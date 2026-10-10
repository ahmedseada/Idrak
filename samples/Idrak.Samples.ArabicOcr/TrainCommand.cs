// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Diagnostics;
using Idrak.Inference.Abstractions;
using Idrak.Nlp;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// <c>train</c>: the line recognizer from a folder of line images and their transcriptions. The alphabet is the data's
/// characters (saved with the model). Lines are scaled to one height and keep their width; a batch is padded to its
/// widest line and each line's steps go to the CTC loss. Augmentations come from the library's registry
/// (<see cref="Augmentations"/>, plus the app's "noise"), seeded per line and epoch, run in parallel off the step. A step
/// the device has no room for is split into micro-batches whose gradients add up (halved again as needed), as the library's
/// image training does. Each epoch reports the loss and, on <c>--eval</c>, CER and WER (<see cref="TuningMetrics"/>);
/// the best epoch is kept. On the console a live line shows the epoch's progress (steps, loss, lines a second, data and
/// step time, device memory, time left); every step and epoch also goes to the library's telemetry (<see cref="Telemetry"/>),
/// which <c>--log FILE</c> writes as JSON Lines.
/// </summary>
internal static class TrainCommand
{
    public const string DefaultAugment = "shift(pixels=2), affine(degrees=1, translate=0.01, scale=0.95:1.05, shear=3), color-jitter(brightness=0.2, contrast=0.3), noise(std=0.03)";

    private const string Help = """
        train --data DIR [--eval DIR] [-o MODEL]
          Trains the line recognizer on line images (PNG, JPEG, BMP, PGM) with UTF-8 transcriptions beside them
          (NAME.png + NAME.txt; cut writes this layout). Only corrected .txt files are used unless --include-drafts
          (then a line without one uses its .aligned.txt, else its .draft.txt).
          --eval DIR           held-out lines: CER and WER every epoch; the best epoch is kept (else the lowest loss)
          -o, --out MODEL      the model folder (default: recognizer): recognizer.ikm + recognizer.json
          --epochs N           (30)        --batch N   lines per step (16)      --lr X   (0.001)      --seed N (1)
          --schedule NAME      the learning rate over the run's steps (the library's FineTuningSchedules): cosine
                               (default), linear, constant, wsd    --warmup F  warm-up share of the steps (0.03)
                               --min-lr X  the rate at the end (0)
          --height N           line height, a multiple of 4 (48)    --channels A,B,C  (32,64,128)
          --hidden N           LSTM size each way (128)              --layers N        (2)
          --direction rtl|ltr  (rtl: lines are flipped so columns run in reading order)
          --profile N          time N steps after the warm ones, then every CUDA kernel of N more (the library's
                               GpuProfiler: calls, ms, TFLOPS), print where a step's time goes, and stop
          --profile-after N    the warm steps before --profile times (20); lines of new widths keep the first epoch
                               cold (new memory blocks, new shapes measured), so a later step shows the warm cost
          --log FILE           every step and epoch as JSON Lines (the library's telemetry: loss, data and step time,
                               epoch summaries with CER, WER, lines a second and device memory)
          --augment PIPELINE   the Augmentations registry's names ("none" for none); default:
                               shift(pixels=2), affine(degrees=1, translate=0.01, scale=0.95:1.05, shear=3),
                               color-jitter(brightness=0.2, contrast=0.3), noise(std=0.03)
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error,
                ["--data", "--eval", "--out", "--epochs", "--batch", "--lr", "--seed", "--height", "--channels", "--hidden", "--layers", "--direction", "--augment", "--log", "--profile", "--profile-after",
                    "--schedule", "--warmup", "--min-lr"],
                ["--include-drafts"], Help) is not { } context)
        {
            return 0;
        }

        var a = context.Args;
        string data = a.Required("--data", "a folder of line images with .txt transcriptions");
        string folder = a.Option("--out") ?? "recognizer";
        int epochs = a.Integer("--epochs", 30, 1), batchSize = a.Integer("--batch", 16, 1), seed = a.Integer("--seed", 1);
        float learningRate = (float)a.Number("--lr", 0.001, 1e-7);
        string scheduleName = (a.Option("--schedule") ?? "cosine").ToLowerInvariant();
        float warmup = (float)a.Number("--warmup", 0.03, 0), minLearningRate = (float)a.Number("--min-lr", 0, 0);
        if (!FineTuningSchedules.Names.Contains(scheduleName) || warmup >= 1)
        {
            throw new UsageException($"--schedule {scheduleName}, --warmup {warmup.ToString(CultureInfo.InvariantCulture)}: use one of "
                                     + $"{string.Join(", ", FineTuningSchedules.Names)} and a warm-up share under 1.");
        }
        bool drafts = a.Flag("--include-drafts");
        string direction = a.Option("--direction") ?? "rtl";
        if (direction is not ("rtl" or "ltr"))
        {
            throw new UsageException($"--direction is rtl or ltr, not '{direction}'.");
        }

        string augmentText = a.Option("--augment") ?? DefaultAugment;
        IReadOnlyList<IAugmentation> augment;
        try
        {
            augment = Augmentations.Parse(augmentText);
        }
        catch (ArgumentException e)
        {
            throw new UsageException($"--augment: {e.Message}");
        }

        // Loading shows where it is: reading tens of thousands of transcriptions and decoding as many images takes a while.
        var live = new LiveLine(error);
        var loading = Stopwatch.StartNew();
        Action<int, int> Phase(string what, string unit) => (done, total) => live.Show(() =>
        {
            double seconds = loading.Elapsed.TotalSeconds, rate = done / Math.Max(1e-9, seconds);
            var left = TimeSpan.FromSeconds(rate > 0 ? (total - done) / rate : 0);
            return FormattableString.Invariant($"{what} {LiveLine.Bar(done / (double)Math.Max(1, total), 10)} {done:N0}/{total:N0} {unit}")
                   + FormattableString.Invariant($" | {rate:F0} {unit}/s | {LiveLine.Time(loading.Elapsed)}, {LiveLine.Time(left)} left");
        }, now: done == total);

        var train = LineFiles.Read(data, drafts, Phase("reading the training transcriptions", "files"));
        if (train.Count == 0)
        {
            throw new UsageException($"{data} has no line images with a transcription ({(drafts ? ".txt, .aligned.txt or .draft.txt" : "a filled .txt; drafts need --include-drafts")}).");
        }

        loading.Restart();
        var evaluation = a.Option("--eval") is { } evalFolder ? LineFiles.Read(evalFolder, drafts: false, Phase("reading the held-out transcriptions", "files")) : [];
        var alphabet = train.SelectMany(s => s.Text.EnumerateRunes()).Select(r => r.ToString()).Distinct().Order(StringComparer.Ordinal).ToArray();
        var settings = new RecognizerSettings
        {
            Alphabet = alphabet,
            Height = a.Integer("--height", 48, 8),
            RightToLeft = direction == "rtl",
            Channels = a.Option("--channels") is { } channels ? ParseChannels(channels) : [32, 64, 128],
            Hidden = a.Integer("--hidden", 128, 1),
            Layers = a.Integer("--layers", 2, 1),
            Augment = augment.Count == 0 ? "none" : augmentText,
        };

        using var recognizer = Recognizer.Create(settings, context.Device, seed);
        var clock = Stopwatch.StartNew();
        loading.Restart();
        var trainLines = Prepare(recognizer, train, Phase("decoding and preparing the training lines", "lines"));
        loading.Restart();
        var evalLines = new ImageData[evaluation.Count];
        var evalProgress = Phase("decoding the held-out lines", "lines");
        int decoded = 0;
        Parallel.For(0, evalLines.Length, ComputeResources.ParallelOptions, i =>
        {
            evalLines[i] = ImageCodecs.Decode(evaluation[i].Image);
            evalProgress(Interlocked.Increment(ref decoded), evalLines.Length);
        });
        live.Clear();
        var labels = train.Select(s => recognizer.Encode(s.Text).Labels).ToArray();
        var unknown = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (var sample in evaluation)
        {
            foreach (string c in recognizer.Encode(sample.Text).Unknown)
            {
                unknown[c] = unknown.GetValueOrDefault(c) + 1;
            }
        }

        int tooNarrow = Enumerable.Range(0, train.Count).Count(i => trainLines[i].Steps < MinimumSteps(labels[i]));
        var kinds = train.GroupBy(s => s.Kind).ToDictionary(g => g.Key, g => g.Count());
        context.Say($"Data      {train.Count:N0} training lines ({string.Join(", ", kinds.Select(k => $"{k.Value} {k.Key}"))}), {evaluation.Count:N0} evaluation lines, "
                    + $"{alphabet.Length} characters, prepared in {clock.Elapsed.TotalSeconds:F1} s");
        if (unknown.Count > 0)
        {
            context.Say($"          evaluation characters not in the training data (left out of the alphabet, counted as errors): "
                        + string.Join(" ", unknown.Select(u => $"{Describe(u.Key)}×{u.Value}")));
        }

        if (tooNarrow > 0)
        {
            context.Say($"          {tooNarrow} lines are too narrow for their text at height {settings.Height} (fewer steps than CTC needs); they count 0 in the loss");
        }

        context.Say($"Network   {recognizer.Network.ParameterCount:N0} parameters on {context.Device} · height {settings.Height} · {(settings.RightToLeft ? "right to left" : "left to right")} · "
                    + $"batch {batchSize} · AdamW {learningRate.ToString("G", CultureInfo.InvariantCulture)} ({scheduleName}, warm-up {warmup.ToString("P0", CultureInfo.InvariantCulture)}) · {epochs} epochs · augment {settings.Augment}");

        if (a.Option("--log") is { } logFolder && Path.GetDirectoryName(Path.GetFullPath(logFolder)) is { } parent)
        {
            Directory.CreateDirectory(parent);
        }

        using var log = a.Option("--log") is { } logFile ? new JsonLinesLogger(logFile, TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Devices) : null;
        using var subscription = log is null ? null : Telemetry.Subscribe(log);
        void Say(string line)
        {
            live.Clear();
            context.Say(line);
        }

        using var optimizer = new AdamW(recognizer.Network.Parameters(), learningRate, weightDecay: 1e-4f);
        var measured = new MeasuredBatches(context.Device);
        var cer = TuningMetrics.Get(TuningMetrics.CharacterErrorRate);
        var wer = TuningMetrics.Get(TuningMetrics.WordErrorRate);
        int micro = batchSize, bestEpoch = 0;
        double best = double.PositiveInfinity, bestLoss = double.PositiveInfinity, bestCer = double.NaN, bestWer = double.NaN;
        var epochSeconds = new List<double>();
        var history = new JsonArray();
        var runClock = Stopwatch.StartNew();
        long step = 0;
        double lastLoss = double.NaN;
        int stepsPerEpoch = (trainLines.Length + batchSize - 1) / batchSize;             // Batches makes ceil(lines / size) of them
        int profile = a.Integer("--profile", 0, 0);
        int profileWarm = a.Integer("--profile-after", 20, 1);
        long profileStart = 0, profiledLines = 0;
        TimeSpan profileData = default, profileCompute = default;
        // The learning rate follows the schedule over every step of the run (warm-up, then the decay), stepped after each
        // optimizer step.
        var schedule = FineTuningSchedules.Create(scheduleName, warmup, minLearningRate)(optimizer, Math.Max(1, epochs * stepsPerEpoch));
        Telemetry.TrainingStarted(new TrainingStarted("ArabicOcr line recognizer", "AdamW", context.Device, epochs, train.Count, evaluation.Count, batchSize,
            stepsPerEpoch, recognizer.Network.ParameterCount, learningRate, Environment.ProcessorCount));
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            var epochClock = Stopwatch.StartNew();
            recognizer.Network.Train();
            double lossSum = 0, stepLoss = double.NaN;
            int seen = 0, done = 0;
            TimeSpan dataTime = default, computeTime = default;
            foreach (var batch in Batches(trainLines, batchSize, seed, epoch))
            {
                // --profile: steps profileWarm + 1 ... + N timed as they run; the N after them with every kernel timed.
                if (profile > 0 && step == profileWarm)
                {
                    context.Device.Synchronize();
                    profileStart = Stopwatch.GetTimestamp();
                }
                else if (profile > 0 && step == profileWarm + profile)
                {
                    context.Device.Synchronize();
                    double wallMs = Stopwatch.GetElapsedTime(profileStart).TotalMilliseconds / profile;
                    Say(FormattableString.Invariant($"Profile   {wallMs:F1} ms a step as it runs ({profiledLines / (double)profile * 1000 / wallMs:F0} lines/s), steps {profileWarm + 1}-{profileWarm + profile}"));
                    Say(FormattableString.Invariant($"          data {profileData.TotalMilliseconds / profile:F1} ms (the batch augmented on the CPU) + step {profileCompute.TotalMilliseconds / profile:F1} ms (forward, CTC, backward, optimizer: host and device)"));
                    GpuProfiler.Start(context.Device);
                }
                else if (profile > 0 && step == profileWarm + 2 * profile)
                {
                    var kernels = GpuProfiler.Stop(context.Device);
                    if (kernels.Count == 0)
                    {
                        Say($"          {context.Device} does not time kernels");
                    }
                    else
                    {
                        Say(FormattableString.Invariant($"          GPU time {kernels.Sum(k => k.Milliseconds) / profile:F1} ms a step in {kernels.Sum(k => k.Calls) / profile:N0} kernel launches (each kernel waited for)"));
                        Say(GpuProfiler.Format(kernels, 30));
                    }

                    return 0;
                }

                long started = Stopwatch.GetTimestamp();
                var lines = Augment(trainLines, batch, augment, settings.Height, seed, epoch);
                long prepared = Stopwatch.GetTimestamp();
                while (true)
                {
                    try
                    {
                        optimizer.ZeroGrad();
                        double batchLoss = 0;
                        for (int start = 0; start < batch.Length; start += micro)
                        {
                            int count = Math.Min(micro, batch.Length - start);
                            using var scope = new TensorScope();
                            var part = lines.Skip(start).Take(count).ToArray();
                            var (images, steps) = recognizer.Batch(part);
                            var logProbs = recognizer.LogProbabilities(images);
                            var loss = Losses.Ctc(logProbs, [.. batch.Skip(start).Take(count).Select(i => labels[i])], steps, blank: 0, zeroInfinity: true, batchFirst: true);
                            var scaled = count == batch.Length ? loss : loss * (count / (float)batch.Length);
                            scaled.Backward();
                            batchLoss += loss.Item() * count;
                        }

                        optimizer.ClipGradientNorm(5f);
                        optimizer.Step();
                        schedule.Step();
                        lossSum += batchLoss;
                        seen += batch.Length;
                        stepLoss = batchLoss / batch.Length;
                        if (profile > 0 && step >= profileWarm && step < profileWarm + profile)
                        {
                            profiledLines += batch.Length;
                        }

                        break;
                    }
                    catch (ResourceLimitExceededException e) when (micro > 1)
                    {
                        micro = (micro + 1) / 2;
                        Say($"  out of device memory ({e.Message.Split(':')[0]}): micro-batches of {micro} lines from this step, their gradients added up");
                    }
                }

                var preparing = Stopwatch.GetElapsedTime(started, prepared);
                var compute = Stopwatch.GetElapsedTime(prepared);
                (dataTime, computeTime) = (dataTime + preparing, computeTime + compute);
                if (profile > 0 && step >= profileWarm && step < profileWarm + profile)
                {
                    (profileData, profileCompute) = (profileData + preparing, profileCompute + compute);
                }

                done++;
                step++;
                if (Telemetry.IsEnabled(TelemetryLevel.Batches))
                {
                    Telemetry.BatchCompleted(new BatchCompleted(epoch, done, stepsPerEpoch, step, batch.Length, stepLoss, optimizer.LearningRate, null, preparing, compute));
                }

                int epochNow = epoch, doneNow = done, seenNow = seen;
                double sumNow = lossSum, lastNow = stepLoss, rateNow = optimizer.LearningRate;
                live.Show(() =>
                {
                    double fraction = doneNow / (double)Math.Max(1, stepsPerEpoch);
                    var elapsed = epochClock.Elapsed;
                    var memory = ComputeResources.GetMemoryUsage(context.Device);
                    var left = TimeSpan.FromSeconds(elapsed.TotalSeconds / Math.Max(fraction, 1e-9) * (1 - fraction));
                    // Most important first: a narrow console cuts the end of the line.
                    return FormattableString.Invariant($"epoch {epochNow}/{epochs} {LiveLine.Bar(fraction, 10)} {fraction * 100:F1}% {doneNow:N0}/{stepsPerEpoch:N0}")
                           + FormattableString.Invariant($" | loss {sumNow / Math.Max(1, seenNow):F4} (step {lastNow:F4}) | lr {rateNow:G3}")
                           + $" | {LiveLine.Time(elapsed)}, {LiveLine.Time(left)} left"
                           + FormattableString.Invariant($" | {seenNow / Math.Max(1e-9, elapsed.TotalSeconds):F0} lines/s")
                           + FormattableString.Invariant($" | data {dataTime.TotalMilliseconds / doneNow:F0} + step {computeTime.TotalMilliseconds / doneNow:F0} ms")
                           + $" | memory {LiveLine.Bytes(memory.InUse)}" + (memory.Peak > 0 ? $", peak {LiveLine.Bytes(memory.Peak)}" : "");
                });
            }

            double trainLoss = lossSum / Math.Max(1, seen);
            double? epochCer = null, epochWer = null;
            if (evaluation.Count > 0)
            {
                live.Show(() => $"epoch {epoch}/{epochs}  reading the {evaluation.Count:N0} held-out lines for CER and WER ...", now: true);
                var readings = recognizer.Read(evalLines, CtcDecoders.Greedy, new CtcDecodeOptions(), measured);
                var pairs = readings.Zip(evaluation).Select(p => (p.First.Text, p.Second.Text)).ToArray();
                epochCer = TuningScore.Sum(pairs.Select(p => cer.Score(p.Item1, p.Item2))).Value;
                epochWer = TuningScore.Sum(pairs.Select(p => wer.Score(p.Item1, p.Item2))).Value;
            }

            double monitored = epochCer ?? trainLoss;                                     // CER on the held-out lines, ties broken by the loss
            bool isBest = monitored < best || monitored == best && trainLoss < bestLoss;
            epochClock.Stop();
            epochSeconds.Add(epochClock.Elapsed.TotalSeconds);
            if (isBest)
            {
                (best, bestLoss, bestEpoch, bestCer, bestWer) = (monitored, trainLoss, epoch, epochCer ?? double.NaN, epochWer ?? double.NaN);
                recognizer.Save(folder, Summary());
            }

            lastLoss = trainLoss;
            var validation = epochCer is null ? null : new Dictionary<string, double> { ["cer"] = epochCer.Value, ["wer"] = epochWer!.Value };
            Telemetry.EpochCompleted(new EpochCompleted(epoch, epochs, trainLoss, new Dictionary<string, double>(), null, validation, optimizer.LearningRate, epochClock.Elapsed,
                seen / Math.Max(1e-9, epochClock.Elapsed.TotalSeconds), ComputeResources.GetMemoryUsage(context.Device), isBest));
            history.Add(new JsonObject { ["epoch"] = epoch, ["loss"] = Round(trainLoss), ["cer"] = epochCer is null ? null : Round(epochCer.Value), ["wer"] = epochWer is null ? null : Round(epochWer.Value), ["seconds"] = Round(epochClock.Elapsed.TotalSeconds), ["lr"] = optimizer.LearningRate });
            Say(FormattableString.Invariant($"epoch {epoch,3}/{epochs}  loss {trainLoss,8:F4}")
                        + (epochCer is null ? "" : FormattableString.Invariant($"  cer {epochCer:F4}  wer {epochWer:F4}"))
                        + FormattableString.Invariant($"  {epochClock.Elapsed.TotalSeconds,6:F1} s  {seen / Math.Max(1e-9, epochClock.Elapsed.TotalSeconds),6:F0} lines/s  lr {optimizer.LearningRate:G3}") + (isBest ? "  best" : ""));
        }

        live.Clear();
        Telemetry.TrainingCompleted(new TrainingCompleted(epochs, runClock.Elapsed, lastLoss, bestEpoch, bestLoss, false, false));

        // The settings file keeps the whole run (every epoch), beside the best epoch's weights.
        var summary = Summary();
        summary["history"] = history;
        string settingsFile = Path.Combine(folder, Recognizer.SettingsFile);
        var saved = (JsonObject)JsonNode.Parse(File.ReadAllText(settingsFile))!;
        saved["training"] = summary.DeepClone();
        OcrText.Write(settingsFile, saved.ToJsonString(Json.Indented));
        context.Say($"Saved     {Path.GetFullPath(folder)} (epoch {bestEpoch}{(double.IsNaN(bestCer) ? "" : FormattableString.Invariant($", cer {bestCer:F4}, wer {bestWer:F4}"))})");
        if (context.Json)
        {
            context.WriteJson(summary);
        }

        return 0;

        JsonObject Summary() => new()
        {
            ["model"] = Path.GetFullPath(folder),
            ["device"] = context.Device.ToString(),
            ["train_lines"] = train.Count,
            ["eval_lines"] = evaluation.Count,
            ["characters"] = alphabet.Length,
            ["best_epoch"] = bestEpoch,
            ["cer"] = double.IsNaN(bestCer) ? null : Round(bestCer),
            ["wer"] = double.IsNaN(bestWer) ? null : Round(bestWer),
            ["seconds_per_epoch"] = epochSeconds.Count == 0 ? null : Round(epochSeconds.Average()),
            ["micro_batch"] = micro,
            ["learning_rate"] = learningRate,
            ["schedule"] = scheduleName,
            ["warmup"] = warmup,
            ["min_learning_rate"] = minLearningRate,
            ["drafts"] = drafts,
            ["unknown_characters"] = new JsonObject([.. unknown.Select(u => KeyValuePair.Create(u.Key, (JsonNode?)u.Value))]),
        };
    }

    private static double Round(double value) => Math.Round(value, 5);

    // CTC needs a step per label and one more between two equal labels.
    private static int MinimumSteps(int[] labels)
    {
        int steps = labels.Length;
        for (int i = 1; i < labels.Length; i++)
        {
            steps += labels[i] == labels[i - 1] ? 1 : 0;
        }

        return steps;
    }

    private static IReadOnlyList<int> ParseChannels(string text)
    {
        var parts = text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        var channels = new List<int>();
        foreach (string part in parts)
        {
            channels.Add(int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int c) && c > 0 ? c : throw new UsageException($"--channels takes whole numbers, such as 32,64,128, not '{text}'."));
        }

        return channels.Count >= 3 ? channels : throw new UsageException("--channels needs at least three blocks, such as 32,64,128.");
    }

    // The lines decoded and prepared once (in parallel), kept as bytes (a quarter of the floats).
    private static StoredLine[] Prepare(Recognizer recognizer, IReadOnlyList<LineSample> samples, Action<int, int> progress)
    {
        var lines = new StoredLine[samples.Count];
        int done = 0;
        Parallel.For(0, samples.Count, ComputeResources.ParallelOptions, i =>
        {
            var prepared = recognizer.Prepare(ImageCodecs.Decode(samples[i].Image));
            var bytes = new byte[prepared.Pixels.Length];
            for (int k = 0; k < bytes.Length; k++)
            {
                bytes[k] = (byte)Math.Round(Math.Clamp(prepared.Pixels[k], 0f, 1f) * 255);
            }

            lines[i] = new StoredLine(bytes, prepared.Width);
            progress(Interlocked.Increment(ref done), samples.Count);
        });
        return lines;
    }

    // An epoch's batches: shuffled, then lines of similar width put together (sorted within windows of 20 batches) so a
    // batch pads little, then the batches shuffled. Seeded by the seed and the epoch.
    private static IEnumerable<int[]> Batches(StoredLine[] lines, int size, int seed, int epoch)
    {
        var random = new Random(HashCode.Combine(seed, epoch));
        var order = Enumerable.Range(0, lines.Length).ToArray();
        random.Shuffle(order);
        var batches = new List<int[]>();
        foreach (var window in order.Chunk(size * 20))
        {
            batches.AddRange(window.OrderBy(i => lines[i].Width).Chunk(size));
        }

        var shuffled = batches.ToArray();
        random.Shuffle(shuffled);
        return shuffled;
    }

    // Each line of the batch through the augmentations (in parallel; each line's random numbers from the seed, the epoch
    // and its index, so a run repeats whatever the threads), back at the network's height.
    private static PreparedLine[] Augment(StoredLine[] lines, int[] batch, IReadOnlyList<IAugmentation> augment, int height, int seed, int epoch)
    {
        var result = new PreparedLine[batch.Length];
        Parallel.For(0, batch.Length, ComputeResources.ParallelOptions, k =>
        {
            var line = lines[batch[k]];
            var pixels = new float[line.Pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = line.Pixels[i] / 255f;
            }

            if (augment.Count == 0)
            {
                result[k] = new PreparedLine(pixels, line.Width);
                return;
            }

            var sample = new AnnotatedImage(new ImageData(pixels, 1, height, line.Width));
            var context = new AugmentationContext(new Random(HashCode.Combine(seed, epoch, batch[k])));
            foreach (var step in augment)
            {
                sample = step.Apply(sample, context);
            }

            var image = sample.Image;
            var values = image.Height == height && image.Channels == 1 ? image.Pixels : image.Resize(1, height, Math.Max(RecognizerSettings.Stride, image.Width * height / image.Height));
            int width = values.Length / height;
            result[k] = new PreparedLine(values, width);
        });
        return result;
    }

    private static string Describe(string character) =>
        string.Join(" ", character.EnumerateRunes().Select(r => $"U+{r.Value:X4}")) + (character.EnumerateRunes().All(r => !Rune.IsControl(r)) ? $" '{character}'" : "");

    private sealed record StoredLine(byte[] Pixels, int Width)
    {
        public int Steps => (Width + RecognizerSettings.Stride - 1) / RecognizerSettings.Stride;
    }
}
