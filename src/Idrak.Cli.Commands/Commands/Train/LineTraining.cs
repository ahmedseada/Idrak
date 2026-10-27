// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Text.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// A line-recognition run of <c>idrak train</c> (<see cref="ImageTask.LineRecognition"/>): a line recognizer read through
/// its family (fine-tuned) or a builder network ending in [steps, classes], on a folder of line images with their
/// transcriptions beside them (<c>NAME.png</c> + <c>NAME.txt</c>, UTF-8). Lines keep their width: each is prepared once
/// (<see cref="ModelLineReader.Prepare(ImageData, LineRecognitionSettings)"/>), kept as bytes; an epoch's lines are shuffled,
/// put in batches of similar widths (padded to the widest), augmented per line on worker threads (<see cref="Augmentations"/>,
/// seeded by the run's seed, the epoch and the line), and trained with the CTC loss. Each epoch reports the loss and, on
/// the evaluation lines, the loss, CER and WER (<see cref="TuningMetrics"/>); the lowest CER (else loss) is the best epoch.
/// A step the device has no room for is split into micro-batches whose gradients add up.
/// </summary>
internal static class LineTraining
{
    private sealed record LineSample(string Image, string Text);

    private sealed record StoredLine(byte[] Pixels, int Width);

    /// <summary>Trains <paramref name="network"/> (a family's, or a builder's) as a line recognizer.</summary>
    public static int Train(CommandContext context, RunSettings settings, ImageRunOptions options, string folder, ImageModel? family, NetworkBuilder? builder,
        Module network, string? resume, int epochsDone, int? epochs)
    {
        var device = context.Device;
        if (options.Loss is not null || options.Matcher is not null || options.Decoder is not null || options.DataFormat is not ("auto" or "lines"))
        {
            throw new UsageException("A line recognizer trains with the CTC loss on a folder of line images and their .txt transcriptions: --loss, --matcher, --decoder and --data-format are not its options.");
        }

        if (options.Metric is { } metricName && !metricName.Equals(TuningMetrics.CharacterErrorRate, StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException($"--metric {metricName}: a line recognizer is scored by {TuningMetrics.CharacterErrorRate} (with {TuningMetrics.WordErrorRate} beside it).");
        }

        // ---------------------------------------------------------------- what the network reads
        LineRecognitionSettings lines;
        IReadOnlyList<string>? labels;
        int channels, classes;
        if (family is not null)
        {
            if (options.Direction is not null || options.ReadingOrder is not null)
            {
                throw new UsageException($"{options.Model}'s family sets its lines' direction and reading order; --direction and --reading-order are for a builder network.");
            }

            (lines, labels, channels, classes) = (family.Lines!, family.Labels!, family.Channels, family.Labels!.Count);
        }
        else
        {
            var shape = builder!.CurrentShape;
            int inputWidth = builder.InputShape[2];
            if (shape.Count != 2 || shape[0] <= 0 || inputWidth % shape[0] != 0)
            {
                throw new UsageException($"A line recognizer's network ends in [steps, classes] (a score per class for each step along the width: ColumnsToSequence, then "
                                         + $"recurrent or linear layers), a step a whole number of its {inputWidth} input columns; this one ends in [{string.Join(", ", shape)}].");
            }

            bool rightToLeft = (options.Direction ?? "ltr") switch
            {
                "rtl" => true,
                "ltr" => false,
                var other => throw new UsageException($"--direction {other}: use ltr or rtl."),
            };
            lines = new LineRecognitionSettings
            {
                Height = builder.InputShape[1], Stride = inputWidth / shape[0], RightToLeft = rightToLeft,
                ReadingOrder = options.ReadingOrder ?? (rightToLeft ? ReadingOrders.Runs : null),
            };
            labels = null;
            channels = builder.InputShape[0];
            classes = shape[1];
        }

        if (lines.ReadingOrder is { } order && ReadingOrders.Find(order) is null)
        {
            throw new UsageException($"--reading-order {order}: use {string.Join(", ", ReadingOrders.Names)}.");
        }

        // ---------------------------------------------------------------- the data
        var watch = Stopwatch.StartNew();
        var train = ReadLines(settings.Data, "--data");
        var evaluation = options.Eval is { } evalPath ? ReadLines(Path.GetFullPath(evalPath), "--eval") : null;
        if (labels is null)
        {
            // A builder network learns the data's characters: class 0 is CTC's blank, the characters in code point order after it.
            string[] alphabet = [.. train.SelectMany(s => s.Text.EnumerateRunes()).Select(r => r.ToString()).Distinct().Order(StringComparer.Ordinal)];
            if (alphabet.Length + 1 != classes)
            {
                throw new InvalidDataException($"{settings.Data} has {alphabet.Length} characters but the network scores {classes} classes; make its last layer "
                                               + $"{alphabet.Length + 1} wide (the characters and CTC's blank).");
            }

            labels = ["", .. alphabet];
        }

        if (train.Count < 2)
        {
            throw new InvalidDataException($"{settings.Data} has {train.Count} transcribed lines; training needs more.");
        }

        if (evaluation is null && settings.Validation > 0)
        {
            var shuffled = train.ToArray();
            new Random(settings.Seed).Shuffle(shuffled);
            int held = Math.Clamp((int)Math.Round(train.Count * settings.Validation), 1, train.Count - 1);
            evaluation = shuffled[..held];
            train = shuffled[held..];
        }

        var reader = new ModelLineReader(network, labels, lines, channels, CtcDecoders.Greedy, device: device);
        var unknown = new SortedDictionary<string, int>(StringComparer.Ordinal);
        int[][] Encode(IReadOnlyList<LineSample> samples) => [.. samples.Select(s =>
        {
            var (ids, missing) = reader.Encode(s.Text);
            foreach (string c in missing)
            {
                unknown[c] = unknown.GetValueOrDefault(c) + 1;
            }

            return ids;
        })];
        var trainLabels = Encode(train);
        var evalLabels = evaluation is null ? null : Encode(evaluation);
        var trainLines = Prepare(reader, train);
        var evalLines = evaluation is null ? null : Prepare(reader, evaluation);
        context.Write($"Data      {train.Count:N0} lines from {settings.Data}{(evaluation is null ? "" : $" · {evaluation.Count:N0} for evaluation{(options.Eval is null ? " (held out)" : $" from {options.Eval}")}")} "
                      + $"({watch.ElapsedMilliseconds} ms)");
        context.Write($"Task      line-recognition, {labels.Count - 1} characters · lines {lines.Height} high, a step every {lines.Stride} column{(lines.Stride == 1 ? "" : "s")} · "
                      + $"{(lines.RightToLeft ? "right to left" : "left to right")}{(lines.ReadingOrder is null ? "" : $" ({lines.ReadingOrder})")}");
        if (unknown.Count > 0)
        {
            context.Write($"          characters no label holds (left out of the targets): {string.Join(" ", unknown.Select(u => $"'{u.Key}'×{u.Value}"))}");
        }

        IReadOnlyList<IAugmentation> pipeline;
        try
        {
            pipeline = Augmentations.Parse(options.Augment);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new UsageException($"--augment: {e.Message} (registered: {string.Join(", ", Augmentations.Names)})");
        }

        // ---------------------------------------------------------------- optimizer and run folder
        if (resume is not null)
        {
            network.Load(resume);
        }

        network.Train();
        string optimizerName = settings.Optimizer.ToLowerInvariant();
        var parameters = network.Parameters().ToList();
        if (parameters.Count == 0)
        {
            throw new UsageException($"{options.Model ?? "The network"} has no trainable parameters.");
        }

        using Optimizer optimizer = optimizerName switch
        {
            "adamw" => new AdamW(parameters, settings.LearningRate, weightDecay: settings.WeightDecay ?? 1e-4f),
            "adam" => new Adam(parameters, settings.LearningRate, weightDecay: settings.WeightDecay ?? 0f),
            "sgd" => new Sgd(parameters, settings.LearningRate, momentum: 0.9f, weightDecay: settings.WeightDecay ?? 0f),
            _ => throw new UsageException($"--optimizer {settings.Optimizer}: use adamw, adam or sgd."),
        };
        int total = epochs ?? Math.Max(0, settings.Epochs - epochsDone);
        if (total == 0)
        {
            context.Error($"The run has done its {settings.Epochs} epochs; give --epochs N to train N more.");
            return ExitCodes.Failed;
        }

        File.WriteAllText(Path.Combine(folder, "run.json"), settings.ToJson().ToJsonString(CommandContext.JsonOutput));
        if (builder is not null)
        {
            File.WriteAllText(Path.Combine(folder, "network.json"), settings.Network.ToJsonString(CommandContext.JsonOutput));
        }

        int steps = (trainLines.Length + settings.Batch - 1) / settings.Batch;
        context.Write($"Training  {network.ParameterCount:N0} parameters on {device} · {optimizerName} {settings.LearningRate.ToString("G", CultureInfo.InvariantCulture)} · batch {settings.Batch} · "
                      + $"loss ctc{(pipeline.Count > 0 ? $" · augment {string.Join(", ", pipeline.Select(a => a.Name))}" : "")} · {(resume is null ? "" : $"from epoch {epochsDone + 1}, ")}up to {total} epochs");
        var started = new TrainingStarted(network.Name ?? options.Model ?? "network", optimizerName, device, total + epochsDone, train.Count, evaluation?.Count,
            settings.Batch, steps, network.ParameterCount, settings.LearningRate, ComputeResources.MaxCpuThreads);
        var writer = new RunLogWriter(Path.Combine(folder, "log.jsonl"), started, epochsDone);
        writer.Append(started);

        // ---------------------------------------------------------------- the loop
        int? patience = evaluation is null ? null : settings.Patience ?? 20;
        int every = context.Verbose ? 1 : Math.Max(1, total / 10);
        int micro = settings.Batch;
        double best = double.PositiveInfinity, bestLoss = double.PositiveInfinity, finalLoss = double.NaN;
        int bestEpoch = 0, ran = 0, sinceBest = 0;
        bool stoppedEarly = false;
        var report = new Dictionary<string, double>();
        var clock = Stopwatch.StartNew();
        string last = Path.Combine(folder, "last.ikw"), bestFile = Path.Combine(folder, "best.ikw");
        using (var progress = new ProgressLine(context, "training", (long)total * steps))
        using (var interrupt = new Interrupt(context))
        {
            for (int epoch = 1; epoch <= total && !interrupt.Requested; epoch++)
            {
                var epochClock = Stopwatch.StartNew();
                network.Train();
                double sum = 0;
                int seen = 0;
                progress.Label = $"epoch {epoch + epochsDone}/{total + epochsDone}";
                foreach (var batch in Batches(trainLines, settings.Batch, settings.Seed, epoch + epochsDone))
                {
                    var prepared = Augment(trainLines, batch, pipeline, lines.Height, lines.Stride, settings.Seed, epoch + epochsDone);
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
                                var loss = Loss(reader, network, prepared.Skip(start).Take(count).ToArray(), [.. batch.Skip(start).Take(count).Select(i => trainLabels[i])], training: true);
                                var scaled = count == batch.Length ? loss : loss * (count / (float)batch.Length);
                                scaled.Backward();
                                batchLoss += loss.Item() * count;
                            }

                            optimizer.Step();
                            sum += batchLoss;
                            seen += batch.Length;
                            break;
                        }
                        catch (ResourceLimitExceededException e) when (micro > 1)
                        {
                            micro = (micro + 1) / 2;
                            progress.Erase();
                            context.Write($"  out of device memory ({e.Message.Split(':')[0]}): micro-batches of {micro} from this step, their gradients added up");
                        }
                    }

                    progress.Advance();
                    if (interrupt.Requested)
                    {
                        break;
                    }
                }

                if (interrupt.Requested && seen < train.Count)
                {
                    break;                                                       // the unfinished epoch is not summarized
                }

                double trainLoss = sum / Math.Max(1, seen);
                double? validationLoss = null;
                Dictionary<string, double>? scores = null;
                if (evalLines is not null)
                {
                    (validationLoss, scores) = Evaluate(reader, network, evalLines, evalLabels!, evaluation!, settings.Batch);
                }

                double monitored = scores?[TuningMetrics.CharacterErrorRate] ?? validationLoss ?? trainLoss;
                double tie = validationLoss ?? trainLoss;
                bool isBest = monitored < best || monitored == best && tie < bestLoss;
                network.Save(last);
                if (isBest)
                {
                    (best, bestLoss, bestEpoch, sinceBest) = (monitored, tie, epoch, 0);
                    report = scores ?? new Dictionary<string, double> { ["loss"] = tie };
                    network.Save(bestFile);
                }
                else
                {
                    sinceBest++;
                }

                ran = epoch;
                finalLoss = trainLoss;
                epochClock.Stop();
                writer.Append(new EpochCompleted(epoch + epochsDone, total + epochsDone, trainLoss, new Dictionary<string, double>(), validationLoss,
                    scores, optimizer.LearningRate, epochClock.Elapsed, seen / Math.Max(epochClock.Elapsed.TotalSeconds, 1e-9), ComputeResources.GetMemoryUsage(device), isBest));
                if (epoch % every == 0 || epoch == 1 || epoch == total)
                {
                    progress.Erase();
                    context.Write($"  epoch {epoch + epochsDone,4}  loss {RunLog.Format(trainLoss)}" + (validationLoss is { } v ? $"  val_loss {RunLog.Format(v)}" : "")
                                  + (scores is null ? "" : $"  cer {RunLog.Format(scores[TuningMetrics.CharacterErrorRate])}  wer {RunLog.Format(scores[TuningMetrics.WordErrorRate])}"));
                }

                if (patience is > 0 && sinceBest >= patience)
                {
                    stoppedEarly = true;
                    break;
                }
            }

            progress.Finish();
            if (interrupt.Requested)
            {
                context.Write(interrupt.TimedOut ? $"Stopped   --timeout ran out; 'idrak resume {Path.GetFileName(folder)}' continues the run" : $"Stopped   'idrak resume {Path.GetFileName(folder)}' continues the run");
            }
        }

        double seconds = clock.Elapsed.TotalSeconds;
        writer.Append(new TrainingCompleted(ran, clock.Elapsed, finalLoss, bestEpoch + epochsDone, bestLoss, stoppedEarly, false));
        if (File.Exists(bestFile))
        {
            network.Load(bestFile);                                              // the best epoch's weights are the ones kept
        }

        // ---------------------------------------------------------------- what is written
        var metrics = new JsonObject();
        foreach (var (key, value) in report)
        {
            metrics[key] = double.IsFinite(value) ? Math.Round(value, 6) : null;
        }

        var meta = new JsonObject
        {
            ["format"] = "idrak-train/1",
            ["task"] = ImageModelPredictors.TaskName(ImageTask.LineRecognition),
            ["input"] = "images",
            ["classes"] = new JsonArray([.. labels.Select(n => (JsonNode)n)]),
            ["lines"] = lines.ToJson(),
            ["inputShape"] = new JsonArray(channels, lines.Height, builder?.InputShape[2] ?? lines.Stride),
            ["data"] = settings.Data,
            ["epochs"] = epochsDone + ran,
            ["metric"] = TuningMetrics.CharacterErrorRate,
            ["metrics"] = metrics,
            ["augment"] = options.Augment,
            ["model"] = options.Model,
        };
        File.WriteAllText(Path.Combine(folder, "training.json"), meta.ToJsonString(CommandContext.JsonOutput));
        if (builder is not null)
        {
            TrainSession.Package(network, settings.Network, null, null, meta, settings.Output);
        }
        else
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(settings.Output))!);
            network.Save(settings.Output);
        }

        string headline = report.Count == 0 ? "" : string.Join(" · ", report.Select(p => $"{p.Key} {RunLog.Format(p.Value)}"));
        context.Write($"Result    {(evaluation is null ? "training" : "evaluation")}: {headline}");
        context.Write($"Best      epoch {bestEpoch + epochsDone}{(stoppedEarly ? ", stopped early" : "")} · {seconds:F1} s");
        context.Write($"Wrote     {settings.Output} · run {folder}");
        context.Write(builder is not null
            ? $"Next      idrak predict {settings.Output} LINE.png   ·   idrak runs show {folder}"
            : $"Next      idrak predict -P PLUGIN {options.Model} LINE.png --weights {settings.Output}   ·   idrak runs show {folder}");
        context.WriteJson(new JsonObject
        {
            ["model"] = Path.GetFullPath(settings.Output),
            ["run"] = Path.GetFullPath(folder),
            ["task"] = meta["task"]!.DeepClone(),
            ["classes"] = meta["classes"]!.DeepClone(),
            ["lines"] = train.Count + (options.Eval is null ? evaluation?.Count ?? 0 : 0),
            ["trainingLines"] = train.Count,
            ["evaluationLines"] = evaluation?.Count ?? 0,
            ["parameters"] = network.ParameterCount,
            ["device"] = device.ToString(),
            ["epochs"] = epochsDone + ran,
            ["bestEpoch"] = bestEpoch + epochsDone,
            ["bestLoss"] = double.IsFinite(bestLoss) ? Math.Round(bestLoss, 6) : null,
            ["finalLoss"] = double.IsFinite(finalLoss) ? Math.Round(finalLoss, 6) : null,
            ["stoppedEarly"] = stoppedEarly,
            ["microBatch"] = micro,
            ["seconds"] = Math.Round(seconds, 3),
            ["metric"] = TuningMetrics.CharacterErrorRate,
            ["metrics"] = metrics.DeepClone(),
            ["unknownCharacters"] = new JsonObject([.. unknown.Select(u => KeyValuePair.Create(u.Key, (JsonNode?)u.Value))]),
        });
        return ExitCodes.Ok;
    }

    // The CTC loss of a batch of prepared lines with their labels (zero for a line too narrow for its labels).
    private static Tensor Loss(ModelLineReader reader, Module network, IReadOnlyList<PreparedLine> lines, IReadOnlyList<int[]> labels, bool training)
    {
        var (images, steps) = reader.Batch(lines);
        var output = training ? network.Forward(images) : network.Predict(images);
        int available = output.Shape[1];
        for (int n = 0; n < steps.Length; n++)
        {
            steps[n] = Math.Min(steps[n], available);
        }

        return Losses.Ctc(reader.LogProbabilities(output), labels, steps, blank: reader.Settings.Blank, zeroInfinity: true, batchFirst: true);
    }

    // The evaluation lines' loss, then their greedy readings scored by CER and WER, in batches of similar widths.
    private static (double Loss, Dictionary<string, double> Scores) Evaluate(ModelLineReader reader, Module network, StoredLine[] lines, int[][] labels,
        IReadOnlyList<LineSample> samples, int batch)
    {
        network.Eval();
        double sum = 0;
        var texts = new string[lines.Length];
        var order = Enumerable.Range(0, lines.Length).OrderByDescending(i => lines[i].Width).ToArray();
        using (Autograd.NoGrad())
        {
            foreach (var chunk in order.Chunk(batch))
            {
                using var scope = new TensorScope();
                var prepared = chunk.Select(i => Unpack(lines[i], reader.Settings.Height)).ToArray();
                sum += Loss(reader, network, prepared, [.. chunk.Select(i => labels[i])], training: false).Item() * chunk.Length;
                var read = reader.Read(prepared);
                for (int k = 0; k < chunk.Length; k++)
                {
                    texts[chunk[k]] = AnswerTexts.Normalize(read[k].Text);
                }
            }
        }

        network.Train();
        var pairs = texts.Zip(samples.Select(s => s.Text)).ToArray();
        return (sum / Math.Max(1, lines.Length), new Dictionary<string, double>
        {
            [TuningMetrics.CharacterErrorRate] = TuningMetrics.Score(TuningMetrics.CharacterErrorRate, pairs).Value,
            [TuningMetrics.WordErrorRate] = TuningMetrics.Score(TuningMetrics.WordErrorRate, pairs).Value,
        });
    }

    // The line images of a folder (and its subfolders) with a transcription beside them (NAME.txt, UTF-8, not empty), in
    // name order; the transcriptions normalized (NFC, white space collapsed).
    private static IReadOnlyList<LineSample> ReadLines(string path, string option)
    {
        if (!Directory.Exists(path))
        {
            throw new UsageException($"{option} {path}: a line recognizer reads a folder of line images with their transcriptions beside them (NAME.png and NAME.txt).");
        }

        var samples = new List<LineSample>();
        foreach (string image in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories).Where(ImageCodecs.CanDecode).Order(StringComparer.Ordinal))
        {
            string text = Path.Combine(Path.GetDirectoryName(image)!, Path.GetFileNameWithoutExtension(image) + ".txt");
            if (File.Exists(text) && AnswerTexts.Normalize(File.ReadAllText(text)) is { Length: > 0 } transcription)
            {
                samples.Add(new LineSample(image, transcription));
            }
        }

        return samples.Count > 0 ? samples
            : throw new InvalidDataException($"{path} has no line images with a transcription beside them (NAME.png and a filled NAME.txt; images: {string.Join(", ", ImageFiles.DecodedExtensions)}).");
    }

    // The lines decoded and prepared once (in parallel), kept as bytes (a quarter of the floats).
    private static StoredLine[] Prepare(ModelLineReader reader, IReadOnlyList<LineSample> samples)
    {
        var lines = new StoredLine[samples.Count];
        Parallel.For(0, samples.Count, ComputeResources.ParallelOptions, i =>
        {
            var prepared = reader.Prepare(ImageCodecs.Decode(samples[i].Image));
            var bytes = new byte[prepared.Pixels.Length];
            for (int k = 0; k < bytes.Length; k++)
            {
                bytes[k] = (byte)Math.Round(Math.Clamp(prepared.Pixels[k], 0f, 1f) * 255);
            }

            lines[i] = new StoredLine(bytes, prepared.Width);
        });
        return lines;
    }

    private static PreparedLine Unpack(StoredLine line, int height)
    {
        var pixels = new float[line.Pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = line.Pixels[i] / 255f;
        }

        return new PreparedLine(pixels, height, line.Width);
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
    private static PreparedLine[] Augment(StoredLine[] lines, int[] batch, IReadOnlyList<IAugmentation> augment, int height, int stride, int seed, int epoch)
    {
        var result = new PreparedLine[batch.Length];
        Parallel.For(0, batch.Length, ComputeResources.ParallelOptions, k =>
        {
            var line = Unpack(lines[batch[k]], height);
            if (augment.Count == 0)
            {
                result[k] = line;
                return;
            }

            var sample = new AnnotatedImage(new ImageData(line.Pixels, 1, height, line.Width));
            var context = new AugmentationContext(new Random(HashCode.Combine(seed, epoch, batch[k])));
            foreach (var step in augment)
            {
                sample = step.Apply(sample, context);
            }

            var image = sample.Image;
            var values = image.Height == height && image.Channels == 1 ? image.Pixels : image.Resize(1, height, Math.Max(stride, image.Width * height / image.Height));
            result[k] = new PreparedLine(values, height, values.Length / height);
        });
        return result;
    }
}
