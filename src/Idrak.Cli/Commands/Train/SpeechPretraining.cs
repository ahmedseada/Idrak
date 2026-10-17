// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak pretrain</c> and <c>idrak resume</c> for a speech model (plan 21, S7): a folder whose config.json names an
/// architecture a speech registry holds (<see cref="SpeechFamilies"/>, <see cref="SpeechSynthesisFamilies"/>; a plug-in
/// given with <c>-P</c> registers it) trains on transcribed speech through <see cref="SpeechTrainer"/> (a recognizer: its
/// objective, "ctc" or "seq2seq", is the model's) or <see cref="SpeechSynthesisTrainer"/> (a synthesizer, on the label
/// durations of an alignments file). The run folder holds run.json (<see cref="RunFormat"/>), log.jsonl and the
/// checkpoints, which <c>resume</c> continues and <c>runs show</c> reads.
/// </summary>
internal static class SpeechPretraining
{
    public const string RunFormat = "idrak-speech-run/1";

    private const string Recognizer = "recognizer", Synthesizer = "synthesizer";

    /// <summary>The value options only a speech run reads.</summary>
    public static readonly string[] ValueOptions =
        ["--batch-seconds", "--metric", "--augment", "--clean", "--clean-settings", "--durations", "--duration-weight", "--min-seconds", "--max-seconds", "--decoder", "--beam", "--source"];

    /// <summary>The flags only a speech run reads.</summary>
    public static readonly string[] Flags = ["--skip-missing"];

    // The pretrain options a speech run does not read (they shape token sequences, images or a decoder's stages).
    private static readonly string[] TextOnly =
        ["--tokens", "--images", "--image-transform", "--vision", "--trained", "--max-length", "--batch-tokens", "--sequences", "--weight", "--top-k"];

    private static readonly string[] TextFlags = ["--recompute", "--no-packing", "--no-graphs"];

    /// <summary>
    /// Whether <paramref name="folder"/> holds a speech model: its config.json's architecture is registered as a speech
    /// recognizer or synthesizer family (and not as a decoder's, which <c>PretrainedModel</c> reads).
    /// </summary>
    public static bool Holds(string folder)
    {
        string config = Path.Combine(folder, "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        try
        {
            var json = JsonNode.Parse(File.ReadAllText(config)) as JsonObject;
            string? architecture = json?["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : (string?)json?["architecture"];
            return architecture is not null && (SpeechFamilies.Find(architecture) is not null || SpeechSynthesisFamilies.Find(architecture) is not null);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>Starts a speech training run of <paramref name="modelFolder"/> in <paramref name="run"/>.</summary>
    public static int Run(CommandContext context, string modelFolder, string run, int logEvery)
    {
        if (TextOnly.FirstOrDefault(o => context.Option(o) is not null || context.Options(o).Count > 0) is { } given)
        {
            throw new UsageException($"{given} is for a text or vision model's pretraining; {modelFolder} is a speech model (it trains on --data transcripts with --audio).");
        }

        if (TextFlags.FirstOrDefault(context.Flag) is { } flag)
        {
            throw new UsageException($"{flag} is for a text model's pretraining; {modelFolder} is a speech model.");
        }

        if (Directory.Exists(run) && SpeechTrainer.LatestCheckpoint(run) is not null)
        {
            throw new UsageException($"{run} holds a training run already; continue it with idrak resume {run}, or give another -o.");
        }

        var (options, train, evaluation) = Read(context);
        using var model = Load(context, modelFolder);
        Directory.CreateDirectory(run);
        File.WriteAllText(Path.Combine(run, "run.json"), new JsonObject
        {
            ["format"] = RunFormat,
            ["model"] = Path.GetFullPath(modelFolder),
            ["trainer"] = model.Synthesizer is null ? Recognizer : Synthesizer,
            ["objective"] = model.Objective,
            ["options"] = options.ToJson(),
            ["train"] = train.ToJson(),
            ["evaluation"] = evaluation?.ToJson(),
            ["log_every"] = logEvery,
        }.ToJsonString(CommandContext.JsonOutput) + "\n");
        context.Write($"{model.Objective} model {Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(modelFolder)))}: {model.Parameters / 1e6:F2}M parameters on {model.Device}");
        options = WithTeacher(context, model, options);
        return Execute(context, run, options.Steps, logEvery, model.Synthesizer is null, (progress, token, trace) => model.Synthesizer is { } synthesizer
            ? SpeechSynthesisTrainer.Train(synthesizer, train, evaluation, options, run, null, progress, token, trace)
            : SpeechTrainer.Train(model.Recognizer!, train, evaluation, options, run, null, progress, token, trace));
    }

    /// <summary>Continues a speech training run from its latest checkpoint (<c>idrak resume</c>).</summary>
    public static int Resume(CommandContext context, string run, JsonObject settings)
    {
        string checkpoint = SpeechTrainer.LatestCheckpoint(run)
            ?? throw new UsageException($"{run} has no checkpoint yet (the run stopped before its first was written); start it again with idrak pretrain.");
        var state = SpeechTrainer.ReadCheckpoint(checkpoint);
        var options = state.Options;
        if (context.Option("--steps") is not null)
        {
            int more = context.IntOption("--steps", 0);
            options = more > 0 ? options with { Steps = options.Steps + more } : throw new UsageException("--steps needs a positive number.");
        }

        bool synthesizer = (string?)settings["trainer"] == Synthesizer;
        if (context.Option("--teacher") is not null)
        {
            if (synthesizer)
            {
                throw new UsageException("--teacher teaches a CTC recognizer; this run trains a synthesizer.");
            }

            options = options with
            {
                Teacher = Teacher(context), DistillationAlpha = Real(context, "--alpha", options.DistillationAlpha),
                DistillationTemperature = Real(context, "--temperature", options.DistillationTemperature),
            };
        }

        context.Write($"Resuming  {Path.GetFileName(run)} after step {state.Step} of {options.Steps}");
        try
        {
            return Execute(context, run, options.Steps, (int?)settings["log_every"] ?? 10, !synthesizer, (progress, token, trace) => synthesizer
                ? SpeechSynthesisTrainer.Resume(checkpoint, options, context.Device, progress, token, trace)
                : SpeechTrainer.Resume(checkpoint, options, context.Device, progress, token, trace));
        }
        finally
        {
            (options.Teacher as IDisposable)?.Dispose();
        }
    }

    // The run's options, training source and evaluation source: --config's (SpeechTrainingOptions' JSON, with "train" and
    // "evaluation" sources), changed by the options given.
    private static (SpeechTrainingOptions Options, SpeechSource Train, SpeechSource? Evaluation) Read(CommandContext context)
    {
        var options = new SpeechTrainingOptions();
        SpeechSource? train = null, evaluation = null;
        if (context.Option("--config") is { } config)
        {
            try
            {
                var json = JsonNode.Parse(File.ReadAllText(config), documentOptions: CliConfig.ReadOptions) as JsonObject ?? throw new UsageException($"{config} is not a JSON object.");
                string folder = Path.GetDirectoryName(Path.GetFullPath(config))!;
                options = SpeechTrainingOptions.FromJson(json);
                if (options.Durations is { } relative)
                {
                    options = options with { Durations = Path.Combine(folder, relative) };
                }

                train = json["train"] is JsonObject t ? Resolved(SpeechSource.FromJson(t), folder) : null;
                evaluation = json["evaluation"] is JsonObject e ? Resolved(SpeechSource.FromJson(e), folder) : null;
            }
            catch (FileNotFoundException)
            {
                throw new UsageException($"No config file {config}.");
            }
            catch (Exception e) when (e is JsonException or InvalidDataException or InvalidOperationException or FormatException or KeyNotFoundException)
            {
                throw new UsageException($"{config}: {e.Message}");
            }
        }

        string? audio = context.Option("--audio") is { } a ? Path.GetFullPath(a) : null;
        SpeechSource Source(IReadOnlyList<string> paths) => new(context.Option("--source") ?? Kind(paths), [.. paths.Select(Path.GetFullPath)])
        {
            Audio = audio,
            MinSeconds = RealD(context, "--min-seconds", 0),
            MaxSeconds = RealD(context, "--max-seconds", 0),
            SkipMissing = context.Flag("--skip-missing"),
        };

        if (context.Options("--data") is { Count: > 0 } data)
        {
            train = Source(data);
        }

        if (context.Options("--eval") is { Count: > 0 } eval)
        {
            evaluation = Source(eval);
        }

        if (train is null)
        {
            throw new UsageException("Give what to train on: --data FILE (transcripts: JSON lines of {\"audio\", \"text\"}) or --data DIR (name.wav beside name.txt), "
                + "--audio DIR|ZIP where the transcripts' sounds are, or a --config with \"train\".");
        }

        bool? Switch(string on, string off) => context.Flag(on) ? true : context.Flag(off) ? false : null;
        options = options with
        {
            Steps = context.IntOption("--steps", options.Steps),
            BatchSeconds = context.Option("--batch-seconds") is { } batch && batch.Equals("auto", StringComparison.OrdinalIgnoreCase) ? 0 : RealD(context, "--batch-seconds", options.BatchSeconds),
            GradientAccumulation = context.IntOption("--accumulate", options.GradientAccumulation),
            LearningRate = Real(context, "--lr", options.LearningRate),
            MinLearningRate = Real(context, "--min-lr", options.MinLearningRate),
            WarmupFraction = Real(context, "--warmup", options.WarmupFraction),
            Schedule = context.Option("--schedule") ?? options.Schedule,
            Optimizer = context.Option("--optimizer") ?? options.Optimizer,
            WeightDecay = Real(context, "--weight-decay", options.WeightDecay),
            MaxGradientNorm = Real(context, "--clip", options.MaxGradientNorm),
            HostOptimizer = context.Flag("--host-optimizer") || options.HostOptimizer,
            BFloat16Activations = context.Flag("--bf16-activations") ? true : options.BFloat16Activations,
            Checkpointing = Switch("--checkpointing", "--no-checkpointing") ?? options.Checkpointing,
            SaveEvery = context.IntOption("--save-every", options.SaveEvery),
            KeepCheckpoints = context.IntOption("--keep", options.KeepCheckpoints),
            EvaluateEvery = context.IntOption("--eval-every", options.EvaluateEvery),
            Metric = context.Option("--metric") ?? options.Metric,
            Decoder = context.Option("--decoder") ?? options.Decoder,
            BeamWidth = context.IntOption("--beam", options.BeamWidth),
            Augmentations = context.Option("--augment") is { } augment ? [.. augment.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)] : options.Augmentations,
            Cleaning = AudioFiles.CleaningStages(context, options.Cleaning),
            CleaningOptions = AudioFiles.CleaningSettings(context, options.CleaningOptions),
            Seed = context.Seed ?? options.Seed,
            FeatureCache = context.Option("--feature-cache") is { } cache ? cache.Equals("none", StringComparison.OrdinalIgnoreCase) ? null : cache : options.FeatureCache,
            Durations = context.Option("--durations") is { } durations ? Path.GetFullPath(durations) : options.Durations,
            DurationLossWeight = Real(context, "--duration-weight", options.DurationLossWeight),
        };
        return (options, train, evaluation);
    }

    // A source's kind from its paths: folders and zips hold name.wav beside name.txt, files are transcripts.
    internal static string Kind(IReadOnlyList<string> paths) =>
        paths.All(p => Directory.Exists(p) || p.EndsWith(".zip", StringComparison.OrdinalIgnoreCase)) ? SpeechSources.AudioFolder : SpeechSources.Transcripts;

    // A config's source with its paths relative to the config's folder.
    private static SpeechSource Resolved(SpeechSource source, string folder) => source with
    {
        Paths = [.. source.Paths.Select(p => Path.Combine(folder, p))],
        Audio = source.Audio is { } audio ? Path.Combine(folder, audio) : null,
    };

    // The teacher (--teacher MODEL): a CTC recognizer, its own device with --teacher-device.
    private static ICtcSpeechModel Teacher(CommandContext context)
    {
        string path = context.Option("--teacher")!;
        var device = context.Option("--teacher-device") is { } td ? Idrak.Abstraction.Device.Parse(td) : context.Device;
        var recognizer = SpeechModels.Load(path, new SpeechModelOptions { Device = device });
        if (recognizer is ICtcSpeechModel ctc)
        {
            return ctc;
        }

        recognizer.Dispose();
        throw new UsageException($"--teacher {path} is a {recognizer.GetType().Name}: a speech student learns frame by frame from a CTC model of its labels; "
            + "a model that writes text teaches through pseudo-labels (idrak transcribe TEACHER clips -o train.jsonl -f json, then --data train.jsonl).");
    }

    private static SpeechTrainingOptions WithTeacher(CommandContext context, LoadedSpeechModel model, SpeechTrainingOptions options)
    {
        if (context.Option("--teacher") is null)
        {
            return options;
        }

        if (model.Synthesizer is not null)
        {
            throw new UsageException("--teacher teaches a CTC recognizer; this model is a synthesizer.");
        }

        var teacher = Teacher(context);
        model.Owned.Add(teacher);
        options = options with
        {
            Teacher = teacher, DistillationAlpha = Real(context, "--alpha", options.DistillationAlpha),
            DistillationTemperature = Real(context, "--temperature", options.DistillationTemperature),
        };
        context.Write(FormattableString.Invariant($"teacher {context.Option("--teacher")}: temperature {options.DistillationTemperature}, alpha {options.DistillationAlpha}"));
        return options;
    }

    // The model a folder holds: a recognizer its family reads, or else a synthesizer (one architecture may name both, the
    // config telling which); a family that refuses both is said with both reasons.
    private static LoadedSpeechModel Load(CommandContext context, string folder)
    {
        string architecture = (string?)(JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))?["architectures"]?[0]) ?? "";
        Exception? refused = null;
        if (SpeechFamilies.Find(architecture) is not null)
        {
            try
            {
                var recognizer = SpeechModels.Load(folder, new SpeechModelOptions { Device = context.Device });
                if (recognizer is ITrainableSpeechModel trainable)
                {
                    return new LoadedSpeechModel(trainable, null, recognizer);
                }

                recognizer.Dispose();
                throw new UsageException($"{folder} holds a {recognizer.GetType().Name}, which does not train (its family gives no ITrainableSpeechModel).");
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException && SpeechSynthesisFamilies.Find(architecture) is not null)
            {
                refused = e;                                                    // a synthesizer's task, say
            }
        }

        try
        {
            var synthesizer = SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = context.Device });
            if (synthesizer is ITrainableSpeechSynthesizer trainable)
            {
                return new LoadedSpeechModel(null, trainable, synthesizer);
            }

            synthesizer.Dispose();
            throw new UsageException($"{folder} holds a {synthesizer.GetType().Name}, which does not train (its family gives no ITrainableSpeechSynthesizer).");
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException && refused is not null)
        {
            throw new UsageException($"{folder} is neither a recognizer nor a synthesizer its family trains: {refused.Message} {e.Message}");
        }
    }

    private static int Execute(CommandContext context, string run, int total, int logEvery, bool recognizer,
        Func<IProgress<SpeechTrainingProgress>, CancellationToken, Action<string>?, SpeechTrainingResult> train)
    {
        using var log = new TelemetryAppender(Path.Combine(run, "log.jsonl"));
        using var subscription = Telemetry.Subscribe(log);
        using var interrupt = new Interrupt(context);
        SpeechTrainingResult result;
        using (var line = new ProgressLine(context, "training", total: total, unit: ProgressUnit.Items))
        {
            try
            {
                result = train(new Reports(context, line, logEvery), interrupt.Token, context.Verbose ? context.Detail : null);
            }
            catch (OperationCanceledException) when (interrupt.Requested)
            {
                line.Erase();
                string? last = SpeechTrainer.LatestCheckpoint(run);
                context.Write(interrupt.TimedOut ? "gave up after --timeout" : "stopped");
                context.Write(last is null ? "no checkpoint was written (no step finished)" : $"checkpoint {last}; idrak resume {run} continues");
                context.WriteJson(new JsonObject { ["run"] = Path.GetFullPath(run), ["stopped"] = true, ["checkpoint"] = last });
                return ExitCodes.Failed;
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or FileNotFoundException)
            {
                line.Erase();
                throw new UsageException(e.Message);
            }

            line.Finish();
        }

        foreach (string note in result.Notes)
        {
            context.Write($"note: {note}");
        }

        (int Step, double Score)? score = result.Evaluations.Count > 0 ? result.Evaluations[^1] : null;
        context.Write(FormattableString.Invariant($"{result.Steps} steps in {result.Elapsed.TotalMinutes:F1} min on {result.Utterances:N0} utterances, loss {(result.Losses.Count > 0 ? result.Losses[^1] : float.NaN):F4}")
            + (score is { } s ? FormattableString.Invariant($", held-out {s.Score:F4} at step {s.Step}") : "")
            + FormattableString.Invariant($" (features {result.FeaturesExtracted:N0} taken, {result.FeaturesCached:N0} cached)"));
        context.Write(result.Checkpoint is null ? "no checkpoint was written"
            : recognizer ? $"model {result.Checkpoint}; next: idrak transcribe \"{result.Checkpoint}\" clip.wav"
            : $"model {result.Checkpoint}; next: idrak speak \"{result.Checkpoint}\" \"...\" -o out.wav");
        context.WriteJson(new JsonObject
        {
            ["run"] = Path.GetFullPath(run), ["steps"] = result.Steps, ["checkpoint"] = result.Checkpoint, ["utterances"] = result.Utterances, ["dropped"] = result.Dropped,
            ["final_loss"] = result.Losses.Count > 0 && float.IsFinite(result.Losses[^1]) ? result.Losses[^1] : null,
            ["evaluations"] = new JsonArray([.. result.Evaluations.Select(e => (JsonNode)new JsonObject { ["step"] = e.Step, ["score"] = e.Score })]),
            ["seconds"] = Math.Round(result.Elapsed.TotalSeconds, 3), ["notes"] = new JsonArray([.. result.Notes.Select(n => (JsonNode)n)]),
        });
        return ExitCodes.Ok;
    }

    private static float Real(CommandContext context, string option, float otherwise) => (float)RealD(context, option, otherwise);

    private static double RealD(CommandContext context, string option, double otherwise) => context.Option(option) is not { } text ? otherwise
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : throw new UsageException($"{option} is a number, not '{text}'.");

    // A speech model loaded for training: the recognizer or the synthesizer, and what the run disposes (the model, a teacher).
    private sealed class LoadedSpeechModel : IDisposable
    {
        public LoadedSpeechModel(ITrainableSpeechModel? recognizer, ITrainableSpeechSynthesizer? synthesizer, IDisposable model)
        {
            Recognizer = recognizer;
            Synthesizer = synthesizer;
            Owned.Add(model);
        }

        public ITrainableSpeechModel? Recognizer { get; }

        public ITrainableSpeechSynthesizer? Synthesizer { get; }

        public List<object> Owned { get; } = [];

        public string Objective => Recognizer?.Objective ?? Synthesizer!.Objective;

        public Idrak.Abstraction.Device Device => Recognizer?.Device ?? Synthesizer!.Device;

        public long Parameters => (Recognizer?.Network ?? Synthesizer!.Network).ParameterCount;

        public void Dispose()
        {
            foreach (var owned in Owned)
            {
                (owned as IDisposable)?.Dispose();
            }
        }
    }

    // The live line, and a line every N steps (loss, rate, real-time factor, memory, time left, the held-out score).
    private sealed class Reports(CommandContext context, ProgressLine line, int every) : IProgress<SpeechTrainingProgress>
    {
        private double? _micro;

        public void Report(SpeechTrainingProgress p)
        {
            // A step's last report ("step") carries its held-out score too, so the "evaluate" and "checkpoint" reports
            // before it print nothing: one line a step.
            if (p.Stage != "step")
            {
                return;
            }

            // The device ran out of memory: say so when it happens (the run's notes repeat it at the end).
            if (_micro is { } before && p.MicroBatchSeconds < before)
            {
                line.Erase();
                context.Write(FormattableString.Invariant(
                    $"step {p.Step}: out of device memory, now micro-batches of {p.MicroBatchSeconds:0.##} audio seconds (gradients added up: the same result, slower)"));
            }

            _micro = p.MicroBatchSeconds;

            line.Report(p.Step);
            if (every > 0 && (p.Step % every == 0 || p.Step == p.Steps) || p.Evaluation is not null)
            {
                line.Erase();
                context.Write(FormattableString.Invariant($"step {p.Step}/{p.Steps} (epoch {p.Epoch} {p.EpochFraction:P0})  loss {p.Loss:F4}  lr {p.LearningRate:G3}  {p.AudioSecondsPerSecond:F1}x real time")
                    + (p.PeakDeviceBytes > 0 ? FormattableString.Invariant($"  {p.PeakDeviceBytes / 1048576.0:F0} MiB") : "")
                    + FormattableString.Invariant($"  {p.Remaining:hh\\:mm\\:ss} left")
                    + (p.Evaluation is { } e ? FormattableString.Invariant($"  held-out {e:F4}") : ""));
            }
        }
    }
}
