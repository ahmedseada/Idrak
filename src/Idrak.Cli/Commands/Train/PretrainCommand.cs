// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak pretrain MODEL -o RUN</c>: every parameter of a model trained on token files and records, in stages
/// (<see cref="Pretrainer"/>, plan 19), with checkpoints that <c>idrak resume</c> continues.
/// </summary>
internal sealed class PretrainCommand : Command
{
    public const string RunFormat = "idrak-pretrain-run/1";

    public override string Name => "pretrain";

    public override string Summary => "Train every weight of a model (one made with idrak create) on token files and records, in stages";

    public override string Usage => """
        MODEL [--config pretrain.json] [-o RUN] [options]

        Trains every parameter of MODEL (a folder idrak create wrote, or any float32 model) on token files (idrak data
        tokens writes them) and conversation records, in the stages of --config, or one stage from the options below.
        Each step draws sequences from the stage's sources by their weights, packs them into rows of --max-length and
        trains on them; the device's memory decides batches, recomputation and checkpointing as it does for idrak tune.
        Checkpoints (RUN/checkpoint-STEP) are model folders with the optimizer's state; Ctrl+C writes one, and idrak
        resume RUN continues where the run stopped. idrak pretrain init writes a pretrain.json to start from.

        A speech model (its family registered by a plug-in, -P: a recognizer whose objective is ctc or seq2seq, or a
        synthesizer) trains on transcribed speech instead: --data transcripts.jsonl ({"audio", "text"} rows; a folder or
        zip of name.wav beside name.txt) with --audio DIR|ZIP, the step, rate and checkpoint options below, and these:
        --batch-seconds S (audio seconds a step, 160), --metric cer|wer (the held-out score), --decoder greedy|beam and
        --beam N (how a CTC model reads the held-out sounds), --augment speed,noise,spec-mask, --min-seconds and
        --max-seconds S (utterances left out), --skip-missing (rows whose sound is missing), --source KIND (the speech
        source kind; told from the paths), --feature-cache memory|disk|none (disk); a synthesizer reads --durations
        alignments.jsonl (idrak data audio --align MODEL writes it) and --duration-weight W (1); a CTC recognizer learns
        from a CTC --teacher's frame posteriors with --alpha and --temperature (0.5, 2). -C takes the trainer's options
        as JSON (batch_seconds, learning_rate, …, with "train" and "evaluation" sources).

        Arguments:
          MODEL  a model folder (idrak create writes one)

        Options:
          -C, --config FILE        the run's stages and settings (pretrain.json); the options below then change it
              --tokens FILE        a token file to train on (repeat for several; one stage)
              --data FILE          conversation records to train on (messages or sharegpt JSON Lines; repeat)
              --images DIR|ZIP     where the records' images are
              --audio DIR|ZIP      where the records' sounds are (an audio-language model's records)
              --image-transform T  how record images are prepared (grayscale, max_width=1024, …; the vision families')
              --vision K=V         a vision option of the model's family (repeat)
              --feature-cache NAME where image features of frozen stages are kept: memory or disk (memory)
              --sequences N        exactly N sequences a step (by default a step's rows are filled)
              --teacher MODEL      learn a teacher's token distributions too (a model of the same vocabulary; one that
                                   reads images for image records)
              --temperature T      the distillation temperature (1)
              --alpha A            the teacher's share of the loss, 0 to 1 (0.5)
              --top-k K            keep the teacher's K most likely tokens per position (all)
              --teacher-device D   the teacher's device (the model's)
              --weight NAME=W      a source's share of each step: tokens=3, data=1 (1 each)
              --steps N            the stage's optimizer steps (1000)
              --trained PARTS      what trains: decoder, embeddings, head, tower, projector, the model's parts'
                                   (audio-encoder, audio-projector), all (decoder,embeddings,head)
              --lr X               the peak learning rate (3e-4)
              --min-lr X           the rate at the end (0)
              --warmup F           the share of steps warming up (0.03)
              --schedule NAME      cosine, linear, constant or wsd (cosine)
              --max-length N       the window and row length in tokens (2048, at most the model's)
              --batch-tokens N     positions per micro-batch (4096)
              --accumulate N       micro-batches per step (1)
              --optimizer NAME     adamw, adam, adamw8bit or sgd (adamw)
              --weight-decay X     (0.1)
              --clip X             the gradient norm clipped to, 0 for none (1)
              --save-every N       steps between checkpoints (each stage's end only)
              --keep N             checkpoints kept, 0 for all (3)
              --eval FILE          held-out token files or records (repeat), scored at each stage's end or --eval-every
              --eval-every N       steps between evaluations
              --checkpointing      recompute each block's activations from the start (else measured)
              --no-checkpointing   never
              --recompute          recompute the feed-forward activations from the start
              --bf16-activations   hold activations as bfloat16 between the passes
              --host-optimizer     the optimizer's update on the CPU, its state in system memory
              --no-packing         one sequence a row
              --no-graphs          no recorded step graphs
              --log-every N        a line every N steps (10)
              --seed N             the data order's seed (0)
          -o, --out RUN            the run folder (a new one under the runs folder)

        Examples:
          idrak data tokens corpus.txt --tokenizer tokenizer.json -o tokens.bin
          idrak create spec.json -o my-model
          idrak pretrain my-model --tokens tokens.bin --steps 2000 --lr 6e-4 -o run -d cuda:0
          idrak pretrain my-vlm --data pages.jsonl --images pages.zip --trained tower,projector -P Idrak.IdrakVision.dll
          idrak pretrain my-audio-lm --data talks.jsonl --audio clips.zip --trained audio-encoder,audio-projector -P Idrak.IdrakAudio.dll
          idrak pretrain my-ctc --data train.jsonl --audio clips --eval dev.jsonl --batch-seconds 120 -P Idrak.IdrakAudio.dll
          idrak pretrain my-tts --data train.jsonl --audio clips --durations alignments.jsonl -P Idrak.IdrakAudio.dll
          idrak pretrain init -o pretrain.json
          idrak pretrain my-model -C pretrain.json -o run
          idrak resume run
          idrak runs show run
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--tokens", "--data", "--images", "--weight", "--steps", "--trained", "--lr", "--min-lr", "--warmup", "--schedule", "--max-length",
        "--batch-tokens", "--accumulate", "--optimizer", "--weight-decay", "--clip", "--save-every", "--keep", "--eval", "--eval-every", "--log-every", "--out",
        "--image-transform", "--vision", "--audio", "--feature-cache", "--sequences", "--teacher", "--temperature", "--alpha", "--top-k", "--teacher-device",
        .. SpeechPretraining.ValueOptions,
    ];

    public override IReadOnlyCollection<string> Flags { get; } =
        ["--checkpointing", "--no-checkpointing", "--recompute", "--bf16-activations", "--host-optimizer", "--no-packing", "--no-graphs", .. SpeechPretraining.Flags];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string modelFolder = context.Argument(0, "MODEL (a model folder; idrak create writes one)");
        string run = context.Option("--out") ?? NewRunFolder(context, Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(modelFolder))));
        if (SpeechPretraining.Holds(modelFolder))
        {
            return SpeechPretraining.Run(context, modelFolder, run, context.IntOption("--log-every", 10));   // a recognizer or synthesizer
        }

        if (SpeechPretraining.ValueOptions.FirstOrDefault(o => context.Option(o) is not null) is { } speech)
        {
            throw new UsageException($"{speech} is for a speech model's training; {modelFolder} is not one (a speech family registered by a plug-in, -P, reads one).");
        }

        var options = Options(context);
        if (Directory.Exists(run) && Pretrainer.LatestCheckpoint(run) is not null)
        {
            throw new UsageException($"{run} holds a pretraining run already; continue it with idrak resume {run}, or give another -o.");
        }

        Directory.CreateDirectory(run);
        File.WriteAllText(Path.Combine(run, "run.json"), new JsonObject
        {
            ["format"] = RunFormat,
            ["model"] = Path.GetFullPath(modelFolder),
            ["options"] = options.ToJson(),
            ["log_every"] = context.IntOption("--log-every", 10),
        }.ToJsonString(CommandContext.JsonOutput) + "\n");
        return Execute(context, modelFolder, options, run, null, context.IntOption("--log-every", 10));
    }

    /// <summary>Continues a pretraining run from its latest checkpoint (<c>idrak resume</c>).</summary>
    public static int Resume(CommandContext context, string run, JsonObject settings)
    {
        string checkpoint = Pretrainer.LatestCheckpoint(run)
            ?? throw new UsageException($"{run} has no checkpoint yet (the run stopped before its first was written); start it again with idrak pretrain.");
        var state = Pretrainer.ReadCheckpoint(checkpoint);
        var options = state.Options;
        if (context.Option("--steps") is not null)
        {
            int more = context.IntOption("--steps", 0);
            if (more <= 0)
            {
                throw new UsageException("--steps needs a positive number.");
            }

            var stages = options.Stages.ToList();
            stages[^1] = stages[^1] with { Steps = stages[^1].Steps + more };
            options = options with { Stages = stages };
            state = state with { Options = options };
        }

        context.Write($"Resuming  {Path.GetFileName(run)} after step {state.Step} ({options.Stages[Math.Min(state.Stage, options.Stages.Count - 1)].Name})");
        return Execute(context, checkpoint, options, run, state, (int?)settings["log_every"] ?? 10);
    }

    private static int Execute(CommandContext context, string modelFolder, PretrainingOptions options, string run, PretrainingCheckpoint? resume, int logEvery)
    {
        PretrainedModel model;
        try
        {
            model = PretrainedModel.Load(modelFolder, new PretrainedOptions { Device = context.Device });
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }

        using (model)
        {
            if (options.MaxLength > model.MaxPositions)
            {
                options = options with { MaxLength = model.MaxPositions };
                context.Write($"--max-length is the model's {model.MaxPositions} positions");
            }

            // A teacher (not kept in checkpoints: given again on resume).
            using var teacherModel = context.Option("--teacher") is { } teacherPath
                ? PretrainedModel.Load(teacherPath, new PretrainedOptions { Device = context.Option("--teacher-device") is { } td ? Idrak.Abstraction.Device.Parse(td) : context.Device })
                : null;
            using var teacher = teacherModel is null ? null : DistillationTeachers.FromModel(teacherModel, context.IntOption("--top-k", 0), options.BatchTokens, options.Images);
            if (teacher is not null)
            {
                float Real(string option, float otherwise) => context.Option(option) is not { } text ? otherwise
                    : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : throw new UsageException($"{option} is a number, not '{text}'.");
                options = options with
                {
                    Teacher = teacher,
                    DistillationTemperature = Real("--temperature", options.DistillationTemperature),
                    DistillationAlpha = Real("--alpha", options.DistillationAlpha),
                };
                context.Write($"teacher {context.Option("--teacher")}: {teacherModel!.Spec.ParameterCount / 1e6:F1}M parameters, temperature {options.DistillationTemperature}, alpha {options.DistillationAlpha}");
            }

            int total = options.Stages.Sum(s => s.Steps);
            using var log = new TelemetryAppender(Path.Combine(run, "log.jsonl"));
            using var subscription = Telemetry.Subscribe(log);
            using var interrupt = new Interrupt(context);
            PretrainingResult result;
            using (var line = new ProgressLine(context, "pretraining", total: total, unit: ProgressUnit.Items))
            {
                var reports = new Reports(context, line, logEvery);
                line.Report(resume?.Step ?? 0);
                try
                {
                    result = Pretrainer.Train(model, options, run, resume, reports, interrupt.Token, context.Verbose ? context.Detail : null);
                }
                catch (OperationCanceledException) when (interrupt.Requested)
                {
                    line.Erase();
                    string? last = Pretrainer.LatestCheckpoint(run);
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
                catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
                {
                    line.Erase();
                    return ImageModelFiles.Refuse(context, e);
                }

                line.Finish();
            }

            context.Write(FormattableString.Invariant($"{result.Steps} steps in {result.Elapsed.TotalMinutes:F1} min, loss {(result.Losses.Count > 0 ? result.Losses[^1] : float.NaN):F4}")
                + (result.EvaluationLosses.Count > 0 ? FormattableString.Invariant($", held-out {result.EvaluationLosses[^1].Loss:F4}") : "") + (result.Settings.Length > 0 ? $" ({result.Settings})" : ""));
            context.Write($"model {result.LastCheckpoint}; next: idrak run \"{result.LastCheckpoint}\" \"...\", or idrak tune it");
            context.WriteJson(new JsonObject
            {
                ["run"] = Path.GetFullPath(run), ["steps"] = result.Steps, ["checkpoint"] = result.LastCheckpoint,
                ["final_loss"] = result.Losses.Count > 0 && float.IsFinite(result.Losses[^1]) ? result.Losses[^1] : null,
                ["evaluations"] = new JsonArray([.. result.EvaluationLosses.Select(e => (JsonNode)new JsonObject { ["step"] = e.Step, ["loss"] = e.Loss })]),
                ["seconds"] = Math.Round(result.Elapsed.TotalSeconds, 3), ["settings"] = result.Settings,
            });
            return ExitCodes.Ok;
        }
    }

    // The run's options: --config's, changed by the options given, or one stage from them.
    private static PretrainingOptions Options(CommandContext context)
    {
        PretrainingOptions options;
        if (context.Option("--config") is { } config)
        {
            try
            {
                var json = JsonNode.Parse(File.ReadAllText(config), documentOptions: CliConfig.ReadOptions) as JsonObject ?? throw new UsageException($"{config} is not a JSON object.");
                options = PretrainingOptions.FromJson(Resolved(json, Path.GetDirectoryName(Path.GetFullPath(config))!));
            }
            catch (FileNotFoundException)
            {
                throw new UsageException($"No config file {config}.");
            }
            catch (Exception e) when (e is JsonException or InvalidDataException or InvalidOperationException or FormatException)
            {
                throw new UsageException($"{config}: {e.Message}");
            }
        }
        else
        {
            options = new PretrainingOptions();
        }

        var weights = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (string pair in context.Options("--weight"))
        {
            int eq = pair.IndexOf('=');
            if (eq <= 0 || !double.TryParse(pair[(eq + 1)..], NumberStyles.Float, CultureInfo.InvariantCulture, out double w))
            {
                throw new UsageException($"--weight is NAME=SHARE (tokens=3, data=1), not '{pair}'.");
            }

            weights[pair[..eq]] = w;
        }

        var sources = new List<PretrainingSource>();
        if (context.Options("--tokens") is { Count: > 0 } tokens)
        {
            sources.Add(new PretrainingSource(PretrainingSources.Tokens, tokens) { Weight = weights.GetValueOrDefault("tokens", 1) });
        }

        if (context.Options("--data") is { Count: > 0 } data)
        {
            sources.Add(new PretrainingSource(PretrainingSources.Records, data) { Weight = weights.GetValueOrDefault("data", 1), Images = context.Option("--images"), Audios = context.Option("--audio") });
        }

        float Real(string option, float otherwise) => context.Option(option) is not { } text ? otherwise
            : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : throw new UsageException($"{option} is a number, not '{text}'.");

        var stage = options.Stages.Count == 1 ? options.Stages[0] : new PretrainingStage("text") { Steps = 1000 };
        bool stageOptions = sources.Count > 0 || new[] { "--steps", "--trained", "--lr", "--min-lr", "--warmup", "--schedule" }.Any(o => context.Option(o) is not null);
        if (stageOptions)
        {
            if (options.Stages.Count > 1)
            {
                throw new UsageException("The config has several stages: change them in the file (the stage options set one stage).");
            }

            stage = stage with
            {
                Sources = sources.Count > 0 ? sources : stage.Sources,
                Steps = context.IntOption("--steps", stage.Steps),
                Trained = context.Option("--trained") is { } trained ? [.. trained.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)] : stage.Trained,
                LearningRate = Real("--lr", stage.LearningRate),
                MinLearningRate = Real("--min-lr", stage.MinLearningRate),
                WarmupFraction = Real("--warmup", stage.WarmupFraction),
                Schedule = context.Option("--schedule") ?? stage.Schedule,
            };
            options = options with { Stages = [stage] };
        }

        if (options.Stages.Count == 0 || options.Stages.Any(s => s.Sources.Count == 0))
        {
            throw new UsageException("Give what to train on: --tokens FILE (idrak data tokens writes one), --data FILE, or a --config with stages.");
        }

        bool? Switch(string on, string off) => context.Flag(on) ? true : context.Flag(off) ? false : null;
        return options with
        {
            MaxLength = context.IntOption("--max-length", options.MaxLength),
            BatchTokens = context.IntOption("--batch-tokens", options.BatchTokens),
            GradientAccumulation = context.IntOption("--accumulate", options.GradientAccumulation),
            Optimizer = context.Option("--optimizer") ?? options.Optimizer,
            WeightDecay = Real("--weight-decay", options.WeightDecay),
            MaxGradientNorm = Real("--clip", options.MaxGradientNorm),
            SaveEvery = context.IntOption("--save-every", options.SaveEvery),
            KeepCheckpoints = context.IntOption("--keep", options.KeepCheckpoints),
            EvaluateEvery = context.IntOption("--eval-every", options.EvaluateEvery),
            Evaluation = context.Options("--eval") is { Count: > 0 } eval
                ? [.. eval.Select(e => new PretrainingSource(e.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase) || e.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                    ? PretrainingSources.Records : PretrainingSources.Tokens, [e]) { Images = context.Option("--images"), Audios = context.Option("--audio") })]
                : options.Evaluation,
            Checkpointing = Switch("--checkpointing", "--no-checkpointing") ?? options.Checkpointing,
            RecomputeFeedForward = context.Flag("--recompute") ? true : options.RecomputeFeedForward,
            BFloat16Activations = context.Flag("--bf16-activations") ? true : options.BFloat16Activations,
            HostOptimizer = context.Flag("--host-optimizer") || options.HostOptimizer,
            Packing = !context.Flag("--no-packing") && options.Packing,
            CudaGraphs = !context.Flag("--no-graphs") && options.CudaGraphs,
            Seed = context.Seed ?? options.Seed,
            SequencesPerStep = context.IntOption("--sequences", options.SequencesPerStep),
            FeatureCache = context.Option("--feature-cache") ?? options.FeatureCache,
            Images = context.Option("--image-transform") is not null || context.Options("--vision").Count > 0
                ? TuningImages.Parse(context.Option("--image-transform"), context.Options("--vision"))
                : options.Images,
        };
    }

    // A config's source paths read from its own folder.
    private static JsonObject Resolved(JsonObject json, string folder)
    {
        IEnumerable<JsonObject> Sources(JsonNode? stages) => stages is JsonArray a ? a.OfType<JsonObject>().SelectMany(s => s["sources"] is JsonArray ss ? ss.OfType<JsonObject>() : []) : [];
        foreach (var source in Sources(json["stages"]).Concat(json["eval"] is JsonArray e ? e.OfType<JsonObject>() : []))
        {
            if (source["paths"] is JsonArray paths)
            {
                source["paths"] = new JsonArray([.. paths.Select(p => (JsonNode)Path.Combine(folder, (string)p!))]);
            }

            if ((string?)source["images"] is { } images)
            {
                source["images"] = Path.Combine(folder, images);
            }

            if ((string?)source["audios"] is { } audios)
            {
                source["audios"] = Path.Combine(folder, audios);
            }
        }

        return json;
    }

    private static string NewRunFolder(CommandContext context, string name)
    {
        string safe = new([.. name.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '-')]);
        string folder = Path.Combine(RunLog.RunsFolder(context), string.Create(CultureInfo.InvariantCulture, $"{DateTime.Now:yyyyMMdd-HHmmss}-pretrain-{safe}"));
        for (int i = 2; Directory.Exists(folder); i++)
        {
            folder = Path.Combine(RunLog.RunsFolder(context), string.Create(CultureInfo.InvariantCulture, $"{DateTime.Now:yyyyMMdd-HHmmss}-pretrain-{safe}-{i}"));
        }

        return folder;
    }

    // The live line, and a line every N steps (loss, rate, speed, memory, time left).
    private sealed class Reports(CommandContext context, ProgressLine line, int every) : IProgress<PretrainingProgress>
    {
        private int _stage = -1;

        public void Report(PretrainingProgress p)
        {
            if (p.StageIndex != _stage)
            {
                _stage = p.StageIndex;
                line.Erase();
                context.Write($"stage {p.StageIndex + 1}: {p.Stage}, {p.StageSteps} steps");
            }

            line.Report(p.Step);
            if (every > 0 && (p.StageStep % every == 0 || p.StageStep == p.StageSteps) || p.EvaluationLoss is not null)
            {
                line.Erase();
                context.Write(FormattableString.Invariant($"step {p.Step}/{p.TotalSteps} ({p.Stage} {p.StageStep}/{p.StageSteps})  loss {p.Loss:F4}  lr {p.LearningRate:G3}  {p.TokensPerSecond:F0} tok/s")
                    + (p.ImagesPerSecond > 0 ? FormattableString.Invariant($"  {p.ImagesPerSecond:F1} img/s") : "")
                    + (p.DeviceBytes > 0 ? FormattableString.Invariant($"  {p.DeviceBytes / 1048576.0:F0} MiB") : "")
                    + FormattableString.Invariant($"  {p.Remaining:hh\\:mm\\:ss} left")
                    + (p.EvaluationLoss is { } e ? FormattableString.Invariant($"  held-out {e:F4}") : ""));
            }
        }
    }
}

/// <summary>Appends the library's telemetry records to a run's log (kept across resumes; <c>idrak runs show</c> reads it).</summary>
internal sealed class TelemetryAppender(string path) : ITelemetryHook, IDisposable
{
    private readonly FileStream _stream = new(path, FileMode.Append, FileAccess.Write, FileShare.Read);
    private readonly object _gate = new();

    public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches;

    public void OnTrainingStarted(in TrainingStarted e) => Write(e);

    public void OnBatchCompleted(in BatchCompleted e) => Write(e);

    public void OnTrainingCompleted(in TrainingCompleted e) => Write(e);

    private void Write(object e)
    {
        lock (_gate)
        {
            using (var writer = new Utf8JsonWriter(_stream))
            {
                TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.Now, e));
            }

            _stream.WriteByte((byte)'\n');
            _stream.Flush();
        }
    }

    public void Dispose() => _stream.Dispose();
}

/// <summary><c>idrak pretrain init</c>: a commented pretrain.json with two stages to start from.</summary>
internal sealed class PretrainInitCommand : Command
{
    public override string Name => "pretrain init";

    public override string Summary => "Write a pretrain.json (stages, sources and settings) to start a pretraining run from";

    public override string Usage => """
        [-o pretrain.json]

        Writes the stages and settings of a pretraining run for idrak pretrain --config: a text stage on token files,
        then a stage on records, every key with a comment. Comments and trailing commas are allowed; paths are read from
        the file's folder.

        Options:
          -o, --out FILE  the file to write (pretrain.json)

        Examples:
          idrak pretrain init -o pretrain.json
          idrak pretrain my-model -C pretrain.json -o run
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? "pretrain.json";
        if (File.Exists(output))
        {
            context.Error($"{output} exists; give another -o.");
            return ExitCodes.Failed;
        }

        File.WriteAllText(output, """
            {
              // idrak pretrain's run (plan 19). Paths are read from this file's folder.
              "format": "idrak-pretrain/1",
              "stages": [
                {
                  "name": "text",
                  // decoder, embeddings, head, tower, projector, or all
                  "trained": ["decoder", "embeddings", "head"],
                  // tokens: token files (idrak data tokens); records: messages or sharegpt JSON Lines, with "images"
                  "sources": [ { "kind": "tokens", "paths": ["tokens.bin"], "weight": 1 } ],
                  "steps": 2000,
                  "learning_rate": 6e-4,
                  "min_learning_rate": 6e-5,
                  "warmup": 0.03,
                  "schedule": "cosine"
                },
                {
                  "name": "answers",
                  "trained": ["all"],
                  "sources": [
                    { "kind": "tokens", "paths": ["tokens.bin"], "weight": 1 },
                    { "kind": "records", "paths": ["records.jsonl"], "weight": 1 }
                  ],
                  "steps": 500,
                  "learning_rate": 1e-4,
                  "schedule": "cosine"
                }
              ],
              "max_length": 1024,      // window and row length (at most the model's positions)
              "batch_tokens": 8192,    // positions per micro-batch
              "accumulate": 1,         // micro-batches per step
              "optimizer": "adamw",    // adamw, adam, adamw8bit, sgd
              "weight_decay": 0.1,
              "beta1": 0.9,
              "beta2": 0.95,
              "clip": 1.0,
              "save_every": 500,       // 0: at each stage's end only
              "keep_checkpoints": 3,
              "eval": [],              // held-out sources, as in "sources"
              "eval_every": 0,
              "seed": 0
              // checkpointing, recompute and bf16_activations are measured on the device unless set (true or false).
            }

            """);
        context.Write($"wrote {output}; next: idrak pretrain MODEL -C {output} -o run");
        context.WriteJson(new JsonObject { ["path"] = Path.GetFullPath(output) });
        return ExitCodes.Ok;
    }
}
