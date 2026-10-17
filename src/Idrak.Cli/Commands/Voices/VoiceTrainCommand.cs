// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Voices;

/// <summary>
/// <c>idrak voice train NAME --from DATA --base MODEL --consent TEXT</c> (plan 21, S9 route C): fine-tunes a synthesizer
/// its plug-in trains (an <see cref="IAdversarialSpeechSynthesizer"/>: VITS) on one person's transcribed recordings with
/// <see cref="AdversarialSynthesisTrainer"/>, then installs the last checkpoint as the voice NAME of kind
/// <see cref="VoiceKinds.Checkpoint"/>. The run folder (run.json, log.jsonl, checkpoint-STEP) continues with
/// <c>--resume RUN</c> or <c>idrak resume RUN</c>.
/// </summary>
internal sealed class VoiceTrainCommand : Command
{
    /// <summary>run.json's format for a voice training run.</summary>
    public const string RunFormat = "idrak-voice-run/1";

    public override string Name => "voice train";

    public override string Summary => "Train a voice on a person's recordings (fine-tune a synthesizer, with their consent) and install it";

    public override string Usage => """
        NAME --from DATA --base MODEL --consent TEXT|--consent-file F [options]
        --resume RUN [options]

        Fine-tunes the synthesizer MODEL (a VITS checkpoint, its plug-in given with -P) on one person's speech, 30 to 60
        minutes for a natural voice, against discriminators (VITS's losses: mel, KL, duration, adversarial, feature
        matching), then installs the last checkpoint as the voice NAME (kind checkpoint) for idrak speak --voice NAME. The
        decoder trains on random segments, so memory follows the batch, not an utterance's length; a batch the device has
        no memory for is split in two and the step run again. A voice is a person's likeness: --consent (who agreed, to
        what) is required, stored with the voice and shown by voice show; the library cannot verify it.

        Arguments:
          NAME                    the voice's name (letters, digits, . - _)

        Options:
              --from DATA         transcripts (JSON lines of {"audio", "text"}) or a folder or zip of name.wav + name.txt
              --audio DIR|ZIP     where the transcripts' sound files are (default: beside the transcripts)
              --eval DATA         held-out speech, scored by its mel error every --eval-every steps and at the end
              --base MODEL        the synthesizer to start from: a folder or a hub id in the cache
              --consent TEXT      the speaker's agreement, in their words
              --consent-file F    the agreement read from a UTF-8 file
              --speaker WHO       who is speaking (default: NAME)
          -l, --language CODE     a language the voice speaks (repeatable; the base model's)
              --steps N           optimizer steps (default 2000)
              --lr X              the learning rate of the generator (default 2e-4)
              --discriminator-lr X
                                  the discriminators' learning rate (default: --lr)
              --batch-seconds S   audio seconds a step, padding included (default 40)
              --segment-frames N  the latent frames of each utterance's slice the decoder makes sound of (default 32)
              --mel-weight W      the mel loss's weight (default 45)
              --kl-weight W       the KL term's weight (default 1)
              --duration-weight W the duration loss's weight (default 1)
              --adversarial-weight W
                                  the adversarial loss's weight (default 1)
              --feature-weight W  the feature-matching loss's weight (default 2)
              --mels N            the mel loss's bands (default 80)
              --discriminator F   new discriminators' description (JSON; default the base's, else VITS's)
              --freeze PARTS      parts kept as the base has them, comma-separated (VITS: decoder keeps its waveform
                                  generator, decoder.upsampler and decoder.conv_pre its frame-to-sample layers; also
                                  text_encoder, flow, duration_predictor, posterior_encoder; default none)
              --schedule NAME     the rate's schedule: constant, cosine, linear, wsd (default constant)
              --warmup F          the share of the steps the rate warms up over (default 0)
              --optimizer NAME    adamw, adam, sgd (default adamw, betas 0.8 and 0.99)
              --clip N            the gradients' norm clipped to N (default 0: none, as VITS)
              --save-every N      a checkpoint every N steps (default 500; the last step always)
              --keep N            keep only the newest N checkpoints (default: all)
              --eval-every N      score --eval every N steps (default: at the end)
              --log-every N       a progress line every N steps (default 10)
              --feature-cache C   disk, memory or none (default disk: a second pass takes no spectra)
              --seed N            the draws' and the order's seed (default 0)
              --resume RUN        continue a voice run from its latest checkpoint (--steps N: N more)
          -o, --out RUN           the run folder (default CACHE/runs/DATE-voice-NAME)
              --no-install        train only; install later with --resume RUN --steps 0
              --profile           time steps instead of training: after a first pass over the recordings, 3 timed and 2
                                  with every GPU kernel timed (wall time, launches, kernel times; nothing installed)
              --overwrite         replace an installed voice of the same name
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice train ahmed --from rec/transcripts.jsonl --base facebook/mms-tts-ara -P Idrak.VitsSpeech.dll -l ar
            --consent "I, Ahmed, agree that this voice is trained for speech made with Idrak." -d cuda
          idrak voice train sara --from clips/ --base ./mms-tts-eng -P Idrak.VitsSpeech.dll --consent-file sara.txt -o runs/sara
          idrak voice train --resume runs/sara --steps 1000 -P Idrak.VitsSpeech.dll -d cuda
          idrak speak facebook/mms-tts-ara "مرحبا" --voice ahmed -P Idrak.VitsSpeech.dll -o ahmed.wav
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--from", "--audio", "--eval", "--base", "--consent", "--consent-file", "--speaker", "--language", "--steps", "--lr", "--discriminator-lr", "--batch-seconds",
        "--segment-frames", "--mel-weight", "--kl-weight", "--duration-weight", "--adversarial-weight", "--feature-weight", "--mels", "--discriminator", "--freeze", "--schedule",
        "--warmup", "--optimizer", "--clip", "--save-every", "--keep", "--eval-every", "--log-every", "--feature-cache", "--resume", "--out", "--voices",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--overwrite", "--no-install", "--profile"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-l"] = "--language", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Option("--resume") is { } resume)
        {
            string folder = Path.GetFullPath(resume), file = Path.Combine(folder, "run.json");
            if (!File.Exists(file))
            {
                throw new UsageException($"{resume} is not a voice training run (no run.json); give the -o folder idrak voice train wrote.");
            }

            return Resume(context, folder, JsonNode.Parse(File.ReadAllText(file))!.AsObject());
        }

        string name = context.Argument(0, "NAME (the voice's name)");
        if (!FolderVoiceStore.IsPlainName(name))
        {
            throw new UsageException($"'{name}' is not a voice name: letters, digits, '.', '-' and '_', starting with a letter or digit, at most 64 characters.");
        }

        string statement = Consent(context);
        string baseArgument = context.Option("--base") ?? throw new UsageException("Missing --base MODEL (the synthesizer to fine-tune; its plug-in given with -P).");
        string baseFolder = Directory.Exists(baseArgument) ? Path.GetFullPath(baseArgument) : ModelChoices.Resolve(context, baseArgument);
        var store = AudioFiles.VoiceStore(context);
        if (store.Find(name) is { } existing && !context.Flag("--overwrite") && !context.Flag("--no-install") && !context.Flag("--profile"))
        {
            throw new UsageException($"Voice '{existing.Name}' is installed already at {existing.Folder}; give --overwrite to replace it, or another NAME.");
        }

        var train = Source(context, context.Option("--from") ?? throw new UsageException("Give the recordings with --from DATA (transcripts, or a folder of name.wav + name.txt)."));
        var evaluation = context.Option("--eval") is { } eval ? Source(context, eval) : null;
        var options = Options(context, Defaults(), resuming: false);
        if (context.Flag("--profile"))
        {
            using var profiled = Load(context, baseFolder);
            string scratch = Path.Combine(Path.GetTempPath(), $"idrak-voice-profile-{Environment.ProcessId}");
            try
            {
                return Profile(context, profiled, train, options, scratch);
            }
            finally
            {
                if (Directory.Exists(scratch))
                {
                    Directory.Delete(scratch, recursive: true);
                }
            }
        }

        string run = Path.GetFullPath(context.Option("--out")
            ?? Path.Combine(RunLog.RunsFolder(context), string.Create(CultureInfo.InvariantCulture, $"{DateTime.Now:yyyyMMdd-HHmmss}-voice-{name}")));
        if (Directory.Exists(run) && SpeechTrainer.LatestCheckpoint(run) is not null)
        {
            throw new UsageException($"{run} holds a training run already; continue it with idrak voice train --resume {run}, or give another -o.");
        }

        var settings = new JsonObject
        {
            ["format"] = RunFormat,
            ["name"] = name,
            ["base"] = baseArgument,
            ["base_folder"] = baseFolder,
            ["consent"] = new JsonObject { ["speaker"] = context.Option("--speaker") ?? name, ["statement"] = statement, ["agreed"] = DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture) },
            ["languages"] = new JsonArray([.. context.Options("--language").Select(l => (JsonNode)l)]),
            ["options"] = options.ToJson(),
            ["train"] = train.ToJson(),
            ["evaluation"] = evaluation?.ToJson(),
            ["log_every"] = context.IntOption("--log-every", 10),
        };
        using var model = Load(context, baseFolder);
        Directory.CreateDirectory(run);
        File.WriteAllText(Path.Combine(run, "run.json"), settings.ToJsonString(CommandContext.JsonOutput) + "\n");
        context.Write(FormattableString.Invariant($"voice '{name}': fine-tuning {baseArgument} ({model.Generator.ParameterCount / 1e6:F2}M parameters) on {model.Device}, run {run}"));
        return Execute(context, run, settings, options, model.TrainingFiles,
            (progress, token, trace) => AdversarialSynthesisTrainer.Train(model, train, evaluation, options, run, null, progress, token, trace));
    }

    /// <summary>Continues a voice run (<c>idrak voice train --resume RUN</c>, <c>idrak resume RUN</c>) and installs its voice.</summary>
    public static int Resume(CommandContext context, string run, JsonObject settings)
    {
        if ((string?)settings["format"] != RunFormat)
        {
            throw new UsageException($"{run} is not a voice training run (run.json's format is {settings["format"]}).");
        }

        string checkpoint = SpeechTrainer.LatestCheckpoint(run)
            ?? throw new UsageException($"{run} has no checkpoint yet (the run stopped before its first was written); start it again with idrak voice train.");
        var state = SpeechTrainer.ReadCheckpoint(checkpoint);
        var options = Options(context, state.Options, resuming: true);
        if (context.Option("--steps") is not null)
        {
            int more = context.IntOption("--steps", 0);
            options = more >= 0 ? options with { Steps = state.Options.Steps + more } : throw new UsageException("--steps takes a number of steps to add, 0 or more.");
        }

        context.Write($"Resuming  {Path.GetFileName(run)} after step {state.Step} of {options.Steps}");
        IReadOnlyList<string> trainingFiles = [];
        return Execute(context, run, settings, options, trainingFiles, (progress, token, trace) =>
        {
            if (state.Step >= options.Steps)
            {
                return new SpeechTrainingResult(state.Step, state.Losses, state.Evaluations, checkpoint, TimeSpan.Zero, 0, 0, state.MicroBatchSeconds, 0, 0, []);
            }

            return AdversarialSynthesisTrainer.Resume(checkpoint, options, context.Device, progress, token, trace);
        });
    }

    // Fine-tuning's starting point: VITS's optimizer (AdamW 2e-4, betas in AdversarialSynthesisOptions), no clipping, a
    // constant rate; batches of 40 audio seconds, a checkpoint every 500 steps.
    private static SpeechTrainingOptions Defaults() => new()
    {
        Steps = 2000, LearningRate = 2e-4f, Schedule = "constant", WarmupFraction = 0, MaxGradientNorm = 0, BatchSeconds = 40, SaveEvery = 500,
        Adversarial = new AdversarialSynthesisOptions(),
    };

    // The options given on the command line over `start`.
    private static SpeechTrainingOptions Options(CommandContext context, SpeechTrainingOptions start, bool resuming)
    {
        var a = start.Adversarial ?? new AdversarialSynthesisOptions();
        JsonObject? description = a.Discriminator;
        if (context.Option("--discriminator") is { } file)
        {
            description = File.Exists(file) ? JsonNode.Parse(File.ReadAllText(file)) as JsonObject ?? throw new UsageException($"{file} is not a JSON object.")
                : throw new UsageException($"No discriminator description {file}.");
        }

        a = a with
        {
            SegmentFrames = context.IntOption("--segment-frames", a.SegmentFrames),
            MelWeight = Real(context, "--mel-weight", a.MelWeight),
            KlWeight = Real(context, "--kl-weight", a.KlWeight),
            DurationWeight = Real(context, "--duration-weight", a.DurationWeight),
            AdversarialWeight = Real(context, "--adversarial-weight", a.AdversarialWeight),
            FeatureMatchingWeight = Real(context, "--feature-weight", a.FeatureMatchingWeight),
            Mels = context.IntOption("--mels", a.Mels),
            DiscriminatorLearningRate = context.Option("--discriminator-lr") is null ? a.DiscriminatorLearningRate : Real(context, "--discriminator-lr", 0),
            Discriminator = description,
            Frozen = context.Option("--freeze") is { } frozen
                ? [.. frozen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)]
                : a.Frozen,
        };
        var options = start with
        {
            Steps = resuming ? start.Steps : context.IntOption("--steps", start.Steps),
            LearningRate = Real(context, "--lr", start.LearningRate),
            BatchSeconds = Real(context, "--batch-seconds", (float)start.BatchSeconds),
            Schedule = context.Option("--schedule") ?? start.Schedule,
            WarmupFraction = Real(context, "--warmup", start.WarmupFraction),
            Optimizer = context.Option("--optimizer") ?? start.Optimizer,
            MaxGradientNorm = Real(context, "--clip", start.MaxGradientNorm),
            SaveEvery = context.IntOption("--save-every", start.SaveEvery),
            KeepCheckpoints = context.IntOption("--keep", start.KeepCheckpoints),
            EvaluateEvery = context.IntOption("--eval-every", start.EvaluateEvery),
            FeatureCache = context.Option("--feature-cache") is { } cache ? cache.Equals("none", StringComparison.OrdinalIgnoreCase) ? null : cache : start.FeatureCache,
            Seed = context.Seed ?? start.Seed,
            Adversarial = a,
        };
        try
        {
            a.Check();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        return options;
    }

    private static string Consent(CommandContext context)
    {
        string statement = context.Option("--consent")
            ?? (context.Option("--consent-file") is { } file ? File.Exists(file) ? File.ReadAllText(file).Trim() : throw new UsageException($"Consent file not found: {file}") : null)
            ?? throw new UsageException("A voice is a person's likeness: give their agreement with --consent TEXT or --consent-file F (it is stored with the voice and shown by voice show).");
        return statement.Length > 0 ? statement : throw new UsageException("The consent is empty: give the speaker's agreement, in their words.");
    }

    // A source from --from or --eval: a folder or zip holds name.wav + name.txt, a file is transcripts.
    private static SpeechSource Source(CommandContext context, string path)
    {
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            throw new UsageException($"Not found: {path}");
        }

        bool folder = Directory.Exists(path) || path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase);
        return new SpeechSource(folder ? SpeechSources.AudioFolder : SpeechSources.Transcripts, [Path.GetFullPath(path)])
        {
            Audio = context.Option("--audio") is { } audio ? Path.GetFullPath(audio) : null,
        };
    }

    // The base loaded to be trained: a family whose model trains against discriminators.
    private static IAdversarialSpeechSynthesizer Load(CommandContext context, string folder)
    {
        ISpeechSynthesizer synthesizer;
        try
        {
            synthesizer = SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = context.Device, Training = true });
        }
        catch (FileNotFoundException e)
        {
            throw new UsageException($"{folder} is not a synthesizer's folder: {e.Message}");
        }

        if (synthesizer is IAdversarialSpeechSynthesizer adversarial)
        {
            return adversarial;
        }

        synthesizer.Dispose();
        throw new UsageException($"{folder} holds a {synthesizer.GetType().Name}, which its family does not train against discriminators (no IAdversarialSpeechSynthesizer); "
            + "voice train fine-tunes a VITS checkpoint (-P Idrak.VitsSpeech.dll).");
    }

    // --profile: steps until the second pass over the recordings (the first takes every clip's training rows), then 3 timed
    // by the clock and 2 with every GPU kernel timed (slower, so timed apart): what a step takes, in how many launches, and on what.
    private static int Profile(CommandContext context, IAdversarialSpeechSynthesizer model, SpeechSource train, SpeechTrainingOptions options, string scratch)
    {
        const int Timed = 3, Profiled = 2;
        var device = model.Device;
        context.Write($"profiling on {device}: the first pass over the recordings (their training rows taken), then {Timed} steps timed and {Profiled} with every GPU kernel timed…");
        long started = 0;
        int first = -1;
        double wallMs = 0, audioPerSecond = 0;
        IReadOnlyList<GpuProfileEntry> kernels = [];
        using var stop = new CancellationTokenSource();
        var progress = new SyncProgress<SpeechTrainingProgress>(p =>
        {
            if (p.Stage != "step")
            {
                return;
            }

            if (first < 0 && p.Epoch >= 2)
            {
                first = p.Step;
                device.Synchronize();
                started = Stopwatch.GetTimestamp();
            }
            else if (first >= 0 && p.Step == first + Timed)
            {
                device.Synchronize();
                wallMs = Stopwatch.GetElapsedTime(started).TotalMilliseconds / Timed;
                audioPerSecond = p.AudioSecondsPerSecond;
                GpuProfiler.Start(device);
            }
            else if (first >= 0 && p.Step == first + Timed + Profiled)
            {
                kernels = GpuProfiler.Stop(device);
                stop.Cancel();
            }
        });
        try
        {
            AdversarialSynthesisTrainer.Train(model, train, null, options with { Steps = 100_000, EvaluateEvery = 0, SaveEvery = 0 }, scratch, null, progress, stop.Token);
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
        }

        if (first < 0)
        {
            throw new UsageException("--profile: the recordings ended before a second pass; give more of them.");
        }

        if (kernels.Count == 0)
        {
            context.Write(FormattableString.Invariant($"per step: {wallMs:F0} ms ({audioPerSecond:F1}x real time); kernel times need a device that times them (cuda)"));
            return ExitCodes.Ok;
        }

        context.Write(GpuProfiler.Format(kernels, rows: 40).TrimEnd());
        double gpuMs = kernels.Sum(k => k.Milliseconds) / Profiled;
        long launches = kernels.Sum(k => k.Calls) / Profiled;
        // Each kernel is timed alone (drained, between two events), so a small one reads its start-up too: the sum bounds the
        // device's work from above, and a sum past the clock means the host's launches, not the device, set the step's time.
        double perLaunchUs = wallMs * 1000 / Math.Max(1, launches);
        context.Write(FormattableString.Invariant(
            $"per step: {wallMs:F0} ms by the clock in {launches:N0} launches ({perLaunchUs:F1} µs a launch; the run so far {audioPerSecond:F1}x real time)"));
        context.Write(gpuMs >= wallMs
            ? FormattableString.Invariant($"the kernels, each timed alone with its start-up, add up to {gpuMs:F0} ms, more than the step: launching them sets its time, so fewer launches make it faster")
            : FormattableString.Invariant($"the kernels, each timed alone with its start-up, add up to {gpuMs:F0} ms: the device works at most {gpuMs / Math.Max(1e-9, wallMs):P0} of the step, the host the rest"));
        return ExitCodes.Ok;
    }

    // Trains, then installs the run's last checkpoint as the voice (unless --no-install).
    private static int Execute(CommandContext context, string run, JsonObject settings, SpeechTrainingOptions options, IReadOnlyList<string> trainingFiles,
        Func<IProgress<SpeechTrainingProgress>, CancellationToken, Action<string>?, SpeechTrainingResult> train)
    {
        SpeechTrainingResult result;
        using (var log = new TelemetryAppender(Path.Combine(run, "log.jsonl")))
        using (var subscription = Telemetry.Subscribe(log))
        using (var interrupt = new Interrupt(context))
        using (var line = new ProgressLine(context, "training the voice", total: options.Steps, unit: ProgressUnit.Items))
        {
            try
            {
                result = train(new Reports(context, line, (int?)settings["log_every"] ?? 10), interrupt.Token, context.Verbose ? context.Detail : null);
            }
            catch (OperationCanceledException) when (interrupt.Requested)
            {
                line.Erase();
                string? last = SpeechTrainer.LatestCheckpoint(run);
                context.Write(interrupt.TimedOut ? "gave up after --timeout" : "stopped");
                context.Write(last is null ? "no checkpoint was written (no step finished)" : $"checkpoint {last}; idrak voice train --resume {run} continues");
                context.WriteJson(new JsonObject { ["run"] = run, ["stopped"] = true, ["checkpoint"] = last });
                return ExitCodes.Failed;
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or FileNotFoundException or NotSupportedException)
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

        context.Write(FormattableString.Invariant($"{result.Steps} steps in {result.Elapsed.TotalMinutes:F1} min on {result.Utterances:N0} utterances, generator loss {(result.Losses.Count > 0 ? result.Losses[^1] : float.NaN):F4}")
            + (result.Evaluations.Count > 0 ? FormattableString.Invariant($", held-out mel error {result.Evaluations[^1].Score:F4}") : ""));
        var output = new JsonObject
        {
            ["run"] = run, ["steps"] = result.Steps, ["checkpoint"] = result.Checkpoint, ["utterances"] = result.Utterances, ["dropped"] = result.Dropped,
            ["final_loss"] = result.Losses.Count > 0 && float.IsFinite(result.Losses[^1]) ? result.Losses[^1] : null,
            ["evaluations"] = new JsonArray([.. result.Evaluations.Select(e => (JsonNode)new JsonObject { ["step"] = e.Step, ["score"] = e.Score })]),
        };
        if (result.Checkpoint is null || context.Flag("--no-install"))
        {
            context.Write(result.Checkpoint is null ? "no checkpoint was written" : $"model {result.Checkpoint}; install it with idrak voice train --resume {run} --steps 0");
            context.WriteJson(output);
            return result.Checkpoint is null ? ExitCodes.Failed : ExitCodes.Ok;
        }

        string name = (string?)settings["name"] ?? throw new UsageException($"{run}/run.json names no voice.");
        var consent = settings["consent"]!;
        var store = AudioFiles.VoiceStore(context);
        var training = new JsonObject
        {
            ["steps"] = result.Steps, ["run"] = run, ["checkpoint"] = Path.GetFileName(result.Checkpoint), ["utterances"] = result.Utterances,
            ["final_loss"] = output["final_loss"]?.DeepClone(), ["options"] = options.ToJson(),
        };
        VoiceProfile voice;
        try
        {
            voice = store.AddCheckpoint(name, result.Checkpoint, new VoiceCheckpointOptions
            {
                Consent = new VoiceConsent((string)consent["speaker"]!, (string)consent["statement"]!, DateTimeOffset.Parse((string)consent["agreed"]!, CultureInfo.InvariantCulture)),
                Languages = [.. settings["languages"]!.AsArray().Select(l => (string)l!)],
                Base = (string?)settings["base"],
                Training = training,
                Exclude = [.. AdversarialSynthesisTrainer.StateFiles, .. trainingFiles.Count > 0 ? trainingFiles : ["discriminator.safetensors", "discriminator.json"]],
                Overwrite = true,
            });
        }
        catch (Exception e) when (e is ArgumentException or IOException)
        {
            context.Error(VoiceText.Options(e.Message));
            return ExitCodes.Failed;
        }

        context.Write($"installed '{voice.Name}' in {store.Location}");
        foreach (string text in VoiceText.Lines(voice, VoiceText.WarnAgreement))
        {
            context.Write(text);
        }

        context.Write($"next: idrak speak \"{settings["base"]}\" \"...\" --voice {voice.Name} -P <the model's plug-in> -o {voice.Name}.wav");
        output["voice"] = VoiceText.Json(voice, VoiceText.WarnAgreement);
        context.WriteJson(output);
        return ExitCodes.Ok;
    }

    private static float Real(CommandContext context, string option, float otherwise) => context.Option(option) is not { } text ? otherwise
        : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : throw new UsageException($"{option} is a number, not '{text}'.");

    // The live line, and every N steps a line of the terms, the pace and the time left.
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
                var t = p.Terms;
                string terms = t is null ? "" : FormattableString.Invariant(
                    $"  mel {t.GetValueOrDefault("mel"):F3}  kl {t.GetValueOrDefault("kl"):F3}  dur {t.GetValueOrDefault("duration"):F3}  adv {t.GetValueOrDefault("adversarial"):F3}  fm {t.GetValueOrDefault("feature_matching"):F3}  disc {t.GetValueOrDefault("discriminator"):F3}");
                context.Write(FormattableString.Invariant($"step {p.Step}/{p.Steps} (epoch {p.Epoch})  loss {p.Loss:F3}") + terms
                    + FormattableString.Invariant($"  {p.AudioSecondsPerSecond:F1}x real time")
                    + (p.PeakDeviceBytes > 0 ? FormattableString.Invariant($"  {p.PeakDeviceBytes / 1048576.0:F0} MiB") : "")
                    + FormattableString.Invariant($"  {p.Remaining:hh\\:mm\\:ss} left")
                    + (p.Evaluation is { } e ? FormattableString.Invariant($"  held-out mel {e:F4}") : ""));
            }
        }
    }
}
