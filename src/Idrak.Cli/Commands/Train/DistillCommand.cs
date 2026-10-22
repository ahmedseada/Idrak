// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak distill --teacher A --student B --data FILE</c>: knowledge distillation (plans/plug-in.md, "Teacher pattern").
/// The student's adapter trains on the teacher's token distributions (computed on the fly, or read from a file written
/// with <c>--precompute</c>), or, with <c>--generate</c>, on answers the teacher writes. The training itself is
/// idrak tune's (the same data reading, options and adapter folder).
/// </summary>
internal sealed class DistillCommand : Command
{
    // distill's own names for the tune options it passes on, where they differ (--alpha is distillation's mixing weight).
    private static readonly Dictionary<string, string> Renamed = new() { ["--lora-alpha"] = "--alpha" };

    // The tune options distill passes on as they are: a session's (TuneTool.SessionValueOptions, --alpha as --lora-alpha),
    // then the data's, the epochs and the evaluation.
    private static readonly string[] TuneValues =
    [
        .. TuneTool.SessionValueOptions.Select(o => Renamed.FirstOrDefault(r => r.Value == o).Key ?? o), "--epochs", "--eval", "--eval-fraction", "--eval-every", "--save-every",
        "--system", "--max-rows", "--data-format", "--images", "--image-transform", "--vision", "--parts", "--feature-cache",
    ];

    private static readonly string[] TuneFlags = [.. TuneTool.SessionFlags, "--no-dedup"];

    public override string Name => "distill";

    public override string Summary => "Distil a teacher model into a student: its token probabilities (on the fly or precomputed) or its answers";

    public override string Usage => """
        --teacher MODEL|FILE --student MODEL --data FILE... -o DIR [options]

        The student's adapter (LoRA, as idrak tune trains it) learns from the teacher. Four ways:
          logits      (default) the teacher's next-token distributions at every trained token, computed as the student
                      trains (both models loaded; the teacher in inference mode, any weight format, any device). The loss is
                      alpha · T² · KL(teacher ‖ student) at temperature T plus (1 − alpha) · the cross-entropy on the data.
                      Teacher and student must share a vocabulary (the same tokenizer); a mismatch is refused.
          precompute  --precompute FILE writes the teacher's top-k logits for the data (k = --top-k, default 16) and
                      stops; train later with --teacher FILE (the teacher is not loaded then). The loss uses the stored
                      top k, renormalized: an approximation that drops the tail of each distribution.
          generate    --generate: the teacher answers the data's prompts (greedy; --reasoning keeps its reasoning) and the
                      student fine-tunes on its answers, written to DIR/teacher-data.jsonl. Works across vocabularies.
          on-policy   --on-policy: the student answers the data's prompts itself (its live adapters, sampled), and learns
                      the teacher's next-token distributions on exactly the tokens it sampled, so it learns where its own
                      answers go wrong (Agarwal et al. 2024, the library's OnPolicyDistiller). A share --mix of each step's
                      prompts trains on the data's own answers instead. Only the assistant's tokens train. A live teacher
                      model of the same vocabulary (stored logits cannot cover new samples).

        Options:
              --teacher MODEL|FILE  the model that teaches (a Hugging Face id, a folder, a .gguf file or an alias), or a
                                    teacher logits file written by --precompute
              --student MODEL       the model that learns (-b, --base MODEL is the same)
              --data FILE           the conversations, texts or prompts (repeatable): files, folders, hf: specs, as
                                    idrak tune reads them
          -o, --out DIR             where the student's adapter goes
              --temperature T       softens both distributions (default 1)
              --alpha A             weight of the distillation term, 0 to 1 (default 0.5; 1: the teacher alone)
              --top-k K             tokens kept per position (default: all on the fly, 16 with --precompute)
              --precompute FILE     write the teacher's top-k logits for the data to FILE and stop
              --generate            train on answers the teacher writes instead of its probabilities
              --reasoning           with --generate: the teacher reasons and its reasoning is kept in the answers
              --keep-cut-off        with --generate: keep answers cut off at --max-new (default: left out)
              --max-new N           with --generate: the longest answer in tokens (default 512); with --on-policy: the
                                    longest sampled answer (default 256)
              --batch-size N        with --generate: prompts answered together (default 8); with --on-policy: the
                                    prompts of each step (default 8)
              --on-policy           the student samples its own answers and learns the teacher's distributions on them
                                    (in one process: on a cluster, idrak reinforce with a policy loss and --teacher)
              --divergence NAME     with --on-policy: forward, reverse (default) or jsd, or a registered one
                                    (TeacherDivergences; IDRAK_RL_DIVERGENCE)
              --beta B              with --on-policy and jsd: the Jensen-Shannon weight, in (0, 1) (default 0.5)
              --mix L               with --on-policy: λ, the share of a step's prompts trained on the data's answers, 0 to 1
                                    (default 0; IDRAK_RL_DISTILL_MIX)
              --steps N             with --on-policy: optimizer steps (default: --epochs passes over the prompts)
              --teacher-weights F   the teacher's weights: int8, int4 or bf16 (default: as stored)
              --teacher-device D    the teacher's device (default: the student's), e.g. cpu, cuda:1
          -w, --weights F           the student's base weights: int8, int4 (QLoRA) or bf16
          and these options of idrak tune train: --rank, --lora-alpha (tune's --alpha), --lr, --epochs, --max-length,
          --batch-tokens, --accumulate, --targets, --optimizer, --weight-decay, --momentum, --schedule, --warmup, --min-lr,
          --decay, --adapter-type, --adapter, --context, --matmul, --gpu-memory, --eval, --eval-fraction, --eval-every,
          --save-every, --system, --max-rows, --no-packing, --checkpointing, --no-checkpointing, --recompute,
          --bf16-activations, --no-graphs, --fp8, --cpu-optimizer, --no-dedup, --offload, and for image
          records --data-format, --images, --image-transform, --vision, --parts and --feature-cache (a vision-language
          teacher reads the student's images as they are prepared; its family's plug-in with -P). --json prints one JSON
          document at the end with the output lines.
              --cluster FILE        train as one rank of a real cluster of the topology file (idrak cluster init writes
                                    one); on a node of several workers IDRAK_RANK names this process's (--rank stays the
                                    adapter's rank)
              --node NAME           this machine's node of the file (the node named after this host)
              --coordinator H:P     where rank 0 listens (the file's first node's address)
              --world-size N        the ranks of a cluster without a file, with --coordinator and IDRAK_RANK

        Examples:
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data chats.jsonl -o adapters/d
          idrak distill --teacher Qwen/Qwen3-8B --teacher-weights int4 --student Qwen/Qwen3-0.6B --data c.jsonl -o d
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data c.jsonl --temperature 2 --alpha 0.8 -o d
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data chats.jsonl --precompute t.topk
          idrak distill --teacher t.topk --student Qwen/Qwen3-0.6B --data chats.jsonl -o adapters/d
          idrak distill --teacher big-model --student small-model --data prompts.jsonl --generate -o adapters/s
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data chats.jsonl --on-policy --mix 0.25 -o adapters/op
          idrak distill --teacher t.topk --student Qwen/Qwen3-0.6B --data chats.jsonl -o adapters/d --cluster lan.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--teacher", "--student", "--base", "--data", "--out", "--temperature", "--alpha", "--top-k", "--precompute", "--max-new", "--batch-size",
        "--teacher-weights", "--teacher-device", "--weights", "--divergence", "--beta", "--mix", "--steps", .. TuneValues, .. ClusterJoin.Options,
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--generate", "--reasoning", "--keep-cut-off", "--on-policy", .. TuneFlags];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-b"] = "--base", ["-o"] = "--out", ["-w"] = "--weights",
    };

    public override int Run(CommandContext context)
    {
        using var cluster = ClusterJoin.Join(context, ownsRank: true);    // --cluster FILE: one rank of a real cluster; --rank is the adapter's
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"distill takes no arguments ('{context.Positional[0]}'): give --teacher, --student and --data.");
        }

        string teacher = context.Option("--teacher") ?? throw new UsageException("Give --teacher MODEL (or a teacher logits file).");
        string student = context.Option("--student") ?? context.Option("--base") ?? throw new UsageException("Give --student MODEL (or -b, --base MODEL).");
        var data = context.Options("--data");
        if (data.Count == 0)
        {
            throw new UsageException("Give --data FILE (the conversations, texts or prompts).");
        }

        bool generate = context.Flag("--generate");
        string? precompute = context.Option("--precompute");
        string? output = context.Option("--out");
        float temperature = Number(context, "--temperature", 1f);
        float alpha = Number(context, "--alpha", 0.5f);
        int? topK = context.Option("--top-k") is null ? null : context.IntOption("--top-k", 0);
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new UsageException("--temperature is a positive number.");
        }

        if (alpha is < 0f or > 1f)
        {
            throw new UsageException("--alpha is between 0 and 1.");
        }

        if (topK is < 0)
        {
            throw new UsageException("--top-k is a whole number, 0 or more (0: the full distribution).");
        }

        bool onPolicy = context.Flag("--on-policy");
        if ((generate ? 1 : 0) + (precompute is not null ? 1 : 0) + (onPolicy ? 1 : 0) > 1)
        {
            throw new UsageException(onPolicy ? "--on-policy, --generate and --precompute are different ways: give one."
                : "--generate and --precompute are two different ways: give one.");
        }

        if (onPolicy && ClusterSession.Current is not null)
        {
            // Plan 27, R3c: OnPolicyDistiller samples every step's prompts on every rank and steps on them without an exchange,
            // so the ranks would agree only where generation is bit-identical across them, and the sampling is not spread.
            throw new UsageException("--on-policy does not run on a cluster: on-policy distillation samples every prompt on every rank. Run it in one process, or "
                                     + "idrak reinforce with a policy loss and --teacher on the cluster (the teacher's term beside the reward, its episodes spread over the ranks).");
        }

        var onPolicyOptions = onPolicy ? OnPolicyOptions(context) : null;

        if (precompute is null && output is null)
        {
            throw new UsageException("Give -o, --out DIR for the student's adapter.");
        }

        bool stored = TeacherLogitsFile.IsTeacherLogits(teacher);
        if (stored && (generate || precompute is not null))
        {
            throw new UsageException($"{teacher} holds stored logits: it cannot write answers or logits; give the teacher model for --generate and --precompute.");
        }

        string mode = onPolicy ? "on-policy" : generate ? "generate" : precompute is not null ? "precompute" : stored ? "stored logits" : "logits";
        Device? teacherDevice = context.Option("--teacher-device") is { } deviceName ? ParseDevice(deviceName) : null;
        var teacherChoice = new ModelChoices.ModelChoice(teacher, context.Option("--teacher-weights"), null, null, null);
        if (teacherChoice.Weights?.ToLowerInvariant() is not (null or "int8" or "int4" or "bf16" or "bfloat16"))
        {
            throw new UsageException($"--teacher-weights {teacherChoice.Weights}: use int8, int4 or bf16.");
        }

        var captured = context.Json ? new StringWriter() : null;
        var console = ToolHost.Console(context, captured ?? (context.Quiet ? TextWriter.Null : context.Output));
        var trainingData = data.ToList();
        int generated = 0;
        if (generate)
        {
            // The teacher's answers first, then ordinary fine-tuning on them.
            Directory.CreateDirectory(output!);
            string rows = Path.Combine(output!, "teacher-data.jsonl");
            generated = Generate(context, console, teacherChoice, teacherDevice ?? context.Device, data, rows);
            if (generated == 0)
            {
                context.Error("idrak distill: the teacher wrote no answer (no row with a prompt, or every answer cut off: raise --max-new).");
                return ExitCodes.Failed;
            }

            trainingData = [rows];
        }

        var choice = ModelChoices.Choose(context, student);
        var args = TuneArguments.Train(context, choice, trainingData, output ?? Path.GetDirectoryName(Path.GetFullPath(precompute!))!, "the student");
        if (generate)
        {
            args.AddRange(["--kind", "chat"]);
        }

        TuneArguments.Pass(context, args, TuneValues, TuneFlags, Renamed);
        var tool = new TuneTool(console);
        try
        {
            tool.Parse(args);
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            throw new UsageException(e.Message);
        }

        // The teacher joins once the student and the data are ready: its logits written (precompute), or its
        // distributions as the loss's soft targets.
        PretrainedModel? teacherModel = null;
        DistillationTeacher? source = null;
        int precomputed = 0;
        IReadOnlyList<float> onPolicyLosses = [];
        if (onPolicy)
        {
            // R3's on-policy distillation on the data's prompts, in a session of the model tune loaded.
            tool.Session = (model, tuning) =>
            {
                var chats = tool.ReadChats();
                int steps = onPolicyOptions!.Steps > 0 ? onPolicyOptions.Steps
                    : Math.Max(1, tuning.Epochs) * Math.Max(1, (chats.Count + onPolicyOptions.PromptsPerStep - 1) / onPolicyOptions.PromptsPerStep);
                (teacherModel, source) = LoadTeacher(context, console, teacherChoice, teacherDevice ?? tool.Device, topK ?? 0, tuning.BatchTokens);
                int result;
                (result, onPolicyLosses) = TrainOnPolicy(context, console, model, tuning, source!, chats,
                    onPolicyOptions! with { Steps = steps, MaxLength = tuning.MaxLength, Temperature = temperature }, output!);
                return result;
            };
        }
        else if (!generate)
        {
            tool.BeforeTraining = (model, train, evaluation, tuning) =>
            {
                if (!stored)
                {
                    teacherModel = LoadTeacherModel(context, console, teacherChoice, teacherDevice ?? tool.Device);
                }

                if (precompute is not null)
                {
                    DistillationTeacher.CheckVocabulary(teacherModel!.Tokenizer ?? throw new InvalidOperationException("The teacher has no tokenizer."),
                        model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer."));
                    var sequences = evaluation is null ? train : [.. train, .. evaluation];
                    int k = topK is > 0 ? topK.Value : 16;
                    var clock = Stopwatch.StartNew();
                    precomputed = TeacherLogitsWriter.Write(precompute, teacherModel, sequences, k,
                        done => console.Progress("teacher logits", done, sequences.Count, clock.Elapsed, unit: "sequences"), tuning.BatchTokens,
                        tuning.Vision?.Images ?? TuningImages.None);
                    console.Finish();
                    console.Out.WriteLine($"teacher logits for {precomputed:N0} sequences ({sequences.Sum(s => (long)s.TrainedTokens):N0} trained tokens, top {k}) written to "
                                          + $"{Path.GetFullPath(precompute)} ({Downloader.Size(new FileInfo(precompute).Length)})");
                    console.Out.WriteLine($"next: idrak distill --teacher {precompute} --student {student} --data {string.Join(" --data ", data)} -o DIR");
                    return null;
                }

                // A vision-language teacher reads the student's images as the student's run prepares them.
                source = stored ? DistillationTeachers.FromFile(teacher)
                    : DistillationTeachers.FromModel(teacherModel!, topK ?? 0, tuning.BatchTokens, tuning.Vision?.Images ?? TuningImages.None);
                source.Check(model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer: its vocabulary cannot be compared with the teacher's."));
                console.Out.WriteLine($"distilling: temperature {temperature.ToString(CultureInfo.InvariantCulture)}, alpha {alpha.ToString(CultureInfo.InvariantCulture)}, "
                                      + (source.TopK > 0 ? $"the teacher's top {source.TopK} tokens per position (renormalized)" : "the teacher's full distributions"));
                return tuning with { Teacher = source, Loss = FineTuningLosses.Distillation(temperature, alpha) };
            };
        }

        int code;
        try
        {
            if (tool.Problem() is { } problem)
            {
                throw new UsageException(problem);
            }

            context.Detail($"idrak-tune {string.Join(' ', args)}");
            code = tool.Execute();
        }
        finally
        {
            source?.Dispose();
            teacherModel?.Dispose();
        }

        if (captured is not null)
        {
            var json = new JsonObject
            {
                ["mode"] = mode,
                ["teacher"] = teacher,
                ["student"] = choice.Model,
                ["data"] = new JsonArray([.. data.Select(d => (JsonNode)d)]),
                ["out"] = precompute is not null ? Path.GetFullPath(precompute) : output,
                ["device"] = tool.Device.ToString(),
                ["exitCode"] = code,
                ["output"] = new JsonArray([.. captured.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => (JsonNode)l)]),
            };
            if (onPolicy)
            {
                json["temperature"] = temperature;
                json["divergence"] = onPolicyOptions!.Divergence;
                json["mix"] = onPolicyOptions.Mix;
                json["losses"] = new JsonArray([.. onPolicyLosses.Select(l => (JsonNode)l)]);
            }
            else if (!generate && precompute is null)
            {
                json["temperature"] = temperature;
                json["alpha"] = alpha;
                json["topK"] = source?.TopK ?? topK ?? 0;
            }

            if (precompute is not null)
            {
                json["sequences"] = precomputed;
            }

            if (generate)
            {
                json["answers"] = generated;
            }

            context.WriteJson(json);
        }

        return code;
    }

    // The teacher's answers to the data's prompts, written as chat rows to `path`; returns how many.
    private static int Generate(CommandContext context, ToolConsole console, ModelChoices.ModelChoice choice, Device device, IReadOnlyList<string> data, string path)
    {
        int maxNew = context.IntOption("--max-new", 512), batchSize = context.IntOption("--batch-size", 8);
        if (maxNew < 1 || batchSize < 1)
        {
            throw new UsageException("--max-new and --batch-size are positive whole numbers.");
        }

        var downloader = console.CreateDownloader();
        IEnumerable<JsonObject> Rows()
        {
            foreach (string spec in data)
            {
                var source = DatasetSpec.Parse(spec);
                var mapping = source.Mapping;
                foreach (var row in source.Open(downloader, CodeFolders.Skipped))
                {
                    if (mapping is null)
                    {
                        yield return row;
                    }
                    else if (ChatRows.Normalize(row, RowKind.Chat, mapping) is { } mapped)
                    {
                        yield return mapped;
                    }
                }
            }
        }

        using var teacher = ModelChoices.Load(context, choice, device);
        var chat = ModelChoices.CreateChat(teacher, choice);
        var options = new TeacherDataOptions
        {
            Reasoning = context.Flag("--reasoning"),
            KeepCutOff = context.Flag("--keep-cut-off"),
            MaxNewTokens = maxNew,
            BatchSize = batchSize,
            Context = Math.Min(ModelChoices.DefaultContext, teacher.MaxPositions),
            System = context.Option("--system"),
            Seed = context.Seed,
        };
        console.Out.WriteLine($"teacher {choice.Model} on {device} answers the prompts of {string.Join(", ", data)}{(options.Reasoning ? ", with its reasoning" : "")}…");
        var clock = Stopwatch.StartNew();
        int count = 0;
        using (var writer = new StreamWriter(path))
        {
            foreach (var row in TeacherData.Generate(chat, Rows(), options, context.TimeoutToken))
            {
                writer.WriteLine(row.ToJsonString());
                count++;
                console.Progress("teacher answers", count, 0, clock.Elapsed, unit: "answers");
            }
        }

        console.Finish();
        console.Out.WriteLine($"teacher wrote {count:N0} answers to {Path.GetFullPath(path)} in {clock.Elapsed.TotalSeconds:F1} s");
        return count;
    }

    // --on-policy's settings from the command line (each library default reads its variable): the divergence, β, λ, the
    // prompts of a step, the longest sampled answer, the rollouts' temperature (IDRAK_RL_TEMPERATURE) and the seed; the
    // steps (0 here: from the data), the lengths and the softening temperature are set once the data and model are read.
    internal static OnPolicyDistillationOptions OnPolicyOptions(CommandContext context)
    {
        var o = new OnPolicyDistillationOptions();
        string divergence = context.Option("--divergence") ?? o.Divergence;
        o = o with
        {
            Divergence = TeacherDivergences.Find(divergence)?.Name ?? throw new UsageException(
                $"{(context.Option("--divergence") is null ? LanguageModelPolicyOptions.DivergenceVariable + "=" : "--divergence ")}{divergence}: use {string.Join(", ", TeacherDivergences.Names)}."),
        };
        if (context.Option("--beta") is not null)
        {
            float beta = Number(context, "--beta", o.Beta);
            o = beta is > 0f and < 1f ? o with { Beta = beta } : throw new UsageException("--beta is between 0 and 1 (both left out).");
        }

        if (context.Option("--mix") is not null)
        {
            float mix = Number(context, "--mix", o.Mix);
            o = mix is >= 0f and <= 1f ? o with { Mix = mix } : throw new UsageException("--mix is between 0 and 1.");
        }

        int steps = context.Option("--steps") is null ? 0 : context.IntOption("--steps", 0);
        int prompts = context.IntOption("--batch-size", o.PromptsPerStep), maxNew = context.IntOption("--max-new", o.MaxNewTokens);
        if (steps < 0 || context.Option("--steps") is not null && steps == 0 || prompts < 1 || maxNew < 1)
        {
            throw new UsageException("--steps, --batch-size and --max-new are positive whole numbers.");
        }

        float sampling;
        try
        {
            sampling = Idrak.Reinforcement.ReinforcementOptions.FromEnvironment().Temperature;
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        return o with { Steps = steps, PromptsPerStep = prompts, MaxNewTokens = maxNew, SamplingTemperature = sampling, Seed = context.Seed ?? 0 };
    }

    // The teacher model, loaded on its device (as stored, or in the weights its choice names).
    internal static PretrainedModel LoadTeacherModel(CommandContext context, ToolConsole console, ModelChoices.ModelChoice choice, Device device)
    {
        var model = ModelChoices.Load(context, choice, device);
        console.Out.WriteLine($"teacher {choice.Model} on {model.Device}: {model.Spec.ParameterCount / 1e6:F0}M parameters, vocabulary {model.Spec.Vocabulary}");
        return model;
    }

    // The teacher of a distillation: a model and its distributions on the fly (its top k, 0: all), or a stored top-k logits
    // file (no model loaded; refused where new samples are scored, by the library).
    internal static (PretrainedModel? Model, DistillationTeacher Teacher) LoadTeacher(CommandContext context, ToolConsole console, ModelChoices.ModelChoice choice,
        Device device, int topK, int batchTokens)
    {
        if (TeacherLogitsFile.IsTeacherLogits(choice.Model))
        {
            return (null, DistillationTeachers.FromFile(choice.Model));
        }

        var model = LoadTeacherModel(context, console, choice, device);
        try
        {
            return (model, DistillationTeachers.FromModel(model, topK, batchTokens));
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    // On-policy distillation alone (R3's OnPolicyDistiller): a session of the student's model with the teacher, its steps on
    // a progress line, the adapter written to `output`. Ctrl+C stops after the current step and keeps what was trained.
    internal static (int Code, IReadOnlyList<float> Losses) TrainOnPolicy(CommandContext context, ToolConsole console, PretrainedModel model, FineTuningOptions tuning,
        DistillationTeacher teacher, IReadOnlyList<ChatTranscript> data, OnPolicyDistillationOptions options, string output)
    {
        if (data.Count == 0)
        {
            console.Error.WriteLine("error: no conversation in the data to take prompts from.");
            return (ExitCodes.Failed, []);
        }

        console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"on-policy distillation: {options.Steps} steps of {options.PromptsPerStep} prompts from {data.Count:N0} conversations, divergence {options.Divergence}, "
            + $"mix {options.Mix}, temperature {options.Temperature}, answers of at most {options.MaxNewTokens} tokens"));
        using var session = new FineTuningSession(model, tuning with { Teacher = teacher }, options.Steps);
        using var stop = new StopKey(context, console.Log, "stopping after the current step and saving the adapter (Ctrl+C again to quit now)…");
        using var progress = new ProgressLine(context, "on-policy distillation", options.Steps);
        OnPolicyDistillationProgress? last = null;
        progress.Detail = () => last is null ? "" : string.Create(CultureInfo.InvariantCulture,
            $"divergence {last.Loss:F4}, lr {last.LearningRate:G3}, {last.Sampled} sampled, {last.FromData} from the data");
        var losses = new List<float>();
        bool stopped = false;
        try
        {
            OnPolicyDistiller.Train(session, data, options, new Reporter<OnPolicyDistillationProgress>(p =>
            {
                last = p;
                losses.Add(p.Loss);
                progress.Report(p.Step);
            }), stop.Token);
            progress.Finish();
        }
        catch (OperationCanceledException) when (stop.Token.IsCancellationRequested)
        {
            stopped = true;
            progress.Clear();
        }

        session.SaveOnce(output);
        console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{(stopped ? "stopped after" : "distilled in")} {losses.Count} steps{(losses.Count > 0 ? FormattableString.Invariant($", divergence {losses[0]:F4} → {losses[^1]:F4}") : "")}; "
            + $"adapter written to {Path.GetFullPath(output)}"));
        return (ExitCodes.Ok, losses);
    }

    // Reports on the calling thread.
    private sealed class Reporter<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    private static float Number(CommandContext context, string option, float fallback)
    {
        if (context.Option(option) is not { } text)
        {
            return fallback;
        }

        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) ? value
            : throw new UsageException($"{option} needs a number, not '{text}'.");
    }

    private static Device ParseDevice(string name)
    {
        try
        {
            return Device.Parse(name);
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException)
        {
            throw new UsageException($"--teacher-device {name}: {e.Message}");
        }
    }
}
