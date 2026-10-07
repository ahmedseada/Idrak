// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.LanguageModels;

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

    // The tune options distill passes on as they are.
    private static readonly string[] TuneValues =
    [
        "--rank", "--lora-alpha", "--lr", "--epochs", "--max-length", "--batch-tokens", "--accumulate", "--targets", "--optimizer", "--weight-decay",
        "--schedule", "--warmup", "--min-lr", "--eval", "--eval-fraction", "--eval-every", "--save-every", "--system", "--max-rows", "--adapter-type", "--context",
    ];

    private static readonly string[] TuneFlags = ["--no-packing", "--checkpointing", "--no-checkpointing", "--recompute", "--bf16-activations", "--no-dedup", "--offload"];

    public override string Name => "distill";

    public override string Summary => "Distil a teacher model into a student: its token probabilities (on the fly or precomputed) or its answers";

    public override string Usage => """
        --teacher MODEL|FILE --student MODEL --data FILE... -o DIR [options]

        The student's adapter (LoRA, as idrak tune trains it) learns from the teacher. Three ways:
          logits      (default) the teacher's next-token distributions at every trained token, computed as the student
                      trains (both models loaded; the teacher in inference mode, any weight format, any device). The loss is
                      alpha · T² · KL(teacher ‖ student) at temperature T plus (1 − alpha) · the cross-entropy on the data.
                      Teacher and student must share a vocabulary (the same tokenizer); a mismatch is refused.
          precompute  --precompute FILE writes the teacher's top-k logits for the data (k = --top-k, default 16) and
                      stops; train later with --teacher FILE (the teacher is not loaded then). The loss uses the stored
                      top k, renormalized: an approximation that drops the tail of each distribution.
          generate    --generate: the teacher answers the data's prompts (greedy; --reasoning keeps its reasoning) and the
                      student fine-tunes on its answers, written to DIR/teacher-data.jsonl. Works across vocabularies.

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
              --max-new N           with --generate: the longest answer in tokens (default 512)
              --batch-size N        with --generate: prompts answered together (default 8)
              --teacher-weights F   the teacher's weights: int8, int4 or bf16 (default: as stored)
              --teacher-device D    the teacher's device (default: the student's), e.g. cpu, cuda:1
          -w, --weights F           the student's base weights: int8, int4 (QLoRA) or bf16
          and these options of idrak tune train: --rank, --lora-alpha (tune's --alpha), --lr, --epochs, --max-length,
          --batch-tokens, --accumulate, --targets, --optimizer, --weight-decay, --schedule, --warmup, --min-lr, --eval,
          --eval-fraction, --eval-every, --save-every, --system, --max-rows, --adapter-type, --context, --no-packing,
          --checkpointing, --no-checkpointing, --recompute, --bf16-activations, --no-dedup, --offload. --json prints one
          JSON document at the end with the output lines.

        Examples:
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data chats.jsonl -o adapters/d
          idrak distill --teacher Qwen/Qwen3-8B --teacher-weights int4 --student Qwen/Qwen3-0.6B --data c.jsonl -o d
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data c.jsonl --temperature 2 --alpha 0.8 -o d
          idrak distill --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B --data chats.jsonl --precompute t.topk
          idrak distill --teacher t.topk --student Qwen/Qwen3-0.6B --data chats.jsonl -o adapters/d
          idrak distill --teacher big-model --student small-model --data prompts.jsonl --generate -o adapters/s
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--teacher", "--student", "--base", "--data", "--out", "--temperature", "--alpha", "--top-k", "--precompute", "--max-new", "--batch-size",
        "--teacher-weights", "--teacher-device", "--weights", .. TuneValues,
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--generate", "--reasoning", "--keep-cut-off", .. TuneFlags];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-b"] = "--base", ["-o"] = "--out", ["-w"] = "--weights",
    };

    public override int Run(CommandContext context)
    {
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

        if (generate && precompute is not null)
        {
            throw new UsageException("--generate and --precompute are two different ways: give one.");
        }

        if (precompute is null && output is null)
        {
            throw new UsageException("Give -o, --out DIR for the student's adapter.");
        }

        bool stored = TeacherLogitsFile.IsTeacherLogits(teacher);
        if (stored && (generate || precompute is not null))
        {
            throw new UsageException($"{teacher} holds stored logits: it cannot write answers or logits; give the teacher model for --generate and --precompute.");
        }

        string mode = generate ? "generate" : precompute is not null ? "precompute" : stored ? "stored logits" : "logits";
        Device? teacherDevice = context.Option("--teacher-device") is { } deviceName ? ParseDevice(deviceName) : null;
        var teacherChoice = new Models.ModelChoice(teacher, context.Option("--teacher-weights"), null, null, null);
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

        var choice = Models.Choose(context, student);
        List<string> args = ["train", choice.Model, .. trainingData, "--out", output ?? Path.GetDirectoryName(Path.GetFullPath(precompute!))!];
        if ((context.Option("--weights") ?? choice.Weights) is { } weights)
        {
            args.Add(weights.ToLowerInvariant() switch
            {
                "int8" => "--int8",
                "int4" => "--int4",
                "bf16" or "bfloat16" => "--bf16",
                _ => throw new UsageException($"--weights {weights}: the student loads as int8, int4 or bf16 (leave it out for float32)."),
            });
        }

        if ((context.Option("--device") ?? context.Config.Get("device")) is { } device)
        {
            args.AddRange(["--device", device]);
        }

        if (context.Option("--seed") is { } seed)
        {
            args.AddRange(["--seed", seed]);
        }

        if (generate)
        {
            args.AddRange(["--kind", "chat"]);
        }

        foreach (string option in TuneValues)
        {
            foreach (string value in context.Options(option))
            {
                args.AddRange([Renamed.GetValueOrDefault(option, option), value]);
            }
        }

        args.AddRange(TuneFlags.Where(context.Flag));
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
        if (!generate)
        {
            tool.BeforeTraining = (model, train, evaluation, tuning) =>
            {
                if (!stored)
                {
                    teacherModel = Models.Load(context, teacherChoice, teacherDevice ?? tool.Device);
                    console.Out.WriteLine($"teacher {teacher} on {teacherModel.Device}: {teacherModel.Spec.ParameterCount / 1e6:F0}M parameters, vocabulary {teacherModel.Spec.Vocabulary}");
                }

                if (precompute is not null)
                {
                    DistillationTeacher.CheckVocabulary(teacherModel!.Tokenizer ?? throw new InvalidOperationException("The teacher has no tokenizer."),
                        model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer."));
                    var sequences = evaluation is null ? train : [.. train, .. evaluation];
                    int k = topK is > 0 ? topK.Value : 16;
                    var clock = Stopwatch.StartNew();
                    precomputed = TeacherLogitsWriter.Write(precompute, teacherModel, sequences, k,
                        done => console.Progress("teacher logits", done, sequences.Count, clock.Elapsed, unit: "sequences"), tuning.BatchTokens);
                    console.Finish();
                    console.Out.WriteLine($"teacher logits for {precomputed:N0} sequences ({sequences.Sum(s => (long)s.TrainedTokens):N0} trained tokens, top {k}) written to "
                                          + $"{Path.GetFullPath(precompute)} ({Downloader.Size(new FileInfo(precompute).Length)})");
                    console.Out.WriteLine($"next: idrak distill --teacher {precompute} --student {student} --data {string.Join(" --data ", data)} -o DIR");
                    return null;
                }

                source = stored ? DistillationTeachers.FromFile(teacher) : DistillationTeachers.FromModel(teacherModel!, topK ?? 0, tuning.BatchTokens);
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
            if (!generate && precompute is null)
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
    private static int Generate(CommandContext context, ToolConsole console, Models.ModelChoice choice, Device device, IReadOnlyList<string> data, string path)
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
                foreach (var row in source.Open(downloader))
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

        using var teacher = Models.Load(context, choice, device);
        var chat = Models.CreateChat(teacher, choice);
        var options = new TeacherDataOptions
        {
            Reasoning = context.Flag("--reasoning"),
            KeepCutOff = context.Flag("--keep-cut-off"),
            MaxNewTokens = maxNew,
            BatchSize = batchSize,
            Context = Math.Min(Models.DefaultContext, teacher.MaxPositions),
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
