// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Layers;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak eval MODEL SET.jsonl</c>: answers each conversation's last user turn greedily and scores the answer
/// against the conversation's own last assistant message (the library's <see cref="ChatEvaluation"/>).
/// </summary>
internal sealed class EvalCommand : Command
{
    public override string Name => "eval";

    public override string Summary => "Answer metrics of a chat model on held-out conversations (accuracy, exact match, F1)";

    public override string Usage =>
        "MODEL SET.jsonl [options]\n\n" +
        "SET.jsonl holds one conversation per line, {\"messages\": [...]}, ending with the reference assistant answer;\n" +
        "the model answers the turns before it (greedy decoding, so results repeat) and the answer is scored.\n\n" +
        "Options:\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format\n" +
        "  -k, --kv FORMAT       the key/value cache format\n" +
        "      --context N       the context length; --adapter DIR: merge an adapter first\n" +
        "      --metric NAME     auto (default: number for '#### n' references, else f1), number, exact, contains, f1\n" +
        "      --max-tokens N    longest answer (default 512)\n" +
        "      --batch N         conversations answered together (default 1)\n" +
        "      --limit N         only the first N conversations\n" +
        "      --think / --no-think  the reasoning mode passed to the chat template (default: the template's)\n" +
        "  -o, --out FILE        write every answer with its score as JSON Lines\n\n" +
        "Metrics beyond these (judged answers, pass@k) wait for the library's evaluation plug-ins (plug-in gap 16).\n\n" +
        "Examples:\n" +
        "  idrak eval org/model test-set.jsonl --limit 100 --no-think\n" +
        "  idrak eval ./tuned held-out.jsonl --metric exact -o answers.jsonl -j\n\n" +
        "Environment: IDRAK_CACHE (models), IDRAK_MATMUL, IDRAK_OFFLOAD, HF_TOKEN (gated downloads)";

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, "--metric", "--max-tokens", "--batch", "--limit", "--out"];

    public override IReadOnlyCollection<string> Flags => ["--think", "--no-think"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>(Models.ShortForms) { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string modelName = context.Argument(0, "MODEL");
        string set = context.Argument(1, "SET.jsonl (the conversations to evaluate)");
        if (context.Positional.Count > 2)
        {
            throw new UsageException("eval takes a model and one set of conversations.");
        }

        if (!File.Exists(set))
        {
            throw new UsageException($"No file {set}: give a JSON Lines file of conversations ({{\"messages\": [...]}} per line).");
        }

        var metric = (context.Option("--metric")?.ToLowerInvariant()) switch
        {
            null or "auto" => AnswerMetric.Auto,
            "number" => AnswerMetric.Number,
            "exact" => AnswerMetric.Exact,
            "contains" => AnswerMetric.Contains,
            "f1" => AnswerMetric.F1,
            var other => throw new UsageException($"--metric takes auto, number, exact, contains or f1, not '{other}'."),
        };
        if (context.Flag("--think") && context.Flag("--no-think"))
        {
            throw new UsageException("Choose --think or --no-think, not both.");
        }

        bool? think = context.Flag("--think") ? true : context.Flag("--no-think") ? false : null;
        int maxTokens = context.IntOption("--max-tokens", 512), batch = context.IntOption("--batch", 1), limit = context.IntOption("--limit", int.MaxValue);
        if (maxTokens <= 0 || batch <= 0 || limit <= 0)
        {
            throw new UsageException("--max-tokens, --batch and --limit need numbers above 0.");
        }

        var rows = ReadRows(set).Take(limit).ToList();
        var choice = Models.Choose(context, modelName);
        using var model = Models.Load(context, choice);
        var chat = model.CreateChat(KeyValueLayouts.Get(choice.Kv ?? "float32"), model.MaxPositions);
        int done = 0;
        var report = ChatEvaluation.Run(chat, rows, metric, maxTokens, think,
            new Progress(a => context.Detail($"  {++done}: score {a.Score:F2}, {a.Tokens} tokens")), model.MaxPositions, batchSize: batch);

        if (context.Option("--out") is { } outPath)
        {
            using var writer = new StreamWriter(outPath);
            foreach (var answer in report.Answers)
            {
                writer.WriteLine(AnswerJson(answer).ToJsonString());
            }
        }

        string metricName = report.Metric.ToString().ToLowerInvariant();
        context.Write($"{choice.Model} on {context.Device}: {report.Answers.Count} of {rows.Count} conversations scored ({rows.Count - report.Answers.Count} without a final assistant answer skipped)");
        context.Write($"{metricName} {report.Score:P1} · {report.MeanTokens:F0} tokens per answer · {report.TokensPerSecond:F1} tokens/s · {report.Duration.TotalSeconds:F1} s");
        if (context.Verbose)
        {
            foreach (var wrong in report.Answers.Where(a => a.Score < 1).Take(5))
            {
                context.Detail($"  expected: {Shorten(wrong.Reference)}\n  answered: {Shorten(wrong.Answer)}");
            }
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = context.Device.ToString(),
            ["metric"] = metricName,
            ["score"] = report.Score,
            ["conversations"] = rows.Count,
            ["scored"] = report.Answers.Count,
            ["mean_tokens"] = report.MeanTokens,
            ["tokens_per_second"] = report.TokensPerSecond,
            ["seconds"] = report.Duration.TotalSeconds,
            ["answers"] = new JsonArray([.. report.Answers.Select(a => (JsonNode)AnswerJson(a))]),
        });
        return ExitCodes.Ok;
    }

    private static IEnumerable<JsonObject> ReadRows(string path)
    {
        int line = 0;
        foreach (string text in File.ReadLines(path))
        {
            line++;
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            JsonObject? row;
            try
            {
                row = JsonNode.Parse(text) as JsonObject;
            }
            catch (System.Text.Json.JsonException e)
            {
                throw new InvalidDataException($"{path}, line {line}: not JSON ({e.Message}). Check it with 'idrak data validate {path} --as chat'.");
            }

            yield return row ?? throw new InvalidDataException($"{path}, line {line}: not a JSON object.");
        }
    }

    private static JsonObject AnswerJson(EvaluatedAnswer a) => new()
    {
        ["question"] = a.Prompt.LastOrDefault(m => m.Role == "user")?.Content,
        ["reference"] = a.Reference,
        ["answer"] = a.Answer,
        ["score"] = a.Score,
        ["tokens"] = a.Tokens,
    };

    private static string Shorten(string text)
    {
        string flat = text.ReplaceLineEndings(" ");
        return flat.Length > 100 ? flat[..100] + "…" : flat;
    }

    private sealed class Progress(Action<EvaluatedAnswer> report) : IProgress<EvaluatedAnswer>
    {
        public void Report(EvaluatedAnswer value) => report(value);
    }
}
