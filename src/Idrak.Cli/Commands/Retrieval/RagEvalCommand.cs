// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Retrieval;

namespace Idrak.Cli.Commands.Retrieval;

/// <summary>
/// <c>idrak rag eval --index INDEX --questions FILE</c>: retrieval quality on questions with known answers: the hit
/// rate at 1 and at N and the mean reciprocal rank of the first passage that answers.
/// </summary>
internal sealed class RagEvalCommand : Command
{
    public override string Name => "rag eval";

    public override string Summary => "Retrieval quality of an index (hit rate, MRR) on question/passage pairs";

    public override string Usage =>
        "--index INDEX --questions FILE [options]\n\n" +
        "FILE is JSON Lines, one question per line: {\"question\": \"...\", \"document\": \"guide.md\"} and/or\n" +
        "{\"passage\": \"text the answer contains\"}. A retrieved chunk answers when it comes from that document (its\n" +
        "path, or a path ending with it) or contains the passage (ignoring case and spacing); with both, it must do both.\n" +
        "Reported: hit rate at 1 and at --top, and the mean reciprocal rank (MRR) of the first chunk that answers.\n\n" +
        "Options:\n" +
        "      --index INDEX     the index file (required)\n" +
        "      --questions FILE  the questions (required)\n" +
        "      --top N           chunks retrieved per question (default 5)\n\n" +
        "Examples:\n" +
        "  idrak rag eval --index docs.idx --questions questions.jsonl\n" +
        "  idrak rag eval --index docs.idx --questions questions.jsonl --top 10 -j";

    public override IReadOnlyCollection<string> ValueOptions => ["--index", "--questions", "--top"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"rag eval takes its files as --index and --questions, not '{context.Positional[0]}'.");
        }

        string path = context.Option("--index") ?? throw new UsageException("Missing --index INDEX (built with 'idrak rag index').");
        string questionsPath = context.Option("--questions") ?? throw new UsageException("Missing --questions FILE (JSON Lines of questions with their document or passage).");
        if (!File.Exists(questionsPath))
        {
            throw new UsageException($"No file {questionsPath}.");
        }

        int top = context.IntOption("--top", 5);
        if (top <= 0)
        {
            throw new UsageException("--top needs a number above 0.");
        }

        var questions = ReadQuestions(questionsPath);
        using var opened = RagIndexFile.Open(context, path);
        var rows = new JsonArray();
        int hitsAtOne = 0, hitsAtTop = 0;
        double reciprocal = 0;
        foreach (var (question, document, passage) in questions)
        {
            var hits = opened.Index.SearchAsync(question, top).AsTask().GetAwaiter().GetResult();
            int rank = 0;
            for (int i = 0; i < hits.Count && rank == 0; i++)
            {
                if (Answers(hits[i].Chunk, document, passage))
                {
                    rank = i + 1;
                }
            }

            hitsAtOne += rank == 1 ? 1 : 0;
            hitsAtTop += rank > 0 ? 1 : 0;
            reciprocal += rank > 0 ? 1.0 / rank : 0;
            context.Detail($"  {(rank > 0 ? $"rank {rank}" : "miss  ")}  {RagIndexFile.Snippet(question, 70)}");
            rows.Add(new JsonObject
            {
                ["question"] = question,
                ["rank"] = rank > 0 ? rank : null,
                ["top"] = hits.Count > 0 ? $"{hits[0].Chunk.DocumentId}#{hits[0].Chunk.Position}" : null,
            });
        }

        int n = questions.Count;
        double hitRate1 = (double)hitsAtOne / n, hitRate = (double)hitsAtTop / n, mrr = reciprocal / n;
        context.Write($"{n} questions on {path} ({opened.Search}), top {top}:");
        context.Write($"  hit rate @1 {hitRate1:P1} · hit rate @{top} {hitRate:P1} · MRR {mrr:F3}");
        context.WriteJson(new JsonObject
        {
            ["index"] = path,
            ["questions"] = n,
            ["top"] = top,
            ["search"] = opened.Embedding is null ? "keywords" : "hybrid",
            ["hit_rate_at_1"] = hitRate1,
            ["hit_rate"] = hitRate,
            ["mrr"] = mrr,
            ["results"] = rows,
        });
        return ExitCodes.Ok;
    }

    /// <summary>Whether <paramref name="chunk"/> answers: from the document, containing the passage, or both when both are given.</summary>
    internal static bool Answers(Chunk chunk, string? document, string? passage)
    {
        bool fromDocument = document is null || chunk.DocumentId == document
            || chunk.DocumentId.EndsWith("/" + document.TrimStart('/'), StringComparison.Ordinal);
        bool hasPassage = passage is null || Normalize(chunk.Text).Contains(Normalize(passage), StringComparison.Ordinal);
        return fromDocument && hasPassage;
    }

    private static string Normalize(string text) => string.Join(' ', text.ToLowerInvariant().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    private static List<(string Question, string? Document, string? Passage)> ReadQuestions(string path)
    {
        var questions = new List<(string, string?, string?)>();
        int line = 0;
        foreach (string text in File.ReadLines(path))
        {
            line++;
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            JsonObject row;
            try
            {
                row = JsonNode.Parse(text) as JsonObject ?? throw new InvalidDataException($"{path}, line {line}: not a JSON object.");
            }
            catch (System.Text.Json.JsonException e)
            {
                throw new InvalidDataException($"{path}, line {line}: not JSON ({e.Message}).");
            }

            string question = (string?)row["question"] ?? (string?)row["query"]
                ?? throw new InvalidDataException($"{path}, line {line}: no \"question\".");
            string? document = (string?)row["document"];
            string? passage = (string?)row["passage"] ?? (string?)row["answer"];
            if (document is null && passage is null)
            {
                throw new InvalidDataException($"{path}, line {line}: give \"document\" or \"passage\" to say which chunks answer.");
            }

            questions.Add((question, document, passage));
        }

        return questions.Count > 0 ? questions : throw new InvalidDataException($"{path} has no questions.");
    }
}
