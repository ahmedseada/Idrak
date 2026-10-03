// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Retrieval;

/// <summary><c>idrak rag search "query" --index INDEX</c>: the best passages with their scores, without generation.</summary>
internal sealed class RagSearchCommand : Command
{
    public override string Name => "rag search";

    public override string Summary => "The best passages of an index for a query, with scores (no generation)";

    public override string Usage =>
        "\"query\" --index INDEX [options]\n\n" +
        "Searches an index built by 'idrak rag index': keywords (BM25), and with a hybrid index also vectors from the\n" +
        "model it was built with, fused by rank. Scores are BM25, or fused scores for a hybrid index.\n\n" +
        "Options:\n" +
        "      --index INDEX     the index file (required)\n" +
        "      --top N           passages to show (default 5)\n\n" +
        "Examples:\n" +
        "  idrak rag search \"reset the device\" --index docs.idx\n" +
        "  idrak rag search \"install on Android\" --index docs.idx --top 10 -j";

    public override IReadOnlyCollection<string> ValueOptions => ["--index", "--top"];

    public override int Run(CommandContext context)
    {
        string query = string.Join(' ', context.Positional);
        if (query.Length == 0)
        {
            throw new UsageException("Missing the query, e.g. idrak rag search \"how do I reset\" --index docs.idx");
        }

        string path = context.Option("--index") ?? throw new UsageException("Missing --index INDEX (built with 'idrak rag index').");
        int top = context.IntOption("--top", 5);
        if (top <= 0)
        {
            throw new UsageException("--top needs a number above 0.");
        }

        using var opened = RagIndexFile.Open(context, path);
        var hits = opened.Index.SearchAsync(query, top).AsTask().GetAwaiter().GetResult();
        context.Write($"{hits.Count} of {opened.Index.Chunks.Count} chunks for \"{query}\" ({opened.Search}):");
        context.Table(["#", "score", "source", "passage"], hits.Select((h, i) => (IReadOnlyList<string>)
            [(i + 1).ToString(), h.Score.ToString("G4"), $"{h.Chunk.DocumentId}#{h.Chunk.Position}", RagIndexFile.Snippet(h.Chunk.Text)]));
        context.WriteJson(new JsonObject
        {
            ["query"] = query,
            ["index"] = path,
            ["search"] = opened.Embedding is null ? "keywords" : "hybrid",
            ["chunks"] = opened.Index.Chunks.Count,
            ["hits"] = new JsonArray([.. hits.Select((h, i) => (JsonNode)RagIndexFile.ChunkJson(h, i + 1))]),
        });
        return ExitCodes.Ok;
    }
}
