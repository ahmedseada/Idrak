// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Retrieval;

namespace Idrak.Cli.Commands.Retrieval;

/// <summary><c>idrak rag index DIR --out INDEX</c>: chunks the text files of a folder and builds a keyword or hybrid index.</summary>
internal sealed class RagIndexCommand : Command
{
    internal const string DefaultExtensions = ".md,.markdown,.txt,.text,.rst";

    public override string Name => "rag index";

    public override string Summary => "Chunk a folder's text files and build a search index (BM25, and vectors with --model)";

    public override string Usage =>
        "DIR --out INDEX [options]\n\n" +
        "Reads the text files under DIR (or one file), splits them into overlapping chunks and builds a keyword index\n" +
        "(BM25). With --model, every chunk is also embedded with that model and searches fuse both results (reciprocal\n" +
        "rank fusion). The embeddings are the model's averaged hidden states: the library has no embedding-model\n" +
        "loader yet, so a chat model's vectors only complement keyword search (gap noted in plans/idrak-cli.md).\n\n" +
        "Options:\n" +
        "  -o, --out INDEX       the index file to write (required)\n" +
        "  -f, --force           replace an existing index file\n" +
        "  -m, --model MODEL     embed the chunks with MODEL for vector search (hybrid index)\n" +
        "  -w, --weights FORMAT  the embedding model's weight format (int8, int4, bf16, ...)\n" +
        "      --chunk N         units per chunk (default 200 words, or 5 sentences)\n" +
        "      --overlap N       units shared by consecutive chunks (default 40 words, or 1 sentence)\n" +
        "      --sentences       count chunks in sentences rather than words\n" +
        "      --extensions LIST the files to read (default " + DefaultExtensions + "; '*' for every file)\n" +
        "      --max-tokens N    tokens of a chunk the model embeds (default 512)\n\n" +
        "Examples:\n" +
        "  idrak rag index ./docs -o docs.idx\n" +
        "  idrak rag index ./docs -o docs.idx -m org/model -w int8 --chunk 120 --overlap 20";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--model", "--weights", "--chunk", "--overlap", "--extensions", "--max-tokens"];

    public override IReadOnlyCollection<string> Flags => ["--force", "--sentences"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-o"] = "--out", ["-f"] = "--force", ["-m"] = "--model", ["-w"] = "--weights",
    };

    public override int Run(CommandContext context)
    {
        string source = context.Argument(0, "DIR (the folder of documents to index)");
        if (context.Positional.Count > 1)
        {
            throw new UsageException("rag index takes one folder (or file); give several with a parent folder.");
        }

        string output = context.Option("--out") ?? throw new UsageException("Missing --out INDEX (the index file to write).");
        if (File.Exists(output) && !context.Flag("--force"))
        {
            throw new UsageException($"{output} exists; add --force to replace it.");
        }

        bool sentences = context.Flag("--sentences");
        int size = context.IntOption("--chunk", sentences ? 5 : 200), overlap = context.IntOption("--overlap", sentences ? 1 : 40);
        if (size <= 0 || overlap < 0 || overlap >= size)
        {
            throw new UsageException($"--chunk needs a number above 0 and --overlap a smaller one (got {size} and {overlap}).");
        }

        int maxTokens = context.IntOption("--max-tokens", 512);
        if (maxTokens <= 0)
        {
            throw new UsageException("--max-tokens needs a number above 0.");
        }

        var documents = ReadDocuments(source, context.Option("--extensions") ?? DefaultExtensions);
        if (documents.Count == 0)
        {
            throw new UsageException($"No documents to index in {source} (extensions {context.Option("--extensions") ?? DefaultExtensions}); choose others with --extensions.");
        }

        var watch = Stopwatch.StartNew();
        var builder = RetrievalIndex.Create().Documents(documents, sentences ? ChunkUnit.Sentences : ChunkUnit.Words, size, overlap).Bm25();
        Recorder? recorder = null;
        RagIndexFile.EmbeddingInfo? embedding = null;
        Idrak.Models.PretrainedModel? model = null;
        try
        {
            if (context.Option("--model") is { } modelName)
            {
                var choice = ModelChoices.Choose(context, modelName);
                model = ModelChoices.Load(context, choice);
                recorder = new Recorder(new ModelEmbedder(model, maxTokens), context);
                builder.Embeddings(recorder, batchSize: 32).Fusion(60, 50);
                embedding = new RagIndexFile.EmbeddingInfo(choice.Model, choice.Weights, maxTokens, model.Spec.Dim);
            }
            else if (context.Option("--weights") is not null)
            {
                throw new UsageException("--weights applies to the embedding model; add --model MODEL.");
            }

            var index = builder.Build();
            if (embedding is not null)
            {
                embedding = embedding with { Dimensions = recorder!.Vectors.Count > 0 ? recorder.Vectors[0].Length : embedding.Dimensions };
            }

            var extra = new JsonObject
            {
                ["source"] = Path.GetFullPath(source),
                ["documents"] = documents.Count,
                ["chunk"] = new JsonObject { ["unit"] = sentences ? "sentences" : "words", ["size"] = size, ["overlap"] = overlap },
            };
            RagIndexFile.Save(output, index, embedding, recorder?.Vectors, extra);
            long bytes = new FileInfo(output).Length;
            string search = embedding is null ? "keywords (BM25)" : $"keywords (BM25) and vectors ({embedding.Model}, {embedding.Dimensions} dimensions)";
            context.Write($"Indexed {documents.Count} document(s) into {index.Chunks.Count} chunks of up to {size} {(sentences ? "sentences" : "words")} ({overlap} shared) in {watch.Elapsed.TotalSeconds:F1} s");
            context.Write($"Search: {search}");
            context.Write($"Wrote {output} ({bytes / 1024.0:F1} KiB). Next: idrak rag search \"query\" --index {output}");
            context.WriteJson(new JsonObject
            {
                ["index"] = output,
                ["documents"] = documents.Count,
                ["chunks"] = index.Chunks.Count,
                ["bytes"] = bytes,
                ["keywords"] = true,
                ["vectors"] = embedding is not null,
                ["embedding_model"] = embedding?.Model,
                ["dimensions"] = embedding?.Dimensions,
                ["chunk"] = extra["chunk"]!.DeepClone(),
                ["seconds"] = watch.Elapsed.TotalSeconds,
            });
            return ExitCodes.Ok;
        }
        finally
        {
            model?.Dispose();
        }
    }

    /// <summary>The documents under <paramref name="source"/> (a folder or a file), ids relative to the folder.</summary>
    internal static List<Document> ReadDocuments(string source, string extensions)
    {
        if (File.Exists(source))
        {
            return [new Document(Path.GetFileName(source), File.ReadAllText(source))];
        }

        if (!Directory.Exists(source))
        {
            throw new UsageException($"No folder or file {source}.");
        }

        var wanted = extensions.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(e => e == "*" ? e : e.StartsWith('.') ? e : "." + e).ToHashSet(StringComparer.OrdinalIgnoreCase);
        return [.. Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories)
            .Where(f => wanted.Contains("*") || wanted.Contains(Path.GetExtension(f)))
            .Where(f => !Path.GetRelativePath(source, f).Split(Path.DirectorySeparatorChar).Any(p => p.StartsWith('.')))   // .git and other hidden folders
            .Order(StringComparer.Ordinal)
            .Select(f => new Document(Path.GetRelativePath(source, f).Replace('\\', '/'), File.ReadAllText(f)))
            .Where(d => d.Text.Length > 0)];
    }

    // Keeps the vectors the builder asks for (chunk order), so they can be written with the index.
    private sealed class Recorder(IEmbedder inner, CommandContext context) : IEmbedder
    {
        public List<float[]> Vectors { get; } = [];

        public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
        {
            var vectors = await inner.EmbedAsync(texts, cancellationToken).ConfigureAwait(false);
            Vectors.AddRange(vectors);
            context.Detail($"  embedded {Vectors.Count} chunks");
            return vectors;
        }
    }
}
