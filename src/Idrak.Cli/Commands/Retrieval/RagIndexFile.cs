// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Retrieval;

namespace Idrak.Cli.Commands.Retrieval;

/// <summary>
/// Embeddings from a language model's hidden states: each text's tokens go through every layer but the output head,
/// the last hidden states are averaged and scaled to length 1. A general chat model is not trained for this, so the
/// vectors are a rough complement to keyword search; a model trained for embeddings gives better ones.
/// </summary>
internal sealed class ModelEmbedder(PretrainedModel model, int maxTokens) : IEmbedder
{
    public PretrainedModel Model { get; } = model;

    public int MaxTokens { get; } = maxTokens;

    public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        var tokenizer = Model.Tokenizer ?? throw new InvalidOperationException("The embedding model has no tokenizer (tokenizer.json).");
        var network = Model.Network;
        var result = new float[texts.Count][];
        using var noGrad = Autograd.NoGrad();
        for (int t = 0; t < texts.Count; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var ids = tokenizer.Encode(texts[t]);
            float[] input = ids.Count == 0 ? [0f] : [.. ids.Take(Math.Min(MaxTokens, Model.MaxPositions)).Select(i => (float)i)];
            using var scope = new TensorScope();
            var hidden = network.ForwardFirst(Tensor.From(input, [1, input.Length], Model.Device), network.Count - 1);   // [1, T, D]
            var values = hidden.ToArray();
            int d = hidden.Shape[^1], steps = values.Length / d;
            var vector = new float[d];
            for (int s = 0; s < steps; s++)
            {
                for (int i = 0; i < d; i++)
                {
                    vector[i] += values[s * d + i];
                }
            }

            double norm = Math.Sqrt(vector.Sum(v => (double)v * v));
            for (int i = 0; i < d; i++)
            {
                vector[i] = norm > 0 ? (float)(vector[i] / norm) : 0f;
            }

            result[t] = vector;
        }

        return ValueTask.FromResult(result);
    }
}

/// <summary>
/// The index file <c>idrak rag</c> writes: the library's retrieval index (<see cref="RetrievalIndex.Save(Stream)"/>,
/// a zip archive) with two more entries when it has vectors: <c>idrak-rag.json</c> (which model made them and how)
/// and <c>embeddings.nsv</c> (the chunk vectors, which the library keeps in a vector store rather than the file).
/// </summary>
internal static class RagIndexFile
{
    private const string Format = "idrak-rag/1";

    /// <summary>What an index file holds about its vectors, or null for a keyword-only index.</summary>
    internal sealed record EmbeddingInfo(string Model, string? Weights, int MaxTokens, int Dimensions);

    /// <summary>An opened index with its embedding model (when it has vectors); dispose to free the model.</summary>
    internal sealed class Opened(RetrievalIndex index, EmbeddingInfo? embedding, ModelEmbedder? embedder, bool ownsModel) : IDisposable
    {
        public RetrievalIndex Index { get; } = index;

        public EmbeddingInfo? Embedding { get; } = embedding;

        public ModelEmbedder? Embedder { get; } = embedder;

        public string Search => Embedding is null ? "keywords (BM25)" : $"keywords (BM25) and vectors ({Embedding.Model}), fused";

        public void Dispose()
        {
            if (ownsModel)
            {
                Embedder?.Model.Dispose();
            }
        }
    }

    /// <summary>Writes <paramref name="index"/> and, for vector search, the vectors and the model that made them.</summary>
    public static void Save(string path, RetrievalIndex index, EmbeddingInfo? embedding, IReadOnlyList<float[]>? vectors, JsonObject? extra = null)
    {
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite);
        index.Save(stream);
        stream.Position = 0;
        using var zip = new ZipArchive(stream, ZipArchiveMode.Update);
        var info = new JsonObject { ["format"] = Format };
        if (extra is not null)
        {
            foreach (var (key, value) in extra)
            {
                info[key] = value?.DeepClone();
            }
        }

        if (embedding is not null && vectors is not null)
        {
            info["embedding"] = new JsonObject
            {
                ["model"] = embedding.Model,
                ["weights"] = embedding.Weights,
                ["max_tokens"] = embedding.MaxTokens,
                ["dimensions"] = embedding.Dimensions,
            };
            var store = new VectorIndex(embedding.Dimensions, VectorMetric.Cosine);
            store.AddRange(vectors);
            using var entry = zip.CreateEntry("embeddings.nsv").Open();
            store.Save(entry);
        }

        using var writer = new StreamWriter(zip.CreateEntry("idrak-rag.json").Open());
        writer.Write(info.ToJsonString(CommandContext.JsonOutput));
    }

    /// <summary>
    /// Opens an index file; one with vectors loads its embedding model on the context's device (or reuses
    /// <paramref name="loaded"/> when that is the same model with the same weights).
    /// </summary>
    public static Opened Open(CommandContext context, string path, (string Model, string? Weights, PretrainedModel Instance)? loaded = null)
    {
        if (!File.Exists(path))
        {
            throw new UsageException($"No index {path}: build one with 'idrak rag index DIR --out {path}'.");
        }

        using var stream = File.OpenRead(path);
        JsonObject info;
        VectorIndex? vectors = null;
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true))
        {
            var entry = zip.GetEntry("idrak-rag.json");
            if (entry is null)
            {
                stream.Position = 0;
                return new Opened(RetrievalIndex.Load(stream, null), null, null, false);       // a library index without vectors
            }

            using (var reader = new StreamReader(entry.Open()))
            {
                info = JsonNode.Parse(reader.ReadToEnd()) as JsonObject ?? [];
            }

            if (zip.GetEntry("embeddings.nsv") is { } stored)
            {
                using var buffer = new MemoryStream();
                using (var s = stored.Open())
                {
                    s.CopyTo(buffer);
                }

                buffer.Position = 0;
                vectors = VectorIndex.Load(buffer);
            }
        }

        stream.Position = 0;
        if (info["embedding"] is not JsonObject e || vectors is null)
        {
            return new Opened(RetrievalIndex.Load(stream, null), null, null, false);
        }

        var embedding = new EmbeddingInfo((string)e["model"]!, (string?)e["weights"], (int?)e["max_tokens"] ?? 512, (int?)e["dimensions"] ?? vectors.Dimensions);
        bool reuse = loaded is { } l && l.Model == embedding.Model && l.Weights == embedding.Weights;
        var model = reuse ? loaded!.Value.Instance : ModelChoices.Load(context, new ModelChoices.ModelChoice(embedding.Model, embedding.Weights, null, null, null));
        var store = new InMemoryVectorStore(vectors.Dimensions, VectorMetric.Cosine);
        store.UpsertAsync([.. Enumerable.Range(0, vectors.Count).Select(i => new VectorRecord(i.ToString(System.Globalization.CultureInfo.InvariantCulture), vectors[i]))])
            .AsTask().GetAwaiter().GetResult();
        var embedder = new ModelEmbedder(model, embedding.MaxTokens);
        return new Opened(RetrievalIndex.Load(stream, embedder, store), embedding, embedder, !reuse);
    }

    /// <summary>A chunk as JSON: where it came from, its score and its text.</summary>
    public static JsonObject ChunkJson(RetrievedChunk hit, int rank) => new()
    {
        ["rank"] = rank,
        ["document"] = hit.Chunk.DocumentId,
        ["position"] = hit.Chunk.Position,
        ["chunk"] = hit.Chunk.Id,
        ["score"] = hit.Score,
        ["keyword_rank"] = hit.KeywordRank,
        ["vector_rank"] = hit.VectorRank,
        ["text"] = hit.Chunk.Text,
    };

    /// <summary>The first <paramref name="length"/> characters of a passage on one line.</summary>
    public static string Snippet(string text, int length = 90)
    {
        string flat = string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        return flat.Length > length ? flat[..length] + "…" : flat;
    }
}
