// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Text.Retrieval.Abstractions;

namespace Idrak.Text.Retrieval;

/// <summary>
/// Re-ranks passages by reading the query and each passage together (a cross-encoder): more accurate than comparing
/// separate vectors, but it runs the model once per pair, so it re-orders a first stage's candidates rather than
/// searching a whole collection. The <see cref="Model"/> maps a batch of encoded pairs [N, ..shape] (the
/// <see cref="Encoder"/>'s) to one relevance score per pair ([N] or [N, 1]).
/// </summary>
public sealed class CrossEncoder : IReranker
{
    /// <summary>Creates the re-ranker.</summary>
    /// <param name="model">Encoded pairs [N, ..encoder.Shape] → scores [N] or [N, 1].</param>
    /// <param name="encoder">Turns (query, passage) into the model's input values for one pair (for example token ids of "&lt;cls&gt; query &lt;sep&gt; passage", padded).</param>
    public CrossEncoder(Module model, IPairEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (encoder.Shape is not { Count: > 0 } shape || shape.Any(d => d <= 0))
        {
            throw new ArgumentException("The pair shape needs at least one positive dimension.", nameof(encoder));
        }

        Model = model;
        Encoder = encoder;
        _shape = [.. shape];
    }

    /// <summary>Creates the re-ranker from a function that returns each pair's values (a <see cref="IPairEncoder"/> that copies them).</summary>
    /// <param name="model">Encoded pairs [N, ..pairShape] → scores [N] or [N, 1].</param>
    /// <param name="encodePair">Turns (query, passage) into the model's input values for one pair.</param>
    /// <param name="pairShape">The shape of one encoded pair (for example [maxLength]).</param>
    public CrossEncoder(Module model, Func<string, string, float[]> encodePair, int[] pairShape) : this(model, new FunctionPairEncoder(encodePair, pairShape))
    {
        _function = (FunctionPairEncoder)Encoder;
    }

    private readonly int[] _shape;
    private readonly FunctionPairEncoder? _function;                     // a function's encoder is given the strings themselves

    /// <summary>Encoded pairs → scores.</summary>
    public Module Model { get; }

    /// <summary>Turns a (query, passage) pair into model input values.</summary>
    public IPairEncoder Encoder { get; }

    /// <summary>The relevance score of every passage for <paramref name="query"/> (higher is more relevant), in one batch.</summary>
    public float[] Score(string query, IReadOnlyList<string> passages)
    {
        if (passages.Count == 0)
        {
            return [];
        }

        int size = _shape.Aggregate(1, (a, b) => a * b);
        var values = new float[passages.Count * size];
        for (int i = 0; i < passages.Count; i++)
        {
            var slot = values.AsSpan(i * size, size);                           // the pair's slot of the batch
            if (_function is not null)
            {
                _function.Encode(query, passages[i], slot);
            }
            else
            {
                Encoder.Encode(query, passages[i], slot);
            }
        }

        var device = Model.WeightsDevice ?? Device.Default;
        using var input = Tensor.From(values, [passages.Count, .. _shape], device);
        using var output = Model.Predict(input);
        var scores = output.ToArray();
        if (scores.Length != passages.Count)
        {
            throw new InvalidOperationException($"The model returned {scores.Length} values for {passages.Count} pairs; it must return one score per pair.");
        }

        return scores;
    }

    /// <summary>
    /// Re-orders <paramref name="candidates"/> by score, best first (ties keep their input order), and keeps the first
    /// <paramref name="keep"/>. Each result carries its new score.
    /// </summary>
    public IReadOnlyList<RetrievedChunk> Rerank(string query, IReadOnlyList<RetrievedChunk> candidates, int keep)
    {
        var scores = Score(query, [.. candidates.Select(c => c.Chunk.Text)]);
        return [.. candidates.Select((c, i) => (c, i)).OrderByDescending(p => scores[p.i]).ThenBy(p => p.i).Take(keep)
            .Select(p => p.c with { Score = scores[p.i] })];
    }

    /// <inheritdoc />
    public ValueTask<IReadOnlyList<RetrievedChunk>> RerankAsync(string query, IReadOnlyList<RetrievedChunk> candidates, int keep,
        CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(Rerank(query, candidates, keep));
}

// A function returning each pair's values, as a pair encoder: the values checked against the shape, then copied.
internal sealed class FunctionPairEncoder(Func<string, string, float[]> encode, int[] shape) : IPairEncoder
{
    private readonly int[] _shape = [.. shape ?? throw new ArgumentNullException(nameof(shape))];

    public IReadOnlyList<int> Shape => _shape;

    public void Encode(ReadOnlySpan<char> query, ReadOnlySpan<char> passage, Span<float> pair) => Encode(query.ToString(), passage.ToString(), pair);

    // The function called with the strings as given (the cross-encoder's own calls make none).
    public void Encode(string query, string passage, Span<float> pair)
    {
        var values = encode(query, passage);
        if (values.Length != pair.Length)
        {
            throw new InvalidOperationException($"The pair encoder returned {values.Length} values; the pair shape [{string.Join(", ", _shape)}] needs {pair.Length}.");
        }

        values.CopyTo(pair);
    }
}
