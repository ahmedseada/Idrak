using Idrak.Layers;

namespace Idrak.Retrieval;

/// <summary>
/// Re-ranks passages by reading the query and each passage together (a cross-encoder): more accurate than comparing
/// separate vectors, but it runs the model once per pair, so it re-orders a first stage's candidates rather than
/// searching a whole collection. The <see cref="Model"/> maps a batch of encoded pairs [N, ..PairShape] to one
/// relevance score per pair ([N] or [N, 1]).
/// </summary>
public sealed class CrossEncoder
{
    /// <summary>Creates the re-ranker.</summary>
    /// <param name="model">Encoded pairs [N, ..pairShape] → scores [N] or [N, 1].</param>
    /// <param name="encodePair">Turns (query, passage) into the model's input values for one pair (for example token ids of "&lt;cls&gt; query &lt;sep&gt; passage", padded).</param>
    /// <param name="pairShape">The shape of one encoded pair (for example [maxLength]).</param>
    public CrossEncoder(Module model, Func<string, string, float[]> encodePair, int[] pairShape)
    {
        ArgumentNullException.ThrowIfNull(pairShape);
        if (pairShape.Length == 0 || pairShape.Any(d => d <= 0))
        {
            throw new ArgumentException("The pair shape needs at least one positive dimension.", nameof(pairShape));
        }

        Model = model;
        EncodePair = encodePair;
        PairShape = [.. pairShape];
    }

    /// <summary>Encoded pairs → scores.</summary>
    public Module Model { get; }

    /// <summary>Turns a (query, passage) pair into model input values.</summary>
    public Func<string, string, float[]> EncodePair { get; }

    /// <summary>The shape of one encoded pair.</summary>
    public IReadOnlyList<int> PairShape { get; }

    /// <summary>The relevance score of every passage for <paramref name="query"/> (higher is more relevant), in one batch.</summary>
    public float[] Score(string query, IReadOnlyList<string> passages)
    {
        if (passages.Count == 0)
        {
            return [];
        }

        int size = PairShape.Aggregate(1, (a, b) => a * b);
        var values = new float[passages.Count * size];
        for (int i = 0; i < passages.Count; i++)
        {
            var pair = EncodePair(query, passages[i]);
            if (pair.Length != size)
            {
                throw new InvalidOperationException($"The pair encoder returned {pair.Length} values; the pair shape [{string.Join(", ", PairShape)}] needs {size}.");
            }

            pair.CopyTo(values, i * size);
        }

        var device = Model.Parameters().FirstOrDefault()?.Device ?? Device.Default;
        using var input = Tensor.From(values, [passages.Count, .. PairShape], device);
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
}
