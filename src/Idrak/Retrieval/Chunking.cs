using System.Text.RegularExpressions;
using Idrak.Generation;

namespace Idrak.Retrieval;

/// <summary>A document to index: an id you choose (a file name, a URL, a database key) and its text.</summary>
public sealed record Document(string Id, string Text);

/// <summary>A passage of a document, the unit that is indexed and retrieved.</summary>
/// <param name="Id">Position in the index (0, 1, 2, …).</param>
/// <param name="DocumentId">The document it came from.</param>
/// <param name="Position">Its number within that document (0 for the first chunk).</param>
/// <param name="Text">The passage.</param>
public sealed record Chunk(int Id, string DocumentId, int Position, string Text);

/// <summary>What a chunk's size and overlap count.</summary>
public enum ChunkUnit
{
    /// <summary>Words (runs of non-space characters).</summary>
    Words,

    /// <summary>Sentences (ending with ., ! or ? followed by a space, or a line break).</summary>
    Sentences,
}

/// <summary>Splits documents into overlapping passages.</summary>
public static partial class Chunker
{
    /// <summary>
    /// Splits every document into chunks of <paramref name="size"/> units; consecutive chunks of a document share
    /// <paramref name="overlap"/> units (0 for none). The last chunk of a document may be shorter. Chunks are numbered
    /// in order across all documents.
    /// </summary>
    public static IReadOnlyList<Chunk> Split(IEnumerable<Document> documents, ChunkUnit unit, int size, int overlap)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        ArgumentOutOfRangeException.ThrowIfNegative(overlap);
        if (overlap >= size)
        {
            throw new ArgumentOutOfRangeException(nameof(overlap), "The overlap must be smaller than the chunk size.");
        }

        var chunks = new List<Chunk>();
        foreach (var document in documents)
        {
            string[] pieces = unit == ChunkUnit.Words
                ? document.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
                : [.. SentenceEnd().Split(document.Text).Select(s => s.Trim()).Where(s => s.Length > 0)];
            int position = 0;
            for (int start = 0; start < pieces.Length; start += size - overlap)
            {
                int count = Math.Min(size, pieces.Length - start);
                chunks.Add(new Chunk(chunks.Count, document.Id, position++, string.Join(' ', pieces, start, count)));
                if (start + count >= pieces.Length)
                {
                    break;
                }
            }
        }

        return chunks;
    }

    [GeneratedRegex(@"(?<=[.!?])\s+|\r?\n+")]
    private static partial Regex SentenceEnd();
}

/// <summary>One search result: the chunk's id and its score (higher is better).</summary>
public readonly record struct SearchHit(int Id, double Score);

/// <summary>The best hits offered so far (best first, ties to the lower id), kept in a heap whose root is the worst of them.</summary>
internal sealed class TopHits(int keep)
{
    private readonly PriorityQueue<int, (double Score, int Id)> _heap = new(Math.Max(keep, 0), WorstFirst.Instance);

    public void Offer(int id, double score)
    {
        if (keep <= 0)
        {
            return;
        }

        if (_heap.Count < keep)
        {
            _heap.Enqueue(id, (score, id));
        }
        else if (_heap.TryPeek(out _, out var worst) && WorstFirst.Instance.Compare((score, id), worst) > 0)
        {
            _heap.DequeueEnqueue(id, (score, id));
        }
    }

    public SearchHit[] ToArray()
    {
        var hits = new SearchHit[_heap.Count];
        for (int r = hits.Length - 1; _heap.TryDequeue(out int id, out var entry); r--)
        {
            hits[r] = new SearchHit(id, entry.Score);
        }

        return hits;
    }

    // A lower score, or the same score with a higher id, is worse.
    private sealed class WorstFirst : IComparer<(double Score, int Id)>
    {
        public static readonly WorstFirst Instance = new();

        public int Compare((double Score, int Id) x, (double Score, int Id) y) =>
            x.Score != y.Score ? x.Score.CompareTo(y.Score) : y.Id.CompareTo(x.Id);
    }
}

/// <summary>
/// Okapi BM25 keyword search: scores a text by the query words it contains, weighting rare words more, damping
/// repeated words and penalizing long texts slightly. Words are split like <see cref="WordTokenizer.Split"/>.
/// </summary>
public sealed class Bm25Index
{
    private readonly int[] _lengths;
    private readonly Dictionary<string, double> _idf = [];
    private readonly Dictionary<string, (int[] Documents, int[] Counts)> _postings = [];   // the texts holding each word, in id order
    private readonly double _averageLength;

    /// <summary>Indexes <paramref name="texts"/> (their ids are their positions).</summary>
    /// <param name="texts">The texts to search.</param>
    /// <param name="k1">Term-frequency saturation (typically 1.2–2.0).</param>
    /// <param name="b">Length normalization, 0 (none) to 1 (full); typically 0.75.</param>
    public Bm25Index(IEnumerable<string> texts, double k1 = 1.2, double b = 0.75)
    {
        K1 = k1;
        B = b;
        var documents = texts.Select(t => WordTokenizer.Split(t).ToArray()).ToList();
        _lengths = [.. documents.Select(d => d.Length)];
        _averageLength = documents.Count == 0 ? 1 : Math.Max(documents.Average(d => d.Length), 1);
        var postings = new Dictionary<string, (List<int> Documents, List<int> Counts)>();
        var counts = new Dictionary<string, int>();
        for (int d = 0; d < documents.Count; d++)
        {
            counts.Clear();
            foreach (var word in documents[d])
            {
                counts[word] = counts.GetValueOrDefault(word) + 1;
            }

            foreach (var (word, count) in counts)
            {
                if (!postings.TryGetValue(word, out var list))
                {
                    postings[word] = list = ([], []);
                }

                list.Documents.Add(d);
                list.Counts.Add(count);
            }
        }

        foreach (var (word, list) in postings)
        {
            int n = list.Documents.Count;
            _idf[word] = Math.Log(1 + (documents.Count - n + 0.5) / (n + 0.5));
            _postings[word] = ([.. list.Documents], [.. list.Counts]);
        }
    }

    /// <summary>Term-frequency saturation.</summary>
    public double K1 { get; }

    /// <summary>Length normalization.</summary>
    public double B { get; }

    /// <summary>Number of indexed texts.</summary>
    public int Count => _lengths.Length;

    /// <summary>The <paramref name="top"/> best texts for <paramref name="query"/>, best first (ties by id); texts sharing no word are left out.</summary>
    public IReadOnlyList<SearchHit> Search(string query, int top)
    {
        var terms = WordTokenizer.Split(query).Where(_idf.ContainsKey).Distinct().ToArray();
        var scores = new double[_lengths.Length];

        // Only the texts holding a query word are scored (term by term in query order, as a full scan would add them).
        foreach (var term in terms)
        {
            double idf = _idf[term];
            var (documents, counts) = _postings[term];
            for (int i = 0; i < documents.Length; i++)
            {
                int d = documents[i], tf = counts[i];
                scores[d] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * _lengths[d] / _averageLength));
            }
        }

        var best = new TopHits(top);
        for (int d = 0; d < scores.Length; d++)
        {
            if (scores[d] > 0)
            {
                best.Offer(d, scores[d]);
            }
        }

        return best.ToArray();
    }
}
