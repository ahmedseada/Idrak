using System.Runtime.InteropServices;
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
        var pieces = new List<(int Start, int Length)>();     // the units of a document, as places in its text
        foreach (var document in documents)
        {
            string text = document.Text;
            pieces.Clear();
            if (unit == ChunkUnit.Words)
            {
                // Runs of non-space characters, as string.Split(null, RemoveEmptyEntries) finds them.
                for (int i = 0; i < text.Length; i++)
                {
                    if (!char.IsWhiteSpace(text[i]))
                    {
                        int start = i;
                        while (i < text.Length && !char.IsWhiteSpace(text[i]))
                        {
                            i++;
                        }

                        pieces.Add((start, i - start));
                    }
                }
            }
            else
            {
                // The pieces between sentence ends, trimmed (string.Trim: the same white space), empty ones dropped.
                foreach (var range in SentenceEnd().EnumerateSplits(text))
                {
                    var (offset, length) = range.GetOffsetAndLength(text.Length);
                    var piece = text.AsSpan(offset, length);
                    int lead = piece.Length - piece.TrimStart().Length;
                    int kept = piece.Trim().Length;
                    if (kept > 0)
                    {
                        pieces.Add((offset + lead, kept));
                    }
                }
            }

            int position = 0;
            for (int start = 0; start < pieces.Count; start += size - overlap)
            {
                int count = Math.Min(size, pieces.Count - start);
                chunks.Add(new Chunk(chunks.Count, document.Id, position++, Join(text, pieces, start, count)));
                if (start + count >= pieces.Count)
                {
                    break;
                }
            }
        }

        return chunks;
    }

    // pieces[start .. start + count) of text joined by single spaces (string.Join(' ', …) without a string per piece).
    private static string Join(string text, List<(int Start, int Length)> pieces, int start, int count)
    {
        int length = count - 1;
        for (int i = start; i < start + count; i++)
        {
            length += pieces[i].Length;
        }

        return string.Create(length, (text, pieces, start, count), static (span, s) =>
        {
            for (int i = s.start; i < s.start + s.count; i++)
            {
                if (i > s.start)
                {
                    span[0] = ' ';
                    span = span[1..];
                }

                var (at, n) = s.pieces[i];
                s.text.AsSpan(at, n).CopyTo(span);
                span = span[n..];
            }
        });
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
        // Words get ids in order of first use (a string only for a new word); each text is counted by id, its words
        // taken in order of first use as a word-keyed count dictionary would list them.
        var ids = new Dictionary<string, int>();
        var lookup = ids.GetAlternateLookup<ReadOnlySpan<char>>();
        var postings = new List<(List<int> Documents, List<int> Counts)>();
        var counts = new List<int>();          // per word id, in the current text
        var used = new List<int>();            // the current text's word ids, in order of first use
        var lengths = new List<int>();
        foreach (var text in texts)
        {
            int length = 0;
            foreach (var word in WordTokenizer.SplitSpans(text))
            {
                length++;
                ref int id = ref CollectionsMarshal.GetValueRefOrAddDefault(lookup, word, out bool exists);
                if (!exists)
                {
                    id = postings.Count;
                    postings.Add(([], []));
                    counts.Add(0);
                }

                if (counts[id]++ == 0)
                {
                    used.Add(id);
                }
            }

            int d = lengths.Count;
            foreach (int id in used)
            {
                postings[id].Documents.Add(d);
                postings[id].Counts.Add(counts[id]);
                counts[id] = 0;
            }

            used.Clear();
            lengths.Add(length);
        }

        _lengths = [.. lengths];
        _averageLength = lengths.Count == 0 ? 1 : Math.Max(lengths.Average(), 1);
        foreach (var (word, id) in ids)
        {
            var list = postings[id];
            int n = list.Documents.Count;
            _idf[word] = Math.Log(1 + (lengths.Count - n + 0.5) / (n + 0.5));
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
        // The indexed words of the query, each once, in query order (the index's own strings: none is allocated).
        var terms = new List<(string Word, double Idf)>();
        var seen = new HashSet<string>();
        var idfs = _idf.GetAlternateLookup<ReadOnlySpan<char>>();
        foreach (var word in WordTokenizer.SplitSpans(query))
        {
            if (idfs.TryGetValue(word, out string? term, out double idf) && seen.Add(term))
            {
                terms.Add((term, idf));
            }
        }

        var scores = new double[_lengths.Length];

        // Only the texts holding a query word are scored (term by term in query order, as a full scan would add them).
        foreach (var (term, idf) in terms)
        {
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
