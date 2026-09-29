using Idrak.Generation;

namespace Idrak.Samples.Rag;

/// <summary>
/// Words of a fixed vocabulary, with two additions for text about invented towns:
/// numbers are spelled digit by digit after a &lt;num&gt; marker ("1450" → &lt;num&gt; 1 4 5 0), so any number can be read
/// and copied with ten digit tokens; and words outside the vocabulary (town names) are kept distinguishable in one of
/// two ways:
/// <list type="bullet">
/// <item>placeholders &lt;w0&gt;, &lt;w1&gt;, … numbered by first appearance in the text: "the same unknown word" is visible
/// within one text (question and passages read together, by the re-ranker and the chat model);</item>
/// <item>hashed buckets &lt;h0&gt; … &lt;h63&gt; (the hashing trick): the same word always gets the same token, in every text,
/// so a question and a passage encoded separately (by the bi-encoder) still share their town's token.</item>
/// </list>
/// Decoding is stateless (placeholders stay as they are), so it works on the pieces a generator decodes as it streams.
/// </summary>
public sealed class TownTokenizer(WordTokenizer words, int placeholders, int hashBuckets) : ITokenizer
{
    public const int Buckets = 64;

    public static readonly string[] Specials = ["<pad>", "<unk>", "<cls>", "<sep>", "<num>", .. Enumerable.Range(0, 10).Select(d => d.ToString()),
        .. Enumerable.Range(0, Buckets).Select(i => $"<h{i}>")];

    public WordTokenizer Words { get; } = words;

    public int VocabularySize => Words.VocabularySize;

    public static string[] Placeholders(int count) => [.. Enumerable.Range(0, count).Select(i => $"<w{i}>")];

    /// <summary>The words a vocabulary is built from: every word of the texts except numbers (spelled as digits).</summary>
    public static IEnumerable<string> WordsOf(string text) => WordTokenizer.Split(text).Where(w => !w.All(char.IsAsciiDigit));

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        var seen = new Dictionary<string, int>();
        foreach (var word in WordTokenizer.Split(text))
        {
            if (word.All(char.IsAsciiDigit))
            {
                ids.Add(Words["<num>"]);
                ids.AddRange(word.Select(d => Words[d.ToString()]));
            }
            else if (Words[word] != Words["<unk>"])
            {
                ids.Add(Words[word]);
            }
            else if (hashBuckets > 0)
            {
                ids.Add(Words[$"<h{Hash(word) % (uint)hashBuckets}>"]);
            }
            else if (seen.TryGetValue(word, out int k) || seen.Count < placeholders)
            {
                ids.Add(Words[$"<w{(seen.TryGetValue(word, out k) ? k : seen[word] = seen.Count)}>"]);
            }
            else
            {
                ids.Add(Words["<unk>"]);
            }
        }

        return ids;
    }

    // FNV-1a: stable across runs and machines (string.GetHashCode is randomized per process).
    private static uint Hash(string word)
    {
        uint hash = 2166136261;
        foreach (char c in word)
        {
            hash = (hash ^ c) * 16777619;
        }

        return hash;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var text = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            string word = (uint)id < (uint)Words.Vocabulary.Count ? Words.Vocabulary[id] : "<unk>";
            if (word == "<num>")
            {
                text.Append(' ');
            }
            else if (word.Length == 1 && char.IsAsciiDigit(word[0]))
            {
                text.Append(word);
            }
            else
            {
                text.Append(Words.Decode([id]));
            }
        }

        return text.ToString();
    }
}
