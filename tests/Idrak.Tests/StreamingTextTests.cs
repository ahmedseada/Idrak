using System.Text;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;

// Streamed text stays whole characters however a byte-level tokenizer splits their UTF-8 bytes across tokens.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] StreamingText =
    [
        ("generation: streamed text keeps every character whole however tokens split its UTF-8 bytes (emoji, stops)", StreamedCharactersStayWhole),
    ];

    private static void StreamedCharactersStayWhole(Device device)
    {
        // Every split of these bytes into tokens of 1 to 4 bytes, one of them ending one emoji and starting the next.
        byte[] tail = Encoding.UTF8.GetBytes("? \U0001F60A\U0001F60A!");
        string expected = "ok today" + Encoding.UTF8.GetString(tail);
        int splits = 0;
        foreach (var widths in ByteSplits(tail.Length, 0))
        {
            var pieces = new List<byte[]> { "P"u8.ToArray() };
            pieces.AddRange(Encoding.UTF8.GetBytes("ok today").Select(b => new[] { b }));
            int at = 0;
            foreach (int w in widths)
            {
                pieces.Add(tail[at..(at + w)]);
                at += w;
            }

            pieces.Add("<|im_end|>"u8.ToArray());
            var tokenizer = new ByteTokens(pieces);
            using var model = ScriptModel(pieces.Count, device);
            var generator = new TextGenerator(model, tokenizer, contextLength: 64);
            var options = new GenerationOptions
            {
                Temperature = 0f, TopK = 1, NumPredict = pieces.Count + 4, Stop = ["<|im_end|>", "<|endoftext|>"], UseCache = false, UseGraph = false,
            };
            var chunks = generator.Stream("P", options).Select(c => c.Text).ToList();
            string text = string.Concat(chunks);
            Check(text == expected, $"split [{string.Join(",", widths)}]: '{text}'");
            Check(chunks.All(WholeCharacters), $"split [{string.Join(",", widths)}]: a chunk holds half of a character");
            var batch = generator.StreamBatch(["P", "P"], options).Where(c => c.Index == 1).Select(c => c.Text).ToList();
            Check(string.Concat(batch) == expected && batch.All(WholeCharacters), $"split [{string.Join(",", widths)}], batch: '{string.Concat(batch)}'");
            splits++;
        }

        Check(splits > 100, $"{splits} splits");
    }

    private static IEnumerable<List<int>> ByteSplits(int length, int at)
    {
        if (at == length)
        {
            yield return [];
            yield break;
        }

        for (int w = 1; w <= 4 && at + w <= length; w++)
        {
            foreach (var rest in ByteSplits(length, at + w))
            {
                rest.Insert(0, w);
                yield return rest;
            }
        }
    }

    private static bool WholeCharacters(string s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(s[i]))
            {
                return false;
            }
        }

        return true;
    }

    // Token i is always followed by token i + 1: an embedding whose row i puts a large logit on i + 1.
    private static Sequential ScriptModel(int vocabulary, Device device)
    {
        var table = new float[vocabulary * vocabulary];
        for (int i = 0; i + 1 < vocabulary; i++)
        {
            table[i * vocabulary + i + 1] = 20f;
        }

        return new Sequential { Embedding.FromWeights(Tensor.From(table, [vocabulary, vocabulary], device)) };
    }

    // Tokens that are raw byte strings (as byte-level BPE tokens are): decoding joins the bytes and reads them as UTF-8.
    private sealed class ByteTokens(List<byte[]> pieces) : ITokenizer
    {
        public int VocabularySize => pieces.Count;

        public IReadOnlyList<int> Encode(string text) => text == "P" ? [0] : throw new ArgumentException(text);

        public string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString([.. ids.SelectMany(i => pieces[i])]);
    }
}
