using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Generation;
using Idrak.LanguageModels;
using Idrak.Retrieval;

// --bench-text: the library's text paths (tokenizer, chat template, streamed output parsing, chunking, keyword search),
// timed on the README as input. CPU only; no model or download needed.
internal static partial class Tests
{
    internal static int BenchText()
    {
        string root = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(root, "README.md")) && Directory.GetParent(root) is { } parent)
        {
            root = parent.FullName;
        }

        string text = File.ReadAllText(Path.Combine(root, "README.md")).ReplaceLineEndings("\n");   // the same input on every OS
        Console.WriteLine($"text paths on the README ({text.Length:N0} characters), .NET {Environment.Version}, {Environment.ProcessorCount} threads");

        // Measures `run` (warmed up first): the best of five rounds, each long enough to time.
        static (double Microseconds, long Bytes) Measure(Action run)
        {
            run();
            int repeats = 1;
            var watch = Stopwatch.StartNew();
            run();
            while (watch.ElapsedMilliseconds < 50 && repeats < 1 << 20)
            {
                repeats *= 2;
                watch.Restart();
                for (int i = 0; i < repeats; i++)
                {
                    run();
                }
            }

            double best = double.MaxValue;
            long bytes = long.MaxValue;
            for (int round = 0; round < 5; round++)
            {
                long allocated = GC.GetAllocatedBytesForCurrentThread();
                watch.Restart();
                for (int i = 0; i < repeats; i++)
                {
                    run();
                }

                best = Math.Min(best, watch.Elapsed.TotalMilliseconds * 1000 / repeats);
                bytes = Math.Min(bytes, (GC.GetAllocatedBytesForCurrentThread() - allocated) / repeats);
            }

            return (best, bytes);
        }

        static void Row(string name, (double Microseconds, long Bytes) result, string? rate = null) =>
            Console.WriteLine($"  {name,-52} {result.Microseconds,12:F1} us {result.Bytes / 1024.0,12:F1} KiB allocated  {rate}");

        var tokenizer = TrainedTokenizer(text, merges: 3000);
        IReadOnlyList<int> ids = tokenizer.Encode(text);
        Console.WriteLine($"tokenizer: byte-level BPE, {tokenizer.VocabularySize:N0} tokens trained on the text; {ids.Count:N0} tokens");
        var encode = Measure(() => tokenizer.Encode(text));
        Row("encode the text", encode, $"{text.Length / encode.Microseconds:F1} MB/s");
        int[] idArray = [.. ids];
        var decode = Measure(() => tokenizer.Decode(idArray));
        Row("decode the text", decode, $"{text.Length / decode.Microseconds:F1} MB/s");
        var single = Measure(() =>
        {
            for (int i = 0; i < 1000; i++)
            {
                tokenizer.Decode(idArray.AsSpan(i, 1));
            }
        });
        Row("decode 1,000 single tokens (streaming)", single);
        var shortText = Measure(() => tokenizer.Encode("What is the capital of France? Answer in one word."));
        Row("encode a 50-character prompt", shortText);

        var template = new JinjaChatTemplate(Qwen3Template, ["<|im_end|>"]);
        var paragraphs = text.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        var messages = Enumerable.Range(0, 20).Select(i => new ChatMessage(i % 2 == 0 ? "user" : "assistant", paragraphs[i % paragraphs.Length])).ToList();
        Row("chat template: render 20 messages", Measure(() => template.Render(messages, [], think: false)));

        string reply = "<think>" + paragraphs[3] + "</think>" + paragraphs[5] + "<tool_call>{\"name\": \"search\", \"arguments\": {\"query\": \"idrak\"}}</tool_call>";
        var pieces = tokenizer.Encode(reply).Select(id => tokenizer.Decode([id])).ToArray();
        Row($"parse streamed reply ({pieces.Length} pieces)", Measure(() =>
        {
            var parser = new ChatOutputParser(template);
            foreach (var piece in pieces)
            {
                parser.Feed(piece);
            }

            parser.Finish();
        }));

        var documents = paragraphs.Select((p, i) => new Document($"d{i}", p)).ToList();
        Row("chunk the text (words, 64 with 16 overlap)", Measure(() => Chunker.Split(documents, ChunkUnit.Words, 64, 16)));
        Row("chunk the text (sentences, 4 with 1 overlap)", Measure(() => Chunker.Split(documents, ChunkUnit.Sentences, 4, 1)));
        Row("BM25: index the paragraphs", Measure(() => new Bm25Index(paragraphs)));
        var index = new Bm25Index(paragraphs);
        Row("BM25: 100 searches", Measure(() =>
        {
            for (int i = 0; i < 100; i++)
            {
                index.Search("offload weights to system memory while training on the GPU", 5);
            }
        }));
        return 0;
    }

    // A GPT-2-style byte-level BPE trained on `text` (word counts, most frequent pair merged first), so the timings use a
    // vocabulary shaped like a real one.
    private static BpeTokenizer TrainedTokenizer(string text, int merges)
    {
        const string Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        var byteMap = new string[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            byteMap[b] = ((char)(printable.Contains(b) ? b : 256 + extra++)).ToString();
        }

        var words = new Dictionary<string, int>();
        foreach (Match match in Regex.Matches(text, Pattern))
        {
            string word = string.Concat(Encoding.UTF8.GetBytes(match.Value).Select(b => byteMap[b]));
            words[word] = words.GetValueOrDefault(word) + 1;
        }

        var split = words.ToDictionary(w => w.Key, w => w.Key.Select(c => c.ToString()).ToList());
        var vocab = new JsonObject();
        foreach (var token in byteMap)
        {
            vocab[token] = vocab.Count;
        }

        var mergeList = new JsonArray();
        for (int m = 0; m < merges; m++)
        {
            var pairs = new Dictionary<(string, string), int>();
            foreach (var (word, parts) in split)
            {
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    pairs[(parts[i], parts[i + 1])] = pairs.GetValueOrDefault((parts[i], parts[i + 1])) + words[word];
                }
            }

            if (pairs.Count == 0)
            {
                break;
            }

            var best = pairs.MaxBy(p => p.Value).Key;
            string merged = best.Item1 + best.Item2;
            if (!vocab.ContainsKey(merged))
            {
                vocab[merged] = vocab.Count;
            }

            mergeList.Add($"{best.Item1} {best.Item2}");
            foreach (var parts in split.Values)
            {
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    if (parts[i] == best.Item1 && parts[i + 1] == best.Item2)
                    {
                        parts[i] = merged;
                        parts.RemoveAt(i + 1);
                    }
                }
            }
        }

        return BpeTokenizer.FromJson(new JsonObject
        {
            ["added_tokens"] = new JsonArray(new JsonObject { ["id"] = vocab.Count, ["content"] = "<|im_end|>", ["special"] = true }),
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(
                    new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = Pattern }, ["behavior"] = "Isolated", ["invert"] = false },
                    new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = false }),
            },
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = mergeList },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        });
    }
}
