// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The library's "wordpiece" tokenizer trainer: BERT's WordPiece model learned from the words its own pipeline makes of
/// each line (BERT's normalizer with <see cref="TokenizerTrainingOptions.Lowercase"/> and
/// <see cref="TokenizerTrainingOptions.StripAccents"/>, the Arabic removals, BERT's pre-tokenizer, and digits apart with
/// <see cref="TokenizerTrainingOptions.SplitDigits"/>). Two scorings (<see cref="TokenizerTrainingOptions.Scoring"/>):
/// <list type="bullet">
/// <item>"frequency": Hugging Face's <c>WordPieceTrainer</c> (tokenizers 0.23): byte-pair merges of the most frequent pair
/// with "##" before every character but a word's first, the merged token taking the right part without its "##". Hugging
/// Face numbers the "##" characters in hash order, so its vocabulary changes from run to run; here they follow the alphabet
/// in code-point order and the words are taken in code-point order, so a run is repeatable.</item>
/// <item>"likelihood": the original WordPiece scoring: the pair whose count over the product of its parts' counts is
/// largest (ties: the smaller pair of ids), among pairs seen at least <see cref="TokenizerTrainingOptions.MinFrequency"/>
/// times.</item>
/// </list>
/// The tokenizer is the one BERT's converter writes (with TemplateProcessing when the special tokens hold [CLS] and
/// [SEP]), so Hugging Face reads its tokenizer.json too.
/// </summary>
internal sealed class WordPieceTrainer : ITokenizerTrainer
{
    private const string Prefix = "##";

    public string Name => TokenizerTrainers.WordPiece;

    public string Summary => "BERT's WordPiece, by frequency (Hugging Face's WordPieceTrainer, its ties in a fixed order) or by likelihood";

    public IPretrainedTokenizer Train(IReadOnlyList<string> files, TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate(Name);
        TrainingFiles.Check(files);
        string unknown = options.UnknownToken ?? "[UNK]";
        var specials = options.SpecialTokens.Distinct(StringComparer.Ordinal).ToList();
        if (!specials.Contains(unknown))
        {
            specials.Add(unknown);
        }

        var json = Describe(options, specials, unknown);
        var counts = CountWords(files, options, json, progress, cancellationToken);
        var settings = options with { SpecialTokens = specials };
        var vocabulary = options.Scoring.Equals(TokenizerTrainers.Likelihood, StringComparison.OrdinalIgnoreCase)
            ? Likelihood(counts, settings, progress, cancellationToken)
            : BpeTrainer.Merge(counts, byteLevel: false, settings, progress, cancellationToken, Prefix).Vocabulary;
        if (json["post_processor"] is JsonObject post)
        {
            foreach (var (token, entry) in post["special_tokens"]!.AsObject())
            {
                entry!["ids"] = new JsonArray(vocabulary[token]);
            }
        }

        var definition = new TokenizerDefinition { Source = $"wordpiece trainer ({string.Join(", ", files.Select(Path.GetFileName))})", Json = json, Vocabulary = vocabulary };
        return TokenizerModels.Create(definition, new TokenizerOptions());
    }

    // The tokenizer.json around the vocabulary, as transformers' BertConverter writes it, with the training's choices.
    private static JsonObject Describe(TokenizerTrainingOptions options, List<string> specials, string unknown)
    {
        var added = new JsonArray();
        for (int id = 0; id < specials.Count; id++)
        {
            added.Add((JsonNode)new JsonObject
            {
                ["id"] = id, ["content"] = specials[id], ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false,
                ["special"] = true,
            });
        }

        JsonNode bert = new JsonObject
        {
            ["type"] = "BertNormalizer", ["clean_text"] = true, ["handle_chinese_chars"] = true,
            ["strip_accents"] = options.StripAccents is { } strip ? JsonValue.Create(strip) : null, ["lowercase"] = options.Lowercase,
        };
        var arabic = ArabicMarks.Of(options).ReplaceSteps().Select(step => (JsonNode)step).ToList();
        JsonNode normalizer = arabic.Count == 0 ? bert : new JsonObject { ["type"] = "Sequence", ["normalizers"] = new JsonArray([.. arabic, bert]) };
        JsonNode preTokenizer = new JsonObject { ["type"] = "BertPreTokenizer" };
        if (options.SplitDigits)
        {
            preTokenizer = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(preTokenizer, new JsonObject { ["type"] = "Digits", ["individual_digits"] = true }),
            };
        }

        JsonObject? post = null;
        if (specials.Contains("[CLS]") && specials.Contains("[SEP]"))
        {
            JsonObject Special(string name, int type = 0) => new() { ["SpecialToken"] = new JsonObject { ["id"] = name, ["type_id"] = type } };
            JsonObject Sequence(string name, int type) => new() { ["Sequence"] = new JsonObject { ["id"] = name, ["type_id"] = type } };
            post = new JsonObject
            {
                ["type"] = "TemplateProcessing",
                ["single"] = new JsonArray(Special("[CLS]"), Sequence("A", 0), Special("[SEP]")),
                ["pair"] = new JsonArray(Special("[CLS]"), Sequence("A", 0), Special("[SEP]"), Sequence("B", 1), Special("[SEP]", 1)),
                ["special_tokens"] = new JsonObject
                {
                    ["[CLS]"] = new JsonObject { ["id"] = "[CLS]", ["ids"] = new JsonArray(0), ["tokens"] = new JsonArray("[CLS]") },
                    ["[SEP]"] = new JsonObject { ["id"] = "[SEP]", ["ids"] = new JsonArray(0), ["tokens"] = new JsonArray("[SEP]") },
                },
            };
        }

        return new JsonObject
        {
            ["version"] = "1.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = added,
            ["normalizer"] = normalizer,
            ["pre_tokenizer"] = preTokenizer,
            ["post_processor"] = post,
            ["decoder"] = new JsonObject { ["type"] = "WordPiece", ["prefix"] = Prefix, ["cleanup"] = true },
            ["model"] = new JsonObject
            {
                ["type"] = TokenizerModels.WordPiece, ["unk_token"] = unknown, ["continuing_subword_prefix"] = Prefix, ["max_input_chars_per_word"] = 100,
            },
        };
    }

    // The words the tokenizer's pipeline makes of every line, counted on every worker (each with a pipeline of its own, as a
    // Regex keeps one matcher and workers sharing one would each make a new matcher per match; its own counts for the whole
    // run; its blocks decoded into one buffer of its own), then added up and sorted in code-point order (the order the
    // merges take them in, so the result does not depend on the workers).
    private static Dictionary<string, long> CountWords(IReadOnlyList<string> files, TokenizerTrainingOptions options, JsonObject tokenizer,
        IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
    {
        long total = files.Sum(f => new FileInfo(f).Length), read = 0;
        int workers = options.Workers > 0 ? options.Workers : ComputeResources.MaxCpuThreads;
        bool crlfAsLf = options.LineEnds == "lf";
        var counted = new Dictionary<string, long>?[workers];
        using (var blocks = new BpeTrainer.BlockQueue(files, cancellationToken))
        {
            Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, worker =>
            {
                var pipeline = new TokenizerPipeline(tokenizer);
                var counts = new Dictionary<string, long>(StringComparer.Ordinal);
                var words = counts.GetAlternateLookup<ReadOnlySpan<char>>();
                var pieces = new PreTokenizedText();
                char[] chars = [];
                try
                {
                    while (blocks.Next(out var block))
                    {
                        int length;
                        try
                        {
                            int most = Encoding.UTF8.GetMaxCharCount(block.Length);
                            if (chars.Length < most)
                            {
                                if (chars.Length > 0)
                                {
                                    ArrayPool<char>.Shared.Return(chars);
                                }

                                chars = [];
                                chars = ArrayPool<char>.Shared.Rent(most);
                            }

                            length = Encoding.UTF8.GetChars(block.Bytes, 0, block.Length, chars, 0);
                        }
                        finally
                        {
                            ArrayPool<byte>.Shared.Return(block.Bytes);
                        }

                        CountLines(chars, length, crlfAsLf, pipeline, pieces, words);
                        progress?.Report(new TokenizerTrainingProgress("words", Interlocked.Add(ref read, block.Length), total));
                    }
                }
                finally
                {
                    pieces.Dispose();
                    if (chars.Length > 0)
                    {
                        ArrayPool<char>.Shared.Return(chars);
                    }
                }

                counted[worker] = counts;
            });
        }

        var merged = BpeTrainer.AddUp(counted);
        var sorted = merged.Keys.ToArray();
        Array.Sort(sorted, CodePointOrder.Compare);
        var ordered = new Dictionary<string, long>(sorted.Length, StringComparer.Ordinal);
        foreach (string word in sorted)
        {
            ordered[word] = merged[word];
        }

        return ordered;
    }

    // Counts the words of a block's lines (text[..length], decoded): each line (without its line end; "\r\n" as "\n" when
    // crlfAsLf) through the pipeline, each piece that is not empty one word.
    private static void CountLines(char[] text, int length, bool crlfAsLf, TokenizerPipeline pipeline, PreTokenizedText pieces,
        Dictionary<string, long>.AlternateLookup<ReadOnlySpan<char>> words)
    {
        int start = 0;
        while (start < length)
        {
            int newline = text.AsSpan(start, length - start).IndexOf('\n');
            int end = newline < 0 ? -1 : start + newline;
            int count = (end < 0 ? length : end) - start;
            if (crlfAsLf && end >= 0 && count > 0 && text[end - 1] == '\r')
            {
                count--;
            }

            if (count > 0)
            {
                pipeline.Run(text, length, start, count, atStart: true, pieces);
                for (int p = 0; p < pieces.Pieces.Count; p++)
                {
                    var word = pieces.Piece(p);
                    if (!word.IsEmpty)
                    {
                        words[word] = words.TryGetValue(word, out long seen) ? seen + 1 : 1;
                    }
                }
            }

            start = end < 0 ? length : end + 1;
        }
    }

    // The original WordPiece scoring: merge the pair with the largest count(pair) / (count(left) × count(right)) until the
    // vocabulary is full. Scores change when a part's count changes, so every pair of a merged part is queued again; an
    // entry whose score is no longer current is dropped (a newer one stands for it) or queued again with its lower score.
    private static Dictionary<string, int> Likelihood(Dictionary<string, long> wordCounts, TokenizerTrainingOptions options,
        IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
    {
        var table = BpeTrainer.Symbols(wordCounts, byteLevel: false, options, Prefix);
        var (ids, tokens, words, lengths, counts) = (table.Ids, table.Tokens, table.Words, table.Lengths, table.Counts);
        var symbolCounts = new Dictionary<int, long>();
        var pairCounts = new Dictionary<long, long>();
        var where = new Dictionary<long, HashSet<int>>();
        var pairsOf = new Dictionary<int, HashSet<long>>();
        void Note(long pair, int word)
        {
            if (!where.TryGetValue(pair, out var set))
            {
                where[pair] = set = [];
            }

            set.Add(word);
            foreach (int symbol in new[] { (int)(pair >> 32), (int)(uint)pair })
            {
                if (!pairsOf.TryGetValue(symbol, out var pairs))
                {
                    pairsOf[symbol] = pairs = [];
                }

                pairs.Add(pair);
            }
        }

        for (int i = 0; i < words.Length; i++)
        {
            for (int k = 0; k < lengths[i]; k++)
            {
                CollectionsMarshal.GetValueRefOrAddDefault(symbolCounts, words[i][k], out _) += counts[i];
                if (k + 1 < lengths[i])
                {
                    long pair = BpeTrainer.Pack(words[i][k], words[i][k + 1]);
                    CollectionsMarshal.GetValueRefOrAddDefault(pairCounts, pair, out _) += counts[i];
                    Note(pair, i);
                }
            }
        }

        long floor = Math.Max(1, options.MinFrequency);
        double Score(long pair) => pairCounts.TryGetValue(pair, out long n) && n >= floor
            ? n / ((double)symbolCounts[(int)(pair >> 32)] * symbolCounts[(int)(uint)pair])
            : double.NaN;

        // Highest score first, then the smaller pair.
        var queue = new PriorityQueue<long, (double Score, long Pair)>(Comparer<(double Score, long Pair)>.Create(
            (x, y) => x.Score != y.Score ? y.Score.CompareTo(x.Score) : x.Pair.CompareTo(y.Pair)));
        void Push(long pair)
        {
            double score = Score(pair);
            if (!double.IsNaN(score))
            {
                queue.Enqueue(pair, (score, pair));
            }
        }

        foreach (long pair in pairCounts.Keys)
        {
            Push(pair);
        }

        var changes = new List<(long Pair, int Change)>();
        var touched = new HashSet<long>();
        while (ids.Count < options.VocabularySize && queue.TryDequeue(out long top, out var priority))
        {
            cancellationToken.ThrowIfCancellationRequested();
            double current = Score(top);
            if (double.IsNaN(current) || current > priority.Score)
            {
                continue;                                                        // gone, or a newer entry stands for it
            }

            if (current < priority.Score)
            {
                queue.Enqueue(top, (current, top));
                continue;
            }

            int left = (int)(top >> 32), right = (int)(uint)top;
            string rightPart = tokens[right].StartsWith(Prefix, StringComparison.Ordinal) ? tokens[right][Prefix.Length..] : tokens[right];
            string merged = tokens[left] + rightPart;
            table.Add(merged);
            int mergedId = ids[merged];
            touched.Clear();
            foreach (int i in where[top].ToArray())
            {
                changes.Clear();
                int before = lengths[i];
                lengths[i] = BpeTrainer.MergeWord(words[i], lengths[i], left, right, mergedId, changes);
                long merges = before - lengths[i];
                if (merges == 0)
                {
                    continue;
                }

                CollectionsMarshal.GetValueRefOrAddDefault(symbolCounts, left, out _) -= merges * counts[i];
                CollectionsMarshal.GetValueRefOrAddDefault(symbolCounts, right, out _) -= merges * counts[i];
                CollectionsMarshal.GetValueRefOrAddDefault(symbolCounts, mergedId, out _) += merges * counts[i];
                CollectionsMarshal.GetValueRefOrAddDefault(pairCounts, top, out _) -= merges * counts[i];
                foreach (var (pair, change) in changes)
                {
                    CollectionsMarshal.GetValueRefOrAddDefault(pairCounts, pair, out _) += change * counts[i];
                    if (change > 0)
                    {
                        Note(pair, i);
                    }

                    touched.Add(pair);
                }
            }

            // Every pair of a part whose count changed scores anew.
            foreach (int symbol in new[] { left, right, mergedId })
            {
                if (pairsOf.TryGetValue(symbol, out var pairs))
                {
                    touched.UnionWith(pairs);
                }
            }

            touched.Remove(top);
            foreach (long pair in touched.Order())
            {
                Push(pair);
            }

            progress?.Report(new TokenizerTrainingProgress("merges", ids.Count, options.VocabularySize));
        }

        return ids;
    }
}
