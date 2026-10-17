// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The library's "bpe" tokenizer trainer: byte-pair encoding learned as Hugging Face's <c>BpeTrainer</c> learns it
/// (tokenizers 0.23, <c>models/bpe/trainer.rs</c>), so the same files give the same vocabulary and the same merges in the
/// same order. Words are counted from the files' bytes in blocks on every processor (each line, with its line end, one
/// sequence, as <c>train_from_files</c> reads); the alphabet is the special tokens, then every character in code-point
/// order; then the most frequent pair (ties: the smaller pair of ids) is merged until the vocabulary is full or no pair
/// is frequent enough. Its bookkeeping is kept as the original's, including a pair queued twice being merged twice
/// (the later rank is the one written).
/// </summary>
internal sealed class BpeTrainer : ITokenizerTrainer
{
    private const string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    private static readonly Regex WordRule = new(Gpt2Pattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private const char MetaspaceMark = '▁';
    private const int BlockBytes = 1 << 20;

    public string Name => TokenizerTrainers.Bpe;

    public string Summary => "byte-pair encoding as Hugging Face's BpeTrainer learns it (the same vocabulary and merges from the same files)";

    public BpeTokenizer Train(IReadOnlyList<string> files, TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(options);
        bool byteLevel = options.PreTokenizer.Equals(TokenizerTrainers.ByteLevel, StringComparison.OrdinalIgnoreCase);
        if (!byteLevel && !options.PreTokenizer.Equals(TokenizerTrainers.Metaspace, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Pre-tokenizer '{options.PreTokenizer}' is not one the bpe trainer writes ({TokenizerTrainers.ByteLevel}, {TokenizerTrainers.Metaspace}).", nameof(options));
        }

        if (options.VocabularySize < 1)
        {
            throw new ArgumentException($"The vocabulary size must be at least 1, not {options.VocabularySize}.", nameof(options));
        }

        if (options.MinFrequency < 0)
        {
            throw new ArgumentException($"The minimum frequency cannot be negative ({options.MinFrequency}).", nameof(options));
        }

        if (options.LineEnds is not ("lf" or "keep"))
        {
            throw new ArgumentException($"LineEnds is \"lf\" (\\r\\n read as \\n) or \"keep\" (the bytes as they are), not '{options.LineEnds}'.", nameof(options));
        }

        if (files.Count == 0)
        {
            throw new ArgumentException("Give at least one text file to learn from.", nameof(files));
        }

        foreach (string file in files)
        {
            if (!File.Exists(file))
            {
                throw new FileNotFoundException($"{file} not found.", file);
            }
        }

        var counts = CountWords(files, byteLevel, options, progress, cancellationToken);
        var (vocabulary, merges) = Merge(counts, byteLevel, options, progress, cancellationToken);
        return BpeTokenizer.FromParts(Describe(byteLevel, options, vocabulary), vocabulary, merges);
    }

    // The words of every file and how often each appears, keyed by the word as the merges see it (byte level: its UTF-8
    // bytes as the byte map's characters).
    private static Dictionary<string, long> CountWords(IReadOnlyList<string> files, bool byteLevel, TokenizerTrainingOptions options,
        IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
    {
        long total = files.Sum(f => new FileInfo(f).Length);
        long read = 0;
        int workers = options.Workers > 0 ? options.Workers : ComputeResources.MaxCpuThreads;
        var merged = new Dictionary<string, long>(StringComparer.Ordinal);
        var parts = new ConcurrentBag<Dictionary<string, long>>();
        Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(Blocks(files, cancellationToken), EnumerablePartitionerOptions.NoBuffering),
            new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken },
            () => new WordCounter(byteLevel, options.AddPrefixSpace, options.LineEnds == "lf"),
            (block, _, counter) =>
            {
                try
                {
                    counter.Count(block.Bytes.AsSpan(0, block.Length));
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(block.Bytes);
                }

                progress?.Report(new TokenizerTrainingProgress("words", Interlocked.Add(ref read, block.Length), total));
                return counter;
            },
            counter =>
            {
                parts.Add(counter.Words);
                counter.Dispose();
            });

        foreach (var part in parts)
        {
            foreach (var (word, count) in part)
            {
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(merged, word, out _) += count;
            }
        }

        if (!byteLevel)
        {
            return merged;
        }

        // Byte level: each distinct word's bytes become characters once (rule 75), not each occurrence.
        var mapped = new Dictionary<string, long>(merged.Count, StringComparer.Ordinal);
        foreach (var (word, count) in merged)
        {
            mapped[BpeTokenizer.ByteLevelText(word)] = count;
        }

        return mapped;
    }

    private readonly record struct Block(byte[] Bytes, int Length);

    // The files' bytes in blocks that end on a line end (a line is never cut), rented from the shared pool; the reader
    // of a block returns it.
    private static IEnumerable<Block> Blocks(IReadOnlyList<string> files, CancellationToken cancellationToken)
    {
        foreach (string file in files)
        {
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
            byte[] carry = [];
            int carried = 0;
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                byte[] buffer = ArrayPool<byte>.Shared.Rent(Math.Max(BlockBytes, carried * 2));
                carry.AsSpan(0, carried).CopyTo(buffer);
                int filled = carried;
                int n;
                while (filled < buffer.Length && (n = stream.Read(buffer, filled, buffer.Length - filled)) > 0)
                {
                    filled += n;
                }

                bool end = filled < buffer.Length;
                int cut = end ? filled : buffer.AsSpan(0, filled).LastIndexOf((byte)'\n') + 1;
                if (cut == 0)
                {
                    // One line longer than the block: keep it and read on with a larger block.
                    carry = buffer.AsSpan(0, filled).ToArray();
                    carried = filled;
                    ArrayPool<byte>.Shared.Return(buffer);
                    continue;
                }

                carried = filled - cut;
                if (carry.Length < carried)
                {
                    carry = new byte[Math.Max(carried, BlockBytes)];
                }

                buffer.AsSpan(cut, carried).CopyTo(carry);
                if (cut > 0)
                {
                    yield return new Block(buffer, cut);
                }
                else
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                }

                if (end)
                {
                    break;
                }
            }
        }
    }

    // Counts one worker's words: a block's bytes become characters once, each line (with its line end) is cut into words
    // by the pre-tokenizer's rule, and each word is counted by its span (no string unless the word is new to this worker).
    private sealed class WordCounter(bool byteLevel, bool prefixSpace, bool crlfAsLf) : IDisposable
    {
        private char[] _chars = [];
        private char[] _line = [];

        public Dictionary<string, long> Words { get; } = new(StringComparer.Ordinal);

        public void Count(ReadOnlySpan<byte> bytes)
        {
            int length = Encoding.UTF8.GetMaxCharCount(bytes.Length);
            if (_chars.Length < length)
            {
                if (_chars.Length > 0)
                {
                    ArrayPool<char>.Shared.Return(_chars);
                }

                _chars = ArrayPool<char>.Shared.Rent(length);
            }

            int count = Encoding.UTF8.GetChars(bytes, _chars);
            Span<char> text = _chars.AsSpan(0, count);
            var words = Words.GetAlternateLookup<ReadOnlySpan<char>>();
            while (!text.IsEmpty)
            {
                int end = text.IndexOf('\n');
                Span<char> line = end < 0 ? text : text[..(end + 1)];
                text = end < 0 ? [] : text[(end + 1)..];
                if (crlfAsLf && line.Length >= 2 && line[^2] == '\r' && line[^1] == '\n')
                {
                    line[^2] = '\n';                                            // "\r\n" read as "\n", in the block's own characters
                    line = line[..^1];
                }

                if (byteLevel)
                {
                    if (prefixSpace && line[0] != ' ')
                    {
                        line = Prefixed(line, ' ');
                    }

                    foreach (var match in WordRule.EnumerateMatches(line))
                    {
                        Add(words, line.Slice(match.Index, match.Length));
                    }
                }
                else
                {
                    CountMetaspace(words, line);
                }
            }
        }

        // Metaspace with prepend "always" and split: spaces become ▁, a ▁ goes before the line unless it starts with one,
        // and each ▁ starts a word (merged with what follows it).
        private void CountMetaspace(Dictionary<string, long>.AlternateLookup<ReadOnlySpan<char>> words, ReadOnlySpan<char> line)
        {
            var marked = line[0] == ' ' || line[0] == MetaspaceMark ? Copy(line, 0) : Prefixed(line, MetaspaceMark);
            System.MemoryExtensions.Replace(marked, ' ', MetaspaceMark);
            int start = 0;
            for (int i = 1; i <= marked.Length; i++)
            {
                if (i == marked.Length || marked[i] == MetaspaceMark)
                {
                    Add(words, marked[start..i]);
                    start = i;
                }
            }
        }

        private Span<char> Prefixed(ReadOnlySpan<char> line, char first)
        {
            var copy = Copy(line, 1);
            copy[0] = first;
            return copy;
        }

        private Span<char> Copy(ReadOnlySpan<char> line, int offset)
        {
            if (_line.Length < line.Length + offset)
            {
                if (_line.Length > 0)
                {
                    ArrayPool<char>.Shared.Return(_line);
                }

                _line = ArrayPool<char>.Shared.Rent(line.Length + offset);
            }

            line.CopyTo(_line.AsSpan(offset));
            return _line.AsSpan(0, line.Length + offset);
        }

        private static void Add(Dictionary<string, long>.AlternateLookup<ReadOnlySpan<char>> words, ReadOnlySpan<char> word)
        {
            if (words.TryGetValue(word, out long seen))
            {
                words[word] = seen + 1;
            }
            else
            {
                words[word] = 1;
            }
        }

        public void Dispose()
        {
            if (_chars.Length > 0)
            {
                ArrayPool<char>.Shared.Return(_chars);
            }

            if (_line.Length > 0)
            {
                ArrayPool<char>.Shared.Return(_line);
            }

            (_chars, _line) = ([], []);
        }
    }

    // One queued merge: a pair (left id << 32 | right id), the words it was seen in when queued.
    private sealed class Candidate(long pair, HashSet<int> words)
    {
        public long Pair { get; } = pair;

        public HashSet<int> Words { get; } = words;
    }

    // The queue's order: the larger count first, then the smaller pair (as the original's Ord on Merge).
    private sealed class Order : IComparer<(ulong Count, long Pair)>
    {
        public static readonly Order Instance = new();

        public int Compare((ulong Count, long Pair) x, (ulong Count, long Pair) y)
            => x.Count != y.Count ? y.Count.CompareTo(x.Count) : x.Pair.CompareTo(y.Pair);
    }

    private static (Dictionary<string, int> Vocabulary, (string Left, string Right)[] Merges) Merge(Dictionary<string, long> wordCounts, bool byteLevel,
        TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
    {
        var ids = new Dictionary<string, int>(Math.Min(options.VocabularySize, 1 << 20), StringComparer.Ordinal);
        var tokens = new List<string>(ids.Capacity);
        void AddToken(string token)
        {
            if (ids.TryAdd(token, tokens.Count))
            {
                tokens.Add(token);
            }
        }

        foreach (string special in options.SpecialTokens)
        {
            AddToken(special);
        }

        // The alphabet: each character's count over the words (the byte map's 256 count as always seen), the rarest
        // dropped past the limit, the rest in code-point order.
        var alphabet = new Dictionary<int, ulong>();
        foreach (var (word, count) in wordCounts)
        {
            foreach (var rune in word.EnumerateRunes())
            {
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(alphabet, rune.Value, out _) += (ulong)count;
            }
        }

        if (byteLevel)
        {
            foreach (char c in BpeTokenizer.ByteLevelAlphabet)
            {
                alphabet[c] = ulong.MaxValue;
            }
        }

        var kept = alphabet.ToList();
        int remove = options.LimitAlphabet is int limit ? Math.Max(0, alphabet.Count - limit) : 0;
        if (remove > 0)
        {
            kept.Sort((a, b) => a.Value != b.Value ? a.Value.CompareTo(b.Value) : a.Key.CompareTo(b.Key));
            kept.RemoveRange(0, remove);
        }

        kept.Sort((a, b) => a.Key.CompareTo(b.Key));
        foreach (var (code, _) in kept)
        {
            AddToken(char.ConvertFromUtf32(code));
        }

        // Every word as symbol ids (characters outside the alphabet are left out, as the original does).
        var words = new int[wordCounts.Count][];
        var lengths = new int[wordCounts.Count];
        var counts = new long[wordCounts.Count];
        int w = 0;
        var symbols = new List<int>();
        var lookup = ids.GetAlternateLookup<ReadOnlySpan<char>>();
        Span<char> runeText = stackalloc char[2];
        foreach (var (word, count) in wordCounts)
        {
            symbols.Clear();
            foreach (var rune in word.EnumerateRunes())
            {
                int n = rune.EncodeToUtf16(runeText);
                if (lookup.TryGetValue(runeText[..n], out int id))
                {
                    symbols.Add(id);
                }
            }

            words[w] = [.. symbols];
            lengths[w] = symbols.Count;
            counts[w] = count;
            w++;
        }

        // Pair counts over every word, and the words each pair is in.
        var pairCounts = new Dictionary<long, long>();
        var where = new Dictionary<long, HashSet<int>>();
        for (int i = 0; i < words.Length; i++)
        {
            var s = words[i];
            for (int k = 0; k + 1 < lengths[i]; k++)
            {
                long pair = Pack(s[k], s[k + 1]);
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(pairCounts, pair, out _) += counts[i];
                if (!where.TryGetValue(pair, out var set))
                {
                    where[pair] = set = [];
                }

                set.Add(i);
            }
        }

        var queue = new PriorityQueue<Candidate, (ulong Count, long Pair)>(pairCounts.Count, Order.Instance);
        void Queue()
        {
            foreach (var (pair, set) in where)
            {
                long count = pairCounts[pair];
                if (count > 0)
                {
                    queue.Enqueue(new Candidate(pair, set), ((ulong)count, pair));
                }
            }

            where.Clear();
        }

        Queue();
        var merges = new List<(long Pair, int Merged)>();
        var changes = new List<(long Pair, int Change)>();
        while (ids.Count < options.VocabularySize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!queue.TryDequeue(out var top, out var priority))
            {
                break;
            }

            ulong current = unchecked((ulong)pairCounts[top.Pair]);
            if (priority.Count != current)
            {
                queue.Enqueue(top, (current, top.Pair));
                continue;
            }

            if (current < 1 || (ulong)options.MinFrequency > current)
            {
                break;
            }

            int left = (int)(top.Pair >> 32), right = (int)(uint)top.Pair;
            string merged = tokens[left] + tokens[right];
            if (!ids.TryGetValue(merged, out int mergedId))
            {
                mergedId = tokens.Count;
                AddToken(merged);
            }

            merges.Add((top.Pair, mergedId));
            foreach (int i in top.Words)
            {
                changes.Clear();
                lengths[i] = MergeWord(words[i], lengths[i], left, right, mergedId, changes);
                foreach (var (pair, change) in changes)
                {
                    System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(pairCounts, pair, out _) += change * counts[i];
                    if (change > 0)
                    {
                        if (!where.TryGetValue(pair, out var set))
                        {
                            where[pair] = set = [];
                        }

                        set.Add(i);
                    }
                }
            }

            Queue();
            progress?.Report(new TokenizerTrainingProgress("merges", ids.Count, options.VocabularySize));
        }

        // A pair merged twice keeps its later rank (the original keeps merges in a map, the last write winning).
        var last = new Dictionary<long, int>(merges.Count);
        for (int r = 0; r < merges.Count; r++)
        {
            last[merges[r].Pair] = r;
        }

        var ordered = new (string, string)[last.Count];
        int at = 0;
        foreach (var (pair, _) in last.OrderBy(p => p.Value))
        {
            ordered[at++] = (tokens[(int)(pair >> 32)], tokens[(int)(uint)pair]);
        }

        return (ids, ordered);
    }

    private static long Pack(int left, int right) => ((long)left << 32) | (uint)right;

    // Merges every (left, right) of a word in place, left to right, recording the pairs that went (-1) and came (+1);
    // returns the new length.
    private static int MergeWord(int[] symbols, int length, int left, int right, int merged, List<(long Pair, int Change)> changes)
    {
        int write = 0, read = 0;
        while (read < length)
        {
            if (symbols[read] == left && read + 1 < length && symbols[read + 1] == right)
            {
                if (write > 0)
                {
                    changes.Add((Pack(symbols[write - 1], left), -1));
                    changes.Add((Pack(symbols[write - 1], merged), 1));
                }

                symbols[write] = merged;
                if (read + 2 < length)
                {
                    changes.Add((Pack(right, symbols[read + 2]), -1));
                    changes.Add((Pack(merged, symbols[read + 2]), 1));
                }

                write++;
                read += 2;
            }
            else
            {
                symbols[write++] = symbols[read++];
            }
        }

        return write;
    }

    // The tokenizer.json around the vocabulary and merges: the added (special) tokens, the pre-tokenizer and decoder of
    // the rule the words were cut by, and a BPE model as Hugging Face writes one.
    private static JsonObject Describe(bool byteLevel, TokenizerTrainingOptions options, Dictionary<string, int> vocabulary)
    {
        var added = new JsonArray();
        foreach (string special in options.SpecialTokens.Distinct(StringComparer.Ordinal))
        {
            added.Add((JsonNode)new JsonObject
            {
                ["id"] = vocabulary[special],
                ["content"] = special,
                ["single_word"] = false,
                ["lstrip"] = false,
                ["rstrip"] = false,
                ["normalized"] = false,
                ["special"] = true,
            });
        }

        JsonObject Rule() => byteLevel
            ? new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = options.AddPrefixSpace, ["trim_offsets"] = true, ["use_regex"] = true }
            : new JsonObject { ["type"] = "Metaspace", ["replacement"] = MetaspaceMark.ToString(), ["prepend_scheme"] = "always", ["split"] = true };

        return new JsonObject
        {
            ["version"] = "1.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = added,
            ["normalizer"] = null,
            ["pre_tokenizer"] = Rule(),
            ["post_processor"] = null,
            ["decoder"] = Rule(),
            ["model"] = new JsonObject
            {
                ["type"] = "BPE",
                ["dropout"] = null,
                ["unk_token"] = null,
                ["continuing_subword_prefix"] = null,
                ["end_of_word_suffix"] = null,
                ["fuse_unk"] = false,
                ["byte_fallback"] = false,
                ["ignore_merges"] = false,
            },
        };
    }
}
