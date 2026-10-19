// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The library's "unigram" tokenizer trainer: SentencePiece's unigram language model learned by SentencePiece's own
/// procedure (sentencepiece 0.2.2, <c>unigram_model_trainer.cc</c> and <c>trainer_interface.cc</c>), in C#:
/// <list type="number">
/// <item>Each line is normalized as SentencePiece normalizes it (its rule, <see cref="TokenizerTrainingOptions.Normalization"/>,
/// with the Arabic removals; extra spaces dropped, a ▁ before the text, or after it with
/// <see cref="TokenizerTrainingOptions.TreatWhitespaceAsSuffix"/>, spaces as ▁), special tokens in it are cut out, and the
/// characters that cover <see cref="TokenizerTrainingOptions.CharacterCoverage"/> of the text are the alphabet (the rest
/// are unknown). Lines over <see cref="TokenizerTrainingOptions.MaxSentenceLength"/> bytes are skipped, when it is set.</item>
/// <item>Seeds: every character, and the substrings found more than once through a suffix array of the whole text
/// (SA-IS), scored by frequency × length, the best <see cref="TokenizerTrainingOptions.SeedSize"/> kept.</item>
/// <item>Rounds of <see cref="TokenizerTrainingOptions.EmIterations"/> EM steps over the distinct words (forward-backward
/// over each word's lattice for the expected counts, then new scores with SentencePiece's digamma correction), then
/// pruning by the loss of likelihood each piece's removal costs, keeping <see cref="TokenizerTrainingOptions.ShrinkingFactor"/>
/// of them, until the vocabulary is within <see cref="TokenizerTrainingOptions.DesiredSizeFactor"/> of its size; then the
/// best pieces and the alphabet fill it.</item>
/// </list>
/// Which substrings may be pieces follows SentencePiece's switches (<see cref="TokenizerTrainingOptions.SplitByWhitespace"/>,
/// <see cref="TokenizerTrainingOptions.SplitByUnicodeScript"/>, <see cref="TokenizerTrainingOptions.SplitByNumber"/>,
/// <see cref="TokenizerTrainingOptions.AllowWhitespaceOnlyPieces"/>, <see cref="TokenizerTrainingOptions.SplitDigits"/>).
/// Scores and sums are in <c>float</c> where SentencePiece keeps them in <c>float</c>. The E step and the pruning's counts
/// are spread over <see cref="TokenizerTrainingOptions.Workers"/> in chunks of words whose sums are added in chunk order, so
/// one worker and many learn the same vocabulary. The tokenizer is the one the sentencepiece reader makes of the
/// <c>tokenizer.model</c> written here (<see cref="ITokenizerCompanionFiles"/>), so the two files give the same ids.
/// </summary>
internal sealed class UnigramTrainer : ITokenizerTrainer
{
    private const char Space = '▁';                 // ▁, SentencePiece's whitespace mark
    private const char UnknownChar = '▅';           // ▅, SentencePiece's mark for a character outside the alphabet
    private const char Boundary = '\t';                  // what special tokens in the text become (never inside a piece)
    private const float RequiredPenaltyStep = 0.0001f;   // keeps the required characters' added scores apart
    private const int ChunkWords = 1024;                 // words per chunk of the E step and the pruning counts
    private const int UnknownNodeId = 0;                 // SentencePiece's trainer gives the unknown node id 0

    public string Name => TokenizerTrainers.Unigram;

    public string Summary => "SentencePiece's unigram language model, learned by its procedure (suffix-array seeds, EM, pruning by loss); writes tokenizer.json and tokenizer.model";

    public IPretrainedTokenizer Train(IReadOnlyList<string> files, TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate(Name);
        TrainingFiles.Check(files);
        int workers = options.Workers > 0 ? options.Workers : ComputeResources.MaxCpuThreads;
        var meta = MetaPieces.Of(options);
        if (meta.Pieces.Count >= options.VocabularySize)
        {
            throw new ArgumentException(
                $"VocabularySize {options.VocabularySize} leaves no room past the {meta.Pieces.Count} special pieces (special tokens, the unknown token{(options.ByteFallback ? " and the 256 byte pieces" : "")}).", nameof(options));
        }

        // The normalization rule (with the Arabic removals) as the charsmap the tokenizer is written with; the identity rule
        // is applied as it always was (the removals, character by character), any other through its charsmap.
        string rule = options.Normalization.ToLowerInvariant();
        var removal = ArabicMarks.Of(options);
        byte[] charsmap = SentencePieceRules.Charsmap(rule, removal);
        var text = new TextRules(removal, rule == TokenizerTrainers.Identity ? null : PrecompiledCharsMap.Read(charsmap), options.TreatWhitespaceAsSuffix,
            options.MaxSentenceLength);
        var corpus = Corpus.Read(files, options, meta, text, workers, progress, cancellationToken);
        if (corpus.Sentences == 0)
        {
            throw new ArgumentException($"The files hold no text to learn from ({string.Join(", ", files)}).", nameof(files));
        }

        var required = corpus.Alphabet(options.CharacterCoverage);
        if (required.Count + meta.Pieces.Count > options.VocabularySize)
        {
            throw new ArgumentException(
                $"VocabularySize {options.VocabularySize} is smaller than the {required.Count} characters CharacterCoverage {options.CharacterCoverage.ToString(CultureInfo.InvariantCulture)} keeps plus the {meta.Pieces.Count} special pieces: raise VocabularySize or lower CharacterCoverage.",
                nameof(options));
        }

        corpus.MarkUnknown(required);
        var rules = new PieceRules(options);
        float unknownPenalty = (float)options.UnknownPenalty, expectedFloor = (float)options.ExpectedCountFloor;
        var model = new Model(Seeds(corpus, required, rules, options.SeedSize, progress, cancellationToken), unknownPenalty);
        var words = corpus.Words(rules);
        corpus = null!;                                                          // the text is not needed past the words

        int desired = (int)(options.VocabularySize * options.DesiredSizeFactor);
        int rounds = EstimateRounds(model.Count, desired, (float)options.ShrinkingFactor);
        int stepsDone = 0, stepsTotal = (rounds + 1) * options.EmIterations;
        int round = 0;
        while (true)
        {
            for (int step = 0; step < options.EmIterations; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var expected = ExpectedCounts(model, words, workers, cancellationToken);
                model = new Model(MStep(model, expected, expectedFloor), unknownPenalty);
                stepsTotal = Math.Max(stepsTotal, ++stepsDone);
                progress?.Report(new TokenizerTrainingProgress("em", stepsDone, stepsTotal));
            }

            if (model.Count <= desired)
            {
                break;
            }

            model = new Model(Prune(model, words, desired, (float)options.ShrinkingFactor, workers, cancellationToken), unknownPenalty);
            progress?.Report(new TokenizerTrainingProgress("prune", ++round, Math.Max(rounds, round)));
        }

        progress?.Report(new TokenizerTrainingProgress("em", stepsDone, stepsDone));
        progress?.Report(new TokenizerTrainingProgress("prune", round, round));
        var final = Finalize(model, required, options.VocabularySize - meta.Pieces.Count);
        return Build(files, options, meta, final, rule, removal, charsmap);
    }

    // How many pruning rounds a vocabulary of seeds takes to reach the desired size (the M step's floor can drop more).
    private static int EstimateRounds(int size, int desired, float shrinking)
    {
        int rounds = 0;
        while (size > desired)
        {
            size = Math.Max(desired, (int)(shrinking * size));
            rounds++;
        }

        return rounds;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The special pieces: the special tokens in their order, the unknown token, the byte pieces.
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class MetaPieces
    {
        public List<string> Pieces { get; } = [];

        public List<SentencePieceType> Types { get; } = [];

        public int Unknown { get; private set; }

        public int Bos { get; private set; } = -1;

        public int Eos { get; private set; } = -1;

        public int Pad { get; private set; } = -1;

        // The texts cut out of the training text (SentencePiece cuts every meta piece), longest first per first character.
        public Dictionary<char, string[]> ByFirst { get; } = [];

        public static MetaPieces Of(TokenizerTrainingOptions options)
        {
            var meta = new MetaPieces();
            string unknown = options.UnknownToken ?? "<unk>";
            foreach (string token in options.SpecialTokens.Distinct(StringComparer.Ordinal))
            {
                meta.Add(token, token == unknown ? SentencePieceType.Unknown : SentencePieceType.Control);
            }

            if (!meta.Pieces.Contains(unknown))
            {
                meta.Add(unknown, SentencePieceType.Unknown);
            }

            meta.Unknown = meta.Pieces.IndexOf(unknown);
            meta.Bos = meta.Pieces.IndexOf("<s>");
            meta.Eos = meta.Pieces.IndexOf("</s>");
            meta.Pad = meta.Pieces.IndexOf("<pad>");
            if (options.ByteFallback)
            {
                for (int b = 0; b < 256; b++)
                {
                    string piece = string.Create(CultureInfo.InvariantCulture, $"<0x{b:X2}>");
                    if (!meta.Pieces.Contains(piece))
                    {
                        meta.Add(piece, SentencePieceType.Byte);
                    }
                }
            }

            foreach (var group in meta.Pieces.GroupBy(p => p[0]))
            {
                meta.ByFirst[group.Key] = [.. group.OrderByDescending(p => p.Length)];
            }

            return meta;
        }

        private void Add(string piece, SentencePieceType type)
        {
            Pieces.Add(piece);
            Types.Add(type);
        }
    }

    // How a line becomes a sentence: the Arabic removals, the rule's charsmap (null: the identity rule, applied by hand),
    // the whitespace as a suffix, and the longest line read (in UTF-8 bytes; 0: no limit).
    private sealed class TextRules(ArabicMarks removal, PrecompiledCharsMap? map, bool suffix, int maxBytes)
    {
        public ArabicMarks Removal => removal;

        public PrecompiledCharsMap? Map => map;

        public bool Suffix => suffix;

        // A UTF-16 unit is at most 3 bytes, so only lines over a third of the limit are measured.
        public bool TooLong(ReadOnlySpan<char> raw) => maxBytes > 0 && raw.Length * 3L > maxBytes && Encoding.UTF8.GetByteCount(raw) > maxBytes;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The text: every line normalized, one string of sentences.
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class Corpus
    {
        private char[] _text = [];
        private int[] _starts = [];                     // sentence i is _text[_starts[i] .. _starts[i + 1])
        private Dictionary<int, long> _counts = [];     // characters (code points) and how often each appears

        public int Sentences => _starts.Length - 1;

        public ReadOnlySpan<char> Sentence(int i) => _text.AsSpan(_starts[i], _starts[i + 1] - _starts[i]);

        public static Corpus Read(IReadOnlyList<string> files, TokenizerTrainingOptions options, MetaPieces meta, TextRules rules, int workers,
            IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
        {
            long total = files.Sum(f => new FileInfo(f).Length), read = 0;
            bool crlfAsLf = options.LineEnds == "lf";
            var parts = new ConcurrentDictionary<int, Part>();
            var blocks = BpeTrainer.Blocks(files, cancellationToken).Select((block, index) => (block, index));
            Parallel.ForEach(
                Partitioner.Create(blocks, EnumerablePartitionerOptions.NoBuffering),
                new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken },
                item =>
                {
                    try
                    {
                        parts[item.index] = Part.Read(item.block.Bytes.AsSpan(0, item.block.Length), rules, crlfAsLf, meta);
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(item.block.Bytes);
                    }

                    progress?.Report(new TokenizerTrainingProgress("words", Interlocked.Add(ref read, item.block.Length), total));
                });

            // The blocks' sentences in file order, as one text.
            var ordered = parts.OrderBy(p => p.Key).Select(p => p.Value).ToList();
            var corpus = new Corpus
            {
                _text = new char[ordered.Sum(p => p.Text.Count)],
                _starts = new int[ordered.Sum(p => p.Starts.Count) + 1],
            };
            int at = 0, sentence = 0;
            foreach (var part in ordered)
            {
                foreach (int start in part.Starts)
                {
                    corpus._starts[sentence++] = at + start;
                }

                CollectionsMarshal.AsSpan(part.Text).CopyTo(corpus._text.AsSpan(at));
                at += part.Text.Count;
                foreach (var (c, n) in part.Counts)
                {
                    CollectionsMarshal.GetValueRefOrAddDefault(corpus._counts, c, out _) += n;
                }
            }

            corpus._starts[sentence] = at;
            return corpus;
        }

        // One block's normalized sentences and character counts.
        private sealed class Part
        {
            public List<char> Text { get; } = [];

            public List<int> Starts { get; } = [];

            public Dictionary<int, long> Counts { get; } = [];

            public static Part Read(ReadOnlySpan<byte> bytes, TextRules rules, bool crlfAsLf, MetaPieces meta)
            {
                var part = new Part();
                char[] chars = ArrayPool<char>.Shared.Rent(Encoding.UTF8.GetMaxCharCount(bytes.Length));
                char[] line = ArrayPool<char>.Shared.Rent(64);
                byte[]? input = null, output = null;                            // a charsmap's UTF-8, in and out
                try
                {
                    int count = Encoding.UTF8.GetChars(bytes, chars);
                    var text = chars.AsSpan(0, count);
                    while (!text.IsEmpty)
                    {
                        int end = text.IndexOf('\n');
                        var raw = end < 0 ? text : text[..end];
                        text = end < 0 ? [] : text[(end + 1)..];
                        if (crlfAsLf && end >= 0 && raw.Length > 0 && raw[^1] == '\r')
                        {
                            raw = raw[..^1];
                        }

                        if (raw.IsEmpty || raw.Contains(UnknownChar) || rules.TooLong(raw))
                        {
                            continue;                                            // SentencePiece skips empty lines, lines holding its unknown mark and long lines
                        }

                        int length;
                        if (rules.Map is null)
                        {
                            if (line.Length < (raw.Length * 2) + 2)
                            {
                                ArrayPool<char>.Shared.Return(line);
                                line = ArrayPool<char>.Shared.Rent((raw.Length * 2) + 2);
                            }

                            length = Normalize(raw, rules, line);
                        }
                        else
                        {
                            length = NormalizeMapped(raw, rules, ref line, ref input, ref output);
                        }

                        length = CutMeta(line, length, meta);
                        if (length > 0)
                        {
                            part.Starts.Add(part.Text.Count);
                            var sentence = line.AsSpan(0, length);
                            part.Text.AddRange(sentence);
                            for (int i = 0; i < sentence.Length; i++)
                            {
                                int c = CodePoint(sentence, i, out int width);
                                i += width - 1;
                                if (c != 0 && c != ' ' && !(c is >= 0xD800 and <= 0xDFFF))
                                {
                                    CollectionsMarshal.GetValueRefOrAddDefault(part.Counts, c, out _)++;
                                }
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<char>.Shared.Return(chars);
                    ArrayPool<char>.Shared.Return(line);
                    if (input is not null)
                    {
                        ArrayPool<byte>.Shared.Return(input);
                    }

                    if (output is not null)
                    {
                        ArrayPool<byte>.Shared.Return(output);
                    }
                }

                return part;
            }
        }

        // SentencePiece's normalization with the identity map (normalizer.cc): the Arabic removals, then leading spaces
        // dropped, runs of spaces made one, a ▁ before the text, spaces as ▁, trailing ▁ dropped, and a ▁ after the text
        // instead of before it when the whitespace is a suffix (unless the line is only spaces). Output has room for twice
        // the line and two. Returns the length written to output.
        private static int Normalize(ReadOnlySpan<char> raw, TextRules rules, Span<char> output)
        {
            var removal = rules.Removal;
            int written = rules.Suffix ? 0 : 1;
            output[0] = Space;
            bool previousSpace = true, any = false;
            foreach (char c in raw)
            {
                any |= c != ' ';
                if (removal.Removes(c))
                {
                    continue;
                }

                if (c == ' ')
                {
                    if (!previousSpace)
                    {
                        output[written++] = Space;
                        previousSpace = true;
                    }

                    continue;
                }

                output[written++] = c;
                previousSpace = false;
            }

            while (written > 0 && output[written - 1] == Space)
            {
                written--;
            }

            if (rules.Suffix && any)
            {
                output[written++] = Space;
            }

            return written;
        }

        // SentencePiece's Normalizer::Normalize with a charsmap (normalizer.cc), on UTF-8: at each place the longest key's
        // replacement (or one character as it is); replacements that are one space are dropped at the start; a
        // replacement's leading spaces are dropped after a space; spaces become ▁; then trailing ▁ dropped; the dummy ▁
        // before the text (or after it, as a suffix). The result goes to line (grown from the pool); returns its length.
        private static int NormalizeMapped(ReadOnlySpan<char> raw, TextRules rules, ref char[] line, ref byte[]? input, ref byte[]? output)
        {
            var map = rules.Map!;
            int count = Encoding.UTF8.GetByteCount(raw);
            Grow(ref input, count, 0);
            Encoding.UTF8.GetBytes(raw, input);
            var text = input.AsSpan(0, count);
            int at = 0;
            while (at < count)
            {
                int used = Chunk(map, text[at..], out var chunk);
                if (chunk.Length != 1 || chunk[0] != (byte)' ')
                {
                    break;
                }

                at += used;
            }

            if (at == count)
            {
                return 0;                                                        // only spaces
            }

            ReadOnlySpan<byte> space = "\u2581"u8;
            int written = 0;
            Grow(ref output, count + space.Length, 0);
            if (!rules.Suffix)
            {
                space.CopyTo(output);
                written = space.Length;
            }

            bool previousSpace = true;
            while (at < count)
            {
                at += Chunk(map, text[at..], out var chunk);
                while (previousSpace && !chunk.IsEmpty && chunk[0] == (byte)' ')
                {
                    chunk = chunk[1..];
                }

                if (chunk.IsEmpty)
                {
                    continue;
                }

                Grow(ref output, written + (chunk.Length * space.Length) + space.Length, written);
                foreach (byte b in chunk)
                {
                    if (b == (byte)' ')
                    {
                        space.CopyTo(output.AsSpan(written));
                        written += space.Length;
                    }
                    else
                    {
                        output[written++] = b;
                    }
                }

                previousSpace = chunk[^1] == (byte)' ';
            }

            while (written >= space.Length && output.AsSpan(written - space.Length, space.Length).SequenceEqual(space))
            {
                written -= space.Length;
            }

            if (rules.Suffix)
            {
                space.CopyTo(output.AsSpan(written));
                written += space.Length;
            }

            int chars = Encoding.UTF8.GetCharCount(output.AsSpan(0, written));
            if (line.Length < chars)
            {
                ArrayPool<char>.Shared.Return(line);
                line = ArrayPool<char>.Shared.Rent(chars);
            }

            return Encoding.UTF8.GetChars(output.AsSpan(0, written), line);

            // The longest key at the start of rest and its replacement, or one character as it is.
            static int Chunk(PrecompiledCharsMap map, ReadOnlySpan<byte> rest, out ReadOnlySpan<byte> chunk)
            {
                int length = map.Prefix(rest, out chunk);
                if (length > 0)
                {
                    return length;
                }

                byte lead = rest[0];
                length = Math.Min(rest.Length, lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4);
                chunk = rest[..length];
                return length;
            }

            // A pooled buffer of at least size bytes, its first keep bytes kept.
            static void Grow([System.Diagnostics.CodeAnalysis.NotNull] ref byte[]? buffer, int size, int keep)
            {
                if (buffer is not null && buffer.Length >= size)
                {
                    return;
                }

                var larger = ArrayPool<byte>.Shared.Rent(Math.Max(size, (buffer?.Length ?? 0) * 2));
                if (buffer is not null)
                {
                    buffer.AsSpan(0, keep).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(buffer);
                }

                buffer = larger;
            }
        }

        // Each meta piece in the normalized text replaced by a tab, in place (trainer_interface.cc). Returns the new length.
        private static int CutMeta(Span<char> output, int written, MetaPieces meta)
        {
            if (written == 0 || meta.ByFirst.Count == 0)
            {
                return written;
            }

            // Meta pieces become tabs, in place (each is at least one character, so the text only shrinks).
            int read = 0, kept = 0;
            while (read < written)
            {
                string? found = null;
                if (meta.ByFirst.TryGetValue(output[read], out var candidates))
                {
                    var rest = output[read..written];
                    foreach (string piece in candidates)
                    {
                        if (rest.StartsWith(piece, StringComparison.Ordinal))
                        {
                            found = piece;
                            break;
                        }
                    }
                }

                if (found is null)
                {
                    output[kept++] = output[read++];
                }
                else
                {
                    output[kept++] = Boundary;
                    read += found.Length;
                }
            }

            return kept;
        }

        // The required characters (SentencePiece's required_chars_): the most frequent until their share reaches the
        // coverage (ties: the smaller code point first); a tab counts toward the share but is never one.
        public Dictionary<int, long> Alphabet(double coverage)
        {
            long all = _counts.Values.Sum();
            float wanted = (float)coverage;
            var required = new Dictionary<int, long>();
            long accumulated = 0;
            foreach (var (c, n) in _counts.OrderByDescending(p => p.Value).ThenBy(p => p.Key))
            {
                float covered = (float)(1.0 * accumulated / all);
                if (covered >= wanted)
                {
                    break;
                }

                accumulated += n;
                if (c != Boundary)
                {
                    required[c] = n;
                }
            }

            return required;
        }

        // Characters outside the alphabet become ▅ (one UTF-16 unit, so a pair of surrogates shrinks the text).
        public void MarkUnknown(Dictionary<int, long> required)
        {
            int write = 0;
            var starts = _starts;
            for (int s = 0; s < Sentences; s++)
            {
                int from = starts[s], to = starts[s + 1];
                starts[s] = write;
                for (int i = from; i < to; i++)
                {
                    int c = CodePoint(_text, i, to, out int width);
                    if (required.ContainsKey(c))
                    {
                        for (int k = 0; k < width; k++)
                        {
                            _text[write++] = _text[i + k];
                        }
                    }
                    else
                    {
                        _text[write++] = UnknownChar;
                    }

                    i += width - 1;
                }
            }

            starts[Sentences] = write;
            Array.Resize(ref _text, write);
            _counts = [];
        }

        // The text as ranked code points, each sentence followed by 0 (SentencePiece's array for its suffix array), with the
        // code point of each rank (rank 0 is the sentence boundary; the order of code points is kept, so the suffix array is
        // the one of the code points themselves).
        public (int[] Text, int[] CodePoints) Ranked(Dictionary<int, long> required)
        {
            var codes = new int[required.Count + 2];
            int r = 1;
            foreach (int c in required.Keys.Append(UnknownChar).Distinct().Order())
            {
                codes[r++] = c;
            }

            Array.Resize(ref codes, r);
            var bmp = new int[0x10000];
            var supplementary = new Dictionary<int, int>();
            for (int i = 1; i < codes.Length; i++)
            {
                if (codes[i] < 0x10000)
                {
                    bmp[codes[i]] = i;
                }
                else
                {
                    supplementary[codes[i]] = i;
                }
            }

            int length = Sentences;
            for (int i = 0; i < _text.Length; i++)
            {
                length += char.IsHighSurrogate(_text[i]) && i + 1 < _text.Length && char.IsLowSurrogate(_text[i + 1]) ? 0 : 1;
            }

            var ranked = new int[length];
            int at = 0;
            for (int s = 0; s < Sentences; s++)
            {
                var sentence = Sentence(s);
                for (int i = 0; i < sentence.Length; i++)
                {
                    int c = CodePoint(sentence, i, out int width);
                    ranked[at++] = c < 0x10000 ? bmp[c] : supplementary[c];
                    i += width - 1;
                }

                ranked[at++] = 0;
            }

            return (ranked, codes);
        }

        // The words SentencePiece learns from: each sentence cut into words (SplitIntoWords), or whole without
        // SplitByWhitespace, counted, sorted by count (then text).
        public Words Words(PieceRules rules)
        {
            var counts = new Dictionary<string, long>(StringComparer.Ordinal);
            var lookup = counts.GetAlternateLookup<ReadOnlySpan<char>>();
            for (int s = 0; s < Sentences; s++)
            {
                var sentence = Sentence(s);
                int start = 0;
                for (int i = 1; i <= sentence.Length; i++)
                {
                    if (i == sentence.Length || (rules.SplitByWhitespace && rules.WordStarts(sentence, i, Space)))
                    {
                        var word = sentence[start..i];
                        lookup[word] = lookup.TryGetValue(word, out long seen) ? seen + 1 : 1;
                        start = i;
                    }
                }
            }

            var sorted = counts.ToArray();
            Array.Sort(sorted, (a, b) => a.Value != b.Value ? b.Value.CompareTo(a.Value) : CodePointOrder.Compare(a.Key, b.Key));
            return new Words(sorted);
        }
    }

    // The distinct words, in UTF-8, with their counts.
    private sealed class Words
    {
        public Words(KeyValuePair<string, long>[] words)
        {
            Starts = new int[words.Length + 1];
            Counts = new long[words.Length];
            int bytes = 0;
            foreach (var (word, _) in words)
            {
                bytes += Encoding.UTF8.GetByteCount(word);
            }

            Bytes = new byte[bytes];
            int at = 0;
            for (int i = 0; i < words.Length; i++)
            {
                Starts[i] = at;
                at += Encoding.UTF8.GetBytes(words[i].Key, Bytes.AsSpan(at));
                Counts[i] = words[i].Value;
            }

            Starts[words.Length] = at;
        }

        public byte[] Bytes { get; }

        public int[] Starts { get; }

        public long[] Counts { get; }

        public int Count => Counts.Length;

        public ReadOnlySpan<byte> this[int i] => Bytes.AsSpan(Starts[i], Starts[i + 1] - Starts[i]);
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The pieces: what SentencePiece's IsValidSentencePiece allows.
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class PieceRules(TokenizerTrainingOptions options)
    {
        private const byte Any = byte.MaxValue;
        private readonly int _maxLength = options.MaxPieceLength;
        private readonly bool _splitDigits = options.SplitDigits, _splitByScript = options.SplitByUnicodeScript, _splitByNumber = options.SplitByNumber;
        private readonly bool _suffix = options.TreatWhitespaceAsSuffix, _spaceOnly = options.AllowWhitespaceOnlyPieces;

        public bool SplitByWhitespace { get; } = options.SplitByWhitespace;

        // SentencePiece's SplitIntoWords: whether a word starts at i (0 < i < text.Length). Before each ▁, or after it as a
        // suffix; with whitespace-only pieces a run of ▁ stays whole (it starts the next word, or ends the last).
        public bool WordStarts<T>(ReadOnlySpan<T> text, int i, T space)
            where T : IEquatable<T> =>
            _suffix
                ? text[i - 1].Equals(space) && (!_spaceOnly || !text[i].Equals(space))
                : text[i].Equals(space) && (!_spaceOnly || !text[i - 1].Equals(space));

        // SentencePiece's IsValidSentencePiece; piece: ranks into codes (the code point of each rank).
        public bool Valid(ReadOnlySpan<int> piece, int[] codes)
        {
            if (piece.IsEmpty || piece.Length > _maxLength)
            {
                return false;
            }

            bool allSpace = true;
            foreach (int rank in piece)
            {
                allSpace &= codes[rank] == Space;
            }

            byte previous = Any;
            for (int pos = 0; pos < piece.Length; pos++)
            {
                int c = codes[piece[pos]];
                if (c is UnknownChar or 0 or Boundary or ' ' || c is >= 0xD800 and <= 0xDFFF || c > 0x10FFFF)
                {
                    return false;
                }

                if (c == Space)
                {
                    // A ▁ starts a piece (ends it, as a suffix), or sits inside it without the whitespace split; a piece
                    // of ▁ alone is any length with whitespace-only pieces.
                    if (!_spaceOnly || !allSpace)
                    {
                        int last = piece.Length - 1;
                        bool refused = _suffix
                            ? (SplitByWhitespace && pos < last) || (!SplitByWhitespace && pos < last && pos == 0)
                            : (SplitByWhitespace && pos > 0) || (!SplitByWhitespace && pos > 0 && pos == last);
                        if (refused)
                        {
                            return false;
                        }
                    }

                    continue;
                }

                byte script = UnicodeScripts.Of(c);
                if (script is UnicodeScripts.Hiragana or UnicodeScripts.Katakana || c == 0x30FC)
                {
                    script = UnicodeScripts.Han;
                }
                else if (script == UnicodeScripts.Inherited)
                {
                    script = previous;
                }

                if (!_splitByNumber && IsDigit(c))
                {
                    script = Any;
                }

                if (_splitDigits && IsDigit(c) && piece.Length > 1)
                {
                    return false;
                }

                if (_splitByScript && script != Any && previous != Any && previous != script)
                {
                    return false;
                }

                previous = script;
            }

            return true;
        }

        private static bool IsDigit(int c) => c is (>= '0' and <= '9') or (>= 0xFF10 and <= 0xFF19);
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Seeds.
    // ---------------------------------------------------------------------------------------------------------------------
    private static List<(string Piece, float Score)> Seeds(Corpus corpus, Dictionary<int, long> required, PieceRules rules, int seedSize,
        IProgress<TokenizerTrainingProgress>? progress, CancellationToken cancellationToken)
    {
        // Every character of the alphabet (SentencePiece's all_chars: the text's characters but ▅), most frequent first.
        var seeds = new List<(string, float)>();
        foreach (var (c, times) in required.OrderByDescending(p => p.Value).ThenBy(p => p.Key))
        {
            seeds.Add((char.ConvertFromUtf32(c), times));
        }

        var (text, codes) = corpus.Ranked(required);
        int n = text.Length;
        var sa = new int[n];
        SuffixArray.Build(text, n, sa, codes.Length);
        var left = new int[n];
        var right = new int[n];
        var depth = new int[n];
        int nodes = SuffixArray.InternalNodes(text, n, sa, left, right, depth);
        progress?.Report(new TokenizerTrainingProgress("seeds", 0, nodes));

        var queue = new BoundedQueue(seedSize, text, codes);
        int spaceRank = Array.BinarySearch(codes, 1, codes.Length - 1, Space);
        for (int node = 0; node < nodes; node++)
        {
            if ((node & 0xFFFF) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progress?.Report(new TokenizerTrainingProgress("seeds", node, nodes));
            }

            int offset = sa[left[node]], length = depth[node];
            if (length <= 1 || offset >= n || offset + length >= n)
            {
                continue;
            }

            long frequency = right[node] - left[node];
            int begin = offset, end = offset + length;
            while (begin < end)
            {
                int delimiter = Array.IndexOf(text, 0, begin, end - begin);
                if (delimiter < 0)
                {
                    delimiter = end;
                }

                int from = begin;
                begin = delimiter + 1;
                if (delimiter - from <= 1)
                {
                    continue;
                }

                // Cut into words (SplitIntoWords), or whole without SplitByWhitespace, then each piece checked.
                int start = from;
                ReadOnlySpan<int> segment = text.AsSpan(from, delimiter - from);
                for (int i = from + 1; i <= delimiter; i++)
                {
                    if (i == delimiter || (rules.SplitByWhitespace && rules.WordStarts(segment, i - from, spaceRank)))
                    {
                        int pieceLength = i - start;
                        if (pieceLength > 1 && rules.Valid(text.AsSpan(start, pieceLength), codes))
                        {
                            queue.Push(start, pieceLength, frequency * pieceLength);
                        }

                        start = i;
                    }
                }
            }
        }

        progress?.Report(new TokenizerTrainingProgress("seeds", nodes, nodes));
        foreach (var (piece, score) in queue.Get())
        {
            seeds.Add((piece, score));
        }

        // Scores as log probabilities (SentencePiece's ToLogProb: the sum in float).
        float sum = 0;
        foreach (var (_, score) in seeds)
        {
            sum += score;
        }

        float logSum = (float)Math.Log(sum);
        for (int i = 0; i < seeds.Count; i++)
        {
            seeds[i] = (seeds[i].Item1, (float)(Math.Log(seeds[i].Item2) - logSum));
        }

        return seeds;
    }

    // SentencePiece's BoundedPriorityQueue: the best score of each substring, the best `capacity` kept (by score, then
    // UTF-8 length, both descending, then the text), pruned to that whenever it holds more than eight times as many.
    // The substrings are keyed by their place in the ranked text (ranks keep the code points' order), so none is made into
    // a string until it is kept.
    private sealed class BoundedQueue(int capacity, int[] ranks, int[] codes)
    {
        private readonly Dictionary<(int Start, int Length), long> _scores = new(new TextComparer(ranks));

        public void Push(int start, int length, long score)
        {
            ref long current = ref CollectionsMarshal.GetValueRefOrAddDefault(_scores, (start, length), out _);
            if (score > current)
            {
                current = score;
            }

            if (_scores.Count > 8L * capacity)
            {
                var kept = Sorted();
                _scores.Clear();
                foreach (var entry in kept.Take(capacity))
                {
                    _scores.Add(entry.Key, entry.Value);
                }
            }
        }

        public IEnumerable<(string Piece, float Score)> Get()
        {
            foreach (var entry in Sorted().Take(capacity))
            {
                yield return (Text(entry.Key), entry.Value);
            }
        }

        private string Text((int Start, int Length) key)
        {
            var builder = new StringBuilder(key.Length);
            foreach (int rank in ranks.AsSpan(key.Start, key.Length))
            {
                builder.Append(char.ConvertFromUtf32(codes[rank]));
            }

            return builder.ToString();
        }

        private KeyValuePair<(int Start, int Length), long>[] Sorted()
        {
            var entries = _scores.ToArray();
            var bytes = new int[entries.Length];
            for (int i = 0; i < entries.Length; i++)
            {
                foreach (int rank in ranks.AsSpan(entries[i].Key.Start, entries[i].Key.Length))
                {
                    int c = codes[rank];
                    bytes[i] += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
                }
            }

            var order = Enumerable.Range(0, entries.Length).ToArray();
            Array.Sort(order, (a, b) =>
            {
                int c = entries[b].Value.CompareTo(entries[a].Value);
                c = c != 0 ? c : bytes[b].CompareTo(bytes[a]);
                return c != 0 ? c : ranks.AsSpan(entries[a].Key.Start, entries[a].Key.Length).SequenceCompareTo(ranks.AsSpan(entries[b].Key.Start, entries[b].Key.Length));
            });
            return [.. order.Select(i => entries[i])];
        }

        private sealed class TextComparer(int[] ranks) : IEqualityComparer<(int Start, int Length)>
        {
            public bool Equals((int Start, int Length) x, (int Start, int Length) y) =>
                x.Length == y.Length && ranks.AsSpan(x.Start, x.Length).SequenceEqual(ranks.AsSpan(y.Start, y.Length));

            public int GetHashCode((int Start, int Length) key)
            {
                var hash = new HashCode();
                hash.AddBytes(MemoryMarshal.AsBytes(ranks.AsSpan(key.Start, key.Length)));
                return hash.ToHashCode();
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The model being trained: pieces, scores and their trie.
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class Model
    {
        public Model(List<(string Piece, float Score)> pieces, float unknownPenalty)
        {
            Pieces = [.. pieces.Select(p => p.Piece)];
            Scores = [.. pieces.Select(p => p.Score)];
            MinScore = float.MaxValue;
            foreach (float score in Scores)
            {
                MinScore = Math.Min(MinScore, score);
            }

            UnknownScore = MinScore - unknownPenalty;
            Trie = DoubleArrayTrie.Build(Pieces, [.. Enumerable.Range(0, Pieces.Length)]);
        }

        public string[] Pieces { get; }

        public float[] Scores { get; }

        public float MinScore { get; }

        // The unknown node's score: the lowest score minus the unknown penalty.
        public float UnknownScore { get; }

        public DoubleArrayTrie Trie { get; }

        public int Count => Pieces.Length;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The lattice of one word (SentencePiece's Lattice with the trainer model's nodes).
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class Lattice
    {
        private int[] _charStart = new int[64];      // char → byte offset
        private int[] _charOf = new int[256];        // byte offset → char (at a character's start)
        private TrieMatch[] _matches = new TrieMatch[64];
        private int[] _endCount = new int[64];
        private float[] _alpha = new float[256], _beta = new float[256], _best = new float[256], _second = new float[256];
        private int[] _prev = new int[256], _secondPrev = new int[256];
        private byte[] _prevRank = new byte[256], _secondRank = new byte[256];

        public int Chars;
        public int Count;
        public int[] Begin = new int[256], End = new int[256], Id = new int[256];
        public float[] Score = new float[256];
        public int[] BeginFirst = new int[64];     // char → first node beginning there (nodes are in begin order)
        public int[] EndOrder = new int[256];      // nodes by end, each end's in insertion order
        public int[] EndFirst = new int[64];       // char → first entry of EndOrder ending there

        // Builds the nodes: every piece the trie finds at each character, and the unknown node where no single character is.
        public void Set(ReadOnlySpan<byte> word, Model model)
        {
            int bytes = word.Length;
            Grow(ref _charOf, bytes + 1);
            Grow(ref _charStart, bytes + 1);
            int chars = 0;
            for (int b = 0; b < bytes; b++)
            {
                if ((word[b] & 0xC0) != 0x80)
                {
                    _charOf[b] = chars;
                    _charStart[chars++] = b;
                }
            }

            _charOf[bytes] = chars;
            _charStart[chars] = bytes;
            Chars = chars;
            Count = 0;
            Grow(ref BeginFirst, chars + 2);
            float unknownScore = model.UnknownScore;
            for (int c = 0; c < chars; c++)
            {
                BeginFirst[c] = Count;
                var rest = word[_charStart[c]..];
                int found = model.Trie.CommonPrefixSearch(rest, _matches);
                if (found > _matches.Length)
                {
                    _matches = new TrieMatch[found * 2];
                    found = model.Trie.CommonPrefixSearch(rest, _matches);
                }

                bool single = false;
                for (int k = 0; k < found; k++)
                {
                    int end = _charOf[_charStart[c] + _matches[k].Length];
                    Add(c, end, _matches[k].Value, model.Scores[_matches[k].Value]);
                    single |= end - c == 1;
                }

                if (!single)
                {
                    Add(c, c + 1, UnknownNodeId, unknownScore);
                }
            }

            BeginFirst[chars] = Count;

            // Nodes by end position, stable: EndFirst[p] .. EndFirst[p + 1] are the nodes ending at p.
            Grow(ref _endCount, chars + 2);
            Grow(ref EndFirst, chars + 2);
            Array.Clear(_endCount, 0, chars + 2);
            for (int i = 0; i < Count; i++)
            {
                _endCount[End[i] + 1]++;
            }

            for (int p = 1; p <= chars + 1; p++)
            {
                _endCount[p] += _endCount[p - 1];
            }

            Array.Copy(_endCount, EndFirst, chars + 2);
            Grow(ref EndOrder, Count);
            for (int i = 0; i < Count; i++)
            {
                EndOrder[_endCount[End[i]]++] = i;
            }
        }

        private void Add(int begin, int end, int id, float score)
        {
            if (Count == Begin.Length)
            {
                int size = Count * 2;
                Array.Resize(ref Begin, size);
                Array.Resize(ref End, size);
                Array.Resize(ref Id, size);
                Array.Resize(ref Score, size);
            }

            Begin[Count] = begin;
            End[Count] = end;
            Id[Count] = id;
            Score[Count] = score;
            Count++;
        }

        private static void Grow(ref int[] array, int size)
        {
            if (array.Length < size)
            {
                array = new int[Math.Max(size, array.Length * 2)];
            }
        }

        // log(exp(x) + exp(y)) as SentencePiece computes it (float, the exponential in double).
        private static float LogSumExp(float x, float y, bool first)
        {
            if (first)
            {
                return y;
            }

            float min = Math.Min(x, y), max = Math.Max(x, y);
            return max > min + 50f ? max : (float)(max + Math.Log(Math.Exp(min - max) + 1.0));
        }

        // Forward-backward; adds each node's expected count, times the word's count, to the piece's (SentencePiece's
        // PopulateMarginal: alpha excludes a node's own score, beta too; BOS and EOS score 0).
        public void Marginals(float frequency, Accumulator expected)
        {
            int n = Count, chars = Chars;
            EnsureNodeArrays(n);
            var alpha = _alpha;
            var beta = _beta;
            for (int r = 0; r < n; r++)
            {
                alpha[r] = Begin[r] == 0 ? 0 : Forward(Begin[r]);
            }

            float z = Forward(chars);
            for (int pos = chars; pos > 0; pos--)
            {
                for (int k = EndFirst[pos]; k < EndFirst[pos + 1]; k++)
                {
                    int l = EndOrder[k];
                    float b = 0;
                    if (pos < chars)
                    {
                        int first = BeginFirst[pos], last = BeginFirst[pos + 1];
                        for (int r = first; r < last; r++)
                        {
                            b = LogSumExp(b, Score[r] + beta[r], r == first);
                        }
                    }

                    beta[l] = b;
                }
            }

            for (int i = 0; i < n; i++)
            {
                float exponent = alpha[i] + Score[i] + beta[i] - z;
                expected.Add(Id[i], frequency * Math.Exp(exponent));
            }

            // alpha of a node beginning at pos > 0 (or of EOS at the end): over the nodes ending there, in order.
            float Forward(int pos)
            {
                float a = 0;
                int first = EndFirst[pos], last = EndFirst[pos + 1];
                for (int k = first; k < last; k++)
                {
                    int l = EndOrder[k];
                    a = LogSumExp(a, (1f * Score[l]) + alpha[l], k == first);
                }

                return a;
            }
        }

        // The best path's nodes (SentencePiece's Viterbi: the first of equal scores wins), appended to path.
        public void Viterbi(List<int> path)
        {
            int n = Count, chars = Chars;
            EnsureNodeArrays(n);
            var best = _best;
            var prev = _prev;
            float eosScore = 0;
            int eosPrev = -1;
            for (int pos = 0; pos <= chars; pos++)
            {
                int from = pos < chars ? BeginFirst[pos] : n, to = pos < chars ? BeginFirst[pos + 1] : n + 1;
                for (int r = from; r < to; r++)
                {
                    bool eos = r == n;
                    float rScore = eos ? 0 : Score[r];
                    float bestScore = 0;
                    int bestNode = -2;
                    if (pos == 0)
                    {
                        bestScore = 0 + rScore;
                        bestNode = -1;                                          // BOS
                    }
                    else
                    {
                        for (int k = EndFirst[pos]; k < EndFirst[pos + 1]; k++)
                        {
                            int l = EndOrder[k];
                            float score = best[l] + rScore;
                            if (bestNode == -2 || score > bestScore)
                            {
                                bestNode = l;
                                bestScore = score;
                            }
                        }
                    }

                    if (eos)
                    {
                        eosScore = bestScore;
                        eosPrev = bestNode;
                    }
                    else
                    {
                        best[r] = bestScore;
                        prev[r] = bestNode;
                    }
                }
            }

            _ = eosScore;
            int start = path.Count;
            for (int node = eosPrev; node >= 0; node = prev[node])
            {
                path.Add(node);
            }

            path.Reverse(start, path.Count - start);
        }

        // The two best paths (SentencePiece's NBest(2)): the best one's nodes and the second's, or null for the second
        // when the lattice has one path. Each node keeps its two best ways in, the first of equal scores first.
        public (List<int> Best, List<int>? Second) TwoBest()
        {
            int n = Count, chars = Chars;
            EnsureNodeArrays(n);
            float eos1 = 0, eos2 = 0;
            int eosPrev1 = -2, eosPrev2 = -2;
            byte eosRank1 = 0, eosRank2 = 0;
            for (int pos = 0; pos <= chars; pos++)
            {
                int from = pos < chars ? BeginFirst[pos] : n, to = pos < chars ? BeginFirst[pos + 1] : n + 1;
                for (int r = from; r < to; r++)
                {
                    bool eos = r == n;
                    float rScore = eos ? 0 : Score[r];
                    float s1 = 0, s2 = 0;
                    int p1 = -2, p2 = -2;
                    byte k1 = 0, k2 = 0;
                    void Offer(float score, int node, byte rank)
                    {
                        if (p1 == -2 || score > s1)
                        {
                            (s2, p2, k2) = (s1, p1, k1);
                            (s1, p1, k1) = (score, node, rank);
                        }
                        else if (p2 == -2 || score > s2)
                        {
                            (s2, p2, k2) = (score, node, rank);
                        }
                    }

                    if (pos == 0)
                    {
                        Offer(rScore, -1, 0);
                    }
                    else
                    {
                        for (int k = EndFirst[pos]; k < EndFirst[pos + 1]; k++)
                        {
                            int l = EndOrder[k];
                            Offer(_best[l] + rScore, l, 0);
                            if (_secondPrev[l] != -2)
                            {
                                Offer(_second[l] + rScore, l, 1);
                            }
                        }
                    }

                    if (eos)
                    {
                        (eos1, eosPrev1, eosRank1, eos2, eosPrev2, eosRank2) = (s1, p1, k1, s2, p2, k2);
                    }
                    else
                    {
                        (_best[r], _prev[r], _prevRank[r]) = (s1, p1, k1);
                        (_second[r], _secondPrev[r], _secondRank[r]) = (s2, p2, k2);
                    }
                }
            }

            _ = (eos1, eos2);
            List<int> Path(int node, byte rank)
            {
                var path = new List<int>();
                while (node >= 0)
                {
                    path.Add(node);
                    (node, rank) = rank == 0 ? (_prev[node], _prevRank[node]) : (_secondPrev[node], _secondRank[node]);
                }

                path.Reverse();
                return path;
            }

            return (Path(eosPrev1, eosRank1), eosPrev2 == -2 ? null : Path(eosPrev2, eosRank2));
        }

        private void EnsureNodeArrays(int n)
        {
            if (_alpha.Length < n)
            {
                int size = Math.Max(n, _alpha.Length * 2);
                _alpha = new float[size];
                _beta = new float[size];
                _best = new float[size];
                _second = new float[size];
                _prev = new int[size];
                _secondPrev = new int[size];
                _prevRank = new byte[size];
                _secondRank = new byte[size];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Sums over chunks of words, added in chunk order (so the worker count does not change a bit of the result).
    // ---------------------------------------------------------------------------------------------------------------------
    private sealed class Accumulator(int size)
    {
        public float[] Values { get; } = new float[size];

        public bool[] Seen { get; } = new bool[size];

        public List<int> Touched { get; } = [];

        // As SentencePiece adds a double to a float.
        public void Add(int id, double amount)
        {
            if (!Seen[id])
            {
                Seen[id] = true;
                Touched.Add(id);
            }

            Values[id] = (float)(Values[id] + amount);
        }

        public void AddTo(float[] total)
        {
            foreach (int id in Touched)
            {
                total[id] += Values[id];
                Values[id] = 0;
                Seen[id] = false;
            }

            Touched.Clear();
        }
    }

    private static float[] SumInChunkOrder(int words, int pieces, int workers, Action<int, int, Lattice, Accumulator> work, CancellationToken cancellationToken)
    {
        var total = new float[pieces];
        int chunks = (words + ChunkWords - 1) / ChunkWords;
        int next = -1, folded = 0;
        var done = new SortedDictionary<int, Accumulator>();
        var spare = new ConcurrentBag<Accumulator>();
        Parallel.For(0, Math.Max(1, Math.Min(workers, chunks)), new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, _ =>
        {
            var lattice = new Lattice();
            int chunk;
            while ((chunk = Interlocked.Increment(ref next)) < chunks)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var accumulator = spare.TryTake(out var reused) ? reused : new Accumulator(pieces);
                int from = chunk * ChunkWords, to = Math.Min(words, from + ChunkWords);
                work(from, to, lattice, accumulator);
                lock (done)
                {
                    done[chunk] = accumulator;
                    while (done.Remove(folded, out var ready))
                    {
                        ready.AddTo(total);
                        spare.Add(ready);
                        folded++;
                    }
                }
            }
        });

        return total;
    }

    // The E step: each piece's expected count over every word (SentencePiece's RunEStep).
    private static float[] ExpectedCounts(Model model, Words words, int workers, CancellationToken cancellationToken) =>
        SumInChunkOrder(words.Count, model.Count, workers, (from, to, lattice, expected) =>
        {
            for (int w = from; w < to; w++)
            {
                lattice.Set(words[w], model);
                lattice.Marginals(words.Counts[w], expected);
            }
        }, cancellationToken);

    // The M step (SentencePiece's RunMStep): pieces expected fewer than floor times (half a time) go; the rest score
    // digamma(count) - digamma(total), the Bayesian EM that acts as a sparse prior.
    private static List<(string Piece, float Score)> MStep(Model model, float[] expected, float floor)
    {
        var kept = new List<(string, float)>(model.Count);
        float sum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] < floor)
            {
                continue;
            }

            kept.Add((model.Pieces[i], expected[i]));
            sum += expected[i];
        }

        float logSum = (float)Digamma(sum);
        for (int i = 0; i < kept.Count; i++)
        {
            kept[i] = (kept[i].Item1, (float)(Digamma(kept[i].Item2) - logSum));
        }

        return kept;
    }

    private static double Digamma(double x)
    {
        double result = 0;
        for (; x < 7; x++)
        {
            result -= 1 / x;
        }

        x -= 1.0 / 2.0;
        double xx = 1.0 / x;
        double xx2 = xx * xx;
        double xx4 = xx2 * xx2;
        result += Math.Log(x) + ((1.0 / 24.0) * xx2) - ((7.0 / 960.0) * xx4) + ((31.0 / 8064.0) * xx4 * xx2) - ((127.0 / 30720.0) * xx4 * xx4);
        return result;
    }

    // Pruning (SentencePiece's PruneSentencePieces): each piece's loss is how much the likelihood falls when its uses go to
    // its second-best segmentation; the pieces with no alternative stay, then the ones with the largest loss, until
    // ShrinkingFactor of the pieces (at least the desired size) are kept.
    private static List<(string Piece, float Score)> Prune(Model model, Words words, int desired, float shrinking, int workers, CancellationToken cancellationToken)
    {
        int count = model.Count;
        var alwaysKeep = new bool[count];
        var alternatives = new int[count][];
        Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, () => new Lattice(), (i, _, lattice) =>
        {
            lattice.Set(Encoding.UTF8.GetBytes(model.Pieces[i]), model);
            var (best, second) = lattice.TwoBest();
            alwaysKeep[i] = true;
            alternatives[i] = [];
            if (second is null)
            {
                return lattice;                                                  // one way to write it: always kept
            }

            if (best.Count >= 2)
            {
                alwaysKeep[i] = false;                                           // its best path splits it: it can go
            }
            else if (best.Count == 1)
            {
                alternatives[i] = [.. second.Select(node => lattice.Id[node])];
            }

            return lattice;
        }, _ => { });

        // Each piece's count on the best paths of every word.
        var frequency = SumInChunkOrder(words.Count, count, workers, (from, to, lattice, counts) =>
        {
            var path = new List<int>();
            for (int w = from; w < to; w++)
            {
                lattice.Set(words[w], model);
                path.Clear();
                lattice.Viterbi(path);
                foreach (int node in path)
                {
                    counts.Add(lattice.Id[node], (float)words.Counts[w]);
                }
            }
        }, cancellationToken);

        double accumulated = 0;
        foreach (float f in frequency)
        {
            accumulated += f;
        }

        float sum = (float)accumulated;
        float logSum = (float)Math.Log(sum);
        var candidates = new List<(int Index, float Loss)>();
        var kept = new List<(string, float)>();
        for (int i = 0; i < count; i++)
        {
            if (frequency[i] == 0 || !alwaysKeep[i])
            {
                continue;                                                        // not on any best path, or splits anyway
            }

            if (alternatives[i].Length == 0)
            {
                kept.Add((model.Pieces[i], model.Scores[i]));
                continue;
            }

            float logProbability = (float)(Math.Log(frequency[i]) - logSum);
            float logSumAlternative = (float)Math.Log(sum + (frequency[i] * (float)(alternatives[i].Length - 1)));
            float logProbabilityAlternative = 0;
            foreach (int n in alternatives[i])
            {
                logProbabilityAlternative = (float)(logProbabilityAlternative + (Math.Log(frequency[n] + frequency[i]) - logSumAlternative));
            }

            float share = frequency[i] / sum;
            candidates.Add((i, share * (logProbability - logProbabilityAlternative)));
        }

        int prunedSize = Math.Max(desired, (int)(shrinking * count));
        candidates.Sort((a, b) => a.Loss != b.Loss ? b.Loss.CompareTo(a.Loss) : a.Index.CompareTo(b.Index));
        foreach (var (index, _) in candidates)
        {
            if (kept.Count == prunedSize)
            {
                break;
            }

            kept.Add((model.Pieces[index], model.Scores[index]));
        }

        return kept;
    }

    // The final pieces (SentencePiece's FinalizeSentencePieces): every required character (with a score just above the
    // lowest when it was pruned), then the best-scoring pieces, to the room left past the special pieces.
    private static List<(string Piece, float Score)> Finalize(Model model, Dictionary<int, long> required, int room)
    {
        var scores = new Dictionary<string, float>(model.Count, StringComparer.Ordinal);
        for (int i = 0; i < model.Count; i++)
        {
            scores[model.Pieces[i]] = model.Scores[i];
        }

        var final = new Dictionary<string, float>(StringComparer.Ordinal);
        float penalty = 0;
        foreach (var (c, _) in required.OrderByDescending(p => p.Value).ThenBy(p => p.Key))
        {
            string piece = char.ConvertFromUtf32(c);
            if (scores.TryGetValue(piece, out float score))
            {
                final[piece] = score;
            }
            else
            {
                final[piece] = model.MinScore + penalty;
                penalty += RequiredPenaltyStep;
            }
        }

        foreach (var (piece, score) in BestFirst(scores))
        {
            if (final.ContainsKey(piece))
            {
                continue;
            }

            if (final.Count == room)
            {
                break;
            }

            final[piece] = score;
        }

        return BestFirst(final);
    }

    // Pieces by score, highest first, then by text (UTF-8 byte order), as SentencePiece's Sorted.
    private static List<(string Piece, float Score)> BestFirst(Dictionary<string, float> pieces)
    {
        var list = pieces.Select(p => (p.Key, p.Value)).ToList();
        list.Sort((a, b) => a.Value != b.Value ? b.Value.CompareTo(a.Value) : CodePointOrder.Compare(a.Key, b.Key));
        return list;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // The tokenizer: the ModelProto written, then read back as the sentencepiece reader reads a tokenizer.model.
    // ---------------------------------------------------------------------------------------------------------------------
    private static IPretrainedTokenizer Build(IReadOnlyList<string> files, TokenizerTrainingOptions options, MetaPieces meta, List<(string Piece, float Score)> final,
        string rule, ArabicMarks removal, byte[] charsmap)
    {
        var pieces = new List<string>(meta.Pieces.Count + final.Count);
        var scores = new List<float>(pieces.Capacity);
        var types = new List<SentencePieceType>(pieces.Capacity);
        pieces.AddRange(meta.Pieces);
        scores.AddRange(meta.Pieces.Select(_ => 0f));
        types.AddRange(meta.Types);
        foreach (var (piece, score) in final)
        {
            pieces.Add(piece);
            scores.Add(score);
            types.Add(SentencePieceType.Normal);
        }

        var settings = new SentencePieceWriter.Settings
        {
            Algorithm = SentencePieceAlgorithm.Unigram,
            ByteFallback = options.ByteFallback,
            UnkId = meta.Unknown,
            BosId = meta.Bos,
            EosId = meta.Eos,
            PadId = meta.Pad,
            TreatWhitespaceAsSuffix = options.TreatWhitespaceAsSuffix,
            NormalizerName = !removal.Any ? rule : rule == TokenizerTrainers.Identity ? "idrak_arabic" : $"{rule}_idrak_arabic",
            PrecompiledCharsmap = charsmap,
            Training = new SentencePieceWriter.TrainingFields
            {
                VocabularySize = pieces.Count,
                CharacterCoverage = (float)options.CharacterCoverage,
                SeedSize = options.SeedSize,
                ShrinkingFactor = (float)options.ShrinkingFactor,
                EmIterations = options.EmIterations,
                MaxPieceLength = options.MaxPieceLength,
                SplitDigits = options.SplitDigits,
                SplitByWhitespace = options.SplitByWhitespace,
                SplitByUnicodeScript = options.SplitByUnicodeScript,
                SplitByNumber = options.SplitByNumber,
                AllowWhitespaceOnlyPieces = options.AllowWhitespaceOnlyPieces,
                MaxSentenceLength = options.MaxSentenceLength,
                ControlSymbols = [.. meta.Pieces.Where((p, i) => meta.Types[i] == SentencePieceType.Control && i != meta.Bos && i != meta.Eos && i != meta.Pad)],
            },
        };
        byte[] proto = SentencePieceWriter.Write(pieces, scores, types, settings);
        string source = $"unigram trainer ({string.Join(", ", files.Select(Path.GetFileName))})";
        var model = SentencePieceModel.Read(proto, source);
        var tokenizer = TokenizerModels.Create(SentencePieceReader.FromModel(model), new TokenizerOptions());
        if (tokenizer is UnigramTokenizer unigram)
        {
            unigram.KeepModelProto(proto);
        }

        return tokenizer;
    }

    private static int CodePoint(ReadOnlySpan<char> text, int at, out int width)
    {
        if (char.IsHighSurrogate(text[at]) && at + 1 < text.Length && char.IsLowSurrogate(text[at + 1]))
        {
            width = 2;
            return char.ConvertToUtf32(text[at], text[at + 1]);
        }

        width = 1;
        return text[at];
    }

    private static int CodePoint(char[] text, int at, int end, out int width) => CodePoint(text.AsSpan(0, end), at, out width);
}

/// <summary>Code-point (UTF-8 byte) order for UTF-16 strings, which ordinal order is not past U+D7FF.</summary>
internal static class CodePointOrder
{
    public static int Compare(string? x, string? y) => Compare(x.AsSpan(), y.AsSpan());

    public static int Compare(ReadOnlySpan<char> x, ReadOnlySpan<char> y)
    {
        int length = Math.Min(x.Length, y.Length);
        int at = x[..length].CommonPrefixLength(y[..length]);
        if (at == length)
        {
            return x.Length.CompareTo(y.Length);
        }

        return Fix(x[at]).CompareTo(Fix(y[at]));

        // Surrogates (supplementary characters) sort after U+E000-U+FFFF, as their code points do.
        static int Fix(char c) => c >= 0xE000 ? c - 0x800 : c >= 0xD800 ? c + 0x2000 : c;
    }
}

/// <summary>The files a trainer reads: each must exist, and at least one is given.</summary>
internal static class TrainingFiles
{
    public static void Check(IReadOnlyList<string> files)
    {
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
    }
}

/// <summary>The Arabic marks <see cref="TokenizerTrainingOptions.Arabic"/> removes.</summary>
internal sealed class ArabicMarks
{
    /// <summary>Tatweel (kashida).</summary>
    public const char Tatweel = 'ـ';

    // The harakat, tanween, shadda, sukun and maddah above (U+064B-U+065F), the honorific and Quranic signs above letters
    // (U+0610-U+061A), the superscript alef (U+0670) and the small Quranic marks (U+06D6-U+06DC, U+06DF-U+06E4, U+06E7,
    // U+06E8, U+06EA-U+06ED).
    private static readonly (int First, int Last)[] DiacriticRanges =
        [(0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06DC), (0x06DF, 0x06E4), (0x06E7, 0x06E8), (0x06EA, 0x06ED)];

    private readonly bool _tatweel, _diacritics;
    private SearchValues<char>? _characters;

    private ArabicMarks(bool tatweel, bool diacritics) => (_tatweel, _diacritics) = (tatweel, diacritics);

    public static ArabicMarks Of(TokenizerTrainingOptions options) =>
        new(options.RemovesArabic(TokenizerTrainers.Tatweel), options.RemovesArabic(TokenizerTrainers.Diacritics));

    public bool Any => _tatweel || _diacritics;

    /// <summary>Which removals these are, as a key (for a cache of what is built from them).</summary>
    public string Key => (_tatweel ? "tatweel" : "") + (_diacritics ? "+diacritics" : "");

    public bool Removes(char c) => (_tatweel && c == Tatweel) || (_diacritics && IsDiacritic(c));

    public IEnumerable<int> CodePoints =>
        (_tatweel ? [Tatweel] : Array.Empty<int>()).Concat(_diacritics ? DiacriticRanges.SelectMany(r => Enumerable.Range(r.First, r.Last - r.First + 1)) : []);

    /// <summary>The removed characters, to search for.</summary>
    public SearchValues<char> Characters => _characters ??= SearchValues.Create([.. CodePoints.Select(c => (char)c)]);

    /// <summary><paramref name="text"/> without the removed characters (the same instance when it holds none).</summary>
    public string Remove(string text)
    {
        if (!Any || !text.AsSpan().ContainsAny(Characters))
        {
            return text;
        }

        var kept = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (!Removes(c))
            {
                kept.Append(c);
            }
        }

        return kept.ToString();
    }

    /// <summary>The standard normalizers that remove the same marks (tokenizer.json <c>Replace</c> steps), or none.</summary>
    public IEnumerable<System.Text.Json.Nodes.JsonObject> ReplaceSteps()
    {
        if (_tatweel)
        {
            yield return Replace(Tatweel.ToString(), regex: false);
        }

        if (_diacritics)
        {
            var range = new StringBuilder("[");
            foreach (var (first, last) in DiacriticRanges)
            {
                range.Append((char)first);
                if (last != first)
                {
                    range.Append('-').Append((char)last);
                }
            }

            yield return Replace(range.Append(']').ToString(), regex: true);
        }

        static System.Text.Json.Nodes.JsonObject Replace(string pattern, bool regex) => new()
        {
            ["type"] = "Replace",
            ["pattern"] = new System.Text.Json.Nodes.JsonObject { [regex ? "Regex" : "String"] = pattern },
            ["content"] = "",
        };
    }

    private static bool IsDiacritic(char c)
    {
        foreach (var (first, last) in DiacriticRanges)
        {
            if (c >= first && c <= last)
            {
                return true;
            }
        }

        return false;
    }
}
