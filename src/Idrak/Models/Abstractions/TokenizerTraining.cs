// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>Settings for <see cref="ITokenizerTrainer.Train"/>.</summary>
public sealed record TokenizerTrainingOptions
{
    /// <summary>The vocabulary to reach: special tokens, the alphabet and the merged tokens together.</summary>
    public int VocabularySize { get; init; } = 32000;

    /// <summary>A pair seen fewer times than this is never merged (training stops at the first such pair).</summary>
    public int MinFrequency { get; init; } = 2;

    /// <summary>Special tokens: the first ids, in this order, written as special added tokens (matched before anything else when encoding).</summary>
    public IReadOnlyList<string> SpecialTokens { get; init; } = [];

    /// <summary>
    /// How text is cut into words before merging: <see cref="TokenizerTrainers.ByteLevel"/> (bytes as characters, the
    /// GPT-2 word rule; every text encodes, the default) or <see cref="TokenizerTrainers.Metaspace"/> (spaces as "▁" that
    /// start a word, SentencePiece's way; characters outside the alphabet are unknown).
    /// </summary>
    public string PreTokenizer { get; init; } = TokenizerTrainers.ByteLevel;

    /// <summary>Byte level only: a space before the text when it does not start with one, so a first word is spelled as later ones.</summary>
    public bool AddPrefixSpace { get; init; }

    /// <summary>Keep at most this many characters in the alphabet (the most frequent), or every one (null). Byte level always keeps its 256.</summary>
    public int? LimitAlphabet { get; init; }

    /// <summary>Threads that count words (0: one per processor).</summary>
    public int Workers { get; init; }

    /// <summary>
    /// How a line's end is read: "lf" (the default) reads "\r\n" as "\n", so a file trains the same tokenizer whichever
    /// system wrote it; "keep" keeps the bytes as they are, "\r" included, as Hugging Face's <c>train_from_files</c> does.
    /// </summary>
    public string LineEnds { get; init; } = "lf";

    /// <summary>
    /// Unigram: the share of the text's characters the alphabet covers (SentencePiece's <c>character_coverage</c>): the
    /// most frequent characters are kept until they cover this share, and the rest are unknown (or bytes, with
    /// <see cref="ByteFallback"/>). In (0, 1]; 1 keeps every character.
    /// </summary>
    public double CharacterCoverage { get; init; } = 0.9995;

    /// <summary>Unigram: the 256 pieces &lt;0x00&gt; to &lt;0xFF&gt;, so a character outside the alphabet encodes as its UTF-8 bytes instead of the unknown piece.</summary>
    public bool ByteFallback { get; init; }

    /// <summary>Unigram: the longest piece, in characters (SentencePiece's <c>max_sentencepiece_length</c>), 1 or more.</summary>
    public int MaxPieceLength { get; init; } = 16;

    /// <summary>Unigram: how many frequent substrings seed the vocabulary before pruning (SentencePiece's <c>seed_sentencepiece_size</c>), besides every character.</summary>
    public int SeedSize { get; init; } = 1_000_000;

    /// <summary>Unigram: the share of the pieces each pruning round keeps, in (0, 1) (SentencePiece's <c>shrinking_factor</c>).</summary>
    public double ShrinkingFactor { get; init; } = 0.75;

    /// <summary>Unigram: the EM steps (expected counts, then new scores) before each pruning round, 1 or more (SentencePiece's <c>num_sub_iterations</c>).</summary>
    public int EmIterations { get; init; } = 2;

    /// <summary>
    /// Every digit (0-9) is a piece of its own: Unigram learns no piece of two digits (SentencePiece's <c>split_digits</c>);
    /// WordPiece cuts numbers into digits before learning and encoding (a <c>Digits</c> pre-tokenizer).
    /// </summary>
    public bool SplitDigits { get; init; }

    /// <summary>WordPiece: text is lowercased before learning and encoding (BERT's uncased models).</summary>
    public bool Lowercase { get; init; }

    /// <summary>WordPiece: accents are removed (NFD, then the nonspacing marks dropped); null follows <see cref="Lowercase"/>, as BERT's normalizer does.</summary>
    public bool? StripAccents { get; init; }

    /// <summary>
    /// WordPiece: how the next merge is chosen. <see cref="TokenizerTrainers.Frequency"/> (the default): the most frequent
    /// pair, as Hugging Face's <c>WordPieceTrainer</c> does (BPE merges with the ## prefix). <see cref="TokenizerTrainers.Likelihood"/>:
    /// the pair whose count over the product of its parts' counts is largest, the original WordPiece scoring.
    /// </summary>
    public string Scoring { get; init; } = TokenizerTrainers.Frequency;

    /// <summary>
    /// Unigram and WordPiece: Arabic marks removed before learning and encoding, written into the tokenizer so every reader
    /// of the file removes them too: <see cref="TokenizerTrainers.Tatweel"/> (U+0640), <see cref="TokenizerTrainers.Diacritics"/>
    /// (the harakat, tanween, shadda, sukun, superscript alef and Quranic marks), or both. Empty: none.
    /// </summary>
    public IReadOnlyList<string> Arabic { get; init; } = [];

    /// <summary>
    /// Unigram: the normalization rule applied before learning and written into the tokenizer, as SentencePiece names its
    /// built-in rules (<c>normalization_rule_name</c>): <see cref="TokenizerTrainers.Identity"/> (the default: the text as
    /// it is), <see cref="TokenizerTrainers.Nfkc"/> (Unicode NFKC) or <see cref="TokenizerTrainers.NmtNfkc"/> (NFKC, with
    /// control characters removed and the other whitespace characters made spaces, SentencePiece's default). The rules are
    /// built as SentencePiece builds them, from the library's own Unicode normalization (its Unicode version).
    /// </summary>
    public string Normalization { get; init; } = TokenizerTrainers.Identity;

    /// <summary>
    /// Unigram: lines longer than this many UTF-8 bytes (without the line end) are skipped, as SentencePiece's
    /// <c>max_sentence_length</c> skips them (its default is 4192); 0, the default, skips none.
    /// </summary>
    public int MaxSentenceLength { get; init; }

    /// <summary>
    /// Unigram: pieces stay within words, each word starting at a ▁ (SentencePiece's <c>split_by_whitespace</c>, true);
    /// false lets a piece span words (▁ inside it, never at its end).
    /// </summary>
    public bool SplitByWhitespace { get; init; } = true;

    /// <summary>Unigram: a piece holds characters of one script (SentencePiece's <c>split_by_unicode_script</c>, true); false lets a piece mix scripts.</summary>
    public bool SplitByUnicodeScript { get; init; } = true;

    /// <summary>
    /// Unigram: digits (0-9 and their full-width forms) are a script of their own, so a piece does not join them to
    /// letters (SentencePiece's <c>split_by_number</c>, true); false lets digits join letters of any script.
    /// </summary>
    public bool SplitByNumber { get; init; } = true;

    /// <summary>
    /// Unigram: the ▁ ends a word instead of starting it (SentencePiece's <c>treat_whitespace_as_suffix</c>, false): the
    /// text gets a ▁ after it, not before, and pieces end with ▁. Written into the tokenizer, whose readers honour it.
    /// </summary>
    public bool TreatWhitespaceAsSuffix { get; init; }

    /// <summary>
    /// Unigram: runs of ▁ may be pieces of their own (SentencePiece's <c>allow_whitespace_only_pieces</c>, false), so a run
    /// of whitespace in the text (indentation written as ▁) stays one word instead of one word a ▁.
    /// </summary>
    public bool AllowWhitespaceOnlyPieces { get; init; }

    /// <summary>
    /// Unigram: pruning stops once the pieces are at most this many times the vocabulary, then the best of them fill it
    /// (SentencePiece's 1.1); 1 or more.
    /// </summary>
    public double DesiredSizeFactor { get; init; } = 1.1;

    /// <summary>Unigram: an EM step drops the pieces expected fewer times than this (SentencePiece's 0.5); 0 or more.</summary>
    public double ExpectedCountFloor { get; init; } = 0.5;

    /// <summary>
    /// Unigram: while learning, a character no piece covers scores this much below the lowest piece (SentencePiece's
    /// 10); 0 or more. Encoding with the learned tokenizer is SentencePiece's, which keeps 10.
    /// </summary>
    public double UnknownPenalty { get; init; } = 10;

    /// <summary>
    /// Unigram and WordPiece: the unknown token; null takes the model's usual one ("&lt;unk&gt;" for Unigram, "[UNK]" for
    /// WordPiece). When <see cref="SpecialTokens"/> does not list it, it takes the first id after them.
    /// </summary>
    public string? UnknownToken { get; init; }

    /// <summary>
    /// Checks the settings every built-in trainer shares and the ones <paramref name="trainer"/> reads (bpe, unigram,
    /// wordpiece), so a bad value is refused before any file is read, with the setting named.
    /// </summary>
    /// <exception cref="ArgumentException">A setting is out of range or unknown; the message names it.</exception>
    public void Validate(string trainer)
    {
        static void Refuse(string message) => throw new ArgumentException(message, "options");
        if (VocabularySize < 1)
        {
            Refuse($"VocabularySize must be at least 1, not {VocabularySize}.");
        }

        if (MinFrequency < 0)
        {
            Refuse($"MinFrequency cannot be negative ({MinFrequency}).");
        }

        if (Workers < 0)
        {
            Refuse($"Workers cannot be negative ({Workers}); 0 takes one per processor.");
        }

        if (LimitAlphabet is < 0)
        {
            Refuse($"LimitAlphabet cannot be negative ({LimitAlphabet}).");
        }

        if (LineEnds is not ("lf" or "keep"))
        {
            Refuse($"LineEnds is \"lf\" (\\r\\n read as \\n) or \"keep\" (the bytes as they are), not '{LineEnds}'.");
        }

        if (SpecialTokens.Any(string.IsNullOrEmpty))
        {
            Refuse("SpecialTokens holds an empty token.");
        }

        if (UnknownToken is { Length: 0 })
        {
            Refuse("UnknownToken is empty: give a token, or null for the model's usual one.");
        }

        if (!(CharacterCoverage > 0 && CharacterCoverage <= 1))
        {
            Refuse($"CharacterCoverage is a share of the characters in (0, 1], not {CharacterCoverage.ToString(System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        if (MaxPieceLength < 1)
        {
            Refuse($"MaxPieceLength must be at least 1 character, not {MaxPieceLength}.");
        }

        if (SeedSize < 1)
        {
            Refuse($"SeedSize must be at least 1, not {SeedSize}.");
        }

        if (!(ShrinkingFactor > 0 && ShrinkingFactor < 1))
        {
            Refuse($"ShrinkingFactor is the share kept per pruning round, in (0, 1), not {ShrinkingFactor.ToString(System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        if (EmIterations < 1)
        {
            Refuse($"EmIterations must be at least 1, not {EmIterations}.");
        }

        if (TokenizerTrainers.NormalizationRules.All(r => !r.Equals(Normalization, StringComparison.OrdinalIgnoreCase)))
        {
            Refuse($"Normalization is {string.Join(", ", TokenizerTrainers.NormalizationRules.Select(r => $"\"{r}\""))}, not '{Normalization}'.");
        }

        if (MaxSentenceLength < 0)
        {
            Refuse($"MaxSentenceLength is a length in bytes, or 0 for no limit, not {MaxSentenceLength}.");
        }

        if (!(DesiredSizeFactor >= 1 && double.IsFinite(DesiredSizeFactor)))
        {
            Refuse($"DesiredSizeFactor must be at least 1 (pruning stops at that many times the vocabulary), not {DesiredSizeFactor.ToString(System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        if (!(ExpectedCountFloor >= 0 && double.IsFinite(ExpectedCountFloor)))
        {
            Refuse($"ExpectedCountFloor must be 0 or more, not {ExpectedCountFloor.ToString(System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        if (!(UnknownPenalty >= 0 && double.IsFinite(UnknownPenalty)))
        {
            Refuse($"UnknownPenalty must be 0 or more, not {UnknownPenalty.ToString(System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        if (!Scoring.Equals(TokenizerTrainers.Frequency, StringComparison.OrdinalIgnoreCase) && !Scoring.Equals(TokenizerTrainers.Likelihood, StringComparison.OrdinalIgnoreCase))
        {
            Refuse($"Scoring is \"{TokenizerTrainers.Frequency}\" or \"{TokenizerTrainers.Likelihood}\", not '{Scoring}'.");
        }

        foreach (string name in Arabic)
        {
            if (!name.Equals(TokenizerTrainers.Tatweel, StringComparison.OrdinalIgnoreCase) && !name.Equals(TokenizerTrainers.Diacritics, StringComparison.OrdinalIgnoreCase))
            {
                Refuse($"Arabic names \"{TokenizerTrainers.Tatweel}\" and \"{TokenizerTrainers.Diacritics}\", not '{name}'.");
            }
        }

        bool bpe = trainer.Equals(TokenizerTrainers.Bpe, StringComparison.OrdinalIgnoreCase);
        if (bpe && !PreTokenizer.Equals(TokenizerTrainers.ByteLevel, StringComparison.OrdinalIgnoreCase)
            && !PreTokenizer.Equals(TokenizerTrainers.Metaspace, StringComparison.OrdinalIgnoreCase))
        {
            Refuse($"PreTokenizer '{PreTokenizer}' is not one the bpe trainer writes ({TokenizerTrainers.ByteLevel}, {TokenizerTrainers.Metaspace}).");
        }

        if (bpe && Arabic.Count > 0)
        {
            Refuse("Arabic is read by the unigram and wordpiece trainers; the bpe trainer writes no normalizer.");
        }

        if (!trainer.Equals(TokenizerTrainers.Unigram, StringComparison.OrdinalIgnoreCase) && !Normalization.Equals(TokenizerTrainers.Identity, StringComparison.OrdinalIgnoreCase))
        {
            Refuse($"Normalization is read by the unigram trainer; the {trainer} trainer writes its own normalizer (or none).");
        }
    }

    /// <summary>Whether <see cref="Arabic"/> names <paramref name="choice"/> (<see cref="TokenizerTrainers.Tatweel"/> or <see cref="TokenizerTrainers.Diacritics"/>).</summary>
    public bool RemovesArabic(string choice) => Arabic.Any(a => a.Equals(choice, StringComparison.OrdinalIgnoreCase));
}

/// <summary>Where a <see cref="ITokenizerTrainer"/> is: reading the files, then learning.</summary>
/// <param name="Stage">
/// "words" (reading and counting; Done and Total in bytes); then BPE and WordPiece: "merges" (Done and Total in tokens of
/// the vocabulary); Unigram: "seeds" (the frequent substrings; Done and Total in suffix-tree nodes), "em" (Done and Total in
/// EM steps, the total estimated from the seeds, the vocabulary and the shrinking factor) and "prune" (Done and Total in
/// pruning rounds, estimated likewise).
/// </param>
/// <param name="Done">How far the stage is.</param>
/// <param name="Total">The stage's end.</param>
public readonly record struct TokenizerTrainingProgress(string Stage, long Done, long Total);

/// <summary>
/// Learns a tokenizer from text files. The result encodes and decodes like one read from a tokenizer.json, and
/// <see cref="IPretrainedTokenizer.Save"/> writes it as one, so a model made from scratch (plan 19) carries its own vocabulary.
/// Register one with <see cref="TokenizerTrainers.Register"/>.
/// </summary>
public interface ITokenizerTrainer
{
    /// <summary>The trainer's name ("bpe").</summary>
    string Name { get; }

    /// <summary>One line on what it learns.</summary>
    string Summary { get; }

    /// <summary>
    /// Learns a tokenizer from UTF-8 text <paramref name="files"/>, each line (with its line end) one sequence, as
    /// Hugging Face's <c>train_from_files</c> reads them.
    /// </summary>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    /// <exception cref="ArgumentException">An option is out of range.</exception>
    IPretrainedTokenizer Train(IReadOnlyList<string> files, TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// A tokenizer that can also be written as files of other kinds than tokenizer.json, each read back by its reader
/// (<see cref="TokenizerFiles"/>) to the same tokenizer: a trained Unigram tokenizer as SentencePiece's tokenizer.model.
/// </summary>
public interface ITokenizerCompanionFiles
{
    /// <summary>The file names it writes ("tokenizer.model"); empty when it writes none.</summary>
    IReadOnlyList<string> CompanionFiles { get; }

    /// <summary>Writes the companion file <paramref name="name"/> (one of <see cref="CompanionFiles"/>) to <paramref name="path"/>.</summary>
    /// <exception cref="ArgumentException">It writes no file of that name.</exception>
    void SaveCompanion(string name, string path);
}

/// <summary>
/// The tokenizer trainers, by name. Built in: "bpe", byte-pair encoding learned as Hugging Face's <c>BpeTrainer</c>
/// learns it (the same vocabulary and merges, in the same order, from the same files); "unigram", SentencePiece's unigram
/// language model learned by its procedure (suffix-array seeds, EM with the digamma correction, pruning by loss), written
/// as tokenizer.json and tokenizer.model; "wordpiece", BERT's model, by frequency (Hugging Face's <c>WordPieceTrainer</c>
/// with its ties in a fixed order) or by likelihood. Others plug in with <see cref="Register"/>.
/// </summary>
public static class TokenizerTrainers
{
    /// <summary>The library's byte-pair-encoding trainer.</summary>
    public const string Bpe = "bpe";

    /// <summary>The library's unigram-language-model trainer (SentencePiece's procedure).</summary>
    public const string Unigram = "unigram";

    /// <summary>The library's WordPiece trainer.</summary>
    public const string WordPiece = "wordpiece";

    /// <summary>WordPiece scoring (<see cref="TokenizerTrainingOptions.Scoring"/>): the most frequent pair, as Hugging Face's trainer.</summary>
    public const string Frequency = "frequency";

    /// <summary>WordPiece scoring (<see cref="TokenizerTrainingOptions.Scoring"/>): the pair's count over its parts' counts.</summary>
    public const string Likelihood = "likelihood";

    /// <summary>An <see cref="TokenizerTrainingOptions.Arabic"/> choice: tatweel (U+0640) removed.</summary>
    public const string Tatweel = "tatweel";

    /// <summary>An <see cref="TokenizerTrainingOptions.Arabic"/> choice: the diacritics removed.</summary>
    public const string Diacritics = "diacritics";

    /// <summary>A <see cref="TokenizerTrainingOptions.Normalization"/> rule: the text as it is (SentencePiece's "identity").</summary>
    public const string Identity = "identity";

    /// <summary>A <see cref="TokenizerTrainingOptions.Normalization"/> rule: Unicode NFKC, as SentencePiece's "nfkc" map applies it.</summary>
    public const string Nfkc = "nfkc";

    /// <summary>
    /// A <see cref="TokenizerTrainingOptions.Normalization"/> rule: SentencePiece's "nmt_nfkc", NFKC with ASCII control
    /// characters removed and the other whitespace characters (tab, line ends, zero-width spaces, U+2581) made spaces.
    /// </summary>
    public const string NmtNfkc = "nmt_nfkc";

    /// <summary>The <see cref="TokenizerTrainingOptions.Normalization"/> rules the unigram trainer knows.</summary>
    public static IReadOnlyList<string> NormalizationRules { get; } = [Identity, Nfkc, NmtNfkc];

    /// <summary>The byte-level pre-tokenizer (<see cref="TokenizerTrainingOptions.PreTokenizer"/>).</summary>
    public const string ByteLevel = "bytelevel";

    /// <summary>The metaspace pre-tokenizer (<see cref="TokenizerTrainingOptions.PreTokenizer"/>).</summary>
    public const string Metaspace = "metaspace";

    private static readonly SlotTable<string, ITokenizerTrainer> Table = BuiltIn();

    private static SlotTable<string, ITokenizerTrainer> BuiltIn()
    {
        var table = new SlotTable<string, ITokenizerTrainer>(nameof(TokenizerTrainers), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a trainer's vocabulary and merges are its own, so another trainer cannot give the same tokenizer");
        table.RegisterDefault(Bpe, new BpeTrainer());
        table.RegisterDefault(Unigram, new UnigramTrainer());
        table.RegisterDefault(WordPiece, new WordPieceTrainer());
        return table;
    }

    /// <summary>Registers <paramref name="trainer"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITokenizerTrainer trainer)
    {
        ArgumentNullException.ThrowIfNull(trainer);
        ArgumentException.ThrowIfNullOrWhiteSpace(trainer.Name);
        Table.Register(trainer.Name, trainer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's trainer of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The trainer of that name, or null.</summary>
    public static ITokenizerTrainer? Find(string name) => Table.Find(name);

    /// <summary>The trainer of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ITokenizerTrainer Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(TokenizerTrainers), name, Table.Keys,
            $"Tokenizer trainer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TokenizerTrainers.Register.");

    /// <summary>The library's trainer of that name, or null.</summary>
    public static ITokenizerTrainer? Default(string name) => Table.Default(name);

    /// <summary>Who registered the trainer of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each trainer as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(t => $"{t.Name}: {t.Summary}"));
}
