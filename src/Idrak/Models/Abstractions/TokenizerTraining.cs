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
}

/// <summary>Where a <see cref="ITokenizerTrainer"/> is: counting words in the files, then merging.</summary>
/// <param name="Stage">"words" (reading and counting; Done and Total in bytes) or "merges" (Done and Total in tokens of the vocabulary).</param>
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
/// The tokenizer trainers, by name. Built in: "bpe", byte-pair encoding learned as Hugging Face's <c>BpeTrainer</c>
/// learns it (the same vocabulary and merges, in the same order, from the same files). Others plug in with
/// <see cref="Register"/>.
/// </summary>
public static class TokenizerTrainers
{
    /// <summary>The library's byte-pair-encoding trainer.</summary>
    public const string Bpe = "bpe";

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
        ?? throw new NotSupportedException($"Tokenizer trainer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TokenizerTrainers.Register.");

    /// <summary>The library's trainer of that name, or null.</summary>
    public static ITokenizerTrainer? Default(string name) => Table.Default(name);

    /// <summary>Who registered the trainer of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each trainer as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(t => $"{t.Name}: {t.Summary}"));
}
