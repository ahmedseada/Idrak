// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// Reads one kind of tokenizer file (a Hugging Face tokenizer.json, a SentencePiece model, a vocabulary file) into the
/// definition a tokenizer model builds from. Register one with <see cref="TokenizerFiles.Register"/>.
/// </summary>
public interface ITokenizerFileReader
{
    /// <summary>The reader's name ("tokenizer.json"), as <see cref="TokenizerOptions.Format"/> names it.</summary>
    string Name { get; }

    /// <summary>One line on the files it reads.</summary>
    string Summary { get; }

    /// <summary>
    /// The file this reader reads at <paramref name="path"/>: in a model folder, its file of this kind if there is one; a
    /// file, itself when it is of this kind. Null when there is none.
    /// </summary>
    string? Find(string path);

    /// <summary>Reads <paramref name="file"/> (one <see cref="Find"/> gave).</summary>
    /// <exception cref="InvalidDataException">The file is not a tokenizer of this kind.</exception>
    TokenizerDefinition Read(string file, TokenizerOptions options);
}

/// <summary>
/// The tokenizer file readers, by name; <see cref="PretrainedTokenizers"/> asks them in registration order (the built-in
/// ones first) for the file of a model folder. Built in: "tokenizer.json" (Hugging Face's, any model type). Others plug in
/// with <see cref="Register"/>.
/// </summary>
public static class TokenizerFiles
{
    /// <summary>The Hugging Face tokenizer.json reader.</summary>
    public const string TokenizerJson = "tokenizer.json";

    private static readonly SlotTable<string, ITokenizerFileReader> Table = new(nameof(TokenizerFiles), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a reader's definition is its own file's, so another reader cannot give the same one");

    static TokenizerFiles() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterTokenizerFiles);   // the built-in readers, on first use

    /// <summary>Registers <paramref name="reader"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITokenizerFileReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentException.ThrowIfNullOrWhiteSpace(reader.Name);
        Table.Register(reader.Name, reader, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's reader of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order the readers are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The readers, in the order they are asked.</summary>
    public static IReadOnlyList<ITokenizerFileReader> Readers => Table.Values;

    /// <summary>The reader of that name, or null.</summary>
    public static ITokenizerFileReader? Find(string name) => Table.Find(name);

    /// <summary>The reader of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ITokenizerFileReader Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Tokenizer file reader '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TokenizerFiles.Register.");

    /// <summary>The library's reader of that name, or null.</summary>
    public static ITokenizerFileReader? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reader of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each reader as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(r => $"{r.Name}: {r.Summary}"));
}
