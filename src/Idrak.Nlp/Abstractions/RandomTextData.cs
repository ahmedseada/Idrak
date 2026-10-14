// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Nlp.Abstractions;

/// <summary>
/// Random text in one language, a line at a time, for drawing training pages when there is no text to draw (plan 18):
/// the words, word shapes, numbers and punctuation a reader of that language meets, not meaningful sentences. Register
/// one with <see cref="RandomTextSources.Register"/>; <see cref="PooledTextSource"/> is one made from word pools.
/// </summary>
public interface IRandomTextSource
{
    /// <summary>The source's name, a language tag for a language's own ("ar").</summary>
    string Name { get; }

    /// <summary>One line on what it writes.</summary>
    string Summary { get; }

    /// <summary>A line of exactly <paramref name="words"/> words (numbers and dates count as words), from <paramref name="random"/>.</summary>
    string Line(Random random, int words, RandomTextOptions options);
}

/// <summary>
/// The random text sources, by language tag. Built in: "ar" (Arabic in the words of official and legal letters, with
/// names, places, Hijri years, dates and amounts) and "en" (the same in English). Any other language is a pools file
/// (<see cref="PooledTextSource.Read"/>) or a word list (<see cref="PooledTextSource.FromWords"/>) registered under its
/// tag (<see cref="Register"/>).
/// </summary>
public static class RandomTextSources
{
    private static readonly SlotTable<string, IRandomTextSource> Table = BuiltIn();

    private static SlotTable<string, IRandomTextSource> BuiltIn()
    {
        var table = new SlotTable<string, IRandomTextSource>(nameof(RandomTextSources), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a text is drawn from its source's own word pools, so another source cannot give the same text");
        table.RegisterDefault("ar", PooledTextSource.Embedded("random-text.ar.json"));
        table.RegisterDefault("en", PooledTextSource.Embedded("random-text.en.json"));
        return table;
    }

    /// <summary>Registers <paramref name="source"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRandomTextSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(source.Name);
        Table.Register(source.Name, source, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's source of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The source of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IRandomTextSource Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Random text source '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); give a pools file or a word list, or add one with RandomTextSources.Register.");

    /// <summary>The library's source of that name, or null.</summary>
    public static IRandomTextSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the source of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each source as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));
}
