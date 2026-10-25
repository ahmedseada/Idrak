// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Text.Abstractions;

/// <summary>
/// One record a synthetic page is drawn from: the answer it was tuned to give (whose page text the page draws) and how to
/// write the record again with a new answer and image (<see cref="Write"/>).
/// </summary>
public sealed class SyntheticRecord
{
    /// <summary>The record's answer as written: JSON (in a code fence or not) or plain text.</summary>
    public required string Answer { get; init; }

    /// <summary>Where the record is, for messages ("train.jsonl line 12").</summary>
    public required string Where { get; init; }

    /// <summary>The record again, with <c>answer</c> as its answer and <c>image</c> (a relative path) as its one image.</summary>
    public required Func<string, string, JsonObject> Write { get; init; }
}

/// <summary>
/// A kind of file that synthetic pages are made from (plan 18): its records' answers in, records with new answers and
/// images out, in the same shape. Register one with <see cref="SyntheticRecordFormats.Register"/>.
/// </summary>
public interface ISyntheticRecordFormat
{
    /// <summary>The format's name ("sharegpt").</summary>
    string Name { get; }

    /// <summary>One line on what it reads.</summary>
    string Summary { get; }

    /// <summary>Whether it reads the file or folder at <paramref name="path"/>.</summary>
    bool Recognizes(string path);

    /// <summary>The records at <paramref name="path"/>, read as they are enumerated.</summary>
    /// <param name="path">The file or folder.</param>
    /// <param name="prompt">The question a written record asks, for formats whose records have none (plain text).</param>
    /// <param name="answerKey">The JSON key a plain text's answer is written under; null for the text itself.</param>
    /// <exception cref="InvalidDataException">A record is not one it reads; the message names it.</exception>
    IEnumerable<SyntheticRecord> Read(string path, string prompt, string? answerKey);
}

/// <summary>
/// The record formats synthetic pages are made from, by name. Built in: "sharegpt" (LlamaFactory's
/// <c>{"conversations", "images"}</c>, JSON Lines or an array), "messages" (<c>{"messages": [{"role", "content"}],
/// "images"}</c>) and "text" (a .txt file of documents separated by lines of <c>---</c>, or a folder of .txt files, one
/// document each; written as ShareGPT). Register another with <see cref="Register"/>.
/// </summary>
public static class SyntheticRecordFormats
{
    private static readonly SlotTable<string, ISyntheticRecordFormat> Table = BuiltIn();

    private static SlotTable<string, ISyntheticRecordFormat> BuiltIn()
    {
        var table = new SlotTable<string, ISyntheticRecordFormat>(nameof(SyntheticRecordFormats), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a file is read as a stream of records, so a reading cannot change format half-way");
        table.RegisterDefault("sharegpt", new ShareGptSyntheticRecords());
        table.RegisterDefault("messages", new MessagesSyntheticRecords());
        table.RegisterDefault("text", new TextSyntheticRecords());
        return table;
    }

    /// <summary>Registers <paramref name="format"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISyntheticRecordFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrWhiteSpace(format.Name);
        Table.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The format of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ISyntheticRecordFormat Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(SyntheticRecordFormats), name, Table.Keys,
            $"Synthetic record format '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with SyntheticRecordFormats.Register.");

    /// <summary>The library's format of that name, or null.</summary>
    public static ISyntheticRecordFormat? Default(string name) => Table.Default(name);

    /// <summary>Who registered the format of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The first registered format that reads <paramref name="path"/>.</summary>
    /// <exception cref="FileNotFoundException">There is no file at <paramref name="path"/>.</exception>
    /// <exception cref="InvalidDataException">None does; the message names them.</exception>
    public static ISyntheticRecordFormat Detect(string path) => !File.Exists(path)
        ? throw new FileNotFoundException($"The records file {path} is not there.", path)
        : Table.Values.FirstOrDefault(f => f.Recognizes(path)) ?? throw new InvalidDataException($"No synthetic record format reads {path} (registered: {string.Join(", ", Table.Keys)}); name one, or register one with SyntheticRecordFormats.Register.");
}

/// <summary>What a page part's slot is filled from: the page's random numbers, its date, and its record's answer fields.</summary>
public sealed class PagePartContext
{
    /// <summary>The page's random numbers (drawn from its seed, so the same page fills the same).</summary>
    public required Random Random { get; init; }

    /// <summary>The page's date, drawn once, so every date slot on a page agrees.</summary>
    public required DateTime Date { get; init; }

    /// <summary>The record answer's top-level text and number fields, by key.</summary>
    public required IReadOnlyDictionary<string, string> Fields { get; init; }

    /// <summary>The folder relative paths in slots are read from (the parts file's).</summary>
    public string? Folder { get; init; }
}

/// <summary>
/// Fills one kind of slot in a page part's text, <c>{NAME}</c> or <c>{NAME:ARGUMENT}</c> ("{digits:6}"). Register one with
/// <see cref="PagePartFillers.Register"/>.
/// </summary>
public interface IPagePartFiller
{
    /// <summary>The slot's name ("digits").</summary>
    string Name { get; }

    /// <summary>One line on what it writes and what its argument is.</summary>
    string Summary { get; }

    /// <summary>The slot's text, or null when it has none (the line holding it is left out).</summary>
    /// <exception cref="ArgumentException">The argument is not one it takes.</exception>
    string? Fill(string? argument, PagePartContext context);
}

/// <summary>
/// The slots of synthetic page parts, by name. Built in: <c>{digits:N}</c> (N random digits), <c>{number:MIN-MAX}</c>,
/// <c>{pick:a|b|c}</c> (one of them), <c>{field:KEY}</c> (the record answer's field; the line is left out when it has
/// none), <c>{line:FILE}</c> (a random line of a text file, relative to the parts file), <c>{date:FORMAT}</c> (the page's
/// date in the Gregorian calendar, yyyy/MM/dd unless given) and <c>{hijri:FORMAT}</c> (the same date in the Umm al-Qura
/// calendar). Register another with <see cref="Register"/>.
/// </summary>
public static class PagePartFillers
{
    private static readonly SlotTable<string, IPagePartFiller> Table = BuiltIn();

    private static SlotTable<string, IPagePartFiller> BuiltIn()
    {
        var table = new SlotTable<string, IPagePartFiller>(nameof(PagePartFillers), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a slot's text is drawn from the page's random numbers, so another filler cannot give the same page");
        foreach (var filler in LibraryPagePartFillers.All)
        {
            table.RegisterDefault(filler.Name, filler);
        }

        return table;
    }

    /// <summary>Registers <paramref name="filler"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPagePartFiller filler)
    {
        ArgumentNullException.ThrowIfNull(filler);
        ArgumentException.ThrowIfNullOrWhiteSpace(filler.Name);
        Table.Register(filler.Name, filler, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's filler of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The filler of that name, or null.</summary>
    public static IPagePartFiller? Find(string name) => Table.Find(name);

    /// <summary>The filler of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IPagePartFiller Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(PagePartFillers), name, Table.Keys,
            $"Page part slot '{{{name}}}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with PagePartFillers.Register.");

    /// <summary>The library's filler of that name, or null.</summary>
    public static IPagePartFiller? Default(string name) => Table.Default(name);

    /// <summary>Who registered the filler of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each filler as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(f => $"{f.Name}: {f.Summary}"));
}
