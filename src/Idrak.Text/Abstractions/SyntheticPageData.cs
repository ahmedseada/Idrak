// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Text.Abstractions;

/// <summary>
/// One record a synthetic page is drawn from: the answer it was tuned to give (whose page text the page draws) and how to
/// write the record again with a new answer and image (<see cref="Write"/>).
/// </summary>
public interface ISyntheticRecord
{
    /// <summary>The record's answer as written: JSON (in a code fence or not) or plain text.</summary>
    string Answer { get; }

    /// <summary>Where the record is, for messages ("train.jsonl line 12").</summary>
    string Where { get; }

    /// <summary>The record again, with <paramref name="answer"/> as its answer and <paramref name="image"/> (a relative path) as its one image.</summary>
    JsonObject Write(string answer, string image);
}

/// <summary>A record whose writing is a function (<see cref="ISyntheticRecord"/>): the library's formats make these.</summary>
public sealed class SyntheticRecord : ISyntheticRecord
{
    /// <inheritdoc />
    public required string Answer { get; init; }

    /// <inheritdoc />
    public required string Where { get; init; }

    /// <summary>Writes the record again with an answer and an image (<see cref="Write"/>).</summary>
    public required Func<string, string, JsonObject> Writer { get; init; }

    /// <inheritdoc />
    public JsonObject Write(string answer, string image) => Writer(answer, image);
}

/// <summary>
/// A kind of file that synthetic pages are made from (plan 18): its records' answers in, records with new answers and
/// images out, in the same shape.
/// </summary>
public interface ISyntheticRecordFormat
{
    /// <summary>The format's name ("sharegpt").</summary>
    string Name { get; }

    /// <summary>Whether it reads the file or folder at <paramref name="path"/>.</summary>
    bool Recognizes(string path);

    /// <summary>The records at <paramref name="path"/>, read as they are enumerated.</summary>
    /// <param name="path">The file or folder.</param>
    /// <param name="prompt">The question a written record asks, for formats whose records have none (plain text).</param>
    /// <param name="answerKey">The JSON key a plain text's answer is written under; null for the text itself.</param>
    /// <exception cref="InvalidDataException">A record is not one it reads; the message names it.</exception>
    IEnumerable<ISyntheticRecord> Read(string path, string prompt, string? answerKey);
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
/// Fills one kind of slot in a page part's text, <c>{NAME}</c> or <c>{NAME:ARGUMENT}</c> ("{digits:6}").
/// </summary>
public interface IPagePartFiller
{
    /// <summary>The slot's name ("digits").</summary>
    string Name { get; }

    /// <summary>The slot's text, or null when it has none (the line holding it is left out).</summary>
    /// <exception cref="ArgumentException">The argument is not one it takes.</exception>
    string? Fill(string? argument, PagePartContext context);
}

