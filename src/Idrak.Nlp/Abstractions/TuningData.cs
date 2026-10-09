// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Nlp.Abstractions;

/// <summary>Where a tuning data file's images are found (<see cref="ITuningDataFormat.Read"/>).</summary>
public sealed record TuningDataOptions
{
    /// <summary>
    /// Where the data's relative image paths are looked up: a folder, or a <c>.zip</c> file opened read-only, of which only
    /// the image entries the data names are read (nothing is unpacked to disk, and no other entry is ever opened). Null:
    /// the data file's folder. Absolute paths in the data are read as they are.
    /// </summary>
    public string? Images { get; init; }
}

/// <summary>
/// A file format of fine-tuning conversations: reads a data file into <see cref="ChatTranscript"/>s whose messages carry
/// the data's images as image parts (<see cref="ChatImage"/>), in place and in order. Registered in
/// <see cref="TuningDataFormats"/> under its <see cref="Name"/>; the testing kit checks one
/// (<c>Conformance.CheckTuningDataFormat</c>).
/// <para>
/// Reading is lazy: records are read as the result is enumerated, each image's encoded bytes as its record is read, and
/// nothing is decoded (decoding is the tuner's, when it needs the pixels), so a large file is never held whole. A record
/// that cannot be read is an <see cref="InvalidDataException"/> naming the file and the record.
/// </para>
/// </summary>
public interface ITuningDataFormat
{
    /// <summary>The format's name in <see cref="TuningDataFormats"/> ("messages", "sharegpt").</summary>
    string Name { get; }

    /// <summary>One line for help and listings: what the format's records look like.</summary>
    string Summary { get; }

    /// <summary>Whether <paramref name="record"/> (a data file's first record) is one of this format's (<see cref="TuningDataFormats.Detect"/>).</summary>
    bool Recognizes(JsonObject record);

    /// <summary>The transcripts of the data file at <paramref name="path"/>, one per record, in the file's order.</summary>
    /// <param name="path">The data file.</param>
    /// <param name="options">Where its images are (next to the file when null).</param>
    /// <exception cref="FileNotFoundException">The file, or an image it names, is not there.</exception>
    /// <exception cref="InvalidDataException">A record is not valid, or its images do not match its placeholders; the message names the file and the record.</exception>
    IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options = null);
}

/// <summary>
/// The file formats of fine-tuning data, by name (ignoring case). The library's:
/// <list type="bullet">
/// <item><see cref="Messages"/>: JSON Lines of <c>{"messages": [{"role", "content"}], "tools"}</c> (OpenAI / Hugging Face
/// chat; see <see cref="ChatTranscript.FromJson"/>). Image parts are the chat JSON's (<c>{"type": "image", "data":
/// base64}</c>), a path (<c>{"type": "image", "path" | "image": "page.png"}</c>, <c>{"type": "image_url",
/// "image_url": {"url": "page.png" | "data:..."}}</c>), or bare (<c>{"type": "image"}</c>) taking the record's next
/// <c>"images"</c> entry; with an <c>"images"</c> list, each <c>&lt;image&gt;</c> in a text takes the next entry too.</item>
/// <item><see cref="ShareGpt"/>: ShareGPT's <c>{"conversations": [{"from", "value"}], "system", "images"}</c> as
/// LlamaFactory writes it: each <c>&lt;image&gt;</c> in a turn's text becomes the next entry of <c>"images"</c>, in
/// place and in order, and the counts must agree.</item>
/// </list>
/// Both read JSON Lines or a JSON array (LlamaFactory's <c>train.json</c>), streamed record by record. Image paths are
/// relative to the data file's folder, or to <see cref="TuningDataOptions.Images"/> (a folder or a zip). Another format is
/// a registration (<see cref="Register"/>); one under a library name shadows the library's, which
/// <see cref="Unregister"/> brings back.
/// </summary>
public static class TuningDataFormats
{
    /// <summary>JSON Lines of chat messages (today's fine-tuning data), with image parts.</summary>
    public const string Messages = "messages";

    /// <summary>ShareGPT conversations with LlamaFactory's <c>"images"</c> list and <c>&lt;image&gt;</c> placeholders.</summary>
    public const string ShareGpt = "sharegpt";

    private static readonly SlotTable<string, ITuningDataFormat> Table = BuiltIn();

    private static SlotTable<string, ITuningDataFormat> BuiltIn()
    {
        var table = new SlotTable<string, ITuningDataFormat>(nameof(TuningDataFormats), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a data file is read as a stream of records, so a reading cannot change format half-way");
        table.RegisterDefault(Messages, new MessagesDataFormat());
        table.RegisterDefault(ShareGpt, new ShareGptDataFormat());
        return table;
    }

    /// <summary>Registers <paramref name="format"/> under its <see cref="ITuningDataFormat.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITuningDataFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrWhiteSpace(format.Name);
        Table.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The format registered as <paramref name="name"/>, or null.</summary>
    public static ITuningDataFormat? Find(string name) => Table.Find(name);

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITuningDataFormat Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Tuning data format '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TuningDataFormats.Register.");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITuningDataFormat? Default(string name) => Table.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered format as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(f => $"{f.Name}: {f.Summary}"));

    /// <summary>
    /// The format of the data file at <paramref name="path"/>: the first registered one that recognizes its first record
    /// (JSON Lines or a JSON array).
    /// </summary>
    /// <exception cref="InvalidDataException">The file has no record, or no registered format recognizes it (the message names them).</exception>
    public static ITuningDataFormat Detect(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var (first, where) = TuningRecords.Read(path).Select(r => (r.Record, r.Where)).FirstOrDefault();
        if (first is null)
        {
            throw new InvalidDataException($"{path} holds no record to tell its format from.");
        }

        return Table.Values.FirstOrDefault(f => f.Recognizes(first))
            ?? throw new InvalidDataException($"{where}: no registered tuning data format recognizes the record (registered: {string.Join(", ", Table.Keys)}); "
                + "name the format, or register one with TuningDataFormats.Register.");
    }

    /// <summary>
    /// The transcripts of the data file at <paramref name="path"/> read by the format <paramref name="format"/>, or by the
    /// one <see cref="Detect"/> finds when it is null.
    /// </summary>
    public static IEnumerable<ChatTranscript> Read(string path, string? format = null, TuningDataOptions? options = null) =>
        (format is null ? Detect(path) : Get(format)).Read(path, options);
}
