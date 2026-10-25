// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Datasets;
using Idrak.Models;

namespace Idrak.Text.Abstractions;

/// <summary>
/// One source of a pretraining stage (<see cref="PretrainingStage.Sources"/>): files of one kind, read by the
/// <see cref="PretrainingSources">source kind</see> named <see cref="Kind"/>, and their share of every step.
/// </summary>
/// <param name="Kind">A <see cref="PretrainingSources"/> name: "tokens" (token files) or "records" (conversations, with images).</param>
/// <param name="Paths">The files.</param>
public sealed record PretrainingSource(string Kind, IReadOnlyList<string> Paths)
{
    /// <summary>The source's share of each step's sequences, against the stage's other sources (normalized over them).</summary>
    public double Weight { get; init; } = 1;

    /// <summary>Records: the data format (a <see cref="TuningDataFormats"/> name), or null to detect it from the first record.</summary>
    public string? Format { get; init; }

    /// <summary>Records: the folder or .zip their images are in, or null for beside each file.</summary>
    public string? Images { get; init; }

    /// <summary>Records: the folder or .zip their sounds are in, or null for beside each file.</summary>
    public string? Audios { get; init; }

    /// <summary>Tokens: how ids are stored in a raw token file (a .npy file says it itself).</summary>
    public TokenType TokenType { get; init; } = TokenType.UInt16;

    /// <summary>The source as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson()
    {
        var json = new JsonObject { ["kind"] = Kind, ["paths"] = new JsonArray([.. Paths.Select(p => (JsonNode)p)]), ["weight"] = Weight };
        if (Format is not null)
        {
            json["format"] = Format;
        }

        if (Images is not null)
        {
            json["images"] = Images;
        }

        if (Audios is not null)
        {
            json["audios"] = Audios;
        }

        if (TokenType != TokenType.UInt16)
        {
            json["token_type"] = TokenType.ToString();
        }

        return json;
    }

    /// <summary>A source from its JSON (<see cref="ToJson"/>); "path" for a single file is read too.</summary>
    /// <exception cref="InvalidDataException">A key is missing or wrong.</exception>
    public static PretrainingSource FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        string kind = (string?)json["kind"] ?? throw new InvalidDataException("A pretraining source has no \"kind\" (tokens or records).");
        var paths = json["paths"] is JsonArray many ? many.Select(p => (string?)p ?? throw new InvalidDataException("A source path is not a string.")).ToList()
            : json["path"] is JsonValue one ? [(string)one!] : throw new InvalidDataException($"The {kind} source names no \"paths\".");
        var type = TokenType.UInt16;
        if ((string?)json["token_type"] is { } t && !Enum.TryParse(t, ignoreCase: true, out type))
        {
            throw new InvalidDataException($"Token type '{t}' is not one of {string.Join(", ", Enum.GetNames<TokenType>())}.");
        }

        return new PretrainingSource(kind, paths)
        {
            Weight = (double?)json["weight"] ?? 1,
            Format = (string?)json["format"],
            Images = (string?)json["images"],
            Audios = (string?)json["audios"],
            TokenType = type,
        };
    }
}

/// <summary>What a source is opened with (<see cref="IPretrainingSource.Open"/>).</summary>
/// <param name="Model">The model being trained (its tokenizer and chat template).</param>
/// <param name="MaxLength">The longest sequence in tokens: token files are cut into windows of this many, conversations shortened to fit.</param>
/// <param name="Encoder">Conversations to training sequences (the model's template and tokenizer, and its vision side when it has one), or null when the model has no chat template.</param>
public sealed record PretrainingContext(PretrainedModel Model, int MaxLength, ChatTranscriptEncoder? Encoder);

/// <summary>
/// An opened source: <see cref="Count"/> training sequences per pass over it, read by index in any order (the
/// pretraining loop shuffles them per pass from its seed and reads one at a time, on one thread at a time).
/// </summary>
public interface IPretrainingReader : IDisposable
{
    /// <summary>Sequences in one pass over the source.</summary>
    long Count { get; }

    /// <summary>The sequence at <paramref name="index"/> (the same on every call), or null when that item gives none (a conversation with nothing to learn).</summary>
    TrainingSequence? Read(long index);
}

/// <summary>A kind of pretraining data: how its files become training sequences. Register one with <see cref="PretrainingSources.Register"/>.</summary>
public interface IPretrainingSource
{
    /// <summary>The kind's name.</summary>
    string Name { get; }

    /// <summary>One line on what it reads.</summary>
    string Summary { get; }

    /// <summary>Opens <paramref name="source"/>'s files.</summary>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    /// <exception cref="InvalidDataException">A file cannot be read as this kind.</exception>
    IPretrainingReader Open(PretrainingSource source, PretrainingContext context);
}

/// <summary>
/// The kinds of pretraining data, by name. Built in: "tokens" (token files, as <c>idrak data tokens</c> writes them, cut
/// into windows of the maximum length, every token trained) and "records" (conversations in a
/// <see cref="TuningDataFormats">tuning data format</see>, with images when the model has a vision part; the answers
/// trained, the prompt and the images not).
/// </summary>
public static class PretrainingSources
{
    /// <summary>Token files.</summary>
    public const string Tokens = "tokens";

    /// <summary>Conversation records.</summary>
    public const string Records = "records";

    private static readonly SlotTable<string, IPretrainingSource> Table = BuiltIn();

    private static SlotTable<string, IPretrainingSource> BuiltIn()
    {
        var table = new SlotTable<string, IPretrainingSource>(nameof(PretrainingSources), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a source kind reads its own files into sequences, so another kind cannot read the same ones");
        table.RegisterDefault(Tokens, new TokenPretrainingSource());
        table.RegisterDefault(Records, new RecordPretrainingSource());
        return table;
    }

    /// <summary>Registers <paramref name="source"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPretrainingSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(source.Name);
        Table.Register(source.Name, source, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The kind of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IPretrainingSource Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(PretrainingSources), name, Table.Keys,
            $"Pretraining source '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with PretrainingSources.Register.");

    /// <summary>The library's kind of that name, or null.</summary>
    public static IPretrainingSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the kind of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each kind as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));
}
