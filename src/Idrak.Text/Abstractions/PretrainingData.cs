// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
        return new PretrainingSource(kind, paths)
        {
            Weight = (double?)json["weight"] ?? 1,
            Format = (string?)json["format"],
            Images = (string?)json["images"],
            Audios = (string?)json["audios"],
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

/// <summary>A kind of pretraining data: how its files become training sequences. Register one with <c>PretrainingSources.Register</c>.</summary>
public interface IPretrainingSource
{
    /// <summary>The kind's name.</summary>
    string Name { get; }

    /// <summary>Opens <paramref name="source"/>'s files.</summary>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    /// <exception cref="InvalidDataException">A file cannot be read as this kind.</exception>
    IPretrainingReader Open(PretrainingSource source, PretrainingContext context);
}

/// <summary>
/// The names of the kinds of pretraining data. Built in: "tokens" (token files, as <c>idrak data tokens</c> writes them, cut
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
}
