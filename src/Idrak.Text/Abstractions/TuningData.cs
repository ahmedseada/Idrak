// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Text.Abstractions;

/// <summary>Where a tuning data file's images and sounds are found (<see cref="ITuningDataFormat.Read"/>).</summary>
public sealed record TuningDataOptions
{
    /// <summary>
    /// Where the data's relative image paths are looked up: a folder, or a <c>.zip</c> file opened read-only, of which only
    /// the image entries the data names are read (nothing is unpacked to disk, and no other entry is ever opened). Null:
    /// the data file's folder. Absolute paths in the data are read as they are.
    /// </summary>
    public string? Images { get; init; }

    /// <summary>
    /// Where the data's relative audio paths (an <c>"audios"</c> list, <c>&lt;audio&gt;</c> markers) are looked up: a
    /// folder or a <c>.zip</c> file, as <see cref="Images"/>. Null: where the images are (<see cref="Images"/>, else the
    /// data file's folder). Audio parts are made by the "audio" kind (Idrak.Audio's);
    /// a file with audio and no such kind is refused naming it.
    /// </summary>
    public string? Audios { get; init; }

    /// <summary>
    /// The "transcripts" format: the user's words after each sound (the owner's prompt for an audio-language model: what it
    /// is asked to do with the sound); empty for the sound alone. "Transcribe." when null.
    /// </summary>
    public string? Prompt { get; init; }
}

/// <summary>
/// A file format of fine-tuning conversations: reads a data file into <see cref="ChatTranscript"/>s whose messages carry
/// the data's images as image parts (<see cref="ChatImage"/>), in place and in order. The testing kit checks one
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

    /// <summary>Whether <paramref name="record"/> (a data file's first record) is one of this format's.</summary>
    bool Recognizes(JsonObject record);

    /// <summary>The transcripts of the data file at <paramref name="path"/>, one per record, in the file's order.</summary>
    /// <param name="path">The data file.</param>
    /// <param name="options">Where its images are (next to the file when null).</param>
    /// <exception cref="FileNotFoundException">The file, or an image it names, is not there.</exception>
    /// <exception cref="InvalidDataException">A record is not valid, or its images do not match its placeholders; the message names the file and the record.</exception>
    IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options = null);

    /// <summary>
    /// The data file's conversations read and checked as <see cref="Read"/> reads them (every record, the count of its
    /// images against its placeholders), each one's images read only when it is taken (<see cref="PendingConversations.Read"/>):
    /// a tuner that holds some conversations out or trains on the first N reads, inflates and hashes the images of the
    /// conversations it takes, and no others. The default reads every conversation with <see cref="Read"/>, images and all;
    /// the library's formats leave the images for later.
    /// </summary>
    /// <param name="path">The data file.</param>
    /// <param name="options">Where its images are (next to the file when null).</param>
    /// <exception cref="FileNotFoundException">The file is not there (an image that is not there fails when its conversation is taken).</exception>
    /// <exception cref="InvalidDataException">A record is not valid, or its images do not match its placeholders; the message names the file and the record.</exception>
    PendingConversations ReadPending(string path, TuningDataOptions? options = null) => new([.. Read(path, options)]);
}

/// <summary>
/// A data file's conversations, read and checked, whose images are read when each conversation is taken
/// (<see cref="ITuningDataFormat.ReadPending"/>). It keeps what the images come from (an images zip) open until disposed.
/// Conversations are taken on one thread at a time.
/// </summary>
public sealed class PendingConversations : IDisposable
{
    private readonly Func<int, ChatTranscript> _read;
    private readonly IDisposable? _source;

    /// <summary>Conversations read already, their images with them.</summary>
    public PendingConversations(IReadOnlyList<ChatTranscript> transcripts)
    {
        ArgumentNullException.ThrowIfNull(transcripts);
        (Count, _read) = (transcripts.Count, i => transcripts[i]);
        foreach (var transcript in transcripts)
        {
            foreach (var message in transcript.Messages)
            {
                Images += message.Parts.Count(p => p is ChatImage);
            }
        }
    }

    /// <summary>
    /// <paramref name="count"/> conversations holding <paramref name="images"/> image parts in all, each taken by
    /// <paramref name="read"/> (its index in, the conversation with its images read out); <paramref name="source"/>, when
    /// given, is disposed with this.
    /// </summary>
    public PendingConversations(int count, int images, Func<int, ChatTranscript> read, IDisposable? source = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegative(images);
        (Count, Images, _read, _source) = (count, images, read ?? throw new ArgumentNullException(nameof(read)), source);
    }

    /// <summary>The conversations, one per record, in the file's order.</summary>
    public int Count { get; }

    /// <summary>The image parts of every conversation together, read or not.</summary>
    public int Images { get; }

    /// <summary>The conversation at <paramref name="index"/> (in the file's order), its images read now (again on each call).</summary>
    /// <exception cref="FileNotFoundException">An image it names is not there.</exception>
    /// <exception cref="InvalidDataException">An image it names cannot be read; the message names the file and the record.</exception>
    public ChatTranscript Read(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Count);
        return _read(index);
    }

    /// <summary>Closes what the images come from.</summary>
    public void Dispose() => _source?.Dispose();
}

/// <summary>
/// The names of the file formats of fine-tuning data. The library's:
/// <list type="bullet">
/// <item><see cref="Messages"/>: JSON Lines of <c>{"messages": [{"role", "content"}], "tools"}</c> (OpenAI / Hugging Face
/// chat; see <see cref="ChatTranscript.FromJson"/>). Image parts are the chat JSON's (<c>{"type": "image", "data":
/// base64}</c>), a path (<c>{"type": "image", "path" | "image": "page.png"}</c>, <c>{"type": "image_url",
/// "image_url": {"url": "page.png" | "data:..."}}</c>), or bare (<c>{"type": "image"}</c>) taking the record's next
/// <c>"images"</c> entry; with an <c>"images"</c> list, each <c>&lt;image&gt;</c> in a text takes the next entry too.
/// Audio parts the same way (<c>{"type": "audio", "path": "a.wav"}</c>, bare ones and <c>&lt;audio&gt;</c> markers
/// taking the record's <c>"audios"</c> list).</item>
/// <item><see cref="ShareGpt"/>: ShareGPT's <c>{"conversations": [{"from", "value"}], "system", "images", "audios"}</c> as
/// LlamaFactory writes it: each <c>&lt;image&gt;</c> (<c>&lt;audio&gt;</c>) in a turn's text becomes the next entry of
/// <c>"images"</c> (<c>"audios"</c>), in place and in order, and the counts must agree.</item>
/// </list>
/// Both read JSON Lines or a JSON array (LlamaFactory's <c>train.json</c>), streamed record by record. Image paths are
/// relative to the data file's folder, or to <see cref="TuningDataOptions.Images"/> (a folder or a zip).
/// </summary>
public static class TuningDataFormats
{
    /// <summary>JSON Lines of chat messages (today's fine-tuning data), with image parts.</summary>
    public const string Messages = "messages";

    /// <summary>ShareGPT conversations with LlamaFactory's <c>"images"</c> list and <c>&lt;image&gt;</c> placeholders.</summary>
    public const string ShareGpt = "sharegpt";

    /// <summary>Speech transcripts, <c>{"audio", "text"}</c> per line: each a conversation of the sound and the prompt (<see cref="TuningDataOptions.Prompt"/>), answered by the text.</summary>
    public const string Transcripts = "transcripts";
}
