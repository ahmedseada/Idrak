// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// One part of a message's content (<see cref="ChatMessage.Parts"/>): text (<see cref="ChatText"/>), an image
/// (<see cref="ChatImage"/>), or a kind of one's own (audio, a video, a document). Every part has a
/// <see cref="Kind"/>, the name of its <see cref="IChatPartKind"/>, which writes and reads it as JSON. A chat model says
/// which kinds it takes (<see cref="IChatModel.PartKinds"/>) and a chat
/// template which it renders (<see cref="ChatTemplate.PartKinds"/>); either fails on any other kind, never drops it.
/// Parts are immutable and compare by value.
/// </summary>
public abstract record ChatPart
{
    /// <summary>The part's kind: "text", "image", or the name of a kind of one's own.</summary>
    public abstract string Kind { get; }
}

/// <summary>Text in a message.</summary>
/// <param name="Text">The text.</param>
public sealed record ChatText(string Text) : ChatPart
{
    /// <summary>The text (never null).</summary>
    public string Text { get; init; } = Text ?? throw new ArgumentNullException(nameof(Text));

    /// <inheritdoc />
    public override string Kind => ChatParts.Text;
}

/// <summary>
/// An image in a message, as it was received: its encoded bytes (PNG, JPEG, ...; never decoded here, decoding is the image
/// codecs'), its media type when known, and the SHA-256 of the bytes (<see cref="Hash"/>), which identifies it:
/// two images are equal when their bytes are, so a model can encode an image once per conversation. Make one with
/// <see cref="FromFile"/>, <see cref="FromBytes"/> or <see cref="FromDataUrl"/>.
/// </summary>
public sealed record ChatImage : ChatPart
{
    private readonly byte[] _data;

    // Takes `data` as the image's own array, not a copy: callers pass an array nothing else holds.
    internal ChatImage(byte[] data, string? mediaType)
    {
        if (data.Length == 0)
        {
            throw new ArgumentException("An image needs its bytes; these are empty.", nameof(data));
        }

        _data = data;
        MediaType = string.IsNullOrWhiteSpace(mediaType) ? Sniff(data) : mediaType.Trim().ToLowerInvariant();
        Hash = Convert.ToHexStringLower(SHA256.HashData(data));
    }

    /// <inheritdoc />
    public override string Kind => ChatParts.Image;

    /// <summary>The encoded bytes, as received.</summary>
    public ReadOnlyMemory<byte> Data => _data;

    /// <summary>
    /// The media type ("image/png"): as given, else read from the bytes' signature (PNG, JPEG, GIF, BMP, WebP, TIFF); null
    /// when neither says.
    /// </summary>
    public string? MediaType { get; }

    /// <summary>The SHA-256 of <see cref="Data"/>, 64 lowercase hex digits: the image's identity.</summary>
    public string Hash { get; }

    /// <summary>The image in the file at <paramref name="path"/> (its bytes as they are; the media type from their signature).</summary>
    public static ChatImage FromFile(string path, string? mediaType = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Image file not found: {path}", path);
        }

        byte[] data = File.ReadAllBytes(path);
        return data.Length > 0 ? new ChatImage(data, mediaType) : throw new ArgumentException($"The image file {path} is empty.", nameof(path));
    }

    /// <summary>The image encoded in <paramref name="data"/> (copied), of <paramref name="mediaType"/> when given.</summary>
    public static ChatImage FromBytes(ReadOnlySpan<byte> data, string? mediaType = null) => new(data.ToArray(), mediaType);

    /// <summary>
    /// The image in the next <paramref name="length"/> bytes of <paramref name="stream"/> (a zip entry, an upload whose
    /// length is known), read straight into the image's own array: one copy of the bytes, where reading them into a buffer
    /// and then <see cref="FromBytes"/> makes two.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is not positive.</exception>
    /// <exception cref="EndOfStreamException">The stream ends before <paramref name="length"/> bytes.</exception>
    public static ChatImage FromStream(Stream stream, int length, string? mediaType = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var data = GC.AllocateUninitializedArray<byte>(length);               // every byte is read over
        stream.ReadExactly(data);
        return new ChatImage(data, mediaType);
    }

    /// <summary>
    /// The image in a base64 data URL, <c>data:image/png;base64,iVBOR…</c> (as OpenAI's <c>image_url</c> parts carry it).
    /// The media type must be an image type (or left out); the data must be base64.
    /// </summary>
    /// <exception cref="FormatException">The text is not a base64 data URL of an image.</exception>
    public static ChatImage FromDataUrl(string url)
    {
        ArgumentNullException.ThrowIfNull(url);
        string text = url.Trim();
        int comma = text.IndexOf(',', StringComparison.Ordinal);
        if (!text.StartsWith("data:", StringComparison.OrdinalIgnoreCase) || comma < 0)
        {
            throw new FormatException($"Not a data URL (data:image/png;base64,...): {Shorten(text)}");
        }

        string[] header = text[5..comma].Split(';', StringSplitOptions.TrimEntries);
        if (!header.Skip(1).Contains("base64", StringComparer.OrdinalIgnoreCase))
        {
            throw new FormatException($"An image data URL must be base64 (data:image/png;base64,...): {Shorten(text)}");
        }

        string? mediaType = header[0].Length == 0 ? null : header[0];
        if (mediaType is not null && !mediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
        {
            throw new FormatException($"The data URL holds '{mediaType}', not an image: {Shorten(text)}");
        }

        byte[] data;
        try
        {
            data = Convert.FromBase64String(text[(comma + 1)..]);
        }
        catch (FormatException)
        {
            throw new FormatException($"The data URL's base64 does not decode: {Shorten(text)}");
        }

        return data.Length > 0 ? new ChatImage(data, mediaType) : throw new FormatException("The data URL holds no bytes.");
    }

    /// <summary>The image as a base64 data URL (<c>data:image/png;base64,…</c>; application/octet-stream when the media type is unknown).</summary>
    public string ToDataUrl() => $"data:{MediaType ?? "application/octet-stream"};base64,{Convert.ToBase64String(_data)}";

    /// <summary>Two images are equal when their bytes are (their <see cref="Hash"/>es match), whatever media type they were given.</summary>
    public bool Equals(ChatImage? other) => other is not null && string.Equals(Hash, other.Hash, StringComparison.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode() => Hash.GetHashCode(StringComparison.Ordinal);

    /// <summary>The media type, size and the start of the hash: "image/png, 12345 bytes, sha256 3f2a9c1b…".</summary>
    public override string ToString() => $"{MediaType ?? "image"}, {_data.Length} bytes, sha256 {Hash[..8]}…";

    private static string Shorten(string text) => text.Length <= 48 ? text : text[..48] + "…";

    // The media type from the bytes' signature; null when it is none of the common image formats.
    private static string? Sniff(ReadOnlySpan<byte> data) => data switch
    {
        [0x89, (byte)'P', (byte)'N', (byte)'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [(byte)'G', (byte)'I', (byte)'F', (byte)'8', ..] => "image/gif",
        [(byte)'B', (byte)'M', ..] => "image/bmp",
        [(byte)'R', (byte)'I', (byte)'F', (byte)'F', _, _, _, _, (byte)'W', (byte)'E', (byte)'B', (byte)'P', ..] => "image/webp",
        [(byte)'I', (byte)'I', 42, 0, ..] or [(byte)'M', (byte)'M', 0, 42, ..] => "image/tiff",
        _ => null,
    };
}

/// <summary>
/// A kind of message part: how parts of this kind are written to and read from the chat JSON (the content-part objects
/// of saved conversations, <c>{"type": name, …}</c>). The library's are "text" and "image"; a plug-in makes its own
/// (audio, video, documents) with its own <see cref="ChatPart"/> type for them. Its <c>Name</c> is the
/// <see cref="ChatPart.Kind"/> of its parts and the "type" of their JSON. Finding a kind by its name is not on the new
/// design yet.
/// </summary>
public interface IChatPartKind
{
    /// <summary>Its name: the <see cref="ChatPart.Kind"/> of its parts and the "type" of their JSON ("text", "image").</summary>
    string Name { get; }

    /// <summary>Writes the fields of <paramref name="part"/> (one of this kind) into <paramref name="json"/>, which already holds its "type".</summary>
    void Write(ChatPart part, JsonObject json);

    /// <summary>The part <paramref name="json"/> (a <c>{"type": name, …}</c> object) describes; a clear exception when it is not valid.</summary>
    ChatPart Read(JsonObject json);
}

/// <summary>
/// A kind of message part whose parts are made from bytes as received (a sound, a video, a document): what a bridge
/// that holds the bytes of a block it cannot name the type of (the MCP bridge's audio blocks, a tuning data file's
/// <c>"audios"</c> list) makes a part with. Finding the kind by its name alone is not on the new design yet.
/// </summary>
public interface IBinaryPartKind : IChatPartKind
{
    /// <summary>A part of this kind holding a copy of <paramref name="data"/>, of <paramref name="mediaType"/> when given (else read from the bytes when the kind can).</summary>
    /// <exception cref="ArgumentException">The bytes are empty.</exception>
    ChatPart FromBytes(ReadOnlySpan<byte> data, string? mediaType = null);

    /// <summary>
    /// The keys data records and request bodies name parts of this kind by (a tuning record's <c>"audios"</c> list and
    /// <c>&lt;audio&gt;</c> markers, a completions request's <c>"input_audio"</c> part), so the readers find them through
    /// the kind instead of names of their own.
    /// </summary>
    BinaryPartRecordKeys RecordKeys { get; }
}

/// <summary>
/// The keys under which data records and request bodies name parts of a binary kind (<see cref="IBinaryPartKind.RecordKeys"/>).
/// </summary>
/// <param name="List">A data record's list of the parts' references, in order ("audios"): each placeholder takes the next.</param>
/// <param name="Marker">The marker in a message's text that stands for the next entry of <paramref name="List"/> ("&lt;audio&gt;").</param>
/// <param name="InputPart">
/// The type of a request's content part that carries a part's bytes as base64 with a format
/// (<c>{"type": "input_audio", "input_audio": {"data", "format"}}</c>); null when requests have none.
/// </param>
public sealed record BinaryPartRecordKeys(string List, string Marker, string? InputPart = null)
{
    /// <summary>
    /// The media types of the formats an <see cref="InputPart"/> names ("mp3" to "audio/mpeg"), by format in lower case;
    /// a format not listed is taken as a media type when it holds a '/', else as the kind's own subtype.
    /// </summary>
    public IReadOnlyDictionary<string, string> Formats { get; init; } = new Dictionary<string, string>();
}

/// <summary>
/// The kinds of message parts and what every chat model and template shares about them: the check that a model or
/// template takes every part it is given (<see cref="ThrowIfUnsupported(IChatModel, ChatRequest, string?)"/>). The kinds:
/// <list type="bullet">
/// <item>"text" (<see cref="ChatText"/>): <c>{"type": "text", "text": "…"}</c>.</item>
/// <item>"image" (<see cref="ChatImage"/>): <c>{"type": "image", "media_type": "image/png", "data": "&lt;base64&gt;"}</c>.</item>
/// <item>"audio" (Idrak.Audio's <c>ChatAudio</c>):
/// <c>{"type": "audio", "media_type": "audio/wav", "data": "&lt;base64&gt;"}</c>.</item>
/// </list>
/// </summary>
public static class ChatParts
{
    /// <summary>The kind of text parts.</summary>
    public const string Text = "text";

    /// <summary>The kind of image parts.</summary>
    public const string Image = "image";

    /// <summary>The kind of audio parts (Idrak.Audio's <c>ChatAudio</c>; a library default only when that package is shipped).</summary>
    public const string Audio = "audio";

    /// <summary>The kinds a text-only model or template takes: "text". The default of <see cref="IChatModel.PartKinds"/>.</summary>
    public static IReadOnlySet<string> TextOnly { get; } = FrozenSet.Create(StringComparer.Ordinal, Text);

    /// <summary>
    /// Throws when <paramref name="request"/> has a part of a kind <paramref name="model"/> does not take
    /// (<see cref="IChatModel.PartKinds"/>): every chat model of the library calls it before it answers, so a text-only
    /// model given an image fails with a message saying which part, and does not answer as if it were not there.
    /// </summary>
    /// <param name="model">The model asked.</param>
    /// <param name="request">The request.</param>
    /// <param name="name">The model's name in the message (its type's name when null).</param>
    /// <exception cref="NotSupportedException">A part of a kind the model does not take.</exception>
    public static void ThrowIfUnsupported(IChatModel model, ChatRequest request, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(request);
        ThrowIfUnsupported(request.Messages, model.PartKinds, $"The chat model {(name is null ? model.GetType().Name : $"'{name}'")}");
    }

    /// <summary>
    /// Throws when a message of <paramref name="messages"/> has a part whose kind is not in <paramref name="kinds"/>, naming
    /// <paramref name="who"/> ("The chat template ChatMLTemplate"), the message and the part's kind.
    /// </summary>
    /// <exception cref="NotSupportedException">A part of a kind not in <paramref name="kinds"/>.</exception>
    public static void ThrowIfUnsupported(IReadOnlyList<ChatMessage> messages, IReadOnlySet<string> kinds, string who)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(kinds);
        for (int i = 0; i < messages.Count; i++)
        {
            foreach (var part in messages[i].Parts)
            {
                if (!kinds.Contains(part.Kind))
                {
                    throw new NotSupportedException(
                        $"{who} takes only {string.Join(", ", kinds.Order(StringComparer.Ordinal).Select(k => $"'{k}'"))} parts, but message {i + 1} " +
                        $"({messages[i].Role}) has a part of kind '{part.Kind}' ({part}). Use a model that takes '{part.Kind}' parts, or remove them from the conversation.");
                }
            }
        }
    }

    internal static T Of<T>(ChatPart part, string kind) where T : ChatPart =>
        part as T ?? throw new ArgumentException($"The '{kind}' kind writes {typeof(T).Name} parts, not {part.GetType().Name}.", nameof(part));
}

internal sealed class TextKind : IChatPartKind
{
    public string Name => ChatParts.Text;

    public void Write(ChatPart part, JsonObject json) => json["text"] = ChatParts.Of<ChatText>(part, Name).Text;

    public ChatPart Read(JsonObject json) =>
        new ChatText(json["text"] is JsonValue v && v.TryGetValue(out string? text) ? text : throw new FormatException($"A text part needs a \"text\" string: {json.ToJsonString()}"));
}

internal sealed class ImageKind : IChatPartKind
{
    public string Name => ChatParts.Image;

    public void Write(ChatPart part, JsonObject json)
    {
        var image = ChatParts.Of<ChatImage>(part, Name);
        if (image.MediaType is { } mediaType)
        {
            json["media_type"] = mediaType;
        }

        json["data"] = Convert.ToBase64String(image.Data.Span);
    }

    public ChatPart Read(JsonObject json)
    {
        string? mediaType = (string?)json["media_type"];
        if ((string?)json["data"] is not { Length: > 0 } data)
        {
            throw new FormatException("An image part needs its bytes as base64 in \"data\".");
        }

        try
        {
            return new ChatImage(Convert.FromBase64String(data), mediaType);   // the decoded array is the image's own: no second copy
        }
        catch (FormatException)
        {
            throw new FormatException("The \"data\" of an image part is not valid base64.");
        }
    }
}
