// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// A sound in a message, as it was received: its encoded bytes (WAV, FLAC, ...; never decoded here, decoding is
/// <see cref="AudioCodecs"/>'s, see <see cref="Decode"/>), its media type when known, and the SHA-256 of the bytes
/// (<see cref="Hash"/>), which identifies it: two sounds are equal when their bytes are, so a model encodes a sound once
/// per conversation. Make one with <see cref="FromFile"/>, <see cref="FromBytes"/>, <see cref="FromStream"/> or
/// <see cref="FromDataUrl"/>. Its kind is <see cref="ChatParts.Audio"/>, registered in <see cref="ChatParts"/> by this
/// package as a library default (<c>{"type": "audio", "media_type": "audio/wav", "data": "&lt;base64&gt;"}</c>).
/// </summary>
public sealed record ChatAudio : ChatPart
{
    private readonly byte[] _data;

    // Takes `data` as the sound's own array, not a copy: callers pass an array nothing else holds.
    internal ChatAudio(byte[] data, string? mediaType)
    {
        if (data.Length == 0)
        {
            throw new ArgumentException("A sound needs its bytes; these are empty.", nameof(data));
        }

        _data = data;
        MediaType = string.IsNullOrWhiteSpace(mediaType) ? Sniff(data) : mediaType.Trim().ToLowerInvariant();
        Hash = Convert.ToHexStringLower(SHA256.HashData(data));
    }

    /// <inheritdoc />
    public override string Kind => ChatParts.Audio;

    /// <summary>The encoded bytes, as received.</summary>
    public ReadOnlyMemory<byte> Data => _data;

    /// <summary>
    /// The media type ("audio/wav"): as given, else read from the bytes' signature (RIFF/WAVE, FLAC, MP3 (an ID3 tag or a
    /// frame sync), Ogg, MP4 audio); null when neither says.
    /// </summary>
    public string? MediaType { get; }

    /// <summary>The SHA-256 of <see cref="Data"/>, 64 lowercase hex digits: the sound's identity.</summary>
    public string Hash { get; }

    /// <summary>The sound in the file at <paramref name="path"/> (its bytes as they are; the media type from their signature).</summary>
    public static ChatAudio FromFile(string path, string? mediaType = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Audio file not found: {path}", path);
        }

        byte[] data = File.ReadAllBytes(path);
        return data.Length > 0 ? new ChatAudio(data, mediaType) : throw new ArgumentException($"The audio file {path} is empty.", nameof(path));
    }

    /// <summary>The sound encoded in <paramref name="data"/> (copied), of <paramref name="mediaType"/> when given.</summary>
    public static ChatAudio FromBytes(ReadOnlySpan<byte> data, string? mediaType = null) => new(data.ToArray(), mediaType);

    /// <summary>
    /// The sound in the next <paramref name="length"/> bytes of <paramref name="stream"/> (a zip entry, an upload whose
    /// length is known), read straight into the sound's own array: one copy of the bytes.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is not positive.</exception>
    /// <exception cref="EndOfStreamException">The stream ends before <paramref name="length"/> bytes.</exception>
    public static ChatAudio FromStream(Stream stream, int length, string? mediaType = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var data = GC.AllocateUninitializedArray<byte>(length);               // every byte is read over
        stream.ReadExactly(data);
        return new ChatAudio(data, mediaType);
    }

    /// <summary>
    /// The sound in a base64 data URL, <c>data:audio/wav;base64,UklGR…</c>. The media type must be an audio type (or left
    /// out); the data must be base64.
    /// </summary>
    /// <exception cref="FormatException">The text is not a base64 data URL of a sound.</exception>
    public static ChatAudio FromDataUrl(string url)
    {
        ArgumentNullException.ThrowIfNull(url);
        string text = url.Trim();
        int comma = text.IndexOf(',', StringComparison.Ordinal);
        if (!text.StartsWith("data:", StringComparison.OrdinalIgnoreCase) || comma < 0)
        {
            throw new FormatException($"Not a data URL (data:audio/wav;base64,...): {Shorten(text)}");
        }

        string[] header = text[5..comma].Split(';', StringSplitOptions.TrimEntries);
        if (!header.Skip(1).Contains("base64", StringComparer.OrdinalIgnoreCase))
        {
            throw new FormatException($"An audio data URL must be base64 (data:audio/wav;base64,...): {Shorten(text)}");
        }

        string? mediaType = header[0].Length == 0 ? null : header[0];
        if (mediaType is not null && !mediaType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
        {
            throw new FormatException($"The data URL holds '{mediaType}', not a sound: {Shorten(text)}");
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

        return data.Length > 0 ? new ChatAudio(data, mediaType) : throw new FormatException("The data URL holds no bytes.");
    }

    /// <summary>The sound as a base64 data URL (<c>data:audio/wav;base64,…</c>; application/octet-stream when the media type is unknown).</summary>
    public string ToDataUrl() => $"data:{MediaType ?? "application/octet-stream"};base64,{Convert.ToBase64String(_data)}";

    /// <summary>The sound decoded by the codec that knows its format (<see cref="AudioCodecs.Decode(ReadOnlySpan{byte})"/>).</summary>
    /// <exception cref="InvalidDataException">No registered codec reads the bytes, or they are damaged.</exception>
    public AudioData Decode() => AudioCodecs.Decode(_data);

    /// <summary>Two sounds are equal when their bytes are (their <see cref="Hash"/>es match), whatever media type they were given.</summary>
    public bool Equals(ChatAudio? other) => other is not null && string.Equals(Hash, other.Hash, StringComparison.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode() => Hash.GetHashCode(StringComparison.Ordinal);

    /// <summary>The media type, size and the start of the hash: "audio/wav, 48044 bytes, sha256 3f2a9c1b…".</summary>
    public override string ToString() => $"{MediaType ?? "audio"}, {_data.Length} bytes, sha256 {Hash[..8]}…";

    private static string Shorten(string text) => text.Length <= 48 ? text : text[..48] + "…";

    // The media type from the bytes' signature; null when it is none of the common sound formats.
    internal static string? Sniff(ReadOnlySpan<byte> data) => data switch
    {
        [(byte)'R', (byte)'I', (byte)'F', (byte)'F', _, _, _, _, (byte)'W', (byte)'A', (byte)'V', (byte)'E', ..] => "audio/wav",
        [(byte)'f', (byte)'L', (byte)'a', (byte)'C', ..] => "audio/flac",
        [(byte)'I', (byte)'D', (byte)'3', ..] => "audio/mpeg",
        [0xFF, var sync, ..] when (sync & 0xE0) == 0xE0 && (sync & 0x06) != 0 => "audio/mpeg",   // an MPEG frame sync with a valid layer
        [(byte)'O', (byte)'g', (byte)'g', (byte)'S', ..] => "audio/ogg",
        [_, _, _, _, (byte)'f', (byte)'t', (byte)'y', (byte)'p', (byte)'M', (byte)'4', (byte)'A', ..] => "audio/mp4",
        _ => null,
    };
}

/// <summary>
/// The "audio" kind of message part: <see cref="ChatAudio"/> as <c>{"type": "audio", "media_type": "audio/wav", "data":
/// "&lt;base64&gt;"}</c>, read back the same, and made from bytes a bridge holds (<see cref="IBinaryPartKind"/>). Registered
/// in <see cref="ChatParts"/> as a library default by this package; an app's kind of the same name shadows it and falls
/// back to it by its policy.
/// </summary>
public sealed class AudioPartKind : IBinaryPartKind
{
    /// <inheritdoc />
    public string Name => ChatParts.Audio;

    /// <inheritdoc />
    public string Summary => "a sound's bytes as base64, with its media type: {\"type\": \"audio\", \"media_type\": …, \"data\": …}";

    /// <inheritdoc />
    public void Write(ChatPart part, JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var audio = part as ChatAudio ?? throw new ArgumentException($"The 'audio' kind writes ChatAudio parts, not {part?.GetType().Name ?? "null"}.", nameof(part));
        if (audio.MediaType is { } mediaType)
        {
            json["media_type"] = mediaType;
        }

        json["data"] = Convert.ToBase64String(audio.Data.Span);
    }

    /// <inheritdoc />
    public ChatPart Read(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        string? mediaType = (string?)json["media_type"];
        if ((string?)json["data"] is not { Length: > 0 } data)
        {
            throw new FormatException("An audio part needs its bytes as base64 in \"data\".");
        }

        try
        {
            return new ChatAudio(Convert.FromBase64String(data), mediaType);   // the decoded array is the sound's own: no second copy
        }
        catch (FormatException)
        {
            throw new FormatException("The \"data\" of an audio part is not valid base64.");
        }
    }

    /// <inheritdoc />
    public ChatPart FromBytes(ReadOnlySpan<byte> data, string? mediaType = null) => ChatAudio.FromBytes(data, mediaType);
}
