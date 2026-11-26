// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Abstractions;

/// <summary>How a sound file is written (<see cref="IAudioEncoder.Encode"/>).</summary>
public sealed record AudioEncodeOptions
{
    /// <summary>Bits per sample: 8, 16, 24 or 32 for integer samples, 32 for float ones.</summary>
    public int BitsPerSample { get; init; } = 16;

    /// <summary>Whether to store floating-point samples (32-bit) instead of integers.</summary>
    public bool Float { get; init; }

    /// <summary>
    /// The format's own options by key, each value as text; null or empty: the encoder's own settings alone. The library's
    /// encoders take their options as their own (MP3's <see cref="Idrak.Audio.Mp3EncodeOptions"/>), so a key given here is refused.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Parameters { get; init; }
}

/// <summary>
/// Writes one sound format. The reading side of a format is its <see cref="IAudioCodec"/>.
/// </summary>
public interface IAudioEncoder
{
    /// <summary>The format's name, e.g. "wav" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extension it writes, with the dot, e.g. ".wav"; <see cref="AudioEncoders.Save"/> picks an encoder by it.</summary>
    string Extension { get; }

    /// <summary>Writes <paramref name="audio"/> to <paramref name="output"/> (16-bit integers unless <paramref name="options"/> say otherwise).</summary>
    void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null);
}

/// <summary>
/// Writes sound files. The library's encoders, each reached directly: <see cref="Idrak.Audio.WavEncoder"/> (PCM 8, 16, 24
/// and 32 bits and float 32; WAVE_FORMAT_EXTENSIBLE past two channels or 16 bits) and <see cref="Idrak.Audio.Mp3Encoder"/>.
/// Finding the encoder of a file's extension is not on the new design yet.
/// </summary>
public static class AudioEncoders
{
    // Finding the encoder of a path's extension is not on the new design yet.
    private static IAudioEncoder EncoderFor(string path) =>
        throw new NotSupportedException($"{path}: finding the audio encoder of its extension is not on the new design yet.");

    /// <summary>Writes <paramref name="audio"/> to <paramref name="path"/> with the encoder of its extension (the folder is created).</summary>
    /// <exception cref="NotSupportedException">Always for now: finding the encoder of the extension is not on the new design yet.</exception>
    public static void Save(string path, AudioData audio, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var encoder = EncoderFor(path);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = File.Create(path);
        encoder.Encode(audio, stream, options);
    }

    /// <summary>The bytes <paramref name="encoder"/> writes for <paramref name="audio"/>.</summary>
    public static byte[] Encode(this IAudioEncoder encoder, AudioData audio, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        using var stream = new MemoryStream();
        encoder.Encode(audio, stream, options);
        return stream.ToArray();
    }
}
