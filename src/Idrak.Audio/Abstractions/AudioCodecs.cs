// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio.Abstractions;

/// <summary>What a sound file's header says.</summary>
/// <param name="SampleRate">Samples a second.</param>
/// <param name="Channels">Channels, interleaved in the file.</param>
/// <param name="Samples">Samples per channel; -1 when the header read does not reach the sound's data (<see cref="AudioCodecs.ReadInfo(string)"/> then asks again with the whole file).</param>
/// <param name="BitsPerSample">Bits per sample as stored (8, 16, 24, 32, 64).</param>
/// <param name="IsFloat">Whether the samples are stored as floating point.</param>
/// <param name="Format">The codec's name, e.g. "wav".</param>
public readonly record struct AudioInfo(int SampleRate, int Channels, long Samples, int BitsPerSample, bool IsFloat, string Format);

/// <summary>
/// Reads one sound format. A codec decides from a file's first bytes whether the file is in its format, so the extension
/// only decides which files a folder source picks up.
/// </summary>
public interface IAudioCodec
{
    /// <summary>The format's name, e.g. "wav" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".wav".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The rate, channels and length from the start of a file (its first <see cref="AudioCodecs.HeaderBytes"/>, or all
    /// of a smaller file), or null when the bytes are not this format. A file whose data chunk lies further in answers
    /// -1 samples; <see cref="AudioCodecs.ReadInfo(string)"/> (and its in-memory sibling) then asks again with the whole file.
    /// </summary>
    AudioInfo? ReadInfo(ReadOnlySpan<byte> header);

    /// <summary>
    /// Decodes a whole file: a damaged file is an <see cref="InvalidDataException"/>, a valid file in a variant the codec
    /// does not read (a compressed WAV, say) a <see cref="NotSupportedException"/>.
    /// </summary>
    AudioData Decode(ReadOnlySpan<byte> file);
}

/// <summary>
/// Reads sound files with the codec that knows their format. The library's codecs, each reached directly: <see
/// cref="Idrak.Audio.WavCodec"/> (RIFF/WAVE: PCM 8, 16, 24 and 32 bits, float 32 and 64, WAVE_FORMAT_EXTENSIBLE; not
/// compressed variants or RF64), <see cref="Idrak.Audio.FlacCodec"/> and <see cref="Idrak.Audio.Mp3Codec"/>. Finding the
/// codec that knows a file is not on the new design yet: finding it stops.
/// </summary>
public static class AudioCodecs
{
    /// <summary>The bytes of a file's start that <see cref="IAudioCodec.ReadInfo"/> receives.</summary>
    public const int HeaderBytes = 64 * 1024;

    // The codecs in the order a file is offered to them: those that list the path's extension first (a .wav file is the
    // WAV codec's before a codec that finds a frame sync inside its samples is asked), then the rest. Finding them by the
    // path's extension and the file's first bytes is not on the new design yet.
    private static IEnumerable<IAudioCodec> CodecsFor(string path) =>
        throw new NotSupportedException($"{path}: finding the audio codec that reads it among the registered codecs is not on the new design yet.");

    /// <summary>Whether a codec lists the extension of <paramref name="path"/>.</summary>
    /// <exception cref="NotSupportedException">Always for now: picking among the codecs is not on the new design yet.</exception>
    public static bool CanDecode(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Length > 0 && CodecsFor(path).Any(c => c.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>The rate, channels and length of a sound file, or null when no codec knows its format.</summary>
    public static AudioInfo? ReadInfo(string path)
    {
        byte[] head = ArrayPool<byte>.Shared.Rent(HeaderBytes);              // one per file of a folder: pooled, not allocated
        try
        {
            int length;
            using (var stream = File.OpenRead(path))
            {
                length = stream.ReadAtLeast(head.AsSpan(0, HeaderBytes), HeaderBytes, throwOnEndOfStream: false);
            }

            byte[]? whole = null;
            foreach (var codec in CodecsFor(path))
            {
                if (codec.ReadInfo(head.AsSpan(0, length)) is not { } info)
                {
                    continue;
                }

                // The data lies past the first bytes (a long LIST chunk, a large ID3 tag): ask with all of them; a codec that
                // then finds the file is not its own leaves it to the next.
                if (info.Samples >= 0 || length < HeaderBytes)
                {
                    return info;
                }

                whole ??= File.ReadAllBytes(path);
                if (codec.ReadInfo(whole) is { } full)
                {
                    return full;
                }
            }

            return null;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(head);
        }
    }

    /// <summary>
    /// The rate, channels and length of a sound file held in memory, or null when no codec knows its format:
    /// from its first bytes, or all of them when the data lies further on. Nothing is decoded.
    /// </summary>
    public static AudioInfo? ReadInfo(ReadOnlySpan<byte> file)
    {
        var header = file[..Math.Min(file.Length, HeaderBytes)];
        foreach (var codec in CodecsFor("audio"))
        {
            if (codec.ReadInfo(header) is not { } info)
            {
                continue;
            }

            if (info.Samples >= 0 || file.Length == header.Length)
            {
                return info;
            }

            if (codec.ReadInfo(file) is { } full)
            {
                return full;                                                    // not its own after all: the next codec is asked
            }
        }

        return null;
    }

    /// <summary>Decodes a sound file with the codec that knows its format.</summary>
    /// <exception cref="InvalidDataException">No codec reads the file, or it is damaged; the message names the file.</exception>
    /// <exception cref="NotSupportedException">The file is in a variant its codec does not read; the message names the file.</exception>
    public static AudioData Decode(string path)
    {
        // The file's bytes in a pooled buffer: a data loader decodes one file per sample, most past the large-object size.
        using var stream = File.OpenRead(path);
        long size = stream.Length;
        if (size > Array.MaxLength)
        {
            throw new IOException($"{path}: {size} bytes is too large for a sound file.");
        }

        byte[] bytes = ArrayPool<byte>.Shared.Rent((int)size);
        try
        {
            long start = Telemetry.StartStage();
            int length = stream.ReadAtLeast(bytes.AsSpan(0, (int)size), (int)size, throwOnEndOfStream: false);
            var audio = Decode(bytes.AsSpan(0, length), path);
            Telemetry.InferenceStage(InferenceStageKind.AudioDecode, null, start, (long)audio.Channels * audio.Length, length);
            return audio;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    /// <summary>Decodes a sound file held in memory (a chat message's audio, a download) with the codec that knows its format.</summary>
    /// <exception cref="InvalidDataException">No codec reads the bytes, or they are damaged.</exception>
    public static AudioData Decode(ReadOnlySpan<byte> file) => Decode(file, "audio");

    private static AudioData Decode(ReadOnlySpan<byte> bytes, string path)
    {
        var header = bytes[..Math.Min(bytes.Length, HeaderBytes)];
        foreach (var codec in CodecsFor(path))
        {
            // A codec that knows the first bytes decodes the file; one that needs all of them to be sure (a large tag
            // before the sound) and then finds it is not its own leaves it to the next.
            if (codec.ReadInfo(header) is { } info && (info.Samples >= 0 || bytes.Length == header.Length || codec.ReadInfo(bytes) is not null))
            {
                try
                {
                    return codec.Decode(bytes);
                }
                catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException or EndOfStreamException)
                {
                    throw new InvalidDataException($"{path}: a damaged {codec.Name} file ({e.Message}).", e);
                }
                catch (InvalidDataException e)
                {
                    throw new InvalidDataException($"{path}: {e.Message}", e);
                }
                catch (NotSupportedException e)
                {
                    throw new NotSupportedException($"{path}: {e.Message}", e);
                }
            }
        }

        throw new InvalidDataException($"{path}: not an audio format the codecs read.");
    }

    /// <summary>
    /// Decodes a sound file to <paramref name="channels"/> channels: 1 is the mean of the file's channels, a file's own
    /// count is returned as it is, any other count is refused (channels are not invented).
    /// </summary>
    /// <exception cref="ArgumentException">The file has neither one nor <paramref name="channels"/> channels and <paramref name="channels"/> is not 1.</exception>
    public static AudioData Load(string path, int channels)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        var audio = Decode(path);
        if (audio.Channels == channels)
        {
            return audio;
        }

        if (channels == 1)
        {
            var mono = GC.AllocateUninitializedArray<float>(audio.Length);
            audio.ToMono(mono);
            return new AudioData(mono, 1, audio.SampleRate);
        }

        throw new ArgumentException($"{path} has {audio.Channels} channels, not {channels} (1 mixes a file's channels down; others are not invented).", nameof(channels));
    }
}
