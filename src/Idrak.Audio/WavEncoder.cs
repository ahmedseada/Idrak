// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;

namespace Idrak.Audio;

/// <summary>
/// Writes RIFF/WAVE files: PCM 8, 16, 24 and 32 bits (<see cref="AudioEncodeOptions.BitsPerSample"/>) or 32-bit float
/// (<see cref="AudioEncodeOptions.Float"/>), as WAVE_FORMAT_EXTENSIBLE past two channels or 16 bits. Sound still held
/// as 16-bit samples is written from them when 16-bit PCM is asked for (no float round trip).
/// </summary>
public sealed class WavEncoder : IAudioEncoder
{
    private const int BlockFrames = 1 << 14;

    /// <inheritdoc />
    public string Name => "wav";

    /// <inheritdoc />
    public string Extension => ".wav";

    /// <inheritdoc />
    public void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(output);
        options ??= new AudioEncodeOptions();
        if (options.Parameters is { Count: > 0 } parameters)
        {
            _ = AudioEncoders.NoParameters.Parse(parameters);                           // WAV takes none: every key refused, by name
        }

        int bits = options.Float ? 32 : options.BitsPerSample;
        if (!options.Float && bits is not (8 or 16 or 24 or 32))
        {
            throw new ArgumentException($"WAV samples are 8, 16, 24 or 32 bits, not {bits}.", nameof(options));
        }

        int channels = audio.Channels, bytes = bits / 8, block = channels * bytes;
        long dataBytes = (long)audio.Length * block;
        if (dataBytes > uint.MaxValue - 100)
        {
            throw new NotSupportedException($"{dataBytes} bytes of samples do not fit a RIFF/WAVE file (4 GiB); RF64 is not written.");
        }

        bool extensible = channels > 2 || bits > 16;
        int formatSize = extensible ? 40 : 16;
        Span<byte> header = stackalloc byte[12 + 8 + 40 + 8];
        int at = 0;
        "RIFF"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..], (uint)(4 + 8 + formatSize + 8 + dataBytes + (dataBytes & 1)));
        "WAVE"u8.CopyTo(header[8..]);
        at = 12;
        "fmt "u8.CopyTo(header[at..]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[(at + 4)..], (uint)formatSize);
        at += 8;
        ushort tag = extensible ? (ushort)0xFFFE : options.Float ? (ushort)3 : (ushort)1;
        BinaryPrimitives.WriteUInt16LittleEndian(header[at..], tag);
        BinaryPrimitives.WriteUInt16LittleEndian(header[(at + 2)..], (ushort)channels);
        BinaryPrimitives.WriteInt32LittleEndian(header[(at + 4)..], audio.SampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(header[(at + 8)..], audio.SampleRate * block);
        BinaryPrimitives.WriteUInt16LittleEndian(header[(at + 12)..], (ushort)block);
        BinaryPrimitives.WriteUInt16LittleEndian(header[(at + 14)..], (ushort)bits);
        at += 16;
        if (extensible)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(header[at..], 22);
            BinaryPrimitives.WriteUInt16LittleEndian(header[(at + 2)..], (ushort)bits);
            BinaryPrimitives.WriteUInt32LittleEndian(header[(at + 4)..], channels switch { 1 => 0x4u, 2 => 0x3u, _ => (1u << channels) - 1 });
            BinaryPrimitives.WriteUInt16LittleEndian(header[(at + 8)..], options.Float ? (ushort)3 : (ushort)1);
            ReadOnlySpan<byte> guid = [0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71];   // the rest of KSDATAFORMAT_SUBTYPE_*
            guid.CopyTo(header[(at + 10)..]);
            at += 24;
        }

        "data"u8.CopyTo(header[at..]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[(at + 4)..], (uint)dataBytes);
        at += 8;
        output.Write(header[..at]);

        if (bits == 16 && !options.Float && audio.TryGetSamples16(out var held, out bool bigEndian))
        {
            if (!bigEndian)
            {
                output.Write(held.Span);
            }
            else
            {
                WriteSwapped(held.Span, output);
            }
        }
        else
        {
            WriteFrames(audio, bits, options.Float, output);
        }

        if ((dataBytes & 1) != 0)
        {
            output.WriteByte(0);
        }
    }

    // 16-bit samples held big-endian, written little-endian through a pooled block.
    private static void WriteSwapped(ReadOnlySpan<byte> source, Stream output)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(BlockFrames * 2);
        try
        {
            for (int done = 0; done < source.Length;)
            {
                int count = Math.Min(buffer.Length, source.Length - done);
                for (int i = 0; i < count; i += 2)
                {
                    buffer[i] = source[done + i + 1];
                    buffer[i + 1] = source[done + i];
                }

                output.Write(buffer, 0, count);
                done += count;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The samples quantized and interleaved, a block of frames at a time through a pooled buffer.
    private static void WriteFrames(AudioData audio, int bits, bool isFloat, Stream output)
    {
        var samples = audio.Samples;
        int channels = audio.Channels, length = audio.Length, bytes = bits / 8, block = channels * bytes;
        byte[] buffer = ArrayPool<byte>.Shared.Rent(BlockFrames * block);
        try
        {
            for (int first = 0; first < length; first += BlockFrames)
            {
                int frames = Math.Min(BlockFrames, length - first);
                for (int c = 0; c < channels; c++)
                {
                    var channel = samples.AsSpan(c * length + first, frames);
                    int at = c * bytes;
                    for (int i = 0; i < frames; i++, at += block)
                    {
                        var target = buffer.AsSpan(at, bytes);
                        float value = channel[i];
                        if (isFloat)
                        {
                            BinaryPrimitives.WriteSingleLittleEndian(target, value);
                            continue;
                        }

                        switch (bits)
                        {
                            case 8:
                                target[0] = (byte)(Quantize(value, 128) + 128);
                                break;
                            case 16:
                                BinaryPrimitives.WriteInt16LittleEndian(target, (short)Quantize(value, 32768));
                                break;
                            case 24:
                                int q = Quantize(value, 8388608);
                                target[0] = (byte)q;
                                target[1] = (byte)(q >> 8);
                                target[2] = (byte)(q >> 16);
                                break;
                            default:
                                BinaryPrimitives.WriteInt32LittleEndian(target, Quantize32(value));
                                break;
                        }
                    }
                }

                output.Write(buffer, 0, frames * block);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // Round to the nearest integer and clamp to [-full, full - 1], as a sound library quantizes.
    private static int Quantize(float value, int full) => (int)Math.Clamp(MathF.Round(value * full), -full, full - 1);

    private static int Quantize32(float value)
    {
        double scaled = Math.Round((double)value * 2147483648.0);
        return scaled >= int.MaxValue ? int.MaxValue : scaled <= int.MinValue ? int.MinValue : (int)scaled;
    }
}
