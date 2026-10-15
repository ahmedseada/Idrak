// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// RIFF/WAVE files: PCM 8 (unsigned), 16, 24 and 32 bits, IEEE float 32 and 64, and WAVE_FORMAT_EXTENSIBLE with the PCM
/// and float sub-formats; the chunks walked in order as bytes (a LIST or other chunk before <c>data</c> is skipped, odd
/// sizes padded), a <c>data</c> chunk that claims more than the file holds cut to the file when decoded (phones that
/// never fix the header; <see cref="ReadInfo"/> reports the header's claim). 16-bit files are kept as their bytes until the floats are read (<see cref="AudioData.TryGetSamples16"/>);
/// the others are converted in one pass per channel. RF64 and compressed variants are refused by name.
/// </summary>
public sealed class WavCodec : IAudioCodec
{
    private const ushort FormatPcm = 1, FormatFloat = 3, FormatExtensible = 0xFFFE;

    // DataLength is what the data chunk's header claims; DataAvailable what the given bytes hold of it (less in a file cut
    // short, or a header that was never fixed: the decoder reads what is there).
    private readonly record struct Header(int Format, int Channels, int SampleRate, int Bits, bool Float, int DataOffset, long DataLength, long DataAvailable, bool DataFound,
        bool FormatFound, bool Rf64, bool Complete);

    /// <inheritdoc />
    public string Name => "wav";

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".wav", ".wave"];

    /// <inheritdoc />
    public AudioInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        if (!Parse(header, out var h))
        {
            return null;
        }

        if (!h.FormatFound || !h.DataFound)
        {
            // Not reached yet (a long chunk before fmt or data): -1 asks for the whole file; a whole file without them is a wav
            // the decoder refuses by name (rate 0).
            return h.Complete ? new AudioInfo(0, 0, -1, 0, false, Name) : new AudioInfo(h.SampleRate, h.Channels, -1, h.Bits, h.Float, Name);
        }

        int block = Math.Max(1, h.Channels * (h.Bits / 8));
        return new AudioInfo(h.SampleRate, h.Channels, h.DataLength / block, h.Bits, h.Float, Name);
    }

    /// <inheritdoc />
    public AudioData Decode(ReadOnlySpan<byte> file)
    {
        if (!Parse(file, out var h))
        {
            throw new InvalidDataException("not a RIFF/WAVE file.");
        }

        if (h.Rf64)
        {
            throw new NotSupportedException("an RF64 file (64-bit sizes, past 4 GiB) is not read; only RIFF/WAVE is.");
        }

        if (!h.FormatFound)
        {
            throw new InvalidDataException("no complete fmt chunk before the sound's data (a cut or damaged file).");
        }

        if (!h.DataFound)
        {
            throw new InvalidDataException("no data chunk (a cut or damaged file).");
        }

        if (h.Format is not (FormatPcm or FormatFloat))
        {
            throw new NotSupportedException($"format tag {h.Format} (a compressed or unusual WAV) is not read; PCM and IEEE float are.");
        }

        if (h.Channels <= 0 || h.SampleRate <= 0)
        {
            throw new InvalidDataException($"{h.Channels} channels at {h.SampleRate} Hz.");
        }

        bool known = h.Float ? h.Bits is 32 or 64 : h.Bits is 8 or 16 or 24 or 32;
        if (!known)
        {
            throw new NotSupportedException($"{h.Bits}-bit {(h.Float ? "float" : "PCM")} samples are not read (8, 16, 24 and 32-bit PCM; 32 and 64-bit float).");
        }

        int block = h.Channels * (h.Bits / 8);
        int frames = (int)(h.DataAvailable / block);
        var data = file.Slice(h.DataOffset, frames * block);
        if (h.Bits == 16 && !h.Float)
        {
            return new AudioData(data.ToArray(), h.Channels, h.SampleRate);        // the floats on their first read
        }

        var planar = GC.AllocateUninitializedArray<float>(h.Channels * frames);   // every value written below
        for (int c = 0; c < h.Channels; c++)
        {
            Convert(data, h, c, frames, planar.AsSpan(c * frames, frames));
        }

        return new AudioData(planar, h.Channels, h.SampleRate);
    }

    // One channel's floats from the interleaved data, in memory order; 32-bit PCM through Vector<int> for mono.
    private static void Convert(ReadOnlySpan<byte> data, in Header h, int channel, int frames, Span<float> output)
    {
        int bytes = h.Bits / 8, stride = h.Channels * bytes, at = channel * bytes;
        if (h.Float)
        {
            if (h.Bits == 32)
            {
                if (h.Channels == 1 && BitConverter.IsLittleEndian)
                {
                    MemoryMarshal.Cast<byte, float>(data[..(frames * 4)]).CopyTo(output);
                    return;
                }

                for (int i = 0; i < frames; i++, at += stride)
                {
                    output[i] = BinaryPrimitives.ReadSingleLittleEndian(data.Slice(at, 4));
                }
            }
            else
            {
                for (int i = 0; i < frames; i++, at += stride)
                {
                    output[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(data.Slice(at, 8));
                }
            }

            return;
        }

        switch (h.Bits)
        {
            case 8:
                const float Scale8 = 1f / 128f;
                for (int i = 0; i < frames; i++, at += stride)
                {
                    output[i] = (data[at] - 128) * Scale8;
                }

                break;
            case 24:
                const float Scale24 = 1f / 8388608f;
                for (int i = 0; i < frames; i++, at += stride)
                {
                    int value = data[at] | (data[at + 1] << 8) | (data[at + 2] << 16);   // 3-byte reads, sign-extended
                    output[i] = ((value << 8) >> 8) * Scale24;
                }

                break;
            default:
                const float Scale32 = 1f / 2147483648f;
                if (h.Channels == 1 && BitConverter.IsLittleEndian && Vector.IsHardwareAccelerated)
                {
                    var ints = MemoryMarshal.Cast<byte, int>(data[..(frames * 4)]);
                    var scale = new Vector<float>(Scale32);
                    int width = Vector<int>.Count, i = 0;
                    for (; i + width <= frames; i += width)
                    {
                        (Vector.ConvertToSingle(new Vector<int>(ints.Slice(i, width))) * scale).CopyTo(output.Slice(i, width));
                    }

                    for (; i < frames; i++)
                    {
                        output[i] = ints[i] * Scale32;
                    }

                    return;
                }

                for (int i = 0; i < frames; i++, at += stride)
                {
                    output[i] = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(at, 4)) * Scale32;
                }

                break;
        }
    }

    // The RIFF/WAVE chunks of `file` (the whole file, or its start): false when the bytes are not a WAVE file at all.
    // Chunk ids are compared as bytes (rule 74). `Complete` says the walk reached the file's end (the given bytes hold
    // every chunk there is), so a missing chunk is missing and not merely further on.
    private static bool Parse(ReadOnlySpan<byte> file, out Header header)
    {
        header = default;
        if (file.Length < 12 || !(file[..4].SequenceEqual("RIFF"u8) || file[..4].SequenceEqual("RF64"u8)) || !file.Slice(8, 4).SequenceEqual("WAVE"u8))
        {
            return false;
        }

        bool rf64 = file[..4].SequenceEqual("RF64"u8);
        int format = 0, channels = 0, rate = 0, bits = 0, dataOffset = 0;
        bool isFloat = false, formatFound = false, dataFound = false, complete = false;
        long dataLength = 0, dataAvailable = 0;
        int at = 12;
        while (true)
        {
            if (at + 8 > file.Length)
            {
                complete = at >= file.Length - 1;                              // a pad byte at the very end is allowed
                break;
            }

            var id = file.Slice(at, 4);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(file.Slice(at + 4, 4));
            int body = at + 8;
            if (id.SequenceEqual("fmt "u8))
            {
                if (size < 16 || body + 16 > file.Length)
                {
                    break;                                                     // cut inside the chunk: not found
                }

                format = BinaryPrimitives.ReadUInt16LittleEndian(file.Slice(body, 2));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(file.Slice(body + 2, 2));
                rate = BinaryPrimitives.ReadInt32LittleEndian(file.Slice(body + 4, 4));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(file.Slice(body + 14, 2));
                if (format == FormatExtensible)
                {
                    if (size < 40 || body + 40 > file.Length)
                    {
                        break;
                    }

                    format = BinaryPrimitives.ReadUInt16LittleEndian(file.Slice(body + 24, 2));   // the sub-format GUID's first word (1 PCM, 3 float)
                }

                isFloat = format == FormatFloat;
                formatFound = true;
            }
            else if (id.SequenceEqual("data"u8))
            {
                dataOffset = body;
                dataLength = size;
                dataAvailable = Math.Min(size, (long)file.Length - body);     // a header that claims more than the file holds
                dataFound = true;
                complete = true;
                break;
            }

            long next = body + size + (size & 1);
            if (next > file.Length)
            {
                complete = false;
                break;
            }

            at = (int)next;
        }

        header = new Header(format, channels, rate, bits, isFloat, dataOffset, dataLength, dataAvailable, dataFound, formatFound, rf64, complete);
        return true;
    }
}
