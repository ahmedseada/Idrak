// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Decoded sound: planar float32 samples in [-1, 1], <c>[Channels, Length]</c>, at <see cref="SampleRate"/> samples a
/// second. Sound may be held as 16-bit interleaved bytes (the WAV codec makes them from a 16-bit file): the floats are
/// made from them on the first read of <see cref="Samples"/>, each sample times 1 / 32768, so they are the values a
/// float decoder would give, and a path that wants the integers (a 16-bit writer) reads them through
/// <see cref="TryGetSamples16"/> without a float round trip. Reading <see cref="Samples"/> ends the 16-bit view.
/// </summary>
public sealed class AudioData
{
    private const float Scale16 = 1f / 32768f;

    private float[]? _samples;
    private byte[]? _pcm16;
    private readonly bool _bigEndian;
    private readonly Lock? _gate;

    /// <summary>Wraps planar samples <c>[channels, samples]</c> in [-1, 1] (used directly, not copied).</summary>
    public AudioData(float[] samples, int channels, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(samples);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (samples.Length % channels != 0)
        {
            throw new ArgumentException($"{samples.Length} samples are not a whole number of frames of {channels} channels.", nameof(samples));
        }

        _samples = samples;
        Channels = channels;
        SampleRate = sampleRate;
        Length = samples.Length / channels;
    }

    /// <summary>
    /// Wraps 16-bit signed samples, interleaved by frame (used directly, not copied), <paramref name="bigEndian"/> when
    /// each sample's high byte comes first (WAV files are little-endian). The floats are made on the first read of
    /// <see cref="Samples"/>.
    /// </summary>
    public AudioData(byte[] pcm16Interleaved, int channels, int sampleRate, bool bigEndian = false)
    {
        ArgumentNullException.ThrowIfNull(pcm16Interleaved);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (pcm16Interleaved.Length % (2 * channels) != 0)
        {
            throw new ArgumentException($"{pcm16Interleaved.Length} bytes are not a whole number of 16-bit frames of {channels} channels.", nameof(pcm16Interleaved));
        }

        (_pcm16, _bigEndian, _gate) = (pcm16Interleaved, bigEndian, new Lock());
        Channels = channels;
        SampleRate = sampleRate;
        Length = pcm16Interleaved.Length / (2 * channels);
    }

    /// <summary>Channels: 1 for mono, 2 for stereo, more for multichannel files.</summary>
    public int Channels { get; }

    /// <summary>Samples a second.</summary>
    public int SampleRate { get; }

    /// <summary>Samples per channel.</summary>
    public int Length { get; }

    /// <summary>The sound's length in time.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds((double)Length / SampleRate);

    /// <summary>
    /// <c>[Channels, Length]</c> values in [-1, 1]: made from the 16-bit samples on the first read when the sound holds
    /// them (then the 16-bit view ends: the array is the caller's to change).
    /// </summary>
    public float[] Samples
    {
        get
        {
            if (Volatile.Read(ref _samples) is { } samples)
            {
                return samples;
            }

            lock (_gate!)
            {
                if (_samples is null)
                {
                    var values = GC.AllocateUninitializedArray<float>(Channels * Length);   // every value written below
                    Convert16(_pcm16!, Channels, Length, _bigEndian, values);
                    Volatile.Write(ref _samples, values);
                    Volatile.Write(ref _pcm16, null);
                }

                return _samples;
            }
        }
    }

    /// <summary>
    /// The 16-bit interleaved samples when the sound still holds them (decoded from 16 bits and <see cref="Samples"/>
    /// not read yet), read-only, with their byte order.
    /// </summary>
    public bool TryGetSamples16(out ReadOnlyMemory<byte> interleaved, out bool bigEndian)
    {
        if (Volatile.Read(ref _pcm16) is { } held)
        {
            (interleaved, bigEndian) = (held, _bigEndian);
            return true;
        }

        (interleaved, bigEndian) = (default, false);
        return false;
    }

    /// <summary>One channel's samples, a view of <see cref="Samples"/>.</summary>
    public ReadOnlySpan<float> Channel(int channel)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(channel);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(channel, Channels);
        return Samples.AsSpan(channel * Length, Length);
    }

    /// <summary>The mean of the channels into <paramref name="destination"/> (<see cref="Length"/> values), in one pass.</summary>
    public void ToMono(Span<float> destination)
    {
        if (destination.Length < Length)
        {
            throw new ArgumentException($"{destination.Length} values hold fewer than the {Length} samples of a channel.", nameof(destination));
        }

        var samples = Samples;
        destination = destination[..Length];
        samples.AsSpan(0, Length).CopyTo(destination);
        for (int c = 1; c < Channels; c++)
        {
            Add(samples.AsSpan(c * Length, Length), destination);
        }

        if (Channels > 1)
        {
            Multiply(destination, 1f / Channels);
        }
    }

    /// <summary>
    /// Channel <paramref name="channel"/> into <paramref name="destination"/>, padded with silence past its end (and cut
    /// to <paramref name="destination"/> when shorter): a segment of a fixed length for a feature extractor.
    /// </summary>
    public void Pad(int channel, Span<float> destination)
    {
        var source = Channel(channel);
        int count = Math.Min(source.Length, destination.Length);
        source[..count].CopyTo(destination);
        destination[count..].Clear();
    }

    /// <summary>Samples <paramref name="from"/> to <paramref name="from"/> + <paramref name="count"/> of every channel, as new planar samples.</summary>
    public AudioData Slice(int from, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(from);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (from + (long)count > Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count), $"Samples {from} to {from + count} of {Length}.");
        }

        var samples = Samples;
        var cut = GC.AllocateUninitializedArray<float>(Channels * count);
        for (int c = 0; c < Channels; c++)
        {
            samples.AsSpan(c * Length + from, count).CopyTo(cut.AsSpan(c * count, count));
        }

        return new AudioData(cut, Channels, SampleRate);
    }

    // The planar floats of 16-bit interleaved samples: one pass per channel over the frames, in memory order (rule 72),
    // the mono little-endian case through Vector<short> widening with a scalar tail (identical results: an integer
    // times a power of two is exact in float32).
    internal static void Convert16(ReadOnlySpan<byte> pcm, int channels, int length, bool bigEndian, Span<float> planar)
    {
        if (channels == 1 && !bigEndian && BitConverter.IsLittleEndian)
        {
            var shorts = MemoryMarshal.Cast<byte, short>(pcm[..(length * 2)]);
            var scale = new Vector<float>(Scale16);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int width = Vector<short>.Count;
                for (; i + width <= length; i += width)
                {
                    Vector.Widen(new Vector<short>(shorts.Slice(i, width)), out var low, out var high);
                    (Vector.ConvertToSingle(low) * scale).CopyTo(planar.Slice(i, Vector<float>.Count));
                    (Vector.ConvertToSingle(high) * scale).CopyTo(planar.Slice(i + Vector<float>.Count, Vector<float>.Count));
                }
            }

            for (; i < length; i++)
            {
                planar[i] = shorts[i] * Scale16;
            }

            return;
        }

        for (int c = 0; c < channels; c++)
        {
            var channel = planar.Slice(c * length, length);
            int at = c * 2;
            int stride = channels * 2;
            for (int i = 0; i < length; i++, at += stride)
            {
                short value = bigEndian ? BinaryPrimitives.ReadInt16BigEndian(pcm.Slice(at, 2)) : BinaryPrimitives.ReadInt16LittleEndian(pcm.Slice(at, 2));
                channel[i] = value * Scale16;
            }
        }
    }

    private static void Add(ReadOnlySpan<float> source, Span<float> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            for (; i + width <= source.Length; i += width)
            {
                (new Vector<float>(destination.Slice(i, width)) + new Vector<float>(source.Slice(i, width))).CopyTo(destination.Slice(i, width));
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] += source[i];
        }
    }

    private static void Multiply(Span<float> values, float factor)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            var scale = new Vector<float>(factor);
            for (; i + width <= values.Length; i += width)
            {
                (new Vector<float>(values.Slice(i, width)) * scale).CopyTo(values.Slice(i, width));
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] *= factor;
        }
    }
}
