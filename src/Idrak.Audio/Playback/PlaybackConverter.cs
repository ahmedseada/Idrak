// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// Blocks of <see cref="AudioData"/> into a playback's device samples, each converted once, a chunk of frames at a time
/// into pooled buffers (returned in finally):
/// <list type="bullet">
/// <item>Channels: as many as the output's are copied; one is copied to every output channel; more are folded onto the
/// output's (output channel c is the mean of source channels c, c + N, c + 2N, ..., the mean of all for a mono output,
/// as <see cref="AudioData.ToMono"/> computes it); fewer are repeated (output channel c plays source channel c mod M).
/// Folding happens before resampling and repeating after, so only the distinct channels are resampled.</item>
/// <item>Rate: by <see cref="Resampler"/>'s own kernel and dot product as one stream over consecutive blocks of the same
/// rate and channels (<see cref="StreamResampler"/>), so the samples played are exactly
/// <see cref="Resampler.Resample"/>'s of the blocks joined, with no edge at each block; the stream ends (its last
/// samples given) at a drain or when a block of another rate or channel count comes.</item>
/// <item>Samples: float to 16-bit as round(x · 32768) clamped (the inverse of the captures' x / 32768, so recorded sound
/// plays back bit for bit), a mono output through vectors with a scalar tail; or float as it is. Little-endian bytes.</item>
/// <item>16-bit sound already at the output's rate and channels (a WAV decoded as 16-bit) is queued as its bytes.</item>
/// </list>
/// </summary>
internal sealed class PlaybackConverter(int rate, int channels, PcmSampleFormat format, ResampleOptions options)
{
    /// <summary>The source frames converted at a time: what bounds the pooled buffers of a write.</summary>
    internal const int ChunkFrames = 4096;

    private readonly int _frameBytes = channels * (format == PcmSampleFormat.F32LE ? sizeof(float) : sizeof(short));
    private StreamResampler[]? _resamplers;
    private int _sourceRate, _sourceChannels;

    /// <summary>Converts <paramref name="block"/> and queues it.</summary>
    public async ValueTask WriteAsync(AudioData block, PlaybackQueue queue, CancellationToken cancellationToken)
    {
        if (_resamplers is not null && (block.SampleRate != _sourceRate || block.Channels != _sourceChannels))
        {
            await FlushAsync(queue, cancellationToken).ConfigureAwait(false);   // the stream before ends here
        }

        if (block.Length == 0)
        {
            return;
        }

        if (block.SampleRate == rate && block.Channels == channels && format == PcmSampleFormat.S16LE && block.TryGetSamples16(out var pcm, out bool bigEndian) && !bigEndian)
        {
            await queue.EnqueueAsync(pcm, cancellationToken).ConfigureAwait(false);
            return;
        }

        bool resample = block.SampleRate != rate;
        int distinct = Math.Min(block.Channels, channels);
        if (resample && _resamplers is null)
        {
            _resamplers = new StreamResampler[distinct];
            for (int d = 0; d < distinct; d++)
            {
                _resamplers[d] = new StreamResampler(block.SampleRate, rate, options, ChunkFrames);
            }

            (_sourceRate, _sourceChannels) = (block.SampleRate, block.Channels);
        }

        int chunk = Math.Min(ChunkFrames, block.Length);
        int stride = resample ? _resamplers![0].MaxOutput(chunk) : chunk;
        float[]? mixed = block.Channels > channels ? ArrayPool<float>.Shared.Rent(distinct * chunk) : null;
        float[]? resampled = resample ? ArrayPool<float>.Shared.Rent(distinct * stride) : null;
        byte[] bytes = ArrayPool<byte>.Shared.Rent(stride * _frameBytes);
        try
        {
            var samples = block.Samples;
            for (int start = 0; start < block.Length; start += chunk)
            {
                int written = Convert(samples, block.Length, block.Channels, start, Math.Min(chunk, block.Length - start), mixed, resampled, stride, bytes);
                if (written > 0)
                {
                    await queue.EnqueueAsync(bytes.AsMemory(0, written), cancellationToken).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
            if (resampled is not null)
            {
                ArrayPool<float>.Shared.Return(resampled);
            }

            if (mixed is not null)
            {
                ArrayPool<float>.Shared.Return(mixed);
            }
        }
    }

    /// <summary>Ends the resampled stream, if one runs: its last samples (the filter's reach past the last block) are queued.</summary>
    public async ValueTask FlushAsync(PlaybackQueue queue, CancellationToken cancellationToken)
    {
        if (_resamplers is not { } resamplers)
        {
            return;
        }

        _resamplers = null;
        int stride = resamplers[0].MaxOutput(0);
        float[] resampled = ArrayPool<float>.Shared.Rent(resamplers.Length * stride);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(stride * _frameBytes);
        try
        {
            int frames = 0;
            for (int d = 0; d < resamplers.Length; d++)
            {
                frames = resamplers[d].Flush(resampled.AsSpan(d * stride, stride));
            }

            int written = Interleave(resampled, 0, stride, frames, resamplers.Length, bytes);
            if (written > 0)
            {
                await queue.EnqueueAsync(bytes.AsMemory(0, written), cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<float>.Shared.Return(resampled);
        }
    }

    // One chunk: the distinct channels at the source rate (views of the block, or folded into `mixed`), resampled into
    // `resampled` (each channel at d · stride), then interleaved into `bytes`; returns the bytes written.
    private int Convert(float[] samples, int length, int sourceChannels, int start, int count, float[]? mixed, float[]? resampled, int stride, byte[] bytes)
    {
        int distinct = Math.Min(sourceChannels, channels);
        float[] source = samples;
        int offset = start, step = length;
        if (mixed is not null)
        {
            for (int d = 0; d < distinct; d++)
            {
                Fold(samples, length, sourceChannels, start, count, d, distinct, mixed.AsSpan(d * count, count));
            }

            (source, offset, step) = (mixed, 0, count);
        }

        int frames = count;
        if (resampled is not null)
        {
            for (int d = 0; d < distinct; d++)
            {
                frames = _resamplers![d].Process(source.AsSpan(offset + d * step, count), resampled.AsSpan(d * stride, stride));
            }

            (source, offset, step) = (resampled, 0, stride);
        }

        return Interleave(source, offset, step, frames, distinct, bytes);
    }

    // Output channel d's mean of the source channels d, d + n, d + 2n, ...: the first copied, the others added in order,
    // then times 1 / their count (AudioData.ToMono's order, so a mono output gets its values).
    private static void Fold(float[] samples, int length, int sourceChannels, int start, int count, int d, int n, Span<float> destination)
    {
        samples.AsSpan(d * length + start, count).CopyTo(destination);
        int folded = 1;
        for (int s = d + n; s < sourceChannels; s += n, folded++)
        {
            Add(samples.AsSpan(s * length + start, count), destination);
        }

        if (folded > 1)
        {
            Multiply(destination, 1f / folded);
        }
    }

    // `frames` frames of the distinct channels (channel d at source[offset + d · step]) into the device's interleaved
    // little-endian samples, output channel c from distinct channel c mod distinct; returns the bytes written.
    private int Interleave(float[] source, int offset, int step, int frames, int distinct, byte[] bytes)
    {
        int written = frames * _frameBytes;
        if (format == PcmSampleFormat.F32LE)
        {
            var output = MemoryMarshal.Cast<byte, float>(bytes.AsSpan(0, written));
            if (channels == 1)
            {
                source.AsSpan(offset, frames).CopyTo(output);
            }
            else
            {
                for (int f = 0, at = 0; f < frames; f++)
                {
                    for (int c = 0; c < channels; c++, at++)
                    {
                        output[at] = source[offset + (c % distinct) * step + f];
                    }
                }
            }

            if (!BitConverter.IsLittleEndian)
            {
                var asInts = MemoryMarshal.Cast<float, int>(output);
                BinaryPrimitives.ReverseEndianness(asInts, asInts);
            }

            return written;
        }

        var shorts = MemoryMarshal.Cast<byte, short>(bytes.AsSpan(0, written));
        if (channels == 1)
        {
            ToInt16(source.AsSpan(offset, frames), shorts);
        }
        else if (channels == 2)
        {
            var left = source.AsSpan(offset, frames);
            var right = source.AsSpan(offset + (distinct == 2 ? step : 0), frames);
            for (int f = 0; f < frames; f++)
            {
                shorts[2 * f] = ToInt16(left[f]);
                shorts[2 * f + 1] = ToInt16(right[f]);
            }
        }
        else
        {
            for (int f = 0, at = 0; f < frames; f++)
            {
                for (int c = 0; c < channels; c++, at++)
                {
                    shorts[at] = ToInt16(source[offset + (c % distinct) * step + f]);
                }
            }
        }

        if (!BitConverter.IsLittleEndian)
        {
            BinaryPrimitives.ReverseEndianness(shorts, shorts);
        }

        return written;
    }

    /// <summary>A float sample as 16 bits: round(x · 32768), to even at halves, clamped to the 16-bit range.</summary>
    internal static short ToInt16(float x) => (short)Math.Clamp(MathF.Round(x * 32768f), -32768f, 32767f);

    // ToInt16 over a span: vectors of floats rounded, clamped, converted and narrowed, then a scalar tail (the same values:
    // the vector rounding is to even as MathF.Round's).
    internal static void ToInt16(ReadOnlySpan<float> source, Span<short> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && source.Length >= 2 * Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            var scale = new Vector<float>(32768f);
            var low = new Vector<float>(-32768f);
            var high = new Vector<float>(32767f);
            for (; i + 2 * width <= source.Length; i += 2 * width)
            {
                var a = Vector.ConvertToInt32(Vector.Min(Vector.Max(Vector.Round(new Vector<float>(source.Slice(i, width)) * scale), low), high));
                var b = Vector.ConvertToInt32(Vector.Min(Vector.Max(Vector.Round(new Vector<float>(source.Slice(i + width, width)) * scale), low), high));
                Vector.Narrow(a, b).CopyTo(destination.Slice(i, 2 * width));
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] = ToInt16(source[i]);
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

/// <summary>
/// One channel resampled as a stream: <see cref="Resampler"/>'s polyphase kernel and dot product over the sound fed so far,
/// the filter's zeros before the first sample, each output frame given as soon as its taps have all arrived, and the
/// rest (the zeros after the last sample) at <see cref="Flush"/>. The samples given are exactly
/// <see cref="Resampler.Resample"/>'s of everything fed, joined. It holds less than one kernel's taps between calls (a
/// buffer made once, sized for its chunks).
/// </summary>
internal sealed class StreamResampler
{
    private readonly Resampler.Kernel _kernel;
    private readonly float[] _input;                                         // the padded sound from padded index _origin
    private int _filled;
    private long _origin;
    private long _frame;                                                     // the next output frame
    private long _fed;
    private long _emitted;

    /// <summary>A stream from <paramref name="from"/> to <paramref name="to"/> fed at most <paramref name="maxChunk"/> samples a call.</summary>
    public StreamResampler(int from, int to, ResampleOptions options, int maxChunk)
    {
        _kernel = Resampler.KernelFor(from, to, options);
        _input = new float[_kernel.Taps + maxChunk + _kernel.PadRight];
        _filled = _kernel.PadLeft;                                           // the zeros before the sound (the array is zeroed)
    }

    /// <summary>The most samples a call fed <paramref name="samples"/> samples gives (<see cref="Flush"/> with 0).</summary>
    public int MaxOutput(int samples) => checked((int)(((long)_kernel.Taps + samples + _kernel.PadRight) / _kernel.From + 1) * _kernel.Phases);

    /// <summary>Feeds <paramref name="samples"/>; the output samples now complete go to <paramref name="output"/> (their count returned).</summary>
    public int Process(ReadOnlySpan<float> samples, Span<float> output)
    {
        samples.CopyTo(_input.AsSpan(_filled));
        _filled += samples.Length;
        _fed += samples.Length;
        return Emit(output, long.MaxValue);
    }

    /// <summary>Ends the stream: the zeros after the sound, and the output samples up to <see cref="Resampler.OutputLength"/> of all fed.</summary>
    public int Flush(Span<float> output)
    {
        _input.AsSpan(_filled, _kernel.PadRight).Clear();
        _filled += _kernel.PadRight;
        return Emit(output, (_fed * _kernel.To + _kernel.From - 1) / _kernel.From);
    }

    // Output sample frame · phases + p is phase p's taps against the padded sound from frame · from, as Resampler computes it.
    private int Emit(Span<float> output, long limit)
    {
        var values = _kernel.Values;
        int taps = _kernel.Taps, phases = _kernel.Phases, from = _kernel.From;
        int written = 0;
        while (_emitted < limit)
        {
            long at = _frame * from - _origin;
            if (at + taps > _filled)
            {
                break;
            }

            var window = _input.AsSpan((int)at, taps);
            for (int p = 0; p < phases && _emitted < limit; p++, _emitted++)
            {
                output[written++] = Resampler.Dot(values.AsSpan(p * taps, taps), window);
            }

            _frame++;
        }

        int done = (int)Math.Min(_frame * from - _origin, _filled);          // no later frame reads before its start
        if (done > 0)
        {
            _input.AsSpan(done, _filled - done).CopyTo(_input);
            _filled -= done;
            _origin += done;
        }

        return written;
    }
}
