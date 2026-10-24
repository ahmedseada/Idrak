// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Numerics;

namespace Idrak.Audio;

/// <summary>How sound is resampled (<see cref="Resampler"/>): torchaudio's windowed-sinc interpolation by default.</summary>
public sealed record ResampleOptions
{
    /// <summary>The low-pass filter's width in zero crossings on each side (torchaudio's lowpass_filter_width).</summary>
    public int Width { get; init; } = 6;

    /// <summary>The cut-off as a share of the Nyquist frequency of the lower rate (torchaudio's rolloff).</summary>
    public double Rolloff { get; init; } = 0.99;

    /// <summary>The window: "sinc-hann" (torchaudio's sinc_interp_hann).</summary>
    public string Method { get; init; } = "sinc-hann";
}

/// <summary>
/// Sound from one rate to another by a polyphase windowed-sinc filter, as torchaudio's <c>resample</c> computes it
/// (<c>sinc_interp_hann</c>): the rate pair reduced by its greatest common divisor, one kernel per output phase, built in
/// double precision once per (from, to, options) and cached. On the host each output sample is one dot product over the
/// taps, channels and blocks of frames in parallel; on a device the same kernel runs as one convolution (the phases as
/// output channels) and a permute, so the two give the same values within 1e-5.
/// </summary>
public static class Resampler
{
    private static readonly ConcurrentDictionary<(int From, int To, int Width, double Rolloff), Kernel> Kernels = new();
    private static readonly ConcurrentDictionary<(Device Device, int From, int To, int Width, double Rolloff), Tensor> DeviceKernels = new();

    /// <summary>The polyphase kernel of a rate pair: [phases, taps], the reduced rates, and the zeros before the sound.</summary>
    internal sealed record Kernel(float[] Values, int Phases, int Taps, int From, int To, int PadLeft, int PadRight);

    /// <summary>The samples a sound of <paramref name="samples"/> samples has after resampling from <paramref name="from"/> to <paramref name="to"/>.</summary>
    public static long OutputLength(long samples, int from, int to)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(from);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(to);
        return (samples * to + from - 1) / from;
    }

    /// <summary><paramref name="audio"/> at <paramref name="sampleRate"/> (itself when it is there already), computed on the host.</summary>
    public static AudioData Resample(AudioData audio, int sampleRate, ResampleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (audio.SampleRate == sampleRate)
        {
            return audio;
        }

        long start = Telemetry.Start(TelemetryLevel.Inference);
        var kernel = KernelFor(audio.SampleRate, sampleRate, options ?? new ResampleOptions());
        int length = (int)OutputLength(audio.Length, audio.SampleRate, sampleRate);
        var output = GC.AllocateUninitializedArray<float>(audio.Channels * length);
        var samples = audio.Samples;
        for (int c = 0; c < audio.Channels; c++)
        {
            ResampleChannel(samples.AsSpan(c * audio.Length, audio.Length), kernel, output.AsSpan(c * length, length));
        }

        Telemetry.InferenceStage(InferenceStageKind.Resample, null, start, (long)audio.Channels * length);
        return new AudioData(output, audio.Channels, sampleRate);
    }

    /// <summary>
    /// <paramref name="audio"/> at <paramref name="sampleRate"/> on <paramref name="device"/>, as [channels, samples]: one
    /// convolution of the padded sound by the phases' kernels, then the phases interleaved.
    /// </summary>
    public static Tensor ResampleOnDevice(AudioData audio, int sampleRate, Device device, ResampleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(device);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (audio.SampleRate == sampleRate)
        {
            var same = Tensor.Empty([audio.Channels, audio.Length], device, zeroed: false);
            same.Load(audio.Samples);
            return same;
        }

        long start = Telemetry.Start(TelemetryLevel.Inference);
        options ??= new ResampleOptions();
        var kernel = KernelFor(audio.SampleRate, sampleRate, options);
        int length = (int)OutputLength(audio.Length, audio.SampleRate, sampleRate);
        int padded = kernel.PadLeft + audio.Length + kernel.PadRight;
        int frames = (padded - kernel.Taps) / kernel.From + 1;

        // The sound with its zeros, [channels, 1, 1, padded]: copied once into the upload buffer (rule 73).
        var host = new float[audio.Channels * padded];
        var samples = audio.Samples;
        for (int c = 0; c < audio.Channels; c++)
        {
            samples.AsSpan(c * audio.Length, audio.Length).CopyTo(host.AsSpan(c * padded + kernel.PadLeft, audio.Length));
        }

        using var input = Tensor.Empty([audio.Channels, 1, 1, padded], device, zeroed: false);
        input.Load(host);
        var weight = DeviceKernels.GetOrAdd((device, audio.SampleRate, sampleRate, options.Width, options.Rolloff), key =>
        {
            var w = Tensor.Empty([kernel.Phases, 1, 1, kernel.Taps], key.Device, zeroed: false, track: false);
            w.Load(kernel.Values);
            return w;
        });
        var geometry = new ConvGeometry(audio.Channels, 1, 1, padded, 1, kernel.Taps, 1, kernel.From, 0, 0);
        using var convolved = input.Convolution(weight, null, geometry);               // [channels, phases, 1, frames]
        using var byPhase = convolved.Reshape(audio.Channels, kernel.Phases, frames);
        using var byFrame = byPhase.Transpose();                                       // [channels, frames, phases]: the phases interleaved
        using var flat = byFrame.Reshape(audio.Channels, frames * kernel.Phases);
        var result = flat.Narrow(1, 0, length);
        Telemetry.InferenceStage(InferenceStageKind.Resample, device, start, (long)audio.Channels * length);
        return result;
    }

    // One channel through the polyphase filter: output sample k·phases + p is the dot product of phase p's taps with the
    // padded sound from k·from, the taps in order (rule 72), blocks of frames in parallel (rule 79).
    private static void ResampleChannel(ReadOnlySpan<float> channel, Kernel kernel, Span<float> output)
    {
        int padded = kernel.PadLeft + channel.Length + kernel.PadRight;
        var input = new float[padded];                                           // one block per channel; the zeros at both ends
        channel.CopyTo(input.AsSpan(kernel.PadLeft));
        int frames = (output.Length + kernel.Phases - 1) / kernel.Phases;
        int length = output.Length;
        var values = kernel.Values;
        var target = output.ToArray();                                           // written by every worker, copied back once
        HostParallel.For(frames, 256, (first, last) =>
        {
            for (int frame = first; frame < last; frame++)
            {
                int at = frame * kernel.From;
                for (int p = 0; p < kernel.Phases; p++)
                {
                    int index = frame * kernel.Phases + p;
                    if (index >= length)
                    {
                        break;
                    }

                    target[index] = Dot(values.AsSpan(p * kernel.Taps, kernel.Taps), input.AsSpan(at, kernel.Taps));
                }
            }
        });
        target.CopyTo(output);
    }

    // The dot product of a phase's taps with the sound; the "resample" processor's stage takes it too, so blocks give these bits.
    internal static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
        {
            var acc = Vector<float>.Zero;
            int width = Vector<float>.Count;
            for (; i + width <= a.Length; i += width)
            {
                acc += new Vector<float>(a.Slice(i, width)) * new Vector<float>(b.Slice(i, width));
            }

            sum = Vector.Sum(acc);
        }

        for (; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>The kernel for a rate pair (built once, cached).</summary>
    internal static Kernel KernelFor(int from, int to, ResampleOptions options)
    {
        if (options.Method != "sinc-hann")
        {
            throw new NotSupportedException($"Resampling method '{options.Method}' is not known (sinc-hann).");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Width, nameof(options));
        if (options.Rolloff is <= 0 or > 1)
        {
            throw new ArgumentException($"Rolloff is a share from 0 to 1, not {options.Rolloff}.", nameof(options));
        }

        return Kernels.GetOrAdd((from, to, options.Width, options.Rolloff), key => Build(key.From, key.To, key.Width, key.Rolloff));
    }

    // torchaudio.functional.resample's kernel (sinc_interp_hann), in double, stored as float32: the sinc at the output
    // phases' offsets, times a Hann window, times the filter's gain; the sound padded by width zeros before and width +
    // from zeros after, so every output sample has its full taps.
    private static Kernel Build(int from, int to, int width, double rolloff)
    {
        int g = (int)BigInteger.GreatestCommonDivisor(from, to);
        int orig = from / g, next = to / g;
        double baseFreq = Math.Min(orig, next) * rolloff;
        int half = (int)Math.Ceiling(width * orig / baseFreq);
        int taps = 2 * half + orig;
        var values = new float[next * taps];
        double scale = baseFreq / orig;
        for (int p = 0; p < next; p++)
        {
            for (int j = 0; j < taps; j++)
            {
                double t = ((double)(j - half) / orig - (double)p / next) * baseFreq;
                t = Math.Clamp(t, -width, width);
                double window = Math.Cos(t * Math.PI / width / 2);
                window *= window;
                t *= Math.PI;
                double sinc = t == 0 ? 1.0 : Math.Sin(t) / t;
                values[p * taps + j] = (float)(sinc * window * scale);
            }
        }

        return new Kernel(values, next, taps, orig, next, half, half + orig);
    }
}
