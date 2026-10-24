// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;

namespace Idrak.Audio;

/// <summary>
/// The short-time Fourier transform of sound as power or magnitude spectra, [frames, bins] (bins = window / 2 + 1), as
/// <c>torch.stft</c> computes it (a periodic window, centred frames with reflect padding, one-sided). Two computations
/// give the same values: the host's (the library's real FFT, frames in parallel) and the composed device path (the frames
/// unfolded by <c>Im2Col</c>, one matrix product with the windowed cosine and sine basis, the halves squared and added:
/// every step an operation each backend has, so it runs on every device with no kernel of its own).
/// </summary>
public static partial class Stft
{
    private static readonly ConcurrentDictionary<(Device Device, int Window, string Name), Tensor> Bases = new();

    /// <summary>The frames a signal of <paramref name="samples"/> samples gives (reflect padding when centred).</summary>
    public static int FramesFor(long samples, AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        long padded = options.Center ? samples + options.WindowSize / 2 * 2 : samples;
        return padded < options.WindowSize ? 0 : (int)((padded - options.WindowSize) / options.HopSize + 1);
    }

    /// <summary>The window's values (periodic: the symmetric window of n + 1 points without its last).</summary>
    public static float[] Window(string name, int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 2);
        var w = new float[size];
        for (int n = 0; n < size; n++)
        {
            double c = Math.Cos(2 * Math.PI * n / size);
            w[n] = (float)(name switch
            {
                "hann" => 0.5 - 0.5 * c,
                "hamming" => 0.54 - 0.46 * c,
                "rectangular" => 1.0,
                _ => throw new ArgumentException($"Window '{name}' is not known (hann, hamming, rectangular).", nameof(name)),
            });
        }

        return w;
    }

    /// <summary>The signal with its reflect padding (half a window at both ends) when centred, else itself.</summary>
    public static float[] Padded(ReadOnlySpan<float> samples, AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.Center)
        {
            return samples.ToArray();
        }

        int pad = options.WindowSize / 2;
        if (samples.Length <= pad)
        {
            throw new ArgumentException($"{samples.Length} samples are too few to reflect-pad by {pad}.", nameof(samples));
        }

        var padded = new float[samples.Length + 2 * pad];
        samples.CopyTo(padded.AsSpan(pad));
        for (int i = 1; i <= pad; i++)
        {
            padded[pad - i] = samples[i];                                        // the reflection leaves the edge sample out
            padded[pad + samples.Length - 1 + i] = samples[samples.Length - 1 - i];
        }

        return padded;
    }

    /// <summary>The power spectra of <paramref name="samples"/> on the host: [frames, bins] into <paramref name="power"/>.</summary>
    public static unsafe void PowerOnHost(ReadOnlySpan<float> samples, AudioFeatureOptions options, Span<float> power)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        int frames = FramesFor(samples.Length, options), bins = options.Bins, size = options.WindowSize, hop = options.HopSize;
        if (power.Length < (long)frames * bins)
        {
            throw new ArgumentException($"{power.Length} values hold fewer than {frames} x {bins}.", nameof(power));
        }

        var padded = Padded(samples, options);
        var window = Window(options.Window, size);
        var fft = Fft.For(size);
        fixed (float* destination = power)
        {
            nint target = (nint)destination;                                    // written in place: no copy of the spectra (a span cannot cross into the body)
            HostParallel.For(frames, 64, (first, last) =>
            {
                var scratch = ArrayPool<double>.Shared.Rent(fft.ScratchLength + 2 * bins);   // once per block of frames
                try
                {
                    var work = scratch.AsSpan(0, fft.ScratchLength);
                    var re = scratch.AsSpan(fft.ScratchLength, bins);
                    var im = scratch.AsSpan(fft.ScratchLength + bins, bins);
                    for (int f = first; f < last; f++)
                    {
                        fft.Forward(padded.AsSpan(f * hop, size), window, re, im, work);   // the window applied as the frame is read (rule 73)
                        var row = new Span<float>((float*)target + (long)f * bins, bins);
                        for (int k = 0; k < bins; k++)
                        {
                            row[k] = (float)(re[k] * re[k] + im[k] * im[k]);
                        }
                    }
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(scratch);
                }
            });
        }
    }

    /// <summary>The power spectra of <paramref name="samples"/> on <paramref name="device"/> by the composed path: [frames, bins].</summary>
    public static Tensor Power(ReadOnlySpan<float> samples, AudioFeatureOptions options, Device device)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(device);
        options.Check();
        int size = options.WindowSize, bins = options.Bins;
        var padded = Padded(samples, options);
        int frames = FramesFor(samples.Length, options);
        if (frames == 0)
        {
            throw new ArgumentException($"{samples.Length} samples give no frame of {size}.", nameof(samples));
        }

        using var signal = Tensor.Empty([1, 1, 1, padded.Length], device, zeroed: false);
        signal.Load(padded);
        using var cols = signal.Im2Col(new ConvGeometry(1, 1, 1, padded.Length, 1, size, 1, options.HopSize, 0, 0));   // [frames, size]
        using var spectrum = cols.MatMul(Basis(options, device));                                                      // [frames, 2 · bins]: cos | -sin
        using var real = spectrum.Narrow(1, 0, bins);
        using var imaginary = spectrum.Narrow(1, bins, bins);
        using var realSquared = real.Square();
        using var imaginarySquared = imaginary.Square();
        return realSquared + imaginarySquared;
    }

    /// <summary>The magnitude spectra (the square root of <see cref="Power"/>).</summary>
    public static Tensor Magnitude(ReadOnlySpan<float> samples, AudioFeatureOptions options, Device device)
    {
        using var power = Power(samples, options, device);
        return power.Sqrt();
    }

    // The windowed DFT basis [size, 2 · bins]: column k is w[n] cos(2π n k / size), column bins + k is -w[n] sin(2π n k /
    // size) (torch's sign), built in double and stored as floats, once per device and window (not the caller's to dispose).
    internal static Tensor Basis(AudioFeatureOptions options, Device device)
    {
        return Bases.GetOrAdd((device, options.WindowSize, options.Window), key =>
        {
            int size = key.Window, bins = size / 2 + 1;
            var window = Window(key.Name, size);
            var values = new float[size * 2 * bins];
            for (int n = 0; n < size; n++)
            {
                for (int k = 0; k < bins; k++)
                {
                    double angle = 2 * Math.PI * n * k / size;
                    values[n * 2 * bins + k] = (float)(window[n] * Math.Cos(angle));
                    values[n * 2 * bins + bins + k] = (float)(-window[n] * Math.Sin(angle));
                }
            }

            var t = Tensor.Empty([size, 2 * bins], key.Device, zeroed: false, track: false);
            t.Load(values);
            return t;
        });
    }
}
