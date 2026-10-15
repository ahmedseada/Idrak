// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Audio;

/// <summary>
/// The mel filterbank, [bins, mels]: triangles on the bins' frequencies between mel-spaced points, as torchaudio's
/// <c>melscale_fbanks</c> builds them (the "slaney" or "htk" scale, "slaney" normalisation giving each band an area of
/// 1, or none), computed in double precision once per set of values and cached, as floats, on the host and per device.
/// </summary>
public static class MelFilterbank
{
    private static readonly ConcurrentDictionary<(int Bins, int Mels, int Rate, float Min, float Max, string Scale, string Norm), float[]> Host = new();
    private static readonly ConcurrentDictionary<(Device Device, int Bins, int Mels, int Rate, float Min, float Max, string Scale, string Norm), Tensor> Devices = new();

    /// <summary>The filterbank [bins, mels] (row-major) for <paramref name="options"/> (cached; the array is shared: copy it to change it).</summary>
    public static float[] Build(AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        var key = (options.Bins, options.Mels, options.SampleRate, options.MinFrequency, options.TopFrequency, options.MelScale, options.MelNorm);
        return Host.GetOrAdd(key, k => Compute(k.Bins, k.Mels, k.Rate, k.Min, k.Max, k.Scale, k.Norm));
    }

    /// <summary>The filterbank as a tensor [bins, mels] on <paramref name="device"/> (cached per device; not the caller's to dispose).</summary>
    public static Tensor Tensor(AudioFeatureOptions options, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var values = Build(options);
        var key = (device, options.Bins, options.Mels, options.SampleRate, options.MinFrequency, options.TopFrequency, options.MelScale, options.MelNorm);
        return Devices.GetOrAdd(key, k =>
        {
            var t = Abstraction.Tensor.Empty([k.Bins, k.Mels], k.Device, zeroed: false, track: false);
            t.Load(values);
            return t;
        });
    }

    private static float[] Compute(int bins, int mels, int rate, float fmin, float fmax, string scale, string norm)
    {
        bool htk = scale == "htk";
        var frequencies = new double[bins];
        for (int i = 0; i < bins; i++)
        {
            frequencies[i] = (double)i * rate / 2 / (bins - 1);                  // linspace(0, rate / 2, bins)
        }

        double melMin = HzToMel(fmin, htk), melMax = HzToMel(fmax, htk);
        var points = new double[mels + 2];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = MelToHz(melMin + (melMax - melMin) * i / (mels + 1), htk);
        }

        var values = new float[bins * mels];
        for (int m = 0; m < mels; m++)
        {
            double lower = points[m], centre = points[m + 1], upper = points[m + 2];
            double enorm = norm == "slaney" ? 2.0 / (upper - lower) : 1.0;
            for (int b = 0; b < bins; b++)
            {
                double f = frequencies[b];
                double down = (f - lower) / (centre - lower);                     // torchaudio: -slopes[:, :-2] / f_diff[:-1]
                double up = (upper - f) / (upper - centre);
                double weight = Math.Max(0, Math.Min(down, up));
                values[b * mels + m] = (float)(weight * enorm);
            }
        }

        return values;
    }

    // torchaudio's _hz_to_mel / _mel_to_hz.
    private static double HzToMel(double hz, bool htk)
    {
        if (htk)
        {
            return 2595.0 * Math.Log10(1.0 + hz / 700.0);
        }

        const double FSp = 200.0 / 3, MinLogHz = 1000.0;
        double minLogMel = MinLogHz / FSp, logStep = Math.Log(6.4) / 27.0;
        return hz >= MinLogHz ? minLogMel + Math.Log(hz / MinLogHz) / logStep : hz / FSp;
    }

    private static double MelToHz(double mel, bool htk)
    {
        if (htk)
        {
            return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
        }

        const double FSp = 200.0 / 3, MinLogHz = 1000.0;
        double minLogMel = MinLogHz / FSp, logStep = Math.Log(6.4) / 27.0;
        return mel >= minLogMel ? MinLogHz * Math.Exp(logStep * (mel - minLogMel)) : FSp * mel;
    }
}
