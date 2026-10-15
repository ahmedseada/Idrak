// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Audio;

/// <summary>How the Griffin-Lim vocoder recovers phases. Every value is a setting.</summary>
public sealed record GriffinLimOptions
{
    /// <summary>The phase iterations (torchaudio's default 32).</summary>
    public int Iterations { get; init; } = 32;

    /// <summary>The momentum of the fast Griffin-Lim update, in [0, 1); 0 is the plain algorithm.</summary>
    public float Momentum { get; init; } = 0.99f;

    /// <summary>The exponent of the spectrogram given to <see cref="GriffinLimVocoder.Reconstruct"/>: 1 for magnitudes, 2 for power.</summary>
    public float Power { get; init; } = 1f;

    /// <summary>Start from random phases (seeded by <see cref="Seed"/>) or, when false, from zero phases.</summary>
    public bool RandomPhase { get; init; } = true;

    /// <summary>The seed of the initial phases: the same spectrogram and seed give the same samples.</summary>
    public int Seed { get; init; }
}

/// <summary>
/// The library's default vocoder: sound from a log-mel spectrogram with no weights. The mel bands go back to linear
/// bins through the filterbank's pseudo-inverse (built once per set of options, rule 75), the log and normalisation of
/// <see cref="AudioFeatureOptions"/> are undone, and the phases are recovered by Griffin-Lim with momentum (Perraudin
/// et al.'s fast variant, as torchaudio's <c>griffinlim</c>): <see cref="Stft.Inverse"/> and the forward transform
/// alternated on the host, frames in parallel.
/// </summary>
public sealed class GriffinLimVocoder : IVocoder
{
    /// <summary>The registry name, "griffin-lim".</summary>
    public const string DefaultName = "griffin-lim";

    private static readonly ConcurrentDictionary<(int Bins, int Mels, int Rate, float Min, float Max, string Scale, string Norm), float[]> PseudoInverses = new();

    /// <summary>A vocoder for the mel layout of <paramref name="options"/> with <paramref name="settings"/> (the defaults when null).</summary>
    public GriffinLimVocoder(AudioFeatureOptions? options = null, GriffinLimOptions? settings = null)
    {
        Options = options ?? new AudioFeatureOptions();
        Options.Check();
        Settings = settings ?? new GriffinLimOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Settings.Iterations);
        if (Settings.Momentum is < 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), $"GriffinLimOptions.Momentum is in [0, 1), got {Settings.Momentum}.");
        }
    }

    /// <inheritdoc />
    public string Name => DefaultName;

    /// <summary>The mel layout the log-mel frames follow.</summary>
    public AudioFeatureOptions Options { get; }

    /// <summary>The phase recovery settings.</summary>
    public GriffinLimOptions Settings { get; }

    /// <inheritdoc />
    public int SampleRate => Options.SampleRate;

    /// <inheritdoc />
    public int Mels => Options.Mels;

    /// <summary>
    /// The mel filterbank's pseudo-inverse [mels, bins] for <paramref name="options"/>: (FᵀF + ridge)⁻¹ Fᵀ of the
    /// [bins, mels] filterbank, computed in double once per set of values and cached (the array is shared: copy it to change it).
    /// </summary>
    public static float[] MelPseudoInverse(AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var key = (options.Bins, options.Mels, options.SampleRate, options.MinFrequency, options.TopFrequency, options.MelScale, options.MelNorm);
        return PseudoInverses.GetOrAdd(key, _ => ComputePseudoInverse(MelFilterbank.Build(options), options.Bins, options.Mels));
    }

    /// <inheritdoc />
    public AudioData Synthesize(Tensor logMel, Device device)
    {
        ArgumentNullException.ThrowIfNull(logMel);
        ArgumentNullException.ThrowIfNull(device);
        int mels = Options.Mels;
        if (logMel.Rank != 2 || logMel.Shape[1] != mels)
        {
            throw new ArgumentException($"A log-mel spectrogram is [frames, {mels}], got {Tensor.FormatShape(logMel.Shape)}.", nameof(logMel));
        }

        int frames = logMel.Shape[0];
        var magnitude = LinearMagnitude(logMel.ToArray(), frames);
        return new AudioData(Reconstruct(magnitude, frames), 1, Options.SampleRate);
    }

    /// <summary>
    /// The linear magnitude spectrogram [frames, bins] a log-mel [frames, mels] came from (the normalisation and log of
    /// <see cref="Options"/> undone, the mel bands projected back through the pseudo-inverse, negative power clipped).
    /// </summary>
    public float[] LinearMagnitude(ReadOnlySpan<float> logMel, int frames)
    {
        int mels = Options.Mels, bins = Options.Bins;
        if (logMel.Length < (long)frames * mels)
        {
            throw new ArgumentException($"{logMel.Length} values hold fewer than {frames} x {mels}.", nameof(logMel));
        }

        var inverse = MelPseudoInverse(Options);                                // [mels, bins]
        var (offset, scale) = Options.Normalize ?? (0f, 1f);
        var power = GC.AllocateUninitializedArray<float>(frames * bins);
        var melPower = new double[mels];
        for (int f = 0; f < frames; f++)
        {
            var row = logMel.Slice(f * mels, mels);
            for (int m = 0; m < mels; m++)
            {
                melPower[m] = Math.Pow(10, row[m] / scale - offset);           // (log10 p + offset) · scale undone
            }

            var target = power.AsSpan(f * bins, bins);
            for (int b = 0; b < bins; b++)
            {
                double sum = 0;
                for (int m = 0; m < mels; m++)
                {
                    sum += melPower[m] * inverse[m * bins + b];
                }

                target[b] = (float)Math.Sqrt(Math.Max(0, sum));
            }
        }

        return power;
    }

    /// <summary>
    /// Sound from a linear spectrogram [frames, bins] (magnitudes, or the power <see cref="GriffinLimOptions.Power"/>
    /// names) at the options' window and hop: the phases recovered by <see cref="GriffinLimOptions.Iterations"/>
    /// Griffin-Lim steps with momentum, then one <see cref="Stft.Inverse"/>.
    /// </summary>
    public float[] Reconstruct(ReadOnlySpan<float> spectrogram, int frames)
    {
        int bins = Options.Bins;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frames);
        if (spectrogram.Length < (long)frames * bins)
        {
            throw new ArgumentException($"{spectrogram.Length} values hold fewer than {frames} x {bins}.", nameof(spectrogram));
        }

        int count = frames * bins;
        var magnitude = new float[count];
        float exponent = 1f / Settings.Power;
        for (int i = 0; i < count; i++)
        {
            magnitude[i] = Settings.Power == 1f ? spectrogram[i] : MathF.Pow(spectrogram[i], exponent);
        }

        // The current phases as unit complex numbers, and the previous rebuilt spectrum for the momentum.
        var angleRe = new float[count];
        var angleIm = new float[count];
        if (Settings.RandomPhase)
        {
            var random = new Random(Settings.Seed);
            for (int i = 0; i < count; i++)
            {
                double phase = random.NextDouble() * 2 * Math.PI;
                angleRe[i] = (float)Math.Cos(phase);
                angleIm[i] = (float)Math.Sin(phase);
            }
        }
        else
        {
            Array.Fill(angleRe, 1f);
        }

        var previousRe = new float[count];
        var previousIm = new float[count];
        var specRe = new float[count];
        var specIm = new float[count];
        var rebuiltRe = new float[count];
        var rebuiltIm = new float[count];
        float momentum = Settings.Momentum / (1 + Settings.Momentum);
        int length = Stft.SamplesFor(frames, Options);
        for (int iteration = 0; iteration < Settings.Iterations; iteration++)
        {
            for (int i = 0; i < count; i++)
            {
                specRe[i] = magnitude[i] * angleRe[i];
                specIm[i] = magnitude[i] * angleIm[i];
            }

            var signal = Stft.Inverse(specRe, specIm, frames, Options, length);
            Stft.ComplexOnHost(signal, Options, rebuiltRe, rebuiltIm);
            for (int i = 0; i < count; i++)
            {
                float re = rebuiltRe[i] - momentum * previousRe[i];
                float im = rebuiltIm[i] - momentum * previousIm[i];
                float norm = MathF.Sqrt(re * re + im * im) + 1e-16f;
                angleRe[i] = re / norm;
                angleIm[i] = im / norm;
            }

            (previousRe, rebuiltRe) = (rebuiltRe, previousRe);
            (previousIm, rebuiltIm) = (rebuiltIm, previousIm);
        }

        for (int i = 0; i < count; i++)
        {
            specRe[i] = magnitude[i] * angleRe[i];
            specIm[i] = magnitude[i] * angleIm[i];
        }

        return Stft.Inverse(specRe, specIm, frames, Options, length);
    }

    // (FᵀF + ridge I)⁻¹ Fᵀ for F [bins, mels] (row-major), the ridge a millionth of the diagonal's mean so an empty band
    // (bins too coarse for it) does not make the system singular; Gaussian elimination with partial pivoting in double.
    private static float[] ComputePseudoInverse(float[] filterbank, int bins, int mels)
    {
        var gram = new double[mels, mels];
        double trace = 0;
        for (int i = 0; i < mels; i++)
        {
            for (int j = i; j < mels; j++)
            {
                double sum = 0;
                for (int b = 0; b < bins; b++)
                {
                    sum += (double)filterbank[b * mels + i] * filterbank[b * mels + j];
                }

                gram[i, j] = sum;
                gram[j, i] = sum;
            }

            trace += gram[i, i];
        }

        double ridge = trace / mels * 1e-6;
        for (int i = 0; i < mels; i++)
        {
            gram[i, i] += ridge;
        }

        // Solve gram · X = Fᵀ for X [mels, bins], the right-hand sides being the columns of Fᵀ (the rows of F).
        var rhs = new double[mels, bins];
        for (int b = 0; b < bins; b++)
        {
            for (int m = 0; m < mels; m++)
            {
                rhs[m, b] = filterbank[b * mels + m];
            }
        }

        for (int column = 0; column < mels; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < mels; row++)
            {
                if (Math.Abs(gram[row, column]) > Math.Abs(gram[pivot, column]))
                {
                    pivot = row;
                }
            }

            if (pivot != column)
            {
                for (int j = 0; j < mels; j++)
                {
                    (gram[column, j], gram[pivot, j]) = (gram[pivot, j], gram[column, j]);
                }

                for (int j = 0; j < bins; j++)
                {
                    (rhs[column, j], rhs[pivot, j]) = (rhs[pivot, j], rhs[column, j]);
                }
            }

            double diagonal = gram[column, column];
            for (int row = 0; row < mels; row++)
            {
                if (row == column)
                {
                    continue;
                }

                double factor = gram[row, column] / diagonal;
                if (factor == 0)
                {
                    continue;
                }

                for (int j = column; j < mels; j++)
                {
                    gram[row, j] -= factor * gram[column, j];
                }

                for (int j = 0; j < bins; j++)
                {
                    rhs[row, j] -= factor * rhs[column, j];
                }
            }
        }

        var inverse = new float[mels * bins];
        for (int m = 0; m < mels; m++)
        {
            double diagonal = gram[m, m];
            for (int b = 0; b < bins; b++)
            {
                inverse[m * bins + b] = (float)(rhs[m, b] / diagonal);
            }
        }

        return inverse;
    }
}
