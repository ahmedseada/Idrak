// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>The power spectra a short-time Fourier transform under test gives: [Frames, Bins], row-major.</summary>
/// <param name="Frames">Frames.</param>
/// <param name="Bins">Frequency bins per frame (window / 2 + 1).</param>
/// <param name="Power">The power (squared magnitude) of each bin, frame by frame.</param>
public readonly record struct PowerSpectra(int Frames, int Bins, float[] Power);

/// <summary>
/// A short-time Fourier transform under test (Idrak.Audio's <c>Stft</c>, or one of your own), given as what the suite
/// needs to check it without depending on the package: its window and hop, whether frames are centred (reflect padding
/// of half a window at both ends), the window's values, and the power spectra of a signal.
/// </summary>
/// <param name="WindowSize">The frame's samples.</param>
/// <param name="HopSize">Samples between frames.</param>
/// <param name="Center">Whether frames are centred with reflect padding.</param>
/// <param name="Window">The window's value at each of the <paramref name="WindowSize"/> samples.</param>
/// <param name="Power">The power spectra of a signal.</param>
public sealed record StftUnderTest(int WindowSize, int HopSize, bool Center, Func<int, float> Window, Func<float[], PowerSpectra> Power);

/// <summary>
/// The checks of a short-time Fourier transform on signals the suite makes (silence, a pure tone on a bin, seeded
/// noise, at fixed and random lengths):
/// <list type="bullet">
/// <item>the frame count follows from the length, window, hop and centring; every frame has window / 2 + 1 bins;</item>
/// <item>silence gives zero power everywhere;</item>
/// <item>a pure tone on bin k has its largest power at bin k in every frame;</item>
/// <item>Parseval's identity holds per frame: the one-sided power summed with its weights (1 for the first and last
/// bins, 2 for the others) equals the window's samples times the windowed frame's energy, within 1e-3 of it;</item>
/// <item>the same spectra on a second call.</item>
/// </list>
/// </summary>
/// <param name="sampleRate">The rate the signals are made at (only the tone's frequency depends on it).</param>
public sealed class AudioFeatureSuite(int sampleRate = 16000) : ContractSuite<StftUnderTest>
{
    /// <inheritdoc />
    public override string Name => "audio-stft";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("silence", 0, 4096, 0);
        yield return Case("tone", 1, 8000, 7);
        yield return Case("noise", 2, 5000, 11);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int kind = random.Next(3);
        return Case(kind switch { 0 => "silence", 1 => "tone", _ => "noise" }, kind, random.Next(600, large ? 200_000 : 20_000), random.Next(1, 60));
    }

    /// <inheritdoc />
    public override void Run(StftUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int kind = (int)@case.Data["kind"]!, length = (int)@case.Data["length"]!, seed = (int)@case.Data["seed"]!;
        int size = implementation.WindowSize, hop = implementation.HopSize, bins = size / 2 + 1;
        int toneBin = Math.Min(bins - 2, 1 + seed % Math.Max(1, bins - 3));
        var signal = new float[length];
        if (kind == 1)
        {
            for (int n = 0; n < length; n++)
            {
                signal[n] = 0.5f * MathF.Cos(2 * MathF.PI * toneBin * n / size);
            }
        }
        else if (kind == 2)
        {
            var random = new Random(seed);
            for (int n = 0; n < length; n++)
            {
                signal[n] = (float)(random.NextDouble() * 2 - 1) * 0.8f;
            }
        }

        var spectra = implementation.Power(signal);
        int pad = implementation.Center ? size / 2 : 0;
        int padded = length + 2 * pad;
        int expectedFrames = padded < size ? 0 : (padded - size) / hop + 1;
        checks.Check("the frame count follows from the length, window, hop and centring", spectra.Frames == expectedFrames, $"{spectra.Frames} frames, expected {expectedFrames} for {length} samples");
        checks.Check("every frame has window / 2 + 1 bins", spectra.Bins == bins && spectra.Power.Length == spectra.Frames * bins, $"{spectra.Bins} bins, {spectra.Power.Length} values");
        if (spectra.Frames != expectedFrames || spectra.Bins != bins)
        {
            return;
        }

        if (kind == 0)
        {
            checks.Check("silence gives zero power", spectra.Power.All(v => v == 0f), "a non-zero bin on silence");
        }
        else
        {
            checks.Skip("silence gives zero power", "not a silent case");
        }

        if (kind == 1)
        {
            int wrong = -1;
            for (int f = 0; f < spectra.Frames && wrong < 0; f++)
            {
                int best = 0;
                for (int k = 1; k < bins; k++)
                {
                    if (spectra.Power[f * bins + k] > spectra.Power[f * bins + best])
                    {
                        best = k;
                    }
                }

                if (best != toneBin)
                {
                    wrong = f;
                }
            }

            checks.Check("a tone on a bin peaks at that bin in every frame", wrong < 0, wrong < 0 ? "" : $"frame {wrong} peaks elsewhere than bin {toneBin}");
        }
        else
        {
            checks.Skip("a tone on a bin peaks at that bin in every frame", "not a tone case");
        }

        // Parseval per frame, the frame rebuilt here with the reflect padding when centred.
        var window = new float[size];
        for (int n = 0; n < size; n++)
        {
            window[n] = implementation.Window(n);
        }

        int bad = -1;
        double worst = 0;
        for (int f = 0; f < spectra.Frames; f++)
        {
            double energy = 0;
            for (int n = 0; n < size; n++)
            {
                int i = f * hop + n - pad;
                float x = i < 0 ? signal[-i] : i >= length ? signal[2 * length - 2 - i] : signal[i];
                double v = window[n] * x;
                energy += v * v;
            }

            double sum = 0;
            for (int k = 0; k < bins; k++)
            {
                double weight = k == 0 || (size % 2 == 0 && k == bins - 1) ? 1 : 2;
                sum += weight * spectra.Power[f * bins + k];
            }

            double expected = size * energy;
            double difference = Math.Abs(sum - expected) / Math.Max(1e-6, Math.Max(Math.Abs(sum), Math.Abs(expected)));
            if (expected > 1e-9 && difference > 1e-3)
            {
                (bad, worst) = (f, difference);
                break;
            }
        }

        checks.Check("Parseval's identity holds per frame", bad < 0, bad < 0 ? "" : $"frame {bad} differs by {worst:E2} (relative)");
        var again = implementation.Power(signal);
        checks.Compare("the same spectra on a second call", Comparisons.Difference(spectra.Power, again.Power, 0f));
    }

    private ContractCase Case(string name, int kind, int length, int seed) =>
        new($"{name}, {length} samples{(kind == 1 ? ", seed " + seed : "")}", new JsonObject { ["kind"] = kind, ["length"] = length, ["seed"] = seed, ["rate"] = sampleRate });
}
