// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
// Ported from xiph's RNNoise v0.1.1 (src/denoise.c: Copyright (c) 2018 Gregor Richards, 2017 Mozilla; src/pitch.c and
// src/celt_lpc.c: Copyright (c) 2007-2009 Xiph.Org Foundation, 2007-2008 CSIRO, 2003-2008 Jean-Marc Valin, 2009-2017
// Jean-Marc Valin; BSD-3-Clause, its text in README.md in this folder).

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Audio.Plugins.RnNoise;

/// <summary>
/// RNNoise on one channel (its <c>DenoiseState</c> and <c>rnnoise_process_frame</c>): frames of 480 samples at 48 kHz in
/// 16-bit scale (±32768). Each frame is high-passed, windowed with its previous frame (960 samples, a power-complementary
/// window) and transformed; the 22 band energies, the pitch (found on the signal decimated by 2 and 4, octave errors
/// removed) and the pitch-delayed spectrum's correlation per band give 42 features; the network gives a gain per band
/// and the voice probability; the pitch filter adds back the pitch-delayed spectrum where the voice is periodic; the
/// gains (held from falling faster than 0.6 a frame) are interpolated over the bins; the inverse transform is
/// overlap-added with the previous frame's second half. The output of a frame is the 480 samples before it: RNNoise's
/// delay is one frame. Every computation is the C code's in its order and in float (double where it takes double), so
/// the samples are its samples. All scratch is held here: nothing is allocated per frame.
/// </summary>
internal sealed class RnNoiseDenoiser
{
    public const int FrameSize = 480;
    public const int WindowSize = 2 * FrameSize;
    public const int FreqSize = FrameSize + 1;
    public const int Bands = RnNoiseModel.Bands;
    private const int PitchMinPeriod = 60, PitchMaxPeriod = 768, PitchFrameSize = 960, PitchBufSize = PitchMaxPeriod + PitchFrameSize;
    private const int CepsMem = 8, DeltaCeps = 6, Features = Bands + 3 * DeltaCeps + 2;

    // Band edges in units of 4 bins (200 Hz at 48 kHz: 0 to 20 kHz).
    private static ReadOnlySpan<short> Edges => [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];

    private static readonly RnNoiseFft Fft = new(WindowSize);
    private static readonly float[] HalfWindow = MakeHalfWindow();
    private static readonly float[] DctTable = MakeDctTable();

    private readonly RnNoiseModel _model;
    private readonly RnNoiseNetworkState _network;
    private readonly float _floor;

    // The state between frames (DenoiseState).
    private readonly float[] _analysisMem = new float[FrameSize];
    private readonly float[] _cepstralMem = new float[CepsMem * Bands];
    private readonly float[] _synthesisMem = new float[FrameSize];
    private readonly float[] _pitchBuf = new float[PitchBufSize];
    private readonly float[] _lastG = new float[Bands];
    private readonly float[] _memHp = new float[2];
    private int _memId, _lastPeriod;
    private float _lastGain;

    // Scratch, reused every frame.
    private readonly float[] _x = new float[FrameSize];
    private readonly float[] _window = new float[WindowSize];
    private readonly RnNoiseComplex[] _fftIn = new RnNoiseComplex[WindowSize];
    private readonly RnNoiseComplex[] _fftOut = new RnNoiseComplex[WindowSize];
    private readonly RnNoiseComplex[] _spectrum = new RnNoiseComplex[FreqSize];
    private readonly RnNoiseComplex[] _pitchSpectrum = new RnNoiseComplex[FreqSize];
    private readonly float[] _ex = new float[Bands], _ep = new float[Bands], _exp = new float[Bands], _g = new float[Bands];
    private readonly float[] _features = new float[Features];
    private readonly float[] _bandScratch = new float[Bands], _bandScratch2 = new float[Bands], _ly = new float[Bands];
    private readonly float[] _gainsPerBin = new float[FreqSize], _gainsPerBin2 = new float[FreqSize];
    private readonly float[] _decimated = new float[PitchBufSize >> 1];
    private readonly float[] _xLp4 = new float[PitchFrameSize >> 2];
    private readonly float[] _yLp4 = new float[(PitchFrameSize + PitchMaxPeriod - 3 * PitchMinPeriod) >> 2];
    private readonly float[] _xcorr = new float[(PitchMaxPeriod - 3 * PitchMinPeriod) >> 1];
    private readonly float[] _yyLookup = new float[(PitchMaxPeriod >> 1) + 1];

    /// <summary>A channel's denoiser on <paramref name="model"/>; the gains are kept at <paramref name="floor"/> or more (0: none, RNNoise's own).</summary>
    public RnNoiseDenoiser(RnNoiseModel model, float floor = 0)
    {
        _model = model;
        _network = new RnNoiseNetworkState(model);
        _floor = floor;
    }

    /// <summary>
    /// rnnoise_process_frame: <paramref name="input"/>'s 480 samples (16-bit scale) in, the 480 samples a frame before
    /// them out (<paramref name="output"/>, which may be <paramref name="input"/>); the frame's voice probability, 0 for
    /// a silent frame (one whose energy is under RNNoise's floor: it is passed through and the network not run).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public float ProcessFrame(ReadOnlySpan<float> input, Span<float> output)
    {
        var x = _x.AsSpan();
        HighPass(input, x);
        bool silence = FrameFeatures(x);
        float vad = 0;
        var spectrum = _spectrum.AsSpan();
        if (!silence)
        {
            var g = _g.AsSpan();
            vad = _model.Run(_network, _features, g);
            PitchFilter(g);
            for (int i = 0; i < Bands; i++)
            {
                float held = .6f * _lastG[i];
                g[i] = g[i] > held ? g[i] : held;                              // the gains fall at most to 0.6 a frame
                _lastG[i] = g[i];
            }

            if (_floor > 0)
            {
                for (int i = 0; i < Bands; i++)
                {
                    g[i] = g[i] > _floor ? g[i] : _floor;
                }
            }

            var gf = _gainsPerBin.AsSpan();
            gf.Clear();                                                          // over 20 kHz: 0 (as RNNoise leaves it)
            InterpolateBands(gf, g);
            for (int i = 0; i < FreqSize; i++)
            {
                spectrum[i].R *= gf[i];
                spectrum[i].I *= gf[i];
            }
        }

        Synthesis(output);
        return vad;
    }

    // biquad with b = (-2, 1), a = (-1.99599, 0.996): the state in float, each update summed in double.
    private void HighPass(ReadOnlySpan<float> input, Span<float> y)
    {
        const float A0 = (float)-1.99599, A1 = (float)0.99600, B0 = -2, B1 = 1;              // C's float array of double literals
        float m0 = _memHp[0], m1 = _memHp[1];
        for (int i = 0; i < FrameSize; i++)
        {
            float xi = input[i];
            float yi = xi + m0;
            m0 = (float)(m1 + (B0 * (double)xi - A0 * (double)yi));
            m1 = (float)(B1 * (double)xi - A1 * (double)yi);
            y[i] = yi;
        }

        (_memHp[0], _memHp[1]) = (m0, m1);
    }

    // compute_frame_features: true when the frame is silent (its features cleared, the cepstral memory not updated).
    private bool FrameFeatures(ReadOnlySpan<float> x)
    {
        var spectrum = _spectrum.AsSpan();
        var ex = _ex.AsSpan();
        Analysis(x, spectrum, ex);

        var pitchBuf = _pitchBuf.AsSpan();
        pitchBuf[FrameSize..].CopyTo(pitchBuf);
        x.CopyTo(pitchBuf[(PitchBufSize - FrameSize)..]);
        var decimated = _decimated.AsSpan();
        PitchDownsample(pitchBuf, decimated);
        int pitchIndex = PitchSearch(decimated[(PitchMaxPeriod >> 1)..], decimated, PitchFrameSize, PitchMaxPeriod - 3 * PitchMinPeriod);
        pitchIndex = PitchMaxPeriod - pitchIndex;
        float gain = RemoveDoubling(decimated, PitchMaxPeriod, PitchMinPeriod, PitchFrameSize, ref pitchIndex, _lastPeriod, _lastGain);
        _lastPeriod = pitchIndex;
        _lastGain = gain;

        var p = _window.AsSpan();
        pitchBuf.Slice(PitchBufSize - WindowSize - pitchIndex, WindowSize).CopyTo(p);
        ApplyWindow(p);
        var pitchSpectrum = _pitchSpectrum.AsSpan();
        Forward(p, pitchSpectrum);
        var ep = _ep.AsSpan();
        var exp = _exp.AsSpan();
        BandEnergy(ep, pitchSpectrum);
        BandCorrelation(exp, spectrum, pitchSpectrum);
        for (int i = 0; i < Bands; i++)
        {
            exp[i] = (float)(exp[i] / Math.Sqrt(.001 + ex[i] * ep[i]));
        }

        var features = _features.AsSpan();
        var tmp = _bandScratch.AsSpan();
        Dct(tmp, exp);
        for (int i = 0; i < DeltaCeps; i++)
        {
            features[Bands + 2 * DeltaCeps + i] = tmp[i];
        }

        features[Bands + 2 * DeltaCeps] = (float)(features[Bands + 2 * DeltaCeps] - 1.3);
        features[Bands + 2 * DeltaCeps + 1] = (float)(features[Bands + 2 * DeltaCeps + 1] - 0.9);
        features[Bands + 3 * DeltaCeps] = (float)(.01 * (pitchIndex - 300));

        float logMax = -2, follow = -2, e = 0;
        var ly = _ly.AsSpan();
        for (int i = 0; i < Bands; i++)
        {
            ly[i] = (float)Math.Log10(1e-2 + ex[i]);
            double fall = follow - 1.5, inner = fall > ly[i] ? fall : ly[i];
            float low = logMax - 7;
            ly[i] = (float)(low > inner ? low : inner);
            logMax = logMax > ly[i] ? logMax : ly[i];
            follow = (float)(fall > ly[i] ? fall : ly[i]);
            e += ex[i];
        }

        if (e < 0.04)
        {
            // No sound: the state is left as it is.
            features.Clear();
            return true;
        }

        Dct(features, ly);
        features[0] -= 12;
        features[1] -= 4;
        var mem = _cepstralMem.AsSpan();
        var ceps0 = mem.Slice(_memId * Bands, Bands);
        var ceps1 = mem.Slice((_memId < 1 ? CepsMem + _memId - 1 : _memId - 1) * Bands, Bands);
        var ceps2 = mem.Slice((_memId < 2 ? CepsMem + _memId - 2 : _memId - 2) * Bands, Bands);
        features[..Bands].CopyTo(ceps0);
        _memId++;
        for (int i = 0; i < DeltaCeps; i++)
        {
            features[i] = ceps0[i] + ceps1[i] + ceps2[i];
            features[Bands + i] = ceps0[i] - ceps2[i];
            features[Bands + DeltaCeps + i] = ceps0[i] - 2 * ceps1[i] + ceps2[i];
        }

        // The spectral variability.
        if (_memId == CepsMem)
        {
            _memId = 0;
        }

        float variability = 0;
        for (int i = 0; i < CepsMem; i++)
        {
            float least = 1e15f;
            var a = mem.Slice(i * Bands, Bands);
            for (int j = 0; j < CepsMem; j++)
            {
                var b = mem.Slice(j * Bands, Bands);
                float dist = 0;
                for (int k = 0; k < Bands; k++)
                {
                    float d = a[k] - b[k];
                    dist += d * d;
                }

                if (j != i)
                {
                    least = least < dist ? least : dist;
                }
            }

            variability += least;
        }

        features[Bands + 3 * DeltaCeps + 1] = (float)(variability / CepsMem - 2.1);
        return false;
    }

    // frame_analysis: the window over the previous frame and this one, its spectrum and band energies.
    private void Analysis(ReadOnlySpan<float> x, Span<RnNoiseComplex> spectrum, Span<float> ex)
    {
        var window = _window.AsSpan();
        _analysisMem.AsSpan().CopyTo(window);
        x.CopyTo(window[FrameSize..]);
        x.CopyTo(_analysisMem);
        ApplyWindow(window);
        Forward(window, spectrum);
        BandEnergy(ex, spectrum);
    }

    // frame_synthesis: the inverse transform, windowed, overlap-added with the previous frame's second half.
    private void Synthesis(Span<float> output)
    {
        var window = _window.AsSpan();
        Inverse(_spectrum, window);
        ApplyWindow(window);
        for (int i = 0; i < FrameSize; i++)
        {
            output[i] = window[i] + _synthesisMem[i];
        }

        window[FrameSize..].CopyTo(_synthesisMem);
    }

    // pitch_filter: the pitch-delayed spectrum added per band by how periodic the band is against the gain the network
    // gave, then each band brought back to its energy.
    private void PitchFilter(ReadOnlySpan<float> g)
    {
        var spectrum = _spectrum.AsSpan();
        var pitch = _pitchSpectrum.AsSpan();
        var ex = _ex.AsSpan();
        var ep = _ep.AsSpan();
        var exp = _exp.AsSpan();
        var r = _bandScratch.AsSpan();
        for (int i = 0; i < Bands; i++)
        {
            float ri;
            if (exp[i] > g[i])
            {
                ri = 1;
            }
            else
            {
                float exp2 = exp[i] * exp[i], g2 = g[i] * g[i];
                ri = (float)(exp2 * (1 - g2) / (.001 + g2 * (1 - exp2)));
            }

            float clamped = 0 > ri ? 0 : ri;
            clamped = 1 < clamped ? 1 : clamped;
            ri = (float)Math.Sqrt(clamped);
            r[i] = (float)(ri * Math.Sqrt(ex[i] / (1e-8 + ep[i])));
        }

        var rf = _gainsPerBin2.AsSpan();
        rf.Clear();
        InterpolateBands(rf, r);
        for (int i = 0; i < FreqSize; i++)
        {
            spectrum[i].R += rf[i] * pitch[i].R;
            spectrum[i].I += rf[i] * pitch[i].I;
        }

        var newE = _bandScratch2.AsSpan();
        BandEnergy(newE, spectrum);
        var norm = _bandScratch.AsSpan();
        for (int i = 0; i < Bands; i++)
        {
            norm[i] = (float)Math.Sqrt(ex[i] / (1e-8 + newE[i]));
        }

        var normf = _gainsPerBin2.AsSpan();
        normf.Clear();
        InterpolateBands(normf, norm);
        for (int i = 0; i < FreqSize; i++)
        {
            spectrum[i].R *= normf[i];
            spectrum[i].I *= normf[i];
        }
    }

    // compute_band_energy: each bin's power shared between the two bands it lies between (triangles), the ends doubled.
    private static void BandEnergy(Span<float> bandE, ReadOnlySpan<RnNoiseComplex> x)
    {
        Span<float> sum = stackalloc float[Bands];
        sum.Clear();
        for (int i = 0; i < Bands - 1; i++)
        {
            int start = Edges[i] << 2, size = (Edges[i + 1] - Edges[i]) << 2;
            for (int j = 0; j < size; j++)
            {
                float frac = (float)j / size;
                float tmp = x[start + j].R * x[start + j].R;
                tmp += x[start + j].I * x[start + j].I;
                sum[i] += (1 - frac) * tmp;
                sum[i + 1] += frac * tmp;
            }
        }

        sum[0] *= 2;
        sum[Bands - 1] *= 2;
        sum.CopyTo(bandE);
    }

    // compute_band_corr: as BandEnergy, of the real part of x times p's conjugate.
    private static void BandCorrelation(Span<float> bandE, ReadOnlySpan<RnNoiseComplex> x, ReadOnlySpan<RnNoiseComplex> p)
    {
        Span<float> sum = stackalloc float[Bands];
        sum.Clear();
        for (int i = 0; i < Bands - 1; i++)
        {
            int start = Edges[i] << 2, size = (Edges[i + 1] - Edges[i]) << 2;
            for (int j = 0; j < size; j++)
            {
                float frac = (float)j / size;
                float tmp = x[start + j].R * p[start + j].R;
                tmp += x[start + j].I * p[start + j].I;
                sum[i] += (1 - frac) * tmp;
                sum[i + 1] += frac * tmp;
            }
        }

        sum[0] *= 2;
        sum[Bands - 1] *= 2;
        sum.CopyTo(bandE);
    }

    // interp_band_gain: the band values interpolated linearly over the bins up to 20 kHz (the bins above are left as they are).
    private static void InterpolateBands(Span<float> g, ReadOnlySpan<float> bandE)
    {
        for (int i = 0; i < Bands - 1; i++)
        {
            int start = Edges[i] << 2, size = (Edges[i + 1] - Edges[i]) << 2;
            for (int j = 0; j < size; j++)
            {
                float frac = (float)j / size;
                g[start + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
            }
        }
    }

    // dct: the 22-point DCT-II of RNNoise's table, scaled by sqrt(2/22) in double.
    private static void Dct(Span<float> output, ReadOnlySpan<float> input)
    {
        var table = DctTable;
        double scale = Math.Sqrt(2.0 / 22);
        for (int i = 0; i < Bands; i++)
        {
            float sum = 0;
            for (int j = 0; j < Bands; j++)
            {
                sum += input[j] * table[j * Bands + i];
            }

            output[i] = (float)(sum * scale);
        }
    }

    private static void ApplyWindow(Span<float> x)
    {
        var half = HalfWindow;
        for (int i = 0; i < FrameSize; i++)
        {
            x[i] *= half[i];
            x[WindowSize - 1 - i] *= half[i];
        }
    }

    // forward_transform: the spectrum (bins 0 to 480) of 960 real samples.
    private void Forward(ReadOnlySpan<float> input, Span<RnNoiseComplex> output)
    {
        var x = _fftIn.AsSpan();
        for (int i = 0; i < WindowSize; i++)
        {
            x[i].R = input[i];
            x[i].I = 0;
        }

        Fft.Forward(x, _fftOut);
        _fftOut.AsSpan(0, FreqSize).CopyTo(output);
    }

    // inverse_transform: the forward transform of the conjugate-symmetric spectrum, read backwards and scaled by 960.
    private void Inverse(ReadOnlySpan<RnNoiseComplex> input, Span<float> output)
    {
        var x = _fftIn.AsSpan();
        input[..FreqSize].CopyTo(x);
        for (int i = FreqSize; i < WindowSize; i++)
        {
            x[i].R = x[WindowSize - i].R;
            x[i].I = -x[WindowSize - i].I;
        }

        var y = _fftOut.AsSpan();
        Fft.Forward(x, y);
        output[0] = WindowSize * y[0].R;
        for (int i = 1; i < WindowSize; i++)
        {
            output[i] = WindowSize * y[WindowSize - i].R;
        }
    }

    // rnn_pitch_downsample (one channel): the signal halved by a [1/4, 1/2, 1/4] filter, then whitened by a 4th-order LPC
    // with a zero added (a 5-tap FIR).
    private static void PitchDownsample(ReadOnlySpan<float> x, Span<float> xLp)
    {
        int len = x.Length;
        for (int i = 1; i < len >> 1; i++)
        {
            xLp[i] = .5f * (.5f * (x[2 * i - 1] + x[2 * i + 1]) + x[2 * i]);
        }

        xLp[0] = .5f * (.5f * x[1] + x[0]);
        Span<float> ac = stackalloc float[5];
        Autocorrelation(xLp[..(len >> 1)], ac, 4);
        ac[0] *= 1.0001f;                                                        // a noise floor at -40 dB
        for (int i = 1; i <= 4; i++)
        {
            ac[i] -= ac[i] * (.008f * i) * (.008f * i);                          // lag windowing
        }

        Span<float> lpc = stackalloc float[4];
        Lpc(lpc, ac, 4);
        float tmp = 1f;
        for (int i = 0; i < 4; i++)
        {
            tmp = .9f * tmp;
            lpc[i] = lpc[i] * tmp;
        }

        const float C1 = .8f;
        Span<float> lpc2 = stackalloc float[5];
        lpc2[0] = lpc[0] + .8f;
        lpc2[1] = lpc[1] + C1 * lpc[0];
        lpc2[2] = lpc[2] + C1 * lpc[1];
        lpc2[3] = lpc[3] + C1 * lpc[2];
        lpc2[4] = C1 * lpc[3];
        Fir5(xLp[..(len >> 1)], lpc2);
    }

    // celt_fir5 in place, its memory starting at 0.
    private static void Fir5(Span<float> x, ReadOnlySpan<float> num)
    {
        float num0 = num[0], num1 = num[1], num2 = num[2], num3 = num[3], num4 = num[4];
        float mem0 = 0, mem1 = 0, mem2 = 0, mem3 = 0, mem4 = 0;
        for (int i = 0; i < x.Length; i++)
        {
            float sum = x[i];
            sum = sum + num0 * mem0;
            sum = sum + num1 * mem1;
            sum = sum + num2 * mem2;
            sum = sum + num3 * mem3;
            sum = sum + num4 * mem4;
            mem4 = mem3;
            mem3 = mem2;
            mem2 = mem1;
            mem1 = mem0;
            mem0 = x[i];
            x[i] = sum;
        }
    }

    // rnn_autocorr without a window: lags 0 to `lag`, the products over the first n - lag samples then the rest.
    private static void Autocorrelation(ReadOnlySpan<float> x, Span<float> ac, int lag)
    {
        int n = x.Length, fastN = n - lag;
        PitchCorrelation(x, x, ac, fastN, lag + 1);
        for (int k = 0; k <= lag; k++)
        {
            float d = 0;
            for (int i = k + fastN; i < n; i++)
            {
                d = d + x[i] * x[i - k];
            }

            ac[k] += d;
        }
    }

    // rnn_lpc (float): Levinson-Durbin, stopping once the error is 30 dB down.
    private static void Lpc(Span<float> lpc, ReadOnlySpan<float> ac, int p)
    {
        float error = ac[0];
        lpc[..p].Clear();
        if (ac[0] == 0)
        {
            return;
        }

        for (int i = 0; i < p; i++)
        {
            float rr = 0;
            for (int j = 0; j < i; j++)
            {
                rr += lpc[j] * ac[i - j];
            }

            rr += ac[i + 1];
            float r = -rr / error;
            lpc[i] = r;
            for (int j = 0; j < (i + 1) >> 1; j++)
            {
                float tmp1 = lpc[j], tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }

            error = error - r * r * error;
            if (error < .001f * ac[0])
            {
                break;
            }
        }
    }

    // rnn_pitch_xcorr: xcorr[i] = the sum over j of x[j] y[i + j], each in order from j = 0 (its order in the C code,
    // whose unrolled kernel keeps it). The lags are taken a SIMD vector at a time (one x[j] times consecutive y), four
    // vectors at once so their sums run side by side, then one vector, then four lags at a time in scalars: none changes
    // the order of a lag's sum.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PitchCorrelation(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> xcorr, int len, int maxPitch)
    {
        // The unchecked loads below stay inside x and y: checked once here (rule 51).
        if (maxPitch <= 0 || len <= 0 || x.Length < len || y.Length < maxPitch + len - 1 || xcorr.Length < maxPitch)
        {
            throw new ArgumentException($"RNNoise's pitch correlation of {maxPitch} lags over {len} samples does not fit its buffers.");
        }

        ref float xs = ref MemoryMarshal.GetReference(x);
        ref float ys = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int v = Vector<float>.Count;
            for (; i + 4 * v <= maxPitch; i += 4 * v)
            {
                Vector<float> s0 = default, s1 = default, s2 = default, s3 = default;
                for (int j = 0; j < len; j++)
                {
                    var xj = new Vector<float>(Unsafe.Add(ref xs, j));
                    nuint at = (nuint)(i + j);
                    s0 += xj * Vector.LoadUnsafe(ref ys, at);
                    s1 += xj * Vector.LoadUnsafe(ref ys, at + (nuint)v);
                    s2 += xj * Vector.LoadUnsafe(ref ys, at + (nuint)(2 * v));
                    s3 += xj * Vector.LoadUnsafe(ref ys, at + (nuint)(3 * v));
                }

                s0.CopyTo(xcorr[i..]);
                s1.CopyTo(xcorr[(i + v)..]);
                s2.CopyTo(xcorr[(i + 2 * v)..]);
                s3.CopyTo(xcorr[(i + 3 * v)..]);
            }

            for (; i + v <= maxPitch; i += v)
            {
                var s = Vector<float>.Zero;
                for (int j = 0; j < len; j++)
                {
                    s += new Vector<float>(Unsafe.Add(ref xs, j)) * Vector.LoadUnsafe(ref ys, (nuint)(i + j));
                }

                s.CopyTo(xcorr[i..]);
            }
        }

        for (; i + 4 <= maxPitch; i += 4)
        {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int j = 0; j < len; j++)
            {
                float xj = Unsafe.Add(ref xs, j);
                ref float yj = ref Unsafe.Add(ref ys, i + j);
                s0 = s0 + xj * yj;
                s1 = s1 + xj * Unsafe.Add(ref yj, 1);
                s2 = s2 + xj * Unsafe.Add(ref yj, 2);
                s3 = s3 + xj * Unsafe.Add(ref yj, 3);
            }

            (xcorr[i], xcorr[i + 1], xcorr[i + 2], xcorr[i + 3]) = (s0, s1, s2, s3);
        }

        for (; i < maxPitch; i++)
        {
            xcorr[i] = InnerProduct(x, y[i..], len);
        }
    }

    private static float InnerProduct(ReadOnlySpan<float> x, ReadOnlySpan<float> y, int n)
    {
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum = sum + x[i] * y[i];
        }

        return sum;
    }

    // rnn_pitch_search: the coarse search on the signal decimated by 4, the fine one by 2 around the two best, then a
    // pseudo-interpolation; the lag found, in samples of the signal decimated by 2.
    private int PitchSearch(ReadOnlySpan<float> xLp, ReadOnlySpan<float> y, int len, int maxPitch)
    {
        int lag = len + maxPitch;
        var xLp4 = _xLp4.AsSpan(0, len >> 2);
        var yLp4 = _yLp4.AsSpan(0, lag >> 2);
        var xcorr = _xcorr.AsSpan(0, maxPitch >> 1);
        for (int j = 0; j < len >> 2; j++)
        {
            xLp4[j] = xLp[2 * j];
        }

        for (int j = 0; j < lag >> 2; j++)
        {
            yLp4[j] = y[2 * j];
        }

        Span<int> best = stackalloc int[2];
        PitchCorrelation(xLp4, yLp4, xcorr, len >> 2, maxPitch >> 2);
        FindBestPitch(xcorr, yLp4, len >> 2, maxPitch >> 2, best);

        // The fine search: the lags within 2 of either best (doubled), each run of them correlated together.
        for (int i = 0, half = maxPitch >> 1; i < half;)
        {
            if (Math.Abs(i - 2 * best[0]) > 2 && Math.Abs(i - 2 * best[1]) > 2)
            {
                xcorr[i++] = 0;
                continue;
            }

            int start = i;
            while (i < half && (Math.Abs(i - 2 * best[0]) <= 2 || Math.Abs(i - 2 * best[1]) <= 2))
            {
                i++;
            }

            PitchCorrelation(xLp, y[start..], xcorr[start..i], len >> 1, i - start);
            for (int k = start; k < i; k++)
            {
                xcorr[k] = -1 > xcorr[k] ? -1 : xcorr[k];
            }
        }

        FindBestPitch(xcorr, y, len >> 1, maxPitch >> 1, best);
        int offset = 0;
        if (best[0] > 0 && best[0] < (maxPitch >> 1) - 1)
        {
            float a = xcorr[best[0] - 1], b = xcorr[best[0]], c = xcorr[best[0] + 1];
            if (c - a > .7f * (b - a))
            {
                offset = 1;
            }
            else if (a - c > .7f * (b - c))
            {
                offset = -1;
            }
        }

        return 2 * best[0] - offset;
    }

    // find_best_pitch: the two lags of the largest normalized correlation (xcorr² / energy), in float.
    private static void FindBestPitch(ReadOnlySpan<float> xcorr, ReadOnlySpan<float> y, int len, int maxPitch, Span<int> bestPitch)
    {
        float syy = 1;
        float bestNum0 = -1, bestNum1 = -1, bestDen0 = 0, bestDen1 = 0;
        bestPitch[0] = 0;
        bestPitch[1] = 1;
        for (int j = 0; j < len; j++)
        {
            syy = syy + y[j] * y[j];
        }

        for (int i = 0; i < maxPitch; i++)
        {
            if (xcorr[i] > 0)
            {
                float xcorr16 = xcorr[i];
                xcorr16 *= 1e-12f;                                               // no underflow or overflow when squared
                float num = xcorr16 * xcorr16;
                if (num * bestDen1 > bestNum1 * syy)
                {
                    if (num * bestDen0 > bestNum0 * syy)
                    {
                        (bestNum1, bestDen1, bestPitch[1]) = (bestNum0, bestDen0, bestPitch[0]);
                        (bestNum0, bestDen0, bestPitch[0]) = (num, syy, i);
                    }
                    else
                    {
                        (bestNum1, bestDen1, bestPitch[1]) = (num, syy, i);
                    }
                }
            }

            syy += y[i + len] * y[i + len] - y[i] * y[i];
            syy = 1 > syy ? 1 : syy;
        }
    }

    private static readonly int[] SecondCheck = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

    // rnn_remove_doubling: a period at T/k (k = 2 to 15) is taken when it correlates well enough (more easily near the
    // previous frame's period); returns the pitch gain, the period (`t0`) refined by a pseudo-interpolation.
    private float RemoveDoubling(ReadOnlySpan<float> signal, int maxPeriod, int minPeriod, int n, ref int t0Ref, int prevPeriod, float prevGain)
    {
        int minPeriod0 = minPeriod;
        maxPeriod /= 2;
        minPeriod /= 2;
        t0Ref /= 2;
        prevPeriod /= 2;
        n /= 2;
        var x = signal[maxPeriod..];
        if (t0Ref >= maxPeriod)
        {
            t0Ref = maxPeriod - 1;
        }

        int t = t0Ref, t0 = t0Ref;
        Span<int> from = stackalloc int[32];
        Span<float> sums = stackalloc float[32];
        (from[0], from[1]) = (maxPeriod, maxPeriod - t0);
        Correlations(x, signal, from[..2], n, sums);
        float xx = sums[0], xy = sums[1];
        var yyLookup = _yyLookup.AsSpan(0, maxPeriod + 1);
        yyLookup[0] = xx;
        float yy = xx;
        for (int i = 1; i <= maxPeriod; i++)
        {
            float a = signal[maxPeriod - i], b = x[n - i];
            yy = yy + a * a - b * b;
            yyLookup[i] = 0 > yy ? 0 : yy;
        }

        // The candidates T/k (until one is under the shortest period) and their second checks, correlated together.
        int candidates = 0;
        for (int k = 2; k <= 15; k++, candidates++)
        {
            int t1 = (2 * t0 + k) / (2 * k);
            if (t1 < minPeriod)
            {
                break;
            }

            int t1b = k == 2 ? (t1 + t0 > maxPeriod ? t0 : t0 + t1) : (2 * SecondCheck[k] * t0 + k) / (2 * k);
            (from[2 * candidates], from[2 * candidates + 1]) = (maxPeriod - t1, maxPeriod - t1b);
        }

        Correlations(x, signal, from[..(2 * candidates)], n, sums);
        yy = yyLookup[t0];
        float bestXy = xy, bestYy = yy;
        float g0 = PitchGain(xy, xx, yy), g = g0;
        for (int c = 0; c < candidates; c++)
        {
            int k = c + 2, t1 = maxPeriod - from[2 * c], t1b = maxPeriod - from[2 * c + 1];
            xy = .5f * (sums[2 * c] + sums[2 * c + 1]);
            yy = .5f * (yyLookup[t1] + yyLookup[t1b]);
            float g1 = PitchGain(xy, xx, yy);
            float cont = Math.Abs(t1 - prevPeriod) <= 1 ? prevGain : Math.Abs(t1 - prevPeriod) <= 2 && 5 * k * k < t0 ? .5f * prevGain : 0;
            float thresh = Max(.3f, .7f * g0 - cont);
            if (t1 < 3 * minPeriod)
            {
                // Against very short periods (false positives from short-term correlation).
                thresh = Max(.4f, .85f * g0 - cont);
            }
            else if (t1 < 2 * minPeriod)
            {
                thresh = Max(.5f, .9f * g0 - cont);
            }

            if (g1 > thresh)
            {
                (bestXy, bestYy, t, g) = (xy, yy, t1, g1);
            }
        }

        bestXy = 0 > bestXy ? 0 : bestXy;
        float pg = bestYy <= bestXy ? 1f : bestXy / (bestYy + 1);
        for (int k = 0; k < 3; k++)
        {
            from[k] = maxPeriod - (t + k - 1);
        }

        Correlations(x, signal, from[..3], n, sums);
        var xcorr = sums[..3];
        int offset = xcorr[2] - xcorr[0] > .7f * (xcorr[1] - xcorr[0]) ? 1 : xcorr[0] - xcorr[2] > .7f * (xcorr[1] - xcorr[2]) ? -1 : 0;
        if (pg > g)
        {
            pg = g;
        }

        t0Ref = 2 * t + offset;
        if (t0Ref < minPeriod0)
        {
            t0Ref = minPeriod0;
        }

        return pg;

        static float Max(float a, float b) => a > b ? a : b;
    }

    // compute_pitch_gain (float): xy / sqrt(1 + xx yy), the root and the division in double.
    private static float PitchGain(float xy, float xx, float yy) => (float)(xy / Math.Sqrt(1 + xx * yy));

    // sums[c] = the sum over i < n of x[i] signal[from[c] + i], each in order from i = 0 (celt_inner_prod and
    // dual_inner_prod), four sums side by side so their additions overlap.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Correlations(ReadOnlySpan<float> x, ReadOnlySpan<float> signal, ReadOnlySpan<int> from, int n, Span<float> sums)
    {
        // The unchecked reads below stay inside x and signal: checked once here (rule 51).
        foreach (int f in from)
        {
            if (f < 0 || f + n > signal.Length || n > x.Length)
            {
                throw new ArgumentException($"RNNoise's pitch correlation at {f} over {n} samples does not fit its buffer of {signal.Length}.");
            }
        }

        ref float xs = ref MemoryMarshal.GetReference(x);
        ref float ys = ref MemoryMarshal.GetReference(signal);
        int c = 0;
        for (; c + 4 <= from.Length; c += 4)
        {
            ref float y0 = ref Unsafe.Add(ref ys, from[c]);
            ref float y1 = ref Unsafe.Add(ref ys, from[c + 1]);
            ref float y2 = ref Unsafe.Add(ref ys, from[c + 2]);
            ref float y3 = ref Unsafe.Add(ref ys, from[c + 3]);
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < n; i++)
            {
                float xi = Unsafe.Add(ref xs, i);
                s0 = s0 + xi * Unsafe.Add(ref y0, i);
                s1 = s1 + xi * Unsafe.Add(ref y1, i);
                s2 = s2 + xi * Unsafe.Add(ref y2, i);
                s3 = s3 + xi * Unsafe.Add(ref y3, i);
            }

            (sums[c], sums[c + 1], sums[c + 2], sums[c + 3]) = (s0, s1, s2, s3);
        }

        for (; c < from.Length; c++)
        {
            sums[c] = InnerProduct(x, signal[from[c]..], n);
        }
    }

    // sin(π/2 sin²(π/2 (i + .5) / 480)), in double, stored as float.
    private static float[] MakeHalfWindow()
    {
        var w = new float[FrameSize];
        for (int i = 0; i < FrameSize; i++)
        {
            double s = Math.Sin(.5 * Math.PI * (i + .5) / FrameSize);
            w[i] = (float)Math.Sin(.5 * Math.PI * s * s);
        }

        return w;
    }

    // cos((i + .5) j π / 22) in double, as float; the first column times sqrt(.5).
    private static float[] MakeDctTable()
    {
        var t = new float[Bands * Bands];
        for (int i = 0; i < Bands; i++)
        {
            for (int j = 0; j < Bands; j++)
            {
                t[i * Bands + j] = (float)Math.Cos((i + .5) * j * Math.PI / Bands);
                if (j == 0)
                {
                    t[i * Bands + j] = (float)(t[i * Bands + j] * Math.Sqrt(.5));
                }
            }
        }

        return t;
    }
}
