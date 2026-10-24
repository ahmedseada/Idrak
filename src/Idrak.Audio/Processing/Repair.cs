// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio;

// The noise and repair processors (plan 31, P3): denoise-live, declick, declip, deess and dereverb (live) and
// spectral-gate (whole recording). Each is checked against tools/audio/make_audio_repair.py, which writes the same
// equations frame by frame or sample by sample in float64 (numpy's FFT for the spectra), and asserts each one's effect
// on made signals. The live stages allocate when opened and nothing per block, give the block back in place, report
// their look-ahead as Latency (their output that many samples late: silence first) and give the tail when flushed, so
// blocks of any size give Apply's samples bit for bit. No option key is a processor's name (an item after a comma that
// names an option of the step before it joins that step; a registered name starts a step of its own).
internal static class RepairProcessors
{
    public const string DenoiseLive = "denoise-live", SpectralGate = "spectral-gate", Declick = "declick", Declip = "declip", Deess = "deess", Dereverb = "dereverb";

    // The library's noise and repair processors, registered after the dynamics (AudioProcessors.BuiltIn).
    public static void RegisterDefaults(SlotTable<string, IAudioProcessor> table)
    {
        ArgumentNullException.ThrowIfNull(table);
        table.RegisterDefault(DenoiseLive, new LiveDenoiseProcessor());
        table.RegisterDefault(SpectralGate, new SpectralGateProcessor());
        table.RegisterDefault(Declick, new DeclickProcessor());
        table.RegisterDefault(Declip, new DeclipProcessor());
        table.RegisterDefault(Deess, new DeessProcessor());
        table.RegisterDefault(Dereverb, new DereverbProcessor());
    }

    // The largest power of two of samples not over `seconds` (at least 16), as denoise takes its window.
    public static int WindowSize(double seconds, int sampleRate)
    {
        int size = 16;
        while (size * 2 <= seconds * sampleRate && size < 1 << 24)
        {
            size *= 2;
        }

        return size;
    }

    // The one-pole coefficient of a time constant of `milliseconds` at one step every `hop` samples (0: at once).
    public static double FrameCoefficient(double milliseconds, int hop, int sampleRate) => milliseconds <= 0 ? 0 : Math.Exp(-hop / (milliseconds / 1000.0 * sampleRate));

    public static void ThrowIfOther(AudioData block, int sampleRate, int channels, string name)
    {
        ArgumentNullException.ThrowIfNull(block);
        if (block.SampleRate != sampleRate || block.Channels != channels)
        {
            throw new ArgumentException($"{name} was opened for {channels} channel(s) at {sampleRate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
        }
    }

    public static AudioProcessorParameter Window(string help) => new("window", "Window", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Max = 1, Step = 0.008, Unit = "s", Optional = true, Default = "0.032", Help = help,
    };

    public static AudioProcessorParameter Floor(string value) => new("floor", "Floor", AudioProcessorInput.Number)
    {
        Max = 0, Step = 1, Unit = "dB", Optional = true, Default = value, Help = "The most a frequency is turned down",
    };

    public static AudioProcessorParameter Order(string value) => new("order", "Predictor order", AudioProcessorInput.Integer)
    {
        Min = 1, Max = 64, Step = 1, Optional = true, Default = value, Help = "How many samples back the linear predictor reaches (the AR model the gaps are filled from)",
    };

    public static AudioProcessorParameter Frame(string value) => new("frame", "Frame", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Max = 1000, Step = 5, Unit = "ms", Optional = true, Default = value,
        Help = "The stretch the predictor is fitted over (and the residual's scale read from); the output lags by it plus the look-ahead",
    };

    public static AudioProcessorParameter Length(string value, string help) => new("length", "Longest", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Max = 100, Step = 0.5, Unit = "ms", Optional = true, Default = value, Help = help,
    };
}

// A real FFT of a power-of-two size N: the complex FFT of N / 2 points (iterative radix 2, decimation in time, its
// twiddles worked out once) of the even samples plus i times the odd ones, then split into the N / 2 + 1 bins
// (numpy's rfft and irfft, in double). Its scratch is its own: one per stage.
internal sealed class RealFft
{
    private readonly int _size, _half;
    private readonly int[] _reverse;
    private readonly double[] _cos, _sin;                                       // e^(-2πi j / (N/2)), j < N / 4
    private readonly double[] _splitCos, _splitSin;                             // e^(-2πi k / N), k ≤ N / 2
    private readonly double[] _re, _im;

    public RealFft(int size)
    {
        if (size < 4 || (size & (size - 1)) != 0)
        {
            throw new ArgumentException($"The real FFT takes a power of two of at least 4 points, not {size}.", nameof(size));
        }

        (_size, _half) = (size, size / 2);
        _reverse = new int[_half];
        int bits = System.Numerics.BitOperations.Log2((uint)_half);
        for (int i = 0; i < _half; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++)
            {
                r |= ((i >> b) & 1) << (bits - 1 - b);
            }

            _reverse[i] = r;
        }

        _cos = new double[Math.Max(1, _half / 2)];
        _sin = new double[_cos.Length];
        for (int j = 0; j < _half / 2; j++)
        {
            _cos[j] = Math.Cos(2 * Math.PI * j / _half);
            _sin[j] = -Math.Sin(2 * Math.PI * j / _half);
        }

        _splitCos = new double[_half + 1];
        _splitSin = new double[_half + 1];
        for (int k = 0; k <= _half; k++)
        {
            _splitCos[k] = Math.Cos(2 * Math.PI * k / size);
            _splitSin[k] = -Math.Sin(2 * Math.PI * k / size);
        }

        _re = new double[_half];
        _im = new double[_half];
    }

    public int Bins => _half + 1;

    // X = rfft(x): x has N samples, real and imaginary N / 2 + 1 (the imaginary parts at 0 Hz and N / 2 exactly 0).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<double> x, Span<double> real, Span<double> imaginary)
    {
        int half = _half;
        for (int i = 0; i < half; i++)
        {
            int r = _reverse[i];
            _re[r] = x[2 * i];
            _im[r] = x[2 * i + 1];
        }

        Butterflies();
        for (int k = 0; k <= half; k++)
        {
            int a = k == half ? 0 : k, b = k == 0 ? 0 : half - k;
            double er = (_re[a] + _re[b]) / 2, ei = (_im[a] - _im[b]) / 2;
            double oddRe = (_im[a] + _im[b]) / 2, oddIm = -(_re[a] - _re[b]) / 2;
            double wr = _splitCos[k], wi = _splitSin[k];
            real[k] = er + (wr * oddRe - wi * oddIm);
            imaginary[k] = ei + (wr * oddIm + wi * oddRe);
        }

        imaginary[0] = 0;
        imaginary[half] = 0;
    }

    // x = irfft(X) of N points (the imaginary parts at 0 Hz and N / 2 left out, as numpy's irfft does).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Inverse(ReadOnlySpan<double> real, ReadOnlySpan<double> imaginary, Span<double> x)
    {
        int half = _half;
        for (int k = 0; k < half; k++)
        {
            int b = half - k;
            double xr = real[k], xi = k == 0 ? 0 : imaginary[k];
            double cr = real[b], ci = b == half ? 0 : -imaginary[b];             // conj(X[N/2 - k])
            double er = (xr + cr) / 2, ei = (xi + ci) / 2;
            double dr = (xr - cr) / 2, di = (xi - ci) / 2;
            double wr = _splitCos[k], wi = -_splitSin[k];                       // W^-k = e^(+2πi k / N)
            double oddRe = dr * wr - di * wi, oddIm = dr * wi + di * wr;
            // Z = E + iO, conjugated for the inverse through the forward butterflies.
            int r = _reverse[k];
            _re[r] = er - oddIm;
            _im[r] = -(ei + oddRe);
        }

        Butterflies();
        double scale = 1.0 / half;
        for (int i = 0; i < half; i++)
        {
            x[2 * i] = _re[i] * scale;
            x[2 * i + 1] = -_im[i] * scale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Butterflies()
    {
        var re = _re.AsSpan(0, _half);
        var im = _im.AsSpan(0, _half);
        for (int length = 2; length <= _half; length <<= 1)
        {
            int halfLength = length / 2, step = _half / length;
            for (int i = 0; i < _half; i += length)
            {
                for (int k = 0; k < halfLength; k++)
                {
                    double wr = _cos[k * step], wi = _sin[k * step];
                    int a = i + k, b = a + halfLength;
                    double tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
}

// The streaming STFT the live spectral stages share (denoise-live, dereverb), each channel on its own. Frame k holds the
// samples (k + 1)H - N to (k + 1)H (N a power of two, H = N / 4; zeros before the start), windowed by a periodic Hann
// (0.5 - 0.5 cos(2π n / N)); its spectrum (RealFft) goes through the stage's frame function, back, windowed again and
// overlap-added in double; a sample, once the last frame over it is in, is that sum divided by the squares of the four
// windows over it (1.5 for a Hann at a quarter hop). Frame k is worked out when its last sample comes in, so a sample
// is ready N - 1 samples after it came in: the output lags by N - 1 (the stage's Latency, silence first) and Flush runs
// N - 1 zeros through to give the tail. A frame is "inside" when it holds no zero from before the start or from the
// flush (the frame function's noise readings take only those).
internal sealed class SpectralStream
{
    private readonly int _size, _hop, _bins, _channels;
    private readonly double[] _window, _norm;
    private readonly RealFft _fft;
    private readonly float[] _input;                                            // per channel, the last N samples (a ring)
    private readonly double[] _sum;                                             // per channel, the overlap-add (a ring of N)
    private readonly float[] _ready;                                            // per channel, the H samples the last frame finished
    private readonly double[] _frame, _real, _imaginary;
    private readonly FrameOf _process;
    private long _taken;                                                        // samples taken per channel so far
    private long _end = long.MaxValue;                                          // the recording's length, once flushing

    public SpectralStream(int size, int sampleRate, int channels, FrameOf process)
    {
        (_size, _hop, _channels, _process, SampleRate) = (size, size / 4, channels, process, sampleRate);
        _fft = new RealFft(size);
        _bins = _fft.Bins;
        _window = new double[size];
        for (int n = 0; n < size; n++)
        {
            _window[n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / size);
        }

        _norm = new double[_hop];
        for (int i = 0; i < _hop; i++)
        {
            double s = 0;
            for (int j = 0; j < 4; j++)
            {
                s += _window[i + j * _hop] * _window[i + j * _hop];
            }

            _norm[i] = s;
        }

        _input = new float[channels * size];
        _sum = new double[channels * size];
        _ready = new float[channels * _hop];
        _frame = new double[size];
        _real = new double[_bins];
        _imaginary = new double[_bins];
    }

    // One frame's spectrum of channel `channel`, changed in place; `inside` when the frame holds only the recording.
    public delegate void FrameOf(int channel, Span<double> real, Span<double> imaginary, bool inside);

    public int Size => _size;

    public int Hop => _hop;

    public int Bins => _bins;

    public int SampleRate { get; }

    public int Latency => _size - 1;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public AudioData Process(AudioData block, string name)
    {
        RepairProcessors.ThrowIfOther(block, SampleRate, _channels, name);
        int length = block.Length;
        var samples = block.Samples;
        for (int c = 0; c < _channels; c++)
        {
            var x = samples.AsSpan(c * length, length);
            var input = _input.AsSpan(c * _size, _size);
            var ready = _ready.AsSpan(c * _hop, _hop);
            long t = _taken;
            int at = (int)(t % _size), phase = (int)((t + 1) % _hop);
            for (int i = 0; i < x.Length; i++, t++)
            {
                input[at] = x[i];
                at = at + 1 == _size ? 0 : at + 1;
                if (phase == 0)
                {
                    Frame(c, t);
                }

                x[i] = t >= _size - 1 ? ready[phase] : 0f;
                phase = phase + 1 == _hop ? 0 : phase + 1;
            }
        }

        _taken += length;
        return block;
    }

    public AudioData Flush(string name)
    {
        _end = Math.Min(_end, _taken);
        return Process(new AudioData(new float[_channels * (_size - 1)], _channels, SampleRate), name);
    }

    // Frame (t + 1) / H - 1 of channel c, its last sample t just in.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Frame(int c, long t)
    {
        var input = _input.AsSpan(c * _size, _size);
        var sum = _sum.AsSpan(c * _size, _size);
        int start = (int)((t + 1) % _size);                                     // the oldest of the last N samples, in the ring
        for (int n = 0; n < _size; n++)
        {
            int k = start + n;
            _frame[n] = input[k < _size ? k : k - _size] * _window[n];
        }

        _fft.Forward(_frame, _real, _imaginary);
        _process(c, _real, _imaginary, t + 1 - _size >= 0 && t + 1 <= _end);
        _fft.Inverse(_real, _imaginary, _frame);
        for (int n = 0; n < _size; n++)
        {
            int k = start + n;
            sum[k < _size ? k : k - _size] += _frame[n] * _window[n];
        }

        var ready = _ready.AsSpan(c * _hop, _hop);
        for (int j = 0; j < _hop; j++)
        {
            int k = start + j;
            k = k < _size ? k : k - _size;
            ready[j] = (float)(sum[k] / _norm[j]);
            sum[k] = 0;                                                         // the slot of sample t + 1 + j, coming next
        }
    }
}

// "denoise-live" (live): spectral gating as denoise does, on the streaming STFT (SpectralStream: Hann of the largest
// power of two of samples under `window` s, a quarter hop; latency N - 1) against a running noise floor (a minimum
// statistic, after Martin 2001, without his bias correction): S, each frequency's magnitude smoothed over the frames
// (one pole, `smooth` ms at the frame rate; the first frame inside the recording starts it); the noise, `bias` times
// S's least value over the last round(track · fs / H) inside frames (bias 1.6: white noise's mean magnitude over that
// minimum at the default window and smoothing). A frame's gain at a frequency is (|X| - value · noise) / |X| within
// [floor, 1] (`floor` where |X| is 0), averaged with the gains of the `frames` - 1 frames before it (no musical noise
// without looking ahead); before the first inside frame the sound passes as it is, and the frames of the flush use the
// last floor read.
internal sealed class LiveDenoiseProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.DenoiseLive;

    public string Label => "Denoise (live)";

    public string Summary => "spectral gating against a running noise floor (the least smoothed magnitude over the last `track` s), taken off value times (default 1.5), no frequency turned down past `floor` dB; lags by the window";

    public bool Live => true;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Strength", AudioProcessorInput.Number)
        {
            Min = 0, Step = 0.1, Unit = "x", Optional = true, Default = "1.5", Help = "How many times the noise floor is taken off (1: the floor itself; more removes more noise and thins the voice)",
        },
        RepairProcessors.Floor("-25"),
        RepairProcessors.Window("The window the spectrum is taken over (rounded down to a power of two of samples; the hop a quarter of it; the output lags by it)"),
        new("track", "Noise memory", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 60, Step = 0.5, Unit = "s", Optional = true, Default = "1.5",
            Help = "How far back the noise floor looks for its least level (longer: steadier under speech, slower to follow a noise that rises)",
        },
        new("smooth", "Smoothing", AudioProcessorInput.Number)
        {
            Min = 0, Max = 10000, Step = 10, Unit = "ms", Optional = true, Default = "50", Help = "The time constant of the magnitudes the noise floor is read from",
        },
        new("bias", "Bias", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 10, Step = 0.1, Unit = "x", Optional = true, Default = "1.6",
            Help = "The noise floor over the least smoothed magnitude (a minimum is under the mean; 1.6 is white noise's at the defaults)",
        },
        new("frames", "Gain frames", AudioProcessorInput.Integer)
        {
            Min = 1, Max = 64, Step = 1, Optional = true, Default = "2", Help = "How many frames each gain is averaged over (the frame and those before it)",
        },
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return new Stage(step, sampleRate, channels);
    }

    private sealed class Stage : IAudioStage
    {
        private readonly SpectralStream _stream;
        private readonly double _strength, _floor, _smooth, _bias;
        private readonly int _bins, _depth, _frames;
        private readonly double[] _smoothed, _noise, _magnitude, _gains;
        private readonly double[] _minValue;                                    // per channel and bin: the running minimum's queue over the last D inside frames
        private readonly long[] _minIndex;
        private readonly int[] _minHead, _minCount, _gainCount, _gainSlot;
        private readonly long[] _inside;                                        // per channel: inside frames so far
        private readonly bool[] _started;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            int size = RepairProcessors.WindowSize(step.Number("window", 0.032), sampleRate);
            _stream = new SpectralStream(size, sampleRate, channels, Frame);
            (Channels, _bins) = (channels, _stream.Bins);
            _strength = step.Number(1.5);
            _floor = DynamicsMath.Gain(step.Number("floor", -25));
            _smooth = RepairProcessors.FrameCoefficient(step.Number("smooth", 50), _stream.Hop, sampleRate);
            _bias = step.Number("bias", 1.6);
            _depth = Math.Max(1, (int)Math.Round(step.Number("track", 1.5) * sampleRate / _stream.Hop));
            _frames = step.Integer("frames", 2);
            (_smoothed, _noise, _magnitude) = (new double[channels * _bins], new double[channels * _bins], new double[_bins]);
            _gains = new double[channels * _frames * _bins];
            (_minValue, _minIndex) = (new double[channels * _bins * _depth], new long[channels * _bins * _depth]);
            (_minHead, _minCount) = (new int[channels * _bins], new int[channels * _bins]);
            (_gainCount, _gainSlot, _inside, _started) = (new int[channels], new int[channels], new long[channels], new bool[channels]);
        }

        public int SampleRate => _stream.SampleRate;

        public int Channels { get; }

        public int Latency => _stream.Latency;

        public AudioData Process(AudioData block) => _stream.Process(block, RepairProcessors.DenoiseLive);

        public AudioData Flush() => _stream.Flush(RepairProcessors.DenoiseLive);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Frame(int c, Span<double> real, Span<double> imaginary, bool inside)
        {
            int bins = _bins;
            var m = _magnitude.AsSpan();
            for (int b = 0; b < bins; b++)
            {
                m[b] = Math.Sqrt(real[b] * real[b] + imaginary[b] * imaginary[b]);
            }

            var s = _smoothed.AsSpan(c * bins, bins);
            var noise = _noise.AsSpan(c * bins, bins);
            if (inside)
            {
                if (!_started[c])
                {
                    m.CopyTo(s);
                    _started[c] = true;
                }
                else
                {
                    for (int b = 0; b < bins; b++)
                    {
                        s[b] = _smooth * s[b] + (1 - _smooth) * m[b];
                    }
                }

                long index = _inside[c]++;
                for (int b = 0; b < bins; b++)
                {
                    noise[b] = _bias * Least(c * bins + b, s[b], index);
                }
            }

            var gains = _gains.AsSpan(c * _frames * bins, _frames * bins);
            var g = gains.Slice(_gainSlot[c] * bins, bins);
            for (int b = 0; b < bins; b++)
            {
                g[b] = !_started[c] ? 1.0 : m[b] > 0 ? Math.Clamp((m[b] - _strength * noise[b]) / m[b], _floor, 1.0) : _floor;
            }

            int count = _gainCount[c] = Math.Min(_gainCount[c] + 1, _frames);
            int newest = _gainSlot[c];
            _gainSlot[c] = newest + 1 == _frames ? 0 : newest + 1;
            int oldest = newest - count + 1 < 0 ? newest - count + 1 + _frames : newest - count + 1;
            for (int b = 0; b < bins; b++)
            {
                double sum = 0;
                for (int k = 0, slot = oldest; k < count; k++, slot = slot + 1 == _frames ? 0 : slot + 1)
                {
                    sum += gains[slot * bins + b];                              // the oldest first, as the reference sums them
                }

                double gain = Math.Clamp(sum / count, _floor, 1.0);
                real[b] *= gain;
                imaginary[b] *= gain;
            }
        }

        // The least of the last D values of queue `q` after `value` (its index-th) is added: a running minimum's queue
        // (values rising from the head; a value is dropped once a later one is no larger, or once it is D values old).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double Least(int q, double value, long index)
        {
            var values = _minValue.AsSpan(q * _depth, _depth);
            var indices = _minIndex.AsSpan(q * _depth, _depth);
            int head = _minHead[q], count = _minCount[q];
            if (count > 0 && indices[head] <= index - _depth)
            {
                head = head + 1 == _depth ? 0 : head + 1;
                count--;
            }

            int back = head + count < _depth ? head + count : head + count - _depth;
            while (count > 0 && values[back == 0 ? _depth - 1 : back - 1] >= value)
            {
                count--;
                back = back == 0 ? _depth - 1 : back - 1;
            }

            (values[back], indices[back]) = (value, index);
            (_minHead[q], _minCount[q]) = (head, count + 1);
            return values[head];
        }
    }
}

// "dereverb" (live): late reverberation taken off (Lebart, Boucher and Denbigh, 2001) on the streaming STFT
// (SpectralStream; latency N - 1). P, each frequency's power smoothed over the frames (one pole, `smooth` ms at the frame
// rate, from 0); the late reverberation's power is P of D = max(1, round(delay · fs / H)) frames before, decayed over
// those frames as an exponential of the value `t60` gives: exp(-2 δ D H / fs), δ = 3 ln 10 / T60; the gain is
// sqrt(1 - value · late / P), no lower than `floor` dB (1 where P is 0).
internal sealed class DereverbProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.Dereverb;

    public string Label => "Dereverb";

    public string Summary => "late reverberation taken off: each frequency's power `delay` ms back, decayed as a room of `t60` s would, taken off value times (default 1), no frequency turned down past `floor` dB; lags by the window";

    public bool Live => true;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Strength", AudioProcessorInput.Number)
        {
            Min = 0, Max = 10, Step = 0.1, Unit = "x", Optional = true, Default = "1", Help = "How many times the late reverberation's estimate is taken off",
        },
        new("t60", "Reverberation time", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 20, Step = 0.1, Unit = "s", Optional = true, Default = "0.5", Help = "The room's T60: how long its echo takes to fall by 60 dB",
        },
        new("delay", "Late after", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1000, Step = 5, Unit = "ms", Optional = true, Default = "50", Help = "Where the late reverberation starts (the early echoes before it are kept)",
        },
        RepairProcessors.Floor("-15"),
        RepairProcessors.Window("The window the spectrum is taken over (rounded down to a power of two of samples; the hop a quarter of it; the output lags by it)"),
        new("smooth", "Smoothing", AudioProcessorInput.Number)
        {
            Min = 0, Max = 10000, Step = 5, Unit = "ms", Optional = true, Default = "20", Help = "The time constant of the power the reverberation is read from",
        },
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return new Stage(step, sampleRate, channels);
    }

    private sealed class Stage : IAudioStage
    {
        private readonly SpectralStream _stream;
        private readonly double _strength, _floor, _smooth, _decay;
        private readonly int _bins, _delay;
        private readonly double[] _history;                                     // per channel: P of the last D + 1 frames (a ring)
        private readonly long[] _count;                                         // per channel: frames so far

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            int size = RepairProcessors.WindowSize(step.Number("window", 0.032), sampleRate);
            _stream = new SpectralStream(size, sampleRate, channels, Frame);
            (Channels, _bins) = (channels, _stream.Bins);
            int hop = _stream.Hop;
            _strength = step.Number(1);
            _floor = DynamicsMath.Gain(step.Number("floor", -15));
            _smooth = RepairProcessors.FrameCoefficient(step.Number("smooth", 20), hop, sampleRate);
            _delay = Math.Max(1, (int)Math.Round(step.Number("delay", 50) / 1000.0 * sampleRate / hop));
            _decay = Math.Exp(-2 * (3 * Math.Log(10) / step.Number("t60", 0.5)) * _delay * hop / sampleRate);
            _history = new double[channels * (_delay + 1) * _bins];
            _count = new long[channels];
        }

        public int SampleRate => _stream.SampleRate;

        public int Channels { get; }

        public int Latency => _stream.Latency;

        public AudioData Process(AudioData block) => _stream.Process(block, RepairProcessors.Dereverb);

        public AudioData Flush() => _stream.Flush(RepairProcessors.Dereverb);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Frame(int c, Span<double> real, Span<double> imaginary, bool inside)
        {
            _ = inside;
            int bins = _bins, slots = _delay + 1;
            long k = _count[c]++;
            var history = _history.AsSpan(c * slots * bins, slots * bins);
            int now = (int)(k % slots), then = (int)((k + 1) % slots);           // the slot of D frames back is the next one
            var before = history.Slice((now == 0 ? slots - 1 : now - 1) * bins, bins);
            var p = history.Slice(now * bins, bins);
            var late = history.Slice(then * bins, bins);
            bool reaches = k >= _delay;
            for (int b = 0; b < bins; b++)
            {
                double power = real[b] * real[b] + imaginary[b] * imaginary[b];
                double smoothed = k == 0 ? (1 - _smooth) * power : _smooth * before[b] + (1 - _smooth) * power;
                double reverb = reaches ? _decay * late[b] : 0;
                p[b] = smoothed;
                double ratio = smoothed > 0 ? reverb / smoothed : 0;
                double gain = Math.Max(Math.Sqrt(Math.Max(0, 1 - _strength * ratio)), _floor);
                real[b] *= gain;
                imaginary[b] *= gain;
            }
        }
    }
}

// "spectral-gate" (whole recording): noisereduce's stationary gate (Sainburg). The spectrum as denoise takes it (the
// library's STFT: Hann of the largest power of two of samples under `window` s, a quarter hop, centred); the noise
// profile, each frequency's mean and standard deviation in dB (10 log10 of the power, at least -200 dB) over the
// quietest `quantile` share of the frames (by their energy; equal energies in order); a bin is signal when its power is
// over mean + value · deviation; the 0/1 mask smoothed by triangles (noisereduce's shape) reaching floor(smooth / hop)
// frames and floor(spread / (fs / N)) bins each side, divided by the weights that fall inside (the edges keep their
// level); the gain reduce · mask + 1 - reduce; the inverse STFT to the recording's length. Needs the whole recording
// for its profile; live sound takes denoise-live.
internal sealed class SpectralGateProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.SpectralGate;

    public string Label => "Spectral gate";

    public string Summary => "noisereduce's stationary gate: a bin passes when over the noise profile's mean plus value (default 1.5) deviations, the mask smoothed over `smooth` ms and `spread` Hz, the rest turned down by `reduce`";

    public bool Live => false;

    public string? LiveAlternative => RepairProcessors.DenoiseLive;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Threshold", AudioProcessorInput.Number)
        {
            Min = -10, Max = 10, Step = 0.1, Unit = "sigma", Optional = true, Default = "1.5", Help = "How many of the noise's standard deviations (in dB) over its mean a bin must be to pass",
        },
        new("reduce", "Reduction", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1, Step = 0.05, Optional = true, Default = "1", Help = "How much of a gated bin is taken off (1: all of it; 0.9: down 20 dB)",
        },
        new("quantile", "Noise share", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1, Step = 0.05, Optional = true, Default = "0.5", Help = "The share of the frames, the quietest, the noise profile is read from (1: all, as noisereduce)",
        },
        new("smooth", "Time smoothing", AudioProcessorInput.Number)
        {
            Min = 0, Max = 10000, Step = 10, Unit = "ms", Optional = true, Default = "50", Help = "How far the mask is smoothed over time on each side",
        },
        new("spread", "Frequency smoothing", AudioProcessorInput.Number)
        {
            Min = 0, Max = 24000, Step = 10, Unit = "Hz", Optional = true, Default = "40", Help = "How far the mask is smoothed over frequency on each side",
        },
        RepairProcessors.Window("The window the spectrum is taken over (rounded down to a power of two of samples; the hop a quarter of it)"),
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(step);
        int rate = audio.SampleRate, size = RepairProcessors.WindowSize(step.Number("window", 0.032), rate), hop = size / 4;
        if (audio.Length < 2 * size)
        {
            return audio;
        }

        var stft = new AudioFeatureOptions { SampleRate = rate, WindowSize = size, HopSize = hop, Window = "hann", Center = true, PadSeconds = null };
        int length = audio.Length, frames = Stft.FramesFor(length, stft), bins = stft.Bins;
        double threshold = step.Number(1.5), reduce = step.Number("reduce", 1);
        int quiet = Math.Clamp((int)Math.Round(step.Number("quantile", 0.5) * frames), 1, frames);
        int across = (int)Math.Floor(step.Number("smooth", 50) / 1000.0 * rate / hop), along = (int)Math.Floor(step.Number("spread", 40) / ((double)rate / size));
        var real = GC.AllocateUninitializedArray<float>(frames * bins);
        var imaginary = GC.AllocateUninitializedArray<float>(frames * bins);
        var power = GC.AllocateUninitializedArray<double>(frames * bins);
        var mask = GC.AllocateUninitializedArray<double>(frames * bins);
        var smoothed = GC.AllocateUninitializedArray<double>(frames * bins);
        var energy = new double[frames];
        var order = new int[frames];
        var mean = new double[bins];
        var deviation = new double[bins];
        var output = GC.AllocateUninitializedArray<float>(audio.Samples.Length);
        Comparison<int> quieter = (a, b) => energy[a] != energy[b] ? energy[a].CompareTo(energy[b]) : a.CompareTo(b);
        for (int c = 0; c < audio.Channels; c++)
        {
            Stft.ComplexOnHost(audio.Channel(c), stft, real, imaginary);
            for (int f = 0; f < frames; f++)
            {
                double sum = 0;
                for (int b = 0; b < bins; b++)
                {
                    int i = f * bins + b;
                    double r = real[i], m = imaginary[i];
                    power[i] = r * r + m * m;
                    sum += power[i];
                }

                energy[f] = sum;
                order[f] = f;
            }

            Array.Sort(order, quieter);                                          // the quietest frames first (equal energies in order)
            Array.Clear(mean);
            Array.Clear(deviation);
            for (int q = 0; q < quiet; q++)
            {
                var p = power.AsSpan(order[q] * bins, bins);
                for (int b = 0; b < bins; b++)
                {
                    mean[b] += Decibels(p[b]);
                }
            }

            for (int b = 0; b < bins; b++)
            {
                mean[b] /= quiet;
            }

            for (int q = 0; q < quiet; q++)
            {
                var p = power.AsSpan(order[q] * bins, bins);
                for (int b = 0; b < bins; b++)
                {
                    double d = Decibels(p[b]) - mean[b];
                    deviation[b] += d * d;
                }
            }

            for (int b = 0; b < bins; b++)
            {
                deviation[b] = Math.Pow(10, (mean[b] + threshold * Math.Sqrt(deviation[b] / quiet)) / 10);   // the power a bin must pass
            }

            for (int i = 0; i < frames * bins; i++)
            {
                mask[i] = power[i] > deviation[i % bins] ? 1 : 0;
            }

            SmoothFrames(mask, smoothed, frames, bins, across);
            SmoothBins(smoothed, mask, frames, bins, along);
            for (int i = 0; i < frames * bins; i++)
            {
                float gain = (float)(reduce * mask[i] + (1 - reduce));
                real[i] *= gain;
                imaginary[i] *= gain;
            }

            var samples = Stft.Inverse(real, imaginary, frames, stft, length);
            samples.AsSpan(0, length).CopyTo(output.AsSpan(c * length, length));
        }

        return new AudioData(output, audio.Channels, audio.SampleRate);
    }

    private static double Decibels(double power) => 10 * Math.Log10(Math.Max(power, 1e-20));

    // noisereduce's triangle: weight (half + 1 - |j|) / (half + 1) at j = -half..half.
    private static double Weight(int j, int half) => (half + 1 - Math.Abs(j)) / (double)(half + 1);

    // Over the frames (rows of `bins`, walked a row at a time): each value the weighted mean of the frames within `half`.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SmoothFrames(double[] source, double[] target, int frames, int bins, int half)
    {
        for (int f = 0; f < frames; f++)
        {
            var row = target.AsSpan(f * bins, bins);
            int lo = Math.Max(0, f - half), hi = Math.Min(frames - 1, f + half);
            row.Clear();
            double total = 0;
            for (int g = lo; g <= hi; g++)
            {
                double w = Weight(g - f, half);
                total += w;
                var from = source.AsSpan(g * bins, bins);
                for (int b = 0; b < bins; b++)
                {
                    row[b] += from[b] * w;
                }
            }

            for (int b = 0; b < bins; b++)
            {
                row[b] /= total;
            }
        }
    }

    // Over the bins of each frame: each value the weighted mean of the bins within `half`.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SmoothBins(double[] source, double[] target, int frames, int bins, int half)
    {
        for (int f = 0; f < frames; f++)
        {
            var from = source.AsSpan(f * bins, bins);
            var row = target.AsSpan(f * bins, bins);
            for (int b = 0; b < bins; b++)
            {
                int lo = Math.Max(0, b - half), hi = Math.Min(bins - 1, b + half);
                double sum = 0, total = 0;
                for (int k = lo; k <= hi; k++)
                {
                    double w = Weight(k - b, half);
                    sum += from[k] * w;
                    total += w;
                }

                row[b] = sum / total;
            }
        }
    }
}

// "deess" (live): a split-band compressor on the sibilance band. The band is split off by a biquad (P1's form, run in
// double, settled on the first sample: its input memory that sample, its output that sample times its gain at 0 Hz):
// mode high, the RBJ low-pass at `frequency`, `q`, the band x less it (everything over the frequency; x - band is the
// low-pass itself); mode band, the RBJ band-pass of 0 dB peak gain, the band its output (x - band the RBJ notch). The
// level is a one-pole mean square (`window`) of the band's largest square over the linked channels; the compressor's
// curve (compress's: the value, `ratio`, `knee`) gives the reduction in dB, smoothed with `attack` (down) and `release`
// (up) as P2's followers are; the output x - (1 - 10^(reduction / 20)) · band: the band turned down, the rest as it was.
internal sealed class DeessProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.Deess;

    public string Label => "De-esser";

    public string Summary => "a split-band compressor on the sibilance (over `frequency` Hz, or the band around it): its level over the value, dBFS (default -30), divided by `ratio`, the rest of the sound untouched";

    public bool Live => true;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Threshold", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-30", Help = "The sibilance level over which it is turned down" },
        new("mode", "Band", AudioProcessorInput.Choice)
        {
            Choices = ["high", "band"], Optional = true, Default = "high", Help = "high: everything over the frequency; band: the band around it (its width from q)",
        },
        new("frequency", "Frequency", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Step = 100, Unit = "Hz", Optional = true, Default = "5000", Help = "Where the sibilance band starts (high) or its centre (band); under half the rate",
        },
        FilterDesign.Q("0.7071", "The split's resonance (high) or the band's narrowness (band: its centre over its width)"),
        new("ratio", "Ratio", AudioProcessorInput.Number) { Min = 1, Step = 0.5, Optional = true, Default = "4", Help = "How many dB of sibilance over the threshold give one dB out" },
        new("knee", "Knee", AudioProcessorInput.Number) { Min = 0, Step = 1, Unit = "dB", Optional = true, Default = "6", Help = "The width of the soft knee around the threshold (0: a hard knee)" },
        DynamicsParameters.Attack("1", "How fast the sibilance is turned down"),
        DynamicsParameters.Release("50", "How fast it comes back"),
        DynamicsParameters.Window("5", "The time constant of the sibilance level (a one-pole mean square)"),
        DynamicsParameters.Link,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        double frequency = step.Number("frequency", 5000);
        FilterDesign.ThrowIfNotBelowHalf(Name, "frequency", frequency, sampleRate);
        return new Stage(step, frequency, sampleRate, channels);
    }

    // The split's biquad: the RBJ low-pass (high) or band-pass of 0 dB peak gain (band).
    public static Biquad Split(bool high, double hertz, double q, int sampleRate)
    {
        if (high)
        {
            return Biquad.LowPass(hertz, sampleRate, 2 * q);
        }

        double w0 = 2 * Math.PI * hertz / sampleRate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / (2 * q);
        double a0 = 1 + alpha;
        return new Biquad(alpha / a0, 0, -alpha / a0, -2 * cos / a0, (1 - alpha) / a0, 0);
    }

    private sealed class Stage : IAudioStage
    {
        private const int Chunk = 1024;
        private readonly Biquad _split;
        private readonly bool _high, _linked;
        private readonly int _groups;
        private readonly double _threshold, _ratio, _knee, _window, _attack, _release, _kneeStart;
        private readonly double[] _memory;                                      // per channel: x1, x2, y1, y2
        private readonly double[] _band;                                        // per channel: a chunk of the band
        private readonly double[] _level = new double[Chunk];
        private readonly double[] _env, _reduction, _last, _lastGain;
        private bool _started;

        public Stage(AudioProcessorStep step, double frequency, int sampleRate, int channels)
        {
            (SampleRate, Channels) = (sampleRate, channels);
            _high = !string.Equals(step.Option("mode", "high"), "band", StringComparison.OrdinalIgnoreCase);
            _split = Split(_high, frequency, step.Number("q", 0.7071), sampleRate);
            _linked = step.Flag("link", true);
            _groups = _linked ? 1 : channels;
            (_threshold, _ratio, _knee) = (step.Number(-30), step.Number("ratio", 4), step.Number("knee", 6));
            _window = DynamicsMath.Coefficient(step.Number("window", 5), sampleRate);
            _attack = DynamicsMath.Coefficient(step.Number("attack", 1), sampleRate);
            _release = DynamicsMath.Coefficient(step.Number("release", 50), sampleRate);
            _kneeStart = Math.Pow(10, (_threshold - _knee / 2) / 10);
            _memory = new double[4 * channels];
            _band = new double[channels * Chunk];
            (_env, _reduction, _last, _lastGain) = (new double[_groups], new double[_groups], new double[_groups], new double[_groups]);
            Array.Fill(_lastGain, 1.0);
        }

        public int SampleRate { get; }

        public int Channels { get; }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public AudioData Process(AudioData block)
        {
            RepairProcessors.ThrowIfOther(block, SampleRate, Channels, RepairProcessors.Deess);
            int length = block.Length, perGroup = _linked ? Channels : 1;
            if (length == 0)
            {
                return block;
            }

            var samples = block.Samples;
            if (!_started)
            {
                for (int c = 0; c < Channels; c++)
                {
                    double first = samples[c * length];
                    double settled = _split.DcGain == 0 ? 0 : first * _split.DcGain;
                    (_memory[4 * c], _memory[4 * c + 1], _memory[4 * c + 2], _memory[4 * c + 3]) = (first, first, settled, settled);
                }

                _started = true;
            }

            for (int start = 0; start < length; start += Chunk)
            {
                int n = Math.Min(Chunk, length - start);
                for (int c = 0; c < Channels; c++)
                {
                    Band(samples.AsSpan(c * length + start, n), _memory.AsSpan(4 * c, 4), _band.AsSpan(c * Chunk, n));
                }

                var level = _level.AsSpan(0, n);
                for (int g = 0; g < _groups; g++)
                {
                    int first = _linked ? 0 : g;
                    for (int c = first; c < first + perGroup; c++)
                    {
                        var band = _band.AsSpan(c * Chunk, n);
                        for (int i = 0; i < n; i++)
                        {
                            double v = band[i] * band[i];
                            level[i] = c == first ? v : Math.Max(level[i], v);
                        }
                    }

                    Gains(g, level);
                    for (int c = first; c < first + perGroup; c++)
                    {
                        var x = samples.AsSpan(c * length + start, n);
                        var band = _band.AsSpan(c * Chunk, n);
                        for (int i = 0; i < n; i++)
                        {
                            x[i] = (float)(x[i] - (1 - level[i]) * band[i]);
                        }
                    }
                }
            }

            return block;
        }

        public AudioData Flush() => new([], Channels, SampleRate);

        // The band of a chunk (its samples read, not written), the split's memory carried.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Band(ReadOnlySpan<float> x, Span<double> s, Span<double> band)
        {
            double b0 = _split.B0, b1 = _split.B1, b2 = _split.B2, a1 = _split.A1, a2 = _split.A2;
            double x1 = s[0], x2 = s[1], y1 = s[2], y2 = s[3];
            for (int i = 0; i < x.Length; i++)
            {
                double xi = x[i];
                double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                band[i] = _high ? xi - yi : yi;
                (x2, x1, y2, y1) = (x1, xi, y1, yi);
            }

            (s[0], s[1], s[2], s[3]) = (x1, x2, y1, y2);
        }

        // The band's gain of each sample of a chunk, from its level (overwritten with the gains).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            double env = _env[group], reduction = _reduction[group], last = _last[group], lastGain = _lastGain[group];
            for (int i = 0; i < level.Length; i++)
            {
                env = DynamicsMath.Follow(env, level[i], _window, DynamicsMath.SilentPower);
                double wanted = env > _kneeStart ? CompressProcessor.Curve(10 * Math.Log10(env), _threshold, _ratio, _knee) : 0;
                reduction = DynamicsMath.Follow(reduction, wanted, wanted < reduction ? _attack : _release, DynamicsMath.GainSnap);
                if (reduction != last)
                {
                    (last, lastGain) = (reduction, DynamicsMath.Gain(reduction));
                }

                level[i] = lastGain;
            }

            (_env[group], _reduction[group], _last[group], _lastGain[group]) = (env, reduction, last, lastGain);
        }
    }
}

// The linear prediction declick and declip share: the predictor x[n] ≈ Σ a_k x[n - k] (k = 1..p), its forward and
// backward residuals, and the least-squares AR interpolation of a gap (Janssen, Veldhuis and Vries, 1986: the gap's
// samples that minimise the prediction error summed over every row that reaches them, given p samples on each side; the
// normal equations are banded, bandwidth p, and solved by a banded Cholesky). Scratch is made once for a stage.
internal sealed class LinearPrediction
{
    private readonly int _order, _longest;
    private readonly double[] _r, _a, _previous, _b, _rb;
    private readonly double[] _lower;                                           // the banded Cholesky factor: row i, column i - j at i (p + 1) + j
    private readonly double[] _known, _rhs, _solution;
    private readonly double[] _covariance;

    public LinearPrediction(int order, int longest)
    {
        (_order, _longest) = (order, longest);
        (_r, _a, _previous, _b, _rb) = (new double[order + 1], new double[order], new double[order + 1], new double[order + 1], new double[order + 1]);
        _lower = new double[longest * (order + 1)];
        (_known, _rhs, _solution) = (new double[longest + order], new double[longest], new double[longest]);
        _covariance = new double[order * (order + 1)];
    }

    // The predictor's coefficients a_1..a_p (the last fitted).
    public ReadOnlySpan<double> Coefficients => _a;

    // The predictor of `segment` by the autocorrelation method (r[j] = Σ s[i] s[i - j], Levinson-Durbin; zeros when
    // r[0] is 0, and the recursion stopped where its error would not stay above 0).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void FitAutocorrelation(ReadOnlySpan<float> segment)
    {
        int p = _order;
        for (int j = 0; j <= p; j++)
        {
            double sum = 0;
            for (int i = j; i < segment.Length; i++)
            {
                sum += (double)segment[i] * segment[i - j];
            }

            _r[j] = sum;
        }

        Array.Clear(_a);
        if (_r[0] <= 0)
        {
            return;
        }

        // a[0] of the reference is _previous's slot; here a_k is _a[k - 1].
        var a = _previous.AsSpan();
        a.Clear();
        double error = _r[0];
        for (int i = 1; i <= p; i++)
        {
            double acc = _r[i];
            for (int j = 1; j < i; j++)
            {
                acc -= a[j] * _r[i - j];
            }

            double k = acc / error;
            for (int j = 1; j < i; j++)
            {
                _b[j] = a[j];
            }

            a[i] = k;
            for (int j = 1; j < i; j++)
            {
                a[j] = _b[j] - k * _b[i - j];
            }

            error *= 1 - k * k;
            if (error <= 0)
            {
                break;
            }
        }

        a.Slice(1, p).CopyTo(_a);
    }

    // The predictor fitted by least squares (the covariance method) over the rows n in [from, to) of x none of whose
    // samples n - p .. n is `bad`, a ridge of 1e-9 of the mean diagonal (and 1e-30) added; false (nothing fitted) when
    // fewer than 2p rows are clean.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool FitCovariance(ReadOnlySpan<float> x, int from, int to, ReadOnlySpan<bool> bad)
    {
        int p = _order, width = p + 1, rows = 0, run = 0;
        var c = _covariance.AsSpan();                                           // p x p, then the right side p
        c.Clear();
        for (int n = from - p; n < to; n++)
        {
            run = bad[n] ? 0 : run + 1;
            if (n < from || run < width)
            {
                continue;
            }

            rows++;
            double target = x[n];
            for (int j = 0; j < p; j++)
            {
                double xj = x[n - 1 - j];
                c[p * p + j] += xj * target;
                for (int k = 0; k <= j; k++)
                {
                    c[j * p + k] += xj * x[n - 1 - k];
                }
            }
        }

        if (rows < 2 * p)
        {
            return false;
        }

        double trace = 0;
        for (int j = 0; j < p; j++)
        {
            trace += c[j * p + j];
            for (int k = 0; k < j; k++)
            {
                c[k * p + j] = c[j * p + k];
            }
        }

        double ridge = 1e-9 * trace / p + 1e-30;
        for (int j = 0; j < p; j++)
        {
            c[j * p + j] += ridge;
        }

        // Cholesky in place (lower), then the two triangular solves into _a.
        for (int j = 0; j < p; j++)
        {
            double d = c[j * p + j];
            for (int k = 0; k < j; k++)
            {
                d -= c[j * p + k] * c[j * p + k];
            }

            if (!(d > 0))
            {
                return false;
            }

            d = Math.Sqrt(d);
            c[j * p + j] = d;
            for (int i = j + 1; i < p; i++)
            {
                double s = c[i * p + j];
                for (int k = 0; k < j; k++)
                {
                    s -= c[i * p + k] * c[j * p + k];
                }

                c[i * p + j] = s / d;
            }
        }

        for (int i = 0; i < p; i++)
        {
            double s = c[p * p + i];
            for (int k = 0; k < i; k++)
            {
                s -= c[i * p + k] * _a[k];
            }

            _a[i] = s / c[i * p + i];
        }

        for (int i = p - 1; i >= 0; i--)
        {
            double s = _a[i];
            for (int k = i + 1; k < p; k++)
            {
                s -= c[k * p + i] * _a[k];
            }

            _a[i] = s / c[i * p + i];
        }

        return true;
    }

    // x[i] - Σ a_k x[i - k].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Forward(ReadOnlySpan<float> x, int i)
    {
        double acc = x[i];
        for (int k = 1; k <= _order; k++)
        {
            acc -= _a[k - 1] * x[i - k];
        }

        return acc;
    }

    // x[i] - Σ a_k x[i + k].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Backward(ReadOnlySpan<float> x, int i)
    {
        double acc = x[i];
        for (int k = 1; k <= _order; k++)
        {
            acc -= _a[k - 1] * x[i + k];
        }

        return acc;
    }

    // The samples s..e of x (e - s < longest; p samples on each side) as the least-squares AR interpolation gives them,
    // into Interpolated (not written to x).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public ReadOnlySpan<double> Interpolate(ReadOnlySpan<float> x, int s, int e)
    {
        int p = _order, m = e - s + 1, width = p + 1;
        _b[0] = 1;
        for (int k = 1; k <= p; k++)
        {
            _b[k] = -_a[k - 1];
        }

        for (int j = 0; j <= p; j++)
        {
            double sum = 0;
            for (int k = 0; k + j <= p; k++)
            {
                sum += _b[k] * _b[k + j];
            }

            _rb[j] = sum;
        }

        for (int t = s; t <= e + p; t++)
        {
            double acc = 0;
            for (int k = 0; k <= p; k++)
            {
                int at = t - k;
                if (at < s || at > e)
                {
                    acc += _b[k] * x[at];
                }
            }

            _known[t - s] = acc;
        }

        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            for (int k = 0; k <= p; k++)
            {
                sum += _b[k] * _known[i + k];
            }

            _rhs[i] = -sum;
        }

        // The banded Cholesky of M[i][j] = rb[|i - j|] (|i - j| <= p): L[i][j] at _lower[i (p + 1) + (i - j)].
        for (int i = 0; i < m; i++)
        {
            int low = Math.Max(0, i - p);
            for (int j = low; j <= i; j++)
            {
                double sum = _rb[i - j];
                for (int k = low; k < j; k++)
                {
                    sum -= _lower[i * width + (i - k)] * _lower[j * width + (j - k)];
                }

                _lower[i * width + (i - j)] = i == j ? Math.Sqrt(sum) : sum / _lower[j * width];
            }
        }

        for (int i = 0; i < m; i++)
        {
            double sum = _rhs[i];
            for (int k = Math.Max(0, i - p); k < i; k++)
            {
                sum -= _lower[i * width + (i - k)] * _solution[k];
            }

            _solution[i] = sum / _lower[i * width];
        }

        for (int i = m - 1; i >= 0; i--)
        {
            double sum = _solution[i];
            for (int k = i + 1; k <= Math.Min(m - 1, i + p); k++)
            {
                sum -= _lower[k * width + (k - i)] * _solution[k];
            }

            _solution[i] = sum / _lower[i * width];
        }

        return _solution.AsSpan(0, m);
    }

    public int Longest => _longest;
}

// The frame walk declick and declip share, each channel on its own: frames of F samples; frame t (samples tF to
// (t + 1)F) is repaired once the p samples before it and the A of its look-ahead after it are in, in a buffer of
// p + F + A samples (zeros before the start) that then moves on by F; the repaired frame comes out one sample per sample
// taken, so the output lags by F + A - 1 (the stage's Latency, silence first) and Flush runs that many zeros through
// for the tail (the repair told where the recording ends, so nothing starts in the zeros). Marks travel with the
// buffer (declip's "rebuilt").
internal sealed class RepairWalk
{
    private readonly int _order, _frame, _look, _width, _channels;
    private readonly float[] _buffer;                                          // per channel: p + F + A samples
    private readonly bool[] _marks;                                             // per channel: one a sample, moving with it
    private readonly float[] _done;                                             // per channel: the last frame repaired
    private readonly RepairOf _repair;
    private readonly double _gain;
    private int _filled;                                                        // samples in each channel's buffer
    private long _frames;                                                       // frames repaired per channel
    private long _taken;
    private long _end = long.MaxValue;

    public RepairWalk(int order, int frame, int look, int sampleRate, int channels, RepairOf repair, double gain = 1)
    {
        (_order, _frame, _look, _channels, _repair, _gain, SampleRate) = (order, frame, look, channels, repair, gain, sampleRate);
        _width = order + frame + look;
        _buffer = new float[channels * _width];
        _marks = new bool[channels * _width];
        _done = new float[channels * frame];
        _filled = order;
    }

    // Repair frame `frame` of channel `channel` in place: the buffer holds the p samples before it, the frame at p and
    // the look-ahead after it; `origin` is the recording's index of buffer[0]; samples from `end` on are the flush's.
    public delegate void RepairOf(int channel, Span<float> buffer, Span<bool> marks, long origin, int end);

    public int SampleRate { get; }

    public int Latency => _frame + _look - 1;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public AudioData Process(AudioData block, string name)
    {
        RepairProcessors.ThrowIfOther(block, SampleRate, _channels, name);
        int length = block.Length, latency = Latency;
        var samples = block.Samples;
        int filled = _filled;
        long frames = _frames;
        for (int c = 0; c < _channels; c++)
        {
            var x = samples.AsSpan(c * length, length);
            var buffer = _buffer.AsSpan(c * _width, _width);
            var marks = _marks.AsSpan(c * _width, _width);
            var done = _done.AsSpan(c * _frame, _frame);
            filled = _filled;
            frames = _frames;
            long t = _taken;
            for (int i = 0; i < x.Length; i++, t++)
            {
                buffer[filled++] = x[i];
                if (filled == _width)
                {
                    long origin = frames * _frame - _order;
                    long end = _end == long.MaxValue ? _width : Math.Min(_width, _end - origin);   // origin is -p at first
                    _repair(c, buffer, marks, origin, (int)end);
                    buffer.Slice(_order, _frame).CopyTo(done);
                    buffer.Slice(_frame).CopyTo(buffer);
                    marks.Slice(_frame).CopyTo(marks);
                    marks.Slice(_width - _frame).Clear();
                    filled -= _frame;
                    frames++;
                }

                if (t < latency)
                {
                    x[i] = 0f;
                }
                else
                {
                    float v = done[(int)((t - latency) % _frame)];
                    x[i] = _gain == 1 ? v : (float)(v * _gain);
                }
            }
        }

        (_filled, _frames) = (filled, frames);
        _taken += length;
        return block;
    }

    public AudioData Flush(string name)
    {
        _end = Math.Min(_end, _taken);
        return Process(new AudioData(new float[_channels * Latency], _channels, SampleRate), name);
    }
}

// "declick" (live, with look-ahead): impulsive clicks found and filled. Per frame of `frame` ms (RepairWalk), the
// order-p predictor of the frame and the p samples before it (the autocorrelation method); sigma, the forward residual's
// median magnitude over the frame / 0.6745 (at least 1e-7: digital silence is not all clicks); a sample is a click's when
// both its forward and its backward residual pass value · sigma (the forward one rings p samples after a click, the
// backward one p before it: both are large only on the click). A click starts at such a sample in the frame (not in the
// recording's first p samples, whose predictor has no past) and runs to the last such sample with no gap over p; it then
// grows back while the forward residual stays over (its first samples) and on while the backward one does (its last); one
// longer than `length` ms is a transient and is left. Its samples become the least-squares AR interpolation
// (LinearPrediction), and the forward residuals of the p samples after it are worked out again. The look-ahead is
// round(length · fs) + 2p samples.
internal sealed class DeclickProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.Declick;

    public string Label => "Declick";

    public string Summary => "clicks found where the linear predictor's residual passes the value (default 10) times its spread, up to `length` ms long, and filled from the sound around them (AR interpolation of `order`)";

    public bool Live => true;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Threshold", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1000, Step = 1, Unit = "sigma", Optional = true, Default = "10",
            Help = "How far over the residual's usual spread a sample must be to count as a click (lower finds more, and may take sharp sounds for clicks)",
        },
        RepairProcessors.Length("2", "The longest click filled; a longer burst is left as it is (it is a sound, not a click)"),
        RepairProcessors.Order("16"),
        RepairProcessors.Frame("20"),
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return new Stage(step, sampleRate, channels);
    }

    private sealed class Stage : IAudioStage
    {
        private readonly RepairWalk _walk;
        private readonly LinearPrediction _prediction;
        private readonly int _order, _frame;
        private readonly double _threshold;
        private readonly double[] _forward, _backward, _spread;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _order = step.Integer("order", 16);
            int longest = Math.Max(1, (int)Math.Round(step.Number("length", 2) * sampleRate / 1000.0));
            _frame = Math.Max(1, (int)Math.Round(step.Number("frame", 20) * sampleRate / 1000.0));
            _threshold = step.Number(10);
            _prediction = new LinearPrediction(_order, longest);
            _walk = new RepairWalk(_order, _frame, longest + 2 * _order, sampleRate, channels, Repair);
            int width = _order + _frame + longest + 2 * _order;
            (_forward, _backward, _spread) = (new double[width], new double[width], new double[_frame]);
            Channels = channels;
        }

        public int SampleRate => _walk.SampleRate;

        public int Channels { get; }

        public int Latency => _walk.Latency;

        public AudioData Process(AudioData block) => _walk.Process(block, RepairProcessors.Declick);

        public AudioData Flush() => _walk.Flush(RepairProcessors.Declick);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Repair(int channel, Span<float> x, Span<bool> marks, long origin, int end)
        {
            _ = channel;
            _ = marks;
            int p = _order, start = p, f = _frame, width = x.Length, longest = _prediction.Longest;
            _prediction.FitAutocorrelation(x[..(p + f)]);
            for (int i = start; i < width - p; i++)
            {
                _forward[i] = _prediction.Forward(x, i);
                _backward[i] = _prediction.Backward(x, i);
            }

            for (int i = 0; i < f; i++)
            {
                _spread[i] = Math.Abs(_forward[start + i]);
            }

            var spread = _spread.AsSpan(0, f);
            spread.Sort();
            double median = f % 2 == 1 ? spread[f / 2] : (spread[f / 2 - 1] + spread[f / 2]) / 2;
            double limit = _threshold * Math.Max(median / 0.6745, 1e-7);
            int low = (int)Math.Max(start, 2 * p - (origin + p));                  // the recording's index p on (a full past for the predictor)
            int stop = Math.Min(start + f, end), bound = Math.Min(width - p, end);
            int at = low;
            while (at < stop)
            {
                if (!(Math.Abs(_forward[at]) > limit && Math.Abs(_backward[at]) > limit))
                {
                    at++;
                    continue;
                }

                int s = at, last = at;
                for (int k = at + 1; k < bound && k - last <= p && last - s + 1 <= longest; k++)
                {
                    if (Math.Abs(_forward[k]) > limit && Math.Abs(_backward[k]) > limit)
                    {
                        last = k;
                    }
                }

                while (s > low && Math.Abs(_forward[s - 1]) > limit && last - s + 1 <= longest)
                {
                    s--;
                }

                while (last + 1 < bound && Math.Abs(_backward[last + 1]) > limit && last - s + 1 <= longest)
                {
                    last++;
                }

                if (last - s + 1 <= longest)
                {
                    var filled = _prediction.Interpolate(x, s, last);
                    for (int j = 0; j < filled.Length; j++)
                    {
                        x[s + j] = (float)filled[j];
                    }

                    for (int j = last + 1; j < Math.Min(last + p + 1, width - p); j++)
                    {
                        _forward[j] = _prediction.Forward(x, j);
                    }
                }

                at = low = last + 1;
            }
        }
    }
}

// "declip" (live, with look-ahead): clipped runs rebuilt. A run is at least `run` samples in a row of one sign at or
// over the clip level less `near` dB and not rebuilt already; the clip level is `level` dBFS or, by default (auto), the
// largest magnitude come in so far (to the end of the look-ahead, the recording's own samples). A run longer than
// `length` ms is left. Per frame of `frame` ms (RepairWalk; look-ahead round(length · fs) + p), the predictor of order
// p is fitted by least squares over the frame and its look-ahead, on the rows none of whose p + 1 samples is clipped
// (LinearPrediction.FitCovariance; the frame's autocorrelation predictor when fewer than 2p rows are clean); a run's
// samples become the least-squares AR interpolation, those that come out under the clip level set to it (a clipped
// sample was at least that), and are marked rebuilt. Every sample out is then times 10^(-headroom / 20) (room for the
// rebuilt peaks; 0 dB leaves the level).
internal sealed class DeclipProcessor : IAudioProcessor
{
    public string Name => RepairProcessors.Declip;

    public string Label => "Declip";

    public string Summary => "clipped runs (at the clip `level`, auto: the peak so far) rebuilt from the sound around them (AR interpolation of `order`, kept over the clip level), then `headroom` dB taken off to make room";

    public bool Live => true;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("level", "Clip level", AudioProcessorInput.Number)
        {
            Max = 0, Step = 0.5, Unit = "dBFS", Optional = true, Default = "auto", Help = "The level the sound was clipped at, or auto (the largest sample come in so far)",
        },
        new("near", "Near the level", AudioProcessorInput.Number)
        {
            Min = 0, Max = 6, Step = 0.01, Unit = "dB", Optional = true, Default = "0.05", Help = "How close under the clip level a sample counts as clipped",
        },
        RepairProcessors.Length("5", "The longest run rebuilt; a longer one is left as it is"),
        new("run", "Shortest run", AudioProcessorInput.Integer)
        {
            Min = 1, Max = 1000, Step = 1, Optional = true, Default = "2", Help = "How many samples in a row at the clip level count as clipping (1: a single sample at the peak too)",
        },
        RepairProcessors.Order("12"),
        RepairProcessors.Frame("20"),
        new("headroom", "Headroom", AudioProcessorInput.Number)
        {
            Min = 0, Max = 24, Step = 1, Unit = "dB", Optional = true, Default = "0", Help = "How far the whole sound is turned down after, to make room for the rebuilt peaks",
        },
    ];

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        (Auto(step) ? new AudioProcessorStep(step.Name, step.Value, step.Options.Where(o => o.Key != "level")) : step).Validate(Parameters);
    }

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        Check(step);
        return new Stage(step, sampleRate, channels);
    }

    private static bool Auto(AudioProcessorStep step) => string.Equals(step.Option("level", "auto"), "auto", StringComparison.OrdinalIgnoreCase);

    private sealed class Stage : IAudioStage
    {
        private readonly RepairWalk _walk;
        private readonly LinearPrediction _prediction;
        private readonly int _order, _frame, _run;
        private readonly double _fixed, _near;
        private readonly double[] _peak;                                        // per channel: the largest magnitude come in
        private readonly long[] _seen;                                          // per channel: the samples it was read over
        private readonly bool[] _bad;                                           // the clipped samples of the buffer, for the fit

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _order = step.Integer("order", 12);
            int longest = Math.Max(1, (int)Math.Round(step.Number("length", 5) * sampleRate / 1000.0));
            _frame = Math.Max(1, (int)Math.Round(step.Number("frame", 20) * sampleRate / 1000.0));
            _run = step.Integer("run", 2);
            _fixed = Auto(step) ? 0 : DynamicsMath.Gain(step.Number("level", 0));
            _near = DynamicsMath.Gain(-step.Number("near", 0.05));
            _prediction = new LinearPrediction(_order, longest);
            _walk = new RepairWalk(_order, _frame, longest + _order, sampleRate, channels, Repair, DynamicsMath.Gain(-step.Number("headroom", 0)));
            (_peak, _seen) = (new double[channels], new long[channels]);
            _bad = new bool[_order + _frame + longest + _order];
            Channels = channels;
        }

        public int SampleRate => _walk.SampleRate;

        public int Channels { get; }

        public int Latency => _walk.Latency;

        public AudioData Process(AudioData block) => _walk.Process(block, RepairProcessors.Declip);

        public AudioData Flush() => _walk.Flush(RepairProcessors.Declip);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Repair(int channel, Span<float> x, Span<bool> marks, long origin, int end)
        {
            int p = _order, start = p, f = _frame, longest = _prediction.Longest;
            int top = end;
            int from = (int)Math.Max(p, _seen[channel] - origin);
            if (top > from)
            {
                _peak[channel] = Math.Max(_peak[channel], DynamicsMath.SamplePeak(x[from..top]));
                _seen[channel] = origin + top;
            }

            double clip = _fixed > 0 ? _fixed : _peak[channel];
            if (clip <= 0)
            {
                return;
            }

            double bar = clip * _near;
            bool fitted = false;
            int stop = Math.Min(start + f, end);
            int i = start;
            while (i < stop)
            {
                int sign = x[i] > 0 ? 1 : x[i] < 0 ? -1 : 0;
                if (!Clipped(x, marks, i, 0, bar) || Clipped(x, marks, i - 1, sign, bar))
                {
                    i++;
                    continue;
                }

                int s = i, e = i;
                while (e + 1 < top && Clipped(x, marks, e + 1, sign, bar) && e + 1 - s < longest)
                {
                    e++;
                }

                if (e - s + 1 >= _run && !(e + 1 < top && Clipped(x, marks, e + 1, sign, bar)))
                {
                    if (!fitted)
                    {
                        Fit(x, marks, start, f, top, bar);
                        fitted = true;
                    }

                    var rebuilt = _prediction.Interpolate(x, s, e);
                    for (int j = 0; j < rebuilt.Length; j++)
                    {
                        double y = rebuilt[j];
                        x[s + j] = (float)(sign * y < clip ? sign * clip : y);
                        marks[s + j] = true;
                    }
                }

                while (e + 1 < top && Clipped(x, marks, e + 1, sign, bar))
                {
                    e++;
                }

                i = e + 1;
            }
        }

        // The predictor over the clean rows of [start, top), else the frame's autocorrelation predictor.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Fit(Span<float> x, Span<bool> marks, int start, int f, int top, double bar)
        {
            for (int j = start - _order; j < top; j++)
            {
                _bad[j] = Clipped(x, marks, j, 0, bar);
            }

            if (!_prediction.FitCovariance(x, start, top, _bad))
            {
                _prediction.FitAutocorrelation(x[(start - _order)..(start + f)]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Clipped(ReadOnlySpan<float> x, ReadOnlySpan<bool> marks, int i, int sign, double bar) =>
            !marks[i] && Math.Abs((double)x[i]) >= bar && (sign == 0 || (x[i] > 0 ? 1 : -1) == sign);
    }
}
