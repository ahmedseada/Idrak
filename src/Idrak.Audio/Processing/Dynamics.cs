// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Idrak.Audio;

// The dynamics processors (plan 31, P2): the level of the sound shaped as it comes (gate, compress, limit, agc,
// loudness-live) or over the whole recording (loudness, peak). Each is checked against tools/audio/make_audio_dynamics.py,
// which writes the same equations sample by sample (and BS.1770's integrated loudness also from pyloudnorm).
// No option key is a processor's name (a gate's threshold is not "gate", peak normalization is not "peak"): an item
// after a comma that names an option of the step before it joins that step, and a registered name would be read as a
// step of its own instead.
internal static class DynamicsProcessors
{
    public const string Gate = "gate", Compress = "compress", Limit = "limit", Agc = "agc", Loudness = "loudness", LoudnessLive = "loudness-live", Peak = "peaknorm";

    public static void RegisterDefaults(KindTable<IAudioProcessor> table)
    {
        table.RegisterDefault(new GateProcessor());
        table.RegisterDefault(new CompressProcessor());
        table.RegisterDefault(new LimitProcessor());
        table.RegisterDefault(new AgcProcessor());
        table.RegisterDefault(new LoudnessProcessor());
        table.RegisterDefault(new LiveLoudnessProcessor());
        table.RegisterDefault(new PeakProcessor());
    }
}

// The arithmetic the dynamics stages share. A time constant of t ms at fs is the one-pole coefficient exp(-1 / (t fs))
// (0: at once); a follower steps value to target + (value - target) a, and takes the target itself once that step is
// under its snap (a mean square under 1e-30, -300 dB, is silence; a gain within 1e-12 of its target is the target), so
// an envelope settles instead of running into denormal numbers.
internal static class DynamicsMath
{
    public const double SilentPower = 1e-30, GainSnap = 1e-12;

    public static double Coefficient(double milliseconds, int sampleRate) => milliseconds <= 0 ? 0 : Math.Exp(-1000.0 / (milliseconds * sampleRate));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Follow(double value, double target, double coefficient, double snap)
    {
        double step = (value - target) * coefficient;
        return Math.Abs(step) >= snap ? target + step : target;
    }

    public static double Gain(double decibels) => Math.Pow(10, decibels / 20);

    // round(ms · fs / 1000) samples, half to even (as the reference's round).
    public static int Samples(double milliseconds, int sampleRate) => (int)Math.Round(milliseconds * sampleRate / 1000.0);

    // The largest magnitude of the samples.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float SamplePeak(ReadOnlySpan<float> samples)
    {
        int i = 0, width = Vector<float>.Count;
        var best = Vector<float>.Zero;
        for (; i + width <= samples.Length; i += width)
        {
            best = Vector.Max(best, Vector.Abs(new Vector<float>(samples.Slice(i, width))));
        }

        float peak = 0;
        for (int k = 0; k < width && samples.Length >= width; k++)
        {
            peak = Math.Max(peak, best[k]);
        }

        for (; i < samples.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(samples[i]));
        }

        return peak;
    }

    // The true peak: the largest magnitude of the samples and of the sound between them, read at oversample - 1 points
    // between each two samples (and after the last) by a Hann-windowed sinc reaching `taps` samples each side. The value
    // at n + p/L is the sum over m in (-taps, taps] of x[n + m] · sinc(p/L - m) · (0.5 + 0.5 cos(π (p/L - m) / taps)),
    // zeros past the ends, summed in float in the order of m (each phase a run of vector multiply-adds over a padded copy).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double TruePeak(AudioData audio, int oversample, int taps)
    {
        var samples = audio.Samples;
        double best = SamplePeak(samples);
        int length = audio.Length;
        if (oversample < 2 || length == 0)
        {
            return best;
        }

        int width = 2 * taps;
        var filter = new float[(oversample - 1) * width];
        for (int p = 1; p < oversample; p++)
        {
            double fraction = (double)p / oversample;
            for (int k = 0; k < width; k++)
            {
                double t = fraction - (k - taps + 1), y = Math.PI * t;
                filter[(p - 1) * width + k] = (float)(Math.Sin(y) / y * (0.5 + 0.5 * Math.Cos(Math.PI * t / taps)));
            }
        }

        const int Chunk = 4096;
        var padded = new float[length + width];
        var sum = new float[Math.Min(Chunk, length)];
        float peak = 0;
        for (int c = 0; c < audio.Channels; c++)
        {
            samples.AsSpan(c * length, length).CopyTo(padded.AsSpan(taps));
            for (int p = 0; p < oversample - 1; p++)
            {
                var h = filter.AsSpan(p * width, width);
                for (int start = 0; start < length; start += Chunk)
                {
                    int n = Math.Min(Chunk, length - start);
                    var acc = sum.AsSpan(0, n);
                    acc.Clear();
                    for (int k = 0; k < width; k++)
                    {
                        MultiplyAdd(padded.AsSpan(start + k + 1, n), h[k], acc);       // x[n + m] for m = k - taps + 1
                    }

                    peak = Math.Max(peak, SamplePeak(acc));
                }
            }
        }

        return Math.Max(best, peak);
    }

    // acc += h · x, a multiply then an add (not fused), so the vector and the scalar tail round alike.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyAdd(ReadOnlySpan<float> x, float h, Span<float> acc)
    {
        int i = 0, width = Vector<float>.Count;
        var g = new Vector<float>(h);
        for (; i + width <= x.Length; i += width)
        {
            (new Vector<float>(acc.Slice(i, width)) + new Vector<float>(x.Slice(i, width)) * g).CopyTo(acc.Slice(i, width));
        }

        for (; i < x.Length; i++)
        {
            acc[i] += h * x[i];
        }
    }
}

// The options several dynamics processors share.
internal static class DynamicsParameters
{
    public static AudioProcessorParameter Attack(string value, string help) => new("attack", "Attack", AudioProcessorInput.Number)
    {
        Min = 0, Step = 1, Unit = "ms", Optional = true, Default = value, Help = help,
    };

    public static AudioProcessorParameter Release(string value, string help) => new("release", "Release", AudioProcessorInput.Number)
    {
        Min = 0, Step = 10, Unit = "ms", Optional = true, Default = value, Help = help,
    };

    public static AudioProcessorParameter Window(string value, string help) => new("window", "Level window", AudioProcessorInput.Number)
    {
        Min = 0, Step = 1, Unit = "ms", Optional = true, Default = value, Help = help,
    };

    public static readonly AudioProcessorParameter Link = new("link", "Link channels", AudioProcessorInput.Switch)
    {
        Optional = true, Default = "true", Help = "One gain for all the channels, from all of them (false: each channel on its own level)",
    };

    public static readonly AudioProcessorParameter MaxGain = new("max", "Most gain", AudioProcessorInput.Number)
    {
        Min = 0, Step = 1, Unit = "dB", Optional = true, Default = "20", Help = "The most the sound is turned up",
    };

    public static readonly AudioProcessorParameter MinGain = new("min", "Least gain", AudioProcessorInput.Number)
    {
        Max = 0, Step = 1, Unit = "dB", Optional = true, Default = "-20", Help = "The most the sound is turned down (a negative gain)",
    };

    public static readonly AudioProcessorParameter Oversample = new("oversample", "True-peak oversampling", AudioProcessorInput.Integer)
    {
        Min = 1, Max = 64, Step = 1, Optional = true, Default = "4", Help = "The points read per sample for the true peak (1: the samples only)",
    };

    public static readonly AudioProcessorParameter Taps = new("taps", "True-peak taps", AudioProcessorInput.Integer)
    {
        Min = 1, Max = 64, Step = 1, Optional = true, Default = "6", Help = "How many samples on each side the true-peak interpolator reaches",
    };

    public static readonly AudioProcessorParameter Weighting = new("weighting", "K-weighting", AudioProcessorInput.Choice)
    {
        Choices = ["bs1770", "rbj"], Optional = true, Default = "bs1770",
        Help = "bs1770: the standard's filters at any rate (De Man's design); rbj: the RBJ cookbook's shelf and high-pass (pyloudnorm's default)",
    };
}

// What a dynamics stage's detector reads from each sample over a group of channels.
internal enum DynamicsDetector
{
    // The largest magnitude.
    Peak,

    // The largest square.
    Square,

    // The mean of the squares.
    MeanSquare,
}

// The walk the live dynamics stages share: a block in chunks of up to Chunk samples; per chunk and group of channels
// (all of them when linked, each alone when not), the detector's input walked channel by channel in memory order into
// one array, the gains worked out from it sample by sample in the group's state (in place), then each channel of the
// group multiplied by them in place. The state lives in the stage, so blocks of any size give the same samples. Each
// stage owns one and hands it its gains (and, for a weighted detector, its detector input).
internal sealed class DynamicsWalk
{
    private const int Chunk = 1024;
    private readonly double[] _level = new double[Chunk];
    private readonly bool _linked;
    private readonly DynamicsDetector _detector;
    private readonly GainsOf _gains;
    private readonly DetectOf _detect;

    public DynamicsWalk(int sampleRate, int channels, bool linked, DynamicsDetector detector, GainsOf gains, DetectOf? detect = null)
    {
        (SampleRate, Channels, _linked, _detector, _gains) = (sampleRate, channels, linked, detector, gains);
        _detect = detect ?? Detect;
        Groups = linked ? 1 : channels;
    }

    // The gain of each sample of a chunk, from the detector's input in `level` (overwritten with the gains).
    public delegate void GainsOf(int group, Span<double> level);

    // The detector's input of channel `channel` over a chunk: written when `first`, else combined with what is there.
    public delegate void DetectOf(int channel, ReadOnlySpan<float> x, Span<double> level, bool first);

    public int SampleRate { get; }

    public int Channels { get; }

    public int Groups { get; }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public AudioData Process(AudioData block)
    {
        ArgumentNullException.ThrowIfNull(block);
        if (block.SampleRate != SampleRate || block.Channels != Channels)
        {
            throw new ArgumentException($"The stage was opened for {Channels} channel(s) at {SampleRate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
        }

        int length = block.Length;
        var samples = block.Samples;
        int perGroup = _linked ? Channels : 1;
        for (int start = 0; start < length; start += Chunk)
        {
            int n = Math.Min(Chunk, length - start);
            var level = _level.AsSpan(0, n);
            for (int g = 0; g < Groups; g++)
            {
                int first = _linked ? 0 : g;
                for (int c = first; c < first + perGroup; c++)
                {
                    _detect(c, samples.AsSpan(c * length + start, n), level, c == first);
                }

                if (_detector == DynamicsDetector.MeanSquare && perGroup > 1)
                {
                    foreach (ref double v in level)
                    {
                        v /= perGroup;
                    }
                }

                _gains(g, level);
                for (int c = first; c < first + perGroup; c++)
                {
                    var x = samples.AsSpan(c * length + start, n);
                    for (int i = 0; i < x.Length; i++)
                    {
                        x[i] = (float)(x[i] * level[i]);
                    }
                }
            }
        }

        return block;
    }

    public AudioData Flush() => new([], Channels, SampleRate);

    // The detector's input by default: each sample's square or magnitude, summed or the largest over the group.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Detect(int channel, ReadOnlySpan<float> x, Span<double> level, bool first)
    {
        for (int i = 0; i < x.Length; i++)
        {
            double v = x[i];
            double d = _detector == DynamicsDetector.Peak ? Math.Abs(v) : v * v;
            level[i] = first ? d : _detector == DynamicsDetector.MeanSquare ? level[i] + d : Math.Max(level[i], d);
        }
    }

}

// "gate" (live): a noise gate. The level is a one-pole mean square (time constant `window`) of the largest square over
// the linked channels, read per sample; the gate opens when the level reaches the value (dBFS), stays open while the
// level is at or above the value less `hysteresis`, and once the level falls under that, after `hold`, closes. The gain
// moves to 1 (open) with `attack` and to `range` dB down (closed) with `release`. It starts closed.
internal sealed class GateProcessor : ILiveAudioProcessor
{
    public string Name => DynamicsProcessors.Gate;

    public string Label => "Noise gate";

    public string Summary => "a noise gate: the sound under the value, dBFS (default -50), is turned down by `range` dB, opening with `attack`, held open `hold` ms, closing with `release`";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Threshold", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-50", Help = "The level at which the gate opens" },
        DynamicsParameters.Attack("1", "How fast the gate opens (the gain's time constant)"),
        new("hold", "Hold", AudioProcessorInput.Number) { Min = 0, Step = 10, Unit = "ms", Optional = true, Default = "50", Help = "How long the gate stays open once the level falls under the closing threshold" },
        DynamicsParameters.Release("100", "How fast the gate closes (the gain's time constant)"),
        new("range", "Range", AudioProcessorInput.Number) { Min = 0, Step = 5, Unit = "dB", Optional = true, Default = "60", Help = "How far down a closed gate turns the sound" },
        new("hysteresis", "Hysteresis", AudioProcessorInput.Number)
        {
            Min = 0, Step = 1, Unit = "dB", Optional = true, Default = "3", Help = "How far under the threshold the level must fall before the gate closes (0: at the threshold)",
        },
        DynamicsParameters.Window("10", "The time constant of the level (a one-pole mean square)"),
        DynamicsParameters.Link,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step, sampleRate, channels);

    private sealed class Stage : IAudioStage
    {
        private readonly DynamicsWalk _walk;

        public int SampleRate => _walk.SampleRate;

        public int Channels => _walk.Channels;

        private int Groups => _walk.Groups;

        public AudioData Process(AudioData block) => _walk.Process(block);

        public AudioData Flush() => _walk.Flush();

        private readonly double _window, _attack, _release, _open, _close, _floor;
        private readonly int _hold;
        private readonly double[] _env, _gain;
        private readonly int[] _left;
        private readonly bool[] _opened;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _walk = new DynamicsWalk(sampleRate, channels, step.Flag("link", true), DynamicsDetector.Square, Gains);
            double threshold = step.Number(-50);
            _window = DynamicsMath.Coefficient(step.Number("window", 10), sampleRate);
            _attack = DynamicsMath.Coefficient(step.Number("attack", 1), sampleRate);
            _release = DynamicsMath.Coefficient(step.Number("release", 100), sampleRate);
            _open = Math.Pow(10, threshold / 10);
            _close = Math.Pow(10, (threshold - step.Number("hysteresis", 3)) / 10);
            _floor = DynamicsMath.Gain(-step.Number("range", 60));
            _hold = DynamicsMath.Samples(step.Number("hold", 50), sampleRate);
            (_env, _gain, _left, _opened) = (new double[Groups], new double[Groups], new int[Groups], new bool[Groups]);
            Array.Fill(_gain, _floor);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            double env = _env[group], gain = _gain[group];
            int left = _left[group];
            bool opened = _opened[group];
            for (int i = 0; i < level.Length; i++)
            {
                env = DynamicsMath.Follow(env, level[i], _window, DynamicsMath.SilentPower);
                if (env >= _open)
                {
                    (opened, left) = (true, _hold);
                }
                else if (opened)
                {
                    if (env >= _close)
                    {
                        left = _hold;
                    }
                    else if (left > 0)
                    {
                        left--;
                    }
                    else
                    {
                        opened = false;
                    }
                }

                double target = opened ? 1 : _floor;
                gain = DynamicsMath.Follow(gain, target, target > gain ? _attack : _release, DynamicsMath.GainSnap);
                level[i] = gain;
            }

            (_env[group], _gain[group], _left[group], _opened[group]) = (env, gain, left, opened);
        }
    }
}

// "compress" (live): a feed-forward compressor. The level is a one-pole mean square (`window`) of the largest square over
// the linked channels (detector=rms) or their largest magnitude (detector=peak), in dB; the static curve gives no gain
// under the knee, a quadratic through the knee (`knee` dB wide around the value) and (level - value)(1/ratio - 1) above
// it (Giannoulis, Massberg and Reiss, 2012); that gain in dB is smoothed with `attack` (more reduction) or `release`
// (less), then `makeup` dB is added (auto: the curve's reduction at 0 dBFS, so a full-scale level stays full scale).
internal sealed class CompressProcessor : ILiveAudioProcessor
{
    public string Name => DynamicsProcessors.Compress;

    public string Label => "Compressor";

    public string Summary => "a feed-forward compressor: the level over the value, dBFS (default -20), divided by `ratio` (soft knee `knee` dB, `attack`/`release` ms, `makeup` dB or auto)";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Threshold", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-20", Help = "The level over which the sound is compressed" },
        new("ratio", "Ratio", AudioProcessorInput.Number) { Min = 1, Step = 0.5, Optional = true, Default = "3", Help = "How many dB of level over the threshold give one dB out" },
        new("knee", "Knee", AudioProcessorInput.Number) { Min = 0, Step = 1, Unit = "dB", Optional = true, Default = "6", Help = "The width of the soft knee around the threshold (0: a hard knee)" },
        DynamicsParameters.Attack("10", "How fast the gain comes down when the level rises"),
        DynamicsParameters.Release("100", "How fast the gain comes back when the level falls"),
        new("makeup", "Makeup", AudioProcessorInput.Number)
        {
            Step = 1, Unit = "dB", Optional = true, Default = "0", Help = "The gain added after the compression, or auto (the reduction the curve gives a full-scale level)",
        },
        new("detector", "Detector", AudioProcessorInput.Choice)
        {
            Choices = ["rms", "peak"], Optional = true, Default = "rms", Help = "rms: the mean square over `window`; peak: each sample's magnitude",
        },
        DynamicsParameters.Window("10", "The time constant of the rms detector's mean square"),
        DynamicsParameters.Link,
    ];

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        (AutoMakeup(step) ? new AudioProcessorStep(step.Name, step.Value, step.Options.Where(o => o.Key != "makeup")) : step).Validate(Parameters);
    }

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step, sampleRate, channels);

    // The static curve's gain in dB (at most 0) for a level in dB.
    public static double Curve(double level, double threshold, double ratio, double knee)
    {
        double over = level - threshold;
        if (2 * over <= -knee)
        {
            return 0;
        }

        if (knee > 0 && 2 * Math.Abs(over) <= knee)
        {
            double t = over + knee / 2;
            return (1.0 / ratio - 1.0) * t * t / (2 * knee);
        }

        return over * (1.0 / ratio - 1.0);
    }

    private static bool AutoMakeup(AudioProcessorStep step) => string.Equals(step.Option("makeup"), "auto", StringComparison.OrdinalIgnoreCase);

    private sealed class Stage : IAudioStage
    {
        private readonly DynamicsWalk _walk;

        public int SampleRate => _walk.SampleRate;

        public int Channels => _walk.Channels;

        private int Groups => _walk.Groups;

        public AudioData Process(AudioData block) => _walk.Process(block);

        public AudioData Flush() => _walk.Flush();

        private readonly double _threshold, _ratio, _knee, _makeup, _makeupGain, _window, _attack, _release, _kneeStart;
        private readonly bool _rms;
        private readonly double[] _env, _reduction, _lastReduction, _lastGain;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _walk = new DynamicsWalk(sampleRate, channels, step.Flag("link", true),
                   string.Equals(step.Option("detector", "rms"), "peak", StringComparison.OrdinalIgnoreCase) ? DynamicsDetector.Peak : DynamicsDetector.Square, Gains);
            _rms = !string.Equals(step.Option("detector", "rms"), "peak", StringComparison.OrdinalIgnoreCase);
            (_threshold, _ratio, _knee) = (step.Number(-20), step.Number("ratio", 3), step.Number("knee", 6));
            _makeup = AutoMakeup(step) ? -Curve(0, _threshold, _ratio, _knee) : step.Number("makeup", 0);
            _makeupGain = DynamicsMath.Gain(_makeup);
            _window = DynamicsMath.Coefficient(step.Number("window", 10), sampleRate);
            _attack = DynamicsMath.Coefficient(step.Number("attack", 10), sampleRate);
            _release = DynamicsMath.Coefficient(step.Number("release", 100), sampleRate);
            // Under the knee's start the curve gives 0 dB: the level is compared there, without a logarithm.
            _kneeStart = Math.Pow(10, (_threshold - _knee / 2) / (_rms ? 10 : 20));
            (_env, _reduction, _lastReduction, _lastGain) = (new double[Groups], new double[Groups], new double[Groups], new double[Groups]);
            Array.Fill(_lastGain, _makeupGain);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            double env = _env[group], reduction = _reduction[group], last = _lastReduction[group], lastGain = _lastGain[group];
            for (int i = 0; i < level.Length; i++)
            {
                double x = level[i];
                if (_rms)
                {
                    env = DynamicsMath.Follow(env, x, _window, DynamicsMath.SilentPower);
                    x = env;
                }

                double wanted = x > _kneeStart ? Curve((_rms ? 10 : 20) * Math.Log10(x), _threshold, _ratio, _knee) : 0;
                reduction = DynamicsMath.Follow(reduction, wanted, wanted < reduction ? _attack : _release, DynamicsMath.GainSnap);
                if (reduction != last)
                {
                    (last, lastGain) = (reduction, DynamicsMath.Gain(reduction + _makeup));
                }

                level[i] = lastGain;
            }

            (_env[group], _reduction[group], _lastReduction[group], _lastGain[group]) = (env, reduction, last, lastGain);
        }
    }
}

// "limit" (live): a look-ahead peak limiter; no sample passes the ceiling (the value, dBFS). L = round(lookahead · fs)
// samples. Each sample's needed gain r = min(1, ceiling / peak), the peak the largest magnitude over the linked channels;
// M the least r over the last L + 1 samples (a running minimum); the envelope follows M down at once and back up with
// `release`; the gain is the mean of the envelope's last L + 1 values, applied to the input L samples back. Every one of
// those L + 1 values is at most the r of the sample it is applied to, so their mean is too; each sample is then clamped
// to the largest float not over the ceiling, which only rounding can reach. The output lags by L (the stage's Latency):
// silence first, and Flush gives the last L samples (the input followed by L samples of silence).
internal sealed class LimitProcessor : ILiveAudioProcessor
{
    public string Name => DynamicsProcessors.Limit;

    public string Label => "Limiter";

    public string Summary => "a look-ahead peak limiter: no sample passes the value, dBFS (default -1); the output lags by `lookahead` ms, the gain comes back with `release`";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Ceiling", AudioProcessorInput.Number) { Max = 0, Step = 0.5, Unit = "dBFS", Optional = true, Default = "-1", Help = "The level no sample passes" },
        new("lookahead", "Look-ahead", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1000, Step = 1, Unit = "ms", Optional = true, Default = "5", Help = "How far ahead the gain sees a peak coming (the output's delay)",
        },
        DynamicsParameters.Release("50", "How fast the gain comes back after a peak"),
        DynamicsParameters.Link,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step, sampleRate, channels);

    private sealed class Stage : IAudioStage
    {
        private const int Chunk = 1024;
        private readonly int _lookahead, _window, _groups;
        private readonly bool _linked;
        private readonly double _ceiling, _release;
        private readonly float _clamp;
        private readonly double[] _level = new double[Chunk];
        private readonly double[] _minValue, _ring, _sum, _env;                 // per group: the running minimum's queue, the envelope's last L + 1 values
        private readonly long[] _minIndex;
        private readonly int[] _minHead, _minCount;
        private readonly float[] _delay;                                         // per channel: its last L input samples
        private long _position;                                                  // samples taken so far

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            (SampleRate, Channels) = (sampleRate, channels);
            _lookahead = DynamicsMath.Samples(step.Number("lookahead", 5), sampleRate);
            _window = _lookahead + 1;
            _linked = step.Flag("link", true);
            _groups = _linked ? 1 : channels;
            _ceiling = DynamicsMath.Gain(step.Number(-1));
            _clamp = (float)_ceiling;
            if (_clamp > _ceiling)
            {
                _clamp = MathF.BitDecrement(_clamp);
            }

            _release = DynamicsMath.Coefficient(step.Number("release", 50), sampleRate);
            (_minValue, _minIndex, _minHead, _minCount) = (new double[_groups * _window], new long[_groups * _window], new int[_groups], new int[_groups]);
            _ring = new double[_groups * _window];
            Array.Fill(_ring, 1.0);                                              // before the start: silence, gain 1
            (_sum, _env) = (new double[_groups], new double[_groups]);
            Array.Fill(_sum, _window);
            Array.Fill(_env, 1.0);
            _delay = new float[channels * _lookahead];
        }

        public int SampleRate { get; }

        public int Channels { get; }

        public int Latency => _lookahead;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public AudioData Process(AudioData block)
        {
            ArgumentNullException.ThrowIfNull(block);
            if (block.SampleRate != SampleRate || block.Channels != Channels)
            {
                throw new ArgumentException($"The limiter was opened for {Channels} channel(s) at {SampleRate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
            }

            int length = block.Length, perGroup = _linked ? Channels : 1;
            var samples = block.Samples;
            for (int start = 0; start < length; start += Chunk)
            {
                int n = Math.Min(Chunk, length - start);
                var level = _level.AsSpan(0, n);
                for (int g = 0; g < _groups; g++)
                {
                    int first = _linked ? 0 : g;
                    for (int c = first; c < first + perGroup; c++)
                    {
                        var x = samples.AsSpan(c * length + start, n);
                        for (int i = 0; i < n; i++)
                        {
                            double v = Math.Abs((double)x[i]);
                            level[i] = c == first ? v : Math.Max(level[i], v);
                        }
                    }

                    Gains(g, level);
                    for (int c = first; c < first + perGroup; c++)
                    {
                        Delay(c, samples.AsSpan(c * length + start, n), level);
                    }
                }

                _position += n;
            }

            return block;
        }

        public AudioData Flush()
        {
            var tail = new AudioData(new float[Channels * _lookahead], Channels, SampleRate);
            return _lookahead == 0 ? tail : Process(tail);
        }

        // The gains of the samples _position.. of a chunk, from their peaks (overwritten with the gains).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            var values = _minValue.AsSpan(group * _window, _window);
            var indices = _minIndex.AsSpan(group * _window, _window);
            var ring = _ring.AsSpan(group * _window, _window);
            int head = _minHead[group], count = _minCount[group];
            double sum = _sum[group], env = _env[group];
            long index = _position;
            int slot = (int)(index % _window);
            for (int i = 0; i < level.Length; i++, index++)
            {
                double needed = level[i] <= _ceiling ? 1.0 : _ceiling / level[i];
                if (count > 0 && indices[head] <= index - _window)
                {
                    head = head + 1 == _window ? 0 : head + 1;                   // out of the last L + 1 samples (one a sample at most)
                    count--;
                }

                int back = head + count < _window ? head + count : head + count - _window;
                while (count > 0 && values[back == 0 ? _window - 1 : back - 1] >= needed)
                {
                    count--;                                                     // a later, smaller need hides it
                    back = back == 0 ? _window - 1 : back - 1;
                }

                (values[back], indices[back]) = (needed, index);
                count++;
                double least = values[head];
                env = least < env ? least : DynamicsMath.Follow(env, least, _release, DynamicsMath.GainSnap);
                sum += env - ring[slot];
                ring[slot] = env;
                slot = slot + 1 == _window ? 0 : slot + 1;
                level[i] = sum / _window;
            }

            (_minHead[group], _minCount[group], _sum[group], _env[group]) = (head, count, sum, env);
        }

        // Channel c's samples of a chunk become its samples L back times the gains, clamped under the ceiling.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Delay(int c, Span<float> x, ReadOnlySpan<double> gains)
        {
            if (_lookahead == 0)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = Math.Clamp((float)(x[i] * gains[i]), -_clamp, _clamp);
                }

                return;
            }

            var delay = _delay.AsSpan(c * _lookahead, _lookahead);
            int at = (int)(_position % _lookahead);
            for (int i = 0; i < x.Length; i++)
            {
                float late = delay[at];
                delay[at] = x[i];
                at = at + 1 == _lookahead ? 0 : at + 1;
                x[i] = Math.Clamp((float)(late * gains[i]), -_clamp, _clamp);
            }
        }
    }
}

// "agc" (live): automatic gain control towards the value, a speech level in dBFS RMS. The level is a one-pole mean square
// (`window`) of the mean square over the linked channels; at or above `quiet` dBFS the wanted gain is value - level,
// within `min` and `max` dB, and under it (a pause, silence) the gain holds; the gain in dB follows the wanted gain with
// `attack` (down) or `release` (up), from 0 dB.
internal sealed class AgcProcessor : ILiveAudioProcessor
{
    public string Name => DynamicsProcessors.Agc;

    public string Label => "Automatic gain control";

    public string Summary => "automatic gain control: the level (an RMS over `window` ms) brought towards the value, dBFS (default -20), within `min` and `max` dB, held through the pauses under `quiet`";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Target", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-20", Help = "The speech level (RMS) aimed at" },
        DynamicsParameters.MaxGain,
        DynamicsParameters.MinGain,
        DynamicsParameters.Attack("100", "How fast the gain comes down when the sound gets louder"),
        DynamicsParameters.Release("1000", "How fast the gain goes up when the sound gets quieter"),
        DynamicsParameters.Window("300", "The time constant of the level (a one-pole mean square)"),
        new("quiet", "Silence under", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-50", Help = "Under this level the gain holds (pauses are not turned up)" },
        DynamicsParameters.Link,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step, sampleRate, channels);

    private sealed class Stage : IAudioStage
    {
        private readonly DynamicsWalk _walk;

        public int SampleRate => _walk.SampleRate;

        public int Channels => _walk.Channels;

        private int Groups => _walk.Groups;

        public AudioData Process(AudioData block) => _walk.Process(block);

        public AudioData Flush() => _walk.Flush();

        private readonly double _target, _max, _min, _attack, _release, _window, _gate;
        private readonly double[] _env, _gain, _last, _lastGain;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _walk = new DynamicsWalk(sampleRate, channels, step.Flag("link", true), DynamicsDetector.MeanSquare, Gains);
            (_target, _max, _min) = (step.Number(-20), step.Number("max", 20), step.Number("min", -20));
            _attack = DynamicsMath.Coefficient(step.Number("attack", 100), sampleRate);
            _release = DynamicsMath.Coefficient(step.Number("release", 1000), sampleRate);
            _window = DynamicsMath.Coefficient(step.Number("window", 300), sampleRate);
            _gate = Math.Pow(10, step.Number("quiet", -50) / 10);
            (_env, _gain, _last, _lastGain) = (new double[Groups], new double[Groups], new double[Groups], new double[Groups]);
            Array.Fill(_lastGain, 1.0);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            double env = _env[group], gain = _gain[group], last = _last[group], lastGain = _lastGain[group];
            for (int i = 0; i < level.Length; i++)
            {
                env = DynamicsMath.Follow(env, level[i], _window, DynamicsMath.SilentPower);
                if (env >= _gate)
                {
                    double wanted = Math.Min(Math.Max(_target - 10 * Math.Log10(env), _min), _max);
                    gain = DynamicsMath.Follow(gain, wanted, wanted < gain ? _attack : _release, DynamicsMath.GainSnap);
                    if (gain != last)
                    {
                        (last, lastGain) = (gain, DynamicsMath.Gain(gain));
                    }
                }

                level[i] = lastGain;
            }

            (_env[group], _gain[group], _last[group], _lastGain[group]) = (env, gain, last, lastGain);
        }
    }
}

// The K-weighting of ITU-R BS.1770-4 (a high shelf, then the RLB high-pass) for any rate, as two biquads in double,
// direct form I from rest: bs1770 is Brecht De Man's design, which gives the standard's 48 kHz coefficients; rbj is the
// RBJ cookbook's 4 dB shelf at 1500 Hz (Q 1/√2) and 38 Hz high-pass (Q 0.5), pyloudnorm's default. A channel's weight
// follows BS.1770's five-channel order (L, R, C, Ls, Rs): 1.41 for the surrounds, else 1.
internal static class KWeighting
{
    private const double ShelfGain = 3.99984385397, ShelfQ = 0.7071752369554193, ShelfHertz = 1681.9744509555319;
    private const double PassQ = 0.5003270373253953, PassHertz = 38.13547087613982;

    // b0, b1, b2, a1, a2 of the shelf, then of the high-pass.
    public static double[] Coefficients(string? weighting, int sampleRate, string name)
    {
        bool rbj = string.Equals(weighting, "rbj", StringComparison.OrdinalIgnoreCase);
        double shelfHertz = rbj ? 1500 : ShelfHertz;
        if (sampleRate <= 2 * shelfHertz)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"{name}: the K-weighting's shelf at {shelfHertz:0.#} Hz needs a rate above {2 * shelfHertz:0.#} Hz, not {sampleRate} Hz."));
        }

        if (rbj)
        {
            double a = Math.Pow(10, 4.0 / 40.0), w0 = 2.0 * Math.PI * 1500.0 / sampleRate;
            double alpha = Math.Sin(w0) / (2.0 * (1 / Math.Sqrt(2))), cos = Math.Cos(w0), root = Math.Sqrt(a);
            double s0 = (a + 1) - (a - 1) * cos + 2 * root * alpha;
            double w1 = 2.0 * Math.PI * 38.0 / sampleRate, alpha1 = Math.Sin(w1) / (2.0 * 0.5), cos1 = Math.Cos(w1), h0 = 1 + alpha1;
            return
            [
                a * ((a + 1) + (a - 1) * cos + 2 * root * alpha) / s0, -2 * a * ((a - 1) + (a + 1) * cos) / s0,
                a * ((a + 1) + (a - 1) * cos - 2 * root * alpha) / s0, 2 * ((a - 1) - (a + 1) * cos) / s0,
                ((a + 1) - (a - 1) * cos - 2 * root * alpha) / s0,
                (1 + cos1) / 2 / h0, -(1 + cos1) / h0, (1 + cos1) / 2 / h0, -2 * cos1 / h0, (1 - alpha1) / h0,
            ];
        }

        double k = Math.Tan(Math.PI * ShelfHertz / sampleRate), vh = Math.Pow(10, ShelfGain / 20.0), vb = Math.Pow(vh, 0.499666774155);
        double a0 = 1.0 + k / ShelfQ + k * k;
        double kp = Math.Tan(Math.PI * PassHertz / sampleRate), p0 = 1.0 + kp / PassQ + kp * kp;
        return
        [
            (vh + vb * k / ShelfQ + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / ShelfQ + k * k) / a0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / ShelfQ + k * k) / a0,
            1.0, -2.0, 1.0, 2.0 * (kp * kp - 1.0) / p0, (1.0 - kp / PassQ + kp * kp) / p0,
        ];
    }

    public static double Weight(int channel) => channel is 3 or 4 ? 1.41 : 1.0;

    // Channel x through both filters (state: x1, x2, y1, y2 of each, kept between calls): each output, squared and
    // weighted, written to `into` (first) or added to it.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Squares(ReadOnlySpan<float> x, double[] f, Span<double> state, double weight, Span<double> into, bool first)
    {
        double x1 = state[0], x2 = state[1], y1 = state[2], y2 = state[3], u1 = state[4], u2 = state[5], z1 = state[6], z2 = state[7];
        for (int i = 0; i < x.Length; i++)
        {
            double xi = x[i];
            double yi = f[0] * xi + f[1] * x1 + f[2] * x2 - f[3] * y1 - f[4] * y2;
            double zi = f[5] * yi + f[6] * u1 + f[7] * u2 - f[8] * z1 - f[9] * z2;
            (x2, x1, y2, y1, u2, u1, z2, z1) = (x1, xi, y1, yi, u1, yi, z1, zi);
            double d = weight * zi * zi;
            into[i] = first ? d : into[i] + d;
        }

        (state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]) = (x1, x2, y1, y2, u1, u2, z1, z2);
    }
}

// "loudness" (whole recording): ITU-R BS.1770-4 / EBU R128 integrated loudness brought to the value, LUFS, by one gain
// over every channel. The K-weighted squares (weighted per channel) summed over the channels; blocks of `block` ms every
// `block` · (1 - `overlap`) (complete blocks only; a recording shorter than one block is one block), each block's
// loudness -0.691 + 10 log10 of its mean; the blocks above `absolute` LUFS, then above their mean's loudness plus
// `relative` LU, averaged. The gain is lowered so the peak (meter=true: the true peak; sample: the sample peak; none: not
// kept) stays at or under `ceiling` dBFS. A silent recording stays as it is.
internal sealed class LoudnessProcessor : IRecordingAudioProcessor
{
    public string Name => DynamicsProcessors.Loudness;

    public string Label => "Loudness (EBU R128)";

    public string Summary => "the integrated loudness (ITU-R BS.1770-4, EBU R128) brought to the value, LUFS (default -23), the true peak kept under `ceiling` dBFS";

    public string? LiveAlternative => DynamicsProcessors.LoudnessLive;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Target", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "LUFS", Optional = true, Default = "-23", Help = "The integrated loudness aimed at (EBU R128: -23; streaming: about -14 to -16)" },
        new("ceiling", "Peak ceiling", AudioProcessorInput.Number) { Max = 0, Step = 0.5, Unit = "dBFS", Optional = true, Default = "-1", Help = "The highest peak allowed after the gain (the gain is lowered to keep under it)" },
        new("meter", "Peak meter", AudioProcessorInput.Choice)
        {
            Choices = ["true", "sample", "none"], Optional = true, Default = "true", Help = "true: the true peak (between the samples too); sample: the samples' peak; none: no ceiling",
        },
        DynamicsParameters.Oversample,
        DynamicsParameters.Taps,
        new("block", "Block", AudioProcessorInput.Number) { Min = 0, MinExclusive = true, Step = 100, Unit = "ms", Optional = true, Default = "400", Help = "The gating block's length" },
        new("overlap", "Block overlap", AudioProcessorInput.Number) { Min = 0, Max = 0.99, Step = 0.05, Optional = true, Default = "0.75", Help = "The share of a block the next one overlaps" },
        new("absolute", "Absolute gate", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "LUFS", Optional = true, Default = "-70", Help = "Blocks under it are left out" },
        new("relative", "Relative gate", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "LU", Optional = true, Default = "-10", Help = "Blocks this far under the absolute-gated loudness are left out" },
        DynamicsParameters.Weighting,
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(step);
        if (audio.Length == 0)
        {
            return audio;
        }

        if (Integrated(audio, step) is not { } level)
        {
            return audio;
        }

        double gain = DynamicsMath.Gain(step.Number(-23) - level);
        string meter = step.Option("meter", "true")!;
        if (!meter.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            double peak = meter.Equals("sample", StringComparison.OrdinalIgnoreCase) ? DynamicsMath.SamplePeak(audio.Samples)
                : DynamicsMath.TruePeak(audio, step.Integer("oversample", 4), step.Integer("taps", 6));
            double ceiling = DynamicsMath.Gain(step.Number("ceiling", -1));
            if (peak * gain > ceiling)
            {
                gain = ceiling / peak;
            }
        }

        return PeakProcessor.Scale(audio, gain);
    }

    // The integrated loudness, LUFS, under step's block, gates and weighting; null when no block passes the gates.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double? Integrated(AudioData audio, AudioProcessorStep step)
    {
        int rate = audio.SampleRate, length = audio.Length;
        var filter = KWeighting.Coefficients(step.Option("weighting"), rate, step.Name);
        int size = DynamicsMath.Samples(step.Number("block", 400), rate);
        if (size < 1)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{step.Name}: a block of {step.Number("block", 400)} ms is under one sample at {rate} Hz."));
        }

        int hop = Math.Max(1, (int)Math.Round(size * (1 - step.Number("overlap", 0.75))));
        double absolute = step.Number("absolute", -70), relative = step.Number("relative", -10);
        var squares = new double[length];
        var state = new double[8];
        for (int c = 0; c < audio.Channels; c++)
        {
            Array.Clear(state);
            KWeighting.Squares(audio.Channel(c), filter, state, KWeighting.Weight(c), squares, c == 0);
        }

        int blocks = length < size ? 1 : (length - size) / hop + 1;
        var means = new double[blocks];
        var loudness = new double[blocks];
        double gated = 0;
        int count = 0;
        for (int j = 0; j < blocks; j++)
        {
            var span = length < size ? squares.AsSpan() : squares.AsSpan(j * hop, size);
            double sum = 0;
            foreach (double v in span)
            {
                sum += v;
            }

            means[j] = sum / span.Length;
            loudness[j] = means[j] > 0 ? -0.691 + 10 * Math.Log10(means[j]) : double.NegativeInfinity;
            if (loudness[j] > absolute)
            {
                gated += means[j];
                count++;
            }
        }

        if (count == 0)
        {
            return null;
        }

        double gamma = -0.691 + 10 * Math.Log10(gated / count) + relative, kept = 0;
        count = 0;
        for (int j = 0; j < blocks; j++)
        {
            if (loudness[j] > absolute && loudness[j] > gamma)
            {
                kept += means[j];
                count++;
            }
        }

        return count == 0 ? null : -0.691 + 10 * Math.Log10(kept / count);
    }
}

// "loudness-live" (live): the loudness steered as it comes. The K-weighted squares (weighted per channel, summed over the
// channels) are summed over hops of `hop` ms; at each hop's end the loudness of the last `window` ms of hops (of those
// so far at the start: 400 is EBU R128's momentary loudness, 3000 its short-term) is measured; at or above `quiet` LUFS
// the wanted gain is the value - loudness, within `min` and `max` dB, else (and before the first hop) the gain holds; the
// gain in dB follows the wanted gain sample by sample with `attack` (down) or `release` (up), from 0 dB. Follow it with
// limit to keep the peaks under a ceiling.
internal sealed class LiveLoudnessProcessor : ILiveAudioProcessor
{
    public string Name => DynamicsProcessors.LoudnessLive;

    public string Label => "Loudness (live)";

    public string Summary => "the short-term loudness (BS.1770, over `window` ms) steered towards the value, LUFS (default -23), smoothly, within `min` and `max` dB";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Target", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "LUFS", Optional = true, Default = "-23", Help = "The loudness aimed at" },
        DynamicsParameters.Window("3000", "The loudness is measured over this much sound (3000: EBU R128's short-term, 400: momentary)") with { Min = 0, MinExclusive = true, Step = 100 },
        new("hop", "Update", AudioProcessorInput.Number) { Min = 0, MinExclusive = true, Step = 10, Unit = "ms", Optional = true, Default = "100", Help = "How often the loudness is measured" },
        DynamicsParameters.Attack("500", "How fast the gain comes down when the sound gets louder") with { Step = 50 },
        DynamicsParameters.Release("2000", "How fast the gain goes up when the sound gets quieter"),
        DynamicsParameters.MaxGain,
        DynamicsParameters.MinGain,
        new("quiet", "Silence under", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "LUFS", Optional = true, Default = "-50", Help = "Under this loudness the gain holds (pauses are not turned up)" },
        DynamicsParameters.Weighting,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step, sampleRate, channels);

    private sealed class Stage : IAudioStage
    {
        private readonly DynamicsWalk _walk;

        public int SampleRate => _walk.SampleRate;

        public int Channels => _walk.Channels;

        private int Groups => _walk.Groups;

        public AudioData Process(AudioData block) => _walk.Process(block);

        public AudioData Flush() => _walk.Flush();

        private readonly double[] _filter, _state, _ring;
        private readonly double _target, _max, _min, _attack, _release, _gate;
        private readonly int _hop;
        private int _at, _filled, _count;
        private double _sum, _gain, _wanted, _last, _lastGain = 1;
        private bool _holding = true;

        public Stage(AudioProcessorStep step, int sampleRate, int channels)
        {
            _walk = new DynamicsWalk(sampleRate, channels, linked: true, DynamicsDetector.Square, Gains, Detect);
            _filter = KWeighting.Coefficients(step.Option("weighting"), sampleRate, step.Name);
            _state = new double[8 * channels];
            double hop = step.Number("hop", 100);
            _hop = Math.Max(1, DynamicsMath.Samples(hop, sampleRate));
            _ring = new double[Math.Max(1, (int)Math.Round(step.Number("window", 3000) / hop))];
            (_target, _max, _min, _gate) = (step.Number(-23), step.Number("max", 20), step.Number("min", -20), step.Number("quiet", -50));
            _attack = DynamicsMath.Coefficient(step.Number("attack", 500), sampleRate);
            _release = DynamicsMath.Coefficient(step.Number("release", 2000), sampleRate);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Detect(int channel, ReadOnlySpan<float> x, Span<double> level, bool first) =>
            KWeighting.Squares(x, _filter, _state.AsSpan(8 * channel, 8), KWeighting.Weight(channel), level, first);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Gains(int group, Span<double> level)
        {
            for (int i = 0; i < level.Length; i++)
            {
                _sum += level[i];
                if (++_count == _hop)
                {
                    _ring[_at] = _sum;
                    _at = _at + 1 == _ring.Length ? 0 : _at + 1;
                    _filled = Math.Min(_filled + 1, _ring.Length);
                    (_sum, _count) = (0, 0);
                    double total = 0;
                    for (int k = 0; k < _filled; k++)
                    {
                        total += _ring[k];
                    }

                    double loudness = total > 0 ? -0.691 + 10 * Math.Log10(total / (_filled * _hop)) : double.NegativeInfinity;
                    _holding = loudness < _gate;
                    if (!_holding)
                    {
                        _wanted = Math.Min(Math.Max(_target - loudness, _min), _max);
                    }
                }

                if (!_holding)
                {
                    _gain = DynamicsMath.Follow(_gain, _wanted, _wanted < _gain ? _attack : _release, DynamicsMath.GainSnap);
                    if (_gain != _last)
                    {
                        (_last, _lastGain) = (_gain, DynamicsMath.Gain(_gain));
                    }
                }

                level[i] = _lastGain;
            }
        }
    }
}

// "peaknorm" (whole recording): peak normalization, the largest peak (meter=sample: of the samples; true: the true peak, between the samples
// too) brought to the value, dBFS, by one gain over every channel. A silent recording stays as it is.
internal sealed class PeakProcessor : IRecordingAudioProcessor
{
    public string Name => DynamicsProcessors.Peak;

    public string Label => "Peak normalize";

    public string Summary => "the peak brought to the value, dBFS (default -1), by one gain (meter=true: the true peak)";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Peak", AudioProcessorInput.Number) { Max = 0, Step = 0.5, Unit = "dBFS", Optional = true, Default = "-1", Help = "The peak aimed at" },
        new("meter", "Peak meter", AudioProcessorInput.Choice)
        {
            Choices = ["sample", "true"], Optional = true, Default = "sample", Help = "sample: the samples' peak; true: the true peak (between the samples too)",
        },
        DynamicsParameters.Oversample,
        DynamicsParameters.Taps,
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(step);
        double peak = string.Equals(step.Option("meter", "sample"), "true", StringComparison.OrdinalIgnoreCase)
            ? DynamicsMath.TruePeak(audio, step.Integer("oversample", 4), step.Integer("taps", 6))
            : DynamicsMath.SamplePeak(audio.Samples);
        return peak > 0 ? Scale(audio, DynamicsMath.Gain(step.Number(-1)) / peak) : audio;
    }

    // The samples times one gain (as a float), new samples; the audio itself when the gain is 1.
    public static AudioData Scale(AudioData audio, double gain)
    {
        if ((float)gain == 1f)
        {
            return audio;
        }

        var output = GC.AllocateUninitializedArray<float>(audio.Samples.Length);
        CleaningLevels.Scale(audio.Samples, (float)gain, output);
        return new AudioData(output, audio.Channels, audio.SampleRate);
    }
}
