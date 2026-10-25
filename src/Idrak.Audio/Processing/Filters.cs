// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;

namespace Idrak.Audio;

// The filters (plan 31, P1), all live: each opens a stage that keeps its memory between blocks, so a recording read in
// blocks of any size gives the samples the whole recording gives, bit for bit. The linear filters are cascades of
// second-order sections (Biquad) run by one stage (BiquadStage): double state, float samples written in place, each
// channel in its memory order, nothing allocated per block. Every filter starts settled on the first sample's level
// (its memory filled as if that level had always been there: scipy's sosfilt_zi or lfilter_zi times the first sample),
// so a DC offset or a level at the start makes no step. Their references are scipy's (tools/audio/make_audio_filters.py).
internal static class FilterProcessors
{
    public const string Gain = "gain", Dc = "dc", LowPass = "lowpass", BandPass = "bandpass", Notch = "notch", Dehum = "dehum", Eq = "eq",
        LowShelf = "lowshelf", HighShelf = "highshelf", Presence = "presence", PreEmphasis = "preemphasis", Mono = "mono", Resample = "resample";

    // The library's filters, registered after the cleaners (AudioProcessors.BuiltIn).
    public static void RegisterDefaults(KindTable<IAudioProcessor> table)
    {
        ArgumentNullException.ThrowIfNull(table);
        table.RegisterDefault(new GainProcessor());
        table.RegisterDefault(new DcProcessor());
        table.RegisterDefault(new LowPassProcessor());
        table.RegisterDefault(new BandPassProcessor());
        table.RegisterDefault(new NotchProcessor());
        table.RegisterDefault(new DehumProcessor());
        table.RegisterDefault(new PeakingProcessor());
        table.RegisterDefault(new ShelfProcessor(high: false));
        table.RegisterDefault(new ShelfProcessor(high: true));
        table.RegisterDefault(new PresenceProcessor());
        table.RegisterDefault(new PreEmphasisProcessor());
        table.RegisterDefault(new MonoProcessor());
        table.RegisterDefault(new ResampleProcessor());
    }
}

// One second-order section, a0 divided out: y = b0·x + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2]; a first-order one
// has b2 = a2 = 0. DcGain is its gain at 0 Hz, given exactly by each design (0 for a high-pass), so a stage settles on
// the first sample's level without a rounding off the formula's value.
internal readonly record struct Biquad(double B0, double B1, double B2, double A1, double A2, double DcGain)
{
    // The RBJ cookbook's high-pass, twoQ = 2·Q; at twoQ = √2 the arithmetic of the high-pass before plan 31, bit for bit.
    public static Biquad HighPass(double hertz, int rate, double twoQ)
    {
        double w0 = 2 * Math.PI * hertz / rate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / twoQ;
        double a0 = 1 + alpha;
        double b0 = (1 + cos) / 2 / a0;
        return new Biquad(b0, -(1 + cos) / a0, b0, -2 * cos / a0, (1 - alpha) / a0, 0);
    }

    // The RBJ cookbook's low-pass.
    public static Biquad LowPass(double hertz, int rate, double twoQ)
    {
        double w0 = 2 * Math.PI * hertz / rate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / twoQ;
        double a0 = 1 + alpha;
        double b0 = (1 - cos) / 2 / a0;
        return new Biquad(b0, (1 - cos) / a0, b0, -2 * cos / a0, (1 - alpha) / a0, 1);
    }

    // The bilinear transform of s / (s + ωc) and ωc / (s + ωc), prewarped at the cutoff (K = tan(π·f / rate)): the odd
    // order's first-order section, as scipy's butter makes it.
    public static Biquad FirstOrder(bool high, double hertz, int rate)
    {
        double k = Math.Tan(Math.PI * hertz / rate), a1 = (k - 1) / (k + 1);
        return high ? new Biquad(1 / (1 + k), -1 / (1 + k), 0, a1, 0, 0) : new Biquad(k / (1 + k), k / (1 + k), 0, a1, 0, 1);
    }

    // scipy.signal.iirnotch's: the -3 dB bandwidth is hertz / q exactly, the gain 1 away from the notch.
    public static Biquad Notch(double hertz, double q, int rate)
    {
        double w0 = 2 * Math.PI * hertz / rate, cos = Math.Cos(w0);
        double gain = 1 / (1 + Math.Tan(w0 / q / 2));
        return new Biquad(gain, -2 * gain * cos, gain, -2 * gain * cos, 2 * gain - 1, 1);
    }

    // The RBJ cookbook's peaking EQ: decibels at hertz, its width q; 1 far from it.
    public static Biquad Peaking(double hertz, double decibels, double q, int rate)
    {
        double w0 = 2 * Math.PI * hertz / rate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / (2 * q), a = Math.Pow(10, decibels / 40);
        double a0 = 1 + alpha / a;
        return new Biquad((1 + alpha * a) / a0, -2 * cos / a0, (1 - alpha * a) / a0, -2 * cos / a0, (1 - alpha / a) / a0, 1);
    }

    // The RBJ cookbook's shelves: decibels below (low) or above (high) hertz, alpha from the slope or q.
    public static Biquad Shelf(bool high, double hertz, double decibels, double alphaOverSin, int rate)
    {
        double w0 = 2 * Math.PI * hertz / rate, cos = Math.Cos(w0), alpha = Math.Sin(w0) * alphaOverSin, a = Math.Pow(10, decibels / 40);
        double root = 2 * Math.Sqrt(a) * alpha;
        if (high)
        {
            double a0 = (a + 1) - (a - 1) * cos + root;
            return new Biquad(a * ((a + 1) + (a - 1) * cos + root) / a0, -2 * a * ((a - 1) + (a + 1) * cos) / a0, a * ((a + 1) + (a - 1) * cos - root) / a0,
                2 * ((a - 1) - (a + 1) * cos) / a0, ((a + 1) - (a - 1) * cos - root) / a0, 1);
        }
        else
        {
            double a0 = (a + 1) + (a - 1) * cos + root;
            return new Biquad(a * ((a + 1) - (a - 1) * cos + root) / a0, 2 * a * ((a - 1) - (a + 1) * cos) / a0, a * ((a + 1) - (a - 1) * cos - root) / a0,
                -2 * ((a - 1) + (a + 1) * cos) / a0, ((a + 1) + (a - 1) * cos - root) / a0, a * a);
        }
    }
}

// The designs and checks the filters share.
internal static class FilterDesign
{
    // A Butterworth filter of `order` at hertz as the bilinear transform prewarped at the cutoff (scipy's butter): order / 2
    // RBJ sections, the k-th with 2·Q = 1 / sin(π(2k + 1) / 2·order), and one first-order section when the order is odd.
    // Order 2 takes 2·Q = √2 as the high-pass always has (1 / sin(π/4) may round otherwise).
    public static Biquad[] Butterworth(bool high, int order, double hertz, int rate)
    {
        var sections = new Biquad[(order + 1) / 2];
        int pairs = order / 2;
        for (int k = 0; k < pairs; k++)
        {
            double twoQ = order == 2 ? Math.Sqrt(2) : 1 / Math.Sin(Math.PI * (2 * k + 1) / (2 * order));
            sections[k] = high ? Biquad.HighPass(hertz, rate, twoQ) : Biquad.LowPass(hertz, rate, twoQ);
        }

        if (order % 2 == 1)
        {
            sections[pairs] = Biquad.FirstOrder(high, hertz, rate);
        }

        return sections;
    }

    // Refuses a frequency at or over half the rate, naming both (as the high-pass always has).
    public static void ThrowIfNotBelowHalf(string name, string what, double hertz, int rate)
    {
        if (hertz >= rate / 2.0)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{name}: the {what} {hertz} Hz is not below half the rate ({rate} Hz)."), "step");
        }
    }

    // Refuses a step without the option `key`, which the processor cannot do without.
    public static void Require(AudioProcessorStep step, string key, string form)
    {
        if (step.Option(key) is null)
        {
            throw new ArgumentException($"{step.Name} needs {key}={form} (given '{step}').");
        }
    }

    public static readonly AudioProcessorParameter Order = new("order", "Order", AudioProcessorInput.Integer)
    {
        Min = 1, Max = 16, Step = 1, Optional = true, Default = "2", Help = "The Butterworth order: 6 dB an octave per order past the cutoff (2: 12 dB an octave)",
    };

    public static AudioProcessorParameter Frequency(string key, string label, string help) => new(key, label, AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Step = 10, Unit = "Hz", Help = help,
    };

    public static AudioProcessorParameter Q(string? fallback, string help) => new("q", "Q", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Max = 1000, Step = 0.1, Optional = true, Default = fallback, Help = help,
    };

    public static readonly AudioProcessorParameter Decibels = new("db", "Gain", AudioProcessorInput.Number)
    {
        Min = -48, Max = 48, Step = 0.5, Unit = "dB", Help = "The boost (above 0) or cut (below 0); required",
    };
}

// A cascade of second-order sections over each channel (see FilterProcessors): each sample through every section in
// double before it is written back as a float. No section passes the sound as it is.
internal sealed class BiquadStage : IAudioStage
{
    private const int Chunk = 1024;                                             // a cascade's piece: its scratch stays in the first cache

    private readonly Biquad[] _sections;
    private readonly double[] _state;                                           // x1, x2, y1, y2 per section, channel by channel
    private readonly double[] _scratch;                                         // a cascade's piece in double, made when opened
    private bool _started;

    public BiquadStage(Biquad[] sections, int sampleRate, int channels)
    {
        (_sections, SampleRate, Channels) = (sections, sampleRate, channels);
        _state = new double[4 * sections.Length * channels];
        _scratch = sections.Length > 1 ? new double[Chunk] : [];
    }

    public int SampleRate { get; }

    public int Channels { get; }

    public AudioData Process(AudioData block)
    {
        ArgumentNullException.ThrowIfNull(block);
        int length = block.Length, count = _sections.Length;
        if (count == 0 || length == 0)
        {
            return block;
        }

        var samples = block.Samples;
        for (int c = 0; c < Channels; c++)
        {
            var x = samples.AsSpan(c * length, length);
            var state = _state.AsSpan(4 * count * c, 4 * count);
            if (!_started)
            {
                Settle(state, x[0]);
            }

            if (count == 1)
            {
                One(x, _sections[0], state);
            }
            else
            {
                Cascade(x, state);
            }
        }

        _started = true;
        return block;
    }

    public AudioData Flush() => new([], Channels, SampleRate);

    // The filters made of second-order sections open through here: the step checked (the processor's Check), then a stage
    // of its design at the rate (a frequency at or over half the rate refused by the design, naming both).
    public static BiquadStage Checked(IAudioProcessor processor, AudioProcessorStep step, int sampleRate, int channels, Func<AudioProcessorStep, int, Biquad[]> design)
    {
        ArgumentNullException.ThrowIfNull(step);
        processor.Check(step);
        return new BiquadStage(design(step, sampleRate), sampleRate, channels);
    }

    // Each section's memory as if `level` had always come in: its input that level, its output that level times its
    // gain at 0 Hz (exactly 0 for a high-pass), the next section's input that output.
    private void Settle(Span<double> state, double level)
    {
        for (int s = 0; s < _sections.Length; s++)
        {
            double gain = _sections[s].DcGain, settled = gain == 0 ? 0 : level * gain;
            (state[4 * s], state[4 * s + 1], state[4 * s + 2], state[4 * s + 3]) = (level, level, settled, settled);
            level = settled;
        }
    }

    // One section, its memory in locals (the high-pass before plan 31's loop, the same arithmetic in the same order).
    private static void One(Span<float> x, in Biquad q, Span<double> s)
    {
        double b0 = q.B0, b1 = q.B1, b2 = q.B2, a1 = q.A1, a2 = q.A2;
        double x1 = s[0], x2 = s[1], y1 = s[2], y2 = s[3];
        for (int i = 0; i < x.Length; i++)
        {
            double xi = x[i];
            double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x[i] = (float)yi;                                                   // in place: x[i] is read before it is written
            (x2, x1, y2, y1) = (x1, xi, y1, yi);
        }

        (s[0], s[1], s[2], s[3]) = (x1, x2, y1, y2);
    }

    // Several sections: the block in pieces of Chunk samples, each piece in double in the scratch, through one section
    // after the other (each with its memory in locals), then back to float: no rounding to float between sections, and
    // each sample meets the same arithmetic as a sample-by-sample cascade, whatever the block's size.
    private void Cascade(Span<float> x, Span<double> s)
    {
        for (int at = 0; at < x.Length; at += Chunk)
        {
            var piece = x.Slice(at, Math.Min(Chunk, x.Length - at));
            var work = _scratch.AsSpan(0, piece.Length);
            for (int i = 0; i < piece.Length; i++)
            {
                work[i] = piece[i];
            }

            for (int k = 0; k < _sections.Length; k++)
            {
                Section(work, _sections[k], s.Slice(4 * k, 4));
            }

            for (int i = 0; i < piece.Length; i++)
            {
                piece[i] = (float)work[i];
            }
        }
    }

    private static void Section(Span<double> v, in Biquad q, Span<double> s)
    {
        double b0 = q.B0, b1 = q.B1, b2 = q.B2, a1 = q.A1, a2 = q.A2;
        double x1 = s[0], x2 = s[1], y1 = s[2], y2 = s[3];
        for (int i = 0; i < v.Length; i++)
        {
            double xi = v[i];
            double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            v[i] = yi;
            (x2, x1, y2, y1) = (x1, xi, y1, yi);
        }

        (s[0], s[1], s[2], s[3]) = (x1, x2, y1, y2);
    }
}

// "gain" (live): every sample times 10^(value / 20), in place (SIMD with a scalar tail). Nothing is clipped: a boost can
// take samples past full scale (a limiter after it keeps them under).
internal sealed class GainProcessor : ILiveAudioProcessor
{
    public string Name => FilterProcessors.Gain;

    public string Label => "Gain";

    public string Summary => "the level changed by the value, dB (above 0 louder, below 0 quieter); nothing is clipped";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Gain", AudioProcessorInput.Number) { Min = -120, Max = 60, Step = 1, Unit = "dB", Help = "Added to the level (6 dB doubles the samples)" },
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return new Stage((float)Math.Pow(10, step.Number() / 20), sampleRate, channels);
    }

    private sealed class Stage(float factor, int sampleRate, int channels) : IAudioStage
    {
        public int SampleRate => sampleRate;

        public int Channels => channels;

        public AudioData Process(AudioData block)
        {
            ArgumentNullException.ThrowIfNull(block);
            if (factor != 1f && block.Length > 0)
            {
                var samples = block.Samples;
                CleaningLevels.Scale(samples, factor, samples);                     // element by element: in place is safe
            }

            return block;
        }

        public AudioData Flush() => new([], channels, sampleRate);
    }
}

// "dc" (live): a DC blocker, y[n] = x[n] - x[n-1] + pole·y[n-1] (a one-pole high-pass with its zero at 0 Hz; its corner
// about rate·(1 - pole) / 2π Hz), started settled on the first sample's level (its output starts at 0, not at the offset).
internal sealed class DcProcessor : ILiveAudioProcessor
{
    public const double DefaultPole = 0.995;

    public string Name => FilterProcessors.Dc;

    public string Label => "DC blocker";

    public string Summary => string.Create(CultureInfo.InvariantCulture, $"a DC blocker (a one-pole high-pass, its pole `pole`, default {DefaultPole}): the offset goes, the sound above a few Hz stays");

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("pole", "Pole", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 0.999999, Step = 0.001, Optional = true, Default = "0.995",
            Help = "Closer to 1, a lower corner (about rate·(1 - pole) / 2π Hz: 13 Hz at 16 kHz for 0.995) and a slower settling",
        },
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double pole = step.Number("pole", DefaultPole);
        return [new Biquad(1, -1, 0, -pole, 0, 0)];
    }
}

// "lowpass" (live): a Butterworth low-pass of `order` (default 2) at the value, Hz: the sound above it turned down.
internal sealed class LowPassProcessor : ILiveAudioProcessor
{
    public string Name => FilterProcessors.LowPass;

    public string Label => "Low-pass filter";

    public string Summary => "a Butterworth low-pass at the value, Hz, of `order` (default 2): the hiss and the sound above it go";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        FilterDesign.Frequency("", "Cutoff", "Above it the sound is turned down, 6 dB an octave per order; below half the rate"),
        FilterDesign.Order,
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double cutoff = step.Number();
        FilterDesign.ThrowIfNotBelowHalf(Name, "cutoff", cutoff, sampleRate);
        return FilterDesign.Butterworth(high: false, step.Integer("order", 2), cutoff, sampleRate);
    }
}

// "bandpass" (live): a Butterworth high-pass at `low` then a Butterworth low-pass at `high`, both of `order`: the band
// between them kept (300 to 3,400 Hz by default: the telephone band). Two cutoffs, not a centre and a Q, so each edge is
// set where it is heard; it is the cascade of the two filters, not scipy's band-pass transform (whose order doubles).
internal sealed class BandPassProcessor : ILiveAudioProcessor
{
    public const double DefaultLow = 300, DefaultHigh = 3400;

    public string Name => FilterProcessors.BandPass;

    public string Label => "Band-pass filter";

    public string Summary => $"the band from `low` to `high` Hz kept (default {DefaultLow} to {DefaultHigh}): a Butterworth high-pass and low-pass of `order`";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        FilterDesign.Frequency("low", "Low edge", "Below it the sound is turned down (a high-pass)") with { Optional = true, Default = "300" },
        FilterDesign.Frequency("high", "High edge", "Above it the sound is turned down (a low-pass); below half the rate") with { Optional = true, Default = "3400" },
        FilterDesign.Order,
    ];

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        double low = step.Number("low", DefaultLow), high = step.Number("high", DefaultHigh);
        if (low >= high)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"bandpass: the low edge {low} Hz is not below the high edge {high} Hz."));
        }
    }

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double low = step.Number("low", DefaultLow), high = step.Number("high", DefaultHigh);
        FilterDesign.ThrowIfNotBelowHalf(Name, "high edge", high, sampleRate);
        int order = step.Integer("order", 2);
        return [.. FilterDesign.Butterworth(high: true, order, low, sampleRate), .. FilterDesign.Butterworth(high: false, order, high, sampleRate)];
    }
}

// "notch" (live): one frequency taken out (scipy's iirnotch: the -3 dB width is value / q Hz), the rest left at its level.
internal sealed class NotchProcessor : ILiveAudioProcessor
{
    public const double DefaultQ = 30;

    public string Name => FilterProcessors.Notch;

    public string Label => "Notch filter";

    public string Summary => $"the value's frequency, Hz, taken out (a notch {DefaultQ} times narrower than it by default, `q`)";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        FilterDesign.Frequency("", "Frequency", "The frequency taken out; below half the rate"),
        FilterDesign.Q("30", "The notch's sharpness: its -3 dB width is the frequency / q"),
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double hertz = step.Number();
        FilterDesign.ThrowIfNotBelowHalf(Name, "frequency", hertz, sampleRate);
        return [Biquad.Notch(hertz, step.Number("q", DefaultQ), sampleRate)];
    }
}

// "dehum" (live): the mains hum taken out: a notch (as "notch") at the value (50 or 60 Hz) and at each multiple of it up to
// `harmonics` notches, all of the same q (so each is as wide as its frequency / q). Multiples at or over half the rate
// are left out; a mains frequency there is refused.
internal sealed class DehumProcessor : ILiveAudioProcessor
{
    public const double DefaultMains = 50, DefaultQ = 30;
    public const int DefaultHarmonics = 4;

    public string Name => FilterProcessors.Dehum;

    public string Label => "Hum remover";

    public string Summary => $"the mains hum taken out: notches at the value, Hz (50 or 60; default {DefaultMains}), and its multiples, `harmonics` in all (default {DefaultHarmonics})";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Mains", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1000, Step = 10, Unit = "Hz", Optional = true, Default = "50", Help = "The mains frequency (50 in most of the world, 60 in the Americas)",
        },
        new("harmonics", "Notches", AudioProcessorInput.Integer)
        {
            Min = 1, Max = 100, Step = 1, Optional = true, Default = "4", Help = "How many notches: the mains frequency and its multiples (1: the mains frequency only)",
        },
        FilterDesign.Q("30", "Each notch's sharpness: its -3 dB width is its frequency / q"),
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double mains = step.Number(DefaultMains), q = step.Number("q", DefaultQ);
        FilterDesign.ThrowIfNotBelowHalf(Name, "mains frequency", mains, sampleRate);
        int harmonics = step.Integer("harmonics", DefaultHarmonics);
        var sections = new List<Biquad>(harmonics);
        for (int k = 1; k <= harmonics && k * mains < sampleRate / 2.0; k++)
        {
            sections.Add(Biquad.Notch(k * mains, q, sampleRate));
        }

        return [.. sections];
    }
}

// "eq" (live): the RBJ cookbook's peaking EQ: `db` dB at the value's frequency, the band as wide as `q` says, the rest
// left at its level.
internal sealed class PeakingProcessor : ILiveAudioProcessor
{
    public const double DefaultQ = 1;

    public string Name => FilterProcessors.Eq;

    public string Label => "Peaking EQ";

    public string Summary => "a band around the value's frequency, Hz, boosted or cut by `db` dB (required), its width `q` (default 1)";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        FilterDesign.Frequency("", "Frequency", "The middle of the band; below half the rate"),
        FilterDesign.Decibels,
        FilterDesign.Q("1", "The band's narrowness (1: about two octaves wide at half the gain; larger, narrower)") with { Max = 100 },
    ];

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        FilterDesign.Require(step, "db", "DECIBELS");
    }

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double hertz = step.Number();
        FilterDesign.ThrowIfNotBelowHalf(Name, "frequency", hertz, sampleRate);
        return [Biquad.Peaking(hertz, step.Number("db", 0), step.Number("q", DefaultQ), sampleRate)];
    }
}

// "lowshelf" / "highshelf" (live): the RBJ cookbook's shelves: the sound below (low) or above (high) the value's frequency
// boosted or cut by `db` dB, the turn as steep as `slope` (1: the steepest without a bump) or, when given instead, `q`.
internal sealed class ShelfProcessor(bool high) : ILiveAudioProcessor
{
    public const double DefaultSlope = 1;

    public string Name => high ? FilterProcessors.HighShelf : FilterProcessors.LowShelf;

    public string Label => high ? "High shelf" : "Low shelf";

    public string Summary => $"the sound {(high ? "above" : "below")} the value's frequency, Hz, boosted or cut by `db` dB (required), the turn as steep as `slope` (default 1) or `q`";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        FilterDesign.Frequency("", "Frequency", "Where the shelf turns (its midpoint, at half the gain); below half the rate"),
        FilterDesign.Decibels,
        new("slope", "Slope", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1, Step = 0.1, Optional = true, Default = "1", Help = "How steep the turn is (1: the steepest without a bump past it)",
        },
        FilterDesign.Q(null, "The turn's sharpness, in place of slope (above 0.71 it bumps past the turn); left out, slope sets it") with { Max = 100 },
    ];

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        FilterDesign.Require(step, "db", "DECIBELS");
        if (step.Option("slope") is not null && step.Option("q") is not null)
        {
            throw new ArgumentException($"{Name} takes slope or q, not both (given '{step}').");
        }
    }

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double hertz = step.Number(), decibels = step.Number("db", 0), a = Math.Pow(10, decibels / 40);
        FilterDesign.ThrowIfNotBelowHalf(Name, "frequency", hertz, sampleRate);
        double alphaOverSin = step.Option("q") is not null
            ? 1 / (2 * step.Number("q", 1))
            : Math.Sqrt((a + 1 / a) * (1 / step.Number("slope", DefaultSlope) - 1) + 2) / 2;
        return [Biquad.Shelf(high, hertz, decibels, alphaOverSin, sampleRate)];
    }
}

// "presence" (live): the speech clarity control (its "sharpness"): a wide peaking boost (as "eq") around 3 kHz, where the
// consonants that tell words apart are; a cut (below 0) softens a harsh voice instead.
internal sealed class PresenceProcessor : ILiveAudioProcessor
{
    public const double DefaultGain = 3, DefaultHertz = 3000, DefaultQ = 0.7;

    public string Name => FilterProcessors.Presence;

    public string Label => "Presence (speech clarity)";

    public string Summary => $"the speech clarity control: the consonants' band around `frequency` (default {DefaultHertz} Hz) boosted by the value, dB (default {DefaultGain}; below 0 softens)";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Clarity", AudioProcessorInput.Number)
        {
            Min = -24, Max = 24, Step = 0.5, Unit = "dB", Optional = true, Default = "3", Help = "The boost of the consonants' band (above 0 clearer, below 0 softer)",
        },
        FilterDesign.Frequency("frequency", "Frequency", "The middle of the band (speech's consonants: 2 to 5 kHz); below half the rate") with { Optional = true, Default = "3000", Step = 100 },
        FilterDesign.Q("0.7", "The band's narrowness (0.7: about two octaves, 1.5 to 6 kHz at 3 kHz)") with { Max = 100 },
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double hertz = step.Number("frequency", DefaultHertz);
        FilterDesign.ThrowIfNotBelowHalf(Name, "frequency", hertz, sampleRate);
        return [Biquad.Peaking(hertz, step.Number(DefaultGain), step.Number("q", DefaultQ), sampleRate)];
    }
}

// "preemphasis" (live): y[n] = x[n] - value·x[n-1] (the speech front ends' first step: the highs lifted against the
// lows), started settled on the first sample's level (x[-1] = x[0], so y[0] = (1 - value)·x[0]; torchaudio's starts
// from x[-1] = 0).
internal sealed class PreEmphasisProcessor : ILiveAudioProcessor
{
    public const double DefaultCoefficient = 0.97;

    public string Name => FilterProcessors.PreEmphasis;

    public string Label => "Pre-emphasis";

    public string Summary => $"y[n] = x[n] - value·x[n-1] (default {DefaultCoefficient}): the highs lifted against the lows, as speech front ends do";

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => BiquadStage.Checked(this, step, sampleRate, channels, Design);

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Coefficient", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1, Step = 0.01, Optional = true, Default = "0.97", Help = "How much of the sample before is taken off (0 leaves the sound; 1 is the difference)",
        },
    ];

    private Biquad[] Design(AudioProcessorStep step, int sampleRate)
    {
        double a = step.Number(DefaultCoefficient);
        return a == 0 ? [] : [new Biquad(1, -a, 0, 0, 0, 1 - a)];
    }
}

// "mono" (live): the channels' mean, sample by sample (float sums, SIMD with a scalar tail): one channel out. Mono sound
// passes as it is; otherwise each block's output is a new block of one channel (its size is not the input's).
internal sealed class MonoProcessor : ILiveAudioProcessor
{
    public string Name => FilterProcessors.Mono;

    public string Label => "Mix to mono";

    public string Summary => "the channels mixed to one (their mean)";

    public IReadOnlyList<AudioProcessorParameter> Parameters => [];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return new Stage(sampleRate, channels);
    }

    private sealed class Stage(int sampleRate, int channels) : IAudioStage
    {
        private readonly AudioData _empty = new([], 1, sampleRate);

        public int SampleRate => sampleRate;

        public int Channels => 1;

        public AudioData Process(AudioData block)
        {
            ArgumentNullException.ThrowIfNull(block);
            int length = block.Length;
            if (channels == 1)
            {
                return block;
            }

            if (length == 0)
            {
                return _empty;
            }

            var samples = block.Samples;
            var mean = GC.AllocateUninitializedArray<float>(length);
            samples.AsSpan(0, length).CopyTo(mean);
            for (int c = 1; c < channels; c++)
            {
                Add(samples.AsSpan(c * length, length), mean);
            }

            CleaningLevels.Scale(mean, 1f / channels, mean);
            return new AudioData(mean, 1, sampleRate);
        }

        public AudioData Flush() => _empty;

        private static void Add(ReadOnlySpan<float> x, Span<float> sum)
        {
            int i = 0, width = Vector<float>.Count;
            if (Vector.IsHardwareAccelerated)
            {
                for (; i + width <= x.Length; i += width)
                {
                    (new Vector<float>(x.Slice(i, width)) + new Vector<float>(sum.Slice(i, width))).CopyTo(sum.Slice(i, width));
                }
            }

            for (; i < x.Length; i++)
            {
                sum[i] += x[i];
            }
        }
    }
}

// "resample" (live): the sound at the value's rate, by the library's resampler (Resampler: torchaudio's windowed-sinc,
// sinc_interp_hann, `width` zero crossings and `rolloff` of the lower rate's Nyquist) run as it comes: the sound is kept
// from the next output frame's first tap, each frame (one output sample per phase) is computed once its last tap has
// come, and the flush adds the zeros the resampler pads with and gives the last frames, cut to its length. So blocks give
// Resampler.Resample's samples bit for bit (the same taps over the same floats); Apply is Resampler.Resample itself. The
// output trails the input by at most Delay samples (the kernel's half width plus one frame); the samples are not
// shifted (Latency 0). Each block's output is a new block (its size is not the input's); a large block's frames are spread over the
// cores, a live block's stay on its thread.
internal sealed class ResampleProcessor : ILiveAudioProcessor
{
    public const int DefaultWidth = 6;
    public const double DefaultRolloff = 0.99;

    public string Name => FilterProcessors.Resample;

    public string Label => "Resample";

    public string Summary => string.Create(CultureInfo.InvariantCulture, $"the sound at the value's rate, Hz (windowed sinc: `width` zero crossings, default {DefaultWidth}; `rolloff` of the lower Nyquist, default {DefaultRolloff})");

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Rate", AudioProcessorInput.Integer) { Min = 1, Max = 1_000_000, Step = 1000, Unit = "Hz", Help = "The rate the sound is given at" },
        new("width", "Width", AudioProcessorInput.Integer)
        {
            Min = 1, Max = 64, Step = 1, Optional = true, Default = "6", Help = "The filter's zero crossings on each side: wider is sharper and slower, with more latency",
        },
        new("rolloff", "Roll-off", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1, Step = 0.01, Optional = true, Default = "0.99", Help = "The cutoff as a share of the lower rate's half (less: less aliasing, a duller top)",
        },
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        var (rate, options) = Read(step);
        return rate == sampleRate ? new Same(sampleRate, channels) : new Stage(Resampler.KernelFor(sampleRate, rate, options), rate, channels);
    }

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var (rate, options) = Read(step);
        return Resampler.Resample(audio, rate, options);
    }

    private (int Rate, ResampleOptions Options) Read(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.Validate(Parameters);
        return (step.Integer(), new ResampleOptions { Width = step.Integer("width", DefaultWidth), Rolloff = step.Number("rolloff", DefaultRolloff) });
    }

    // At the rate it is already: the sound as it is.
    private sealed class Same(int sampleRate, int channels) : IAudioStage
    {
        public int SampleRate => sampleRate;

        public int Channels => channels;

        public AudioData Process(AudioData block) => block;

        public AudioData Flush() => new([], channels, sampleRate);
    }

    private sealed class Stage : IAudioStage
    {
        private const long ParallelWork = 1 << 18;                              // multiply-adds a piece needs before its frames are spread over the cores

        private readonly Resampler.Kernel _kernel;
        private readonly int _capacity;
        private readonly float[] _buffer;                                       // per channel: the padded sound from the next frame's first tap
        private readonly AudioData _empty;
        private int _filled;                                                    // samples held per channel
        private long _received, _given;                                         // input samples taken, output samples given

        public Stage(Resampler.Kernel kernel, int rate, int channels)
        {
            (_kernel, SampleRate, Channels) = (kernel, rate, channels);
            _capacity = kernel.Taps + kernel.From * Math.Max(64, 4096 / kernel.From);   // a block is taken in pieces that fit (64 frames or more)
            _buffer = new float[channels * _capacity];
            _filled = kernel.PadLeft;                                           // the resampler's zeros before the sound
            _empty = new AudioData([], channels, rate);
            Delay = (int)((((long)kernel.PadLeft + kernel.From - 1) * kernel.To + kernel.From - 1) / kernel.From);   // ⌈(half + from - 1)·to / from⌉, the most held back
        }

        public int SampleRate { get; }

        public int Channels { get; }

        public int Delay { get; }

        public AudioData Process(AudioData block)
        {
            ArgumentNullException.ThrowIfNull(block);
            int length = block.Length;
            if (length == 0)
            {
                return _empty;
            }

            // Every frame whose last tap has come is whole and inside the recording's output, whatever comes after.
            long before = Frames(_received);
            _received += length;
            int count = checked((int)((Frames(_received) - before) * _kernel.Phases));
            var output = count == 0 ? [] : GC.AllocateUninitializedArray<float>(Channels * count);
            Feed(block.Samples, length, output, count);
            return count == 0 ? _empty : new AudioData(output, Channels, SampleRate);
        }

        public AudioData Flush()
        {
            int count = (int)(Resampler.OutputLength(_received, _kernel.From, _kernel.To) - _given);
            if (count <= 0)
            {
                return _empty;
            }

            var output = GC.AllocateUninitializedArray<float>(Channels * count);
            Feed(null, _kernel.PadRight, output, count);                         // the resampler's zeros after the sound
            return new AudioData(output, Channels, SampleRate);
        }

        // The frames whose last tap is among the first PadLeft + received samples of the padded sound.
        private long Frames(long received)
        {
            long end = _kernel.PadLeft + received;
            return end < _kernel.Taps ? 0 : (end - _kernel.Taps) / _kernel.From + 1;
        }

        // `length` samples of each channel of `samples` (zeros when null) through the kernel in pieces that fit the buffer;
        // the frames that become whole are written to `output` (each channel `count` samples), no more than `count`.
        private void Feed(float[]? samples, int length, float[] output, int count)
        {
            int taps = _kernel.Taps, from = _kernel.From, phases = _kernel.Phases, written = 0;
            for (int taken = 0; taken < length;)
            {
                int piece = Math.Min(length - taken, _capacity - _filled), filled = _filled + piece;
                int frames = filled < taps ? 0 : (filled - taps) / from + 1;
                for (int c = 0; c < Channels; c++)
                {
                    var buffer = _buffer.AsSpan(c * _capacity, _capacity);
                    if (samples is null)
                    {
                        buffer.Slice(_filled, piece).Clear();
                    }
                    else
                    {
                        samples.AsSpan(c * length + taken, piece).CopyTo(buffer[_filled..]);
                    }

                    if ((long)frames * phases * taps >= ParallelWork)
                    {
                        ComputeOnAllCores(c, output, count, written, frames);
                    }
                    else
                    {
                        Compute(c, output, count, written, 0, frames);
                    }

                    buffer[(frames * from)..filled].CopyTo(buffer);                  // the next frame's first tap to the front
                }

                _filled = filled - frames * from;
                written = Math.Min(count, written + frames * phases);
                taken += piece;
            }

            _given += written;
        }

        // Frames first to last of channel c's buffer, each phase's dot product (Resampler's) written from `written` on,
        // no more than `count` in all.
        private void Compute(int c, float[] output, int count, int written, int first, int last)
        {
            int taps = _kernel.Taps, from = _kernel.From, phases = _kernel.Phases;
            var values = _kernel.Values;
            var buffer = _buffer.AsSpan(c * _capacity, _capacity);
            var target = output.AsSpan(c * count, count);
            for (int f = first; f < last; f++)
            {
                var window = buffer.Slice(f * from, taps);
                for (int p = 0, at = written + f * phases; p < phases && at < count; p++, at++)
                {
                    target[at] = Resampler.Dot(values.AsSpan(p * taps, taps), window);
                }
            }
        }

        // A large piece's frames spread over the cores (ComputeResources bounds them; rule 79): each output is the same
        // dot product, so the samples do not change. Its own method, so a small piece makes no closure.
        private void ComputeOnAllCores(int c, float[] output, int count, int written, int frames)
        {
            int grain = (int)Math.Max(1, ParallelWork / ((long)_kernel.Phases * _kernel.Taps));
            HostParallel.For(frames, grain, (first, last) => Compute(c, output, count, written, first, last));
        }
    }
}
