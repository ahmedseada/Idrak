// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Audio.Plugins.RnNoise;

/// <summary>
/// "rnnoise" (live): xiph's RNNoise (its classic recurrent network, ported in <see cref="RnNoiseDenoiser"/>) as an audio
/// processor. RNNoise works on 48 kHz in frames of 480 samples: sound at another rate is resampled to 48 kHz and back
/// inside the stage by the library's own resample processor (<c>convert=false</c> refuses it instead), and each channel
/// is denoised alone. The output lags the input by <see cref="RnNoiseStream.Delay"/> samples at 48 kHz (a frame to
/// fill and RNNoise's own frame of delay): silence first, the last of the sound given by the flush. That shift is the
/// stage's latency, which a recording's Apply takes off, and blocks of any size give the whole recording's samples. The model is read
/// from RNNoise's model file (<see cref="RnNoisePlugin.LocateModel"/>), once per process.
/// </summary>
internal sealed class RnNoiseProcessor : ILiveAudioProcessor
{
    private const int Rate = 48000;

    public string Name => RnNoisePlugin.ProcessorName;

    public string Label => "RNNoise denoiser";

    public string Summary => "RNNoise's learned denoiser (xiph's recurrent network on 48 kHz frames of 10 ms; another rate is resampled to 48 kHz and back): "
        + "`mix` of the denoised sound against the original (default 1), `attenuation` the most a band is turned down, dB (default none); "
        + $"`model=PATH` its model file (default {RnNoisePlugin.ModelVariable}, else rnnoise/rnnoise.rnnn in the cache); the output lags by 959 samples at 48 kHz (about 20 ms)";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("mix", "Mix", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1, Step = 0.05, Unit = "fraction", Optional = true, Default = "1",
            Help = "How much of the denoised sound is heard: 1 the denoised sound alone, 0 the original (delayed as the output is)",
        },
        new("attenuation", "Attenuation limit", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 120, Step = 1, Unit = "dB", Optional = true,
            Help = "The most a band is turned down (its gain kept at 10^(-dB/20) or more), so some of the noise stays and the speech keeps its body; left out: RNNoise's own gains",
        },
        new("convert", "Convert the rate", AudioProcessorInput.Switch)
        {
            Optional = true, Default = "true",
            Help = "Sound at another rate than 48 kHz is resampled to 48 kHz and back inside the stage (the library's resample processor); false refuses it",
        },
    ];

    // `model` takes a path, which no parameter kind describes: it is a key, checked here, and named in the summary.
    public IReadOnlySet<string> Keys => field ??= new OptionKeys([.. Parameters.Where(p => p.Key.Length > 0).Select(p => p.Key), RnNoisePlugin.ModelOption]);

    public void Check(AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        step.ThrowIfUnknownOptions(Keys);
        var rest = new AudioProcessorStep(step.Name, step.Value, step.Options.Where(o => !o.Key.Equals(RnNoisePlugin.ModelOption, StringComparison.OrdinalIgnoreCase)));
        rest.Validate(Parameters);
    }

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        Check(step);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        bool convert = step.Flag("convert", true);
        if (sampleRate != Rate && !convert)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"rnnoise runs at {Rate} Hz and convert=false refuses sound at {sampleRate} Hz: leave convert out (the stage resamples), or put resample={Rate} before it."));
        }

        var model = RnNoiseModel.Load(RnNoisePlugin.LocateModel(step.Option(RnNoisePlugin.ModelOption)));
        float mix = (float)step.Number("mix", 1);
        float floor = step.Option("attenuation") is null ? 0 : (float)Math.Pow(10, -step.Number("attenuation", 0) / 20);
        var core = new RnNoiseStream(model, channels, mix, floor);
        return sampleRate == Rate ? new Stage(core, channels) : new ResampledStage(core, sampleRate, channels);
    }

    private static void ThrowIfOther(AudioData block, int rate, int channels)
    {
        ArgumentNullException.ThrowIfNull(block);
        if (block.SampleRate != rate || block.Channels != channels)
        {
            throw new ArgumentException($"rnnoise was opened for {channels} channel(s) at {rate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
        }
    }

    // a then b (the same rate and channels), planar; either alone when the other is empty.
    private static AudioData Concat(AudioData a, AudioData b)
    {
        if (b.Length == 0)
        {
            return a;
        }

        if (a.Length == 0)
        {
            return b;
        }

        int channels = a.Channels, length = a.Length + b.Length;
        var samples = GC.AllocateUninitializedArray<float>(channels * length);
        for (int c = 0; c < channels; c++)
        {
            a.Channel(c).CopyTo(samples.AsSpan(c * length, a.Length));
            b.Channel(c).CopyTo(samples.AsSpan(c * length + a.Length, b.Length));
        }

        return new AudioData(samples, channels, a.SampleRate);
    }

    // At 48 kHz: the blocks are denoised in place (nothing allocated per block); the flush gives the last Delay samples.
    private sealed class Stage(RnNoiseStream core, int channels) : IAudioStage
    {
        private readonly AudioData _empty = new([], channels, Rate);

        public int SampleRate => Rate;

        public int Channels => channels;

        public int Latency => RnNoiseStream.Delay;

        public AudioData Process(AudioData block)
        {
            ThrowIfOther(block, Rate, channels);
            if (block.Length == 0)
            {
                return _empty;
            }

            core.Process(block.Samples, block.Length);
            return block;
        }

        public AudioData Flush()
        {
            var tail = new float[channels * RnNoiseStream.Delay];
            core.Flush(tail);
            return new AudioData(tail, channels, Rate);
        }
    }

    // At another rate: the library's resample stages to 48 kHz and back around the denoiser (each gives its own blocks).
    // The resamplers do not shift the sound, only hold it back (their Delay); the denoiser shifts it by Delay samples at
    // 48 kHz, which is a whole number of samples at the stage's rate only when it is a multiple of 48000 / gcd(48000,
    // rate). So the shift is padded up to the next such multiple with silence fed first (one sample at 16 kHz: 960, a
    // shift of exactly 320), and the stage's Latency is that shift at its rate: Apply takes it off exactly. A rate whose
    // multiple is over four times the delay is not padded and its Latency is the shift rounded (off by under a sample).
    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return a;
    }

    private sealed class ResampledStage : IAudioStage
    {
        private readonly RnNoiseStream _core;
        private readonly IAudioStage _up, _down;
        private readonly AudioData _empty;
        private readonly float[] _lead;                                         // the silence fed first: the shift padded to whole samples
        private readonly int _leadLength;
        private bool _started;

        public ResampledStage(RnNoiseStream core, int rate, int channels)
        {
            var resample = AudioProcessors.Default("resample")
                ?? throw new InvalidOperationException("rnnoise resamples with the library's resample processor, which is not registered.");
            _core = core;
            (SampleRate, Channels) = (rate, channels);
            _up = resample.Open(new AudioProcessorStep("resample", Rate.ToString(CultureInfo.InvariantCulture)), rate, channels);
            try
            {
                _down = resample.Open(new AudioProcessorStep("resample", rate.ToString(CultureInfo.InvariantCulture)), Rate, channels);
            }
            catch
            {
                _up.Dispose();
                throw;
            }

            _empty = new AudioData([], channels, rate);
            int whole = Rate / Gcd(Rate, rate);                                 // 48 kHz samples that are a whole number at the rate
            int shift = whole <= 4 * RnNoiseStream.Delay ? (RnNoiseStream.Delay + whole - 1) / whole * whole : RnNoiseStream.Delay;
            _leadLength = shift - RnNoiseStream.Delay;
            _lead = new float[channels * _leadLength];
            Latency = (int)Math.Round((double)shift * rate / Rate);
            Delay = Latency + (int)Math.Ceiling((double)_up.Delay * rate / Rate) + _down.Delay;
        }

        public int SampleRate { get; }

        public int Channels { get; }

        public int Latency { get; }

        public int Delay { get; }

        public AudioData Process(AudioData block)
        {
            ThrowIfOther(block, SampleRate, Channels);
            if (block.Length == 0)
            {
                return _empty;
            }

            var lead = Lead();
            var high = _up.Process(block);
            if (high.Length == 0)
            {
                return lead.Length > 0 ? lead : _empty;
            }

            _core.Process(high.Samples, high.Length);
            var low = Concat(lead, _down.Process(high));
            return low.Length > 0 ? low : _empty;
        }

        // Once, before the first sound: the padding silence through the denoiser (it gives silence back, its own delay
        // being longer) and down to the stage's rate.
        private AudioData Lead()
        {
            if (_started || _leadLength == 0)
            {
                _started = true;
                return _empty;
            }

            _started = true;
            Array.Clear(_lead);
            _core.Process(_lead, _leadLength);
            return _down.Process(new AudioData(_lead, Channels, Rate));
        }

        public AudioData Flush()
        {
            if (!_started)
            {
                return _empty;
            }

            var high = _up.Flush();
            if (high.Length > 0)
            {
                _core.Process(high.Samples, high.Length);
            }

            var tail = new float[Channels * RnNoiseStream.Delay];
            _core.Flush(tail);
            var low = _down.Process(Concat(high, new AudioData(tail, Channels, Rate)));
            var rest = Concat(low, _down.Flush());
            return rest.Length > 0 ? rest : _empty;
        }

        public void Dispose()
        {
            _up.Dispose();
            _down.Dispose();
        }
    }
}

/// <summary>
/// RNNoise over planar 48 kHz blocks of every channel, in place: each channel's sound is gathered into frames of 480,
/// each frame denoised (in 16-bit scale: times 32768 in, divided out) and its output, the frame before it (RNNoise's
/// first output, from before the sound, is left out), queued; every sample taken gives back the one
/// <see cref="Delay"/> samples before it (silence first). Mixed with the original when <c>mix</c> is under 1. Nothing
/// is allocated per block; a long block's channels run on separate cores.
/// </summary>
internal sealed class RnNoiseStream
{
    /// <summary>
    /// The samples (at 48 kHz) the output lags the input: the denoised frame of samples n to n + 479 is ready once sample
    /// n + 959 has come (RNNoise needs the next frame), so a sample taken gives back the one 959 samples before it.
    /// </summary>
    public const int Delay = 2 * RnNoiseDenoiser.FrameSize - 1;

    private const int ParallelLength = 4800;                                     // samples a channel before the channels are spread over the cores (0.1 s)
    private readonly Channel[] _channels;

    public RnNoiseStream(RnNoiseModel model, int channels, float mix, float floor)
    {
        _channels = new Channel[channels];
        for (int c = 0; c < channels; c++)
        {
            _channels[c] = new Channel(new RnNoiseDenoiser(model, floor), mix);
        }
    }

    /// <summary>Denoises <paramref name="length"/> samples of each channel of <paramref name="samples"/> (planar) in place.</summary>
    public void Process(float[] samples, int length)
    {
        if (_channels.Length > 1 && length >= ParallelLength)
        {
            ProcessOnAllCores(samples, length);
            return;
        }

        for (int c = 0; c < _channels.Length; c++)
        {
            _channels[c].Feed(samples.AsSpan(c * length, length));
        }
    }

    /// <summary>The last <see cref="Delay"/> samples of each channel (planar into <paramref name="tail"/>): the sound's end, from frames padded with silence.</summary>
    public void Flush(float[] tail)
    {
        for (int c = 0; c < _channels.Length; c++)
        {
            _channels[c].Flush(tail.AsSpan(c * Delay, Delay));
        }
    }

    // Its own method, so a short block makes no closure (rule 79: the channels are independent).
    private void ProcessOnAllCores(float[] samples, int length) =>
        HostParallel.For(_channels.Length, 1, (first, last) =>
        {
            for (int c = first; c < last; c++)
            {
                _channels[c].Feed(samples.AsSpan(c * length, length));
            }
        });

    private sealed class Channel(RnNoiseDenoiser denoiser, float mix)
    {
        private const int Frame = RnNoiseDenoiser.FrameSize;
        private const float Scale = 32768f, Unscale = 1f / 32768f;
        private readonly float[] _frame = new float[Frame];                     // the frame being gathered
        private readonly float[] _dry = mix < 1 ? new float[Frame] : [];        // the previous frame as it came, for the mix
        private readonly float[] _work = new float[Frame];
        private readonly float[] _queue = new float[Delay + 2 * Frame];         // the output to give: Delay samples of silence first
        private int _filled, _head, _count = Delay;
        private long _frames, _received;

        public void Feed(Span<float> x)
        {
            for (int at = 0; at < x.Length;)
            {
                int take = Math.Min(x.Length - at, Frame - _filled);
                var piece = x.Slice(at, take);
                piece.CopyTo(_frame.AsSpan(_filled));
                _filled += take;
                if (_filled == Frame)
                {
                    Compute();
                }

                Take(piece);
                at += take;
            }

            _received += x.Length;
        }

        // Frames of silence after the sound until every sample of it has its output, then the last Delay samples.
        public void Flush(Span<float> tail)
        {
            long needed = (_received + Frame - 1) / Frame + 1;
            while (_frames < needed)
            {
                _frame.AsSpan(_filled).Clear();
                Compute();
            }

            Take(tail);
        }

        private void Compute()
        {
            var work = _work.AsSpan();
            var frame = _frame.AsSpan();
            for (int i = 0; i < Frame; i++)
            {
                work[i] = frame[i] * Scale;
            }

            denoiser.ProcessFrame(work, work);
            if (_frames > 0)
            {
                for (int i = 0; i < Frame; i++)
                {
                    work[i] *= Unscale;
                }

                if (mix < 1)
                {
                    float wet = mix, dry = 1 - mix;
                    for (int i = 0; i < Frame; i++)
                    {
                        work[i] = wet * work[i] + dry * _dry[i];
                    }
                }

                Put(work);
            }

            if (mix < 1)
            {
                frame.CopyTo(_dry);
            }

            _frames++;
            _filled = 0;
        }

        private void Put(ReadOnlySpan<float> values)
        {
            int end = (_head + _count) % _queue.Length, first = Math.Min(values.Length, _queue.Length - end);
            values[..first].CopyTo(_queue.AsSpan(end));
            values[first..].CopyTo(_queue);
            _count += values.Length;
        }

        private void Take(Span<float> into)
        {
            int first = Math.Min(into.Length, _queue.Length - _head);
            _queue.AsSpan(_head, first).CopyTo(into);
            _queue.AsSpan(0, into.Length - first).CopyTo(into[first..]);
            _head = (_head + into.Length) % _queue.Length;
            _count -= into.Length;
        }
    }
}
