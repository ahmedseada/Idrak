// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P0: the audio processor contract (IAudioProcessor, IAudioStage), its registry (AudioProcessors) and the
// pipeline (AudioPipeline: text, JSON, checks, Apply, blocks, live inputs); the four cleaning stages as processors, the
// same samples as before bit for bit.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioProcessingGroup =
    [
        ("audio processing: the registry names highpass (live), denoise, trim and normalize (whole recording) as the library's; an unknown name is refused naming the registered ones and AudioProcessors.Register, a name with a comma, '=' or a space is refused; an app's processors, one implementing only Open (live) and one only Apply (whole recording), register, run in a pipeline and unregister", AudioProcessingRegistry),
        ("audio processing: pipeline text: values and space-separated options round-trip through text and JSON (strings and objects), an option after a comma joins the step before it, none or empty is the empty pipeline, Then and Contains; refused naming the cause: an unknown processor or option, a value that is not a number, one out of range (with its range and unit), a value to a processor that takes none", AudioProcessingText),
        ("audio processing: a pipeline run block by block (blocks of 1, 7, 160 and 4,096 samples, mono and stereo, a stage with latency, whole-recording steps flushed at the end) gives Apply's samples bit for bit once its latency is taken off; Apply keeps the length and leaves its input unchanged; a live pipeline wraps an IAudioInput and gives the same samples; Wrap refuses a whole-recording step naming it; a block of another rate is refused", AudioProcessingBlocks),
        ("audio processing: the cleaners run through their processors give the samples of the cleaners before plan 31 bit for bit (each stage and the chain highpass, denoise, trim, normalize, mono and stereo, a tone in bursts under noise with a DC offset), through AudioCleaners and through the pipeline text", AudioProcessingCleanersUnchanged),
    ];

    private static void AudioProcessingRegistry(Device device)
    {
        _ = device;
        string[] library = [AudioProcessors.HighPass, AudioProcessors.Denoise, AudioProcessors.Trim, AudioProcessors.Normalize];
        Check(library.All(n => AudioProcessors.Names.Contains(n) && AudioProcessors.Origin(n) == Overrides.Library && AudioProcessors.Default(n) is not null),
            $"the library's processors: {string.Join(", ", AudioProcessors.Names)}");
        Check(AudioProcessors.Get("HIGHPASS").Live && !AudioProcessors.Get("denoise").Live && !AudioProcessors.Get("trim").Live && !AudioProcessors.Get("normalize").Live,
            "highpass is live, the other three need the whole recording");
        Check(AudioProcessors.Find("vacuum") is null, "an unknown name is null");
        ThrowsNaming<ArgumentException>(() => AudioProcessors.Get("vacuum"), "an unregistered processor", "vacuum", "highpass", "normalize", "AudioProcessors.Register");
        Check(AudioProcessors.Describe().Contains("trim: ", StringComparison.Ordinal) && AudioProcessors.Describe().Contains("(whole recording)", StringComparison.Ordinal),
            $"Describe lists each with its summary: {AudioProcessors.Describe()}");
        Check(AudioProcessors.Get("denoise").Keys.SequenceEqual(["quantile", "floor", "window"]), $"keys follow the parameters: {string.Join(", ", AudioProcessors.Get("denoise").Keys)}");
        foreach (string bad in new[] { "a,b", "a=b", "a b" })
        {
            ThrowsNaming<ArgumentException>(() => AudioProcessors.Register(new ProcessingGain(bad)), $"the name '{bad}'", bad);
        }

        AudioProcessors.Register(new ProcessingGain());
        AudioProcessors.Register(new ProcessingReverse());
        try
        {
            Check(AudioProcessors.Origin(ProcessingGain.Id) is { } origin && origin != Overrides.Library, "an app's processor is the app's");
            var tone = ProcessingTone(16000, 0.25, 1);
            var pipeline = AudioPipeline.Parse($"{ProcessingGain.Id}=-6.0206, {ProcessingReverse.Id}");
            Check(!pipeline.Live, "a whole-recording step makes the pipeline not live");
            var output = pipeline.Apply(tone);
            int n = tone.Length;
            float half = (float)Math.Pow(10, -6.0206 / 20);
            Check(output.Length == n && Enumerable.Range(0, n).All(i => output.Samples[i] == tone.Samples[n - 1 - i] * half),
                "a processor with only Open (gain) then one with only Apply (reverse) run in a pipeline");
            Check(AudioProcessors.Get(ProcessingReverse.Id).Open(new AudioProcessorStep(ProcessingReverse.Id), 16000, 1) is { SampleRate: 16000, Channels: 1, Latency: 0 },
                "a whole-recording processor's default stage keeps the rate and channels");
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse(ProcessingGain.Id), "the gain without its value", ProcessingGain.Id, "needs a value");
        }
        finally
        {
            Check(AudioProcessors.Unregister(ProcessingGain.Id) && AudioProcessors.Unregister(ProcessingReverse.Id), "the app's processors unregister");
        }

        Check(AudioProcessors.Find(ProcessingGain.Id) is null && !AudioProcessors.Unregister(AudioProcessors.Trim), "gone after Unregister; the library's cannot be unregistered");
    }

    private static void AudioProcessingText(Device device)
    {
        _ = device;
        var pipeline = AudioPipeline.Parse("HighPass=80, denoise=2 Quantile=0.2 floor=-30, trim, silence=30, normalize=-18 peak=0.9");
        Check(pipeline.ToString() == "highpass=80, denoise=2 floor=-30 quantile=0.2, trim silence=30, normalize=-18 peak=0.9", $"the text: {pipeline}");
        Check(pipeline.Steps.Count == 4 && pipeline.Steps[2].Option("silence") == "30", "an option after a comma joins the step before it");
        Check(AudioPipeline.Parse(pipeline.ToString()).Equals(pipeline), "the text reads back the same pipeline");
        Check(AudioPipeline.FromJson(pipeline.ToJson()).Equals(pipeline), "its JSON reads back the same pipeline");
        var json = JsonNode.Parse("""["highpass=80", {"name": "denoise", "value": 2, "quantile": 0.2, "floor": -30}, "trim silence=30", {"name": "normalize", "value": "-18", "peak": 0.9}]""");
        Check(AudioPipeline.FromJson(json).Equals(pipeline), $"JSON strings and objects: {AudioPipeline.FromJson(json)}");
        Check(AudioPipeline.Parse(null).IsEmpty && AudioPipeline.Parse(" none ").IsEmpty && AudioPipeline.Parse(" , ").IsEmpty && AudioPipeline.FromJson(null).IsEmpty
              && AudioPipeline.Empty.ToJson() is null && AudioPipeline.Empty.Live, "none, empty and null are the empty pipeline (live)");
        var first = AudioPipeline.Parse("highpass");
        Check(first.Then(AudioPipeline.Parse("normalize")).ToString() == "highpass, normalize" && first.Then(null) == first && AudioPipeline.Empty.Then(first) == first,
            "Then joins two pipelines");
        Check(pipeline.Contains("TRIM") && !pipeline.Contains("gate"), "Contains ignores case");
        Check(first.Steps[0].Number(HighPassProcessor.DefaultHertz) == 60 && pipeline.Steps[1].Number("quantile", 0) == 0.2 && pipeline.Steps[1].Number("window", 0.5) == 0.5,
            "the step reads its value and options, or the default");

        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass, vacuum"), "an unknown processor", "vacuum", "registered", "denoise");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("trim, peak=0.9"), "an option of another processor after a comma", "peak", "silence, hop");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise depth=2"), "an unknown option", "denoise", "depth", "quantile, floor, window");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise quantile"), "an option without a value", "KEY=VALUE", "quantile");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass=low"), "a value that is not a number", "highpass", "number", "low");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass=-5"), "a negative cutoff", "highpass", "at least 0 Hz", "-5");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise quantile=2"), "a share over 1", "denoise quantile", "0 to 1", "2");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("normalize=-20 peak=0"), "a peak ceiling of 0", "normalize peak", "above 0 to 1");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("normalize=3"), "a target above full scale", "normalize", "at most 0 dBFS");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.FromJson(JsonNode.Parse("""[{"value": 3}]""")), "an object without a name", "\"name\"");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.FromJson(JsonNode.Parse("3")), "a number for a pipeline", "string or an array");
        AudioProcessors.Register(new ProcessingReverse());
        try
        {
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse($"{ProcessingReverse.Id}=2"), "a value to a processor that takes none", ProcessingReverse.Id, "takes no value", "2");
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse($"{ProcessingReverse.Id} fast=1"), "an option to a processor that takes none", "takes no options", "fast");
        }
        finally
        {
            AudioProcessors.Unregister(ProcessingReverse.Id);
        }
    }

    private static void AudioProcessingBlocks(Device device)
    {
        _ = device;
        AudioProcessors.Register(new ProcessingGain());
        AudioProcessors.Register(new ProcessingDelay());
        try
        {
            foreach (int channels in new[] { 1, 2 })
            {
                var sound = ProcessingTone(16000, 0.6, channels, noise: 0.05f, dc: 0.1f);
                var before = sound.Samples.AsSpan().ToArray();
                foreach (string text in new[] { "highpass=120", $"{ProcessingDelay.Id}=37, highpass=80, {ProcessingGain.Id}=3", "highpass=80, denoise, trim, normalize=-18",
                             $"{ProcessingDelay.Id}=5, normalize, highpass=200, {ProcessingDelay.Id}=11" })
                {
                    var pipeline = AudioPipeline.Parse(text);
                    var whole = pipeline.Apply(sound);
                    Check(ProcessingSame(sound.Samples, before), $"{text}: Apply leaves its input unchanged");
                    Check(!ReferenceEquals(whole.Samples, sound.Samples), $"{text}: Apply gives new samples");
                    Check(text.Contains("trim", StringComparison.Ordinal) || whole.Length == sound.Length, $"{text}: Apply takes the look-ahead off, so the length is kept ({whole.Length} of {sound.Length})");
                    foreach (int size in new[] { 1, 7, 160, 4096 })
                    {
                        using var stage = pipeline.Open(sound.SampleRate, channels);
                        var pieces = new List<AudioData>();
                        foreach (var block in ProcessingBlocks(sound, size))
                        {
                            pieces.Add(stage.Process(block));
                        }

                        pieces.Add(stage.Flush());
                        var joined = ProcessingAligned(ProcessingJoin(pieces, stage.Channels, stage.SampleRate), stage.Latency);
                        Check(joined.Length == whole.Length && ProcessingSame(joined.Samples, whole.Samples),
                            $"{text}, {channels} channel(s), blocks of {size}: {joined.Length} samples (its latency {stage.Latency} off) against Apply's {whole.Length}, the same bits");
                    }
                }

                using (var delayed = AudioPipeline.Parse($"{ProcessingDelay.Id}=37, highpass, {ProcessingDelay.Id}=11").Open(16000, channels))
                {
                    Check(delayed.Latency == 48 && delayed.Delay == 48, $"the latencies add up: {delayed.Latency}, held back {delayed.Delay}");
                }

                var live = AudioPipeline.Parse($"highpass=100, {ProcessingGain.Id}=-3, {ProcessingDelay.Id}=9");
                Check(live.Live, "a pipeline of live steps is live");
                var input = new ProcessingInput(sound, 333);
                using var wrapped = live.Wrap(input);
                Check(wrapped.SampleRate == 16000 && wrapped.Channels == channels && wrapped.Description.Contains("cleaned by highpass=100", StringComparison.Ordinal),
                    $"the wrapped input: {wrapped.Description}");
                int latency;
                using (var liveStage = live.Open(16000, channels))
                {
                    latency = liveStage.Latency;
                }

                var heard = ProcessingAligned(ProcessingJoin([.. wrapped.ReadAsync().ToBlockingEnumerable()], channels, 16000), latency);
                Check(latency == 9 && ProcessingSame(heard.Samples, live.Apply(sound).Samples), "a wrapped live input gives the samples Apply gives, shifted by its latency");
                wrapped.Dispose();
                Check(input.Disposed, "disposing the wrapped input disposes the input");
            }

            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass, normalize, trim").Wrap(new ProcessingInput(ProcessingTone(16000, 0.1, 1), 100)),
                "a live pipeline with a whole-recording step", "normalize", "whole recording");
            Check(AudioPipeline.Empty.Wrap(new ProcessingInput(ProcessingTone(16000, 0.1, 1), 100)) is ProcessingInput, "the empty pipeline gives the input itself");
            using var opened = AudioPipeline.Parse("highpass").Open(16000, 1);
            ThrowsNaming<ArgumentException>(() => opened.Process(ProcessingTone(8000, 0.1, 1)), "a block of another rate", "16000 Hz", "8000 Hz");
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass=5000").Open(8000, 1), "a cutoff over half the rate", "5000", "half the rate");
        }
        finally
        {
            AudioProcessors.Unregister(ProcessingGain.Id);
            AudioProcessors.Unregister(ProcessingDelay.Id);
        }
    }

    private static void AudioProcessingCleanersUnchanged(Device device)
    {
        _ = device;
        var options = new AudioCleaningOptions { HighPassHertz = 75, TargetDecibels = -17.5, DenoiseStrength = 1.75, NoiseQuantile = 0.15, TrimPadSeconds = 0.07 };
        foreach (var settings in new[] { new AudioCleaningOptions(), options })
        {
            foreach (int channels in new[] { 1, 2 })
            {
                var sound = ProcessingTone(16000, 1.5, channels, noise: 0.02f, dc: 0.05f, bursts: true);
                foreach (string name in new[] { "highpass", "denoise", "trim", "normalize" })
                {
                    var now = AudioCleaners.Get(name).Clean(sound, settings);
                    var then = CleanersBefore31.Clean(name, sound, settings);
                    Check(now.Length == then.Length && now.Channels == then.Channels && ProcessingSame(now.Samples, then.Samples),
                        $"{name}, {channels} channel(s): {now.Length} samples, the cleaner's before plan 31 {then.Length}, the same bits");
                }

                string[] chain = ["highpass", "denoise", "trim", "normalize"];
                var expected = chain.Aggregate(sound, (a, n) => CleanersBefore31.Clean(n, a, settings));
                Check(ProcessingSame(AudioCleaners.Apply(sound, chain, settings).Samples, expected.Samples), $"the chain, {channels} channel(s), through AudioCleaners");
                var text = AudioPipeline.Parse($"highpass={settings.HighPassHertz:R}, denoise={settings.DenoiseStrength:R} quantile={settings.NoiseQuantile:R}, "
                                               + $"trim={settings.TrimPadSeconds:R}, normalize={settings.TargetDecibels:R}");
                Check(ProcessingSame(text.Apply(sound).Samples, expected.Samples), $"the chain, {channels} channel(s), through the pipeline {text}");
            }
        }
    }

    // A tone of 440 Hz at half scale, `channels` copies (the second a little quieter), optionally in bursts (0.2 s on,
    // 0.2 s off), with white noise and a DC offset.
    private static AudioData ProcessingTone(int rate, double seconds, int channels, float noise = 0, float dc = 0, bool bursts = false)
    {
        int length = (int)Math.Round(seconds * rate);
        var random = new Random(31);
        var samples = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < length; i++)
            {
                bool on = !bursts || (i / (rate / 5)) % 2 == 0;
                double tone = on ? 0.5 * Math.Sin(2 * Math.PI * 440 * i / rate) / (1 + c) : 0;
                samples[c * length + i] = (float)(tone + noise * (2 * random.NextDouble() - 1) + dc);
            }
        }

        return new AudioData(samples, channels, rate);
    }

    // The sound in blocks of `size` (copies: a block is the stage's).
    private static IEnumerable<AudioData> ProcessingBlocks(AudioData sound, int size)
    {
        for (int at = 0; at < sound.Length; at += size)
        {
            yield return sound.Slice(at, Math.Min(size, sound.Length - at));
        }
    }

    private static AudioData ProcessingJoin(IReadOnlyList<AudioData> pieces, int channels, int rate)
    {
        int length = pieces.Sum(p => p.Length);
        var samples = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            int at = c * length;
            foreach (var piece in pieces)
            {
                piece.Channel(c).CopyTo(samples.AsSpan(at, piece.Length));
                at += piece.Length;
            }
        }

        return new AudioData(samples, channels, rate);
    }

    // The sound with its first `latency` samples (a stage's shift) left out: aligned with what Apply gives.
    private static AudioData ProcessingAligned(AudioData sound, int latency) => latency == 0 ? sound : sound.Slice(latency, sound.Length - latency);

    // Planar samples [channels, n] with each channel's first `skip` left out.
    private static float[] ProcessingAligned(float[] planar, int channels, int skip)
    {
        int length = planar.Length / channels, kept = length - skip;
        var aligned = new float[channels * kept];
        for (int c = 0; c < channels; c++)
        {
            planar.AsSpan(c * length + skip, kept).CopyTo(aligned.AsSpan(c * kept));
        }

        return aligned;
    }

    // Bit for bit: the floats' bits, not their values (so -0 and 0 differ and NaN equals itself).
    private static bool ProcessingSame(float[] a, float[] b) => MemoryMarshal.Cast<float, int>(a.AsSpan()).SequenceEqual(MemoryMarshal.Cast<float, int>(b.AsSpan()));

    // An app's live processor implementing only Open: a gain in dB, in place.
    private sealed class ProcessingGain(string name = ProcessingGain.Id) : ILiveAudioProcessor
    {
        public const string Id = "test-gain";

        public string Name => name;

        public string Summary => "a gain in dB (a test's)";

        public IReadOnlyList<AudioProcessorParameter> Parameters { get; } = [new("", "Gain", AudioProcessorInput.Number) { Min = -60, Max = 24, Unit = "dB" }];

        public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage((float)Math.Pow(10, step.Number() / 20), sampleRate, channels);

        private sealed class Stage(float gain, int sampleRate, int channels) : IAudioStage
        {
            public int SampleRate => sampleRate;

            public int Channels => channels;

            public AudioData Process(AudioData block)
            {
                foreach (ref float x in block.Samples.AsSpan())
                {
                    x *= gain;
                }

                return block;
            }

            public AudioData Flush() => new([], channels, sampleRate);
        }
    }

    // An app's live processor with a latency: the sound delayed by the value's samples (silence first, the tail at the flush).
    private sealed class ProcessingDelay : ILiveAudioProcessor
    {
        public const string Id = "test-delay";

        public string Name => Id;

        public string Summary => "the sound delayed by N samples (a test's)";

        public IReadOnlyList<AudioProcessorParameter> Parameters { get; } = [new("", "Samples", AudioProcessorInput.Integer) { Min = 1 }];

        public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => new Stage(step.Integer(), sampleRate, channels);

        private sealed class Stage(int delay, int sampleRate, int channels) : IAudioStage
        {
            private float[] _held = new float[channels * delay];

            public int SampleRate => sampleRate;

            public int Channels => channels;

            public int Latency => delay;

            public AudioData Process(AudioData block)
            {
                int length = block.Length, total = delay + length;
                var all = new float[channels * total];
                for (int c = 0; c < channels; c++)
                {
                    _held.AsSpan(c * delay, delay).CopyTo(all.AsSpan(c * total));
                    block.Channel(c).CopyTo(all.AsSpan(c * total + delay));
                }

                var output = new float[channels * length];
                _held = new float[channels * delay];
                for (int c = 0; c < channels; c++)
                {
                    all.AsSpan(c * total, length).CopyTo(output.AsSpan(c * length));
                    all.AsSpan(c * total + length, delay).CopyTo(_held.AsSpan(c * delay));
                }

                return new AudioData(output, channels, sampleRate);
            }

            public AudioData Flush() => new(_held, channels, sampleRate);
        }
    }

    // An app's whole-recording processor implementing only Apply: the sound backwards.
    private sealed class ProcessingReverse : IRecordingAudioProcessor
    {
        public const string Id = "test-reverse";

        public string Name => Id;

        public string Summary => "the sound backwards (a test's)";

        public AudioData Apply(AudioData audio, AudioProcessorStep step)
        {
            var samples = audio.Samples.AsSpan().ToArray();
            for (int c = 0; c < audio.Channels; c++)
            {
                samples.AsSpan(c * audio.Length, audio.Length).Reverse();
            }

            return new AudioData(samples, audio.Channels, audio.SampleRate);
        }
    }

    // Live sound in blocks of `size`.
    private sealed class ProcessingInput(AudioData sound, int size) : IAudioInput
    {
        public bool Disposed { get; private set; }

        public string Description => "a test's blocks";

        public int SampleRate => sound.SampleRate;

        public int Channels => sound.Channels;

        public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var block in ProcessingBlocks(sound, size))
            {
                await Task.Yield();
                yield return block;
            }
        }

        public void Dispose() => Disposed = true;
    }

    // The cleaning stages as they were before plan 31 (src/Idrak.Audio/AudioCleaning.cs at d957cd00), kept verbatim as the
    // reference the processors must match bit for bit.
    private static class CleanersBefore31
    {
        public static AudioData Clean(string name, AudioData audio, AudioCleaningOptions options) => name switch
        {
            "highpass" => HighPass(audio, options),
            "trim" => Trim(audio, options),
            "normalize" => Normalize(audio, options),
            "denoise" => Denoise(audio, options),
            _ => throw new ArgumentException(name),
        };

        private static (float[] Decibels, float Loudest, int Hop) Frames(AudioData audio, double hopSeconds)
        {
            int hop = Math.Max(1, (int)Math.Round(hopSeconds * audio.SampleRate));
            int length = audio.Length, channels = audio.Channels, frames = (length + hop - 1) / hop;
            var samples = audio.Samples;
            var decibels = new float[frames];
            float loudest = -200f;
            for (int f = 0; f < frames; f++)
            {
                int start = f * hop, end = Math.Min(length, start + hop);
                double sum = 0;
                if (channels == 1)
                {
                    foreach (float x in samples.AsSpan(start, end - start))
                    {
                        sum += (double)x * x;
                    }
                }
                else
                {
                    for (int i = start; i < end; i++)
                    {
                        double mix = 0;
                        for (int c = 0; c < channels; c++)
                        {
                            mix += samples[c * length + i];
                        }

                        mix /= channels;
                        sum += mix * mix;
                    }
                }

                double meanSquare = sum / (end - start);
                decibels[f] = meanSquare > 0 ? (float)(10 * Math.Log10(meanSquare)) : -200f;
                loudest = Math.Max(loudest, decibels[f]);
            }

            return (decibels, loudest, hop);
        }

        private static void Scale(ReadOnlySpan<float> x, float gain, Span<float> y)
        {
            int i = 0, width = System.Numerics.Vector<float>.Count;
            var g = new System.Numerics.Vector<float>(gain);
            for (; i + width <= x.Length; i += width)
            {
                (new System.Numerics.Vector<float>(x.Slice(i, width)) * g).CopyTo(y.Slice(i, width));
            }

            for (; i < x.Length; i++)
            {
                y[i] = x[i] * gain;
            }
        }

        private static AudioData HighPass(AudioData audio, AudioCleaningOptions options)
        {
            double cutoff = options.HighPassHertz;
            if (cutoff <= 0 || audio.Length == 0)
            {
                return audio;
            }

            double w0 = 2 * Math.PI * cutoff / audio.SampleRate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / Math.Sqrt(2);
            double a0 = 1 + alpha;
            double b0 = (1 + cos) / 2 / a0, b1 = -(1 + cos) / a0, b2 = b0, a1 = -2 * cos / a0, a2 = (1 - alpha) / a0;
            int length = audio.Length;
            var samples = audio.Samples;
            var output = GC.AllocateUninitializedArray<float>(samples.Length);
            for (int c = 0; c < audio.Channels; c++)
            {
                var x = samples.AsSpan(c * length, length);
                var y = output.AsSpan(c * length, length);
                double x1 = x[0], x2 = x[0], y1 = 0, y2 = 0;
                for (int i = 0; i < length; i++)
                {
                    double xi = x[i];
                    double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                    y[i] = (float)yi;
                    (x2, x1, y2, y1) = (x1, xi, y1, yi);
                }
            }

            return new AudioData(output, audio.Channels, audio.SampleRate);
        }

        private static AudioData Trim(AudioData audio, AudioCleaningOptions options)
        {
            if (audio.Length == 0)
            {
                return audio;
            }

            var (decibels, loudest, hop) = Frames(audio, options.HopSeconds);
            float threshold = loudest - (float)options.SilenceDecibels;
            int first = Array.FindIndex(decibels, d => d >= threshold), last = Array.FindLastIndex(decibels, d => d >= threshold);
            if (first < 0)
            {
                return audio;
            }

            int pad = (int)Math.Round(options.TrimPadSeconds * audio.SampleRate);
            int start = Math.Max(0, first * hop - pad), end = Math.Min(audio.Length, (last + 1) * hop + pad);
            return start == 0 && end == audio.Length ? audio : audio.Slice(start, end - start);
        }

        private static AudioData Normalize(AudioData audio, AudioCleaningOptions options)
        {
            if (audio.Length == 0)
            {
                return audio;
            }

            var (decibels, loudest, _) = Frames(audio, options.HopSeconds);
            float threshold = loudest - (float)options.SilenceDecibels;
            double sum = 0;
            int count = 0;
            foreach (float d in decibels)
            {
                if (d >= threshold && d > -200f)
                {
                    sum += Math.Pow(10, d / 10);
                    count++;
                }
            }

            if (count == 0 || !(sum > 0))
            {
                return audio;
            }

            double level = 10 * Math.Log10(sum / count);
            double gain = Math.Pow(10, (options.TargetDecibels - level) / 20);
            var samples = audio.Samples;
            float peak = 0;
            foreach (float x in samples)
            {
                peak = Math.Max(peak, Math.Abs(x));
            }

            if (peak * gain > options.PeakCeiling)
            {
                gain = options.PeakCeiling / peak;
            }

            if (Math.Abs(gain - 1) < 1e-6)
            {
                return audio;
            }

            var output = GC.AllocateUninitializedArray<float>(samples.Length);
            Scale(samples, (float)gain, output);
            return new AudioData(output, audio.Channels, audio.SampleRate);
        }

        private static AudioData Denoise(AudioData audio, AudioCleaningOptions options)
        {
            int size = 16;
            while (size * 2 <= options.DenoiseWindowSeconds * audio.SampleRate)
            {
                size *= 2;
            }

            if (audio.Length < 2 * size)
            {
                return audio;
            }

            var stft = new AudioFeatureOptions { SampleRate = audio.SampleRate, WindowSize = size, HopSize = size / 4, Window = "hann", Center = true, PadSeconds = null };
            int length = audio.Length, frames = Stft.FramesFor(length, stft), bins = stft.Bins;
            float floorGain = (float)Math.Pow(10, options.DenoiseFloorDecibels / 20);
            float strength = (float)options.DenoiseStrength;
            var real = new float[frames * bins];
            var imaginary = new float[frames * bins];
            var magnitude = new float[frames * bins];
            var gain = new float[frames * bins];
            var energy = new float[frames];
            var order = new int[frames];
            var floor = new float[bins];
            var output = new float[audio.Samples.Length];
            int quietest = Math.Clamp((int)Math.Round(options.NoiseQuantile * frames), 1, frames);
            for (int c = 0; c < audio.Channels; c++)
            {
                Stft.ComplexOnHost(audio.Channel(c), stft, real, imaginary);
                for (int f = 0; f < frames; f++)
                {
                    float sum = 0;
                    for (int b = 0; b < bins; b++)
                    {
                        int i = f * bins + b;
                        magnitude[i] = MathF.Sqrt(real[i] * real[i] + imaginary[i] * imaginary[i]);
                        sum += magnitude[i] * magnitude[i];
                    }

                    energy[f] = sum;
                    order[f] = f;
                }

                Array.Sort(energy, order);
                Array.Clear(floor);
                for (int q = 0; q < quietest; q++)
                {
                    var m = magnitude.AsSpan(order[q] * bins, bins);
                    for (int b = 0; b < bins; b++)
                    {
                        floor[b] += m[b] / quietest;
                    }
                }

                for (int f = 0; f < frames; f++)
                {
                    var m = magnitude.AsSpan(f * bins, bins);
                    var g = gain.AsSpan(f * bins, bins);
                    for (int b = 0; b < bins; b++)
                    {
                        g[b] = m[b] > 0 ? Math.Clamp((m[b] - strength * floor[b]) / m[b], floorGain, 1f) : floorGain;
                    }
                }

                for (int f = 0; f < frames; f++)
                {
                    int f0 = Math.Max(0, f - 1), f1 = Math.Min(frames - 1, f + 1);
                    for (int b = 0; b < bins; b++)
                    {
                        float sum = 0;
                        for (int ff = f0; ff <= f1; ff++)
                        {
                            sum += gain[ff * bins + b];
                        }

                        float smoothed = Math.Clamp(sum / (f1 - f0 + 1), floorGain, 1f);
                        real[f * bins + b] *= smoothed;
                        imaginary[f * bins + b] *= smoothed;
                    }
                }

                var samples = Stft.Inverse(real, imaginary, frames, stft, length);
                samples.AsSpan(0, length).CopyTo(output.AsSpan(c * length, length));
            }

            return new AudioData(output, audio.Channels, audio.SampleRate);
        }
    }
}
