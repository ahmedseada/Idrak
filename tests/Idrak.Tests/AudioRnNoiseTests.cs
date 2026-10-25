// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins;
using Idrak.Audio.Plugins.RnNoise;

// Plan 31, P5: RNNoise as the plug-in processor "rnnoise" (Idrak.Audio.Plugins.RnNoise) against xiph's RNNoise v0.1.1
// built from its C sources (tools/audio/make_rnnoise_reference.py writes the fixture and the model file), block by block
// bit for bit at 48 kHz and through the resampler at 16 kHz, its text, its refusals, its registration and its time.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioRnNoiseGroup =
    [
        ("audio processing: rnnoise: registered as \"rnnoise\" by Idrak.Audio.Plugins (RnNoisePlugin, in AudioPlugins.All), live, not a library default; its options (model, mix, attenuation, convert) each with a help line, none a processor's name; Unregister takes it out of the registry and Register puts it back", AudioRnNoiseRegistry),
        ("audio processing: rnnoise: against xiph's RNNoise v0.1.1 built from C (2 s of speech-like sound at 48 kHz: silence, voiced syllables, a fricative over a noise floor): the samples within 1e-4 (Apply takes the 959-sample delay off), each frame's voice probability within 1e-4, the model file's layers", AudioRnNoiseReference),
        ("audio processing: rnnoise: block by block (blocks of 1, 7, 160 and 4,096) gives Apply's samples bit for bit at 48 kHz (mono and stereo, with mix and attenuation) and at 16 kHz (resampled to 48 kHz and back), the output longer by the latency at 48 kHz; mix=0 gives the high-pass-free original delayed; nothing allocated per block at 48 kHz once open", AudioRnNoiseBlocks),
        ("audio processing: rnnoise: pipeline text (highpass=80, rnnoise mix=0.8 attenuation=30, agc) round-trips and an option after a comma joins it; refused naming the cause: mix over 1, an attenuation of 0, convert that is not true or false, an unknown option, convert=false at 16 kHz, a missing model (where it looked and how to get it), a file that is not a model, a model whose layers do not fit, a weight outside a signed byte", AudioRnNoiseText),
        ("audio processing: rnnoise: cold and warm time on 10 s of 16 kHz mono and 48 kHz stereo (reported, not asserted)", AudioRnNoiseTiming),
    ];

    private static string RnNoiseData(string relative) => AudioData(Path.Combine("rnnoise", relative));

    private static string RnNoiseModelPath => RnNoiseData("rnnoise.rnnn");

    // The step with the fixture's model (the path as an option, so it may hold spaces) and `options`.
    private static AudioProcessorStep RnNoiseStep(params (string Key, string Value)[] options) =>
        new(RnNoisePlugin.ProcessorName, null, options.Select(o => KeyValuePair.Create(o.Key, o.Value)).Prepend(KeyValuePair.Create(RnNoisePlugin.ModelOption, RnNoiseModelPath)));

    private static AudioPipeline RnNoisePipeline(params (string Key, string Value)[] options)
    {
        RnNoisePlugin.Register();
        return new AudioPipeline([RnNoiseStep(options)]);
    }

    private static AudioData RnNoiseInput() => new(ReadNpyFloat32(RnNoiseData("input.npy")), 1, 48000);

    private static void AudioRnNoiseRegistry(Device device)
    {
        _ = device;
        RnNoisePlugin.Register();
        Check(AudioPlugins.All.Contains(RnNoisePlugin.Plugin) && RnNoisePlugin.IsRegistered, "RnNoisePlugin is one of AudioPlugins.All and registered");
        var processor = AudioProcessors.Find("rnnoise");
        Check(processor is not null && processor.Live && AudioProcessors.Origin("rnnoise") == AudioPlugins.Package.Name,
            $"rnnoise is registered, live, by {AudioPlugins.Package.Name} (origin {AudioProcessors.Origin("rnnoise")})");
        Check(AudioProcessors.Default("rnnoise") is null, "the library has no rnnoise of its own (no learned denoiser is a library default)");
        Check(AudioProcessors.Describe().Contains("rnnoise: ", StringComparison.Ordinal), "rnnoise is listed");
        Check(processor!.Keys.OrderBy(k => k, StringComparer.Ordinal).SequenceEqual(["attenuation", "convert", "mix", "model"]), $"its options: {string.Join(", ", processor.Keys)}");
        Check(!processor.Keys.Any(k => AudioProcessors.Find(k) is not null), $"no option is a processor's name ({string.Join(", ", processor.Keys)})");
        Check(processor.Parameters.All(p => p.Help is { Length: > 0 } && (p.Input != AudioProcessorInput.Number || (p.Unit is not null && p.Min is not null && p.Max is not null))),
            "every input has a help line, a number its unit and range");
        Check(processor.Summary.Contains("model=PATH", StringComparison.Ordinal) && processor.Summary.Contains(RnNoisePlugin.ModelVariable, StringComparison.Ordinal),
            "the summary names model=PATH and IDRAK_RNNOISE_MODEL");

        RnNoisePlugin.Unregister();
        try
        {
            Check(!RnNoisePlugin.IsRegistered && AudioProcessors.Find("rnnoise") is null && !AudioProcessors.Names.Contains("rnnoise"), "Unregister takes rnnoise out");
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("highpass=80, rnnoise"), "a pipeline naming rnnoise without the plug-in", "Unknown audio processor 'rnnoise'");
        }
        finally
        {
            RnNoisePlugin.Register();
        }

        Check(AudioProcessors.Find("rnnoise") is not null, "Register puts it back");
    }

    private static void AudioRnNoiseReference(Device device)
    {
        _ = device;
        var manifest = JsonNode.Parse(File.ReadAllText(RnNoiseData("rnnoise.json")))!.AsObject();
        var input = RnNoiseInput();
        var expected = ReadNpyFloat32(RnNoiseData("output.npy"));
        var vadExpected = ReadNpyFloat32(RnNoiseData("vad.npy"));
        int frames = (int)manifest["frames"]!;
        Check(input.Length == frames * 480 && expected.Length == input.Length - 480 && vadExpected.Length == frames, $"the fixture: {input.Length} samples, {expected.Length} out, {vadExpected.Length} frames");

        var model = RnNoiseModel.Load(RnNoiseModelPath);
        int[] neurons = [model.InputDense.Neurons, model.VadGru.Neurons, model.NoiseGru.Neurons, model.DenoiseGru.Neurons, model.DenoiseOutput.Neurons, model.VadOutput.Neurons];
        int[] stated = [.. manifest["layers"]!.AsArray().Select(l => (int)l!["neurons"]!)];
        Check(neurons.SequenceEqual(stated) && ReferenceEquals(model, RnNoiseModel.Load(RnNoiseModelPath)), $"the model's layers {string.Join(", ", neurons)} as the reference states, read once");

        // Through the processor: Apply takes the delay off, so sample i is the reference's sample i.
        var output = RnNoisePipeline().Apply(input);
        Check(output.Length == input.Length && output.SampleRate == 48000 && output.Channels == 1, $"{output.Length} samples out for {input.Length} in (the delay taken off)");
        double largest = 0;
        int at = 0, exact = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float mine = output.Samples[i];
            double d = Math.Abs((double)mine - expected[i]);
            exact += mine == expected[i] ? 1 : 0;
            (largest, at) = d > largest ? (d, i) : (largest, at);
        }

        Check(largest <= 1e-4, $"the samples within 1e-4 of RNNoise's: the largest difference {largest:G3} at sample {at}");
        Console.WriteLine($"    samples: the largest difference from RNNoise's {largest:G3} (at {at}), {exact} of {expected.Length} equal bit for bit");

        // Frame by frame through the denoiser: the voice probabilities.
        var denoiser = new RnNoiseDenoiser(model);
        var frame = new float[480];
        double vadLargest = 0;
        for (int f = 0; f < frames; f++)
        {
            for (int i = 0; i < 480; i++)
            {
                frame[i] = input.Samples[f * 480 + i] * 32768f;
            }

            float vad = denoiser.ProcessFrame(frame, frame);
            vadLargest = Math.Max(vadLargest, Math.Abs(vad - vadExpected[f]));
        }

        Check(vadLargest <= 1e-4, $"each frame's voice probability within 1e-4 of RNNoise's: the largest difference {vadLargest:G3}");
        int voiced = vadExpected.Count(v => v > 0.5f);
        Check(voiced > frames / 4 && voiced < frames, $"the fixture has voiced and unvoiced frames ({voiced} of {frames} over 0.5)");
        Console.WriteLine($"    voice probability: the largest difference {vadLargest:G3}; {voiced} of {frames} frames voiced");
    }

    private static void AudioRnNoiseBlocks(Device device)
    {
        _ = device;
        var mono = RnNoiseInput();
        var stereoSamples = new float[2 * mono.Length];
        for (int i = 0; i < mono.Length; i++)
        {
            stereoSamples[i] = mono.Samples[i];
            stereoSamples[mono.Length + i] = 0.5f * mono.Samples[(i + 4321) % mono.Length];
        }

        var stereo = new AudioData(stereoSamples, 2, 48000);
        var low = Resampler.Resample(mono, 16000);
        var cases = new List<(string What, AudioPipeline Pipeline, AudioData Sound)>
        {
            ("48 kHz mono", RnNoisePipeline(), mono),
            ("48 kHz stereo, mix 0.7, attenuation 20 dB", RnNoisePipeline(("mix", "0.7"), ("attenuation", "20")), stereo),
            ("16 kHz mono", RnNoisePipeline(), low),
            ("16 kHz mono, attenuation 12 dB", RnNoisePipeline(("attenuation", "12")), low),
        };
        foreach (var (what, pipeline, sound) in cases)
        {
            var whole = pipeline.Apply(sound);
            using (var probe = pipeline.Open(sound.SampleRate, sound.Channels))
            {
                Check(whole.SampleRate == sound.SampleRate && whole.Channels == sound.Channels && whole.Length == sound.Length,
                    $"{what}: {whole.Length} samples out for {sound.Length} in (latency {probe.Latency} taken off)");
            }

            foreach (int size in new[] { 1, 7, 160, 4096 })
            {
                using var stage = pipeline.Open(sound.SampleRate, sound.Channels);
                var pieces = new List<AudioData>();
                foreach (var block in ProcessingBlocks(sound, size))
                {
                    pieces.Add(stage.Process(block));
                }

                pieces.Add(stage.Flush());
                var joined = ProcessingAligned(ProcessingJoin(pieces, stage.Channels, stage.SampleRate), stage.Latency);
                Check(joined.Length == whole.Length && ProcessingSame(joined.Samples, whole.Samples),
                    $"{what}, blocks of {size}: {joined.Length} samples (its latency {stage.Latency} off) against Apply's {whole.Length}, the same bits");
            }
        }

        using (var at48 = RnNoisePipeline().Open(48000, 1))
        using (var at16 = RnNoisePipeline().Open(16000, 1))
        {
            Check(at48.Latency == RnNoiseStream.Delay && at48.Delay == RnNoiseStream.Delay && at16.Latency == 320 && at16.Delay > 320,
                $"the latency: {at48.Latency} at 48 kHz (959), {at16.Latency} at 16 kHz (960 at 48 kHz: the delay padded to whole samples), held back {at16.Delay} (with the resamplers')");
        }

        // mix=0: the original, the delay taken off by Apply.
        var dry = RnNoisePipeline(("mix", "0")).Apply(mono);
        Check(dry.Samples.AsSpan().SequenceEqual(mono.Samples), "mix=0 gives the original sound, aligned");

        // Nothing allocated per block once open (48 kHz, stereo, blocks of 10 ms).
        using var live = RnNoisePipeline().Open(48000, 2);
        var blocks = Enumerable.Range(0, 120).Select(k => new AudioData(stereo.Slice(480 * (k % 150), 480).Samples, 2, 48000)).ToArray();
        for (int k = 0; k < 60; k++)
        {
            live.Process(blocks[k]);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int k = 60; k < 120; k++)
        {
            live.Process(blocks[k]);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0, $"60 blocks of 480 stereo samples allocated {allocated} bytes once open");
    }

    private static void AudioRnNoiseText(Device device)
    {
        _ = device;
        RnNoisePlugin.Register();
        var pipeline = AudioPipeline.Parse("highpass=80, rnnoise mix=0.8 attenuation=30, agc");
        Check(pipeline.ToString() == "highpass=80, rnnoise attenuation=30 mix=0.8, agc" && AudioPipeline.Parse(pipeline.ToString()).Equals(pipeline) && pipeline.Live,
            $"the text: {pipeline}");
        Check(AudioPipeline.FromJson(pipeline.ToJson()).Equals(pipeline), "the JSON form round-trips");
        Check(AudioPipeline.Parse("rnnoise, mix=0.5").Steps[0].Option("mix") == "0.5" && AudioPipeline.Parse("rnnoise, convert=false").Steps.Count == 1,
            "an option after a comma joins rnnoise");
        Check(AudioPipeline.Parse("rnnoise model=D:\\Models\\sh.rnnn").Steps[0].Option("model") == "D:\\Models\\sh.rnnn", "model=PATH is kept as written");
        using (var stage = RnNoisePipeline().Open(48000, 1))
        {
            Check(stage.SampleRate == 48000 && stage.Channels == 1, "the stage keeps the rate and channels");
        }

        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("rnnoise mix=1.5"), "a mix over 1", "rnnoise mix", "0 to 1", "1.5");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("rnnoise attenuation=0"), "an attenuation of 0", "rnnoise attenuation", "above 0 to 120 dB", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("rnnoise convert=maybe"), "a convert that is not a switch", "rnnoise convert", "true or false", "maybe");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("rnnoise strength=3"), "an unknown option", "rnnoise", "strength", "mix, attenuation, convert, model");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("rnnoise=3"), "a value", "rnnoise takes no value");
        ThrowsNaming<ArgumentException>(() => RnNoisePipeline(("convert", "false")).Open(16000, 1), "convert=false at 16 kHz", "48000 Hz", "16000 Hz", "resample=48000");

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-rnnoise-{Environment.ProcessId}");
        Directory.CreateDirectory(folder);
        string? cache = Environment.GetEnvironmentVariable("IDRAK_CACHE"), variable = Environment.GetEnvironmentVariable(RnNoisePlugin.ModelVariable);
        try
        {
            Environment.SetEnvironmentVariable("IDRAK_CACHE", folder);
            Environment.SetEnvironmentVariable(RnNoisePlugin.ModelVariable, null);
            string expectedDefault = Path.Combine(folder, "rnnoise", "rnnoise.rnnn");
            Check(RnNoisePlugin.DefaultModelPath == expectedDefault, $"the default model is rnnoise/rnnoise.rnnn in the cache: {RnNoisePlugin.DefaultModelPath}");
            ThrowsNaming<FileNotFoundException>(() => AudioPipeline.Parse("rnnoise").Apply(RnNoiseInput()), "no model anywhere",
                "RNNoise's model was not found", RnNoisePlugin.ModelVariable, expectedDefault, "make_rnnoise_reference.py --model-only", "model=PATH");
            ThrowsNaming<FileNotFoundException>(() => AudioPipeline.Parse("rnnoise model=sh").Open(48000, 1), "a model named but not in the cache",
                "rnnoise model=sh", Path.Combine(folder, "rnnoise", "sh.rnnn"), "make_rnnoise_reference.py");
            Environment.SetEnvironmentVariable(RnNoisePlugin.ModelVariable, Path.Combine(folder, "missing.rnnn"));
            ThrowsNaming<FileNotFoundException>(() => AudioPipeline.Parse("rnnoise").Open(48000, 1), "the variable naming no file", RnNoisePlugin.ModelVariable, "missing.rnnn");

            // Found through the variable, and by name in the cache's rnnoise folder.
            Environment.SetEnvironmentVariable(RnNoisePlugin.ModelVariable, RnNoiseModelPath);
            using (var stage = AudioPipeline.Parse("rnnoise").Open(48000, 1))
            {
                Check(stage.Latency == RnNoiseStream.Delay, "the model named by IDRAK_RNNOISE_MODEL opens");
            }

            Environment.SetEnvironmentVariable(RnNoisePlugin.ModelVariable, null);
            Directory.CreateDirectory(Path.Combine(folder, "rnnoise"));
            File.Copy(RnNoiseModelPath, Path.Combine(folder, "rnnoise", "classic.rnnn"), overwrite: true);
            using (var stage = AudioPipeline.Parse("rnnoise model=classic").Open(48000, 1))
            {
                Check(stage.Latency == RnNoiseStream.Delay, "model=classic opens rnnoise/classic.rnnn in the cache");
            }

            File.Copy(RnNoiseModelPath, expectedDefault, overwrite: true);
            using (var stage = AudioPipeline.Parse("rnnoise").Open(48000, 1))
            {
                Check(stage.Latency == RnNoiseStream.Delay, "the default model in the cache opens");
            }

            string notModel = Path.Combine(folder, "not-a-model.rnnn");
            File.WriteAllText(notModel, "hello\n");
            ThrowsNaming<InvalidDataException>(() => RnNoiseModel.Load(notModel), "a file that is not a model", "not an RNNoise model file", RnNoiseModel.Header);
            string wrong = Path.Combine(folder, "wrong.rnnn");
            File.WriteAllText(wrong, RnNoiseModelText(40, 0));
            ThrowsNaming<InvalidDataException>(() => RnNoiseModel.Load(wrong), "a model of 40 features", "the dense input layer has 40", "needs 42");
            string wide = Path.Combine(folder, "wide.rnnn");
            File.WriteAllText(wide, RnNoiseModelText(42, 300));
            ThrowsNaming<InvalidDataException>(() => RnNoiseModel.Load(wide), "a weight of 300", "300", "signed bytes");
            string cut = Path.Combine(folder, "cut.rnnn");
            File.WriteAllText(cut, RnNoiseModelText(42, 0)[..200]);
            ThrowsNaming<InvalidDataException>(() => RnNoiseModel.Load(cut), "a model cut short", "cut short");
            var tiny = RnNoiseModel.Load(Path.Combine(folder, WriteText(folder, "zeros.rnnn", RnNoiseModelText(42, 0))));
            Check(tiny.DenoiseGru.Neurons == 96 && tiny.VadOutput.Neurons == 1, "a model of zeros in the right shape is read");
        }
        finally
        {
            Environment.SetEnvironmentVariable("IDRAK_CACHE", cache);
            Environment.SetEnvironmentVariable(RnNoisePlugin.ModelVariable, variable);
            Directory.Delete(folder, recursive: true);
        }

        static string WriteText(string folder, string name, string text)
        {
            File.WriteAllText(Path.Combine(folder, name), text);
            return name;
        }
    }

    // A model file of RNNoise's classic shape whose dense input layer takes `features` inputs, every weight `weight`.
    private static string RnNoiseModelText(int features, int weight)
    {
        var text = new StringBuilder("rnnoise-nu model file version 1\n");
        void Layer(int inputs, int neurons, int activation, bool gru)
        {
            int gates = gru ? 3 : 1;
            text.Append(inputs).Append(' ').Append(neurons).Append(' ').Append(activation).Append('\n');
            text.Append(string.Join(' ', Enumerable.Repeat(weight, inputs * neurons * gates))).Append('\n');
            if (gru)
            {
                text.Append(string.Join(' ', Enumerable.Repeat(weight, neurons * neurons * gates))).Append('\n');
            }

            text.Append(string.Join(' ', Enumerable.Repeat(0, neurons * gates))).Append('\n');
        }

        Layer(features, 24, 0, false);
        Layer(24, 24, 2, true);
        Layer(90, 48, 2, true);
        Layer(114, 96, 2, true);
        Layer(96, 22, 1, false);
        Layer(24, 1, 1, false);
        return text.ToString();
    }

    private static void AudioRnNoiseTiming(Device device)
    {
        _ = device;
        var pipeline = RnNoisePipeline();
        var speech = RnNoiseInput();
        foreach (var (rate, channels) in new[] { (16000, 1), (48000, 2) })
        {
            var one = rate == 48000 ? speech : Resampler.Resample(speech, rate);
            int length = 10 * rate;
            var samples = new float[channels * length];
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < length; i++)
                {
                    samples[c * length + i] = one.Samples[(i + 977 * c) % one.Length];
                }
            }

            var sound = new AudioData(samples, channels, rate);
            long start = Stopwatch.GetTimestamp();
            _ = pipeline.Apply(sound);
            double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var warm = new List<double>();
            for (int k = 0; k < 5; k++)
            {
                start = Stopwatch.GetTimestamp();
                _ = pipeline.Apply(sound);
                warm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }

            double average = warm.Average();
            Console.WriteLine($"    rnnoise on 10 s of {rate / 1000} kHz, {channels} channel(s): cold {cold,7:F1} ms, warm {average,6:F1} ms ({10000 / average:N0}x real time)");
        }
    }
}
