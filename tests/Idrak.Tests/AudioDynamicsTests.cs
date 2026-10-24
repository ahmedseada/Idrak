// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P2: the dynamics processors (gate, compress, limit, agc, loudness-live live; loudness and peaknorm over the whole
// recording) against tools/audio/make_audio_dynamics.py's reference (the same equations sample by sample, and
// pyloudnorm's integrated loudness), block by block bit for bit, their text and refusals, and their time.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioDynamicsGroup =
    [
        ("audio processing: dynamics: cold and warm time of each stage on 10 s of 16 kHz mono (reported, not asserted)", AudioDynamicsTiming),
        ("audio processing: dynamics: the registry names gate, compress, limit, agc and loudness-live (live), loudness and peaknorm (whole recording) as the library's; no option of theirs is a processor's name (an option after a comma joins its step: agc, quiet=-40; normalize, peak=0.9); a live pipeline refuses loudness naming loudness-live, normalize naming agc, and peaknorm; the limiter reports its look-ahead as latency", AudioDynamicsRegistry),
        ("audio processing: dynamics: each stage against its reference (gate, compress, limit, agc, loudness-live within 1e-5; the limiter never over its ceiling; the integrated loudness within 0.01 LU of pyloudnorm with the standard's and RBJ's filters; the sample and true peaks; the loudness and peaknorm gains)", AudioDynamicsReference),
        ("audio processing: dynamics: every stage run block by block (blocks of 1, 7, 160 and 4,096, mono and stereo, the limiter's latency and tail included) gives Apply's samples bit for bit, and a live stage allocates nothing per block once open", AudioDynamicsBlocks),
        ("audio processing: dynamics: pipeline text (compress=-20 ratio=3, limit=-1; makeup=auto) round-trips; refused naming the cause: a ratio under 1, a makeup that is not a number or auto, an unknown detector, a ceiling over 0 dBFS, a negative look-ahead, an overlap of 1, a rate too low for the K-weighting", AudioDynamicsText),
    ];

    private static string DynamicsData(string relative) => AudioData(Path.Combine("dynamics", relative));

    private static JsonObject DynamicsReference() => JsonNode.Parse(File.ReadAllText(DynamicsData("dynamics.json")))!.AsObject();

    private static AudioData DynamicsInput(JsonNode entry) =>
        new(ReadNpyFloat32(DynamicsData((string)entry["input"]!)), (int)entry["channels"]!, (int)entry["rate"]!);

    private static void AudioDynamicsTiming(Device device)
    {
        _ = device;
        var sound = ProcessingTone(16000, 10, 1, noise: 0.01f, bursts: true);
        foreach (string text in new[] { "gate", "compress", "limit", "agc", "loudness-live", "loudness", "loudness meter=sample", "peaknorm", "peaknorm meter=true" })
        {
            var pipeline = AudioPipeline.Parse(text);
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
            Console.WriteLine($"    {text,-22} 10 s of 16 kHz mono: cold {cold,7:F2} ms, warm {average,6:F2} ms ({10000 / average:N0}x real time)");
        }
    }

    private static void AudioDynamicsRegistry(Device device)
    {
        _ = device;
        string[] live = ["gate", "compress", "limit", "agc", "loudness-live"];
        string[] whole = ["loudness", "peaknorm"];
        Check(live.Concat(whole).All(n => AudioProcessors.Names.Contains(n) && AudioProcessors.Origin(n) == Overrides.Library),
            $"the library's dynamics processors: {string.Join(", ", AudioProcessors.Names)}");
        Check(live.All(n => AudioProcessors.Get(n).Live) && whole.All(n => !AudioProcessors.Get(n).Live), "the live ones are live, loudness and peaknorm need the whole recording");
        Check(AudioProcessors.Get("loudness").LiveAlternative == "loudness-live" && AudioProcessors.Get("normalize").LiveAlternative == "agc" && AudioProcessors.Get("peaknorm").LiveAlternative is null,
            "loudness's live alternative is loudness-live, normalize's is agc, peaknorm has none");
        foreach (string name in live.Concat(whole))
        {
            var processor = AudioProcessors.Get(name);
            Check(processor.Parameters.Count > 0 && processor.Parameters.All(p => p.Help is { Length: > 0 } && (p.Input != AudioProcessorInput.Number || p.Unit is not null || p.Key is "ratio" or "overlap")),
                $"{name}: every input has a help line and its unit");
            Check(AudioProcessors.Describe().Contains($"{name}: ", StringComparison.Ordinal), $"{name} is listed");
            Check(!processor.Keys.Any(k => AudioProcessors.Find(k) is not null), $"{name}: no option is a processor's name ({string.Join(", ", processor.Keys)})");
        }

        Check(AudioPipeline.Parse("agc, quiet=-40").ToString() == "agc quiet=-40" && AudioPipeline.Parse("normalize=-20, peak=0.9").ToString() == "normalize=-20 peak=0.9",
            "an option after a comma joins the step before it");

        var input = new ProcessingInput(ProcessingTone(16000, 0.1, 1), 100);
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("agc, loudness").Wrap(input), "a live pipeline with loudness", "loudness", "whole recording", "use loudness-live instead");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("normalize").Wrap(input), "a live pipeline with normalize", "normalize", "use agc instead");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("peaknorm=-3").Wrap(input), "a live pipeline with peaknorm", "peaknorm", "leave it out");
        using var wrapped = AudioPipeline.Parse("gate=-40, compress=-20 ratio=3, agc, loudness-live, limit=-1").Wrap(input);
        Check(wrapped.SampleRate == 16000, "a pipeline of the live dynamics wraps live sound");
        using var limiter = AudioPipeline.Parse("limit lookahead=2.5").Open(48000, 2);
        Check(limiter.Latency == 120, $"the limiter's look-ahead is its latency: {limiter.Latency}");
        using var plain = AudioPipeline.Parse("gate, compress, agc, loudness-live").Open(48000, 2);
        Check(plain.Latency == 0, "the other live stages have no latency");
    }

    private static void AudioDynamicsReference(Device device)
    {
        _ = device;
        var reference = DynamicsReference();
        foreach (var entry in reference["live"]!.AsArray())
        {
            string text = (string)entry!["pipeline"]!;
            var input = DynamicsInput(entry);
            var expected = ReadNpyFloat32(DynamicsData((string)entry["output"]!));
            var output = AudioPipeline.Parse(text).Apply(input);
            Check(output.Channels == input.Channels && output.Samples.Length == expected.Length,
                $"{text}: {output.Channels} x {output.Length} samples, the reference {expected.Length / input.Channels}");
            double largest = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                largest = Math.Max(largest, Math.Abs(expected[i] - output.Samples[i]));
            }

            Check(largest <= 1e-5, $"{text} on {entry["input"]}: the largest difference from the reference is {largest:G3}");
            Console.WriteLine($"    {text} on {entry["input"]}: largest difference {largest:G3}");
            if (text.StartsWith("limit", StringComparison.Ordinal))
            {
                double ceiling = Math.Pow(10, AudioPipeline.Parse(text).Steps[0].Number(-1) / 20);
                Check(output.Samples.All(x => Math.Abs(x) <= ceiling), $"{text}: no sample over the ceiling {ceiling:G6}");
                Check(output.Length == input.Length + AudioPipeline.Parse(text).Open(input.SampleRate, input.Channels).Latency, $"{text}: the output is longer by the look-ahead");
            }
        }

        // The limiter's promise on sound far over full scale (a loud transient, square waves, noise at 4x).
        var loud = DynamicsInput(reference["live"]![8]!);
        foreach (ref float x in loud.Samples.AsSpan())
        {
            x *= 4;
        }

        foreach (string text in new[] { "limit", "limit=-6 lookahead=0", "limit=-0.1 lookahead=0.5 release=1", "limit=-12 lookahead=20 release=500 link=false" })
        {
            var limited = AudioPipeline.Parse(text).Apply(loud);
            float ceiling = (float)Math.Pow(10, AudioPipeline.Parse(text).Steps[0].Number(-1) / 20);
            Check(limited.Samples.All(x => Math.Abs((double)x) <= Math.Pow(10, AudioPipeline.Parse(text).Steps[0].Number(-1) / 20)),
                $"{text} on sound at 4x full scale: the largest sample {limited.Samples.Max(MathF.Abs):G9}, the ceiling {ceiling:G9}");
        }

        var standard = new AudioProcessorStep("loudness");
        var rbj = new AudioProcessorStep("loudness", null, [KeyValuePair.Create("weighting", "rbj")]);
        foreach (var node in reference["loudness"]!.AsArray())
        {
            var entry = node!;
            var input = DynamicsInput(entry);
            double mine = LoudnessProcessor.Integrated(input, standard)!.Value, mineRbj = LoudnessProcessor.Integrated(input, rbj)!.Value;
            double pyloudnorm = (double)entry["pyloudnorm_deman"]!, pyloudnormRbj = (double)entry["pyloudnorm_kweighting"]!;
            Check(Math.Abs(mine - pyloudnorm) <= 0.01 && Math.Abs(mineRbj - pyloudnormRbj) <= 0.01,
                $"{entry["input"]}: integrated {mine:F4} LUFS against pyloudnorm's {pyloudnorm:F4} (DeMan), {mineRbj:F4} against {pyloudnormRbj:F4} (K-weighting)");
            Check(Math.Abs(mine - (double)entry["integrated"]!) <= 1e-6 && Math.Abs(mineRbj - (double)entry["integrated_rbj"]!) <= 1e-6,
                $"{entry["input"]}: integrated {mine:R} and {mineRbj:R} against the script's {entry["integrated"]} and {entry["integrated_rbj"]}");
            Console.WriteLine($"    {entry["input"]}: integrated {mine:F4} LUFS, pyloudnorm {pyloudnorm:F4} (difference {mine - pyloudnorm:G3}); RBJ {mineRbj:F4}, pyloudnorm {pyloudnormRbj:F4} (difference {mineRbj - pyloudnormRbj:G3})");
            double truePeak = DynamicsMath.TruePeak(input, 4, 6), samplePeak = DynamicsMath.SamplePeak(input.Samples);
            Check(samplePeak == (double)entry["sample_peak"]! && Math.Abs(truePeak - (double)entry["true_peak"]!) <= 1e-6 * truePeak && truePeak >= samplePeak,
                $"{entry["input"]}: sample peak {samplePeak:R} (the reference {entry["sample_peak"]}), true peak {truePeak:R} (the reference {entry["true_peak"]})");
        }

        foreach (var entry in reference["whole"]!.AsArray())
        {
            string text = (string)entry!["pipeline"]!;
            var input = DynamicsInput(entry);
            float gain = (float)(double)entry["gain"]!;
            var output = AudioPipeline.Parse(text).Apply(input);
            double largest = 0;
            for (int i = 0; i < input.Samples.Length; i++)
            {
                largest = Math.Max(largest, Math.Abs(input.Samples[i] * gain - output.Samples[i]));
            }

            Check(output.Samples.Length == input.Samples.Length && largest <= 1e-5, $"{text} on {entry["input"]}: the samples times the reference's gain {gain:R}, by up to {largest:G3}");
            Console.WriteLine($"    {text} on {entry["input"]}: gain {gain:G6}, largest difference {largest:G3}");
        }

        var silence = new AudioData(new float[16000], 1, 16000);
        Check(ReferenceEquals(AudioPipeline.Parse("loudness").Apply(silence), silence) && ReferenceEquals(AudioPipeline.Parse("peaknorm").Apply(silence), silence),
            "loudness and peaknorm leave a silent recording as it is");
        var tone = ProcessingTone(16000, 0.2, 1);
        Check(LoudnessProcessor.Integrated(tone, standard) is { } shortLevel && double.IsFinite(shortLevel), "a recording shorter than one block is measured as one block");
    }

    private static void AudioDynamicsBlocks(Device device)
    {
        _ = device;
        var reference = DynamicsReference();
        var texts = new List<(string Text, AudioData Sound)>();
        foreach (var entry in reference["live"]!.AsArray())
        {
            texts.Add(((string)entry!["pipeline"]!, DynamicsInput(entry)));
        }

        var speech = texts[0].Sound;
        var stereo = texts[2].Sound;
        texts.Add(("gate=-40, compress=-20 ratio=3, agc, loudness-live, limit=-1", stereo));
        texts.Add(("gate=-40, compress=-20 ratio=3, limit=-1, loudness, peaknorm meter=true", speech));
        foreach (var (text, sound) in texts)
        {
            var pipeline = AudioPipeline.Parse(text);
            var whole = pipeline.Apply(sound);
            foreach (int size in new[] { 1, 7, 160, 4096 })
            {
                using var stage = pipeline.Open(sound.SampleRate, sound.Channels);
                var pieces = new List<AudioData>();
                foreach (var block in ProcessingBlocks(sound, size))
                {
                    pieces.Add(stage.Process(block));
                }

                pieces.Add(stage.Flush());
                var joined = ProcessingJoin(pieces, stage.Channels, stage.SampleRate);
                Check(joined.Length == whole.Length && ProcessingSame(joined.Samples, whole.Samples),
                    $"{text}, {sound.Channels} channel(s), blocks of {size}: {joined.Length} samples against Apply's {whole.Length}, the same bits");
            }
        }

        foreach (string text in new[] { "gate", "compress detector=peak", "compress link=false", "limit", "limit lookahead=0", "agc link=false", "loudness-live" })
        {
            using var stage = AudioPipeline.Parse(text).Open(48000, 2);
            var blocks = Enumerable.Range(0, 120).Select(_ => AudioStages.Copy(stereo.Slice(4800, 480))).ToArray();
            for (int k = 0; k < 60; k++)
            {
                stage.Process(blocks[k]);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int k = 60; k < 120; k++)
            {
                stage.Process(blocks[k]);
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated == 0, $"{text}: 60 blocks of 480 stereo samples allocated {allocated} bytes once open");
        }
    }

    private static void AudioDynamicsText(Device device)
    {
        _ = device;
        var pipeline = AudioPipeline.Parse("compress=-20 ratio=3, limit=-1");
        Check(pipeline.ToString() == "compress=-20 ratio=3, limit=-1" && pipeline.Live && AudioPipeline.Parse(pipeline.ToString()).Equals(pipeline), $"the text: {pipeline}");
        var full = AudioPipeline.Parse("gate=-45 Hold=80 hysteresis=4, compress=-24 ratio=4 knee=0 makeup=AUTO detector=peak, limit=-1 lookahead=3, agc=-18 max=12, loudness-live=-20 window=400, loudness=-16 meter=sample, peaknorm=-2");
        Check(full.ToString() == "gate=-45 hold=80 hysteresis=4, compress=-24 detector=peak knee=0 makeup=AUTO ratio=4, limit=-1 lookahead=3, agc=-18 max=12, loudness-live=-20 window=400, loudness=-16 meter=sample, peaknorm=-2"
              && AudioPipeline.FromJson(full.ToJson()).Equals(full) && !full.Live, $"every dynamics step round-trips: {full}");
        Check(AudioPipeline.Parse("compress, ratio=2").Steps[0].Option("ratio") == "2", "an option after a comma joins the compressor");

        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("compress ratio=0.5"), "a ratio under 1", "compress ratio", "at least 1", "0.5");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("compress makeup=loud"), "a makeup that is neither a number nor auto", "compress makeup", "number", "loud");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("compress detector=avg"), "an unknown detector", "compress detector", "rms, peak", "avg");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("compress=3"), "a threshold over full scale", "compress", "at most 0 dBFS", "3");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("limit=1"), "a ceiling over 0 dBFS", "limit", "at most 0 dBFS", "1");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("limit lookahead=-1"), "a negative look-ahead", "limit lookahead", "0 to 1000 ms", "-1");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("gate range=-5"), "a negative range", "gate range", "at least 0 dB", "-5");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("gate link=maybe"), "a link that is not true or false", "gate link", "true or false", "maybe");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("agc max=-3"), "a most gain under 0", "agc max", "at least 0 dB");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("agc depth=3"), "an unknown option", "agc", "depth", "max, min, attack");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("agc quiet=3"), "a silence level over full scale", "agc quiet", "at most 0 dBFS");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness overlap=1"), "an overlap of a whole block", "loudness overlap", "0 to 0.99", "1");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness meter=rms"), "an unknown meter", "loudness meter", "true, sample, none", "rms");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness-live window=0"), "an empty window", "loudness-live window", "above 0", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("peaknorm oversample=0"), "no true-peak points", "peaknorm oversample", "1 to 64", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness").Apply(ProcessingTone(3000, 1, 1)), "a rate too low for the K-weighting", "loudness", "3363.9 Hz", "3000 Hz");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness-live weighting=rbj").Open(2000, 1), "a rate too low for RBJ's shelf", "loudness-live", "3000 Hz", "2000 Hz");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("loudness block=0.01").Apply(ProcessingTone(16000, 1, 1)), "a block under one sample", "loudness", "under one sample");
    }
}
