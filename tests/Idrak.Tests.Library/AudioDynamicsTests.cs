// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P2: the dynamics processors (gate, compress, limit, agc, loudness-live live; loudness and peaknorm over the whole
// recording), each reached directly with its step: the limiter's ceiling, the integrated loudness against
// tools/audio/make_audio_dynamics.py's reference (pyloudnorm's), their refusals, their allocations and their time.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioDynamicsGroup =
    [
        ("audio processing: dynamics: cold and warm time of each stage on 10 s of 16 kHz mono (reported, not asserted)", AudioDynamicsTiming),
        ("audio processing: dynamics: gate, compress, limit, agc and loudness-live are live, loudness and peaknorm need the whole recording (loudness's live alternative is loudness-live, normalize's agc, peaknorm has none); every input has a help line and its unit; the limiter reports its look-ahead as latency, the other live stages none", AudioDynamicsKinds),
        ("audio processing: dynamics: the limiter never over its ceiling on sound at 4x full scale; the integrated loudness within 0.01 LU of pyloudnorm with the standard's and RBJ's filters; the sample and true peaks; loudness and peaknorm leave a silent recording as it is", AudioDynamicsReference),
        ("audio processing: dynamics: a live stage allocates nothing per block once open", AudioDynamicsBlocks),
        ("audio processing: dynamics: refused naming the cause: a ratio under 1, a makeup that is not a number or auto, an unknown detector, a ceiling over 0 dBFS, a negative look-ahead, an overlap of 1, a rate too low for the K-weighting", AudioDynamicsRefusals),
    ];

    internal static string DynamicsData(string relative) => AudioData(Path.Combine("dynamics", relative));

    internal static JsonObject DynamicsReference() => JsonNode.Parse(File.ReadAllText(DynamicsData("dynamics.json")))!.AsObject();

    internal static AudioData DynamicsInput(JsonNode entry) =>
        new(ReadNpyFloat32(DynamicsData((string)entry["input"]!)), (int)entry["channels"]!, (int)entry["rate"]!);

    internal static void AudioDynamicsTiming(Device device)
    {
        _ = device;
        var sound = ProcessingTone(16000, 10, 1, noise: 0.01f, bursts: true);
        (IAudioProcessor Processor, AudioProcessorStep Step)[] stages =
        [
            (new GateProcessor(), new(DynamicsProcessors.Gate)),
            (new CompressProcessor(), new(DynamicsProcessors.Compress)),
            (new LimitProcessor(), new(DynamicsProcessors.Limit)),
            (new AgcProcessor(), new(DynamicsProcessors.Agc)),
            (new LiveLoudnessProcessor(), new(DynamicsProcessors.LoudnessLive)),
            (new LoudnessProcessor(), new(DynamicsProcessors.Loudness)),
            (new LoudnessProcessor(), new(DynamicsProcessors.Loudness, null, [KeyValuePair.Create("meter", "sample")])),
            (new PeakProcessor(), new(DynamicsProcessors.Peak)),
            (new PeakProcessor(), new(DynamicsProcessors.Peak, null, [KeyValuePair.Create("meter", "true")])),
        ];
        foreach (var (processor, step) in stages)
        {
            long start = Stopwatch.GetTimestamp();
            _ = processor.Apply(sound, step);
            double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var warm = new List<double>();
            for (int k = 0; k < 5; k++)
            {
                start = Stopwatch.GetTimestamp();
                _ = processor.Apply(sound, step);
                warm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }

            double average = warm.Average();
            Console.WriteLine($"    {step,-22} 10 s of 16 kHz mono: cold {cold,7:F2} ms, warm {average,6:F2} ms ({10000 / average:N0}x real time)");
        }
    }

    internal static void AudioDynamicsKinds(Device device)
    {
        _ = device;
        IAudioProcessor gate = new GateProcessor(), compress = new CompressProcessor(), limit = new LimitProcessor(), agc = new AgcProcessor(),
            loudnessLive = new LiveLoudnessProcessor(), loudness = new LoudnessProcessor(), peaknorm = new PeakProcessor(), normalize = new NormalizeProcessor();
        IAudioProcessor[] live = [gate, compress, limit, agc, loudnessLive];
        IAudioProcessor[] whole = [loudness, peaknorm];
        Check(live.All(p => p.Live) && whole.All(p => !p.Live), "the live ones are live, loudness and peaknorm need the whole recording");
        Check(loudness.LiveAlternative == "loudness-live" && normalize.LiveAlternative == "agc" && peaknorm.LiveAlternative is null,
            "loudness's live alternative is loudness-live, normalize's is agc, peaknorm has none");
        foreach (var processor in live.Concat(whole))
        {
            Check(processor.Parameters.Count > 0 && processor.Parameters.All(p => p.Help is { Length: > 0 } && (p.Input != AudioProcessorInput.Number || p.Unit is not null || p.Key is "ratio" or "overlap")),
                $"{processor.Name}: every input has a help line and its unit");
        }

        using (var limiter = limit.Open(new AudioProcessorStep(DynamicsProcessors.Limit, null, [KeyValuePair.Create("lookahead", "2.5")]), 48000, 2))
        {
            Check(limiter.Latency == 120, $"the limiter's look-ahead is its latency: {limiter.Latency}");
        }

        foreach (var processor in new[] { gate, compress, agc, loudnessLive })
        {
            using var stage = processor.Open(new AudioProcessorStep(processor.Name), 48000, 2);
            Check(stage.Latency == 0, $"{processor.Name} has no latency");
        }
    }

    internal static void AudioDynamicsReference(Device device)
    {
        _ = device;
        var reference = DynamicsReference();
        IAudioProcessor limit = new LimitProcessor();

        // The limiter's promise on sound far over full scale (a loud transient, square waves, noise at 4x).
        var loud = DynamicsInput(reference["live"]![8]!);
        foreach (ref float x in loud.Samples.AsSpan())
        {
            x *= 4;
        }

        foreach (var step in new AudioProcessorStep[]
                 {
                     new(DynamicsProcessors.Limit),
                     new(DynamicsProcessors.Limit, "-6", [KeyValuePair.Create("lookahead", "0")]),
                     new(DynamicsProcessors.Limit, "-0.1", [KeyValuePair.Create("lookahead", "0.5"), KeyValuePair.Create("release", "1")]),
                     new(DynamicsProcessors.Limit, "-12", [KeyValuePair.Create("lookahead", "20"), KeyValuePair.Create("release", "500"), KeyValuePair.Create("link", "false")]),
                 })
        {
            var limited = limit.Apply(loud, step);
            float ceiling = (float)Math.Pow(10, step.Number(-1) / 20);
            Check(limited.Samples.All(x => Math.Abs((double)x) <= Math.Pow(10, step.Number(-1) / 20)),
                $"{step} on sound at 4x full scale: the largest sample {limited.Samples.Max(MathF.Abs):G9}, the ceiling {ceiling:G9}");
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

        var silence = new AudioData(new float[16000], 1, 16000);
        IAudioProcessor loudness = new LoudnessProcessor(), peaknorm = new PeakProcessor();
        Check(ReferenceEquals(loudness.Apply(silence, standard), silence) && ReferenceEquals(peaknorm.Apply(silence, new AudioProcessorStep(DynamicsProcessors.Peak)), silence),
            "loudness and peaknorm leave a silent recording as it is");
        var tone = ProcessingTone(16000, 0.2, 1);
        Check(LoudnessProcessor.Integrated(tone, standard) is { } shortLevel && double.IsFinite(shortLevel), "a recording shorter than one block is measured as one block");
    }

    internal static void AudioDynamicsBlocks(Device device)
    {
        _ = device;
        var stereo = DynamicsInput(DynamicsReference()["live"]![2]!);
        (IAudioProcessor Processor, AudioProcessorStep Step)[] stages =
        [
            (new GateProcessor(), new(DynamicsProcessors.Gate)),
            (new CompressProcessor(), new(DynamicsProcessors.Compress, null, [KeyValuePair.Create("detector", "peak")])),
            (new CompressProcessor(), new(DynamicsProcessors.Compress, null, [KeyValuePair.Create("link", "false")])),
            (new LimitProcessor(), new(DynamicsProcessors.Limit)),
            (new LimitProcessor(), new(DynamicsProcessors.Limit, null, [KeyValuePair.Create("lookahead", "0")])),
            (new AgcProcessor(), new(DynamicsProcessors.Agc, null, [KeyValuePair.Create("link", "false")])),
            (new LiveLoudnessProcessor(), new(DynamicsProcessors.LoudnessLive)),
        ];
        foreach (var (processor, step) in stages)
        {
            using var stage = processor.Open(step, 48000, 2);
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
            Check(allocated == 0, $"{step}: 60 blocks of 480 stereo samples allocated {allocated} bytes once open");
        }
    }

    internal static void AudioDynamicsRefusals(Device device)
    {
        _ = device;
        IAudioProcessor gate = new GateProcessor(), compress = new CompressProcessor(), limit = new LimitProcessor(), agc = new AgcProcessor(),
            loudness = new LoudnessProcessor(), loudnessLive = new LiveLoudnessProcessor(), peaknorm = new PeakProcessor();
        (IAudioProcessor Processor, AudioProcessorStep Step, string What, string[] Named)[] refused =
        [
            (compress, new("compress", null, [KeyValuePair.Create("ratio", "0.5")]), "a ratio under 1", ["compress ratio", "at least 1", "0.5"]),
            (compress, new("compress", null, [KeyValuePair.Create("makeup", "loud")]), "a makeup that is neither a number nor auto", ["compress makeup", "number", "loud"]),
            (compress, new("compress", null, [KeyValuePair.Create("detector", "avg")]), "an unknown detector", ["compress detector", "rms, peak", "avg"]),
            (compress, new("compress", "3"), "a threshold over full scale", ["compress", "at most 0 dBFS", "3"]),
            (limit, new("limit", "1"), "a ceiling over 0 dBFS", ["limit", "at most 0 dBFS", "1"]),
            (limit, new("limit", null, [KeyValuePair.Create("lookahead", "-1")]), "a negative look-ahead", ["limit lookahead", "0 to 1000 ms", "-1"]),
            (gate, new("gate", null, [KeyValuePair.Create("range", "-5")]), "a negative range", ["gate range", "at least 0 dB", "-5"]),
            (gate, new("gate", null, [KeyValuePair.Create("link", "maybe")]), "a link that is not true or false", ["gate link", "true or false", "maybe"]),
            (agc, new("agc", null, [KeyValuePair.Create("max", "-3")]), "a most gain under 0", ["agc max", "at least 0 dB"]),
            (agc, new("agc", null, [KeyValuePair.Create("depth", "3")]), "an unknown option", ["agc", "depth", "max, min, attack"]),
            (agc, new("agc", null, [KeyValuePair.Create("quiet", "3")]), "a silence level over full scale", ["agc quiet", "at most 0 dBFS"]),
            (loudness, new("loudness", null, [KeyValuePair.Create("overlap", "1")]), "an overlap of a whole block", ["loudness overlap", "0 to 0.99", "1"]),
            (loudness, new("loudness", null, [KeyValuePair.Create("meter", "rms")]), "an unknown meter", ["loudness meter", "true, sample, none", "rms"]),
            (loudnessLive, new("loudness-live", null, [KeyValuePair.Create("window", "0")]), "an empty window", ["loudness-live window", "above 0", "0"]),
            (peaknorm, new("peaknorm", null, [KeyValuePair.Create("oversample", "0")]), "no true-peak points", ["peaknorm oversample", "1 to 64", "0"]),
        ];
        foreach (var (processor, step, what, named) in refused)
        {
            ThrowsNaming<ArgumentException>(() => processor.Check(step), what, named);
        }

        ThrowsNaming<ArgumentException>(() => loudness.Apply(ProcessingTone(3000, 1, 1), new AudioProcessorStep("loudness")), "a rate too low for the K-weighting", "loudness", "3363.9 Hz", "3000 Hz");
        ThrowsNaming<ArgumentException>(() => loudnessLive.Open(new AudioProcessorStep("loudness-live", null, [KeyValuePair.Create("weighting", "rbj")]), 2000, 1),
            "a rate too low for RBJ's shelf", "loudness-live", "3000 Hz", "2000 Hz");
        ThrowsNaming<ArgumentException>(() => loudness.Apply(ProcessingTone(16000, 1, 1), new AudioProcessorStep("loudness", null, [KeyValuePair.Create("block", "0.01")])),
            "a block under one sample", "loudness", "under one sample");
    }
}
