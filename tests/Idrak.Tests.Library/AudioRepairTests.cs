// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P3: the noise and repair processors (denoise-live, declick, declip, deess, dereverb live; spectral-gate over
// the whole recording), each reached directly with its step: what they are, their latency, silence kept silent, their
// refusals, their allocations and their time.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioRepairGroup =
    [
        ("audio processing: repair: cold and warm time of each stage on 10 s of 16 kHz mono (reported, not asserted)", AudioRepairTiming),
        ("audio processing: repair: denoise-live, declick, declip, deess and dereverb are live, spectral-gate needs the whole recording; denoise's and spectral-gate's live alternative is denoise-live; every input has a help line and its default, every number its unit; each stage reports its look-ahead as latency", AudioRepairKinds),
        ("audio processing: repair: spectral-gate leaves silence silent and a recording shorter than two windows as it is; the live stages leave silence silent", AudioRepairSilence),
        ("audio processing: repair: a live stage allocates nothing per block once open", AudioRepairBlocks),
        ("audio processing: repair: refused naming the cause: gain frames under 1, an empty noise memory, a reduction over 1, a click threshold of 0, a clip level that is not a number or auto or is over 0 dBFS, a value to declip, an unknown de-esser mode, a de-esser frequency over half the rate, a T60 of 0, an unknown option", AudioRepairRefusals),
    ];

    internal static string RepairData(string relative) => AudioData(Path.Combine("repair", relative));

    internal static JsonObject RepairReference() => JsonNode.Parse(File.ReadAllText(RepairData("repair.json")))!.AsObject();

    internal static AudioData RepairInput(JsonNode entry) =>
        new(ReadNpyFloat32(RepairData((string)entry["input"]!)), (int)entry["channels"]!, (int)entry["rate"]!);

    internal static void AudioRepairTiming(Device device)
    {
        _ = device;
        var sound = ProcessingTone(16000, 10, 1, noise: 0.01f, bursts: true);
        (IAudioProcessor Processor, AudioProcessorStep Step)[] stages =
        [
            (new LiveDenoiseProcessor(), new(RepairProcessors.DenoiseLive)),
            (new SpectralGateProcessor(), new(RepairProcessors.SpectralGate)),
            (new DeclickProcessor(), new(RepairProcessors.Declick)),
            (new DeclipProcessor(), new(RepairProcessors.Declip)),
            (new DeessProcessor(), new(RepairProcessors.Deess)),
            (new DereverbProcessor(), new(RepairProcessors.Dereverb)),
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
            Console.WriteLine($"    {step,-14} 10 s of 16 kHz mono: cold {cold,7:F2} ms, warm {average,6:F2} ms ({10000 / average:N0}x real time)");
        }
    }

    internal static void AudioRepairKinds(Device device)
    {
        _ = device;
        IAudioProcessor denoiseLive = new LiveDenoiseProcessor(), declick = new DeclickProcessor(), declip = new DeclipProcessor(), deess = new DeessProcessor(),
            dereverb = new DereverbProcessor(), spectralGate = new SpectralGateProcessor(), denoise = new DenoiseProcessor();
        IAudioProcessor[] live = [denoiseLive, declick, declip, deess, dereverb];
        Check(live.All(p => p.Live) && !spectralGate.Live, "the live ones are live, spectral-gate needs the whole recording");
        Check(denoise.LiveAlternative == "denoise-live" && spectralGate.LiveAlternative == "denoise-live", "denoise's and spectral-gate's live alternative is denoise-live");
        foreach (var processor in live.Append(spectralGate))
        {
            Check(processor.Parameters.Count > 0 && processor.Parameters.All(p => p.Help is { Length: > 0 } && (!p.Optional || p.Default is not null)),
                $"{processor.Name}: every input has a help line and its default");
            Check(processor.Parameters.All(p => p.Input != AudioProcessorInput.Number || p.Unit is not null || p.Key is "reduce" or "quantile" or "ratio" or "q"),
                $"{processor.Name}: every number has its unit");
        }

        foreach (var (processor, step, rate, latency) in new (IAudioProcessor, AudioProcessorStep, int, int)[]
                 {
                     (denoiseLive, new(RepairProcessors.DenoiseLive), 16000, 511), (denoiseLive, new(RepairProcessors.DenoiseLive), 48000, 1023),
                     (denoiseLive, new(RepairProcessors.DenoiseLive, null, [KeyValuePair.Create("window", "0.016")]), 16000, 255), (dereverb, new(RepairProcessors.Dereverb), 16000, 511),
                     (declick, new(RepairProcessors.Declick), 16000, 320 + 32 + 32 - 1),
                     (declick, new(RepairProcessors.Declick, null, [KeyValuePair.Create("length", "1"), KeyValuePair.Create("order", "8"), KeyValuePair.Create("frame", "10")]), 48000, 480 + 48 + 16 - 1),
                     (declip, new(RepairProcessors.Declip), 16000, 320 + 80 + 12 - 1),
                     (declip, new(RepairProcessors.Declip, null, [KeyValuePair.Create("length", "2"), KeyValuePair.Create("order", "16"), KeyValuePair.Create("frame", "5")]), 48000, 240 + 96 + 16 - 1),
                     (deess, new(RepairProcessors.Deess), 16000, 0),
                 })
        {
            using var stage = processor.Open(step, rate, 2);
            Check(stage.Latency == latency, $"{step} at {rate} Hz: latency {stage.Latency}, expected {latency}");
        }
    }

    internal static void AudioRepairSilence(Device device)
    {
        _ = device;
        IAudioProcessor spectralGate = new SpectralGateProcessor();
        var silence = new AudioData(new float[16000], 1, 16000);
        var gate = new AudioProcessorStep(RepairProcessors.SpectralGate);
        Check(spectralGate.Apply(silence, gate).Samples.All(x => x == 0), "spectral-gate leaves silence silent");
        var shortSound = ProcessingTone(16000, 0.03, 1);
        Check(ReferenceEquals(spectralGate.Apply(shortSound, gate), shortSound), "spectral-gate leaves a recording shorter than two windows as it is");
        foreach (var processor in new IAudioProcessor[] { new LiveDenoiseProcessor(), new DeclickProcessor(), new DeclipProcessor(), new DeessProcessor(), new DereverbProcessor() })
        {
            var quiet = processor.Apply(silence, new AudioProcessorStep(processor.Name));
            Check(quiet.Samples.All(x => x == 0), $"{processor.Name} leaves silence silent (no click or clip found in it)");
        }
    }

    internal static void AudioRepairBlocks(Device device)
    {
        _ = device;
        var stereo = RepairReference()["cases"]!.AsArray().Select(c => RepairInput(c!)).First(s => s.Channels == 2);
        (IAudioProcessor Processor, AudioProcessorStep Step)[] stages =
        [
            (new LiveDenoiseProcessor(), new(RepairProcessors.DenoiseLive)),
            (new LiveDenoiseProcessor(), new(RepairProcessors.DenoiseLive, null, [KeyValuePair.Create("frames", "4"), KeyValuePair.Create("smooth", "0")])),
            (new DereverbProcessor(), new(RepairProcessors.Dereverb)),
            (new DeclickProcessor(), new(RepairProcessors.Declick)),
            (new DeclipProcessor(), new(RepairProcessors.Declip)),
            (new DeclipProcessor(), new(RepairProcessors.Declip, null, [KeyValuePair.Create("level", "-3"), KeyValuePair.Create("headroom", "3")])),
            (new DeessProcessor(), new(RepairProcessors.Deess)),
            (new DeessProcessor(), new(RepairProcessors.Deess, null, [KeyValuePair.Create("mode", "band"), KeyValuePair.Create("link", "false")])),
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

    internal static void AudioRepairRefusals(Device device)
    {
        _ = device;
        IAudioProcessor denoiseLive = new LiveDenoiseProcessor(), spectralGate = new SpectralGateProcessor(), declick = new DeclickProcessor(), declip = new DeclipProcessor(),
            deess = new DeessProcessor(), dereverb = new DereverbProcessor();
        (IAudioProcessor Processor, AudioProcessorStep Step, string What, string[] Named)[] refused =
        [
            (denoiseLive, new("denoise-live", null, [KeyValuePair.Create("frames", "0")]), "no gain frames", ["denoise-live frames", "1 to 64", "0"]),
            (denoiseLive, new("denoise-live", null, [KeyValuePair.Create("track", "0")]), "an empty noise memory", ["denoise-live track", "above 0", "0"]),
            (spectralGate, new("spectral-gate", null, [KeyValuePair.Create("reduce", "2")]), "a reduction over 1", ["spectral-gate reduce", "0 to 1", "2"]),
            (declick, new("declick", "0"), "a click threshold of 0", ["declick", "above 0", "0"]),
            (declick, new("declick", null, [KeyValuePair.Create("length", "0")]), "no click length", ["declick length", "above 0", "ms"]),
            (declip, new("declip", null, [KeyValuePair.Create("level", "loud")]), "a clip level that is neither a number nor auto", ["declip level", "number", "loud"]),
            (declip, new("declip", null, [KeyValuePair.Create("level", "3")]), "a clip level over full scale", ["declip level", "at most 0 dBFS", "3"]),
            (declip, new("declip", "3"), "a value to declip", ["declip", "takes no value"]),
            (declip, new("declip", null, [KeyValuePair.Create("run", "0")]), "a run of no samples", ["declip run", "1 to 1000"]),
            (deess, new("deess", null, [KeyValuePair.Create("mode", "wide")]), "an unknown de-esser mode", ["deess mode", "high, band", "wide"]),
            (deess, new("deess", null, [KeyValuePair.Create("ratio", "0.5")]), "a ratio under 1", ["deess ratio", "at least 1"]),
            (dereverb, new("dereverb", null, [KeyValuePair.Create("t60", "0")]), "a T60 of 0", ["dereverb t60", "above 0", "0"]),
            (dereverb, new("dereverb", null, [KeyValuePair.Create("depth", "3")]), "an unknown option", ["dereverb", "depth", "t60, delay"]),
        ];
        foreach (var (processor, step, what, named) in refused)
        {
            ThrowsNaming<ArgumentException>(() => processor.Check(step), what, named);
        }

        ThrowsNaming<ArgumentException>(() => deess.Open(new AudioProcessorStep("deess", null, [KeyValuePair.Create("frequency", "9000")]), 16000, 1),
            "a de-esser frequency over half the rate", "deess", "9000 Hz", "16000 Hz");
    }
}
