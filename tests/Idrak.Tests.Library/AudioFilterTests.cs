// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P1: the filters as live audio processors (gain, dc, highpass and lowpass of any order, bandpass, notch, dehum,
// eq, lowshelf, highshelf, presence, preemphasis, mono, resample), each reached directly with its step: what they are,
// what they refuse, the resampler's samples and the time per stage.
internal abstract partial class Tests
{
    internal static readonly string[] FilterNames =
        ["gain", "dc", "highpass", "lowpass", "bandpass", "notch", "dehum", "eq", "lowshelf", "highshelf", "presence", "preemphasis", "mono", "resample"];

    internal static readonly (string Name, Action<Device> Run)[] AudioFilterGroup =
    [
        ("audio processing: filters: resample's Apply is Resampler.Resample and its stage gives those samples bit for bit", AudioFilterResample),
        ("audio processing: filters: the in-place filters allocate nothing per block", AudioFilterBlocks),
        ("audio processing: filters: every filter is live and describes its inputs and its label, the option keys follow the parameters; highpass order=2 is the highpass bit for bit; a resample to the same rate, a pre-emphasis of 0, a gain of 0 dB, a high-pass at 0 Hz and mono on mono sound leave the samples; a filter keeps the rate and channels with no latency; mono and resample give their channels and rate, resample holds back no more than its latency", AudioFilterKinds),
        ("audio processing: filters: refused naming the cause: a missing value, a value or option out of range (with its range and unit), a frequency at or over half the rate (with the rate), a low edge over the high one, an eq or shelf without its gain, a shelf given slope and q", AudioFilterRefusals),
        ("audio processing: filters: cold and warm time per stage on 10 s of 16 kHz mono (the first call, the best of 20 after it; printed, not checked)", AudioFilterTimes),
    ];

    internal static JsonObject FilterCases() => JsonNode.Parse(File.ReadAllText(AudioData("filters/cases.json")))!.AsObject();

    internal static AudioData FilterSignal(JsonObject cases, string name)
    {
        var entry = cases["inputs"]![name]!;
        return new AudioData(ReadNpyFloat32(AudioData($"filters/signal-{name}.npy")), (int)entry["channels"]!, (int)entry["rate"]!);
    }

    internal static void AudioFilterResample(Device device)
    {
        _ = device;
        var cases = FilterCases();
        IAudioProcessor resample = new ResampleProcessor();

        // Rule 71: the streaming resampler is the library's resampler, the same bits.
        foreach (var (signal, rate) in new[] { ("speech", 8000), ("tones", 44100), ("stereo", 12000), ("tones8k", 16000), ("noise", 22050) })
        {
            var input = FilterSignal(cases, signal);
            var direct = Resampler.Resample(input, rate);
            var step = new AudioProcessorStep(FilterProcessors.Resample, rate.ToString(CultureInfo.InvariantCulture));
            var applied = resample.Apply(input, step);
            AudioData staged;
            using (var stage = resample.Open(step, input.SampleRate, input.Channels))
            {
                staged = AudioStages.Run(stage, AudioStages.Copy(input), stage.Latency);
            }

            Check(applied.Length == direct.Length && ProcessingSame(applied.Samples, direct.Samples), $"{signal} to {rate} Hz: Apply is Resampler.Resample");
            Check(staged.Length == direct.Length && ProcessingSame(staged.Samples, direct.Samples),
                $"{signal} to {rate} Hz: the stage gives Resampler.Resample's samples bit for bit ({staged.Length} against {direct.Length})");
        }
    }

    internal static void AudioFilterBlocks(Device device)
    {
        _ = device;

        // The filters that write in place keep their memory in arrays made when opened: nothing is allocated per block.
        var sound = ProcessingTone(16000, 0.5, 2, noise: 0.05f, dc: 0.1f);
        var blocks = ProcessingBlocks(sound, 160).ToArray();
        (IAudioProcessor Processor, AudioProcessorStep Step)[] filters =
        [
            (new GainProcessor(), new(FilterProcessors.Gain, "-3")),
            (new DcProcessor(), new(FilterProcessors.Dc)),
            (new HighPassProcessor(), new(AudioProcessors.HighPass)),
            (new HighPassProcessor(), new(AudioProcessors.HighPass, "80", [KeyValuePair.Create("order", "5")])),
            (new LowPassProcessor(), new(FilterProcessors.LowPass, "3000", [KeyValuePair.Create("order", "4")])),
            (new BandPassProcessor(), new(FilterProcessors.BandPass)),
            (new NotchProcessor(), new(FilterProcessors.Notch, "1000")),
            (new DehumProcessor(), new(FilterProcessors.Dehum, null, [KeyValuePair.Create("harmonics", "8")])),
            (new PeakingProcessor(), new(FilterProcessors.Eq, "1000", [KeyValuePair.Create("db", "3")])),
            (new ShelfProcessor(high: false), new(FilterProcessors.LowShelf, "200", [KeyValuePair.Create("db", "3")])),
            (new ShelfProcessor(high: true), new(FilterProcessors.HighShelf, "4000", [KeyValuePair.Create("db", "-3"), KeyValuePair.Create("q", "0.8")])),
            (new PresenceProcessor(), new(FilterProcessors.Presence)),
            (new PreEmphasisProcessor(), new(FilterProcessors.PreEmphasis)),
        ];
        foreach (var (processor, step) in filters)
        {
            using (var warm = processor.Open(step, 16000, 2))
            {
                Array.ForEach(blocks, b => warm.Process(b));
            }

            using var stage = processor.Open(step, 16000, 2);
            long before = GC.GetAllocatedBytesForCurrentThread();
            foreach (var block in blocks)
            {
                stage.Process(block);
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated == 0, $"{step}: {allocated} bytes allocated over {blocks.Length} blocks (none expected)");
        }
    }

    internal static void AudioFilterKinds(Device device)
    {
        _ = device;
        IAudioProcessor gain = new GainProcessor(), highpass = new HighPassProcessor(), bandpass = new BandPassProcessor(), dehum = new DehumProcessor(),
            lowshelf = new ShelfProcessor(high: false), presence = new PresenceProcessor(), preemphasis = new PreEmphasisProcessor(), mono = new MonoProcessor(),
            resample = new ResampleProcessor();
        IAudioProcessor[] filters = [gain, new DcProcessor(), highpass, new LowPassProcessor(), bandpass, new NotchProcessor(), dehum, new PeakingProcessor(), lowshelf,
            new ShelfProcessor(high: true), presence, preemphasis, mono, resample];
        Check(filters.Select(f => f.Name).SequenceEqual(FilterNames), $"the filters: {string.Join(", ", filters.Select(f => f.Name))}");
        foreach (var processor in filters)
        {
            Check(processor.Live && processor.Parameters.All(p => p.Help is { Length: > 0 }) && processor.Label != processor.Name,
                $"{processor.Name} is live and describes its inputs and its label");
        }

        Check(dehum.Keys.SequenceEqual(["harmonics", "q"]) && bandpass.Keys.SequenceEqual(["low", "high", "order"])
              && highpass.Keys.SequenceEqual(["order"]) && lowshelf.Keys.SequenceEqual(["db", "slope", "q"]),
            "the option keys follow the parameters");

        var stereo = ProcessingTone(16000, 0.3, 2, noise: 0.05f, dc: 0.1f);
        Check(ProcessingSame(highpass.Apply(stereo, new(AudioProcessors.HighPass, "60", [KeyValuePair.Create("order", "2")])).Samples,
                highpass.Apply(stereo, new(AudioProcessors.HighPass)).Samples),
            "highpass order=2 is the highpass");
        var tone = ProcessingTone(16000, 0.1, 1);
        Check(ProcessingSame(resample.Apply(stereo, new(FilterProcessors.Resample, "16000")).Samples, stereo.Samples)
              && ProcessingSame(preemphasis.Apply(stereo, new(FilterProcessors.PreEmphasis, "0")).Samples, stereo.Samples)
              && ProcessingSame(gain.Apply(tone, new(FilterProcessors.Gain, "0")).Samples, tone.Samples)
              && ProcessingSame(highpass.Apply(tone, new(AudioProcessors.HighPass, "0")).Samples, tone.Samples)
              && ProcessingSame(mono.Apply(tone, new(FilterProcessors.Mono)).Samples, tone.Samples),
            "a resample to the same rate, a pre-emphasis of 0, a gain of 0 dB, a high-pass at 0 Hz and mono on mono sound leave the samples");

        using (var stage = mono.Open(new(FilterProcessors.Mono), 16000, 3))
        {
            Check(stage is { Channels: 1, SampleRate: 16000, Latency: 0 }, "mono gives one channel");
            var three = new AudioData([1f, 2f, 3f, 4f, 5f, 6f], 3, 16000);
            Check(stage.Process(three).Samples.SequenceEqual([3f, 4f]), "mono is the channels' mean");
        }

        foreach (var (processor, step) in new (IAudioProcessor, AudioProcessorStep)[]
                 {
                     (highpass, new(AudioProcessors.HighPass, "100", [KeyValuePair.Create("order", "6")])), (dehum, new(FilterProcessors.Dehum)),
                     (presence, new(FilterProcessors.Presence)),
                 })
        {
            using var filter = processor.Open(step, 16000, 2);
            Check(filter is { Channels: 2, SampleRate: 16000, Latency: 0 }, $"{step} keeps the rate and channels with no latency");
        }

        foreach (var (from, to) in new[] { (16000, 8000), (16000, 44100), (8000, 16000), (44100, 16000) })
        {
            using var stage = resample.Open(new(FilterProcessors.Resample, to.ToString(CultureInfo.InvariantCulture)), from, 1);
            Check(stage.SampleRate == to && stage.Channels == 1 && stage.Latency == 0 && stage.Delay > 0, $"resample {from} to {to}: {stage.SampleRate} Hz, latency {stage.Latency} (not shifted), held back {stage.Delay}");
            var sound = ProcessingTone(from, 0.5, 1);
            long taken = 0, given = 0, held = 0;
            foreach (var block in ProcessingBlocks(sound, 97))
            {
                taken += block.Length;
                given += stage.Process(block).Length;
                held = Math.Max(held, (taken * to + from - 1) / from - given);
            }

            Check(held <= stage.Delay && held > stage.Delay / 2, $"resample {from} to {to}: held back up to {held} samples, its delay {stage.Delay}");
        }
    }

    internal static void AudioFilterRefusals(Device device)
    {
        _ = device;
        IAudioProcessor gain = new GainProcessor(), dc = new DcProcessor(), highpass = new HighPassProcessor(), lowpass = new LowPassProcessor(),
            bandpass = new BandPassProcessor(), notch = new NotchProcessor(), dehum = new DehumProcessor(), eq = new PeakingProcessor(),
            lowshelf = new ShelfProcessor(high: false), highshelf = new ShelfProcessor(high: true), presence = new PresenceProcessor(),
            preemphasis = new PreEmphasisProcessor(), mono = new MonoProcessor(), resample = new ResampleProcessor();
        (IAudioProcessor Processor, AudioProcessorStep Step, string[] Named)[] checkedSteps =
        [
            (gain, new("gain"), ["gain needs a value", "-120 to 60 dB"]),
            (gain, new("gain", "200"), ["gain is -120 to 60 dB", "200"]),
            (dc, new("dc", "3"), ["dc takes no value", "3"]),
            (dc, new("dc", null, [KeyValuePair.Create("pole", "1")]), ["dc pole is above 0 to 0.999999", "not 1"]),
            (highpass, new("highpass", "100", [KeyValuePair.Create("order", "0")]), ["highpass order is 1 to 16", "not 0"]),
            (highpass, new("highpass", null, [KeyValuePair.Create("order", "2.5")]), ["highpass order takes a whole number", "2.5"]),
            (lowpass, new("lowpass"), ["lowpass needs a value", "above 0 Hz"]),
            (lowpass, new("lowpass", "0"), ["lowpass is above 0 Hz", "not 0"]),
            (lowpass, new("lowpass", "1000", [KeyValuePair.Create("order", "17")]), ["lowpass order is 1 to 16", "17"]),
            (bandpass, new("bandpass", null, [KeyValuePair.Create("low", "3000"), KeyValuePair.Create("high", "300")]), ["bandpass", "low edge 3000 Hz", "high edge 300 Hz"]),
            (bandpass, new("bandpass", null, [KeyValuePair.Create("high", "-5")]), ["bandpass high is above 0 Hz", "-5"]),
            (notch, new("notch"), ["notch needs a value"]),
            (notch, new("notch", "50", [KeyValuePair.Create("q", "0")]), ["notch q is above 0 to 1000", "not 0"]),
            (dehum, new("dehum", "0"), ["dehum is above 0 to 1000 Hz", "not 0"]),
            (dehum, new("dehum", null, [KeyValuePair.Create("harmonics", "0")]), ["dehum harmonics is 1 to 100", "not 0"]),
            (dehum, new("dehum", null, [KeyValuePair.Create("width", "3")]), ["dehum", "width", "harmonics, q"]),
            (eq, new("eq", "1000"), ["eq needs db=DECIBELS", "eq=1000"]),
            (eq, new("eq", "1000", [KeyValuePair.Create("db", "60")]), ["eq db is -48 to 48 dB", "60"]),
            (eq, new("eq", null, [KeyValuePair.Create("db", "3")]), ["eq needs a value", "FREQUENCY"]),
            (lowshelf, new("lowshelf", "200"), ["lowshelf needs db=DECIBELS"]),
            (lowshelf, new("lowshelf", "200", [KeyValuePair.Create("db", "3"), KeyValuePair.Create("slope", "0.5"), KeyValuePair.Create("q", "1")]), ["lowshelf takes slope or q, not both"]),
            (highshelf, new("highshelf", "200", [KeyValuePair.Create("db", "3"), KeyValuePair.Create("slope", "2")]), ["highshelf slope is above 0 to 1", "2"]),
            (presence, new("presence", "40"), ["presence is -24 to 24 dB", "40"]),
            (presence, new("presence", null, [KeyValuePair.Create("q", "0")]), ["presence q is above 0 to 100", "not 0"]),
            (preemphasis, new("preemphasis", "1.5"), ["preemphasis is 0 to 1", "1.5"]),
            (mono, new("mono", "2"), ["mono takes no value", "2"]),
            (mono, new("mono", null, [KeyValuePair.Create("fast", "1")]), ["mono takes no options", "fast"]),
            (resample, new("resample"), ["resample needs a value", "1 to 1000000 Hz"]),
            (resample, new("resample", "0"), ["resample is 1 to 1000000 Hz", "not 0"]),
            (resample, new("resample", "8000.5"), ["resample takes a whole number", "8000.5"]),
            (resample, new("resample", "16000", [KeyValuePair.Create("width", "0")]), ["resample width is 1 to 64", "not 0"]),
            (resample, new("resample", "16000", [KeyValuePair.Create("rolloff", "0")]), ["resample rolloff is above 0 to 1", "not 0"]),
        ];
        foreach (var (processor, step, named) in checkedSteps)
        {
            ThrowsNaming<ArgumentException>(() => processor.Check(step), $"'{step}'", named);
        }

        (IAudioProcessor Processor, AudioProcessorStep Step, int Rate, string[] Named)[] opened =
        [
            (highpass, new("highpass", "8000"), 16000, ["highpass: the cutoff 8000 Hz", "half the rate (16000 Hz)"]),
            (highpass, new("highpass", "5000", [KeyValuePair.Create("order", "4")]), 8000, ["highpass: the cutoff 5000 Hz", "half the rate (8000 Hz)"]),
            (lowpass, new("lowpass", "8000"), 16000, ["lowpass: the cutoff 8000 Hz", "half the rate (16000 Hz)"]),
            (bandpass, new("bandpass", null, [KeyValuePair.Create("high", "9000")]), 16000, ["bandpass: the high edge 9000 Hz", "half the rate (16000 Hz)"]),
            (notch, new("notch", "4000"), 8000, ["notch: the frequency 4000 Hz", "half the rate (8000 Hz)"]),
            (dehum, new("dehum", "600"), 1000, ["dehum: the mains frequency 600 Hz", "half the rate (1000 Hz)"]),
            (eq, new("eq", "9000", [KeyValuePair.Create("db", "3")]), 16000, ["eq: the frequency 9000 Hz", "half the rate (16000 Hz)"]),
            (lowshelf, new("lowshelf", "4000", [KeyValuePair.Create("db", "3")]), 8000, ["lowshelf: the frequency 4000 Hz", "half the rate (8000 Hz)"]),
            (highshelf, new("highshelf", "12000", [KeyValuePair.Create("db", "3")]), 22050, ["highshelf: the frequency 12000 Hz", "half the rate (22050 Hz)"]),
            (presence, new("presence"), 6000, ["presence: the frequency 3000 Hz", "half the rate (6000 Hz)"]),
            (presence, new("presence", null, [KeyValuePair.Create("frequency", "5000")]), 8000, ["presence: the frequency 5000 Hz", "half the rate (8000 Hz)"]),
        ];
        foreach (var (processor, step, rate, named) in opened)
        {
            ThrowsNaming<ArgumentException>(() => processor.Open(step, rate, 1), $"'{step}' at {rate} Hz", named);
            ThrowsNaming<ArgumentException>(() => processor.Apply(ProcessingTone(rate, 0.05, 1), step), $"'{step}' applied at {rate} Hz", named);
        }

        // Opened without Check first, a step is still checked.
        ThrowsNaming<ArgumentException>(() => eq.Open(new AudioProcessorStep("eq", "1000"), 16000, 1), "eq opened without its gain", "eq needs db=DECIBELS");
        ThrowsNaming<ArgumentException>(() => resample.Open(new AudioProcessorStep("resample", "-1"), 16000, 1), "resample opened at -1 Hz", "resample is 1 to 1000000 Hz");

        // The harmonics at or over half the rate are left out (dehum=1000 harmonics=10 at 8 kHz: 1 to 3 kHz), not refused.
        var tone = ProcessingTone(8000, 0.2, 1);
        Check(dehum.Apply(tone, new AudioProcessorStep("dehum", "1000", [KeyValuePair.Create("harmonics", "10")])).Samples.All(float.IsFinite), "dehum leaves out the harmonics past half the rate");
    }

    internal static void AudioFilterTimes(Device device)
    {
        _ = device;
        var mono = ProcessingTone(16000, 10, 1, noise: 0.05f, dc: 0.1f, bursts: true);
        var stereo = ProcessingTone(16000, 10, 2, noise: 0.05f, dc: 0.1f, bursts: true);
        var lines = new List<string>();
        (IAudioProcessor Processor, AudioProcessorStep Step)[] stages =
        [
            (new GainProcessor(), new(FilterProcessors.Gain, "-3")),
            (new DcProcessor(), new(FilterProcessors.Dc)),
            (new HighPassProcessor(), new(AudioProcessors.HighPass)),
            (new HighPassProcessor(), new(AudioProcessors.HighPass, "80", [KeyValuePair.Create("order", "8")])),
            (new LowPassProcessor(), new(FilterProcessors.LowPass, "4000", [KeyValuePair.Create("order", "4")])),
            (new BandPassProcessor(), new(FilterProcessors.BandPass)),
            (new NotchProcessor(), new(FilterProcessors.Notch, "1000")),
            (new DehumProcessor(), new(FilterProcessors.Dehum)),
            (new DehumProcessor(), new(FilterProcessors.Dehum, null, [KeyValuePair.Create("harmonics", "20")])),
            (new PeakingProcessor(), new(FilterProcessors.Eq, "1000", [KeyValuePair.Create("db", "3")])),
            (new ShelfProcessor(high: false), new(FilterProcessors.LowShelf, "200", [KeyValuePair.Create("db", "3")])),
            (new ShelfProcessor(high: true), new(FilterProcessors.HighShelf, "4000", [KeyValuePair.Create("db", "3")])),
            (new PresenceProcessor(), new(FilterProcessors.Presence)),
            (new PreEmphasisProcessor(), new(FilterProcessors.PreEmphasis)),
            (new MonoProcessor(), new(FilterProcessors.Mono)),
            (new ResampleProcessor(), new(FilterProcessors.Resample, "8000")),
            (new ResampleProcessor(), new(FilterProcessors.Resample, "44100")),
        ];
        foreach (var (processor, step) in stages)
        {
            bool isMono = step.Name == FilterProcessors.Mono;
            var sound = isMono ? stereo : mono;
            var watch = Stopwatch.StartNew();
            _ = processor.Apply(sound, step);
            double cold = watch.Elapsed.TotalMilliseconds;
            var warm = new List<double>();
            for (int i = 0; i < 20; i++)
            {
                watch.Restart();
                _ = processor.Apply(sound, step);
                warm.Add(watch.Elapsed.TotalMilliseconds);
            }

            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{step}{(isMono ? " (stereo)" : "")}: cold {cold:F2} ms, warm {warm.Min():F2} ms"));
        }

        Console.WriteLine("    10 s of 16 kHz mono, Apply (a copy and the stage): " + string.Join("; ", lines));
    }
}
