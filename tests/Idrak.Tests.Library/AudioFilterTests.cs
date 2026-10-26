// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P1: the filters as live audio processors (gain, dc, highpass and lowpass of any order, bandpass, notch, dehum,
// eq, lowshelf, highshelf, presence, preemphasis, mono, resample), each against scipy's samples (data/audio/filters,
// written by tools/audio/make_audio_filters.py) and block by block against its whole-recording output.
internal abstract partial class Tests
{
    internal static readonly string[] FilterNames =
        ["gain", "dc", "highpass", "lowpass", "bandpass", "notch", "dehum", "eq", "lowshelf", "highshelf", "presence", "preemphasis", "mono", "resample"];

    internal static readonly (string Name, Action<Device> Run)[] AudioFilterGroup =
    [
        ("audio processing: filters: every filter (gain, dc, highpass and lowpass of orders 1 to 8, bandpass, notch, dehum, eq, lowshelf, highshelf, presence, preemphasis, mono, resample, and chains of them) gives scipy's samples within 1e-5 on tones, speech, noise, hum and a DC offset, mono and stereo; resample's Apply is Resampler.Resample and its stage gives those samples bit for bit", AudioFilterReference),
        ("audio processing: filters: every reference pipeline run block by block (blocks of 1, 7, 160 and 4,096 samples) gives Apply's samples bit for bit, at the rate and channels its last step gives; the in-place filters allocate nothing per block", AudioFilterBlocks),
        ("audio processing: filters: the registry names every filter as the library's and live; the text form round-trips (dehum=50 harmonics=4, presence=4, every filter in one pipeline, JSON), an option after a comma joins its step; highpass order=2 is the highpass bit for bit; mono and resample give their channels and rate, resample holds back no more than its latency", AudioFilterText),
        ("audio processing: filters: refused naming the cause: a missing value, a value or option out of range (with its range and unit), a frequency at or over half the rate (with the rate), a low edge over the high one, an eq or shelf without its gain, a shelf given slope and q", AudioFilterRefusals),
        ("audio processing: filters: cold and warm time per stage on 10 s of 16 kHz mono (the first call, the best of 20 after it; printed, not checked)", AudioFilterTimes),
    ];

    internal static JsonObject FilterCases() => JsonNode.Parse(File.ReadAllText(AudioData("filters/cases.json")))!.AsObject();

    internal static AudioData FilterSignal(JsonObject cases, string name)
    {
        var entry = cases["inputs"]![name]!;
        return new AudioData(ReadNpyFloat32(AudioData($"filters/signal-{name}.npy")), (int)entry["channels"]!, (int)entry["rate"]!);
    }

    internal static void AudioFilterReference(Device device)
    {
        _ = device;
        var cases = FilterCases();
        var largest = new SortedDictionary<string, double>(StringComparer.Ordinal);
        foreach (var item in cases["cases"]!.AsArray())
        {
            string name = (string)item!["name"]!, text = (string)item["pipeline"]!;
            var input = FilterSignal(cases, (string)item["input"]!);
            var expected = ReadNpyFloat32(AudioData($"filters/{name}.npy"));
            var pipeline = AudioPipeline.Parse(text);
            var output = pipeline.Apply(input);
            Check(output.SampleRate == (int)item["rate"]! && output.Channels == (int)item["channels"]! && output.Length == (int)item["length"]!,
                $"{name} ({text}): {output.Channels} channel(s) of {output.Length} at {output.SampleRate} Hz; scipy's {item["channels"]} of {item["length"]} at {item["rate"]} Hz");
            var samples = output.Samples;
            double gap = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                gap = Math.Max(gap, Math.Abs((double)expected[i] - samples[i]));
            }

            Check(gap <= 1e-5, $"{name} ({text}): off scipy's by up to {gap:G3}");
            string key = pipeline.Steps.Count == 1 ? pipeline.Steps[0].Name : "chains";
            largest[key] = Math.Max(largest.GetValueOrDefault(key), gap);
        }

        Check(FilterNames.All(largest.ContainsKey), $"every filter has a reference: {string.Join(", ", largest.Keys)}");
        Console.WriteLine("    largest gap to scipy: " + string.Join(", ", largest.Select(p => string.Create(CultureInfo.InvariantCulture, $"{p.Key} {p.Value:G2}"))));

        // Rule 71: the streaming resampler is the library's resampler, the same bits.
        foreach (var (signal, rate) in new[] { ("speech", 8000), ("tones", 44100), ("stereo", 12000), ("tones8k", 16000), ("noise", 22050) })
        {
            var input = FilterSignal(cases, signal);
            var direct = Resampler.Resample(input, rate);
            var applied = AudioProcessors.Get("resample").Apply(input, new AudioProcessorStep("resample", rate.ToString(CultureInfo.InvariantCulture)));
            var staged = AudioPipeline.Parse(string.Create(CultureInfo.InvariantCulture, $"resample={rate}")).Apply(input);
            Check(applied.Length == direct.Length && ProcessingSame(applied.Samples, direct.Samples), $"{signal} to {rate} Hz: Apply is Resampler.Resample");
            Check(staged.Length == direct.Length && ProcessingSame(staged.Samples, direct.Samples),
                $"{signal} to {rate} Hz: the stage gives Resampler.Resample's samples bit for bit ({staged.Length} against {direct.Length})");
        }
    }

    internal static void AudioFilterBlocks(Device device)
    {
        _ = device;
        var cases = FilterCases();
        foreach (var item in cases["cases"]!.AsArray())
        {
            string name = (string)item!["name"]!, text = (string)item["pipeline"]!;
            var input = FilterSignal(cases, (string)item["input"]!);
            var pipeline = AudioPipeline.Parse(text);
            var whole = pipeline.Apply(input);
            foreach (int size in new[] { 1, 7, 160, 4096 })
            {
                using var stage = pipeline.Open(input.SampleRate, input.Channels);
                var pieces = new List<AudioData>();
                foreach (var block in ProcessingBlocks(input, size))
                {
                    pieces.Add(stage.Process(block));
                }

                pieces.Add(stage.Flush());
                Check(pieces.All(p => p.SampleRate == stage.SampleRate && p.Channels == stage.Channels), $"{name}, blocks of {size}: every piece at the stage's rate and channels");
                var joined = ProcessingJoin(pieces, stage.Channels, stage.SampleRate);
                Check(stage.SampleRate == whole.SampleRate && stage.Channels == whole.Channels && joined.Length == whole.Length && ProcessingSame(joined.Samples, whole.Samples),
                    $"{name} ({text}), blocks of {size}: {joined.Length} samples at {stage.SampleRate} Hz against Apply's {whole.Length} at {whole.SampleRate} Hz, the same bits");
            }
        }

        // The filters that write in place keep their memory in arrays made when opened: nothing is allocated per block.
        var sound = ProcessingTone(16000, 0.5, 2, noise: 0.05f, dc: 0.1f);
        var blocks = ProcessingBlocks(sound, 160).ToArray();
        foreach (string text in new[] { "gain=-3", "dc", "highpass", "highpass=80 order=5", "lowpass=3000 order=4", "bandpass", "notch=1000", "dehum harmonics=8",
                     "eq=1000 db=3", "lowshelf=200 db=3", "highshelf=4000 db=-3 q=0.8", "presence", "preemphasis" })
        {
            var step = AudioPipeline.Parse(text).Steps[0];
            var processor = AudioProcessors.Get(step.Name);
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
            Check(allocated == 0, $"{text}: {allocated} bytes allocated over {blocks.Length} blocks (none expected)");
        }
    }

    internal static void AudioFilterText(Device device)
    {
        _ = device;
        foreach (string name in FilterNames)
        {
            Check(AudioProcessors.Names.Contains(name) && AudioProcessors.Origin(name) == Overrides.Library && AudioProcessors.Get(name).Live,
                $"{name} is the library's and live (registered: {string.Join(", ", AudioProcessors.Names)})");
            var processor = AudioProcessors.Get(name);
            Check(processor.Parameters.All(p => p.Help is { Length: > 0 }) && processor.Summary.Length > 0 && processor.Label != name,
                $"{name} describes its inputs, its summary and its label");
        }

        Check(AudioProcessors.Describe().Contains("presence: the speech clarity control", StringComparison.Ordinal), $"presence is the speech clarity control: {AudioProcessors.Describe()}");
        Check(AudioProcessors.Get("dehum").Keys.SequenceEqual(["harmonics", "q"]) && AudioProcessors.Get("bandpass").Keys.SequenceEqual(["low", "high", "order"])
              && AudioProcessors.Get("highpass").Keys.SequenceEqual(["order"]) && AudioProcessors.Get("lowshelf").Keys.SequenceEqual(["db", "slope", "q"]),
            "the option keys follow the parameters");

        var pipeline = AudioPipeline.Parse("dehum=50 harmonics=4, presence=4");
        Check(pipeline.ToString() == "dehum=50 harmonics=4, presence=4" && pipeline.Steps.Count == 2 && pipeline.Steps[0].Integer("harmonics", 0) == 4 && pipeline.Steps[1].Number() == 4 && pipeline.Live,
            $"the text: {pipeline}");
        Check(AudioPipeline.Parse("dehum=60, harmonics=3, q=20, presence").ToString() == "dehum=60 harmonics=3 q=20, presence", "options after commas join the step before them");
        const string Every = "gain=-3, dc pole=0.99, highpass=80 order=4, lowpass=7000 order=3, bandpass high=3000 low=200 order=2, notch=1000 q=10, dehum=60 harmonics=3, "
                             + "eq=1500 db=2 q=1.5, lowshelf=120 db=-3, highshelf=6000 db=2 slope=0.7, presence=4 frequency=3500, preemphasis=0.95, mono, resample=8000";
        var every = AudioPipeline.Parse(Every);
        Check(every.ToString() == Every && every.Live && AudioPipeline.Parse(every.ToString()).Equals(every) && AudioPipeline.FromJson(every.ToJson()).Equals(every),
            $"every filter in one pipeline round-trips through text and JSON: {every}");
        var json = JsonNode.Parse("""[{"name": "dehum", "value": 60, "harmonics": 3}, {"name": "presence", "value": 4, "frequency": 3500}, "resample=8000 width=8"]""");
        Check(AudioPipeline.FromJson(json).ToString() == "dehum=60 harmonics=3, presence=4 frequency=3500, resample=8000 width=8", $"JSON objects: {AudioPipeline.FromJson(json)}");
        var stereo = ProcessingTone(16000, 0.3, 2, noise: 0.05f, dc: 0.1f);
        var result = every.Apply(stereo);
        Check(result.Channels == 1 && result.SampleRate == 8000 && result.Length == 2400 && result.Samples.All(float.IsFinite),
            $"every filter in one pipeline: {result.Channels} channel(s) of {result.Length} at {result.SampleRate} Hz");

        Check(ProcessingSame(AudioPipeline.Parse("highpass=60 order=2").Apply(stereo).Samples, AudioPipeline.Parse("highpass").Apply(stereo).Samples), "highpass order=2 is the highpass");
        Check(ProcessingSame(AudioPipeline.Parse("resample=16000").Apply(stereo).Samples, stereo.Samples) && ProcessingSame(AudioPipeline.Parse("preemphasis=0").Apply(stereo).Samples, stereo.Samples)
              && ProcessingSame(AudioPipeline.Parse("gain=0, highpass=0, mono").Apply(ProcessingTone(16000, 0.1, 1)).Samples, ProcessingTone(16000, 0.1, 1).Samples),
            "a resample to the same rate, a pre-emphasis of 0, a gain of 0 dB, a high-pass at 0 Hz and mono on mono sound leave the samples");

        using (var mono = AudioPipeline.Parse("mono").Open(16000, 3))
        {
            Check(mono is { Channels: 1, SampleRate: 16000, Latency: 0 }, "mono gives one channel");
            var three = new AudioData([1f, 2f, 3f, 4f, 5f, 6f], 3, 16000);
            Check(mono.Process(three).Samples.SequenceEqual([3f, 4f]), "mono is the channels' mean");
        }

        using (var filter = AudioPipeline.Parse("highpass=100 order=6, dehum, presence").Open(16000, 2))
        {
            Check(filter is { Channels: 2, SampleRate: 16000, Latency: 0 }, "a filter keeps the rate and channels with no latency");
        }

        foreach (var (from, to) in new[] { (16000, 8000), (16000, 44100), (8000, 16000), (44100, 16000) })
        {
            using var stage = AudioPipeline.Parse(string.Create(CultureInfo.InvariantCulture, $"resample={to}")).Open(from, 1);
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
        (string Text, string[] Named)[] parsed =
        [
            ("gain", ["gain needs a value", "-120 to 60 dB"]),
            ("gain=200", ["gain is -120 to 60 dB", "200"]),
            ("dc=3", ["dc takes no value", "3"]),
            ("dc pole=1", ["dc pole is above 0 to 0.999999", "not 1"]),
            ("highpass=100 order=0", ["highpass order is 1 to 16", "not 0"]),
            ("highpass order=2.5", ["highpass order takes a whole number", "2.5"]),
            ("lowpass", ["lowpass needs a value", "above 0 Hz"]),
            ("lowpass=0", ["lowpass is above 0 Hz", "not 0"]),
            ("lowpass=1000 order=17", ["lowpass order is 1 to 16", "17"]),
            ("bandpass low=3000 high=300", ["bandpass", "low edge 3000 Hz", "high edge 300 Hz"]),
            ("bandpass high=-5", ["bandpass high is above 0 Hz", "-5"]),
            ("notch", ["notch needs a value"]),
            ("notch=50 q=0", ["notch q is above 0 to 1000", "not 0"]),
            ("dehum=0", ["dehum is above 0 to 1000 Hz", "not 0"]),
            ("dehum harmonics=0", ["dehum harmonics is 1 to 100", "not 0"]),
            ("dehum width=3", ["dehum", "width", "harmonics, q"]),
            ("eq=1000", ["eq needs db=DECIBELS", "eq=1000"]),
            ("eq=1000 db=60", ["eq db is -48 to 48 dB", "60"]),
            ("eq db=3", ["eq needs a value", "FREQUENCY"]),
            ("lowshelf=200", ["lowshelf needs db=DECIBELS"]),
            ("lowshelf=200 db=3 slope=0.5 q=1", ["lowshelf takes slope or q, not both"]),
            ("highshelf=200 db=3 slope=2", ["highshelf slope is above 0 to 1", "2"]),
            ("presence=40", ["presence is -24 to 24 dB", "40"]),
            ("presence q=0", ["presence q is above 0 to 100", "not 0"]),
            ("preemphasis=1.5", ["preemphasis is 0 to 1", "1.5"]),
            ("mono=2", ["mono takes no value", "2"]),
            ("mono fast=1", ["mono takes no options", "fast"]),
            ("resample", ["resample needs a value", "1 to 1000000 Hz"]),
            ("resample=0", ["resample is 1 to 1000000 Hz", "not 0"]),
            ("resample=8000.5", ["resample takes a whole number", "8000.5"]),
            ("resample=16000 width=0", ["resample width is 1 to 64", "not 0"]),
            ("resample=16000 rolloff=0", ["resample rolloff is above 0 to 1", "not 0"]),
        ];
        foreach (var (text, named) in parsed)
        {
            ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse(text), $"'{text}'", named);
        }

        (string Text, int Rate, string[] Named)[] opened =
        [
            ("highpass=8000", 16000, ["highpass: the cutoff 8000 Hz", "half the rate (16000 Hz)"]),
            ("highpass=5000 order=4", 8000, ["highpass: the cutoff 5000 Hz", "half the rate (8000 Hz)"]),
            ("lowpass=8000", 16000, ["lowpass: the cutoff 8000 Hz", "half the rate (16000 Hz)"]),
            ("bandpass high=9000", 16000, ["bandpass: the high edge 9000 Hz", "half the rate (16000 Hz)"]),
            ("notch=4000", 8000, ["notch: the frequency 4000 Hz", "half the rate (8000 Hz)"]),
            ("dehum=600", 1000, ["dehum: the mains frequency 600 Hz", "half the rate (1000 Hz)"]),
            ("eq=9000 db=3", 16000, ["eq: the frequency 9000 Hz", "half the rate (16000 Hz)"]),
            ("lowshelf=4000 db=3", 8000, ["lowshelf: the frequency 4000 Hz", "half the rate (8000 Hz)"]),
            ("highshelf=12000 db=3", 22050, ["highshelf: the frequency 12000 Hz", "half the rate (22050 Hz)"]),
            ("presence", 6000, ["presence: the frequency 3000 Hz", "half the rate (6000 Hz)"]),
            ("presence frequency=5000", 8000, ["presence: the frequency 5000 Hz", "half the rate (8000 Hz)"]),
        ];
        foreach (var (text, rate, named) in opened)
        {
            var pipeline = AudioPipeline.Parse(text);
            ThrowsNaming<ArgumentException>(() => pipeline.Open(rate, 1), $"'{text}' at {rate} Hz", named);
            ThrowsNaming<ArgumentException>(() => pipeline.Apply(ProcessingTone(rate, 0.05, 1)), $"'{text}' applied at {rate} Hz", named);
        }

        // Opened directly (not through a pipeline), a step is still checked.
        ThrowsNaming<ArgumentException>(() => AudioProcessors.Get("eq").Open(new AudioProcessorStep("eq", "1000"), 16000, 1), "eq opened without its gain", "eq needs db=DECIBELS");
        ThrowsNaming<ArgumentException>(() => AudioProcessors.Get("resample").Open(new AudioProcessorStep("resample", "-1"), 16000, 1), "resample opened at -1 Hz", "resample is 1 to 1000000 Hz");

        // The harmonics at or over half the rate are left out (dehum=1000 harmonics=10 at 8 kHz: 1 to 3 kHz), not refused.
        var tone = ProcessingTone(8000, 0.2, 1);
        Check(AudioPipeline.Parse("dehum=1000 harmonics=10").Apply(tone).Samples.All(float.IsFinite), "dehum leaves out the harmonics past half the rate");
    }

    internal static void AudioFilterTimes(Device device)
    {
        _ = device;
        var mono = ProcessingTone(16000, 10, 1, noise: 0.05f, dc: 0.1f, bursts: true);
        var stereo = ProcessingTone(16000, 10, 2, noise: 0.05f, dc: 0.1f, bursts: true);
        var lines = new List<string>();
        foreach (string text in new[] { "gain=-3", "dc", "highpass", "highpass=80 order=8", "lowpass=4000 order=4", "bandpass", "notch=1000", "dehum", "dehum harmonics=20",
                     "eq=1000 db=3", "lowshelf=200 db=3", "highshelf=4000 db=3", "presence", "preemphasis", "mono", "resample=8000", "resample=44100",
                     "dc, highpass=80, dehum, presence=4, gain=-3" })
        {
            var pipeline = AudioPipeline.Parse(text);
            var sound = text == "mono" ? stereo : mono;
            var watch = Stopwatch.StartNew();
            _ = pipeline.Apply(sound);
            double cold = watch.Elapsed.TotalMilliseconds;
            var warm = new List<double>();
            for (int i = 0; i < 20; i++)
            {
                watch.Restart();
                _ = pipeline.Apply(sound);
                warm.Add(watch.Elapsed.TotalMilliseconds);
            }

            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{text}{(text == "mono" ? " (stereo)" : "")}: cold {cold:F2} ms, warm {warm.Min():F2} ms"));
        }

        Console.WriteLine("    10 s of 16 kHz mono, Apply (a copy and the stage): " + string.Join("; ", lines));
    }
}
