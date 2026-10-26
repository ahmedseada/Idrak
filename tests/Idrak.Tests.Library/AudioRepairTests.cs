// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P3: the noise and repair processors (denoise-live, declick, declip, deess, dereverb live; spectral-gate over
// the whole recording) against tools/audio/make_audio_repair.py's reference (the same equations frame by frame or sample
// by sample), their effects on made signals, block by block bit for bit, their text and refusals, and their time.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioRepairGroup =
    [
        ("audio processing: repair: cold and warm time of each stage on 10 s of 16 kHz mono (reported, not asserted)", AudioRepairTiming),
        ("audio processing: repair: the registry names denoise-live, declick, declip, deess and dereverb (live) and spectral-gate (whole recording) as the library's; denoise's and spectral-gate's live alternative is denoise-live; no option of any registered processor is a processor's name; a live pipeline refuses spectral-gate naming denoise-live; each stage reports its look-ahead as latency", AudioRepairRegistry),
        ("audio processing: repair: each stage against its reference within 1e-5 (Apply aligned: the reference's look-ahead silence taken off), and its effect: the noise in the pauses down 10 dB with the bursts within 1 dB, the clicks' error down 20 dB, the clipped samples within a tenth of their error, the sibilance down 6 dB with the voice within 0.5 dB, the late reverberation down", AudioRepairReference),
        ("audio processing: repair: every stage run block by block (blocks of 1, 7, 160 and 4,096, mono and stereo, the latency and tail included) once its latency is off gives Apply's samples bit for bit, and a live stage allocates nothing per block once open", AudioRepairBlocks),
        ("audio processing: repair: pipeline text (declip level=auto headroom=3, an option after a comma) round-trips; refused naming the cause: gain frames under 1, an empty noise memory, a reduction over 1, a click threshold of 0, a clip level that is not a number or auto or is over 0 dBFS, a value to declip, an unknown de-esser mode, a de-esser frequency over half the rate, a T60 of 0, an unknown option", AudioRepairText),
    ];

    internal static readonly string[] RepairLive = ["denoise-live", "declick", "declip", "deess", "dereverb"];

    internal static string RepairData(string relative) => AudioData(Path.Combine("repair", relative));

    internal static JsonObject RepairReference() => JsonNode.Parse(File.ReadAllText(RepairData("repair.json")))!.AsObject();

    internal static AudioData RepairInput(JsonNode entry) =>
        new(ReadNpyFloat32(RepairData((string)entry["input"]!)), (int)entry["channels"]!, (int)entry["rate"]!);

    internal static AudioData RepairFile(string name, int rate) => new(ReadNpyFloat32(RepairData(name)), 1, rate);

    internal static void AudioRepairTiming(Device device)
    {
        _ = device;
        var sound = ProcessingTone(16000, 10, 1, noise: 0.01f, bursts: true);
        foreach (string text in new[] { "denoise-live", "spectral-gate", "declick", "declip", "deess", "dereverb" })
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
            Console.WriteLine($"    {text,-14} 10 s of 16 kHz mono: cold {cold,7:F2} ms, warm {average,6:F2} ms ({10000 / average:N0}x real time)");
        }
    }

    internal static void AudioRepairRegistry(Device device)
    {
        _ = device;
        string[] all = [.. RepairLive, "spectral-gate"];
        Check(all.All(n => AudioProcessors.Names.Contains(n) && AudioProcessors.Origin(n) == Overrides.Library),
            $"the library's repair processors: {string.Join(", ", AudioProcessors.Names)}");
        Check(RepairLive.All(n => AudioProcessors.Get(n).Live) && !AudioProcessors.Get("spectral-gate").Live, "the live ones are live, spectral-gate needs the whole recording");
        Check(AudioProcessors.Get("denoise").LiveAlternative == "denoise-live" && AudioProcessors.Get("spectral-gate").LiveAlternative == "denoise-live",
            "denoise's and spectral-gate's live alternative is denoise-live");
        foreach (string name in all)
        {
            var processor = AudioProcessors.Get(name);
            Check(processor.Parameters.Count > 0 && processor.Parameters.All(p => p.Help is { Length: > 0 } && (!p.Optional || p.Default is not null)),
                $"{name}: every input has a help line and its default");
            Check(processor.Parameters.All(p => p.Input != AudioProcessorInput.Number || p.Unit is not null || p.Key is "reduce" or "quantile" or "ratio" or "q"),
                $"{name}: every number has its unit");
            Check(AudioProcessors.Describe().Contains($"{name}: ", StringComparison.Ordinal), $"{name} is listed");
        }

        // After a comma a registered name starts a step, so no option may be one.
        var clashes = AudioProcessors.Names.SelectMany(n => AudioProcessors.Get(n).Keys.Where(k => AudioProcessors.Find(k) is not null).Select(k => (n, k))).ToList();
        Check(clashes.Count == 0, $"no option of a registered processor is a processor's name: {string.Join(", ", clashes.Select(c => $"{c.Item1} {c.Item2}"))}");

        Check(AudioPipeline.Parse("declip, level=-3").ToString() == "declip level=-3" && AudioPipeline.Parse("dereverb, t60=0.8").ToString() == "dereverb t60=0.8",
            "an option after a comma joins the step before it");
        var input = new ProcessingInput(ProcessingTone(16000, 0.1, 1), 100);
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declick, spectral-gate").Wrap(input), "a live pipeline with spectral-gate", "spectral-gate", "whole recording", "use denoise-live instead");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise").Wrap(input), "a live pipeline with denoise", "denoise", "use denoise-live instead");
        using var wrapped = AudioPipeline.Parse("denoise-live, dereverb, declick, declip, deess").Wrap(input);
        Check(wrapped.SampleRate == 16000, "a pipeline of the live repair stages wraps live sound");

        foreach (var (text, rate, latency) in new[]
                 {
                     ("denoise-live", 16000, 511), ("denoise-live", 48000, 1023), ("denoise-live window=0.016", 16000, 255), ("dereverb", 16000, 511),
                     ("declick", 16000, 320 + 32 + 32 - 1), ("declick length=1 order=8 frame=10", 48000, 480 + 48 + 16 - 1),
                     ("declip", 16000, 320 + 80 + 12 - 1), ("declip length=2 order=16 frame=5", 48000, 240 + 96 + 16 - 1), ("deess", 16000, 0),
                     ("denoise-live, declick", 16000, 511 + 383),
                 })
        {
            using var stage = AudioPipeline.Parse(text).Open(rate, 2);
            Check(stage.Latency == latency, $"{text} at {rate} Hz: latency {stage.Latency}, expected {latency}");
        }
    }

    internal static void AudioRepairReference(Device device)
    {
        _ = device;
        var reference = RepairReference();
        var outputs = new Dictionary<string, AudioData>();
        foreach (var node in reference["cases"]!.AsArray())
        {
            var entry = node!;
            string text = (string)entry["pipeline"]!;
            var input = RepairInput(entry);
            var pipeline = AudioPipeline.Parse(text);
            var output = pipeline.Apply(input);
            int latency = (int)entry["latency"]!;
            var expected = ProcessingAligned(ReadNpyFloat32(RepairData((string)entry["output"]!)), input.Channels, latency);   // the reference's look-ahead silence off
            using (var stage = pipeline.Open(input.SampleRate, input.Channels))
            {
                Check(stage.Latency == latency, $"{text}: latency {stage.Latency}, the reference's {latency}");
            }

            Check(output.Channels == input.Channels && output.Samples.Length == expected.Length && output.Length == input.Length,
                $"{text}: {output.Channels} x {output.Length} samples, the reference {expected.Length / input.Channels} (the input's {input.Length}; Apply takes the latency {latency} off)");
            double largest = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                largest = Math.Max(largest, Math.Abs(expected[i] - output.Samples[i]));
            }

            Check(largest <= 1e-5, $"{text} on {entry["input"]}: the largest difference from the reference is {largest:G3}");
            Console.WriteLine($"    {text} on {entry["input"]}: largest difference {largest:G3}");
            outputs[text] = output;
        }

        var clicks = RepairFile("clicks16k.npy", 16000);
        var clicksClean = RepairFile("clicksclean16k.npy", 16000);
        var clipped = RepairFile("clipped16k.npy", 16000);
        var clippedClean = RepairFile("clippedclean16k.npy", 16000);
        foreach (var node in reference["effects"]!.AsArray())
        {
            var effect = node!;
            string text = (string)effect["pipeline"]!, kind = (string)effect["kind"]!;
            var y = outputs[text];
            var source = RepairInput(reference["cases"]!.AsArray().First(c => (string)c!["pipeline"]! == text)!);
            switch (kind)
            {
                case "noise":
                {
                    double down = RepairLevel(source.Channel(0), effect["pauses"]!) - RepairLevel(y.Channel(0), effect["pauses"]!);
                    double kept = RepairLevel(y.Channel(0), effect["bursts"]!) - RepairLevel(source.Channel(0), effect["bursts"]!);
                    RepairEffect(text, down >= 10 && Math.Abs(kept) <= 1, $"the pauses' noise down {down:F2} dB (at least 10), the bursts by {kept:F2} dB (within 1)",
                        (down, effect["noise_down_db"]!), (kept, effect["burst_change_db"]!));
                    break;
                }

                case "clicks":
                {
                    var regions = effect["regions"]!;
                    double before = RepairLevel(RepairDifference(clicks.Channel(0), clicksClean.Channel(0)), regions);
                    double after = RepairLevel(RepairDifference(y.Channel(0), clicksClean.Channel(0)), regions);
                    float moved = 0;
                    var inside = new bool[y.Length];
                    foreach (var range in regions.AsArray())
                    {
                        for (int i = (int)range![0]!; i < (int)range[1]!; i++)
                        {
                            inside[i] = true;
                        }
                    }

                    for (int i = 0; i < y.Length; i++)
                    {
                        moved = inside[i] ? moved : Math.Max(moved, Math.Abs(y.Samples[i] - clicks.Samples[i]));
                    }

                    RepairEffect(text, before - after >= 20 && moved <= 0.05, $"the clicks' error {before:F2} dB before, {after:F2} after (down {before - after:F2}, at least 20); elsewhere moved by at most {moved:G3}",
                        (before - after, effect["down_db"]!));
                    break;
                }

                case "clipping":
                {
                    double gain = (double)effect["gain"]!, bar = 0.5 * Math.Pow(10, -0.05 / 20), before = 0, after = 0, worst = 0;
                    int count = 0;
                    for (int i = 320; i < clipped.Length - 320; i++)
                    {
                        if (Math.Abs(clipped.Samples[i]) >= bar)
                        {
                            double clean = clippedClean.Samples[i], rebuilt = y.Samples[i] / gain;
                            before += (clipped.Samples[i] - clean) * (clipped.Samples[i] - clean);
                            after += (rebuilt - clean) * (rebuilt - clean);
                            worst = Math.Max(worst, Math.Abs(rebuilt - clean));
                            count++;
                        }
                    }

                    (before, after) = (Math.Sqrt(before / count), Math.Sqrt(after / count));
                    RepairEffect(text, after <= 0.1 * before && worst <= 0.1, $"the {count} clipped samples' RMS error {before:F4} before, {after:F4} after (a tenth or less), the largest {worst:F4} (at most 0.1)",
                        (after, effect["rms_error_after"]!));
                    break;
                }

                case "sibilance":
                {
                    double low = (double)effect["band"]![0]!, high = (double)effect["band"]![1]!;
                    double down = RepairBand(source, effect["sibilants"]!, low, high) - RepairBand(y, effect["sibilants"]!, low, high);
                    double voice = RepairBand(y, effect["voice"]!, 0, 3000) - RepairBand(source, effect["voice"]!, 0, 3000);
                    RepairEffect(text, down >= 6 && Math.Abs(voice) <= 0.5, $"the sibilants from {low} to {high} Hz down {down:F2} dB (at least 6), the voice under 3 kHz by {voice:F2} dB (within 0.5)",
                        (down, effect["band_down_db"]!), (voice, effect["voice_change_db"]!));
                    break;
                }

                case "reverb":
                {
                    double least = text == "dereverb" ? 4 : 6;
                    double down = RepairLevel(source.Channel(0), effect["tails"]!) - RepairLevel(y.Channel(0), effect["tails"]!);
                    double kept = RepairLevel(y.Channel(0), effect["speech"]!) - RepairLevel(source.Channel(0), effect["speech"]!);
                    RepairEffect(text, down >= least && kept >= -3, $"the late reverberation down {down:F2} dB (at least {least}), the bursts by {kept:F2} dB (at least -3)",
                        (down, effect["tail_down_db"]!), (kept, effect["speech_change_db"]!));
                    break;
                }

                default:
                    Check(false, $"an effect of an unknown kind {kind}");
                    break;
            }
        }

        var cross = reference["noisereduce"]!;
        Console.WriteLine($"    noisereduce's stationary gate on {cross["input"]} (the script's cross-check): the pauses down {(double)cross["noise_down_db"]!:F2} dB, the bursts {(double)cross["burst_change_db"]!:F2} dB");
        var silence = new AudioData(new float[16000], 1, 16000);
        Check(AudioPipeline.Parse("spectral-gate").Apply(silence).Samples.All(x => x == 0), "spectral-gate leaves silence silent");
        var shortSound = ProcessingTone(16000, 0.03, 1);
        Check(ReferenceEquals(AudioPipeline.Parse("spectral-gate").Apply(shortSound), shortSound), "spectral-gate leaves a recording shorter than two windows as it is");
        foreach (string text in RepairLive)
        {
            var quiet = AudioPipeline.Parse(text).Apply(silence);
            Check(quiet.Samples.All(x => x == 0), $"{text} leaves silence silent (no click or clip found in it)");
        }
    }

    internal static void RepairEffect(string text, bool met, string what, params (double Mine, JsonNode Script)[] numbers)
    {
        Check(met, $"{text}: {what}");
        foreach (var (mine, script) in numbers)
        {
            Check(Math.Abs(mine - (double)script) <= 0.05 * Math.Max(1, Math.Abs((double)script)), $"{text}: {mine:G6} measured here, {(double)script:G6} by the script");
        }

        Console.WriteLine($"    {text}: {what}");
    }

    // The level in dB of the samples in the ranges ([[from, to], ...]), joined.
    internal static double RepairLevel(ReadOnlySpan<float> x, JsonNode ranges)
    {
        double sum = 0;
        long count = 0;
        foreach (var range in ranges.AsArray())
        {
            for (int i = (int)range![0]!; i < Math.Min((int)range[1]!, x.Length); i++)       // past the end: to the end, as the script's slices
            {
                sum += (double)x[i] * x[i];
                count++;
            }
        }

        return 10 * Math.Log10(Math.Max(sum / count, 1e-30));
    }

    internal static float[] RepairDifference(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var d = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            d[i] = (float)((double)a[i] - b[i]);
        }

        return d;
    }

    // The power from `low` to `high` Hz of the frames centred in the ranges, in dB (the library's STFT, the 32 ms window), as the script's band_level.
    internal static double RepairBand(AudioData sound, JsonNode ranges, double low, double high)
    {
        int size = 16;
        while (size * 2 <= 0.032 * sound.SampleRate)
        {
            size *= 2;
        }

        var options = new AudioFeatureOptions { SampleRate = sound.SampleRate, WindowSize = size, HopSize = size / 4, Window = "hann", Center = true, PadSeconds = null };
        int frames = Stft.FramesFor(sound.Length, options), bins = options.Bins;
        var power = new float[frames * bins];
        Stft.PowerOnHost(sound.Channel(0), options, power);
        double total = 0;
        int count = 0;
        for (int f = 0; f < frames; f++)
        {
            int centre = f * (size / 4);
            if (!ranges.AsArray().Any(r => (int)r![0]! <= centre && centre < (int)r[1]!))
            {
                continue;
            }

            double sum = 0;
            for (int k = 0; k < bins; k++)
            {
                double hertz = (double)k * sound.SampleRate / size;
                sum += hertz >= low && hertz < high ? power[f * bins + k] : 0;
            }

            total += sum;
            count++;
        }

        return 10 * Math.Log10(Math.Max(total / count, 1e-30));
    }

    internal static void AudioRepairBlocks(Device device)
    {
        _ = device;
        var reference = RepairReference();
        var texts = new List<(string Text, AudioData Sound)>();
        foreach (var entry in reference["cases"]!.AsArray())
        {
            texts.Add(((string)entry!["pipeline"]!, RepairInput(entry)));
        }

        var stereo = texts.First(t => t.Sound.Channels == 2).Sound;
        var clipped = RepairFile("clipped16k.npy", 16000);
        texts.Add(("denoise-live, dereverb, declick, declip, deess", stereo));
        texts.Add(("declick, spectral-gate, deess", stereo));
        texts.Add(("declip level=-6.0206 headroom=3, declick=6", clipped));
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
                var joined = ProcessingAligned(ProcessingJoin(pieces, stage.Channels, stage.SampleRate), stage.Latency);
                Check(joined.Length == whole.Length && ProcessingSame(joined.Samples, whole.Samples),
                    $"{text}, {sound.Channels} channel(s), blocks of {size}: {joined.Length} samples against Apply's {whole.Length}, the same bits");
            }
        }

        foreach (string text in new[] { "denoise-live", "denoise-live frames=4 smooth=0", "dereverb", "declick", "declip", "declip level=-3 headroom=3", "deess", "deess mode=band link=false" })
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

    internal static void AudioRepairText(Device device)
    {
        _ = device;
        var full = AudioPipeline.Parse("denoise-live=2 Track=1, spectral-gate=1.5 reduce=0.9, declick=8, declip level=AUTO headroom=3, deess=-30 mode=band, dereverb=1 t60=0.6");
        Check(full.ToString() == "denoise-live=2 track=1, spectral-gate=1.5 reduce=0.9, declick=8, declip headroom=3 level=AUTO, deess=-30 mode=band, dereverb=1 t60=0.6"
              && AudioPipeline.FromJson(full.ToJson()).Equals(full) && !full.Live, $"every repair step round-trips: {full}");
        var live = AudioPipeline.Parse("denoise-live, declip, level=-1, deess, frequency=6000");
        Check(live.Live && live.ToString() == "denoise-live, declip level=-1, deess frequency=6000", $"options after commas join their steps: {live}");

        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise-live frames=0"), "no gain frames", "denoise-live frames", "1 to 64", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("denoise-live track=0"), "an empty noise memory", "denoise-live track", "above 0", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("spectral-gate reduce=2"), "a reduction over 1", "spectral-gate reduce", "0 to 1", "2");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declick=0"), "a click threshold of 0", "declick", "above 0", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declick length=0"), "no click length", "declick length", "above 0", "ms");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declip level=loud"), "a clip level that is neither a number nor auto", "declip level", "number", "loud");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declip level=3"), "a clip level over full scale", "declip level", "at most 0 dBFS", "3");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declip=3"), "a value to declip", "declip", "takes no value");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("declip run=0"), "a run of no samples", "declip run", "1 to 1000");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("deess mode=wide"), "an unknown de-esser mode", "deess mode", "high, band", "wide");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("deess ratio=0.5"), "a ratio under 1", "deess ratio", "at least 1");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("deess frequency=9000").Open(16000, 1), "a de-esser frequency over half the rate", "deess", "9000 Hz", "16000 Hz");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("dereverb t60=0"), "a T60 of 0", "dereverb t60", "above 0", "0");
        ThrowsNaming<ArgumentException>(() => AudioPipeline.Parse("dereverb depth=3"), "an unknown option", "dereverb", "depth", "t60, delay");
    }
}
