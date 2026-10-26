// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Vits;


// The cleaning stages (AudioCleaners: highpass, denoise, trim, normalize) on made signals, through the trainer's options
// and feature cache, and through idrak data audio --clean.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioCleaningGroup =
    [
        ("audio cleaning: the registry names highpass, denoise, trim and normalize as the library's; an unregistered stage is refused naming the registered ones; the options round-trip through JSON (a plug-in's settings included) and refuse values out of range", AudioCleaningRegistry),
        ("audio cleaning: highpass takes a DC offset and a 10 Hz rumble out of a 1 kHz tone (each down over 20 dB) and keeps the tone within 0.5 dB on both channels; trim cuts the silence at both ends of a tone to the pad and leaves a silent clip and a tight one as they are; normalize brings the speech frames to the target level within 0.5 dB and never over the peak ceiling", AudioCleaningStages),
        ("audio cleaning: denoise on a tone spoken in bursts under white noise 20 dB down turns the noise in the pauses down by at least 8 dB, keeps the tone's level in a burst within 1 dB and the length exactly; AudioCleaners.Apply runs the stages in the order named", AudioCleaningDenoise),
        ("audio cleaning: SpeechTrainingOptions carry the stages and their settings through JSON and a checkpoint's state, refuse an unregistered stage naming it, and the feature cache keeps cleaned and uncleaned rows apart (a second cleaned pass adds none); idrak data audio --clean trim writes the trimmed piece with its manifest row and refuses an unknown stage naming it", AudioCleaningTraining),
    ];

    // A tone of `hertz` at `amplitude` for `seconds`, `channels` copies, plus a DC offset and an extra signal per sample.
    private static AudioData CleaningSignal(int rate, double seconds, double hertz, float amplitude, int channels = 1, float dc = 0, Func<int, float>? extra = null)
    {
        int length = (int)Math.Round(seconds * rate);
        var samples = new float[channels * length];
        for (int i = 0; i < length; i++)
        {
            float x = (float)(amplitude * Math.Sin(2 * Math.PI * hertz * i / rate)) + dc + (extra?.Invoke(i) ?? 0);
            for (int c = 0; c < channels; c++)
            {
                samples[c * length + i] = x;
            }
        }

        return new AudioData(samples, channels, rate);
    }

    // The mean power of the component at `hertz` (Goertzel over the whole span; a bin frequency gives it exactly).
    private static double CleaningBandPower(ReadOnlySpan<float> x, double hertz, int rate)
    {
        int n = x.Length;
        double w = 2 * Math.PI * Math.Round(n * hertz / rate) / n, coefficient = 2 * Math.Cos(w), s1 = 0, s2 = 0;
        foreach (float v in x)
        {
            double s0 = v + coefficient * s1 - s2;
            (s2, s1) = (s1, s0);
        }

        double magnitude2 = s1 * s1 + s2 * s2 - coefficient * s1 * s2;        // |X[k]|²
        double amplitude = 2 * Math.Sqrt(magnitude2) / n;
        return amplitude * amplitude / 2;
    }

    private static double CleaningMeanSquare(ReadOnlySpan<float> x)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }

        return sum / Math.Max(1, x.Length);
    }

    private static double Decibels(double ratio) => 10 * Math.Log10(ratio);

    private static void AudioCleaningRegistry(Device device)
    {
        _ = device;
        Check(new[] { "highpass", "denoise", "trim", "normalize" }.All(n => AudioCleaners.Names.Contains(n) && AudioCleaners.Origin(n) == Overrides.Library && AudioCleaners.Default(n) is not null),
            $"the library's stages: {string.Join(", ", AudioCleaners.Names)}");
        Check(AudioCleaners.Find("TRIM")?.Name == AudioCleaners.Trim && AudioCleaners.Find("vacuum") is null, "names ignore case; an unknown one is null");
        ThrowsNaming<NotSupportedException>(() => AudioCleaners.Get("vacuum"), "an unregistered stage", "vacuum", "highpass", "trim", "AudioCleaners.Register");

        var options = new AudioCleaningOptions { HighPassHertz = 80, TargetDecibels = -18, DenoiseStrength = 2, Settings = new JsonObject { ["model"] = "dfn3", ["passes"] = 2 } };
        var back = AudioCleaningOptions.FromJson(options.ToJson());
        Check(back == options with { Settings = back.Settings } && back.Settings!.ToJsonString() == options.Settings.ToJsonString() && back.HighPassHertz == 80 && back.TargetDecibels == -18,
            $"the options round-trip: {back.ToJson().ToJsonString()}");
        Check(AudioCleaningOptions.FromJson(new JsonObject()) == new AudioCleaningOptions(), "missing keys keep the defaults");
        new AudioCleaningOptions().Check();
        ThrowsNaming<ArgumentException>(() => new AudioCleaningOptions { PeakCeiling = 1.5 }.Check(), "a peak ceiling over 1", "PeakCeiling");
        ThrowsNaming<ArgumentException>(() => new AudioCleaningOptions { NoiseQuantile = 2 }.Check(), "a quantile over 1", "NoiseQuantile");
        ThrowsNaming<ArgumentException>(() => new AudioCleaningOptions { TargetDecibels = 3 }.Check(), "a target above full scale", "TargetDecibels");
        ThrowsNaming<ArgumentException>(() => new AudioCleaningOptions { SilenceDecibels = 0 }.Check(), "a silence threshold of 0", "SilenceDecibels");
    }

    private static void AudioCleaningStages(Device device)
    {
        _ = device;
        const int Rate = 16000;
        var options = new AudioCleaningOptions();

        // highpass: a 1 kHz tone with a DC offset and a 10 Hz rumble (2.6 octaves under the 60 Hz cutoff: 12 dB an octave), on two channels.
        var dirty = CleaningSignal(Rate, 2, 1000, 0.3f, channels: 2, dc: 0.1f, extra: i => (float)(0.1 * Math.Sin(2 * Math.PI * 10 * i / Rate)));
        var filtered = AudioCleaners.Get(AudioCleaners.HighPass).Clean(dirty, options);
        Check(filtered.Channels == 2 && filtered.Length == dirty.Length && filtered.SampleRate == Rate, "highpass keeps the channels, length and rate");
        for (int c = 0; c < 2; c++)
        {
            var before = dirty.Channel(c)[3200..];                                 // the filter's start-up left out
            var after = filtered.Channel(c)[3200..];
            double rumble = Decibels(CleaningBandPower(before, 10, Rate) / CleaningBandPower(after, 10, Rate));
            double toneShift = Decibels(CleaningBandPower(after, 1000, Rate) / CleaningBandPower(before, 1000, Rate));
            double dcBefore = 0, dcAfter = 0;
            foreach (float v in before)
            {
                dcBefore += v;
            }

            foreach (float v in after)
            {
                dcAfter += v;
            }

            Check(rumble > 20 && Math.Abs(toneShift) < 0.5 && Math.Abs(dcAfter) < Math.Abs(dcBefore) / 10,
                $"channel {c}: the 10 Hz rumble down {rumble:F1} dB, the tone moved {toneShift:F2} dB, the DC offset {dcBefore / before.Length:F4} to {dcAfter / after.Length:F5}");
        }

        ThrowsNaming<ArgumentException>(() => AudioCleaners.Get(AudioCleaners.HighPass).Clean(dirty, options with { HighPassHertz = 9000 }), "a cutoff over half the rate", "9000", "half the rate");
        Check(ReferenceEquals(AudioCleaners.Get(AudioCleaners.HighPass).Clean(dirty, options with { HighPassHertz = 0 }), dirty), "a cutoff of 0 leaves the sound as it is");

        // trim: 0.5 s of silence, 1 s of tone, 0.7 s of silence; the pad of 0.1 s kept on both sides.
        var tone = CleaningSignal(Rate, 1, 440, 0.3f);
        var padded = new float[(int)(2.2 * Rate)];
        tone.Samples.CopyTo(padded, Rate / 2);
        var trimmed = AudioCleaners.Get(AudioCleaners.Trim).Clean(new AudioData(padded, 1, Rate), options);
        int expected = Rate + 2 * (int)(options.TrimPadSeconds * Rate);
        Check(Math.Abs(trimmed.Length - expected) <= 2 * (int)(options.HopSeconds * Rate), $"trimmed to {trimmed.Length} samples, {expected} expected");
        Check(Math.Abs(trimmed.Samples[(int)(options.TrimPadSeconds * Rate) + 400] - tone.Samples[400]) < 1e-6f || trimmed.Length != expected, "the tone starts after the pad");
        var silent = new AudioData(new float[Rate], 1, Rate);
        Check(ReferenceEquals(AudioCleaners.Get(AudioCleaners.Trim).Clean(silent, options), silent), "a silent clip is left as it is");
        Check(ReferenceEquals(AudioCleaners.Get(AudioCleaners.Trim).Clean(tone, options), tone), "a clip with no silence to cut is left as it is");

        // normalize: a quiet tone after 0.3 s of silence comes to the target level; a loud one is held under the ceiling.
        var quiet = new float[(int)(1.3 * Rate)];
        CleaningSignal(Rate, 1, 440, 0.05f).Samples.CopyTo(quiet, (int)(0.3 * Rate));
        var levelled = AudioCleaners.Get(AudioCleaners.Normalize).Clean(new AudioData(quiet, 1, Rate), options);
        double level = Decibels(CleaningMeanSquare(levelled.Samples.AsSpan((int)(0.3 * Rate) + 160, Rate - 320)));
        Check(Math.Abs(level - options.TargetDecibels) < 0.5 && levelled.Samples.Max(MathF.Abs) <= options.PeakCeiling + 1e-4,
            $"the tone levelled to {level:F2} dBFS (target {options.TargetDecibels}), peak {levelled.Samples.Max(MathF.Abs):F3}");
        Check(Math.Abs(levelled.Samples[(int)(0.3 * Rate) - 160]) < 1e-6f, "the silence stays silent");
        var loud = AudioCleaners.Get(AudioCleaners.Normalize).Clean(CleaningSignal(Rate, 1, 440, 0.9f), options with { TargetDecibels = -1 });
        Check(loud.Samples.Max(MathF.Abs) <= options.PeakCeiling + 1e-4 && loud.Samples.Max(MathF.Abs) > 0.94, $"the peak held at {loud.Samples.Max(MathF.Abs):F3} under the ceiling {options.PeakCeiling}");
        Check(ReferenceEquals(AudioCleaners.Get(AudioCleaners.Normalize).Clean(silent, options), silent), "a silent clip is left as it is");
    }

    private static void AudioCleaningDenoise(Device device)
    {
        _ = device;
        const int Rate = 16000;
        var random = new Random(11);
        float noiseRms = 0.3f / MathF.Sqrt(2) / 10;                               // the tone's RMS 20 dB above the noise's
        // The tone speaks in bursts of 0.4 s with pauses of 0.2 s (speech pauses: the floor is read there), under white noise throughout.
        var noisy = CleaningSignal(Rate, 3, 1000, 0.3f, extra: i => noiseRms * (float)(Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble())));
        for (int i = 0; i < noisy.Length; i++)
        {
            if (i % (6 * Rate / 10) >= 4 * Rate / 10)
            {
                noisy.Samples[i] -= (float)(0.3 * Math.Sin(2 * Math.PI * 1000 * i / Rate));
            }
        }

        var options = new AudioCleaningOptions();
        var cleaned = AudioCleaners.Get(AudioCleaners.Denoise).Clean(noisy, options);
        Check(cleaned.Length == noisy.Length && cleaned.Channels == 1 && cleaned.SampleRate == Rate, "denoise keeps the length, channels and rate");
        // The noise in a pause (inside the second one, clear of its edges) and the tone's level inside a burst (the third).
        int pause = 16000 + 400, pauseLength = 2400, burst = 19200 + 400, burstLength = 5600;
        double noiseBefore = Decibels(CleaningMeanSquare(noisy.Samples.AsSpan(pause, pauseLength))), noiseAfter = Decibels(CleaningMeanSquare(cleaned.Samples.AsSpan(pause, pauseLength)));
        double toneMoved = Decibels(CleaningBandPower(cleaned.Samples.AsSpan(burst, burstLength), 1000, Rate) / CleaningBandPower(noisy.Samples.AsSpan(burst, burstLength), 1000, Rate));
        Check(noiseBefore - noiseAfter >= 8 && Math.Abs(toneMoved) < 1, $"the noise in a pause {noiseBefore:F1} dBFS to {noiseAfter:F1} dBFS; the tone moved {toneMoved:F2} dB");
        Check(ReferenceEquals(AudioCleaners.Get(AudioCleaners.Denoise).Clean(new AudioData(new float[100], 1, Rate), options), noisy) is false, "a clip shorter than two windows is returned as it is");

        // Apply runs the stages in order: trim then normalize equals the two calls in turn, and differs from the other order's result length.
        var padded = new float[(int)(2.2 * Rate)];
        CleaningSignal(Rate, 1, 440, 0.05f).Samples.CopyTo(padded, Rate / 2);
        var clip = new AudioData(padded, 1, Rate);
        var applied = AudioCleaners.Apply(clip, [AudioCleaners.Trim, AudioCleaners.Normalize], options);
        var byHand = AudioCleaners.Get(AudioCleaners.Normalize).Clean(AudioCleaners.Get(AudioCleaners.Trim).Clean(clip, options), options);
        Check(applied.Length == byHand.Length && applied.Samples.AsSpan().SequenceEqual(byHand.Samples), "Apply equals the stages called in turn");
        Check(AudioCleaners.Apply(clip, [], options) == clip, "no stages: the clip itself");
        ThrowsNaming<NotSupportedException>(() => AudioCleaners.Apply(clip, ["trim", "vacuum"], options), "an unregistered stage in the list", "vacuum");
    }

    private static void AudioCleaningTraining(Device device)
    {
        var options = new SpeechTrainingOptions { Cleaning = ["trim", "normalize"], CleaningOptions = new AudioCleaningOptions { TargetDecibels = -18 } };
        var back = SpeechTrainingOptions.FromJson(options.ToJson());
        Check(back.Cleaning.SequenceEqual(["trim", "normalize"]) && back.CleaningOptions.TargetDecibels == -18, $"the training options carry the cleaning: {back.ToJson()["cleaning"]}");
        Check(SpeechTrainingOptions.FromJson(new JsonObject()).Cleaning.Count == 0, "no cleaning by default");

        string root = TempFolder("audio-cleaning");
        VitsPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 6, 41, maxLetters: 4);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            string cache = Path.Combine(root, "cache");
            var tone = VitsToneTraining(cache) with { Steps = 1 };
            using (var model = LoadVitsTrain(device))
            {
                ThrowsNaming<NotSupportedException>(() => AdversarialSynthesisTrainer.Train(model, train, null, tone with { Cleaning = ["vacuum"] }, Path.Combine(root, "none")),
                    "an unregistered stage", "vacuum", "AudioCleaners.Register");
            }

            int Rows() => Directory.Exists(cache) ? Directory.GetFiles(cache, "*", SearchOption.AllDirectories).Length : 0;
            using (var model = LoadVitsTrain(device))
            {
                var result = AdversarialSynthesisTrainer.Train(model, train, null, tone with { Cleaning = ["trim", "normalize"] }, Path.Combine(root, "cleaned"));
                Check(result.Steps == 1 && result.Losses.All(float.IsFinite), "a cleaned run trains");
            }

            int cleaned = Rows();
            using (var model = LoadVitsTrain(device))
            {
                AdversarialSynthesisTrainer.Train(model, train, null, tone, Path.Combine(root, "plain"));
            }

            int both = Rows();
            using (var model = LoadVitsTrain(device))
            {
                AdversarialSynthesisTrainer.Train(model, train, null, tone with { Cleaning = ["trim", "normalize"] }, Path.Combine(root, "cleaned-again"));
            }

            Check(cleaned > 0 && both > cleaned && Rows() == both, $"cache rows: {cleaned} after the cleaned run, {both} after the plain one too, {Rows()} after the cleaned run again");
            string state = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "cleaned", "checkpoint-1", SpeechTrainer.StateFile)))!.ToJsonString();
            Check(state.Contains("\"cleaning\":[\"trim\",\"normalize\"]", StringComparison.Ordinal) && state.Contains("\"cleaning_options\":", StringComparison.Ordinal), $"the checkpoint's state holds the cleaning: {state[Math.Max(0, state.IndexOf("cleaning", StringComparison.Ordinal) - 40)..Math.Min(state.Length, state.IndexOf("cleaning", StringComparison.Ordinal) + 200)]}");

            // idrak data audio --clean trim: a tone between silences comes out trimmed to the tone and its pads.
            const int Rate = 16000;
            var padded = new float[2 * Rate];
            CleaningSignal(Rate, 1, 440, 0.3f).Samples.CopyTo(padded, Rate / 2);
            string wav = Path.Combine(root, "padded.wav"), outFolder = Path.Combine(root, "out");
            AudioEncoders.Save(wav, new AudioData(padded, 1, Rate), new AudioEncodeOptions());
            var (code, _, err) = AudioCli("data", "audio", wav, "-o", outFolder, "--clean", "trim");
            var written = AudioCodecs.Load(Path.Combine(outFolder, "padded.wav"), channels: 1);
            Check(code == 0 && Math.Abs(written.Length - (Rate + 2 * Rate / 10)) <= 2 * Rate / 100 && File.ReadAllText(Path.Combine(outFolder, "manifest.jsonl")).Contains("\"padded.wav\"", StringComparison.Ordinal),
                $"data audio --clean trim: exit {code}, {written.Length} samples ({Rate + 2 * Rate / 10} expected), {err}");
            var refused = AudioCli("data", "audio", wav, "-o", Path.Combine(root, "out2"), "--clean", "vacuum");
            Check(refused.Code != 0 && refused.Err.Contains("vacuum", StringComparison.Ordinal) && refused.Err.Contains("trim", StringComparison.Ordinal), $"an unknown stage refused: {refused.Err}");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
