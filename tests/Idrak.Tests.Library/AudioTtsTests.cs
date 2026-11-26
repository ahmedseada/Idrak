// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.EspeakNg;
using Idrak.Audio.Plugins.Vits;

// Plan 21, phase S5: the inverse STFT, the Griffin-Lim vocoder, and the VITS plug-in's tokenizer and Arabic romanizer.
// The fixture comes from tools/audio/make_tiny_audio.py (tiny-vits/).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioTtsGroup =
    [
        ("audio tts: Stft.Inverse gives a random signal back from ComplexOnHost within 1e-4 away from the edges (centred and not), SamplesFor names the length, and a hop the window does not cover is refused", AudioTtsInverseStft),
        ("audio tts: Griffin-Lim on a tone's magnitudes gives a tone of the same frequency, finite and of the frames' length; the same spectrogram and seed twice give the same samples; the vocoder makes a sound from a log-mel whose peak band is the tone's", AudioTtsGriffinLim),
        ("audio tts: the VITS tokenizer maps the fixture's symbols to ids, lower-cases and drops symbols outside the vocabulary, intersperses blanks when asked; espeak-ng not found refused naming the remedy, punctuation kept in place, the phonemes' language en-us without tokenizer_config.json; a uroman checkpoint takes Latin text; the Arabic romanizer gives uroman's output exactly on the tool's cases; and SplitText bounds the pieces at sentence ends", AudioTtsVitsTokenizer),
    ];

    internal static string TinyVits => AudioData("tiny-vits");

    internal static void AudioTtsInverseStft(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host only: once
        }

        var random = new Random(2026);
        var signal = new float[4000];
        for (int i = 0; i < signal.Length; i++)
        {
            signal[i] = random.NextSingle() * 2f - 1f;
        }

        foreach (bool center in new[] { true, false })
        {
            var options = new AudioFeatureOptions { WindowSize = 400, HopSize = 160, Center = center };
            int frames = Stft.FramesFor(signal.Length, options), bins = options.Bins;
            var real = new float[frames * bins];
            var imaginary = new float[frames * bins];
            Stft.ComplexOnHost(signal, options, real, imaginary);
            var power = new float[frames * bins];
            Stft.PowerOnHost(signal, options, power);
            for (int i = 0; i < power.Length; i++)
            {
                Check(MathF.Abs(real[i] * real[i] + imaginary[i] * imaginary[i] - power[i]) <= 1e-3f * Math.Max(1f, power[i]), $"centre {center}: the complex spectrum's power at {i} is {real[i] * real[i] + imaginary[i] * imaginary[i]}, PowerOnHost {power[i]}");
            }

            var back = Stft.Inverse(real, imaginary, frames, options, signal.Length);
            Check(back.Length == signal.Length, $"centre {center}: {back.Length} samples back for {signal.Length}");
            int margin = center ? 0 : options.WindowSize;                       // without padding the first samples lie under one window's edge only
            int checkedTo = center ? signal.Length - options.WindowSize : frames * options.HopSize - options.WindowSize;
            AssertClose(signal[margin..checkedTo], back[margin..checkedTo], 1e-4f, $"centre {center}: the signal back from its transform");
            int expected = Stft.SamplesFor(frames, options);
            Check(expected == (center ? options.HopSize * (frames - 1) : options.WindowSize + options.HopSize * (frames - 1)), $"centre {center}: SamplesFor {expected} for {frames} frames");
            Check(Stft.Inverse(real, imaginary, frames, options).Length == expected, "Inverse without a length gives SamplesFor samples");
        }

        var sparse = new AudioFeatureOptions { WindowSize = 400, HopSize = 500, Center = false };
        int sparseFrames = Stft.FramesFor(signal.Length, sparse);
        var sparseSpectrum = new float[sparseFrames * sparse.Bins];
        Check(Throws(() => Stft.Inverse(sparseSpectrum, sparseSpectrum, sparseFrames, sparse)), "a hop larger than the window leaves samples uncovered: refused");
    }

    internal static void AudioTtsGriffinLim(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host only: once
        }

        var options = new AudioFeatureOptions { SampleRate = 16000, WindowSize = 400, HopSize = 160, PadSeconds = null };
        const double Frequency = 1000;
        var tone = new float[8000];
        for (int i = 0; i < tone.Length; i++)
        {
            tone[i] = 0.5f * (float)Math.Sin(2 * Math.PI * Frequency * i / options.SampleRate);
        }

        int frames = Stft.FramesFor(tone.Length, options), bins = options.Bins;
        var magnitude = new float[frames * bins];
        Stft.PowerOnHost(tone, options, magnitude);
        for (int i = 0; i < magnitude.Length; i++)
        {
            magnitude[i] = MathF.Sqrt(magnitude[i]);
        }

        var vocoder = new GriffinLimVocoder(options, new GriffinLimOptions { Iterations = 32, Seed = 7 });
        var samples = vocoder.Reconstruct(magnitude, frames);
        Check(samples.Length == Stft.SamplesFor(frames, options) && samples.All(float.IsFinite), $"{samples.Length} finite samples for {frames} frames");
        int expectedBin = (int)Math.Round(Frequency / options.SampleRate * options.WindowSize);
        Check(PeakBin(samples, options) == expectedBin, $"the reconstructed tone peaks at bin {PeakBin(samples, options)}, the tone's is {expectedBin}");
        var again = new GriffinLimVocoder(options, new GriffinLimOptions { Iterations = 32, Seed = 7 }).Reconstruct(magnitude, frames);
        Check(samples.AsSpan().SequenceEqual(again), "the same spectrogram and seed give the same samples");

        var defaults = new GriffinLimVocoder(options);
        Check(defaults is { SampleRate: 16000, Mels: 80 }, $"the vocoder at {defaults.SampleRate} Hz with {defaults.Mels} mels");

        using var logMel = new LogMelFeatures(options).Extract(new AudioData(tone, 1, options.SampleRate), device);
        var audio = defaults.Synthesize(logMel, device);
        Check(audio.SampleRate == 16000 && audio.Channels == 1 && audio.Length == Stft.SamplesFor(logMel.Shape[0], options) && audio.Samples.All(float.IsFinite),
            $"the sound from the log-mel: {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz for {logMel.Shape[0]} frames");
        int peak = PeakBin(audio.Samples, options);
        Check(Math.Abs(peak - expectedBin) <= 2, $"the sound from the log-mel peaks at bin {peak}, the tone's is {expectedBin} (the mel bands are coarser than the bins)");
    }

    // The bin of the largest mean power over the frames.
    internal static int PeakBin(float[] samples, AudioFeatureOptions options)
    {
        int frames = Stft.FramesFor(samples.Length, options), bins = options.Bins;
        var power = new float[frames * bins];
        Stft.PowerOnHost(samples, options, power);
        var mean = new double[bins];
        for (int f = 0; f < frames; f++)
        {
            for (int b = 0; b < bins; b++)
            {
                mean[b] += power[f * bins + b];
            }
        }

        return Array.IndexOf(mean, mean.Max());
    }

    internal static void AudioTtsVitsTokenizer(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host only: once
        }

        var tokenizer = VitsTokenizer.Read(TinyVits);
        Check(!tokenizer.AddBlank && tokenizer.Normalize && !tokenizer.Phonemize && !tokenizer.IsUroman && tokenizer.Size == 20, "the fixture's tokenizer: no tokenizer_config.json, so no blanks");
        Check(tokenizer.Encode("abc ab").SequenceEqual([2, 3, 4, 1, 2, 3]), $"'abc ab' gives [{string.Join(", ", tokenizer.Encode("abc ab"))}]");
        Check(tokenizer.Encode(" ABC, ab! ").SequenceEqual([2, 3, 4, 1, 2, 3]), "upper case is lowered, symbols outside the vocabulary dropped, the ends trimmed");
        tokenizer.AddBlank = true;
        Check(tokenizer.Encode("ab").SequenceEqual([0, 2, 0, 3, 0]), $"blanks interspersed: [{string.Join(", ", tokenizer.Encode("ab"))}]");
        tokenizer.AddBlank = false;

        var missing = new EspeakNgPhonemizer { Executable = Path.Combine(Path.GetTempPath(), $"no-espeak-{Guid.NewGuid():N}") };
        Check(ThrowsSaying<NotSupportedException>(() => missing.Phonemize("hello", "en-us"), EspeakNgPhonemizer.ExecutableVariable), "espeak-ng not found is refused naming IDRAK_ESPEAK");
        Check(TextPhonemizers.KeepingPunctuation("Hello, my  world!  Yes", w => $"<{w}>") == "<Hello>, <my world>! <Yes>"
              && TextPhonemizers.KeepingPunctuation("«a» b", w => w.ToUpperInvariant()) == "«A» B", "punctuation kept in place between the phonemized stretches, white space collapsed");
        Check(VitsTokenizer.Read(TinyVits).PhonemeLanguage == "en-us", "without tokenizer_config.json's language the phonemizer reads en-us");
        tokenizer.IsUroman = true;
        Check(tokenizer.Encode("ab").SequenceEqual([2, 3]), "a uroman checkpoint accepts Latin text");
        tokenizer.IsUroman = false;

        // uroman's own output on the tool's cases (sentences, diacritics, digits, presentation forms, decomposed marks, random strings).
        var cases = JsonNode.Parse(File.ReadAllText(AudioData("uroman/arabic.json")))!.AsObject();
        var arabic = new UromanArabicRomanizer();
        int wrong = 0;
        string first = "";
        foreach (var row in cases["cases"]!.AsArray())
        {
            string text = (string)row!["text"]!, expected = (string)row["uroman"]!, got = arabic.Romanize(text);
            if (got != expected && wrong++ == 0)
            {
                first = $"\"{text}\": \"{got}\", uroman \"{expected}\"";
            }
        }

        Check(wrong == 0, $"{wrong} of {cases["cases"]!.AsArray().Count} cases differ from uroman {cases["uroman"]} (first: {first})");
        Check(UromanArabicRomanizer.Version == (string)cases["uroman"]!, "the table is from the cases' uroman");

        var pieces = VitsSynthesizer.SplitText("One two. Three four! Five? Six seven eight nine", 12);
        Check(pieces.SequenceEqual(["One two.", "Three four!", "Five?", "Six seven", "eight nine"]), $"the split: [{string.Join(" | ", pieces)}]");
        Check(VitsSynthesizer.SplitText("a. b", 0).SequenceEqual(["a. b"]), "MaxCharacters 0: one piece");
    }

    internal static bool ThrowsSaying<TException>(Action action, string containing) where TException : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (TException e)
        {
            return e.Message.Contains(containing, StringComparison.Ordinal);
        }
    }
}
