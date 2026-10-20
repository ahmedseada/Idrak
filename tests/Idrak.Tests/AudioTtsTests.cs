// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.EspeakNg;
using Idrak.Audio.Plugins.Vits;

// Plan 21, phase S5: the inverse STFT, the Griffin-Lim vocoder, the synthesis contracts and the VITS plug-in. The
// fixture comes from tools/audio/make_tiny_audio.py (tiny-vits/).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioTtsGroup =
    [
        ("audio tts: Stft.Inverse gives a random signal back from ComplexOnHost within 1e-4 away from the edges (centred and not), SamplesFor names the length, and a hop the window does not cover is refused", AudioTtsInverseStft),
        ("audio tts: Griffin-Lim on a tone's magnitudes gives a tone of the same frequency, finite and of the frames' length; the same spectrogram and seed twice give the same samples; the vocoder registry has griffin-lim as the library default and makes it from a log-mel whose peak band is the tone's", AudioTtsGriffinLim),
        ("audio tts: the VITS plug-in loads tiny-vits and speaks the fixture's ids with noise 0 within 1e-3 of reference/waveform.npy's largest sample (not silent) at the config's rate (the durations' frames times the hop), the same twice, Rate 2 no longer, text through the tokenizer equal to the ids, SpeakStream joined equal to Speak, the WAV saved and read back; without the plug-in the family is refused", AudioTtsVitsFixture),
        ("audio tts: the VITS decoder in pieces (DecoderChunkSeconds) with its context frames on both sides gives one pass's samples within 1e-6 (pieces of 3, 7 and 50 frames, through Synthesize and the vocoder), where 50 frames decoded without context differ near the cut 100 times more; on the CPU its peak memory grows only by the latents' own copy while they grow fivefold", AudioTtsDecoderPieces),
        ("audio tts: the VITS tokenizer maps the fixture's symbols to ids, lower-cases and drops symbols outside the vocabulary, intersperses blanks when asked, phonemizes for a phonemize checkpoint through TextPhonemizers (the language from tokenizer_config.json, punctuation kept in place; an unregistered phonemizer and espeak-ng not found refused naming the remedy), romanises Arabic for a uroman checkpoint (uroman's output exactly on the tool's cases) and refuses a script no romanizer covers, and SplitText bounds the pieces at sentence ends", AudioTtsVitsTokenizer),
    ];

    private static string TinyVits => AudioData("tiny-vits");

    private static void AudioTtsInverseStft(Device device)
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

    private static void AudioTtsGriffinLim(Device device)
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

        Check(Vocoders.Names.Contains(Vocoders.DefaultName) && Vocoders.Origin(Vocoders.DefaultName) == Overrides.Library && Vocoders.Default(Vocoders.DefaultName) is not null,
            $"griffin-lim is the library's default vocoder ({string.Join(", ", Vocoders.Names)})");
        var fromRegistry = Vocoders.Create(null, options);
        Check(fromRegistry is GriffinLimVocoder { SampleRate: 16000, Mels: 80 }, $"Vocoders.Create(null) is {fromRegistry.Name} at {fromRegistry.SampleRate} Hz with {fromRegistry.Mels} mels");
        Check(Throws<NotSupportedException>(() => Vocoders.Create("no-such-vocoder")), "an unknown vocoder is refused");

        using var logMel = new LogMelFeatures(options).Extract(new AudioData(tone, 1, options.SampleRate), device);
        var audio = fromRegistry.Synthesize(logMel, device);
        Check(audio.SampleRate == 16000 && audio.Channels == 1 && audio.Length == Stft.SamplesFor(logMel.Shape[0], options) && audio.Samples.All(float.IsFinite),
            $"the sound from the log-mel: {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz for {logMel.Shape[0]} frames");
        int peak = PeakBin(audio.Samples, options);
        Check(Math.Abs(peak - expectedBin) <= 2, $"the sound from the log-mel peaks at bin {peak}, the tone's is {expectedBin} (the mel bands are coarser than the bins)");
    }

    // The bin of the largest mean power over the frames.
    private static int PeakBin(float[] samples, AudioFeatureOptions options)
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

    private static void AudioTtsVitsFixture(Device device)
    {
        string folder = TinyVits;
        var input = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "reference", "input.json")))!.AsObject();
        var ids = input["ids"]!.AsArray().Select(v => (int)v!).ToArray();
        var expected = ReadNpyFloat32(Path.Combine(folder, "reference", "waveform.npy"));
        var loadOptions = new SpeechSynthesisLoadOptions { Device = device };
        VitsPlugin.Unregister();
        Check(ThrowsSaying<NotSupportedException>(() => SpeechSynthesisModels.Load(folder, loadOptions), "is not registered"), "without the plug-in the VitsModel family is refused naming the registry");
        VitsPlugin.Register();
        try
        {
            Check(SpeechSynthesisFamilies.Names.Contains(VitsSpeechFamily.Architecture) && SpeechSynthesisFamilies.Origin(VitsSpeechFamily.Architecture) != Overrides.Library,
                "the plug-in's family is registered by an outside assembly");
            using var model = SpeechSynthesisModels.Load(folder, loadOptions);
            var vits = model as VitsSynthesizer ?? throw new Exception($"the family read a {model.GetType().Name}");
            Check(model.SampleRate == 16000 && model.Voices.Count == 0 && vits.Config.HopLength == 16, $"the config: {model.SampleRate} Hz, {model.Voices.Count} voices, hop {vits.Config.HopLength}");
            var deterministic = new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0 };
            var (audio, durations) = vits.Synthesize(ids, deterministic);
            Check(audio.SampleRate == 16000 && audio.Channels == 1, $"the sound is {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz");
            Check(audio.Length == durations.Sum() * 16 && audio.Length == expected.Length, $"{audio.Length} samples for durations [{string.Join(", ", durations)}] (x 16), the reference has {expected.Length}");
            // Relative to the reference's largest sample (about 6e-5 here, under any absolute tolerance), and not silent.
            float largest = expected.Max(MathF.Abs);
            Check(largest > 0 && audio.Samples.Max(MathF.Abs) >= 0.5f * largest, $"the waveform is not silent: largest {audio.Samples.Max(MathF.Abs):G3}, the reference's {largest:G3}");
            AssertClose(expected, audio.Samples, 1e-3f * largest, $"the waveform against reference/waveform.npy (largest sample {largest:G3})");

            var (again, _) = vits.Synthesize(ids, deterministic);
            Check(again.Samples.AsSpan().SequenceEqual(audio.Samples), "the same ids twice give the same samples");
            var (faster, fasterDurations) = vits.Synthesize(ids, deterministic with { Rate = 2f });
            Check(faster.Length <= audio.Length && fasterDurations.Zip(durations).All(p => p.First <= p.Second), $"Rate 2 gives {faster.Length} samples for {audio.Length}");
            var (seeded, _) = vits.Synthesize(ids, new SpeechSynthesisOptions { Seed = 3 });
            var (seededAgain, _) = vits.Synthesize(ids, new SpeechSynthesisOptions { Seed = 3 });
            Check(seeded.Samples.AsSpan().SequenceEqual(seededAgain.Samples) && seeded.Samples.All(float.IsFinite), "the configured noise with one seed gives the same samples twice");

            var (spoken, stats) = model.Speak((string)input["text"]!, deterministic);
            Check(spoken.Samples.AsSpan().SequenceEqual(audio.Samples), $"the text through the tokenizer gives the ids' samples ({spoken.Length} for {audio.Length})");
            Check(stats.Model > TimeSpan.Zero && stats.Vocoder > TimeSpan.Zero && stats.RealTimeFactor > 0, $"the stats: {stats}");
            var pieces = new List<AudioData>();
            var split = deterministic with { MaxCharacters = 3 };
            foreach (var piece in model.SpeakStream((string)input["text"]!, split).ToBlockingEnumerable())
            {
                pieces.Add(piece);
            }

            var (joined, _) = model.Speak((string)input["text"]!, split);
            Check(pieces.Count == 2 && pieces.Sum(p => p.Length) == joined.Length && pieces.SelectMany(p => p.Samples).SequenceEqual(joined.Samples),
                $"SpeakStream's {pieces.Count} pieces ({pieces.Sum(p => p.Length)} samples) joined equal Speak's {joined.Length} with the same split");

            string temp = TempFolder("audio-tts");
            try
            {
                string path = Path.Combine(temp, "abc.wav");
                AudioEncoders.Save(path, audio, new AudioEncodeOptions { Float = true, BitsPerSample = 32 });
                var read = AudioCodecs.Decode(path);
                Check(read.SampleRate == 16000 && read.Channels == 1 && read.Length == audio.Length, $"the saved WAV reads back as {read.Channels} x {read.Length} at {read.SampleRate} Hz");
                AssertClose(audio.Samples, read.Samples, 1e-6f, "the saved samples");
            }
            finally
            {
                Directory.Delete(temp, recursive: true);
            }
        }
        finally
        {
            VitsPlugin.Unregister();
        }
    }

    // A phonemizer of the test: each letter of the text to one of the fixture's symbols, the last call recorded.
    private sealed class TestPhonemizer : ITextPhonemizer
    {
        public string Name => "tts-test-phonemes";

        public (string Text, string Language) Asked { get; private set; }

        public static string Of(string text) => new([.. text.Select(c => char.IsLetter(c) ? (char)('a' + c % 7) : c)]);

        public string Phonemize(string text, string language)
        {
            Asked = (text, language);
            return Of(text);
        }
    }

    private static void AudioTtsVitsTokenizer(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host only: once
        }

        // The romanizer and the phonemizer are the VITS family's (Idrak.Audio.Plugins), registered with it: the library
        // has none of its own (it starts no outside program for text).
        Check(TextPhonemizers.Names.All(n => TextPhonemizers.Origin(n) != Overrides.Library) && TextRomanizers.Names.All(n => TextRomanizers.Origin(n) != Overrides.Library)
              && TextRomanizers.Default(VitsPlugin.UromanArabic) is null, "the library registers no phonemizer and no romanizer");
        VitsPlugin.Register();
        try
        {
            AudioTtsVitsTokenizerRegistered();
        }
        finally
        {
            VitsPlugin.Unregister();
        }
    }

    private static void AudioTtsVitsTokenizerRegistered()
    {
        var tokenizer = VitsTokenizer.Read(TinyVits);
        Check(!tokenizer.AddBlank && tokenizer.Normalize && !tokenizer.Phonemize && !tokenizer.IsUroman && tokenizer.Size == 20, "the fixture's tokenizer: no tokenizer_config.json, so no blanks");
        Check(tokenizer.Encode("abc ab").SequenceEqual([2, 3, 4, 1, 2, 3]), $"'abc ab' gives [{string.Join(", ", tokenizer.Encode("abc ab"))}]");
        Check(tokenizer.Encode(" ABC, ab! ").SequenceEqual([2, 3, 4, 1, 2, 3]), "upper case is lowered, symbols outside the vocabulary dropped, the ends trimmed");
        tokenizer.AddBlank = true;
        Check(tokenizer.Encode("ab").SequenceEqual([0, 2, 0, 3, 0]), $"blanks interspersed: [{string.Join(", ", tokenizer.Encode("ab"))}]");
        tokenizer.AddBlank = false;
        tokenizer.Phonemize = true;
        var phonemes = new TestPhonemizer();
        TextPhonemizers.Register(phonemes);
        try
        {
            tokenizer.PhonemeLanguage = "en-gb";
            var phonemized = tokenizer.Encode("Hi, Bo!");
            Check(phonemes.Asked == ("hi, bo!", "en-gb"), $"the phonemizer reads the lowered text in the tokenizer's language: {phonemes.Asked}");
            tokenizer.Phonemize = false;
            Check(phonemized.SequenceEqual(tokenizer.Encode(TestPhonemizer.Of("hi, bo!"))), "the phonemes are tokenized as characters");
            tokenizer.Phonemize = true;
            tokenizer.Phonemizer = "no-such-phonemizer";
            Check(ThrowsSaying<NotSupportedException>(() => tokenizer.Encode("ab"), "TextPhonemizers.Register"), "an unregistered phonemizer is refused naming the registry");
            tokenizer.Phonemizer = EspeakNgPlugin.Phonemizer;
        }
        finally
        {
            TextPhonemizers.Unregister(phonemes.Name);
        }

        var missing = new EspeakNgPhonemizer { Executable = Path.Combine(Path.GetTempPath(), $"no-espeak-{Guid.NewGuid():N}") };
        Check(ThrowsSaying<NotSupportedException>(() => missing.Phonemize("hello", "en-us"), EspeakNgPhonemizer.ExecutableVariable), "espeak-ng not found is refused naming IDRAK_ESPEAK");
        Check(TextPhonemizers.KeepingPunctuation("Hello, my  world!  Yes", w => $"<{w}>") == "<Hello>, <my world>! <Yes>"
              && TextPhonemizers.KeepingPunctuation("«a» b", w => w.ToUpperInvariant()) == "«A» B", "punctuation kept in place between the phonemized stretches, white space collapsed");
        Check(VitsTokenizer.Read(TinyVits).PhonemeLanguage == "en-us", "without tokenizer_config.json's language the phonemizer reads en-us");
        tokenizer.Phonemize = false;
        tokenizer.Phonemizer = null;
        tokenizer.IsUroman = true;
        Check(tokenizer.Encode("ab").SequenceEqual([2, 3]), "a uroman checkpoint accepts Latin text");
        Check(tokenizer.Romanize("بكا بك") == "bka bk" && tokenizer.Encode("بكا بك").SequenceEqual(tokenizer.Encode("bka bk")), $"Arabic is romanised before its ids: {tokenizer.Romanize("بكا بك")}");
        Check(ThrowsSaying<NotSupportedException>(() => tokenizer.Encode("你好"), "TextRomanizers.Register"), "a script no romanizer covers is refused naming the registry");
        tokenizer.Romanizer = "none";
        Check(ThrowsSaying<NotSupportedException>(() => tokenizer.Encode("بك"), "not registered"), "a romanizer named but not registered is refused");
        tokenizer.Romanizer = null;
        tokenizer.IsUroman = false;

        // uroman's own output on the tool's cases (sentences, diacritics, digits, presentation forms, decomposed marks, random strings).
        var cases = JsonNode.Parse(File.ReadAllText(AudioData("uroman/arabic.json")))!.AsObject();
        var arabic = TextRomanizers.Get(VitsPlugin.UromanArabic);
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
        Check(UromanArabicRomanizer.Version == (string)cases["uroman"]! && TextRomanizers.Origin(VitsPlugin.UromanArabic) == typeof(VitsPlugin).Assembly.GetName().Name && TextRomanizers.For("مرحبا") == arabic && TextRomanizers.For("hello") is null,
            "the table is from the cases' uroman; registered by the VITS plug-in, the registry picks it for Arabic, none for Latin");
        var upper = new UpperRomanizer();
        TextRomanizers.Register(upper);
        try
        {
            Check(TextRomanizers.For("Ωμέγα") == upper && TextRomanizers.Names[0] == "test-greek", "an app's romanizer is asked first for its script");
        }
        finally
        {
            TextRomanizers.Unregister("test-greek");
        }

        Check(!TextRomanizers.Names.Contains("test-greek"), "and unregisters");

        var pieces = VitsSynthesizer.SplitText("One two. Three four! Five? Six seven eight nine", 12);
        Check(pieces.SequenceEqual(["One two.", "Three four!", "Five?", "Six seven", "eight nine"]), $"the split: [{string.Join(" | ", pieces)}]");
        Check(VitsSynthesizer.SplitText("a. b", 0).SequenceEqual(["a. b"]), "MaxCharacters 0: one piece");
    }

    private sealed class UpperRomanizer : ITextRomanizer
    {
        public string Name => "test-greek";

        public bool Covers(System.Text.Rune rune) => rune.Value is >= 0x0370 and <= 0x03FF or >= 0x1F00 and <= 0x1FFF;

        public string Romanize(string text) => text;
    }

    private static void AudioTtsDecoderPieces(Device device)
    {
        VitsPlugin.Register();
        using var model = (VitsSynthesizer)SpeechSynthesisModels.Load(TinyVits, new SpeechSynthesisLoadOptions { Device = device });
        int flow = model.Config.FlowSize, hop = model.Config.HopLength, rate = model.Config.SamplingRate;
        var random = new Random(77);
        float[] Latents(int frames)
        {
            var values = new float[frames * flow];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = random.NextSingle() * 2 - 1;
            }

            return values;
        }

        float[] Decode(float[] values, int frames, int chunkFrames)
        {
            model.DecoderChunkSeconds = chunkFrames * (double)hop / rate;
            using var latents = Tensor.From(values, [frames, flow], device);
            return model.Vocoder.Synthesize(latents, device).Samples;
        }

        var long200 = Latents(200);
        var whole = Decode(long200, 200, 0);
        Check(whole.Length == 200 * hop, $"one pass gives {whole.Length} samples for 200 frames of hop {hop}");
        float worstPieces = 0;
        foreach (int chunk in new[] { 3, 7, 50 })
        {
            float[] pieces = Decode(long200, 200, chunk);
            float worst = 0;
            for (int i = 0; i < whole.Length; i++)
            {
                worst = Math.Max(worst, MathF.Abs(whole[i] - pieces[i]));
            }

            worstPieces = Math.Max(worstPieces, worst);
            Check(pieces.Length == whole.Length && worst <= 1e-6f, $"pieces of {chunk} frames differ from one pass by {worst:E2} (samples lie in [-1, 1]; the bound is 1e-6)");
        }

        // The context is what makes pieces exact: 50 frames decoded alone differ near their cut by far more.
        var alone = Decode(long200[..(50 * flow)], 50, 0);
        float edge = 0;
        for (int i = 45 * hop; i < 50 * hop; i++)
        {
            edge = Math.Max(edge, MathF.Abs(whole[i] - alone[i]));
        }

        Check(edge > 100 * Math.Max(worstPieces, 1e-9f), $"50 frames decoded without context differ by {edge:E2} near the cut, pieces with context by {worstPieces:E2}");

        var ids = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyVits, "reference", "input.json")))!["ids"]!.AsArray().Select(v => (int)v!).ToArray();
        var deterministic = new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0 };
        model.DecoderChunkSeconds = 0;
        var (onePass, _) = model.Synthesize(ids, deterministic);
        model.DecoderChunkSeconds = 2.0 * hop / rate;
        var (inPieces, _) = model.Synthesize(ids, deterministic);
        AssertClose(onePass.Samples, inPieces.Samples, 1e-6f, "Synthesize with the decoder in pieces of 2 frames against one pass");

        if (device.Type == DeviceType.Cpu)
        {
            long Peak(int frames)
            {
                var values = Latents(frames);
                model.DecoderChunkSeconds = 50.0 * hop / rate;
                using var latents = Tensor.From(values, [frames, flow], device);
                device.Backend.ResetPeakMemoryUsage();
                long before = device.Backend.GetMemoryUsage().InUse;
                model.Vocoder.Synthesize(latents, device);
                return device.Backend.GetMemoryUsage().Peak - before;
            }

            // The vocoder holds one transposed copy of its latents; beyond that the pieces' working set is the same at any length.
            long small = Peak(400), large = Peak(2000), latentsCopy = (2000 - 400) * flow * sizeof(float);
            Check(large - small <= latentsCopy + small / 10,
                $"the decoder's peak memory in pieces of 50 frames: {small} bytes for 400 frames, {large} for 2000; beyond the latents' own transposed copy ({latentsCopy} more bytes) it should not grow with the length");
        }

        model.DecoderChunkSeconds = 10;
    }

    private static bool ThrowsSaying<TException>(Action action, string containing) where TException : Exception
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
