// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.VitsSpeech;

// Plan 21, phase S5: the inverse STFT, the Griffin-Lim vocoder, the synthesis contracts and the VITS plug-in. The
// fixture comes from tools/audio/make_tiny_audio.py (tiny-vits/).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioTtsGroup =
    [
        ("audio tts: Stft.Inverse gives a random signal back from ComplexOnHost within 1e-4 away from the edges (centred and not), SamplesFor names the length, and a hop the window does not cover is refused", AudioTtsInverseStft),
        ("audio tts: Griffin-Lim on a tone's magnitudes gives a tone of the same frequency, finite and of the frames' length; the same spectrogram and seed twice give the same samples; the vocoder registry has griffin-lim as the library default and makes it from a log-mel whose peak band is the tone's", AudioTtsGriffinLim),
        ("audio tts: the VITS plug-in loads tiny-vits and speaks the fixture's ids with noise 0 within 1e-4 of reference/waveform.npy at the config's rate (the durations' frames times the hop), the same twice, Rate 2 no longer, text through the tokenizer equal to the ids, SpeakStream joined equal to Speak, the WAV saved and read back; without the plug-in the family is refused", AudioTtsVitsFixture),
        ("audio tts: the VITS tokenizer maps the fixture's symbols to ids, lower-cases and drops symbols outside the vocabulary, intersperses blanks when asked, refuses phonemes and non-Latin text for a uroman checkpoint, and SplitText bounds the pieces at sentence ends", AudioTtsVitsTokenizer),
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
        VitsSpeechPlugin.Unregister();
        Check(ThrowsSaying<NotSupportedException>(() => SpeechSynthesisModels.Load(folder, loadOptions), "is not registered"), "without the plug-in the VitsModel family is refused naming the registry");
        VitsSpeechPlugin.Register();
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
            AssertClose(expected, audio.Samples, 1e-4f, "the waveform against reference/waveform.npy");

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
            VitsSpeechPlugin.Unregister();
        }
    }

    private static void AudioTtsVitsTokenizer(Device device)
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
        tokenizer.Phonemize = true;
        Check(ThrowsSaying<NotSupportedException>(() => tokenizer.Encode("ab"), "phonemize"), "a phonemizing checkpoint is refused naming the flag");
        tokenizer.Phonemize = false;
        tokenizer.IsUroman = true;
        Check(ThrowsSaying<NotSupportedException>(() => tokenizer.Encode("مرحبا"), "is_uroman"), "a uroman checkpoint refuses non-Latin text naming the flag");
        Check(tokenizer.Encode("ab").SequenceEqual([2, 3]), "a uroman checkpoint accepts Latin text");

        var pieces = VitsSynthesizer.SplitText("One two. Three four! Five? Six seven eight nine", 12);
        Check(pieces.SequenceEqual(["One two.", "Three four!", "Five?", "Six seven", "eight nine"]), $"the split: [{string.Join(" | ", pieces)}]");
        Check(VitsSynthesizer.SplitText("a. b", 0).SequenceEqual(["a. b"]), "MaxCharacters 0: one piece");
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
