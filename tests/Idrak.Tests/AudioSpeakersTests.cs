// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Vits;

// Plan 21, S9 route A: a multi-speaker VITS checkpoint (VCTK's shape) speaks each recorded speaker by its embedding,
// against transformers' VitsModel on tiny-vits-speakers/ (the stochastic duration predictor) and tiny-vits-speakers-det/
// (the deterministic one), from tools/audio/make_tiny_audio.py --sections vits-speakers.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioSpeakersGroup =
    [
        ("audio voices: recorded speakers: the VITS plug-in reads tiny-vits-speakers and tiny-vits-speakers-det (3 speakers, an 8-wide speaker embedding conditioning the duration predictor, the flow and the decoder) and speaks the fixture's ids as each speaker within 1e-3 of the largest sample of transformers' waveform (noise 0), the speakers named by speakers.json (any case) or by index, no voice the first speaker, the speakers' sounds different; an unknown speaker refused listing them", AudioSpeakersFixture),
        ("audio voices: recorded speakers: speakers.json as a list or as a map of names to indices (the others keep their indices), none gives the indices, a name out of range or repeated refused; VoicedSynthesizer resolves a checkpoint speaker by name, any case or index before the store; idrak speak --voice p226 writes the library's samples and --list-voices lists the speakers", d => { if (d == Device.Cpu) AudioSpeakersNames(); }),
        ("audio voices: recorded speakers: a synthesis run again gives its first run's values bit for bit at every stage (the speaker's embedding, the text encoder, the durations, the prior, the flow, the decoder), the first stage that differs named", d => { if (d == Device.Cpu) AudioSpeakersRepeatStages(); }),
    ];

    private static readonly string[] FixtureSpeakers = ["p225", "p226", "p227"];

    // VitsSynthesizer.SynthesizeCore's stages (noise 0, speaker p226) run twice on the same model; the first stage whose
    // output differs between the runs is the one whose arithmetic follows something other than its inputs.
    private static void AudioSpeakersRepeatStages()
    {
        VitsPlugin.Register();
        try
        {
            string folder = AudioData("tiny-vits-speakers");
            var ids = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "reference", "input.json")))!["ids"]!.AsArray().Select(v => (int)v!).ToArray();
            using var model = (VitsSynthesizer)SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = Device.Cpu });
            T Field<T>(string name) => (T)typeof(VitsSynthesizer).GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(model)!;
            var encoder = Field<VitsTextEncoder>("_encoder");
            var predictor = Field<Idrak.Audio.Plugins.Vits.Abstractions.IVitsDurationPredictor>("_durations");
            var flow = Field<Idrak.Audio.Layers.ResidualCouplingBlock>("_flow");
            var decoder = Field<Idrak.Audio.Layers.HifiGanGenerator>("_decoder");
            var speakers = Field<Tensor>("_speakers");
            var runs = new List<List<(string Stage, float[] Values)>>();
            for (int run = 0; run < 2; run++)
            {
                var stages = new List<(string, float[])>();
                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                var speaker = speakers.EmbeddingLookup(Tensor.Indices([model.SpeakerIndex("p226")], Device.Cpu)).Transpose();
                stages.Add(("speaker", speaker.ToArray()));
                var (hidden, means, logVariances) = encoder.Forward(ids);
                stages.Add(("text encoder: hidden", hidden.ToArray()));
                stages.Add(("text encoder: means", means.ToArray()));
                stages.Add(("text encoder: log-variances", logVariances.ToArray()));
                float[] logDurations = predictor.LogDurations(hidden.Transpose(), speaker, 0f, new Random(0));
                stages.Add(("durations (log)", logDurations));
                var rows = new List<int>();
                for (int i = 0; i < logDurations.Length; i++)
                {
                    rows.AddRange(Enumerable.Repeat(i, (int)Math.Ceiling(Math.Exp(logDurations[i]))));
                }

                var prior = means.EmbeddingLookup(Tensor.Indices([.. rows], Device.Cpu));
                stages.Add(("prior", prior.ToArray()));
                var latents = flow.Reverse(prior.Transpose(), speaker);
                stages.Add(("flow", latents.ToArray()));
                stages.Add(("decoder", decoder.Samples(latents, speaker, model.DecoderChunkFrames)));
                runs.Add(stages);
            }

            for (int i = 0; i < runs[0].Count; i++)
            {
                var (stage, first) = runs[0][i];
                var second = runs[1][i].Values;
                if (!first.AsSpan().SequenceEqual(second))
                {
                    int differ = first.Length == second.Length ? first.Zip(second).Count(p => BitConverter.SingleToInt32Bits(p.First) != BitConverter.SingleToInt32Bits(p.Second)) : -1;
                    float largest = first.Length == second.Length ? first.Zip(second).Max(p => MathF.Abs(p.First - p.Second)) : float.NaN;
                    Check(false, $"the first stage that differs between two runs is {stage}: {differ} of {first.Length} values, by up to {largest:G3}"
                                 + $" (the stages before it equal: {string.Join(", ", runs[0].Take(i).Select(s => s.Stage))})");
                }
            }
        }
        finally
        {
            VitsPlugin.Unregister();
        }
    }

    private static void AudioSpeakersFixture(Device device)
    {
        VitsPlugin.Register();
        try
        {
            foreach (string name in (string[])["tiny-vits-speakers", "tiny-vits-speakers-det"])
            {
                string folder = AudioData(name);
                var input = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "reference", "input.json")))!.AsObject();
                var ids = input["ids"]!.AsArray().Select(v => (int)v!).ToArray();
                using var model = (VitsSynthesizer)SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = device });
                Check(model.Voices.SequenceEqual(FixtureSpeakers) && model.Config.Speakers == 3 && model.Config.SpeakerEmbeddingSize == 8
                      && model.Config.StochasticDurations == (name == "tiny-vits-speakers"), $"{name}: the speakers {string.Join(", ", model.Voices)}");
                var deterministic = new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0 };
                var spoken = new List<float[]>();
                for (int k = 0; k < FixtureSpeakers.Length; k++)
                {
                    var expected = ReadNpyFloat32(Path.Combine(folder, "reference", $"waveform-{k}.npy"));
                    var (audio, durations) = model.Synthesize(ids, deterministic with { Voice = FixtureSpeakers[k] });
                    float largest = expected.Max(MathF.Abs);
                    Check(audio.Length == expected.Length && audio.Length == durations.Sum() * model.Config.HopLength,
                        $"{name} {FixtureSpeakers[k]}: {audio.Length} samples (durations [{string.Join(", ", durations)}]), transformers' {expected.Length}");
                    AssertClose(expected, audio.Samples, 1e-3f * largest, $"{name} {FixtureSpeakers[k]} against transformers' waveform (largest sample {largest:G3})");
                    var (byIndex, _) = model.Synthesize(ids, deterministic with { Voice = k.ToString(System.Globalization.CultureInfo.InvariantCulture) });
                    var (byCase, _) = model.Synthesize(ids, deterministic with { Voice = FixtureSpeakers[k].ToUpperInvariant() });
                    Check(byIndex.Samples.AsSpan().SequenceEqual(audio.Samples) && byCase.Samples.AsSpan().SequenceEqual(audio.Samples), $"{name}: speaker {k} by index and in capitals");
                    spoken.Add(audio.Samples);
                }

                var (first, _) = model.Synthesize(ids, deterministic);
                Check(first.Samples.AsSpan().SequenceEqual(spoken[0]), $"{name}: no voice is the first speaker");
                Check(!spoken[0].AsSpan().SequenceEqual(spoken[1]) && !spoken[1].AsSpan().SequenceEqual(spoken[2]), $"{name}: the speakers sound different");
                ThrowsNaming<ArgumentException>(() => model.Synthesize(ids, deterministic with { Voice = "p999" }), "an unknown speaker", "'p999'", "p225, p226, p227", "0 to 2");
            }
        }
        finally
        {
            VitsPlugin.Unregister();
        }
    }

    private static void AudioSpeakersNames()
    {
        string root = TempFolder("vits-speakers");
        VitsPlugin.Register();
        try
        {
            string folder = Path.Combine(root, "model");
            Directory.CreateDirectory(folder);
            foreach (string file in Directory.GetFiles(AudioData("tiny-vits-speakers")))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            string names = Path.Combine(folder, VitsSynthesizer.SpeakersFile);
            IReadOnlyList<string> Voices()
            {
                using var model = SpeechSynthesisModels.Load(folder);
                return [.. model.Voices];
            }

            File.Delete(names);
            Check(Voices().SequenceEqual(["0", "1", "2"]), "without speakers.json the speakers are their indices");
            File.WriteAllText(names, """{"sara": 2}""");
            Check(Voices().SequenceEqual(["0", "1", "sara"]), "a map names the speakers it lists, the others keep their indices");
            File.WriteAllText(names, """{"sara": 3}""");
            ThrowsNaming<InvalidDataException>(() => Voices(), "a name out of range", "'sara' at 3", "0 to 2");
            File.WriteAllText(names, """["a", "A", "b"]""");
            ThrowsNaming<InvalidDataException>(() => Voices(), "a repeated name", "two speakers");
            File.Copy(AudioData("tiny-vits-speakers/speakers.json"), names, overwrite: true);

            // VoicedSynthesizer: a checkpoint speaker comes before the store, by name in any case or by index.
            string store = Path.Combine(root, "voices");
            var ids = JsonNode.Parse(File.ReadAllText(AudioData("tiny-vits-speakers/reference/input.json")))!["ids"]!.AsArray().Select(v => (int)v!).ToArray();
            using (var model = (VitsSynthesizer)SpeechSynthesisModels.Load(folder))
            using (var voiced = new VoicedSynthesizer(model, new FolderVoiceStore(store)))
            {
                foreach (string voice in (string[])["p226", "P226", "1"])
                {
                    var (speaker, options, converter, _) = voiced.Resolve(new SpeechSynthesisOptions { Voice = voice });
                    Check(ReferenceEquals(speaker, model) && options.Voice == voice && converter is null, $"'{voice}' resolves to the checkpoint's speaker");
                }

                ThrowsNaming<ArgumentException>(() => voiced.Resolve(new SpeechSynthesisOptions { Voice = "5" }), "an index past the speakers", "'5'", "p225, p226, p227");
            }

            // idrak speak --voice and --list-voices.
            string wav = Path.Combine(root, "p226.wav");
            float[] expected;
            using (var model = (VitsSynthesizer)SpeechSynthesisModels.Load(folder))
            {
                expected = model.Synthesize(ids, new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0, Voice = "p226" }).Audio.Samples;
            }

            VitsPlugin.Unregister();                                       // the commands register it through -P
            string text = (string)JsonNode.Parse(File.ReadAllText(AudioData("tiny-vits-speakers/reference/input.json")))!["text"]!;
            var spoken = AudioCli("speak", "-P", VitsPluginPath, folder, text, "--voice", "p226", "--noise", "0", "--duration-noise", "0", "--bits", "float",
                "--voices", store, "-o", wav, "-d", "cpu");
            Check(spoken.Code == 0, $"speak --voice p226: {spoken.Code} {spoken.Out} {spoken.Err}");
            float[] written = AudioCodecs.Decode(wav).Samples;
            string which = "";
            if (!written.AsSpan().SequenceEqual(expected))
            {
                // Which side moved: the library's call again (after the tool ran in this process), and the tool's path (the
                // text through VoicedSynthesizer with the command's options) in process.
                VitsPlugin.Register();
                float[] again, voicedText;
                using (var model = (VitsSynthesizer)SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = Device.Cpu }))
                {
                    again = model.Synthesize(ids, new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0, Voice = "p226" }).Audio.Samples;
                    using var voiced = new VoicedSynthesizer(model, new FolderVoiceStore(store));
                    voicedText = voiced.Speak(text, new SpeechSynthesisOptions
                    {
                        Voice = "p226", NoiseScale = 0, DurationNoiseScale = 0, Conversion = new VoiceConversionOptions { Tau = new VoiceConversionOptions().Tau },
                    }).Audio.Samples;
                }

                VitsPlugin.Unregister();
                string Like(float[] values) => values.AsSpan().SequenceEqual(written) ? "the WAV's" : values.AsSpan().SequenceEqual(expected) ? "the first call's" : "neither";
                which = $" (the library's call again gives {Like(again)} samples, the text through VoicedSynthesizer {Like(voicedText)})";
            }

            AssertClose(expected, written, 0f, $"speak --voice p226's WAV against the library's samples{which}");
            var listed = AudioCliJson("speak", "-P", VitsPluginPath, folder, "--list-voices", "--voices", store);
            Check(listed["speakers"]!.AsArray().Select(v => (string?)v).SequenceEqual(FixtureSpeakers), $"--list-voices names the speakers: {listed}");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
