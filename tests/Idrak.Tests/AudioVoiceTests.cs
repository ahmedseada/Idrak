// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Models;
using Idrak.OpenVoice;
using Idrak.VitsSpeech;

// Plan 21, phase S9 route B: voices from a person's clips and voice conversion. The fixture comes from
// tools/audio/make_tiny_audio.py --sections voices (tiny-openvoice/: a random OpenVoice converter from OpenVoice's own
// model code, with its spectrogram, embeddings and conversions).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioVoiceGroup =
    [
        ("audio voices: the OpenVoice plug-in loads tiny-openvoice and gives OpenVoice's spectrogram of wav/pcm16, the embeddings of the source, of each clip and of both (their mean) within 1e-4, and the conversion of the source to the target with tau 0 within 1e-4 of the largest sample for v2 (zero_g) and v1; the same twice; a target of the wrong width refused; without the plug-in the family is refused naming VoiceConverters", AudioVoiceOpenVoiceFixture),
        ("audio voices: a long clip is embedded in windows of EmbedSeconds averaged; the VITS flow forward then reverse with one embedding gives the latents back within 1e-4; a seeded tau gives the same samples twice and others for another seed; speech converted in pieces (MaxSeconds) keeps its length and stays finite; another rate is resampled; the kit's VoiceConverterSuite passes", AudioVoiceConversion),
        ("audio voices: the folder store installs a voice from WAV clips with its consent, clips' hashes, seconds and agreements in voice.json and the encoder's embedding of the trimmed clips, finds it ignoring case, refuses no consent, a bad name, a second voice of the name, a clip too short and a disagreeing clip, keeps clips only when asked, exports, removes and imports it equal, refuses a folder without consent; the kit's VoiceStoreSuite passes", AudioVoiceStore),
        ("audio voices: VoicedSynthesizer speaks an installed embedding voice as the synthesizer's speech converted to it (the converter loaded from voice.json's folder or given), \"off\" unconverted, another family refused, a checkpoint voice with its own model, a speaker voice as that checkpoint speaker, an unknown name refused listing the speakers and the installed voices, and SpeakStream as Speak", AudioVoiceResolution),
    ];

    private static string TinyOpenVoice => AudioData("tiny-openvoice");

    private static float[] OpenVoiceReference(string name) => ReadNpyFloat32(Path.Combine(TinyOpenVoice, "reference", name));

    // The fixture's sounds: wav/pcm16 (the source) and the second channel of wav/stereo16 and its second half (the clips), all read as 16 kHz.
    private static (AudioData Source, AudioData[] Clips) OpenVoiceInputs()
    {
        var source = new AudioData(ReadNpyFloat32(AudioData("wav/pcm16.npy")), 1, 16000);
        var stereo = ReadNpyFloat32(AudioData("wav/stereo16.npy"));
        var other = stereo[(stereo.Length / 2)..];
        return (source, [new AudioData(other, 1, 16000), new AudioData(other[(other.Length / 2)..], 1, 16000)]);
    }

    // The largest difference over the largest expected value, checked against `relative`.
    private static void CloseRelative(float[] expected, float[] actual, float relative, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values for {expected.Length}");
        float peak = 0, worst = 0;
        int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(expected[i]));
            float d = Math.Abs(expected[i] - actual[i]);
            if (!(d <= worst))
            {
                worst = d;
                at = i;
            }
        }

        Check(worst <= relative * Math.Max(peak, 1e-12f), $"{what}: differs by {worst:E3} at {at} ({expected[at]} against {actual[at]}), {worst / Math.Max(peak, 1e-12f):E3} of the peak {peak:E3}, over {relative:E1}");
    }

    private static IVoiceConverter LoadOpenVoice(string folder, Device device) =>
        VoiceConverters.Load(folder, new VoiceConverterLoadOptions { Device = device });

    // A copy of tiny-openvoice whose config.json says zero_g false (OpenVoice v1).
    private static string OpenVoiceV1(string root)
    {
        string folder = Path.Combine(root, "openvoice-v1");
        Directory.CreateDirectory(folder);
        File.Copy(Path.Combine(TinyOpenVoice, "model.safetensors"), Path.Combine(folder, "model.safetensors"));
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyOpenVoice, "config.json")))!.AsObject();
        config["_version_"] = "v1";
        config["model"]!["zero_g"] = false;
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        return folder;
    }

    private static void AudioVoiceOpenVoiceFixture(Device device)
    {
        OpenVoicePlugin.Unregister();
        ThrowsNaming<NotSupportedException>(() => LoadOpenVoice(TinyOpenVoice, device).Dispose(), "without the plug-in", "is not registered", "VoiceConverters.Register");
        OpenVoicePlugin.Register();
        string root = TempFolder("openvoice");
        try
        {
            Check(VoiceConverters.Names.Contains(OpenVoiceFamily.FamilyName) && VoiceConverters.Origin(OpenVoiceFamily.FamilyName) != Overrides.Library,
                "the plug-in's family is registered by an outside assembly");
            var (source, clips) = OpenVoiceInputs();
            using var converter = LoadOpenVoice(TinyOpenVoice, device);
            var open = (OpenVoiceConverter)converter;
            Check(converter.SampleRate == 16000 && converter.EmbeddingSize == 16 && open.Config.ZeroG && open.Config.HopLength == 16
                && converter.Folder == Path.GetFullPath(TinyOpenVoice), $"the config: {converter.SampleRate} Hz, {converter.EmbeddingSize} wide, zero_g {open.Config.ZeroG}, folder {converter.Folder}");
            using (var spectrogram = open.Spectrogram(source.Samples))
            {
                var expected = OpenVoiceReference("spectrogram.npy");                     // [bins, frames]
                Check(spectrogram.Shape[0] == 1500 && spectrogram.Shape[1] == 33, $"the spectrogram is [{spectrogram.Shape[0]}, {spectrogram.Shape[1]}], [1500, 33] expected");
                using var transposed = spectrogram.Transpose();
                CloseRelative(expected, transposed.ToArray(), 1e-4f, "the spectrogram against reference/spectrogram.npy");
            }

            CloseRelative(OpenVoiceReference("embedding-source.npy"), converter.Embed([source]), 1e-4f, "the source's embedding");
            var clipEmbeddings = OpenVoiceReference("embedding-clips.npy");
            for (int i = 0; i < clips.Length; i++)
            {
                CloseRelative(clipEmbeddings[(i * 16)..((i + 1) * 16)], converter.Embed([clips[i]]), 1e-4f, $"clip {i}'s embedding");
            }

            var target = converter.Embed(clips);
            CloseRelative(OpenVoiceReference("embedding-target.npy"), target, 1e-4f, "the clips' embedding (their mean)");

            var exact = new VoiceConversionOptions { Tau = 0, Source = OpenVoiceReference("embedding-source.npy") };
            var converted = converter.Convert(source, target, exact);
            Check(converted.SampleRate == 16000 && converted.Channels == 1, $"the conversion is {converted.Channels} x {converted.Length} at {converted.SampleRate} Hz");
            CloseRelative(OpenVoiceReference("converted-v2.npy"), converted.Samples, 1e-4f, "v2's conversion against reference/converted-v2.npy");
            Check(converter.Convert(source, target, exact).Samples.AsSpan().SequenceEqual(converted.Samples), "the same conversion twice");
            var embedded = converter.Convert(source, target, exact with { Source = null });
            CloseRelative(converted.Samples, embedded.Samples, 1e-4f, "the source embedded from the speech itself, as given");
            ThrowsNaming<ArgumentException>(() => converter.Convert(source, new float[15]), "a target of 15 values", "15", "16");

            using var v1 = LoadOpenVoice(OpenVoiceV1(root), device);
            Check(!((OpenVoiceConverter)v1).Config.ZeroG, "the v1 copy has zero_g off");
            CloseRelative(OpenVoiceReference("converted-v1.npy"), v1.Convert(source, target, exact).Samples, 1e-4f, "v1's conversion against reference/converted-v1.npy");
            ThrowsNaming<NotSupportedException>(() => VoiceConverters.Load(TinyOpenVoice, new VoiceConverterLoadOptions { Device = device, Family = "no-such" }).Dispose(),
                "an unregistered family asked for", "'no-such' is not registered", OpenVoiceFamily.FamilyName);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceConversion(Device device)
    {
        OpenVoicePlugin.Register();
        var (source, clips) = OpenVoiceInputs();
        using var converter = LoadOpenVoice(TinyOpenVoice, device);
        var target = converter.Embed(clips);

        // A clip longer than 1.5 EmbedSeconds is embedded in equal windows, their embeddings averaged.
        var open = (OpenVoiceConverter)converter;
        open.EmbedSeconds = 0.5;
        var windowed = converter.Embed([source]);
        var thirds = new double[16];
        for (int k = 0; k < 3; k++)
        {
            var part = converter.Embed([source.Slice(k * 8000, 8000)]);
            for (int i = 0; i < 16; i++)
            {
                thirds[i] += part[i] / 3.0;
            }
        }

        CloseRelative([.. thirds.Select(v => (float)v)], windowed, 1e-6f, "1.5 s embedded in windows of 0.5 s against the mean of its thirds");
        open.EmbedSeconds = 10;

        // The flow is invertible: forward with an embedding, then reverse with the same, is the identity.
        using (var store = SafeTensorsReader.Open(Path.Combine(TinyOpenVoice, "model.safetensors")))
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var config = OpenVoiceConfig.FromJson(JsonNode.Parse(File.ReadAllText(Path.Combine(TinyOpenVoice, "config.json")))!.AsObject());
            using var flow = OpenVoiceConverter.Flow(new StoredLayers(new OpenVoiceTensors(store), device), config);
            var random = new Random(5);
            var values = new float[8 * 40];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = random.NextSingle() * 2 - 1;
            }

            var z = Tensor.From(values, [8, 40], device);
            var g = Tensor.From(target, [16, 1], device);
            CloseRelative(values, flow.Reverse(flow.Forward(z, g), g).ToArray(), 1e-4f, "the flow forward then reverse");
        }

        var seeded = new VoiceConversionOptions { Tau = 0.5f, Seed = 9 };
        var a = converter.Convert(source, target, seeded);
        Check(a.Samples.AsSpan().SequenceEqual(converter.Convert(source, target, seeded).Samples), "a seeded tau gives the same samples twice");
        Check(!a.Samples.AsSpan().SequenceEqual(converter.Convert(source, target, seeded with { Seed = 10 }).Samples), "another seed gives other samples");
        Check(!a.Samples.AsSpan().SequenceEqual(converter.Convert(source, target, seeded with { Tau = 0 }).Samples), "tau 0 gives other samples than tau 0.5");

        foreach (double seconds in new[] { 0.2, 0.37 })
        {
            var pieces = converter.Convert(source, target, new VoiceConversionOptions { Tau = 0, MaxSeconds = seconds });
            Check(pieces.Length == source.Length / 16 * 16 && pieces.Samples.All(float.IsFinite), $"in pieces of {seconds} s: {pieces.Length} samples for {source.Length}, finite {pieces.Samples.All(float.IsFinite)}");
        }

        var resampled = converter.Convert(Resampler.Resample(source, 22050), target, new VoiceConversionOptions { Tau = 0 });
        Check(resampled.SampleRate == 16000 && Math.Abs(resampled.Length - source.Length) <= 32, $"speech at 22050 Hz converted at the converter's 16000: {resampled.Length} samples for {source.Length}");

        var report = Conformance.CheckVoiceConverter(new VoiceConverterUnderTest(converter.SampleRate, converter.EmbeddingSize,
            c => converter.Embed([.. c.Select(s => new AudioData(s, 1, converter.SampleRate))]),
            (speech, t) => converter.Convert(new AudioData(speech, 1, converter.SampleRate), t).Samples, 16));
        Check(report.Passed, $"the kit's voice converter suite: {report}");
    }

    private static VoiceAddOptions Consenting(bool keepClips = false) => new()
    {
        Consent = new VoiceConsent("A tester", "I agree that this test voice is installed.", new DateTimeOffset(2026, 10, 15, 12, 0, 0, TimeSpan.Zero)),
        Languages = ["ar", "en"],
        KeepClips = keepClips,
    };

    private static void AudioVoiceStore(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the store is host work: once
        }

        OpenVoicePlugin.Register();
        string root = TempFolder("voices");
        try
        {
            using var converter = LoadOpenVoice(TinyOpenVoice, device);
            var store = new FolderVoiceStore(Path.Combine(root, "store"));
            Check(store.List().Count == 0 && store.Find("nobody") is null, "an empty store lists nothing");
            string[] clips = [AudioData("wav/stereo16.wav"), AudioData("wav/pcm16.wav")];
            ThrowsNaming<ArgumentException>(() => store.Add("tester", clips, converter, new VoiceAddOptions()), "no consent", "consent");
            ThrowsNaming<ArgumentException>(() => store.Add("../escape", clips, converter, Consenting()), "a name with a path", "not a voice name");
            Check(store.List().Count == 0, "nothing is installed by a refused Add");

            var voice = store.Add("Tester", clips, converter, Consenting());
            Check(voice.Name == "Tester" && voice.Kind == VoiceKinds.Embedding && voice.Languages.SequenceEqual(["ar", "en"]) && voice.EmbeddingSize == 16
                && voice.Encoder == OpenVoiceFamily.FamilyName && voice.EncoderFolder == converter.Folder, $"the profile: {voice.Name}, {voice.Kind}, [{string.Join(", ", voice.Languages)}], {voice.EmbeddingSize}, {voice.Encoder} at {voice.EncoderFolder}");
            Check(voice.Consent is { Speaker: "A tester" } c && c.Statement.StartsWith("I agree", StringComparison.Ordinal) && c.Agreed.Year == 2026, $"the consent: {voice.Consent}");
            var prepared = clips.Select(p => FolderVoiceStore.Prepare(AudioCodecs.Decode(p), converter.SampleRate, Consenting())).ToArray();
            CloseRelative(converter.Embed(prepared), store.ReadEmbedding(voice), 1e-6f, "the stored embedding against the encoder's of the trimmed clips");
            var recorded = voice.Clips;
            Check(recorded.Count == 2, $"{recorded.Count} clips recorded");
            for (int i = 0; i < clips.Length; i++)
            {
                string sha = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(clips[i])));
                Check(recorded[i].File == Path.GetFileName(clips[i]) && recorded[i].Sha256 == sha && Math.Abs(recorded[i].Seconds - prepared[i].Duration.TotalSeconds) < 1e-3
                    && recorded[i].Agreement is > -1.001f and < 1.001f, $"clip {i}: {recorded[i]}");
            }

            Check(!Directory.Exists(Path.Combine(voice.Folder, "clips")), "the clips are not kept unless asked");
            Check(store.Find("tester")?.Folder == voice.Folder && store.List().Select(v => v.Name).SequenceEqual(["Tester"]), "found ignoring case and listed");
            ThrowsNaming<IOException>(() => store.Add("tester", clips, converter, Consenting()), "a second voice of the name", "installed already", "Overwrite");
            ThrowsNaming<ArgumentException>(() => store.Add("short", clips, converter, Consenting() with { MinClipSeconds = 1000 }), "clips too short", "MinClipSeconds");
            ThrowsNaming<ArgumentException>(() => store.Add("picky", clips, converter, Consenting() with { MinAgreement = 1.5f }), "a disagreeing clip", "MinAgreement", clips[0]);
            Check(store.List().Count == 1, "the refused voices left nothing behind");

            var kept = store.Add("keeper", [clips[0], clips[0]], converter, Consenting(keepClips: true));
            Check(File.Exists(Path.Combine(kept.Folder, "clips", "stereo16.wav")) && File.Exists(Path.Combine(kept.Folder, "clips", "stereo16-2.wav"))
                && kept.Clips.Select(c => c.File).SequenceEqual(["stereo16.wav", "stereo16-2.wav"]) && kept.Manifest["clipsKept"]!.GetValue<bool>(), "kept clips are copied under unique names");

            string zip = Path.Combine(root, "tester.zip");
            store.Export("tester", zip);
            var embedding = store.ReadEmbedding(voice);
            string manifest = File.ReadAllText(Path.Combine(voice.Folder, FolderVoiceStore.ManifestFile));
            Check(store.Remove("TESTER") && !store.Remove("tester") && store.Find("tester") is null, "removed once");
            var back = store.Import(zip);
            Check(back.Name == "Tester" && File.ReadAllText(Path.Combine(back.Folder, FolderVoiceStore.ManifestFile)) == manifest
                && store.ReadEmbedding(back).AsSpan().SequenceEqual(embedding), "imported from the zip equal");
            ThrowsNaming<IOException>(() => store.Import(zip), "importing over an installed voice", "installed already");
            Check(store.Import(zip, overwrite: true).Name == "Tester", "importing with overwrite replaces it");
            var other = new FolderVoiceStore(Path.Combine(root, "other"));
            Check(other.Import(back.Folder).Name == "Tester" && other.ReadEmbedding(other.Find("tester")!).AsSpan().SequenceEqual(embedding), "imported from a folder");

            string unconsented = Path.Combine(root, "unconsented");
            Directory.CreateDirectory(unconsented);
            var bare = JsonNode.Parse(manifest)!.AsObject();
            bare.Remove("consent");
            bare["name"] = "bare";
            File.WriteAllText(Path.Combine(unconsented, FolderVoiceStore.ManifestFile), bare.ToJsonString());
            File.Copy(Path.Combine(back.Folder, FolderVoiceStore.EmbeddingFile), Path.Combine(unconsented, FolderVoiceStore.EmbeddingFile));
            ThrowsNaming<InvalidDataException>(() => store.Import(unconsented), "a folder without consent", "no consent");
            Check(store.Find("bare") is null && !Directory.EnumerateDirectories(store.Root).Any(d => Path.GetFileName(d).StartsWith('.')), "nothing is left of a refused import");

            var suiteStore = new FolderVoiceStore(Path.Combine(root, "suite"));
            var report = Conformance.CheckVoiceStore(new VoiceStoreUnderTest(
                (name, files, consent) => suiteStore.Add(name, files, converter, new VoiceAddOptions { Consent = consent is null ? null : new VoiceConsent("suite", consent, default), MinClipSeconds = 0.1 }),
                name => suiteStore.Find(name) is { } v ? suiteStore.ReadEmbedding(v) : null,
                () => [.. suiteStore.List().Select(v => v.Name)],
                suiteStore.Export,
                zipPath => suiteStore.Import(zipPath).Name,
                suiteStore.Remove), Path.Combine(root, "suite-work"));
            Check(report.Passed, $"the kit's voice store suite: {report}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceResolution(Device device)
    {
        VitsSpeechPlugin.Register();
        OpenVoicePlugin.Register();
        string root = TempFolder("voiced");
        try
        {
            var store = new FolderVoiceStore(Path.Combine(root, "store"));
            using var converter = LoadOpenVoice(TinyOpenVoice, device);
            store.Add("friend", [AudioData("wav/stereo16.wav")], converter, Consenting());

            string modelVoice = Path.Combine(store.Root, "trained");                // a checkpoint voice: its own model folder
            CopyTree(TinyVits, Path.Combine(modelVoice, "model"));
            File.WriteAllText(Path.Combine(modelVoice, FolderVoiceStore.ManifestFile), new JsonObject
            {
                ["version"] = 1, ["name"] = "trained", ["kind"] = VoiceKinds.Checkpoint, ["checkpoint"] = "model",
                ["consent"] = new JsonObject { ["speaker"] = "A tester", ["statement"] = "I agree.", ["agreed"] = "2026-10-15T12:00:00+00:00" },
            }.ToJsonString());
            string speakerVoice = Path.Combine(store.Root, "numbered");
            Directory.CreateDirectory(speakerVoice);
            File.WriteAllText(Path.Combine(speakerVoice, FolderVoiceStore.ManifestFile), new JsonObject
            {
                ["version"] = 1, ["name"] = "numbered", ["kind"] = VoiceKinds.Speaker, ["speaker"] = "7",
                ["consent"] = new JsonObject { ["speaker"] = "A tester", ["statement"] = "I agree.", ["agreed"] = "2026-10-15T12:00:00+00:00" },
            }.ToJsonString());

            var loadOptions = new SpeechSynthesisLoadOptions { Device = device };
            string text = (string)JsonNode.Parse(File.ReadAllText(Path.Combine(TinyVits, "reference", "input.json")))!["text"]!;
            var plain = new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0, Seed = 4 };
            using var voiced = new VoicedSynthesizer(SpeechSynthesisModels.Load(TinyVits, loadOptions), store);
            var (own, _) = voiced.Synthesizer.Speak(text, plain);
            Check(voiced.Voices.SequenceEqual(["friend", "numbered", "trained"]), $"the voices: [{string.Join(", ", voiced.Voices)}]");
            Check(voiced.Speak(text, plain).Audio.Samples.AsSpan().SequenceEqual(own.Samples), "no voice: the synthesizer's own");

            var friend = plain with { Voice = "friend" };
            var expected = converter.Convert(own, store.ReadEmbedding(store.Find("friend")!), new VoiceConversionOptions { Seed = 4 });
            var (converted, stats) = voiced.Speak(text, friend);
            Check(converted.SampleRate == converter.SampleRate, $"the converted voice is at {converted.SampleRate} Hz");
            CloseRelative(expected.Samples, converted.Samples, 1e-6f, "the embedding voice (converter loaded from voice.json's folder) against the speech converted by hand");
            Check(stats.Conversion > TimeSpan.Zero && stats.RealTimeFactor > 0, $"the stats: {stats}");
            using (var given = new VoicedSynthesizer(voiced.Synthesizer, store, converter, ownsSynthesizer: false))
            {
                Check(given.Speak(text, friend).Audio.Samples.AsSpan().SequenceEqual(converted.Samples), "the given converter gives the same as the one loaded");
            }

            Check(voiced.Speak(text, friend with { SampleRate = 8000 }).Audio.SampleRate == 8000, "the options' rate applies after the conversion");
            Check(voiced.Speak(text, friend with { VoiceConversion = "off" }).Audio.Samples.AsSpan().SequenceEqual(own.Samples), "\"off\": the synthesizer's own voice");
            ThrowsNaming<ArgumentException>(() => voiced.Speak(text, friend with { VoiceConversion = "another" }), "another family", "'openvoice'", "'another'");
            var streamed = voiced.SpeakStream(text, friend).ToBlockingEnumerable().ToList();
            Check(streamed.Count == 1 && streamed[0].Samples.AsSpan().SequenceEqual(converted.Samples), $"SpeakStream gives Speak's samples ({streamed.Count} pieces)");

            using (var direct = SpeechSynthesisModels.Load(TinyVits, loadOptions))
            {
                Check(voiced.Speak(text, plain with { Voice = "trained" }).Audio.Samples.AsSpan().SequenceEqual(direct.Speak(text, plain).Audio.Samples), "a checkpoint voice speaks with its own model");
            }

            ThrowsNaming<ArgumentException>(() => voiced.Speak(text, plain with { Voice = "numbered" }), "a speaker voice on a one-voice model", "'7'");
            ThrowsNaming<ArgumentException>(() => voiced.Speak(text, plain with { Voice = "stranger" }), "an unknown voice", "'stranger'", "friend, numbered, trained", "one voice");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void CopyTree(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (string file in Directory.EnumerateFiles(from))
        {
            File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
        }

        foreach (string folder in Directory.EnumerateDirectories(from))
        {
            CopyTree(folder, Path.Combine(to, Path.GetFileName(folder)));
        }
    }
}
