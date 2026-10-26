// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.IdrakAudio;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 21, phase S6: training. The fixture comes from tools/audio/make_tiny_audio.py --sections ctc (ctc/: the IdrakAudio
// "ctc" network in torch with two weight sets, padded log-mel inputs, torch's CTC loss and gradients, three SGD steps and
// torchaudio's forced alignment of the step-1 log-probabilities).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioTrainGroup =
    [
        ("audio train: CtcAlignment.Viterbi gives torchaudio's forced_align path of each utterance of ctc/reference/logp-step1.npy exactly, its log-probability within 1e-4, each label's start and length from the path; MinimumSteps counts a step between equal labels; too few steps, a blank label and a label out of range refused", AudioTrainViterbi),
        ("audio train: SpeechSources \"transcripts\" reads the fixture's train.jsonl with its audio folder, rows relative to the file, NeMo's audio_filepath and duration, a zip of the transcripts and clips, and audio in another zip, each utterance's text, seconds (the row's or the header's) and bytes; MinSeconds and MaxSeconds leave rows out with a note; a missing file refused naming the row unless SkipMissing; \"audio-folder\" pairs name.wav with name.txt in a folder and a zip, noting files without text; the JSON round trip; an unregistered kind refused; Order a seeded permutation per pass", AudioTrainSpeechSources),
        ("audio train: AudioAugmentations: \"speed\" makes the sound 1/factor as long (a factor of 1 the same sound), \"noise\" adds white noise and a noise file's stretch at the drawn signal-to-noise ratio, \"spec-mask\" fills at most its masks' widths with the mean feature and leaves the rest; RandomFor gives the same draws for the same seed, pass and utterance and others for another pass; the options' JSON round trip and refusals; an unregistered name refused", AudioTrainAugmentations),
        ("audio train: the IdrakAudio plug-in reads ctc/model and ctc/model-conv (refused naming SpeechFamilies.Register without it) and gives torch's step-1 CTC loss within 1e-4 and every tensor's gradient within 1e-4 of that tensor's largest, through CtcObjective.Loss; three SGD steps' losses within 1e-4 and the weights after them within 1e-4; Viterbi on the model's own log-probabilities gives torchaudio's paths; each utterance's logits in the padded batch equal its logits alone within 1e-5; a config that does not match the weights refused", AudioTrainFixture),
        ("audio train: SpeechModels.Create makes an IdrakAudio ctc model from a spec and a trained tokenizer into a folder SpeechModels.Load reads back with the same weights (an unregistered architecture, a used folder, an unbuilt task and the tts task refused, the last naming SpeechSynthesisModels.Create); it transcribes tone speech with greedy and beam decoding, batch and chunked and streamed alike, and passes the kit's SpeechRecognizerSuite", AudioTrainCreate),
        ("audio train: SpeechTrainer: a run stopped after step 2 and resumed from checkpoint-2 gives steps 3 to 6 and the final weights bit for bit (augmentations on, an epoch boundary crossed); checkpoints are models SpeechModels.Load reads with train-state.json; the feature cache takes each utterance's features once (no AudioFeatures stage after the first pass); an utterance too short for its text is counted and left out; progress reports every step with the time left", AudioTrainResume),
        ("audio train: BatchSeconds 0 sizes the batch automatically: the longest utterance is probed through a step's passes without an optimizer step, the note names the measured batch (at least the longest utterance, at most one pass of the audio), the run trains in it and its checkpoint resumes with it", AudioTrainAutomaticBatch),
        ("audio train: SpeechTrainer halves the micro-batch when the device runs out of memory (a model refusing more than two utterances at once) and repeats the step: the same losses within 1e-5 as a run that never ran out, the note and the final micro-batch reported; with gradient accumulation of two batches a step, the run one utterance at a time gives the same losses within 1e-5", AudioTrainMemory),
        ("audio train: a 200-step CPU run of a tiny IdrakAudio ctc model on tone speech halves its loss (the mean of the last 10 steps under half of step 1), its evaluation CER at step 200 is at most its CER at step 100 and under 0.5, and the trained checkpoint transcribes (step time and CER printed)", AudioTrainSmoke),
    ];

    private static string CtcData(string relative) => AudioData(Path.Combine("ctc", relative));

    private static JsonObject CtcReference => JsonNode.Parse(File.ReadAllText(CtcData("reference/loss.json")))!.AsObject();

    private static void AudioTrainViterbi(Device device)
    {
        var reference = CtcReference;
        var alignment = JsonNode.Parse(File.ReadAllText(CtcData("reference/alignment.json")))!.AsObject();
        int classes = (int)reference["classes"]!, blank = (int)reference["blank"]!;
        int[][] labels = [.. reference["labels"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
        int[] steps = [.. reference["results"]!["model"]!["steps"]!.AsArray().Select(v => (int)v!)];
        var logp = ReadNpyFloat32(CtcData("reference/logp-step1.npy"));
        int padded = steps.Max();
        Check(logp.Length == labels.Length * padded * classes, $"logp-step1.npy holds {logp.Length} values");
        for (int n = 0; n < labels.Length; n++)
        {
            var rows = logp.AsSpan(n * padded * classes, steps[n] * classes);
            var path = CtcAlignment.Viterbi(rows, steps[n], classes, labels[n], blank);
            int[] expected = [.. alignment["alignments"]![n]!.AsArray().Select(v => (int)v!)];
            Check(path.Frames.SequenceEqual(expected), $"utterance {n}: the path {string.Join(",", path.Frames)} is not torchaudio's {string.Join(",", expected)}");
            float score = (float)alignment["path_log_probs"]![n]!;
            Check(MathF.Abs(path.LogProbability - score) <= 1e-4f * MathF.Abs(score), $"utterance {n}: the path's log-probability {path.LogProbability} is not {score}");
            float sum = 0;
            for (int t = 0; t < steps[n]; t++)
            {
                sum += rows[t * classes + path.Frames[t]];
            }

            Check(MathF.Abs(sum - path.LogProbability) <= 1e-4f * MathF.Abs(sum), $"utterance {n}: the log-probability {path.LogProbability} is not the frames' sum {sum}");
            for (int l = 0; l < labels[n].Length; l++)
            {
                Check(path.Lengths[l] > 0 && Enumerable.Range(path.Starts[l], path.Lengths[l]).All(t => path.Frames[t] == labels[n][l]),
                    $"utterance {n}, label {l}: steps {path.Starts[l]} + {path.Lengths[l]} do not read label {labels[n][l]}");
                Check(l == 0 || path.Starts[l] >= path.Starts[l - 1] + path.Lengths[l - 1], $"utterance {n}: label {l} starts inside label {l - 1}");
            }

            Check(path.Lengths.Sum() == path.Frames.Count(f => f != blank), $"utterance {n}: the labels' lengths do not cover the label steps");
        }

        // The same path from the tensor overload.
        using (var tensor = Tensor.From(logp.AsSpan(0, steps[0] * classes).ToArray(), [steps[0], classes], device))
        {
            var path = CtcAlignment.Viterbi(tensor, labels[0], blank);
            Check(path.Frames.SequenceEqual(alignment["alignments"]![0]!.AsArray().Select(v => (int)v!)), "the tensor overload gives the same path");
        }

        Check(CtcAlignment.MinimumSteps([4, 4, 4]) == 5 && CtcAlignment.MinimumSteps([1, 2, 2, 3]) == 5 && CtcAlignment.MinimumSteps([]) == 0, "MinimumSteps");
        var few = logp.AsSpan(0, 4 * classes).ToArray();
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [4, 4, 4], blank), "four steps for three equal labels", "cannot hold", "need 5");
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [blank, 3], blank), "a blank label", "is the blank");
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [classes], blank), "a label out of range", $"not one of the {classes} classes");
    }

    private static void AudioTrainSpeechSources(Device device)
    {
        string root = TempFolder("speech-sources");
        try
        {
            // The fixture's transcripts: paths relative to the audio folder (data/audio), given as the source's Audio.
            using (var fixture = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [AudioData("transcripts/train.jsonl")]) { Audio = AudioData("") }))
            {
                var rows = File.ReadAllLines(AudioData("transcripts/train.jsonl")).Select(l => JsonNode.Parse(l)!.AsObject()).ToArray();
                Check(fixture.Count == rows.Length && fixture.Notes.Count == 0, $"the fixture's {rows.Length} rows read as {fixture.Count} ({string.Join("; ", fixture.Notes)})");
                for (int i = 0; i < rows.Length; i++)
                {
                    string file = AudioData((string)rows[i]["audio"]!);
                    var info = AudioCodecs.ReadInfo(file)!.Value;
                    Check(fixture[i].Text == (string)rows[i]["text"]! && Math.Abs(fixture[i].Seconds - info.Samples / (double)info.SampleRate) < 1e-9,
                        $"row {i}: {fixture[i].Text} for {fixture[i].Seconds} s");
                    Check(fixture.ReadBytes(i).AsSpan().SequenceEqual(File.ReadAllBytes(file)), $"row {i}: the bytes are the file's");
                }
            }

            // Rows relative to their file, NeMo's keys, a row's own seconds, an escaped text.
            string clips = Path.Combine(root, "clips");
            Directory.CreateDirectory(clips);
            File.Copy(AudioData("wav/pcm16.wav"), Path.Combine(clips, "a.wav"));
            File.Copy(AudioData("wav/rate8k.wav"), Path.Combine(clips, "b.wav"));
            File.Copy(AudioData("wav/stereo16.wav"), Path.Combine(clips, "c.wav"));
            string jsonl = Path.Combine(root, "train.jsonl");
            File.WriteAllText(jsonl, "\uFEFF{\"audio\": \"clips/a.wav\", \"text\": \"first \\\"one\\\"\", \"speaker\": {\"id\": 3}}\r\n"
                + "\n{\"audio_filepath\": \"clips/b.wav\", \"duration\": 2.25, \"text\": \"second\"}\n"
                + "{\"text\": \"third\", \"audio\": \"./clips/../clips/c.wav\"}");
            using (var reader = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [jsonl])))
            {
                Check(reader.Count == 3 && reader[0].Text == "first \"one\"" && reader[1].Text == "second" && reader[2].Text == "third", "the three rows and their texts");
                Check(reader[1].Seconds == 2.25 && Math.Abs(reader[0].Seconds - 1.5) < 1e-9 && reader[0].Name.EndsWith("train.jsonl:1", StringComparison.Ordinal),
                    $"seconds {reader[0].Seconds}, {reader[1].Seconds}; name {reader[0].Name}");
                Check(reader.ReadBytes(2).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(clips, "c.wav"))), "a path with . and .. resolves");
            }

            using (var limited = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [jsonl]) { MinSeconds = 1.6, MaxSeconds = 2 }))
            {
                Check(limited.Count == 0 && limited.Notes.Count == 2 && limited.Notes.Any(n => n.Contains("shorter than 1.6", StringComparison.Ordinal))
                    && limited.Notes.Any(n => n.Contains("1 utterances longer than 2", StringComparison.Ordinal)), $"the limits' notes: {string.Join("; ", limited.Notes)}");
            }

            // A zip of the transcripts and clips (the jsonl in a folder of the zip), and a transcript whose audio is another zip.
            string zip = Path.Combine(root, "data.zip");
            using (var archive = ZipFile.Open(zip, ZipArchiveMode.Create))
            {
                archive.CreateEntryFromFile(Path.Combine(clips, "a.wav"), "set/clips/a.wav");
                archive.CreateEntryFromFile(Path.Combine(clips, "b.wav"), "set/clips/b.wav");
                using var writer = new StreamWriter(archive.CreateEntry("set/train.jsonl").Open());
                writer.Write("{\"audio\": \"clips/a.wav\", \"text\": \"zipped a\"}\n{\"audio\": \"clips/b.wav\", \"text\": \"zipped b\"}\n");
            }

            using (var zipped = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [zip])))
            {
                Check(zipped.Count == 2 && zipped[1].Text == "zipped b" && zipped[1].Audio.EndsWith("data.zip!set/clips/b.wav", StringComparison.Ordinal), $"the zip's rows ({zipped.Count}, {zipped[0].Audio})");
                var info = AudioCodecs.ReadInfo(Path.Combine(clips, "b.wav"))!.Value;
                Check(Math.Abs(zipped[1].Seconds - info.Samples / (double)info.SampleRate) < 1e-9, "an entry's seconds from its header");
                Check(zipped.ReadBytes(0).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(clips, "a.wav"))), "an entry's bytes");
            }

            string loose = Path.Combine(root, "loose.jsonl");
            File.WriteAllText(loose, "{\"audio\": \"set/clips/a.wav\", \"text\": \"from the audio zip\"}\n");
            using (var other = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [loose]) { Audio = zip }))
            {
                Check(other.Count == 1 && other.ReadBytes(0).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(clips, "a.wav"))), "audio in the source's Audio zip");
            }

            string missing = Path.Combine(root, "missing.jsonl");
            File.WriteAllText(missing, "{\"audio\": \"clips/a.wav\", \"text\": \"here\"}\n{\"audio\": \"clips/gone.wav\", \"text\": \"gone\"}\n");
            ThrowsNaming<FileNotFoundException>(() => SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [missing])).Dispose(), "a missing audio file", "missing.jsonl:2", "gone.wav", "SkipMissing");
            using (var skipped = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [missing]) { SkipMissing = true }))
            {
                Check(skipped.Count == 1 && skipped.Notes.Single().StartsWith("1 rows name audio that was not found", StringComparison.Ordinal), $"skipped: {string.Join("; ", skipped.Notes)}");
            }

            string bad = Path.Combine(root, "bad.jsonl");
            File.WriteAllText(bad, "{\"audio\": \"clips/a.wav\"}\n");
            ThrowsNaming<InvalidDataException>(() => SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [bad])).Dispose(), "a row without text", "bad.jsonl, line 1", "\"text\"");

            // "audio-folder": name.wav beside name.txt; a sound without text is left out with a note.
            File.WriteAllText(Path.Combine(clips, "a.txt"), "\uFEFF the a clip \n");
            File.WriteAllText(Path.Combine(clips, "c.txt"), "the c clip");
            using (var folder = SpeechSources.Open(new SpeechSource(SpeechSources.AudioFolder, [clips])))
            {
                Check(folder.Count == 2 && folder[0].Text == "the a clip" && folder[1].Text == "the c clip" && folder.Notes.Single().StartsWith("1 sound files have no .txt", StringComparison.Ordinal),
                    $"the folder's pairs ({folder.Count}: {string.Join("; ", folder.Notes)})");
                Check(folder.ReadBytes(1).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(clips, "c.wav"))), "the folder's bytes");
            }

            string folderZip = Path.Combine(root, "folder.zip");
            ZipFile.CreateFromDirectory(clips, folderZip);
            using (var zippedFolder = SpeechSources.Open(new SpeechSource(SpeechSources.AudioFolder, [folderZip])))
            {
                Check(zippedFolder.Count == 2 && zippedFolder[1].Text == "the c clip" && Math.Abs(zippedFolder[0].Seconds - 1.5) < 1e-9, $"the zipped folder ({zippedFolder.Count})");
            }

            var spec = new SpeechSource(SpeechSources.Transcripts, [jsonl, zip]) { Audio = clips, MinSeconds = 0.5, MaxSeconds = 20, SkipMissing = true };
            var back = SpeechSource.FromJson(JsonNode.Parse(spec.ToJson().ToJsonString())!.AsObject());
            Check(back.Paths.SequenceEqual(spec.Paths) && back with { Paths = spec.Paths } == spec, "the source's JSON round trip");
            ThrowsNaming<NotSupportedException>(() => SpeechSources.Open(new SpeechSource("podcasts", [jsonl])), "an unregistered kind", "'podcasts' is not registered", "SpeechSources.Register");
            Check(SpeechSources.Names.Contains(SpeechSources.Transcripts) && SpeechSources.Origin(SpeechSources.AudioFolder) == Idrak.Abstraction.Overrides.Library, "the built-in kinds");

            var order = SpeechSources.Order(50, 7, 2);
            Check(order.Order().SequenceEqual(Enumerable.Range(0, 50)) && order.SequenceEqual(SpeechSources.Order(50, 7, 2)) && !order.SequenceEqual(SpeechSources.Order(50, 7, 3))
                && !order.SequenceEqual(SpeechSources.Order(50, 8, 2)) && !order.SequenceEqual(Enumerable.Range(0, 50)), "Order: a seeded permutation per pass");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioTrainAugmentations(Device device)
    {
        var tone = new AudioData(ReadNpyFloat32(AudioData("wav/pcm16.npy")), 1, 16000);
        var options = new AudioAugmentationOptions();
        var speed = (ISoundAugmentation)AudioAugmentations.Get(AudioAugmentations.Speed);
        Check(speed.Stage == AudioAugmentationStage.Sound, "speed works on the sound");
        foreach (double factor in new[] { 0.9, 1.1 })
        {
            var faster = speed.Augment(tone, new Random(1), options with { SpeedFactors = [factor] });
            long expected = Resampler.OutputLength(tone.Length, (int)Math.Round(16000 * factor), 16000);
            Check(faster.SampleRate == 16000 && faster.Length == expected && Math.Abs(faster.Length - tone.Length / factor) <= 1, $"speed {factor}: {faster.Length} samples for {tone.Length}");
        }

        Check(ReferenceEquals(speed.Augment(tone, new Random(1), options with { SpeedFactors = [1.0] }), tone), "a speed factor of 1 leaves the sound");

        static double SnrDb(float[] clean, float[] noisy)
        {
            double signal = 0, noise = 0;
            for (int i = 0; i < clean.Length; i++)
            {
                signal += clean[i] * (double)clean[i];
                noise += (noisy[i] - clean[i]) * (double)(noisy[i] - clean[i]);
            }

            return 10 * Math.Log10(signal / noise);
        }

        var noise = (ISoundAugmentation)AudioAugmentations.Get(AudioAugmentations.Noise);
        var at20 = options with { MinSnrDb = 20, MaxSnrDb = 20 };
        var white = noise.Augment(tone, new Random(3), at20);
        Check(white.Length == tone.Length && Math.Abs(SnrDb(tone.Samples, white.Samples) - 20) < 0.5, $"white noise at 20 dB measures {SnrDb(tone.Samples, white.Samples):F2} dB");
        var fromFile = noise.Augment(tone, new Random(3), at20 with { NoiseFiles = [AudioData("wav/rate8k.wav")] });
        Check(Math.Abs(SnrDb(tone.Samples, fromFile.Samples) - 20) < 0.01, $"file noise at 20 dB measures {SnrDb(tone.Samples, fromFile.Samples):F3} dB");
        Check(noise.Augment(tone, new Random(3), at20).Samples.SequenceEqual(white.Samples), "the same draws give the same noise");

        var mask = (IFeatureAugmentation)AudioAugmentations.Get(AudioAugmentations.SpecMask);
        Check(mask.Stage == AudioAugmentationStage.Features, "spec-mask works on the features");
        int frames = 200, mels = 80;
        var features = RandomArray(new Random(5), frames * mels);
        float mean = (float)(features.Sum(f => (double)f) / features.Length);
        var masked = features.ToArray();
        var limits = options with { TimeMasks = 1, TimeMaskFrames = 30, TimeMaskFraction = 0.1, FrequencyMasks = 1, FrequencyMaskBands = 12 };
        mask.Augment(masked, frames, mels, new Random(11), limits);
        int changedFrames = Enumerable.Range(0, frames).Count(t => Enumerable.Range(0, mels).All(m => masked[t * mels + m] != features[t * mels + m]));
        int changedBands = Enumerable.Range(0, mels).Count(m => Enumerable.Range(0, frames).All(t => masked[t * mels + m] != features[t * mels + m]));
        Check(changedFrames <= 20 && changedBands <= 12 && changedFrames + changedBands > 0, $"masks of {changedFrames} frames and {changedBands} bands (at most 20 and 12)");
        Check(Enumerable.Range(0, masked.Length).All(i => masked[i] == features[i] || masked[i] == mean), "a masked value is the mean feature, the rest unchanged");

        var a = AudioAugmentations.RandomFor(5, 2, 9, "noise");
        var b = AudioAugmentations.RandomFor(5, 2, 9, "NOISE");
        var c = AudioAugmentations.RandomFor(5, 3, 9, "noise");
        var first = Enumerable.Range(0, 8).Select(_ => a.Next()).ToArray();
        Check(first.SequenceEqual(Enumerable.Range(0, 8).Select(_ => b.Next())) && !first.SequenceEqual(Enumerable.Range(0, 8).Select(_ => c.Next())),
            "RandomFor: the same draws for the same seed, pass and utterance, others for another pass");

        var json = new AudioAugmentationOptions { Probability = 0.5, SpeedFactors = [0.95, 1.05], NoiseFiles = ["n.wav"], TimeMasks = 3 };
        var back = AudioAugmentationOptions.FromJson(JsonNode.Parse(json.ToJson().ToJsonString())!.AsObject());
        Check(back.SpeedFactors.SequenceEqual(json.SpeedFactors) && back.NoiseFiles.SequenceEqual(json.NoiseFiles) && back with { SpeedFactors = json.SpeedFactors, NoiseFiles = json.NoiseFiles } == json,
            "the options' JSON round trip");
        ThrowsNaming<ArgumentException>(() => (options with { Probability = 2 }).Check(), "a probability of 2", "Probability is 0 to 1");
        ThrowsNaming<NotSupportedException>(() => AudioAugmentations.Get("reverb"), "an unregistered augmentation", "'reverb' is not registered", "AudioAugmentations.Register");
        Check(AudioAugmentations.Names.Count >= 3 && AudioAugmentations.Origin(AudioAugmentations.SpecMask) == Idrak.Abstraction.Overrides.Library, "the built-in augmentations");
    }

    // ---------------------------------------------------------------- the plug-in against torch

    private static (float[] Mels, int[] Frames, int[][] Labels) CtcInputs()
    {
        var reference = CtcReference;
        int[] frames = [.. reference["frames"]!.AsArray().Select(v => (int)v!)];
        int[][] labels = [.. reference["labels"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
        return (ReadNpyFloat32(CtcData("mels.npy")), frames, labels);
    }

    private static Dictionary<string, (int[] Shape, float[] Values)> ReadSafeTensors(string path)
    {
        using var reader = SafeTensorsReader.Open(path);
        ITensorStore store = reader;
        return store.Names.ToDictionary(n => n, n => (store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal);
    }

    // Each tensor's values within `relative` of that tensor's largest expected value.
    private static void TensorsClose(Dictionary<string, (int[] Shape, float[] Values)> expected, IEnumerable<(string Name, float[] Values)> actual, float relative, string what)
    {
        int seen = 0;
        foreach (var (name, values) in actual)
        {
            Check(expected.TryGetValue(name, out var reference), $"{what}: no reference for {name}");
            CloseRelative(reference.Values, values, relative, $"{what} {name}");
            seen++;
        }

        Check(seen == expected.Count, $"{what}: {seen} tensors for the reference's {expected.Count}");
    }

    private static IEnumerable<(string Name, float[] Values)> Gradients(IdrakAudioCtcModel model) =>
        model.NamedParameters().Select(p => (p.Name, IdrakAudioWeights.Stored(p.Parameter.Grad ?? throw new Exception($"{p.Name} has no gradient"), p.Transposed)));

    private static void AudioTrainFixture(Device device)
    {
        IdrakAudioPlugin.Unregister();
        ThrowsNaming<NotSupportedException>(() => SpeechModels.Load(CtcData("model")).Dispose(), "without the plug-in", "'IdrakAudio' is not registered", "SpeechFamilies.Register");
        IdrakAudioPlugin.Register();
        try
        {
            var (mels, frames, labels) = CtcInputs();
            var reference = CtcReference;
            int padded = frames.Max();
            foreach (var (folder, grads) in new[] { ("model", "grads-step1.safetensors"), ("model-conv", "grads-step1-conv.safetensors") })
            {
                using var model = (IdrakAudioCtcModel)SpeechModels.Load(CtcData(folder), new SpeechModelOptions { Device = device });
                model.Network.Train();
                using (var scope = new TensorScope())
                {
                    var input = Tensor.From(mels, [frames.Length, padded, 80], device);
                    var loss = CtcObjective.Loss(model, input, frames, labels);
                    float expected = (float)reference["results"]![folder]!["loss_step1"]!;
                    Check(MathF.Abs(loss.Item() - expected) <= 1e-4f * MathF.Abs(expected), $"{folder}: the step-1 loss {loss.Item()} is not torch's {expected}");
                    loss.Backward();
                    TensorsClose(ReadSafeTensors(CtcData("reference/" + grads)), Gradients(model), 1e-4f, $"{folder} step-1 gradient");
                }

                // Each utterance alone gives the batch's logits on its own steps (its padding is not attended to).
                using (var scope = new TensorScope())
                using (Autograd.NoGrad())
                {
                    var batch = model.Logits(Tensor.From(mels, [frames.Length, padded, 80], device), frames).ToArray();
                    int steps = model.StepsFor(padded), classes = model.Classes;
                    for (int n = 0; n < frames.Length; n++)
                    {
                        var alone = model.Logits(Tensor.From(mels.AsSpan(n * padded * 80, frames[n] * 80), [1, frames[n], 80], device), [frames[n]]).ToArray();
                        int own = model.StepsFor(frames[n]);
                        CloseRelative(alone[..(own * classes)], batch.AsSpan(n * steps * classes, own * classes).ToArray(), 1e-5f, $"{folder} utterance {n} alone against in the batch");
                    }
                }
            }

            // Three SGD steps as torch took them, and the weights after them; the alignment of the model's own log-probabilities.
            using (var model = (IdrakAudioCtcModel)SpeechModels.Load(CtcData("model"), new SpeechModelOptions { Device = device }))
            {
                model.Network.Train();
                using var sgd = new Sgd(model.Network.Parameters(), 0.1f, 0f, 0f);
                float[] expected = [.. reference["results"]!["model"]!["sgd_losses"]!.AsArray().Select(v => (float)v!)];
                var losses = new List<float>();
                for (int s = 0; s < 4; s++)
                {
                    using var scope = new TensorScope();
                    if (s > 0)
                    {
                        sgd.Step();
                    }

                    sgd.ZeroGrad();
                    var loss = CtcObjective.Loss(model, Tensor.From(mels, [frames.Length, padded, 80], device), frames, labels);
                    losses.Add(loss.Item());
                    loss.Backward();
                }

                for (int s = 0; s < 4; s++)
                {
                    Check(MathF.Abs(losses[s] - expected[s]) <= 1e-4f * MathF.Abs(expected[s]), $"SGD loss {s}: {losses[s]}, torch {expected[s]} ({string.Join(", ", losses)})");
                }

                TensorsClose(ReadSafeTensors(CtcData("reference/weights-step3.safetensors")), model.Export().Select(e => (e.Name, e.Values)), 1e-4f, "the weights after 3 SGD steps");
            }

            using (var model = (IdrakAudioCtcModel)SpeechModels.Load(CtcData("model"), new SpeechModelOptions { Device = device }))
            using (var scope = new TensorScope())
            using (Autograd.NoGrad())
            {
                var alignment = JsonNode.Parse(File.ReadAllText(CtcData("reference/alignment.json")))!.AsObject();
                var logp = model.Logits(Tensor.From(mels, [frames.Length, padded, 80], device), frames).LogSoftmax().ToArray();
                int steps = model.StepsFor(padded), classes = model.Classes;
                for (int n = 0; n < frames.Length; n++)
                {
                    var path = CtcAlignment.Viterbi(logp.AsSpan(n * steps * classes, model.StepsFor(frames[n]) * classes), model.StepsFor(frames[n]), classes, labels[n], model.Blank);
                    Check(path.Frames.SequenceEqual(alignment["alignments"]![n]!.AsArray().Select(v => (int)v!)), $"utterance {n}: the model's own path differs from torchaudio's");
                }
            }

            // A config that does not match the weights: the convolution module's tensors unread.
            string root = TempFolder("ctc-mismatch");
            try
            {
                foreach (string file in Directory.GetFiles(CtcData("model-conv")))
                {
                    File.Copy(file, Path.Combine(root, Path.GetFileName(file)));
                }

                var config = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "config.json")))!.AsObject();
                config["convolution_module"] = false;
                File.WriteAllText(Path.Combine(root, "config.json"), config.ToJsonString());
                ThrowsNaming<InvalidDataException>(() => SpeechModels.Load(root).Dispose(), "conv weights read without the module", "does not use", "layers.0.conv");
                config["convolution_module"] = true;
                config["dim"] = 48;
                File.WriteAllText(Path.Combine(root, "config.json"), config.ToJsonString());
                ThrowsNaming<InvalidDataException>(() => SpeechModels.Load(root).Dispose(), "a width the weights do not have", "the config gives");
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
        }
    }

    // ---------------------------------------------------------------- tone speech: letters as tone bursts

    private const string ToneLetters = "abcdefgh";

    // `count` utterances of 2 to 6 letters, each letter a 0.12 s tone of its own pitch (a Hann envelope) and 0.04 s of
    // silence, 0.1 s of silence around them, a little seeded noise: transcripts.jsonl and the WAV files in `folder`.
    private static (string Transcripts, string[] Texts) ToneSpeech(string folder, int count, int seed, int minLetters = 2, int maxLetters = 6)
    {
        Directory.CreateDirectory(folder);
        var random = new Random(seed);
        var texts = new string[count];
        var rows = new System.Text.StringBuilder();
        const int Rate = 16000;
        for (int u = 0; u < count; u++)
        {
            int letters = random.Next(minLetters, maxLetters + 1);
            var text = new char[letters];
            for (int i = 0; i < letters; i++)
            {
                text[i] = ToneLetters[(u + i * 3 + random.Next(ToneLetters.Length)) % ToneLetters.Length];
            }

            texts[u] = new string(text);
            int tone = (int)(0.12 * Rate), gap = (int)(0.04 * Rate), edge = (int)(0.1 * Rate);
            var samples = new float[2 * edge + letters * (tone + gap)];
            for (int i = 0; i < letters; i++)
            {
                double frequency = 300 + 150 * ToneLetters.IndexOf(text[i], StringComparison.Ordinal);
                int start = edge + i * (tone + gap);
                for (int j = 0; j < tone; j++)
                {
                    double envelope = 0.5 - 0.5 * Math.Cos(2 * Math.PI * j / (tone - 1));
                    samples[start + j] = (float)(0.5 * envelope * Math.Sin(2 * Math.PI * frequency * j / Rate));
                }
            }

            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] += (float)(0.003 * (random.NextDouble() * 2 - 1));
            }

            AudioEncoders.Save(Path.Combine(folder, $"u{u:D3}.wav"), new AudioData(samples, 1, Rate));
            rows.Append($"{{\"audio\": \"u{u:D3}.wav\", \"text\": \"{texts[u]}\"}}\n");
        }

        string transcripts = Path.Combine(folder, "transcripts.jsonl");
        File.WriteAllText(transcripts, rows.ToString());
        return (transcripts, texts);
    }

    // A character tokenizer for the tone letters: Metaspace BPE trained to its alphabet alone (no merges): the letters, "▁"
    // and the line end (the trainer reads each line with its end, as Hugging Face's train_from_files does).
    private static IPretrainedTokenizer ToneTokenizer(string folder, IEnumerable<string> texts)
    {
        string file = Path.Combine(folder, "texts.txt");
        File.WriteAllLines(file, texts);
        int alphabet = string.Concat(texts).Distinct().Count() + 2;
        return TokenizerTrainers.Get(TokenizerTrainers.Bpe).Train([file],
            new TokenizerTrainingOptions { VocabularySize = alphabet, MinFrequency = 1, PreTokenizer = TokenizerTrainers.Metaspace, Workers = 1 });
    }

    private static JsonObject ToneSpec(bool convolution = false) => new()
    {
        ["architecture"] = IdrakAudioFamily.Architecture, ["task"] = "ctc", ["mels"] = 80, ["dim"] = 32, ["layers"] = 2, ["heads"] = 4, ["ffn_dim"] = 64,
        ["subsample_layers"] = 2, ["subsample_kernel"] = 3, ["convolution_module"] = convolution, ["convolution_kernel"] = 5,
    };

    private static IdrakAudioCtcModel CreateTone(string folder, IPretrainedTokenizer tokenizer, Device device, int seed = 7, bool convolution = false) =>
        (IdrakAudioCtcModel)SpeechModels.Create(folder, ToneSpec(convolution), tokenizer, new SpeechCreateOptions { Device = device, Seed = seed });

    private static void AudioTrainCreate(Device device)
    {
        string root = TempFolder("ctc-create");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 6, 11);
            var tokenizer = ToneTokenizer(root, texts);
            Check(tokenizer.VocabularySize == texts.SelectMany(t => t).Distinct().Count() + 2 && tokenizer.IdOf("▁") is not null, $"the tone tokenizer has {tokenizer.VocabularySize} tokens ({string.Join(" ", Enumerable.Range(0, tokenizer.VocabularySize).Select(tokenizer.TokenOf))}) for the texts {string.Join(" ", texts)}");
            string folder = Path.Combine(root, "model");
            using var model = CreateTone(folder, tokenizer, device, convolution: true);
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            Check((string?)config["architectures"]![0] == "IdrakAudio" && (int)config["classes"]! == tokenizer.VocabularySize + 1 && File.Exists(Path.Combine(folder, "tokenizer.json")),
                "config.json names the architecture and the classes; tokenizer.json is written");
            Check(model.Labels(texts[0]).Length == texts[0].Length + 1 && model.Text(model.Labels(texts[0])) == texts[0], $"labels of '{texts[0]}' read back as '{model.Text(model.Labels(texts[0]))}'");
            using (var again = (IdrakAudioCtcModel)SpeechModels.Load(folder, new SpeechModelOptions { Device = device }))
            {
                Check(again.Export().Zip(model.Export()).All(p => p.First.Name == p.Second.Name && p.First.Values.AsSpan().SequenceEqual(p.Second.Values)), "the folder reads back to the same weights");
            }

            using (var same = CreateTone(Path.Combine(root, "same"), tokenizer, device, convolution: true))
            {
                Check(same.Export().Zip(model.Export()).All(p => p.First.Values.AsSpan().SequenceEqual(p.Second.Values)), "the same seed draws the same weights");
            }

            ThrowsNaming<IOException>(() => CreateTone(folder, tokenizer, device).Dispose(), "a used folder", "already holds files");
            ThrowsNaming<NotSupportedException>(() => SpeechModels.Create(Path.Combine(root, "other"), new JsonObject { ["architecture"] = "NoSuchAudio" }, tokenizer).Dispose(),
                "an unregistered architecture", "'NoSuchAudio' is not registered");
            ThrowsNaming<NotSupportedException>(() => SpeechModels.Create(Path.Combine(root, "tts"), new JsonObject { ["architecture"] = "IdrakAudio", ["task"] = "tts" }, tokenizer).Dispose(),
                "a synthesizer task", "task 'tts' is a speech synthesizer", "SpeechSynthesisModels.Create");
            ThrowsNaming<NotSupportedException>(() => SpeechModels.Create(Path.Combine(root, "other-task"), new JsonObject { ["architecture"] = "IdrakAudio", ["task"] = "no-such-task" }, tokenizer).Dispose(),
                "an unbuilt task", "task 'no-such-task' is not built yet");

            using var reader = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [transcripts]));
            var audio = AudioCodecs.Decode(reader.ReadBytes(0));
            var greedy = model.Transcribe(audio);
            var beam = model.Transcribe(audio, new TranscriptionOptions { Beams = 4 });
            Check(greedy.Segments.Count == 1 && greedy.Segments[0].End == audio.Duration.TotalSeconds && beam.Segments.Count == 1, "one segment for a short sound");
            var chunked = model.Transcribe(audio, new TranscriptionOptions { ChunkSeconds = 0.4 });
            int pieces = (int)Math.Ceiling(audio.Duration.TotalSeconds / 0.4 - 1e-9);
            Check(chunked.Segments.Count == pieces && chunked.Segments.Zip(chunked.Segments.Skip(1)).All(p => Math.Abs(p.First.End - p.Second.Start) < 1e-9), $"{chunked.Segments.Count} chunks of 0.4 s for {pieces}");
            async IAsyncEnumerable<AudioData> Pieces()
            {
                var samples = audio.Samples;
                for (int start = 0; start < samples.Length; start += 1000)
                {
                    await Task.Yield();
                    yield return new AudioData(samples[start..Math.Min(samples.Length, start + 1000)], 1, audio.SampleRate);
                }
            }

            var streamed = new List<TranscriptSegment>();
            var enumerator = model.TranscribeStream(Pieces(), new TranscriptionOptions { ChunkSeconds = 0.4 }).GetAsyncEnumerator();
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                streamed.Add(enumerator.Current);
            }

            Check(streamed.Select(s => s.Text).SequenceEqual(chunked.Segments.Select(s => s.Text)) && streamed.Select(s => s.Start).SequenceEqual(chunked.Segments.Select(s => s.Start)),
                "the stream gives the chunks' segments");
            var report = Conformance.CheckSpeechRecognizer(new SpeechRecognizerUnderTest(16000, samples =>
            {
                var t = model.Transcribe(new AudioData(samples, 1, 16000));
                return new RecognizedSpeech(t.Text, [.. t.Segments.Select(s => new RecognizedSegment(s.Start, s.End, s.Text))]);
            }), silenceMaxChars: 1000);
            Check(report.Passed, report.ToString());
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    // ---------------------------------------------------------------- the trainer

    private sealed class FeatureStages : ITelemetryHook
    {
        private int _count;

        public int Count => Volatile.Read(ref _count);

        public TelemetryLevel Levels => TelemetryLevel.Inference;

        public void OnInferenceStage(in InferenceStage e)
        {
            if (e.Stage == InferenceStageKind.AudioFeatures)
            {
                Interlocked.Increment(ref _count);
            }
        }
    }

    private static SpeechTrainingOptions ToneTraining(string cache) => new()
    {
        Steps = 6, BatchSeconds = 3, LearningRate = 3e-3f, WarmupFraction = 0.1f, Seed = 3, FeatureCache = FeatureCaches.Disk, FeatureCacheFolder = cache,
    };

    private static void AudioTrainResume(Device device)
    {
        string root = TempFolder("ctc-resume");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 12, 21);
            // One more utterance whose text is far longer than its sound can hold.
            File.Copy(Path.Combine(root, "data", "u000.wav"), Path.Combine(root, "data", "short.wav"));
            File.AppendAllText(transcripts, "{\"audio\": \"short.wav\", \"text\": \"" + new string('a', 80) + "\"}\n");
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = ToneTraining(Path.Combine(root, "cache")) with { Augmentations = ["speed", "noise", "spec-mask"], AugmentationOptions = new() { Probability = 0.5 }, SaveEvery = 2 };

            // The whole run.
            SpeechTrainingResult whole;
            var reports = new List<SpeechTrainingProgress>();
            using (var model = CreateTone(Path.Combine(root, "start-a"), tokenizer, device))
            {
                whole = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"), progress: new Progress(reports));
            }

            Check(whole.Steps == 6 && whole.Losses.Count == 6 && whole.Losses.All(float.IsFinite) && whole.Dropped == 1 && whole.Utterances == 12,
                $"the run: {whole.Steps} steps, {whole.Dropped} dropped, {whole.Utterances} used");
            Check(whole.Notes.Any(n => n.Contains("1 utterances give fewer steps than their labels need", StringComparison.Ordinal)), $"the dropped row is noted: {string.Join("; ", whole.Notes)}");
            var steps = reports.Where(r => r.Stage == "step").ToList();
            Check(steps.Select(r => r.Step).SequenceEqual(Enumerable.Range(1, 6)) && steps[^1].Remaining == TimeSpan.Zero && steps.Any(r => r.Epoch == 2)
                && steps[2].Remaining > TimeSpan.Zero && steps.All(r => r.MicroBatchSeconds == 3) && steps.All(r => r.BatchAudioSeconds > 0),
                "progress after every step, the time left, the step's audio seconds, an epoch boundary crossed");
            foreach (int step in new[] { 2, 4, 6 })
            {
                Check(File.Exists(Path.Combine(root, "run-a", $"checkpoint-{step}", SpeechTrainer.StateFile)), $"checkpoint-{step} holds its state");
            }

            // The same run stopped after step 2 (cancelled: the checkpoint of the last finished step is written), then resumed.
            using (var model = CreateTone(Path.Combine(root, "start-b"), tokenizer, device))
            using (var stop = new CancellationTokenSource())
            {
                var seen = new List<SpeechTrainingProgress>();
                var stopper = new Progress(seen, r =>
                {
                    if (r.Stage == "step" && r.Step == 2)
                    {
                        stop.Cancel();
                    }
                });
                ThrowsNaming<OperationCanceledException>(() => SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-b"), progress: stopper, cancellationToken: stop.Token),
                    "the run cancelled after step 2");
                Check(seen.Count(r => r.Stage == "step") == 2, "two steps ran before the run stopped");
            }

            var state = SpeechTrainer.ReadCheckpoint(Path.Combine(root, "run-b", "checkpoint-2"));
            Check(state.Step == 2 && state.Losses.SequenceEqual(whole.Losses.Take(2)) && state.Train.Paths.SequenceEqual(train.Paths) && state.Options.Steps == 6, "train-state.json");
            var resumed = SpeechTrainer.Resume(Path.Combine(root, "run-b", "checkpoint-2"), device: device);
            Check(resumed.Losses.SequenceEqual(whole.Losses), $"resumed losses {string.Join(", ", resumed.Losses)} are not the whole run's {string.Join(", ", whole.Losses)}");
            Check(File.ReadAllBytes(Path.Combine(root, "run-b", "checkpoint-6", "model.safetensors")).AsSpan()
                .SequenceEqual(File.ReadAllBytes(Path.Combine(root, "run-a", "checkpoint-6", "model.safetensors"))), "the resumed run's final weights are the whole run's, bit for bit");
            Check(SpeechTrainer.LatestCheckpoint(Path.Combine(root, "run-b"))!.EndsWith("checkpoint-6", StringComparison.Ordinal), "the latest checkpoint");
            using (var trained = SpeechModels.Load(Path.Combine(root, "run-a", "checkpoint-6"), new SpeechModelOptions { Device = device }))
            {
                Check(trained is ICtcSpeechModel && trained.Transcribe(AudioCodecs.Decode(Path.Combine(root, "data", "u001.wav"))).Segments.Count == 1, "a checkpoint is a model that transcribes");
            }

            // The feature cache: each utterance's features taken once, in the first pass only.
            var hook = new FeatureStages();
            using (Telemetry.Subscribe(hook))
            using (var model = CreateTone(Path.Combine(root, "start-c"), tokenizer, device))
            {
                var cached = SpeechTrainer.Train(model, train, null, ToneTraining(Path.Combine(root, "cache-c")) with { Steps = 16 }, Path.Combine(root, "run-c"));
                Check(cached.FeaturesExtracted == 12 && cached.FeaturesCached >= 12 && hook.Count == 12,
                    $"features taken {cached.FeaturesExtracted} times (telemetry {hook.Count}), {cached.FeaturesCached} from the cache, over three passes of 12");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private sealed class Progress(List<SpeechTrainingProgress> reports, Action<SpeechTrainingProgress>? then = null) : IProgress<SpeechTrainingProgress>
    {
        public void Report(SpeechTrainingProgress value)
        {
            reports.Add(value);
            then?.Invoke(value);
        }
    }

    // A model whose device takes at most `most` utterances at once: more is a ResourceLimitExceededException, as a full device.
    private sealed class TightCtcModel(IdrakAudioCtcModel inner, int most) : ICtcSpeechModel
    {
        public int Refused { get; private set; }

        public string Architecture => inner.Architecture;

        public Module Network => inner.Network;

        public IAudioFeatureExtractor Features => inner.Features;

        public int Classes => inner.Classes;

        public int Blank => inner.Blank;

        public Device Device => inner.Device;

        public IReadOnlySet<string> Languages => inner.Languages;

        public int StepsFor(int frames) => inner.StepsFor(frames);

        public int[] Labels(string text) => inner.Labels(text);

        public string Text(ReadOnlySpan<int> labels) => inner.Text(labels);

        public Tensor Logits(Tensor features, IReadOnlyList<int> frames, SpeechForwardOptions? options = null)
        {
            if (features.Shape[0] > most)
            {
                Refused++;
                throw new ResourceLimitExceededException($"device memory: {features.Shape[0]} utterances at once");
            }

            return inner.Logits(features, frames, options);
        }

        public void Save(string folder) => inner.Save(folder);

        public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default) =>
            inner.Transcribe(audio, options, progress, cancellationToken);

        public IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default) =>
            inner.TranscribeStream(chunks, options, cancellationToken);

        public void Dispose() => inner.Dispose();
    }

    private static void AudioTrainAutomaticBatch(Device device)
    {
        string root = TempFolder("ctc-auto-batch");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 12, 37);
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = ToneTraining(Path.Combine(root, "cache")) with { Steps = 3, BatchSeconds = 0, SaveEvery = 3 };
            SpeechTrainingResult result;
            using (var model = CreateTone(Path.Combine(root, "a"), tokenizer, device))
            {
                var before = model.Network.Parameters().Select(p => p.ToArray()).ToList();
                result = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"));
                Check(model.Network.Parameters().Select(p => p.ToArray()).Zip(before).Any(pair => !pair.First.SequenceEqual(pair.Second)), "the run trained (the probe alone changes no weight)");
            }

            string note = result.Notes.FirstOrDefault(n => n.StartsWith("batch: ", StringComparison.Ordinal)) ?? "";
            Check(note.Contains("automatic", StringComparison.Ordinal) || note.Contains("reports no free memory", StringComparison.Ordinal), $"the note names the automatic batch: '{note}' ({string.Join("; ", result.Notes)})");
            Check(result.MicroBatchSeconds > 0, $"the run has a batch of {result.MicroBatchSeconds} s");
            var state = SpeechTrainer.ReadCheckpoint(Path.Combine(root, "run-a", "checkpoint-3"));
            Check(state.Options.BatchSeconds == result.MicroBatchSeconds, $"the checkpoint stores the measured batch ({state.Options.BatchSeconds} s against {result.MicroBatchSeconds})");
            Check(state.Options.BatchSeconds >= 0.9 && state.Options.BatchSeconds <= 12 * 1.1,
                $"the batch is at least the longest utterance and at most one pass ({state.Options.BatchSeconds} s against 0.9-{12 * 1.1:0.#} s on {device}; "
                + $"{device.Backend.AvailableMemory()?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "no"} bytes available now; the measure: '{note}')");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioTrainMemory(Device device)
    {
        string root = TempFolder("ctc-memory");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 12, 31);
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = ToneTraining(Path.Combine(root, "cache")) with { Steps = 4, BatchSeconds = 6 };
            SpeechTrainingResult free, tight;
            using (var model = CreateTone(Path.Combine(root, "a"), tokenizer, device))
            {
                free = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"));
            }

            using (var model = new TightCtcModel(CreateTone(Path.Combine(root, "b"), tokenizer, device), 2))
            {
                tight = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-b"));
                Check(model.Refused > 0, "the tight model refused a batch");
            }

            Check(free.MicroBatchSeconds == 6 && tight.MicroBatchSeconds < 6 && tight.Notes.Any(n => n.Contains("out of device memory", StringComparison.Ordinal)),
                $"micro-batches of {tight.MicroBatchSeconds} s after running out ({string.Join("; ", tight.Notes)})");
            for (int s = 0; s < free.Losses.Count; s++)
            {
                Check(MathF.Abs(free.Losses[s] - tight.Losses[s]) <= 1e-5f * Math.Max(1f, MathF.Abs(free.Losses[s])), $"step {s + 1}: {tight.Losses[s]} in halves against {free.Losses[s]}");
            }

            // Gradient accumulation (two batches of 3 s a step), and the same run on a device that takes one utterance at a time.
            SpeechTrainingResult accumulated, accumulatedTight;
            using (var model = CreateTone(Path.Combine(root, "c"), tokenizer, device))
            {
                accumulated = SpeechTrainer.Train(model, train, null, options with { BatchSeconds = 3, GradientAccumulation = 2 }, Path.Combine(root, "run-c"));
            }

            using (var model = new TightCtcModel(CreateTone(Path.Combine(root, "d"), tokenizer, device), 1))
            {
                accumulatedTight = SpeechTrainer.Train(model, train, null, options with { BatchSeconds = 3, GradientAccumulation = 2 }, Path.Combine(root, "run-d"));
            }

            for (int s = 0; s < accumulated.Losses.Count; s++)
            {
                Check(MathF.Abs(accumulated.Losses[s] - accumulatedTight.Losses[s]) <= 1e-5f * Math.Max(1f, MathF.Abs(accumulated.Losses[s])),
                    $"accumulated step {s + 1}: {accumulatedTight.Losses[s]} one utterance at a time against {accumulated.Losses[s]}");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioTrainSmoke(Device device)
    {
        string root = TempFolder("ctc-smoke");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 24, 41);
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = ToneTraining(Path.Combine(root, "cache")) with { Steps = 200, BatchSeconds = 8, LearningRate = 3e-3f, WarmupFraction = 0.05f, EvaluateEvery = 100, Checkpointing = false };
            var clock = Stopwatch.StartNew();
            SpeechTrainingResult result;
            using (var model = CreateTone(Path.Combine(root, "start"), tokenizer, device))
            {
                result = SpeechTrainer.Train(model, train, train, options, Path.Combine(root, "run"));
            }

            double seconds = clock.Elapsed.TotalSeconds;
            float first = result.Losses[0], last = result.Losses.TakeLast(10).Average();
            Console.WriteLine(FormattableString.Invariant($"    200 steps in {seconds:F1} s ({seconds * 1000 / 200:F1} ms a step); loss {first:F3} at step 1, {last:F3} over the last 10; cer {string.Join(", ", result.Evaluations.Select(e => $"{e.Score:F3} at step {e.Step}"))}"));
            Check(last < first / 2, FormattableString.Invariant($"the loss went from {first:F3} to {last:F3}, not under half"));
            Check(result.Evaluations.Count == 2 && result.Evaluations[1].Score <= result.Evaluations[0].Score && result.Evaluations[1].Score < 0.5,
                $"the CER at steps 100 and 200: {string.Join(", ", result.Evaluations)}");
            using var trained = SpeechModels.Load(result.Checkpoint!, new SpeechModelOptions { Device = device });
            var text = trained.Transcribe(AudioCodecs.Decode(Path.Combine(root, "data", "u000.wav"))).Text;
            Console.WriteLine($"    u000 '{texts[0]}' reads as '{text}'");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
