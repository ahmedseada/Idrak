// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

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
        var speed = AudioAugmentations.Get(AudioAugmentations.Speed);
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

        var noise = AudioAugmentations.Get(AudioAugmentations.Noise);
        var at20 = options with { MinSnrDb = 20, MaxSnrDb = 20 };
        var white = noise.Augment(tone, new Random(3), at20);
        Check(white.Length == tone.Length && Math.Abs(SnrDb(tone.Samples, white.Samples) - 20) < 0.5, $"white noise at 20 dB measures {SnrDb(tone.Samples, white.Samples):F2} dB");
        var fromFile = noise.Augment(tone, new Random(3), at20 with { NoiseFiles = [AudioData("wav/rate8k.wav")] });
        Check(Math.Abs(SnrDb(tone.Samples, fromFile.Samples) - 20) < 0.01, $"file noise at 20 dB measures {SnrDb(tone.Samples, fromFile.Samples):F3} dB");
        Check(noise.Augment(tone, new Random(3), at20).Samples.SequenceEqual(white.Samples), "the same draws give the same noise");

        var mask = AudioAugmentations.Get(AudioAugmentations.SpecMask);
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
}
