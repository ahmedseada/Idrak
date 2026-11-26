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

// Plan 21, phase S6: training. The fixture comes from tools/audio/make_tiny_audio.py --sections ctc (ctc/: torchaudio's
// forced alignment of the step-1 log-probabilities); the speech sources' JSON and order, the augmentations, and the tone
// speech, tokenizer and spec the other audio tests build on.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioTrainGroup =
    [
        ("audio train: CtcAlignment.Viterbi gives torchaudio's forced_align path of each utterance of ctc/reference/logp-step1.npy exactly, its log-probability within 1e-4, each label's start and length from the path; MinimumSteps counts a step between equal labels; too few steps, a blank label and a label out of range refused", AudioTrainViterbi),
        ("audio train: a speech source's JSON round trip; SpeechSources.Order a seeded permutation per pass", AudioTrainSpeechSources),
        ("audio train: AudioAugmentations: \"speed\" makes the sound 1/factor as long (a factor of 1 the same sound), \"noise\" adds white noise at the drawn signal-to-noise ratio, \"spec-mask\" fills at most its masks' widths with the mean feature and leaves the rest; RandomFor gives the same draws for the same seed, pass and utterance and others for another pass; the options' JSON round trip and refusals", AudioTrainAugmentations),
    ];

    internal static string CtcData(string relative) => AudioData(Path.Combine("ctc", relative));

    internal static JsonObject CtcReference => JsonNode.Parse(File.ReadAllText(CtcData("reference/loss.json")))!.AsObject();

    internal static void AudioTrainViterbi(Device device)
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

    internal static void AudioTrainSpeechSources(Device device)
    {
        string root = TempFolder("speech-sources");
        try
        {
            string jsonl = Path.Combine(root, "train.jsonl"), zip = Path.Combine(root, "data.zip"), clips = Path.Combine(root, "clips");
            var spec = new SpeechSource(SpeechSources.Transcripts, [jsonl, zip]) { Audio = clips, MinSeconds = 0.5, MaxSeconds = 20, SkipMissing = true };
            var back = SpeechSource.FromJson(JsonNode.Parse(spec.ToJson().ToJsonString())!.AsObject());
            Check(back.Paths.SequenceEqual(spec.Paths) && back with { Paths = spec.Paths } == spec, "the source's JSON round trip");

            var order = SpeechSources.Order(50, 7, 2);
            Check(order.Order().SequenceEqual(Enumerable.Range(0, 50)) && order.SequenceEqual(SpeechSources.Order(50, 7, 2)) && !order.SequenceEqual(SpeechSources.Order(50, 7, 3))
                && !order.SequenceEqual(SpeechSources.Order(50, 8, 2)) && !order.SequenceEqual(Enumerable.Range(0, 50)), "Order: a seeded permutation per pass");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    internal static void AudioTrainAugmentations(Device device)
    {
        var tone = new AudioData(ReadNpyFloat32(AudioData("wav/pcm16.npy")), 1, 16000);
        var options = new AudioAugmentationOptions();
        ISoundAugmentation speed = new SpeedAugmentation();
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

        ISoundAugmentation noise = new NoiseAugmentation();
        var at20 = options with { MinSnrDb = 20, MaxSnrDb = 20 };
        var white = noise.Augment(tone, new Random(3), at20);
        Check(white.Length == tone.Length && Math.Abs(SnrDb(tone.Samples, white.Samples) - 20) < 0.5, $"white noise at 20 dB measures {SnrDb(tone.Samples, white.Samples):F2} dB");
        Check(noise.Augment(tone, new Random(3), at20).Samples.SequenceEqual(white.Samples), "the same draws give the same noise");

        IFeatureAugmentation mask = new SpecMaskAugmentation();
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
    }

    // ---------------------------------------------------------------- tensors against torch's

    internal static Dictionary<string, (int[] Shape, float[] Values)> ReadSafeTensors(string path)
    {
        using var reader = SafeTensorsReader.Open(path);
        ITensorStore store = reader;
        return store.Names.ToDictionary(n => n, n => (store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal);
    }

    // Each tensor's values within `relative` of that tensor's largest expected value.
    internal static void TensorsClose(Dictionary<string, (int[] Shape, float[] Values)> expected, IEnumerable<(string Name, float[] Values)> actual, float relative, string what)
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

    // Each tensor's gradient within `relative` of that tensor's largest expected value, or of a thousandth of the largest
    // gradient of any tensor when its own is smaller: an attention's key bias adds the same to every score of a row, which
    // the softmax takes away, so its gradient is zero in exact arithmetic and rounding noise in both (about 1e-9).
    internal static void GradientsClose(Dictionary<string, (int[] Shape, float[] Values)> expected, IEnumerable<(string Name, float[] Values)> actual, float relative, string what)
    {
        float overall = expected.Values.Max(e => e.Values.Max(MathF.Abs));
        int seen = 0;
        foreach (var (name, values) in actual)
        {
            Check(expected.TryGetValue(name, out var reference) && reference.Values.Length == values.Length, $"{what}: no reference of {values.Length} values for {name}");
            float peak = Math.Max(reference.Values.Max(MathF.Abs), overall * 1e-3f), worst = 0;
            int at = 0;
            for (int i = 0; i < values.Length; i++)
            {
                float d = MathF.Abs(reference.Values[i] - values[i]);
                if (!(d <= worst))
                {
                    (worst, at) = (d, i);
                }
            }

            Check(worst <= relative * peak, $"{what} {name}: differs by {worst:E3} at {at} ({reference.Values[at]} against {values[at]}), over {relative:E1} of {peak:E3}");
            seen++;
        }

        Check(seen == expected.Count, $"{what}: {seen} tensors for the reference's {expected.Count}");
    }

    // ---------------------------------------------------------------- tone speech: letters as tone bursts

    internal const string ToneLetters = "abcdefgh";

    // `count` utterances of 2 to 6 letters, each letter a 0.12 s tone of its own pitch (a Hann envelope) and 0.04 s of
    // silence, 0.1 s of silence around them, a little seeded noise: transcripts.jsonl and the WAV files in `folder`.
    internal static (string Transcripts, string[] Texts) ToneSpeech(string folder, int count, int seed, int minLetters = 2, int maxLetters = 6)
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

            File.WriteAllBytes(Path.Combine(folder, $"u{u:D3}.wav"), new WavEncoder().Encode(new AudioData(samples, 1, Rate)));
            rows.Append($"{{\"audio\": \"u{u:D3}.wav\", \"text\": \"{texts[u]}\"}}\n");
        }

        string transcripts = Path.Combine(folder, "transcripts.jsonl");
        File.WriteAllText(transcripts, rows.ToString());
        return (transcripts, texts);
    }

    // A character tokenizer for the tone letters: Metaspace BPE trained to its alphabet alone (no merges): the letters, "▁"
    // and the line end (the trainer reads each line with its end, as Hugging Face's train_from_files does).
    internal static IPretrainedTokenizer ToneTokenizer(string folder, IEnumerable<string> texts)
    {
        string file = Path.Combine(folder, "texts.txt");
        File.WriteAllLines(file, texts);
        int alphabet = string.Concat(texts).Distinct().Count() + 2;
        return new BpeTrainer().Train([file],
            new TokenizerTrainingOptions { VocabularySize = alphabet, MinFrequency = 1, PreTokenizer = TokenizerTrainers.Metaspace, Workers = 1 });
    }

    internal static JsonObject ToneSpec(bool convolution = false) => new()
    {
        ["architecture"] = IdrakAudioFamily.Architecture, ["task"] = "ctc", ["mels"] = 80, ["dim"] = 32, ["layers"] = 2, ["heads"] = 4, ["ffn_dim"] = 64,
        ["subsample_layers"] = 2, ["subsample_kernel"] = 3, ["convolution_module"] = convolution, ["convolution_kernel"] = 5,
    };

    // ---------------------------------------------------------------- the trainer

    internal sealed class Progress(List<SpeechTrainingProgress> reports, Action<SpeechTrainingProgress>? then = null) : IProgress<SpeechTrainingProgress>
    {
        public void Report(SpeechTrainingProgress value)
        {
            reports.Add(value);
            then?.Invoke(value);
        }
    }
}
