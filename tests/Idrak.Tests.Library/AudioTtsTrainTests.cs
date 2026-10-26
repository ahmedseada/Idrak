// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.IdrakAudio;
using Idrak.Models;

// Plan 21, phase S6: the IdrakAudio plug-in's "tts" task, the "durations" synthesis objective, length regulation, the
// alignments file and SpeechSynthesisTrainer. The fixture comes from tools/audio/make_tiny_audio.py --sections tts (tts/:
// the IdrakAudio synthesizer in torch, padded labels with their durations and target log-mels, the step-1 loss and its
// gradients, three SGD steps).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioTtsTrainGroup =
    [
        ("audio tts train: the IdrakAudio plug-in reads tts/model through SpeechSynthesisModels.Load (refused naming SpeechSynthesisFamilies.Register without it, and by SpeechModels.Load naming SpeechSynthesisModels) and gives torch's step-1 log-mel within 1e-4, its log durations, its log-mel L1 and log-duration MSE and their sum within 1e-4, and every tensor's gradient within 1e-4 of that tensor's largest (of a thousandth of the largest of any, for the key biases' zero gradients), through DurationSynthesisObjective; each utterance's log-mel and log durations in the padded batch equal its own alone within 1e-5; three SGD steps' losses and the weights after them within 1e-4", AudioTtsTrainFixture),
        ("audio tts train: LengthRegulation repeats each label's row for its frames exactly (zero durations skipped, the padding after a text's frames its first row), refuses a negative duration; Durations rounds exp(x) - offset over the rate (2 halves them), never below 0, one frame when all round to none; SpeechAlignment.FrameDurations sums to the frames, gives each label the steps to the next label's start (the leading ones to the first) scaled to the frames", AudioTtsTrainRegulation),
        ("audio tts train: SpeechSynthesisModels.Create makes an IdrakAudio tts model from a spec and a trained tokenizer into a folder SpeechSynthesisModels.Load reads back with the same weights (the same seed the same weights; classes that are not the tokenizer's, a used folder, a recognizer task refused; SpeechModels.Create refuses the tts task naming SpeechSynthesisModels.Create); it speaks through griffin-lim sound of the regulated length (Stft.SamplesFor of the durations' frames, also for one or two frames) at the features' rate, the same twice, SpeakStream joined equal to Speak with the same split, resampled when asked, a character the tokenizer cannot read left out (or refused when asked); an app's vocoder used; a vocoder of another mel layout, an unregistered vocoder and a voice refused", AudioTtsTrainCreate),
        ("audio tts train: SpeechAlignments.Align of a tone ctc model gives each utterance's CtcAlignment.Viterbi path of its own log-probabilities exactly (one utterance a batch), the same labels and log-probabilities within 1e-3 in batches of several; AlignToFile writes JSON lines Read gives back exactly; unknown keys skipped, a bad row refused naming its line, a missing file refused; the index finds an utterance by its path, or its file name and text", AudioTtsTrainAlignments),
        ("audio tts train: SpeechSynthesisTrainer trains an IdrakAudio tts model with the \"durations\" objective: a run stopped after step 2 and resumed from checkpoint-2 gives the whole run's losses and final weights bit for bit, its state naming the objective and the alignments file; an utterance without an alignment counted and left out, a run without Durations and a feature augmentation refused", AudioTtsTrainResume),
        ("audio tts train: a 200-step CPU run of a tiny IdrakAudio tts model on tone speech with the tones' durations halves its loss (the mean of the last 10 steps under half of step 1) and its log-mel L1, the trained checkpoint predicts a letter's frames within a third of the truth and speaks finite sound of the regulated length (step time printed)", AudioTtsTrainSmoke),
    ];

    internal static string TtsData(string relative) => AudioData(Path.Combine("tts", relative));

    internal static JsonObject TtsReference => JsonNode.Parse(File.ReadAllText(TtsData("reference/loss.json")))!.AsObject();

    internal static IEnumerable<(string Name, float[] Values)> Gradients(IdrakAudioTtsModel model) =>
        model.NamedParameters().Select(p => (p.Name, IdrakAudioWeights.Stored(p.Parameter.Grad ?? throw new Exception($"{p.Name} has no gradient"), p.Transposed)));

    // The fixture's labels as the objective builds them: [n, longest] ids as floats, 0 after each text's own.
    internal static float[] TtsIds(int[][] labels, int length)
    {
        var ids = new float[labels.Length * length];
        for (int b = 0; b < labels.Length; b++)
        {
            for (int i = 0; i < labels[b].Length; i++)
            {
                ids[b * length + i] = labels[b][i];
            }
        }

        return ids;
    }

    internal static void AudioTtsTrainFixture(Device device)
    {
        IdrakAudioPlugin.Unregister();
        ThrowsNaming<NotSupportedException>(() => SpeechSynthesisModels.Load(TtsData("model")).Dispose(), "without the plug-in", "'IdrakAudio' is not registered", "SpeechSynthesisFamilies.Register");
        IdrakAudioPlugin.Register();
        try
        {
            ThrowsNaming<NotSupportedException>(() => SpeechModels.Load(TtsData("model")).Dispose(), "a tts model through the recognizers' loader", "SpeechSynthesisModels.Load");
            var reference = TtsReference;
            var results = reference["results"]!;
            int[][] labels = [.. reference["labels"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
            int[][] durations = [.. reference["durations"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
            int[] frames = [.. reference["frames"]!.AsArray().Select(v => (int)v!)];
            int mels = (int)reference["mels"]!, n = labels.Length, longest = frames.Max(), length = labels.Max(l => l.Length);
            var targets = ReadNpyFloat32(TtsData("log-mels.npy"));
            using (var model = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(TtsData("model"), new SpeechSynthesisLoadOptions { Device = device }))
            {
                Check(model is ITrainableSpeechSynthesizer { Objective: SpeechSynthesisObjectives.Durations } && model.DurationOffset == (float)reference["duration_offset"]!
                    && model.Config.Mels == mels, "the durations objective, the offset and the mels");
                model.Network.Train();
                using (var scope = new TensorScope())
                {
                    var features = Tensor.From(targets, [n, longest, mels], device);
                    var (melLoss, durationLoss) = DurationSynthesisObjective.Parts(model, features, frames, labels, durations);
                    var loss = melLoss + durationLoss * (float)reference["duration_weight"]!;
                    foreach (var (actual, key) in new[] { (melLoss.Item(), "mel_l1_step1"), (durationLoss.Item(), "duration_mse_step1"), (loss.Item(), "loss_step1") })
                    {
                        float expected = (float)results[key]!;
                        Check(MathF.Abs(actual - expected) <= 1e-4f * MathF.Abs(expected), $"{key}: {actual}, torch {expected}");
                    }

                    loss.Backward();
                    GradientsClose(ReadSafeTensors(TtsData("reference/grads-step1.safetensors")), Gradients(model), 1e-4f, "tts step-1 gradient");
                }

                model.Network.Eval();
                using (var scope = new TensorScope())
                using (Autograd.NoGrad())
                {
                    int[] lengths = [.. labels.Select(l => l.Length)];
                    var hidden = model.Encode(Tensor.From(TtsIds(labels, length), [n, length], device), lengths);
                    var logDurations = model.LogDurations(hidden, lengths).ToArray();
                    var mel = model.Decode(hidden, lengths, durations).ToArray();
                    var torchMel = ReadNpyFloat32(TtsData("reference/log-mel-step1.npy"));
                    var torchDurations = results["log_durations_step1"]!.AsArray();
                    for (int b = 0; b < n; b++)
                    {
                        CloseRelative(torchMel.AsSpan(b * longest * mels, frames[b] * mels).ToArray(), mel.AsSpan(b * longest * mels, frames[b] * mels).ToArray(), 1e-4f, $"utterance {b}'s log-mel against torch's");
                        float[] expected = [.. torchDurations[b]!.AsArray().Take(labels[b].Length).Select(v => (float)v!)];
                        CloseRelative(expected, logDurations.AsSpan(b * length, labels[b].Length).ToArray(), 1e-4f, $"utterance {b}'s log durations against torch's");
                        var own = model.Encode(Tensor.From(TtsIds([labels[b]], labels[b].Length), [1, labels[b].Length], device), [labels[b].Length]);
                        var alone = model.Decode(own, [labels[b].Length], [durations[b]]).ToArray();
                        CloseRelative(alone, mel.AsSpan(b * longest * mels, frames[b] * mels).ToArray(), 1e-5f, $"utterance {b}'s log-mel alone against in the batch");
                        CloseRelative(model.LogDurations(own, [labels[b].Length]).ToArray(), logDurations.AsSpan(b * length, labels[b].Length).ToArray(), 1e-5f,
                            $"utterance {b}'s log durations alone against in the batch");
                    }
                }
            }

            // Three SGD steps as torch took them, and the weights after them.
            using (var model = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(TtsData("model"), new SpeechSynthesisLoadOptions { Device = device }))
            {
                model.Network.Train();
                using var sgd = new Sgd(model.Network.Parameters(), (float)reference["sgd_lr"]!, 0f, 0f);
                float[] expected = [.. results["sgd_losses"]!.AsArray().Select(v => (float)v!)];
                var losses = new List<float>();
                for (int s = 0; s < 4; s++)
                {
                    using var scope = new TensorScope();
                    if (s > 0)
                    {
                        sgd.Step();
                    }

                    sgd.ZeroGrad();
                    var loss = DurationSynthesisObjective.Loss(model, Tensor.From(targets, [n, longest, mels], device), frames, labels, durations);
                    losses.Add(loss.Item());
                    loss.Backward();
                }

                for (int s = 0; s < 4; s++)
                {
                    Check(MathF.Abs(losses[s] - expected[s]) <= 1e-4f * MathF.Abs(expected[s]), $"SGD loss {s}: {losses[s]}, torch {expected[s]} ({string.Join(", ", losses)})");
                }

                TensorsClose(ReadSafeTensors(TtsData("reference/weights-step3.safetensors")), model.Export().Select(e => (e.Name, e.Values)), 1e-4f, "the weights after 3 SGD steps");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
        }
    }

    internal static void AudioTtsTrainRegulation(Device device)
    {
        // A gather: each frame is its label's row exactly.
        int[][] durations = [[2, 0, 3], [1, 4]];
        const int Length = 3, Width = 5;
        var values = new float[2 * Length * Width];
        var random = new Random(5);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var (rows, lengths) = LengthRegulation.Rows(durations, Length);
        Check(lengths.SequenceEqual([5, 5]) && rows.SequenceEqual([0, 0, 2, 2, 2, 3, 4, 4, 4, 4]), $"rows {string.Join(" ", rows)} for frames {string.Join(" ", lengths)}");
        var (padRows, padLengths) = LengthRegulation.Rows([[1, 1], [2, 0, 1]], Length);
        Check(padLengths.SequenceEqual([2, 3]) && padRows.SequenceEqual([0, 1, 0, 3, 3, 5]), $"the shorter text's padding reads its first row: {string.Join(" ", padRows)}");
        using (var scope = new TensorScope())
        {
            var (expanded, frames) = LengthRegulation.Expand(Tensor.From(values, [2, Length, Width], device), durations);
            Check(expanded.Shape.SequenceEqual([2, 5, Width]) && frames.SequenceEqual([5, 5]), $"expanded to {Tensor.FormatShape(expanded.Shape)}");
            var got = expanded.ToArray();
            for (int f = 0; f < rows.Length; f++)
            {
                Check(got.AsSpan(f * Width, Width).SequenceEqual(values.AsSpan(rows[f] * Width, Width)), $"frame {f} is row {rows[f]} exactly");
            }
        }

        ThrowsNaming<ArgumentException>(() => LengthRegulation.Rows([[1, -1]], 2), "a negative duration", "negative duration");
        ThrowsNaming<ArgumentException>(() => LengthRegulation.Rows([[1, 1, 1]], 2), "more durations than positions", "3 durations for 2 positions");

        float[] logs = [MathF.Log(5 + 1), MathF.Log(0.4f + 1), MathF.Log(10.6f + 1), MathF.Log(0.2f)];
        Check(LengthRegulation.Durations(logs, 1f).SequenceEqual([5, 0, 11, 0]), $"durations {string.Join(" ", LengthRegulation.Durations(logs, 1f))}");
        Check(LengthRegulation.Durations(logs, 1f, rate: 2f).SequenceEqual([3, 0, 5, 0]), $"at rate 2: {string.Join(" ", LengthRegulation.Durations(logs, 1f, 2f))}");
        Check(LengthRegulation.Durations([-3f, -1f, -2f], 1f).SequenceEqual([0, 1, 0]), "all none: the label predicted longest gets a frame");
        ThrowsNaming<ArgumentOutOfRangeException>(() => LengthRegulation.Durations(logs, 1f, 0f), "rate 0", "speaking rate is positive");

        // Three labels at steps 2, 5 and 9 of 12: the first from step 0, each to the next one's start, the last to the end.
        var alignment = new SpeechAlignment("a.wav", "abc", [1, 2, 3], [2, 5, 9], [2, 3, 1], 12, 0.04, -3f);
        Check(alignment.FrameDurations(12).SequenceEqual([5, 4, 3]), $"frames at the steps' scale: {string.Join(" ", alignment.FrameDurations(12))}");
        Check(alignment.FrameDurations(48).SequenceEqual([20, 16, 12]), $"four frames a step: {string.Join(" ", alignment.FrameDurations(48))}");
        var odd = alignment.FrameDurations(31);
        Check(odd.Sum() == 31 && odd.All(d => d >= 0), $"31 frames: {string.Join(" ", odd)}");
        Check(alignment.FrameDurations(0).All(d => d == 0), "no frames: none to any label");
    }

    internal static JsonObject ToneTtsSpec() => new()
    {
        ["architecture"] = IdrakAudioFamily.Architecture, ["task"] = "tts", ["mels"] = 80, ["dim"] = 32, ["layers"] = 2, ["heads"] = 2, ["ffn_dim"] = 64,
        ["decoder_layers"] = 2, ["decoder_dim"] = 32, ["decoder_heads"] = 2, ["decoder_ffn_dim"] = 64, ["duration_filters"] = 32,
    };

    internal static IdrakAudioTtsModel CreateToneTts(string folder, IPretrainedTokenizer tokenizer, Device device, int seed = 7) =>
        (IdrakAudioTtsModel)SpeechSynthesisModels.Create(folder, ToneTtsSpec(), tokenizer, new SpeechCreateOptions { Device = device, Seed = seed });

    // A vocoder of another mel layout, to be refused.
    internal sealed class OtherMelsVocoder : IVocoder
    {
        public string Name => "tts-test-other-mels";

        public int SampleRate => 16000;

        public int Mels => 7;

        public AudioData Synthesize(Tensor logMel, Device device) => new(new float[160], 1, 16000);
    }

    internal static void AudioTtsTrainCreate(Device device)
    {
        string root = TempFolder("tts-create");
        IdrakAudioPlugin.Register();
        Vocoders.Register(TestKinds.Vocoder("tts-test-other-mels", _ => new OtherMelsVocoder()));
        try
        {
            var (_, texts) = ToneSpeech(Path.Combine(root, "data"), 6, 11);
            var tokenizer = ToneTokenizer(root, texts);
            string folder = Path.Combine(root, "model");
            using var model = CreateToneTts(folder, tokenizer, device);
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            Check((string?)config["task"] == "tts" && (int)config["classes"]! == tokenizer.VocabularySize && (int)config["decoder_layers"]! == 2 && (int)config["duration_filters"]! == 32
                && (int)config["duration_kernel"]! == 3 && (float)config["duration_offset"]! == 1f && !config.ContainsKey("blank_id") && !config.ContainsKey("subsample_layers")
                && File.Exists(Path.Combine(folder, "tokenizer.json")), $"config.json holds the synthesizer's settings: {config.ToJsonString()}");
            Check(model.Labels(texts[0]).Length == texts[0].Length + 1 && model.SampleRate == 16000 && model.Voices.Count == 0, $"labels of '{texts[0]}', the rate, no voices");
            using (var again = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = device }))
            {
                Check(again.Export().Zip(model.Export()).All(p => p.First.Name == p.Second.Name && p.First.Values.AsSpan().SequenceEqual(p.Second.Values)), "the folder reads back to the same weights");
                Check(again.Config.ToJson().ToJsonString() == model.Config.ToJson().ToJsonString(), "the config reads back the same");
            }

            using (var same = CreateToneTts(Path.Combine(root, "same"), tokenizer, device))
            {
                Check(same.Export().Zip(model.Export()).All(p => p.First.Values.AsSpan().SequenceEqual(p.Second.Values)), "the same seed draws the same weights");
            }

            var wrong = ToneTtsSpec();
            wrong["classes"] = tokenizer.VocabularySize + 1;
            ThrowsNaming<ArgumentException>(() => SpeechSynthesisModels.Create(Path.Combine(root, "wrong"), wrong, tokenizer).Dispose(), "classes that are not the tokenizer's", "the tokenizer's");
            ThrowsNaming<IOException>(() => CreateToneTts(folder, tokenizer, device).Dispose(), "a used folder", "already holds files");
            ThrowsNaming<NotSupportedException>(() => SpeechSynthesisModels.Create(Path.Combine(root, "ctc"), ToneSpec(), tokenizer).Dispose(), "a recognizer task", "is a speech recognizer", "SpeechModels.Load");
            ThrowsNaming<NotSupportedException>(() => SpeechModels.Create(Path.Combine(root, "tts"), ToneTtsSpec(), tokenizer).Dispose(), "the tts task through SpeechModels", "SpeechSynthesisModels.Create");

            // Speaking: the spectrogram's frames through griffin-lim.
            string text = texts[1];
            var labels = model.Labels(text);
            var (logMel, frames, durations) = model.Spectrogram(labels);
            Check(frames == durations.Sum() && logMel.Length == frames * 80 && logMel.All(float.IsFinite), $"{frames} frames for durations {string.Join(" ", durations)}");
            var (audio, stats) = model.Speak(text);
            Check(audio.SampleRate == 16000 && audio.Channels == 1 && audio.Length == Stft.SamplesFor(frames, model.Features.Options) && audio.Samples.All(float.IsFinite),
                $"{audio.Length} samples for {frames} frames ({Stft.SamplesFor(frames, model.Features.Options)})");
            Check(stats.Model > TimeSpan.Zero && stats.Vocoder > TimeSpan.Zero && stats.RealTimeFactor > 0, $"the stats: {stats}");
            Check(model.Speak(text).Audio.Samples.AsSpan().SequenceEqual(audio.Samples), "the same text twice gives the same samples");
            var griffinLim = new GriffinLimVocoder(model.Features.Options);
            foreach (int few in (int[])[1, 2, 3])
            {
                using var scope = new TensorScope();
                var short_ = griffinLim.Synthesize(Tensor.From(new float[few * 80], [few, 80], device), device);
                Check(short_.Length == Stft.SamplesFor(few, model.Features.Options) && short_.Samples.All(float.IsFinite), $"griffin-lim on {few} frames: {short_.Length} samples");
            }

            var given = model.Spectrogram(labels, durations: [.. Enumerable.Repeat(3, labels.Length)]);
            Check(given.Frames == 3 * labels.Length && given.Durations.All(d => d == 3), "given durations are used as they are");
            var resampled = model.Speak(text, new SpeechSynthesisOptions { SampleRate = 8000 }).Audio;
            Check(resampled.SampleRate == 8000 && Math.Abs(resampled.Length - audio.Length / 2) <= 2, $"resampled to 8 kHz: {resampled.Length} samples");

            string two = texts[2] + ". " + texts[3] + ".";
            var split = new SpeechSynthesisOptions { MaxCharacters = 8 };
            var whole = model.Speak(two, split).Audio.Samples;
            var joined = new List<float>();
            var stream = model.SpeakStream(two, split).GetAsyncEnumerator();
            int pieces = 0;
            while (stream.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                joined.AddRange(stream.Current.Samples);
                pieces++;
            }

            Check(pieces == 2 && joined.ToArray().AsSpan().SequenceEqual(whole), $"SpeakStream gives {pieces} pieces joined equal to Speak");
            Check(model.Speak(text + "!").Audio.Samples.AsSpan().SequenceEqual(audio.Samples), "a character the tokenizer has no token for is left out");
            model.DropUnknownCharacters = false;
            ThrowsNaming<InvalidDataException>(() => model.Speak(text + "!"), "an unknown character kept", "'!' is not in the vocabulary");
            model.DropUnknownCharacters = true;

            model.Vocoder = new GriffinLimVocoder(model.Features.Options, new GriffinLimOptions { Iterations = 4, Seed = 9 });
            var own = model.Speak(text).Audio;
            Check(own.Length == audio.Length && !own.Samples.AsSpan().SequenceEqual(audio.Samples), "the app's vocoder (4 iterations) is used when the call names none");
            Check(model.Speak(text, new SpeechSynthesisOptions { Vocoder = "griffin-lim" }).Audio.Samples.AsSpan().SequenceEqual(audio.Samples), "a named vocoder over the app's");
            model.Vocoder = null;
            ThrowsNaming<NotSupportedException>(() => model.Speak(text, new SpeechSynthesisOptions { Vocoder = "tts-test-other-mels" }), "a vocoder of 7 mels", "reads 7 mel bands", "makes 80");
            ThrowsNaming<NotSupportedException>(() => model.Speak(text, new SpeechSynthesisOptions { Vocoder = "no-such-vocoder" }), "an unregistered vocoder", "No vocoder 'no-such-vocoder'");
            ThrowsNaming<ArgumentException>(() => model.Speak(text, new SpeechSynthesisOptions { Voice = "0" }), "a voice", "one voice");
        }
        finally
        {
            Vocoders.Unregister("tts-test-other-mels");
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    internal static bool SameAlignment(SpeechAlignment a, SpeechAlignment b) =>
        a.Audio == b.Audio && a.Text == b.Text && a.Labels.SequenceEqual(b.Labels) && a.Starts.SequenceEqual(b.Starts) && a.Durations.SequenceEqual(b.Durations)
        && a.Steps == b.Steps && a.StepSeconds == b.StepSeconds && a.LogProbability == b.LogProbability;

    internal static void AudioTtsTrainAlignments(Device device)
    {
        string root = TempFolder("tts-align");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 8, 13);
            var tokenizer = ToneTokenizer(root, texts);
            using var model = CreateTone(Path.Combine(root, "ctc"), tokenizer, device);
            var source = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var reports = new List<SpeechAlignmentProgress>();
            string file = Path.Combine(root, "alignments.jsonl");
            var single = SpeechAlignments.AlignToFile(model, source, file, new SpeechAlignmentOptions { BatchSeconds = 1e-3 }, new ListProgress<SpeechAlignmentProgress>(reports));
            Check(single.Alignments.Count == 8 && reports.Count == 8 && reports[^1].Done == 8 && reports[^1].Total == 8, $"{single.Alignments.Count} alignments, {reports.Count} reports");
            using var reader = SpeechSources.Open(source);
            model.Network.Eval();
            for (int u = 0; u < reader.Count; u++)
            {
                var a = single.Alignments[u];
                Check(a.Audio == reader[u].Audio && a.Text == texts[u] && a.Labels.SequenceEqual(model.Labels(texts[u])), $"utterance {u}: its audio, text and labels, in the source's order");
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var audio = AudioCodecs.Decode(reader.ReadBytes(u));
                float[] values;
                int frames;
                using (var extracted = model.Features.Extract(audio, Device.Cpu))                          // on the host, as Align takes them
                {
                    (values, frames) = (extracted.ToArray(), extracted.Shape[0]);
                }

                int steps = model.StepsFor(frames);
                var logProbs = model.Logits(Tensor.From(values, [1, frames, 80], device), [frames]).LogSoftmax().Reshape(steps, model.Classes);
                var path = CtcAlignment.Viterbi(logProbs, a.Labels, model.Blank);
                Check(a.Steps == steps && a.Starts.SequenceEqual(path.Starts) && a.Durations.SequenceEqual(path.Lengths) && a.LogProbability == path.LogProbability,
                    $"utterance {u}: the alignment is the Viterbi path of its own log-probabilities");
                Check(Math.Abs(a.StepSeconds * steps - frames * 0.01) < 1e-9, $"utterance {u}: {a.StepSeconds} s a step");
            }

            var batched = SpeechAlignments.Align(model, source, new SpeechAlignmentOptions { BatchSeconds = 30 });
            Check(batched.Alignments.Zip(single.Alignments).All(p => p.First.Labels.SequenceEqual(p.Second.Labels)
                && MathF.Abs(p.First.LogProbability - p.Second.LogProbability) <= 1e-3f * MathF.Abs(p.Second.LogProbability)), "batches of several give the same paths' log-probabilities");

            var read = SpeechAlignments.Read(file);
            Check(read.Count == 8 && read.Zip(single.Alignments).All(p => SameAlignment(p.First, p.Second)), "the file reads back exactly");
            string line = File.ReadLines(file).First();
            Check(line.StartsWith("{\"audio\":", StringComparison.Ordinal) && line.Contains("\"durations\":[", StringComparison.Ordinal) && line.Contains("\"step_seconds\":", StringComparison.Ordinal),
                $"a row: {line}");

            string extra = Path.Combine(root, "extra.jsonl");
            File.WriteAllText(extra, line[..^1] + ",\"speaker\":\"x\",\"more\":[1,{\"a\":2}]}\n\n");
            Check(SameAlignment(SpeechAlignments.Read(extra)[0], read[0]), "unknown keys are skipped");
            string bad = Path.Combine(root, "bad.jsonl");
            File.WriteAllText(bad, line + "\n{\"audio\":\"x.wav\",\"text\":\"ab\",\"labels\":[1,2],\"starts\":[0],\"durations\":[1,1],\"steps\":4}\n");
            ThrowsNaming<InvalidDataException>(() => SpeechAlignments.Read(bad), "a row whose starts do not match its labels", "bad.jsonl, line 2", "2 labels with 1 starts");
            ThrowsNaming<FileNotFoundException>(() => SpeechAlignments.Read(Path.Combine(root, "none.jsonl")), "a missing file", "was not found");

            var index = SpeechAlignments.Index(read);
            Check(ReferenceEquals(index(reader[3]), read[3]), "found by its path");
            Check(ReferenceEquals(index(new SpeechUtterance("moved", "/elsewhere/" + Path.GetFileName(reader[3].Audio), texts[3], 1)), read[3]), "found by its file name and text");
            Check(index(new SpeechUtterance("other", "/elsewhere/" + Path.GetFileName(reader[3].Audio), "zzz", 1)) is null, "another text is not found");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    internal sealed class ListProgress<T>(List<T> reports) : IProgress<T>
    {
        public void Report(T value) => reports.Add(value);
    }

    // The tones' alignments (ToneSpeech's layout: 0.1 s of silence, each letter a 0.12 s tone and 0.04 s of silence), in
    // feature frames: the word marker "▁" over the leading silence, each letter over its tone; `skip` left out.
    internal static string ToneAlignments(string transcripts, string path, IdrakAudioTtsModel model, int skip = -1)
    {
        using var reader = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [transcripts]));
        double step = model.Features.Options.HopSize / (double)model.SampleRate;
        var rows = new List<SpeechAlignment>();
        for (int u = 0; u < reader.Count; u++)
        {
            if (u == skip)
            {
                continue;
            }

            var utterance = reader[u];
            var labels = model.Labels(utterance.Text);
            int letters = utterance.Text.Length;
            Check(labels.Length == letters + 1, $"'{utterance.Text}': the word marker and a label a letter");
            var starts = new int[letters + 1];
            var durations = new int[letters + 1];
            durations[0] = (int)Math.Round(0.1 / step);
            for (int i = 0; i < letters; i++)
            {
                starts[i + 1] = (int)Math.Round((0.1 + i * 0.16) / step);
                durations[i + 1] = (int)Math.Round(0.12 / step);
            }

            int frames = model.Features.FramesFor((long)Math.Round(utterance.Seconds * model.SampleRate));
            rows.Add(new SpeechAlignment(utterance.Audio, utterance.Text, labels, starts, durations, frames, step, 0f));
        }

        SpeechAlignments.Write(path, rows);
        return path;
    }

    internal static void AudioTtsTrainResume(Device device)
    {
        string root = TempFolder("tts-resume");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 12, 21);
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            string alignments;
            using (var probe = CreateToneTts(Path.Combine(root, "probe"), tokenizer, device))
            {
                alignments = ToneAlignments(transcripts, Path.Combine(root, "alignments.jsonl"), probe, skip: 5);
            }

            var options = ToneTraining(Path.Combine(root, "cache")) with { SaveEvery = 2, Durations = alignments };
            SpeechTrainingResult whole;
            using (var model = CreateToneTts(Path.Combine(root, "start-a"), tokenizer, device))
            {
                ThrowsNaming<ArgumentException>(() => SpeechSynthesisTrainer.Train(model, train, null, options with { Durations = null }, Path.Combine(root, "none")),
                    "no alignments file", "SpeechTrainingOptions.Durations");
                ThrowsNaming<ArgumentException>(() => SpeechSynthesisTrainer.Train(model, train, null, options with { Augmentations = ["spec-mask"] }, Path.Combine(root, "none")),
                    "a feature augmentation", "'spec-mask'", "sound augmentations only");
                whole = SpeechSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"));
            }

            Check(whole.Steps == 6 && whole.Losses.All(float.IsFinite) && whole.Dropped == 1 && whole.Utterances == 11, $"the run: {whole.Steps} steps, {whole.Dropped} dropped, {whole.Utterances} used");
            Check(whole.Notes.Any(n => n.Contains("1 utterances have no alignment in", StringComparison.Ordinal)), $"the dropped row is noted: {string.Join("; ", whole.Notes)}");
            using (var model = CreateToneTts(Path.Combine(root, "start-b"), tokenizer, device))
            using (var stop = new CancellationTokenSource())
            {
                var stopper = new Progress([], r =>
                {
                    if (r.Stage == "step" && r.Step == 2)
                    {
                        stop.Cancel();
                    }
                });
                ThrowsNaming<OperationCanceledException>(() => SpeechSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run-b"), progress: stopper, cancellationToken: stop.Token),
                    "the run cancelled after step 2");
            }

            var state = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "run-b", "checkpoint-2", SpeechTrainer.StateFile)))!;
            Check(state["objective"]?.GetValue<string>() == SpeechSynthesisObjectives.Durations && (string?)state["options"]!["durations"] == alignments,
                "the state names the durations objective and the alignments file");
            var resumed = SpeechSynthesisTrainer.Resume(Path.Combine(root, "run-b", "checkpoint-2"), device: device);
            Check(resumed.Losses.SequenceEqual(whole.Losses), $"resumed losses {string.Join(", ", resumed.Losses)} are not the whole run's {string.Join(", ", whole.Losses)}");
            Check(File.ReadAllBytes(Path.Combine(root, "run-b", "checkpoint-6", "model.safetensors")).AsSpan()
                .SequenceEqual(File.ReadAllBytes(Path.Combine(root, "run-a", "checkpoint-6", "model.safetensors"))), "the resumed run's final weights are the whole run's, bit for bit");
            using var trained = SpeechSynthesisModels.Load(Path.Combine(root, "run-a", "checkpoint-6"), new SpeechSynthesisLoadOptions { Device = device });
            Check(trained is IdrakAudioTtsModel && trained.Speak(texts[0]).Audio.Samples.All(float.IsFinite), "a checkpoint is a synthesizer that speaks");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    internal static void AudioTtsTrainSmoke(Device device)
    {
        string root = TempFolder("tts-smoke");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 24, 41);
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            string alignments;
            float firstMel;
            using (var probe = CreateToneTts(Path.Combine(root, "start"), tokenizer, device))
            {
                alignments = ToneAlignments(transcripts, Path.Combine(root, "alignments.jsonl"), probe);
                firstMel = MelLoss(probe, transcripts, alignments, device);
            }

            var options = ToneTraining(Path.Combine(root, "cache")) with
            {
                Steps = 200, BatchSeconds = 8, LearningRate = 3e-3f, WarmupFraction = 0.05f, Checkpointing = false, Durations = alignments,
            };
            var clock = Stopwatch.StartNew();
            SpeechTrainingResult result;
            using (var model = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(Path.Combine(root, "start"), new SpeechSynthesisLoadOptions { Device = device }))
            {
                result = SpeechSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run"));
            }

            double seconds = clock.Elapsed.TotalSeconds;
            float first = result.Losses[0], last = result.Losses.TakeLast(10).Average();
            using var trained = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(result.Checkpoint!, new SpeechSynthesisLoadOptions { Device = device });
            float lastMel = MelLoss(trained, transcripts, alignments, device);
            var labels = trained.Labels(texts[0]);
            var (_, frames, durations) = trained.Spectrogram(labels);
            using var reader = SpeechSources.Open(train);
            int ownFrames = trained.Features.FramesFor((long)Math.Round(reader[0].Seconds * trained.SampleRate));
            double letter = durations.Skip(1).Average(), truth = SpeechAlignments.Index(SpeechAlignments.Read(alignments))(reader[0])!.FrameDurations(ownFrames).Skip(1).Average();
            Console.WriteLine(FormattableString.Invariant($"    200 steps in {seconds:F1} s ({seconds * 1000 / 200:F1} ms a step); loss {first:F3} at step 1, {last:F3} over the last 10; log-mel L1 {firstMel:F3} before, {lastMel:F3} after"));
            Console.WriteLine($"    '{texts[0]}' predicted durations {string.Join(" ", durations)} (a letter {letter:F1} frames, the alignment's {truth:F1})");
            Check(last < first / 2, FormattableString.Invariant($"the loss went from {first:F3} to {last:F3}, not under half"));
            Check(lastMel < firstMel / 2, FormattableString.Invariant($"the log-mel L1 went from {firstMel:F3} to {lastMel:F3}, not under half"));
            Check(Math.Abs(letter - truth) <= truth / 3, FormattableString.Invariant($"a letter's predicted frames {letter:F1}, the alignment's {truth:F1}"));
            var audio = trained.Speak(texts[0]).Audio;
            Check(audio.Length == Stft.SamplesFor(frames, trained.Features.Options) && audio.Samples.All(float.IsFinite), $"{audio.Length} finite samples for {frames} frames");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    // The mean log-mel L1 of the source's utterances with their alignments' durations (gradients off, each utterance alone).
    internal static float MelLoss(IdrakAudioTtsModel model, string transcripts, string alignments, Device device)
    {
        using var reader = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [transcripts]));
        var index = SpeechAlignments.Index(SpeechAlignments.Read(alignments));
        bool training = model.Network.IsTraining;
        model.Network.Eval();
        try
        {
            double sum = 0;
            for (int u = 0; u < reader.Count; u++)
            {
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var features = model.Features.Extract(AudioCodecs.Decode(reader.ReadBytes(u)), device);
                int frames = features.Shape[0];
                var labels = model.Labels(reader[u].Text);
                var durations = index(reader[u])!.FrameDurations(frames);
                sum += DurationSynthesisObjective.Parts(model, features.Reshape(1, frames, model.Config.Mels), [frames], [labels], [durations]).Mel.Item();
            }

            return (float)(sum / reader.Count);
        }
        finally
        {
            model.Network.Train(training);
        }
    }
}
