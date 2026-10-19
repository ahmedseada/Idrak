// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.IdrakAudio;
using Idrak.Models;

// Plan 21, phase S6: the "seq2seq" objective and the IdrakAudio plug-in's "seq2seq" task. The fixture comes from
// tools/audio/make_tiny_audio.py --sections seq2seq (seq2seq/: the IdrakAudio encoder and a text decoder in torch with
// two weight sets, padded log-mel inputs, each utterance's mean next-token cross-entropy and its gradients, greedy texts
// and three SGD steps).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioSeq2SeqGroup =
    [
        ("audio seq2seq: the IdrakAudio plug-in reads seq2seq/model (the head tied to the embedding) and seq2seq/model-untied (its own head, a decoder narrower than the encoder) and gives torch's step-1 loss within 1e-4 and every tensor's gradient within 1e-4 of that tensor's largest (of a thousandth of the largest of any, for the key biases' zero gradients), through Seq2SeqObjective.Loss; each utterance's decoder states in the padded batch equal its states alone within 1e-5; greedy generation from each utterance's memory gives torch's tokens; three SGD steps' losses and the weights after them within 1e-4", AudioSeq2SeqFixture),
        ("audio seq2seq: SpeechModels.Create makes an IdrakAudio seq2seq model from a spec and a trained tokenizer (the start and end token one class past its tokens) into a folder SpeechModels.Load reads back with the same weights, the decoder's settings in config.json; labels and text round trip; a spec whose classes do not fit the tokenizer refused; it transcribes with greedy and beam decoding, chunked and streamed alike, and passes the kit's SpeechRecognizerSuite", AudioSeq2SeqCreate),
        ("audio seq2seq: SpeechTrainer trains an IdrakAudio seq2seq model with the \"seq2seq\" objective: a run stopped after step 2 and resumed from checkpoint-2 gives the whole run's losses and final weights bit for bit, its state naming the objective; an utterance whose text does not fit the decoder's positions is counted and left out; a 200-step CPU run on tone speech halves its loss and its evaluation CER at step 200 is under 0.5 (step time and CER printed)", AudioSeq2SeqTrain),
    ];

    private static string SeqData(string relative) => AudioData(Path.Combine("seq2seq", relative));

    private static JsonObject SeqReference => JsonNode.Parse(File.ReadAllText(SeqData("reference/loss.json")))!.AsObject();

    private static IEnumerable<(string Name, float[] Values)> Gradients(IdrakAudioSeq2SeqModel model) =>
        model.NamedParameters().Select(p => (p.Name, IdrakAudioWeights.Stored(p.Parameter.Grad ?? throw new Exception($"{p.Name} has no gradient"), p.Transposed)));

    // Each tensor's gradient within `relative` of that tensor's largest expected value, or of a thousandth of the largest
    // gradient of any tensor when its own is smaller: an attention's key bias adds the same to every score of a row, which
    // the softmax takes away, so its gradient is zero in exact arithmetic and rounding noise in both (about 1e-9).
    private static void GradientsClose(Dictionary<string, (int[] Shape, float[] Values)> expected, IEnumerable<(string Name, float[] Values)> actual, float relative, string what)
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

    // The decoder's input for a batch as the objective builds it: the start token, the labels, the end token as padding.
    private static float[] SeqTokens(int[][] labels, int length, int start, int end)
    {
        var tokens = new float[labels.Length * length];
        for (int b = 0; b < labels.Length; b++)
        {
            tokens.AsSpan(b * length, length).Fill(end);
            tokens[b * length] = start;
            for (int i = 0; i < labels[b].Length; i++)
            {
                tokens[b * length + i + 1] = labels[b][i];
            }
        }

        return tokens;
    }

    private static void AudioSeq2SeqFixture(Device device)
    {
        IdrakAudioPlugin.Register();
        try
        {
            var reference = SeqReference;
            int[] frames = [.. reference["frames"]!.AsArray().Select(v => (int)v!)];
            int[][] labels = [.. reference["labels"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
            var mels = ReadNpyFloat32(SeqData("mels.npy"));
            int padded = frames.Max(), n = frames.Length;
            foreach (var (folder, grads) in new[] { ("model", "grads-step1.safetensors"), ("model-untied", "grads-step1-untied.safetensors") })
            {
                using var model = (IdrakAudioSeq2SeqModel)SpeechModels.Load(SeqData(folder), new SpeechModelOptions { Device = device });
                var results = reference["results"]![folder]!;
                Check(model.StartToken == (int)reference["start_token"]! && model.EndToken == (int)reference["end_token"]! && model.Vocabulary == (int)reference["classes"]!,
                    $"{folder}: the start and end tokens and the classes");
                Check(frames.Select(model.StepsFor).SequenceEqual(results["steps"]!.AsArray().Select(v => (int)v!)), $"{folder}: the encoder's steps");
                model.Network.Train();
                using (var scope = new TensorScope())
                {
                    var loss = Seq2SeqObjective.Loss(model, Tensor.From(mels, [n, padded, 80], device), frames, labels);
                    float expected = (float)results["loss_step1"]!;
                    Check(MathF.Abs(loss.Item() - expected) <= 1e-4f * MathF.Abs(expected), $"{folder}: the step-1 loss {loss.Item()} is not torch's {expected}");
                    loss.Backward();
                    GradientsClose(ReadSafeTensors(SeqData("reference/" + grads)), Gradients(model), 1e-4f, $"{folder} step-1 gradient");
                }

                model.Network.Eval();
                using (var scope = new TensorScope())
                using (Autograd.NoGrad())
                {
                    // Each utterance alone (its own frames and labels) gives the batch's decoder states on its own positions.
                    int length = labels.Max(l => l.Length) + 1, width = model.Config.DecoderDim;
                    var memory = model.Encode(Tensor.From(mels, [n, padded, 80], device), frames);
                    int[] steps = [.. frames.Select(model.StepsFor)];
                    var batch = model.Hidden(Tensor.From(SeqTokens(labels, length, model.StartToken, model.EndToken), [n, length], device), memory, steps).ToArray();
                    var greedy = results["greedy_step1"]!.AsArray();
                    for (int b = 0; b < n; b++)
                    {
                        var own = model.Encode(Tensor.From(mels.AsSpan(b * padded * 80, frames[b] * 80), [1, frames[b], 80], device), [frames[b]]);
                        int positions = labels[b].Length + 1;
                        var alone = model.Hidden(Tensor.From(SeqTokens([labels[b]], positions, model.StartToken, model.EndToken), [1, positions], device), own, [steps[b]]).ToArray();
                        CloseRelative(alone, batch.AsSpan(b * length * width, positions * width).ToArray(), 1e-5f, $"{folder} utterance {b}'s decoder states alone against in the batch");
                        int[] generated = model.Generate(memory.Narrow(0, b, 1).Narrow(1, 0, steps[b]), beams: 1);
                        int[] expected = [.. greedy[b]!.AsArray().Select(v => (int)v!)];
                        Check(generated.SequenceEqual(expected), $"{folder} utterance {b}: greedy gives [{string.Join(" ", generated)}], torch [{string.Join(" ", expected)}]");
                    }
                }
            }

            // Three SGD steps as torch took them, and the weights after them.
            using (var model = (IdrakAudioSeq2SeqModel)SpeechModels.Load(SeqData("model"), new SpeechModelOptions { Device = device }))
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
                    var loss = Seq2SeqObjective.Loss(model, Tensor.From(mels, [n, padded, 80], device), frames, labels);
                    losses.Add(loss.Item());
                    loss.Backward();
                }

                for (int s = 0; s < 4; s++)
                {
                    Check(MathF.Abs(losses[s] - expected[s]) <= 1e-4f * MathF.Abs(expected[s]), $"SGD loss {s}: {losses[s]}, torch {expected[s]} ({string.Join(", ", losses)})");
                }

                TensorsClose(ReadSafeTensors(SeqData("reference/weights-step3.safetensors")), model.Export().Select(e => (e.Name, e.Values)), 1e-4f, "the weights after 3 SGD steps");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
        }
    }

    private static JsonObject ToneSeqSpec(int maxTargetPositions = 32) => new()
    {
        ["architecture"] = IdrakAudioFamily.Architecture, ["task"] = "seq2seq", ["mels"] = 80, ["dim"] = 32, ["layers"] = 2, ["heads"] = 4, ["ffn_dim"] = 64,
        ["subsample_layers"] = 2, ["subsample_kernel"] = 3, ["decoder_layers"] = 2, ["max_target_positions"] = maxTargetPositions,
    };

    private static IdrakAudioSeq2SeqModel CreateToneSeq(string folder, IPretrainedTokenizer tokenizer, Device device, int seed = 7, int maxTargetPositions = 32) =>
        (IdrakAudioSeq2SeqModel)SpeechModels.Create(folder, ToneSeqSpec(maxTargetPositions), tokenizer, new SpeechCreateOptions { Device = device, Seed = seed });

    private static void AudioSeq2SeqCreate(Device device)
    {
        string root = TempFolder("seq2seq-create");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 6, 11);
            var tokenizer = ToneTokenizer(root, texts);
            string folder = Path.Combine(root, "model");
            using var model = CreateToneSeq(folder, tokenizer, device);
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            int vocabulary = tokenizer.VocabularySize;
            Check((string?)config["task"] == "seq2seq" && (int)config["classes"]! == vocabulary + 1 && (int)config["start_token"]! == vocabulary && (int)config["end_token"]! == vocabulary
                && (int)config["decoder_layers"]! == 2 && (int)config["decoder_dim"]! == 32 && (int)config["max_target_positions"]! == 32 && (bool)config["tie_embeddings"]! && !config.ContainsKey("blank_id"),
                $"config.json holds the decoder's settings: {config.ToJsonString()}");
            Check(model.StartToken == vocabulary && model.EndToken == vocabulary && model is ISequenceSpeechModel { Objective: SpeechTrainingObjectives.Seq2Seq },
                "the start and end token one class past the tokenizer's, the seq2seq objective");
            Check(model.Labels(texts[0]).Length == texts[0].Length + 1 && model.Text(model.Labels(texts[0])) == texts[0], $"labels of '{texts[0]}' read back as '{model.Text(model.Labels(texts[0]))}'");
            Check(model.Text([model.StartToken, .. model.Labels(texts[0]), model.EndToken]) == texts[0], "the start and end tokens are not text");
            using (var again = (IdrakAudioSeq2SeqModel)SpeechModels.Load(folder, new SpeechModelOptions { Device = device }))
            {
                Check(again.Export().Zip(model.Export()).All(p => p.First.Name == p.Second.Name && p.First.Values.AsSpan().SequenceEqual(p.Second.Values)), "the folder reads back to the same weights");
                Check(again.Config.ToJson().ToJsonString() == model.Config.ToJson().ToJsonString(), "the config reads back the same");
            }

            var wrong = ToneSeqSpec();
            wrong["classes"] = vocabulary + 3;
            ThrowsNaming<ArgumentException>(() => SpeechModels.Create(Path.Combine(root, "wrong"), wrong, tokenizer).Dispose(), "classes that do not fit the tokenizer", "start and end token");

            using var reader = SpeechSources.Open(new SpeechSource(SpeechSources.Transcripts, [transcripts]));
            var audio = AudioCodecs.Decode(reader.ReadBytes(0));
            var greedy = model.Transcribe(audio);
            var beam = model.Transcribe(audio, new TranscriptionOptions { Beams = 3 });
            Check(greedy.Segments.Count == 1 && greedy.Segments[0].End == audio.Duration.TotalSeconds && beam.Segments.Count == 1 && greedy.Segments[0].Tokens.Count <= 31,
                $"one segment for a short sound, at most the decoder's positions less the start token ({greedy.Segments[0].Tokens.Count} tokens)");
            var chunked = model.Transcribe(audio, new TranscriptionOptions { ChunkSeconds = 0.4 });
            int pieces = (int)Math.Ceiling(audio.Duration.TotalSeconds / 0.4 - 1e-9);
            Check(chunked.Segments.Count == pieces, $"{chunked.Segments.Count} chunks of 0.4 s for {pieces}");
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

    private static void AudioSeq2SeqTrain(Device device)
    {
        string root = TempFolder("seq2seq-train");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 12, 21);
            // One more utterance whose text does not fit the decoder's 32 positions.
            File.Copy(Path.Combine(root, "data", "u000.wav"), Path.Combine(root, "data", "long.wav"));
            File.AppendAllText(transcripts, "{\"audio\": \"long.wav\", \"text\": \"" + new string('a', 40) + "\"}\n");
            var tokenizer = ToneTokenizer(root, texts);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = ToneTraining(Path.Combine(root, "cache")) with { SaveEvery = 2 };

            SpeechTrainingResult whole;
            using (var model = CreateToneSeq(Path.Combine(root, "start-a"), tokenizer, device))
            {
                whole = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"));
            }

            Check(whole.Steps == 6 && whole.Losses.All(float.IsFinite) && whole.Dropped == 1 && whole.Utterances == 12, $"the run: {whole.Steps} steps, {whole.Dropped} dropped, {whole.Utterances} used");
            Check(whole.Notes.Any(n => n.Contains("1 utterances have more labels than the decoder's positions hold", StringComparison.Ordinal)), $"the dropped row is noted: {string.Join("; ", whole.Notes)}");
            using (var model = CreateToneSeq(Path.Combine(root, "start-b"), tokenizer, device))
            using (var stop = new CancellationTokenSource())
            {
                var stopper = new Progress([], r =>
                {
                    if (r.Stage == "step" && r.Step == 2)
                    {
                        stop.Cancel();
                    }
                });
                ThrowsNaming<OperationCanceledException>(() => SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run-b"), progress: stopper, cancellationToken: stop.Token),
                    "the run cancelled after step 2");
            }

            string state = File.ReadAllText(Path.Combine(root, "run-b", "checkpoint-2", SpeechTrainer.StateFile));
            Check(JsonNode.Parse(state)!["objective"]?.GetValue<string>() == SpeechTrainingObjectives.Seq2Seq, "the state names the seq2seq objective");
            var resumed = SpeechTrainer.Resume(Path.Combine(root, "run-b", "checkpoint-2"), device: device);
            Check(resumed.Losses.SequenceEqual(whole.Losses), $"resumed losses {string.Join(", ", resumed.Losses)} are not the whole run's {string.Join(", ", whole.Losses)}");
            Check(File.ReadAllBytes(Path.Combine(root, "run-b", "checkpoint-6", "model.safetensors")).AsSpan()
                .SequenceEqual(File.ReadAllBytes(Path.Combine(root, "run-a", "checkpoint-6", "model.safetensors"))), "the resumed run's final weights are the whole run's, bit for bit");

            // The smoke run: the loss halves and the evaluation reads the tone letters.
            var (smokeTranscripts, smokeTexts) = ToneSpeech(Path.Combine(root, "smoke"), 24, 41);
            var smokeTokenizer = ToneTokenizer(Path.Combine(root, "smoke"), smokeTexts);
            var smoke = new SpeechSource(SpeechSources.Transcripts, [smokeTranscripts]);
            var smokeOptions = ToneTraining(Path.Combine(root, "smoke-cache")) with { Steps = 200, BatchSeconds = 8, LearningRate = 3e-3f, WarmupFraction = 0.05f, EvaluateEvery = 100, Checkpointing = false };
            var clock = Stopwatch.StartNew();
            SpeechTrainingResult result;
            using (var model = CreateToneSeq(Path.Combine(root, "smoke-start"), smokeTokenizer, device))
            {
                result = SpeechTrainer.Train(model, smoke, smoke, smokeOptions, Path.Combine(root, "smoke-run"));
            }

            double seconds = clock.Elapsed.TotalSeconds;
            float first = result.Losses[0], last = result.Losses.TakeLast(10).Average();
            Console.WriteLine(FormattableString.Invariant($"    200 steps in {seconds:F1} s ({seconds * 1000 / 200:F1} ms a step); loss {first:F3} at step 1, {last:F3} over the last 10; cer {string.Join(", ", result.Evaluations.Select(e => $"{e.Score:F3} at step {e.Step}"))}"));
            Check(last < first / 2, FormattableString.Invariant($"the loss went from {first:F3} to {last:F3}, not under half"));
            Check(result.Evaluations.Count == 2 && result.Evaluations[1].Score < 0.5, $"the CER at steps 100 and 200: {string.Join(", ", result.Evaluations)}");
            using var trained = SpeechModels.Load(result.Checkpoint!, new SpeechModelOptions { Device = device });
            Console.WriteLine($"    u000 '{smokeTexts[0]}' reads as '{trained.Transcribe(AudioCodecs.Decode(Path.Combine(root, "smoke", "u000.wav"))).Text}'");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
