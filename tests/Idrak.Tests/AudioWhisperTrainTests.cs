// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Whisper;
using Idrak.Models;

// Plan 29: fine-tuning Whisper. The plug-in's recognizer as an ISequenceSpeechModel on the fixture
// tests/Idrak.Tests/data/audio/tiny-whisper, its loss and gradients against transformers' (tools/audio/make_tiny_audio.py,
// section whisper-train), its labels, its saved folder, a short SpeechTrainer run and the command line.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioWhisperTrainGroup =
    [
        ("audio whisper train: the seq2seq loss of a two-utterance batch and every trained parameter's gradient under transformers' names within 1e-4 of transformers' (tiny-whisper; the labels the prompt after <|startoftranscript|> then the text's tokens, as transformers'; the encoder's fixed positions get none, as OpenAI's Whisper keeps them)", WhisperTrainReference),
        ("audio whisper train: the labels name the language and task given (or generation_config.json's), a multilingual checkpoint without a language refuses naming --language, Text drops the prompt's tokens, the encoder steps of a segment and past it, a network of packed (bfloat16) weights refused naming SpeechModelOptions.Weights", WhisperTrainLabels),
        ("audio whisper train: Save writes every stored tensor under its name bit for bit (float32), config.json, generation_config.json with the language and task, the tokenizer and preprocessor; the folder loads back with its language and the same encoder output", WhisperTrainSave),
        ("audio whisper train: a short SpeechTrainer run on tone speech lowers the loss, a clip longer than the 30 s segment is left out with a note, the checkpoint loads as Whisper in its language and transcribes", WhisperTrainRun),
        ("audio whisper train: idrak pretrain on a Whisper folder refuses without --language naming it; with --language ar two steps write a checkpoint under transformers' names", WhisperTrainCli),
    ];

    private static WhisperRecognizer TrainableWhisper(Device device, string? language = "ar", string? task = null, string? folder = null, EncoderWeights weights = EncoderWeights.Float32)
    {
        WhisperPlugin.Register();
        return (WhisperRecognizer)SpeechModels.Load(folder ?? AudioData("tiny-whisper"),
            new SpeechModelOptions { Device = device, Weights = weights, Language = language, Task = task });
    }

    private static void WhisperTrainReference(Device device)
    {
        var reference = JsonNode.Parse(File.ReadAllText(AudioData("tiny-whisper/reference/train.json")))!.AsObject();
        string[] texts = [.. reference["texts"]!.AsArray().Select(t => (string)t!)];
        int[][] labels = [.. reference["labels"]!.AsArray().Select(row => row!.AsArray().Select(v => (int)v!).ToArray())];
        using var model = TrainableWhisper(device);
        for (int i = 0; i < texts.Length; i++)
        {
            Check(model.Labels(texts[i]).SequenceEqual(labels[i]), $"the labels of '{texts[i]}': [{string.Join(", ", model.Labels(texts[i]))}], transformers' [{string.Join(", ", labels[i])}]");
        }

        // The fixture's log-mel [80, 3000] as the trainer lays utterances out ([frames, mels]), and the same reversed in time.
        var mel = ReadNpyFloat32(AudioData("features/log-mel.npy"));
        const int Frames = 3000, Mels = 80;
        var features = new float[2 * Frames * Mels];
        for (int t = 0; t < Frames; t++)
        {
            for (int m = 0; m < Mels; m++)
            {
                features[t * Mels + m] = mel[m * Frames + t];
                features[Frames * Mels + t * Mels + m] = mel[m * Frames + Frames - 1 - t];
            }
        }

        model.Network.Train();
        using var input = Tensor.From(features, [2, Frames, Mels], device);
        var loss = Seq2SeqObjective.Loss(model, input, [Frames, Frames], labels);
        float expected = (float)reference["loss"]!;
        Check(MathF.Abs(loss.Item() - expected) <= 1e-4f * Math.Max(1f, MathF.Abs(expected)), $"the loss {loss.Item()} on {device}, transformers' {expected}");
        loss.Backward();

        using var grads = SafeTensorsReader.Open(AudioData("tiny-whisper/reference/train-grads.safetensors"));
        var ours = model.Gradients().ToDictionary(g => g.Name);
        int compared = 0;
        // The encoder's sinusoidal positions stay fixed here, as in OpenAI's Whisper (a buffer) and transformers 4 (frozen);
        // transformers 5.19 records a gradient for them, the one tensor not compared.
        const string Positions = "model.encoder.embed_positions.weight";
        foreach (string name in grads.Tensors.Keys.Where(n => n != Positions))
        {
            Check(ours.TryGetValue(name, out var mine) && mine.Gradient is not null, $"transformers' gradient '{name}' has no counterpart with a gradient here");
            var theirs = grads.Read(name);
            AssertClose(theirs, mine.Gradient!, 1e-4f * Largest(theirs), $"the gradient of {name} on {device}");
            compared++;
        }

        var extra = ours.Values.Where(g => g.Gradient is not null && !grads.Contains(g.Name)).Select(g => g.Name).ToList();
        Check(extra.Count == 0, $"gradients transformers has none of: {string.Join(", ", extra)}");
        Check(ours[Positions].Gradient is null, "the encoder's fixed positions got a gradient");
        Console.WriteLine($"    loss {loss.Item():F5} (transformers' {expected:F5}); {compared} gradients compared");
    }

    private static void WhisperTrainLabels(Device device)
    {
        int[] Ids(params string[] tokens)
        {
            var specials = JsonNode.Parse(File.ReadAllText(AudioData("tiny-whisper/special_tokens.json")))!.AsObject();
            return [.. tokens.Select(t => (int)specials[t]!)];
        }

        const string English = "the quick brown fox jumps over the lazy dog";
        using (var model = TrainableWhisper(device, language: null))
        {
            ThrowsNaming<InvalidOperationException>(() => model.Labels(English), "labels without a language", "--language", "SpeechModelOptions.Language", "generation_config.json");
            Check(model.StepsFor(3000) == 1500 && model.StepsFor(2999) == 1500 && model.StepsFor(3002) == 0 && model.MaxPositions == 448,
                $"steps {model.StepsFor(3000)}, {model.StepsFor(2999)}, {model.StepsFor(3002)}; positions {model.MaxPositions}");
        }

        using (var model = TrainableWhisper(device, language: "<|AR|>", task: "translate"))
        {
            var labels = model.Labels(English);
            Check(labels.Take(3).SequenceEqual(Ids("<|ar|>", "<|translate|>", "<|notimestamps|>")) && model.Language == "ar" && model.Task == "translate",
                $"the labels start [{string.Join(", ", labels.Take(3))}], language {model.Language}, task {model.Task}");
            Check(model.Text(labels) == English && model.Text([.. labels, model.EndToken]) == English, $"Text gives '{model.Text(labels)}'");
            string objective = ((ITrainableSpeechModel)model).Objective;
            Check(objective == SpeechTrainingObjectives.Seq2Seq && model.Architecture == WhisperFamily.Architecture && model.StartToken == Ids("<|startoftranscript|>")[0],
                $"objective {objective}, architecture {model.Architecture}, start {model.StartToken}");
        }

        using (var packed = TrainableWhisper(device, weights: EncoderWeights.BFloat16))
        {
            ThrowsNaming<InvalidOperationException>(() => _ = packed.Network, "the network of bfloat16 weights", "SpeechModelOptions.Weights");
        }
    }

    private static void WhisperTrainSave(Device device)
    {
        string root = TempFolder("whisper-save");
        try
        {
            string saved = Path.Combine(root, "saved");
            float[] before;
            using (var model = TrainableWhisper(device, task: "transcribe"))
            {
                using var mel = FixtureMel(device);
                using var encoded = model.Encode(mel);
                before = encoded.ToArray();
                model.Save(saved);
            }

            foreach (string file in new[] { "config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json", "special_tokens.json", "model.safetensors" })
            {
                Check(File.Exists(Path.Combine(saved, file)), $"{file} was not written");
            }

            Check(!Directory.Exists(Path.Combine(saved, "reference")), "a subfolder of the checkpoint was copied");
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(saved, "config.json")))!.AsObject();
            var generation = JsonNode.Parse(File.ReadAllText(Path.Combine(saved, "generation_config.json")))!.AsObject();
            Check((string?)config["dtype"] == "float32" && (string?)generation["language"] == "ar" && (string?)generation["task"] == "transcribe"
                  && generation["lang_to_id"] is JsonObject, $"config dtype {config["dtype"]}; generation_config {generation.ToJsonString()}");

            using (var original = SafeTensorsReader.Open(AudioData("tiny-whisper/model.safetensors")))
            using (var written = SafeTensorsReader.Open(Path.Combine(saved, "model.safetensors")))
            {
                Check(original.Tensors.Keys.Order().SequenceEqual(written.Tensors.Keys.Order()),
                    $"names: missing {string.Join(", ", original.Tensors.Keys.Except(written.Tensors.Keys))}; extra {string.Join(", ", written.Tensors.Keys.Except(original.Tensors.Keys))}");
                foreach (string name in original.Tensors.Keys)
                {
                    Check(original.Tensors[name].Shape.SequenceEqual(written.Tensors[name].Shape) && original.Read(name).SequenceEqual(written.Read(name)), $"{name} changed in the round trip");
                }

                Console.WriteLine($"    {original.Tensors.Count} tensors written back bit for bit");
            }

            using var loaded = TrainableWhisper(device, language: null, folder: saved);
            Check(loaded.Language == "ar" && loaded.Task == "transcribe", $"the saved folder loads in {loaded.Language}, {loaded.Task}");
            using var mel2 = FixtureMel(device);
            using var again = loaded.Encode(mel2);
            AssertClose(before, again.ToArray(), 0f, "the encoder output of the saved folder");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void WhisperTrainRun(Device device)
    {
        string root = TempFolder("whisper-run");
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 8, 51);
            AudioEncoders.Save(Path.Combine(root, "data", "long.wav"), new AudioData(new float[31 * 16000], 1, 16000));   // past the 30 s segment
            File.AppendAllText(transcripts, "{\"audio\": \"long.wav\", \"text\": \"aaa\"}\n");
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = new SpeechTrainingOptions { Steps = 12, BatchSeconds = 4, LearningRate = 1e-3f, WarmupFraction = 0f, Seed = 3, FeatureCache = FeatureCaches.Memory };
            SpeechTrainingResult result;
            using (var model = TrainableWhisper(device))
            {
                result = SpeechTrainer.Train(model, train, null, options, Path.Combine(root, "run"));
            }

            float first = result.Losses.Take(3).Average(), last = result.Losses.TakeLast(3).Average();
            Console.WriteLine(FormattableString.Invariant($"    12 steps in {result.Elapsed.TotalSeconds:F1} s; loss {first:F3} over the first 3, {last:F3} over the last 3"));
            Check(last < first, FormattableString.Invariant($"the loss went from {first:F3} to {last:F3}"));
            Check(result.Dropped == 1 && result.Notes.Any(n => n.Contains("longer than the features' 30 s segment", StringComparison.Ordinal)),
                $"dropped {result.Dropped}; notes: {string.Join(" | ", result.Notes)}");
            using var trained = SpeechModels.Load(result.Checkpoint!, new SpeechModelOptions { Device = device });
            Check(trained is WhisperRecognizer { Language: "ar" }, $"the checkpoint loads as {trained.GetType().Name} in {(trained as WhisperRecognizer)?.Language}");
            var text = trained.Transcribe(AudioCodecs.Decode(Path.Combine(root, "data", "u000.wav")), new TranscriptionOptions { MaxTokens = 8 }).Text;
            Console.WriteLine($"    u000 '{texts[0]}' reads as '{text}'");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void WhisperTrainCli(Device device)
    {
        string root = TempFolder("whisper-cli");
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 4, 61);
            string model = AudioData("tiny-whisper"), d = device.ToString();
            var refused = TrainCli(["pretrain", model, "--data", transcripts, "-o", Path.Combine(root, "refused"), "-P", IdrakAudioPluginPath, "-d", d, .. CliToneTrainingArgs(2)]);
            Check(refused.Code != 0 && refused.Err.Contains("--language", StringComparison.Ordinal), $"without --language: exit {refused.Code}\n{refused.Err}");

            string run = Path.Combine(root, "run");
            var result = TrainCliJson(["pretrain", model, "--data", transcripts, "--language", "ar", "-o", run, "-P", IdrakAudioPluginPath, "-d", d, .. CliToneTrainingArgs(2)]);
            Check((int)result["steps"]! == 2 && result["checkpoint"] is not null, $"pretrain: {result}");
            string checkpoint = (string)result["checkpoint"]!;
            using var written = SafeTensorsReader.Open(Path.Combine(checkpoint, "model.safetensors"));
            Check(written.Contains("model.encoder.layers.0.self_attn.k_proj.weight") && !written.Contains("model.encoder.layers.0.self_attn.k_proj.bias")
                  && written.Contains("model.decoder.embed_tokens.weight"), $"the checkpoint's names: {string.Join(", ", written.Tensors.Keys.Take(8))}");
            var generation = JsonNode.Parse(File.ReadAllText(Path.Combine(checkpoint, "generation_config.json")))!.AsObject();
            Check((string?)generation["language"] == "ar", $"generation_config.json: {generation.ToJsonString()}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
