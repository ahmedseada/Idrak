// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Abstraction.Generation;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.IdrakAudio;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp.Generation;

// Plan 21, phase S6: the IdrakAudio plug-in's audio-language model, against tools/audio/make_tiny_audio.py --sections
// audio-language (audio-language/: a Llama decoder with the IdrakAudio encoder and projector in torch, three
// conversations with sounds, the features, ids and soft-token blocks, joint and align steps' losses, gradients and weights).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioLanguageFamilyGroup =
    [
        ("audio language: the IdrakAudio plug-in reads audio-language/model (its audio part claimed, nothing unused): each sound's log-mel within 1e-4 of torch's, its soft tokens and the records' ids and blocks the fixture's; joint pretraining (every parameter) gives PyTorch's step-1 loss and gradients and its losses and weights after three steps", AudioLanguageJoint),
        ("audio language: align pretraining (audio-encoder and audio-projector; the decoder stays) gives PyTorch's losses, gradients and weights; the decoder's tensors are as they were", AudioLanguageAlign),
        ("audio language: the \"transcripts\" tuning format reads the fixture's train.jsonl as conversations of the sound and the prompt answered by the text (NeMo's keys too, the prompt a setting, an empty one the sound alone; a row without text refused naming it) and passes the kit's CheckTuningDataFormat; FineTuner trains the audio-language model's projector on them through FineTuningOptions.Parts", AudioLanguageTranscripts),
        ("audio language: PretrainedModel.Create with CreateOptions.Parts[\"audio\"] gives the fixture's tensor names and shapes and reads back with its part; align then joint stages stopped between them and resumed give the uninterrupted losses; the trained model answers about a sound through ChatGenerator with its part", AudioLanguageStages),
    ];

    private static string AudioLanguageData(string relative) => AudioData(Path.Combine("audio-language", relative));

    private static PretrainingOptions AudioLanguageStage(string name, IReadOnlyList<string> trained, int steps) => new()
    {
        Stages = [new PretrainingStage(name)
        {
            Trained = trained, Steps = steps, LearningRate = 0.05f, Schedule = "constant", WarmupFraction = 0,
            Sources = [new PretrainingSource(PretrainingSources.Records, [AudioLanguageData("records.jsonl")]) { Format = "messages" }],
        }],
        MaxLength = 128, BatchTokens = 384, SequencesPerStep = 3, Optimizer = "sgd", Momentum = 0, WeightDecay = 0, MaxGradientNorm = 0,
        CudaGraphs = false, KeepCheckpoints = 0,
    };

    private static void CheckAudioLanguageRun(Device device, string kind, IReadOnlyList<string> trained, string folder)
    {
        var expected = JsonNode.Parse(File.ReadAllText(AudioLanguageData("steps.json")))![kind]!["losses"]!.AsArray().Select(l => (float)l!).ToArray();
        var initial = ReadTensors(AudioLanguageData("model/model.safetensors"));
        using (var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device }))
        {
            var one = Pretrainer.Train(model, AudioLanguageStage(kind, trained, 1), Path.Combine(folder, $"{kind}-one"));
            Check(MathF.Abs(one.Losses[0] - expected[0]) < 1e-4f, $"{kind}: step 1's loss {one.Losses[0]:F6}, PyTorch's {expected[0]:F6}");
            var after = ReadTensors(Path.Combine(one.LastCheckpoint!, "model.safetensors"));
            var grads = ReadTensors(AudioLanguageData($"{kind}/grads-step1.safetensors"));
            float largest = grads.Values.Max(g => g.Values.Max(MathF.Abs));
            foreach (var (name, (_, g)) in grads)
            {
                var moved = initial[name].Values.Zip(after[name].Values, (a, b) => (a - b) / 0.05f).ToArray();
                AssertClose(g, moved, 1e-4f * largest, $"{kind}: the gradient of {name}");
            }

            foreach (var name in initial.Keys.Where(n => !grads.ContainsKey(n)))
            {
                Check(after[name].Values.AsSpan().SequenceEqual(initial[name].Values), $"{kind}: {name} is frozen and stays");
            }
        }

        using (var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device }))
        {
            var three = Pretrainer.Train(model, AudioLanguageStage(kind, trained, 3), Path.Combine(folder, $"{kind}-three"));
            AssertClose(expected, [.. three.Losses], 1e-4f, $"{kind}: three steps' losses");
            var after = ReadTensors(Path.Combine(three.LastCheckpoint!, "model.safetensors"));
            foreach (var (name, (_, w)) in ReadTensors(AudioLanguageData($"{kind}/weights-step3.safetensors")))
            {
                AssertClose(w, after[name].Values, 5e-5f, $"{kind}: {name} after three steps");
            }
        }
    }

    private static void AudioLanguageJoint(Device device)
    {
        IdrakAudioPlugin.Register();
        string folder = TempFolder("audio-language-joint");
        try
        {
            using (var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device }))
            {
                Check(model.Parts.SequenceEqual([ChatParts.Audio]) && !model.Notes.Any(n => n.Contains("not used", StringComparison.Ordinal)),
                    $"the audio part is claimed: [{string.Join(", ", model.Parts)}]; {string.Join("; ", model.Notes)}");
                var tuning = JsonNode.Parse(File.ReadAllText(AudioLanguageData("tuning.json")))!.AsObject();
                using var source = model.OpenPart(ChatParts.Audio, device);
                var part = (IdrakAudioLanguagePart)((ChatAudioParts)source).Audio;
                foreach (var (path, tokens) in tuning["tokens_per_audio"]!.AsObject())
                {
                    string name = Path.GetFileNameWithoutExtension(path);
                    var audio = AudioCodecs.Decode(AudioLanguageData(path));
                    using var scope = new TensorScope();
                    var mel = part.Features.Extract(audio, device).ToArray();
                    AssertClose(ReadNpyFloat32(AudioLanguageData($"features-{name}.npy")), mel, 1e-4f, $"{name}'s log-mel against torch's");
                    Check(source.Blocks(ChatAudio.FromFile(AudioLanguageData(path))).Single().Tokens == (int)tokens!, $"{name}: {(int)tokens!} soft tokens");
                }

                var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Parts = [source] };
                using var records = TuningDataFormats.Get("messages").ReadPending(AudioLanguageData("records.jsonl"));
                var expected = tuning["records"]!.AsArray();
                for (int i = 0; i < expected.Count; i++)
                {
                    var sequence = encoder.Encode(records.Read(i), 128)!;
                    int[] ids = [.. expected[i]!["input_ids"]!.AsArray().Select(v => (int)v!)];
                    Check(sequence.Tokens.SequenceEqual(ids), $"record {i}: ids [{string.Join(" ", sequence.Tokens)}] against [{string.Join(" ", ids)}]");
                    var blocks = expected[i]!["blocks"]!.AsArray().Select(b => ((int)b![0]!, (int)b[1]!)).ToList();
                    Check(sequence.Parts.SelectMany(p => p.Blocks).SequenceEqual(blocks), $"record {i}: blocks [{string.Join(", ", sequence.Parts.SelectMany(p => p.Blocks))}]");
                }
            }

            CheckAudioLanguageRun(device, "joint", ["all"], folder);
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void AudioLanguageAlign(Device device)
    {
        IdrakAudioPlugin.Register();
        string folder = TempFolder("audio-language-align");
        try
        {
            CheckAudioLanguageRun(device, "align", [IdrakAudioLanguageFamily.EncoderPart, IdrakAudioLanguageFamily.ProjectorPart], folder);
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void AudioLanguageTranscripts(Device device)
    {
        IdrakAudioPlugin.Register();
        string folder = TempFolder("audio-language-transcripts");
        try
        {
            var format = TuningDataFormats.Get(TuningDataFormats.Transcripts);
            string fixture = AudioData("transcripts/train.jsonl");
            var rows = File.ReadAllLines(fixture).Where(l => l.Length > 0).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
            var read = format.Read(fixture, new TuningDataOptions { Audios = AudioData("") }).ToList();
            Check(read.Count == rows.Count && read.Zip(rows).All(p => p.First.Messages is [{ Role: "user", Parts: [ChatAudio a, ChatText { Text: "Transcribe." }] }, { Role: "assistant" } answer]
                    && a.Data.Span.SequenceEqual(File.ReadAllBytes(AudioData((string)p.Second["audio"]!))) && answer.Content == (string)p.Second["text"]!),
                $"each row is the sound and the prompt answered by the text ({read.Count} of {rows.Count})");
            Check(format.Recognizes(rows[0]) && !format.Recognizes(new JsonObject { ["messages"] = new JsonArray() }), "it recognizes transcript rows only");

            string nemo = Path.Combine(folder, "nemo.jsonl");
            var row = new JsonObject { ["audio_filepath"] = AudioData("wav/pcm16.wav"), ["duration"] = 1.5, ["text"] = "the quick brown fox" };
            File.WriteAllLines(nemo, [row.ToJsonString()]);
            var bare = format.Read(nemo, new TuningDataOptions { Prompt = "" }).Single();
            Check(bare.Messages[0].Parts is [ChatAudio] && bare.Messages[1].Content == "the quick brown fox", "NeMo's audio_filepath; an empty prompt gives the sound alone");
            string bad = Path.Combine(folder, "bad.jsonl");
            File.WriteAllLines(bad, [new JsonObject { ["audio"] = "wav/pcm16.wav" }.ToJsonString()]);
            ThrowsNaming<InvalidDataException>(() => format.Read(bad).ToList(), "a row without text", "bad.jsonl", "needs a sound");
            Conformance.CheckTuningDataFormat(path => format.Read(path, new TuningDataOptions { Audios = AudioData("") }).Select(t => t.Messages), [new TuningDataSample(fixture)]).ThrowIfFailed();

            // Tuning the audio-language model's projector on transcripts.
            string english = Path.Combine(folder, "train.jsonl");
            File.WriteAllLines(english, [.. new[] { ("wav/pcm16.wav", "the quick brown fox"), ("wav/pcm8.wav", "hello there"), ("wav/list-chunk16.wav", "a short tone") }
                .Select(r => new JsonObject { ["audio"] = r.Item1, ["text"] = r.Item2 }.ToJsonString())]);
            using var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device });
            using var source = model.OpenPart(ChatParts.Audio, device);
            source.Train([IdrakAudioLanguageFamily.ProjectorPart]);
            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Parts = [source] };
            var sequences = format.Read(english, new TuningDataOptions { Audios = AudioData("") }).Select(t => encoder.Encode(t, 128)!).ToList();
            var before = source.Export([IdrakAudioLanguageFamily.ProjectorPart]).ToDictionary(t => t.Name, t => t.Values);
            float initial = FineTuner.Evaluate(model, sequences, 384, 512, parts: [source]);
            FineTuner.Train(model, sequences, null, new FineTuningOptions { Epochs = 3, BatchTokens = 384, MaxLength = 128, LearningRate = 1e-2f, Parts = [source] }, Path.Combine(folder, "tuned"));
            float after = FineTuner.Evaluate(model, sequences, 384, 512, parts: [source]);
            var moved = source.Export([IdrakAudioLanguageFamily.ProjectorPart]);
            Check(moved.Any(t => !t.Values.AsSpan().SequenceEqual(before[t.Name])) && after < initial, $"the projector trained on transcripts: loss {initial} then {after}");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void AudioLanguageStages(Device device)
    {
        IdrakAudioPlugin.Register();
        string folder = TempFolder("audio-language-stages");
        try
        {
            // A model made from a spec has the fixture's tensors (names and shapes) and reads back with its part.
            var config = JsonNode.Parse(File.ReadAllText(AudioLanguageData("model/config.json")))!.AsObject();
            var spec = PretrainedArchitectures.Get(IdrakAudioLanguageFamily.Architecture).Spec(config, []);
            var description = new JsonObject
            {
                ["audio_config"] = config["audio_config"]!.DeepClone(),
                ["audio_projector"] = new JsonObject { ["stack"] = 4, ["type"] = "mlp" },
                ["audio_token"] = "<image>", ["begin_audio"] = "<start_of_image>", ["end_audio"] = "<end_of_image>",
            };
            string made = Path.Combine(folder, "made");
            var template = JsonNode.Parse(File.ReadAllText(AudioLanguageData("model/tokenizer_config.json")))!["chat_template"]!.GetValue<string>();
            using (var model = PretrainedModel.Create(made, spec, IdrakAudioLanguageFamily.Architecture,
                new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")), ChatTemplate = template, BosToken = "<s>", PadToken = "<pad>", StopTokens = ["</s>"] },
                new CreateOptions { Device = device, Seed = 4, Parts = new Dictionary<string, JsonObject> { [ChatParts.Audio] = description } }))
            {
                var expected = ReadTensors(AudioLanguageData("model/model.safetensors"));
                var created = ReadTensors(Path.Combine(made, "model.safetensors"));
                Check(created.Keys.Order(StringComparer.Ordinal).SequenceEqual(expected.Keys.Order(StringComparer.Ordinal)),
                    $"the family's tensor names: {string.Join(", ", created.Keys.Except(expected.Keys).Take(5))} | {string.Join(", ", expected.Keys.Except(created.Keys).Take(5))}");
                Check(created.All(t => t.Value.Shape.SequenceEqual(expected[t.Key].Shape)), "and shapes");
                Check(model.Parts.SequenceEqual([ChatParts.Audio]) && !model.Notes.Any(n => n.Contains("not used", StringComparison.Ordinal)), "the made model reads back with its audio part");
                var madeConfig = JsonNode.Parse(File.ReadAllText(Path.Combine(made, "config.json")))!.AsObject();
                foreach (string key in new[] { "audio_token_index", "boa_token_index", "eoa_token_index", "audio_attention" })
                {
                    Check(JsonNode.DeepEquals(madeConfig[key], config[key]), $"config.json's {key}: {madeConfig[key]} against {config[key]}");
                }
            }

            // Align then joint, stopped between them and resumed.
            var align = AudioLanguageStage("align", [IdrakAudioLanguageFamily.EncoderPart, IdrakAudioLanguageFamily.ProjectorPart], 2).Stages[0];
            var joint = AudioLanguageStage("joint", ["all"], 2).Stages[0];
            var options = AudioLanguageStage("x", ["all"], 1) with { Stages = [align, joint], SequencesPerStep = 3, Optimizer = "adamw", Seed = 3 };
            IReadOnlyList<float> whole;
            using (var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device }))
            {
                whole = Pretrainer.Train(model, options, Path.Combine(folder, "whole")).Losses;
            }

            Check(whole.Count == 4 && whole.All(float.IsFinite), $"two stages of two steps: {string.Join(", ", whole)}");
            string run = Path.Combine(folder, "stopped");
            using (var model = PretrainedModel.Load(AudioLanguageData("model"), new PretrainedOptions { Device = device }))
            {
                using var stop = new CancellationTokenSource();
                Throws<OperationCanceledException>(() => Pretrainer.Train(model, options, run, null,
                    new SynchronousProgress<PretrainingProgress>(p => { if (p.Step == 2) { stop.Cancel(); } }), stop.Token), "stopped after the align stage");
            }

            var checkpoint = Pretrainer.ReadCheckpoint(Pretrainer.LatestCheckpoint(run)!);
            var aligned = ReadTensors(Path.Combine(checkpoint.Folder, "model.safetensors"));
            var initial = ReadTensors(AudioLanguageData("model/model.safetensors"));
            Check(!aligned["audio_tower.norm.weight"].Values.SequenceEqual(initial["audio_tower.norm.weight"].Values)
                  && aligned["language_model.model.norm.weight"].Values.SequenceEqual(initial["language_model.model.norm.weight"].Values),
                "the checkpoint holds the trained audio part and the untouched decoder");
            using (var model = PretrainedModel.Load(checkpoint.Folder, new PretrainedOptions { Device = device }))
            {
                var resumed = Pretrainer.Train(model, checkpoint.Options, run, checkpoint);
                Check(resumed.Losses.SequenceEqual(whole), $"resumed [{string.Join(", ", resumed.Losses)}] against [{string.Join(", ", whole)}]");
                using var source = model.OpenPart(ChatParts.Audio, device);
                var chat = new ChatGenerator(model.CreateChat(KeyValueFormat.Float32, 128).Generator, model.ChatTemplate) { Parts = [source] };
                var answer = chat.Chat(new ChatRequest([new ChatMessage("user", [ChatAudio.FromFile(AudioLanguageData("wav/pcm16.wav")), new ChatText("Transcribe.")])],
                    Options: new GenerationOptions { Temperature = 0, NumPredict = 4 }));
                Check(answer.Done, "the trained model answers about a sound");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }
}
