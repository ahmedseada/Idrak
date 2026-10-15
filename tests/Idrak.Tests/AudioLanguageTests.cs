// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 21, phase S6: the audio-language task's plumbing. A model part stored beside the decoder (ModelParts: made with
// the model, claimed when it loads, carried by every save, opened as a trainable soft-part source) and the pretraining
// step reading it: here with a fake "audio" part (a projector of fixed per-sound inputs), so the library's wiring is
// checked apart from any family.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioLanguageGroup =
    [
        ("audio language: a model made with a fake \"audio\" part (CreateOptions.Parts) writes its config keys, tensors and files; Load counts its tensors as used and lists the part; OpenPart opens it with the checkpoint's tensors, KeepTrainedParts replaces them there and in SaveHuggingFace; a tensor of no part and an unregistered kind refused", AudioLanguageParts),
        ("audio language: pretraining on conversations with sounds trains the part: step 1's loss is the evaluated loss, the projector's step-1 gradient (SGD) matches central differences of it; a stage training the part alone leaves the decoder as it was; the next stage starts from the kept part and every checkpoint holds it; an unknown stage name refused naming the parts", AudioLanguagePretrain),
    ];

    private const string FakePartKey = "fake_audio", FakePartTensor = "fake_audio.proj.weight", FakePartName = "fake-projector";
    private const int FakePartInputs = 4, FakePartTokens = 2;

    // A sound's fixed inputs [tokens, inputs] (drawn from its hash): the part's features are these times its projector.
    private static float[] FakePartInputsOf(ChatPart part)
    {
        var random = new Random(((ChatAudio)part).Hash.GetHashCode(StringComparison.Ordinal));
        return [.. Enumerable.Range(0, FakePartTokens * FakePartInputs).Select(_ => (float)(random.NextDouble() * 2 - 1))];
    }

    private sealed class FakeModelPart : IModelPart
    {
        public string Kind => ChatParts.Audio;

        public bool In(JsonObject config) => config[FakePartKey] is JsonObject;

        public IReadOnlySet<string> StoredTensors(ModelPartCheckpoint checkpoint) =>
            checkpoint.Tensors.Contains(FakePartTensor) ? new HashSet<string> { FakePartTensor } : throw new InvalidDataException($"no {FakePartTensor}");

        public CreatedModelPart Create(ModelPartRequest request)
        {
            var random = new Random(request.Seed);
            var values = Enumerable.Range(0, FakePartInputs * request.TextDim).Select(_ => (float)((random.NextDouble() * 2 - 1) * 0.5)).ToArray();
            return new CreatedModelPart(new JsonObject { [FakePartKey] = new JsonObject { ["inputs"] = FakePartInputs } },
                [(FakePartTensor, [FakePartInputs, request.TextDim], values)], new Dictionary<string, string> { ["fake_audio.json"] = "{}" });
        }

        public ITrainableSoftParts Open(ModelPartCheckpoint checkpoint, ModelPartOptions options)
        {
            var (shape, values) = options.Trained is { } trained && trained.TryGetValue(FakePartTensor, out var kept)
                ? kept : (checkpoint.Tensors.ShapeOf(FakePartTensor), checkpoint.Tensors.Read(FakePartTensor));
            return new FakeTrainableSource(Tensor.From(values, shape, options.Device));
        }
    }

    private sealed class FakeTrainableSource(Tensor projector) : ITrainableSoftParts
    {
        public Tensor Projector { get; } = projector;

        public string Kind => ChatParts.Audio;

        public int Width => Projector.Shape[1];

        public IImagePromptFormat Format { get; } = new ImageTokenFormat("fake-audio", TinyLlamaAudioMarker[0], TinyLlamaAudioMarker[1], end: TinyLlamaAudioMarker[2]);

        public IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

        public IReadOnlyList<string> TrainableParts => [FakePartName];

        public IReadOnlyList<Tensor> Parameters { get; private set; } = [];

        public IReadOnlyList<ImageTokenLayout> Blocks(ChatPart part) => [new ImageTokenLayout(FakePartTokens)];

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ChatPart> parts)
        {
            using var noGrad = Autograd.NoGrad();
            return EncodeTraining(parts, false);
        }

        public IReadOnlyList<ImageFeatures> EncodeTraining(IReadOnlyList<ChatPart> parts, bool checkpointing) =>
            [.. parts.Select(p => new ImageFeatures(Tensor.From(FakePartInputsOf(p), [FakePartTokens, FakePartInputs], Projector.Device).MatMul(Projector), new ImageTokenLayout(FakePartTokens)))];

        public IReadOnlyList<Tensor> Train(IReadOnlyCollection<string> parts)
        {
            var unknown = parts.FirstOrDefault(p => p != FakePartName);
            if (unknown is not null)
            {
                throw new ArgumentException($"no part '{unknown}'");
            }

            Projector.RequiresGrad = parts.Count > 0;
            Parameters = parts.Count > 0 ? [Projector] : [];
            return Parameters;
        }

        public void SetTraining(bool training)
        {
        }

        public IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IReadOnlyCollection<string> parts) =>
            parts.Contains(FakePartName) ? [(FakePartTensor, [.. Projector.Shape], Projector.ToArray())] : [];

        public void Dispose() => Projector.Dispose();
    }

    // The tiny Llama's spec as a model made with the fake part, its ChatML template with an audio branch.
    private static PretrainedModel CreateWithFakePart(string folder, Device device) =>
        PretrainedModel.Create(folder, TinyLlamaSpec(), "LlamaForCausalLM",
            new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")), ChatTemplate = AudioTemplate, BosToken = "<s>", PadToken = "<pad>", StopTokens = ["</s>"] },
            new CreateOptions { Device = device, Seed = 5, Parts = new Dictionary<string, JsonObject> { [ChatParts.Audio] = new JsonObject() } });

    private static void AudioLanguageParts(Device device)
    {
        ModelParts.Register(new FakeModelPart());
        string folder = TempFolder("audio-language-parts");
        try
        {
            ThrowsNaming<NotSupportedException>(() => ModelParts.Get("no-such-part"), "an unregistered kind", "'no-such-part' is not registered", "ModelParts.Register");
            string made = Path.Combine(folder, "made");
            using var model = CreateWithFakePart(made, device);
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(made, "config.json")))!.AsObject();
            var stored = ReadTensors(Path.Combine(made, "model.safetensors"));
            Check(config[FakePartKey] is JsonObject && File.Exists(Path.Combine(made, "fake_audio.json")) && stored[FakePartTensor].Shape.SequenceEqual([FakePartInputs, model.Spec.Dim]),
                "the part's config key, file and tensor are written");
            Check(model.Parts.SequenceEqual([ChatParts.Audio]) && !model.Notes.Any(n => n.Contains("not used", StringComparison.Ordinal)),
                $"the part is listed and its tensor counted as used: [{string.Join(", ", model.Parts)}]; {string.Join("; ", model.Notes)}");
            using (var source = model.OpenPart(ChatParts.Audio, device))
            {
                Check(((FakeTrainableSource)source).Projector.ToArray().SequenceEqual(stored[FakePartTensor].Values), "OpenPart reads the checkpoint's tensor");
            }

            var trained = stored[FakePartTensor].Values.Select(v => v + 1f).ToArray();
            Check(model.KeepTrainedParts([(FakePartTensor, [FakePartInputs, model.Spec.Dim], trained)]) == 1 && model.TrainedPartTensors.SequenceEqual([FakePartTensor]), "the trained tensor is kept");
            using (var source = model.OpenPart(ChatParts.Audio, device))
            {
                Check(((FakeTrainableSource)source).Projector.ToArray().SequenceEqual(trained), "OpenPart puts the kept tensor in");
            }

            string saved = Path.Combine(folder, "saved");
            model.SaveHuggingFace(saved, SafeTensorType.F32);
            Check(ReadTensors(Path.Combine(saved, "model.safetensors"))[FakePartTensor].Values.SequenceEqual(trained), "the save writes the kept tensor");
            using (var again = PretrainedModel.Load(saved, new PretrainedOptions { Device = device }))
            {
                Check(again.Parts.SequenceEqual([ChatParts.Audio]), "the saved model holds the part");
            }

            ThrowsNaming<ArgumentException>(() => model.KeepTrainedParts([("model.norm.weight", [1], [0f])]), "a tensor of no part", "not a tensor of the model's parts");
            ThrowsNaming<InvalidOperationException>(() => model.OpenPart("video"), "a part the model lacks", "no 'video' part");
        }
        finally
        {
            ModelParts.Unregister(ChatParts.Audio);
            Directory.Delete(folder, recursive: true);
        }
    }

    // Three conversations, each a sound and its answer, as ShareGPT records whose sounds are in the fixture's wav folder.
    private static string FakePartRecords(string folder)
    {
        string file = Path.Combine(folder, "records.jsonl");
        string[] rows =
        [
            """{"conversations": [{"from": "human", "value": "<audio>Transcribe."}, {"from": "gpt", "value": "the quick brown fox"}], "audios": ["pcm16.wav"]}""",
            """{"conversations": [{"from": "human", "value": "<audio>What is said?"}, {"from": "gpt", "value": "hello there"}], "audios": ["rate8k.wav"]}""",
            """{"conversations": [{"from": "human", "value": "Listen: <audio>"}, {"from": "gpt", "value": "a short tone"}], "audios": ["stereo16.wav"]}""",
        ];
        File.WriteAllLines(file, rows);
        return file;
    }

    private static PretrainingOptions FakePartStage(string records, IReadOnlyList<string> trained, int steps) => new()
    {
        Stages = [new PretrainingStage("stage")
        {
            Trained = trained, Steps = steps, LearningRate = 0.05f, Schedule = "constant", WarmupFraction = 0,
            Sources = [new PretrainingSource(PretrainingSources.Records, [records]) { Format = "sharegpt", Audios = AudioData("wav") }],
        }],
        MaxLength = 64, BatchTokens = 192, SequencesPerStep = 3, Optimizer = "sgd", Momentum = 0, WeightDecay = 0, MaxGradientNorm = 0,
        CudaGraphs = false, KeepCheckpoints = 0,
    };

    // The mean loss over the records' trained tokens with the part's projector as it is.
    private static float FakePartLoss(PretrainedModel model, ITrainableSoftParts source, string records)
    {
        var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Parts = [source] };
        var format = TuningDataFormats.Get("sharegpt");
        var file = format.ReadPending(records, new TuningDataOptions { Audios = AudioData("wav") });
        var sequences = Enumerable.Range(0, (int)file.Count).Select(i => encoder.Encode(file.Read(i), 64)!).ToList();
        return FineTuner.Evaluate(model, sequences, 192, 512, parts: [source]);
    }

    private static void AudioLanguagePretrain(Device device)
    {
        ModelParts.Register(new FakeModelPart());
        string folder = TempFolder("audio-language-pretrain");
        try
        {
            string records = FakePartRecords(folder);
            string made = Path.Combine(folder, "made");
            using (CreateWithFakePart(made, device))
            {
            }

            var initial = ReadTensors(Path.Combine(made, "model.safetensors"));

            // One SGD step training everything: its loss is the evaluated one, the projector's move is the gradient.
            float evaluated;
            float[] numeric = new float[3];
            int[] probes = [0, 7, 3 * 32 + 5];
            using (var model = PretrainedModel.Load(made, new PretrainedOptions { Device = device }))
            using (var source = model.OpenPart(ChatParts.Audio, device))
            {
                evaluated = FakePartLoss(model, source, records);
                var projector = ((FakeTrainableSource)source).Projector;
                var values = projector.ToArray();
                const float Eps = 1e-2f;
                for (int k = 0; k < probes.Length; k++)
                {
                    var plus = (float[])values.Clone();
                    plus[probes[k]] += Eps;
                    projector.CopyFrom(plus);
                    float up = FakePartLoss(model, source, records);
                    var minus = (float[])values.Clone();
                    minus[probes[k]] -= Eps;
                    projector.CopyFrom(minus);
                    float down = FakePartLoss(model, source, records);
                    numeric[k] = (up - down) / (2 * Eps);
                }
            }

            string one = Path.Combine(folder, "one");
            using (var model = PretrainedModel.Load(made, new PretrainedOptions { Device = device }))
            {
                var result = Pretrainer.Train(model, FakePartStage(records, ["all"], 1), one);
                Check(MathF.Abs(result.Losses[0] - evaluated) <= 1e-5f * MathF.Max(1, evaluated), $"step 1's loss {result.Losses[0]} is the evaluated {evaluated}");
                var after = ReadTensors(Path.Combine(result.LastCheckpoint!, "model.safetensors"));
                var moved = initial[FakePartTensor].Values.Zip(after[FakePartTensor].Values, (a, b) => (a - b) / 0.05f).ToArray();
                for (int k = 0; k < probes.Length; k++)
                {
                    float analytic = moved[probes[k]];
                    Check(MathF.Abs(analytic - numeric[k]) <= 2e-2f * MathF.Max(MathF.Abs(numeric[k]), 1e-2f), $"the projector's gradient at {probes[k]}: {analytic}, central differences {numeric[k]}");
                }

                Check(moved.Any(g => g != 0f), "the part trained");
            }

            // The part alone: the decoder stays; the next stage starts from the kept part.
            string align = Path.Combine(folder, "align");
            string alignCheckpoint;
            using (var model = PretrainedModel.Load(made, new PretrainedOptions { Device = device }))
            {
                alignCheckpoint = Pretrainer.Train(model, FakePartStage(records, [FakePartName], 1), align).LastCheckpoint!;
            }

            var aligned = ReadTensors(Path.Combine(alignCheckpoint, "model.safetensors"));
            Check(!aligned[FakePartTensor].Values.SequenceEqual(initial[FakePartTensor].Values)
                  && initial.Keys.Where(k => k != FakePartTensor).All(k => aligned[k].Values.SequenceEqual(initial[k].Values)), "the part trained alone; every decoder tensor as it was");
            float fromAligned;
            using (var model = PretrainedModel.Load(alignCheckpoint, new PretrainedOptions { Device = device }))
            using (var source = model.OpenPart(ChatParts.Audio, device))
            {
                fromAligned = FakePartLoss(model, source, records);
            }

            var stage = FakePartStage(records, [FakePartName], 1).Stages[0];
            var twoStages = FakePartStage(records, ["all"], 1) with { Stages = [stage with { Name = "align" }, stage with { Name = "joint", Trained = ["all"] }] };
            using (var model = PretrainedModel.Load(made, new PretrainedOptions { Device = device }))
            {
                var result = Pretrainer.Train(model, twoStages, Path.Combine(folder, "two"));
                Check(result.Losses.Count == 2 && MathF.Abs(result.Losses[1] - fromAligned) <= 1e-5f * MathF.Max(1, fromAligned),
                    $"the joint stage starts from the aligned part: its first loss {result.Losses[1]}, the aligned model's {fromAligned}");
            }

            using (var model = PretrainedModel.Load(made, new PretrainedOptions { Device = device }))
            {
                ThrowsNaming<ArgumentException>(() => Pretrainer.Train(model, FakePartStage(records, ["audio-tower"], 1), Path.Combine(folder, "unknown")),
                    "an unknown part", "'audio-tower'", FakePartName);
            }
        }
        finally
        {
            ModelParts.Unregister(ChatParts.Audio);
            Directory.Delete(folder, recursive: true);
        }
    }
}
