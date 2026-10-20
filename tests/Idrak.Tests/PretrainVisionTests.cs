// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Vision.Plugins.IdrakVision;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// Plan 19, phase B3: the pretraining loop with images: every parameter of a vision-language model, or its vision side alone.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PretrainVisionGroup =
    [
        ("pretrain vision: joint, every parameter of the tiny IdrakVision model trains on image records: PyTorch's loss and gradients at step 1, its losses and weights after three", PretrainVisionJoint),
        ("pretrain vision: align, only the tower and the projector train (the decoder stays): PyTorch's losses, gradients and weights", PretrainVisionAlign),
        ("pretrain vision: an align stage then a joint stage, stopped between them and resumed, gives the uninterrupted losses; text windows and image records mix; the result answers about an image", PretrainVisionStages),
    ];

    private static PretrainingOptions VisionStage(string name, IReadOnlyList<string> trained, int steps) => new()
    {
        Stages = [new PretrainingStage(name)
        {
            Trained = trained, Steps = steps, LearningRate = 0.05f, Schedule = "constant", WarmupFraction = 0,
            Sources = [new PretrainingSource(PretrainingSources.Records, [ScratchData("records.jsonl")]) { Format = "messages" }],
        }],
        MaxLength = 128, BatchTokens = 384, SequencesPerStep = 3, Optimizer = "sgd", Momentum = 0, WeightDecay = 0, MaxGradientNorm = 0,
        CudaGraphs = false, KeepCheckpoints = 0,
    };

    private static void CheckVisionRun(Device device, string kind, IReadOnlyList<string> trained, string folder)
    {
        var steps = JsonNode.Parse(File.ReadAllText(ScratchData("steps.json")))![kind]!;
        var expected = steps["losses"]!.AsArray().Select(l => (float)l!).ToArray();
        var initial = ReadTensors(ScratchData("tiny-idrak-vision/model.safetensors"));
        using (var model = LoadScratch(device))
        {
            var one = Pretrainer.Train(model, VisionStage(kind, trained, 1), Path.Combine(folder, $"{kind}-one"));
            Check(Math.Abs(one.Losses[0] - expected[0]) < 1e-4f, $"{kind}: step 1's loss {one.Losses[0]:F6}, PyTorch's {expected[0]:F6}");
            var after = ReadTensors(Path.Combine(one.LastCheckpoint!, "model.safetensors"));
            var grads = ReadTensors(ScratchData($"{kind}/grads-step1.safetensors"));
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

        using (var model = LoadScratch(device))
        {
            var three = Pretrainer.Train(model, VisionStage(kind, trained, 3), Path.Combine(folder, $"{kind}-three"));
            AssertClose(expected, [.. three.Losses], 1e-4f, $"{kind}: three steps' losses");
            var after = ReadTensors(Path.Combine(three.LastCheckpoint!, "model.safetensors"));
            foreach (var (name, (_, w)) in ReadTensors(ScratchData($"{kind}/weights-step3.safetensors")))
            {
                AssertClose(w, after[name].Values, 5e-5f, $"{kind}: {name} after three steps");
            }
        }
    }

    private static void PretrainVisionJoint(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("pretrain-vision-joint");
        try
        {
            CheckVisionRun(device, "joint", ["all"], folder);
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void PretrainVisionAlign(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("pretrain-vision-align");
        try
        {
            CheckVisionRun(device, "align", ["tower", "projector"], folder);
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void PretrainVisionStages(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("pretrain-vision-stages");
        try
        {
            var align = VisionStage("align", ["tower", "projector"], 2).Stages[0];
            var joint = VisionStage("joint", ["all"], 2).Stages[0] with
            {
                Sources = [align.Sources[0], new PretrainingSource(PretrainingSources.Tokens, [PretrainData("tokens.bin")]) { Weight = 2 }],
            };
            var options = VisionStage("x", ["all"], 1) with { Stages = [align, joint], SequencesPerStep = 4, Optimizer = "adamw", Seed = 3 };
            IReadOnlyList<float> whole;
            using (var model = LoadScratch(device))
            {
                whole = Pretrainer.Train(model, options, Path.Combine(folder, "whole")).Losses;
            }

            Check(whole.Count == 4 && whole.All(float.IsFinite), $"two stages of two steps: {string.Join(", ", whole)}");
            string run = Path.Combine(folder, "stopped");
            using (var model = LoadScratch(device))
            {
                using var stop = new CancellationTokenSource();
                Throws<OperationCanceledException>(() => Pretrainer.Train(model, options, run, null,
                    new SynchronousProgress<PretrainingProgress>(p => { if (p.Step == 2) { stop.Cancel(); } }), stop.Token), "stopped after the align stage");
            }

            var checkpoint = Pretrainer.ReadCheckpoint(Pretrainer.LatestCheckpoint(run)!);
            // Cancelled as the joint stage's first step began: the checkpoint is the run after the align stage (stage 1, none of its steps done).
            Check(checkpoint.Step == 2 && (checkpoint.Stage, checkpoint.StageStep) is (0, 2) or (1, 0), $"the checkpoint ends the align stage: stage {checkpoint.Stage}, step {checkpoint.StageStep}");
            var aligned = ReadTensors(Path.Combine(checkpoint.Folder, "model.safetensors"));
            var initial = ReadTensors(ScratchData("tiny-idrak-vision/model.safetensors"));
            Check(!aligned["vision_tower.post_layernorm.weight"].Values.SequenceEqual(initial["vision_tower.post_layernorm.weight"].Values)
                  && aligned["language_model.model.norm.weight"].Values.SequenceEqual(initial["language_model.model.norm.weight"].Values),
                "the checkpoint holds the trained tower and the untouched decoder");
            using (var model = PretrainedModel.Load(checkpoint.Folder, new PretrainedOptions { Device = device }))
            {
                var resumed = Pretrainer.Train(model, checkpoint.Options, run, checkpoint);
                Check(resumed.Losses.SequenceEqual(whole), $"resumed [{string.Join(", ", resumed.Losses)}] against [{string.Join(", ", whole)}]");
                using var encoder = model.CreateVisionEncoder(new VisionEncoderOptions { Device = device });
                var chat = new ChatGenerator(model.CreateChat(KeyValueFormat.Float32, 128).Generator, model.ChatTemplate)
                {
                    Images = new ChatImages(encoder, model.Vision!.PromptFormat, model.Vision.Attention),
                };
                var answer = chat.Chat(new ChatRequest([new ChatMessage("user", [ChatImage.FromFile(ScratchData("images/image.png")), new ChatText("Read the scan.")])],
                    Options: new GenerationOptions { Temperature = 0, NumPredict = 4 }));
                Check(answer.Done, "the trained model answers about an image");
            }
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }
}
