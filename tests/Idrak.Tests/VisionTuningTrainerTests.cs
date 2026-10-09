// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Gemma3Vision;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 12, phase 2: images through the tuner. The fixtures of phase 0 (tests/Idrak.Tests/data/vlm-tuning) run through
// FineTuner.Train with the family's vision side (TuningVision): the losses, adapters and projector of transformers' runs,
// the feature cache, checkpointed blocks with images, the saved adapter folder, and the refusals.
internal static partial class Tests
{
    // The fixture's records encoded by the library's own encoder with the family's vision side.
    private static List<TrainingSequence> TuningSequences(PretrainedModel model, JsonNode tuning, TuningVision vision)
    {
        var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Vision = vision };
        return [.. tuning["records"]!.AsArray().Select(r => encoder.Encode(TuningTranscript(r!), 4096)!)];
    }

    // The fixture's training settings: rank 2, alpha 4, q/k/v/o, SGD at its rate, no clipping, no schedule, every record in one batch.
    private static FineTuningOptions TuningOptions(JsonNode tuning, TuningVision vision, int epochs) => new()
    {
        Rank = (int)tuning["settings"]!["lora"]!["rank"]!,
        Alpha = (float)tuning["settings"]!["lora"]!["alpha"]!,
        Targets = ["q", "k", "v", "o"],
        Optimizer = ps => new Idrak.Abstraction.Training.Sgd(ps, (float)tuning["settings"]!["optimizer"]!["learning_rate"]!),
        Scheduler = FineTuningSchedules.Constant(),
        MaxGradientNorm = 0f,
        Epochs = epochs,
        BatchTokens = 4096,
        Vision = vision,
    };

    // The model with the fixture's initial adapters.
    private static (PretrainedModel Model, JsonNode Tuning) TuningModelWithAdapters(string family, Device device)
    {
        var (model, tuning) = LoadTuningModel(family, device);
        var lora = tuning["settings"]!["lora"]!;
        model.AddAdapters((int)lora["rank"]!, (float)lora["alpha"]!, ["q", "k", "v", "o"]);
        model.LoadAdapter(TuningData($"{family}/adapter-init"));
        return (model, tuning);
    }

    private static void VisionTuningTrainer(Device device)
    {
        foreach (string family in new[] { "gemma3", "llava" })
        {
            foreach (string run in new[] { "lora", "projector" })
            {
                VisionTuningTrainerRun(family, run, device);
            }
        }
    }

    private static void VisionTuningTrainerRun(string family, string run, Device device)
    {
        string what = $"{family} {run} through FineTuner.Train";
        var (model, tuning) = TuningModelWithAdapters(family, device);
        using var owned = model;
        string[] parts = run == "projector" ? [VisionTuningParts.Projector] : [];
        using var vision = TuningVision.Create(model, TuningImages.Parse(null), parts);
        var train = TuningSequences(model, tuning, vision);
        var expectedRun = tuning["runs"]![run]!;
        float[] expected = [.. expectedRun["losses"]!.AsArray().Select(n => (float)n!)];
        var losses = new List<float>();
        var lines = new List<string>();
        string folder = TempFolder();
        try
        {
            FineTuner.Train(model, train, null, TuningOptions(tuning, vision, expected.Length), folder,
                new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)), trace: lines.Add);
            AssertClose(expected, [.. losses], 1e-4f, $"{what}: the losses of {expected.Length} steps");

            // Two distinct images: the tower runs once for each; every later step (epoch) takes them from the cache.
            Check(vision.Encoded == 2 && vision.CacheHits == 2 * (expected.Length - 1),
                $"{what}: {vision.Encoded} images encoded, {vision.CacheHits} cache hits (expected 2 and {2 * (expected.Length - 1)})");
            Check(lines.Any(l => l.Contains("images (", StringComparison.Ordinal) && l.Contains("image tokens, packed with the others)", StringComparison.Ordinal))
                  && lines.Any(l => l.Contains($"image features: 2 from the cache, 0 encoded, 2 images in {(run == "projector" ? "1 pass," : "0 passes,")}", StringComparison.Ordinal)),
                $"{what}: the trace shows the images and the cache:\n{string.Join("\n", lines)}");

            // The saved folder: the adapters and the projector (modules_to_save) as transformers' after the last step.
            using (var saved = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors")))
            using (var adapters = SafeTensorsReader.Open(TuningData($"{family}/{run}/adapter-step{expected.Length}/adapter_model.safetensors")))
            {
                foreach (string key in adapters.Tensors.Keys)
                {
                    AssertClose(adapters.Read(key), saved.Read(key), 1e-5f, $"{what}: {key} after the steps");
                }

                int projectorTensors = 0;
                if (run == "projector")
                {
                    using var projector = SafeTensorsReader.Open(TuningData($"{family}/{run}/projector-step{expected.Length}.safetensors"));
                    foreach (string key in projector.Tensors.Keys)
                    {
                        Check(saved.Tensors[$"base_model.model.{key}"].Shape.SequenceEqual(projector.Tensors[key].Shape), $"{what}: {key} saved as stored");
                        AssertClose(projector.Read(key), saved.Read($"base_model.model.{key}"), 1e-5f, $"{what}: {key} after the steps");
                    }

                    projectorTensors = projector.Tensors.Count;
                }

                Check(saved.Tensors.Count == adapters.Tensors.Count + projectorTensors, $"{what}: {saved.Tensors.Count} tensors saved");
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "adapter_config.json")))!;
            Check(run == "projector" ? config["modules_to_save"]!.AsArray().Select(n => (string)n!).SequenceEqual(["multi_modal_projector"]) : config["modules_to_save"] is null,
                $"{what}: modules_to_save {config["modules_to_save"]?.ToJsonString()}");
            Check(TuningImages.Read(folder) == TuningImages.None, $"{what}: tuning_images.json saved");

            // Read back: LoadAdapter and MergeAdapter keep the trained projector, and the vision encoder takes it.
            if (run == "projector")
            {
                foreach (var reload in new Func<PretrainedModel>[]
                         {
                             () => { var m = PretrainedModel.Load(model.Folder, new PretrainedOptions { Device = device }); m.LoadAdapter(folder); return m; },
                             () => PretrainedModel.Load(model.Folder, new PretrainedOptions { Device = device, MergeAdapter = folder }),
                         })
                {
                    using var again = reload();
                    using var encoder = again.CreateVisionEncoder();
                    var exported = ((IVisionTuningPart)again.Vision!).Export(encoder, VisionTuningParts.Projector);
                    using var projector = SafeTensorsReader.Open(TuningData($"{family}/{run}/projector-step{expected.Length}.safetensors"));
                    Check(again.TrainedVisionTensors.Count == projector.Tensors.Count && exported.Count == projector.Tensors.Count, $"{what}: {again.TrainedVisionTensors.Count} trained vision tensors read back");
                    foreach (var (key, tensor) in exported)
                    {
                        AssertClose(projector.Read(key), tensor.ToArray(), 1e-5f, $"{what}: {key} in the reloaded encoder");
                        tensor.Dispose();
                    }
                }
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        Console.WriteLine($"    {what} on {device}: losses {string.Join(", ", losses.Select(l => l.ToString("F6")))} (transformers {string.Join(", ", expected.Select(l => l.ToString("F6")))})");
    }

    private static void VisionTuningCheckpointed(Device device)
    {
        // Through the public scope: the pass module by module, decoder blocks checkpointed, the scope closed before the
        // backward pass; against the stored pass (ImagePrefill.Forward). Gemma 3's image blocks attend both ways, so a
        // recompute without them would give other gradients.
        foreach (string family in new[] { "gemma3", "llava" })
        {
            var (model, tuning) = TuningModelWithAdapters(family, device);
            using var owned = model;
            var vision = model.Vision!;
            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            var record = tuning["records"]![2]!;
            int[] ids = JsonInts(record["input_ids"]);
            var features = record["images"]!.AsArray().Select(f => encoder.Encode([Idrak.Data.Abstractions.ImageCodecs.Decode(TestData($"vlm/{(string)f!}"))])[0].Features).ToList();
            var adapters = model.Network.Descendants().OfType<Linear>().Select(l => l.Lora).OfType<LoraAdapter>().ToList();
            float[][] Gradients(bool checkpointed)
            {
                foreach (var a in adapters)
                {
                    a.A.ReleaseGrad();
                    a.B.ReleaseGrad();
                }

                using var scope = new TensorScope();
                using var input = Ids(ids, device);
                var images = ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, features);
                Tensor logits;
                if (checkpointed)
                {
                    var modules = model.Network.ToList();
                    using (var imageScope = model.Network.Begin(input, images, vision.Attention))
                    {
                        var hidden = input;
                        for (int i = 0; i < modules.Count; i++)
                        {
                            hidden = i == imageScope.FirstBlock ? imageScope.Substitute(hidden) : hidden;
                            hidden = modules[i] is DecoderBlock ? modules[i].ForwardCheckpointed(hidden) : modules[i].Forward(hidden);
                        }

                        logits = hidden;
                    }
                }
                else
                {
                    logits = model.Network.Forward(input, images, vision.Attention);
                }

                var weights = Tensor.From([.. Enumerable.Range(0, logits.Size).Select(i => (float)Math.Sin(i * 0.37))], logits.Shape.ToArray(), device);
                (logits.LogSoftmax() * weights).Sum().Backward();                          // the image scope is closed by now
                return [.. adapters.SelectMany(a => new[] { a.A.Grad!.ToArray(), a.B.Grad!.ToArray() })];
            }

            var stored = Gradients(checkpointed: false);
            var recomputed = Gradients(checkpointed: true);
            int differ = Enumerable.Range(0, stored.Length).Count(i => !stored[i].SequenceEqual(recomputed[i]));
            Check(differ == 0, $"{family}: {differ} of {stored.Length} adapter gradients differ between checkpointed and stored blocks (largest {Enumerable.Range(0, stored.Length).Max(i => MaxDifference(stored[i], recomputed[i])):G3})");
            features.ForEach(f => f.Dispose());
        }

        // Through the tuner: one step with the projector trained, checkpointing on and off, gives the same adapters and projector.
        foreach (string family in new[] { "gemma3", "llava" })
        {
            float[][] Step(bool checkpointing)
            {
                var (model, tuning) = TuningModelWithAdapters(family, device);
                using var owned = model;
                using var vision = TuningVision.Create(model, parts: [VisionTuningParts.Projector]);
                var options = TuningOptions(tuning, vision, epochs: 1) with { Checkpointing = checkpointing };
                FineTuner.Train(model, TuningSequences(model, tuning, vision), null, options);
                return [.. model.Network.Descendants().OfType<Linear>().Select(l => l.Lora).OfType<LoraAdapter>().SelectMany(a => new[] { a.A.ToArray(), a.B.ToArray() }),
                        .. vision.Parameters.Select(p => p.ToArray())];
            }

            var plain = Step(checkpointing: false);
            var checkpointed = Step(checkpointing: true);
            int differ = Enumerable.Range(0, plain.Length).Count(i => !plain[i].SequenceEqual(checkpointed[i]));
            Check(plain.Length == checkpointed.Length && differ == 0, $"{family}: {differ} of {plain.Length} trained tensors differ after a step with and without checkpointing");
            Console.WriteLine($"    {family}: checkpointed and stored blocks with images give the same gradients ({plain.Length} trained tensors after a step)");
        }
    }

    private static void VisionTuningRefusals(Device device)
    {
        var (model, tuning) = LoadTuningModel("gemma3", device);
        using var owned = model;
        List<TrainingSequence> train;
        using (var vision = TuningVision.Create(model))
        {
            train = TuningSequences(model, tuning, vision);
            Check(Failure<ArgumentException>(() => TuningVision.Create(model, TuningImages.Parse(null, ["no_such_option=1"]))).Message.Contains("no_such_option", StringComparison.Ordinal),
                "an unknown vision option is refused");
        }

        // The family unregistered after loading: the tuner says so with the registry's message, before any step (no adapter is added).
        VisionFamilies.Unregister(Gemma3VisionFamily.Architecture);
        try
        {
            var options = new FineTuningOptions { Rank = 2, Alpha = 4, Targets = ["q"], Epochs = 1 };
            foreach (var (how, action) in new (string, Action)[]
                     {
                         ("TuningVision.Create", () => TuningVision.Create(model).Dispose()),
                         ("FineTuner.Train", () => FineTuner.Train(model, train, null, options)),
                         ("FineTuner.Evaluate", () => FineTuner.Evaluate(model, train)),
                     })
            {
                string message = Failure<NotSupportedException>(action).Message;
                Check(message.Contains($"Vision family '{Gemma3VisionFamily.Architecture}' is not registered", StringComparison.Ordinal)
                      && message.Contains("VisionFamilies.Register", StringComparison.Ordinal), $"{how}: {message}");
            }

            Check(!model.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null), "nothing ran before the refusal");
        }
        finally
        {
            RegisterGemma3Vision();
        }

        // A text model given sequences with images, and a text-only encoder given a transcript with images.
        using (var text = PretrainedModel.Load(TestData("gguf/tiny-llama-hf"), new PretrainedOptions { Device = device }))
        {
            string message = Failure<InvalidOperationException>(() => FineTuner.Train(text, train, null, new FineTuningOptions { Epochs = 1 })).Message;
            Check(message.Contains("no vision part", StringComparison.Ordinal), $"a text model given images: {message}");
        }

        var transcript = TuningTranscript(tuning["records"]![0]!);
        string refused = Failure<InvalidOperationException>(() => new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!).Encode(transcript, 4096)).Message;
        Check(refused.Contains("ChatTranscriptEncoder.Vision", StringComparison.Ordinal), $"a text-only encoder given images: {refused}");
    }
}
