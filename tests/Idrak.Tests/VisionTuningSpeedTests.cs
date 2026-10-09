// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 12, phase 6: speed and packing. Sequences with images packed into rows (each sequence its own key ranges by the
// family's rule), the recordable image inputs of a graphed step (checked on the CPU without recording: the same pass from
// fixed buffers), the tower trained in the step (gradients, checkpointing, saved and read back), a trained projector over
// a bfloat16 tower (the tower keeps its precision), and a batch's images through the vision side in one pass.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionTuningPhase6Group =
    [
        ("vision tuning packing: sequences with images packed into rows (with padding) give each sequence the logits and adapter gradients it gets alone, for Gemma 3 (image blocks) and LLaVA (causal); checkpointed packed blocks give the stored gradients bit for bit; the recordable inputs (ImagePrefillInputs, fixed buffers loaded again for another layout) give the same pass", VisionTuningPacking),
        ("vision tuning packing: FineTuner.Train with packing and with padded rows gives the same losses and adapters (and projector) on the phase-0 fixtures, both within 1e-4 of transformers' losses", VisionTuningPackedTrainer),
        ("vision tuning graph: a forward and backward pass with images recorded as a graph through ImagePrefillInputs and replayed for another batch (features, rows and key ranges copied in) gives the ordinary pass's loss and adapter gradients, for Gemma 3 and LLaVA; a device without graphs runs the same pass from the fixed inputs", VisionTuningGraphInputs),
        ("vision tuning tower: Gemma 3's tower exports every SigLIP tensor as the checkpoint stores it (the fused q, k, v split back) and imports it again; gradients reach the tower's weights through the decoder (finite differences); checkpointed tower blocks give the stored gradients bit for bit", VisionTuningTowerGradients),
        ("vision tuning tower: FineTuner.Train trains the tower in the step (pixel values cached, the tower and projector run each step), checkpointing on and off give the same weights, and the tower saved as modules_to_save is read back by LoadAdapter into a new encoder", VisionTuningTowerTrainer),
        ("vision tuning memory: a trained projector over a bfloat16 checkpoint keeps the tower in bfloat16 (its weight bytes are the bfloat16 tower's plus the float32 projector's), not the whole encoder in float32", VisionTuningTrainedProjectorMemory),
        ("vision tuning speed: a batch's images through the frozen tower and the projector in one pass give each image's features alone; the time of one pass against one image at a time", VisionTuningBatchedImages),
    ];

    // The fixture's records of a family: their ids and each image's features from the encoder.
    private static List<(int[] Ids, List<Tensor> Features)> TuningRecordFeatures(JsonNode tuning, IVisionEncoder encoder) =>
        [.. tuning["records"]!.AsArray().Select(r => (JsonInts(r!["input_ids"]),
            r["images"]!.AsArray().Select(f => encoder.Encode([ImageCodecs.Decode(TestData($"vlm/{(string)f!}"))])[0].Features).ToList()))];

    // The decoder module by module from an image scope, decoder blocks checkpointed or not.
    private static Tensor ScopedPass(PretrainedModel model, Tensor tokens, ImagePrefillScope scope, bool checkpointed)
    {
        var modules = model.Network.ToList();
        var hidden = tokens;
        for (int i = 0; i < modules.Count; i++)
        {
            hidden = i == scope.FirstBlock ? scope.Substitute(hidden) : hidden;
            hidden = checkpointed && modules[i] is DecoderBlock ? modules[i].ForwardCheckpointed(hidden) : modules[i].Forward(hidden);
        }

        return hidden;
    }

    private static List<LoraAdapter> Adapters(PretrainedModel model) => [.. model.Network.Descendants().OfType<Linear>().Select(l => l.Lora).OfType<LoraAdapter>()];

    private static float[][] AdapterGradients(PretrainedModel model) => [.. Adapters(model).SelectMany(a => new[] { a.A.Grad!.ToArray(), a.B.Grad!.ToArray() })];

    private static void ReleaseAdapterGradients(PretrainedModel model)
    {
        foreach (var a in Adapters(model))
        {
            a.A.ReleaseGrad();
            a.B.ReleaseGrad();
        }
    }

    private static void VisionTuningPacking(Device device)
    {
        foreach (string family in new[] { "gemma3", "llava" })
        {
            var (model, tuning) = TuningModelWithAdapters(family, device);
            using var owned = model;
            var vision = model.Vision!;
            if (vision.Attention.Causal && !PackedSequences.Supports(model.Network))
            {
                Console.WriteLine($"    {family}: {device} has no packed causal attention for this model; its packed rows are not checked here");
                continue;
            }

            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            var records = TuningRecordFeatures(tuning, encoder);
            int imageToken = vision.PromptFormat.ImageToken;
            float close = device.Type == DeviceType.Cpu ? 2e-5f : 2e-3f;                   // a GPU sums in its own order (and may round to bfloat16)

            // Each record's loss weights: one per position and vocabulary entry.
            int vocabulary;
            using (Autograd.NoGrad())
            using (var probe = model.Network.Forward(Ids(records[0].Ids, device), ImagePrefill.Locate(records[0].Ids, imageToken, records[0].Features), vision.Attention))
            {
                vocabulary = probe.Shape[^1];
            }

            var coefficients = records.Select((r, k) => Weights(r.Ids.Length * vocabulary, 100 + k)).ToList();

            // Alone: each record its own [1, n] pass, the gradients summed.
            ReleaseAdapterGradients(model);
            var aloneLogits = new List<float[]>();
            using (var scope = new TensorScope())
            {
                foreach (var (r, k) in records.Select((r, k) => (r, k)))
                {
                    var logits = model.Network.Forward(Ids(r.Ids, device), ImagePrefill.Locate(r.Ids, imageToken, r.Features), vision.Attention);
                    (logits * Tensor.From(coefficients[k], logits.Shape.ToArray(), device)).Sum().Backward();
                    aloneLogits.Add(logits.ToArray());
                }
            }

            var aloneGradients = AdapterGradients(model);

            // Packed: rows of records one after another, the rest of each row padding (weight 0).
            (List<float[]> Logits, float[][] Gradients) Packed(int[][] rows, int length, Func<Tensor, IReadOnlyList<PromptImage>, PackedSequences, Tensor> pass)
            {
                ReleaseAdapterGradients(model);
                var values = new float[rows.Length * length];
                var weights = new float[rows.Length * length * vocabulary];
                var images = new List<PromptImage>();
                for (int row = 0; row < rows.Length; row++)
                {
                    int offset = 0;
                    foreach (int k in rows[row])
                    {
                        var r = records[k];
                        for (int t = 0; t < r.Ids.Length; t++)
                        {
                            values[row * length + offset + t] = r.Ids[t];
                        }

                        coefficients[k].CopyTo(weights, (row * length + offset) * vocabulary);
                        images.AddRange(ImagePrefill.Locate(r.Ids, imageToken, r.Features, sequence: row).Select(p => p with { Position = p.Position + offset }));
                        offset += r.Ids.Length;
                    }
                }

                using var scope = new TensorScope();
                using var packing = PackedSequences.Create([.. rows.Select(r => r.Select(k => records[k].Ids.Length).ToArray())], length, device);
                float[] all;
                using (packing.Use())
                {
                    var tokens = Tensor.From(values, [rows.Length, length], device);
                    var logits = pass(tokens, images, packing);
                    (logits * Tensor.From(weights, logits.Shape.ToArray(), device)).Sum().Backward();
                    all = logits.ToArray();
                }

                var perRecord = new List<float[]>(new float[records.Count][]);
                for (int row = 0; row < rows.Length; row++)
                {
                    int offset = 0;
                    foreach (int k in rows[row])
                    {
                        perRecord[k] = all[((row * length + offset) * vocabulary)..((row * length + offset + records[k].Ids.Length) * vocabulary)];
                        offset += records[k].Ids.Length;
                    }
                }

                return (perRecord, AdapterGradients(model));
            }

            int[][] layout = [[0, 2], [1]];
            int width = layout.Max(r => r.Sum(k => records[k].Ids.Length)) + 5;
            var forward = Packed(layout, width, (tokens, images, _) => model.Network.Forward(tokens, images, vision.Attention));
            for (int k = 0; k < records.Count; k++)
            {
                AssertClose(aloneLogits[k], forward.Logits[k], close, $"{family}: record {k}'s logits packed against alone");
            }

            float largest = 0f;
            for (int i = 0; i < aloneGradients.Length; i++)
            {
                float scale = Math.Max(1e-3f, aloneGradients[i].Max(MathF.Abs));
                AssertClose(aloneGradients[i], forward.Gradients[i], close * Math.Max(1f, scale), $"{family}: adapter gradient {i} packed against alone");
                largest = Math.Max(largest, MaxDifference(aloneGradients[i], forward.Gradients[i]));
            }

            // Module by module, checkpointed, the scope closed before the backward pass: the stored packed pass's gradients.
            var stored = Packed(layout, width, (tokens, images, _) =>
            {
                using var scope = model.Network.Begin(tokens, images, vision.Attention);
                return ScopedPass(model, tokens, scope, checkpointed: false);
            });
            var checkpointed = Packed(layout, width, (tokens, images, _) =>
            {
                using var scope = model.Network.Begin(tokens, images, vision.Attention);
                return ScopedPass(model, tokens, scope, checkpointed: true);
            });
            int differ = Enumerable.Range(0, stored.Gradients.Length).Count(i => !stored.Gradients[i].SequenceEqual(checkpointed.Gradients[i]));
            Check(differ == 0, $"{family}: {differ} of {stored.Gradients.Length} adapter gradients differ between checkpointed and stored packed blocks");
            Check(Enumerable.Range(0, records.Count).All(k => stored.Logits[k].SequenceEqual(forward.Logits[k])), $"{family}: the scoped packed pass gives ImagePrefill.Forward's logits");

            // The recordable inputs: fixed buffers for up to every image token, loaded for this layout, then again for another.
            int tokensAll = records.Sum(r => r.Features.Sum(f => f.Shape[0]));
            using var inputs = ImagePrefillInputs.Create(model.Network, layout.Length, width, tokensAll, vision.Attention);
            Check(inputs.Fits(layout.Length, width, tokensAll) && !inputs.Fits(layout.Length, width, tokensAll + 1) && !inputs.Fits(layout.Length + 1, width, 1), "the inputs fit their shape");
            var fixedPass = Packed(layout, width, (tokens, images, packing) =>
            {
                inputs.Load(images, packing);
                using var scope = inputs.Begin();
                return ScopedPass(model, tokens, scope, checkpointed: true);
            });
            Check(Enumerable.Range(0, records.Count).All(k => fixedPass.Logits[k].SequenceEqual(forward.Logits[k]))
                  && Enumerable.Range(0, stored.Gradients.Length).All(i => fixedPass.Gradients[i].SequenceEqual(stored.Gradients[i])),
                $"{family}: the fixed inputs give the scoped pass's logits and gradients");
            int[][] other = [[1, 0], [2]];
            var again = Packed(other, width, (tokens, images, packing) =>
            {
                inputs.Load(images, packing);
                using var scope = inputs.Begin();
                return ScopedPass(model, tokens, scope, checkpointed: false);
            });
            for (int k = 0; k < records.Count; k++)
            {
                AssertClose(aloneLogits[k], again.Logits[k], close, $"{family}: record {k}'s logits from the inputs loaded again for another layout");
            }

            var tooMany = Enumerable.Range(0, 3).SelectMany(_ => records.SelectMany(r => r.Features)).Select((f, i) => new PromptImage(i * 40, f) { Sequence = 0 }).ToList();
            Check(Failure<ArgumentException>(() => inputs.Load(tooMany)).Message.Contains("image tokens", StringComparison.Ordinal), $"{family}: more image tokens than the inputs hold are refused");

            // An image across two packed sequences is refused.
            var crossing = ImagePrefill.Locate(records[0].Ids, imageToken, records[0].Features).Select(p => p with { Position = records[0].Ids.Length - 1 }).Take(1).ToList();
            int n0 = records[0].Ids.Length;
            using (var packing = PackedSequences.Create([[n0, 40]], n0 + 64, device))
            using (packing.Use())
            using (var tokens = Tensor.From(new float[n0 + 64], [1, n0 + 64], device))
            {
                Check(Failure<ArgumentException>(() => model.Network.Begin(tokens, crossing, vision.Attention).Dispose()).Message.Contains("packed sequences", StringComparison.Ordinal),
                    $"{family}: an image across two packed sequences is refused");
            }

            records.ForEach(r => r.Features.ForEach(f => f.Dispose()));
            Console.WriteLine($"    {family}: {records.Count} records packed in {layout.Length} rows of {width}: logits and gradients as alone (largest gradient difference {largest:G3}); "
                              + "checkpointed and fixed-input passes bit for bit");
        }
    }

    private static void VisionTuningPackedTrainer(Device device)
    {
        foreach (string family in new[] { "gemma3", "llava" })
        {
            foreach (string run in new[] { "lora", "projector" })
            {
                (List<float> Losses, float[][] Trained, List<string> Trace) Train(bool packing)
                {
                    var (model, tuning) = TuningModelWithAdapters(family, device);
                    using var owned = model;
                    using var vision = TuningVision.Create(model, parts: run == "projector" ? [VisionTuningParts.Projector] : []);
                    var losses = new List<float>();
                    var lines = new List<string>();
                    FineTuner.Train(model, TuningSequences(model, tuning, vision), null, TuningOptions(tuning, vision, epochs: 3) with { Packing = packing }, null,
                        new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)), trace: lines.Add);
                    return (losses, [.. Adapters(model).SelectMany(a => new[] { a.A.ToArray(), a.B.ToArray() }), .. vision.Parameters.Select(p => p.ToArray())], lines);
                }

                var (model0, tuning0) = LoadTuningModel(family, device);
                bool supported = PackedSequences.Supports(model0.Network);
                model0.Dispose();
                if (!supported)
                {
                    Console.WriteLine($"    {family} {run}: {device} does not pack this model's sequences (FineTuner pads them); not compared");
                    continue;
                }
                float[] expected = [.. tuning0["runs"]![run]!["losses"]!.AsArray().Select(n => (float)n!)];
                var packed = Train(packing: true);
                var padded = Train(packing: false);
                Check(packed.Trace.Any(l => l.Contains("packed in", StringComparison.Ordinal)) && padded.Trace.Any(l => l.Contains("3 sequences × ", StringComparison.Ordinal)),
                    $"{family} {run}: one run packed, the other padded:\n{string.Join("\n", packed.Trace.Concat(padded.Trace))}");
                AssertClose(expected, [.. packed.Losses], 1e-4f, $"{family} {run}: packed losses against transformers'");
                AssertClose(expected, [.. padded.Losses], 1e-4f, $"{family} {run}: padded losses against transformers'");
                AssertClose([.. padded.Losses], [.. packed.Losses], 1e-5f, $"{family} {run}: packed against padded losses");
                for (int i = 0; i < padded.Trained.Length; i++)
                {
                    AssertClose(padded.Trained[i], packed.Trained[i], 1e-5f, $"{family} {run}: trained tensor {i} packed against padded");
                }

                Console.WriteLine($"    {family} {run}: packed {string.Join(", ", packed.Losses.Select(l => l.ToString("F6")))}; padded {string.Join(", ", padded.Losses.Select(l => l.ToString("F6")))}");
            }
        }
    }

    private static void VisionTuningGraphInputs(Device device)
    {
        var backend = device.Backend;
        bool graphs = backend.SupportsGraphs;
        float tolerance = device.Type == DeviceType.Cpu ? 1e-6f : 1e-3f;
        foreach (string family in new[] { "gemma3", "llava" })
        {
            var (model, tuning) = TuningModelWithAdapters(family, device);
            using var owned = model;
            var vision = model.Vision!;
            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            var records = TuningRecordFeatures(tuning, encoder);
            int imageToken = vision.PromptFormat.ImageToken, rows = records.Count, width = records.Max(r => r.Ids.Length) + 3;
            int vocabulary;
            using (Autograd.NoGrad())
            using (var probe = model.Network.Forward(Ids(records[0].Ids, device), ImagePrefill.Locate(records[0].Ids, imageToken, records[0].Features), vision.Attention))
            {
                vocabulary = probe.Shape[^1];
            }

            var coefficients = records.Select((r, k) => Weights(r.Ids.Length * vocabulary, 200 + k)).ToList();

            // A batch of padded rows: record order[row] in each row, its loss weights there, 0 elsewhere.
            (float[] Tokens, float[] Weights, List<PromptImage> Images) Batch(int[] order)
            {
                var values = new float[rows * width];
                var weights = new float[rows * width * vocabulary];
                var images = new List<PromptImage>();
                for (int row = 0; row < rows; row++)
                {
                    var r = records[order[row]];
                    for (int t = 0; t < r.Ids.Length; t++)
                    {
                        values[row * width + t] = r.Ids[t];
                    }

                    coefficients[order[row]].CopyTo(weights, row * width * vocabulary);
                    images.AddRange(ImagePrefill.Locate(r.Ids, imageToken, r.Features, sequence: row));
                }

                return (values, weights, images);
            }

            // The ordinary pass (the image blocks uploaded as it runs), fresh gradients.
            (float Loss, float[][] Gradients) Ordinary(int[] order)
            {
                ReleaseAdapterGradients(model);
                var (values, weights, images) = Batch(order);
                using var scope = new TensorScope();
                var tokens = Tensor.From(values, [rows, width], device);
                Tensor logits;
                using (var imageScope = model.Network.Begin(tokens, images, vision.Attention))
                {
                    logits = ScopedPass(model, tokens, imageScope, checkpointed: false);
                }

                var loss = (logits * Tensor.From(weights, [rows, width, vocabulary], device)).Sum();
                loss.Backward();
                return (loss.Item(), AdapterGradients(model));
            }

            int[] first = [0, 1, 2], second = [2, 0, 1];
            var expectedFirst = Ordinary(first);
            var expectedSecond = Ordinary(second);                        // the adapters keep these gradient buffers from here on

            // The recorded pass reads fixed buffers only: the tokens, the loss weights and the images' inputs.
            using var tokensBuffer = Tensor.Persistent(new float[rows * width], [rows, width], device, requiresGrad: false);
            using var weightsBuffer = Tensor.Persistent(new float[rows * width * vocabulary], [rows, width, vocabulary], device, requiresGrad: false);
            using var lossBuffer = Tensor.Persistent([0f], [1], device, requiresGrad: false);
            using var inputs = ImagePrefillInputs.Create(model.Network, rows, width, records.Sum(r => r.Features.Sum(f => f.Shape[0])), vision.Attention);
            void Load(int[] order)
            {
                var (values, weights, images) = Batch(order);
                tokensBuffer.Load(values);
                weightsBuffer.Load(weights);
                inputs.Load(images);
            }

            void Pass()
            {
                foreach (var a in Adapters(model))
                {
                    a.A.ZeroGrad();
                    a.B.ZeroGrad();
                }

                using var scope = new TensorScope();
                Tensor logits;
                using (var imageScope = inputs.Begin())
                {
                    logits = ScopedPass(model, tokensBuffer, imageScope, checkpointed: false);
                }

                var loss = (logits * weightsBuffer).Sum();
                loss.Backward();
                backend.Copy(loss.Storage, lossBuffer.Storage, 1);
            }

            Load(first);
            IntPtr executable = IntPtr.Zero, graph = IntPtr.Zero;
            List<Idrak.Abstraction.Devices.Storage> kept = [];
            if (graphs)
            {
                device.Synchronize();
                backend.BeginCapture();
                try
                {
                    Pass();
                    (executable, graph, kept) = backend.EndCapture();
                }
                catch
                {
                    foreach (var storage in backend.AbortCapture())
                    {
                        storage.Release();
                    }

                    throw;
                }
            }

            try
            {
                void Run()
                {
                    if (graphs)
                    {
                        backend.ReplayGraph(executable);
                    }
                    else
                    {
                        Pass();
                    }
                }

                foreach (var (order, expected, which) in new[] { (first, expectedFirst, "the recorded batch"), (second, expectedSecond, "another batch loaded") })
                {
                    Load(order);
                    Run();
                    AssertClose([expected.Loss], [lossBuffer.Item()], tolerance, $"{family}: {which}'s loss {(graphs ? "replayed" : "from the fixed inputs")}");
                    var gradients = AdapterGradients(model);
                    for (int i = 0; i < gradients.Length; i++)
                    {
                        AssertClose(expected.Gradients[i], gradients[i], tolerance * Math.Max(1f, expected.Gradients[i].Max(MathF.Abs)), $"{family}: {which}'s adapter gradient {i}");
                    }
                }
            }
            finally
            {
                if (graphs)
                {
                    device.Synchronize();
                    backend.DestroyGraph(executable, graph);
                    kept.ForEach(storage => storage.Release());
                }
            }

            records.ForEach(r => r.Features.ForEach(f => f.Dispose()));
            Console.WriteLine($"    {family}: {(graphs ? "recorded once, replayed for two batches" : "no graphs on this device: the pass from the fixed inputs")}, losses {expectedFirst.Loss:F4} and {expectedSecond.Loss:F4} as the ordinary pass");
        }
    }

    private static void VisionTuningTowerGradients(Device device)
    {
        // Export: every SigLIP tensor under its checkpoint name, the checkpoint's values exactly; Import puts changed values back.
        var (model, tuning) = TuningModelWithAdapters("gemma3", device);
        using var owned = model;
        var vision = (Gemma3Vision)model.Vision!;
        var part = (IVisionTuningPart)vision;
        using var encoder = (Gemma3ImageEncoder)vision.CreateEncoder(new VisionEncoderOptions { Device = device, TrainedParts = [VisionTuningParts.Tower] });
        Check(!encoder.Encoder.Buffers().Any(), "TrainedParts = [tower] builds the tower with float32 weights");
        var exported = part.Export(encoder, VisionTuningParts.Tower);
        var towerNames = vision.Tensors.Where(t => t.Key.StartsWith("vision.", StringComparison.Ordinal)).Select(t => t.Value.Stored).ToHashSet(StringComparer.Ordinal);
        Check(exported.Count == towerNames.Count && exported.Keys.All(towerNames.Contains), $"the tower exports {exported.Count} of its {towerNames.Count} tensors");
        using (var reader = SafeTensorsReader.Open(model.Folder))
        {
            foreach (var (name, tensor) in exported)
            {
                Check(tensor.Shape.SequenceEqual(reader.Tensors[name].Shape) && reader.Read(name).SequenceEqual(tensor.ToArray()), $"{name}: exported as the checkpoint stores it");
            }
        }

        Check(part.PartsOf(exported.Keys).SequenceEqual([VisionTuningParts.Tower]) && part.PartsOf(part.Export(encoder, VisionTuningParts.Projector).Keys).SequenceEqual([VisionTuningParts.Projector]),
            "PartsOf tells the tower's tensors from the projector's");
        var changed = exported.ToDictionary(p => p.Key, p => Tensor.From([.. p.Value.ToArray().Select((v, i) => v * 1.5f + 0.01f * (i % 7))], p.Value.Shape.ToArray(), Device.Cpu), StringComparer.Ordinal);
        using (var other = vision.CreateEncoder(new VisionEncoderOptions { Device = device, TrainedParts = [VisionTuningParts.Tower] }))
        {
            Check(part.Import(other, changed).Count == changed.Count, "every tower tensor imported");
            foreach (var (name, tensor) in part.Export(other, VisionTuningParts.Tower))
            {
                Check(tensor.ToArray().SequenceEqual(changed[name].ToArray()), $"{name}: imported and exported again");
            }
        }

        using (var packed = vision.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.BFloat16 }))
        {
            string refused = Failure<InvalidOperationException>(() => part.Import(packed, changed)).Message;
            Check(refused.Contains("TrainedParts", StringComparison.Ordinal), $"a bfloat16 tower takes no trained values: {refused}");
        }

        // Gradients through the decoder: record 2's two images, the tower then the projector, the decoder's log-probabilities weighted.
        var record = tuning["records"]![2]!;
        int[] ids = JsonInts(record["input_ids"]);
        var stages = (IVisionEncoderStages)encoder;
        var pixels = record["images"]!.AsArray().Select(f => stages.PixelValues(ImageCodecs.Decode(TestData($"vlm/{(string)f!}")))).ToList();
        using var joined = Tensor.Concat([.. pixels.Select(p => Tensor.From(p.ToArray(), p.Shape.ToArray(), device))], 0);
        var parameters = part.Parameters(encoder, VisionTuningParts.Tower);
        foreach (var p in encoder.Parameters())
        {
            p.RequiresGrad = false;
        }

        foreach (var p in parameters)
        {
            p.RequiresGrad = true;
        }

        float[]? weights = null;
        float Loss(bool checkpointed)
        {
            using var scope = new TensorScope();
            using var blocks = checkpointed ? ActivationMemory.CheckpointBlocks() : (ActivationMemory.Scope?)null;
            var features = part.Features(encoder, part.Tower(encoder, joined));               // [images, tokens, width]
            var list = Enumerable.Range(0, features.Shape[0]).Select(i => features.Narrow(0, i, 1).Reshape(features.Shape[1], features.Shape[2])).ToList();
            var logits = model.Network.Forward(Ids(ids, device), ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, list), vision.Attention);
            weights ??= Weights(logits.Size, 7);
            var loss = (logits.LogSoftmax() * Tensor.From(weights, logits.Shape.ToArray(), device)).Sum();
            if (Autograd.IsEnabled)
            {
                loss.Backward();
            }

            return loss.Item();
        }

        float[][] Gradients(bool checkpointed)
        {
            foreach (var p in parameters)
            {
                p.ReleaseGrad();
            }

            Loss(checkpointed);
            return [.. parameters.Select(p => p.Grad!.ToArray())];
        }

        var storedGradients = Gradients(checkpointed: false);
        var checkpointedGradients = Gradients(checkpointed: true);
        int differ = Enumerable.Range(0, storedGradients.Length).Count(i => !storedGradients[i].SequenceEqual(checkpointedGradients[i]));
        Check(differ == 0, $"{differ} of {storedGradients.Length} tower gradients differ between checkpointed and stored blocks");
        Check(storedGradients.Count(g => g.Any(v => v != 0f) && g.All(float.IsFinite)) >= parameters.Count / 2, "gradients reach the tower's weights");

        // Finite differences on the largest gradient of a few tower tensors: the first (the patch convolution), one in the
        // middle and the last of its matrices.
        var matrices = parameters.Where(p => p.Rank >= 2).ToList();
        var probed = new[] { matrices[0], matrices[matrices.Count / 2], matrices[^1] };
        foreach (var p in probed)
        {
            int index = parameters.ToList().FindIndex(x => ReferenceEquals(x, p));
            Check(index >= 0, "a probed tensor is a tower parameter");
            float[] values = p.ToArray(), gradient = storedGradients[index];
            int at = Enumerable.Range(0, gradient.Length).MaxBy(i => MathF.Abs(gradient[i]));
            const float Step = 1e-2f;
            float original = values[at];
            float Shifted(float by)
            {
                values[at] = original + by;
                p.Load(values);
                using var noGrad = Autograd.NoGrad();
                return Loss(checkpointed: false);
            }

            float numeric = (Shifted(Step) - Shifted(-Step)) / (2 * Step);
            values[at] = original;
            p.Load(values);
            Check(MathF.Abs(numeric - gradient[at]) <= 2e-2f * MathF.Max(1f, MathF.Abs(gradient[at])),
                $"d loss / d {Tensor.FormatShape(p.Shape)}[{at}] is {gradient[at]}, finite differences {numeric}");
            Console.WriteLine($"    tower {Tensor.FormatShape(p.Shape)}[{at}]: gradient {gradient[at]:G5}, finite differences {numeric:G5}");
        }

        pixels.ForEach(p => p.Dispose());
        foreach (var t in exported.Values.Concat(changed.Values))
        {
            t.Dispose();
        }
    }

    private static void VisionTuningTowerTrainer(Device device)
    {
        string folder = TempFolder();
        try
        {
            var (model, tuning) = TuningModelWithAdapters("gemma3", device);
            using var owned = model;
            using var vision = TuningVision.Create(model, parts: [VisionTuningParts.Tower]);
            Check(vision.TrainsTower && vision.CachedStage == FeatureCacheKey.PixelsStage && vision.Parameters.Count > 4, $"the tower trains: {vision}");
            var before = vision.Parameters.Select(p => p.ToArray()).ToList();
            var losses = new List<float>();
            var lines = new List<string>();
            FineTuner.Train(model, TuningSequences(model, tuning, vision), null, TuningOptions(tuning, vision, epochs: 3), folder,
                new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)), trace: lines.Add);
            Check(losses.Count == 3 && losses.All(float.IsFinite) && losses[2] < losses[0], $"losses {string.Join(", ", losses)}");
            Check(vision.Parameters.Select((p, i) => !p.ToArray().SequenceEqual(before[i])).Count(c => c) >= vision.Parameters.Count / 2, "the tower's weights changed");
            Check(lines.Any(l => l.Contains("image features: 0 from the cache, 2 encoded, 2 images in 2 passes", StringComparison.Ordinal))
                  && lines.Count(l => l.Contains("image features: 2 from the cache, 0 encoded, 2 images in 2 passes", StringComparison.Ordinal)) == 2,
                $"pixel values cached, the tower and the projector one pass each per step:\n{string.Join("\n", lines)}");

            // Saved as modules_to_save under the checkpoint's names; read back into a new model's encoder.
            var part = (IVisionTuningPart)model.Vision!;
            var exported = part.Export(vision.Encoder, VisionTuningParts.Tower);
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "adapter_config.json")))!;
            Check(config["modules_to_save"]!.AsArray().Count > 0, $"modules_to_save {config["modules_to_save"]?.ToJsonString()}");
            using (var saved = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors")))
            {
                foreach (var (name, tensor) in exported)
                {
                    Check(saved.Read($"base_model.model.{name}").SequenceEqual(tensor.ToArray()), $"{name} saved");
                }
            }

            using var again = PretrainedModel.Load(model.Folder, new PretrainedOptions { Device = device });
            again.LoadAdapter(folder);
            Check(again.TrainedVisionTensors.Count == exported.Count, $"{again.TrainedVisionTensors.Count} trained vision tensors read back");
            using var encoder = (Gemma3ImageEncoder)again.CreateVisionEncoder();
            foreach (var (name, tensor) in ((IVisionTuningPart)again.Vision!).Export(encoder, VisionTuningParts.Tower))
            {
                Check(tensor.ToArray().SequenceEqual(exported[name].ToArray()), $"{name} in the reloaded encoder");
                tensor.Dispose();
            }

            foreach (var t in exported.Values)
            {
                t.Dispose();
            }

            Console.WriteLine($"    tower trained in the step: losses {string.Join(", ", losses.Select(l => l.ToString("F6")))}; {exported.Count} tensors saved and read back");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        // One step with the tower and the projector trained, checkpointing on and off: the same weights after it.
        float[][] Step(bool checkpointing)
        {
            var (model, tuning) = TuningModelWithAdapters("gemma3", device);
            using var owned = model;
            using var vision = TuningVision.Create(model, parts: [VisionTuningParts.Projector, VisionTuningParts.Tower]);
            FineTuner.Train(model, TuningSequences(model, tuning, vision), null, TuningOptions(tuning, vision, epochs: 1) with { Checkpointing = checkpointing });
            return [.. Adapters(model).SelectMany(a => new[] { a.A.ToArray(), a.B.ToArray() }), .. vision.Parameters.Select(p => p.ToArray())];
        }

        var plain = Step(checkpointing: false);
        var checkpointed = Step(checkpointing: true);
        int differ = Enumerable.Range(0, plain.Length).Count(i => !plain[i].SequenceEqual(checkpointed[i]));
        Check(plain.Length == checkpointed.Length && differ == 0, $"{differ} of {plain.Length} trained tensors differ after a step with and without checkpointing (tower and projector trained)");
        Console.WriteLine($"    tower and projector: checkpointed and stored steps give the same {plain.Length} trained tensors");
    }

    // The bytes an encoder's weights take: float32 parameters, and bfloat16 projections at 2 bytes a value.
    private static long WeightBytes(Module module) =>
        module.Parameters().Sum(p => 4L * p.Size) + module.Descendants().OfType<Linear>().Sum(l => l.BFloat16?.Bytes ?? 0L);

    private static void VisionTuningTrainedProjectorMemory(Device device)
    {
        RegisterGemma3Vision();
        string folder = BFloat16Copy(TestData("vlm/tiny-gemma3"));
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var vision = (Gemma3Vision)model.Vision!;
            using (var trained = vision.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.Float32 }))
            {
                Check(model.KeepTrainedVision(trained, [VisionTuningParts.Projector]) == 2, "the projector kept as trained");
            }

            using var encoder = (Gemma3ImageEncoder)model.CreateVisionEncoder();
            using var half = vision.CreateEncoder(device);
            using var full = vision.CreateEncoder(device, weights: EncoderWeights.Float32);
            long after = WeightBytes(encoder), expected = WeightBytes(half.Encoder) + WeightBytes(full.Projector), wholeFloat32 = WeightBytes(full);
            Check(encoder.Encoder.Descendants().OfType<Linear>().All(l => l.BFloat16 is not null) && !encoder.Projector.Buffers().Any(),
                "the tower's projections stay bfloat16, the trained projector is float32");
            Check(after == expected && after < wholeFloat32, $"the encoder's weights take {after} bytes; bfloat16 tower + float32 projector {expected}; all float32 {wholeFloat32}");

            // The trained projector's values are the ones the encoder uses.
            var exported = ((IVisionTuningPart)vision).Export(encoder, VisionTuningParts.Projector);
            using (var trained = vision.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.Float32 }))
            {
                foreach (var (name, tensor) in ((IVisionTuningPart)vision).Export(trained, VisionTuningParts.Projector))
                {
                    Check(tensor.ToArray().SequenceEqual(exported[name].ToArray()), $"{name}: the trained values");
                    tensor.Dispose();
                }
            }

            foreach (var t in exported.Values)
            {
                t.Dispose();
            }

            Console.WriteLine($"    trained projector over a bfloat16 tower: {after} bytes of weights (phase 2 built the whole encoder in float32: {wholeFloat32} bytes; "
                              + $"the tower's share {WeightBytes(half.Encoder)} against {WeightBytes(full.Encoder)} in float32)");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void VisionTuningBatchedImages(Device device)
    {
        const int Count = 12, Repeats = 20;
        var baseImage = ImageCodecs.Decode(TestData("vlm/image.png"));
        var random = new Random(5);
        var images = Enumerable.Range(0, Count).Select(_ => new ImageData([.. baseImage.Pixels.Select(v => v * (0.8f + 0.4f * (float)random.NextDouble()))],
            baseImage.Channels, baseImage.Height, baseImage.Width)).ToList();
        foreach (string family in new[] { "gemma3", "llava" })
        {
            var (model, _) = LoadTuningModel(family, device);
            using var owned = model;
            var part = (IVisionTuningPart)model.Vision!;
            using var encoder = model.Vision!.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.Float32 });
            var stages = (IVisionEncoderStages)encoder;
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            var pixels = images.Select(i => stages.PixelValues(i)).Select(p => Tensor.From(p.ToArray(), p.Shape.ToArray(), device)).ToList();

            // The frozen tower and the projector: joined against one at a time, each image's rows the same.
            var towerAlone = pixels.Select(stages.Tower).ToList();
            var towerJoined = stages.Tower(Tensor.Concat(pixels, 0));
            var featuresAlone = towerAlone.Select(t => part.Features(encoder, t)).ToList();
            var featuresJoined = part.Features(encoder, Tensor.Concat(towerAlone, 0));
            float towerDifference = 0f, featureDifference = 0f;
            int at = 0;
            for (int i = 0; i < Count; i++)
            {
                int rows = towerAlone[i].Shape[0];
                towerDifference = Math.Max(towerDifference, MaxDifference(towerAlone[i].ToArray(), towerJoined.Narrow(0, at, rows).ToArray()));
                featureDifference = Math.Max(featureDifference, MaxDifference(featuresAlone[i].ToArray(), featuresJoined.Narrow(0, at, rows).ToArray()));
                at += rows;
            }

            Check(towerDifference <= 1e-5f && featureDifference <= 1e-5f, $"{family}: an image's tower output and features alone and in a batch differ by {towerDifference:G3} and {featureDifference:G3}");

            double Time(Action pass)
            {
                pass();
                var watch = Stopwatch.StartNew();
                for (int r = 0; r < Repeats; r++)
                {
                    using var inner = new TensorScope();
                    pass();
                }

                device.Synchronize();
                return watch.Elapsed.TotalMilliseconds / Repeats;
            }

            double projectorAlone = Time(() => towerAlone.ForEach(t => part.Features(encoder, t).ToArray()));
            double projectorJoined = Time(() => part.Features(encoder, Tensor.Concat(towerAlone, 0)).ToArray());
            double frozenAlone = Time(() => pixels.ForEach(p => stages.Tower(p).ToArray()));
            double frozenJoined = Time(() => stages.Tower(Tensor.Concat(pixels, 0)).ToArray());
            Console.WriteLine($"    {family}: {Count} images; alone and joined differ by {towerDifference:G3} (tower) and {featureDifference:G3} (features); "
                              + $"projector {projectorAlone:F2} ms one at a time, {projectorJoined:F2} ms in one pass; frozen tower {frozenAlone:F2} ms against {frozenJoined:F2} ms");
        }
    }
}
