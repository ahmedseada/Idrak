// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Gemma3Vision;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

// Plan 12, phase 0: the tiny vision fine-tune of tools/vlm/make_tiny_tuning.py (tests/Idrak.Tests/data/vlm-tuning)
// checked with what the library has today, before the tuner takes images: the records' rendering, image expansion, token
// ids and trained mask through ChatTranscriptEncoder and the family's IImagePromptFormat; then three SGD steps by hand
// (ImagePrefill.Forward with autograd, the fixture's LoRA values, a masked mean cross-entropy written out here): the
// loss of every step, the gradients of step 1 and the adapters after step 3, against transformers. peft's lora_A
// [rank, in] and lora_B [out, rank] are transposed (TransposeRows) to Idrak's [in, rank] and [rank, out].
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionTuningReferenceGroup =
    [
        ("vision tuning reference: the tiny Gemma 3 and LLaVA records render, expand and tokenize as transformers' processor does, the answer and its end of turn trained and the prompt, image tokens and their markers not (LlamaFactory's gemma3 template: the same ids, plus the \"\\n\" after <end_of_turn>)", VisionTuningRecords),
        ("vision tuning reference: three SGD steps of LoRA on q, k, v, o with images (ImagePrefill.Forward with autograd, the fixture's initial adapters, the masked mean cross-entropy) give transformers' losses, step-1 gradients and final adapters; with the Gemma 3 projector trained too, its gradients and values", VisionTuningSteps),
        ("vision tuning trainer: FineTuner.Train on the tiny Gemma 3 and LLaVA records (LoRA alone, and with the projector) gives transformers' losses and final adapters and projector; the tower runs once per distinct image (the feature cache serves the later epochs); adapters, projector (modules_to_save) and tuning_images.json saved, read back by LoadAdapter and MergeAdapter into the vision encoder", VisionTuningTrainer),
        ("vision tuning trainer: checkpointed blocks with images give the gradients of stored activations, bit for bit (the image blocks travel with the recompute, the scope closed before the backward pass), through ImagePrefill.Begin and through FineTuner.Train", VisionTuningCheckpointed),
        ("vision tuning trainer: a vision family no longer registered fails with the registry's message before any step; a text model given images, and a text-only encoder given a transcript with images, fail clearly", VisionTuningRefusals),
    ];

    private static string TuningData(string name) => TestData($"vlm-tuning/{name}");

    private static int[] JsonInts(JsonNode? node) => [.. node!.AsArray().Select(n => (int)n!)];

    private static (PretrainedModel Model, JsonNode Tuning) LoadTuningModel(string family, Device device)
    {
        if (family == "gemma3")
        {
            RegisterGemma3Vision();
        }
        else
        {
            Idrak.PluginTests.LlavaPlugin.Register();
        }

        var tuning = JsonNode.Parse(File.ReadAllText(TuningData($"{family}/tuning.json")))!;
        string folder = ((string)tuning["settings"]!["model"]!)["../".Length..];
        return (PretrainedModel.Load(TestData(folder), new PretrainedOptions { Device = device }), tuning);
    }

    // The record's messages as the tuner will read them: text as strings, the user's parts with its images (by file name in data/vlm).
    private static ChatTranscript TuningTranscript(JsonNode record)
    {
        var files = new Queue<string>(record["images"]!.AsArray().Select(n => (string)n!));
        var messages = new List<ChatMessage>();
        foreach (var m in record["messages"]!.AsArray())
        {
            string role = (string)m!["role"]!;
            if (m["content"] is JsonValue text)
            {
                messages.Add(new ChatMessage(role, (string)text!));
                continue;
            }

            messages.Add(new ChatMessage(role, [.. m["content"]!.AsArray().Select(p => (string)p!["type"]! == "image"
                ? (ChatPart)ChatImage.FromFile(TestData($"vlm/{files.Dequeue()}"))
                : new ChatText((string)p["text"]!))]));
        }

        return new ChatTranscript(messages, []);
    }

    private static void VisionTuningRecords(Device device)
    {
        foreach (string family in new[] { "gemma3", "llava" })
        {
            var (model, tuning) = LoadTuningModel(family, device);
            using var owned = model;
            var vision = model.Vision!;
            var tokenizer = model.Tokenizer!;
            var settings = tuning["settings"]!;
            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer);
            Check(encoder.AssistantHeader == (string)settings["assistant_header"]! && encoder.AssistantEnd == (string)settings["assistant_end"]!,
                $"{family}: the assistant's turn is '{encoder.AssistantHeader}' ... '{encoder.AssistantEnd}'");
            using var images = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            using var tuningVision = TuningVision.Create(model);
            var imageEncoder = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer) { Vision = tuningVision };
            int index = 0;
            foreach (var record in tuning["records"]!.AsArray())
            {
                string what = $"{family} record {index++}";
                var transcript = TuningTranscript(record!);
                var (text, spans) = encoder.Render(transcript);
                Check(text == (string)record!["rendered"]!, $"{what}: rendered\n{text}\nexpected\n{record["rendered"]}");
                int[] span = JsonInts(record["assistant_span"]);
                Check(spans is [var only] && only == (span[0], span[1]), $"{what}: the answer's span {string.Join(", ", spans)}");
                var (start, end) = spans[0];

                // The image markers (all in the prompt) expanded by the family's format, each image as its encoder's blocks.
                var layouts = record["images"]!.AsArray().Select(f => images.Blocks(Idrak.Data.Abstractions.ImageCodecs.Decode(TestData($"vlm/{(string)f!}")))).ToList();
                string prompt = vision.PromptFormat.Expand(text[..start], layouts, tokenizer);
                Check(prompt + text[start..] == (string)record["expanded"]!, $"{what}: expanded\n{prompt + text[start..]}\nexpected\n{record["expanded"]}");

                // Prompt, answer and what follows tokenized each on its own (as the encoder does), the answer trained.
                var ids = new List<int>();
                var trained = new List<bool>();
                foreach (var (segment, train) in new[] { (prompt, false), (text[start..end], true), (text[end..], false) })
                {
                    var part = tokenizer.Encode(segment);
                    ids.AddRange(part);
                    trained.AddRange(Enumerable.Repeat(train, part.Count));
                }

                int[] expectedIds = JsonInts(record["input_ids"]);
                bool[] expectedTrained = [.. record["trained"]!.AsArray().Select(n => (bool)n!)];
                Check(ids.SequenceEqual(expectedIds), $"{what}: ids {string.Join(" ", ids)}\nexpected {string.Join(" ", expectedIds)}");
                Check(trained.SequenceEqual(expectedTrained), $"{what}: trained mask");
                Check(trained.Skip(1).Count(t => t) == (int)record["trained_tokens"]!, $"{what}: trained targets");

                // Nothing of the prompt is trained (its image tokens, their begin and end tokens, the turn headers); the answer is,
                // up to the token that ends the turn.
                int imageToken = vision.PromptFormat.ImageToken;
                int promptTokens = tokenizer.Encode(prompt).Count, lastTrained = trained.LastIndexOf(true);
                Check(ids.LastIndexOf(imageToken) < promptTokens && !trained.Take(promptTokens).Any(t => t) && trained.Skip(promptTokens).Take(lastTrained + 1 - promptTokens).All(t => t),
                    $"{what}: the prompt's {promptTokens} tokens untrained, the answer trained");
                string endText = tokenizer.Decode([ids[lastTrained]]);
                Check(family == "llava" || endText == "<end_of_turn>", $"{what}: the last trained token is '{endText}'");

                // The image blocks where the prefill finds them.
                var blocks = record["image_blocks"]!.AsArray();
                var features = blocks.Select(b => Tensor.From(new float[(int)b!["tokens"]! * vision.Width], [(int)b["tokens"]!, vision.Width], device)).ToList();
                var located = ImagePrefill.Locate(ids, imageToken, features);
                Check(located.Select(p => p.Position).SequenceEqual(blocks.Select(b => (int)b!["position"]!)), $"{what}: image blocks at {string.Join(", ", located.Select(p => p.Position))}");
                features.ForEach(f => f.Dispose());

                // The encoder's own sequence with the family's vision side (phase 2): the same ids and trained mask, the images
                // where the prefill finds them; without a vision side, a transcript with images is refused.
                var sequence = imageEncoder.Encode(transcript, 4096)!;
                Check(sequence.Tokens.SequenceEqual(expectedIds) && sequence.Trained.SequenceEqual(expectedTrained), $"{what}: the encoder's ids or trained mask differ");
                var placed = sequence.Images.SelectMany(i => i.Blocks).ToList();
                Check(placed.Select(b => b.Position).SequenceEqual(blocks.Select(b => (int)b!["position"]!)) && placed.Select(b => b.Tokens).SequenceEqual(blocks.Select(b => (int)b!["tokens"]!)),
                    $"{what}: the encoder's image blocks at {string.Join(", ", placed)}");
                Check(sequence.Images.Select(i => i.Image).SequenceEqual(record["images"]!.AsArray().Select(f => ChatImage.FromFile(TestData($"vlm/{(string)f!}")))),
                    $"{what}: the encoder's images");
                Check(Failure<InvalidOperationException>(() => encoder.Encode(transcript, 4096)).Message.Contains("ChatTranscriptEncoder.Vision", StringComparison.Ordinal),
                    $"{what}: a text-only encoder refuses images");

                // Cut inside the second image's block: the cut moves to its start (whole blocks only); nothing trainable is left.
                if (blocks.Count > 1)
                {
                    int inside = (int)blocks[1]!["position"]! + 1;
                    var cut = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer) { Vision = tuningVision, ShortenToFit = false }.Encode(transcript, inside);
                    Check(cut is null, $"{what}: a sequence cut inside an image keeps no trainable token");
                }

                // LlamaFactory's gemma3 template: the same ids; it trains the "\n" after <end_of_turn> too (plan 12's decision: not trained).
                if (record["llamafactory"] is { } lf)
                {
                    bool[] lfTrained = [.. lf["trained"]!.AsArray().Select(n => (bool)n!)];
                    int[] differs = [.. Enumerable.Range(0, ids.Count).Where(k => lfTrained[k] != trained[k])];
                    Check(JsonInts(lf["input_ids"]).SequenceEqual(ids) && differs is [var k] && k == ids.Count - 1 && lfTrained[k] && tokenizer.Decode([ids[k]]) == "\n",
                        $"{what}: LlamaFactory differs at {string.Join(", ", differs)}");
                }
            }

            Console.WriteLine($"    {family}: {index} records, the answers (and the end of turn the template writes) trained ({string.Join(", ", tuning["records"]!.AsArray().Select(r => (int)r!["trained_tokens"]!))} tokens)");
        }
    }

    private static void VisionTuningSteps(Device device)
    {
        foreach (string family in new[] { "gemma3", "llava" })
        {
            // By hand, Gemma 3's projector through its encoder's module; LLaVA's projector run goes through IVisionTuningPart
            // in "vision tuning trainer" (FineTuner.Train).
            foreach (string run in family == "gemma3" ? new[] { "lora", "projector" } : ["lora"])
            {
                VisionTuningRun(family, run, device);
            }
        }
    }

    private static void VisionTuningRun(string family, string run, Device device)
    {
        string what = $"{family} {run}";
        var (model, tuning) = LoadTuningModel(family, device);
        using var owned = model;
        var settings = tuning["settings"]!;
        var lora = settings["lora"]!;
        var vision = model.Vision!;
        int added = model.AddAdapters((int)lora["rank"]!, (float)lora["alpha"]!, ["q", "k", "v", "o"]);
        int loaded = model.LoadAdapter(TuningData($"{family}/adapter-init"));
        Check(added == loaded && loaded == 4 * model.Network.Descendants().OfType<DecoderBlock>().Count(), $"{what}: {added} adapters added, {loaded} loaded");

        // The adapters by their checkpoint names (as the fixture names them: "base_model.model." + the module's tensor name).
        var adapters = new SortedDictionary<string, (LoraAdapter Adapter, int In, int Out)>(StringComparer.Ordinal);
        foreach (var (path, module) in model.NamedModules())
        {
            if (module is Linear { Lora: { } adapter } linear)
            {
                string weight = model.Architecture.TensorName($"{path}.weight")!;
                adapters["base_model.model." + weight[..^".weight".Length]] = (adapter, linear.InFeatures, linear.OutFeatures);
            }
        }

        // The images' features: the encoder's (frozen), against the fixture's; with the projector trained, through it.
        using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
        var files = tuning["records"]!.AsArray().SelectMany(r => r!["images"]!.AsArray().Select(f => (string)f!)).Distinct().Order(StringComparer.Ordinal).ToList();
        var projector = run == "projector" ? ((Gemma3ImageEncoder)encoder).Projector : null;
        var trainedProjector = projector?.Parameters().ToList() ?? [];
        var features = new Dictionary<string, Tensor>();
        foreach (string file in files)
        {
            var image = Idrak.Data.Abstractions.ImageCodecs.Decode(TestData($"vlm/{file}"));
            Tensor frozen;
            using (Autograd.NoGrad())
            {
                frozen = encoder.Encode([image])[0].Features;
            }

            float[] expected = ReadNpyFloat32(TuningData($"{family}/features/{Path.ChangeExtension(file, ".npy")}"));
            AssertClose(expected, frozen.ToArray(), 1e-4f, $"{what}: the features of {file}");
            features[file] = frozen;
        }

        var parameters = adapters.Values.SelectMany(a => new[] { a.Adapter.A, a.Adapter.B }).Concat(trainedProjector).ToList();
        var optimizer = new Idrak.Abstraction.Training.Sgd(parameters, (float)settings["optimizer"]!["learning_rate"]!);
        var expectedRun = tuning["runs"]![run]!;
        float[] expectedLosses = [.. expectedRun["losses"]!.AsArray().Select(n => (float)n!)];
        var records = tuning["records"]!.AsArray();
        int total = records.Sum(r => (int)r!["trained_tokens"]!);
        var losses = new List<float>();
        for (int step = 0; step < expectedLosses.Length; step++)
        {
            optimizer.ZeroGrad();
            Dictionary<string, Tensor> stepFeatures = features;
            if (projector is not null)
            {
                var stages = (IVisionEncoderStages)encoder;
                stepFeatures = files.ToDictionary(f => f, f =>
                {
                    var tower = stages.Tower(stages.PixelValues(Idrak.Data.Abstractions.ImageCodecs.Decode(TestData($"vlm/{f}"))));
                    return projector.Forward(tower);                                     // gradients reach the projector (the tower is frozen)
                });
            }

            Tensor? loss = null;
            int index = 0;
            foreach (var record in records)
            {
                int[] ids = JsonInts(record!["input_ids"]);
                bool[] trained = [.. record["trained"]!.AsArray().Select(n => (bool)n!)];
                var images = ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, [.. record["images"]!.AsArray().Select(f => stepFeatures[(string)f!])]);
                using var input = Ids(ids, device);
                var logits = model.Network.Forward(input, images, vision.Attention);         // [1, steps, vocabulary]
                int vocabulary = logits.Shape[2];
                var weights = new float[ids.Length * vocabulary];
                for (int t = 0; t + 1 < ids.Length; t++)
                {
                    if (trained[t + 1])
                    {
                        weights[t * vocabulary + ids[t + 1]] = 1f / total;
                    }
                }

                var logProbabilities = logits.LogSoftmax();
                var term = (logProbabilities * Tensor.From(weights, [1, ids.Length, vocabulary], device)).Sum() * -1f;
                loss = loss is null ? term : loss + term;
                if (step == 0)
                {
                    // Each trained token's cross-entropy, as transformers computed it.
                    float[] lp = logProbabilities.ToArray();
                    float[] mine = [.. Enumerable.Range(0, ids.Length - 1).Where(t => trained[t + 1]).Select(t => -lp[t * vocabulary + ids[t + 1]])];
                    float[] theirs = [.. expectedRun["step1_per_record"]![index]!["trained_cross_entropy"]!.AsArray().Select(n => (float)n!)];
                    AssertClose(theirs, mine, 1e-4f, $"{what}: record {index}'s token cross-entropies at step 1");
                }

                index++;
            }

            loss!.Backward();
            losses.Add(loss.ToArray()[0]);
            if (step == 0)
            {
                CheckTuningGradients(what, adapters, trainedProjector, TuningData($"{family}/{run}/grads-step1.safetensors"), settings);
            }

            optimizer.Step();
        }

        AssertClose(expectedLosses, [.. losses], 1e-4f, $"{what}: the losses of {expectedLosses.Length} steps");

        // The adapters (and projector) after the last step.
        using (var reader = SafeTensorsReader.Open(TuningData($"{family}/{run}/adapter-step{expectedLosses.Length}/adapter_model.safetensors")))
        {
            foreach (var (key, (adapter, inputs, outputs)) in adapters)
            {
                AssertClose(TransposeRows(reader.Read($"{key}.lora_A.weight"), adapter.Rank, inputs), adapter.A.ToArray(), 1e-5f, $"{what}: {key} A after the steps");
                AssertClose(TransposeRows(reader.Read($"{key}.lora_B.weight"), outputs, adapter.Rank), adapter.B.ToArray(), 1e-5f, $"{what}: {key} B after the steps");
            }
        }

        if (projector is not null)
        {
            using var reader = SafeTensorsReader.Open(TuningData($"{family}/{run}/projector-step{expectedLosses.Length}.safetensors"));
            foreach (var p in trainedProjector)
            {
                AssertClose(reader.Read(ProjectorName(p, settings)), p.ToArray(), 1e-5f, $"{what}: {ProjectorName(p, settings)} after the steps");
            }
        }

        Console.WriteLine($"    {what} on {device}: losses {string.Join(", ", losses.Select(l => l.ToString("F6")))} (transformers {string.Join(", ", expectedLosses.Select(l => l.ToString("F6")))})");
    }

    // Gemma 3's projector parameters by shape: the norm's gain [vision width], the projection [vision width, text width] (stored as Idrak keeps it).
    private static string ProjectorName(Tensor parameter, JsonNode settings) =>
        settings["projector_parameters"]!.AsArray().Select(n => (string)n!).Single(n => n.EndsWith(parameter.Rank == 1 ? "mm_soft_emb_norm.weight" : "mm_input_projection_weight", StringComparison.Ordinal));

    private static void CheckTuningGradients(string what, SortedDictionary<string, (LoraAdapter Adapter, int In, int Out)> adapters, List<Tensor> projector, string file, JsonNode settings)
    {
        using var reader = SafeTensorsReader.Open(file);
        float largest = reader.Tensors.Keys.Max(k => reader.Read(k).Max(MathF.Abs));
        float tolerance = 1e-4f * largest;
        float worst = 0f;
        void Compare(float[] expected, Tensor parameter, string name)
        {
            Check(parameter.Grad is not null, $"{what}: {name} has no gradient");
            float[] actual = parameter.Grad!.ToArray();
            worst = MathF.Max(worst, MaxDifference(expected, actual));
            AssertClose(expected, actual, tolerance, $"{what}: the gradient of {name} at step 1");
        }

        foreach (var (key, (adapter, inputs, outputs)) in adapters)
        {
            Compare(TransposeRows(reader.Read($"{key}.lora_A.weight"), adapter.Rank, inputs), adapter.A, $"{key}.lora_A");
            Compare(TransposeRows(reader.Read($"{key}.lora_B.weight"), outputs, adapter.Rank), adapter.B, $"{key}.lora_B");
        }

        foreach (var p in projector)
        {
            Compare(reader.Read(ProjectorName(p, settings)), p, ProjectorName(p, settings));
        }

        Check(reader.Tensors.Count == 2 * adapters.Count + projector.Count, $"{what}: {reader.Tensors.Count} gradients in the fixture");
        Console.WriteLine($"    {what}: step-1 gradients of {reader.Tensors.Count} tensors within {worst:G3} (largest {largest:G3})");
    }
}
