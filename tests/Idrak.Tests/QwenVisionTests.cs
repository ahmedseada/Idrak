// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.QwenVision;

// Plan 16: Qwen-VL as a plug-in (samples/QwenVision) on the library's multi-axis rotary positions, checked against
// transformers 5.19's outputs for tiny random Qwen2.5-VL and Qwen2-VL checkpoints (tools/vlm/make_tiny_qwen_vl.py,
// tests/Idrak.Tests/data/vlm-qwen).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] QwenVisionGroup =
    [
        ("qwen vl: the grid position rule places the fixture's prompts as transformers' get_rope_index does (position ids on three axes and the rope delta); the decoder reads mrope_section as three rotary axes", QwenPositionRule),
        ("qwen vl: tiny Qwen2.5-VL and Qwen2-VL (transformers 5's config.json and the hub's) give transformers' grids, pixels, tower (windows and full attention) and features for three pages, prompt logits in one pass and cached, 20 greedy tokens, two pages in one prompt, and a text-only prompt", QwenMatchesReference),
        ("qwen vl: the chat API (ChatGenerator with the family's encoder, format and rule) renders transformers' prompt and answers with its 20 greedy tokens", QwenChat),
        ("qwen vl: a tuning example with an image gives transformers' logits and loss with gradients on, the gradients reach the LoRA adapters; FineTuner.Train runs with the merger trained (the images placed on their axes) and saves it under the checkpoint's names", QwenTuning),
        ("qwen vl: the decoding context's rotary offsets: decode steps add the row's offset (recorded steps the one set last), truncating before a pass's images restores the offsets before it, inside them is refused, Reset clears them", QwenRotaryOffsets),
        ("qwen vl: without the plug-in a Qwen-VL checkpoint is not registered; the plug-in registers both families and their text decoders, and nothing of Qwen in the library", QwenNotRegistered),
    ];

    private static readonly (string Folder, string Tag)[] QwenModels =
        [("tiny-qwen2.5-vl", "qwen2.5-vl"), ("tiny-qwen2.5-vl-flat", "qwen2.5-vl"), ("tiny-qwen2-vl", "qwen2-vl"), ("tiny-qwen2-vl-flat", "qwen2-vl")];

    private static string QwenData(string name) => TestData($"vlm-qwen/{name}");

    private static JsonNode QwenManifest() => JsonNode.Parse(File.ReadAllText(QwenData("manifest.json")))!;

    private static int[] QwenInts(string path) => [.. ReadNpyInt64(path).Select(i => (int)i)];

    private static void QwenPositionRule(Device device)
    {
        QwenVisionPlugin.Register();
        var manifest = QwenManifest();
        var rule = ImagePositionRules.Get(ImagePositionRules.Grid);
        Check(ImagePositionRules.Origin(ImagePositionRules.Grid) == Idrak.Abstraction.Overrides.Library, "the grid rule is the library's");
        int pad = (int)manifest["token_ids"]!["<|image_pad|>"]!;
        foreach (var (prompt, pages) in new[] { ("prompt-image", new[] { "page" }), ("prompt-two-images", new[] { "page", "page-tall" }) })
        {
            var facts = manifest["facts"]!["tiny-qwen2.5-vl"]!;
            int[] ids = QwenInts(QwenData($"reference/qwen2.5-vl-{prompt}-input_ids.npy"));
            int[] expected = QwenInts(QwenData($"reference/qwen2.5-vl-{prompt}-position_ids.npy"));
            var runs = new List<ImageRun>();
            int k = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] != pad)
                {
                    continue;
                }

                var grid = facts["pages"]![pages[k++]]!["grid_thw"]!.AsArray().Select(n => (int)n!).ToArray();
                int rows = grid[1] / 2, columns = grid[2] / 2, tokens = rows * columns;
                var relative = new float[3 * tokens];
                for (int t = 0; t < tokens; t++)
                {
                    (relative[tokens + t], relative[2 * tokens + t]) = (t / columns, t % columns);
                }

                runs.Add(new ImageRun(i, tokens, relative));
                i += tokens - 1;
            }

            var placed = new float[3 * ids.Length];
            int next = rule.Place(0, ids.Length, runs, 3, placed);
            Check(placed.Select(p => (int)p).SequenceEqual(expected), $"{prompt}: positions {string.Join(" ", placed.Take(40))}…");
            Check(next - ids.Length == (int)facts[prompt]!["rope_delta"]!, $"{prompt}: rope delta {next - ids.Length}, transformers' {facts[prompt]!["rope_delta"]}");
        }

        // The decoder's rotary embedding: three axes of 2, 3 and 3 frequency pairs.
        using var model = PretrainedModel.Load(QwenData("tiny-qwen2.5-vl"), new PretrainedOptions { Device = device });
        Check(model.Spec.Rope?.Sections is { Axes: 3 } sections && sections.Pairs.SequenceEqual([2, 3, 3]), $"rotary sections {model.Spec.Rope?.Sections}");
    }

    private static void QwenMatchesReference(Device device)
    {
        QwenVisionPlugin.Register();
        var manifest = QwenManifest();
        using var noGrad = Autograd.NoGrad();
        foreach (var (folder, tag) in QwenModels)
        {
            var facts = manifest["facts"]![$"tiny-{tag}"]!;
            using var model = PretrainedModel.Load(QwenData(folder), new PretrainedOptions { Device = device });
            var vision = model.Vision as QwenVision ?? throw new InvalidOperationException($"{folder}: no Qwen-VL vision part");
            Check(vision.Attention.Causal && vision.PromptFormat.ImageToken == (int)manifest["token_ids"]!["<|image_pad|>"]!, $"{folder}: {vision.Describe()}");
            using var encoder = (QwenImageEncoder)vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            var differences = new List<string>();
            var pageFeatures = new Dictionary<string, ImageFeatures>();
            foreach (string page in (string[])["page", "page-tall", "page-small"])
            {
                var image = ImageCodecs.Decode(QwenData($"{page}.png"));
                var pageFacts = facts["pages"]![page]!;
                var grid = encoder.Processor.Grid(image);
                Check(new[] { grid.Frames, grid.Rows, grid.Columns }.SequenceEqual(pageFacts["grid_thw"]!.AsArray().Select(n => (int)n!)), $"{folder} {page}: grid {grid}");
                using var pixels = encoder.PixelValues(image);
                AssertClose(ReadNpyFloat32(QwenData($"reference/{tag}-{page}-pixel_values.npy")), pixels.ToArray(), 1e-5f, $"{folder} {page}: pixel values");
                using var tower = encoder.Tower(pixels);
                float[] expectedTower = ReadNpyFloat32(QwenData($"reference/{tag}-{page}-tower.npy"));
                differences.Add($"{page} tower {MaxDifference(expectedTower, tower.ToArray()):G3}");
                AssertClose(expectedTower, tower.ToArray(), 1e-4f, $"{folder} {page}: the tower's output");
                var features = encoder.Encode([image]);
                float[] expectedFeatures = ReadNpyFloat32(QwenData($"reference/{tag}-{page}-features.npy"));
                differences.Add($"features {MaxDifference(expectedFeatures, features[0].Features.ToArray()):G3}");
                AssertClose(expectedFeatures, features[0].Features.ToArray(), 1e-4f, $"{folder} {page}: features");
                Check(features[0].Tokens == (int)pageFacts["tokens"]! && features[0].Positions is { } p && p.ToArray().SequenceEqual(encoder.AxisPositions(features[0].Layout)),
                    $"{folder} {page}: {features[0].Tokens} tokens with their axis positions");
                pageFeatures[page] = features[0];
            }

            var chat = model.CreateChat(KeyValueFormat.Float32, 256);
            foreach (var (prompt, pages) in new[] { ("prompt-image", new[] { "page" }), ("prompt-two-images", new[] { "page", "page-tall" }), ("prompt-text", Array.Empty<string>()) })
            {
                var promptFacts = facts[prompt]!;
                int[] ids = QwenInts(QwenData($"reference/{tag}-{prompt}-input_ids.npy"));
                ChatMessage[] messages = prompt switch
                {
                    "prompt-image" => [new("user", [ChatImage.FromFile(QwenData("page.png")), new ChatText("Read the scan.")])],
                    "prompt-two-images" => [new("system", [new ChatText("Read the scans.")]),
                        new("user", [ChatImage.FromFile(QwenData("page.png")), new ChatText("and"), ChatImage.FromFile(QwenData("page-tall.png")), new ChatText("What is in these images?")])],
                    _ => [new("user", "Write one line about the moon and the sun in the night.")],
                };
                string rendered = chat.Template.Render(messages, [], null);
                Check(rendered == (string)promptFacts["rendered"]!, $"{folder} {prompt}: rendered {rendered}");
                var layouts = pages.Select(p => (IReadOnlyList<ImageTokenLayout>)[pageFeatures[p].Layout]).ToList();
                var own = model.Tokenizer!.Encode(vision.PromptFormat.Expand(rendered, layouts, model.Tokenizer));
                Check(own.SequenceEqual(ids), $"{folder} {prompt}: ids {string.Join(" ", own)}");

                float[] expectedLogits = ReadNpyFloat32(QwenData($"reference/{tag}-{prompt}-logits.npy"));
                float[] expectedSteps = ReadNpyFloat32(QwenData($"reference/{tag}-{prompt}-generate-logits.npy"));
                int[] newTokens = [.. promptFacts["new_tokens"]!.AsArray().Select(n => (int)n!)];
                var images = pages.Length == 0 ? [] : ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, [.. pages.Select(p => pageFeatures[p])]);
                using var input = Ids(ids, device);
                var whole = (images.Count == 0 ? model.Network.Forward(input) : model.Network.Forward(input, images, vision.Attention)).ToArray();
                differences.Add($"{prompt} {MaxDifference(expectedLogits, whole):G3}");
                AssertClose(expectedLogits, whole, 1e-4f, $"{folder} {prompt}: logits without the cache");
                using var context = new DecodingContext(device, 1, 256);
                var prefill = (images.Count == 0 ? model.Network.ForwardCached(input, context) : model.Network.ForwardCached(input, images, vision.Attention, context)).ToArray();
                AssertClose(expectedLogits, prefill, 1e-4f, $"{folder} {prompt}: logits of the cached prefill");
                int delta = (int)promptFacts["rope_delta"]!;
                Check(images.Count == 0 ? context.RotaryOffsets is null : context.RotaryOffsets is [var offset] && offset == delta,
                    $"{folder} {prompt}: rotary offset {string.Join(",", context.RotaryOffsets ?? [])}, transformers' rope delta {delta}");
                int vocabulary = model.Spec.Vocabulary;
                var generated = new List<int>();
                var stepLogits = new List<float>();
                float[] last = prefill[^vocabulary..];
                for (int step = 0; step < newTokens.Length; step++)
                {
                    stepLogits.AddRange(last);
                    generated.Add(ArgMax(last));
                    using var next = Ids([generated[^1]], device);
                    last = model.Network.ForwardCached(next, context).ToArray();
                }

                Check(generated.SequenceEqual(newTokens), $"{folder} {prompt}: greedy tokens {string.Join(" ", generated)}, expected {string.Join(" ", newTokens)}");
                differences.Add($"steps {MaxDifference(expectedSteps, [.. stepLogits]):G3}");
                AssertClose(expectedSteps, [.. stepLogits], 1e-4f, $"{folder} {prompt}: the greedy steps' logits");
            }

            foreach (var f in pageFeatures.Values)
            {
                f.Dispose();
            }

            Console.WriteLine($"    {folder} on {device}: largest differences {string.Join(", ", differences)}");
        }
    }

    private static void QwenChat(Device device)
    {
        QwenVisionPlugin.Register();
        var facts = QwenManifest()["facts"]!["tiny-qwen2.5-vl"]!["prompt-image"]!;
        int[] newTokens = [.. facts["new_tokens"]!.AsArray().Select(n => (int)n!)];
        using var model = PretrainedModel.Load(QwenData("tiny-qwen2.5-vl"), new PretrainedOptions { Device = device });
        var vision = model.Vision!;
        using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
        var chat = model.CreateChat(KeyValueFormat.Float32, 256);
        var withImages = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };
        var request = new ChatRequest([new("user", [ChatImage.FromFile(QwenData("page.png")), new ChatText("Read the scan.")])])
        {
            Options = new GenerationOptions { Temperature = 0, NumPredict = 20, RepeatPenalty = 1 },
        };
        var reply = withImages.Chat(request);
        string expected = model.Tokenizer!.Decode(newTokens);
        Check(reply.Stats is { GeneratedTokens: 20 } && reply.Stats.PromptTokens == facts["input_ids"]!.AsArray().Count, $"20 tokens after the prompt: {reply.Stats}");
        Check(reply.Message!.Content == expected, $"the answer: '{reply.Message.Content}', transformers' '{expected}'");
        Check(withImages.Chat(request).Message!.Content == expected, "the same request again (the kept cache up to the image) answers the same");
    }

    private static void QwenTuning(Device device)
    {
        QwenVisionPlugin.Register();
        var facts = QwenManifest()["facts"]!["tiny-qwen2.5-vl"]!["tuning"]!;
        using var model = PretrainedModel.Load(QwenData("tiny-qwen2.5-vl"), new PretrainedOptions { Device = device });
        var vision = model.Vision!;
        int[] ids = QwenInts(QwenData("reference/qwen2.5-vl-tuning-input_ids.npy"));
        int[] answer = [.. facts["answer_ids"]!.AsArray().Select(n => (int)n!)];
        float[] expectedLogits = ReadNpyFloat32(QwenData("reference/qwen2.5-vl-tuning-logits.npy"));
        model.AddAdapters(2, 4f, ["q", "k", "v", "o"]);                                     // B starts at 0: the logits are the base model's
        IReadOnlyList<ImageFeatures> features;
        using (var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device }))
        {
            features = encoder.Encode([ImageCodecs.Decode(QwenData("page.png"))]);
        }

        int vocabulary = model.Spec.Vocabulary;
        using (var scope = new TensorScope())
        {
            using var input = Ids(ids, device);
            var logits = model.Network.Forward(input, ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, features), vision.Attention);
            float[] values = logits.ToArray();
            AssertClose(expectedLogits, values, 1e-4f, "the tuning example's logits with gradients on");
            double loss = 0;
            for (int i = 0; i < answer.Length; i++)
            {
                int row = ids.Length - answer.Length - 1 + i;                             // the logits before each answer token
                var r = values.AsSpan(row * vocabulary, vocabulary);
                float max = float.NegativeInfinity;
                foreach (float v in r)
                {
                    max = Math.Max(max, v);
                }

                double sum = 0;
                foreach (float v in r)
                {
                    sum += Math.Exp(v - max);
                }

                loss -= r[answer[i]] - max - Math.Log(sum);
            }

            loss /= answer.Length;
            Check(Math.Abs(loss - (double)facts["loss"]!) < 1e-4, $"loss {loss}, transformers' {facts["loss"]}");
            logits.Sum().Backward();
            var adapters = model.Network.Parameters().Where(p => p.RequiresGrad).ToList();
            Check(adapters.Count > 0 && adapters.Any(p => p.Grad is { } g && g.ToArray().Any(v => v != 0f && float.IsFinite(v))),
                $"gradients reach the {adapters.Count} adapter tensors through the multi-axis positions");
        }

        foreach (var f in features)
        {
            f.Dispose();
        }

        // FineTuner.Train with the merger trained: the image placed on its axes by the encoder's AxisPositions.
        using var tuningVision = TuningVision.Create(model, TuningImages.Parse(null), [VisionTuningParts.Projector]);
        var transcript = new ChatTranscript([new ChatMessage("user", [ChatImage.FromFile(QwenData("page.png")), new ChatText("Read the scan.")]),
            new ChatMessage("assistant", (string)facts["answer"]!)], []);
        var sequence = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Vision = tuningVision }.Encode(transcript, 512)!;
        Check(sequence.Images is [{ AxisPositions: [{ } axes] }] && axes.Length == 3 * sequence.Images[0].Tokens, "the training image carries its axis positions");
        var losses = new List<float>();
        string folder = TempFolder();
        try
        {
            FineTuner.Train(model, [sequence], null, new FineTuningOptions
            {
                Rank = 2, Alpha = 4f, Targets = ["q", "k", "v", "o"], Epochs = 3, BatchTokens = 512, Vision = tuningVision,
                Optimizer = ps => new Idrak.Abstraction.Training.Sgd(ps, 0.05f), Scheduler = FineTuningSchedules.Constant(), MaxGradientNorm = 0f,
            }, folder, new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)));
            Check(losses.Count == 3 && losses.All(float.IsFinite) && losses[^1] < losses[0], $"losses {string.Join(", ", losses)}");
            using var saved = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
            Check(saved.Tensors.Keys.Any(k => k.EndsWith("visual.merger.mlp.0.weight", StringComparison.Ordinal)), $"the merger saved: {string.Join(", ", saved.Tensors.Keys.Take(6))}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void QwenRotaryOffsets(Device device)
    {
        using var context = new DecodingContext(device, 2, 64);
        context.BeginStep(5);
        context.EndStep(5);
        Check(context.RotaryOffsets is null && context.Recordable, "no offsets until a pass sets them");
        context.SetRotaryOffsets([-3, -3], from: 2);                                   // a pass whose images start at 2 ended at 5
        Check(context.Recordable, "one offset for every row is recordable");
        context.BeginStep(1);
        Check(context.RotaryPositions is { } rp && rp.ToArray() is [2f], $"one step at 5 with offset -3 rotates by 2: {string.Join(",", context.RotaryPositions?.ToArray() ?? [])}");
        context.EndStep(1);
        context.BeginStep(2);
        Check(context.RotaryPositions!.ToArray().SequenceEqual([3f, 4f]), $"two steps at 6: {string.Join(",", context.RotaryPositions.ToArray())}");
        context.EndStep(2);
        context.SetRotaryOffsets([-3, -1], from: 6);                                   // a second pass, images from 6 to 8
        Check(!context.Recordable, "rows with different offsets are not recordable");
        context.BeginStep(2);
        Check(context.RotaryPositions!.ToArray().SequenceEqual([5f, 6f, 7f, 8f]), $"per row: {string.Join(",", context.RotaryPositions.ToArray())}");
        context.EndStep(2);
        Check(Fails(() => context.Truncate(7)), "inside a pass's images is refused");
        context.Truncate(6);
        Check(context.RotaryOffsets is [-3, -3], $"truncating to the second pass's images restores the first's offsets: {string.Join(",", context.RotaryOffsets ?? [])}");
        context.Truncate(1);
        Check(context.RotaryOffsets is null, "truncating before the first pass's images clears them");
        context.SetRotaryOffsets([4, 4], from: 0);
        context.Reset();
        Check(context.RotaryOffsets is null, "Reset clears them");
    }

    private static void QwenNotRegistered(Device device)
    {
        foreach (string architecture in (string[])[QwenVisionFamily.Qwen25VlArchitecture, QwenVisionFamily.Qwen2VlArchitecture])
        {
            VisionFamilies.Unregister(architecture);
            PretrainedArchitectures.Unregister(architecture);
            Check(VisionFamilies.Default(architecture) is null && PretrainedArchitectures.Default(architecture) is null, $"{architecture}: nothing of it in the library");
        }

        try
        {
            using var _ = PretrainedModel.Load(QwenData("tiny-qwen2.5-vl"), new PretrainedOptions { Device = device });
            Check(false, "a Qwen-VL checkpoint without the plug-in is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains($"No architecture '{QwenVisionFamily.Qwen25VlArchitecture}' is registered", StringComparison.Ordinal), e.Message);
        }

        QwenVisionPlugin.Register();
        Check(VisionFamilies.Find(QwenVisionFamily.Qwen25VlArchitecture) is QwenVisionFamily && PretrainedArchitectures.Names.Contains(QwenVisionFamily.Qwen2VlArchitecture),
            "the plug-in registers both families and their text decoders");
    }
}
