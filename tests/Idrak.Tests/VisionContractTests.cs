// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Gemma3Vision;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Nlp;

// Plan 11, "Vision contracts (families are applications)": the library registers no vision family. Gemma 3's is a
// registration from outside it (samples/Gemma3Vision), registered by the tests that use it; without it a Gemma 3
// vision checkpoint is refused, naming the registry.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionContractGroup =
    [
        ("vision contracts: a tiny LLaVA registered from outside the library (CLIP tower, vision_feature_layer, select strategy, MLP projector, causal image tokens) gives transformers' pixels, features, prompt logits and 20 greedy tokens, in both of its configurations", LlavaMatchesReference),
        ("vision contracts: the tiny LLaVA through the public chat API (ChatGenerator with the family's encoder, format and rule) answers with transformers' 20 greedy tokens", LlavaChat),
        ("vision contracts: the testing kit's vision-encoder suite passes for the Gemma 3 and LLaVA encoders (deterministic, token counts, batch equals single, device equals CPU)", VisionEncoderConformance),
        ("vision contracts: the library registers no vision family; image token formats expand by each image's blocks (several with Join's text); images whose tokens touch are told apart by their counts; M-RoPE position ids are refused by the decoder", VisionContractRules),
    ];

    private static string LlavaData(string name) => TestData($"vlm-llava/{name}");

    private static void LlavaMatchesReference(Device device)
    {
        Idrak.PluginTests.LlavaPlugin.Register();
        var manifest = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(LlavaData("manifest.json")))!;
        var image = ImageCodecs.Decode(TestData("vlm/image.png"));
        using var noGrad = Autograd.NoGrad();
        foreach (var (folder, tag) in new[] { ("tiny-llava", ""), ("tiny-llava-full", "-full") })
        {
            var facts = manifest["facts"]![folder]!;
            int[] ids = [.. ReadNpyInt64(LlavaData($"reference/prompt-image-input_ids{tag}.npy")).Select(i => (int)i)];
            int[] newTokens = [.. facts["new_tokens"]!.AsArray().Select(n => (int)n!)];
            float[] expectedLogits = ReadNpyFloat32(LlavaData($"reference/prompt-image-logits{tag}.npy"));
            float[] expectedSteps = ReadNpyFloat32(LlavaData($"reference/prompt-image-generate-logits{tag}.npy"));
            float[] referenceFeatures = ReadNpyFloat32(LlavaData($"reference/image_features{tag}.npy"));
            int tokens = (int)facts["image_tokens"]!;
            using var model = PretrainedModel.Load(LlavaData(folder), new PretrainedOptions { Device = device });
            Check(model.Notes.Count == 0, $"{folder}: {string.Join("; ", model.Notes)}");
            var vision = model.Vision as Idrak.PluginTests.LlavaVision ?? throw new InvalidOperationException($"{folder}: no LLaVA vision part");
            Check(vision is { Family: "LlavaForConditionalGeneration", Width: 32 } && vision.Layout.Tokens == tokens && vision.Attention.Causal
                  && vision.PromptFormat.ImageToken == (int)manifest["token_ids"]!["<image>"]!, $"{folder}: {vision.Layout}, rule {vision.Attention.Name}");

            // Pixels (shortest edge, bicubic, center crop), the selected hidden states, the projected features.
            using var encoder = (Idrak.PluginTests.LlavaImageEncoder)vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            float[] pixels = encoder.Preprocessor.Pixels(image);
            AssertClose(ReadNpyFloat32(LlavaData($"reference/pixel_values{tag}.npy")), pixels, 1e-5f, $"{folder}: pixel values");
            using var pixelTensor = Tensor.From(pixels, [1, 3, 28, 28], device);
            float[] selected;
            using (var tower = encoder.Tower(pixelTensor))
            using (var cut = vision.Full ? null : tower.Narrow(1, 1, 16))
            {
                selected = (cut ?? tower).ToArray();
            }

            float[] referenceSelected = ReadNpyFloat32(LlavaData($"reference/vision_selected{tag}.npy"));
            float selectedDifference = MaxDifference(referenceSelected, selected);
            AssertClose(referenceSelected, selected, 1e-5f, $"{folder}: the selected hidden states");

            // The projector alone, from transformers' selected hidden states (its error apart from the tower's).
            float projectorDifference;
            using (var given = Tensor.From(referenceSelected, [1, tokens, referenceSelected.Length / tokens], device))
            using (var projected = encoder.Projector.Predict(given))
            {
                projectorDifference = MaxDifference(referenceFeatures, projected.ToArray());
            }
            var features = encoder.Encode([image]);
            float[] featureValues = features[0].Features.ToArray();
            float featureDifference = MaxDifference(referenceFeatures, featureValues);
            AssertClose(referenceFeatures, featureValues, 3e-5f, $"{folder}: projected features");

            // The prompt: the template and the family's format give the processor's ids.
            var chat = model.CreateChat(KeyValueFormat.Float32, 128);
            var rendered = chat.Template.Render([new("system", "Read the scan."), new("user", [ChatImage.FromFile(TestData("vlm/image.png")), new ChatText("What is in this image?")])], [], null);
            Check(rendered == (string)facts["rendered_by_chat_template"]!, $"{folder}: rendered {rendered}");
            var own = model.Tokenizer!.Encode(vision.PromptFormat.Expand(rendered, [[vision.Layout]], model.Tokenizer));
            Check(own.SequenceEqual(ids), $"{folder}: ids {string.Join(" ", own)}");

            // Prompt logits (one pass and cached), then 20 greedy steps over the cache.
            int vocabulary = model.Spec.Vocabulary;
            var images = ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, features);
            Check(images is [{ Position: 11 }] && images[0].Tokens == tokens, $"{folder}: the image's tokens");
            using var input = Ids(ids, device);
            var whole = model.Network.Forward(input, images, vision.Attention).ToArray();
            float wholeDifference = MaxDifference(expectedLogits, whole);
            AssertClose(expectedLogits, whole, 2e-5f, $"{folder}: prompt logits without the cache");
            using var context = new DecodingContext(device, 1, 128);
            var prefill = model.Network.ForwardCached(input, images, vision.Attention, context).ToArray();
            float prefillDifference = MaxDifference(expectedLogits, prefill);
            AssertClose(expectedLogits, prefill, 2e-5f, $"{folder}: prompt logits of the cached prefill");
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

            Check(generated.SequenceEqual(newTokens), $"{folder}: greedy tokens {string.Join(" ", generated)}, expected {string.Join(" ", newTokens)}");
            float stepDifference = MaxDifference(expectedSteps, [.. stepLogits]);
            AssertClose(expectedSteps, [.. stepLogits], 2e-5f, $"{folder}: the greedy steps' logits");
            foreach (var f in features)
            {
                f.Dispose();
            }

            Console.WriteLine($"    {folder} on {device}: {tokens} image tokens; largest differences {selectedDifference:G3} (selected hidden states), {projectorDifference:G3} (projector alone), {featureDifference:G3} (features), {wholeDifference:G3} (prompt, one pass), "
                + $"{prefillDifference:G3} (cached prefill), {stepDifference:G3} (20 greedy steps); tokens {string.Join(" ", generated)}");
        }
    }

    private static void LlavaChat(Device device)
    {
        Idrak.PluginTests.LlavaPlugin.Register();
        var facts = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(LlavaData("manifest.json")))!["facts"]!["tiny-llava"]!;
        int[] newTokens = [.. facts["new_tokens"]!.AsArray().Select(n => (int)n!)];
        using var model = PretrainedModel.Load(LlavaData("tiny-llava"), new PretrainedOptions { Device = device });
        var vision = model.Vision!;
        using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
        var chat = model.CreateChat(KeyValueFormat.Float32, 128);
        var withImages = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };
        var request = new ChatRequest([new("system", "Read the scan."), new("user", [ChatImage.FromFile(TestData("vlm/image.png")), new ChatText("What is in this image?")])])
        {
            Options = new GenerationOptions { Temperature = 0, NumPredict = 20, RepeatPenalty = 1 },
        };
        var reply = withImages.Chat(request);
        string expected = model.Tokenizer!.Decode(newTokens);
        Check(reply.Stats is { GeneratedTokens: 20 } && reply.Stats.PromptTokens == ((int[])[.. facts["input_ids"]!.AsArray().Select(n => (int)n!)]).Length,
            $"20 tokens after the prompt: {reply.Stats}");
        Check(reply.Message!.Content.Trim() == expected.Trim(), $"the answer: '{reply.Message.Content}', transformers' '{expected}'");
    }

    private static void VisionEncoderConformance(Device device)
    {
        RegisterGemma3Vision();
        Idrak.PluginTests.LlavaPlugin.Register();
        var samples = new[] { ImageCodecs.Decode(TestData("vlm/image.png")), ImageCodecs.Decode(TestData("vlm/image-gray.jpg")) };
        foreach (string folder in new[] { "vlm/tiny-gemma3", "vlm-llava/tiny-llava", "vlm-llava/tiny-llava-full" })
        {
            using var model = PretrainedModel.Load(TestData(folder), new PretrainedOptions { Device = device });
            using var encoder = model.Vision!.CreateEncoder(new VisionEncoderOptions { Device = device });
            using var cpu = model.Vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
            var report = Idrak.Abstraction.Testing.Conformance.CheckVisionEncoder(encoder, cpu, samples);
            Console.WriteLine($"    {folder} on {device}: {report.Cases} cases, {report.Entries.Count(e => e.Status == Idrak.Abstraction.Testing.CheckStatus.Passed)} properties pass");
            report.ThrowIfFailed();
        }
    }

    private static void VisionContractRules(Device device)
    {
        Gemma3VisionNotRegistered(device);
        Check(VisionFamilies.Names.Count == 0 || VisionFamilies.Names.All(n => VisionFamilies.Default(n) is null), "no vision family is a library default");

        // An image token format expands each marker by its own image's count, with or without begin and end tokens.
        var tokenizer = BpeTokenizer.Load(LlavaData("tiny-llava"));
        int image = tokenizer.Encode("<image>").Last();
        var llava = new ImageTokenFormat("llava", image, image);
        string two = llava.Expand("A <image> and <image>.", [[new ImageTokenLayout(2)], [new ImageTokenLayout(3)]], tokenizer);
        Check(two == "A <image><image> and <image><image><image>.", $"expanded by layouts: {two}");
        Check(Fails(() => llava.Expand("<image>", [[new ImageTokenLayout(1)], [new ImageTokenLayout(1)]], tokenizer)), "one marker for two images is refused");

        // An image of several blocks: refused by a format without Join, written by Join's text otherwise (each block whole).
        try
        {
            llava.Expand("A <image>.", [[new ImageTokenLayout(1), new ImageTokenLayout(2)]], tokenizer);
            Check(false, "several blocks without Join are refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("2 blocks", StringComparison.Ordinal) && e.Message.Contains("Join", StringComparison.Ordinal), e.Message);
        }

        var views = new ImageTokenFormat("views", image, image, before: "[", after: "]") { Join = blocks => "whole " + blocks[0] + " parts " + string.Join(" ", blocks.Skip(1)) };
        string several = views.Expand("A <image>, <image>.", [[new ImageTokenLayout(1), new ImageTokenLayout(2), new ImageTokenLayout(1)], [new ImageTokenLayout(2)]], tokenizer);
        Check(several == "A whole [<image>] parts [<image><image>] [<image>], [<image><image>].", $"several blocks joined: {several}");
        Check(views.Block(new ImageTokenLayout(3), tokenizer) == "[<image><image><image>]", "one block's text");

        // Two images whose tokens touch (LLaVA writes <image><image>): told apart by their counts.
        using var a = Tensor.From(new float[2 * 4], [2, 4], device);
        using var b = Tensor.From(new float[3 * 4], [3, 4], device);
        int[] ids = [1, image, image, image, image, image, 9];
        var found = ImagePrefill.Locate(ids, image, [a, b]);
        Check(found is [{ Position: 1, Tokens: 2 }, { Position: 3, Tokens: 3 }], "adjacent images by their counts");
        Check(Fails(() => ImagePrefill.Locate(ids, image, [a, a])), "image tokens left over are refused");
        Check(Fails(() => ImagePrefill.Locate(ids, image, [b, b])), "too few image tokens are refused");
        Check(Fails(() => _ = new ImageTokenLayout(6) { Grid = [2, 2] }), "a grid that does not hold the tokens is refused");

        // The M-RoPE slot: position ids reach the decoder, which refuses them (no multi-axis rotary embedding yet).
        Idrak.PluginTests.LlavaPlugin.Register();
        using var model = PretrainedModel.Load(LlavaData("tiny-llava"), new PretrainedOptions { Device = device });
        using var features = new ImageFeatures(Tensor.From(new float[16 * 32], [16, 32], device), new ImageTokenLayout(16) { Grid = [4, 4] },
            Tensor.From(new float[3 * 16], [3, 16], device));
        int[] prompt = [.. Enumerable.Repeat(1, 3), .. Enumerable.Repeat(image, 16), 5];
        var placed = ImagePrefill.Locate(prompt, image, [features]);
        Check(placed[0].Positions is not null, "position ids are carried to the prefill");
        using var input = Ids(prompt, device);
        try
        {
            model.Network.Forward(input, placed, ImageAttentionRules.Get(ImageAttentionRules.Causal)).Dispose();
            Check(false, "M-RoPE position ids are refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("M-RoPE", StringComparison.Ordinal), e.Message);
        }

        RegisterGemma3Vision();
    }
    /// <summary>Registers the Gemma 3 vision family (the sample plug-in's entry point), as an app or <c>-P</c> does.</summary>
    private static void RegisterGemma3Vision() => Gemma3VisionPlugin.Register();

    /// <summary>
    /// Removes the Gemma 3 vision family and its attention rule, and checks that the tiny Gemma 3 vision checkpoint then
    /// fails to load with the "not registered" error naming the registry (no fallback to any family).
    /// </summary>
    private static void Gemma3VisionNotRegistered(Device device)
    {
        VisionFamilies.Unregister(Gemma3VisionFamily.Architecture);
        ImageAttentionRules.Unregister(Gemma3VisionPlugin.ImageBlocks);
        Check(VisionFamilies.Find(Gemma3VisionFamily.Architecture) is null && VisionFamilies.Default(Gemma3VisionFamily.Architecture) is null
              && ImageAttentionRules.Find(Gemma3VisionPlugin.ImageBlocks) is null && ImageAttentionRules.Names.SequenceEqual([ImageAttentionRules.Causal]),
            "the library registers no vision family and no image attention rule but the causal one");
        try
        {
            using var _ = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
            Check(false, "a vision checkpoint whose family is not registered is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("Vision family 'Gemma3ForConditionalGeneration' is not registered", StringComparison.Ordinal)
                  && e.Message.Contains("VisionFamilies.Register", StringComparison.Ordinal), $"the error names the registry: {e.Message}");
        }

        Check(VisionFamilies.For("Gemma3ForCausalLM", new System.Text.Json.Nodes.JsonObject()) is null, "a text checkpoint needs no vision family");
    }
}
