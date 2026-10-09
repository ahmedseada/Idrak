// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

// Gemma 3's pan and scan through the Gemma 3 plug-in (plan 11, "Pan and scan"): one image becomes the whole image and
// its crops, each a block of image tokens with Gemma3Processor's text between them, matching transformers on the tiny
// model (tools/vlm/make_pan_scan.py → data/vlm-pan-scan); the settings as vision options, the model's
// preprocessor_config.json as the defaults.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PanAndScanGroup =
    [
        ("pan and scan: a tall and a wide image give transformers' crops, pixels, features, prompt ids, prompt logits and 20 greedy tokens (tiny Gemma 3, the plug-in's vision options); a near-square image gives exactly what pan and scan off gives", PanAndScanMatchesReference),
        ("pan and scan: the crop rule, the vision options (unknown keys name Gemma 3's, values checked), preprocessor_config.json as the defaults, the testing kit with crops", PanAndScanSettings),
    ];

    // The reference's settings: Gemma3Processor's defaults with 32-pixel crops (the tiny model reads 56-pixel images).
    private static readonly VisionOptions PanAndScanOn = new([KeyValuePair.Create("do_pan_and_scan", "true"), KeyValuePair.Create("pan_and_scan_min_crop_size", "32")]);

    private static void PanAndScanMatchesReference(Device device)
    {
        RegisterGemma3Vision();
        var manifest = JsonNode.Parse(File.ReadAllText(TestData("vlm-pan-scan/manifest.json")))!;
        var facts = manifest["facts"]!;
        const int vocabulary = 366, tokens = 4;
        using var noGrad = Autograd.NoGrad();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
        var vision = (Gemma3Vision)model.Vision!;
        using var encoder = vision.CreateEncoder(device);
        IVisionEncoder contract = encoder;
        var chat = model.CreateChat(KeyValueFormat.Float32, 256);
        var reader = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };
        foreach (string name in new[] { "tall", "wide", "square" })
        {
            var entry = facts["images"]![name]!;
            var image = ImageCodecs.Decode(TestData($"vlm-pan-scan/image-{name}.png"));
            int crops = (int)entry["num_crops"]!;

            // The crops: transformers' rectangles of the original, one block each after the whole image.
            var boxes = encoder.PanAndScan.With(PanAndScanOn).Crops(image.Height, image.Width).Select(c => new[] { c.Top, c.Left, c.Height, c.Width }).ToList();
            var expectedBoxes = entry["crops_top_left_height_width"]!.AsArray().Select(b => b!.AsArray().Select(v => (int)v!).ToArray()).ToList();
            Check(boxes.Count == crops && boxes.Zip(expectedBoxes).All(p => p.First.SequenceEqual(p.Second)),
                $"{name}: crops {string.Join("; ", boxes.Select(b => string.Join(",", b)))}");
            var blocks = contract.Blocks(image, PanAndScanOn);
            Check(blocks.Count == 1 + crops && blocks.All(b => b.Tokens == tokens), $"{name}: {blocks.Count} blocks");

            // Pixels: the whole image, then each crop resized on its own.
            using (var pixels = encoder.PixelValues(image, PanAndScanOn))
            {
                var expected = ReadNpyFloat32(TestData($"vlm-pan-scan/reference/pixel_values-{name}.npy"));
                Check(pixels.Shape[0] == 1 + crops, $"{name}: pixel values {Tensor.FormatShape(pixels.Shape)}");
                AssertClose(expected, pixels.ToArray(), 1e-5f, $"{name}: pixel values");
            }

            // Features: one per block.
            var features = contract.Encode([image], PanAndScanOn);
            try
            {
                Check(features.Count == 1 + crops, $"{name}: {features.Count} blocks' features");
                float[] own = [.. features.SelectMany(f => f.Features.ToArray())];
                float featureDifference = MaxDifference(ReadNpyFloat32(TestData($"vlm-pan-scan/reference/image_features-{name}.npy")), own);
                AssertClose(ReadNpyFloat32(TestData($"vlm-pan-scan/reference/image_features-{name}.npy")), own, 3e-5f, $"{name}: features");

                // The prompt: Gemma3Processor's text around each crop's block, the same ids.
                var request = new ChatRequest([new("system", (string)facts["system"]!), new("user", [ChatImage.FromFile(TestData($"vlm-pan-scan/image-{name}.png")), new ChatText((string)facts["question"]!)])])
                {
                    Options = new GenerationOptions { Temperature = 0, NumPredict = 20, RepeatPenalty = 1 },
                    VisionOptions = PanAndScanOn,
                };
                int[] ids = [.. entry["input_ids"]!.AsArray().Select(i => (int)i!)];
                var own_ids = model.Tokenizer!.Encode(reader.RenderPrompt(request));
                Check(own_ids.SequenceEqual(ids), $"{name}: prompt ids {string.Join(" ", own_ids)}");
                var located = ImagePrefill.Locate(ids, vision.ImageTokens.ImageToken, features);
                int[] starts = [.. entry["image_block_starts"]!.AsArray().Select(i => (int)i!)];
                Check(located.Select(i => i.Position).SequenceEqual(starts), $"{name}: blocks at {string.Join(", ", located.Select(i => i.Position))}");

                // Logits: one pass, the cached prefill, 20 greedy steps.
                float[] expectedLogits = ReadNpyFloat32(TestData($"vlm-pan-scan/reference/prompt-{name}-logits.npy"));
                float[] expectedSteps = ReadNpyFloat32(TestData($"vlm-pan-scan/reference/prompt-{name}-generate-logits.npy"));
                int[] newTokens = [.. entry["new_tokens"]!.AsArray().Select(i => (int)i!)];
                using var input = Ids(ids, device);
                var whole = model.Network.Forward(input, located, vision.Attention).ToArray();
                float wholeDifference = MaxDifference(expectedLogits, whole);
                AssertClose(expectedLogits, whole, 2e-5f, $"{name}: prompt logits without the cache");
                using var context = new DecodingContext(device, batch: 1, capacity: ids.Length + newTokens.Length + 1);
                var prefill = model.Network.ForwardCached(input, located, vision.Attention, context).ToArray();
                AssertClose(expectedLogits, prefill, 2e-5f, $"{name}: prompt logits of the cached prefill");
                var generated = new List<int>();
                var steps = new List<float>();
                float[] last = prefill[^vocabulary..];
                for (int step = 0; step < newTokens.Length; step++)
                {
                    steps.AddRange(last);
                    generated.Add(ArgMax(last));
                    using var next = Ids([generated[^1]], device);
                    last = model.Network.ForwardCached(next, context).ToArray();
                }

                Check(generated.SequenceEqual(newTokens), $"{name}: greedy tokens {string.Join(" ", generated)}, expected {string.Join(" ", newTokens)}");
                float stepDifference = MaxDifference(expectedSteps, [.. steps]);
                AssertClose(expectedSteps, [.. steps], 2e-5f, $"{name}: the greedy steps' logits");

                // The public chat API with the request's vision options: the same 20 tokens (their text up to the first byte
                // token that is not whole UTF-8, which the streamed text and a one-shot decode may show differently).
                var reply = reader.Chat(request);
                string answer = model.Tokenizer.Decode(newTokens).Trim(), wholeText = answer.Split('�')[0];
                Check(reply.Stats is { GeneratedTokens: 20 } && reply.Stats.PromptTokens == ids.Length && reply.Message!.Content.Trim().StartsWith(wholeText, StringComparison.Ordinal),
                    $"{name}: the chat API's answer '{reply.Message?.Content}' ({reply.Stats}), transformers' '{answer}'");

                if (crops == 0)
                {
                    // Not cropped: exactly what pan and scan off gives (pixels, features, ids, logits), bit for bit.
                    using var offPixels = encoder.PixelValues(image);
                    using var onPixels = encoder.PixelValues(image, PanAndScanOn);
                    Check(offPixels.ToArray().AsSpan().SequenceEqual(onPixels.ToArray()), $"{name}: the same pixels");
                    var off = contract.Encode([image]);
                    try
                    {
                        Check(off.Count == 1 && off[0].Features.ToArray().AsSpan().SequenceEqual(features[0].Features.ToArray()), $"{name}: the same features");
                        var offIds = model.Tokenizer.Encode(reader.RenderPrompt(request with { VisionOptions = null }));
                        Check(offIds.SequenceEqual(ids), $"{name}: the same prompt ids");
                        var offLogits = model.Network.Forward(input, ImagePrefill.Locate(ids, vision.ImageTokens.ImageToken, off), vision.Attention).ToArray();
                        Check(offLogits.AsSpan().SequenceEqual(whole), $"{name}: the same logits");
                    }
                    finally
                    {
                        off.ToList().ForEach(f => f.Dispose());
                    }
                }

                Console.WriteLine($"    {name} on {device}: {crops} crops, {ids.Length} prompt tokens; largest difference: features {featureDifference:G3}, "
                    + $"logits {wholeDifference:G3}, 20 greedy steps {stepDifference:G3}; tokens {string.Join(" ", generated)}");
            }
            finally
            {
                foreach (var f in features)
                {
                    f.Dispose();
                }
            }
        }
    }

    private static void PanAndScanSettings(Device device)
    {
        RegisterGemma3Vision();

        // The rule (transformers' pan_and_scan), Gemma3Processor's defaults: 256-pixel crops, 4 at most, ratio 1.2.
        var on = Gemma3PanAndScan.Default with { Enabled = true };
        Check(Gemma3PanAndScan.Default.Crops(1000, 700).Count == 0, "off by default");
        var scan = on.Crops(1000, 700);                    // a 700 x 1000 page: ratio 1.43 rounds to 1, at least 2 crops
        Check(scan is [(0, 0, 500, 700), (500, 0, 500, 700)], $"a 700 x 1000 page: {string.Join("; ", scan)}");
        Check(on.Crops(3508, 2480).Count == 2 && on.Crops(896, 896).Count == 0 && on.Crops(1100, 1000).Count == 0, "an A4 scan, a square, ratio 1.1");
        Check(on.Crops(1000, 4000) is { Count: 4 } wide && wide.All(c => c.Width == 1000 && c.Height == 1000), "ratio 4: four crops across");
        Check(on.Crops(1000, 9000).Count == 4, "ratio 9: no more than 4");
        Check(on.Crops(300, 500).Count == 0, "crops under 256 pixels: none");
        Check(on.Crops(1001, 600) is [(0, 0, 501, 600), (501, 0, 500, 600)], "the last crop cut at the edge");
        Check((on with { MaxCrops = 1 }).Crops(1000, 3000) is [(0, 0, 1000, 3000)], "max 1: the whole image as its one crop, as transformers does");

        // Options: transformers' names (and pan_and_scan for the switch); anything else is refused, naming Gemma 3's keys.
        var parsed = Gemma3PanAndScan.Default.With(VisionOptions.Parse(["pan_and_scan=true", "pan_and_scan_max_num_crops=2", "pan_and_scan_min_ratio_to_activate=1.5"]));
        Check(parsed is { Enabled: true, MaxCrops: 2, MinCropSize: 256, MinRatio: 1.5 }, $"parsed: {parsed}");
        foreach (var (options, words) in new[]
        {
            (VisionOptions.Parse(["crops=4"]), "do_pan_and_scan, pan_and_scan, pan_and_scan_min_crop_size"),
            (VisionOptions.Parse(["do_pan_and_scan=maybe"]), "true or false"),
            (VisionOptions.Parse(["pan_and_scan_max_num_crops=0"]), "at least 1"),
            (VisionOptions.Parse(["pan_and_scan_min_crop_size=1.5"]), "whole number"),
            (VisionOptions.Parse(["do_pan_and_scan=true", "pan_and_scan=false"]), "disagree"),
        })
        {
            try
            {
                Gemma3PanAndScan.Default.With(options);
                Check(false, $"{options} is refused");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains(words, StringComparison.Ordinal), $"{options}: {e.Message}");
            }
        }

        // The model folder's preprocessor_config.json sets the defaults (null keys: Gemma3Processor's); the encoder's
        // options go over them, and a request's over those.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-pan-scan-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            foreach (string file in Directory.GetFiles(TestData("vlm/tiny-gemma3-v5")))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "preprocessor_config.json")))!.AsObject();
            config["do_pan_and_scan"] = true;
            config["pan_and_scan_min_crop_size"] = 32;
            File.WriteAllText(Path.Combine(folder, "preprocessor_config.json"), config.ToJsonString());
            var tall = ImageCodecs.Decode(TestData("vlm-pan-scan/image-tall.png"));
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var vision = (Gemma3Vision)model.Vision!;
            Check(vision.PanAndScan is { Enabled: true, MinCropSize: 32, MaxCrops: 4, MinRatio: 1.2 } && vision.VisionOptionKeys.Contains("do_pan_and_scan"),
                $"from preprocessor_config.json: {vision.PanAndScan}");
            using (var fromConfig = vision.CreateEncoder(new VisionEncoderOptions { Device = device }))
            {
                Check(fromConfig.Blocks(tall).Count == 4 && fromConfig.Blocks(tall, VisionOptions.Parse(["do_pan_and_scan=false"])).Count == 1
                      && fromConfig.Blocks(tall, VisionOptions.Parse(["pan_and_scan_max_num_crops=2"])).Count == 3, "the config's defaults, a request's options over them");

                // The same features as the reference's (with the request options it was made with).
                var features = fromConfig.Encode([tall]);
                try
                {
                    AssertClose(ReadNpyFloat32(TestData("vlm-pan-scan/reference/image_features-tall.npy")), [.. features.SelectMany(f => f.Features.ToArray())], 3e-5f,
                        "the config's pan and scan: transformers' features");
                }
                finally
                {
                    features.ToList().ForEach(f => f.Dispose());
                }
            }

            using (var off = vision.CreateEncoder(new VisionEncoderOptions { Device = device, VisionOptions = VisionOptions.Parse(["do_pan_and_scan=false"]) }))
            {
                Check(off.Blocks(tall).Count == 1 && off.Blocks(tall, VisionOptions.Parse(["pan_and_scan=true"])).Count == 4, "the encoder's options over the config, a request's over those");
            }

            try
            {
                using var _ = vision.CreateEncoder(new VisionEncoderOptions { Device = device, VisionOptions = VisionOptions.Parse(["tiles=2"]) });
                Check(false, "an unknown option is refused when the encoder is built");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains("Gemma3ForConditionalGeneration", StringComparison.Ordinal) && e.Message.Contains("pan_and_scan_max_num_crops", StringComparison.Ordinal), e.Message);
            }

            // A null do_pan_and_scan (the real checkpoints') keeps it off.
            config["do_pan_and_scan"] = null;
            File.WriteAllText(Path.Combine(folder, "preprocessor_config.json"), config.ToJsonString());
            using var plain = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            Check(((Gemma3Vision)plain.Vision!).PanAndScan is { Enabled: false, MinCropSize: 32 }, "null: off");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        // The testing kit with crops: every block deterministic, counted, the same alone and in a batch and on the CPU.
        using var tiny = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
        using var encoder = tiny.Vision!.CreateEncoder(new VisionEncoderOptions { Device = device });
        using var cpu = tiny.Vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
        var samples = new[] { ImageCodecs.Decode(TestData("vlm-pan-scan/image-tall.png")), ImageCodecs.Decode(TestData("vlm-pan-scan/image-wide.png")) };
        var report = Idrak.Abstraction.Testing.Conformance.CheckVisionEncoder(encoder, cpu, samples, visionOptions: PanAndScanOn);
        Console.WriteLine($"    the kit with pan and scan on {device}: {report.Cases} cases, {report.Entries.Count(e => e.Status == Idrak.Abstraction.Testing.CheckStatus.Passed)} properties pass");
        report.ThrowIfFailed();
    }
}
