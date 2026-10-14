// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Data;
using Idrak.IdrakVision;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp.Generation;

// Plan 19, phases B1 and B2: vision-language models made from scratch, IdrakVision as the plug-in that makes and reads them.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VlmScratchGroup =
    [
        ("vlm scratch: IdrakVision reads its folder; pixel values, tower outputs and features are PyTorch's, the encoder passes the contract", VlmScratchRead),
        ("vlm scratch: the chat encoder gives each record's ids, trained mask and image blocks, and the decoder with images gives PyTorch's logits", VlmScratchRecords),
        ("vlm scratch: a model made from a vision description has the family's tensors and shapes, loads, answers about an image, and saves its vision part back", VlmScratchCreate),
        ("vlm scratch: without its plug-in the family is not registered; a description the family cannot make and a family that reads only are refused", VlmScratchRefusals),
    ];

    private static string ScratchData(string relative) => TestData($"vlm-scratch/{relative}");

    private static float[] Npy(string relative) => ReadNpyFloat32(ScratchData(relative));

    private static PretrainedModel LoadScratch(Device device) => PretrainedModel.Load(ScratchData("tiny-idrak-vision"), new PretrainedOptions { Device = device });

    private static void VlmScratchRead(Device device)
    {
        IdrakVisionPlugin.Register();
        try
        {
            using var model = LoadScratch(device);
            Check(model.Vision is IdrakVisionPart, $"the vision part is IdrakVision's: {model.Vision}");
            var vision = (IdrakVisionPart)model.Vision!;
            Check(vision.Tokens.TokensPerImage == 4 && vision.Attention.Name == IdrakVisionPlugin.ImageBlocks, vision.Describe());
            using var encoder = (IdrakVisionEncoder)vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            IVisionEncoderStages stages = encoder;
            foreach (string stem in new[] { "image", "image-palette" })
            {
                var image = ImageCodecs.Decode(ScratchData($"images/{stem}.png"));
                using var pixels = stages.PixelValues(image);
                AssertClose(Npy($"reference/{stem}-pixel_values.npy"), pixels.ToArray(), 1e-5f, $"{stem}: pixel values");
                using var tower = stages.Tower(pixels);
                AssertClose(Npy($"reference/{stem}-tower.npy"), tower.ToArray(), 1e-4f, $"{stem}: the tower's output");
                using var features = stages.Features(pixels);
                AssertClose(Npy($"reference/{stem}-features.npy"), features.ToArray(), 1e-4f, $"{stem}: the features");
            }

            using var cpu = vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
            var samples = new[] { ImageCodecs.Decode(ScratchData("images/image.png")), ImageCodecs.Decode(ScratchData("images/image-palette.png")) };
            Idrak.Abstraction.Testing.Conformance.CheckVisionEncoder(encoder, cpu, samples).ThrowIfFailed();
            Check(VisionFamilies.Origin(IdrakVisionFamily.Architecture) is { } origin && origin != Idrak.Abstraction.Overrides.Library
                && VisionFamilies.Default(IdrakVisionFamily.Architecture) is null, "the family is the plug-in's, not the library's");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
        }
    }

    private static void VlmScratchRecords(Device device)
    {
        IdrakVisionPlugin.Register();
        try
        {
            using var model = LoadScratch(device);
            using var vision = TuningVision.Create(model);
            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Vision = vision };
            var expected = JsonNode.Parse(File.ReadAllText(ScratchData("tuning.json")))!["records"]!.AsArray();
            using var records = TuningDataFormats.Get("messages").ReadPending(ScratchData("records.jsonl"));
            Check(records.Count == expected.Count, $"{records.Count} records");
            var sequences = new List<TrainingSequence>();
            for (int r = 0; r < records.Count; r++)
            {
                var sequence = encoder.Encode(records.Read(r), 128)!;
                sequences.Add(sequence);
                var ids = expected[r]!["input_ids"]!.AsArray().Select(i => (int)i!).ToArray();
                var trained = expected[r]!["trained"]!.AsArray().Select(t => (bool)t!).ToArray();
                Check(sequence.Tokens.SequenceEqual(ids), $"record {r}: ids [{string.Join(" ", sequence.Tokens)}] against [{string.Join(" ", ids)}]");
                Check(sequence.Trained.SequenceEqual(trained), $"record {r}: the trained mask [{string.Join("", sequence.Trained.Select(t => t ? 1 : 0))}] against [{string.Join("", trained.Select(t => t ? 1 : 0))}]");
                var blocks = expected[r]!["image_blocks"]!.AsArray().Select(b => ((int)b![0]!, (int)b[1]!)).ToArray();
                Check(sequence.Images.SelectMany(i => i.Blocks).SequenceEqual(blocks), $"record {r}: image blocks");
            }

            // The first record's logits through the decoder with its image's features.
            var first = sequences[0];
            using var features = ((IdrakVisionEncoder)vision.Encoder).Encode([ImageCodecs.Decode(ScratchData("images/image.png"))]);
            using var ids0 = Tensor.From(first.Tokens.Select(t => (float)t).ToArray(), [1, first.Tokens.Length], device);
            var images = first.Images[0].Blocks.Select(b => new PromptImage(b.Position, features.Reshape(4, model.Spec.Dim))).ToList();
            using var logits = model.Network.Forward(ids0, images, vision.Attention);
            AssertClose(Npy("reference/record0-logits.npy"), logits.ToArray(), 1e-4f, "the first record's logits");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
        }
    }

    private static JsonObject ScratchDescription()
    {
        var config = JsonNode.Parse(File.ReadAllText(ScratchData("tiny-idrak-vision/config.json")))!.AsObject();
        return new JsonObject
        {
            ["vision_config"] = config["vision_config"]!.DeepClone(),
            ["projector_config"] = config["projector_config"]!.DeepClone(),
            ["image_attention"] = "image-blocks",
        };
    }

    private static void VlmScratchCreate(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("vlm-scratch-create");
        try
        {
            var config = JsonNode.Parse(File.ReadAllText(ScratchData("tiny-idrak-vision/config.json")))!.AsObject();
            var spec = PretrainedArchitectures.Get(IdrakVisionFamily.Architecture).Spec(config, []);
            string made = Path.Combine(folder, "made");
            using (var model = PretrainedModel.Create(made, spec, IdrakVisionFamily.Architecture, new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")), BosToken = "<s>", PadToken = "<pad>", StopTokens = ["</s>"] },
                new CreateOptions { Device = device, Seed = 4, Vision = ScratchDescription() }))
            {
                var expected = ReadTensors(ScratchData("tiny-idrak-vision/model.safetensors"));
                var created = ReadTensors(Path.Combine(made, "model.safetensors"));
                Check(created.Keys.Order(StringComparer.Ordinal).SequenceEqual(expected.Keys.Order(StringComparer.Ordinal)),
                    $"the family's tensor names: {string.Join(", ", created.Keys.Except(expected.Keys).Take(5))} | {string.Join(", ", expected.Keys.Except(created.Keys).Take(5))}");
                Check(created.All(t => t.Value.Shape.SequenceEqual(expected[t.Key].Shape)), "and shapes");
                var madeConfig = JsonNode.Parse(File.ReadAllText(Path.Combine(made, "config.json")))!.AsObject();
                foreach (string key in new[] { "image_token_index", "boi_token_index", "eoi_token_index", "mm_tokens_per_image", "image_attention" })
                {
                    Check(JsonNode.DeepEquals(madeConfig[key], config[key]), $"{key}: {madeConfig[key]} against {config[key]}");
                }

                Check(model.Vision is IdrakVisionPart && File.Exists(Path.Combine(made, "preprocessor_config.json")), "the made model reads back with its vision part and preprocessing");

                // It answers about an image (nonsense, as expected).
                using var encoder = model.CreateVisionEncoder(new VisionEncoderOptions { Device = device });
                var chat = new ChatGenerator(model.CreateChat(KeyValueFormat.Float32, 128).Generator, model.ChatTemplate)
                {
                    Images = new ChatImages(encoder, model.Vision!.PromptFormat, model.Vision.Attention),
                };
                var message = new ChatMessage("user", [ChatImage.FromFile(ScratchData("images/image.png")), new ChatText("read the scan")]);
                var answer = chat.Chat(new ChatRequest([message], Options: new GenerationOptions { Temperature = 0, NumPredict = 3 }));
                Check(answer.Done, "a made vision-language model answers about an image");
            }

            // A loaded model saved as a checkpoint writes its vision part too, as stored.
            string saved = Path.Combine(folder, "saved");
            using (var model = LoadScratch(device))
            {
                model.SaveHuggingFace(saved, SafeTensorType.F32);
            }

            var original = ReadTensors(ScratchData("tiny-idrak-vision/model.safetensors"));
            var written = ReadTensors(Path.Combine(saved, "model.safetensors"));
            Check(written.Count == original.Count && written.All(t => original[t.Key].Values.AsSpan().SequenceEqual(t.Value.Values)),
                $"every tensor, the vision part's included, written back as read ({written.Count} of {original.Count})");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void VlmScratchRefusals(Device device)
    {
        if (!FirstRun(nameof(VlmScratchRefusals)))
        {
            return;
        }

        string message = "";
        try
        {
            using var _ = LoadScratch(Device.Cpu);
        }
        catch (Exception e) when (e is NotSupportedException)
        {
            message = e.Message;
        }

        Check(message.Contains(IdrakVisionFamily.Architecture, StringComparison.Ordinal) && message.Contains("is registered", StringComparison.Ordinal)
              && message.Contains("Register", StringComparison.Ordinal), $"without the plug-in, the family is refused by name: {message}");
        IdrakVisionPlugin.Register();
        string folder = TempFolder("vlm-scratch-refusals");
        try
        {
            var config = JsonNode.Parse(File.ReadAllText(ScratchData("tiny-idrak-vision/config.json")))!.AsObject();
            var spec = PretrainedArchitectures.Get(IdrakVisionFamily.Architecture).Spec(config, []);
            var files = new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")) };
            var bad = ScratchDescription();
            bad["vision_config"]!["patch_size"] = 5;
            Throws<InvalidDataException>(() => PretrainedModel.Create(Path.Combine(folder, "a"), spec, IdrakVisionFamily.Architecture, files, new CreateOptions { Vision = bad }), "28 is not a whole number of 5-pixel patches");
            var noToken = ScratchDescription();
            noToken["image_token"] = "<picture>";
            Throws<InvalidDataException>(() => PretrainedModel.Create(Path.Combine(folder, "b"), spec, IdrakVisionFamily.Architecture, files, new CreateOptions { Vision = noToken }), "a token the tokenizer lacks");
            Throws<NotSupportedException>(() => PretrainedModel.Create(Path.Combine(folder, "c"), spec, "LlamaForCausalLM", files, new CreateOptions { Vision = ScratchDescription() }), "a text family has no vision family");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }
}
