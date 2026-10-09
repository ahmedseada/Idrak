// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

namespace Idrak.PluginTests;

/// <summary>
/// A vision family of its own, registered from outside the library (plan 11, "Vision contracts"): a trivial encoder (each
/// image's mean colour per band of rows, through a fixed matrix: one token per band, the bands one per 16 rows, so the
/// count depends on the image: dynamic resolution), its own prompt format and its own attention rule (its image tokens
/// see each other). Its text decoder is LLaVA's tiny one (this assembly's <see cref="LlavaPlugin"/>), under its own
/// architecture name. With its vision option <c>halves=true</c> an image becomes two blocks of image tokens (its top
/// half and its bottom half, " and " between them in the prompt): one image as several blocks with text between, as a
/// family that adds views of an image does. No library change was needed for it.
/// </summary>
public static class VisionFamilyPluginTests
{
    /// <summary>The architecture name the outside family is registered under.</summary>
    public const string Architecture = "OutsideToyVlmForConditionalGeneration";

    /// <summary>Its attention rule's name: image tokens see their whole image.</summary>
    public const string Rule = "outside-whole-image";

    /// <summary>
    /// Registers the family (text and vision) and its rule, loads a copy of the tiny LLaVA folder renamed to it, checks the
    /// encoder with the testing kit, answers a prompt with two images of different sizes through the public chat API, and
    /// unregisters everything: unregistered, the checkpoint is refused naming the registry.
    /// </summary>
    public static void OutsideFamily(Device device)
    {
        string source = Path.Combine(DataFolder(), "vlm-llava", "tiny-llava");
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-outside-vlm-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            foreach (string file in Directory.GetFiles(source))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            config["architectures"] = new JsonArray(Architecture);
            File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());

            LlavaPlugin.Register();
            PretrainedArchitectures.Register(Architecture, PretrainedArchitectures.Get(LlavaPlugin.Architecture));
            try
            {
                Load(folder, device).Dispose();
                Check(false, "an unregistered vision family is refused");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains($"Vision family '{Architecture}' is not registered", StringComparison.Ordinal) && e.Message.Contains("VisionFamilies.Register", StringComparison.Ordinal),
                    $"refused naming the registry: {e.Message}");
            }

            ImageAttentionRules.Register(new WholeImageRule());
            VisionFamilies.Register(new ToyFamily());
            Check(VisionFamilies.Origin(Architecture) == typeof(ToyFamily).Assembly.GetName().Name && VisionFamilies.Default(Architecture) is null,
                "registered from this assembly, with no library default");

            using var model = Load(folder, device);
            var vision = model.Vision as ToyVision ?? throw new InvalidOperationException("the outside family read the vision part");
            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            using var cpu = vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
            Conformance.CheckVisionEncoder(encoder, cpu).ThrowIfFailed();
            var halves = new VisionOptions([KeyValuePair.Create("halves", "true")]);
            Conformance.CheckVisionEncoder(encoder, cpu, visionOptions: halves).ThrowIfFailed();
            Check(vision.VisionOptionKeys.SequenceEqual(["halves"]), "the family names its option");

            // A prompt with two images of different sizes: 2 and 4 tokens, the outside rule, through the public chat API.
            var chat = model.CreateChat(KeyValueFormat.Float32, 128);
            var withImages = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };
            var small = Image(40, 24, 0.2f);
            var tall = Image(64, 30, 0.7f);
            var request = new ChatRequest([new ChatMessage("user", [small, tall, new ChatText("What is in this image?")])])
            {
                Options = new GenerationOptions { Temperature = 0, NumPredict = 6, RepeatPenalty = 1 },
            };
            var ids = model.Tokenizer!.Encode(withImages.RenderPrompt(request));
            Check(ids.Count(i => i == vision.PromptFormat.ImageToken) == 2 + 4, $"the prompt holds 2 + 4 image tokens: {string.Join(" ", ids)}");
            var reply = withImages.Chat(request);
            Check(reply.Done && reply.Stats is { GeneratedTokens: 6 }, $"a reply of 6 tokens: {reply.Message?.Content}");
            var swapped = withImages.Chat(request with { Messages = [new ChatMessage("user", [tall, small, new ChatText("What is in this image?")])] });
            Check(swapped.Done && swapped.Stats is { GeneratedTokens: 6 }, "the images in the other order");

            // Per request, each image in two blocks (1 + 1 and 2 + 2 tokens) with the family's text between them: four runs.
            var split = request with { VisionOptions = halves };
            var splitIds = model.Tokenizer!.Encode(withImages.RenderPrompt(split));
            int runs = splitIds.Where((id, i) => id == vision.PromptFormat.ImageToken && (i == 0 || splitIds[i - 1] != id)).Count();
            Check(splitIds.Count(i => i == vision.PromptFormat.ImageToken) == 2 + 4 && runs == 4, $"two blocks per image, four runs: {string.Join(" ", splitIds)}");
            var splitReply = withImages.Chat(split);
            Check(splitReply.Done && splitReply.Stats is { GeneratedTokens: 6 } && splitReply.Stats.PromptTokens == splitIds.Count, "a reply with each image in two blocks");
            try
            {
                withImages.Chat(request with { VisionOptions = new VisionOptions([KeyValuePair.Create("quarters", "true")]) });
                Check(false, "an unknown vision option is refused");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains("quarters", StringComparison.Ordinal) && e.Message.Contains("halves", StringComparison.Ordinal), $"refused naming the keys it takes: {e.Message}");
            }
        }
        finally
        {
            VisionFamilies.Unregister(Architecture);
            ImageAttentionRules.Unregister(Rule);
            PretrainedArchitectures.Unregister(Architecture);
            LlavaPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static PretrainedModel Load(string folder, Device device) => PretrainedModel.Load(folder, new PretrainedOptions { Device = device });

    // A binary PPM (read by the library's Netpbm codec) of a shade with a gradient across, height x width.
    private static ChatImage Image(int height, int width, float shade)
    {
        var header = System.Text.Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n");
        var bytes = new byte[header.Length + 3 * height * width];
        header.CopyTo(bytes, 0);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    bytes[header.Length + (y * width + x) * 3 + c] = (byte)Math.Round(255 * Math.Clamp(shade + 0.3f * x / width - 0.1f * c + 0.2f * y / height, 0f, 1f));
                }
            }
        }

        return ChatImage.FromBytes(bytes, "image/x-portable-pixmap");
    }

    // tests/Idrak.Tests/data, found from the build output folder.
    private static string DataFolder()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string data = Path.Combine(directory.FullName, "tests", "Idrak.Tests", "data");
            if (Directory.Exists(Path.Combine(data, "vlm-llava")))
            {
                return data;
            }
        }

        throw new DirectoryNotFoundException("tests/Idrak.Tests/data/vlm-llava");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class WholeImageRule : IImageAttentionRule
    {
        public string Name => Rule;

        public bool Causal => false;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.ImageBlocks(rows, blocks, window);
    }

    private sealed class ToyFamily : IVisionFamily
    {
        public string Name => Architecture;

        public PretrainedVision? Read(VisionCheckpoint checkpoint)
        {
            int imageToken = (int?)checkpoint.Config["image_token_index"] ?? throw new InvalidDataException("no image_token_index");
            int width = PretrainedArchitectures.Get(checkpoint.Architecture).Spec(checkpoint.Config, []).Dim;
            return new ToyVision(width, new ImageTokenFormat("outside-toy", imageToken, imageToken) { Join = blocks => string.Join(" and ", blocks) });
        }
    }

    private sealed class ToyVision(int width, IImagePromptFormat format) : PretrainedVision
    {
        public override string Family => Architecture;

        public override int Width => width;

        public override IImagePromptFormat PromptFormat => format;

        public override IImageAttentionRule Attention => ImageAttentionRules.Get(Rule);

        public override IReadOnlyCollection<string> StoredTensors => [];

        public override IReadOnlyCollection<string> VisionOptionKeys => [ToyEncoder.Halves];

        public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null) => new ToyEncoder(width, options?.Device ?? Device.Default, options?.VisionOptions);
    }

    // One token per 16 rows (at least one): the mean of each channel over the band, times a fixed [3, width] matrix.
    // With halves=true, the top and the bottom half of an image (at least 2 rows) are two blocks.
    private sealed class ToyEncoder(int width, Device device, VisionOptions? defaults) : IVisionEncoder
    {
        public const string Halves = "halves";

        public int Width => width;

        public Device Device => device;

        public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null) => [.. Parts(image, options).Select(Layout)];

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null) =>
            [.. images.SelectMany(image => Parts(image, options)).Select(Encode)];

        private static ImageTokenLayout Layout(ImageData image) => new(Math.Max(1, image.Height / 16)) { Grid = [Math.Max(1, image.Height / 16), 1] };

        private IReadOnlyList<ImageData> Parts(ImageData image, VisionOptions? options)
        {
            var all = (defaults ?? VisionOptions.Empty).With(options);
            all.ThrowIfUnknown(Architecture, [Halves]);
            if (!all.Flag(Halves, false) || image.Height < 2)
            {
                return [image];
            }

            int top = image.Height / 2;
            return [Rows(image, 0, top), Rows(image, top, image.Height - top)];
        }

        private static ImageData Rows(ImageData image, int from, int count)
        {
            var pixels = new float[image.Channels * count * image.Width];
            for (int c = 0; c < image.Channels; c++)
            {
                Array.Copy(image.Pixels, (c * image.Height + from) * image.Width, pixels, c * count * image.Width, count * image.Width);
            }

            return new ImageData(pixels, image.Channels, count, image.Width);
        }

        private ImageFeatures Encode(ImageData image)
        {
            var layout = Layout(image);
            int bands = layout.Tokens, rows = image.Height / bands, channels = image.Channels;
            var values = new float[bands * width];
            for (int b = 0; b < bands; b++)
            {
                for (int c = 0; c < 3; c++)
                {
                    double sum = 0;
                    int plane = Math.Min(c, channels - 1);
                    for (int y = b * rows; y < (b + 1) * rows; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            sum += image.Pixels[(plane * image.Height + y) * image.Width + x];
                        }
                    }

                    float mean = (float)(sum / (rows * image.Width));
                    for (int j = 0; j < width; j++)
                    {
                        values[b * width + j] += mean * MathF.Sin(1 + c * 7 + j * 0.37f);
                    }
                }
            }

            return new ImageFeatures(Tensor.From(values, [bands, width], device), layout);
        }

        public void Dispose()
        {
        }
    }
}
