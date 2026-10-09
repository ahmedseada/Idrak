// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Nlp;

// Plan 11, image transforms: the contract and registry (Idrak.Abstraction.Data), the library's transforms against
// Pillow 12.3 byte for byte, and the JPEG encoder against libjpeg-turbo 3.1 (tools/vlm/make_image_transforms.py writes
// tests/Idrak.Tests/data/image-transforms).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ImageTransformGroup =
    [
        ("image transforms: grayscale, max_width/max_height (every resample), contrast, brightness, sharpness, autocontrast and jpeg give Pillow 12.3's bytes exactly (fixtures and a scan-like page)", TransformsMatchPillow),
        ("image transforms: the JPEG encoder's files decode to the pixels of Pillow's at the same quality (grey, 4:2:0, 4:2:2, 4:4:4, odd sizes, qualities 10 to 100)", JpegEncoderMatchesPillow),
        ("image transforms: pipelines parse from text and JSON in the user's order; unknown names, options and bad values name what is registered", TransformPipelineParsing),
        ("image transforms: the library's are library defaults; an app's transform registers from outside, runs in a pipeline and unregisters", TransformRegistry),
        ("image transforms: a chat request's transforms give transformers' pixels, features and 20 greedy tokens on the same Pillow-transformed image (tiny Gemma 3, compare_real.py --image-transform)", TransformedRequestMatchesTransformers),
    ];

    private static string TransformFixtures => Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "image-transforms");

    private static JsonObject TransformManifest() => JsonNode.Parse(File.ReadAllText(Path.Combine(TransformFixtures, "manifest.json")))!.AsObject();

    // An input of the manifest, decoded as a chat image is (codecs, EXIF upright).
    private static ImageData TransformInput(string name)
    {
        string path = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", name.Split('#')[0].Replace('/', Path.DirectorySeparatorChar));
        var image = ChatImageDecoder.Decode(File.ReadAllBytes(path));
        return name.EndsWith("#L", StringComparison.Ordinal) ? ChatImageDecoder.Grayscale(image) : image;
    }

    // The 8-bit planes of an image, concatenated (Pillow's bytes in planar order).
    private static byte[] PlanarBytes(ImageData image)
    {
        var bytes = new byte[image.Pixels.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(image.Pixels[i] * 255f), 0, 255);
        }

        return bytes;
    }

    private static void TransformsMatchPillow(Device device)
    {
        _ = device;
        var manifest = TransformManifest();
        var cases = manifest["transforms"]!.AsArray();
        Check(cases.Count >= 200, $"{cases.Count} cases");
        var inputs = new Dictionary<string, ImageData>();
        var failures = new List<string>();
        foreach (var node in cases)
        {
            var entry = node!.AsObject();
            string input = (string)entry["input"]!, pipeline = (string)entry["pipeline"]!;
            if (!inputs.TryGetValue(input, out var image))
            {
                inputs[input] = image = TransformInput(input);
            }

            var output = ImageTransformPipeline.Parse(pipeline).Apply(image);
            int channels = (string)entry["mode"]! == "L" ? 1 : 3;
            if (output.Channels != channels || output.Width != (int)entry["width"]! || output.Height != (int)entry["height"]!)
            {
                failures.Add($"{input} | {pipeline}: {output.Channels} x {output.Height} x {output.Width}, Pillow {channels} x {entry["height"]} x {entry["width"]}");
                continue;
            }

            var bytes = PlanarBytes(output);
            if (Convert.ToHexStringLower(SHA256.HashData(bytes)) == (string)entry["sha256"]!)
            {
                continue;
            }

            string worst = "";
            if (entry["png"] is { } png)
            {
                var expected = PlanarBytes(ImageCodecs.Decode(Path.Combine(TransformFixtures, (string)png!)));
                int most = 0, count = 0;
                for (int i = 0; i < bytes.Length; i++)
                {
                    int d = Math.Abs(bytes[i] - expected[i]);
                    most = Math.Max(most, d);
                    count += d > 0 ? 1 : 0;
                }

                worst = $" (largest difference {most}, {count} of {bytes.Length} bytes)";
            }

            failures.Add($"{input} | {pipeline}: differs from Pillow{worst}");
        }

        Check(failures.Count == 0, $"{failures.Count} of {cases.Count} differ from Pillow: {string.Join("; ", failures.Take(12))}");
    }

    private static void JpegEncoderMatchesPillow(Device device)
    {
        _ = device;
        var cases = TransformManifest()["jpeg"]!.AsArray();
        Check(cases.Count >= 40, $"{cases.Count} JPEG cases");
        var failures = new List<string>();
        int identical = 0;
        foreach (var node in cases)
        {
            var entry = node!.AsObject();
            string input = (string)entry["input"]!, sub = (string)entry["subsampling"]!;
            int quality = (int)entry["quality"]!;
            var image = TransformInput(input);
            var subsampling = sub switch { "4:4:4" => JpegSubsampling.Full444, "4:2:2" => JpegSubsampling.Half422, _ => JpegSubsampling.Half420 };
            var file = JpegEncoder.Encode(image, quality, optimize: true, subsampling);
            identical += Convert.ToHexStringLower(SHA256.HashData(file)) == (string)entry["sha256"]! ? 1 : 0;
            var decoded = ImageCodecs.Decode(file);
            var bytes = PlanarBytes(decoded);
            if (Convert.ToHexStringLower(SHA256.HashData(bytes)) == (string)entry["decoded_sha256"]!)
            {
                continue;
            }

            string worst = "";
            if (entry["file"] is { } pillow)
            {
                var expected = PlanarBytes(ImageCodecs.Decode(Path.Combine(TransformFixtures, (string)pillow!)));
                int most = 0;
                for (int i = 0; i < Math.Min(bytes.Length, expected.Length); i++)
                {
                    most = Math.Max(most, Math.Abs(bytes[i] - expected[i]));
                }

                worst = $" (largest difference {most})";
            }

            failures.Add($"{input} q{quality} {sub}: decoded pixels differ from Pillow's file{worst}");
        }

        Check(failures.Count == 0, $"{failures.Count} of {cases.Count}: {string.Join("; ", failures.Take(12))}");

        // Optimized Huffman tables change the size, never the pixels; the files are libjpeg-turbo's, byte for byte.
        Check(identical == cases.Count, $"{identical} of {cases.Count} files are byte for byte Pillow's");
        var sample = TransformInput("vlm/image.png");
        var plain = ImageCodecs.Decode(JpegEncoder.Encode(sample, 90));
        var optimized = ImageCodecs.Decode(JpegEncoder.Encode(sample, 90, optimize: true));
        Check(PlanarBytes(plain).SequenceEqual(PlanarBytes(optimized)), "optimize changed the pixels");
        Check(JpegEncoder.Encode(sample, 90, optimize: true).Length < JpegEncoder.Encode(sample, 90).Length, "optimized tables are not smaller");
    }

    private static void TransformPipelineParsing(Device device)
    {
        _ = device;
        var card = ImageTransformPipeline.Parse("grayscale, max_width=1024 ,contrast=1.5");
        Check(card.ToString() == "grayscale,max_width=1024,contrast=1.5" && card.Steps.Count == 3, $"the card's pipeline: {card}");
        Check(card.Steps[1].Integer() == 1024 && card.Steps[2].Number() == 1.5, "values");
        var resample = ImageTransformPipeline.Parse("max_width=1024,resample=bicubic,contrast=1.5,jpeg=95");
        Check(resample.Steps.Count == 3 && resample.Steps[0].Option("resample") == "bicubic" && resample.ToString() == "max_width=1024,resample=bicubic,contrast=1.5,jpeg=95",
            $"an option belongs to the step before it: {resample}");
        Check(ImageTransformPipeline.Parse("MAX_WIDTH=10,Resample=Box").Steps[0].Option("resample") == "Box", "names and keys ignore case");
        Check(ImageTransformPipeline.Parse(null).IsEmpty && ImageTransformPipeline.Parse(" none ").IsEmpty && ImageTransformPipeline.Parse("").IsEmpty, "empty");

        // The user's order.
        var order = ImageTransformPipeline.Parse("contrast=1.5,grayscale");
        Check(order.Steps[0].Name == "contrast" && order.Steps[1].Name == "grayscale", "order kept");
        var rgb = new ImageData([.. Enumerable.Range(0, 3 * 4 * 5).Select(i => i * 37 % 256 / 255f)], 3, 4, 5);
        Check(!PlanarBytes(order.Apply(rgb)).SequenceEqual(PlanarBytes(ImageTransformPipeline.Parse("grayscale,contrast=1.5").Apply(rgb))), "order changes the result");

        // JSON: a string, or an array of strings and objects.
        Check(ImageTransformPipeline.FromJson(JsonValue.Create("grayscale,contrast=1.5")).ToString() == "grayscale,contrast=1.5", "JSON string");
        var array = ImageTransformPipeline.FromJson(JsonNode.Parse("""["grayscale", {"name": "max_width", "value": 1024, "resample": "lanczos"}, {"name": "contrast", "value": 1.5}]"""));
        Check(array.ToString() == "grayscale,max_width=1024,resample=lanczos,contrast=1.5", $"JSON array: {array}");
        Check(ImageTransformPipeline.FromJson(null).IsEmpty, "JSON null");
        Check(array.Equals(ImageTransformPipeline.Parse(array.ToString())) && ImageTransformPipeline.Parse(array.ToJson()!.GetValue<string>()).Equals(array), "round trip");
        Check(card.Then(ImageTransformPipeline.Parse("jpeg=95")).ToString() == "grayscale,max_width=1024,contrast=1.5,jpeg=95", "Then");

        void Refused(Func<object> parse, string words, string what)
        {
            try
            {
                parse();
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains(words, StringComparison.Ordinal), $"{what}: {e.Message}");
                return;
            }

            throw new InvalidOperationException($"{what}: accepted");
        }

        Refused(() => ImageTransformPipeline.Parse("grayscale,blur=2"), "Unknown image transform 'blur' (registered: grayscale, max_width, max_height, contrast", "unknown name");
        Refused(() => ImageTransformPipeline.Parse("max_width=1024,resample=bicubic,colour=2"), "Unknown image transform 'colour'", "unknown after a step");
        Refused(() => ImageTransformPipeline.Parse("resample=bicubic"), "Unknown image transform 'resample'", "an option without its step");
        Refused(() => ImageTransformPipeline.Parse("contrast=1.5,resample=bicubic"), "Unknown image transform 'resample'", "an option of another step");
        Refused(() => ImageTransformPipeline.Parse("max_width=1024,resample=nearest"), "lanczos, bicubic, bilinear, box or hamming", "a bad resample");
        Refused(() => ImageTransformPipeline.Parse("max_width"), "max_width needs a value", "a missing value");
        Refused(() => ImageTransformPipeline.Parse("max_width=-3"), "positive", "a negative size");
        Refused(() => ImageTransformPipeline.Parse("contrast=high"), "contrast takes a number, not 'high'", "a bad number");
        Refused(() => ImageTransformPipeline.Parse("grayscale=1"), "grayscale takes no value", "a value where none is taken");
        Refused(() => ImageTransformPipeline.Parse("jpeg=101"), "quality is 1 to 100", "a bad quality");
        Refused(() => ImageTransformPipeline.Parse("=3"), "NAME or NAME=VALUE", "no name");
        Refused(() => ImageTransformPipeline.FromJson(JsonNode.Parse("""[{"name": "max_width", "value": 10, "radius": 2}]""")), "does not take 'radius'; it takes: resample", "JSON unknown option");
        Refused(() => ImageTransformPipeline.FromJson(JsonNode.Parse("""[{"value": 2}]""")), "needs a \"name\"", "JSON without a name");
        Refused(() => ImageTransformPipeline.FromJson(JsonNode.Parse("""{"contrast": 2}""")), "a string or an array", "JSON object");
        Refused(() => ImageTransformPipeline.FromJson(JsonNode.Parse("""[3]""")), "a string or an object", "JSON number");

        // The card's arithmetic: a wider image to 1024 with int truncation of the height; a narrower one kept.
        var wide = new ImageData(new float[2000 * 3], 1, 3, 2000);
        Check(ImageTransformPipeline.Parse("max_width=1024").Apply(wide) is { Width: 1024, Height: 1 }, "int(3 * 1024 / 2000) = 1");
        var page = new ImageData(new float[700 * 1000], 1, 1000, 700);
        Check(ReferenceEquals(ImageTransformPipeline.Parse("max_width=1024").Apply(page), page), "a narrower image is not touched");
        Check(ImageTransformPipeline.Parse("max_height=896").Apply(page) is { Height: 896, Width: 627 }, "int(700 * 896 / 1000) = 627");
    }

    private static void TransformedRequestMatchesTransformers(Device device)
    {
        RegisterGemma3Vision();
        string reference = TestData("vlm/compare/transformed");
        var manifest = JsonNode.Parse(File.ReadAllText(Path.Combine(reference, "manifest.json")))!;
        var pipeline = ImageTransformPipeline.FromJson(manifest["image_transforms"]);
        Check(pipeline.ToString() == "grayscale,max_width=64,resample=lanczos,contrast=1.5", $"the reference's transforms: {pipeline}");
        using var model = Idrak.Models.PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new Idrak.Models.PretrainedOptions { Device = device });
        var vision = model.Vision!;
        using var encoder = vision.CreateEncoder(new Idrak.Models.Abstractions.VisionEncoderOptions { Device = model.Device });
        var image = ChatImage.FromFile(TestData("vlm/image.jpg"));

        // Pixels and features: the transformed image as the encoder reads it, against transformers' on Pillow's.
        var transformed = pipeline.Apply(ChatImageDecoder.Decode(image));
        Check(transformed is { Channels: 1, Width: 64, Height: 48 }, $"transformed to {transformed.Channels} x {transformed.Height} x {transformed.Width}");
        using (var pixels = ((IVisionEncoderStages)encoder).PixelValues(transformed))
        {
            AssertClose(ReadNpyFloat32(Path.Combine(reference, "pixel_values.npy")), pixels.ToArray(), 1e-5f, "pixels after the transforms");
        }

        var features = encoder.Encode([transformed]);
        try
        {
            AssertClose(ReadNpyFloat32(Path.Combine(reference, "image_features.npy")), features[0].Features.ToArray(), 1e-5f, "features after the transforms");
        }
        finally
        {
            foreach (var f in features)
            {
                f.Dispose();
            }
        }

        // The chat request: the image as sent, its transforms on the request (the generator's own are none).
        int[] expected = [.. ReadNpyInt64(Path.Combine(reference, "generated_ids.npy")).Select(t => (int)t)];
        var chat = model.CreateChat();
        var reader = new Idrak.Generation.ChatGenerator(chat.Generator, chat.Template) { Images = new Idrak.Generation.ChatImages(encoder, vision.PromptFormat, vision.Attention) };
        var request = new ChatRequest([new ChatMessage("user", [image, new ChatText("What is in this image?")])])
        {
            Options = new GenerationOptions { Temperature = 0, NumPredict = 20, RepeatPenalty = 1 },
            ImageTransforms = ImageTransformPipeline.Parse("grayscale,max_width=64,contrast=1.5"),
        };
        var sampled = new List<int>();
        var answer = WithRecordedTokens(sampled, () => reader.Chat(request));
        Check(sampled.SequenceEqual(expected), $"greedy tokens with the request's transforms: {string.Join(" ", sampled)}, transformers {string.Join(" ", expected)}");
        Check(answer.Stats!.PromptTokens == (int)manifest["prompt_tokens"]!, $"prompt tokens {answer.Stats.PromptTokens}");

        // The same transforms as the images' own (ChatImages.Transforms), and none on the request overriding them.
        var own = new Idrak.Generation.ChatGenerator(chat.Generator, chat.Template)
        {
            Images = new Idrak.Generation.ChatImages(encoder, vision.PromptFormat, vision.Attention) { Transforms = request.ImageTransforms },
        };
        sampled.Clear();
        WithRecordedTokens(sampled, () => own.Chat(request with { ImageTransforms = null }));
        Check(sampled.SequenceEqual(expected), "ChatImages.Transforms");
        Check(own.Images.Read(image, request with { ImageTransforms = ImageTransformPipeline.Empty }) is { Width: 100, Height: 75, Channels: 3 }
              && own.Images.Read(image, request with { ImageTransforms = null }) is { Width: 64, Height: 48, Channels: 1 }, "an empty pipeline on the request: no transforms");
    }

    private sealed class InvertTransform : IImageTransform
    {
        public string Name => "invert";

        public string Summary => "invert: 1 - value (a test's own)";

        public IReadOnlyCollection<string> Keys => ["strength"];

        public void Check(ImageTransformStep step) => step.ThrowIfValue();

        public ImageData Apply(ImageData image, ImageTransformStep step) =>
            new([.. image.Pixels.Select(v => 1f - v)], image.Channels, image.Height, image.Width);
    }

    private static void TransformRegistry(Device device)
    {
        _ = device;
        string[] library = ["grayscale", "max_width", "max_height", "contrast", "brightness", "sharpness", "autocontrast", "jpeg"];
        Check(library.All(n => ImageTransforms.Origin(n) == Overrides.Library && ImageTransforms.Default(n) is not null), $"library defaults: {string.Join(", ", ImageTransforms.Names)}");
        Check(ImageTransforms.Describe().Contains("contrast: contrast=F", StringComparison.Ordinal), "Describe");
        Check(ImageTransforms.Find("nope") is null, "Find");
        ImageTransforms.Register(new InvertTransform());
        try
        {
            Check(ImageTransforms.Origin("invert") != Overrides.Library && ImageTransforms.Names.Contains("invert"), "registered from outside");
            var pipeline = ImageTransformPipeline.Parse("grayscale,invert,strength=1");
            var grey = new ImageData([0f, 1f, 0.2f, 0.6f], 1, 2, 2);
            var output = pipeline.Apply(grey);
            Check(PlanarBytes(output).SequenceEqual(PlanarBytes(new ImageData([1f, 0f, 0.8f, 0.4f], 1, 2, 2))), "the app's transform runs");
        }
        finally
        {
            Check(ImageTransforms.Unregister("invert"), "unregistered");
        }

        Check(ImageTransforms.Find("invert") is null && !ImageTransforms.Unregister("contrast"), "a library default is never removed");
    }
}
