// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Gemma3Vision;
using Idrak.Generation.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

// Images on the command line (plan 11, phase 7): run --image and chat --image / /image with the tiny Gemma 3 of phase 0
// give transformers' greedy tokens through the CLI's own path (the real encoder, and phase 0's features fed in);
// --grayscale reads the pixels Pillow's convert("L") gives; EXIF orientations turn images as exif_transpose does; the
// chat template's image parts expand to the processor's ids; a text-only model refuses images.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliImageGroup =
    [
        ("cli images: EXIF orientations 1 to 8 (JPEG APP1, PNG eXIf) turn images as Pillow's exif_transpose", CliImageExif),
        ("cli images: the Jinja chat template renders image parts; Gemma 3's image prompt format expands them to the processor's text and ids", CliImageTemplate),
        ("cli images: run --image with the tiny Gemma 3 gives transformers' 20 greedy tokens (real encoder and reference features), -j, --schema, --grayscale, an alias; chat --image and /image; a text-only model refuses", CliImageRun),
        ("cli images: vlm check reads tools/vlm/compare_real.py's folders (colour, and grey from a JPEG) for the tiny Gemma 3 and reports exact agreement; a changed token is a near-tie or a real difference by --tie", CliImageCompare),
        ("cli images: run --out writes the answer, or the -j document, to a file as UTF-8 without a BOM", CliImageRunOut),
    ];

    private static string VlmModel => TestData("vlm/tiny-gemma3");

    private static void CliImageExif(Device device)
    {
        _ = device;
        foreach (string file in Directory.GetFiles(TestData("vlm/exif"), "exif-*").Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(file);
            int n = int.Parse(name[5..6], System.Globalization.CultureInfo.InvariantCulture);
            var image = ChatImage.FromFile(file);
            Check(ChatImageDecoder.Orientation(image.Data.Span) == n, $"{name}: orientation {ChatImageDecoder.Orientation(image.Data.Span)}, expected {n}");
            var upright = ImageInputs.Decode(image);
            var expected = ImageCodecs.Decode(TestData($"vlm/exif/upright-{n}{Path.GetExtension(file)}.png"));
            Check(upright.Width == expected.Width && upright.Height == expected.Height && upright.Channels == expected.Channels
                  && upright.Pixels.Select(v => (int)MathF.Round(v * 255)).SequenceEqual(expected.Pixels.Select(v => (int)MathF.Round(v * 255))), $"{name}: {upright.Width} x {upright.Height} differs from Pillow's exif_transpose ({expected.Width} x {expected.Height})");
        }

        Check(ChatImageDecoder.Orientation(File.ReadAllBytes(TestData("vlm/image.png"))) == 1 && ChatImageDecoder.Orientation(File.ReadAllBytes(TestData("vlm/image.jpg"))) == 1
              && ChatImageDecoder.Orientation(new byte[] { 0xFF, 0xD8, 0xFF, 0xE1, 0x00 }) == 1, "no EXIF (or a cut one): orientation 1");
    }

    private static void CliImageTemplate(Device device)
    {
        _ = device;
        var facts = JsonNode.Parse(File.ReadAllText(TestData("vlm/manifest.json")))!["facts"]!["image_prompt"]!;
        var tokenizer = BpeTokenizer.Load(VlmModel);
        var template = JinjaChatTemplate.Load(VlmModel, tokenizer)!;
        Check(template.PartKinds.SetEquals([ChatParts.Text, ChatParts.Image]), $"Gemma 3's template renders images: {string.Join(", ", template.PartKinds)}");
        Check(JinjaChatTemplate.Load(CliModel, BpeTokenizer.Load(CliModel))!.PartKinds.SetEquals(ChatParts.TextOnly), "a ChatML template renders text only");

        var image = ChatImage.FromFile(TestData("vlm/image.png"));
        List<ChatMessage> messages = [new("system", "Read the scan."), new("user", [image, new ChatText("What is in this image?")])];
        string rendered = template.Render(messages, [], null, addGenerationPrompt: true);
        Check(rendered == (string)facts["rendered_by_chat_template"]!, $"rendered: {rendered}");
        Check(template.Render([new("user", "hi")], [], null) == template.Render([new("user", [new ChatText("hi")])], [], null), "text-only messages render as before");

        RegisterGemma3Vision();
        var model = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = Device.Cpu });
        using (model)
        {
            // The prompt format and its token ids come from the family's registration (samples/Gemma3Vision).
            var vision = model.Vision!;
            var format = vision.PromptFormat;
            var layout = new ImageTokenLayout(4) { Grid = [2, 2] };
            string expanded = format.Expand(rendered, [layout], tokenizer);
            Check(expanded == (string)facts["expanded_text"]!, $"expanded: {expanded}");
            int[] ids = [.. facts["input_ids"]!.AsArray().Select(i => (int)i!)];
            Check(tokenizer.Encode(expanded).SequenceEqual(ids), $"ids: {string.Join(" ", tokenizer.Encode(expanded))}");
            Check(Fails(() => format.Expand(rendered, [layout, layout], tokenizer)), "one marker for two images is refused");

            // A chat generator without images takes text only; with them, images, and batches refuse them.
            var chat = model.CreateChat(KeyValueFormat.Float32, 64);
            Check(chat.PartKinds.SetEquals(ChatParts.TextOnly), "a chat generator takes text unless given images");
            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
            var withImages = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, format, vision.Attention) };
            Check(withImages.PartKinds.Contains(ChatParts.Image) && withImages.RenderPrompt(new ChatRequest(messages)) == expanded, "RenderPrompt expands the images");
            Check(Fails(() => withImages.ChatBatch([new ChatRequest(messages)])), "a chat batch with images is refused");
        }
    }

    private static bool Fails(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or ArgumentException)
        {
            return true;
        }
    }

    // The default sampler, recording the ids of the first row it hands out (the CLI's generations, seen from outside).
    private sealed class RecordingSampler(ITokenSampler inner, List<int> ids) : ITokenSampler
    {
        public int Rows => inner.Rows;

        public int Vocabulary => inner.Vocabulary;

        public Tensor Ids => inner.Ids;

        public bool Recordable => inner.Recordable;

        public void Sample(Tensor logits) => inner.Sample(logits);

        public void SetHistory(IReadOnlyList<int> tokens) => inner.SetHistory(tokens);

        public void Reset() => inner.Reset();

        public SampledToken[][] Read(int fromStep, int toStep)
        {
            var steps = inner.Read(fromStep, toStep);
            if (fromStep > 0 || toStep > 1)
            {
                ids.AddRange(steps.Select(s => s[0].Id));                // the prompt's own step (0 to 1) is read again later
            }

            return steps;
        }

        public void Dispose() => inner.Dispose();
    }

    private static void CliImageRun(Device device)
    {
        // Without the Gemma 3 vision family registered, the vision checkpoint is refused, naming the registry.
        Gemma3VisionNotRegistered(device);
        var (refusedCode, _, refused) = RunIdrakOn(device, null, "run", VlmModel, "--image", TestData("vlm/image.png"), "hi");
        Check(refusedCode != 0 && refused.Contains("Vision family 'Gemma3ForConditionalGeneration' is not registered", StringComparison.Ordinal)
              && refused.Contains("VisionFamilies.Register", StringComparison.Ordinal), $"run --image without the family: {refusedCode} {refused}");
        RegisterGemma3Vision();
        var facts = JsonNode.Parse(File.ReadAllText(TestData("vlm/manifest.json")))!["facts"]!;
        int[] newTokens = [.. facts["generate"]!["new_tokens"]!.AsArray().Select(n => (int)n!)];
        string expected = BpeTokenizer.Load(VlmModel).Decode(newTokens).Trim();
        float[] reference = ReadNpyFloat32(TestData("vlm/reference/image_features.npy"));
        string png = TestData("vlm/image.png");
        string[] greedy = ["-s", "Read the scan.", "--temperature", "0", "--max-tokens", "20"];

        // The real encoder (SigLIP and the projector of phase 3b) through the CLI; the tokens seen by a sampler that records them.
        var sampled = new List<int>();
        var library = TokenSamplers.Default(TokenSamplers.DefaultName)!;
        TokenSamplers.Register(TokenSamplers.DefaultName, request => new RecordingSampler(library(request), sampled));
        int code;
        string text, error;
        JsonObject json;
        try
        {
            (code, text, error) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy, "-j"]);
        }
        finally
        {
            TokenSamplers.Unregister(TokenSamplers.DefaultName);
        }

        json = JsonOf(text, "run --image");
        string answer = (string?)json["text"] ?? "";
        Check(sampled.SequenceEqual(newTokens), $"run --image: greedy tokens {string.Join(" ", sampled)}, expected {string.Join(" ", newTokens)}");
        Check(code == 0 && ((string)json["text"]!).Length == expected.Length && (int)json["prompt_tokens"]! == 38 && (int)json["generated_tokens"]! == 20
              && json["images"] is JsonArray { Count: 1 } && (bool)json["grayscale"]! == false, $"run --image: {code} {text} {error}; expected '{expected}'");

        // Reference features fed in, the pixels each image is read with recorded.
        var pixels = new List<float[]>();
        int calls = 0;
        ImageInputs.EncoderFactory = (model, options) => new ReferenceEncoder(((Gemma3Vision)model.Vision!).Preprocessor(options.Grayscale), reference, model.Device, images =>
        {
            calls++;
            return images;
        }, pixels);
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-image-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy, "-j"]);
            Check(code == 0 && (string?)JsonOf(text, "run, reference features")["text"] == answer, $"run with phase 0's features: {text}");
            AssertClose(ReadNpyFloat32(TestData("vlm/reference/pixel_values-png.npy")), pixels[^1], 1e-5f, "the pixels read");

            // --grayscale: Pillow's convert("L"), as phase 0 wrote it; then the same as an alias setting.
            (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "--grayscale", "What is in this image?", .. greedy, "-j"]);
            Check(code == 0 && (bool)JsonOf(text, "run --grayscale")["grayscale"]!, $"run --grayscale: {text}");
            float[] grey = ReadNpyFloat32(TestData("vlm/reference/pixel_values-png-gray.npy"));
            AssertClose(grey, pixels[^1], 1e-5f, "--grayscale pixels");
            string config = Path.Combine(folder, "config.json");
            File.WriteAllText(config, new JsonObject { ["aliases"] = new JsonObject { ["ocr"] = new JsonObject { ["model"] = VlmModel, ["grayscale"] = true } } }.ToJsonString());
            int before = pixels.Count;
            var output = new StringWriter();
            code = StandardInput.With(new StringReader(""), () => Idrak.Cli.CommandLine.Run(
                ["run", "ocr", "--image", png, "hi", "--max-tokens", "2", "-q", "-d", device.ToString(), "-C", config], output, new StringWriter()));
            Check(code == 0 && pixels.Count == before + 1, "run with an alias");
            AssertClose(grey, pixels[^1], 1e-5f, "an alias's grayscale setting");

            // Two images, and --schema with -j (the tiny model's answer is not JSON: exit 1, schema_valid false).
            (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "--image", TestData("vlm/image.jpg"), "Compare.", "--max-tokens", "3", "-j"]);
            Check(code == 0 && (int)JsonOf(text, "two images")["prompt_tokens"]! > 38 && calls > 0, $"two images: {text}");
            string schema = Path.Combine(folder, "schema.json");
            File.WriteAllText(schema, """{"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}""");
            (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "Read it.", "--schema", schema, "--max-tokens", "4", "-j"]);
            json = JsonOf(text, "run --schema");
            Check(code == 1 && (bool)json["schema_valid"]! == false && json["images"] is JsonArray, $"run --image --schema: {code} {text}");

            // Chat: --image goes with the first message, /image with the next; each answer reads the conversation's images again.
            calls = 0;
            (code, text, error) = RunIdrakOn(device, "What is in this image?\n/image " + png + "\nAnd this one?\n/exit\n", ["chat", VlmModel, "--image", png, .. greedy, "-j"]);
            json = JsonOf(text, "chat --image");
            var messages = json["messages"]!.AsArray();
            Check(code == 0 && messages.Count == 5 && (string?)messages[2]!["content"] == answer && calls == 2, $"chat --image: {code} {calls} calls {text} {error}");
            Check(messages[1]!["content"] is JsonArray { Count: 2 } first && (string?)first[0]!["type"] == "image"
                  && messages[3]!["content"] is JsonArray { Count: 2 }, "chat: the images are parts of their messages");

            // A text-only model refuses images; a missing file is a usage error.
            (code, _, error) = RunIdrakOn(device, null, "run", CliModel, "--image", png, "hi");
            Check(code == 2 && error.Contains("reads text only") && error.Contains("--image"), $"a text model given --image: {error}");
            (code, text, _) = RunIdrakOn(device, "/image " + png + "\n/exit\n", "chat", CliModel);
            Check(code == 0 && text.Contains("reads text only"), $"a text model given /image: {text}");
            (code, _, error) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", "--context", "32", "--max-tokens", "2"]);
            Check(code == 1 && error.Contains("context window"), $"a prompt with an image never cut to the window: {error}");
            (code, _, error) = RunIdrakOn(device, null, "run", VlmModel, "--image", Path.Combine(folder, "missing.png"), "hi");
            Check(code == 2 && error.Contains("Image file not found"), $"a missing image: {error}");
        }
        finally
        {
            ImageInputs.EncoderFactory = null;
            Directory.Delete(folder, recursive: true);
        }
    }

    // Phase 0's features for every image, recording the pixels each image is read with (the family's preprocessing).
    private sealed class ReferenceEncoder(ImagePreprocessor preprocessor, float[] features, Device device, Func<IReadOnlyList<ImageData>, IReadOnlyList<ImageData>> seen,
        List<float[]> pixels) : IVisionEncoder
    {
        private static readonly ImageTokenLayout Grid = new(4) { Grid = [2, 2] };

        public int Width => 24;

        public Device Device => device;

        public ImageTokenLayout Layout(ImageData image) => Grid;

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images)
        {
            pixels.AddRange(seen(images).Select(preprocessor.Pixels));
            return [.. images.Select(_ => new ImageFeatures(Tensor.From(features, [4, 24], device), Grid))];
        }

        public void Dispose()
        {
        }
    }

    private static void CliImageCompare(Device device)
    {
        RegisterGemma3Vision();
        foreach (string name in new[] { "color", "gray" })
        {
            string reference = TestData($"vlm/compare/{name}");
            var (code, text, error) = RunIdrakOn(device, null, "vlm", "check", VlmModel, "--reference", reference, "-j");
            var json = JsonOf(text, $"vlm check {name}");
            var forced = json["teacher_forced"]!;
            Check(code == 0 && (bool)json["ok"]! && (int)forced["agree"]! == 20 && (int)forced["steps"]! == 20
                  && (int)json["teacher_forced_reference_features"]!["agree"]! == 20 && json["greedy"]!["first_divergence"] is null
                  && (int)forced["prompt_rows_agree"]! == (int)forced["prompt_rows"]!, $"vlm check {name}: {code} {error} {text}");
            Check((float)json["pixels"]!["max_abs"]! <= 1e-5f && (double)json["features"]!["cosine"]! > 0.99999 && (float)json["features"]!["max_abs"]! < 1e-4f
                  && (bool)json["prompt"]!["same"]! && (bool)json["grayscale"]! == (name == "gray") && (float)forced["max_top5_logit_difference"]! < 1e-4f,
                $"vlm check {name}: pixels, features, prompt: {text}");
            (code, text, _) = RunIdrakOn(device, null, "vlm", "check", VlmModel, "--reference", reference);
            Check(code == 0 && text.Contains("Verdict: Idrak picks transformers' token at all 20 steps and its own greedy answer is the same.", StringComparison.Ordinal), $"vlm check {name} as text: {text}");
        }

        // A reference whose last answer token (step 19, so nothing is fed after it) is its second choice: Idrak disagrees there
        // only; a near-tie or a real difference by --tie.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-vlm-check-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            foreach (string file in Directory.GetFiles(TestData("vlm/compare/color")))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            File.WriteAllText(Path.Combine(folder, "manifest.json"), File.ReadAllText(Path.Combine(folder, "manifest.json")).Replace("\"../../image.png\"", "null", StringComparison.Ordinal));
            long[] second = ReadNpyInt64(Path.Combine(folder, "gen_top_ids.npy"));
            string ids = Path.Combine(folder, "generated_ids.npy");
            byte[] bytes = File.ReadAllBytes(ids);
            int start = 10 + BitConverter.ToUInt16(bytes, 8);
            BitConverter.TryWriteBytes(bytes.AsSpan(start + 19 * 8), second[19 * 5 + 1]);
            File.WriteAllBytes(ids, bytes);

            var (code, text, _) = RunIdrakOn(device, null, "vlm", "check", VlmModel, "--reference", folder, "--tie", "1000", "-j");
            var json = JsonOf(text, "vlm check, a changed token");
            var first = json["teacher_forced"]!["disagreements"]![0]!;
            Check(code == 0 && (int)first["step"]! == 19 && (bool)first["near_tie"]! && (int)first["idrak"]![0]!["id"]! == (int)second[19 * 5]
                  && json["teacher_forced"]!["disagreements"]!.AsArray().Count == 1 && (int)json["greedy"]!["first_divergence"]! == 19 && json["pixels"] is null
                  && ((string)json["verdict"]!).Contains("Every disagreement (1) is a near-tie", StringComparison.Ordinal), $"a changed token, --tie 1000: {code} {text}");
            (code, text, _) = RunIdrakOn(device, null, "vlm", "check", VlmModel, "--reference", folder, "--tie", "0");
            Check(code == 1 && text.Contains("REAL DIFFERENCE", StringComparison.Ordinal) && text.Contains("reference top-5:", StringComparison.Ordinal)
                  && text.Contains("NOT a near-tie", StringComparison.Ordinal) && text.Contains("first divergence at step 19", StringComparison.Ordinal), $"a changed token, --tie 0: {code} {text}");

            Check(RunIdrakOn(device, null, "vlm", "check", VlmModel).Code == 2 && RunIdrakOn(device, null, "vlm", "check", VlmModel, "--reference", TestData("vlm/reference")).Code == 2,
                "vlm check: --reference is required and must be compare_real.py's folder");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliImageRunOut(Device device)
    {
        RegisterGemma3Vision();
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-run-out-{Guid.NewGuid():N}");
        try
        {
            string png = TestData("vlm/image.png");
            string answerFile = Path.Combine(folder, "sub", "answer.txt"), jsonFile = Path.Combine(folder, "answer.json");
            string[] greedy = ["-s", "Read the scan.", "--temperature", "0", "--max-tokens", "20"];
            var (code, text, error) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy, "-j", "--out", jsonFile]);
            byte[] bytes = File.ReadAllBytes(jsonFile);
            string written = new System.Text.UTF8Encoding(false, throwOnInvalidBytes: true).GetString(bytes);
            Check(code == 0 && !(bytes is [0xEF, 0xBB, 0xBF, ..]) && written == text && JsonOf(written, "--out -j")["text"] is not null, $"run -j --out: {code} {error}\n{written}\n{text}");
            string answer = (string)JsonOf(text, "run -j")["text"]!;
            Check(answer.Any(c => c > 127), $"the tiny model's answer holds non-ASCII text (byte fallback): {answer}");

            (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy, "-o", answerFile]);
            bytes = File.ReadAllBytes(answerFile);
            written = new System.Text.UTF8Encoding(false, throwOnInvalidBytes: true).GetString(bytes);
            Check(code == 0 && !(bytes is [0xEF, 0xBB, 0xBF, ..]) && written == answer.Trim() + "\n" && text.Contains(answer.Trim(), StringComparison.Ordinal), $"run -o: {code}\n{written}\n{text}");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }
}
