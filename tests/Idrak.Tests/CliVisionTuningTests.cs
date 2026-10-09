// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Gemma3Vision;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

// Plan 12, phase 4: the command line. idrak tune on LlamaFactory's ShareGPT (<image> placeholders, an "images" list) with
// the tiny Gemma 3 (its plug-in) and the tiny LLaVA (the test plug-in), images from a folder and from a zip: the adapters,
// the trained projector and tuning_images.json written, a generated-text score (cer), the run plan; run --adapter applies
// the projector and the saved preparation; the refusals (an unregistered family, an adapter of another family); tune init
// writes a vision example; tune.json's image keys; tune evaluate with --metric cer.
internal static partial class Tests
{
    // The fixture's three records as LlamaFactory writes them (train.json): a JSON array of ShareGPT records.
    private static string WriteShareGpt(string folder, string family, string name, int records = 3)
    {
        var tuning = JsonNode.Parse(File.ReadAllText(TuningData($"{family}/tuning.json")))!;
        var array = new JsonArray();
        foreach (var record in tuning["records"]!.AsArray().Take(records))
        {
            var sharegpt = record!["sharegpt"]?.DeepClone() ?? ShareGptOf(record);
            array.Add(sharegpt);
        }

        string path = Path.Combine(folder, name);
        File.WriteAllText(path, array.ToJsonString());
        return path;
    }

    // A record's messages in ShareGPT (for a model whose fixture keeps none): <image> where each image part is.
    private static JsonNode ShareGptOf(JsonNode record)
    {
        var conversations = new JsonArray();
        string? system = null;
        foreach (var m in record["messages"]!.AsArray())
        {
            string role = (string)m!["role"]!;
            string text = m["content"] is JsonValue v ? (string)v! : string.Concat(m["content"]!.AsArray().Select(p => (string?)p!["type"] == "image" ? "<image>" : (string?)p!["text"]));
            if (role == "system")
            {
                system = text;
                continue;
            }

            conversations.Add(new JsonObject { ["from"] = role == "user" ? "human" : "gpt", ["value"] = text });
        }

        var json = new JsonObject { ["conversations"] = conversations, ["images"] = record["images"]!.DeepClone() };
        if (system is not null)
        {
            json["system"] = system;
        }

        return json;
    }

    // The images the records name, in a folder and in a zip.
    private static (string Folder, string Zip) WriteTuningImages(string folder)
    {
        string images = Path.Combine(folder, "images");
        Directory.CreateDirectory(images);
        foreach (string name in new[] { "image.png", "image-palette.png" })
        {
            File.Copy(TestData($"vlm/{name}"), Path.Combine(images, name));
        }

        string zip = Path.Combine(folder, "images.zip");
        ZipFile.CreateFromDirectory(images, zip);
        return (images, zip);
    }

    private static readonly string[] TinyTuning =
    [
        "--rank", "2", "--alpha", "4", "--targets", "q,k,v,o", "--lr", "0.2", "--optimizer", "sgd", "--schedule", "constant", "--warmup", "0",
        "--batch-tokens", "4096", "--max-new", "8", "-d", "cpu",
    ];

    private static void CliVisionTuning(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        RegisterGemma3Vision();
        string folder = TempFolder();
        try
        {
            var (images, zip) = WriteTuningImages(folder);
            string train = WriteShareGpt(folder, "gemma3", "train.json");
            string validation = WriteShareGpt(folder, "gemma3", "val.json", records: 2);

            // Images from a folder: LoRA and the projector, a transform and a vision option saved, cer every step, -v's plan.
            string adapter = Path.Combine(folder, "adapter");
            var tuned = TrainCli(["tune", "-b", VlmModel, "--data", train, "--eval", validation, "--images", images, "--train-projector",
                "--image-transform", "contrast=1.5", "--vision", "do_pan_and_scan=false", "--metric", "cer", "--metric-every", "1", "--epochs", "3",
                "-o", adapter, "-v", .. TinyTuning]);
            string text = tuned.Out.ReplaceLineEndings("\n");
            Check(tuned.Code == 0, $"tune with images: {tuned.Code}\n{tuned.Out}\n{tuned.Err}");
            Check(text.Contains("training: 3 conversations (sharegpt)", StringComparison.Ordinal) && text.Contains("4 images", StringComparison.Ordinal),
                $"the records and images read: {text}");
            Check(text.Contains("training images: 4 in 3 of 3 sequences (2 distinct), blocks per image 1", StringComparison.Ordinal)
                  && text.Contains("image tokens", StringComparison.Ordinal), $"the plan's images: {text}");
            Check(text.Contains("before training: cer ", StringComparison.Ordinal) && text.Contains("step 1: cer ", StringComparison.Ordinal)
                  && text.Contains("step 3: cer ", StringComparison.Ordinal) && System.Text.RegularExpressions.Regex.IsMatch(text, @"trained in [^\n]*, cer [0-9.]+ on 2 answers"),
                $"cer before, at every step and in the summary: {text}");
            Check(text.Contains("memory: layers", StringComparison.Ordinal) && text.Contains("memory: encoder", StringComparison.Ordinal)
                  && text.Contains("memory: activations", StringComparison.Ordinal) && text.Contains("from the cache, 2 encoded", StringComparison.Ordinal),
                $"-v: the memory and the cache: {text}");
            Check((tuned.Out + tuned.Err).Contains("image features: 2 from the cache, 0 encoded", StringComparison.Ordinal), $"-v: each batch's cache hits: {tuned.Out}{tuned.Err}");

            // The folder: adapters, the projector (modules_to_save), the preparation with its family, the manifest.
            using (var saved = SafeTensorsReader.Open(Path.Combine(adapter, "adapter_model.safetensors")))
            {
                Check(saved.Tensors.Keys.Count(k => k.Contains("lora_A", StringComparison.Ordinal)) == 4 * 2
                      && saved.Tensors.Keys.Any(k => k.Contains(".multi_modal_projector.", StringComparison.Ordinal)),
                    $"adapters and the projector saved: {string.Join(", ", saved.Tensors.Keys)}");
            }

            var preparation = TuningImages.Read(adapter)!;
            Check(preparation.Transforms.ToString() == "contrast=1.5" && preparation.VisionOptions.ToString() == "do_pan_and_scan=false"
                  && preparation.Family == Gemma3VisionFamily.Architecture, $"tuning_images.json: {preparation.ToJson()}");
            Check(TuningManifest.Read(adapter)?.BaseModel == Path.GetFullPath(VlmModel), "idrak-tuning.json names the base model");

            CliVisionTuningRun(adapter, preparation);

            // Images from a zip (read in place), the language model alone; the adapter folder as the model of tune evaluate.
            string zipped = Path.Combine(folder, "zipped");
            var fromZip = TrainCli(["tune", "train", VlmModel, train, "--images", zip, "--epochs", "1", "-o", zipped, .. TinyTuning]);
            Check(fromZip.Code == 0 && File.Exists(Path.Combine(zipped, "adapter_model.safetensors")) && TuningImages.Read(zipped)?.IsEmpty == true
                  && JsonNode.Parse(File.ReadAllText(Path.Combine(zipped, "adapter_config.json")))!["modules_to_save"] is null,
                $"tune with images from a zip: {fromZip.Code} {fromZip.Out} {fromZip.Err}");

            // tune evaluate: the loss and cer of the base model and the adapter, side by side.
            string answers = Path.Combine(folder, "answers.jsonl");
            var evaluated = TrainCli(["tune", "evaluate", VlmModel, validation, "--images", images, "--adapter", adapter, "--metric", "cer", "--max-new", "8", "-o", answers, "-d", "cpu"]);
            string evaluation = evaluated.Out.ReplaceLineEndings("\n");
            Check(evaluated.Code == 0 && System.Text.RegularExpressions.Regex.Matches(evaluation, @"loss [0-9.]+, cer [0-9.]+ on 2 answers").Count == 2
                  && evaluation.Contains("the adapter scores better than the base model on", StringComparison.Ordinal), $"tune evaluate --metric cer: {evaluated.Code} {evaluation} {evaluated.Err}");
            var lines = File.ReadAllLines(answers);
            Check(lines.Length == 2 && JsonNode.Parse(lines[0])!["adapter adapter"]!["cer"] is not null, $"tune evaluate -o: {string.Join("\n", lines)}");

            // The tiny LLaVA (the test plug-in): the same command line, another family.
            Idrak.PluginTests.LlavaPlugin.Register();
            string llava = Path.Combine(folder, "llava");
            string llavaTrain = WriteShareGpt(folder, "llava", "llava.json");
            var other = TrainCli(["tune", "-b", TestData("vlm-llava/tiny-llava"), "--data", llavaTrain, "--images", zip, "--train-projector", "--epochs", "2", "-o", llava, .. TinyTuning]);
            Check(other.Code == 0 && TuningImages.Read(llava)?.Family == "LlavaForConditionalGeneration", $"tune the tiny LLaVA: {other.Code} {other.Out} {other.Err}");
            using (var saved = SafeTensorsReader.Open(Path.Combine(llava, "adapter_model.safetensors")))
            {
                Check(saved.Tensors.Keys.Any(k => k.Contains("multi_modal_projector.linear_1", StringComparison.Ordinal)), $"LLaVA's projector saved: {string.Join(", ", saved.Tensors.Keys)}");
            }

            // An adapter of one family on a model of another: a usage error naming both, before any weight is read.
            var (code, _, error) = RunIdrakOn(Device.Cpu, null, ["run", TestData("vlm-llava/tiny-llava"), "--adapter", adapter, "--image", TestData("vlm/image.png"), "hi"]);
            Check(code == 2 && error.Contains($"vision family {Gemma3VisionFamily.Architecture}", StringComparison.Ordinal)
                  && error.Contains("LlavaForConditionalGeneration", StringComparison.Ordinal), $"run --adapter of another family: {code} {error}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // run --adapter: the trained projector (CreateVisionEncoder) and the saved preparation, as the library generates with them.
    private static void CliVisionTuningRun(string adapter, TuningImages preparation)
    {
        string png = TestData("vlm/image.png");
        string[] greedy = ["-s", "Read the scan.", "--temperature", "0", "--max-tokens", "12", "-j"];
        var (code, text, error) = RunIdrakOn(Device.Cpu, null, ["run", VlmModel, "--image", png, "What is in this image?", "--adapter", adapter, .. greedy]);
        var json = JsonOf(text, "run --adapter");
        Check(code == 0 && (string?)json["image_transforms"] == "contrast=1.5" && (string?)json["vision_options"]?["do_pan_and_scan"] == "false",
            $"run --adapter takes the saved preparation: {code} {text} {error}");

        // The library with the same adapter: its encoder with the trained projector, the saved transforms and options.
        using var model = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = Device.Cpu, MergeAdapter = adapter });
        var vision = model.Vision!;
        using var trained = model.CreateVisionEncoder();
        using var frozen = vision.CreateEncoder(new VisionEncoderOptions { Device = Device.Cpu });
        var pixels = preparation.Apply(ChatImageDecoder.Decode(ChatImage.FromFile(png)));
        float[] withProjector = trained.Encode([pixels])[0].Features.ToArray(), base_ = frozen.Encode([pixels])[0].Features.ToArray();
        Check(withProjector.Length == base_.Length && withProjector.Zip(base_).Max(p => MathF.Abs(p.First - p.Second)) > 1e-3f,
            "the trained projector changes the image features");
        string Generate(IVisionEncoder encoder)
        {
            var chat = model.CreateChat(Idrak.Abstraction.Generation.KeyValueLayouts.Get("float32"), 4096);
            var withImages = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) { Transforms = preparation.Pipeline } };
            var request = new ChatRequest([new ChatMessage("system", "Read the scan."), new ChatMessage("user", [ChatImage.FromFile(png), new ChatText("What is in this image?")])],
                Options: new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 12, NumCtx = 4096 }) { VisionOptions = preparation.VisionOptions };
            return withImages.Chat(request).Message!.Content;
        }

        string expected = Generate(trained);
        Check((string?)json["text"] == expected, $"run --adapter answers as the library with the trained projector: '{json["text"]}', expected '{expected}' (frozen projector: '{Generate(frozen)}')");

        // The command line's transforms replace the adapter's.
        (code, text, _) = RunIdrakOn(Device.Cpu, null, ["run", VlmModel, "--image", png, "What is in this image?", "--adapter", adapter, "--image-transform", "none", .. greedy]);
        Check(code == 0 && (string?)JsonOf(text, "run --adapter --image-transform none")["image_transforms"] == "", $"--image-transform over the adapter's: {text}");
    }

    private static void CliVisionTuningRefusals(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // An unregistered family: idrak tune exits with the registry's message before it reads the data (none exists here).
        VisionFamilies.Unregister(Gemma3VisionFamily.Architecture);
        ImageAttentionRules.Unregister(Gemma3VisionPlugin.ImageBlocks);
        try
        {
            var refused = TrainCli("tune", "-b", VlmModel, "--data", Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.json"), "-o", "unused", "-d", "cpu");
            Check(refused.Code == 1 && refused.Err.Contains($"Vision family '{Gemma3VisionFamily.Architecture}' is not registered", StringComparison.Ordinal)
                  && refused.Err.Contains("-P", StringComparison.Ordinal) && !refused.Err.Contains("missing-", StringComparison.Ordinal),
                $"tune without the family: {refused.Code} {refused.Err}");
        }
        finally
        {
            RegisterGemma3Vision();
        }

        // Image options on a text model, and bad names: refused, naming the choices.
        string text = WriteChatModel(TinyChatSpec());
        string folder = TempFolder();
        try
        {
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllText(data, "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}, {\"role\": \"assistant\", \"content\": \"hello\"}]}\n");
            var textModel = TrainCli("tune", "-b", text, "--data", data, "--train-projector", "-o", Path.Combine(folder, "a"), "-d", "cpu");
            Check(textModel.Code == 1 && textModel.Err.Contains("no vision part", StringComparison.Ordinal), $"--train-projector on a text model: {textModel.Err}");
            var format = TrainCli("tune", "-b", text, "--data", data, "--data-format", "alpaca", "-o", Path.Combine(folder, "a"));
            Check(format.Code == 2 && format.Err.Contains("messages, sharegpt", StringComparison.Ordinal), $"--data-format alpaca: {format.Err}");
            var metric = TrainCli("tune", "-b", text, "--data", data, "--metric", "bleu", "-o", Path.Combine(folder, "a"));
            Check(metric.Code == 2 && metric.Err.Contains("cer, wer", StringComparison.Ordinal), $"--metric bleu: {metric.Err}");
            var noEval = TrainCli("tune", "-b", text, "--data", data, "--metric", "wer", "-o", Path.Combine(folder, "a"), "-d", "cpu", "--max-length", "64");
            Check(noEval.Code == 1 && noEval.Err.Contains("--eval FILE", StringComparison.Ordinal), $"--metric without evaluation data: {noEval.Err}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(text, true);
        }
    }

    private static void CliVisionTuneInit(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        RegisterGemma3Vision();
        string folder = TempFolder();
        try
        {
            // tune init -P with a plug-in that registers a vision family: the vision example, the plug-in kept.
            string plugin = typeof(Gemma3VisionPlugin).Assembly.Location;
            string settings = Path.Combine(folder, "tune.json");
            var init = TrainCli("tune", "init", "-P", plugin, "-b", VlmModel, "-o", settings);
            string written = File.ReadAllText(settings).ReplaceLineEndings("\n");
            Check(init.Code == 0 && written.Contains("\"data\": [\"train.json\"]", StringComparison.Ordinal) && written.Contains("\"eval\": \"val.json\"", StringComparison.Ordinal)
                  && written.Contains("\"metric\": \"cer\"", StringComparison.Ordinal) && written.Contains("\"train-projector\": false", StringComparison.Ordinal)
                  && written.Contains(Gemma3VisionFamily.Architecture, StringComparison.Ordinal) && written.Contains("\"plugins\": [", StringComparison.Ordinal),
                $"tune init -P: {init.Out} {init.Err}\n{written}");
            Check(JsonNode.Parse(written, documentOptions: new System.Text.Json.JsonDocumentOptions { CommentHandling = System.Text.Json.JsonCommentHandling.Skip })!["plugins"]![0]!.ToString()
                  == Path.GetFullPath(plugin), "the plug-in's full path kept");
            var plain = TrainCli("tune", "init", "-b", VlmModel, "-o", Path.Combine(folder, "text.json"));
            Check(plain.Code == 0 && !File.ReadAllText(Path.Combine(folder, "text.json")).Contains("\"metric\"", StringComparison.Ordinal), "without -P: the text example");

            // A tune.json with the image keys as the plan writes them (underscores, "format" naming the data's, "vision" an
            // object, "parts" a list) and the disk feature cache under --cache.
            var (images, _) = WriteTuningImages(folder);
            string train = WriteShareGpt(folder, "gemma3", "train.json");
            string adapter = Path.Combine(folder, "adapter");
            string config = Path.Combine(folder, "vision.json");
            File.WriteAllText(config, new JsonObject
            {
                ["format"] = "sharegpt", ["command"] = "train", ["model"] = VlmModel, ["data"] = new JsonArray(train), ["out"] = adapter,
                ["images"] = images, ["image_transform"] = "max_width=64", ["vision"] = new JsonObject { ["do_pan_and_scan"] = "false" },
                ["parts"] = new JsonArray("projector"), ["feature_cache"] = "disk", ["epochs"] = 2,
            }.ToJsonString());
            string cache = Path.Combine(folder, "cache");
            var tuned = TrainCli(["tune", "--config", config, "--cache", cache, .. TinyTuning]);
            var preparation = TuningImages.Read(adapter);
            Check(tuned.Code == 0 && preparation?.Transforms.ToString() == "max_width=64" && preparation.VisionOptions.ToString() == "do_pan_and_scan=false"
                  && JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")))!["modules_to_save"] is JsonArray { Count: 1 },
                $"tune --config with image keys: {tuned.Code} {tuned.Out} {tuned.Err}");
            Check(Directory.Exists(Path.Combine(cache, "features")) && Directory.EnumerateFiles(Path.Combine(cache, "features"), "*", SearchOption.AllDirectories).Count() == 2,
                "the disk feature cache under --cache keeps the two images");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
