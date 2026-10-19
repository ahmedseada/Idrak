// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

// Plan 19, phase A4: models made from scratch on the command line (data tokenizer, data tokens, create, pretrain, resume).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliPretrainGroup =
    [
        ("cli pretrain: data tokenizer learns the reference's tokenizer from the corpus, and data tokens writes the reference's token file; --model unigram writes tokenizer.json and tokenizer.model (with SentencePiece's switches and normalization rules carried into it), --model wordpiece BERT's tokenizer, and bad options and --model with --trainer are refused naming them", CliPretrainData),
        ("cli pretrain: create makes a model from a spec file (and create init's), pretrain trains it, Ctrl+C-like stops checkpoint, resume finishes, runs show reads the steps, run answers", CliPretrainRun),
        ("cli pretrain: an unknown architecture, a missing spec key, a missing tokenizer, no sources and a run folder in use are refused", CliPretrainRefusals),
        ("cli pretrain vision: create init --vision, create and pretrain an IdrakVision model with its plug-in (-P), resume, run with an image; without the plug-in the family is refused naming -P", CliPretrainVision),
    ];

    private static void CliPretrainVision(Device device)
    {
        string plugin = typeof(Idrak.IdrakVision.IdrakVisionPlugin).Assembly.Location;
        string folder = TempFolder("cli-pretrain-vision");
        Idrak.IdrakVision.IdrakVisionPlugin.Unregister();
        try
        {
            string spec = Path.Combine(folder, "spec.json");
            TrainCliJson("create", "init", "--vision", "-o", spec);
            string text = File.ReadAllText(spec);
            Check(text.Contains("IdrakVisionForConditionalGeneration", StringComparison.Ordinal) && text.Contains("\"vision\"", StringComparison.Ordinal), "create init --vision writes a vision part");
            text = text.Replace("\"dim\": 256", "\"dim\": 32").Replace("\"layers\": 4", "\"layers\": 2").Replace("\"ffDim\": 768", "\"ffDim\": 64")
                .Replace("\"maxPositions\": 1024", "\"maxPositions\": 128")
                .Replace("\"tokenizer\": \"tokenizer.json\"", $"\"tokenizer\": {JsonValue.Create(PretrainData("tokenizer/tokenizer.json")).ToJsonString()}")
                .Replace("\"hidden_size\": 192", "\"hidden_size\": 16").Replace("\"intermediate_size\": 768", "\"intermediate_size\": 32")
                .Replace("\"num_hidden_layers\": 4", "\"num_hidden_layers\": 1").Replace("\"num_attention_heads\": 3", "\"num_attention_heads\": 2")
                .Replace("\"image_size\": 224", "\"image_size\": 28").Replace("\"patch_size\": 14", "\"patch_size\": 7");
            File.WriteAllText(spec, text);
            string model = Path.Combine(folder, "model");
            var (code, _, error) = TrainCli("create", spec, "-o", model);
            Check(code == 1 && error.Contains("IdrakVisionForConditionalGeneration", StringComparison.Ordinal) && error.Contains("-P", StringComparison.Ordinal), $"without the plug-in: {error}");

            var created = TrainCliJson("create", spec, "-o", model, "--seed", "2", "-P", plugin);
            Check(File.Exists(Path.Combine(model, "preprocessor_config.json")) && JsonNode.Parse(File.ReadAllText(Path.Combine(model, "config.json")))!["vision_config"] is JsonObject,
                $"a vision-language model made: {created["files"]}");
            string run = Path.Combine(folder, "run");
            var trained = TrainCliJson("pretrain", model, "-P", plugin, "--data", ScratchData("records.jsonl"), "--trained", "tower,projector", "--sequences", "3",
                "--max-length", "128", "--batch-tokens", "384", "--steps", "2", "--lr", "1e-3", "-o", run, "-d", device.ToString());
            Check((int)trained["steps"]! == 2, $"{trained}");
            var resumed = TrainCliJson("resume", run, "-P", plugin, "--steps", "1", "-d", device.ToString());
            Check((int)resumed["steps"]! == 3, $"resume: {resumed}");
            var (runCode, answer, runError) = RunIdrakOn(device, null, "run", (string)resumed["checkpoint"]!, "-P", plugin, "--image", ScratchData("images/image.png"),
                "read the scan", "--max-tokens", "3", "--temperature", "0");
            Check(runCode == 0, $"run with an image: {answer} {runError}");
        }
        finally
        {
            Idrak.IdrakVision.IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliPretrainData(Device device)
    {
        if (!FirstRun(nameof(CliPretrainData)))
        {
            return;
        }

        string folder = TempFolder("cli-pretrain-data");
        try
        {
            string tokenizer = Path.Combine(folder, "tokenizer.json");
            var learned = TrainCliJson("data", "tokenizer", PretrainData("corpus.txt"), "--vocab", "512", "--min-frequency", "2", "-o", tokenizer);
            Check((int)learned["vocabulary"]! == 512 && (int)learned["special"]! == 8, $"{learned}");
            var expected = JsonNode.Parse(File.ReadAllText(PretrainData("tokenizer/tokenizer.json")))!["model"]!;
            var written = JsonNode.Parse(File.ReadAllText(tokenizer))!["model"]!;
            Check(JsonNode.DeepEquals(expected["vocab"], written["vocab"]) && JsonNode.DeepEquals(expected["merges"], written["merges"]),
                "data tokenizer learns the reference's vocabulary and merges");

            string tokens = Path.Combine(folder, "tokens.bin");
            var encoded = TrainCliJson("data", "tokens", PretrainData("corpus.txt"), "--tokenizer", tokenizer, "-o", tokens);
            Check(File.ReadAllBytes(tokens).AsSpan().SequenceEqual(File.ReadAllBytes(PretrainData("tokens.bin"))), $"data tokens writes the reference's token file: {encoded}");
            Check((long)encoded["documents"]! == 96 && File.Exists(tokens + ".json"), "96 documents, and the description beside it");
            var (code, _, error) = TrainCli("data", "tokens", PretrainData("corpus.txt"), "--tokenizer", tokenizer, "--separator", "<nope>", "-o", tokens);
            Check(code == 2 && error.Contains("no token '<nope>'", StringComparison.Ordinal), $"an unknown separator: {error}");
            (code, _, error) = TrainCli("data", "tokenizer", PretrainData("corpus.txt"), "--trainer", "nope", "-o", tokenizer);
            Check(code == 1 && error.Contains("is not registered", StringComparison.Ordinal) && error.Contains("-P", StringComparison.Ordinal), $"an unregistered trainer: {error}");

            // Plan 24, T4: --model picks a built-in trainer; unigram writes tokenizer.model beside tokenizer.json.
            string unigramJson = Path.Combine(folder, "unigram", "tokenizer.json");
            var unigram = TrainCliJson("data", "tokenizer", PretrainData("corpus.txt"), "--model", "unigram", "--vocab", "400", "--byte-fallback", "--no-special", "-o", unigramJson);
            string unigramModel = Path.Combine(folder, "unigram", "tokenizer.model");
            Check((string?)unigram["model"] == "Unigram" && (string?)unigram["trainer"] == "unigram" && File.Exists(unigramModel)
                && unigram["companions"]!.AsArray().Select(c => (string)c!).SequenceEqual([unigramModel]), $"data tokenizer --model unigram: {unigram}");
            var fromJson = PretrainedTokenizers.Load(unigramJson);
            var fromModel = PretrainedTokenizers.Load(unigramModel);
            foreach (string line in File.ReadLines(PretrainData("corpus.txt")).Take(50))
            {
                Check(fromJson.Encode(line).SequenceEqual(fromModel.Encode(line)), $"the CLI's tokenizer.json and tokenizer.model encode '{line}' alike");
            }

            // SentencePiece's switches, the normalization rule and the algorithm's constants from the command line, into
            // tokenizer.model.
            string switchedJson = Path.Combine(folder, "switched", "tokenizer.json");
            var switched = TrainCliJson("data", "tokenizer", PretrainData("corpus.txt"), "--model", "unigram", "--vocab", "400", "--no-special", "--normalization", "nmt_nfkc",
                "--max-sentence", "4192", "--no-whitespace-split", "--no-script-split", "--no-number-split", "--whitespace-suffix", "--whitespace-pieces",
                "--desired-size", "1.2", "--expected-floor", "0.4", "--unknown-penalty", "12", "-o", switchedJson);
            string switchedModel = Path.Combine(folder, "switched", "tokenizer.model");
            var spec = Idrak.Models.SentencePieceModel.Load(switchedModel);
            Check((string?)switched["model"] == "Unigram" && spec.NormalizerSpec.Name == "nmt_nfkc" && spec.NormalizerSpec.PrecompiledCharsmap.Length > 0
                && spec.TrainerSpec.MaxSentenceLength == 4192 && !spec.TrainerSpec.SplitByWhitespace && !spec.TrainerSpec.SplitByUnicodeScript && !spec.TrainerSpec.SplitByNumber
                && spec.TrainerSpec.TreatWhitespaceAsSuffix && spec.TrainerSpec.AllowWhitespaceOnlyPieces, $"data tokenizer --model unigram with the switches: {switched}");
            var switchedFromJson = PretrainedTokenizers.Load(switchedJson);
            var switchedFromModel = PretrainedTokenizers.Load(switchedModel);
            foreach (string line in File.ReadLines(PretrainData("corpus.txt")).Take(50))
            {
                Check(switchedFromJson.Encode(line).SequenceEqual(switchedFromModel.Encode(line)), $"with the switches, tokenizer.json and tokenizer.model encode '{line}' alike");
            }

            string wordpieceJson = Path.Combine(folder, "wordpiece.json");
            var wordpiece = TrainCliJson("data", "tokenizer", PretrainData("corpus.txt"), "--model", "wordpiece", "--vocab", "600", "--lowercase", "--scoring", "likelihood",
                "--arabic", "tatweel,diacritics", "-o", wordpieceJson);
            Check((string?)wordpiece["model"] == "WordPiece" && (int)wordpiece["special"]! == 5 && PretrainedTokenizers.Load(wordpieceJson).IdOf("[CLS]") == 2,
                $"data tokenizer --model wordpiece: BERT's special tokens by default: {wordpiece}");
            var (_, text, _) = TrainCli("data", "tokenizer", PretrainData("corpus.txt"), "--model", "wordpiece", "--vocab", "600", "-o", wordpieceJson);
            Check(text.Contains("a WordPiece tokenizer", StringComparison.Ordinal) && !text.Contains("merges", StringComparison.Ordinal), $"the summary names the model: {text}");

            foreach (var (args, named) in new (string[] Args, string Named)[]
                     {
                         (["--model", "unigram", "--trainer", "bpe"], "not both"), (["--model", "sentencepiece"], "--model"),
                         (["--model", "unigram", "--character-coverage", "2"], "--character-coverage"), (["--model", "unigram", "--shrinking", "x"], "--shrinking"),
                         (["--model", "unigram", "--em-steps", "0"], "--em-steps"), (["--model", "wordpiece", "--scoring", "best"], "--scoring"),
                         (["--model", "wordpiece", "--arabic", "kashida"], "--arabic"), (["--model", "wordpiece", "--strip-accents", "--keep-accents"], "not both"),
                         (["--model", "unigram", "--normalization", "nfd"], "--normalization"), (["--model", "wordpiece", "--normalization", "nfkc"], "--normalization"),
                         (["--model", "unigram", "--max-sentence", "-1"], "--max-sentence"), (["--model", "unigram", "--max-sentence", "long"], "--max-sentence"),
                         (["--model", "unigram", "--desired-size", "0.5"], "--desired-size"), (["--model", "unigram", "--expected-floor", "-1"], "--expected-floor"),
                         (["--model", "unigram", "--unknown-penalty", "x"], "--unknown-penalty"), (["--model", "unigram", "--unknown-penalty", "-2"], "--unknown-penalty"),
                     })
            {
                (code, _, error) = TrainCli(["data", "tokenizer", PretrainData("corpus.txt"), .. args, "-o", tokenizer]);
                Check(code == 2 && error.Contains(named, StringComparison.Ordinal), $"data tokenizer {string.Join(' ', args)}: exit {code}, {error}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliPretrainRun(Device device)
    {
        string folder = TempFolder("cli-pretrain-run");
        try
        {
            // create init's file, made small, then create.
            string spec = Path.Combine(folder, "spec.json");
            TrainCliJson("create", "init", "-o", spec);
            string text = File.ReadAllText(spec).Replace("\"dim\": 256", "\"dim\": 32").Replace("\"layers\": 4", "\"layers\": 2").Replace("\"ffDim\": 768", "\"ffDim\": 64")
                .Replace("\"maxPositions\": 1024", "\"maxPositions\": 64").Replace("\"tokenizer\": \"tokenizer.json\"", $"\"tokenizer\": {JsonValue.Create(PretrainData("tokenizer/tokenizer.json")).ToJsonString()}");
            File.WriteAllText(spec, text);
            string model = Path.Combine(folder, "model");
            var created = TrainCliJson("create", spec, "-o", model, "--seed", "3", "-d", device.ToString());
            Check((long)created["parameters"]! > 0 && (string?)created["architecture"] == "LlamaForCausalLM", $"{created}");
            Check(created["spec"]!["vocabulary"]!.GetValue<int>() == 512 && created["spec"]!["kvHeads"]!.GetValue<int>() == 2 && created["spec"]!["headDim"]!.GetValue<int>() == 8,
                "the vocabulary from the tokenizer, kvHeads from the file, headDim from dim / heads");
            // The made model generates before any training: a failure here is the device's inference on this shape
            // (dim 32, heads 4 of 8, 2 key/value heads, vocabulary 512), not the training.
            var (madeCode, madeAnswer, madeError) = RunIdrakOn(device, null, "run", model, "read the scan", "--max-tokens", "3", "--temperature", "0");
            Check(madeCode == 0, $"run on the made model, before training (inference on {device}): {madeAnswer} {madeError}");

            // pretrain 6 steps, stopped by --timeout's interrupt? A run is stopped after 2 steps with --steps 2 and resumed with --steps 4.
            string run = Path.Combine(folder, "run");
            var first = TrainCliJson("pretrain", model, "--tokens", PretrainData("tokens.bin"), "--steps", "2", "--max-length", "32", "--batch-tokens", "64",
                "--lr", "3e-3", "--warmup", "0", "--schedule", "constant", "-o", run, "-d", device.ToString(), "--log-every", "1");
            Check((int)first["steps"]! == 2 && File.Exists(Path.Combine(run, "run.json")) && File.Exists(Path.Combine(run, "log.jsonl")), $"{first}");
            Check(first["final_loss"] is not null, $"the first run's losses are finite (training on {device}, {first["settings"]}): {File.ReadAllText(Path.Combine(run, "log.jsonl"))}");
            var resumed = TrainCliJson("resume", run, "--steps", "2", "-d", device.ToString());
            Check((int)resumed["steps"]! == 4 && ((string)resumed["checkpoint"]!).EndsWith("checkpoint-4", StringComparison.Ordinal), $"resume: {resumed}");
            Check(resumed["final_loss"] is not null, $"the resumed run's losses are finite (training on {device} from the saved moments, {resumed["settings"]}): {File.ReadAllText(Path.Combine(run, "log.jsonl"))}");
            var (firstCode, firstAnswer, firstError) = RunIdrakOn(device, null, "run", (string)first["checkpoint"]!, "read the scan", "--max-tokens", "3", "--temperature", "0");
            Check(firstCode == 0, $"run on the first run's checkpoint (2 steps, losses finite: the update or the save on {device} made the weights not finite): {firstAnswer} {firstError}");
            var again = TrainCliJson("resume", run, "-d", device.ToString());
            Check((int)again["steps"]! == 4 && ((string?)again["checkpoint"])?.EndsWith("checkpoint-4", StringComparison.Ordinal) == true,
                $"resuming a finished run trains nothing and names its last checkpoint: {again}");
            var shown = TrainCliJson("runs", "show", run);
            Check((int)shown["epochs"]! == 4, $"runs show reads the 4 steps: {shown["epochs"]}");
            var (code, output, _) = TrainCli("runs", "show", run);
            Check(code == 0 && output.Contains("Steps", StringComparison.Ordinal), output);
            Check(output.Contains("step 1", StringComparison.Ordinal) && !output.Contains("epoch 1", StringComparison.Ordinal), $"the chart counts steps, not epochs: {output}");
            var (runCode, answer, runError) = RunIdrakOn(device, null, "run", (string)resumed["checkpoint"]!, "read the scan", "--max-tokens", "3", "--temperature", "0");
            Check(runCode == 0, $"run on the resumed run's checkpoint (4 steps; the first run's ran: the resumed steps on {device} made the weights not finite): {answer} {runError}");

            // A config with two stages.
            string config = Path.Combine(folder, "pretrain.json");
            TrainCliJson("pretrain", "init", "-o", config);
            var json = JsonNode.Parse(File.ReadAllText(config), documentOptions: new System.Text.Json.JsonDocumentOptions { CommentHandling = System.Text.Json.JsonCommentHandling.Skip })!.AsObject();
            foreach (var stage in json["stages"]!.AsArray())
            {
                stage!["steps"] = 1;
                foreach (var source in stage["sources"]!.AsArray())
                {
                    source!["paths"] = new JsonArray(PretrainData("tokens.bin"));
                    source["kind"] = "tokens";
                }
            }

            json["max_length"] = 32;
            json["batch_tokens"] = 64;
            File.WriteAllText(config, json.ToJsonString());
            var staged = TrainCliJson("pretrain", model, "-C", config, "-o", Path.Combine(folder, "staged"), "-d", device.ToString());
            Check((int)staged["steps"]! == 2, $"two stages of one step: {staged}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliPretrainRefusals(Device device)
    {
        if (!FirstRun(nameof(CliPretrainRefusals)))
        {
            return;
        }

        string folder = TempFolder("cli-pretrain-refusals");
        try
        {
            string spec = Path.Combine(folder, "spec.json");
            string tokenizerJson = JsonValue.Create(PretrainData("tokenizer/tokenizer.json")).ToJsonString();
            File.WriteAllText(spec, "{\"format\": \"idrak-create/1\", \"architecture\": \"NoSuchForCausalLM\", \"tokenizer\": " + tokenizerJson
                + ", \"spec\": {\"dim\": 32, \"layers\": 1, \"heads\": 4, \"ffDim\": 64, \"maxPositions\": 32}}");
            var (code, _, error) = TrainCli("create", spec, "-o", Path.Combine(folder, "a"));
            Check(code == 1 && error.Contains("NoSuchForCausalLM", StringComparison.Ordinal), $"an unknown architecture: {error}");
            File.WriteAllText(spec, """{"format": "idrak-create/1", "tokenizer": "missing.json", "spec": {"dim": 32}}""");
            (code, _, error) = TrainCli("create", spec, "-o", Path.Combine(folder, "b"));
            Check(code == 2 && error.Contains("No tokenizer at", StringComparison.Ordinal), $"a missing tokenizer: {error}");
            File.WriteAllText(spec, "{\"tokenizer\": " + tokenizerJson + ", \"spec\": {\"dim\": 32, \"heads\": 4}}");
            (code, _, error) = TrainCli("create", spec, "-o", Path.Combine(folder, "c"));
            Check(code == 2 && error.Contains("\"layers\"", StringComparison.Ordinal), $"a missing spec key: {error}");
            (code, _, error) = TrainCli("pretrain", PretrainData("tiny-llama"), "-o", Path.Combine(folder, "d"));
            Check(code == 2 && error.Contains("--tokens", StringComparison.Ordinal), $"no sources: {error}");
            string run = Path.Combine(folder, "run");
            Check(TrainCli("pretrain", PretrainData("tiny-llama"), "--tokens", PretrainData("tokens.bin"), "--steps", "1", "--max-length", "32", "--batch-tokens", "32", "-o", run).Code == 0, "one step");
            (code, _, error) = TrainCli("pretrain", PretrainData("tiny-llama"), "--tokens", PretrainData("tokens.bin"), "--steps", "1", "-o", run);
            Check(code == 2 && error.Contains("idrak resume", StringComparison.Ordinal), $"a run folder in use: {error}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
