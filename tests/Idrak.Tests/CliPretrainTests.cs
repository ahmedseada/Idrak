// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

// Plan 19, phase A4: models made from scratch on the command line (data tokenizer, data tokens, create, pretrain, resume).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliPretrainGroup =
    [
        ("cli pretrain: data tokenizer learns the reference's tokenizer from the corpus, and data tokens writes the reference's token file", CliPretrainData),
        ("cli pretrain: create makes a model from a spec file (and create init's), pretrain trains it, Ctrl+C-like stops checkpoint, resume finishes, runs show reads the steps, run answers", CliPretrainRun),
        ("cli pretrain: an unknown architecture, a missing spec key, a missing tokenizer, no sources and a run folder in use are refused", CliPretrainRefusals),
    ];

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
            (code, _, error) = TrainCli("data", "tokenizer", PretrainData("corpus.txt"), "--trainer", "wordpiece", "-o", tokenizer);
            Check(code == 1 && error.Contains("is not registered", StringComparison.Ordinal) && error.Contains("-P", StringComparison.Ordinal), $"an unregistered trainer: {error}");
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

            // pretrain 6 steps, stopped by --timeout's interrupt? A run is stopped after 2 steps with --steps 2 and resumed with --steps 4.
            string run = Path.Combine(folder, "run");
            var first = TrainCliJson("pretrain", model, "--tokens", PretrainData("tokens.bin"), "--steps", "2", "--max-length", "32", "--batch-tokens", "64",
                "--lr", "3e-3", "--warmup", "0", "--schedule", "constant", "-o", run, "-d", device.ToString(), "--log-every", "1");
            Check((int)first["steps"]! == 2 && File.Exists(Path.Combine(run, "run.json")) && File.Exists(Path.Combine(run, "log.jsonl")), $"{first}");
            var resumed = TrainCliJson("resume", run, "--steps", "2", "-d", device.ToString());
            Check((int)resumed["steps"]! == 4 && ((string)resumed["checkpoint"]!).EndsWith("checkpoint-4", StringComparison.Ordinal), $"resume: {resumed}");
            var shown = TrainCliJson("runs", "show", run);
            Check((int)shown["epochs"]! == 4, $"runs show reads the 4 steps: {shown["epochs"]}");
            var (code, output, _) = TrainCli("runs", "show", run);
            Check(code == 0 && output.Contains("Steps", StringComparison.Ordinal), output);
            var (runCode, answer, runError) = RunIdrakOn(device, null, "run", (string)resumed["checkpoint"]!, "read the scan", "--max-tokens", "3", "--temperature", "0");
            Check(runCode == 0, $"run on the pretrained model: {answer} {runError}");

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
            Check(code == 2 && error.Contains("No tokenizer.json", StringComparison.Ordinal), $"a missing tokenizer: {error}");
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
