// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Commands.Data;
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// The idrak CLI's Train and data group: tune, train, resume, runs, predict, package, distill, and data with its
// subcommands, run in-process on data built here and tiny models.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliTrainDataGroup =
    [
        ("cli train data: help, short forms and examples of every command", CliTrainDataHelp),
        ("cli train data: tune and data usage, errors and exit codes", CliTuneDataUsage),
        ("cli train data: tune init, tune --config, tune info and tune train on a tiny model", CliTune),
        ("cli train data: data show, count, build, preview, validate, stats", CliDataInspect),
        ("cli train data: data convert, dedupe, split, sample, mix", CliDataTransform),
        ("cli train data: train, predict, runs, resume, package on CSV", CliTrainCsv),
        ("cli train data: train and predict on an image folder", CliTrainImages),
        ("cli train data: distill on the fly, precomputed top-k logits then trained from the file, teacher-written answers; a vocabulary mismatch is refused", CliDistill),
    ];

    private static (int Code, string Out, string Err) TrainCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject TrainCliJson(params string[] args)
    {
        var (code, output, error) = TrainCli([.. args, "--json"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} --json: exit {code}\n{output}\n{error}");
        return JsonNode.Parse(output) as JsonObject ?? throw new Exception($"not one JSON object: {output}");
    }

    private static void CliTrainDataHelp(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        string[] providers = ["ollama", "openai", "anthropic", "claude"];
        foreach (var command in Idrak.Cli.Commands.TrainCommands.All.Concat(Idrak.Cli.Commands.DataCommands.All))
        {
            var (code, output, _) = TrainCli(["help", .. command.Name.Split(' ')]);
            Check(code == 0 && output.Contains($"idrak {command.Name}", StringComparison.Ordinal), $"help {command.Name}: {code}\n{output}");
            Check(output.Contains("Example", StringComparison.Ordinal) && output.Contains("Environment (idrak help env for all):", StringComparison.Ordinal), $"help {command.Name} has examples and its environment");
            Check(!providers.Any(p => System.Text.RegularExpressions.Regex.Replace(output, @"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", "").Contains(p, StringComparison.OrdinalIgnoreCase)), $"help {command.Name} names no provider");
            Check(command.ShortForms.Keys.All(k => k.Length == 2 && !CommandContext.CommonShortForms.ContainsKey(k)), $"{command.Name}: short forms are one letter and not common ones");
            Check(command.ShortForms.Values.Distinct().Count() == command.ShortForms.Count, $"{command.Name}: no two short forms for one option");
            Check(command.ShortForms.Values.All(v => command.ValueOptions.Contains(v) || command.Flags.Contains(v)), $"{command.Name}: every short form names an option");
        }

        // The same letter means the same option in every command of the group.
        var meanings = Idrak.Cli.Commands.TrainCommands.All.Concat(Idrak.Cli.Commands.DataCommands.All).SelectMany(c => c.ShortForms).GroupBy(p => p.Key);
        foreach (var group in meanings)
        {
            Check(group.Select(p => p.Value).Distinct().Count() == 1, $"{group.Key} means {string.Join(" and ", group.Select(p => p.Value).Distinct())}");
        }

        Check(TrainCli("train").Code == 2, "train without a spec is a usage error");
        Check(TrainCli("tune").Code == 2 && TrainCli("tune", "fly", "m").Code == 2 && TrainCli("tune", "train", "m", "d.jsonl").Code == 2, "tune usage errors exit with 2");
        Check(TrainCli("tune", "train", "m", "d.jsonl", "-o", "x", "--optimizer", "nope").Err.Contains("--optimizer nope", StringComparison.Ordinal), "a bad tune option is named");
        Check(TrainCli("data", "fly").Code == 2 && TrainCli("data", "validate", "x.jsonl").Code == 2, "data usage errors exit with 2");
    }

    private static void CliTuneDataUsage(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // The usage text is idrak's own: no other tool is named.
        var tuneHelp = TrainCli("help", "tune");
        Check(tuneHelp.Code == 0 && tuneHelp.Out.Contains("Fine-tune pretrained language models", StringComparison.Ordinal)
            && tuneHelp.Out.Contains("idrak tune train <model> <data…> --out <dir>", StringComparison.Ordinal)
            && tuneHelp.Out.Contains("as idrak data reads them", StringComparison.Ordinal), $"help tune: {tuneHelp.Out}");
        var dataHelp = TrainCli("data", "help");
        Check(dataHelp.Code == 0 && dataHelp.Out.Contains("Inspect, download and assemble datasets.", StringComparison.Ordinal)
            && dataHelp.Out.Contains("idrak data show <spec…>", StringComparison.Ordinal), $"data help: {dataHelp.Out}");
        Check(!(tuneHelp.Out + dataHelp.Out).Contains("idrak-tune ", StringComparison.Ordinal) && !(tuneHelp.Out + dataHelp.Out).Contains("idrak-data", StringComparison.Ordinal), "the tune and data help name no separate tool");

        // Usage errors exit with 2 and name the problem.
        Check(TrainCli("tune", "train", "m", "d", "-o", "x", "--rank", "x").Code == 2, "tune: a bad number is a usage error");
        var unknown = TrainCli("tune", "train", "m", "d", "-o", "x", "--bogus");
        Check(unknown.Code == 2 && unknown.Err.Contains("--bogus", StringComparison.Ordinal), $"tune names an unknown option: {unknown.Err}");
        Check(TrainCli("data").Code == 2 && TrainCli("data", "build", "x.jsonl").Code == 2, "data without a command, or build without --out, is a usage error");

        string folder = TempFolder();
        try
        {
            string file = Path.Combine(folder, "rows.jsonl");
            File.WriteAllLines(file, ["{\"question\": \"1+1?\", \"answer\": \"2\"}", "{\"question\": \"2+2?\", \"answer\": \"4\"}"]);
            var count = TrainCli("data", "count", file, "--cache", folder);
            Check(count.Code == 0 && count.Out.Contains(": 2 rows", StringComparison.Ordinal), $"data count: {count.Out}");
            var missing = TrainCli("data", "count", Path.Combine(folder, "missing.jsonl"), "--cache", folder);
            Check(missing.Code == 1 && missing.Err.Length > 0, $"data exits with 1 on a missing file: {missing.Code} {missing.Err}");
            var built = TrainCli("data", "build", file, "--out", Path.Combine(folder, "out.jsonl"), "--cache", folder);
            Check(built.Code == 0 && File.ReadAllLines(Path.Combine(folder, "out.jsonl")).Length == 2, $"data build: {built.Out}{built.Err}");

            // A tune.json's format, when given, must be a tune.json's.
            string other = Path.Combine(folder, "train.json");
            File.WriteAllText(other, "{ \"format\": \"idrak-train/1\", \"command\": \"info\", \"model\": \"x\" }");
            var wrong = TrainCli("tune", "--config", other);
            Check(wrong.Code == 2 && wrong.Err.Contains("is not a tune.json", StringComparison.Ordinal), $"tune --config with another file's format: {wrong.Err}");
            File.WriteAllText(other, "{ \"format\": \"idrak-tune/1\", \"command\": \"info\", \"model\": \"no-such-model\" }");
            var right = TrainCli("tune", "--config", other, "--offline");
            Check(!right.Err.Contains("format", StringComparison.Ordinal) && right.Err.Contains("no-such-model", StringComparison.Ordinal), $"tune --config accepts its own format (as idrak suggest writes it): {right.Err}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliTune(Device device)
    {
        string model = WriteChatModel(TinyChatSpec());
        string folder = TempFolder();
        try
        {
            string d = device.ToString();
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, Enumerable.Range(0, 12).Select(i =>
                new JsonObject { ["question"] = $"What is {i} plus {i}?", ["answer"] = $"{i + i}" }.ToJsonString()));

            // tune init writes a commented tune.json that tune --config reads (the command line wins).
            string settings = Path.Combine(folder, "tune.json");
            var init = TrainCli("tune", "init", "-b", model, "--data", data, "-o", settings, "-d", d);
            Check(init.Code == 0 && File.ReadAllText(settings).Contains("// What to do", StringComparison.Ordinal), $"tune init: {init.Out}{init.Err}");
            Check(File.ReadAllText(settings).Contains($"\"format\": \"{TuneCommand.Format}\"", StringComparison.Ordinal), "tune init writes the format");
            Check(TrainCli("tune", "init", "-b", model, "-o", settings).Code == 1, "tune init does not overwrite without --force");
            var info = TrainCliJson("tune", "info", "--config", settings);
            Check((string?)info["command"] == "info" && (string?)info["model"] == model && (string?)info["device"] == d, $"tune info from tune.json: {info}");
            Check(info["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("chat template:", StringComparison.Ordinal)), $"tune info output: {info}");

            var (defaultDevice, precision) = (Device.Default, MixedPrecision.Default);

            // A few steps of LoRA training, through idrak's options (-b, -o, a tune.json for the rest).
            string adapter = Path.Combine(folder, "adapter");
            var train = TrainCli("tune", "train", data, "-b", model, "-o", adapter, "--config", settings, "--rank", "4", "--alpha", "8", "--max-length", "64",
                "--batch-tokens", "256", "--epochs", "1", "--eval-fraction", "0", "-d", d);
            Check(train.Code == 0 && File.Exists(Path.Combine(adapter, "idrak-tuning.json")), $"tune train: {train.Out}\n{train.Err}");
            Check(train.Out.Contains("trained in", StringComparison.Ordinal) && train.Out.Contains($"training on {d}", StringComparison.Ordinal), $"tune train output: {train.Out}");
            Check(Device.Default == defaultDevice && MixedPrecision.Default == precision, "tune puts back the default device and precision");

            // The adapter folder as the model: its base model and the adapter.
            var adapted = TrainCli("tune", "info", adapter, "-d", d, "-q");
            Check(adapted.Code == 0 && adapted.Out.Length == 0, $"tune info ADAPTER -q: {adapted.Code} {adapted.Err}");
            var unknown = TrainCli("tune", "--config", Path.Combine(folder, "bad.json"));
            Check(unknown.Code != 0, "a missing tune.json fails");
            File.WriteAllText(Path.Combine(folder, "bad.json"), "{ \"command\": \"info\", \"model\": \"x\", \"rnak\": 4 }");
            Check(TrainCli("tune", "--config", Path.Combine(folder, "bad.json")).Err.Contains("unknown key 'rnak'", StringComparison.Ordinal), "a misspelt tune.json key is named");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    private static void CliDataInspect(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        string folder = TempFolder();
        string model = WriteChatModel(TinyChatSpec());
        try
        {
            string chats = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(chats,
            [
                "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello there\"}, {\"role\": \"assistant\", \"content\": \"Hi! How can I help?\"}]}",
                "{\"question\": \"What is two plus two?\", \"answer\": \"Four.\"}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"No answer here\"}]}",
            ]);

            // show, count and build, with idrak's cache.
            var show = TrainCli("data", "show", chats, "--take", "2", "--cache", folder);
            Check(show.Code == 0 && show.Out.Contains("normalized:", StringComparison.Ordinal), $"data show: {show.Out}{show.Err}");
            var count = TrainCliJson("data", "count", chats, "--cache", folder);
            Check((string?)count["command"] == "count" && count["output"]![0]!.ToString().Contains(": 3 rows", StringComparison.Ordinal), $"data count: {count}");
            var built = TrainCli("data", "build", chats, "-o", Path.Combine(folder, "train.jsonl"), "--kind", "chat", "--cache", folder);
            Check(built.Code == 0 && File.ReadAllLines(Path.Combine(folder, "train.jsonl")).Length == 3, $"data build: {built.Out}{built.Err}");

            // preview: columns and types of a CSV and a Parquet file.
            string csv = Path.Combine(folder, "table.csv");
            File.WriteAllText(csv, "id,size,city,price\n1,50.5,Cairo,100\n2,70,Giza,140\n3,,Cairo,90\n");
            var preview = TrainCliJson("data", "preview", csv, "-n", "2");
            Check((long?)preview["rows"] == 3 && preview["columns"]!.AsArray().Count == 4 && preview["preview"]!.AsArray().Count == 2, $"data preview: {preview}");
            Check((string?)preview["columns"]![1]!["type"] == "number" && (string?)preview["columns"]![2]!["type"] == "text", $"data preview types: {preview}");
            string parquet = Path.Combine(AppContext.BaseDirectory, "data", "parquet", "snappy-v1.parquet");
            if (!File.Exists(parquet))
            {
                parquet = Path.Combine(FindRepositoryRoot(), "tests", "Idrak.Tests", "data", "parquet", "snappy-v1.parquet");
            }

            Check(TrainCli("data", "preview", parquet).Code == 0, "data preview of a Parquet file");

            // validate: the first bad rows and exit 1; a good table exits 0.
            var chat = TrainCli("data", "validate", chats, "--as", "chat");
            Check(chat.Code == 1 && chat.Out.Contains("2 valid, 1 invalid", StringComparison.Ordinal) && chat.Out.Contains("row 3: no assistant turn", StringComparison.Ordinal), $"validate chat: {chat.Out}");
            Check(TrainCli("data", "validate", csv, "--as", "table", "-t", "price").Code == 0, "a good table validates");
            File.AppendAllText(csv, "4,big,Cairo,10\n");
            var tableText = TrainCli("data", "validate", csv, "--as", "table");
            Check(tableText.Code == 1 && tableText.Out.Contains("'size' is big, not a number", StringComparison.Ordinal), $"validate table: {tableText.Out}");
            string prefs = Path.Combine(folder, "prefs.jsonl");
            File.WriteAllLines(prefs, ["{\"prompt\": \"Hi\", \"chosen\": \"Hello!\", \"rejected\": \"Go away.\"}", "{\"prompt\": \"Hi\", \"chosen\": \"Same\", \"rejected\": \"Same\"}"]);
            var preference = TrainCli("data", "validate", prefs, "--as", "preference");
            Check(preference.Code == 1 && preference.Out.Contains("chosen and rejected are the same", StringComparison.Ordinal), $"validate preference: {preference.Out}");

            // stats: characters and words, and tokens with a model's tokenizer and chat template.
            var stats = TrainCliJson("data", "stats", chats, "-m", model, "--context", "20");
            Check((int?)stats["rows"] == 3 && (int?)stats["conversations"] == 3 && stats["tokens"]?["max"] is not null, $"data stats: {stats}");
            Check((long?)stats["overContext"] >= 1, $"data stats counts rows over the context: {stats}");
            var plain = TrainCli("data", "stats", csv, "--column", "city");
            Check(plain.Code == 0 && plain.Out.Contains("characters", StringComparison.Ordinal), $"data stats without a model: {plain.Out}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Idrak.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName ?? Directory.GetCurrentDirectory();
    }

    private static void CliDataTransform(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        string folder = TempFolder();
        try
        {
            string rows = Path.Combine(folder, "rows.jsonl");
            var lines = Enumerable.Range(0, 40).Select(i => new JsonObject { ["instruction"] = $"Say {i % 20}", ["output"] = i % 20 == 3 ? "THREE!" : $"{i % 20}", ["label"] = i % 4 == 0 ? "a" : "b" }.ToJsonString()).ToList();
            lines.Add(new JsonObject { ["instruction"] = "say  1", ["output"] = "1", ["label"] = "b" }.ToJsonString());
            File.WriteAllLines(rows, lines);

            // convert: to CSV and back, and Alpaca rows to chat rows.
            string csv = Path.Combine(folder, "rows.csv");
            Check(TrainCliJson("data", "convert", rows, csv)["written"]!.GetValue<long>() == 41, "data convert to CSV");
            Check(TrainCli("data", "convert", rows, csv).Code == 1 && TrainCli("data", "convert", rows, csv, "-f").Code == 0, "data convert needs -f to overwrite");
            Check(File.ReadAllLines(csv)[0] == "instruction,output,label", "CSV header");
            string chats = Path.Combine(folder, "chats.jsonl");
            Check(TrainCli("data", "convert", csv, chats, "--as", "chat", "-s", "Be brief.").Code == 0, "data convert --as chat");
            var first = JsonNode.Parse(File.ReadLines(chats).First())!;
            Check((string?)first["messages"]![0]!["role"] == "system" && (string?)first["messages"]![2]!["content"] == "0", $"chat rows: {first}");

            // dedupe: exact (20 repeats), near (one more: "say  1" is "Say 1"), in place only with -y.
            var dry = TrainCliJson("data", "dedupe", rows, "--dry-run", "--columns", "instruction,output");
            Check((int?)dry["duplicates"] == 20 && dry["output"] is null, $"dedupe dry run: {dry}");
            Check(TrainCli("data", "dedupe", rows) is { Code: 2 } inPlace && inPlace.Err.Contains("--yes", StringComparison.Ordinal), "dedupe in place needs --yes (no terminal to ask on)");
            var near = TrainCliJson("data", "dedupe", rows, "--near", "--columns", "instruction,output", "-o", Path.Combine(folder, "unique.jsonl"));
            Check((int?)near["duplicates"] == 21 && File.ReadAllLines(Path.Combine(folder, "unique.jsonl")).Length == 20, $"near dedupe: {near}");

            // split: stratified, with every part keeping the label balance; sample: stratified too.
            var split = TrainCliJson("data", "split", rows, "--validation", "0.2", "--test", "0.2", "-t", "label", "--seed", "3", "-o", Path.Combine(folder, "parts"));
            Check((int?)split["parts"]!["train"]!["rows"] + (int?)split["parts"]!["validation"]!["rows"] + (int?)split["parts"]!["test"]!["rows"] == 41, $"split: {split}");
            Check(File.Exists(Path.Combine(folder, "parts.test.jsonl")), "split files");
            var testRows = File.ReadAllLines(Path.Combine(folder, "parts.test.jsonl")).Select(l => (string?)JsonNode.Parse(l)!["label"]).ToList();
            Check(testRows.Count(l => l == "a") == 2, $"stratified test part: {string.Join(",", testRows)}");
            var sample = TrainCliJson("data", "sample", rows, "-n", "8", "-t", "label");
            var labels = sample["sample"]!.AsArray().Select(r => (string?)r!["label"]).ToList();
            Check(labels.Count == 8 && labels.Count(l => l == "a") == 2, $"stratified sample: {string.Join(",", labels)}");
            Check(TrainCli("data", "sample", rows).Code == 2, "sample needs -n");

            // mix: a recipe of two local sources.
            string recipe = Path.Combine(folder, "recipe.json");
            File.WriteAllText(recipe, new JsonObject
            {
                ["sources"] = new JsonArray(rows, new JsonObject { ["source"] = chats, ["take"] = 5 }),
                ["seed"] = 1,
                ["eval_fraction"] = 0.1,
            }.ToJsonString());
            var mix = TrainCliJson("data", "mix", recipe, "-o", Path.Combine(folder, "mixed.jsonl"), "--cache", folder);
            Check((long?)mix["rows"] > 0 && File.Exists(Path.Combine(folder, "mixed.eval.jsonl")), $"data mix: {mix}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliTrainCsv(Device device)
    {
        string folder = TempFolder();
        try
        {
            string d = device.ToString(), cache = Path.Combine(folder, "cache");
            var random = new Random(5);
            var csv = new StringBuilder("id,x1,x2,y\n");
            var classes = new StringBuilder("x1,x2,colour\n");
            for (int i = 0; i < 240; i++)
            {
                double a = random.NextDouble() * 4 - 2, b = random.NextDouble() * 4 - 2;
                csv.Append(CultureInfo.InvariantCulture, $"{i},{a:F4},{b:F4},{3 * a - 2 * b + 10:F4}\n");
                classes.Append(CultureInfo.InvariantCulture, $"{a:F4},{b:F4},{(a + b > 0.5 ? "red" : a - b > 0 ? "blue" : "green")}\n");
            }

            File.WriteAllText(Path.Combine(folder, "reg.csv"), csv.ToString());
            File.WriteAllText(Path.Combine(folder, "cls.csv"), classes.ToString());
            File.WriteAllText(Path.Combine(folder, "reg.json"), Network.Input(2).Linear(16).ReLU().Linear(1).Named("reg").ToJson().ToJsonString());
            File.WriteAllText(Path.Combine(folder, "cls.json"), Network.Input(2).Linear(24).ReLU().Linear(3).Named("cls").ToJson().ToJsonString());

            // Regression: settings from a train.json, the command line winning.
            File.WriteAllText(Path.Combine(folder, "train.json"), "{ // from idrak suggest\n \"target\": \"y\", \"epochs\": 5, \"learning_rate\": 0.01, \"batch_size\": 16 }");
            var reg = TrainCliJson("train", Path.Combine(folder, "reg.json"), "--data", Path.Combine(folder, "reg.csv"), "--ignore", "id", "--config", Path.Combine(folder, "train.json"),
                "--epochs", "80", "-o", Path.Combine(folder, "reg.ikm"), "--cache", cache, "-d", d);
            Check((string?)reg["task"] == "regression" && (int?)reg["epochs"] >= 40 && File.Exists(Path.Combine(folder, "reg.ikm")), $"train regression: {reg}");
            Check((double?)reg["metrics"]!["r2"] > 0.9, $"the regression fits: {reg["metrics"]}");
            string run = (string)reg["run"]!;
            Check(File.Exists(Path.Combine(run, "last.ikw")) && File.Exists(Path.Combine(run, "log.jsonl")) && File.Exists(Path.Combine(run, "run.json")), "the run folder");

            // predict: rows without the target, by column name.
            File.WriteAllText(Path.Combine(folder, "new.csv"), "x2,x1\n0,1\n1,0\n");
            var predicted = TrainCliJson("predict", Path.Combine(folder, "reg.ikm"), "-i", Path.Combine(folder, "new.csv"), "-d", d);
            double p0 = (double)predicted["predictions"]![0]!["prediction"]!, p1 = (double)predicted["predictions"]![1]!["prediction"]!;
            Check(Math.Abs(p0 - 13) < 1.5 && Math.Abs(p1 - 8) < 1.5, $"predictions 13 and 8: {p0}, {p1}");
            Check(TrainCli("predict", Path.Combine(folder, "reg.ikm"), "-i", Path.Combine(folder, "new.csv"), "-o", Path.Combine(folder, "out.csv")).Code == 0
                  && File.ReadAllLines(Path.Combine(folder, "out.csv"))[0] == "x2,x1,prediction", "predict -o writes the rows with the prediction");

            // Classification with class names in the CSV.
            var cls = TrainCliJson("train", Path.Combine(folder, "cls.json"), "--data", Path.Combine(folder, "cls.csv"), "--epochs", "60", "--lr", "0.01",
                "-o", Path.Combine(folder, "cls.ikm"), "--cache", cache, "-d", d);
            Check((string?)cls["task"] == "classification" && cls["classes"]!.AsArray().Count == 3 && (double?)cls["metrics"]!["accuracy"] > 0.8, $"train classification: {cls}");
            var labels = TrainCliJson("predict", Path.Combine(folder, "cls.ikm"), "-i", Path.Combine(folder, "cls.csv"), "--top", "2", "-d", d);
            int right = labels["predictions"]!.AsArray().Count(r => (string?)r!["prediction"] == (string?)r["colour"]);
            Check(right > 190 && labels["predictions"]![0]!["top"]!.AsArray().Count == 2, $"classes predicted: {right} of 240");

            // runs: list, show, compare; resume; package.
            var list = TrainCliJson("runs", "list", "--cache", cache);
            Check(list["runs"]!.AsArray().Count == 2, $"runs list: {list}");
            Check(TrainCli("runs", "--cache", cache).Out.Contains("completed", StringComparison.Ordinal), "runs (alias) lists the runs");
            var show = TrainCli("runs", "show", Path.GetFileName(run), "--cache", cache);
            Check(show.Code == 0 && show.Out.Contains("val_loss", StringComparison.Ordinal) && show.Out.Contains("epoch 1", StringComparison.Ordinal), $"runs show: {show.Out}");
            var shown = TrainCliJson("runs", "show", Path.Combine(run, "log.jsonl"));
            int epochs = (int)shown["epochs"]!;
            Check(epochs == (int)reg["epochs"]! && shown["history"]!.AsArray().Count == epochs, $"runs show --json: {shown["epochs"]}");
            var compared = TrainCliJson("runs", "compare", run, (string)cls["run"]!);
            Check(compared["runs"]!.AsArray().Count == 2 && compared["lowest"] is not null, "runs compare");
            var resumed = TrainCliJson("resume", run, "--epochs", "3", "--cache", cache, "-d", d);
            Check((int?)resumed["epochs"] == epochs + 3, $"resume: {resumed["epochs"]} after {epochs}");
            Check(TrainCliJson("runs", "show", run)["epochs"]!.GetValue<int>() == epochs + 3, "the resumed run's log continues");
            var packaged = TrainCliJson("package", "--model", run, "-o", Path.Combine(folder, "again.ikm"), "--checkpoint", "last");
            Check(packaged["included"]!.AsArray().Any(i => (string?)i == "training.json"), $"package: {packaged}");
            var again = TrainCliJson("predict", Path.Combine(folder, "again.ikm"), "-i", Path.Combine(folder, "new.csv"), "-d", d);
            Check(Math.Abs((double)again["predictions"]![0]!["prediction"]! - 13) < 1.5, "the packaged run predicts");
            Check(TrainCli("resume", "nothing-here", "--cache", cache).Code == 2, "resume of an unknown run is a usage error");
            var mismatch = TrainCli("train", Path.Combine(folder, "reg.json"), "--data", Path.Combine(folder, "reg.csv"), "--cache", cache, "-d", d);
            Check(mismatch.Code == 1 && mismatch.Err.Contains("3 feature columns", StringComparison.Ordinal), $"a network that does not fit the data is explained: {mismatch.Err}");
            var notClasses = TrainCli("train", Path.Combine(folder, "cls.json"), "--data", Path.Combine(folder, "reg.csv"), "--ignore", "id", "--cache", cache, "-d", d);
            Check(notClasses.Code == 1 && notClasses.Err.Contains("not class indices", StringComparison.Ordinal), $"a classifier on a numeric target is explained: {notClasses.Err}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliTrainImages(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Two classes of 8x8 grey images: a bright left half or a bright right half (with noise), as PGM files.
            var random = new Random(9);
            foreach (string label in new[] { "left", "right" })
            {
                Directory.CreateDirectory(Path.Combine(folder, "images", label));
                for (int n = 0; n < 24; n++)
                {
                    var text = new StringBuilder("P2\n8 8\n255\n");
                    for (int y = 0; y < 8; y++)
                    {
                        for (int x = 0; x < 8; x++)
                        {
                            bool bright = label == "left" ? x < 4 : x >= 4;
                            text.Append(Math.Clamp((bright ? 200 : 40) + random.Next(-30, 30), 0, 255)).Append(' ');
                        }

                        text.Append('\n');
                    }

                    File.WriteAllText(Path.Combine(folder, "images", label, $"{n}.pgm"), text.ToString());
                }
            }

            File.WriteAllText(Path.Combine(folder, "cnn.json"),
                Network.Image(1, 8, 8).Conv2d(4, 3, padding: 1).ReLU().MaxPool2d(2).Flatten().Linear(2).Named("halves").ToJson().ToJsonString());
            string d = device.ToString();
            var trained = TrainCliJson("train", Path.Combine(folder, "cnn.json"), "--data", Path.Combine(folder, "images"), "--epochs", "30", "--lr", "0.01", "--batch", "8",
                "-o", Path.Combine(folder, "halves.ikm"), "--cache", Path.Combine(folder, "cache"), "-d", d);
            Check((string?)trained["task"] == "classification" && (double?)trained["metrics"]!["accuracy"] >= 0.9, $"train on images: {trained}");
            var predicted = TrainCliJson("predict", Path.Combine(folder, "halves.ikm"), "-i", Path.Combine(folder, "images", "right"), "-d", d);
            int right = predicted["predictions"]!.AsArray().Count(r => (string?)r!["prediction"] == "right");
            Check(right >= 22, $"images predicted: {right} of 24");
            var table = TrainCli("predict", Path.Combine(folder, "halves.ikm"), "-i", Path.Combine(folder, "images", "right"), "-d", d);
            var lines = table.Out.Split('\n').Select(l => l.TrimEnd('\r')).ToList();
            Check(table.Code == 0 && lines.Any(l => l.StartsWith("12.pgm ", StringComparison.Ordinal)) && lines.Any(l => l.StartsWith("3.pgm ", StringComparison.Ordinal)),
                $"the table names each file relative to the input folder:\n{table.Out}");

            // A package from Predictor.Save (no training entry): its class names make it a classifier for predict too.
            var images = new ImageFolderSource(Path.Combine(folder, "images"), 1, 8, 8);
            var network = Network.Image(1, 8, 8).OnDevice(device).Seed(3).Conv2d(4, 3, padding: 1).ReLU().MaxPool2d(2).Flatten().Linear(2);
            using var model = network.Build();
            new TrainingRun
            {
                Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new Adam(p, 0.01f),
                Train = images.ToDataset().Batches(8, shuffle: true, device: device, seed: 1), Epochs = 30,
            }.Fit();
            using (var saved = Predictor.For(model).InputShape(1, 8, 8).Softmax().Classes(images.Classes).Build())
            {
                saved.Save(Path.Combine(folder, "library.ikm"));
            }

            var fromLibrary = TrainCliJson("predict", Path.Combine(folder, "library.ikm"), "-i", Path.Combine(folder, "images", "right"), "--top", "2", "-d", d);
            int libraryRight = fromLibrary["predictions"]!.AsArray().Count(r => (string?)r!["prediction"] == "right" && r["top"]!.AsArray().Count == 2);
            Check((string?)fromLibrary["task"] == "classification" && libraryRight >= 22, $"Predictor.Save package: {libraryRight} of 24, task {fromLibrary["task"]}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
