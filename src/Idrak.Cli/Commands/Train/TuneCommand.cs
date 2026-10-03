// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak tune</c>: the fine-tuning tool (idrak-tune) as a subcommand, with idrak's model arguments (aliases,
/// <c>-w</c>/<c>--weights</c>, <c>-k</c>/<c>--kv</c>), <c>-b</c>/<c>--base</c> for the model, and settings from a
/// tune.json given with <c>--config</c> (written by <c>idrak tune init</c>; the command line wins).
/// </summary>
internal sealed class TuneCommand : Command
{
    // Keys of a tune.json that are not tune options: the config file's own and the command, model and data.
    private static readonly string[] OwnKeys = ["device", "cache", "plugins", "aliases", "token", "hf_token", "profiles", "command", "model", "data", "base", "weights"];

    public override string Name => "tune";

    public override string Summary => "Fine-tune language models (LoRA, QLoRA, DoRA, DPO/ORPO/SimPO); evaluate, chat, export, download, info";

    public override string Usage => "COMMAND MODEL [DATA...] [options]\n\n"
        + TuneTool.Usage.Replace("idrak-tune: fine-tune", "Fine-tune", StringComparison.Ordinal).Replace("idrak-tune ", "idrak tune ", StringComparison.Ordinal)
            .Replace("as idrak-data reads them", "as idrak data reads them", StringComparison.Ordinal)
            .Replace("a .gguf file, ollama:name, or an", "a .gguf file, or an", StringComparison.Ordinal)
        + """

        idrak tune also takes:
          -b, --base MODEL     the model as an option (then every argument after the command is data)
          -w, --weights F      base weights: int8, int4 (QLoRA) or bf16 (as --int8, --int4, --bf16)
          -k, --kv F           the KV cache format; -o, --out DIR; -s, --system TEXT
          -C, --config FILE    a tune.json (idrak tune init writes one): its keys are these options without the dashes,
                               plus "command", "model" and "data"; options on the command line win
          MODEL may be an alias from the config (its weights and KV format apply unless given).
          --json prints one JSON document at the end with the command's output lines.

        Examples:
          idrak tune init -b Qwen/Qwen3-0.6B --data chats.jsonl && idrak tune --config tune.json
          idrak tune train Qwen/Qwen3-0.6B chats.jsonl -o adapters/chat -w int4 --epochs 2
          idrak tune train -b qwen prefs.jsonl --loss dpo -o adapters/dpo -d cuda:0
          idrak tune evaluate adapters/chat held-out.jsonl --adapter adapters/chat
          idrak tune export adapters/chat -o merged
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        [.. TuneTool.ValueOptions.Where(o => o is not ("--device" or "--seed")), "--weights", "--base"];      // --device and --seed are common options

    public override IReadOnlyCollection<string> Flags => TuneTool.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-o"] = "--out", ["-b"] = "--base", ["-w"] = "--weights", ["-k"] = "--kv", ["-s"] = "--system",
    };

    public override int Run(CommandContext context)
    {
        var settings = ReadSettings(context);
        var positional = context.Positional.ToList();
        string? command = positional.Count > 0 && TuneTool.Commands.Contains(positional[0]) ? positional[0] : null;
        if (command is not null)
        {
            positional.RemoveAt(0);
        }

        command ??= Text(settings, "command")
            ?? throw new UsageException($"Give a command: {string.Join(", ", TuneTool.Commands)} (or --config tune.json from idrak tune init).");
        if (!TuneTool.Commands.Contains(command))
        {
            throw new UsageException($"Unknown tune command '{command}': use {string.Join(", ", TuneTool.Commands)}.");
        }

        // The model: --base, else the first argument, else the tune.json's; the data: the other arguments, else the tune.json's.
        string? model = context.Option("--base") ?? Text(settings, "base");
        if (context.Option("--base") is null && positional.Count > 0)
        {
            model = positional[0];
            positional.RemoveAt(0);
        }

        model ??= Text(settings, "model");
        var data = positional.Count > 0 ? positional : List(settings, "data");
        if (model is null)
        {
            throw new UsageException(command == "download" ? "download needs one or more model ids." : $"{command} needs a model (MODEL or -b, --base MODEL).");
        }

        var choice = Models.Choose(context, model);
        var args = new List<string> { command, choice.Model };
        args.AddRange(command == "download" ? data.Select(id => Models.Choose(context, id).Model) : data);

        // Options: the command line's, else the tune.json's.
        string? weights = context.Option("--weights") ?? choice.Weights ?? Text(settings, "weights");
        if (weights is not null)
        {
            args.Add(weights.ToLowerInvariant() switch
            {
                "int8" => "--int8",
                "int4" => "--int4",
                "bf16" or "bfloat16" => "--bf16",
                _ => throw new UsageException($"--weights {weights}: tune loads the base weights as int8, int4 or bf16 (leave it out for float32)."),
            });
        }

        if ((context.Option("--device") ?? context.Config.Get("device")) is { } device)
        {
            args.AddRange(["--device", device]);
        }

        foreach (string option in ValueOptions.Where(o => o is not ("--weights" or "--base")).Append("--seed"))
        {
            var values = option == "--kv" && choice.Kv is { } kv ? [kv] : context.Options(option);
            if (values.Count == 0 && Text(settings, option[2..]) is { } fromFile)
            {
                values = [fromFile];
            }

            foreach (string value in values)
            {
                args.AddRange([option, value]);
            }
        }

        foreach (string flag in Flags)
        {
            if (context.Flag(flag) || settings?[flag[2..]] is JsonValue v && v.TryGetValue(out bool on) && on)
            {
                args.Add(flag);
            }
        }

        // The tool itself: its output to idrak's writers (captured for --json, dropped with --quiet).
        var captured = context.Json ? new StringWriter() : null;
        var output = captured ?? (context.Quiet ? TextWriter.Null : context.Output);
        var console = ToolHost.Console(context, output);
        var tool = new TuneTool(console);
        try
        {
            tool.Parse(args);
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            throw new UsageException(e.Message);
        }

        if (tool.Problem() is { } problem)
        {
            throw new UsageException(problem);
        }

        context.Detail($"idrak-tune {string.Join(' ', args)}");
        int code = tool.Execute();
        if (captured is not null)
        {
            context.WriteJson(new JsonObject
            {
                ["command"] = command,
                ["model"] = choice.Model,
                ["data"] = new JsonArray([.. data.Select(d => (JsonNode)d)]),
                ["out"] = tool.Output,
                ["device"] = tool.Device.ToString(),
                ["exitCode"] = code,
                ["output"] = new JsonArray([.. captured.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => (JsonNode)l)]),
            });
        }

        return code;
    }

    // The tune.json given with --config (only when given: the default config file holds no tune settings).
    private JsonObject? ReadSettings(CommandContext context)
    {
        if (context.Option("--config") is not { } path)
        {
            return null;
        }

        var json = JsonNode.Parse(File.ReadAllText(path), documentOptions: CliConfig.ReadOptions) as JsonObject
            ?? throw new UsageException($"{path} is not a JSON object.");
        var known = ValueOptions.Append("--seed").Concat(Flags).Select(o => o[2..]).ToHashSet(StringComparer.Ordinal);
        foreach (var (key, _) in json)
        {
            if (!known.Contains(key) && !OwnKeys.Contains(key))
            {
                throw new UsageException($"{path}: unknown key '{key}'. The keys are idrak tune's options without the dashes (idrak help tune), and command, model and data.");
            }
        }

        return json;
    }

    private static string? Text(JsonObject? settings, string key) => settings?[key] switch
    {
        null => null,
        JsonValue v when v.TryGetValue(out string? s) => s,
        JsonValue v when v.GetValueKind() == JsonValueKind.Number => v.ToJsonString(),
        JsonArray a => string.Join(',', a.Select(n => n?.ToString())),
        var other => throw new UsageException($"tune.json: \"{key}\" should be a text or a number, not {other.ToJsonString()}."),
    };

    private static List<string> List(JsonObject? settings, string key) => settings?[key] switch
    {
        null => [],
        JsonArray a => [.. a.Select(n => n?.ToString()).OfType<string>()],
        var other => [Text(settings, key)!],
    };
}

/// <summary><c>idrak tune init</c>: writes a commented tune.json for a model and data, with the library's defaults.</summary>
internal sealed class TuneInitCommand : Command
{
    public override string Name => "tune init";

    public override string Summary => "Write a commented tune.json (model, data, adapter and training settings) for idrak tune --config";

    public override string Usage => """
        [-b MODEL] [--data FILE]... [-o tune.json] [--loss sft|dpo|orpo|simpo] [-w int8|int4|bf16] [-f]

          -b, --base MODEL    the model to tune (a Hugging Face id, a folder, a .gguf file or an alias)
          --data FILE         the data (repeatable): files, folders, hf:... specs or a recipe .json
          -o, --out FILE      where to write the settings (default tune.json)
          --adapter-out DIR   where idrak tune writes the adapter (default adapters/<model name>)
          --loss NAME         sft (default), or dpo, orpo, simpo for preference rows
          -w, --weights F     base weights: int8, int4 (QLoRA) or bf16
          -f, --force         overwrite an existing file

        The values are the library's defaults (rank, alpha, learning rate, lengths); the device given with -d is written
        too. Sizing them to the data and the device's memory is what idrak suggest does, when it is available.

        Examples:
          idrak tune init -b Qwen/Qwen3-0.6B --data chats.jsonl
          idrak tune init -b qwen --data prefs.jsonl --loss dpo -w int4 -o dpo.json && idrak tune --config dpo.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--base", "--data", "--out", "--adapter-out", "--loss", "--weights"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-b"] = "--base", ["-o"] = "--out", ["-w"] = "--weights", ["-f"] = "--force",
    };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"tune init takes no arguments ('{context.Positional[0]}'); give the model with -b and the data with --data.");
        }

        string path = context.Option("--out") ?? "tune.json";
        if (File.Exists(path) && !context.Flag("--force"))
        {
            context.Error($"{path} exists; give -f, --force to overwrite it, or -o FILE for another name.");
            return ExitCodes.Failed;
        }

        string loss = (context.Option("--loss") ?? "sft").ToLowerInvariant();
        if (loss is not ("sft" or "dpo" or "orpo" or "simpo"))
        {
            throw new UsageException($"--loss {loss}: use sft, dpo, orpo or simpo.");
        }

        string? weights = context.Option("--weights")?.ToLowerInvariant();
        if (weights is not (null or "int8" or "int4" or "bf16"))
        {
            throw new UsageException($"--weights {weights}: use int8, int4 or bf16.");
        }

        string model = context.Option("--base") ?? "owner/model";
        var data = context.Options("--data").Count > 0 ? context.Options("--data") : [loss == "sft" ? "chats.jsonl" : "preferences.jsonl"];
        string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(model.Replace(':', '-'))).ToLowerInvariant();
        string adapters = context.Option("--adapter-out") ?? $"adapters/{(name.Length > 0 ? name : "model")}";
        string? device = context.Option("--device") ?? context.Config.Get("device");
        var defaults = new Idrak.LanguageModels.FineTuningOptions();
        string Json(string text) => JsonValue.Create(text).ToJsonString(CommandContext.JsonOutput);
        string Number(double value) => value.ToString("R", CultureInfo.InvariantCulture);
        string text = $$"""
            // idrak tune settings, written by idrak tune init. Run them with: idrak tune --config {{path}}
            // Options on the command line win over these. The keys are idrak tune's options without the dashes
            // (idrak help tune lists every one), plus "command", "model" and "data".
            {
              // What to do: train, evaluate, chat, export, download or info.
              "command": "train",
              // The base model: a Hugging Face id, a model folder, a .gguf file or an alias.
              "model": {{Json(model)}},
              // The data: files, folders, hf:, github:, kaggle:, zenodo: specs or URLs, or a recipe .json with "sources".
              "data": [{{string.Join(", ", data.Select(Json))}}],
              // Where the adapter (PEFT format) and idrak-tuning.json are written.
              "out": {{Json(adapters)}},
              // The device: cpu, cuda:0, vulkan:0, hip:0 (left out: the best GPU found, else the CPU).
              {{(device is null ? "// \"device\": \"cuda:0\"," : $"\"device\": {Json(device)},")}}
              // Base weights: int8, int4 (QLoRA) or bf16 (left out: float32).
              {{(weights is null ? "// \"weights\": \"int4\"," : $"\"weights\": {Json(weights)},")}}
              // The loss: sft (conversations and texts), or dpo, orpo, simpo (rows with prompt, chosen and rejected).
              "loss": {{Json(loss)}},
              // The adapters: lora or dora, their rank and alpha (the update is scaled by alpha / rank), and the layers they go on.
              "adapter-type": "lora",
              "rank": {{defaults.Rank}},
              "alpha": {{Number(defaults.Alpha)}},
              "targets": {{Json(string.Join(',', defaults.Targets))}},
              // Optimization: learning rate, epochs, optimizer (adamw, adam, adamw8bit, sgd), schedule (cosine, linear,
              // constant, wsd) and the warm-up as a fraction of the steps.
              "lr": {{Number(defaults.LearningRate)}},
              "epochs": {{defaults.Epochs}},
              "optimizer": "adamw",
              "schedule": "cosine",
              "warmup": {{Number(defaults.WarmupFraction)}},
              // Sequences: the longest kept (tokens) and the tokens in a batch.
              "max-length": {{defaults.MaxLength}},
              "batch-tokens": {{defaults.BatchTokens}},
              // A fraction of the data held out to report the evaluation loss while training.
              "eval-fraction": 0.02,
              "seed": 0
            }

            """;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, text);
        context.Write($"Wrote {path} ({(loss == "sft" ? "supervised fine-tuning" : loss)} of {model} on {string.Join(", ", data)})");
        if (model == "owner/model")
        {
            context.Write("Set \"model\" in it (or give -b MODEL next time).");
        }

        context.Write($"Next      idrak tune --config {path}");
        context.WriteJson(new JsonObject
        {
            ["path"] = Path.GetFullPath(path),
            ["model"] = model,
            ["data"] = new JsonArray([.. data.Select(d => (JsonNode)d)]),
            ["out"] = adapters,
            ["loss"] = loss,
        });
        return ExitCodes.Ok;
    }
}
