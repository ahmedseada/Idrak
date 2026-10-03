// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Commands;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;
using Idrak.Generation;

// The idrak run commands (chat, run, batch, compare, complete, embed, tokenize, template, agent, tools), run in-process
// through CommandLine.Run with captured output against the tiny model in tests/Idrak.Tests/data (no network).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliRunGroup =
    [
        ("cli run: help of every run command (examples, environment), distinct short forms, no provider names", CliRunHelp),
        ("cli run: generation settings, /set values, tool sample arguments and schema checks", CliRunSettings),
        ("cli run: run answers a prompt (argument, --input, piped), --json with text, tokens and speed; usage errors", CliRunRun),
        ("cli run: chat drives a conversation with slash commands, --file, --history saved and resumed, --json", CliRunChat),
        ("cli run: batch answers JSON Lines and resumes after an interruption", CliRunBatch),
        ("cli run: complete, tokenize (--chat, --count), template, embed (JSON and .npy), compare", CliRunText),
        ("cli run: tools list/test on an assembly, run with --tools, agent in a read-only workspace, run --schema", CliRunToolsAndAgent),
    ];

    private static string CliModel => TestData("gguf/tiny-llama-hf");

    // Runs idrak with args (and input for chat or piped text); a config file that does not exist keeps the user's out.
    private static (int Code, string Output, string Error) RunIdrakOn(Device device, string? input, params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        string[] all = [.. args, "-d", device.ToString(), "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-test-no-config.json")];
        int code = StandardInput.With(new StringReader(input ?? ""), () => CommandLine.Run(all, output, error));
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject JsonOf(string text, string what)
    {
        try
        {
            return JsonNode.Parse(text) as JsonObject ?? throw new InvalidOperationException("not an object");
        }
        catch (Exception e)
        {
            throw new InvalidOperationException($"{what}: output is not one JSON object ({e.Message}): {text}");
        }
    }

    private static void CliRunHelp(Device device)
    {
        string[] providers = ["ollama", "openai", "anthropic", "llama.cpp", "lm studio"];
        foreach (var command in RunCommands.All)
        {
            var (code, text, _) = RunIdrakOn(device, null, [.. command.Name.Split(' '), "--help"]);
            Check(code == 0 && text.Contains("Examples:") && text.Contains("Environment (idrak help env for all):") && text.Contains($"idrak {command.Name}"), $"{command.Name}: help");
            Check(!providers.Any(p => System.Text.RegularExpressions.Regex.Replace(text, @"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", "").Contains(p, StringComparison.OrdinalIgnoreCase)), $"{command.Name}: help names no provider");
            var shorts = command.ShortForms.Keys.ToList();
            Check(shorts.Distinct().Count() == shorts.Count && !shorts.Any(CommandContext.CommonShortForms.ContainsKey), $"{command.Name}: short forms");
            Check(command.ShortForms.Values.All(l => command.ValueOptions.Contains(l) || command.Flags.Contains(l)), $"{command.Name}: short forms name its options");
        }

        Check(CommandLine.Find(CommandTable.All, ["c"], out _)?.Name == "chat" && CommandLine.Find(CommandTable.All, ["r"], out _)?.Name == "run", "aliases c and r");
        Check(CommandLine.Find(CommandTable.All, ["tools", "test"], out int used)?.Name == "tools test" && used == 2, "tools test is two words");
    }

    private static void CliRunSettings(Device device)
    {
        _ = device;
        var settings = new GenerationSettings();
        Check(settings.Set("temperature", "0") is null && settings.ToOptions(64) is { TopK: 1, Temperature: 1f, NumCtx: 64 }, "temperature 0 is greedy");
        Check(settings.Set("top-p", "1.5") is { } error && error.Contains("--top-p"), "top-p out of range");
        Check(settings.Set("seed", "7") is null && settings.Seed == 7 && settings.Set("seed", "random") is null && settings.Seed is null, "seed");
        Check(settings.Set("think", "off") is null && settings.Think == false && settings.Set("colour", "1")!.Contains("Unknown setting"), "think and unknown settings");
        Check(settings.Set("max-tokens", "12") is null && settings.ToOptions(64).NumPredict == 12, "max-tokens");

        var schema = (JsonObject)JsonNode.Parse("""
            {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 3},
             "unit": {"enum": ["c", "f"]}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}
            """)!;
        var sample = ToolAssemblies.SampleArguments(schema);
        Check((string?)sample["city"] == "test" && (int?)sample["days"] == 3 && (string?)sample["unit"] == "c" && sample["tags"] is JsonArray, "sample arguments");
        Check(StructuredOutput.Check("```json\n{\"city\": \"Cairo\", \"days\": 2}\n```", schema).Problem is null, "a fenced JSON answer matches");
        Check(StructuredOutput.Check("Sure: {\"days\": 2}", schema).Problem!.Contains("'city'"), "a missing required property");
        Check(StructuredOutput.Check("{\"city\": \"x\", \"days\": 2.5}", schema).Problem!.Contains("$.days"), "a wrong type");
        Check(StructuredOutput.Check("no json here", schema).Problem!.Contains("not JSON"), "not JSON");
        Check(CompareCommand.Wrap("one two three four", 9).SequenceEqual(["one two", "three", "four"]), "wrapping");
    }

    private static void CliRunRun(Device device)
    {
        var (code, text, error) = RunIdrakOn(device, null, "run", CliModel, "Hello", "--max-tokens", "6", "--seed", "1");
        Check(code == 0 && text.Length > 0 && error.Length == 0, $"run: exit {code}, error {error}");

        (code, text, _) = RunIdrakOn(device, null, "r", CliModel, "Hello", "--max-tokens", "5", "--temperature", "0", "-j");
        var json = JsonOf(text, "run --json");
        Check(code == 0 && json["text"] is JsonValue && (int)json["generated_tokens"]! == 5 && (int)json["prompt_tokens"]! > 0
              && (double)json["tokens_per_second"]! > 0 && (string?)json["done_reason"] == "length", $"run --json: {text}");
        var (_, again, _) = RunIdrakOn(device, null, "run", CliModel, "Hello", "--max-tokens", "5", "--temperature", "0", "-j");
        Check((string?)JsonOf(again, "run again")["text"] == (string?)json["text"], "greedy answers repeat");

        // The prompt from piped input and from a file, after the argument.
        string file = Path.Combine(Path.GetTempPath(), $"idrak-cli-prompt-{Guid.NewGuid():N}.txt");
        File.WriteAllText(file, "a note");
        try
        {
            (code, text, _) = RunIdrakOn(device, "piped text", "run", CliModel, "Summarize", "-i", file, "--max-tokens", "3", "-j", "-s", "Be brief.");
            Check(code == 0 && (int)JsonOf(text, "run piped")["prompt_tokens"]! > (int)json["prompt_tokens"]!, "piped and file text join the prompt");
        }
        finally
        {
            File.Delete(file);
        }

        (code, _, error) = RunIdrakOn(device, null, "run", CliModel);
        Check(code == 2 && error.Contains("Missing a prompt"), $"run without a prompt: {code} {error}");
        (code, _, error) = RunIdrakOn(device, null, "run", "./no-such-model", "hi");
        Check(code == 2 && error.Contains("Model not found"), $"run with a missing model: {code} {error}");
        (code, _, error) = RunIdrakOn(device, null, "run", CliModel, "hi", "--top-k", "many");
        Check(code == 2 && error.Contains("--top-k"), "a bad --top-k");
        (code, _, error) = RunIdrakOn(device, null, "run", CliModel, "hi", "-k", "nosuch");
        Check(code == 2 && error.Contains("Unknown --kv format"), $"a bad --kv: {error}");
        (code, _, _) = RunIdrakOn(device, null, "run", CliModel, "hi", "-k", "int8", "--max-tokens", "2", "-q");
        Check(code == 0, "int8 KV cache");
    }

    private static void CliRunChat(Device device)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-chat-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string history = Path.Combine(folder, "history.json"), saved = Path.Combine(folder, "saved.json"), notes = Path.Combine(folder, "notes.txt");
            File.WriteAllText(notes, "remember the milk");
            string input = string.Join('\n', "hello", "/set temperature 0", "/set max-tokens 4", "/set top-p 7", "/system Be brief.", "/system",
                "how are \\", "you", "/stats", "/retry", "/copy", "/think off", "/set", "/tools", "/file " + notes, "and this?", $"/save {saved}", "/bogus", "/help",
                "/exit", "never sent");
            var (code, text, error) = RunIdrakOn(device, input, "chat", CliModel, "--max-tokens", "6", "--seed", "3", "--history", history, "--file", notes);
            Check(code == 0 && error.Length == 0, $"chat: exit {code}, {error}");
            Check(text.Contains("temperature = 0") && text.Contains("max-tokens = 4") && text.Contains("--top-p needs"), "chat: /set");
            Check(text.Contains("System prompt set.") && text.Contains("System prompt: Be brief."), "chat: /system");
            Check(text.Contains("Context ") && text.Contains("tokens/s on average") && text.Contains("Memory "), "chat: /stats");
            Check(text.Contains("Reasoning: off.") && text.Contains("No tools") && text.Contains("Unknown command /bogus") && text.Contains("/retry"), "chat: /think, /tools, unknown, /help");
            Check(text.Split('\n').Count(l => l.StartsWith("[4 tokens", StringComparison.Ordinal)) == 3, "chat: three answers after /set max-tokens 4 (two turns and a retry)");

            var messages = ChatHistory.Load(history);
            Check(messages.Count == 7 && messages[0].Role == "system" && messages[1].Content.Contains("remember the milk") && messages[1].Content.EndsWith("hello")
                  && messages[3].Content == "how are \nyou" && messages[6].Role == "assistant", $"chat: history has {messages.Count} messages");
            Check(messages[5].Content.Contains("remember the milk") && messages[5].Content.EndsWith("and this?"), "chat: /file joins the next message");
            Check(ChatHistory.Load(saved).Count == 7, "chat: /save");

            // Resumed from the history: the conversation continues, and --json gives the whole of it at the end.
            (code, text, _) = RunIdrakOn(device, "one more\n", "c", CliModel, "--max-tokens", "2", "--history", history, "-j");
            var json = JsonOf(text, "chat --json");
            Check(code == 0 && json["messages"] is JsonArray { Count: 9 } && json["answers"] is JsonArray { Count: 1 },
                $"chat: resumed with --json: {text}");
            Check(ChatHistory.Load(history).Count == 9 && ChatHistory.Load(history)[0].Content == "Be brief.", "chat: history updated, system prompt kept");

            (code, _, error) = RunIdrakOn(device, "", "chat", CliModel, "extra words");
            Check(code == 2 && error.Contains("idrak run"), "chat with a prompt points to run");
            (code, _, error) = RunIdrakOn(device, "", "chat", CliModel, "--file", Path.Combine(folder, "missing.txt"));
            Check(code == 2 && error.Contains("File not found"), "chat --file missing");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliRunBatch(Device device)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-batch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string prompts = Path.Combine(folder, "prompts.jsonl"), answers = Path.Combine(folder, "answers.jsonl");
            File.WriteAllLines(prompts, ["\"first\"", "{\"prompt\": \"second\", \"id\": \"b\"}", "", "plain third",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"fourth\"}]}"]);
            var (code, text, error) = RunIdrakOn(device, null, "batch", CliModel, "-i", prompts, "-o", answers, "--max-tokens", "3", "--batch-size", "3", "--temperature", "0", "-j");
            var json = JsonOf(text, "batch --json");
            Check(code == 0 && (int)json["prompts"]! == 4 && (int)json["answered"]! == 4 && (int)json["skipped"]! == 0, $"batch: {text} {error}");
            var lines = File.ReadAllLines(answers).Select(l => JsonNode.Parse(l)!).ToList();
            Check(lines.Count == 4 && lines.Select(l => (int)l["index"]!).Order().SequenceEqual([0, 1, 2, 3]) && (string?)lines.First(l => (int)l["index"]! == 1)["id"] == "b"
                  && lines.All(l => (int)l["generated_tokens"]! == 3), "batch: one line per prompt");

            // An interruption: two lines kept and a torn one; the rerun answers only the rest.
            File.WriteAllText(answers, File.ReadAllLines(answers)[0] + "\n" + File.ReadAllLines(answers)[1] + "\n{\"index\": 2, \"ans");
            (code, text, _) = RunIdrakOn(device, null, "batch", CliModel, "--input", prompts, "--out", answers, "--max-tokens", "3", "-j");
            json = JsonOf(text, "batch resumed");
            Check(code == 0 && (int)json["skipped"]! == 2 && (int)json["answered"]! == 2 && File.ReadAllLines(answers).Length == 4, $"batch resumed: {text}");
            (code, text, _) = RunIdrakOn(device, null, "batch", CliModel, "-i", prompts, "-o", answers);
            Check(code == 0 && text.Contains("0 to go"), "batch: nothing left");
            (code, _, error) = RunIdrakOn(device, null, "batch", CliModel, "-i", prompts);
            Check(code == 2 && error.Contains("--out"), "batch without --out");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliRunText(Device device)
    {
        var (code, text, _) = RunIdrakOn(device, null, "complete", CliModel, "Once upon", "--max-tokens", "4", "--seed", "2", "-j");
        var json = JsonOf(text, "complete --json");
        Check(code == 0 && (int)json["generated_tokens"]! == 4 && (string?)json["prompt"] == "Once upon", $"complete: {text}");

        var tokenizer = Idrak.LanguageModels.BpeTokenizer.Load(CliModel);
        (code, text, _) = RunIdrakOn(device, null, "tokenize", CliModel, "hello world", "--count");
        Check(code == 0 && text.Trim() == tokenizer.Encode("hello world").Count.ToString(), $"tokenize --count: {text}");
        (code, text, _) = RunIdrakOn(device, null, "tokenize", CliModel, "hello world", "-j");
        json = JsonOf(text, "tokenize --json");
        Check(json["ids"]!.AsArray().Select(i => (int)i!).SequenceEqual(tokenizer.Encode("hello world")), "tokenize: ids");
        (code, text, _) = RunIdrakOn(device, null, "tokenize", CliModel, "hi", "--chat", "-s", "sys");
        Check(code == 0 && text.Contains("<|im_start|>") && text.Contains(" tokens"), "tokenize --chat renders the template");

        (code, text, _) = RunIdrakOn(device, null, "template", CliModel, "-j");
        json = JsonOf(text, "template --json");
        Check(code == 0 && ((string)json["source"]!).Contains("im_start") && ((string)json["sample"]!).Contains("Cairo") && json["tool_call_format"] is JsonValue, "template");
        (code, text, _) = RunIdrakOn(device, null, "template", CliModel, "--source");
        Check(code == 0 && text.StartsWith("{%-", StringComparison.Ordinal), "template --source");

        string npy = Path.Combine(Path.GetTempPath(), $"idrak-cli-{Guid.NewGuid():N}.npy");
        try
        {
            (code, text, _) = RunIdrakOn(device, null, "embed", CliModel, "first text", "second", "--max-length", "16");
            json = JsonOf(text, "embed");
            var vectors = json["embeddings"]!.AsArray().Select(e => e!["vector"]!.AsArray().Select(v => (double)v!).ToArray()).ToList();
            Check(code == 0 && (int)json["dimensions"]! == 64 && vectors.Count == 2 && vectors.All(v => Math.Abs(Math.Sqrt(v.Sum(x => x * x)) - 1) < 1e-3), "embed: unit vectors");
            (code, _, _) = RunIdrakOn(device, null, "embed", CliModel, "a", "b", "c", "-o", npy, "-q");
            var bytes = File.ReadAllBytes(npy);
            Check(code == 0 && bytes[0] == 0x93 && System.Text.Encoding.ASCII.GetString(bytes, 10, 70).Contains("'shape': (3, 64)") && bytes.Length == 128 + 3 * 64 * 4, "embed: .npy");
        }
        finally
        {
            File.Delete(npy);
        }

        (code, text, _) = RunIdrakOn(device, null, "compare", CliModel, CliModel, "Hello", "--max-tokens", "3", "--kv-b", "int8", "-j");
        json = JsonOf(text, "compare --json");
        Check(code == 0 && json["results"] is JsonArray { Count: 2 } results && ((string)results[1]!["label"]!).Contains("-k int8"), $"compare: {text}");
        (code, text, _) = RunIdrakOn(device, null, "compare", CliModel, CliModel, "Hello", "--max-tokens", "3", "--width", "60");
        Check(code == 0 && text.Contains(" | ") && text.Contains("tokens/s"), "compare side by side");
    }

    private static void CliRunToolsAndAgent(Device device)
    {
        string assembly = typeof(CliTestTools).Assembly.Location;
        var (code, text, _) = RunIdrakOn(device, null, "tools", "list", assembly, "-j");
        var names = JsonOf(text, "tools list")["tools"]!.AsArray().Select(t => (string)t!["function"]!["name"]!).ToList();
        Check(code == 0 && names.Contains("cli_add") && names.Contains("cli_echo") && names.Contains("cli_fail"), $"tools list: {string.Join(", ", names)}");
        (code, text, _) = RunIdrakOn(device, null, "tools", "test", assembly, "--tool", "cli_add", "--tool", "cli_echo");
        Check(code == 0 && text.Contains("ok   cli_add") && text.Contains("2 of 2 tools answered"), $"tools test: {text}");
        (code, text, _) = RunIdrakOn(device, null, "tools", "test", assembly, "--tool", "cli_add", "--args", "{\"a\": 2, \"b\": 40}", "-j");
        Check(code == 0 && (string?)JsonOf(text, "tools test --args")["results"]![0]!["result"] == "42", "tools test --args");
        (code, _, _) = RunIdrakOn(device, null, "tools", "test", assembly);
        Check(code == 1, "tools test: a failing tool exits with 1");
        (code, _, _) = RunIdrakOn(device, null, "tools", "test", assembly, "--tool", "nope");
        Check(code == 2, "tools test: an unknown tool");

        (code, text, _) = RunIdrakOn(device, "/tools\n/exit\n", "chat", CliModel, "--tools", assembly);
        Check(code == 0 && text.Contains("cli_add: Adds two numbers."), "chat --tools lists them");
        (code, text, _) = RunIdrakOn(device, null, "run", CliModel, "add 1 and 2", "--tools", assembly, "--max-tokens", "3", "-j");
        Check(code == 0 && JsonOf(text, "run --tools")["generated_tokens"] is not null, "run --tools");

        string workspace = Path.Combine(Path.GetTempPath(), $"idrak-cli-agent-{Guid.NewGuid():N}");
        Directory.CreateDirectory(workspace);
        try
        {
            File.WriteAllText(Path.Combine(workspace, "a.txt"), "x");
            (code, text, _) = RunIdrakOn(device, null, "agent", CliModel, "list the files", "--workspace", workspace, "--read-only", "--max-tokens", "4", "-j");
            var json = JsonOf(text, "agent --json");
            Check(code == 0 && (int)json["tasks"]! == 1 && ((string)json["messages"]![0]!["content"]!).Length > 100, $"agent: {text}");
            (code, text, _) = RunIdrakOn(device, "first task\n/exit\n", "agent", CliModel, "--workspace", workspace, "--max-tokens", "3");
            Check(code == 0 && text.Contains("tool calls"), "agent reads tasks from the input");
            (code, _, _) = RunIdrakOn(device, null, "agent", CliModel, "x", "--workspace", Path.Combine(workspace, "missing"));
            Check(code == 2, "agent: a missing workspace");
        }
        finally
        {
            Directory.Delete(workspace, recursive: true);
        }

        string schema = Path.Combine(Path.GetTempPath(), $"idrak-cli-schema-{Guid.NewGuid():N}.json");
        File.WriteAllText(schema, "{\"type\": \"object\", \"required\": [\"name\"]}");
        try
        {
            // The tiny model writes no JSON: the check reports it and the exit code says so.
            var (schemaCode, schemaText, schemaError) = RunIdrakOn(device, null, "run", CliModel, "name?", "--schema", schema, "--max-tokens", "3", "-j");
            Check(schemaCode == 1 && JsonOf(schemaText, "run --schema")["schema_valid"]?.GetValue<bool>() == false && schemaError.Contains("does not match"), "run --schema");
        }
        finally
        {
            File.Delete(schema);
        }
    }

}

/// <summary>Tools for the CLI tests (idrak tools list/test, --tools): found by the [Tool] attribute.</summary>
public sealed class CliTestTools
{
    [Tool("cli_add", "Adds two numbers.")]
    public static int Add(int a, int b) => a + b;

    [Tool("cli_echo", "Returns the text.")]
    public string Echo(string text) => text;

    [Tool("cli_fail", "Always fails.")]
    public static string Fail() => throw new InvalidOperationException("broken on purpose");
}
