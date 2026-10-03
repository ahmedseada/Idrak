// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands.Run;

/// <summary>A model's tokenizer and chat template, read without its weights (tokenize, template).</summary>
internal sealed record ModelText(string Model, string Folder, BpeTokenizer Tokenizer, JinjaChatTemplate? Template)
{
    /// <summary>Reads the tokenizer and template of the model named <paramref name="name"/> (alias, id, folder or .gguf file).</summary>
    public static ModelText Load(CommandContext context, string name)
    {
        var choice = Models.Choose(context, name);
        string path = Models.Resolve(context, choice.Model);
        string folder = CheckpointFormats.For(path).Prepare(path);   // a .gguf file: its tokenizer and template from the metadata
        if (!File.Exists(Path.Combine(folder, "tokenizer.json")))
        {
            throw new InvalidOperationException($"{choice.Model} has no tokenizer.json.");
        }

        var tokenizer = BpeTokenizer.Load(folder);
        return new ModelText(choice.Model, folder, tokenizer, JinjaChatTemplate.Load(folder, tokenizer));
    }
}

/// <summary><c>idrak tokenize MODEL "text"</c>: the tokens and ids of a text.</summary>
internal sealed class TokenizeCommand : Command
{
    public override string Name => "tokenize";

    public override string Summary => "Tokens and ids of a text (--chat renders the chat template first, --count only counts)";

    public override string Usage => """
        MODEL ["text"] [options]

        The text is the arguments after MODEL, then --input files and piped standard input. Only the tokenizer is read,
        not the weights.

        Options:
          -i, --input FILE   read (more of) the text from FILE (repeatable)
              --chat         render the text as a user message in the model's chat template (with the assistant's
                             turn opened) and tokenize that
          -s, --system TEXT  with --chat: a system message first
              --count        print only the number of tokens

        Examples:
          idrak tokenize Qwen/Qwen3-0.6B "Hello, world"
          idrak tokenize qwen --chat -s "Be brief." "Hi"
          cat essay.txt | idrak tokenize ./model.gguf --count
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--input", "--system"];

    public override IReadOnlyCollection<string> Flags => ["--chat", "--count"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-i"] = "--input", ["-s"] = "--system" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string text = PromptInput.Read(context, 1, "the text");
        var model = ModelText.Load(context, name);
        if (context.Flag("--chat"))
        {
            var template = model.Template ?? throw new InvalidOperationException($"{name} has no chat template; tokenize without --chat.");
            var messages = new List<ChatMessage>();
            if (context.Option("--system") is { } system)
            {
                messages.Add(new ChatMessage("system", system));
            }

            messages.Add(new ChatMessage("user", text));
            text = template.Render(messages, [], null, addGenerationPrompt: true);
        }

        var ids = model.Tokenizer.Encode(text);
        if (context.Flag("--count"))
        {
            if (!context.Json)
            {
                context.Output.WriteLine(ids.Count);
            }

            context.WriteJson(new JsonObject { ["model"] = model.Model, ["count"] = ids.Count });
            return ExitCodes.Ok;
        }

        context.Table(["#", "ID", "Token"], ids.Select((id, i) => (IReadOnlyList<string>)[i.ToString(), id.ToString(), Show(model.Tokenizer.TokenOf(id))]));
        context.Write($"{ids.Count} tokens");
        context.WriteJson(new JsonObject
        {
            ["model"] = model.Model,
            ["text"] = text,
            ["count"] = ids.Count,
            ["ids"] = new JsonArray([.. ids.Select(i => (JsonNode)i)]),
            ["tokens"] = new JsonArray([.. ids.Select(i => (JsonNode)model.Tokenizer.TokenOf(i))]),
        });
        return ExitCodes.Ok;
    }

    // Control characters made visible.
    private static string Show(string token) => token.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
}

/// <summary><c>idrak template MODEL</c>: the chat template, the tool-call format and a rendered sample conversation.</summary>
internal sealed class TemplateCommand : Command
{
    public override string Name => "template";

    public override string Summary => "The chat template, its tool-call format and a rendered sample conversation with a tool call";

    public override string Usage => """
        MODEL [options]

        Prints the template's source, its stop sequences, reasoning tags and detected tool-call format, then a sample
        conversation (system, user, a tool call, its result, the answer) rendered by the template. Only the tokenizer
        files are read, not the weights.

        Options:
              --source       print only the template's source
              --think        render the sample with reasoning on (default: the template's own default)

        Examples:
          idrak template Qwen/Qwen3-0.6B
          idrak template ./model.gguf --source > template.jinja
          idrak template qwen --json
        """;

    public override IReadOnlyCollection<string> Flags => ["--source", "--think"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        var model = ModelText.Load(context, name);
        var template = model.Template ?? throw new InvalidOperationException($"{name} has no chat template (tokenizer_config.json, chat_template.jinja).");
        if (context.Flag("--source"))
        {
            if (!context.Json)
            {
                context.Output.WriteLine(template.Source);
            }

            context.WriteJson(new JsonObject { ["model"] = model.Model, ["source"] = template.Source });
            return ExitCodes.Ok;
        }

        var tool = new ToolDefinition("get_weather", "The current weather in a city.", new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject { ["city"] = new JsonObject { ["type"] = "string", ["description"] = "The city." } },
            ["required"] = new JsonArray("city"),
        });
        var sample = new List<ChatMessage>
        {
            new("system", "You are a helpful assistant."),
            new("user", "What is the weather in Cairo?"),
            new("assistant", "", ToolCalls: [new ToolCall("get_weather", new JsonObject { ["city"] = "Cairo" })]),
            new("tool", "{\"temperature\": 31, \"sky\": \"clear\"}", ToolName: "get_weather"),
            new("assistant", "It is 31 degrees and clear in Cairo."),
        };
        string rendered;
        try
        {
            rendered = template.Render(sample, [tool], context.Flag("--think") ? true : null, addGenerationPrompt: false);
        }
        catch (InvalidOperationException e)
        {
            rendered = $"(the template cannot render the sample: {e.Message})";
        }

        string format = template.ToolCallFormatName;
        context.Write($"Template   {template.Source.Length} characters");
        context.Write($"Stops      {string.Join(", ", template.StopSequences.Select(s => $"\"{s}\""))}");
        context.Write($"Reasoning  {template.ThinkTags.Open} ... {template.ThinkTags.Close}");
        context.Write($"Tool calls {format}");
        context.Write("");
        context.Write("Source:");
        context.Write(template.Source);
        context.Write("");
        context.Write("Sample conversation:");
        context.Write(rendered);
        context.WriteJson(new JsonObject
        {
            ["model"] = model.Model,
            ["source"] = template.Source,
            ["stop_sequences"] = new JsonArray([.. template.StopSequences.Select(s => (JsonNode)s)]),
            ["think_tags"] = new JsonArray(template.ThinkTags.Open, template.ThinkTags.Close),
            ["tool_call_format"] = format,
            ["sample"] = rendered,
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak complete MODEL "text"</c>: raw completion of a text, without the chat template.</summary>
internal sealed class CompleteCommand : Command
{
    public override string Name => "complete";

    public override string Summary => "Raw completion of a text, without the chat template";

    public override string Usage => """
        MODEL ["text"] [options]

        Continues the text as written (no chat template, no system prompt); the text is the arguments after MODEL, then
        --input files and piped standard input. The continuation streams to the output; --json prints it with the token
        counts and speed.

        Options:
          -i, --input FILE       read (more of) the text from FILE (repeatable)
              --stop TEXT        end at this text (repeatable)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
              --temperature T, --top-k N, --top-p P, --max-tokens N, --seed N   as for run

        Examples:
          idrak complete Qwen/Qwen3-0.6B "def fibonacci(n):"
          idrak complete ./model.gguf "Once upon a time" --max-tokens 100 --seed 1
        """;

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. Models.ValueOptions, "--temperature", "--top-k", "--top-p", "--max-tokens", "--input", "--stop"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(Models.ShortForms.Append(KeyValuePair.Create("-i", "--input")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string text = PromptInput.Read(context, 1, "the text");
        var settings = GenerationSettings.From(context);
        var choice = Models.Choose(context, name);
        using var model = Models.Load(context, choice);
        var generator = Models.CreateGenerator(model, choice);
        int contextLength = Models.ContextLength(choice, model);
        var options = settings.ToOptions(contextLength) with { Stop = context.Options("--stop") };
        var output = new System.Text.StringBuilder();
        GenerationStats? stats = null;
        string? reason = null;
        foreach (var chunk in generator.Stream(text, options))
        {
            output.Append(chunk.Text);
            if (!context.Json)
            {
                context.Output.Write(chunk.Text);
                context.Output.Flush();
            }

            if (chunk.Done)
            {
                (stats, reason) = (chunk.Stats, chunk.DoneReason);
            }
        }

        if (!context.Json)
        {
            context.Output.WriteLine();
        }

        if (context.Verbose && !context.Json && stats is not null)
        {
            context.ErrorOutput.WriteLine($"[{stats.GeneratedTokens} tokens, {stats.TokensPerSecond:F1} tokens/s, prompt {stats.PromptTokens} tokens in {stats.PromptDuration.TotalSeconds:F2} s]");
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = context.Device.ToString(),
            ["prompt"] = text,
            ["text"] = output.ToString(),
            ["done_reason"] = reason,
            ["prompt_tokens"] = stats?.PromptTokens ?? 0,
            ["generated_tokens"] = stats?.GeneratedTokens ?? 0,
            ["tokens_per_second"] = Math.Round(stats?.TokensPerSecond ?? 0, 2),
        });
        return ExitCodes.Ok;
    }
}
