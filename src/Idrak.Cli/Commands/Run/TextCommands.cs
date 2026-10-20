// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;

namespace Idrak.Cli.Commands.Run;

/// <summary>A model's tokenizer and chat template, read without its weights (tokenize, template).</summary>
internal sealed record ModelText(string Model, string Folder, IPretrainedTokenizer Tokenizer, JinjaChatTemplate? Template)
{
    /// <summary>Reads the tokenizer and template of the model named <paramref name="name"/> (alias, id, folder or .gguf file).</summary>
    public static ModelText Load(CommandContext context, string name)
    {
        var choice = ModelChoices.Choose(context, name);
        string path = ModelChoices.Resolve(context, choice.Model);
        string folder = CheckpointFormats.For(path).Prepare(path);   // a .gguf file: its tokenizer and template from the metadata
        var tokenizer = PretrainedTokenizers.TryLoad(folder)
            ?? throw new InvalidOperationException($"{choice.Model} has no tokenizer (readers: {string.Join(", ", TokenizerFiles.Names)}).");
        return new ModelText(choice.Model, folder, tokenizer, JinjaChatTemplate.Load(folder, tokenizer));
    }
}

/// <summary><c>idrak tokenize MODEL "text"</c>: the tokens and ids of a text.</summary>
internal sealed class TokenizeCommand : Command
{
    public override string Name => "tokenize";

    public override string Summary => "Tokens, ids, offsets and type ids of text (--chat renders the chat template first, --count only counts)";

    public override string Usage => """
        MODEL ["text"] [options]

        The text is the arguments after MODEL, then --input files and piped standard input. Only the tokenizer is read,
        not the weights. Without --offsets, --pair, --special, --max-length and --truncation, the tokens are the
        tokenizer's plain encoding (no special tokens, not cut).

        Options:
          -i, --input FILE      read (more of) the text from FILE (repeatable)
              --chat            render the text as a user message in the model's chat template (with the assistant's
                                turn opened) and tokenize that
          -s, --system TEXT     with --chat: a system message first
              --count           print only the number of tokens
              --offsets         show where each token comes from: its UTF-16 start and end in its text, and the text it
                                covers (a special token covers none)
              --pair TEXT       encode TEXT as the second text of a pair; shows each token's type id and sequence (0 for
                                the text, 1 for the pair, -1 for a special token)
              --special         add the tokenizer's special tokens where its post-processor puts them ([CLS] ... [SEP],
                                <s> ... </s>); by default none are added, as a chat template writes its own
              --max-length N    cut the encoding to N tokens, special tokens included (default: not cut); the
                                tokenizer.json's own truncation and padding are not applied
              --truncation S    with --max-length, which text loses tokens: longest_first (default), only_first or
                                only_second

        Examples:
          idrak tokenize Qwen/Qwen3-0.6B "Hello, world"
          idrak tokenize qwen --chat -s "Be brief." "Hi"
          cat essay.txt | idrak tokenize ./model.gguf --count
          idrak tokenize ./bert-base-uncased "Hello, World!" --pair "Second" --special
          idrak tokenize ./t5-small "Hello, world" --offsets --max-length 8
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--input", "--system", "--pair", "--max-length", "--truncation"];

    public override IReadOnlyCollection<string> Flags => ["--chat", "--count", "--offsets", "--special"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-i"] = "--input", ["-s"] = "--system" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        var detailed = DetailedOptions(context);                     // checked before the text is read and the model loaded
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

        if (detailed is { } options)
        {
            return RunDetailed(context, model, text, context.Option("--pair"), options);
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

        context.Table(["#", "ID", "Token"], ids.Select((id, i) => (IReadOnlyList<string>)[i.ToString(CultureInfo.InvariantCulture), id.ToString(CultureInfo.InvariantCulture), Show(model.Tokenizer.TokenOf(id))]));
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

    // The encode options of --offsets, --pair, --special, --max-length and --truncation, or null when none is given (then
    // the output is the plain encoding's, as before these options). Nothing is padded, and without --max-length nothing is
    // cut: the tokenizer.json's own truncation and padding are left out, so the tokens are the text's.
    private static TokenizerEncodeOptions? DetailedOptions(CommandContext context)
    {
        string? maxLength = context.Option("--max-length"), truncation = context.Option("--truncation");
        if (!context.Flag("--offsets") && !context.Flag("--special") && context.Option("--pair") is null && maxLength is null && truncation is null)
        {
            return null;
        }

        int? max = null;
        if (maxLength is not null)
        {
            int value = context.IntOption("--max-length", 0);
            max = value >= 1 ? value : throw new UsageException($"--max-length needs a whole number of 1 or more, not '{maxLength}'.");
        }

        if (truncation is not (null or "longest_first" or "only_first" or "only_second"))
        {
            throw new UsageException($"--truncation takes longest_first, only_first or only_second, not '{truncation}'.");
        }

        if (truncation is not null && max is null)
        {
            throw new UsageException("--truncation needs --max-length.");
        }

        return new TokenizerEncodeOptions
        {
            AddSpecialTokens = context.Flag("--special"), MaxLength = max ?? int.MaxValue, Truncation = truncation ?? "longest_first", Padding = "none",
        };
    }

    // The tokenizer's detailed encoding: ids and tokens, with type ids and sequences (--pair, --special) and offsets with
    // the text each token covers (--offsets).
    private static int RunDetailed(CommandContext context, ModelText model, string text, string? pair, TokenizerEncodeOptions options)
    {
        var encoding = model.Tokenizer.Encode(text, pair, options);
        if (context.Flag("--count"))
        {
            if (!context.Json)
            {
                context.Output.WriteLine(encoding.Count);
            }

            context.WriteJson(new JsonObject { ["model"] = model.Model, ["count"] = encoding.Count });
            return ExitCodes.Ok;
        }

        bool sequences = pair is not null || options.AddSpecialTokens, offsets = context.Flag("--offsets");
        string Covered(int i)
        {
            var (start, end) = encoding.Offsets[i];
            string? source = encoding.SequenceIds[i] switch { 0 => text, 1 => pair, _ => null };
            return source is not null && 0 <= start && start < end && end <= source.Length ? Show(source[start..end]) : "";
        }

        string Number(int value) => value.ToString(CultureInfo.InvariantCulture);
        List<string> headers = ["#", "ID", "Token"];
        if (sequences)
        {
            headers.AddRange(["Type", "Sequence"]);
        }

        if (offsets)
        {
            headers.AddRange(["Start", "End", "Text"]);
        }

        context.Table(headers, Enumerable.Range(0, encoding.Count).Select(i =>
        {
            List<string> row = [Number(i), Number(encoding.Ids[i]), Show(model.Tokenizer.TokenOf(encoding.Ids[i]))];
            if (sequences)
            {
                row.AddRange([Number(encoding.TypeIds[i]), Number(encoding.SequenceIds[i])]);
            }

            if (offsets)
            {
                row.AddRange([Number(encoding.Offsets[i].Start), Number(encoding.Offsets[i].End), Covered(i)]);
            }

            return (IReadOnlyList<string>)row;
        }));
        context.Write($"{encoding.Count} tokens");
        static JsonArray Numbers(int[] values) => new([.. values.Select(v => (JsonNode)v)]);
        var json = new JsonObject { ["model"] = model.Model, ["text"] = text };
        if (pair is not null)
        {
            json["pair"] = pair;
        }

        json["count"] = encoding.Count;
        json["ids"] = Numbers(encoding.Ids);
        json["tokens"] = new JsonArray([.. encoding.Ids.Select(i => (JsonNode)model.Tokenizer.TokenOf(i))]);
        json["type_ids"] = Numbers(encoding.TypeIds);
        json["sequence_ids"] = Numbers(encoding.SequenceIds);
        json["special_tokens_mask"] = Numbers(encoding.SpecialTokensMask);
        json["offsets"] = new JsonArray([.. encoding.Offsets.Select(o => (JsonNode)new JsonArray(o.Start, o.End))]);
        context.WriteJson(json);
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
              --source  print only the template's source
              --think   render the sample with reasoning on (default: the template's own default)

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
          -i, --input FILE      read (more of) the text from FILE (repeatable)
              --stop TEXT       end at this text (repeatable)
          -w, --weights FORMAT  int8, int4, bf16 or a registered packed format (default: as stored)
          --head-weights F      the output head's format beside -w (bf16: the checkpoint's own scores and one
                                vocabulary table for tied models, less memory than int8/int4 heads, slower)
          -k, --kv FORMAT       KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N       context window in tokens (default 4096, at most the model's)
              --adapter DIR     merge a LoRA adapter into the weights as they are read
              --temperature T   sampling, as for run; also --top-k N, --top-p P, --max-tokens N and --seed N

        Examples:
          idrak complete Qwen/Qwen3-0.6B "def fibonacci(n):"
          idrak complete ./model.gguf "Once upon a time" --max-tokens 100 --seed 1
        """;

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. ModelChoices.ValueOptions, "--temperature", "--top-k", "--top-p", "--max-tokens", "--input", "--stop"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(ModelChoices.ShortForms.Append(KeyValuePair.Create("-i", "--input")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string text = PromptInput.Read(context, 1, "the text");
        var settings = GenerationSettings.From(context);
        var choice = ModelChoices.Choose(context, name);
        using var model = ModelChoices.Load(context, choice);
        var generator = ModelChoices.CreateGenerator(model, choice);
        int contextLength = ModelChoices.ContextLength(choice, model);
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
