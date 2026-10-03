// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak run MODEL "prompt"</c> (<c>r</c>): one answer, streamed to the output, then exit.</summary>
internal sealed class RunCommand : Command
{
    public override string Name => "run";

    public override IReadOnlyCollection<string> Aliases => ["r"];

    public override string Summary => "One answer to a prompt (argument, --input file or piped), then exit";

    public override string Usage => """
        MODEL ["prompt"] [options]

        The prompt is the arguments after MODEL, followed by the text of --input files and of piped standard input, so
        `cat notes.txt | idrak run MODEL "Summarize"` asks about the notes. The answer streams to the output; --json
        prints one document with the text, the token counts and the speed instead; -v adds the figures on the error output.

        Options:
          -i, --input FILE       read (more of) the prompt from FILE (repeatable)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
              --schema FILE      ask for JSON matching this JSON schema and check the answer (exit 1 when it is not)
              --mcp SERVER       let the model call an MCP server's tools (an http(s) URL or a command; repeatable)

        Gap: the library has no JSON-schema constrained sampling yet (a logits processor; plans/plug-in.md gap 13), so
        --schema gives the schema to the model as an instruction and checks the answer afterwards (it is JSON, has the
        required properties and their types) instead of guaranteeing it while generating.
        """ + "\n" + GenerationSettings.Help + """

        Examples:
          idrak run Qwen/Qwen3-0.6B "Why is the sky blue?"
          cat notes.txt | idrak r qwen "Summarize in three bullets"
          idrak run ./model.gguf -i question.txt --temperature 0 --json
          idrak run qwen "Extract the name and age: Sara is 31." --schema person.json
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, .. GenerationSettings.ValueOptions, "--input", "--schema", "--mcp"];

    public override IReadOnlyCollection<string> Flags => GenerationSettings.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(Models.ShortForms.Concat(GenerationSettings.ShortForms).Append(KeyValuePair.Create("-i", "--input")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string prompt = PromptInput.Read(context, 1);
        var settings = GenerationSettings.From(context);
        JsonObject? schema = null;
        if (context.Option("--schema") is { } schemaPath)
        {
            schema = StructuredOutput.Read(schemaPath);
            settings.System = StructuredOutput.Instruction(settings.System, schema);
            context.Detail("--schema: the schema is an instruction and the answer is checked afterwards (no constrained sampling in the library yet)");
        }

        using var loaded = LoadedChat.Load(context, name, settings);
        var messages = new List<ChatMessage>();
        if (settings.System is { Length: > 0 } system)
        {
            messages.Add(new ChatMessage("system", system));
        }

        messages.Add(new ChatMessage("user", prompt));
        var responder = new ChatResponder(loaded.Chat, loaded.Registry)
        {
            Stream = context.Json ? null : context.Output,
            ThinkingStream = context.Verbose && !context.Json ? context.ErrorOutput : null,
        };
        using var interrupt = new Interrupt(context);           // the first Ctrl+C (or --timeout) stops after the current token
        ChatAnswer answer;
        try
        {
            answer = responder.Answer(messages, settings, loaded.Context, interrupt.Token);
        }
        catch (OperationCanceledException)
        {
            context.Error(interrupt.TimedOut ? $"stopped: --timeout {ProgressLine.Duration(context.Timeout!.Value)} ran out" : "stopped");
            return ExitCodes.Failed;
        }

        if (context.Verbose && !context.Json)
        {
            context.ErrorOutput.WriteLine(answer.Summary(loaded.Context));
        }

        var json = new JsonObject
        {
            ["model"] = loaded.Choice.Model,
            ["device"] = context.Device.ToString(),
            ["text"] = answer.Message.Content,
        };
        if (answer.Message.Thinking is { } thinking)
        {
            json["thinking"] = thinking;
        }

        if (answer.Message.ToolCalls is { Count: > 0 } calls)
        {
            json["requested_tool_calls"] = new JsonArray([.. calls.Select(c => (JsonNode)new JsonObject { ["name"] = c.Name, ["arguments"] = c.Arguments.DeepClone() })]);
        }

        foreach (var (key, value) in answer.Figures())
        {
            json[key] = value?.DeepClone();
        }

        if (schema is not null)
        {
            var (value, problem) = StructuredOutput.Check(answer.Message.Content, schema);
            json["json"] = value?.DeepClone();
            json["schema_valid"] = problem is null;
            if (problem is not null)
            {
                json["schema_error"] = problem;
                context.Error($"The answer does not match the schema: {problem}");
            }

            context.WriteJson(json);
            return problem is null ? ExitCodes.Ok : ExitCodes.Failed;
        }

        context.WriteJson(json);
        return ExitCodes.Ok;
    }
}
