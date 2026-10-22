// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak agent MODEL</c>: the coding agent (read, search, edit, run commands in a folder).</summary>
internal sealed class AgentCommand : Command
{
    public override string Name => "agent";

    public override string Summary => "The coding agent: reads, searches, edits files and runs commands in a folder";

    public override string Usage => """
        MODEL ["task"] [options]

        The model works in the folder (--workspace, default the current one) with the coding tools: list_files,
        read_file, search, edit_file, write_file and run_command (allowlisted programs started without a shell:
        git and those of the coding languages, such as dotnet, npm, node; paths cannot leave the folder). The
        coding languages (Idrak.Text.Plugins registers C#, TypeScript, Angular and Python) set the programs, the
        folders skipped and the prompt's build and test commands: all registered ones by default, --languages
        detected for those the folder shows, or names. A folder no chosen language detects gets a note naming
        the plug-in to register. With a task it works on it and exits; without one it reads tasks line by line,
        as chat does, keeping the conversation (/exit ends). Every command the model wants to run is shown and
        asked about (y to run it) unless --yes is given; without a terminal and without --yes, commands are
        refused.

        Options:
              --workspace DIR    the folder to work in (default: the current folder)
          -y, --yes              run commands without asking
              --read-only        only the reading tools (no edits, writes or commands)
              --rounds N         model replies allowed per task (default 40)
              --languages LIST   the coding languages: all, detected, none or names such as csharp,typescript
                                 (default: IDRAK_CODING_LANGUAGES, else all)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
          --head-weights F       the output head's format beside -w (bf16: the checkpoint's own scores and one
                                 vocabulary table for tied models, less memory than int8/int4 heads, slower)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
        """ + "\n" + GenerationSettings.Help.Replace("  -s, --system TEXT      the system prompt\n", "  -s, --system TEXT      instructions instead of the agent's own\n") + """

        Examples:
          idrak agent Qwen/Qwen3-8B "Add a unit test for Parse" --workspace ./src
          idrak agent qwen -y --context 16384
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, .. GenerationSettings.ValueOptions, "--workspace", "--rounds", "--languages"];

    public override IReadOnlyCollection<string> Flags => [.. GenerationSettings.Flags, "--yes", "--read-only"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(ModelChoices.ShortForms.Concat(GenerationSettings.ShortForms).Append(KeyValuePair.Create("-y", "--yes")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string workspace = Path.GetFullPath(context.Option("--workspace") ?? Directory.GetCurrentDirectory());
        if (!Directory.Exists(workspace))
        {
            throw new UsageException($"Workspace folder not found: {workspace}");
        }

        int rounds = context.IntOption("--rounds", 40);
        var settings = GenerationSettings.From(context);
        string? task = context.Positional.Count > 1 ? string.Join(' ', context.Positional.Skip(1)) : null;
        var input = StandardInput.Reader;
        bool interactive = !StandardInput.IsRedirected;
        bool yes = context.Flag("--yes");

        CodingTools coding;
        try
        {
            coding = new CodingTools(workspace, new CodingToolOptions { ReadOnly = context.Flag("--read-only"), Languages = context.Option("--languages") ?? CodingLanguages.DefaultChoice });
        }
        catch (NotSupportedException e) when (e.Message.Contains("Coding language", StringComparison.Ordinal))
        {
            throw new UsageException(e.Message);
        }

        if (coding.LanguageNotice is { } notice)
        {
            context.Error(notice);
        }

        var builder = ToolRegistry.Create().Add(coding.All);
        if (!yes && coding.All.Any(t => t.Definition.Name == "run_command"))
        {
            builder.RequireApproval("run_command", (call, _) =>
            {
                string command = (string?)call.Arguments["command"] ?? call.Arguments.ToJsonString();
                if (!interactive && StandardInput.Override is null)
                {
                    context.Error($"Refused '{command}': no terminal to ask; give --yes to allow commands.");
                    return ValueTask.FromResult(false);
                }

                context.Output.Write($"Run '{command}'? [y/N] ");
                context.Output.Flush();
                string answer = input.ReadLine()?.Trim().ToLowerInvariant() ?? "";
                return ValueTask.FromResult(answer is "y" or "yes");
            });
        }

        var extra = ToolAssemblies.Load(settings.ToolAssemblies);
        var registry = builder.Add(extra).Build();
        settings.ToolAssemblies = [];
        using var loaded = LoadedChat.Load(context, name, settings);
        var color = Terminal.UseColour(context);
        var responder = new ChatResponder(loaded.Chat, registry)
        {
            MaxToolRounds = rounds,
            Stream = context.Json ? null : context.Output,
            ThinkingStream = context.Json || context.Quiet ? null : context.Output,
            Color = color,
        };
        var messages = new List<ChatMessage> { new("system", settings.System ?? CodingAgent.SystemPrompt(coding.Languages)) };
        var answers = new List<ChatAnswer>();
        void Work(string text)
        {
            messages.Add(new ChatMessage("user", text));
            var answer = responder.Answer(messages, settings, loaded.Context);
            answers.Add(answer);
            context.Write($"[{answer.Rounds} replies, {answer.ToolResults.Count} tool calls ({answer.ToolResults.Count(r => !r.Succeeded)} errors), "
                          + $"{answer.GeneratedTokens} tokens at {answer.TokensPerSecond:F1} tokens/s]");
        }

        if (task is not null)
        {
            Work(task);
        }
        else
        {
            if (interactive)
            {
                context.Write($"Coding agent with {loaded.Describe(context)} in {workspace}. Type a task; /exit ends.");
            }

            while (true)
            {
                if (interactive && !context.Json)
                {
                    context.Output.Write("> ");
                }

                if (input.ReadLine()?.Trim() is not { } line || line is "/exit" or "/quit")
                {
                    break;
                }

                if (line.Length > 0)
                {
                    Work(line);
                }
            }
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = loaded.Choice.Model,
            ["workspace"] = workspace,
            ["tasks"] = answers.Count,
            ["tool_calls"] = answers.Sum(a => a.ToolResults.Count),
            ["tool_errors"] = answers.Sum(a => a.ToolResults.Count(r => !r.Succeeded)),
            ["generated_tokens"] = answers.Sum(a => a.GeneratedTokens),
            ["messages"] = ChatJson.Messages(messages),
        });
        return ExitCodes.Ok;
    }
}
