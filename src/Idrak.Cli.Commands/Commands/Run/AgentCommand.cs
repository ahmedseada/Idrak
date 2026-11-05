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
        git and the coding languages' build and test runners, such as dotnet, npm, tsc; --commands or
        IDRAK_AGENT_COMMANDS names others, such as node; paths cannot leave the folder, nor follow a link out of
        it). Commands run in the sandbox --sandbox names (IDRAK_SANDBOX, default process: a plain process with
        your rights, nothing isolated, said in the first line; container runs each in Docker or Podman, for a
        model you do not trust). The coding languages (Idrak.Text.Plugins registers C#, TypeScript, Angular and Python) set the programs, the
        folders skipped and the prompt's build and test commands: all registered ones by default, --languages
        detected for those the folder shows, or names. A folder no chosen language detects gets a note naming
        the plug-in to register. With a task it works on it and exits; without one it reads tasks line by line,
        as chat does, keeping the conversation (/exit ends). Every command the model wants to run, every file
        it wants to write or edit and every call of a --tools tool is shown and asked about (y to go on) unless
        --yes is given; without a terminal and without --yes, they are refused. Calls the model writes as text (markdown json fences, JSON in its prose) naming an offered tool
        are read and run unless --strict-calls is given; a reply that looks like calls none of which can be read
        gets a correction showing the template's call format, at most --nudges in a row, then the task ends
        saying so.

        Options:
              --workspace DIR    the folder to work in (default: the current folder)
          -y, --yes              run commands, write files and call --tools tools without asking
              --sandbox NAME     where commands run: process (a plain process, the default) or container
                                 (Docker or Podman; IDRAK_SANDBOX)
              --commands LIST    the programs commands may start, such as dotnet,npm,node,git (default:
                                 IDRAK_AGENT_COMMANDS, else the languages' build and test runners and git)
              --read-only        only the reading tools (no edits, writes or commands)
              --rounds N         model replies allowed per task (default 40)
              --nudges N         corrections in a row for calls that could not be read (default 2; 0: none)
              --strict-calls     read calls only as the chat template writes them, not from fences or bare JSON
              --languages LIST   the coding languages: all, detected, none or names such as csharp,typescript
                                 (default: IDRAK_CODING_LANGUAGES, else all)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: auto)
          --head-weights F       the output head's format beside -w (bf16: the checkpoint's own scores and one
                                 vocabulary table for tied models, less memory than int8/int4 heads, slower)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default: the run's type)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
        """ + "\n" + GenerationSettings.Help.Replace("  -s, --system TEXT      the system prompt\n", "  -s, --system TEXT      instructions instead of the agent's own\n") + """

        Examples:
          idrak agent Qwen/Qwen3-8B "Add a unit test for Parse" --workspace ./src
          idrak agent qwen -y --context 16384
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, .. GenerationSettings.ValueOptions, "--workspace", "--rounds", "--languages", "--nudges", "--sandbox", "--commands"];

    public override IReadOnlyCollection<string> Flags => [.. GenerationSettings.Flags, "--yes", "--read-only", "--strict-calls"];

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
        int nudges = context.IntOption("--nudges", 2);
        if (nudges < 0)
        {
            throw new UsageException($"--nudges takes 0 or more corrections, not {nudges}.");
        }
        var settings = GenerationSettings.From(context);
        string? task = context.Positional.Count > 1 ? string.Join(' ', context.Positional.Skip(1)) : null;
        var input = StandardInput.Reader;
        bool interactive = !StandardInput.IsRedirected;
        bool yes = context.Flag("--yes");

        // Writes are asked about as commands are (the "ask" write policy, registered for this run), unless --yes.
        using var asking = AskWrites(context, input, interactive);
        CodingTools coding;
        try
        {
            var options = new CodingToolOptions
            {
                ReadOnly = context.Flag("--read-only"), Languages = context.Option("--languages") ?? CodingLanguages.DefaultChoice,
                WritePolicy = yes ? WritePolicies.Allow : WritePolicies.Ask,
            };
            coding = new CodingTools(workspace, options with
            {
                Sandbox = context.Option("--sandbox") ?? options.Sandbox,
                Commands = context.Option("--commands") is { } commands ? commands.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) : options.Commands,
            });
        }
        catch (NotRegisteredException e) when (e.Registry is nameof(CodingLanguages) or nameof(CommandSandboxes))
        {
            throw new UsageException(e.Message);
        }

        if (coding.All.Any(t => t.Definition.Name == "run_command"))
        {
            // The first line: where the model's commands run, and how to isolate them.
            context.Error(coding.Sandbox.Name.Equals(CommandSandboxes.Process, StringComparison.OrdinalIgnoreCase)
                ? $"Commands run in the {CommandSandboxes.Process} sandbox: a plain process of this machine with your rights, nothing isolated (for models you trust); "
                  + $"for one you do not, run them in a container: --sandbox {CommandSandboxes.Container} or {CommandSandboxes.EnvironmentVariable}={CommandSandboxes.Container} (Docker or Podman)."
                : $"Commands run in the {coding.Sandbox.Name} sandbox: {coding.Sandbox.Summary}.");
        }

        if (coding.LanguageNotice is { } notice)
        {
            context.Error(notice);
        }

        var extra = ToolAssemblies.Load(settings.ToolAssemblies);
        ValueTask<bool> Ask(string what)
        {
            if (!interactive && StandardInput.Override is null)
            {
                context.Error($"Refused {what}: no terminal to ask; give --yes to allow it.");
                return ValueTask.FromResult(false);
            }

            context.Output.Write($"{char.ToUpperInvariant(what[0])}{what[1..]}? [y/N] ");
            context.Output.Flush();
            string answer = input.ReadLine()?.Trim().ToLowerInvariant() ?? "";
            return ValueTask.FromResult(answer is "y" or "yes");
        }

        var registry = Tools(coding, extra, yes ? null : Ask);
        settings.ToolAssemblies = [];
        using var loaded = LoadedChat.Load(context, name, settings);
        var color = Terminal.UseColour(context);
        var responder = new ChatResponder(loaded.Chat, registry)
        {
            MaxToolRounds = rounds,
            Stream = context.Json ? null : context.Output,
            ThinkingStream = context.Json || context.Quiet ? null : context.Output,
            Color = color,
            TextCalls = CodingAgent.CallContext(loaded.Chat.Template, registry.Definitions),
            ReadCallsInText = !context.Flag("--strict-calls"),
            Nudges = nudges,
        };
        var messages = new List<ChatMessage> { new("system", settings.System ?? CodingAgent.SystemPrompt(coding.Languages)) };
        var answers = new List<ChatAnswer>();
        void Work(string text)
        {
            messages.Add(new ChatMessage("user", text));
            var answer = responder.Answer(messages, settings, loaded.Context);
            answers.Add(answer);
            if (answer.CallsNotRead)
            {
                context.Error($"The model wrote tool calls that could not be read{(answer.Nudges > 0 ? $", after {answer.Nudges} correction{(answer.Nudges == 1 ? "" : "s")}" : "")}; "
                              + "nothing from its last reply was run.");
            }

            context.Write($"[{answer.Rounds} replies, {answer.ToolResults.Count} tool calls ({answer.ToolResults.Count(r => !r.Succeeded)} errors), "
                          + $"{(answer.Nudges > 0 ? $"{answer.Nudges} corrections, " : "")}{answer.GeneratedTokens} tokens at {answer.TokensPerSecond:F1} tokens/s]");
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
            ["nudges"] = answers.Sum(a => a.Nudges),
            ["calls_not_read"] = answers.Count(a => a.CallsNotRead),
            ["messages"] = ChatJson.Messages(messages),
        });
        return ExitCodes.Ok;
    }

    /// <summary>
    /// The agent's tools: the coding tools, then the <c>--tools</c> assemblies' <paramref name="extra"/>. Every tool that
    /// changes something outside the model's text is asked about through <paramref name="ask"/> (null: <c>--yes</c>, none
    /// is): run_command, and every extra tool, whatever it does; write_file and edit_file ask through the write policy.
    /// </summary>
    internal static ToolRegistry Tools(CodingTools coding, IReadOnlyList<ITool> extra, Func<string, ValueTask<bool>>? ask)
    {
        var builder = ToolRegistry.Create().Add(coding.All).Add(extra);
        if (ask is not null)
        {
            if (coding.All.Any(t => t.Definition.Name == "run_command"))
            {
                builder.RequireApproval("run_command", (call, _) => ask($"run '{(string?)call.Arguments["command"] ?? call.Arguments.ToJsonString()}'"));
            }

            foreach (var tool in extra)
            {
                builder.RequireApproval(tool.Definition.Name, (call, _) => ask($"call the tool '{call.Name}' with {call.Arguments.ToJsonString()}"));
            }
        }

        return builder.Build();
    }

    // The "ask" write policy for this run (WritePolicies.Ask): each write_file and edit_file shown and asked about on the
    // terminal (y to write), refused without one; unregistered when the run ends.
    private static AskRegistration AskWrites(CommandContext context, TextReader input, bool interactive)
    {
        WritePolicies.Register(new AskingWritePolicy((write, _) =>
        {
            string what = $"{(write.Tool == "edit_file" ? "Edit" : "Write")} '{write.Path}' ({(write.Creates ? "new file, " : "")}{write.Change})";
            if (!interactive && StandardInput.Override is null)
            {
                context.Error($"Refused: {what}: no terminal to ask; give --yes to allow writes.");
                return ValueTask.FromResult(false);
            }

            context.Output.Write($"{what}? [y/N] ");
            context.Output.Flush();
            string answer = input.ReadLine()?.Trim().ToLowerInvariant() ?? "";
            return ValueTask.FromResult(answer is "y" or "yes");
        }));
        return new AskRegistration();
    }

    private sealed class AskRegistration : IDisposable
    {
        public void Dispose() => WritePolicies.Unregister(WritePolicies.Ask);
    }
}
