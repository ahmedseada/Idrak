// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli;

/// <summary>
/// Parses <c>idrak WORDS... [arguments] [options]</c>: the longest run of leading words that names a command selects it
/// ("cache info" before "cache"), then options (<c>--name value</c>, <c>--name=value</c>, flags) and positional
/// arguments in any order; <c>--</c> ends the options.
/// </summary>
internal static class CommandLine
{
    /// <summary>Runs the tool with <paramref name="args"/>; returns the exit code. Used by Main and by the tests.</summary>
    public static int Run(IReadOnlyList<string> args, TextWriter output, TextWriter error)
    {
        var commands = CommandTable.All;
        try
        {
            args = Shared.ResponseFiles.Expand(args);
        }
        catch (UsageException e)
        {
            error.WriteLine(e.Message);
            return ExitCodes.Usage;
        }

        if (args.Count > 0 && args[0] is "--version" or "-V")
        {
            args = ["version", .. args.Skip(1)];
        }

        if (args.Count == 0 || args[0] is "--help" or "-h" or "help" && args.Count == 1)
        {
            output.Write(Help.Overview(commands));
            return args.Count == 0 ? ExitCodes.Usage : ExitCodes.Ok;
        }

        if (args[0] == "help")
        {
            // A concept page (idrak help topics, help devices, help exit codes) first, then the command of that name.
            var target = Find(commands, args.Skip(1).ToList(), out _);
            string? topic = Shared.HelpTopics.Page(string.Join(' ', args.Skip(1)));
            if (topic is not null)
            {
                output.Write(target is null ? topic : target.Name == "env" ? Help.For(target) : $"{topic}\n{Help.For(target)}");
                return ExitCodes.Ok;
            }

            output.Write(target is null ? Help.Overview(commands) : Help.For(target));
            return target is null ? ExitCodes.Usage : ExitCodes.Ok;
        }

        var command = Find(commands, args, out int used);
        if (command is null)
        {
            // A group word alone ("idrak cache") lists the group's commands.
            var group = commands.Where(c => c.Name.StartsWith(args[0] + " ", StringComparison.Ordinal)).Select(c => "idrak " + c.Name).ToList();
            error.WriteLine(group.Count > 0
                ? $"'{args[0]}' needs a subcommand: {string.Join(", ", group)}."
                : $"Unknown command '{string.Join(' ', args.TakeWhile(a => !a.StartsWith('-')).Take(2))}'. Run 'idrak help' for the commands.");
            return ExitCodes.Usage;
        }

        CommandContext? context = null;
        try
        {
            var (positional, options, flags) = Parse(command, args.Skip(used).ToList());
            if (flags.Contains("--help"))
            {
                output.Write(Help.For(command));
                return ExitCodes.Ok;
            }

            context = new CommandContext(command, positional, options, flags, output, error);
            command.BeforePlugins(context);
            context.LoadPlugins();
            return command.Run(context);
        }
        catch (UsageException e)
        {
            error.WriteLine(e.Message);
            error.WriteLine($"Usage: idrak {command.Name} {command.Usage.Split('\n')[0]}".TrimEnd());
            return ExitCodes.Usage;
        }
        catch (Exception e) when (e is not OutOfMemoryException && context?.Timeout is { } limit && context.TimeoutToken.IsCancellationRequested)
        {
            // Whatever the cancelled call threw (a cancelled task, a closed connection), the reason is the time limit.
            error.WriteLine($"idrak {command.Name}: gave up after --timeout {Shared.Units.Duration(limit)} ({e.Message.Split('\n')[0]})");
            return ExitCodes.Failed;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            error.WriteLine($"idrak {command.Name}: {e.Message}");
            if (Environment.GetEnvironmentVariable("IDRAK_TRACE") is "1" or "true")
            {
                error.WriteLine(e.ToString());
            }

            return ExitCodes.Failed;
        }
        finally
        {
            context?.Dispose();
        }
    }

    // The command whose name is the longest run of leading words of args.
    internal static Command? Find(IReadOnlyList<Command> commands, IReadOnlyList<string> args, out int used)
    {
        used = 0;
        Command? best = null;
        foreach (var command in commands)
        {
            foreach (string name in command.Aliases.Prepend(command.Name))
            {
                var words = name.Split(' ');
                if (words.Length > used && words.Length <= args.Count && words.Select((w, i) => args[i] == w).All(m => m))
                {
                    best = command;
                    used = words.Length;
                }
            }
        }

        return best;
    }

    internal static (List<string> Positional, Dictionary<string, List<string>> Options, HashSet<string> Flags) Parse(Command command, IReadOnlyList<string> args)
    {
        var valueOptions = new HashSet<string>(CommandContext.CommonValueOptions.Concat(command.ValueOptions), StringComparer.Ordinal);
        var knownFlags = new HashSet<string>(CommandContext.CommonFlags.Concat(command.Flags), StringComparer.Ordinal);
        var positional = new List<string>();
        var options = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var flags = new HashSet<string>(StringComparer.Ordinal);
        bool optionsEnded = false;
        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            if (optionsEnded || !arg.StartsWith('-') || arg == "-")
            {
                positional.Add(arg);
            }
            else if (arg == "--")
            {
                optionsEnded = true;
            }
            else
            {
                string name = arg, value = "";
                int equals = arg.IndexOf('=');
                bool inline = equals > 0;
                if (inline)
                {
                    (name, value) = (arg[..equals], arg[(equals + 1)..]);
                }

                if (!name.StartsWith("--", StringComparison.Ordinal))
                {
                    name = CommandContext.CommonShortForms.TryGetValue(name, out string? common) ? common
                        : command.ShortForms.TryGetValue(name, out string? own) ? own
                        : throw new UsageException($"Unknown option {name}.");
                }

                if (valueOptions.Contains(name))
                {
                    if (!inline)
                    {
                        value = i + 1 < args.Count ? args[++i] : throw new UsageException($"{name} needs a value.");
                    }

                    (options.TryGetValue(name, out var list) ? list : options[name] = []).Add(value);
                }
                else if (inline)
                {
                    throw new UsageException($"{name} takes no value.");
                }
                else if (knownFlags.Contains(name))
                {
                    flags.Add(name);
                }
                else
                {
                    throw new UsageException($"Unknown option {name}.");
                }
            }
        }

        return (positional, options, flags);
    }
}

/// <summary>Help text.</summary>
internal static class Help
{
    /// <summary>
    /// <c>idrak help</c>: the commands by group (plans/idrak-cli.md: setup and health, run models, serve, ...), each with
    /// its aliases and summary, then the common options.
    /// </summary>
    public static string Overview(IReadOnlyList<Command> commands)
    {
        var text = new System.Text.StringBuilder("idrak: Idrak's command-line tool\n\nUsage: idrak COMMAND [arguments] [options]\n");
        string Name(Command c) => c.Aliases.Count > 0 ? $"{c.Name} ({string.Join(", ", c.Aliases)})" : c.Name;
        int width = commands.Count == 0 ? 0 : commands.Max(c => Name(c).Length);
        var groups = CommandTable.Groups.Select(g => (Title: g.Title, Commands: g.Commands.Where(commands.Contains).ToList()))
            .Append((Title: "Other", Commands: commands.Where(c => !CommandTable.All.Contains(c)).ToList()));
        foreach (var (title, members) in groups.Where(g => g.Commands.Count > 0))
        {
            text.Append('\n').Append(title).Append(":\n");
            foreach (var c in members)
            {
                text.Append("  ").Append(Name(c).PadRight(width)).Append("  ").Append(c.Summary).Append('\n');
            }
        }

        return text.Append('\n').Append(CommonOptions)
            .Append("\nRun 'idrak help COMMAND' (or 'idrak COMMAND --help') for a command's options; 'idrak help topics' for concept pages.\n").ToString();
    }

    /// <summary>
    /// <c>idrak help COMMAND</c>, in one layout for every command: the summary, the usage line and aliases, the command's
    /// own text (what it does, its arguments and options with their short forms, examples, limits and gaps), any short
    /// form that text does not show, the common options, and the environment variables that affect it (generated from
    /// <see cref="Shared.EnvironmentVariables"/>).
    /// </summary>
    public static string For(Command command)
    {
        string synopsis = command.Usage.Split('\n')[0].Trim();
        string body = Body(command);
        var text = new System.Text.StringBuilder($"idrak {command.Name}: {command.Summary}\n\n");
        text.Append($"Usage: idrak {command.Name} {synopsis}".TrimEnd()).Append('\n');
        if (command.Aliases.Count > 0)
        {
            text.Append("Aliases: ").Append(string.Join(", ", command.Aliases.Select(a => "idrak " + a))).Append('\n');
        }

        if (body.Length > 0)
        {
            text.Append('\n').Append(body).Append('\n');
        }

        var unshown = command.ShortForms.Where(p => !body.Contains($"{p.Key}, {p.Value}", StringComparison.Ordinal)).ToList();
        if (unshown.Count > 0)
        {
            text.Append("\nShort forms: ").Append(string.Join(", ", unshown.Select(p => $"{p.Key} {p.Value}"))).Append('\n');
        }

        return text.Append('\n').Append(CommonOptions).Append(Shared.EnvironmentVariables.HelpSection(command.Name)).ToString();
    }

    /// <summary>The command's own text in its help (the usage after its first line), as <see cref="For"/> shows it.</summary>
    public static string Body(Command command)
    {
        string usage = command.Usage.Replace("\r", "", StringComparison.Ordinal);
        int newline = usage.IndexOf('\n');
        return newline < 0 ? "" : usage[(newline + 1)..].Trim('\n').TrimEnd();
    }

    private const string CommonOptions =
        "Common options:\n" +
        "  -d, --device NAME   cpu, cuda:0, vulkan:1, hip:0 (default: the default device)\n" +
        "  -P, --plugin PATH   load an assembly that registers formats, families, ... (repeatable)\n" +
        "  -j, --json          machine-readable output\n" +
        "  -q, --quiet         less output; -v, --verbose: more\n" +
        "      --cache DIR     the cache folder (default IDRAK_CACHE or ~/.cache/idrak)\n" +
        "  -C, --config FILE   the config file (default IDRAK_CONFIG or ~/.idrak/config.json)\n" +
        "      --log FILE      also write every line, verbose ones included, to FILE\n" +
        "      @FILE           read arguments from FILE, one per line (@@text: a literal @text)\n" +
        "  -O, --output FILE   write the command's main output to FILE\n" +
        "      --format F      text, json, csv or md (tables)\n" +
        "      --color WHEN    auto, always or never (NO_COLOR still wins); --plain: no Unicode or progress line\n" +
        "      --offline       use only what is cached; a download is an error\n" +
        "      --threads N     CPU threads; --seed N: one seed for sampling, shuffling and initialization\n" +
        "      --timeout D     give up after a duration (30s, 5m, 1h30m)\n" +
        "  -h, --help          this help; idrak -V, --version: the versions; idrak help topics: concept pages\n";
}
