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
            var target = Find(commands, args.Skip(1).ToList(), out _);
            output.Write(target is null ? Help.Overview(commands) : Help.For(target));
            return target is null ? ExitCodes.Usage : ExitCodes.Ok;
        }

        var command = Find(commands, args, out int used);
        if (command is null)
        {
            error.WriteLine($"Unknown command '{string.Join(' ', args.TakeWhile(a => !a.StartsWith('-')).Take(2))}'. Run 'idrak help' for the commands.");
            return ExitCodes.Usage;
        }

        try
        {
            var (positional, options, flags) = Parse(command, args.Skip(used).ToList());
            if (flags.Contains("--help"))
            {
                output.Write(Help.For(command));
                return ExitCodes.Ok;
            }

            var context = new CommandContext(command, positional, options, flags, output, error);
            context.LoadPlugins();
            return command.Run(context);
        }
        catch (UsageException e)
        {
            error.WriteLine(e.Message);
            error.WriteLine($"Usage: idrak {command.Name} {command.Usage}".TrimEnd());
            return ExitCodes.Usage;
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
    public static string Overview(IReadOnlyList<Command> commands)
    {
        var text = new System.Text.StringBuilder("idrak: Idrak's command-line tool\n\nUsage: idrak COMMAND [arguments] [options]\n\nCommands:\n");
        int width = commands.Count == 0 ? 0 : commands.Max(c => c.Name.Length);
        foreach (var c in commands.OrderBy(c => c.Name, StringComparer.Ordinal))
        {
            text.Append("  ").Append(c.Name.PadRight(width)).Append("  ").Append(c.Summary)
                .Append(c.Aliases.Count > 0 ? $" ({string.Join(", ", c.Aliases)})" : "").Append('\n');
        }

        return text.Append('\n').Append(CommonOptions).Append("\nRun 'idrak help COMMAND' for a command's options.\n").ToString();
    }

    public static string For(Command command)
    {
        var text = new System.Text.StringBuilder($"idrak {command.Name}: {command.Summary}\n\nUsage: idrak {command.Name} {command.Usage}".TrimEnd()).Append('\n');
        if (command.Aliases.Count > 0)
        {
            text.Append("Aliases: ").Append(string.Join(", ", command.Aliases.Select(a => "idrak " + a))).Append('\n');
        }

        if (command.ShortForms.Count > 0)
        {
            text.Append("Short forms: ").Append(string.Join(", ", command.ShortForms.Select(p => $"{p.Key} {p.Value}"))).Append('\n');
        }

        return text.Append('\n').Append(CommonOptions).ToString();
    }

    private const string CommonOptions =
        "Common options:\n" +
        "  -d, --device NAME   cpu, cuda:0, vulkan:1, hip:0 (default: the default device)\n" +
        "  -P, --plugin PATH   load an assembly that registers formats, families, ... (repeatable)\n" +
        "  -j, --json          machine-readable output\n" +
        "  -q, --quiet         less output; -v, --verbose: more\n" +
        "      --cache DIR     the cache folder (default IDRAK_CACHE or ~/.cache/idrak)\n" +
        "  -C, --config FILE   the config file (default IDRAK_CONFIG or ~/.idrak/config.json)\n" +
        "  -h, --help          this help; idrak -V, --version: the versions\n";
}
