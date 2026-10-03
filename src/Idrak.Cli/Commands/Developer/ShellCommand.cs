// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak shell</c>: an interactive prompt for idrak commands (without the leading "idrak"), run in this process
/// through <see cref="CommandLine.Run"/>, so plug-ins, loaded devices and kernel caches stay warm between commands.
/// On a terminal it has a line editor with history (up and down arrows, kept in a file) and Tab completion of command
/// names and options; otherwise (or when a test sets <see cref="Input"/>) it reads plain lines.
/// </summary>
internal sealed class ShellCommand : Command
{
    /// <summary>Where lines come from instead of the terminal (tests set it).</summary>
    internal static TextReader? Input { get; set; }

    public override string Name => "shell";

    public override string Summary => "An interactive prompt for idrak commands, with history and completion";

    public override string Usage =>
        "[--history FILE]\n\n" +
        "Type commands without 'idrak' (e.g. 'devices', 'demo xor -d vulkan:0'). Tab completes command names and\n" +
        "options; the arrows recall earlier lines. Inside the shell: 'history' lists the lines, '!N' runs line N again,\n" +
        "'exit' or 'quit' (or end of input) leaves. The common options given to shell (--device, --plugin, --config,\n" +
        "--cache, --verbose, --quiet) apply to every command typed.\n\n" +
        "Options:\n" +
        "      --history FILE  where the history is kept (default ~/.idrak/history; 'none' keeps none)\n\n" +
        "Examples:\n" +
        "  idrak shell\n" +
        "  idrak shell -d vulkan:0 -P ./MyFormat.dll\n\n" +
        "Environment: IDRAK_CONFIG (the config file, also for the commands typed), NO_COLOR.";

    public override IReadOnlyCollection<string> ValueOptions => ["--history"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"Unexpected argument '{context.Positional[0]}'; type commands at the prompt.");
        }

        if (context.Json)
        {
            throw new UsageException("idrak shell is interactive and has no JSON output; pass --json to the commands you type.");
        }

        string? historyFile = context.Option("--history") is "none" ? null
            : context.Option("--history") ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".idrak", "history");
        var history = new List<string>();
        if (historyFile is not null && File.Exists(historyFile))
        {
            history.AddRange(File.ReadAllLines(historyFile).Where(l => l.Length > 0).TakeLast(1000));
        }

        int loaded = history.Count;
        var inherited = new List<string>();
        foreach (string option in CommandContext.CommonValueOptions)
        {
            inherited.AddRange(context.Options(option).SelectMany(v => new[] { option, v }));
        }

        inherited.AddRange(new[] { "--quiet", "--verbose" }.Where(context.Flag));
        bool editor = Input is null && !Console.IsInputRedirected && !Console.IsOutputRedirected && ReferenceEquals(context.Output, Console.Out);
        var reader = Input ?? Console.In;
        if (!context.Quiet)
        {
            context.Output.WriteLine("idrak shell: type a command ('help' for the list, Tab to complete, 'exit' to leave).");
        }

        int last = ExitCodes.Ok;
        while (true)
        {
            context.Output.Write("idrak> ");
            context.Output.Flush();
            string? line = editor ? ReadLine(history) : reader.ReadLine();
            if (line is null)
            {
                context.Output.WriteLine();
                break;
            }

            line = line.Trim();
            if (line.Length == 0)
            {
                continue;
            }

            if (line.StartsWith('!') && int.TryParse(line[1..], out int number))
            {
                if (number < 1 || number > history.Count)
                {
                    context.Error($"No line {number} in the history ({history.Count} lines).");
                    continue;
                }

                line = history[number - 1];
                context.Output.WriteLine(line);
            }

            if (line is "exit" or "quit")
            {
                break;
            }

            if (history.Count == 0 || history[^1] != line)
            {
                history.Add(line);
            }

            if (line == "history")
            {
                for (int i = 0; i < history.Count; i++)
                {
                    context.Output.WriteLine($"{i + 1,5}  {history[i]}");
                }

                continue;
            }

            List<string> args;
            try
            {
                args = Split(line);
            }
            catch (FormatException e)
            {
                context.Error(e.Message);
                continue;
            }

            if (args.Count > 0 && args[0] == "idrak")
            {
                args.RemoveAt(0);
            }

            if (args.Count > 0 && CommandLine.Find(CommandTable.All, args, out _) is ShellCommand)
            {
                context.Error("Already in the shell.");
                continue;
            }

            last = CommandLine.Run([.. args, .. inherited], context.Output, context.ErrorOutput);
            if (last != ExitCodes.Ok)
            {
                context.Detail($"(exit code {last})");
            }
        }

        if (historyFile is not null && history.Count > loaded)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(historyFile))!);
                File.WriteAllLines(historyFile, history.TakeLast(1000));
            }
            catch (IOException e)
            {
                context.Error($"Could not save the history to {historyFile}: {e.Message}");
            }
        }

        return ExitCodes.Ok;
    }

    /// <summary>Splits a typed line into arguments: spaces separate, single or double quotes group, a backslash escapes.</summary>
    internal static List<string> Split(string line)
    {
        var args = new List<string>();
        var current = new StringBuilder();
        bool any = false;
        char? quote = null;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '\\' && i + 1 < line.Length && quote != '\'')
            {
                current.Append(line[++i]);
                any = true;
            }
            else if (quote is not null)
            {
                if (c == quote)
                {
                    quote = null;
                }
                else
                {
                    current.Append(c);
                }
            }
            else if (c is '"' or '\'')
            {
                quote = c;
                any = true;
            }
            else if (char.IsWhiteSpace(c))
            {
                if (any)
                {
                    args.Add(current.ToString());
                    current.Clear();
                    any = false;
                }
            }
            else
            {
                current.Append(c);
                any = true;
            }
        }

        if (quote is not null)
        {
            throw new FormatException($"Unclosed {quote} quote.");
        }

        if (any)
        {
            args.Add(current.ToString());
        }

        return args;
    }

    /// <summary>
    /// The completions of the last word of <paramref name="line"/>: the next word of a command name while the command
    /// is not chosen yet, else the chosen command's options and the common ones.
    /// </summary>
    internal static IReadOnlyList<string> Complete(string line)
    {
        var words = line.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
        string partial = line.Length == 0 || line.EndsWith(' ') ? "" : words[^1];
        var before = partial.Length == 0 ? words : words[..^1];
        var command = CommandLine.Find(CommandTable.All, before, out int used);
        if (partial.StartsWith('-'))
        {
            if (command is null)
            {
                return [];
            }

            return [.. CommandContext.CommonValueOptions.Concat(CommandContext.CommonFlags).Concat(command.ValueOptions).Concat(command.Flags)
                .Where(o => o.StartsWith(partial, StringComparison.Ordinal)).Distinct().Order(StringComparer.Ordinal)];
        }

        // Command names (and aliases) that continue the words typed so far.
        var names = CommandTable.All.SelectMany(c => c.Aliases.Prepend(c.Name)).Select(n => n.Split(' '))
            .Where(n => n.Length > before.Count && n.Take(before.Count).SequenceEqual(before) && n[before.Count].StartsWith(partial, StringComparison.Ordinal))
            .Select(n => n[before.Count]).Distinct().Order(StringComparer.Ordinal).ToList();
        return command is not null && used == before.Count && names.Count == 0 ? [] : names;
    }

    // A small line editor: printable keys, Backspace, Left/Right, Home/End, Up/Down through the history, Tab completion.
    private static string? ReadLine(List<string> history)
    {
        var text = new StringBuilder();
        int cursor = 0, recalled = history.Count;
        int origin = Console.CursorLeft, drawn = 0;
        void Redraw()
        {
            Console.CursorLeft = origin;
            Console.Write(text.ToString() + new string(' ', Math.Max(0, drawn - text.Length)));   // clears what a longer line left
            drawn = text.Length;
            Console.CursorLeft = Math.Min(origin + cursor, Math.Max(0, Console.BufferWidth - 1));
        }

        while (true)
        {
            var key = Console.ReadKey(intercept: true);
            switch (key.Key)
            {
                case ConsoleKey.Enter:
                    Console.WriteLine();
                    return text.ToString();
                case ConsoleKey.D when key.Modifiers.HasFlag(ConsoleModifiers.Control) && text.Length == 0:
                    return null;
                case ConsoleKey.Backspace when cursor > 0:
                    text.Remove(--cursor, 1);
                    break;
                case ConsoleKey.Delete when cursor < text.Length:
                    text.Remove(cursor, 1);
                    break;
                case ConsoleKey.LeftArrow when cursor > 0:
                    cursor--;
                    break;
                case ConsoleKey.RightArrow when cursor < text.Length:
                    cursor++;
                    break;
                case ConsoleKey.Home:
                    cursor = 0;
                    break;
                case ConsoleKey.End:
                    cursor = text.Length;
                    break;
                case ConsoleKey.UpArrow when recalled > 0:
                case ConsoleKey.DownArrow when recalled < history.Count:
                    recalled += key.Key == ConsoleKey.UpArrow ? -1 : 1;
                    text.Clear().Append(recalled < history.Count ? history[recalled] : "");
                    cursor = text.Length;
                    break;
                case ConsoleKey.Tab:
                    string typed = text.ToString(0, cursor);
                    var options = Complete(typed);
                    string partial = typed.Length == 0 || typed.EndsWith(' ') ? "" : typed.Split(' ')[^1];
                    if (options.Count == 1)
                    {
                        string rest = options[0][partial.Length..] + " ";
                        text.Insert(cursor, rest);
                        cursor += rest.Length;
                    }
                    else if (options.Count > 1)
                    {
                        string common = options.Aggregate((a, b) => new string([.. a.Zip(b).TakeWhile(p => p.First == p.Second).Select(p => p.First)]));
                        if (common.Length > partial.Length)
                        {
                            text.Insert(cursor, common[partial.Length..]);
                            cursor += common.Length - partial.Length;
                        }
                        else
                        {
                            Console.WriteLine();
                            Console.WriteLine(string.Join("  ", options));
                            Console.Write("idrak> ");
                            origin = Console.CursorLeft;
                            drawn = 0;
                        }
                    }

                    break;
                default:
                    if (!char.IsControl(key.KeyChar))
                    {
                        text.Insert(cursor++, key.KeyChar);
                    }

                    break;
            }

            Redraw();
        }
    }
}
