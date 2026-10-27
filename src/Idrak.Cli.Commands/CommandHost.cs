// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Shared;

namespace Idrak.Cli;

/// <summary>
/// What a command needs from the process it runs in, beyond the library: declared by the command (<c>Command.Needs</c>,
/// or for the options given, <c>Command.NeedsFor</c>) or asked for where one of its options needs it
/// (<c>CommandContext.Require</c>), and refused with its reason where the host does not allow it
/// (<see cref="CommandHost.Allows"/>): an app runs every other command in its own process.
/// </summary>
[Flags]
public enum CommandNeeds
{
    /// <summary>Nothing beyond the library.</summary>
    None = 0,

    /// <summary>Starting other programs: dotnet, a browser, the ranks of a localhost cluster.</summary>
    ChildProcesses = 1,

    /// <summary>Opening a network listener: a server, a dashboard, a cluster's rendezvous.</summary>
    NetworkListener = 2,

    /// <summary>The console's keyboard, or standard input as a stream: an interactive shell, a protocol over standard input.</summary>
    ConsoleInput = 4,

    /// <summary>Everything above (a console tool's process).</summary>
    All = ChildProcesses | NetworkListener | ConsoleInput,
}

/// <summary>
/// The process the commands run in: what it allows them (<see cref="Allows"/>), what stands for standard input, and how
/// their progress lines are drawn. <see cref="Console"/> is the idrak tool's (everything allowed, the console's input, the
/// progress the terminal and <c>IDRAK_PROGRESS</c> choose); an app gives its own (<see cref="IdrakCommands.Run"/>).
/// </summary>
public sealed record CommandHost
{
    /// <summary>The idrak tool's process: a console, where every command may run.</summary>
    public static CommandHost Console { get; } = new();

    /// <summary>The host's name in a refusal ("the Idrak app"); "this process" by default.</summary>
    public string Name { get; init; } = "this process";

    /// <summary>What the commands may do here (<see cref="CommandNeeds.All"/> by default); a command that needs more is refused, its reason named.</summary>
    public CommandNeeds Allows { get; init; } = CommandNeeds.All;

    /// <summary>
    /// The folder the commands run in (null: the process's current folder): relative paths are read from it, and where the
    /// platform gives no home folder (<c>~</c>: the config file, the history), it is the home for the run. The cache and
    /// model folder is the library's (<see cref="ComputeResources.CacheFolder"/>, <c>IDRAK_CACHE</c>), so the commands see
    /// what the host's own model store pulled; <c>--cache</c> still chooses another.
    /// </summary>
    public string? WorkingFolder { get; init; }

    /// <summary>What a command reads as standard input (null: the console's): an app with no console gives an empty reader.</summary>
    public TextReader? Input { get; init; }

    /// <summary>
    /// The progress reporter the commands' progress lines use (a <see cref="Idrak.Diagnostics.ProgressReporters"/> name:
    /// "console" redraws a line in place with a carriage return, for a screen that does), over <c>IDRAK_PROGRESS</c>; null:
    /// <c>IDRAK_PROGRESS</c>, else "console" on a terminal and none elsewhere.
    /// </summary>
    public string? ProgressReporter { get; init; }

    /// <summary>Whether <paramref name="needs"/> are all allowed here.</summary>
    public bool Permits(CommandNeeds needs) => (needs & ~Allows) == 0;

    /// <summary>
    /// Why the command <paramref name="command"/> cannot run here with <paramref name="needs"/> ("idrak serve is not available
    /// in the Idrak app: it opens a network listener (a server)."), or null when it can; <paramref name="reason"/> adds what
    /// asked for it (an option).
    /// </summary>
    public string? Refusal(string command, CommandNeeds needs, string? reason = null)
    {
        ArgumentNullException.ThrowIfNull(command);
        var missing = needs & ~Allows;
        if (missing == 0)
        {
            return null;
        }

        var what = new List<string>();
        if ((missing & CommandNeeds.ChildProcesses) != 0)
        {
            what.Add(Messages.T("it starts other programs"));
        }

        if ((missing & CommandNeeds.NetworkListener) != 0)
        {
            what.Add(Messages.T("it opens a network listener (a server)"));
        }

        if ((missing & CommandNeeds.ConsoleInput) != 0)
        {
            what.Add(Messages.T("it reads the console's keyboard or standard input"));
        }

        string text = Messages.T("idrak {0} is not available in {1}: {2}.", command, Name, string.Join("; ", what));
        return reason is { Length: > 0 } ? $"{text} ({reason})" : text;
    }
}

/// <summary>A command (or one of its options) needs what its host does not allow (<see cref="CommandHost.Allows"/>): the tool says why and exits with 3.</summary>
internal sealed class CommandUnavailableException(string message) : Exception(message);

/// <summary>One command as a host lists it: its words, short names, one line of help, its group and what it needs from the host.</summary>
/// <param name="Name">The words after <c>idrak</c> ("cache info").</param>
/// <param name="Aliases">Its short names ("ls").</param>
/// <param name="Summary">Its line in <c>idrak help</c>.</param>
/// <param name="Group">Its group's title in <c>idrak help</c> ("Models").</param>
/// <param name="Needs">What it always needs from the host (an option may need more: the command says when it is given).</param>
/// <param name="Options">Its own options, those taking a value and its flags ("--port", "--json"), without the common ones (<see cref="IdrakCommands.CommonOptions"/>).</param>
public sealed record IdrakCommandInfo(string Name, IReadOnlyList<string> Aliases, string Summary, string Group, CommandNeeds Needs, IReadOnlyList<string> Options);

/// <summary>
/// The idrak commands for a host that runs them in its own process (an app's terminal): <see cref="Run"/> runs one command
/// line as the tool would, its output written as it comes, and <see cref="Commands"/> lists them from the tool's own table.
/// One command at a time: the tool's process-wide settings (the language, the threads of <c>--threads</c>) are a process's.
/// </summary>
public static class IdrakCommands
{
    /// <summary>Every command, in <c>idrak help</c>'s order (read from the tool's command table).</summary>
    public static IReadOnlyList<IdrakCommandInfo> Commands { get; } =
        [.. CommandTable.Groups.SelectMany(g => g.Commands.Select(c => new IdrakCommandInfo(c.Name, [.. c.Aliases], c.Summary, g.Title, c.Needs,
            [.. c.ValueOptions.Concat(c.Flags).Distinct(StringComparer.Ordinal)])))];

    /// <summary>The options every command takes ("--device", "--json", "--help"…).</summary>
    public static IReadOnlyList<string> CommonOptions { get; } = [.. CommandContext.CommonValueOptions.Concat(CommandContext.CommonFlags)];

    /// <summary>The help topics <c>idrak help TOPIC</c> shows (their names).</summary>
    public static IReadOnlyList<string> HelpTopics { get; } = [.. Shared.HelpTopics.All.Select(t => t.Name)];

    /// <summary>The exit code of a command its host does not allow (<see cref="CommandHost.Refusal"/>).</summary>
    public const int UnavailableExitCode = ExitCodes.Unavailable;

    /// <summary>The exit code of a command its host stopped (the cancellation given to <see cref="Run"/>).</summary>
    public const int StoppedExitCode = ExitCodes.Stopped;

    /// <summary>
    /// Runs <c>idrak ARGS</c> in this process on <paramref name="host"/> (<see cref="CommandHost.Console"/> when null): its
    /// output and errors written to <paramref name="output"/> and <paramref name="error"/> as the command writes them, a
    /// command that needs what the host does not allow refused with the reason (<see cref="UnavailableExitCode"/>), and
    /// <paramref name="cancellation"/> stopping it as Ctrl+C would (the command stops at its next check:
    /// <see cref="StoppedExitCode"/>). Returns the tool's exit code. Blocks: call it off the UI thread.
    /// </summary>
    public static int Run(IReadOnlyList<string> args, TextWriter output, TextWriter error, CommandHost? host = null, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        return CommandLine.Run(args, output, error, host ?? CommandHost.Console, cancellation);
    }

    /// <summary>
    /// Why <paramref name="command"/> (its words, or one of its aliases) cannot run on <paramref name="host"/>, or null when
    /// it can (or names no command).
    /// </summary>
    public static string? Refusal(string command, CommandHost host)
    {
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(host);
        return CommandLine.Find(CommandTable.All, command.Split(' ', StringSplitOptions.RemoveEmptyEntries), out _) is { } found
            ? host.Refusal(found.Name, found.Needs) : null;
    }

    /// <summary>
    /// Splits a typed command line into its words as a shell would: spaces separate, double or single quotes keep a word's
    /// spaces, a backslash before a quote or a backslash keeps it; a leading "idrak" is dropped.
    /// </summary>
    /// <exception cref="FormatException">A quote is not closed.</exception>
    public static IReadOnlyList<string> Split(string line)
    {
        ArgumentNullException.ThrowIfNull(line);
        var words = new List<string>();
        var word = new System.Text.StringBuilder();
        bool inWord = false;
        char quote = '\0';
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '\\' && i + 1 < line.Length && (line[i + 1] is '"' or '\'' or '\\') && quote != '\'')
            {
                word.Append(line[++i]);
                inWord = true;
            }
            else if (quote != '\0')
            {
                if (c == quote)
                {
                    quote = '\0';
                }
                else
                {
                    word.Append(c);
                }
            }
            else if (c is '"' or '\'')
            {
                quote = c;
                inWord = true;
            }
            else if (char.IsWhiteSpace(c))
            {
                if (inWord)
                {
                    words.Add(word.ToString());
                    word.Clear();
                    inWord = false;
                }
            }
            else
            {
                word.Append(c);
                inWord = true;
            }
        }

        if (quote != '\0')
        {
            throw new FormatException($"The quote {quote} is not closed.");
        }

        if (inWord)
        {
            words.Add(word.ToString());
        }

        if (words.Count > 0 && words[0] == "idrak")
        {
            words.RemoveAt(0);
        }

        return words;
    }
}
