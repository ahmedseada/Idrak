// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// What the Terminal's commands may do in the app's process beyond the library: start other programs, open a network
/// listener (<see cref="IdrakMauiOptions.TerminalAllow"/>, the setting <c>idrak.terminal.allow</c>). A command that needs
/// more is refused by the runner with its reason. Console input is never among them: the Terminal is the shell itself.
/// </summary>
[Flags]
public enum TerminalAllowance
{
    /// <summary>Nothing beyond the library (a phone's).</summary>
    None = 0,

    /// <summary>Starting other programs: dotnet (<c>idrak test</c>), a browser, the ranks of a localhost cluster.</summary>
    ChildProcesses = 1,

    /// <summary>Opening a network listener: a server (<c>idrak serve</c>), a dashboard, a cluster's rendezvous.</summary>
    NetworkListener = 2,
}

/// <summary>The Terminal's allowances by platform, decided at run time (<see cref="IdrakPlatform.IsDesktop"/>), and their setting's names.</summary>
public static class TerminalAllowances
{
    /// <summary>Both allowances: what a desktop allows by default.</summary>
    public const TerminalAllowance Desktop = TerminalAllowance.ChildProcesses | TerminalAllowance.NetworkListener;

    /// <summary>
    /// What the app's platform can allow: <see cref="Desktop"/> on a desktop, none on a phone (Android starts no other
    /// program for an app, and keeps no server of an app running behind the screen).
    /// </summary>
    public static TerminalAllowance Possible => For(Desktop, IdrakPlatform.IsDesktop);

    /// <summary>The allowance when nothing is chosen: all a platform can allow (<see cref="Possible"/>).</summary>
    public static TerminalAllowance Default => Possible;

    /// <summary>
    /// What <paramref name="chosen"/> allows on a desktop (<paramref name="desktop"/>) or a phone: the choice (null: all the
    /// platform can), held to what the platform can (a phone: none).
    /// </summary>
    public static TerminalAllowance For(TerminalAllowance? chosen, bool desktop) => desktop ? (chosen ?? Desktop) & Desktop : TerminalAllowance.None;

    /// <summary>The setting's name of <paramref name="allowance"/>: "none", "programs", "listener" or "programs+listener".</summary>
    public static string Name(TerminalAllowance allowance) => (allowance & Desktop) switch
    {
        TerminalAllowance.ChildProcesses => "programs",
        TerminalAllowance.NetworkListener => "listener",
        Desktop => "programs+listener",
        _ => "none",
    };

    /// <summary>
    /// The allowance a setting's text names: "none", "programs", "listener", both joined by "+", "," or a space, or "all";
    /// "" or "default" is null (the platform's own).
    /// </summary>
    /// <exception cref="FormatException">A word is none of those (the message names what is taken).</exception>
    public static TerminalAllowance? Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Trim() is "" or "default")
        {
            return null;
        }

        var allowance = TerminalAllowance.None;
        foreach (string word in text.Split(['+', ',', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            allowance |= word.ToLowerInvariant() switch
            {
                "none" => TerminalAllowance.None,
                "programs" or "processes" => TerminalAllowance.ChildProcesses,
                "listener" or "server" => TerminalAllowance.NetworkListener,
                "all" => Desktop,
                _ => throw new FormatException($"'{text}' is not what the Terminal allows: default, none, programs, listener or programs+listener."),
            };
        }

        return allowance;
    }
}

/// <summary>One command a runner offers, as the terminal lists it: its words, short names, help, group, and whether it runs here (and why not).</summary>
/// <param name="Name">The words after the runner's name ("models list").</param>
/// <param name="Aliases">Its short names.</param>
/// <param name="Summary">One line of help.</param>
/// <param name="Group">Its group's title ("Models").</param>
/// <param name="IsAvailable">Whether it runs in this app.</param>
/// <param name="Unavailable">Why it does not (when <paramref name="IsAvailable"/> is false), or "".</param>
/// <param name="Options">Its own options ("--json"), without the common ones.</param>
public sealed record TerminalCommand(string Name, IReadOnlyList<string> Aliases, string Summary, string Group, bool IsAvailable, string Unavailable, IReadOnlyList<string> Options);

/// <summary>How one command line ended.</summary>
/// <param name="ExitCode">Its exit code (0: it worked).</param>
/// <param name="Stopped">Whether it was stopped (cancelled).</param>
/// <param name="Refused">Whether the app refused it (the reason is in its output).</param>
/// <param name="Elapsed">How long it ran.</param>
public sealed record CommandRunResult(int ExitCode, bool Stopped, bool Refused, TimeSpan Elapsed);

/// <summary>
/// Runs command lines in the app's own process for the Terminal page: the library's command-line tool as a library (the
/// app registers its implementation; Idrak.Mobile's runs the idrak commands). One command at a time: a runner refuses a
/// second while one runs. Output is written to the writers as the command writes it, from the command's thread; a progress
/// line is redrawn in place with a carriage return.
/// </summary>
public interface ICommandRunner
{
    /// <summary>The tool's name, typed before its commands ("idrak"; a line may start with it).</summary>
    string Name { get; }

    /// <summary>
    /// Every command, in its help's order, with whether it runs here: as the app allows now, so a new list when what it
    /// allows changes (<see cref="IdrakMauiOptions.TerminalAllow"/>); the terminal reads it again when shown.
    /// </summary>
    IReadOnlyList<TerminalCommand> Commands { get; }

    /// <summary>The options every command takes ("--json", "--help"…).</summary>
    IReadOnlyList<string> CommonOptions { get; }

    /// <summary>The topics of its help.</summary>
    IReadOnlyList<string> HelpTopics { get; }

    /// <summary>The words of a typed line, as a shell splits them (quotes kept together; a leading tool name dropped).</summary>
    /// <exception cref="FormatException">A quote is not closed.</exception>
    IReadOnlyList<string> Split(string line);

    /// <summary>
    /// Runs <paramref name="args"/> in <paramref name="workingFolder"/> on a worker thread, its output and errors written as
    /// they come; <paramref name="cancellationToken"/> stops it as Ctrl+C would. A command the app does not allow is refused
    /// with its reason written out.
    /// </summary>
    /// <exception cref="InvalidOperationException">A command runs already (one at a time).</exception>
    Task<CommandRunResult> RunAsync(IReadOnlyList<string> args, TextWriter output, TextWriter error, string workingFolder, CancellationToken cancellationToken = default);
}
