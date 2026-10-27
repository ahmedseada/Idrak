// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

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

    /// <summary>Every command, in its help's order, with whether it runs here.</summary>
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
