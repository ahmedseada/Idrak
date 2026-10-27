// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Cli;
using Idrak.Maui;
using Idrak.Maui.Abstractions;

namespace Idrak.Mobile;

/// <summary>
/// The Terminal page's runner: the idrak commands (Idrak.Cli.Commands) run in the app's own process, on a host that allows
/// what the app's setting allows (<see cref="IdrakMauiOptions.TerminalAllow"/>: on a desktop other programs and a network
/// listener by default, on a phone nothing), read at each command and each listing, so a change in Settings applies when
/// the Terminal is next shown. Console input is never allowed (standard input is empty: the page is the shell), so what
/// needs it is refused with its reason; the progress lines are the console's (redrawn in place with a carriage return, which
/// the terminal's output draws as the tool's console does). One command at a time.
/// </summary>
public sealed class IdrakCommandRunner : ICommandRunner
{
    private readonly SemaphoreSlim _one = new(1, 1);
    private readonly CommandHost _host;
    private readonly Func<TerminalAllowance> _allows;
    private (CommandNeeds Allows, IReadOnlyList<TerminalCommand> Commands)? _listed;

    /// <summary>
    /// A runner on a host named <paramref name="hostName"/> allowing what <paramref name="allows"/> says at each use (null:
    /// <see cref="TerminalAllowances.Default"/>, all the platform can).
    /// </summary>
    public IdrakCommandRunner(string hostName = "the Idrak app", Func<TerminalAllowance>? allows = null, string progressReporter = "console")
    {
        _allows = allows ?? (() => TerminalAllowances.Default);
        _host = new CommandHost { Name = hostName, Allows = CommandNeeds.None, Input = TextReader.Null, ProgressReporter = progressReporter };
    }

    /// <inheritdoc />
    public string Name => "idrak";

    /// <summary>What the host allows now (the setting's), as the commands' needs.</summary>
    public CommandNeeds Allows => Needs(_allows());

    /// <inheritdoc />
    /// <remarks>The same list while what the host allows stays the same; a new one when it changes.</remarks>
    public IReadOnlyList<TerminalCommand> Commands
    {
        get
        {
            var allows = Allows;
            if (_listed is { } listed && listed.Allows == allows)
            {
                return listed.Commands;
            }

            var host = _host with { Allows = allows };
            IReadOnlyList<TerminalCommand> commands = [.. IdrakCommands.Commands.Select(c => new TerminalCommand(c.Name, c.Aliases, c.Summary, c.Group, host.Permits(c.Needs),
                host.Refusal(c.Name, c.Needs) ?? "", c.Options))];
            _listed = (allows, commands);
            return commands;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<string> CommonOptions => IdrakCommands.CommonOptions;

    /// <inheritdoc />
    public IReadOnlyList<string> HelpTopics => IdrakCommands.HelpTopics;

    /// <inheritdoc />
    public IReadOnlyList<string> Split(string line) => IdrakCommands.Split(line);

    /// <inheritdoc />
    /// <remarks>
    /// A server started here (<c>idrak serve</c>) runs until <paramref name="cancellationToken"/> stops it, as Ctrl+C would
    /// in the tool's console.
    /// </remarks>
    public async Task<CommandRunResult> RunAsync(IReadOnlyList<string> args, TextWriter output, TextWriter error, string workingFolder, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        if (!_one.Wait(0))
        {
            throw new InvalidOperationException("A command runs already: one at a time (the tool's settings are the process's).");
        }

        try
        {
            var host = _host with { WorkingFolder = workingFolder, Allows = Allows };
            long start = Stopwatch.GetTimestamp();
            int code = await Task.Factory.StartNew(() => IdrakCommands.Run(args, output, error, host, cancellationToken), CancellationToken.None,
                TaskCreationOptions.LongRunning, TaskScheduler.Default).ConfigureAwait(false);
            return new CommandRunResult(code, code == IdrakCommands.StoppedExitCode, code == IdrakCommands.UnavailableExitCode, Stopwatch.GetElapsedTime(start));
        }
        finally
        {
            _one.Release();
        }
    }

    // The app's allowance as the commands' needs (console input never: the page gives no console).
    private static CommandNeeds Needs(TerminalAllowance allowance) =>
        ((allowance & TerminalAllowance.ChildProcesses) != 0 ? CommandNeeds.ChildProcesses : CommandNeeds.None)
        | ((allowance & TerminalAllowance.NetworkListener) != 0 ? CommandNeeds.NetworkListener : CommandNeeds.None);
}
