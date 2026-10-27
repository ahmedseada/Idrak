// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Cli;
using Idrak.Maui.Abstractions;

namespace Idrak.Mobile;

/// <summary>
/// The Terminal page's runner: the idrak commands (Idrak.Cli.Commands) run in the app's own process, on a host that allows
/// none of what an app cannot do (no child process, no network listener, no console input), so those commands are refused
/// with their reason; standard input is empty, the progress lines are the console's (redrawn in place with a carriage
/// return, which the terminal's output draws as the tool's console does). One command at a time.
/// </summary>
public sealed class IdrakCommandRunner : ICommandRunner
{
    private readonly SemaphoreSlim _one = new(1, 1);
    private readonly CommandHost _host;

    /// <summary>A runner on a host named <paramref name="hostName"/> allowing <paramref name="allows"/> (nothing by default).</summary>
    public IdrakCommandRunner(string hostName = "the Idrak app", CommandNeeds allows = CommandNeeds.None, string progressReporter = "console")
    {
        _host = new CommandHost { Name = hostName, Allows = allows, Input = TextReader.Null, ProgressReporter = progressReporter };
        Commands = [.. IdrakCommands.Commands.Select(c => new TerminalCommand(c.Name, c.Aliases, c.Summary, c.Group, _host.Permits(c.Needs),
            _host.Refusal(c.Name, c.Needs) ?? "", c.Options))];
    }

    /// <inheritdoc />
    public string Name => "idrak";

    /// <inheritdoc />
    public IReadOnlyList<TerminalCommand> Commands { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> CommonOptions => IdrakCommands.CommonOptions;

    /// <inheritdoc />
    public IReadOnlyList<string> HelpTopics => IdrakCommands.HelpTopics;

    /// <inheritdoc />
    public IReadOnlyList<string> Split(string line) => IdrakCommands.Split(line);

    /// <inheritdoc />
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
            var host = _host with { WorkingFolder = workingFolder };
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
}
