// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;

namespace Idrak.Text.Generation;

/// <summary>
/// The <c>process</c> command sandbox (<see cref="CommandSandboxes.Process"/>, registered by Idrak.Text; the default): the
/// command runs as a plain process of this machine, started without a shell, in its folder, with the programs allowed
/// (<see cref="CommandLimits.Programs"/>, found on the PATH), standard input closed, the time limit and the memory limit
/// enforced by stopping the whole process tree, and the output trimmed to its cap. The memory limit is checked on the
/// started process's working set (<see cref="Process.WorkingSet64"/>, which .NET reports on every platform), every
/// <see cref="PollInterval"/>; the processes it starts in turn are not counted. It cannot turn the network off nor limit
/// processes or CPUs, so it refuses those limits, naming <see cref="CommandSandboxes.Container"/>. Each line of output
/// goes to <see cref="CommandLimits.Output"/> as it is written, when one is given.
/// </summary>
public sealed class ProcessCommandSandbox : ICommandSandbox
{
    /// <summary>How often the memory of a command with a memory limit is read (50 ms by default).</summary>
    public TimeSpan PollInterval { get; init; } = TimeSpan.FromMilliseconds(50);

    /// <inheritdoc />
    public string Name => CommandSandboxes.Process;

    /// <inheritdoc />
    public string Summary => "a plain process (no shell): the programs allowed, a time limit, an output cap and a memory limit on the started process; the network as the machine has it";

    /// <inheritdoc />
    public string? Check(CommandLimits limits)
    {
        ArgumentNullException.ThrowIfNull(limits);
        string? missing = !limits.Network ? "turn the network off"
            : limits.MaxProcesses > 0 ? "limit the processes a command starts"
            : limits.MaxCpus > 0 ? "limit the CPUs a command uses"
            : null;
        return missing is null ? null
            : $"The {CommandSandboxes.Process} sandbox cannot {missing}: run the commands in the {CommandSandboxes.Container} sandbox "
              + $"(CodingToolOptions.Sandbox = \"{CommandSandboxes.Container}\", {CommandSandboxes.EnvironmentVariable}={CommandSandboxes.Container}; Idrak.Reinforcement.Plugins), which can.";
    }

    /// <inheritdoc />
    public async ValueTask<CommandResult> RunAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(limits);
        if (Check(limits) is { } refusal)
        {
            throw new NotSupportedException(refusal);
        }

        static CommandResult Refuse(string why) => new(null, false, why, TimeSpan.Zero, true);
        if (command.Count == 0 || command[0].Length == 0)
        {
            return Refuse("Error: empty command.");
        }

        string program = Path.GetFileNameWithoutExtension(command[0]);
        if (limits.Programs is { } allowed && (!allowed.Contains(program, StringComparer.OrdinalIgnoreCase) || command[0].Contains('/') || command[0].Contains('\\')))
        {
            return Refuse($"Error: '{command[0]}' is not an allowed program. Allowed: {string.Join(", ", allowed)}.");
        }

        string? executable = limits.Programs is null && Path.IsPathRooted(command[0]) && File.Exists(command[0]) ? command[0] : FindProgram(program);
        if (executable is null)
        {
            return Refuse($"Error: '{program}' is not installed (not found on the PATH).");
        }

        var start = new ProcessStartInfo(executable)
        {
            WorkingDirectory = directory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        for (int i = 1; i < command.Count; i++)
        {
            start.ArgumentList.Add(command[i]);
        }

        foreach (var (name, value) in limits.Environment ?? new Dictionary<string, string>())
        {
            start.Environment[name] = value;
        }

        var output = new StringBuilder();
        var watch = Stopwatch.StartNew();
        using var process = new Process { StartInfo = start };
        var live = limits.Output;
        void Line(string? data)
        {
            if (data is null)
            {
                return;
            }

            lock (output)
            {
                output.Append(data).Append('\n');
            }

            if (live is not null)
            {
                try
                {
                    live(data);
                }
#pragma warning disable CA1031 // A watcher's failure never stops the command (CommandLimits.Output).
                catch (Exception)
#pragma warning restore CA1031
                {
                }
            }
        }

        process.OutputDataReceived += (_, e) => Line(e.Data);
        process.ErrorDataReceived += (_, e) => Line(e.Data);
        try
        {
            process.Start();
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            return Refuse($"Error: could not start '{program}': {ex.Message}");
        }

        process.StandardInput.Close();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        limit.CancelAfter(limits.Time);
        bool timedOut = false, outOfMemory = false;
        using var stopWatching = new CancellationTokenSource();
        var memory = limits.MaxMemoryBytes > 0 ? WatchMemoryAsync(process, limits.MaxMemoryBytes, limit, () => outOfMemory = true, stopWatching.Token) : Task.CompletedTask;
        try
        {
            await process.WaitForExitAsync(limit.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            timedOut = !cancellationToken.IsCancellationRequested && !outOfMemory;
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
            }

            if (!timedOut && !outOfMemory)
            {
                throw;
            }
        }
        finally
        {
            await stopWatching.CancelAsync().ConfigureAwait(false);
            await memory.ConfigureAwait(false);
        }

        process.WaitForExit();                                      // flushes the output events
        string text;
        lock (output)
        {
            text = Trim(output.ToString().TrimEnd(), limits.MaxOutputCharacters);
        }

        bool stopped = timedOut || outOfMemory;
        return new CommandResult(stopped ? null : process.ExitCode, timedOut, text, watch.Elapsed, false) { OutOfMemory = outOfMemory };
    }

    /// <summary>The program <paramref name="name"/> found on the PATH (with the Windows executable extensions: a program installed as a .cmd script is found there), or null.</summary>
    public static string? FindProgram(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        var extensions = OperatingSystem.IsWindows()
            ? (Environment.GetEnvironmentVariable("PATHEXT") ?? ".COM;.EXE;.BAT;.CMD").Split(';', StringSplitOptions.RemoveEmptyEntries)
            : [""];
        foreach (var folder in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (var extension in extensions)
            {
                string candidate = Path.Combine(folder.Trim('"'), name + extension.ToLowerInvariant());
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    // Reads the started process's working set every poll; over the limit, marks it and stops the wait (the caller kills the tree).
    private async Task WatchMemoryAsync(Process process, long limit, CancellationTokenSource stop, Action exceeded, CancellationToken done)
    {
        try
        {
            while (!done.IsCancellationRequested)
            {
                await Task.Delay(PollInterval, done).ConfigureAwait(false);
                long used;
                try
                {
                    process.Refresh();
                    if (process.HasExited)
                    {
                        return;
                    }

                    used = process.WorkingSet64;
                }
                catch (InvalidOperationException)
                {
                    return;                                         // exited between the checks
                }

                if (used > limit)
                {
                    exceeded();
                    await stop.CancelAsync().ConfigureAwait(false);
                    return;
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    // The start and the end of a long output, the middle cut (a third before, the rest after).
    private static string Trim(string text, int max)
    {
        if (text.Length <= max)
        {
            return text;
        }

        int head = max / 3, tail = max - head;
        return $"{text[..head]}\n… {text.Length - max} characters cut …\n{text[^tail..]}";
    }
}
