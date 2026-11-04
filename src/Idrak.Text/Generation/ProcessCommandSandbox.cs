// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;

namespace Idrak.Text.Generation;

/// <summary>
/// The <c>process</c> command sandbox (<see cref="CommandSandboxes.Process"/>, registered by Idrak.Text; the default): the
/// command runs as a plain process of this machine, with the user's rights and nothing isolating the files, network or
/// processes it reaches (for models you trust; <see cref="CommandSandboxes.Container"/> isolates them), started without a
/// shell, in its folder, with the programs allowed (<see cref="CommandLimits.Programs"/>, found on the PATH: this
/// sandbox's check is the one the coding tools rely on, they keep no copy of it), standard input closed, the time limit and the memory limit
/// enforced by stopping the whole process tree, and the output trimmed to its cap. The memory limit is checked on the
/// started process's working set (<see cref="Process.WorkingSet64"/>, which .NET reports on every platform), every
/// <see cref="PollInterval"/>; the processes it starts in turn are not counted. It cannot turn the network off nor limit
/// processes or CPUs, so it refuses those limits, naming <see cref="CommandSandboxes.Container"/>. Each line of output
/// goes to <see cref="CommandLimits.Output"/> as it is written, when one is given. <see cref="StartAsync"/> starts a command
/// interactively the same way (the programs allowed, the variables, the time and memory limits), its standard input kept
/// open and its three standard streams handed to the caller as the process's own pipes (.NET's <see cref="Process"/> on
/// every platform; nothing buffered between them and the caller).
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
        if (Prepare(command, directory, limits, out string program) is not { } start)
        {
            return Refuse(program);
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

    /// <inheritdoc />
    public string? CheckStart(CommandLimits limits) => Check(limits);

    /// <inheritdoc />
    public ValueTask<CommandStart> StartAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(limits);
        if (Check(limits) is { } refusal)
        {
            throw new NotSupportedException(refusal);
        }

        cancellationToken.ThrowIfCancellationRequested();
        if (Prepare(command, directory, limits, out string program) is not { } start)
        {
            return ValueTask.FromResult(CommandStart.Refused(program));
        }

        var process = new Process { StartInfo = start };
        try
        {
            process.Start();
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            process.Dispose();
            return ValueTask.FromResult(CommandStart.Refused($"Error: could not start '{program}': {ex.Message}"));
        }

        return ValueTask.FromResult(CommandStart.Of(new ProcessCommandSession(this, process, limits)));
    }

    // The start of a command: the program checked against the ones allowed and found, the standard streams redirected, the
    // variables set. Null with the reason in "program" when it is refused; else the program's name in it.
    private static ProcessStartInfo? Prepare(IReadOnlyList<string> command, string directory, CommandLimits limits, out string program)
    {
        if (command.Count == 0 || command[0].Length == 0)
        {
            program = "Error: empty command.";
            return null;
        }

        program = Path.GetFileNameWithoutExtension(command[0]);
        if (limits.Programs is { } allowed && (!allowed.Contains(program, StringComparer.OrdinalIgnoreCase) || command[0].Contains('/') || command[0].Contains('\\')))
        {
            program = $"Error: '{command[0]}' is not an allowed program. Allowed: {string.Join(", ", allowed)}.";
            return null;
        }

        string? executable = limits.Programs is null && Path.IsPathRooted(command[0]) && File.Exists(command[0]) ? command[0] : FindProgram(program);
        if (executable is null)
        {
            program = $"Error: '{program}' is not installed (not found on the PATH).";
            return null;
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

        if (limits.Environment is { } environment)
        {
            foreach (var (name, value) in environment)
            {
                start.Environment[name] = value;
            }
        }

        return start;
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
    internal async Task WatchMemoryAsync(Process process, long limit, CancellationTokenSource stop, Action exceeded, CancellationToken done)
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

/// <summary>
/// The process sandbox's interactive command (<see cref="ProcessCommandSandbox.StartAsync"/>): the process's own pipes as
/// its streams (the input flushed on every write), its time and memory limits watched, <see cref="Stop"/> ending its tree.
/// </summary>
internal sealed class ProcessCommandSession : ICommandSession
{
    private readonly Process _process;
    private readonly CancellationTokenSource _stop = new();
    private int _stopped;
    private int _disposed;

    public ProcessCommandSession(ProcessCommandSandbox sandbox, Process process, CommandLimits limits)
    {
        _process = process;
        ProcessId = process.Id;
        Input = new InputStream(process.StandardInput.BaseStream);
        Output = process.StandardOutput.BaseStream;
        Error = process.StandardError.BaseStream;
        Completion = WatchAsync(sandbox, limits);
    }

    public int ProcessId { get; }

    public Stream Input { get; }

    public Stream Output { get; }

    public Stream Error { get; }

    public Task<CommandResult> Completion { get; }

    public void Stop()
    {
        if (Volatile.Read(ref _disposed) != 0 || Completion.IsCompleted)
        {
            return;
        }

        Volatile.Write(ref _stopped, 1);
        try
        {
            _stop.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // Disposed meanwhile: the command was stopped by the dispose.
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (!Completion.IsCompleted)
        {
            Volatile.Write(ref _stopped, 1);
            await _stop.CancelAsync().ConfigureAwait(false);
        }

        await Completion.ConfigureAwait(false);
        try
        {
            await Input.DisposeAsync().ConfigureAwait(false);
        }
        catch (IOException)
        {
            // The command closed its end first: nothing is left to close.
        }

        _process.Dispose();
        _stop.Dispose();
    }

    // Waits for the end, stopping the tree at the time limit, the memory limit or Stop.
    private async Task<CommandResult> WatchAsync(ProcessCommandSandbox sandbox, CommandLimits limits)
    {
        var watch = Stopwatch.StartNew();
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);
        limit.CancelAfter(limits.Time);
        bool timedOut = false, outOfMemory = false;
        using var stopWatching = new CancellationTokenSource();
        var memory = limits.MaxMemoryBytes > 0 ? sandbox.WatchMemoryAsync(_process, limits.MaxMemoryBytes, limit, () => outOfMemory = true, stopWatching.Token) : Task.CompletedTask;
        try
        {
            await _process.WaitForExitAsync(limit.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            timedOut = Volatile.Read(ref _stopped) == 0 && !outOfMemory;
            try
            {
                _process.Kill(entireProcessTree: true);
            }
            catch (Exception error) when (error is InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                // It ended meanwhile, or a child of it is not ours to end.
            }

            await _process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            await stopWatching.CancelAsync().ConfigureAwait(false);
            await memory.ConfigureAwait(false);
        }

        bool stopped = Volatile.Read(ref _stopped) != 0 && !timedOut && !outOfMemory;
        return new CommandResult(timedOut || outOfMemory || stopped ? null : _process.ExitCode, timedOut, "", watch.Elapsed, false) { OutOfMemory = outOfMemory, Stopped = stopped };
    }

    // The input pipe, flushed on every write (a FileStream over a pipe buffers on some platforms), so what is typed arrives.
    private sealed class InputStream(Stream inner) : Stream
    {
        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => inner.CanWrite;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override void Flush() => inner.Flush();

        public override Task FlushAsync(CancellationToken cancellationToken) => inner.FlushAsync(cancellationToken);

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            inner.Write(buffer);
            inner.Flush();
        }

        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await inner.WriteAsync(buffer, cancellationToken).ConfigureAwait(false);
            await inner.FlushAsync(cancellationToken).ConfigureAwait(false);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                inner.Dispose();
            }

            base.Dispose(disposing);
        }

        public override async ValueTask DisposeAsync()
        {
            await inner.DisposeAsync().ConfigureAwait(false);
            await base.DisposeAsync().ConfigureAwait(false);
        }
    }
}
