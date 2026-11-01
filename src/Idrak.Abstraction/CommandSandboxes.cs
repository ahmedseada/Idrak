// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction;

/// <summary>The outcome of a command run in an <see cref="ICommandSandbox"/>.</summary>
/// <param name="ExitCode">The exit code, or null when the command was stopped (time or memory limit) or refused.</param>
/// <param name="TimedOut">Stopped at the time limit.</param>
/// <param name="Output">Standard output and error, interleaved and trimmed (for a refused command: why).</param>
/// <param name="Duration">How long it ran.</param>
/// <param name="Refused">Not started (not allowed, not installed, outside the workspace).</param>
public sealed record CommandResult(int? ExitCode, bool TimedOut, string Output, TimeSpan Duration, bool Refused)
{
    /// <summary>Ran and exited with 0.</summary>
    public bool Succeeded => ExitCode == 0;

    /// <summary>Stopped because it used more memory than <see cref="CommandLimits.MaxMemoryBytes"/> allows.</summary>
    public bool OutOfMemory { get; init; }

    /// <summary>Stopped by its caller (<see cref="ICommandSession.Stop"/>), with its process tree: no exit code.</summary>
    public bool Stopped { get; init; }
}

/// <summary>
/// What a command run in an <see cref="ICommandSandbox"/> may do and see. A sandbox enforces every limit given, or
/// refuses the command (<see cref="ICommandSandbox.Check"/>), naming a sandbox that can: it never runs a command under
/// weaker limits than asked.
/// </summary>
public sealed record CommandLimits
{
    /// <summary>The longest the command may run; it is stopped (its whole process tree) after it.</summary>
    public TimeSpan Time { get; init; } = TimeSpan.FromMinutes(5);

    /// <summary>Characters of output returned (the start and the end are kept, the middle is cut).</summary>
    public int MaxOutputCharacters { get; init; } = 12_000;

    /// <summary>The most memory the command may use, in bytes; 0 for no limit.</summary>
    public long MaxMemoryBytes { get; init; }

    /// <summary>The most processes the command may start; 0 for no limit.</summary>
    public int MaxProcesses { get; init; }

    /// <summary>The most CPUs the command may use (fractions allowed); 0 for no limit.</summary>
    public double MaxCpus { get; init; }

    /// <summary>Whether the command may reach the network (true: as a plain process does).</summary>
    public bool Network { get; init; } = true;

    /// <summary>The programs the command may start (file names without extension, ignoring case); null for any.</summary>
    public IReadOnlyList<string>? Programs { get; init; }

    /// <summary>Variables set for the command, over the ones it inherits; null for none.</summary>
    public IReadOnlyDictionary<string, string>? Environment { get; init; }

    /// <summary>
    /// The folder the command works in and may see (a workspace; a container mounts it), holding the command's folder;
    /// null: the command's folder itself.
    /// </summary>
    public string? Root { get; init; }

    /// <summary>
    /// Receives each line of the command's output (standard output and error, as the command writes them) while it runs,
    /// on the sandbox's reader thread: a terminal that shows the output live. The result's <see cref="CommandResult.Output"/>
    /// still holds the whole output, trimmed to <see cref="MaxOutputCharacters"/>; this sees every line, untrimmed. It must
    /// return quickly and not throw (an exception is dropped, the command goes on). Null: none.
    /// </summary>
    public Action<string>? Output { get; init; }
}

/// <summary>
/// Where an outside command runs, and under which limits (<see cref="CommandLimits"/>: time, output, memory, processes,
/// CPUs, network, the programs allowed), registered in <see cref="CommandSandboxes"/> under <see cref="Name"/>: any
/// package that runs a command a model asked for (the coding tools, a coding environment's builds and tests) runs it
/// through one. A sandbox that cannot enforce a limit it is asked for refuses the command, naming one that can; it never
/// runs it under weaker limits. Thread-safe: many commands run at once.
/// </summary>
public interface ICommandSandbox
{
    /// <summary>The sandbox's name in <see cref="CommandSandboxes"/> ("process", "container").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>
    /// Null when the sandbox can run commands under <paramref name="limits"/>; else why not (the limit it cannot enforce,
    /// or what it lacks, such as its program), naming a sandbox that can. Cheap: callers check once, before the first command.
    /// </summary>
    string? Check(CommandLimits limits);

    /// <summary>
    /// Runs <paramref name="command"/> (the program, then its arguments, as separate words: no shell) in
    /// <paramref name="directory"/> under <paramref name="limits"/>. A program that is not allowed or not found gives a
    /// refused result whose output says why; a command over its time or memory limit is stopped with its process tree.
    /// </summary>
    /// <param name="command">The program and its arguments, at least the program.</param>
    /// <param name="directory">The folder to run in (absolute), inside <see cref="CommandLimits.Root"/> when one is given.</param>
    /// <param name="limits">What the command may do.</param>
    /// <param name="cancellationToken">Stops the command (its process tree) and throws <see cref="OperationCanceledException"/>.</param>
    /// <exception cref="NotSupportedException">The sandbox cannot enforce <paramref name="limits"/> (<see cref="Check"/>'s reason).</exception>
    ValueTask<CommandResult> RunAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default);

    /// <summary>
    /// Null when the sandbox can start a command interactively under <paramref name="limits"/> (<see cref="StartAsync"/>:
    /// its standard input, output and error as streams the caller reads and writes); else why not, naming a sandbox that
    /// can. By default a sandbox cannot (it runs a command to its end, <see cref="RunAsync"/>), and says so.
    /// </summary>
    string? CheckStart(CommandLimits limits) =>
        $"The {Name} sandbox cannot start a command interactively (its standard input, output and error as streams); "
        + $"the {CommandSandboxes.Process} sandbox (Idrak.Text) can.";

    /// <summary>
    /// Starts <paramref name="command"/> (the program, then its arguments, as separate words: no shell) in
    /// <paramref name="directory"/> under <paramref name="limits"/> and hands its standard input, output and error over as
    /// streams (<see cref="ICommandSession"/>): a terminal tab that takes the keyboard, a language server, a script that
    /// asks a question. The time and memory limits stop it with its process tree as for <see cref="RunAsync"/>; the output
    /// cap and <see cref="CommandLimits.Output"/> do not apply (the caller reads the output). A program that is not allowed
    /// or not found gives a refused start whose reason says why. <paramref name="cancellationToken"/> cancels the start
    /// only (a question the sandbox asks first); once started, the command runs until it ends, its limit stops it, or
    /// <see cref="ICommandSession.Stop"/>. By default a sandbox cannot (<see cref="CheckStart"/>).
    /// </summary>
    /// <param name="command">The program and its arguments, at least the program.</param>
    /// <param name="directory">The folder to run in (absolute), inside <see cref="CommandLimits.Root"/> when one is given.</param>
    /// <param name="limits">What the command may do.</param>
    /// <param name="cancellationToken">Cancels the start (not the started command) and throws <see cref="OperationCanceledException"/>.</param>
    /// <exception cref="NotSupportedException">The sandbox cannot start a command interactively under <paramref name="limits"/> (<see cref="CheckStart"/>'s reason).</exception>
    ValueTask<CommandStart> StartAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default) =>
        ValueTask.FromException<CommandStart>(new NotSupportedException(CheckStart(limits) ?? $"The {Name} sandbox does not start commands interactively."));
}

/// <summary>
/// A command started interactively (<see cref="ICommandSandbox.StartAsync"/>): its standard input, output and error as
/// streams, and a handle that waits for its end and stops its process tree. The streams are the process's own pipes, so
/// nothing is kept between them and the caller: memory does not grow with what the command reads or writes. Disposing it
/// stops the command when it still runs and frees its handles. <see cref="Stop"/> and <c>DisposeAsync</c> may be called
/// from any thread.
/// </summary>
public interface ICommandSession : IAsyncDisposable
{
    /// <summary>The started process's id.</summary>
    int ProcessId { get; }

    /// <summary>
    /// The command's standard input: what is written is flushed to it at once; disposing the stream closes the input (the
    /// command reads its end).
    /// </summary>
    Stream Input { get; }

    /// <summary>The command's standard output, read as it writes it; a read gives 0 once it closed its output.</summary>
    Stream Output { get; }

    /// <summary>The command's standard error, read as it writes it; a read gives 0 once it closed it.</summary>
    Stream Error { get; }

    /// <summary>
    /// Completes when the command ended: its exit code, or none when its time or memory limit or <see cref="Stop"/> stopped
    /// it (<see cref="CommandResult.TimedOut"/>, <see cref="CommandResult.OutOfMemory"/>, <see cref="CommandResult.Stopped"/>).
    /// Its <see cref="CommandResult.Output"/> is empty: the output went through <see cref="Output"/> and <see cref="Error"/>.
    /// </summary>
    Task<CommandResult> Completion { get; }

    /// <summary>Stops the command with its whole process tree (nothing when it ended); <see cref="Completion"/> then completes.</summary>
    void Stop();
}

/// <summary>An interactive start's outcome (<see cref="ICommandSandbox.StartAsync"/>): the started command, or why it did not start.</summary>
public sealed record CommandStart
{
    private CommandStart(ICommandSession? session, string? refusal)
    {
        Session = session;
        Refusal = refusal;
    }

    /// <summary>The started command, or null when it was refused.</summary>
    public ICommandSession? Session { get; }

    /// <summary>Why it did not start (not allowed, not installed, the user said no), or null when it started.</summary>
    public string? Refusal { get; }

    /// <summary>Whether it started (<see cref="Session"/> is set).</summary>
    [System.Diagnostics.CodeAnalysis.MemberNotNullWhen(true, nameof(Session))]
    [System.Diagnostics.CodeAnalysis.MemberNotNullWhen(false, nameof(Refusal))]
    public bool Started => Session is not null;

    /// <summary>A started command.</summary>
    public static CommandStart Of(ICommandSession session) => new(session ?? throw new ArgumentNullException(nameof(session)), null);

    /// <summary>A refused start, saying why.</summary>
    public static CommandStart Refused(string why)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(why);
        return new(null, why);
    }
}

/// <summary>
/// The command sandboxes by name (ignoring case). Idrak.Text registers <see cref="Process"/> (a plain process with the
/// programs allowed, a time limit, an output cap and a memory limit; no isolation of the network; it also starts a command
/// interactively, <see cref="ICommandSandbox.StartAsync"/>), and the coding
/// plug-in of Idrak.Reinforcement.Plugins <see cref="Container"/> (Docker or Podman: the network off, memory, CPU and
/// process limits). A sandbox that is not registered is refused with "not registered"; nothing falls back to another.
/// Register another with <see cref="Register"/>; one under a library name takes the library's place, which
/// <see cref="Unregister"/> brings back.
/// </summary>
public static class CommandSandboxes
{
    /// <summary>The plain-process sandbox (Idrak.Text): the default.</summary>
    public const string Process = "process";

    /// <summary>The container sandbox (Docker or Podman; the coding plug-in of Idrak.Reinforcement.Plugins).</summary>
    public const string Container = "container";

    /// <summary>The variable that names the sandbox commands run in when an option does not: "IDRAK_SANDBOX".</summary>
    public const string EnvironmentVariable = "IDRAK_SANDBOX";

    private static readonly SlotTable<string, ICommandSandbox> Table = new(nameof(CommandSandboxes), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a sandbox is a safety boundary: falling back to another when one fails would run a model's command under other limits than the ones asked for");

    static CommandSandboxes() => LibraryDefaults.Ensure(typeof(CommandSandboxes));

    /// <summary>The sandbox named by <see cref="EnvironmentVariable"/>, or <see cref="Process"/> when it is not set.</summary>
    public static string DefaultName => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Process;

    /// <summary>Registers <paramref name="sandbox"/> under its <see cref="ICommandSandbox.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ICommandSandbox sandbox)
    {
        ArgumentNullException.ThrowIfNull(sandbox);
        ArgumentException.ThrowIfNullOrWhiteSpace(sandbox.Name);
        Table.Register(sandbox.Name, sandbox, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's sandbox <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The sandbox registered as <paramref name="name"/>, or null.</summary>
    public static ICommandSandbox? Find(string name) => Table.Find(name);

    /// <summary>The sandbox registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and where the others come from.</exception>
    public static ICommandSandbox Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(CommandSandboxes), name, Table.Keys,
            $"Command sandbox '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); "
            + $"add it with CommandSandboxes.Register; '{Process}' comes with Idrak.Text, '{Container}' with Idrak.Reinforcement.Plugins "
            + "(ReinforcementPlugins.RegisterAll()).");

    /// <summary>The library's sandbox <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ICommandSandbox? Default(string name) => Table.Default(name);

    /// <summary>Who registered the sandbox <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered sandbox as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));
}
