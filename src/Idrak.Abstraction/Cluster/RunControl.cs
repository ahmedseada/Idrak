// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Thrown on every rank of a cluster run as it begins the step after the one a stop was asked at (plan 23, phase C9: the
/// dashboard's "stop and checkpoint", or a launcher's control file): every rank stops at the same step boundary, and the
/// trainer writes the checkpoint of the last finished step before it lets this go on. A cancellation, so code that ends
/// on a cancelled token ends on it alike.
/// </summary>
public sealed class ClusterRunStoppedException : OperationCanceledException
{
    /// <summary>A stop at no particular step.</summary>
    public ClusterRunStoppedException()
        : this(-1, "the run was stopped as asked")
    {
    }

    /// <summary>A stop described by <paramref name="message"/>.</summary>
    public ClusterRunStoppedException(string message)
        : this(-1, message)
    {
    }

    /// <summary>A stop described by <paramref name="message"/>, caused by <paramref name="inner"/>.</summary>
    public ClusterRunStoppedException(string message, Exception inner)
        : base(message, inner) => Step = -1;

    /// <summary>A stop as rank begins <paramref name="step"/>, which no rank trains.</summary>
    /// <param name="step">The step the ranks did not begin: the run finished the steps before it.</param>
    /// <param name="message">Why, and who asked.</param>
    public ClusterRunStoppedException(long step, string message)
        : base(message) => Step = step;

    /// <summary>The step the ranks did not begin (the run finished the steps before it), or -1 when unknown.</summary>
    public long Step { get; }

    /// <summary>
    /// Whether the ranks agreed on the stop one of them asked for (Plan 27, R7: a cancelled token, a timeout, a stop request;
    /// <see cref="IStopAgreement"/>), rather than a launcher's control or the dashboard. A command ends such a run as it ends
    /// one stopped by Ctrl+C in one process.
    /// </summary>
    public bool Agreed { get; init; }
}

/// <summary>What a launcher tells the ranks of a run through its control file (<see cref="ClusterControlFile"/>).</summary>
public enum ClusterControlCommand
{
    /// <summary>Train on (the file's absence means this too).</summary>
    Run = 0,

    /// <summary>Wait at the next step boundary until told to run or stop.</summary>
    Pause = 1,

    /// <summary>Stop at the next step boundary, the last finished step's checkpoint written.</summary>
    Stop = 2,
}

/// <summary>
/// The control channel between a launcher and the ranks it started as processes of their own (plan 23, phase C9): a small
/// text file the launcher writes (one word: run, pause or stop; replaced whole, so a reader never sees half of it) and the
/// ranks' rank 0 reads as it begins each step (<see cref="SimulationOptions.Control"/>, <c>IDRAK_CLUSTER_CONTROL</c>), then
/// tells every rank in one small broadcast, so every rank pauses or stops at the same step boundary. Rank 0 reports what
/// the ranks did in a file beside it (<see cref="StatePath"/>: "paused 12", "running 12", "stopped 12"). Faults are not
/// sent this way: they belong to the simulated transport.
/// </summary>
public static class ClusterControlFile
{
    /// <summary>The file rank 0 reports the ranks' state in, beside the control file.</summary>
    public static string StatePath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return path + ".state";
    }

    /// <summary>Writes <paramref name="command"/> to the control file at <paramref name="path"/> (replaced whole).</summary>
    public static void Write(string path, ClusterControlCommand command) => Replace(path, Word(command));

    /// <summary>The command in the control file at <paramref name="path"/>: <see cref="ClusterControlCommand.Run"/> when it is missing or empty.</summary>
    /// <exception cref="InvalidDataException">The file holds something else; the message names it.</exception>
    public static ClusterControlCommand Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string text;
        try
        {
            text = Shared(() => ReadShared(path)).Trim();
        }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException)
        {
            return ClusterControlCommand.Run;
        }

        return text.ToLowerInvariant() switch
        {
            "" or "run" or "resume" => ClusterControlCommand.Run,
            "pause" => ClusterControlCommand.Pause,
            "stop" => ClusterControlCommand.Stop,
            _ => throw new InvalidDataException($"The cluster control file {path} holds '{text}', not run, pause or stop."),
        };
    }

    /// <summary>Rank 0 reports that the ranks are <paramref name="state"/> ("paused", "running", "stopped") at <paramref name="step"/>.</summary>
    public static void Report(string path, string state, long step)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(state);
        Replace(StatePath(path), string.Create(CultureInfo.InvariantCulture, $"{state} {step}"));
    }

    /// <summary>What rank 0 last reported, or null before any report.</summary>
    public static (string State, long Step)? ReadState(string path)
    {
        try
        {
            string[] parts = Shared(() => ReadShared(StatePath(path))).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            return parts.Length == 2 && long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out long step) ? (parts[0], step) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;                                       // not written yet (or being replaced): no report
        }
    }

    private static string Word(ClusterControlCommand command) => command switch
    {
        ClusterControlCommand.Pause => "pause",
        ClusterControlCommand.Stop => "stop",
        _ => "run",
    };

    // A whole new file moved over the old one: a reader sees the old text or the new, never a part.
    private static void Replace(string path, string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        string temporary = full + "." + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture) + ".tmp";
        File.WriteAllText(temporary, text + "\n");
        try
        {
            Shared(() =>
            {
                File.Move(temporary, full, overwrite: true);
                return true;
            });
        }
        catch
        {
            File.Delete(temporary);
            throw;
        }
    }

    // The file read while a writer may replace it: shared for reading, writing and deleting, so the reader holds up no
    // replacement where the system allows it.
    private static string ReadShared(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    // Windows refuses to open a file while another process replaces it, and to replace one a reader has open (a sharing
    // violation: an IOException or UnauthorizedAccessException); both last a moment, so the call is tried again, waiting
    // a millisecond longer each time (about a second in all). A missing file is an answer, not a conflict.
    private static T Shared<T>(Func<T> access)
    {
        const int attempts = 45;
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                return access();
            }
            catch (Exception e) when (attempt < attempts
                                      && e is UnauthorizedAccessException or IOException and not (FileNotFoundException or DirectoryNotFoundException))
            {
                Thread.Sleep(attempt);
            }
        }
    }
}

/// <summary>
/// A trainer's control point at its own boundaries (Plan 27, R11): the dashboard's pause, resume and stop (or a launcher's
/// control file) taken where the trainer's unit of work ends, an iteration of a reinforcement run, a round of an off-policy
/// one, instead of at each gradient step its synchronisation begins (<see cref="IStepGroup.BeginStep"/>, the control file
/// read as <c>IGradientSync.BeginStep</c> begins a step), where a stop would cut an iteration's updates in two. Every rank
/// calls <see cref="Boundary"/> at each boundary, at the same point of its run: it returns at once while the run goes on,
/// holds every rank at this same boundary while the run is paused (no rank inside a collective), and says
/// <see cref="ClusterControlCommand.Stop"/> on every rank alike once a stop was asked. <see cref="RunSteerings.For"/> gives a
/// run's steering; <see cref="RunSteerings.Stops"/> takes a stop through the ranks' agreement (<see cref="IStopAgreement"/>).
/// </summary>
public interface IRunSteering
{
    /// <summary>The way the run is steered ("in-process", "control-file").</summary>
    string Name { get; }

    /// <summary>
    /// The boundary before <paramref name="step"/> (every rank calls it at each boundary, at the same point: a collective
    /// point): waits while the run is paused, then answers <see cref="ClusterControlCommand.Run"/> to go on or
    /// <see cref="ClusterControlCommand.Stop"/>, the same answer on every rank (never <see cref="ClusterControlCommand.Pause"/>).
    /// </summary>
    /// <param name="step">The step (an iteration, a round) about to begin: the boundaries go forward.</param>
    /// <param name="cancellationToken">A cancellation while paused ends the wait as a stop where the steering can tell every rank so (a control file's rank 0); else it is the trainer's to see.</param>
    ClusterControlCommand Boundary(long step, CancellationToken cancellationToken = default);
}

/// <summary>One way of steering a run at its trainer's boundaries, registered by name in <see cref="RunSteerings"/>.</summary>
public interface IRunSteeringProvider
{
    /// <summary>The name ("in-process").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The steering of a run over <paramref name="collectives"/> (null: a run in one process) with <paramref name="options"/>, or null when this way does not steer it.</summary>
    IRunSteering? Create(ICollectives? collectives, ClusterOptions options);
}

/// <summary>
/// The ways a run is steered at its trainer's boundaries (Plan 27, R11), by name, asked in registration order:
/// Idrak.Cluster registers "in-process" (a simulated run's ranks, held and stopped through the run's
/// <c>ClusterRunControl</c>, the dashboard's in-process control), Idrak.Network "control-file" (a launcher's control file,
/// <see cref="SimulationOptions.Control"/>: read by rank 0 at each boundary and told to every rank in one broadcast of one
/// value; in one process read as it is, at most every <see cref="SimulationOptions.ControlPoll"/>).
/// </summary>
public static class RunSteerings
{
    /// <summary>Why a trainer asks the ranks to stop when its steering says so.</summary>
    public const string StopReason = "a stop asked through the run's control (the dashboard, or its control file)";

    private static readonly SlotTable<string, IRunSteeringProvider> Table = new(nameof(RunSteerings), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a steering holds every rank at the same boundary, so another cannot take over a run");

    // The steerings handed out per rank's collectives: their step gates let every step through (Steered).
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<ICollectives, IRunSteering> Taken = new();

    static RunSteerings() => LibraryDefaults.Ensure(typeof(RunSteerings));

    /// <summary>The registered ways, in registration order (the order <see cref="For"/> asks them in).</summary>
    public static IReadOnlyList<IRunSteeringProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRunSteeringProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's way of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The way of that name, or null.</summary>
    public static IRunSteeringProvider? Find(string name) => Table.Find(name);

    /// <summary>The way of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IRunSteeringProvider Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(RunSteerings), name, Table.Keys,
            $"Run steering '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with RunSteerings.Register, or reference Idrak.Cluster or Idrak.Network.");

    /// <summary>The library's way of that name, or null.</summary>
    public static IRunSteeringProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the way of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each way as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));

    /// <summary>
    /// The steering of a trainer that pauses and stops at its own boundaries, over <paramref name="collectives"/> (null: a run
    /// in one process) with <paramref name="options"/> (else the collectives' own, else <see cref="ClusterOptions.FromEnvironment"/>):
    /// the first registered way that steers the run, or null when nothing steers it (no run control, no control file). The
    /// trainer takes the run's control over: from then on these collectives' step gates let every step through
    /// (<see cref="Steered"/>), so a run is held and stopped at the trainer's boundaries alone. One steering per rank's
    /// collectives: a later call gives the same one.
    /// </summary>
    /// <exception cref="ArgumentException">A cluster variable that does not parse (when the options come from the environment).</exception>
    public static IRunSteering? For(ICollectives? collectives, ClusterOptions? options = null)
    {
        if (collectives is not null && Taken.TryGetValue(collectives, out var taken))
        {
            return taken;
        }

        var o = options ?? collectives?.Context?.Options ?? ClusterOptions.FromEnvironment();
        foreach (var provider in Table.Values)
        {
            if (provider.Create(collectives, o) is { } steering)
            {
                return collectives is null ? steering : Taken.GetValue(collectives, _ => steering);
            }
        }

        return null;
    }

    /// <summary>
    /// Whether a trainer steers the run over <paramref name="collectives"/> at its own boundaries (<see cref="For"/> gave it
    /// a steering): their step gates (a simulated rank's <see cref="IStepGroup.BeginStep"/>, the control file read as a
    /// group of processes begins a gradient step) then let every step through.
    /// </summary>
    public static bool Steered(ICollectives collectives)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        return Taken.TryGetValue(collectives, out _);
    }

    /// <summary>
    /// A trainer's boundary before <paramref name="step"/>: <paramref name="steering"/>'s <see cref="IRunSteering.Boundary"/>
    /// (held there while the run is paused), then a stop it says taken as the run's stop. In one process (no
    /// <paramref name="agreement"/>) true: the trainer stops here, as on a cancelled token. On a cluster every rank asks the
    /// ranks' agreement at this same boundary (<see cref="StopReason"/>) and, every rank having asked, hands it that sum
    /// (<see cref="ICarriedStopAgreement.Carried"/> when it is carried), so the trainer's <see cref="IStopAgreement.Agree"/> right after stops every
    /// rank here; false (the agreement says it). False when nothing steers the run.
    /// </summary>
    public static bool Stops(this IRunSteering? steering, long step, IStopAgreement? agreement, CancellationToken cancellationToken = default)
    {
        if (steering is null || steering.Boundary(step, cancellationToken) != ClusterControlCommand.Stop)
        {
            return false;
        }

        if (agreement is null)
        {
            return true;
        }

        agreement.Request(StopReason);
        if (agreement is ICarriedStopAgreement carried)
        {
            carried.Carried(agreement.WorldSize);              // every rank asked at this boundary: the sum is known with no message
        }

        return false;
    }
}
