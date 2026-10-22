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
            text = File.ReadAllText(path).Trim();
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
            string[] parts = File.ReadAllText(StatePath(path)).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            return parts.Length == 2 && long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out long step) ? (parts[0], step) : null;
        }
        catch (IOException)
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
        File.Move(temporary, full, overwrite: true);
    }
}
