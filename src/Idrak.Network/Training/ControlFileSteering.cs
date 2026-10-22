// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Networking.Training;

/// <summary>The "control-file" run steering (Plan 27, R11), registered in <see cref="RunSteerings"/>.</summary>
internal sealed class ControlFileSteeringProvider : IRunSteeringProvider
{
    public string Name => "control-file";

    public string Summary => "a launcher's control file (IDRAK_CLUSTER_CONTROL): rank 0 reads it at each trainer boundary and tells every rank in one broadcast; one process reads it alone";

    public IRunSteering? Create(ICollectives? collectives, ClusterOptions options) =>
        options.Simulation.Control is { } path ? new ControlFileSteering(path, collectives, options.Simulation.ControlPoll) : null;
}

/// <summary>
/// A run steered through a launcher's control file (<see cref="ClusterControlFile"/>, plan 23 C9) at its trainer's boundaries
/// (Plan 27, R11). On a cluster rank 0 reads the file at each boundary and broadcasts what it says (one value), so every rank
/// goes on, waits (rank 0 reading again every <see cref="SimulationOptions.ControlPoll"/> and broadcasting again, so no wait
/// reaches the collectives' timeout) or stops at this same boundary; rank 0 reports in the state file beside it ("paused N",
/// "running N", "stopped N"). The loop is the one a group of processes' gradient synchronisation runs as it begins a step
/// (<see cref="Hold"/>). In one process the file is read as it is, at most every <see cref="SimulationOptions.ControlPoll"/>
/// while the run goes (a boundary can come every few microseconds: an off-policy round), and at once at the first boundary.
/// </summary>
internal sealed class ControlFileSteering : IRunSteering
{
    private readonly string _path;
    private readonly HostValues? _values;
    private readonly int _rank;
    private readonly TimeSpan _poll;
    private long _read;

    public ControlFileSteering(string path, ICollectives? collectives, TimeSpan poll)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _path = Path.GetFullPath(path);
        _rank = collectives?.Rank ?? 0;
        _values = collectives is { WorldSize: > 1 } ? new HostValues(collectives) : null;
        _poll = poll;
    }

    public string Name => "control-file";

    public ClusterControlCommand Boundary(long step, CancellationToken cancellationToken = default)
    {
        if (_values is null)
        {
            // One process: the file read at the first boundary, then at most every poll (a pause holds here and reads on).
            long now = Stopwatch.GetTimestamp();
            if (_read != 0 && Stopwatch.GetElapsedTime(_read, now) < _poll)
            {
                return ClusterControlCommand.Run;
            }

            _read = now;
        }

        return Hold(_path, step, _rank, _values, _poll, cancellationToken);
    }

    /// <summary>
    /// The boundary before <paramref name="step"/> of a run steered through the control file at <paramref name="path"/>: rank
    /// 0 reads it and <paramref name="values"/> tells every rank (null: one rank, nothing to tell), so every rank goes on
    /// (<see cref="ClusterControlCommand.Run"/>), waits while it says pause, or stops (<see cref="ClusterControlCommand.Stop"/>)
    /// at this same boundary. While paused rank 0 reads it again every <paramref name="poll"/> and tells every rank again; its
    /// <paramref name="cancellationToken"/> cancelled while paused ends the wait as a stop on every rank. Rank 0 reports what
    /// the ranks do in the state file.
    /// </summary>
    internal static ClusterControlCommand Hold(string path, long step, int rank, HostValues? values, TimeSpan poll, CancellationToken cancellationToken)
    {
        Span<float> said = stackalloc float[1];
        bool paused = false;
        while (true)
        {
            if (rank == 0)
            {
                said[0] = (float)(paused && cancellationToken.IsCancellationRequested ? ClusterControlCommand.Stop : ClusterControlFile.Read(path));
            }

            values?.Broadcast(said, 0);
            switch ((ClusterControlCommand)(int)said[0])
            {
                case ClusterControlCommand.Run:
                    if (paused && rank == 0)
                    {
                        ClusterControlFile.Report(path, "running", step);
                    }

                    return ClusterControlCommand.Run;
                case ClusterControlCommand.Stop:
                    if (rank == 0)
                    {
                        ClusterControlFile.Report(path, "stopped", step);
                    }

                    return ClusterControlCommand.Stop;
                default:
                    if (rank == 0)
                    {
                        if (!paused)
                        {
                            ClusterControlFile.Report(path, "paused", step);
                        }

                        if (cancellationToken.CanBeCanceled)
                        {
                            cancellationToken.WaitHandle.WaitOne(poll);
                        }
                        else
                        {
                            Thread.Sleep(poll);
                        }
                    }

                    paused = true;
                    break;
            }
        }
    }
}
