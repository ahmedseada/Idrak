// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>The "shared" stop agreement (Plan 27, R7), registered in <see cref="StopAgreements"/>.</summary>
internal sealed class SharedStopAgreementProvider : IStopAgreementProvider
{
    public string Name => "shared";

    public string Summary => "an in-process simulated cluster's ranks agree on a stop through the run's memory: no message, whatever ranks take part in a step";

    public IStopAgreement Create(ICollectives collectives, ClusterOptions options) => collectives is SimStepper stepper
        ? new SharedStopAgreement(stepper.World.Stops, stepper.RealRank, stepper.World.World, options.StopWait)
        : throw new NotSupportedException($"The shared stop agreement serves an in-process simulated cluster's ranks (the sim transport), not {collectives.Name} collectives over "
                                          + $"{collectives.Transport?.Name ?? "a transport of their own"}: use all-reduce (IDRAK_CLUSTER_STOP=all-reduce, or auto).");
}

/// <summary>
/// The ranks of one in-process simulated run agree on a stop through memory they share (Plan 27, R7), as the dashboard's
/// stop does (<see cref="ClusterRunControl.Stop"/>): a rank that asks sets the stop after the last boundary any rank has
/// passed, so every rank finishes the steps begun and leaves at the boundary after it. No message travels (the virtual
/// clock and the event log are those of a run without the agreement), and a rank that sits a step out (a late joiner, a
/// replay's end) agrees all the same, since every rank calls the boundary of every step.
/// </summary>
internal sealed class SharedStopAgreement(SharedStops stops, int rank, int worldSize, TimeSpan wait) : StopAgreement(rank, worldSize, wait)
{
    private volatile int _run;                                 // this rank's runs, counted alike on every rank
    private long _boundaries;

    public override string Name => "shared";

    protected override void Began()
    {
        _run++;
        _boundaries = 0;
    }

    protected override void Asked(string reason) => stops.Ask(_run);

    protected override int Decide(long step) => stops.Decide(_run, ++_boundaries);
}

/// <summary>
/// The boundaries of every rank of one simulated run, per run of a trainer on them (a rank that ends a stopped run and
/// begins the next does not clear what the others still use): the highest boundary any rank passed, and the boundary a stop
/// was set after.
/// </summary>
internal sealed class SharedStops
{
    private readonly Lock _gate = new();
    private readonly Dictionary<int, Board> _runs = [];

    // A rank of `run` asked: the stop goes after the last boundary any rank passed (once a run).
    public void Ask(int run)
    {
        lock (_gate)
        {
            var board = BoardOf(run);
            board.Askers++;
            board.After ??= board.Passed;
        }
    }

    // A rank reaches its `boundary`-th boundary of `run`: the ranks that asked once it is past the stop's boundary, else 0
    // (and the boundary counts as passed).
    public int Decide(int run, long boundary)
    {
        lock (_gate)
        {
            var board = BoardOf(run);
            if (board.After is { } after && boundary > after)
            {
                return board.Agreed ??= board.Askers;
            }

            board.Passed = Math.Max(board.Passed, boundary);
            return 0;
        }
    }

    private Board BoardOf(int run)
    {
        if (!_runs.TryGetValue(run, out var board))
        {
            _runs[run] = board = new Board();
        }

        return board;
    }

    private sealed class Board
    {
        public long Passed;
        public long? After;
        public int Askers;
        public int? Agreed;
    }
}
