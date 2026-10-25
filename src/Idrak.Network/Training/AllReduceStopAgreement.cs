// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Training;

/// <summary>The "all-reduce" stop agreement (Plan 27, R7), registered in <see cref="StopAgreements"/>.</summary>
internal sealed class AllReduceStopAgreementProvider : IStopAgreementProvider
{
    public string Name => "all-reduce";

    public string Summary => "the ranks' stop flags summed in the gradient reduction's header (4 bytes a step), else in a one-value all-reduce at the step boundary";

    public IStopAgreement Create(ICollectives collectives, ClusterOptions options) => new AllReduceStopAgreement(collectives, options);
}

/// <summary>
/// The ranks agree on a stop by summing their flags (Plan 27, R7): a gradient reduction adds this rank's flag to its header
/// every step (<see cref="Carried"/>), so a boundary after such a step decides from that sum with no message of its own;
/// a boundary that follows no such collective (a trainer whose step reduces nothing every rank shares: a pipeline's stages,
/// local SGD's steps between averages) sums the flags in one all-reduce of one value, every
/// <see cref="ClusterOptions.StopEvery"/> boundaries. Every rank makes the same collectives between two boundaries, so every
/// rank decides from the same sum: the run stops at the same boundary on all of them.
/// </summary>
internal sealed class AllReduceStopAgreement : StopAgreement, ICarriedStopAgreement, IDisposable
{
    private readonly HostValues _values;
    private readonly int _every;
    private float? _carried;
    private long _boundaries;

    public AllReduceStopAgreement(ICollectives collectives, ClusterOptions options)
        : base(collectives.Rank, collectives.WorldSize, options.StopWait)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(options.StopEvery, 1, "ClusterOptions.StopEvery (IDRAK_CLUSTER_STOP_EVERY)");
        _values = new HostValues(collectives);
        _every = options.StopEvery;
    }

    public override string Name => "all-reduce";

    public float Contribution => Requested ? 1f : 0f;

    /// <summary>The boundaries decided by a one-value all-reduce of their own (the rest by a carried sum, or skipped).</summary>
    internal long Exchanges { get; private set; }

    public void Carried(float sum) => _carried = Math.Max(_carried ?? 0f, sum);

    public void Dispose() => _values.Dispose();

    protected override void Began()
    {
        _carried = null;
        _boundaries = 0;
    }

    protected override int Decide(long step)
    {
        long boundary = _boundaries++;
        if (_carried is { } carried)
        {
            _carried = null;
            return (int)carried;
        }

        if (WorldSize == 1 || boundary % _every != 0)
        {
            return WorldSize == 1 && Requested ? 1 : 0;
        }

        Span<float> flag = [Contribution];
        _values.AllReduce(flag, WireType.Float32);
        Exchanges++;
        return (int)flag[0];
    }
}
