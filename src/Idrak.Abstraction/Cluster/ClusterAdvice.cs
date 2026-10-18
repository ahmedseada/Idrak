// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// What a data-parallel run measured over its first warm steps (plan 23, decision 4): the subject an
/// <see cref="Diagnostics.ISettingAdvisor"/> is asked about (<see cref="Diagnostics.AdviceSituation.Subject"/>) when a
/// gradient synchronisation has timed its steps; the library's "cluster" advisor (Idrak.Network) suggests from it how to
/// send less (a bfloat16 wire, local SGD, gradient accumulation, the bucket size, top-k, the hierarchical collective).
/// Every time is the mean over the measured steps and the ranks; on a simulated transport the network's times are the
/// virtual clock's, the rest the wall's.
/// </summary>
/// <param name="Options">The run's cluster settings.</param>
/// <param name="Sync">The gradient synchronisation ("all-reduce", "sharded", …).</param>
/// <param name="Collective">The collective algorithm ("ring", "hierarchical", …).</param>
/// <param name="Compression">The buckets' compression ("none", "bf16", "top-k", …).</param>
/// <param name="WorldSize">The ranks of the step's group.</param>
/// <param name="Nodes">The topology's nodes (machines), or 1 when the collectives do not say.</param>
/// <param name="Steps">The warm steps measured (the first, cold step left out).</param>
/// <param name="Step">A step's time: its wall time with the network time it waited for counted as the cluster's (the virtual clock's on a simulated transport).</param>
/// <param name="Exposed">The network time a step waited for after its backward pass.</param>
/// <param name="Network">The network time of a step's gradient reduction, whether or not the backward pass hid it.</param>
/// <param name="Backward">A step's backward pass (zero when the trainer gave no overlap scope).</param>
/// <param name="GradientBytes">The gradient bytes a step reduces, before compression.</param>
/// <param name="BucketBytes">The bytes of a bucket.</param>
/// <param name="Buckets">The buckets a step reduces.</param>
/// <param name="Simulated">Whether the network's times are a simulated transport's virtual clock.</param>
/// <param name="BFloat16Error">
/// The precision a bfloat16 wire would lose: a sample slice of the step's gradients reduced over a float32 and over a
/// bfloat16 wire, the largest difference over the largest float32 value; null when it was not measured (the wire is
/// bfloat16 already, or the compression sends something else).
/// </param>
/// <param name="SliceValues">The values of the sample slice (0 when none was reduced).</param>
public sealed record ClusterStepMeasure(ClusterOptions Options, string Sync, string Collective, string Compression, int WorldSize, int Nodes, int Steps,
    TimeSpan Step, TimeSpan Exposed, TimeSpan Network, TimeSpan Backward, long GradientBytes, long BucketBytes, int Buckets, bool Simulated,
    double? BFloat16Error, int SliceValues)
{
    /// <summary>The share of a step spent waiting on the network (<see cref="Exposed"/> over <see cref="Step"/>), from 0 to 1.</summary>
    public double Share => Step > TimeSpan.Zero ? Math.Clamp(Exposed / Step, 0, 1) : 0;

    /// <summary>A step's time apart from the network it waited for (<see cref="Step"/> less <see cref="Exposed"/>).</summary>
    public TimeSpan Compute => Step > Exposed ? Step - Exposed : TimeSpan.Zero;
}
