// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Training;

// Plan 23, phase C7: pipeline parallelism. GPipe's schedule, a layout of stages and replicas over the ranks, and the
// pipeline settings read from the environment.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterPipelineGroup =
    [
        ("cluster: the gpipe schedule runs every forward, then the backward passes in reverse, with the bubble (S-1)/(M+S-1); stages that do not divide the ranks are refused by name, a layout keeps each stage's replicas together; the stage layers, the cuts and the placement read from the environment", ClusterPipelinePlan),
    ];

    internal static ClusterOptions PipeOptions(int? stages = null, int? microBatches = null) =>
        DpOptions() with { Placement = "pipeline", Stages = stages, MicroBatches = microBatches };

    internal static void ClusterPipelinePlan(Device device)
    {
        if (!FirstRun(nameof(ClusterPipelinePlan)))
        {
            return;
        }

        _ = device;
        var gpipe = new Idrak.Networking.Training.GPipeSchedule();
        Check(gpipe.Order(1, 2, 3).SequenceEqual(new PipelineWork[] { new(PipelineWorkKind.Forward, 0), new(PipelineWorkKind.Forward, 1), new(PipelineWorkKind.Forward, 2),
                  new(PipelineWorkKind.Backward, 2), new(PipelineWorkKind.Backward, 1), new(PipelineWorkKind.Backward, 0) })
              && Math.Abs(gpipe.IdealBubble(2, 4) - 0.2) < 1e-12 && gpipe.InFlight(0, 2, 4) == 4,
            $"gpipe: every forward, then the backward passes in reverse; bubble (S-1)/(M+S-1): {gpipe.Summary}");

        Check(ClusterOptions.From(n => n == "IDRAK_CLUSTER_STAGE_LAYERS" ? "2, 7" : null, null).PipelineStageLayers!.SequenceEqual([2, 7]), "the layers parse from the environment");
        ThrowsNaming<ArgumentException>(() => PipelineLayout.For(6, PipeOptions(stages: 4)), "stages that do not divide the ranks", "IDRAK_CLUSTER_STAGES", "1, 2, 3, 6");
        var layout = PipelineLayout.For(6, PipeOptions(stages: 3));
        Check(layout is { Replicas: 2 } && layout.StageOf(3) == 1 && layout.ReplicaOf(3) == 1 && layout.RankOf(2, 0) == 4 && layout.StageRanks(1).SequenceEqual([2, 3]),
            $"6 ranks in 3 stages: each stage's 2 replicas together, stage by stage: {layout}");
        Check(ClusterOptions.From(n => n == "IDRAK_CLUSTER_PIPELINE_CUTS" ? "3, 6" : null, null).PipelineCuts!.SequenceEqual([3, 6]), "the cuts parse from the environment");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_PLACEMENT" ? "tensor" : null, null), "an unknown placement", "IDRAK_CLUSTER_PLACEMENT", "single or pipeline");
    }
}
