// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Training;

/// <summary>
/// GPipe (plan 23, phase C7): every micro-batch's forward pass, then every backward pass in reverse. Each stage holds the
/// activations of every micro-batch at once and stands idle (stages - 1) / (micro-batches + stages - 1) of a step on stages
/// of equal cost: more micro-batches shrink the bubble, at the price of their activations. 1F1B (one forward, one backward)
/// leaves the same bubble and only bounds the activations held to the stages; it is not built (the measured bubble is
/// GPipe's arithmetic's, see the plan's C7 note).
/// </summary>
internal sealed class GPipeSchedule : IPipelineSchedule
{
    public string Name => "gpipe";

    public string Summary => "every micro-batch's forward pass, then every backward pass in reverse; bubble (stages - 1) / (micro-batches + stages - 1)";

    public IReadOnlyList<PipelineWork> Order(int stage, int stages, int microBatches)
    {
        Check(stage, stages, microBatches);
        var order = new PipelineWork[2 * microBatches];
        for (int m = 0; m < microBatches; m++)
        {
            order[m] = new PipelineWork(PipelineWorkKind.Forward, m);
            order[microBatches + m] = new PipelineWork(PipelineWorkKind.Backward, microBatches - 1 - m);
        }

        return order;
    }

    public double IdealBubble(int stages, int microBatches)
    {
        Check(0, stages, microBatches);
        return (stages - 1.0) / (microBatches + stages - 1.0);
    }

    public int InFlight(int stage, int stages, int microBatches)
    {
        Check(stage, stages, microBatches);
        return microBatches;
    }

    private static void Check(int stage, int stages, int microBatches)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stages);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(microBatches);
        ArgumentOutOfRangeException.ThrowIfNegative(stage);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(stage, stages);
    }
}
