// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;

namespace Idrak.Text;

/// <summary>
/// A rung of the tuner's activation ladder: what the training step keeps between its forward and backward passes. The
/// ladder runs held → recomputed feed-forward → recomputed feed-forward and bfloat16 → checkpointing, each lighter on
/// device memory than the one before.
/// </summary>
internal readonly record struct ActivationRung(bool Recompute, bool BFloat16, bool Checkpointing)
{
    /// <summary>The rung <paramref name="options"/> train with.</summary>
    public static ActivationRung Of(FineTuningOptions options) =>
        new(options.RecomputeFeedForward == true, options.BFloat16Activations == true, options.Checkpointing == true);

    /// <summary>How far down the ladder (0: held, 3: checkpointing).</summary>
    public int Depth => Checkpointing ? 3 : Recompute && BFloat16 ? 2 : Recompute || BFloat16 ? 1 : 0;

    /// <summary>The rung as a log line says it.</summary>
    public override string ToString() => Checkpointing ? "checkpointing"
        : Recompute && BFloat16 ? "recomputed feed-forward and bfloat16 activations"
        : Recompute ? "recomputed feed-forward activations"
        : BFloat16 ? "bfloat16 activations" : "activations held";

    /// <summary>The rung as the stop state writes it.</summary>
    public JsonObject ToJson() => new() { ["recompute"] = Recompute, ["bfloat16"] = BFloat16, ["checkpointing"] = Checkpointing };

    /// <summary>The rung a stop state wrote, or null when it wrote none (a checkpoint from before rungs were kept).</summary>
    public static ActivationRung? FromJson(JsonNode? node) => node is JsonObject json
        ? new ActivationRung((bool?)json["recompute"] ?? false, (bool?)json["bfloat16"] ?? false, (bool?)json["checkpointing"] ?? false)
        : null;
}

/// <summary>
/// The device memory a training step of a decoder needs at its peak on each rung of the activation ladder, from the
/// model's own dimensions (<see cref="DecoderSpec"/>) and the batch's rows × positions: what each layer keeps between
/// the passes (the normalized inputs of the projections, queries, keys, values, the attention's output before and after
/// its heads are merged, the feed-forward gate, up and activation; float32, or bfloat16 where the rung compresses them,
/// the activation dropped where it is recomputed; each norm's normalized values, float32 on every rung; only each
/// block's input with checkpointing, plus one block run again),
/// one block's gradients while the backward pass runs through it, the residual stream and its gradient, the loss's
/// chunk of logits and their gradient, the score matrices where the device composes attention instead of tiling it, and
/// the gradients of trained parameters that have none yet. Card-agnostic: the room it is compared with is what the
/// device reports at run time. An estimate, not a bound: the out-of-memory ladder stays behind it.
/// </summary>
internal static class ActivationEstimate
{
    /// <summary>The variable that turns the estimate off (0 or false: start from the options as given, as before).</summary>
    public const string EnvironmentVariable = "IDRAK_ACTIVATION_ESTIMATE";

    /// <summary>Whether <see cref="EnvironmentVariable"/> leaves the estimate on (unset: on).</summary>
    public static bool FromEnvironment() =>
        Environment.GetEnvironmentVariable(EnvironmentVariable)?.Trim() is not { } value || value != "0" && !value.Equals("false", StringComparison.OrdinalIgnoreCase);

    /// <summary>The peak bytes of a step over <paramref name="rows"/> × <paramref name="length"/> positions on <paramref name="rung"/>.</summary>
    /// <param name="spec">The decoder's dimensions.</param>
    /// <param name="capabilities">The device's (whether its attention is tiled at the model's head size).</param>
    /// <param name="rows">The batch's rows.</param>
    /// <param name="length">The batch's positions per row.</param>
    /// <param name="rung">What the step keeps between the passes.</param>
    /// <param name="lossChunkRows">The rows of the output head the loss computes at once (<see cref="FineTuningOptions.LossChunkRows"/>).</param>
    /// <param name="pendingGradients">The elements of trained parameters that have no gradient yet (the first step makes them).</param>
    public static long Bytes(DecoderSpec spec, BackendCapabilities capabilities, int rows, int length, ActivationRung rung, int lossChunkRows, long pendingGradients = 0)
    {
        long positions = (long)rows * length;
        long d = spec.Dim, queries = (long)spec.Heads * spec.HeadDim, keys = (long)spec.KvHeads * spec.HeadDim;
        long ff = spec.Experts > 0 ? (long)spec.ExpertHidden * Math.Max(1, spec.ExpertsPerToken) + spec.SharedExpertFfDim : spec.FfDim;
        long projected = ff * (spec.Gated ? 2 : 1);                         // gate and up (up alone without a gate)

        // Per position of one layer, in elements: the two norms' outputs, queries, keys, values, the attention's output
        // before and after merging its heads, the feed-forward's inputs and its activation; the norms' and attention's
        // statistics stay float32.
        long compressible = 2 * d + 3 * queries + 2 * keys + projected;
        long statistics = 2 + spec.Heads;
        // Each RMS norm of a block also keeps its normalized values beside its scaled output (the normalization's result,
        // which the gain's backward reads), float32 on every rung: the block's norms, the norms after attention and the
        // feed-forward where the family has them, and each head's query and key norms.
        long normalized = d * ((spec.ParallelBlocks ? 1 : 2) + (spec.PostNorms ? 2 : 0)) + (spec.QkNorm ? queries + keys : 0);
        long full = (compressible + ff + statistics + normalized) * sizeof(float);   // a block's every result, float32
        long kept = rung.Checkpointing ? d * sizeof(float)
            : (compressible + (rung.Recompute ? 0 : ff)) * (rung.BFloat16 ? 2 : sizeof(float)) + (statistics + normalized) * sizeof(float);

        // Attention composed from score matrices (the head size above what the device tiles): each layer keeps its
        // weights, and the backward pass holds the scores' gradient beside them.
        long scores = spec.HeadDim > capabilities.TiledAttentionHeadDim ? (long)rows * spec.Heads * length * length * sizeof(float) : 0;

        long held = spec.Layers * (positions * kept + (rung.Checkpointing ? 0 : scores))
            + (rung.Checkpointing ? positions * full + scores : 0);         // the block run again in the backward pass
        long backward = positions * full + scores;                          // the gradients of the block the backward pass is in
        long residual = 2 * positions * d * sizeof(float);                 // the residual stream and its gradient
        long loss = 2 * Math.Min(Math.Max(1, lossChunkRows), positions) * spec.Vocabulary * sizeof(float);   // a chunk's logits and their gradient
        return held + backward + residual + loss + pendingGradients * sizeof(float);
    }

    /// <summary>Bytes as a line says them (GB from a gigabyte, MB below).</summary>
    public static string Size(double bytes) => bytes >= 1 << 30
        ? string.Create(CultureInfo.InvariantCulture, $"{bytes / (1 << 30):F1} GB")
        : string.Create(CultureInfo.InvariantCulture, $"{bytes / (1 << 20):F0} MB");
}
