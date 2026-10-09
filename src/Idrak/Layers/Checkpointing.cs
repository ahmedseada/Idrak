// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;
using Idrak.Layers;
using System.Numerics;

namespace Idrak.Layers;

/// <summary>Gradient checkpointing: a forward run again during backward instead of keeping its activations (dropout replays its seeds).</summary>
internal static class Checkpointing
{
    /// <summary>
    /// <paramref name="forward"/>(input) without storing its intermediate results (activation checkpointing): the
    /// forward pass runs without recording, and the backward pass runs it again with recording, back-propagates through
    /// it and frees it at once. Parameters used inside receive their gradients then. Memory drops to the checkpointed
    /// outputs at the cost of a second forward pass. The function must be deterministic (no dropout). The
    /// recompute runs with the image blocks the first pass ran with (<see cref="ImageBlocks.Current"/>).
    /// </summary>
    internal static Tensor Checkpoint(Func<Tensor, Tensor> forward, Tensor input)
    {
        if (!Autograd.IsEnabled)
        {
            return forward(input);
        }

        // Only the output survives the first pass: the module's intermediate results are freed at once. The image blocks
        // the pass ran with (ImagePrefill's, thread-local) travel with the recompute, so the block attends by the same key
        // ranges in the backward pass wherever and whenever that runs.
        var images = ImageBlocks.Current;
        Tensor output;
        var seeds = new List<uint>();                                   // dropout masks are drawn again identically
        using (Layers.DropoutSeeds.Record(seeds))
        using (Autograd.NoGrad())
        using (var pass = new TensorScope())
        {
            output = pass.Keep(forward(input));
        }

        if (ReferenceEquals(output, input))
        {
            return output;
        }

        output.Record("checkpoint", g =>
        {
            using var scope = new TensorScope();
            var replay = input.Detach();
            replay.RequiresGrad = input.RequiresGrad;
            Tensor recomputed;
            using (Layers.DropoutSeeds.Replay(seeds))
            using (images is not null && !ReferenceEquals(ImageBlocks.Current, images) ? images.Use() : (ImageBlocks.Scope?)null)
            {
                recomputed = forward(replay);
            }

            if (recomputed.RequiresGrad)
            {
                recomputed.Backward(g);
            }

            if (input.RequiresGrad && replay.Grad is { } grad)
            {
                input.AddGradient(grad, adopt: true);             // the replay dies with this scope
            }
        }, input);
        return output;
    }
}
