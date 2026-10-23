// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;
using Idrak.Layers;
using System.Numerics;

namespace Idrak.Layers;

/// <summary>
/// Gradient checkpointing: a forward run again during backward instead of keeping its activations (dropout replays its
/// seeds). <see cref="ModuleExtensions.ForwardCheckpointed"/> checkpoints a module; <see cref="Checkpoint"/> any function of
/// one tensor, so a model built from its own layers (a plug-in's) keeps only its blocks' outputs too.
/// </summary>
public static class Checkpointing
{
    /// <summary>
    /// Whether a checkpointed forward pass runs on this thread without recording (the pass the backward pass repeats with
    /// recording; <see cref="Autograd.IsFirstPass"/>): layers then compute as they do with gradients (no inference-only
    /// fused kernels), so the kept output is the one the recompute gives, bit for bit.
    /// </summary>
    internal static bool FirstPass => Autograd.IsFirstPass;

    /// <summary>
    /// <paramref name="forward"/>(<paramref name="input"/>) checkpointed when <see cref="ActivationMemory.CheckpointsBlocks"/>
    /// holds on this thread and gradients are recorded, else run as it is: what a module made of blocks of its own calls for
    /// each block, so a trainer's checkpointing scope reaches it.
    /// </summary>
    public static Tensor RunBlock(Func<Tensor, Tensor> forward, Tensor input) =>
        ActivationMemory.CheckpointsBlocks && Autograd.IsEnabled ? Checkpoint(forward, input) : forward(input);

    /// <summary>
    /// <paramref name="forward"/>(input) without storing its intermediate results (activation checkpointing): the
    /// forward pass runs without recording, and the backward pass runs it again with recording, back-propagates through
    /// it and frees it at once. Parameters used inside receive their gradients then. Memory drops to the checkpointed
    /// outputs at the cost of a second forward pass. The function must be deterministic (no dropout); the recompute takes
    /// the paths the first pass took (<see cref="Autograd.FirstPass"/>), so it gives the kept output again bit for bit. The
    /// recompute runs with the image blocks the first pass ran with (<see cref="ImageBlocks.Current"/>).
    /// </summary>
    public static Tensor Checkpoint(Func<Tensor, Tensor> forward, Tensor input)
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
        var choices = new List<int>();                                  // and the paths taken (Autograd.FirstPass)
        using (Layers.DropoutSeeds.Record(seeds))
        using (Autograd.FirstPass(choices))
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
            using (Autograd.Replay(choices))
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
