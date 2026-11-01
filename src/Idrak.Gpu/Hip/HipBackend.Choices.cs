// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Cuda;

namespace Idrak.Gpu.Hip;

// The choices CUDA makes for its callers, made on HIP (plan 35, HIP parity):
// - PrefersComposedAttention: CUDA's rule, measured on this device. Attention over key ranges runs AttentionSpans (HIP's
//   GCN attention kernels, G4b and G6b) or the composed full scores (HIP's products and softmax); which is faster for a
//   shape and precision, inference or training (with the gradients), is timed once on the device into scratch memory and
//   kept in HIP's tuning table (per GPU, driver and build, G7a), the key CUDA's (TuneOp.AttentionPath). While nothing can
//   be measured (IDRAK_AUTOTUNE=0, a profile, a timing under way) the answer is CUDA's default: AttentionSpans, which
//   never holds the scores.
// - ReuseQuantizedOperands: CUDA's scope reuses the 8-bit form of an activation its float8 tensor-core products quantize.
//   HIP's products quantize no operand (its packed products read weights quantized once, at load; the matrix units that
//   would take 8-bit activations are G9's), so there is nothing to reuse and the scope is none, as the contract says of a
//   device that quantizes nothing.
internal sealed unsafe partial class HipBackend
{
    public override bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale, bool training = false)
    {
        using var operation = EnterOperation();                            // addresses read until the work is queued (the move guard)
        // Candidates: 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed; keyed by the
        // precision and by whether the gradient is timed with the forward pass, as CUDA's.
        var key = new TuneKey(TuneOp.AttentionPath, (int)MixedPrecision.Current, heads, TuneSizes.Class(rows), TuneSizes.Class(keyRows), dim,
            (mask is null ? 0 : 1) | (training ? 2 : 0));
        int[] candidates = [0, 1];
        if (!Autotune)
        {
            return false;
        }

        if (Tuning.TryKnown(key, candidates, out int known))
        {
            return known == 1;
        }

        long scores = (long)heads * rows * keyRows, outputs = (long)heads * rows * dim, keyFloats = (long)heads * keyRows * dim;
        if (!CanMeasure || scores > int.MaxValue || outputs > int.MaxValue || keyFloats > int.MaxValue)
        {
            return false;
        }

        var scratch = new List<Storage>();
        Storage Scratch(long length)
        {
            var storage = Allocate((int)length, zeroed: true);
            scratch.Add(storage);
            return storage;
        }

        try
        {
            var (s, w, y) = (Scratch(scores), Scratch(scores), Scratch(outputs));
            Storage? g = null, lse = null, dq = null, dk = null, dv = null;
            if (training)
            {
                (g, lse, dq, dk, dv) = (Scratch(scores), Scratch((long)heads * rows), Scratch(outputs), Scratch(keyFloats), Scratch(keyFloats));
            }

            return Tune(key, candidates, 0, c =>
            {
                if (!training)
                {
                    if (c == 0)
                    {
                        AttentionSpans(q, keys, values, starts, ends, y, null, heads, heads, heads, rows, keyRows, dim, scale);
                    }
                    else
                    {
                        ComposedAttention(q, keys, values, mask, s, w, y, heads, rows, keyRows, dim, scale);
                    }
                }
                else if (c == 0)
                {
                    AttentionSpans(q, keys, values, starts, ends, y, lse, heads, heads, heads, rows, keyRows, dim, scale);
                    AttentionSpansBackward(q, keys, values, starts, ends, y, lse!, y, dq!, dk!, dv!, heads, heads, heads, rows, keyRows, dim, scale);
                }
                else
                {
                    ComposedAttentionTraining(q, keys, values, mask, s, w, g!, y, y, dq!, dk!, dv!, heads, rows, keyRows, dim, scale);
                }
            }) == 1;
        }
        catch (ResourceLimitExceededException)
        {
            return false;                                                  // no room for the scores: the spans, which never hold them
        }
        finally
        {
            foreach (var storage in scratch)
            {
                storage.Release();
            }
        }
    }

    public override IDisposable? ReuseQuantizedOperands() => null;
}
