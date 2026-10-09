// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// CTC loss and its gradient (PtxKernels.Ctc.cs): a block per sequence. The rows of α and β stay in shared memory when the
// longest sequence's 2L + 1 states fit the block (KernelShapes.BlockSize, derived from the device's limits), else in a
// scratch buffer; the gradient keeps α of every step in another ([batch, steps, states]). The lengths and offsets go up
// with the call (3 ints a sequence). The labels are not checked on the device (the CPU checks them).
internal sealed unsafe partial class CudaBackend
{
    public override void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        int states = CtcStates(targetLengths);
        if (batch <= 0 || (long)steps * batch * classes > int.MaxValue || (long)batch * 3 * states > int.MaxValue)
        {
            base.CtcLossKernel(logProbs, targets, losses, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
            return;
        }

        bool shared = states <= _shapes.BlockSize;
        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var work = shared ? null : Allocate(batch * 2 * states, zeroed: false);
        try
        {
            Launch(K(shared ? "ctc_loss_f32" : "ctc_loss_global_f32"), (uint)batch, 1, (uint)_shapes.BlockSize, 1, P(logProbs), P(targets), P(meta), P(losses),
                P(work ?? losses), U(steps), U(batch), U(classes), U(blank), U(batchFirst ? 1 : 0), U(zeroInfinity ? 1 : 0), U(states));
        }
        finally
        {
            meta.Release();
            work?.Release();
        }
    }

    public override void CtcLossBackwardKernel(Storage logProbs, Storage targets, Storage lossGrads, Storage dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        int states = CtcStates(targetLengths);
        long alphaFloats = (long)batch * steps * states;
        if (batch <= 0 || (long)steps * batch * classes > int.MaxValue || alphaFloats > int.MaxValue || (long)batch * 3 * states > int.MaxValue)
        {
            base.CtcLossBackwardKernel(logProbs, targets, lossGrads, dLogProbs, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst,
                zeroInfinity);
            return;
        }

        bool shared = states <= _shapes.BlockSize;
        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var alpha = Allocate((int)Math.Max(1, alphaFloats), zeroed: false);
        var work = shared ? null : Allocate(batch * 3 * states, zeroed: false);
        try
        {
            Launch(K(shared ? "ctc_loss_bwd_f32" : "ctc_loss_bwd_global_f32"), (uint)batch, 1, (uint)_shapes.BlockSize, 1, P(logProbs), P(targets), P(meta),
                P(lossGrads), P(dLogProbs), P(alpha), P(work ?? alpha), U(steps), U(batch), U(classes), U(blank), U(batchFirst ? 1 : 0), U(zeroInfinity ? 1 : 0),
                U(states));
        }
        finally
        {
            meta.Release();
            alpha.Release();
            work?.Release();
        }
    }

    // The longest sequence's extended states, 2L + 1.
    private static int CtcStates(ReadOnlySpan<int> targetLengths)
    {
        int longest = 0;
        foreach (int length in targetLengths)
        {
            longest = Math.Max(longest, length);
        }

        return 2 * longest + 1;
    }

    // Per sequence: input length, target length, target offset (ints as float bits), on the device.
    private Storage CtcMeta(ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets)
    {
        var values = new float[3 * inputLengths.Length];
        for (int n = 0; n < inputLengths.Length; n++)
        {
            values[3 * n] = BitConverter.Int32BitsToSingle(inputLengths[n]);
            values[3 * n + 1] = BitConverter.Int32BitsToSingle(targetLengths[n]);
            values[3 * n + 2] = BitConverter.Int32BitsToSingle(targetOffsets[n]);
        }

        var meta = Allocate(values.Length, zeroed: false);
        Upload(values, meta);
        return meta;
    }
}
