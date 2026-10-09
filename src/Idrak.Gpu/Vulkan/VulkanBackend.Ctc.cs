// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// CTC loss and its gradient as generated kernels (VulkanKernels.Ctc.cs): a workgroup per sequence. The rows of α and β
// stay in workgroup memory when the longest sequence's 2L + 1 states fit the device's workgroup width (which its
// workgroup memory and invocation limits set), else in a scratch buffer; the gradient keeps α of every step in another
// ([batch, steps, states]: the CPU's O(T·S) a sequence). The lengths and offsets go up with the call (3 ints a sequence).
// The labels are not checked on the device (the CPU checks them); a case past the kernels' index range or a binding goes
// to the host fallback.
internal sealed partial class VulkanBackend
{
    public override void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        int states = CtcStates(targetLengths);
        bool shared = states <= Width;
        if (batch <= 0 || !Fit(logProbs, targets, losses) || (long)steps * batch * classes > int.MaxValue || (long)batch * 2 * states > int.MaxValue)
        {
            base.CtcLossKernel(logProbs, targets, losses, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
            return;
        }

        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var work = shared ? null : Allocate(batch * 2 * states, zeroed: false);
        try
        {
            Span<byte> b = stackalloc byte[28];
            Run(shared ? "ctc_loss" : "ctc_loss_global", RowGroups(batch), 1, 1, [logProbs, targets, meta, losses, work ?? losses],
                new Push(b).I(steps).I(batch).I(classes).I(blank).B(batchFirst).B(zeroInfinity).I(states).Bytes);
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
        bool shared = states <= Width;
        long alphaFloats = (long)batch * steps * states;
        if (batch <= 0 || !Fit(logProbs, targets, lossGrads, dLogProbs) || (long)steps * batch * classes > int.MaxValue || alphaFloats > int.MaxValue
            || BlockBytes((int)Math.Max(1, alphaFloats)) > MaxStorageBytes)
        {
            base.CtcLossBackwardKernel(logProbs, targets, lossGrads, dLogProbs, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst,
                zeroInfinity);
            return;
        }

        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var alpha = Allocate((int)Math.Max(1, alphaFloats), zeroed: false);
        var work = shared ? null : Allocate(batch * 3 * states, zeroed: false);
        try
        {
            Span<byte> b = stackalloc byte[28];
            Run(shared ? "ctc_loss_backward" : "ctc_loss_backward_global", RowGroups(batch), 1, 1, [logProbs, targets, meta, lossGrads, dLogProbs, alpha, work ?? alpha],
                new Push(b).I(steps).I(batch).I(classes).I(blank).B(batchFirst).B(zeroInfinity).I(states).Bytes);
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
