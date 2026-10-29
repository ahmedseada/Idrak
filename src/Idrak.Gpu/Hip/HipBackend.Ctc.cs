// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;

namespace Idrak.Gpu.Hip;

// CTC loss and its gradient on HIP's GCN kernels (GcnKernels.Ctc.cs), CUDA's launch: a block per sequence (within the
// device's grid, grid-stride past it). The rows of α and β (stride: the longest sequence's 2L + 1 states) stay in LDS when
// they fit the rows the shared kernels declare and the device lets a block have that LDS (its reported limit, rule 84),
// else in a scratch buffer (the "_global" kernels); the gradient keeps α of every step in another ([batch, steps, states],
// then each step's row maximum, [batch, steps]). The lengths and offsets go up with the call (3 ints a sequence). The
// labels are not checked on the device (the CPU checks them). Past the kernels' 32-bit byte offsets the host fallback runs.
internal sealed unsafe partial class HipBackend
{
    /// <summary>Tests set it (on their thread) to run the scratch-row kernels where the LDS ones would run.</summary>
    [ThreadStatic]
    internal static bool t_ctcScratchRows;

    /// <summary>The cost of a CTC kernel per state and step (GcnKernels.Ctc.cs); else the recurrent ones'.</summary>
    internal static (double Flops, double Bytes) CtcCost(string kernel) => kernel switch
    {
        "ctc_loss_f32" or "ctc_loss_global_f32" => (12, 12),
        "ctc_loss_bwd_f32" or "ctc_loss_bwd_global_f32" => (24, 24),
        _ => RecurrentCost(kernel, 0),
    };

    public override void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        int states = CtcStates(targetLengths);
        bool shared = CtcShared(backward: false, states);
        if (batch <= 0 || !KernelSizes([(long)steps * batch * classes, (long)batch * GcnKernels.CtcRows(false) * states], [classes])
            || Gcn(shared ? "ctc_loss_f32" : "ctc_loss_global_f32") is not { } f)
        {
            base.CtcLossKernel(logProbs, targets, losses, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
            return;
        }

        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var work = shared ? null : Allocate(batch * GcnKernels.CtcRows(false) * states, zeroed: false);
        try
        {
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{steps}x{states}"), Cost(f.Name).Flops * CtcWork(inputLengths, targetLengths));
            }

            LaunchKernel(f, GridFor(batch), 1, P(logProbs), P(targets), P(meta), P(losses), P(work ?? losses), U(steps), U(batch), U(classes), U(blank), U(batchFirst ? 1 : 0),
                U(zeroInfinity ? 1 : 0), U(states));
        }
        finally
        {
            meta.Release();                                                      // reused in stream order
            work?.Release();
        }
    }

    public override void CtcLossBackwardKernel(Storage logProbs, Storage targets, Storage lossGrads, Storage dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        int states = CtcStates(targetLengths);
        long alphaFloats = (long)batch * steps * (states + 1);
        bool shared = CtcShared(backward: true, states);
        if (batch <= 0 || !KernelSizes([(long)steps * batch * classes, (long)batch * GcnKernels.CtcRows(true) * states, (long)steps * states], [classes])
            || alphaFloats > int.MaxValue || Gcn(shared ? "ctc_loss_bwd_f32" : "ctc_loss_bwd_global_f32") is not { } f)
        {
            base.CtcLossBackwardKernel(logProbs, targets, lossGrads, dLogProbs, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst,
                zeroInfinity);
            return;
        }

        var meta = CtcMeta(inputLengths, targetLengths, targetOffsets);
        var alpha = Allocate((int)Math.Max(1, alphaFloats), zeroed: false);
        var work = shared ? null : Allocate(batch * GcnKernels.CtcRows(true) * states, zeroed: false);
        try
        {
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{steps}x{states}"), Cost(f.Name).Flops * CtcWork(inputLengths, targetLengths));
            }

            LaunchKernel(f, GridFor(batch), 1, P(logProbs), P(targets), P(meta), P(lossGrads), P(dLogProbs), P(alpha), P(work ?? alpha), U(steps), U(batch), U(classes),
                U(blank), U(batchFirst ? 1 : 0), U(zeroInfinity ? 1 : 0), U(states));
        }
        finally
        {
            meta.Release();
            alpha.Release();
            work?.Release();
        }
    }

    // Whether the rows of `states` states fit the LDS kernels' rows and the device lets a block have their LDS (rule 84).
    private bool CtcShared(bool backward, int states) =>
        !t_ctcScratchRows && 4L * GcnKernels.CtcRows(backward) * states <= GcnKernels.CtcRowsBytes && GcnKernels.CtcLdsBytes <= DeviceLimits.SharedMemoryPerBlock;

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

    // States times steps over the batch: the profile's unit count.
    private static double CtcWork(ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths)
    {
        double work = 0;
        for (int n = 0; n < inputLengths.Length; n++)
        {
            work += (double)inputLengths[n] * (2 * targetLengths[n] + 1);
        }

        return work;
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
