// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;

namespace Idrak.Gpu.Hip;

// One step of an LSTM or GRU cell and its gradient on HIP's GCN kernels (GcnKernels.Recurrent.cs), as CUDA implements
// IRecurrentKernels: the cell kernels a work-item a (row, unit), one launch a step; the step kernels with the recurrent
// product inside, a block a (row, 64 units), one launch a step. A storage the caller leaves out is passed as another valid
// pointer with its flag clear; the kernels never touch it. Sizes past the kernels' 32-bit byte offsets, a grid past the
// device's, or unavailable kernels return false, and the callers compose the step (the cell kernels after a product, or
// products and element-wise operations). The sequence kernels (G6c) run every step in one launch with a grid barrier
// between steps. Each launch is labelled for --profile (RecurrentCost; a sequence's per step).
internal sealed unsafe partial class HipBackend : IRecurrentKernels
{
    /// <summary>The cost of a recurrent kernel per (row, unit) of a step, for hidden size <paramref name="hidden"/> (the step kernels' products).</summary>
    internal static (double Flops, double Bytes) RecurrentCost(string kernel, int hidden) => kernel switch
    {
        "lstm_cell_f32" => (150, 64),
        "lstm_cell_bwd_f32" => (60, 72),
        "gru_cell_f32" => (110, 56),
        "gru_cell_bwd_f32" => (20, 68),
        "lstm_step_f32" => (150 + 8.0 * hidden, 48 + 16.0 * hidden),
        "lstm_step_bwd_f32" => (60 + 8.0 * hidden, 56 + 16.0 * hidden),
        "gru_step_f32" => (110 + 6.0 * hidden, 40 + 12.0 * hidden),
        "gru_step_bwd_f32" => (20 + 6.0 * hidden, 52 + 12.0 * hidden),
        "lstm_seq_f32" or "lstm_seq_bwd_f32" or "gru_seq_f32" or "gru_seq_bwd_f32" => RecurrentCost(kernel.Replace("_seq", "_step", StringComparison.Ordinal), hidden),
        _ => (0, 0),
    };

    // A cell kernel for `batch` rows of H units of a sequence of `steps` steps and rows `gates` · H wide, or null: the
    // kernels unavailable or an index past 32 bits (byte offsets).
    private GcnFunction? CellKernel(string name, int batch, int steps, int hiddenSize, int gates) =>
        KernelSizes([(long)batch * steps * gates * hiddenSize, (long)batch * hiddenSize * gates], [hiddenSize, steps]) && batch > 0 ? Gcn(name) : null;

    // A step kernel for these sizes, or null: also the weights within the byte offsets and the blocks within the grid.
    private GcnFunction? StepKernel(string name, int batch, int steps, int hiddenSize) =>
        batch > 0 && KernelSizes([(long)batch * steps * 4 * hiddenSize, (long)hiddenSize * 4 * hiddenSize], [hiddenSize, steps])
        && StepBlocks(batch, hiddenSize) <= DeviceLimits.MaxGridDimX ? Gcn(name) : null;

    private static long StepBlocks(int batch, int hiddenSize) => (long)batch * ((hiddenSize + GcnKernels.StepUnits - 1) / GcnKernels.StepUnits);

    private void LaunchCell(GcnFunction f, int batch, int hiddenSize, params ReadOnlySpan<ulong> values)
    {
        int n = batch * hiddenSize;
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{hiddenSize}"), RecurrentCost(f.Name, hiddenSize).Flops * n);
        }

        LaunchKernel(f, PointwiseBlocks(n), 1, values);
    }

    private void LaunchStep(GcnFunction f, int batch, int hiddenSize, params ReadOnlySpan<ulong> values)
    {
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{hiddenSize}"), RecurrentCost(f.Name, hiddenSize).Flops * batch * hiddenSize);
        }

        LaunchKernel(f, (uint)StepBlocks(batch, hiddenSize), 1, values);
    }

    bool IRecurrentKernels.LstmCell(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("lstm_cell_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2);
        LaunchCell(f, batch, hiddenSize, P(projected), P(recurrent), P(cell), P(hidden), P(output), P(gates ?? output), P(cells ?? output), U(step), U(steps),
            U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.LstmCellBackward(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("lstm_cell_bwd_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        LaunchCell(f, batch, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(dHidden), P(dCell), P(dGates), P(dStep), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.GruCell(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("gru_cell_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2);
        LaunchCell(f, batch, hiddenSize, P(projected), P(recurrent), P(hiddenBias ?? recurrent), P(hidden), P(output), P(gates ?? output), U(step), U(steps),
            U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.GruCellBackward(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("gru_cell_bwd_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        LaunchCell(f, batch, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(dHidden), P(dGates), P(dRecurrent), P(dStep), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.LstmStep(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step,
        int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("lstm_step_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(projected), P(weights), P(cell), P(output), P(gates ?? output), P(cells ?? output), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.LstmStepBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step,
        int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("lstm_step_bwd_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(weightsT), P(dCell), P(dGates), U(step), U(Math.Max(next, 0)),
            U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.GruStep(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous,
        int steps, int batch, int hiddenSize)
    {
        if (StepKernel("gru_step_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(projected), P(weights), P(hiddenBias ?? weights), P(output), P(gates ?? output), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.GruStepBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int step, int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("gru_step_bwd_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(weightsT), P(dHidden), P(dGates), P(dRecurrent), U(step),
            U(Math.Max(next, 0)), U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    // The sequence kernels (plan 35, G6c): every step in one launch, the blocks meeting at a grid barrier between steps
    // (GcnKernels.RecurrentSequence). The grid takes at most one block a compute unit, as the device reports them: a block
    // of four waves fits any gfx9 compute unit (which holds up to forty waves), so every block of the launch is resident
    // while the others wait at the barrier, without a cooperative launch. The barrier's two words are allocated once and
    // zeroed (the generation word moves on with every barrier, so they are not the completion counters, which stay zero).
    private Storage? _gridBarrier;

    private Storage GridBarrier() => _gridBarrier ??= Allocate(2, zeroed: true);

    // A sequence kernel for these sizes (the step kernel's sizes, and the kernel available), or null.
    private GcnFunction? SequenceKernel(string name, int batch, int steps, int hiddenSize) =>
        DeviceLimits.ComputeUnits > 0 && StepKernel(name.Replace("_seq", "_step", StringComparison.Ordinal), Math.Max(batch, 1), steps, hiddenSize) is not null
            ? Gcn(name) : null;

    // The blocks of a sequence launch: one an item of a step (batch · ⌈H / 64⌉), at most one a compute unit.
    private int SequenceBlocks(int batch, int hiddenSize) => (int)Math.Min(StepBlocks(batch, hiddenSize), DeviceLimits.ComputeUnits);

    private void LaunchSequence(GcnFunction f, int batch, int steps, int hiddenSize, params ReadOnlySpan<ulong> values)
    {
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{hiddenSize}x{steps}"),
                RecurrentCost(f.Name, hiddenSize).Flops * batch * hiddenSize * steps);
        }

        LaunchKernel(f, (uint)SequenceBlocks(batch, hiddenSize), 1, values);
    }

    bool IRecurrentKernels.LstmSequence(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps,
        int batch, int hiddenSize, bool reverse)
    {
        if (SequenceKernel("lstm_seq_f32", batch, steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || steps == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2) | (reverse ? GcnKernels.SequenceReverse : 0);
        LaunchSequence(f, batch, steps, hiddenSize, P(projected), P(weights), P(cell), P(output), P(gates ?? output), P(cells ?? output), P(GridBarrier()),
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    bool IRecurrentKernels.LstmSequenceBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps,
        int batch, int hiddenSize, bool reverse)
    {
        if (SequenceKernel("lstm_seq_bwd_f32", batch, steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || steps == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (reverse ? GcnKernels.SequenceReverse : 0);
        LaunchSequence(f, batch, steps, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(weightsT), P(dCell), P(dGates), P(GridBarrier()),
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    bool IRecurrentKernels.GruSequence(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch,
        int hiddenSize, bool reverse)
    {
        if (SequenceKernel("gru_seq_f32", batch, steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || steps == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2) | (reverse ? GcnKernels.SequenceReverse : 0);
        LaunchSequence(f, batch, steps, hiddenSize, P(projected), P(weights), P(hiddenBias ?? weights), P(output), P(gates ?? output), P(GridBarrier()),
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    bool IRecurrentKernels.GruSequenceBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int steps, int batch, int hiddenSize, bool reverse)
    {
        if (SequenceKernel("gru_seq_bwd_f32", batch, steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || steps == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (reverse ? GcnKernels.SequenceReverse : 0);
        LaunchSequence(f, batch, steps, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(weightsT), P(dHidden), P(dGates), P(dRecurrent),
            P(GridBarrier()), U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }
}
