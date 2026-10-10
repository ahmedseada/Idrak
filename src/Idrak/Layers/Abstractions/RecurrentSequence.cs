// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Diagnostics;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// The time loop of one layer and direction of an LSTM or a GRU as one autograd operation, on the device's fused cell
/// kernels (<see cref="Backend.LstmCell"/>, <see cref="Backend.GruCell"/> and their gradients). Per step the forward pass
/// launches two kernels, the recurrent product h_{t-1}·U and the cell, and the backward pass two, the cell's gradient and
/// dz_t·Uᵀ; the hidden weights' gradient is one product over every step at the end. The composed path (a step of
/// <see cref="RecurrentModule"/>'s cell, about 13 operations, each an autograd node, and twice as many in the backward
/// pass) stays the reference and the path of devices without the kernels.
/// </summary>
/// <remarks>
/// Layouts, batch first as the layer's input: the projected input [N, T, G·H] (the step's input through the input weights
/// and bias, G = 4 for an LSTM, 3 for a GRU), the output [N, T, H] (every step's hidden state, the row of step t at (n,
/// t)), the states [N, H]. Saved for the backward pass (training only; inference saves nothing): the activated gates
/// [N, T, 4H] (LSTM: i, f, g, o; GRU: r, u, the candidate and its recurrent term) and, for an LSTM, the cells [N, T, H];
/// the output itself gives the previous hidden states. The backward pass writes dz into the projected input's gradient
/// buffer ([N, T, G·H], the input projection's existing autograd takes it from there), a GRU's recurrent part into one
/// more [N, T, 3H] buffer, and reads the previous hidden states shifted by one step ([N, T, H], one strided copy) for
/// dU = Σ_t h_{t-1}ᵀ·dz_t; all are released when the step ends.
/// </remarks>
internal static class RecurrentSequence
{
    /// <summary>
    /// Every step's hidden state [N, T, H] of one layer and direction, computed as one operation from the projected input
    /// [N, T, G·H] (<paramref name="reverse"/>: the steps taken from the last to the first), with zero initial states; null
    /// when the device has no fused kernels for it (or the sequence is empty), so the caller composes the steps.
    /// </summary>
    internal static Tensor? Run(Tensor projected, RecurrentWeights weights, int hiddenSize, bool gru, bool reverse)
    {
        int batch = projected.Shape[0], steps = projected.Shape[1], h = hiddenSize, width = projected.Shape[2];
        if (batch == 0 || steps == 0 || width != (gru ? 3 : 4) * h)
        {
            return null;
        }

        var backend = projected.Backend;
        Tensor u = weights.HiddenWeight;
        Tensor? candidateBias = gru ? weights.HiddenBias : null;
        bool record = Autograd.IsEnabled && (projected.RequiresGrad || u.RequiresGrad || candidateBias?.RequiresGrad == true);

        // With batch 0 the kernels only say whether the device has them.
        var any = projected.Storage;
        bool fused = gru
            ? backend.GruCell(any, any, null, any, any, null, 0, steps, 0, h)
              && (!record || backend.GruCellBackward(any, any, null, any, any, any, any, 0, -1, steps, 0, h))
            : backend.LstmCell(any, any, any, any, any, null, null, 0, steps, 0, h)
              && (!record || backend.LstmCellBackward(any, any, null, any, any, any, any, 0, -1, steps, 0, h));
        if (!fused)
        {
            return null;
        }

        string name = gru ? "gru_sequence" : "lstm_sequence";
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var device = projected.Device;
        var y = Tensor.Empty([batch, steps, h], device);
        var gates = record ? Tensor.Empty([batch, steps, 4 * h], device, track: false) : null;
        var cells = record && !gru ? Tensor.Empty([batch, steps, h], device, track: false) : null;
        using (var hidden = Tensor.Empty([batch, h], device, zeroed: true, track: false))
        using (var recurrent = Tensor.Empty([batch, width], device, zeroed: true, track: false))
        using (var cell = gru ? null : Tensor.Empty([batch, h], device, zeroed: true, track: false))
        {
            for (int i = 0; i < steps; i++)
            {
                int t = reverse ? steps - 1 - i : i;
                if (i > 0)
                {
                    backend.MatMul(hidden.Storage, u.Storage, recurrent.Storage, batch, width, h, false, false, 0f);   // h_{t-1}·U (0 at the start)
                }

                bool ran = gru
                    ? backend.GruCell(projected.Storage, recurrent.Storage, candidateBias?.Storage, hidden.Storage, y.Storage, gates?.Storage, t, steps, batch, h)
                    : backend.LstmCell(projected.Storage, recurrent.Storage, cell!.Storage, hidden.Storage, y.Storage, gates?.Storage, cells?.Storage, t, steps, batch, h);
                if (!ran && i == 0)
                {
                    // The device has the kernel but not for these sizes: the caller composes the steps.
                    y.Dispose();
                    gates?.Dispose();
                    cells?.Dispose();
                    return null;
                }

                if (!ran)
                {
                    throw new InvalidOperationException($"{backend.Name}: the {(gru ? "GRU" : "LSTM")} cell kernel stopped running after its first step.");
                }
            }
        }

        if (!record)
        {
            return Tensor.Traced(name, y, start);
        }

        Tensor[] inputs = candidateBias is null ? [projected, u] : [projected, u, candidateBias];
        y.Record(name, g => Backward(g, y, projected, u, candidateBias, gates!, cells, h, gru, reverse), inputs);
        return Tensor.Traced(name, y, start);
    }

    // The gradient of the whole loop: the steps from the last taken to the first, each the cell's gradient (dz_t, written
    // into the projected input's gradient) and dz_t·Uᵀ into the previous step's dh; then dU in one product and a GRU's
    // candidate bias as column sums.
    private static void Backward(Tensor g, Tensor y, Tensor projected, Tensor u, Tensor? candidateBias, Tensor gates, Tensor? cells, int h, bool gru, bool reverse)
    {
        var backend = projected.Backend;
        var device = projected.Device;
        int batch = projected.Shape[0], steps = projected.Shape[1], width = projected.Shape[2];
        try
        {
            // dz goes straight into the projected input's gradient when that is its first gradient, else into a buffer added to it.
            float projectedBeta = 1f;
            Storage? projectedGrad = projected.RequiresGrad ? projected.GradientTarget(out projectedBeta) : null;
            using var dzBuffer = projectedGrad is null || projectedBeta != 0f ? Tensor.Empty([batch, steps, width], device, track: false) : null;
            var dz = dzBuffer?.Storage ?? projectedGrad!;
            using var dRecurrent = gru ? Tensor.Empty([batch, steps, width], device, track: false) : null;
            using var dStep = Tensor.Empty([batch, width], device, track: false);
            using var dHidden = Tensor.Empty([batch, h], device, zeroed: true, track: false);
            using var dCell = gru ? null : Tensor.Empty([batch, h], device, zeroed: true, track: false);
            for (int i = steps - 1; i >= 0; i--)
            {
                int t = reverse ? steps - 1 - i : i;
                int previous = i == 0 ? -1 : reverse ? t + 1 : t - 1;
                bool ran = gru
                    ? backend.GruCellBackward(gates.Storage, y.Storage, g.Storage, dHidden.Storage, dz, dRecurrent!.Storage, dStep.Storage, t, previous, steps, batch, h)
                    : backend.LstmCellBackward(gates.Storage, cells!.Storage, g.Storage, dHidden.Storage, dCell!.Storage, dz, dStep.Storage, t, previous, steps, batch, h);
                if (!ran)
                {
                    throw new InvalidOperationException($"{backend.Name}: the {(gru ? "GRU" : "LSTM")} cell gradient kernel stopped running after it said it runs.");
                }

                if (i > 0)
                {
                    // The previous step's dh: dz_t·Uᵀ (a GRU's kernel left its direct part dh·u there to add to).
                    backend.MatMul(dStep.Storage, u.Storage, dHidden.Storage, batch, h, width, false, true, gru ? 1f : 0f);
                }
            }

            var dGates = gru ? dRecurrent!.Storage : dz;
            if (u.RequiresGrad)
            {
                // dU = Σ_t h_{t-1}ᵀ·dz_t as one product over [N·T] rows: the outputs shifted one step (zero before the first).
                using var previousStates = Tensor.Empty([batch, steps, h], device, zeroed: true, track: false);
                if (steps > 1)
                {
                    backend.Copy2D(y.Storage, reverse ? h : 0, steps * h, previousStates.Storage, reverse ? 0 : h, steps * h, batch, (steps - 1) * h, accumulate: false);
                }

                var target = u.GradientTarget(out float beta);
                backend.MatMul(previousStates.Storage, dGates, target, h, width, batch * steps, true, false, beta);
            }

            if (candidateBias is { RequiresGrad: true })
            {
                backend.SumColumns(dGates, 2 * h, width, candidateBias.GradStorage(), batch * steps, h);
            }

            if (projectedGrad is not null && dzBuffer is not null)
            {
                backend.Axpy(dzBuffer.Storage, projectedGrad, batch * steps * width, 1f);
            }
        }
        finally
        {
            gates.Dispose();
            cells?.Dispose();
        }
    }
}
