// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The recurrent layers' kernels (IRecurrentKernels): one time step of an LSTM or GRU cell and its gradient, each written
// once over the float lanes (a row of the batch a row kernel, its units a map), run by the width measured for its shape
// (Rows); the step kernels take the recurrent product first (the float32 product over the previous step's rows read where
// they are, each sum fused in the hidden units' order from zero, or from the gradient it adds to), then the cell; the
// sequence kernels take every step in the step kernels' order, so they give the step kernels' values bit for bit.
internal sealed unsafe partial class CpuBackend : IRecurrentKernels
{
    // Element operations a hidden unit of a step costs (the activations), for the parallel cut-over.
    private const int CellWork = 24;

    // ------------------------------------------------------------------ LSTM

    bool IRecurrentKernels.LstmCell(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pp = V(projected), pq = V(recurrent), pc = V(cell), ph = V(hidden), po = V(output), pg = gates is null ? null : V(gates),
                pcs = cells is null ? null : V(cells))
            {
                LstmForward(pp, pq, pc, ph, po, pg, pcs, step, steps, batch, hiddenSize);
            }
        }

        return true;
    }

    bool IRecurrentKernels.LstmStep(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step, int previous,
        int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * 4 * hiddenSize);
            try
            {
                fixed (float* pp = V(projected), pu = V(weights), pc = V(cell), po = V(output), pg = gates is null ? null : V(gates),
                    pcs = cells is null ? null : V(cells), pr = buffer)
                {
                    LstmStepRun(pp, pu, pc, po, pg, pcs, step, previous, steps, batch, hiddenSize, pr);
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    bool IRecurrentKernels.LstmSequence(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps, int batch,
        int hiddenSize, bool reverse)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * 4 * hiddenSize);
            try
            {
                fixed (float* pp = V(projected), pu = V(weights), pc = V(cell), po = V(output), pg = gates is null ? null : V(gates),
                    pcs = cells is null ? null : V(cells), pr = buffer)
                {
                    for (int i = 0; i < steps; i++)
                    {
                        LstmStepRun(pp, pu, pc, po, pg, pcs, Taken(i, steps, reverse), i == 0 ? -1 : Taken(i - 1, steps, reverse), steps, batch, hiddenSize, pr);
                    }
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    bool IRecurrentKernels.LstmCellBackward(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pg = V(gates), pcs = V(cells), pdo = dOutput is null ? null : V(dOutput), pdh = V(dHidden), pdc = V(dCell), pdz = V(dGates),
                pds = V(dStep))
            {
                LstmBackward(pg, pcs, pdo, pdh, hiddenSize, pdc, pdz, pds, step, previous, steps, batch, hiddenSize);
            }
        }

        return true;
    }

    bool IRecurrentKernels.LstmStepBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step, int next,
        int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * hiddenSize);
            try
            {
                fixed (float* pg = V(gates), pcs = V(cells), pdo = dOutput is null ? null : V(dOutput), put = V(weightsT), pdc = V(dCell), pdz = V(dGates),
                    pdh = buffer)
                {
                    LstmStepBackRun(pg, pcs, pdo, put, pdc, pdz, step, next, previous, steps, batch, hiddenSize, pdh);
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    bool IRecurrentKernels.LstmSequenceBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps,
        int batch, int hiddenSize, bool reverse)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * hiddenSize);
            try
            {
                fixed (float* pg = V(gates), pcs = V(cells), pdo = dOutput is null ? null : V(dOutput), put = V(weightsT), pdc = V(dCell), pdz = V(dGates),
                    pdh = buffer)
                {
                    for (int i = steps - 1; i >= 0; i--)
                    {
                        LstmStepBackRun(pg, pcs, pdo, put, pdc, pdz, Taken(i, steps, reverse), i == steps - 1 ? -1 : Taken(i + 1, steps, reverse),
                            i == 0 ? -1 : Taken(i - 1, steps, reverse), steps, batch, hiddenSize, pdh);
                    }
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    // The step a sequence takes i-th (reverse: from the last).
    private static int Taken(int i, int steps, bool reverse) => reverse ? steps - 1 - i : i;

    // An LSTM step from the recurrent product held in `recurrent` [batch, 4H]; `hidden` [batch, H] or null.
    private static void LstmForward(float* projected, float* recurrent, float* cell, float* hidden, float* output, float* gates, float* cells, int step, int steps,
        int batch, int h) =>
        Rows("LstmCell", new LstmRows(projected, recurrent, cell, hidden, output, gates, cells, step, steps, h, batch), batch, h, CellWork);

    // The step kernel: the previous step's hidden rows (output[n, previous]) through U [H, 4H] into `recurrent` [batch,
    // 4H], then the cell (no product at the first step: the zero initial state).
    private static void LstmStepRun(float* projected, float* weights, float* cell, float* output, float* gates, float* cells, int step, int previous, int steps,
        int batch, int h, float* recurrent)
    {
        if (previous >= 0)
        {
            Product(output + (long)previous * h, weights, null, recurrent, 1, batch, 4 * h, h, false, false, 0f, (long)steps * h);
        }
        else
        {
            new Span<float>(recurrent, batch * 4 * h).Clear();
        }

        LstmForward(projected, recurrent, cell, null, output, gates, cells, step, steps, batch, h);
    }

    // A step's buffer of `count` floats, from the shared pool (given back by the caller).
    private static float[] StepBuffer(long count) => System.Buffers.ArrayPool<float>.Shared.Rent((int)Math.Max(1, count));

    private static void GiveBack(float[] buffer) => System.Buffers.ArrayPool<float>.Shared.Return(buffer);

    // The gradient of an LSTM step with dh read from `dHidden` (`hiddenStride` floats a row); dStep may be null.
    private static void LstmBackward(float* gates, float* cells, float* dOutput, float* dHidden, long hiddenStride, float* dCell, float* dGates, float* dStep, int step,
        int previous, int steps, int batch, int h) =>
        Rows("LstmCellBackward", new LstmBackRows(gates, cells, dOutput, dHidden, hiddenStride, dCell, dGates, dStep, step, previous, steps, h, batch), batch, h, CellWork);

    // The step's gradient: dh = dGates[n, next] · Uᵀ into `dHidden` [batch, H] (0 for the last step taken), then the
    // cell's gradient.
    private static void LstmStepBackRun(float* gates, float* cells, float* dOutput, float* weightsT, float* dCell, float* dGates, int step, int next, int previous,
        int steps, int batch, int h, float* dHidden)
    {
        if (next >= 0)
        {
            Product(dGates + (long)next * 4 * h, weightsT, null, dHidden, 1, batch, h, 4 * h, false, false, 0f, (long)steps * 4 * h);
        }
        else
        {
            new Span<float>(dHidden, batch * h).Clear();
        }

        LstmBackward(gates, cells, dOutput, dHidden, h, dCell, dGates, null, step, previous, steps, batch, h);
    }

    // Row n: z_k = projected[n, step, k·H + j] + recurrent[n, k·H + j]; i, f, o = σ(z_0, z_1, z_3), g = tanh(z_2);
    // c = f · c_prev + i · g (fused), h = o · tanh(c).
    private struct LstmRows(float* projected, float* recurrent, float* cell, float* hidden, float* output, float* gates, float* cells, int step, int steps, int h,
        int batch) : IRowKernel
    {
        private readonly float* _cellIn = cell;
        private float* _cell = cell, _hidden = hidden, _output = output, _gates = gates, _cells = cells;

        public readonly void Row<TV, TF>(int n) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long row = (long)n * steps + step;
            var map = new LstmMap(projected + row * 4 * h, recurrent + (long)n * 4 * h, _cellIn + (long)n * h, _cell + (long)n * h, _hidden == null ? null : _hidden + (long)n * h,
                _output + row * h, _gates == null ? null : _gates + row * 4 * h, _cells == null ? null : _cells + row * h, h);
            RowMap<TV, TF, LstmMap>(ref map, h);
        }

        public void Redirect(ref Scratch scratch)
        {
            _cell = scratch.Take((long)batch * h);
            _hidden = _hidden == null ? null : scratch.Take((long)batch * h);
            _output = scratch.Take((long)batch * steps * h);
            _gates = _gates == null ? null : scratch.Take((long)batch * steps * 4 * h);
            _cells = _cells == null ? null : scratch.Take((long)batch * steps * h);
        }
    }

    private struct LstmMap(float* p, float* q, float* cellIn, float* cellOut, float* hidden, float* output, float* gates, float* cells, int h) : IRowMap
    {
        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var input = Fn<TV, TF>.Sigmoid(TF.Add(TF.Load(p + j), TF.Load(q + j)));
            var forget = Fn<TV, TF>.Sigmoid(TF.Add(TF.Load(p + h + j), TF.Load(q + h + j)));
            var candidate = Fn<TV, TF>.Tanh(TF.Add(TF.Load(p + 2 * h + j), TF.Load(q + 2 * h + j)));
            var gate = Fn<TV, TF>.Sigmoid(TF.Add(TF.Load(p + 3 * h + j), TF.Load(q + 3 * h + j)));
            var next = TF.Fma(forget, TF.Load(cellIn + j), TF.Mul(input, candidate));
            var state = TF.Mul(gate, Fn<TV, TF>.Tanh(next));
            TF.Store(cellOut + j, next);
            TF.Store(output + j, state);
            if (hidden != null)
            {
                TF.Store(hidden + j, state);
            }

            if (gates != null)
            {
                TF.Store(gates + j, input);
                TF.Store(gates + h + j, forget);
                TF.Store(gates + 2 * h + j, candidate);
                TF.Store(gates + 3 * h + j, gate);
            }

            if (cells != null)
            {
                TF.Store(cells + j, next);
            }
        }
    }

    // Row n of an LSTM step's gradient (Backend.LstmCellBackwardKernel).
    private struct LstmBackRows(float* gates, float* cells, float* dOutput, float* dHidden, long hiddenStride, float* dCell, float* dGates, float* dStep, int step,
        int previous, int steps, int h, int batch) : IRowKernel
    {
        private readonly float* _dCellIn = dCell;
        private float* _dCell = dCell, _dGates = dGates, _dStep = dStep;

        public readonly void Row<TV, TF>(int n) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long row = (long)n * steps + step;
            var map = new LstmBackMap(gates + row * 4 * h, cells + row * h, previous >= 0 ? cells + ((long)n * steps + previous) * h : null,
                dOutput == null ? null : dOutput + row * h, dHidden + n * hiddenStride, _dCellIn + (long)n * h, _dCell + (long)n * h, _dGates + row * 4 * h,
                _dStep == null ? null : _dStep + (long)n * 4 * h, h);
            RowMap<TV, TF, LstmBackMap>(ref map, h);
        }

        public void Redirect(ref Scratch scratch)
        {
            _dCell = scratch.Take((long)batch * h);
            _dGates = scratch.Take((long)batch * steps * 4 * h);
            _dStep = _dStep == null ? null : scratch.Take((long)batch * 4 * h);
        }
    }

    // dh = dHidden + dOutput, t = tanh(c), dc' = dc + dh · o · (1 − t²); dz = (dc'·g·i(1 − i), dc'·c_prev·f(1 − f),
    // dc'·i·(1 − g²), dh·t·o(1 − o)); dCell = dc'·f.
    private struct LstmBackMap(float* g, float* c, float* cPrevious, float* dy, float* dh, float* dcIn, float* dcOut, float* dz, float* ds, int h) : IRowMap
    {
        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var one = TF.All(1f);
            TV input = TF.Load(g + j), forget = TF.Load(g + h + j), candidate = TF.Load(g + 2 * h + j), gate = TF.Load(g + 3 * h + j);
            var dState = dy == null ? TF.Load(dh + j) : TF.Add(TF.Load(dh + j), TF.Load(dy + j));
            var t = Fn<TV, TF>.Tanh(TF.Load(c + j));
            var dNext = TF.Fma(TF.Mul(dState, gate), TF.Fma(Fn<TV, TF>.Neg(t), t, one), TF.Load(dcIn + j));
            var before = cPrevious == null ? TF.All(0f) : TF.Load(cPrevious + j);
            var di = TF.Mul(TF.Mul(dNext, candidate), TF.Mul(input, TF.Sub(one, input)));
            var df = TF.Mul(TF.Mul(dNext, before), TF.Mul(forget, TF.Sub(one, forget)));
            var dg = TF.Mul(TF.Mul(dNext, input), TF.Fma(Fn<TV, TF>.Neg(candidate), candidate, one));
            var dO = TF.Mul(TF.Mul(dState, t), TF.Mul(gate, TF.Sub(one, gate)));
            TF.Store(dcOut + j, TF.Mul(dNext, forget));
            TF.Store(dz + j, di);
            TF.Store(dz + h + j, df);
            TF.Store(dz + 2 * h + j, dg);
            TF.Store(dz + 3 * h + j, dO);
            if (ds != null)
            {
                TF.Store(ds + j, di);
                TF.Store(ds + h + j, df);
                TF.Store(ds + 2 * h + j, dg);
                TF.Store(ds + 3 * h + j, dO);
            }
        }
    }

    // ------------------------------------------------------------------ GRU

    bool IRecurrentKernels.GruCell(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates, int step, int steps,
        int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pp = V(projected), pq = V(recurrent), pb = hiddenBias is null ? null : V(hiddenBias), ph = V(hidden), po = V(output),
                pg = gates is null ? null : V(gates))
            {
                GruForward(pp, pq, pb, ph, hiddenSize, ph, po, pg, step, steps, batch, hiddenSize);
            }
        }

        return true;
    }

    bool IRecurrentKernels.GruStep(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous, int steps,
        int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * 3 * hiddenSize);
            try
            {
                fixed (float* pp = V(projected), pu = V(weights), pb = hiddenBias is null ? null : V(hiddenBias), po = V(output), pg = gates is null ? null : V(gates),
                    pr = buffer)
                {
                    GruStepRun(pp, pu, pb, po, pg, step, previous, steps, batch, hiddenSize, pr);
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    bool IRecurrentKernels.GruSequence(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch,
        int hiddenSize, bool reverse)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            var buffer = StepBuffer((long)batch * 3 * hiddenSize);
            try
            {
                fixed (float* pp = V(projected), pu = V(weights), pb = hiddenBias is null ? null : V(hiddenBias), po = V(output), pg = gates is null ? null : V(gates),
                    pr = buffer)
                {
                    for (int i = 0; i < steps; i++)
                    {
                        GruStepRun(pp, pu, pb, po, pg, Taken(i, steps, reverse), i == 0 ? -1 : Taken(i - 1, steps, reverse), steps, batch, hiddenSize, pr);
                    }
                }
            }
            finally
            {
                GiveBack(buffer);
            }
        }

        return true;
    }

    bool IRecurrentKernels.GruCellBackward(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pg = V(gates), po = V(output), pdo = dOutput is null ? null : V(dOutput), pdh = V(dHidden), pdz = V(dGates), pdr = V(dRecurrent),
                pds = V(dStep))
            {
                GruBackward(pg, po, pdo, pdh, pdz, pdr, pds, step, previous, steps, batch, hiddenSize);
            }
        }

        return true;
    }

    bool IRecurrentKernels.GruStepBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates, Storage dRecurrent,
        int step, int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pg = V(gates), po = V(output), pdo = dOutput is null ? null : V(dOutput), put = V(weightsT), pdh = V(dHidden), pdz = V(dGates),
                pdr = V(dRecurrent))
            {
                GruStepBackRun(pg, po, pdo, put, pdh, pdz, pdr, step, next, previous, steps, batch, hiddenSize);
            }
        }

        return true;
    }

    bool IRecurrentKernels.GruSequenceBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int steps, int batch, int hiddenSize, bool reverse)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            fixed (float* pg = V(gates), po = V(output), pdo = dOutput is null ? null : V(dOutput), put = V(weightsT), pdh = V(dHidden), pdz = V(dGates),
                pdr = V(dRecurrent))
            {
                for (int i = steps - 1; i >= 0; i--)
                {
                    GruStepBackRun(pg, po, pdo, put, pdh, pdz, pdr, Taken(i, steps, reverse), i == steps - 1 ? -1 : Taken(i + 1, steps, reverse),
                        i == 0 ? -1 : Taken(i - 1, steps, reverse), steps, batch, hiddenSize);
                }
            }
        }

        return true;
    }

    // A GRU step from the recurrent product in `recurrent` [batch, 3H]: the previous state read from `hiddenIn` (`stride`
    // floats a row; null: 0), the new one written to `hiddenOut` [batch, H] (or null) and output[n, step].
    private static void GruForward(float* projected, float* recurrent, float* bias, float* hiddenIn, long stride, float* hiddenOut, float* output, float* gates,
        int step, int steps, int batch, int h) =>
        Rows("GruCell", new GruRows(projected, recurrent, bias, hiddenIn, stride, hiddenOut, output, gates, step, steps, h, batch), batch, h, CellWork);

    // The step kernel: the previous step's rows (output[n, previous]) through U [H, 3H] into `recurrent` [batch, 3H], then
    // the cell reading them as the previous state (none at the first step).
    private static void GruStepRun(float* projected, float* weights, float* bias, float* output, float* gates, int step, int previous, int steps, int batch, int h,
        float* recurrent)
    {
        float* before = null;
        if (previous >= 0)
        {
            before = output + (long)previous * h;
            Product(before, weights, null, recurrent, 1, batch, 3 * h, h, false, false, 0f, (long)steps * h);
        }
        else
        {
            new Span<float>(recurrent, batch * 3 * h).Clear();
        }

        GruForward(projected, recurrent, bias, before, (long)steps * h, null, output, gates, step, steps, batch, h);
    }

    private static void GruBackward(float* gates, float* output, float* dOutput, float* dHidden, float* dGates, float* dRecurrent, float* dStep, int step, int previous,
        int steps, int batch, int h) =>
        Rows("GruCellBackward", new GruBackRows(gates, output, dOutput, dHidden, dGates, dRecurrent, dStep, step, previous, steps, h, batch), batch, h, CellWork);

    // The step's gradient: dHidden += dRecurrent[n, next] · Uᵀ (each sum from dHidden, the terms fused in order; none for
    // the last step taken), then the cell's gradient, which overwrites dHidden with the direct part dh · u.
    private static void GruStepBackRun(float* gates, float* output, float* dOutput, float* weightsT, float* dHidden, float* dGates, float* dRecurrent, int step,
        int next, int previous, int steps, int batch, int h)
    {
        if (next >= 0)
        {
            Product(dRecurrent + (long)next * 3 * h, weightsT, null, dHidden, 1, batch, h, 3 * h, false, false, 1f, (long)steps * 3 * h);
        }

        GruBackward(gates, output, dOutput, dHidden, dGates, dRecurrent, null, step, previous, steps, batch, h);
    }

    // Row n: r = σ(p_0 + q_0), u = σ(p_1 + q_1), a = q_2 + bias, c = tanh(p_2 + r · a) (fused), h = (1 − u) · c + u · h_prev.
    private struct GruRows(float* projected, float* recurrent, float* bias, float* hiddenIn, long stride, float* hiddenOut, float* output, float* gates, int step,
        int steps, int h, int batch) : IRowKernel
    {
        private float* _hidden = hiddenOut, _output = output, _gates = gates;

        public readonly void Row<TV, TF>(int n) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long row = (long)n * steps + step;
            var map = new GruMap(projected + row * 3 * h, recurrent + (long)n * 3 * h, bias, hiddenIn == null ? null : hiddenIn + n * stride,
                _hidden == null ? null : _hidden + (long)n * h, _output + row * h, _gates == null ? null : _gates + row * 4 * h, h);
            RowMap<TV, TF, GruMap>(ref map, h);
        }

        public void Redirect(ref Scratch scratch)
        {
            _hidden = _hidden == null ? null : scratch.Take((long)batch * h);
            _output = scratch.Take((long)batch * steps * h);
            _gates = _gates == null ? null : scratch.Take((long)batch * steps * 4 * h);
        }
    }

    private struct GruMap(float* p, float* q, float* bias, float* hiddenIn, float* hiddenOut, float* output, float* gates, int h) : IRowMap
    {
        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var reset = Fn<TV, TF>.Sigmoid(TF.Add(TF.Load(p + j), TF.Load(q + j)));
            var update = Fn<TV, TF>.Sigmoid(TF.Add(TF.Load(p + h + j), TF.Load(q + h + j)));
            var term = bias == null ? TF.Load(q + 2 * h + j) : TF.Add(TF.Load(q + 2 * h + j), TF.Load(bias + j));
            var candidate = Fn<TV, TF>.Tanh(TF.Fma(reset, term, TF.Load(p + 2 * h + j)));
            var before = hiddenIn == null ? TF.All(0f) : TF.Load(hiddenIn + j);
            var state = TF.Fma(update, before, TF.Mul(TF.Sub(TF.All(1f), update), candidate));
            TF.Store(output + j, state);
            if (hiddenOut != null)
            {
                TF.Store(hiddenOut + j, state);
            }

            if (gates != null)
            {
                TF.Store(gates + j, reset);
                TF.Store(gates + h + j, update);
                TF.Store(gates + 2 * h + j, candidate);
                TF.Store(gates + 3 * h + j, term);
            }
        }
    }

    // Row n of a GRU step's gradient (Backend.GruCellBackwardKernel).
    private struct GruBackRows(float* gates, float* output, float* dOutput, float* dHidden, float* dGates, float* dRecurrent, float* dStep, int step, int previous,
        int steps, int h, int batch) : IRowKernel
    {
        private readonly float* _dHiddenIn = dHidden;
        private float* _dHidden = dHidden, _dGates = dGates, _dRecurrent = dRecurrent, _dStep = dStep;

        public readonly void Row<TV, TF>(int n) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long row = (long)n * steps + step;
            var map = new GruBackMap(gates + row * 4 * h, previous >= 0 ? output + ((long)n * steps + previous) * h : null, dOutput == null ? null : dOutput + row * h,
                _dHiddenIn + (long)n * h, _dHidden + (long)n * h, _dGates + row * 3 * h, _dRecurrent + row * 3 * h, _dStep == null ? null : _dStep + (long)n * 3 * h, h);
            RowMap<TV, TF, GruBackMap>(ref map, h);
        }

        public void Redirect(ref Scratch scratch)
        {
            _dHidden = scratch.Take((long)batch * h);
            _dGates = scratch.Take((long)batch * steps * 3 * h);
            _dRecurrent = scratch.Take((long)batch * steps * 3 * h);
            _dStep = _dStep == null ? null : scratch.Take((long)batch * 3 * h);
        }
    }

    // dh = dHidden + dOutput, dc = dh · (1 − u) · (1 − c²); dp = (dc·a·r(1 − r), dh·(h_prev − c)·u(1 − u), dc),
    // dq = (dp_0, dp_1, dc·r); dHidden = dh·u.
    private struct GruBackMap(float* g, float* hPrevious, float* dy, float* dhIn, float* dhOut, float* dp, float* dq, float* ds, int h) : IRowMap
    {
        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var one = TF.All(1f);
            TV reset = TF.Load(g + j), update = TF.Load(g + h + j), candidate = TF.Load(g + 2 * h + j), term = TF.Load(g + 3 * h + j);
            var dState = dy == null ? TF.Load(dhIn + j) : TF.Add(TF.Load(dhIn + j), TF.Load(dy + j));
            var before = hPrevious == null ? TF.All(0f) : TF.Load(hPrevious + j);
            var dc = TF.Mul(TF.Mul(dState, TF.Sub(one, update)), TF.Fma(Fn<TV, TF>.Neg(candidate), candidate, one));
            var dr = TF.Mul(TF.Mul(dc, term), TF.Mul(reset, TF.Sub(one, reset)));
            var du = TF.Mul(TF.Mul(dState, TF.Sub(before, candidate)), TF.Mul(update, TF.Sub(one, update)));
            var dt = TF.Mul(dc, reset);
            TF.Store(dhOut + j, TF.Mul(dState, update));
            TF.Store(dp + j, dr);
            TF.Store(dp + h + j, du);
            TF.Store(dp + 2 * h + j, dc);
            TF.Store(dq + j, dr);
            TF.Store(dq + h + j, du);
            TF.Store(dq + 2 * h + j, dt);
            if (ds != null)
            {
                TF.Store(ds + j, dr);
                TF.Store(ds + h + j, du);
                TF.Store(ds + 2 * h + j, dt);
            }
        }
    }
}
