// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Abstraction.Devices.Cpu;

// One time step of an LSTM or GRU cell and its gradient (Backend.LstmCell, GruCell and their backward kernels): each row
// of the batch in one fused pass over its hidden units (a row's gate values are contiguous, so every load is a vector
// along the row), with the activations of the element-wise kernels (σ(x) = 1 / (1 + e^-x), tanh(x) = 1 - 2 / (e^2x + 1)
// in the vector lanes, MathF in the scalar tail), so the step agrees with the composed operations. Rows split across
// cores only when the step is large enough to pay for it.
internal sealed partial class CpuBackend
{
    // Element operations a hidden unit of a step costs (the activations and the products), to decide whether rows split.
    private const int CellWork = 24;

    public override bool LstmCellKernel(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            RunRows(new LstmCellRows(D(projected), D(recurrent), D(cell), D(hidden), D(output), gates is null ? null : D(gates), cells is null ? null : D(cells),
                step, steps, hiddenSize), batch, (long)batch * hiddenSize * CellWork);
        }

        return true;
    }

    public override bool LstmCellBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            RunRows(new LstmCellBackwardRows(D(gates), D(cells), dOutput is null ? null : D(dOutput), D(dHidden), D(dCell), D(dGates), D(dStep), step, previous, steps,
                hiddenSize), batch, (long)batch * hiddenSize * CellWork);
        }

        return true;
    }

    public override bool GruCellKernel(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            RunRows(new GruCellRows(D(projected), D(recurrent), hiddenBias is null ? null : D(hiddenBias), D(hidden), D(output), gates is null ? null : D(gates),
                step, steps, hiddenSize), batch, (long)batch * hiddenSize * CellWork);
        }

        return true;
    }

    public override bool GruCellBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            RunRows(new GruCellBackwardRows(D(gates), D(output), dOutput is null ? null : D(dOutput), D(dHidden), D(dGates), D(dRecurrent), D(dStep), step, previous,
                steps, hiddenSize), batch, (long)batch * hiddenSize * CellWork);
        }

        return true;
    }

    // Runs rows [0, rows) of a kernel inline, or in chunks of rows across cores when the work pays for it.
    private static void RunRows<TKernel>(TKernel kernel, int rows, long work)
        where TKernel : struct, IRangeKernel
    {
        if (rows <= 1 || !CpuTuning.SplitElements(work))
        {
            kernel.Execute(0, rows);
            return;
        }

        RunRowsInParallel(kernel, rows);
    }

    // Apart from RunRows, so the closure exists only when the rows do split.
    private static void RunRowsInParallel<TKernel>(TKernel kernel, int rows)
        where TKernel : struct, IRangeKernel
    {
        int chunks = Math.Min(rows, ComputeResources.MaxCpuThreads * 2);
        int size = (rows + chunks - 1) / chunks;
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
        {
            int start = c * size;
            if (start < rows)
            {
                kernel.Execute(start, Math.Min(start + size, rows));
            }
        });
    }

    private static Vector<float> SigmoidLanes(Vector<float> x) => Vector<float>.One / (Vector<float>.One + Vector.Exp(-x));

    private static Vector<float> TanhLanes(Vector<float> x)
    {
        var two = new Vector<float>(2f);
        return Vector<float>.One - two / (Vector.Exp(x * two) + Vector<float>.One);
    }

    // An LSTM step for rows [start, end): reads the step's projected row and the recurrent product, updates the cell and
    // hidden states in place, writes the output row and (training) the gates and the cell.
    private readonly struct LstmCellRows(float[] projected, float[] recurrent, float[] cell, float[] hidden, float[] output, float[]? gates, float[]? cells,
        int step, int steps, int h) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int w = 4 * h;
            for (int n = start; n < end; n++)
            {
                int row = n * steps + step;
                ReadOnlySpan<float> p = projected.AsSpan(row * w, w), q = recurrent.AsSpan(n * w, w);
                Span<float> c = cell.AsSpan(n * h, h), hid = hidden.AsSpan(n * h, h), o = output.AsSpan(row * h, h);
                Span<float> g = gates is null ? default : gates.AsSpan(row * w, w), cs = cells is null ? default : cells.AsSpan(row * h, h);
                int j = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int lanes = Vector<float>.Count;
                    for (; j <= h - lanes; j += lanes)
                    {
                        var input = SigmoidLanes(new Vector<float>(p[j..]) + new Vector<float>(q[j..]));
                        var forget = SigmoidLanes(new Vector<float>(p[(h + j)..]) + new Vector<float>(q[(h + j)..]));
                        var candidate = TanhLanes(new Vector<float>(p[(2 * h + j)..]) + new Vector<float>(q[(2 * h + j)..]));
                        var outputGate = SigmoidLanes(new Vector<float>(p[(3 * h + j)..]) + new Vector<float>(q[(3 * h + j)..]));
                        var next = forget * new Vector<float>(c[j..]) + input * candidate;
                        var state = outputGate * TanhLanes(next);
                        next.CopyTo(c[j..]);
                        state.CopyTo(hid[j..]);
                        state.CopyTo(o[j..]);
                        if (!g.IsEmpty)
                        {
                            input.CopyTo(g[j..]);
                            forget.CopyTo(g[(h + j)..]);
                            candidate.CopyTo(g[(2 * h + j)..]);
                            outputGate.CopyTo(g[(3 * h + j)..]);
                        }

                        if (!cs.IsEmpty)
                        {
                            next.CopyTo(cs[j..]);
                        }
                    }
                }

                for (; j < h; j++)
                {
                    float input = SigmoidOf(p[j] + q[j]), forget = SigmoidOf(p[h + j] + q[h + j]), candidate = MathF.Tanh(p[2 * h + j] + q[2 * h + j]);
                    float outputGate = SigmoidOf(p[3 * h + j] + q[3 * h + j]);
                    float next = forget * c[j] + input * candidate, state = outputGate * MathF.Tanh(next);
                    c[j] = next;
                    hid[j] = state;
                    o[j] = state;
                    if (!g.IsEmpty)
                    {
                        g[j] = input;
                        g[h + j] = forget;
                        g[2 * h + j] = candidate;
                        g[3 * h + j] = outputGate;
                    }

                    if (!cs.IsEmpty)
                    {
                        cs[j] = next;
                    }
                }
            }
        }
    }

    // The gradient of an LSTM step for rows [start, end) (Backend.LstmCellBackwardKernel).
    private readonly struct LstmCellBackwardRows(float[] gates, float[] cells, float[]? dOutput, float[] dHidden, float[] dCell, float[] dGates, float[] dStep,
        int step, int previous, int steps, int h) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int w = 4 * h;
            for (int n = start; n < end; n++)
            {
                int row = n * steps + step;
                ReadOnlySpan<float> g = gates.AsSpan(row * w, w), c = cells.AsSpan(row * h, h), dh = dHidden.AsSpan(n * h, h);
                ReadOnlySpan<float> cp = previous >= 0 ? cells.AsSpan((n * steps + previous) * h, h) : default;
                ReadOnlySpan<float> dy = dOutput is null ? default : dOutput.AsSpan(row * h, h);
                Span<float> dc = dCell.AsSpan(n * h, h), dz = dGates.AsSpan(row * w, w), ds = dStep.AsSpan(n * w, w);
                int j = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int lanes = Vector<float>.Count;
                    var one = Vector<float>.One;
                    for (; j <= h - lanes; j += lanes)
                    {
                        var input = new Vector<float>(g[j..]);
                        var forget = new Vector<float>(g[(h + j)..]);
                        var candidate = new Vector<float>(g[(2 * h + j)..]);
                        var outputGate = new Vector<float>(g[(3 * h + j)..]);
                        var dState = new Vector<float>(dh[j..]);
                        if (!dy.IsEmpty)
                        {
                            dState += new Vector<float>(dy[j..]);
                        }

                        var t = TanhLanes(new Vector<float>(c[j..]));
                        var dNext = new Vector<float>(dc[j..]) + dState * outputGate * (one - t * t);
                        var before = cp.IsEmpty ? Vector<float>.Zero : new Vector<float>(cp[j..]);
                        var di = dNext * candidate * input * (one - input);
                        var df = dNext * before * forget * (one - forget);
                        var dg = dNext * input * (one - candidate * candidate);
                        var dout = dState * t * outputGate * (one - outputGate);
                        (dNext * forget).CopyTo(dc[j..]);
                        di.CopyTo(dz[j..]);
                        df.CopyTo(dz[(h + j)..]);
                        dg.CopyTo(dz[(2 * h + j)..]);
                        dout.CopyTo(dz[(3 * h + j)..]);
                        di.CopyTo(ds[j..]);
                        df.CopyTo(ds[(h + j)..]);
                        dg.CopyTo(ds[(2 * h + j)..]);
                        dout.CopyTo(ds[(3 * h + j)..]);
                    }
                }

                for (; j < h; j++)
                {
                    float input = g[j], forget = g[h + j], candidate = g[2 * h + j], outputGate = g[3 * h + j];
                    float dState = dh[j] + (dy.IsEmpty ? 0f : dy[j]);
                    float t = MathF.Tanh(c[j]);
                    float dNext = dc[j] + dState * outputGate * (1f - t * t);
                    float before = cp.IsEmpty ? 0f : cp[j];
                    float di = dNext * candidate * input * (1f - input), df = dNext * before * forget * (1f - forget);
                    float dg = dNext * input * (1f - candidate * candidate), dout = dState * t * outputGate * (1f - outputGate);
                    dc[j] = dNext * forget;
                    dz[j] = ds[j] = di;
                    dz[h + j] = ds[h + j] = df;
                    dz[2 * h + j] = ds[2 * h + j] = dg;
                    dz[3 * h + j] = ds[3 * h + j] = dout;
                }
            }
        }
    }

    // A GRU step for rows [start, end): reads the step's projected row and the recurrent product, updates the hidden state
    // in place, writes the output row and (training) r, u, the candidate and its recurrent term.
    private readonly struct GruCellRows(float[] projected, float[] recurrent, float[]? hiddenBias, float[] hidden, float[] output, float[]? gates,
        int step, int steps, int h) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int w = 3 * h;
            for (int n = start; n < end; n++)
            {
                int row = n * steps + step;
                ReadOnlySpan<float> p = projected.AsSpan(row * w, w), q = recurrent.AsSpan(n * w, w);
                ReadOnlySpan<float> b = hiddenBias is null ? default : hiddenBias.AsSpan(0, h);
                Span<float> hid = hidden.AsSpan(n * h, h), o = output.AsSpan(row * h, h);
                Span<float> g = gates is null ? default : gates.AsSpan(row * 4 * h, 4 * h);
                int j = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int lanes = Vector<float>.Count;
                    for (; j <= h - lanes; j += lanes)
                    {
                        var reset = SigmoidLanes(new Vector<float>(p[j..]) + new Vector<float>(q[j..]));
                        var update = SigmoidLanes(new Vector<float>(p[(h + j)..]) + new Vector<float>(q[(h + j)..]));
                        var term = new Vector<float>(q[(2 * h + j)..]);
                        if (!b.IsEmpty)
                        {
                            term += new Vector<float>(b[j..]);
                        }

                        var candidate = TanhLanes(new Vector<float>(p[(2 * h + j)..]) + reset * term);
                        var state = (Vector<float>.One - update) * candidate + update * new Vector<float>(hid[j..]);
                        state.CopyTo(hid[j..]);
                        state.CopyTo(o[j..]);
                        if (!g.IsEmpty)
                        {
                            reset.CopyTo(g[j..]);
                            update.CopyTo(g[(h + j)..]);
                            candidate.CopyTo(g[(2 * h + j)..]);
                            term.CopyTo(g[(3 * h + j)..]);
                        }
                    }
                }

                for (; j < h; j++)
                {
                    float reset = SigmoidOf(p[j] + q[j]), update = SigmoidOf(p[h + j] + q[h + j]);
                    float term = q[2 * h + j] + (b.IsEmpty ? 0f : b[j]);
                    float candidate = MathF.Tanh(p[2 * h + j] + reset * term);
                    float state = (1f - update) * candidate + update * hid[j];
                    hid[j] = state;
                    o[j] = state;
                    if (!g.IsEmpty)
                    {
                        g[j] = reset;
                        g[h + j] = update;
                        g[2 * h + j] = candidate;
                        g[3 * h + j] = term;
                    }
                }
            }
        }
    }

    // The gradient of a GRU step for rows [start, end) (Backend.GruCellBackwardKernel).
    private readonly struct GruCellBackwardRows(float[] gates, float[] output, float[]? dOutput, float[] dHidden, float[] dGates, float[] dRecurrent, float[] dStep,
        int step, int previous, int steps, int h) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int w = 3 * h;
            for (int n = start; n < end; n++)
            {
                int row = n * steps + step;
                ReadOnlySpan<float> g = gates.AsSpan(row * 4 * h, 4 * h);
                ReadOnlySpan<float> hp = previous >= 0 ? output.AsSpan((n * steps + previous) * h, h) : default;
                ReadOnlySpan<float> dy = dOutput is null ? default : dOutput.AsSpan(row * h, h);
                Span<float> dh = dHidden.AsSpan(n * h, h), dp = dGates.AsSpan(row * w, w), dq = dRecurrent.AsSpan(row * w, w), ds = dStep.AsSpan(n * w, w);
                int j = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int lanes = Vector<float>.Count;
                    var one = Vector<float>.One;
                    for (; j <= h - lanes; j += lanes)
                    {
                        var reset = new Vector<float>(g[j..]);
                        var update = new Vector<float>(g[(h + j)..]);
                        var candidate = new Vector<float>(g[(2 * h + j)..]);
                        var term = new Vector<float>(g[(3 * h + j)..]);
                        var dState = new Vector<float>(dh[j..]);
                        if (!dy.IsEmpty)
                        {
                            dState += new Vector<float>(dy[j..]);
                        }

                        var before = hp.IsEmpty ? Vector<float>.Zero : new Vector<float>(hp[j..]);
                        var dc = dState * (one - update) * (one - candidate * candidate);
                        var dr = dc * term * reset * (one - reset);
                        var du = dState * (before - candidate) * update * (one - update);
                        var dt = dc * reset;
                        (dState * update).CopyTo(dh[j..]);
                        dr.CopyTo(dp[j..]);
                        du.CopyTo(dp[(h + j)..]);
                        dc.CopyTo(dp[(2 * h + j)..]);
                        dr.CopyTo(dq[j..]);
                        du.CopyTo(dq[(h + j)..]);
                        dt.CopyTo(dq[(2 * h + j)..]);
                        dr.CopyTo(ds[j..]);
                        du.CopyTo(ds[(h + j)..]);
                        dt.CopyTo(ds[(2 * h + j)..]);
                    }
                }

                for (; j < h; j++)
                {
                    float reset = g[j], update = g[h + j], candidate = g[2 * h + j], term = g[3 * h + j];
                    float dState = dh[j] + (dy.IsEmpty ? 0f : dy[j]);
                    float before = hp.IsEmpty ? 0f : hp[j];
                    float dc = dState * (1f - update) * (1f - candidate * candidate);
                    float dr = dc * term * reset * (1f - reset), du = dState * (before - candidate) * update * (1f - update), dt = dc * reset;
                    dh[j] = dState * update;
                    dp[j] = dq[j] = ds[j] = dr;
                    dp[h + j] = dq[h + j] = ds[h + j] = du;
                    dp[2 * h + j] = dc;
                    dq[2 * h + j] = ds[2 * h + j] = dt;
                }
            }
        }
    }
}
