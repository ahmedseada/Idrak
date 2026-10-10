// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// The cases of recurrent cells: an LSTM and a GRU step and their gradients against plain loops in double.
public static partial class DeviceCases
{
    private static double SigmoidOf(double x) => 1 / (1 + Math.Exp(-x));

    // Random sizes; a step in the middle, the last and the first, each with a previous step before it, after it (a reverse
    // direction) or none (the first step taken).
    private static void RecurrentCells(DeviceCaseContext c)
    {
        int batch = c.Size(1, 5), steps = c.Size(1, 6), h = c.Size(1, 40);
        foreach (int step in new[] { 0, steps - 1, steps / 2 })
        {
            int previous = c.Random.Next(3) switch { 0 => -1, 1 => step - 1, _ => step + 1 };
            previous = previous >= steps ? -1 : previous;
            bool save = c.Random.Next(4) != 0, withOutput = c.Random.Next(3) != 0, withBias = c.Random.Next(2) == 0;
            LstmStep(c, batch, steps, h, step, save);
            LstmStepGradient(c, batch, steps, h, step, previous, withOutput);
            GruStep(c, batch, steps, h, step, save, withBias);
            GruStepGradient(c, batch, steps, h, step, previous, withOutput);
        }
    }

    // z = projected[n, step] + recurrent[n]; i, f, o = σ(z_0, z_1, z_3), g = tanh(z_2); c = f·c_prev + i·g, h = o·tanh(c).
    private static void LstmStep(DeviceCaseContext c, int batch, int steps, int h, int step, bool save)
    {
        float[] projected = c.Values(batch * steps * 4 * h, 2f), recurrent = c.Values(batch * 4 * h, 2f), cell = c.Values(batch * h), output = c.Values(batch * steps * h);
        Storage cs = c.Storage(cell), hs = Random(c, batch * h), os = c.Storage(output);
        Storage? gs = save ? Random(c, batch * steps * 4 * h) : null, ss = save ? Random(c, batch * steps * h) : null;
        if (!c.Backend.LstmCell(c.Storage(projected), c.Storage(recurrent), cs, hs, os, gs, ss, step, steps, batch, h))
        {
            return;
        }

        var expectedCell = new float[batch * h];
        var expectedOutput = (float[])output.Clone();
        var expectedGates = new float[batch * 4 * h];
        for (int n = 0; n < batch; n++)
        {
            for (int j = 0; j < h; j++)
            {
                double Z(int k) => projected[(n * steps + step) * 4 * h + k * h + j] + recurrent[n * 4 * h + k * h + j];
                double i = SigmoidOf(Z(0)), f = SigmoidOf(Z(1)), g = Math.Tanh(Z(2)), o = SigmoidOf(Z(3));
                double next = f * cell[n * h + j] + i * g;
                expectedCell[n * h + j] = (float)next;
                expectedOutput[(n * steps + step) * h + j] = (float)(o * Math.Tanh(next));
                int at = n * 4 * h + j;
                (expectedGates[at], expectedGates[at + h], expectedGates[at + 2 * h], expectedGates[at + 3 * h]) = ((float)i, (float)f, (float)g, (float)o);
            }
        }

        c.ExpectClose(expectedCell, Read(cs), 1e-4f, "LSTM cell state");
        c.ExpectClose(Read(hs), StepRows(Read(os), batch, steps, h, step), 0f, "LSTM hidden state equals the output row");
        c.ExpectClose(expectedOutput, Read(os), 1e-4f, "LSTM output");
        if (gs is not null && ss is not null)
        {
            c.ExpectClose(expectedGates, StepRows(Read(gs), batch, steps, 4 * h, step), 1e-4f, "LSTM saved gates");
            c.ExpectClose(expectedCell, StepRows(Read(ss), batch, steps, h, step), 1e-4f, "LSTM saved cells");
        }
    }

    // From saved gates and cells: dh = dHidden + dOutput, t = tanh(c), dc' = dc + dh·o·(1 - t²); dz and dc_prev = dc'·f.
    private static void LstmStepGradient(DeviceCaseContext c, int batch, int steps, int h, int step, int previous, bool withOutput)
    {
        float[] gates = [.. c.Values(batch * steps * 4 * h).Select(v => 0.5f + 0.49f * v)], cells = c.Values(batch * steps * h);
        float[] dOutput = c.Values(batch * steps * h), dHidden = c.Values(batch * h), dCell = c.Values(batch * h);
        Storage dcs = c.Storage(dCell), dzs = Random(c, batch * steps * 4 * h), dss = Random(c, batch * 4 * h);
        if (!c.Backend.LstmCellBackward(c.Storage(gates), c.Storage(cells), withOutput ? c.Storage(dOutput) : null, c.Storage(dHidden), dcs, dzs, dss, step, previous, steps,
                batch, h))
        {
            return;
        }

        var expectedDz = new float[batch * 4 * h];
        var expectedDc = new float[batch * h];
        for (int n = 0; n < batch; n++)
        {
            for (int j = 0; j < h; j++)
            {
                int at = (n * steps + step) * 4 * h + j, d = n * 4 * h + j;
                double i = gates[at], f = gates[at + h], g = gates[at + 2 * h], o = gates[at + 3 * h];
                double dh = dHidden[n * h + j] + (withOutput ? dOutput[(n * steps + step) * h + j] : 0);
                double t = Math.Tanh(cells[(n * steps + step) * h + j]);
                double dc = dCell[n * h + j] + dh * o * (1 - t * t);
                double before = previous < 0 ? 0 : cells[(n * steps + previous) * h + j];
                expectedDz[d] = (float)(dc * g * i * (1 - i));
                expectedDz[d + h] = (float)(dc * before * f * (1 - f));
                expectedDz[d + 2 * h] = (float)(dc * i * (1 - g * g));
                expectedDz[d + 3 * h] = (float)(dh * t * o * (1 - o));
                expectedDc[n * h + j] = (float)(dc * f);
            }
        }

        c.ExpectClose(expectedDz, StepRows(Read(dzs), batch, steps, 4 * h, step), 1e-4f, "LSTM gate gradients");
        c.ExpectClose(expectedDz, Read(dss), 1e-4f, "LSTM step gradients");
        c.ExpectClose(expectedDc, Read(dcs), 1e-4f, "LSTM cell gradient");
    }

    // r = σ(p_0 + q_0), u = σ(p_1 + q_1), a = q_2 + b, c = tanh(p_2 + r·a), h = (1 - u)·c + u·h_prev.
    private static void GruStep(DeviceCaseContext c, int batch, int steps, int h, int step, bool save, bool withBias)
    {
        float[] projected = c.Values(batch * steps * 3 * h, 2f), recurrent = c.Values(batch * 3 * h, 2f), bias = c.Values(h), hidden = c.Values(batch * h);
        float[] output = c.Values(batch * steps * h);
        Storage hs = c.Storage(hidden), os = c.Storage(output);
        Storage? gs = save ? Random(c, batch * steps * 4 * h) : null;
        if (!c.Backend.GruCell(c.Storage(projected), c.Storage(recurrent), withBias ? c.Storage(bias) : null, hs, os, gs, step, steps, batch, h))
        {
            return;
        }

        var expectedHidden = new float[batch * h];
        var expectedGates = new float[batch * 4 * h];
        for (int n = 0; n < batch; n++)
        {
            for (int j = 0; j < h; j++)
            {
                int p = (n * steps + step) * 3 * h + j, q = n * 3 * h + j, at = n * 4 * h + j;
                double r = SigmoidOf(projected[p] + recurrent[q]), u = SigmoidOf(projected[p + h] + recurrent[q + h]);
                double a = recurrent[q + 2 * h] + (withBias ? bias[j] : 0);
                double candidate = Math.Tanh(projected[p + 2 * h] + r * a);
                expectedHidden[n * h + j] = (float)((1 - u) * candidate + u * hidden[n * h + j]);
                (expectedGates[at], expectedGates[at + h], expectedGates[at + 2 * h], expectedGates[at + 3 * h]) = ((float)r, (float)u, (float)candidate, (float)a);
            }
        }

        c.ExpectClose(expectedHidden, Read(hs), 1e-4f, "GRU hidden state");
        c.ExpectClose(expectedHidden, StepRows(Read(os), batch, steps, h, step), 1e-4f, "GRU output");
        if (gs is not null)
        {
            c.ExpectClose(expectedGates, StepRows(Read(gs), batch, steps, 4 * h, step), 1e-4f, "GRU saved gates");
        }
    }

    // From saved r, u, c, a: dh = dHidden + dOutput, dc = dh·(1 - u)·(1 - c²); dp = (dc·a·r(1 - r), dh·(h_prev - c)·u(1 - u), dc),
    // dq = (dp_0, dp_1, dc·r), dHidden = dh·u.
    private static void GruStepGradient(DeviceCaseContext c, int batch, int steps, int h, int step, int previous, bool withOutput)
    {
        float[] gates = c.Values(batch * steps * 4 * h);
        for (int i = 0; i < gates.Length; i++)
        {
            gates[i] = i / h % 4 == 3 ? 2f * gates[i] : 0.5f + 0.49f * gates[i];
        }

        float[] states = c.Values(batch * steps * h), dOutput = c.Values(batch * steps * h), dHidden = c.Values(batch * h);
        Storage dhs = c.Storage(dHidden), dps = Random(c, batch * steps * 3 * h), dqs = Random(c, batch * steps * 3 * h), dss = Random(c, batch * 3 * h);
        if (!c.Backend.GruCellBackward(c.Storage(gates), c.Storage(states), withOutput ? c.Storage(dOutput) : null, dhs, dps, dqs, dss, step, previous, steps, batch, h))
        {
            return;
        }

        var expectedDp = new float[batch * 3 * h];
        var expectedDq = new float[batch * 3 * h];
        var expectedDh = new float[batch * h];
        for (int n = 0; n < batch; n++)
        {
            for (int j = 0; j < h; j++)
            {
                int at = (n * steps + step) * 4 * h + j, d = n * 3 * h + j;
                double r = gates[at], u = gates[at + h], candidate = gates[at + 2 * h], a = gates[at + 3 * h];
                double dh = dHidden[n * h + j] + (withOutput ? dOutput[(n * steps + step) * h + j] : 0);
                double before = previous < 0 ? 0 : states[(n * steps + previous) * h + j];
                double dc = dh * (1 - u) * (1 - candidate * candidate);
                double dr = dc * a * r * (1 - r), du = dh * (before - candidate) * u * (1 - u);
                (expectedDp[d], expectedDp[d + h], expectedDp[d + 2 * h]) = ((float)dr, (float)du, (float)dc);
                (expectedDq[d], expectedDq[d + h], expectedDq[d + 2 * h]) = ((float)dr, (float)du, (float)(dc * r));
                expectedDh[n * h + j] = (float)(dh * u);
            }
        }

        c.ExpectClose(expectedDp, StepRows(Read(dps), batch, steps, 3 * h, step), 1e-4f, "GRU input gradients");
        c.ExpectClose(expectedDq, StepRows(Read(dqs), batch, steps, 3 * h, step), 1e-4f, "GRU recurrent gradients");
        c.ExpectClose(expectedDq, Read(dss), 1e-4f, "GRU step gradients");
        c.ExpectClose(expectedDh, Read(dhs), 1e-4f, "GRU direct hidden gradient");
    }

    // The rows of one step of a [batch, steps, width] sequence, [batch, width].
    private static float[] StepRows(float[] sequence, int batch, int steps, int width, int step)
    {
        var rows = new float[batch * width];
        for (int n = 0; n < batch; n++)
        {
            Array.Copy(sequence, (n * steps + step) * width, rows, n * width, width);
        }

        return rows;
    }
}
