// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

// The fused time loop of LSTM and GRU layers (RecurrentSequence over Backend.LstmCell, GruCell and their gradients) against
// the composed steps (RecurrentModule.Cell), the cell kernels of each device against the CPU's, and finite differences.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] RecurrentGroup =
    [
        ("recurrent: the fused time loop matches the composed steps (LSTM and GRU with its candidate bias, one and two directions, one and two layers, every step and the last states, 7 and 50 steps): outputs and every gradient", RecurrentFusedMatchesComposed),
        ("recurrent: inference through the fused time loop matches training's output and the composed steps'; the device runs the cell kernels, not the steps", RecurrentFusedInference),
        ("recurrent: gradient: the fused LSTM and GRU against finite differences (inputs and weights, both directions, the candidate bias)", RecurrentFusedGradients),
        ("recurrent: the device's LSTM and GRU cell kernels and their gradients match the CPU's (forward and reverse steps, the first step, with and without saved gates and the candidate bias, units past a vector width)", RecurrentCellKernelsMatchCpu),
        ("recurrent: the device's LSTM and GRU step kernels (the recurrent product inside) and their gradients match the CPU's product and cell kernels (forward and reverse steps, the first and last steps, with and without saved gates, dOutput and the candidate bias, units past a warp)", RecurrentStepKernelsMatchCpu),
        ("recurrent: the device's LSTM and GRU sequence kernels (every step in one launch) and their gradients give its step kernels' values bit for bit (forward and reverse, with and without saved gates, dOutput and the candidate bias, units past a warp, more rows than one round of resident blocks)", RecurrentSequenceKernelsMatchSteps),
    ];

    private static RecurrentModule RecurrentLayer(bool gru, int inputs, int hidden, bool sequences, bool bidirectional, int layers, Device device, int seed)
    {
        RecurrentModule module = gru
            ? new GRU(inputs, hidden, sequences, bidirectional, layers, candidateBias: true, device: device, random: new Random(seed))
            : new LSTM(inputs, hidden, sequences, bidirectional, layers, device, new Random(seed));
        if (gru)
        {
            foreach (var w in module.Weights)
            {
                w.HiddenBias!.Load(RandomArray(new Random(seed + 1), hidden, 0.5f));
            }
        }

        return module;
    }

    // Forward and backward once, either path: the output, the input's gradient and every parameter's, and whether the cell kernels ran.
    private static (float[] Output, float[] InputGrad, float[][] Grads, long CellCalls) RecurrentPass(RecurrentModule module, float[] input, int[] shape, bool composed,
        Device device)
    {
        bool before = RecurrentModule.ComposedOnly;
        RecurrentModule.ComposedOnly = composed;
        try
        {
            using var scope = new TensorScope();
            foreach (var p in module.Parameters())
            {
                p.ZeroGrad();
            }

            var x = Tensor.From(input, shape, device, requiresGrad: true);
            long calls;
            Tensor y;
            using (var trace = Kernels.Trace(device.Backend))
            {
                y = module.Forward(x);
                var weights = Tensor.From(RandomArray(new Random(y.Size), y.Size), y.Shape, device);
                (y * weights).Sum().Backward();
                calls = trace.Calls(Ops.LstmCell) + trace.Calls(Ops.GruCell) + trace.Calls(Ops.LstmCellBackward) + trace.Calls(Ops.GruCellBackward)
                        + trace.Calls(Ops.LstmStep) + trace.Calls(Ops.GruStep) + trace.Calls(Ops.LstmStepBackward) + trace.Calls(Ops.GruStepBackward)
                        + trace.Calls(Ops.LstmSequence) + trace.Calls(Ops.GruSequence) + trace.Calls(Ops.LstmSequenceBackward) + trace.Calls(Ops.GruSequenceBackward);
            }

            return (y.ToArray(), x.Grad!.ToArray(), [.. module.Parameters().Select(p => p.Grad!.ToArray())], calls);
        }
        finally
        {
            RecurrentModule.ComposedOnly = before;
        }
    }

    // Whether the device has the fused kernels (asked with batch 0).
    private static bool HasCellKernels(Backend backend, bool gru)
    {
        var probe = backend.Allocate(1, zeroed: true);
        try
        {
            return gru
                ? backend.GruCell(probe, probe, null, probe, probe, null, 0, 1, 0, 1) && backend.GruCellBackward(probe, probe, null, probe, probe, probe, probe, 0, -1, 1, 0, 1)
                : backend.LstmCell(probe, probe, probe, probe, probe, null, null, 0, 1, 0, 1)
                  && backend.LstmCellBackward(probe, probe, null, probe, probe, probe, probe, 0, -1, 1, 0, 1);
        }
        finally
        {
            probe.Release();
        }
    }

    private static void RecurrentFusedMatchesComposed(Device device)
    {
        Check(device.Type != DeviceType.Cpu || HasCellKernels(device.Backend, false) && HasCellKernels(device.Backend, true), "the CPU has the fused cell kernels");
        const int batch = 3, inputs = 4, hidden = 5;
        int seed = 80;
        foreach (bool gru in new[] { false, true })
        {
            bool fusedHere = HasCellKernels(device.Backend, gru);
            foreach (int steps in new[] { 7, 50 })
            {
                foreach (bool bidirectional in new[] { false, true })
                {
                    foreach (int layers in new[] { 1, 2 })
                    {
                        foreach (bool sequences in new[] { true, false })
                        {
                            string what = $"{(gru ? "GRU" : "LSTM")}, {steps} steps, {(bidirectional ? "bidirectional" : "one direction")}, {layers} layer(s), {(sequences ? "every step" : "last states")}";
                            using var module = RecurrentLayer(gru, inputs, hidden, sequences, bidirectional, layers, device, seed++);
                            int[] shape = [batch, steps, inputs];
                            var input = RandomArray(new Random(seed), batch * steps * inputs, 2f);
                            var composed = RecurrentPass(module, input, shape, composed: true, device);
                            var fused = RecurrentPass(module, input, shape, composed: false, device);
                            Check(composed.CellCalls == 0 && (fused.CellCalls > 0) == fusedHere, $"{what}: cell kernel calls {composed.CellCalls} composed, {fused.CellCalls} fused");
                            AssertClose(composed.Output, fused.Output, 1e-5f, $"{device}: {what}: output");
                            AssertClose(composed.InputGrad, fused.InputGrad, 1e-4f, $"{device}: {what}: input gradient");
                            for (int i = 0; i < composed.Grads.Length; i++)
                            {
                                AssertClose(composed.Grads[i], fused.Grads[i], 1e-4f, $"{device}: {what}: parameter {i} gradient");
                            }
                        }
                    }
                }
            }
        }
    }

    private static void RecurrentFusedInference(Device device)
    {
        foreach (bool gru in new[] { false, true })
        {
            using var module = RecurrentLayer(gru, 3, 6, sequences: true, bidirectional: true, layers: 2, device, 90);
            using var scope = new TensorScope();
            var x = Tensor.From(RandomArray(new Random(91), 2 * 9 * 3), [2, 9, 3], device);
            float[] inference;
            long calls, sequences;
            using (Autograd.NoGrad())
            using (var trace = Kernels.Trace(device.Backend))
            {
                inference = module.Forward(x).ToArray();
                calls = trace.Calls(gru ? Ops.GruCell : Ops.LstmCell) + trace.Calls(gru ? Ops.GruStep : Ops.LstmStep);
                sequences = trace.Calls(gru ? Ops.GruSequence : Ops.LstmSequence);
                Check(trace.Calls(Ops.LstmCellBackward) + trace.Calls(Ops.GruCellBackward) + trace.Calls(Ops.LstmStepBackward) + trace.Calls(Ops.GruStepBackward)
                      + trace.Calls(Ops.LstmSequenceBackward) + trace.Calls(Ops.GruSequenceBackward) == 0, "inference asks for no gradient kernel");
            }

            Check(!HasCellKernels(device.Backend, gru) || sequences >= 2 * 2 || calls >= 2 * 2 * 9,
                $"{device}: {sequences} sequence kernel calls (one a layer and direction: 2 layers x 2 directions) and {calls} cell kernel calls (a step each: 2 x 2 x 9)");
            var training = module.Forward(x).ToArray();
            AssertClose(training, inference, 0f, $"{(gru ? "GRU" : "LSTM")}: inference and training give the same output");
            RecurrentModule.ComposedOnly = true;
            try
            {
                AssertClose(module.Forward(x).ToArray(), inference, 1e-5f, $"{(gru ? "GRU" : "LSTM")}: the composed steps");
            }
            finally
            {
                RecurrentModule.ComposedOnly = false;
            }
        }
    }

    private static void RecurrentFusedGradients(Device device)
    {
        using var lstm = RecurrentLayer(false, 3, 4, sequences: true, bidirectional: true, layers: 1, device, 95);
        GradCheck(device, [2, 5, 3], x => lstm.Forward(x).Square().Sum());
        ParameterGradCheck(lstm, Tensor.From(RandomArray(new Random(96), 30), [2, 5, 3], device), y => y.Square().Sum());
        using var gru = RecurrentLayer(true, 3, 4, sequences: false, bidirectional: true, layers: 2, device, 97);
        GradCheck(device, [2, 5, 3], x => gru.Forward(x).Square().Sum());
        ParameterGradCheck(gru, Tensor.From(RandomArray(new Random(98), 30), [2, 5, 3], device), y => y.Square().Sum());
        using var single = RecurrentLayer(false, 2, 3, sequences: false, bidirectional: false, layers: 1, device, 99);
        GradCheck(device, [1, 4, 2], x => single.Forward(x).Sum());
    }

    private static void RecurrentCellKernelsMatchCpu(Device device)
    {
        var random = new Random(100);
        float[] R(int n, float scale = 1f) => RandomArray(random, n, scale);
        foreach (int h in new[] { 3, 37 })
        {
            const int batch = 3, steps = 4;
            if (HasCellKernels(device.Backend, gru: false))
            {
                foreach (var (step, previous) in new[] { (2, 1), (1, 2), (0, -1) })
                {
                    string label = $"LSTM, {h} units, step {step} after {previous}";

                    // [0] projected, [1] recurrent, [2] cell, [3] hidden, [4] output, [5] gates, [6] cells.
                    OnBoth(device, label, [R(batch * steps * 4 * h, 2f), R(batch * 4 * h, 2f), R(batch * h), R(batch * h), R(batch * steps * h), R(batch * steps * 4 * h),
                        R(batch * steps * h)], (b, s) => Check(b.LstmCell(s[0], s[1], s[2], s[3], s[4], s[5], s[6], step, steps, batch, h), "the LSTM cell runs"), 1e-5f);
                    OnBoth(device, label + " without saving", [R(batch * steps * 4 * h, 2f), R(batch * 4 * h, 2f), R(batch * h), R(batch * h), R(batch * steps * h)],
                        (b, s) => b.LstmCell(s[0], s[1], s[2], s[3], s[4], null, null, step, steps, batch, h), 1e-5f);

                    // [0] gates (activated: in (0, 1), the candidate in (-1, 1)), [1] cells, [2] dOutput, [3] dHidden, [4] dCell, [5] dGates, [6] dStep.
                    var gates = R(batch * steps * 4 * h).Select(v => 0.5f + 0.49f * v).ToArray();
                    OnBoth(device, label + " gradient", [gates, R(batch * steps * h), R(batch * steps * h), R(batch * h), R(batch * h), R(batch * steps * 4 * h), R(batch * 4 * h)],
                        (b, s) => b.LstmCellBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], step, previous, steps, batch, h), 1e-5f);
                    OnBoth(device, label + " gradient without dOutput", [gates, R(batch * steps * h), R(batch * h), R(batch * h), R(batch * steps * 4 * h), R(batch * 4 * h)],
                        (b, s) => b.LstmCellBackward(s[0], s[1], null, s[2], s[3], s[4], s[5], step, previous, steps, batch, h), 1e-5f);
                }
            }

            if (HasCellKernels(device.Backend, gru: true))
            {
                foreach (var (step, previous) in new[] { (2, 1), (1, 2), (0, -1) })
                {
                    string label = $"GRU, {h} units, step {step} after {previous}";

                    // [0] projected, [1] recurrent, [2] candidate bias, [3] hidden, [4] output, [5] gates.
                    OnBoth(device, label, [R(batch * steps * 3 * h, 2f), R(batch * 3 * h, 2f), R(h), R(batch * h), R(batch * steps * h), R(batch * steps * 4 * h)],
                        (b, s) => Check(b.GruCell(s[0], s[1], s[2], s[3], s[4], s[5], step, steps, batch, h), "the GRU cell runs"), 1e-5f);
                    OnBoth(device, label + " without the candidate bias or saving", [R(batch * steps * 3 * h, 2f), R(batch * 3 * h, 2f), R(batch * h), R(batch * steps * h)],
                        (b, s) => b.GruCell(s[0], s[1], null, s[2], s[3], null, step, steps, batch, h), 1e-5f);

                    // [0] gates (r, u, c in range; the recurrent term any value), [1] output, [2] dOutput, [3] dHidden, [4] dGates, [5] dRecurrent, [6] dStep.
                    var gates = R(batch * steps * 4 * h);
                    for (int i = 0; i < gates.Length; i++)
                    {
                        gates[i] = i / h % 4 == 3 ? 2f * gates[i] : 0.5f + 0.49f * gates[i];
                    }

                    OnBoth(device, label + " gradient", [gates, R(batch * steps * h), R(batch * steps * h), R(batch * h), R(batch * steps * 3 * h), R(batch * steps * 3 * h), R(batch * 3 * h)],
                        (b, s) => b.GruCellBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], step, previous, steps, batch, h), 1e-5f);
                    OnBoth(device, label + " gradient without dOutput", [gates, R(batch * steps * h), R(batch * h), R(batch * steps * 3 * h), R(batch * steps * 3 * h), R(batch * 3 * h)],
                        (b, s) => b.GruCellBackward(s[0], s[1], null, s[2], s[3], s[4], s[5], step, previous, steps, batch, h), 1e-5f);
                }
            }
        }
    }

    // The step kernels against the CPU's two kernels: the previous step's hidden rows (or the later step's gate gradient
    // rows) through U (or Uᵀ) by a product, then the cell kernel. The device runs only its step kernel.
    private static void RecurrentStepKernelsMatchCpu(Device device)
    {
        var any = device.Backend.Allocate(1, zeroed: true);
        bool lstm = device.Backend.LstmStep(any, any, any, any, null, null, 0, -1, 4, 0, 1) && device.Backend.LstmStepBackward(any, any, null, any, any, any, 0, -1, -1, 4, 0, 1);
        bool gru = device.Backend.GruStep(any, any, null, any, null, 0, -1, 4, 0, 1) && device.Backend.GruStepBackward(any, any, null, any, any, any, any, 0, -1, -1, 4, 0, 1);
        any.Release();
        if (!lstm && !gru)
        {
            return;                                                                       // the device has no step kernels
        }

        var random = new Random(101);
        float[] R(int n, float scale = 1f) => RandomArray(random, n, scale);
        const int batch = 3, steps = 4;

        // rows[n, ·] = sequence[n, row, ·] (width floats a row), or 0 without a row.
        static Storage Rows(Backend b, Storage sequence, int row, int width)
        {
            var rows = b.Allocate(batch * width, zeroed: true);
            if (row >= 0)
            {
                b.Copy2D(sequence, row * width, steps * width, rows, 0, width, batch, width, accumulate: false);
            }

            return rows;
        }

        foreach (int h in new[] { 3, 37 })
        {
            foreach (var (step, previous, next) in new[] { (2, 1, 3), (1, 2, 0), (0, -1, 1), (3, 2, -1) })
            {
                if (lstm)
                {
                    string label = $"LSTM step, {h} units, step {step} after {previous}";

                    // [0] projected, [1] U [H, 4H], [2] cell, [3] output, [4] gates, [5] cells.
                    OnBoth(device, label, [R(batch * steps * 4 * h, 2f), R(h * 4 * h, 0.5f), R(batch * h), R(batch * steps * h), R(batch * steps * 4 * h), R(batch * steps * h)],
                        (b, s) =>
                        {
                            if (b != CpuBackend.Instance)
                            {
                                Check(b.LstmStep(s[0], s[1], s[2], s[3], s[4], s[5], step, previous, steps, batch, h), "the LSTM step runs");
                                return;
                            }

                            var hidden = Rows(b, s[3], previous, h);
                            var recurrent = b.Allocate(batch * 4 * h, zeroed: false);
                            b.MatMul(hidden, s[1], recurrent, batch, 4 * h, h, false, false, 0f);
                            b.LstmCell(s[0], recurrent, s[2], hidden, s[3], s[4], s[5], step, steps, batch, h);
                            hidden.Release();
                            recurrent.Release();
                        }, 1e-4f);

                    // [0] gates (activated), [1] cells, [2] dOutput, [3] Uᵀ [4H, H], [4] dCell, [5] dGates (the later step's row read).
                    var gates = R(batch * steps * 4 * h).Select(v => 0.5f + 0.49f * v).ToArray();
                    foreach (bool output in new[] { true, false })
                    {
                        OnBoth(device, label + " gradient" + (output ? "" : " without dOutput"),
                            [gates, R(batch * steps * h), R(batch * steps * h), R(4 * h * h, 0.5f), R(batch * h), R(batch * steps * 4 * h)],
                            (b, s) =>
                            {
                                if (b != CpuBackend.Instance)
                                {
                                    Check(b.LstmStepBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], step, next, previous, steps, batch, h),
                                        "the LSTM step gradient runs");
                                    return;
                                }

                                var later = Rows(b, s[5], next, 4 * h);
                                var dHidden = b.Allocate(batch * h, zeroed: true);
                                var dStep = b.Allocate(batch * 4 * h, zeroed: false);
                                b.MatMul(later, s[3], dHidden, batch, h, 4 * h, false, false, 0f);
                                b.LstmCellBackward(s[0], s[1], output ? s[2] : null, dHidden, s[4], s[5], dStep, step, previous, steps, batch, h);
                                later.Release();
                                dHidden.Release();
                                dStep.Release();
                            }, 1e-4f);
                    }
                }

                if (gru)
                {
                    string label = $"GRU step, {h} units, step {step} after {previous}";
                    foreach (bool bias in new[] { true, false })
                    {
                        // [0] projected, [1] U [H, 3H], [2] candidate bias, [3] output, [4] gates.
                        OnBoth(device, label + (bias ? "" : " without the candidate bias"),
                            [R(batch * steps * 3 * h, 2f), R(h * 3 * h, 0.5f), R(h), R(batch * steps * h), R(batch * steps * 4 * h)],
                            (b, s) =>
                            {
                                if (b != CpuBackend.Instance)
                                {
                                    Check(b.GruStep(s[0], s[1], bias ? s[2] : null, s[3], s[4], step, previous, steps, batch, h), "the GRU step runs");
                                    return;
                                }

                                var hidden = Rows(b, s[3], previous, h);
                                var recurrent = b.Allocate(batch * 3 * h, zeroed: false);
                                b.MatMul(hidden, s[1], recurrent, batch, 3 * h, h, false, false, 0f);
                                b.GruCell(s[0], recurrent, bias ? s[2] : null, hidden, s[3], s[4], step, steps, batch, h);
                                hidden.Release();
                                recurrent.Release();
                            }, 1e-4f);
                    }

                    // [0] gates (r, u, c in range; the recurrent term any value), [1] output, [2] dOutput, [3] Uᵀ [3H, H], [4] dHidden (the
                    // direct part), [5] dGates, [6] dRecurrent (the later step's row read).
                    var gates = R(batch * steps * 4 * h);
                    for (int i = 0; i < gates.Length; i++)
                    {
                        gates[i] = i / h % 4 == 3 ? 2f * gates[i] : 0.5f + 0.49f * gates[i];
                    }

                    foreach (bool output in new[] { true, false })
                    {
                        OnBoth(device, label + " gradient" + (output ? "" : " without dOutput"),
                            [gates, R(batch * steps * h), R(batch * steps * h), R(3 * h * h, 0.5f), R(batch * h), R(batch * steps * 3 * h), R(batch * steps * 3 * h)],
                            (b, s) =>
                            {
                                if (b != CpuBackend.Instance)
                                {
                                    Check(b.GruStepBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], s[6], step, next, previous, steps, batch, h),
                                        "the GRU step gradient runs");
                                    return;
                                }

                                var later = Rows(b, s[6], next, 3 * h);
                                var dStep = b.Allocate(batch * 3 * h, zeroed: false);
                                b.MatMul(later, s[3], s[4], batch, h, 3 * h, false, false, 1f);
                                b.GruCellBackward(s[0], s[1], output ? s[2] : null, s[4], s[5], s[6], dStep, step, previous, steps, batch, h);
                                later.Release();
                                dStep.Release();
                            }, 1e-4f);
                    }
                }
            }
        }
    }

    // The device's sequence kernels against its step kernels called a step at a time: the same values bit for bit (the same
    // sums in the same order; only the launches differ). 40 rows of 300 units are 400 blocks' work a step, more than one
    // round of resident blocks, so blocks take several in turn.
    private static void RecurrentSequenceKernelsMatchSteps(Device device)
    {
        var backend = device.Backend;
        var any = backend.Allocate(1, zeroed: true);
        bool lstm = backend.LstmSequence(any, any, any, any, null, null, 4, 0, 1, false) && backend.LstmSequenceBackward(any, any, null, any, any, any, 4, 0, 1, false);
        bool gru = backend.GruSequence(any, any, null, any, null, 4, 0, 1, false) && backend.GruSequenceBackward(any, any, null, any, any, any, any, 4, 0, 1, false);
        any.Release();
        if (!lstm && !gru)
        {
            return;                                                                       // the device has no sequence kernels
        }

        var random = new Random(103);
        float[] R(int n, float scale = 1f) => RandomArray(random, n, scale);
        foreach (var (batch, steps, h) in new[] { (3, 5, 3), (3, 6, 37), (40, 4, 300) })
        {
            foreach (bool reverse in new[] { false, true })
            {
                // The step taken i-th, and the steps taken before and after it (-1: none), as the layer's loop takes them.
                int T(int i) => reverse ? steps - 1 - i : i;
                int Before(int i) => i == 0 ? -1 : T(i - 1);
                int After(int i) => i == steps - 1 ? -1 : T(i + 1);
                string label = $"{batch} rows, {steps} steps, {h} units{(reverse ? ", reverse" : "")}";
                if (lstm)
                {
                    // [0] projected, [1] U [H, 4H], [2] cell (the zero initial state), [3] output, [4] gates, [5] cells.
                    foreach (bool save in new[] { true, false })
                    {
                        SameOnDevice(backend, $"LSTM sequence, {label}{(save ? "" : ", no saved gates")}",
                            [R(batch * steps * 4 * h, 2f), R(h * 4 * h, 0.5f), new float[batch * h], R(batch * steps * h), R(batch * steps * 4 * h), R(batch * steps * h)],
                            s => Check(backend.LstmSequence(s[0], s[1], s[2], s[3], save ? s[4] : null, save ? s[5] : null, steps, batch, h, reverse), "the LSTM sequence runs"),
                            s =>
                            {
                                for (int i = 0; i < steps; i++)
                                {
                                    Check(backend.LstmStep(s[0], s[1], s[2], s[3], save ? s[4] : null, save ? s[5] : null, T(i), Before(i), steps, batch, h), "the LSTM step runs");
                                }
                            });
                    }

                    // [0] gates (activated), [1] cells, [2] dOutput, [3] Uᵀ [4H, H], [4] dCell (zero at the loss's end), [5] dGates.
                    var gates = R(batch * steps * 4 * h).Select(v => 0.5f + 0.49f * v).ToArray();
                    foreach (bool output in new[] { true, false })
                    {
                        SameOnDevice(backend, $"LSTM sequence gradient, {label}{(output ? "" : ", no dOutput")}",
                            [gates, R(batch * steps * h), R(batch * steps * h), R(4 * h * h, 0.5f), new float[batch * h], R(batch * steps * 4 * h)],
                            s => Check(backend.LstmSequenceBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], steps, batch, h, reverse),
                                "the LSTM sequence gradient runs"),
                            s =>
                            {
                                for (int i = steps - 1; i >= 0; i--)
                                {
                                    Check(backend.LstmStepBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], T(i), After(i), Before(i), steps, batch, h),
                                        "the LSTM step gradient runs");
                                }
                            });
                    }
                }

                if (gru)
                {
                    // [0] projected, [1] U [H, 3H], [2] candidate bias, [3] output, [4] gates.
                    foreach (bool bias in new[] { true, false })
                    {
                        SameOnDevice(backend, $"GRU sequence, {label}{(bias ? "" : ", no candidate bias")}",
                            [R(batch * steps * 3 * h, 2f), R(h * 3 * h, 0.5f), R(h), R(batch * steps * h), R(batch * steps * 4 * h)],
                            s => Check(backend.GruSequence(s[0], s[1], bias ? s[2] : null, s[3], s[4], steps, batch, h, reverse), "the GRU sequence runs"),
                            s =>
                            {
                                for (int i = 0; i < steps; i++)
                                {
                                    Check(backend.GruStep(s[0], s[1], bias ? s[2] : null, s[3], s[4], T(i), Before(i), steps, batch, h), "the GRU step runs");
                                }
                            });
                    }

                    // [0] gates (r, u, c in range; the recurrent term any value), [1] output, [2] dOutput, [3] Uᵀ [3H, H], [4] dHidden (zero at the
                    // loss's end), [5] dGates, [6] dRecurrent.
                    var gates = R(batch * steps * 4 * h);
                    for (int i = 0; i < gates.Length; i++)
                    {
                        gates[i] = i / h % 4 == 3 ? 2f * gates[i] : 0.5f + 0.49f * gates[i];
                    }

                    foreach (bool output in new[] { true, false })
                    {
                        SameOnDevice(backend, $"GRU sequence gradient, {label}{(output ? "" : ", no dOutput")}",
                            [gates, R(batch * steps * h), R(batch * steps * h), R(3 * h * h, 0.5f), new float[batch * h], R(batch * steps * 3 * h), R(batch * steps * 3 * h)],
                            s => Check(backend.GruSequenceBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], s[6], steps, batch, h, reverse),
                                "the GRU sequence gradient runs"),
                            s =>
                            {
                                for (int i = steps - 1; i >= 0; i--)
                                {
                                    Check(backend.GruStepBackward(s[0], s[1], output ? s[2] : null, s[3], s[4], s[5], s[6], T(i), After(i), Before(i), steps, batch, h),
                                        "the GRU step gradient runs");
                                }
                            });
                    }
                }
            }
        }
    }

    // `first` and `second` on copies of the same inputs on the device: every storage must end the same, bit for bit, with
    // no operation on the host.
    private static void SameOnDevice(Backend backend, string what, float[][] inputs, Action<Storage[]> first, Action<Storage[]> second)
    {
        Storage[] Upload() => [.. inputs.Select(d =>
        {
            var s = backend.Allocate(Math.Max(1, d.Length), false);
            backend.Upload(d, s);
            return s;
        })];

        Storage[] a = Upload(), b = Upload();
        try
        {
            using (var trace = Kernels.Trace(backend))
            {
                first(a);
                second(b);
                backend.Synchronize();
                Check(trace.HostCalls == 0, $"{backend.Name}: {what} took the host fallback ({string.Join(", ", trace.HostCallsByOperation.Select(p => $"{p.Key} ×{p.Value}"))})");
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                var (x, y) = (new float[inputs[i].Length], new float[inputs[i].Length]);
                backend.Download(a[i], x);
                backend.Download(b[i], y);
                int at = -1;
                for (int k = 0; k < x.Length && at < 0; k++)
                {
                    at = BitConverter.SingleToInt32Bits(x[k]) != BitConverter.SingleToInt32Bits(y[k]) ? k : -1;
                }
                Check(at < 0, $"{backend.Name}: {what}, storage {i}: element {at} is {(at < 0 ? 0 : x[at])} in one launch, {(at < 0 ? 0 : y[at])} a step at a time");
            }
        }
        finally
        {
            foreach (var s in a.Concat(b))
            {
                s.Release();
            }
        }
    }
}
