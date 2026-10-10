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
                calls = trace.Calls(Ops.LstmCell) + trace.Calls(Ops.GruCell) + trace.Calls(Ops.LstmCellBackward) + trace.Calls(Ops.GruCellBackward);
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
            long calls;
            using (Autograd.NoGrad())
            using (var trace = Kernels.Trace(device.Backend))
            {
                inference = module.Forward(x).ToArray();
                calls = trace.Calls(gru ? Ops.GruCell : Ops.LstmCell);
                Check(trace.Calls(Ops.LstmCellBackward) + trace.Calls(Ops.GruCellBackward) == 0, "inference asks for no gradient kernel");
            }

            Check(!HasCellKernels(device.Backend, gru) || calls >= 2 * 2 * 9, $"{device}: {calls} cell kernel calls for 2 layers x 2 directions x 9 steps");
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
}
