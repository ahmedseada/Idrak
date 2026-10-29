// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's recurrent kernels in GCN (plan 35, G5c): one step of an LSTM or GRU cell and its gradient (IRecurrentKernels),
// with CUDA's names and arguments (PtxKernels.Recurrent.cs) and the CPU backend's arithmetic (CpuBackend.Recurrent.cs):
// every product and sum rounded alone in the CPU's order, sigmoid and tanh as HIP's element-wise kernels compute them.
// Sequences are [batch, steps, width] row-major, the step's row of sequence n at n·steps + step; states [batch, H].
// The optional storages are passed as any valid pointer with their flag bit clear; the kernels never touch them.
//
// - The cell kernels (lstm_cell_f32, gru_cell_f32 and their gradients) take the recurrent product as an input: a
//   work-item a (row n, unit j), grid-stride (GcnKernels.Training.cs's frame "Single").
// - The step kernels (lstm_step_f32, gru_step_f32 and their gradients) take the recurrent product inside, so a time step is
//   one launch: a block of 256 (four waves) a (row n, 64 units), the waves the product's four slices (wave s takes the
//   sum's terms m = s, s + 4, …, two a pass, each value in a register of its own; the source row's value read once a wave on
//   the scalar unit, the weights read along j, coalesced). The slices' sums meet in LDS and the first wave adds them in
//   slice order, then runs the cell's arithmetic, the same as the cell kernels'. The forward product reads the previous
//   hidden state from the output sequence (another row: no work-item writes what another reads) against U [H, G·H]; the
//   gradient reads the later step's gate gradient against Uᵀ [G·H, H]. Grid x = batch · ⌈H / 64⌉ (block b: row b / ⌈H /
//   64⌉, units 64 · (b mod ⌈H / 64⌉) on), within the device's grid (else the backend declines and the layers compose).
// - CUDA's sequence kernels (every step in one cooperative launch, a grid barrier between steps) have no GCN form yet:
//   HIP declines them, and the layers run the step kernels a step at a time.
//
// For --profile's labels and --bench-hip's rates (HipBackend.RecurrentCost), per (row, unit) of a step:
//
// | Kernel             | Arguments                                                                        | Flops      | Bytes        |
// |--------------------|----------------------------------------------------------------------------------|------------|--------------|
// | lstm_cell_f32      | projected, recurrent, cell, hidden, output, gates, cells; step, steps, h, flags, n | 150      | 64           |
// | lstm_cell_bwd_f32  | gates, cells, doutput, dhidden, dcell, dgates, dstep; step, previous, steps, h, flags, n | 60 | 72        |
// | gru_cell_f32       | projected, recurrent, hbias, hidden, output, gates; step, steps, h, flags, n       | 110        | 56           |
// | gru_cell_bwd_f32   | gates, output, doutput, dhidden, dgates, drecurrent, dstep; step, previous, steps, h, flags, n | 20 | 68 |
// | lstm_step_f32      | projected, weights, cell, output, gates, cells; step, previous, steps, h, flags  | 150 + 8 H  | 48 + 16 H    |
// | lstm_step_bwd_f32  | gates, cells, doutput, weightst, dcell, dgates; step, next, previous, steps, h, flags | 60 + 8 H | 56 + 16 H |
// | gru_step_f32       | projected, weights, hbias, output, gates; step, previous, steps, h, flags        | 110 + 6 H  | 40 + 12 H    |
// | gru_step_bwd_f32   | gates, output, doutput, weightst, dhidden, dgates, drecurrent; step, next, previous, steps, h, flags | 20 + 6 H | 52 + 12 H |
//
// Flags: the cell kernels' and the step kernels' as CUDA's (lstm_cell_f32: 1 gates, 2 cells; gradients: 1 dOutput, 2 a
// previous step; gru_cell_f32: 1 the candidate's bias, 2 gates; the step kernels add 4: a previous step (forward) or a
// later one (gradients), whose product they take; without it the recurrent sums are 0).
internal static partial class GcnKernels
{
    /// <summary>The recurrent kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Recurrent => _recurrent ??= BuildRecurrent();

    private static GcnKernelSource[]? _recurrent;

    /// <summary>Units of a step kernel's block (a wave's lanes): grid x = batch · ⌈H / StepUnits⌉.</summary>
    internal const int StepUnits = 64;

    /// <summary>The slices of a step kernel's recurrent sum: the block's waves.</summary>
    internal const int StepSlices = RowThreads / StepUnits;

    private static GcnKernelSource[] BuildRecurrent() =>
    [
        RecurrentCell("lstm_cell_f32", ["projected", "recurrent", "cell", "hidden", "output", "gates", "cells"], ["step", "steps", "h", "flags"], 4, (w, c) =>
            LstmForward(w, c, recurrentOf: g => c.Load(c.Pointer("recurrent"), c.StateGate, g))),
        RecurrentCell("lstm_cell_bwd_f32", ["gates", "cells", "doutput", "dhidden", "dcell", "dgates", "dstep"], ["step", "previous", "steps", "h", "flags"], 4, (w, c) =>
            LstmBackward(w, c, c.Load(c.Pointer("dhidden"), c.State, 0))),
        RecurrentCell("gru_cell_f32", ["projected", "recurrent", "hbias", "hidden", "output", "gates"], ["step", "steps", "h", "flags"], 3, (w, c) =>
            GruForward(w, c, recurrentOf: g => c.Load(c.Pointer("recurrent"), c.StateGate, g))),
        RecurrentCell("gru_cell_bwd_f32", ["gates", "output", "doutput", "dhidden", "dgates", "drecurrent", "dstep"], ["step", "previous", "steps", "h", "flags"], 3, (w, c) =>
            GruBackward(w, c, c.Load(c.Pointer("dhidden"), c.State, 0))),
        RecurrentStep("lstm_step_f32", ["projected", "weights", "cell", "output", "gates", "cells"], ["step", "previous", "steps", "h", "flags"], 4,
            new("output", "previous", Gates: 1, Weights: "weights", WeightGates: 4, Terms: 4), (w, c, sums) => LstmForward(w, c, recurrentOf: g => sums[g])),
        RecurrentStep("lstm_step_bwd_f32", ["gates", "cells", "doutput", "weightst", "dcell", "dgates"], ["step", "next", "previous", "steps", "h", "flags"], 4,
            new("dgates", "next", Gates: 4, Weights: "weightst", WeightGates: 1, Terms: 1), (w, c, sums) => LstmBackward(w, c, sums[0])),
        RecurrentStep("gru_step_f32", ["projected", "weights", "hbias", "output", "gates"], ["step", "previous", "steps", "h", "flags"], 3,
            new("output", "previous", Gates: 1, Weights: "weights", WeightGates: 3, Terms: 3), (w, c, sums) => GruForward(w, c, recurrentOf: g => sums[g])),
        RecurrentStep("gru_step_bwd_f32", ["gates", "output", "doutput", "weightst", "dhidden", "dgates", "drecurrent"], ["step", "next", "previous", "steps", "h", "flags"], 3,
            new("drecurrent", "next", Gates: 3, Weights: "weightst", WeightGates: 1, Terms: 1), (w, c, sums) =>
            {
                string incoming = w.V();
                w.I(TrainingText($"v_add_f32_e32 {incoming}, {sums[0]}, {c.Load(c.Pointer("dhidden"), c.State, 0)}"));
                GruBackward(w, c, incoming);
            }),
    ];

    /// <summary>
    /// One (row n, unit j) of a recurrent kernel as its body sees it: the arguments' registers, the byte offsets of the
    /// unit in a row of G·H gates (Gate), of the state's G·H gates (StateGate, the cell kernels' recurrent and dstep), of
    /// the row's units (Unit), of the state (State, n·H + j), of the previous step's units (Previous) and of the row's four
    /// gates (Gates4, a GRU's saved gates); each gate's stride (g · H · 4 bytes); the flags.
    /// </summary>
    private sealed class RecurrentContext
    {
        public required string[] Buffers { get; init; }

        public required string[] Pointers { get; init; }

        public required string[] ValueNames { get; init; }

        public required string[] Values { get; init; }

        public required string Gate { get; init; }

        public required string StateGate { get; init; }

        public required string Unit { get; init; }

        public required string State { get; init; }

        public required string Previous { get; init; }

        public required string Gates4 { get; init; }

        public required string[] Strides { get; init; }

        public required string UnitIndex { get; init; }

        public required string Label { get; init; }

        public required PointwiseWriter W { get; init; }

        public string Pointer(string name) => Pointers[Array.IndexOf(Buffers, name)];

        public string Value(string name) => Values[Array.IndexOf(ValueNames, name)];

        public string Flags => Value("flags");

        // The float at byte offset `at` plus gate g's stride of the buffer `pointer`, into a fresh register.
        public string Load(string pointer, string at, int gate)
        {
            string target = W.V();
            W.I(TrainingText($"global_load_dword {target}, {Address(at, gate)}, {pointer}"));
            return target;
        }

        // Stores `values` at byte offset `at` plus each gate's stride of the buffer `pointer`.
        public void Store(string pointer, string at, params string[] values)
        {
            for (int g = 0; g < values.Length; g++)
            {
                W.I(TrainingText($"global_store_dword {Address(at, g)}, {values[g]}, {pointer}"));
            }
        }

        // The float at `at` when the flag bit is set, else 0, into a fresh register.
        public string LoadIf(string pointer, string at, int bit)
        {
            string target = W.V(), skip = TrainingText($"{Label}_load{_labels++}");
            W.I(TrainingText($"v_mov_b32_e32 {target}, 0"));
            TrainingGate(W, Flags, bit, skip);
            W.I(TrainingText($"global_load_dword {target}, {at}, {pointer}"));
            W.I(TrainingText($"{skip}:"));
            return target;
        }

        // `store` when the flag bit is set.
        public void If(int bit, Action store)
        {
            string skip = TrainingText($"{Label}_store{_labels++}");
            TrainingGate(W, Flags, bit, skip);
            store();
            W.I(TrainingText($"{skip}:"));
        }

        private int _labels;

        private string Address(string at, int gate)
        {
            if (gate == 0)
            {
                return at;
            }

            string address = W.V();
            W.I(TrainingText($"v_add_u32_e32 {address}, {Strides[gate]}, {at}"));
            return address;
        }
    }

    // target = σ(x) and the others' math through the element writer (HIP's element-wise kernels).
    private static string RecurrentSigmoid(PointwiseWriter w, string x)
    {
        string target = w.V();
        w.Sigmoid(target, x);
        return target;
    }

    private static string RecurrentTanh(PointwiseWriter w, string x)
    {
        string target = w.V();
        w.Tanh(target, x);
        return target;
    }

    private static string RecurrentOp(PointwiseWriter w, string op, string a, string b)
    {
        string target = w.V();
        w.I(TrainingText($"{op} {target}, {a}, {b}"));
        return target;
    }

    private static string RcMul(PointwiseWriter w, string a, string b) => RecurrentOp(w, "v_mul_f32_e32", a, b);

    private static string RcAdd(PointwiseWriter w, string a, string b) => RecurrentOp(w, "v_add_f32_e32", a, b);

    private static string RcSub(PointwiseWriter w, string a, string b) => RecurrentOp(w, "v_sub_f32_e32", a, b);

    // An LSTM step (CpuBackend's LstmCellRows): i, f, g, o = σ, σ, tanh, σ of projected + recurrent; next = f · c + i · g;
    // state = o · tanh(next); the cell and (the cell kernel) the hidden state updated, the output row, the cells (flag 2)
    // and the gates (flag 1) written.
    private static void LstmForward(PointwiseWriter w, RecurrentContext c, Func<int, string> recurrentOf)
    {
        var activations = new string[4];
        for (int g = 0; g < 4; g++)
        {
            string sum = RcAdd(w, c.Load(c.Pointer("projected"), c.Gate, g), recurrentOf(g));
            activations[g] = g == 2 ? RecurrentTanh(w, sum) : RecurrentSigmoid(w, sum);
        }

        var (input, forget, candidate, output) = (activations[0], activations[1], activations[2], activations[3]);
        string next = RcAdd(w, RcMul(w, forget, c.Load(c.Pointer("cell"), c.State, 0)), RcMul(w, input, candidate));
        string state = RcMul(w, output, RecurrentTanh(w, next));
        c.Store(c.Pointer("cell"), c.State, next);
        if (c.Buffers.Contains("hidden"))
        {
            c.Store(c.Pointer("hidden"), c.State, state);
        }

        c.Store(c.Pointer("output"), c.Unit, state);
        c.If(1, () => c.Store(c.Pointer("cells"), c.Unit, next));
        c.If(0, () => c.Store(c.Pointer("gates"), c.Gate, activations));
    }

    // The gradient of an LSTM step (CpuBackend's LstmCellBackwardRows) from the incoming gradient of the hidden state
    // (dHidden, or the step kernel's product), dOutput added (flag 1): dcell and dgates (and the cell kernel's dstep) written.
    private static void LstmBackward(PointwiseWriter w, RecurrentContext c, string incoming)
    {
        var gates = new string[4];
        for (int g = 0; g < 4; g++)
        {
            gates[g] = c.Load(c.Pointer("gates"), c.Gate, g);
        }

        var (input, forget, candidate, output) = (gates[0], gates[1], gates[2], gates[3]);
        string dState = RcAdd(w, incoming, c.LoadIf(c.Pointer("doutput"), c.Unit, 0));
        string t = RecurrentTanh(w, c.Load(c.Pointer("cells"), c.Unit, 0));
        string dNext = RcAdd(w, c.Load(c.Pointer("dcell"), c.State, 0), RcMul(w, RcMul(w, dState, output), RcSub(w, "1.0", RcMul(w, t, t))));
        string before = c.LoadIf(c.Pointer("cells"), c.Previous, 1);
        string di = RcMul(w, RcMul(w, RcMul(w, dNext, candidate), input), RcSub(w, "1.0", input));
        string df = RcMul(w, RcMul(w, RcMul(w, dNext, before), forget), RcSub(w, "1.0", forget));
        string dg = RcMul(w, RcMul(w, dNext, input), RcSub(w, "1.0", RcMul(w, candidate, candidate)));
        string dout = RcMul(w, RcMul(w, RcMul(w, dState, t), output), RcSub(w, "1.0", output));
        c.Store(c.Pointer("dcell"), c.State, RcMul(w, dNext, forget));
        c.Store(c.Pointer("dgates"), c.Gate, di, df, dg, dout);
        if (c.Buffers.Contains("dstep"))
        {
            c.Store(c.Pointer("dstep"), c.StateGate, di, df, dg, dout);
        }
    }

    // A GRU step (CpuBackend's GruCellRows): r, u = σ of projected + recurrent; a = the recurrent candidate term plus the
    // bias (flag 1, else 0); the candidate tanh(projected + r · a); state = (1 - u) · candidate + u · the previous state
    // (the cell kernel's hidden state, the step kernel's previous output row or 0); the gates r, u, candidate, a (flag 2).
    private static void GruForward(PointwiseWriter w, RecurrentContext c, Func<int, string> recurrentOf)
    {
        string reset = RecurrentSigmoid(w, RcAdd(w, c.Load(c.Pointer("projected"), c.Gate, 0), recurrentOf(0)));
        string update = RecurrentSigmoid(w, RcAdd(w, c.Load(c.Pointer("projected"), c.Gate, 1), recurrentOf(1)));
        string bias = c.LoadIf(c.Pointer("hbias"), c.UnitIndex, 0);
        string term = RcAdd(w, recurrentOf(2), bias);
        string candidate = RecurrentTanh(w, RcAdd(w, c.Load(c.Pointer("projected"), c.Gate, 2), RcMul(w, reset, term)));
        bool cell = c.Buffers.Contains("hidden");
        string before = cell ? c.Load(c.Pointer("hidden"), c.State, 0) : c.LoadIf(c.Pointer("output"), c.Previous, 2);
        string state = RcAdd(w, RcMul(w, RcSub(w, "1.0", update), candidate), RcMul(w, update, before));
        if (cell)
        {
            c.Store(c.Pointer("hidden"), c.State, state);
        }

        c.Store(c.Pointer("output"), c.Unit, state);
        c.If(1, () => c.Store(c.Pointer("gates"), c.Gates4, reset, update, candidate, term));
    }

    // The gradient of a GRU step (CpuBackend's GruCellBackwardRows) from the incoming gradient of the hidden state, dOutput
    // added (flag 1): dhidden, dgates, drecurrent (and the cell kernel's dstep) written.
    private static void GruBackward(PointwiseWriter w, RecurrentContext c, string incoming)
    {
        var gates = new string[4];
        for (int g = 0; g < 4; g++)
        {
            gates[g] = c.Load(c.Pointer("gates"), c.Gates4, g);
        }

        var (reset, update, candidate, term) = (gates[0], gates[1], gates[2], gates[3]);
        string dState = RcAdd(w, incoming, c.LoadIf(c.Pointer("doutput"), c.Unit, 0));
        string before = c.LoadIf(c.Pointer("output"), c.Previous, 1);
        string dc = RcMul(w, RcMul(w, dState, RcSub(w, "1.0", update)), RcSub(w, "1.0", RcMul(w, candidate, candidate)));
        string dr = RcMul(w, RcMul(w, RcMul(w, dc, term), reset), RcSub(w, "1.0", reset));
        string du = RcMul(w, RcMul(w, RcMul(w, dState, RcSub(w, before, candidate)), update), RcSub(w, "1.0", update));
        string dt = RcMul(w, dc, reset);
        c.Store(c.Pointer("dhidden"), c.State, RcMul(w, dState, update));
        c.Store(c.Pointer("dgates"), c.Gate, dr, du, dc);
        c.Store(c.Pointer("drecurrent"), c.Gate, dr, du, dt);
        if (c.Buffers.Contains("dstep"))
        {
            c.Store(c.Pointer("dstep"), c.StateGate, dr, du, dt);
        }
    }

    // A cell kernel: a work-item a (row n, unit j), i = n·H + j, grid-stride; the byte offsets of the context, then `body`.
    private static GcnKernelSource RecurrentCell(string name, string[] buffers, string[] values, int gates, Action<PointwiseWriter, RecurrentContext> body) =>
        TrainingElementwise(name, new(buffers, values, [], [])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            var r = new Dictionary<string, string>();
            var (n, j) = CvDivMod(w, e.Index, e.Values[Array.IndexOf(values, "h")], r);
            var context = Context(w, e.Label, buffers, e.Pointers, values, e.Values, gates, n, j);
            body(w, context);
        });

    // The context of unit j of row n (both vector registers): the byte offsets (see
    // RecurrentContext), each gate's stride made once.
    private static RecurrentContext Context(PointwiseWriter w, string label, string[] buffers, string[] pointers, string[] valueNames, string[] values, int gates,
        string n, string j)
    {
        string Value(string v) => values[Array.IndexOf(valueNames, v)];
        string h = Value("h"), steps = Value("steps"), step = Value("step");
        string previous = Array.IndexOf(valueNames, "previous") >= 0 ? Value("previous") : step;
        var strides = new string[4];
        for (int g = 1; g < 4; g++)
        {
            strides[g] = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {h}, {4 * g}"));
        }

        string width = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {h}, {gates}")), width4 = CvHeld(w, TrainingText($"s_lshl_b32 {{0}}, {h}, 2"));
        string row = w.V(), prior = w.V(), gate = w.V(), stateGate = w.V(), unit = w.V(), state = w.V(), before = w.V(), gates4 = w.V(), jBytes = w.V();
        CvLines(w, $"""
            v_mul_lo_u32 {row}, {n}, {steps}
            v_mov_b32_e32 {prior}, {row}
            v_add_u32_e32 {row}, {step}, {row}
            v_add_u32_e32 {prior}, {previous}, {prior}
            v_mul_lo_u32 {gate}, {row}, {width}
            v_add_u32_e32 {gate}, {gate}, {j}
            v_lshlrev_b32_e32 {gate}, 2, {gate}
            v_mul_lo_u32 {stateGate}, {n}, {width}
            v_add_u32_e32 {stateGate}, {stateGate}, {j}
            v_lshlrev_b32_e32 {stateGate}, 2, {stateGate}
            v_mul_lo_u32 {unit}, {row}, {h}
            v_add_u32_e32 {unit}, {unit}, {j}
            v_lshlrev_b32_e32 {unit}, 2, {unit}
            v_mul_lo_u32 {state}, {n}, {h}
            v_add_u32_e32 {state}, {state}, {j}
            v_lshlrev_b32_e32 {state}, 2, {state}
            v_mul_lo_u32 {before}, {prior}, {h}
            v_add_u32_e32 {before}, {before}, {j}
            v_lshlrev_b32_e32 {before}, 2, {before}
            v_mul_lo_u32 {gates4}, {row}, {width4}
            v_add_u32_e32 {gates4}, {gates4}, {j}
            v_lshlrev_b32_e32 {gates4}, 2, {gates4}
            v_lshlrev_b32_e32 {jBytes}, 2, {j}
            """);
        return new RecurrentContext
        {
            Buffers = buffers, Pointers = pointers, ValueNames = valueNames, Values = values, Gate = gate, StateGate = stateGate, Unit = unit, State = state,
            Previous = before, Gates4 = gates4, Strides = strides, UnitIndex = jBytes, Label = label, W = w,
        };
    }

    // The product a step kernel takes inside: the source sequence and the value naming its row (the previous or the later
    // step), its width in gates of H (1 or G), the weights and their width in gates of H, the terms (sums) per unit.
    private sealed record StepProduct(string Source, string Row, int Gates, string Weights, int WeightGates, int Terms);

    // Scalar registers of the step kernels' frame (the arguments take %s0 up; the element writer's %s64 up).
    private const string StepSlice = "%s40", StepChunks = "%s41", StepRow = "%s42", StepChunk = "%s43", StepT0 = "%s44", StepT1 = "%s45", StepT2 = "%s46",
        StepT3 = "%s47", StepM = "%s48", StepMw = "%s49", StepAddress = "%s[50:51]", StepA = "%s52", StepB = "%s53", StepSaved = "%s[54:55]", StepWw = "%s56",
        StepAhead = "%s57", StepSourceBase = "%s[58:59]";

    // A step kernel (see the file's remarks): the frame, the product, the slices' sums in LDS, then `body` in the first
    // wave's lanes below H with the sums in registers.
    private static GcnKernelSource RecurrentStep(string name, string[] buffers, string[] values, int gates, StepProduct product,
        Action<PointwiseWriter, RecurrentContext, string[]> body)
    {
        var w = new PointwiseWriter();
        int dwords = 2 * buffers.Length + values.Length;
        string[] pointers = [.. Enumerable.Range(0, buffers.Length).Select(b => TrainingText($"%s[{2 * b}:{2 * b + 1}]"))];
        string[] registers = [.. Enumerable.Range(0, values.Length).Select(k => TrainingText($"%s{2 * buffers.Length + k}"))];
        string Value(string v) => registers[Array.IndexOf(values, v)];
        string Pointer(string b) => pointers[Array.IndexOf(buffers, b)];
        string h = Value("h"), steps = Value("steps"), flags = Value("flags");
        int terms = product.Terms;
        string label = TrainingText($".L{name}");

        // The block's row and units, the lane's unit.
        const string Lane = "%v1", Unit = "%v2", Lane4 = "%v3", WeightAt = "%v4", WeightAhead = "%v5", PartAt = "%v6";
        var body1 = new StringBuilder();
        w.Into(body1);
        CvLines(w, $"""
            v_readfirstlane_b32 {StepSlice}, v0
            s_lshr_b32 {StepSlice}, {StepSlice}, 6
            v_and_b32_e32 {Lane}, 63, v0
            v_lshlrev_b32_e32 {Lane4}, 2, {Lane}
            s_add_u32 {StepChunks}, {h}, 63
            s_lshr_b32 {StepChunks}, {StepChunks}, 6
            v_cvt_f32_u32_e32 {WeightAt}, {StepChunks}
            s_sub_i32 {StepT0}, 0, {StepChunks}
            v_rcp_iflag_f32_e32 {WeightAt}, {WeightAt}
            v_mul_f32_e32 {WeightAt}, 0x4f7ffffe, {WeightAt}
            v_cvt_u32_f32_e32 {WeightAt}, {WeightAt}
            v_readfirstlane_b32 {StepT1}, {WeightAt}
            s_mul_i32 {StepT0}, {StepT0}, {StepT1}
            s_mul_hi_u32 {StepT0}, {StepT1}, {StepT0}
            s_add_i32 {StepT1}, {StepT1}, {StepT0}
            s_mul_hi_u32 {StepRow}, s2, {StepT1}
            s_mul_i32 {StepT2}, {StepRow}, {StepChunks}
            s_sub_i32 {StepChunk}, s2, {StepT2}
            s_add_u32 {StepT3}, {StepRow}, 1
            s_sub_i32 {StepT2}, {StepChunk}, {StepChunks}
            s_cmp_ge_u32 {StepChunk}, {StepChunks}
            s_cselect_b32 {StepChunk}, {StepT2}, {StepChunk}
            s_cselect_b32 {StepRow}, {StepT3}, {StepRow}
            s_add_u32 {StepT3}, {StepRow}, 1
            s_sub_i32 {StepT2}, {StepChunk}, {StepChunks}
            s_cmp_ge_u32 {StepChunk}, {StepChunks}
            s_cselect_b32 {StepChunk}, {StepT2}, {StepChunk}
            s_cselect_b32 {StepRow}, {StepT3}, {StepRow}
            s_lshl_b32 {StepT0}, {StepChunk}, 6
            v_add_u32_e32 {Unit}, {StepT0}, {Lane}
            """);

        // The product: wave s sums m = s, s + 4, … of source[n, row, m] · weights[m, g·H + j] per term g.
        var sums = new string[terms];
        for (int g = 0; g < terms; g++)
        {
            sums[g] = w.V();
            CvLines(w, $"v_mov_b32_e32 {sums[g]}, 0");
        }

        string width = product.Gates == 1 ? h : CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {h}, {product.Gates}"));
        var termStrides = new string[terms];
        for (int g = 1; g < terms; g++)
        {
            termStrides[g] = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {h}, {4 * g}"));
        }
        CvLines(w, $"""
            s_mul_i32 {StepWw}, {h}, {product.WeightGates}
            s_lshl_b32 {StepAhead}, {StepWw}, 4
            s_mul_i32 {StepT0}, {StepRow}, {steps}
            s_add_u32 {StepT0}, {StepT0}, {Value(product.Row)}
            s_mul_i32 {StepT1}, {StepT0}, {width}
            s_mul_hi_u32 {RowWriter.Hi(StepSourceBase)}, {StepT0}, {width}
            s_mov_b32 {RowWriter.Lo(StepSourceBase)}, {StepT1}
            s_lshl_b64 {StepSourceBase}, {StepSourceBase}, 2
            s_add_u32 {RowWriter.Lo(StepSourceBase)}, {RowWriter.Lo(Pointer(product.Source))}, {RowWriter.Lo(StepSourceBase)}
            s_addc_u32 {RowWriter.Hi(StepSourceBase)}, {RowWriter.Hi(Pointer(product.Source))}, {RowWriter.Hi(StepSourceBase)}
            s_bitcmp1_b32 {flags}, 2
            s_cbranch_scc0 {label}_product_end
            v_cmp_gt_u32_e32 vcc, {h}, {Unit}
            s_and_saveexec_b64 {StepSaved}, vcc
            s_cbranch_execz {label}_product_out
            s_mov_b32 {StepM}, {StepSlice}
            {label}_pair:
            s_add_u32 {StepT0}, {StepM}, {StepSlices}
            s_cmp_ge_u32 {StepT0}, {width}
            s_cbranch_scc1 {label}_one
            s_lshl_b32 {StepT1}, {StepM}, 2
            s_add_u32 {RowWriter.Lo(StepAddress)}, {RowWriter.Lo(StepSourceBase)}, {StepT1}
            s_addc_u32 {RowWriter.Hi(StepAddress)}, {RowWriter.Hi(StepSourceBase)}, 0
            s_load_dword {StepA}, {StepAddress}, 0x0
            s_load_dword {StepB}, {StepAddress}, 0x{4 * StepSlices:x}
            s_mul_i32 {StepMw}, {StepM}, {StepWw}
            v_add_u32_e32 {WeightAt}, {StepMw}, {Unit}
            v_lshlrev_b32_e32 {WeightAt}, 2, {WeightAt}
            v_add_u32_e32 {WeightAhead}, {StepAhead}, {WeightAt}
            """);
        var first = new string[terms];
        var second = new string[terms];
        for (int g = 0; g < terms; g++)
        {
            first[g] = w.V();
            second[g] = w.V();
            string at = w.V(), ahead = w.V();
            CvLines(w, g == 0
                ? (FormattableString)$"""
                    global_load_dword {first[g]}, {WeightAt}, {Pointer(product.Weights)}
                    global_load_dword {second[g]}, {WeightAhead}, {Pointer(product.Weights)}
                    """
                : $"""
                    v_add_u32_e32 {at}, {termStrides[g]}, {WeightAt}
                    v_add_u32_e32 {ahead}, {termStrides[g]}, {WeightAhead}
                    global_load_dword {first[g]}, {at}, {Pointer(product.Weights)}
                    global_load_dword {second[g]}, {ahead}, {Pointer(product.Weights)}
                    """);
        }

        for (int g = 0; g < terms; g++)
        {
            CvLines(w, $"""
                v_fma_f32 {sums[g]}, {StepA}, {first[g]}, {sums[g]}
                v_fma_f32 {sums[g]}, {StepB}, {second[g]}, {sums[g]}
                """);
        }

        CvLines(w, $"""
            s_add_u32 {StepM}, {StepM}, {2 * StepSlices}
            s_branch {label}_pair
            {label}_one:
            s_cmp_ge_u32 {StepM}, {width}
            s_cbranch_scc1 {label}_product_out
            s_lshl_b32 {StepT1}, {StepM}, 2
            s_add_u32 {RowWriter.Lo(StepAddress)}, {RowWriter.Lo(StepSourceBase)}, {StepT1}
            s_addc_u32 {RowWriter.Hi(StepAddress)}, {RowWriter.Hi(StepSourceBase)}, 0
            s_load_dword {StepA}, {StepAddress}, 0x0
            s_mul_i32 {StepMw}, {StepM}, {StepWw}
            v_add_u32_e32 {WeightAt}, {StepMw}, {Unit}
            v_lshlrev_b32_e32 {WeightAt}, 2, {WeightAt}
            """);
        for (int g = 0; g < terms; g++)
        {
            string value = w.V(), at = w.V();
            CvLines(w, g == 0
                ? (FormattableString)$"global_load_dword {value}, {WeightAt}, {Pointer(product.Weights)}"
                : $"""
                    v_add_u32_e32 {at}, {termStrides[g]}, {WeightAt}
                    global_load_dword {value}, {at}, {Pointer(product.Weights)}
                    """);
            CvLines(w, $"v_fma_f32 {sums[g]}, {StepA}, {value}, {sums[g]}");
        }

        // The slices' sums meet in LDS ([slice][term][64 lanes]); the first wave adds them in slice order.
        CvLines(w, $"""
            {label}_product_out:
            s_mov_b64 exec, {StepSaved}
            {label}_product_end:
            s_mul_i32 {StepT0}, {StepSlice}, {terms * StepUnits * 4}
            v_add_u32_e32 {PartAt}, {StepT0}, {Lane4}
            """);
        for (int g = 0; g < terms; g++)
        {
            CvLines(w, $"ds_write_b32 {PartAt}, {sums[g]}{(g == 0 ? "" : TrainingText($" offset:{g * StepUnits * 4}"))}");
        }

        CvLines(w, $"""
            s_barrier
            s_cmp_lg_u32 {StepSlice}, 0
            s_cbranch_scc1 {label}_done
            v_cmp_gt_u32_e32 vcc, {h}, {Unit}
            s_and_saveexec_b64 {StepSaved}, vcc
            s_cbranch_execz {label}_done
            """);
        var totals = new string[terms];
        for (int g = 0; g < terms; g++)
        {
            totals[g] = w.V();
            for (int s = 0; s < StepSlices; s++)
            {
                string part = s == 0 ? totals[g] : w.V();
                int offset = (s * terms + g) * StepUnits * 4;
                CvLines(w, $"ds_read_b32 {part}, {Lane4}{(offset == 0 ? "" : TrainingText($" offset:{offset}"))}");
                if (s > 0)
                {
                    CvLines(w, $"v_add_f32_e32 {totals[g]}, {totals[g]}, {part}");
                }
            }
        }

        string n = w.V();
        CvLines(w, $"v_mov_b32_e32 {n}, {StepRow}");
        var context = Context(w, label + "_b", buffers, pointers, values, registers, gates, n, Unit);
        body(w, context, totals);
        CvLines(w, $"""
            {label}_done:
              s_endpgm
            """);

        // The head: directives, the arguments, then the held lines.
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $".kernel {name}\n");
        foreach (string buffer in buffers)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {buffer}, 8, global\n");
        }

        foreach (string value in values)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {value}, 4, value\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"""
            .reqd_workgroup_size {RowThreads}, 1, 1
            .lds {StepSlices * terms * StepUnits * 4}
            .user_sgpr kernarg_segment_ptr
            .system_sgpr workgroup_id_x

            """);
        for (int at = 0; at < dwords;)
        {
            int size = dwords - at >= 8 ? 8 : dwords - at >= 4 ? 4 : dwords - at >= 2 ? 2 : 1;
            text.Append(size == 1
                ? TrainingText($"  s_load_dword %s{at}, s[0:1], 0x{4 * at:x}\n")
                : TrainingText($"  s_load_dwordx{size} %s[{at}:{at + size - 1}], s[0:1], 0x{4 * at:x}\n"));
            at += size;
        }

        text.Append(w.Held).Append(body1);
        return new GcnKernelSource(name, Encoding.UTF8.GetBytes(text.ToString()));
    }
}
