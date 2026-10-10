// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for one step of an LSTM or GRU cell and its gradient (Backend.LstmCell, GruCell and their backward kernels): one
// thread per (row n, unit j) of the step, i = n·H + j, reading the unit's gate values (a stride of H apart in the row),
// with the activations of sigmoid_f32 and tanh_f32 (σ(x) = 1 / (1 + 2^(-x·log2 e)), tanh(x) = 1 - 2 / (2^(2x·log2 e) +
// 1)) and every product and sum rounded on its own (.rn, no contraction), as the composed element-wise kernels round
// them. Sequences are [batch, steps, width] row-major, the step's row of sequence n at n·steps + step; states [batch, H].
// The optional storages are passed as any valid pointer with their flag bit clear. Parameters: the storages, then step,
// (previous,) steps, h, flags, and n = batch·H.
//
// The step kernels (Backend.LstmStep, GruStep and their backward kernels) take the recurrent product inside, so a time
// step is one launch: a block per (row n, 32 units), 32 lanes (units j) by S slices of the product's sum, S = the block's
// warps (up to MaxStepSlices; the launch takes as many as the device's reported block size allows). Slice s takes m = s,
// s + S, …, two m a pass with every value in a register of its own; the weights are read along j, coalesced. The slices'
// sums meet in shared memory and are added in slice order (warp g adds term g), then the
// first warp runs the cell's arithmetic, the same as the cell kernels'. The forward product reads the previous hidden
// state from the output sequence (another row, so no thread writes what another reads) against U [H, G·H]; the gradient
// reads the later step's gate gradient against Uᵀ [G·H, H].
//
// The sequence kernels (Backend.LstmSequence, GruSequence and their gradients) run the step kernels' product and body for
// every step in one launch (RecurrentSequence below): the blocks take the step's (row, 32 units) items in turn, then meet
// at a grid barrier before the next step.
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] RecurrentNames = ["lstm_cell_f32", "lstm_cell_bwd_f32", "gru_cell_f32", "gru_cell_bwd_f32", "lstm_step_f32",
        "lstm_step_bwd_f32", "gru_step_f32", "gru_step_bwd_f32", "lstm_seq_f32", "lstm_seq_bwd_f32", "gru_seq_f32", "gru_seq_bwd_f32"];

    /// <summary>The most slices of the recurrent product's sum a step kernel's block takes (a warp each, 32 units a warp).</summary>
    public const int MaxStepSlices = 32;

    /// <summary>The sequence kernels' flag bit: the steps taken from the last to the first.</summary>
    public const int SequenceReverse = 256;

    private static readonly (string, string)[] StepScalars = [("u32", "step"), ("u32", "previous"), ("u32", "steps"), ("u32", "h"), ("u32", "flags")];

    private static readonly (string, string)[] StepBackwardScalars =
        [("u32", "step"), ("u32", "next"), ("u32", "previous"), ("u32", "steps"), ("u32", "h"), ("u32", "flags")];

    private static readonly (string, string)[] RecurrentScalars = [("u32", "step"), ("u32", "steps"), ("u32", "h"), ("u32", "flags")];

    private static readonly (string, string)[] RecurrentBackwardScalars = [("u32", "step"), ("u32", "previous"), ("u32", "steps"), ("u32", "h"), ("u32", "flags")];

    private static void BuildRecurrent(StringBuilder sb)
    {
        // flags: 1 = save the gates, 2 = save the cells.
        Elementwise(sb, "lstm_cell_f32", ["projected", "recurrent", "cell", "hidden", "output", "gates", "cells"], RecurrentScalars,
            $"""
            {RecurrentIndices(4)}
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_projected, %rd1;
            mul.wide.u32 %rd2, %r6, 4;
            add.u64 %rd2, %b_recurrent, %rd2;
            {GatePairSum("%f1", 0)}
            {GatePairSum("%f2", 1)}
            {GatePairSum("%f3", 2)}
            {GatePairSum("%f4", 3)}
            {Sigmoid("%f5", "%f1")}
            {Sigmoid("%f6", "%f2")}
            {Tanh("%f7", "%f3")}
            {Sigmoid("%f8", "%f4")}
            ld.global.f32 %f9, [%a_cell];
            mul.rn.f32 %f10, %f6, %f9;
            mul.rn.f32 %f11, %f5, %f7;
            add.rn.f32 %f10, %f10, %f11;
            {Tanh("%f12", "%f10")}
            mul.rn.f32 %f13, %f8, %f12;
            st.global.f32 [%a_cell], %f10;
            st.global.f32 [%a_hidden], %f13;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_output, %rd4;
            st.global.f32 [%rd5], %f13;
            and.b32 %r10, %s_flags, 2;
            setp.ne.u32 %p1, %r10, 0;
            add.u64 %rd5, %b_cells, %rd4;
            @%p1 st.global.f32 [%rd5], %f10;
            and.b32 %r10, %s_flags, 1;
            setp.eq.u32 %p1, %r10, 0;
            @%p1 bra DONE;
            mul.wide.u32 %rd6, %r5, 4;
            add.u64 %rd6, %b_gates, %rd6;
            st.global.f32 [%rd6], %f5;
            add.u64 %rd6, %rd6, %rd3;
            st.global.f32 [%rd6], %f6;
            add.u64 %rd6, %rd6, %rd3;
            st.global.f32 [%rd6], %f7;
            add.u64 %rd6, %rd6, %rd3;
            st.global.f32 [%rd6], %f8;
            """);

        // flags: 1 = dOutput given, 2 = a previous step.
        Elementwise(sb, "lstm_cell_bwd_f32", ["gates", "cells", "doutput", "dhidden", "dcell", "dgates", "dstep"], RecurrentBackwardScalars,
            $"""
            {RecurrentIndices(4)}
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_gates, %rd1;
            ld.global.f32 %f1, [%rd1];
            add.u64 %rd2, %rd1, %rd3;
            ld.global.f32 %f2, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f3, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f4, [%rd2];
            {IncomingHidden("%f5")}
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_cells, %rd4;
            ld.global.f32 %f6, [%rd5];
            {Tanh("%f7", "%f6")}
            mul.rn.f32 %f8, %f7, %f7;
            sub.rn.f32 %f8, {One}, %f8;
            mul.rn.f32 %f9, %f5, %f4;
            mul.rn.f32 %f9, %f9, %f8;
            ld.global.f32 %f10, [%a_dcell];
            add.rn.f32 %f10, %f10, %f9;
            {PreviousRow("%f11", "%b_cells")}
            sub.rn.f32 %f12, {One}, %f1;
            mul.rn.f32 %f12, %f1, %f12;
            mul.rn.f32 %f13, %f10, %f3;
            mul.rn.f32 %f13, %f13, %f12;
            sub.rn.f32 %f12, {One}, %f2;
            mul.rn.f32 %f12, %f2, %f12;
            mul.rn.f32 %f14, %f10, %f11;
            mul.rn.f32 %f14, %f14, %f12;
            mul.rn.f32 %f12, %f3, %f3;
            sub.rn.f32 %f12, {One}, %f12;
            mul.rn.f32 %f15, %f10, %f1;
            mul.rn.f32 %f15, %f15, %f12;
            sub.rn.f32 %f12, {One}, %f4;
            mul.rn.f32 %f12, %f4, %f12;
            mul.rn.f32 %f16, %f5, %f7;
            mul.rn.f32 %f16, %f16, %f12;
            mul.rn.f32 %f17, %f10, %f2;
            st.global.f32 [%a_dcell], %f17;
            {StoreGates("%b_dgates", "%r5", ["%f13", "%f14", "%f15", "%f16"])}
            {StoreGates("%b_dstep", "%r6", ["%f13", "%f14", "%f15", "%f16"])}
            """);

        // flags: 1 = the candidate bias given, 2 = save the gates (r, u, c and a, a [batch, steps, 4H] sequence).
        Elementwise(sb, "gru_cell_f32", ["projected", "recurrent", "hbias", "hidden", "output", "gates"], RecurrentScalars,
            $"""
            {RecurrentIndices(3)}
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_projected, %rd1;
            mul.wide.u32 %rd2, %r6, 4;
            add.u64 %rd2, %b_recurrent, %rd2;
            {GatePairSum("%f1", 0)}
            {GatePairSum("%f2", 1)}
            {Sigmoid("%f5", "%f1")}
            {Sigmoid("%f6", "%f2")}
            add.u64 %rd7, %rd1, %rd3;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f3, [%rd7];
            add.u64 %rd7, %rd2, %rd3;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f4, [%rd7];
            and.b32 %r10, %s_flags, 1;
            setp.ne.u32 %p1, %r10, 0;
            mul.wide.u32 %rd7, %r2, 4;
            add.u64 %rd7, %b_hbias, %rd7;
            @%p1 ld.global.f32 %f9, [%rd7];
            @%p1 add.rn.f32 %f4, %f4, %f9;
            mul.rn.f32 %f10, %f5, %f4;
            add.rn.f32 %f10, %f3, %f10;
            {Tanh("%f7", "%f10")}
            ld.global.f32 %f11, [%a_hidden];
            sub.rn.f32 %f12, {One}, %f6;
            mul.rn.f32 %f12, %f12, %f7;
            mul.rn.f32 %f13, %f6, %f11;
            add.rn.f32 %f12, %f12, %f13;
            st.global.f32 [%a_hidden], %f12;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_output, %rd4;
            st.global.f32 [%rd5], %f12;
            and.b32 %r10, %s_flags, 2;
            setp.eq.u32 %p1, %r10, 0;
            @%p1 bra DONE;
            {GateIndex(4, "%r8")}
            {StoreGates("%b_gates", "%r8", ["%f5", "%f6", "%f7", "%f4"])}
            """);

        // flags: 1 = dOutput given, 2 = a previous step.
        Elementwise(sb, "gru_cell_bwd_f32", ["gates", "output", "doutput", "dhidden", "dgates", "drecurrent", "dstep"], RecurrentBackwardScalars,
            $"""
            {RecurrentIndices(3)}
            {GateIndex(4, "%r8")}
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd1, %b_gates, %rd1;
            ld.global.f32 %f1, [%rd1];
            add.u64 %rd2, %rd1, %rd3;
            ld.global.f32 %f2, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f3, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f4, [%rd2];
            {IncomingHidden("%f5")}
            {PreviousRow("%f11", "%b_output")}
            sub.rn.f32 %f12, {One}, %f2;
            mul.rn.f32 %f13, %f3, %f3;
            sub.rn.f32 %f13, {One}, %f13;
            mul.rn.f32 %f14, %f5, %f12;
            mul.rn.f32 %f14, %f14, %f13;
            sub.rn.f32 %f13, {One}, %f1;
            mul.rn.f32 %f13, %f1, %f13;
            mul.rn.f32 %f15, %f14, %f4;
            mul.rn.f32 %f15, %f15, %f13;
            sub.rn.f32 %f16, %f11, %f3;
            mul.rn.f32 %f16, %f5, %f16;
            mul.rn.f32 %f13, %f2, %f12;
            mul.rn.f32 %f16, %f16, %f13;
            mul.rn.f32 %f17, %f14, %f1;
            mul.rn.f32 %f18, %f5, %f2;
            st.global.f32 [%a_dhidden], %f18;
            {StoreGates("%b_dgates", "%r5", ["%f15", "%f16", "%f14"])}
            {StoreGates("%b_drecurrent", "%r5", ["%f15", "%f16", "%f17"])}
            {StoreGates("%b_dstep", "%r6", ["%f15", "%f16", "%f17"])}
            """);

        // flags: 1 = save the gates, 2 = save the cells, 4 = a previous step (the product; else the zero initial state).
        StepAndSequence(sb, "lstm_step_f32", "lstm_seq_f32", ["projected", "weights", "cell", "output", "gates", "cells"], false, 4, 4,
            ("%b_output", "%s_previous", "%s_h", "%r4", "%b_weights"),
            $"""
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_projected, %rd1;
            {ProjectedPlus("%f1", 0, "%f30")}
            {ProjectedPlus("%f2", 1, "%f31")}
            {ProjectedPlus("%f3", 2, "%f32")}
            {ProjectedPlus("%f4", 3, "%f33")}
            {Sigmoid("%f5", "%f1")}
            {Sigmoid("%f6", "%f2")}
            {Tanh("%f7", "%f3")}
            {Sigmoid("%f8", "%f4")}
            ld.global.f32 %f9, [%a_cell];
            mul.rn.f32 %f10, %f6, %f9;
            mul.rn.f32 %f11, %f5, %f7;
            add.rn.f32 %f10, %f10, %f11;
            {Tanh("%f12", "%f10")}
            mul.rn.f32 %f13, %f8, %f12;
            st.global.f32 [%a_cell], %f10;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_output, %rd4;
            st.global.f32 [%rd5], %f13;
            and.b32 %r10, %s_flags, 2;
            setp.ne.u32 %p1, %r10, 0;
            add.u64 %rd5, %b_cells, %rd4;
            @%p1 st.global.f32 [%rd5], %f10;
            and.b32 %r10, %s_flags, 1;
            setp.eq.u32 %p1, %r10, 0;
            @%p1 bra DONE;
            {StoreGates("%b_gates", "%r5", ["%f5", "%f6", "%f7", "%f8"])}
            """);

        // flags: 1 = dOutput given, 2 = a previous step, 4 = a later step (the product of its gate gradient; else dh's
        // recurrent part is 0).
        StepAndSequence(sb, "lstm_step_bwd_f32", "lstm_seq_bwd_f32", ["gates", "cells", "doutput", "weightst", "dcell", "dgates"], true, 4, 1,
            ("%b_dgates", "%s_next", "%r4", "%s_h", "%b_weightst"),
            $"""
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_gates, %rd1;
            ld.global.f32 %f1, [%rd1];
            add.u64 %rd2, %rd1, %rd3;
            ld.global.f32 %f2, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f3, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f4, [%rd2];
            {IncomingOutput("%f5", "%f30")}
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_cells, %rd4;
            ld.global.f32 %f6, [%rd5];
            {Tanh("%f7", "%f6")}
            mul.rn.f32 %f8, %f7, %f7;
            sub.rn.f32 %f8, {One}, %f8;
            mul.rn.f32 %f9, %f5, %f4;
            mul.rn.f32 %f9, %f9, %f8;
            ld.global.f32 %f10, [%a_dcell];
            add.rn.f32 %f10, %f10, %f9;
            {PreviousRow("%f11", "%b_cells")}
            sub.rn.f32 %f12, {One}, %f1;
            mul.rn.f32 %f12, %f1, %f12;
            mul.rn.f32 %f13, %f10, %f3;
            mul.rn.f32 %f13, %f13, %f12;
            sub.rn.f32 %f12, {One}, %f2;
            mul.rn.f32 %f12, %f2, %f12;
            mul.rn.f32 %f14, %f10, %f11;
            mul.rn.f32 %f14, %f14, %f12;
            mul.rn.f32 %f12, %f3, %f3;
            sub.rn.f32 %f12, {One}, %f12;
            mul.rn.f32 %f15, %f10, %f1;
            mul.rn.f32 %f15, %f15, %f12;
            sub.rn.f32 %f12, {One}, %f4;
            mul.rn.f32 %f12, %f4, %f12;
            mul.rn.f32 %f16, %f5, %f7;
            mul.rn.f32 %f16, %f16, %f12;
            mul.rn.f32 %f17, %f10, %f2;
            st.global.f32 [%a_dcell], %f17;
            {StoreGates("%b_dgates", "%r5", ["%f13", "%f14", "%f15", "%f16"])}
            """);

        // flags: 1 = the candidate bias given, 2 = save the gates, 4 = a previous step.
        StepAndSequence(sb, "gru_step_f32", "gru_seq_f32", ["projected", "weights", "hbias", "output", "gates"], false, 3, 3,
            ("%b_output", "%s_previous", "%s_h", "%r4", "%b_weights"),
            $"""
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_projected, %rd1;
            {ProjectedPlus("%f1", 0, "%f30")}
            {ProjectedPlus("%f2", 1, "%f31")}
            {Sigmoid("%f5", "%f1")}
            {Sigmoid("%f6", "%f2")}
            add.u64 %rd7, %rd1, %rd3;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f3, [%rd7];
            mov.f32 %f4, %f32;
            and.b32 %r10, %s_flags, 1;
            setp.ne.u32 %p1, %r10, 0;
            mul.wide.u32 %rd7, %r2, 4;
            add.u64 %rd7, %b_hbias, %rd7;
            @%p1 ld.global.f32 %f9, [%rd7];
            @%p1 add.rn.f32 %f4, %f4, %f9;
            mul.rn.f32 %f10, %f5, %f4;
            add.rn.f32 %f10, %f3, %f10;
            {Tanh("%f7", "%f10")}
            {PreviousRow("%f11", "%b_output", 4)}
            sub.rn.f32 %f12, {One}, %f6;
            mul.rn.f32 %f12, %f12, %f7;
            mul.rn.f32 %f13, %f6, %f11;
            add.rn.f32 %f12, %f12, %f13;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd5, %b_output, %rd4;
            st.global.f32 [%rd5], %f12;
            and.b32 %r10, %s_flags, 2;
            setp.eq.u32 %p1, %r10, 0;
            @%p1 bra DONE;
            {GateIndex(4, "%r8")}
            {StoreGates("%b_gates", "%r8", ["%f5", "%f6", "%f7", "%f4"])}
            """);

        // flags: 1 = dOutput given, 2 = a previous step, 4 = a later step (the product of its recurrent gradient).
        StepAndSequence(sb, "gru_step_bwd_f32", "gru_seq_bwd_f32", ["gates", "output", "doutput", "weightst", "dhidden", "dgates", "drecurrent"], true, 3, 1,
            ("%b_drecurrent", "%s_next", "%r4", "%s_h", "%b_weightst"),
            $"""
            {GateIndex(4, "%r8")}
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd1, %b_gates, %rd1;
            ld.global.f32 %f1, [%rd1];
            add.u64 %rd2, %rd1, %rd3;
            ld.global.f32 %f2, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f3, [%rd2];
            add.u64 %rd2, %rd2, %rd3;
            ld.global.f32 %f4, [%rd2];
            ld.global.f32 %f5, [%a_dhidden];
            add.rn.f32 %f30, %f30, %f5;
            {IncomingOutput("%f5", "%f30")}
            {PreviousRow("%f11", "%b_output")}
            sub.rn.f32 %f12, {One}, %f2;
            mul.rn.f32 %f13, %f3, %f3;
            sub.rn.f32 %f13, {One}, %f13;
            mul.rn.f32 %f14, %f5, %f12;
            mul.rn.f32 %f14, %f14, %f13;
            sub.rn.f32 %f13, {One}, %f1;
            mul.rn.f32 %f13, %f1, %f13;
            mul.rn.f32 %f15, %f14, %f4;
            mul.rn.f32 %f15, %f15, %f13;
            sub.rn.f32 %f16, %f11, %f3;
            mul.rn.f32 %f16, %f5, %f16;
            mul.rn.f32 %f13, %f2, %f12;
            mul.rn.f32 %f16, %f16, %f13;
            mul.rn.f32 %f17, %f14, %f1;
            mul.rn.f32 %f18, %f5, %f2;
            st.global.f32 [%a_dhidden], %f18;
            {StoreGates("%b_dgates", "%r5", ["%f15", "%f16", "%f14"])}
            {StoreGates("%b_drecurrent", "%r5", ["%f15", "%f16", "%f17"])}
            """);
    }

    // A step kernel (`gates` = G: 4 for an LSTM, 3 for a GRU; `terms` sums in its product): the parameters (the
    // storages, then the scalars), the indices of RecurrentIndices with n = the block's row and j = its unit (%p8: j < H;
    // the %a_ pointers at n·H + j), the shared slices, the product (all threads, its barriers included), then `body` in
    // the first warp's valid threads, ending at DONE. 32 · S threads a block for S slices, at most 32 · MaxStepSlices
    // (.maxntid, so ptxas keeps the registers within what such a block may use); grid x = batch · ⌈H / 32⌉ (block b: row
    // b / ⌈H / 32⌉, units 32 · (b mod ⌈H / 32⌉) on).
    private static void RecurrentStep(StringBuilder sb, string name, string[] pointers, (string Type, string Name)[] scalars, int gates, int terms,
        string product, string body)
    {
        var invariant = CultureInfo.InvariantCulture;
        sb.Append(invariant, $".visible .entry {name}(\n");
        for (int k = 0; k < pointers.Length; k++)
        {
            sb.Append(invariant, $"    .param .u64 p_{pointers[k]},\n");
        }

        for (int k = 0; k < scalars.Length; k++)
        {
            sb.Append(invariant, $"    .param .{scalars[k].Type} p_{scalars[k].Name}{(k + 1 < scalars.Length ? "," : "")}\n");
        }

        sb.Append(invariant, $$"""
            )
            .maxntid {{32 * MaxStepSlices}}, 1, 1
            {
                .reg .pred %p<16>;
                .reg .f32 %f<48>;
                .reg .b32 %r<32>;
                .reg .b64 %rd<26>;
                .reg .u32 %i, %lane, %slice, %slices;
                .reg .u64 %off;
                .shared .align 4 .f32 {{name}}_part[{{MaxStepSlices * terms * 32}}];
                .shared .align 4 .f32 {{name}}_sum[{{terms * 32}}];

            """);
        foreach (string pointer in pointers)
        {
            sb.Append(invariant, $"    .reg .u64 %a_{pointer}, %b_{pointer};\n");
        }

        foreach (var (type, scalar) in scalars)
        {
            sb.Append(invariant, $"    .reg .{type} %s_{scalar};\n");
            sb.Append(invariant, $"    ld.param.{type} %s_{scalar}, [p_{scalar}];\n");
        }

        sb.Append("""
                mov.u32 %r20, %tid.x;
                and.b32 %lane, %r20, 31;
                shr.u32 %slice, %r20, 5;
                mov.u32 %slices, %ntid.x;
                shr.u32 %slices, %slices, 5;
                add.u32 %r21, %s_h, 31;
                shr.u32 %r21, %r21, 5;
                mov.u32 %r20, %ctaid.x;
                div.u32 %r1, %r20, %r21;
                rem.u32 %r20, %r20, %r21;
                shl.b32 %r20, %r20, 5;
                add.u32 %r2, %r20, %lane;
                setp.lt.u32 %p8, %r2, %s_h;
                mad.lo.u32 %i, %r1, %s_h, %r2;
                mul.wide.u32 %off, %i, 4;

            """);
        foreach (string pointer in pointers)
        {
            sb.Append(invariant, $"    ld.param.u64 %b_{pointer}, [p_{pointer}];\n");
            sb.Append(invariant, $"    cvta.to.global.u64 %b_{pointer}, %b_{pointer};\n");
            sb.Append(invariant, $"    add.u64 %a_{pointer}, %b_{pointer}, %off;\n");
        }

        AppendIndented(sb, RecurrentRowIndices(gates));
        AppendIndented(sb, product);
        AppendIndented(sb, body);
        sb.Append("""
            DONE:
                ret;
            }

            """);
    }

    // A step kernel and its sequence kernel: the same product (`product`: source, row, width, weights' width, weights; see
    // StepProduct) and `body`, one step a launch and every step in one launch.
    private static void StepAndSequence(StringBuilder sb, string name, string sequence, string[] pointers, bool backward, int gates, int terms,
        (string Source, string Row, string Width, string WeightsWidth, string Weights) product, string body)
    {
        RecurrentStep(sb, name, pointers, backward ? StepBackwardScalars : StepScalars, gates, terms,
            StepProduct(name, terms, product.Source, product.Row, 4, product.Width, product.WeightsWidth, product.Weights, "ld.global.f32"), body);
        RecurrentSequence(sb, sequence, pointers, backward, gates, terms,
            StepProduct(sequence, terms, product.Source, product.Row, 4, product.Width, product.WeightsWidth, product.Weights, "ld.global.cg.f32"), body);
    }

    // A sequence kernel (Backend.LstmSequence, GruSequence and their gradients): the step kernel's work for every step in
    // one launch. Parameters: the storages, the grid barrier (two u32 words: arrivals and generation, both 0 between
    // launches), then steps, h, flags (the step kernel's own bits 1 and 2 as given; SequenceReverse: the steps taken from
    // the last; the bits of the previous and the later step are set here) and batch. The items of a step are the step
    // kernel's blocks (row n, 32 units), item = n·⌈H/32⌉ + chunk; block b takes items b, b + grid, … in turn, so each
    // item's sums are the step kernel's, in its order. Between steps every block meets at a grid barrier: the launch is
    // cooperative (all blocks resident), the host sizes the grid from the occupancy the device reports. The rows other
    // blocks wrote (the previous output, the later gate gradient) are read past the multiprocessor's own cache.
    private static void RecurrentSequence(StringBuilder sb, string name, string[] pointers, bool backward, int gates, int terms, string product, string body)
    {
        var invariant = CultureInfo.InvariantCulture;
        sb.Append(invariant, $".visible .entry {name}(\n");
        foreach (string pointer in pointers)
        {
            sb.Append(invariant, $"    .param .u64 p_{pointer},\n");
        }

        sb.Append(invariant, $$"""
                .param .u64 p_barrier,
                .param .u32 p_steps,
                .param .u32 p_h,
                .param .u32 p_flags,
                .param .u32 p_batch
            )
            .maxntid {{32 * MaxStepSlices}}, 1, 1
            {
                .reg .pred %p<16>;
                .reg .f32 %f<48>;
                .reg .b32 %r<32>;
                .reg .b64 %rd<26>;
                .reg .u32 %i, %lane, %slice, %slices;
                .reg .u32 %s_steps, %s_h, %s_flags, %s_step, %s_previous, %s_next;
                .reg .u32 %base, %batch, %k, %last, %item, %items, %chunks;
                .reg .u64 %off, %barrier;
                .shared .align 4 .f32 {{name}}_part[{{MaxStepSlices * terms * 32}}];
                .shared .align 4 .f32 {{name}}_sum[{{terms * 32}}];

            """);
        foreach (string pointer in pointers)
        {
            sb.Append(invariant, $"    .reg .u64 %a_{pointer}, %b_{pointer};\n");
            sb.Append(invariant, $"    ld.param.u64 %b_{pointer}, [p_{pointer}];\n");
            sb.Append(invariant, $"    cvta.to.global.u64 %b_{pointer}, %b_{pointer};\n");
        }

        sb.Append(invariant, $$"""
                ld.param.u64 %barrier, [p_barrier];
                cvta.to.global.u64 %barrier, %barrier;
                ld.param.u32 %s_steps, [p_steps];
                ld.param.u32 %s_h, [p_h];
                ld.param.u32 %base, [p_flags];
                ld.param.u32 %batch, [p_batch];
                mov.u32 %r20, %tid.x;
                and.b32 %lane, %r20, 31;
                shr.u32 %slice, %r20, 5;
                mov.u32 %slices, %ntid.x;
                shr.u32 %slices, %slices, 5;
                add.u32 %chunks, %s_h, 31;
                shr.u32 %chunks, %chunks, 5;
                mul.lo.u32 %items, %batch, %chunks;
                sub.u32 %last, %s_steps, 1;
                mov.u32 %k, 0;
            STEP:
                and.b32 %r28, %base, {{SequenceReverse}};
                setp.ne.u32 %p13, %r28, 0;
                sub.u32 %r29, %last, %k;

            """);

        // The step taken k-th, its neighbours and flags. Forward: t = k (reverse: T-1-k), the previous step t∓1 when k > 0
        // (flag 4). Backward, the steps from the last taken: t = T-1-k (reverse: k), the previous step t∓1 unless it is the
        // first taken (flag 2), the later step t±1 unless it is the last taken (flag 4).
        AppendIndented(sb, backward
            ? """
              selp.u32 %s_step, %k, %r29, %p13;
              add.u32 %r29, %s_step, 1;
              sub.u32 %r30, %s_step, 1;
              selp.u32 %s_previous, %r29, %r30, %p13;
              selp.u32 %s_next, %r30, %r29, %p13;
              setp.eq.u32 %p14, %k, %last;
              setp.eq.u32 %p15, %k, 0;
              selp.u32 %s_previous, 0, %s_previous, %p14;
              selp.u32 %s_next, 0, %s_next, %p15;
              and.b32 %s_flags, %base, 1;
              or.b32 %r29, %s_flags, 2;
              selp.u32 %s_flags, %s_flags, %r29, %p14;
              or.b32 %r29, %s_flags, 4;
              selp.u32 %s_flags, %s_flags, %r29, %p15;
              """
            : """
              selp.u32 %s_step, %r29, %k, %p13;
              add.u32 %r29, %s_step, 1;
              sub.u32 %r30, %s_step, 1;
              selp.u32 %s_previous, %r29, %r30, %p13;
              mov.u32 %s_next, 0;
              setp.eq.u32 %p14, %k, 0;
              selp.u32 %s_previous, 0, %s_previous, %p14;
              and.b32 %s_flags, %base, 3;
              or.b32 %r29, %s_flags, 4;
              selp.u32 %s_flags, %s_flags, %r29, %p14;
              """);

        // The block's items of this step: the step kernel's block preamble from the item.
        sb.Append("""
                mov.u32 %item, %ctaid.x;
            ITEM:
                setp.ge.u32 %p12, %item, %items;
                @%p12 bra STEP_END;
                div.u32 %r1, %item, %chunks;
                rem.u32 %r20, %item, %chunks;
                shl.b32 %r20, %r20, 5;
                add.u32 %r2, %r20, %lane;
                setp.lt.u32 %p8, %r2, %s_h;
                mad.lo.u32 %i, %r1, %s_h, %r2;
                mul.wide.u32 %off, %i, 4;

            """);
        foreach (string pointer in pointers)
        {
            sb.Append(invariant, $"    add.u64 %a_{pointer}, %b_{pointer}, %off;\n");
        }

        AppendIndented(sb, RecurrentRowIndices(gates));
        AppendIndented(sb, product.Replace("bra DONE;", "bra ITEM_END;", StringComparison.Ordinal));
        AppendIndented(sb, body.Replace("bra DONE;", "bra ITEM_END;", StringComparison.Ordinal));

        // The next item; after the last step's, the end. Between steps the grid barrier: the block's threads meet, its
        // first thread makes the block's writes visible to the device and arrives (the last to arrive clears the count and
        // advances the generation), then waits for the generation to move; the block's threads meet again.
        sb.Append("""
            ITEM_END:
                mov.u32 %r20, %nctaid.x;
                add.u32 %item, %item, %r20;
                bra ITEM;
            STEP_END:
                setp.ge.u32 %p12, %k, %last;
                @%p12 bra EXIT;
                bar.sync 0;
                mov.u32 %r28, %tid.x;
                setp.ne.u32 %p12, %r28, 0;
                @%p12 bra GRID_PASSED;
                membar.gl;
                ld.volatile.global.u32 %r29, [%barrier+4];
                atom.global.add.u32 %r30, [%barrier], 1;
                mov.u32 %r31, %nctaid.x;
                sub.u32 %r31, %r31, 1;
                setp.ne.u32 %p12, %r30, %r31;
                @%p12 bra GRID_WAIT;
                atom.global.exch.b32 %r30, [%barrier], 0;
                membar.gl;
                atom.global.add.u32 %r30, [%barrier+4], 1;
                bra GRID_FENCE;
            GRID_WAIT:
                ld.volatile.global.u32 %r30, [%barrier+4];
                setp.eq.u32 %p12, %r30, %r29;
                @%p12 bra GRID_WAIT;
            GRID_FENCE:
                membar.gl;
            GRID_PASSED:
                bar.sync 0;
                add.u32 %k, %k, 1;
                bra STEP;
            EXIT:
                ret;
            }

            """);
    }

    // Each line of `text` into `sb`, indented four spaces, its trailing spaces dropped (spans: no line strings).
    private static void AppendIndented(StringBuilder sb, string text)
    {
        foreach (var line in text.AsSpan().EnumerateLines())
        {
            sb.Append("    ").Append(line.TrimEnd()).Append('\n');
        }
    }

    // The step's recurrent sums for unit j: term g = Σ_m source[m] · weights[m, g·H + j] over m < `width` (the source row
    // `row` of sequence n, `width` floats wide; the weights `weightsWidth` floats a row; nothing without the flag bit
    // `flag`, every sum 0). Slice s of S (%slices) takes m = s, s + S, …, two m a pass (every value in a register of its
    // own), each term's sum in m order; the slices' sums meet in shared memory and warp g adds term g's in slice order. Then
    // the first warp's valid threads hold term g in %f{30 + g} (the others go to DONE). `load` reads the source (ld.global.cg
    // where other blocks of the same launch wrote it: past the multiprocessor's own cache). Uses %r20 … %r27, %rd16 …
    // %rd25, %f30 … %f47, %p8 … %p11; %rd3 (H·4: a term's stride in the weights) from RecurrentRowIndices.
    private static string StepProduct(string name, int terms, string source, string row, int flag, string width, string weightsWidth, string weights,
        string load)
    {
        var invariant = CultureInfo.InvariantCulture;
        var s = new StringBuilder();
        for (int g = 0; g < terms; g++)
        {
            s.Append(invariant, $"mov.f32 %f{30 + g}, {Zero};\n");
        }

        s.Append(invariant, $"""
            and.b32 %r20, %s_flags, {flag};
            setp.eq.u32 %p9, %r20, 0;
            @%p9 bra PRODUCT_END;
            @!%p8 bra PRODUCT_END;
            mad.lo.u32 %r21, %r1, %s_steps, {row};
            mul.lo.u32 %r21, %r21, {width};
            add.u32 %r21, %r21, %slice;
            mul.wide.u32 %rd16, %r21, 4;
            add.u64 %rd16, {source}, %rd16;
            mad.lo.u32 %r22, %slice, {weightsWidth}, %r2;
            mul.wide.u32 %rd17, %r22, 4;
            add.u64 %rd17, {weights}, %rd17;
            mul.lo.u32 %r27, {weightsWidth}, %slices;
            mul.wide.u32 %rd18, %r27, 4;
            mul.wide.u32 %rd24, %slices, 4;
            mov.u32 %r23, %slice;
            PRODUCT2:
            add.u32 %r24, %r23, %slices;
            setp.ge.u32 %p9, %r24, {width};
            @%p9 bra PRODUCT1;
            {load} %f38, [%rd16];
            add.u64 %rd25, %rd16, %rd24;
            {load} %f39, [%rd25];
            add.u64 %rd19, %rd17, %rd18;

            """);
        for (int g = 0; g < terms; g++)
        {
            string step = g == 0 ? "" : "add.u64 %rd20, %rd20, %rd3;\nadd.u64 %rd21, %rd21, %rd3;\n";
            if (g == 0)
            {
                s.Append("mov.u64 %rd20, %rd17;\nmov.u64 %rd21, %rd19;\n");
            }

            s.Append(step);
            s.Append(invariant, $"ld.global.f32 %f{34 + g}, [%rd20];\nld.global.f32 %f{40 + g}, [%rd21];\n");
        }

        for (int g = 0; g < terms; g++)
        {
            s.Append(invariant, $"fma.rn.f32 %f{30 + g}, %f38, %f{34 + g}, %f{30 + g};\nfma.rn.f32 %f{30 + g}, %f39, %f{40 + g}, %f{30 + g};\n");
        }

        s.Append("""
            add.u64 %rd16, %rd25, %rd24;
            add.u64 %rd17, %rd19, %rd18;
            add.u32 %r23, %r24, %slices;
            bra PRODUCT2;
            PRODUCT1:
            setp.ge.u32 %p9, %r23, %s_width_placeholder;
            @%p9 bra PRODUCT_END;
            %load_placeholder %f38, [%rd16];
            mov.u64 %rd20, %rd17;

            """.Replace("%s_width_placeholder", width, StringComparison.Ordinal).Replace("%load_placeholder", load, StringComparison.Ordinal));
        for (int g = 0; g < terms; g++)
        {
            if (g > 0)
            {
                s.Append("add.u64 %rd20, %rd20, %rd3;\n");
            }

            s.Append(invariant, $"ld.global.f32 %f{34 + g}, [%rd20];\n");
        }

        for (int g = 0; g < terms; g++)
        {
            s.Append(invariant, $"fma.rn.f32 %f{30 + g}, %f38, %f{34 + g}, %f{30 + g};\n");
        }

        s.Append(invariant, $"""
            PRODUCT_END:
            mov.u64 %rd22, {name}_part;
            mad.lo.u32 %r25, %slice, {terms * 32}, %lane;
            mul.wide.u32 %rd23, %r25, 4;
            add.u64 %rd23, %rd22, %rd23;

            """);
        for (int g = 0; g < terms; g++)
        {
            s.Append(invariant, $"st.shared.f32 [%rd23+{g * 128}], %f{30 + g};\n");
        }

        s.Append(invariant, $"""
            bar.sync 0;
            setp.ge.u32 %p10, %slice, {terms};
            @%p10 bra REDUCE_END;
            mad.lo.u32 %r26, %slice, 32, %lane;
            mul.wide.u32 %rd23, %r26, 4;
            add.u64 %rd23, %rd22, %rd23;
            ld.shared.f32 %f44, [%rd23];
            mov.u32 %r27, 1;
            REDUCE:
            setp.ge.u32 %p11, %r27, %slices;
            @%p11 bra REDUCE_STORE;
            add.u64 %rd23, %rd23, {terms * 128};
            ld.shared.f32 %f45, [%rd23];
            add.rn.f32 %f44, %f44, %f45;
            add.u32 %r27, %r27, 1;
            bra REDUCE;
            REDUCE_STORE:
            mov.u64 %rd22, {name}_sum;
            mul.wide.u32 %rd23, %r26, 4;
            add.u64 %rd23, %rd22, %rd23;
            st.shared.f32 [%rd23], %f44;
            REDUCE_END:
            bar.sync 0;
            setp.ne.u32 %p10, %slice, 0;
            @%p10 bra DONE;
            @!%p8 bra DONE;
            mov.u64 %rd22, {name}_sum;
            mul.wide.u32 %rd23, %lane, 4;
            add.u64 %rd23, %rd22, %rd23;

            """);
        for (int g = 0; g < terms; g++)
        {
            s.Append(invariant, $"ld.shared.f32 %f{30 + g}, [%rd23+{g * 128}];\n");
        }

        return s.ToString();
    }

    // %r1 = n, %r2 = j, %r3 = the step's row n·steps + step, %r4 = the width G·H, %r5 = the row's gate index row·G·H + j,
    // %r6 = the state's gate index n·G·H + j, %r7 = the row's unit index row·H + j, %rd3 = H·4 (a gate's stride in bytes).
    private static string RecurrentIndices(int gates) => $"""
        div.u32 %r1, %i, %s_h;
        rem.u32 %r2, %i, %s_h;
        {RecurrentRowIndices(gates)}
        """;

    // RecurrentIndices from %r1 = n and %r2 = j.
    private static string RecurrentRowIndices(int gates) => string.Create(CultureInfo.InvariantCulture, $"""
        mad.lo.u32 %r3, %r1, %s_steps, %s_step;
        mul.lo.u32 %r4, %s_h, {gates};
        mad.lo.u32 %r5, %r3, %r4, %r2;
        mad.lo.u32 %r6, %r1, %r4, %r2;
        mad.lo.u32 %r7, %r3, %s_h, %r2;
        mul.wide.u32 %rd3, %s_h, 4;
        """);

    // The row's index in a sequence of `gates`·H wide rows into `target`: row·gates·H + j.
    private static string GateIndex(int gates, string target) => string.Create(CultureInfo.InvariantCulture, $"""
        mul.lo.u32 %r9, %s_h, {gates};
        mad.lo.u32 {target}, %r3, %r9, %r2;
        """);

    // projected[gate] + recurrent[gate] into `target` (the gate's values at %rd1 and %rd2 plus gate·H; uses %rd7, %rd8, %f20).
    private static string GatePairSum(string target, int gate) => string.Create(CultureInfo.InvariantCulture, $"""
        mul.lo.u32 %r11, %s_h, {gate * 4};
        cvt.u64.u32 %rd7, %r11;
        add.u64 %rd8, %rd1, %rd7;
        ld.global.f32 {target}, [%rd8];
        add.u64 %rd8, %rd2, %rd7;
        ld.global.f32 %f20, [%rd8];
        add.rn.f32 {target}, {target}, %f20;
        """);

    // projected[gate] + `product` into `target` (the step kernels' sum; projected's gate values at %rd1 plus gate·H; uses %rd7, %rd8).
    private static string ProjectedPlus(string target, int gate, string product) => string.Create(CultureInfo.InvariantCulture, $"""
        mul.lo.u32 %r11, %s_h, {gate * 4};
        cvt.u64.u32 %rd7, %r11;
        add.u64 %rd8, %rd1, %rd7;
        ld.global.f32 {target}, [%rd8];
        add.rn.f32 {target}, {target}, {product};
        """);

    // dh = `product` + dOutput[row, j] (flag 1) into `target`: the step kernels' incoming gradient (uses %r10, %p2, %rd9, %f21).
    private static string IncomingOutput(string target, string product) => $"""
        mov.f32 {target}, {product};
        and.b32 %r10, %s_flags, 1;
        setp.ne.u32 %p2, %r10, 0;
        mul.wide.u32 %rd9, %r7, 4;
        add.u64 %rd9, %b_doutput, %rd9;
        @%p2 ld.global.f32 %f21, [%rd9];
        @%p2 add.rn.f32 {target}, {target}, %f21;
        """;

    // dh = dHidden[n, j] + dOutput[row, j] (flag 1) into `target` (uses %r10, %p2, %rd9, %f21).
    private static string IncomingHidden(string target) => $"""
        ld.global.f32 {target}, [%a_dhidden];
        and.b32 %r10, %s_flags, 1;
        setp.ne.u32 %p2, %r10, 0;
        mul.wide.u32 %rd9, %r7, 4;
        add.u64 %rd9, %b_doutput, %rd9;
        @%p2 ld.global.f32 %f21, [%rd9];
        @%p2 add.rn.f32 {target}, {target}, %f21;
        """;

    // The previous step's value of unit j, sequence[n·steps + previous, j], into `target`; 0 without one (flag bit `flag`;
    // uses %r12, %p3, %rd10).
    private static string PreviousRow(string target, string sequence, int flag = 2) => string.Create(CultureInfo.InvariantCulture, $"""
        mov.f32 {target}, {Zero};
        and.b32 %r10, %s_flags, {flag};
        setp.ne.u32 %p3, %r10, 0;
        mad.lo.u32 %r12, %r1, %s_steps, %s_previous;
        mad.lo.u32 %r12, %r12, %s_h, %r2;
        mul.wide.u32 %rd10, %r12, 4;
        add.u64 %rd10, {sequence}, %rd10;
        @%p3 ld.global.f32 {target}, [%rd10];
        """);

    // Stores the values one gate stride apart from index `index` of the storage at `pointer` (uses %rd11).
    private static string StoreGates(string pointer, string index, string[] values)
    {
        var s = new StringBuilder();
        s.AppendLine($"mul.wide.u32 %rd11, {index}, 4;");
        s.AppendLine($"add.u64 %rd11, {pointer}, %rd11;");
        for (int k = 0; k < values.Length; k++)
        {
            if (k > 0)
            {
                s.AppendLine("add.u64 %rd11, %rd11, %rd3;");
            }

            s.AppendLine($"st.global.f32 [%rd11], {values[k]};");
        }

        return s.ToString().TrimEnd();
    }

    // σ(x) as sigmoid_f32 computes it (uses %f22).
    private static string Sigmoid(string target, string x) => $"""
        mul.f32 %f22, {x}, {F(-1.4426950408889634f)};
        ex2.approx.ftz.f32 %f22, %f22;
        add.f32 %f22, %f22, {One};
        rcp.rn.f32 {target}, %f22;
        """;

    // tanh(x) as tanh_f32 computes it (uses %f22).
    private static string Tanh(string target, string x) => $"""
        mul.f32 %f22, {x}, {F(2.8853900817779268f)};
        ex2.approx.ftz.f32 %f22, %f22;
        add.f32 %f22, %f22, {One};
        rcp.rn.f32 %f22, %f22;
        fma.rn.f32 {target}, %f22, {F(-2f)}, {One};
        """;
}
