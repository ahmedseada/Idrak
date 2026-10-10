// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for one step of an LSTM or GRU cell and its gradient (Backend.LstmCell, GruCell and their backward kernels): one
// thread per (row n, unit j) of the step, i = n·H + j, reading the unit's gate values (a stride of H apart in the row),
// with the activations of sigmoid_f32 and tanh_f32 (σ(x) = 1 / (1 + 2^(-x·log2 e)), tanh(x) = 1 - 2 / (2^(2x·log2 e) +
// 1)) and every product and sum rounded on its own (.rn, no contraction), as the composed element-wise kernels round
// them. Sequences are [batch, steps, width] row-major, the step's row of sequence n at n·steps + step; states [batch, H].
// The optional storages are passed as any valid pointer with their flag bit clear. Parameters: the storages, then step,
// (previous,) steps, h, flags, and n = batch·H.
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] RecurrentNames = ["lstm_cell_f32", "lstm_cell_bwd_f32", "gru_cell_f32", "gru_cell_bwd_f32"];

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
    }

    // %r1 = n, %r2 = j, %r3 = the step's row n·steps + step, %r4 = the width G·H, %r5 = the row's gate index row·G·H + j,
    // %r6 = the state's gate index n·G·H + j, %r7 = the row's unit index row·H + j, %rd3 = H·4 (a gate's stride in bytes).
    private static string RecurrentIndices(int gates) => $"""
        div.u32 %r1, %i, %s_h;
        rem.u32 %r2, %i, %s_h;
        mad.lo.u32 %r3, %r1, %s_steps, %s_step;
        mul.lo.u32 %r4, %s_h, {gates};
        mad.lo.u32 %r5, %r3, %r4, %r2;
        mad.lo.u32 %r6, %r1, %r4, %r2;
        mad.lo.u32 %r7, %r3, %s_h, %r2;
        mul.wide.u32 %rd3, %s_h, 4;
        """;

    // The row's index in a sequence of `gates`·H wide rows into `target`: row·gates·H + j.
    private static string GateIndex(int gates, string target) => $"""
        mul.lo.u32 %r9, %s_h, {gates};
        mad.lo.u32 {target}, %r3, %r9, %r2;
        """;

    // projected[gate] + recurrent[gate] into `target` (the gate's values at %rd1 and %rd2 plus gate·H; uses %rd7, %rd8, %f20).
    private static string GatePairSum(string target, int gate) => $"""
        mul.lo.u32 %r11, %s_h, {gate * 4};
        cvt.u64.u32 %rd7, %r11;
        add.u64 %rd8, %rd1, %rd7;
        ld.global.f32 {target}, [%rd8];
        add.u64 %rd8, %rd2, %rd7;
        ld.global.f32 %f20, [%rd8];
        add.rn.f32 {target}, {target}, %f20;
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

    // The previous step's value of unit j, sequence[n·steps + previous, j], into `target`; 0 without one (flag 2; uses %r12, %p3, %rd10).
    private static string PreviousRow(string target, string sequence) => $"""
        mov.f32 {target}, {Zero};
        and.b32 %r10, %s_flags, 2;
        setp.ne.u32 %p3, %r10, 0;
        mad.lo.u32 %r12, %r1, %s_steps, %s_previous;
        mad.lo.u32 %r12, %r12, %s_h, %r2;
        mul.wide.u32 %rd10, %r12, 4;
        add.u64 %rd10, {sequence}, %rd10;
        @%p3 ld.global.f32 {target}, [%rd10];
        """;

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
