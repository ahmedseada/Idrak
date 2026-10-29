// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's element-wise kernels in GCN (plan 35, G3a), with CUDA's names (PtxKernels.cs, PtxKernels.Advanced.cs,
// PtxKernels.Pointwise.cs) and CUDA's arguments in CUDA's order (the buffers, the values, then the count n), computed as
// the CPU computes them (CpuBackend*.cs, CpuMath.cs): the reference their results are proven against.
//
// Every kernel has one shape. Blocks of PointwiseBlock work-items, the grid any size the backend bounds it to (rule 81):
// the dispatch packet's grid size (grid_size_x, in work-items) is read once, and both loops stride by it.
// - The main loop: chunks of 256 elements, one a wave, 4 consecutive floats a lane (one global_load_dwordx4 per buffer
//   read, a register each, one global_store_dwordx4), coalesced; chunk c goes to the grid's wave c mod waves. It runs
//   with every lane on, as long as the wave's chunk lies wholly below n rounded down to 4, under a scalar loop, so no
//   value is written lane by lane. It runs only when every buffer it moves 4 at a time is 16-byte aligned (checked
//   once, on the scalar unit); otherwise no chunk is whole and the tail takes every element.
// - The tail: what the chunks leave (fewer than 260 elements when aligned), one float a work-item, grid-stride.
// The count stays below 2^30 (byte offsets are 32-bit) and the grid's work-items below 2^29 (so a stride added to an
// index never wraps): PointwiseMostElements, PointwiseMostWorkItems.
//
// Arithmetic as the CPU rounds it: adds, products and fused multiply-adds where the CPU takes them; float32 denormals
// kept (the code object's mode). The functions: LLVM 18's own sequences for amdgcn, read from llc's output (-mcpu
// gfx90c, 2026-10-29): e^x (x·log2 e split in two floats, v_exp_f32 on the reduced part, v_ldexp_f32, 0 below
// -103.28 and ∞ above 88.72), ln x (v_log_f32 with denormals scaled by 2^32 first, the product by ln 2 split in two
// floats), √x (v_sqrt_f32 corrected by one ulp either way through two fused residues, small values scaled by 2^32),
// the correctly rounded division (v_div_scale, v_rcp_f32, fused Newton steps, v_div_fmas, v_div_fixup) and the
// unsigned remainder (v_rcp_iflag_f32 and two corrections). LLVM lowers sin and cos to the hardware's v_sin_f32 alone
// (accurate only near zero); the device library's sinf is not here, so sin and cos are CUDA's: quarter turns taken off
// by Cody-Waite in three floats, minimax polynomials on [-π/4, π/4], the quadrant picking ±sin or ±cos. tanh is the
// CPU's own rational (TanhRational), bit for bit with LLVM's division; sigmoid, SiLU and both GELUs follow the CPU's
// formulas over these. max and min are MathF.Max and MathF.Min (a NaN on either side gives NaN; +0 above -0), ReLU is
// MathF.Max(x, 0) (NaN stays NaN), sign is the CPU's (±1, ±0 and NaN kept).
//
// For G2b (--profile labels and --bench-hip rows): each kernel, its floating-point operations an element (a fused
// multiply-add counts 2, a hardware function or a division step 1 or 2 as the instruction does; compares, selects,
// conversions and integer work not counted; an operation made of selects alone counts 1) and its bytes an element
// (each buffer read and written once; the row vector's read counted with each element).
//
//   kernel            flops  bytes   arguments
//   affine_f32            2      8   x, y, alpha, beta, n
//   axpy_f32              2     12   x, y, alpha, n
//   muladd_f32            2     16   a, b, c, n
//   add_f32               1     12   a, b, c, n
//   sub_f32               1     12   a, b, c, n
//   mul_f32               1     12   a, b, c, n
//   max_f32               2     12   a, b, c, n
//   min_f32               2     12   a, b, c, n
//   add_rowvec_f32        1     12   a, v, c, cols, n
//   relu_f32              1      8   x, y, n
//   sigmoid_f32          28      8   x, y, n
//   tanh_f32             38      8   x, y, n
//   gelu_f32             46      8   x, y, n
//   gelu_erf_f32         45      8   x, y, n
//   silu_f32             29      8   x, y, n
//   square_f32            1      8   x, y, n
//   abs_f32               1      8   x, y, n
//   exp_f32              10      8   x, y, n
//   log_f32               9      8   x, y, n
//   sqrt_f32              7      8   x, y, n
//   sin_f32              32      8   x, y, n
//   cos_f32              32      8   x, y, n
//   sign_f32              1      8   x, y, n
//   pow_f32              12      8   x, y, p, flags, n (flags: PtxKernels.PowFlags(p))
//   clamp_f32             4      8   x, y, lo, hi, n
//   leaky_relu_f32        1      8   x, y, slope, n
//   where_f32             1     16   condition, a, b, y, n
//   inv_sqrt_f32         25      8   x, y, eps, n
//   bias_gelu_f32        47     12   x, bias, y, cols, n (the row vector's add_rowvec_f32 shape)
//   add_scalar_f32        1      8   s, y, scale, n (y += s[0] · scale; s[0] read once)
//   gated_silu_f32       30     12   gate, up, y, n (y = act(gate) · up; CUDA's gated_act_f32 takes the kind,
//   gated_gelu_f32       47     12   gate, up, y, n  here one kernel a kind, so no element branches on it)
//   gated_relu_f32        2     12   gate, up, y, n
//   gated_tanh_f32       67     12   gate, up, y, n
//   (fill_f32, the gate kernel in GcnKernels.cs, one element a work-item: 0 flops, 4 bytes; out, value, count)
internal static partial class GcnKernels
{
    /// <summary>The element-wise kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Pointwise => _pointwise ??= BuildPointwise();

    // Built on first use: the text reads this file's lists (GatedKinds), whose static initializers come later.
    private static GcnKernelSource[]? _pointwise;

    /// <summary>The block every element-wise kernel is written for (its required workgroup size): four waves of 64.</summary>
    internal const int PointwiseBlock = 256;

    /// <summary>The most elements an element-wise kernel takes: its byte offsets are 32-bit unsigned values (element × 4), so 2^30.</summary>
    internal const int PointwiseMostElements = 1 << 30;

    /// <summary>The most work-items an element-wise kernel's grid may have: an index plus four strides stays below 2^32.</summary>
    internal const int PointwiseMostWorkItems = 1 << 29;

    /// <summary>
    /// The unary kernels' arguments (relu_f32, sigmoid_f32, tanh_f32, gelu_f32, gelu_erf_f32, silu_f32, square_f32,
    /// abs_f32, exp_f32, log_f32, sqrt_f32, sin_f32, cos_f32, sign_f32): x at 0, y at 8, n at 16; y[i] = f(x[i]).
    /// </summary>
    internal static IReadOnlyList<GcnArgument> UnaryArguments { get; } = [GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "y"), GcnArgument.Value(16, 4, "n")];

    /// <summary>Bytes of a unary kernel's arguments.</summary>
    internal const int UnaryArgumentBytes = 20;

    /// <summary>
    /// The binary kernels' arguments (add_f32, sub_f32, mul_f32, max_f32, min_f32: c[i] = a[i] op b[i]; muladd_f32:
    /// c[i] = fma(a[i], b[i], c[i])): a at 0, b at 8, c at 16, n at 24.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> BinaryArguments { get; } =
        [GcnArgument.Buffer(0, "a"), GcnArgument.Buffer(8, "b"), GcnArgument.Buffer(16, "c"), GcnArgument.Value(24, 4, "n")];

    /// <summary>Bytes of a binary kernel's arguments.</summary>
    internal const int BinaryArgumentBytes = 28;

    /// <summary>affine_f32's arguments: x at 0, y at 8, alpha at 16, beta at 20, n at 24; y[i] = fma(x[i], alpha, beta).</summary>
    internal static IReadOnlyList<GcnArgument> AffineArguments { get; } =
        [GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "y"), GcnArgument.Value(16, 4, "alpha"), GcnArgument.Value(20, 4, "beta"), GcnArgument.Value(24, 4, "n")];

    /// <summary>Bytes of affine_f32's arguments.</summary>
    internal const int AffineArgumentBytes = 28;

    /// <summary>axpy_f32's arguments: x at 0, y at 8, alpha at 16, n at 20; y[i] = fma(x[i], alpha, y[i]).</summary>
    internal static IReadOnlyList<GcnArgument> AxpyArguments { get; } =
        [GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "y"), GcnArgument.Value(16, 4, "alpha"), GcnArgument.Value(20, 4, "n")];

    /// <summary>Bytes of axpy_f32's arguments.</summary>
    internal const int AxpyArgumentBytes = 24;

    /// <summary>add_rowvec_f32's arguments: a at 0, v at 8, c at 16, cols at 24, n at 28; c[i] = a[i] + v[i mod cols] (n = rows × cols).</summary>
    internal static IReadOnlyList<GcnArgument> RowVectorArguments { get; } =
        [GcnArgument.Buffer(0, "a"), GcnArgument.Buffer(8, "v"), GcnArgument.Buffer(16, "c"), GcnArgument.Value(24, 4, "cols"), GcnArgument.Value(28, 4, "n")];

    /// <summary>Bytes of add_rowvec_f32's arguments.</summary>
    internal const int RowVectorArgumentBytes = 32;

    /// <summary>A unary kernel's arguments (<see cref="UnaryArguments"/>) into <paramref name="arguments"/>, <see cref="UnaryArgumentBytes"/> long.</summary>
    internal static void PackUnary(Span<byte> arguments, ulong x, ulong y, uint n)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, UnaryArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments, x);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[8..], y);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[16..], n);
    }

    /// <summary>A binary kernel's arguments (<see cref="BinaryArguments"/>) into <paramref name="arguments"/>, <see cref="BinaryArgumentBytes"/> long.</summary>
    internal static void PackBinary(Span<byte> arguments, ulong a, ulong b, ulong c, uint n)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, BinaryArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments, a);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[8..], b);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[16..], c);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[24..], n);
    }

    /// <summary>affine_f32's arguments (<see cref="AffineArguments"/>) into <paramref name="arguments"/>, <see cref="AffineArgumentBytes"/> long.</summary>
    internal static void PackAffine(Span<byte> arguments, ulong x, ulong y, float alpha, float beta, uint n)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, AffineArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments, x);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[8..], y);
        BinaryPrimitives.WriteSingleLittleEndian(arguments[16..], alpha);
        BinaryPrimitives.WriteSingleLittleEndian(arguments[20..], beta);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[24..], n);
    }

    /// <summary>axpy_f32's arguments (<see cref="AxpyArguments"/>) into <paramref name="arguments"/>, <see cref="AxpyArgumentBytes"/> long.</summary>
    internal static void PackAxpy(Span<byte> arguments, ulong x, ulong y, float alpha, uint n)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, AxpyArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments, x);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[8..], y);
        BinaryPrimitives.WriteSingleLittleEndian(arguments[16..], alpha);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[20..], n);
    }

    /// <summary>add_rowvec_f32's arguments (<see cref="RowVectorArguments"/>) into <paramref name="arguments"/>, <see cref="RowVectorArgumentBytes"/> long.</summary>
    internal static void PackAddRowVector(Span<byte> arguments, ulong a, ulong v, ulong c, uint cols, uint n)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, RowVectorArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments, a);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[8..], v);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[16..], c);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[24..], cols);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[28..], n);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The kernels. A shape names the buffers and values (in argument order, n after them), the buffers read (in the
    // order the body takes them), the buffer written, and whether the body's second input is the row vector's element.
    // A body writes one element's result into `output` from its inputs' registers and the values' scalar registers.
    // ---------------------------------------------------------------------------------------------------------------

    private delegate void PointwiseBody(PointwiseWriter w, string[] inputs, string[] values, string output);

    private sealed record PointwiseShape(string[] Buffers, string[] Values, int[] Read, int Written, bool RowVector = false);

    private static GcnKernelSource[] BuildPointwise()
    {
        PointwiseShape unary = new(["x", "y"], [], [0], 1);
        PointwiseShape binary = new(["a", "b", "c"], [], [0, 1], 2);
        return
        [
            Kernel("affine_f32", new(["x", "y"], ["alpha", "beta"], [0], 1), (w, i, v, o) => w.I($"v_fma_f32 {o}, {i[0]}, {v[0]}, {w.VFrom(v[1])}")),
            Kernel("axpy_f32", new(["x", "y"], ["alpha"], [0, 1], 1), (w, i, v, o) => w.I($"v_fma_f32 {o}, {i[0]}, {v[0]}, {i[1]}")),
            Kernel("muladd_f32", new(["a", "b", "c"], [], [0, 1, 2], 2), (w, i, _, o) => w.I($"v_fma_f32 {o}, {i[0]}, {i[1]}, {i[2]}")),
            Kernel("add_f32", binary, (w, i, _, o) => w.I($"v_add_f32_e32 {o}, {i[0]}, {i[1]}")),
            Kernel("sub_f32", binary, (w, i, _, o) => w.I($"v_sub_f32_e32 {o}, {i[0]}, {i[1]}")),
            Kernel("mul_f32", binary, (w, i, _, o) => w.I($"v_mul_f32_e32 {o}, {i[0]}, {i[1]}")),
            Kernel("max_f32", binary, (w, i, _, o) => w.Extreme(o, i[0], i[1], minimum: false)),
            Kernel("min_f32", binary, (w, i, _, o) => w.Extreme(o, i[0], i[1], minimum: true)),
            Kernel("add_rowvec_f32", new(["a", "v", "c"], ["cols"], [0], 2, RowVector: true), (w, i, _, o) => w.I($"v_add_f32_e32 {o}, {i[0]}, {i[1]}")),
            Kernel("relu_f32", unary, (w, i, _, o) =>
            {
                w.I($"v_cmp_ge_f32_e32 vcc, 0, {i[0]}               ; x <= 0 (not NaN): +0");
                w.I($"v_cndmask_b32_e64 {o}, {i[0]}, 0, vcc");
            }),
            Kernel("sigmoid_f32", unary, (w, i, _, o) => w.Sigmoid(o, i[0])),
            Kernel("tanh_f32", unary, (w, i, _, o) => w.Tanh(o, i[0])),
            Kernel("gelu_f32", unary, (w, i, _, o) => w.GeluTanh(o, i[0])),
            Kernel("gelu_erf_f32", unary, (w, i, _, o) => w.GeluErf(o, i[0])),
            Kernel("silu_f32", unary, (w, i, _, o) =>
            {
                string s = w.V();
                w.Sigmoid(s, i[0]);
                w.I($"v_mul_f32_e32 {o}, {i[0]}, {s}");
            }),
            Kernel("square_f32", unary, (w, i, _, o) => w.I($"v_mul_f32_e32 {o}, {i[0]}, {i[0]}")),
            Kernel("abs_f32", unary, (w, i, _, o) => w.I($"v_and_b32_e32 {o}, 0x7fffffff, {i[0]}")),
            Kernel("exp_f32", unary, (w, i, _, o) => w.Exp(o, i[0])),
            Kernel("log_f32", unary, (w, i, _, o) => w.Log(o, i[0])),
            Kernel("sqrt_f32", unary, (w, i, _, o) => w.Sqrt(o, i[0])),
            Kernel("sin_f32", unary, (w, i, _, o) => w.SinCos(o, i[0], cosine: false)),
            Kernel("cos_f32", unary, (w, i, _, o) => w.SinCos(o, i[0], cosine: true)),
            Kernel("sign_f32", unary, (w, i, _, o) =>
            {
                string t = w.V();
                w.I($"v_cmp_lt_f32_e32 vcc, 0, {i[0]}               ; x > 0: 1");
                w.I($"v_cndmask_b32_e64 {t}, {i[0]}, 1.0, vcc");
                w.I($"v_cmp_gt_f32_e32 vcc, 0, {i[0]}               ; x < 0: -1; else x (±0, NaN)");
                w.I($"v_cndmask_b32_e64 {o}, {t}, -1.0, vcc");
            }),

            // G5a: the element-wise operations with parameters (PtxKernels.Pointwise.cs, PtxKernels.Advanced.cs).
            Kernel("pow_f32", new(["x", "y"], ["p", "flags"], [0], 1), (w, i, v, o) => w.Pow(o, i[0], v[0], v[1])),
            Kernel("clamp_f32", new(["x", "y"], ["lo", "hi"], [0], 1), (w, i, v, o) =>
            {
                string t = w.V();
                w.Extreme(t, i[0], w.VFrom(v[0]), minimum: false);
                w.Extreme(o, t, w.VFrom(v[1]), minimum: true);
            }),
            Kernel("leaky_relu_f32", new(["x", "y"], ["slope"], [0], 1), (w, i, v, o) =>
            {
                string t = w.V();
                w.I($"v_mul_f32_e32 {t}, {v[0]}, {i[0]}");
                w.I($"v_cmp_lt_f32_e32 vcc, 0, {i[0]}               ; x > 0: x (NaN: slope · NaN)");
                w.I($"v_cndmask_b32_e32 {o}, {t}, {i[0]}, vcc");
            }),
            Kernel("where_f32", new(["condition", "a", "b", "y"], [], [0, 1, 2], 3), (w, i, _, o) =>
            {
                w.I($"v_cmp_neq_f32_e32 vcc, 0, {i[0]}              ; condition ≠ 0 (NaN too): a");
                w.I($"v_cndmask_b32_e32 {o}, {i[2]}, {i[1]}, vcc");
            }),
            Kernel("inv_sqrt_f32", new(["x", "y"], ["eps"], [0], 1), (w, i, v, o) =>
            {
                string sum = w.V(), root = w.V();
                w.I($"v_add_f32_e32 {sum}, {v[0]}, {i[0]}");
                w.Sqrt(root, sum);
                w.Div(o, "1.0", root);
            }),
            Kernel("bias_gelu_f32", new(["x", "bias", "y"], ["cols"], [0], 2, RowVector: true), (w, i, _, o) =>
            {
                string sum = w.V();
                w.I($"v_add_f32_e32 {sum}, {i[0]}, {i[1]}");
                w.GeluTanh(o, sum);
            }),
            Kernel("add_scalar_f32", new(["s", "y"], ["scale"], [1], 1), (w, i, v, o) => w.I($"v_add_f32_e32 {o}, {w.HeldProduct("%s[0:1]", v[0])}, {i[0]}")),
            .. GatedKinds.Select((kind, k) => Kernel(kind, new(["gate", "up", "y"], [], [0, 1], 2), (w, i, _, o) => w.Gated(o, i[0], i[1], k))),
        ];
    }

    /// <summary>The gated activations' kernels by kind (0 SiLU, 1 GELU, 2 ReLU, 3 tanh · sigmoid): one a kind, no branch per element.</summary>
    internal static readonly string[] GatedKinds = ["gated_silu_f32", "gated_gelu_f32", "gated_relu_f32", "gated_tanh_f32"];

    // Scalar registers of the frame (the arguments take %s0 up, a dword each, in argument order; the writer's constants
    // and masks %s64 up).
    private const string GridItems = "%s40", WaveChunk = "%s41", Waves = "%s42", Chunks = "%s43", ChunkBytes = "%s44", PassBytes = "%s45",
        Misaligned = "%s46", TailFirst = "%s47", NegCols = "%s48", Scratch = "%s49", SavedExec = "%s[50:51]";

    // Vector registers of the frame: the work-item's index in the grid, its lane's bytes in a chunk, the chunk's byte
    // offset, the tail's element and byte offset; the row vector's reciprocal of cols, column and steps (the writer's
    // registers %v16 up).
    private const string Item = "%v0", LaneBytes = "%v1", Offset = "%v2", TailElement = "%v3", TailOffset = "%v4", Reciprocal = "%v5",
        Column = "%v6", ColumnStep = "%v7", TailColumn = "%v8", TailStep = "%v9";

    // One element-wise kernel's text (see the file's remarks).
    private static GcnKernelSource Kernel(string name, PointwiseShape shape, PointwiseBody body)
    {
        var w = new PointwiseWriter();
        int dwords = 2 * shape.Buffers.Length + shape.Values.Length + 1;
        string Pointer(int buffer) => $"%s[{2 * buffer}:{2 * buffer + 1}]";
        string Low(int buffer) => $"%s{2 * buffer}";
        string[] values = [.. Enumerable.Range(0, shape.Values.Length).Select(k => $"%s{2 * shape.Buffers.Length + k}")];
        string count = $"%s{dwords - 1}";
        string cols = shape.RowVector ? values[0] : "";

        // The main loop: a chunk of 256 elements a wave a pass, 4 a lane.
        var main = new StringBuilder();
        w.Into(main);
        w.I($"v_add_u32_e32 {Offset}, {ChunkBytes}, {LaneBytes}       ; this lane's 16 bytes of the wave's chunk");
        var quads = new int[shape.Read.Length];
        for (int r = 0; r < shape.Read.Length; r++)
        {
            quads[r] = w.Range(4);
            w.I($"global_load_dwordx4 %v[{quads[r]}:{quads[r] + 3}], {Offset}, {Pointer(shape.Read[r])}");
        }

        var rowValues = new string[4];
        if (shape.RowVector)
        {
            string column = Column;
            for (int e = 0; e < 4; e++)
            {
                column = e == 0 ? column : w.NextColumn(column, cols);
                string bytes = w.V();
                rowValues[e] = w.V();
                w.I($"v_lshlrev_b32_e32 {bytes}, 2, {column}");
                w.I($"global_load_dword {rowValues[e]}, {bytes}, {Pointer(1)}");
            }
        }

        int results = w.Range(4);
        for (int e = 0; e < 4; e++)
        {
            string[] inputs = shape.RowVector ? [$"%v{quads[0] + e}", rowValues[e]] : [.. quads.Select(q => $"%v{q + e}")];
            body(w, inputs, values, $"%v{results + e}");
        }

        w.I($"global_store_dwordx4 {Offset}, %v[{results}:{results + 3}], {Pointer(shape.Written)}");
        if (shape.RowVector)
        {
            w.AddColumns(Column, ColumnStep, cols);
        }

        // The tail: one element a work-item, grid-stride, with the lanes past n off.
        var tail = new StringBuilder();
        w.Into(tail);
        w.I($"v_lshlrev_b32_e32 {TailOffset}, 2, {TailElement}");
        var singles = new string[shape.Read.Length];
        for (int r = 0; r < shape.Read.Length; r++)
        {
            singles[r] = w.V();
            w.I($"global_load_dword {singles[r]}, {TailOffset}, {Pointer(shape.Read[r])}");
        }

        if (shape.RowVector)
        {
            string bytes = w.V();
            string value = w.V();
            w.I($"v_lshlrev_b32_e32 {bytes}, 2, {TailColumn}");
            w.I($"global_load_dword {value}, {bytes}, {Pointer(1)}");
            singles = [singles[0], value];
        }

        string result = w.V();
        body(w, singles, values, result);
        w.I($"global_store_dword {TailOffset}, {result}, {Pointer(shape.Written)}");
        if (shape.RowVector)
        {
            w.AddColumns(TailColumn, TailStep, cols);
        }

        // The head: directives, the arguments, the frame; then the constants the bodies hold, set once.
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $".kernel {name}\n");
        foreach (string buffer in shape.Buffers)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {buffer}, 8, global\n");
        }

        foreach (string value in shape.Values)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {value}, 4, value\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"""
            .arg n, 4, value
            .reqd_workgroup_size {PointwiseBlock}, 1, 1
            .user_sgpr dispatch_ptr
            .user_sgpr kernarg_segment_ptr
            .system_sgpr workgroup_id_x

            """);
        var head = new StringBuilder();
        w.Into(head);
        w.I($"s_load_dword {GridItems}, s[0:1], 0xc              ; the grid's work-items (the dispatch packet's grid_size_x)");
        for (int at = 0; at < dwords;)
        {
            int size = dwords - at >= 8 ? 8 : dwords - at >= 4 ? 4 : dwords - at >= 2 ? 2 : 1;
            w.I(size == 1
                ? string.Create(CultureInfo.InvariantCulture, $"s_load_dword %s{at}, s[2:3], 0x{4 * at:x}")
                : string.Create(CultureInfo.InvariantCulture, $"s_load_dwordx{size} %s[{at}:{at + size - 1}], s[2:3], 0x{4 * at:x}"));
            at += size;
        }

        w.I($"v_readfirstlane_b32 {WaveChunk}, v0                ; the wave's first work-item in its block");
        w.I($"s_lshl_b32 {Scratch}, s4, 8                        ; the block's first work-item");
        w.I($"v_add_u32_e32 {Item}, {Scratch}, v0");
        w.I($"v_and_b32_e32 {LaneBytes}, 63, v0");
        w.I($"v_lshlrev_b32_e32 {LaneBytes}, 4, {LaneBytes}         ; the lane's bytes in a chunk");
        w.I($"s_lshr_b32 {WaveChunk}, {WaveChunk}, 6");
        w.I($"s_lshl_b32 {Scratch}, s4, 2");
        w.I($"s_add_u32 {WaveChunk}, {Scratch}, {WaveChunk}       ; the wave's first chunk: its wave in the grid");
        w.I($"s_lshr_b32 {Waves}, {GridItems}, 6                  ; the grid's waves: chunks a pass");
        w.I($"s_and_b32 {Chunks}, {count}, -4                     ; n rounded down to 4");
        int[] moved = [.. shape.Read.Where(b => !shape.RowVector || b != 1).Append(shape.Written).Distinct()];
        w.I($"s_mov_b32 {Misaligned}, {Low(moved[0])}");
        foreach (int buffer in moved.Skip(1))
        {
            w.I($"s_or_b32 {Misaligned}, {Misaligned}, {Low(buffer)}");
        }

        w.I($"s_and_b32 {Misaligned}, {Misaligned}, 15            ; SCC: a buffer moved 4 at a time is not 16-byte aligned");
        w.I($"s_cselect_b32 {Chunks}, 0, {Chunks}                 ; then no chunk is whole: the tail takes every element");
        w.I($"s_lshr_b32 {Chunks}, {Chunks}, 8                    ; whole chunks of 256");
        w.I($"s_lshl_b32 {ChunkBytes}, {WaveChunk}, 10            ; the wave's chunk's byte offset");
        w.I($"s_lshl_b32 {PassBytes}, {Waves}, 10                 ; bytes a pass");
        w.I($"s_lshl_b32 {TailFirst}, {Chunks}, 8                 ; the tail's first element");
        if (shape.RowVector)
        {
            // The remainder by cols as LLVM computes it: a reciprocal once, then each column.
            w.I($"v_cvt_f32_u32_e32 {Reciprocal}, {cols}");
            w.I($"s_sub_i32 {NegCols}, 0, {cols}");
            w.I($"v_rcp_iflag_f32_e32 {Reciprocal}, {Reciprocal}");
            w.I($"v_mul_f32_e32 {Reciprocal}, 0x4f7ffffe, {Reciprocal}");
            w.I($"v_cvt_u32_f32_e32 {Reciprocal}, {Reciprocal}");
            string t = w.V();
            w.I($"v_mul_lo_u32 {t}, {NegCols}, {Reciprocal}");
            w.I($"v_mul_hi_u32 {t}, {Reciprocal}, {t}");
            w.I($"v_add_u32_e32 {Reciprocal}, {Reciprocal}, {t}");
            string first = w.V();
            w.I($"v_lshrrev_b32_e32 {first}, 2, {LaneBytes}");
            w.I($"s_lshl_b32 {Scratch}, {WaveChunk}, 8");
            w.I($"v_add_u32_e32 {first}, {Scratch}, {first}          ; the lane's first element");
            w.Remainder(Column, first, cols);
            string step = w.V();
            w.I($"s_lshl_b32 {Scratch}, {Waves}, 8");
            w.I($"v_mov_b32_e32 {step}, {Scratch}                    ; elements a pass");
            w.Remainder(ColumnStep, step, cols);
            string tailItem = w.V();
            w.I($"v_add_u32_e32 {tailItem}, {TailFirst}, {Item}");
            w.Remainder(TailColumn, tailItem, cols);
            string grid = w.V();
            w.I($"v_mov_b32_e32 {grid}, {GridItems}");
            w.Remainder(TailStep, grid, cols);
        }

        string label = $".L{name}";
        text.Append(head).Append(w.Held);
        text.Append(CultureInfo.InvariantCulture, $"""
              s_cmp_lt_u32 {WaveChunk}, {Chunks}
              s_cbranch_scc0 {label}_tail
            {label}_main:

            """);
        text.Append(main);
        text.Append(CultureInfo.InvariantCulture, $"""
              s_add_u32 {WaveChunk}, {WaveChunk}, {Waves}
              s_add_u32 {ChunkBytes}, {ChunkBytes}, {PassBytes}
              s_cmp_lt_u32 {WaveChunk}, {Chunks}
              s_cbranch_scc1 {label}_main
            {label}_tail:
              v_add_u32_e32 {TailElement}, {TailFirst}, {Item}
              v_cmp_gt_u32_e32 vcc, {count}, {TailElement}
              s_and_saveexec_b64 {SavedExec}, vcc
              s_cbranch_execz {label}_done
            {label}_tail_loop:

            """);
        text.Append(tail);
        text.Append(CultureInfo.InvariantCulture, $"""
              v_add_u32_e32 {TailElement}, {GridItems}, {TailElement}
              v_cmp_gt_u32_e32 vcc, {count}, {TailElement}
              s_and_b64 exec, exec, vcc
              s_cbranch_execnz {label}_tail_loop
            {label}_done:
              s_endpgm

            """);
        return new GcnKernelSource(name, Encoding.UTF8.GetBytes(text.ToString()));
    }

    /// <summary>
    /// Writes element-wise bodies with virtual registers: fresh registers, the constants they hold (each set once at
    /// the kernel's start, in a scalar register, or in a vector one where the instruction already reads a scalar or
    /// VCC), and the math sequences (see the file's remarks).
    /// </summary>
    private sealed class PointwiseWriter(int firstVector = 16, int firstScalar = 64)
    {
        private readonly Dictionary<(bool Vector, uint Bits), string> _held = [];
        private readonly Dictionary<string, string> _copies = [];
        private StringBuilder _text = new();
        private int _v = firstVector, _s = firstScalar;

        /// <summary>The lines that set the held constants and copies.</summary>
        public StringBuilder Held { get; } = new();

        public void Into(StringBuilder text) => _text = text;

        public void I(string line) => _text.Append("  ").Append(line).Append('\n');

        /// <summary>A fresh vector register.</summary>
        public string V() => string.Create(CultureInfo.InvariantCulture, $"%v{_v++}");

        /// <summary>The first of <paramref name="count"/> fresh adjacent vector registers.</summary>
        public int Range(int count)
        {
            int first = _v;
            _v += count;
            return first;
        }

        /// <summary>A fresh scalar pair (a lane mask).</summary>
        public string Mask()
        {
            int first = _s;
            _s += 2;
            return string.Create(CultureInfo.InvariantCulture, $"%s[{first}:{first + 1}]");
        }

        /// <summary>A scalar register holding <paramref name="value"/>.</summary>
        public string S(float value) => Hold(false, BitConverter.SingleToUInt32Bits(value));

        /// <summary>A scalar register holding the bits <paramref name="bits"/>.</summary>
        public string S(uint bits) => Hold(false, bits);

        /// <summary>A vector register holding <paramref name="value"/> in every lane.</summary>
        public string VC(float value) => Hold(true, BitConverter.SingleToUInt32Bits(value));

        /// <summary>A vector register holding the bits <paramref name="bits"/> in every lane.</summary>
        public string VC(uint bits) => Hold(true, bits);

        /// <summary>A vector register holding scalar register <paramref name="scalar"/>'s value (an argument), copied once.</summary>
        public string VFrom(string scalar)
        {
            if (!_copies.TryGetValue(scalar, out string? register))
            {
                register = V();
                Held.Append(CultureInfo.InvariantCulture, $"  v_mov_b32_e32 {register}, {scalar}\n");
                _copies[scalar] = register;
            }

            return register;
        }

        /// <summary>
        /// A lane mask (all lanes or none) that is set when scalar register <paramref name="flags"/> (an argument) has any
        /// of <paramref name="bits"/>, made once at the kernel's start.
        /// </summary>
        public string FlagMask(string flags, uint bits)
        {
            string key = $"flag {flags} {bits}";
            if (!_copies.TryGetValue(key, out string? mask))
            {
                mask = Mask();
                string t = string.Create(CultureInfo.InvariantCulture, $"%s{_s++}");
                Held.Append(CultureInfo.InvariantCulture, $"  s_and_b32 {t}, {flags}, {Literal(bits)}\n");
                Held.Append(CultureInfo.InvariantCulture, $"  s_cmp_lg_u32 {t}, 0\n");
                Held.Append(CultureInfo.InvariantCulture, $"  s_cselect_b64 {mask}, -1, 0\n");
                _copies[key] = mask;
            }

            return mask;
        }

        /// <summary>
        /// A vector register holding <paramref name="pointer"/>[0] · <paramref name="scale"/> (a buffer's first float and
        /// a scalar argument), read and multiplied once at the kernel's start.
        /// </summary>
        public string HeldProduct(string pointer, string scale)
        {
            string key = $"product {pointer} {scale}";
            if (!_copies.TryGetValue(key, out string? register))
            {
                register = V();
                string t = string.Create(CultureInfo.InvariantCulture, $"%s{_s++}");
                Held.Append(CultureInfo.InvariantCulture, $"  s_load_dword {t}, {pointer}, 0x0\n");
                Held.Append(CultureInfo.InvariantCulture, $"  v_mov_b32_e32 {register}, {scale}\n");
                Held.Append(CultureInfo.InvariantCulture, $"  v_mul_f32_e32 {register}, {t}, {register}\n");
                _copies[key] = register;
            }

            return register;
        }

        /// <summary>A literal of a float's bits.</summary>
        public static string Literal(float value) => Literal(BitConverter.SingleToUInt32Bits(value));

        /// <summary>A literal of 32 bits.</summary>
        public static string Literal(uint bits) => "0x" + bits.ToString("x8", CultureInfo.InvariantCulture);

        private string Hold(bool vector, uint bits)
        {
            if (!_held.TryGetValue((vector, bits), out string? register))
            {
                register = vector ? V() : string.Create(CultureInfo.InvariantCulture, $"%s{_s++}");
                Held.Append(CultureInfo.InvariantCulture, $"  {(vector ? "v_mov_b32_e32" : "s_mov_b32")} {register}, {Literal(bits)}\n");
                _held[(vector, bits)] = register;
            }

            return register;
        }

        // ---- the row vector's columns -----------------------------------------------------------------------------

        /// <summary>target = x mod cols, through the frame's reciprocal (LLVM's unsigned remainder).</summary>
        public void Remainder(string target, string x, string cols)
        {
            string q = V(), t = V();
            I($"v_mul_hi_u32 {q}, {x}, {Reciprocal}");
            I($"v_mul_lo_u32 {q}, {q}, {cols}");
            I($"v_sub_u32_e32 {target}, {x}, {q}");
            I($"v_subrev_u32_e32 {t}, {cols}, {target}");
            I($"v_cmp_le_u32_e32 vcc, {cols}, {target}");
            I($"v_cndmask_b32_e32 {target}, {target}, {t}, vcc");
            I($"v_subrev_u32_e32 {t}, {cols}, {target}");
            I($"v_cmp_le_u32_e32 vcc, {cols}, {target}");
            I($"v_cndmask_b32_e32 {target}, {target}, {t}, vcc");
        }

        /// <summary>The column after <paramref name="column"/> (wrapping at cols).</summary>
        public string NextColumn(string column, string cols)
        {
            string next = V(), t = V();
            I($"v_add_u32_e32 {next}, 1, {column}");
            I($"v_subrev_u32_e32 {t}, {cols}, {next}");
            I($"v_cmp_le_u32_e32 vcc, {cols}, {next}");
            I($"v_cndmask_b32_e32 {next}, {next}, {t}, vcc");
            return next;
        }

        /// <summary>column = (column + step) mod cols, both below cols.</summary>
        public void AddColumns(string column, string step, string cols)
        {
            string t = V();
            I($"v_add_u32_e32 {column}, {column}, {step}");
            I($"v_subrev_u32_e32 {t}, {cols}, {column}");
            I($"v_cmp_le_u32_e32 vcc, {cols}, {column}");
            I($"v_cndmask_b32_e32 {column}, {column}, {t}, vcc");
        }

        // ---- the math -------------------------------------------------------------------------------------------

        /// <summary>target = MathF.Max(a, b) or MathF.Min(a, b): NaN if either is; equal values by their bits (+0 above -0).</summary>
        public void Extreme(string target, string a, string b, bool minimum)
        {
            string t = V(), z = V(), nan = V();
            I($"{(minimum ? "v_min_f32_e32" : "v_max_f32_e32")} {t}, {a}, {b}");
            I($"{(minimum ? "v_or_b32_e32" : "v_and_b32_e32")} {z}, {a}, {b}            ; equal: -0 for min, +0 for max");
            I($"v_cmp_eq_f32_e32 vcc, {a}, {b}");
            I($"v_cndmask_b32_e32 {t}, {t}, {z}, vcc");
            I($"v_add_f32_e32 {nan}, {a}, {b}");
            I($"v_cmp_u_f32_e32 vcc, {a}, {b}                     ; a NaN on either side: NaN");
            I($"v_cndmask_b32_e32 {target}, {t}, {nan}, vcc");
        }

        /// <summary>target = e^x (LLVM's expansion of llvm.exp.f32).</summary>
        public void Exp(string target, string x)
        {
            string high = V(), low = V(), whole = V(), k = V();
            I($"v_mul_f32_e32 {high}, {Literal(0x3fb8aa3bu)}, {x}         ; x·log2 e, high");
            I($"v_fma_f32 {low}, {x}, {S(0x3fb8aa3bu)}, -{high}           ; its rounding error");
            I($"v_rndne_f32_e32 {whole}, {high}");
            I($"v_fma_f32 {low}, {x}, {S(0x32a5705fu)}, {low}             ; plus x·(log2 e's second float)");
            I($"v_sub_f32_e32 {high}, {high}, {whole}");
            I($"v_add_f32_e32 {high}, {high}, {low}");
            I($"v_cvt_i32_f32_e32 {k}, {whole}");
            I($"v_exp_f32_e32 {high}, {high}");
            I($"v_ldexp_f32 {high}, {high}, {k}");
            I($"v_cmp_gt_f32_e32 vcc, {S(0xc2ce8ed0u)}, {x}               ; below -103.28: 0");
            I($"v_cndmask_b32_e64 {high}, {high}, 0, vcc");
            I($"v_cmp_lt_f32_e32 vcc, {S(0x42b17218u)}, {x}               ; above 88.72: infinity");
            I($"v_cndmask_b32_e32 {target}, {high}, {VC(0x7f800000u)}, vcc");
        }

        /// <summary>target = ln x (LLVM's expansion of llvm.log.f32).</summary>
        public void Log(string target, string x)
        {
            string scaled = V(), log = V(), shift = V(), high = V(), low = V(), finite = Mask();
            I($"v_cmp_gt_f32_e32 vcc, {S(0x00800000u)}, {x}               ; below the smallest normal: scaled by 2^32");
            I($"v_cndmask_b32_e32 {scaled}, 1.0, {VC(0x4f800000u)}, vcc");
            I($"v_mul_f32_e32 {scaled}, {x}, {scaled}");
            I($"v_log_f32_e32 {log}, {scaled}");
            I($"v_cndmask_b32_e32 {shift}, 0, {VC(0x41b17218u)}, vcc      ; 32·ln 2 taken off after");
            I($"v_mul_f32_e32 {high}, {Literal(0x3f317217u)}, {log}       ; log2·ln 2, high");
            I($"v_fma_f32 {low}, {log}, {S(0x3f317217u)}, -{high}");
            I($"v_fma_f32 {low}, {log}, {S(0x3377d1cfu)}, {low}           ; plus log2·(ln 2's second float)");
            I($"v_add_f32_e32 {high}, {high}, {low}");
            I($"v_cmp_lt_f32_e64 {finite}, |{log}|, {S(0x7f800000u)}");
            I($"v_cndmask_b32_e64 {log}, {log}, {high}, {finite}");
            I($"v_sub_f32_e32 {target}, {log}, {shift}");
        }

        /// <summary>target = √x correctly rounded (LLVM's expansion of llvm.sqrt.f32).</summary>
        public void Sqrt(string target, string x)
        {
            string big = V(), y = V(), root = V(), down = V(), up = V(), rd = V(), ru = V(), small = V(), mask = Mask();
            I($"v_mul_f32_e32 {big}, {Literal(0x4f800000u)}, {x}");
            I($"v_cmp_gt_f32_e32 vcc, {S(0x0f800000u)}, {x}               ; below 2^-96: scaled by 2^32");
            I($"v_cndmask_b32_e32 {y}, {x}, {big}, vcc");
            I($"v_sqrt_f32_e32 {root}, {y}");
            I($"v_add_u32_e32 {down}, -1, {root}                          ; the floats either side");
            I($"v_add_u32_e32 {up}, 1, {root}");
            I($"v_fma_f32 {rd}, -{down}, {root}, {y}");
            I($"v_fma_f32 {ru}, -{up}, {root}, {y}");
            I($"v_cmp_ge_f32_e64 {mask}, 0, {rd}");
            I($"v_cndmask_b32_e64 {root}, {root}, {down}, {mask}");
            I($"v_cmp_lt_f32_e64 {mask}, 0, {ru}");
            I($"v_cndmask_b32_e64 {root}, {root}, {up}, {mask}");
            I($"v_mul_f32_e32 {small}, {Literal(0x37800000u)}, {root}     ; 2^-16 undoes the scale");
            I($"v_cndmask_b32_e32 {root}, {root}, {small}, vcc");
            I($"v_cmp_class_f32_e32 vcc, {y}, {VC(0x260u)}                ; ±0 and +infinity: themselves");
            I($"v_cndmask_b32_e32 {target}, {root}, {y}, vcc");
        }

        /// <summary>target = n / d correctly rounded (LLVM's expansion of fdiv, float32 denormals kept).</summary>
        public void Div(string target, string n, string d)
        {
            string ds = V(), ns = V(), r = V(), e = V(), q = V(), e2 = V(), unused = Mask();
            I($"v_div_scale_f32 {ds}, {unused}, {d}, {d}, {n}");
            I($"v_div_scale_f32 {ns}, vcc, {n}, {d}, {n}");
            I($"v_rcp_f32_e32 {r}, {ds}");
            I($"v_fma_f32 {e}, -{ds}, {r}, 1.0");
            I($"v_fma_f32 {r}, {e}, {r}, {r}");
            I($"v_mul_f32_e32 {q}, {ns}, {r}");
            I($"v_fma_f32 {e2}, -{ds}, {q}, {ns}");
            I($"v_fma_f32 {q}, {e2}, {r}, {q}");
            I($"v_fma_f32 {ds}, -{ds}, {q}, {ns}");
            I($"v_div_fmas_f32 {ds}, {ds}, {r}, {q}");
            I($"v_div_fixup_f32 {target}, {ds}, {d}, {n}");
        }

        /// <summary>target = 1 / (1 + e^-x) (the CPU's sigmoid).</summary>
        public void Sigmoid(string target, string x)
        {
            string negated = V(), e = V(), d = V();
            I($"v_xor_b32_e32 {negated}, 0x80000000, {x}");
            Exp(e, negated);
            I($"v_add_f32_e32 {d}, 1.0, {e}");
            Div(target, "1.0", d);
        }

        /// <summary>target = x / (1 + e^-x), SiLU as CpuMath.Gated computes it (one division).</summary>
        public void Silu(string target, string x)
        {
            string negated = V(), e = V(), d = V();
            I($"v_xor_b32_e32 {negated}, 0x80000000, {x}");
            Exp(e, negated);
            I($"v_add_f32_e32 {d}, 1.0, {e}");
            Div(target, x, d);
        }

        /// <summary>
        /// target = act(g) · u as CpuMath.Gated computes it: kind 0 SiLU, 1 GELU (tanh), 2 ReLU, 3 tanh(g) · sigmoid(u).
        /// </summary>
        public void Gated(string target, string g, string u, int kind)
        {
            string a = V();
            switch (kind)
            {
                case 0:
                    Silu(a, g);
                    break;
                case 1:
                    GeluTanh(a, g);
                    break;
                case 3:
                    string s = V();
                    Tanh(a, g);
                    Sigmoid(s, u);
                    I($"v_mul_f32_e32 {target}, {a}, {s}");
                    return;
                default:
                    I($"v_cmp_ge_f32_e32 vcc, 0, {g}                    ; g <= 0: +0");
                    I($"v_cndmask_b32_e64 {a}, {g}, 0, vcc");
                    break;
            }

            I($"v_mul_f32_e32 {target}, {a}, {u}");
        }

        /// <summary>
        /// target = MathF.Pow(x, p) as CUDA's pow_f32 computes it (PowOf, with PtxKernels.PowFlags' bits in the scalar
        /// <paramref name="flags"/>): 2^(p · log2 |x|) with the product's rounding error taken back by a fused step, a
        /// denormal |x| scaled by 2^32 before the logarithm and a result below 2^-126 scaled after the exponential (as
        /// LLVM does), then the sign, NaN and the exponents 0, 1 and 2 as MathF.Pow has them. The one pow sequence:
        /// pow_f32 and pow_bwd_f32 both write it (plan 35, G6b).
        /// </summary>
        public void Pow(string target, string x, string p, string flags)
        {
            string a = V(), scaled = V(), l = V(), shift = V(), hi = V(), lo = V(), small = V(), r = V(), t = V(), negated = V(), xx = V();
            string finite = Mask(), one = Mask(), unit = Mask(), xIsOne = Mask(), chosen = Mask();
            I($"v_and_b32_e32 {a}, 0x7fffffff, {x}");
            I($"v_cmp_gt_f32_e32 vcc, {S(0x00800000u)}, {a}               ; below the smallest normal: scaled by 2^32");
            I($"v_cndmask_b32_e32 {scaled}, 1.0, {VC(0x4f800000u)}, vcc");
            I($"v_mul_f32_e32 {scaled}, {a}, {scaled}");
            I($"v_log_f32_e32 {l}, {scaled}");
            I($"v_cndmask_b32_e32 {shift}, 0, {VC(32f)}, vcc");
            I($"v_sub_f32_e32 {l}, {l}, {shift}                          ; log2 |x|");
            I($"v_mul_f32_e32 {hi}, {p}, {l}");
            I($"v_fma_f32 {lo}, {l}, {p}, -{hi}                          ; the product's rounding error");
            I($"v_mul_f32_e32 {lo}, {Literal(0x3f317218u)}, {lo}");
            I($"v_cmp_gt_f32_e32 vcc, {S(-126f)}, {hi}                     ; below 2^-126: 2^(hi + 64) · 2^-64");
            I($"v_cndmask_b32_e32 {small}, 0, {VC(64f)}, vcc");
            I($"v_add_f32_e32 {small}, {hi}, {small}");
            I($"v_exp_f32_e32 {r}, {small}");
            I($"v_cndmask_b32_e32 {small}, 1.0, {VC(0x1f800000u)}, vcc");
            I($"v_mul_f32_e32 {r}, {r}, {small}");
            I($"v_fma_f32 {t}, {r}, {lo}, {r}");
            I($"v_cmp_class_f32_e64 {finite}, {hi}, {S(0x1f8u)}          ; a finite product: corrected");
            I($"v_cndmask_b32_e64 {r}, {r}, {t}, {finite}");
            I($"v_cmp_eq_f32_e64 {unit}, 1.0, {a}");
            I($"v_cmp_eq_f32_e64 {xIsOne}, 1.0, {x}");
            string nanExponent = FlagMask(flags, 4);
            I($"s_and_b64 {chosen}, {nanExponent}, {xIsOne}              ; |x| = 1 gives 1 (x = 1 alone for a NaN exponent)");
            I($"s_andn2_b64 {one}, {unit}, {nanExponent}");
            I($"s_or_b64 {one}, {one}, {chosen}");
            I($"v_cndmask_b32_e64 {r}, {r}, 1.0, {one}");
            I($"v_xor_b32_e32 {negated}, 0x80000000, {r}");
            I($"v_cmp_gt_i32_e32 vcc, 0, {x}                              ; a negative base (its sign bit), an odd exponent: negated");
            I($"s_and_b64 vcc, vcc, {FlagMask(flags, 1)}");
            I($"v_cndmask_b32_e32 {r}, {r}, {negated}, vcc");
            I($"v_cmp_gt_f32_e32 vcc, 0, {x}                              ; a finite negative base, a fractional exponent: NaN");
            I($"v_cmp_class_f32_e64 {finite}, {x}, {S(0x1f8u)}");
            I($"s_and_b64 vcc, vcc, {finite}");
            I($"s_and_b64 vcc, vcc, {FlagMask(flags, 2)}");
            I($"v_cndmask_b32_e32 {r}, {r}, {VC(0xffc00000u)}, vcc");
            I($"v_cndmask_b32_e64 {r}, {r}, 1.0, {FlagMask(flags, 8)}     ; the exponent 0: 1, NaN too");
            I($"v_cndmask_b32_e64 {r}, {r}, {x}, {FlagMask(flags, 16)}    ; 1: x");
            I($"v_mul_f32_e32 {xx}, {x}, {x}");
            I($"v_cndmask_b32_e64 {target}, {r}, {xx}, {FlagMask(flags, 32)}   ; 2: x · x");
        }

        /// <summary>target = tanh x as the CPU computes it (CpuMath.Tanh, TanhRational's coefficients).</summary>
        public void Tanh(string target, string x)
        {
            var (a, b) = (TanhRational.A, TanhRational.B);
            string c = V(), x2 = V(), p = V(), q = V(), r = V(), s = V();
            I($"v_med3_f32 {c}, {x}, {S(-TanhRational.Clamp)}, {VC(TanhRational.Clamp)}");
            I($"v_mul_f32_e32 {x2}, {c}, {c}");
            I($"v_fma_f32 {p}, {x2}, {S(a[6])}, {VC(a[5])}");
            for (int k = 4; k >= 0; k--)
            {
                I($"v_fma_f32 {p}, {x2}, {p}, {S(a[k])}");
            }

            I($"v_mul_f32_e32 {p}, {p}, {c}");
            I($"v_fma_f32 {q}, {x2}, {S(b[3])}, {VC(b[2])}");
            I($"v_fma_f32 {q}, {x2}, {q}, {S(b[1])}");
            I($"v_fma_f32 {q}, {x2}, {q}, {S(b[0])}");
            Div(r, p, q);
            I($"v_cmp_lt_f32_e32 vcc, {Literal(TanhRational.Saturated)}, {x}   ; past ±9: ±1");
            I($"v_cndmask_b32_e64 {s}, {r}, 1.0, vcc");
            I($"v_cmp_gt_f32_e32 vcc, {Literal(-TanhRational.Saturated)}, {x}");
            I($"v_cndmask_b32_e64 {s}, {s}, -1.0, vcc");
            I($"v_cmp_u_f32_e32 vcc, {x}, {x}                     ; NaN kept");
            I($"v_cndmask_b32_e32 {target}, {s}, {x}, vcc");
        }

        /// <summary>target = 0.5 x (1 + tanh(k (x + 0.044715 x³))) with the CPU's lanes' rounding: k · fma(c·x·x, x, x).</summary>
        public void GeluTanh(string target, string x)
        {
            string u = V(), t = V(), h = V(), s = V();
            I($"v_mul_f32_e32 {u}, {Literal(0.044715f)}, {x}");
            I($"v_mul_f32_e32 {u}, {u}, {x}");
            I($"v_fma_f32 {u}, {u}, {x}, {x}");
            I($"v_mul_f32_e32 {u}, {Literal(0.7978845608f)}, {u}");
            Tanh(t, u);
            I($"v_mul_f32_e32 {h}, 0.5, {x}");
            I($"v_add_f32_e32 {s}, 1.0, {t}");
            I($"v_mul_f32_e32 {target}, {h}, {s}");
        }

        /// <summary>target = 0.5 x (1 + erf(x · 0.70710678)), erf as the CPU computes it (CpuMath.Erf).</summary>
        public void GeluErf(string target, string x)
        {
            string z = V(), a = V(), d = V(), t = V(), poly = V(), m = V(), e = V(), erf = V(), negated = V(), h = V(), s = V();
            I($"v_mul_f32_e32 {z}, {Literal(0.70710678f)}, {x}");
            I($"v_and_b32_e32 {a}, 0x7fffffff, {z}");
            I($"v_fma_f32 {d}, {a}, {S(0.3275911f)}, 1.0");
            Div(t, "1.0", d);
            I($"v_fma_f32 {poly}, {t}, {S(1.061405429f)}, {VC(-1.453152027f)}");
            I($"v_fma_f32 {poly}, {poly}, {t}, {S(1.421413741f)}");
            I($"v_fma_f32 {poly}, {poly}, {t}, {S(-0.284496736f)}");
            I($"v_fma_f32 {poly}, {poly}, {t}, {S(0.254829592f)}");
            I($"v_mul_f32_e32 {poly}, {poly}, {t}");
            I($"v_mul_f32_e64 {m}, -{a}, {a}");
            Exp(e, m);
            I($"v_mul_f32_e32 {poly}, {poly}, {e}");
            I($"v_sub_f32_e32 {erf}, 1.0, {poly}");
            I($"v_xor_b32_e32 {negated}, 0x80000000, {erf}");
            I($"v_cmp_gt_f32_e32 vcc, 0, {z}                      ; z < 0: -erf");
            I($"v_cndmask_b32_e32 {erf}, {erf}, {negated}, vcc");
            I($"v_mul_f32_e32 {h}, 0.5, {x}");
            I($"v_add_f32_e32 {s}, 1.0, {erf}");
            I($"v_mul_f32_e32 {target}, {h}, {s}");
        }

        /// <summary>target = sin x or cos x as CUDA's sin_f32 and cos_f32 compute them (PtxKernels.Pointwise.cs).</summary>
        public void SinCos(string target, string x, bool cosine)
        {
            const uint TwoOverPi = 0x3F22F983, NegHalfPi1 = 0xBFC90FDB, NegHalfPi2 = 0x333BBD2E, NegHalfPi3 = 0x26F72CED;
            string j = V(), quadrant = V(), r = V(), more = V(), r2 = V(), sine = V(), cos = V(), bit = V(), picked = V();
            I($"v_mul_f32_e32 {j}, {Literal(TwoOverPi)}, {x}              ; quarter turns");
            I($"v_rndne_f32_e32 {j}, {j}");
            I($"v_cvt_i32_f32_e32 {quadrant}, {j}");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi1)}, {x}                  ; x - j·π/2 in three floats");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi2)}, {r}");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi3)}, {r}");
            I($"v_mul_f32_e32 {j}, {Literal(TwoOverPi)}, {r}              ; the turn a rounding missed");
            I($"v_rndne_f32_e32 {j}, {j}");
            I($"v_cvt_i32_f32_e32 {more}, {j}");
            I($"v_add_u32_e32 {quadrant}, {quadrant}, {more}");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi1)}, {r}");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi2)}, {r}");
            I($"v_fma_f32 {r}, {j}, {S(NegHalfPi3)}, {r}");
            if (cosine)
            {
                I($"v_add_u32_e32 {quadrant}, 1, {quadrant}               ; cos x = sin(x + π/2)");
            }

            I($"v_mul_f32_e32 {r2}, {r}, {r}");
            I($"v_fma_f32 {sine}, {r2}, {S(-1.9515295891e-4f)}, {VC(8.3321608736e-3f)}");
            I($"v_fma_f32 {sine}, {sine}, {r2}, {S(-1.6666654611e-1f)}");
            I($"v_mul_f32_e32 {sine}, {sine}, {r2}");
            I($"v_fma_f32 {sine}, {sine}, {r}, {r}");
            I($"v_fma_f32 {cos}, {r2}, {S(2.443315711809948e-5f)}, {VC(-1.388731625493765e-3f)}");
            I($"v_fma_f32 {cos}, {cos}, {r2}, {S(4.166664568298827e-2f)}");
            I($"v_fma_f32 {cos}, {cos}, {r2}, -0.5");
            I($"v_fma_f32 {cos}, {cos}, {r2}, 1.0");
            I($"v_and_b32_e32 {bit}, 1, {quadrant}");
            I($"v_cmp_ne_u32_e32 vcc, 0, {bit}                    ; odd quadrants: the cosine");
            I($"v_cndmask_b32_e32 {picked}, {sine}, {cos}, vcc");
            I($"v_and_b32_e32 {bit}, 2, {quadrant}");
            I($"v_lshlrev_b32_e32 {bit}, 30, {bit}                ; quadrants 2 and 3: negated");
            if (cosine)
            {
                I($"v_xor_b32_e32 {target}, {picked}, {bit}");
                return;
            }

            I($"v_xor_b32_e32 {picked}, {picked}, {bit}");
            I($"v_cmp_eq_f32_e32 vcc, 0, {x}                      ; sin ±0 = ±0");
            I($"v_cndmask_b32_e32 {target}, {picked}, {x}, vcc");
        }
    }
}
