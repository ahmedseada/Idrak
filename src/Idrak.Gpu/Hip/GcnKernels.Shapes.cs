// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's shape kernels in GCN (plan 35, G4b): the key/value cache writes (float32, bfloat16, int8), the decoder mask, the
// gathers (float32, bfloat16 rows, bfloat16 columns), the permute and the strided copy (adding or not), with CUDA's names
// and arguments in CUDA's order (PtxKernels.Decoding.cs, .Quantized.cs, .Advanced.cs: the buffers, then the 32-bit
// values, the count n last), each computing what the CPU backend computes (CpuBackend.Decoding.cs, .Quantized.cs,
// .Advanced.cs), the reference their results are proven against.
//
// Every kernel has one shape: one output element (a word for kv_write_bf16, a cached row for kv_write_int8) a work-item,
// in blocks of ShapeBlock (declared with .reqd_workgroup_size), walked grid-stride: the dispatch packet's grid size is
// read once and each work-item steps by it while its element is below n, so the backend may launch any grid up to the
// element-wise kernels' bound (PointwiseMostWorkItems: an index plus the stride never wraps). Divisions by a count (an
// element's row and column) are LLVM's unsigned division by a value the whole grid shares: its reciprocal is made once at
// the kernel's start (rule 85), then each element takes a high product and two corrections. A position read from the
// device (pos, position: a float holding a whole number) is read once. Byte offsets are 32-bit: every buffer a kernel
// indexes stays below 2^30 floats (the backend checks and otherwise takes the host fallback). Out-of-range indices are
// clamped to the table's last row as CUDA's are (the CPU refuses them).
//
// For the profile's labels and --bench-hip's rates (HipBackend.Cost): each kernel, its arguments, its unit of work and
// its floating-point operations and global-memory bytes per unit (an index's read counted with its row's elements, so
// left out):
//
// | Kernel               | Arguments                                                            | Unit                 | Flops | Bytes        |
// |----------------------|----------------------------------------------------------------------|----------------------|-------|--------------|
// | kv_write_f32         | src, cache, pos; stepsdim, capdim, dim, n                            | a source float       | 0     | 8            |
// | kv_write_bf16        | src, cache, pos; steps, cap, dim, words, n                           | a source float       | 0     | 6            |
// | kv_write_int8        | src, cache, scales, position; steps, capacity, dim, words, n         | a source float       | 3     | 9            |
// | decoder_mask_f32     | pos, mask; cap, n                                                    | a mask float         | 0     | 4            |
// | gather_f32           | table, indices, y; dim, maxindex, n                                  | an output float      | 0     | 8            |
// | gather_bf16_f32      | table, indices, y; dim, maxindex, words, n                           | an output float      | 0     | 6            |
// | gather_bf16_cols_f32 | table, indices, y; dim, maxindex, words, n                           | an output float      | 0     | 8            |
// | permute_f32          | x, y; s0..s5, t0..t5, accumulate, n                                  | an output float      | 0 (1) | 8 (12)       |
// | copy2d_f32           | src, dst; srcoff, srcstride, dstoff, dststride, cols, accumulate, n  | a copied float       | 0 (1) | 8 (12)       |
//
// (In brackets: adding into the output.) kv_write_bf16's work-item writes a word of two values (n = heads · steps ·
// words); kv_write_int8's quantizes a whole cached row (n = heads · steps), reading it twice (its largest magnitude, then
// each value). gather_bf16_cols_f32 reads a whole word for each value (the column's half). permute_f32's dimensions are
// right-aligned to six (unused leading ones of size 1 and stride 0), as CUDA's.
internal static partial class GcnKernels
{
    /// <summary>The block every shape kernel is written for (its required workgroup size): four waves of 64.</summary>
    internal const int ShapeBlock = 256;

    /// <summary>The dimensions permute_f32 takes (CUDA's <c>MaxPermuteRank</c>).</summary>
    internal const int ShapePermuteRank = 6;

    /// <summary>The most floats a buffer a shape kernel indexes may hold: its byte offsets are 32-bit (element × 4).</summary>
    internal const long ShapeMostElements = 1L << 30;

    /// <summary>The shape kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Shapes { get; } =
    [
        ShapeKeyValueWrite(),
        ShapeKeyValueWriteBFloat16(),
        ShapeKeyValueWriteInt8(),
        ShapeDecoderMask(),
        ShapeGather(),
        ShapeGatherBFloat16(columns: false),
        ShapeGatherBFloat16(columns: true),
        ShapePermute(),
        ShapeCopy2D(),
    ];

    // cache[h, pos + t, d] = src[h, t, d]: one source float a work-item (h, rest) = i / (steps · dim).
    private static GcnKernelSource ShapeKeyValueWrite()
    {
        var w = new ElementWriter("kv_write_f32", ["src", "cache", "pos"], ["stepsdim", "capdim", "dim", "n"]);
        string stepsDim = w.Scalar("stepsdim"), capDim = w.Scalar("capdim"), dim = w.Scalar("dim");
        string pos = w.Position(w.Pointer("pos"));
        string posDim = w.S(), reciprocal = w.Reciprocal(stepsDim);
        w.L($"s_mul_i32 {posDim}, {pos}, {dim}");
        w.Begin();
        var (head, rest) = w.DivMod(w.Item, stepsDim, reciprocal);
        string at = w.V(), from = w.V(), value = w.V();
        w.L($"""
            v_mul_lo_u32 {at}, {head}, {capDim}
            v_add_u32_e32 {at}, {posDim}, {at}
            v_add_u32_e32 {at}, {at}, {rest}
            v_lshlrev_b32_e32 {at}, 2, {at}
            v_lshlrev_b32_e32 {from}, 2, {w.Item}
            global_load_dword {value}, {from}, {w.Pointer("src")}
            global_store_dword {at}, {value}, {w.Pointer("cache")}
            """);
        return w.End();
    }

    // Word wi of cached row (h, pos + t) = the pair src[h, t, 2wi], src[h, t, 2wi + 1] (0 past dim), each rounded to
    // bfloat16 as BFloat16Bits.Round does (to nearest, ties to even; a NaN kept quiet): one word a work-item.
    private static GcnKernelSource ShapeKeyValueWriteBFloat16()
    {
        var w = new ElementWriter("kv_write_bf16", ["src", "cache", "pos"], ["steps", "cap", "dim", "words", "n"]);
        string steps = w.Scalar("steps"), cap = w.Scalar("cap"), dim = w.Scalar("dim"), words = w.Scalar("words");
        string pos = w.Position(w.Pointer("pos"));
        string byWords = w.Reciprocal(words), bySteps = w.Reciprocal(steps);
        w.Begin();
        var (row, word) = w.DivMod(w.Item, words, byWords);
        var (head, step) = w.DivMod(row, steps, bySteps);
        string at = w.V(), first = w.V(), from = w.V(), low = w.V(), high = w.V(), second = w.V(), saved = w.SPair();
        w.L($"""
            v_mul_lo_u32 {at}, {head}, {cap}
            v_add_u32_e32 {at}, {pos}, {at}
            v_add_u32_e32 {at}, {at}, {step}
            v_mul_lo_u32 {at}, {at}, {words}
            v_add_u32_e32 {at}, {at}, {word}
            v_lshlrev_b32_e32 {at}, 2, {at}
            v_lshlrev_b32_e32 {first}, 1, {word}
            v_mul_lo_u32 {from}, {row}, {dim}
            v_add_u32_e32 {from}, {from}, {first}
            v_lshlrev_b32_e32 {from}, 2, {from}
            global_load_dword {low}, {from}, {w.Pointer("src")}
            v_mov_b32_e32 {high}, 0
            v_add_u32_e32 {second}, 1, {first}
            v_cmp_gt_u32_e32 vcc, {dim}, {second}
            s_and_saveexec_b64 {saved}, vcc
            global_load_dword {high}, {from}, {w.Pointer("src")} offset:4
            s_or_b64 exec, exec, {saved}
            """);
        string a = w.RoundBFloat16(low), b = w.RoundBFloat16(high), packed = w.V();
        w.L($"""
            v_lshl_or_b32 {packed}, {b}, 16, {a}
            global_store_dword {at}, {packed}, {w.Pointer("cache")}
            """);
        return w.End();
    }

    // Cached row (h, position + t) of int8 values from source row i = h · steps + t, as the CPU quantizes it: the largest
    // magnitude m, scale = m / 127 (stored in scales), each value round-to-even(x · (1 / scale)) clamped to ±127 (0 when
    // m is 0), four bytes a word, the padding bytes 0. One cached row a work-item.
    private static GcnKernelSource ShapeKeyValueWriteInt8()
    {
        var w = new ElementWriter("kv_write_int8", ["src", "cache", "scales", "position"], ["steps", "capacity", "dim", "words", "n"]);
        string steps = w.Scalar("steps"), capacity = w.Scalar("capacity"), dim = w.Scalar("dim"), words = w.Scalar("words");
        string pos = w.Position(w.Pointer("position"));
        string bySteps = w.Reciprocal(steps), rowBytes = w.S(), wordBytes = w.S();
        w.L($"s_lshl_b32 {rowBytes}, {dim}, 2");
        w.L($"s_lshl_b32 {wordBytes}, {words}, 2");
        w.Begin();
        var (head, step) = w.DivMod(w.Item, steps, bySteps);
        string slot = w.V(), from = w.V(), at = w.V(), max = w.V(), x = w.V(), d = w.S();
        w.L($"""
            v_mul_lo_u32 {slot}, {head}, {capacity}
            v_add_u32_e32 {slot}, {pos}, {slot}
            v_add_u32_e32 {slot}, {slot}, {step}
            v_mul_lo_u32 {from}, {w.Item}, {rowBytes}
            v_mov_b32_e32 {at}, {from}
            v_mov_b32_e32 {max}, 0
            s_mov_b32 {d}, 0
            .Lmax:
            s_cmp_ge_u32 {d}, {dim}
            s_cbranch_scc1 .Lmax_end
            global_load_dword {x}, {at}, {w.Pointer("src")}
            v_max_f32_e64 {max}, {max}, |{x}|
            v_add_u32_e32 {at}, 4, {at}
            s_add_u32 {d}, {d}, 1
            s_branch .Lmax
            .Lmax_end:
            """);
        string scale = w.V(), inverse = w.V(), scaleAt = w.V(), hundred = w.V();
        w.L($"v_mov_b32_e32 {hundred}, 0x42fe0000                ; 127");
        w.Div(scale, max, hundred);
        w.L($"v_mov_b32_e32 {inverse}, 1.0");
        w.Div(inverse, inverse, scale);
        w.L($"""
            v_cmp_lt_f32_e32 vcc, 0, {max}
            v_cndmask_b32_e32 {inverse}, 0, {inverse}, vcc
            v_lshlrev_b32_e32 {scaleAt}, 2, {slot}
            global_store_dword {scaleAt}, {scale}, {w.Pointer("scales")}
            """);
        string target = w.V(), packed = w.V(), wi = w.S(), lane = w.S();
        w.L($"""
            v_mul_lo_u32 {target}, {slot}, {wordBytes}
            s_mov_b32 {wi}, 0
            s_mov_b32 {d}, 0
            .Lwords:
            s_cmp_ge_u32 {wi}, {words}
            s_cbranch_scc1 .Lwords_end
            v_mov_b32_e32 {packed}, 0
            """);
        for (int e = 0; e < 4; e++)
        {
            string value = w.V(), q = w.V();
            w.L($"""
                s_add_u32 {lane}, {d}, {e}
                v_mov_b32_e32 {value}, 0
                s_cmp_ge_u32 {lane}, {dim}
                s_cbranch_scc1 .Lbyte{e}
                global_load_dword {value}, {from}, {w.Pointer("src")} offset:{4 * e}
                .Lbyte{e}:
                v_mul_f32_e32 {q}, {value}, {inverse}
                v_rndne_f32_e32 {q}, {q}
                v_cvt_i32_f32_e32 {q}, {q}
                v_max_i32_e32 {q}, 0xffffff81, {q}
                v_min_i32_e32 {q}, 0x7f, {q}
                v_and_b32_e32 {q}, 0xff, {q}
                v_lshlrev_b32_e32 {q}, {8 * e}, {q}
                v_or_b32_e32 {packed}, {packed}, {q}
                """);
        }

        w.L($"""
            global_store_dword {target}, {packed}, {w.Pointer("cache")}
            v_add_u32_e32 {target}, 4, {target}
            v_add_u32_e32 {from}, 16, {from}
            s_add_u32 {d}, {d}, 4
            s_add_u32 {wi}, {wi}, 1
            s_branch .Lwords
            .Lwords_end:
            """);
        return w.End();
    }

    // mask[i, j] = j ≤ pos + i ? 0 : -1e9, the position read from the device.
    private static GcnKernelSource ShapeDecoderMask()
    {
        var w = new ElementWriter("decoder_mask_f32", ["pos", "mask"], ["cap", "n"]);
        string cap = w.Scalar("cap");
        string pos = w.Position(w.Pointer("pos"));
        string byCap = w.Reciprocal(cap), masked = w.V();
        w.L($"v_mov_b32_e32 {masked}, 0xce6e6b28                 ; -1e9");
        w.Begin();
        var (row, column) = w.DivMod(w.Item, cap, byCap);
        string limit = w.V(), value = w.V(), at = w.V();
        w.L($"""
            v_add_u32_e32 {limit}, {pos}, {row}
            v_cmp_le_u32_e32 vcc, {column}, {limit}
            v_cndmask_b32_e64 {value}, {masked}, 0, vcc
            v_lshlrev_b32_e32 {at}, 2, {w.Item}
            global_store_dword {at}, {value}, {w.Pointer("mask")}
            """);
        return w.End();
    }

    // The row of a gather's output element i: (row, column) = i / dim, the index indices[row] as an unsigned whole number
    // (saturated: a negative one is 0), clamped to maxindex.
    private static (string Index, string Column) ShapeGatherIndex(ElementWriter w, string dim, string byDim)
    {
        var (row, column) = w.DivMod(w.Item, dim, byDim);
        string at = w.V(), index = w.V();
        w.L($"""
            v_lshlrev_b32_e32 {at}, 2, {row}
            global_load_dword {index}, {at}, {w.Pointer("indices")}
            v_cvt_u32_f32_e32 {index}, {index}
            v_min_u32_e32 {index}, {w.Scalar("maxindex")}, {index}
            """);
        return (index, column);
    }

    // y[i, j] = table[indices[i], j].
    private static GcnKernelSource ShapeGather()
    {
        var w = new ElementWriter("gather_f32", ["table", "indices", "y"], ["dim", "maxindex", "n"]);
        string dim = w.Scalar("dim"), byDim = w.Reciprocal(dim);
        w.Begin();
        var (index, column) = ShapeGatherIndex(w, dim, byDim);
        string from = w.V(), value = w.V(), at = w.V();
        w.L($"""
            v_mul_lo_u32 {from}, {index}, {dim}
            v_add_u32_e32 {from}, {from}, {column}
            v_lshlrev_b32_e32 {from}, 2, {from}
            global_load_dword {value}, {from}, {w.Pointer("table")}
            v_lshlrev_b32_e32 {at}, 2, {w.Item}
            global_store_dword {at}, {value}, {w.Pointer("y")}
            """);
        return w.End();
    }

    // From a bfloat16 table packed two a word: rows [vocabulary, words] (gather_bf16_f32: y[i, j] = half j of row
    // indices[i]), or a [dim, words] table read by columns (gather_bf16_cols_f32: y[i, j] = half indices[i] of row j). The
    // half is the low one for an even position, as the CPU's halves are.
    private static GcnKernelSource ShapeGatherBFloat16(bool columns)
    {
        var w = new ElementWriter(columns ? "gather_bf16_cols_f32" : "gather_bf16_f32", ["table", "indices", "y"], ["dim", "maxindex", "words", "n"]);
        string dim = w.Scalar("dim"), words = w.Scalar("words"), byDim = w.Reciprocal(dim);
        w.Begin();
        var (index, column) = ShapeGatherIndex(w, dim, byDim);
        var (row, half) = columns ? (column, index) : (index, column);
        string word = w.V(), bits = w.V(), odd = w.V(), low = w.V(), at = w.V();
        w.L($"""
            v_lshrrev_b32_e32 {word}, 1, {half}
            v_mul_lo_u32 {at}, {row}, {words}
            v_add_u32_e32 {word}, {at}, {word}
            v_lshlrev_b32_e32 {word}, 2, {word}
            global_load_dword {bits}, {word}, {w.Pointer("table")}
            v_and_b32_e32 {odd}, 1, {half}
            v_cmp_eq_u32_e32 vcc, 0, {odd}
            v_lshlrev_b32_e32 {low}, 16, {bits}
            v_and_b32_e32 {bits}, 0xffff0000, {bits}
            v_cndmask_b32_e32 {bits}, {bits}, {low}, vcc
            v_lshlrev_b32_e32 {at}, 2, {w.Item}
            global_store_dword {at}, {bits}, {w.Pointer("y")}
            """);
        return w.End();
    }

    // y[i] (+)= x[Σ_k c_k · t_k] for output element i at coordinates c of the shape s (six dimensions, right-aligned):
    // the coordinates taken off right to left, each by its own reciprocal made once.
    private static GcnKernelSource ShapePermute()
    {
        string[] scalars = [.. Enumerable.Range(0, ShapePermuteRank).Select(d => $"s{d}"), .. Enumerable.Range(0, ShapePermuteRank).Select(d => $"t{d}"), "accumulate", "n"];
        var w = new ElementWriter("permute_f32", ["x", "y"], scalars);
        var reciprocals = new string[ShapePermuteRank];
        for (int d = 0; d < ShapePermuteRank; d++)
        {
            reciprocals[d] = w.Reciprocal(w.Scalar($"s{d}"));
        }

        w.Begin();
        string rest = w.Item, offset = w.V(), term = w.V();
        w.L($"v_mov_b32_e32 {offset}, 0");
        for (int d = ShapePermuteRank - 1; d >= 0; d--)
        {
            var (quotient, coordinate) = w.DivMod(rest, w.Scalar($"s{d}"), reciprocals[d]);
            w.L($"v_mul_lo_u32 {term}, {coordinate}, {w.Scalar($"t{d}")}");
            w.L($"v_add_u32_e32 {offset}, {offset}, {term}");
            rest = quotient;
        }

        string from = w.V(), value = w.V(), at = w.V(), old = w.V();
        w.L($"""
            v_lshlrev_b32_e32 {from}, 2, {offset}
            global_load_dword {value}, {from}, {w.Pointer("x")}
            v_lshlrev_b32_e32 {at}, 2, {w.Item}
            s_cmp_eq_u32 {w.Scalar("accumulate")}, 0
            s_cbranch_scc1 .Lstore
            global_load_dword {old}, {at}, {w.Pointer("y")}
            v_add_f32_e32 {value}, {old}, {value}
            .Lstore:
            global_store_dword {at}, {value}, {w.Pointer("y")}
            """);
        return w.End();
    }

    // dst[dstoff + r · dststride + c] (+)= src[srcoff + r · srcstride + c] for element i = r · cols + c.
    private static GcnKernelSource ShapeCopy2D()
    {
        var w = new ElementWriter("copy2d_f32", ["src", "dst"], ["srcoff", "srcstride", "dstoff", "dststride", "cols", "accumulate", "n"]);
        string cols = w.Scalar("cols"), byCols = w.Reciprocal(cols);
        w.Begin();
        var (row, column) = w.DivMod(w.Item, cols, byCols);
        string from = w.V(), to = w.V(), value = w.V(), old = w.V();
        w.L($"""
            v_mul_lo_u32 {from}, {row}, {w.Scalar("srcstride")}
            v_add_u32_e32 {from}, {w.Scalar("srcoff")}, {from}
            v_add_u32_e32 {from}, {from}, {column}
            v_lshlrev_b32_e32 {from}, 2, {from}
            v_mul_lo_u32 {to}, {row}, {w.Scalar("dststride")}
            v_add_u32_e32 {to}, {w.Scalar("dstoff")}, {to}
            v_add_u32_e32 {to}, {to}, {column}
            v_lshlrev_b32_e32 {to}, 2, {to}
            global_load_dword {value}, {from}, {w.Pointer("src")}
            s_cmp_eq_u32 {w.Scalar("accumulate")}, 0
            s_cbranch_scc1 .Lstore
            global_load_dword {old}, {to}, {w.Pointer("dst")}
            v_add_f32_e32 {value}, {old}, {value}
            .Lstore:
            global_store_dword {to}, {value}, {w.Pointer("dst")}
            """);
        return w.End();
    }

    // ---- The writer ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Writes a one-element-a-work-item kernel's GCN text with virtual registers: its directives (the block of
    /// <see cref="ShapeBlock"/>, the dispatch packet for the grid's size), its arguments loaded once into scalar registers
    /// (pointers, then 32-bit values, in order), the grid-stride loop (<see cref="Begin"/>, <see cref="End"/>) over
    /// <see cref="Item"/> &lt; n, and the arithmetic the kernels share. Each kernel's text is written once a process.
    /// </summary>
    private sealed class ElementWriter
    {
        private readonly StringBuilder _text = new();
        private readonly string _name;
        private readonly string[] _pointers, _scalars;
        private readonly int _first;
        private readonly string _grid;
        private int _s = 100, _v = 100;

        public ElementWriter(string name, string[] pointers, string[] scalars)
        {
            (_name, _pointers, _scalars) = (name, pointers, scalars);
            Raw($".kernel {name}");
            foreach (string p in pointers)
            {
                Raw($".arg {p}, 8, global");
            }

            foreach (string s in scalars)
            {
                Raw($".arg {s}, 4, value");
            }

            Raw($".reqd_workgroup_size {ShapeBlock}, 1, 1");
            Raw($".user_sgpr dispatch_ptr");
            Raw($".user_sgpr kernarg_segment_ptr");
            Raw($".system_sgpr workgroup_id_x");
            int words = 2 * pointers.Length + scalars.Length;
            _first = _s;
            _s += words;
            for (int at = 0; at < words;)
            {
                int take = words - at >= 8 ? 8 : words - at >= 4 ? 4 : words - at >= 2 ? 2 : 1;
                string load = take == 1 ? F($"s_load_dword %s{_first + at}") : F($"s_load_dwordx{take} %s[{_first + at}:{_first + at + take - 1}]");
                L($"{load}, s[2:3], 0x{4 * at:x}");
                at += take;
            }

            _grid = S();
            string block = S();
            L($"s_load_dword {_grid}, s[0:1], 0xc                ; the grid's work-items (grid_size_x)");
            L($"s_lshl_b32 {block}, s4, 8");
            L($"v_add_u32_e32 {Item}, {block}, v0                ; this work-item's first element");
        }

        /// <summary>The work-item's element (stepped by the grid's work-items).</summary>
        public string Item => "%v1";

        public string Pointer(string name) => F($"%s[{_first + 2 * Array.IndexOf(_pointers, name)}:{_first + 2 * Array.IndexOf(_pointers, name) + 1}]");

        public string Scalar(string name) => F($"%s{_first + 2 * _pointers.Length + Array.IndexOf(_scalars, name)}");

        public string S() => F($"%s{_s++}");

        public string SPair()
        {
            int id = _s;
            _s += 2;
            return F($"%s[{id}:{id + 1}]");
        }

        public string V() => F($"%v{_v++}");

        /// <summary>Instruction lines (several split at their line breaks); a line ending in a colon is a label.</summary>
        public void L(FormattableString lines)
        {
            foreach (var part in lines.ToString(CultureInfo.InvariantCulture).Split('\n'))
            {
                string line = part.Trim();
                if (line.Length > 0)
                {
                    _text.Append(line.EndsWith(':') ? "" : "  ").Append(line).Append('\n');
                }
            }
        }

        private void Raw(FormattableString line) => _text.Append(line.ToString(CultureInfo.InvariantCulture)).Append('\n');

        private static string F(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

        /// <summary>The whole number a float at <paramref name="pointer"/> holds (a position on the device), read once, in a scalar register.</summary>
        public string Position(string pointer)
        {
            string bits = S(), value = V(), position = S();
            L($"""
                s_load_dword {bits}, {pointer}, 0x0
                v_cvt_u32_f32_e32 {value}, {bits}
                v_readfirstlane_b32 {position}, {value}
                """);
            return position;
        }

        /// <summary>The reciprocal LLVM's unsigned division by <paramref name="count"/> (a scalar, at least 1) takes, made once.</summary>
        public string Reciprocal(string count)
        {
            string r = V(), negated = S(), t = V();
            L($"""
                v_cvt_f32_u32_e32 {r}, {count}
                v_rcp_iflag_f32_e32 {r}, {r}
                v_mul_f32_e32 {r}, 0x4f7ffffe, {r}
                v_cvt_u32_f32_e32 {r}, {r}
                s_sub_i32 {negated}, 0, {count}
                v_mul_lo_u32 {t}, {negated}, {r}
                v_mul_hi_u32 {t}, {r}, {t}
                v_add_u32_e32 {r}, {r}, {t}
                """);
            return r;
        }

        /// <summary>x / count and x mod count (unsigned) through the count's <see cref="Reciprocal"/>: a high product, two corrections.</summary>
        public (string Quotient, string Remainder) DivMod(string x, string count, string reciprocal)
        {
            string q = V(), r = V(), t = V(), next = V();
            L($"""
                v_mul_hi_u32 {q}, {x}, {reciprocal}
                v_mul_lo_u32 {t}, {q}, {count}
                v_sub_u32_e32 {r}, {x}, {t}
                v_add_u32_e32 {next}, 1, {q}
                v_subrev_u32_e32 {t}, {count}, {r}
                v_cmp_le_u32_e32 vcc, {count}, {r}
                v_cndmask_b32_e32 {r}, {r}, {t}, vcc
                v_cndmask_b32_e32 {q}, {q}, {next}, vcc
                v_add_u32_e32 {next}, 1, {q}
                v_subrev_u32_e32 {t}, {count}, {r}
                v_cmp_le_u32_e32 vcc, {count}, {r}
                v_cndmask_b32_e32 {r}, {r}, {t}, vcc
                v_cndmask_b32_e32 {q}, {q}, {next}, vcc
                """);
            return (q, r);
        }

        /// <summary>target = n / d correctly rounded (LLVM's expansion of fdiv, float32 denormals kept), as the CPU divides.</summary>
        public void Div(string target, string n, string d)
        {
            string ds = V(), ns = V(), r = V(), e = V(), q = V(), e2 = V(), unused = SPair();
            L($"""
                v_div_scale_f32 {ds}, {unused}, {d}, {d}, {n}
                v_div_scale_f32 {ns}, vcc, {n}, {d}, {n}
                v_rcp_f32_e32 {r}, {ds}
                v_fma_f32 {e}, -{ds}, {r}, 1.0
                v_fma_f32 {r}, {e}, {r}, {r}
                v_mul_f32_e32 {q}, {ns}, {r}
                v_fma_f32 {e2}, -{ds}, {q}, {ns}
                v_fma_f32 {q}, {e2}, {r}, {q}
                v_fma_f32 {ds}, -{ds}, {q}, {ns}
                v_div_fmas_f32 {ds}, {ds}, {r}, {q}
                v_div_fixup_f32 {target}, {ds}, {d}, {n}
                """);
        }

        /// <summary>The bfloat16 bits of float <paramref name="x"/> as BFloat16Bits.Round makes them (to nearest, ties to even; a NaN quiet), in the low half.</summary>
        public string RoundBFloat16(string x)
        {
            string tie = V(), rounded = V(), nan = V();
            L($"""
                v_bfe_u32 {tie}, {x}, 16, 1
                v_add_u32_e32 {rounded}, 0x7fff, {x}
                v_add_u32_e32 {rounded}, {rounded}, {tie}
                v_lshrrev_b32_e32 {rounded}, 16, {rounded}
                v_lshrrev_b32_e32 {nan}, 16, {x}
                v_or_b32_e32 {nan}, 64, {nan}
                v_cmp_u_f32_e32 vcc, {x}, {x}
                v_cndmask_b32_e32 {rounded}, {rounded}, {nan}, vcc
                """);
            return rounded;
        }

        /// <summary>The loop's head: the lanes whose element is below n, the end when none is.</summary>
        public void Begin()
        {
            string saved = SPair();
            L($"""
                v_cmp_gt_u32_e32 vcc, {Scalar("n")}, {Item}
                s_and_saveexec_b64 {saved}, vcc
                s_cbranch_execz .Ldone
                .Lloop:
                """);
        }

        /// <summary>The loop's tail (each element steps by the grid's work-items) and the kernel's end; the kernel's source.</summary>
        public GcnKernelSource End()
        {
            L($"""
                v_add_u32_e32 {Item}, {_grid}, {Item}
                v_cmp_gt_u32_e32 vcc, {Scalar("n")}, {Item}
                s_and_b64 exec, exec, vcc
                s_cbranch_execnz .Lloop
                .Ldone:
                s_endpgm
                """);
            return new GcnKernelSource(_name, Encoding.UTF8.GetBytes(_text.ToString()));
        }
    }
}
