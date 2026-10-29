// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's convolution, pooling and resampling kernels in GCN (plan 35, G5c), with CUDA's names and arguments
// (PtxKernels.Advanced.cs, .Convolution.cs, .Resampling.cs) and the CPU backend's results (CpuBackend.Advanced.cs,
// .Vision.cs, .Resampling.cs) within the device-against-CPU tolerance, most of them bit for bit:
// - the unfolded patches (im2col_f32) and their adjoint (col2im_f32), which the convolution's composed path runs with the
//   products of GcnKernels.Products.cs (Backend.Convolution.cs), as CUDA's composed candidate does;
// - maximum pooling (maxpool_f32 with its arg-max, maxpool_bwd_gather_f32), average pooling (avgpool_f32,
//   avgpool_bwd_f32, the padding counted or not), adaptive pooling (average and maximum, with gradients), interpolation
//   (nearest and bilinear, with gradients) and the resampling with per-channel normalization (resize_normalize_f32);
// - the depthwise convolution (one input channel a group) and both its gradients.
//
// Every kernel but conv_dw_bwd_weight_f32 has one work-item an output, grid-stride (GcnKernels.Training.cs's frame
// "Single": blocks of 256, the lanes past n off): the forward passes read their window, the gradients gather what reads
// each input element (the windows that cover it, in output order), so every sum is in a fixed order, never atomics (rule
// 86), and the maximum pooling's gradient is deterministic. The divisions by a geometry value go through its reciprocal,
// made once at the kernel's start (LLVM's unsigned division); the windows' loops run on the scalar unit where their bounds
// are the geometry's (each lane's element masked in or out), else lane by lane, a lane leaving as its range ends.
// conv_dw_bwd_weight_f32 is RowWriter's: a block a weight (grid-stride), its threads over the positions, then the block's
// fixed LDS tree. CUDA's atomic maxpool_bwd_f32 (dx[argmax[i]] += dy[i] without the windows) has no kernel here: the
// library calls the gathered gradient, which needs the windows.
//
// For --profile's labels and --bench-hip's rates (HipBackend.ConvolutionCost): each kernel, its arguments (pointers, then
// 32-bit values, the float ones as their bits; PackArguments packs them), its unit, and its floating-point operations and
// global-memory bytes per unit (a division counted as one operation; integer work and selects not counted):
//
// | Kernel                    | Arguments                                                                          | Unit                    | Flops | Bytes |
// |---------------------------|------------------------------------------------------------------------------------|-------------------------|-------|-------|
// | im2col_f32                | x, cols; C, H, W, KW, SH, SW, PH, PW, OW, patch, khkw, ohow, DH, DW, n             | column entry            | 0     | 8     |
// | col2im_f32                | dcols, dx; C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, patch, DH, DW, n               | element of dx x window  | 1     | 4     |
// | maxpool_f32               | x, y, argmax; H, W, KH, KW, SH, SW, PH, PW, OW, ohow, n                            | output x window         | 1     | 4     |
// | maxpool_bwd_gather_f32    | dy, argmax, dx; H, W, KH, KW, SH, SW, PH, PW, OH, OW, n                            | element of dx           | 1     | 16    |
// | avgpool_f32               | x, y; H, W, KH, KW, SH, SW, PH, PW, OH, OW, countpad, bottom, right, n             | output x window         | 1     | 4     |
// | avgpool_bwd_f32           | dy, dx; (as avgpool_f32), n                                                        | element of dx x window  | 2     | 4     |
// | adaptive_avgpool_f32      | x, y; planes, height, width, outheight, outwidth, n                                | element of x            | 1     | 4     |
// | adaptive_avgpool_bwd_f32  | dy, dx; (as adaptive_avgpool_f32), n                                               | element of dx           | 3     | 12    |
// | adaptive_maxpool_f32      | x, y, argmax; (as adaptive_avgpool_f32), n                                         | element of x            | 1     | 4     |
// | adaptive_maxpool_bwd_f32  | dy, argmax, dx; (as adaptive_avgpool_f32), n                                       | element of dx           | 1     | 16    |
// | interpolate_f32           | x, y; planes, height, width, outheight, outwidth, mode, align, scaleh, scalew, n   | output                  | 7     | 20    |
// | interpolate_bwd_f32       | dy, dx; (as interpolate_f32), n                                                    | element of dx           | 12    | 24    |
// | resize_normalize_f32      | x, coefficients, values, y; planes, channels, height, width, outheight, outwidth, xtaps, ytaps, bytes, n | output x tap | 2 | 8 |
// | conv_dw_f32               | x, w, bias, y; N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW, F, G, flags, splits, n | output x window | 2 | 8     |
// | conv_dw_bwd_input_f32     | dy, w, dx; (the 18 geometry values), n                                             | element of dx x window  | 2     | 8     |
// | conv_dw_bwd_weight_f32    | x, dy, dw; (the 18 geometry values)                                                | weight x position       | 2     | 8     |
//
// The counts stay below 2^30 (byte offsets are 32-bit): the backend takes the host fallback past them. Flags of the
// depthwise forward: bit 0 a bias, the activation (ConvActivation) from bit 1 (CUDA's).
internal static partial class GcnKernels
{
    /// <summary>The convolution, pooling and resampling kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Convolution => _convolution ??= BuildConvolution();

    private static GcnKernelSource[]? _convolution;

    /// <summary>The geometry values every depthwise convolution kernel takes after its pointers, in order (CUDA's).</summary>
    internal static readonly string[] ConvolutionScalars = ["N", "C", "H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW", "DH", "DW", "F", "G", "flags", "splits"];

    private static GcnKernelSource[] BuildConvolution() =>
    [
        Im2Col(),
        Col2Im(),
        MaxPool(),
        MaxPoolGather(),
        AvgPool(backward: false),
        AvgPool(backward: true),
        AdaptivePool(maximum: false, backward: false),
        AdaptivePool(maximum: false, backward: true),
        AdaptivePool(maximum: true, backward: false),
        AdaptivePool(maximum: true, backward: true),
        Interpolate(backward: false),
        Interpolate(backward: true),
        ResizeNormalize(),
        DepthwiseConvolution(),
        DepthwiseInput(),
        DepthwiseWeight(),
    ];

    // ---- Helpers of the per-output kernels -----------------------------------------------------------------------------

    // Instruction lines (several split at their line breaks), each indented.
    private static void CvLines(PointwiseWriter w, FormattableString lines)
    {
        foreach (string line in lines.ToString(CultureInfo.InvariantCulture).Split('\n'))
        {
            string t = line.Trim();
            if (t.Length > 0)
            {
                w.I(t);
            }
        }
    }

    // A fresh scalar register of the element writer (the first of a fresh pair, the second never named).
    private static string CvScalar(PointwiseWriter w) => RowWriter.Lo(w.Mask());

    // A scalar register holding `line`'s result, set once at the kernel's start (the held lines): `line` names its target {0}.
    private static string CvHeld(PointwiseWriter w, string format)
    {
        string s = CvScalar(w);
        w.Held.Append("  ").Append(string.Format(CultureInfo.InvariantCulture, format, s)).Append('\n');
        return s;
    }

    // num / den of two uniform scalars, set once at the kernel's start (LLVM's unsigned division on the scalar unit).
    private static string CvHeldQuotient(PointwiseWriter w, string num, string den)
    {
        string f = w.V(), u = w.V(), negated = CvScalar(w), r = CvScalar(w), q = CvScalar(w), m = CvScalar(w), t = CvScalar(w), next = CvScalar(w);
        w.Held.Append(TrainingText($"""
              v_cvt_f32_u32_e32 {f}, {den}
              s_sub_i32 {negated}, 0, {den}
              v_rcp_iflag_f32_e32 {f}, {f}
              v_mul_f32_e32 {f}, 0x4f7ffffe, {f}
              v_cvt_u32_f32_e32 {u}, {f}
              v_readfirstlane_b32 {r}, {u}
              s_mul_i32 {negated}, {negated}, {r}
              s_mul_hi_u32 {negated}, {r}, {negated}
              s_add_i32 {r}, {r}, {negated}
              s_mul_hi_u32 {q}, {num}, {r}
              s_mul_i32 {t}, {q}, {den}
              s_sub_i32 {m}, {num}, {t}
              s_add_u32 {next}, {q}, 1
              s_sub_i32 {t}, {m}, {den}
              s_cmp_ge_u32 {m}, {den}
              s_cselect_b32 {m}, {t}, {m}
              s_cselect_b32 {q}, {next}, {q}
              s_add_u32 {next}, {q}, 1
              s_cmp_ge_u32 {m}, {den}
              s_cselect_b32 {q}, {next}, {q}

            """));
        return q;
    }

    // x / d and x mod d (x a vector, d a uniform scalar), the reciprocal of d made once.
    private static (string Quotient, string Remainder) CvDivMod(PointwiseWriter w, string x, string d, Dictionary<string, string> reciprocals)
    {
        if (!reciprocals.TryGetValue(d, out string? r))
        {
            r = TrainingReciprocal(w, d);
            reciprocals[d] = r;
        }

        return TrainingDivMod(w, x, d, r);
    }

    // A loop of the scalar unit: counter = 0 … count - 1 (count a uniform scalar), `body` inside.
    private static void CvScalarLoop(PointwiseWriter w, string label, string counter, string count, Action body)
    {
        CvLines(w, $"""
            s_mov_b32 {counter}, 0
            {label}:
            s_cmp_ge_u32 {counter}, {count}
            s_cbranch_scc1 {label}_end
            """);
        body();
        CvLines(w, $"""
            s_add_u32 {counter}, {counter}, 1
            s_branch {label}
            {label}_end:
            """);
    }

    // A loop lane by lane: counter = first … last - 1 (unsigned, per lane), each lane leaving once its range ends.
    private static void CvLaneLoop(PointwiseWriter w, string label, string counter, string first, string last, Action body)
    {
        string saved = w.Mask();
        CvLines(w, $"""
            s_mov_b64 {saved}, exec
            v_mov_b32_e32 {counter}, {first}
            v_cmp_gt_u32_e32 vcc, {last}, {counter}
            s_and_b64 exec, exec, vcc
            s_cbranch_execz {label}_end
            {label}:
            """);
        body();
        CvLines(w, $"""
            v_add_u32_e32 {counter}, 1, {counter}
            v_cmp_gt_u32_e32 vcc, {last}, {counter}
            s_and_b64 exec, exec, vcc
            s_cbranch_execnz {label}
            {label}_end:
            s_mov_b64 exec, {saved}
            """);
    }

    // `body` for the lanes where `condition` (vcc or a lane mask, set just before) holds; skipped when none does.
    private static void CvMasked(PointwiseWriter w, string label, string condition, Action body)
    {
        string saved = w.Mask();
        CvLines(w, $"""
            s_and_saveexec_b64 {saved}, {condition}
            s_cbranch_execz {label}_masked
            """);
        body();
        CvLines(w, $"""
            {label}_masked:
            s_mov_b64 exec, {saved}
            """);
    }

    // target = a · b + c (unsigned 32-bit; a or b a vector register).
    private static void CvMad(PointwiseWriter w, string target, string a, string b, string c)
    {
        CvLines(w, $"""
            v_mul_lo_u32 {target}, {a}, {b}
            v_add_u32_e32 {target}, {c}, {target}
            """);
    }

    // A byte offset (element · 4) into a fresh register.
    private static string CvBytes(PointwiseWriter w, string element)
    {
        string at = w.V();
        CvLines(w, $"v_lshlrev_b32_e32 {at}, 2, {element}");
        return at;
    }

    // A lane mask of every lane when the uniform scalar `value` is not 0 (or, `equal`, when it equals `other`), of none otherwise.
    private static string CvUniformMask(PointwiseWriter w, string value, string? other = null)
    {
        string mask = w.Mask();
        CvLines(w, other is null
            ? (FormattableString)$"""
                s_cmp_lg_u32 {value}, 0
                s_cselect_b64 {mask}, -1, 0
                """
            : $"""
                s_cmp_eq_u32 {value}, {other}
                s_cselect_b64 {mask}, -1, 0
                """);
        return mask;
    }

    // ---- Unfolded patches ----------------------------------------------------------------------------------------------

    // cols[i] = the input value column entry i reads (0 outside the image): i = (position, (c, kh, kw)), the position
    // (n, oh, ow), the input (n, c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW).
    private static GcnKernelSource Im2Col() =>
        TrainingElementwise("im2col_f32", new(["x", "cols"], ["C", "H", "W", "KW", "SH", "SW", "PH", "PW", "OW", "patch", "khkw", "ohow", "DH", "DW"], [], [1])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (position, column) = CvDivMod(w, e.Index, v[9], r);
            var (image, inImage) = CvDivMod(w, position, v[11], r);
            var (oh, ow) = CvDivMod(w, inImage, v[8], r);
            var (c, k) = CvDivMod(w, column, v[10], r);
            var (kh, kw) = CvDivMod(w, k, v[3], r);
            string ih = w.V(), iw = w.V(), t = w.V(), at = w.V(), inside = w.Mask(), value = e.Out[1];
            CvLines(w, $"""
                v_mul_lo_u32 {ih}, {oh}, {v[4]}
                v_mul_lo_u32 {t}, {kh}, {v[12]}
                v_add_u32_e32 {ih}, {ih}, {t}
                v_subrev_u32_e32 {ih}, {v[6]}, {ih}
                v_mul_lo_u32 {iw}, {ow}, {v[5]}
                v_mul_lo_u32 {t}, {kw}, {v[13]}
                v_add_u32_e32 {iw}, {iw}, {t}
                v_subrev_u32_e32 {iw}, {v[7]}, {iw}
                v_mul_lo_u32 {at}, {image}, {v[0]}
                v_add_u32_e32 {at}, {at}, {c}
                v_mul_lo_u32 {at}, {at}, {v[1]}
                v_add_u32_e32 {at}, {at}, {ih}
                v_mul_lo_u32 {at}, {at}, {v[2]}
                v_add_u32_e32 {at}, {at}, {iw}
                v_lshlrev_b32_e32 {at}, 2, {at}
                v_mov_b32_e32 {value}, 0
                v_cmp_gt_u32_e32 vcc, {v[1]}, {ih}
                v_cmp_gt_u32_e64 {inside}, {v[2]}, {iw}
                s_and_b64 vcc, vcc, {inside}
                """);
            CvMasked(w, e.Label + "_in", "vcc", () => CvLines(w, $"global_load_dword {value}, {at}, {e.Pointers[0]}"));
        });

    // dx[i] += the column entries that read input element i, the windows in position order (oh, then ow, rising: kh, then
    // kw, falling), added to dx's value as the CPU's fold adds them.
    private static GcnKernelSource Col2Im() =>
        TrainingElementwise("col2im_f32", new(["dcols", "dx"], ["C", "H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW", "patch", "DH", "DW"], [1], [1])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (planeRow, iw) = CvDivMod(w, e.Index, v[2], r);
            var (plane, ih) = CvDivMod(w, planeRow, v[1], r);
            var (image, c) = CvDivMod(w, plane, v[0], r);
            string sum = w.V(), ihp = w.V(), iwp = w.V(), rowBase = w.V(), colBase = w.V();
            CvLines(w, $"""
                v_mov_b32_e32 {sum}, {e.In[1]}
                v_add_u32_e32 {ihp}, {v[7]}, {ih}
                v_add_u32_e32 {iwp}, {v[8]}, {iw}
                v_mul_lo_u32 {rowBase}, {image}, {v[9]}
                v_mul_lo_u32 {colBase}, {c}, {v[3]}
                """);
            string kh = CvScalar(w), khDh = CvScalar(w), kw = CvScalar(w), kwDw = CvScalar(w);
            string t = w.V(), u = w.V(), rowAt = w.V(), colAt = w.V(), at = w.V(), value = w.V(), ok = w.Mask(), ok2 = w.Mask();
            string rows = e.Label + "_kh", columns = e.Label + "_kw";
            CvLines(w, $"""
                s_mov_b32 {kh}, {v[3]}
                {rows}:
                s_cmp_eq_u32 {kh}, 0
                s_cbranch_scc1 {rows}_end
                s_sub_u32 {kh}, {kh}, 1
                s_mul_i32 {khDh}, {kh}, {v[12]}
                v_subrev_u32_e32 {t}, {khDh}, {ihp}
                """);
            var (oh, ohRest) = CvDivMod(w, t, v[5], r);
            CvLines(w, $"""
                v_cmp_lt_i32_e32 vcc, -1, {t}
                v_cmp_eq_u32_e64 {ok}, 0, {ohRest}
                s_and_b64 vcc, vcc, {ok}
                v_cmp_gt_u32_e64 {ok}, {v[9]}, {oh}
                s_and_b64 vcc, vcc, {ok}
                """);
            CvMasked(w, rows, "vcc", () =>
            {
                CvLines(w, $"""
                    v_add_u32_e32 {rowAt}, {rowBase}, {oh}
                    v_mul_lo_u32 {rowAt}, {rowAt}, {v[10]}
                    v_add_u32_e32 {colAt}, {kh}, {colBase}
                    v_mul_lo_u32 {colAt}, {colAt}, {v[4]}
                    s_mov_b32 {kw}, {v[4]}
                    {columns}:
                    s_cmp_eq_u32 {kw}, 0
                    s_cbranch_scc1 {columns}_end
                    s_sub_u32 {kw}, {kw}, 1
                    s_mul_i32 {kwDw}, {kw}, {v[13]}
                    v_subrev_u32_e32 {u}, {kwDw}, {iwp}
                    """);
                var (ow, owRest) = CvDivMod(w, u, v[6], r);
                CvLines(w, $"""
                    v_add_u32_e32 {at}, {rowAt}, {ow}
                    v_mul_lo_u32 {at}, {at}, {v[11]}
                    v_add_u32_e32 {at}, {at}, {colAt}
                    v_add_u32_e32 {at}, {kw}, {at}
                    v_lshlrev_b32_e32 {at}, 2, {at}
                    v_cmp_lt_i32_e32 vcc, -1, {u}
                    v_cmp_eq_u32_e64 {ok2}, 0, {owRest}
                    s_and_b64 vcc, vcc, {ok2}
                    v_cmp_gt_u32_e64 {ok2}, {v[10]}, {ow}
                    s_and_b64 vcc, vcc, {ok2}
                    """);
                CvMasked(w, columns, "vcc", () => CvLines(w, $"""
                    global_load_dword {value}, {at}, {e.Pointers[0]}
                    v_add_f32_e32 {sum}, {sum}, {value}
                    """));
                CvLines(w, $"""
                    s_branch {columns}
                    {columns}_end:
                    """);
            });
            CvLines(w, $"""
                s_branch {rows}
                {rows}_end:
                v_mov_b32_e32 {e.Out[1]}, {sum}
                """);
        });

    // ---- Maximum pooling -----------------------------------------------------------------------------------------------

    // y = the window's maximum, argmax = its flat input index (int bits), the first maximum in row order kept (CUDA's).
    private static GcnKernelSource MaxPool() =>
        TrainingElementwise("maxpool_f32", new(["x", "y", "argmax"], ["H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OW", "ohow"], [], [1, 2])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (plane, inPlane) = CvDivMod(w, e.Index, v[9], r);
            var (oh, ow) = CvDivMod(w, inPlane, v[8], r);
            string area = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {v[0]}, {v[1]}"));
            string planeAt = w.V(), r0 = w.V(), c0 = w.V(), best = w.V(), bestAt = w.V(), ih = w.V(), iw = w.V(), rowAt = w.V(), at = w.V(), bytes = w.V(), value = w.V();
            CvLines(w, $"""
                v_mul_lo_u32 {planeAt}, {plane}, {area}
                v_mul_lo_u32 {r0}, {oh}, {v[4]}
                v_subrev_u32_e32 {r0}, {v[6]}, {r0}
                v_mul_lo_u32 {c0}, {ow}, {v[5]}
                v_subrev_u32_e32 {c0}, {v[7]}, {c0}
                v_mov_b32_e32 {best}, {RowWriter.NegativeInfinity}
                v_mov_b32_e32 {bestAt}, {planeAt}
                """);
            string kh = CvScalar(w), kw = CvScalar(w);
            CvScalarLoop(w, e.Label + "_kh", kh, v[2], () =>
            {
                CvLines(w, $"""
                    v_add_u32_e32 {ih}, {kh}, {r0}
                    v_cmp_gt_u32_e32 vcc, {v[0]}, {ih}
                    """);
                CvMasked(w, e.Label + "_row", "vcc", () =>
                {
                    CvMad(w, rowAt, ih, v[1], planeAt);
                    CvScalarLoop(w, e.Label + "_kw", kw, v[3], () =>
                    {
                        CvLines(w, $"""
                            v_add_u32_e32 {iw}, {kw}, {c0}
                            v_cmp_gt_u32_e32 vcc, {v[1]}, {iw}
                            """);
                        CvMasked(w, e.Label + "_col", "vcc", () => CvLines(w, $"""
                            v_add_u32_e32 {at}, {rowAt}, {iw}
                            v_lshlrev_b32_e32 {bytes}, 2, {at}
                            global_load_dword {value}, {bytes}, {e.Pointers[0]}
                            v_cmp_gt_f32_e32 vcc, {value}, {best}
                            v_cndmask_b32_e32 {best}, {best}, {value}, vcc
                            v_cndmask_b32_e32 {bestAt}, {bestAt}, {at}, vcc
                            """));
                    });
                });
            });
            CvLines(w, $"""
                v_mov_b32_e32 {e.Out[1]}, {best}
                v_mov_b32_e32 {e.Out[2]}, {bestAt}
                """);
        });

    // The windows [first, last) along one axis whose span covers input coordinate i (o·stride - pad ≤ i < o·stride - pad +
    // size), the CPU's Covering: first = ⌈(i + pad - size + 1) / stride⌉ (0 when that is not positive), last = min((i + pad)
    // / stride + 1, outputs).
    private static (string First, string Last) CvCovering(PointwiseWriter w, string i, string pad, string size, string stride, string outputs,
        Dictionary<string, string> r)
    {
        string ip = w.V(), low = w.V(), up = w.V(), first = w.V(), last = w.V();
        CvLines(w, $"""
            v_add_u32_e32 {ip}, {pad}, {i}
            v_subrev_u32_e32 {low}, {size}, {ip}
            v_add_u32_e32 {low}, 1, {low}
            v_add_u32_e32 {up}, {stride}, {low}
            v_add_u32_e32 {up}, -1, {up}
            """);
        var (ceiling, _) = CvDivMod(w, up, stride, r);
        var (floor, _) = CvDivMod(w, ip, stride, r);
        CvLines(w, $"""
            v_cmp_lt_i32_e32 vcc, 0, {low}
            v_cndmask_b32_e32 {first}, 0, {ceiling}, vcc
            v_add_u32_e32 {last}, 1, {floor}
            v_min_u32_e32 {last}, {outputs}, {last}
            """);
        return (first, last);
    }

    // dx[i] += dy of each window whose maximum is element i (argmax = i), the covering windows in output order (oh, then ow).
    private static GcnKernelSource MaxPoolGather() =>
        TrainingElementwise("maxpool_bwd_gather_f32", new(["dy", "argmax", "dx"], ["H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW"], [2], [2])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (planeRow, iw) = CvDivMod(w, e.Index, v[1], r);
            var (plane, ih) = CvDivMod(w, planeRow, v[0], r);
            var (rowFirst, rowLast) = CvCovering(w, ih, v[6], v[2], v[4], v[8], r);
            var (colFirst, colLast) = CvCovering(w, iw, v[7], v[3], v[5], v[9], r);
            string sum = w.V(), planeAt = w.V(), oh = w.V(), ow = w.V(), rowAt = w.V(), at = w.V(), index = w.V(), gradient = w.V(), added = w.V();
            CvLines(w, $"""
                v_mov_b32_e32 {sum}, {e.In[2]}
                v_mul_lo_u32 {planeAt}, {plane}, {v[8]}
                """);
            CvLaneLoop(w, e.Label + "_oh", oh, rowFirst, rowLast, () =>
            {
                CvLines(w, $"""
                    v_add_u32_e32 {rowAt}, {planeAt}, {oh}
                    v_mul_lo_u32 {rowAt}, {rowAt}, {v[9]}
                    """);
                CvLaneLoop(w, e.Label + "_ow", ow, colFirst, colLast, () => CvLines(w, $"""
                    v_add_u32_e32 {at}, {rowAt}, {ow}
                    v_lshlrev_b32_e32 {at}, 2, {at}
                    global_load_dword {index}, {at}, {e.Pointers[1]}
                    global_load_dword {gradient}, {at}, {e.Pointers[0]}
                    v_add_f32_e32 {added}, {sum}, {gradient}
                    v_cmp_eq_u32_e32 vcc, {index}, {e.Index}
                    v_cndmask_b32_e32 {sum}, {sum}, {added}, vcc
                    """));
            });
            CvLines(w, $"v_mov_b32_e32 {e.Out[2]}, {sum}");
        });

    // ---- Average pooling -----------------------------------------------------------------------------------------------

    // The divisor of the window at rows from r0 and columns from c0 (signed), as a float: the positions it covers in the
    // image (at least 1), or (countpad) its rows and columns up to the padded end (H + bottom, W + right).
    private static string PoolDivisor(PointwiseWriter w, string[] v, string r0, string c0, string countPad, string paddedH, string paddedW)
    {
        string re = w.V(), rs = w.V(), ce = w.V(), cs = w.V(), area = w.V(), padded = w.V(), t = w.V(), divisor = w.V();
        CvLines(w, $"""
            v_add_u32_e32 {re}, {v[2]}, {r0}
            v_min_i32_e32 {t}, {v[0]}, {re}
            v_max_i32_e32 {rs}, 0, {r0}
            v_sub_u32_e32 {rs}, {t}, {rs}
            v_add_u32_e32 {ce}, {v[3]}, {c0}
            v_min_i32_e32 {t}, {v[1]}, {ce}
            v_max_i32_e32 {cs}, 0, {c0}
            v_sub_u32_e32 {cs}, {t}, {cs}
            v_mul_lo_u32 {area}, {rs}, {cs}
            v_max_i32_e32 {area}, 1, {area}
            v_min_i32_e32 {re}, {paddedH}, {re}
            v_sub_u32_e32 {re}, {re}, {r0}
            v_min_i32_e32 {ce}, {paddedW}, {ce}
            v_sub_u32_e32 {ce}, {ce}, {c0}
            v_mul_lo_u32 {padded}, {re}, {ce}
            v_cndmask_b32_e64 {area}, {area}, {padded}, {countPad}
            v_cvt_f32_i32_e32 {divisor}, {area}
            """);
        return divisor;
    }

    // Forward: y = the window's sum (rows, then columns, in order) / its divisor. Gradient: dx[i] += dy / divisor of each
    // window covering element i, in window order (oh, then ow).
    private static GcnKernelSource AvgPool(bool backward) =>
        TrainingElementwise(backward ? "avgpool_bwd_f32" : "avgpool_f32",
            new([backward ? "dy" : "x", backward ? "dx" : "y"], ["H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW", "countpad", "bottom", "right"],
                backward ? [1] : [], [1])
            {
                Index = true,
                Single = true,
            }, (w, e) =>
            {
                string[] v = e.Values;
                var r = new Dictionary<string, string>();
                string countPad = CvUniformMask(w, v[10]);
                string paddedH = CvHeld(w, TrainingText($"s_add_u32 {{0}}, {v[0]}, {v[11]}")), paddedW = CvHeld(w, TrainingText($"s_add_u32 {{0}}, {v[1]}, {v[12]}"));
                if (!backward)
                {
                    var (planeRow, ow) = CvDivMod(w, e.Index, v[9], r);
                    var (plane, oh) = CvDivMod(w, planeRow, v[8], r);
                    string area = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {v[0]}, {v[1]}"));
                    string planeAt = w.V(), r0 = w.V(), c0 = w.V(), sum = w.V(), ih = w.V(), iw = w.V(), rowAt = w.V(), at = w.V(), value = w.V();
                    CvLines(w, $"""
                        v_mul_lo_u32 {planeAt}, {plane}, {area}
                        v_mul_lo_u32 {r0}, {oh}, {v[4]}
                        v_subrev_u32_e32 {r0}, {v[6]}, {r0}
                        v_mul_lo_u32 {c0}, {ow}, {v[5]}
                        v_subrev_u32_e32 {c0}, {v[7]}, {c0}
                        v_mov_b32_e32 {sum}, 0
                        """);
                    string kh = CvScalar(w), kw = CvScalar(w);
                    CvScalarLoop(w, e.Label + "_kh", kh, v[2], () =>
                    {
                        CvLines(w, $"""
                            v_add_u32_e32 {ih}, {kh}, {r0}
                            v_cmp_gt_u32_e32 vcc, {v[0]}, {ih}
                            """);
                        CvMasked(w, e.Label + "_row", "vcc", () =>
                        {
                            CvMad(w, rowAt, ih, v[1], planeAt);
                            CvScalarLoop(w, e.Label + "_kw", kw, v[3], () =>
                            {
                                CvLines(w, $"""
                                    v_add_u32_e32 {iw}, {kw}, {c0}
                                    v_cmp_gt_u32_e32 vcc, {v[1]}, {iw}
                                    """);
                                CvMasked(w, e.Label + "_col", "vcc", () => CvLines(w, $"""
                                    v_add_u32_e32 {at}, {rowAt}, {iw}
                                    v_lshlrev_b32_e32 {at}, 2, {at}
                                    global_load_dword {value}, {at}, {e.Pointers[0]}
                                    v_add_f32_e32 {sum}, {sum}, {value}
                                    """));
                            });
                        });
                    });
                    string divisor = PoolDivisor(w, v, r0, c0, countPad, paddedH, paddedW);
                    w.Div(e.Out[1], sum, divisor);
                    return;
                }

                var (inRow, iwx) = CvDivMod(w, e.Index, v[1], r);
                var (planeIndex, ihx) = CvDivMod(w, inRow, v[0], r);
                var (rowFirst, rowLast) = CvCovering(w, ihx, v[6], v[2], v[4], v[8], r);
                var (colFirst, colLast) = CvCovering(w, iwx, v[7], v[3], v[5], v[9], r);
                string total = w.V(), planeOut = w.V(), ohv = w.V(), owv = w.V(), r0v = w.V(), c0v = w.V(), rowOut = w.V(), atOut = w.V(), g = w.V(), share = w.V();
                CvLines(w, $"""
                    v_mov_b32_e32 {total}, {e.In[1]}
                    v_mul_lo_u32 {planeOut}, {planeIndex}, {v[8]}
                    """);
                CvLaneLoop(w, e.Label + "_oh", ohv, rowFirst, rowLast, () =>
                {
                    CvLines(w, $"""
                        v_mul_lo_u32 {r0v}, {ohv}, {v[4]}
                        v_subrev_u32_e32 {r0v}, {v[6]}, {r0v}
                        v_add_u32_e32 {rowOut}, {planeOut}, {ohv}
                        v_mul_lo_u32 {rowOut}, {rowOut}, {v[9]}
                        """);
                    CvLaneLoop(w, e.Label + "_ow", owv, colFirst, colLast, () =>
                    {
                        CvLines(w, $"""
                            v_mul_lo_u32 {c0v}, {owv}, {v[5]}
                            v_subrev_u32_e32 {c0v}, {v[7]}, {c0v}
                            v_add_u32_e32 {atOut}, {rowOut}, {owv}
                            v_lshlrev_b32_e32 {atOut}, 2, {atOut}
                            global_load_dword {g}, {atOut}, {e.Pointers[0]}
                            """);
                        string divisor = PoolDivisor(w, v, r0v, c0v, countPad, paddedH, paddedW);
                        w.Div(share, g, divisor);
                        CvLines(w, $"v_add_f32_e32 {total}, {total}, {share}");
                    });
                });
                CvLines(w, $"v_mov_b32_e32 {e.Out[1]}, {total}");
            });

    // ---- Adaptive pooling ----------------------------------------------------------------------------------------------

    // Adaptive window o of an axis: [⌊o · size / outSize⌋, ⌈(o + 1) · size / outSize⌉).
    private static (string Start, string End) AdaptiveWindow(PointwiseWriter w, string o, string size, string outSize, Dictionary<string, string> r)
    {
        string t = w.V(), u = w.V();
        CvLines(w, $"""
            v_mul_lo_u32 {t}, {o}, {size}
            v_add_u32_e32 {u}, {size}, {t}
            v_add_u32_e32 {u}, {outSize}, {u}
            v_add_u32_e32 {u}, -1, {u}
            """);
        var (start, _) = CvDivMod(w, t, outSize, r);
        var (end, _) = CvDivMod(w, u, outSize, r);
        return (start, end);
    }

    // The adaptive windows that can cover input position i: [⌊i · outSize / size⌋ - 1, ⌈(i + 1) · outSize / size⌉ + 1), clipped.
    private static (string First, string Last) AdaptiveCovers(PointwiseWriter w, string i, string size, string outSize, Dictionary<string, string> r)
    {
        string t = w.V(), u = w.V(), first = w.V(), last = w.V();
        CvLines(w, $"""
            v_mul_lo_u32 {t}, {i}, {outSize}
            v_add_u32_e32 {u}, {outSize}, {t}
            v_add_u32_e32 {u}, {size}, {u}
            v_add_u32_e32 {u}, -1, {u}
            """);
        var (low, _) = CvDivMod(w, t, size, r);
        var (high, _) = CvDivMod(w, u, size, r);
        CvLines(w, $"""
            v_add_u32_e32 {first}, -1, {low}
            v_max_i32_e32 {first}, 0, {first}
            v_add_u32_e32 {last}, 1, {high}
            v_min_u32_e32 {last}, {outSize}, {last}
            """);
        return (first, last);
    }

    // PyTorch's adaptive pooling (the CPU's): the average of each window (sum / rows / columns) or its maximum (the first
    // in row order, a NaN wins) with the flat input index; the gradients gather over the windows that can cover each
    // element, in window order.
    private static GcnKernelSource AdaptivePool(bool maximum, bool backward)
    {
        string name = (maximum ? "adaptive_maxpool" : "adaptive_avgpool") + (backward ? "_bwd_f32" : "_f32");
        string[] buffers = (maximum, backward) switch
        {
            (false, false) => ["x", "y"],
            (false, true) => ["dy", "dx"],
            (true, false) => ["x", "y", "argmax"],
            _ => ["dy", "argmax", "dx"],
        };
        int output = backward ? buffers.Length - 1 : 1;
        int[] written = maximum && !backward ? [1, 2] : [output];
        return TrainingElementwise(name, new(buffers, ["planes", "height", "width", "outheight", "outwidth"], backward ? [output] : [], written)
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            if (!backward)
            {
                var (planeRow, ow) = CvDivMod(w, e.Index, v[4], r);
                var (plane, oh) = CvDivMod(w, planeRow, v[3], r);
                var (rs, re) = AdaptiveWindow(w, oh, v[1], v[3], r);
                var (cs, ce) = AdaptiveWindow(w, ow, v[2], v[4], r);
                string rowBase = w.V(), rowAt = w.V(), row = w.V(), col = w.V(), at = w.V(), value = w.V(), result = w.V(), local = w.V(), best = w.Mask();
                CvLines(w, $"v_mul_lo_u32 {rowBase}, {plane}, {v[1]}");
                CvLines(w, maximum
                    ? (FormattableString)$"""
                        v_mov_b32_e32 {result}, {RowWriter.NegativeInfinity}
                        v_mul_lo_u32 {local}, {rs}, {v[2]}
                        v_add_u32_e32 {local}, {local}, {cs}
                        """
                    : $"v_mov_b32_e32 {result}, 0");
                CvLaneLoop(w, e.Label + "_r", row, rs, re, () =>
                {
                    CvLines(w, $"""
                        v_add_u32_e32 {rowAt}, {rowBase}, {row}
                        v_mul_lo_u32 {rowAt}, {rowAt}, {v[2]}
                        """);
                    CvLaneLoop(w, e.Label + "_c", col, cs, ce, () =>
                    {
                        CvLines(w, $"""
                            v_add_u32_e32 {at}, {rowAt}, {col}
                            v_lshlrev_b32_e32 {at}, 2, {at}
                            global_load_dword {value}, {at}, {e.Pointers[0]}
                            """);
                        CvLines(w, maximum
                            ? (FormattableString)$"""
                                v_cmp_gt_f32_e64 {best}, {value}, {result}
                                v_cmp_u_f32_e32 vcc, {value}, {value}
                                s_or_b64 vcc, vcc, {best}
                                v_cndmask_b32_e32 {result}, {result}, {value}, vcc
                                v_mul_lo_u32 {at}, {row}, {v[2]}
                                v_add_u32_e32 {at}, {at}, {col}
                                v_cndmask_b32_e32 {local}, {local}, {at}, vcc
                                """
                            : $"v_add_f32_e32 {result}, {result}, {value}");
                    });
                });
                if (maximum)
                {
                    string area = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {v[1]}, {v[2]}")), planeAt = w.V();
                    CvLines(w, $"""
                        v_mul_lo_u32 {planeAt}, {plane}, {area}
                        v_add_u32_e32 {e.Out[2]}, {planeAt}, {local}
                        v_mov_b32_e32 {e.Out[1]}, {result}
                        """);
                    return;
                }

                string rows = w.V(), cols = w.V(), half = w.V();
                CvLines(w, $"""
                    v_sub_u32_e32 {rows}, {re}, {rs}
                    v_cvt_f32_u32_e32 {rows}, {rows}
                    v_sub_u32_e32 {cols}, {ce}, {cs}
                    v_cvt_f32_u32_e32 {cols}, {cols}
                    """);
                w.Div(half, result, rows);
                w.Div(e.Out[1], half, cols);
                return;
            }

            var (inRow, iw) = CvDivMod(w, e.Index, v[2], r);
            var (planeIndex, ih) = CvDivMod(w, inRow, v[1], r);
            var (rowFirst, rowLast) = AdaptiveCovers(w, ih, v[1], v[3], r);
            var (colFirst, colLast) = AdaptiveCovers(w, iw, v[2], v[4], r);
            string sum = w.V(), planeOut = w.V(), o = w.V(), oc = w.V(), rowOut = w.V(), atOut = w.V(), gradient = w.V(), added = w.V(), rowsF = w.V(), colsF = w.V(),
                part = w.V(), index = w.V(), inRows = w.Mask(), inCols = w.Mask();
            CvLines(w, $"""
                v_mov_b32_e32 {sum}, {e.In[output]}
                v_mul_lo_u32 {planeOut}, {planeIndex}, {v[3]}
                """);
            CvLaneLoop(w, e.Label + "_o", o, rowFirst, rowLast, () =>
            {
                CvLines(w, $"""
                    v_add_u32_e32 {rowOut}, {planeOut}, {o}
                    v_mul_lo_u32 {rowOut}, {rowOut}, {v[4]}
                    """);
                if (!maximum)
                {
                    var (rs, re) = AdaptiveWindow(w, o, v[1], v[3], r);
                    string notBelow = w.Mask();
                    CvLines(w, $"""
                        v_cmp_le_u32_e64 {notBelow}, {rs}, {ih}
                        v_cmp_gt_u32_e64 {inRows}, {re}, {ih}
                        s_and_b64 {inRows}, {inRows}, {notBelow}
                        v_sub_u32_e32 {rowsF}, {re}, {rs}
                        v_cvt_f32_u32_e32 {rowsF}, {rowsF}
                        """);
                }

                CvLaneLoop(w, e.Label + "_oc", oc, colFirst, colLast, () =>
                {
                    CvLines(w, $"""
                        v_add_u32_e32 {atOut}, {rowOut}, {oc}
                        v_lshlrev_b32_e32 {atOut}, 2, {atOut}
                        global_load_dword {gradient}, {atOut}, {e.Pointers[0]}
                        """);
                    if (maximum)
                    {
                        CvLines(w, $"""
                            global_load_dword {index}, {atOut}, {e.Pointers[1]}
                            v_add_f32_e32 {added}, {sum}, {gradient}
                            v_cmp_eq_u32_e32 vcc, {index}, {e.Index}
                            v_cndmask_b32_e32 {sum}, {sum}, {added}, vcc
                            """);
                        return;
                    }

                    var (cs, ce) = AdaptiveWindow(w, oc, v[2], v[4], r);
                    string notLeft = w.Mask();
                    CvLines(w, $"""
                        v_cmp_le_u32_e64 {notLeft}, {cs}, {iw}
                        v_cmp_gt_u32_e64 {inCols}, {ce}, {iw}
                        s_and_b64 {inCols}, {inCols}, {notLeft}
                        v_sub_u32_e32 {colsF}, {ce}, {cs}
                        v_cvt_f32_u32_e32 {colsF}, {colsF}
                        """);
                    w.Div(part, gradient, rowsF);
                    w.Div(added, part, colsF);
                    CvLines(w, $"""
                        v_add_f32_e32 {added}, {sum}, {added}
                        s_and_b64 vcc, {inRows}, {inCols}
                        v_cndmask_b32_e32 {sum}, {sum}, {added}, vcc
                        """);
                });
            });
            CvLines(w, $"v_mov_b32_e32 {e.Out[output]}, {sum}");
        });
    }

    // ---- Interpolation -------------------------------------------------------------------------------------------------

    // The input positions output position `o` of one axis reads and their weights, as the CPU computes them (CpuBackend's
    // Axis): nearest min(⌊o · scale⌋, size - 1) with the exact cases output = input and output = 2 · input; bilinear ⌊s⌋ and
    // the next position (the last reads itself twice) with weights 1 - λ and λ, s = o · scale (align) or max((o + 0.5) ·
    // scale - 0.5, 0), every step rounded alone. `nearest`, `align`, `same` and `twice` are uniform lane masks.
    private static (string First, string Second, string W0, string W1) AxisTaps(PointwiseWriter w, string o, string scale, string last, string nearest,
        string align, string same, string twice)
    {
        string fo = w.V(), s = w.V(), n = w.V(), half = w.V(), centered = w.V(), floor = w.V(), index = w.V(), fi = w.V(), lambda = w.V(), next = w.V();
        string first = w.V(), second = w.V(), w0 = w.V(), w1 = w.V(), bilinear = w.V();
        CvLines(w, $"""
            v_cvt_f32_u32_e32 {fo}, {o}
            v_mul_f32_e32 {s}, {scale}, {fo}
            v_floor_f32_e32 {floor}, {s}
            v_cvt_i32_f32_e32 {n}, {floor}
            v_min_i32_e32 {n}, {last}, {n}
            v_lshrrev_b32_e32 {half}, 1, {o}
            v_cndmask_b32_e64 {n}, {n}, {half}, {twice}
            v_cndmask_b32_e64 {n}, {n}, {o}, {same}
            v_add_f32_e32 {centered}, 0.5, {fo}
            v_mul_f32_e32 {centered}, {scale}, {centered}
            v_add_f32_e32 {centered}, -0.5, {centered}
            v_max_f32_e32 {centered}, 0, {centered}
            v_cndmask_b32_e64 {s}, {centered}, {s}, {align}
            v_floor_f32_e32 {floor}, {s}
            v_cvt_i32_f32_e32 {index}, {floor}
            v_min_i32_e32 {index}, {last}, {index}
            v_cvt_f32_i32_e32 {fi}, {index}
            v_sub_f32_e32 {lambda}, {s}, {fi}
            v_max_f32_e32 {lambda}, 0, {lambda}
            v_min_f32_e32 {lambda}, 1.0, {lambda}
            v_add_u32_e32 {next}, 1, {index}
            v_cmp_gt_i32_e32 vcc, {last}, {index}
            v_cndmask_b32_e32 {bilinear}, {index}, {next}, vcc
            v_sub_f32_e32 {w0}, 1.0, {lambda}
            v_cndmask_b32_e64 {first}, {index}, {n}, {nearest}
            v_cndmask_b32_e64 {second}, {bilinear}, {n}, {nearest}
            v_cndmask_b32_e64 {w0}, {w0}, 1.0, {nearest}
            v_cndmask_b32_e64 {w1}, {lambda}, 0, {nearest}
            """);
        return (first, second, w0, w1);
    }

    // The weight the taps put on input position i (nearest: w0 where first is i; bilinear: w0 where first is, plus w1
    // where second is), and whether either reads it (into `reads`).
    private static string WeightOn(PointwiseWriter w, (string First, string Second, string W0, string W1) taps, string i, string bilinear, string reads)
    {
        string weight = w.V(), more = w.V(), onSecond = w.Mask();
        CvLines(w, $"""
            v_cmp_eq_u32_e32 vcc, {taps.First}, {i}
            v_cndmask_b32_e32 {weight}, 0, {taps.W0}, vcc
            v_cmp_eq_u32_e64 {onSecond}, {taps.Second}, {i}
            s_or_b64 {reads}, vcc, {onSecond}
            s_and_b64 {onSecond}, {onSecond}, {bilinear}
            v_add_f32_e32 {more}, {weight}, {taps.W1}
            v_cndmask_b32_e64 {weight}, {weight}, {more}, {onSecond}
            """);
        return weight;
    }

    // The outputs [first, last) whose source position can lie within two of input position i (all when the scale is not
    // above 0): [⌊(i - 2) / scale⌋ - 2, ⌊(i + 2) / scale⌋ + 4), clipped.
    private static (string First, string Last) Readers(PointwiseWriter w, string i, string scale, string outSize)
    {
        string t = w.V(), f = w.V(), q = w.V(), qf = w.V(), low = w.V(), high = w.V(), first = w.V(), last = w.V(), scaleV = w.VFrom(scale), positive = w.Mask();
        CvLines(w, $"""
            v_add_u32_e32 {t}, -2, {i}
            v_cvt_f32_i32_e32 {f}, {t}
            """);
        w.Div(q, f, scaleV);
        CvLines(w, $"""
            v_floor_f32_e32 {qf}, {q}
            v_cvt_i32_f32_e32 {low}, {qf}
            v_add_u32_e32 {low}, -2, {low}
            v_max_i32_e32 {low}, 0, {low}
            v_min_i32_e32 {low}, {outSize}, {low}
            v_add_u32_e32 {t}, 2, {i}
            v_cvt_f32_u32_e32 {f}, {t}
            """);
        w.Div(q, f, scaleV);
        CvLines(w, $"""
            v_floor_f32_e32 {qf}, {q}
            v_cvt_i32_f32_e32 {high}, {qf}
            v_add_u32_e32 {high}, 4, {high}
            v_max_i32_e32 {high}, 0, {high}
            v_min_i32_e32 {high}, {outSize}, {high}
            v_cmp_lt_f32_e64 {positive}, 0, {scaleV}
            v_cndmask_b32_e64 {first}, 0, {low}, {positive}
            v_mov_b32_e32 {last}, {outSize}
            v_cndmask_b32_e64 {last}, {last}, {high}, {positive}
            """);
        return (first, last);
    }

    // The uniform masks of one axis: same size, twice the size, and the last input position.
    private static (string Same, string Twice, string Last) AxisMasks(PointwiseWriter w, string size, string outSize)
    {
        string twiceSize = CvHeld(w, TrainingText($"s_lshl_b32 {{0}}, {size}, 1")), last = CvHeld(w, TrainingText($"s_add_u32 {{0}}, {size}, -1"));
        return (CvUniformMask(w, size, outSize), CvUniformMask(w, twiceSize, outSize), last);
    }

    // Interpolation (nearest, bilinear) of planes [height, width] to [outheight, outwidth]: a work-item an output reads its
    // taps (y = h0 (w0 a + w1 b) + h1 (w0 c + w1 d), each product and sum rounded alone, as the CPU); the gradient gathers
    // the outputs that read each element (dx += (wr · wc) · dy, rows then columns in order).
    private static GcnKernelSource Interpolate(bool backward) =>
        TrainingElementwise(backward ? "interpolate_bwd_f32" : "interpolate_f32",
            new([backward ? "dy" : "x", backward ? "dx" : "y"], ["planes", "height", "width", "outheight", "outwidth", "mode", "align", "scaleh", "scalew"],
                backward ? [1] : [], [1])
            {
                Index = true,
                Single = true,
            }, (w, e) =>
            {
                string[] v = e.Values;
                var r = new Dictionary<string, string>();
                string bilinear = CvUniformMask(w, v[5]), nearest = w.Mask(), align = CvUniformMask(w, v[6]);
                CvLines(w, $"s_not_b64 {nearest}, {bilinear}");
                var (rowsSame, rowsTwice, rowsLast) = AxisMasks(w, v[1], v[3]);
                var (colsSame, colsTwice, colsLast) = AxisMasks(w, v[2], v[4]);
                if (!backward)
                {
                    var (planeRow, ow) = CvDivMod(w, e.Index, v[4], r);
                    var (plane, oh) = CvDivMod(w, planeRow, v[3], r);
                    var rows = AxisTaps(w, oh, v[7], rowsLast, nearest, align, rowsSame, rowsTwice);
                    var cols = AxisTaps(w, ow, v[8], colsLast, nearest, align, colsSame, colsTwice);
                    string row0 = w.V(), row1 = w.V(), a = w.V(), b = w.V(), c = w.V(), d = w.V(), at = w.V(), p = w.V(), q = w.V(), t = w.V(), mixed = w.V();
                    CvLines(w, $"""
                        v_mul_lo_u32 {row0}, {plane}, {v[1]}
                        v_add_u32_e32 {row1}, {row0}, {rows.Second}
                        v_add_u32_e32 {row0}, {row0}, {rows.First}
                        v_mul_lo_u32 {row0}, {row0}, {v[2]}
                        v_mul_lo_u32 {row1}, {row1}, {v[2]}
                        v_add_u32_e32 {at}, {row0}, {cols.First}
                        v_lshlrev_b32_e32 {at}, 2, {at}
                        global_load_dword {a}, {at}, {e.Pointers[0]}
                        v_add_u32_e32 {at}, {row0}, {cols.Second}
                        v_lshlrev_b32_e32 {at}, 2, {at}
                        global_load_dword {b}, {at}, {e.Pointers[0]}
                        v_add_u32_e32 {at}, {row1}, {cols.First}
                        v_lshlrev_b32_e32 {at}, 2, {at}
                        global_load_dword {c}, {at}, {e.Pointers[0]}
                        v_add_u32_e32 {at}, {row1}, {cols.Second}
                        v_lshlrev_b32_e32 {at}, 2, {at}
                        global_load_dword {d}, {at}, {e.Pointers[0]}
                        v_mul_f32_e32 {p}, {cols.W0}, {a}
                        v_mul_f32_e32 {t}, {cols.W1}, {b}
                        v_add_f32_e32 {p}, {p}, {t}
                        v_mul_f32_e32 {q}, {cols.W0}, {c}
                        v_mul_f32_e32 {t}, {cols.W1}, {d}
                        v_add_f32_e32 {q}, {q}, {t}
                        v_mul_f32_e32 {p}, {rows.W0}, {p}
                        v_mul_f32_e32 {q}, {rows.W1}, {q}
                        v_add_f32_e32 {mixed}, {p}, {q}
                        v_cndmask_b32_e64 {e.Out[1]}, {mixed}, {a}, {nearest}
                        """);
                    return;
                }

                var (inRow, iw) = CvDivMod(w, e.Index, v[2], r);
                var (planeIndex, ih) = CvDivMod(w, inRow, v[1], r);
                var (rowFirst, rowLast) = Readers(w, ih, v[7], v[3]);
                var (colFirst, colLast) = Readers(w, iw, v[8], v[4]);
                string sum = w.V(), planeOut = w.V(), o = w.V(), oc = w.V(), rowOut = w.V(), atOut = w.V(), g = w.V(), share = w.V(), added = w.V();
                string readsRow = w.Mask(), readsCol = w.Mask();
                CvLines(w, $"""
                    v_mov_b32_e32 {sum}, {e.In[1]}
                    v_mul_lo_u32 {planeOut}, {planeIndex}, {v[3]}
                    """);
                CvLaneLoop(w, e.Label + "_o", o, rowFirst, rowLast, () =>
                {
                    var rowTaps = AxisTaps(w, o, v[7], rowsLast, nearest, align, rowsSame, rowsTwice);
                    string wr = WeightOn(w, rowTaps, ih, bilinear, readsRow);
                    CvLines(w, $"""
                        v_add_u32_e32 {rowOut}, {planeOut}, {o}
                        v_mul_lo_u32 {rowOut}, {rowOut}, {v[4]}
                        """);
                    CvLaneLoop(w, e.Label + "_oc", oc, colFirst, colLast, () =>
                    {
                        var colTaps = AxisTaps(w, oc, v[8], colsLast, nearest, align, colsSame, colsTwice);
                        string wc = WeightOn(w, colTaps, iw, bilinear, readsCol);
                        CvLines(w, $"""
                            v_add_u32_e32 {atOut}, {rowOut}, {oc}
                            v_lshlrev_b32_e32 {atOut}, 2, {atOut}
                            global_load_dword {g}, {atOut}, {e.Pointers[0]}
                            v_mul_f32_e32 {share}, {wr}, {wc}
                            v_mul_f32_e32 {share}, {share}, {g}
                            v_add_f32_e32 {added}, {sum}, {share}
                            s_and_b64 vcc, {readsRow}, {readsCol}
                            v_cndmask_b32_e32 {sum}, {sum}, {added}, vcc
                            """);
                    });
                });
                CvLines(w, $"v_mov_b32_e32 {e.Out[1]}, {sum}");
            });

    // ---- Resampling with normalization ---------------------------------------------------------------------------------

    // Resampling and per-channel normalization (Backend.ResizeNormalize, CUDA's resize_normalize_f32): a work-item an output
    // takes the vertical taps in order, each the horizontal pass of its row (the taps in order) computed in place: the
    // two-pass result without the intermediate image; floats, or Pillow's 8-bit passes in integers (2^21 + Σ byte · weight,
    // >> 22, clipped). The coefficients hold, per output column then per output row, the first input position and the tap
    // count (int bits) and the taps' weights (floats, or int bits for bytes).
    private static GcnKernelSource ResizeNormalize() =>
        TrainingElementwise("resize_normalize_f32", new(["x", "coefficients", "values", "y"],
            ["planes", "channels", "height", "width", "outheight", "outwidth", "xtaps", "ytaps", "bytes"], [], [3])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            string[] p = e.Pointers;
            var r = new Dictionary<string, string>();
            var (planeRow, ox) = CvDivMod(w, e.Index, v[5], r);
            var (plane, oy) = CvDivMod(w, planeRow, v[4], r);
            string area = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {v[2]}, {v[3]}"));
            string xStride = CvHeld(w, TrainingText($"s_add_u32 {{0}}, {v[6]}, 2")), yStride = CvHeld(w, TrainingText($"s_add_u32 {{0}}, {v[7]}, 2"));
            string yBase = CvHeld(w, TrainingText($"s_mul_i32 {{0}}, {v[5]}, {xStride}"));
            w.Held.Append(TrainingText($"""
                  s_cmp_eq_u32 {v[6]}, 0
                  s_cselect_b32 {yBase}, 0, {yBase}

                """));
            string maxByte = w.S(255u);
            string planeAt = w.V(), xAt = w.V();
            CvLines(w, $"""
                v_mul_lo_u32 {planeAt}, {plane}, {area}
                v_mul_lo_u32 {xAt}, {ox}, {xStride}
                """);

            // Byte k of the packed words into `target` (an unsigned value 0 … 255).
            void ReadByte(string k, string target)
            {
                string word = w.V(), at = w.V(), shift = w.V();
                CvLines(w, $"""
                    v_and_b32_e32 {at}, -4, {k}
                    global_load_dword {word}, {at}, {p[0]}
                    v_and_b32_e32 {shift}, 3, {k}
                    v_lshlrev_b32_e32 {shift}, 3, {shift}
                    v_bfe_u32 {target}, {word}, {shift}, 8
                    """);
            }

            // The integer sum's clipped result (sum >> 22 within [0, 255]) as a float.
            void Clip(string sum, string target)
            {
                CvLines(w, $"""
                    v_ashrrev_i32_e32 {target}, 22, {sum}
                    v_max_i32_e32 {target}, 0, {target}
                    v_min_i32_e32 {target}, {maxByte}, {target}
                    v_cvt_f32_i32_e32 {target}, {target}
                    """);
            }

            // The horizontal pass of input row `row` at output column ox into `target` (a float; bytes as a float 0 … 255).
            int rows = 0;
            void Row(string row, string target)
            {
                string label = TrainingText($"{e.Label}_row{rows++}");
                string start = w.V(), first = w.V(), count = w.V(), at = w.V(), tap = w.V(), weight = w.V(), value = w.V(), product = w.V(), isum = w.V(), k = w.V();
                CvLines(w, $"""
                    v_mul_lo_u32 {start}, {row}, {v[3]}
                    v_add_u32_e32 {start}, {start}, {planeAt}
                    s_cmp_eq_u32 {v[6]}, 0
                    s_cbranch_scc1 {label}_none
                    v_lshlrev_b32_e32 {at}, 2, {xAt}
                    global_load_dword {first}, {at}, {p[1]}
                    global_load_dword {count}, {at}, {p[1]} offset:4
                    v_add_u32_e32 {start}, {start}, {first}
                    s_cmp_lg_u32 {v[8]}, 0
                    s_cbranch_scc1 {label}_bytes
                    v_mov_b32_e32 {target}, 0
                    """);
                CvLaneLoop(w, label + "_f", tap, "0", count, () => CvLines(w, $"""
                    v_add_u32_e32 {k}, {xAt}, {tap}
                    v_lshlrev_b32_e32 {k}, 2, {k}
                    global_load_dword {weight}, {k}, {p[1]} offset:8
                    v_add_u32_e32 {k}, {start}, {tap}
                    v_lshlrev_b32_e32 {k}, 2, {k}
                    global_load_dword {value}, {k}, {p[0]}
                    v_mul_f32_e32 {product}, {value}, {weight}
                    v_add_f32_e32 {target}, {target}, {product}
                    """));
                CvLines(w, $"""
                    s_branch {label}_done
                    {label}_bytes:
                    v_mov_b32_e32 {isum}, 0x200000
                    """);
                CvLaneLoop(w, label + "_b", tap, "0", count, () =>
                {
                    CvLines(w, $"""
                        v_add_u32_e32 {k}, {xAt}, {tap}
                        v_lshlrev_b32_e32 {k}, 2, {k}
                        global_load_dword {weight}, {k}, {p[1]} offset:8
                        v_add_u32_e32 {k}, {start}, {tap}
                        """);
                    ReadByte(k, value);
                    CvLines(w, $"""
                        v_mul_lo_u32 {product}, {value}, {weight}
                        v_add_u32_e32 {isum}, {isum}, {product}
                        """);
                });
                Clip(isum, target);
                CvLines(w, $"""
                    s_branch {label}_done
                    {label}_none:
                    v_add_u32_e32 {k}, {start}, {ox}
                    s_cmp_lg_u32 {v[8]}, 0
                    s_cbranch_scc1 {label}_none_bytes
                    v_lshlrev_b32_e32 {at}, 2, {k}
                    global_load_dword {target}, {at}, {p[0]}
                    s_branch {label}_done
                    {label}_none_bytes:
                    """);
                ReadByte(k, value);
                CvLines(w, $"""
                    v_cvt_f32_u32_e32 {target}, {value}
                    {label}_done:
                    """);
            }

            string result = w.V(), yAt = w.V(), yFirst = w.V(), yCount = w.V(), yTap = w.V(), rowIndex = w.V(), rowValue = w.V(), yWeight = w.V(), yProduct = w.V(),
                yInt = w.V(), whole = w.V(), yk = w.V();
            string down = e.Label + "_down";
            var (_, channel) = CvDivMod(w, plane, v[1], r);
            CvLines(w, $"""
                s_cmp_eq_u32 {v[7]}, 0
                s_cbranch_scc1 {down}_none
                v_mul_lo_u32 {yAt}, {oy}, {yStride}
                v_add_u32_e32 {yAt}, {yBase}, {yAt}
                v_lshlrev_b32_e32 {yk}, 2, {yAt}
                global_load_dword {yFirst}, {yk}, {p[1]}
                global_load_dword {yCount}, {yk}, {p[1]} offset:4
                v_mov_b32_e32 {result}, 0
                v_mov_b32_e32 {yInt}, 0x200000
                """);
            CvLaneLoop(w, down + "_tap", yTap, "0", yCount, () =>
            {
                CvLines(w, $"v_add_u32_e32 {rowIndex}, {yFirst}, {yTap}");
                Row(rowIndex, rowValue);
                CvLines(w, $"""
                    v_add_u32_e32 {yk}, {yAt}, {yTap}
                    v_lshlrev_b32_e32 {yk}, 2, {yk}
                    global_load_dword {yWeight}, {yk}, {p[1]} offset:8
                    v_mul_f32_e32 {yProduct}, {rowValue}, {yWeight}
                    v_add_f32_e32 {result}, {result}, {yProduct}
                    v_cvt_i32_f32_e32 {whole}, {rowValue}
                    v_mul_lo_u32 {whole}, {whole}, {yWeight}
                    v_add_u32_e32 {yInt}, {yInt}, {whole}
                    """);
            });
            string clipped = w.V();
            Clip(yInt, clipped);
            string bytes = CvUniformMask(w, v[8]);
            CvLines(w, $"""
                v_cndmask_b32_e64 {result}, {result}, {clipped}, {bytes}
                s_branch {down}_map
                {down}_none:
                """);
            Row(oy, result);
            string mapAt = w.V(), scale = w.V(), shift = w.V(), mapped = w.V(), code = w.V(), normalized = w.V();
            CvLines(w, $"""
                {down}_map:
                v_lshlrev_b32_e32 {mapAt}, 3, {channel}
                global_load_dword {scale}, {mapAt}, {p[2]}
                global_load_dword {shift}, {mapAt}, {p[2]} offset:4
                v_mul_f32_e32 {normalized}, {result}, {scale}
                v_add_f32_e32 {normalized}, {normalized}, {shift}
                s_cmp_lg_u32 {v[8]}, 0
                s_cbranch_scc0 {down}_float
                v_cvt_u32_f32_e32 {code}, {result}
                v_lshl_add_u32 {code}, {channel}, 8, {code}
                v_lshlrev_b32_e32 {code}, 2, {code}
                global_load_dword {mapped}, {code}, {p[2]}
                v_mov_b32_e32 {normalized}, {mapped}
                {down}_float:
                v_mov_b32_e32 {e.Out[3]}, {normalized}
                """);
        });

    // ---- The depthwise convolution -------------------------------------------------------------------------------------

    // The activation of code `code` (ConvActivation: 0 none, 1 ReLU, 2 sigmoid, 3 tanh, 4 GELU, 5 SiLU) on `x` into
    // `target`, as HIP's element-wise kernels compute each (the CPU's).
    private static void ConvActivationOf(PointwiseWriter w, string label, string code, string x, string target)
    {
        CvLines(w, $"""
            v_mov_b32_e32 {target}, {x}
            s_cmp_eq_u32 {code}, 1
            s_cbranch_scc0 {label}_not_relu
            v_cmp_ge_f32_e32 vcc, 0, {x}
            v_cndmask_b32_e64 {target}, {x}, 0, vcc
            s_branch {label}_done
            {label}_not_relu:
            s_cmp_eq_u32 {code}, 2
            s_cbranch_scc0 {label}_not_sigmoid
            """);
        w.Sigmoid(target, x);
        CvLines(w, $"""
            s_branch {label}_done
            {label}_not_sigmoid:
            s_cmp_eq_u32 {code}, 3
            s_cbranch_scc0 {label}_not_tanh
            """);
        w.Tanh(target, x);
        CvLines(w, $"""
            s_branch {label}_done
            {label}_not_tanh:
            s_cmp_eq_u32 {code}, 4
            s_cbranch_scc0 {label}_not_gelu
            """);
        w.GeluTanh(target, x);
        CvLines(w, $"""
            s_branch {label}_done
            {label}_not_gelu:
            s_cmp_eq_u32 {code}, 5
            s_cbranch_scc0 {label}_done
            """);
        string s = w.V();
        w.Sigmoid(s, x);
        CvLines(w, $"""
            v_mul_f32_e32 {target}, {x}, {s}
            {label}_done:
            """);
    }

    // y[n, f, oh, ow] = act(bias[f] + Σ w[f, kh, kw] · x[n, c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW]) with c = f / (F / C):
    // a work-item an output, its window rows then columns in order (fused multiply-adds, as CUDA's).
    private static GcnKernelSource DepthwiseConvolution() =>
        TrainingElementwise("conv_dw_f32", new(["x", "w", "bias", "y"], ConvolutionScalars, [], [3])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (planeRow, ow) = CvDivMod(w, e.Index, v[11], r);
            var (planeIndex, oh) = CvDivMod(w, planeRow, v[10], r);
            var (image, f) = CvDivMod(w, planeIndex, v[14], r);
            string multiplier = CvHeldQuotient(w, v[14], v[1]);
            var (c, _) = CvDivMod(w, f, multiplier, r);
            string planeAt = w.V(), weightAt = w.V(), r0 = w.V(), c0 = w.V(), sum = w.V(), ih = w.V(), iw = w.V(), rowAt = w.V(), wRow = w.V(), at = w.V(),
                value = w.V(), weight = w.V(), bias = w.V(), biased = w.V();
            CvLines(w, $"""
                v_mul_lo_u32 {planeAt}, {image}, {v[1]}
                v_add_u32_e32 {planeAt}, {planeAt}, {c}
                v_mul_lo_u32 {planeAt}, {planeAt}, {v[2]}
                v_mul_lo_u32 {weightAt}, {f}, {v[4]}
                v_mul_lo_u32 {r0}, {oh}, {v[6]}
                v_subrev_u32_e32 {r0}, {v[8]}, {r0}
                v_mul_lo_u32 {c0}, {ow}, {v[7]}
                v_subrev_u32_e32 {c0}, {v[9]}, {c0}
                v_mov_b32_e32 {sum}, 0
                """);
            string kh = CvScalar(w), kw = CvScalar(w), khDh = CvScalar(w), kwDw = CvScalar(w);
            CvScalarLoop(w, e.Label + "_kh", kh, v[4], () =>
            {
                CvLines(w, $"""
                    s_mul_i32 {khDh}, {kh}, {v[12]}
                    v_add_u32_e32 {ih}, {khDh}, {r0}
                    v_cmp_gt_u32_e32 vcc, {v[2]}, {ih}
                    """);
                CvMasked(w, e.Label + "_row", "vcc", () =>
                {
                    CvLines(w, $"""
                        v_add_u32_e32 {rowAt}, {planeAt}, {ih}
                        v_mul_lo_u32 {rowAt}, {rowAt}, {v[3]}
                        v_add_u32_e32 {wRow}, {kh}, {weightAt}
                        v_mul_lo_u32 {wRow}, {wRow}, {v[5]}
                        """);
                    CvScalarLoop(w, e.Label + "_kw", kw, v[5], () =>
                    {
                        CvLines(w, $"""
                            s_mul_i32 {kwDw}, {kw}, {v[13]}
                            v_add_u32_e32 {iw}, {kwDw}, {c0}
                            v_cmp_gt_u32_e32 vcc, {v[3]}, {iw}
                            """);
                        CvMasked(w, e.Label + "_col", "vcc", () => CvLines(w, $"""
                            v_add_u32_e32 {at}, {rowAt}, {iw}
                            v_lshlrev_b32_e32 {at}, 2, {at}
                            global_load_dword {value}, {at}, {e.Pointers[0]}
                            v_add_u32_e32 {at}, {kw}, {wRow}
                            v_lshlrev_b32_e32 {at}, 2, {at}
                            global_load_dword {weight}, {at}, {e.Pointers[1]}
                            v_fma_f32 {sum}, {weight}, {value}, {sum}
                            """));
                    });
                });
            });
            string hasBias = TrainingFlagMask(w, v[16], 0), code = CvScalar(w), at2 = CvBytes(w, f);
            CvLines(w, $"""
                global_load_dword {bias}, {at2}, {e.Pointers[2]}
                v_add_f32_e32 {biased}, {sum}, {bias}
                v_cndmask_b32_e64 {sum}, {sum}, {biased}, {hasBias}
                s_lshr_b32 {code}, {v[16]}, 1
                """);
            ConvActivationOf(w, e.Label + "_act", code, sum, e.Out[3]);
        });

    // dx[n, c, ih, iw] += Σ w[f, kh, kw] · dy[n, f, oh, ow] over the channel's filters f in order and the windows covering
    // the element in window order (ih + PH - kh·DH = oh·SH, iw + PW - kw·DW = ow·SW).
    private static GcnKernelSource DepthwiseInput() =>
        TrainingElementwise("conv_dw_bwd_input_f32", new(["dy", "w", "dx"], ConvolutionScalars, [2], [2])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string[] v = e.Values;
            var r = new Dictionary<string, string>();
            var (planeRow, iw) = CvDivMod(w, e.Index, v[3], r);
            var (planeIndex, ih) = CvDivMod(w, planeRow, v[2], r);
            var (image, c) = CvDivMod(w, planeIndex, v[1], r);
            string multiplier = CvHeldQuotient(w, v[14], v[1]);
            string sum = w.V(), ihp = w.V(), iwp = w.V(), filter = w.V(), dyRow = w.V(), wFilter = w.V(), t = w.V(), u = w.V(), dyAt = w.V(), wRow = w.V(), at = w.V(),
                g = w.V(), weight = w.V(), ok = w.Mask(), ok2 = w.Mask();
            CvLines(w, $"""
                v_mov_b32_e32 {sum}, {e.In[2]}
                v_add_u32_e32 {ihp}, {v[8]}, {ih}
                v_add_u32_e32 {iwp}, {v[9]}, {iw}
                """);
            string m = CvScalar(w), kh = CvScalar(w), kw = CvScalar(w), khDh = CvScalar(w), kwDw = CvScalar(w);
            CvScalarLoop(w, e.Label + "_m", m, multiplier, () =>
            {
                CvLines(w, $"""
                    v_mul_lo_u32 {filter}, {c}, {multiplier}
                    v_add_u32_e32 {filter}, {m}, {filter}
                    v_mul_lo_u32 {dyRow}, {image}, {v[14]}
                    v_add_u32_e32 {dyRow}, {dyRow}, {filter}
                    v_mul_lo_u32 {dyRow}, {dyRow}, {v[10]}
                    v_mul_lo_u32 {wFilter}, {filter}, {v[4]}
                    """);
                CvScalarLoop(w, e.Label + "_kh", kh, v[4], () =>
                {
                    CvLines(w, $"""
                        s_mul_i32 {khDh}, {kh}, {v[12]}
                        v_subrev_u32_e32 {t}, {khDh}, {ihp}
                        """);
                    var (oh, ohRest) = CvDivMod(w, t, v[6], r);
                    CvLines(w, $"""
                        v_cmp_lt_i32_e32 vcc, -1, {t}
                        v_cmp_eq_u32_e64 {ok}, 0, {ohRest}
                        s_and_b64 vcc, vcc, {ok}
                        v_cmp_gt_u32_e64 {ok}, {v[10]}, {oh}
                        s_and_b64 vcc, vcc, {ok}
                        """);
                    CvMasked(w, e.Label + "_row", "vcc", () =>
                    {
                        CvLines(w, $"""
                            v_add_u32_e32 {dyAt}, {dyRow}, {oh}
                            v_mul_lo_u32 {dyAt}, {dyAt}, {v[11]}
                            v_add_u32_e32 {wRow}, {kh}, {wFilter}
                            v_mul_lo_u32 {wRow}, {wRow}, {v[5]}
                            """);
                        CvScalarLoop(w, e.Label + "_kw", kw, v[5], () =>
                        {
                            CvLines(w, $"""
                                s_mul_i32 {kwDw}, {kw}, {v[13]}
                                v_subrev_u32_e32 {u}, {kwDw}, {iwp}
                                """);
                            var (ow, owRest) = CvDivMod(w, u, v[7], r);
                            CvLines(w, $"""
                                v_cmp_lt_i32_e32 vcc, -1, {u}
                                v_cmp_eq_u32_e64 {ok2}, 0, {owRest}
                                s_and_b64 vcc, vcc, {ok2}
                                v_cmp_gt_u32_e64 {ok2}, {v[11]}, {ow}
                                s_and_b64 vcc, vcc, {ok2}
                                """);
                            CvMasked(w, e.Label + "_col", "vcc", () => CvLines(w, $"""
                                v_add_u32_e32 {at}, {dyAt}, {ow}
                                v_lshlrev_b32_e32 {at}, 2, {at}
                                global_load_dword {g}, {at}, {e.Pointers[0]}
                                v_add_u32_e32 {at}, {kw}, {wRow}
                                v_lshlrev_b32_e32 {at}, 2, {at}
                                global_load_dword {weight}, {at}, {e.Pointers[1]}
                                v_fma_f32 {sum}, {weight}, {g}, {sum}
                                """));
                        });
                    });
                });
            });
            CvLines(w, $"v_mov_b32_e32 {e.Out[2]}, {sum}");
        });

    // dw[f, kh, kw] += Σ over the positions (n, oh, ow) of dy · x: a block a weight (grid-stride over the weights), each
    // thread positions tid, tid + 256, … in order, then the block's fixed LDS tree; the first thread adds the sum to dw.
    private static GcnKernelSource DepthwiseWeight()
    {
        var w = new RowWriter("conv_dw_bwd_weight_f32", arrays: 1);
        var a = w.Arguments(["x", "dy", "dw"], ConvolutionScalars);
        string S(string name) => a.Scalar(name);
        string weights = w.S(), total = w.S();
        w.L($"""
            s_mul_i32 {weights}, {S("F")}, {S("KH")}
            s_mul_i32 {weights}, {weights}, {S("KW")}
            s_mul_i32 {total}, {S("N")}, {S("OH")}
            s_mul_i32 {total}, {total}, {S("OW")}
            """);
        string byKw = AttentionReciprocal(w, S("KW")), byKh = AttentionReciprocal(w, S("KH"));
        string byC = AttentionReciprocal(w, S("C"));
        var (multiplier, _) = AttentionDivMod(w, S("F"), S("C"), byC);
        string byMultiplier = AttentionReciprocal(w, multiplier), byOw = AttentionReciprocal(w, S("OW")), byOh = AttentionReciprocal(w, S("OH"));
        w.BeginRows(weights);
        var (fk, kw) = AttentionDivMod(w, w.Row, S("KW"), byKw);
        var (f, kh) = AttentionDivMod(w, fk, S("KH"), byKh);
        var (c, _) = AttentionDivMod(w, f, multiplier, byMultiplier);
        string offH = w.S(), offW = w.S();
        w.L($"""
            s_mul_i32 {offH}, {kh}, {S("DH")}
            s_sub_u32 {offH}, {offH}, {S("PH")}
            s_mul_i32 {offW}, {kw}, {S("DW")}
            s_sub_u32 {offW}, {offW}, {S("PW")}
            """);
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        BlockElements(w, ".Lpositions", "0", total, j =>
        {
            var (imageRow, ow) = RowDivMod(w, j, S("OW"), byOw);
            var (image, oh) = RowDivMod(w, imageRow, S("OH"), byOh);
            string ih = w.V(), iw = w.V(), inside = w.SPair(), saved = w.SPair(), dyAt = w.V(), xAt = w.V(), g = w.V(), value = w.V();
            w.L($"""
                v_mul_lo_u32 {ih}, {oh}, {S("SH")}
                v_add_u32_e32 {ih}, {offH}, {ih}
                v_mul_lo_u32 {iw}, {ow}, {S("SW")}
                v_add_u32_e32 {iw}, {offW}, {iw}
                v_cmp_gt_u32_e32 vcc, {S("H")}, {ih}
                v_cmp_gt_u32_e64 {inside}, {S("W")}, {iw}
                s_and_b64 vcc, vcc, {inside}
                s_and_saveexec_b64 {saved}, vcc
                v_mul_lo_u32 {dyAt}, {image}, {S("F")}
                v_add_u32_e32 {dyAt}, {f}, {dyAt}
                v_mul_lo_u32 {dyAt}, {dyAt}, {S("OH")}
                v_add_u32_e32 {dyAt}, {dyAt}, {oh}
                v_mul_lo_u32 {dyAt}, {dyAt}, {S("OW")}
                v_add_u32_e32 {dyAt}, {dyAt}, {ow}
                v_lshlrev_b32_e32 {dyAt}, 2, {dyAt}
                v_mul_lo_u32 {xAt}, {image}, {S("C")}
                v_add_u32_e32 {xAt}, {c}, {xAt}
                v_mul_lo_u32 {xAt}, {xAt}, {S("H")}
                v_add_u32_e32 {xAt}, {xAt}, {ih}
                v_mul_lo_u32 {xAt}, {xAt}, {S("W")}
                v_add_u32_e32 {xAt}, {xAt}, {iw}
                v_lshlrev_b32_e32 {xAt}, 2, {xAt}
                global_load_dword {g}, {dyAt}, {a.Pointer("dy")}
                global_load_dword {value}, {xAt}, {a.Pointer("x")}
                v_fma_f32 {sum}, {g}, {value}, {sum}
                s_mov_b64 exec, {saved}
                """);
        });
        w.Reduce(RowWriter.Add, sum);
        string at = w.ElementAddress(a.Pointer("dw"), w.Row), first = w.SPair(), old = w.V();
        w.L($"""
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {first}, vcc
            global_load_dword {old}, {w.Zero}, {at}
            v_add_f32_e32 {old}, {old}, {sum}
            global_store_dword {w.Zero}, {old}, {at}
            s_or_b64 exec, exec, {first}
            """);
        w.EndRows();
        return w.Source();
    }
}
