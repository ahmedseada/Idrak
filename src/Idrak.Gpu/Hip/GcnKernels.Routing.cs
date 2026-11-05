// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's mixture-of-experts kernels in GCN (plan 37, R0b; Backend.Routing.cs), the same operations, order and arithmetic
// as Vulkan's (VulkanKernels.Routing.cs) and the CPU's (CpuBackend.Routing.cs), whose results they equal bit for bit:
// - route_choose_f32: one work-item per token: its k most probable experts (ties to the lowest index) walked in as the
//   CPU walks a row, the list kept in the token's own slots of `choices` (int32), sorted into increasing expert order;
//   the weights' sum added in that order and each weight divided by it (LLVM's correctly rounded division) when asked.
// - route_counts_f32 (one work-item per expert), route_offsets_f32 (one work-item: the counts added in expert order),
//   route_place_f32 (one work-item per expert: its rows in slot order): offsets, tokenOfRow and slotRow (int32).
// - expert_gated_{silu,gelu,relu,tanh}_f32 and expert_matmul_f32: one launch per expert, whose grid-stride loop runs
//   over its rows × n outputs (the count read from the offsets on the device); each output one fused multiply-add chain
//   over k from zero, in order (rule 86, the CPU's product), the bias added after it (a null bias pointer: none), the
//   gated activation as the CPU's (PointwiseWriter.Gated).
// Every kernel walks its work grid-stride in blocks of ShapeBlock, as the shape kernels do; byte offsets are 32-bit (the
// backend checks every buffer stays below 2^30 floats, else the host fallback).
//
// | Kernel             | Arguments                                                        | Unit             | Flops      | Bytes        |
// |--------------------|------------------------------------------------------------------|------------------|------------|--------------|
// | route_choose_f32   | probabilities, weights, choices; experts, k, normalize, n        | a token's expert | 2          | 16           |
// | route_counts_f32   | choices, counts; slots, n                                        | an expert's slot | 0          | 4            |
// | route_offsets_f32  | counts, offsets; slots, experts, n                               | an expert        | 0          | 8            |
// | route_place_f32    | choices, offsets, tokenOfRow, slotRow; slots, k, n               | an expert's slot | 0          | 4            |
// | expert_gated_*_f32 | x, offsets, gate, gatebias, up, upbias, hidden; expert, k, n     | an output × k    | 4          | 16           |
// | expert_matmul_f32  | x, offsets, weight, bias, y; expert, k, n                        | an output × k    | 2          | 8            |
internal static partial class GcnKernels
{
    /// <summary>The expert gated pairs' kernels by the activation's kind (0 SiLU, 1 GELU, 2 ReLU, 3 tanh · sigmoid), as <see cref="GatedKinds"/>.</summary>
    internal static readonly string[] ExpertGatedKinds = ["expert_gated_silu_f32", "expert_gated_gelu_f32", "expert_gated_relu_f32", "expert_gated_tanh_f32"];

    /// <summary>route_choose_f32's arguments.</summary>
    internal static IReadOnlyList<GcnArgument> RouteChooseArguments { get; } =
    [
        GcnArgument.Buffer(0, "probabilities"), GcnArgument.Buffer(8, "weights"), GcnArgument.Buffer(16, "choices"),
        GcnArgument.Value(24, 4, "experts"), GcnArgument.Value(28, 4, "k"), GcnArgument.Value(32, 4, "normalize"), GcnArgument.Value(36, 4, "n"),
    ];

    /// <summary>route_counts_f32's arguments.</summary>
    internal static IReadOnlyList<GcnArgument> RouteCountsArguments { get; } =
        [GcnArgument.Buffer(0, "choices"), GcnArgument.Buffer(8, "counts"), GcnArgument.Value(16, 4, "slots"), GcnArgument.Value(20, 4, "n")];

    /// <summary>route_offsets_f32's arguments.</summary>
    internal static IReadOnlyList<GcnArgument> RouteOffsetsArguments { get; } =
        [GcnArgument.Buffer(0, "counts"), GcnArgument.Buffer(8, "offsets"), GcnArgument.Value(16, 4, "slots"), GcnArgument.Value(20, 4, "experts"), GcnArgument.Value(24, 4, "n")];

    /// <summary>route_place_f32's arguments.</summary>
    internal static IReadOnlyList<GcnArgument> RoutePlaceArguments { get; } =
    [
        GcnArgument.Buffer(0, "choices"), GcnArgument.Buffer(8, "offsets"), GcnArgument.Buffer(16, "tokenOfRow"), GcnArgument.Buffer(24, "slotRow"),
        GcnArgument.Value(32, 4, "slots"), GcnArgument.Value(36, 4, "k"), GcnArgument.Value(40, 4, "n"),
    ];

    /// <summary>An expert gated pair's arguments (gatebias and upbias: 0 for none).</summary>
    internal static IReadOnlyList<GcnArgument> ExpertGatedArguments { get; } =
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "offsets"), GcnArgument.Buffer(16, "gate"), GcnArgument.Buffer(24, "gatebias"), GcnArgument.Buffer(32, "up"),
        GcnArgument.Buffer(40, "upbias"), GcnArgument.Buffer(48, "hidden"), GcnArgument.Value(56, 4, "expert"), GcnArgument.Value(60, 4, "k"), GcnArgument.Value(64, 4, "n"),
    ];

    /// <summary>expert_matmul_f32's arguments (bias: 0 for none).</summary>
    internal static IReadOnlyList<GcnArgument> ExpertMatMulArguments { get; } =
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "offsets"), GcnArgument.Buffer(16, "weight"), GcnArgument.Buffer(24, "bias"), GcnArgument.Buffer(32, "y"),
        GcnArgument.Value(40, 4, "expert"), GcnArgument.Value(44, 4, "k"), GcnArgument.Value(48, 4, "n"),
    ];

    /// <summary>The routing kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Routing => _routing ??=
        [RouteChoose(), RouteCounts(), RouteOffsets(), RoutePlace(), .. Enumerable.Range(0, ExpertGatedKinds.Length).Select(ExpertGated), ExpertMatMul()];

    private static GcnKernelSource[]? _routing;

    // `target` = buffer[at] (an element index in a vector register), through a fresh byte offset.
    private static void LoadAt(ProductWriter w, string target, string at, string buffer)
    {
        string bytes = w.V();
        w.L($"v_lshlrev_b32_e32 {bytes}, 2, {at}");
        w.L($"global_load_dword {target}, {bytes}, {buffer}");
    }

    // buffer[at] = value (an element index in a vector register), through a fresh byte offset.
    private static void StoreAt(ProductWriter w, string value, string at, string buffer)
    {
        string bytes = w.V();
        w.L($"v_lshlrev_b32_e32 {bytes}, 2, {at}");
        w.L($"global_store_dword {bytes}, {value}, {buffer}");
    }

    // A scalar loop: counter from 0 while below `end` (a scalar), body between; the counter is the body's to read.
    private static void ScalarLoop(ProductWriter w, string counter, string end, Action body)
    {
        string head = w.NewLabel(), done = w.NewLabel();
        w.L($"s_mov_b32 {counter}, 0");
        w.Label(head);
        w.L($"s_cmp_ge_u32 {counter}, {end}");
        w.L($"s_cbranch_scc1 {done}");
        body();
        w.L($"s_add_u32 {counter}, {counter}, 1");
        w.L($"s_branch {head}");
        w.Label(done);
    }

    private static GcnKernelSource RouteChoose()
    {
        var w = new ProductWriter("route_choose_f32");
        w.Head(RouteChooseArguments, ShapeBlock, dispatch: true);
        string probabilities = w.Pointer("probabilities"), weights = w.Pointer("weights"), choices = w.Pointer("choices");
        string experts = w.Scalar("experts"), k = w.Scalar("k"), normalize = w.Scalar("normalize"), n = w.Scalar("n");
        string lastExpert = w.S(), lastPlace = w.S(), normalized = w.SPair();
        w.L($"s_sub_u32 {lastExpert}, {experts}, 1");
        w.L($"s_sub_u32 {lastPlace}, {k}, 1");
        w.L($"s_cmp_lg_u32 {normalize}, 0");
        w.L($"s_cselect_b64 {normalized}, -1, 0");

        // The division, LLVM's correctly rounded one (PointwiseWriter's registers apart from this writer's).
        var p = new PointwiseWriter(firstVector: 3000, firstScalar: 3000);
        var division = new StringBuilder();
        p.Into(division);
        string quotient = w.V(), dividend = w.V(), divisor = w.V();
        p.Div(quotient, dividend, divisor);
        w.Lines(p.Held);

        string index = w.GridIndex();
        w.GridLoop(n, index, () =>
        {
            string row = w.V(), first = w.V(), count = w.V(), e = w.S();
            w.L($"v_mul_lo_u32 {row}, {index}, {experts}                ; the token's probabilities");
            w.L($"v_mul_lo_u32 {first}, {index}, {k}                    ; its slots, its list while it is made");
            w.L($"v_mov_b32_e32 {count}, 0");

            // The expert at list place `place` (clamped into the list), clamped into the experts (read where it counts).
            string Chosen(string place)
            {
                string clamped = w.V(), at = w.V(), chosen = w.V();
                w.L($"v_max_i32_e32 {clamped}, 0, {place}");
                w.L($"v_min_i32_e32 {clamped}, {lastPlace}, {clamped}");
                w.L($"v_add_u32_e32 {at}, {clamped}, {first}");
                LoadAt(w, chosen, at, choices);
                w.L($"v_max_i32_e32 {chosen}, 0, {chosen}");
                w.L($"v_min_i32_e32 {chosen}, {lastExpert}, {chosen}");
                return chosen;
            }

            // probabilities[row + expert].
            string Probability(string expert)
            {
                string at = w.V(), value = w.V();
                w.L($"v_add_u32_e32 {at}, {expert}, {row}");
                LoadAt(w, value, at, probabilities);
                return value;
            }

            ScalarLoop(w, e, experts, () =>
            {
                // Insert e after every chosen expert at least as probable (earlier indices win ties): walk back from the
                // list's end while e is more probable than the one before.
                string at = w.V(), walking = w.V(), probability = w.V(), at0 = w.V(), t = w.S();
                w.L($"v_add_u32_e32 {at0}, {e}, {row}");
                LoadAt(w, probability, at0, probabilities);
                w.L($"v_mov_b32_e32 {at}, {count}");
                w.L($"v_mov_b32_e32 {walking}, 1");
                ScalarLoop(w, t, k, () =>
                {
                    string before = w.V(), dec = w.V(), moving = w.SPair(), positive = w.SPair(), still = w.SPair();
                    w.L($"v_add_u32_e32 {before}, -1, {at}");
                    string other = Probability(Chosen(before));
                    w.L($"v_cmp_gt_f32_e32 vcc, {probability}, {other}");
                    w.L($"v_cmp_lt_i32_e64 {positive}, 0, {at}");
                    w.L($"v_cmp_ne_u32_e64 {still}, 0, {walking}");
                    w.L($"s_and_b64 {moving}, {positive}, vcc");
                    w.L($"s_and_b64 {moving}, {moving}, {still}");
                    w.L($"v_add_u32_e32 {dec}, -1, {at}");
                    w.L($"v_cndmask_b32_e64 {at}, {at}, {dec}, {moving}");
                    w.L($"v_cndmask_b32_e64 {walking}, 0, 1, {moving}");
                });

                // Inserted where at < k: places min(count, k - 1) down to at + 1 move up one, e goes to place at.
                string inserting = w.SPair(), last = w.V(), j = w.S(), shiftHead = w.NewLabel(), shiftDone = w.NewLabel();
                w.L($"v_cmp_gt_u32_e64 {inserting}, {k}, {at}");
                w.L($"v_min_u32_e32 {last}, {lastPlace}, {count}");
                w.L($"s_mov_b32 {j}, {lastPlace}");
                w.Label(shiftHead);
                w.L($"s_cmp_eq_u32 {j}, 0");
                w.L($"s_cbranch_scc1 {shiftDone}");
                {
                    string to = w.V(), from = w.V(), current = w.V(), previous = w.V(), moved = w.V(), above = w.SPair(), within = w.SPair();
                    w.L($"v_add_u32_e32 {to}, {j}, {first}");
                    w.L($"v_add_u32_e32 {from}, -1, {to}");
                    LoadAt(w, current, to, choices);
                    LoadAt(w, previous, from, choices);
                    w.L($"v_cmp_gt_u32_e64 {above}, {j}, {at}");
                    w.L($"v_cmp_le_u32_e64 {within}, {j}, {last}");
                    w.L($"s_and_b64 {above}, {above}, {within}");
                    w.L($"s_and_b64 {above}, {above}, {inserting}");
                    w.L($"v_cndmask_b32_e64 {moved}, {current}, {previous}, {above}");
                    StoreAt(w, moved, to, choices);
                }

                w.L($"s_sub_u32 {j}, {j}, 1");
                w.L($"s_branch {shiftHead}");
                w.Label(shiftDone);
                string place = w.V(), placeAt = w.V(), old = w.V(), expert = w.V(), written = w.V(), grown = w.V();
                w.L($"v_min_u32_e32 {place}, {lastPlace}, {at}");
                w.L($"v_add_u32_e32 {placeAt}, {place}, {first}");
                LoadAt(w, old, placeAt, choices);
                w.L($"v_mov_b32_e32 {expert}, {e}");
                w.L($"v_cndmask_b32_e64 {written}, {old}, {expert}, {inserting}");
                StoreAt(w, written, placeAt, choices);
                w.L($"v_add_u32_e32 {grown}, 1, {count}");
                w.L($"v_min_u32_e32 {grown}, {k}, {grown}");
                w.L($"v_cndmask_b32_e64 {count}, {count}, {grown}, {inserting}");
            });

            // The k choices in increasing expert order (insertion sort; they are distinct).
            string a = w.S(), sortHead = w.NewLabel(), sortDone = w.NewLabel();
            w.L($"s_mov_b32 {a}, 1");
            w.Label(sortHead);
            w.L($"s_cmp_ge_u32 {a}, {k}");
            w.L($"s_cbranch_scc1 {sortDone}");
            {
                string valueAt = w.V(), value = w.V(), b = w.V(), moving = w.V(), t = w.S();
                w.L($"v_add_u32_e32 {valueAt}, {a}, {first}");
                LoadAt(w, value, valueAt, choices);
                w.L($"v_mov_b32_e32 {b}, {a}");
                w.L($"v_mov_b32_e32 {moving}, 1");
                ScalarLoop(w, t, a, () =>
                {
                    string before = w.V(), beforeAt = w.V(), at = w.V(), previous = w.V(), current = w.V(), moved = w.V(), dec = w.V();
                    string shift = w.SPair(), positive = w.SPair(), still = w.SPair();
                    w.L($"v_add_u32_e32 {before}, -1, {b}");
                    w.L($"v_max_i32_e32 {before}, 0, {before}");
                    w.L($"v_add_u32_e32 {beforeAt}, {before}, {first}");
                    w.L($"v_add_u32_e32 {at}, {b}, {first}");
                    LoadAt(w, previous, beforeAt, choices);
                    LoadAt(w, current, at, choices);
                    w.L($"v_cmp_gt_i32_e32 vcc, {previous}, {value}");
                    w.L($"v_cmp_lt_i32_e64 {positive}, 0, {b}");
                    w.L($"v_cmp_ne_u32_e64 {still}, 0, {moving}");
                    w.L($"s_and_b64 {shift}, {positive}, vcc");
                    w.L($"s_and_b64 {shift}, {shift}, {still}");
                    w.L($"v_cndmask_b32_e64 {moved}, {current}, {previous}, {shift}");
                    StoreAt(w, moved, at, choices);
                    w.L($"v_add_u32_e32 {dec}, -1, {b}");
                    w.L($"v_cndmask_b32_e64 {b}, {b}, {dec}, {shift}");
                    w.L($"v_cndmask_b32_e64 {moving}, 0, 1, {shift}");
                });

                string bAt = w.V();
                w.L($"v_add_u32_e32 {bAt}, {b}, {first}");
                StoreAt(w, value, bAt, choices);
            }

            w.L($"s_add_u32 {a}, {a}, 1");
            w.L($"s_branch {sortHead}");
            w.Label(sortDone);

            // The sum in increasing expert order, then each weight: the probability, divided by the sum when asked.
            string sum = w.V(), s = w.S();
            w.L($"v_mov_b32_e32 {sum}, 0");
            ScalarLoop(w, s, k, () =>
            {
                string slot = w.V();
                w.L($"v_mov_b32_e32 {slot}, {s}");
                w.L($"v_add_f32_e32 {sum}, {sum}, {Probability(Chosen(slot))}");
            });
            ScalarLoop(w, s, k, () =>
            {
                string slot = w.V(), weight = w.V(), weightAt = w.V();
                w.L($"v_mov_b32_e32 {slot}, {s}");
                string probability = Probability(Chosen(slot));
                w.L($"v_mov_b32_e32 {dividend}, {probability}");
                w.L($"v_mov_b32_e32 {divisor}, {sum}");
                w.Lines(division);
                w.L($"v_cndmask_b32_e64 {weight}, {probability}, {quotient}, {normalized}");
                w.L($"v_add_u32_e32 {weightAt}, {s}, {first}");
                StoreAt(w, weight, weightAt, weights);
            });
        });
        return w.Source();
    }

    // counts[e] = the slots whose choice is e: one work-item per expert, walking the slots.
    private static GcnKernelSource RouteCounts()
    {
        var w = new ProductWriter("route_counts_f32");
        w.Head(RouteCountsArguments, ShapeBlock, dispatch: true);
        string choices = w.Pointer("choices"), counts = w.Pointer("counts"), slots = w.Scalar("slots"), n = w.Scalar("n");
        string index = w.GridIndex();
        w.GridLoop(n, index, () =>
        {
            string count = w.V(), s = w.S();
            w.L($"v_mov_b32_e32 {count}, 0");
            ScalarLoop(w, s, slots, () =>
            {
                string slot = w.V(), choice = w.V(), hit = w.V();
                w.L($"v_mov_b32_e32 {slot}, {s}");
                LoadAt(w, choice, slot, choices);
                w.L($"v_cmp_eq_u32_e32 vcc, {choice}, {index}");
                w.L($"v_cndmask_b32_e64 {hit}, 0, 1, vcc");
                w.L($"v_add_u32_e32 {count}, {count}, {hit}");
            });
            StoreAt(w, count, index, counts);
        });
        return w.Source();
    }

    // offsets[e] = the counts before e, added in expert order; offsets[experts] = slots: one work-item.
    private static GcnKernelSource RouteOffsets()
    {
        var w = new ProductWriter("route_offsets_f32");
        w.Head(RouteOffsetsArguments, ShapeBlock, dispatch: true);
        string counts = w.Pointer("counts"), offsets = w.Pointer("offsets"), slots = w.Scalar("slots"), experts = w.Scalar("experts"), n = w.Scalar("n");
        string index = w.GridIndex();
        w.GridLoop(n, index, () =>
        {
            string start = w.V(), e = w.S();
            w.L($"v_mov_b32_e32 {start}, 0");
            ScalarLoop(w, e, experts, () =>
            {
                string at = w.V(), count = w.V();
                w.L($"v_mov_b32_e32 {at}, {e}");
                StoreAt(w, start, at, offsets);
                LoadAt(w, count, at, counts);
                w.L($"v_add_u32_e32 {start}, {start}, {count}");
            });
            string end = w.V(), total = w.V();
            w.L($"v_mov_b32_e32 {end}, {experts}");
            w.L($"v_mov_b32_e32 {total}, {slots}");
            StoreAt(w, total, end, offsets);
        });
        return w.Source();
    }

    // Expert e's rows in slot order: tokenOfRow[row] = slot / k and slotRow[slot] = row, row from offsets[e] on: one
    // work-item per expert, walking the slots.
    private static GcnKernelSource RoutePlace()
    {
        var w = new ProductWriter("route_place_f32");
        w.Head(RoutePlaceArguments, ShapeBlock, dispatch: true);
        string choices = w.Pointer("choices"), offsets = w.Pointer("offsets"), tokenOfRow = w.Pointer("tokenOfRow"), slotRow = w.Pointer("slotRow");
        string slots = w.Scalar("slots"), k = w.Scalar("k"), n = w.Scalar("n");
        string index = w.GridIndex();
        w.GridLoop(n, index, () =>
        {
            string cursor = w.V(), s = w.S(), token = w.S(), within = w.S();
            LoadAt(w, cursor, index, offsets);
            w.L($"s_mov_b32 {token}, 0");
            w.L($"s_mov_b32 {within}, 0");
            ScalarLoop(w, s, slots, () =>
            {
                string slot = w.V(), choice = w.V(), tokenValue = w.V(), saved = w.SPair(), skip = w.NewLabel();
                w.L($"v_mov_b32_e32 {slot}, {s}");
                LoadAt(w, choice, slot, choices);
                w.L($"v_cmp_eq_u32_e32 vcc, {choice}, {index}");
                w.L($"s_and_saveexec_b64 {saved}, vcc");
                w.L($"s_cbranch_execz {skip}");
                w.L($"v_mov_b32_e32 {tokenValue}, {token}");
                StoreAt(w, tokenValue, cursor, tokenOfRow);
                StoreAt(w, cursor, slot, slotRow);
                w.L($"v_add_u32_e32 {cursor}, 1, {cursor}");
                w.Label(skip);
                w.L($"s_or_b64 exec, exec, {saved}");
                // The next slot's token: within counts up to k, then the token moves on.
                w.L($"s_add_u32 {within}, {within}, 1");
                w.L($"s_cmp_eq_u32 {within}, {k}");
                w.L($"s_cselect_b32 {within}, 0, {within}");
                w.L($"s_addc_u32 {token}, {token}, 0");
            });
        });
        return w.Source();
    }

    // The bounds of expert `expert`'s rows (offsets[expert], offsets[expert + 1], read on the scalar unit) and its outputs'
    // count (rows · n): the grid-stride loop's bound.
    private static (string First, string Count) ExpertRows(ProductWriter w, string expert, string n)
    {
        string at = w.Offset64(w.Pointer("offsets"), expert), first = w.S(), end = w.S(), rows = w.S(), count = w.S();
        w.L($"s_load_dword {first}, {at}, 0x0");
        w.L($"s_load_dword {end}, {at}, 0x4");
        w.L($"s_sub_u32 {rows}, {end}, {first}");
        w.L($"s_mul_i32 {count}, {rows}, {n}");
        return (first, count);
    }

    // `sum` = Σ over p < k of x[row · k + p] · w[p · n + col] (one fused chain from zero, p in order), then + bias[col]
    // unless the bias pointer is 0; another weight's chain alongside when `other` is given (the gated pair's up side).
    private static void ExpertChains(ProductWriter w, string x, string row, string col, string k, string n, (string Weight, string Bias, string Sum)[] chains)
    {
        string rowStart = w.V(), p = w.S(), pn = w.S();
        w.L($"v_mul_lo_u32 {rowStart}, {row}, {k}");
        foreach (var chain in chains)
        {
            w.L($"v_mov_b32_e32 {chain.Sum}, 0");
        }

        w.L($"s_mov_b32 {pn}, 0");
        ScalarLoop(w, p, k, () =>
        {
            string xAt = w.V(), xValue = w.V(), wAt = w.V();
            w.L($"v_add_u32_e32 {xAt}, {p}, {rowStart}");
            LoadAt(w, xValue, xAt, x);
            w.L($"v_add_u32_e32 {wAt}, {pn}, {col}");
            foreach (var chain in chains)
            {
                string weight = w.V();
                LoadAt(w, weight, wAt, chain.Weight);
                w.L($"v_fma_f32 {chain.Sum}, {xValue}, {weight}, {chain.Sum}");
            }

            w.L($"s_add_u32 {pn}, {pn}, {n}");
        });

        foreach (var chain in chains)
        {
            string unbiased = w.NewLabel(), bias = w.V();
            w.L($"s_cmp_eq_u64 {chain.Bias}, 0");
            w.L($"s_cbranch_scc1 {unbiased}");
            LoadAt(w, bias, col, chain.Bias);
            w.L($"v_add_f32_e32 {chain.Sum}, {chain.Sum}, {bias}");
            w.Label(unbiased);
        }
    }

    // One expert's gate and up projections over its rows and their gated activation into those rows of hidden.
    private static GcnKernelSource ExpertGated(int kind)
    {
        var w = new ProductWriter(ExpertGatedKinds[kind]);
        w.Head(ExpertGatedArguments, ShapeBlock, dispatch: true);
        string x = w.Pointer("x"), hidden = w.Pointer("hidden"), expert = w.Scalar("expert"), k = w.Scalar("k"), n = w.Scalar("n");
        var p = new PointwiseWriter(firstVector: 3000, firstScalar: 3000);
        var activation = new StringBuilder();
        p.Into(activation);
        string g = w.V(), u = w.V(), h = w.V();
        p.Gated(h, g, u, kind);
        w.Lines(p.Held);
        var (first, count) = ExpertRows(w, expert, n);
        string reciprocal = w.Reciprocal(n);
        string index = w.GridIndex();
        w.GridLoop(count, index, () =>
        {
            var (offset, col) = w.DivMod(index, n, reciprocal);
            string row = w.V(), outAt = w.V();
            w.L($"v_add_u32_e32 {row}, {first}, {offset}");
            ExpertChains(w, x, row, col, k, n, [(w.Pointer("gate"), w.Pointer("gatebias"), g), (w.Pointer("up"), w.Pointer("upbias"), u)]);
            w.Lines(activation);
            w.L($"v_mul_lo_u32 {outAt}, {row}, {n}");
            w.L($"v_add_u32_e32 {outAt}, {outAt}, {col}");
            StoreAt(w, h, outAt, hidden);
        });
        return w.Source();
    }

    // One expert's product over its rows into those rows of y.
    private static GcnKernelSource ExpertMatMul()
    {
        var w = new ProductWriter("expert_matmul_f32");
        w.Head(ExpertMatMulArguments, ShapeBlock, dispatch: true);
        string x = w.Pointer("x"), y = w.Pointer("y"), expert = w.Scalar("expert"), k = w.Scalar("k"), n = w.Scalar("n");
        var (first, count) = ExpertRows(w, expert, n);
        string reciprocal = w.Reciprocal(n);
        string index = w.GridIndex();
        w.GridLoop(count, index, () =>
        {
            var (offset, col) = w.DivMod(index, n, reciprocal);
            string row = w.V(), sum = w.V(), outAt = w.V();
            w.L($"v_add_u32_e32 {row}, {first}, {offset}");
            ExpertChains(w, x, row, col, k, n, [(w.Pointer("weight"), w.Pointer("bias"), sum)]);
            w.L($"v_mul_lo_u32 {outAt}, {row}, {n}");
            w.L($"v_add_u32_e32 {outAt}, {outAt}, {col}");
            StoreAt(w, sum, outAt, y);
        });
        return w.Source();
    }
}
