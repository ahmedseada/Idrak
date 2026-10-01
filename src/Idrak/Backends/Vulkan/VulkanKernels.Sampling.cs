// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Kernels for decoding steps without host round trips: the sampler (see Backend.SampleRows), repetition penalties, the
// token history, and the queries' and keys' RMS normalization with rotary positions. They mirror the CPU's passes, so the
// same seed draws the same tokens (up to rounding in sums of weights, which only matters at an exact tie of the draw).
internal static partial class VulkanKernels
{
    /// <summary>Scores of one row a <c>topk_slots</c> workgroup takes (its slice, held in workgroup memory).</summary>
    public const int SliceLength = 2048;

    /// <summary>Largest top-k the two-stage sampler takes (slots each slice keeps).</summary>
    public const int MaxTopKSlots = 64;

    /// <summary>Largest vocabulary the narrow sampler and penalties (one invocation per row, no barriers) are meant for.</summary>
    public const int NarrowVocabulary = 256;

    // The statistics stored per sampled token (Backend.SampleRows).
    private const int StatsPerToken = 13;

    private static IEnumerable<(string, Func<SpirvKernel>)> SamplingKernels()
    {
        // history[r, length % capacity] = ids[r] (length read from device memory, not advanced). One invocation per row.
        yield return ("history_push", () =>
        {
            var k = new KernelBuilder("history_push", Block);
            var (ids, history, length) = (k.Buffer("ids"), k.Buffer("history"), k.Buffer("length"));
            var (rows, capacity) = (k.PushInt("rows"), k.PushInt("capacity"));
            var at = (length[k.Int(0)].ToUInt() % capacity.ToUInt()).ToInt();
            Grid(k, rows, r => history[r * capacity + at] = ids[r]);
            return k.Build();
        });

        // Penalties and the sampler: a workgroup per row, or one invocation per row ("_narrow") for small vocabularies.
        foreach (bool narrow in new[] { false, true })
        {
            string penalize = narrow ? "penalize_rows_narrow" : "penalize_rows";
            yield return (penalize, () => Penalize(penalize, narrow));
            string sample = narrow ? "sample_rows_narrow" : "sample_rows";
            yield return (sample, () => Sampler(sample, slots: false, narrow));
        }

        // The sampler for top-k over the slots topk_slots filled (see Sampler).
        yield return ("sample_rows_slots", () => Sampler("sample_rows_slots", slots: true, narrow: false));

        // Stage one of top-k sampling over a large vocabulary: one workgroup per (row, slice) pair g = r·blocks + b (pairs
        // looped with a stride of the group count) takes scores [b·SliceLength, …) of row r, scaled by invT (NaN as -inf,
        // which no cut-off keeps), into workgroup memory, and keeps its topK largest distinct scores, largest first, with how
        // many times each occurs: slots[g·topK + t] and counts[…] (-inf and 0 past its distinct scores). Pass t takes the
        // largest score below pass t - 1's, and counts it (workgroup reductions, as the sampler's own top-k). The row's k-th
        // largest distinct score is among the slots, and every score at or above it is counted there.
        yield return ("topk_slots", () =>
        {
            var k = new KernelBuilder("topk_slots", Block);
            var (logits, slots, counts) = (k.Buffer("logits"), k.Buffer("slots"), k.Buffer("counts"));
            var (vocabulary, rowStride, rowOffset, invT, topK, blocks, rows) = (k.PushInt("vocabulary"), k.PushInt("rowStride"),
                k.PushInt("rowOffset"), k.PushFloat("invT"), k.PushInt("topK"), k.PushInt("blocks"), k.PushInt("rows"));
            var slice = k.Shared("slice", SliceLength);
            var scratch = k.Shared("scratch", Block);
            var lane = k.LocalX;
            k.For(k.GroupX, rows * blocks, g =>
            {
                var (r, b) = (g / blocks, g % blocks);
                var first = b * SliceLength;
                var length = k.Min(vocabulary - first, k.Int(SliceLength));
                var source = r * rowStride + rowOffset + first;
                k.For(lane, k.Int(SliceLength), Block, i =>
                {
                    var v = k.Local(float.NegativeInfinity);
                    k.If(i < length, () =>
                    {
                        var s = logits[source + i] * invT;
                        v.V = k.Select(s.IsNan(), k.Float(float.NegativeInfinity), s);
                    });
                    slice[i] = v.V;
                });
                k.Barrier();
                var threshold = k.Local(float.PositiveInfinity);
                var o = g * topK;
                k.For(k.Int(0), topK, 1, t =>
                {
                    var below = k.Local(float.NegativeInfinity);
                    k.For(lane, k.Int(SliceLength), Block, i =>
                    {
                        var v = slice[i];
                        below.V = k.Select((v < threshold.V) & (v > below.V), v, below.V);
                    });
                    var next = k.ReduceMax(scratch, below.V);
                    var found = next > float.NegativeInfinity;
                    var same = k.Local(0f);
                    k.For(lane, k.Int(SliceLength), Block, i => same.V = same.V + k.Select(found & slice[i].Eq(next), k.Float(1f), k.Float(0f)));
                    var count = k.ReduceSum(scratch, same.V);
                    k.If(lane.Eq(0), () =>
                    {
                        slots[o + t] = next;
                        counts[o + t] = count;
                    });
                    threshold.V = next;
                });
                k.Barrier();                                             // the slice is read before the next pair overwrites it
            }, k.GroupsX);
            return k.Build();
        });

        foreach (bool narrow in new[] { false, true })
        {
            string single = narrow ? "rms_norm_rope_narrow" : "rms_norm_rope";
            yield return (single, () => NormRope(single, pair: false, narrow));
            string pair = narrow ? "rms_norm_rope_pair_narrow" : "rms_norm_rope_pair";
            yield return (pair, () => NormRope(pair, pair: true, narrow));
        }
    }

    // Repetition penalties: the row's logits are copied to work[r, :]; then history entries k (newest first) of the window
    // n = min(length, window) are taken in turn and, at the token's most recent occurrence, penalized once with the token's
    // count in the window (as CpuBackend.PenalizeRows).
    private static SpirvKernel Penalize(string name, bool narrow)
    {
        var k = new KernelBuilder(name, Block);
        var (logits, work, history, length) = (k.Buffer("logits"), k.Buffer("work"), k.Buffer("history"), k.Buffer("length"));
        var (rows, vocabulary, rowStride, rowOffset) = (k.PushInt("rows"), k.PushInt("vocabulary"), k.PushInt("rowStride"), k.PushInt("rowOffset"));
        var (capacity, window) = (k.PushInt("capacity"), k.PushInt("window"));
        var (repeat, presence, frequency) = (k.PushFloat("repeat"), k.PushFloat("presence"), k.PushFloat("frequency"));
        var len = length[k.Int(0)].ToUInt();
        var n = k.Min(len, window.ToUInt()).ToInt();
        var cap = capacity.ToUInt();
        var w = new RowWork(k, narrow);
        w.Each(rows, r =>
        {
            var source = r * rowStride + rowOffset;
            var o = r * vocabulary;
            w.Columns(vocabulary, j => work[o + j] = logits[source + j]);
            if (!narrow)
            {
                k.BufferBarrier();                                       // the copy lands before any penalty overwrites it
            }

            // History entry q (0 = newest) of row r.
            Val Entry(Val q) => history[r * capacity + ((len - 1 - q.ToUInt()) % cap).ToInt()].ToInt();

            w.Columns(n, q0 =>
            {
                var id = Entry(q0);
                var count = k.Local(0);
                var seen = k.Local(k.Bool(false));
                k.For(k.Int(0), n, 1, q =>
                {
                    k.If(Entry(q).Eq(id), () =>
                    {
                        count.V = count.V + 1;
                        seen.V = seen.V | (q < q0);
                    });
                });
                k.If(!seen.V & (id >= 0) & (id < vocabulary), () =>
                {
                    var x = logits[source + id];
                    var penalized = k.Select(x > 0f, x / repeat, x * repeat);
                    work[o + id] = penalized - presence - frequency * count.V.ToFloat();
                });
            });
            if (!narrow)
            {
                k.BufferBarrier();                                       // before the next row reuses no memory, but keeps passes apart
            }
        });
        return k.Build();
    }

    // u in [0, 1) from (seed, step, row): the counter-based stream (CounterRandom.Uniform).
    private static Val Uniform(KernelBuilder k, Val seed, Val step, Val row)
    {
        var h = k.Local(seed ^ (step * k.UInt(0x9E3779B9u)) ^ (row.ToUInt() * k.UInt(0x85EBCA6Bu)));
        h.V = h.V ^ (h.V >> 16);
        h.V = h.V * k.UInt(0x85EBCA6Bu);
        h.V = h.V ^ (h.V >> 13);
        h.V = h.V * k.UInt(0xC2B2AE35u);
        h.V = h.V ^ (h.V >> 16);
        return (h.V >> 8).ToFloat() * (1f / 16777216f);
    }

    // The sampler (Backend.SampleRows; CpuBackend.SampleRows pass by pass): 1. max of the scaled scores s = logit · invT;
    // 2. top-k: the k-th largest distinct score, by k passes each taking the largest score below the last one; 3. min-p: at
    // least max + ln(minP); 4. top-p: the highest cut-off keeping topP of the mass, by 24 bisection steps; 5. the draw at
    // u · Σ e^(s - max) over the kept scores (u from the counter-based stream); 6. entropy and the five most likely tokens
    // in one pass. A workgroup per row: the draw sums contiguous chunks (one per invocation), invocation 0 finds the chunk
    // where the running total passes the target and that chunk's invocation finds the token; each invocation keeps its own
    // five most likely tokens, merged by five arg-max reductions. Narrow: one invocation per row does every pass in token
    // order, as the CPU. With slots (topk_slots), steps 1 to 4 read the slots instead of the row: every score the cut-offs
    // keep is there with its count.
    private static SpirvKernel Sampler(string name, bool slots, bool narrow)
    {
        var k = new KernelBuilder(name, Block);
        var (logits, ids, stats, step) = (k.Buffer("logits"), k.Buffer("ids"), k.Buffer("stats"), k.Buffer("step"));
        var (slotValues, slotCounts) = slots ? (k.Buffer("slots"), k.Buffer("counts")) : (null, null);
        var (vocabulary, rowStride, rowOffset, rows) = (k.PushInt("vocabulary"), k.PushInt("rowStride"), k.PushInt("rowOffset"), k.PushInt("rows"));
        var (invT, topK, topP, minP, logMinP) = (k.PushFloat("invT"), k.PushInt("topK"), k.PushFloat("topP"), k.PushFloat("minP"), k.PushFloat("logMinP"));
        var (seed, slotCount) = (k.PushUInt("seed"), k.PushInt("slotCount"));
        var stepNumber = step[k.Int(0)].ToUInt();
        var w = new RowWork(k, narrow);
        var (values, indices, parts, shared) = narrow ? (null, null, null, null)
            : (k.Shared("values", Block), k.Shared("indices", Block), k.Shared("parts", Block), k.Shared("shared", 4));

        w.Each(rows, r =>
        {
            var source = r * rowStride + rowOffset;
            Val Score(Val j) => logits[source + j] * invT;

            // The candidates of steps 1 to 4: the row's scores (each once), or the row's slots (each `count` times).
            var candidates = slots ? slotCount : vocabulary;
            var slotBase = r * slotCount;
            Val Candidate(Val c) => slots ? slotValues![slotBase + c] : Score(c);
            Val Times(Val c) => slots ? slotCounts![slotBase + c] : k.Float(1f);

            var max = w.RowMax(candidates, Candidate);
            var threshold = k.Local(float.NegativeInfinity);
            k.If((topK > 0) & (topK < vocabulary), () =>
            {
                var cut = k.Local(float.PositiveInfinity);
                k.For(k.Int(0), topK, 1, _ =>
                {
                    var below = k.Local(float.NegativeInfinity);
                    w.Columns(candidates, c =>
                    {
                        var v = Candidate(c);
                        below.V = k.Select((v < cut.V) & (v > below.V), v, below.V);
                    });
                    cut.V = w.Max(below.V);
                });
                threshold.V = cut.V;
            });
            k.If(minP > 0f, () => threshold.V = k.Max(threshold.V, max + logMinP));
            k.If((topP > 0f) & (topP < 1f), () =>
            {
                var total = w.RowSum(candidates, c =>
                {
                    var v = Candidate(c);
                    return k.Select(v >= threshold.V, Times(c) * k.Exp(v - max), k.Float(0f));
                });
                var goal = total * topP;
                var lo = k.Local(k.Max(max - 40f, threshold.V));
                var hi = k.Local(max);
                k.For(k.Int(0), k.Int(24), 1, _ =>
                {
                    var mid = (lo.V + hi.V) * 0.5f;
                    var mass = k.Local(0f);
                    w.Columns(candidates, c =>
                    {
                        var v = Candidate(c);
                        k.If(v >= mid, () => mass.V = mass.V + Times(c) * k.Exp(v - max));
                    });
                    var kept = w.Sum(mass.V) >= goal;
                    lo.V = k.Select(kept, mid, lo.V);
                    hi.V = k.Select(kept, hi.V, mid);
                });
                threshold.V = lo.V;
            });

            var cutoff = threshold.V;
            Val Weight(Val j)
            {
                var s = Score(j);
                return k.Select(s >= cutoff, k.Exp(s - max), k.Float(0f));
            }

            var u = Uniform(k, seed, stepNumber, r);
            var (token, tokenWeight, total) = narrow ? DrawInOrder(k, vocabulary, u, Weight) : DrawByChunks(k, vocabulary, u, Weight, parts!, shared!);

            // Entropy (bits) and the five heaviest tokens of each invocation (heaviest first; the lower id first on ties:
            // an invocation meets its tokens in increasing order and a newcomer only passes strictly lighter ones).
            var entropy = k.Local(0f);
            var topWeights = Enumerable.Range(0, 5).Select(_ => k.Local(0f)).ToArray();
            var topIds = Enumerable.Range(0, 5).Select(_ => k.Local(-1)).ToArray();
            w.Columns(vocabulary, j =>
            {
                var e = Weight(j);
                k.If(e > 0f, () =>
                {
                    var p = e / total;
                    entropy.V = entropy.V - p * k.Log2(p);
                    var carry = k.Local(e);
                    var carryId = k.Local(j);
                    var shifting = k.Local(k.Bool(false));                // once inserted, the rest shift down a place
                    for (int s = 0; s < 5; s++)
                    {
                        var take = shifting.V | (carry.V > topWeights[s].V);
                        shifting.V = take;
                        var (kept, keptId) = (topWeights[s].V, topIds[s].V);
                        topWeights[s].V = k.Select(take, carry.V, kept);
                        topIds[s].V = k.Select(take, carryId.V, keptId);
                        carry.V = k.Select(take, kept, carry.V);
                        carryId.V = k.Select(take, keptId, carryId.V);
                    }
                });
            });
            var bits = w.Sum(entropy.V);
            var o = (stepNumber.ToInt() * rows + r) * StatsPerToken;
            w.Leader(() =>
            {
                ids[r] = token.ToFloat();
                stats[o] = token.ToFloat();
                stats[o + 1] = tokenWeight / total;
                stats[o + 2] = bits;
            });

            // The five heaviest: an invocation's own (narrow), or five rounds where the heaviest head of the invocations'
            // lists wins (the lower id on ties) and leaves its list.
            for (int a = 0; a < 5; a++)
            {
                Val best, bestId;
                if (narrow)
                {
                    (best, bestId) = (topWeights[a].V, topIds[a].V);
                }
                else
                {
                    (best, bestId) = ReduceArgMax(k, values!, indices!, topWeights[0].V, topIds[0].V);
                    var won = topIds[0].V.Eq(bestId) & (bestId >= 0);
                    for (int s = 0; s < 5; s++)
                    {
                        var (nextWeight, nextId) = s < 4 ? (topWeights[s + 1].V, topIds[s + 1].V) : (k.Float(0f), k.Int(-1));
                        topWeights[s].V = k.Select(won, nextWeight, topWeights[s].V);
                        topIds[s].V = k.Select(won, nextId, topIds[s].V);
                    }
                }

                int slot = a;
                var (weight, id) = (best, bestId);
                w.Leader(() =>
                {
                    var present = (id >= 0) & (weight > 0f);
                    stats[o + 3 + 2 * slot] = k.Select(present, id, k.Int(-1)).ToFloat();
                    stats[o + 4 + 2 * slot] = k.Select(present, weight / total, k.Float(0f));
                });
            }
        });
        return k.Build();
    }

    // The draw by one invocation in token order (as the CPU): the sum of the weights, then the first token where the running
    // total passes u · sum (the last kept one if rounding never passes it). Returns the token, its weight and the sum.
    private static (Val Token, Val Weight, Val Total) DrawInOrder(KernelBuilder k, Val vocabulary, Val u, Func<Val, Val> weight)
    {
        var sum = k.Local(0f);
        k.For(k.Int(0), vocabulary, 1, j => sum.V = sum.V + weight(j));
        var target = u * sum.V;
        var chosen = k.Local(-1);
        var chosenWeight = k.Local(0f);
        var running = k.Local(0f);
        var walk = k.Local(0);
        var done = k.Local(k.Bool(false));
        k.While(() => (walk.V < vocabulary) & !done.V, () =>
        {
            var e = weight(walk.V);
            k.If(e > 0f, () =>
            {
                chosen.V = walk.V;
                chosenWeight.V = e;
                running.V = running.V + e;
                done.V = running.V > target;
            });
            walk.V = walk.V + 1;
        });
        return (chosen.V, chosenWeight.V, sum.V);
    }

    // The draw by a workgroup: each invocation sums a contiguous chunk of the weights; invocation 0 adds the chunks up and
    // finds the one where the running total passes u · sum; that chunk's invocation walks it. Returns, to every
    // invocation, the token, its weight and the sum.
    private static (Val Token, Val Weight, Val Total) DrawByChunks(KernelBuilder k, Val vocabulary, Val u, Func<Val, Val> weight,
        SharedArray parts, SharedArray shared)
    {
        var lane = k.LocalX;
        var chunk = (vocabulary + (Block - 1)) / Block;
        var start = k.Min(lane * chunk, vocabulary);
        var end = k.Min(start + chunk, vocabulary);
        var part = k.Local(0f);
        k.For(start, end, 1, j => part.V = part.V + weight(j));
        parts[lane] = part.V;
        k.Barrier();
        k.If(lane.Eq(0), () =>
        {
            var sum = k.Local(0f);
            k.For(k.Int(0), k.Int(Block), 1, c => sum.V = sum.V + parts[c]);
            var target = u * sum.V;
            var at = k.Local(-1);
            var before = k.Local(0f);
            var running = k.Local(0f);
            var done = k.Local(k.Bool(false));
            k.For(k.Int(0), k.Int(Block), 1, c =>
            {
                var p = parts[c];
                k.If(!done.V & (p > 0f), () =>
                {
                    at.V = c;
                    before.V = running.V;
                    running.V = running.V + p;
                    done.V = running.V > target;
                });
            });
            shared[k.Int(0)] = sum.V;
            shared[k.Int(1)] = target;
            shared[k.Int(2)] = at.V.AsFloat();
            shared[k.Int(3)] = before.V;
        });
        k.Barrier();
        var total = shared[k.Int(0)];
        var goal = shared[k.Int(1)];
        var chosenChunk = shared[k.Int(2)].AsInt();
        var preceding = shared[k.Int(3)];
        k.Barrier();
        k.If(lane.Eq(chosenChunk), () =>
        {
            var chosen = k.Local(-1);
            var chosenWeight = k.Local(0f);
            var running = k.Local(preceding);
            var walk = k.Local(start);
            var done = k.Local(k.Bool(false));
            k.While(() => (walk.V < end) & !done.V, () =>
            {
                var e = weight(walk.V);
                k.If(e > 0f, () =>
                {
                    chosen.V = walk.V;
                    chosenWeight.V = e;
                    running.V = running.V + e;
                    done.V = running.V > goal;
                });
                walk.V = walk.V + 1;
            });
            shared[k.Int(2)] = chosen.V.AsFloat();
            shared[k.Int(3)] = chosenWeight.V;
        });
        k.Barrier();
        var token = shared[k.Int(2)].AsInt();
        var tokenWeight = shared[k.Int(3)];
        k.Barrier();
        return (token, tokenWeight, total);
    }

    // The largest (value, id) over the workgroup (the lower id on equal values), returned to every invocation.
    private static (Val Value, Val Id) ReduceArgMax(KernelBuilder k, SharedArray values, SharedArray indices, Val value, Val id)
    {
        var lane = k.LocalX;
        values[lane] = value;
        indices[lane] = id.AsFloat();
        k.Barrier();
        for (int stride = Block / 2; stride > 0; stride /= 2)
        {
            int s = stride;
            k.If(lane < s, () =>
            {
                var (va, vb) = (values[lane], values[lane + s]);
                var (ia, ib) = (indices[lane].AsInt(), indices[lane + s].AsInt());
                k.If((vb > va) | (vb.Eq(va) & (ib < ia) & (ib >= 0)) | ((ia < 0) & (ib >= 0)), () =>
                {
                    values[lane] = vb;
                    indices[lane] = ib.AsFloat();
                });
            });
            k.Barrier();
        }

        var result = (values[k.Int(0)], indices[k.Int(0)].AsInt());
        k.Barrier();
        return result;
    }

    // RMS normalization with gain of each row (a head's vector), then the rotary embedding with sign 1 (Backend.RmsNormRope:
    // normalized = x · inv · (gain + offset); y = rope(normalized) on the first 2·half dimensions, normalized beyond). Rows
    // are batch·steps·heads; row r's position is positions[r / heads % steps]. The pair variant takes rows1 rows of x and
    // then rows2 of x2 (each with its own gain, eps, offset and heads per step) in one dispatch.
    private static SpirvKernel NormRope(string name, bool pair, bool narrow)
    {
        var k = new KernelBuilder(name, Block);
        var (x, gain, y) = (k.Buffer("x"), k.Buffer("gain"), k.Buffer("y"));
        var (x2, gain2, y2) = pair ? (k.Buffer("x2"), k.Buffer("gain2"), k.Buffer("y2")) : (null, null, null);
        var (cos, sin, positions) = (k.Buffer("cos"), k.Buffer("sin"), k.Buffer("positions"));
        var (rows, cols, eps, offset, heads) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"), k.PushFloat("offset"), k.PushInt("heads"));
        var (rows2, eps2, offset2, heads2) = pair ? (k.PushInt("rows2"), k.PushFloat("eps2"), k.PushFloat("offset2"), k.PushInt("heads2")) : default;
        var (steps, half, interleaved) = (k.PushInt("steps"), k.PushInt("half"), k.PushInt("interleaved"));
        var w = new RowWork(k, narrow);
        var pairs = interleaved.Ne(0);

        void Row(Buf xs, Buf g, Buf ys, Val r, Val epsilon, Val shift, Val perStep)
        {
            var o = r * cols;
            var squares = k.Local(0f);
            w.Columns(cols, j => squares.V = k.Fma(xs[o + j], xs[o + j], squares.V));
            var scale = 1f / k.Sqrt(w.Sum(squares.V) / cols.ToFloat() + epsilon);
            var position = positions[r / perStep % steps].ToInt();
            w.Columns(half, p =>
            {
                var i = k.Select(pairs, 2 * p, p);
                var j = k.Select(pairs, 2 * p + 1, p + half);
                var a = xs[o + i] * scale * (g[i] + shift);
                var b = xs[o + j] * scale * (g[j] + shift);
                var c = cos[position * half + p];
                var s = sin[position * half + p];
                ys[o + i] = k.Fma(a, c, -(b * s));
                ys[o + j] = k.Fma(b, c, a * s);
            });

            // Dimensions past the rotated ones: normalized only.
            w.Columns(cols - 2 * half, q =>
            {
                var j = 2 * half + q;
                ys[o + j] = xs[o + j] * scale * (g[j] + shift);
            });
        }

        w.Each(pair ? rows + rows2 : rows, r =>
        {
            if (!pair)
            {
                Row(x, gain, y, r, eps, offset, heads);
                return;
            }

            k.If(r < rows, () => Row(x, gain, y, r, eps, offset, heads), () => Row(x2!, gain2!, y2!, r - rows, eps2, offset2, heads2));
        });
        return k.Build();
    }
}
