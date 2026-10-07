// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Matrix products: the float32 product tiled through workgroup memory, and the packed weight products (int8, int4,
// bfloat16) with their expansions to float.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> MatMulKernels()
    {
        // The tiled float32 products: register-blocked (4 × 4 outputs per invocation) and one output per invocation, for
        // workgroups of side × side invocations (side = MatSide(width)); the backend measures which (or the small kernel)
        // is fastest per shape.
        yield return ("batched_matmul", BatchedMatMul);
        yield return ("batched_matmul_tile", BatchedMatMulTile);

        // batched_matmul for small products (attention heads, few rows): one invocation per output, no workgroup memory or
        // barriers; any number of groups (grid stride over batch · m · n).
        yield return ("batched_matmul_small", () =>
        {
            var k = new KernelBuilder("batched_matmul_small", Block);
            var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
            var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
            var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
            var (ta, tb) = (transA.Ne(0), transB.Ne(0));
            Grid(k, batch * m * n, o =>
            {
                var (bi, rest) = (o / (m * n), o % (m * n));
                var (row, col) = (rest / n, rest % n);
                var aBase = bi * m * depth;
                var bBase = bi * depth * n;
                // op(a)[row, kk] = a[aBase + aStart + kk · aStep]; op(b)[kk, col] = b[bBase + bStart + kk · bStep].
                var (aStart, aStep) = (k.Select(ta, row, row * depth), k.Select(ta, m, k.Int(1)));
                var (bStart, bStep) = (k.Select(tb, col * depth, col), k.Select(tb, k.Int(1), n));
                var acc = k.Local(0f);
                k.For(k.Int(0), depth, 1, kk => acc.V = k.Fma(a[aBase + aStart + kk * aStep], b[bBase + bStart + kk * bStep], acc.V));
                k.If(beta.Eq(0), () => c[o] = acc.V, () => c[o] = acc.V + beta * c[o]);
            });
            return k.Build();
        });

        // The packed weight products y[m, n] = x[m, k] · w[k, n], w read from its packed words (see PackedGemv): for
        // each format, one kernel per row block (1, 2, 4 or 8 rows of x per workgroup), and the pass adding the
        // partial sums of split-k dispatches.
        foreach (var format in new[] { PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16 })
        {
            foreach (int rows in GemvRowBlocks)
            {
                foreach (int words in GemvWordCounts)
                {
                    string name = GemvName(format, rows, words);
                    yield return (name, () => PackedGemv(name, format, rows, words));
                }
            }
        }

        // y[i] = Σ_s part[s, i] over the splits in order (· scales[i % n] for int8): one invocation per output.
        yield return ("gemv_reduce", () => GemvReduce("gemv_reduce", scaled: false));
        yield return ("int8_gemv_reduce", () => GemvReduce("int8_gemv_reduce", scaled: true));

        yield return ("int8_dequantize", () =>
        {
            var k = new KernelBuilder("int8_dequantize", Block);
            var (q, scales, w) = (k.Buffer("q"), k.Buffer("scales"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                w[i] = Int8At(k, q, r * ((n + 3) / 4), j) * scales[j];
            });
            return k.Build();
        });

        yield return ("int4_dequantize", () =>
        {
            var k = new KernelBuilder("int4_dequantize", Block);
            var (q, scales, w) = (k.Buffer("q"), k.Buffer("scales"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                var words = (n + 7) / 8;
                w[i] = Int4At(k, q, r * words, j) * scales[r / 32 * (words * 8) + j];
            });
            return k.Build();
        });

        yield return ("bf16_dequantize", () =>
        {
            var k = new KernelBuilder("bf16_dequantize", Block);
            var (packed, w) = (k.Buffer("packed"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                w[i] = BFloat16At(k, packed, r * ((n + 1) / 2) + (j >> 1), j);
            });
            return k.Build();
        });

        // Word i of packed = x[2i], x[2i + 1] rounded to bfloat16 (to nearest, ties to even; low half first; 0 past n).
        yield return ("bf16_pack", () =>
        {
            var k = new KernelBuilder("bf16_pack", Block);
            var (x, packed) = (k.Buffer("x"), k.Buffer("packed"));
            var n = k.PushInt("n");
            Grid(k, (n + 1) / 2, i =>
            {
                var low = RoundBFloat16(k, x[2 * i]);
                var high = k.Local(k.UInt(0));
                k.If(2 * i + 1 < n, () => high.V = RoundBFloat16(k, x[2 * i + 1]));
                packed[i] = (low | high.V << 16).AsFloat();
            });
            return k.Build();
        });
    }

    /// <summary>
    /// Side of the square workgroup of the tiled products for a width: the largest power of two whose square is at most
    /// the width (16 for 256 invocations, 8 for 64 or 128).
    /// </summary>
    public static int MatSide(int width)
    {
        int side = 1;
        while ((side * 2) * (side * 2) <= width)
        {
            side *= 2;
        }

        return side;
    }

    /// <summary>Outputs along each side an invocation of <c>batched_matmul</c> keeps (4 × 4): its block of c is 4 · side.</summary>
    public const int MatPer = 4;

    // Rows of k staged per step of batched_matmul.
    private const int MatDepth = 16;

    // c = op(a) · op(b) + beta · c for batch contiguous triples; a [m, k] (transA: stored [k, m]), b [k, n] (transB: [n, k]).
    // A workgroup of S × S invocations (S = MatSide(width), 16 for 256) computes one T × T block of c, T = 4S:
    // invocation (tx, ty) = (i % S, i / S) keeps the 4 × 4 outputs (ty + S r, tx + S s) in registers. Per step of 16
    // along k, the workgroup stages op(a)[T × 16] and op(b)[16 × T] in workgroup memory (each invocation loading 64 / S
    // elements of each, consecutive invocations reading consecutive addresses of a and b, transposed or not; zeros
    // outside), then each invocation reads 4 + 4 staged values per k for 16 multiply-adds. Every output adds its k terms
    // in order. beta = 0 writes the product without reading c (whatever c held).
    //
    // Workgroup memory: tileA[kk · (T + 1) + r] and tileB[kk · (T + 1) + col] (rows padded by one: the stores along k
    // and the reads along m or n fall in different banks) are written only between the barrier ending the previous
    // step's reads and the barrier before this step's reads, each element by exactly one invocation.
    private static SpirvKernel BatchedMatMul()
    {
        int side = MatSide(Block), threads = side * side, Per = MatPer, tileEdge = Per * side, stride = tileEdge + 1;
        int sideShift = System.Numerics.BitOperations.Log2((uint)side), edgeShift = System.Numerics.BitOperations.Log2((uint)tileEdge);
        var k = new KernelBuilder("batched_matmul", threads);
        var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
        var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
        var tileA = k.Shared("tileA", MatDepth * stride);
        var tileB = k.Shared("tileB", MatDepth * stride);
        var tid = k.LocalX;
        var (tx, ty) = (tid & (side - 1), tid >> sideShift);
        var (ta, tb) = (transA.Ne(0), transB.Ne(0));
        var rowBase = k.GroupY * tileEdge;
        var colBase = k.GroupX * tileEdge;
        var acc = new Var[Per * Per];
        for (int i = 0; i < acc.Length; i++)
        {
            acc[i] = k.Local(ScalarKind.Float);
        }

        // Where invocation tid's p-th element of each staged tile comes from: (r, kk) of op(a), (kk, col) of op(b).
        // Untransposed a and transposed b are read along k (16 consecutive invocations per row), the others along m or n.
        var loads = new (Val R, Val KA, Val Col, Val KB)[tileEdge * MatDepth / threads];
        for (int p = 0; p < loads.Length; p++)
        {
            var l = tid + p * threads;
            loads[p] = (k.Select(ta, l & (tileEdge - 1), l >> 4), k.Select(ta, l >> edgeShift, l & (MatDepth - 1)),
                k.Select(tb, l >> 4, l & (tileEdge - 1)), k.Select(tb, l & (MatDepth - 1), l >> edgeShift));
        }

        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            k.For(k.Int(0), depth, MatDepth, t =>
            {
                foreach (var (r, ka, col, kb) in loads)
                {
                    var (row, kka) = (rowBase + r, t + ka);
                    var va = k.Local(0f);
                    k.If((row < m) & (kka < depth), () => va.V = a[aBase + k.Select(ta, kka * m + row, row * depth + kka)]);
                    tileA[ka * stride + r] = va.V;
                    var (column, kkb) = (colBase + col, t + kb);
                    var vb = k.Local(0f);
                    k.If((kkb < depth) & (column < n), () => vb.V = b[bBase + k.Select(tb, column * depth + kkb, kkb * n + column)]);
                    tileB[kb * stride + col] = vb.V;
                }

                k.Barrier();
                var sums = acc.Select(v => v.V).ToArray();
                for (int e = 0; e < MatDepth; e++)
                {
                    var av = new Val[Per];
                    var bv = new Val[Per];
                    for (int i = 0; i < Per; i++)
                    {
                        av[i] = tileA[k.Int(e * stride + side * i) + ty];
                        bv[i] = tileB[k.Int(e * stride + side * i) + tx];
                    }

                    for (int i = 0; i < Per; i++)
                    {
                        for (int j = 0; j < Per; j++)
                        {
                            sums[i * Per + j] = k.Fma(av[i], bv[j], sums[i * Per + j]);
                        }
                    }
                }

                for (int i = 0; i < acc.Length; i++)
                {
                    acc[i].V = sums[i];
                }

                k.Barrier();
            });

            for (int i = 0; i < Per; i++)
            {
                for (int j = 0; j < Per; j++)
                {
                    var (row, col, value) = (rowBase + ty + side * i, colBase + tx + side * j, acc[i * Per + j]);
                    k.If((row < m) & (col < n), () =>
                    {
                        var o = bi * m * n + row * n + col;
                        k.If(beta.Eq(0), () => c[o] = value.V, () => c[o] = value.V + beta * c[o]);
                    });
                }
            }
        }, k.GroupsZ);
        return k.Build();
    }

    // batched_matmul with one output per invocation: each S × S workgroup (S = MatSide(width)) computes one S × S block of
    // c, staging S × S tiles of a and b in workgroup memory (fewer registers, more workgroups: better where the blocks of
    // batched_matmul are too few or too large for the device).
    //
    // Workgroup memory: tileA[ty · S + tx], tileB[ty · S + tx] written by invocation (tx, ty) only, between the barrier
    // ending the previous step's reads and the barrier before this step's reads.
    private static SpirvKernel BatchedMatMulTile()
    {
        int side = MatSide(Block);
        var k = new KernelBuilder("batched_matmul_tile", side, side);
        var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
        var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
        var tileA = k.Shared("tileA", side * side);
        var tileB = k.Shared("tileB", side * side);
        var (tx, ty) = (k.LocalX, k.LocalY);
        var row = k.GroupY * side + ty;
        var col = k.GroupX * side + tx;
        var at = ty * side + tx;
        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            var acc = k.Local(0f);
            k.For(k.Int(0), depth, side, t =>
            {
                // tileA[ty, tx] = op(a)[row, t + tx]; tileB[ty, tx] = op(b)[t + ty, col]; zeros outside.
                var (ka, kb) = (t + tx, t + ty);
                var va = k.Local(0f);
                k.If((row < m) & (ka < depth), () => va.V = a[aBase + k.Select(transA.Ne(0), ka * m + row, row * depth + ka)]);
                tileA[at] = va.V;
                var vb = k.Local(0f);
                k.If((kb < depth) & (col < n), () => vb.V = b[bBase + k.Select(transB.Ne(0), col * depth + kb, kb * n + col)]);
                tileB[at] = vb.V;
                k.Barrier();
                var sum = acc.V;
                for (int e = 0; e < side; e++)
                {
                    sum = k.Fma(tileA[ty * side + e], tileB[k.Int(e * side) + tx], sum);
                }

                acc.V = sum;
                k.Barrier();
            });
            k.If((row < m) & (col < n), () =>
            {
                var o = bi * m * n + row * n + col;
                k.If(beta.Eq(0), () => c[o] = acc.V, () => c[o] = acc.V + beta * c[o]);
            });
        }, k.GroupsZ);
        return k.Build();
    }

    /// <summary>The packed weight formats the products read.</summary>
    internal enum PackedFormat
    {
        Int8,
        Int4,
        BFloat16,
    }

    /// <summary>Rows of x one packed-product workgroup takes (the kernel variants; more rows run as several row blocks).</summary>
    public static readonly int[] GemvRowBlocks = [1, 2, 4, 8];

    /// <summary>
    /// Packed words (along n) one packed-product workgroup reads per row of k, the kernel variants: 32 (128 contiguous
    /// bytes) or 64; the rest of the workgroup's width is slices of k (width / words). The backend measures which is
    /// faster per shape.
    /// </summary>
    public static readonly int[] GemvWordCounts = [32, 64];

    /// <summary>Rows of k the chunk of each split of a packed product is a multiple of (one int4 scale group).</summary>
    public const int GemvChunkAlign = 32;

    /// <summary>The packed-product kernel for a format, a row block and words per row of k.</summary>
    public static string GemvName(PackedFormat format, int rows, int words) => format switch
    {
        PackedFormat.Int8 => "int8_gemv_",
        PackedFormat.Int4 => "int4_gemv_",
        _ => "bf16_gemv_",
    } + rows + (words == GemvWordCounts[0] ? "" : "_w" + words);

    /// <summary>Columns per packed word: 4 (int8), 8 (int4), 2 (bfloat16).</summary>
    public static int ColumnsPerWord(PackedFormat format) => format switch
    {
        PackedFormat.Int8 => 4,
        PackedFormat.Int4 => 8,
        _ => 2,
    };

    // y[r, j] = Σ_kk x[r, kk] · w(kk, j) for the packed formats, with the weights' layouts:
    //   int8: signed bytes, four per word along each row of k (⌈n / 4⌉ words, byte c = column 4w + c), one scale per
    //         column applied to the sum;
    //   int4: signed nibbles, eight per word (⌈n / 8⌉ words, nibble c = column 8w + c); w = nibble · scales[kk / 32, j]
    //         (scale rows of 8·⌈n / 8⌉ values per 32 rows of k), each weight scaled before its multiply-add;
    //   bfloat16: two per word (⌈n / 2⌉ words, the even column in the low half).
    // Grid: x = ⌈words / L⌉ column blocks, y = splits of k, z = ⌈m / rows⌉ row blocks. A workgroup of W (the width) is
    // L lanes (consecutive words of a row of k: 32 lanes read 128 contiguous bytes) × S = W / L slices of its split's
    // rows [s · chunk, min(k, (s + 1) · chunk)), s = first + y (chunk a multiple of 32, from the host): slice t takes
    // rows ≡ t (mod S), so the workgroup reads S consecutive rows of L words at a time, and each word an invocation reads
    // feeds rows × (columns per word) multiply-adds kept in registers. Then the slices' sums are added in slice order
    // through workgroup memory (`batch` sums per lane per round, two barriers a round) and stored: to y[r, j] when the
    // product has one split in all (`splits`; int8 scaled), else to part[s, r, j] for gemv_reduce. Lanes past the last
    // word read the last word again and store nothing; rows of the block past m read row m - 1 and store nothing.
    //
    // The weights' binding (and int4's scales') starts at row first · chunk of k: a dispatch over one window of weights
    // larger than a binding (VulkanBackend.LargeStorage.cs) takes the splits of that window; otherwise first is 0.
    //
    // Workgroup memory: partial[b · W + slice · L + lane] is written by invocation (slice, lane) only, after a barrier
    // that ends the previous round's reads; it is read after the next barrier.
    private static SpirvKernel PackedGemv(string name, PackedFormat format, int rows, int lanes)
    {
        int threads = Block, slices = threads / lanes, laneShift = System.Numerics.BitOperations.Log2((uint)lanes);
        int perWord = ColumnsPerWord(format);
        int sums = rows * perWord, batch = Math.Min(sums, 8);
        var k = new KernelBuilder(name, threads);
        var x = k.Buffer("x");
        var q = k.Buffer(format == PackedFormat.BFloat16 ? "packed" : "q");
        var scales = format == PackedFormat.BFloat16 ? null : k.Buffer("scales");
        var y = k.Buffer("y");
        var (m, n, depth, chunk) = (k.PushInt("m"), k.PushInt("n"), k.PushInt("k"), k.PushInt("chunk"));
        var (firstSplit, splits) = (k.PushInt("first"), k.PushInt("splits"));
        var partial = k.Shared("partial", threads * batch);
        var tid = k.LocalX;
        var (lane, slice) = (tid & (lanes - 1), tid >> laneShift);
        var words = (n + (perWord - 1)) / perWord;
        var wordAt = k.Min(k.GroupX * lanes + lane, words - 1);
        var split = firstSplit + k.GroupY;
        var baseRow = firstSplit * chunk;                                   // the row of k the weights' binding starts at
        var kBegin = split * chunk;
        var kEnd = k.Min(depth, kBegin + chunk);
        var rowBase = k.GroupZ * rows;
        var xRows = new Val[rows];
        for (int r = 0; r < rows; r++)
        {
            xRows[r] = k.Min(rowBase + r, m - 1) * depth;
        }

        var acc = new Var[sums];
        for (int i = 0; i < sums; i++)
        {
            acc[i] = k.Local(0f);
        }

        // The weights of word `bits` (int4: times the group's scales), then every row's multiply-adds for row kk of k.
        void Accumulate(Val kk, Val bits, Val[]? groupScales)
        {
            var w = new Val[perWord];
            for (int c = 0; c < perWord; c++)
            {
                w[c] = format switch
                {
                    PackedFormat.Int8 => ((bits << (24 - 8 * c)) >> 24).ToFloat(),
                    PackedFormat.Int4 => ((bits << (28 - 4 * c)) >> 28).ToFloat() * groupScales![c],
                    _ => c == 0 ? (bits.AsUInt() << 16).AsFloat() : (bits.AsUInt() & k.UInt(0xFFFF0000u)).AsFloat(),
                };
            }

            for (int r = 0; r < rows; r++)
            {
                var xv = x[xRows[r] + kk];
                for (int c = 0; c < perWord; c++)
                {
                    acc[r * perWord + c].V = k.Fma(xv, w[c], acc[r * perWord + c].V);
                }
            }
        }

        Val Word(Val kk) => q.Int((kk - baseRow) * words + wordAt);

        if (format == PackedFormat.Int4)
        {
            // Group by group (32 rows sharing scales; kBegin is a multiple of 32): this invocation's eight scales once per
            // group, then its 32 / S rows of the group (fewer in the last group of k).
            k.For(kBegin / 32, (kEnd + 31) / 32, 1, g =>
            {
                var groupScales = new Val[perWord];
                for (int c = 0; c < perWord; c++)
                {
                    groupScales[c] = scales![(g - baseRow / 32) * (words * 8) + wordAt * 8 + c];
                }

                var first = g * 32 + slice;
                var end = k.Min(g * 32 + 32, kEnd);
                k.If((g * 32 + 32).Eq(end), () =>
                {
                    var loaded = new Val[32 / slices];
                    for (int u = 0; u < loaded.Length; u++)
                    {
                        loaded[u] = Word(first + u * slices);
                    }

                    for (int u = 0; u < loaded.Length; u++)
                    {
                        Accumulate(first + u * slices, loaded[u], groupScales);
                    }
                }, () => k.For(first, end, slices, kk => Accumulate(kk, Word(kk), groupScales)));
            });
        }
        else
        {
            // Four rows per step (their words loaded before any is used), then the rest one by one.
            const int Unroll = 4;
            var kk = k.Local(kBegin + slice);
            k.While(() => kk.V + (Unroll - 1) * slices < kEnd, () =>
            {
                var at = kk.V;
                var loaded = new Val[Unroll];
                for (int u = 0; u < Unroll; u++)
                {
                    loaded[u] = Word(at + u * slices);
                }

                for (int u = 0; u < Unroll; u++)
                {
                    Accumulate(at + u * slices, loaded[u], null);
                }

                kk.V = at + Unroll * slices;
            });
            k.While(() => kk.V < kEnd, () =>
            {
                var at = kk.V;
                Accumulate(at, Word(at), null);
                kk.V = at + slices;
            });
        }

        // The slices' sums, `batch` per lane per round: output o < L · batch (invocations o, o + W, …) adds sum b = o / L
        // of lane o % L over the S slices in order and stores it.
        var single = splits.Eq(1);
        for (int round = 0; round < sums / batch; round++)
        {
            if (round > 0)
            {
                k.Barrier();                                                   // the previous round's reads are done
            }

            for (int b = 0; b < batch; b++)
            {
                partial[k.Int(b * threads) + tid] = acc[round * batch + b].V;
            }

            k.Barrier();
            int first = round * batch;
            for (int pass = 0; pass * threads < lanes * batch; pass++)
            {
                var o = tid + pass * threads;
                k.If(o < lanes * batch, () =>
                {
                    var (l, b) = (o & (lanes - 1), o >> laneShift);
                    var total = partial[b * threads + l];
                    for (int s = 1; s < slices; s++)
                    {
                        total = total + partial[b * threads + k.Int(s * lanes) + l];
                    }

                    var index = first + b;
                    var (r, c) = (index / perWord, index % perWord);
                    var (row, j) = (rowBase + r, (k.GroupX * lanes + l) * perWord + c);
                    k.If((row < m) & (j < n), () =>
                    {
                        var value = format == PackedFormat.Int8 ? k.Select(single, total * scales![j], total) : total;
                        y[(split * m + row) * n + j] = value;
                    });
                });
            }
        }

        return k.Build();
    }

    // y[i] = Σ_s part[s · count + i] for s < splits, in split order (int8: times scales[i % n]).
    private static SpirvKernel GemvReduce(string name, bool scaled)
    {
        var k = new KernelBuilder(name, Block);
        var part = k.Buffer("part");
        var scales = scaled ? k.Buffer("scales") : null;
        var y = k.Buffer("y");
        var (count, n, splits) = (k.PushInt("count"), k.PushInt("n"), k.PushInt("splits"));
        Grid(k, count, i =>
        {
            var total = k.Local(part[i]);
            k.For(k.Int(1), splits, 1, s => total.V = total.V + part[s * count + i]);
            y[i] = scales is null ? total.V : total.V * scales[i % n];
        });
        return k.Build();
    }

    // Signed byte j of the row of words starting at word rowStart.
    private static Val Int8At(KernelBuilder k, Buf q, Val rowStart, Val j)
    {
        var word = q.Int(rowStart + (j >> 2));
        return word.ShiftLeft(24 - 8 * (j & 3)).ShiftRight(k.Int(24)).ToFloat();
    }

    // Signed nibble j of the row of words starting at word rowStart.
    private static Val Int4At(KernelBuilder k, Buf q, Val rowStart, Val j)
    {
        var word = q.Int(rowStart + (j >> 3));
        return word.ShiftLeft(28 - 4 * (j & 7)).ShiftRight(k.Int(28)).ToFloat();
    }

    // The bfloat16 value in word wordIndex: its high half for odd j, its low half for even j.
    private static Val BFloat16At(KernelBuilder k, Buf packed, Val wordIndex, Val j)
    {
        var word = packed.UInt(wordIndex);
        return k.Select((j & 1).Eq(1), word & k.UInt(0xFFFF0000u), word << 16).AsFloat();
    }

    // The bfloat16 bits of v (as uint in the low half): to nearest, ties to even; NaN → 0x7FC0.
    private static Val RoundBFloat16(KernelBuilder k, Val v)
    {
        var u = v.AsUInt();
        var rounded = (u + k.UInt(0x7FFFu) + ((u >> 16) & 1)) >> 16;
        return k.Select(v.IsNan(), k.UInt(0x7FC0u), rounded);
    }
}
