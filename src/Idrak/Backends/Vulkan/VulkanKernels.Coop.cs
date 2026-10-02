// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Backends.Vulkan;

// Matrix products on the device's matrix units (tensor cores, XMX, WMMA) through cooperative matrices
// (SPV_KHR_cooperative_matrix): 16-bit float operands, 32-bit float sums, in the tile shape the device reports
// (VulkanBackend.Matrix.cs). These kernels are built only for devices that report such a shape, and run only where the
// measured tuning finds them faster than the float32 kernels.
//
// Precision. The products of the library compare with the CPU's float32 sums (relative 1e-4 of the largest output in the
// tests), so a float32 operand is never just rounded to 16 bits (about 5e-4 relative per value). Instead each operand is
// split into two 16-bit floats, x = high + low (high: x's sign, exponent and first 10 mantissa bits, cut with an integer
// mask so no compiler folds the split away; low = x - high, exact, then rounded to 16 bits: 22 of float32's 24
// significant bits), and the product takes high · high + high · low + low · high, each exact in the 32-bit sums: about
// 1e-6 relative, as close to the CPU as two float32 summation orders are. A bfloat16 or int8 weight is exact in one 16-bit float (8
// significant bits), so those products need only high · w + low · w. The 16-bit exponent range (6e-5 to 65504) is not
// float32's, so every row of the left operand and every column of the right one is first multiplied by a power of two
// (exact) that brings its largest magnitude into [2^14, 2^15), and the outputs divided by the same powers: no value
// overflows, and only values below 2^-29 of their row's (or column's) largest lose bits, which changes no sum by more
// than float32 rounding would. (An infinite input, whose row or column is lost anyway, may give NaN where the float32
// kernels give an infinity: its low part is 0 and infinity · 0 is NaN.)
internal static partial class VulkanKernels
{
    /// <summary>A cooperative-matrix shape a device reports: A is M × K, B is K × N, the accumulator M × N.</summary>
    internal readonly record struct CoopShape(int M, int N, int K)
    {
        /// <summary>"16x16x16" (M × N × K).</summary>
        public override string ToString() => $"{M}x{N}x{K}";
    }

    /// <summary>
    /// A cooperative-matrix product kernel: the right operand (null: float32 for batched_matmul; else a packed weight
    /// format), the matrix shape, the workgroup width, for tests the subgroup size of an emulation (0: the device's
    /// cooperative matrices), and the rows of k staged per step (<see cref="CoopDepths"/>, at least K).
    /// </summary>
    internal readonly record struct CoopSpec(PackedFormat? Format, CoopShape Shape, int Width, int Emulated = 0, int Depth = 32)
    {
        /// <summary>The kernel's name: "coop_f32_m16n16k16_w128_d32" (with "_emulated8" for an emulation).</summary>
        public string Name => $"coop_{Format switch { null => "f32", PackedFormat.Int8 => "int8", PackedFormat.Int4 => "int4", _ => "bf16" }}_" +
            $"m{Shape.M}n{Shape.N}k{Shape.K}_w{Width}_d{Depth}{(Emulated > 0 ? $"_emulated{Emulated}" : "")}";
    }

    /// <summary>Rows (and columns) of the output one cooperative-matrix workgroup computes: 2 × 2 subgroup blocks of 32 × 32.</summary>
    public const int CoopBlock = 64;

    /// <summary>
    /// Rows of k a cooperative-matrix product may stage per step (one or more matrix depths), deepest first: 32 halves the
    /// barriers of 16 and takes about 36 KiB of workgroup memory (float32), 16 about 27 KiB. The backend takes the deepest
    /// whose workgroup memory the device has.
    /// </summary>
    public static readonly int[] CoopDepths = [32, 16];

    // Elements of padding at the end of each staged row (16 bytes of 16-bit floats: rows stay 16-byte aligned).
    private const int CoopPad = 8;

    private static readonly ConcurrentDictionary<CoopSpec, SpirvKernel> BuiltCoop = new();

    /// <summary>Whether the kernels can use a reported shape: M, N and K each 8, 16 or 32 (whole tiles of a 32 × 32 subgroup block).</summary>
    public static bool CoopShapeUsable(CoopShape shape) => Usable(shape.M) && Usable(shape.N) && Usable(shape.K);

    private static bool Usable(int size) => size is 8 or 16 or 32;

    /// <summary>
    /// The cooperative-matrix product kernel of <paramref name="spec"/>, built on first use. The width is a power of two
    /// from 16 to <see cref="MaxWidth"/> (the kernel's subgroups take the four subgroup blocks of the workgroup in turns,
    /// whatever their number), and an emulation's subgroup size divides the matrix sizes.
    /// </summary>
    public static SpirvKernel Coop(CoopSpec spec)
    {
        if (!CoopShapeUsable(spec.Shape))
        {
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Shape, "Cooperative-matrix sizes are 8, 16 or 32.");
        }

        if (spec.Width is < 16 or > MaxWidth || (spec.Width & (spec.Width - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Width, $"A power of two from 16 to {MaxWidth}.");
        }

        if (Array.IndexOf(CoopDepths, spec.Depth) < 0 || spec.Depth < spec.Shape.K)
        {
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Depth, "A depth of CoopDepths, at least K.");
        }

        if (spec.Emulated < 0 || spec.Emulated > 0 && (spec.Shape.M * spec.Shape.N % spec.Emulated != 0 || spec.Width % spec.Emulated != 0))
        {
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Emulated, "An emulation's subgroup size divides M · N and the width.");
        }

        return BuiltCoop.GetOrAdd(spec, CoopGemm);
    }

    // y = op(a) · op(b) (+ beta · y for float32; · scales[col] for int8) for one 64 × 64 block of the output per workgroup
    // (grid x = ⌈n / 64⌉, y = ⌈m / 64⌉, batches along z for float32), as four 32 × 32 subgroup blocks of TM × TN matrix
    // tiles (TM = 32 / M, TN = 32 / N). Per batch:
    //   1. the largest magnitude of each of the block's 64 rows of op(a) and 64 columns of op(b) over all of k (16
    //      invocations per line where its elements are contiguous, else invocations along the lines), and from it the
    //      line's power of two (shift s: max · 2^s in [2^14, 2^15));
    //   2. rounds: subgroup g takes subgroup block r · (subgroups) + g, so any number of subgroups computes the four (one
    //      round when there are four or more, the others idle in the products but help stage); per step of D along k (32
    //      or 16) the workgroup stages op(a)[64 × D] and op(b)[D × 64] scaled and split into 16-bit high and low parts
    //      (zeros past m, n or k; the low part 0 where a value is infinite), and each subgroup adds its tiles'
    //      products: low · high, high · low (float32 and int4 b only), then high · high, per matrix depth; at the end of
    //      a round it stores its tiles (float32 sums) into the block's staging;
    //   3. every output in range: its staged sum divided by its row's and column's powers of two, then beta or the int8
    //      scale, written by consecutive invocations along a row.
    //
    // Workgroup memory, each phase separated by a barrier: staged[64 · 64] holds the maxima of step 1 (lines · 16 + part)
    // until the shifts are computed, then the rounds' tiles (each subgroup block written by one subgroup), read in step 3;
    // shifts[64 + 64] written in step 1 by one invocation per line; aHigh, aLow[64 · (D + 8)], bHigh, bLow[D · 72] written
    // only between the barrier ending a step's products and the barrier before the next step's.
    private static SpirvKernel CoopGemm(CoopSpec spec)
    {
        var (M, N, K) = (spec.Shape.M, spec.Shape.N, spec.Shape.K);
        int tm = 32 / M, tn = 32 / N, width = spec.Width;
        const int Edge = CoopBlock, EdgeShift = 6, StrideB = Edge + CoopPad;   // EdgeShift = log2 Edge
        int Depth = spec.Depth, DepthShift = System.Numerics.BitOperations.Log2((uint)spec.Depth), StrideA = Depth + CoopPad;
        var format = spec.Format;
        bool splitB = format is null or PackedFormat.Int4;               // float32 and int4 · scale are not exact in 16 bits
        var k = new KernelBuilder(spec.Name, width);
        Buf a, b, c;
        Buf? scales = null;
        Val batch, m, n, depth, beta;
        Val? ta = null, tb = null;
        if (format is null)
        {
            (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
            (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
            var (transA, transB) = (k.PushInt("transA"), k.PushInt("transB"));
            beta = k.PushFloat("beta");
            (ta, tb) = (transA.Ne(0), transB.Ne(0));
        }
        else
        {
            a = k.Buffer("x");
            b = k.Buffer(format == PackedFormat.BFloat16 ? "packed" : "q");
            scales = format == PackedFormat.BFloat16 ? null : k.Buffer("scales");
            c = k.Buffer("y");
            (m, n, depth) = (k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
            (batch, beta) = (k.Int(1), k.Float(0f));
        }

        var aHigh = k.SharedHalf("aHigh", Edge * StrideA);
        var aLow = k.SharedHalf("aLow", Edge * StrideA);
        var bHigh = k.SharedHalf("bHigh", Depth * StrideB);
        var bLow = splitB ? k.SharedHalf("bLow", Depth * StrideB) : null;
        var staged = k.Shared("staged", Edge * Edge);
        var shifts = k.Shared("shifts", 2 * Edge);                       // rows of op(a), then columns of op(b)
        var ops = new MatrixOps(k, spec.Shape, spec.Emulated);
        var accumulators = new MatrixOps.Accumulator[tm * tn];
        for (int i = 0; i < accumulators.Length; i++)
        {
            accumulators[i] = ops.NewAccumulator();
        }

        var tid = k.LocalX;
        var rowBase = k.GroupY * Edge;
        var colBase = k.GroupX * Edge;
        int perWord = format is { } packed ? ColumnsPerWord(packed) : 1;
        var words = (n + (perWord - 1)) / perWord;

        // op(a)[row, kk] and op(b)[kk, col] (in range).
        Val LoadA(Val aBase, Val row, Val kk) => a[aBase + (ta is { } t ? k.Select(t, kk * m + row, row * depth + kk) : row * depth + kk)];
        Val LoadB(Val bBase, Val kk, Val col) => format switch
        {
            null => b[bBase + k.Select(tb!.Value, col * depth + kk, kk * n + col)],
            PackedFormat.Int8 => Int8At(k, b, kk * words, col),
            PackedFormat.Int4 => Int4At(k, b, kk * words, col) * scales![kk / 32 * (words * 8) + col],
            _ => BFloat16At(k, b, kk * words + (col >> 1), col),
        };

        // 2^s for a shift s in [-126, 126] (a normal float, built from its exponent bits).
        Val Power(Val shift) => ((shift + 127) << 23).AsFloat();

        // x's sign, exponent and first 10 mantissa bits: a 16-bit float wherever x is in its normal range (every value of
        // a line within 2^29 of the line's largest), and x - High(x) is exact (13 bits at most). An infinity stays one.
        Val High(Val x) => (x.AsUInt() & k.UInt(0xFFFFE000u)).AsFloat();

        // Runs `contiguous` where a line's elements are consecutive in memory (op(a) not transposed, op(b) transposed),
        // `strided` otherwise: chosen at run time for float32 operands, known for the packed ones.
        void Layout(Val? flag, bool known, Action contiguous, Action strided)
        {
            if (flag is { } f)
            {
                k.If(f, contiguous, strided);
            }
            else if (known)
            {
                contiguous();
            }
            else
            {
                strided();
            }
        }

        int stridedParts = Math.Max(1, width / Edge);                    // invocations per line along k where lines are strided

        // Step 1: each line's largest magnitude in parts: staged[offset + line · 16 + part].
        void Maxima(Val? contiguousFlag, bool contiguousKnown, int offset, Val lineBase, Val lineLimit, Func<Val, Val, Val> load) => Layout(contiguousFlag, contiguousKnown,
            () => k.For(tid >> 4, k.Int(Edge), width / 16, line =>
            {
                var lane = tid & 15;
                var best = k.Local(0f);
                k.If(lineBase + line < lineLimit, () => k.For(lane, depth, 16, kk => best.V = k.Max(best.V, k.Abs(load(line, kk)))));
                staged[offset + line * 16 + lane] = best.V;
            }),
            () => k.For(width >= Edge ? tid & (Edge - 1) : tid, k.Int(Edge), Math.Min(width, Edge), line =>
            {
                var part = width >= Edge ? tid >> EdgeShift : k.Int(0);
                var best = k.Local(0f);
                k.If(lineBase + line < lineLimit, () => k.For(part, depth, stridedParts, kk => best.V = k.Max(best.V, k.Abs(load(line, kk)))));
                staged[offset + line * 16 + part] = best.V;
            }));

        // … and from the parts each line's shift: shifts[shiftOffset + line] (as int bits).
        void Shifts(Val? contiguousFlag, bool contiguousKnown, int offset, int shiftOffset) => k.For(tid, k.Int(Edge), width, line =>
        {
            var parts = contiguousFlag is { } f ? k.Select(f, k.Int(16), k.Int(stridedParts)) : k.Int(contiguousKnown ? 16 : stridedParts);
            var best = k.Local(0f);
            k.For(k.Int(0), parts, 1, j => best.V = k.Max(best.V, staged[offset + line * 16 + j]));
            var exponent = (best.V.AsInt() >> 23) & 0xFF;                // 0 for zeros, 255 for infinities
            shifts[shiftOffset + line] = k.Clamp(141 - exponent, k.Int(-126), k.Int(126)).AsFloat();
        });

        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            Val? aContiguous = ta is { } flagA ? !flagA : null;
            Maxima(aContiguous, true, 0, rowBase, m, (line, kk) => LoadA(aBase, rowBase + line, kk));
            Maxima(tb, false, Edge * 16, colBase, n, (line, kk) => LoadB(bBase, kk, colBase + line));
            k.Barrier();
            Shifts(aContiguous, true, 0, 0);
            Shifts(tb, false, Edge * 16, Edge);
            k.Barrier();

            var rounds = (4 + k.NumSubgroups - 1) / k.NumSubgroups;
            k.For(k.Int(0), rounds, 1, round =>
            {
                var sub = round * k.NumSubgroups + k.SubgroupId;
                var active = sub < 4;
                var at = k.Min(sub, k.Int(3));
                var (subRow, subCol) = ((at >> 1) * 32, (at & 1) * 32);
                foreach (var accumulator in accumulators)
                {
                    ops.Zero(accumulator);
                }

                k.For(k.Int(0), depth, Depth, t0 =>
                {
                    // op(a)[64 × D]: consecutive invocations along k (not transposed) or along m (transposed).
                    k.For(tid, k.Int(Edge * Depth), width, l =>
                    {
                        Val r = l >> DepthShift, kk = l & (Depth - 1);
                        if (ta is { } transposedA)
                        {
                            (r, kk) = (k.Select(transposedA, l & (Edge - 1), r), k.Select(transposedA, l >> EdgeShift, kk));
                        }

                        var (row, kg) = (rowBase + r, t0 + kk);
                        var value = k.Local(0f);
                        k.If((row < m) & (kg < depth), () => value.V = LoadA(aBase, row, kg));
                        var scaled = value.V * Power(shifts[r].AsInt());
                        var high = High(scaled);
                        aHigh[r * StrideA + kk] = high;
                        aLow[r * StrideA + kk] = k.Select(k.IsInf(scaled), k.Float(0f), scaled - high);
                    });

                    // op(b)[D × 64]: consecutive invocations along n (not transposed, packed) or along k (transposed).
                    k.For(tid, k.Int(Depth * Edge), width, l =>
                    {
                        Val kk = l >> EdgeShift, cc = l & (Edge - 1);
                        if (tb is { } transposedB)
                        {
                            (kk, cc) = (k.Select(transposedB, l & (Depth - 1), kk), k.Select(transposedB, l >> DepthShift, cc));
                        }

                        var (col, kg) = (colBase + cc, t0 + kk);
                        var value = k.Local(0f);
                        k.If((col < n) & (kg < depth), () => value.V = LoadB(bBase, kg, col));
                        var scaled = value.V * Power(shifts[Edge + cc].AsInt());
                        var high = High(scaled);
                        bHigh[kk * StrideB + cc] = high;
                        if (bLow is not null)
                        {
                            bLow[kk * StrideB + cc] = k.Select(k.IsInf(scaled), k.Float(0f), scaled - high);
                        }
                    });

                    k.Barrier();
                    k.If(active, () =>
                    {
                        for (int kq = 0; kq < Depth; kq += K)
                        {
                            var aHi = new MatrixOps.Operand[tm];
                            var aLo = new MatrixOps.Operand[tm];
                            for (int i = 0; i < tm; i++)
                            {
                                var offset = (subRow + i * M) * StrideA + kq;
                                (aHi[i], aLo[i]) = (ops.LoadA(aHigh, offset, StrideA), ops.LoadA(aLow, offset, StrideA));
                            }

                            var bHi = new MatrixOps.Operand[tn];
                            var bLo = new MatrixOps.Operand?[tn];
                            for (int j = 0; j < tn; j++)
                            {
                                var offset = subCol + (kq * StrideB + j * N);
                                bHi[j] = ops.LoadB(bHigh, offset, StrideB);
                                bLo[j] = bLow is null ? null : ops.LoadB(bLow, offset, StrideB);
                            }

                            for (int i = 0; i < tm; i++)
                            {
                                for (int j = 0; j < tn; j++)
                                {
                                    var accumulator = accumulators[i * tn + j];
                                    ops.MulAdd(accumulator, aLo[i], bHi[j]);
                                    if (bLo[j] is { } low)
                                    {
                                        ops.MulAdd(accumulator, aHi[i], low);
                                    }

                                    ops.MulAdd(accumulator, aHi[i], bHi[j]);
                                }
                            }
                        }
                    });
                    k.Barrier();                                          // the step is read before the next is staged
                });

                k.If(active, () =>
                {
                    for (int i = 0; i < tm; i++)
                    {
                        for (int j = 0; j < tn; j++)
                        {
                            ops.Store(accumulators[i * tn + j], staged, (subRow + i * M) * Edge + subCol + j * N, Edge);
                        }
                    }
                });
            });
            k.Barrier();

            // Step 3: the outputs, consecutive invocations along a row.
            k.For(tid, k.Int(Edge * Edge), width, l =>
            {
                var (r, cc) = (l >> EdgeShift, l & (Edge - 1));
                var (row, col) = (rowBase + r, colBase + cc);
                k.If((row < m) & (col < n), () =>
                {
                    var value = staged[l] * Power(-shifts[r].AsInt()) * Power(-shifts[Edge + cc].AsInt());
                    switch (format)
                    {
                        case null:
                            var o = bi * m * n + row * n + col;
                            k.If(beta.Eq(0), () => c[o] = value, () => c[o] = value + beta * c[o]);
                            break;
                        case PackedFormat.Int8:
                            c[row * n + col] = value * scales![col];
                            break;
                        default:
                            c[row * n + col] = value;
                            break;
                    }
                });
            });
            k.Barrier();                                                  // staged and shifts read before the next batch writes them
        }, k.GroupsZ);
        return k.Build();
    }

    // The matrix operations of the products: cooperative matrices, or (tests, on devices without them) the same
    // operations emulated, each invocation of a subgroup of S (the size the kernel was built for) keeping elements lane,
    // lane + S, … (row-major) of every accumulator and reading its operands from workgroup memory. The emulation runs the
    // rest of the kernel (scaling, splitting, staging, rounds, partial blocks) exactly as the device's kernel does, so a
    // device without matrix units can check it against the CPU; it needs the subgroup size to be S.
    private sealed class MatrixOps
    {
        private readonly KernelBuilder _k;
        private readonly CoopShape _shape;
        private readonly int _emulated;
        private readonly uint _aType, _bType, _accumulatorType;

        public MatrixOps(KernelBuilder k, CoopShape shape, int emulated)
        {
            (_k, _shape, _emulated) = (k, shape, emulated);
            if (emulated == 0)
            {
                _aType = k.MatrixType(shape.M, shape.K, MatrixUse.A);
                _bType = k.MatrixType(shape.K, shape.N, MatrixUse.B);
                _accumulatorType = k.MatrixType(shape.M, shape.N, MatrixUse.Accumulator);
            }
        }

        /// <summary>An accumulator: a local cooperative matrix, or an emulation's elements of this invocation.</summary>
        public sealed class Accumulator(uint local, Var[]? lanes)
        {
            public uint Local => local;

            public Var[]? Lanes => lanes;
        }

        /// <summary>A loaded operand: the matrix (its id), or where an emulation reads it.</summary>
        public readonly record struct Operand(uint Id, HalfArray Source, Val Offset, int Stride);

        public Accumulator NewAccumulator()
        {
            if (_emulated == 0)
            {
                return new Accumulator(_k.MatrixLocal(_accumulatorType), null);
            }

            var lanes = new Var[_shape.M * _shape.N / _emulated];
            for (int e = 0; e < lanes.Length; e++)
            {
                lanes[e] = _k.Local(ScalarKind.Float);
            }

            return new Accumulator(0, lanes);
        }

        public void Zero(Accumulator accumulator)
        {
            if (accumulator.Lanes is { } lanes)
            {
                foreach (var v in lanes)
                {
                    v.V = _k.Float(0f);
                }
            }
            else
            {
                _k.MatrixZero(accumulator.Local, _accumulatorType);
            }
        }

        public Operand LoadA(HalfArray source, Val offset, int stride) =>
            new(_emulated == 0 ? _k.MatrixLoad(_aType, source, offset, stride) : 0, source, offset, stride);

        public Operand LoadB(HalfArray source, Val offset, int stride) =>
            new(_emulated == 0 ? _k.MatrixLoad(_bType, source, offset, stride) : 0, source, offset, stride);

        // accumulator += a · b.
        public void MulAdd(Accumulator accumulator, Operand a, Operand b)
        {
            if (accumulator.Lanes is not { } lanes)
            {
                _k.MatrixMulAdd(accumulator.Local, _accumulatorType, a.Id, b.Id);
                return;
            }

            var (rows, cols) = Elements();
            _k.For(_k.Int(0), _k.Int(_shape.K), 1, kk =>
            {
                for (int e = 0; e < lanes.Length; e++)
                {
                    lanes[e].V = _k.Fma(a.Source[a.Offset + rows[e] * a.Stride + kk], b.Source[b.Offset + kk * b.Stride + cols[e]], lanes[e].V);
                }
            });
        }

        // Stores the accumulator row-major at target[offset + row · stride + col].
        public void Store(Accumulator accumulator, SharedArray target, Val offset, int stride)
        {
            if (accumulator.Lanes is not { } lanes)
            {
                _k.MatrixStore(accumulator.Local, _accumulatorType, target, offset, stride);
                return;
            }

            var (rows, cols) = Elements();
            for (int e = 0; e < lanes.Length; e++)
            {
                target[offset + rows[e] * stride + cols[e]] = lanes[e].V;
            }
        }

        // The (row, column) of each element this invocation keeps in an emulation: element e is lane + S · e, row-major.
        private (Val[] Rows, Val[] Cols) Elements()
        {
            int count = _shape.M * _shape.N / _emulated;
            var (rows, cols) = (new Val[count], new Val[count]);
            for (int e = 0; e < count; e++)
            {
                var index = _k.SubgroupLocalId + _emulated * e;
                (rows[e], cols[e]) = (index / _shape.N, index % _shape.N);
            }

            return (rows, cols);
        }
    }
}
