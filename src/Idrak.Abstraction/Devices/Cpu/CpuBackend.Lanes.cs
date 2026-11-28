// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

// The CPU's described lane for weights stored in the file's layout: y = x · Wᵀ for W stored as n rows of k values of any
// type, read from its description alone (TypeDescription: no name, no code of a type's own), with a scale per row or none.
// Its kernels are written once over IInstructions and run on any CPU by .NET's portable vector types
// (PortableInstructions); the instruction sets only some CPUs have (x86's AVX2 and FMA, AVX-VNNI; a 64-bit Arm CPU's NEON
// and its dot products) give the lane extra variants where the CPU has them, with the same bits, the fastest at each shape
// measured once on the real operands (Chosen).
internal sealed partial class CpuBackend
{
    // A stored weight's words, k, the bytes of a row and where the rows' read part starts in each (bytes), and a row's
    // scale where the lane reads the codes alone (RowCodesOf: each output times its row's scale, read from the row's
    // block as its number type reads it): a value made on every product, on the stack (nothing allocated a call).
    private readonly record struct StoredLane(float[] Words, int Length, int K, long RowBytes, int Offset = 0, BlockScale? RowScale = null);

    // The bytes of a row of k values of a description (whole groups of blocks).
    private static long RowBytesOf(TypeDescription d, int k) => (long)k / d.UnitValues * d.UnitBytes;

    /// <summary>
    /// The product of a stored weight by the described lane, in the CPU's own memory. False when the weight is not such a
    /// one, or the described lane does not read its description at k.
    /// </summary>
    private bool StoredRouted(Storage x, Storage weight, Storage y, int m, int n, int k)
    {
        if (weight.Description is not { } description || weight is MappedStorage || m == 0 || n == 0 || k == 0)
        {
            return false;
        }

        var lane = new StoredLane(D(weight), weight.Length, k, RowBytesOf(description, k));
        if (!DescribedProduct(description, lane, D(x), D(y), m, n, k) && !RowScaleProduct(description, weight, D(x), D(y), m, n, k))
        {
            return false;
        }

        s_describedMultiplied = true;
        return true;
    }

    // The codes alone of a description whose row of k values is one block under one scale covering it (its codes one plane
    // of whole bytes within the block, no minimum, sum or codebook), or null where it is not one. Made once and held by the
    // description itself (TypeDescription.CpuRowCodes, by CodesAlone), with its own plan.
    private static TypeDescription? RowCodesOf(TypeDescription d, int k)
    {
        if (d is not { Values: var values, Group: 1, Mins: null or [], Sums: null or [], Scales: [{ Pieces: null } scale], Planes: [{ Group: 1, Stride: 0, Stream: false, Places: null, Radix: 0 } plane] }
            || values != k || scale.Values != values || scale.Type.Width != 1 || d.Code.Width != 1 || plane.Bits % 8 != 0
            || (long)plane.Offset + (long)k * (plane.Bits / 8) > d.Bytes)
        {
            return null;
        }

        var codes = d.CpuRowCodes;
        return codes.Valid ? codes : null;
    }

    // The codes' description of a row-scaled description (RowCodesOf: one plane of whole bytes a value, no scale).
    internal static TypeDescription CodesAlone(TypeDescription r) => new(1, r.Planes[0].Bits / 8, r.Code, [r.Planes[0] with { Offset = 0 }], r.CodeOffset);

    /// <summary>
    /// How the described lane multiplies rows of <paramref name="k"/> values of <paramref name="description"/>: its arithmetic
    /// ("integer", "float32", "float64"), that over the codes alone with each row's scale after ("... over the codes, the row's
    /// scale after"), or null where it does not (the default product serves). For tests.
    /// </summary>
    internal static string? DescribedRoute(TypeDescription description, int k) =>
        DescribedArithmetic(description, k) ?? (RowCodesOf(description, k) is { } codes && DescribedArithmetic(codes, k) is { } a ? a + " over the codes, the row's scale after" : null);

    // A row a block whose one scale covers it (its codes one plane of whole bytes, no minimum, no sum, no codebook): the
    // scale factors out of the row's sum, y[i, j] = scale_j · Σ x[i, p] · code[j, p]. Where the lane has no plan for the
    // description itself (16-bit codes under a scale), it takes the plan of the codes alone (the same code, no scale) over
    // each row's codes where they lie, then multiplies each output by its row's scale read through its number type.
    private static bool RowScaleProduct(TypeDescription d, Storage weight, float[] x, float[] y, int m, int n, int k)
    {
        if (RowCodesOf(d, k) is not { } codes)
        {
            return false;
        }

        return DescribedProduct(codes, RowCodesLane(d, D(weight), weight.Length, k), x, y, m, n, k);
    }

    // The lane of a row-scaled description's codes where they lie in each row's block, each output times its row's scale
    // (FormatColumns, a tile's columns once their sums are made: the same product, a float32 multiply of each sum by its
    // row's scale, whichever thread makes it).
    private static StoredLane RowCodesLane(TypeDescription d, float[] words, int length, int k) => new(words, length, k, d.Bytes, d.Planes[0].Offset, d.Scales![0]);

    // Row c's scale (a row's block at `block`) as its number type reads it: a float32 as it is stored, any other through
    // its type (the same number).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float RowScaleAt(BlockScale scale, byte* block, int bytes) =>
        scale is { Pieces: null, Bias: 0d, Type: { Kind: NumberKind.Float, Bits: 32, Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Specials: FloatSpecials.Ieee, Subnormals: true, Width: 1 } }
            ? Unsafe.ReadUnaligned<float>(block + scale.Offset)
            : (float)scale.Value(new ReadOnlySpan<byte>(block, bytes), 0);

    // Whether an instruction set shifts lanes by their own counts in one instruction (IInstructions.ShiftsByLane).
    private static bool ShiftsLanes(Instructions with) => with switch
    {
        Instructions.Avx2 => Avx2Instructions.ShiftsByLane,
        Instructions.Vnni => VnniInstructions.ShiftsByLane,
        Instructions.Neon => NeonInstructions.ShiftsByLane,
        Instructions.DotProd => DotProdInstructions.ShiftsByLane,
        Instructions.DotProdWide => WideDotProdInstructions.ShiftsByLane,
        _ => PortableInstructions.ShiftsByLane,
    };

    // Bit deposit and extract bit by bit (IInstructions.Deposit and Extract where no instruction does them): the same values.
    private static ulong BitsDeposited(ulong value, ulong mask)
    {
        ulong result = 0;
        for (ulong bit = 1; mask != 0; mask &= mask - 1, bit <<= 1)
        {
            result |= (value & bit) != 0 ? mask & (~mask + 1) : 0;
        }

        return result;
    }

    private static ulong BitsExtracted(ulong value, ulong mask)
    {
        ulong result = 0;
        for (ulong bit = 1; mask != 0; mask &= mask - 1, bit <<= 1)
        {
            result |= (value & mask & (~mask + 1)) != 0 ? bit : 0;
        }

        return result;
    }

    // The described lane's kind of lane (a lane id's low 4 bits: LaneOf).
    private const int DescribedLaneId = 11;

    // The described lane whose one-row kernels make a codebook's indices in general registers by bit extract and deposit
    // (DescribedPlan.DepositSteps: x86's BMI2) and read its entries a qword each, a candidate for a row of x alone where the
    // plan's steps take it and the CPU has the instructions; its other kernels the described lane's own (the same bits).
    private const int DepositLaneId = 13;

    // The float product's kind of lane, a candidate for float32 rows (DescribedPlan.FloatRows): x by W as the float32 product
    // takes a B transposed (its tiles packing W's rows as it reads them; ProductOverRows), measured against the described
    // lane's at each shape, the faster kept (its sums in their own order, so its own bits).
    private const int ProductLaneId = 12;

    // The product of x by float32 rows (W [n, k] where they lie, its rows `lane.RowBytes` apart) by the float32 product.
    private static unsafe void ProductOverRows(StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        fixed (float* a = x, w = lane.Words, c = y)
        {
            Product(a, (float*)((byte*)w + lane.Offset), null, c, 1, m, n, k, false, true, 0f, ldb: lane.RowBytes / 4);
        }
    }

    // The described lane's name in the recorded choices.
    private const string DescribedLaneName = "described lane (any type read from its description alone, its values decoded in registers inside the dot products)";

    // ------------------------------------------------------------------ the instructions a lane is compiled for

    // Every lane's kernels are written once over IInstructions, the operations of their decoding and dot products that
    // instruction sets do differently, and compiled for each set they run with: PortableInstructions, .NET's portable
    // vector types alone (Vector256 and Vector128, each operation the CPU's own instructions where .NET maps it to them:
    // any CPU), which every lane of every format takes on every CPU; and the instruction sets only some CPUs have, each an
    // extra candidate of the measured choice where the CPU has them (x86's AVX2 and FMA: Avx2Instructions, the int8 dot
    // products by its multiply-adds; with AVX-VNNI's dot products: VnniInstructions; a 64-bit Arm CPU's NEON, which every
    // one has: NeonInstructions; with its dot products, FEAT_DotProd: DotProdInstructions, and for codes past 127
    // WideDotProdInstructions). Each gives the bits the others give (the same operations in the same order: integer sums
    // exact, every multiply-add fused, float sums in the lanes' fixed order), so a shape's measured choice among them
    // changes its speed alone, and a CPU that has none of them runs every lane, with the same results, by the portable
    // instructions. Each lane by the portable instructions, NEON or Arm's dot products has a 128-bit twin (NarrowLane),
    // its four-row kernels on 128-bit vectors by the sets' 128-bit forms, another candidate with the same bits.

    /// <summary>The instruction set a lane's kernels are compiled for (a lane id's bits 4-6: LaneOf).</summary>
    private enum Instructions : byte
    {
        Portable,
        Avx2,
        Vnni,
        Neon,
        DotProd,
        DotProdWide,
    }

    // Whether the CPU has the instruction sets of the extra candidates (as the runtime reports them: its DOTNET_Enable*
    // settings turn them off): x86's AVX2 and FMA; AVX-VNNI with them; a 64-bit Arm CPU's NEON (AdvSimd: off only with
    // every intrinsic, DOTNET_EnableHWIntrinsic=0) and its dot products (FEAT_DotProd, DOTNET_EnableArm64Dp). The dispatch
    // to a lane's kernels tests AdvSimd.Arm64.IsSupported first, a constant to the JIT, so an x86 process compiles none of
    // the Arm branches and an Arm process none of the x86 ones.
    private static readonly bool s_avx2Lanes = Avx2.IsSupported && Fma.IsSupported, s_vnniLanes = s_avx2Lanes && AvxVnni.IsSupported,
        s_neonLanes = AdvSimd.Arm64.IsSupported, s_dotProdLanes = s_neonLanes && Dp.IsSupported;

    // A lane id: its kind (DescribedLaneId) in the low 4 bits, its instruction set in the next 3, and in the
    // top bit whether its four-row kernels run on 128-bit vectors (NarrowLane: the lane's 128-bit twin).
    private static int LaneOf(int kind, Instructions with, bool narrow = false) => kind | (int)with << 4 | (narrow ? NarrowLane : 0);

    // The lane id's bit of a 128-bit twin: the same kind, instructions and one-row kernels, its four-row kernels written on
    // 128-bit vectors (four rows of x sharing each stored row's decode where 256-bit vectors are not accelerated too).
    private const int NarrowLane = 0x80;

    private static int KindOf(int lane) => lane & 15;

    private static Instructions InstructionsOf(int lane) => (Instructions)(lane >> 4 & 7);

    private static bool NarrowOf(int lane) => (lane & NarrowLane) != 0;

    // A lane's name: its kind's, the instruction set where it is not the portable one, and a 128-bit twin's vectors.
    private static string LaneName(int lane) => KindOf(lane) == ProductLaneId ? "the float32 product (W's rows packed as it reads them)" : (InstructionsOf(lane) switch
    {
        Instructions.Avx2 => DescribedLaneName + " by AVX2 and FMA",
        Instructions.Vnni => DescribedLaneName + " by AVX-VNNI",
        Instructions.Neon => DescribedLaneName + " by NEON",
        Instructions.DotProd => DescribedLaneName + " by Arm's dot products (DotProd)",
        Instructions.DotProdWide => DescribedLaneName + " by Arm's dot products, two a step for codes past 127 (DotProd)",
        _ => DescribedLaneName,
    }) + (NarrowOf(lane) ? ", four rows of x at a time on 128-bit vectors" : "") + (KindOf(lane) == DepositLaneId ? ", a row's codebook indices by bit deposit (BMI2)" : "");

    // A lane's short name in a measurement's times (its kind's first word, its instruction set's, and /128 for a 128-bit twin).
    private static string LaneShortName(int lane) => KindOf(lane) == ProductLaneId ? "product" : DescribedLaneName.Split(' ')[0] + InstructionsOf(lane) switch
    {
        Instructions.Avx2 => "/avx2",
        Instructions.Vnni => "/avx-vnni",
        Instructions.Neon => "/neon",
        Instructions.DotProd => "/dotprod",
        Instructions.DotProdWide => "/dotprod2",
        _ => "",
    } + (NarrowOf(lane) ? "/128" : "") + (KindOf(lane) == DepositLaneId ? "/deposit" : "");

    /// <summary>
    /// A format's candidate lanes that run on this CPU (up to 16, a byte each), the first the one the others are timed
    /// against: each kind of lane by the portable instructions, then by each instruction set the CPU has that the kind takes,
    /// then its 128-bit twins (by the portable instructions, and by each of Arm's sets the CPU has that the kind takes).
    /// </summary>
    private readonly struct LaneList
    {
        private readonly UInt128 lanes;

        private LaneList(UInt128 lanes, int count)
        {
            this.lanes = lanes;
            Count = count;
        }

        public int Count { get; }

        public int this[int index] => (byte)(lanes >> (8 * index));

        public LaneList With(int lane) => new(lanes | (UInt128)(uint)lane << (8 * Count), Count + 1);

        /// <summary>
        /// A kind of lane by the portable instructions, then by AVX2 and FMA, and (int8 dot products) by AVX-VNNI where the CPU
        /// has them, or by NEON, and (int8 dot products) by Arm's dot products where the CPU has those;
        /// <paramref name="pairs"/>: the kind's int8 dot products take AVX2's multiply-adds (the codes at most 127, so no
        /// pair of products saturates), so its AVX2 lane is one, as are its NEON lane (16-bit products of signed bytes) and
        /// its DotProd lane (Arm's signed dot products of its codes as they are; else two a step: DotProdWide). Then the
        /// kind's 128-bit twins (NarrowLane): by the portable instructions on any CPU, and by NEON and Arm's dot products as
        /// the kind takes them where the CPU has them. The vector paths allowed (ComputeResources.CpuVectors,
        /// IDRAK_CPU_VECTORS) narrow them: native, the platform's intrinsics alone (the portable lanes only for a kind with
        /// none here); portable, the portable lane and its 128-bit twin; scalar, the portable lane alone.
        /// </summary>
        public LaneList WithKind(int kind, bool dots = false, bool pairs = true)
        {
            var paths = ComputeResources.CpuVectors;
            bool intrinsics = paths is CpuVectorPaths.Automatic or CpuVectorPaths.Native;
            bool avx2 = intrinsics && s_avx2Lanes && (!dots || pairs), neon = intrinsics && s_neonLanes && (!dots || pairs),
                dotProd = intrinsics && dots && s_dotProdLanes, vnni = intrinsics && dots && s_vnniLanes;
            bool portable = paths != CpuVectorPaths.Native || !(avx2 || neon || dotProd || vnni);
            var list = this;
            if (portable)
            {
                list = list.With(LaneOf(kind, Instructions.Portable));
            }

            if (avx2)
            {
                list = list.With(LaneOf(kind, Instructions.Avx2));
            }

            if (neon)
            {
                list = list.With(LaneOf(kind, Instructions.Neon));
            }

            if (dotProd)
            {
                list = list.With(LaneOf(kind, pairs ? Instructions.DotProd : Instructions.DotProdWide));
            }

            if (vnni)
            {
                list = list.With(LaneOf(kind, Instructions.Vnni));
            }

            if (portable && paths != CpuVectorPaths.Scalar)
            {
                list = list.With(LaneOf(kind, Instructions.Portable, narrow: true));
            }

            if (neon)
            {
                list = list.With(LaneOf(kind, Instructions.Neon, narrow: true));
            }

            return dotProd ? list.With(LaneOf(kind, pairs ? Instructions.DotProd : Instructions.DotProdWide, narrow: true)) : list;
        }

        /// <summary>
        /// The lanes a product of <paramref name="m"/> rows of x takes as candidates: all of them from 4 rows, where its four
        /// rows of x run together; below that the 256-bit lanes alone (a 128-bit twin's kernels for fewer rows are its 256-bit
        /// lane's own).
        /// </summary>
        public LaneList ForRows(int m)
        {
            var kept = default(LaneList);
            for (int i = 0; i < Count; i++)
            {
                // (A 128-bit twin from 4 rows; a lane whose one-row kernels alone differ, DepositLaneId's, for one row.)
                if (!(NarrowOf(this[i]) && m < 4 || KindOf(this[i]) == DepositLaneId && m != 1))
                {
                    kept = kept.With(this[i]);
                }
            }

            return kept;
        }

        /// <summary>
        /// The described lane's deposit lanes (DepositLaneId) by AVX2 and FMA and by AVX-VNNI where the CPU has them with
        /// BMI2 and the plan's one-row steps take them (<paramref name="deposits"/>), as WithKind lists its own.
        /// </summary>
        public LaneList WithDeposits(bool deposits, bool pairs)
        {
            bool intrinsics = ComputeResources.CpuVectors is CpuVectorPaths.Automatic or CpuVectorPaths.Native;
            if (!(deposits && intrinsics && Bmi2.X64.IsSupported))
            {
                return this;
            }

            var list = this;
            if (s_avx2Lanes && pairs)
            {
                list = list.With(LaneOf(DepositLaneId, Instructions.Avx2));
            }

            return s_vnniLanes ? list.With(LaneOf(DepositLaneId, Instructions.Vnni)) : list;
        }

        public bool Contains(int lane)
        {
            for (int i = 0; i < Count; i++)
            {
                if (this[i] == lane)
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>
    /// The operations of the lanes' kernels that instruction sets do differently, each giving the same bits whichever set
    /// does it (the lanes' kernels are generic over it: PortableInstructions, Avx2Instructions, VnniInstructions,
    /// NeonInstructions, DotProdInstructions, WideDotProdInstructions).
    /// </summary>
    internal unsafe interface IInstructions
    {
        /// <summary>
        /// <paramref name="sum"/> + in each 32-bit lane the dot product of its 4 unsigned bytes of <paramref name="codes"/>
        /// by its 4 signed bytes of <paramref name="x"/> (x's within ±127), exactly (codes of a kind that takes AVX2's
        /// multiply-adds at most 127: LaneList.WithKind).
        /// </summary>
        static abstract Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x);

        /// <summary>a · b + c, rounded once.</summary>
        static abstract Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c);

        /// <summary>a · b + c, rounded once.</summary>
        static abstract Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c);

        /// <summary>Each 128-bit half's low two 32-bit lanes of a and b, interleaved (a0, b0, a1, b1 | a4, b4, a5, b5).</summary>
        static abstract Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b);

        /// <summary>Each 128-bit half's high two 32-bit lanes of a and b, interleaved (a2, b2, a3, b3 | a6, b6, a7, b7).</summary>
        static abstract Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b);

        /// <summary>Each 128-bit half's low 64-bit lanes of a and b (a0, b0 | a2, b2).</summary>
        static abstract Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b);

        /// <summary>Each 128-bit half's high 64-bit lanes of a and b (a1, b1 | a3, b3).</summary>
        static abstract Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b);

        /// <summary>The 128-bit halves swapped.</summary>
        static abstract Vector256<uint> Swap(Vector256<uint> v);

        /// <summary>a's low 128 bits, then b's.</summary>
        static abstract Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b);

        /// <summary>a's high 128 bits, then b's.</summary>
        static abstract Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b);

        /// <summary>Each 32-bit lane shifted right (zeros in) by its count (below 32).</summary>
        static abstract Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts);

        /// <summary>Each 32-bit lane shifted left by its count (below 32).</summary>
        static abstract Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts);

        /// <summary>Each 64-bit lane shifted right (zeros in) by its count (below 64).</summary>
        static abstract Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts);

        /// <summary>The lower 128-bit half as it is and the upper shifted right by <paramref name="count"/>, each 32-bit lane.</summary>
        static abstract Vector256<uint> ShiftUpper(Vector256<uint> v, int count);

        /// <summary>The dword at <paramref name="at"/> + offsets[l] in lane l.</summary>
        static abstract Vector256<int> Gather(byte* at, int* offsets);

        /// <summary>The dword at <paramref name="at"/> + offsets[l] in the lanes l below <paramref name="count"/> (0 elsewhere, nothing read there).</summary>
        static abstract Vector256<int> Gather(byte* at, int* offsets, int count);

        /// <summary>table[indices[l]] in lane l.</summary>
        static abstract Vector256<float> Gather(float* table, Vector256<int> indices);

        /// <summary>The dword at <paramref name="at"/> + offsets[l] in lane l (the offsets in a register).</summary>
        static abstract Vector256<int> Gather(byte* at, Vector256<int> offsets);

        /// <summary>
        /// Each 128-bit half's bytes picked by its own bytes of <paramref name="control"/> from its own 16 of
        /// <paramref name="bytes"/>: a control byte below 16 picks that byte, one with its top bit set gives 0 (none other
        /// is passed).
        /// </summary>
        static abstract Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control);

        /// <summary>The 8 bytes at <paramref name="at"/>, each a 32-bit lane (zero-extended).</summary>
        static abstract Vector256<int> Widen(byte* at);

        /// <summary>The 4 float32 values at <paramref name="at"/> as float64, exactly.</summary>
        static abstract Vector256<double> Doubles(float* at);

        /// <summary>The 16 bytes at <paramref name="at"/> in both 128-bit halves.</summary>
        static abstract Vector256<byte> Halves(byte* at);

        /// <summary>The 8 values at <paramref name="at"/>, each a 32-bit lane (zero-extended).</summary>
        static abstract Vector256<int> Widen(ushort* at);

        /// <summary>The 8 values at <paramref name="at"/>, each a 32-bit lane (sign-extended).</summary>
        static abstract Vector256<int> Widen(short* at);

        /// <summary>The low 8 bytes, each a 32-bit lane (sign-extended).</summary>
        static abstract Vector256<int> Widen(Vector128<sbyte> v);

        /// <summary>The low 8 bytes, each a 32-bit lane (sign- or zero-extended as <paramref name="signed"/> says).</summary>
        static abstract Vector256<int> Widen(Vector128<byte> v, bool signed);

        /// <summary>
        /// Each 128-bit half's bytes picked from its own 16 by <paramref name="control"/> (a constant: the lower half's bytes
        /// 0-15, the upper half's 16-31, each picking within its own half).
        /// </summary>
        static abstract Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control);

        /// <summary>Lane <paramref name="lane"/> of v in every lane.</summary>
        static abstract Vector256<float> Broadcast(Vector256<float> v, int lane);

        /// <summary>v[indices[l]] in lane l (indices 0-7, of which an instruction set may read the low 3 bits alone: Pairs).</summary>
        static abstract Vector256<float> Permute(Vector256<float> v, Vector256<int> indices);

        /// <summary>v[indices[l]] in lane l (indices 0-7).</summary>
        static abstract Vector256<int> Permute(Vector256<int> v, Vector256<int> indices);

        /// <summary>
        /// Lane l: <paramref name="word"/>'s bits first + 2l, first + 2l + 1 and first + 2l + 2 in its low 3 (the lane's
        /// others 0, or bits this instruction set's Permute does not read); <paramref name="first"/> 0 or 16.
        /// </summary>
        static abstract Vector256<uint> Pairs(Vector256<uint> word, int first);

        /// <summary>Whether lanes shift by their own counts in one instruction (else IRunLoad shifts by the counts as numbers).</summary>
        static abstract bool ShiftsByLane { get; }

        /// <summary>
        /// Whether the integer lane's tile dots several rows of x with a group's held steps in one pass of its loop (four, two
        /// on 128-bit vectors: IntegerTile): where a step's dot product is the set's own instructions. The portable
        /// instructions split each step into its even and odd bytes first, and with four rows in a pass the JIT held those
        /// halves across the rows and spilled them (measured: 423-501 vector stack accesses in the loop of q8_0's, q4_k's and
        /// tq1_0's portable tiles, none a row at a time), so they take a row a pass.
        /// </summary>
        static abstract bool RowsTogether { get; }

        /// <summary>Each 16-bit lane's unsigned product a · b, its high 16 bits.</summary>
        static abstract Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b);

        /// <summary>Whether MultiplyHigh is one instruction (else a digit plane is read by low multiplies and shifts alone).</summary>
        static abstract bool MultipliesHigh { get; }

        /// <summary>The qword table[indices[2j]] in 64-bit lane j (each pair of 32-bit lanes' first index; indices non-negative).</summary>
        static abstract Vector256<long> GatherPairs(long* table, Vector256<int> indices);

        /// <summary>
        /// The low bits of <paramref name="value"/> put in the set bits of <paramref name="mask"/>, in order (x86's bit deposit,
        /// BMI2's PDEP, where the instruction set has it; else bit by bit, the same value).
        /// </summary>
        static abstract ulong Deposit(ulong value, ulong mask);

        /// <summary>The bits of <paramref name="value"/> under the set bits of <paramref name="mask"/>, in order, as the low bits (BMI2's PEXT; else bit by bit).</summary>
        static abstract ulong Extract(ulong value, ulong mask);

        /// <summary>The cache line at <paramref name="at"/> asked for (a hint, where the instruction set has one; it never faults).</summary>
        static abstract void Prefetch(void* at);

        // The 128-bit forms the 128-bit twins' four-row kernels take (lanes 0-3 of a 256-bit lane set in one vector, 4-7 in
        // another: each the 256-bit operation's on its half, so the same bits).

        /// <summary>Dot over 128 bits: <paramref name="sum"/> + in each 32-bit lane its 4 bytes of codes by x's 4, exactly.</summary>
        static abstract Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x);

        /// <summary>a · b + c over 128 bits, rounded once.</summary>
        static abstract Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c);

        /// <summary>a · b + c over 128 bits, rounded once.</summary>
        static abstract Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c);

        /// <summary>The low two 32-bit lanes of a and b, interleaved (a0, b0, a1, b1): UnpackLow over one 128-bit half.</summary>
        static abstract Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b);

        /// <summary>The high two 32-bit lanes of a and b, interleaved (a2, b2, a3, b3): UnpackHigh over one 128-bit half.</summary>
        static abstract Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b);

        /// <summary>The low 64-bit lanes of a and b (a0, b0): UnpackLow over one 128-bit half.</summary>
        static abstract Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b);

        /// <summary>The high 64-bit lanes of a and b (a1, b1): UnpackHigh over one 128-bit half.</summary>
        static abstract Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b);
    }

    /// <summary>
    /// The lanes' operations by .NET's portable vector types alone (any CPU): 256-bit operations where .NET reports them
    /// accelerated, else their 128-bit halves; int8 dot products by 16-bit products of even and odd bytes; multiply-adds
    /// fused by the CPU where FusedMultiplyAdd measures faster, else exactly from separate operations (ExactFused).
    /// </summary>
    private unsafe struct PortableInstructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x)
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                return Vector256.Create(Dot(sum.GetLower(), codes.GetLower(), x.GetLower()), Dot(sum.GetUpper(), codes.GetUpper(), x.GetUpper()));
            }

            // Even and odd bytes as 16-bit lanes (codes zero-extended, x sign-extended), each product exact in 16 bits (at most
            // 255 · 128), each 32-bit lane's two of each added as 32-bit integers.
            var c = codes.AsInt16();
            var v = x.AsInt16();
            var even = ((c & Vector256.Create((short)0xFF)) * ((v << 8) >> 8)).AsInt32();
            var odd = ((c >>> 8) * (v >> 8)).AsInt32();
            return sum + (((even << 16) >> 16) + (even >> 16))
                + (((odd << 16) >> 16) + (odd >> 16));
        }

        // Dot over 128 bits (a half of a 256-bit lane set where those are not accelerated, and the 128-bit twins': the same
        // operations, lane by lane). The shifts by their operators, which the JIT takes as its own instructions however much
        // it has inlined into a kernel (measured: Vector128.ShiftLeft and the shifts right left as calls, 33 in q4_k's
        // 128-bit integer kernel by the portable instructions on 64-bit Arm).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x)
        {
            var c = codes.AsInt16();
            var v = x.AsInt16();
            var even = ((c & Vector128.Create((short)0xFF)) * ((v << 8) >> 8)).AsInt32();
            var odd = ((c >>> 8) * (v >> 8)).AsInt32();
            return sum + (((even << 16) >> 16) + (even >> 16)) + (((odd << 16) >> 16) + (odd >> 16));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) =>
            FusedInHardware.Single ? Vector256.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) =>
            FusedInHardware.Double ? Vector256.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                var low = Vector256.Create(0u, 0, 1, 1, 4, 4, 5, 5);
                return Vector256.ConditionalSelect(Vector256.Create(0u, ~0u, 0u, ~0u, 0u, ~0u, 0u, ~0u), Vector256.Shuffle(b, low), Vector256.Shuffle(a, low));
            }

            return Vector256.Create(Interleave(a.GetLower(), b.GetLower(), 0), Interleave(a.GetUpper(), b.GetUpper(), 0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                var high = Vector256.Create(2u, 2, 3, 3, 6, 6, 7, 7);
                return Vector256.ConditionalSelect(Vector256.Create(0u, ~0u, 0u, ~0u, 0u, ~0u, 0u, ~0u), Vector256.Shuffle(b, high), Vector256.Shuffle(a, high));
            }

            return Vector256.Create(Interleave(a.GetLower(), b.GetLower(), 2), Interleave(a.GetUpper(), b.GetUpper(), 2));
        }

        // x's and y's 32-bit lanes from `from` and from + 1, interleaved.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<uint> Interleave(Vector128<uint> x, Vector128<uint> y, uint from)
        {
            var pick = Vector128.Create(from, from, from + 1, from + 1);
            return Vector128.ConditionalSelect(Vector128.Create(0u, ~0u, 0u, ~0u), Vector128.Shuffle(y, pick), Vector128.Shuffle(x, pick));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                var low = Vector256.Create(0ul, 0, 2, 2);
                return Vector256.ConditionalSelect(Vector256.Create(0ul, ~0ul, 0ul, ~0ul), Vector256.Shuffle(b, low), Vector256.Shuffle(a, low));
            }

            return Vector256.Create(Vector128.Create(a.GetElement(0), b.GetElement(0)), Vector128.Create(a.GetElement(2), b.GetElement(2)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                var high = Vector256.Create(1ul, 1, 3, 3);
                return Vector256.ConditionalSelect(Vector256.Create(0ul, ~0ul, 0ul, ~0ul), Vector256.Shuffle(b, high), Vector256.Shuffle(a, high));
            }

            return Vector256.Create(Vector128.Create(a.GetElement(1), b.GetElement(1)), Vector128.Create(a.GetElement(3), b.GetElement(3)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => Vector256.Create(v.GetUpper(), v.GetLower());

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => Vector256.Create(a.GetLower(), b.GetLower());

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => Vector256.Create(a.GetUpper(), b.GetUpper());

        // A shift by each bit of the counts in turn: 1, 2, 4, 8 and 16 places where it is set.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts)
        {
            v = Vector256.ConditionalSelect(Bit(counts, 0), v >>> 1, v);
            v = Vector256.ConditionalSelect(Bit(counts, 1), v >>> 2, v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), v >>> 4, v);
            v = Vector256.ConditionalSelect(Bit(counts, 3), v >>> 8, v);
            return Vector256.ConditionalSelect(Bit(counts, 4), v >>> 16, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts)
        {
            v = Vector256.ConditionalSelect(Bit(counts, 0), v << 1, v);
            v = Vector256.ConditionalSelect(Bit(counts, 1), v << 2, v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), v << 4, v);
            v = Vector256.ConditionalSelect(Bit(counts, 3), v << 8, v);
            return Vector256.ConditionalSelect(Bit(counts, 4), v << 16, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts)
        {
            v = Vector256.ConditionalSelect(Bit(counts, 1), v >>> 1, v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), v >>> 2, v);
            v = Vector256.ConditionalSelect(Bit(counts, 4), v >>> 4, v);
            v = Vector256.ConditionalSelect(Bit(counts, 8), v >>> 8, v);
            v = Vector256.ConditionalSelect(Bit(counts, 16), v >>> 16, v);
            return Vector256.ConditionalSelect(Bit(counts, 32), v >>> 32, v);
        }

        // All bits set in the lanes whose count has bit `b` set: that bit shifted to the top and copied down by an arithmetic
        // shift, two operations and no constant (a mask and compare by each bit's constant held five registers across a
        // kernel that shifts by lanes).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> Bit(Vector256<uint> counts, int b) => ((counts << (31 - b)).AsInt32() >> 31).AsUInt32();

        // All bits set in the lanes whose count has `bit` set (64-bit lanes: no arithmetic shift of them among the 256-bit
        // instructions an AVX2 CPU has).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<ulong> Bit(Vector256<ulong> counts, ulong bit) => Vector256.Equals(counts & Vector256.Create(bit), Vector256.Create(bit));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) =>
            Vector256.Create(v.GetLower(), Vector128.ShiftRightLogical(v.GetUpper(), count));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => Vector256.Create(
            Unsafe.ReadUnaligned<int>(at + offsets[0]), Unsafe.ReadUnaligned<int>(at + offsets[1]), Unsafe.ReadUnaligned<int>(at + offsets[2]), Unsafe.ReadUnaligned<int>(at + offsets[3]),
            Unsafe.ReadUnaligned<int>(at + offsets[4]), Unsafe.ReadUnaligned<int>(at + offsets[5]), Unsafe.ReadUnaligned<int>(at + offsets[6]), Unsafe.ReadUnaligned<int>(at + offsets[7]));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count)
        {
            if (count >= 8)
            {
                return Gather(at, offsets);
            }

            var v = Vector256<int>.Zero;
            for (int l = 0; l < count; l++)
            {
                v = v.WithElement(l, Unsafe.ReadUnaligned<int>(at + offsets[l]));
            }

            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => Vector256.Create(
            table[indices.GetElement(0)], table[indices.GetElement(1)], table[indices.GetElement(2)], table[indices.GetElement(3)],
            table[indices.GetElement(4)], table[indices.GetElement(5)], table[indices.GetElement(6)], table[indices.GetElement(7)]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => Vector256.Create(
            Unsafe.ReadUnaligned<int>(at + offsets.GetElement(0)), Unsafe.ReadUnaligned<int>(at + offsets.GetElement(1)), Unsafe.ReadUnaligned<int>(at + offsets.GetElement(2)),
            Unsafe.ReadUnaligned<int>(at + offsets.GetElement(3)), Unsafe.ReadUnaligned<int>(at + offsets.GetElement(4)), Unsafe.ReadUnaligned<int>(at + offsets.GetElement(5)),
            Unsafe.ReadUnaligned<int>(at + offsets.GetElement(6)), Unsafe.ReadUnaligned<int>(at + offsets.GetElement(7)));

        // Each half by Vector128.Shuffle (a control byte past 15 gives 0, as the top bit's does on x86's byte shuffle).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) =>
            Vector256.Create(Vector128.Shuffle(bytes.GetLower(), control.GetLower()), Vector128.Shuffle(bytes.GetUpper(), control.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => Widen(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(at)).AsByte(), false);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => FloatsToDoubles(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at)
        {
            var v = Vector128.Load(at);
            return Vector256.Create(v, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at)
        {
            var v = Vector128.Load(at);
            return Vector256.IsHardwareAccelerated
                ? Vector256.WidenLower(v.ToVector256Unsafe()).AsInt32()
                : Vector256.Create(Vector128.WidenLower(v), Vector128.WidenUpper(v)).AsInt32();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at)
        {
            var v = Vector128.Load(at);
            return Vector256.IsHardwareAccelerated
                ? Vector256.WidenLower(v.ToVector256Unsafe())
                : Vector256.Create(Vector128.WidenLower(v), Vector128.WidenUpper(v));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                return Vector256.WidenLower(Vector256.WidenLower(v.ToVector256Unsafe()));
            }

            var shorts = Vector128.WidenLower(v);
            return Vector256.Create(Vector128.WidenLower(shorts), Vector128.WidenUpper(shorts));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed)
        {
            if (signed)
            {
                return Widen(v.AsSByte());
            }

            if (Vector256.IsHardwareAccelerated)
            {
                return Vector256.WidenLower(Vector256.WidenLower(v.ToVector256Unsafe())).AsInt32();
            }

            var shorts = Vector128.WidenLower(v);
            return Vector256.Create(Vector128.WidenLower(shorts), Vector128.WidenUpper(shorts)).AsInt32();
        }

        // A constant shuffle within each half: one byte shuffle where 256-bit vectors are accelerated, else each half's own.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => Vector256.IsHardwareAccelerated
            ? Vector256.Shuffle(bytes, control)
            : Vector256.Create(Vector128.Shuffle(bytes.GetLower(), control.GetLower()), Vector128.Shuffle(bytes.GetUpper(), control.GetUpper() - Vector128.Create((byte)16)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane) => Vector256.Create(v.GetElement(lane));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) =>
            Vector256.IsHardwareAccelerated ? Vector256.ShuffleNative(v, indices) : Vector256.Shuffle(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices) =>
            Vector256.IsHardwareAccelerated ? Vector256.ShuffleNative(v, indices) : Vector256.Shuffle(v, indices);

        // Each lane's 3 bits moved to the top by a multiply (a shift left by 29 − 2l), then 29 places right: 0-7 exactly.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first)
        {
            var w = first == 0 ? word : word >>> first;
            return (w * Vector256.Create(1u << 29, 1u << 27, 1u << 25, 1u << 23, 1u << 21, 1u << 19, 1u << 17, 1u << 15)) >>> 29;
        }

        public static bool ShiftsByLane => false;

        public static bool RowsTogether => false;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => Vector256.Narrow(Vector256.WidenLower(a) * Vector256.WidenLower(b) >>> 16, Vector256.WidenUpper(a) * Vector256.WidenUpper(b) >>> 16);

        public static bool MultipliesHigh => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Vector256.Create(table[indices.GetElement(0)], table[indices.GetElement(2)], table[indices.GetElement(4)], table[indices.GetElement(6)]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => BitsExtracted(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at)
        {
        }

        // Fused as the 256-bit form chooses it (FusedInHardware: whether the CPU fuses in hardware).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) =>
            FusedInHardware.Single ? Vector128.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) =>
            FusedInHardware.Double ? Vector128.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => Interleave(a, b, 0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => Interleave(a, b, 2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => Vector128.Create(a.GetElement(0), b.GetElement(0));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => Vector128.Create(a.GetElement(1), b.GetElement(1));
    }

    /// <summary>The lanes' operations by x86's AVX2 and FMA (the int8 dot products by its multiply-adds: codes at most 127).</summary>
    private unsafe struct Avx2Instructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x) =>
            sum + Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes, x), Vector256.Create((short)1));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b) => Avx2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b) => Avx2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b) => Avx2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b) => Avx2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => Avx2.Permute2x128(v, v, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => Avx2.Permute2x128(a, b, 0x20);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => Avx2.Permute2x128(a, b, 0x31);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts) => Avx2.ShiftRightLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts) => Avx2.ShiftLeftLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts) => Avx2.ShiftRightLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) =>
            Avx2.ShiftRightLogicalVariable(v, Vector256.Create(0u, 0u, 0u, 0u, (uint)count, (uint)count, (uint)count, (uint)count));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => Avx2.GatherVector256((int*)at, Vector256.Load(offsets), 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count) => count >= 8
            ? Avx2.GatherVector256((int*)at, Vector256.Load(offsets), 1)
            : Avx2.GatherMaskVector256(Vector256<int>.Zero, (int*)at, Vector256.Load(offsets), GroupMask(count), 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => Avx2.GatherVector256(table, indices, 4);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => Avx2.GatherVector256((int*)at, offsets, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) => Avx2.Shuffle(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => Avx2.ConvertToVector256Int32(at);

        // The load folded into the conversion.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => Avx.ConvertToVector256Double(Sse.LoadVector128(at));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at) => Avx2.BroadcastVector128ToVector256(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at) => Avx2.ConvertToVector256Int32(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at) => Avx2.ConvertToVector256Int32(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v) => Avx2.ConvertToVector256Int32(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed) => signed ? Avx2.ConvertToVector256Int32(v.AsSByte()) : Avx2.ConvertToVector256Int32(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => Avx2.Shuffle(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane) => Avx2.PermuteVar8x32(v, Vector256.Create(lane));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) => Avx2.PermuteVar8x32(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices) => Avx2.PermuteVar8x32(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first) =>
            Avx2.ShiftRightLogicalVariable(word, Vector256.Create((uint)first, (uint)first + 2, (uint)first + 4, (uint)first + 6, (uint)first + 8, (uint)first + 10, (uint)first + 12, (uint)first + 14));

        public static bool ShiftsByLane => true;

        public static bool RowsTogether => true;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => Avx2.MultiplyHigh(a, b);

        public static bool MultipliesHigh => true;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Avx2.GatherVector256(table, indices.AsInt64() & Vector256.Create(0xFFFFFFFFL), 8);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => Bmi2.X64.IsSupported ? Bmi2.X64.ParallelBitDeposit(value, mask) : BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => Bmi2.X64.IsSupported ? Bmi2.X64.ParallelBitExtract(value, mask) : BitsExtracted(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at) => Sse.Prefetch0(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x) =>
            sum + Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(codes, x), Vector128.Create((short)1));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => Sse2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => Sse2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => Sse2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => Sse2.UnpackHigh(a, b);
    }

    /// <summary>The lanes' operations by AVX2 and FMA, the int8 dot products by AVX-VNNI's (any unsigned codes).</summary>
    private unsafe struct VnniInstructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x) => AvxVnni.MultiplyWideningAndAdd(sum, codes, x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b) => Avx2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b) => Avx2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b) => Avx2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b) => Avx2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => Avx2.Permute2x128(v, v, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => Avx2.Permute2x128(a, b, 0x20);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => Avx2.Permute2x128(a, b, 0x31);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts) => Avx2.ShiftRightLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts) => Avx2.ShiftLeftLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts) => Avx2.ShiftRightLogicalVariable(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) => Avx2Instructions.ShiftUpper(v, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => Avx2Instructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count) => Avx2Instructions.Gather(at, offsets, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => Avx2.GatherVector256(table, indices, 4);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => Avx2.GatherVector256((int*)at, offsets, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) => Avx2.Shuffle(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => Avx2.ConvertToVector256Int32(at);

        // The load folded into the conversion.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => Avx.ConvertToVector256Double(Sse.LoadVector128(at));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at) => Avx2.BroadcastVector128ToVector256(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at) => Avx2.ConvertToVector256Int32(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at) => Avx2.ConvertToVector256Int32(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v) => Avx2.ConvertToVector256Int32(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed) => Avx2Instructions.Widen(v, signed);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => Avx2.Shuffle(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane) => Avx2.PermuteVar8x32(v, Vector256.Create(lane));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) => Avx2.PermuteVar8x32(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices) => Avx2.PermuteVar8x32(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first) => Avx2Instructions.Pairs(word, first);

        public static bool ShiftsByLane => true;

        public static bool RowsTogether => true;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => Avx2.MultiplyHigh(a, b);

        public static bool MultipliesHigh => true;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Avx2.GatherVector256(table, indices.AsInt64() & Vector256.Create(0xFFFFFFFFL), 8);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => Bmi2.X64.IsSupported ? Bmi2.X64.ParallelBitDeposit(value, mask) : BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => Bmi2.X64.IsSupported ? Bmi2.X64.ParallelBitExtract(value, mask) : BitsExtracted(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at) => Sse.Prefetch0(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x) => AvxVnni.MultiplyWideningAndAdd(sum, codes, x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) => Fma.MultiplyAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => Sse2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => Sse2.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => Sse2.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => Sse2.UnpackHigh(a, b);
    }

    /// <summary>
    /// The lanes' operations by a 64-bit Arm CPU's NEON (AdvSimd, which every one has), each 256-bit operation by its two
    /// 128-bit halves: the int8 dot products by widening multiplies and pairwise adds (SMULL, SMULL2, SADDLP, ADDP; codes at
    /// most 127, as AVX2's multiply-adds take them); interleaves by ZIP1 and ZIP2; shifts by each lane's own count by USHL
    /// (right by the negated count); byte shuffles, permutes and broadcasts by table lookups over the two halves' 32 bytes
    /// (TBL, a run-time lane too); multiply-adds fused by FMLA. The rest as the portable instructions do them, which .NET
    /// already makes these instructions on Arm (the widening moves: SXTL, UXTL, FCVTL) or none (halves moved).
    /// </summary>
    private unsafe struct NeonInstructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x) =>
            Vector256.Create(Dot(sum.GetLower(), codes.GetLower(), x.GetLower()), Dot(sum.GetUpper(), codes.GetUpper(), x.GetUpper()));

        // Each byte's product exact in 16 bits (at most 127 · 128: the codes signed bytes as they are), each pair of them
        // widened and added, then each 4 bytes' two pairs.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x)
        {
            var c = codes.AsSByte();
            var low = AdvSimd.AddPairwiseWidening(AdvSimd.MultiplyWideningLower(c.GetLower(), x.GetLower()));
            var high = AdvSimd.AddPairwiseWidening(AdvSimd.MultiplyWideningUpper(c, x));
            return sum + AdvSimd.Arm64.AddPairwise(low, high);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Vector256.Create(
            AdvSimd.FusedMultiplyAdd(c.GetLower(), a.GetLower(), b.GetLower()), AdvSimd.FusedMultiplyAdd(c.GetUpper(), a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) => Vector256.Create(
            AdvSimd.Arm64.FusedMultiplyAdd(c.GetLower(), a.GetLower(), b.GetLower()), AdvSimd.Arm64.FusedMultiplyAdd(c.GetUpper(), a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b) =>
            Vector256.Create(AdvSimd.Arm64.ZipLow(a.GetLower(), b.GetLower()), AdvSimd.Arm64.ZipLow(a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b) =>
            Vector256.Create(AdvSimd.Arm64.ZipHigh(a.GetLower(), b.GetLower()), AdvSimd.Arm64.ZipHigh(a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b) =>
            Vector256.Create(AdvSimd.Arm64.ZipLow(a.GetLower(), b.GetLower()), AdvSimd.Arm64.ZipLow(a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b) =>
            Vector256.Create(AdvSimd.Arm64.ZipHigh(a.GetLower(), b.GetLower()), AdvSimd.Arm64.ZipHigh(a.GetUpper(), b.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => PortableInstructions.Swap(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => PortableInstructions.Lows(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => PortableInstructions.Highs(a, b);

        // USHL shifts left by each lane's signed count (its low byte), right where it is negative.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts) => Vector256.Create(
            AdvSimd.ShiftLogical(v.GetLower(), -counts.GetLower().AsInt32()), AdvSimd.ShiftLogical(v.GetUpper(), -counts.GetUpper().AsInt32()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts) => Vector256.Create(
            AdvSimd.ShiftLogical(v.GetLower(), counts.GetLower().AsInt32()), AdvSimd.ShiftLogical(v.GetUpper(), counts.GetUpper().AsInt32()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts) => Vector256.Create(
            AdvSimd.ShiftLogical(v.GetLower(), -counts.GetLower().AsInt64()), AdvSimd.ShiftLogical(v.GetUpper(), -counts.GetUpper().AsInt64()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) => PortableInstructions.ShiftUpper(v, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => PortableInstructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count) => PortableInstructions.Gather(at, offsets, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => PortableInstructions.Gather(table, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => PortableInstructions.Gather(at, offsets);

        // TBL: a control byte past 15 gives 0, as the top bit's does on x86's byte shuffle.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) => Vector256.Create(
            AdvSimd.Arm64.VectorTableLookup(bytes.GetLower(), control.GetLower()), AdvSimd.Arm64.VectorTableLookup(bytes.GetUpper(), control.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => PortableInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => PortableInstructions.Doubles(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at) => PortableInstructions.Halves(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at) => PortableInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at) => PortableInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v) => PortableInstructions.Widen(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed) => PortableInstructions.Widen(v, signed);

        // The upper half's control (16-31) picks from the two halves' 32 bytes as it is, no subtraction.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => Vector256.Create(
            AdvSimd.Arm64.VectorTableLookup(bytes.GetLower(), control.GetLower()),
            AdvSimd.Arm64.VectorTableLookup((bytes.GetLower(), bytes.GetUpper()), control.GetUpper()));

        // Lane `lane` of the 32 bytes picked into every lane of a half (a lane known only at run time: no stack store).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane)
        {
            var picked = AdvSimd.Arm64.VectorTableLookup((v.GetLower().AsByte(), v.GetUpper().AsByte()),
                Vector128.Create((byte)(4 * lane)) + Vector128.Create((byte)0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)).AsSingle();
            return Vector256.Create(picked, picked);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) => Permute(v.AsInt32(), indices).AsSingle();

        // Each lane's 4 bytes picked from the 32 by TBL: index i's bytes 4i..4i + 3 (indices 0-7, as Pairs gives them).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices)
        {
            var table = (v.GetLower().AsByte(), v.GetUpper().AsByte());
            return Vector256.Create(AdvSimd.Arm64.VectorTableLookup(table, ByteIndices(indices.GetLower())).AsInt32(),
                AdvSimd.Arm64.VectorTableLookup(table, ByteIndices(indices.GetUpper())).AsInt32());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> ByteIndices(Vector128<int> indices) =>
            (indices.AsUInt32() * Vector128.Create(0x04040404u) + Vector128.Create(0x03020100u)).AsByte();

        // Each lane shifted right by first + 2l by USHL (its negated count), then its low 3 bits: 0-7 exactly.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first)
        {
            var seven = Vector128.Create(7u);
            var low = first == 0 ? Vector128.Create(0, -2, -4, -6) : Vector128.Create(-16, -18, -20, -22);
            var high = first == 0 ? Vector128.Create(-8, -10, -12, -14) : Vector128.Create(-24, -26, -28, -30);
            return Vector256.Create(AdvSimd.ShiftLogical(word.GetLower(), low) & seven, AdvSimd.ShiftLogical(word.GetUpper(), high) & seven);
        }

        public static bool ShiftsByLane => true;

        public static bool RowsTogether => true;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => PortableInstructions.MultiplyHigh(a, b);

        public static bool MultipliesHigh => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Vector256.Create(table[indices.GetElement(0)], table[indices.GetElement(2)], table[indices.GetElement(4)], table[indices.GetElement(6)]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => BitsExtracted(value, mask);

        // .NET reaches no prefetch on Arm outside SVE (Sve.Prefetch*, experimental, and only where SVE runs).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) => AdvSimd.FusedMultiplyAdd(c, a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) => AdvSimd.Arm64.FusedMultiplyAdd(c, a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => AdvSimd.Arm64.ZipLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => AdvSimd.Arm64.ZipHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => AdvSimd.Arm64.ZipLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => AdvSimd.Arm64.ZipHigh(a, b);
    }

    /// <summary>
    /// The lanes' operations by NEON (NeonInstructions), the int8 dot products by a 64-bit Arm CPU's dot products
    /// (FEAT_DotProd, optional from Armv8.2 and mandatory from Armv8.4: SDOT, each 32-bit lane plus the products of its 4
    /// signed bytes by x's 4, exactly): the codes of a kind that takes them at most 127 (LaneList.WithKind), so signed bytes
    /// as they are.
    /// </summary>
    private unsafe struct DotProdInstructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x) => Vector256.Create(
            Dp.DotProduct(sum.GetLower(), codes.GetLower().AsSByte(), x.GetLower()), Dp.DotProduct(sum.GetUpper(), codes.GetUpper().AsSByte(), x.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => NeonInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) => NeonInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b) => NeonInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b) => NeonInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b) => NeonInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b) => NeonInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => NeonInstructions.Swap(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => NeonInstructions.Lows(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => NeonInstructions.Highs(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts) => NeonInstructions.ShiftRight(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts) => NeonInstructions.ShiftLeft(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts) => NeonInstructions.ShiftRight(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) => NeonInstructions.ShiftUpper(v, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => NeonInstructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count) => NeonInstructions.Gather(at, offsets, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => NeonInstructions.Gather(table, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => NeonInstructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) => NeonInstructions.ShuffleBytes(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => NeonInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => NeonInstructions.Doubles(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at) => NeonInstructions.Halves(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at) => NeonInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at) => NeonInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v) => NeonInstructions.Widen(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed) => NeonInstructions.Widen(v, signed);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => NeonInstructions.ShuffleHalves(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane) => NeonInstructions.Broadcast(v, lane);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) => NeonInstructions.Permute(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices) => NeonInstructions.Permute(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first) => NeonInstructions.Pairs(word, first);

        public static bool ShiftsByLane => NeonInstructions.ShiftsByLane;

        public static bool RowsTogether => NeonInstructions.RowsTogether;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => NeonInstructions.MultiplyHigh(a, b);

        public static bool MultipliesHigh => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Vector256.Create(table[indices.GetElement(0)], table[indices.GetElement(2)], table[indices.GetElement(4)], table[indices.GetElement(6)]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => BitsExtracted(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at) => NeonInstructions.Prefetch(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x) => Dp.DotProduct(sum, codes.AsSByte(), x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) => NeonInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) => NeonInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => NeonInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => NeonInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => NeonInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => NeonInstructions.UnpackHigh(a, b);
    }

    /// <summary>
    /// The lanes' operations as DotProdInstructions does them, the int8 dot products of any unsigned codes (0-255) by two of
    /// Arm's signed ones a step: c · x = (c − 128) · x + (−128) · (−x), the codes with their top bit flipped read as signed
    /// bytes, then −128 by −x (x within ±127, so −x is a signed byte too); every sum exact in 32 bits.
    /// </summary>
    private unsafe struct WideDotProdInstructions : IInstructions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, Vector256<sbyte> x) =>
            Vector256.Create(Dot(sum.GetLower(), codes.GetLower(), x.GetLower()), Dot(sum.GetUpper(), codes.GetUpper(), x.GetUpper()));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Dot(Vector128<int> sum, Vector128<byte> codes, Vector128<sbyte> x) =>
            Dp.DotProduct(Dp.DotProduct(sum, (codes ^ Vector128.Create((byte)0x80)).AsSByte(), x), Vector128.Create((sbyte)-128), -x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Fused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => DotProdInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Fused(Vector256<double> a, Vector256<double> b, Vector256<double> c) => DotProdInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackLow(Vector256<uint> a, Vector256<uint> b) => DotProdInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> UnpackHigh(Vector256<uint> a, Vector256<uint> b) => DotProdInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackLow(Vector256<ulong> a, Vector256<ulong> b) => DotProdInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> UnpackHigh(Vector256<ulong> a, Vector256<ulong> b) => DotProdInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Swap(Vector256<uint> v) => DotProdInstructions.Swap(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Lows(Vector256<ulong> a, Vector256<ulong> b) => DotProdInstructions.Lows(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> Highs(Vector256<ulong> a, Vector256<ulong> b) => DotProdInstructions.Highs(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftRight(Vector256<uint> v, Vector256<uint> counts) => DotProdInstructions.ShiftRight(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts) => DotProdInstructions.ShiftLeft(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts) => DotProdInstructions.ShiftRight(v, counts);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftUpper(Vector256<uint> v, int count) => DotProdInstructions.ShiftUpper(v, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets) => DotProdInstructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, int* offsets, int count) => DotProdInstructions.Gather(at, offsets, count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Gather(float* table, Vector256<int> indices) => DotProdInstructions.Gather(table, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Gather(byte* at, Vector256<int> offsets) => DotProdInstructions.Gather(at, offsets);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleBytes(Vector256<byte> bytes, Vector256<byte> control) => DotProdInstructions.ShuffleBytes(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(byte* at) => DotProdInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Doubles(float* at) => DotProdInstructions.Doubles(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Halves(byte* at) => DotProdInstructions.Halves(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(ushort* at) => DotProdInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(short* at) => DotProdInstructions.Widen(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<sbyte> v) => DotProdInstructions.Widen(v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Widen(Vector128<byte> v, bool signed) => DotProdInstructions.Widen(v, signed);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> ShuffleHalves(Vector256<byte> bytes, Vector256<byte> control) => DotProdInstructions.ShuffleHalves(bytes, control);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Broadcast(Vector256<float> v, int lane) => DotProdInstructions.Broadcast(v, lane);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Permute(Vector256<float> v, Vector256<int> indices) => DotProdInstructions.Permute(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Permute(Vector256<int> v, Vector256<int> indices) => DotProdInstructions.Permute(v, indices);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> Pairs(Vector256<uint> word, int first) => DotProdInstructions.Pairs(word, first);

        public static bool ShiftsByLane => DotProdInstructions.ShiftsByLane;

        public static bool RowsTogether => DotProdInstructions.RowsTogether;

        // Each 16-bit lane's unsigned product's high 16 bits.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ushort> MultiplyHigh(Vector256<ushort> a, Vector256<ushort> b) => DotProdInstructions.MultiplyHigh(a, b);

        public static bool MultipliesHigh => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<long> GatherPairs(long* table, Vector256<int> indices) => Vector256.Create(table[indices.GetElement(0)], table[indices.GetElement(2)], table[indices.GetElement(4)], table[indices.GetElement(6)]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Deposit(ulong value, ulong mask) => BitsDeposited(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Extract(ulong value, ulong mask) => BitsExtracted(value, mask);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Prefetch(void* at) => DotProdInstructions.Prefetch(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Fused(Vector128<float> a, Vector128<float> b, Vector128<float> c) => DotProdInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> Fused(Vector128<double> a, Vector128<double> b, Vector128<double> c) => DotProdInstructions.Fused(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackLow(Vector128<uint> a, Vector128<uint> b) => DotProdInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<uint> UnpackHigh(Vector128<uint> a, Vector128<uint> b) => DotProdInstructions.UnpackHigh(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackLow(Vector128<ulong> a, Vector128<ulong> b) => DotProdInstructions.UnpackLow(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<ulong> UnpackHigh(Vector128<ulong> a, Vector128<ulong> b) => DotProdInstructions.UnpackHigh(a, b);
    }

    /// <summary>
    /// Whether Vector256.FusedMultiplyAdd, for float32 and for float64, runs faster than ExactFused (the CPU fuses in
    /// hardware; else the runtime's own fused multiply-add is a scalar call a value): each timed once on the same values in
    /// 8 chains, the least of 5 rounds, the faster kept. Both give the same bits, so the choice changes speed alone. Made
    /// before a lane's kernels are compiled (Ensure), so their code reads it as a constant.
    /// </summary>
    private static class FusedInHardware
    {
        public static readonly bool Single, Double;

        private static double s_sink;

        static FusedInHardware()
        {
            Single = SingleTime(hardware: true) < SingleTime(hardware: false);
            Double = DoubleTime(hardware: true) < DoubleTime(hardware: false);
        }

        /// <summary>The choice made (the type's initializer run), before a kernel reads it.</summary>
        public static void Ensure()
        {
        }

        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private static long SingleTime(bool hardware)
        {
            var a = Vector256.Create(1.0001f);
            var b = Vector256.Create(0.4990234375f);
            long least = long.MaxValue;
            for (int round = 0; round < 5; round++)
            {
                Vector256<float> s0 = b, s1 = b, s2 = b, s3 = b, s4 = b, s5 = b, s6 = b, s7 = b;
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                for (int i = 0; i < 1024; i++)
                {
                    if (hardware)
                    {
                        (s0, s1, s2, s3) = (Vector256.FusedMultiplyAdd(a, s0, b), Vector256.FusedMultiplyAdd(a, s1, b), Vector256.FusedMultiplyAdd(a, s2, b), Vector256.FusedMultiplyAdd(a, s3, b));
                        (s4, s5, s6, s7) = (Vector256.FusedMultiplyAdd(a, s4, b), Vector256.FusedMultiplyAdd(a, s5, b), Vector256.FusedMultiplyAdd(a, s6, b), Vector256.FusedMultiplyAdd(a, s7, b));
                    }
                    else
                    {
                        (s0, s1, s2, s3) = (ExactFused(a, s0, b), ExactFused(a, s1, b), ExactFused(a, s2, b), ExactFused(a, s3, b));
                        (s4, s5, s6, s7) = (ExactFused(a, s4, b), ExactFused(a, s5, b), ExactFused(a, s6, b), ExactFused(a, s7, b));
                    }
                }

                least = Math.Min(least, System.Diagnostics.Stopwatch.GetTimestamp() - t0);
                s_sink += Vector256.Sum(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
            }

            return least;
        }

        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private static long DoubleTime(bool hardware)
        {
            var a = Vector256.Create(1.0001);
            var b = Vector256.Create(0.4990234375);
            long least = long.MaxValue;
            for (int round = 0; round < 5; round++)
            {
                Vector256<double> s0 = b, s1 = b, s2 = b, s3 = b, s4 = b, s5 = b, s6 = b, s7 = b;
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                for (int i = 0; i < 1024; i++)
                {
                    if (hardware)
                    {
                        (s0, s1, s2, s3) = (Vector256.FusedMultiplyAdd(a, s0, b), Vector256.FusedMultiplyAdd(a, s1, b), Vector256.FusedMultiplyAdd(a, s2, b), Vector256.FusedMultiplyAdd(a, s3, b));
                        (s4, s5, s6, s7) = (Vector256.FusedMultiplyAdd(a, s4, b), Vector256.FusedMultiplyAdd(a, s5, b), Vector256.FusedMultiplyAdd(a, s6, b), Vector256.FusedMultiplyAdd(a, s7, b));
                    }
                    else
                    {
                        (s0, s1, s2, s3) = (ExactFused(a, s0, b), ExactFused(a, s1, b), ExactFused(a, s2, b), ExactFused(a, s3, b));
                        (s4, s5, s6, s7) = (ExactFused(a, s4, b), ExactFused(a, s5, b), ExactFused(a, s6, b), ExactFused(a, s7, b));
                    }
                }

                least = Math.Min(least, System.Diagnostics.Stopwatch.GetTimestamp() - t0);
                s_sink += Vector256.Sum(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
            }

            return least;
        }
    }

    // Where 256-bit vectors are not accelerated every Vector256 operation is a method the JIT inlines, and four rows' dot
    // products inlined into one kernel pass its inlining budget, the rest left as calls (measured: 26-37 calls in a four-row
    // kernel's code, none in a one-row kernel's). There the four-row kernels take a row at a time by their one-row kernels
    // (the same operations in the same order, so the same bits), each compiled on its own budget (the integer lane's tiles
    // dot one row of x at a time in their loop, so they take no such step). Every lane's 128-bit twin (NarrowLane) has four-row kernels
    // written on 128-bit vectors (each 256-bit vector of the 256-bit kernel as its two halves, every operation the 256-bit
    // one's on its half, and the halves' sums added as Vector256.Sum adds them: the same bits), whose operations are the
    // CPU's own instructions wherever 128-bit vectors are accelerated: four rows of x share each stored row's decode there
    // too (the integer lane's twins make a group's steps as its 256-bit kernels do, then dot each step's halves). The
    // measured choice times the twins against the 256-bit lanes for four rows of x or more.

    // The sum of a 256-bit lane set held as its two 128-bit halves, as Vector256.Sum adds them (the lower half's sum, then
    // the upper's: each half's lanes in pairs).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SumOf(Vector128<float> lower, Vector128<float> upper) => Vector128.Sum(lower) + Vector128.Sum(upper);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SumOf(Vector128<double> lower, Vector128<double> upper) => Vector128.Sum(lower) + Vector128.Sum(upper);

    // x's 16 bytes from `at` (a half of a 256-bit step's 32).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Dot<TI>(Vector128<int> sum, Vector128<byte> codes, ref sbyte x, nint at) where TI : struct, IInstructions =>
        TI.Dot(sum, codes, Vector128.LoadUnsafe(ref x, (nuint)at));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<int> DotAt<TI>(Vector128<int> sum, Vector128<byte> codes, sbyte* x) where TI : struct, IInstructions =>
        TI.Dot(sum, codes, Vector128.Load(x));

    // a · b + c rounded once from separate operations: each product exact in float64, its sum with c rounded to odd there,
    // then to float32 to nearest (float64's 53 bits are float32's 24 and two more: the two roundings give the one).
    // Where 256-bit vectors are not accelerated, by their 128-bit halves (the same operations, lane by lane).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ExactFused(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Vector256.IsHardwareAccelerated
        ? Vector256.Narrow(OddSum(Vector256.WidenLower(a) * Vector256.WidenLower(b), Vector256.WidenLower(c)),
            OddSum(Vector256.WidenUpper(a) * Vector256.WidenUpper(b), Vector256.WidenUpper(c)))
        : Vector256.Create(ExactFused(a.GetLower(), b.GetLower(), c.GetLower()), ExactFused(a.GetUpper(), b.GetUpper(), c.GetUpper()));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ExactFused(Vector128<float> a, Vector128<float> b, Vector128<float> c) =>
        Vector128.Narrow(OddSum(Vector128.WidenLower(a) * Vector128.WidenLower(b), Vector128.WidenLower(c)),
            OddSum(Vector128.WidenUpper(a) * Vector128.WidenUpper(b), Vector128.WidenUpper(c)));

    // a · b + c rounded once from separate operations: a · b = ph + pl exactly (Dekker's product, each halved by 2^27 + 1),
    // c + ph = th + tl exactly, tl + pl rounded to odd, then th plus that rounded to nearest (Boldo and Melquiond's emulated
    // fused multiply-add). An operand past 1e299 (its halves' product would overflow) or not finite takes the plain sum,
    // as do products whose low part falls below float64's normal range (2^-969): neither comes up in the lanes.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> ExactFused(Vector256<double> a, Vector256<double> b, Vector256<double> c)
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return Vector256.Create(ExactFused(a.GetLower(), b.GetLower(), c.GetLower()), ExactFused(a.GetUpper(), b.GetUpper(), c.GetUpper()));
        }

        var ph = a * b;
        var split = Vector256.Create(134217729.0);
        var ta = a * split;
        var ah = ta - (ta - a);
        var al = a - ah;
        var tb = b * split;
        var bh = tb - (tb - b);
        var bl = b - bh;
        var pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;
        var th = c + ph;
        var cc = th - ph;
        var tl = (c - cc) + (ph - (th - cc));
        var v = OddSum(tl, pl);
        var sum = Vector256.ConditionalSelect(Vector256.Equals(v, Vector256<double>.Zero), th, th + v);
        var most = Vector256.Create(1e299);
        var finite = Vector256.LessThan(Vector256.Abs(a), most) & Vector256.LessThan(Vector256.Abs(b), most) & Vector256.LessThan(Vector256.Abs(c), Vector256.Create(double.PositiveInfinity));
        return Vector256.ConditionalSelect(finite, sum, ph + c);
    }

    // ExactFused over 128 bits (the same operations, lane by lane).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> ExactFused(Vector128<double> a, Vector128<double> b, Vector128<double> c)
    {
        var ph = a * b;
        var split = Vector128.Create(134217729.0);
        var ta = a * split;
        var ah = ta - (ta - a);
        var al = a - ah;
        var tb = b * split;
        var bh = tb - (tb - b);
        var bl = b - bh;
        var pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;
        var th = c + ph;
        var cc = th - ph;
        var tl = (c - cc) + (ph - (th - cc));
        var v = OddSum(tl, pl);
        var sum = Vector128.ConditionalSelect(Vector128.Equals(v, Vector128<double>.Zero), th, th + v);
        var most = Vector128.Create(1e299);
        var finite = Vector128.LessThan(Vector128.Abs(a), most) & Vector128.LessThan(Vector128.Abs(b), most) & Vector128.LessThan(Vector128.Abs(c), Vector128.Create(double.PositiveInfinity));
        return Vector128.ConditionalSelect(finite, sum, ph + c);
    }

    // OddSum over 128 bits (the same operations, lane by lane).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> OddSum(Vector128<double> p, Vector128<double> c)
    {
        var s = p + c;
        var pp = s - c;
        var e = (p - pp) + (c - (s - pp));
        var bits = s.AsInt64();
        var exponent = Vector128.Create(0x7FF0000000000000L);
        var inexact = ~Vector128.Equals(e, Vector128<double>.Zero).AsInt64() & Vector128.Equals(bits & Vector128<long>.One, Vector128<long>.Zero)
            & ~Vector128.Equals(bits & exponent, exponent);
        var step = Vector128.LessThan(bits ^ e.AsInt64(), Vector128<long>.Zero) | Vector128<long>.One;
        return (bits + (step & inexact)).AsDouble();
    }

    // p + c rounded to odd: their sum where it is exact, else the neighbour of the exact sum whose last bit is 1 (the sum
    // rounded to nearest, its error by Knuth's two-sum, then one step toward the error where that last bit is 0).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> OddSum(Vector256<double> p, Vector256<double> c)
    {
        var s = p + c;
        var pp = s - c;
        var e = (p - pp) + (c - (s - pp));
        var bits = s.AsInt64();
        var exponent = Vector256.Create(0x7FF0000000000000L);
        var inexact = ~Vector256.Equals(e, Vector256<double>.Zero).AsInt64() & Vector256.Equals(bits & Vector256<long>.One, Vector256<long>.Zero)
            & ~Vector256.Equals(bits & exponent, exponent);
        var step = Vector256.LessThan(bits ^ e.AsInt64(), Vector256<long>.Zero) | Vector256<long>.One;   // -1 toward zero, 1 away
        return (bits + (step & inexact)).AsDouble();
    }

    // ------------------------------------------------------------------ the described lane's measured choice and its tiles

    /// <summary>
    /// A product with candidate lanes (up to MostCandidates), timed on the real operands once per shape and the fastest kept
    /// for the shape (Chosen): a weight's product over its stored rows by the described lane (LaneCandidates), by the
    /// portable instructions and by those the CPU has, and their 128-bit twins for 4 rows of x or more.
    /// </summary>
    private interface ILaneChoice
    {
        /// <summary>The weight's description: its block in the recorded choices' names, its fingerprint in the machine's file.</summary>
        TypeDescription Description { get; }

        /// <summary>
        /// The lanes measured for the description's shapes, held by the description's own CPU plan (DescribedPlan.Chosen):
        /// by shape alone (the vector paths, the rows' class, n and k), each with the generation it was measured in.
        /// </summary>
        ConcurrentDictionary<(CpuVectorPaths Vectors, int Threads, int Rows, int N, int K), (int Lane, int Generation)> Chosen { get; }

        /// <summary>The candidates that run on this CPU (1 to MostCandidates).</summary>
        int Count { get; }

        /// <summary>Candidate <paramref name="index"/>'s lane, the first the one the others are timed against.</summary>
        int Candidate(int index);

        /// <summary>The lane without measuring (CpuTuning.Autotune off).</summary>
        int Unmeasured { get; }

        /// <summary>The product by <paramref name="lane"/>: all of y written.</summary>
        void Run(int lane);
    }

    /// <summary>
    /// A lane's format over a weight's rows (k values each, a stored row's bytes apart) where they are stored (the described
    /// lane's, DescribedFormat): read by its description alone; one or two kinds of lane, each a candidate by the portable instructions and by each instruction
    /// set the CPU has that it takes (LaneList), the fastest measured per shape; what each lane makes of x first; the stored
    /// rows a group of a tile takes, the values of k a pass takes and the scratch its kernels take; its kernels over a tile.
    /// </summary>
    private unsafe interface IRowFormat<TFormat, TKernels> where TFormat : struct, IRowFormat<TFormat, TKernels> where TKernels : struct, IRowKernels
    {
        /// <summary>The format of rows of <paramref name="k"/> values of <paramref name="description"/>; false where it is not one the format reads.</summary>
        static abstract bool Reads(TypeDescription description, int k, out TFormat format);

        /// <summary>The candidate lanes that run on this CPU, the first the one the others are timed against.</summary>
        LaneList Lanes { get; }

        /// <summary>The format's first kind of lane (its lanes of one kind give the same bits, whatever their instructions).</summary>
        int First { get; }

        /// <summary>Its second kind (First where it has one).</summary>
        int Second { get; }

        /// <summary>The lane without measuring (by the portable instructions).</summary>
        int Unmeasured { get; }

        /// <summary>Which of x's preparations <paramref name="lane"/> reads (Prepare: the lanes of one kind share one).</summary>
        int InputOf(int lane);

        /// <summary>Stored bytes a group of a tile takes with more than one row of x (RowGroups).</summary>
        int GroupBytes { get; }

        /// <summary>
        /// The columns and the rows of x a tile takes with more than one row of x (RowTiles), of <paramref name="total"/>
        /// columns and <paramref name="m"/> rows over <paramref name="threads"/> threads, the stored rows
        /// <paramref name="rowBytes"/> apart.
        /// </summary>
        (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes);

        /// <summary>Values of k four rows of x pass over a group at a time (k or more: all of k in one pass).</summary>
        int Span(int k);

        /// <summary>x as <paramref name="lane"/> reads it: x itself, or what the lane makes of it first (pooled; LaneInput.Release).</summary>
        LaneInput Prepare(int lane, float[] x, int m, int k);

        /// <summary>The bytes of scratch a tile's kernels take: a row's last period, where it is short of one, copied into zeros as it is stored (0: none).</summary>
        int Scratch(int k);

        /// <summary>The kernels of <paramref name="lane"/> (its kind, by its instructions) for a tile: x and the scratch (its bytes cleared) pinned.</summary>
        TKernels Kernels(int lane, int k, LanePointers x, byte* bytes);
    }

    /// <summary>A lane's kernels over a weight's rows read where they are stored (RowGroups'), each stored value made in registers as it is read.</summary>
    private unsafe interface IRowKernels
    {
        /// <summary>The rows of x the kernels take at once against a group of stored rows (Tile), or 0: four at a time (Four).</summary>
        int Block { get; }

        /// <summary>
        /// Rows i..i + count − 1 of x (count at most Block) by <paramref name="columns"/> stored rows from
        /// <paramref name="rows"/> (<paramref name="rowBytes"/> apart): stored row c by row r's product into
        /// made[c · count + r], each pair's sums in its 32 bytes of <paramref name="sums"/>.
        /// </summary>
        void Tile(byte* rows, nint rowBytes, int columns, int i, int count, byte* sums, float* made);

        /// <summary>Rows i..i + 3 of x by the stored row at <paramref name="row"/>, over values [p0, p1) of k.</summary>
        void Four(byte* row, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3);

        /// <summary>Whether the kernels take eight rows of x at a time (Eight) where they would take four (Four).</summary>
        bool Eights { get; }

        /// <summary>Rows i..i + 7 of x by the stored row at <paramref name="row"/>, over values [p0, p1) of k, into y[0..8).</summary>
        void Eight(byte* row, int i, int p0, int p1, float* y);

        /// <summary>Row i of x by the stored row at <paramref name="row"/>.</summary>
        float Single(byte* row, int i);
    }

    /// <summary>
    /// x as a lane reads it: x itself, or what the lane made of it first (int8 codes and their scales; values reordered).
    /// The codes and scales lie from CodesAt and ScalesAt (elements) of their arrays: where the arrays' data start on a
    /// cache line, so no 32-byte read of a row of x splits two lines.
    /// </summary>
    private readonly record struct LaneInput(float[] Values, bool Owned, sbyte[]? Codes, float[]? Scales, bool Natural = false, int CodesAt = 0, int ScalesAt = 0)
    {
        /// <summary>The pooled buffers back to the pool.</summary>
        public void Release()
        {
            if (Owned)
            {
                ArrayPool<float>.Shared.Return(Values);
            }

            if (Codes is not null)
            {
                ArrayPool<sbyte>.Shared.Return(Codes);
            }

            if (Scales is not null)
            {
                ArrayPool<float>.Shared.Return(Scales);
            }
        }
    }

    /// <summary>A lane's x, pinned for a tile (LaneInput's arrays).</summary>
    private readonly unsafe struct LanePointers
    {
        public LanePointers(float* values, sbyte* codes, float* scales, bool natural)
        {
            Values = values;
            Codes = codes;
            Scales = scales;
            Natural = natural;
        }

        /// <summary>Whether x's int8 codes lie in the natural layout (LaneInput.Natural: the plan's Natural reads them).</summary>
        public bool Natural { get; }

        public float* Values { get; }

        public sbyte* Codes { get; }

        public float* Scales { get; }
    }

    // The shape's lane of `product`: kept by its description's plan from its measuring in this generation, or measured now
    // (Measure).
    private static int Chosen<TC>(in TC product, int m, int n, int k) where TC : struct, ILaneChoice =>
        product.Chosen.TryGetValue(ShapeOf(m, n, k), out var known) && known.Generation == Volatile.Read(ref s_lanesForgotten) ? known.Lane : Measure(product, m, n, k);

    // How many times this process's choices were forgotten (ForgetChoices, tests): a lane a plan kept from an earlier
    // generation is measured again.
    private static int s_lanesForgotten;

    // A shape, as a description's plan keeps its measured lane (DescribedPlan.Chosen): the vector paths allowed (a choice
    // made under one setting is not taken under another), the threads it runs on (a lane's time is its share's, so the
    // fastest on one thread need not be on many), the rows by class (exact up to 8, then powers of two, so prompts of every
    // length measure a few classes), n and k.
    private static (CpuVectorPaths Vectors, int Threads, int Rows, int N, int K) ShapeOf(int m, int n, int k) =>
        (ComputeResources.CpuVectors, ComputeResources.MaxCpuThreads, m <= 8 ? m : (int)BitOperations.RoundUpToPowerOf2((uint)m), n, k);

    // A shape's name in the recorded choices (ProductRoutes.Chose): the description's block and the product's shape.
    private static string ShapeName(TypeDescription d, int m, int n, int k) =>
        string.Create(System.Globalization.CultureInfo.InvariantCulture, $"described ({d.Values} values in {d.Bytes} bytes) {m}x{k}->{n}");

    // The most candidates a product's measuring takes: the described lane by the portable instructions and by each
    // instruction set the CPU has that it takes, with the 128-bit twins for 4 rows of x or more.
    private const int MostCandidates = 16;

    // The fastest of the product's candidates, timed once on the real operands, each against the fastest so far (each run
    // writes all of y, which the caller writes again: Faster until the times settle) and kept for the shape; a lone
    // candidate unmeasured (recorded as the only one); without measuring (CpuTuning.Autotune off), the product's
    // unmeasured lane.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Measure<TC>(TC product, int m, int n, int k) where TC : struct, ILaneChoice
    {
        // The unmeasured lane where it is among the candidates (the vector paths allowed may leave it out), else the first.
        int count = Math.Clamp(product.Count, 1, MostCandidates);
        bool listed = false;
        for (int c = 0; c < count; c++)
        {
            listed |= product.Candidate(c) == product.Unmeasured;
        }

        int best = count > 1 && listed ? product.Unmeasured : product.Candidate(0);
        if (count == 1)
        {
            ProductRoutes.Chose("cpu", ShapeName(product.Description, m, n, k), $"{LaneName(best)} (the only candidate)");
        }
        else if (KeptLane(product, m, n, k, count) is { } kept)
        {
            best = kept;
            if (!ProductRoutes.Choices.ContainsKey("cpu: " + ShapeName(product.Description, m, n, k)))
            {
                // (A shape measured in this process keeps its times in the record; one read from the file is listed as kept.)
                ProductRoutes.Chose("cpu", ShapeName(product.Description, m, n, k), $"{LaneName(best)} (kept from this machine's measured choices)");
            }
        }
        else if (CpuTuning.Autotune)
        {
            // Every candidate timed alike (CpuChoices.RoundTimes: warmed once, then rounds in the same order, each one's
            // median), none timed against a holder and none given a head start; the lowest kept for the machine.
            var each = new ChoiceRuns<TC>(product);
            var times = CpuChoices.RoundTimes(ref each, count);
            best = product.Candidate(CpuChoices.Lowest(times));
            ProductRoutes.Chose("cpu", ShapeName(product.Description, m, n, k),
                string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{LaneName(best)} ({string.Join(", ", Enumerable.Range(0, count).Select(i => $"{LaneShortName(product.Candidate(i))} {times[i]:0.###} ms"))})"));
            CpuChoices.Keep(LaneFileKey(product, m, n, k), LaneVariant(product, count), best.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        product.Chosen[ShapeOf(m, n, k)] = (best, Volatile.Read(ref s_lanesForgotten));
        return best;
    }

    // A product's candidates as RoundTimes runs them.
    private struct ChoiceRuns<TC>(TC product) : CpuChoices.ICandidates where TC : struct, ILaneChoice
    {
        private TC _product = product;

        public void Run(int index) => _product.Run(_product.Candidate(index));
    }

    // The machine's kept lane for the shape (CpuChoices) among these candidates as they are now, where it is one of them.
    private static int? KeptLane<TC>(in TC product, int m, int n, int k, int count) where TC : struct, ILaneChoice
    {
        if (!CpuChoices.TryGet(LaneFileKey(product, m, n, k), LaneVariant(product, count), out var kept)
            || !int.TryParse(kept, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int lane))
        {
            return null;
        }

        for (int c = 0; c < count; c++)
        {
            if (product.Candidate(c) == lane)
            {
                return lane;
            }
        }

        return null;
    }

    // A shape's key in the machine's kept choices: the description's fingerprint (from its fields alone), the vector paths
    // allowed, the threads, the rows' class, n and k.
    private static string LaneFileKey<TC>(in TC product, int m, int n, int k) where TC : struct, ILaneChoice
    {
        var (vectors, threads, rows, _, _) = ShapeOf(m, n, k);
        return string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described {product.Description.CpuFingerprint} {rows}x{k}->{n} {vectors} t{threads}");
    }

    // What a shape's candidates are, as its kept choice names them (CpuChoices' variant): the code they run (CpuCode's
    // fingerprint of what the described lane's candidates reach: every lane's kernels, by every instruction set, and the
    // float32 product), then their lanes.
    private static string LaneVariant<TC>(in TC product, int count) where TC : struct, ILaneChoice
    {
        var text = new System.Text.StringBuilder("c").Append(CpuCode.Of(typeof(LaneCandidates))).Append('/');
        for (int c = 0; c < count; c++)
        {
            text.Append(c == 0 ? "" : ".").Append(product.Candidate(c).ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        return text.ToString();
    }

    // A description's fingerprint: SHA-256 of its fields as text (each list's items, never a type's name), 16 hex digits.
    // Made once and held by the description itself (TypeDescription.CpuFingerprint).
    internal static string FingerprintOf(TypeDescription x)
    {
        var text = new System.Text.StringBuilder();
        static void Number(System.Text.StringBuilder t, NumberType n)
        {
            t.Append(System.Globalization.CultureInfo.InvariantCulture, $"[{n.Bits} {n.Kind} {n.Signed} {n.Exponent} {n.Mantissa} {n.ExponentBias} {n.Subnormals} {n.Specials} {n.Width}");
            foreach (double v in n.Table ?? [])
            {
                t.Append(' ').Append(BitConverter.DoubleToInt64Bits(v));
            }

            t.Append(']');
        }

        static void Scales(System.Text.StringBuilder t, IReadOnlyList<BlockScale>? scales)
        {
            foreach (var s in scales ?? [])
            {
                Number(t, s.Type);
                t.Append(System.Globalization.CultureInfo.InvariantCulture, $" {s.Values} {s.Offset} {BitConverter.DoubleToInt64Bits(s.Bias)}");
                foreach (var pieces in s.Pieces ?? [])
                {
                    t.Append(" (");
                    foreach (var piece in pieces)
                    {
                        t.Append(System.Globalization.CultureInfo.InvariantCulture, $"{piece.Offset}.{piece.Shift}.{piece.Bits} ");
                    }

                    t.Append(')');
                }

                t.Append(';');
            }
        }

        text.Append(System.Globalization.CultureInfo.InvariantCulture, $"{x.Values} {x.Bytes} {BitConverter.DoubleToInt64Bits(x.CodeOffset)} {x.MinsAdded} {x.Group} {x.GroupBytes} ");
        Number(text, x.Code);
        foreach (var plane in x.Planes)
        {
            text.Append(System.Globalization.CultureInfo.InvariantCulture, $" <{plane.Offset} {plane.Bits} {plane.Group} {plane.Stride} {plane.Stream} {plane.Radix}");
            foreach (var place in plane.Places ?? [])
            {
                text.Append(System.Globalization.CultureInfo.InvariantCulture, $" {place.Byte}.{place.Shift}");
            }

            text.Append('>');
        }

        text.Append(" scales ");
        Scales(text, x.Scales);
        text.Append(" mins ");
        Scales(text, x.Mins);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(text.ToString())))[..16];
    }

    // A weight's candidates as a choice of lanes (Chosen): the described lane by the portable instructions and by each
    // instruction set the CPU has that its plan takes, the 128-bit twins among them for 4 rows of x or more
    // (LaneList.ForRows); its product by any of them.
    private readonly struct LaneCandidates(DescribedFormat described, TypeDescription description, StoredLane lane, float[] x, float[] y, int m, int n, int k)
        : ILaneChoice
    {
        private LaneList Lanes => described.Lanes.ForRows(m);

        public TypeDescription Description => description;

        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Threads, int Rows, int N, int K), (int Lane, int Generation)> Chosen => described.Chosen;

        public int Count => Lanes.Count;

        public int Candidate(int index) => Lanes[index];

        public int Unmeasured => described.Unmeasured;

        public void Run(int laneId)
        {
            if (KindOf(laneId) == ProductLaneId)
            {
                ProductOverRows(lane, x, y, m, n, k);
                return;
            }

            FormatProduct<DescribedFormat, DescribedKernels>(described, laneId, lane, x, y, m, n, k);
        }
    }

    // The product of a weight of `description` by the described lane: the shape's lane among its candidates (Chosen), then
    // the product by it; false where the lane does not read the description at k. The CPU's way of fusing multiply-adds is
    // measured first (FusedInHardware), so the lanes' kernels compile with it.
    private static bool DescribedProduct(TypeDescription description, StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        FusedInHardware.Ensure();
        if (!DescribedFormat.Reads(description, k, out var described))
        {
            return false;
        }

        var candidates = new LaneCandidates(described, description, lane, x, y, m, n, k);
        candidates.Run(Chosen(candidates, m, n, k));
        return true;
    }

    // Up to three products of a grouped pass, on the stack.
    [InlineArray(3)]
    private struct ThreeProducts
    {
        private (Storage Values, Storage Output, int Columns) _product;
    }

    /// <summary>
    /// y [m, n] = x [m, k] · Wᵀ by the described lane (any type read from its description), by its lane
    /// <paramref name="index"/> among those that run here (DescribedLanesOf: by the portable instructions first), W's n rows
    /// stored as <paramref name="words"/>. False where the described lane does not read the description here or has no such lane. For tests, apart from the
    /// measured choice.
    /// </summary>
    internal static bool DescribedRowLane(int index, TypeDescription description, float[] words, int length, float[] x, float[] y, int m, int n, int k)
    {
        if (!DescribedFormat.Reads(description, k, out var described) || index >= described.Lanes.Count)
        {
            return false;
        }

        FusedInHardware.Ensure();
        var stored = new StoredLane(words, length, k, RowBytesOf(description, k));
        if (KindOf(described.Lanes[index]) == ProductLaneId)
        {
            ProductOverRows(stored, x, y, m, n, k);
            return true;
        }

        FormatProduct<DescribedFormat, DescribedKernels>(described, described.Lanes[index], stored, x, y, m, n, k);
        return true;
    }

    /// <summary>The described lane by the portable instructions (DescribedRowLane's first lane). For tests.</summary>
    internal static bool DescribedRowLane(TypeDescription description, float[] words, int length, float[] x, float[] y, int m, int n, int k) =>
        DescribedRowLane(0, description, words, length, x, y, m, n, k);

    /// <summary>The names of the described lane's lanes that run here for rows of <paramref name="k"/> values of <paramref name="description"/> (DescribedRowLane's indices); empty where it does not read it. For tests.</summary>
    internal static IReadOnlyList<string> DescribedLanesOf(TypeDescription description, int k)
    {
        if (!DescribedFormat.Reads(description, k, out var described))
        {
            return [];
        }

        var lanes = described.Lanes;
        return [.. Enumerable.Range(0, lanes.Count).Select(i => LaneName(lanes[i]))];
    }

    /// <summary>The arithmetic the described lane takes for rows of <paramref name="k"/> values of <paramref name="description"/> ("integer", "float32", "float64"), or null where it does not read it here. For tests.</summary>
    internal static string? DescribedArithmetic(TypeDescription description, int k) => DescribedFormat.Reads(description, k, out var described) ? described.Arithmetic : null;

    /// <summary>
    /// How the described lane makes each span of 32 values of <paramref name="description"/>'s period (its integer and
    /// float32 lanes over codes of at most 8 bits): the spans in runs, from two windows, as a stream, by a codebook's
    /// entries, and one value at a time; null where it does not read the description so. For tests.
    /// </summary>
    internal static (int Runs, int Windows, int Streams, int Codebook, int OneByOne)? DescribedSpans(TypeDescription description, int k) =>
        DescribedFormat.Reads(description, k, out var described) ? described.Spans() : null;

    /// <summary>
    /// The readings of <paramref name="description"/> its plan takes that no built-in type's takes (DescribedPlan.Fallbacks):
    /// "one by one" (a span's codes a value at a time), "table" (codes through the 256-byte table), "codebook" (a codebook
    /// the fast reading does not take), "scalar" (a scale or minimum a lane at a time); null where the described lane does
    /// not read it. For tests.
    /// </summary>
    internal static IReadOnlyList<string>? DescribedFallbacks(TypeDescription description, int k) =>
        DescribedFormat.Reads(description, k, out var described) ? described.Fallbacks() : null;

    // The product by one lane (FormatMany with one part).
    private static void FormatProduct<TF, TK>(TF format, int laneId, StoredLane lane, float[] x, float[] y, int m, int n, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        var parts = new ThreeParts();
        parts[0] = new RowPart(lane, y, n, laneId);
        FormatMany<TF, TK>(format, ((ReadOnlySpan<RowPart>)parts)[..1], x, m, k);
    }

    // 2 or 3 weights of one description sharing x: each by its own lane for the shape among the described lane's candidates
    // (measured here first where it has not been), then one pass (FormatMany), so each output is its own product's, bit for
    // bit; false where the described lane does not read the description at k.
    private static bool DescribedGrouped(TypeDescription description, float[] x, int m, int k, ThreeProducts products, int count)
    {
        FusedInHardware.Ensure();
        // A row-scaled description the lane has no plan for: its codes alone, each output times its row's scale.
        TypeDescription? codes = null;
        if (!DescribedFormat.Reads(description, k, out var described))
        {
            codes = RowCodesOf(description, k);
            if (codes is null || !DescribedFormat.Reads(codes, k, out described))
            {
                return false;
            }
        }

        ThreeParts parts = new();
        for (int j = 0; j < count; j++)
        {
            var (values, output, n) = products[j];
            var lane = codes is null ? new StoredLane(D(values), values.Length, k, RowBytesOf(description, k)) : RowCodesLane(description, D(values), values.Length, k);
            float[] y = D(output);
            var candidates = new LaneCandidates(described, codes ?? description, lane, x, y, m, n, k);
            int chosen = Chosen(candidates, m, n, k);
            if (KindOf(chosen) == ProductLaneId)
            {
                candidates.Run(chosen);                                  // (the float product's: its own, not a pass's part)
                parts[j] = new RowPart(lane, y, 0, chosen);
                continue;
            }

            parts[j] = new RowPart(lane, y, n, chosen);
        }

        // The parts the float product ran (none columns left) out of the pass.
        int kept = 0;
        for (int j = 0; j < count; j++)
        {
            if (parts[j].N > 0)
            {
                parts[kept++] = parts[j];
            }
        }

        if (kept == 0)
        {
            return true;
        }

        count = kept;

        FormatMany<DescribedFormat, DescribedKernels>(described, ((ReadOnlySpan<RowPart>)parts)[..count], x, m, k);
        return true;
    }

    // Products of 2 or 3 weights of one description sharing x (the queries, keys and values; the gate and up) by the
    // described lane in one pass, any rows; offered once such a weight has been multiplied here, so the groups of weights
    // it does not read are not asked (their outputs allocated) only to run one by one.
    public override bool StoredMatMulManyAvailable(int m) => s_describedMultiplied;

    // Whether a weight has been multiplied by the described lane in this process.
    private static volatile bool s_describedMultiplied;

    // 2 or 3 such products in one pass (DescribedGrouped): each by its own lane for the shape (measured first where it has
    // not been), x prepared once, the weights' columns spread over the threads as one range, so each output is its own
    // product's, bit for bit. False for any other weights: the callers then run them one by one.
    public override bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage Output, int Columns)> products)
    {
        var description = products.Length > 0 ? products[0].Values.Description : null;
        if (products.Length is < 2 or > 3 || m <= 0 || k <= 0 || description is null)
        {
            return false;
        }

        var group = new ThreeProducts();
        for (int j = 0; j < products.Length; j++)
        {
            var (values, output, n) = products[j];
            if (values is not CpuStorage || !Equals(values.Description, description) || n <= 0 || values.ElementCount < (long)n * k || output is not CpuStorage)
            {
                return false;
            }

            group[j] = products[j];
        }

        if (!DescribedGrouped(description, D(x), m, k, group, products.Length))
        {
            return false;
        }

        s_describedMultiplied = true;
        return true;
    }

    // One product of a pass over stored rows: its weight, output, columns and lane.
    private readonly record struct RowPart(StoredLane Lane, float[] Y, int N, int LaneId);

    // Up to three parts, on the stack.
    [InlineArray(3)]
    private struct ThreeParts
    {
        private RowPart _part;
    }

    // Up to three preparations of x, on the stack.
    [InlineArray(3)]
    private struct ThreeInputs
    {
        private LaneInput _input;
    }

    // The products of `parts` (1 to 3 weights of one format sharing x): x prepared once for each preparation their lanes
    // read (IRowFormat.InputOf: the lanes of one kind share one), then their tiles over the threads (RowTiles, FormatColumns).
    private static void FormatMany<TF, TK>(TF format, ReadOnlySpan<RowPart> parts, float[] x, int m, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        ThreeInputs inputs = new();
        Span<int> kinds = stackalloc int[3];
        Span<int> slots = stackalloc int[3];
        int prepared = 0;
        try
        {
            for (int p = 0; p < parts.Length; p++)
            {
                int kind = format.InputOf(parts[p].LaneId), slot = kinds[..prepared].IndexOf(kind);
                if (slot < 0)
                {
                    inputs[prepared] = format.Prepare(parts[p].LaneId, x, m, k);
                    kinds[prepared] = kind;
                    slot = prepared++;
                }

                slots[p] = slot;
            }

            RowTiles<TF, TK>(format, parts, inputs, slots, m, k);
        }
        finally
        {
            for (int i = 0; i < prepared; i++)
            {
                inputs[i].Release();
            }
        }
    }

    // Stored bytes a tile takes with one row of x: columns of about this many bytes (small enough that the last tiles keep
    // every thread busy to the end). With more rows a tile is a group of stored rows of a format's own bytes
    // (IRowFormat.GroupBytes: the rows a tile's loop keeps in the cache while it passes the rows of x over them).
    private const int RowTileBytes = 16384;

    // Rows of x a tile of a group of stored rows takes (GroupTile: the tiles split x's rows as well as the columns).
    private const int RowTileRows = 32;

    // A tile of a group of stored rows (`groupBytes`) by RowTileRows rows of x.
    private static (int Columns, int Rows) GroupTile(int total, nint rowBytes, int groupBytes) => (RowGroup(rowBytes, total, groupBytes), RowTileRows);

    // The most stored rows a group of a tile's loop takes (its four rows' partial sums on the stack).
    private const int MostGroupRows = 64;

    // The most rows' sums a block of rows of x by a group of stored rows keeps (IRowKernels.Block: 32 bytes each, on the stack).
    private const int MostBlockSums = 1024;

    // The stored rows a group of a tile's loop takes: those within `groupBytes`, at least one, at most MostGroupRows.
    private static int RowGroup(nint rowBytes, int columns, int groupBytes) =>
        (int)Math.Clamp(groupBytes / Math.Max(1, rowBytes), 1, Math.Clamp(columns, 1, MostGroupRows));

    // The tiles of `parts` in one range over the threads (CpuRanges, a tile a chunk): columns and rows of x, each part's
    // tiles after the other's; each output its own product's, whichever thread runs it. One row of x: columns in whole cache
    // lines of y (the machine's line's floats), so tiles on other threads do not write into the same lines as each output
    // is written. More rows: the format's tile (IRowFormat.Tile). Part p reads x as inputs[slots[p]].
    private static void RowTiles<TF, TK>(TF format, ReadOnlySpan<RowPart> parts, ThreeInputs inputs, ReadOnlySpan<int> slots, int m, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        int total = 0;
        foreach (var part in parts)
        {
            total += part.N;
        }

        int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
        long rowBytes = Math.Max(1, parts[0].Lane.RowBytes);
        int lineFloats = Math.Max(1, LineBytes / sizeof(float));
        // One row of x: RowTileBytes of stored rows a tile, but at most a quarter of a thread's share (as the prompts' tiles
        // are), so a pass of few columns still gives each thread several tiles and the slower cores' last ones keep no other
        // waiting (tq1_0 at 1x2048->2048: 48 columns a tile, 43 tiles for 24 threads, 0.021 ms; 32 columns, 64 tiles,
        // 0.016); at most 64 tiles a thread.
        long share = (total + 4L * threads - 1) / (4L * threads);
        var (wanted, rows) = m == 1
            ? ((Math.Max(Math.Min(RowTileBytes / rowBytes, share), (total + 64L * threads - 1) / (64L * threads)) + lineFloats - 1) / lineFloats * lineFloats, 1)
            : format.Tile(total, m, threads, (nint)rowBytes);
        int columns = (int)Math.Clamp(wanted, 1, total);
        rows = Math.Clamp(rows, 1, m);
        int rowBlocks = (m + rows - 1) / rows;
        int t1 = (parts[0].N + columns - 1) / columns * rowBlocks;
        int t2 = parts.Length > 1 ? t1 + (parts[1].N + columns - 1) / columns * rowBlocks : t1;
        int tiles = parts.Length > 2 ? t2 + (parts[2].N + columns - 1) / columns * rowBlocks : t2;
        var body = new RowTileRange<TF, TK>(format, m, k, columns, rows, rowBlocks, t1, t2,
            parts[0], inputs[slots[0]], parts.Length > 1 ? parts[1] : default, parts.Length > 1 ? inputs[slots[1]] : default,
            parts.Length > 2 ? parts[2] : default, parts.Length > 2 ? inputs[slots[2]] : default);
        CpuRanges.Run(tiles, 1, tiles, threads, ref body);                             // `body` is this frame's, on the stack
    }

    // RowTiles' range: tile t of part p (its tiles from t_p) is its column block t / rowBlocks and its block of rows of x
    // t mod rowBlocks, by the part's lane and that lane's x.
    private readonly struct RowTileRange<TF, TK>(TF format, int m, int k, int columns, int rows, int rowBlocks, int t1, int t2,
        RowPart p0, LaneInput x0, RowPart p1, LaneInput x1, RowPart p2, LaneInput x2) : IRangeBody
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        public void Run(int from, int to)
        {
            for (int t = from; t < to; t++)
            {
                var (part, input, local) = t < t1 ? (p0, x0, t) : t < t2 ? (p1, x1, t - t1) : (p2, x2, t - t2);
                int j0 = local / rowBlocks * columns, i0 = local % rowBlocks * rows;
                FormatColumns<TF, TK>(format, part, input, k, j0, Math.Min(part.N, j0 + columns), i0, Math.Min(m, i0 + rows));
            }
        }
    }

    // Columns [j0, j1) of a part's y for rows [i0, i1) of x by the part's lane (RowGroups with the format's kernels): the
    // scratch the kernels take (its bytes cleared), pinned with the operands for the tile.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void FormatColumns<TF, TK>(TF format, in RowPart part, LaneInput input, int k, int j0, int j1, int i0, int i1)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        var lane = part.Lane;
        nint rowBytes = (nint)lane.RowBytes;
        int bytes = format.Scratch(k);
        byte[]? byteArray = bytes > 0 ? ArrayPool<byte>.Shared.Rent(bytes) : null;
        try
        {
            fixed (float* words = lane.Words)
            fixed (float* xv = input.Values)
            fixed (sbyte* xc = input.Codes)
            fixed (float* xs = input.Scales)
            fixed (byte* scratchBytes = byteArray)
            {
                if (scratchBytes is not null)
                {
                    new Span<byte>(scratchBytes, bytes).Clear();
                }

                var kernels = format.Kernels(part.LaneId, k, new LanePointers(xv, xc + input.CodesAt, xs + input.ScalesAt, input.Natural), scratchBytes);
                RowGroups(ref kernels, (byte*)words + lane.Offset, rowBytes, k, format.Span(k), format.GroupBytes, part.Y, part.N, j0, j1, i0, i1);
                if (lane.RowScale is { } scale)
                {
                    // The codes alone multiplied: each output times its row's scale.
                    float[] y = part.Y;
                    int n = part.N;
                    for (int c = j0; c < j1; c++)
                    {
                        float factor = RowScaleAt(scale, (byte*)words + c * rowBytes, (int)rowBytes);
                        for (int i = i0; i < i1; i++)
                        {
                            y[(long)i * n + c] *= factor;
                        }
                    }
                }
            }
        }
        finally
        {
            if (byteArray is not null)
            {
                ArrayPool<byte>.Shared.Return(byteArray);
            }
        }
    }

    // Columns [j0, j1) of y (n a row) for rows [i0, i1) of x (k a row) by `kernels` over the stored rows at `stored`. One row
    // of x: each stored row read once, in order. More: a group of stored rows (RowGroup, `groupBytes`) at a time, four rows
    // of x at a time against each of the group's rows (each row's values made in registers again for each four), the rows
    // left over one at a time.
    // The four rows of x pass over the group's rows `span` values of k at a time (all of k where span ≥ k), so that much of
    // them stays in the cache across the group, and the group's stored rows across the rows of x; a span's partial sums are
    // added in order. Compiled optimized from its first call,
    // as FormatColumns and RowTileRange.Run are: a format's kernels inline into it only when it is optimized, and the
    // first passes of a run would otherwise call them through unoptimized code.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowGroups<TK>(ref TK kernels, byte* stored, nint rowBytes, int k, int span, int groupBytes, float[] y, int n,
        int j0, int j1, int i0, int i1)
        where TK : struct, IRowKernels
    {
        if (i1 - i0 == 1)
        {
            for (int c = j0; c < j1; c++)
            {
                y[i0 * n + c] = kernels.Single(stored + c * rowBytes, i0);
            }

            return;
        }

        int block = kernels.Block;
        if (block > 0)
        {
            // Blocks of rows of x by groups of stored rows (the kernels' tiles: each stored row's values made in registers once
            // a block of rows of x, a group of values at a time, the group's x staying in the cache across the stored rows).
            int columns = Math.Min(RowGroup(rowBytes, j1 - j0, groupBytes), Math.Max(1, MostBlockSums / block));
            byte* pairRoom = stackalloc byte[32 * columns * block + LineBytes - 1];
            byte* pairSums = (byte*)LineAbove((nint)pairRoom);                         // whole cache lines: no sum split across two
            float* made = stackalloc float[columns * block];
            for (int i = i0; i < i1; i += block)
            {
                int rows = Math.Min(block, i1 - i);
                for (int c0 = j0; c0 < j1; c0 += columns)
                {
                    int count = Math.Min(columns, j1 - c0);
                    kernels.Tile(stored + c0 * rowBytes, rowBytes, count, i, rows, pairSums, made);
                    // (A row of y at a time: its outputs side by side, where a column at a time wrote one a row, n floats apart.)
                    for (int r = 0; r < rows; r++)
                    {
                        float* from = made + r;
                        int at = (i + r) * n + c0;
                        for (int c = 0; c < count; c++)
                        {
                            y[at + c] = from[c * rows];
                        }
                    }
                }
            }

            return;
        }

        int group = RowGroup(rowBytes, j1 - j0, groupBytes);
        float* sums = stackalloc float[8 * MostGroupRows];
        float* eight = stackalloc float[8];

        // Eight rows of x pass over half the values four would: the same bytes of x in the cache.
        int span8 = Math.Max(1, span / 2);
        bool eights = kernels.Eights;
        for (int c0 = j0; c0 < j1; c0 += group)
        {
            int count = Math.Min(group, j1 - c0);
            int i = i0;
            for (; eights && i + 8 <= i1; i += 8)
            {
                // Spans of k (all of k where span8 ≥ k) by each stored row of the group: each span's partial sums added in
                // order, the outputs written at the end.
                new Span<float>(sums, 8 * count).Clear();
                for (int p0 = 0; p0 < k; p0 += span8)
                {
                    int p1 = Math.Min(k, p0 + span8);
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Eight(stored + (c0 + c) * rowBytes, i, p0, p1, eight);
                        float* into = sums + 8 * c;
                        for (int r = 0; r < 8; r++)
                        {
                            into[r] += eight[r];
                        }
                    }
                }

                for (int r = 0; r < 8; r++)
                {
                    int at = (i + r) * n + c0;
                    for (int c = 0; c < count; c++)
                    {
                        y[at + c] = sums[8 * c + r];
                    }
                }
            }

            for (; i + 4 <= i1; i += 4)
            {
                if (span >= k)
                {
                    // One pass over k: each output written as its four rows are done.
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Four(stored + (c0 + c) * rowBytes, i, 0, k, out float s0, out float s1, out float s2, out float s3);
                        int at = i * n + c0 + c;
                        y[at] = s0;
                        y[at + n] = s1;
                        y[at + 2 * n] = s2;
                        y[at + 3 * n] = s3;
                    }

                    continue;
                }

                // Passes over k a span at a time: each span's partial sums added in order, the outputs written at the end.
                new Span<float>(sums, 4 * count).Clear();
                for (int p0 = 0; p0 < k; p0 += span)
                {
                    int p1 = Math.Min(k, p0 + span);
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Four(stored + (c0 + c) * rowBytes, i, p0, p1, out float s0, out float s1, out float s2, out float s3);
                        sums[4 * c] += s0;
                        sums[4 * c + 1] += s1;
                        sums[4 * c + 2] += s2;
                        sums[4 * c + 3] += s3;
                    }
                }

                for (int c = 0; c < count; c++)
                {
                    int at = i * n + c0 + c;
                    y[at] = sums[4 * c];
                    y[at + n] = sums[4 * c + 1];
                    y[at + 2 * n] = sums[4 * c + 2];
                    y[at + 3 * n] = sums[4 * c + 3];
                }
            }

            for (; i < i1; i++)
            {
                for (int c = 0; c < count; c++)
                {
                    y[i * n + c0 + c] = kernels.Single(stored + (c0 + c) * rowBytes, i);
                }
            }
        }
    }

    // ------------------------------------------------------------------ helpers of the described lane's kernels

    // Stored bytes a group of rows of codes read through a table takes with more than one row of x (RowGroups).
    private const int ScaledGroupBytes = 16384;

    // Bytes 8-15 of v in its low 8 (the rest as they come: only the low 8 are read).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<sbyte> UpperEight(Vector128<sbyte> v) =>
        Vector128.Shuffle(v, Vector128.Create((sbyte)8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));

    // Four float32 values (at `at`) as float64, exactly.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> FloatsToDoubles(float* at)
    {
        var v = Vector128.Load(at);
        return Vector256.IsHardwareAccelerated ? Vector256.WidenLower(v.ToVector256Unsafe()) : Vector256.Create(Vector128.WidenLower(v), Vector128.WidenUpper(v));
    }

    // x as float64 for the float64 lane (each value exactly), row by row, over the threads where x is large; `into` holds
    // the doubles' bytes (two floats' room a value).
    private static unsafe void WidenRows(float[] x, float[] into, int m, int k)
    {
        static void Row(ReadOnlySpan<float> from, Span<double> to)
        {
            int p = 0;
            fixed (float* f = from)
            fixed (double* t = to)
            {
                for (; p + 4 <= from.Length; p += 4)
                {
                    FloatsToDoubles(f + p).Store(t + p);
                }
            }

            for (; p < from.Length; p++)
            {
                to[p] = from[p];
            }
        }

        var doubles = MemoryMarshal.Cast<float, double>(into.AsSpan(0, 2 * m * k));
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            fixed (float* wide = into)
            {
                nint at = (nint)wide;
                Parallel.For(0, m, ComputeResources.ParallelOptions, i => Row(x.AsSpan(i * k, k), new Span<double>((double*)at + (nint)i * k, k)));
            }

            return;
        }

        for (int i = 0; i < m; i++)
        {
            Row(x.AsSpan(i * k, k), doubles.Slice(i * k, k));
        }
    }

    // Four int64 codes (at `at`) as float64: exact up to 2^53, else rounded once (the high 16 bits and the low 48 each made
    // a float64 exactly through the bits of 3 · 2^67 and 2^52, then added).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> LongsToDoubles(long* at)
    {
        var v = Vector256.Load(at);
        var high = (Vector256.ShiftRightArithmetic(v.AsInt32(), 16).AsInt64() & Vector256.Create(unchecked((long)0xFFFFFFFF00000000))) + Vector256.Create(442721857769029238784.0).AsInt64();
        var low = ((v & Vector256.Create(0x0000FFFFFFFFFFFFL)) | Vector256.Create(4503599627370496.0).AsInt64()).AsDouble();
        return (high.AsDouble() - Vector256.Create(442726361368656609280.0)) + low;
    }

    // LongsToDoubles over two codes (the same operations, lane by lane).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<double> LongsToDoubles128(long* at)
    {
        var v = Vector128.Load(at);
        var high = ((v.AsInt32() >> 16).AsInt64() & Vector128.Create(unchecked((long)0xFFFFFFFF00000000))) + Vector128.Create(442721857769029238784.0).AsInt64();
        var low = ((v & Vector128.Create(0x0000FFFFFFFFFFFFL)) | Vector128.Create(4503599627370496.0).AsInt64()).AsDouble();
        return (high.AsDouble() - Vector128.Create(442726361368656609280.0)) + low;
    }

    // ShuffleHalves' controls: each 128-bit half's 4 × 4 bytes transposed (its byte 4j + i to 4i + j); each half's even bytes of
    // each 8, then its odd ones (two 16-bit codes' bytes apart). Fields, so code where 256-bit vectors are not accelerated
    // loads them rather than building them in each loop (the JIT takes them as constants where they are).
    private static readonly Vector256<byte> TransposeQuads =
        Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31);

    private static readonly Vector256<byte> EvenOddPairs =
        Vector256.Create((byte)0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15, 16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31);

    // The lanes of a group of `count` blocks (all for a whole group of 8): lane l set where l < count.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> GroupMask(int count) => Vector256.LessThan(Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7), Vector256.Create(count));

    // ------------------------------------------------------------------ the described lane: any type from its description

    // One lane for every type (DescribedFormat), reading nothing but the type's description (TypeDescription: a
    // block's values and bytes; each value's code from its bit planes, an integer, a float of its bits or a table's entry; a
    // code offset; the scales and minimums covering it, each a number of its own type), compiled once a description into a
    // plan (DescribedPlan) of what the description implies, never the type's name and no code of a type's own, and the same
    // plan on every CPU. The plan takes the arithmetic the values allow, and the lane decodes them in registers inside the
    // dot products, writing no weight's value anywhere, whole or a panel:
    // - integer: codes of at most 8 bits whose numbers are whole multiples of one unit spanning at most 255 units, or whole
    //   numbers of 2, 4 or 8 bytes (each byte a plane of its own): x rounded to int8 a block of 32 values, each code made an
    //   unsigned byte in registers (its plane's bits shifted and masked, then its number less the least: as it is, its sign
    //   bit flipped or by a byte shuffle), 8 chunks' codes transposed into the int8 dot products' lanes (or, a chunk's codes
    //   16 bytes or fewer, gathered straight into them: GatheredSteps), the least added back through x's code sums; the
    //   scales and minimums of the 8 chunks gathered and read by their number types at once (an integer, a table of an 8-bit
    //   type's numbers, a 16-bit float by its bits, its pieces joined first); where every span's codes load alike, the load
    //   and the map are the kernels' type parameters (UniformOne, LoadedGroup, GatheredRunsGroup); a wide integer's byte planes are made by
    //   its plane count's own step (IStepLoad.Plane); a plane of a radix's digits by its runs' bytes times each run's power
    //   of the radix (DigitsOf); a codebook's entries by an index of up to 16 bits over any planes, each plane's field read
    //   from a word and the fields joined (Entry); a block of rows of x makes each group's steps, scales and minimums in
    //   registers once and dots every row with them (DescribedPlan.TileKernel), a row alone as it reads them (OneKernel);
    // - float32: values a float32 holds (16-bit floats by their bits, 16-bit integers, float32, codes of up to 8 bits
    //   through a table of their numbers times their scales), float32 FMAs with x;
    // - float64: 64-bit floats as stored, 32- and 64-bit integers made float64 in registers, FMAs with x in float64 (no
    //   narrowing of a 64-bit value to float32).
    // Each by the portable instructions and by each instruction set the CPU has that the arithmetic takes (IInstructions).

    // How the described lane computes, by what the type's values allow (DescribedPlan.Make).
    internal enum DescribedMode : byte
    {
        Integer,
        Single,
        Double,
    }

    // How the integer lane makes a code an unsigned byte: as it is, its sign bit flipped, through a 16-entry byte shuffle, or
    // through a 256-entry table a byte at a time.
    internal enum CodeMap : byte
    {
        Same,
        Flip,
        Shuffle,
        Table,

        // A codebook: each entry's index read and its values' bytes copied from the plan's table, the sign vectors applied
        // as bytes, then made unsigned (DescribedPlan.CodebookFast, CodebookAny).
        Codebook,
    }

    // How a block's scale or minimum reads (DescribedPlan.Factor): an integer (unsigned, two's complement), through a table
    // of its numbers (8 bits or fewer), a 16-bit float by its bits, a float32 as it is, or one lane at a time by its type.
    private enum FactorRead : byte
    {
        Unsigned,
        Signed,
        Table,
        Half,
        Single,
        Scalar,
    }

    // How the float lanes make a stored value: a 16-bit float by its bits, a 16-bit integer, a float32 as it is, codes of up
    // to 8 bits through a table of their numbers (float32); a float64 as it is, a 64-bit or 32-bit integer (float64).
    internal enum WideRead : byte
    {
        Half,
        Short,
        Single,
        Codes,
        Double,
        Long,
        Int,
    }

    /// <summary>
    /// What the described lane makes of a type's description (<see cref="TypeDescription"/>), once a description and alike
    /// on every CPU: the arithmetic its values allow; the layout's period (whole blocks, spans of 32 values and groups of 8
    /// chunks); for codes of at most 8 bits, each span's runs of bytes a plane's fields fill (one load, a shift and a mask a
    /// plane) and the order they give the span's values; how a code becomes the integer lane's byte; each value of the
    /// period's place in x's layout; for each scale and minimum, the 8 lanes' bytes, shifts and masks at each group of the
    /// period and how its number reads. Null where the description is one the lane cannot read in registers (codes wider
    /// than 8 bits that are not one plane of whole bytes a value; a codebook's index of more than 16 bits, or of a radix's
    /// digits).
    /// </summary>
    internal sealed unsafe class DescribedPlan
    {
        // The pinned arrays the kernels read through pointers, kept alive with the plan.
        private readonly List<Array> held = [];

        /// <summary>
        /// The lanes measured for this plan's shapes (Chosen, Measure): the plan's own, held with it by its description (an
        /// equal description made apart measures its own), by shape alone (the vector paths, the rows' class, n and k), each
        /// with the generation it was measured in (a lane of an earlier one, forgotten, is measured again).
        /// </summary>
        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Threads, int Rows, int N, int K), (int Lane, int Generation)> Chosen { get; } = new();

        private DescribedPlan(TypeDescription d)
        {
            D = d;
            Values = d.Values;
            Bytes = d.Bytes;
        }

        public TypeDescription D { get; }

        /// <summary>Values a block holds.</summary>
        public int Values { get; }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes { get; }

        public DescribedMode Mode { get; private set; }

        /// <summary>The float lanes' reading of a value.</summary>
        public WideRead Wide { get; private set; }

        /// <summary>Values and bytes of the layout's period: whole blocks, spans of 32 values and groups.</summary>
        public int Period { get; private set; }

        public int PeriodBytes { get; private set; }

        /// <summary>Byte planes of a code in the integer lane: 1, or a wide integer's bytes (2, 4, 8).</summary>
        public int P { get; private set; } = 1;

        /// <summary>Values a chunk takes (one scale and one lane of the dot products).</summary>
        public int C { get; private set; } = 32;

        /// <summary>Chunks a group takes (8 / P).</summary>
        public int Cg { get; private set; } = 8;

        /// <summary>The integer lane's steps a group takes (C / 4).</summary>
        public int T { get; private set; } = 8;

        /// <summary>Values a group takes (Cg · C).</summary>
        public int GroupValues { get; private set; } = 256;

        /// <summary>Groups a period takes.</summary>
        public int GroupsPerPeriod { get; private set; }

        /// <summary>The integer lane's codes are its numbers over this.</summary>
        public float Unit { get; private set; } = 1f;

        /// <summary>
        /// Whether the integer lane's unsigned codes are at most 127, so AVX2's multiply-adds take them (one byte plane: its
        /// AVX2 lane among the candidates).
        /// </summary>
        public bool PairsFit { get; private set; }

        /// <summary>Whether the type has minimums (a float added a chunk: x's sums times them).</summary>
        public bool Mins { get; private set; }

        /// <summary>The floats of x's lane data a group takes in the integer lane: scales, code-sum starts, then (minimums) sums.</summary>
        public int LaneFloats { get; private set; }

        /// <summary>A wide integer's top byte plane's lanes (all bits set), where its code-sum start goes.</summary>
        public Vector256<int> TopLanes { get; private set; }

        /// <summary>One scale of one piece and no bias or minimums, read as a 16-bit float by its bits (1) or through its table (2); else 0.</summary>
        internal int OneScale => oneScale;

        private int oneScale;

        // OneScale: a 16-bit float scale's reading as vectors, the unit folded in; a table scale's numbers times the unit.
        private HalfVectors oneHalf;
        private float* oneTable;

        /// <summary>Each lane's weight in the final sum (2^(8p) for byte plane p), or null: every lane 1.</summary>
        public Vector256<float>? LaneWeights { get; private set; }

        /// <summary>Code planes (codes of at most 8 bits).</summary>
        public int Planes { get; private set; }

        public CodeMap Map { get; private set; }

        public Vector256<byte> Flip { get; private set; }

        public Vector256<byte> Shuffle { get; private set; }

        /// <summary>The bytes the top byte plane of a signed wide integer is flipped by (0x80 in its lanes).</summary>
        public Vector256<byte> TopFlip { get; private set; }

        /// <summary>
        /// Where every span's codes are one plane loaded the same way (1, 2, 4 or 8 runs) and mapped by a flip or a 16-entry
        /// shuffle or not at all: the runs (the kernels then take that load and that map as type parameters); else 0.
        /// </summary>
        public int UniformRuns { get; private set; }

        /// <summary>Each span's run's first byte (UniformRuns).</summary>
        public int* RunOffsets => runOffset;

        /// <summary>Each span's run's shifts (UniformRuns), per 64-bit lane (per 32-bit lane for 8 runs).</summary>
        public Vector256<ulong>* RunShifts => runShift;

        /// <summary>The same shifts as numbers, 8 a span (RunShifts' lanes: the first 4 for fewer than 8 runs).</summary>
        public int* RunCounts => runCount;

        /// <summary>The one plane's mask (UniformRuns).</summary>
        public Vector256<byte> PlaneMask0 { get; private set; }

        /// <summary>The map's vector: the flip, or the shuffle's table (UniformRuns).</summary>
        public Vector256<byte> MapVector { get; private set; }

        /// <summary>
        /// Whether a group's steps are gathered (UniformRuns of 2 or more, chunks of 32: each chunk's run is 16 bytes or fewer,
        /// so dword d of the 8 chunks' runs is one gather, and each run's shift of it a step: step q · (n / 4) + d), rather than
        /// loaded a chunk at a time and transposed.
        /// </summary>
        public bool Gathered { get; private set; }

        /// <summary>Gathered with flipped codes: each byte's fields' top bits, XORed into the gathered words (the steps then mapped as they are).</summary>
        public Vector256<uint> WordFlip { get; private set; }

        /// <summary>Gathered: per group of the period, the 8 chunks' runs' first bytes, 8 a group.</summary>
        public int* GatherOffsets { get; private set; }

        /// <summary>Gathered: per group of the period and run, the 8 chunks' shifts of that run, a lane each.</summary>
        public Vector256<uint>* GatherShifts { get; private set; }

        /// <summary>Gathered: per group of the period and run, the shift where every chunk's is the same, else −1.</summary>
        public int* GatherShift { get; private set; }

        /// <summary>
        /// Per span of the period, the two byte shuffles that put a block of 32 of x's codes (value order) in the span's order
        /// (from its low 16 values, from its high 16: OrderShuffles[2r], [2r + 1]), or null where every span is in value order.
        /// </summary>
        public Vector256<byte>* OrderShuffles { get; private set; }

        // The shuffles of OrderShuffles, once Order is known.
        private void Shuffles()
        {
            if (Identity)
            {
                return;
            }

            int spans = Period / 32;
            OrderShuffles = Pin<Vector256<byte>>(2 * spans);
            Span<byte> low = stackalloc byte[32], high = stackalloc byte[32];
            for (int r = 0; r < spans; r++)
            {
                for (int t = 0; t < 32; t++)
                {
                    int from = Order[32 * r + t] - 32 * r;
                    low[t] = from < 16 ? (byte)from : (byte)0x80;
                    high[t] = from >= 16 ? (byte)(from - 16) : (byte)0x80;
                }

                OrderShuffles[2 * r] = Vector256.Create<byte>(low);
                OrderShuffles[2 * r + 1] = Vector256.Create<byte>(high);
            }
        }

        /// <summary>Each byte plane's least, added through x's code sums (plane 0: the code offset; the top: −128 where signed).</summary>
        public int* PlaneLeast { get; private set; }

        /// <summary>Each value of the period's byte in x's integer layout.</summary>
        public int* XPlace { get; private set; }

        /// <summary>Each value of the period's chunk.</summary>
        public int* ChunkOf { get; private set; }

        /// <summary>Position t of span r's value (32r + t: the order its codes are made in).</summary>
        public int* Order { get; private set; }

        /// <summary>Whether every span's order is value order (the float lanes read x as it is).</summary>
        public bool Identity { get; private set; }

        /// <summary>The float lanes' table of each code's number (2^bits floats: the number plus the code offset).</summary>
        public float* CodeValues { get; private set; }

        /// <summary>Whether the type has no scales or minimums (a code's number its value: the float lanes skip the factors).</summary>
        public bool Plain => factors == 0;

        /// <summary>Whether a row is float32 values as they are (float32's own bits, no offset, no factor): the float product reads them too.</summary>
        public bool FloatRows => Mode == DescribedMode.Single && Wide == WideRead.Single && Offset == 0 && Plain;

        /// <summary>The 16-bit float reading: its sign shift, magnitude mask, mantissa shift, the power of two it is times, and where its specials start (0: none).</summary>
        public (int SignShift, int Magnitude, int Shift, float Times, int Special) Half16 { get; private set; }

        /// <summary>Half16 as vectors made once (HalfVectors, no factor): the float32 lane's reading of 16-bit floats (ScaledHalfRead).</summary>
        public HalfVectors Half16Vectors { get; private set; }

        /// <summary>Added to a wide value's number (the code offset).</summary>
        public double Offset { get; private set; }

        /// <summary>Whether a wide integer is signed.</summary>
        public bool Signed { get; private set; }

        // Codes of at most 8 bits: per span of the period and plane, the run's first byte, how many runs (1, 2, 4, 8; 0: read
        // one value at a time) and their shifts (per 64-bit lane, or per 32-bit lane for 8 runs; and as numbers, 8 an entry);
        // per plane its mask and low bit.
        private int* runOffset;
        private byte* runKind;
        private Vector256<ulong>* runShift;
        private int* runCount;
        private Vector256<byte>* planeMask;
        private int* planeLow;
        private byte* byteTable;

        // Scales then minimums: count, scales' count, pieces at most; per factor its reading, bits, bias, pieces and table;
        // per group of the period, factor and piece, the lanes' offsets (as numbers, 8 an entry), shifts, masks and lefts
        // (and the shift and the left where every lane's is the same, else −1).
        private int factors, scaleFactors, pieces;
        private FactorRead[] reads = [];
        private int[] factorBits = [], factorPieces = [];
        private float[] factorBias = [];
        private float*[] factorTables = null!;
        private (int SignShift, int Magnitude, int Shift, float Times, int Special)[] factorHalves = [];
        private int* fOffset;
        private Vector256<int>* fShift;
        private Vector256<int>* fMask;
        private Vector256<int>* fLeft;
        private int* fShiftOne;
        private int* fLeftOne;

        // Per group of the period and factor, whether every lane reads the same bits (a level of the block's or the group's
        // own: a super-scale): then read once by scalar loads and broadcast, no gather; per piece its mask as a number.
        private byte* fUniform;
        private int* fMaskOne;

        // The factors read a lane at a time (FactorRead.Scalar: ReadOneByOne): per factor its number's reading
        // (ScalarNumber); per group of the period, factor and lane its bits' pieces, scalarPieces an entry, each a byte of
        // the period (from its first), a shift, a mask and the place its bits go (a piece past a lane's own has mask 0).
        // Places and the description's numbers, made once.
        private ScalarNumber* scalarNumbers;
        private int scalarPieces;
        private int* sOffset;
        private int* sShift;
        private int* sMask;
        private int* sLeft;

        // The factors a chunk takes (the scales, then the minimums; a codebook's vector factors are not among them).
        private List<BlockScale> factorList = [];

        // A codebook (CodeMap.Codebook): each entry's values as the integer lane's signed bytes; the bytes adding the least
        // back; per span each entry's index field (its byte from the period's first, its shift) and the indices' mask; per
        // vector factor its width, its masks (0xFF where its value is -1, an entry a width) and per span and group of it its
        // index's pieces (from vecStart: byte, shift, mask, place).
        private sbyte* entryBytes;
        private Vector256<byte> leastBytes;
        private int entryWidth = 1, entryMask;
        private int* entryAt;
        private int* entryShift;

        // An index wider than a byte's field (CodebookSpans): per entry and plane its 32-bit word's first byte and shift; per
        // plane its mask and its bits' place in the index.
        private bool entryWide;
        private uint* entryMasks;
        private int* entryLows;
        private int vectorCount;
        private int[] vectorWidth = [];
        private byte*[] vectorMasks = [];
        private int[] vecStart = [], vecOffset = [], vecShift = [], vecMask = [], vecPlace = [];

        // The codebook's fast reading (entries of 4 or 8 values; at most one vector factor, of groups of 8 whose every index
        // lies within 16 bits of its first byte): per span and group the index's first byte and shift, and its mask.
        private bool codebookFast;
        private int* signAt;
        private int* signShift;
        private int signMask;

        /// <summary>Groups a row of <paramref name="k"/> values takes (the last may be short).</summary>
        public int Groups(int k) => (k + GroupValues - 1) / GroupValues;

        /// <summary>
        /// Bytes a row of x takes as the integer lane's int8 codes (RoundDescribedRow): its whole groups, then a cache line
        /// (the machine's: CpuInfo), so the rows of a block of x (a whole group a multiple of 256 bytes) do not all fall in
        /// the same sets of the cache.
        /// </summary>
        public int CodesRow(int k) => Groups(k) * GroupValues + LineBytes;

        /// <summary>Floats a row of x's lane data takes in the integer lane (LaneFloats a group, then a cache line's floats).</summary>
        public int LanesRow(int k) => Groups(k) * LaneFloats + LineBytes / sizeof(float);

        /// <summary>
        /// Rows of x the integer lane's tiles take at once (IRowKernels.Block): as many as one group's codes and lane data of
        /// them fit DescribedBlockBytes, a multiple of 4 from 4 to 128.
        /// </summary>
        public int BlockRows { get; private set; }

        // A pinned array of n T's, held by the plan.
        private T* Pin<T>(int n) where T : unmanaged
        {
            var array = GC.AllocateArray<T>(Math.Max(1, n), pinned: true);
            held.Add(array);
            return (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
        }

        /// <summary>
        /// The plan of <paramref name="d"/>, or null where the lane does not read it: made once a description, by the
        /// description itself (<see cref="TypeDescription.CpuPlan"/>), which holds it.
        /// </summary>
        internal static DescribedPlan? Make(TypeDescription d)
        {
            if (!(BitConverter.IsLittleEndian && d.Valid))
            {
                return null;
            }

            var plan = new DescribedPlan(d);
            if (!(plan.Wide64() || plan.Narrow()))
            {
                return null;
            }

            plan.Finish();
            if (plan.StepsGathered && plan.C == 32 && plan.T == 8 && plan.Cg == 8)
            {
                // Its one-row kernel's plan: the same description with a chunk a step (natural), where its steps gather so.
                var own = new DescribedPlan(d) { natural = true };
                if (own.Narrow())
                {
                    own.Finish();
                    plan.Natural = own.StepsGathered ? own : null;
                }
            }

            return plan;
        }

        /// <summary>
        /// The plan a row of x alone is read by (NaturalOne), where this one's steps are gathered (chunks of 32): the same
        /// description, each of its steps one chunk's 32 values in their order (lane l the chunk's 4l..4l + 3), so x's codes
        /// lie a chunk at a time as they are rounded (no transpose: RoundDescribedRow's natural layout) and a step's dwords
        /// are loaded where they lie together in the stored bytes (Load), not gathered; each chunk's dot product summed
        /// across its lanes after (LaneTotals). Null where there is none.
        /// </summary>
        public DescribedPlan? Natural { get; private set; }

        /// <summary>Whether this plan is another's Natural one (its steps a chunk each: StepPosition).</summary>
        public bool IsNatural => natural;

        private bool natural;

        // A code of one plane of 2, 4 or 8 whole bytes a value, no scales: the integer lane over its byte planes where the code
        // is an integer with no offset, else a float lane.
        private bool Wide64()
        {
            var d = D;
            if (d is not { Values: 1, Planes: [{ Offset: 0, Group: 1, Bits: 16 or 32 or 64, Places: null, Radix: 0 } plane], Scales: null or [], Mins: null or [] } || d.Bytes != plane.Bits / 8)
            {
                return false;
            }

            var code = d.Code;
            int bytes = plane.Bits / 8;
            Offset = d.CodeOffset;
            Signed = code.Signed;
            Period = 32;
            PeriodBytes = 32 * bytes;
            if (code.Kind == NumberKind.Integer && d.CodeOffset == 0)
            {
                Mode = DescribedMode.Integer;
                P = bytes;
                C = 32;
                Cg = 8 / P;
                T = 8;
                GroupValues = 32 * Cg;
                Period = GroupValues;
                PeriodBytes = GroupValues * bytes;
                PlaneLeast = Pin<int>(P);
                PlaneLeast[P - 1] = code.Signed ? -128 : 0;
                Span<float> weights = stackalloc float[8];
                Span<byte> flip = stackalloc byte[32];
                Span<int> top = stackalloc int[8];
                for (int l = 0; l < 8; l++)
                {
                    weights[l] = MathF.ScaleB(1f, 8 * (l % P));
                    top[l] = l % P == P - 1 ? -1 : 0;
                    if (code.Signed && l % P == P - 1)
                    {
                        flip[4 * l] = flip[4 * l + 1] = flip[4 * l + 2] = flip[4 * l + 3] = 0x80;
                    }
                }

                LaneWeights = Vector256.Create<float>(weights);
                TopFlip = Vector256.Create<byte>(flip);
                TopLanes = Vector256.Create<int>(top);
                Identity = true;
                Places();
                return true;
            }

            switch (code.Kind, plane.Bits)
            {
                case (NumberKind.Float, 16) when HalfOf(code) is { } half:
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Half;
                    Half16 = half;
                    Half16Vectors = new HalfVectors(half, 1f);
                    return true;
                case (NumberKind.Integer, 16):
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Short;
                    return true;
                case (NumberKind.Float, 32) when code is { Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Specials: FloatSpecials.Ieee, Subnormals: true }:
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Single;
                    return true;
                case (NumberKind.Float, 64) when code is { Exponent: 11, Mantissa: 52, ExponentBias: 1023, Signed: true, Specials: FloatSpecials.Ieee, Subnormals: true }:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Double;
                    return true;
                case (NumberKind.Integer, 64) when code.Signed:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Long;
                    return true;
                case (NumberKind.Integer, 32) when code.Signed:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Int;
                    return true;
                default:
                    return false;
            }
        }

        // A 16-bit float read by its bits (a float32's exponent range or less, subnormals, IEEE's or no specials): the sign's
        // shift to bit 31, the magnitude's mask, the shift putting its exponent at float32's, the power of two that rebiases it
        // (1: none), and the magnitude its specials start at (0: none: they come through as they are, or there are none).
        private static (int SignShift, int Magnitude, int Shift, float Times, int Special)? HalfOf(NumberType t)
        {
            if (t is not { Kind: NumberKind.Float, Bits: 16, Subnormals: true, Specials: FloatSpecials.Ieee or FloatSpecials.None } || t.Exponent > 8
                || t.ExponentBias is <= 0 or >= 254 || t.Exponent == 8 && t.ExponentBias != 127)
            {
                return null;
            }

            int magnitude = t.Signed ? 0x7FFF : 0xFFFF;
            int special = t.Exponent < 8 && t.Specials == FloatSpecials.Ieee ? ((1 << t.Exponent) - 1) << t.Mantissa : 0;
            return (t.Signed ? 16 : 0, magnitude, 23 - t.Mantissa, MathF.ScaleB(1f, 127 - t.ExponentBias), special);
        }

        // Codes of at most 8 bits, from planes of 1, 2, 4 or 8 bits: each span's runs and order, the chunk (one scale a lane)
        // the factors allow, the integer lane where the codes' numbers are whole multiples of a unit within a byte's span, else
        // the float32 lane through a table of the codes' numbers.
        private bool Narrow()
        {
            var d = D;
            bool book = d.HasCodebook;
            // Codes of at most 8 bits from fields of at most 8 (bits, or a radix's digits); a codebook's entry indices of up
            // to 16 bits from fields of bits, each read whole however many planes and bytes it takes (CodebookSpans).
            if (book ? d.Code.Bits > 16 || d.Planes.Any(p => p.Radix != 0) : d.Code.Bits > 8 || d.Planes.Any(p => p.Bits > 8))
            {
                return false;
            }

            // A codebook's vector factors (scales a vector of ±1 each: sign masks) are applied to the integer codes as bytes;
            // every other factor is a chunk's (one number for a chunk's values).
            var vectors = (d.Scales ?? []).Where(f => f.Type.Width > 1).ToList();
            if ((d.Mins ?? []).Any(f => f.Type.Width > 1) || vectors.Any(f => f.Type.Table!.Any(v => v is not (1d or -1d))))
            {
                return false;
            }

            var all = (d.Scales ?? []).Where(f => f.Type.Width == 1).Concat(d.Mins ?? []).ToList();
            factorList = all;
            int chunk = all.All(f => f.Values % 32 == 0) ? 32 : all.All(f => f.Values % 16 == 0) ? 16 : all.All(f => f.Values % 8 == 0) ? 8 : 0;
            if (chunk == 0)
            {
                return false;
            }

            // The codes' numbers (a codebook's: every entry's values, and their negations under sign vectors), and the unit
            // that makes them whole within a byte's span (the integer lanes' dot products take unsigned bytes of any value:
            // IInstructions.Dot).
            int count = 1 << d.Code.Bits;
            var numbers = new double[count];
            for (int c = 0; c < count; c++)
            {
                numbers[c] = d.Code.Value((ulong)c) + d.CodeOffset;
            }

            var entries = book && d.Code.Width > 1 ? d.Code.Table!.Select(v => v + d.CodeOffset).ToArray() : numbers;
            double[] reached = vectors.Count > 0 ? [.. entries, .. entries.Select(v => -v)] : entries;
            int unitShift = -1, least = 0, span = 255;
            for (int s = 0; s <= 7 && unitShift < 0 && chunk >= 16 && reached.All(double.IsFinite); s++)
            {
                var scaled = reached.Select(n => Math.ScaleB(n, s)).ToArray();
                if (scaled.All(v => v == Math.Round(v)) && scaled.Max() - scaled.Min() <= span)
                {
                    unitShift = s;
                    least = (int)scaled.Min();
                    PairsFit = scaled.Max() - scaled.Min() <= 127;
                }
            }

            Mode = unitShift >= 0 ? DescribedMode.Integer : DescribedMode.Single;
            if (book && (Mode != DescribedMode.Integer || 32 % d.Code.Width != 0 || d.Code.Width > 8 || vectors.Any(f => 32 % f.Values != 0 || f.Values > 8)))
            {
                return false;                                                  // a codebook reads in the integer lane alone
            }

            Wide = WideRead.Codes;
            C = chunk;
            Cg = 8;
            T = chunk / 4;
            GroupValues = 8 * chunk;
            Period = Lcm(Lcm(d.UnitValues, 32), GroupValues);
            PeriodBytes = Period / d.UnitValues * d.UnitBytes;
            Planes = d.Planes.Count;
            if (!(book ? CodebookSpans(vectors) : Spans()) || !Places() || !Factors(all, (d.Scales ?? []).Count(f => f.Type.Width == 1)))
            {
                return false;
            }

            if (book)
            {
                // Each entry's values as the integer lane's signed bytes (number · 2^unit), and the bytes that make them unsigned.
                Mins = d.Mins is { Count: > 0 };
                CodeValues = Pin<float>(count);
                Unit = MathF.ScaleB(1f, -unitShift);
                PlaneLeast = Pin<int>(1);
                PlaneLeast[0] = least;
                entryBytes = Pin<sbyte>(entries.Length);
                for (int i = 0; i < entries.Length; i++)
                {
                    entryBytes[i] = (sbyte)Math.ScaleB(entries[i], unitShift);
                }

                leastBytes = Vector256.Create(unchecked((byte)-least));
                Map = CodeMap.Codebook;
                Ready();
                return true;
            }

            Mins = d.Mins is { Count: > 0 };
            CodeValues = Pin<float>(count);
            for (int c = 0; c < count; c++)
            {
                CodeValues[c] = (float)numbers[c];
            }

            if (Mode == DescribedMode.Integer)
            {
                Unit = MathF.ScaleB(1f, -unitShift);
                PlaneLeast = Pin<int>(1);
                PlaneLeast[0] = least;
                var unsigned = new byte[count];
                for (int c = 0; c < count; c++)
                {
                    unsigned[c] = (byte)(Math.ScaleB(numbers[c], unitShift) - least);
                }

                bool same = Enumerable.Range(0, count).All(c => unsigned[c] == c);
                bool flip = Enumerable.Range(0, count).All(c => unsigned[c] == (c ^ count / 2));
                if (same)
                {
                    Map = CodeMap.Same;
                }
                else if (flip)
                {
                    Map = CodeMap.Flip;
                    Flip = Vector256.Create((byte)(count / 2));
                }
                else if (count <= 16)
                {
                    Map = CodeMap.Shuffle;
                    Span<byte> table = stackalloc byte[32];
                    for (int i = 0; i < 32; i++)
                    {
                        table[i] = unsigned[i % 16 % count];
                    }

                    Shuffle = Vector256.Create<byte>(table);
                }
                else
                {
                    // The codes' bytes, made once from the description (each code's number in units less the least, a byte
                    // each): 3 bytes past the 256 so the dword Through reads at any code's lies within it.
                    Map = CodeMap.Table;
                    byteTable = Pin<byte>(256 + 3);
                    for (int i = 0; i < 256; i++)
                    {
                        byteTable[i] = unsigned[i % count];
                    }
                }

                int spans = Period / 32;
                // (A plane of digits is made by Raw alone: its runs' bytes times each run's power of the radix.)
                if (Planes == 1 && d.Planes[0].Radix == 0 && Map != CodeMap.Table && Enumerable.Range(0, spans).All(r => runKind[r] == runKind[0]) && runKind[0] is 1 or 2 or 4 or 8)
                {
                    UniformRuns = runKind[0];
                }

                if (UniformRuns >= 2 && C == 32)
                {
                    // Chunk l of group g is span 8g + l: its run's first byte, and each run q's shift (a 64-bit lane's, or
                    // a 32-bit lane's for 8 runs), and that shift as a number where every chunk's is the same.
                    int runs = UniformRuns, groups = Period / GroupValues;
                    GatherOffsets = Pin<int>(8 * groups);
                    GatherShifts = Pin<Vector256<uint>>(groups * runs);
                    GatherShift = Pin<int>(groups * runs);
                    Span<int> lanes = stackalloc int[8];
                    for (int g = 0; g < groups; g++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            GatherOffsets[8 * g + l] = runOffset[8 * g + l];
                        }

                        for (int q = 0; q < runs; q++)
                        {
                            for (int l = 0; l < 8; l++)
                            {
                                lanes[l] = runCount[8 * (8 * g + l) + (runs == 8 ? q : 4 * q / runs)];
                            }

                            GatherShifts[g * runs + q] = Vector256.Create<int>(lanes).AsUInt32();
                            GatherShift[g * runs + q] = One(lanes);
                        }
                    }

                    Gathered = true;
                    if (Map == CodeMap.Flip)
                    {
                        // Each field's top bit flipped in the gathered words at once (every run's shift of the flip),
                        // rather than in each step.
                        int fieldTops = 0;
                        for (int q = 0; q < runs; q++)
                        {
                            fieldTops |= (count / 2) << runCount[runs == 8 ? q : 4 * q / runs];
                        }

                        WordFlip = Vector256.Create((uint)(fieldTops * 0x01010101));
                        // Every span's runs at the same shifts, or the steps are loaded and transposed (their map then a step at a time).
                        Gathered = Enumerable.Range(0, spans).All(r => runShift[r].Equals(runShift[0]));
                    }
                }
            }

            Ready();
            return true;
        }

        // The one-scale read's vectors and table, once the unit is known (OneScale).
        private void Ready()
        {
            if (oneScale == 1)
            {
                oneHalf = new HalfVectors(factorHalves[0], Unit);
            }
            else if (oneScale == 2)
            {
                oneTable = Pin<float>(256);
                for (int c = 0; c < 256; c++)
                {
                    oneTable[c] = factorTables[0][c] * Unit;
                }
            }
        }

        // What the kernels read of the plan on every row, stored once it is made.
        private void Finish()
        {
            GroupsPerPeriod = Period / GroupValues;
            LaneFloats = P > 1 ? 2 * Cg : Mins ? 24 : 16;
            BlockRows = Math.Clamp(DescribedBlockBytes / (GroupValues + 4 * LaneFloats) & ~3, 4, 128);
            PlaneMask0 = planeMask == null ? default : planeMask[0];
            MapVector = Map == CodeMap.Flip ? Flip : Shuffle;
            oneKernels = Pin<nint>(16);
            tileKernels = Pin<nint>(16);
            if (Mode == DescribedMode.Integer && P == 1 && UniformRuns == 0)
            {
                GatherSteps();                                              // false: the spans made and transposed (Steps)
                if (natural && stepsGathered)
                {
                    Deposits();                                             // false: the steps' own (PlaneSteps)
                }
            }
        }

        // The lanes' one value where every lane's is the same, else −1.
        private static int One(ReadOnlySpan<int> lanes)
        {
            foreach (int lane in lanes)
            {
                if (lane != lanes[0])
                {
                    return -1;
                }
            }

            return lanes[0];
        }

        private static int Lcm(int a, int b) => a / Gcd(a, b) * b;

        private static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);

        // Each span of the period: per plane the bytes its 32 values' fields lie in, as runs of n bytes at one shift each
        // (n · runs = 32, every run over the same bytes): one load, a shift a run and a mask make the span's codes, in the
        // order the runs give; a span whose planes do not fall so (or give other orders) is read one value at a time, in value
        // order.
        private bool Spans()
        {
            var d = D;
            int spans = Period / 32;
            runOffset = Pin<int>(spans * Planes);
            runKind = Pin<byte>(spans * Planes);
            runShift = Pin<Vector256<ulong>>(spans * Planes);
            runCount = Pin<int>(8 * spans * Planes);
            planeMask = Pin<Vector256<byte>>(Planes);
            planeLow = Pin<int>(Planes);
            streamAt = Pin<int>(4 * spans * Planes);
            planeRadix = Pin<int>(Planes);
            planeBits = Pin<int>(Planes);
            digitTimes = d.Planes.Any(p => p.Radix != 0) ? Pin<Vector256<byte>>(spans * Planes) : null;
            Order = Pin<int>(Period);
            int low = 0;
            for (int p = 0; p < Planes; p++)
            {
                planeMask[p] = Vector256.Create((byte)((1 << d.Planes[p].Bits) - 1));
                planeLow[p] = low;
                planeRadix[p] = d.Planes[p].Radix;
                planeBits[p] = d.Planes[p].Bits;
                low += d.Planes[p].Bits;
            }

            Span<int> at = stackalloc int[32];
            Span<int> shift = stackalloc int[32];
            Span<int> mine = stackalloc int[32];
            Span<int> order = stackalloc int[32];
            Span<int> shifts = stackalloc int[8];
            Span<byte> times = stackalloc byte[32];
            for (int r = 0; r < spans; r++)
            {
                bool vector = true;
                for (int p = 0; p < Planes && vector; p++)
                {
                    var plane = d.Planes[p];
                    if (plane.Stream)
                    {
                        // A bit stream (fields across byte boundaries): the span's four groups of 8 fields, each Bits whole
                        // bytes from a byte boundary within one block, made in value order (StreamLoad).
                        for (int q = 0; q < 4 && vector; q++)
                        {
                            int v0 = 32 * r + 8 * q;
                            var (b0, s0) = plane.Place(v0 % d.Values);
                            vector = s0 == 0 && v0 % d.Values + 8 <= d.Values;
                            streamAt[4 * (r * Planes + p) + q] = v0 / d.UnitValues * d.UnitBytes + d.BlockStart(v0) + b0;
                        }

                        for (int t = 0; t < 32; t++)
                        {
                            mine[t] = 32 * r + t;
                        }

                        if (!vector || p > 0 && !mine.SequenceEqual(order))
                        {
                            vector = false;
                            break;
                        }

                        mine.CopyTo(order);
                        runKind[r * Planes + p] = StreamRuns;
                        continue;
                    }

                    // Runs of bytes at one shift (a plane of digits: at one digit, the runs loaded as they are and each
                    // byte's digit made by its run's power of the radix); a listed field running into the next byte is
                    // read one value at a time.
                    int first = int.MaxValue, runs = 0;
                    for (int t = 0; t < 32; t++)
                    {
                        int v = 32 * r + t;
                        var (b, s) = plane.Place(v % d.Values);
                        if (plane.Radix == 0 && s + plane.Bits > 8)
                        {
                            vector = false;
                            break;
                        }

                        at[t] = v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b;
                        shift[t] = s;
                        first = Math.Min(first, at[t]);
                        if (shifts[..runs].IndexOf(s) < 0)
                        {
                            if (runs == 8)
                            {
                                vector = false;
                                break;
                            }

                            shifts[runs++] = s;
                        }
                    }

                    if (!vector || runs is not (1 or 2 or 4 or 8))
                    {
                        vector = false;
                        break;
                    }

                    shifts[..runs].Sort();
                    int n = 32 / runs;
                    mine.Fill(-1);
                    for (int t = 0; t < 32 && vector; t++)
                    {
                        int q = shifts[..runs].IndexOf(shift[t]), b = at[t] - first;
                        vector = b < n && mine[q * n + b] < 0;
                        if (vector)
                        {
                            mine[q * n + b] = 32 * r + t;
                        }
                    }

                    if (!vector || p > 0 && !mine.SequenceEqual(order))
                    {
                        vector = false;
                        break;
                    }

                    mine.CopyTo(order);
                    int i = r * Planes + p;
                    runOffset[i] = first;
                    runKind[i] = (byte)runs;
                    if (plane.Radix != 0)
                    {
                        // Digits: the runs' bytes loaded unshifted, byte j of the 32 times the radix to its run's digit.
                        for (int j = 0; j < 32; j++)
                        {
                            int power = 1;
                            for (int e = 0; e < shifts[j / n]; e++)
                            {
                                power *= plane.Radix;
                            }

                            times[j] = (byte)power;
                        }

                        digitTimes[i] = Vector256.Create<byte>(times);
                        runShift[i] = default;
                        new Span<int>(runCount + 8 * i, 8).Clear();
                        continue;
                    }

                    // 8 runs: a shift a 32-bit lane; fewer: a shift a 64-bit lane, run q's in lanes [4q / runs, 4(q + 1) / runs).
                    runShift[i] = runs == 8
                        ? Vector256.Create(shifts[0], shifts[1], shifts[2], shifts[3], shifts[4], shifts[5], shifts[6], shifts[7]).AsUInt64()
                        : Vector256.Create((ulong)shifts[0], (ulong)shifts[runs / 4], (ulong)shifts[2 * runs / 4], (ulong)shifts[3 * runs / 4]);
                    for (int l = 0; l < 8; l++)
                    {
                        runCount[8 * i + l] = runs == 8 ? shifts[l] : l < 4 ? shifts[l * runs / 4] : 0;
                    }
                }

                for (int t = 0; t < 32; t++)
                {
                    Order[32 * r + t] = vector ? order[t] : 32 * r + t;
                }

                if (!vector)
                {
                    // Not in runs: in value order from two windows of 16 bytes where every plane's fields fit them (Windows),
                    // else one value at a time.
                    bool windows = Windows(r);
                    for (int p = 0; p < Planes; p++)
                    {
                        runKind[r * Planes + p] = windows ? WindowRuns : (byte)0;
                    }

                    if (!windows)
                    {
                        OneValues(r);
                    }
                }
            }

            Identity = Enumerable.Range(0, Period).All(v => Order[v] == v);
            Shuffles();
            return true;
        }

        // A span's run kind made from two windows of 16 bytes (Spans; Raw reads it by WindowLoad).
        private const byte WindowRuns = 17;

        // A span read one value at a time (OneByOne): per value of the period and plane, the dword its field lies in (its first
        // byte from the period's first, kept within the period) and the field's shift in it (a digit plane's: its byte's), and
        // a digit's power of the radix; per plane its field's mask. Places, made once from the description.
        private int* oneAt;
        private byte* oneShift;
        private int* onePower;
        private uint* oneMask;

        // Span r's values' places, one value at a time (Spans, where neither runs nor windows take the span): the place
        // CodeBits reads each plane's field from.
        private void OneValues(int r)
        {
            var d = D;
            if (oneAt == null)
            {
                oneAt = Pin<int>(Period * Planes);
                oneShift = Pin<byte>(Period * Planes);
                onePower = Pin<int>(Period * Planes);
                oneMask = Pin<uint>(Planes);
                for (int p = 0; p < Planes; p++)
                {
                    oneMask[p] = (1u << d.Planes[p].Bits) - 1;
                }
            }

            for (int t = 0; t < 32; t++)
            {
                int v = 32 * r + t;
                for (int p = 0; p < Planes; p++)
                {
                    var plane = d.Planes[p];
                    var (b, s) = plane.Place(v % d.Values);
                    int at = v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b, kept = Math.Max(0, Math.Min(at, PeriodBytes - 4)), i = v * Planes + p;
                    int power = 1;
                    for (int e = 0; plane.Radix != 0 && e < s; e++)
                    {
                        power *= plane.Radix;
                    }

                    // (A field within the period lies within its kept dword: at most 8 bits from a bit below 8 of its byte.)
                    (oneAt[i], oneShift[i], onePower[i]) = (kept, (byte)(8 * (at - kept) + (plane.Radix != 0 ? 0 : s)), power);
                }
            }
        }

        /// <summary>The period's spans by how their codes are made: in runs, from two windows, as a stream, by a codebook's entries, one value at a time (tests).</summary>
        public (int Runs, int Windows, int Streams, int Codebook, int OneByOne) SpanKinds()
        {
            int spans = Period / 32;
            if (Map == CodeMap.Codebook)
            {
                return (0, 0, 0, spans, 0);
            }

            int runs = 0, windows = 0, streams = 0, one = 0;
            for (int r = 0; runKind != null && r < spans; r++)
            {
                var kinds = new ReadOnlySpan<byte>(runKind + r * Planes, Planes);
                if (kinds.Contains((byte)0))
                {
                    one++;
                }
                else if (kinds.Contains(WindowRuns))
                {
                    windows++;
                }
                else if (kinds.Contains(StreamRuns))
                {
                    streams++;
                }
                else
                {
                    runs++;
                }
            }

            return (runs, windows, streams, 0, one);
        }

        /// <summary>
        /// The plan's readings no built-in type's takes (tests): "one by one" (a span's codes a value at a time: OneByOne),
        /// "table" (codes through the 256-byte table: Through), "codebook" (a codebook the fast reading does not take:
        /// CodebookAny), "scalar" (a scale or minimum a lane at a time: ReadOneByOne).
        /// </summary>
        public IReadOnlyList<string> Fallbacks()
        {
            var taken = new List<string>();
            if (SpanKinds().OneByOne > 0)
            {
                taken.Add("one by one");
            }

            if (Map == CodeMap.Table)
            {
                taken.Add("table");
            }

            if (Map == CodeMap.Codebook && !codebookFast)
            {
                taken.Add("codebook");
            }

            if (reads.Contains(FactorRead.Scalar))
            {
                taken.Add("scalar");
            }

            return taken;
        }

        // Per span and plane: its two windows' first bytes (from the period's first), the byte shuffles taking each value's
        // byte from them (0x80: from the other), and the 16-bit multipliers making its field of the byte (its low 16 values'
        // and its high 16's): ((byte · first) mod 256) · second >> 8.
        private int* windowAt;
        private Vector256<byte>* windowShuffle;
        private Vector256<ushort>* windowTimes;

        // Whether span r's 32 values' fields lie, in every plane, in two windows of 16 bytes of the period (a field within its
        // byte: bits at a shift, or a radix's digit): then each plane's span is two loads, two byte shuffles into value order,
        // and one multiply-mask-multiply per byte (a field of bits at shift s: times 1, then 2^(8 − s); a digit i of a radix
        // r: times r^i, then r), so the span is made in registers like the runs, in value order as one value at a time.
        private bool Windows(int r)
        {
            var d = D;
            if (PeriodBytes < 16)
            {
                return false;
            }

            if (windowAt == null)
            {
                int spans = Period / 32;
                windowAt = Pin<int>(2 * spans * Planes);
                windowShuffle = Pin<Vector256<byte>>(2 * spans * Planes);
                windowTimes = Pin<Vector256<ushort>>(4 * spans * Planes);
            }

            Span<int> at = stackalloc int[32];
            Span<int> shift = stackalloc int[32];
            Span<byte> first = stackalloc byte[32], second = stackalloc byte[32];
            Span<ushort> times = stackalloc ushort[32], then = stackalloc ushort[32];
            for (int p = 0; p < Planes; p++)
            {
                var plane = d.Planes[p];
                if (plane.Stream)
                {
                    return false;
                }

                int lo = int.MaxValue, hi = int.MinValue;
                for (int t = 0; t < 32; t++)
                {
                    int v = 32 * r + t;
                    var (b, s) = plane.Place(v % d.Values);
                    if (plane.Radix == 0 && s + plane.Bits > 8)
                    {
                        return false;
                    }

                    (at[t], shift[t]) = (v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b, s);
                    (lo, hi) = (Math.Min(lo, at[t]), Math.Max(hi, at[t]));
                }

                int a = Math.Min(lo, PeriodBytes - 16), c = Math.Clamp(hi - 15, 0, PeriodBytes - 16);
                for (int t = 0; t < 32; t++)
                {
                    bool fromA = at[t] - a is >= 0 and < 16, fromC = !fromA && at[t] - c is >= 0 and < 16;
                    if (!fromA && !fromC)
                    {
                        return false;
                    }

                    first[t] = fromA ? (byte)(at[t] - a) : (byte)0x80;
                    second[t] = fromC ? (byte)(at[t] - c) : (byte)0x80;
                    int power = 1;
                    for (int e = 0; plane.Radix != 0 && e < shift[t]; e++)
                    {
                        power *= plane.Radix;
                    }

                    (times[t], then[t]) = plane.Radix != 0 ? ((ushort)power, (ushort)plane.Radix) : ((ushort)1, (ushort)(256 >> shift[t]));
                }

                int i = r * Planes + p;
                (windowAt[2 * i], windowAt[2 * i + 1]) = (a, c);
                windowShuffle[2 * i] = Vector256.Create<byte>(first);
                windowShuffle[2 * i + 1] = Vector256.Create<byte>(second);
                windowTimes[4 * i] = Vector256.Create<ushort>(times[..16]);
                windowTimes[4 * i + 1] = Vector256.Create<ushort>(times[16..]);
                windowTimes[4 * i + 2] = Vector256.Create<ushort>(then[..16]);
                windowTimes[4 * i + 3] = Vector256.Create<ushort>(then[16..]);
            }

            return true;
        }

        // A plane's span from its two windows (Windows), its 32 fields as bytes in value order.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> WindowLoad<TI>(byte* period, int i) where TI : struct, IInstructions
        {
            var a = Vector128.Load(period + windowAt[2 * i]);
            var c = Vector128.Load(period + windowAt[2 * i + 1]);
            var bytes = TI.ShuffleBytes(Vector256.Create(a, a), windowShuffle[2 * i]) | TI.ShuffleBytes(Vector256.Create(c, c), windowShuffle[2 * i + 1]);
            var (low, high) = Vector256.Widen(bytes);
            var mask = Vector256.Create((ushort)0xFF);
            Vector256<ushort>* times = windowTimes + 4 * i;
            return Vector256.Narrow(Vector256.ShiftRightLogical((low * times[0] & mask) * times[2], 8), Vector256.ShiftRightLogical((high * times[1] & mask) * times[3], 8));
        }

        // A codebook's spans, in value order: each span's entries' index fields and each vector factor's index pieces of each
        // of its groups (one plane of indices; entries and vector groups dividing a span).
        private bool CodebookSpans(List<BlockScale> vectors)
        {
            var d = D;
            int spans = Period / 32, w = d.Code.Width, perSpan = 32 / w;
            runOffset = Pin<int>(spans * Planes);
            runKind = Pin<byte>(spans * Planes);
            runShift = Pin<Vector256<ulong>>(spans * Planes);
            runCount = Pin<int>(8 * spans * Planes);
            planeMask = Pin<Vector256<byte>>(Planes);
            planeLow = Pin<int>(Planes);
            planeMask[0] = Vector256.Create((byte)((1 << d.Planes[0].Bits) - 1));
            Order = Pin<int>(Period);
            for (int v = 0; v < Period; v++)
            {
                Order[v] = v;
            }

            Identity = true;
            (entryWidth, entryMask) = (w, (1 << d.Planes[0].Bits) - 1);
            // Each entry's index: one plane whose fields each lie within a byte, a byte read a field; else (more planes, fields
            // across bytes or of more than 8 bits) each plane's field from a 32-bit word kept within the period, its bits
            // joined (the first plane the lowest).
            entryWide = Planes > 1 || Enumerable.Range(0, d.Values / w).Any(e => d.Planes[0].Place(e).Shift + d.Planes[0].Bits > 8);
            if (entryWide && PeriodBytes < 4)
            {
                return false;
            }

            entryAt = Pin<int>(spans * perSpan * Planes);
            entryShift = Pin<int>(spans * perSpan * Planes);
            entryMasks = Pin<uint>(Planes);
            entryLows = Pin<int>(Planes);
            for (int p = 0, low = 0; p < Planes; low += d.Planes[p].Bits, p++)
            {
                (entryMasks[p], entryLows[p]) = ((1u << d.Planes[p].Bits) - 1, low);
            }

            for (int r = 0; r < spans; r++)
            {
                for (int j = 0; j < perSpan; j++)
                {
                    int v = 32 * r + j * w;
                    for (int p = 0; p < Planes; p++)
                    {
                        var (b, s) = d.Planes[p].Place(v % d.Values / w);
                        int at = v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b, i = (r * perSpan + j) * Planes + p;
                        if (entryWide)
                        {
                            int kept = Math.Max(0, Math.Min(at, PeriodBytes - 4));
                            (at, s) = (kept, s + 8 * (at - kept));
                            if (s + d.Planes[p].Bits > 32)
                            {
                                return false;
                            }
                        }

                        (entryAt[i], entryShift[i]) = (at, s);
                    }
                }
            }

            vectorCount = vectors.Count;
            vectorFactors = vectors;
            vectorWidth = new int[vectorCount];
            vectorMasks = new byte*[vectorCount];
            var (start, offset, shift, mask, place) = (new List<int>(), new List<int>(), new List<int>(), new List<int>(), new List<int>());
            for (int f = 0; f < vectorCount; f++)
            {
                var factor = vectors[f];
                int vw = factor.Values;
                vectorWidth[f] = vw;
                vectorMasks[f] = Pin<byte>(factor.Type.Table!.Count);
                for (int i = 0; i < factor.Type.Table!.Count; i++)
                {
                    vectorMasks[f][i] = factor.Type.Table[i] < 0 ? (byte)0xFF : (byte)0;
                }

                for (int r = 0; r < spans; r++)
                {
                    for (int q = 0; q < 32 / vw; q++)
                    {
                        int v = 32 * r + q * vw, block = v / d.UnitValues * d.UnitBytes + d.ScaleBase(factor, v), index = d.ScaleIndex(factor, v);
                        start.Add(offset.Count);
                        if (factor.Pieces is { } pieces)
                        {
                            int at = 0;
                            foreach (var piece in pieces[index])
                            {
                                offset.Add(block + piece.Offset);
                                shift.Add(piece.Shift);
                                mask.Add((1 << piece.Bits) - 1);
                                place.Add(at);
                                at += piece.Bits;
                            }
                        }
                        else
                        {
                            // Its bits one after another from Offset: a piece a byte they touch.
                            long bit = 8L * factor.Offset + (long)index * factor.Type.Bits;
                            for (int left = factor.Type.Bits, at = 0; left > 0;)
                            {
                                int s = (int)(bit % 8), take = Math.Min(left, 8 - s);
                                offset.Add(block + (int)(bit / 8));
                                shift.Add(s);
                                mask.Add((1 << take) - 1);
                                place.Add(at);
                                (bit, left, at) = (bit + take, left - take, at + take);
                            }
                        }
                    }
                }
            }

            start.Add(offset.Count);
            (vecStart, vecOffset, vecShift, vecMask, vecPlace) = ([.. start], [.. offset], [.. shift], [.. mask], [.. place]);

            // The fast reading where every vector group's index is contiguous bits within 16 of its first byte.
            codebookFast = w is 4 or 8 && (vectorCount == 0 || vectorCount == 1 && vectorWidth[0] == 8);
            if (codebookFast && vectorCount == 1)
            {
                int groups = spans * 4;
                signAt = Pin<int>(groups);
                signShift = Pin<int>(groups);
                for (int q = 0; q < groups && codebookFast; q++)
                {
                    int bit = -1, bits = 0;
                    for (int piece = vecStart[q]; piece < vecStart[q + 1]; piece++)
                    {
                        int at = 8 * vecOffset[piece] + vecShift[piece];
                        bit = bit < 0 ? at : bit;
                        codebookFast &= at == bit + bits && vecPlace[piece] == bits;
                        bits += BitOperations.PopCount((uint)vecMask[piece]);
                    }

                    codebookFast &= bit >= 0 && bit % 8 + bits <= 16 && bit / 8 + 2 <= PeriodBytes;
                    if (codebookFast)
                    {
                        (signAt[q], signShift[q], signMask) = (bit / 8, bit % 8, (1 << bits) - 1);
                    }
                }
            }

            return true;
        }

        // A codebook span's 32 codes as the integer lane's unsigned bytes: each entry's values' bytes from the table (a word an
        // entry of 8 or 4 values), the sign vector's -1s applied ((v ^ m) - m a byte; each group's index one unaligned 16-bit
        // load), the least added back (the spans-made kernels' FastBookSpans); any other codebook by CodebookAny (AnyBookSpans).
        // The fast reading (codebookFast: entries of 8 or 4 values, at most one sign vector of 8).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> CodebookFast(byte* period, int span)
        {
            Vector256<byte> v;
            if (entryWidth == 8)
            {
                int at = span * 4;
                v = Vector256.Create(*(ulong*)(entryBytes + 8 * Entry(period, at)),
                    *(ulong*)(entryBytes + 8 * Entry(period, at + 1)),
                    *(ulong*)(entryBytes + 8 * Entry(period, at + 2)),
                    *(ulong*)(entryBytes + 8 * Entry(period, at + 3))).AsByte();
            }
            else
            {
                int at = span * 8;
                v = Vector256.Create(*(uint*)(entryBytes + 4 * Entry(period, at)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 1)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 2)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 3)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 4)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 5)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 6)),
                    *(uint*)(entryBytes + 4 * Entry(period, at + 7))).AsByte();
            }

            if (vectorCount == 1)
            {
                int g = span * 4, sm = signMask;
                ulong* masks = (ulong*)vectorMasks[0];
                var m = Vector256.Create(masks[Unsafe.ReadUnaligned<ushort>(period + signAt[g]) >> signShift[g] & sm],
                    masks[Unsafe.ReadUnaligned<ushort>(period + signAt[g + 1]) >> signShift[g + 1] & sm],
                    masks[Unsafe.ReadUnaligned<ushort>(period + signAt[g + 2]) >> signShift[g + 2] & sm],
                    masks[Unsafe.ReadUnaligned<ushort>(period + signAt[g + 3]) >> signShift[g + 3] & sm]).AsByte();
                v = (v ^ m) - m;
            }

            return v + leastBytes;
        }

        // Any codebook span's codes (entries of 1, 2, 4 or 8 values, vector factors of any group width and pieces), in registers:
        // the entries' bytes 64-bit words (8 / w entries a word for entries of 1 or 2 values, each word put in its lane), the
        // vector factors' masks applied, the least added back. Nothing is written.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> CodebookAny(byte* period, int span)
        {
            int w = entryWidth, per = 32 / w, at = span * per;
            Vector256<byte> v;
            switch (w)
            {
                case 8:
                    v = Vector256.Create(*(ulong*)(entryBytes + 8 * Entry(period, at)),
                        *(ulong*)(entryBytes + 8 * Entry(period, at + 1)),
                        *(ulong*)(entryBytes + 8 * Entry(period, at + 2)),
                        *(ulong*)(entryBytes + 8 * Entry(period, at + 3))).AsByte();
                    break;
                case 4:
                    v = Vector256.Create(*(uint*)(entryBytes + 4 * Entry(period, at)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 1)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 2)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 3)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 4)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 5)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 6)),
                        *(uint*)(entryBytes + 4 * Entry(period, at + 7))).AsByte();
                    break;
                default:
                    var words = Vector256<ulong>.Zero;
                    for (int q = 0, j = at; q < 4; q++)
                    {
                        ulong word = 0;
                        for (int b = 0; b < 64; b += 8 * w, j++)
                        {
                            word |= (ulong)(w == 1 ? *(byte*)(entryBytes + Entry(period, j)) : *(ushort*)(entryBytes + 2 * Entry(period, j))) << b;
                        }

                        words |= Vector256.Create(word) & Vector256.Equals(Vector256.Create((ulong)q), Vector256<ulong>.Indices);
                    }

                    v = words.AsByte();
                    break;
            }

            for (int f = 0; f < vectorCount; f++)
            {
                var m = VectorMask(period, span, f);
                v = (v ^ m) - m;
            }

            return v + leastBytes;
        }

        // Entry e's index (e counted over the period's spans): its byte's field, or its planes' fields from their words, joined.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private int Entry(byte* period, int e)
        {
            if (!entryWide)
            {
                return period[entryAt[e]] >> entryShift[e] & entryMask;
            }

            uint index = 0;
            for (int p = 0, at = e * Planes; p < Planes; p++, at++)
            {
                index |= (Unsafe.ReadUnaligned<uint>(period + entryAt[at]) >> entryShift[at] & entryMasks[p]) << entryLows[p];
            }

            return (int)index;
        }

        // Vector factor f's masks of a span's 32 values (0xFF where the value is negated): each of its groups' index read from
        // its pieces, its entry's mask bytes.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> VectorMask(byte* period, int span, int f)
        {
            int vw = vectorWidth[f], groups = 32 / vw, first = (f * (Period / 32) + span) * groups;
            byte* masks = vectorMasks[f];
            ulong w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            for (int q = 0; q < groups; q++)
            {
                int index = 0;
                for (int p = vecStart[first + q]; p < vecStart[first + q + 1]; p++)
                {
                    index |= (period[vecOffset[p]] >> vecShift[p] & vecMask[p]) << vecPlace[p];
                }

                // The group's mask bytes into its place of the 32 (a whole word for groups of 8; groups divide 8).
                {
                    int pos = q * vw;
                    ulong bits = 0;
                    if (vw == 8)
                    {
                        bits = *(ulong*)(masks + index * 8);
                    }
                    else
                    {
                        for (int b = 0; b < vw; b++)
                        {
                            bits |= (ulong)masks[index * vw + b] << (8 * (pos % 8 + b));
                        }
                    }

                    switch (pos / 8)
                    {
                        case 0:
                            w0 |= bits;
                            break;
                        case 1:
                            w1 |= bits;
                            break;
                        case 2:
                            w2 |= bits;
                            break;
                        default:
                            w3 |= bits;
                            break;
                    }
                }
            }

            return Vector256.Create(w0, w1, w2, w3).AsByte();
        }

        // Each value of the period's place in x's integer layout (group g's bytes from g · GroupValues: step s, lane l = j · P
        // + p, 4 bytes of chunk j's positions 4s..4s + 3 in its span's order) and its chunk; false where a chunk's values do
        // not share every scale (the order mixes them).
        private bool Places()
        {
            XPlace = Pin<int>(Period);
            ChunkOf = Pin<int>(Period);
            if (Order == null)
            {
                Order = Pin<int>(Period);
                for (int v = 0; v < Period; v++)
                {
                    Order[v] = v;
                }
            }

            for (int position = 0; position < Period; position++)
            {
                int v = Order[position], c = position / C, inChunk = position % C;
                int g = c / Cg, j = c % Cg;
                XPlace[v] = g * GroupValues + inChunk / 4 * 4 * Cg + 4 * j + inChunk % 4;
                ChunkOf[v] = c;
            }

            foreach (var f in factorList)
            {
                for (int c = 0; c < Period / C; c++)
                {
                    int v0 = Order[c * C];
                    for (int t = 1; t < C; t++)
                    {
                        int v = Order[c * C + t];
                        if (v / D.UnitValues != v0 / D.UnitValues || D.ScaleBase(f, v) != D.ScaleBase(f, v0) || D.ScaleIndex(f, v) != D.ScaleIndex(f, v0))
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        // Each scale and minimum: how it reads, and at each group of the period the 8 lanes' dwords (from the period's first
        // byte, kept within the period), the shift and mask of its bits (each piece's), and where a piece's bits go.
        private bool Factors(List<BlockScale> all, int scales)
        {
            factors = all.Count;
            scaleFactors = scales;
            pieces = Math.Max(1, all.Count == 0 ? 1 : all.Max(f => f.Pieces is { } p ? p.Max(x => x.Count) : 1));
            int groups = Period / GroupValues;
            reads = new FactorRead[factors];
            factorBits = new int[factors];
            factorPieces = new int[factors];
            factorBias = new float[factors];
            factorTables = new float*[factors];
            factorHalves = new (int, int, int, float, int)[factors];
            fOffset = Pin<int>(8 * groups * factors * pieces);
            fShift = Pin<Vector256<int>>(groups * factors * pieces);
            fMask = Pin<Vector256<int>>(groups * factors * pieces);
            fLeft = Pin<Vector256<int>>(groups * factors * pieces);
            fShiftOne = Pin<int>(groups * factors * pieces);
            fLeftOne = Pin<int>(groups * factors * pieces);
            Span<int> offsets = stackalloc int[8], shifts = stackalloc int[8], masks = stackalloc int[8], lefts = stackalloc int[8];
            for (int f = 0; f < factors; f++)
            {
                var factor = all[f];
                var type = factor.Type;
                factorBits[f] = factor.Pieces is { } given ? given[0].Sum(x => x.Bits) : type.Bits;
                factorPieces[f] = factor.Pieces is { } listed ? listed.Max(x => x.Count) : 1;
                factorBias[f] = (float)factor.Bias;
                if (type.Kind == NumberKind.Integer && factorBits[f] <= 24)
                {
                    reads[f] = type.Signed ? FactorRead.Signed : FactorRead.Unsigned;
                }
                else if (factorBits[f] <= 8)
                {
                    reads[f] = FactorRead.Table;
                    var table = Pin<float>(256);
                    for (int c = 0; c < 256; c++)
                    {
                        table[c] = (float)(type.Value((ulong)c) + factor.Bias);
                    }

                    factorTables[f] = table;
                }
                else if (HalfOf(type) is { } half)
                {
                    // (Its bits from their pieces where it has them: Read joins them before reading them as a 16-bit float.)
                    reads[f] = FactorRead.Half;
                    factorHalves[f] = half;
                }
                else if (type is { Kind: NumberKind.Float, Bits: 32, Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Specials: FloatSpecials.Ieee } && factor.Pieces is null)
                {
                    reads[f] = FactorRead.Single;
                }
                else
                {
                    reads[f] = FactorRead.Scalar;
                }

                for (int g = 0; g < groups; g++)
                {
                    for (int q = 0; q < pieces; q++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            int c = g * Cg + l / P, v0 = Order[c * C], index = D.ScaleIndex(factor, v0);
                            int start = v0 / D.UnitValues * D.UnitBytes + D.ScaleBase(factor, v0), bit, bits, left = 0;
                            if (factor.Pieces is { } p)
                            {
                                var own = p[index];
                                if (q >= own.Count)
                                {
                                    (offsets[l], shifts[l], masks[l], lefts[l]) = (start, 0, 0, 0);
                                    continue;
                                }

                                for (int e = 0; e < q; e++)
                                {
                                    left += own[e].Bits;
                                }

                                bit = 8 * own[q].Offset + own[q].Shift;
                                bits = own[q].Bits;
                            }
                            else if (q > 0)
                            {
                                (offsets[l], shifts[l], masks[l], lefts[l]) = (start, 0, 0, 0);
                                continue;
                            }
                            else
                            {
                                bit = 8 * factor.Offset + index * type.Bits;
                                bits = type.Bits;
                            }

                            int dword = start + bit / 8, s = bit % 8;
                            int kept = Math.Max(0, Math.Min(dword, PeriodBytes - 4));
                            s += 8 * (dword - kept);
                            if (s + bits > 32 && reads[f] != FactorRead.Scalar)
                            {
                                reads[f] = FactorRead.Scalar;
                            }

                            (offsets[l], shifts[l], masks[l], lefts[l]) = (kept, Math.Min(s, 31), bits >= 32 ? -1 : (1 << bits) - 1, left);
                        }

                        int i = (g * factors + f) * pieces + q;
                        offsets.CopyTo(new Span<int>(fOffset + 8 * i, 8));
                        fShift[i] = Vector256.Create<int>(shifts);
                        fMask[i] = Vector256.Create<int>(masks);
                        fLeft[i] = Vector256.Create<int>(lefts);
                        fShiftOne[i] = One(shifts);
                        fLeftOne[i] = One(lefts);
                    }
                }
            }

            fUniform = Pin<byte>(groups * factors);
            fMaskOne = Pin<int>(groups * factors * pieces);
            for (int g = 0; g < groups; g++)
            {
                for (int f = 0; f < factors; f++)
                {
                    bool same = reads[f] != FactorRead.Scalar;
                    for (int q = 0; q < pieces; q++)
                    {
                        int i = (g * factors + f) * pieces + q;
                        fMaskOne[i] = fMask[i].ToScalar();
                        same &= One(new ReadOnlySpan<int>(fOffset + 8 * i, 8)) >= 0 && fShiftOne[i] >= 0 && fLeftOne[i] >= 0 && fMask[i] == Vector256.Create(fMaskOne[i]);
                    }

                    fUniform[g * factors + f] = (byte)(same ? 1 : 0);
                }
            }

            if (factors == 1 && scaleFactors == 1 && factorPieces[0] == 1 && factorBias[0] == 0f)
            {
                oneScale = reads[0] switch
                {
                    FactorRead.Half => 1,
                    FactorRead.Table => 2,
                    _ => 0,
                };
            }

            // Each factor's numbers for the reads as type parameters, pinned (FactorKinds): its bias, bits, pieces, table and
            // 16-bit float reading as vectors.
            minsAdded = D.MinsAdded;
            fBiasAt = Pin<float>(factors);
            fBitsAt = Pin<int>(factors);
            fPiecesAt = Pin<int>(factors);
            fTablesAt = Pin<nint>(factors);
            fHalvesAt = Pin<HalfVectors>(factors);
            for (int f = 0; f < factors; f++)
            {
                (fBiasAt[f], fBitsAt[f], fPiecesAt[f], fTablesAt[f]) = (factorBias[f], factorBits[f], factorPieces[f], (nint)factorTables[f]);
                fHalvesAt[f] = reads[f] == FactorRead.Half ? new HalfVectors(factorHalves[f], 1f) : default;
            }

            // Each entry's 8 dwords where they lie within 16 bytes of the period (a block's own scales, a super-block's packed
            // ones): that window's first byte and the byte shuffle that puts each lane's dword in place, the same bits a gather
            // takes (−1: they do not).
            int entries = groups * factors * pieces;
            fWindow = Pin<int>(entries);
            fWindowBytes = Pin<Vector256<byte>>(entries);
            Span<byte> window = stackalloc byte[32];
            for (int i = 0; i < entries; i++)
            {
                int least = int.MaxValue, most = int.MinValue;
                for (int l = 0; l < 8; l++)
                {
                    (least, most) = (Math.Min(least, fOffset[8 * i + l]), Math.Max(most, fOffset[8 * i + l]));
                }

                int first = Math.Min(least, PeriodBytes - 16);
                fWindow[i] = first >= 0 && most + 4 <= first + 16 ? first : -1;
                for (int l = 0; l < 8 && fWindow[i] >= 0; l++)
                {
                    for (int b = 0; b < 4; b++)
                    {
                        window[4 * l + b] = (byte)(fOffset[8 * i + l] - first + b);
                    }
                }

                fWindowBytes[i] = fWindow[i] >= 0 ? Vector256.Create<byte>(window) : default;
            }

            // The kinds: up to two scales and two minimums, each read alike in every group of the period (FactorRead's
            // readings but the one lane at a time); one scale alone reads by its own way (OneScale), and any other set by
            // the loop over the factors (Factor). A field's bits: plain loads where every lane's are the same, else one
            // 16-byte load and a byte shuffle where each piece's dwords lie within 16 bytes, else gathers.
            int minCount = factors - scaleFactors;
            if (oneScale == 0 && scaleFactors is 1 or 2 && minCount <= 2 && reads.All(r => r != FactorRead.Scalar))
            {
                int kinds = 0;
                for (int f = 0; f < factors; f++)
                {
                    bool uniform = true, windowed = true;
                    for (int g = 0; g < groups; g++)
                    {
                        uniform &= fUniform[g * factors + f] != 0;
                        for (int q = 0; q < factorPieces[f]; q++)
                        {
                            windowed &= fWindow[(g * factors + f) * pieces + q] >= 0;
                        }
                    }

                    int fetch = uniform ? 1 : windowed ? (factorPieces[f] == 1 ? 4 : 5) : factorPieces[f] == 1 ? 2 : 3, slot = f < scaleFactors ? f : 2 + f - scaleFactors;
                    // (A table of up to 8 numbers, a field of up to 3 bits: its numbers in one register, a lane each by a
                    // permute: SmallTableNumber, 6; any other reading its FactorRead.)
                    int read = reads[f] == FactorRead.Table && factorBits[f] <= 3 ? 6 : (int)reads[f];
                    kinds |= (fetch << 4 | read) << (8 * slot);
                }

                FactorKinds = kinds;
            }

            ScalarReads(all);
            return PeriodBytes >= 4 || factors == 0;
        }

        // A factor's number as NumberType.Value reads it, for the factors read a lane at a time (ReadOneByOne, ScalarOf): its
        // type's fields and its bias; a table's entries, their bias added and rounded, 2^bits floats made once.
        private struct ScalarNumber
        {
            public NumberKind Kind;
            public bool Signed, Subnormals;
            public FloatSpecials Specials;
            public int Bits, Exponent, Mantissa, ExponentBias;
            public double Bias;
            public float* Table;
        }

        // The factors read a lane at a time: each one's number (ScalarNumber), and per group, lane and piece the byte, shift,
        // mask and place of its bits (BlockScale.Bits': its pieces as listed, or its bits from Offset a byte at a time).
        private void ScalarReads(List<BlockScale> all)
        {
            if (!reads.Contains(FactorRead.Scalar))
            {
                return;
            }

            var d = D;
            ReadsScalar = true;
            int groups = Period / GroupValues;
            var lanes = new List<(int Offset, int Shift, int Mask, int Left)>[groups * factors * 8];
            scalarNumbers = Pin<ScalarNumber>(factors);
            for (int f = 0; f < factors; f++)
            {
                if (reads[f] != FactorRead.Scalar)
                {
                    continue;
                }

                var factor = all[f];
                var type = factor.Type;
                float* table = null;
                if (type.Kind == NumberKind.Table)
                {
                    table = Pin<float>(1 << type.Bits);
                    for (int c = 0; c < 1 << type.Bits; c++)
                    {
                        table[c] = (float)(type.Value((ulong)c) + factor.Bias);
                    }
                }

                scalarNumbers[f] = new ScalarNumber
                {
                    Kind = type.Kind, Signed = type.Signed, Subnormals = type.Subnormals, Specials = type.Specials, Bits = type.Bits, Exponent = type.Exponent,
                    Mantissa = type.Mantissa, ExponentBias = type.ExponentBias, Bias = factor.Bias, Table = table,
                };
                for (int g = 0; g < groups; g++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        int c = g * Cg + l / P, v0 = Order[c * C], index = d.ScaleIndex(factor, v0);
                        int start = v0 / d.UnitValues * d.UnitBytes + d.ScaleBase(factor, v0);
                        var own = lanes[(g * factors + f) * 8 + l] = [];
                        if (factor.Pieces is { } listed)
                        {
                            int left = 0;
                            foreach (var piece in listed[index])
                            {
                                own.Add((start + piece.Offset, piece.Shift, (1 << piece.Bits) - 1, left));
                                left += piece.Bits;
                            }

                            continue;
                        }

                        long bit = 8L * factor.Offset + (long)index * type.Bits;
                        for (int remaining = type.Bits, at = 0; remaining > 0;)
                        {
                            int s = (int)(bit % 8), take = Math.Min(remaining, 8 - s);
                            own.Add((start + (int)(bit / 8), s, (1 << take) - 1, at));
                            (bit, remaining, at) = (bit + take, remaining - take, at + take);
                        }
                    }
                }
            }

            scalarPieces = Math.Max(1, lanes.Max(own => own?.Count ?? 0));
            int entries = groups * factors * 8 * scalarPieces;
            sOffset = Pin<int>(entries);
            sShift = Pin<int>(entries);
            sMask = Pin<int>(entries);
            sLeft = Pin<int>(entries);
            for (int i = 0; i < lanes.Length; i++)
            {
                for (int q = 0; q < (lanes[i]?.Count ?? 0); q++)
                {
                    int at = i * scalarPieces + q;
                    (sOffset[at], sShift[at], sMask[at], sLeft[at]) = lanes[i]![q];
                }
            }
        }

        /// <summary>
        /// The plan's factors as the group loop's type parameters (CpuBackend.WithFactors): 8 bits a slot (the first scale,
        /// the second, the first minimum, the second), the field's reading in the high 4 (0: no factor; 1: the same bits in
        /// every lane, by plain loads; 2: gathered, one piece; 3: gathered, its pieces joined; 4: one piece, by a window of 16
        /// bytes; 5: its pieces, each by its window) and its number's in the low 4
        /// (FactorRead: unsigned, signed, a table, a 16-bit float, a float32). 0 where the loop over the factors reads them
        /// (Factor): one scale alone (its own reading), none, more than two of either, or a field read one lane at a time.
        /// </summary>
        public int FactorKinds { get; private set; }

        /// <summary>Whether a factor is read a lane at a time (FactorRead.Scalar: ReadOneByOne), the loop over the factors then taking it (CpuBackend.FactorLoop{Yes}).</summary>
        public bool ReadsScalar { get; private set; }

        /// <summary>Whether a codebook's spans take the fast reading (CodebookFast; else CodebookAny).</summary>
        public bool CodebookIsFast => codebookFast;

        // FactorKinds' numbers: whether minimums are added (else subtracted); per factor its bias, bits, pieces, table and
        // 16-bit float reading.
        private bool minsAdded;
        private float* fBiasAt;
        private int* fBitsAt;
        private int* fPiecesAt;
        private nint* fTablesAt;
        private HalfVectors* fHalvesAt;

        // Per entry (group, factor, piece) the first byte of the 16 its 8 dwords lie within (−1: none), and the byte shuffle
        // that puts each lane's dword in place from them (WindowBits).
        private int* fWindow;
        private Vector256<byte>* fWindowBytes;

        /// <summary>A factor's bits for the 8 lanes, at entry <c>at</c> ((group · factors + f) · pieces) of factor <c>f</c>.</summary>
        internal interface IFactorBits
        {
            static abstract Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions;
        }

        /// <summary>The same bits in every lane: each piece by a plain load, joined, broadcast (Read's uniform branch).</summary>
        internal readonly struct UniformBits : IFactorBits
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions
            {
                uint value = 0;
                for (int q = 0, i = at, count = plan.fPiecesAt[f]; q < count; q++, i++)
                {
                    value |= (Unsafe.ReadUnaligned<uint>(period + plan.fOffset[8 * i]) >> plan.fShiftOne[i] & (uint)plan.fMaskOne[i]) << plan.fLeftOne[i];
                }

                return Vector256.Create((int)value);
            }
        }

        /// <summary>One piece, gathered: a dword a lane, shifted and masked.</summary>
        internal readonly struct GatherBits : IFactorBits
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions =>
                plan.Shifted<TI>(TI.Gather(period, plan.fOffset + 8 * at), at) & plan.fMask[at];
        }

        /// <summary>One piece whose dwords lie within 16 bytes (fWindow): those bytes loaded once and shuffled to the lanes, shifted and masked.</summary>
        internal readonly struct WindowBits : IFactorBits
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions =>
                plan.Shifted<TI>(TI.ShuffleBytes(TI.Halves(period + plan.fWindow[at]), plan.fWindowBytes[at]).AsInt32(), at) & plan.fMask[at];
        }

        /// <summary>Pieces whose dwords each lie within 16 bytes: each by its window, shifted, masked and put in place.</summary>
        internal readonly struct WindowPieces : IFactorBits
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions
            {
                var bits = WindowBits.Bits<TI>(plan, period, at, f);
                for (int q = 1, count = plan.fPiecesAt[f]; q < count; q++)
                {
                    var piece = WindowBits.Bits<TI>(plan, period, at + q, f).AsUInt32();
                    int one = plan.fLeftOne[at + q];
                    bits |= (TI.ShiftsByLane || one < 0 ? TI.ShiftLeft(piece, plan.fLeft[at + q].AsUInt32()) : Vector256.ShiftLeft(piece, one)).AsInt32();
                }

                return bits;
            }
        }

        /// <summary>Its pieces gathered, each shifted, masked and put in place (Read's gathered branch).</summary>
        internal readonly struct PieceBits : IFactorBits
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<int> Bits<TI>(DescribedPlan plan, byte* period, int at, int f) where TI : struct, IInstructions
            {
                var bits = plan.Shifted<TI>(TI.Gather(period, plan.fOffset + 8 * at), at) & plan.fMask[at];
                for (int q = 1, count = plan.fPiecesAt[f]; q < count; q++)
                {
                    var piece = (plan.Shifted<TI>(TI.Gather(period, plan.fOffset + 8 * (at + q)), at + q) & plan.fMask[at + q]).AsUInt32();
                    int one = plan.fLeftOne[at + q];
                    bits |= (TI.ShiftsByLane || one < 0 ? TI.ShiftLeft(piece, plan.fLeft[at + q].AsUInt32()) : Vector256.ShiftLeft(piece, one)).AsInt32();
                }

                return bits;
            }
        }

        /// <summary>A factor's bits as its numbers (Read's readings).</summary>
        internal interface IFactorNumber
        {
            static abstract Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions;
        }

        internal readonly struct UnsignedNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions =>
                Vector256.ConvertToSingle(bits) + Vector256.Create(plan.fBiasAt[f]);
        }

        internal readonly struct SignedNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions
            {
                int up = 32 - plan.fBitsAt[f];
                return Vector256.ConvertToSingle(Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(bits, up), up)) + Vector256.Create(plan.fBiasAt[f]);
            }
        }

        internal readonly struct TableNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions =>
                TI.Gather((float*)plan.fTablesAt[f], bits);
        }

        // A table of up to 8 numbers (a field of up to 3 bits): the table's first 8 in one register, each lane's by a permute,
        // where a gather would read them from memory (the same numbers).
        internal readonly struct SmallTableNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions =>
                TI.Permute(Vector256.Load((float*)plan.fTablesAt[f]), bits);
        }

        internal readonly struct HalfNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions =>
                plan.fHalvesAt[f].Of(bits) + Vector256.Create(plan.fBiasAt[f]);
        }

        internal readonly struct SingleNumber : IFactorNumber
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Of<TI>(DescribedPlan plan, Vector256<int> bits, int f) where TI : struct, IInstructions =>
                bits.AsSingle() + Vector256.Create(plan.fBiasAt[f]);
        }

        /// <summary>A slot of FactorKinds: factor <c>f</c>'s numbers for group <c>gp</c>'s 8 lanes, or none.</summary>
        internal interface IFactorSlot
        {
            static abstract bool Present { get; }

            static abstract Vector256<float> Read<TI>(DescribedPlan plan, byte* period, int gp, int f) where TI : struct, IInstructions;
        }

        internal readonly struct NoFactor : IFactorSlot
        {
            public static bool Present => false;

            public static Vector256<float> Read<TI>(DescribedPlan plan, byte* period, int gp, int f) where TI : struct, IInstructions => default;
        }

        internal readonly struct Field<TB, TN> : IFactorSlot where TB : struct, IFactorBits where TN : struct, IFactorNumber
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<float> Read<TI>(DescribedPlan plan, byte* period, int gp, int f) where TI : struct, IInstructions =>
                TN.Of<TI>(plan, TB.Bits<TI>(plan, period, (gp * plan.factors + f) * plan.pieces, f), f);
        }

        /// <summary>Factor's arithmetic over FactorKinds' slots, in its order (the scales' product, the unit, the minimums' product, its sign, the lanes kept).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal void Slots<T0, T1, T2, T3, TI>(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min)
            where T0 : struct, IFactorSlot where T1 : struct, IFactorSlot where T2 : struct, IFactorSlot where T3 : struct, IFactorSlot where TI : struct, IInstructions
        {
            var s = T0.Read<TI>(this, period, gp, 0);
            if (T1.Present)
            {
                s *= T1.Read<TI>(this, period, gp, 1);
            }

            if (Unit != 1f)
            {
                s *= Vector256.Create(Unit);
            }

            var mn = Vector256<float>.Zero;
            if (T2.Present)
            {
                int f = T1.Present ? 2 : 1;
                var product = T2.Read<TI>(this, period, gp, f);
                if (T3.Present)
                {
                    product *= T3.Read<TI>(this, period, gp, f + 1);
                }

                mn = minsAdded ? product : -product;
            }

            if (valid < 8)
            {
                var keep = GroupMask(valid).AsSingle();
                s &= keep;
                mn &= keep;
            }

            scale = s;
            min = mn;
        }

        // The one-row integer kernel per instruction set (Instructions' value), and the tile kernel per instruction set and
        // width (8 more: the 128-bit twin's), resolved on first use (CpuBackend.ResolveOne, ResolveTile): 0 not yet, −1 none
        // (the one-row kernel's own dispatch reads the plan).
        private nint* oneKernels, tileKernels;

        /// <summary>The one-row integer kernel by TI (<paramref name="with"/>: its Instructions value), or null where DescribedIntegerOne's dispatch reads the plan.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> OneKernel<TI>(int with) where TI : struct, IInstructions
        {
            nint f = oneKernels[with];
            if (f == 0)
            {
                f = ResolveOne<TI>(this);
                oneKernels[with] = f = f == 0 ? -1 : f;
            }

            return f == -1 ? null : (delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float>)f;
        }

        /// <summary>
        /// The one-row integer kernel by TI whose codebook indices are made by bit deposit (DepositSteps), or the plan's own
        /// (OneKernel) where its steps do not take it.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> DepositKernel<TI>(int with) where TI : struct, IInstructions
        {
            if (!depositSteps)
            {
                return OneKernel<TI>(with);
            }

            nint f = oneKernels[8 + with];
            if (f == 0)
            {
                oneKernels[8 + with] = f = ResolveDeposit<TI>(this);
            }

            return (delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float>)f;
        }

        /// <summary>
        /// The integer tile kernel (a block of rows of x by a group of stored rows: IntegerTile, WideTile) by TI (<paramref name="with"/>:
        /// its Instructions value) on 256-bit vectors, or on 128-bit ones where <paramref name="narrow"/> (the 128-bit twin's):
        /// every plan has one.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal delegate*<DescribedPlan, byte*, nint, int, byte*, int, int, int, sbyte*, float*, nint, nint, int, byte*, float*, void> TileKernel<TI>(int with, bool narrow)
            where TI : struct, IInstructions
        {
            int at = narrow ? with + 8 : with;
            nint f = tileKernels[at];
            if (f == 0)
            {
                tileKernels[at] = f = narrow ? ResolveTile<TI, Vectors128>(this) : ResolveTile<TI, Vectors256>(this);
            }

            return (delegate*<DescribedPlan, byte*, nint, int, byte*, int, int, int, sbyte*, float*, nint, nint, int, byte*, float*, void>)f;
        }

        /// <summary>
        /// The codes of span <paramref name="span"/> of the period at <paramref name="period"/> (its first byte), 32 bytes in the
        /// span's order: each plane's run loaded once, shifted (a run each) and masked, the planes joined (the first the lowest
        /// bits); a span not in runs one value at a time.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> Raw<TI>(byte* period, int span) where TI : struct, IInstructions
        {
            Vector256<byte> code = default;
            int at = span * Planes;
            for (int p = 0; p < Planes; p++, at++)
            {
                byte* from = period + runOffset[at];
                Vector256<byte> v;
                switch (runKind[at])
                {
                    case 1:
                        v = OneRun.Load<TI>(from, runShift, runCount, at);
                        break;
                    case 2:
                        v = TwoRuns.Load<TI>(from, runShift, runCount, at);
                        break;
                    case 4:
                        v = FourRuns.Load<TI>(from, runShift, runCount, at);
                        break;
                    case 8:
                        v = EightRuns.Load<TI>(from, runShift, runCount, at);
                        break;
                    case StreamRuns:
                        v = StreamLoad(period, at, planeBits[p]);
                        break;
                    case WindowRuns:
                        code |= Vector256.ShiftLeft((WindowLoad<TI>(period, at) & planeMask[p]).AsUInt16(), planeLow[p]).AsByte();
                        continue;
                    default:
                        return OneByOne(period, span);
                }

                if (planeRadix[p] != 0)
                {
                    v = DigitsOf(v, digitTimes[at], planeRadix[p]);
                }

                code |= Vector256.ShiftLeft((v & planeMask[p]).AsUInt16(), planeLow[p]).AsByte();
            }

            return code;
        }

        // A plane of digits' planes' radix (0: bits), and per span and plane each byte's power of the radix (its run's digit);
        // each plane's bits (a stream's field's: StreamLoad), from the plan, not the description, on the kernels' path.
        private int* planeRadix;
        private int* planeBits;
        private Vector256<byte>* digitTimes;

        // Each byte's digit: ⌊radix · ((byte · times) mod 256) / 256⌋ (CodePlane.Digit), over 16-bit lanes.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Vector256<byte> DigitsOf(Vector256<byte> bytes, Vector256<byte> times, int radix)
        {
            var (low, high) = Vector256.Widen(bytes);
            var (lowTimes, highTimes) = Vector256.Widen(times);
            var mask = Vector256.Create((ushort)0xFF);
            var r = Vector256.Create((ushort)radix);
            return Vector256.Narrow(Vector256.ShiftRightLogical((low * lowTimes & mask) * r, 8), Vector256.ShiftRightLogical((high * highTimes & mask) * r, 8));
        }

        // A span's run kind for a bit stream (Spans; Raw reads it by StreamLoad).
        private const byte StreamRuns = 16;

        // Per span and stream plane, its four groups' first bytes (from the period's first).
        private int* streamAt;

        // A stream plane's span of 32 fields (`bits` bits each, across byte boundaries) as bytes in value order: each group of 8
        // fields one word of its `bits` bytes, a 64-bit lane each, and field j of every lane shifted out and masked at once.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> StreamLoad(byte* period, int at, int bits)
        {
            int* group = streamAt + 4 * at;
            if (bits == 1)
            {
                // Fields of one bit: the 4 groups' bytes in every 32-bit lane, byte j / 8 of them to byte j, and bit j mod 8
                // of it tested.
                uint four = period[group[0]] | (uint)period[group[1]] << 8 | (uint)period[group[2]] << 16 | (uint)period[group[3]] << 24;
                // (Byte j / 8 of the 4 repeated to byte j, each 128-bit half picking from its own copy of them: a constant to the
                // JIT, where a static field read by a kernel compiled before its class is made costs a call and the spills
                // around it.)
                var bitBytes = Vector256.Create((byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19);
                var spread = Vector256.Shuffle(Vector256.Create(four).AsByte(), bitBytes);
                var test = Vector256.Create(0x8040201008040201UL).AsByte();
                return Vector256.Equals(spread & test, test) & Vector256.Create((byte)1);
            }

            var words = Vector256.Create(StreamWord(period + group[0], bits), StreamWord(period + group[1], bits), StreamWord(period + group[2], bits),
                StreamWord(period + group[3], bits));
            var mask = Vector256.Create((ulong)((1 << bits) - 1));
            var codes = words & mask;
            for (int j = 1; j < 8; j++)
            {
                codes |= Vector256.ShiftLeft(Vector256.ShiftRightLogical(words, j * bits) & mask, 8 * j);
            }

            return codes.AsByte();
        }

        // The `bits` bytes of 8 stream fields from `p` (little-endian), and no byte past them.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong StreamWord(byte* p, int bits) => bits switch
        {
            1 => p[0],
            2 => Unsafe.ReadUnaligned<ushort>(p),
            3 => Unsafe.ReadUnaligned<ushort>(p) | (ulong)p[2] << 16,
            4 => Unsafe.ReadUnaligned<uint>(p),
            5 => Unsafe.ReadUnaligned<uint>(p) | (ulong)p[4] << 32,
            6 => Unsafe.ReadUnaligned<uint>(p) | (ulong)Unsafe.ReadUnaligned<ushort>(p + 4) << 32,
            7 => Unsafe.ReadUnaligned<uint>(p) | (ulong)Unsafe.ReadUnaligned<ushort>(p + 4) << 32 | (ulong)p[6] << 48,
            _ => Unsafe.ReadUnaligned<ulong>(p),
        };

        // A span's codes one value at a time, in value order (a layout the runs and windows do not take), in registers: each
        // value's planes' fields from their dwords at the plan's places (OneValues), joined as CodeBits joins them (a digit
        // plane's digit of its byte), 8 values a 64-bit word and each word put in its lane of the span. Nothing is written.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> OneByOne(byte* period, int span)
        {
            var codes = Vector256<ulong>.Zero;
            for (int q = 0, i = 32 * span * Planes; q < 4; q++)
            {
                ulong word = 0;
                for (int j = 0; j < 64; j += 8)
                {
                    uint code = 0;
                    for (int p = 0; p < Planes; p++, i++)
                    {
                        uint field = Unsafe.ReadUnaligned<uint>(period + oneAt[i]) >> oneShift[i];
                        int radix = planeRadix[p];
                        field = radix == 0 ? field & oneMask[p] : ((field & 0xFF) * (uint)onePower[i] & 0xFF) * (uint)radix >> 8;
                        code |= field << planeLow[p];
                    }

                    word |= (ulong)code << j;
                }

                codes |= Vector256.Create(word) & Vector256.Equals(Vector256.Create((ulong)q), Vector256<ulong>.Indices);
            }

            return codes.AsByte();
        }

        // Each byte through the 256-entry table (codes of 5 to 8 bits whose numbers are not their bits), in registers: 8 codes
        // at a time each a 32-bit lane, the table's dword at each code's byte gathered (its first byte the code's), and the 32
        // lanes' first bytes narrowed back into place. The table is the plan's own (made once from the description), not a copy
        // of stored values; nothing is written.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> Through<TI>(Vector256<byte> code) where TI : struct, IInstructions
        {
            var (low, high) = Vector256.Widen(code);
            var (c0, c1) = Vector256.Widen(low);
            var (c2, c3) = Vector256.Widen(high);
            byte* table = byteTable;
            return Vector256.Narrow(Vector256.Narrow(TI.Gather(table, c0.AsInt32()).AsUInt32(), TI.Gather(table, c1.AsInt32()).AsUInt32()),
                Vector256.Narrow(TI.Gather(table, c2.AsInt32()).AsUInt32(), TI.Gather(table, c3.AsInt32()).AsUInt32()));
        }

        /// <summary>
        /// Group <paramref name="gp"/> of the period at <paramref name="period"/> as the integer lane's steps (codes of at most 8
        /// bits: a wide integer's are made by IStepLoad.Plane): lane l of step s holds 4 bytes of chunk l, its positions
        /// 4s..4s + 3 in its span's order. T = 4: s4..s7 are 0. The group's spans made one at a time (TS: the plan's way of
        /// making them, ISpanCodes), each one's dwords put in their steps' lanes as it is made (a dword broadcast and a blend
        /// a step), so a kernel holds one span's making, its plan's way alone, and the JIT inlines all of it: every span made
        /// and moved in registers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Steps<TI, TS>(byte* period, int gp, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions where TS : struct, ISpanCodes
        {
            int span = gp * (GroupValues / 32);
            Vector256<int> t0 = default, t1 = default, t2 = default, t3 = default, t4 = default, t5 = default, t6 = default, t7 = default;
            if (C == 32)
            {
                // Chunk l is span l: its dword s to lane l of step s.
                for (int l = 0; l < 8; l++)
                {
                    var v = TS.Codes<TI>(this, period, span + l).AsInt32();
                    var lane = Vector256.Equals(Vector256.Create(l), Vector256<int>.Indices);
                    t0 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(0)), t0);
                    t1 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(1)), t1);
                    t2 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(2)), t2);
                    t3 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(3)), t3);
                    t4 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(4)), t4);
                    t5 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(5)), t5);
                    t6 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(6)), t6);
                    t7 = Vector256.ConditionalSelect(lane, TI.Permute(v, Vector256.Create(7)), t7);
                }
            }
            else
            {
                // Chunks 2q and 2q + 1 are span q's halves: its dwords s and 4 + s to lanes 2q and 2q + 1 of step s.
                for (int q = 0; q < 4; q++)
                {
                    var v = TS.Codes<TI>(this, period, span + q).AsInt32();
                    var lanes = Vector256.Equals(Vector256.Create(q), Vector256.ShiftRightLogical(Vector256<int>.Indices, 1));
                    t0 = Vector256.ConditionalSelect(lanes, TI.Permute(v, Vector256.Create(0, 4, 0, 4, 0, 4, 0, 4)), t0);
                    t1 = Vector256.ConditionalSelect(lanes, TI.Permute(v, Vector256.Create(1, 5, 1, 5, 1, 5, 1, 5)), t1);
                    t2 = Vector256.ConditionalSelect(lanes, TI.Permute(v, Vector256.Create(2, 6, 2, 6, 2, 6, 2, 6)), t2);
                    t3 = Vector256.ConditionalSelect(lanes, TI.Permute(v, Vector256.Create(3, 7, 3, 7, 3, 7, 3, 7)), t3);
                }
            }

            (s0, s1, s2, s3, s4, s5, s6, s7) = (t0.AsByte(), t1.AsByte(), t2.AsByte(), t3.AsByte(), t4.AsByte(), t5.AsByte(), t6.AsByte(), t7.AsByte());
        }

        // Steps gathered straight into the dot products' lanes (GatherSteps), where the spans do not load alike: step s of a
        // group (lane l: 4 bytes of chunk l, its positions 4s..4s + 3) by one gather a plane of the 8 chunks' fields and a
        // fixed reading of each, no span made and no transpose. Per plane its reading (StepRead), per step of the period and
        // plane the 8 lanes' dwords' first bytes (from the period's first) and their shifts, a digit plane's multipliers;
        // for a codebook per step each lane's dword within its entry, and per vector factor its index's dwords, shifts and
        // dword within the vector.
        private bool stepsGathered;
        private StepRead[] stepRead = [];
        private int* stepOffset;
        private Vector256<uint>* stepShift;
        private int* stepShiftOne;
        private Vector256<ushort>* stepTimes, stepTimesHigh;
        private Vector256<uint>[] stepMask = [], stepField = [];
        private Vector256<int>* bookWithin;
        private int* vectorOffset;
        private Vector256<uint>* vectorShift;
        private Vector256<int>* vectorWithin;
        private uint[] vectorIndexMask = [];
        private List<BlockScale> vectorFactors = [];

        // How a plane's 4 fields of a step's lane come out of its dword: 4 bytes at one shift; a stream of fields from a bit,
        // spread a byte each; 4 bytes' digits of one place; a codebook's index field.
        private enum StepRead : byte
        {
            Bytes,
            Stream,
            Digits,
            Index,
        }

        // Whether the integer lane's steps can be gathered (codes of one byte plane: every plane's fields of each step's lane
        // within one dword read alike, or a codebook of entries of 4 or 8 values whose index fields and vector factors' fields
        // each lie within a dword): then the plan's per-step places, false otherwise (the spans made and transposed).
        private bool GatherSteps()
        {
            var d = D;
            int groups = Period / GroupValues, steps = groups * T;
            bool book = Map == CodeMap.Codebook;
            if (Mode != DescribedMode.Integer || P != 1 || PeriodBytes < 4 || book && (entryWidth is not (4 or 8) || vectorWidth.Any(w => w is not (4 or 8))))
            {
                return false;
            }

            // A natural plan's steps of entries of 8 (lanes 2e and 2e + 1 one entry's two dwords: its 4 entries) make their
            // indices two steps at once (PairSteps), where they have no vector factors and up to three planes.
            bool paired = natural && book && entryWidth == 8 && Enumerable.Range(0, steps * 8).All(i => StepValue(i / 8, i % 8) % entryWidth / 4 == i % 2
                && StepValue(i / 8, i % 8) / entryWidth == StepValue(i / 8, i % 8 & ~1) / entryWidth);
            pairSteps = paired && vectorCount == 0 && Planes <= 3 && T % 2 == 0;
            int indexSteps = pairSteps ? steps / 2 : steps;
            stepRead = new StepRead[Planes];
            stepMask = new Vector256<uint>[Planes];
            stepField = new Vector256<uint>[Planes];
            stepOffset = Pin<int>(8 * steps * Planes);
            stepShift = Pin<Vector256<uint>>(steps * Planes);
            stepShiftOne = Pin<int>(steps * Planes);
            stepTimes = Pin<Vector256<ushort>>(steps * Planes);
            stepTimesHigh = Pin<Vector256<ushort>>(steps * Planes);
            Span<int> offsets = stackalloc int[8];
            Span<uint> shifts = stackalloc uint[8];
            Span<ushort> times = stackalloc ushort[16];
            Span<int> current = stackalloc int[8];
            Span<long> fieldBits = stackalloc long[8];
            for (int p = 0; p < Planes; p++)
            {
                var plane = d.Planes[p];
                int bits = plane.Bits;
                if (book)
                {
                    stepRead[p] = StepRead.Index;
                }
                else
                {
                    // The plane's reading: the first that every step's every lane takes.
                    var reads = plane.Radix != 0 ? [StepRead.Digits] : new[] { StepRead.Bytes, StepRead.Stream };
                    int found = Array.FindIndex(reads, read => Enumerable.Range(0, steps * 8).All(i => StepPlace(plane, i / 8, i % 8, read, out _, out _)));
                    if (found < 0)
                    {
                        return false;
                    }

                    stepRead[p] = reads[found];
                }

                stepMask[p] = Vector256.Create(stepRead[p] == StepRead.Index ? (1u << bits) - 1 : 0x01010101u * (uint)((1 << bits) - 1));
                stepField[p] = Vector256.Create((uint)((1 << bits) - 1));
                int width = stepRead[p] == StepRead.Index ? bits : 4 * bits;
                for (int s = 0; s < indexSteps; s++)
                {
                    int i = s * Planes + p;
                    for (int l = 0; l < 8; l++)
                    {
                        if (!StepPlace(plane, s, l, stepRead[p], out offsets[l], out int shift))
                        {
                            return false;
                        }

                        // An index's or a stream's fields from the dword an earlier step of the group reads where they fit it (a
                        // natural plan's from the dword its step's lanes share: Share).
                        if (stepRead[p] is StepRead.Index or StepRead.Stream && !natural && !Keep(current, l, s % T == 0, 8L * offsets[l] + shift, width, out offsets[l], out shift))
                        {
                            return false;
                        }

                        // (A digit plane's "shift" is its digit: its bytes unshifted, times the radix to the digit.)
                        shifts[l] = stepRead[p] == StepRead.Digits ? 0u : (uint)shift;
                        int power = 1;
                        for (int e = 0; stepRead[p] == StepRead.Digits && e < shift; e++)
                        {
                            power *= plane.Radix;
                        }

                        times[2 * l] = times[2 * l + 1] = (ushort)power;
                    }

                    if (natural && stepRead[p] is StepRead.Index or StepRead.Stream && !Share(offsets, shifts, fieldBits, width))
                    {
                        return false;
                    }

                    offsets.CopyTo(new Span<int>(stepOffset + 8 * i, 8));
                    stepShift[i] = Vector256.Create<uint>(shifts);
                    stepShiftOne[i] = shifts.IndexOfAnyExcept(shifts[0]) < 0 ? (int)shifts[0] : -1;
                    stepTimes[i] = Vector256.Create<ushort>(times);
                    stepTimesHigh[i] = stepTimes[i] << 8;
                }
            }

            if (book)
            {
                // Each lane's dword within its entry; each vector factor's index (one field within a dword) and dword within its vector.
                bookWithin = Pin<Vector256<int>>(steps);
                vectorOffset = Pin<int>(8 * steps * Math.Max(1, vectorCount));
                vectorShift = Pin<Vector256<uint>>(steps * Math.Max(1, vectorCount));
                vectorWithin = Pin<Vector256<int>>(steps * Math.Max(1, vectorCount));
                vectorIndexMask = new uint[vectorCount];
                Span<int> within = stackalloc int[8];
                Span<int> factorCurrent = stackalloc int[8 * Math.Max(1, vectorCount)];
                // (A natural plan's steps of entries of 8: lanes 2e and 2e + 1 the two dwords of one entry, a qword each pair.)
                pairedEntries = paired;
                for (int s = 0; s < steps; s++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        within[l] = StepValue(s, l) % entryWidth / 4;
                    }

                    bookWithin[s] = Vector256.Create<int>(within);
                    for (int f = 0; f < vectorCount; f++)
                    {
                        var factor = vectorFactors[f];
                        int i = s * vectorCount + f, bitsOf = 0;
                        for (int l = 0; l < 8; l++)
                        {
                            int v = StepValue(s, l), index = d.ScaleIndex(factor, v), origin = v / d.UnitValues * d.UnitBytes + d.ScaleBase(factor, v);
                            long bit;
                            if (factor.Pieces is { } pieces)
                            {
                                // Its pieces one run of bits (each after the last), or no gather.
                                var own = pieces[index];
                                bit = 8L * (origin + own[0].Offset) + own[0].Shift;
                                bitsOf = 0;
                                foreach (var piece in own)
                                {
                                    if (8L * (origin + piece.Offset) + piece.Shift != bit + bitsOf)
                                    {
                                        return false;
                                    }

                                    bitsOf += piece.Bits;
                                }
                            }
                            else
                            {
                                (bit, bitsOf) = (8L * (origin + factor.Offset) + (long)index * factor.Type.Bits, factor.Type.Bits);
                            }

                            within[l] = v % vectorWidth[f] / 4;
                            if (natural)
                            {
                                (offsets[l], shifts[l]) = ((int)(bit / 8), (uint)(bit % 8));
                                continue;
                            }

                            if (!Keep(factorCurrent.Slice(8 * f, 8), l, s % T == 0, bit, bitsOf, out int kept, out int shift))
                            {
                                return false;
                            }

                            (offsets[l], shifts[l]) = (kept, (uint)shift);
                        }

                        if (natural && !Share(offsets, shifts, fieldBits, bitsOf))
                        {
                            return false;
                        }

                        vectorIndexMask[f] = bitsOf >= 32 ? uint.MaxValue : (1u << bitsOf) - 1;
                        offsets.CopyTo(new Span<int>(vectorOffset + 8 * i, 8));
                        vectorShift[i] = Vector256.Create<uint>(shifts);
                        vectorWithin[i] = Vector256.Create<int>(within);
                    }
                }
            }

            // The kernels' constants pinned: per plane its reading, mask, field mask, bits (then its radix), place in the code;
            // per vector factor its index mask, its masks' table and its dwords an entry.
            stepReads = Pin<byte>(Planes);
            stepMasks = Pin<Vector256<uint>>(Planes);
            stepFields = Pin<Vector256<uint>>(Planes);
            stepBits = Pin<int>(2 * Planes);
            stepLows = Pin<int>(Planes);
            for (int p = 0; p < Planes; p++)
            {
                (stepReads[p], stepMasks[p], stepFields[p]) = ((byte)stepRead[p], stepMask[p], stepField[p]);
                (stepBits[p], stepBits[Planes + p]) = (d.Planes[p].Bits, d.Planes[p].Radix);
                stepLows[p] = book ? entryLows[p] : planeLow[p];
            }

            vectorIndexMasks = Pin<Vector256<uint>>(vectorCount);
            vectorMaskTables = Pin<nint>(vectorCount);
            vectorDwords = Pin<int>(vectorCount);
            vectorFromBits = Pin<byte>(vectorCount);
            for (int f = 0; f < vectorCount; f++)
            {
                (vectorIndexMasks[f], vectorMaskTables[f], vectorDwords[f]) = (Vector256.Create(vectorIndexMask[f]), (nint)vectorMasks[f], vectorWidth[f] / 4);
                // Whether the vectors are the index's bits (entry s's value j −1 exactly where bit j of s is set).
                var table = vectorFactors[f].Type;
                int width = vectorWidth[f];
                vectorFromBits[f] = (byte)(table.Bits == width && Enumerable.Range(0, table.Table!.Count).All(e => table.Table[e] == ((e / width >> e % width & 1) != 0 ? -1d : 1d)) ? 1 : 0);
            }

            // Each group's distinct dword vectors (its planes' and vector factors' over its steps), gathered once a group:
            // each step's plane and factor reads one of them (its index from the group's first).
            var words = new List<int[]>();
            var known = new Dictionary<(int, int, int, int, int, int, int, int), int>();
            groupWordStart = Pin<int>(groups + 1);
            stepWord = Pin<int>(steps * Planes);
            vectorWord = Pin<int>(steps * Math.Max(1, vectorCount));
            int groupSteps = pairSteps ? T / 2 : T;
            for (int gp = 0; gp < groups; gp++)
            {
                known.Clear();
                int start = words.Count;
                for (int s = gp * groupSteps; s < (gp + 1) * groupSteps; s++)
                {
                    for (int p = 0; p < Planes; p++)
                    {
                        stepWord[s * Planes + p] = Word(stepOffset + 8 * (s * Planes + p)) - start;
                    }

                    for (int f = 0; f < vectorCount; f++)
                    {
                        vectorWord[s * vectorCount + f] = Word(vectorOffset + 8 * (s * vectorCount + f)) - start;
                    }
                }

                groupWordStart[gp + 1] = words.Count;
                MostGroupWords = Math.Max(MostGroupWords, words.Count - start);
            }

            wordOffsets = Pin<int>(8 * Math.Max(1, words.Count));
            for (int u = 0; u < words.Count; u++)
            {
                words[u].CopyTo(new Span<int>(wordOffsets + 8 * u, 8));
            }

            // A natural plan's steps' words each from two 16-byte windows of the period (Window), or none: the plan is not one.
            if (natural && !(Windows(stepOffset, indexSteps * Planes, out stepWindows, out stepControls) && Windows(vectorOffset, steps * vectorCount, out vectorWindows, out vectorControls)))
            {
                return false;
            }

            // The readings as type parameters for up to three planes (StepKinds).
            if (Planes is >= 1 and <= 3)
            {
                int kinds = ((int)Map + 1) << 12;
                for (int p = 0; p < Planes; p++)
                {
                    int kind = stepRead[p] switch
                    {
                        StepRead.Bytes or StepRead.Index => 1,
                        StepRead.Stream => d.Planes[p].Bits == 1 ? 3 : 4,
                        _ => 5,
                    };
                    kinds |= kind << (4 * p);
                }

                StepKinds = kinds;
            }

            StepShiftsUniform = Enumerable.Range(0, indexSteps * Planes).All(i => stepShiftOne[i] >= 0);
            stepsGathered = true;
            return true;

            // The dword lane l reads a field of `width` bits from `bit` from: its current one where the field fits it (not at
            // a group's first step), else the field's own (kept within the period); false where the field does not fit it.
            bool Keep(Span<int> lanes, int l, bool groupStart, long bit, int width, out int kept, out int shift)
            {
                if (groupStart || bit < 8L * lanes[l] || bit - 8L * lanes[l] + width > 32)
                {
                    lanes[l] = (int)Math.Clamp(bit / 8, 0, PeriodBytes - 4);
                }

                (kept, shift) = (lanes[l], (int)(bit - 8L * lanes[l]));
                return shift >= 0 && shift + width <= 32;
            }

            // A natural plan's step (its lanes' fields one chunk's, at `at`'s bytes and `shift`'s bits, `width` bits each):
            // every lane's read from one dword where all 8 fit it, else each 128-bit half's from one where its 4 fit it, else
            // each lane's from its field's byte (kept within the period), so the step's word is a dword broadcast, or one a
            // half (Load), where its fields allow; false where a field does not fit its dword.
            bool Share(Span<int> at, Span<uint> shift, Span<long> bit, int width)
            {
                for (int l = 0; l < 8; l++)
                {
                    bit[l] = 8L * at[l] + shift[l];
                }

                if (!Base(bit, 0, 8, width, at))
                {
                    for (int h = 0; h < 8; h += 4)
                    {
                        if (!Base(bit, h, 4, width, at))
                        {
                            for (int l = h; l < h + 4; l++)
                            {
                                at[l] = (int)Math.Clamp(bit[l] / 8, 0, PeriodBytes - 4);
                            }
                        }
                    }
                }

                for (int l = 0; l < 8; l++)
                {
                    long within = bit[l] - 8L * at[l];
                    if (within < 0 || within + width > 32)
                    {
                        return false;
                    }

                    shift[l] = (uint)within;
                }

                return true;
            }

            // Lanes [from, from + count)'s fields from one dword (its first byte into `at`), where they all fit it.
            bool Base(Span<long> bit, int from, int count, int width, Span<int> at)
            {
                long low = long.MaxValue, high = long.MinValue;
                for (int l = from; l < from + count; l++)
                {
                    (low, high) = (Math.Min(low, bit[l]), Math.Max(high, bit[l] + width));
                }

                int first = (int)Math.Clamp(low / 8, 0, PeriodBytes - 4);
                if (low < 8L * first || high > 8L * first + 32)
                {
                    return false;
                }

                for (int l = from; l < from + count; l++)
                {
                    at[l] = first;
                }

                return true;
            }

            int Word(int* lanes)
            {
                var key = (lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
                if (!known.TryGetValue(key, out int at))
                {
                    known[key] = at = words.Count;
                    words.Add(new ReadOnlySpan<int>(lanes, 8).ToArray());
                }

                return at;
            }

            // Each of `count` dword vectors (8 offsets each at `offsets`) as two 16-byte windows of the period, its low half's
            // 4 dwords within the first and its high half's within the second: the windows' first bytes (two a vector) and
            // the byte shuffle that puts each lane's dword in place; false where a half's dwords lie further apart.
            bool Windows(int* offsets, int count, out int* windows, out Vector256<byte>* controls)
            {
                windows = Pin<int>(2 * Math.Max(1, count));
                controls = Pin<Vector256<byte>>(count);
                Span<byte> control = stackalloc byte[32];
                for (int i = 0; i < count; i++)
                {
                    for (int h = 0; h < 2; h++)
                    {
                        int* lanes = offsets + 8 * i + 4 * h;
                        int low = Math.Min(Math.Min(lanes[0], lanes[1]), Math.Min(lanes[2], lanes[3]));
                        int high = Math.Max(Math.Max(lanes[0], lanes[1]), Math.Max(lanes[2], lanes[3]));
                        int first = Math.Clamp(low, 0, PeriodBytes - 16);
                        if (PeriodBytes < 16 || high + 4 > first + 16)
                        {
                            return false;
                        }

                        windows[2 * i + h] = first;
                        for (int j = 0; j < 16; j++)
                        {
                            control[16 * h + j] = (byte)(lanes[j / 4] - first + j % 4);
                        }
                    }

                    controls[i] = Vector256.Create<byte>(control);
                }

                return true;
            }
        }

        // GatheredStep's constants (GatherSteps): per plane its reading, mask, field mask, bits and radix, place; per vector
        // factor its index mask, masks' table and dwords an entry.
        private byte* stepReads;
        private Vector256<uint>* stepMasks;
        private Vector256<uint>* stepFields;
        private int* stepBits;
        private int* stepLows;
        private Vector256<uint>* vectorIndexMasks;
        private nint* vectorMaskTables;
        private int* vectorDwords;
        private byte* vectorFromBits;

        // The gathered steps' dword vectors: each group's (from groupWordStart[gp], 8 offsets each in wordOffsets), and per
        // step and plane (and vector factor) the one it reads; the most a group gathers.
        private int* groupWordStart;
        private int* wordOffsets;
        // A natural plan's steps' and vector factors' words: each entry's two 16-byte windows' first bytes and its byte
        // shuffle (Window).
        private int* stepWindows, vectorWindows;
        private Vector256<byte>* stepControls, vectorControls;

        // Whether each step's lanes 2e and 2e + 1 are one entry's two dwords (a natural plan's entries of 8: GatherPairs).
        private bool pairedEntries;

        /// <summary>Whether each step's lanes 2e and 2e + 1 are one entry's two dwords (a natural plan's entries of 8: BookBytes' qword gathers).</summary>
        public bool PairedEntries => pairedEntries;

        // Whether the steps' indices are made two steps at once (a natural plan's entries of 8, no vector factors, up to
        // three planes): index step s's lanes 2e and 2e + 1 entry e's of steps 2s and 2s + 1 (IndexPosition), its planes'
        // places, windows and shifts so (half the steps' making: each step's 4 entries are 4 of its 8 lanes'), each
        // step's entries then gathered from its half (BookPair).
        private bool pairSteps;

        /// <summary>Whether the steps are made two at once (PairSteps; the one-row kernel's Pair).</summary>
        public bool PairSteps => pairSteps;
        private int* stepWord;
        private int* vectorWord;

        /// <summary>The most dword vectors a group of the gathered steps gathers (GatherSteps).</summary>
        public int MostGroupWords { get; private set; }

        /// <summary>
        /// Whether every gathered step's planes shift their 8 lanes' dwords alike (one shift each: stepShiftOne), so an
        /// instruction set that does not shift lanes by their own counts in one instruction shifts by that number (PlaneSteps).
        /// </summary>
        public bool StepShiftsUniform { get; private set; }

        /// <summary>Whether the integer lane's steps are gathered (GatherSteps: GatheredOne, GatheredFour).</summary>
        public bool StepsGathered => stepsGathered;

        /// <summary>Group <paramref name="gp"/>'s distinct dword vectors of the period at <paramref name="period"/> into <paramref name="words"/> (GatherSteps).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void GatherWords<TI>(byte* period, int gp, Vector256<uint>* words) where TI : struct, IInstructions
        {
            int start = groupWordStart[gp], count = groupWordStart[gp + 1] - start;
            int* offsets = wordOffsets + 8 * start;
            for (int u = 0; u < count; u++)
            {
                words[u] = TI.Gather(period, offsets + 8 * u).AsUInt32();
            }
        }

        /// <summary>
        /// A natural plan's dword vector of the period at <paramref name="period"/>: 16 bytes from each of its two windows
        /// (<paramref name="windows"/>: their first bytes), the low half's dwords picked from the first and the high's from
        /// the second by one byte shuffle (<paramref name="control"/>): every step's word so, no gather and no branch.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<uint> Window<TI>(byte* period, int* windows, Vector256<byte> control) where TI : struct, IInstructions =>
            TI.ShuffleBytes(Vector256.Create(Vector128.Load(period + windows[0]), Vector128.Load(period + windows[1])), control).AsUInt32();

        /// <summary>
        /// Step-and-plane entry <paramref name="i"/>'s dword vector: a natural plan's loaded from the period
        /// (<paramref name="words"/> is its first byte: Load), any other's from its group's gathered words.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<uint> Fetch<TI>(Vector256<uint>* words, int i) where TI : struct, IInstructions =>
            natural ? Window<TI>((byte*)words, stepWindows + 2 * i, stepControls[i]) : words[stepWord[i]];

        // Fetch of vector factor entry i.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<uint> FetchVector<TI>(Vector256<uint>* words, int i) where TI : struct, IInstructions =>
            natural ? Window<TI>((byte*)words, vectorWindows + 2 * i, vectorControls[i]) : words[vectorWord[i]];

        // The value at the first of the 4 positions step s's lane l takes (the step counted over the period: its group s / T):
        // the lane's chunk's positions 4s..4s + 3, or (a chunk a step: natural) the step's chunk's 4l..4l + 3.
        private int StepValue(int s, int l) => Order[StepPosition(s, l)];

        // The period's position of StepValue.
        private int StepPosition(int s, int l) => natural ? ((s / T) * Cg + s % T) * C + 4 * l : ((s / T) * Cg + l) * C + 4 * (s % T);

        // The position index step s's lane l makes the index of (GatherSteps' planes): step s's lane l's, or (PairSteps) pair
        // s's: lane 2e entry e of step 2s, lane 2e + 1 entry e of step 2s + 1 (each the entry's first value's).
        private int IndexPosition(int s, int l) => pairSteps ? StepPosition(2 * s + (l & 1), l & ~1) : StepPosition(s, l);

        // Where step s's lane l of `plane` reads its 4 fields (positions 4s..4s + 3 of its chunk, in its span's order) by
        // `read`: 4 bytes' or digits' first byte and their shift (a digit plane: the digit); an index's or a stream's first
        // field's byte and bit in it (GatherSteps picks the dword it is read from); false where the 4 fields do not lie so.
        private bool StepPlace(CodePlane plane, int s, int l, StepRead read, out int offset, out int shift)
        {
            var d = D;
            (offset, shift) = (0, 0);
            int position = IndexPosition(s, l);
            if (read == StepRead.Index)
            {
                // A codebook entry's index (entries of 4 or 8 values: the 4 positions one entry's).
                int v = Order[position];
                var (b, sh) = plane.Place(v % d.Values / d.Code.Width);
                (offset, shift) = (v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b, sh);
                return true;
            }

            long first = 0;
            for (int q = 0; q < 4; q++)
            {
                int v = Order[position + q];
                var (b, sh) = plane.Place(v % d.Values);
                int at = v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b;
                long bit = 8L * at + sh;
                if (q == 0)
                {
                    (first, offset, shift) = (bit, at, sh);
                }

                bool fits = read switch
                {
                    StepRead.Bytes => at == offset + q && sh == shift && sh + plane.Bits <= 8,
                    StepRead.Digits => at == offset + q && sh == shift,
                    _ => bit == first + (long)q * plane.Bits,
                };
                if (!fits)
                {
                    return false;
                }
            }

            if (read == StepRead.Stream)
            {
                (offset, shift) = ((int)(first / 8), (int)(first % 8));
                return true;
            }

            return offset + 4 <= PeriodBytes;
        }

        /// <summary>
        /// Step <paramref name="s"/> of the period (counted over the period) made from its group's gathered dword vectors
        /// (<paramref name="words"/>: the gathered group's, GatherWords): the integer lane's bytes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> GatheredStep<TI>(Vector256<uint>* words, int s) where TI : struct, IInstructions
        {
            // (Every constant from the plan's pinned arrays: no description, list or bounds check on this path.)
            Vector256<uint> code = default;
            for (int p = 0, i = s * Planes; p < Planes; p++, i++)
            {
                var word = TI.ShiftRight(Fetch<TI>(words, i), stepShift[i]);
                Vector256<uint> v;
                switch (stepReads[p])
                {
                    case (byte)StepRead.Bytes:
                    case (byte)StepRead.Index:
                        v = word & stepMasks[p];
                        break;
                    case (byte)StepRead.Stream:
                        int bits = stepBits[p];
                        if (bits == 1)
                        {
                            // Fields of one bit: the 4 bits to the 4 bytes' low bits by one multiply (bit i times 2^(7i) is at bit
                            // 8i, no two products share a bit).
                            v = (word & Vector256.Create(0xFu)) * Vector256.Create(0x00204081u) & Vector256.Create(0x01010101u);
                            break;
                        }

                        var field = stepFields[p];
                        v = (word & field) | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, bits) & field, 8)
                            | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, 2 * bits) & field, 16) | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, 3 * bits) & field, 24);
                        break;
                    default:
                        v = Digits<TI>(Fetch<TI>(words, i), i, p);
                        break;
                }

                code |= Vector256.ShiftLeft(v, stepLows[p]);
            }

            if (Map == CodeMap.Codebook)
            {
                return pairedEntries ? BookBytes<TI, Yes, Yes>(code, words, s) : BookBytes<TI, Yes, No>(code, words, s);
            }

            var bytes = code.AsByte();
            return Map switch
            {
                CodeMap.Flip => bytes ^ Flip,
                CodeMap.Shuffle => TI.ShuffleBytes(Shuffle, bytes),
                CodeMap.Table => Through<TI>(bytes),
                _ => bytes,
            };
        }

        // A digit plane's step from its dword vector (`word`, as stored: each byte a field's digits), step and plane entry i,
        // plane p: each byte v times its place's power t of the radix r, mod 256, times r, its top byte (the digit), even and
        // odd bytes in their own 16-bit lanes: the even byte's (v · t mod 256) · 256 is the lane times t · 256 (mod 2^16; the
        // odd byte's product lies past 16 bits), whose high 16 bits by r are the digit; the odd byte's is the lane's high
        // byte times t, whose product by r · 256 has the digit in its high byte (exactly: v · t mod 256 · r < 2^16).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal Vector256<uint> Digits<TI>(Vector256<uint> word, int i, int p) where TI : struct, IInstructions
        {
            var lane = word.AsUInt16();
            var radix = Vector256.Create((ushort)stepBits[Planes + p]);
            if (!TI.MultipliesHigh)
            {
                // (Each byte's (v · t mod 256) · r, its top byte, by low multiplies: the same digits.)
                var low = Vector256.Create((ushort)0xFF);
                var times = stepTimes[i];
                var evens = ((lane & low) * times & low) * radix >>> 8;
                var odds = ((lane >>> 8) * times & low) * radix >>> 8;
                return (evens | odds << 8).AsUInt32();
            }

            var high = Vector256.Create((ushort)0xFF00);
            var even = TI.MultiplyHigh(lane * stepTimesHigh[i], radix);
            var odd = TI.MultiplyHigh((lane & high) * stepTimes[i], radix << 8) & high;
            return (even | odd).AsUInt32();
        }

        // Step s's codebook indices (`code`) as the integer lane's bytes: the index's entry dword (each lane's within its
        // entry), the vector factors' -1s applied (TV: the plan has some, else no code for them), the least added back.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        // TP: each step's lane pairs one entry's dwords (PairedEntries: its entries gathered a qword a pair).
        private Vector256<byte> BookBytes<TI, TV, TP>(Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions where TV : struct, IFlag where TP : struct, IFlag
        {
            var entries = typeof(TP) == typeof(Yes) ? TI.GatherPairs((long*)entryBytes, code.AsInt32()).AsByte() : TI.Gather((float*)entryBytes, code.AsInt32() * (entryWidth / 4) + bookWithin[s]).AsByte();
            for (int f = 0; typeof(TV) == typeof(Yes) && f < vectorCount; f++)
            {
                int i = s * vectorCount + f;
                var index = TI.ShiftRight(FetchVector<TI>(words, i), vectorShift[i]) & vectorIndexMasks[f];
                Vector256<byte> m;
                if (vectorFromBits[f] != 0)
                {
                    // The index's bits are the signs (bit j set: value j negative): this lane's 4 bits, its dword's low
                    // byte copied to its 4 bytes, each byte's own bit tested.
                    var bits = TI.ShiftRight(index, Vector256.ShiftLeft(vectorWithin[i], 2).AsUInt32()).AsByte();
                    var one = Vector256.Create(0x08040201u).AsByte();
                    // (Byte 0 of each dword to its 4 bytes, within each 128-bit half: a constant to the JIT, where a static
                    // field read by a kernel compiled before the plan's class is made costs a call and the spills around it.)
                    var lowBytes = Vector256.Create((byte)0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
                    m = Vector256.Equals(TI.ShuffleBytes(bits, lowBytes) & one, one);
                }
                else
                {
                    m = TI.Gather((float*)vectorMaskTables[f], index.AsInt32() * vectorDwords[f] + vectorWithin[i]).AsByte();
                }

                entries = (entries ^ m) - m;
            }

            return entries + leastBytes;
        }

        // A natural plan's steps of a codebook's entries of 8 (each step 4 entries, PairedEntries) with no vector factors and
        // up to three planes, their indices made in general registers (DepositSteps): per step and plane the 8 bytes its 4
        // entries' fields lie in (from the period's first, within it), their bits extracted in order (depositExtract) and
        // deposited in 16-bit lanes at the plane's place (depositPlace), or (one field the 4 entries share) multiplied into
        // each lane's place; per plane its way (DepositKinds: 1 spread, 2 shared, 3 a run, 2 bits a plane).
        private bool depositSteps;
        private int* depositOffset;
        private ulong* depositExtract, depositPlace;

        /// <summary>Whether a row's steps make their codebook indices by bit deposit (the deposit lane's one-row kernels).</summary>
        public bool DepositSteps => depositSteps;

        /// <summary>Each plane's way of depositing its fields (2 bits a plane: 1 spread, 2 shared by the step's entries, 3 a run, its fields back to back).</summary>
        public int DepositKinds { get; private set; }

        // The deposit steps' places (Deposits), where every step's every plane takes one way: false otherwise.
        private bool Deposits()
        {
            var d = D;
            int steps = Period / GroupValues * T;
            if (!(Map == CodeMap.Codebook && pairedEntries && vectorCount == 0 && Planes is >= 1 and <= 3 && PeriodBytes >= 8))
            {
                return false;
            }

            depositOffset = Pin<int>(steps * Planes);
            depositExtract = Pin<ulong>(steps * Planes);
            depositPlace = Pin<ulong>(steps * Planes);
            Span<long> bit = stackalloc long[4];
            int kinds = 0;
            for (int p = 0; p < Planes; p++)
            {
                var plane = d.Planes[p];
                int bits = plane.Bits, low = entryLows[p], kind = 0;
                if (bits > 16 || low + bits > 16)
                {
                    return false;
                }

                ulong field = (1UL << bits) - 1;
                for (int s = 0; s < steps; s++)
                {
                    for (int e = 0; e < 4; e++)
                    {
                        int v = StepValue(s, 2 * e);
                        var (b, sh) = plane.Place(v % d.Values / d.Code.Width);
                        bit[e] = 8L * (v / d.UnitValues * d.UnitBytes + d.BlockStart(v) + b) + sh;
                    }

                    // (A run: the 4 fields back to back, read by a shift; else each extracted where it lies.)
                    bool shared = bit[1] == bit[0] && bit[2] == bit[0] && bit[3] == bit[0];
                    bool run = bit[1] == bit[0] + bits && bit[2] == bit[1] + bits && bit[3] == bit[2] + bits;
                    bool spread = bit[0] + bits <= bit[1] && bit[1] + bits <= bit[2] && bit[2] + bits <= bit[3];
                    int own = shared ? 2 : run ? 3 : spread ? 1 : 0;
                    long first = bit[0], last = bit[3] + bits;
                    int offset = (int)Math.Clamp(first / 8, 0, PeriodBytes - 8);
                    if (own == 0 || kind != 0 && own != kind || first < 8L * offset || last > 8L * offset + 64)
                    {
                        return false;
                    }

                    kind = own;
                    int i = s * Planes + p;
                    depositOffset[i] = offset;
                    ulong extract = 0, place = 0;
                    for (int e = 0; e < 4; e++)
                    {
                        extract |= field << (int)(bit[e] - 8L * offset);
                        place |= (shared ? 1UL : field) << (16 * e + low);
                    }

                    // (A run's "extract" is its shift: its first field's bit in the 8 bytes.)
                    depositExtract[i] = own == 3 ? (ulong)(first - 8L * offset) : extract;
                    depositPlace[i] = place;
                }

                kinds |= kind << (2 * p);
            }

            DepositKinds = kinds;
            depositSteps = true;
            return true;
        }

        /// <summary>
        /// Step <paramref name="s"/> of the period at <paramref name="period"/> by bit deposit (DepositSteps): its 4 entries'
        /// indices in the 16-bit lanes of one general register, each plane's fields extracted and deposited (TK0..TK2), the
        /// entries read a qword each, the least added back: the bytes BookBytes makes of the same entries.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal Vector256<byte> DepositStep<TI, TK0, TK1, TK2>(byte* period, int s)
            where TI : struct, IInstructions where TK0 : struct, IDepositPlane where TK1 : struct, IDepositPlane where TK2 : struct, IDepositPlane
        {
            int i = s * (TK2.Present ? 3 : TK1.Present ? 2 : 1);
            ulong index = TK0.Make<TI>(this, period, i);
            if (TK1.Present)
            {
                index |= TK1.Make<TI>(this, period, i + 1);
            }

            if (TK2.Present)
            {
                index |= TK2.Make<TI>(this, period, i + 2);
            }

            long* table = (long*)entryBytes;
            return Vector256.Create(table[(ushort)index], table[(ushort)(index >> 16)], table[(ushort)(index >> 32)], table[index >> 48]).AsByte() + leastBytes;
        }

        /// <summary>A plane's fields of a deposit step (DepositSteps), into their lanes' places of the index.</summary>
        internal interface IDepositPlane
        {
            static abstract bool Present { get; }

            static abstract ulong Make<TI>(DescribedPlan plan, byte* period, int i) where TI : struct, IInstructions;
        }

        internal readonly struct NoDeposit : IDepositPlane
        {
            public static bool Present => false;

            public static ulong Make<TI>(DescribedPlan plan, byte* period, int i) where TI : struct, IInstructions => 0;
        }

        // The 4 entries' own fields, in order: extracted, then deposited a 16-bit lane each.
        internal readonly struct SpreadDeposit : IDepositPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong Make<TI>(DescribedPlan plan, byte* period, int i) where TI : struct, IInstructions =>
                TI.Deposit(TI.Extract(Unsafe.ReadUnaligned<ulong>(period + plan.depositOffset[i]), plan.depositExtract[i]), plan.depositPlace[i]);
        }

        // The 4 entries' fields back to back: shifted down, then deposited a 16-bit lane each (the deposit takes as many low bits
        // as its mask has, so no extract).
        internal readonly struct RunDeposit : IDepositPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong Make<TI>(DescribedPlan plan, byte* period, int i) where TI : struct, IInstructions =>
                TI.Deposit(Unsafe.ReadUnaligned<ulong>(period + plan.depositOffset[i]) >> (int)plan.depositExtract[i], plan.depositPlace[i]);
        }

        // One field the 4 entries share: extracted, then multiplied into each lane's place.
        internal readonly struct SharedDeposit : IDepositPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong Make<TI>(DescribedPlan plan, byte* period, int i) where TI : struct, IInstructions =>
                TI.Extract(Unsafe.ReadUnaligned<ulong>(period + plan.depositOffset[i]), plan.depositExtract[i]) * plan.depositPlace[i];
        }

        /// <summary>The steps of a natural plan by bit deposit (DepositStep), its planes' ways as type parameters.</summary>
        internal readonly struct DepositedSteps<TK0, TK1, TK2> : IGatheredSteps where TK0 : struct, IDepositPlane where TK1 : struct, IDepositPlane where TK2 : struct, IDepositPlane
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                plan.DepositStep<TI, TK0, TK1, TK2>((byte*)words, s);

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static void Pair<TI>(DescribedPlan plan, Vector256<uint>* words, int s, out Vector256<byte> even, out Vector256<byte> odd) where TI : struct, IInstructions
            {
                even = plan.DepositStep<TI, TK0, TK1, TK2>((byte*)words, 2 * s);
                odd = plan.DepositStep<TI, TK0, TK1, TK2>((byte*)words, 2 * s + 1);
            }
        }

        // Pair step s's codebook indices (`code`: step 2s's entries' in the even lanes, 2s + 1's in the odd: PairSteps) as the
        // two steps' integer lane bytes, each its entries' qwords, the least added back: the same bytes BookBytes makes.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal void BookPair<TI>(Vector256<uint> code, out Vector256<byte> even, out Vector256<byte> odd) where TI : struct, IInstructions
        {
            even = TI.GatherPairs((long*)entryBytes, code.AsInt32()).AsByte() + leastBytes;
            odd = TI.GatherPairs((long*)entryBytes, (code.AsUInt64() >>> 32).AsInt32()).AsByte() + leastBytes;
        }

        /// <summary>
        /// The gathered steps' readings as type parameters (CpuBackend.WithSteps): plane 0's reading in the low 4 bits, plane
        /// 1's and plane 2's in the next 4 each (0: no such plane; 1: masked bytes or an index; 3: fields of one bit spread to
        /// bytes; 4: a stream of fields; 5: digits), the map plus one in the next (CodeMap); 0 where GatheredStep's loop reads
        /// them (more planes).
        /// </summary>
        public int StepKinds { get; private set; }

        /// <summary>How a step's plane is made from its shifted dword (GatheredStep's readings).</summary>
        internal interface IStepPlane
        {
            static abstract bool Present { get; }

            /// <summary>Whether its dword is shifted first (a digit plane's is read as it is: its fields' shift is 0).</summary>
            static abstract bool Shifted { get; }

            static abstract Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions;
        }

        internal readonly struct NoPlane : IStepPlane
        {
            public static bool Present => false;

            public static bool Shifted => true;

            public static Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions => default;
        }

        internal readonly struct MaskedPlane : IStepPlane
        {
            public static bool Present => true;

            public static bool Shifted => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions => word & plan.stepMasks[p];
        }

        internal readonly struct BitPlane : IStepPlane
        {
            public static bool Present => true;

            public static bool Shifted => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions =>
                (word & Vector256.Create(0xFu)) * Vector256.Create(0x00204081u) & Vector256.Create(0x01010101u);
        }

        internal readonly struct StreamPlane : IStepPlane
        {
            public static bool Present => true;

            public static bool Shifted => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions
            {
                int bits = plan.stepBits[p];
                var field = plan.stepFields[p];
                return (word & field) | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, bits) & field, 8)
                    | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, 2 * bits) & field, 16) | Vector256.ShiftLeft(Vector256.ShiftRightLogical(word, 3 * bits) & field, 24);
            }
        }

        internal readonly struct DigitPlane : IStepPlane
        {
            public static bool Present => true;

            public static bool Shifted => false;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read<TI>(DescribedPlan plan, Vector256<uint> word, int i, int p) where TI : struct, IInstructions =>
                plan.Digits<TI>(word, i, p);
        }

        /// <summary>How a step's code becomes the integer lane's bytes (GatheredStep's map).</summary>
        /// <summary>
        /// How the spans-made kernels (Steps) make a span's codes as the integer lane's bytes, the plan's way as a type
        /// parameter (CpuBackend.WithSpans), so each kernel holds that way's code alone: a span's bits (Raw) through the map
        /// TM, a codebook's entries by its fast reading or by any (CodebookFast, CodebookAny), or any plan's (Codes).
        /// </summary>
        internal interface ISpanCodes
        {
            static abstract Vector256<byte> Codes<TI>(DescribedPlan plan, byte* period, int span) where TI : struct, IInstructions;
        }

        internal readonly struct RawSpans<TM> : ISpanCodes where TM : struct, IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Codes<TI>(DescribedPlan plan, byte* period, int span) where TI : struct, IInstructions =>
                TM.Map<TI>(plan, plan.Raw<TI>(period, span).AsUInt32(), null, 0);
        }

        internal readonly struct FastBookSpans : ISpanCodes
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Codes<TI>(DescribedPlan plan, byte* period, int span) where TI : struct, IInstructions => plan.CodebookFast(period, span);
        }

        internal readonly struct AnyBookSpans : ISpanCodes
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Codes<TI>(DescribedPlan plan, byte* period, int span) where TI : struct, IInstructions => plan.CodebookAny(period, span);
        }

        internal interface IStepMap
        {
            static abstract Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions;
        }

        internal readonly struct SameStepMap : IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions => code.AsByte();
        }

        internal readonly struct FlipStepMap : IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions => code.AsByte() ^ plan.Flip;
        }

        internal readonly struct ShuffleStepMap : IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                TI.ShuffleBytes(plan.Shuffle, code.AsByte());
        }

        internal readonly struct TableStepMap : IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                plan.Through<TI>(code.AsByte());
        }

        // TV: the plan has vector factors (VectorCount; else its kernels hold no code for them).
        internal readonly struct BookStepMap<TV, TP> : IStepMap where TV : struct, IFlag where TP : struct, IFlag
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                plan.BookBytes<TI, TV, TP>(code, words, s);
        }

        /// <summary>The codebook's vector factors (sign vectors and the like) a gathered step applies (BookBytes).</summary>
        public int VectorCount => vectorCount;

        /// <summary>Step s of the period made from its group's gathered dword vectors (GatheredStep, or its readings as type parameters).</summary>
        internal interface IGatheredSteps
        {
            static abstract Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions;

            /// <summary>Steps 2s and 2s + 1 of the period made from pair step s's indices (PairSteps: the plan's planes up to three, a codebook's entries of 8).</summary>
            static abstract void Pair<TI>(DescribedPlan plan, Vector256<uint>* words, int s, out Vector256<byte> even, out Vector256<byte> odd) where TI : struct, IInstructions;
        }

        internal readonly struct PlanSteps : IGatheredSteps
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions => plan.GatheredStep<TI>(words, s);

            // (Never reached: a plan of pair steps has its planes' readings as type parameters, PlaneSteps.)
            public static void Pair<TI>(DescribedPlan plan, Vector256<uint>* words, int s, out Vector256<byte> even, out Vector256<byte> odd) where TI : struct, IInstructions =>
                throw new NotSupportedException("A plan of pair steps reads its planes by PlaneSteps.");
        }

        // TH: shift by the one number every lane takes (StepShiftsUniform, chosen where the instruction set does not shift
        // lanes by their own counts in one instruction: CpuBackend.StepShiftsOf; the portable instructions' shift by lanes is
        // a select for each bit of the counts, some twenty operations, one shift is one), else by the lanes' own.
        // TL: the plan is natural (its words loaded from the period, which `words` is: Load), else gathered words.
        internal readonly struct PlaneSteps<TP0, TP1, TP2, TM, TH, TL> : IGatheredSteps
            where TP0 : struct, IStepPlane where TP1 : struct, IStepPlane where TP2 : struct, IStepPlane where TM : struct, IStepMap where TH : struct, IFlag
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                TM.Map<TI>(plan, Code<TI>(plan, words, s), words, s);

            // Steps 2s and 2s + 1 from pair step s's code (the planes' places a pair step's: PairSteps), its entries by the codebook.
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static void Pair<TI>(DescribedPlan plan, Vector256<uint>* words, int s, out Vector256<byte> even, out Vector256<byte> odd) where TI : struct, IInstructions =>
                plan.BookPair<TI>(Code<TI>(plan, words, s), out even, out odd);

            // Index step s's code: its planes' fields, each in its place.
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private static Vector256<uint> Code<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions
            {
                // (The planes a constant to the JIT, as their readings are; plane 0's bits the code's lowest: no shift.)
                int i = s * (TP2.Present ? 3 : TP1.Present ? 2 : 1);
                var code = TP0.Read<TI>(plan, Word<TI, TP0>(plan, words, i), i, 0);
                if (TP1.Present)
                {
                    code |= Vector256.ShiftLeft(TP1.Read<TI>(plan, Word<TI, TP1>(plan, words, i + 1), i + 1, 1), plan.stepLows[1]);
                }

                if (TP2.Present)
                {
                    code |= Vector256.ShiftLeft(TP2.Read<TI>(plan, Word<TI, TP2>(plan, words, i + 2), i + 2, 2), plan.stepLows[2]);
                }

                return code;
            }

            // Entry i's gathered dword vector shifted right by its lanes' shifts (a plane read unshifted as it is).
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private static Vector256<uint> Word<TI, TP>(DescribedPlan plan, Vector256<uint>* words, int i) where TI : struct, IInstructions where TP : struct, IStepPlane
            {
                var word = typeof(TL) == typeof(Yes) ? Window<TI>((byte*)words, plan.stepWindows + 2 * i, plan.stepControls[i]) : words[plan.stepWord[i]];
                return !TP.Shifted ? word : typeof(TH) == typeof(Yes) ? word >>> plan.stepShiftOne[i] : TI.ShiftRight(word, plan.stepShift[i]);
            }
        }

        /// <summary>
        /// The scales (each lane's product of the scales covering its chunk, times the unit) and minimums (their product, less
        /// where they are subtracted) of group <paramref name="gp"/> of the period at <paramref name="period"/>, read for the 8
        /// lanes at once; lanes from <paramref name="valid"/> on 0 (any factor's reading: Factor{TI, TS} with Yes).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Factor<TI>(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions =>
            Factor<TI, Yes>(period, gp, valid, out scale, out min);

        /// <summary>
        /// Factor, where <typeparamref name="TS"/> says whether a factor is read a lane at a time (FactorRead.Scalar:
        /// ReadsScalar): a kernel of a plan without one holds no code for it (CpuBackend.WithFactors' FactorLoop{No}).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Factor<TI, TS>(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions where TS : struct, IFlag
        {
            if (OneScale != 0)
            {
                OneScaleFactor<TI>(period, gp, valid, out scale, out min);
                return;
            }

            var s = Vector256.Create(1f);
            var product = s;
            for (int f = 0; f < factors; f++)
            {
                var v = Read<TI, TS>(period, gp, f);
                if (f < scaleFactors)
                {
                    s = f == 0 ? v : s * v;
                }
                else
                {
                    product = f == scaleFactors ? v : product * v;
                }
            }

            if (Unit != 1f)
            {
                s *= Vector256.Create(Unit);
            }

            var mn = !Mins ? Vector256<float>.Zero : D.MinsAdded ? product : -product;
            if (valid < 8)
            {
                var keep = GroupMask(valid).AsSingle();
                s &= keep;
                mn &= keep;
            }

            scale = s;
            min = mn;
        }

        /// <summary>
        /// Factor's one scale (OneScale): its field gathered and read with the unit folded in (a 16-bit float's constants as
        /// vectors made once a plan, or a table of its numbers times the unit), no loop over factors; the kernels of a plan
        /// with one scale take it alone (CpuBackend.OneScaleRead), so no code of the loop is theirs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void OneScaleFactor<TI>(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions
        {
            var bits = fUniform[gp] != 0 ? Vector256.Create((int)(Unsafe.ReadUnaligned<uint>(period + fOffset[8 * gp]) >> fShiftOne[gp]))
                : Shifted<TI>(TI.Gather(period, fOffset + 8 * gp), gp);
            var one = OneScale == 1 ? oneHalf.Of(bits) : TI.Gather(oneTable, bits & fMask[gp]);
            scale = valid < 8 ? one & GroupMask(valid).AsSingle() : one;
            min = default;
        }

        // Entry i's gathered dwords shifted right by its lanes' shifts (by one shift where every lane's is the same and the
        // instruction set does not shift lanes by their own counts in one instruction).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector256<int> Shifted<TI>(Vector256<int> words, int i) where TI : struct, IInstructions
        {
            if (TI.ShiftsByLane)
            {
                return TI.ShiftRight(words.AsUInt32(), fShift[i].AsUInt32()).AsInt32();
            }

            int one = fShiftOne[i];
            return one switch
            {
                0 => words,
                > 0 => Vector256.ShiftRightLogical(words.AsUInt32(), one).AsInt32(),
                _ => TI.ShiftRight(words.AsUInt32(), fShift[i].AsUInt32()).AsInt32(),
            };
        }

        // Factor f's numbers for group gp's 8 lanes: its bits gathered (a dword a lane, each piece's shifted, masked and put
        // in place), then read by its type; a layout or a number the gathers do not take a lane at a time (ReadOneByOne,
        // where TS says the plan has one).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> Read<TI, TS>(byte* period, int gp, int f) where TI : struct, IInstructions where TS : struct, IFlag
        {
            if (typeof(TS) == typeof(Yes) && reads[f] == FactorRead.Scalar)
            {
                return ReadOneByOne(period, gp, f);
            }

            int at = (gp * factors + f) * pieces;
            Vector256<int> bits;
            if (fUniform[gp * factors + f] != 0)
            {
                // Every lane's the same bits: its pieces read once and broadcast.
                uint value = 0;
                for (int q = 0, i = at; q < factorPieces[f]; q++, i++)
                {
                    value |= (Unsafe.ReadUnaligned<uint>(period + fOffset[8 * i]) >> fShiftOne[i] & (uint)fMaskOne[i]) << fLeftOne[i];
                }

                bits = Vector256.Create((int)value);
            }
            else
            {
                bits = Shifted<TI>(TI.Gather(period, fOffset + 8 * at), at) & fMask[at];
                for (int q = 1; q < factorPieces[f]; q++)
                {
                    var piece = (Shifted<TI>(TI.Gather(period, fOffset + 8 * (at + q)), at + q) & fMask[at + q]).AsUInt32();
                    int one = fLeftOne[at + q];
                    bits |= (TI.ShiftsByLane || one < 0 ? TI.ShiftLeft(piece, fLeft[at + q].AsUInt32()) : Vector256.ShiftLeft(piece, one)).AsInt32();
                }
            }

            float bias = factorBias[f];
            switch (reads[f])
            {
                case FactorRead.Unsigned:
                    return Vector256.ConvertToSingle(bits) + Vector256.Create(bias);
                case FactorRead.Signed:
                    int up = 32 - factorBits[f];
                    return Vector256.ConvertToSingle(Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(bits, up), up)) + Vector256.Create(bias);
                case FactorRead.Table:
                    return TI.Gather(factorTables[f], bits);
                case FactorRead.Half:
                    return HalfBits(bits, factorHalves[f]) + Vector256.Create(bias);
                default:
                    return bits.AsSingle() + Vector256.Create(bias);       // FactorRead.Single
            }
        }

        // Factor f's numbers for group gp's 8 lanes one lane at a time, by its type (a layout or a number the gathers do not
        // take), in registers: each lane's bits from their pieces (ScalarReads' places), read as BlockScale.Value reads them
        // (ScalarOf: NumberType.Value's operations in float64, the bias added) and rounded to float32, put in its lane.
        // Nothing is written.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> ReadOneByOne(byte* period, int gp, int f)
        {
            ScalarNumber* number = scalarNumbers + f;
            int count = scalarPieces;
            var values = Vector256<float>.Zero;
            for (int l = 0, at = (gp * factors + f) * 8 * count; l < 8; l++)
            {
                ulong bits = 0;
                for (int q = 0; q < count; q++, at++)
                {
                    bits |= (ulong)(uint)(period[sOffset[at]] >> sShift[at] & sMask[at]) << sLeft[at];
                }

                var lane = Vector256.Equals(Vector256.Create(l), Vector256<int>.Indices).AsSingle();
                values = Vector256.ConditionalSelect(lane, Vector256.Create(ScalarOf(bits, number)), values);
            }

            return values;
        }

        // The number a factor's bits stand for, its bias added, rounded to float32: NumberType.Value's operations on them in
        // float64 (Math.ScaleB's as TimesTwoTo), then the bias's addition and the rounding BlockScale.Value and the lane's
        // read take, so the same bits; a table's from the plan's table of them (each the same number, made once). No call.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static float ScalarOf(ulong bits, ScalarNumber* n)
        {
            int width = n->Bits, mantissa = n->Mantissa;
            bits = width == 64 ? bits : bits & ((1UL << width) - 1);
            double value;
            switch (n->Kind)
            {
                case NumberKind.Table:
                    return n->Table[(int)bits];
                case NumberKind.Integer:
                    value = n->Signed && width < 64 ? (long)(bits << (64 - width)) >> (64 - width) : n->Signed ? (long)bits : (double)bits;
                    break;
                default:
                    // (The sign taken before the scaling, which rounds alike either way: x · 2^p's sign is x's.)
                    bool negative = n->Signed && (bits >> (width - 1) & 1) != 0;
                    ulong magnitude = n->Signed ? bits & ((1UL << (width - 1)) - 1) : bits;
                    long e = (long)(magnitude >> mantissa), top = (1L << n->Exponent) - 1;
                    ulong m = magnitude & ((1UL << mantissa) - 1);
                    if (n->Specials == FloatSpecials.Ieee && e == top)
                    {
                        value = m == 0 ? double.PositiveInfinity : double.NaN;
                        value = negative ? -value : value;
                    }
                    else if (n->Specials == FloatSpecials.AllOnesNaN && magnitude == (n->Signed ? (1UL << (width - 1)) - 1 : width == 64 ? ulong.MaxValue : (1UL << width) - 1))
                    {
                        value = double.NaN;
                        value = negative ? -value : value;
                    }
                    else
                    {
                        bool subnormal = e == 0 && n->Subnormals;
                        double x = subnormal ? m : (double)((1UL << mantissa) | m);
                        value = TimesTwoTo(negative ? -x : x, (subnormal ? 1 : (int)e) - n->ExponentBias - mantissa);
                    }

                    break;
            }

            return (float)(value + n->Bias);
        }

        // x · 2^p as Math.ScaleB makes it (musl's scalbn: x times at most three powers of two, each made from its bits), so
        // the same bits; no call.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double TimesTwoTo(double x, int p)
        {
            const double Up = 8.98846567431158E+307, Down = 2.2250738585072014E-308 * 9007199254740992;    // 2^1023, 2^-1022 · 2^53
            if (p > 1023)
            {
                x *= Up;
                p -= 1023;
                if (p > 1023)
                {
                    x *= Up;
                    p = p - 1023 > 1023 ? 1023 : p - 1023;
                }
            }
            else if (p < -1022)
            {
                x *= Down;
                p += 1022 - 53;
                if (p < -1022)
                {
                    x *= Down;
                    p = p + 1022 - 53 < -1022 ? -1022 : p + 1022 - 53;
                }
            }

            return x * BitConverter.Int64BitsToDouble((long)(0x3ff + p) << 52);
        }

        /// <summary>16-bit floats (each lane's low 16 bits) as float32 by their bits (HalfOf): exactly, specials too.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<float> HalfBits(Vector256<int> bits, (int SignShift, int Magnitude, int Shift, float Times, int Special) h)
        {
            var magnitude = bits & Vector256.Create(h.Magnitude);
            var value = Vector256.ShiftLeft(magnitude, h.Shift).AsSingle();
            if (h.Times != 1f)
            {
                value *= Vector256.Create(h.Times);
            }

            var result = value.AsInt32();
            if (h.Special != 0)
            {
                result |= Vector256.GreaterThanOrEqual(magnitude, Vector256.Create(h.Special)) & Vector256.Create(0x7F800000);
            }

            if (h.SignShift != 0)
            {
                result |= Vector256.ShiftLeft(bits & Vector256.Create(0x8000), h.SignShift);
            }

            return result.AsSingle();
        }

        /// <summary>HalfBits over 128 bits (the same operations, lane by lane).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector128<float> HalfBits(Vector128<int> bits, (int SignShift, int Magnitude, int Shift, float Times, int Special) h)
        {
            var magnitude = bits & Vector128.Create(h.Magnitude);
            var value = (magnitude << h.Shift).AsSingle();
            if (h.Times != 1f)
            {
                value *= Vector128.Create(h.Times);
            }

            var result = value.AsInt32();
            if (h.Special != 0)
            {
                result |= Vector128.GreaterThanOrEqual(magnitude, Vector128.Create(h.Special)) & Vector128.Create(0x7F800000);
            }

            if (h.SignShift != 0)
            {
                result |= (bits & Vector128.Create(0x8000)) << h.SignShift;
            }

            return result.AsSingle();
        }
    }

    // The integer lane's tiles (IntegerTile, WideTile) by the core's L1 data cache (CpuInfo): a third of it for x's codes and
    // lane data of one group of values of a block of rows of x (DescribedPlan.BlockRows), which every stored row of a tile's
    // group then reads, and a third for the stored rows of that group. Measured, 128 rows of x by 2048 → 6144 at 24 threads
    // (AVX-VNNI, 48 KB L1, ms): q4_k 1.23 / 1.17 / 1.29 by 32, 48 and 56 rows a block; 1.32 / 1.17 / 1.22 by groups of 32,
    // 16 and 8 KB of stored rows.
    private static readonly int DescribedBlockBytes = CpuInfo.Current.L1 / 3;

    // Rows of x a tile takes at most where x is int8 codes or the stored values are 4 or more bytes each (each stored row
    // read from memory once a prompt of that many rows); the float32 lane's 16-bit values take a group of stored rows by
    // RowTileRows rows.
    private const int DescribedTileRows = 128;

    // Stored bytes a group of a heavy type's rows (4 or more bytes a value) takes in the float64 lane (its tile's x passes
    // once a group, 4 rows of x in L1 a span of k).
    private const int DescribedDoubleGroupBytes = 1 << 20;

    // Stored bytes a group of 16-bit (or float32) rows takes in the float32 lane, and values of k four rows of x pass over it
    // at a time (32 KB of x stays in L1); the float64 lane's pass (four rows of x as float64, 16 KB).
    private const int DescribedSingleGroupBytes = 65536, DescribedSingleSpan = 2048, DescribedDoubleSpan = 512;

    /// <summary>
    /// The described lane as a format read where it is stored (IRowFormat): one kind of lane (DescribedLaneId) over any type
    /// with a description the plan reads (DescribedPlan), by the portable instructions and by each instruction set the CPU
    /// has that its arithmetic takes; x rounded to int8 in the plan's layout (integer), as it is or in the spans' order
    /// (float32), or widened to float64 (float64). Its tiles are columns over up to DescribedTileRows rows of x, but the
    /// float32 lane's over 16-bit values (a group of stored rows by RowTileRows rows).
    /// </summary>
    private readonly unsafe struct DescribedFormat : IRowFormat<DescribedFormat, DescribedKernels>
    {
        private readonly DescribedPlan plan;
        private readonly int k;

        private DescribedFormat(DescribedPlan plan, int k)
        {
            this.plan = plan;
            this.k = k;
        }

        public static bool Reads(TypeDescription description, int k, out DescribedFormat format)
        {
            format = k > 0 && k % description.UnitValues == 0 && description.CpuPlan is { } plan ? new DescribedFormat(plan, k) : default;
            return format.plan is not null;
        }

        /// <summary>The lanes measured for the plan's shapes (DescribedPlan.Chosen).</summary>
        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Threads, int Rows, int N, int K), (int Lane, int Generation)> Chosen => plan.Chosen;

        // The integer arithmetic's int8 dot products take AVX2's multiply-adds where its unsigned codes are at most 127 and
        // a byte plane each (DescribedPlan.PairsFit).
        public LaneList Lanes => plan.Mode == DescribedMode.Integer
            ? default(LaneList).WithKind(DescribedLaneId, dots: true, pairs: plan.P == 1 && plan.PairsFit).WithDeposits(plan.Natural is { DepositSteps: true }, plan.P == 1 && plan.PairsFit)
            : plan.FloatRows ? default(LaneList).WithKind(DescribedLaneId).With(LaneOf(ProductLaneId, Instructions.Portable)) : default(LaneList).WithKind(DescribedLaneId);

        public int First => DescribedLaneId;

        public int Second => DescribedLaneId;

        public int Unmeasured => DescribedLaneId;

        // (A lane whose instructions shift lanes by their own counts reads a row of x alone by the plan's natural one, its x
        // laid out so: Prepare's natural layout, its own preparation.)
        public int InputOf(int lane) => plan.Natural is not null && ShiftsLanes(InstructionsOf(lane)) ? 1 : 0;

        /// <summary>The plan's spans by how their codes are made (DescribedPlan.SpanKinds; tests).</summary>
        public (int Runs, int Windows, int Streams, int Codebook, int OneByOne) Spans() => plan.SpanKinds();

        /// <summary>The plan's readings no built-in type's takes (DescribedPlan.Fallbacks; tests).</summary>
        public IReadOnlyList<string> Fallbacks() => plan.Fallbacks();

        /// <summary>The arithmetic the plan took: "integer", "float32" or "float64".</summary>
        public string Arithmetic => plan.Mode switch
        {
            DescribedMode.Integer => "integer",
            DescribedMode.Single => "float32",
            _ => "float64",
        };

        private bool Heavy => plan.Bytes >= 4 * plan.Values;

        public int GroupBytes => plan.Mode switch
        {
            DescribedMode.Integer => DescribedBlockBytes,
            DescribedMode.Double => DescribedDoubleGroupBytes,
            _ => plan.Wide == WideRead.Codes ? ScaledGroupBytes : DescribedSingleGroupBytes,
        };

        // Integer arithmetic and codes: columns over up to DescribedTileRows rows of x, a quarter of a thread's share, so each
        // stored row is read from memory once a prompt of that many rows. (Groups of stored rows by 32 rows of x read
        // each row four times a 128-row prompt: q4_k's prompt 611 tokens a second under the full measured choice against 634
        // by its written kernels, though the lane measured faster at each shape on one weight held in the cache.)
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => Heavy
            ? ((total + 8 * threads - 1) / (8 * threads), Math.Min(m, DescribedTileRows))
            : plan.Mode == DescribedMode.Integer || plan.Wide == WideRead.Codes
                ? ((total + 4 * threads - 1) / (4 * threads), Math.Min(m, DescribedTileRows))
                : GroupTile(total, rowBytes, GroupBytes);

        public int Span(int k) => plan.Mode == DescribedMode.Double ? DescribedDoubleSpan : plan.Mode == DescribedMode.Single && plan.Wide != WideRead.Codes ? DescribedSingleSpan : k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    // (A cache line more of each, so that the rows can start on one: a split read cost the tile 23% on a P-core.)
                    var codes = ArrayPool<sbyte>.Shared.Rent(m * plan.CodesRow(k) + LineBytes);
                    var lanes = ArrayPool<float>.Shared.Rent(m * plan.LanesRow(k) + LineBytes / sizeof(float));
                    int codesAt = LineStart(codes), lanesAt = LineStart(lanes);
                    // A row of x alone, read by the plan's natural one (its codes a chunk at a time, as rounded), by a lane whose
                    // instructions shift each lane by its own count in one instruction, as the natural steps shift theirs (the
                    // portable instructions shift by a select for each bit of the counts: measured slower than the gathered
                    // steps for q5_0, q5_1, iq2_xxs and iq3_s, so those lanes keep them).
                    bool natural = m == 1 && InputOf(lane) == 1;
                    if (InstructionsOf(lane) == Instructions.Portable)
                    {
                        RoundDescribedRows<PortableInstructions>(plan, x, codes, codesAt, lanes, lanesAt, m, k, natural);
                    }
                    else if (AdvSimd.Arm64.IsSupported)
                    {
                        RoundDescribedRows<NeonInstructions>(plan, x, codes, codesAt, lanes, lanesAt, m, k, natural);
                    }
                    else
                    {
                        RoundDescribedRows<Avx2Instructions>(plan, x, codes, codesAt, lanes, lanesAt, m, k, natural);
                    }
                    return new LaneInput(x, false, codes, lanes, natural, codesAt, lanesAt);
                case DescribedMode.Double:
                    var wide = ArrayPool<float>.Shared.Rent(2 * m * k);
                    WidenRows(x, wide, m, k);
                    return new LaneInput(wide, true, null, null);
                default:
                    if (plan.Wide != WideRead.Codes)
                    {
                        return new LaneInput(x, false, null, null);
                    }

                    int padded = plan.Groups(k) * plan.GroupValues;
                    var ordered = ArrayPool<float>.Shared.Rent(m * padded);
                    OrderDescribedRows(plan, x, ordered, m, k, padded);
                    return new LaneInput(ordered, true, null, null);
            }
        }

        // A row's last period where it is short of one, copied into zeros as it is stored (its bytes cleared): the float32
        // lane's over codes one; the integer lane's one for each stored row of a tile's group where its rows are short (at
        // most MostGroupRows: RowGroups), else one (a row read alone).
        public int Scratch(int k) => plan.Mode switch
        {
            DescribedMode.Integer => RowBytesOf(plan.D, k) > (long)(k / plan.Period) * plan.PeriodBytes ? MostGroupRows * plan.PeriodBytes : plan.PeriodBytes,
            DescribedMode.Single when plan.Wide == WideRead.Codes => plan.PeriodBytes,
            _ => 0,
        };

        public DescribedKernels Kernels(int lane, int k, LanePointers x, byte* bytes) => new(plan, InstructionsOf(lane), NarrowOf(lane), k, x, bytes, KindOf(lane) == DepositLaneId);
    }

    // The described lane as RowGroups' kernels, each call by the lane's instruction set. Every stored value is made in
    // registers as the dot products read it: integer, a group's steps, scales and minimums made from the stored row once
    // for a block of rows of x (DescribedPlan.TileKernel) or for a row read alone (DescribedPlan.OneKernel); float32 and
    // float64, each value as it is read.
    private readonly unsafe struct DescribedKernels : IRowKernels
    {
        private readonly DescribedPlan plan;

        // The plan a row alone is read by: the natural one where x's codes lie so (LanePointers.Natural), else the plan.
        private readonly DescribedPlan one;
        private readonly Instructions with;
        private readonly bool narrow, deposit;
        private readonly int k, groups, full, tailBytes, chunks;
        private readonly nint xRow, lRow;
        private readonly float* x, lanes;
        private readonly sbyte* xq;
        private readonly byte* tail;

        public DescribedKernels(DescribedPlan plan, Instructions with, bool narrow, int k, LanePointers input, byte* tail, bool deposit = false)
        {
            this.plan = plan;
            one = input.Natural && plan.Natural is { } own ? own : plan;
            this.with = with;
            this.narrow = narrow;
            this.deposit = deposit;
            this.k = k;
            this.tail = tail;
            x = input.Values;
            xq = input.Codes;
            lanes = input.Scales;
            groups = plan.Groups(k);
            full = k / plan.Period;
            // The bytes past the whole periods: whole groups of blocks with their shared levels (a block's bytes alone would
            // leave out a group's levels stored after its blocks, so the last period's scales read as zeros).
            tailBytes = (int)(RowBytesOf(plan.D, k) - (long)full * plan.PeriodBytes);
            chunks = (k + plan.C - 1) / plan.C;
            // x's row: the integer layout's (DescribedPlan.CodesRow, LanesRow), or (float32 codes) the spans' order in whole groups.
            xRow = plan.Mode == DescribedMode.Integer ? plan.CodesRow(k) : (nint)groups * plan.GroupValues;
            lRow = plan.LanesRow(k);
        }

        // A row's last period copied (its bytes, as stored) into the zeros of `tail`, where the row is short of whole periods.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Tail(byte* row)
        {
            if (tailBytes > 0)
            {
                new ReadOnlySpan<byte>(row + (nint)full * plan.PeriodBytes, tailBytes).CopyTo(new Span<byte>(tail, tailBytes));
            }
        }

        // Integer: blocks of rows of x by groups of stored rows (its tile kernel); the float lanes four rows at a time.
        public int Block => plan.Mode == DescribedMode.Integer ? plan.BlockRows : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Tile(byte* rows, nint rowBytes, int columns, int i, int count, byte* sums, float* made)
        {
            if (tailBytes > 0)
            {
                // Each stored row's last period, short of one, copied as it is stored into its own zeros.
                for (int c = 0; c < columns; c++)
                {
                    new ReadOnlySpan<byte>(rows + c * rowBytes + (nint)full * plan.PeriodBytes, tailBytes).CopyTo(new Span<byte>(tail + (nint)c * plan.PeriodBytes, tailBytes));
                }
            }

            if (AdvSimd.Arm64.IsSupported)
            {
                switch (with)
                {
                    case Instructions.DotProd:
                        TileBy<DotProdInstructions>(rows, rowBytes, columns, i, count, sums, made);
                        return;
                    case Instructions.DotProdWide:
                        TileBy<WideDotProdInstructions>(rows, rowBytes, columns, i, count, sums, made);
                        return;
                    case Instructions.Neon:
                        TileBy<NeonInstructions>(rows, rowBytes, columns, i, count, sums, made);
                        return;
                    default:
                        TileBy<PortableInstructions>(rows, rowBytes, columns, i, count, sums, made);
                        return;
                }
            }

            switch (with)
            {
                case Instructions.Vnni:
                    TileBy<VnniInstructions>(rows, rowBytes, columns, i, count, sums, made);
                    return;
                case Instructions.Avx2:
                    TileBy<Avx2Instructions>(rows, rowBytes, columns, i, count, sums, made);
                    return;
                default:
                    TileBy<PortableInstructions>(rows, rowBytes, columns, i, count, sums, made);
                    return;
            }
        }

        // The plan's tile kernel by its choices as type parameters, made once (DescribedPlan.TileKernel); the stored rows'
        // short last periods in `tail` already, a period each.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void TileBy<TI>(byte* rows, nint rowBytes, int columns, int i, int count, byte* sums, float* made) where TI : struct, IInstructions =>
            plan.TileKernel<TI>((int)with, narrow)(plan, rows, rowBytes, columns, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow, xRow, lRow, count, sums, made);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Four(byte* row, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (AdvSimd.Arm64.IsSupported)
            {
                switch (with)
                {
                    case Instructions.DotProd:
                        FourBy<DotProdInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    case Instructions.DotProdWide:
                        FourBy<WideDotProdInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    case Instructions.Neon:
                        FourBy<NeonInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    default:
                        FourBy<PortableInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                }
            }

            switch (with)
            {
                case Instructions.Vnni:
                    FourBy<VnniInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
                case Instructions.Avx2:
                    FourBy<Avx2Instructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
                default:
                    FourBy<PortableInstructions>(row, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
            }
        }

        // The float32 lane over values it reads as they are (16-bit or float32): eight rows of x a pass, each stored value
        // made float32 once for the eight (DescribedSingleEight), or (the 128-bit twin) once for each four of them, its sums
        // in the same order: every width the same bits.
        public bool Eights => plan.Mode == DescribedMode.Single && plan.Wide != WideRead.Codes;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Eight(byte* row, int i, int p0, int p1, float* y)
        {
            if (AdvSimd.Arm64.IsSupported)
            {
                if (with == Instructions.Portable)
                {
                    EightBy<PortableInstructions>(row, i, p0, p1, y);
                    return;
                }

                EightBy<NeonInstructions>(row, i, p0, p1, y);
                return;
            }

            switch (with)
            {
                case Instructions.Vnni:
                    EightBy<VnniInstructions>(row, i, p0, p1, y);
                    return;
                case Instructions.Avx2:
                    EightBy<Avx2Instructions>(row, i, p0, p1, y);
                    return;
                default:
                    EightBy<PortableInstructions>(row, i, p0, p1, y);
                    return;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void EightBy<TI>(byte* row, int i, int p0, int p1, float* y) where TI : struct, IInstructions
        {
            if (narrow)
            {
                DescribedSingleEight<TI, Yes>(plan, row + (nint)p0 * plan.Bytes, x + (nint)i * k + p0, k, p1 - p0, y);
                DescribedSingleEight<TI, Yes>(plan, row + (nint)p0 * plan.Bytes, x + (nint)(i + 4) * k + p0, k, p1 - p0, y + 4);
                return;
            }

            DescribedSingleEight<TI, No>(plan, row + (nint)p0 * plan.Bytes, x + (nint)i * k + p0, k, p1 - p0, y);
        }

        // Four rows by the 256-bit kernels, or (the 128-bit twin) by the 128-bit ones.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void FourBy<TI>(byte* row, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    // Never four at a time: RowGroups takes the integer lane's rows in its blocks (Block, Rows).
                    throw new NotSupportedException("The integer lane takes rows of x in blocks.");
                case DescribedMode.Double:
                    DescribedDoubleFour<TI>(plan, narrow, row + (nint)p0 * plan.Bytes, (double*)x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
                    return;
                default:
                    if (plan.Wide == WideRead.Codes)
                    {
                        Tail(row);
                        if (narrow)
                        {
                            DescribedCodesFour128<TI>(plan, row, tail, full, groups, chunks, x + i * xRow, xRow, out y0, out y1, out y2, out y3);
                            return;
                        }

                        DescribedCodesFour<TI>(plan, row, tail, full, groups, chunks, x + i * xRow, xRow, out y0, out y1, out y2, out y3);
                        return;
                    }

                    DescribedSingleFour<TI>(plan, narrow, row + (nint)p0 * plan.Bytes, x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
                    return;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float Single(byte* row, int i)
        {
            if (AdvSimd.Arm64.IsSupported)
            {
                return with switch
                {
                    Instructions.DotProd => SingleBy<DotProdInstructions>(row, i),
                    Instructions.DotProdWide => SingleBy<WideDotProdInstructions>(row, i),
                    Instructions.Neon => SingleBy<NeonInstructions>(row, i),
                    _ => SingleBy<PortableInstructions>(row, i),
                };
            }

            return with switch
            {
                Instructions.Vnni => SingleBy<VnniInstructions>(row, i),
                Instructions.Avx2 => SingleBy<Avx2Instructions>(row, i),
                _ => SingleBy<PortableInstructions>(row, i),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private float SingleBy<TI>(byte* row, int i) where TI : struct, IInstructions
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    Tail(row);
                    return IntegerOne<TI>(row, i);
                case DescribedMode.Double:
                    return DescribedDoubleOne<TI>(plan, row, (double*)x + (nint)i * k, k);
                default:
                    if (plan.Wide == WideRead.Codes)
                    {
                        Tail(row);
                        return DescribedCodesOne<TI>(plan, row, tail, full, groups, chunks, x + i * xRow);
                    }

                    return DescribedSingleOne<TI>(plan, row, x + (nint)i * k, k);
            }
        }

        // Row i of x by the stored row at `row` (its last period in `tail` already): the plan's kernel by its choices as type
        // parameters, made once (DescribedPlan.OneKernel), or DescribedIntegerOne's dispatch.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private float IntegerOne<TI>(byte* row, int i) where TI : struct, IInstructions
        {
            var kernel = deposit ? one.DepositKernel<TI>((int)with) : one.OneKernel<TI>((int)with);
            return kernel != null ? kernel(one, row, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow)
                : DescribedIntegerOne<TI>(plan, row, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow);
        }
    }

    // x rounded to int8 for the described lane's integer arithmetic, row by row (RoundDescribedRow), over the threads where
    // x is large (a row a chunk, CpuRanges: no closure or task allocated a call, as the rows alone ran before).
    private static void RoundDescribedRows<TI>(DescribedPlan plan, float[] x, sbyte[] codes, int codesAt, float[] lanes, int lanesAt, int m, int k, bool natural)
        where TI : struct, IInstructions
    {
        var rows = new RoundedRows<TI>(plan, x, codes, codesAt, lanes, lanesAt, k, natural);
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            CpuRanges.Run(m, 1, m, Math.Max(1, ComputeResources.MaxCpuThreads), ref rows);   // `rows` is this frame's, on the stack
            return;
        }

        rows.Run(0, m);
    }

    // RoundDescribedRows' rows [first, last), each rounded on its own (the same bytes whichever thread rounds it).
    private readonly struct RoundedRows<TI>(DescribedPlan plan, float[] x, sbyte[] codes, int codesAt, float[] lanes, int lanesAt, int k, bool natural) : IRangeBody
        where TI : struct, IInstructions
    {
        private readonly int xRow = plan.CodesRow(k), lRow = plan.LanesRow(k);

        public void Run(int first, int last)
        {
            for (int i = first; i < last; i++)
            {
                RoundDescribedRow<TI>(plan, x.AsSpan(i * k, k), codes.AsSpan(codesAt + i * xRow, xRow), lanes.AsSpan(lanesAt + i * lRow, lRow), natural);
            }
        }
    }

    // The element of `array` from which its data start on a cache line (where it lies now: a large array, as these are, is
    // not moved; one moved later is read from the same element, only not on a line's start).
    private static unsafe int LineStart<T>(T[] array) where T : unmanaged
    {
        nint at = (nint)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
        return (int)(LineAbove(at) - at) / sizeof(T);
    }

    // The machine's cache line in bytes (CpuInfo: the system's, or its fallback).
    private static readonly int LineBytes = CpuInfo.Current.CacheLine;

    // The first address at or past `at` on a cache line's start.
    private static nint LineAbove(nint at) => at + (LineBytes - at % LineBytes) % LineBytes;

    // A row of x rounded to int8 a block of 32 values at a time (the block's scale amax / 127, 0 where every value is 0; its
    // codes value · 127 / amax rounded to even), laid out as the plan's steps are (XPlace): each block put in its span's
    // order by two byte shuffles (OrderShuffles; none where spans are in value order), a group's blocks transposed as the
    // stored codes are (one chunk a lane of each step; a wide integer's chunks 4 bytes each of a step's 4 · Cg); per group
    // LaneFloats floats: each lane's block scale, as int32 bits its plane's least times its chunk's code sum, and (minimums)
    // its block scale times its chunk's code sum (in plane 0's lanes). Blocks past k are zero codes, scales and sums. The
    // integer lane's product is then the product of x so rounded. Made by the instruction set of the lane that prepares it
    // (values moved and integers summed: the same bytes by any set, so the lanes of every set share it).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RoundDescribedRow<TI>(DescribedPlan plan, ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> lanes, bool natural) where TI : struct, IInstructions
    {
        int k = x.Length, groups = plan.Groups(k), spansPerPeriod = plan.Period / 32, spans = plan.GroupValues / 32, p = plan.P, laneFloats = plan.LaneFloats;
        bool half = plan.C == 16, mins = plan.Mins;
        int* least = plan.PlaneLeast;
        Vector256<byte>* order = plan.OrderShuffles;
        Span<float> block = stackalloc float[32];
        Span<float> scales = stackalloc float[8];
        Span<int> sums = stackalloc int[16];
        Vector256<byte>* made = stackalloc Vector256<byte>[8];
        ref sbyte into = ref MemoryMarshal.GetReference(codes);
        for (int g = 0; g < groups; g++)
        {
            for (int q = 0; q < spans; q++)
            {
                int from = g * plan.GroupValues + 32 * q, count = Math.Clamp(k - from, 0, 32);
                if (count == 0)
                {
                    made[q] = default;
                    scales[q] = 0f;
                    sums[2 * q] = sums[2 * q + 1] = 0;
                    continue;
                }

                scoped ReadOnlySpan<float> values = x.Slice(from, count);
                if (count < 32)
                {
                    block.Clear();
                    values.CopyTo(block);
                    values = block;
                }

                ref float v = ref MemoryMarshal.GetReference(values);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var top = Vector128.Max(most.GetLower(), most.GetUpper());
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(2, 3, 0, 1)));
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(1, 0, 3, 2)));
                float amax = top.ToScalar();
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var rounded = Vector256.Narrow(
                    Vector256.Narrow(Vector256.ConvertToInt32(Vector256.Round(v0 * inverse)), Vector256.ConvertToInt32(Vector256.Round(v1 * inverse))),
                    Vector256.Narrow(Vector256.ConvertToInt32(Vector256.Round(v2 * inverse)), Vector256.ConvertToInt32(Vector256.Round(v3 * inverse)))).AsByte();
                if (order != null)
                {
                    int r = from / 32 % spansPerPeriod;
                    var lower = rounded.GetLower();
                    var upper = rounded.GetUpper();
                    rounded = TI.ShuffleBytes(Vector256.Create(lower, lower), order[2 * r]) | TI.ShuffleBytes(Vector256.Create(upper, upper), order[2 * r + 1]);
                }

                made[q] = rounded;
                scales[q] = amax / 127f;
                var codes16 = rounded.AsSByte();
                sums[2 * q] = Vector128.Sum(Vector128.WidenLower(codes16.GetLower()) + Vector128.WidenUpper(codes16.GetLower()));
                sums[2 * q + 1] = Vector128.Sum(Vector128.WidenLower(codes16.GetUpper()) + Vector128.WidenUpper(codes16.GetUpper()));
            }

            byte* to = (byte*)Unsafe.AsPointer(ref Unsafe.Add(ref into, g * plan.GroupValues));
            switch (p)
            {
                case 1 when natural:
                    // (The natural layout: each chunk's codes as rounded, chunk q at 32q.)
                    for (int q = 0; q < 8; q++)
                    {
                        made[q].Store(to + 32 * q);
                    }

                    break;
                case 1 when !half:
                    Transpose8<TI>(made[0], made[1], made[2], made[3], made[4], made[5], made[6], made[7], out var s0, out var s1, out var s2, out var s3,
                        out var s4, out var s5, out var s6, out var s7);
                    s0.Store(to);
                    s1.Store(to + 32);
                    s2.Store(to + 64);
                    s3.Store(to + 96);
                    s4.Store(to + 128);
                    s5.Store(to + 160);
                    s6.Store(to + 192);
                    s7.Store(to + 224);
                    break;
                case 1:
                    Transpose4<TI>(made[0], made[1], made[2], made[3], out var h0, out var h1, out var h2, out var h3);
                    h0.Store(to);
                    h1.Store(to + 32);
                    h2.Store(to + 64);
                    h3.Store(to + 96);
                    break;
                case 8:
                    made[0].Store(to);
                    break;
                case 4:
                    var lo = TI.UnpackLow(made[0].AsUInt32(), made[1].AsUInt32()).AsUInt64();
                    var hi = TI.UnpackHigh(made[0].AsUInt32(), made[1].AsUInt32()).AsUInt64();
                    TI.Lows(lo, hi).AsByte().Store(to);
                    TI.Highs(lo, hi).AsByte().Store(to + 32);
                    break;
                default:
                    var t0 = TI.UnpackLow(made[0].AsUInt32(), made[1].AsUInt32());
                    var t1 = TI.UnpackHigh(made[0].AsUInt32(), made[1].AsUInt32());
                    var t2 = TI.UnpackLow(made[2].AsUInt32(), made[3].AsUInt32());
                    var t3 = TI.UnpackHigh(made[2].AsUInt32(), made[3].AsUInt32());
                    var u0 = TI.UnpackLow(t0.AsUInt64(), t2.AsUInt64());
                    var u1 = TI.UnpackHigh(t0.AsUInt64(), t2.AsUInt64());
                    var u2 = TI.UnpackLow(t1.AsUInt64(), t3.AsUInt64());
                    var u3 = TI.UnpackHigh(t1.AsUInt64(), t3.AsUInt64());
                    TI.Lows(u0, u1).AsByte().Store(to);
                    TI.Lows(u2, u3).AsByte().Store(to + 32);
                    TI.Highs(u0, u1).AsByte().Store(to + 64);
                    TI.Highs(u2, u3).AsByte().Store(to + 96);
                    break;
            }

            var own = lanes.Slice(g * laneFloats, laneFloats);
            if (p > 1)
            {
                // A wide integer's chunks: each its block scale, then its top plane's least times its code sum (spread to the
                // chunk's lanes as x's codes are: IStepLoad).
                var tops = MemoryMarshal.Cast<float, int>(own[plan.Cg..]);
                for (int j = 0; j < plan.Cg; j++)
                {
                    own[j] = scales[j];
                    tops[j] = least[p - 1] * (sums[2 * j] + sums[2 * j + 1]);
                }

                continue;
            }

            var starts = MemoryMarshal.Cast<float, int>(own.Slice(8, 8));
            for (int l = 0; l < 8; l++)
            {
                int j = l / p, plane = l % p;
                float xs = half ? scales[j / 2] : scales[j];
                int sum = half ? sums[j] : sums[2 * j] + sums[2 * j + 1];
                own[l] = xs;
                starts[l] = least[plane] * sum;
                if (mins)
                {
                    own[16 + l] = plane == 0 ? xs * sum : 0f;
                }
            }
        }
    }

    // x in the spans' order for the float32 lane over codes (OrderDescribedRow), row by row, each `padded` (whole groups).
    private static void OrderDescribedRows(DescribedPlan plan, float[] x, float[] into, int m, int k, int padded)
    {
        for (int i = 0; i < m; i++)
        {
            OrderDescribedRow(plan, x.AsSpan(i * k, k), into.AsSpan(i * padded, padded));
        }
    }

    // A row of x in the order each span's codes are made (DescribedPlan.Order: position t of a period takes the value at
    // Order[t]), whole groups: the positions of values past the row 0.
    private static unsafe void OrderDescribedRow(DescribedPlan plan, ReadOnlySpan<float> x, Span<float> into)
    {
        int period = plan.Period;
        int* order = plan.Order;
        for (int at = 0; at < into.Length; at++)
        {
            int inPeriod = at % period, v = at - inPeriod + order[inPeriod];
            into[at] = v < x.Length ? x[v] : 0f;
        }
    }

    /// <summary>
    /// How the integer lane loads a step of x's codes (P = 1, 8 chunks' 4 bytes; P = 2, 4 chunks' each in 2 lanes; P = 4, 2
    /// chunks' each in 4; P = 8, one chunk's in all 8), and, for a wide integer (P > 1), makes step s of a group of its
    /// stored codes (chunk j's from `at` + 32 · P · j): lane j · P + p byte p of chunk j's codes 4s..4s + 3.
    /// </summary>
    private unsafe interface IStepLoad
    {
        static abstract Vector256<sbyte> Load<TI>(sbyte* at) where TI : struct, IInstructions;

        /// <summary>Load's lower 128-bit half (lanes 0-3), or its upper (4-7): the 128-bit twins' four-row kernels'.</summary>
        static abstract Vector128<sbyte> Half<TI>(sbyte* at, bool upper) where TI : struct, IInstructions;

        static abstract Vector256<byte> Plane<TI>(byte* at, int s) where TI : struct, IInstructions;
    }

    private unsafe struct EightChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load<TI>(sbyte* at) where TI : struct, IInstructions => Vector256.Load(at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector128<sbyte> Half<TI>(sbyte* at, bool upper) where TI : struct, IInstructions => Vector128.Load(upper ? at + 16 : at);

        public static Vector256<byte> Plane<TI>(byte* at, int s) where TI : struct, IInstructions => throw new NotSupportedException("Codes of at most 8 bits have one byte plane.");
    }

    private unsafe struct FourChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load<TI>(sbyte* at) where TI : struct, IInstructions
        {
            var v = Vector128.Load((int*)at);
            return TI.Permute(Vector256.Create(v, v), Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3)).AsSByte();
        }

        // Dwords 0, 0, 1, 1 (the lower half) or 2, 2, 3, 3 (the upper): each dword interleaved with itself.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector128<sbyte> Half<TI>(sbyte* at, bool upper) where TI : struct, IInstructions
        {
            var v = Vector128.Load((uint*)at);
            return (upper ? TI.UnpackHigh(v, v) : TI.UnpackLow(v, v)).AsSByte();
        }

        // 16-bit codes: 4 chunks' 4 codes (8 bytes each, 64 bytes apart), each 128-bit half's two chunks transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane<TI>(byte* at, int s) where TI : struct, IInstructions =>
            TI.ShuffleHalves(Vector256.Create(*(ulong*)(at + 8 * s), *(ulong*)(at + 64 + 8 * s), *(ulong*)(at + 128 + 8 * s), *(ulong*)(at + 192 + 8 * s)).AsByte(),
                EvenOddPairs);
    }

    private unsafe struct TwoChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load<TI>(sbyte* at) where TI : struct, IInstructions =>
            TI.Permute(Vector256.CreateScalarUnsafe(*(long*)at).AsInt32(), Vector256.Create(0, 0, 0, 0, 1, 1, 1, 1)).AsSByte();

        // Dword 0 in every lane of the lower half, dword 1 in the upper's.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector128<sbyte> Half<TI>(sbyte* at, bool upper) where TI : struct, IInstructions => Vector128.Create(*((int*)at + (upper ? 1 : 0))).AsSByte();

        // 32-bit codes: 2 chunks' 4 codes (16 bytes each, 128 bytes apart), each 128-bit half's 4 × 4 bytes transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane<TI>(byte* at, int s) where TI : struct, IInstructions =>
            TI.ShuffleHalves(Vector256.Create(Vector128.Load(at + 16 * s), Vector128.Load(at + 128 + 16 * s)),
                TransposeQuads);
    }

    private unsafe struct OneChunk : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load<TI>(sbyte* at) where TI : struct, IInstructions => Vector256.Create(*(int*)at).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector128<sbyte> Half<TI>(sbyte* at, bool upper) where TI : struct, IInstructions => Vector128.Create(*(int*)at).AsSByte();

        // 64-bit codes: one chunk's 4 codes (32 bytes), their low dwords then their high ones, each half's 4 × 4 bytes transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane<TI>(byte* at, int s) where TI : struct, IInstructions =>
            TI.ShuffleHalves(TI.Permute(Vector256.Load((int*)(at + 32 * s)), Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7)).AsByte(),
                TransposeQuads);
    }

    // f + a wide integer's group's sums for a row of x: its 8 steps dotted with x's (TX), each lane from x's start (the top
    // plane's least times the chunk's code sum), then times x's block scales (both spread to the chunks' lanes by TX).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<float> WideRow<TX, TI>(Vector256<float> f, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, sbyte* x, int xStep, float* lanes, int cg, Vector256<int> top)
        where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        var i = TX.Load<TI>((sbyte*)(lanes + cg)).AsInt32() & top;
        i = TI.Dot(i, s0, TX.Load<TI>(x));
        i = TI.Dot(i, s1, TX.Load<TI>(x + xStep));
        i = TI.Dot(i, s2, TX.Load<TI>(x + 2 * xStep));
        i = TI.Dot(i, s3, TX.Load<TI>(x + 3 * xStep));
        i = TI.Dot(i, s4, TX.Load<TI>(x + 4 * xStep));
        i = TI.Dot(i, s5, TX.Load<TI>(x + 5 * xStep));
        i = TI.Dot(i, s6, TX.Load<TI>(x + 6 * xStep));
        i = TI.Dot(i, s7, TX.Load<TI>(x + 7 * xStep));
        return TI.Fused(Vector256.ConvertToSingle(i), TX.Load<TI>((sbyte*)lanes).AsSingle(), f);
    }

    // A wide integer's one row of x by one stored row read alone: each group's 8 steps made in registers (TX.Plane) as they
    // are dotted with x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float WideMadeOne<TX, TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, sbyte* x, float* lanes)
        where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        int periodBytes = plan.PeriodBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var flip = plan.TopFlip;
        var top = plan.TopLanes;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = g < full ? row + (nint)g * periodBytes : tail;
            f = WideRow<TX, TI>(f, TX.Plane<TI>(at, 0) ^ flip, TX.Plane<TI>(at, 1) ^ flip, TX.Plane<TI>(at, 2) ^ flip, TX.Plane<TI>(at, 3) ^ flip, TX.Plane<TI>(at, 4) ^ flip,
                TX.Plane<TI>(at, 5) ^ flip, TX.Plane<TI>(at, 6) ^ flip, TX.Plane<TI>(at, 7) ^ flip, x + (nint)g * groupValues, xStep, lanes + (nint)g * laneFloats, cg, top);
        }

        return Vector256.Sum(f * plan.LaneWeights!.Value);
    }

    /// <summary>
    /// Whether a tile kernel of the integer lane runs on 256-bit vectors (Vectors256) or, a 128-bit twin's (NarrowLane), on
    /// 128-bit ones (Vectors128), each 256-bit vector of the other as its two halves: a type parameter the kernels test by
    /// its type, so the JIT imports the one width's code alone.
    /// </summary>
    private interface IVectorWidth
    {
    }

    private readonly struct Vectors256 : IVectorWidth
    {
    }

    private readonly struct Vectors128 : IVectorWidth
    {
    }

    /// <summary>A plan's yes or no as a type parameter (Yes, No), tested by its type: a constant to the JIT in the tile kernels' loops.</summary>
    internal interface IFlag
    {
    }

    internal readonly struct Yes : IFlag
    {
    }

    internal readonly struct No : IFlag
    {
    }

    /// <summary>
    /// How the integer lane's tile kernel (P = 1: IntegerTile) makes a group's steps in registers from the stored bytes of
    /// its period (lane l of step s: 4 bytes of chunk l, its positions 4s..4s + 3 in its span's order; T = 4: s4..s7 not
    /// written): the plan's uniform runs loaded (LoadedGroup) or gathered (GatheredRunsGroup), its gathered planes
    /// (GatheredGroup) or its own spans (MadeGroup), each as its one-row kernel makes them. Made once a call from the plan
    /// (Of); `words`: the call's room for a group's gathered dwords, the stored bytes as they are (WordCount of them).
    /// </summary>
    private unsafe interface IGroupSteps<TSelf> where TSelf : struct, IGroupSteps<TSelf>
    {
        static abstract TSelf Of(DescribedPlan plan);

        static abstract int WordCount(DescribedPlan plan);

        void Make<TI>(DescribedPlan plan, byte* at, int gp, Vector256<uint>* words, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions;
    }

    // UniformOne's steps loaded a span at a time (TR) and mapped (TM), then transposed (UniformSteps): 8 spans (TE: chunks
    // of 32), else 4. A plan whose steps are gathered takes GatheredRunsGroup instead, chosen once a plan (OneByMap,
    // TileByMap), so neither kernel holds the other's code.
    private readonly unsafe struct LoadedGroup<TR, TM, TE> : IGroupSteps<LoadedGroup<TR, TM, TE>> where TR : struct, IRunLoad where TM : struct, ICodeMap where TE : struct, IFlag
    {
        private readonly int* offsets, counts;
        private readonly Vector256<ulong>* shifts;
        private readonly Vector256<byte> mask, table;

        // (Inlined, as Of: a call would leave the struct in memory, its vectors read from the stack in the loops.)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private LoadedGroup(DescribedPlan plan)
        {
            offsets = plan.RunOffsets;
            shifts = plan.RunShifts;
            counts = plan.RunCounts;
            mask = plan.PlaneMask0;
            table = plan.MapVector;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static LoadedGroup<TR, TM, TE> Of(DescribedPlan plan) => new(plan);

        public static int WordCount(DescribedPlan plan) => 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Make<TI>(DescribedPlan plan, byte* at, int gp, Vector256<uint>* words, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions =>
            UniformSteps<TR, TM, TI>(at, gp, offsets, shifts, counts, mask, table, typeof(TE) == typeof(Yes), out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
    }

    // UniformOne's steps gathered (DescribedPlan.Gathered: chunks of 32, TR's runs a gather each, GatheredSteps), mapped by
    // TM (flipped codes: the flip XORed into the gathered words, the steps then taken as they are).
    private readonly unsafe struct GatheredRunsGroup<TR, TM> : IGroupSteps<GatheredRunsGroup<TR, TM>> where TR : struct, IRunLoad where TM : struct, ICodeMap
    {
        private readonly int* gatherOffsets, gatherShift;
        private readonly Vector256<uint>* gatherShifts;
        private readonly Vector256<byte> mask, table;
        private readonly Vector256<uint> wordFlip;

        // (Inlined, as Of: a call would leave the struct in memory, its vectors read from the stack in the loops.)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private GatheredRunsGroup(DescribedPlan plan)
        {
            mask = plan.PlaneMask0;
            table = plan.MapVector;
            gatherOffsets = plan.GatherOffsets;
            gatherShifts = plan.GatherShifts;
            gatherShift = plan.GatherShift;
            wordFlip = plan.WordFlip;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static GatheredRunsGroup<TR, TM> Of(DescribedPlan plan) => new(plan);

        public static int WordCount(DescribedPlan plan) => 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Make<TI>(DescribedPlan plan, byte* at, int gp, Vector256<uint>* words, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions
        {
            int runs = TR.Count;
            if (typeof(TM) == typeof(FlippedCodes))
            {
                GatheredSteps<SameCodes, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, wordFlip, mask, table,
                    out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                return;
            }

            GatheredSteps<TM, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, default, mask, table,
                out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
        }
    }

    // GatheredOne's steps: the group's dword vectors gathered as they are stored (DescribedPlan.GatherWords), each step made
    // from them by TS.
    private readonly unsafe struct GatheredGroup<TS> : IGroupSteps<GatheredGroup<TS>> where TS : struct, DescribedPlan.IGatheredSteps
    {
        public static GatheredGroup<TS> Of(DescribedPlan plan) => default;

        public static int WordCount(DescribedPlan plan) => Math.Max(1, plan.MostGroupWords);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Make<TI>(DescribedPlan plan, byte* at, int gp, Vector256<uint>* words, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions
        {
            int first = gp * plan.T;
            plan.GatherWords<TI>(at, gp, words);
            s0 = TS.Step<TI>(plan, words, first);
            s1 = TS.Step<TI>(plan, words, first + 1);
            s2 = TS.Step<TI>(plan, words, first + 2);
            s3 = TS.Step<TI>(plan, words, first + 3);
            if (plan.T == 8)
            {
                s4 = TS.Step<TI>(plan, words, first + 4);
                s5 = TS.Step<TI>(plan, words, first + 5);
                s6 = TS.Step<TI>(plan, words, first + 6);
                s7 = TS.Step<TI>(plan, words, first + 7);
                return;
            }

            s4 = s5 = s6 = s7 = default;
        }
    }

    // MadeOne's steps: the plan's spans made one at a time, TS's way, and put in their steps (DescribedPlan.Steps).
    private readonly unsafe struct MadeGroup<TS> : IGroupSteps<MadeGroup<TS>> where TS : struct, DescribedPlan.ISpanCodes
    {
        public static MadeGroup<TS> Of(DescribedPlan plan) => default;

        public static int WordCount(DescribedPlan plan) => 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Make<TI>(DescribedPlan plan, byte* at, int gp, Vector256<uint>* words, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions =>
            plan.Steps<TI, TS>(at, gp, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
    }

    // The integer lane's tile (P = 1): `columns` stored rows (from `rows`, `rowBytes` apart, each read where it is stored;
    // row c's last period from tails + c · PeriodBytes where it is short) by `count` rows of x (codes from `x`, a row every
    // `xRow`; lane data from `lanes`, a row every `lRow`). A group at a time, then each stored row: its group's steps (TG),
    // scales and minimums (TF) made in registers from the stored bytes once and held while each row of x is dotted with them
    // in turn (ChunkRowMin), so the group's x stays in the cache across the stored rows; each pair's sums in its own vector
    // of `sums` (columns · count of them, 32 bytes each: the outputs', no stored value), its product into
    // made[c · count + r]. Each pair's operations its one-row kernel's (UniformOne, GatheredOne, MadeOne), in their order.
    // TW: on 128-bit vectors (each step's halves, ChunkRow128; each pair's sums two halves). TE: 8 steps a group (else 4,
    // chunks of 16); TM: minimums; TT: four rows of x a pass (two on 128-bit vectors), where the instruction set dots them
    // together (IInstructions.RowsTogether, read once a plan: IntegerTileUse), else a row a pass; constants to the JIT, so no
    // row tests them.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void IntegerTile<TI, TG, TF, TW, TE, TM, TT>(DescribedPlan plan, byte* rows, nint rowBytes, int columns, byte* tails, int full, int groups, int chunks,
        sbyte* x, float* lanes, nint xRow, nint lRow, int count, byte* sums, float* made)
        where TI : struct, IInstructions where TG : struct, IGroupSteps<TG> where TF : struct, IFactors where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag
        where TT : struct, IFlag
    {
        int gpp = plan.GroupsPerPeriod, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        bool mins = typeof(TM) == typeof(Yes), eight = typeof(TE) == typeof(Yes), together = typeof(TT) == typeof(Yes);
        var steps = TG.Of(plan);
        Vector256<uint>* words = stackalloc Vector256<uint>[TG.WordCount(plan)];
        new Span<byte>(sums, 32 * columns * count).Clear();
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            nint offset = (nint)period * periodBytes;
            bool whole = period < full;
            sbyte* x0 = x + (nint)g * groupValues;
            float* l0 = lanes + (nint)g * laneFloats;
            for (int c = 0; c < columns; c++)
            {
                byte* at = whole ? rows + c * rowBytes + offset : tails + (nint)c * periodBytes;
                steps.Make<TI>(plan, at, gp, words, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
                TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
                sbyte* xg = x0;
                float* lg = l0;
                int r = 0;
                if (typeof(TW) == typeof(Vectors128))
                {
                    var h = (Vector128<float>*)sums + 2 * c * count;
                    if (!mins)
                    {
                        // Every row's lower half (chunks 0-3), then every row's upper half (4-7): eight half-steps held at a time.
                        HalfRows<TI>(h, count, eight, together, s0.GetLower(), s1.GetLower(), s2.GetLower(), s3.GetLower(), s4.GetLower(), s5.GetLower(),
                            s6.GetLower(), s7.GetLower(), scale.GetLower(), xg, xRow, lg, lRow);
                        HalfRows<TI>(h + 1, count, eight, together, s0.GetUpper(), s1.GetUpper(), s2.GetUpper(), s3.GetUpper(), s4.GetUpper(), s5.GetUpper(),
                            s6.GetUpper(), s7.GetUpper(), scale.GetUpper(), xg + 16, xRow, lg + 4, lRow);
                        continue;
                    }

                    // With minimums, both halves a row at a time (the half passes measured 3% slower for q4_1 and q4_k on one
                    // core without AVX2, the same with it).
                    Vector128<byte> s0l = s0.GetLower(), s0u = s0.GetUpper(), s1l = s1.GetLower(), s1u = s1.GetUpper(), s2l = s2.GetLower(), s2u = s2.GetUpper(),
                        s3l = s3.GetLower(), s3u = s3.GetUpper(), s4l = s4.GetLower(), s4u = s4.GetUpper(), s5l = s5.GetLower(), s5u = s5.GetUpper(),
                        s6l = s6.GetLower(), s6u = s6.GetUpper(), s7l = s7.GetLower(), s7u = s7.GetUpper();
                    Vector128<float> scalel = scale.GetLower(), scaleu = scale.GetUpper(), minl = min.GetLower(), minu = min.GetUpper();
                    for (; together && r + 2 <= count; r += 2, xg += 2 * xRow, lg += 2 * lRow)
                    {
                        ChunkHalves<TI>(h + 2 * r, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scalel, scaleu, minl, minu, xg, lg);
                        ChunkHalves<TI>(h + 2 * r + 2, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scalel, scaleu, minl, minu,
                            xg + xRow, lg + lRow);
                    }

                    for (; r < count; r++, xg += xRow, lg += lRow)
                    {
                        ChunkHalves<TI>(h + 2 * r, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scalel, scaleu, minl, minu, xg, lg);
                    }

                    continue;
                }

                var f = (Vector256<float>*)sums + c * count;
                for (; together && r + 4 <= count; r += 4, xg += 4 * xRow, lg += 4 * lRow)
                {
                    f[r] = ChunkRowMin<TI>(f[r], eight, mins, s0, s1, s2, s3, s4, s5, s6, s7, scale, min, xg, lg);
                    f[r + 1] = ChunkRowMin<TI>(f[r + 1], eight, mins, s0, s1, s2, s3, s4, s5, s6, s7, scale, min, xg + xRow, lg + lRow);
                    f[r + 2] = ChunkRowMin<TI>(f[r + 2], eight, mins, s0, s1, s2, s3, s4, s5, s6, s7, scale, min, xg + 2 * xRow, lg + 2 * lRow);
                    f[r + 3] = ChunkRowMin<TI>(f[r + 3], eight, mins, s0, s1, s2, s3, s4, s5, s6, s7, scale, min, xg + 3 * xRow, lg + 3 * lRow);
                }

                for (; r < count; r++, xg += xRow, lg += lRow)
                {
                    f[r] = ChunkRowMin<TI>(f[r], eight, mins, s0, s1, s2, s3, s4, s5, s6, s7, scale, min, xg, lg);
                }
            }
        }

        int pairs = columns * count;
        if (typeof(TW) == typeof(Vectors128))
        {
            var h = (Vector128<float>*)sums;
            for (int q = 0; q < pairs; q++)
            {
                made[q] = SumOf(h[2 * q], h[2 * q + 1]);
            }

            return;
        }

        var all = (Vector256<float>*)sums;
        for (int q = 0; q < pairs; q++)
        {
            made[q] = Vector256.Sum(all[q]);
        }
    }

    // A wide integer's tile (P > 1), as IntegerTile: a group at a time, then each stored row, its 8 steps made in registers
    // once (TX.Plane, as WideMadeOne makes them) and held while each row of x is dotted with them in turn (WideRow), each
    // pair's sums weighted by the planes at the end: each pair's operations WideMadeOne's, in its order. TW: on 128-bit
    // vectors (each step's halves dotted with x's matching halves, WideRow128).
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void WideTile<TX, TI, TW>(DescribedPlan plan, byte* rows, nint rowBytes, int columns, byte* tails, int full, int groups, int chunks,
        sbyte* x, float* lanes, nint xRow, nint lRow, int count, byte* sums, float* made) where TX : struct, IStepLoad where TI : struct, IInstructions where TW : struct, IVectorWidth
    {
        int periodBytes = plan.PeriodBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var flip = plan.TopFlip;
        var top = plan.TopLanes;
        var w = plan.LaneWeights!.Value;
        new Span<byte>(sums, 32 * columns * count).Clear();
        for (int g = 0; g < groups; g++)
        {
            nint offset = (nint)g * periodBytes;
            bool whole = g < full;
            sbyte* x0 = x + (nint)g * groupValues;
            float* l0 = lanes + (nint)g * laneFloats;
            for (int c = 0; c < columns; c++)
            {
                byte* at = whole ? rows + c * rowBytes + offset : tails + (nint)c * periodBytes;
                Vector256<byte> s0 = TX.Plane<TI>(at, 0) ^ flip, s1 = TX.Plane<TI>(at, 1) ^ flip, s2 = TX.Plane<TI>(at, 2) ^ flip, s3 = TX.Plane<TI>(at, 3) ^ flip,
                    s4 = TX.Plane<TI>(at, 4) ^ flip, s5 = TX.Plane<TI>(at, 5) ^ flip, s6 = TX.Plane<TI>(at, 6) ^ flip, s7 = TX.Plane<TI>(at, 7) ^ flip;
                sbyte* xg = x0;
                float* lg = l0;
                int r = 0;
                if (typeof(TW) == typeof(Vectors128))
                {
                    var h = (Vector128<float>*)sums + 2 * c * count;
                    Vector128<int> topl = top.GetLower(), topu = top.GetUpper();
                    Vector128<byte> s0l = s0.GetLower(), s0u = s0.GetUpper(), s1l = s1.GetLower(), s1u = s1.GetUpper(), s2l = s2.GetLower(), s2u = s2.GetUpper(),
                        s3l = s3.GetLower(), s3u = s3.GetUpper(), s4l = s4.GetLower(), s4u = s4.GetUpper(), s5l = s5.GetLower(), s5u = s5.GetUpper(),
                        s6l = s6.GetLower(), s6u = s6.GetUpper(), s7l = s7.GetLower(), s7u = s7.GetUpper();
                    for (; r < count; r++, xg += xRow, lg += lRow)
                    {
                        h[2 * r] = WideRow128<TX, TI>(h[2 * r], s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, xg, xStep, lg, cg, topl, false);
                        h[2 * r + 1] = WideRow128<TX, TI>(h[2 * r + 1], s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, xg, xStep, lg, cg, topu, true);
                    }

                    continue;
                }

                var f = (Vector256<float>*)sums + c * count;
                for (; r + 4 <= count; r += 4, xg += 4 * xRow, lg += 4 * lRow)
                {
                    f[r] = WideRow<TX, TI>(f[r], s0, s1, s2, s3, s4, s5, s6, s7, xg, xStep, lg, cg, top);
                    f[r + 1] = WideRow<TX, TI>(f[r + 1], s0, s1, s2, s3, s4, s5, s6, s7, xg + xRow, xStep, lg + lRow, cg, top);
                    f[r + 2] = WideRow<TX, TI>(f[r + 2], s0, s1, s2, s3, s4, s5, s6, s7, xg + 2 * xRow, xStep, lg + 2 * lRow, cg, top);
                    f[r + 3] = WideRow<TX, TI>(f[r + 3], s0, s1, s2, s3, s4, s5, s6, s7, xg + 3 * xRow, xStep, lg + 3 * lRow, cg, top);
                }

                for (; r < count; r++, xg += xRow, lg += lRow)
                {
                    f[r] = WideRow<TX, TI>(f[r], s0, s1, s2, s3, s4, s5, s6, s7, xg, xStep, lg, cg, top);
                }
            }
        }

        int pairs = columns * count;
        if (typeof(TW) == typeof(Vectors128))
        {
            var h = (Vector128<float>*)sums;
            Vector128<float> wl = w.GetLower(), wu = w.GetUpper();
            for (int q = 0; q < pairs; q++)
            {
                made[q] = SumOf(h[2 * q] * wl, h[2 * q + 1] * wu);
            }

            return;
        }

        var all = (Vector256<float>*)sums;
        for (int q = 0; q < pairs; q++)
        {
            made[q] = Vector256.Sum(all[q] * w);
        }
    }

    // WideRow over one 128-bit half (`upper`: lanes 4-7, else 0-3): the half's steps dotted with x's matching half of each.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector128<float> WideRow128<TX, TI>(Vector128<float> f, Vector128<byte> s0, Vector128<byte> s1, Vector128<byte> s2, Vector128<byte> s3,
        Vector128<byte> s4, Vector128<byte> s5, Vector128<byte> s6, Vector128<byte> s7, sbyte* x, int xStep, float* lanes, int cg, Vector128<int> top, bool upper)
        where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        var i = TX.Half<TI>((sbyte*)(lanes + cg), upper).AsInt32() & top;
        i = TI.Dot(i, s0, TX.Half<TI>(x, upper));
        i = TI.Dot(i, s1, TX.Half<TI>(x + xStep, upper));
        i = TI.Dot(i, s2, TX.Half<TI>(x + 2 * xStep, upper));
        i = TI.Dot(i, s3, TX.Half<TI>(x + 3 * xStep, upper));
        i = TI.Dot(i, s4, TX.Half<TI>(x + 4 * xStep, upper));
        i = TI.Dot(i, s5, TX.Half<TI>(x + 5 * xStep, upper));
        i = TI.Dot(i, s6, TX.Half<TI>(x + 6 * xStep, upper));
        i = TI.Dot(i, s7, TX.Half<TI>(x + 7 * xStep, upper));
        return TI.Fused(Vector128.ConvertToSingle(i), TX.Half<TI>((sbyte*)lanes, upper).AsSingle(), f);
    }

    // ChunkRow, then (minimums) the group's minimums times x's (`lanes` + 16): a row's sums for a group, as the one-row
    // kernels add them.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<float> ChunkRowMin<TI>(Vector256<float> f, bool eight, bool mins, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2,
        Vector256<byte> s3, Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, Vector256<float> scale, Vector256<float> min, sbyte* x,
        float* lanes) where TI : struct, IInstructions
    {
        f = ChunkRow<TI>(f, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, x, lanes);
        return mins ? TI.Fused(min, Vector256.Load(lanes + 16), f) : f;
    }

    // The 128-bit tile's rows for one half of the lanes, a plan without minimums (h, x and lanes at that half's: the lower,
    // chunks 0-3, or the upper, 4-7, at h + 1, x + 16, lanes + 4): each row's sums of that half (h[2r]) by its steps' halves
    // (ChunkRow128), each half's sums its own as a row's two halves were (the same bits). The two halves in two passes hold
    // eight half-steps at a time: sixteen and their scales did not fit the registers (measured: 33-36 vector stack accesses
    // in q8_0's and iq4_nl's portable 128-bit tiles, the steps reloaded in the rows' loop; 5-9% faster on one core without
    // AVX2 for q4_0, q8_0, tq1_0, iq1_s, iq4_nl and mxfp4).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void HalfRows<TI>(Vector128<float>* h, int count, bool eight, bool together, Vector128<byte> s0, Vector128<byte> s1, Vector128<byte> s2, Vector128<byte> s3,
        Vector128<byte> s4, Vector128<byte> s5, Vector128<byte> s6, Vector128<byte> s7, Vector128<float> scale, sbyte* x, nint xRow, float* lanes, nint lRow)
        where TI : struct, IInstructions
    {
        int r = 0;
        for (; together && r + 2 <= count; r += 2, x += 2 * xRow, lanes += 2 * lRow)
        {
            h[2 * r] = ChunkRow128<TI>(h[2 * r], eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, x, lanes);
            h[2 * r + 2] = ChunkRow128<TI>(h[2 * r + 2], eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, x + xRow, lanes + lRow);
        }

        for (; r < count; r++, x += xRow, lanes += lRow)
        {
            h[2 * r] = ChunkRow128<TI>(h[2 * r], eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, x, lanes);
        }
    }

    // ChunkRowMin on 128-bit vectors, a plan with minimums: a row's two halves of sums (h[0]: chunks 0-3, h[1]: 4-7) by the
    // steps' halves, then the minimums' halves times x's (`lanes` + 16, + 20).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ChunkHalves<TI>(Vector128<float>* h, bool eight, Vector128<byte> s0l, Vector128<byte> s1l, Vector128<byte> s2l, Vector128<byte> s3l,
        Vector128<byte> s4l, Vector128<byte> s5l, Vector128<byte> s6l, Vector128<byte> s7l, Vector128<byte> s0u, Vector128<byte> s1u, Vector128<byte> s2u, Vector128<byte> s3u,
        Vector128<byte> s4u, Vector128<byte> s5u, Vector128<byte> s6u, Vector128<byte> s7u, Vector128<float> scalel, Vector128<float> scaleu, Vector128<float> minl,
        Vector128<float> minu, sbyte* x, float* lanes) where TI : struct, IInstructions
    {
        var fl = ChunkRow128<TI>(h[0], eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, scalel, x, lanes);
        var fu = ChunkRow128<TI>(h[1], eight, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scaleu, x + 16, lanes + 4);
        h[0] = TI.Fused(minl, Vector128.Load(lanes + 16), fl);
        h[1] = TI.Fused(minu, Vector128.Load(lanes + 20), fu);
    }

    // ChunkRow over one 128-bit half: its steps' halves dotted with x's matching 16 bytes of each 32 (from `x`), from x's
    // starts (`lanes` + 8), then times the half's scales and x's (`lanes`).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector128<float> ChunkRow128<TI>(Vector128<float> f, bool eight, Vector128<byte> s0, Vector128<byte> s1, Vector128<byte> s2, Vector128<byte> s3,
        Vector128<byte> s4, Vector128<byte> s5, Vector128<byte> s6, Vector128<byte> s7, Vector128<float> scale, sbyte* x, float* lanes) where TI : struct, IInstructions
    {
        var i = Vector128.Load((int*)(lanes + 8));
        i = DotAt<TI>(i, s0, x);
        i = DotAt<TI>(i, s1, x + 32);
        i = DotAt<TI>(i, s2, x + 64);
        i = DotAt<TI>(i, s3, x + 96);
        if (eight)
        {
            i = DotAt<TI>(i, s4, x + 128);
            i = DotAt<TI>(i, s5, x + 160);
            i = DotAt<TI>(i, s6, x + 192);
            i = DotAt<TI>(i, s7, x + 224);
        }

        return TI.Fused(Vector128.ConvertToSingle(i), scale * Vector128.Load(lanes), f);
    }

    // f + a group's sums for a row of x: its steps dotted with x's (32 bytes a step from `x`), each lane's sum from x's start
    // (`lanes` + 8), then times the lanes' scales and x's (`lanes`).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<float> ChunkRow<TI>(Vector256<float> f, bool eight, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, Vector256<float> scale, sbyte* x, float* lanes) where TI : struct, IInstructions
    {
        var i = Vector256.Load((int*)(lanes + 8));
        i = TI.Dot(i, s0, Vector256.Load(x));
        i = TI.Dot(i, s1, Vector256.Load(x + 32));
        i = TI.Dot(i, s2, Vector256.Load(x + 64));
        i = TI.Dot(i, s3, Vector256.Load(x + 96));
        if (eight)
        {
            i = TI.Dot(i, s4, Vector256.Load(x + 128));
            i = TI.Dot(i, s5, Vector256.Load(x + 160));
            i = TI.Dot(i, s6, Vector256.Load(x + 192));
            i = TI.Dot(i, s7, Vector256.Load(x + 224));
        }

        return TI.Fused(Vector256.ConvertToSingle(i), scale * Vector256.Load(lanes), f);
    }

    // The described integer lane for one row of x read alone where the plan's one-row kernel is not resolved
    // (DescribedPlan.OneKernel: byte planes of wide integers, P > 1; every plan of one plane has its own kernel, ResolveOne).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedIntegerOne<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TI : struct, IInstructions => plan.P switch
        {
            2 => WideMadeOne<FourChunks, TI>(plan, row, tail, full, groups, x, lanes),
            4 => WideMadeOne<TwoChunks, TI>(plan, row, tail, full, groups, x, lanes),
            _ => WideMadeOne<OneChunk, TI>(plan, row, tail, full, groups, x, lanes),
        };

    /// <summary>
    /// How the integer lane's group loop reads a group's scales and minimums: the plan's loop over its factors
    /// (FactorLoop), or the plan's factors as type parameters (DescribedPlan.FactorKinds: SlotFactors), each group then
    /// straight-line code. The same numbers either way.
    /// </summary>
    private unsafe interface IFactors
    {
        static abstract void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions;
    }

    // The loop over the factors; TS: whether a factor is read a lane at a time (DescribedPlan.ReadsScalar), so a plan
    // without one has no code for it in its kernels.
    private readonly unsafe struct FactorLoop<TS> : IFactors where TS : struct, IFlag
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions =>
            plan.Factor<TI, TS>(period, gp, valid, out scale, out min);
    }

    // A plan's one scale alone (DescribedPlan.OneScale: OneScaleFactor), no loop over factors in its kernels.
    private readonly unsafe struct OneScaleRead : IFactors
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions =>
            plan.OneScaleFactor<TI>(period, gp, valid, out scale, out min);
    }

    private readonly unsafe struct SlotFactors<T0, T1, T2, T3> : IFactors
        where T0 : struct, DescribedPlan.IFactorSlot where T1 : struct, DescribedPlan.IFactorSlot where T2 : struct, DescribedPlan.IFactorSlot where T3 : struct, DescribedPlan.IFactorSlot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions =>
            plan.Slots<T0, T1, T2, T3, TI>(period, gp, valid, out scale, out min);
    }

    /// <summary>A kernel's address with its factors as the type parameter (WithFactors' last step).</summary>
    private interface IFactorsUse
    {
        static abstract nint Use<TF>() where TF : struct, IFactors;
    }

    // The plan's factors as type parameters (DescribedPlan.FactorKinds), a slot at a time: its field's reading, then its
    // number's, the last handing the set to TU; the loop over the factors where the plan has no kinds. Made once a plan and
    // kernel (DescribedPlan.OneKernel, TileKernel), never a call.
    private static nint WithFactors<TU>(DescribedPlan plan) where TU : struct, IFactorsUse =>
        plan.FactorKinds != 0 ? FactorsFirst<TU>(plan.FactorKinds) : plan.OneScale != 0 ? TU.Use<OneScaleRead>()
            : plan.ReadsScalar ? TU.Use<FactorLoop<Yes>>() : TU.Use<FactorLoop<No>>();

    private static nint FactorsFirst<TU>(int k) where TU : struct, IFactorsUse => (k >> 4 & 15) switch
    {
        1 => FirstNumber<TU, DescribedPlan.UniformBits>(k),
        2 => FirstNumber<TU, DescribedPlan.GatherBits>(k),
        3 => FirstNumber<TU, DescribedPlan.PieceBits>(k),
        4 => FirstNumber<TU, DescribedPlan.WindowBits>(k),
        _ => FirstNumber<TU, DescribedPlan.WindowPieces>(k),
    };

    private static nint FirstNumber<TU, TB>(int k) where TU : struct, IFactorsUse where TB : struct, DescribedPlan.IFactorBits => (k & 15) switch
    {
        0 => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.UnsignedNumber>>(k >> 8),
        1 => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.SignedNumber>>(k >> 8),
        2 => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.TableNumber>>(k >> 8),
        6 => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.SmallTableNumber>>(k >> 8),
        3 => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.HalfNumber>>(k >> 8),
        _ => FactorsSecond<TU, DescribedPlan.Field<TB, DescribedPlan.SingleNumber>>(k >> 8),
    };

    private static nint FactorsSecond<TU, T0>(int k) where TU : struct, IFactorsUse where T0 : struct, DescribedPlan.IFactorSlot => (k >> 4 & 15) switch
    {
        0 => FactorsThird<TU, T0, DescribedPlan.NoFactor>(k >> 8),
        1 => SecondNumber<TU, T0, DescribedPlan.UniformBits>(k),
        2 => SecondNumber<TU, T0, DescribedPlan.GatherBits>(k),
        3 => SecondNumber<TU, T0, DescribedPlan.PieceBits>(k),
        4 => SecondNumber<TU, T0, DescribedPlan.WindowBits>(k),
        _ => SecondNumber<TU, T0, DescribedPlan.WindowPieces>(k),
    };

    private static nint SecondNumber<TU, T0, TB>(int k) where TU : struct, IFactorsUse where T0 : struct, DescribedPlan.IFactorSlot where TB : struct, DescribedPlan.IFactorBits =>
        (k & 15) switch
        {
            0 => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.UnsignedNumber>>(k >> 8),
            1 => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.SignedNumber>>(k >> 8),
            2 => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.TableNumber>>(k >> 8),
        6 => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.SmallTableNumber>>(k >> 8),
            3 => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.HalfNumber>>(k >> 8),
            _ => FactorsThird<TU, T0, DescribedPlan.Field<TB, DescribedPlan.SingleNumber>>(k >> 8),
        };

    private static nint FactorsThird<TU, T0, T1>(int k) where TU : struct, IFactorsUse where T0 : struct, DescribedPlan.IFactorSlot where T1 : struct, DescribedPlan.IFactorSlot =>
        (k >> 4 & 15) switch
        {
            0 => FactorsFourth<TU, T0, T1, DescribedPlan.NoFactor>(k >> 8),
            1 => ThirdNumber<TU, T0, T1, DescribedPlan.UniformBits>(k),
            2 => ThirdNumber<TU, T0, T1, DescribedPlan.GatherBits>(k),
            3 => ThirdNumber<TU, T0, T1, DescribedPlan.PieceBits>(k),
            4 => ThirdNumber<TU, T0, T1, DescribedPlan.WindowBits>(k),
            _ => ThirdNumber<TU, T0, T1, DescribedPlan.WindowPieces>(k),
        };

    private static nint ThirdNumber<TU, T0, T1, TB>(int k)
        where TU : struct, IFactorsUse where T0 : struct, DescribedPlan.IFactorSlot where T1 : struct, DescribedPlan.IFactorSlot where TB : struct, DescribedPlan.IFactorBits =>
        (k & 15) switch
        {
            0 => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.UnsignedNumber>>(k >> 8),
            1 => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.SignedNumber>>(k >> 8),
            2 => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.TableNumber>>(k >> 8),
        6 => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.SmallTableNumber>>(k >> 8),
            3 => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.HalfNumber>>(k >> 8),
            _ => FactorsFourth<TU, T0, T1, DescribedPlan.Field<TB, DescribedPlan.SingleNumber>>(k >> 8),
        };

    private static nint FactorsFourth<TU, T0, T1, T2>(int k)
        where TU : struct, IFactorsUse where T0 : struct, DescribedPlan.IFactorSlot where T1 : struct, DescribedPlan.IFactorSlot where T2 : struct, DescribedPlan.IFactorSlot =>
        (k >> 4 & 15) switch
        {
            0 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.NoFactor>>(),
            1 => FourthNumber<TU, T0, T1, T2, DescribedPlan.UniformBits>(k),
            2 => FourthNumber<TU, T0, T1, T2, DescribedPlan.GatherBits>(k),
            3 => FourthNumber<TU, T0, T1, T2, DescribedPlan.PieceBits>(k),
            4 => FourthNumber<TU, T0, T1, T2, DescribedPlan.WindowBits>(k),
            _ => FourthNumber<TU, T0, T1, T2, DescribedPlan.WindowPieces>(k),
        };

    private static nint FourthNumber<TU, T0, T1, T2, TB>(int k) where TU : struct, IFactorsUse
        where T0 : struct, DescribedPlan.IFactorSlot where T1 : struct, DescribedPlan.IFactorSlot where T2 : struct, DescribedPlan.IFactorSlot where TB : struct, DescribedPlan.IFactorBits =>
        (k & 15) switch
        {
            0 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.UnsignedNumber>>>(),
            1 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.SignedNumber>>>(),
            2 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.TableNumber>>>(),
            6 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.SmallTableNumber>>>(),
            3 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.HalfNumber>>>(),
            _ => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.SingleNumber>>>(),
        };

    // The kernels WithFactors hands a plan's factors to: the one-row kernels, the tile kernels (TW: their vectors; TE, TM: their flags).
    private readonly unsafe struct UniformOneUse<TG, TI, TE, TM> : IFactorsUse
        where TG : struct, IGroupSteps<TG> where TI : struct, IInstructions where TE : struct, IFlag where TM : struct, IFlag
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &UniformOne<TG, TI, TF, TE, TM>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct GatheredOneUse<TI, TS, TE, TM> : IFactorsUse
        where TI : struct, IInstructions where TS : struct, DescribedPlan.IGatheredSteps where TE : struct, IFlag where TM : struct, IFlag
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &GatheredOne<TI, TF, TS, TE, TM>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct MadeOneUse<TI, TS> : IFactorsUse where TI : struct, IInstructions where TS : struct, DescribedPlan.ISpanCodes
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &MadeOne<EightChunks, TI, TF, TS>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct IntegerTileUse<TI, TG, TW, TE, TM> : IFactorsUse
        where TI : struct, IInstructions where TG : struct, IGroupSteps<TG> where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, nint, int, byte*, int, int, int, sbyte*, float*, nint, nint, int, byte*, float*, void> kernel = TI.RowsTogether
                ? &IntegerTile<TI, TG, TF, TW, TE, TM, Yes> : &IntegerTile<TI, TG, TF, TW, TE, TM, No>;
            return (nint)kernel;
        }
    }

    /// <summary>A spans-made kernel's address with its plan's way of making a span as the type parameter (WithSpans), its factors then chosen.</summary>
    private interface ISpansUse
    {
        static abstract nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.ISpanCodes;
    }

    private readonly struct MadeOneSpans<TI> : ISpansUse where TI : struct, IInstructions
    {
        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.ISpanCodes => WithFactors<MadeOneUse<TI, TS>>(plan);
    }

    private readonly struct MadeTileSpans<TI, TW, TE, TM> : ISpansUse
        where TI : struct, IInstructions where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag
    {
        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.ISpanCodes => WithFactors<IntegerTileUse<TI, MadeGroup<TS>, TW, TE, TM>>(plan);
    }

    // The spans-made kernels' way of making a span as a type parameter (DescribedPlan.ISpanCodes): a codebook's by its fast
    // reading or by any, else the span's bits through the plan's map; so each kernel holds its plan's way alone and stays
    // within what the JIT inlines. Made once a plan and kernel, never a call.
    private static nint WithSpans<TU>(DescribedPlan plan) where TU : struct, ISpansUse => plan.Map switch
    {
        CodeMap.Codebook => plan.CodebookIsFast ? TU.Use<DescribedPlan.FastBookSpans>(plan) : TU.Use<DescribedPlan.AnyBookSpans>(plan),
        CodeMap.Flip => TU.Use<DescribedPlan.RawSpans<DescribedPlan.FlipStepMap>>(plan),
        CodeMap.Shuffle => TU.Use<DescribedPlan.RawSpans<DescribedPlan.ShuffleStepMap>>(plan),
        CodeMap.Table => TU.Use<DescribedPlan.RawSpans<DescribedPlan.TableStepMap>>(plan),
        _ => TU.Use<DescribedPlan.RawSpans<DescribedPlan.SameStepMap>>(plan),
    };

    /// <summary>A gathered kernel's address with its steps' readings as the type parameter (WithSteps' last step), its factors then chosen.</summary>
    private interface IStepsUse
    {
        /// <summary>The kernel's instruction set's IInstructions.ShiftsByLane (read here, once a plan, not in the kernel).</summary>
        static abstract bool ShiftsByLane { get; }

        static abstract nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps;
    }

    // (Its steps a group, T = 8, and its minimums as flags, as the tile takes them: no code for the other in the kernel.)
    private readonly struct GatheredOneSteps<TI> : IStepsUse where TI : struct, IInstructions
    {
        public static bool ShiftsByLane => TI.ShiftsByLane;

        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps => plan.T == 8
            ? plan.Mins ? WithFactors<GatheredOneUse<TI, TS, Yes, Yes>>(plan) : WithFactors<GatheredOneUse<TI, TS, Yes, No>>(plan)
            : plan.Mins ? WithFactors<GatheredOneUse<TI, TS, No, Yes>>(plan) : WithFactors<GatheredOneUse<TI, TS, No, No>>(plan);
    }

    // The natural plan's one-row kernel (8 steps a group, each a chunk; its minimums as a flag).
    private readonly struct NaturalOneSteps<TI> : IStepsUse where TI : struct, IInstructions
    {
        public static bool ShiftsByLane => TI.ShiftsByLane;

        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps => plan.PairSteps ? Groups<TS, Yes>(plan) : Groups<TS, No>(plan);

        private static nint Groups<TS, TP>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps where TP : struct, IFlag =>
            plan.GroupsPerPeriod == 1
                ? plan.Mins ? WithFactors<NaturalOneUse<TI, TS, Yes, Yes, TP>>(plan) : WithFactors<NaturalOneUse<TI, TS, No, Yes, TP>>(plan)
                : plan.Mins ? WithFactors<NaturalOneUse<TI, TS, Yes, No, TP>>(plan) : WithFactors<NaturalOneUse<TI, TS, No, No, TP>>(plan);
    }

    private readonly unsafe struct NaturalOneUse<TI, TS, TM, TG, TP> : IFactorsUse
        where TI : struct, IInstructions where TS : struct, DescribedPlan.IGatheredSteps where TM : struct, IFlag where TG : struct, IFlag where TP : struct, IFlag
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &NaturalOne<TI, TF, TS, TM, TG, TP>;
            return (nint)kernel;
        }
    }

    private readonly struct GatheredTileSteps<TI, TW, TE, TM> : IStepsUse
        where TI : struct, IInstructions where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag
    {
        public static bool ShiftsByLane => TI.ShiftsByLane;

        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps => WithFactors<IntegerTileUse<TI, GatheredGroup<TS>, TW, TE, TM>>(plan);
    }

    // The gathered steps' readings as type parameters (DescribedPlan.StepKinds): plane 0's, plane 1's (or none), the map;
    // GatheredStep's loop where the plan has no kinds. Made once a plan and kernel, never a call.
    private static nint WithSteps<TU>(DescribedPlan plan) where TU : struct, IStepsUse
    {
        int k = plan.StepKinds;
        return k == 0 ? TU.Use<DescribedPlan.PlanSteps>(plan) : (k & 15) switch
        {
            1 => SecondPlane<TU, DescribedPlan.MaskedPlane>(plan, k),
            3 => SecondPlane<TU, DescribedPlan.BitPlane>(plan, k),
            4 => SecondPlane<TU, DescribedPlan.StreamPlane>(plan, k),
            _ => SecondPlane<TU, DescribedPlan.DigitPlane>(plan, k),
        };
    }

    private static nint SecondPlane<TU, TP0>(DescribedPlan plan, int k) where TU : struct, IStepsUse where TP0 : struct, DescribedPlan.IStepPlane => (k >> 4 & 15) switch
    {
        0 => StepMapOf<TU, TP0, DescribedPlan.NoPlane, DescribedPlan.NoPlane>(plan, k),
        1 => ThirdPlane<TU, TP0, DescribedPlan.MaskedPlane>(plan, k),
        3 => ThirdPlane<TU, TP0, DescribedPlan.BitPlane>(plan, k),
        4 => ThirdPlane<TU, TP0, DescribedPlan.StreamPlane>(plan, k),
        _ => ThirdPlane<TU, TP0, DescribedPlan.DigitPlane>(plan, k),
    };

    private static nint ThirdPlane<TU, TP0, TP1>(DescribedPlan plan, int k)
        where TU : struct, IStepsUse where TP0 : struct, DescribedPlan.IStepPlane where TP1 : struct, DescribedPlan.IStepPlane => (k >> 8 & 15) switch
        {
            0 => StepMapOf<TU, TP0, TP1, DescribedPlan.NoPlane>(plan, k),
            1 => StepMapOf<TU, TP0, TP1, DescribedPlan.MaskedPlane>(plan, k),
            3 => StepMapOf<TU, TP0, TP1, DescribedPlan.BitPlane>(plan, k),
            4 => StepMapOf<TU, TP0, TP1, DescribedPlan.StreamPlane>(plan, k),
            _ => StepMapOf<TU, TP0, TP1, DescribedPlan.DigitPlane>(plan, k),
        };

    private static nint StepMapOf<TU, TP0, TP1, TP2>(DescribedPlan plan, int k)
        where TU : struct, IStepsUse where TP0 : struct, DescribedPlan.IStepPlane where TP1 : struct, DescribedPlan.IStepPlane where TP2 : struct, DescribedPlan.IStepPlane =>
        (CodeMap)((k >> 12) - 1) switch
        {
            CodeMap.Same => StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.SameStepMap>(plan),
            CodeMap.Flip => StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.FlipStepMap>(plan),
            CodeMap.Shuffle => StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.ShuffleStepMap>(plan),
            CodeMap.Table => StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.TableStepMap>(plan),
            _ => plan.VectorCount > 0
                ? plan.PairedEntries ? StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.BookStepMap<Yes, Yes>>(plan) : StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.BookStepMap<Yes, No>>(plan)
                : plan.PairedEntries ? StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.BookStepMap<No, Yes>>(plan) : StepShiftsOf<TU, TP0, TP1, TP2, DescribedPlan.BookStepMap<No, No>>(plan),
        };

    // Whether the gathered steps shift by one number a plane and step (DescribedPlan.StepShiftsUniform, where the kernel's
    // instruction set does not shift lanes by their own counts in one instruction), as the steps' last type parameter.
    private static nint StepShiftsOf<TU, TP0, TP1, TP2, TM>(DescribedPlan plan)
        where TU : struct, IStepsUse where TP0 : struct, DescribedPlan.IStepPlane where TP1 : struct, DescribedPlan.IStepPlane where TP2 : struct, DescribedPlan.IStepPlane
        where TM : struct, DescribedPlan.IStepMap => plan.StepShiftsUniform && !TU.ShiftsByLane
        ? StepLoadsOf<TU, TP0, TP1, TP2, TM, Yes>(plan) : StepLoadsOf<TU, TP0, TP1, TP2, TM, No>(plan);

    // Whether the steps' words are loaded from the period (a natural plan's: DescribedPlan.IsNatural), as their last.
    private static nint StepLoadsOf<TU, TP0, TP1, TP2, TM, TH>(DescribedPlan plan)
        where TU : struct, IStepsUse where TP0 : struct, DescribedPlan.IStepPlane where TP1 : struct, DescribedPlan.IStepPlane where TP2 : struct, DescribedPlan.IStepPlane
        where TM : struct, DescribedPlan.IStepMap where TH : struct, IFlag => plan.IsNatural
        ? TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, TM, TH, Yes>>(plan) : TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, TM, TH, No>>(plan);

    // The one-row integer kernel of a plan by TI, its choices as type parameters (DescribedIntegerOne's, and the factors'):
    // made once a plan and instruction set (DescribedPlan.OneKernel); 0 for byte planes of wide integers (their dispatch).
    private static nint ResolveOne<TI>(DescribedPlan plan) where TI : struct, IInstructions
    {
        if (plan.P != 1)
        {
            return 0;
        }

        if (plan.UniformRuns != 0)
        {
            return plan.UniformRuns switch
            {
                1 => OneByMap<OneRun, TI>(plan),
                2 => OneByMap<TwoRuns, TI>(plan),
                4 => OneByMap<FourRuns, TI>(plan),
                _ => OneByMap<EightRuns, TI>(plan),
            };
        }

        return plan.StepsGathered ? plan.IsNatural ? WithSteps<NaturalOneSteps<TI>>(plan) : WithSteps<GatheredOneSteps<TI>>(plan) : WithSpans<MadeOneSpans<TI>>(plan);
    }

    // The deposit lane's one-row kernel of a natural plan (DescribedPlan.DepositSteps) by TI, its planes' ways as type
    // parameters: NaturalOne over DepositSteps, its factors chosen as the plan's own.
    private static nint ResolveDeposit<TI>(DescribedPlan plan) where TI : struct, IInstructions
    {
        int k = plan.DepositKinds;
        return (k & 3) switch
        {
            1 => DepositSecond<TI, DescribedPlan.SpreadDeposit>(plan, k),
            2 => DepositSecond<TI, DescribedPlan.SharedDeposit>(plan, k),
            _ => DepositSecond<TI, DescribedPlan.RunDeposit>(plan, k),
        };
    }

    private static nint DepositSecond<TI, TK0>(DescribedPlan plan, int k) where TI : struct, IInstructions where TK0 : struct, DescribedPlan.IDepositPlane => (k >> 2 & 3) switch
    {
        0 => NaturalOneSteps<TI>.Use<DescribedPlan.DepositedSteps<TK0, DescribedPlan.NoDeposit, DescribedPlan.NoDeposit>>(plan),
        1 => DepositThird<TI, TK0, DescribedPlan.SpreadDeposit>(plan, k),
        2 => DepositThird<TI, TK0, DescribedPlan.SharedDeposit>(plan, k),
        _ => DepositThird<TI, TK0, DescribedPlan.RunDeposit>(plan, k),
    };

    private static nint DepositThird<TI, TK0, TK1>(DescribedPlan plan, int k)
        where TI : struct, IInstructions where TK0 : struct, DescribedPlan.IDepositPlane where TK1 : struct, DescribedPlan.IDepositPlane => (k >> 4 & 3) switch
        {
            0 => NaturalOneSteps<TI>.Use<DescribedPlan.DepositedSteps<TK0, TK1, DescribedPlan.NoDeposit>>(plan),
            1 => NaturalOneSteps<TI>.Use<DescribedPlan.DepositedSteps<TK0, TK1, DescribedPlan.SpreadDeposit>>(plan),
            2 => NaturalOneSteps<TI>.Use<DescribedPlan.DepositedSteps<TK0, TK1, DescribedPlan.SharedDeposit>>(plan),
            _ => NaturalOneSteps<TI>.Use<DescribedPlan.DepositedSteps<TK0, TK1, DescribedPlan.RunDeposit>>(plan),
        };

    private static nint OneByMap<TR, TI>(DescribedPlan plan) where TR : struct, IRunLoad where TI : struct, IInstructions => plan.Map switch
    {
        CodeMap.Same => OneBySteps<TR, SameCodes, TI>(plan),
        CodeMap.Flip => OneBySteps<TR, FlippedCodes, TI>(plan),
        _ => OneBySteps<TR, ShuffledCodes, TI>(plan),
    };

    // UniformOne's steps gathered or loaded (DescribedPlan.Gathered: chunks of 32), its steps a group and its minimums as
    // flags: a kernel holds its plan's one way of making them.
    private static nint OneBySteps<TR, TM, TI>(DescribedPlan plan) where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions =>
        plan.Gathered ? OneByMins<GatheredRunsGroup<TR, TM>, TI, Yes>(plan)
        : plan.T == 8 ? OneByMins<LoadedGroup<TR, TM, Yes>, TI, Yes>(plan) : OneByMins<LoadedGroup<TR, TM, No>, TI, No>(plan);

    private static nint OneByMins<TG, TI, TE>(DescribedPlan plan) where TG : struct, IGroupSteps<TG> where TI : struct, IInstructions where TE : struct, IFlag =>
        plan.Mins ? WithFactors<UniformOneUse<TG, TI, TE, Yes>>(plan) : WithFactors<UniformOneUse<TG, TI, TE, No>>(plan);

    // The tile kernel of a plan by TI and TW (its vectors): its steps' maker and factors as type parameters as ResolveOne
    // takes them, its steps a group and minimums as flags, a wide integer's by its plane count; made once a plan,
    // instruction set and width (DescribedPlan.TileKernel).
    private static unsafe nint ResolveTile<TI, TW>(DescribedPlan plan) where TI : struct, IInstructions where TW : struct, IVectorWidth
    {
        if (plan.P != 1)
        {
            delegate*<DescribedPlan, byte*, nint, int, byte*, int, int, int, sbyte*, float*, nint, nint, int, byte*, float*, void> wide;
            if (plan.P == 2)
            {
                wide = &WideTile<FourChunks, TI, TW>;
            }
            else if (plan.P == 4)
            {
                wide = &WideTile<TwoChunks, TI, TW>;
            }
            else
            {
                wide = &WideTile<OneChunk, TI, TW>;
            }

            return (nint)wide;
        }

        if (plan.T == 8)
        {
            return plan.Mins ? ChunkTile<TI, TW, Yes, Yes>(plan) : ChunkTile<TI, TW, Yes, No>(plan);
        }

        return plan.Mins ? ChunkTile<TI, TW, No, Yes>(plan) : ChunkTile<TI, TW, No, No>(plan);
    }

    private static nint ChunkTile<TI, TW, TE, TM>(DescribedPlan plan)
        where TI : struct, IInstructions where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag
    {
        if (plan.UniformRuns != 0)
        {
            return plan.UniformRuns switch
            {
                1 => TileByMap<OneRun, TI, TW, TE, TM>(plan),
                2 => TileByMap<TwoRuns, TI, TW, TE, TM>(plan),
                4 => TileByMap<FourRuns, TI, TW, TE, TM>(plan),
                _ => TileByMap<EightRuns, TI, TW, TE, TM>(plan),
            };
        }

        return plan.StepsGathered ? WithSteps<GatheredTileSteps<TI, TW, TE, TM>>(plan) : WithSpans<MadeTileSpans<TI, TW, TE, TM>>(plan);
    }

    private static nint TileByMap<TR, TI, TW, TE, TM>(DescribedPlan plan)
        where TR : struct, IRunLoad where TI : struct, IInstructions where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag => plan.Map switch
        {
            CodeMap.Same => TileBySteps<TR, SameCodes, TI, TW, TE, TM>(plan),
            CodeMap.Flip => TileBySteps<TR, FlippedCodes, TI, TW, TE, TM>(plan),
            _ => TileBySteps<TR, ShuffledCodes, TI, TW, TE, TM>(plan),
        };

    // The tile's steps gathered or loaded, as OneBySteps chooses them.
    private static nint TileBySteps<TR, TC, TI, TW, TE, TM>(DescribedPlan plan)
        where TR : struct, IRunLoad where TC : struct, ICodeMap where TI : struct, IInstructions where TW : struct, IVectorWidth where TE : struct, IFlag where TM : struct, IFlag =>
        plan.Gathered ? WithFactors<IntegerTileUse<TI, GatheredRunsGroup<TR, TC>, TW, TE, TM>>(plan) : WithFactors<IntegerTileUse<TI, LoadedGroup<TR, TC, TE>, TW, TE, TM>>(plan);

    /// <summary>
    /// How the described lane loads a span's run of its one code plane and shifts it (every span the same:
    /// DescribedPlan.UniformRuns): span s by each lane's shift (shifts[s]) where the instruction set shifts lanes by their
    /// own counts in one instruction (IInstructions.ShiftsByLane), else by the shifts as numbers (RunCounts' 8 of span s).
    /// The arrays and s are passed, not the span's entries, so the branch not taken reads nothing (the taken one's shift
    /// read in place, within the shift instruction).
    /// </summary>
    private unsafe interface IRunLoad
    {
        /// <summary>The runs a span's 32 bytes are in (DescribedPlan.UniformRuns).</summary>
        static abstract int Count { get; }

        static abstract Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions;
    }

    // One run: 32 bytes at one shift.
    private unsafe struct OneRun : IRunLoad
    {
        public static int Count => 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions => TI.ShiftsByLane
            ? TI.ShiftRight(Vector256.Load(at).AsUInt64(), shifts[s]).AsByte()
            : Vector256.ShiftRightLogical(Vector256.Load(at).AsUInt64(), counts[8 * s]).AsByte();
    }

    // Two runs: 16 bytes at two shifts (a 128-bit half each).
    private unsafe struct TwoRuns : IRunLoad
    {
        public static int Count => 2;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions
        {
            if (TI.ShiftsByLane)
            {
                return TI.ShiftRight(TI.Halves(at).AsUInt64(), shifts[s]).AsByte();
            }

            var v = Vector128.Load(at).AsUInt64();
            return Vector256.Create(Vector128.ShiftRightLogical(v, counts[8 * s]), Vector128.ShiftRightLogical(v, counts[8 * s + 2])).AsByte();
        }
    }

    // Four runs: 8 bytes at four shifts (a 64-bit lane each).
    private unsafe struct FourRuns : IRunLoad
    {
        public static int Count => 4;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions
        {
            ulong v = *(ulong*)at;
            if (TI.ShiftsByLane)
            {
                return TI.ShiftRight(Vector256.Create(v), shifts[s]).AsByte();
            }

            int* c = counts + 8 * s;
            return Vector256.Create(v >> c[0], v >> c[1], v >> c[2], v >> c[3]).AsByte();
        }
    }

    // Eight runs: 4 bytes at eight shifts (a 32-bit lane each).
    private unsafe struct EightRuns : IRunLoad
    {
        public static int Count => 8;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions
        {
            uint v = *(uint*)at;
            if (TI.ShiftsByLane)
            {
                return TI.ShiftRight(Vector256.Create(v), shifts[s].AsUInt32()).AsByte();
            }

            int* c = counts + 8 * s;
            return Vector256.Create(v >> c[0], v >> c[1], v >> c[2], v >> c[3], v >> c[4], v >> c[5], v >> c[6], v >> c[7]).AsByte();
        }
    }

    /// <summary>How the integer lane makes a code its byte (DescribedPlan.Map): as it is, flipped, or by a 16-entry byte shuffle.</summary>
    private interface ICodeMap
    {
        static abstract Vector256<byte> Map<TI>(Vector256<byte> code, Vector256<byte> table) where TI : struct, IInstructions;
    }

    private struct SameCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map<TI>(Vector256<byte> code, Vector256<byte> table) where TI : struct, IInstructions => code;
    }

    private struct FlippedCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map<TI>(Vector256<byte> code, Vector256<byte> table) where TI : struct, IInstructions => code ^ table;
    }

    private struct ShuffledCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map<TI>(Vector256<byte> code, Vector256<byte> table) where TI : struct, IInstructions => TI.ShuffleBytes(table, code);
    }

    // Span s's codes (of the period at `at`) as the integer lane's bytes, by TR and TM.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<byte> SpanCodes<TR, TM, TI>(byte* at, int* offsets, Vector256<ulong>* shifts, int* counts, int s, Vector256<byte> mask, Vector256<byte> table)
        where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions =>
        TM.Map<TI>(TR.Load<TI>(at + offsets[s], shifts, counts, s) & mask, table);

    // Group gp of the period at `at` as the integer lane's steps (DescribedPlan.Steps, a plan with uniform runs): 8 spans
    // transposed (chunks of 32), or 4 (chunks of 16: s4..s7 not written).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void UniformSteps<TR, TM, TI>(byte* at, int gp, int* offsets, Vector256<ulong>* shifts, int* counts, Vector256<byte> mask, Vector256<byte> table, bool c32,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3, out Vector256<byte> s4, out Vector256<byte> s5,
        out Vector256<byte> s6, out Vector256<byte> s7) where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions
    {
        if (c32)
        {
            int s = 8 * gp;
            Transpose8<TI>(SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 1, mask, table),
                SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 2, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 3, mask, table),
                SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 4, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 5, mask, table),
                SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 6, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, s + 7, mask, table),
                out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            return;
        }

        int r = 4 * gp;
        Transpose4<TI>(SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, r, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, r + 1, mask, table),
            SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, r + 2, mask, table), SpanCodes<TR, TM, TI>(at, offsets, shifts, counts, r + 3, mask, table), out s0, out s1, out s2, out s3);
        s4 = s5 = s6 = s7 = default;
    }

    // Group gp's steps gathered (DescribedPlan.Gathered): dword d of the 8 chunks' runs (each its own first byte, a lane
    // each) in one gather, each run q's shifts of it masked and mapped: step q · D + d, D = 8 / runs (the positions 4s..4s + 3
    // of each chunk in its span's order, as the transposed loads give them).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void GatheredSteps<TM, TI>(byte* at, int* offsets, Vector256<uint>* shifts, int* shift, int runs, Vector256<uint> flip, Vector256<byte> mask, Vector256<byte> table,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3, out Vector256<byte> s4, out Vector256<byte> s5,
        out Vector256<byte> s6, out Vector256<byte> s7) where TM : struct, ICodeMap where TI : struct, IInstructions
    {
        switch (runs)
        {
            case 8:
                var g = TI.Gather(at, offsets).AsUInt32() ^ flip;
                s0 = GatheredStep<TM, TI>(g, shifts, shift, 0, mask, table);
                s1 = GatheredStep<TM, TI>(g, shifts, shift, 1, mask, table);
                s2 = GatheredStep<TM, TI>(g, shifts, shift, 2, mask, table);
                s3 = GatheredStep<TM, TI>(g, shifts, shift, 3, mask, table);
                s4 = GatheredStep<TM, TI>(g, shifts, shift, 4, mask, table);
                s5 = GatheredStep<TM, TI>(g, shifts, shift, 5, mask, table);
                s6 = GatheredStep<TM, TI>(g, shifts, shift, 6, mask, table);
                s7 = GatheredStep<TM, TI>(g, shifts, shift, 7, mask, table);
                return;
            case 4:
                var a0 = TI.Gather(at, offsets).AsUInt32() ^ flip;
                var a1 = TI.Gather(at + 4, offsets).AsUInt32() ^ flip;
                s0 = GatheredStep<TM, TI>(a0, shifts, shift, 0, mask, table);
                s1 = GatheredStep<TM, TI>(a1, shifts, shift, 0, mask, table);
                s2 = GatheredStep<TM, TI>(a0, shifts, shift, 1, mask, table);
                s3 = GatheredStep<TM, TI>(a1, shifts, shift, 1, mask, table);
                s4 = GatheredStep<TM, TI>(a0, shifts, shift, 2, mask, table);
                s5 = GatheredStep<TM, TI>(a1, shifts, shift, 2, mask, table);
                s6 = GatheredStep<TM, TI>(a0, shifts, shift, 3, mask, table);
                s7 = GatheredStep<TM, TI>(a1, shifts, shift, 3, mask, table);
                return;
            default:
                // Two runs of 16 bytes: the 8 chunks' runs loaded (chunk l and l + 4 a vector) and their dwords transposed by
                // unpacks (8 loads and 8 unpacks: fewer than 4 gathers), dword d of every chunk in b_d.
                var r0 = Vector256.Create(Vector128.Load(at + offsets[0]), Vector128.Load(at + offsets[4])).AsUInt32();
                var r1 = Vector256.Create(Vector128.Load(at + offsets[1]), Vector128.Load(at + offsets[5])).AsUInt32();
                var r2 = Vector256.Create(Vector128.Load(at + offsets[2]), Vector128.Load(at + offsets[6])).AsUInt32();
                var r3 = Vector256.Create(Vector128.Load(at + offsets[3]), Vector128.Load(at + offsets[7])).AsUInt32();
                var w0 = TI.UnpackLow(r0, r1).AsUInt64();
                var w1 = TI.UnpackHigh(r0, r1).AsUInt64();
                var w2 = TI.UnpackLow(r2, r3).AsUInt64();
                var w3 = TI.UnpackHigh(r2, r3).AsUInt64();
                var b0 = TI.UnpackLow(w0, w2).AsUInt32() ^ flip;
                var b1 = TI.UnpackHigh(w0, w2).AsUInt32() ^ flip;
                var b2 = TI.UnpackLow(w1, w3).AsUInt32() ^ flip;
                var b3 = TI.UnpackHigh(w1, w3).AsUInt32() ^ flip;
                s0 = GatheredStep<TM, TI>(b0, shifts, shift, 0, mask, table);
                s1 = GatheredStep<TM, TI>(b1, shifts, shift, 0, mask, table);
                s2 = GatheredStep<TM, TI>(b2, shifts, shift, 0, mask, table);
                s3 = GatheredStep<TM, TI>(b3, shifts, shift, 0, mask, table);
                s4 = GatheredStep<TM, TI>(b0, shifts, shift, 1, mask, table);
                s5 = GatheredStep<TM, TI>(b1, shifts, shift, 1, mask, table);
                s6 = GatheredStep<TM, TI>(b2, shifts, shift, 1, mask, table);
                s7 = GatheredStep<TM, TI>(b3, shifts, shift, 1, mask, table);
                return;
        }
    }

    // A gathered dword vector at run q's shifts (shifts[q]; by one[q] where every lane's is the same and the instruction set
    // does not shift lanes by their own counts in one instruction), masked and mapped: a step. The arrays and q are passed,
    // not the run's entries, so the branch not taken reads nothing.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<byte> GatheredStep<TM, TI>(Vector256<uint> words, Vector256<uint>* shifts, int* one, int q, Vector256<byte> mask, Vector256<byte> table)
        where TM : struct, ICodeMap where TI : struct, IInstructions =>
        TM.Map<TI>((TI.ShiftsByLane || one[q] < 0 ? TI.ShiftRight(words, shifts[q]) : Vector256.ShiftRightLogical(words, one[q])).AsByte() & mask, table);

    // MadeOne for a plan whose steps are gathered (DescribedPlan.GatherSteps): each group's dword vectors gathered into one
    // scratch made once a call, each step made from them straight into its dot product.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    // TE: 8 steps a group (else 4); TM: minimums. Constants to the JIT, as the tile's (GatheredOneSteps).
    private static unsafe float GatheredOne<TI, TF, TS, TE, TM>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TI : struct, IInstructions where TF : struct, IFactors where TS : struct, DescribedPlan.IGatheredSteps where TE : struct, IFlag where TM : struct, IFlag
    {
        bool mins = typeof(TM) == typeof(Yes), eight = typeof(TE) == typeof(Yes);
        int gpp = plan.GroupsPerPeriod, t = eight ? 8 : 4, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        Vector256<uint>* words = stackalloc Vector256<uint>[Math.Max(1, plan.MostGroupWords)];
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp, first = gp * t;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            plan.GatherWords<TI>(at, gp, words);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var i = Vector256.Load((int*)(lg + 8));
            i = TI.Dot(i, TS.Step<TI>(plan, words, first), Vector256.Load(xg));
            i = TI.Dot(i, TS.Step<TI>(plan, words, first + 1), Vector256.Load(xg + 32));
            i = TI.Dot(i, TS.Step<TI>(plan, words, first + 2), Vector256.Load(xg + 64));
            i = TI.Dot(i, TS.Step<TI>(plan, words, first + 3), Vector256.Load(xg + 96));
            if (eight)
            {
                i = TI.Dot(i, TS.Step<TI>(plan, words, first + 4), Vector256.Load(xg + 128));
                i = TI.Dot(i, TS.Step<TI>(plan, words, first + 5), Vector256.Load(xg + 160));
                i = TI.Dot(i, TS.Step<TI>(plan, words, first + 6), Vector256.Load(xg + 192));
                i = TI.Dot(i, TS.Step<TI>(plan, words, first + 7), Vector256.Load(xg + 224));
            }

            TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            f = TI.Fused(Vector256.ConvertToSingle(i), scale * Vector256.Load(lg), f);
            if (mins)
            {
                f = TI.Fused(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(f);
    }

    // GatheredOne by a natural plan (DescribedPlan.Natural: x's codes a chunk at a time as rounded): each step's dword
    // vectors loaded where they lie (DescribedPlan.Load), step c chunk c's 32 values in their order dotted with its 32 codes of x on
    // its own, the 8 chunks' sums then each summed across its lanes into lane c (LaneTotals). The same integer sum a chunk,
    // exactly, as GatheredOne's lane c, then the same float operations: the same bits.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    // TG: one group a period (its steps then the period's first 8, constants to the JIT). TP: its steps made two at once
    // (DescribedPlan.PairSteps: pair step q makes steps 2q and 2q + 1).
    private static unsafe float NaturalOne<TI, TF, TS, TM, TG, TP>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TI : struct, IInstructions where TF : struct, IFactors where TS : struct, DescribedPlan.IGatheredSteps where TM : struct, IFlag where TG : struct, IFlag where TP : struct, IFlag
    {
        bool mins = typeof(TM) == typeof(Yes);
        int gpp = plan.GroupsPerPeriod, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = typeof(TG) == typeof(Yes) ? g : g / gpp, gp = g - period * gpp, first = typeof(TG) == typeof(Yes) ? 0 : gp * 8;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            var words = (Vector256<uint>*)at;                                       // (a natural plan's steps load from the period)
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            Vector256<int> d0, d1, d2, d3, d4, d5, d6, d7;
            if (typeof(TP) == typeof(Yes))
            {
                int q = first / 2;
                TS.Pair<TI>(plan, words, q, out var s0, out var s1);
                d0 = TI.Dot(default, s0, Vector256.Load(xg));
                d1 = TI.Dot(default, s1, Vector256.Load(xg + 32));
                TS.Pair<TI>(plan, words, q + 1, out var s2, out var s3);
                d2 = TI.Dot(default, s2, Vector256.Load(xg + 64));
                d3 = TI.Dot(default, s3, Vector256.Load(xg + 96));
                TS.Pair<TI>(plan, words, q + 2, out var s4, out var s5);
                d4 = TI.Dot(default, s4, Vector256.Load(xg + 128));
                d5 = TI.Dot(default, s5, Vector256.Load(xg + 160));
                TS.Pair<TI>(plan, words, q + 3, out var s6, out var s7);
                d6 = TI.Dot(default, s6, Vector256.Load(xg + 192));
                d7 = TI.Dot(default, s7, Vector256.Load(xg + 224));
            }
            else
            {
                d0 = TI.Dot(default, TS.Step<TI>(plan, words, first), Vector256.Load(xg));
                d1 = TI.Dot(default, TS.Step<TI>(plan, words, first + 1), Vector256.Load(xg + 32));
                d2 = TI.Dot(default, TS.Step<TI>(plan, words, first + 2), Vector256.Load(xg + 64));
                d3 = TI.Dot(default, TS.Step<TI>(plan, words, first + 3), Vector256.Load(xg + 96));
                d4 = TI.Dot(default, TS.Step<TI>(plan, words, first + 4), Vector256.Load(xg + 128));
                d5 = TI.Dot(default, TS.Step<TI>(plan, words, first + 5), Vector256.Load(xg + 160));
                d6 = TI.Dot(default, TS.Step<TI>(plan, words, first + 6), Vector256.Load(xg + 192));
                d7 = TI.Dot(default, TS.Step<TI>(plan, words, first + 7), Vector256.Load(xg + 224));
            }

            var i = LaneTotals<TI>(d0, d1, d2, d3, d4, d5, d6, d7) + Vector256.Load((int*)(lg + 8));
            TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            f = TI.Fused(Vector256.ConvertToSingle(i), scale * Vector256.Load(lg), f);
            if (mins)
            {
                f = TI.Fused(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(f);
    }

    // Lane c: the sum of d_c's 8 lanes (32-bit integers, exactly: wrapping adds, every sum within them): pairs of lanes by
    // 32-bit unpacks, then 64-bit, then the 128-bit halves.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<int> LaneTotals<TI>(Vector256<int> d0, Vector256<int> d1, Vector256<int> d2, Vector256<int> d3, Vector256<int> d4, Vector256<int> d5,
        Vector256<int> d6, Vector256<int> d7) where TI : struct, IInstructions
    {
        var a0 = TI.UnpackLow(d0.AsUInt32(), d1.AsUInt32()) + TI.UnpackHigh(d0.AsUInt32(), d1.AsUInt32());
        var a1 = TI.UnpackLow(d2.AsUInt32(), d3.AsUInt32()) + TI.UnpackHigh(d2.AsUInt32(), d3.AsUInt32());
        var a2 = TI.UnpackLow(d4.AsUInt32(), d5.AsUInt32()) + TI.UnpackHigh(d4.AsUInt32(), d5.AsUInt32());
        var a3 = TI.UnpackLow(d6.AsUInt32(), d7.AsUInt32()) + TI.UnpackHigh(d6.AsUInt32(), d7.AsUInt32());
        var b0 = TI.UnpackLow(a0.AsUInt64(), a1.AsUInt64()).AsUInt32() + TI.UnpackHigh(a0.AsUInt64(), a1.AsUInt64()).AsUInt32();
        var b1 = TI.UnpackLow(a2.AsUInt64(), a3.AsUInt64()).AsUInt32() + TI.UnpackHigh(a2.AsUInt64(), a3.AsUInt64()).AsUInt32();
        return (TI.Lows(b0.AsUInt64(), b1.AsUInt64()).AsUInt32() + TI.Highs(b0.AsUInt64(), b1.AsUInt64()).AsUInt32()).AsInt32();
    }

    // MadeOne for a plan with uniform runs (P = 1): its steps loaded or gathered (TG: LoadedGroup, GatheredRunsGroup, by
    // the plan's load and map), 8 a group (TE, else 4), minimums (TM), each a type parameter chosen once a plan (OneBySteps).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float UniformOne<TG, TI, TF, TE, TM>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TG : struct, IGroupSteps<TG> where TI : struct, IInstructions where TF : struct, IFactors where TE : struct, IFlag where TM : struct, IFlag
    {
        int gpp = plan.GroupsPerPeriod, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        bool mins = typeof(TM) == typeof(Yes), c32 = typeof(TE) == typeof(Yes);
        var steps = TG.Of(plan);
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            steps.Make<TI>(plan, at, gp, null, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
            TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var i = Vector256.Load((int*)(lg + 8));
            i = TI.Dot(i, s0, Vector256.Load(xg));
            i = TI.Dot(i, s1, Vector256.Load(xg + 32));
            i = TI.Dot(i, s2, Vector256.Load(xg + 64));
            i = TI.Dot(i, s3, Vector256.Load(xg + 96));
            if (c32)
            {
                i = TI.Dot(i, s4, Vector256.Load(xg + 128));
                i = TI.Dot(i, s5, Vector256.Load(xg + 160));
                i = TI.Dot(i, s6, Vector256.Load(xg + 192));
                i = TI.Dot(i, s7, Vector256.Load(xg + 224));
            }

            f = TI.Fused(Vector256.ConvertToSingle(i), scale * Vector256.Load(lg), f);
            if (mins)
            {
                f = TI.Fused(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(f);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float MadeOne<TX, TI, TF, TS>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TX : struct, IStepLoad where TI : struct, IInstructions where TF : struct, IFactors where TS : struct, DescribedPlan.ISpanCodes
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, xStep = 4 * plan.Cg, cg = plan.Cg, p = plan.P;
        bool mins = plan.Mins;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Steps<TI, TS>(at, gp, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
            TF.Read<TI>(plan, at, gp, Math.Min(cg, chunks - g * cg) * p, out var scale, out var min);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var xs = Vector256.Load(lg);
            var i = Vector256.Load((int*)(lg + 8));
            i = TI.Dot(i, s0, TX.Load<TI>(xg));
            i = TI.Dot(i, s1, TX.Load<TI>(xg + xStep));
            i = TI.Dot(i, s2, TX.Load<TI>(xg + 2 * xStep));
            i = TI.Dot(i, s3, TX.Load<TI>(xg + 3 * xStep));
            if (t == 8)
            {
                i = TI.Dot(i, s4, TX.Load<TI>(xg + 4 * xStep));
                i = TI.Dot(i, s5, TX.Load<TI>(xg + 5 * xStep));
                i = TI.Dot(i, s6, TX.Load<TI>(xg + 6 * xStep));
                i = TI.Dot(i, s7, TX.Load<TI>(xg + 7 * xStep));
            }

            f = TI.Fused(Vector256.ConvertToSingle(i), scale * xs, f);
            if (mins)
            {
                f = TI.Fused(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(plan.LaneWeights is { } w ? f * w : f);
    }

    /// <summary>How the float32 lane makes 8 stored values of 2 or 4 bytes each float32 in registers (and one, for a row's tail).</summary>
    private unsafe interface ISingleRead
    {
        Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions;

        /// <summary>Eight's first 4 values, over 128 bits (the 128-bit twins' four-row kernels).</summary>
        Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions;

        float One(byte* at);
    }

    // A 16-bit float with float32's exponent bits and bias (bfloat16's description): float32's high half, by a shift.
    private readonly unsafe struct HighHalfRead : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions => Vector256.ShiftLeft(TI.Widen((ushort*)at), 16).AsSingle();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions => (FourShorts(at) << 16).AsSingle();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => BitConverter.Int32BitsToSingle(*(ushort*)at << 16);
    }

    // The 4 16-bit values at `at`, each a 32-bit lane (zero-extended).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<int> FourShorts(byte* at) => Vector128.WidenLower(Vector128.CreateScalarUnsafe(*(ulong*)at).AsUInt16()).AsInt32();

    // Any other 16-bit float by its bits (DescribedPlan.HalfBits).
    private readonly unsafe struct HalfRead((int SignShift, int Magnitude, int Shift, float Times, int Special) h) : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions => DescribedPlan.HalfBits(TI.Widen((ushort*)at), h);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions => DescribedPlan.HalfBits(FourShorts(at), h);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => DescribedPlan.HalfBits(Vector256.Create((int)*(ushort*)at), h).ToScalar();
    }

    // A 16-bit float whose reading rebiases by a power of two other than 1 (float16's: HalfBits multiplies every value, so
    // its vectors made once a plan give its bits exactly, DescribedPlan.Half16Vectors): 8 values made in registers by one
    // mask, shift, multiply, compare and the sign's two operations, with no test of the reading's fields a step.
    private readonly unsafe struct ScaledHalfRead(HalfVectors v, (int SignShift, int Magnitude, int Shift, float Times, int Special) h) : ISingleRead
    {
        /// <summary>Whether a reading is one this read gives the bits of: its power not 1, its sign (where it has one) bit 15's.</summary>
        public static bool Takes((int SignShift, int Magnitude, int Shift, float Times, int Special) half) => half.Times != 1f && half.SignShift is 0 or 16;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions => v.OfSigned(TI.Widen((short*)at));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions => DescribedPlan.HalfBits(FourShorts(at), h);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => DescribedPlan.HalfBits(Vector256.Create((int)*(ushort*)at), h).ToScalar();
    }

    // A 16-bit integer (two's complement where signed).
    private readonly unsafe struct ShortRead(bool signed) : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions =>
            Vector256.ConvertToSingle(signed ? TI.Widen((short*)at) : TI.Widen((ushort*)at));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions =>
            Vector128.ConvertToSingle(signed ? Vector128.WidenLower(Vector128.CreateScalarUnsafe(*(long*)at).AsInt16()) : FourShorts(at));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => signed ? *(short*)at : *(ushort*)at;
    }

    // float32 as stored.
    private readonly unsafe struct SingleRead : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions => Vector256.Load((float*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions => Vector128.Load((float*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => *(float*)at;
    }

    // A read plus the code offset (a description with one; one without takes the read alone).
    private readonly unsafe struct SingleOffset<TR>(TR read, float offset) : ISingleRead where TR : struct, ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight<TI>(byte* at) where TI : struct, IInstructions => read.Eight<TI>(at) + Vector256.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<float> Four<TI>(byte* at) where TI : struct, IInstructions => read.Four<TI>(at) + Vector128.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => read.One(at) + offset;
    }

    /// <summary>How the float64 lane makes 4 stored values float64 in registers (and one, for a row's tail).</summary>
    private unsafe interface IDoubleRead
    {
        Vector256<double> Four(byte* at);

        /// <summary>Four's first two values, over 128 bits (the 128-bit twins' four-row kernels).</summary>
        Vector128<double> Two(byte* at);

        double One(byte* at);
    }

    // float64 as stored.
    private readonly unsafe struct DoubleRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => Vector256.Load((double*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<double> Two(byte* at) => Vector128.Load((double*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(double*)at;
    }

    // A 64-bit two's complement integer, rounded once to float64 (LongsToDoubles).
    private readonly unsafe struct LongRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => LongsToDoubles((long*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<double> Two(byte* at) => LongsToDoubles128((long*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(long*)at;
    }

    // A 32-bit two's complement integer, exactly: each sign-extended to 64 bits and made float64 through the bits of
    // 2^52 + 2^51 (exact for every 32-bit integer).
    private readonly unsafe struct IntRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at)
        {
            var v = Vector128.Load((int*)at);
            var wide = Vector256.IsHardwareAccelerated ? Vector256.WidenLower(v.ToVector256Unsafe()) : Vector256.Create(Vector128.WidenLower(v), Vector128.WidenUpper(v));
            return (wide + Vector256.Create(6755399441055744.0).AsInt64()).AsDouble() - Vector256.Create(6755399441055744.0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<double> Two(byte* at)
        {
            var wide = Vector128.WidenLower(Vector128.CreateScalarUnsafe(*(long*)at).AsInt32());
            return (wide + Vector128.Create(6755399441055744.0).AsInt64()).AsDouble() - Vector128.Create(6755399441055744.0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(int*)at;
    }

    // A read plus the code offset.
    private readonly unsafe struct DoubleOffset<TR>(TR read, double offset) : IDoubleRead where TR : struct, IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => read.Four(at) + Vector256.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector128<double> Two(byte* at) => read.Two(at) + Vector128.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => read.One(at) + offset;
    }

    // The described float64 lane: four rows of x (float64, `stride` apart) by `count` stored values, dispatched on how a value
    // reads (and whether the description adds a code offset); by the 128-bit kernel where `narrow` (the 128-bit twin).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedDoubleFour<TI>(DescribedPlan plan, bool narrow, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TI : struct, IInstructions
    {
        switch (plan.Wide)
        {
            case WideRead.Long:
                DoubleFourBy<LongRead, TI>(new LongRead(), 8, plan.Offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Int:
                DoubleFourBy<IntRead, TI>(new IntRead(), 4, plan.Offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            default:
                DoubleFourBy<DoubleRead, TI>(new DoubleRead(), 8, plan.Offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoubleFourBy<TR, TI>(TR read, int bytes, double offset, bool narrow, byte* row, double* x, int stride, int count,
        out float y0, out float y1, out float y2, out float y3) where TR : struct, IDoubleRead where TI : struct, IInstructions
    {
        if (offset == 0)
        {
            if (narrow)
            {
                DoubleFour128<TR, TI>(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
                return;
            }

            DoubleFour<TR, TI>(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        if (narrow)
        {
            DoubleFour128<DoubleOffset<TR>, TI>(new DoubleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        DoubleFour<DoubleOffset<TR>, TI>(new DoubleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedDoubleOne<TI>(DescribedPlan plan, byte* row, double* x, int k) where TI : struct, IInstructions => plan.Wide switch
    {
        WideRead.Long => (float)DoubleOneBy<LongRead, TI>(new LongRead(), 8, plan.Offset, row, x, k),
        WideRead.Int => (float)DoubleOneBy<IntRead, TI>(new IntRead(), 4, plan.Offset, row, x, k),
        _ => (float)DoubleOneBy<DoubleRead, TI>(new DoubleRead(), 8, plan.Offset, row, x, k),
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe double DoubleOneBy<TR, TI>(TR read, int bytes, double offset, byte* row, double* x, int k) where TR : struct, IDoubleRead where TI : struct, IInstructions =>
        offset == 0 ? DoubleOne<TR, TI>(read, bytes, row, x, k) : DoubleOne<DoubleOffset<TR>, TI>(new DoubleOffset<TR>(read, offset), bytes, row, x, k);

    // Four rows of x (float64, `stride` apart: widened once a call) by `count` values of `bytes` bytes each: 8 values a step
    // made float64 in registers (TR), each multiplied with the four rows' matching values in float64 FMAs; two sums a row,
    // the last count mod 8 one at a time; each output its float64 sum rounded to float32 once. (x read as float32 and
    // widened in registers instead measured slower: 171 tokens a second on Qwen3-1.7B's prompts against 177.)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoubleFour<TR, TI>(TR read, int bytes, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, IDoubleRead where TI : struct, IInstructions
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 8 <= count; p += 8)
        {
            var w0 = read.Four(row + (nint)p * bytes);
            var w1 = read.Four(row + (nint)(p + 4) * bytes);
            double* xp = x + p;
            a0 = TI.Fused(w0, Vector256.Load(xp), a0);
            b0 = TI.Fused(w1, Vector256.Load(xp + 4), b0);
            a1 = TI.Fused(w0, Vector256.Load(xp + r1), a1);
            b1 = TI.Fused(w1, Vector256.Load(xp + r1 + 4), b1);
            a2 = TI.Fused(w0, Vector256.Load(xp + r2), a2);
            b2 = TI.Fused(w1, Vector256.Load(xp + r2 + 4), b2);
            a3 = TI.Fused(w0, Vector256.Load(xp + r3), a3);
            b3 = TI.Fused(w1, Vector256.Load(xp + r3 + 4), b3);
        }

        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (; p < count; p++)
        {
            double w = read.One(row + (nint)p * bytes);
            t0 = Math.FusedMultiplyAdd(w, x[p], t0);
            t1 = Math.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = Math.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = Math.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = (float)(Vector256.Sum(a0 + b0) + t0);
        y1 = (float)(Vector256.Sum(a1 + b1) + t1);
        y2 = (float)(Vector256.Sum(a2 + b2) + t2);
        y3 = (float)(Vector256.Sum(a3 + b3) + t3);
    }

    // DoubleFour on 128-bit vectors (its 128-bit twin): each 4 values made as two halves of 2 (TR.Two), each sum a register
    // a half.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoubleFour128<TR, TI>(TR read, int bytes, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, IDoubleRead where TI : struct, IInstructions
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector128<double> a0l = default, a0u = default, a1l = default, a1u = default, a2l = default, a2u = default, a3l = default, a3u = default;
        Vector128<double> b0l = default, b0u = default, b1l = default, b1u = default, b2l = default, b2u = default, b3l = default, b3u = default;
        int p = 0;
        for (; p + 8 <= count; p += 8)
        {
            byte* at = row + (nint)p * bytes;
            var w0l = read.Two(at);
            var w0u = read.Two(at + 2 * bytes);
            var w1l = read.Two(at + 4 * bytes);
            var w1u = read.Two(at + 6 * bytes);
            double* xp = x + p;
            a0l = TI.Fused(w0l, Vector128.Load(xp), a0l);
            a0u = TI.Fused(w0u, Vector128.Load(xp + 2), a0u);
            b0l = TI.Fused(w1l, Vector128.Load(xp + 4), b0l);
            b0u = TI.Fused(w1u, Vector128.Load(xp + 6), b0u);
            a1l = TI.Fused(w0l, Vector128.Load(xp + r1), a1l);
            a1u = TI.Fused(w0u, Vector128.Load(xp + r1 + 2), a1u);
            b1l = TI.Fused(w1l, Vector128.Load(xp + r1 + 4), b1l);
            b1u = TI.Fused(w1u, Vector128.Load(xp + r1 + 6), b1u);
            a2l = TI.Fused(w0l, Vector128.Load(xp + r2), a2l);
            a2u = TI.Fused(w0u, Vector128.Load(xp + r2 + 2), a2u);
            b2l = TI.Fused(w1l, Vector128.Load(xp + r2 + 4), b2l);
            b2u = TI.Fused(w1u, Vector128.Load(xp + r2 + 6), b2u);
            a3l = TI.Fused(w0l, Vector128.Load(xp + r3), a3l);
            a3u = TI.Fused(w0u, Vector128.Load(xp + r3 + 2), a3u);
            b3l = TI.Fused(w1l, Vector128.Load(xp + r3 + 4), b3l);
            b3u = TI.Fused(w1u, Vector128.Load(xp + r3 + 6), b3u);
        }

        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (; p < count; p++)
        {
            double w = read.One(row + (nint)p * bytes);
            t0 = Math.FusedMultiplyAdd(w, x[p], t0);
            t1 = Math.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = Math.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = Math.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = (float)(SumOf(a0l + b0l, a0u + b0u) + t0);
        y1 = (float)(SumOf(a1l + b1l, a1u + b1u) + t1);
        y2 = (float)(SumOf(a2l + b2l, a2u + b2u) + t2);
        y3 = (float)(SumOf(a3l + b3l, a3u + b3u) + t3);
    }

    // DoubleFour for one row of x, 16 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe double DoubleOne<TR, TI>(TR read, int bytes, byte* row, double* x, int k) where TR : struct, IDoubleRead where TI : struct, IInstructions
    {
        Vector256<double> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 16 <= k; p += 16)
        {
            byte* at = row + (nint)p * bytes;
            a = TI.Fused(read.Four(at), Vector256.Load(x + p), a);
            b = TI.Fused(read.Four(at + 4 * bytes), Vector256.Load(x + p + 4), b);
            c = TI.Fused(read.Four(at + 8 * bytes), Vector256.Load(x + p + 8), c);
            d = TI.Fused(read.Four(at + 12 * bytes), Vector256.Load(x + p + 12), d);
        }

        for (; p + 4 <= k; p += 4)
        {
            a = TI.Fused(read.Four(row + (nint)p * bytes), Vector256.Load(x + p), a);
        }

        double t = 0;
        for (; p < k; p++)
        {
            t = Math.FusedMultiplyAdd(read.One(row + (nint)p * bytes), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // The described float32 lane over values of 2 or 4 bytes: four rows of x by `count` values of a stored row, dispatched on
    // how a value reads (and whether the description adds a code offset); by the 128-bit kernel where `narrow` (the
    // 128-bit twin).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedSingleFour<TI>(DescribedPlan plan, bool narrow, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TI : struct, IInstructions
    {
        float offset = (float)plan.D.CodeOffset;
        switch (plan.Wide)
        {
            case WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 }:
                SingleFourBy<HighHalfRead, TI>(new HighHalfRead(), 2, offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Half when ScaledHalfRead.Takes(plan.Half16):
                SingleFourBy<ScaledHalfRead, TI>(new ScaledHalfRead(plan.Half16Vectors, plan.Half16), 2, offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Half:
                SingleFourBy<HalfRead, TI>(new HalfRead(plan.Half16), 2, offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Short:
                SingleFourBy<ShortRead, TI>(new ShortRead(plan.D.Code.Signed), 2, offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            default:
                SingleFourBy<SingleRead, TI>(new SingleRead(), 4, offset, narrow, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleFourBy<TR, TI>(TR read, int bytes, float offset, bool narrow, byte* row, float* x, int stride, int count,
        out float y0, out float y1, out float y2, out float y3) where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        if (offset == 0f)
        {
            if (narrow)
            {
                SingleFour128<TR, TI>(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
                return;
            }

            SingleFour<TR, TI>(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        if (narrow)
        {
            SingleFour128<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        SingleFour<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedSingleOne<TI>(DescribedPlan plan, byte* row, float* x, int k) where TI : struct, IInstructions
    {
        float offset = (float)plan.D.CodeOffset;
        return plan.Wide switch
        {
            WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 } => SingleOneBy<HighHalfRead, TI>(new HighHalfRead(), 2, offset, row, x, k),
            WideRead.Half when ScaledHalfRead.Takes(plan.Half16) => SingleOneBy<ScaledHalfRead, TI>(new ScaledHalfRead(plan.Half16Vectors, plan.Half16), 2, offset, row, x, k),
            WideRead.Half => SingleOneBy<HalfRead, TI>(new HalfRead(plan.Half16), 2, offset, row, x, k),
            WideRead.Short => SingleOneBy<ShortRead, TI>(new ShortRead(plan.D.Code.Signed), 2, offset, row, x, k),
            _ => SingleOneBy<SingleRead, TI>(new SingleRead(), 4, offset, row, x, k),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SingleOneBy<TR, TI>(TR read, int bytes, float offset, byte* row, float* x, int k) where TR : struct, ISingleRead where TI : struct, IInstructions =>
        offset == 0f ? SingleOne<TR, TI>(read, bytes, row, x, k) : SingleOne<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, k);

    // DescribedSingleFour for eight rows of x (SingleEight), the plan's read of its values as there; TH (the 128-bit twin):
    // four rows a call (SingleEightHalf128).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedSingleEight<TI, TH>(DescribedPlan plan, byte* row, float* x, int stride, int count, float* y)
        where TI : struct, IInstructions where TH : struct, IFlag
    {
        float offset = (float)plan.D.CodeOffset;
        switch (plan.Wide)
        {
            case WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 }:
                SingleEightBy<HighHalfRead, TI, TH>(new HighHalfRead(), 2, offset, row, x, stride, count, y);
                return;
            case WideRead.Half when ScaledHalfRead.Takes(plan.Half16):
                SingleEightBy<ScaledHalfRead, TI, TH>(new ScaledHalfRead(plan.Half16Vectors, plan.Half16), 2, offset, row, x, stride, count, y);
                return;
            case WideRead.Half:
                SingleEightBy<HalfRead, TI, TH>(new HalfRead(plan.Half16), 2, offset, row, x, stride, count, y);
                return;
            case WideRead.Short:
                SingleEightBy<ShortRead, TI, TH>(new ShortRead(plan.D.Code.Signed), 2, offset, row, x, stride, count, y);
                return;
            default:
                SingleEightBy<SingleRead, TI, TH>(new SingleRead(), 4, offset, row, x, stride, count, y);
                return;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleEightBy<TR, TI, TH>(TR read, int bytes, float offset, byte* row, float* x, int stride, int count, float* y)
        where TR : struct, ISingleRead where TI : struct, IInstructions where TH : struct, IFlag
    {
        if (typeof(TH) == typeof(Yes))
        {
            if (offset == 0f)
            {
                SingleEightHalf128<TR, TI>(read, bytes, row, x, stride, count, y);
                return;
            }

            SingleEightHalf128<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, stride, count, y);
            return;
        }

        if (offset == 0f)
        {
            SingleEight<TR, TI>(read, bytes, row, x, stride, count, y);
            return;
        }

        SingleEight<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, stride, count, y);
    }

    // Eight rows of x (`stride` apart) by `count` values of `bytes` bytes each: 8 values a step made float32 in registers
    // (TR) once for the eight rows, each multiplied with every row's matching values; a sum a row (lane j the values j mod 8
    // in order), the last count mod 8 one at a time. The values are made once for every eight rows of x, where four rows a
    // pass made each twice as often: the eight sums and the values made fit the registers, two sums a row would not.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleEight<TR, TI>(TR read, int bytes, byte* row, float* x, int stride, int count, float* y)
        where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        float* x0 = x, x1 = x + stride, x2 = x + 2 * (nint)stride, x3 = x + 3 * (nint)stride, x4 = x + 4 * (nint)stride, x5 = x + 5 * (nint)stride,
            x6 = x + 6 * (nint)stride, x7 = x + 7 * (nint)stride;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default, a6 = default, a7 = default;
        nint p = 0;
        for (; p + 8 <= count; p += 8)
        {
            var w = read.Eight<TI>(row + p * bytes);
            a0 = TI.Fused(w, Vector256.Load(x0 + p), a0);
            a1 = TI.Fused(w, Vector256.Load(x1 + p), a1);
            a2 = TI.Fused(w, Vector256.Load(x2 + p), a2);
            a3 = TI.Fused(w, Vector256.Load(x3 + p), a3);
            a4 = TI.Fused(w, Vector256.Load(x4 + p), a4);
            a5 = TI.Fused(w, Vector256.Load(x5 + p), a5);
            a6 = TI.Fused(w, Vector256.Load(x6 + p), a6);
            a7 = TI.Fused(w, Vector256.Load(x7 + p), a7);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f, t4 = 0f, t5 = 0f, t6 = 0f, t7 = 0f;
        for (; p < count; p++)
        {
            float v = read.One(row + p * bytes);
            t0 = MathF.FusedMultiplyAdd(v, x0[p], t0);
            t1 = MathF.FusedMultiplyAdd(v, x1[p], t1);
            t2 = MathF.FusedMultiplyAdd(v, x2[p], t2);
            t3 = MathF.FusedMultiplyAdd(v, x3[p], t3);
            t4 = MathF.FusedMultiplyAdd(v, x4[p], t4);
            t5 = MathF.FusedMultiplyAdd(v, x5[p], t5);
            t6 = MathF.FusedMultiplyAdd(v, x6[p], t6);
            t7 = MathF.FusedMultiplyAdd(v, x7[p], t7);
        }

        y[0] = Vector256.Sum(a0) + t0;
        y[1] = Vector256.Sum(a1) + t1;
        y[2] = Vector256.Sum(a2) + t2;
        y[3] = Vector256.Sum(a3) + t3;
        y[4] = Vector256.Sum(a4) + t4;
        y[5] = Vector256.Sum(a5) + t5;
        y[6] = Vector256.Sum(a6) + t6;
        y[7] = Vector256.Sum(a7) + t7;
    }

    // SingleEight on 128-bit vectors (its 128-bit twin), four rows of x a call: each 8 values made as two halves of 4
    // (TR.Four), a row's sum two registers (lanes 0-3, 4-7: each lane the values j mod 8 in order, as SingleEight's), added
    // as Vector256.Sum adds a vector's halves (SumOf): the same bits as SingleEight's rows.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleEightHalf128<TR, TI>(TR read, int bytes, byte* row, float* x, int stride, int count, float* y)
        where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        float* x0 = x, x1 = x + stride, x2 = x + 2 * (nint)stride, x3 = x + 3 * (nint)stride;
        Vector128<float> a0l = default, a0u = default, a1l = default, a1u = default, a2l = default, a2u = default, a3l = default, a3u = default;
        nint p = 0;
        for (; p + 8 <= count; p += 8)
        {
            byte* at = row + p * bytes;
            var lo = read.Four<TI>(at);
            var hi = read.Four<TI>(at + 4 * bytes);
            a0l = TI.Fused(lo, Vector128.Load(x0 + p), a0l);
            a0u = TI.Fused(hi, Vector128.Load(x0 + p + 4), a0u);
            a1l = TI.Fused(lo, Vector128.Load(x1 + p), a1l);
            a1u = TI.Fused(hi, Vector128.Load(x1 + p + 4), a1u);
            a2l = TI.Fused(lo, Vector128.Load(x2 + p), a2l);
            a2u = TI.Fused(hi, Vector128.Load(x2 + p + 4), a2u);
            a3l = TI.Fused(lo, Vector128.Load(x3 + p), a3l);
            a3u = TI.Fused(hi, Vector128.Load(x3 + p + 4), a3u);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float v = read.One(row + p * bytes);
            t0 = MathF.FusedMultiplyAdd(v, x0[p], t0);
            t1 = MathF.FusedMultiplyAdd(v, x1[p], t1);
            t2 = MathF.FusedMultiplyAdd(v, x2[p], t2);
            t3 = MathF.FusedMultiplyAdd(v, x3[p], t3);
        }

        y[0] = SumOf(a0l, a0u) + t0;
        y[1] = SumOf(a1l, a1u) + t1;
        y[2] = SumOf(a2l, a2u) + t2;
        y[3] = SumOf(a3l, a3u) + t3;
    }

    // Four rows of x (`stride` apart) by `count` values of `bytes` bytes each: 16 values a step made float32 in registers
    // (TR), each multiplied with the four rows' matching values; two sums a row, the last count mod 16 one at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleFour<TR, TI>(TR read, int bytes, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 16 <= count; p += 16)
        {
            var lo = read.Eight<TI>(row + (nint)p * bytes);
            var hi = read.Eight<TI>(row + (nint)(p + 8) * bytes);
            float* xp = x + p;
            a0 = TI.Fused(lo, Vector256.Load(xp), a0);
            b0 = TI.Fused(hi, Vector256.Load(xp + 8), b0);
            a1 = TI.Fused(lo, Vector256.Load(xp + r1), a1);
            b1 = TI.Fused(hi, Vector256.Load(xp + r1 + 8), b1);
            a2 = TI.Fused(lo, Vector256.Load(xp + r2), a2);
            b2 = TI.Fused(hi, Vector256.Load(xp + r2 + 8), b2);
            a3 = TI.Fused(lo, Vector256.Load(xp + r3), a3);
            b3 = TI.Fused(hi, Vector256.Load(xp + r3 + 8), b3);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float w = read.One(row + (nint)p * bytes);
            t0 = MathF.FusedMultiplyAdd(w, x[p], t0);
            t1 = MathF.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = MathF.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = MathF.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = Vector256.Sum(a0 + b0) + t0;
        y1 = Vector256.Sum(a1 + b1) + t1;
        y2 = Vector256.Sum(a2 + b2) + t2;
        y3 = Vector256.Sum(a3 + b3) + t3;
    }

    // SingleFour on 128-bit vectors (its 128-bit twin): each 8 values made as two halves of 4 (TR.Four), each sum a register
    // a half.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleFour128<TR, TI>(TR read, int bytes, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector128<float> a0l = default, a0u = default, a1l = default, a1u = default, a2l = default, a2u = default, a3l = default, a3u = default;
        Vector128<float> b0l = default, b0u = default, b1l = default, b1u = default, b2l = default, b2u = default, b3l = default, b3u = default;
        int p = 0;
        for (; p + 16 <= count; p += 16)
        {
            byte* at = row + (nint)p * bytes;
            var lol = read.Four<TI>(at);
            var lou = read.Four<TI>(at + 4 * bytes);
            var hil = read.Four<TI>(at + 8 * bytes);
            var hiu = read.Four<TI>(at + 12 * bytes);
            float* xp = x + p;
            a0l = TI.Fused(lol, Vector128.Load(xp), a0l);
            a0u = TI.Fused(lou, Vector128.Load(xp + 4), a0u);
            b0l = TI.Fused(hil, Vector128.Load(xp + 8), b0l);
            b0u = TI.Fused(hiu, Vector128.Load(xp + 12), b0u);
            a1l = TI.Fused(lol, Vector128.Load(xp + r1), a1l);
            a1u = TI.Fused(lou, Vector128.Load(xp + r1 + 4), a1u);
            b1l = TI.Fused(hil, Vector128.Load(xp + r1 + 8), b1l);
            b1u = TI.Fused(hiu, Vector128.Load(xp + r1 + 12), b1u);
            a2l = TI.Fused(lol, Vector128.Load(xp + r2), a2l);
            a2u = TI.Fused(lou, Vector128.Load(xp + r2 + 4), a2u);
            b2l = TI.Fused(hil, Vector128.Load(xp + r2 + 8), b2l);
            b2u = TI.Fused(hiu, Vector128.Load(xp + r2 + 12), b2u);
            a3l = TI.Fused(lol, Vector128.Load(xp + r3), a3l);
            a3u = TI.Fused(lou, Vector128.Load(xp + r3 + 4), a3u);
            b3l = TI.Fused(hil, Vector128.Load(xp + r3 + 8), b3l);
            b3u = TI.Fused(hiu, Vector128.Load(xp + r3 + 12), b3u);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float w = read.One(row + (nint)p * bytes);
            t0 = MathF.FusedMultiplyAdd(w, x[p], t0);
            t1 = MathF.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = MathF.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = MathF.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = SumOf(a0l + b0l, a0u + b0u) + t0;
        y1 = SumOf(a1l + b1l, a1u + b1u) + t1;
        y2 = SumOf(a2l + b2l, a2u + b2u) + t2;
        y3 = SumOf(a3l + b3l, a3u + b3u) + t3;
    }

    // SingleFour for one row of x, 32 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SingleOne<TR, TI>(TR read, int bytes, byte* row, float* x, int k) where TR : struct, ISingleRead where TI : struct, IInstructions
    {
        Vector256<float> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 32 <= k; p += 32)
        {
            byte* at = row + (nint)p * bytes;
            a = TI.Fused(read.Eight<TI>(at), Vector256.Load(x + p), a);
            b = TI.Fused(read.Eight<TI>(at + 8 * bytes), Vector256.Load(x + p + 8), b);
            c = TI.Fused(read.Eight<TI>(at + 16 * bytes), Vector256.Load(x + p + 16), c);
            d = TI.Fused(read.Eight<TI>(at + 24 * bytes), Vector256.Load(x + p + 24), d);
        }

        for (; p + 8 <= k; p += 8)
        {
            a = TI.Fused(read.Eight<TI>(row + (nint)p * bytes), Vector256.Load(x + p), a);
        }

        float t = 0f;
        for (; p < k; p++)
        {
            t = MathF.FusedMultiplyAdd(read.One(row + (nint)p * bytes), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // The described float32 lane over codes of up to 8 bits (no integer arithmetic for their numbers): four rows of x (in
    // the spans' order) by one stored row, each group's scales and minimums read at once (DescribedPlan.Factor), each span's
    // codes made (DescribedPlan.Raw) and 8 at a time looked up in the table of their numbers, times their chunk's scale less
    // its minimum, each multiplied with the four rows; the row's last period from `tail`.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedCodesFour<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x, nint stride,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        bool plain = plan.Plain;                                            // no factors: the numbers as they are (x · 1 + 0 is x)
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Factor<TI>(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            for (int s = 0; s < spans; s++)
            {
                var codes = plan.Raw<TI>(at, gp * spans + s);
                for (int e = 0; e < 4; e++)
                {
                    int lane = (32 * s + 8 * e) / c;
                    var g8 = TI.Gather(table, TI.Widen(EightAt(codes, e), false));
                    var w = plain ? g8 : TI.Fused(g8, TI.Broadcast(scale, lane), TI.Broadcast(min, lane));
                    float* xp = x + (nint)g * plan.GroupValues + 32 * s + 8 * e;
                    a0 = TI.Fused(w, Vector256.Load(xp), a0);
                    a1 = TI.Fused(w, Vector256.Load(xp + stride), a1);
                    a2 = TI.Fused(w, Vector256.Load(xp + 2 * stride), a2);
                    a3 = TI.Fused(w, Vector256.Load(xp + 3 * stride), a3);
                }
            }
        }

        y0 = Vector256.Sum(a0);
        y1 = Vector256.Sum(a1);
        y2 = Vector256.Sum(a2);
        y3 = Vector256.Sum(a3);
    }

    // DescribedCodesFour on 128-bit vectors (its 128-bit twin): each group's scales and minimums and each span's codes made
    // in registers once for the four rows (the plan's Factor and Raw, as the 256-bit kernel makes them), then each 8 codes'
    // numbers gathered from the table and taken as two halves, times their chunk's scale less its minimum (the lane the
    // 256-bit kernel broadcasts, its lower half), each sum a register a half.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedCodesFour128<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x, nint stride,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        nint s2 = 2 * stride, s3 = 3 * stride;
        Vector128<float> a0l = default, a0u = default, a1l = default, a1u = default, a2l = default, a2u = default, a3l = default, a3u = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Factor<TI>(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            for (int s = 0; s < spans; s++)
            {
                var codes = plan.Raw<TI>(at, gp * spans + s);
                for (int e = 0; e < 4; e++)
                {
                    int lane = (32 * s + 8 * e) / c;
                    var d = TI.Broadcast(scale, lane).GetLower();
                    var m = TI.Broadcast(min, lane).GetLower();
                    var g8 = TI.Gather(table, TI.Widen(EightAt(codes, e), false));
                    var wl = TI.Fused(g8.GetLower(), d, m);
                    var wu = TI.Fused(g8.GetUpper(), d, m);
                    float* xp = x + (nint)g * plan.GroupValues + 32 * s + 8 * e;
                    a0l = TI.Fused(wl, Vector128.Load(xp), a0l);
                    a0u = TI.Fused(wu, Vector128.Load(xp + 4), a0u);
                    a1l = TI.Fused(wl, Vector128.Load(xp + stride), a1l);
                    a1u = TI.Fused(wu, Vector128.Load(xp + stride + 4), a1u);
                    a2l = TI.Fused(wl, Vector128.Load(xp + s2), a2l);
                    a2u = TI.Fused(wu, Vector128.Load(xp + s2 + 4), a2u);
                    a3l = TI.Fused(wl, Vector128.Load(xp + s3), a3l);
                    a3u = TI.Fused(wu, Vector128.Load(xp + s3 + 4), a3u);
                }
            }
        }

        y0 = SumOf(a0l, a0u);
        y1 = SumOf(a1l, a1u);
        y2 = SumOf(a2l, a2u);
        y3 = SumOf(a3l, a3u);
    }

    // DescribedCodesFour for one row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedCodesOne<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x) where TI : struct, IInstructions
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        bool plain = plan.Plain;
        Vector256<float> a = default, b = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Factor<TI>(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            for (int s = 0; s < spans; s++)
            {
                var codes = plan.Raw<TI>(at, gp * spans + s);
                float* xp = x + (nint)g * plan.GroupValues + 32 * s;
                for (int e = 0; e < 4; e += 2)
                {
                    int lane = (32 * s + 8 * e) / c, next = (32 * s + 8 * e + 8) / c;
                    var g0 = TI.Gather(table, TI.Widen(EightAt(codes, e), false));
                    var g1 = TI.Gather(table, TI.Widen(EightAt(codes, e + 1), false));
                    a = TI.Fused(plain ? g0 : TI.Fused(g0, TI.Broadcast(scale, lane), TI.Broadcast(min, lane)), Vector256.Load(xp + 8 * e), a);
                    b = TI.Fused(plain ? g1 : TI.Fused(g1, TI.Broadcast(scale, next), TI.Broadcast(min, next)), Vector256.Load(xp + 8 * e + 8), b);
                }
            }
        }

        return Vector256.Sum(a + b);
    }

    // Bytes 8e..8e + 7 of a span's codes, in the low 8 bytes.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector128<byte> EightAt(Vector256<byte> codes, int e) => e switch
    {
        0 => codes.GetLower(),
        1 => UpperEight(codes.GetLower().AsSByte()).AsByte(),
        2 => codes.GetUpper(),
        _ => UpperEight(codes.GetUpper().AsSByte()).AsByte(),
    };

    /// <summary>
    /// A 16-bit float's reading by its bits (DescribedPlan.HalfOf) as vectors made once, a factor (a power of two: the
    /// integer lane's unit) folded in: the magnitude masked and shifted to float32's place, times the power of two that
    /// rebiases it and the factor, its specials' exponent set (none: never), its sign moved to bit 31 (none: masked off).
    /// </summary>
    internal readonly struct HalfVectors((int SignShift, int Magnitude, int Shift, float Times, int Special) h, float factor)
    {
        private readonly Vector256<int> magnitude = Vector256.Create(h.Magnitude), special = Vector256.Create(h.Special == 0 ? int.MaxValue : h.Special),
            sign = Vector256.Create(h.SignShift == 0 ? 0 : 0x8000), below = Vector256.Create(h.Special == 0 ? int.MaxValue : h.Special - 1),
            top = Vector256.Create(h.SignShift == 0 ? 0 : int.MinValue);
        private readonly Vector256<float> times = Vector256.Create(h.Times * factor);
        private readonly int shift = h.Shift;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Of(Vector256<int> bits)
        {
            var m = bits & magnitude;
            var value = (Vector256.ShiftLeft(m, shift).AsSingle() * times).AsInt32();
            value |= Vector256.GreaterThanOrEqual(m, special) & Vector256.Create(0x7F800000);
            return (value | Vector256.ShiftLeft(bits & sign, 16)).AsSingle();
        }

        /// <summary>
        /// Of from the 16-bit values sign-extended (each lane's bits 15-31 its bit 15): the sign is the lane's top bit as it
        /// is (no shift), the magnitude its low bits as Of's, and the specials by one compare, greater than their start less
        /// one: the same number of each value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> OfSigned(Vector256<int> bits)
        {
            var m = bits & magnitude;
            var value = (Vector256.ShiftLeft(m, shift).AsSingle() * times).AsInt32();
            value |= Vector256.GreaterThan(m, below) & Vector256.Create(0x7F800000);
            return (value | (bits & top)).AsSingle();
        }
    }

    // 8 chunks' 32 bytes (a chunk a vector) as the integer lane's 8 steps: step s holds dword s of each chunk, in lane order.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void Transpose8<TI>(Vector256<byte> r0, Vector256<byte> r1, Vector256<byte> r2, Vector256<byte> r3, Vector256<byte> r4, Vector256<byte> r5,
        Vector256<byte> r6, Vector256<byte> r7, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
        out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions
    {
        var t0 = TI.UnpackLow(r0.AsUInt32(), r1.AsUInt32());
        var t1 = TI.UnpackHigh(r0.AsUInt32(), r1.AsUInt32());
        var t2 = TI.UnpackLow(r2.AsUInt32(), r3.AsUInt32());
        var t3 = TI.UnpackHigh(r2.AsUInt32(), r3.AsUInt32());
        var t4 = TI.UnpackLow(r4.AsUInt32(), r5.AsUInt32());
        var t5 = TI.UnpackHigh(r4.AsUInt32(), r5.AsUInt32());
        var t6 = TI.UnpackLow(r6.AsUInt32(), r7.AsUInt32());
        var t7 = TI.UnpackHigh(r6.AsUInt32(), r7.AsUInt32());
        var u0 = TI.UnpackLow(t0.AsUInt64(), t2.AsUInt64());
        var u1 = TI.UnpackHigh(t0.AsUInt64(), t2.AsUInt64());
        var u2 = TI.UnpackLow(t1.AsUInt64(), t3.AsUInt64());
        var u3 = TI.UnpackHigh(t1.AsUInt64(), t3.AsUInt64());
        var u4 = TI.UnpackLow(t4.AsUInt64(), t6.AsUInt64());
        var u5 = TI.UnpackHigh(t4.AsUInt64(), t6.AsUInt64());
        var u6 = TI.UnpackLow(t5.AsUInt64(), t7.AsUInt64());
        var u7 = TI.UnpackHigh(t5.AsUInt64(), t7.AsUInt64());
        s0 = TI.Lows(u0, u4).AsByte();
        s4 = TI.Highs(u0, u4).AsByte();
        s1 = TI.Lows(u1, u5).AsByte();
        s5 = TI.Highs(u1, u5).AsByte();
        s2 = TI.Lows(u2, u6).AsByte();
        s6 = TI.Highs(u2, u6).AsByte();
        s3 = TI.Lows(u3, u7).AsByte();
        s7 = TI.Highs(u3, u7).AsByte();
    }

    // 4 spans of two 16-value chunks each (chunk 2q in span q's low half, 2q + 1 in its high half) as the integer lane's 4
    // steps: step s holds dword s of each of the 8 chunks, in lane order.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void Transpose4<TI>(Vector256<byte> r0, Vector256<byte> r1, Vector256<byte> r2, Vector256<byte> r3,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3) where TI : struct, IInstructions
    {
        var pairs = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        var w0 = TI.Permute(r0.AsInt32(), pairs).AsUInt64();
        var w1 = TI.Permute(r1.AsInt32(), pairs).AsUInt64();
        var w2 = TI.Permute(r2.AsInt32(), pairs).AsUInt64();
        var w3 = TI.Permute(r3.AsInt32(), pairs).AsUInt64();
        var a = TI.UnpackLow(w0, w1);
        var b = TI.UnpackHigh(w0, w1);
        var c = TI.UnpackLow(w2, w3);
        var d = TI.UnpackHigh(w2, w3);
        s0 = TI.Lows(a, c).AsByte();
        s2 = TI.Highs(a, c).AsByte();
        s1 = TI.Lows(b, d).AsByte();
        s3 = TI.Highs(b, d).AsByte();
    }
}
