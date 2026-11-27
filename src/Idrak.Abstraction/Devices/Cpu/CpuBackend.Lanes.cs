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

    // The described lane's kind of lane (a lane id's low 4 bits: LaneOf).
    private const int DescribedLaneId = 11;

    // The described lane's second kind, where its float32 lane reads 16-bit values (DescribedKernels.Decodes): the same
    // lane with a group's stored rows made float32 once into the tile's scratch, which every four rows of x then read
    // (each value the number the first kind makes in registers, the same multiply-adds in the same order: the same bits);
    // the measured choice takes the faster at each shape.
    private const int DecodedLaneId = 12;

    // The described lane's name in the recorded choices.
    private const string DescribedLaneName = "described lane (any type read from its description alone, its values decoded in registers inside the dot products)";

    // Its second kind's (DecodedLaneId).
    private const string DecodedLaneName = "described lane, a group's 16-bit rows made float32 once for four rows of x at a time";

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
    private static string LaneName(int lane) => (KindOf(lane) == DecodedLaneId ? DecodedLaneName : DescribedLaneName) + InstructionsOf(lane) switch
    {
        Instructions.Avx2 => " by AVX2 and FMA",
        Instructions.Vnni => " by AVX-VNNI",
        Instructions.Neon => " by NEON",
        Instructions.DotProd => " by Arm's dot products (DotProd)",
        Instructions.DotProdWide => " by Arm's dot products, two a step for codes past 127 (DotProd)",
        _ => "",
    } + (NarrowOf(lane) ? ", four rows of x at a time on 128-bit vectors" : "");

    // A lane's short name in a measurement's times (its kind's first word, "decoded" for the second kind, its instruction
    // set's, and /128 for a 128-bit twin).
    private static string LaneShortName(int lane) => (KindOf(lane) == DecodedLaneId ? "decoded" : DescribedLaneName.Split(' ')[0]) + InstructionsOf(lane) switch
    {
        Instructions.Avx2 => "/avx2",
        Instructions.Vnni => "/avx-vnni",
        Instructions.Neon => "/neon",
        Instructions.DotProd => "/dotprod",
        Instructions.DotProdWide => "/dotprod2",
        _ => "",
    } + (NarrowOf(lane) ? "/128" : "");

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
            if (m >= 4)
            {
                return this;
            }

            var wide = default(LaneList);
            for (int i = 0; i < Count; i++)
            {
                if (!NarrowOf(this[i]) && KindOf(this[i]) != DecodedLaneId)
                {
                    wide = wide.With(this[i]);
                }
            }

            return wide;
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
            var even = ((c & Vector256.Create((short)0xFF)) * Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(v, 8), 8)).AsInt32();
            var odd = (Vector256.ShiftRightLogical(c, 8) * Vector256.ShiftRightArithmetic(v, 8)).AsInt32();
            return sum + (Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(even, 16), 16) + Vector256.ShiftRightArithmetic(even, 16))
                + (Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(odd, 16), 16) + Vector256.ShiftRightArithmetic(odd, 16));
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
            v = Vector256.ConditionalSelect(Bit(counts, 1), Vector256.ShiftRightLogical(v, 1), v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), Vector256.ShiftRightLogical(v, 2), v);
            v = Vector256.ConditionalSelect(Bit(counts, 4), Vector256.ShiftRightLogical(v, 4), v);
            v = Vector256.ConditionalSelect(Bit(counts, 8), Vector256.ShiftRightLogical(v, 8), v);
            return Vector256.ConditionalSelect(Bit(counts, 16), Vector256.ShiftRightLogical(v, 16), v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<uint> ShiftLeft(Vector256<uint> v, Vector256<uint> counts)
        {
            v = Vector256.ConditionalSelect(Bit(counts, 1), Vector256.ShiftLeft(v, 1), v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), Vector256.ShiftLeft(v, 2), v);
            v = Vector256.ConditionalSelect(Bit(counts, 4), Vector256.ShiftLeft(v, 4), v);
            v = Vector256.ConditionalSelect(Bit(counts, 8), Vector256.ShiftLeft(v, 8), v);
            return Vector256.ConditionalSelect(Bit(counts, 16), Vector256.ShiftLeft(v, 16), v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<ulong> ShiftRight(Vector256<ulong> v, Vector256<ulong> counts)
        {
            v = Vector256.ConditionalSelect(Bit(counts, 1), Vector256.ShiftRightLogical(v, 1), v);
            v = Vector256.ConditionalSelect(Bit(counts, 2), Vector256.ShiftRightLogical(v, 2), v);
            v = Vector256.ConditionalSelect(Bit(counts, 4), Vector256.ShiftRightLogical(v, 4), v);
            v = Vector256.ConditionalSelect(Bit(counts, 8), Vector256.ShiftRightLogical(v, 8), v);
            v = Vector256.ConditionalSelect(Bit(counts, 16), Vector256.ShiftRightLogical(v, 16), v);
            return Vector256.ConditionalSelect(Bit(counts, 32), Vector256.ShiftRightLogical(v, 32), v);
        }

        // All bits set in the lanes whose count has `bit` set.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> Bit(Vector256<uint> counts, uint bit) => Vector256.Equals(counts & Vector256.Create(bit), Vector256.Create(bit));

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
            var w = first == 0 ? word : Vector256.ShiftRightLogical(word, first);
            return Vector256.ShiftRightLogical(w * Vector256.Create(1u << 29, 1u << 27, 1u << 25, 1u << 23, 1u << 21, 1u << 19, 1u << 17, 1u << 15), 29);
        }

        public static bool ShiftsByLane => false;

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
    // (the same operations in the same order, so the same bits), which are never inlined, so each is compiled on its own
    // budget: IntegerOne, ScaledStepsOne, SixIntegerOne, StepsOne, WideStepsOne. Every lane's 128-bit twin (NarrowLane) has
    // four-row kernels written on 128-bit vectors (each 256-bit vector of the 256-bit kernel as its two halves, every
    // operation the 256-bit one's on its half, and the halves' sums added as Vector256.Sum adds them: the same bits), whose
    // operations are the CPU's own instructions wherever 128-bit vectors are accelerated: four rows of x share each stored
    // row's decode there too. The measured choice times the twins against the 256-bit lanes for four rows of x or more.

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
        ConcurrentDictionary<(CpuVectorPaths Vectors, int Rows, int N, int K), (int Lane, int Generation)> Chosen { get; }

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

        /// <summary>The floats and bytes of scratch a tile's kernels of <paramref name="lane"/> take, for a group of <paramref name="group"/> stored rows (0: rows read alone).</summary>
        (int Floats, int Bytes) Scratch(int lane, int k, int group);

        /// <summary>The kernels of <paramref name="lane"/> (its kind, by its instructions) for a tile: x and the scratch (its bytes cleared) pinned.</summary>
        TKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes);
    }

    /// <summary>A lane's kernels over a weight's rows read where they are stored (RowGroups').</summary>
    private unsafe interface IRowKernels
    {
        /// <summary>Before stored rows [c0, c0 + count) are multiplied by more than one row of x.</summary>
        void Group(byte* stored, nint rowBytes, int c0, int count);

        /// <summary>Rows i..i + 3 of x by the group's stored row c (at <paramref name="row"/>), over values [p0, p1) of k.</summary>
        void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3);

        /// <summary>Row i of x by the group's stored row c.</summary>
        float One(byte* row, int c, int i);

        /// <summary>Row i of x by a stored row read alone (one row of x: each stored row read once, no group).</summary>
        float Single(byte* row, int i);
    }

    /// <summary>x as a lane reads it: x itself, or what the lane made of it first (int8 codes and their scales; values reordered).</summary>
    private readonly record struct LaneInput(float[] Values, bool Owned, sbyte[]? Codes, float[]? Scales)
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
        public LanePointers(float* values, sbyte* codes, float* scales)
        {
            Values = values;
            Codes = codes;
            Scales = scales;
        }

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
    // made under one setting is not taken under another), the rows by class (exact up to 8, then powers of two, so prompts
    // of every length measure a few classes), n and k.
    private static (CpuVectorPaths Vectors, int Rows, int N, int K) ShapeOf(int m, int n, int k) =>
        (ComputeResources.CpuVectors, m <= 8 ? m : (int)BitOperations.RoundUpToPowerOf2((uint)m), n, k);

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
            CpuChoices.Keep(LaneFileKey(product, m, n, k), best.ToString(System.Globalization.CultureInfo.InvariantCulture));
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

    // The machine's kept lane for the shape (CpuChoices), where it is among the candidates.
    private static int? KeptLane<TC>(in TC product, int m, int n, int k, int count) where TC : struct, ILaneChoice
    {
        if (!CpuChoices.TryGet(LaneFileKey(product, m, n, k), out var kept)
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
    // allowed, the rows' class, n and k; and, where the candidates hold the lane's second kind (DecodedLaneId), its name, so
    // a choice kept from among the first kind's lanes alone is measured again with it.
    private static string LaneFileKey<TC>(in TC product, int m, int n, int k) where TC : struct, ILaneChoice
    {
        var (vectors, rows, _, _) = ShapeOf(m, n, k);
        bool second = false;
        for (int c = 0; c < product.Count; c++)
        {
            second |= KindOf(product.Candidate(c)) == DecodedLaneId;
        }

        return string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described {product.Description.CpuFingerprint} {rows}x{k}->{n} {vectors}{(second ? " decoded" : "")}");
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

        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Rows, int N, int K), (int Lane, int Generation)> Chosen => described.Chosen;

        public int Count => Lanes.Count;

        public int Candidate(int index) => Lanes[index];

        public int Unmeasured => described.Unmeasured;

        public void Run(int laneId) => FormatProduct<DescribedFormat, DescribedKernels>(described, laneId, lane, x, y, m, n, k);
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
        FormatProduct<DescribedFormat, DescribedKernels>(described, described.Lanes[index], new StoredLane(words, length, k, RowBytesOf(description, k)), x, y, m, n, k);
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
            int chosen = Chosen(new LaneCandidates(described, codes ?? description, lane, x, y, m, n, k), m, n, k);
            parts[j] = new RowPart(lane, y, n, chosen);
        }

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

    // The stored rows a group of a tile's loop takes: those within `groupBytes`, at least one, at most MostGroupRows.
    private static int RowGroup(nint rowBytes, int columns, int groupBytes) =>
        (int)Math.Clamp(groupBytes / Math.Max(1, rowBytes), 1, Math.Clamp(columns, 1, MostGroupRows));

    // The tiles of `parts` in one range over the threads (CpuRanges, a tile a chunk): columns and rows of x, each part's
    // tiles after the other's; each output its own product's, whichever thread runs it. One row of x: columns in whole cache
    // lines of y (16 floats), so tiles on other threads do not write into the same lines as each output is written. More
    // rows: the format's tile (IRowFormat.Tile). Part p reads x as inputs[slots[p]].
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
        var (wanted, rows) = m == 1
            ? ((Math.Max(RowTileBytes / rowBytes, (total + 64L * threads - 1) / (64L * threads)) + 15) / 16 * 16, 1)
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
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
        var (floats, bytes) = format.Scratch(part.LaneId, k, i1 - i0 > 1 ? RowGroup(rowBytes, j1 - j0, format.GroupBytes) : 0);
        float[]? floatArray = floats > 0 ? ArrayPool<float>.Shared.Rent(floats) : null;
        byte[]? byteArray = bytes > 0 ? ArrayPool<byte>.Shared.Rent(bytes) : null;
        try
        {
            fixed (float* words = lane.Words)
            fixed (float* xv = input.Values)
            fixed (sbyte* xc = input.Codes)
            fixed (float* xs = input.Scales)
            fixed (float* scratch = floatArray)
            fixed (byte* scratchBytes = byteArray)
            {
                if (scratchBytes is not null)
                {
                    new Span<byte>(scratchBytes, bytes).Clear();
                }

                var kernels = format.Kernels(part.LaneId, k, new LanePointers(xv, xc, xs), scratch, scratchBytes);
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
            if (floatArray is not null)
            {
                ArrayPool<float>.Shared.Return(floatArray);
            }

            if (byteArray is not null)
            {
                ArrayPool<byte>.Shared.Return(byteArray);
            }
        }
    }

    // Columns [j0, j1) of y (n a row) for rows [i0, i1) of x (k a row) by `kernels` over the stored rows at `stored`. One row
    // of x: each stored row read once, in order. More: a group of stored rows (RowGroup, `groupBytes`) at a time, set up once
    // (IRowKernels.Group), then four rows of x at a time against each of the group's rows, the rows left over one at a time.
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

        int group = RowGroup(rowBytes, j1 - j0, groupBytes);
        float* sums = stackalloc float[4 * MostGroupRows];
        for (int c0 = j0; c0 < j1; c0 += group)
        {
            int count = Math.Min(group, j1 - c0);
            kernels.Group(stored, rowBytes, c0, count);
            int i = i0;
            for (; i + 4 <= i1; i += 4)
            {
                if (span >= k)
                {
                    // One pass over k: each output written as its four rows are done.
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Four(stored + (c0 + c) * rowBytes, c, i, 0, k, out float s0, out float s1, out float s2, out float s3);
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
                        kernels.Four(stored + (c0 + c) * rowBytes, c, i, p0, p1, out float s0, out float s1, out float s2, out float s3);
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
                    y[i * n + c0 + c] = kernels.One(stored + (c0 + c) * rowBytes, c, i);
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
    //   and the map are the kernels' type parameters (UniformOne, UniformRowSteps); a wide integer's byte planes are made by
    //   its plane count's own step (IStepLoad.Plane); a plane of a radix's digits by its runs' bytes times each run's power
    //   of the radix (DigitsOf); a codebook's entries by an index of up to 16 bits over any planes, each plane's field read
    //   from a word and the fields joined (Entry);
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
        // as bytes, then made unsigned (DescribedPlan.CodebookCodes).
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
        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Rows, int N, int K), (int Lane, int Generation)> Chosen { get; } = new();

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
        private int OneScale => oneScale;

        private int oneScale;

        // OneScale: a 16-bit float scale's reading as vectors, the unit folded in; a table scale's numbers times the unit.
        private HalfVectors oneHalf;
        private float* oneTable;

        /// <summary>Bytes a group's made steps take in the integer lane: its steps, its scales, then (minimums) its minimums.</summary>
        public int GroupStepBytes { get; private set; }

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

        /// <summary>The 16-bit float reading: its sign shift, magnitude mask, mantissa shift, the power of two it is times, and where its specials start (0: none).</summary>
        public (int SignShift, int Magnitude, int Shift, float Times, int Special) Half16 { get; private set; }

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
        // (and the shift and the left where every lane's is the same, else −1); per group, factor and lane the block and
        // index (one lane at a time).
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
        private int[] scalarBlock = [], scalarIndex = [];

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
            return plan;
        }

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
                    Map = CodeMap.Table;
                    byteTable = Pin<byte>(256);
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
            GroupStepBytes = P > 1 ? T * 32 : T * 32 + (Mins ? 64 : 32);
            PlaneMask0 = planeMask == null ? default : planeMask[0];
            MapVector = Map == CodeMap.Flip ? Flip : Shuffle;
            oneKernels = Pin<nint>(8);
            rowKernels = Pin<nint>(8);
            if (Mode == DescribedMode.Integer && P == 1 && UniformRuns == 0)
            {
                GatherSteps();                                              // false: the spans made and transposed (Steps)
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
            digitTimes = d.Planes.Any(p => p.Radix != 0) ? Pin<Vector256<byte>>(spans * Planes) : null;
            Order = Pin<int>(Period);
            int low = 0;
            for (int p = 0; p < Planes; p++)
            {
                planeMask[p] = Vector256.Create((byte)((1 << d.Planes[p].Bits) - 1));
                planeLow[p] = low;
                planeRadix[p] = d.Planes[p].Radix;
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
                }
            }

            Identity = Enumerable.Range(0, Period).All(v => Order[v] == v);
            Shuffles();
            return true;
        }

        // A span's run kind made from two windows of 16 bytes (Spans; Raw reads it by WindowLoad).
        private const byte WindowRuns = 17;

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
        // load), the least added back; any other codebook by CodebookAny.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> CodebookCodes(byte* period, int span)
        {
            if (!codebookFast)
            {
                return CodebookAny(period, span);
            }

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

        // Any codebook span's codes (entries of 1, 2, 4 or 8 values, vector factors of any group width and pieces).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
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
                    Span<byte> bytes = stackalloc byte[32];
                    for (int j = 0; j < per; j++)
                    {
                        new ReadOnlySpan<byte>(entryBytes + w * Entry(period, at + j), w).CopyTo(bytes.Slice(j * w, w));
                    }

                    v = Vector256.Create<byte>(bytes);
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
            scalarBlock = new int[groups * factors * 8];
            scalarIndex = new int[groups * factors * 8];
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
                            scalarBlock[(g * factors + f) * 8 + l] = start;
                            scalarIndex[(g * factors + f) * 8 + l] = index;
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
                    kinds |= (fetch << 4 | (int)reads[f]) << (8 * slot);
                }

                FactorKinds = kinds;
            }

            return PeriodBytes >= 4 || factors == 0;
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

        // The one-row integer kernel and the row steps' maker per instruction set (Instructions' value), resolved on first
        // use (CpuBackend.ResolveOne, ResolveRowSteps): 0 not yet, −1 none (their own dispatch reads the plan).
        private nint* oneKernels, rowKernels;

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

        /// <summary>The row steps' maker by TI (<paramref name="with"/>: its Instructions value), or null where RowStepsOf's dispatch reads the plan.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal delegate*<DescribedPlan, byte*, byte*, int, int, int, byte*, void> RowKernel<TI>(int with) where TI : struct, IInstructions
        {
            nint f = rowKernels[with];
            if (f == 0)
            {
                f = ResolveRowSteps<TI>(this);
                rowKernels[with] = f = f == 0 ? -1 : f;
            }

            return f == -1 ? null : (delegate*<DescribedPlan, byte*, byte*, int, int, int, byte*, void>)f;
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
                        v = StreamLoad(period, at, D.Planes[p].Bits);
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

        // A plane of digits' planes' radix (0: bits), and per span and plane each byte's power of the radix (its run's digit).
        private int* planeRadix;
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
                var spread = Vector256.Shuffle(Vector256.Create(four).AsByte(), BitBytes);
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

        // Byte j / 8 of 4 bytes repeated to byte j (each 128-bit half picks from its own copy of them, so the shuffle stays within
        // the halves: StreamLoad's one-bit fields).
        private static readonly Vector256<byte> BitBytes = Vector256.Create((byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19);

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

        // A span's codes one value at a time, in value order, by the description (a layout the runs do not take).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> OneByOne(byte* period, int span)
        {
            Span<byte> codes = stackalloc byte[32];
            for (int t = 0; t < 32; t++)
            {
                int v = 32 * span + t;
                codes[t] = (byte)D.CodeBits(new ReadOnlySpan<byte>(period + v / D.UnitValues * D.UnitBytes, D.UnitBytes), v % D.UnitValues);
            }

            return Vector256.Create<byte>(codes);
        }

        /// <summary>The integer lane's bytes of span <paramref name="span"/>'s codes: each its number in units less the least.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> Codes<TI>(byte* period, int span) where TI : struct, IInstructions
        {
            if (Map == CodeMap.Codebook)
            {
                return CodebookCodes(period, span);
            }

            var code = Raw<TI>(period, span);
            return Map switch
            {
                CodeMap.Flip => code ^ Flip,
                CodeMap.Shuffle => TI.ShuffleBytes(Shuffle, code),
                CodeMap.Table => Through(code),
                _ => code,
            };
        }

        // Each byte through the 256-entry table (codes of 5 to 8 bits whose numbers are not their bits).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> Through(Vector256<byte> code)
        {
            Span<byte> bytes = stackalloc byte[32];
            code.CopyTo(bytes);
            for (int i = 0; i < 32; i++)
            {
                bytes[i] = byteTable[bytes[i]];
            }

            return Vector256.Create<byte>(bytes);
        }

        /// <summary>
        /// Group <paramref name="gp"/> of the period at <paramref name="period"/> as the integer lane's steps (codes of at most 8
        /// bits: a wide integer's are made by IStepLoad.Plane): lane l of step s holds 4 bytes of chunk l, its positions
        /// 4s..4s + 3 in its span's order. T = 4: s4..s7 are not written.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Steps<TI>(byte* period, int gp, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions
        {
            if (stepsGathered)
            {
                GroupSteps<TI>(period, gp, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                return;
            }

            int span = gp * (GroupValues / 32);
            if (C == 32)
            {
                Transpose8<TI>(Codes<TI>(period, span), Codes<TI>(period, span + 1), Codes<TI>(period, span + 2), Codes<TI>(period, span + 3), Codes<TI>(period, span + 4),
                    Codes<TI>(period, span + 5), Codes<TI>(period, span + 6), Codes<TI>(period, span + 7), out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                return;
            }

            Transpose4<TI>(Codes<TI>(period, span), Codes<TI>(period, span + 1), Codes<TI>(period, span + 2), Codes<TI>(period, span + 3), out s0, out s1, out s2, out s3);
            s4 = s5 = s6 = s7 = default;
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
        private Vector256<ushort>* stepTimes;
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

            stepRead = new StepRead[Planes];
            stepMask = new Vector256<uint>[Planes];
            stepField = new Vector256<uint>[Planes];
            stepOffset = Pin<int>(8 * steps * Planes);
            stepShift = Pin<Vector256<uint>>(steps * Planes);
            stepTimes = Pin<Vector256<ushort>>(steps * Planes);
            Span<int> offsets = stackalloc int[8];
            Span<uint> shifts = stackalloc uint[8];
            Span<ushort> times = stackalloc ushort[16];
            Span<int> current = stackalloc int[8];
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
                for (int s = 0; s < steps; s++)
                {
                    int i = s * Planes + p;
                    for (int l = 0; l < 8; l++)
                    {
                        if (!StepPlace(plane, s, l, stepRead[p], out offsets[l], out int shift))
                        {
                            return false;
                        }

                        // An index's or a stream's fields from the dword an earlier step of the group reads where they fit it.
                        if (stepRead[p] is StepRead.Index or StepRead.Stream && !Keep(current, l, s % T == 0, 8L * offsets[l] + shift, width, out offsets[l], out shift))
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

                    offsets.CopyTo(new Span<int>(stepOffset + 8 * i, 8));
                    stepShift[i] = Vector256.Create<uint>(shifts);
                    stepTimes[i] = Vector256.Create<ushort>(times);
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

                            if (!Keep(factorCurrent.Slice(8 * f, 8), l, s % T == 0, bit, bitsOf, out int kept, out int shift))
                            {
                                return false;
                            }

                            (offsets[l], shifts[l], within[l]) = (kept, (uint)shift, v % vectorWidth[f] / 4);
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
            for (int gp = 0; gp < groups; gp++)
            {
                known.Clear();
                int start = words.Count;
                for (int s = gp * T; s < (gp + 1) * T; s++)
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

        // Byte 0 of each dword to its 4 bytes (within each 128-bit half, as the byte shuffles take it).
        private static readonly Vector256<byte> LowBytes = Vector256.Create((byte)0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);

        // The gathered steps' dword vectors: each group's (from groupWordStart[gp], 8 offsets each in wordOffsets), and per
        // step and plane (and vector factor) the one it reads; the most a group gathers.
        private int* groupWordStart;
        private int* wordOffsets;
        private int* stepWord;
        private int* vectorWord;

        /// <summary>The most dword vectors a group of the gathered steps gathers (GatherSteps).</summary>
        public int MostGroupWords { get; private set; }

        /// <summary>Whether the integer lane's steps are gathered (GatherSteps: GatheredOne, GatheredRowSteps).</summary>
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
        /// Group <paramref name="gp"/>'s steps gathered (GatherSteps): its distinct dword vectors gathered once (on the stack,
        /// not cleared: each written before it is read), each step made from the ones it reads. T = 4: s4..s7 not written.
        /// </summary>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private void GroupSteps<TI>(byte* period, int gp, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7) where TI : struct, IInstructions
        {
            int start = groupWordStart[gp], count = groupWordStart[gp + 1] - start, first = gp * T;
            Vector256<uint>* words = stackalloc Vector256<uint>[Math.Max(1, count)];
            for (int u = 0; u < count; u++)
            {
                words[u] = TI.Gather(period, wordOffsets + 8 * (start + u)).AsUInt32();
            }

            s0 = GatheredStep<TI>(words, first);
            s1 = GatheredStep<TI>(words, first + 1);
            s2 = GatheredStep<TI>(words, first + 2);
            s3 = GatheredStep<TI>(words, first + 3);
            if (T == 8)
            {
                s4 = GatheredStep<TI>(words, first + 4);
                s5 = GatheredStep<TI>(words, first + 5);
                s6 = GatheredStep<TI>(words, first + 6);
                s7 = GatheredStep<TI>(words, first + 7);
                return;
            }

            s4 = s5 = s6 = s7 = default;
        }

        // The value at position 4s of step s's lane l (the step counted over the period: its group s / T; the lane's chunk).
        private int StepValue(int s, int l) => Order[((s / T) * Cg + l) * C + 4 * (s % T)];

        // Where step s's lane l of `plane` reads its 4 fields (positions 4s..4s + 3 of its chunk, in its span's order) by
        // `read`: 4 bytes' or digits' first byte and their shift (a digit plane: the digit); an index's or a stream's first
        // field's byte and bit in it (GatherSteps picks the dword it is read from); false where the 4 fields do not lie so.
        private bool StepPlace(CodePlane plane, int s, int l, StepRead read, out int offset, out int shift)
        {
            var d = D;
            (offset, shift) = (0, 0);
            int position = ((s / T) * Cg + l) * C + 4 * (s % T);
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
        /// (<paramref name="words"/>: GroupSteps): the integer lane's bytes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> GatheredStep<TI>(Vector256<uint>* words, int s) where TI : struct, IInstructions
        {
            // (Every constant from the plan's pinned arrays: no description, list or bounds check on this path.)
            Vector256<uint> code = default;
            for (int p = 0, i = s * Planes; p < Planes; p++, i++)
            {
                var word = TI.ShiftRight(words[stepWord[i]], stepShift[i]);
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
                        // Digits: each byte times its place's power of the radix, mod 256, times the radix, its top byte; even
                        // and odd bytes a 16-bit lane each.
                        var low = Vector256.Create(0x00FF00FFu);
                        var times = stepTimes[i];
                        var radix = Vector256.Create((ushort)stepBits[Planes + p]);
                        var even = Vector256.ShiftRightLogical(((word & low).AsUInt16() * times & low.AsUInt16()) * radix, 8);
                        var odd = Vector256.ShiftRightLogical(((Vector256.ShiftRightLogical(word, 8) & low).AsUInt16() * times & low.AsUInt16()) * radix, 8);
                        v = even.AsUInt32() | Vector256.ShiftLeft(odd.AsUInt32(), 8);
                        break;
                }

                code |= Vector256.ShiftLeft(v, stepLows[p]);
            }

            if (Map == CodeMap.Codebook)
            {
                return BookBytes<TI>(code, words, s);
            }

            var bytes = code.AsByte();
            return Map switch
            {
                CodeMap.Flip => bytes ^ Flip,
                CodeMap.Shuffle => TI.ShuffleBytes(Shuffle, bytes),
                CodeMap.Table => Through(bytes),
                _ => bytes,
            };
        }

        // Step s's codebook indices (`code`) as the integer lane's bytes: the index's entry dword (each lane's within its
        // entry), the vector factors' -1s applied, the least added back.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> BookBytes<TI>(Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions
        {
            var entries = TI.Gather((float*)entryBytes, code.AsInt32() * (entryWidth / 4) + bookWithin[s]).AsByte();
            for (int f = 0; f < vectorCount; f++)
            {
                int i = s * vectorCount + f;
                var index = TI.ShiftRight(words[vectorWord[i]], vectorShift[i]) & vectorIndexMasks[f];
                Vector256<byte> m;
                if (vectorFromBits[f] != 0)
                {
                    // The index's bits are the signs (bit j set: value j negative): this lane's 4 bits, its dword's low
                    // byte copied to its 4 bytes, each byte's own bit tested.
                    var bits = TI.ShiftRight(index, Vector256.ShiftLeft(vectorWithin[i], 2).AsUInt32()).AsByte();
                    var one = Vector256.Create(0x08040201u).AsByte();
                    m = Vector256.Equals(TI.ShuffleBytes(bits, LowBytes) & one, one);
                }
                else
                {
                    m = TI.Gather((float*)vectorMaskTables[f], index.AsInt32() * vectorDwords[f] + vectorWithin[i]).AsByte();
                }

                entries = (entries ^ m) - m;
            }

            return entries + leastBytes;
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

            static abstract Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p);
        }

        internal readonly struct NoPlane : IStepPlane
        {
            public static bool Present => false;

            public static Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p) => default;
        }

        internal readonly struct MaskedPlane : IStepPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p) => word & plan.stepMasks[p];
        }

        internal readonly struct BitPlane : IStepPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p) =>
                (word & Vector256.Create(0xFu)) * Vector256.Create(0x00204081u) & Vector256.Create(0x01010101u);
        }

        internal readonly struct StreamPlane : IStepPlane
        {
            public static bool Present => true;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p)
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

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<uint> Read(DescribedPlan plan, Vector256<uint> word, int i, int p)
            {
                var low = Vector256.Create(0x00FF00FFu);
                var times = plan.stepTimes[i];
                var radix = Vector256.Create((ushort)plan.stepBits[plan.Planes + p]);
                var even = Vector256.ShiftRightLogical(((word & low).AsUInt16() * times & low.AsUInt16()) * radix, 8);
                var odd = Vector256.ShiftRightLogical(((Vector256.ShiftRightLogical(word, 8) & low).AsUInt16() * times & low.AsUInt16()) * radix, 8);
                return even.AsUInt32() | Vector256.ShiftLeft(odd.AsUInt32(), 8);
            }
        }

        /// <summary>How a step's code becomes the integer lane's bytes (GatheredStep's map).</summary>
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
                plan.Through(code.AsByte());
        }

        internal readonly struct BookStepMap : IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Map<TI>(DescribedPlan plan, Vector256<uint> code, Vector256<uint>* words, int s) where TI : struct, IInstructions =>
                plan.BookBytes<TI>(code, words, s);
        }

        /// <summary>Step s of the period made from its group's gathered dword vectors (GatheredStep, or its readings as type parameters).</summary>
        internal interface IGatheredSteps
        {
            static abstract Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions;
        }

        internal readonly struct PlanSteps : IGatheredSteps
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions => plan.GatheredStep<TI>(words, s);
        }

        internal readonly struct PlaneSteps<TP0, TP1, TP2, TM> : IGatheredSteps
            where TP0 : struct, IStepPlane where TP1 : struct, IStepPlane where TP2 : struct, IStepPlane where TM : struct, IStepMap
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<byte> Step<TI>(DescribedPlan plan, Vector256<uint>* words, int s) where TI : struct, IInstructions
            {
                int i = s * plan.Planes;
                var code = Vector256.ShiftLeft(TP0.Read(plan, TI.ShiftRight(words[plan.stepWord[i]], plan.stepShift[i]), i, 0), plan.stepLows[0]);
                if (TP1.Present)
                {
                    code |= Vector256.ShiftLeft(TP1.Read(plan, TI.ShiftRight(words[plan.stepWord[i + 1]], plan.stepShift[i + 1]), i + 1, 1), plan.stepLows[1]);
                }

                if (TP2.Present)
                {
                    code |= Vector256.ShiftLeft(TP2.Read(plan, TI.ShiftRight(words[plan.stepWord[i + 2]], plan.stepShift[i + 2]), i + 2, 2), plan.stepLows[2]);
                }

                return TM.Map<TI>(plan, code, words, s);
            }
        }

        /// <summary>
        /// The scales (each lane's product of the scales covering its chunk, times the unit) and minimums (their product, less
        /// where they are subtracted) of group <paramref name="gp"/> of the period at <paramref name="period"/>, read for the 8
        /// lanes at once; lanes from <paramref name="valid"/> on 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Factor<TI>(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions
        {
            if (OneScale != 0)
            {
                // One scale: its field gathered and read with the unit folded in (a 16-bit float's constants as vectors made
                // once a plan, or a table of its numbers times the unit), no loop over factors.
                var bits = fUniform[gp] != 0 ? Vector256.Create((int)(Unsafe.ReadUnaligned<uint>(period + fOffset[8 * gp]) >> fShiftOne[gp]))
                    : Shifted<TI>(TI.Gather(period, fOffset + 8 * gp), gp);
                var one = OneScale == 1 ? oneHalf.Of(bits) : TI.Gather(oneTable, bits & fMask[gp]);
                scale = valid < 8 ? one & GroupMask(valid).AsSingle() : one;
                min = default;
                return;
            }

            var s = Vector256.Create(1f);
            var product = s;
            for (int f = 0; f < factors; f++)
            {
                var v = Read<TI>(period, gp, f);
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
        // in place), then read by its type.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> Read<TI>(byte* period, int gp, int f) where TI : struct, IInstructions
        {
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
                case FactorRead.Single:
                    return bits.AsSingle() + Vector256.Create(bias);
                default:
                    return ReadOneByOne(period, gp, f);
            }
        }

        // Factor f's numbers one lane at a time, by its type (a layout or a number the gathers do not take).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> ReadOneByOne(byte* period, int gp, int f)
        {
            var factor = factorList[f];
            Span<float> values = stackalloc float[8];
            for (int l = 0; l < 8; l++)
            {
                int i = (gp * factors + f) * 8 + l;
                values[l] = (float)factor.Value(new ReadOnlySpan<byte>(period + scalarBlock[i], PeriodBytes - scalarBlock[i]), scalarIndex[i]);
            }

            return Vector256.Create<float>(values);
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

    // Steps of the described lane's integer arithmetic a group of stored rows makes at most with more than one row of x (its
    // rows' steps, scales and minimums stay in a core's L2 while four rows of x pass over them; int2's prompt measured 634
    // tokens a second with 32 KB, 660 with 64 KB, 682 with 128 KB, 673 with 256 KB).
    private const int DescribedStepBytes = 131072;

    // Rows of x a tile takes at most where x is int8 codes or the stored values are 4 or more bytes each (each stored row
    // read from memory, and made into steps, once a prompt of that many rows); the float32 lane's 16-bit values take a group
    // of stored rows by RowTileRows rows.
    private const int DescribedTileRows = 128;

    // Stored bytes a group of a heavy type's rows (4 or more bytes a value) takes: the float64 lane's (its tile's x passes once
    // a group, 4 rows of x in L1 a span of k), the integer lane's over byte planes.
    private const int DescribedDoubleGroupBytes = 1 << 20, DescribedPlaneGroupBytes = 65536;

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
        public ConcurrentDictionary<(CpuVectorPaths Vectors, int Rows, int N, int K), (int Lane, int Generation)> Chosen => plan.Chosen;

        // The integer arithmetic's int8 dot products take AVX2's multiply-adds where its unsigned codes are at most 127 and
        // a byte plane each (DescribedPlan.PairsFit).
        public LaneList Lanes => plan.Mode == DescribedMode.Integer
            ? default(LaneList).WithKind(DescribedLaneId, dots: true, pairs: plan.P == 1 && plan.PairsFit)
            : DescribedKernels.Decodes(plan) ? default(LaneList).WithKind(DescribedLaneId).WithKind(DecodedLaneId) : default(LaneList).WithKind(DescribedLaneId);

        public int First => DescribedLaneId;

        public int Second => DescribedLaneId;

        public int Unmeasured => DescribedLaneId;

        public int InputOf(int lane) => 0;

        /// <summary>The plan's spans by how their codes are made (DescribedPlan.SpanKinds; tests).</summary>
        public (int Runs, int Windows, int Streams, int Codebook, int OneByOne) Spans() => plan.SpanKinds();

        /// <summary>The arithmetic the plan took: "integer", "float32" or "float64".</summary>
        public string Arithmetic => plan.Mode switch
        {
            DescribedMode.Integer => "integer",
            DescribedMode.Single => "float32",
            _ => "float64",
        };

        private bool Heavy => plan.Bytes >= 4 * plan.Values;

        // A row's bytes: its whole groups of blocks (a group's shared levels with them), as RowBytesOf counts them.
        private long RowBytes => RowBytesOf(plan.D, k);

        public int GroupBytes => plan.Mode switch
        {
            DescribedMode.Integer when plan.P > 1 => DescribedPlaneGroupBytes,
            DescribedMode.Integer => (int)Math.Min(int.MaxValue, RowBytes * Math.Max(1, DescribedStepBytes / Math.Max(1, plan.Groups(k) * plan.GroupStepBytes))),
            DescribedMode.Double => DescribedDoubleGroupBytes,
            _ => plan.Wide == WideRead.Codes ? ScaledGroupBytes : DescribedSingleGroupBytes,
        };

        // Integer arithmetic and codes: columns over up to DescribedTileRows rows of x, a quarter of a thread's share, so each
        // stored row is read (and made into steps) once a prompt of that many rows. (Groups of stored rows by 32 rows of x read
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
                    int groups = plan.Groups(k);
                    var codes = ArrayPool<sbyte>.Shared.Rent(m * groups * plan.GroupValues);
                    var lanes = ArrayPool<float>.Shared.Rent(m * groups * plan.LaneFloats);
                    if (InstructionsOf(lane) == Instructions.Portable)
                    {
                        RoundDescribedRows<PortableInstructions>(plan, x, codes, lanes, m, k);
                    }
                    else if (AdvSimd.Arm64.IsSupported)
                    {
                        RoundDescribedRows<NeonInstructions>(plan, x, codes, lanes, m, k);
                    }
                    else
                    {
                        RoundDescribedRows<Avx2Instructions>(plan, x, codes, lanes, m, k);
                    }
                    return new LaneInput(x, false, codes, lanes);
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

        // The integer lane's made steps of a group's rows, as floats (every byte written before it is read); a row's last
        // period where it is short of one, copied into zeros (its bytes cleared).
        // The float32 lane's 16-bit values: a group's stored rows made float32 once (DescribedKernels.Group), which every
        // four rows of x then read as float32.
        public (int Floats, int Bytes) Scratch(int lane, int k, int group) => plan.Mode switch
        {
            DescribedMode.Integer => (group * plan.Groups(k) * plan.GroupStepBytes / sizeof(float), plan.PeriodBytes),
            DescribedMode.Single when plan.Wide == WideRead.Codes => (0, plan.PeriodBytes),
            DescribedMode.Single when group > 0 && KindOf(lane) == DecodedLaneId => (group * k, 0),
            _ => (0, 0),
        };

        public DescribedKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) =>
            new(plan, InstructionsOf(lane), NarrowOf(lane), KindOf(lane) == DecodedLaneId, k, x, (byte*)floats, bytes);
    }

    // The described lane as RowGroups' kernels, each call by the lane's instruction set. Integer: a group's stored rows made
    // into steps, scales and minimums once (DescribedRowSteps), which every four rows of x then read; a row read alone makes
    // them in registers as it reads them. Float32 and float64: each value made in registers as the dot products read it,
    // nothing to set up.
    private readonly unsafe struct DescribedKernels : IRowKernels
    {
        private readonly DescribedPlan plan;
        private readonly Instructions with;
        private readonly bool narrow, decoded;
        private readonly int k, groups, full, tailBytes, rowSteps, chunks;
        private readonly nint xRow, lRow;
        private readonly float* x, lanes;
        private readonly sbyte* xq;
        private readonly byte* steps, tail;

        // `decodedKind`: the second kind (DecodedLaneId), a group's 16-bit rows made float32 once where its scratch is given.
        public DescribedKernels(DescribedPlan plan, Instructions with, bool narrow, bool decodedKind, int k, LanePointers input, byte* steps, byte* tail)
        {
            this.plan = plan;
            this.with = with;
            this.narrow = narrow;
            decoded = steps != null && decodedKind && Decodes(plan);
            this.k = k;
            this.steps = steps;
            this.tail = tail;
            x = input.Values;
            xq = input.Codes;
            lanes = input.Scales;
            groups = plan.Groups(k);
            full = k / plan.Period;
            // The bytes past the whole periods: whole groups of blocks with their shared levels (a block's bytes alone would
            // leave out a group's levels stored after its blocks, so the last period's scales read as zeros).
            tailBytes = (int)(RowBytesOf(plan.D, k) - (long)full * plan.PeriodBytes);
            rowSteps = groups * plan.GroupStepBytes;
            chunks = (k + plan.C - 1) / plan.C;
            // x's row: the integer layout's whole groups, or (float32 codes) the spans' order in whole groups.
            xRow = (nint)groups * plan.GroupValues;
            lRow = (nint)groups * plan.LaneFloats;
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

        /// <summary>Whether the float32 lane makes a group's stored values float32 once for four rows of x and more (16-bit values: made, not loaded).</summary>
        public static bool Decodes(DescribedPlan plan) => plan.Mode == DescribedMode.Single && plan.Wide is WideRead.Half or WideRead.Short;

        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            if (decoded)
            {
                // Each stored row's values made float32 into the scratch, a row of k after another (as each four rows of x
                // would make them: the same numbers).
                for (int c = 0; c < count; c++)
                {
                    DecodeSingleRow(plan, stored + (c0 + c) * rowBytes, (float*)steps + (nint)c * k, k);
                }

                return;
            }

            if (plan.Mode != DescribedMode.Integer)
            {
                return;
            }

            if (with == Instructions.Portable)
            {
                GroupBy<PortableInstructions>(stored, rowBytes, c0, count, Instructions.Portable);
            }
            else if (AdvSimd.Arm64.IsSupported)
            {
                GroupBy<NeonInstructions>(stored, rowBytes, c0, count, Instructions.Neon);
            }
            else
            {
                GroupBy<Avx2Instructions>(stored, rowBytes, c0, count, Instructions.Avx2);
            }
        }

        // Compiled optimized from its first call, so the shape's lane is measured on the code it then runs. The plan's maker
        // by its choices as type parameters, made once (DescribedPlan.RowKernel), where it has one.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void GroupBy<TI>(byte* stored, nint rowBytes, int c0, int count, Instructions by) where TI : struct, IInstructions
        {
            var make = plan.RowKernel<TI>((int)by);
            for (int c = 0; c < count; c++)
            {
                byte* row = stored + (c0 + c) * rowBytes;
                Tail(row);
                if (make != null)
                {
                    make(plan, row, tail, full, groups, chunks, steps + (nint)c * rowSteps);
                }
                else
                {
                    RowStepsOf<TI>(plan, row, tail, full, groups, chunks, steps + (nint)c * rowSteps);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (AdvSimd.Arm64.IsSupported)
            {
                switch (with)
                {
                    case Instructions.DotProd:
                        FourBy<DotProdInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    case Instructions.DotProdWide:
                        FourBy<WideDotProdInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    case Instructions.Neon:
                        FourBy<NeonInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                    default:
                        FourBy<PortableInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                        return;
                }
            }

            switch (with)
            {
                case Instructions.Vnni:
                    FourBy<VnniInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
                case Instructions.Avx2:
                    FourBy<Avx2Instructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
                default:
                    FourBy<PortableInstructions>(row, c, i, p0, p1, out y0, out y1, out y2, out y3);
                    return;
            }
        }

        // Four rows by the 256-bit kernels, or (the 128-bit twin) by the 128-bit ones.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void FourBy<TI>(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    DescribedIntegerFour<TI>(plan, narrow, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow, xRow, lRow, out y0, out y1, out y2, out y3);
                    return;
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

                    if (decoded)
                    {
                        // The group's row made float32 (Group): read as float32, its offset already added.
                        SingleFourBy<SingleRead, TI>(new SingleRead(), 4, 0f, narrow, (byte*)((float*)steps + (nint)c * k + p0), x + (nint)i * k + p0, k, p1 - p0,
                            out y0, out y1, out y2, out y3);
                        return;
                    }

                    DescribedSingleFour<TI>(plan, narrow, row + (nint)p0 * plan.Bytes, x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
                    return;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* row, int c, int i)
        {
            if (plan.Mode != DescribedMode.Integer)
            {
                return Single(row, i);
            }

            if (AdvSimd.Arm64.IsSupported)
            {
                return with switch
                {
                    Instructions.DotProd => DescribedIntegerSteps<DotProdInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                    Instructions.DotProdWide => DescribedIntegerSteps<WideDotProdInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                    Instructions.Neon => DescribedIntegerSteps<NeonInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                    _ => DescribedIntegerSteps<PortableInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                };
            }

            return with switch
            {
                Instructions.Vnni => DescribedIntegerSteps<VnniInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                Instructions.Avx2 => DescribedIntegerSteps<Avx2Instructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
                _ => DescribedIntegerSteps<PortableInstructions>(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow),
            };
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
                    // The plan's kernel by its choices as type parameters, made once (DescribedPlan.OneKernel).
                    var one = plan.OneKernel<TI>((int)with);
                    return one != null ? one(plan, row, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow)
                        : DescribedIntegerOne<TI>(plan, row, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow);
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
    }

    // x rounded to int8 for the described lane's integer arithmetic, row by row (RoundDescribedRow), over the threads where
    // x is large (a row a chunk, CpuRanges: no closure or task allocated a call, as the rows alone ran before).
    private static void RoundDescribedRows<TI>(DescribedPlan plan, float[] x, sbyte[] codes, float[] lanes, int m, int k) where TI : struct, IInstructions
    {
        var rows = new RoundedRows<TI>(plan, x, codes, lanes, k);
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            CpuRanges.Run(m, 1, m, Math.Max(1, ComputeResources.MaxCpuThreads), ref rows);   // `rows` is this frame's, on the stack
            return;
        }

        rows.Run(0, m);
    }

    // RoundDescribedRows' rows [first, last), each rounded on its own (the same bytes whichever thread rounds it).
    private readonly struct RoundedRows<TI>(DescribedPlan plan, float[] x, sbyte[] codes, float[] lanes, int k) : IRangeBody where TI : struct, IInstructions
    {
        private readonly int xRow = plan.Groups(k) * plan.GroupValues, lRow = plan.Groups(k) * plan.LaneFloats;

        public void Run(int first, int last)
        {
            for (int i = first; i < last; i++)
            {
                RoundDescribedRow<TI>(plan, x.AsSpan(i * k, k), codes.AsSpan(i * xRow, xRow), lanes.AsSpan(i * lRow, lRow));
            }
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time (the block's scale amax / 127, 0 where every value is 0; its
    // codes value · 127 / amax rounded to even), laid out as the plan's steps are (XPlace): each block put in its span's
    // order by two byte shuffles (OrderShuffles; none where spans are in value order), a group's blocks transposed as the
    // stored codes are (one chunk a lane of each step; a wide integer's chunks 4 bytes each of a step's 4 · Cg); per group
    // LaneFloats floats: each lane's block scale, as int32 bits its plane's least times its chunk's code sum, and (minimums)
    // its block scale times its chunk's code sum (in plane 0's lanes). Blocks past k are zero codes, scales and sums. The
    // integer lane's product is then the product of x so rounded. Made by the instruction set of the lane that prepares it
    // (values moved and integers summed: the same bytes by any set, so the lanes of every set share it).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RoundDescribedRow<TI>(DescribedPlan plan, ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> lanes) where TI : struct, IInstructions
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

    // A stored row's integer steps, scales and minimums (DescribedPlan.Steps and Factor), a group after another, into `into`
    // (GroupStepBytes a group); its last period from `tail` where it is short.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedRowSteps<TI, TF>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
        where TI : struct, IInstructions where TF : struct, IFactors
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, stepBytes = plan.GroupStepBytes;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            byte* to = into + (nint)g * stepBytes;
            plan.Steps<TI>(at, gp, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
            s0.Store(to);
            s1.Store(to + 32);
            s2.Store(to + 64);
            s3.Store(to + 96);
            if (t == 8)
            {
                s4.Store(to + 128);
                s5.Store(to + 160);
                s6.Store(to + 192);
                s7.Store(to + 224);
            }

            TF.Read<TI>(plan, at, gp, Math.Min(plan.Cg, chunks - g * plan.Cg) * plan.P, out var scale, out var min);
            scale.Store((float*)(to + 32 * t));
            if (plan.Mins)
            {
                min.Store((float*)(to + 32 * t + 32));
            }
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

    // A wide integer's (P > 1) stored row made into the integer lane's steps (8 a group, 32 bytes each, no scales: x's block
    // scales and the planes' weights apply), its last period from `tail`; TX makes each step.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void WideRowSteps<TX, TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, byte* into)
        where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        int periodBytes = plan.PeriodBytes;
        var flip = plan.TopFlip;
        for (int g = 0; g < groups; g++)
        {
            byte* at = g < full ? row + (nint)g * periodBytes : tail;
            byte* to = into + (nint)g * 256;
            (TX.Plane<TI>(at, 0) ^ flip).Store(to);
            (TX.Plane<TI>(at, 1) ^ flip).Store(to + 32);
            (TX.Plane<TI>(at, 2) ^ flip).Store(to + 64);
            (TX.Plane<TI>(at, 3) ^ flip).Store(to + 96);
            (TX.Plane<TI>(at, 4) ^ flip).Store(to + 128);
            (TX.Plane<TI>(at, 5) ^ flip).Store(to + 160);
            (TX.Plane<TI>(at, 6) ^ flip).Store(to + 192);
            (TX.Plane<TI>(at, 7) ^ flip).Store(to + 224);
        }
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

    // A wide integer's one row of x by a stored row made into steps (WideRowSteps).
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float WideStepsOne<TX, TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        int groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var top = plan.TopLanes;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * 256;
            f = WideRow<TX, TI>(f, Vector256.Load(at), Vector256.Load(at + 32), Vector256.Load(at + 64), Vector256.Load(at + 96), Vector256.Load(at + 128),
                Vector256.Load(at + 160), Vector256.Load(at + 192), Vector256.Load(at + 224), x + (nint)g * groupValues, xStep, lanes + (nint)g * laneFloats, cg, top);
        }

        return Vector256.Sum(f * plan.LaneWeights!.Value);
    }

    // The described integer lane, four rows of x by one stored row made into steps (DescribedRowSteps), dispatched on the
    // step's x load (the plan's byte planes); by the 128-bit kernels where `narrow` (the 128-bit twin).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedIntegerFour<TI>(DescribedPlan plan, bool narrow, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        if (narrow)
        {
            switch (plan.P)
            {
                case 1:
                    ChunkStepsFour128<TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                    break;
                case 2:
                    StepsFour128<FourChunks, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                    break;
                case 4:
                    StepsFour128<TwoChunks, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                    break;
                default:
                    StepsFour128<OneChunk, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                    break;
            }

            return;
        }

        switch (plan.P)
        {
            case 1:
                ChunkStepsFour<TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            case 2:
                StepsFour<FourChunks, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            case 4:
                StepsFour<TwoChunks, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            default:
                StepsFour<OneChunk, TI>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
        }
    }

    // Four rows of x (codes from `x`, a row every `xRow`; lane data from `lanes`, a row every `lRow`) by one stored row's steps
    // of a wide integer (P > 1, `groups` groups of GroupStepBytes, no scales): each step loaded once and dotted with each
    // row's step of x, a lane's sum from x's start (the top plane's least times its chunk's code sum); then times x's block
    // scales (spread to the chunks' lanes as x's codes are), and the lanes weighted by their planes at the end.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void StepsFour<TX, TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            // Each row by itself (the same operations in the same order: WideStepsOne), never inlined, so compiled on its own.
            y0 = WideStepsOne<TX, TI>(plan, steps, groups, x, lanes);
            y1 = WideStepsOne<TX, TI>(plan, steps, groups, x + xRow, lanes + lRow);
            y2 = WideStepsOne<TX, TI>(plan, steps, groups, x + 2 * xRow, lanes + 2 * lRow);
            y3 = WideStepsOne<TX, TI>(plan, steps, groups, x + 3 * xRow, lanes + 3 * lRow);
            return;
        }

        int groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var top = plan.TopLanes;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * 256;
            var s0 = Vector256.Load(at);
            var s1 = Vector256.Load(at + 32);
            var s2 = Vector256.Load(at + 64);
            var s3 = Vector256.Load(at + 96);
            var s4 = Vector256.Load(at + 128);
            var s5 = Vector256.Load(at + 160);
            var s6 = Vector256.Load(at + 192);
            var s7 = Vector256.Load(at + 224);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0 = WideRow<TX, TI>(f0, s0, s1, s2, s3, s4, s5, s6, s7, xg, xStep, lg, cg, top);
            f1 = WideRow<TX, TI>(f1, s0, s1, s2, s3, s4, s5, s6, s7, xg + xRow, xStep, lg + lRow, cg, top);
            f2 = WideRow<TX, TI>(f2, s0, s1, s2, s3, s4, s5, s6, s7, xg + 2 * xRow, xStep, lg + 2 * lRow, cg, top);
            f3 = WideRow<TX, TI>(f3, s0, s1, s2, s3, s4, s5, s6, s7, xg + 3 * xRow, xStep, lg + 3 * lRow, cg, top);
        }

        var w = plan.LaneWeights!.Value;
        y0 = Vector256.Sum(f0 * w);
        y1 = Vector256.Sum(f1 * w);
        y2 = Vector256.Sum(f2 * w);
        y3 = Vector256.Sum(f3 * w);
    }

    // StepsFour on 128-bit vectors (its 128-bit twin): each step's halves (`l`: lanes 0-3, `u`: lanes 4-7) loaded once and
    // dotted with each row's matching half of x's step (IStepLoad.Half), the lanes' starts, scales and weights a half each.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void StepsFour128<TX, TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        int groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var top = plan.TopLanes;
        var weights = plan.LaneWeights!.Value;
        Vector128<int> topl = top.GetLower(), topu = top.GetUpper();
        Vector128<float> wl = weights.GetLower(), wu = weights.GetUpper();
        Vector128<float> f0l = default, f0u = default, f1l = default, f1u = default, f2l = default, f2u = default, f3l = default, f3u = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * 256;
            var s0l = Vector128.Load(at);
            var s0u = Vector128.Load(at + 16);
            var s1l = Vector128.Load(at + 32);
            var s1u = Vector128.Load(at + 48);
            var s2l = Vector128.Load(at + 64);
            var s2u = Vector128.Load(at + 80);
            var s3l = Vector128.Load(at + 96);
            var s3u = Vector128.Load(at + 112);
            var s4l = Vector128.Load(at + 128);
            var s4u = Vector128.Load(at + 144);
            var s5l = Vector128.Load(at + 160);
            var s5u = Vector128.Load(at + 176);
            var s6l = Vector128.Load(at + 192);
            var s6u = Vector128.Load(at + 208);
            var s7l = Vector128.Load(at + 224);
            var s7u = Vector128.Load(at + 240);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0l = WideRow128<TX, TI>(f0l, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, xg, xStep, lg, cg, topl, false);
            f0u = WideRow128<TX, TI>(f0u, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, xg, xStep, lg, cg, topu, true);
            f1l = WideRow128<TX, TI>(f1l, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, xg + xRow, xStep, lg + lRow, cg, topl, false);
            f1u = WideRow128<TX, TI>(f1u, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, xg + xRow, xStep, lg + lRow, cg, topu, true);
            f2l = WideRow128<TX, TI>(f2l, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, xg + 2 * xRow, xStep, lg + 2 * lRow, cg, topl, false);
            f2u = WideRow128<TX, TI>(f2u, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, xg + 2 * xRow, xStep, lg + 2 * lRow, cg, topu, true);
            f3l = WideRow128<TX, TI>(f3l, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, xg + 3 * xRow, xStep, lg + 3 * lRow, cg, topl, false);
            f3u = WideRow128<TX, TI>(f3u, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, xg + 3 * xRow, xStep, lg + 3 * lRow, cg, topu, true);
        }

        y0 = SumOf(f0l * wl, f0u * wu);
        y1 = SumOf(f1l * wl, f1u * wu);
        y2 = SumOf(f2l * wl, f2u * wu);
        y3 = SumOf(f3l * wl, f3u * wu);
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

    // StepsFour for codes of at most 8 bits (P = 1: a chunk a lane, x's steps 32 bytes): a group's steps (8, or 4 for chunks
    // of 16) loaded once and held while each of the four rows of x is dotted with them in turn.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ChunkStepsFour<TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            // Each row by itself (the same operations in the same order: StepsOne), never inlined, so compiled on its own.
            y0 = StepsOne<EightChunks, TI>(plan, steps, groups, x, lanes);
            y1 = StepsOne<EightChunks, TI>(plan, steps, groups, x + xRow, lanes + lRow);
            y2 = StepsOne<EightChunks, TI>(plan, steps, groups, x + 2 * xRow, lanes + 2 * lRow);
            y3 = StepsOne<EightChunks, TI>(plan, steps, groups, x + 3 * xRow, lanes + 3 * lRow);
            return;
        }

        int t = plan.T, stepBytes = plan.GroupStepBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats;
        bool mins = plan.Mins, eight = t == 8;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * stepBytes;
            var s0 = Vector256.Load(at);
            var s1 = Vector256.Load(at + 32);
            var s2 = Vector256.Load(at + 64);
            var s3 = Vector256.Load(at + 96);
            Vector256<byte> s4 = default, s5 = default, s6 = default, s7 = default;
            if (eight)
            {
                s4 = Vector256.Load(at + 128);
                s5 = Vector256.Load(at + 160);
                s6 = Vector256.Load(at + 192);
                s7 = Vector256.Load(at + 224);
            }

            var scale = Vector256.Load((float*)(at + 32 * t));
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0 = ChunkRow<TI>(f0, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg, lg);
            f1 = ChunkRow<TI>(f1, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + xRow, lg + lRow);
            f2 = ChunkRow<TI>(f2, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + 2 * xRow, lg + 2 * lRow);
            f3 = ChunkRow<TI>(f3, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + 3 * xRow, lg + 3 * lRow);
            if (mins)
            {
                var min = Vector256.Load((float*)(at + 32 * t + 32));
                f0 = TI.Fused(min, Vector256.Load(lg + 16), f0);
                f1 = TI.Fused(min, Vector256.Load(lg + lRow + 16), f1);
                f2 = TI.Fused(min, Vector256.Load(lg + 2 * lRow + 16), f2);
                f3 = TI.Fused(min, Vector256.Load(lg + 3 * lRow + 16), f3);
            }
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // ChunkStepsFour on 128-bit vectors (its 128-bit twin): a group's steps' halves (`l`: chunks 0-3, `u`: chunks 4-7) loaded
    // once and held while each row of x is dotted with them, its scales and minimums a half each.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ChunkStepsFour128<TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        int t = plan.T, stepBytes = plan.GroupStepBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats;
        bool mins = plan.Mins, eight = t == 8;
        Vector128<float> f0l = default, f0u = default, f1l = default, f1u = default, f2l = default, f2u = default, f3l = default, f3u = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * stepBytes;
            var s0l = Vector128.Load(at);
            var s0u = Vector128.Load(at + 16);
            var s1l = Vector128.Load(at + 32);
            var s1u = Vector128.Load(at + 48);
            var s2l = Vector128.Load(at + 64);
            var s2u = Vector128.Load(at + 80);
            var s3l = Vector128.Load(at + 96);
            var s3u = Vector128.Load(at + 112);
            Vector128<byte> s4l = default, s4u = default, s5l = default, s5u = default, s6l = default, s6u = default, s7l = default, s7u = default;
            if (eight)
            {
                s4l = Vector128.Load(at + 128);
                s4u = Vector128.Load(at + 144);
                s5l = Vector128.Load(at + 160);
                s5u = Vector128.Load(at + 176);
                s6l = Vector128.Load(at + 192);
                s6u = Vector128.Load(at + 208);
                s7l = Vector128.Load(at + 224);
                s7u = Vector128.Load(at + 240);
            }

            var scalel = Vector128.Load((float*)(at + 32 * t));
            var scaleu = Vector128.Load((float*)(at + 32 * t + 16));
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0l = ChunkRow128<TI>(f0l, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, scalel, xg, lg);
            f0u = ChunkRow128<TI>(f0u, eight, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scaleu, xg + 16, lg + 4);
            f1l = ChunkRow128<TI>(f1l, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, scalel, xg + xRow, lg + lRow);
            f1u = ChunkRow128<TI>(f1u, eight, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scaleu, xg + xRow + 16, lg + lRow + 4);
            f2l = ChunkRow128<TI>(f2l, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, scalel, xg + 2 * xRow, lg + 2 * lRow);
            f2u = ChunkRow128<TI>(f2u, eight, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scaleu, xg + 2 * xRow + 16, lg + 2 * lRow + 4);
            f3l = ChunkRow128<TI>(f3l, eight, s0l, s1l, s2l, s3l, s4l, s5l, s6l, s7l, scalel, xg + 3 * xRow, lg + 3 * lRow);
            f3u = ChunkRow128<TI>(f3u, eight, s0u, s1u, s2u, s3u, s4u, s5u, s6u, s7u, scaleu, xg + 3 * xRow + 16, lg + 3 * lRow + 4);
            if (mins)
            {
                var minl = Vector128.Load((float*)(at + 32 * t + 32));
                var minu = Vector128.Load((float*)(at + 32 * t + 48));
                f0l = TI.Fused(minl, Vector128.Load(lg + 16), f0l);
                f0u = TI.Fused(minu, Vector128.Load(lg + 20), f0u);
                f1l = TI.Fused(minl, Vector128.Load(lg + lRow + 16), f1l);
                f1u = TI.Fused(minu, Vector128.Load(lg + lRow + 20), f1u);
                f2l = TI.Fused(minl, Vector128.Load(lg + 2 * lRow + 16), f2l);
                f2u = TI.Fused(minu, Vector128.Load(lg + 2 * lRow + 20), f2u);
                f3l = TI.Fused(minl, Vector128.Load(lg + 3 * lRow + 16), f3l);
                f3u = TI.Fused(minu, Vector128.Load(lg + 3 * lRow + 20), f3u);
            }
        }

        y0 = SumOf(f0l, f0u);
        y1 = SumOf(f1l, f1u);
        y2 = SumOf(f2l, f2u);
        y3 = SumOf(f3l, f3u);
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

    // StepsFour for one row of x (the rows of a group left over after its fours).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedIntegerSteps<TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) where TI : struct, IInstructions => plan.P switch
    {
        1 => StepsOne<EightChunks, TI>(plan, steps, groups, x, lanes),
        2 => WideStepsOne<FourChunks, TI>(plan, steps, groups, x, lanes),
        4 => WideStepsOne<TwoChunks, TI>(plan, steps, groups, x, lanes),
        _ => WideStepsOne<OneChunk, TI>(plan, steps, groups, x, lanes),
    };

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float StepsOne<TX, TI>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) where TX : struct, IStepLoad where TI : struct, IInstructions
    {
        int t = plan.T, stepBytes = plan.GroupStepBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, xStep = 4 * plan.Cg;
        bool mins = plan.Mins;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * stepBytes;
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var xs = Vector256.Load(lg);
            var i = Vector256.Load((int*)(lg + 8));
            for (int s = 0; s < t; s++)
            {
                i = TI.Dot(i, Vector256.Load(at + 32 * s), TX.Load<TI>(xg + s * xStep));
            }

            f = TI.Fused(Vector256.ConvertToSingle(i), Vector256.Load((float*)(at + 32 * t)) * xs, f);
            if (mins)
            {
                f = TI.Fused(Vector256.Load((float*)(at + 32 * t + 32)), Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(plan.LaneWeights is { } w ? f * w : f);
    }

    // The described integer lane for one row of x read alone (one row of x, or a tile's one row): each group's steps,
    // scales and minimums made in registers from the stored row (its last period from `tail`) as they are dotted with x;
    // a plan whose spans are one plane loaded the same way takes its load and map as type parameters (UniformOne).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedIntegerOne<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TI : struct, IInstructions
    {
        if (plan.P == 1 && plan.UniformRuns != 0)
        {
            return plan.UniformRuns switch
            {
                1 => UniformOneBy<OneRun, TI>(plan, row, tail, full, groups, chunks, x, lanes),
                2 => UniformOneBy<TwoRuns, TI>(plan, row, tail, full, groups, chunks, x, lanes),
                4 => UniformOneBy<FourRuns, TI>(plan, row, tail, full, groups, chunks, x, lanes),
                _ => UniformOneBy<EightRuns, TI>(plan, row, tail, full, groups, chunks, x, lanes),
            };
        }

        if (plan.StepsGathered)
        {
            return GatheredOne<TI, FactorLoop, DescribedPlan.PlanSteps>(plan, row, tail, full, groups, chunks, x, lanes);
        }

        return plan.P switch
        {
            1 => MadeOne<EightChunks, TI, FactorLoop>(plan, row, tail, full, groups, chunks, x, lanes),
            2 => WideMadeOne<FourChunks, TI>(plan, row, tail, full, groups, x, lanes),
            4 => WideMadeOne<TwoChunks, TI>(plan, row, tail, full, groups, x, lanes),
            _ => WideMadeOne<OneChunk, TI>(plan, row, tail, full, groups, x, lanes),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float UniformOneBy<TR, TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TR : struct, IRunLoad where TI : struct, IInstructions => plan.Map switch
        {
            CodeMap.Same => UniformOne<TR, SameCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, x, lanes),
            CodeMap.Flip => UniformOne<TR, FlippedCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, x, lanes),
            _ => UniformOne<TR, ShuffledCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, x, lanes),
        };

    // A stored row's integer steps (DescribedRowSteps), by the plan's uniform load and map as type parameters where it has them.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowStepsOf<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into) where TI : struct, IInstructions
    {
        switch (plan.P)
        {
            case 2:
                WideRowSteps<FourChunks, TI>(plan, row, tail, full, groups, into);
                return;
            case 4:
                WideRowSteps<TwoChunks, TI>(plan, row, tail, full, groups, into);
                return;
            case 8:
                WideRowSteps<OneChunk, TI>(plan, row, tail, full, groups, into);
                return;
        }

        if (plan.UniformRuns == 0)
        {
            if (plan.StepsGathered)
            {
                GatheredRowSteps<TI, FactorLoop, DescribedPlan.PlanSteps>(plan, row, tail, full, groups, chunks, into);
                return;
            }

            DescribedRowSteps<TI, FactorLoop>(plan, row, tail, full, groups, chunks, into);
            return;
        }

        switch (plan.UniformRuns)
        {
            case 1:
                RowStepsBy<OneRun, TI>(plan, row, tail, full, groups, chunks, into);
                break;
            case 2:
                RowStepsBy<TwoRuns, TI>(plan, row, tail, full, groups, chunks, into);
                break;
            case 4:
                RowStepsBy<FourRuns, TI>(plan, row, tail, full, groups, chunks, into);
                break;
            default:
                RowStepsBy<EightRuns, TI>(plan, row, tail, full, groups, chunks, into);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowStepsBy<TR, TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
        where TR : struct, IRunLoad where TI : struct, IInstructions
    {
        switch (plan.Map)
        {
            case CodeMap.Same:
                UniformRowSteps<TR, SameCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, into);
                break;
            case CodeMap.Flip:
                UniformRowSteps<TR, FlippedCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, into);
                break;
            default:
                UniformRowSteps<TR, ShuffledCodes, TI, FactorLoop>(plan, row, tail, full, groups, chunks, into);
                break;
        }
    }

    /// <summary>
    /// How the integer lane's group loop reads a group's scales and minimums: the plan's loop over its factors
    /// (FactorLoop), or the plan's factors as type parameters (DescribedPlan.FactorKinds: SlotFactors), each group then
    /// straight-line code. The same numbers either way.
    /// </summary>
    private unsafe interface IFactors
    {
        static abstract void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions;
    }

    private readonly unsafe struct FactorLoop : IFactors
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Read<TI>(DescribedPlan plan, byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min) where TI : struct, IInstructions =>
            plan.Factor<TI>(period, gp, valid, out scale, out min);
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
    // kernel (DescribedPlan.OneKernel, RowKernel), never a call.
    private static nint WithFactors<TU>(DescribedPlan plan) where TU : struct, IFactorsUse =>
        plan.FactorKinds == 0 ? TU.Use<FactorLoop>() : FactorsFirst<TU>(plan.FactorKinds);

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
            3 => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.HalfNumber>>>(),
            _ => TU.Use<SlotFactors<T0, T1, T2, DescribedPlan.Field<TB, DescribedPlan.SingleNumber>>>(),
        };

    // The kernels WithFactors hands a plan's factors to: the one-row kernels, the row steps' makers.
    private readonly unsafe struct UniformOneUse<TR, TM, TI> : IFactorsUse where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &UniformOne<TR, TM, TI, TF>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct GatheredOneUse<TI, TS> : IFactorsUse where TI : struct, IInstructions where TS : struct, DescribedPlan.IGatheredSteps
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &GatheredOne<TI, TF, TS>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct MadeOneUse<TI> : IFactorsUse where TI : struct, IInstructions
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, sbyte*, float*, float> kernel = &MadeOne<EightChunks, TI, TF>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct UniformRowStepsUse<TR, TM, TI> : IFactorsUse where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, byte*, void> kernel = &UniformRowSteps<TR, TM, TI, TF>;
            return (nint)kernel;
        }
    }

    private readonly unsafe struct GatheredRowStepsUse<TI, TS> : IFactorsUse where TI : struct, IInstructions where TS : struct, DescribedPlan.IGatheredSteps
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, byte*, void> kernel = &GatheredRowSteps<TI, TF, TS>;
            return (nint)kernel;
        }
    }

    /// <summary>A gathered kernel's address with its steps' readings as the type parameter (WithSteps' last step), its factors then chosen.</summary>
    private interface IStepsUse
    {
        static abstract nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps;
    }

    private readonly struct GatheredOneSteps<TI> : IStepsUse where TI : struct, IInstructions
    {
        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps => WithFactors<GatheredOneUse<TI, TS>>(plan);
    }

    private readonly struct GatheredRowStepsSteps<TI> : IStepsUse where TI : struct, IInstructions
    {
        public static nint Use<TS>(DescribedPlan plan) where TS : struct, DescribedPlan.IGatheredSteps => WithFactors<GatheredRowStepsUse<TI, TS>>(plan);
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
            CodeMap.Same => TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, DescribedPlan.SameStepMap>>(plan),
            CodeMap.Flip => TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, DescribedPlan.FlipStepMap>>(plan),
            CodeMap.Shuffle => TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, DescribedPlan.ShuffleStepMap>>(plan),
            CodeMap.Table => TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, DescribedPlan.TableStepMap>>(plan),
            _ => TU.Use<DescribedPlan.PlaneSteps<TP0, TP1, TP2, DescribedPlan.BookStepMap>>(plan),
        };

    private readonly unsafe struct DescribedRowStepsUse<TI> : IFactorsUse where TI : struct, IInstructions
    {
        public static nint Use<TF>() where TF : struct, IFactors
        {
            delegate*<DescribedPlan, byte*, byte*, int, int, int, byte*, void> kernel = &DescribedRowSteps<TI, TF>;
            return (nint)kernel;
        }
    }

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

        return plan.StepsGathered ? WithSteps<GatheredOneSteps<TI>>(plan) : WithFactors<MadeOneUse<TI>>(plan);
    }

    private static nint OneByMap<TR, TI>(DescribedPlan plan) where TR : struct, IRunLoad where TI : struct, IInstructions => plan.Map switch
    {
        CodeMap.Same => WithFactors<UniformOneUse<TR, SameCodes, TI>>(plan),
        CodeMap.Flip => WithFactors<UniformOneUse<TR, FlippedCodes, TI>>(plan),
        _ => WithFactors<UniformOneUse<TR, ShuffledCodes, TI>>(plan),
    };

    // The row steps' maker of a plan by TI (RowStepsOf's choice, and the factors'), made once a plan and instruction set
    // (DescribedPlan.RowKernel); 0 for byte planes of wide integers.
    private static nint ResolveRowSteps<TI>(DescribedPlan plan) where TI : struct, IInstructions
    {
        if (plan.P != 1)
        {
            return 0;
        }

        if (plan.UniformRuns != 0)
        {
            return plan.UniformRuns switch
            {
                1 => RowStepsByMap<OneRun, TI>(plan),
                2 => RowStepsByMap<TwoRuns, TI>(plan),
                4 => RowStepsByMap<FourRuns, TI>(plan),
                _ => RowStepsByMap<EightRuns, TI>(plan),
            };
        }

        return plan.StepsGathered ? WithSteps<GatheredRowStepsSteps<TI>>(plan) : WithFactors<DescribedRowStepsUse<TI>>(plan);
    }

    private static nint RowStepsByMap<TR, TI>(DescribedPlan plan) where TR : struct, IRunLoad where TI : struct, IInstructions => plan.Map switch
    {
        CodeMap.Same => WithFactors<UniformRowStepsUse<TR, SameCodes, TI>>(plan),
        CodeMap.Flip => WithFactors<UniformRowStepsUse<TR, FlippedCodes, TI>>(plan),
        _ => WithFactors<UniformRowStepsUse<TR, ShuffledCodes, TI>>(plan),
    };

    /// <summary>
    /// How the described lane loads a span's run of its one code plane and shifts it (every span the same:
    /// DescribedPlan.UniformRuns): span s by each lane's shift (shifts[s]) where the instruction set shifts lanes by their
    /// own counts in one instruction (IInstructions.ShiftsByLane), else by the shifts as numbers (RunCounts' 8 of span s).
    /// The arrays and s are passed, not the span's entries, so the branch not taken reads nothing (the taken one's shift
    /// read in place, within the shift instruction).
    /// </summary>
    private unsafe interface IRunLoad
    {
        static abstract Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions;
    }

    // One run: 32 bytes at one shift.
    private unsafe struct OneRun : IRunLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load<TI>(byte* at, Vector256<ulong>* shifts, int* counts, int s) where TI : struct, IInstructions => TI.ShiftsByLane
            ? TI.ShiftRight(Vector256.Load(at).AsUInt64(), shifts[s]).AsByte()
            : Vector256.ShiftRightLogical(Vector256.Load(at).AsUInt64(), counts[8 * s]).AsByte();
    }

    // Two runs: 16 bytes at two shifts (a 128-bit half each).
    private unsafe struct TwoRuns : IRunLoad
    {
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
    private static unsafe float GatheredOne<TI, TF, TS>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TI : struct, IInstructions where TF : struct, IFactors where TS : struct, DescribedPlan.IGatheredSteps
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins, eight = t == 8;
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

    // DescribedRowSteps for a plan whose steps are gathered (GatherSteps): each group's dword vectors gathered into one
    // scratch made once a call, its steps, scales and minimums stored.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void GatheredRowSteps<TI, TF, TS>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
        where TI : struct, IInstructions where TF : struct, IFactors where TS : struct, DescribedPlan.IGatheredSteps
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, stepBytes = plan.GroupStepBytes, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins;
        Vector256<uint>* words = stackalloc Vector256<uint>[Math.Max(1, plan.MostGroupWords)];
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp, first = gp * t;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            byte* to = into + (nint)g * stepBytes;
            plan.GatherWords<TI>(at, gp, words);
            for (int s = 0; s < t; s++)
            {
                TS.Step<TI>(plan, words, first + s).Store(to + 32 * s);
            }

            TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            scale.Store((float*)(to + 32 * t));
            if (mins)
            {
                min.Store((float*)(to + 32 * t + 32));
            }
        }
    }

    // MadeOne for a plan with uniform runs (P = 1), its load and map as type parameters.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float UniformOne<TR, TM, TI, TF>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions where TF : struct, IFactors
    {
        int gpp = plan.GroupsPerPeriod, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins, c32 = plan.C == 32;
        int* offsets = plan.RunOffsets;
        Vector256<ulong>* shifts = plan.RunShifts;
        int* counts = plan.RunCounts;
        var mask = plan.PlaneMask0;
        var table = plan.MapVector;
        bool gathered = plan.Gathered;
        int runs = plan.UniformRuns;
        int* gatherOffsets = plan.GatherOffsets;
        Vector256<uint>* gatherShifts = plan.GatherShifts;
        int* gatherShift = plan.GatherShift;
        var wordFlip = plan.WordFlip;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            Vector256<byte> s0, s1, s2, s3, s4, s5, s6, s7;
            if (gathered)
            {
                if (typeof(TM) == typeof(FlippedCodes))
                {
                    GatheredSteps<SameCodes, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, wordFlip, mask, table,
                        out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
                else
                {
                    GatheredSteps<TM, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, default, mask, table,
                        out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
            }
            else
            {
                UniformSteps<TR, TM, TI>(at, gp, offsets, shifts, counts, mask, table, c32, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            }

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

    // DescribedRowSteps for a plan with uniform runs (P = 1), its load and map as type parameters.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void UniformRowSteps<TR, TM, TI, TF>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
        where TR : struct, IRunLoad where TM : struct, ICodeMap where TI : struct, IInstructions where TF : struct, IFactors
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, stepBytes = plan.GroupStepBytes, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins, c32 = t == 8;
        int* offsets = plan.RunOffsets;
        Vector256<ulong>* shifts = plan.RunShifts;
        int* counts = plan.RunCounts;
        var mask = plan.PlaneMask0;
        var table = plan.MapVector;
        bool gathered = plan.Gathered;
        int runs = plan.UniformRuns;
        int* gatherOffsets = plan.GatherOffsets;
        Vector256<uint>* gatherShifts = plan.GatherShifts;
        int* gatherShift = plan.GatherShift;
        var wordFlip = plan.WordFlip;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            byte* to = into + (nint)g * stepBytes;
            Vector256<byte> s0, s1, s2, s3, s4, s5, s6, s7;
            if (gathered)
            {
                if (typeof(TM) == typeof(FlippedCodes))
                {
                    GatheredSteps<SameCodes, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, wordFlip, mask, table,
                        out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
                else
                {
                    GatheredSteps<TM, TI>(at, gatherOffsets + 8 * gp, gatherShifts + gp * runs, gatherShift + gp * runs, runs, default, mask, table,
                        out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
            }
            else
            {
                UniformSteps<TR, TM, TI>(at, gp, offsets, shifts, counts, mask, table, c32, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            }

            s0.Store(to);
            s1.Store(to + 32);
            s2.Store(to + 64);
            s3.Store(to + 96);
            if (c32)
            {
                s4.Store(to + 128);
                s5.Store(to + 160);
                s6.Store(to + 192);
                s7.Store(to + 224);
            }

            TF.Read<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            scale.Store((float*)(to + 32 * t));
            if (mins)
            {
                min.Store((float*)(to + 32 * t + 32));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float MadeOne<TX, TI, TF>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TX : struct, IStepLoad where TI : struct, IInstructions where TF : struct, IFactors
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, xStep = 4 * plan.Cg, cg = plan.Cg, p = plan.P;
        bool mins = plan.Mins;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Steps<TI>(at, gp, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
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

    // A stored row's k values of 2 bytes made float32 into `into` as the float32 lane reads them (DescribedSingleFour's
    // readers, the code offset added): 8 at a time, the rest one at a time. By the portable instructions (moves, shifts and
    // conversions: the same numbers by any set).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DecodeSingleRow(DescribedPlan plan, byte* row, float* into, int k)
    {
        float offset = (float)plan.D.CodeOffset;
        switch (plan.Wide)
        {
            case WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 }:
                DecodeRowBy(new HighHalfRead(), offset, row, into, k);
                break;
            case WideRead.Half:
                DecodeRowBy(new HalfRead(plan.Half16), offset, row, into, k);
                break;
            default:
                DecodeRowBy(new ShortRead(plan.D.Code.Signed), offset, row, into, k);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DecodeRowBy<TR>(TR read, float offset, byte* row, float* into, int k) where TR : struct, ISingleRead
    {
        if (offset != 0f)
        {
            DecodeRowOf(new SingleOffset<TR>(read, offset), row, into, k);
            return;
        }

        DecodeRowOf(read, row, into, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DecodeRowOf<TR>(TR read, byte* row, float* into, int k) where TR : struct, ISingleRead
    {
        int p = 0;
        if (!Vector256.IsHardwareAccelerated)
        {
            for (; p + 4 <= k; p += 4)
            {
                read.Four<PortableInstructions>(row + 2 * (nint)p).Store(into + p);
            }
        }

        for (; p + 8 <= k; p += 8)
        {
            read.Eight<PortableInstructions>(row + 2 * (nint)p).Store(into + p);
        }

        for (; p < k; p++)
        {
            into[p] = read.One(row + 2 * (nint)p);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedSingleOne<TI>(DescribedPlan plan, byte* row, float* x, int k) where TI : struct, IInstructions
    {
        float offset = (float)plan.D.CodeOffset;
        return plan.Wide switch
        {
            WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 } => SingleOneBy<HighHalfRead, TI>(new HighHalfRead(), 2, offset, row, x, k),
            WideRead.Half => SingleOneBy<HalfRead, TI>(new HalfRead(plan.Half16), 2, offset, row, x, k),
            WideRead.Short => SingleOneBy<ShortRead, TI>(new ShortRead(plan.D.Code.Signed), 2, offset, row, x, k),
            _ => SingleOneBy<SingleRead, TI>(new SingleRead(), 4, offset, row, x, k),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SingleOneBy<TR, TI>(TR read, int bytes, float offset, byte* row, float* x, int k) where TR : struct, ISingleRead where TI : struct, IInstructions =>
        offset == 0f ? SingleOne<TR, TI>(read, bytes, row, x, k) : SingleOne<SingleOffset<TR>, TI>(new SingleOffset<TR>(read, offset), bytes, row, x, k);

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
    // once for the four rows (by the plan's Factor and Raw, compiled on their own: CodesFactors, CodesRaw), then each 4 codes'
    // numbers read from the table as a half, times their chunk's scale less its minimum (the lanes the 256-bit kernel
    // broadcasts), each sum a register a half.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedCodesFour128<TI>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x, nint stride,
        out float y0, out float y1, out float y2, out float y3) where TI : struct, IInstructions
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        float* factors = stackalloc float[16];
        byte* codes = stackalloc byte[32];
        nint s2 = 2 * stride, s3 = 3 * stride;
        Vector128<float> a0l = default, a0u = default, a1l = default, a1u = default, a2l = default, a2u = default, a3l = default, a3u = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            CodesFactors<TI>(plan, at, gp, Math.Min(8, chunks - g * 8), factors);
            for (int s = 0; s < spans; s++)
            {
                CodesRaw<TI>(plan, at, gp * spans + s, codes);
                for (int e = 0; e < 4; e++)
                {
                    int lane = (32 * s + 8 * e) / c;
                    var d = Vector128.Create(factors[lane]);
                    var m = Vector128.Create(factors[8 + lane]);
                    byte* ce = codes + 8 * e;
                    var wl = TI.Fused(Vector128.Create(table[ce[0]], table[ce[1]], table[ce[2]], table[ce[3]]), d, m);
                    var wu = TI.Fused(Vector128.Create(table[ce[4]], table[ce[5]], table[ce[6]], table[ce[7]]), d, m);
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

    // A group's scales (at `into`) and minimums (at `into` + 8) as DescribedPlan.Factor reads them, compiled on its own.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void CodesFactors<TI>(DescribedPlan plan, byte* at, int gp, int valid, float* into) where TI : struct, IInstructions
    {
        plan.Factor<TI>(at, gp, valid, out var scale, out var min);
        scale.Store(into);
        min.Store(into + 8);
    }

    // A span's 32 codes as DescribedPlan.Raw makes them, into `into`, compiled on its own.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void CodesRaw<TI>(DescribedPlan plan, byte* at, int span, byte* into) where TI : struct, IInstructions => plan.Raw<TI>(at, span).Store(into);

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
    private readonly struct HalfVectors((int SignShift, int Magnitude, int Shift, float Times, int Special) h, float factor)
    {
        private readonly Vector256<int> magnitude = Vector256.Create(h.Magnitude), special = Vector256.Create(h.Special == 0 ? int.MaxValue : h.Special),
            sign = Vector256.Create(h.SignShift == 0 ? 0 : 0x8000);
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
