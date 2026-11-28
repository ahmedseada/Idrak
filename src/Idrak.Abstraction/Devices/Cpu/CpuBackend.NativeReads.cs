// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// The 16-bit floats of a machine format (a type whose native kind is bfloat16 or float16, IElementType.Native) read by
// their bits in the vector lanes: a candidate against the type's default read (IElementType.Read), the faster at each size
// measured once for the machine and kept (CpuChoices), with the same bits (every finite value, ±0 and ±∞ exact, a NaN as
// the type's own read gives it). .NET 10 offers no instruction of AVX-512 FP16 or BF16 or of Arm's FP16 arithmetic to
// compute such values directly, so this lane converts them in registers; where it is slower, the type's read serves.
internal sealed unsafe partial class CpuBackend
{
    // A choice of how a native kind's values are read: by the kind's size class and the vector paths setting.
    private readonly record struct ReadKey(NativeKind Kind, int Size, CpuVectorPaths Vectors);

    private static readonly ConcurrentDictionary<ReadKey, bool> s_readChosen = new();

    /// <summary>Whether typed reads take the native kind's lane (true), the type's read (false), or the measured choice (null). For tests.</summary>
    internal static bool? ForcedNativeRead;

    // Whether the native kind's lane reads `count` values here (measured where it was not, on these bytes).
    private static bool NativeReadChosen(IElementType type, NativeKind kind, ReadOnlySpan<byte> stored, int count)
    {
        if (ForcedNativeRead is { } forced)
        {
            return forced;
        }

        var key = new ReadKey(kind, SizeClass(count), ComputeResources.CpuVectors);
        return s_readChosen.TryGetValue(key, out bool native) ? native : MeasureRead(key, type, stored, count);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool MeasureRead(ReadKey key, IElementType type, ReadOnlySpan<byte> stored, int count)
    {
        string shape = string.Create(CultureInfo.InvariantCulture, $"read {key.Kind} 2^{key.Size}");
        string fileKey = string.Create(CultureInfo.InvariantCulture, $"{shape} {key.Vectors}");
        // What the candidates are: the code of both reads (CpuCode: the kind's lane and the type's own read).
        string variant = "c" + CpuCode.Of(typeof(ReadCandidates), type.GetType()) + "/type.kind";
        bool native = false;
        if (CpuChoices.TryGet(fileKey, variant, out var kept) && kept is "kind" or "type")
        {
            native = kept == "kind";
        }
        else if (CpuTuning.Autotune)
        {
            var scratch = GC.AllocateUninitializedArray<float>(Math.Max(1, count), pinned: true);
            fixed (byte* s = stored)
            fixed (float* v = scratch)
            {
                var each = new ReadCandidates(type, key.Kind, s, stored.Length, v, count);
                var times = CpuChoices.RoundTimes(ref each, 2);
                native = CpuChoices.Lowest(times) == 1;
                ProductRoutes.Chose("cpu", shape, string.Create(CultureInfo.InvariantCulture,
                    $"{(native ? "the kind's lane" : "the type's read")} (the type's read {times[0]:0.####} ms, the kind's lane {times[1]:0.####} ms)"));
            }

            CpuChoices.Keep(fileKey, variant, native ? "kind" : "type");
        }

        s_readChosen[key] = native;
        return native;
    }

    private readonly struct ReadCandidates(IElementType type, NativeKind kind, byte* stored, int bytes, float* values, int count) : CpuChoices.ICandidates
    {
        public void Run(int index)
        {
            if (index == 0)
            {
                type.Read(new ReadOnlySpan<byte>(stored, bytes), new Span<float>(values, count));
            }
            else
            {
                NativeRead(type, kind, (ushort*)stored, values, count);
            }
        }
    }

    /// <summary>The 16-bit floats <paramref name="codes"/> of <paramref name="type"/>'s native kind as float32 by the kind's lane (tests).</summary>
    internal static void NativeRead(IElementType type, ReadOnlySpan<ushort> codes, Span<float> values)
    {
        fixed (ushort* s = codes)
        fixed (float* v = values)
        {
            NativeRead(type, type.Native, s, v, Math.Min(codes.Length, values.Length));
        }
    }

    /// <summary>
    /// <paramref name="count"/> 16-bit floats of <paramref name="kind"/> at <paramref name="stored"/> as float32 by their
    /// bits: bfloat16 shifted into float32's high half; float16's exponent rebiased (a subnormal its mantissa times 2^-24,
    /// exact); a NaN as the type's own read gives a NaN of its sign.
    /// </summary>
    internal static void NativeRead(IElementType type, NativeKind kind, ushort* stored, float* values, int count)
    {
        // The type's own read of a positive and a negative NaN code: what every NaN of that sign reads as.
        bool brain = kind == NativeKind.BFloat16;
        ushort* codes = stackalloc ushort[2];
        codes[0] = brain ? (ushort)0x7FC0 : (ushort)0x7E00;
        codes[1] = brain ? (ushort)0xFFC0 : (ushort)0xFE00;
        float* nans = stackalloc float[2];
        type.Read(new ReadOnlySpan<byte>(codes, 4), new Span<float>(nans, 2));
        uint positive = BitConverter.SingleToUInt32Bits(nans[0]), negative = BitConverter.SingleToUInt32Bits(nans[1]);
        int i = 0;
        if (VectorsAllowed && Vector256.IsHardwareAccelerated)
        {
            var nanPositive = Vector256.Create(positive);
            var nanNegative = Vector256.Create(negative);
            for (; i + 8 <= count; i += 8)
            {
                var h = Vector128.Load(stored + i);
                var w = Vector256.Create(Vector128.WidenLower(h), Vector128.WidenUpper(h));
                var bits = brain ? Brain(w) : Half(w);
                var nan = Vector256.GreaterThan(w & Vector256.Create(0x7FFFu), Vector256.Create(brain ? 0x7F80u : 0x7C00u));
                var sign = Vector256.Equals(w & Vector256.Create(0x8000u), Vector256.Create(0x8000u));
                bits = Vector256.ConditionalSelect(nan, Vector256.ConditionalSelect(sign, nanNegative, nanPositive), bits);
                bits.AsSingle().Store(values + i);
            }
        }

        for (; i < count; i++)
        {
            uint w = stored[i];
            uint bits = (w & 0x7FFF) > (brain ? 0x7F80u : 0x7C00u) ? ((w & 0x8000) != 0 ? negative : positive) : brain ? w << 16 : HalfBits(w);
            values[i] = BitConverter.UInt32BitsToSingle(bits);
        }
    }

    // bfloat16's bits as float32's: the high half.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> Brain(Vector256<uint> w) => Vector256.ShiftLeft(w, 16);

    // float16's bits as float32's (NaNs set apart by the caller): ±0 and subnormals m · 2^-24, normals rebiased, ±∞.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> Half(Vector256<uint> w)
    {
        var sign = Vector256.ShiftLeft(w & Vector256.Create(0x8000u), 16);
        var e = Vector256.ShiftRightLogical(w, 10) & Vector256.Create(0x1Fu);
        var m = w & Vector256.Create(0x3FFu);
        var normal = Vector256.ShiftLeft(e + Vector256.Create(112u), 23) | Vector256.ShiftLeft(m, 13);
        var small = (Vector256.ConvertToSingle(m.AsInt32()) * Vector256.Create(5.9604645E-08f)).AsUInt32();   // m · 2^-24, exact
        var magnitude = Vector256.ConditionalSelect(Vector256.Equals(e, Vector256<uint>.Zero), small,
            Vector256.ConditionalSelect(Vector256.Equals(e, Vector256.Create(31u)), Vector256.Create(0x7F800000u), normal));
        return sign | magnitude;
    }

    // One float16's bits as float32's (no NaN).
    private static uint HalfBits(uint w)
    {
        uint sign = (w & 0x8000) << 16, e = w >> 10 & 0x1F, m = w & 0x3FF;
        uint magnitude = e == 0 ? BitConverter.SingleToUInt32Bits(m * 5.9604645E-08f) : e == 31 ? 0x7F800000u : (e + 112) << 23 | m << 13;
        return sign | magnitude;
    }
}
