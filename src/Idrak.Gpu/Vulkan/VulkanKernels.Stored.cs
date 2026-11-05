// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Weights kept in their file's format (plan 37, Q0): each kernel widens a stored [n, k] weight (n rows of k values, the
// file's order) into the float32 [k, n] weights the product reads, value for value as the CPU's ElementType conversions
// widen them (the same products, each rounded on its own: never fused into another operation). One invocation per
// output value, written in order. q8_0 and q4_0 blocks (32 values: a half scale d, then 32 signed bytes or 16 bytes of
// nibbles); one-byte codes (int8, float8 E4M3 or E5M2) through a 256-entry table of their values, then (value − zero) ·
// scale where the weight has scales.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> StoredKernels()
    {
        yield return ("stored_q8_0", () =>
        {
            var k = new KernelBuilder("stored_q8_0", Block);
            var (weight, w) = (k.Buffer("weight"), k.Buffer("w"));
            var (n, columns) = (k.PushInt("n"), k.PushInt("k"));
            Grid(k, n * columns, i =>
            {
                var (p, j) = (i / n, i % n);
                var at = j * (columns / 32 * 34) + (p >> 5) * 34;            // the block's first byte
                var d = HalfAtByte(k, weight, at);
                var q = SignedByteAt(k, weight, at + 2 + (p & 31));
                w[i] = k.MulRounded(q.ToFloat(), d);
            });
            return k.Build();
        });

        yield return ("stored_q4_0", () =>
        {
            var k = new KernelBuilder("stored_q4_0", Block);
            var (weight, w) = (k.Buffer("weight"), k.Buffer("w"));
            var (n, columns) = (k.PushInt("n"), k.PushInt("k"));
            Grid(k, n * columns, i =>
            {
                var (p, j) = (i / n, i % n);
                var at = j * (columns / 32 * 18) + (p >> 5) * 18;
                var d = HalfAtByte(k, weight, at);
                var e = p & 31;
                var high = e >= 16;
                var b = ByteAt(k, weight, at + 2 + k.Select(high, e - 16, e));
                var q = k.Select(high, b.ShiftRight(k.Int(4)), b & 0xF);
                w[i] = k.MulRounded((q - 8).ToFloat(), d);
            });
            return k.Build();
        });

        yield return ("stored_codes", () =>
        {
            var k = new KernelBuilder("stored_codes", Block);
            var (weight, table, scales, zeros, w) = (k.Buffer("weight"), k.Buffer("table"), k.Buffer("scales"), k.Buffer("zeros"), k.Buffer("w"));
            var (n, columns) = (k.PushInt("n"), k.PushInt("k"));
            var (blockRows, blockColumns, scaleColumns) = (k.PushInt("blockRows"), k.PushInt("blockColumns"), k.PushInt("scaleColumns"));
            var form = k.PushInt("form");                                       // 0 no scales, 1 scales, 2 scales and zero points
            Grid(k, n * columns, i =>
            {
                var (p, j) = (i / n, i % n);
                var v = table[ByteAt(k, weight, j * columns + p)];
                k.If(form.Eq(0), () => w[i] = v);
                k.If(form.Ne(0), () =>
                {
                    var s = j / k.Max(blockRows, k.Int(1)) * scaleColumns + p / k.Max(blockColumns, k.Int(1));
                    k.If(form.Eq(1), () => w[i] = k.MulRounded(v, scales[s]));
                    k.If(form.Eq(2), () => w[i] = k.MulRounded(k.AddRounded(v, -zeros[s]), scales[s]));
                });
            });
            return k.Build();
        });
    }

    // The byte at byte offset `at` (an int) of a buffer of words (little-endian), 0..255.
    private static Val ByteAt(KernelBuilder k, Buf buffer, Val at) => buffer.Int(at >> 2).ShiftRight((at & 3) * 8) & 0xFF;

    // The byte at `at` read as a signed byte.
    private static Val SignedByteAt(KernelBuilder k, Buf buffer, Val at) => ByteAt(k, buffer, at).ShiftLeft(k.Int(24)).ShiftRight(k.Int(24));

    // The IEEE half at even byte offset `at`, widened exactly (Float16At: its word, the half by the offset's parity of halves).
    private static Val HalfAtByte(KernelBuilder k, Buf buffer, Val at) => Float16At(k, buffer, at >> 2, at >> 1);
}
