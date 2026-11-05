// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// The conversions between float32 and the 16-bit floats on the device (plan 37: no fixed type; ConvertElements), bit for
// bit the CPU's (ElementConversions: BFloat16Bits.Round, (Half) for float16, HalfFloats' widening), NaNs included, as
// Vulkan's and HIP's kernels compute them:
//   - to bfloat16 (convert_f32_bf16): round to nearest, ties to even, in integer arithmetic (bits + 0x7FFF + the kept
//     lowest bit); a NaN keeps its sign and top payload bits, made quiet ((bits >> 16) | 0x40), never rounded into an
//     infinity.
//   - to float16 (convert_f32_f16): cvt.rn.f16.f32 (round to nearest, ties to even, subnormal halves kept, from 65520 an
//     infinity, as IEEE and .NET round); a NaN in integer arithmetic, as .NET and the x86 conversion give it: sign |
//     0x7E00 | the payload's top ten bits, where the device would give its own canonical NaN.
//   - from either (convert_bf16_f32, convert_f16_f32): exact. bfloat16 is the float32's high half; float16 through
//     cvt.f32.f16, a NaN in integer arithmetic (sign | 0x7F800000 | mantissa << 13: never quieted, as HalfFloats widens it).
// bf16_pack_f32 (PackBFloat16: the packed weights of fine-tuning) rounds as the CPU's PackBFloat16 does: the same, but a
// NaN becomes 0x7FC0 (that operation's own rule on the CPU; the conversions keep the payload).
// The 16-bit values are two a word, element 2i in the low half. The narrowing kernels run a thread per output word (n =
// words; past the last element 0), the widening ones a thread per element. bf16_pack_f32 is listed in Names, the others
// in AdvancedNames.
internal static partial class PtxKernels
{
    private static void Conversions(StringBuilder sb)
    {
        PackSixteen(sb, "bf16_pack_f32", Sixteen.PackBFloat16);
        PackSixteen(sb, "convert_f32_bf16", Sixteen.BFloat16);
        PackSixteen(sb, "convert_f32_f16", Sixteen.Float16);
        WidenSixteen(sb, "convert_bf16_f32", bfloat16: true);
        WidenSixteen(sb, "convert_f16_f32", bfloat16: false);
    }

    // How a narrowing kernel rounds: bfloat16 as the conversions (a NaN's payload kept) or as PackBFloat16 (a NaN 0x7FC0),
    // or float16.
    private enum Sixteen
    {
        BFloat16,
        PackBFloat16,
        Float16,
    }

    // Word i of y = x[2i], x[2i + 1] narrowed (low half first; past `count` elements 0): n = words.
    private static void PackSixteen(StringBuilder sb, string name, Sixteen rounding)
    {
        Elementwise(sb, name, ["x", "y"], [("u32", "count")],
            $$"""
            {
            .reg .b16 %h<2>;
            mul.wide.u32 %rd1, %i, 8;
            add.u64 %rd1, %rd1, %b_x;
            ld.global.b32 %r1, [%rd1];
            mov.u32 %r2, 0;
            shl.b32 %r3, %i, 1;
            add.u32 %r3, %r3, 1;
            setp.lt.u32 %p1, %r3, %s_count;
            @%p1 ld.global.b32 %r2, [%rd1+4];
            {{Narrow("%r1", "%r4", rounding)}}
            {{Narrow("%r2", "%r5", rounding)}}
            @!%p1 mov.u32 %r5, 0;
            shl.b32 %r5, %r5, 16;
            or.b32 %r6, %r4, %r5;
            st.global.b32 [%a_y], %r6;
            }
            """);
    }

    // The 16-bit bits of float32 bits `source` into `target` (low 16 bits; uses %r10 … %r12, %f10, %h1, %p2).
    private static string Narrow(string source, string target, Sixteen rounding) => rounding switch
    {
        Sixteen.Float16 => $"""
            mov.b32 %f10, {source};
            cvt.rn.f16.f32 %h1, %f10;
            cvt.u32.u16 {target}, %h1;
            and.b32 %r11, {source}, 0x7FFFFFFF;
            setp.gt.u32 %p2, %r11, 0x7F800000;
            shr.u32 %r11, {source}, 16;
            and.b32 %r11, %r11, 0x8000;
            shr.u32 %r12, {source}, 13;
            and.b32 %r12, %r12, 0x3FF;
            or.b32 %r11, %r11, %r12;
            or.b32 %r11, %r11, 0x7E00;
            selp.b32 {target}, %r11, {target}, %p2;
            """,
        _ => BFloat16Round(source, target, packNaN: rounding == Sixteen.PackBFloat16),
    };

    /// <summary>
    /// PTX setting <paramref name="target"/> (its low 16 bits) to the bfloat16 bits of the float32 bits in
    /// <paramref name="source"/> (they may be one register): to nearest, ties to even, in integer arithmetic; a NaN as the
    /// conversions keep it ((bits &gt;&gt; 16) | 0x40) or, <paramref name="packNaN"/>, as PackBFloat16 writes it (0x7FC0). In
    /// a block of its own registers, so any kernel may use it.
    /// </summary>
    internal static string BFloat16Round(string source, string target, bool packNaN = false) => $$"""
        {
        .reg .b32 %bt<3>;
        .reg .pred %bnan;
        shr.u32 %bt0, {{source}}, 16;
        and.b32 %bt0, %bt0, 1;
        add.u32 %bt0, %bt0, 0x7FFF;
        add.u32 %bt1, {{source}}, %bt0;
        shr.u32 %bt1, %bt1, 16;
        and.b32 %bt2, {{source}}, 0x7FFFFFFF;
        setp.gt.u32 %bnan, %bt2, 0x7F800000;
        {{(packNaN ? "mov.u32 %bt2, 0x7FC0;" : $"shr.u32 %bt2, {source}, 16;\n        or.b32 %bt2, %bt2, 0x40;")}}
        selp.b32 {{target}}, %bt2, %bt1, %bnan;
        }
        """;

    // y[i] = element i of the 16-bit words x widened: n = elements.
    private static void WidenSixteen(StringBuilder sb, string name, bool bfloat16)
    {
        string widen = bfloat16
            ? "shl.b32 %r4, %r2, 16;"
            : """
              and.b32 %r2, %r2, 0xFFFF;
              cvt.u16.u32 %h1, %r2;
              cvt.f32.f16 %f1, %h1;
              mov.b32 %r4, %f1;
              and.b32 %r5, %r2, 0x7FFF;
              setp.gt.u32 %p1, %r5, 0x7C00;
              and.b32 %r6, %r2, 0x8000;
              shl.b32 %r6, %r6, 16;
              and.b32 %r7, %r2, 0x3FF;
              shl.b32 %r7, %r7, 13;
              or.b32 %r6, %r6, %r7;
              or.b32 %r6, %r6, 0x7F800000;
              selp.b32 %r4, %r6, %r4, %p1;
              """;
        Elementwise(sb, name, ["x", "y"], [],
            $$"""
            {
            .reg .b16 %h<2>;
            shr.u32 %r1, %i, 1;
            mul.wide.u32 %rd1, %r1, 4;
            add.u64 %rd1, %rd1, %b_x;
            ld.global.b32 %r2, [%rd1];
            and.b32 %r3, %i, 1;
            shl.b32 %r3, %r3, 4;
            shr.u32 %r2, %r2, %r3;
            {{widen}}
            st.global.b32 [%a_y], %r4;
            }
            """);
    }
}
