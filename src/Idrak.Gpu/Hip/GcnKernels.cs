// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>
/// HIP's kernels in GCN, AMD's own instruction set, written in C# as <c>Cuda/PtxKernels.cs</c> writes CUDA's in PTX:
/// with virtual registers, which Idrak's assembler (<see cref="GcnAssembler.AssembleKernel"/>) allocates, adding the
/// waits and wait states, for the family of the target a device reports, then into the code object the HIP runtime
/// loads (<see cref="GcnCodeObject"/>). No SDK, no compiler: the driver's runtime alone loads them (plan 35). So far
/// the gate kernel, <c>fill_f32</c>.
/// </summary>
internal static partial class GcnKernels
{
    /// <summary>The block <c>fill_f32</c> is written for (its required workgroup size): one element a work-item.</summary>
    internal const int FillBlock = 256;

    /// <summary>
    /// The most elements <c>fill_f32</c> fills: its byte offset is a 32-bit unsigned value (element × 4), so 2^30.
    /// </summary>
    internal const int FillMostElements = 1 << 30;

    /// <summary><c>fill_f32</c>'s arguments: out (a global float pointer) at 0, value (a float) at 8, count (a uint32) at 12.</summary>
    internal static IReadOnlyList<GcnArgument> FillArguments { get; } =
        [GcnArgument.Buffer(0, "out"), GcnArgument.Value(8, 4, "value"), GcnArgument.Value(12, 4, "count")];

    /// <summary>Bytes of <c>fill_f32</c>'s kernel arguments.</summary>
    internal const int FillArgumentBytes = 16;

    /// <summary>
    /// <c>fill_f32</c>'s text: out[i] = value for every i &lt; count, one element a work-item in blocks of
    /// <see cref="FillBlock"/> (ceil(count / 256) blocks). The kernel arguments' address arrives in s[0:1], the workgroup
    /// id x in s2, the work-item id x in v0. Its allocation is the code of tests/Idrak.Tests/data/gcn/kernels/fill_f32.s,
    /// whose code object llvm-mc and ld.lld made, byte for byte.
    /// </summary>
    internal static ReadOnlySpan<byte> FillText => """
        .kernel fill_f32
        .arg out, 8, global
        .arg value, 4, value
        .arg count, 4, value
        .reqd_workgroup_size 256, 1, 1
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dwordx4 %s[0:3], s[0:1], 0x0      ; %s[0:1] out, %s2 value, %s3 count
          s_lshl_b32 %s4, s2, 8                    ; the block's first element: workgroup id x * 256
          v_add_u32_e32 %v0, %s4, v0               ; this work-item's element
          v_cmp_gt_u32_e32 vcc, %s3, %v0           ; element < count
          s_and_saveexec_b64 %s[6:7], vcc          ; only those lanes
          s_cbranch_execz .Lfill_f32_done
          v_lshlrev_b32_e32 %v1, 2, %v0            ; the byte offset
          v_mov_b32_e32 %v2, %s2
          global_store_dword %v1, %v2, %s[0:1]
        .Lfill_f32_done:
          s_endpgm
        """u8;

    /// <summary><c>fill_f32</c> (<see cref="FillText"/>) for <paramref name="target"/>'s family.</summary>
    internal static GcnKernelCode Fill(IGcnTarget target)
    {
        using var asm = new GcnAssembler(target, bytes: 64);
        return asm.AssembleKernel(FillText);
    }

    /// <summary>The code object of <c>fill_f32</c> for <paramref name="isa"/> in code object format <paramref name="format"/>.</summary>
    /// <exception cref="NotRegisteredException">No registered family handles the target.</exception>
    internal static byte[] FillCodeObject(GcnIsaVersion isa, int format)
    {
        var target = GcnTargets.For(isa);
        return GcnCodeObject.Write(target, isa, format, [Fill(target)]);
    }

    /// <summary>
    /// <c>fill_f32</c>'s kernel arguments packed as its metadata lays them out (<see cref="FillArguments"/>: each at its
    /// offset, of its size, little-endian as the device reads them) into <paramref name="arguments"/>, <see cref="FillArgumentBytes"/> long.
    /// </summary>
    internal static void PackFill(Span<byte> arguments, ulong output, float value, uint count)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, FillArgumentBytes);
        BinaryPrimitives.WriteUInt64LittleEndian(arguments[FillArguments[0].Offset..], output);
        BinaryPrimitives.WriteSingleLittleEndian(arguments[FillArguments[1].Offset..], value);
        BinaryPrimitives.WriteUInt32LittleEndian(arguments[FillArguments[2].Offset..], count);
    }
}
