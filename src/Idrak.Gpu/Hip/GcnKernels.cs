// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Numerics;
using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Hip.GcnOperand;

namespace Idrak.Gpu.Hip;

/// <summary>
/// HIP's kernels in GCN, AMD's own instruction set, written in C# as <c>Cuda/PtxKernels.cs</c> writes CUDA's in PTX, and
/// turned into machine code by Idrak's assembler (<see cref="GcnAssembler"/>) for the family of the target a device
/// reports, then into the code object the HIP runtime loads (<see cref="GcnCodeObject"/>). No SDK, no compiler: the
/// driver's runtime alone loads them (plan 35). So far the gate kernel, <c>fill_f32</c>, with physical registers; the
/// register allocator (virtual registers, waits and hazards from use) comes next.
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
    /// <c>fill_f32</c>: out[i] = value for every i &lt; count, one element a work-item in blocks of <see cref="FillBlock"/>
    /// (ceil(count / 256) blocks), for <paramref name="target"/>'s family. On entry s[0:1] holds the kernel arguments'
    /// address (the only user SGPRs) and s2 the workgroup id x (the first system SGPR); v0 holds the work-item id x.
    /// The same kernel as tests/Idrak.Tests/data/gcn/kernels/fill_f32.s, whose code object llvm-mc and ld.lld made.
    /// </summary>
    internal static GcnKernelCode Fill(IGcnTarget target)
    {
        using var asm = new GcnAssembler(target, bytes: 64);
        var done = asm.NewLabel();
        asm.Smem(GcnOp.SLoadDwordx4, S(4, 4), S(0, 2), 0);                    // s[4:5] out, s6 value, s7 count
        asm.Sop2(GcnOp.SLshlB32, S(2), S(2), Int(BitOperations.Log2(FillBlock)));   // the block's first element
        asm.Vop2(GcnOp.VAddU32, V(0), S(2), V(0));                              // this work-item's element
        asm.Waitcnt(lgkmcnt: 0);                                                // the arguments have arrived
        asm.Vopc(GcnOp.VCmpGtU32, S(7), V(0));                                  // element < count
        asm.Sop1(GcnOp.SAndSaveexecB64, S(0, 2), Vcc);                          // only those lanes
        asm.Branch(GcnOp.SCbranchExecz, done);
        asm.Vop2(GcnOp.VLshlrevB32, V(0), Int(2), V(0));                        // the byte offset
        asm.Vop1(GcnOp.VMovB32, V(1), S(6));
        asm.GlobalStore(GcnOp.GlobalStoreDword, V(0), V(1), S(4, 2));
        asm.Bind(done);
        asm.Sopp(GcnOp.SEndpgm);
        return new GcnKernelCode
        {
            Name = "fill_f32",
            Code = asm.ToArray(),
            Vgprs = 2,
            Sgprs = 8,
            UsesVcc = true,
            Arguments = FillArguments,
            MaxFlatWorkgroupSize = FillBlock,
            RequiredWorkgroupSize = (FillBlock, 1, 1),
        };
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
