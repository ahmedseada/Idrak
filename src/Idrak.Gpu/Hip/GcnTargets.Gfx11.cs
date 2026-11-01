// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;

namespace Idrak.Gpu.Abstractions;

public static partial class GcnTargets
{
    /// <summary>The gfx11 family (RDNA 3), made but not registered until its kernels are (plan 35, G8a).</summary>
    internal static IGcnTarget Gfx11Family { get; } = new Gfx11();

    /// <summary>
    /// The gfx11 family: ISA 11.0.x (RDNA 3), run by Idrak's kernels in waves of 64. Its encodings are ISA 11's
    /// (<see cref="GcnEncoding.Isa11"/>), its instructions those of its table, in gfx11's own names.
    /// </summary>
    private sealed partial class Gfx11 : Gfx10And11
    {
        public override string Family => "gfx11";

        public override GcnEncoding Encoding => GcnEncoding.Isa11;

        public override bool Handles(GcnIsaVersion isa) => isa is { Major: 11, Minor: 0 } && TryMachine(isa, out _, out _, out _);

        // 11.0.0 and 11.0.1 hold half as many vector registers again as RDNA 2 (1,536 a SIMD in wave32, 768 in wave64,
        // allocated in twelves in wave64); 11.0.2 and 11.0.3 RDNA 2's (LLVM 18's FeatureGFX11FullVGPRs).
        protected override int VectorFile(GcnIsaVersion isa) => FullVectorFile(isa) ? 768 : 512;

        protected override int VectorAllocation(GcnIsaVersion isa) => FullVectorFile(isa) ? 12 : 4;

        private static bool FullVectorFile(GcnIsaVersion isa) => isa.Stepping is 0 or 1;

        // M0 is 125, null 124 (gfx10's swapped).
        protected override int M0 => 125;

        protected override int Null => 124;

        protected override GcnInstruction?[] ByOp => Index.ByOp;

        protected override FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic => Index.ByMnemonic;

        // s_waitcnt: expcnt at 2:0, lgkmcnt (six bits) at 9:4, vmcnt (six bits) at 15:10.
        public override ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt) => (ushort)((expcnt & 7) | (lgkmcnt & 0x3F) << 4 | (vmcnt & 0x3F) << 10);

        public override (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits) => (bits >> 10 & 0x3F, bits & 7, bits >> 4 & 0x3F);

        private static class Index
        {
            public static readonly GcnInstruction?[] ByOp = IndexByOp(Instructions, "gfx11");

            public static readonly FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic =
                Instructions.ToFrozenDictionary(i => i.Mnemonic, StringComparer.Ordinal).GetAlternateLookup<ReadOnlySpan<char>>();
        }
    }
}
