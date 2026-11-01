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

        // Measured on gfx1100 and gfx1102 (llc 18): a transcendental's result read too soon, and an SGPR written by an
        // SALU after a VALU read it as a lane mask.
        public override GcnHazardFixes HazardFixes(GcnIsaVersion? isa) => GcnHazardFixes.TransResultThenValu | GcnHazardFixes.LaneMaskThenSaluWrite;

        // M0 is 125, null 124 (gfx10's swapped).
        protected override int M0 => 125;

        protected override int Null => 124;

        protected override GcnInstruction?[] ByOp => Index.ByOp;

        protected override FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic => Index.ByMnemonic;

        protected override FrozenDictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> Aliases => Index.Aliases;

        // s_waitcnt: expcnt at 2:0, lgkmcnt (six bits) at 9:4, vmcnt (six bits) at 15:10.
        public override ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt) => (ushort)((expcnt & 7) | (lgkmcnt & 0x3F) << 4 | (vmcnt & 0x3F) << 10);

        public override (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits) => (bits >> 10 & 0x3F, bits & 7, bits >> 4 & 0x3F);

        private static class Index
        {
            public static readonly GcnInstruction?[] ByOp = IndexByOp(Instructions, "gfx11");

            public static readonly FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic =
                Instructions.ToFrozenDictionary(i => i.Mnemonic, StringComparer.Ordinal).GetAlternateLookup<ReadOnlySpan<char>>();

            // gfx9's names of the instructions gfx11 renamed: LLVM 18's gfx11 aliases (llvm-mc -mcpu=gfx1100 takes each and
            // prints the right-hand name, 2026-11-02), the ones Idrak's kernels write.
            public static readonly FrozenDictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> Aliases = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["v_add_u32"] = "v_add_nc_u32",
                ["v_sub_u32"] = "v_sub_nc_u32",
                ["v_subrev_u32"] = "v_subrev_nc_u32",
                ["s_load_dword"] = "s_load_b32",
                ["s_load_dwordx2"] = "s_load_b64",
                ["s_load_dwordx4"] = "s_load_b128",
                ["s_load_dwordx8"] = "s_load_b256",
                ["global_load_dword"] = "global_load_b32",
                ["global_load_dwordx4"] = "global_load_b128",
                ["global_load_ubyte"] = "global_load_u8",
                ["global_store_dword"] = "global_store_b32",
                ["global_store_dwordx4"] = "global_store_b128",
                ["global_store_byte"] = "global_store_b8",
                ["global_store_short"] = "global_store_b16",
                ["global_atomic_add"] = "global_atomic_add_u32",
                ["ds_read_b32"] = "ds_load_b32",
                ["ds_read_b128"] = "ds_load_b128",
                ["ds_write_b32"] = "ds_store_b32",
                ["s_andn2_b64"] = "s_and_not1_b64",
            }.ToFrozenDictionary(StringComparer.Ordinal).GetAlternateLookup<ReadOnlySpan<char>>();
        }
    }
}
