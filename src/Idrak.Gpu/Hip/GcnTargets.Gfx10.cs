// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Diagnostics.CodeAnalysis;

namespace Idrak.Gpu.Abstractions;

public static partial class GcnTargets
{
    /// <summary>The gfx10 family (RDNA 1 and 2), made but not registered until its kernels are (plan 35, G8a).</summary>
    internal static IGcnTarget Gfx10Family { get; } = new Gfx10();

    /// <summary>
    /// The gfx10 family: ISA 10.1.x and 10.3.x (RDNA 1 and 2), run by Idrak's kernels in waves of 64. Its encodings are
    /// ISA 10's (<see cref="GcnEncoding.Isa10"/>), its instructions those of the table below.
    /// </summary>
    private sealed partial class Gfx10 : Gfx10And11
    {
        public override string Family => "gfx10";

        public override GcnEncoding Encoding => GcnEncoding.Isa10;

        public override bool Handles(GcnIsaVersion isa) => isa is { Major: 10, Minor: 1 or 3 } && TryMachine(isa, out _, out _, out _);

        // RDNA 1 and 2: 1,024 vector registers a SIMD in wave32, 512 in wave64 (LLVM 18's getTotalNumVGPRs).
        protected override int VectorFile(GcnIsaVersion isa) => 512;

        protected override int VectorAllocation(GcnIsaVersion isa) => 4;

        // ISA 10.1 (gfx1010-gfx1013) only, measured with llc 18 (gfx1030 has neither); an unknown version gets them.
        public override GcnHazardFixes HazardFixes(GcnIsaVersion? isa) => isa is { Minor: not 1 }
            ? GcnHazardFixes.None
            : GcnHazardFixes.VmemSgprThenScalarWrite | GcnHazardFixes.SmemSgprThenValuWrite;

        // M0 is 124, null (a global instruction's off, SMEM's absent offset) 125.
        protected override int M0 => 124;

        protected override int Null => 125;

        protected override GcnInstruction?[] ByOp => Index.ByOp;

        protected override FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic => Index.ByMnemonic;

        protected override FrozenDictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> Aliases => Index.Aliases;

        // s_waitcnt: vmcnt's low four bits at 3:0 and high two at 15:14, expcnt at 6:4, lgkmcnt (six bits) at 13:8.
        public override ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt) =>
            (ushort)((vmcnt & 0xF) | (expcnt & 7) << 4 | (lgkmcnt & 0x3F) << 8 | (vmcnt >> 4 & 3) << 14);

        public override (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits) => ((bits & 0xF) | (bits >> 14 & 3) << 4, bits >> 4 & 7, bits >> 8 & 0x3F);

        private static class Index
        {
            public static readonly GcnInstruction?[] ByOp = IndexByOp(Instructions, "gfx10");

            public static readonly FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic =
                Instructions.ToFrozenDictionary(i => i.Mnemonic, StringComparer.Ordinal).GetAlternateLookup<ReadOnlySpan<char>>();

            // gfx9's names for the adds ISA 10 renamed (no carry: v_add_nc_u32). llvm-mc 18 takes them on gfx1100 (its
            // gfx11 aliases) but not on gfx1030; Idrak takes them on both, so one kernel text serves every family.
            public static readonly FrozenDictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> Aliases = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["v_add_u32"] = "v_add_nc_u32",
                ["v_sub_u32"] = "v_sub_nc_u32",
                ["v_subrev_u32"] = "v_subrev_nc_u32",
            }.ToFrozenDictionary(StringComparer.Ordinal).GetAlternateLookup<ReadOnlySpan<char>>();
        }
    }

    /// <summary>
    /// What the RDNA families (gfx10, gfx11) share: wave64 as Idrak's kernels run it, 106 scalar registers, the descriptor's
    /// bits, the occupancy and the wait counters' limits. Every descriptor number here was read from what <c>llvm-mc</c> 18
    /// wrote for gfx1010, gfx1030, gfx1100 and gfx1102 with <c>+wavefrontsize64</c> (2026-11-01): the vector blocks in
    /// eights of registers, no scalar blocks, the mode bits. The occupancy follows LLVM 18's AMDGPUBaseInfo (20 waves a
    /// SIMD, the vector file and its allocation granule by wave size; scalar registers do not bound it on ISA 10 and later).
    /// </summary>
    private abstract class Gfx10And11 : IGcnTarget, IGcnInstructionSet
    {
        // COMPUTE_PGM_RSRC1: DX10_CLAMP (21) and IEEE_MODE (23), as on gfx9, and MEM_ORDERED (30). WGP_MODE (29) is off:
        // a workgroup runs on one compute unit (LLVM's cumode), so its waves share one L0 cache and LDS as on gfx9, and the
        // kernels' workgroup-scope memory needs no L0 invalidation between waves.
        private const uint ModeBits = (1u << 21) | (1u << 23) | (1u << 30);

        public abstract string Family { get; }

        public abstract GcnEncoding Encoding { get; }

        public IReadOnlyList<int> WaveSizes { get; } = [64];

        public uint PaddingWord => 0xBF800000;   // s_nop 0

        public abstract bool Handles(GcnIsaVersion isa);

        protected abstract int VectorFile(GcnIsaVersion isa);

        protected abstract int VectorAllocation(GcnIsaVersion isa);

        protected abstract int M0 { get; }

        protected abstract int Null { get; }

        protected abstract GcnInstruction?[] ByOp { get; }

        protected abstract FrozenDictionary<string, GcnInstruction>.AlternateLookup<ReadOnlySpan<char>> ByMnemonic { get; }

        // gfx9's spellings of the family's instructions, to the family's mnemonic (which the printer writes): Idrak's
        // kernels are one text for every family, written in gfx9's names, as PTX is one text for every SM.
        protected abstract FrozenDictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> Aliases { get; }

        public GcnRegisterFile Registers(GcnIsaVersion isa, int waveSize)
        {
            CheckWave(waveSize);
            return new GcnRegisterFile(Vgprs: 256, VgprGranule: VectorAllocation(isa), Sgprs: 106, SgprGranule: 8, MaxWavesPerSimd: 20);
        }

        // ISA 10 and later hold no xnack mask or flat scratch among a wave's scalar registers; VCC is 2 more (LLVM's count).
        public int SgprCount(GcnIsaVersion isa, int sgprs, bool vcc, bool flatScratch)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sgprs);
            return sgprs + (vcc ? 2 : 0);
        }

        public int WavesPerSimd(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize)
        {
            CheckCounts(vgprs, sgprs, waveSize);
            int granule = VectorAllocation(isa);
            return vgprs == 0 ? 20 : Math.Min(20, VectorFile(isa) / AlignUp(vgprs, granule));
        }

        public GcnDescriptorBits Descriptor(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize)
        {
            CheckCounts(vgprs, sgprs, waveSize);
            uint vgprBlocks = (uint)(AlignUp(Math.Max(1, vgprs), 8) / 8 - 1);
            return new GcnDescriptorBits(vgprBlocks | ModeBits, Rsrc3: 0, CodeProperties: 0);   // wave64: ENABLE_WAVEFRONT_SIZE32 (bit 10) clear
        }

        public uint ElfFlags(GcnIsaVersion isa, int codeObjectVersion) => ElfFlagsOf(isa, codeObjectVersion);

        public int Ttmps => 16;

        public GcnInstruction? Find(GcnOp op) => (uint)op < (uint)ByOp.Length ? ByOp[(int)op] : null;

        public bool TryFind(ReadOnlySpan<char> mnemonic, [NotNullWhen(true)] out GcnInstruction? instruction)
        {
            if (ByMnemonic.TryGetValue(mnemonic, out instruction))
            {
                return true;
            }

            instruction = null;
            return Aliases.TryGetValue(mnemonic, out string? name) && ByMnemonic.TryGetValue(name, out instruction);
        }

        // The table holds the instructions every version of the family has (the forms Idrak's kernels use).
        public GcnFeatures Features(GcnIsaVersion isa) => GcnFeatures.None;

        // The scalar operand codes: s0-s105, VCC 106, the trap registers 108-123, M0 and null (the family's), EXEC 126; the
        // apertures and the wave's id 235-239, VCCZ, EXECZ, SCC and LDS direct 251-254. Off is null.
        public int Special(GcnSpecial register) => register switch
        {
            GcnSpecial.Vcc or GcnSpecial.VccLo => 106,
            GcnSpecial.VccHi => 107,
            GcnSpecial.M0 => M0,
            GcnSpecial.Exec or GcnSpecial.ExecLo => 126,
            GcnSpecial.ExecHi => 127,
            GcnSpecial.SharedBase => 235,
            GcnSpecial.SharedLimit => 236,
            GcnSpecial.PrivateBase => 237,
            GcnSpecial.PrivateLimit => 238,
            GcnSpecial.PopsExitingWaveId => 239,
            GcnSpecial.Vccz => 251,
            GcnSpecial.Execz => 252,
            GcnSpecial.Scc => 253,
            GcnSpecial.LdsDirect => 254,
            GcnSpecial.Off or GcnSpecial.Null => Null,
            _ => throw new ArgumentOutOfRangeException(nameof(register), register, $"Not a special register the {Family} family encodes (it has no flat scratch or xnack mask among its scalar registers)."),
        };

        // The hardware registers s_getreg_b32 and s_setreg_b32 name (llvm-mc's ids for ISA 10 and 11).
        public int HardwareRegister(ReadOnlySpan<char> name) => name switch
        {
            "HW_REG_MODE" => 1,
            "HW_REG_STATUS" => 2,
            "HW_REG_TRAPSTS" => 3,
            "HW_REG_GPR_ALLOC" => 5,
            "HW_REG_LDS_ALLOC" => 6,
            "HW_REG_IB_STS" => 7,
            "HW_REG_SH_MEM_BASES" => 15,
            "HW_REG_HW_ID1" => 23,
            "HW_REG_HW_ID2" => 24,
            _ => -1,
        };

        public int Ttmp(int index) => (uint)index < 16 ? 108 + index : throw new ArgumentOutOfRangeException(nameof(index), index, $"{Family} has ttmp0 to ttmp15.");

        // The wait counters (ISA 10 and 11): vmcnt counts the vector loads (and atomics with a return) in order, vscnt the
        // stores and atomics without one, in order (LLVM 18's SIInsertWaitcnts: s_waitcnt_vscnt null, n for them); lgkmcnt
        // LDS in order and scalar memory out of order, as on gfx9.
        public GcnCounting Counting(GcnMemoryClass memory) => memory switch
        {
            GcnMemoryClass.Vector => new(GcnCounter.Vm, InOrder: true),
            GcnMemoryClass.VectorStore => new(GcnCounter.Vs, InOrder: true),
            GcnMemoryClass.Flat => new(GcnCounter.Vm | GcnCounter.Lgkm, InOrder: false),
            GcnMemoryClass.Lds => new(GcnCounter.Lgkm, InOrder: true),
            GcnMemoryClass.Gds => new(GcnCounter.Lgkm, InOrder: true, DataLock: GcnCounter.Exp),
            GcnMemoryClass.Scalar => new(GcnCounter.Lgkm, InOrder: false),
            _ => default,
        };

        public int CounterLimit(GcnCounter counter) => counter switch
        {
            GcnCounter.Vm => 63,
            GcnCounter.Lgkm => 63,
            GcnCounter.Exp => 7,
            GcnCounter.Vs => 63,
            _ => throw new ArgumentOutOfRangeException(nameof(counter), counter, "One counter: vmcnt, lgkmcnt, expcnt or vscnt."),
        };

        public abstract ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt);

        public abstract (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits);

        // s_barrier gets every wait before it, as on gfx9: the kernels' text names no fence, so the barrier's wait is the
        // release that makes a wave's LDS and memory writes visible to the others.
        public bool BarrierWaitsForMemory => false;

        // The hazards (LLVM 18's GCNHazardRecognizer, llc -run-pass=post-RA-hazard-rec on gfx1010, gfx1030, gfx1100 and
        // gfx1102 in wave64, 2026-11-01/02; tools/gcn/measure_hazards.py and measure_hazard_windows.py): none of gfx9's s_nop
        // cases on any of them, so every count here is 0. The fixes LLVM makes otherwise are instructions of their own, the
        // four of GcnHazardFixes (each family's HazardFixes: ISA 10.1's two, ISA 11's two), which the hazard pass writes.
        // KNOWN GAP, not inserted (measured for some shapes only, so no rule is guessed):
        // - ISA 10.1: LDS, a branch, then vector memory (or the reverse): s_waitcnt_vscnt null, 0 before the second;
        // - ISA 11: a VALU reads VGPRs written on both sides of an EXEC write (partial forwarding): s_waitcnt_depctr 0x0fff,
        //   measured for one shape only (5 VALUs between clear it, SALUs do not).
        // gfx1030 (ISA 10.3) needs none of them.
        public virtual int HazardWaitStates(GcnHazard hazard, GcnFeatureSetting xnack) => 0;

        public abstract GcnHazardFixes HazardFixes(GcnIsaVersion? isa);

        // Waves of a workgroup spread over one compute unit's two SIMDs (CU mode, as the descriptor sets).
        public int SimdsPerComputeUnit => 2;

        private void CheckWave(int waveSize)
        {
            if (waveSize != 64)
            {
                throw new ArgumentOutOfRangeException(nameof(waveSize), waveSize, $"Idrak's {Family} kernels run waves of 64 lanes.");
            }
        }

        private void CheckCounts(int vgprs, int sgprs, int waveSize)
        {
            CheckWave(waveSize);
            ArgumentOutOfRangeException.ThrowIfNegative(vgprs);
            ArgumentOutOfRangeException.ThrowIfNegative(sgprs);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(vgprs, 256);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(sgprs, 106 + 2);   // the most a kernel addresses, and VCC
        }

        private static int AlignUp(int value, int granule) => (value + granule - 1) / granule * granule;
    }

    private static GcnInstruction Row(GcnOp op, string mnemonic, GcnFormat format, int code, string operands, GcnInstructionFlags flags = 0) =>
        new(op, mnemonic, format, code, operands, flags, GcnFeatures.None);

    private static GcnInstruction?[] IndexByOp(GcnInstruction[] instructions, string family)
    {
        var byOp = new GcnInstruction?[Enum.GetValues<GcnOp>().Length];
        foreach (var instruction in instructions)
        {
            byOp[(int)instruction.Op] = byOp[(int)instruction.Op] is null
                ? instruction
                : throw new InvalidOperationException($"{instruction.Op} is in {family}'s table twice.");
        }

        return byOp;
    }
}
