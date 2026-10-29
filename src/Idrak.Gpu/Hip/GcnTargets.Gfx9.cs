// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Gpu.Abstractions;

public static partial class GcnTargets
{
    /// <summary>
    /// The gfx9 family: ISA 9.0.x, GCN 5 and CDNA's first targets (wave64, 256 vector registers a lane, 102 scalar ones).
    /// 9.0.10 and 9.4.x share the instruction set but not the register file (vector and accumulation registers in one
    /// file, the descriptor's accumulation offset): they are another family's. Every number here was read from LLVM 18
    /// (2026-10-28): the descriptor's blocks and reserved registers from what <c>llvm-mc</c> wrote for each
    /// <c>.amdhsa_next_free_vgpr</c>, <c>.amdhsa_next_free_sgpr</c> and <c>.amdhsa_reserve_*</c>, the limits from what it
    /// refused (257 vector, 103 scalar registers), the waves per SIMD from clang's reported occupancy.
    /// </summary>
    private sealed partial class Gfx9 : IGcnTarget, IGcnInstructionSet
    {
        private static readonly GcnRegisterFile Wave64 = new(Vgprs: 256, VgprGranule: 4, Sgprs: 102, SgprGranule: 8, MaxWavesPerSimd: 10);

        // COMPUTE_PGM_RSRC1: DX10_CLAMP (bit 21) and IEEE_MODE (bit 23), on as LLVM sets them for compute kernels.
        private const uint ModeBits = (1u << 21) | (1u << 23);

        public string Family => "gfx9";

        public IReadOnlyList<int> WaveSizes { get; } = [64];

        public uint PaddingWord => 0xBF800000;   // s_nop 0

        public bool Handles(GcnIsaVersion isa) => isa is { Major: 9, Minor: 0 } && isa.Stepping != 0xA && TryMachine(isa, out _, out _, out _);

        public GcnRegisterFile Registers(GcnIsaVersion isa, int waveSize)
        {
            CheckWave(waveSize);
            return Wave64;
        }

        // VCC, the xnack mask and flat scratch sit at the top of a wave's scalar registers, in that order down: a kernel
        // holds 2 more for VCC alone, 4 while xnack is not off, 6 with flat scratch (LLVM's count, not a sum).
        public int SgprCount(GcnIsaVersion isa, int sgprs, bool vcc, bool flatScratch)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sgprs);
            int reserved = flatScratch ? 6 : isa.Xnack != GcnFeatureSetting.Off ? 4 : vcc ? 2 : 0;
            return sgprs + reserved;
        }

        public int WavesPerSimd(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize)
        {
            CheckCounts(vgprs, sgprs, waveSize);
            int byVgprs = vgprs == 0 ? Wave64.MaxWavesPerSimd : Math.Min(Wave64.MaxWavesPerSimd, Wave64.Vgprs / AlignUp(vgprs, Wave64.VgprGranule));
            int bySgprs = sgprs <= 80 ? 10 : sgprs <= 88 ? 9 : sgprs <= 100 ? 8 : 7;   // LLVM's steps for 800 scalar registers a SIMD
            return Math.Min(byVgprs, bySgprs);
        }

        public GcnDescriptorBits Descriptor(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize)
        {
            CheckCounts(vgprs, sgprs, waveSize);
            uint vgprBlocks = (uint)(AlignUp(Math.Max(1, vgprs), Wave64.VgprGranule) / Wave64.VgprGranule - 1);
            uint sgprBlocks = (uint)(AlignUp(Math.Max(1, sgprs), Wave64.SgprGranule) / Wave64.SgprGranule - 1);
            return new GcnDescriptorBits(vgprBlocks | (sgprBlocks << 6) | ModeBits, Rsrc3: 0, CodeProperties: 0);
        }

        public uint ElfFlags(GcnIsaVersion isa, int codeObjectVersion) => ElfFlagsOf(isa, codeObjectVersion);

        public GcnEncoding Encoding => GcnEncoding.Isa9;

        public int Ttmps => 16;

        public GcnInstruction? Find(GcnOp op) => (uint)op < (uint)ByOp.Length ? ByOp[(int)op] : null;

        public bool TryFind(ReadOnlySpan<char> mnemonic, [NotNullWhen(true)] out GcnInstruction? instruction) => ByMnemonic.TryGetValue(mnemonic, out instruction);

        // The instruction groups by version (LLVM 18's FeatureISAVersion9_0_*): 9.0.0, 9.0.2, 9.0.9 and 9.0.12 mix without
        // fusing; 9.0.4 fuses; 9.0.6 adds the dl instructions and the dot products; 9.0.8 the accumulating dot products, the
        // accumulation registers and the float atomics without a return. Read from the version, never from a card.
        public GcnFeatures Features(GcnIsaVersion isa) => isa is { Major: 9, Minor: 0 }
            ? isa.Stepping switch
            {
                0x0 or 0x2 or 0x9 or 0xC => GcnFeatures.MadMix,
                0x4 => GcnFeatures.FmaMix,
                0x6 => GcnFeatures.FmaMix | GcnFeatures.DeepLearning | GcnFeatures.Dot,
                0x8 => GcnFeatures.FmaMix | GcnFeatures.DeepLearning | GcnFeatures.Dot | GcnFeatures.DotAccumulate | GcnFeatures.AccumulationRegisters
                       | GcnFeatures.AtomicAddNoReturn,
                _ => GcnFeatures.None,
            }
            : GcnFeatures.None;

        // The scalar operand codes: s0-s101 are 0-101, then flat scratch (102), the xnack mask (104), VCC (106), the trap
        // registers (108-123), M0 (124) and EXEC (126); the apertures and the wave's id 235-239, VCCZ, EXECZ, SCC and LDS
        // direct 251-254 (llvm-mc's encodings). A global instruction's SADDR of 0x7F is off.
        public int Special(GcnSpecial register) => register switch
        {
            GcnSpecial.FlatScratch or GcnSpecial.FlatScratchLo => 102,
            GcnSpecial.FlatScratchHi => 103,
            GcnSpecial.XnackMask or GcnSpecial.XnackMaskLo => 104,
            GcnSpecial.XnackMaskHi => 105,
            GcnSpecial.Vcc or GcnSpecial.VccLo => 106,
            GcnSpecial.VccHi => 107,
            GcnSpecial.M0 => 124,
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
            GcnSpecial.Off => 0x7F,
            _ => throw new ArgumentOutOfRangeException(nameof(register), register, "Not a special register the gfx9 family encodes."),
        };

        // The hardware registers s_getreg_b32 and s_setreg_b32 name (llvm-mc's ids for this family).
        public int HardwareRegister(ReadOnlySpan<char> name) => name switch
        {
            "HW_REG_MODE" => 1,
            "HW_REG_STATUS" => 2,
            "HW_REG_TRAPSTS" => 3,
            "HW_REG_HW_ID" => 4,
            "HW_REG_GPR_ALLOC" => 5,
            "HW_REG_LDS_ALLOC" => 6,
            "HW_REG_IB_STS" => 7,
            "HW_REG_SH_MEM_BASES" => 15,
            _ => -1,
        };

        public int Ttmp(int index) => (uint)index < 16 ? 108 + index : throw new ArgumentOutOfRangeException(nameof(index), index, "gfx9 has ttmp0 to ttmp15.");

        // The wait counters (ISA 9; LLVM 18's SIInsertWaitcnts, checked with llc -run-pass=si-insert-waitcnts on gfx90c,
        // 2026-10-29): vmcnt counts global loads, stores and atomics alike, in order (a load, a store, then the load's use
        // waits vmcnt(1)); lgkmcnt counts LDS in order (an LDS read, an LDS write, then the read's use: lgkmcnt(1)) and
        // scalar memory out of order (its use waits lgkmcnt(0)); FLAT counts on both and may complete through either, so
        // its use waits both to 0; GDS counts on lgkmcnt and holds its data registers on expcnt (overwriting them waits
        // expcnt(0)). Stores need no wait before their data registers are written again (ISA 9 reads them at issue).
        public GcnCounting Counting(GcnMemoryClass memory) => memory switch
        {
            GcnMemoryClass.Vector => new(GcnCounter.Vm, InOrder: true),
            GcnMemoryClass.Flat => new(GcnCounter.Vm | GcnCounter.Lgkm, InOrder: false),
            GcnMemoryClass.Lds => new(GcnCounter.Lgkm, InOrder: true),
            GcnMemoryClass.Gds => new(GcnCounter.Lgkm, InOrder: true, DataLock: GcnCounter.Exp),
            GcnMemoryClass.Scalar => new(GcnCounter.Lgkm, InOrder: false),
            _ => default,
        };

        public int CounterLimit(GcnCounter counter) => counter switch
        {
            GcnCounter.Vm => 63,
            GcnCounter.Lgkm => 15,
            GcnCounter.Exp => 7,
            _ => throw new ArgumentOutOfRangeException(nameof(counter), counter, "One counter: vmcnt, lgkmcnt or expcnt."),
        };

        // vmcnt's low four bits at 3:0 and its high two at 15:14, expcnt at 6:4, lgkmcnt at 11:8.
        public ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt) => (ushort)((vmcnt & 0xF) | (expcnt & 7) << 4 | (lgkmcnt & 0xF) << 8 | (vmcnt >> 4 & 3) << 14);

        public (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits) => ((bits & 0xF) | (bits >> 14 & 3) << 4, bits >> 4 & 7, bits >> 8 & 0xF);

        // gfx9 has no automatic wait before a barrier (LLVM's hasAutoWaitcntBeforeBarrier and back-off barriers come with
        // gfx90a and gfx10): s_barrier after an LDS write takes lgkmcnt(0) first, after a store vmcnt(0).
        public bool BarrierWaitsForMemory => false;

        // The hazards, as LLVM 18's GCNHazardRecognizer inserts s_nop for them on gfx900, gfx906, gfx908 and gfx90c (llc
        // -run-pass=post-RA-hazard-rec on two-instruction MIR cases, 2026-10-29; the counts in wait states, s_nop n being
        // n + 1). The cases LLVM leaves alone on gfx9 are 0: a VALU's SGPR read by a VALU or an SALU, an SALU's SGPR read
        // by SMEM (Southern Islands only), an SALU's EXEC before DPP, a VALU's VGPR read by v_readlane.
        public int HazardWaitStates(GcnHazard hazard, GcnFeatureSetting xnack) => hazard switch
        {
            GcnHazard.VmemSgprAfterValu => 5,       // checkVMEMHazards: v_readfirstlane s2 or v_cmpx (EXEC), then global_load ... s[2:3]: s_nop 4
            GcnHazard.LaneSelectAfterValu => 4,     // checkRWLaneHazards: v_readfirstlane s4 (or v_cmp vcc), then v_readlane/v_writelane lane s4 (vcc_lo): s_nop 3
            GcnHazard.DivFmasVccAfterValu => 4,     // checkDivFMasHazards: v_div_scale ... vcc, then v_div_fmas: s_nop 3
            GcnHazard.DppVgprAfterValu => 2,        // checkDPPHazards: v_add v1, then v_mov_b32_dpp ..., v1: s_nop 1
            GcnHazard.DppAfterValuExec => 5,        // checkDPPHazards: v_cmpx (EXEC), then any DPP: s_nop 4
            GcnHazard.GetregAfterSetreg => 2,       // checkGetRegHazards: s_setreg(_imm32) hwreg(n), then s_getreg hwreg(n): s_nop 1
            GcnHazard.SetregAfterSetreg => 2,       // checkSetRegHazards: s_setreg hwreg(n) twice: s_nop 1
            GcnHazard.WideStoreDataAfterValu => 1,  // checkVALUHazards (has12DWordStoreHazard): global_store_dwordx3/x4, then a VALU writes a data VGPR: s_nop 0
            GcnHazard.M0AfterSalu => 1,             // checkReadM0Hazards: s_mov m0, then a GDS DS instruction or s_movrels: s_nop 0
            GcnHazard.SmemSgprAfterSalu => 0,       // checkSMRDHazards: Southern Islands only
            GcnHazard.SoftClause => xnack == GcnFeatureSetting.Off ? 0 : 1,   // checkSoftClauseHazards: with xnack on or any, s_nop 0 breaks the clause
            _ => 0,
        };

        public int SimdsPerComputeUnit => 4;

        private static void CheckWave(int waveSize)
        {
            if (waveSize != 64)
            {
                throw new ArgumentOutOfRangeException(nameof(waveSize), waveSize, "The gfx9 family runs waves of 64 lanes only.");
            }
        }

        private static void CheckCounts(int vgprs, int sgprs, int waveSize)
        {
            CheckWave(waveSize);
            ArgumentOutOfRangeException.ThrowIfNegative(vgprs);
            ArgumentOutOfRangeException.ThrowIfNegative(sgprs);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(vgprs, Wave64.Vgprs);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(sgprs, Wave64.Sgprs + 6);   // the most a kernel addresses, and the reserved ones
        }

        private static int AlignUp(int value, int granule) => (value + granule - 1) / granule * granule;
    }
}
