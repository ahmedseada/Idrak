// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
    private sealed class Gfx9 : IGcnTarget, IGcnInstructionSet
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

        // The opcode numbers of AMD's ISA reference for this family, each checked against llvm-mc's bytes for it
        // (tests/Idrak.Tests/data/gcn/gfx9/forms.golden). A switch over the enum: a jump table, nothing allocated.
        public bool TryOpcode(GcnOp op, out GcnOpcode opcode)
        {
            opcode = op switch
            {
                GcnOp.SLoadDwordx4 => new(GcnFormat.Smem, 0x02),
                GcnOp.SLshlB32 => new(GcnFormat.Sop2, 0x1C),
                GcnOp.SAndSaveexecB64 => new(GcnFormat.Sop1, 0x20),
                GcnOp.SWaitcnt => new(GcnFormat.Sopp, 0x0C),
                GcnOp.SCbranchExecz => new(GcnFormat.Sopp, 0x08),
                GcnOp.SEndpgm => new(GcnFormat.Sopp, 0x01),
                GcnOp.VAddU32 => new(GcnFormat.Vop2, 0x34),
                GcnOp.VCmpGtU32 => new(GcnFormat.Vopc, 0xCC),
                GcnOp.VLshlrevB32 => new(GcnFormat.Vop2, 0x12),
                GcnOp.VMovB32 => new(GcnFormat.Vop1, 0x01),
                GcnOp.GlobalStoreDword => new(GcnFormat.Global, 0x1C),
                _ => default,
            };
            return opcode.Format != GcnFormat.None;
        }

        // The scalar operand codes: s0-s101 are 0-101, then flat scratch (102), the xnack mask (104), VCC (106), the trap
        // registers, M0 (124) and EXEC (126); a global instruction's SADDR of 0x7F is off.
        public int Special(GcnSpecial register) => register switch
        {
            GcnSpecial.Vcc or GcnSpecial.VccLo => 106,
            GcnSpecial.VccHi => 107,
            GcnSpecial.M0 => 124,
            GcnSpecial.Exec or GcnSpecial.ExecLo => 126,
            GcnSpecial.ExecHi => 127,
            GcnSpecial.Off => 0x7F,
            _ => throw new ArgumentOutOfRangeException(nameof(register), register, "Not a special register the gfx9 family encodes."),
        };

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
