// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Gpu.Abstractions;

/// <summary>
/// The GCN instructions Idrak's assembler (<c>Hip/GcnAssembler*.cs</c>) knows, by meaning; each family gives its own
/// mnemonic, format, number and operands for each (<see cref="IGcnInstructionSet.Find"/>), or none where it lacks one.
/// Each name is its mnemonic as LLVM writes it, in Pascal case (<c>v_cvt_f32_ubyte0</c> is <see cref="VCvtF32Ubyte0"/>),
/// grouped by the format gfx9 encodes it in. New instructions are added here and to each family's table, with their
/// goldens (<c>tests/Idrak.Tests/data/gcn</c>).
/// </summary>
internal enum GcnOp : ushort
{
    // SOP1: one scalar source
    SMovB32,
    SMovB64,
    SCmovB32,
    SCmovB64,
    SNotB32,
    SNotB64,
    SWqmB64,
    SBrevB32,
    SBcnt1I32B32,
    SBcnt1I32B64,
    SFf1I32B32,
    SFf1I32B64,
    SFlbitI32B32,
    SFlbitI32B64,
    SFlbitI32,
    SSextI32I8,
    SSextI32I16,
    SBitset0B32,
    SBitset1B32,
    SAbsI32,
    SGetpcB64,
    SSetpcB64,
    SSwappcB64,
    SAndSaveexecB64,
    SOrSaveexecB64,
    SXorSaveexecB64,
    SAndn2SaveexecB64,
    SOrn2SaveexecB64,
    SAndn1SaveexecB64,
    SMovrelsB32,
    SMovreldB32,

    // SOP2: two scalar sources
    SAddU32,
    SAddcU32,
    SSubU32,
    SSubbU32,
    SAddI32,
    SSubI32,
    SMulI32,
    SMulHiU32,
    SMulHiI32,
    SMinU32,
    SMinI32,
    SMaxU32,
    SMaxI32,
    SLshlB32,
    SLshlB64,
    SLshrB32,
    SLshrB64,
    SAshrI32,
    SAshrI64,
    SAndB32,
    SAndB64,
    SOrB32,
    SOrB64,
    SXorB32,
    SXorB64,
    SAndn2B32,
    SAndn2B64,
    SOrn2B64,
    SNandB64,
    SNorB64,
    SXnorB64,
    SCselectB32,
    SCselectB64,
    SBfeU32,
    SBfeI32,
    SBfeU64,
    SBfmB32,
    SBfmB64,
    SAbsdiffI32,
    SLshl1AddU32,
    SLshl2AddU32,
    SLshl3AddU32,
    SLshl4AddU32,
    SPackLlB32B16,
    SPackLhB32B16,
    SPackHhB32B16,

    // SOPK: a 16-bit constant
    SMovkI32,
    SCmovkI32,
    SAddkI32,
    SMulkI32,
    SCmpkEqU32,
    SCmpkLgU32,
    SCmpkGtU32,
    SCmpkLtI32,
    SCmpkGeI32,
    SGetregB32,
    SSetregB32,
    SSetregImm32B32,

    // SOPC: scalar compares
    SCmpEqU32,
    SCmpLgU32,
    SCmpGtU32,
    SCmpGeU32,
    SCmpLtU32,
    SCmpLeU32,
    SCmpEqI32,
    SCmpLgI32,
    SCmpGtI32,
    SCmpGeI32,
    SCmpLtI32,
    SCmpLeI32,
    SCmpEqU64,
    SCmpLgU64,
    SBitcmp0B32,
    SBitcmp1B32,
    SBitcmp1B64,
    SSetGprIdxOn,

    // SOPP: program control
    SSetGprIdxOff,
    SNop,
    SWaitcnt,
    SBarrier,
    SSleep,
    SSetprio,
    SSethalt,
    SIcacheInv,
    STrap,
    SBranch,
    SCbranchScc0,
    SCbranchScc1,
    SCbranchVccz,
    SCbranchVccnz,
    SCbranchExecz,
    SCbranchExecnz,
    SEndpgm,

    // SMEM: scalar memory
    SLoadDword,
    SLoadDwordx2,
    SLoadDwordx4,
    SLoadDwordx8,
    SLoadDwordx16,
    SBufferLoadDword,
    SBufferLoadDwordx2,
    SBufferLoadDwordx4,
    SStoreDword,
    SStoreDwordx2,
    SDcacheWb,
    SDcacheInv,
    SMemtime,
    SMemrealtime,

    // VOP1: one vector source
    VMovB32,
    VNop,
    VReadfirstlaneB32,
    VCvtF32I32,
    VCvtF32U32,
    VCvtU32F32,
    VCvtI32F32,
    VCvtF16F32,
    VCvtF32F16,
    VCvtRpiI32F32,
    VCvtFlrI32F32,
    VCvtF32F64,
    VCvtF64F32,
    VCvtI32F64,
    VCvtF64I32,
    VCvtF64U32,
    VCvtU32F64,
    VCvtF32Ubyte0,
    VCvtF32Ubyte1,
    VCvtF32Ubyte2,
    VCvtF32Ubyte3,
    VCvtF16U16,
    VCvtF16I16,
    VCvtU16F16,
    VCvtI16F16,
    VCvtOffF32I4,
    VCvtNormI16F16,
    VFractF32,
    VTruncF32,
    VCeilF32,
    VRndneF32,
    VFloorF32,
    VExpF32,
    VLogF32,
    VRcpF32,
    VRcpIflagF32,
    VRsqF32,
    VSqrtF32,
    VSinF32,
    VCosF32,
    VExpF16,
    VLogF16,
    VRcpF16,
    VRsqF16,
    VSqrtF16,
    VRcpF64,
    VRsqF64,
    VSqrtF64,
    VFrexpExpI32F32,
    VFrexpMantF32,
    VFrexpExpI32F64,
    VNotB32,
    VBfrevB32,
    VFfbhU32,
    VFfblB32,
    VFfbhI32,
    VSwapB32,

    // VOP2: two vector sources
    VAddF32,
    VSubF32,
    VSubrevF32,
    VMulF32,
    VMulLegacyF32,
    VMacF32,
    VMadmkF32,
    VMadakF32,
    VMaxF32,
    VMinF32,
    VMaxI32,
    VMinI32,
    VMaxU32,
    VMinU32,
    VMulI32I24,
    VMulU32U24,
    VMulHiU32U24,
    VMulHiI32I24,
    VLshrrevB32,
    VAshrrevI32,
    VLshlrevB32,
    VLshlrevB16,
    VAndB32,
    VOrB32,
    VXorB32,
    VAddU32,
    VSubU32,
    VSubrevU32,
    VAddCoU32,
    VAddcCoU32,
    VSubCoU32,
    VSubbCoU32,
    VSubrevCoU32,
    VCndmaskB32,
    VAddF16,
    VSubF16,
    VMulF16,
    VMacF16,
    VMaxF16,
    VMinF16,
    VLdexpF16,
    VAddU16,
    VMulLoU16,
    VFmacF32,
    VXnorB32,
    VDot2cF32F16,
    VDot4cI32I8,

    // VOPC: vector compares
    VCmpLtF32,
    VCmpEqF32,
    VCmpLeF32,
    VCmpGtF32,
    VCmpLgF32,
    VCmpGeF32,
    VCmpOF32,
    VCmpUF32,
    VCmpNeqF32,
    VCmpNltF32,
    VCmpEqU32,
    VCmpNeU32,
    VCmpLtU32,
    VCmpGtU32,
    VCmpLeU32,
    VCmpGeU32,
    VCmpLtI32,
    VCmpGtI32,
    VCmpEqI32,
    VCmpNeI32,
    VCmpEqU64,
    VCmpLtU64,
    VCmpLtF64,
    VCmpLtF16,
    VCmpClassF32,
    VCmpxLtU32,
    VCmpxGtU32,
    VCmpxEqU32,
    VCmpxLtF32,

    // VOP3: three sources (VOP3-only)
    VLdexpF32,
    VFmaF32,
    VMadF32,
    VMadLegacyF32,
    VFmaF64,
    VFmaF16,
    VMadF16,
    VAddF64,
    VMulF64,
    VMaxF64,
    VLdexpF64,
    VMadU32U24,
    VMadI32I24,
    VMadU64U32,
    VMadI64I32,
    VMulLoU32,
    VMulHiU32,
    VMulHiI32,
    VAdd3U32,
    VLshlAddU32,
    VAddLshlU32,
    VLshlOrB32,
    VAndOrB32,
    VOr3B32,
    VXadU32,
    VLshlrevB64,
    VLshrrevB64,
    VAshrrevI64,
    VBfeU32,
    VBfeI32,
    VBfiB32,
    VBfmB32,
    VAlignbitB32,
    VAlignbyteB32,
    VPermB32,
    VBcntU32B32,
    VMbcntLoU32B32,
    VMbcntHiU32B32,
    VMed3F32,
    VMed3I32,
    VMed3U32,
    VMax3F32,
    VMin3F32,
    VMax3U32,
    VMin3I32,
    VSadU32,
    VCvtPkrtzF16F32,
    VCvtPknormI16F32,
    VCvtPkU16U32,
    VCvtPkI16I32,
    VCvtPkU8F32,
    VPackB32F16,
    VDivScaleF32,
    VDivFmasF32,
    VDivFixupF32,
    VDivScaleF64,
    VDivFmasF64,
    VDivFixupF64,
    VDivFixupF16,
    VTrigPreopF64,
    VReadlaneB32,
    VWritelaneB32,
    VMadU16,

    // VOP3P: packed and mixed precision
    VMadMixloF16,
    VPkAddF16,
    VPkMulF16,
    VPkFmaF16,
    VPkMaxF16,
    VPkMinF16,
    VPkAddU16,
    VPkAddI16,
    VPkSubI16,
    VPkMulLoU16,
    VPkMadU16,
    VPkLshlrevB16,
    VPkAshrrevI16,
    VMadMixF32,
    VFmaMixF32,
    VFmaMixloF16,
    VFmaMixhiF16,
    VDot2F32F16,
    VDot2I32I16,
    VDot2U32U16,
    VDot4I32I8,
    VDot4U32U8,
    VDot8I32I4,
    VDot8U32U4,
    VAccvgprReadB32,
    VAccvgprWriteB32,

    // DS: the workgroup's shared memory
    DsReadB32,
    DsReadB64,
    DsReadB128,
    DsRead2B32,
    DsRead2st64B32,
    DsRead2B64,
    DsReadU8,
    DsReadI8,
    DsReadU16,
    DsReadI16,
    DsReadU16D16,
    DsReadU16D16Hi,
    DsWriteB32,
    DsWriteB64,
    DsWriteB128,
    DsWrite2B32,
    DsWrite2st64B32,
    DsWrite2B64,
    DsWriteB8,
    DsWriteB16,
    DsWriteB8D16Hi,
    DsWriteB16D16Hi,
    DsAddU32,
    DsAddRtnU32,
    DsAddF32,
    DsAddRtnF32,
    DsSubU32,
    DsMaxF32,
    DsMinF32,
    DsMaxU32,
    DsAddU64,
    DsCmpstRtnB32,
    DsWrxchgRtnB32,
    DsBpermuteB32,
    DsPermuteB32,
    DsSwizzleB32,
    DsAppend,
    DsConsume,

    // GLOBAL: global memory
    GlobalLoadDword,
    GlobalLoadDwordx2,
    GlobalLoadDwordx3,
    GlobalLoadDwordx4,
    GlobalLoadUbyte,
    GlobalLoadSbyte,
    GlobalLoadUshort,
    GlobalLoadSshort,
    GlobalLoadShortD16,
    GlobalLoadShortD16Hi,
    GlobalLoadUbyteD16,
    GlobalLoadUbyteD16Hi,
    GlobalStoreDword,
    GlobalStoreDwordx2,
    GlobalStoreDwordx3,
    GlobalStoreDwordx4,
    GlobalStoreByte,
    GlobalStoreByteD16Hi,
    GlobalStoreShort,
    GlobalStoreShortD16Hi,
    GlobalAtomicAdd,
    GlobalAtomicUmax,
    GlobalAtomicCmpswap,
    GlobalAtomicAddX2,
    GlobalAtomicAddF32,
    GlobalAtomicPkAddF16,

    // FLAT: memory in any segment
    FlatLoadDword,
    FlatLoadDwordx2,
    FlatLoadDwordx4,
    FlatLoadUbyte,
    FlatLoadSbyte,
    FlatLoadUshort,
    FlatLoadSshort,
    FlatStoreDword,
    FlatStoreDwordx2,
    FlatStoreDwordx4,
    FlatStoreByte,
    FlatStoreShort,
    FlatAtomicAdd,
}

/// <summary>The instruction formats: how an instruction's word (or words) is laid out.</summary>
internal enum GcnFormat : byte
{
    /// <summary>No format: the family has no such instruction.</summary>
    None,

    /// <summary>Scalar, one source.</summary>
    Sop1,

    /// <summary>Scalar, two sources.</summary>
    Sop2,

    /// <summary>Scalar, a 16-bit constant.</summary>
    Sopk,

    /// <summary>Scalar compare, into SCC.</summary>
    Sopc,

    /// <summary>Scalar program control: branches, waits, the end.</summary>
    Sopp,

    /// <summary>Scalar memory (two dwords).</summary>
    Smem,

    /// <summary>Vector, one source (the 32-bit form; it has DPP, SDWA and VOP3 forms where the instruction allows).</summary>
    Vop1,

    /// <summary>Vector, two sources (the second a vector register in the 32-bit form).</summary>
    Vop2,

    /// <summary>Vector compare, into VCC (the 32-bit form) or any scalar pair (its VOP3 form).</summary>
    Vopc,

    /// <summary>Vector, up to three sources and modifiers (two dwords): the instructions that have no 32-bit form.</summary>
    Vop3,

    /// <summary>Vector, packed 16-bit pairs, mixed precision and dot products (two dwords).</summary>
    Vop3p,

    /// <summary>The workgroup's shared memory (LDS) and the wave's lane exchanges (two dwords).</summary>
    Ds,

    /// <summary>Memory through a 64-bit address in any segment (two dwords).</summary>
    Flat,

    /// <summary>Global memory (two dwords).</summary>
    Global,
}

/// <summary>Which word layouts a family's formats follow (the bit fields of each <see cref="GcnFormat"/>).</summary>
internal enum GcnEncoding : byte
{
    /// <summary>ISA version 9's (GCN 5 and CDNA's): among others a 20-bit SMEM offset and 13-bit signed global offsets.</summary>
    Isa9,
}

/// <summary>
/// Instruction groups only some versions of a family have, decided by the version a device reports (never its name):
/// <see cref="IGcnInstructionSet.Features"/> says which a version has, <see cref="GcnInstruction.Needs"/> which an
/// instruction needs. The names follow LLVM's subtarget features.
/// </summary>
[Flags]
internal enum GcnFeatures : ushort
{
    /// <summary>Every version of the family has it.</summary>
    None = 0,

    /// <summary>The mixed-precision multiply-adds that do not fuse: v_mad_mix_f32, v_mad_mixlo_f16 (LLVM's mad-mix-insts).</summary>
    MadMix = 1 << 0,

    /// <summary>The fused mixed-precision multiply-adds: v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 (fma-mix-insts).</summary>
    FmaMix = 1 << 1,

    /// <summary>v_fmac_f32 and v_xnor_b32 (LLVM's dl-insts).</summary>
    DeepLearning = 1 << 2,

    /// <summary>The packed dot products v_dot2_*, v_dot4_* and v_dot8_* (dot1 to dot7-insts).</summary>
    Dot = 1 << 3,

    /// <summary>The dot products that accumulate into their destination: v_dot2c_f32_f16, v_dot4c_i32_i8.</summary>
    DotAccumulate = 1 << 4,

    /// <summary>The accumulation registers (a0-a255) and their moves, v_accvgpr_read_b32 and v_accvgpr_write_b32 (mai-insts).</summary>
    AccumulationRegisters = 1 << 5,

    /// <summary>global_atomic_add_f32 and global_atomic_pk_add_f16 without a returned value (atomic-fadd-no-rtn-insts).</summary>
    AtomicAddNoReturn = 1 << 6,
}

/// <summary>What an instruction allows beyond its operands: its other encodings, its modifiers, its implicit reads.</summary>
[Flags]
internal enum GcnInstructionFlags : uint
{
    /// <summary>Nothing beyond its operands.</summary>
    None = 0,

    /// <summary>A VOP1, VOP2 or VOPC instruction with a VOP3 form (_e64); without it, it is written without a suffix.</summary>
    E64 = 1u << 0,

    /// <summary>The result may be clamped (clamp).</summary>
    Clamp = 1u << 1,

    /// <summary>The result may be scaled (mul:2, mul:4, div:2).</summary>
    Omod = 1u << 2,

    /// <summary>The 16-bit halves of the sources and the result may be chosen (op_sel; op_sel_hi too in VOP3P).</summary>
    OpSel = 1u << 3,

    /// <summary>A VOP3P instruction whose halves may be negated (neg_lo, neg_hi).</summary>
    NegLoHi = 1u << 4,

    /// <summary>It has a DPP form (_dpp): src0 read from another lane.</summary>
    Dpp = 1u << 5,

    /// <summary>It has an SDWA form (_sdwa): bytes and words of the sources and the result chosen.</summary>
    Sdwa = 1u << 6,

    /// <summary>It reads VCC without naming it (v_div_fmas): a read of the constant bus.</summary>
    ReadsVcc = 1u << 7,

    /// <summary>A mixed-precision VOP3P instruction: op_sel_hi chooses each source's precision (0 by default), and its sources take -, | |.</summary>
    Mix = 1u << 8,

    /// <summary>Source 0 may be negated (-v1, neg(v1)).</summary>
    Neg0 = 1u << 9,

    /// <summary>Source 0's absolute value may be taken (|v1|, abs(v1)).</summary>
    Abs0 = 1u << 10,

    /// <summary>Source 1 may be negated.</summary>
    Neg1 = 1u << 11,

    /// <summary>Source 1's absolute value may be taken.</summary>
    Abs1 = 1u << 12,

    /// <summary>Source 2 may be negated.</summary>
    Neg2 = 1u << 13,

    /// <summary>Source 2's absolute value may be taken.</summary>
    Abs2 = 1u << 14,

    /// <summary>A DS instruction with two 8-bit offsets (offset0, offset1) instead of one 16-bit offset.</summary>
    TwoOffsets = 1u << 15,

    /// <summary>A memory atomic: with glc it returns the old value into a destination written first; without, it has none.</summary>
    Atomic = 1u << 16,

    /// <summary>An atomic without the returning form (glc).</summary>
    NoReturn = 1u << 17,
}

/// <summary>What an operand of an instruction is, in the order the instruction is written.</summary>
internal enum GcnRole : byte
{
    /// <summary>No operand.</summary>
    None,

    /// <summary>A scalar destination: scalar registers or a special register.</summary>
    ScalarDestination,

    /// <summary>A scalar source: scalar registers, a special register or a constant.</summary>
    ScalarSource,

    /// <summary>A 16-bit constant (SOPK's and SOPP's simm16).</summary>
    Immediate16,

    /// <summary>A hardware register field: hwreg(name, offset, size).</summary>
    HardwareRegister,

    /// <summary>A 32-bit constant always written after the instruction.</summary>
    Literal32,

    /// <summary>The operands indexed by M0: gpr_idx(SRC0, SRC1, SRC2, DST).</summary>
    GprIndex,

    /// <summary>The counts s_waitcnt waits for: vmcnt(n) expcnt(n) lgkmcnt(n), or their 16 bits.</summary>
    WaitCounts,

    /// <summary>A branch's target: a label, or the 16-bit count of dwords.</summary>
    Target,

    /// <summary>SMEM's base address: a scalar register pair, or quad for a buffer.</summary>
    MemoryBase,

    /// <summary>SMEM's offset: bytes (0 to 0xFFFFF) or a scalar register.</summary>
    MemoryOffset,

    /// <summary>A vector destination.</summary>
    VectorDestination,

    /// <summary>A scalar destination of a vector instruction, written in its vector destination field (v_readlane_b32).</summary>
    ScalarResult,

    /// <summary>A scalar register pair a vector instruction writes per lane: the carry out, or a compare's mask (VCC in the 32-bit form).</summary>
    LaneMask,

    /// <summary>A vector instruction's source: a register of either file, a special register or a constant.</summary>
    Source,

    /// <summary>A source that is a vector register only.</summary>
    VectorSource,

    /// <summary>A source that is a scalar register, a special register or a constant (a lane select).</summary>
    ScalarOnlySource,

    /// <summary>A source that is a vector register or an inline constant.</summary>
    VectorOrInline,

    /// <summary>A scalar register pair a vector instruction reads per lane: a carry in, or v_cndmask's mask (VCC in the 32-bit form).</summary>
    LaneMaskSource,

    /// <summary>A 32-bit constant in a vector instruction's literal (v_madmk_f32's and v_madak_f32's K).</summary>
    Constant32,

    /// <summary>A memory instruction's vector address.</summary>
    Address,

    /// <summary>A memory instruction's first data.</summary>
    Data0,

    /// <summary>A DS instruction's second data.</summary>
    Data1,

    /// <summary>A global instruction's scalar address: a scalar register pair, or off.</summary>
    ScalarAddress,

    /// <summary>An accumulation register destination.</summary>
    AccumulationDestination,

    /// <summary>An accumulation register source.</summary>
    AccumulationSource,
}

/// <summary>
/// An instruction's operands, in the order they are written: each one's role and its width in 32-bit registers (up to
/// 6 operands, packed in 64 bits). Made from a short text, one token an operand: the role's letter, then the width
/// (<c>"d2 s2 s1"</c>: a 64-bit destination, a 64-bit and a 32-bit source).
/// </summary>
internal readonly struct GcnSignature
{
    /// <summary>The most operands an instruction has.</summary>
    public const int MaxOperands = 6;

    private readonly ulong _packed;

    private GcnSignature(ulong packed, int count)
    {
        _packed = packed;
        Count = count;
    }

    /// <summary>How many operands.</summary>
    public int Count { get; }

    /// <summary>Operand <paramref name="index"/>'s role.</summary>
    public GcnRole Role(int index) => (GcnRole)(_packed >> (10 * index) & 0x1F);

    /// <summary>Operand <paramref name="index"/>'s width in 32-bit registers (1 for constants and labels).</summary>
    public int Width(int index) => (int)(_packed >> (10 * index + 5) & 0x1F);

    /// <summary>The index of the first operand of <paramref name="role"/>, or -1.</summary>
    public int IndexOf(GcnRole role)
    {
        for (int i = 0; i < Count; i++)
        {
            if (Role(i) == role)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// The operands of <paramref name="format"/>'s instruction from their text: a token an operand, a letter and a width.
    /// Scalar formats: d destination, s source, k 16-bit constant, h hwreg, l literal, g gpr_idx, w wait counts, j branch
    /// target; SMEM: d or s data, b base, o offset. Vector formats: v destination, D scalar result, c lane mask out, x
    /// source, y vector source, z scalar source, u vector or inline source, i lane mask in, k K literal, A and a
    /// accumulation registers. Memory: v destination, r address, e and f data, S scalar address.
    /// </summary>
    /// <exception cref="ArgumentException">A token is not one of the format's.</exception>
    public static GcnSignature Parse(GcnFormat format, string text)
    {
        ulong packed = 0;
        int count = 0;
        bool scalar = format is GcnFormat.Sop1 or GcnFormat.Sop2 or GcnFormat.Sopk or GcnFormat.Sopc or GcnFormat.Sopp or GcnFormat.Smem;
        bool memory = format is GcnFormat.Ds or GcnFormat.Flat or GcnFormat.Global;
        foreach (var token in text.AsSpan().Split(' '))
        {
            ReadOnlySpan<char> item = text.AsSpan()[token];
            if (item.IsEmpty)
            {
                continue;
            }

            var role = (item[0], scalar, memory) switch
            {
                ('d', true, _) => GcnRole.ScalarDestination,
                ('s', true, _) => GcnRole.ScalarSource,
                ('k', true, _) => GcnRole.Immediate16,
                ('h', true, _) => GcnRole.HardwareRegister,
                ('l', true, _) => GcnRole.Literal32,
                ('g', true, _) => GcnRole.GprIndex,
                ('w', true, _) => GcnRole.WaitCounts,
                ('j', true, _) => GcnRole.Target,
                ('b', true, _) => GcnRole.MemoryBase,
                ('o', true, _) => GcnRole.MemoryOffset,
                ('v', false, false) => GcnRole.VectorDestination,
                ('D', false, false) => GcnRole.ScalarResult,
                ('c', false, false) => GcnRole.LaneMask,
                ('x', false, false) => GcnRole.Source,
                ('y', false, false) => GcnRole.VectorSource,
                ('z', false, false) => GcnRole.ScalarOnlySource,
                ('u', false, false) => GcnRole.VectorOrInline,
                ('i', false, false) => GcnRole.LaneMaskSource,
                ('k', false, false) => GcnRole.Constant32,
                ('A', false, false) => GcnRole.AccumulationDestination,
                ('a', false, false) => GcnRole.AccumulationSource,
                ('v', false, true) => GcnRole.VectorDestination,
                ('r', false, true) => GcnRole.Address,
                ('e', false, true) => GcnRole.Data0,
                ('f', false, true) => GcnRole.Data1,
                ('S', false, true) => GcnRole.ScalarAddress,
                _ => throw new ArgumentException($"'{item}' in \"{text}\" is not an operand of a {format} instruction.", nameof(text)),
            };
            int width = item.Length > 1 ? int.Parse(item[1..], System.Globalization.CultureInfo.InvariantCulture) : 1;
            if (count == MaxOperands || width is < 1 or > 16)
            {
                throw new ArgumentException($"\"{text}\": at most {MaxOperands} operands of 1 to 16 registers.", nameof(text));
            }

            packed |= ((ulong)role | (ulong)width << 5) << (10 * count);
            count++;
        }

        return new GcnSignature(packed, count);
    }
}

/// <summary>The type of an instruction's sources, which decides how a constant written for them is encoded.</summary>
internal enum GcnValueType : byte
{
    /// <summary>32 or 64 bits, integer or float (a float constant is taken as float32, or float64 for a 64-bit source).</summary>
    Bits,

    /// <summary>16-bit floats: a constant's inline values and literals are float16's.</summary>
    Half,

    /// <summary>16-bit integers.</summary>
    Short,

    /// <summary>64-bit floats: a constant's inline values are float64's.</summary>
    Double,
}

/// <summary>
/// One instruction of a family: its mnemonic, its format and number there, its operands, what it allows and the
/// features it needs. A family makes each once (<see cref="IGcnInstructionSet"/>); the assembler reads them, allocating
/// nothing.
/// </summary>
internal sealed class GcnInstruction
{
    /// <summary>An instruction of <paramref name="format"/> numbered <paramref name="code"/>, its operands as <see cref="GcnSignature.Parse"/> reads them.</summary>
    public GcnInstruction(GcnOp op, string mnemonic, GcnFormat format, int code, string operands, GcnInstructionFlags flags = GcnInstructionFlags.None,
        GcnFeatures needs = GcnFeatures.None)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mnemonic);
        ArgumentOutOfRangeException.ThrowIfNegative(code);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(code, 0x3FF);
        Op = op;
        Mnemonic = mnemonic;
        Format = format;
        Code = (ushort)code;
        Operands = GcnSignature.Parse(format, operands);
        Flags = flags;
        Needs = needs;
        SourceType = TypeOf(mnemonic);
    }

    /// <summary>What the instruction is.</summary>
    public GcnOp Op { get; }

    /// <summary>Its mnemonic as LLVM writes it, without an encoding suffix (_e32, _e64, _dpp, _sdwa).</summary>
    public string Mnemonic { get; }

    /// <summary>Its format: for VOP1, VOP2 and VOPC instructions the 32-bit one, from which the others follow.</summary>
    public GcnFormat Format { get; }

    /// <summary>Its number in that format.</summary>
    public ushort Code { get; }

    /// <summary>Its operands.</summary>
    public GcnSignature Operands { get; }

    /// <summary>What it allows beyond them.</summary>
    public GcnInstructionFlags Flags { get; }

    /// <summary>The features a version needs to have it.</summary>
    public GcnFeatures Needs { get; }

    /// <summary>Its sources' type, from the mnemonic's last type (v_cvt_f32_f16 reads float16s, v_add_u16 16-bit integers).</summary>
    public GcnValueType SourceType { get; }

    /// <summary>Whether it has <paramref name="flag"/>.</summary>
    public bool Has(GcnInstructionFlags flag) => (Flags & flag) == flag;

    /// <inheritdoc/>
    public override string ToString() => Mnemonic;

    // The last of the mnemonic's types (_f16, _u16, _f64 and so on) decides; scalar instructions read whole registers.
    private static GcnValueType TypeOf(string mnemonic)
    {
        if (!mnemonic.StartsWith("v_", StringComparison.Ordinal))
        {
            return GcnValueType.Bits;
        }

        for (int end = mnemonic.Length; end > 0;)
        {
            int start = mnemonic.LastIndexOf('_', end - 1);
            ReadOnlySpan<char> part = mnemonic.AsSpan(start + 1, end - start - 1);
            switch (part)
            {
                case "f16":
                    return GcnValueType.Half;
                case "i16" or "u16" or "b16":
                    return GcnValueType.Short;
                case "f64":
                    return GcnValueType.Double;
                case "f32" or "i32" or "u32" or "b32" or "i64" or "u64" or "b64" or "i8" or "u8" or "i4" or "u4" or "i24" or "u24":
                    return GcnValueType.Bits;
            }

            if (start < 0)
            {
                break;
            }

            end = start;
        }

        return GcnValueType.Bits;
    }
}

/// <summary>Registers and values a source or destination names by role; each family gives their operand codes.</summary>
internal enum GcnSpecial : byte
{
    /// <summary>The vector condition code, a register pair (vcc).</summary>
    Vcc,

    /// <summary>Its low half (vcc_lo).</summary>
    VccLo,

    /// <summary>Its high half (vcc_hi).</summary>
    VccHi,

    /// <summary>The execution mask, a register pair (exec).</summary>
    Exec,

    /// <summary>Its low half (exec_lo).</summary>
    ExecLo,

    /// <summary>Its high half (exec_hi).</summary>
    ExecHi,

    /// <summary>The memory descriptor register (m0).</summary>
    M0,

    /// <summary>No scalar address in a global instruction (off): the vector address is 64-bit.</summary>
    Off,

    /// <summary>The scratch base, a register pair (flat_scratch).</summary>
    FlatScratch,

    /// <summary>Its low half (flat_scratch_lo).</summary>
    FlatScratchLo,

    /// <summary>Its high half (flat_scratch_hi).</summary>
    FlatScratchHi,

    /// <summary>The lanes whose memory accesses replay, a register pair (xnack_mask).</summary>
    XnackMask,

    /// <summary>Its low half (xnack_mask_lo).</summary>
    XnackMaskLo,

    /// <summary>Its high half (xnack_mask_hi).</summary>
    XnackMaskHi,

    /// <summary>The scalar condition code as a source (src_scc).</summary>
    Scc,

    /// <summary>Whether VCC is zero, as a source (src_vccz).</summary>
    Vccz,

    /// <summary>Whether EXEC is zero, as a source (src_execz).</summary>
    Execz,

    /// <summary>The shared (LDS) aperture's base (src_shared_base).</summary>
    SharedBase,

    /// <summary>The shared aperture's limit (src_shared_limit).</summary>
    SharedLimit,

    /// <summary>The private (scratch) aperture's base (src_private_base).</summary>
    PrivateBase,

    /// <summary>The private aperture's limit (src_private_limit).</summary>
    PrivateLimit,

    /// <summary>The POPS exiting wave's id (src_pops_exiting_wave_id).</summary>
    PopsExitingWaveId,

    /// <summary>A value read from LDS at M0's address (src_lds_direct).</summary>
    LdsDirect,
}

/// <summary>A memory operation as a family's wait counters count it (<see cref="IGcnInstructionSet.Counting"/>).</summary>
internal enum GcnMemoryClass : byte
{
    /// <summary>Not a memory operation.</summary>
    None,

    /// <summary>Global memory through the vector memory unit (GLOBAL): loads, stores and atomics.</summary>
    Vector,

    /// <summary>FLAT: global memory or LDS through one address, which a lane decides at run time.</summary>
    Flat,

    /// <summary>The workgroup's shared memory (DS without gds), and the lane exchanges through it.</summary>
    Lds,

    /// <summary>The global data share (DS with gds).</summary>
    Gds,

    /// <summary>Scalar memory (SMEM): loads, stores, cache operations, the clocks.</summary>
    Scalar,
}

/// <summary>A family's wait counters (s_waitcnt's fields).</summary>
[Flags]
internal enum GcnCounter : byte
{
    /// <summary>No counter.</summary>
    None = 0,

    /// <summary>vmcnt: vector memory operations.</summary>
    Vm = 1,

    /// <summary>lgkmcnt: LDS, GDS, scalar memory and messages.</summary>
    Lgkm = 2,

    /// <summary>expcnt: exports, and the data registers of GDS operations until they are read.</summary>
    Exp = 4,
}

/// <summary>
/// How a family counts one class of memory operation: the counters it increments, whether they complete in the order
/// they were issued (else a wait for one of them waits for the counter to reach 0), and the counter that holds the
/// operation's data registers until they have been read (overwriting them waits for it).
/// </summary>
internal readonly record struct GcnCounting(GcnCounter Counters, bool InOrder, GcnCounter DataLock = GcnCounter.None);

/// <summary>
/// The cases where the hardware does not check a dependency itself and needs wait states (s_nop) between two
/// instructions; a family gives each its count (<see cref="IGcnInstructionSet.HazardWaitStates"/>), 0 where it has none.
/// Each case is named as LLVM's GCNHazardRecognizer checks it.
/// </summary>
internal enum GcnHazard : byte
{
    /// <summary>A VALU writes an SGPR (or VCC, EXEC), then a vector memory instruction reads it (its scalar address, or EXEC).</summary>
    VmemSgprAfterValu,

    /// <summary>A VALU writes an SGPR, then v_readlane_b32 or v_writelane_b32 reads it as its lane select.</summary>
    LaneSelectAfterValu,

    /// <summary>A VALU writes VCC, then v_div_fmas reads it.</summary>
    DivFmasVccAfterValu,

    /// <summary>A VALU writes a VGPR, then a DPP instruction reads it (a source, or the destination it partly keeps).</summary>
    DppVgprAfterValu,

    /// <summary>A VALU writes EXEC (v_cmpx), then a DPP instruction runs.</summary>
    DppAfterValuExec,

    /// <summary>s_setreg writes a hardware register, then s_getreg reads it.</summary>
    GetregAfterSetreg,

    /// <summary>s_setreg writes a hardware register, then another s_setreg writes it.</summary>
    SetregAfterSetreg,

    /// <summary>A FLAT or GLOBAL store of more than 64 bits, then a VALU writes one of its data VGPRs.</summary>
    WideStoreDataAfterValu,

    /// <summary>An SALU writes M0, then a GDS instruction or s_movrel reads it.</summary>
    M0AfterSalu,

    /// <summary>An SALU writes an SGPR, then an SMEM instruction reads it (its address or offset).</summary>
    SmemSgprAfterSalu,

    /// <summary>
    /// With xnack, consecutive memory instructions of one kind (SMEM, or vector memory) form a soft clause the hardware
    /// may replay: one that writes a register another of the clause reads, or a store after a load, breaks it.
    /// </summary>
    SoftClause,
}

/// <summary>
/// What Idrak's GCN assembler asks of a family (<see cref="IGcnTarget"/>) to encode its instructions: the word layouts
/// it follows, its instructions (by meaning and by mnemonic), the features each version has, and the operand codes of
/// its special registers; and, for kernels written with virtual registers, its wait counters and hazards. A family
/// without it is described by the code object writer but not assembled for. Internal: only the library's assembler
/// reads it, and an app's family (<see cref="GcnTargets.Register"/>) describes code objects without encoding instructions.
/// </summary>
internal interface IGcnInstructionSet
{
    /// <summary>The counters <paramref name="memory"/> increments, whether they complete in order, and its data lock.</summary>
    GcnCounting Counting(GcnMemoryClass memory);

    /// <summary>The most a counter holds (s_waitcnt's field for it, and the operations the hardware keeps in flight).</summary>
    int CounterLimit(GcnCounter counter);

    /// <summary>s_waitcnt's 16 bits for the counts (a counter at its limit is not waited on).</summary>
    ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt);

    /// <summary>The counts s_waitcnt's 16 bits wait for.</summary>
    (int Vmcnt, int Expcnt, int Lgkmcnt) WaitCounts(ushort bits);

    /// <summary>Whether s_barrier waits for the wave's memory operations itself; when not, a wait for every counter goes before it.</summary>
    bool BarrierWaitsForMemory { get; }

    /// <summary>The wait states <paramref name="hazard"/> needs on this family (0: none), with xnack as <paramref name="xnack"/> sets it.</summary>
    int HazardWaitStates(GcnHazard hazard, GcnFeatureSetting xnack);

    /// <summary>The SIMDs of a compute unit, over which a workgroup's waves are spread (what bounds a block's registers).</summary>
    int SimdsPerComputeUnit { get; }

    /// <summary>The word layouts the family's formats follow.</summary>
    GcnEncoding Encoding { get; }

    /// <summary>The family's <paramref name="op"/>; null when it has no such instruction.</summary>
    GcnInstruction? Find(GcnOp op);

    /// <summary>The instruction written <paramref name="mnemonic"/> (lowercase, without an encoding suffix); false when the family has none.</summary>
    bool TryFind(ReadOnlySpan<char> mnemonic, [NotNullWhen(true)] out GcnInstruction? instruction);

    /// <summary>The instruction groups <paramref name="isa"/> has beyond the family's common ones.</summary>
    GcnFeatures Features(GcnIsaVersion isa);

    /// <summary>The operand code of <paramref name="register"/> (the value a source or destination field holds).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The family has no such register.</exception>
    int Special(GcnSpecial register);

    /// <summary>The operand code of trap register <paramref name="index"/> (ttmp0 and up).</summary>
    int Ttmp(int index);

    /// <summary>How many trap registers (ttmp) there are.</summary>
    int Ttmps { get; }

    /// <summary>The id of the hardware register named <paramref name="name"/> (as in hwreg(HW_REG_MODE)); -1 when the family has none.</summary>
    int HardwareRegister(ReadOnlySpan<char> name);
}
