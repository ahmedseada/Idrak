// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Abstractions;

/// <summary>
/// The GCN instructions Idrak's assembler (<c>Hip/GcnAssembler*.cs</c>) knows, by meaning; each family gives its own
/// number and format for each (<see cref="IGcnInstructionSet.TryOpcode"/>), or none where it lacks one. The comment on each
/// is its mnemonic as LLVM writes it. New instructions are added here and to each family's table.
/// </summary>
internal enum GcnOp : ushort
{
    /// <summary>s_load_dwordx4: four dwords from constant memory into four scalar registers.</summary>
    SLoadDwordx4,

    /// <summary>s_lshl_b32: a 32-bit scalar shift left.</summary>
    SLshlB32,

    /// <summary>s_and_saveexec_b64: saves EXEC into a register pair and ANDs it with the source.</summary>
    SAndSaveexecB64,

    /// <summary>s_waitcnt: waits until the memory counters fall to the given counts.</summary>
    SWaitcnt,

    /// <summary>s_cbranch_execz: branches when EXEC is zero (no lane active).</summary>
    SCbranchExecz,

    /// <summary>s_endpgm: ends the wave.</summary>
    SEndpgm,

    /// <summary>v_add_u32: a 32-bit vector add without carry out.</summary>
    VAddU32,

    /// <summary>v_cmp_gt_u32: unsigned src0 &gt; src1 per lane, into VCC (the 32-bit form).</summary>
    VCmpGtU32,

    /// <summary>v_lshlrev_b32: vsrc1 shifted left by src0 (the reversed operands).</summary>
    VLshlrevB32,

    /// <summary>v_mov_b32: a 32-bit vector move.</summary>
    VMovB32,

    /// <summary>global_store_dword: one dword per lane to global memory.</summary>
    GlobalStoreDword,
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

    /// <summary>Vector, one source.</summary>
    Vop1,

    /// <summary>Vector, two sources (the second a vector register).</summary>
    Vop2,

    /// <summary>Vector compare, into VCC.</summary>
    Vopc,

    /// <summary>Vector, three sources and modifiers (two dwords).</summary>
    Vop3,

    /// <summary>Global memory (two dwords).</summary>
    Global,
}

/// <summary>Which word layouts a family's formats follow (the bit fields of each <see cref="GcnFormat"/>).</summary>
internal enum GcnEncoding : byte
{
    /// <summary>ISA version 9's (GCN 5 and CDNA's): among others a 20-bit SMEM offset and 13-bit signed global offsets.</summary>
    Isa9,
}

/// <summary>An instruction as a family encodes it: its format and its opcode number in that format.</summary>
internal readonly record struct GcnOpcode(GcnFormat Format, ushort Code);

/// <summary>Registers a source or destination names by role; each family gives their operand codes.</summary>
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
}

/// <summary>
/// What Idrak's GCN assembler asks of a family (<see cref="IGcnTarget"/>) to encode its instructions: the word layouts
/// it follows, each instruction's format and number, and the operand codes of its special registers. A family without
/// it is described by the code object writer but not assembled for.
/// </summary>
internal interface IGcnInstructionSet
{
    /// <summary>The word layouts the family's formats follow.</summary>
    GcnEncoding Encoding { get; }

    /// <summary>The format and number of <paramref name="op"/>; false when the family has no such instruction.</summary>
    bool TryOpcode(GcnOp op, out GcnOpcode opcode);

    /// <summary>The operand code of <paramref name="register"/> (the value a source or destination field holds).</summary>
    int Special(GcnSpecial register);
}
