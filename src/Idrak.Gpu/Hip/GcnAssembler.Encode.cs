// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.CompilerServices;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>What an operand is: a scalar, vector or accumulation register (or a run of them), a special register, a value.</summary>
internal enum GcnOperandKind : byte
{
    None,
    Sgpr,
    Vgpr,
    Special,

    /// <summary>A value the hardware holds itself (an inline constant: -16 to 64, ±0.5, ±1, ±2, ±4, 1/(2π)).</summary>
    Constant,

    /// <summary>A 32-bit value written after the instruction (a literal).</summary>
    Literal,

    /// <summary>Trap registers (ttmp0-ttmp15).</summary>
    Ttmp,

    /// <summary>Accumulation registers (a0-a255), where the version has them.</summary>
    Agpr,

    /// <summary>A label a branch names (<see cref="GcnLabel"/>).</summary>
    Label,

    /// <summary>A field's value written into the instruction's own bits (an offset, a simm16, a hwreg or gpr_idx field).</summary>
    Immediate,

    /// <summary>Virtual scalar registers (<c>%s3</c>, <c>%s[4:7]</c>) of a kernel's text, before the allocator gives them physical ones.</summary>
    VirtualSgpr,

    /// <summary>Virtual vector registers (<c>%v12</c>, <c>%v[0:3]</c>) of a kernel's text, before the allocator gives them physical ones.</summary>
    VirtualVgpr,
}

/// <summary>
/// One operand of a GCN instruction: a run of scalar, vector or accumulation registers (or, in a kernel's text, of
/// virtual ones the allocator places), a special register (VCC, EXEC, M0, off), a label, or a value: an inline
/// constant when the hardware holds it (the integers -16 to 64 and the floats ±0.5, ±1, ±2, ±4 and 1/(2π), by their
/// bits, as LLVM chooses), a literal otherwise, or an immediate written into the instruction's own field.
/// </summary>
internal readonly record struct GcnOperand
{
    private GcnOperand(GcnOperandKind kind, int index, int count, uint value)
    {
        Kind = kind;
        Index = index;
        Count = count;
        Value = value;
    }

    public GcnOperandKind Kind { get; }

    /// <summary>The first register; for a special register its <see cref="GcnSpecial"/>; for a constant its operand code; for a label its id.</summary>
    public int Index { get; }

    /// <summary>How many 32-bit registers it names (1 for a value; 0 for off and for the special values of any width).</summary>
    public int Count { get; }

    /// <summary>A value's 32 bits.</summary>
    public uint Value { get; }

    /// <summary>Scalar registers s[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand S(int index, int count = 1) => Registers(GcnOperandKind.Sgpr, index, count);

    /// <summary>Vector registers v[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand V(int index, int count = 1) => Registers(GcnOperandKind.Vgpr, index, count);

    /// <summary>Accumulation registers a[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand A(int index, int count = 1) => Registers(GcnOperandKind.Agpr, index, count);

    /// <summary>Trap registers ttmp[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand Ttmp(int index, int count = 1) => Registers(GcnOperandKind.Ttmp, index, count);

    public static GcnOperand Vcc => Special(GcnSpecial.Vcc);

    public static GcnOperand VccLo => Special(GcnSpecial.VccLo);

    public static GcnOperand VccHi => Special(GcnSpecial.VccHi);

    public static GcnOperand Exec => Special(GcnSpecial.Exec);

    public static GcnOperand ExecLo => Special(GcnSpecial.ExecLo);

    public static GcnOperand ExecHi => Special(GcnSpecial.ExecHi);

    public static GcnOperand M0 => Special(GcnSpecial.M0);

    /// <summary>No scalar address (a global instruction's SADDR): the vector address is a 64-bit pair.</summary>
    public static GcnOperand Off => Special(GcnSpecial.Off);

    /// <summary>A special register or value, of the width it has (pairs 2, halves and values 1; the apertures and off 0, any).</summary>
    public static GcnOperand Special(GcnSpecial register) => new(GcnOperandKind.Special, (int)register, register switch
    {
        GcnSpecial.Vcc or GcnSpecial.Exec or GcnSpecial.FlatScratch or GcnSpecial.XnackMask => 2,
        GcnSpecial.Off or GcnSpecial.SharedBase or GcnSpecial.SharedLimit or GcnSpecial.PrivateBase or GcnSpecial.PrivateLimit => 0,
        _ => 1,
    }, 0);

    /// <summary>A branch's target.</summary>
    public static GcnOperand Target(GcnLabel label) => new(GcnOperandKind.Label, label.Id, 1, 0);

    /// <summary>A value written into the instruction's own field (an offset, a simm16, a hwreg or gpr_idx field).</summary>
    public static GcnOperand Immediate(int value) => new(GcnOperandKind.Immediate, 0, 1, unchecked((uint)value));

    /// <summary>Virtual registers <paramref name="id"/> to id + count - 1 of a file (<see cref="GcnOperandKind.VirtualSgpr"/> or <see cref="GcnOperandKind.VirtualVgpr"/>).</summary>
    internal static GcnOperand Virtual(GcnOperandKind kind, int id, int count) => new(kind, id, count, 0);

    /// <summary>A 32-bit integer: inline from -16 to 64, a literal otherwise.</summary>
    public static GcnOperand Int(int value) => Bits(unchecked((uint)value));

    /// <summary>A float: inline for ±0.5, ±1, ±2, ±4 and 1/(2π), a literal otherwise.</summary>
    public static GcnOperand Float(float value) => Bits(BitConverter.SingleToUInt32Bits(value));

    /// <summary>32 bits: an inline constant when an integer from -16 to 64 or one of the inline floats has them, else a literal.</summary>
    public static GcnOperand Bits(uint bits)
    {
        int code = InlineCode(bits);
        return code >= 0 ? new(GcnOperandKind.Constant, code, 1, bits) : new(GcnOperandKind.Literal, LiteralCode, 1, bits);
    }

    /// <summary>An inline constant by its operand code (128-208, 240-248), holding <paramref name="bits"/> as its value.</summary>
    internal static GcnOperand Inline(int code, uint bits) => new(GcnOperandKind.Constant, code, 1, bits);

    /// <summary>A literal of <paramref name="bits"/>, whatever they are (a 64-bit or 16-bit source's literal).</summary>
    internal static GcnOperand LiteralOf(uint bits) => new(GcnOperandKind.Literal, LiteralCode, 1, bits);

    /// <summary>The operand code of a literal (the value follows the instruction's word).</summary>
    internal const int LiteralCode = 255;

    // The inline constants' operand codes: 128 + n for 0 to 64, 192 + n for -1 to -16, then the floats.
    internal static int InlineCode(uint bits)
    {
        int value = unchecked((int)bits);
        if (value is >= 0 and <= 64)
        {
            return 128 + value;
        }

        if (value is >= -16 and <= -1)
        {
            return 192 - value;
        }

        return bits switch
        {
            0x3F000000 => 240,                                                   // 0.5
            0xBF000000 => 241,                                                   // -0.5
            0x3F800000 => 242,                                                   // 1.0
            0xBF800000 => 243,                                                   // -1.0
            0x40000000 => 244,                                                   // 2.0
            0xC0000000 => 245,                                                   // -2.0
            0x40800000 => 246,                                                   // 4.0
            0xC0800000 => 247,                                                   // -4.0
            0x3E22F983 => 248,                                                   // 1/(2π)
            _ => -1,
        };
    }

    private static GcnOperand Registers(GcnOperandKind kind, int index, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        return new(kind, index, count, 0);
    }

    /// <summary>As LLVM writes it: s7, s[4:7], v0, a1, ttmp2, vcc, off, 2, 0x3fb8aa3b.</summary>
    public override string ToString() => Kind switch
    {
        GcnOperandKind.Sgpr or GcnOperandKind.Vgpr or GcnOperandKind.Agpr or GcnOperandKind.Ttmp when Count == 1 => $"{Prefix(Kind)}{Index}",
        GcnOperandKind.Sgpr or GcnOperandKind.Vgpr or GcnOperandKind.Agpr or GcnOperandKind.Ttmp => $"{Prefix(Kind)}[{Index}:{Index + Count - 1}]",
        GcnOperandKind.Special => SpecialName((GcnSpecial)Index),
        GcnOperandKind.Constant when Index < 209 => unchecked((int)Value).ToString(CultureInfo.InvariantCulture),
        GcnOperandKind.Constant => BitConverter.UInt32BitsToSingle(Value).ToString(CultureInfo.InvariantCulture),
        GcnOperandKind.Literal => string.Create(CultureInfo.InvariantCulture, $"0x{Value:x}"),
        GcnOperandKind.Label => string.Create(CultureInfo.InvariantCulture, $"label {Index}"),
        GcnOperandKind.Immediate => unchecked((int)Value).ToString(CultureInfo.InvariantCulture),
        GcnOperandKind.VirtualSgpr or GcnOperandKind.VirtualVgpr when Count == 1 => $"%{(Kind == GcnOperandKind.VirtualSgpr ? 's' : 'v')}{Index}",
        GcnOperandKind.VirtualSgpr or GcnOperandKind.VirtualVgpr => $"%{(Kind == GcnOperandKind.VirtualSgpr ? 's' : 'v')}[{Index}:{Index + Count - 1}]",
        _ => "(none)",
    };

    private static string Prefix(GcnOperandKind kind) => kind switch
    {
        GcnOperandKind.Sgpr => "s",
        GcnOperandKind.Vgpr => "v",
        GcnOperandKind.Agpr => "a",
        _ => "ttmp",
    };

    /// <summary>A special register's name as LLVM prints it.</summary>
    internal static string SpecialName(GcnSpecial register) => register switch
    {
        GcnSpecial.Vcc => "vcc",
        GcnSpecial.VccLo => "vcc_lo",
        GcnSpecial.VccHi => "vcc_hi",
        GcnSpecial.Exec => "exec",
        GcnSpecial.ExecLo => "exec_lo",
        GcnSpecial.ExecHi => "exec_hi",
        GcnSpecial.M0 => "m0",
        GcnSpecial.Off => "off",
        GcnSpecial.FlatScratch => "flat_scratch",
        GcnSpecial.FlatScratchLo => "flat_scratch_lo",
        GcnSpecial.FlatScratchHi => "flat_scratch_hi",
        GcnSpecial.XnackMask => "xnack_mask",
        GcnSpecial.XnackMaskLo => "xnack_mask_lo",
        GcnSpecial.XnackMaskHi => "xnack_mask_hi",
        GcnSpecial.Scc => "src_scc",
        GcnSpecial.Vccz => "src_vccz",
        GcnSpecial.Execz => "src_execz",
        GcnSpecial.SharedBase => "src_shared_base",
        GcnSpecial.SharedLimit => "src_shared_limit",
        GcnSpecial.PrivateBase => "src_private_base",
        GcnSpecial.PrivateLimit => "src_private_limit",
        GcnSpecial.PopsExitingWaveId => "src_pops_exiting_wave_id",
        GcnSpecial.LdsDirect => "src_lds_direct",
        _ => register.ToString(),
    };
}

/// <summary>A place in the code a branch can name, made by <see cref="GcnAssembler.NewLabel"/> and placed by <see cref="GcnAssembler.Bind"/>.</summary>
internal readonly record struct GcnLabel(int Id);

/// <summary>Which encoding of a VOP1, VOP2 or VOPC instruction is written: chosen by the suffix, or by its operands.</summary>
internal enum GcnVariant : byte
{
    /// <summary>The 32-bit form when the operands fit it, else VOP3; DPP or SDWA when their controls are given.</summary>
    Auto,

    /// <summary>_e32: the 32-bit form.</summary>
    E32,

    /// <summary>_e64: the VOP3 form.</summary>
    E64,

    /// <summary>_dpp: src0 read from another lane.</summary>
    Dpp,

    /// <summary>_sdwa: bytes and words of the sources and result chosen.</summary>
    Sdwa,
}

/// <summary>
/// A GCN instruction's text was refused: an unknown mnemonic or operand, an operand the instruction cannot take, a form
/// the version lacks. The message names the line and column (1-based) and the reason.
/// </summary>
internal sealed class GcnSyntaxException(int line, int column, string reason)
    : FormatException(string.Create(CultureInfo.InvariantCulture, $"GCN line {line}, column {column}: {reason}"))
{
    /// <summary>The line, from 1.</summary>
    public int Line { get; } = line;

    /// <summary>The column, from 1 (a byte offset in the line's UTF-8).</summary>
    public int Column { get; } = column;

    /// <summary>Why, without the place.</summary>
    public string Reason { get; } = reason;
}

/// <summary>
/// Idrak's GCN assembler (plan 35), what <c>ptxas</c> is to PTX: GCN instructions to the machine code a code object
/// holds. This part encodes: each format's words from an instruction (<see cref="GcnInstruction"/>, the family's) and its
/// operands and modifiers, given as text (<see cref="Assemble"/>, in <c>GcnAssembler.Parse.cs</c>) or through the typed
/// methods, branches to labels resolved at the end. The words are written into one pooled buffer (returned by
/// <see cref="Dispose"/>), so an instruction allocates nothing; <see cref="ToArray"/> gives the code in an array of its
/// exact size. ISA 9's layouts: SOP1, SOP2, SOPK, SOPC, SOPP, SMEM, VOP1, VOP2, VOPC and their VOP3, DPP and SDWA forms,
/// VOP3, VOP3P, DS, FLAT and GLOBAL.
/// </summary>
internal sealed partial class GcnAssembler : IDisposable
{
    private readonly IGcnInstructionSet _set;
    private readonly IGcnTarget _target;
    private readonly GcnIsaVersion? _isa;
    private readonly string _family;
    private readonly GcnFeatures _features;
    private readonly string _version;
    private byte[] _code;
    private int _length;

    // Each label's byte offset (-1 until bound), and each branch as (its byte offset, its label, its line), pooled like
    // the code. Named labels keep their names' UTF-8 bytes in one pooled buffer: (start, length, hash) a label.
    private int[] _labels;
    private int _labelCount;
    private int[] _branches;
    private int _branchCount;

    // The instruction being encoded: its definition, its form, its operands and modifiers (reused, nothing allocated).
    private GcnInstruction _ins = null!;
    private GcnVariant _variant;

    // The form the encoder chose (the suffix a printed instruction carries).
    private GcnVariant _encoded;
    private Operands _ops;
    private int _count;
    private int _negMask, _absMask, _sextMask;
    private Modifiers _m;

    // ISA 9's limits: scalar operand codes 0-101 are registers, vector and accumulation ones 256 a lane; one scalar value
    // (a scalar register, a special register or a literal) read by a vector instruction (the constant bus).
    private const int ScalarRegisters = 102, VectorRegisters = 256, ConstantBus = 1;

    /// <summary>
    /// An assembler for <paramref name="target"/>'s family, its buffer sized for <paramref name="bytes"/> of code (it grows
    /// past them). Without a version, only the instructions every version of the family has are taken.
    /// </summary>
    /// <exception cref="NotSupportedException">The family has no instruction encodings, or layouts this assembler does not write.</exception>
    public GcnAssembler(IGcnTarget target, int bytes = 256)
        : this(target, null, bytes)
    {
    }

    /// <summary>
    /// An assembler for <paramref name="isa"/>, a version <paramref name="target"/>'s family handles: the instructions
    /// only some versions have (<see cref="GcnFeatures"/>) are taken where this one has them, and refused by name elsewhere.
    /// </summary>
    /// <exception cref="ArgumentException">The family does not handle <paramref name="isa"/>.</exception>
    /// <exception cref="NotSupportedException">The family has no instruction encodings, or layouts this assembler does not write.</exception>
    public GcnAssembler(IGcnTarget target, GcnIsaVersion isa, int bytes = 256)
        : this(target, (GcnIsaVersion?)isa, bytes)
    {
    }

    private GcnAssembler(IGcnTarget target, GcnIsaVersion? isa, int bytes)
    {
        ArgumentNullException.ThrowIfNull(target);
        _target = target;
        _isa = isa;
        _family = target.Family;
        _set = target as IGcnInstructionSet
               ?? throw new NotSupportedException($"The GCN family {target.Family} has no instruction encodings (IGcnInstructionSet): Idrak's assembler cannot write code for it.");
        if (_set.Encoding != GcnEncoding.Isa9)
        {
            throw new NotSupportedException($"The GCN family {target.Family} uses the {_set.Encoding} layouts, which Idrak's assembler does not write yet.");
        }

        if (isa is { } version)
        {
            if (!target.Handles(version))
            {
                throw new ArgumentException($"The GCN family {target.Family} does not handle '{version}'.", nameof(isa));
            }

            _features = _set.Features(version);
            _version = $"ISA {version.Major}.{version.Minor}.{version.Stepping} ({version.Processor})";
        }
        else
        {
            _version = $"the {target.Family} family without a version";
        }

        _code = ArrayPool<byte>.Shared.Rent(Math.Max(64, bytes));
        _labels = ArrayPool<int>.Shared.Rent(32);
        _branches = ArrayPool<int>.Shared.Rent(48);
        _names = ArrayPool<byte>.Shared.Rent(256);
        _chars = ArrayPool<char>.Shared.Rent(CharBuffer);
        RentKernelBuffers();
    }

    /// <summary>Bytes of code written so far.</summary>
    public int Length => _length;

    /// <summary>The instruction groups the version given has (none without one).</summary>
    public GcnFeatures Features => _features;

    /// <summary>Forgets the code, the labels and the line count, keeping the buffers: the assembler starts again.</summary>
    public void Reset()
    {
        _length = 0;
        _labelCount = 0;
        _branchCount = 0;
        _namesLength = 0;
        _line = 0;
        _printLength = 0;
        ResetKernel();
    }

    /// <summary>A new label, placed later with <see cref="Bind"/>.</summary>
    public GcnLabel NewLabel() => new(AddLabel(0, 0, 0));

    /// <summary>Places <paramref name="label"/> at the next instruction.</summary>
    public void Bind(GcnLabel label)
    {
        if ((uint)label.Id >= (uint)_labelCount || _labels[LabelSlot * label.Id] >= 0)
        {
            throw new InvalidOperationException($"Label {label.Id} is not this assembler's, or is bound already.");
        }

        _labels[LabelSlot * label.Id] = _length;
    }

    /// <summary>SOP1: <paramref name="sdst"/> = op(<paramref name="ssrc0"/>).</summary>
    public void Sop1(GcnOp op, GcnOperand sdst, GcnOperand ssrc0) => Typed(op, GcnFormat.Sop1, sdst, ssrc0);

    /// <summary>SOP2: <paramref name="sdst"/> = op(<paramref name="ssrc0"/>, <paramref name="ssrc1"/>).</summary>
    public void Sop2(GcnOp op, GcnOperand sdst, GcnOperand ssrc0, GcnOperand ssrc1) => Typed(op, GcnFormat.Sop2, sdst, ssrc0, ssrc1);

    /// <summary>SOPP with its 16-bit field (a wait's counts, a branch's dword count, or nothing).</summary>
    public void Sopp(GcnOp op, ushort simm16 = 0)
    {
        var ins = Definition(op, GcnFormat.Sopp);
        if (ins.Operands.Count == 0)
        {
            Typed(op, GcnFormat.Sopp);
        }
        else
        {
            Typed(op, GcnFormat.Sopp, GcnOperand.Immediate(simm16));
        }
    }

    /// <summary>
    /// s_waitcnt: waits until at most <paramref name="vmcnt"/> vector memory operations (0-63),
    /// <paramref name="expcnt"/> exports (0-7) and <paramref name="lgkmcnt"/> LDS, GDS, constant and message operations
    /// (0-15) are outstanding; a counter left null is not waited on.
    /// </summary>
    public void Waitcnt(int? vmcnt = null, int? expcnt = null, int? lgkmcnt = null)
    {
        int vm = Count(vmcnt, 63, nameof(vmcnt)), exp = Count(expcnt, 7, nameof(expcnt)), lgkm = Count(lgkmcnt, 15, nameof(lgkmcnt));
        Sopp(GcnOp.SWaitcnt, WaitBits(vm, exp, lgkm));

        static int Count(int? count, int most, string name)
        {
            if (count is { } value)
            {
                ArgumentOutOfRangeException.ThrowIfNegative(value, name);
                ArgumentOutOfRangeException.ThrowIfGreaterThan(value, most, name);
                return value;
            }

            return most;
        }
    }

    /// <summary>A SOPP branch to <paramref name="target"/>, resolved by <see cref="ToArray"/>: the dwords from the next instruction.</summary>
    public void Branch(GcnOp op, GcnLabel target)
    {
        if ((uint)target.Id >= (uint)_labelCount)
        {
            throw new ArgumentException($"{op}: label {target.Id} is not this assembler's.", nameof(target));
        }

        Typed(op, GcnFormat.Sopp, GcnOperand.Target(target));
    }

    /// <summary>
    /// SMEM with an immediate offset: <paramref name="sdata"/> = the dwords at <paramref name="sbase"/> (a register pair
    /// holding an address) + <paramref name="offset"/> bytes (0 to 0xFFFFF).
    /// </summary>
    public void Smem(GcnOp op, GcnOperand sdata, GcnOperand sbase, int offset) => Typed(op, GcnFormat.Smem, sdata, sbase, GcnOperand.Immediate(offset));

    /// <summary>VOP1: <paramref name="vdst"/> = op(<paramref name="src0"/>), the 32-bit form (_e32).</summary>
    public void Vop1(GcnOp op, GcnOperand vdst, GcnOperand src0) => Typed(op, GcnFormat.Vop1, vdst, src0);

    /// <summary>VOP2: <paramref name="vdst"/> = op(<paramref name="src0"/>, <paramref name="vsrc1"/>), the 32-bit form (_e32).</summary>
    public void Vop2(GcnOp op, GcnOperand vdst, GcnOperand src0, GcnOperand vsrc1) => Typed(op, GcnFormat.Vop2, vdst, src0, vsrc1);

    /// <summary>VOPC: VCC = op(<paramref name="src0"/>, <paramref name="vsrc1"/>) per lane, the 32-bit form (_e32).</summary>
    public void Vopc(GcnOp op, GcnOperand src0, GcnOperand vsrc1) => Typed(op, GcnFormat.Vopc, GcnOperand.Vcc, src0, vsrc1);

    /// <summary>
    /// A GLOBAL store: <paramref name="vdata"/> to the address <paramref name="saddr"/> (a scalar pair) + <paramref name="vaddr"/>
    /// (a 32-bit byte offset per lane) + <paramref name="offset"/>, or, with <see cref="GcnOperand.Off"/>, to
    /// <paramref name="vaddr"/> (a 64-bit address pair per lane) + <paramref name="offset"/> (-4096 to 4095 bytes).
    /// </summary>
    public void GlobalStore(GcnOp op, GcnOperand vaddr, GcnOperand vdata, GcnOperand saddr, int offset = 0, bool glc = false, bool slc = false)
    {
        Begin(Definition(op, GcnFormat.Global), GcnVariant.Auto, typed: true);
        _m.Offset = offset;
        _m.Glc = glc;
        _m.Slc = slc;
        Add(vaddr);
        Add(vdata);
        Add(saddr);
        Encode();
    }

    /// <summary>The code with every branch resolved, in an array of its exact size.</summary>
    /// <exception cref="InvalidOperationException">A branch names a label never bound, or one too far for 16 bits.</exception>
    public byte[] ToArray()
    {
        for (int i = 0; i < _branchCount; i++)
        {
            int at = _branches[BranchSlot * i], label = _branches[BranchSlot * i + 1], line = _branches[BranchSlot * i + 2];
            int target = _labels[LabelSlot * label];
            if (target < 0)
            {
                throw new InvalidOperationException(_labels[LabelSlot * label + 2] > 0
                    ? $"GCN line {line}: the branch names label '{LabelName(label)}', which is never defined."
                    : $"The branch at byte {at} names label {label}, which was never bound.");
            }

            int dwords = (target - (at + 4)) / 4;
            if (dwords is < short.MinValue or > short.MaxValue)
            {
                throw new InvalidOperationException($"The branch at byte {at} is {dwords} dwords from its label, past what 16 bits hold.");
            }

            BinaryPrimitives.WriteUInt16LittleEndian(_code.AsSpan(at), unchecked((ushort)dwords));
        }

        return _code.AsSpan(0, _length).ToArray();
    }

    /// <summary>Returns the pooled buffers; the assembler is not used after.</summary>
    public void Dispose()
    {
        Return(ref _code);
        Return(ref _labels);
        Return(ref _branches);
        Return(ref _names);
        Return(ref _chars);
        ReturnKernelBuffers();

        static void Return<T>(ref T[] array)
        {
            if (array.Length > 0)
            {
                ArrayPool<T>.Shared.Return(array);
                array = [];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The instruction being written: begun, its operands added, encoded.
    // ---------------------------------------------------------------------------------------------------------------

    // An instruction's operands, in written order.
    [InlineArray(GcnSignature.MaxOperands)]
    private struct Operands
    {
        private GcnOperand _first;
    }

    // An instruction's modifiers, as written (defaults: no offset, masks 0xF, SDWA's DWORD and UNUSED_PRESERVE).
    private struct Modifiers
    {
        public bool Clamp, Glc, Slc, Gds, BoundCtrl, Swizzle;
        public int Omod;
        public int OpSel, OpSelCount, OpSelHi, OpSelHiGiven, NegLo, NegHi;
        public int Offset, Offset0, Offset1;
        public int DppControl, RowMask, BankMask;
        public int DstSel, DstUnused, Src0Sel, Src1Sel;
        public bool SdwaGiven, DstGiven;

        public static Modifiers Default => new() { DppControl = -1, RowMask = 0xF, BankMask = 0xF, DstSel = 6, DstUnused = 2, Src0Sel = 6, Src1Sel = 6 };
    }

    private void Begin(GcnInstruction instruction, GcnVariant variant, bool typed)
    {
        _ins = instruction;
        _variant = variant;
        _encoded = variant;
        _count = 0;
        _negMask = _absMask = _sextMask = 0;
        _m = Modifiers.Default;
        if (typed)
        {
            _parsing = false;
        }
    }

    private void Add(GcnOperand operand) => _ops[_count++] = operand;

    private void Typed(GcnOp op, GcnFormat format, params ReadOnlySpan<GcnOperand> operands)
    {
        Begin(Definition(op, format), format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc ? GcnVariant.E32 : GcnVariant.Auto, typed: true);
        foreach (var operand in operands)
        {
            Add(operand);
        }

        Encode();
    }

    // The family's op, which must be of the format the caller writes, in a version that has it.
    private GcnInstruction Definition(GcnOp op, GcnFormat format)
    {
        var ins = _set.Find(op) ?? throw new NotSupportedException($"The GCN family {_family} has no {op}.");
        if (ins.Format != format)
        {
            throw new ArgumentException($"{op} ({ins.Mnemonic}) is a {ins.Format} instruction in the {_family} family, not {format}.", nameof(op));
        }

        if ((ins.Needs & ~_features) != 0)
        {
            throw new NotSupportedException(Missing(ins));
        }

        return ins;
    }

    private string Missing(GcnInstruction ins) =>
        $"{ins.Mnemonic} needs {FeatureName(ins.Needs & ~_features)}, which {_version} does not have.";

    private static string FeatureName(GcnFeatures features) => features switch
    {
        GcnFeatures.MadMix => "the unfused mixed-precision instructions (mad-mix)",
        GcnFeatures.FmaMix => "the fused mixed-precision instructions (fma-mix)",
        GcnFeatures.DeepLearning => "the dl instructions (v_fmac_f32, v_xnor_b32)",
        GcnFeatures.Dot => "the packed dot products (dot-insts)",
        GcnFeatures.DotAccumulate => "the accumulating dot products (v_dot2c, v_dot4c)",
        GcnFeatures.AccumulationRegisters => "the accumulation registers (mai-insts)",
        GcnFeatures.AtomicAddNoReturn => "the float atomics without a return (atomic-fadd-no-rtn-insts)",
        _ => features.ToString(),
    };

    // Encodes the instruction begun, by its format.
    private void Encode()
    {
        var ins = _ins;
        if (_count != ins.Operands.Count)
        {
            throw Refuse(-1, $"takes {ins.Operands.Count} operand(s), not {_count}");
        }

        switch (ins.Format)
        {
            case GcnFormat.Sop1:
                EncodeSop1();
                break;
            case GcnFormat.Sop2:
                EncodeSop2();
                break;
            case GcnFormat.Sopk:
                EncodeSopk();
                break;
            case GcnFormat.Sopc:
                EncodeSopc();
                break;
            case GcnFormat.Sopp:
                EncodeSopp();
                break;
            case GcnFormat.Smem:
                EncodeSmem();
                break;
            case GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc or GcnFormat.Vop3 or GcnFormat.Vop3p:
                EncodeVector();
                break;
            case GcnFormat.Ds:
                EncodeDs();
                break;
            case GcnFormat.Flat or GcnFormat.Global:
                EncodeFlat();
                break;
            default:
                throw Refuse(-1, $"the {ins.Format} format is not one this assembler writes");
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Scalar formats.
    // ---------------------------------------------------------------------------------------------------------------

    // SOP1: 0xBE800000, SDST at 22:16, the opcode at 15:8, SSRC0 at 7:0; then a literal.
    private void EncodeSop1()
    {
        var sig = _ins.Operands;
        int d = sig.IndexOf(GcnRole.ScalarDestination), s = sig.IndexOf(GcnRole.ScalarSource);
        int sdst = d >= 0 ? ScalarDestination(d) : 0;
        int ssrc = s >= 0 ? ScalarSource(s) : 0;
        Word(0xBE800000 | (uint)sdst << 16 | (uint)_ins.Code << 8 | (uint)ssrc);
        Literals(s, -1);
    }

    // SOP2: 0x80000000, the opcode at 29:23, SDST at 22:16, SSRC1 at 15:8, SSRC0 at 7:0; then one literal.
    private void EncodeSop2()
    {
        int ssrc0 = ScalarSource(1), ssrc1 = ScalarSource(2);
        Word(0x80000000 | (uint)_ins.Code << 23 | (uint)ScalarDestination(0) << 16 | (uint)ssrc1 << 8 | (uint)ssrc0);
        Literals(1, 2);
    }

    // SOPK: 0xB0000000, the opcode at 27:23, SDST at 22:16, SIMM16 at 15:0 (s_setreg_imm32_b32: then its literal).
    private void EncodeSopk()
    {
        var sig = _ins.Operands;
        int register = 0, simm16 = 0;
        uint? literal = null;
        for (int i = 0; i < _count; i++)
        {
            switch (sig.Role(i))
            {
                case GcnRole.ScalarDestination:
                    register = ScalarDestination(i);
                    break;
                case GcnRole.ScalarSource:
                    register = ScalarRegisterOnly(i);
                    break;
                case GcnRole.Immediate16 or GcnRole.HardwareRegister:
                    simm16 = Immediate(i, short.MinValue, ushort.MaxValue);
                    break;
                case GcnRole.Literal32:
                    literal = ImmediateBits(i);
                    break;
            }
        }

        Word(0xB0000000 | (uint)_ins.Code << 23 | (uint)register << 16 | (uint)(simm16 & 0xFFFF));
        if (literal is { } bits)
        {
            Word(bits);
        }
    }

    // SOPC: 0xBF000000, the opcode at 22:16, SSRC1 at 15:8, SSRC0 at 7:0; then one literal.
    private void EncodeSopc()
    {
        int ssrc0 = ScalarSource(0);
        bool gprIndex = _ins.Operands.Role(1) == GcnRole.GprIndex;
        int ssrc1 = gprIndex ? Immediate(1, 0, 15) : ScalarSource(1);
        Word(0xBF000000 | (uint)_ins.Code << 16 | (uint)ssrc1 << 8 | (uint)ssrc0);
        Literals(0, gprIndex ? -1 : 1);
    }

    // SOPP: 0xBF800000, the opcode at 22:16, SIMM16 at 15:0 (a branch's is resolved by ToArray).
    private void EncodeSopp()
    {
        int simm16 = 0;
        if (_count == 1)
        {
            var operand = _ops[0];
            if (operand.Kind == GcnOperandKind.Label)
            {
                if ((uint)operand.Index >= (uint)_labelCount)
                {
                    throw Refuse(0, "the label is not this assembler's");
                }

                Grow(ref _branches, BranchSlot * _branchCount + BranchSlot);
                _branches[BranchSlot * _branchCount] = _length;
                _branches[BranchSlot * _branchCount + 1] = operand.Index;
                _branches[BranchSlot * _branchCount + 2] = _parsing ? _line : 0;
                _branchCount++;
            }
            else
            {
                simm16 = Immediate(0, short.MinValue, ushort.MaxValue);
            }
        }

        Word(0xBF800000 | (uint)_ins.Code << 16 | (uint)(simm16 & 0xFFFF));
    }

    // SMEM: SBASE (the pair's index / 2) at 5:0, SDATA at 12:6, GLC at 16, IMM at 17, the opcode at 25:18, 0xC0000000; then
    // the offset (IMM, 20 bits) or the offset's scalar register.
    private void EncodeSmem()
    {
        var sig = _ins.Operands;
        int data = 0, sbase = 0;
        uint offsetWord = 0;
        bool imm = false;
        for (int i = 0; i < _count; i++)
        {
            var operand = _ops[i];
            int width = sig.Width(i);
            switch (sig.Role(i))
            {
                case GcnRole.ScalarDestination or GcnRole.ScalarSource:
                    if (operand.Kind != GcnOperandKind.Sgpr || operand.Count != width || !Aligned(operand) || operand.Index + width > ScalarRegisters)
                    {
                        throw Refuse(i, $"the data is {width} aligned scalar register(s)");
                    }

                    data = operand.Index;
                    break;
                case GcnRole.MemoryBase:
                    if (operand.Kind != GcnOperandKind.Sgpr || operand.Count != width || !Aligned(operand) || operand.Index + width > ScalarRegisters)
                    {
                        throw Refuse(i, width == 2 ? "the base is an even scalar register pair" : $"the base is {width} aligned scalar registers");
                    }

                    sbase = operand.Index >> 1;
                    break;
                case GcnRole.MemoryOffset:
                    if (operand.Kind is GcnOperandKind.Immediate or GcnOperandKind.Constant or GcnOperandKind.Literal)
                    {
                        int offset = unchecked((int)operand.Value);
                        if (offset is < 0 or > 0xFFFFF)
                        {
                            throw OutOfRange(i, "offset", offset, "an immediate offset is 0 to 0xFFFFF bytes");
                        }

                        offsetWord = (uint)offset;
                        imm = true;
                    }
                    else
                    {
                        offsetWord = (uint)ScalarRegisterOnly(i);
                    }

                    break;
            }
        }

        Word(0xC0000000 | (uint)_ins.Code << 18 | (imm ? 1u << 17 : 0) | (_m.Glc ? 1u << 16 : 0) | (uint)data << 6 | (uint)sbase);
        Word(offsetWord);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Vector formats: VOP1, VOP2, VOPC (32-bit, DPP, SDWA, VOP3), VOP3 and VOP3P.
    // ---------------------------------------------------------------------------------------------------------------

    // The operands by place: the destination, the lane mask out, the sources (up to 3), the lane mask in and K.
    private struct Places
    {
        public int Destination, MaskOut, MaskIn, K, Sources, Src0, Src1, Src2;
    }

    private Places Place()
    {
        var sig = _ins.Operands;
        var p = new Places { Destination = -1, MaskOut = -1, MaskIn = -1, K = -1, Src0 = -1, Src1 = -1, Src2 = -1 };
        for (int i = 0; i < _count; i++)
        {
            switch (sig.Role(i))
            {
                case GcnRole.VectorDestination or GcnRole.ScalarResult or GcnRole.AccumulationDestination:
                    p.Destination = i;
                    break;
                case GcnRole.LaneMask:
                    p.MaskOut = i;
                    break;
                case GcnRole.LaneMaskSource:
                    p.MaskIn = i;
                    break;
                case GcnRole.Constant32:
                    p.K = i;
                    break;
                default:
                    switch (p.Sources++)
                    {
                        case 0:
                            p.Src0 = i;
                            break;
                        case 1:
                            p.Src1 = i;
                            break;
                        default:
                            p.Src2 = i;
                            break;
                    }

                    break;
            }
        }

        return p;
    }

    private void EncodeVector()
    {
        var ins = _ins;
        var p = Place();
        CheckSourceModifiers(p);
        var variant = _variant;
        if (ins.Format is GcnFormat.Vop3 or GcnFormat.Vop3p)
        {
            if (variant is not (GcnVariant.Auto or GcnVariant.E64))
            {
                throw Refuse(-1, $"has no {Suffix(variant)} form");
            }

            if (ins.Format == GcnFormat.Vop3p)
            {
                EncodeVop3p(p);
            }
            else
            {
                EncodeVop3(p, ins.Code);
            }

            return;
        }

        if (variant == GcnVariant.Auto)
        {
            variant = _m.DppControl >= 0 ? GcnVariant.Dpp
                : _m.SdwaGiven ? GcnVariant.Sdwa
                : FitsE32(p, out _, out _) ? GcnVariant.E32
                : ins.Has(GcnInstructionFlags.E64) ? GcnVariant.E64
                : GcnVariant.E32;
        }

        _encoded = variant;
        switch (variant)
        {
            case GcnVariant.E32:
                if (!FitsE32(p, out string why, out int at))
                {
                    throw Refuse(at, $"its 32-bit form (_e32) {why}");
                }

                EncodeE32(p);
                break;
            case GcnVariant.E64:
                if (!ins.Has(GcnInstructionFlags.E64))
                {
                    throw Refuse(-1, "has no VOP3 form (_e64)");
                }

                EncodeVop3(p, ins.Format switch { GcnFormat.Vop1 => 0x140 + ins.Code, GcnFormat.Vop2 => 0x100 + ins.Code, _ => ins.Code });
                break;
            case GcnVariant.Dpp:
                EncodeDpp(p);
                break;
            default:
                EncodeSdwa(p);
                break;
        }
    }

    private static string Suffix(GcnVariant variant) => variant switch
    {
        GcnVariant.E32 => "32-bit (_e32)",
        GcnVariant.E64 => "VOP3 (_e64)",
        GcnVariant.Dpp => "DPP (_dpp)",
        _ => "SDWA (_sdwa)",
    };

    // Whether the operands and modifiers fit the 32-bit form: no modifiers, the lane masks VCC, src1 a vector register;
    // when not, why and the operand at fault (-1 for a modifier).
    private bool FitsE32(in Places p, out string why, out int at)
    {
        at = -1;
        if (_m.Clamp || _m.Omod != 0 || _m.OpSelCount > 0 || (_negMask | _absMask | _sextMask) != 0)
        {
            why = "takes no modifiers (clamp, mul, div, op_sel, neg, abs)";
            return false;
        }

        if (p.MaskOut >= 0 && !IsVcc(_ops[p.MaskOut]) || p.MaskIn >= 0 && !IsVcc(_ops[p.MaskIn]))
        {
            at = p.MaskOut >= 0 && !IsVcc(_ops[p.MaskOut]) ? p.MaskOut : p.MaskIn;
            why = "has vcc as its lane mask";
            return false;
        }

        if (_ins.Format is GcnFormat.Vop2 or GcnFormat.Vopc && p.Src1 >= 0 && _ops[p.Src1].Kind != GcnOperandKind.Vgpr)
        {
            at = p.Src1;
            why = "reads src1 from a vector register";
            return false;
        }

        why = "";
        return true;
    }

    private bool IsVcc(GcnOperand operand) => operand is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Vcc };

    // The 32-bit forms. VOP1: 0x7E000000, VDST at 24:17, the opcode at 16:9, SRC0 at 8:0. VOP2: the opcode at 30:25, VDST,
    // VSRC1 at 16:9, SRC0. VOPC: 0x7C000000, the opcode at 24:17, VSRC1, SRC0. Then a literal (src0's, or K).
    private void EncodeE32(in Places p)
    {
        var ins = _ins;
        int src0 = p.Src0 >= 0 ? VectorSource(p.Src0) : 0;
        int vsrc1 = p.Src1 >= 0 ? VectorRegisterOnly(p.Src1) : 0;
        int vdst = p.Destination >= 0 ? Destination(p.Destination) : 0;
        CheckConstantBus(p, implicitVcc: p.MaskIn >= 0);
        uint word = ins.Format switch
        {
            GcnFormat.Vop1 => 0x7E000000 | (uint)vdst << 17 | (uint)ins.Code << 9 | (uint)src0,
            GcnFormat.Vop2 => (uint)ins.Code << 25 | (uint)vdst << 17 | (uint)vsrc1 << 9 | (uint)src0,
            _ => 0x7C000000 | (uint)ins.Code << 17 | (uint)vsrc1 << 9 | (uint)src0,
        };
        Word(word);
        if (p.K >= 0)
        {
            uint k = ImmediateBits(p.K);
            if (p.Src0 >= 0 && _ops[p.Src0].Kind == GcnOperandKind.Literal && _ops[p.Src0].Value != k)
            {
                throw Refuse(p.Src0, "an instruction holds one literal, and K is it");
            }

            Word(k);
        }
        else if (p.Src0 >= 0 && _ops[p.Src0].Kind == GcnOperandKind.Literal)
        {
            Word(_ops[p.Src0].Value);
        }
    }

    // VOP3: 0xD0000000, the opcode at 25:16, CLAMP at 15, then OP_SEL at 14:11 and ABS at 10:8 (VOP3a) or SDST at 14:8
    // (VOP3b: the carry-out instructions), VDST at 7:0; SRC0 at 8:0, SRC1 at 17:9, SRC2 at 26:18, OMOD at 28:27, NEG at
    // 31:29. ISA 9 holds no literal in it.
    private void EncodeVop3(in Places p, int code)
    {
        bool carryOut = p.MaskOut >= 0 && p.Destination >= 0;
        int vdst = p.Destination >= 0 ? Destination(p.Destination) : p.MaskOut >= 0 ? ScalarPair(p.MaskOut) : 0;
        int sdst = carryOut ? ScalarPair(p.MaskOut) : 0;
        int src0 = p.Src0 >= 0 ? Vop3Source(p.Src0) : 0;
        int src1 = p.Src1 >= 0 ? Vop3Source(p.Src1) : 0;
        int src2 = p.Src2 >= 0 ? Vop3Source(p.Src2) : p.MaskIn >= 0 ? ScalarPair(p.MaskIn) : 0;
        CheckConstantBus(p, implicitVcc: false);
        int neg = SourceBits(p, _negMask), abs = SourceBits(p, _absMask);
        if (carryOut && abs != 0)
        {
            throw Refuse(-1, "the carry-out (VOP3b) layout has no abs");
        }

        CheckClampOmod(vop3: true);
        int opSel = 0;
        if (_m.OpSelCount > 0)
        {
            if (!_ins.Has(GcnInstructionFlags.OpSel) || _m.OpSelCount > p.Sources + 1)
            {
                throw Refuse(-1, _ins.Has(GcnInstructionFlags.OpSel) ? $"op_sel lists its {p.Sources} sources and the destination" : "takes no op_sel");
            }

            // [src0, src1, (src2,) dst]: a list one longer than the sources ends with the destination's (bit 3).
            for (int i = 0; i < _m.OpSelCount; i++)
            {
                if ((_m.OpSel >> i & 1) != 0)
                {
                    opSel |= i == _m.OpSelCount - 1 && _m.OpSelCount == p.Sources + 1 ? 8 : 1 << i;
                }
            }
        }

        Word(0xD0000000 | (uint)code << 16 | (_m.Clamp ? 1u << 15 : 0) | (carryOut ? (uint)sdst << 8 : (uint)opSel << 11 | (uint)abs << 8) | (uint)vdst);
        Word((uint)neg << 29 | (uint)_m.Omod << 27 | (uint)src2 << 18 | (uint)src1 << 9 | (uint)src0);
    }

    // VOP3P: 0xD3800000, the opcode at 22:16, CLAMP at 15, OP_SEL_HI[2] at 14, OP_SEL at 13:11, NEG_HI at 10:8, VDST at
    // 7:0; SRC0, SRC1, SRC2, OP_SEL_HI[1:0] at 28:27, NEG_LO at 31:29. op_sel_hi is all ones unless written (none for the
    // mixed-precision instructions, whose -x and |x| are NEG_LO and NEG_HI).
    private void EncodeVop3p(in Places p)
    {
        int vdst = p.Destination >= 0 ? Destination(p.Destination) : 0;
        int src0 = p.Src0 >= 0 ? Vop3Source(p.Src0) : 0;
        int src1 = p.Src1 >= 0 ? Vop3Source(p.Src1) : 0;
        int src2 = p.Src2 >= 0 ? Vop3Source(p.Src2) : 0;
        CheckConstantBus(p, implicitVcc: false);
        CheckClampOmod(vop3: true);
        bool mix = _ins.Has(GcnInstructionFlags.Mix);
        int negLo = _m.NegLo, negHi = _m.NegHi;
        if ((negLo | negHi) != 0 && !_ins.Has(GcnInstructionFlags.NegLoHi))
        {
            throw Refuse(-1, "takes no neg_lo or neg_hi");
        }

        if ((_m.OpSelCount | _m.OpSelHiGiven) != 0 && !_ins.Has(GcnInstructionFlags.OpSel))
        {
            throw Refuse(-1, "takes no op_sel or op_sel_hi");
        }

        if (_m.OpSelCount > 3)
        {
            throw Refuse(-1, "a VOP3P op_sel lists at most 3 sources");
        }

        if (mix)
        {
            negLo |= SourceBits(p, _negMask);
            negHi |= SourceBits(p, _absMask);
        }

        int opSelHi = (mix ? 0 : 7) & ~_m.OpSelHiGiven | _m.OpSelHi;
        Word(0xD3800000 | (uint)_ins.Code << 16 | (_m.Clamp ? 1u << 15 : 0) | (uint)(opSelHi >> 2 & 1) << 14 | (uint)(_m.OpSel & 7) << 11 | (uint)negHi << 8 | (uint)vdst);
        Word((uint)negLo << 29 | (uint)(opSelHi & 3) << 27 | (uint)src2 << 18 | (uint)src1 << 9 | (uint)src0);
    }

    // DPP: the 32-bit form with SRC0 = 0xFA, then SRC0 (a vector register) at 7:0, DPP_CTRL at 16:8, BOUND_CTRL at 19,
    // SRC0_NEG 20, SRC0_ABS 21, SRC1_NEG 22, SRC1_ABS 23, BANK_MASK at 27:24, ROW_MASK at 31:28.
    private void EncodeDpp(in Places p)
    {
        var ins = _ins;
        if (!ins.Has(GcnInstructionFlags.Dpp))
        {
            throw Refuse(-1, "has no DPP form (_dpp)");
        }

        if (_m.DppControl < 0)
        {
            throw Refuse(-1, "its DPP form needs a control (quad_perm, row_shl, row_shr, row_ror, wave_*, row_mirror, row_half_mirror, row_bcast)");
        }

        if (_m.Clamp || _m.Omod != 0 || _sextMask != 0 || p.MaskOut >= 0 && !IsVcc(_ops[p.MaskOut]) || p.MaskIn >= 0 && !IsVcc(_ops[p.MaskIn]))
        {
            throw Refuse(-1, "its DPP form takes no clamp, mul, div or sext, and its lane masks are vcc");
        }

        int src0 = p.Src0 >= 0 ? VectorRegisterOnly(p.Src0) : 0;
        int vsrc1 = p.Src1 >= 0 ? VectorRegisterOnly(p.Src1) : 0;
        int vdst = p.Destination >= 0 ? VectorRegisterOnly(p.Destination) : 0;
        uint word = ins.Format == GcnFormat.Vop1
            ? 0x7E000000 | (uint)vdst << 17 | (uint)ins.Code << 9 | 0xFA
            : (uint)ins.Code << 25 | (uint)vdst << 17 | (uint)vsrc1 << 9 | 0xFA;
        int neg = SourceBits(p, _negMask), abs = SourceBits(p, _absMask);
        Word(word);
        Word((uint)_m.RowMask << 28 | (uint)_m.BankMask << 24 | (uint)(abs >> 1 & 1) << 23 | (uint)(neg >> 1 & 1) << 22 | (uint)(abs & 1) << 21
             | (uint)(neg & 1) << 20 | (_m.BoundCtrl ? 1u << 19 : 0) | (uint)_m.DppControl << 8 | (uint)src0);
    }

    // SDWA: the 32-bit form with SRC0 = 0xF9, then SRC0 at 7:0, DST_SEL at 10:8, DST_UNUSED at 12:11, CLAMP 13, OMOD at
    // 15:14 (a compare's SDST at 14:8 and SD at 15 instead), SRC0_SEL at 18:16, SEXT 19, NEG 20, ABS 21, S0 23 (src0 is a
    // scalar operand code), SRC1_SEL at 26:24, SEXT 27, NEG 28, ABS 29, S1 31.
    private void EncodeSdwa(in Places p)
    {
        var ins = _ins;
        if (!ins.Has(GcnInstructionFlags.Sdwa))
        {
            throw Refuse(-1, "has no SDWA form (_sdwa)");
        }

        bool compare = ins.Format == GcnFormat.Vopc;
        if (p.MaskIn >= 0 && !IsVcc(_ops[p.MaskIn]))
        {
            throw Refuse(p.MaskIn, "its SDWA form reads the mask from vcc");
        }

        if (compare && (_m.Clamp || _m.Omod != 0 || _m.DstGiven))
        {
            throw Refuse(-1, "a compare's SDWA form has no dst_sel, dst_unused, clamp or omod");
        }

        if (p.MaskOut >= 0 && !compare && !IsVcc(_ops[p.MaskOut]))
        {
            throw Refuse(p.MaskOut, "its SDWA form writes the carry to vcc");
        }

        CheckClampOmod(vop3: false);
        int src0 = p.Src0 >= 0 ? SdwaSource(p.Src0) : 0;
        int src1 = p.Src1 >= 0 ? SdwaSource(p.Src1) : 256;
        int vdst = p.Destination >= 0 ? VectorRegisterOnly(p.Destination) : 0;
        CheckConstantBus(p, implicitVcc: p.MaskIn >= 0);
        uint vsrc1 = (uint)src1 & 0xFF;
        uint word = ins.Format switch
        {
            GcnFormat.Vop1 => 0x7E000000 | (uint)vdst << 17 | (uint)ins.Code << 9 | 0xF9,
            GcnFormat.Vop2 => (uint)ins.Code << 25 | (uint)vdst << 17 | vsrc1 << 9 | 0xF9,
            _ => 0x7C000000 | (uint)ins.Code << 17 | vsrc1 << 9 | 0xF9,
        };
        int neg = SourceBits(p, _negMask), abs = SourceBits(p, _absMask), sext = SourceBits(p, _sextMask);
        uint dst = compare
            ? IsVcc(_ops[p.MaskOut]) ? 0 : 1u << 15 | (uint)ScalarPair(p.MaskOut) << 8
            : (uint)_m.DstSel << 8 | (uint)_m.DstUnused << 11 | (_m.Clamp ? 1u << 13 : 0) | (uint)_m.Omod << 14;
        Word(word);
        int src1Sel = p.Src1 >= 0 ? _m.Src1Sel : 0;                                // a VOP1 has no src1: its field is 0
        Word((src1 < 256 ? 1u << 31 : 0) | (uint)(abs >> 1 & 1) << 29 | (uint)(neg >> 1 & 1) << 28 | (uint)(sext >> 1 & 1) << 27 | (uint)src1Sel << 24
             | (src0 < 256 ? 1u << 23 : 0) | (uint)(abs & 1) << 21 | (uint)(neg & 1) << 20 | (uint)(sext & 1) << 19 | (uint)_m.Src0Sel << 16 | dst
             | (uint)src0 & 0xFF);
    }

    // The modifiers a source took, checked against what the instruction allows (neg and abs on float sources, sext on
    // integer ones in SDWA).
    private void CheckSourceModifiers(in Places p)
    {
        int written = _negMask | _absMask | _sextMask;
        if (written == 0)
        {
            return;
        }

        for (int i = 0; i < _count; i++)
        {
            if ((written >> i & 1) == 0)
            {
                continue;
            }

            int source = i == p.Src0 ? 0 : i == p.Src1 ? 1 : i == p.Src2 ? 2 : -1;
            if (source < 0)
            {
                throw Refuse(i, "only a source takes -, | | or sext");
            }

            bool negAllowed = _ins.Has((GcnInstructionFlags)((uint)GcnInstructionFlags.Neg0 << (2 * source)));
            bool absAllowed = _ins.Has((GcnInstructionFlags)((uint)GcnInstructionFlags.Abs0 << (2 * source)));
            if ((_negMask >> i & 1) != 0 && !negAllowed)
            {
                throw Refuse(i, "this source takes no negation (an integer source, or one the instruction cannot negate)");
            }

            if ((_absMask >> i & 1) != 0 && !absAllowed)
            {
                throw Refuse(i, "this source takes no absolute value");
            }

            if ((_sextMask >> i & 1) != 0 && (negAllowed || _variant != GcnVariant.Sdwa && _m is { SdwaGiven: false }))
            {
                throw Refuse(i, "sext is for an integer source of an SDWA form");
            }
        }
    }

    // The bits of `mask` (by operand) for src0, src1 and src2.
    private static int SourceBits(in Places p, int mask) =>
        (p.Src0 >= 0 ? mask >> p.Src0 & 1 : 0) | (p.Src1 >= 0 ? (mask >> p.Src1 & 1) << 1 : 0) | (p.Src2 >= 0 ? (mask >> p.Src2 & 1) << 2 : 0);

    private void CheckClampOmod(bool vop3)
    {
        if (_m.Clamp && !_ins.Has(GcnInstructionFlags.Clamp))
        {
            throw Refuse(-1, "takes no clamp");
        }

        if (_m.Omod != 0 && !_ins.Has(GcnInstructionFlags.Omod))
        {
            throw Refuse(-1, "takes no output modifier (mul:2, mul:4, div:2)");
        }

        if (!vop3 && _m.OpSelCount > 0)
        {
            throw Refuse(-1, "takes op_sel only in its VOP3 form");
        }
    }

    // ISA 9 lets a vector instruction read one scalar value: distinct scalar registers, special registers and literals
    // count (the inline constants do not), and VCC where the 32-bit form reads it as its mask or the instruction reads it
    // unnamed.
    private void CheckConstantBus(in Places p, bool implicitVcc)
    {
        int first = -1, firstCount = 0, reads = 0;
        uint literal = 0;
        bool hasLiteral = false;
        if (implicitVcc || _ins.Has(GcnInstructionFlags.ReadsVcc))
        {
            first = _set.Special(GcnSpecial.Vcc);
            firstCount = 2;
            reads = 1;
        }

        for (int i = 0; i < _count; i++)
        {
            if (i != p.Src0 && i != p.Src1 && i != p.Src2 && (i != p.MaskIn || implicitVcc) && i != p.K)
            {
                continue;
            }

            var operand = _ops[i];
            if (operand.Kind is GcnOperandKind.Literal || i == p.K)
            {
                uint value = operand.Value;
                if (!hasLiteral)
                {
                    hasLiteral = true;
                    literal = value;
                    reads++;
                }
                else if (literal != value)
                {
                    throw Refuse(i, "an instruction holds one literal");
                }

                continue;
            }

            int code = operand.Kind switch
            {
                GcnOperandKind.Sgpr => operand.Index,
                GcnOperandKind.Ttmp => _set.Ttmp(operand.Index),
                GcnOperandKind.Special => _set.Special((GcnSpecial)operand.Index),
                _ => -1,
            };
            if (code < 0 || (code == first && operand.Count == firstCount))
            {
                continue;
            }

            if (first < 0)
            {
                first = code;
                firstCount = operand.Count;
            }

            reads++;
            if (reads > ConstantBus)
            {
                throw Refuse(i, $"ISA 9 reads at most {ConstantBus} scalar value (a scalar or special register, or a literal) in a vector instruction");
            }
        }

        if (reads > ConstantBus)
        {
            throw Refuse(-1, $"ISA 9 reads at most {ConstantBus} scalar value (a scalar or special register, or a literal) in a vector instruction, and this one also reads vcc");
        }
    }

    // A vector instruction's destination field: a vector or accumulation register's number, or a scalar's operand code.
    private int Destination(int i)
    {
        var operand = _ops[i];
        int width = _ins.Operands.Width(i);
        return _ins.Operands.Role(i) switch
        {
            GcnRole.ScalarResult => ScalarDestination(i),
            GcnRole.AccumulationDestination when operand.Kind == GcnOperandKind.Agpr && operand.Count == width && operand.Index + width <= VectorRegisters => operand.Index,
            GcnRole.VectorDestination when operand.Kind == GcnOperandKind.Vgpr && operand.Count == width && operand.Index + width <= VectorRegisters => operand.Index,
            GcnRole.AccumulationDestination => throw Refuse(i, $"the destination is {width} accumulation register(s)"),
            _ => throw Refuse(i, $"the destination is {width} vector register(s)"),
        };
    }

    // A vector instruction's first source in the 32-bit form: a register of either file, a special register or a constant.
    private int VectorSource(int i)
    {
        var operand = _ops[i];
        int width = _ins.Operands.Width(i);
        switch (_ins.Operands.Role(i))
        {
            case GcnRole.VectorSource when operand.Kind != GcnOperandKind.Vgpr:
                throw Refuse(i, "this source is a vector register");
            case GcnRole.ScalarOnlySource when operand.Kind == GcnOperandKind.Vgpr:
                throw Refuse(i, "this source is a scalar register or a constant");
            case GcnRole.VectorOrInline when operand.Kind is not (GcnOperandKind.Vgpr or GcnOperandKind.Constant):
                throw Refuse(i, "this source is a vector register or an inline constant");
            case GcnRole.AccumulationSource:
                return operand.Kind == GcnOperandKind.Agpr && operand.Count == width && operand.Index + width <= VectorRegisters
                    ? 256 + operand.Index
                    : throw Refuse(i, $"this source is {width} accumulation register(s)");
        }

        if (operand.Kind == GcnOperandKind.Vgpr)
        {
            return operand.Count == width && operand.Index + width <= VectorRegisters
                ? 256 + operand.Index
                : throw Refuse(i, $"this source is {width} register(s)");
        }

        return ScalarSource(i);
    }

    // A source of the VOP3 forms: as the 32-bit form's src0, without a literal (ISA 9 has none there).
    private int Vop3Source(int i)
    {
        int code = VectorSource(i);
        return code != GcnOperand.LiteralCode
            ? code
            : throw Refuse(i, "ISA 9 holds no literal in a VOP3 or VOP3P instruction (only inline constants)");
    }

    // An SDWA source: a vector register, or a scalar operand (register or inline constant) with its S bit.
    private int SdwaSource(int i)
    {
        int code = VectorSource(i);
        if (code == GcnOperand.LiteralCode)
        {
            throw Refuse(i, "an SDWA form holds no literal (only inline constants)");
        }

        if (_ops[i].Count > 1)
        {
            throw Refuse(i, "an SDWA form reads 32-bit sources");
        }

        return code;
    }

    // One vector register (a 32-bit form's second source, or any DPP operand).
    private int VectorRegisterOnly(int i)
    {
        var operand = _ops[i];
        int width = _ins.Operands.Width(i);
        return operand.Kind == GcnOperandKind.Vgpr && operand.Count == width && operand.Index + width <= VectorRegisters
            ? operand.Index
            : throw Refuse(i, width == 1 ? "this operand is one vector register" : $"this operand is {width} vector registers");
    }

    // A scalar register pair (or VCC, EXEC...) a vector instruction writes or reads per lane.
    private int ScalarPair(int i)
    {
        var operand = _ops[i];
        return operand.Count == 2 && operand.Kind is GcnOperandKind.Sgpr or GcnOperandKind.Ttmp or GcnOperandKind.Special
            ? ScalarDestination(i, 2)
            : throw Refuse(i, "this operand is a scalar register pair (vcc, or s[n:n+1] with n even)");
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Memory formats: DS, FLAT and GLOBAL.
    // ---------------------------------------------------------------------------------------------------------------

    // DS: OFFSET0 at 7:0 and OFFSET1 at 15:8 (or a 16-bit OFFSET), GDS at 16, the opcode at 24:17, 0xD8000000; then ADDR at
    // 7:0, DATA0 at 15:8, DATA1 at 23:16, VDST at 31:24.
    private void EncodeDs()
    {
        var sig = _ins.Operands;
        int vdst = 0, addr = 0, data0 = 0, data1 = 0;
        for (int i = 0; i < _count; i++)
        {
            switch (sig.Role(i))
            {
                case GcnRole.VectorDestination:
                    vdst = VectorRegisterOnly(i);
                    break;
                case GcnRole.Address:
                    addr = VectorRegisterOnly(i);
                    break;
                case GcnRole.Data0:
                    data0 = VectorRegisterOnly(i);
                    break;
                case GcnRole.Data1:
                    data1 = VectorRegisterOnly(i);
                    break;
            }
        }

        int offset;
        if (_ins.Has(GcnInstructionFlags.TwoOffsets))
        {
            if (_m.Offset != 0)
            {
                throw Refuse(-1, "takes offset0 and offset1, not offset");
            }

            if (_m.Offset0 is < 0 or > 255 || _m.Offset1 is < 0 or > 255)
            {
                throw OutOfRange(-1, "offset", _m.Offset0 is < 0 or > 255 ? _m.Offset0 : _m.Offset1, "offset0 and offset1 are 0 to 255");
            }

            offset = _m.Offset0 | _m.Offset1 << 8;
        }
        else
        {
            if ((_m.Offset0 | _m.Offset1) != 0)
            {
                throw Refuse(-1, "takes one offset (offset), not offset0 and offset1");
            }

            if (_m.Swizzle && _ins.Op != GcnOp.DsSwizzleB32)
            {
                throw Refuse(-1, "only ds_swizzle_b32 takes swizzle()");
            }

            if (_m.Offset is < 0 or > 0xFFFF)
            {
                throw OutOfRange(-1, "offset", _m.Offset, "the offset is 0 to 65535 bytes");
            }

            offset = _m.Offset;
        }

        Word(0xD8000000 | (uint)_ins.Code << 17 | (_m.Gds ? 1u << 16 : 0) | (uint)offset);
        Word((uint)vdst << 24 | (uint)data1 << 16 | (uint)data0 << 8 | (uint)addr);
    }

    // FLAT and GLOBAL: OFFSET at 12:0 (global: 13 bits signed; flat: 12 unsigned), SEG at 15:14 (0 flat, 2 global), GLC 16,
    // SLC 17, the opcode at 24:18, 0xDC000000; then ADDR at 7:0, DATA at 15:8, SADDR at 22:16 (0x7F: off), VDST at 31:24.
    // An atomic returns the old value (into a destination written first) when glc is set.
    private void EncodeFlat()
    {
        var ins = _ins;
        var sig = ins.Operands;
        bool global = ins.Format == GcnFormat.Global;
        int vdst = 0, addr = 0, data = 0, saddr = global ? _set.Special(GcnSpecial.Off) : 0;
        bool off = true, returns = false;
        int addressAt = -1;
        for (int i = 0; i < _count; i++)
        {
            var operand = _ops[i];
            switch (sig.Role(i))
            {
                case GcnRole.VectorDestination when operand.Kind != GcnOperandKind.None:   // an atomic without glc has none
                    vdst = VectorRegisterOnly(i);
                    returns = true;
                    break;
                case GcnRole.Address:
                    addressAt = i;
                    break;
                case GcnRole.Data0:
                    data = VectorRegisterOnly(i);
                    break;
                case GcnRole.ScalarAddress:
                    off = operand is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Off };
                    if (!off)
                    {
                        if (operand.Count != 2 || operand.Kind == GcnOperandKind.Vgpr || operand.Kind == GcnOperandKind.Sgpr && operand.Index % 2 != 0)
                        {
                            throw Refuse(i, "the scalar address is an even scalar register pair, or off");
                        }

                        saddr = ScalarDestination(i, 2);
                    }

                    break;
            }
        }

        if (addressAt >= 0)
        {
            var address = _ops[addressAt];
            int width = global && !off ? 1 : 2;
            if (address.Kind != GcnOperandKind.Vgpr || address.Count != width || address.Index + width > VectorRegisters)
            {
                throw Refuse(addressAt, !global ? "the address is a vector register pair"
                    : off ? "with off, the address is a vector register pair" : "with a scalar address, the offset is one vector register");
            }

            addr = address.Index;
        }

        if (ins.Has(GcnInstructionFlags.Atomic))
        {
            if (returns != _m.Glc)
            {
                throw Refuse(-1, returns ? "an atomic returns the old value only with glc" : "with glc an atomic returns the old value: name its destination first");
            }

            if (returns && ins.Has(GcnInstructionFlags.NoReturn))
            {
                throw Refuse(-1, $"has no returning form (glc) in {_version}");
            }
        }

        int offset = _m.Offset;
        if (global ? offset is < -4096 or > 4095 : offset is < 0 or > 4095)
        {
            throw OutOfRange(-1, "offset", offset, global ? "the offset is -4096 to 4095" : "the offset is 0 to 4095 bytes");
        }

        Word(0xDC000000 | (uint)ins.Code << 18 | (_m.Slc ? 1u << 17 : 0) | (_m.Glc ? 1u << 16 : 0) | (global ? 2u << 14 : 0) | ((uint)offset & (global ? 0x1FFFu : 0xFFFu)));
        Word((uint)vdst << 24 | (uint)saddr << 16 | (uint)data << 8 | (uint)addr);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Operands to codes.
    // ---------------------------------------------------------------------------------------------------------------

    // A run of scalar registers at a multiple of its size, up to 4 (as the hardware addresses pairs and quads).
    private static bool Aligned(GcnOperand registers) => registers.Index % Math.Min(4, registers.Count) == 0;

    // A scalar destination: registers of the operand's width, or a special register of that width.
    private int ScalarDestination(int i) => ScalarDestination(i, _ins.Operands.Width(i));

    private int ScalarDestination(int i, int width)
    {
        var operand = _ops[i];
        switch (operand.Kind)
        {
            case GcnOperandKind.Sgpr when operand.Count == width && Aligned(operand) && operand.Index + width <= ScalarRegisters:
                return operand.Index;
            case GcnOperandKind.Ttmp when operand.Count == width && Aligned(operand) && operand.Index + width <= _set.Ttmps:
                return _set.Ttmp(operand.Index);
            case GcnOperandKind.Special when operand.Count == width && IsRegister((GcnSpecial)operand.Index):
                return _set.Special((GcnSpecial)operand.Index);
            default:
                throw Refuse(i, $"the destination is {width} scalar register(s)");
        }
    }

    // A scalar source: registers or a special register of the operand's width, or a constant.
    private int ScalarSource(int i)
    {
        var operand = _ops[i];
        int width = _ins.Operands.Width(i);
        switch (operand.Kind)
        {
            case GcnOperandKind.Constant or GcnOperandKind.Literal:
                return operand.Index;
            case GcnOperandKind.Sgpr when operand.Count == width && Aligned(operand) && operand.Index + width <= ScalarRegisters:
                return operand.Index;
            case GcnOperandKind.Ttmp when operand.Count == width && Aligned(operand) && operand.Index + width <= _set.Ttmps:
                return _set.Ttmp(operand.Index);
            case GcnOperandKind.Special when operand.Index != (int)GcnSpecial.Off && (operand.Count == width || operand.Count == 0):
                return _set.Special((GcnSpecial)operand.Index);
            default:
                throw Refuse(i, $"a source is {width} scalar register(s) or a constant");
        }
    }

    // A scalar register (or M0 and the others), never a constant: SOPK's register, SMEM's offset register.
    private int ScalarRegisterOnly(int i)
    {
        var operand = _ops[i];
        return operand.Kind is GcnOperandKind.Sgpr or GcnOperandKind.Ttmp or GcnOperandKind.Special && operand.Count == 1
            ? ScalarSource(i)
            : throw Refuse(i, "this operand is one scalar register");
    }

    private static bool IsRegister(GcnSpecial special) => special is GcnSpecial.Vcc or GcnSpecial.VccLo or GcnSpecial.VccHi or GcnSpecial.Exec or GcnSpecial.ExecLo
        or GcnSpecial.ExecHi or GcnSpecial.M0 or GcnSpecial.FlatScratch or GcnSpecial.FlatScratchLo or GcnSpecial.FlatScratchHi or GcnSpecial.XnackMask
        or GcnSpecial.XnackMaskLo or GcnSpecial.XnackMaskHi;

    // An immediate field's value, from lowest to highest.
    private int Immediate(int i, int lowest, int highest)
    {
        var operand = _ops[i];
        if (operand.Kind is not (GcnOperandKind.Immediate or GcnOperandKind.Constant or GcnOperandKind.Literal))
        {
            throw Refuse(i, "this operand is a number");
        }

        int value = unchecked((int)operand.Value);
        return value >= lowest && value <= highest ? value : throw OutOfRange(i, "operand", value, $"this field holds {lowest} to {highest}");
    }

    private uint ImmediateBits(int i) => _ops[i].Kind is GcnOperandKind.Immediate or GcnOperandKind.Constant or GcnOperandKind.Literal
        ? _ops[i].Value
        : throw Refuse(i, "this operand is a 32-bit constant");

    // The literal of a scalar instruction's sources a and b (one at most, or two of one value).
    private void Literals(int a, int b)
    {
        bool hasA = a >= 0 && _ops[a].Kind == GcnOperandKind.Literal, hasB = b >= 0 && _ops[b].Kind == GcnOperandKind.Literal;
        if (hasA && hasB && _ops[a].Value != _ops[b].Value)
        {
            throw Refuse(b, "an instruction holds one literal");
        }

        if (hasA || hasB)
        {
            Word(_ops[hasA ? a : b].Value);
        }
    }

    /// <summary>s_waitcnt's 16 bits in ISA 9: vmcnt's low four bits at 3:0 and its high two at 15:14, expcnt at 6:4, lgkmcnt at 11:8.</summary>
    internal static ushort WaitBits(int vmcnt, int expcnt, int lgkmcnt) => (ushort)((vmcnt & 0xF) | expcnt << 4 | lgkmcnt << 8 | (vmcnt >> 4) << 14);

    // A refusal naming the operand (or the instruction): with the line and column when assembling text.
    private Exception Refuse(int operand, string rule)
    {
        string reason = operand >= 0 ? $"{_ins.Mnemonic} cannot take {(_standIns ? _written[operand] : _ops[operand])}: {rule}." : $"{_ins.Mnemonic}: {rule}.";
        return _parsing ? new GcnSyntaxException(_line, operand >= 0 ? _columns[operand] : _column, reason) : new ArgumentException(reason, "operand");
    }

    private Exception OutOfRange(int operand, string name, int value, string rule)
    {
        string reason = $"{_ins.Mnemonic}: {rule}, not {value}.";
        return _parsing ? new GcnSyntaxException(_line, operand >= 0 ? _columns[operand] : _column, reason) : new ArgumentOutOfRangeException(name, value, reason);
    }

    private void Word(uint word)
    {
        if (_length + 4 > _code.Length)
        {
            Grow(ref _code, _length + 4);
        }

        BinaryPrimitives.WriteUInt32LittleEndian(_code.AsSpan(_length), word);
        _length += 4;
    }

    // A pooled array of at least `needed` items, its contents kept: twice the size, the old one returned.
    private static void Grow<T>(ref T[] array, int needed)
    {
        if (needed <= array.Length)
        {
            return;
        }

        var bigger = ArrayPool<T>.Shared.Rent(Math.Max(needed, 2 * array.Length));
        array.AsSpan().CopyTo(bigger);
        ArrayPool<T>.Shared.Return(array);
        array = bigger;
    }
}
