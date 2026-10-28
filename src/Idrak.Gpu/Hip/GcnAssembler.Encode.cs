// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>What an operand is: a scalar or vector register (or a run of them), a special register, a constant.</summary>
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
}

/// <summary>
/// One operand of a GCN instruction with physical registers (virtual ones come with the register allocator): a run of
/// scalar or vector registers, a special register (VCC, EXEC, M0, off), or a 32-bit value, which is an inline constant
/// when the hardware holds it (the integers -16 to 64 and the floats ±0.5, ±1, ±2, ±4 and 1/(2π), by their bits, as LLVM
/// chooses) and a literal otherwise.
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

    /// <summary>The first register; for a special register its <see cref="GcnSpecial"/>; for a constant its operand code.</summary>
    public int Index { get; }

    /// <summary>How many 32-bit registers it names (1 for a value).</summary>
    public int Count { get; }

    /// <summary>A value's 32 bits.</summary>
    public uint Value { get; }

    /// <summary>Scalar registers s[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand S(int index, int count = 1) => Registers(GcnOperandKind.Sgpr, index, count);

    /// <summary>Vector registers v[<paramref name="index"/> : index + count - 1].</summary>
    public static GcnOperand V(int index, int count = 1) => Registers(GcnOperandKind.Vgpr, index, count);

    public static GcnOperand Vcc => new(GcnOperandKind.Special, (int)GcnSpecial.Vcc, 2, 0);

    public static GcnOperand VccLo => new(GcnOperandKind.Special, (int)GcnSpecial.VccLo, 1, 0);

    public static GcnOperand VccHi => new(GcnOperandKind.Special, (int)GcnSpecial.VccHi, 1, 0);

    public static GcnOperand Exec => new(GcnOperandKind.Special, (int)GcnSpecial.Exec, 2, 0);

    public static GcnOperand ExecLo => new(GcnOperandKind.Special, (int)GcnSpecial.ExecLo, 1, 0);

    public static GcnOperand ExecHi => new(GcnOperandKind.Special, (int)GcnSpecial.ExecHi, 1, 0);

    public static GcnOperand M0 => new(GcnOperandKind.Special, (int)GcnSpecial.M0, 1, 0);

    /// <summary>No scalar address (a global instruction's SADDR): the vector address is a 64-bit pair.</summary>
    public static GcnOperand Off => new(GcnOperandKind.Special, (int)GcnSpecial.Off, 0, 0);

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

    /// <summary>The operand code of a literal (the value follows the instruction's word).</summary>
    internal const int LiteralCode = 255;

    // The inline constants' operand codes: 128 + n for 0 to 64, 192 + n for -1 to -16, then the floats.
    private static int InlineCode(uint bits)
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

    /// <summary>As LLVM writes it: s7, s[4:7], v0, vcc, off, 2, 0x3fb8aa3b.</summary>
    public override string ToString() => Kind switch
    {
        GcnOperandKind.Sgpr or GcnOperandKind.Vgpr when Count == 1 => $"{(Kind == GcnOperandKind.Sgpr ? 's' : 'v')}{Index}",
        GcnOperandKind.Sgpr or GcnOperandKind.Vgpr => $"{(Kind == GcnOperandKind.Sgpr ? 's' : 'v')}[{Index}:{Index + Count - 1}]",
        GcnOperandKind.Special => ((GcnSpecial)Index).ToString().ToLowerInvariant() switch { "vcclo" => "vcc_lo", "vcchi" => "vcc_hi", "execlo" => "exec_lo", "exechi" => "exec_hi", var name => name },
        GcnOperandKind.Constant when Index < 209 => unchecked((int)Value).ToString(CultureInfo.InvariantCulture),
        GcnOperandKind.Constant => BitConverter.UInt32BitsToSingle(Value).ToString(CultureInfo.InvariantCulture),
        GcnOperandKind.Literal => string.Create(CultureInfo.InvariantCulture, $"0x{Value:x}"),
        _ => "(none)",
    };
}

/// <summary>A place in the code a branch can name, made by <see cref="GcnAssembler.NewLabel"/> and placed by <see cref="GcnAssembler.Bind"/>.</summary>
internal readonly record struct GcnLabel(int Id);

/// <summary>
/// Idrak's GCN assembler (plan 35), what <c>ptxas</c> is to PTX: GCN instructions to the machine code a code object
/// holds. This part encodes: each format's words from an instruction (<see cref="GcnOp"/>) and typed operands, with the
/// family's opcode numbers and operand codes (<see cref="IGcnInstructionSet"/>, in <c>GcnTargets.*.cs</c>), branches to
/// labels resolved at the end. The words are written into one pooled buffer (returned by <see cref="Dispose"/>), so an
/// instruction allocates nothing; <see cref="ToArray"/> gives the code in an array of its exact size.
/// Encoded so far: the forms <c>fill_f32</c> uses, in ISA 9's layouts (SOP1, SOP2, SOPP, SMEM, VOP1, VOP2, VOPC, GLOBAL).
/// </summary>
internal sealed partial class GcnAssembler : IDisposable
{
    private readonly IGcnInstructionSet _set;
    private readonly string _family;
    private byte[] _code;
    private int _length;

    // Each label's byte offset (-1 until bound), and each branch as (its byte offset, its label), pooled like the code.
    private int[] _labels;
    private int _labelCount;
    private int[] _branches;
    private int _branchCount;

    // ISA 9's limits: scalar operand codes 0-101 are registers, vector ones 256 a lane.
    private const int ScalarRegisters = 102, VectorRegisters = 256;

    /// <summary>An assembler for <paramref name="target"/>'s family, its buffer sized for <paramref name="bytes"/> of code (it grows past them).</summary>
    /// <exception cref="NotSupportedException">The family has no instruction encodings, or layouts this assembler does not write.</exception>
    public GcnAssembler(IGcnTarget target, int bytes = 256)
    {
        ArgumentNullException.ThrowIfNull(target);
        _family = target.Family;
        _set = target as IGcnInstructionSet
               ?? throw new NotSupportedException($"The GCN family {target.Family} has no instruction encodings (IGcnInstructionSet): Idrak's assembler cannot write code for it.");
        if (_set.Encoding != GcnEncoding.Isa9)
        {
            throw new NotSupportedException($"The GCN family {target.Family} uses the {_set.Encoding} layouts, which Idrak's assembler does not write yet.");
        }

        _code = ArrayPool<byte>.Shared.Rent(Math.Max(64, bytes));
        _labels = ArrayPool<int>.Shared.Rent(8);
        _branches = ArrayPool<int>.Shared.Rent(16);
    }

    /// <summary>Bytes of code written so far.</summary>
    public int Length => _length;

    /// <summary>A new label, placed later with <see cref="Bind"/>.</summary>
    public GcnLabel NewLabel()
    {
        Grow(ref _labels, _labelCount + 1);
        _labels[_labelCount] = -1;
        return new GcnLabel(_labelCount++);
    }

    /// <summary>Places <paramref name="label"/> at the next instruction.</summary>
    public void Bind(GcnLabel label)
    {
        if ((uint)label.Id >= (uint)_labelCount || _labels[label.Id] >= 0)
        {
            throw new InvalidOperationException($"Label {label.Id} is not this assembler's, or is bound already.");
        }

        _labels[label.Id] = _length;
    }

    /// <summary>SOP1: <paramref name="sdst"/> = op(<paramref name="ssrc0"/>).</summary>
    public void Sop1(GcnOp op, GcnOperand sdst, GcnOperand ssrc0)
    {
        int code = Opcode(op, GcnFormat.Sop1);
        int width = Dwords(op);
        uint word = 0xBE800000 | (uint)ScalarDestination(op, sdst, width) << 16 | (uint)code << 8 | (uint)ScalarSource(op, ssrc0, width);
        Emit(word, ssrc0);
    }

    /// <summary>SOP2: <paramref name="sdst"/> = op(<paramref name="ssrc0"/>, <paramref name="ssrc1"/>).</summary>
    public void Sop2(GcnOp op, GcnOperand sdst, GcnOperand ssrc0, GcnOperand ssrc1)
    {
        int code = Opcode(op, GcnFormat.Sop2);
        int width = Dwords(op);
        OneLiteral(op, ssrc0, ssrc1);
        uint word = 0x80000000 | (uint)code << 23 | (uint)ScalarDestination(op, sdst, width) << 16 | (uint)ScalarSource(op, ssrc1, width) << 8
                    | (uint)ScalarSource(op, ssrc0, width);
        Emit(word, ssrc0.Kind == GcnOperandKind.Literal ? ssrc0 : ssrc1);
    }

    /// <summary>SOPP with its 16-bit field (a wait's counts, a branch's dword count, or nothing).</summary>
    public void Sopp(GcnOp op, ushort simm16 = 0) => Word(0xBF800000 | (uint)Opcode(op, GcnFormat.Sopp) << 16 | simm16);

    /// <summary>
    /// s_waitcnt: waits until at most <paramref name="vmcnt"/> vector memory operations (0-63),
    /// <paramref name="expcnt"/> exports (0-7) and <paramref name="lgkmcnt"/> LDS, GDS, constant and message operations
    /// (0-15) are outstanding; a counter left null is not waited on.
    /// </summary>
    public void Waitcnt(int? vmcnt = null, int? expcnt = null, int? lgkmcnt = null)
    {
        int vm = Count(vmcnt, 63, nameof(vmcnt)), exp = Count(expcnt, 7, nameof(expcnt)), lgkm = Count(lgkmcnt, 15, nameof(lgkmcnt));

        // ISA 9: vmcnt's low four bits at 3:0 and its high two at 15:14, expcnt at 6:4, lgkmcnt at 11:8.
        Sopp(GcnOp.SWaitcnt, (ushort)((vm & 0xF) | exp << 4 | lgkm << 8 | (vm >> 4) << 14));

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

        Grow(ref _branches, 2 * _branchCount + 2);
        _branches[2 * _branchCount] = _length;
        _branches[2 * _branchCount + 1] = target.Id;
        _branchCount++;
        Sopp(op);
    }

    /// <summary>
    /// SMEM with an immediate offset: <paramref name="sdata"/> = the dwords at <paramref name="sbase"/> (a register pair
    /// holding an address) + <paramref name="offset"/> bytes (0 to 0xFFFFF).
    /// </summary>
    public void Smem(GcnOp op, GcnOperand sdata, GcnOperand sbase, int offset)
    {
        int code = Opcode(op, GcnFormat.Smem);
        int width = Dwords(op);
        if (sdata.Kind != GcnOperandKind.Sgpr || sdata.Count != width || !Aligned(sdata) || sdata.Index + width > ScalarRegisters)
        {
            throw Refused(op, sdata, $"the data is {width} aligned scalar register(s)");
        }

        if (sbase.Kind != GcnOperandKind.Sgpr || sbase.Count != 2 || sbase.Index % 2 != 0 || sbase.Index + 2 > ScalarRegisters)
        {
            throw Refused(op, sbase, "the base is an even scalar register pair");
        }

        if (offset is < 0 or > 0xFFFFF)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), offset, $"{op}: an immediate offset is 0 to 0xFFFFF bytes.");
        }

        // ISA 9: SBASE (the pair's index / 2) at 5:0, SDATA at 12:6, IMM at 17, the opcode at 25:18; then the offset.
        Word(0xC0000000 | (uint)code << 18 | 1u << 17 | (uint)sdata.Index << 6 | (uint)sbase.Index >> 1);
        Word((uint)offset);
    }

    /// <summary>VOP1: <paramref name="vdst"/> = op(<paramref name="src0"/>), the 32-bit form (_e32).</summary>
    public void Vop1(GcnOp op, GcnOperand vdst, GcnOperand src0)
    {
        int code = Opcode(op, GcnFormat.Vop1);
        Emit(0x7E000000 | (uint)VectorRegister(op, vdst) << 17 | (uint)code << 9 | (uint)VectorSource(op, src0), src0);
    }

    /// <summary>VOP2: <paramref name="vdst"/> = op(<paramref name="src0"/>, <paramref name="vsrc1"/>), the 32-bit form (_e32).</summary>
    public void Vop2(GcnOp op, GcnOperand vdst, GcnOperand src0, GcnOperand vsrc1)
    {
        int code = Opcode(op, GcnFormat.Vop2);
        Emit((uint)code << 25 | (uint)VectorRegister(op, vdst) << 17 | (uint)VectorRegister(op, vsrc1) << 9 | (uint)VectorSource(op, src0), src0);
    }

    /// <summary>VOPC: VCC = op(<paramref name="src0"/>, <paramref name="vsrc1"/>) per lane, the 32-bit form (_e32).</summary>
    public void Vopc(GcnOp op, GcnOperand src0, GcnOperand vsrc1)
    {
        int code = Opcode(op, GcnFormat.Vopc);
        Emit(0x7C000000 | (uint)code << 17 | (uint)VectorRegister(op, vsrc1) << 9 | (uint)VectorSource(op, src0), src0);
    }

    /// <summary>
    /// A GLOBAL store: <paramref name="vdata"/> to the address <paramref name="saddr"/> (a scalar pair) + <paramref name="vaddr"/>
    /// (a 32-bit byte offset per lane) + <paramref name="offset"/>, or, with <see cref="GcnOperand.Off"/>, to
    /// <paramref name="vaddr"/> (a 64-bit address pair per lane) + <paramref name="offset"/> (-4096 to 4095 bytes).
    /// </summary>
    public void GlobalStore(GcnOp op, GcnOperand vaddr, GcnOperand vdata, GcnOperand saddr, int offset = 0, bool glc = false, bool slc = false)
    {
        int code = Opcode(op, GcnFormat.Global);
        if (vdata.Kind != GcnOperandKind.Vgpr || vdata.Count != Dwords(op))
        {
            throw Refused(op, vdata, $"the data is {Dwords(op)} vector register(s)");
        }

        Global(op, code, vaddr, saddr, offset, glc, slc, data: vdata.Index, vdst: 0);
    }

    /// <summary>The code with every branch resolved, in an array of its exact size.</summary>
    /// <exception cref="InvalidOperationException">A branch names a label never bound, or one too far for 16 bits.</exception>
    public byte[] ToArray()
    {
        for (int i = 0; i < _branchCount; i++)
        {
            int at = _branches[2 * i], label = _branches[2 * i + 1];
            int target = _labels[label];
            if (target < 0)
            {
                throw new InvalidOperationException($"The branch at byte {at} names label {label}, which was never bound.");
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

        static void Return<T>(ref T[] array)
        {
            if (array.Length > 0)
            {
                ArrayPool<T>.Shared.Return(array);
                array = [];
            }
        }
    }

    // ISA 9's GLOBAL words: the offset (13 bits, signed) at 12:0, SEG 2 (global) at 15:14, GLC 16, SLC 17, the opcode at
    // 24:18; then ADDR at 7:0, DATA at 15:8, SADDR at 22:16 and VDST at 31:24.
    private void Global(GcnOp op, int code, GcnOperand vaddr, GcnOperand saddr, int offset, bool glc, bool slc, int data, int vdst)
    {
        bool off = saddr is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Off };
        if (!off && (saddr.Kind != GcnOperandKind.Sgpr || saddr.Count != 2 || saddr.Index % 2 != 0 || saddr.Index + 2 > ScalarRegisters))
        {
            throw Refused(op, saddr, "the scalar address is an even scalar register pair, or off");
        }

        if (vaddr.Kind != GcnOperandKind.Vgpr || vaddr.Count != (off ? 2 : 1) || vaddr.Index + vaddr.Count > VectorRegisters)
        {
            throw Refused(op, vaddr, off ? "with off, the address is a vector register pair" : "with a scalar address, the offset is one vector register");
        }

        if (offset is < -4096 or > 4095)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), offset, $"{op}: the offset is -4096 to 4095 bytes.");
        }

        int saddrCode = off ? _set.Special(GcnSpecial.Off) : saddr.Index;
        Word(0xDC000000 | (uint)code << 18 | (slc ? 1u << 17 : 0) | (glc ? 1u << 16 : 0) | 2u << 14 | ((uint)offset & 0x1FFF));
        Word((uint)vdst << 24 | (uint)saddrCode << 16 | (uint)data << 8 | (uint)vaddr.Index);
    }

    // The family's number for op, which must be of the format the caller writes.
    private int Opcode(GcnOp op, GcnFormat format)
    {
        if (!_set.TryOpcode(op, out var opcode))
        {
            throw new NotSupportedException($"The GCN family {_family} has no {op}.");
        }

        if (opcode.Format != format)
        {
            throw new ArgumentException($"{op} is a {opcode.Format} instruction in the {_family} family, not {format}.", nameof(op));
        }

        return opcode.Code;
    }

    // The dwords an instruction's scalar operands (or its data) hold: 1 unless its name says otherwise.
    private static int Dwords(GcnOp op) => op switch
    {
        GcnOp.SAndSaveexecB64 => 2,
        GcnOp.SLoadDwordx4 => 4,
        _ => 1,
    };

    // A run of scalar registers at a multiple of its size, up to 4 (as the hardware addresses pairs and quads).
    private static bool Aligned(GcnOperand registers) => registers.Index % Math.Min(4, registers.Count) == 0;

    // A scalar destination: registers of the instruction's width, or a special register of that width.
    private int ScalarDestination(GcnOp op, GcnOperand sdst, int width)
    {
        if (sdst.Kind == GcnOperandKind.Sgpr && sdst.Count == width && Aligned(sdst) && sdst.Index + width <= ScalarRegisters)
        {
            return sdst.Index;
        }

        if (sdst is { Kind: GcnOperandKind.Special } && sdst.Index != (int)GcnSpecial.Off && sdst.Count == width)
        {
            return _set.Special((GcnSpecial)sdst.Index);
        }

        throw Refused(op, sdst, $"the destination is {width} scalar register(s)");
    }

    // A scalar source: registers or a special register of the instruction's width, or a constant.
    private int ScalarSource(GcnOp op, GcnOperand source, int width)
    {
        switch (source.Kind)
        {
            case GcnOperandKind.Constant or GcnOperandKind.Literal:
                return source.Index;
            case GcnOperandKind.Sgpr when source.Count == width && Aligned(source) && source.Index + width <= ScalarRegisters:
                return source.Index;
            case GcnOperandKind.Special when source.Index != (int)GcnSpecial.Off && source.Count == width:
                return _set.Special((GcnSpecial)source.Index);
            default:
                throw Refused(op, source, $"a source is {width} scalar register(s) or a constant");
        }
    }

    // A vector instruction's first source: one register of either file, a special register's half, or a constant.
    private int VectorSource(GcnOp op, GcnOperand source)
    {
        switch (source.Kind)
        {
            case GcnOperandKind.Vgpr when source.Count == 1 && source.Index < VectorRegisters:
                return 256 + source.Index;
            case GcnOperandKind.Sgpr or GcnOperandKind.Special or GcnOperandKind.Constant or GcnOperandKind.Literal:
                return ScalarSource(op, source, 1);
            default:
                throw Refused(op, source, "src0 is one register or a constant");
        }
    }

    // One vector register (a destination or the second source).
    private static int VectorRegister(GcnOp op, GcnOperand register) => register.Kind == GcnOperandKind.Vgpr && register.Count == 1 && register.Index < VectorRegisters
        ? register.Index
        : throw Refused(op, register, "this operand is one vector register");

    private static void OneLiteral(GcnOp op, GcnOperand a, GcnOperand b)
    {
        if (a.Kind == GcnOperandKind.Literal && b.Kind == GcnOperandKind.Literal)
        {
            throw Refused(op, b, "an instruction holds one literal");
        }
    }

    private static ArgumentException Refused(GcnOp op, GcnOperand operand, string rule) => new($"{op} cannot take {operand}: {rule}.", nameof(operand));

    // The word, then the literal when the source is one.
    private void Emit(uint word, GcnOperand source)
    {
        Word(word);
        if (source.Kind == GcnOperandKind.Literal)
        {
            Word(source.Value);
        }
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
