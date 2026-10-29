// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>
/// The text half of Idrak's GCN assembler: GCN assembly as UTF-8 bytes, read over spans with nothing allocated per
/// token or line, into the encoder's instructions. The syntax is LLVM's (what <c>llvm-mc</c> reads and prints), so a
/// kernel Idrak prints with physical registers is valid input for LLVM, the test oracle:
/// <list type="bullet">
/// <item>One statement a line: <c>label:</c>, then an instruction; <c>;</c> and <c>//</c> start comments.</item>
/// <item>A mnemonic, lowercase or not, with an encoding suffix where the instruction has several: <c>_e32</c>,
/// <c>_e64</c>, <c>_dpp</c>, <c>_sdwa</c>. Without one, the 32-bit form when the operands fit it, else VOP3; DPP or SDWA
/// when their controls are written.</item>
/// <item>Operands split by commas: registers (<c>s7</c>, <c>s[4:7]</c>, <c>v[2:3]</c>, <c>a1</c>, <c>ttmp[2:3]</c>, a
/// range of one as <c>s[3]</c>), special registers and values (<c>vcc</c>, <c>vcc_lo</c>, <c>exec_hi</c>, <c>m0</c>,
/// <c>flat_scratch</c>, <c>xnack_mask</c>, <c>src_scc</c>, <c>src_shared_base</c>, <c>off</c>), integers (decimal,
/// <c>0x</c>, <c>0b</c>, signed) and floats (<c>0.5</c>, <c>1e0</c>), labels, and the fields <c>hwreg(...)</c>,
/// <c>gpr_idx(...)</c>, <c>vmcnt(n) expcnt(n) lgkmcnt(n)</c>. Sources take <c>-x</c>, <c>|x|</c>, <c>-|x|</c>,
/// <c>neg(x)</c>, <c>abs(x)</c> and, in SDWA, <c>sext(x)</c>.</item>
/// <item>Modifiers after the operands, split by spaces: <c>clamp</c>, <c>mul:2</c>, <c>mul:4</c>, <c>div:2</c>,
/// <c>op_sel:[..]</c>, <c>op_sel_hi:[..]</c>, <c>neg_lo:[..]</c>, <c>neg_hi:[..]</c>, <c>offset:n</c>,
/// <c>offset:swizzle(..)</c>, <c>offset0:n</c>, <c>offset1:n</c>, <c>glc</c>, <c>slc</c>, <c>gds</c>; DPP's
/// <c>quad_perm:[a,b,c,d]</c>, <c>row_shl:n</c>, <c>row_shr:n</c>, <c>row_ror:n</c>, <c>wave_shl:1</c>,
/// <c>wave_rol:1</c>, <c>wave_shr:1</c>, <c>wave_ror:1</c>, <c>row_mirror</c>, <c>row_half_mirror</c>,
/// <c>row_bcast:15</c>, <c>row_bcast:31</c>, <c>row_mask:n</c>, <c>bank_mask:n</c>, <c>bound_ctrl:0</c> (or 1);
/// SDWA's <c>dst_sel:</c>, <c>dst_unused:</c>, <c>src0_sel:</c>, <c>src1_sel:</c>.</item>
/// </list>
/// A kernel's text (<see cref="AssembleKernel"/>, in <c>GcnAssembler.Kernel.cs</c>) adds virtual registers (a
/// <c>%</c> operand: <c>%v12</c>, <c>%s3</c>, a range <c>%v[4:7]</c> of adjacent ones) and directives (a line starting
/// with <c>.</c>); <see cref="Assemble"/>, which encodes physical registers as written, refuses both by line and
/// column. Anything unknown is refused the same way.
/// </summary>
internal sealed partial class GcnAssembler
{
    // A label is (its byte offset or -1, its name's start and length in _names, the name's hash); a branch is (its byte
    // offset, its label, its line).
    private const int LabelSlot = 4, BranchSlot = 3, CharBuffer = 128;

    private byte[] _names;
    private int _namesLength;
    private char[] _chars;
    private bool _parsing;
    private int _line, _column;
    private Columns _columns;

    // Each operand's column, for the refusals.
    [InlineArray(GcnSignature.MaxOperands)]
    private struct Columns
    {
        private int _first;
    }

    // The characters of numbers and of identifiers (labels, registers, modifiers).
    private static readonly SearchValues<byte> NameBytes = SearchValues.Create("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$"u8);

    // The bytes a comment may start with (';', or '/' of '//'). The searches here are non-generic (rules 88, 89).
    private static readonly SearchValues<byte> CommentBytes = SearchValues.Create(";/"u8);

    /// <summary>
    /// Assembles GCN text (UTF-8): each line's label and instruction, appended to the code; labels are this assembler's,
    /// across calls, and lines are counted across calls (<see cref="Reset"/> starts again).
    /// </summary>
    /// <exception cref="GcnSyntaxException">A line is refused: its line, column and reason.</exception>
    public void Assemble(ReadOnlySpan<byte> text)
    {
        _parsing = true;
        try
        {
            while (true)
            {
                int newline = text.IndexOf((byte)'\n');
                ReadOnlySpan<byte> line = newline < 0 ? text : text[..newline];
                _line++;
                Statement(line.Length > 0 && line[^1] == (byte)'\r' ? line[..^1] : line);
                if (newline < 0)
                {
                    break;
                }

                text = text[(newline + 1)..];
            }
        }
        finally
        {
            _parsing = false;
        }
    }

    // One line: an optional label, then an optional instruction.
    private void Statement(ReadOnlySpan<byte> line)
    {
        line = line[..CommentStart(line)];
        int at = SkipSpaces(line, 0);
        if (at == line.Length)
        {
            return;
        }

        int end = NameEnd(line, at);
        if (end > at && end < line.Length && line[end] == (byte)':' && !IsDigit(line[at]))
        {
            DefineLabel(line[at..end], at);
            at = SkipSpaces(line, end + 1);
            if (at == line.Length)
            {
                return;
            }

            end = NameEnd(line, at);
        }

        if (line[at] == (byte)'.')
        {
            if (!_kernel)
            {
                throw Syntax(at, "directives (a line starting with '.') are taken in a kernel's text (AssembleKernel), where the register allocator reads them");
            }

            Directive(line, at);
            return;
        }

        if (end == at)
        {
            throw Syntax(at, $"expected an instruction, found '{Text(line[at..(at + 1)])}'");
        }

        _column = at + 1;
        var ins = Lookup(line[at..end], at, out var variant);
        Begin(ins, variant, typed: false);
        if ((ins.Needs & ~_features) != 0)
        {
            throw Syntax(at, Missing(ins));
        }

        at = ParseOperands(line, end);
        ParseModifiers(line, at);
        if (_kernel)
        {
            Record();
        }
        else
        {
            Encode();
            if (Printing)
            {
                PrintInstruction();
            }
        }
    }

    // The family's instruction for a mnemonic, its encoding suffix taken off.
    private GcnInstruction Lookup(ReadOnlySpan<byte> mnemonic, int at, out GcnVariant variant)
    {
        variant = GcnVariant.Auto;
        if (mnemonic.Length <= CharBuffer && Ascii.ToLower(mnemonic, _chars, out int length) == OperationStatus.Done)
        {
            ReadOnlySpan<char> name = _chars.AsSpan(0, length);
            var suffix = name.EndsWith("_e32") ? GcnVariant.E32
                : name.EndsWith("_e64") ? GcnVariant.E64
                : name.EndsWith("_dpp") ? GcnVariant.Dpp
                : name.EndsWith("_sdwa") ? GcnVariant.Sdwa
                : GcnVariant.Auto;
            if (suffix != GcnVariant.Auto && _set.TryFind(name[..name.LastIndexOf('_')], out var stripped))
            {
                variant = suffix;
                return stripped;
            }

            if (_set.TryFind(name, out var instruction))
            {
                return instruction;
            }
        }

        throw Syntax(at, $"'{Text(mnemonic)}' is not an instruction of the {_family} family");
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Operands.
    // ---------------------------------------------------------------------------------------------------------------

    // The operands the instruction's signature lists, split by commas; an atomic without glc leaves out its destination.
    private int ParseOperands(ReadOnlySpan<byte> line, int at)
    {
        var sig = _ins.Operands;
        int first = 0;
        if (_ins.Has(GcnInstructionFlags.Atomic) && sig.Role(0) == GcnRole.VectorDestination && CountOperands(line, at) == sig.Count - 1)
        {
            Add(default);
            _columns[0] = _column;
            first = 1;
        }

        for (int i = first; i < sig.Count; i++)
        {
            at = SkipSpaces(line, at);
            if (i > first)
            {
                if (at == line.Length || line[at] != (byte)',')
                {
                    throw Syntax(at, $"{_ins.Mnemonic} takes {sig.Count} operands: expected ',' and {RoleName(sig.Role(i))}");
                }

                at = SkipSpaces(line, at + 1);
            }

            if (at == line.Length)
            {
                throw Syntax(at, $"{_ins.Mnemonic}: expected {RoleName(sig.Role(i))}");
            }

            _columns[i] = at + 1;
            at = ParseOperand(line, at, i);
        }

        at = SkipSpaces(line, at);
        if (at < line.Length && line[at] == (byte)',')
        {
            throw Syntax(at, $"{_ins.Mnemonic} takes {sig.Count} operand(s), and more are written");
        }

        return at;
    }

    // Commas outside brackets and parentheses, up to the modifiers: the operands written.
    private static int CountOperands(ReadOnlySpan<byte> line, int at)
    {
        int depth = 0, count = 1;
        for (int i = at; i < line.Length; i++)
        {
            switch (line[i])
            {
                case (byte)'[' or (byte)'(':
                    depth++;
                    break;
                case (byte)']' or (byte)')':
                    depth--;
                    break;
                case (byte)',' when depth == 0:
                    count++;
                    break;
            }
        }

        return count;
    }

    private int ParseOperand(ReadOnlySpan<byte> line, int at, int i)
    {
        switch (_ins.Operands.Role(i))
        {
            case GcnRole.WaitCounts:
                return ParseWaitCounts(line, at);
            case GcnRole.HardwareRegister:
                return ParseHardwareRegister(line, at);
            case GcnRole.GprIndex:
                return ParseGprIndex(line, at);
            case GcnRole.Target when !IsNumberStart(line, at):
                {
                    int end = NameEnd(line, at);
                    if (end == at)
                    {
                        throw Syntax(at, "expected a label or a number");
                    }

                    Add(GcnOperand.Target(new GcnLabel(FindLabel(line[at..end]))));
                    return end;
                }

            case GcnRole.Immediate16 or GcnRole.Literal32 or GcnRole.Constant32 or GcnRole.Target:
                {
                    int end = ParseNumber(line, at, out bool isFloat, out long integer, out double real);
                    Add(GcnOperand.Immediate(unchecked((int)ImmediateBits(at, isFloat, integer, real))));
                    return end;
                }

            case GcnRole.MemoryOffset when IsNumberStart(line, at):
                {
                    int end = ParseNumber(line, at, out bool isFloat, out long integer, out _);
                    if (isFloat || integer is < int.MinValue or > int.MaxValue)
                    {
                        throw Syntax(at, "an offset is an integer");
                    }

                    Add(GcnOperand.Immediate((int)integer));
                    return end;
                }

            default:
                return ParseSource(line, at, i);
        }
    }

    // A register, special register or constant, with -, | |, neg(), abs() or sext() around it.
    private int ParseSource(ReadOnlySpan<byte> line, int at, int i)
    {
        if (line[at] == (byte)'-' && !IsNumberStart(line, at))
        {
            _negMask |= 1 << i;
            at++;
        }

        if (line[at] == (byte)'|')
        {
            _absMask |= 1 << i;
            at = ParseValue(line, at + 1, i);
            return Expect(line, at, (byte)'|');
        }

        int name = NameEnd(line, at);
        if (name < line.Length && line[name] == (byte)'(')
        {
            ReadOnlySpan<byte> function = line[at..name];
            if (function.SequenceEqual("neg"u8) || function.SequenceEqual("abs"u8) || function.SequenceEqual("sext"u8))
            {
                int bit = 1 << i;
                if (function[0] == (byte)'n')
                {
                    _negMask |= bit;
                }
                else if (function[0] == (byte)'a')
                {
                    _absMask |= bit;
                }
                else
                {
                    _sextMask |= bit;
                }

                at = ParseSource(line, SkipSpaces(line, name + 1), i);
                return Expect(line, SkipSpaces(line, at), (byte)')');
            }
        }

        return ParseValue(line, at, i);
    }

    // A register, a special register or value, or a constant for operand i.
    private int ParseValue(ReadOnlySpan<byte> line, int at, int i)
    {
        if (at == line.Length)
        {
            throw Syntax(at, "expected an operand");
        }

        if (IsNumberStart(line, at))
        {
            int end = ParseNumber(line, at, out bool isFloat, out long integer, out double real);
            Add(Constant(at, i, isFloat, integer, real));
            return end;
        }

        if (line[at] == (byte)'%')
        {
            if (!_kernel)
            {
                throw Syntax(at, $"'{Text(line[at..NameEnd(line, at + 1)])}' is a virtual register: Assemble takes physical registers (a kernel's text, AssembleKernel, takes virtual ones)");
            }

            return ParseVirtual(line, at);
        }

        int nameEnd = NameEnd(line, at);
        ReadOnlySpan<byte> name = line[at..nameEnd];
        if (nameEnd < line.Length && line[nameEnd] == (byte)'[')
        {
            var kind = RegisterFile(name);
            if (kind == GcnOperandKind.None)
            {
                throw Syntax(at, $"'{Text(name)}' is not a register file (s, v, a, ttmp)");
            }

            int close = line[nameEnd..].IndexOf((byte)']');
            if (close < 0)
            {
                throw Syntax(nameEnd, "a register range ends with ']'");
            }

            close += nameEnd;
            ReadOnlySpan<byte> range = line[(nameEnd + 1)..close];
            int colon = range.IndexOf((byte)':');
            if (!TryIndex(colon < 0 ? range : range[..colon], out int low) || !TryIndex(colon < 0 ? range : range[(colon + 1)..], out int high) || high < low)
            {
                throw Syntax(nameEnd, $"'{Text(line[at..(close + 1)])}' is not a register range (low:high)");
            }

            Add(Register(kind, low, high - low + 1));
            return close + 1;
        }

        var single = RegisterFile(name, out int index);
        if (single != GcnOperandKind.None)
        {
            Add(Register(single, index, 1));
            return nameEnd;
        }

        if (name.Length <= CharBuffer && Ascii.ToUtf16(name, _chars, out int length) == OperationStatus.Done && SpecialNamed(_chars.AsSpan(0, length), out var special))
        {
            Add(GcnOperand.Special(special));
            return nameEnd;
        }

        throw Syntax(at, name.IsEmpty ? $"expected an operand, found '{Text(line[at..(at + 1)])}'" : $"'{Text(name)}' is not an operand {_family} knows");
    }

    private GcnOperand Register(GcnOperandKind kind, int index, int count) => kind switch
    {
        GcnOperandKind.Sgpr => GcnOperand.S(index, count),
        GcnOperandKind.Vgpr => GcnOperand.V(index, count),
        GcnOperandKind.Agpr => GcnOperand.A(index, count),
        _ => GcnOperand.Ttmp(index, count),
    };

    private static GcnOperandKind RegisterFile(ReadOnlySpan<byte> name) => name switch
    {
        [(byte)'s'] => GcnOperandKind.Sgpr,
        [(byte)'v'] => GcnOperandKind.Vgpr,
        [(byte)'a'] => GcnOperandKind.Agpr,
        _ when name.SequenceEqual("ttmp"u8) => GcnOperandKind.Ttmp,
        _ => GcnOperandKind.None,
    };

    // s7, v255, a1, ttmp3: a file's letter (or ttmp) and digits.
    private static GcnOperandKind RegisterFile(ReadOnlySpan<byte> name, out int index)
    {
        index = 0;
        int digits = name.Length > 4 && name.StartsWith("ttmp"u8) ? 4 : 1;
        if (name.Length <= digits || !TryIndex(name[digits..], out index))
        {
            return GcnOperandKind.None;
        }

        return RegisterFile(name[..digits]);
    }

    private static bool TryIndex(ReadOnlySpan<byte> digits, out int index)
    {
        int first = SkipSpaces(digits, 0), last = digits.Length;
        while (last > first && digits[last - 1] == (byte)' ')
        {
            last--;
        }

        digits = digits[first..last];
        index = 0;
        if (digits.IsEmpty || digits.Length > 4 || (digits.Length > 1 && digits[0] == (byte)'0'))
        {
            return false;
        }

        foreach (byte b in digits)
        {
            if (!IsDigit(b))
            {
                return false;
            }

            index = 10 * index + (b - '0');
        }

        return true;
    }

    private static bool SpecialNamed(ReadOnlySpan<char> name, out GcnSpecial special)
    {
        special = name switch
        {
            "vcc" => GcnSpecial.Vcc,
            "vcc_lo" => GcnSpecial.VccLo,
            "vcc_hi" => GcnSpecial.VccHi,
            "exec" => GcnSpecial.Exec,
            "exec_lo" => GcnSpecial.ExecLo,
            "exec_hi" => GcnSpecial.ExecHi,
            "m0" => GcnSpecial.M0,
            "off" => GcnSpecial.Off,
            "flat_scratch" => GcnSpecial.FlatScratch,
            "flat_scratch_lo" => GcnSpecial.FlatScratchLo,
            "flat_scratch_hi" => GcnSpecial.FlatScratchHi,
            "xnack_mask" => GcnSpecial.XnackMask,
            "xnack_mask_lo" => GcnSpecial.XnackMaskLo,
            "xnack_mask_hi" => GcnSpecial.XnackMaskHi,
            "scc" or "src_scc" => GcnSpecial.Scc,
            "vccz" or "src_vccz" => GcnSpecial.Vccz,
            "execz" or "src_execz" => GcnSpecial.Execz,
            "shared_base" or "src_shared_base" => GcnSpecial.SharedBase,
            "shared_limit" or "src_shared_limit" => GcnSpecial.SharedLimit,
            "private_base" or "src_private_base" => GcnSpecial.PrivateBase,
            "private_limit" or "src_private_limit" => GcnSpecial.PrivateLimit,
            "pops_exiting_wave_id" or "src_pops_exiting_wave_id" => GcnSpecial.PopsExitingWaveId,
            "lds_direct" or "src_lds_direct" => GcnSpecial.LdsDirect,
            _ => (GcnSpecial)byte.MaxValue,
        };
        return special != (GcnSpecial)byte.MaxValue;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Constants: a written number as the operand's width and type encode it.
    // ---------------------------------------------------------------------------------------------------------------

    // A source's constant: inline when the hardware holds it for the source's type, a literal otherwise.
    private GcnOperand Constant(int at, int i, bool isFloat, long integer, double real)
    {
        int width = _ins.Operands.Width(i);
        var type = _ins.SourceType;
        if (width == 2)
        {
            return Constant64(at, isFloat, integer, real, type == GcnValueType.Double);
        }

        if (type is GcnValueType.Half or GcnValueType.Short)
        {
            return Constant16(at, isFloat, integer, real, type == GcnValueType.Half);
        }

        if (isFloat)
        {
            return GcnOperand.Float((float)real);
        }

        return integer is >= int.MinValue and <= uint.MaxValue
            ? GcnOperand.Bits(unchecked((uint)integer))
            : throw Syntax(at, $"{integer} does not fit 32 bits");
    }

    // 16-bit sources: the integers -16 to 64 and float16's ±0.5, ±1, ±2, ±4, 1/(2π) inline; else a 16-bit literal.
    private GcnOperand Constant16(int at, bool isFloat, long integer, double real, bool half)
    {
        if (isFloat)
        {
            if (!half)
            {
                throw Syntax(at, "a float for a 16-bit integer source");
            }

            int code = InlineFloat(real, (float)real == 0.15915494f);
            if (code >= 0)
            {
                return GcnOperand.Inline(code, BitConverter.HalfToUInt16Bits((Half)real));
            }

            var value = (Half)real;
            return Half.IsInfinity(value) && !double.IsInfinity(real)
                ? throw Syntax(at, $"{real.ToString(CultureInfo.InvariantCulture)} is past float16's range")
                : GcnOperand.LiteralOf(BitConverter.HalfToUInt16Bits(value));
        }

        if (integer is >= -16 and <= 64)
        {
            return GcnOperand.Bits(unchecked((uint)(int)integer));
        }

        if (integer is < short.MinValue or > ushort.MaxValue)
        {
            throw Syntax(at, $"{integer} does not fit 16 bits");
        }

        ushort bits = unchecked((ushort)integer);
        int inline = half ? bits switch
        {
            0x3800 => 240,
            0xB800 => 241,
            0x3C00 => 242,
            0xBC00 => 243,
            0x4000 => 244,
            0xC000 => 245,
            0x4400 => 246,
            0xC400 => 247,
            0x3118 => 248,
            _ => -1,
        } : -1;
        return inline >= 0 ? GcnOperand.Inline(inline, bits) : GcnOperand.LiteralOf(bits);
    }

    // 64-bit sources: the integers -16 to 64 and float64's inline values; else a 32-bit literal (an integer that fits
    // it, or a float64 whose low half is zero, as its high half).
    private GcnOperand Constant64(int at, bool isFloat, long integer, double real, bool isDouble)
    {
        if (isFloat)
        {
            if (!isDouble)
            {
                throw Syntax(at, "a float for a 64-bit integer source");
            }

            int code = InlineFloat(real, real == 0.15915494309189532);
            if (code >= 0)
            {
                return GcnOperand.Inline(code, 0);
            }

            ulong bits = BitConverter.DoubleToUInt64Bits(real);
            return (uint)bits == 0 ? GcnOperand.LiteralOf((uint)(bits >> 32)) : throw Syntax(at, "a float64 literal holds its high 32 bits only: this one needs more");
        }

        if (integer is >= -16 and <= 64)
        {
            return GcnOperand.Bits(unchecked((uint)(int)integer));
        }

        if (isDouble)
        {
            int code = InlineFloat(BitConverter.Int64BitsToDouble(integer), BitConverter.Int64BitsToDouble(integer) == 0.15915494309189532);
            if (code >= 0)
            {
                return GcnOperand.Inline(code, 0);
            }
        }

        return integer is >= int.MinValue and <= uint.MaxValue
            ? GcnOperand.LiteralOf(unchecked((uint)integer))
            : throw Syntax(at, $"{integer} does not fit the 32-bit literal of a 64-bit source");
    }

    // The inline float codes by value: 0.5 240, -0.5 241, 1 242, -1 243, 2 244, -2 245, 4 246, -4 247, 1/(2π) 248.
    private static int InlineFloat(double value, bool inverseTwoPi) => value switch
    {
        0.5 => 240,
        -0.5 => 241,
        1.0 => 242,
        -1.0 => 243,
        2.0 => 244,
        -2.0 => 245,
        4.0 => 246,
        -4.0 => 247,
        _ => inverseTwoPi ? 248 : -1,
    };

    // An immediate field's 32 bits: an integer, or a float's float32 bits.
    private uint ImmediateBits(int at, bool isFloat, long integer, double real)
    {
        if (isFloat)
        {
            return BitConverter.SingleToUInt32Bits((float)real);
        }

        return integer is >= int.MinValue and <= uint.MaxValue ? unchecked((uint)integer) : throw Syntax(at, $"{integer} does not fit 32 bits");
    }

    private static bool IsNumberStart(ReadOnlySpan<byte> line, int at)
    {
        if (at < line.Length && line[at] is (byte)'-' or (byte)'+')
        {
            at++;
        }

        return at < line.Length && (IsDigit(line[at]) || (line[at] == (byte)'.' && at + 1 < line.Length && IsDigit(line[at + 1])));
    }

    // A signed integer (decimal, 0x hex, 0b binary) or float (with a '.' or an exponent); the end of it.
    private int ParseNumber(ReadOnlySpan<byte> line, int at, out bool isFloat, out long integer, out double real)
    {
        isFloat = false;
        integer = 0;
        real = 0;
        int start = at;
        bool negative = false;
        if (at < line.Length && line[at] is (byte)'-' or (byte)'+')
        {
            negative = line[at] == (byte)'-';
            at++;
        }

        int digits = at;
        if (at + 1 < line.Length && line[at] == (byte)'0' && (line[at + 1] | 0x20) is (byte)'x' or (byte)'b')
        {
            bool hex = (line[at + 1] | 0x20) == (byte)'x';
            int end = at + 2;
            while (end < line.Length && (hex ? char.IsAsciiHexDigit((char)line[end]) : line[end] is (byte)'0' or (byte)'1'))
            {
                end++;
            }

            if (end == at + 2 || !ulong.TryParse(line[(at + 2)..end], hex ? NumberStyles.AllowHexSpecifier : NumberStyles.AllowBinarySpecifier, CultureInfo.InvariantCulture, out ulong value)
                || value > long.MaxValue)
            {
                throw Syntax(start, $"'{Text(line[start..NameEnd(line, at)])}' is not a number Idrak reads (64 bits at most)");
            }

            integer = negative ? -(long)value : (long)value;
            return end;
        }

        int last = at;
        while (last < line.Length && (IsDigit(line[last]) || line[last] == (byte)'.' || (line[last] | 0x20) == (byte)'e'
                                      || (line[last] is (byte)'-' or (byte)'+' && (line[last - 1] | 0x20) == (byte)'e')))
        {
            isFloat |= !IsDigit(line[last]);
            last++;
        }

        ReadOnlySpan<byte> number = line[digits..last];
        bool read = isFloat
            ? double.TryParse(number, NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, CultureInfo.InvariantCulture, out real)
            : long.TryParse(number, NumberStyles.None, CultureInfo.InvariantCulture, out integer);
        if (!read)
        {
            throw Syntax(start, $"'{Text(line[start..Math.Max(last, start + 1)])}' is not a number");
        }

        real = negative ? -real : real;
        integer = negative ? -integer : integer;
        return last;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Fields written as functions: s_waitcnt's counts, hwreg(), gpr_idx(), swizzle().
    // ---------------------------------------------------------------------------------------------------------------

    // vmcnt(n) expcnt(n) lgkmcnt(n), split by spaces, '&' or ','; a counter not named is not waited on. Or the 16 bits.
    private int ParseWaitCounts(ReadOnlySpan<byte> line, int at)
    {
        if (IsNumberStart(line, at))
        {
            int end = ParseNumber(line, at, out bool isFloat, out long integer, out _);
            if (isFloat || integer is < 0 or > ushort.MaxValue)
            {
                throw Syntax(at, "s_waitcnt's 16 bits are 0 to 65535");
            }

            Add(GcnOperand.Immediate((int)integer));
            return end;
        }

        int vm = 63, exp = 7, lgkm = 15, named = 0;
        while (true)
        {
            at = SkipSpaces(line, at);
            if (at < line.Length && line[at] is (byte)'&' or (byte)',')
            {
                at = SkipSpaces(line, at + 1);
            }

            int name = NameEnd(line, at);
            if (name == at || name >= line.Length || line[name] != (byte)'(')
            {
                break;
            }

            ReadOnlySpan<byte> counter = line[at..name];
            int which = counter.SequenceEqual("vmcnt"u8) ? 0 : counter.SequenceEqual("expcnt"u8) ? 1 : counter.SequenceEqual("lgkmcnt"u8) ? 2 : -1;
            if (which < 0 || (named >> which & 1) != 0)
            {
                throw Syntax(at, which < 0 ? $"'{Text(counter)}' is not a counter (vmcnt, expcnt, lgkmcnt)" : $"{Text(counter)} is named twice");
            }

            int most = which switch { 0 => 63, 1 => 7, _ => 15 };
            int end = ParseNumber(line, SkipSpaces(line, name + 1), out bool isFloat, out long count, out _);
            if (isFloat || count < 0 || count > most)
            {
                throw Syntax(name + 1, $"{Text(counter)} is 0 to {most}");
            }

            named |= 1 << which;
            switch (which)
            {
                case 0:
                    vm = (int)count;
                    break;
                case 1:
                    exp = (int)count;
                    break;
                default:
                    lgkm = (int)count;
                    break;
            }

            at = Expect(line, SkipSpaces(line, end), (byte)')');
        }

        if (named == 0)
        {
            throw Syntax(at, "s_waitcnt waits for vmcnt(n), expcnt(n) or lgkmcnt(n), or takes their 16 bits");
        }

        Add(GcnOperand.Immediate(WaitBits(vm, exp, lgkm)));
        return at;
    }

    // hwreg(name or id[, offset, size]): the id at 5:0, the offset at 10:6, the size - 1 at 15:11 (all 32 bits by default).
    private int ParseHardwareRegister(ReadOnlySpan<byte> line, int at)
    {
        int open = Function(line, at, "hwreg"u8);
        int id;
        int end;
        if (IsNumberStart(line, open))
        {
            end = ParseNumber(line, open, out bool isFloat, out long value, out _);
            id = isFloat || value is < 0 or > 63 ? throw Syntax(open, "a hardware register's id is 0 to 63") : (int)value;
        }
        else
        {
            end = NameEnd(line, open);
            ReadOnlySpan<byte> name = line[open..end];
            id = name.Length <= CharBuffer && Ascii.ToUtf16(name, _chars, out int length) == OperationStatus.Done ? _set.HardwareRegister(_chars.AsSpan(0, length)) : -1;
            if (id < 0)
            {
                throw Syntax(open, $"'{Text(name)}' is not a hardware register of the {_family} family");
            }
        }

        int offset = 0, size = 32;
        end = SkipSpaces(line, end);
        if (end < line.Length && line[end] == (byte)',')
        {
            int at2 = SkipSpaces(line, end + 1);
            end = ParseNumber(line, at2, out bool f1, out long o, out _);
            end = Expect(line, SkipSpaces(line, end), (byte)',');
            int at3 = SkipSpaces(line, end);
            end = ParseNumber(line, at3, out bool f2, out long s, out _);
            if (f1 || f2 || o is < 0 or > 31 || s is < 1 or > 32)
            {
                throw Syntax(at2, "hwreg's offset is 0 to 31 and its size 1 to 32 bits");
            }

            (offset, size) = ((int)o, (int)s);
        }

        Add(GcnOperand.Immediate(id | offset << 6 | (size - 1) << 11));
        return Expect(line, SkipSpaces(line, end), (byte)')');
    }

    // gpr_idx(SRC0, SRC1, SRC2, DST): the operands M0 indexes, bits 0 to 3; or the 4 bits.
    private int ParseGprIndex(ReadOnlySpan<byte> line, int at)
    {
        if (IsNumberStart(line, at))
        {
            int end = ParseNumber(line, at, out bool isFloat, out long value, out _);
            Add(GcnOperand.Immediate(isFloat || value is < 0 or > 15 ? throw Syntax(at, "gpr_idx's bits are 0 to 15") : (int)value));
            return end;
        }

        int open = Function(line, at, "gpr_idx"u8);
        int bits = 0;
        at = SkipSpaces(line, open);
        while (at < line.Length && line[at] != (byte)')')
        {
            int end = NameEnd(line, at);
            ReadOnlySpan<byte> name = line[at..end];
            int bit = name.SequenceEqual("SRC0"u8) ? 1 : name.SequenceEqual("SRC1"u8) ? 2 : name.SequenceEqual("SRC2"u8) ? 4 : name.SequenceEqual("DST"u8) ? 8 : 0;
            if (bit == 0 || (bits & bit) != 0)
            {
                throw Syntax(at, $"'{Text(name)}' is not one of gpr_idx's SRC0, SRC1, SRC2, DST, or is named twice");
            }

            bits |= bit;
            at = SkipSpaces(line, end);
            if (at < line.Length && line[at] == (byte)',')
            {
                at = SkipSpaces(line, at + 1);
            }
        }

        Add(GcnOperand.Immediate(bits));
        return Expect(line, at, (byte)')');
    }

    // swizzle(QUAD_PERM,a,b,c,d), swizzle(BITMASK_PERM,"01pip"), swizzle(SWAP,n), swizzle(REVERSE,n),
    // swizzle(BROADCAST,size,lane): ds_swizzle_b32's 16-bit offset (a quad permutation with bit 15, else the and, or and
    // xor masks of the lane id at 4:0, 9:5 and 14:10).
    private int ParseSwizzle(ReadOnlySpan<byte> line, int at, out int offset)
    {
        int open = Function(line, at, "swizzle"u8);
        int end = NameEnd(line, open);
        ReadOnlySpan<byte> mode = line[open..end];
        at = Expect(line, SkipSpaces(line, end), (byte)',');
        Span<int> values = stackalloc int[4];
        if (mode.SequenceEqual("BITMASK_PERM"u8))
        {
            at = SkipSpaces(line, at);
            if (at + 7 > line.Length || line[at] != (byte)'"' || line[at + 6] != (byte)'"')
            {
                throw Syntax(at, "BITMASK_PERM takes five characters in quotes, each 0, 1, p or i (bit 4 first)");
            }

            int and = 0, or = 0, xor = 0;
            for (int k = 0; k < 5; k++)
            {
                int bit = 1 << (4 - k);
                switch (line[at + 1 + k])
                {
                    case (byte)'0':
                        break;
                    case (byte)'1':
                        or |= bit;
                        break;
                    case (byte)'p':
                        and |= bit;
                        break;
                    case (byte)'i':
                        and |= bit;
                        xor |= bit;
                        break;
                    default:
                        throw Syntax(at + 1 + k, "BITMASK_PERM's characters are 0, 1, p or i");
                }
            }

            offset = and | or << 5 | xor << 10;
            return Expect(line, SkipSpaces(line, at + 7), (byte)')');
        }

        int count = 0;
        while (true)
        {
            at = SkipSpaces(line, at);
            int next = ParseNumber(line, at, out bool isFloat, out long value, out _);
            if (isFloat || count == values.Length || value is < 0 or > 32)
            {
                throw Syntax(at, "swizzle's numbers are integers 0 to 32");
            }

            values[count++] = (int)value;
            at = SkipSpaces(line, next);
            if (at < line.Length && line[at] == (byte)',')
            {
                at++;
                continue;
            }

            break;
        }

        at = Expect(line, at, (byte)')');
        bool power(int v) => v > 0 && (v & (v - 1)) == 0;
        if (mode.SequenceEqual("QUAD_PERM"u8) && count == 4 && values[0] < 4 && values[1] < 4 && values[2] < 4 && values[3] < 4)
        {
            offset = 0x8000 | values[0] | values[1] << 2 | values[2] << 4 | values[3] << 6;
        }
        else if (mode.SequenceEqual("SWAP"u8) && count == 1 && power(values[0]) && values[0] <= 16)
        {
            offset = 0x1F | values[0] << 10;
        }
        else if (mode.SequenceEqual("REVERSE"u8) && count == 1 && power(values[0]) && values[0] is >= 2 and <= 32)
        {
            offset = 0x1F | (values[0] - 1) << 10;
        }
        else if (mode.SequenceEqual("BROADCAST"u8) && count == 2 && power(values[0]) && values[0] is >= 2 and <= 32 && values[1] < values[0])
        {
            offset = (0x1F & ~(values[0] - 1)) | values[1] << 5;
        }
        else
        {
            throw Syntax(open, "swizzle takes QUAD_PERM and four lanes 0-3, BITMASK_PERM and a mask, SWAP and a group of 1-16, REVERSE and a group of 2-32, or BROADCAST, a group of 2-32 and a lane in it");
        }

        return at;
    }

    // name( : the position after the parenthesis.
    private int Function(ReadOnlySpan<byte> line, int at, ReadOnlySpan<byte> name)
    {
        int end = NameEnd(line, at);
        if (!line[at..end].SequenceEqual(name) || end >= line.Length || line[end] != (byte)'(')
        {
            throw Syntax(at, $"expected {Encoding.UTF8.GetString(name)}(...)");
        }

        return SkipSpaces(line, end + 1);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Modifiers.
    // ---------------------------------------------------------------------------------------------------------------

    // The modifiers after the operands, split by spaces: each checked against the format here, against the instruction
    // by the encoder.
    private void ParseModifiers(ReadOnlySpan<byte> line, int at)
    {
        var format = _ins.Format;
        bool vector = format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc or GcnFormat.Vop3 or GcnFormat.Vop3p;
        bool e32Family = format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc;
        bool memory = format is GcnFormat.Ds or GcnFormat.Flat or GcnFormat.Global;
        while (true)
        {
            at = SkipSpaces(line, at);
            if (at == line.Length)
            {
                return;
            }

            int start = at, end = NameEnd(line, at);
            if (end == at)
            {
                throw Syntax(at, $"expected a modifier, found '{Text(line[at..(at + 1)])}'");
            }

            ReadOnlySpan<byte> name = line[at..end];
            bool valued = end < line.Length && line[end] == (byte)':';
            at = valued ? end + 1 : end;
            if (name.Length > CharBuffer || Ascii.ToUtf16(name, _chars, out int length) != OperationStatus.Done)
            {
                throw Syntax(start, "unknown modifier");
            }

            ReadOnlySpan<char> key = _chars.AsSpan(0, length);
            bool dpp = key is "quad_perm" or "row_shl" or "row_shr" or "row_ror" or "wave_shl" or "wave_rol" or "wave_shr" or "wave_ror" or "row_mirror"
                or "row_half_mirror" or "row_bcast" or "row_mask" or "bank_mask" or "bound_ctrl";
            bool sdwa = key is "dst_sel" or "dst_unused" or "src0_sel" or "src1_sel";
            bool allowed = key switch
            {
                "glc" => format is GcnFormat.Smem or GcnFormat.Flat or GcnFormat.Global,
                "slc" => format is GcnFormat.Flat or GcnFormat.Global,
                "gds" or "offset0" or "offset1" => format == GcnFormat.Ds,
                "offset" => memory,
                "clamp" or "mul" or "div" => vector,
                "op_sel" => format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc or GcnFormat.Vop3 or GcnFormat.Vop3p,
                "op_sel_hi" or "neg_lo" or "neg_hi" => format == GcnFormat.Vop3p,
                _ when dpp => e32Family && format != GcnFormat.Vopc && _variant is GcnVariant.Auto or GcnVariant.Dpp,
                _ when sdwa => e32Family && _variant is GcnVariant.Auto or GcnVariant.Sdwa,
                _ => throw Syntax(start, $"'{Text(name)}' is not a modifier Idrak knows"),
            };
            if (!allowed)
            {
                throw Syntax(start, $"{_ins.Mnemonic}{SuffixText(_variant)} takes no {Text(name)}");
            }

            bool flag = key is "glc" or "slc" or "gds" or "clamp" or "row_mirror" or "row_half_mirror";
            if (flag == valued)
            {
                throw Syntax(start, flag ? $"{Text(name)} takes no value" : $"{Text(name)} takes a value ({Text(name)}:...)");
            }

            switch (key)
            {
                case "glc":
                    _m.Glc = true;
                    break;
                case "slc":
                    _m.Slc = true;
                    break;
                case "gds":
                    _m.Gds = true;
                    break;
                case "clamp":
                    _m.Clamp = true;
                    break;
                case "mul" or "div":
                    {
                        int value = Integer(line, ref at, 2, 4);
                        _m.Omod = key is "div" ? value == 2 ? 3 : throw Syntax(start, "the output divides by 2 only (div:2)") : value == 2 ? 1 : value == 4 ? 2 : throw Syntax(start, "the output multiplies by 2 or 4 (mul:2, mul:4)");
                        break;
                    }

                case "offset":
                    if (line[at..NameEnd(line, at)].SequenceEqual("swizzle"u8))
                    {
                        at = ParseSwizzle(line, at, out _m.Offset);
                        _m.Swizzle = true;
                    }
                    else
                    {
                        _m.Offset = Integer(line, ref at, -0x100000, 0xFFFFF);
                    }

                    break;
                case "offset0":
                    _m.Offset0 = Integer(line, ref at, 0, 255);
                    break;
                case "offset1":
                    _m.Offset1 = Integer(line, ref at, 0, 255);
                    break;
                case "op_sel":
                    _m.OpSel = Bits(line, ref at, out _m.OpSelCount, 4);
                    break;
                case "op_sel_hi":
                    _m.OpSelHi = Bits(line, ref at, out int given, 3);
                    _m.OpSelHiGiven = (1 << given) - 1;
                    break;
                case "neg_lo":
                    _m.NegLo = Bits(line, ref at, out _, 3);
                    break;
                case "neg_hi":
                    _m.NegHi = Bits(line, ref at, out _, 3);
                    break;
                case "row_mask":
                    _m.RowMask = Integer(line, ref at, 0, 15);
                    break;
                case "bank_mask":
                    _m.BankMask = Integer(line, ref at, 0, 15);
                    break;
                case "bound_ctrl":
                    Integer(line, ref at, 0, 1);
                    _m.BoundCtrl = true;                                            // LLVM reads bound_ctrl:0 and :1 alike
                    break;
                case "dst_sel":
                    _m.DstSel = Select(line, ref at);
                    _m.DstGiven = true;
                    break;
                case "dst_unused":
                    _m.DstUnused = Unused(line, ref at);
                    _m.DstGiven = true;
                    break;
                case "src0_sel":
                    _m.Src0Sel = Select(line, ref at);
                    break;
                case "src1_sel":
                    _m.Src1Sel = Select(line, ref at);
                    break;
                default:
                    if (_m.DppControl >= 0)
                    {
                        throw Syntax(start, "a DPP form takes one control");
                    }

                    _m.DppControl = key switch
                    {
                        "quad_perm" => QuadPermutation(line, ref at),
                        "row_shl" => 0x100 + Integer(line, ref at, 1, 15),
                        "row_shr" => 0x110 + Integer(line, ref at, 1, 15),
                        "row_ror" => 0x120 + Integer(line, ref at, 1, 15),
                        "wave_shl" => WaveShift(line, ref at, 0x130),
                        "wave_rol" => WaveShift(line, ref at, 0x134),
                        "wave_shr" => WaveShift(line, ref at, 0x138),
                        "wave_ror" => WaveShift(line, ref at, 0x13C),
                        "row_mirror" => 0x140,
                        "row_half_mirror" => 0x141,
                        _ => Integer(line, ref at, 15, 31) switch
                        {
                            15 => 0x142,
                            31 => 0x143,
                            _ => throw Syntax(start, "row_bcast is 15 or 31"),
                        },
                    };
                    break;
            }

            if (sdwa)
            {
                _m.SdwaGiven = true;
            }

            if (at < line.Length && line[at] is not ((byte)' ' or (byte)'\t'))
            {
                throw Syntax(at, $"expected a space after {Text(name)}");
            }
        }
    }

    private static string SuffixText(GcnVariant variant) => variant switch
    {
        GcnVariant.E32 => "_e32",
        GcnVariant.E64 => "_e64",
        GcnVariant.Dpp => "_dpp",
        GcnVariant.Sdwa => "_sdwa",
        _ => "",
    };

    // An integer value from lowest to highest.
    private int Integer(ReadOnlySpan<byte> line, ref int at, int lowest, int highest)
    {
        int start = at;
        at = ParseNumber(line, at, out bool isFloat, out long value, out _);
        return !isFloat && value >= lowest && value <= highest ? (int)value : throw Syntax(start, $"expected an integer from {lowest} to {highest}");
    }

    // [b0,b1,...]: up to `most` bits of 0 or 1; how many were written.
    private int Bits(ReadOnlySpan<byte> line, ref int at, out int count, int most)
    {
        int bits = 0;
        count = 0;
        at = Expect(line, at, (byte)'[');
        while (true)
        {
            at = SkipSpaces(line, at);
            int start = at;
            int bit = Integer(line, ref at, 0, 1);
            if (count == most)
            {
                throw Syntax(start, $"at most {most} values");
            }

            bits |= bit << count++;
            at = SkipSpaces(line, at);
            if (at < line.Length && line[at] == (byte)',')
            {
                at++;
                continue;
            }

            at = Expect(line, at, (byte)']');
            return bits;
        }
    }

    // wave_shl:1 and the other whole-wave moves: by one lane only.
    private int WaveShift(ReadOnlySpan<byte> line, ref int at, int control)
    {
        Integer(line, ref at, 1, 1);
        return control;
    }

    // quad_perm:[a,b,c,d], each lane 0-3: the DPP control a | b << 2 | c << 4 | d << 6.
    private int QuadPermutation(ReadOnlySpan<byte> line, ref int at)
    {
        at = Expect(line, at, (byte)'[');
        int control = 0;
        for (int k = 0; k < 4; k++)
        {
            if (k > 0)
            {
                at = Expect(line, SkipSpaces(line, at), (byte)',');
            }

            at = SkipSpaces(line, at);
            control |= Integer(line, ref at, 0, 3) << (2 * k);
        }

        at = Expect(line, SkipSpaces(line, at), (byte)']');
        return control;
    }

    // SDWA's selections: BYTE_0-3 0-3, WORD_0 4, WORD_1 5, DWORD 6.
    private int Select(ReadOnlySpan<byte> line, ref int at)
    {
        int end = NameEnd(line, at);
        ReadOnlySpan<byte> name = line[at..end];
        int sel = name.Length == 6 && name.StartsWith("BYTE_"u8) && name[5] is >= (byte)'0' and <= (byte)'3' ? name[5] - '0'
            : name.SequenceEqual("WORD_0"u8) ? 4
            : name.SequenceEqual("WORD_1"u8) ? 5
            : name.SequenceEqual("DWORD"u8) ? 6
            : throw Syntax(at, $"'{Text(name)}' is not a selection (BYTE_0-3, WORD_0, WORD_1, DWORD)");
        at = end;
        return sel;
    }

    // SDWA's dst_unused: UNUSED_PAD 0, UNUSED_SEXT 1, UNUSED_PRESERVE 2.
    private int Unused(ReadOnlySpan<byte> line, ref int at)
    {
        int end = NameEnd(line, at);
        ReadOnlySpan<byte> name = line[at..end];
        int unused = name.SequenceEqual("UNUSED_PAD"u8) ? 0
            : name.SequenceEqual("UNUSED_SEXT"u8) ? 1
            : name.SequenceEqual("UNUSED_PRESERVE"u8) ? 2
            : throw Syntax(at, $"'{Text(name)}' is not UNUSED_PAD, UNUSED_SEXT or UNUSED_PRESERVE");
        at = end;
        return unused;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Labels by name.
    // ---------------------------------------------------------------------------------------------------------------

    private int AddLabel(int start, int length, int hash)
    {
        Grow(ref _labels, LabelSlot * (_labelCount + 1));
        int slot = LabelSlot * _labelCount;
        _labels[slot] = -1;
        _labels[slot + 1] = start;
        _labels[slot + 2] = length;
        _labels[slot + 3] = hash;
        return _labelCount++;
    }

    // The label of that name, made (unbound) on its first mention.
    private int FindLabel(ReadOnlySpan<byte> name)
    {
        int hash = (int)Hash(name);
        for (int id = 0; id < _labelCount; id++)
        {
            int slot = LabelSlot * id;
            if (_labels[slot + 3] == hash && _labels[slot + 2] == name.Length && name.Length > 0 && _names.AsSpan(_labels[slot + 1], name.Length).SequenceEqual(name))
            {
                return id;
            }
        }

        Grow(ref _names, _namesLength + name.Length);
        name.CopyTo(_names.AsSpan(_namesLength));
        _namesLength += name.Length;
        return AddLabel(_namesLength - name.Length, name.Length, hash);
    }

    private void DefineLabel(ReadOnlySpan<byte> name, int at)
    {
        int id = FindLabel(name);
        if (_labels[LabelSlot * id] >= 0)
        {
            throw Syntax(at, $"label '{Text(name)}' is defined twice");
        }

        _labels[LabelSlot * id] = _kernel ? _itemCount : _length;
        if (_kernel)
        {
            Grow(ref _labelOrder, _labelOrderCount + 1);
            _labelOrder[_labelOrderCount++] = id;
        }
        else if (Printing)
        {
            PrintLabel(id);
        }
    }

    private string LabelName(int id) => Encoding.UTF8.GetString(_names, _labels[LabelSlot * id + 1], _labels[LabelSlot * id + 2]);

    // FNV-1a over the name's bytes.
    private static uint Hash(ReadOnlySpan<byte> name)
    {
        uint hash = 2166136261;
        foreach (byte b in name)
        {
            hash = (hash ^ b) * 16777619;
        }

        return hash;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Characters.
    // ---------------------------------------------------------------------------------------------------------------

    // Where a comment (';' or '//') starts, or the line's end.
    private static int CommentStart(ReadOnlySpan<byte> line)
    {
        for (int at = 0; at < line.Length;)
        {
            int found = line[at..].IndexOfAny(CommentBytes);
            if (found < 0)
            {
                break;
            }

            at += found;
            if (line[at] == (byte)';' || (at + 1 < line.Length && line[at + 1] == (byte)'/'))
            {
                return at;
            }

            at++;
        }

        return line.Length;
    }

    private static int SkipSpaces(ReadOnlySpan<byte> line, int at)
    {
        while (at < line.Length && line[at] is (byte)' ' or (byte)'\t')
        {
            at++;
        }

        return at;
    }

    // The end of the name (letters, digits, '_', '.', '$') starting at `at`.
    private static int NameEnd(ReadOnlySpan<byte> line, int at)
    {
        if (at >= line.Length)
        {
            return at;
        }

        int length = line[at..].IndexOfAnyExcept(NameBytes);
        return length < 0 ? line.Length : at + length;
    }

    private int Expect(ReadOnlySpan<byte> line, int at, byte expected) => at < line.Length && line[at] == expected
        ? at + 1
        : throw Syntax(at, $"expected '{(char)expected}'");

    private static bool IsDigit(byte b) => b is >= (byte)'0' and <= (byte)'9';

    private static string RoleName(GcnRole role) => role switch
    {
        GcnRole.ScalarDestination or GcnRole.ScalarResult => "a scalar destination",
        GcnRole.ScalarSource or GcnRole.ScalarOnlySource => "a scalar source",
        GcnRole.Immediate16 or GcnRole.Literal32 or GcnRole.Constant32 => "a constant",
        GcnRole.HardwareRegister => "hwreg(...)",
        GcnRole.GprIndex => "gpr_idx(...)",
        GcnRole.WaitCounts => "the counts",
        GcnRole.Target => "a label",
        GcnRole.MemoryBase => "the base registers",
        GcnRole.MemoryOffset => "the offset",
        GcnRole.VectorDestination or GcnRole.AccumulationDestination => "a destination",
        GcnRole.LaneMask or GcnRole.LaneMaskSource => "a lane mask (vcc or s[n:n+1])",
        GcnRole.Address => "the address",
        GcnRole.Data0 or GcnRole.Data1 => "the data",
        GcnRole.ScalarAddress => "the scalar address or off",
        _ => "a source",
    };

    // The text of a few bytes, for a refusal (made only when one is thrown).
    private static string Text(ReadOnlySpan<byte> bytes) => Encoding.UTF8.GetString(bytes);

    private GcnSyntaxException Syntax(int at, string reason) => new(_line, at + 1, reason);
}
