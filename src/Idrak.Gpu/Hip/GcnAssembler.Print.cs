// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;
using System.Text;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

// The printer: each instruction as encoded, in LLVM's syntax (what llvm-mc reads), one a line, labels on their own
// lines. The form chosen is written as its suffix (_e32, _e64, _dpp, _sdwa; none for an instruction with no VOP3
// form), constants as their value (integers, the inline floats, other bits in hex), registers physical. Printed text
// reads back, through Assemble, to the same bytes. UTF-8 into a pooled buffer, numbers formatted into it: nothing is
// allocated per instruction.
internal sealed partial class GcnAssembler
{
    private byte[] _print = [];
    private int _printLength;

    /// <summary>Whether <see cref="Assemble"/> also prints what it encodes into <see cref="Printed"/> (a kernel's text always is).</summary>
    public bool Printing { get; set; }

    /// <summary>The code printed so far, with physical registers, as UTF-8 (LLVM's syntax; <see cref="Reset"/> clears it).</summary>
    public ReadOnlySpan<byte> Printed => _print.AsSpan(0, _printLength);

    /// <summary>The code printed so far, as a string.</summary>
    public string PrintedText => Encoding.UTF8.GetString(Printed);

    private void PrintLabel(int id)
    {
        int length = _labels[LabelSlot * id + 2];
        if (length == 0)
        {
            Print(".L"u8);
            PrintInt(id);
        }
        else
        {
            Print(_names.AsSpan(_labels[LabelSlot * id + 1], length));
        }

        Print(":\n"u8);
    }

    // The instruction just encoded.
    private void PrintInstruction()
    {
        var ins = _ins;
        Print("  "u8);
        PrintAscii(ins.Mnemonic);
        if (ins.Format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc)
        {
            Print(_encoded switch
            {
                GcnVariant.E64 => "_e64"u8,
                GcnVariant.Dpp => "_dpp"u8,
                GcnVariant.Sdwa => "_sdwa"u8,
                _ => ins.Has(GcnInstructionFlags.E64) ? "_e32"u8 : ""u8,
            });
        }

        bool first = true;
        for (int i = 0; i < _count; i++)
        {
            if (_ops[i].Kind == GcnOperandKind.None)
            {
                continue;                                                      // an atomic without glc has no destination
            }

            Print(first ? " "u8 : ", "u8);
            first = false;
            PrintOperand(i);
        }

        PrintModifiers();
        Print("\n"u8);
    }

    private void PrintOperand(int i)
    {
        var operand = _ops[i];
        switch (_ins.Operands.Role(i))
        {
            case GcnRole.WaitCounts:
                {
                    var (vm, exp, lgkm) = _set.WaitCounts((ushort)operand.Value);
                    bool all = vm == 63 && exp == 7 && lgkm == 15;
                    bool space = false;
                    PrintCount("vmcnt("u8, vm, all || vm != 63, ref space);
                    PrintCount("expcnt("u8, exp, all || exp != 7, ref space);
                    PrintCount("lgkmcnt("u8, lgkm, all || lgkm != 15, ref space);
                    return;
                }

            case GcnRole.HardwareRegister:
                {
                    uint value = operand.Value;
                    Print("hwreg("u8);
                    PrintInt((int)(value & 63));
                    int offset = (int)(value >> 6 & 31), size = (int)(value >> 11 & 31) + 1;
                    if (offset != 0 || size != 32)
                    {
                        Print(", "u8);
                        PrintInt(offset);
                        Print(", "u8);
                        PrintInt(size);
                    }

                    Print(")"u8);
                    return;
                }

            case GcnRole.Literal32 or GcnRole.Constant32:
                PrintHex(operand.Value);
                return;
            case GcnRole.MemoryOffset when operand.Kind is GcnOperandKind.Immediate or GcnOperandKind.Constant or GcnOperandKind.Literal:
                PrintHex(operand.Value);
                return;
        }

        if (operand.Kind == GcnOperandKind.Label)
        {
            PrintLabelName(operand.Index);
            return;
        }

        bool neg = (_negMask >> i & 1) != 0, abs = (_absMask >> i & 1) != 0, sext = (_sextMask >> i & 1) != 0;
        bool constant = operand.Kind is GcnOperandKind.Constant or GcnOperandKind.Literal;
        Print(neg ? constant ? "neg("u8 : "-"u8 : ""u8);
        Print(abs ? "|"u8 : sext ? "sext("u8 : ""u8);
        PrintValue(operand);
        Print(abs ? "|"u8 : sext ? ")"u8 : ""u8);
        Print(neg && constant ? ")"u8 : ""u8);
    }

    private void PrintCount(ReadOnlySpan<byte> name, int count, bool shown, ref bool space)
    {
        if (!shown)
        {
            return;
        }

        Print(space ? " "u8 : ""u8);
        space = true;
        Print(name);
        PrintInt(count);
        Print(")"u8);
    }

    private void PrintLabelName(int id)
    {
        int length = _labels[LabelSlot * id + 2];
        if (length == 0)
        {
            Print(".L"u8);
            PrintInt(id);
        }
        else
        {
            Print(_names.AsSpan(_labels[LabelSlot * id + 1], length));
        }
    }

    private void PrintValue(GcnOperand operand)
    {
        switch (operand.Kind)
        {
            case GcnOperandKind.Sgpr or GcnOperandKind.Vgpr or GcnOperandKind.Agpr or GcnOperandKind.Ttmp:
                Print(operand.Kind switch { GcnOperandKind.Sgpr => "s"u8, GcnOperandKind.Vgpr => "v"u8, GcnOperandKind.Agpr => "a"u8, _ => "ttmp"u8 });
                if (operand.Count == 1)
                {
                    PrintInt(operand.Index);
                }
                else
                {
                    Print("["u8);
                    PrintInt(operand.Index);
                    Print(":"u8);
                    PrintInt(operand.Index + operand.Count - 1);
                    Print("]"u8);
                }

                return;
            case GcnOperandKind.Special:
                PrintAscii(GcnOperand.SpecialName((GcnSpecial)operand.Index));
                return;
            case GcnOperandKind.Constant when operand.Index <= 192:
                PrintInt(operand.Index - 128);
                return;
            case GcnOperandKind.Constant when operand.Index <= 208:
                PrintInt(192 - operand.Index);
                return;
            case GcnOperandKind.Constant:
                Print(operand.Index switch
                {
                    240 => "0.5"u8,
                    241 => "-0.5"u8,
                    242 => "1.0"u8,
                    243 => "-1.0"u8,
                    244 => "2.0"u8,
                    245 => "-2.0"u8,
                    246 => "4.0"u8,
                    247 => "-4.0"u8,
                    _ => _ins.SourceType == GcnValueType.Double ? "0.15915494309189532"u8 : "0.15915494"u8,
                });
                return;
            case GcnOperandKind.Literal:
                PrintHex(operand.Value);
                return;
            case GcnOperandKind.Immediate:
                PrintInt(unchecked((int)operand.Value));
                return;
            default:
                Print("?"u8);
                return;
        }
    }

    // The modifiers the encoder took, each as Assemble reads it back.
    private void PrintModifiers()
    {
        var format = _ins.Format;
        switch (format)
        {
            case GcnFormat.Smem:
                PrintFlag(_m.Glc, " glc"u8);
                return;
            case GcnFormat.Ds:
                if (_ins.Has(GcnInstructionFlags.TwoOffsets))
                {
                    PrintNumber(_m.Offset0 != 0, " offset0:"u8, _m.Offset0);
                    PrintNumber(_m.Offset1 != 0, " offset1:"u8, _m.Offset1);
                }
                else
                {
                    PrintNumber(_m.Offset != 0, " offset:"u8, _m.Offset);
                }

                PrintFlag(_m.Gds, " gds"u8);
                return;
            case GcnFormat.Flat or GcnFormat.Global:
                PrintNumber(_m.Offset != 0, " offset:"u8, _m.Offset);
                PrintFlag(_m.Glc, " glc"u8);
                PrintFlag(_m.Slc, " slc"u8);
                return;
            case GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc or GcnFormat.Vop3 or GcnFormat.Vop3p:
                break;
            default:
                return;
        }

        PrintFlag(_m.Clamp, " clamp"u8);
        Print(_m.Omod switch { 1 => " mul:2"u8, 2 => " mul:4"u8, 3 => " div:2"u8, _ => ""u8 });
        if (_encoded == GcnVariant.Sdwa)
        {
            PrintSdwa();
        }

        PrintBits(_m.OpSelCount > 0, " op_sel:["u8, _m.OpSel, _m.OpSelCount);
        PrintBits(_m.OpSelHiGiven != 0, " op_sel_hi:["u8, _m.OpSelHi, BitOperations.PopCount((uint)_m.OpSelHiGiven));
        int sources = Place().Sources;
        PrintBits(_m.NegLo != 0, " neg_lo:["u8, _m.NegLo, sources);
        PrintBits(_m.NegHi != 0, " neg_hi:["u8, _m.NegHi, sources);
        if (_encoded == GcnVariant.Dpp)
        {
            PrintDpp();
        }
    }

    private void PrintSdwa()
    {
        bool compare = _ins.Format == GcnFormat.Vopc;
        if (!compare)
        {
            Print(" dst_sel:"u8);
            PrintSelect(_m.DstSel);
            Print(_m.DstUnused switch { 0 => " dst_unused:UNUSED_PAD"u8, 1 => " dst_unused:UNUSED_SEXT"u8, _ => " dst_unused:UNUSED_PRESERVE"u8 });
        }

        Print(" src0_sel:"u8);
        PrintSelect(_m.Src0Sel);
        if (Place().Src1 >= 0)
        {
            Print(" src1_sel:"u8);
            PrintSelect(_m.Src1Sel);
        }
    }

    private void PrintSelect(int select) => Print(select switch
    {
        0 => "BYTE_0"u8,
        1 => "BYTE_1"u8,
        2 => "BYTE_2"u8,
        3 => "BYTE_3"u8,
        4 => "WORD_0"u8,
        5 => "WORD_1"u8,
        _ => "DWORD"u8,
    });

    private void PrintDpp()
    {
        int control = _m.DppControl;
        switch (control)
        {
            case <= 0xFF:
                Print(" quad_perm:["u8);
                for (int k = 0; k < 4; k++)
                {
                    Print(k > 0 ? ","u8 : ""u8);
                    PrintInt(control >> (2 * k) & 3);
                }

                Print("]"u8);
                break;
            case >= 0x101 and <= 0x10F:
                Print(" row_shl:"u8);
                PrintInt(control - 0x100);
                break;
            case >= 0x111 and <= 0x11F:
                Print(" row_shr:"u8);
                PrintInt(control - 0x110);
                break;
            case >= 0x121 and <= 0x12F:
                Print(" row_ror:"u8);
                PrintInt(control - 0x120);
                break;
            default:
                Print(control switch
                {
                    0x130 => " wave_shl:1"u8,
                    0x134 => " wave_rol:1"u8,
                    0x138 => " wave_shr:1"u8,
                    0x13C => " wave_ror:1"u8,
                    0x140 => " row_mirror"u8,
                    0x141 => " row_half_mirror"u8,
                    0x142 => " row_bcast:15"u8,
                    _ => " row_bcast:31"u8,
                });
                break;
        }

        Print(" row_mask:"u8);
        PrintHex((uint)_m.RowMask);
        Print(" bank_mask:"u8);
        PrintHex((uint)_m.BankMask);
        PrintFlag(_m.BoundCtrl, " bound_ctrl:0"u8);
    }

    private void PrintFlag(bool on, ReadOnlySpan<byte> text)
    {
        if (on)
        {
            Print(text);
        }
    }

    private void PrintNumber(bool shown, ReadOnlySpan<byte> name, int value)
    {
        if (shown)
        {
            Print(name);
            PrintInt(value);
        }
    }

    private void PrintBits(bool shown, ReadOnlySpan<byte> name, int bits, int count)
    {
        if (!shown)
        {
            return;
        }

        Print(name);
        for (int k = 0; k < count; k++)
        {
            Print(k > 0 ? ","u8 : ""u8);
            PrintInt(bits >> k & 1);
        }

        Print("]"u8);
    }

    private void Print(ReadOnlySpan<byte> text)
    {
        Grow(ref _print, _printLength + text.Length);
        text.CopyTo(_print.AsSpan(_printLength));
        _printLength += text.Length;
    }

    private void PrintAscii(string text)
    {
        Grow(ref _print, _printLength + text.Length);
        _printLength += Encoding.ASCII.GetBytes(text, _print.AsSpan(_printLength));
    }

    private void PrintInt(int value)
    {
        Grow(ref _print, _printLength + 16);
        value.TryFormat(_print.AsSpan(_printLength), out int written, default, CultureInfo.InvariantCulture);
        _printLength += written;
    }

    private void PrintHex(uint value)
    {
        Grow(ref _print, _printLength + 16);
        _print[_printLength++] = (byte)'0';
        _print[_printLength++] = (byte)'x';
        value.TryFormat(_print.AsSpan(_printLength), out int written, "x", CultureInfo.InvariantCulture);
        _printLength += written;
    }
}
