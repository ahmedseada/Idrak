// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Idrak.Tests.Gcn;

// The GCN interpreter's decoder (plan 35, G1b): gfx9 machine words to instructions, written from the instruction
// formats and from what LLVM's disassembler prints for probe words (GcnInterpreter.Opcodes.cs), apart from Idrak's
// encoder (GcnAssembler, its opcode tables): the interpreter is the check on the encoder, so it keeps its own reading of
// the ISA. An encoding it does not know is refused with its offset and words, never guessed.

/// <summary>The gfx9 processors the interpreter decodes for: the opcode table's columns.</summary>
internal enum GcnCpu
{
    Gfx900,
    Gfx906,
    Gfx908,
    Gfx90c,
}

/// <summary>The gfx9 instruction formats the decoder reads.</summary>
internal enum GcnFormat : byte
{
    Sop1,
    Sop2,
    Sopk,
    Sopc,
    Sopp,
    Smem,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vop3p,
    Ds,
    Flat,
    Global,
    Scratch,
}

/// <summary>A VOP1, VOP2 or VOPC instruction's form: its own 32-bit word, or a DPP or SDWA word after it.</summary>
internal enum GcnVariant : byte
{
    None,
    Dpp,
    Sdwa,
}

/// <summary>An operand field of an instruction, as the opcode table's templates name them.</summary>
internal enum GcnField : byte
{
    D,
    SD,
    S0,
    S1,
    S2,
    A,
    B,
    K,
    O,
    Fixed,
}

/// <summary>One operand of an opcode's template: its field, register class (v, s or a) and width in registers, or fixed text.</summary>
internal readonly record struct GcnSlot(GcnField Field, char Class, int Width, string Text);

/// <summary>An opcode of one format on one processor: LLVM's mnemonic, its operand template and its semantics.</summary>
internal sealed class GcnOpcode
{
    public required GcnFormat Format { get; init; }

    public required int Code { get; init; }

    /// <summary>The mnemonic as LLVM prints it (with _e32 or _e64 where LLVM writes one).</summary>
    public required string Name { get; init; }

    /// <summary>The mnemonic without _e32 or _e64: the key of its semantics, and the stem of its _dpp and _sdwa forms.</summary>
    public required string Base { get; init; }

    public required GcnSlot[] Slots { get; init; }

    /// <summary>FLAT's template when glc is set (an atomic then returns the old value), when it differs.</summary>
    public GcnSlot[]? GlcSlots { get; init; }

    /// <summary>The immediate's kind: hex16, hwreg, lit, dec, waitcnt, sendmsg, gpridx; empty when there is none.</summary>
    public string Immediate { get; init; } = "";

    /// <summary>The offset's kind: 16 (one 16-bit offset), 8x2 (offset0 and offset1), swz (a swizzle pattern), hex (SMEM); empty when there is none.</summary>
    public string OffsetKind { get; init; } = "";

    /// <summary>VOP3P's mixed-precision forms (fma_mix, mad_mix), whose op_sel_hi defaults to 0, not 1.</summary>
    public bool Mix { get; init; }

    /// <summary>How many of S0, S1 and S2 the template prints.</summary>
    public int Sources { get; init; }

    /// <summary>The semantics, or null when the interpreter does not model this opcode (running it is refused).</summary>
    internal GcnExecute? Execute { get; set; }

    public override string ToString() => $"{Format} 0x{Code:X} {Name}";
}

/// <summary>The opcode table (GcnInterpreter.Opcodes.cs), read once: per processor, per format, per opcode.</summary>
internal static partial class GcnOpcodeTable
{
    private static readonly GcnOpcode?[][][] Opcodes = Build();

    /// <summary>The opcode, or null when the processor has none there.</summary>
    internal static GcnOpcode? Find(GcnCpu cpu, GcnFormat format, int code)
    {
        var table = Opcodes[(int)cpu][(int)format];
        return (uint)code < (uint)table.Length ? table[code] : null;
    }

    /// <summary>Every opcode of a processor (for the tests' counts).</summary>
    internal static IEnumerable<GcnOpcode> All(GcnCpu cpu) => Opcodes[(int)cpu].SelectMany(t => t).OfType<GcnOpcode>();

    private static GcnOpcode?[][][] Build()
    {
        string[] formats = ["sop1", "sop2", "sopk", "sopc", "sopp", "smem", "vop1", "vop2", "vopc", "vop3", "vop3p", "ds", "flat", "global", "scratch"];
        int cpuCount = Enum.GetValues<GcnCpu>().Length;
        var result = new GcnOpcode?[cpuCount][][];
        for (int c = 0; c < cpuCount; c++)
        {
            result[c] = new GcnOpcode?[formats.Length][];
            for (int f = 0; f < formats.Length; f++)
            {
                result[c][f] = new GcnOpcode?[1024];
            }
        }

        foreach (string raw in Text.Split('\n'))
        {
            string line = raw.Trim();
            if (line.Length == 0)
            {
                continue;
            }

            string[] halves = line.Split(" | ");
            string[] parts = halves[0].Split(' ');
            int format = Array.IndexOf(formats, parts[0]);
            if (format < 0)
            {
                throw new InvalidOperationException($"opcode table: unknown format in '{line}'");
            }

            int code = int.Parse(parts[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            string mask = parts[2];
            string name = parts[3];
            var slots = Template(parts.AsSpan(4), out string immediate, out string offset, out bool mix);
            var glc = halves.Length > 1 ? Template(halves[1].Split(' '), out _, out _, out _) : null;
            string stem = name.EndsWith("_e32", StringComparison.Ordinal) || name.EndsWith("_e64", StringComparison.Ordinal) ? name[..^4] : name;
            for (int c = 0; c < cpuCount; c++)
            {
                if (mask[c] != '1')
                {
                    continue;
                }

                result[c][format][code] = new GcnOpcode
                {
                    Format = (GcnFormat)format,
                    Code = code,
                    Name = name,
                    Base = stem,
                    Slots = slots,
                    GlcSlots = glc,
                    Immediate = immediate,
                    OffsetKind = offset,
                    Mix = mix,
                    Sources = slots.Count(s => s.Field is GcnField.S0 or GcnField.S1 or GcnField.S2),
                    Execute = GcnSemantics.Find(stem),
                };
            }
        }

        return result;
    }

    private static GcnSlot[] Template(ReadOnlySpan<string> tokens, out string immediate, out string offset, out bool mix)
    {
        immediate = "";
        offset = "";
        mix = false;
        var slots = new List<GcnSlot>();
        foreach (string token in tokens)
        {
            if (token == "-")
            {
                continue;
            }

            if (token == "mix")
            {
                mix = true;
            }
            else if (token[0] == '=')
            {
                slots.Add(new GcnSlot(GcnField.Fixed, ' ', 0, token[1..]));
            }
            else
            {
                int colon = token.IndexOf(':', StringComparison.Ordinal);
                string field = token[..colon];
                string kind = token[(colon + 1)..];
                switch (field)
                {
                    case "K":
                        immediate = immediate.Length == 0 ? kind : immediate + "," + kind;
                        slots.Add(new GcnSlot(GcnField.K, ' ', 0, kind));
                        break;
                    case "O":
                        offset = kind;
                        break;
                    default:
                        var f = Enum.Parse<GcnField>(field);
                        if (kind is "gpridx")
                        {
                            immediate = "gpridx";
                            slots.Add(new GcnSlot(GcnField.K, ' ', 0, "gpridx"));
                        }
                        else
                        {
                            slots.Add(new GcnSlot(f, kind[0], int.Parse(kind.AsSpan(1), CultureInfo.InvariantCulture), ""));
                        }

                        break;
                }
            }
        }

        return [.. slots];
    }
}

/// <summary>
/// A decoded instruction. Register operands are held as gfx9's 9-bit source values: 0-255 the scalar registers and
/// constants (255 a literal), 256 + n vector register n (or accumulation register n where the template says 'a').
/// </summary>
internal sealed class GcnInstruction
{
    public required int Offset { get; init; }

    public required int Size { get; init; }

    public required GcnOpcode Opcode { get; init; }

    public required GcnCpu Cpu { get; init; }

    public GcnVariant Variant { get; init; }

    public uint Word0 { get; init; }

    public uint Word1 { get; init; }

    public int D { get; init; } = -1;

    public int SD { get; init; } = -1;

    public int S0 { get; init; } = -1;

    public int S1 { get; init; } = -1;

    public int S2 { get; init; } = -1;

    /// <summary>DS and FLAT: the address register; SMEM: the base register pair (or quad).</summary>
    public int A { get; init; } = -1;

    /// <summary>A literal constant (the 32-bit word after the instruction) when HasLiteral.</summary>
    public uint Literal { get; init; }

    public bool HasLiteral { get; init; }

    /// <summary>SOPP, SOPK: the 16-bit immediate.</summary>
    public int Simm16 { get; init; }

    // VOP3, VOP3P, DPP and SDWA modifiers (bit i for source i).
    public int Abs { get; init; }

    public int Neg { get; init; }

    public int NegHi { get; init; }

    public int OpSel { get; init; }

    public int OpSelHi { get; init; }

    public bool Clamp { get; init; }

    public int Omod { get; init; }

    // DPP.
    public int DppControl { get; init; }

    public int RowMask { get; init; }

    public int BankMask { get; init; }

    public bool BoundControl { get; init; }

    // SDWA (selections: 0-3 bytes, 4-5 words, 6 the whole dword).
    public int DstSel { get; init; } = 6;

    public int DstUnused { get; init; }

    public int Src0Sel { get; init; } = 6;

    public int Src1Sel { get; init; } = 6;

    public int Sext { get; init; }

    // Memory.
    public int MemoryOffset { get; init; }

    public int Offset0 { get; init; }

    public int Offset1 { get; init; }

    public bool Glc { get; init; }

    public bool Slc { get; init; }

    public bool Gds { get; init; }

    /// <summary>SMEM: the offset is an immediate (else a register, MemoryOffset its source value).</summary>
    public bool Imm { get; init; }

    /// <summary>GLOBAL: the scalar base pair's source value, or -1 for off (a 64-bit vector address).</summary>
    public int SAddr { get; init; } = -1;

    /// <summary>The template the operands follow (FLAT's glc template when it applies).</summary>
    public GcnSlot[] Slots => Glc && Opcode.GlcSlots is { } g ? g : Opcode.Slots;

    /// <summary>The mnemonic as LLVM prints it for this form.</summary>
    public string Name => Variant switch
    {
        GcnVariant.Dpp => Opcode.Base + "_dpp",
        GcnVariant.Sdwa => Opcode.Base + "_sdwa",
        _ => Opcode.Name,
    };

    private string? text;

    /// <summary>The instruction as LLVM prints it (a branch's target as its 16-bit dword count, unsigned, as LLVM prints it).</summary>
    public string Text => text ??= GcnPrinter.Print(this);

    public string Words => Size == 4 ? $"{Word0:X8}" : Size == 8 ? $"{Word0:X8} {Word1:X8}" : $"{Word0:X8} {Word1:X8} {Literal:X8}";

    public override string ToString() => $"0x{Offset:X4}: {Text}";
}

/// <summary>An encoding the decoder refuses: its offset, its words and why.</summary>
internal sealed class GcnDecodeException(string message) : Exception(message);

/// <summary>Decodes gfx9 machine code for one processor.</summary>
internal sealed class GcnDecoder(GcnCpu cpu)
{
    public GcnCpu Cpu => cpu;

    /// <summary>The instruction at <paramref name="offset"/> of <paramref name="code"/>.</summary>
    /// <exception cref="GcnDecodeException">The words are not an instruction of this processor the decoder knows, or are cut short.</exception>
    public GcnInstruction Decode(ReadOnlySpan<byte> code, int offset)
    {
        if (offset < 0 || offset % 4 != 0 || offset + 4 > code.Length)
        {
            throw new GcnDecodeException($"0x{offset:X4}: no instruction there (the code is {code.Length} bytes, instructions are 4-byte aligned)");
        }

        uint w0 = BinaryPrimitives.ReadUInt32LittleEndian(code[offset..]);
        uint w1 = offset + 8 <= code.Length ? BinaryPrimitives.ReadUInt32LittleEndian(code[(offset + 4)..]) : 0;
        bool hasW1 = offset + 8 <= code.Length;
        uint w2 = offset + 12 <= code.Length ? BinaryPrimitives.ReadUInt32LittleEndian(code[(offset + 8)..]) : 0;
        bool hasW2 = offset + 12 <= code.Length;

        if ((w0 & 0x80000000) == 0)
        {
            return DecodeVector(offset, w0, w1, hasW1, w2, hasW2);
        }

        if (w0 >> 30 == 2)
        {
            return DecodeScalar(offset, w0, w1, hasW1);
        }

        switch (w0 >> 26)
        {
            case 0x30:
                Need(hasW1, offset, w0, "SMEM");
                return DecodeSmem(offset, w0, w1);
            case 0x34:
                Need(hasW1, offset, w0, "VOP3");
                return w0 >> 23 == 0x1A7 ? DecodeVop3p(offset, w0, w1) : DecodeVop3(offset, w0, w1);
            case 0x36:
                Need(hasW1, offset, w0, "DS");
                return DecodeDs(offset, w0, w1);
            case 0x37:
                Need(hasW1, offset, w0, "FLAT");
                return DecodeFlat(offset, w0, w1);
            default:
                string kind = (w0 >> 26) switch
                {
                    0x31 => "EXP",
                    0x35 => "VINTRP",
                    0x38 => "MUBUF",
                    0x3A => "MTBUF",
                    0x3C => "MIMG",
                    _ => "a reserved encoding",
                };
                throw Refuse(offset, $"{w0:X8}", $"{kind}: not decoded by the interpreter");
        }
    }

    /// <summary>Every instruction of <paramref name="code"/> from offset 0 to its end, in order.</summary>
    public List<GcnInstruction> DecodeAll(ReadOnlySpan<byte> code)
    {
        var list = new List<GcnInstruction>();
        for (int offset = 0; offset < code.Length;)
        {
            var instruction = Decode(code, offset);
            list.Add(instruction);
            offset += instruction.Size;
        }

        return list;
    }

    private static void Need(bool present, int offset, uint w0, string format)
    {
        if (!present)
        {
            throw new GcnDecodeException($"0x{offset:X4}: {w0:X8}: a {format} instruction is 8 bytes, the code ends after 4");
        }
    }

    private GcnDecodeException Refuse(int offset, string words, string why) => new($"0x{offset:X4}: {words}: {why} ({cpu.ToString().ToLowerInvariant()})");

    private GcnOpcode Opcode(GcnFormat format, int code, int offset, string words) =>
        GcnOpcodeTable.Find(cpu, format, code) ?? throw Refuse(offset, words, $"no {format.ToString().ToUpperInvariant()} instruction with opcode 0x{code:X}");

    private GcnInstruction DecodeScalar(int offset, uint w0, uint w1, bool hasW1)
    {
        GcnFormat format;
        int code, d = -1, s0 = -1, s1 = -1, simm = 0;
        if (w0 >> 28 == 0xB)
        {
            switch (w0 >> 23)
            {
                case 0x17D:
                    format = GcnFormat.Sop1;
                    code = (int)(w0 >> 8) & 0xFF;
                    d = (int)(w0 >> 16) & 0x7F;
                    s0 = (int)w0 & 0xFF;
                    break;
                case 0x17E:
                    format = GcnFormat.Sopc;
                    code = (int)(w0 >> 16) & 0x7F;
                    s0 = (int)w0 & 0xFF;
                    s1 = (int)(w0 >> 8) & 0xFF;
                    break;
                case 0x17F:
                    format = GcnFormat.Sopp;
                    code = (int)(w0 >> 16) & 0x7F;
                    simm = (int)w0 & 0xFFFF;
                    break;
                default:
                    format = GcnFormat.Sopk;
                    code = (int)(w0 >> 23) & 0x1F;
                    d = (int)(w0 >> 16) & 0x7F;
                    simm = (int)w0 & 0xFFFF;
                    break;
            }
        }
        else
        {
            format = GcnFormat.Sop2;
            code = (int)(w0 >> 23) & 0x7F;
            d = (int)(w0 >> 16) & 0x7F;
            s0 = (int)w0 & 0xFF;
            s1 = (int)(w0 >> 8) & 0xFF;
        }

        var opcode = Opcode(format, code, offset, $"{w0:X8}");
        bool literal = s0 == 255 || s1 == 255 || opcode.Immediate.Contains("lit", StringComparison.Ordinal);
        if (format == GcnFormat.Sopc && opcode.Immediate == "gpridx")
        {
            literal = s0 == 255;
        }

        if (literal && !hasW1)
        {
            throw Refuse(offset, $"{w0:X8}", "a literal constant follows, but the code ends");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = literal ? 8 : 4,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = literal ? w1 : 0,
            D = d,
            S0 = s0,
            S1 = s1,
            Simm16 = simm,
            Literal = literal ? w1 : 0,
            HasLiteral = literal,
        };
    }

    private GcnInstruction DecodeVector(int offset, uint w0, uint w1, bool hasW1, uint w2, bool hasW2)
    {
        int op6 = (int)(w0 >> 25) & 0x3F;
        GcnFormat format;
        int code, d = -1, s0 = (int)w0 & 0x1FF, s1 = -1;
        if (op6 == 0x3F)
        {
            format = GcnFormat.Vop1;
            code = (int)(w0 >> 9) & 0xFF;
            d = (int)(w0 >> 17) & 0xFF;
        }
        else if (op6 == 0x3E)
        {
            format = GcnFormat.Vopc;
            code = (int)(w0 >> 17) & 0xFF;
            s1 = 256 + ((int)(w0 >> 9) & 0xFF);
        }
        else
        {
            format = GcnFormat.Vop2;
            code = op6;
            d = (int)(w0 >> 17) & 0xFF;
            s1 = 256 + ((int)(w0 >> 9) & 0xFF);
        }

        var opcode = Opcode(format, code, offset, $"{w0:X8}");
        int dest = d < 0 ? -1 : opcode.Slots.FirstOrDefault(s => s.Field == GcnField.D) is { Class: 's' } ? d : 256 + d;
        bool kLiteral = opcode.Immediate.Contains("lit", StringComparison.Ordinal);
        if (s0 is 0xFA or 0xF9)
        {
            if (kLiteral)
            {
                throw Refuse(offset, $"{w0:X8}", $"{opcode.Name} takes no DPP or SDWA word");
            }

            if (!hasW1)
            {
                throw Refuse(offset, $"{w0:X8}", "a DPP or SDWA word follows, but the code ends");
            }

            return s0 == 0xFA ? Dpp(offset, w0, w1, opcode, dest, s1) : Sdwa(offset, w0, w1, opcode, dest, s1, format);
        }

        bool literal = s0 == 255 || kLiteral;
        if (s0 == 255 && kLiteral)
        {
            throw Refuse(offset, $"{w0:X8}", $"{opcode.Name} holds one literal, its source 0 cannot be another");
        }

        if (literal && !hasW1)
        {
            throw Refuse(offset, $"{w0:X8}", "a literal constant follows, but the code ends");
        }

        _ = w2;
        _ = hasW2;
        return new GcnInstruction
        {
            Offset = offset,
            Size = literal ? 8 : 4,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = literal ? w1 : 0,
            D = dest,
            S0 = s0,
            S1 = s1,
            Literal = literal ? w1 : 0,
            HasLiteral = literal,
        };
    }

    private GcnInstruction Dpp(int offset, uint w0, uint w1, GcnOpcode opcode, int dest, int s1)
    {
        int control = (int)(w1 >> 8) & 0x1FF;
        bool valid = control <= 0xFF || control is (>= 0x101 and <= 0x10F) or (>= 0x111 and <= 0x11F) or (>= 0x121 and <= 0x12F)
                     or 0x130 or 0x134 or 0x138 or 0x13C or 0x140 or 0x141 or 0x142 or 0x143;
        if (!valid || (w1 & 0x00060000) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", $"DPP control 0x{control:X3} is not one gfx9 defines");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Variant = GcnVariant.Dpp,
            Word0 = w0,
            Word1 = w1,
            D = dest,
            S0 = 256 + ((int)w1 & 0xFF),
            S1 = s1,
            DppControl = control,
            BoundControl = (w1 & (1u << 19)) != 0,
            Neg = (int)((w1 >> 20) & 1) | (int)((w1 >> 22) & 1) << 1,
            Abs = (int)((w1 >> 21) & 1) | (int)((w1 >> 23) & 1) << 1,
            BankMask = (int)(w1 >> 24) & 0xF,
            RowMask = (int)(w1 >> 28) & 0xF,
        };
    }

    private GcnInstruction Sdwa(int offset, uint w0, uint w1, GcnOpcode opcode, int dest, int s1, GcnFormat format)
    {
        int src0 = (w1 & (1u << 23)) != 0 ? (int)w1 & 0xFF : 256 + ((int)w1 & 0xFF);
        int src1 = s1;
        if (s1 >= 0 && (w1 & (1u << 31)) != 0)
        {
            src1 = s1 - 256;
        }

        int sd = dest;
        bool clamp = false;
        int omod = 0, dstSel = 6, dstUnused = 0;
        if (format == GcnFormat.Vopc)
        {
            sd = (w1 & (1u << 15)) != 0 ? (int)(w1 >> 8) & 0x7F : 106;
        }
        else
        {
            dstSel = (int)(w1 >> 8) & 7;
            dstUnused = (int)(w1 >> 11) & 3;
            clamp = (w1 & (1u << 13)) != 0;
            omod = (int)(w1 >> 14) & 3;
        }

        int src0Sel = (int)(w1 >> 16) & 7, src1Sel = (int)(w1 >> 24) & 7;
        if (dstSel > 6 || src0Sel > 6 || src1Sel > 6 || dstUnused > 2 || (w1 & (1u << 22)) != 0 || (w1 & (1u << 30)) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "an SDWA selection or bit gfx9 does not define");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Variant = GcnVariant.Sdwa,
            Word0 = w0,
            Word1 = w1,
            D = format == GcnFormat.Vopc ? sd : dest,
            SD = format == GcnFormat.Vopc ? sd : -1,
            S0 = src0,
            S1 = src1,
            Clamp = clamp,
            Omod = omod,
            DstSel = dstSel,
            DstUnused = dstUnused,
            Src0Sel = src0Sel,
            Src1Sel = src1Sel,
            Sext = (int)((w1 >> 19) & 1) | (int)((w1 >> 27) & 1) << 1,
            Neg = (int)((w1 >> 20) & 1) | (int)((w1 >> 28) & 1) << 1,
            Abs = (int)((w1 >> 21) & 1) | (int)((w1 >> 29) & 1) << 1,
        };
    }

    private GcnInstruction DecodeVop3(int offset, uint w0, uint w1)
    {
        int code = (int)(w0 >> 16) & 0x3FF;
        var opcode = Opcode(GcnFormat.Vop3, code, offset, $"{w0:X8} {w1:X8}");
        bool vop3b = opcode.Slots.Any(s => s.Field == GcnField.SD);
        int s0 = (int)w1 & 0x1FF, s1 = (int)(w1 >> 9) & 0x1FF, s2 = (int)(w1 >> 18) & 0x1FF;
        foreach (int s in (ReadOnlySpan<int>)[s0, s1, s2])
        {
            if (s is 255 or 0xF9 or 0xFA)
            {
                throw Refuse(offset, $"{w0:X8} {w1:X8}", "a VOP3 source on gfx9 cannot be a literal, DPP or SDWA");
            }
        }

        var dSlot = opcode.Slots.FirstOrDefault(s => s.Field == GcnField.D);
        int d = (int)w0 & 0xFF;
        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            D = dSlot.Class == 's' ? d : 256 + d,
            SD = vop3b ? (int)(w0 >> 8) & 0x7F : -1,
            S0 = s0,
            S1 = s1,
            S2 = s2,
            Abs = vop3b ? 0 : (int)(w0 >> 8) & 7,
            OpSel = vop3b ? 0 : (int)(w0 >> 11) & 0xF,
            Clamp = (w0 & 0x8000) != 0,
            Omod = (int)(w1 >> 27) & 3,
            Neg = (int)(w1 >> 29) & 7,
        };
    }

    private GcnInstruction DecodeVop3p(int offset, uint w0, uint w1)
    {
        int code = (int)(w0 >> 16) & 0x7F;
        var opcode = Opcode(GcnFormat.Vop3p, code, offset, $"{w0:X8} {w1:X8}");
        int s0 = (int)w1 & 0x1FF, s1 = (int)(w1 >> 9) & 0x1FF, s2 = (int)(w1 >> 18) & 0x1FF;
        foreach (int s in (ReadOnlySpan<int>)[s0, s1, s2])
        {
            if (s is 255 or 0xF9 or 0xFA)
            {
                throw Refuse(offset, $"{w0:X8} {w1:X8}", "a VOP3P source on gfx9 cannot be a literal, DPP or SDWA");
            }
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            D = 256 + ((int)w0 & 0xFF),
            S0 = s0,
            S1 = s1,
            S2 = s2,
            NegHi = (int)(w0 >> 8) & 7,
            OpSel = (int)(w0 >> 11) & 7,
            OpSelHi = (int)(w1 >> 27) & 3 | (int)((w0 >> 14) & 1) << 2,
            Clamp = (w0 & 0x8000) != 0,
            Neg = (int)(w1 >> 29) & 7,
        };
    }

    private GcnInstruction DecodeSmem(int offset, uint w0, uint w1)
    {
        int code = (int)(w0 >> 18) & 0xFF;
        var opcode = Opcode(GcnFormat.Smem, code, offset, $"{w0:X8} {w1:X8}");
        if ((w0 & (3u << 14)) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "SMEM with soe or nv set is not decoded by the interpreter");
        }

        bool imm = (w0 & (1u << 17)) != 0;
        if (imm ? (w1 & 0xFFE00000) != 0 : (w1 & 0xFFFFFF00) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "SMEM offset bits gfx9 does not define");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            A = ((int)w0 & 0x3F) << 1,
            D = (int)(w0 >> 6) & 0x7F,
            Glc = (w0 & (1u << 16)) != 0,
            Imm = imm,
            MemoryOffset = imm ? (int)(w1 & 0x1FFFFF) : (int)(w1 & 0xFF),
        };
    }

    private GcnInstruction DecodeDs(int offset, uint w0, uint w1)
    {
        int code = (int)(w0 >> 17) & 0xFF;
        var opcode = Opcode(GcnFormat.Ds, code, offset, $"{w0:X8} {w1:X8}");
        if ((w0 & (1u << 25)) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "DS bit 25 is not defined on gfx9");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            Offset0 = (int)w0 & 0xFF,
            Offset1 = (int)(w0 >> 8) & 0xFF,
            MemoryOffset = (int)w0 & 0xFFFF,
            Gds = (w0 & (1u << 16)) != 0,
            A = 256 + ((int)w1 & 0xFF),
            S0 = 256 + ((int)(w1 >> 8) & 0xFF),
            S1 = 256 + ((int)(w1 >> 16) & 0xFF),
            D = 256 + ((int)(w1 >> 24) & 0xFF),
        };
    }

    private GcnInstruction DecodeFlat(int offset, uint w0, uint w1)
    {
        int segment = (int)(w0 >> 14) & 3;
        if (segment == 3)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "FLAT segment 3 is reserved");
        }

        if ((w0 & (1u << 13)) != 0 || (w0 & (1u << 25)) != 0 || (w1 & (1u << 23)) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "FLAT with lds or nv set is not decoded by the interpreter");
        }

        var format = segment switch { 0 => GcnFormat.Flat, 1 => GcnFormat.Scratch, _ => GcnFormat.Global };
        int code = (int)(w0 >> 18) & 0x7F;
        var opcode = Opcode(format, code, offset, $"{w0:X8} {w1:X8}");
        int imm = format == GcnFormat.Flat ? (int)w0 & 0xFFF : ((int)w0 & 0x1FFF) << 19 >> 19;
        if (format == GcnFormat.Flat && (w0 & 0x1000) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "FLAT's offset is 12 bits on gfx9");
        }

        int saddr = (int)(w1 >> 16) & 0x7F;
        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            MemoryOffset = imm,
            Glc = (w0 & (1u << 16)) != 0,
            Slc = (w0 & (1u << 17)) != 0,
            A = 256 + ((int)w1 & 0xFF),
            S0 = 256 + ((int)(w1 >> 8) & 0xFF),
            D = 256 + ((int)(w1 >> 24) & 0xFF),
            SAddr = format == GcnFormat.Flat || saddr == 0x7F ? -1 : saddr,
        };
    }
}

/// <summary>Prints a decoded instruction as LLVM 18's disassembler does (a branch's target as its 16-bit dword count, unsigned).</summary>
internal static class GcnPrinter
{
    private static readonly string[] Selections = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD"];

    private static readonly string[] Unused = ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"];

    internal static string Print(GcnInstruction i)
    {
        var text = new StringBuilder(i.Name);
        var op = i.Opcode;
        var slots = i.Slots;
        bool first = true;
        foreach (var slot in slots)
        {
            string part = Slot(i, slot);
            if (part.Length == 0)
            {
                continue;
            }

            text.Append(first ? " " : ", ").Append(part);
            first = false;
        }

        switch (op.Format)
        {
            case GcnFormat.Smem:
                if (op.OffsetKind == "hex")
                {
                    text.Append(", ").Append(i.Imm ? "0x" + i.MemoryOffset.ToString("x", CultureInfo.InvariantCulture) : GcnPrinter.Operand(i.MemoryOffset, 1, 's', i, false));
                }

                if (i.Glc)
                {
                    text.Append(" glc");
                }

                break;
            case GcnFormat.Vop3:
                if (i.OpSel != 0)
                {
                    int count = op.Sources;
                    text.Append(" op_sel:[");
                    for (int s = 0; s < count; s++)
                    {
                        text.Append((i.OpSel >> s) & 1).Append(',');
                    }

                    text.Append((i.OpSel >> 3) & 1).Append(']');
                }

                AppendClampOmod(text, i);
                break;
            case GcnFormat.Vop3p:
                AppendPacked(text, i);
                break;
            case GcnFormat.Ds:
                AppendDsOffsets(text, i);
                if (i.Gds)
                {
                    text.Append(" gds");
                }

                break;
            case GcnFormat.Flat or GcnFormat.Global or GcnFormat.Scratch:
                if (i.MemoryOffset != 0)
                {
                    text.Append(" offset:").Append(i.MemoryOffset.ToString(CultureInfo.InvariantCulture));
                }

                if (i.Glc)
                {
                    text.Append(" glc");
                }

                if (i.Slc)
                {
                    text.Append(" slc");
                }

                break;
            case GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc:
                if (i.Variant == GcnVariant.Dpp)
                {
                    text.Append(' ').Append(DppControl(i.DppControl))
                        .Append(" row_mask:0x").Append(i.RowMask.ToString("x", CultureInfo.InvariantCulture))
                        .Append(" bank_mask:0x").Append(i.BankMask.ToString("x", CultureInfo.InvariantCulture));
                    if (i.BoundControl)
                    {
                        text.Append(" bound_ctrl:1");
                    }
                }
                else if (i.Variant == GcnVariant.Sdwa)
                {
                    AppendClampOmod(text, i);
                    if (op.Format != GcnFormat.Vopc)
                    {
                        text.Append(" dst_sel:").Append(Selections[i.DstSel]).Append(" dst_unused:").Append(Unused[i.DstUnused]);
                    }

                    text.Append(" src0_sel:").Append(Selections[i.Src0Sel]);
                    if (op.Format != GcnFormat.Vop1)
                    {
                        text.Append(" src1_sel:").Append(Selections[i.Src1Sel]);
                    }
                }

                break;
        }

        return text.ToString();
    }

    private static void AppendClampOmod(StringBuilder text, GcnInstruction i)
    {
        if (i.Clamp)
        {
            text.Append(" clamp");
        }

        text.Append(i.Omod switch { 1 => " mul:2", 2 => " mul:4", 3 => " div:2", _ => "" });
    }

    private static void AppendPacked(StringBuilder text, GcnInstruction i)
    {
        int count = i.Opcode.Sources;
        int full = (1 << count) - 1;
        int defaultHi = i.Opcode.Mix ? 0 : full;
        Bits(text, "op_sel", i.OpSel & full, count, i.OpSel != 0);
        Bits(text, "op_sel_hi", i.OpSelHi & full, count, (i.OpSelHi & full) != defaultHi);
        Bits(text, "neg_lo", i.Neg & full, count, i.Neg != 0);
        Bits(text, "neg_hi", i.NegHi & full, count, i.NegHi != 0);
        if (i.Clamp)
        {
            text.Append(" clamp");
        }
    }

    private static void Bits(StringBuilder text, string name, int bits, int count, bool print)
    {
        if (!print)
        {
            return;
        }

        text.Append(' ').Append(name).Append(":[");
        for (int s = 0; s < count; s++)
        {
            text.Append(s == 0 ? "" : ",").Append((bits >> s) & 1);
        }

        text.Append(']');
    }

    private static void AppendDsOffsets(StringBuilder text, GcnInstruction i)
    {
        switch (i.Opcode.OffsetKind)
        {
            case "8x2":
                if (i.Offset0 != 0)
                {
                    text.Append(" offset0:").Append(i.Offset0.ToString(CultureInfo.InvariantCulture));
                }

                if (i.Offset1 != 0)
                {
                    text.Append(" offset1:").Append(i.Offset1.ToString(CultureInfo.InvariantCulture));
                }

                break;
            case "swz":
                if (i.MemoryOffset != 0)
                {
                    text.Append(" offset:").Append(Swizzle(i.MemoryOffset));
                }

                break;
            default:
                if (i.MemoryOffset != 0)
                {
                    text.Append(" offset:").Append(i.MemoryOffset.ToString(CultureInfo.InvariantCulture));
                }

                break;
        }
    }

    /// <summary>A ds_swizzle_b32 offset as LLVM prints it.</summary>
    internal static string Swizzle(int imm)
    {
        if ((imm & 0xFF00) == 0x8000)
        {
            return $"swizzle(QUAD_PERM,{imm & 3},{(imm >> 2) & 3},{(imm >> 4) & 3},{(imm >> 6) & 3})";
        }

        if ((imm & 0x8000) != 0)
        {
            return imm.ToString(CultureInfo.InvariantCulture);
        }

        int and = imm & 0x1F, or = (imm >> 5) & 0x1F, xor = (imm >> 10) & 0x1F;
        if (and == 0x1F && or == 0 && System.Numerics.BitOperations.PopCount((uint)xor) == 1)
        {
            return $"swizzle(SWAP,{xor})";
        }

        if (and == 0x1F && or == 0 && xor > 0 && ((xor + 1) & xor) == 0)
        {
            return $"swizzle(REVERSE,{xor + 1})";
        }

        int group = 0x1F - and + 1;
        if (group > 1 && (group & (group - 1)) == 0 && or < group && xor == 0)
        {
            return $"swizzle(BROADCAST,{group},{or})";
        }

        int probe0 = or ^ xor, probe1 = ((0x1F & and) | or) ^ xor;
        var mask = new StringBuilder("swizzle(BITMASK_PERM,\"");
        for (int bit = 4; bit >= 0; bit--)
        {
            int p0 = (probe0 >> bit) & 1, p1 = (probe1 >> bit) & 1;
            mask.Append(p0 == p1 ? (p0 == 0 ? '0' : '1') : (p0 == 0 ? 'p' : 'i'));
        }

        return mask.Append("\")").ToString();
    }

    private static string DppControl(int c) => c switch
    {
        <= 0xFF => $"quad_perm:[{c & 3},{(c >> 2) & 3},{(c >> 4) & 3},{(c >> 6) & 3}]",
        >= 0x101 and <= 0x10F => $"row_shl:{c & 0xF}",
        >= 0x111 and <= 0x11F => $"row_shr:{c & 0xF}",
        >= 0x121 and <= 0x12F => $"row_ror:{c & 0xF}",
        0x130 => "wave_shl:1",
        0x134 => "wave_rol:1",
        0x138 => "wave_shr:1",
        0x13C => "wave_ror:1",
        0x140 => "row_mirror",
        0x141 => "row_half_mirror",
        0x142 => "row_bcast:15",
        _ => "row_bcast:31",
    };

    private static string Slot(GcnInstruction i, GcnSlot slot)
    {
        var op = i.Opcode;
        switch (slot.Field)
        {
            case GcnField.Fixed:
                if (slot.Text == "off" && i.SAddr >= 0)
                {
                    return Operand(i.SAddr, 2, 's', i, false);
                }

                if (slot.Text == "vcc" && i.Variant == GcnVariant.Sdwa && op.Format == GcnFormat.Vopc)
                {
                    return Operand(i.SD, 2, 's', i, false);
                }

                return slot.Text;
            case GcnField.K:
                return Immediate(i, slot.Text);
            case GcnField.D:
                return Operand(i.D, slot.Width, slot.Class, i, false);
            case GcnField.SD:
                return Operand(i.SD, slot.Width, 's', i, false);
            case GcnField.A:
                return Operand(i.A, op.Format == GcnFormat.Global && i.SAddr >= 0 ? 1 : slot.Width, slot.Class, i, false);
            case GcnField.B:
                return Operand(i.A, slot.Width, 's', i, false);
            default:
                int index = slot.Field - GcnField.S0;
                int value = slot.Field switch { GcnField.S0 => i.S0, GcnField.S1 => i.S1, _ => i.S2 };
                string operand = Operand(value, slot.Width, slot.Class, i, op.Format == GcnFormat.Vop2 && op.Immediate == "lit");
                bool modifiers = op.Format == GcnFormat.Vop3 || i.Variant != GcnVariant.None;
                if (op.Format == GcnFormat.Vop3p || !modifiers || index > 1 && i.Variant != GcnVariant.None)
                {
                    return operand;
                }

                if ((i.Sext >> index & 1) != 0)
                {
                    operand = $"sext({operand})";
                }

                if ((i.Abs >> index & 1) != 0)
                {
                    operand = $"|{operand}|";
                }

                if ((i.Neg >> index & 1) != 0)
                {
                    operand = "-" + operand;
                }

                return operand;
        }
    }

    private static string Immediate(GcnInstruction i, string kind)
    {
        int k = i.Simm16;
        switch (kind)
        {
            case "hex16":
                return "0x" + k.ToString("x", CultureInfo.InvariantCulture);
            case "lit":
                return i.Opcode.Format == GcnFormat.Vop2 ? "0x" + i.Literal.ToString("x", CultureInfo.InvariantCulture) : Literal32(i.Literal);
            case "hwreg":
                return Hwreg(k);
            case "waitcnt":
                return Waitcnt(k);
            case "gpridx":
                int mode = i.Opcode.Format == GcnFormat.Sopc ? i.S1 : k;
                var names = new List<string>();
                foreach (var (bit, label) in (ReadOnlySpan<(int, string)>)[(1, "SRC0"), (2, "SRC1"), (4, "SRC2"), (8, "DST")])
                {
                    if ((mode & bit) != 0)
                    {
                        names.Add(label);
                    }
                }

                return $"gpr_idx({string.Join(",", names)})";
            case "sendmsg":
                return $"sendmsg({k})";
            default:
                string name = i.Opcode.Name;
                if (name == "s_endpgm" && k == 0)
                {
                    return "";
                }

                return k.ToString(CultureInfo.InvariantCulture);
        }
    }

    private static string Hwreg(int k)
    {
        int id = k & 0x3F, offset = (k >> 6) & 0x1F, width = ((k >> 11) & 0x1F) + 1;
        string name = id switch
        {
            1 => "HW_REG_MODE",
            2 => "HW_REG_STATUS",
            3 => "HW_REG_TRAPSTS",
            4 => "HW_REG_HW_ID",
            5 => "HW_REG_GPR_ALLOC",
            6 => "HW_REG_LDS_ALLOC",
            7 => "HW_REG_IB_STS",
            15 => "HW_REG_SH_MEM_BASES",
            _ => id.ToString(CultureInfo.InvariantCulture),
        };
        return offset == 0 && width == 32 ? $"hwreg({name})" : $"hwreg({name}, {offset}, {width})";
    }

    /// <summary>s_waitcnt's counters as LLVM prints them: those below their maximum, or all three when none is.</summary>
    internal static string Waitcnt(int k)
    {
        int vm = (k & 0xF) | ((k >> 14) & 3) << 4, exp = (k >> 4) & 7, lgkm = (k >> 8) & 0xF;
        if (vm == 63 && exp == 7 && lgkm == 15)
        {
            return "vmcnt(63) expcnt(7) lgkmcnt(15)";
        }

        var parts = new List<string>();
        if (vm != 63)
        {
            parts.Add($"vmcnt({vm})");
        }

        if (exp != 7)
        {
            parts.Add($"expcnt({exp})");
        }

        if (lgkm != 15)
        {
            parts.Add($"lgkmcnt({lgkm})");
        }

        return string.Join(" ", parts);
    }

    /// <summary>An operand by its 9-bit source value: a register, a named register, an inline constant or the literal.</summary>
    internal static string Operand(int value, int width, char cls, GcnInstruction i, bool literalHex)
    {
        if (value >= 256)
        {
            int n = value - 256;
            char prefix = cls == 'a' ? 'a' : 'v';
            return width <= 1 ? $"{prefix}{n}" : $"{prefix}[{n}:{n + width - 1}]";
        }

        switch (value)
        {
            case < 102:
                return width <= 1 ? $"s{value}" : $"s[{value}:{value + width - 1}]";
            case 102 or 103:
                return Named("flat_scratch", value - 102, width);
            case 104 or 105:
                return Named("xnack_mask", value - 104, width);
            case 106 or 107:
                return Named("vcc", value - 106, width);
            case >= 108 and <= 123:
                int t = value - 108;
                return width <= 1 ? $"ttmp{t}" : $"ttmp[{t}:{t + width - 1}]";
            case 124:
                return "m0";
            case 126 or 127:
                return Named("exec", value - 126, width);
            case >= 128 and <= 192:
                return (value - 128).ToString(CultureInfo.InvariantCulture);
            case >= 193 and <= 208:
                return (192 - value).ToString(CultureInfo.InvariantCulture);
            case 235:
                return "src_shared_base";
            case 236:
                return "src_shared_limit";
            case 237:
                return "src_private_base";
            case 238:
                return "src_private_limit";
            case 239:
                return "src_pops_exiting_wave_id";
            case >= 240 and <= 248:
                if (value == 248 && width == 2 && i.Opcode.Base.Contains("_f64", StringComparison.Ordinal))
                {
                    return "0.15915494309189532";
                }

                return FloatConstants[value - 240];
            case 251:
                return "src_vccz";
            case 252:
                return "src_execz";
            case 253:
                return "src_scc";
            case 254:
                return "src_lds_direct";
            case 255:
                return literalHex ? "0x" + i.Literal.ToString("x", CultureInfo.InvariantCulture) : Literal32(i.Literal);
            default:
                return $"<reserved {value}>";
        }
    }

    private static readonly string[] FloatConstants = ["0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0", "0.15915494"];

    private static string Named(string name, int half, int width) => width >= 2 && half == 0 ? name : half == 0 ? name + "_lo" : name + "_hi";

    /// <summary>A 32-bit literal as LLVM prints it: an integer from -16 to 64 in decimal, an inline float's bits as that float, else hex.</summary>
    internal static string Literal32(uint value)
    {
        int signed = (int)value;
        if (signed is >= -16 and <= 64)
        {
            return signed.ToString(CultureInfo.InvariantCulture);
        }

        return value switch
        {
            0x3F000000 => "0.5",
            0xBF000000 => "-0.5",
            0x3F800000 => "1.0",
            0xBF800000 => "-1.0",
            0x40000000 => "2.0",
            0xC0000000 => "-2.0",
            0x40800000 => "4.0",
            0xC0800000 => "-4.0",
            0x3E22F983 => "0.15915494",
            _ => "0x" + value.ToString("x", CultureInfo.InvariantCulture),
        };
    }
}
