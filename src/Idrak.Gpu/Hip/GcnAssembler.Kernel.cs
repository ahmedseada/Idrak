// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>
/// A kernel needs more registers than its declared block lets a lane (or a wave) hold: the kernel, the register file
/// ('v' or 's'), how many it needs where it ran out and how many fit. Idrak's assembler does not spill yet.
/// </summary>
internal sealed class GcnAllocationException(string kernel, char file, int needed, int budget, string message) : InvalidOperationException(message)
{
    /// <summary>The kernel's name.</summary>
    public string Kernel { get; } = kernel;

    /// <summary>'v' for vector registers, 's' for scalar ones.</summary>
    public char File { get; } = file;

    /// <summary>The registers live where it ran out, the one being placed included.</summary>
    public int Needed { get; } = needed;

    /// <summary>The registers that fit.</summary>
    public int Budget { get; } = budget;
}

/// <summary>
/// Idrak's GCN assembler, the <c>ptxas</c> part (plan 35, G2a): a kernel written with virtual registers, as PTX is,
/// turned into machine code. Its text is <see cref="Assemble"/>'s syntax, plus:
/// <list type="bullet">
/// <item>Virtual registers: <c>%v12</c> a vector register (a value per lane), <c>%s3</c> a scalar one (a value per wave),
/// and ranges <c>%v[4:7]</c>, <c>%s[0:3]</c> of adjacent ones, numbered as the writer likes (0 to 9999 a file). A range
/// is placed where the hardware takes it (a scalar pair even, a scalar quad or more 4-aligned); a register named both
/// alone and in a range is the range's.</item>
/// <item>Directives, before the first instruction: <c>.kernel name</c> (first); <c>.arg name, bytes, global|value[,
/// offset]</c> for each kernel argument in order (a global buffer pointer is 8 bytes; offsets follow, each aligned to
/// its size up to 8, unless given); <c>.max_flat_workgroup_size n</c> and <c>.reqd_workgroup_size x, y, z</c> (the
/// block the kernel is written for: it bounds the registers, as <c>.maxntid</c> bounds ptxas; without either 256);
/// <c>.lds bytes</c>; and the values the hardware loads before the kernel starts: <c>.user_sgpr dispatch_ptr</c>,
/// <c>queue_ptr</c>, <c>kernarg_segment_ptr</c>, <c>dispatch_id</c> (pairs from s0, in that order), <c>.system_sgpr
/// workgroup_id_x</c>, <c>_y</c>, <c>_z</c> (one each after them; y brings x, z both), <c>.system_vgpr workitem_id_x</c>,
/// <c>_y</c>, <c>_z</c> (v0, v1, v2; x is always loaded).</item>
/// <item>Physical registers only for those system values, read where the directives put them (with the kernel
/// arguments' pointer alone, s[0:1]; then the workgroup id x, s2; the work-item id x, v0), never written: every value
/// the kernel makes is a virtual register. VCC, EXEC, M0 and SCC are named as usual.</item>
/// </list>
/// The assembler then finds the basic blocks, whether EXEC is narrowed at each instruction (a vector register written
/// under a narrowed EXEC keeps its other lanes, so where a value of it reaches the write it is live from before), the
/// registers' liveness, gives each virtual register a physical one within the declared block's budget, inserts the
/// waits the memory counters need and the wait states (s_nop) the family's hazards need, and encodes. <see cref="Printed"/> holds the kernel with physical
/// registers in LLVM's syntax, valid input for <c>llvm-mc</c>. Everything is kept in pooled buffers: once warm, a
/// kernel allocates nothing per instruction (its code, name and arguments only).
/// </summary>
internal sealed partial class GcnAssembler
{
    // An instruction of a kernel's text: what the parser read, kept until the registers are known.
    private struct Item
    {
        public GcnInstruction Ins;
        public GcnVariant Variant;
        public GcnVariant Encoded;
        public int First;
        public int Count;
        public int Neg, Abs, Sext;
        public Modifiers M;
        public int Line;
        public Columns Columns;
    }

    // A line of the emitted code: an instruction of the text, or a wait or wait states inserted before one.
    private struct Row
    {
        public int Item;
        public ushort Value;
        public byte Kind;
    }

    private const byte RowItem = 0, RowWait = 1, RowNop = 2;

    // How an instruction touches one of its operands.
    [Flags]
    private enum Access : byte
    {
        None = 0,
        Read = 1,
        Write = 2,

        // A vector register written lane by lane: lanes EXEC leaves out keep their values.
        LaneWise = 4,
    }

    private bool _kernel;
    private bool _layoutFixed;
    private string? _kernelName;
    private List<GcnArgument>? _arguments;
    private int _argumentEnd;
    private int _maxFlat, _lds;
    private bool _maxGiven;
    private (int X, int Y, int Z)? _required;
    private GcnUserSgprs _userSgprs;
    private int _workgroupIds, _workitemIds;
    private int _systemSgprs;
    private bool _usesVcc;

    private Item[] _items = [];
    private int _itemCount;
    private GcnOperand[] _itemOps = [];
    private int _itemOpCount;
    private int[] _labelOrder = [];
    private int _labelOrderCount;
    private int _sIds, _vIds;
    private Row[] _rows = [];
    private int _rowCount;

    // While an instruction is checked with stand-in registers: the operands as written, for the refusals.
    private bool _standIns;
    private Operands _written;

    private GcnInstruction _waitIns = null!, _nopIns = null!;

    /// <summary>
    /// Assembles one kernel's text (UTF-8; see the class remarks): the code with physical registers, waits and wait
    /// states, and what its descriptor and metadata say. The assembler starts again first (<see cref="Reset"/>); after
    /// it, <see cref="Printed"/> holds the kernel's text with physical registers and <see cref="ToArray"/> its code.
    /// </summary>
    /// <param name="text">The kernel's text.</param>
    /// <param name="wavesPerSimd">The waves of this kernel one SIMD must hold at once (the occupancy aimed for); null: what the declared block needs.</param>
    /// <exception cref="GcnSyntaxException">A line is refused: its line, column and reason.</exception>
    /// <exception cref="GcnAllocationException">The registers do not fit the block's budget (no spilling yet).</exception>
    public GcnKernelCode AssembleKernel(ReadOnlySpan<byte> text, int? wavesPerSimd = null)
    {
        Reset();
        _kernel = true;
        try
        {
            Assemble(text);
            if (_kernelName is null)
            {
                throw new GcnSyntaxException(1, 1, "a kernel's text starts with .kernel <name>");
            }

            if (_itemCount == 0)
            {
                throw new GcnSyntaxException(_line, 1, $"kernel '{_kernelName}' has no instructions");
            }

            _parsing = true;
            BuildBlocks();
            NarrowExec();
            BuildUnits();
            Liveness();
            Allocate(wavesPerSimd);
            PlaceWaitsAndHazards();
            Emit();
            _parsing = false;
            int vgprs = Math.Max(_maxVgpr + 1, _workitemIds);
            int sgprs = Math.Max(_maxSgpr + 1, _systemSgprs);
            return new GcnKernelCode
            {
                Name = _kernelName,
                Code = ToArray(),
                Vgprs = vgprs,
                Sgprs = sgprs,
                UsesVcc = _usesVcc,
                Arguments = _arguments is null ? [] : _arguments,
                LdsBytes = _lds,
                UserSgprs = _userSgprs,
                WorkgroupIds = _workgroupIds,
                WorkitemIds = _workitemIds,
                WaveSize = _target.WaveSizes[0],
                MaxFlatWorkgroupSize = _maxFlat,
                RequiredWorkgroupSize = _required,
            };
        }
        finally
        {
            _kernel = false;
            _parsing = false;
        }
    }

    /// <summary>
    /// A kernel's declarations as <see cref="AssembleKernel"/> reads them (<c>.kernel</c>, each <c>.arg</c> laid out at its
    /// offset, the block and <c>.lds</c>), without its instructions: the lines up to the first instruction or label. The
    /// directives are Idrak's syntax, the same for every family, so what a plug-in's kernel (<see cref="GcnKernel"/>)
    /// checks its launches against is known before any device assembles it. The assembler starts again first.
    /// </summary>
    /// <exception cref="GcnSyntaxException">A directive is refused, or the text does not start with <c>.kernel</c>: its line, column and reason.</exception>
    public GcnKernelDeclarations ReadDeclarations(ReadOnlySpan<byte> text)
    {
        Reset();
        _kernel = true;
        _parsing = true;
        try
        {
            while (true)
            {
                int newline = text.IndexOf((byte)'\n');
                ReadOnlySpan<byte> line = newline < 0 ? text : text[..newline];
                line = line.Length > 0 && line[^1] == (byte)'\r' ? line[..^1] : line;
                _line++;
                line = line[..CommentStart(line)];
                int at = SkipSpaces(line, 0);
                if (at < line.Length)
                {
                    int end = NameEnd(line, at);
                    bool label = end > at && end < line.Length && line[end] == (byte)':' && !IsDigit(line[at]);
                    if (label || line[at] != (byte)'.')
                    {
                        break;                                                      // the first label or instruction: the declarations end
                    }

                    Directive(line, at);
                }

                if (newline < 0)
                {
                    break;
                }

                text = text[(newline + 1)..];
            }

            if (_kernelName is null)
            {
                throw new GcnSyntaxException(1, 1, "a kernel's text starts with .kernel <name>");
            }

            FixLayout();
            return new GcnKernelDeclarations(_kernelName, _arguments is null ? [] : [.. _arguments], _maxFlat, _required, _lds);
        }
        finally
        {
            _kernel = false;
            _parsing = false;
        }
    }

    /// <summary>The first physical register the last kernel gave virtual register <paramref name="id"/> of file <paramref name="file"/> ('s' or 'v').</summary>
    internal int PhysicalRegister(char file, int id)
    {
        int u = file == 's' ? id : _sIds + id;
        if ((uint)id >= (uint)(file == 's' ? _sIds : _vIds) || _unitOf[u] < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(id), id, $"%{file}{id} is not a register of the last kernel.");
        }

        int unit = _unitOf[u];
        return _unitPhys[unit] + (u - _unitLo[unit]);
    }

    private void RentKernelBuffers()
    {
        _items = ArrayPool<Item>.Shared.Rent(64);
        _itemOps = ArrayPool<GcnOperand>.Shared.Rent(192);
        _labelOrder = ArrayPool<int>.Shared.Rent(16);
        _rows = ArrayPool<Row>.Shared.Rent(96);
        _print = ArrayPool<byte>.Shared.Rent(1024);
        RentRegisterBuffers();
        RentWaitBuffers();
        _waitIns = _set.Find(GcnOp.SWaitcnt) ?? throw new NotSupportedException($"The GCN family {_family} has no s_waitcnt.");
        _nopIns = _set.Find(GcnOp.SNop) ?? throw new NotSupportedException($"The GCN family {_family} has no s_nop.");
    }

    private void ReturnKernelBuffers()
    {
        ReturnPooled(ref _items);
        ReturnPooled(ref _itemOps);
        ReturnPooled(ref _labelOrder);
        ReturnPooled(ref _rows);
        ReturnPooled(ref _print);
        ReturnRegisterBuffers();
        ReturnWaitBuffers();
    }

    // Returns a pooled array, cleared when it holds references (so the pool keeps no instructions alive).
    private static void ReturnPooled<T>(ref T[] array)
    {
        if (array.Length > 0)
        {
            ArrayPool<T>.Shared.Return(array, clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());
            array = [];
        }
    }

    private void ResetKernel()
    {
        _kernel = false;
        _layoutFixed = false;
        _kernelName = null;
        _arguments = null;
        _argumentEnd = 0;
        _maxFlat = 256;
        _maxGiven = false;
        _required = null;
        _lds = 0;
        _userSgprs = GcnUserSgprs.None;
        _workgroupIds = 0;
        _workitemIds = 1;
        _systemSgprs = 0;
        _usesVcc = false;
        _itemCount = 0;
        _itemOpCount = 0;
        _labelOrderCount = 0;
        _sIds = 0;
        _vIds = 0;
        _rowCount = 0;
        _standIns = false;
        _blockCount = 0;
        _unitCount = 0;
        _maxSgpr = -1;
        _maxVgpr = -1;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Directives.
    // ---------------------------------------------------------------------------------------------------------------

    private void Directive(ReadOnlySpan<byte> line, int at)
    {
        int end = NameEnd(line, at + 1);
        ReadOnlySpan<byte> name = line[at..end];
        if (_layoutFixed)
        {
            throw Syntax(at, "directives come before the kernel's first instruction");
        }

        bool kernel = name.SequenceEqual(".kernel"u8);
        if (kernel != (_kernelName is null))
        {
            throw Syntax(at, kernel ? "one kernel a text: .kernel is given once" : "a kernel's text starts with .kernel <name>");
        }

        int next = SkipSpaces(line, end);
        if (kernel)
        {
            int nameEnd = NameEnd(line, next);
            if (nameEnd == next || IsDigit(line[next]))
            {
                throw Syntax(next, ".kernel takes the kernel's name (letters, digits, '_', '.', '$')");
            }

            _kernelName = Text(line[next..nameEnd]);
            next = nameEnd;
        }
        else if (name.SequenceEqual(".arg"u8))
        {
            next = ArgumentDirective(line, next);
        }
        else if (name.SequenceEqual(".max_flat_workgroup_size"u8))
        {
            _maxFlat = Integer(line, ref next, 1, 1024);
            _maxGiven = true;
        }
        else if (name.SequenceEqual(".reqd_workgroup_size"u8))
        {
            int x = Integer(line, ref next, 1, 1024);
            next = SkipSpaces(line, Expect(line, SkipSpaces(line, next), (byte)','));
            int y = Integer(line, ref next, 1, 1024);
            next = SkipSpaces(line, Expect(line, SkipSpaces(line, next), (byte)','));
            int z = Integer(line, ref next, 1, 64);
            if (x * y * z > 1024)
            {
                throw Syntax(at, $"a workgroup of {x} x {y} x {z} is past 1024 work-items");
            }

            _required = (x, y, z);
        }
        else if (name.SequenceEqual(".lds"u8))
        {
            _lds = Integer(line, ref next, 0, 65536);
        }
        else if (name.SequenceEqual(".user_sgpr"u8))
        {
            int valueEnd = NameEnd(line, next);
            ReadOnlySpan<byte> value = line[next..valueEnd];
            var flag = value.SequenceEqual("dispatch_ptr"u8) ? GcnUserSgprs.DispatchPtr
                : value.SequenceEqual("queue_ptr"u8) ? GcnUserSgprs.QueuePtr
                : value.SequenceEqual("kernarg_segment_ptr"u8) ? GcnUserSgprs.KernargSegmentPtr
                : value.SequenceEqual("dispatch_id"u8) ? GcnUserSgprs.DispatchId
                : throw Syntax(next, $"'{Text(value)}' is not a user SGPR Idrak loads (dispatch_ptr, queue_ptr, kernarg_segment_ptr, dispatch_id)");
            if ((_userSgprs & flag) != 0)
            {
                throw Syntax(next, $"{Text(value)} is enabled twice");
            }

            _userSgprs |= flag;
            next = valueEnd;
        }
        else if (name.SequenceEqual(".system_sgpr"u8) || name.SequenceEqual(".system_vgpr"u8))
        {
            bool vector = name[8] == (byte)'v';
            int valueEnd = NameEnd(line, next);
            ReadOnlySpan<byte> value = line[next..valueEnd];
            ReadOnlySpan<byte> prefix = vector ? "workitem_id_"u8 : "workgroup_id_"u8;
            int axis = value.Length == prefix.Length + 1 && value.StartsWith(prefix) ? value[^1] - 'x' : -1;
            if (axis is < 0 or > 2)
            {
                throw Syntax(next, vector ? "the system VGPRs are workitem_id_x, workitem_id_y and workitem_id_z" : "the system SGPRs are workgroup_id_x, workgroup_id_y and workgroup_id_z");
            }

            if (vector)
            {
                _workitemIds = Math.Max(_workitemIds, axis + 1);
            }
            else
            {
                _workgroupIds = Math.Max(_workgroupIds, axis + 1);
            }

            next = valueEnd;
        }
        else
        {
            throw Syntax(at, $"'{Text(name)}' is not a directive Idrak knows (.kernel, .arg, .max_flat_workgroup_size, .reqd_workgroup_size, .lds, .user_sgpr, .system_sgpr, .system_vgpr)");
        }

        next = SkipSpaces(line, next);
        if (next < line.Length)
        {
            throw Syntax(next, $"unexpected '{Text(line[next..])}' after {Text(name)}");
        }
    }

    // .arg name, bytes, global|value[, offset]
    private int ArgumentDirective(ReadOnlySpan<byte> line, int at)
    {
        int nameEnd = NameEnd(line, at);
        if (nameEnd == at)
        {
            throw Syntax(at, ".arg takes a name, its size in bytes and global or value (and an offset)");
        }

        string argument = Text(line[at..nameEnd]);
        int next = SkipSpaces(line, Expect(line, SkipSpaces(line, nameEnd), (byte)','));
        int sizeAt = next;
        int size = Integer(line, ref next, 1, 256);
        next = SkipSpaces(line, Expect(line, SkipSpaces(line, next), (byte)','));
        int kindEnd = NameEnd(line, next);
        ReadOnlySpan<byte> kind = line[next..kindEnd];
        bool global = kind.SequenceEqual("global"u8);
        if (!global && !kind.SequenceEqual("value"u8))
        {
            throw Syntax(next, $"'{Text(kind)}' is not an argument kind: global (a pointer to global memory) or value");
        }

        if (global && size != 8)
        {
            throw Syntax(sizeAt, "a global buffer pointer is 8 bytes");
        }

        int align = size >= 8 ? 8 : size >= 4 ? 4 : size >= 2 ? 2 : 1;
        int offset = (_argumentEnd + align - 1) / align * align;
        next = SkipSpaces(line, kindEnd);
        if (next < line.Length && line[next] == (byte)',')
        {
            next = SkipSpaces(line, next + 1);
            int offsetAt = next;
            offset = Integer(line, ref next, 0, 1 << 20);
            if (offset < _argumentEnd)
            {
                throw Syntax(offsetAt, $"argument '{argument}' at {offset} overlaps the one before it (which ends at {_argumentEnd})");
            }
        }

        _arguments ??= [];
        foreach (var other in _arguments)
        {
            if (other.Name == argument)
            {
                throw Syntax(at, $"argument '{argument}' is declared twice");
            }
        }

        _arguments.Add(global ? GcnArgument.Buffer(offset, argument) : GcnArgument.Value(offset, size, argument));
        _argumentEnd = offset + size;
        return next;
    }

    // The block and the system registers, fixed by the directives once the first instruction comes.
    private void FixLayout()
    {
        if (_required is { } required)
        {
            int product = required.X * required.Y * required.Z;
            if (_maxGiven && _maxFlat < product)
            {
                throw Syntax(0, $".max_flat_workgroup_size {_maxFlat} is less than the required workgroup's {product} work-items");
            }

            if (!_maxGiven)
            {
                _maxFlat = product;
            }
        }

        _systemSgprs = 2 * BitOperations.PopCount((uint)_userSgprs) + _workgroupIds;
        _layoutFixed = true;
    }

    // What the system registers hold, for a refusal.
    private string SystemLayout()
    {
        var text = new StringBuilder();
        int at = 0;
        foreach (var (flag, what) in (ReadOnlySpan<(GcnUserSgprs, string)>)
                 [(GcnUserSgprs.DispatchPtr, "dispatch_ptr"), (GcnUserSgprs.QueuePtr, "queue_ptr"), (GcnUserSgprs.KernargSegmentPtr, "kernarg_segment_ptr"), (GcnUserSgprs.DispatchId, "dispatch_id")])
        {
            if ((_userSgprs & flag) != 0)
            {
                text.Append(System.Globalization.CultureInfo.InvariantCulture, $"s[{at}:{at + 1}] {what}, ");
                at += 2;
            }
        }

        for (int axis = 0; axis < _workgroupIds; axis++)
        {
            text.Append(System.Globalization.CultureInfo.InvariantCulture, $"s{at++} workgroup_id_{(char)('x' + axis)}, ");
        }

        for (int axis = 0; axis < _workitemIds; axis++)
        {
            text.Append(System.Globalization.CultureInfo.InvariantCulture, $"v{axis} workitem_id_{(char)('x' + axis)}, ");
        }

        return text.ToString(0, text.Length - 2);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Instructions of a kernel's text: checked, then kept.
    // ---------------------------------------------------------------------------------------------------------------

    // %s3, %v12, %s[4:7], %v[0:3]
    private int ParseVirtual(ReadOnlySpan<byte> line, int at)
    {
        int start = at++;
        if (at >= line.Length || line[at] is not ((byte)'s' or (byte)'v'))
        {
            throw Syntax(start, "a virtual register is %v or %s and a number, or a range (%v12, %s[4:7])");
        }

        var kind = line[at] == (byte)'s' ? GcnOperandKind.VirtualSgpr : GcnOperandKind.VirtualVgpr;
        at++;
        int low, count;
        if (at < line.Length && line[at] == (byte)'[')
        {
            int close = line[at..].IndexOf((byte)']');
            ReadOnlySpan<byte> range = close < 0 ? [] : line[(at + 1)..(at + close)];
            int colon = range.IndexOf((byte)':');
            if (close < 0 || colon < 0 || !TryIndex(range[..colon], out low) || !TryIndex(range[(colon + 1)..], out int high) || high < low || high - low >= 16)
            {
                throw Syntax(start, $"'{Text(line[start..(close < 0 ? line.Length : at + close + 1)])}' is not a range of 1 to 16 virtual registers (%v[low:high])");
            }

            count = high - low + 1;
            at += close + 1;
        }
        else
        {
            int end = NameEnd(line, at);
            if (!TryIndex(line[at..end], out low))
            {
                throw Syntax(start, $"'{Text(line[start..end])}' is not a virtual register (%v or %s and 0 to 9999)");
            }

            count = 1;
            at = end;
        }

        Add(GcnOperand.Virtual(kind, low, count));
        return at;
    }

    // The instruction parsed: checked against what a kernel's text allows, encoded once with stand-in registers (so
    // every refusal comes now, with its line and column), and kept.
    private void Record()
    {
        if (!_layoutFixed)
        {
            FixLayout();
        }

        var ins = _ins;
        if (ins.Op is GcnOp.SMovrelsB32 or GcnOp.SMovreldB32 or GcnOp.SSetGprIdxOn or GcnOp.SSetGprIdxOff or GcnOp.SSetpcB64 or GcnOp.SSwappcB64)
        {
            throw Syntax(_column - 1, $"{ins.Mnemonic} is not taken in a kernel's text: indexing registers by M0 and jumping through registers need physical registers and a known flow");
        }

        if (IsBranch(ins.Op) && _ops[0].Kind != GcnOperandKind.Label)
        {
            throw Syntax(_columns[0] - 1, "a branch in a kernel's text names a label (the allocator follows the flow through it)");
        }

        for (int i = 0; i < _count; i++)
        {
            CheckKernelOperand(i);
        }

        Validate();
        Grow(ref _items, _itemCount + 1);
        Grow(ref _itemOps, _itemOpCount + _count);
        ref var item = ref _items[_itemCount++];
        item.Ins = ins;
        item.Variant = _variant;
        item.Encoded = _encoded;
        item.First = _itemOpCount;
        item.Count = _count;
        item.Neg = _negMask;
        item.Abs = _absMask;
        item.Sext = _sextMask;
        item.M = _m;
        item.Line = _line;
        item.Columns = _columns;
        for (int i = 0; i < _count; i++)
        {
            var operand = _ops[i];
            _itemOps[_itemOpCount++] = operand;
            if (operand.Kind == GcnOperandKind.VirtualSgpr)
            {
                _sIds = Math.Max(_sIds, operand.Index + operand.Count);
            }
            else if (operand.Kind == GcnOperandKind.VirtualVgpr)
            {
                _vIds = Math.Max(_vIds, operand.Index + operand.Count);
            }
            else if (operand.Kind == GcnOperandKind.Special && (GcnSpecial)operand.Index is GcnSpecial.Vcc or GcnSpecial.VccLo or GcnSpecial.VccHi)
            {
                _usesVcc = true;
            }
        }

        _usesVcc |= ReadsVccImplicitly(ins);
    }

    // A physical register in a kernel's text: only a system value the directives enable, and only read.
    private void CheckKernelOperand(int i)
    {
        var operand = _ops[i];
        int at = _columns[i] - 1;
        switch (operand.Kind)
        {
            case GcnOperandKind.Agpr or GcnOperandKind.Ttmp:
                throw Syntax(at, $"{operand} is not allocated: a kernel's text uses virtual registers (%v, %s) and the system values its directives enable");
            case GcnOperandKind.Special when (GcnSpecial)operand.Index is GcnSpecial.FlatScratch or GcnSpecial.FlatScratchLo or GcnSpecial.FlatScratchHi
                or GcnSpecial.XnackMask or GcnSpecial.XnackMaskLo or GcnSpecial.XnackMaskHi:
                throw Syntax(at, $"{operand} is reserved by the target in a kernel's text");
            case GcnOperandKind.Sgpr or GcnOperandKind.Vgpr:
                {
                    bool scalar = operand.Kind == GcnOperandKind.Sgpr;
                    int limit = scalar ? _systemSgprs : _workitemIds;
                    if (operand.Index + operand.Count > limit)
                    {
                        throw Syntax(at, $"{operand} holds no system value this kernel enables ({SystemLayout()}): every other register is a virtual one (%{(scalar ? 's' : 'v')})");
                    }

                    if ((AccessOf(_ins, i, _variant, _m) & Access.Write) != 0)
                    {
                        throw Syntax(at, $"{operand} holds a system value, which a kernel's text reads only: write a virtual register (%{(scalar ? 's' : 'v')}) instead");
                    }

                    break;
                }
        }
    }

    // Encodes the instruction with each virtual register replaced by a physical stand-in of its size and alignment (the
    // n-th distinct one at s[16n] or v[32n]), so the encoder's checks (widths, alignment, one scalar value, literals) are
    // made now; the code written is taken back.
    private void Validate()
    {
        int length = _length, branches = _branchCount;
        int scalars = 0, vectors = 0;
        for (int i = 0; i < _count; i++)
        {
            var operand = _ops[i];
            _written[i] = operand;
            if (operand.Kind is not (GcnOperandKind.VirtualSgpr or GcnOperandKind.VirtualVgpr))
            {
                continue;
            }

            _standIns = true;
            int at = -1;
            for (int j = 0; j < i; j++)
            {
                if (_written[j].Kind == operand.Kind && _written[j].Index == operand.Index)
                {
                    at = _ops[j].Index;
                    break;
                }
            }

            bool scalar = operand.Kind == GcnOperandKind.VirtualSgpr;
            if (at < 0)
            {
                at = scalar ? 16 * scalars++ : 32 * vectors++;
            }

            _ops[i] = scalar ? GcnOperand.S(at, operand.Count) : GcnOperand.V(at, operand.Count);
        }

        try
        {
            Encode();
        }
        finally
        {
            _standIns = false;
            _length = length;
            _branchCount = branches;
            for (int i = 0; i < _count; i++)
            {
                _ops[i] = _written[i];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // What an instruction touches: its operands' access, its implicit registers, its memory class.
    // ---------------------------------------------------------------------------------------------------------------

    private static Access AccessOf(GcnInstruction ins, int i, GcnVariant encoded, in Modifiers m)
    {
        switch (ins.Operands.Role(i))
        {
            case GcnRole.ScalarDestination:
                return ins.Op is GcnOp.SCmovB32 or GcnOp.SCmovB64 or GcnOp.SCmovkI32 or GcnOp.SAddkI32 or GcnOp.SMulkI32 or GcnOp.SBitset0B32 or GcnOp.SBitset1B32
                    ? Access.Read | Access.Write
                    : Access.Write;
            case GcnRole.ScalarResult or GcnRole.LaneMask:
                return Access.Write;
            case GcnRole.VectorDestination or GcnRole.AccumulationDestination:
                return KeepsDestination(ins, encoded, m) ? Access.Read | Access.Write | Access.LaneWise : Access.Write | Access.LaneWise;
            case GcnRole.VectorSource when ins.Op == GcnOp.VSwapB32:
                return Access.Read | Access.Write | Access.LaneWise;
            case GcnRole.None or GcnRole.Immediate16 or GcnRole.HardwareRegister or GcnRole.Literal32 or GcnRole.GprIndex or GcnRole.WaitCounts or GcnRole.Target or GcnRole.Constant32:
                return Access.None;
            default:
                return Access.Read;
        }
    }

    // Instructions whose vector destination keeps part of what it held whatever EXEC is: accumulating into it, writing
    // one lane or one half (the 16-bit loads, the mixed-precision halves, gfx9's 16-bit multiply-adds), SDWA keeping the
    // unused bits, DPP with lanes masked or sources out of bounds, a VOP3 op_sel.
    private static bool KeepsDestination(GcnInstruction ins, GcnVariant encoded, in Modifiers m) =>
        ins.Op is GcnOp.VMacF32 or GcnOp.VMacF16 or GcnOp.VFmacF32 or GcnOp.VDot2cF32F16 or GcnOp.VDot4cI32I8 or GcnOp.VWritelaneB32 or GcnOp.VMadMixloF16
            or GcnOp.VFmaMixloF16 or GcnOp.VFmaMixhiF16 or GcnOp.VMadF16 or GcnOp.VFmaF16 or GcnOp.VDivFixupF16 or GcnOp.VMadU16 or GcnOp.DsReadU16D16
            or GcnOp.DsReadU16D16Hi or GcnOp.GlobalLoadShortD16 or GcnOp.GlobalLoadShortD16Hi or GcnOp.GlobalLoadUbyteD16 or GcnOp.GlobalLoadUbyteD16Hi
        || (encoded == GcnVariant.Sdwa && m.DstSel != 6 && m.DstUnused == 2)
        || (encoded == GcnVariant.Dpp && (m.RowMask != 0xF || m.BankMask != 0xF || !m.BoundCtrl))
        || m.OpSelCount > 0;

    private static bool IsSaveexec(GcnOp op) => op is GcnOp.SAndSaveexecB64 or GcnOp.SOrSaveexecB64 or GcnOp.SXorSaveexecB64 or GcnOp.SAndn2SaveexecB64
        or GcnOp.SOrn2SaveexecB64 or GcnOp.SAndn1SaveexecB64;

    private static bool IsVector(GcnFormat format) => format is GcnFormat.Vop1 or GcnFormat.Vop2 or GcnFormat.Vopc or GcnFormat.Vop3 or GcnFormat.Vop3p;

    private static bool IsScalarAlu(GcnFormat format) => format is GcnFormat.Sop1 or GcnFormat.Sop2 or GcnFormat.Sopk or GcnFormat.Sopc;

    private static bool ReadsExec(GcnInstruction ins) => IsVector(ins.Format) || ins.Format is GcnFormat.Ds or GcnFormat.Flat or GcnFormat.Global
        || IsSaveexec(ins.Op) || ins.Op is GcnOp.SCbranchExecz or GcnOp.SCbranchExecnz;

    private static bool WritesExec(GcnInstruction ins) => IsSaveexec(ins.Op) || ins.Op is GcnOp.VCmpxLtU32 or GcnOp.VCmpxGtU32 or GcnOp.VCmpxEqU32 or GcnOp.VCmpxLtF32;

    private static bool ReadsVccImplicitly(GcnInstruction ins) => ins.Has(GcnInstructionFlags.ReadsVcc) || ins.Op is GcnOp.SCbranchVccz or GcnOp.SCbranchVccnz;

    private static bool ReadsM0(GcnInstruction ins, in Modifiers m) => (ins.Format == GcnFormat.Ds && m.Gds) || ins.Op is GcnOp.DsAppend or GcnOp.DsConsume;

    private static GcnMemoryClass MemoryOf(GcnInstruction ins, in Modifiers m) => ins.Format switch
    {
        GcnFormat.Smem => GcnMemoryClass.Scalar,
        GcnFormat.Ds => m.Gds ? GcnMemoryClass.Gds : GcnMemoryClass.Lds,
        GcnFormat.Global => GcnMemoryClass.Vector,
        GcnFormat.Flat => GcnMemoryClass.Flat,
        _ => GcnMemoryClass.None,
    };

    // A memory instruction that writes memory: stores, atomics, SMEM's stores and write-back.
    private static bool IsStore(GcnInstruction ins) =>
        (ins.Format is GcnFormat.Global or GcnFormat.Flat && (ins.Operands.IndexOf(GcnRole.VectorDestination) < 0 || ins.Has(GcnInstructionFlags.Atomic)))
        || ins.Op is GcnOp.SStoreDword or GcnOp.SStoreDwordx2 or GcnOp.SDcacheWb;

    private static bool IsBranch(GcnOp op) => op is GcnOp.SBranch or GcnOp.SCbranchScc0 or GcnOp.SCbranchScc1 or GcnOp.SCbranchVccz or GcnOp.SCbranchVccnz
        or GcnOp.SCbranchExecz or GcnOp.SCbranchExecnz;

    private static bool IsAllOnes(GcnOperand operand) => operand.Kind == GcnOperandKind.Constant && operand.Value == uint.MaxValue;

    // ---------------------------------------------------------------------------------------------------------------
    // The code: each instruction with its physical registers, after the waits and wait states placed before it.
    // ---------------------------------------------------------------------------------------------------------------

    private GcnOperand Physical(GcnOperand operand)
    {
        switch (operand.Kind)
        {
            case GcnOperandKind.VirtualSgpr:
                {
                    int unit = _unitOf[operand.Index];
                    return GcnOperand.S(_unitPhys[unit] + operand.Index - _unitLo[unit], operand.Count);
                }

            case GcnOperandKind.VirtualVgpr:
                {
                    int u = _sIds + operand.Index;
                    int unit = _unitOf[u];
                    return GcnOperand.V(_unitPhys[unit] + u - _unitLo[unit], operand.Count);
                }

            default:
                return operand;
        }
    }

    private void AddRow(int item, byte kind, ushort value)
    {
        Grow(ref _rows, _rowCount + 1);
        _rows[_rowCount++] = new Row { Item = item, Kind = kind, Value = value };
    }

    // Encodes the rows into the code (labels bound as their instruction comes) and prints them.
    private void Emit()
    {
        _length = 0;
        _branchCount = 0;
        _printLength = 0;
        int label = 0, current = -1;
        for (int r = 0; r < _rowCount; r++)
        {
            var row = _rows[r];
            if (row.Item != current)
            {
                current = row.Item;
                while (label < _labelOrderCount && _labels[LabelSlot * _labelOrder[label]] == current)
                {
                    int id = _labelOrder[label++];
                    _labels[LabelSlot * id] = _length;
                    PrintLabel(id);
                }
            }

            switch (row.Kind)
            {
                case RowWait:
                    Begin(_waitIns, GcnVariant.Auto, typed: false);
                    Add(GcnOperand.Immediate(row.Value));
                    _line = _items[row.Item].Line;
                    break;
                case RowNop:
                    Begin(_nopIns, GcnVariant.Auto, typed: false);
                    Add(GcnOperand.Immediate(row.Value));
                    _line = _items[row.Item].Line;
                    break;
                default:
                    {
                        ref var item = ref _items[row.Item];
                        Begin(item.Ins, item.Variant, typed: false);
                        _negMask = item.Neg;
                        _absMask = item.Abs;
                        _sextMask = item.Sext;
                        _m = item.M;
                        _line = item.Line;
                        _columns = item.Columns;
                        for (int i = 0; i < item.Count; i++)
                        {
                            Add(Physical(_itemOps[item.First + i]));
                        }

                        break;
                    }
            }

            _column = 1;
            Encode();
            PrintInstruction();
        }
    }
}
