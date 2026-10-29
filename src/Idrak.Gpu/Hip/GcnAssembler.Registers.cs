// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

// The register allocator of a kernel's text: its basic blocks, where EXEC is narrowed, the virtual registers' liveness
// over the blocks, their units (a register, or a range kept adjacent and aligned), and a linear scan that gives each
// unit physical registers within the budget the declared block leaves. Every table lives in pooled arrays sized by
// the kernel and kept by the assembler, so a warm assembler allocates nothing here.
internal sealed partial class GcnAssembler
{
    private const byte Narrowed = 1, StartsBlock = 2;
    private const byte Mentioned = 1, Written = 2;

    // Registers as the waits and hazards index them: scalar operand codes 0-127, then VGPR n at 128 + n.
    private const int RegisterSpace = 384, VgprBase = 128;

    private int _blockCount;
    private int[] _blockStart = [], _blockEnd = [], _succ = [], _itemBlock = [];
    private byte[] _itemFlags = [], _execIn = [], _saved = [], _keeps = [];
    private ulong[] _live = [], _running = [];
    private int _words;

    // Per virtual register (scalar ids first, then vector ones): its group's parent, its unit, where it is first and
    // last live, whether it is mentioned and written.
    private int[] _parent = [], _unitOf = [], _idFirst = [], _idLast = [];
    private byte[] _idFlags = [];

    // Per unit: its first register (as a virtual id), its size, file (0 scalar, 1 vector), the residues mod 4 its first
    // physical register may have, where it is live, the block of its load (or -1), and its first physical register.
    private int _unitCount;
    private int[] _unitLo = [], _unitSize = [], _unitStart = [], _unitEnd = [], _unitLoad = [], _unitPhys = [];
    private byte[] _unitFile = [], _unitMask = [];
    private long[] _order = [];

    // Per physical register: the position its value is last used at, and the block whose load last wrote it.
    private readonly int[] _busy = new int[RegisterSpace], _loadOwner = new int[RegisterSpace];
    private int _maxSgpr = -1, _maxVgpr = -1;

    private void RentRegisterBuffers()
    {
        _blockStart = ArrayPool<int>.Shared.Rent(16);
        _blockEnd = ArrayPool<int>.Shared.Rent(16);
        _succ = ArrayPool<int>.Shared.Rent(32);
        _itemBlock = ArrayPool<int>.Shared.Rent(64);
        _itemFlags = ArrayPool<byte>.Shared.Rent(64);
        _keeps = ArrayPool<byte>.Shared.Rent(64);
        _running = ArrayPool<ulong>.Shared.Rent(4);
        _execIn = ArrayPool<byte>.Shared.Rent(16);
        _saved = ArrayPool<byte>.Shared.Rent(64);
        _live = ArrayPool<ulong>.Shared.Rent(64);
        _parent = ArrayPool<int>.Shared.Rent(64);
        _unitOf = ArrayPool<int>.Shared.Rent(64);
        _idFirst = ArrayPool<int>.Shared.Rent(64);
        _idLast = ArrayPool<int>.Shared.Rent(64);
        _idFlags = ArrayPool<byte>.Shared.Rent(64);
        _unitLo = ArrayPool<int>.Shared.Rent(64);
        _unitSize = ArrayPool<int>.Shared.Rent(64);
        _unitStart = ArrayPool<int>.Shared.Rent(64);
        _unitEnd = ArrayPool<int>.Shared.Rent(64);
        _unitLoad = ArrayPool<int>.Shared.Rent(64);
        _unitPhys = ArrayPool<int>.Shared.Rent(64);
        _unitFile = ArrayPool<byte>.Shared.Rent(64);
        _unitMask = ArrayPool<byte>.Shared.Rent(64);
        _order = ArrayPool<long>.Shared.Rent(64);
    }

    private void ReturnRegisterBuffers()
    {
        ReturnPooled(ref _blockStart);
        ReturnPooled(ref _blockEnd);
        ReturnPooled(ref _succ);
        ReturnPooled(ref _itemBlock);
        ReturnPooled(ref _itemFlags);
        ReturnPooled(ref _keeps);
        ReturnPooled(ref _running);
        ReturnPooled(ref _execIn);
        ReturnPooled(ref _saved);
        ReturnPooled(ref _live);
        ReturnPooled(ref _parent);
        ReturnPooled(ref _unitOf);
        ReturnPooled(ref _idFirst);
        ReturnPooled(ref _idLast);
        ReturnPooled(ref _idFlags);
        ReturnPooled(ref _unitLo);
        ReturnPooled(ref _unitSize);
        ReturnPooled(ref _unitStart);
        ReturnPooled(ref _unitEnd);
        ReturnPooled(ref _unitLoad);
        ReturnPooled(ref _unitPhys);
        ReturnPooled(ref _unitFile);
        ReturnPooled(ref _unitMask);
        ReturnPooled(ref _order);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Basic blocks: an instruction starts one when it is the first, a label is placed at it, or the one before it is a
    // branch or s_endpgm. s_branch goes to its label; s_cbranch_* to its label or on; s_endpgm nowhere.
    // ---------------------------------------------------------------------------------------------------------------

    private void BuildBlocks()
    {
        int n = _itemCount;
        Grow(ref _itemFlags, n);
        Grow(ref _itemBlock, n);
        _itemFlags.AsSpan(0, n).Clear();
        _itemFlags[0] = StartsBlock;
        for (int l = 0; l < _labelOrderCount; l++)
        {
            int id = _labelOrder[l], at = _labels[LabelSlot * id];
            if (at >= n)
            {
                throw new GcnSyntaxException(_items[n - 1].Line, 1, $"label '{LabelName(id)}' ends the kernel: an instruction follows every label");
            }

            _itemFlags[at] |= StartsBlock;
        }

        for (int k = 0; k < n; k++)
        {
            ref var item = ref _items[k];
            var op = item.Ins.Op;
            if (IsBranch(op))
            {
                int label = _itemOps[item.First].Index;
                if (_labels[LabelSlot * label] < 0)
                {
                    throw new GcnSyntaxException(item.Line, item.Columns[0], $"the branch names label '{LabelName(label)}', which is never defined");
                }
            }

            if ((IsBranch(op) || op == GcnOp.SEndpgm) && k + 1 < n)
            {
                _itemFlags[k + 1] |= StartsBlock;
            }
        }

        if (_items[n - 1].Ins.Op is not (GcnOp.SEndpgm or GcnOp.SBranch))
        {
            throw new GcnSyntaxException(_items[n - 1].Line, 1, $"kernel '{_kernelName}' runs past its last instruction: end it with s_endpgm or s_branch");
        }

        int blocks = 0;
        for (int k = 0; k < n; k++)
        {
            blocks += _itemFlags[k] >> 1 & 1;
            _itemBlock[k] = blocks - 1;
        }

        _blockCount = blocks;
        Grow(ref _blockStart, blocks);
        Grow(ref _blockEnd, blocks);
        Grow(ref _succ, 2 * blocks);
        for (int k = 0; k < n; k++)
        {
            if ((_itemFlags[k] & StartsBlock) != 0)
            {
                _blockStart[_itemBlock[k]] = k;
            }

            _blockEnd[_itemBlock[k]] = k + 1;
        }

        for (int b = 0; b < blocks; b++)
        {
            ref var last = ref _items[_blockEnd[b] - 1];
            var op = last.Ins.Op;
            int target = IsBranch(op) ? _itemBlock[_labels[LabelSlot * _itemOps[last.First].Index]] : -1;
            (_succ[2 * b], _succ[2 * b + 1]) = op switch
            {
                GcnOp.SEndpgm => (-1, -1),
                GcnOp.SBranch => (target, -1),
                _ when target >= 0 => (target, b + 1 < blocks ? b + 1 : -1),
                _ => (b + 1 < blocks ? b + 1 : -1, -1),
            };
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // EXEC: full (the wave's lanes, as the kernel starts) or narrowed, before each instruction. Saving EXEC and narrowing
    // it (s_*_saveexec) narrows it, and the pair it saved holds the state before; restoring from that pair (s_mov_b64 exec,
    // s_or_b64 exec with exec) gives that state back; s_mov_b64 exec, -1 and s_or_saveexec_b64 with -1 make it full.
    // Any other write of EXEC (v_cmpx too), a pair saved under different states or written otherwise, an instruction no
    // path reaches: narrowed, the safe answer.
    // ---------------------------------------------------------------------------------------------------------------

    private void NarrowExec()
    {
        int blocks = _blockCount;
        Grow(ref _execIn, blocks);
        Grow(ref _saved, Math.Max(1, _sIds));
        _saved.AsSpan(0, _sIds).Clear();

        // A scalar pair written by anything but a saveexec is not a saved EXEC: 2 (narrowed, unknown).
        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            for (int i = 0; i < item.Count; i++)
            {
                var operand = _itemOps[item.First + i];
                if (operand.Kind == GcnOperandKind.VirtualSgpr && (AccessOf(item.Ins, i, item.Encoded, item.M) & Access.Write) != 0
                    && !(IsSaveexec(item.Ins.Op) && i == 0 && operand.Count == 2))
                {
                    _saved.AsSpan(operand.Index, operand.Count).Fill(2);
                }
            }
        }

        bool savedChanged = true;
        while (savedChanged)
        {
            savedChanged = false;
            _execIn.AsSpan(0, blocks).Clear();
            _execIn[0] = 1;
            bool moved = true;
            while (moved)
            {
                moved = false;
                for (int b = 0; b < blocks; b++)
                {
                    int state = _execIn[b];
                    if (state == 0)
                    {
                        continue;
                    }

                    for (int k = _blockStart[b]; k < _blockEnd[b]; k++)
                    {
                        _itemFlags[k] = (byte)((_itemFlags[k] & ~Narrowed) | (state == 2 ? Narrowed : 0));
                        state = ExecAfter(k, state, ref savedChanged);
                    }

                    for (int s = 2 * b; s < 2 * b + 2; s++)
                    {
                        int next = _succ[s];
                        if (next >= 0 && _execIn[next] < state)
                        {
                            _execIn[next] = (byte)state;
                            moved = true;
                        }
                    }
                }
            }
        }

        for (int b = 0; b < blocks; b++)
        {
            if (_execIn[b] == 0)
            {
                for (int k = _blockStart[b]; k < _blockEnd[b]; k++)
                {
                    _itemFlags[k] |= Narrowed;
                }
            }
        }
    }

    // EXEC's state after item k (1 full, 2 narrowed), from the state before it.
    private int ExecAfter(int k, int state, ref bool savedChanged)
    {
        ref var item = ref _items[k];
        var ins = item.Ins;
        if (IsSaveexec(ins.Op))
        {
            var saved = _itemOps[item.First];
            if (saved.Kind == GcnOperandKind.VirtualSgpr && saved.Count == 2 && _saved[saved.Index] < state)
            {
                _saved[saved.Index] = (byte)state;
                savedChanged = true;
            }

            return ins.Op == GcnOp.SOrSaveexecB64 && IsAllOnes(_itemOps[item.First + 1]) ? 1 : 2;
        }

        if (WritesExec(ins))
        {
            return 2;
        }

        int written = -1;
        for (int i = 0; i < item.Count; i++)
        {
            var operand = _itemOps[item.First + i];
            if (operand.Kind == GcnOperandKind.Special && (GcnSpecial)operand.Index is GcnSpecial.Exec or GcnSpecial.ExecLo or GcnSpecial.ExecHi
                && (AccessOf(ins, i, item.Encoded, item.M) & Access.Write) != 0)
            {
                written = i;
            }
        }

        if (written < 0)
        {
            return state;
        }

        bool whole = _itemOps[item.First + written] is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Exec };
        if (whole && ins.Op == GcnOp.SMovB64)
        {
            return SavedState(_itemOps[item.First + 1]);
        }

        if (whole && ins.Op == GcnOp.SOrB64)
        {
            var a = _itemOps[item.First + 1];
            var b = _itemOps[item.First + 2];
            var other = a is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Exec } ? b : b is { Kind: GcnOperandKind.Special, Index: (int)GcnSpecial.Exec } ? a : default;
            return state == 1 && other.Kind != GcnOperandKind.None ? 1 : other.Kind == GcnOperandKind.None ? 2 : SavedState(other);
        }

        return 2;
    }

    // EXEC's state restored from a source: -1 is full, a pair a saveexec wrote is the state it saved.
    private int SavedState(GcnOperand source) =>
        IsAllOnes(source) ? 1
        : source.Kind == GcnOperandKind.VirtualSgpr && source.Count == 2 && _saved[source.Index] == 1 ? 1
        : 2;

    // ---------------------------------------------------------------------------------------------------------------
    // Units: a virtual register alone, or the registers ranges join (%v[4:7] with %v[6:9]: 4 to 9), placed together,
    // a scalar range at a multiple of its size up to 4 (ISA 9's pairs and quads; vector ranges anywhere).
    // ---------------------------------------------------------------------------------------------------------------

    private void BuildUnits()
    {
        int ids = _sIds + _vIds;
        Grow(ref _parent, Math.Max(1, ids));
        Grow(ref _unitOf, Math.Max(1, ids));
        Grow(ref _idFlags, Math.Max(1, ids));
        for (int u = 0; u < ids; u++)
        {
            _parent[u] = u;
            _unitOf[u] = -1;
            _idFlags[u] = 0;
        }

        for (int o = 0; o < _itemOpCount; o++)
        {
            var operand = _itemOps[o];
            int u = Universe(operand);
            if (u < 0)
            {
                continue;
            }

            for (int j = 0; j < operand.Count; j++)
            {
                _idFlags[u + j] |= Mentioned;
                if (j > 0)
                {
                    Union(u, u + j);
                }
            }
        }

        int units = 0;
        for (int u = 0; u < ids; u++)
        {
            if ((_idFlags[u] & Mentioned) == 0)
            {
                continue;
            }

            int root = Find(u);
            if (_unitOf[root] < 0)
            {
                Grow(ref _unitLo, units + 1);
                Grow(ref _unitSize, units + 1);
                Grow(ref _unitFile, units + 1);
                Grow(ref _unitMask, units + 1);
                _unitLo[units] = u;
                _unitSize[units] = 0;
                _unitFile[units] = (byte)(u < _sIds ? 0 : 1);
                _unitMask[units] = 0xF;
                _unitOf[root] = units++;
            }

            int unit = _unitOf[root];
            _unitOf[u] = unit;
            _unitSize[unit] = u - _unitLo[unit] + 1;
        }

        _unitCount = units;

        // Each scalar range's alignment, as residues mod 4 of the unit's first register.
        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            for (int i = 0; i < item.Count; i++)
            {
                var operand = _itemOps[item.First + i];
                if (operand.Kind != GcnOperandKind.VirtualSgpr || operand.Count < 2)
                {
                    continue;
                }

                int unit = _unitOf[operand.Index], modulus = operand.Count >= 4 ? 4 : 2, offset = operand.Index - _unitLo[unit];
                int allowed = 0;
                for (int r = 0; r < 4; r++)
                {
                    allowed |= (r + offset) % modulus == 0 ? 1 << r : 0;
                }

                if ((_unitMask[unit] & allowed) == 0)
                {
                    throw new GcnSyntaxException(item.Line, item.Columns[i], $"{operand} cannot be aligned as the hardware needs ({(modulus == 2 ? "an even pair" : "a multiple of 4")}) together with the other ranges its registers are part of");
                }

                _unitMask[unit] &= (byte)allowed;
            }
        }
    }

    // The virtual register's index among both files (scalar ids, then vector ones); -1 for anything else.
    private int Universe(GcnOperand operand) => operand.Kind switch
    {
        GcnOperandKind.VirtualSgpr => operand.Index,
        GcnOperandKind.VirtualVgpr => _sIds + operand.Index,
        _ => -1,
    };

    // A register's index for liveness: the virtual ones, then the system values the text reads (scalar, then vector),
    // whose registers are held until they are last live on any path, not until their last read in the text's order.
    private int LiveIndex(GcnOperand operand) => operand.Kind switch
    {
        GcnOperandKind.VirtualSgpr => operand.Index,
        GcnOperandKind.VirtualVgpr => _sIds + operand.Index,
        GcnOperandKind.Sgpr => _sIds + _vIds + operand.Index,
        GcnOperandKind.Vgpr => _sIds + _vIds + _systemSgprs + operand.Index,
        _ => -1,
    };

    private int Find(int u)
    {
        while (_parent[u] != u)
        {
            _parent[u] = _parent[_parent[u]];
            u = _parent[u];
        }

        return u;
    }

    private void Union(int a, int b)
    {
        a = Find(a);
        b = Find(b);
        if (a != b)
        {
            _parent[Math.Max(a, b)] = Math.Min(a, b);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Liveness over the blocks (backwards to a fixed point), then where each unit is live: from its first definition
    // (or the start of a block it is live into) to its last use (or the end of a block it is live out of). An instruction
    // reads at position 2k and writes at 2k + 1, so a register last read by an instruction can be written by it. A
    // vector register written lane by lane under a narrowed EXEC keeps its other lanes: where a value of it may reach
    // that write (a forward pass finds where each register may be defined), the write reads it too, so it is live from
    // before; where none can, its other lanes hold nothing and the write starts it.
    // ---------------------------------------------------------------------------------------------------------------

    private static bool Uses(Access access, bool keeps) => (access & Access.Read) != 0 || keeps;

    private void Liveness()
    {
        int virtuals = _sIds + _vIds, ids = virtuals + _systemSgprs + _workitemIds, blocks = _blockCount;
        int words = _words = Math.Max(1, (ids + 63) >> 6);
        int gen = 0, kill = blocks * words, live = 2 * blocks * words, liveOut = 3 * blocks * words, defined = 4 * blocks * words, written = 5 * blocks * words;
        Grow(ref _live, 6 * blocks * words);
        Grow(ref _running, words);
        Grow(ref _keeps, _itemCount);
        Grow(ref _idFlags, ids);
        _idFlags.AsSpan(virtuals, ids - virtuals).Fill(Mentioned | Written);       // the system values: the hardware writes them
        _live.AsSpan(0, 6 * blocks * words).Clear();

        // Where each register may be defined as a block starts: forwards, the union of its predecessors' ends.
        for (int b = 0; b < blocks; b++)
        {
            for (int k = _blockStart[b]; k < _blockEnd[b]; k++)
            {
                ref var item = ref _items[k];
                for (int i = 0; i < item.Count; i++)
                {
                    var operand = _itemOps[item.First + i];
                    int u = LiveIndex(operand);
                    if (u >= 0 && (AccessOf(item.Ins, i, item.Encoded, item.M) & Access.Write) != 0)
                    {
                        for (int j = u; j < u + operand.Count; j++)
                        {
                            Set(written + b * words, j);
                            _idFlags[j] |= Written;
                        }
                    }
                }
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int b = 0; b < blocks; b++)
            {
                for (int s = 2 * b; s < 2 * b + 2; s++)
                {
                    int next = _succ[s];
                    for (int w = 0; next >= 0 && w < words; w++)
                    {
                        ulong bits = _live[defined + b * words + w] | _live[written + b * words + w];
                        if ((bits & ~_live[defined + next * words + w]) != 0)
                        {
                            _live[defined + next * words + w] |= bits;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Each block's reads before writes (gen) and full writes (kill); which lane-wise writes keep a value.
        for (int b = 0; b < blocks; b++)
        {
            _live.AsSpan(defined + b * words, words).CopyTo(_running);
            for (int k = _blockStart[b]; k < _blockEnd[b]; k++)
            {
                ref var item = ref _items[k];
                bool narrowed = (_itemFlags[k] & Narrowed) != 0;
                int keeps = 0;
                for (int i = 0; i < item.Count; i++)
                {
                    var operand = _itemOps[item.First + i];
                    int u = LiveIndex(operand);
                    var access = AccessOf(item.Ins, i, item.Encoded, item.M);
                    if (u >= 0 && narrowed && (access & (Access.Write | Access.LaneWise)) == (Access.Write | Access.LaneWise))
                    {
                        for (int j = u; j < u + operand.Count; j++)
                        {
                            keeps |= (_running[j >> 6] >> (j & 63) & 1) != 0 ? 1 << i : 0;
                        }
                    }
                }

                _keeps[k] = (byte)keeps;
                for (int pass = 0; pass < 2; pass++)
                {
                    for (int i = 0; i < item.Count; i++)
                    {
                        var operand = _itemOps[item.First + i];
                        int u = LiveIndex(operand);
                        if (u < 0)
                        {
                            continue;
                        }

                        var access = AccessOf(item.Ins, i, item.Encoded, item.M);
                        bool uses = Uses(access, (keeps >> i & 1) != 0);
                        for (int j = u; j < u + operand.Count; j++)
                        {
                            if (pass == 0 && uses && !Test(kill + b * words, j))
                            {
                                Set(gen + b * words, j);
                            }
                            else if (pass == 1 && (access & Access.Write) != 0)
                            {
                                _running[j >> 6] |= 1UL << (j & 63);
                                if (!uses)
                                {
                                    Set(kill + b * words, j);
                                }
                            }
                        }
                    }
                }
            }
        }

        changed = true;
        while (changed)
        {
            changed = false;
            for (int b = blocks - 1; b >= 0; b--)
            {
                for (int w = 0; w < words; w++)
                {
                    ulong outBits = 0;
                    for (int s = 2 * b; s < 2 * b + 2; s++)
                    {
                        outBits |= _succ[s] >= 0 ? _live[live + _succ[s] * words + w] : 0;
                    }

                    ulong inBits = _live[gen + b * words + w] | (outBits & ~_live[kill + b * words + w]);
                    _live[liveOut + b * words + w] = outBits;
                    if (inBits != _live[live + b * words + w])
                    {
                        _live[live + b * words + w] = inBits;
                        changed = true;
                    }
                }
            }
        }

        // A register live as the kernel starts and never written is read but never set.
        for (int w = 0; w < words; w++)
        {
            for (ulong bits = _live[live + w]; bits != 0; bits &= bits - 1)
            {
                int u = 64 * w + BitOperations.TrailingZeroCount(bits);
                if ((_idFlags[u] & Written) == 0)
                {
                    ThrowNeverWritten(u);
                }
            }
        }

        Grow(ref _idFirst, Math.Max(1, ids));
        Grow(ref _idLast, Math.Max(1, ids));
        _idFirst.AsSpan(0, ids).Fill(int.MaxValue);
        _idLast.AsSpan(0, ids).Fill(-1);
        for (int b = 0; b < blocks; b++)
        {
            int start = 2 * _blockStart[b], end = 2 * _blockEnd[b];
            for (int w = 0; w < words; w++)
            {
                for (ulong bits = _live[live + b * words + w]; bits != 0; bits &= bits - 1)
                {
                    int u = 64 * w + BitOperations.TrailingZeroCount(bits);
                    _idFirst[u] = Math.Min(_idFirst[u], start);
                    _idLast[u] = Math.Max(_idLast[u], start);
                }

                for (ulong bits = _live[liveOut + b * words + w]; bits != 0; bits &= bits - 1)
                {
                    _idLast[64 * w + BitOperations.TrailingZeroCount(bits)] = Math.Max(_idLast[64 * w + BitOperations.TrailingZeroCount(bits)], end);
                }
            }
        }

        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            for (int i = 0; i < item.Count; i++)
            {
                var operand = _itemOps[item.First + i];
                int u = LiveIndex(operand);
                if (u < 0)
                {
                    continue;
                }

                var access = AccessOf(item.Ins, i, item.Encoded, item.M);
                int first = Uses(access, (_keeps[k] >> i & 1) != 0) ? 2 * k : 2 * k + 1, last = (access & Access.Write) != 0 ? 2 * k + 1 : 2 * k;
                for (int j = u; j < u + operand.Count; j++)
                {
                    _idFirst[j] = Math.Min(_idFirst[j], first);
                    _idLast[j] = Math.Max(_idLast[j], last);
                }
            }
        }

        int units = _unitCount;
        Grow(ref _unitStart, units);
        Grow(ref _unitEnd, units);
        Grow(ref _unitLoad, units);
        Grow(ref _unitPhys, units);
        _unitStart.AsSpan(0, units).Fill(int.MaxValue);
        _unitEnd.AsSpan(0, units).Fill(-1);
        _unitLoad.AsSpan(0, units).Fill(-1);
        _unitPhys.AsSpan(0, units).Fill(-1);
        for (int u = 0; u < virtuals; u++)
        {
            int unit = _unitOf[u];
            if (unit >= 0)
            {
                _unitStart[unit] = Math.Min(_unitStart[unit], _idFirst[u]);
                _unitEnd[unit] = Math.Max(_unitEnd[unit], _idLast[u]);
            }
        }

        // Rule 80: a load's destination is noted with its block, so no other load of that block shares its registers.
        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            if (MemoryOf(item.Ins, item.M) == GcnMemoryClass.None)
            {
                continue;
            }

            for (int i = 0; i < item.Count; i++)
            {
                int u = Universe(_itemOps[item.First + i]);
                if (u >= 0 && (AccessOf(item.Ins, i, item.Encoded, item.M) & Access.Write) != 0 && _unitLoad[_unitOf[u]] < 0)
                {
                    _unitLoad[_unitOf[u]] = _itemBlock[k];
                }
            }
        }
    }

    private bool Test(int at, int u) => (_live[at + (u >> 6)] >> (u & 63) & 1) != 0;

    private void Set(int at, int u) => _live[at + (u >> 6)] |= 1UL << (u & 63);

    private void ThrowNeverWritten(int u)
    {
        bool scalar = u < _sIds;
        int id = scalar ? u : u - _sIds;
        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            for (int i = 0; i < item.Count; i++)
            {
                var operand = _itemOps[item.First + i];
                int v = Universe(operand);
                if (v >= 0 && u >= v && u < v + operand.Count)
                {
                    throw new GcnSyntaxException(item.Line, item.Columns[i], $"%{(scalar ? 's' : 'v')}{id} is read but never written");
                }
            }
        }

        throw new GcnSyntaxException(1, 1, $"%{(scalar ? 's' : 'v')}{id} is read but never written");
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Allocation: a linear scan over the units by where they start. A unit takes the register an operand of its first
    // instruction frees there when it fits (so v0 = s2 + v0 keeps v0), else the lowest place that fits: free from its
    // start, aligned, within the budget, and (rule 80) not written by another load of the same block. The system values
    // the text reads hold their registers until they are last live, on any path. The budget: the most registers that
    // still let a SIMD hold the waves the declared block needs (a workgroup's waves spread over a compute unit's SIMDs),
    // as .maxntid bounds ptxas; the reserved scalar registers (VCC, the xnack mask, flat scratch) are above those.
    // ---------------------------------------------------------------------------------------------------------------

    private void Allocate(int? wavesPerSimd)
    {
        var isa = _isa ?? default;
        int wave = _target.WaveSizes[0];
        var file = _target.Registers(isa, wave);
        int waves = (_maxFlat + wave - 1) / wave;
        int needed = wavesPerSimd ?? Math.Max(1, (waves + _set.SimdsPerComputeUnit - 1) / _set.SimdsPerComputeUnit);
        if (needed < 1 || needed > file.MaxWavesPerSimd)
        {
            throw new ArgumentOutOfRangeException(nameof(wavesPerSimd), needed, $"Kernel '{_kernelName}': {_family} holds 1 to {file.MaxWavesPerSimd} waves a SIMD.");
        }

        int vectorBudget = 0, scalarBudget = 0;
        for (int v = file.Vgprs; v >= 1 && vectorBudget == 0; v--)
        {
            vectorBudget = _target.WavesPerSimd(isa, v, 0, wave) >= needed ? v : 0;
        }

        for (int s = file.Sgprs; s >= 1 && scalarBudget == 0; s--)
        {
            scalarBudget = _target.WavesPerSimd(isa, 1, _target.SgprCount(isa, s, vcc: true, flatScratch: false), wave) >= needed ? s : 0;
        }

        _busy.AsSpan().Fill(-1);
        _loadOwner.AsSpan().Fill(-1);
        _maxSgpr = _maxVgpr = -1;
        for (int k = 0; k < _itemCount; k++)
        {
            ref var item = ref _items[k];
            for (int i = 0; i < item.Count; i++)
            {
                var operand = _itemOps[item.First + i];
                if (operand.Kind is GcnOperandKind.Sgpr or GcnOperandKind.Vgpr)
                {
                    bool scalar = operand.Kind == GcnOperandKind.Sgpr;
                    int live = LiveIndex(operand);
                    for (int j = 0; j < operand.Count; j++)
                    {
                        _busy[(scalar ? 0 : VgprBase) + operand.Index + j] = _idLast[live + j];
                    }

                    if (scalar)
                    {
                        _maxSgpr = Math.Max(_maxSgpr, operand.Index + operand.Count - 1);
                    }
                    else
                    {
                        _maxVgpr = Math.Max(_maxVgpr, operand.Index + operand.Count - 1);
                    }
                }
            }
        }

        int units = _unitCount;
        Grow(ref _order, Math.Max(1, units));
        for (int unit = 0; unit < units; unit++)
        {
            _order[unit] = (long)_unitStart[unit] << 24 | (uint)unit;
        }

        _order.AsSpan(0, units).Sort();
        for (int n = 0; n < units; n++)
        {
            int unit = (int)(_order[n] & 0xFFFFFF);
            bool scalar = _unitFile[unit] == 0;
            int budget = scalar ? scalarBudget : vectorBudget;
            int at = Hint(unit, budget);
            for (int p = 0; at < 0 && p + _unitSize[unit] <= budget; p++)
            {
                at = Fits(unit, p, budget) ? p : -1;
            }

            if (at < 0)
            {
                ThrowOutOfRegisters(unit, budget, needed);
            }

            int offset = scalar ? 0 : VgprBase;
            for (int j = 0; j < _unitSize[unit]; j++)
            {
                _busy[offset + at + j] = _unitEnd[unit];
                if (_unitLoad[unit] >= 0)
                {
                    _loadOwner[offset + at + j] = _unitLoad[unit];
                }
            }

            _unitPhys[unit] = at;
            if (scalar)
            {
                _maxSgpr = Math.Max(_maxSgpr, at + _unitSize[unit] - 1);
            }
            else
            {
                _maxVgpr = Math.Max(_maxVgpr, at + _unitSize[unit] - 1);
            }
        }
    }

    // The register an operand of the unit's first instruction frees there, when the unit fits it; else -1.
    private int Hint(int unit, int budget)
    {
        int start = _unitStart[unit];
        if ((start & 1) == 0)
        {
            return -1;
        }

        int k = start >> 1;
        bool scalar = _unitFile[unit] == 0;
        ref var item = ref _items[k];
        for (int i = 0; i < item.Count; i++)
        {
            var operand = _itemOps[item.First + i];
            if ((AccessOf(item.Ins, i, item.Encoded, item.M) & Access.Read) == 0)
            {
                continue;
            }

            int at = operand.Kind switch
            {
                GcnOperandKind.Sgpr when scalar => operand.Index,
                GcnOperandKind.Vgpr when !scalar => operand.Index,
                GcnOperandKind.VirtualSgpr when scalar && _unitPhys[_unitOf[operand.Index]] >= 0 => Physical(operand).Index,
                GcnOperandKind.VirtualVgpr when !scalar && _unitPhys[_unitOf[_sIds + operand.Index]] >= 0 => Physical(operand).Index,
                _ => -1,
            };
            if (at >= 0 && _busy[(scalar ? 0 : VgprBase) + at] == start - 1 && Fits(unit, at, budget))
            {
                return at;
            }
        }

        return -1;
    }

    private bool Fits(int unit, int at, int budget)
    {
        bool scalar = _unitFile[unit] == 0;
        int size = _unitSize[unit], offset = scalar ? 0 : VgprBase, start = _unitStart[unit], load = _unitLoad[unit];
        if (at + size > budget || (scalar && (_unitMask[unit] >> (at & 3) & 1) == 0))
        {
            return false;
        }

        for (int j = offset + at; j < offset + at + size; j++)
        {
            if (_busy[j] >= start || (load >= 0 && _loadOwner[j] == load))
            {
                return false;
            }
        }

        return true;
    }

    private void ThrowOutOfRegisters(int unit, int budget, int waves)
    {
        bool scalar = _unitFile[unit] == 0;
        int start = _unitStart[unit], size = _unitSize[unit], offset = scalar ? 0 : VgprBase, live = 0;
        for (int r = 0; r < budget; r++)
        {
            live += _busy[offset + r] >= start ? 1 : 0;
        }

        // The kernel's peak: the most registers of the file live at once (where a unit starts), system values included.
        int system = _sIds + _vIds + (scalar ? 0 : _systemSgprs), systemCount = scalar ? _systemSgprs : _workitemIds;
        int peak = 0, peakAt = start;
        for (int a = 0; a < _unitCount; a++)
        {
            int at = _unitStart[a], here = 0;
            if (_unitFile[a] != _unitFile[unit])
            {
                continue;
            }

            for (int b = 0; b < _unitCount; b++)
            {
                here += _unitFile[b] == _unitFile[unit] && _unitStart[b] <= at && _unitEnd[b] >= at ? _unitSize[b] : 0;
            }

            for (int r = 0; r < systemCount; r++)
            {
                here += _idLast[system + r] >= at ? 1 : 0;
            }

            (peak, peakAt) = here > peak ? (here, at) : (peak, peakAt);
        }

        string what = scalar ? "scalar" : "vector";
        int line = _items[Math.Min(start >> 1, _itemCount - 1)].Line, peakLine = _items[Math.Min(peakAt >> 1, _itemCount - 1)].Line;
        int lo = _unitLo[unit] - (scalar ? 0 : _sIds);
        string name = size == 1 ? $"%{(scalar ? 's' : 'v')}{lo}" : $"%{(scalar ? 's' : 'v')}[{lo}:{lo + size - 1}]";
        int needed = peak > budget ? peak : live + size;
        string reason = peak > budget
            ? $"needs {peak} {what} registers at once (at line {peakLine}), and {budget} fit"
            : $"cannot place {name} ({size} adjacent {what} registers{(scalar && size > 1 ? ", aligned" : "")}) at line {line}: {live} of {budget} hold live values{(_unitLoad[unit] >= 0 ? ", and a load's destination takes none another load of its block wrote" : "")}";
        throw new GcnAllocationException(_kernelName!, scalar ? 's' : 'v', needed, budget,
            $"GCN kernel '{_kernelName}' {reason}: its block of {_maxFlat} work-items needs {waves} wave(s) a SIMD on {_family}, which leaves a {(scalar ? "wave" : "lane")} {budget} {what} registers. Idrak's assembler does not spill yet.");
    }
}
