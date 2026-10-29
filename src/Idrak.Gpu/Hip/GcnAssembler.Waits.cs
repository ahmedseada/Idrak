// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

// The waits of a kernel's text: a scoreboard of the memory counters (vmcnt, lgkmcnt, expcnt) as the family counts
// each class of operation (IGcnInstructionSet.Counting), and s_waitcnt placed as late as possible: just before the
// first instruction that reads or writes a register a memory operation has not returned yet. Each counter holds a
// score for every operation issued (its upper bound) and what a wait has retired (its lower bound); a register holds
// the score of the operation that writes it. Counters that complete in order wait for just the operations up to that
// one (vmcnt(1) when one newer is in flight); a counter with an operation that completes out of order in flight (SMEM,
// FLAT), or two classes mixed, waits to 0. s_barrier waits for every counter first where the family's barrier does not.
// Across blocks the state at a block's start merges its predecessors' (the most in flight, each register's newest
// pending write), to a fixed point over loops; the hazards (GcnAssembler.Hazards.cs) are walked alongside, so the
// wait states count the waits placed.
internal sealed partial class GcnAssembler
{
    private const int Counters = 3, Classes = 8;
    private const int CounterBytes = 400, WaitBytes = Counters * CounterBytes, StateBytes = WaitBytes + HazardBytes;
    private const byte Seen = 1, Dirty = 2;

    // Counter 0 vmcnt, 1 lgkmcnt, 2 expcnt.
    private readonly int[] _score = new int[Counters * RegisterSpace];
    private readonly int[] _classScore = new int[Counters * Classes];
    private readonly int[] _ub = new int[Counters], _lb = new int[Counters], _limit = new int[Counters], _need = new int[Counters];
    private readonly GcnCounting[] _counting = new GcnCounting[Classes];

    // The physical registers the instruction being walked reads and writes (scalar codes, VGPRs at 128 + n).
    private readonly int[] _reads = new int[256], _writes = new int[256];
    private int _readCount, _writeCount;

    private byte[] _states = [], _visited = [];
    private readonly byte[] _exit = new byte[StateBytes];

    private void RentWaitBuffers()
    {
        _states = ArrayPool<byte>.Shared.Rent(4 * StateBytes);
        _visited = ArrayPool<byte>.Shared.Rent(16);
        _limit[0] = _set.CounterLimit(GcnCounter.Vm);
        _limit[1] = _set.CounterLimit(GcnCounter.Lgkm);
        _limit[2] = _set.CounterLimit(GcnCounter.Exp);
        for (int c = 0; c < Classes; c++)
        {
            _counting[c] = c <= (int)GcnMemoryClass.Scalar ? _set.Counting((GcnMemoryClass)c) : default;
        }

        for (int h = 0; h < _hazardStates.Length; h++)
        {
            _hazardStates[h] = _set.HazardWaitStates((GcnHazard)h, _isa?.Xnack ?? GcnFeatureSetting.Any);
        }
    }

    private void ReturnWaitBuffers()
    {
        ReturnPooled(ref _states);
        ReturnPooled(ref _visited);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The walk: blocks to a fixed point (nothing emitted), then once more, emitting each block's rows.
    // ---------------------------------------------------------------------------------------------------------------

    private void PlaceWaitsAndHazards()
    {
        int blocks = _blockCount;
        Grow(ref _states, blocks * StateBytes);
        Grow(ref _visited, blocks);
        _visited.AsSpan(0, blocks).Clear();
        InitialState(_states.AsSpan(0, StateBytes));
        _visited[0] = Seen | Dirty;
        bool again = true;
        for (int sweeps = 0; again; sweeps++)
        {
            if (sweeps > 2 * blocks + 64)
            {
                throw new InvalidOperationException($"GCN kernel '{_kernelName}': the waits and hazards did not settle over its {blocks} blocks.");
            }

            again = false;
            for (int b = 0; b < blocks; b++)
            {
                if ((_visited[b] & Dirty) == 0)
                {
                    continue;
                }

                _visited[b] &= unchecked((byte)~Dirty);
                LoadState(_states.AsSpan(b * StateBytes, StateBytes));
                Walk(b, emit: false);
                SaveState(_exit);
                for (int s = 2 * b; s < 2 * b + 2; s++)
                {
                    int next = _succ[s];
                    if (next < 0)
                    {
                        continue;
                    }

                    var into = _states.AsSpan(next * StateBytes, StateBytes);
                    if ((_visited[next] & Seen) == 0)
                    {
                        _exit.CopyTo(into);
                        _visited[next] = Seen | Dirty;
                        again = true;
                    }
                    else if (MergeState(into, _exit))
                    {
                        _visited[next] |= Dirty;
                        again = true;
                    }
                }
            }
        }

        _rowCount = 0;
        for (int b = 0; b < blocks; b++)
        {
            var state = _states.AsSpan(b * StateBytes, StateBytes);
            if ((_visited[b] & Seen) == 0)
            {
                InitialState(state);
            }

            LoadState(state);
            Walk(b, emit: true);
        }
    }

    // One block from the state loaded: before each instruction the wait it needs, then the wait states, then the
    // instruction's own effect on both.
    private void Walk(int b, bool emit)
    {
        for (int k = _blockStart[b]; k < _blockEnd[b]; k++)
        {
            ref var item = ref _items[k];
            Touch(k);
            if (item.Ins.Op == GcnOp.SWaitcnt)
            {
                var (vm, exp, lgkm) = _set.WaitCounts((ushort)_itemOps[item.First].Value);
                _need[0] = vm < _limit[0] ? vm : -1;
                _need[1] = lgkm < _limit[1] ? lgkm : -1;
                _need[2] = exp < _limit[2] ? exp : -1;
                ApplyWait();
            }
            else if (WaitNeeded(item))
            {
                ushort bits = _set.WaitBits(_need[0] < 0 ? _limit[0] : _need[0], _need[2] < 0 ? _limit[2] : _need[2], _need[1] < 0 ? _limit[1] : _need[1]);
                ApplyWait();
                HazardAdvance(1);
                if (emit)
                {
                    AddRow(k, RowWait, bits);
                }
            }

            for (int nops = HazardNeed(k); nops > 0;)
            {
                int n = Math.Min(nops, MaxNop);
                HazardAdvance(n);
                nops -= n;
                if (emit)
                {
                    AddRow(k, RowNop, (ushort)(n - 1));
                }
            }

            Issue(k);
            HazardRecord(k);
            if (emit)
            {
                AddRow(k, RowItem, 0);
            }
        }
    }

    // The physical registers item k reads and writes, its implicit ones (EXEC, VCC, M0) included. A vector register
    // whose write keeps the lanes EXEC leaves out is read too.
    private void Touch(int k)
    {
        _readCount = _writeCount = 0;
        ref var item = ref _items[k];
        for (int i = 0; i < item.Count; i++)
        {
            var access = AccessOf(item.Ins, i, item.Encoded, item.M);
            if (access == Access.None || !RegisterCodes(Physical(_itemOps[item.First + i]), out int code, out int count))
            {
                continue;
            }

            bool read = Uses(access, (_keeps[k] >> i & 1) != 0), write = (access & Access.Write) != 0;
            for (int j = code; j < code + count; j++)
            {
                if (read)
                {
                    _reads[_readCount++] = j;
                }

                if (write)
                {
                    _writes[_writeCount++] = j;
                }
            }
        }

        var ins = item.Ins;
        if (ReadsExec(ins))
        {
            AddTouch(_reads, ref _readCount, _set.Special(GcnSpecial.Exec), 2);
        }

        if (WritesExec(ins))
        {
            AddTouch(_writes, ref _writeCount, _set.Special(GcnSpecial.Exec), 2);
        }

        if (ReadsVccImplicitly(ins))
        {
            AddTouch(_reads, ref _readCount, _set.Special(GcnSpecial.Vcc), 2);
        }

        if (ReadsM0(ins, item.M))
        {
            AddTouch(_reads, ref _readCount, _set.Special(GcnSpecial.M0), 1);
        }
    }

    private static void AddTouch(int[] list, ref int count, int code, int registers)
    {
        for (int j = code; j < code + registers; j++)
        {
            list[count++] = j;
        }
    }

    // A register operand's first code (scalar codes 0-127, VGPR n as 128 + n) and how many; false for anything else.
    private bool RegisterCodes(GcnOperand operand, out int code, out int count)
    {
        switch (operand.Kind)
        {
            case GcnOperandKind.Sgpr:
                (code, count) = (operand.Index, operand.Count);
                return true;
            case GcnOperandKind.Vgpr:
                (code, count) = (VgprBase + operand.Index, operand.Count);
                return true;
            case GcnOperandKind.Special when IsRegister((GcnSpecial)operand.Index):
                (code, count) = (_set.Special((GcnSpecial)operand.Index), Math.Max(1, operand.Count));
                return true;
            default:
                (code, count) = (0, 0);
                return false;
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The scoreboard.
    // ---------------------------------------------------------------------------------------------------------------

    // The counts a wait before the instruction needs, in _need (-1: not waited on); false when it needs none.
    private bool WaitNeeded(in Item item)
    {
        _need[0] = _need[1] = _need[2] = 0;
        for (int n = 0; n < _readCount; n++)
        {
            Pending(_reads[n], counters: 2);
        }

        for (int n = 0; n < _writeCount; n++)
        {
            Pending(_writes[n], counters: 3);
        }

        bool barrier = item.Ins.Op == GcnOp.SBarrier && !_set.BarrierWaitsForMemory;
        bool any = false;
        for (int c = 0; c < Counters; c++)
        {
            if (barrier && _ub[c] > _lb[c])
            {
                _need[c] = _ub[c];
            }

            if (_need[c] > 0)
            {
                _need[c] = OutOfOrder(c) ? 0 : Math.Min(_ub[c] - _need[c], _limit[c] - 1);
                any = true;
            }
            else
            {
                _need[c] = -1;
            }
        }

        return any;
    }

    // Notes the newest pending write of register r on the first `counters` counters (expcnt's data locks are waited
    // for by writes only).
    private void Pending(int r, int counters)
    {
        for (int c = 0; c < counters; c++)
        {
            int score = _score[c * RegisterSpace + r];
            if (score > _lb[c] && score > _need[c])
            {
                _need[c] = score;
            }
        }
    }

    // Whether counter c may complete out of order now: an operation of a class that does is in flight, or two classes are.
    private bool OutOfOrder(int c)
    {
        int classes = 0;
        for (int cls = 1; cls < Classes; cls++)
        {
            if (_classScore[c * Classes + cls] > _lb[c])
            {
                if (!_counting[cls].InOrder)
                {
                    return true;
                }

                classes++;
            }
        }

        return classes > 1;
    }

    private void ApplyWait()
    {
        for (int c = 0; c < Counters; c++)
        {
            if (_need[c] >= 0)
            {
                _lb[c] = Math.Max(_lb[c], _ub[c] - _need[c]);
            }
        }
    }

    // The instruction's memory operation: a new score on each counter its class increments, its destinations pending on
    // them, its data registers locked on the data lock's counter.
    private void Issue(int k)
    {
        ref var item = ref _items[k];
        var cls = MemoryOf(item.Ins, item.M);
        if (cls == GcnMemoryClass.None)
        {
            return;
        }

        var counting = _counting[(int)cls];
        for (int c = 0; c < 2; c++)
        {
            if ((counting.Counters & (c == 0 ? GcnCounter.Vm : GcnCounter.Lgkm)) == 0)
            {
                continue;
            }

            int score = ++_ub[c];
            _classScore[c * Classes + (int)cls] = score;
            for (int n = 0; n < _writeCount; n++)
            {
                _score[c * RegisterSpace + _writes[n]] = score;
            }

            _lb[c] = Math.Max(_lb[c], _ub[c] - _limit[c]);
        }

        if ((counting.DataLock & GcnCounter.Exp) != 0)
        {
            int score = ++_ub[2];
            _classScore[2 * Classes + (int)cls] = score;
            for (int i = 0; i < item.Count; i++)
            {
                if (item.Ins.Operands.Role(i) is GcnRole.Data0 or GcnRole.Data1 && RegisterCodes(Physical(_itemOps[item.First + i]), out int code, out int count))
                {
                    _score.AsSpan(2 * RegisterSpace + code, count).Fill(score);
                }
            }

            _lb[2] = Math.Max(_lb[2], _ub[2] - _limit[2]);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // States at block starts, relative to each counter's newest operation: per counter the operations in flight, then
    // per register and per class how many newer ones there are (255: none pending); then the hazards' part.
    // ---------------------------------------------------------------------------------------------------------------

    private void InitialState(Span<byte> state)
    {
        for (int c = 0; c < Counters; c++)
        {
            state[c * CounterBytes] = 0;
            state.Slice(c * CounterBytes + 1, CounterBytes - 1).Fill(byte.MaxValue);
        }

        InitialHazards(state[WaitBytes..]);
    }

    private void SaveState(Span<byte> state)
    {
        for (int c = 0; c < Counters; c++)
        {
            int ub = _ub[c], lb = _lb[c];
            var counter = state.Slice(c * CounterBytes, CounterBytes);
            counter[0] = (byte)Math.Min(ub - lb, 254);
            for (int r = 0; r < RegisterSpace; r++)
            {
                int score = _score[c * RegisterSpace + r];
                counter[1 + r] = score > lb ? (byte)Math.Min(ub - score, 254) : byte.MaxValue;
            }

            for (int cls = 0; cls < Classes; cls++)
            {
                int score = _classScore[c * Classes + cls];
                counter[1 + RegisterSpace + cls] = score > lb ? (byte)Math.Min(ub - score, 254) : byte.MaxValue;
            }
        }

        SaveHazards(state[WaitBytes..]);
    }

    private void LoadState(ReadOnlySpan<byte> state)
    {
        for (int c = 0; c < Counters; c++)
        {
            var counter = state.Slice(c * CounterBytes, CounterBytes);
            int ub = counter[0];
            _ub[c] = ub;
            _lb[c] = 0;
            for (int r = 0; r < RegisterSpace; r++)
            {
                _score[c * RegisterSpace + r] = counter[1 + r] == byte.MaxValue ? 0 : ub - counter[1 + r];
            }

            for (int cls = 0; cls < Classes; cls++)
            {
                _classScore[c * Classes + cls] = counter[1 + RegisterSpace + cls] == byte.MaxValue ? 0 : ub - counter[1 + RegisterSpace + cls];
            }
        }

        LoadHazards(state[WaitBytes..]);
    }

    // Merges a predecessor's exit into a block's start: the most operations in flight, each register's and class's newest
    // pending write; the hazards' nearest writes and the clauses joined. True when the start changed.
    private static bool MergeState(Span<byte> into, ReadOnlySpan<byte> from)
    {
        bool changed = false;
        for (int c = 0; c < Counters; c++)
        {
            int at = c * CounterBytes;
            if (from[at] > into[at])
            {
                into[at] = from[at];
                changed = true;
            }

            for (int i = at + 1; i < at + CounterBytes; i++)
            {
                if (from[i] < into[i])
                {
                    into[i] = from[i];
                    changed = true;
                }
            }
        }

        return MergeHazards(into[WaitBytes..], from[WaitBytes..]) | changed;
    }
}
