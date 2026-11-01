// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

// The hazards of a kernel's text: where the hardware does not check a dependency itself, the wait states (s_nop) the
// family needs between two instructions (IGcnInstructionSet.HazardWaitStates, by case). A clock counts wait states
// (an instruction 1, s_nop n n + 1); each register remembers when a VALU or an SALU last wrote it, each hardware
// register when s_setreg wrote it, each VGPR when a wide store last read it; the current soft clause (consecutive SMEM,
// or consecutive vector memory, instructions) keeps what its members read and wrote. Before an instruction, each case
// that applies asks for its count less the wait states since the write; the most asked for is inserted. Across blocks
// the nearest write of every predecessor counts, so a hazard is covered on every path, as LLVM's recognizer does.
// The hazards a family fixes with instructions of their own (IGcnInstructionSet.HazardFixes: ISA 10.1's and 11's) keep
// their own state: the SGPRs vector memory read, those SMEM read and those VALUs read as lane masks since what clears
// each, and for every VGPR the VALUs and transcendentals since a transcendental wrote it.
internal sealed partial class GcnAssembler
{
    // Where each kind of write is kept: VALU writes of scalar codes 0-127 and VGPRs, SALU writes of scalar codes, s_setreg
    // by hardware register id, wide stores' data VGPRs; then the clause's kind (1 SMEM, 2 vector memory) and the
    // registers it wrote and read (384 bits each).
    private const int ValuSgpr = 0, ValuVgpr = 128, SaluSgpr = 384, Setreg = 512, StoreData = 576, HazardEntries = 832;
    private const int ClauseKindAt = HazardEntries, ClauseDefsAt = HazardEntries + 1, ClauseUsesAt = ClauseDefsAt + 48;

    // Then the fixes' masks (vector memory's, SMEM's and the lane masks' reads of scalar codes 0-127, 16 bytes each), and
    // per VGPR the VALUs (256 bytes), then the transcendentals (256), since a transcendental wrote it.
    private const int FixMasksAt = 936, TransAt = FixMasksAt + 48, HazardBytes = TransAt + 512;

    // A transcendental's result is safe to read past 5 VALUs or 1 transcendental between (llc 18, gfx1100 and gfx1102).
    private const int TransValus = 6, TransTrans = 2;

    // s_waitcnt_depctr's bits: vm_vsrc(0), va_vdst(0), sa_sdst(0), the other counters left as they are.
    private const ushort DepctrVmVsrc = 0xFFE3, DepctrVaVdst = 0x0FFF, DepctrSaSdst = 0xFFFE;

    // Wait states past the longest any case needs: a write this far back is no hazard.
    private const int Far = 16;

    // The most wait states one s_nop gives (s_nop 7), as LLVM inserts them.
    private const int MaxNop = 8;

    private readonly int[] _hzTime = new int[HazardEntries];
    private readonly int[] _hazardStates = new int[(int)GcnHazard.SoftClause + 1];
    private readonly ulong[] _clauseDefs = new ulong[6], _clauseUses = new ulong[6], _xDefs = new ulong[6], _xUses = new ulong[6];
    private int _clock, _clauseKind;

    // The fixes this version needs; their masks (0-1 vector memory's reads, 2-3 SMEM's, 4-5 the lane masks'), the clocks
    // of VALUs and transcendentals and when a transcendental last wrote each VGPR; the family's null, EXEC, M0 and VCC.
    private GcnHazardFixes _fixes;
    private readonly ulong[] _fixMasks = new ulong[6];
    private readonly int[] _transValu = new int[256], _transTrans = new int[256];
    private int _valuClock, _transClock, _nullCode = -1, _execCode = -1, _m0Code = -1, _vccCode = -1;

    private int Since(int entry) => _clock - _hzTime[entry];

    // The wait states item k needs before it, from every case that applies (Touch(k) has listed its registers).
    private int HazardNeed(int k)
    {
        ref var item = ref _items[k];
        var ins = item.Ins;
        bool vmem = ins.Format is GcnFormat.Global or GcnFormat.Flat;
        int need = 0, states;

        // A VALU writes an SGPR (VCC, EXEC), then vector memory reads it: its scalar address, or EXEC.
        if (vmem && (states = _hazardStates[(int)GcnHazard.VmemSgprAfterValu]) > 0)
        {
            for (int n = 0; n < _readCount; n++)
            {
                need = _reads[n] < VgprBase ? Math.Max(need, states - Since(ValuSgpr + _reads[n])) : need;
            }
        }

        // A VALU writes an SGPR, then v_readlane or v_writelane selects its lane with it (operand 2).
        if (ins.Op is GcnOp.VReadlaneB32 or GcnOp.VWritelaneB32 && (states = _hazardStates[(int)GcnHazard.LaneSelectAfterValu]) > 0
            && RegisterCodes(Physical(_itemOps[item.First + 2]), out int lane, out int laneCount) && lane < VgprBase)
        {
            for (int r = lane; r < lane + laneCount; r++)
            {
                need = Math.Max(need, states - Since(ValuSgpr + r));
            }
        }

        // A VALU writes VCC, then v_div_fmas reads it.
        if (ins.Has(GcnInstructionFlags.ReadsVcc) && (states = _hazardStates[(int)GcnHazard.DivFmasVccAfterValu]) > 0)
        {
            int vcc = _set.Special(GcnSpecial.Vcc);
            need = Math.Max(need, Math.Max(states - Since(ValuSgpr + vcc), states - Since(ValuSgpr + vcc + 1)));
        }

        if (item.Encoded == GcnVariant.Dpp)
        {
            // A VALU writes a VGPR, then DPP reads it (a source, or the destination whose lanes it may keep).
            if ((states = _hazardStates[(int)GcnHazard.DppVgprAfterValu]) > 0)
            {
                for (int n = 0; n < _readCount; n++)
                {
                    need = _reads[n] >= VgprBase ? Math.Max(need, states - Since(ValuVgpr + _reads[n] - VgprBase)) : need;
                }

                for (int n = 0; n < _writeCount; n++)
                {
                    need = _writes[n] >= VgprBase ? Math.Max(need, states - Since(ValuVgpr + _writes[n] - VgprBase)) : need;
                }
            }

            // A VALU writes EXEC (v_cmpx), then DPP.
            if ((states = _hazardStates[(int)GcnHazard.DppAfterValuExec]) > 0)
            {
                int exec = _set.Special(GcnSpecial.Exec);
                need = Math.Max(need, Math.Max(states - Since(ValuSgpr + exec), states - Since(ValuSgpr + exec + 1)));
            }
        }

        // s_setreg writes a hardware register, then s_getreg reads it or s_setreg writes it again.
        if (ins.Op is GcnOp.SGetregB32 or GcnOp.SSetregB32 or GcnOp.SSetregImm32B32)
        {
            states = _hazardStates[(int)(ins.Op == GcnOp.SGetregB32 ? GcnHazard.GetregAfterSetreg : GcnHazard.SetregAfterSetreg)];
            need = states > 0 ? Math.Max(need, states - Since(Setreg + HardwareRegisterId(item))) : need;
        }

        // A wide FLAT or GLOBAL store, then a VALU writes one of its data VGPRs.
        if (IsVector(ins.Format) && (states = _hazardStates[(int)GcnHazard.WideStoreDataAfterValu]) > 0)
        {
            for (int n = 0; n < _writeCount; n++)
            {
                need = _writes[n] >= VgprBase ? Math.Max(need, states - Since(StoreData + _writes[n] - VgprBase)) : need;
            }
        }

        // An SALU writes M0, then a GDS instruction reads it.
        if (ins.Format == GcnFormat.Ds && item.M.Gds && (states = _hazardStates[(int)GcnHazard.M0AfterSalu]) > 0)
        {
            need = Math.Max(need, states - Since(SaluSgpr + _set.Special(GcnSpecial.M0)));
        }

        // An SALU writes an SGPR, then SMEM reads it (none on gfx9).
        if (ins.Format == GcnFormat.Smem && (states = _hazardStates[(int)GcnHazard.SmemSgprAfterSalu]) > 0)
        {
            for (int n = 0; n < _readCount; n++)
            {
                need = _reads[n] < VgprBase ? Math.Max(need, states - Since(SaluSgpr + _reads[n])) : need;
            }
        }

        // A soft clause (with xnack): a store after a member that wrote registers, or a member's write meeting a member's
        // read, breaks it.
        if ((ins.Format == GcnFormat.Smem || vmem) && (states = _hazardStates[(int)GcnHazard.SoftClause]) > need
            && (_clauseKind & (ins.Format == GcnFormat.Smem ? 1 : 2)) != 0 && Any(_clauseDefs) && (IsStore(ins) || ClauseMeets()))
        {
            need = states;
        }

        return need;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The fixes that are instructions (ISA 10.1 and 11), as LLVM 18's GCNHazardRecognizer writes them.
    // ---------------------------------------------------------------------------------------------------------------

    private void SetHazardFixes()
    {
        _fixes = _set.HazardFixes(_isa);
        if (_fixes != GcnHazardFixes.None)
        {
            _nullCode = _set.Special(GcnSpecial.Null);
            _execCode = _set.Special(GcnSpecial.Exec);
            _m0Code = _set.Special(GcnSpecial.M0);
            _vccCode = _set.Special(GcnSpecial.Vcc);
        }
    }

    // Before item k (Touch(k) has listed its registers): the fixes it needs, as rows before it when emitting.
    private void HazardFixBefore(int k, bool emit)
    {
        if (_fixes == GcnHazardFixes.None)
        {
            return;
        }

        var format = _items[k].Ins.Format;
        bool valu = IsVector(format);

        // Vector memory read an SGPR, then an SALU or SMEM writes it (ISA 10.1).
        if ((_fixes & GcnHazardFixes.VmemSgprThenScalarWrite) != 0 && (IsScalarAlu(format) || format == GcnFormat.Smem) && WritesMasked(0))
        {
            InsertDepctr(k, DepctrVmVsrc, emit);
        }

        // SMEM read an SGPR, then a VALU writes it (ISA 10.1): s_mov_b32 null, 0, an SALU, clears it.
        if ((_fixes & GcnHazardFixes.SmemSgprThenValuWrite) != 0 && valu && WritesMasked(2))
        {
            _fixMasks[2] = _fixMasks[3] = 0;
            HazardAdvance(1);
            if (emit)
            {
                AddRow(k, RowNullMov, 0);
            }
        }

        // A transcendental wrote a VGPR this VALU reads, too few VALUs ago (ISA 11).
        if ((_fixes & GcnHazardFixes.TransResultThenValu) != 0 && valu && ReadsTransResult())
        {
            InsertDepctr(k, DepctrVaVdst, emit);
        }
    }

    // After item k ran (and was recorded): what it does to the fixes' state, and the fix that goes right after it (an SALU
    // writing an SGPR a VALU read as its lane mask, ISA 11).
    private void HazardFixAfter(int k, bool emit)
    {
        if (_fixes == GcnHazardFixes.None)
        {
            return;
        }

        ref var item = ref _items[k];
        var ins = item.Ins;
        var format = ins.Format;
        if (ins.Op == GcnOp.SWaitcnt)
        {
            var (vm, exp, lgkm) = _set.WaitCounts((ushort)_itemOps[item.First].Value);
            HazardWaited(vm, exp, lgkm);
        }
        else if (ins.Op == GcnOp.SWaitcntDepctr)
        {
            Depctr((ushort)_itemOps[item.First].Value);
        }
        else if (format is GcnFormat.Global or GcnFormat.Flat or GcnFormat.Ds or GcnFormat.Mubuf)
        {
            MaskReads(0);
            ClearTrans();
        }
        else if (format == GcnFormat.Smem)
        {
            MaskReads(2);
        }
        else if (IsVector(format))
        {
            _fixMasks[0] = _fixMasks[1] = 0;
            _valuClock++;
            if (IsTranscendental(ins.Op))
            {
                _transClock++;
                for (int n = 0; n < _writeCount; n++)
                {
                    if (_writes[n] >= VgprBase)
                    {
                        _transValu[_writes[n] - VgprBase] = _valuClock;
                        _transTrans[_writes[n] - VgprBase] = _transClock;
                    }
                }
            }

            LaneMaskReads(k);
        }
        else if (IsScalarAlu(format))
        {
            if (ins.Op != GcnOp.SWaitcntVscnt)
            {
                _fixMasks[2] = _fixMasks[3] = 0;
            }

            int d = ins.Operands.IndexOf(GcnRole.ScalarDestination);
            if ((_fixes & GcnHazardFixes.LaneMaskThenSaluWrite) != 0 && d >= 0
                && RegisterCodes(Physical(_itemOps[item.First + d]), out int code, out int count))
            {
                for (int r = code; r < code + count; r++)
                {
                    if (r < 128 && r != _execCode && r != _execCode + 1 && r != _m0Code && r != _nullCode && (_fixMasks[4 + (r >> 6)] >> (r & 63) & 1) != 0)
                    {
                        InsertDepctr(k, DepctrSaSdst, emit);
                        break;
                    }
                }
            }
        }
    }

    // A wait for these counts (the text's, or one placed): s_waitcnt 0 clears vector memory's reads, lgkmcnt(0) SMEM's.
    private void HazardWaited(int vm, int exp, int lgkm)
    {
        if (_fixes == GcnHazardFixes.None)
        {
            return;
        }

        if (vm == 0 && exp == 0 && lgkm == 0)
        {
            _fixMasks[0] = _fixMasks[1] = 0;
        }

        if (lgkm == 0)
        {
            _fixMasks[2] = _fixMasks[3] = 0;
        }
    }

    private void InsertDepctr(int k, ushort bits, bool emit)
    {
        Depctr(bits);
        HazardAdvance(1);
        if (emit)
        {
            AddRow(k, RowDepctr, bits);
        }
    }

    // s_waitcnt_depctr: each counter it waits to 0 clears what it guards.
    private void Depctr(ushort bits)
    {
        if ((bits >> 2 & 7) == 0)
        {
            _fixMasks[0] = _fixMasks[1] = 0;
        }

        if ((bits >> 12 & 0xF) == 0)
        {
            ClearTrans();
        }

        if ((bits & 1) == 0)
        {
            _fixMasks[4] = _fixMasks[5] = 0;
        }
    }

    // Whether the instruction writes a scalar register (not null) of mask pair `at`.
    private bool WritesMasked(int at)
    {
        for (int n = 0; n < _writeCount; n++)
        {
            int r = _writes[n];
            if (r < 128 && r != _nullCode && (_fixMasks[at + (r >> 6)] >> (r & 63) & 1) != 0)
            {
                return true;
            }
        }

        return false;
    }

    // The scalar registers the instruction reads (EXEC included, not null) join mask pair `at`.
    private void MaskReads(int at)
    {
        for (int n = 0; n < _readCount; n++)
        {
            int r = _reads[n];
            if (r < 128 && r != _nullCode)
            {
                _fixMasks[at + (r >> 6)] |= 1UL << (r & 63);
            }
        }
    }

    // A VALU that reads an SGPR (VCC included, not EXEC) or a literal ends what the lane masks read before it; the masks
    // it reads itself (v_cndmask's, v_div_fmas's VCC) are then the ones a later SALU write must wait for.
    private void LaneMaskReads(int k)
    {
        ref var item = ref _items[k];
        bool scalar = false;
        for (int n = 0; n < _readCount && !scalar; n++)
        {
            int r = _reads[n];
            scalar = r < 128 && r != _execCode && r != _execCode + 1 && r != _nullCode;
        }

        for (int i = 0; i < item.Count && !scalar; i++)
        {
            scalar = _itemOps[item.First + i].Kind == GcnOperandKind.Literal;
        }

        if (!scalar)
        {
            return;
        }

        _fixMasks[4] = _fixMasks[5] = 0;
        int code, count;
        int at = item.Ins.Operands.IndexOf(GcnRole.LaneMaskSource);
        if (at >= 0)
        {
            if (!RegisterCodes(Physical(_itemOps[item.First + at]), out code, out count))
            {
                return;
            }
        }
        else if (item.Ins.Op is GcnOp.VDivFmasF32 or GcnOp.VDivFmasF64)
        {
            (code, count) = (_vccCode, 2);
        }
        else
        {
            return;
        }

        for (int r = code; r < code + count; r++)
        {
            if (r < 128)
            {
                _fixMasks[4 + (r >> 6)] |= 1UL << (r & 63);
            }
        }
    }

    // Whether a VGPR the VALU reads was written by a transcendental fewer than 6 VALUs and 2 transcendentals ago.
    private bool ReadsTransResult()
    {
        for (int n = 0; n < _readCount; n++)
        {
            int v = _reads[n] - VgprBase;
            if (v >= 0 && _valuClock - _transValu[v] < TransValus && _transClock - _transTrans[v] < TransTrans)
            {
                return true;
            }
        }

        return false;
    }

    private void ClearTrans()
    {
        _transValu.AsSpan().Fill(_valuClock - TransValus);
        _transTrans.AsSpan().Fill(_transClock - TransTrans);
    }

    // LLVM's transcendentals (isTRANS): exp, log, rcp, rsq, sqrt, sin and cos.
    private static bool IsTranscendental(GcnOp op) => op is GcnOp.VExpF32 or GcnOp.VLogF32 or GcnOp.VRcpF32 or GcnOp.VRcpIflagF32
        or GcnOp.VRsqF32 or GcnOp.VSqrtF32 or GcnOp.VSinF32 or GcnOp.VCosF32 or GcnOp.VExpF16 or GcnOp.VLogF16 or GcnOp.VRcpF16
        or GcnOp.VRsqF16 or GcnOp.VSqrtF16 or GcnOp.VRcpF64 or GcnOp.VRsqF64 or GcnOp.VSqrtF64;

    // s_getreg's and s_setreg's hardware register id (hwreg's bits 5:0).
    private int HardwareRegisterId(in Item item)
    {
        int at = item.Ins.Operands.IndexOf(GcnRole.HardwareRegister);
        return (int)(_itemOps[item.First + at].Value & 63);
    }

    // Whether the clause with this instruction in it has a write meeting a read (the instruction's included).
    private bool ClauseMeets()
    {
        _xDefs.AsSpan().Clear();
        _xUses.AsSpan().Clear();
        for (int n = 0; n < _writeCount; n++)
        {
            _xDefs[_writes[n] >> 6] |= 1UL << (_writes[n] & 63);
        }

        for (int n = 0; n < _readCount; n++)
        {
            _xUses[_reads[n] >> 6] |= 1UL << (_reads[n] & 63);
        }

        for (int w = 0; w < 6; w++)
        {
            if (((_clauseDefs[w] | _xDefs[w]) & (_clauseUses[w] | _xUses[w])) != 0)
            {
                return true;
            }
        }

        return false;
    }

    private static bool Any(ulong[] bits) => (bits[0] | bits[1] | bits[2] | bits[3] | bits[4] | bits[5]) != 0;

    // Wait states inserted (a wait, or s_nop): the clock moves, and a clause ends.
    private void HazardAdvance(int states)
    {
        _clock += states;
        ResetClause();
    }

    private void ResetClause()
    {
        _clauseKind = 0;
        _clauseDefs.AsSpan().Clear();
        _clauseUses.AsSpan().Clear();
    }

    // Item k ran: the clock moves by its wait states, its writes are noted by kind, the clause takes it or ends.
    private void HazardRecord(int k)
    {
        ref var item = ref _items[k];
        var ins = item.Ins;
        _clock += ins.Op == GcnOp.SNop ? (int)(_itemOps[item.First].Value & 0xF) + 1 : 1;
        if (IsVector(ins.Format))
        {
            for (int n = 0; n < _writeCount; n++)
            {
                _hzTime[_writes[n] < VgprBase ? ValuSgpr + _writes[n] : ValuVgpr + _writes[n] - VgprBase] = _clock;
            }
        }
        else if (IsScalarAlu(ins.Format))
        {
            for (int n = 0; n < _writeCount; n++)
            {
                if (_writes[n] < VgprBase)
                {
                    _hzTime[SaluSgpr + _writes[n]] = _clock;
                }
            }
        }

        if (ins.Op is GcnOp.SSetregB32 or GcnOp.SSetregImm32B32)
        {
            _hzTime[Setreg + HardwareRegisterId(item)] = _clock;
        }

        if (ins.Format is GcnFormat.Global or GcnFormat.Flat && IsStore(ins))
        {
            int data = ins.Operands.IndexOf(GcnRole.Data0);
            if (data >= 0 && RegisterCodes(Physical(_itemOps[item.First + data]), out int code, out int count) && count > 2)
            {
                for (int r = code; r < code + count; r++)
                {
                    _hzTime[StoreData + r - VgprBase] = _clock;
                }
            }
        }

        int kind = ins.Format == GcnFormat.Smem ? 1 : ins.Format is GcnFormat.Global or GcnFormat.Flat ? 2 : 0;
        if ((_clauseKind & kind) == 0)
        {
            ResetClause();
        }

        if (kind == 0)
        {
            return;
        }

        _clauseKind = kind;
        for (int n = 0; n < _writeCount; n++)
        {
            _clauseDefs[_writes[n] >> 6] |= 1UL << (_writes[n] & 63);
        }

        for (int n = 0; n < _readCount; n++)
        {
            _clauseUses[_reads[n] >> 6] |= 1UL << (_reads[n] & 63);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The hazards' part of a block's start state: wait states since each write (Far: none near), the clause.
    // ---------------------------------------------------------------------------------------------------------------

    private static void InitialHazards(Span<byte> state)
    {
        state[..HazardEntries].Fill(Far);
        state[HazardEntries..TransAt].Clear();
        state.Slice(TransAt, 256).Fill(TransValus);
        state.Slice(TransAt + 256, 256).Fill(TransTrans);
    }

    private void SaveHazards(Span<byte> state)
    {
        for (int e = 0; e < HazardEntries; e++)
        {
            state[e] = (byte)Math.Min(Since(e), Far);
        }

        state[ClauseKindAt] = (byte)_clauseKind;
        MemoryMarshal.AsBytes(_clauseDefs.AsSpan()).CopyTo(state[ClauseDefsAt..]);
        MemoryMarshal.AsBytes(_clauseUses.AsSpan()).CopyTo(state[ClauseUsesAt..]);
        MemoryMarshal.AsBytes(_fixMasks.AsSpan()).CopyTo(state[FixMasksAt..]);
        for (int v = 0; v < 256; v++)
        {
            state[TransAt + v] = (byte)Math.Min(_valuClock - _transValu[v], TransValus);
            state[TransAt + 256 + v] = (byte)Math.Min(_transClock - _transTrans[v], TransTrans);
        }
    }

    private void LoadHazards(ReadOnlySpan<byte> state)
    {
        _clock = 2 * Far;
        for (int e = 0; e < HazardEntries; e++)
        {
            _hzTime[e] = _clock - state[e];
        }

        _clauseKind = state[ClauseKindAt];
        state.Slice(ClauseDefsAt, 48).CopyTo(MemoryMarshal.AsBytes(_clauseDefs.AsSpan()));
        state.Slice(ClauseUsesAt, 48).CopyTo(MemoryMarshal.AsBytes(_clauseUses.AsSpan()));
        state.Slice(FixMasksAt, 48).CopyTo(MemoryMarshal.AsBytes(_fixMasks.AsSpan()));
        _valuClock = _transClock = 2 * Far;
        for (int v = 0; v < 256; v++)
        {
            _transValu[v] = _valuClock - state[TransAt + v];
            _transTrans[v] = _transClock - state[TransAt + 256 + v];
        }
    }

    private static bool MergeHazards(Span<byte> into, ReadOnlySpan<byte> from)
    {
        bool changed = false;
        for (int e = 0; e < HazardEntries; e++)
        {
            if (from[e] < into[e])
            {
                into[e] = from[e];
                changed = true;
            }
        }

        for (int i = ClauseKindAt; i < TransAt; i++)
        {
            byte joined = (byte)(into[i] | from[i]);
            changed |= joined != into[i];
            into[i] = joined;
        }

        for (int i = TransAt; i < HazardBytes; i++)
        {
            if (from[i] < into[i])
            {
                into[i] = from[i];
                changed = true;
            }
        }

        return changed;
    }
}
