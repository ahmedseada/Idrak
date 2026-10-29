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
internal sealed partial class GcnAssembler
{
    // Where each kind of write is kept: VALU writes of scalar codes 0-127 and VGPRs, SALU writes of scalar codes, s_setreg
    // by hardware register id, wide stores' data VGPRs; then the clause's kind (1 SMEM, 2 vector memory) and the
    // registers it wrote and read (384 bits each).
    private const int ValuSgpr = 0, ValuVgpr = 128, SaluSgpr = 384, Setreg = 512, StoreData = 576, HazardEntries = 832;
    private const int ClauseKindAt = HazardEntries, ClauseDefsAt = HazardEntries + 1, ClauseUsesAt = ClauseDefsAt + 48, HazardBytes = 936;

    // Wait states past the longest any case needs: a write this far back is no hazard.
    private const int Far = 16;

    // The most wait states one s_nop gives (s_nop 7), as LLVM inserts them.
    private const int MaxNop = 8;

    private readonly int[] _hzTime = new int[HazardEntries];
    private readonly int[] _hazardStates = new int[(int)GcnHazard.SoftClause + 1];
    private readonly ulong[] _clauseDefs = new ulong[6], _clauseUses = new ulong[6], _xDefs = new ulong[6], _xUses = new ulong[6];
    private int _clock, _clauseKind;

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
        state[HazardEntries..HazardBytes].Clear();
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

        for (int i = ClauseKindAt; i < ClauseUsesAt + 48; i++)
        {
            byte joined = (byte)(into[i] | from[i]);
            changed |= joined != into[i];
            into[i] = joined;
        }

        return changed;
    }
}
