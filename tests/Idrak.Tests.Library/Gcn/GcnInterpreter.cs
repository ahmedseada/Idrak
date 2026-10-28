// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Numerics;

namespace Idrak.Tests.Gcn;

// The GCN interpreter (plan 35, G1b): runs gfx9 machine code on the CPU, so the GCN kernels Idrak writes are checked
// here against the CPU backend, where no AMD GPU is. Waves of 64 lanes with EXEC, VCC, SCC, M0, SGPRs and VGPRs up to
// the descriptor's counts; LDS per workgroup; s_barrier across a workgroup's waves (run in lockstep, in a seeded order);
// global memory as a flat byte space with storages mapped at addresses. Loads and LDS reads complete only when an
// s_waitcnt retires them: an instruction that reads (or overwrites) a register a load has not been waited for is
// reported as a missing wait, with both offsets, so wrong code fails loudly and never passes by luck. Every access out
// of a mapped storage or the workgroup's LDS is reported; so are races a barrier or a wait would have prevented.

/// <summary>What went wrong in a run: the kind tests check for.</summary>
internal enum GcnFailure
{
    /// <summary>An instruction read or overwrote a register a load (or LDS read) still writes: no s_waitcnt retired it.</summary>
    MissingWait,

    /// <summary>A memory access outside every mapped storage, the kernel arguments or the workgroup's LDS.</summary>
    OutOfRange,

    /// <summary>Two waves (or workgroups) touched the same bytes with no barrier, or no wait, between them.</summary>
    Race,

    /// <summary>A register past the descriptor's counts, an odd register pair, or a register the interpreter does not model.</summary>
    Register,

    /// <summary>An instruction or setting the interpreter does not model.</summary>
    NotModelled,

    /// <summary>The words at the program counter are not an instruction the decoder knows.</summary>
    Decode,

    /// <summary>A wave ran past its instruction budget or off the end of the code.</summary>
    Hang,
}

/// <summary>A run that went wrong: the kind, and a message naming the instruction (offset and text), the workgroup and the wave.</summary>
internal sealed class GcnInterpreterException(GcnFailure kind, int offset, string message) : Exception(message)
{
    public GcnFailure Kind => kind;

    /// <summary>The offset in the code of the instruction that failed, or -1.</summary>
    public int Offset => offset;
}

/// <summary>How a kernel is dispatched in the interpreter.</summary>
internal sealed record GcnDispatch
{
    /// <summary>Workgroups in x, y and z.</summary>
    public (int X, int Y, int Z) Workgroups { get; init; } = (1, 1, 1);

    /// <summary>Work-items in a workgroup, in x, y and z.</summary>
    public (int X, int Y, int Z) WorkgroupSize { get; init; } = (64, 1, 1);

    /// <summary>The kernel arguments, exactly the descriptor's kernarg_size bytes.</summary>
    public byte[] Arguments { get; init; } = [];

    /// <summary>
    /// Seeds the order waves of a workgroup take their turns in, and the values registers and LDS hold before the
    /// kernel writes them (so code that reads what it never wrote gives different results under different seeds).
    /// </summary>
    public ulong Seed { get; init; } = 1;

    /// <summary>The most instructions one wave may run before the run is reported as hung.</summary>
    public long MaxInstructionsPerWave { get; init; } = 50_000_000;

    /// <summary>Whether races between waves and workgroups are tracked (on by default).</summary>
    public bool TrackRaces { get; init; } = true;
}

/// <summary>What a run did.</summary>
internal readonly record struct GcnRunResult(long Instructions, int Waves, int Workgroups);

/// <summary>A pending memory operation: counted by vmcnt, lgkmcnt or both until a wait (or the wave's end) retires it.</summary>
internal sealed class GcnPending
{
    public required GcnInstruction Instruction { get; init; }

    /// <summary>Counted by vmcnt (VMEM and FLAT).</summary>
    public bool Vm { get; init; }

    /// <summary>Counted by lgkmcnt (LDS, SMEM, FLAT).</summary>
    public bool Lgkm { get; init; }

    /// <summary>On lgkmcnt, completes in order with the other in-order entries (LDS); SMEM and FLAT complete in any order.</summary>
    public bool InOrder { get; init; }

    /// <summary>The first vector (or accumulation) register written, and how many; -1 when none.</summary>
    public int VFirst { get; init; } = -1;

    public int VCount { get; init; }

    /// <summary>The values for VCount registers × 64 lanes, written for the lanes in Lanes.</summary>
    public uint[]? VValues { get; init; }

    public ulong Lanes { get; init; }

    /// <summary>0 the whole register; 1 the low 16 bits (the high kept); 2 the high 16 bits (the low kept).</summary>
    public int Merge { get; init; }

    public int SFirst { get; init; } = -1;

    public int SCount { get; init; }

    public uint[]? SValues { get; init; }

    /// <summary>The store this entry is, when it is one.</summary>
    public GcnStoreRecord? Store { get; init; }
}

/// <summary>A store's place in the race tracking: who wrote, in which barrier interval, and whether a wait retired it.</summary>
internal sealed class GcnStoreRecord
{
    public int Group { get; init; }

    public int Wave { get; init; }

    public int Epoch { get; init; }

    public bool Retired { get; set; }

    public GcnInstruction? Instruction { get; init; }
}

/// <summary>One wave: 64 lanes, its registers and its pending memory operations.</summary>
internal sealed class GcnWave
{
    internal GcnWave(int index, int vgprs, bool acc)
    {
        Index = index;
        V = new uint[vgprs * 64];
        VOwner = new GcnPending?[vgprs];
        Acc = acc ? new uint[vgprs * 64] : [];
        AOwner = new GcnPending?[acc ? vgprs : 0];
    }

    public int Index { get; }

    /// <summary>The scalar registers by their encoding: s0-s101, flat_scratch, xnack_mask, VCC (106, 107), ttmp, M0 (124), EXEC (126, 127).</summary>
    internal readonly uint[] S = new uint[128];

    /// <summary>The vector registers, register-major: V[r * 64 + lane].</summary>
    internal readonly uint[] V;

    internal readonly uint[] Acc;

    internal readonly GcnPending?[] SOwner = new GcnPending?[128];

    internal readonly GcnPending?[] VOwner;

    internal readonly GcnPending?[] AOwner;

    internal readonly List<GcnPending> VmQueue = [];

    internal readonly List<GcnPending> LgkmQueue = [];

    internal bool Scc;

    internal int Pc;

    internal int Next;

    internal bool Ended;

    internal bool AtBarrier;

    internal long Executed;

    internal ulong Exec
    {
        get => S[126] | (ulong)S[127] << 32;
        set
        {
            S[126] = (uint)value;
            S[127] = (uint)(value >> 32);
        }
    }

    internal ulong Vcc
    {
        get => S[106] | (ulong)S[107] << 32;
        set
        {
            S[106] = (uint)value;
            S[107] = (uint)(value >> 32);
        }
    }
}

/// <summary>Runs a kernel image over a grid of workgroups against a <see cref="GcnMemory"/>.</summary>
internal static class GcnInterpreter
{
    /// <summary>
    /// Runs <paramref name="kernel"/> over <paramref name="dispatch"/>'s grid. Global memory is <paramref name="memory"/>;
    /// the kernel arguments are mapped for the run. Workgroups run one after another (x fastest), each workgroup's
    /// waves in lockstep up to its barriers.
    /// </summary>
    /// <exception cref="GcnInterpreterException">The kernel misbehaved (a missing wait, an access out of range, a race) or did what the interpreter does not model.</exception>
    public static GcnRunResult Run(GcnKernelImage kernel, GcnMemory memory, GcnDispatch dispatch) => new GcnMachine(kernel, memory, dispatch).Run();

    /// <summary>Runs kernel <paramref name="name"/> of a code object.</summary>
    public static GcnRunResult Run(byte[] codeObject, string name, GcnMemory memory, GcnDispatch dispatch) =>
        Run(GcnCodeObjectReader.Read(codeObject, name), memory, dispatch);
}

/// <summary>One dispatch's machine state: the current workgroup's waves and LDS, the decoded code, the scratch lanes.</summary>
internal sealed partial class GcnMachine
{
    private readonly GcnKernelImage kernel;
    private readonly GcnMemory memory;
    private readonly GcnDispatch dispatch;
    private readonly GcnDecoder decoder;
    private readonly GcnInstruction?[] decoded;
    private readonly GcnKernelDescriptor descriptor;
    private readonly List<GcnWave> wavePool = [];
    private ulong random;
    private long instructions;

    internal readonly List<GcnStoreRecord> Stores = [new GcnStoreRecord()];

    internal int Group;

    internal (int X, int Y, int Z) GroupId;

    internal int Epoch;

    internal GcnWave[] Waves = [];

    internal byte[] Lds = [];

    internal int[] LdsWriter = [];

    internal long[] LdsReader = [];

    internal int LdsTouched;

    internal ulong KernargAddress;

    internal ulong PacketAddress;

    internal readonly uint[] L0 = new uint[64], L1 = new uint[64], L2 = new uint[64], L3 = new uint[64], L4 = new uint[64], L5 = new uint[64], R0 = new uint[64], R1 = new uint[64];

    internal GcnMachine(GcnKernelImage kernel, GcnMemory memory, GcnDispatch dispatch)
    {
        this.kernel = kernel;
        this.memory = memory;
        this.dispatch = dispatch;
        descriptor = kernel.Descriptor;
        decoder = new GcnDecoder(kernel.Cpu);
        decoded = new GcnInstruction?[(kernel.Code.Length + 3) / 4];
        random = dispatch.Seed * 0x9E3779B97F4A7C15UL + 0x632BE59BD9B4E019UL;
    }

    internal GcnKernelDescriptor Descriptor => descriptor;

    internal GcnKernelImage Kernel => kernel;

    internal GcnMemory Memory => memory;

    internal bool TrackRaces => dispatch.TrackRaces;

    /// <summary>The next value of the run's seeded generator (SplitMix64).</summary>
    internal ulong NextRandom()
    {
        ulong z = random += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal GcnRunResult Run()
    {
        Validate();
        var (gx, gy, gz) = dispatch.Workgroups;
        var (bx, by, bz) = dispatch.WorkgroupSize;
        byte[] arguments = new byte[Math.Max(descriptor.KernargBytes, 0)];
        dispatch.Arguments.CopyTo(arguments, 0);
        KernargAddress = descriptor.KernargBytes > 0 ? memory.Map("the kernel arguments", arguments) : 0;
        if (descriptor.DispatchPtr)
        {
            var packet = new byte[64];
            BinaryPrimitives.WriteUInt16LittleEndian(packet.AsSpan(4), (ushort)bx);
            BinaryPrimitives.WriteUInt16LittleEndian(packet.AsSpan(6), (ushort)by);
            BinaryPrimitives.WriteUInt16LittleEndian(packet.AsSpan(8), (ushort)bz);
            BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(12), (uint)(gx * bx));
            BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(16), (uint)(gy * by));
            BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(20), (uint)(gz * bz));
            BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(28), (uint)descriptor.LdsBytes);
            BinaryPrimitives.WriteUInt64LittleEndian(packet.AsSpan(40), KernargAddress);
            PacketAddress = memory.Map("the dispatch packet", packet);
        }

        memory.BeginDispatch();
        int threads = bx * by * bz;
        int waveCount = (threads + 63) / 64;
        try
        {
            for (int z = 0; z < gz; z++)
            {
                for (int y = 0; y < gy; y++)
                {
                    for (int x = 0; x < gx; x++)
                    {
                        GroupId = (x, y, z);
                        RunGroup(waveCount);
                        Group++;
                    }
                }
            }
        }
        finally
        {
            if (KernargAddress != 0)
            {
                memory.Unmap(KernargAddress);
            }

            if (PacketAddress != 0)
            {
                memory.Unmap(PacketAddress);
            }
        }

        return new GcnRunResult(instructions, Group * waveCount, Group);
    }

    private void Validate()
    {
        var (bx, by, bz) = dispatch.WorkgroupSize;
        var (gx, gy, gz) = dispatch.Workgroups;
        string? why = null;
        if (bx < 1 || by < 1 || bz < 1 || bx * by * bz > 1024)
        {
            why = $"a workgroup of {bx} x {by} x {bz} work-items (1 to 1,024 on gfx9)";
        }
        else if (gx < 1 || gy < 1 || gz < 1)
        {
            why = $"a grid of {gx} x {gy} x {gz} workgroups";
        }
        else if (dispatch.Arguments.Length != descriptor.KernargBytes)
        {
            why = $"{dispatch.Arguments.Length} bytes of kernel arguments where the descriptor's kernarg_size is {descriptor.KernargBytes}";
        }
        else if (descriptor.PrivateBytes > 0 || descriptor.PrivateSegmentBuffer || descriptor.FlatScratchInit || descriptor.PrivateSegmentSize || descriptor.PrivateSegmentWaveOffset)
        {
            why = "scratch (private segment) memory: the interpreter models none";
        }
        else if (descriptor.QueuePtr || descriptor.WorkgroupInfo)
        {
            why = "the queue pointer and the workgroup info SGPRs: the interpreter models neither";
        }
        else if (descriptor.LdsBytes > 65536)
        {
            why = $"{descriptor.LdsBytes} bytes of LDS (gfx9 has 65,536 a workgroup)";
        }
        else if (descriptor.WorkitemIds == 3)
        {
            why = "work-item ids mode 3 (reserved)";
        }

        if (why is not null)
        {
            throw new GcnInterpreterException(GcnFailure.NotModelled, -1, $"'{kernel.Name}': {why}");
        }
    }

    private void RunGroup(int waveCount)
    {
        var (bx, by, _) = dispatch.WorkgroupSize;
        int threads = bx * by * dispatch.WorkgroupSize.Z;
        while (wavePool.Count < waveCount)
        {
            wavePool.Add(new GcnWave(wavePool.Count, descriptor.Vgprs, kernel.Cpu == GcnCpu.Gfx908));
        }

        if (Waves.Length != waveCount)
        {
            Waves = wavePool.Take(waveCount).ToArray();
        }

        Epoch = 0;
        if (Lds.Length != descriptor.LdsBytes)
        {
            Lds = new byte[descriptor.LdsBytes];
            LdsWriter = new int[descriptor.LdsBytes];
            LdsReader = new long[descriptor.LdsBytes];
        }

        Poison(Lds);
        if (LdsTouched > 0)
        {
            Array.Clear(LdsWriter, 0, LdsTouched);
            Array.Clear(LdsReader, 0, LdsTouched);
            LdsTouched = 0;
        }

        foreach (var wave in Waves)
        {
            StartWave(wave, threads, bx, by);
        }

        int count = Waves.Length;
        while (true)
        {
            bool ran = false;
            int start = count == 1 ? 0 : (int)(NextRandom() % (ulong)count);
            for (int k = 0; k < count; k++)
            {
                var wave = Waves[(start + k) % count];
                if (!wave.Ended && !wave.AtBarrier)
                {
                    Step(wave);
                    ran = true;
                }
            }

            if (ran)
            {
                continue;
            }

            bool waiting = false;
            foreach (var wave in Waves)
            {
                if (wave.AtBarrier)
                {
                    wave.AtBarrier = false;
                    waiting = true;
                }
            }

            if (!waiting)
            {
                break;
            }

            Epoch++;
        }
    }

    private void StartWave(GcnWave wave, int threads, int bx, int by)
    {
        Poison(wave.V);
        Poison(wave.Acc);
        for (int r = 0; r < 128; r++)
        {
            wave.S[r] = (uint)NextRandom();
        }

        Array.Clear(wave.SOwner);
        Array.Clear(wave.VOwner);
        Array.Clear(wave.AOwner);
        wave.VmQueue.Clear();
        wave.LgkmQueue.Clear();
        wave.Scc = (NextRandom() & 1) != 0;
        wave.Pc = kernel.Entry;
        wave.Ended = false;
        wave.AtBarrier = false;
        wave.Executed = 0;
        int first = wave.Index * 64;
        int lanes = Math.Min(64, threads - first);
        wave.Exec = lanes == 64 ? ulong.MaxValue : (1UL << lanes) - 1;

        // User SGPRs in gfx9's order, then the system SGPRs.
        int s = 0;
        if (descriptor.DispatchPtr)
        {
            SetPair(wave, ref s, PacketAddress);
        }

        if (descriptor.KernargSegmentPtr)
        {
            SetPair(wave, ref s, KernargAddress);
        }

        if (descriptor.DispatchId)
        {
            SetPair(wave, ref s, 0);
        }

        if (s != descriptor.UserSgprCount)
        {
            throw new GcnInterpreterException(GcnFailure.NotModelled, -1,
                $"'{kernel.Name}': the descriptor's enables set {s} user SGPRs, its USER_SGPR_COUNT says {descriptor.UserSgprCount}");
        }

        if (descriptor.WorkgroupIdX)
        {
            wave.S[s++] = (uint)GroupId.X;
        }

        if (descriptor.WorkgroupIdY)
        {
            wave.S[s++] = (uint)GroupId.Y;
        }

        if (descriptor.WorkgroupIdZ)
        {
            wave.S[s++] = (uint)GroupId.Z;
        }

        for (int lane = 0; lane < lanes; lane++)
        {
            int t = first + lane;
            wave.V[lane] = (uint)(t % bx);
            if (descriptor.WorkitemIds >= 1 && wave.V.Length > 64)
            {
                wave.V[64 + lane] = (uint)(t / bx % by);
            }

            if (descriptor.WorkitemIds >= 2 && wave.V.Length > 128)
            {
                wave.V[128 + lane] = (uint)(t / (bx * by));
            }
        }
    }

    private static void SetPair(GcnWave wave, ref int s, ulong value)
    {
        wave.S[s++] = (uint)value;
        wave.S[s++] = (uint)(value >> 32);
    }

    private void Poison(Span<uint> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (uint)NextRandom();
        }
    }

    private void Poison(Span<byte> bytes)
    {
        int i = 0;
        for (; i + 8 <= bytes.Length; i += 8)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(bytes[i..], NextRandom());
        }

        for (; i < bytes.Length; i++)
        {
            bytes[i] = (byte)NextRandom();
        }
    }

    private void Step(GcnWave wave)
    {
        if (++wave.Executed > dispatch.MaxInstructionsPerWave)
        {
            throw Fail(GcnFailure.Hang, wave, null, $"the wave ran {dispatch.MaxInstructionsPerWave:N0} instructions without ending (at 0x{wave.Pc:X4})");
        }

        var instruction = Fetch(wave);
        var execute = instruction.Opcode.Execute
                      ?? throw Fail(GcnFailure.NotModelled, wave, instruction, $"{instruction.Opcode.Base} is not modelled by the interpreter");
        if (instruction.Variant != GcnVariant.None && !GcnSemantics.TakesLaneForms(instruction.Opcode.Base))
        {
            throw Fail(GcnFailure.NotModelled, wave, instruction, $"the {instruction.Variant.ToString().ToUpperInvariant()} form of {instruction.Opcode.Base} is not modelled");
        }

        wave.Next = wave.Pc + instruction.Size;
        instructions++;
        execute(this, wave, instruction);
        wave.Pc = wave.Next;
    }

    private GcnInstruction Fetch(GcnWave wave)
    {
        int pc = wave.Pc;
        if (pc < 0 || pc >= kernel.Code.Length)
        {
            throw Fail(GcnFailure.Hang, wave, null, $"the wave ran off the code (to 0x{pc:X4} of {kernel.Code.Length} bytes) without s_endpgm");
        }

        if (decoded[pc / 4] is { } known)
        {
            return known;
        }

        try
        {
            return decoded[pc / 4] = decoder.Decode(kernel.Code, pc);
        }
        catch (GcnDecodeException e)
        {
            throw Fail(GcnFailure.Decode, wave, null, e.Message);
        }
    }

    /// <summary>An exception naming the failure, the instruction (offset and text), the workgroup and the wave.</summary>
    internal GcnInterpreterException Fail(GcnFailure kind, GcnWave? wave, GcnInstruction? instruction, string what)
    {
        string kindText = kind switch
        {
            GcnFailure.MissingWait => "missing s_waitcnt",
            GcnFailure.OutOfRange => "out of range",
            GcnFailure.Race => "race",
            GcnFailure.Register => "register",
            GcnFailure.NotModelled => "not modelled",
            GcnFailure.Decode => "not decoded",
            _ => "hang",
        };
        string where = instruction is null ? "" : $" at 0x{instruction.Offset:X4} '{instruction.Text}'";
        string who = wave is null ? "" : $", workgroup ({GroupId.X}, {GroupId.Y}, {GroupId.Z}) wave {wave.Index}";
        return new GcnInterpreterException(kind, instruction?.Offset ?? wave?.Pc ?? -1, $"'{kernel.Name}': {kindText}{where}{who}: {what}");
    }

    // ---- Registers ---------------------------------------------------------------------------------------------

    /// <summary>The name of scalar register <paramref name="r"/> for messages.</summary>
    internal static string ScalarName(int r) => r switch
    {
        < 102 => $"s{r}",
        106 => "vcc_lo",
        107 => "vcc_hi",
        124 => "m0",
        126 => "exec_lo",
        127 => "exec_hi",
        _ => GcnPrinter.Operand(r, 1, 's', null!, false),
    };

    private void CheckScalar(GcnWave wave, int r, GcnInstruction i)
    {
        if (r < kernel.UsableSgprs || r is 106 or 107 or 124 or 126 or 127)
        {
            return;
        }

        throw r < 102
            ? Fail(GcnFailure.Register, wave, i, $"s{r} is past the {kernel.UsableSgprs} SGPRs the descriptor allocates ({descriptor.SgprBlocks}, less VCC{(kernel.Xnack ? " and the xnack mask" : "")})")
            : Fail(GcnFailure.Register, wave, i, $"{ScalarName(r)} is not modelled by the interpreter");
    }

    /// <summary>Scalar register <paramref name="r"/> (by encoding), checked against the allocation and pending loads.</summary>
    internal uint ReadScalar(GcnWave wave, int r, GcnInstruction i)
    {
        CheckScalar(wave, r, i);
        if (wave.SOwner[r] is { } pending)
        {
            throw PendingRead(wave, i, ScalarName(r), pending);
        }

        return wave.S[r];
    }

    internal void WriteScalar(GcnWave wave, int r, uint value, GcnInstruction i)
    {
        CheckScalar(wave, r, i);
        if (wave.SOwner[r] is { } pending)
        {
            throw PendingWrite(wave, i, ScalarName(r), pending);
        }

        wave.S[r] = value;
    }

    internal void CheckPair(GcnWave wave, int r, GcnInstruction i)
    {
        if (r < 102 && (r & 1) != 0)
        {
            throw Fail(GcnFailure.Register, wave, i, $"s[{r}:{r + 1}]: a 64-bit scalar operand starts at an even register");
        }
    }

    internal ulong ReadScalar64(GcnWave wave, int r, GcnInstruction i)
    {
        CheckPair(wave, r, i);
        return ReadScalar(wave, r, i) | (ulong)ReadScalar(wave, r + 1, i) << 32;
    }

    internal void WriteScalar64(GcnWave wave, int r, ulong value, GcnInstruction i)
    {
        CheckPair(wave, r, i);
        WriteScalar(wave, r, (uint)value, i);
        WriteScalar(wave, r + 1, (uint)(value >> 32), i);
    }

    /// <summary>A 32-bit scalar operand: a register, an inline constant (as f16 bits when <paramref name="half"/>) or the literal.</summary>
    internal uint Scalar32(GcnWave wave, int value, GcnInstruction i, bool half = false)
    {
        switch (value)
        {
            case < 128:
                return ReadScalar(wave, value, i);
            case <= 192:
                return (uint)(value - 128);
            case <= 208:
                return (uint)(192 - value);
            case >= 240 and <= 248:
                return half ? Half16[value - 240] : Float32[value - 240];
            case 251:
                return wave.Vcc == 0 ? 1u : 0u;
            case 252:
                return wave.Exec == 0 ? 1u : 0u;
            case 253:
                return wave.Scc ? 1u : 0u;
            case 255:
                return i.Literal;
            default:
                throw Fail(GcnFailure.NotModelled, wave, i, $"source {GcnPrinter.Operand(value, 1, 's', i, false)} is not modelled");
        }
    }

    /// <summary>A 64-bit scalar operand: a register pair, an inline constant (f64 bits for the float ones) or the literal (zero-extended; the high half for a float).</summary>
    internal ulong Scalar64(GcnWave wave, int value, GcnInstruction i, bool f64 = false)
    {
        switch (value)
        {
            case < 128:
                return ReadScalar64(wave, value, i);
            case <= 192:
                return (ulong)(value - 128);
            case <= 208:
                return (ulong)(long)(192 - value);
            case >= 240 and <= 248:
                return Float64[value - 240];
            case 255:
                return f64 ? (ulong)i.Literal << 32 : i.Literal;
            default:
                return Scalar32(wave, value, i);
        }
    }

    private static readonly uint[] Float32 = [0x3F000000, 0xBF000000, 0x3F800000, 0xBF800000, 0x40000000, 0xC0000000, 0x40800000, 0xC0800000, 0x3E22F983];

    private static readonly uint[] Half16 = [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118];

    private static readonly ulong[] Float64 =
        [0x3FE0000000000000, 0xBFE0000000000000, 0x3FF0000000000000, 0xBFF0000000000000, 0x4000000000000000, 0xC000000000000000, 0x4010000000000000, 0xC010000000000000, 0x3FC45F306DC9C882];

    internal void CheckVector(GcnWave wave, int r, int count, GcnInstruction i, bool acc = false)
    {
        int limit = descriptor.Vgprs;
        if (r < 0 || r + count > limit)
        {
            string name = count == 1 ? $"{(acc ? 'a' : 'v')}{r}" : $"{(acc ? 'a' : 'v')}[{r}:{r + count - 1}]";
            throw Fail(GcnFailure.Register, wave, i, $"{name} is past the {limit} {(acc ? "AGPRs" : "VGPRs")} the descriptor allocates");
        }
    }

    /// <summary>Reads vector register <paramref name="r"/> (all 64 lanes) into <paramref name="lanes"/>, checked against pending loads.</summary>
    internal void ReadVector(GcnWave wave, int r, GcnInstruction i, Span<uint> lanes, bool acc = false)
    {
        CheckVector(wave, r, 1, i, acc);
        var owner = acc ? wave.AOwner : wave.VOwner;
        if (owner[r] is { } pending)
        {
            throw PendingRead(wave, i, $"{(acc ? 'a' : 'v')}{r}", pending);
        }

        (acc ? wave.Acc : wave.V).AsSpan(r * 64, 64).CopyTo(lanes);
    }

    /// <summary>Writes <paramref name="lanes"/> into vector register <paramref name="r"/> for the lanes in <paramref name="mask"/>.</summary>
    internal void WriteVector(GcnWave wave, int r, GcnInstruction i, ReadOnlySpan<uint> lanes, ulong mask, bool acc = false)
    {
        CheckVector(wave, r, 1, i, acc);
        var owner = acc ? wave.AOwner : wave.VOwner;
        if (owner[r] is { } pending)
        {
            throw PendingWrite(wave, i, $"{(acc ? 'a' : 'v')}{r}", pending);
        }

        var target = (acc ? wave.Acc : wave.V).AsSpan(r * 64, 64);
        if (mask == ulong.MaxValue)
        {
            lanes.CopyTo(target);
            return;
        }

        while (mask != 0)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            target[lane] = lanes[lane];
            mask &= mask - 1;
        }
    }

    /// <summary>A 32-bit source operand for every lane: a vector register, or a scalar operand repeated.</summary>
    internal void Source32(GcnWave wave, int value, GcnInstruction i, Span<uint> lanes, bool half = false)
    {
        if (value >= 256)
        {
            ReadVector(wave, value - 256, i, lanes);
        }
        else
        {
            lanes.Fill(Scalar32(wave, value, i, half));
        }
    }

    /// <summary>A 64-bit source operand for every lane, low and high halves apart.</summary>
    internal void Source64(GcnWave wave, int value, GcnInstruction i, Span<uint> low, Span<uint> high, bool f64 = false)
    {
        if (value >= 256)
        {
            ReadVector(wave, value - 256, i, low);
            ReadVector(wave, value - 255, i, high);
        }
        else
        {
            ulong v = Scalar64(wave, value, i, f64);
            low.Fill((uint)v);
            high.Fill((uint)(v >> 32));
        }
    }

    /// <summary>EXEC, for an instruction that reads it (every vector instruction), checked against pending loads.</summary>
    internal ulong ExecOf(GcnWave wave, GcnInstruction i)
    {
        if (wave.SOwner[126] is { } low)
        {
            throw PendingRead(wave, i, "exec", low);
        }

        if (wave.SOwner[127] is { } high)
        {
            throw PendingRead(wave, i, "exec", high);
        }

        return wave.Exec;
    }

    private GcnInterpreterException PendingRead(GcnWave wave, GcnInstruction i, string register, GcnPending pending) =>
        Fail(GcnFailure.MissingWait, wave, i,
            $"it reads {register}, which '{pending.Instruction.Text}' at 0x{pending.Instruction.Offset:X4} still writes: no s_waitcnt {Counter(pending)} retired it");

    private GcnInterpreterException PendingWrite(GcnWave wave, GcnInstruction i, string register, GcnPending pending) =>
        Fail(GcnFailure.MissingWait, wave, i,
            $"it writes {register}, which '{pending.Instruction.Text}' at 0x{pending.Instruction.Offset:X4} still writes (its late return would overwrite this): no s_waitcnt {Counter(pending)} retired it");

    private static string Counter(GcnPending p) => p.Vm && p.Lgkm ? "vmcnt or lgkmcnt" : p.Vm ? "vmcnt" : "lgkmcnt";

    // ---- Pending memory operations and s_waitcnt ----------------------------------------------------------------

    /// <summary>
    /// Issues a memory operation: its destination registers are pending until a wait retires it. Writing a register
    /// another pending operation still writes is reported (its return would race this one's).
    /// </summary>
    internal void Issue(GcnWave wave, GcnPending pending)
    {
        var i = pending.Instruction;
        if (pending.VFirst >= 0)
        {
            bool acc = false;
            CheckVector(wave, pending.VFirst, pending.VCount, i, acc);
            for (int r = pending.VFirst; r < pending.VFirst + pending.VCount; r++)
            {
                if (wave.VOwner[r] is { } other)
                {
                    throw PendingWrite(wave, i, $"v{r}", other);
                }
            }

            for (int r = pending.VFirst; r < pending.VFirst + pending.VCount; r++)
            {
                wave.VOwner[r] = pending;
            }
        }

        if (pending.SFirst >= 0)
        {
            for (int r = pending.SFirst; r < pending.SFirst + pending.SCount; r++)
            {
                CheckScalar(wave, r, i);
                if (wave.SOwner[r] is { } other)
                {
                    throw PendingWrite(wave, i, ScalarName(r), other);
                }
            }

            for (int r = pending.SFirst; r < pending.SFirst + pending.SCount; r++)
            {
                wave.SOwner[r] = pending;
            }
        }

        if (pending.Vm)
        {
            wave.VmQueue.Add(pending);
            if (wave.VmQueue.Count > 63)
            {
                Retire(wave, wave.VmQueue[0]);
            }
        }

        if (pending.Lgkm)
        {
            wave.LgkmQueue.Add(pending);
            if (wave.LgkmQueue.Count > 15)
            {
                Retire(wave, wave.LgkmQueue[0]);
            }
        }
    }

    /// <summary>
    /// s_waitcnt vmcnt(<paramref name="vm"/>) lgkmcnt(<paramref name="lgkm"/>): retires what the counters guarantee.
    /// vmcnt counts loads and stores, which complete in order on gfx9: the oldest beyond vm are done. lgkmcnt counts
    /// LDS operations, which complete in order among themselves, and SMEM and FLAT, which complete in any order: the
    /// oldest LDS operations beyond lgkm are done, and SMEM and FLAT only when lgkm is 0 (FLAT also through vmcnt).
    /// </summary>
    internal void Wait(GcnWave wave, int vm, int lgkm)
    {
        while (wave.VmQueue.Count > vm)
        {
            Retire(wave, wave.VmQueue[0]);
        }

        if (lgkm == 0)
        {
            while (wave.LgkmQueue.Count > 0)
            {
                Retire(wave, wave.LgkmQueue[0]);
            }

            return;
        }

        int inOrder = 0;
        foreach (var p in wave.LgkmQueue)
        {
            inOrder += p.InOrder ? 1 : 0;
        }

        for (int done = inOrder - lgkm; done > 0; done--)
        {
            Retire(wave, wave.LgkmQueue.First(p => p.InOrder));
        }
    }

    /// <summary>Retires everything (the wave's end: the hardware completes what is in flight).</summary>
    internal void RetireAll(GcnWave wave)
    {
        while (wave.VmQueue.Count > 0)
        {
            Retire(wave, wave.VmQueue[0]);
        }

        while (wave.LgkmQueue.Count > 0)
        {
            Retire(wave, wave.LgkmQueue[0]);
        }
    }

    private static void Retire(GcnWave wave, GcnPending pending)
    {
        if (pending.VFirst >= 0)
        {
            for (int k = 0; k < pending.VCount; k++)
            {
                int r = pending.VFirst + k;
                var target = wave.V.AsSpan(r * 64, 64);
                var values = pending.VValues.AsSpan(k * 64, 64);
                for (ulong mask = pending.Lanes; mask != 0; mask &= mask - 1)
                {
                    int lane = BitOperations.TrailingZeroCount(mask);
                    target[lane] = pending.Merge switch
                    {
                        1 => (target[lane] & 0xFFFF0000) | (values[lane] & 0xFFFF),
                        2 => (target[lane] & 0xFFFF) | values[lane] << 16,
                        _ => values[lane],
                    };
                }

                if (wave.VOwner[r] == pending)
                {
                    wave.VOwner[r] = null;
                }
            }
        }

        if (pending.SFirst >= 0)
        {
            for (int k = 0; k < pending.SCount; k++)
            {
                wave.S[pending.SFirst + k] = pending.SValues![k];
                if (wave.SOwner[pending.SFirst + k] == pending)
                {
                    wave.SOwner[pending.SFirst + k] = null;
                }
            }
        }

        if (pending.Store is { } store)
        {
            store.Retired = true;
        }

        wave.VmQueue.Remove(pending);
        wave.LgkmQueue.Remove(pending);
    }

    /// <summary>A new store record of the current workgroup, wave and barrier interval; its index is what the race tracking keeps per byte.</summary>
    internal int NewStore(GcnWave wave, GcnInstruction i, out GcnStoreRecord record)
    {
        record = new GcnStoreRecord { Group = Group, Wave = wave.Index, Epoch = Epoch, Instruction = i };
        Stores.Add(record);
        return Stores.Count - 1;
    }
}
