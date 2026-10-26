// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// What the machine reports about its processor: logical and physical cores, the vector instructions the runtime
/// accelerates, and the cache sizes. Read once, from the operating system (Linux and Android: /sys/devices/system/cpu; Windows:
/// GetLogicalProcessorInformationEx; macOS: sysctl), with a fallback per value when it cannot be read. Every CPU tiling,
/// blocking and threading choice is derived from these (see <see cref="CpuTuning"/>); no processor name or vendor is
/// read for a decision.
/// </summary>
internal sealed record CpuInfo
{
    /// <summary>L1 data cache per core when the system does not report it.</summary>
    internal const int FallbackL1 = 32 << 10;

    /// <summary>L2 cache per core when the system does not report it.</summary>
    internal const int FallbackL2 = 1 << 20;

    /// <summary>Cache line when the system does not report it.</summary>
    internal const int FallbackLine = 64;

    private static readonly Lazy<CpuInfo> s_current = new(Read);

    /// <summary>This machine.</summary>
    public static CpuInfo Current => s_current.Value;

    /// <summary>Logical processors (hardware threads) the process may use.</summary>
    public required int LogicalProcessors { get; init; }

    /// <summary>Physical cores (logical processors when the system does not say).</summary>
    public required int PhysicalCores { get; init; }

    /// <summary>Floats in a <see cref="Vector{T}"/> (the portable kernels' width).</summary>
    public required int VectorFloats { get; init; }

    /// <summary>The widest accelerated vector, in floats (16: AVX-512; 8: AVX2/AVX; 4: SSE, NEON).</summary>
    public required int WidestFloats { get; init; }

    /// <summary>Architectural vector registers (32 with AVX-512 or on ARM64; 16 otherwise).</summary>
    public required int VectorRegisters { get; init; }

    /// <summary>Fused multiply-add in hardware (x86 FMA3, ARM64 AdvSimd).</summary>
    public required bool FusedMultiplyAdd { get; init; }

    /// <summary>The instruction sets the runtime accelerates (for diagnostics and the tuning cache key).</summary>
    public required string InstructionSets { get; init; }

    /// <summary>L1 data cache per core in bytes, and whether the system reported it.</summary>
    public required int L1 { get; init; }

    /// <summary>L2 cache per core in bytes (a shared L2 divided by the cores sharing it).</summary>
    public required int L2 { get; init; }

    /// <summary>Last-level cache in bytes (0 when not reported).</summary>
    public required long L3 { get; init; }

    /// <summary>Cache line in bytes.</summary>
    public required int CacheLine { get; init; }

    /// <summary>Whether the cache sizes came from the system (false: the fallbacks).</summary>
    public required bool CachesReported { get; init; }

    /// <summary>
    /// The processor's name as the system reports it (its "model name"; where there is none, as on ARM: the system's
    /// "Hardware" name or the core groups); only a key of the tuning cache, never a reason for a choice.
    /// </summary>
    public required string Model { get; init; }

    /// <summary>
    /// The cores by kind, in the order the system numbers them (Linux and Android: each processor's part number in
    /// /proc/cpuinfo and its highest frequency in cpufreq; several groups on processors that mix core types). Raw numbers
    /// only: no table names a part. Empty where the system does not say.
    /// </summary>
    public CoreGroup[] CoreGroups { get; init; } = [];

    /// <summary>The vector paths the runtime reports: platform intrinsics, portable vectors only (the runtime reports none), or none.</summary>
    public VectorSupport Vectors { get; init; }

    public override string ToString() =>
        $"{LogicalProcessors} threads / {PhysicalCores} cores, {InstructionSets}, {VectorRegisters} vector registers, "
        + $"L1d {L1 >> 10} KiB, L2 {L2 >> 10} KiB per core, L3 {(L3 > 0 ? $"{L3 >> 20} MiB" : "unknown")}, line {CacheLine} B"
        + (CachesReported ? "" : " (cache sizes not reported: fallbacks)");

    /// <summary>The machine's report with the given cache sizes (tests: the formulas on other machines).</summary>
    internal static CpuInfo With(int l1, int l2, int registers = 0, int threads = 0) => Current with
    {
        L1 = l1,
        L2 = l2,
        VectorRegisters = registers > 0 ? registers : Current.VectorRegisters,
        LogicalProcessors = threads > 0 ? threads : Current.LogicalProcessors,
    };

    private static CpuInfo Read()
    {
        var caches = new Caches();
        int physical = 0;
        string model = "";
        CoreGroup[] groups = [];
        try
        {
            if (OperatingSystem.IsLinux() || OperatingSystem.IsAndroid())
            {
                (physical, model, groups) = ReadLinux(caches);
            }
            else if (OperatingSystem.IsWindows())
            {
                physical = ReadWindows(caches);
                model = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "";
            }
            else if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst())
            {
                (physical, model) = ReadApple(caches);
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException or DllNotFoundException
            or EntryPointNotFoundException or OverflowException)
        {
            // Whatever was read stays; the rest falls back.
        }

        int logical = Environment.ProcessorCount;
        int widest = Vector512.IsHardwareAccelerated ? 16 : Vector256.IsHardwareAccelerated ? 8 : Vector128.IsHardwareAccelerated ? 4 : 1;
        bool x86 = Sse.IsSupported;
        var vectors = VectorsReported(x86 || AdvSimd.IsSupported, Vector128.IsHardwareAccelerated);
        // ARM64 has 32 vector registers whether or not the runtime reports its intrinsics (Mono reports none).
        bool arm64 = AdvSimd.Arm64.IsSupported || RuntimeInformation.ProcessArchitecture == Architecture.Arm64;
        int registers = x86 ? (Avx512F.IsSupported ? 32 : 16) : arm64 ? 32 : 16;
        var sets = new List<string>();
        void Add(bool supported, string name)
        {
            if (supported)
            {
                sets.Add(name);
            }
        }

        Add(Avx512F.IsSupported, "AVX-512F");
        Add(Avx512BW.IsSupported, "AVX-512BW");
        Add(Avx512Vbmi.IsSupported, "AVX-512VBMI");
        Add(Avx512Vbmi2.IsSupported, "AVX-512VBMI2");
        Add(Avx10v1.IsSupported, "AVX10.1");
        Add(Avx10v2.IsSupported, "AVX10.2");
        Add(Avx2.IsSupported, "AVX2");
        Add(Fma.IsSupported, "FMA");
        Add(AvxVnni.IsSupported, "AVX-VNNI");
        Add(AvxVnniInt8.IsSupported, "AVX-VNNI-INT8");
        Add(x86 && !Avx2.IsSupported && Sse2.IsSupported, "SSE2");
        Add(AdvSimd.IsSupported, "NEON");
        Add(AdvSimd.Arm64.IsSupported, "NEON64");
        Add(Dp.IsSupported, "DotProd");
        Add(Rdm.IsSupported, "RDM");
        return new CpuInfo
        {
            LogicalProcessors = logical,
            PhysicalCores = physical is > 0 and var p && p <= logical ? p : logical,
            VectorFloats = Vector<float>.Count,
            WidestFloats = widest,
            VectorRegisters = registers,
            // Every ARMv8-A core multiplies and adds fused in its vector unit (reported or not by the runtime).
            FusedMultiplyAdd = Fma.IsSupported || AdvSimd.IsSupported || arm64 && vectors == VectorSupport.Portable,
            InstructionSets = DescribeSets(sets, vectors, RuntimeInformation.ProcessArchitecture),
            Vectors = vectors,
            CoreGroups = groups,
            L1 = caches.L1 > 0 ? caches.L1 : FallbackL1,
            L2 = caches.L2 > 0 ? caches.L2 : FallbackL2,
            L3 = caches.L3,
            CacheLine = caches.Line > 0 ? caches.Line : FallbackLine,
            CachesReported = caches.L1 > 0 && caches.L2 > 0,
            Model = model.Trim(),
        };
    }

    /// <summary>What the runtime reports: platform intrinsics, only accelerated portable vectors, or neither.</summary>
    internal static VectorSupport VectorsReported(bool intrinsics, bool vector128Accelerated) =>
        intrinsics ? VectorSupport.Intrinsics : vector128Accelerated ? VectorSupport.Portable : VectorSupport.None;

    /// <summary>
    /// The instruction sets for listings: the platform sets the runtime reports, or, where it reports none but accelerates
    /// 128-bit vectors (Mono on ARM64), the architecture's vector unit run through portable vectors; "no SIMD" only where
    /// neither is accelerated.
    /// </summary>
    internal static string DescribeSets(IReadOnlyList<string> sets, VectorSupport vectors, Architecture architecture)
    {
        if (sets.Count > 0)
        {
            return string.Join(" ", sets);
        }

        string unit = architecture switch
        {
            Architecture.Arm64 or Architecture.Arm => "NEON",
            Architecture.X64 or Architecture.X86 => "SSE",
            _ => "128-bit vectors",
        };
        return vectors == VectorSupport.Portable ? $"{unit} (portable vectors: the runtime reports no {unit} intrinsics)" : "no SIMD";
    }

    // The largest per-core size of each level (on hybrid processors: the performance cores').
    private sealed class Caches
    {
        public int L1, L2, Line;
        public long L3;

        public void Add(int level, long bytes, int sharedCores)
        {
            long perCore = bytes / Math.Max(1, sharedCores);
            switch (level)
            {
                case 1: L1 = (int)Math.Max(L1, Math.Min(perCore, int.MaxValue)); break;
                case 2: L2 = (int)Math.Max(L2, Math.Min(perCore, int.MaxValue)); break;
                case >= 3: L3 = Math.Max(L3, bytes); break;
            }
        }
    }

    // /sys/devices/system/cpu/cpu0/cache/index*/{level,type,size,shared_cpu_list,coherency_line_size} and the cores in
    // /sys/devices/system/cpu/cpu*/topology/{core_id,physical_package_id} (or cluster_cpus_list on some ARM boards).
    private static (int Physical, string Model, CoreGroup[] Groups) ReadLinux(Caches caches)
    {
        const string Root = "/sys/devices/system/cpu";
        var cores = new HashSet<(string Package, string Core)>();
        var siblings = new Dictionary<int, int>();                            // logical CPU → logical CPUs on its core
        if (Directory.Exists(Root))
        {
            foreach (var dir in Directory.EnumerateDirectories(Root, "cpu*"))
            {
                string name = Path.GetFileName(dir);
                if (!int.TryParse(name.AsSpan(3), NumberStyles.Integer, CultureInfo.InvariantCulture, out int cpu))
                {
                    continue;
                }

                string topology = Path.Combine(dir, "topology");
                if (File.Exists(Path.Combine(topology, "core_id")))
                {
                    string package = File.Exists(Path.Combine(topology, "physical_package_id")) ? File.ReadAllText(Path.Combine(topology, "physical_package_id")).Trim() : "0";
                    cores.Add((package, File.ReadAllText(Path.Combine(topology, "core_id")).Trim()));
                    string threads = Path.Combine(topology, "thread_siblings_list");
                    siblings[cpu] = File.Exists(threads) ? CountList(File.ReadAllText(threads)) : 1;
                }
            }
        }

        int smt = siblings.TryGetValue(0, out int s) ? Math.Max(1, s) : 1;
        string cache = Path.Combine(Root, "cpu0", "cache");
        if (Directory.Exists(cache))
        {
            foreach (var index in Directory.EnumerateDirectories(cache, "index*"))
            {
                string Read(string file) => File.Exists(Path.Combine(index, file)) ? File.ReadAllText(Path.Combine(index, file)).Trim() : "";
                string type = Read("type");
                if (type == "Instruction" || !int.TryParse(Read("level"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int level))
                {
                    continue;
                }

                long bytes = ParseSize(Read("size"));
                int shared = Read("shared_cpu_list") is { Length: > 0 } list ? CountList(list) : 1;
                if (bytes > 0)
                {
                    caches.Add(level, bytes, (shared + smt - 1) / smt);
                }

                if (int.TryParse(Read("coherency_line_size"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int line) && line > 0)
                {
                    caches.Line = Math.Max(caches.Line, line);
                }
            }
        }

        string model = "";
        CoreGroup[] groups = [];
        if (File.Exists("/proc/cpuinfo"))
        {
            var (found, name) = ParseCores(File.ReadAllBytes("/proc/cpuinfo"), cpu => MaxKilohertz(Root, cpu));
            groups = found;
            model = name;
        }

        return (cores.Count, model, groups);
    }

    // cpuN/cpufreq/cpuinfo_max_freq (kHz), or null.
    private static long? MaxKilohertz(string root, int cpu)
    {
        string file = Path.Combine(root, string.Create(CultureInfo.InvariantCulture, $"cpu{cpu}"), "cpufreq", "cpuinfo_max_freq");
        try
        {
            return File.Exists(file) && SystemMemory.ParseBytes(File.ReadAllBytes(file)) is { } khz && khz > 0 ? khz : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// The core groups and the processor's name from /proc/cpuinfo's bytes: each "processor" block's part (ARM: "CPU
    /// part", with "CPU implementer"; x86: "model"), its highest frequency (<paramref name="maxKilohertz"/> of its number),
    /// consecutive processors of one part and frequency in one group. The name is the "model name" where the system gives
    /// one (x86: as before), else the system's "Hardware" or "Model" line with the groups, else the groups.
    /// </summary>
    internal static (CoreGroup[] Groups, string Model) ParseCores(ReadOnlySpan<byte> cpuinfo, Func<int, long?> maxKilohertz)
    {
        var groups = new List<CoreGroup>();
        string? modelName = null, system = null;
        int processor = -1;
        string? part = null, implementer = null;
        void Flush()
        {
            if (processor < 0)
            {
                return;
            }

            string label = part is null ? "" : implementer is null ? part : $"{implementer}:{part}";
            long? khz = maxKilohertz(processor);
            if (groups.Count > 0 && groups[^1].Part == label && groups[^1].MaxKilohertz == khz)
            {
                groups[^1] = groups[^1] with { Count = groups[^1].Count + 1 };
            }
            else
            {
                groups.Add(new CoreGroup(1, label, khz));
            }

            processor = -1;
            part = implementer = null;
        }

        foreach (var range in cpuinfo.Split((byte)'\n'))
        {
            var line = cpuinfo[range];
            int colon = line.IndexOf((byte)':');
            if (colon < 0)
            {
                continue;
            }

            var key = line[..colon].Trim("\t "u8);
            string value = System.Text.Encoding.UTF8.GetString(line[(colon + 1)..].Trim("\t \r"u8));
            if (key.SequenceEqual("processor"u8))
            {
                Flush();
                processor = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : groups.Sum(g => g.Count);
            }
            else if (key.SequenceEqual("CPU part"u8))
            {
                part = value;
            }
            else if (key.SequenceEqual("CPU implementer"u8))
            {
                implementer = value;
            }
            else if (key.SequenceEqual("model"u8) && part is null)
            {
                part = "model " + value;
            }
            else if (key.SequenceEqual("model name"u8))
            {
                modelName ??= value;
            }
            else if ((key.SequenceEqual("Hardware"u8) || key.SequenceEqual("Model"u8)) && value.Length > 0)
            {
                system ??= value;
            }
        }

        Flush();
        CoreGroup[] found = [.. groups];
        string described = found.Length > 0 ? DescribeGroups(found) : "";
        string model = modelName is { Length: > 0 } ? modelName
            : system is not null ? (described.Length > 0 ? $"{system} ({described})" : system)
            : described;
        return (found, model);
    }

    /// <summary>"8 cores: 4 × part 0x41:0xd05 up to 1.80 GHz, 4 × part 0x41:0xd0d up to 2.40 GHz" (a frequency where reported).</summary>
    internal static string DescribeGroups(IReadOnlyList<CoreGroup> groups)
    {
        int cores = groups.Sum(g => g.Count);
        var parts = groups.Select(g =>
        {
            string part = g.Part.Length == 0 ? "unknown part" : g.Part.StartsWith("model ", StringComparison.Ordinal) ? g.Part : "part " + g.Part;
            return g.MaxKilohertz is { } khz
                ? string.Create(CultureInfo.InvariantCulture, $"{g.Count} × {part} up to {khz / 1e6:F2} GHz")
                : string.Create(CultureInfo.InvariantCulture, $"{g.Count} × {part}");
        });
        return string.Create(CultureInfo.InvariantCulture, $"{cores} core{(cores == 1 ? "" : "s")}: {string.Join(", ", parts)}");
    }

    // "0-3,8,10-11" → 7.
    internal static int CountList(string list)
    {
        int count = 0;
        foreach (var part in list.Trim().Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            int dash = part.IndexOf('-');
            count += dash < 0 ? 1 : int.Parse(part[(dash + 1)..], CultureInfo.InvariantCulture) - int.Parse(part[..dash], CultureInfo.InvariantCulture) + 1;
        }

        return Math.Max(1, count);
    }

    // "32K", "1024K", "33 MiB", "48K" → bytes.
    internal static long ParseSize(string text)
    {
        text = text.Trim();
        int end = 0;
        while (end < text.Length && char.IsAsciiDigit(text[end]))
        {
            end++;
        }

        if (end == 0 || !long.TryParse(text.AsSpan(0, end), NumberStyles.Integer, CultureInfo.InvariantCulture, out long value))
        {
            return 0;
        }

        return text[end..].TrimStart().ToUpperInvariant() switch
        {
            "" or "B" => value,
            ['K', ..] => value << 10,
            ['M', ..] => value << 20,
            ['G', ..] => value << 30,
            _ => 0,
        };
    }

    // GetLogicalProcessorInformationEx(RelationAll): RelationProcessorCore (0) records count the physical cores,
    // RelationCache (2) records give level, line, size, type and the processors sharing the cache.
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static unsafe int ReadWindows(Caches caches)
    {
        uint length = 0;
        GetLogicalProcessorInformationEx(0xFFFF, null, ref length);
        if (length == 0)
        {
            return 0;
        }

        var buffer = new byte[length];
        fixed (byte* start = buffer)
        {
            if (!GetLogicalProcessorInformationEx(0xFFFF, start, ref length))
            {
                return 0;
            }

            var records = new List<(int Level, long Bytes, int Line, int Shared)>();
            int physical = 0, smt = 1;
            for (uint at = 0; at + 8 <= length;)
            {
                byte* record = start + at;
                int relationship = *(int*)record, size = *(int*)(record + 4);
                if (size <= 0)
                {
                    break;
                }

                if (relationship == 0)
                {
                    physical++;
                    if ((record[8] & 1) != 0)                                     // LTP_PC_SMT: two (or more) threads per core
                    {
                        smt = Math.Max(smt, BitOperations.PopCount(*(ulong*)(record + 32)));
                    }
                }
                else if (relationship == 2 && *(int*)(record + 16) != 1)        // CacheInstruction = 1
                {
                    int shared = Math.Max(1, BitOperations.PopCount(*(ulong*)(record + 40)));
                    records.Add((record[8], *(uint*)(record + 12), *(ushort*)(record + 10), shared));
                }

                at += (uint)size;
            }

            foreach (var (level, bytes, line, shared) in records)
            {
                caches.Add(level, bytes, (shared + smt - 1) / smt);
                caches.Line = Math.Max(caches.Line, line);
            }

            return physical;
        }
    }

    [DllImport("kernel32", SetLastError = true)]
    private static extern unsafe bool GetLogicalProcessorInformationEx(int relationship, byte* buffer, ref uint length);

    // sysctl: the performance cores' caches where the processor has core types (hw.perflevel0.*), else hw.*.
    private static (int Physical, string Model) ReadApple(Caches caches)
    {
        long l1 = Sysctl("hw.perflevel0.l1dcachesize") is > 0 and var p1 ? p1 : Sysctl("hw.l1dcachesize");
        long l2 = Sysctl("hw.perflevel0.l2cachesize") is > 0 and var p2 ? p2 : Sysctl("hw.l2cachesize");
        long perL2 = Sysctl("hw.perflevel0.cpusperl2") is > 0 and var c2 ? c2 : 1;
        caches.Add(1, l1, 1);
        caches.Add(2, l2, (int)perL2);
        caches.Add(3, Sysctl("hw.l3cachesize"), 1);
        caches.Line = (int)Math.Max(0, Sysctl("hw.cachelinesize"));
        return ((int)Math.Max(0, Sysctl("hw.physicalcpu")), "");
    }

    private static unsafe long Sysctl(string name)
    {
        long value = 0;
        nint size = sizeof(long);
        if (sysctlbyname(name, &value, ref size, null, 0) != 0)
        {
            return 0;
        }

        return size == sizeof(int) ? (int)value : value;
    }

    [DllImport("libc", CharSet = CharSet.Ansi)]
    private static extern unsafe int sysctlbyname([MarshalAs(UnmanagedType.LPStr)] string name, void* value, ref nint size, void* newValue, nint newSize);
}

/// <summary>Consecutive cores of one kind (<see cref="CpuInfo.CoreGroups"/>).</summary>
/// <param name="Count">How many.</param>
/// <param name="Part">The raw part as the system reports it ("0x41:0xd05" on ARM, "model 85" on x86; "" when not reported).</param>
/// <param name="MaxKilohertz">The highest frequency the system reports for them, in kHz (null when not reported).</param>
internal readonly record struct CoreGroup(int Count, string Part, long? MaxKilohertz);

/// <summary>Which vector paths the runtime reports (<see cref="CpuInfo.Vectors"/>).</summary>
internal enum VectorSupport : byte
{
    /// <summary>No accelerated vectors.</summary>
    None,

    /// <summary>Accelerated 128-bit portable vectors without the platform's intrinsics (Mono on ARM64 reports no AdvSimd).</summary>
    Portable,

    /// <summary>The platform's intrinsics (SSE and up, AdvSimd).</summary>
    Intrinsics,
}
