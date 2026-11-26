// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// The CPU's measured choices kept per machine: each shape's candidate (a width of the float lanes, a lane of the
/// described lane), measured once with every candidate timed alike (<see cref="RoundTimes{TC}"/>) and kept in a file of
/// this machine's own (the processor, its threads and the instruction sets the runtime runs with), so the next process
/// reads it instead of measuring again. The key holds the vector paths setting (IDRAK_CPU_VECTORS): a choice made under
/// one setting is never taken under another. <c>IDRAK_TUNING_CACHE=0</c> keeps none (measured each process); a path keeps
/// them there. Every candidate gives the same bits, so a choice changes speed alone.
/// </summary>
internal static class CpuChoices
{
    private const string Magic = "idrak-cpu-choices 1";

    // Rounds RoundTimes takes: each candidate's median over them.
    private const int Rounds = 9;

    // A round's run of a candidate at least this long (its calls repeated, every candidate the same number of times).
    private const double SampleMs = 0.05;

    private static readonly ConcurrentDictionary<string, string> s_kept = new(StringComparer.Ordinal);
    private static readonly Lock s_gate = new();
    private static volatile bool s_loaded;

    /// <summary>What the choices apply to: the processor, its threads, the architecture, the runtime and the instruction sets it runs with.</summary>
    public static string Identity { get; } = MakeIdentity();

    private static string MakeIdentity()
    {
        var cpu = CpuInfo.Current;
        var sets = new StringBuilder();
        foreach (var (name, on) in new (string, bool)[]
                 {
                     ("avx512f", Avx512F.IsSupported), ("avx2", Avx2.IsSupported), ("fma", Fma.IsSupported), ("avxvnni", AvxVnni.IsSupported),
                     ("advsimd", AdvSimd.IsSupported), ("dp", Dp.IsSupported), ("v512", Vector512.IsHardwareAccelerated),
                     ("v256", Vector256.IsHardwareAccelerated), ("v128", Vector128.IsHardwareAccelerated),
                 })
        {
            sets.Append(on ? name : "no-" + name).Append(' ');
        }

        return string.Create(CultureInfo.InvariantCulture,
            $"{(cpu.Model.Length > 0 ? cpu.Model : "unnamed processor")}; {Environment.ProcessorCount} threads; {RuntimeInformation.ProcessArchitecture}; {RuntimeInformation.FrameworkDescription}; {sets.ToString().TrimEnd()}");
    }

    /// <summary>The file this machine's choices are kept in, or null when none are kept (IDRAK_TUNING_CACHE=0).</summary>
    public static string? FilePath
    {
        get
        {
            string? setting = Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE");
            if (TuningFiles.IsOff(setting))
            {
                return null;
            }

            string folder = !string.IsNullOrWhiteSpace(setting) && !setting.Trim().Equals("1", StringComparison.Ordinal)
                && !setting.Trim().Equals("on", StringComparison.OrdinalIgnoreCase) && !setting.Trim().Equals("true", StringComparison.OrdinalIgnoreCase)
                ? Path.Combine(Path.GetFullPath(setting.Trim()), "cpu")
                : Path.Combine(ComputeResources.CacheFolder, "tuning", "cpu");
            string hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(Identity)))[..16];
            return Path.Combine(folder, "choices-" + hash + ".txt");
        }
    }

    /// <summary>The kept choice under <paramref name="key"/> (with the vector paths setting in it), if any.</summary>
    public static bool TryGet(string key, out string value)
    {
        EnsureLoaded();
        return s_kept.TryGetValue(key, out value!);
    }

    /// <summary>Keeps <paramref name="value"/> under <paramref name="key"/>: here, and in this machine's file when one is kept.</summary>
    public static void Keep(string key, string value)
    {
        EnsureLoaded();
        s_kept[key] = value;
        if (FilePath is not { } file)
        {
            return;
        }

        try
        {
            using var held = TuningFiles.Lock(file);
            var entries = Read(file);
            entries[key] = value;
            string temporary = file + "." + Environment.ProcessId.ToString(CultureInfo.InvariantCulture) + ".tmp";
            var text = new StringBuilder();
            text.Append(Magic).Append('\n').Append(Identity).Append('\n');
            foreach (var (k, v) in entries.OrderBy(e => e.Key, StringComparer.Ordinal))
            {
                text.Append(k).Append('\t').Append(v).Append('\n');
            }

            File.WriteAllText(temporary, text.ToString(), new UTF8Encoding(false));
            File.Move(temporary, file, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // A folder that cannot be written keeps the choice in this process alone.
        }
    }

    /// <summary>Forgets the choices read or made in this process (tests); the file stays.</summary>
    public static void Forget()
    {
        lock (s_gate)
        {
            s_kept.Clear();
            s_loaded = false;
        }
    }

    private static void EnsureLoaded()
    {
        if (s_loaded)
        {
            return;
        }

        lock (s_gate)
        {
            if (s_loaded)
            {
                return;
            }

            if (FilePath is { } file)
            {
                try
                {
                    foreach (var (k, v) in Read(file))
                    {
                        s_kept.TryAdd(k, v);
                    }
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException)
                {
                    // An unreadable file: measured again.
                }
            }

            s_loaded = true;
        }
    }

    // A file's entries: none when it is missing, not a choices file, or another machine's.
    private static Dictionary<string, string> Read(string file)
    {
        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(file))
        {
            return entries;
        }

        string[] lines = File.ReadAllLines(file, Encoding.UTF8);
        if (lines.Length < 2 || lines[0] != Magic || lines[1] != Identity)
        {
            return entries;
        }

        for (int i = 2; i < lines.Length; i++)
        {
            int tab = lines[i].IndexOf('\t', StringComparison.Ordinal);
            if (tab > 0)
            {
                entries[lines[i][..tab]] = lines[i][(tab + 1)..];
            }
        }

        return entries;
    }

    /// <summary>Candidates of a choice, each run on the operands it is measured on.</summary>
    internal interface ICandidates
    {
        /// <summary>Runs candidate <paramref name="index"/> once.</summary>
        void Run(int index);
    }

    /// <summary>
    /// Every candidate timed alike, none against a holder and none given a head start: each run once (compiled and warmed),
    /// then <see cref="Rounds"/> rounds of every candidate in the same order, each round's run of a candidate repeated as
    /// often as the others' (enough that the fastest warm run fills <see cref="SampleMs"/>), each candidate's median over
    /// the rounds in ms a call.
    /// </summary>
    public static double[] RoundTimes<TC>(ref TC candidates, int count) where TC : struct, ICandidates
    {
        double fastest = double.MaxValue;
        for (int c = 0; c < count; c++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            candidates.Run(c);
            fastest = Math.Min(fastest, System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }

        int calls = (int)Math.Clamp(Math.Ceiling(SampleMs / Math.Max(fastest, 1e-5)), 1, 64);
        var rounds = new double[count * Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            for (int c = 0; c < count; c++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                for (int i = 0; i < calls; i++)
                {
                    candidates.Run(c);
                }

                rounds[c * Rounds + r] = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds / calls;
            }
        }

        var times = new double[count];
        for (int c = 0; c < count; c++)
        {
            var own = rounds.AsSpan(c * Rounds, Rounds);
            own.Sort();
            times[c] = own[Rounds / 2];
        }

        return times;
    }

    /// <summary>The index of the lowest of <paramref name="times"/> (the first of equal ones).</summary>
    public static int Lowest(ReadOnlySpan<double> times)
    {
        int best = 0;
        for (int c = 1; c < times.Length; c++)
        {
            if (times[c] < times[best])
            {
                best = c;
            }
        }

        return best;
    }
}
