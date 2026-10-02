// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Backends.Cuda;

/// <summary>
/// How measured candidates are compared (see CudaBackend.Tuning.cs). Pure: the timer is passed in, so tests check the
/// choice without a GPU.
/// </summary>
internal static class TuneTiming
{
    /// <summary>
    /// Rounds of paired timings per candidate; the median of the pair ratios is kept, so up to three outliers (either
    /// way) cannot move the choice. Even, so each candidate runs as often before its reference as after it.
    /// </summary>
    internal const int Rounds = 8;

    /// <summary>A candidate replaces the formula's choice only when its median is at least this much faster (3%).</summary>
    internal const float Margin = 0.97f;

    /// <summary>GPU time (ms) every candidate runs, in turns, before timing, at the least (Warm)…</summary>
    internal const float WarmMinMs = 25f;

    /// <summary>… and at the most, when its speed has not settled before.</summary>
    internal const float WarmMaxMs = 250f;

    /// <summary>
    /// The index of the candidate to keep. Each candidate is timed in pairs with the reference (the formula's choice,
    /// <paramref name="fallback"/>, when it is a candidate, else the first), the two timings back to back, over
    /// <paramref name="rounds"/> rounds; the reference runs first in even rounds and second in odd ones, and the
    /// candidates take their turns forwards, then backwards. Each pair gives the ratio of the candidate's timing to the
    /// reference's, and the medians of these ratios are compared. The formula's choice stays unless the fastest beats it
    /// by the margin. Pairs cancel a clock that changes during the measurement: a GPU still raising its clocks (laptops:
    /// several times faster within a few milliseconds) or lowering them under a power limit. Before, every candidate ran
    /// once per round against the reference's single timing of that round, four rounds of seven with the reference
    /// first: a steady drift then moved each candidate's median by its distance from the reference in the round (with
    /// the reference first, the most splits), up to the whole drift of a round, and a candidate a few percent slower than
    /// the formula's choice could be kept. <paramref name="time"/>(index) returns one timing of that candidate (ms per run);
    /// <paramref name="medians"/>, when given, receives each candidate's median ratio (diagnostics).
    /// </summary>
    internal static int Choose(ReadOnlySpan<int> candidates, int fallback, Func<int, float> time, int rounds = Rounds, float margin = Margin,
        float[]? medians = null)
    {
        int count = candidates.Length;
        int formula = candidates.IndexOf(fallback);
        int reference = Math.Max(formula, 0);
        var ratios = new float[count * rounds];
        for (int round = 0; round < rounds; round++)
        {
            for (int i = 0; i < count; i++)
            {
                int c = round % 2 == 0 ? i : count - 1 - i;
                if (c == reference)
                {
                    continue;
                }

                float t, r;
                if (round % 2 == 0)
                {
                    r = time(reference);
                    t = time(c);
                }
                else
                {
                    t = time(c);
                    r = time(reference);
                }

                ratios[c * rounds + round] = t / Math.Max(r, 1e-9f);
            }
        }

        medians ??= new float[count];
        for (int c = 0; c < count; c++)
        {
            medians[c] = c == reference ? 1f : Median(ratios.AsSpan(c * rounds, rounds));
        }

        int fastest = 0;
        for (int c = 1; c < count; c++)
        {
            if (medians[c] < medians[fastest])
            {
                fastest = c;
            }
        }

        return formula >= 0 && medians[fastest] >= medians[formula] * margin ? formula : fastest;
    }

    /// <summary>
    /// Runs <paramref name="pass"/> (every candidate once; returns its GPU time in ms) until the GPU's speed has settled:
    /// at least <paramref name="minMs"/> of GPU time, then until two consecutive windows of about
    /// <paramref name="windowMs"/> each take the same time per pass within <paramref name="tolerance"/>, or
    /// <paramref name="maxMs"/> in all. Returns the GPU time spent. A GPU that was idle runs at a fraction of its clocks
    /// for the first milliseconds of work; timings taken then compare candidates at another speed than decoding runs at.
    /// </summary>
    internal static float Warm(Func<float> pass, float minMs = WarmMinMs, float maxMs = WarmMaxMs, float windowMs = 5f, float tolerance = 0.03f)
    {
        float total = 0f, window = 0f, previous = float.NaN;
        int passes = 0;
        for (int i = 0; i < 100_000; i++)
        {
            float t = Math.Max(0f, pass());
            total += t;
            window += t;
            passes++;
            if (total >= maxMs)
            {
                break;
            }

            if (window >= windowMs)
            {
                float perPass = window / passes;
                if (total >= minMs && !float.IsNaN(previous) && MathF.Abs(perPass - previous) <= tolerance * previous)
                {
                    break;
                }

                previous = perPass;
                (window, passes) = (0f, 0);
            }
        }

        return total;
    }

    /// <summary>The median of <paramref name="values"/> (sorted in place; the mean of the middle two for an even count).</summary>
    internal static float Median(Span<float> values)
    {
        values.Sort();
        int n = values.Length;
        return n == 0 ? float.NaN : n % 2 == 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
    }
}

/// <summary>
/// What a cache file of measured choices belongs to: the GPU (name, compute capability, SM count, memory), the driver
/// (the CUDA version it supports and its own release) and the library build with its kernels. A file whose identity
/// differs in any part is ignored and rewritten, so a driver update or new kernels are measured again.
/// </summary>
internal sealed record TuningIdentity(string Device, string Driver, string Library);

/// <summary>
/// Measured choices kept per GPU in the user's cache folder (<c>IDRAK_CACHE</c> or <c>~/.cache/idrak</c>, under
/// <c>tuning/cuda/</c>), so they are not measured again at every start. <c>IDRAK_TUNING_CACHE=0</c> turns it off;
/// <c>IDRAK_TUNING_CACHE=&lt;folder&gt;</c> uses that folder. One text file per GPU:
/// <code>
/// idrak-tuning 2
/// device: NVIDIA ... ; compute 12.0; 70 SMs; 16303 MiB
/// driver: CUDA 13000; 580.82.07
/// library: 0.1.7+...; kernels 1a2b...
/// GemvSplits 0 1 2048 1024 0 0 0 = 16
/// </code>
/// Pure functions (text in, text out) apart from <see cref="Load"/> and <see cref="Save"/>, so tests check them without a GPU.
/// </summary>
internal static class TuningCache
{
    /// <summary>
    /// The file format; a file of another version is ignored and rewritten. 2: choices measured with the warm-up, the
    /// ratios per round and the fused kernels timed whole (those of version 1 could be the plain kernel's, or noise).
    /// 3: candidates timed in pairs with the formula's choice (a drifting clock moved version 2's choices toward the
    /// candidates timed furthest from it), and decoding-sized products timed with a cold L2 cache, as decoding reads them.
    /// </summary>
    internal const int FormatVersion = 3;

    private const string Magic = "idrak-tuning";

    private static readonly Lock FileLock = new();

    /// <summary>The folder for cache files, or null when the cache is off (<c>IDRAK_TUNING_CACHE=0</c>).</summary>
    internal static string? Folder(Func<string, string?>? environment = null)
    {
        environment ??= Environment.GetEnvironmentVariable;
        string? setting = environment("IDRAK_TUNING_CACHE");
        if (setting is "0" || string.Equals(setting, "false", StringComparison.OrdinalIgnoreCase) || string.Equals(setting, "off", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (setting is { Length: > 0 } && setting is not "1" && !string.Equals(setting, "true", StringComparison.OrdinalIgnoreCase))
        {
            return setting;
        }

        string root = environment("IDRAK_CACHE") is { Length: > 0 } cache
            ? cache
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
        return Path.Combine(root, "tuning", "cuda");
    }

    /// <summary>A file name for the device: its identity with every character a file system may reject replaced.</summary>
    internal static string FileName(string device)
    {
        var name = new StringBuilder(device.Length);
        foreach (char ch in device)
        {
            name.Append(char.IsAsciiLetterOrDigit(ch) || ch is '.' or '-' ? ch : '_');
        }

        string text = name.ToString().Trim('_');
        while (text.Contains("__", StringComparison.Ordinal))
        {
            text = text.Replace("__", "_", StringComparison.Ordinal);
        }

        return (text.Length > 120 ? text[..120] : text) + ".txt";
    }

    /// <summary>Text of a cache file: the header, then one line per choice in a stable order.</summary>
    internal static string Serialize(TuningIdentity identity, IEnumerable<KeyValuePair<TuneKey, int>> entries)
    {
        var text = new StringBuilder();
        text.Append(Magic).Append(' ').Append(FormatVersion.ToString(CultureInfo.InvariantCulture)).Append('\n');
        text.Append("device: ").Append(OneLine(identity.Device)).Append('\n');
        text.Append("driver: ").Append(OneLine(identity.Driver)).Append('\n');
        text.Append("library: ").Append(OneLine(identity.Library)).Append('\n');
        foreach (var (key, value) in entries.OrderBy(e => e.Key.Op).ThenBy(e => e.Key.Variant).ThenBy(e => e.Key.A).ThenBy(e => e.Key.B)
                     .ThenBy(e => e.Key.C).ThenBy(e => e.Key.D).ThenBy(e => e.Key.E).ThenBy(e => e.Key.F))
        {
            text.Append(CultureInfo.InvariantCulture, $"{key.Op} {key.Variant} {key.A} {key.B} {key.C} {key.D} {key.E} {key.F} = {value}\n");
        }

        return text.ToString();
    }

    /// <summary>
    /// The choices in <paramref name="text"/>, or null when it is not a cache file of this format for
    /// <paramref name="expected"/> (another GPU, driver, library build or format version). Lines that do not parse (an
    /// operation this build does not know) are skipped.
    /// </summary>
    internal static Dictionary<TuneKey, int>? Parse(string text, TuningIdentity expected)
    {
        var lines = text.Split('\n');
        if (lines.Length < 4 || lines[0].TrimEnd('\r') != $"{Magic} {FormatVersion.ToString(CultureInfo.InvariantCulture)}"
            || lines[1].TrimEnd('\r') != "device: " + OneLine(expected.Device)
            || lines[2].TrimEnd('\r') != "driver: " + OneLine(expected.Driver)
            || lines[3].TrimEnd('\r') != "library: " + OneLine(expected.Library))
        {
            return null;
        }

        var entries = new Dictionary<TuneKey, int>();
        for (int i = 4; i < lines.Length; i++)
        {
            var parts = lines[i].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 10 || parts[8] != "=" || !Enum.TryParse(parts[0], ignoreCase: false, out TuneOp op) || !Enum.IsDefined(op))
            {
                continue;
            }

            var numbers = new int[8];
            bool valid = true;
            for (int j = 1; j < 10 && valid; j++)
            {
                if (j == 8)
                {
                    continue;
                }

                valid = int.TryParse(parts[j], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out numbers[j < 8 ? j - 1 : 7]);
            }

            if (valid)
            {
                entries[new TuneKey(op, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6])] = numbers[7];
            }
        }

        return entries;
    }

    /// <summary>The choices kept in <paramref name="path"/> for <paramref name="identity"/>; empty when there are none or the file is another identity's.</summary>
    internal static Dictionary<TuneKey, int> Load(string path, TuningIdentity identity)
    {
        try
        {
            lock (FileLock)
            {
                return File.Exists(path) ? Parse(File.ReadAllText(path), identity) ?? [] : [];
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    /// <summary>
    /// Writes <paramref name="entries"/> to <paramref name="path"/>, keeping the choices another process wrote there for
    /// the same identity (ours win) and replacing a file of another identity. Written to a temporary file first and moved
    /// into place, so a reader never sees half a file. False when it could not be written (a read-only folder: the
    /// choices are then measured again next time).
    /// </summary>
    internal static bool Save(string path, TuningIdentity identity, IReadOnlyDictionary<TuneKey, int> entries)
    {
        try
        {
            lock (FileLock)
            {
                var merged = File.Exists(path) ? Parse(File.ReadAllText(path), identity) ?? [] : [];
                foreach (var (key, value) in entries)
                {
                    merged[key] = value;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                string temporary = $"{path}.{Environment.ProcessId}.tmp";
                File.WriteAllText(temporary, Serialize(identity, merged));
                File.Move(temporary, path, overwrite: true);
                return true;
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// The driver's own release where the system shows it (Linux: /proc/driver/nvidia/version; Windows: the version of
    /// nvcuda.dll), else empty. The CUDA version from cuDriverGetVersion is part of the identity as well.
    /// </summary>
    internal static string DriverRelease()
    {
        try
        {
            if (OperatingSystem.IsLinux() && File.Exists("/proc/driver/nvidia/version"))
            {
                // "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  Fri Aug ..."
                string line = File.ReadLines("/proc/driver/nvidia/version").FirstOrDefault() ?? "";
                int at = line.IndexOf("Kernel Module", StringComparison.Ordinal);
                var words = (at >= 0 ? line[(at + "Kernel Module".Length)..] : line).Split(' ', StringSplitOptions.RemoveEmptyEntries);
                return words.Length > 0 ? words[0] : "";
            }

            if (OperatingSystem.IsWindows())
            {
                string dll = Path.Combine(Environment.SystemDirectory, "nvcuda.dll");
                return File.Exists(dll) ? System.Diagnostics.FileVersionInfo.GetVersionInfo(dll).FileVersion ?? "" : "";
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
        }

        return "";
    }

    /// <summary>
    /// The library build: its version without a development build's time stamp (the commit stays when the build records
    /// it), and a hash of the generated kernels, so a build with changed kernels measures again.
    /// </summary>
    internal static string LibraryBuild(string kernels)
    {
        string version = typeof(TuningCache).Assembly
            .GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .OfType<System.Reflection.AssemblyInformationalVersionAttribute>().FirstOrDefault()?.InformationalVersion ?? "";
        int dev = version.IndexOf("-dev.", StringComparison.Ordinal);
        if (dev >= 0)
        {
            int end = version.IndexOf('+', dev);
            version = version[..dev] + (end >= 0 ? version[end..] : "");
        }

        string hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(kernels)))[..16];
        return $"{version}; kernels {hash}";
    }

    private static string OneLine(string text) => text.Replace('\n', ' ').Replace('\r', ' ').Trim();
}
