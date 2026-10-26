// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Text;
using System.Diagnostics;
using System.Globalization;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// The machine's memory as the operating system reports it (Linux and Android: /proc/meminfo's MemTotal and MemAvailable),
/// and the limit of the control group the process runs in, where one applies (a container: cgroup v2 memory.max, v1
/// memory.limit_in_bytes, the lowest on the way from the process's group to the hierarchy's root), with what that group
/// uses (less its inactive file cache, which the kernel reclaims first). Null where nothing is reported (another
/// system, a file the process may not read).
/// </summary>
/// <param name="Total">MemTotal, in bytes.</param>
/// <param name="Available">MemAvailable, in bytes (what can be taken without swapping).</param>
/// <param name="Limit">The lowest control group limit above the process, in bytes (none: null).</param>
/// <param name="LimitUsage">What the group at that limit uses, less its inactive file cache, in bytes.</param>
/// <param name="LimitSource">"cgroup v2" or "cgroup v1", with the limit.</param>
internal readonly record struct SystemMemoryReport(long? Total, long? Available, long? Limit, long? LimitUsage, string? LimitSource)
{
    /// <summary>What the process may use in all: the machine's memory, or the group's limit where that is lower.</summary>
    public long? ProcessTotal => Min(Total, Limit);

    /// <summary>What the process may still take: MemAvailable, or what the group's limit leaves where that is lower.</summary>
    public long? ProcessAvailable => Min(Available, Limit is { } limit && LimitUsage is { } used ? Math.Max(0, limit - used) : null);

    private static long? Min(long? a, long? b) => a is { } x ? (b is { } y ? Math.Min(x, y) : x) : b;
}

/// <summary>Reads <see cref="SystemMemoryReport"/>: the files parsed as bytes (rule 74), the groups' directories found once.</summary>
internal static class SystemMemory
{
    // Limits at or above this are "no limit" (cgroup v1 reports 0x7FFFFFFFFFFFF000, rounded to the page size).
    private const long NoLimit = 1L << 62;

    private static readonly Lazy<(string Directory, string Mount, bool V2)[]> s_groups = new(() => GroupsUnder(""));

    /// <summary>Whether the system reports its memory here (Linux and Android).</summary>
    public static bool Supported => OperatingSystem.IsLinux() || OperatingSystem.IsAndroid();

    /// <summary>The report now (empty where <see cref="Supported"/> is false or nothing could be read).</summary>
    public static SystemMemoryReport Read() => Supported ? Read("", s_groups.Value) : default;

    /// <summary>
    /// The report from the files under <paramref name="root"/> ("" for the system's own; tests pass a folder holding
    /// proc/meminfo, proc/self/cgroup, proc/self/mountinfo and the groups' files at their mount points).
    /// </summary>
    internal static SystemMemoryReport Read(string root) => Read(root, GroupsUnder(root));

    private static SystemMemoryReport Read(string root, (string Directory, string Mount, bool V2)[] groups)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(16 << 10);
        try
        {
            var (total, available) = ReadAll(root + "/proc/meminfo", buffer) is var n and > 0 ? ParseMeminfo(buffer.AsSpan(0, n)) : (null, null);
            long? limit = null, usage = null;
            string? source = null;
            foreach (var (directory, mount, v2) in groups)
            {
                // Each group from the process's own up to the hierarchy's mount: the lowest limit binds, with what its
                // group uses.
                for (string dir = directory; ; dir = Path.GetDirectoryName(dir) ?? mount)
                {
                    if (ReadAll(Path.Combine(dir, v2 ? "memory.max" : "memory.limit_in_bytes"), buffer) is var read and > 0
                        && ParseBytes(buffer.AsSpan(0, read)) is { } bytes && bytes < NoLimit && (limit is null || bytes < limit))
                    {
                        limit = bytes;
                        source = v2 ? "cgroup v2" : "cgroup v1";
                        usage = GroupUsage(dir, v2, buffer);
                    }

                    if (dir.Length <= mount.Length)
                    {
                        break;
                    }
                }
            }

            return new SystemMemoryReport(total, available, limit, usage, source);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // memory.current (v2) or memory.usage_in_bytes (v1) less the inactive file cache of memory.stat, or null.
    private static long? GroupUsage(string dir, bool v2, byte[] buffer)
    {
        if (ReadAll(Path.Combine(dir, v2 ? "memory.current" : "memory.usage_in_bytes"), buffer) is not (var n and > 0)
            || ParseBytes(buffer.AsSpan(0, n)) is not { } used)
        {
            return null;
        }

        long inactive = ReadAll(Path.Combine(dir, "memory.stat"), buffer) is var s and > 0
            ? StatValue(buffer.AsSpan(0, s), v2 ? "inactive_file"u8 : "total_inactive_file"u8) ?? 0 : 0;
        return Math.Max(0, used - Math.Min(inactive, used));
    }

    /// <summary>
    /// The memory control groups' directories of this process under <paramref name="root"/>: for each hierarchy that has
    /// memory in /proc/self/cgroup (v2: "0::path"; v1: a line naming the memory controller), its mount point from
    /// /proc/self/mountinfo joined with the process's path below the mount's root.
    /// </summary>
    internal static (string Directory, string Mount, bool V2)[] GroupsUnder(string root)
    {
        try
        {
            string cgroupFile = root + "/proc/self/cgroup", mountFile = root + "/proc/self/mountinfo";
            if (!File.Exists(cgroupFile) || !File.Exists(mountFile))
            {
                return [];
            }

            return Groups(File.ReadAllBytes(cgroupFile), File.ReadAllBytes(mountFile), root);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    /// <summary>The directories of <see cref="GroupsUnder"/> from the two files' bytes.</summary>
    internal static (string Directory, string Mount, bool V2)[] Groups(ReadOnlySpan<byte> cgroup, ReadOnlySpan<byte> mountinfo, string root)
    {
        string? v1Path = null, v2Path = null;
        foreach (var range in cgroup.Split((byte)'\n'))
        {
            var line = cgroup[range];
            int first = line.IndexOf((byte)':');
            int second = first < 0 ? -1 : line[(first + 1)..].IndexOf((byte)':');
            if (second < 0)
            {
                continue;
            }

            var controllers = line.Slice(first + 1, second);
            string path = System.Text.Encoding.UTF8.GetString(line[(first + second + 2)..].TrimEnd((byte)'\r'));
            if (controllers.IsEmpty && line[..first].SequenceEqual("0"u8))
            {
                v2Path = path;
            }
            else if (HasItem(controllers, "memory"u8))
            {
                v1Path = path;
            }
        }

        var found = new List<(string, string, bool)>();
        Span<Range> fields = stackalloc Range[5];
        Span<Range> parts = stackalloc Range[3];
        foreach (var range in mountinfo.Split((byte)'\n'))
        {
            // id parent major:minor root mount-point options [optional fields] - type source super-options
            var line = mountinfo[range];
            int dash = line.IndexOf(" - "u8);
            if (dash < 0)
            {
                continue;
            }

            var head = line[..dash];
            var tail = line[(dash + 3)..];
            if (Words(head, fields) < 5 || Words(tail, parts) < 1)
            {
                continue;
            }

            var type = tail[parts[0]];
            bool v2 = type.SequenceEqual("cgroup2"u8);
            bool v1 = type.SequenceEqual("cgroup"u8) && Words(tail, parts) >= 3 && HasItem(tail[parts[2]], "memory"u8);
            string? path = v2 ? v2Path : v1 ? v1Path : null;
            if (path is null)
            {
                continue;
            }

            string mountRoot = System.Text.Encoding.UTF8.GetString(head[fields[3]]);
            string mount = root + System.Text.Encoding.UTF8.GetString(head[fields[4]]).TrimEnd('/');
            string below = mountRoot == "/" ? path : path.StartsWith(mountRoot, StringComparison.Ordinal) ? path[mountRoot.Length..] : "/";
            below = below.TrimEnd('/');
            found.Add((below.Length == 0 ? mount : mount + (below[0] == '/' ? "" : "/") + below, mount, v2));
            if (v2)
            {
                v2Path = null;                                                     // one mount per hierarchy
            }
            else
            {
                v1Path = null;
            }
        }

        return [.. found];
    }

    // The ranges of the first destination.Length space-separated words of `line`; returns how many were found.
    private static int Words(ReadOnlySpan<byte> line, Span<Range> destination)
    {
        int count = 0, at = 0;
        while (count < destination.Length)
        {
            while (at < line.Length && line[at] == (byte)' ')
            {
                at++;
            }

            if (at >= line.Length)
            {
                break;
            }

            int end = line[at..].IndexOf((byte)' ') is var space and >= 0 ? at + space : line.Length;
            destination[count++] = at..end;
            at = end;
        }

        return count;
    }

    // Whether a comma list ("rw,memory") holds `item`.
    private static bool HasItem(ReadOnlySpan<byte> list, ReadOnlySpan<byte> item)
    {
        foreach (var range in list.Split((byte)','))
        {
            if (list[range].SequenceEqual(item))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>MemTotal and MemAvailable of /proc/meminfo ("MemTotal:   16318412 kB"), in bytes; null where a line is missing.</summary>
    internal static (long? Total, long? Available) ParseMeminfo(ReadOnlySpan<byte> text)
    {
        long? total = null, available = null;
        foreach (var range in text.Split((byte)'\n'))
        {
            var line = text[range];
            if (line.StartsWith("MemTotal:"u8))
            {
                total = Kibibytes(line["MemTotal:"u8.Length..]);
            }
            else if (line.StartsWith("MemAvailable:"u8))
            {
                available = Kibibytes(line["MemAvailable:"u8.Length..]);
            }
        }

        return (total, available);
    }

    // "   16318412 kB" → bytes.
    private static long? Kibibytes(ReadOnlySpan<byte> value)
    {
        value = value.Trim((byte)' ').Trim((byte)'\t');
        if (!Utf8Parser.TryParse(value, out long number, out int used) || number < 0)
        {
            return null;
        }

        var unit = value[used..].Trim((byte)' ').TrimEnd((byte)'\r');
        return unit.SequenceEqual("kB"u8) ? number << 10 : unit.IsEmpty ? number : null;
    }

    /// <summary>A control group's number file ("14341881856\n"), in bytes; null for "max" (no limit) or anything else.</summary>
    internal static long? ParseBytes(ReadOnlySpan<byte> text)
    {
        text = text.TrimEnd("\r\n "u8);
        return Utf8Parser.TryParse(text, out long value, out int used) && used == text.Length && value >= 0 ? value : null;
    }

    /// <summary>The value of <paramref name="key"/> in a memory.stat ("inactive_file 3399880704" lines), or null.</summary>
    internal static long? StatValue(ReadOnlySpan<byte> stat, ReadOnlySpan<byte> key)
    {
        foreach (var range in stat.Split((byte)'\n'))
        {
            var line = stat[range];
            if (line.Length > key.Length && line.StartsWith(key) && line[key.Length] == (byte)' ')
            {
                return ParseBytes(line[(key.Length + 1)..]);
            }
        }

        return null;
    }

    // Reads a small system file whole into `buffer` (its first buffer.Length bytes); 0 when it is missing or unreadable.
    private static int ReadAll(string path, byte[] buffer)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, bufferSize: 0);
            int total = 0;
            while (total < buffer.Length && stream.Read(buffer.AsSpan(total)) is var read and > 0)
            {
                total += read;
            }

            return total;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            return 0;
        }
    }

    /// <summary>"15.7 GiB", "143.8 MiB": a size for listings.</summary>
    internal static string Format(long bytes) => bytes switch
    {
        >= 1L << 30 => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 30):F1} GiB"),
        >= 1L << 20 => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 20):F1} MiB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes >> 10} KiB"),
    };

    /// <summary>A timestamp for the refresh of a cached report (<see cref="Stopwatch.GetTimestamp"/>).</summary>
    internal static long Now => Stopwatch.GetTimestamp();
}
