// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Imaging.Plugins.Fonts;

/// <summary>What the WOFF reader and writer share: their settings' variables, the measured limit and the file's numbers.</summary>
internal static class WoffSettings
{
    /// <summary>The WOFF header's size and a table directory entry's (WOFF 1.0, sections 3 and 4).</summary>
    public const int HeaderSize = 44, EntrySize = 20;

    /// <summary>An sfnt's header and table directory entry.</summary>
    public const int SfntHeaderSize = 12, SfntEntrySize = 16;

    /// <summary>The signature "wOFF", WOFF 2.0's "wOF2" and a collection's "ttcf": none is the flavor of a WOFF 1.0 font.</summary>
    public const uint Woff = 0x774F4646, Woff2 = 0x774F4632, Collection = 0x74746366;

    /// <summary>"head", the table whose fontRevision gives the WOFF header's version.</summary>
    public const uint Head = 0x68656164;

    // A variable's value, trimmed; null when unset or empty.
    private static string? Text(string variable) => Environment.GetEnvironmentVariable(variable)?.Trim() is { Length: > 0 } value ? value : null;

    /// <summary>A number of bytes from <paramref name="variable"/>: a whole number, or one followed by k, m or g (KiB, MiB, GiB; any case, an optional "b" or "ib" after it); null when unset.</summary>
    /// <exception cref="ArgumentException">Set to anything else; the message names the variable.</exception>
    public static long? Bytes(string variable)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        var span = text.AsSpan();
        long unit = 1;
        if (span.EndsWith("ib", StringComparison.OrdinalIgnoreCase))
        {
            span = span[..^2];
        }
        else if (span.EndsWith("b", StringComparison.OrdinalIgnoreCase))
        {
            span = span[..^1];
        }

        if (span.Length > 0 && char.ToLowerInvariant(span[^1]) is 'k' or 'm' or 'g')
        {
            unit = char.ToLowerInvariant(span[^1]) switch { 'k' => 1L << 10, 'm' => 1L << 20, _ => 1L << 30 };
            span = span[..^1];
        }

        return long.TryParse(span.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value > 0 && value <= long.MaxValue / unit
            ? value * unit
            : throw new ArgumentException($"{variable}={text}: a number of bytes more than 0 (such as 268435456, 256m or 1g).", variable);
    }

    /// <summary>A whole number from <paramref name="min"/> to <paramref name="max"/> from <paramref name="variable"/>; null when unset.</summary>
    /// <exception cref="ArgumentException">Set to anything else; the message names the variable.</exception>
    public static int? Int(string variable, int min, int max)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value
            : throw new ArgumentException($"{variable}={text}: a whole number from {min} to {max}.", variable);
    }

    /// <summary>
    /// The default limit on a font inside a WOFF: a quarter of the memory free now (what the CPU device reports, within
    /// <see cref="ComputeResources.CpuMemoryLimit"/> when the user set one), at most the largest array .NET makes.
    /// </summary>
    public static long MeasuredLimit()
    {
        long free = Device.Cpu.Backend.AvailableMemory() ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (ComputeResources.CpuMemoryLimit is { } limit)
        {
            free = Math.Min(free, limit);
        }

        return Math.Min(Array.MaxLength, Math.Max(1L << 20, free) / 4);
    }

    /// <summary>A table tag as text: its four characters, or its number when they are not printable ASCII.</summary>
    public static string Tag(uint tag)
    {
        for (int shift = 24; shift >= 0; shift -= 8)
        {
            if (((tag >> shift) & 0xFF) is < 0x20 or > 0x7E)
            {
                return string.Create(CultureInfo.InvariantCulture, $"0x{tag:X8}");
            }
        }

        return string.Create(4, tag, static (chars, t) =>
        {
            for (int i = 0; i < 4; i++)
            {
                chars[i] = (char)((t >> (24 - 8 * i)) & 0xFF);
            }
        });
    }

    /// <summary><paramref name="length"/> rounded up to a whole number of 4-byte words (how sfnt and WOFF lay tables out).</summary>
    public static long Padded(long length) => (length + 3) & ~3L;
}
