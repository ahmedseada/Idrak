// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction;

/// <summary>
/// How two outputs of one contract differ: the library default's and another implementation's. Each returns null when
/// they agree, otherwise a short description of the first difference. <see cref="SlotPolicy.Shadow"/> compares with
/// these; so can a conformance check.
/// </summary>
public static class Comparisons
{
    /// <summary>Equal by <see cref="EqualityComparer{T}.Default"/>.</summary>
    public static string? Exact<T>(T library, T other) =>
        EqualityComparer<T>.Default.Equals(library, other) ? null : $"{Show(library)} != {Show(other)}";

    /// <summary>The same items in the same order (each equal by <see cref="EqualityComparer{T}.Default"/>).</summary>
    public static string? Sequences<T>(IReadOnlyList<T>? library, IReadOnlyList<T>? other)
    {
        if (library is null || other is null)
        {
            return library is null && other is null ? null : $"{(library is null ? "null" : "a list")} != {(other is null ? "null" : "a list")}";
        }

        int count = Math.Min(library.Count, other.Count);
        for (int i = 0; i < count; i++)
        {
            if (!EqualityComparer<T>.Default.Equals(library[i], other[i]))
            {
                return $"item {i}: {Show(library[i])} != {Show(other[i])}";
            }
        }

        return library.Count == other.Count ? null : $"{library.Count} items != {other.Count}";
    }

    /// <summary>The same count of numbers, each within <paramref name="tolerance"/> (absolute, or relative to the larger magnitude); NaN only where the other is NaN.</summary>
    public static string? Numbers(ReadOnlySpan<double> library, ReadOnlySpan<double> other, double tolerance = 1e-9)
    {
        if (library.Length != other.Length)
        {
            return $"{library.Length} values != {other.Length}";
        }

        for (int i = 0; i < library.Length; i++)
        {
            if (!Close(library[i], other[i], tolerance))
            {
                return $"value {i}: {library[i].ToString("R", CultureInfo.InvariantCulture)} != {other[i].ToString("R", CultureInfo.InvariantCulture)}";
            }
        }

        return null;
    }

    /// <summary>The same count of numbers, each within <paramref name="tolerance"/> (see <see cref="Numbers(ReadOnlySpan{double}, ReadOnlySpan{double}, double)"/>).</summary>
    public static string? Numbers(ReadOnlySpan<float> library, ReadOnlySpan<float> other, double tolerance = 1e-5)
    {
        if (library.Length != other.Length)
        {
            return $"{library.Length} values != {other.Length}";
        }

        for (int i = 0; i < library.Length; i++)
        {
            if (!Close(library[i], other[i], tolerance))
            {
                return $"value {i}: {library[i].ToString("R", CultureInfo.InvariantCulture)} != {other[i].ToString("R", CultureInfo.InvariantCulture)}";
            }
        }

        return null;
    }

    /// <summary>The same shape and values within <paramref name="tolerance"/> (read back to the host).</summary>
    public static string? Tensors(Tensor library, Tensor other, double tolerance = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(library);
        ArgumentNullException.ThrowIfNull(other);
        if (!library.Shape.SequenceEqual(other.Shape))
        {
            return $"shape {Tensor.FormatShape(library.Shape)} != {Tensor.FormatShape(other.Shape)}";
        }

        return Numbers(library.ToArray(), other.ToArray(), tolerance);
    }

    private static bool Close(double a, double b, double tolerance) =>
        a.Equals(b) || Math.Abs(a - b) <= tolerance * Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(b)));

    private static string Show<T>(T value) => value switch
    {
        null => "null",
        string s => s.Length <= 40 ? $"\"{s}\"" : $"\"{s[..37]}...\"",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? "",
    };
}
