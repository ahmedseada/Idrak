// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The comparisons the kit makes between an implementation and the library default: floating-point values within a
/// tolerance relative to the expected value (absolute below 1), token ids and text exactly. Each returns null when the
/// two agree, else a message naming the first difference, so a test framework can print it as it is.
/// </summary>
public static class Comparisons
{
    /// <summary>
    /// Whether <paramref name="actual"/> agrees with <paramref name="expected"/>: equal bit for bit, both NaN, or
    /// |actual - expected| ≤ <paramref name="tolerance"/> · max(1, |expected|).
    /// </summary>
    public static bool Close(float expected, float actual, float tolerance) =>
        BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual)
        || (float.IsNaN(expected) && float.IsNaN(actual))
        || MathF.Abs(expected - actual) <= tolerance * MathF.Max(1f, MathF.Abs(expected));

    /// <summary>
    /// Null when every element agrees (<see cref="Close(float, float, float)"/>); else the first element that does not,
    /// how many do not, and the largest difference.
    /// </summary>
    public static string? Difference(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance)
    {
        if (expected.Length != actual.Length)
        {
            return $"{actual.Length} values, {expected.Length} expected";
        }

        int first = -1, count = 0;
        float largest = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!Close(expected[i], actual[i], tolerance))
            {
                first = first < 0 ? i : first;
                count++;
                float difference = MathF.Abs(expected[i] - actual[i]);
                largest = float.IsNaN(difference) ? float.NaN : float.IsNaN(largest) ? largest : MathF.Max(largest, difference);
            }
        }

        return count == 0 ? null
            : $"element {first} is {Format(actual[first])}, expected {Format(expected[first])} (tolerance {tolerance:G3}); "
              + $"{count} of {expected.Length} differ, by up to {Format(largest)}";
    }

    /// <summary>The same as <see cref="Difference(ReadOnlySpan{float}, ReadOnlySpan{float}, float)"/> for doubles.</summary>
    public static string? Difference(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tolerance)
    {
        if (expected.Length != actual.Length)
        {
            return $"{actual.Length} values, {expected.Length} expected";
        }

        for (int i = 0; i < expected.Length; i++)
        {
            bool close = expected[i].Equals(actual[i]) || Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i]));
            if (!close)
            {
                return $"element {i} is {actual[i]:R}, expected {expected[i]:R} (tolerance {tolerance:G3})";
            }
        }

        return null;
    }

    /// <summary>Null when the ids are the same; else the first position where they differ.</summary>
    public static string? Difference(IReadOnlyList<int> expected, IReadOnlyList<int> actual)
    {
        for (int i = 0; i < Math.Min(expected.Count, actual.Count); i++)
        {
            if (expected[i] != actual[i])
            {
                return $"id {i} is {actual[i]}, expected {expected[i]}";
            }
        }

        return expected.Count == actual.Count ? null : $"{actual.Count} ids, {expected.Count} expected";
    }

    /// <summary>Null when the texts are the same; else the first character where they differ, with the text around it.</summary>
    public static string? Difference(string expected, string actual)
    {
        if (string.Equals(expected, actual, StringComparison.Ordinal))
        {
            return null;
        }

        int i = 0;
        while (i < expected.Length && i < actual.Length && expected[i] == actual[i])
        {
            i++;
        }

        return $"text differs at character {i}: \"{Around(actual, i)}\", expected \"{Around(expected, i)}\" (lengths {actual.Length} and {expected.Length})";
    }

    // Up to 12 characters either side of position i, escaped.
    private static string Around(string text, int i)
    {
        int start = Math.Max(0, i - 12), end = Math.Min(text.Length, i + 12);
        string part = text[start..end].Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\n", "\\n", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal).Replace("\t", "\\t", StringComparison.Ordinal);
        return (start > 0 ? "…" : "") + part + (end < text.Length ? "…" : "");
    }

    private static string Format(float value) => value.ToString("G9", System.Globalization.CultureInfo.InvariantCulture);
}
