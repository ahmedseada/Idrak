// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Cuda;

/// <summary>
/// The PTX target the tensor-core modules declare (the described lane's integer products), and reading the target a
/// module declares: a device loads such a module when the compute capability it reports reaches that target, never by a
/// list of cards.
/// </summary>
internal static partial class PtxKernels
{
    /// <summary>
    /// The PTX target the tensor-core modules declare (<c>.target</c>): what their <c>mma.sync</c> shapes and
    /// <c>ldmatrix</c> need, as the PTX ISA defines them.
    /// </summary>
    internal const string TensorCoreTarget = "sm_80";

    /// <summary>
    /// The compute capability a PTX target names, as one number ("sm_89" → 89, "sm_90a" → 90), or -1 when the text is not
    /// "sm_" followed by digits.
    /// </summary>
    internal static int TargetNumber(ReadOnlySpan<char> target)
    {
        if (!target.StartsWith("sm_", StringComparison.Ordinal))
        {
            return -1;
        }

        int end = SkipDigits(target, 3);
        return end > 3 && int.TryParse(target[3..end], NumberStyles.None, CultureInfo.InvariantCulture, out int number) ? number : -1;
    }

    /// <summary>The target a module's PTX declares (the word after <c>.target</c>), or an empty span when it declares none.</summary>
    internal static ReadOnlySpan<char> DeclaredTarget(string source)
    {
        int at = source.AsSpan().IndexOf(".target", StringComparison.Ordinal);
        if (at < 0)
        {
            return [];
        }

        int start = SkipWhiteSpace(source, at + ".target".Length);
        return source.AsSpan(start, SkipWord(source, start) - start);
    }
}
