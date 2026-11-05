// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction;

/// <summary>
/// The double-precision functions of <see cref="Math"/> whose last bits the C runtime decides (exp, log, pow, sin, cos,
/// tan, tanh, atan), computed in plain IEEE arithmetic instead (fdlibm's algorithms): the same bits on every machine and
/// operating system (rule 70), within an ulp or two of the runtime's. Host code whose numbers reach a result (a table, a
/// weight, a schedule, a feature, a score) calls these in place of <see cref="Math"/>'s; the signatures are the same.
/// </summary>
public static class PortableMath
{
    /// <summary>e^x (fdlibm's exp).</summary>
    public static double Exp(double x) => CpuMath.ExpDouble(x);

    /// <summary>ln x (fdlibm's log).</summary>
    public static double Log(double x) => CpuMath.LogDouble(x);

    /// <summary>log₂ x, exact at powers of two.</summary>
    public static double Log2(double x) => CpuMath.Log2Double(x);

    /// <summary>log₁₀ x, exact at the powers of ten a double holds exactly.</summary>
    public static double Log10(double x) => CpuMath.Log10Double(x);

    /// <summary>x^y with <see cref="Math.Pow"/>'s special cases; integer powers exact where the result is.</summary>
    public static double Pow(double x, double y) => CpuMath.PowDouble(x, y);

    /// <summary>sin x (fdlibm's; within an ulp below |x| ≈ 1.6e6).</summary>
    public static double Sin(double x) => CpuMath.SinDouble(x);

    /// <summary>cos x (fdlibm's; within an ulp below |x| ≈ 1.6e6).</summary>
    public static double Cos(double x) => CpuMath.CosDouble(x);

    /// <summary>tan x (sin x / cos x of one reduction).</summary>
    public static double Tan(double x) => CpuMath.TanDouble(x);

    /// <summary>tanh x (within about 5e-15 relative).</summary>
    public static double Tanh(double x) => CpuMath.TanhDouble(x);

    /// <summary>atan x (fdlibm's).</summary>
    public static double Atan(double x) => CpuMath.AtanDouble(x);

    /// <summary>atan(y / x) over the four quadrants, with <see cref="Math.Atan2"/>'s special cases.</summary>
    public static double Atan2(double y, double x) => CpuMath.Atan2Double(y, x);
}

/// <summary>
/// The single-precision functions of <see cref="MathF"/> whose last bits the C runtime decides, computed as the CPU's kernels
/// compute them (their vector lanes and scalar tails alike): the same bits on every machine (rule 70). The signatures are
/// <see cref="MathF"/>'s.
/// </summary>
public static class PortableMathF
{
    /// <summary>e^x as the CPU's kernels compute it (within 1.2e-7 relative).</summary>
    public static float Exp(float x) => CpuMath.Exp(x);

    /// <summary>ln x as the CPU's kernels compute it.</summary>
    public static float Log(float x) => CpuMath.Log(x);

    /// <summary>x^y: the double power rounded once, <see cref="MathF.Pow"/>'s special cases.</summary>
    public static float Pow(float x, float y) => CpuMath.Pow(x, y);

    /// <summary>sin x, reduced and evaluated in double and rounded once.</summary>
    public static float Sin(float x) => CpuMath.Sin(x);

    /// <summary>cos x, reduced and evaluated in double and rounded once.</summary>
    public static float Cos(float x) => CpuMath.Cos(x);

    /// <summary>tanh x as the CPU's kernels compute it (an odd rational, within a few ulp).</summary>
    public static float Tanh(float x) => CpuMath.Tanh(x);
}
