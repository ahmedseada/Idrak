// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>The arithmetic used for large float32 matrix products.</summary>
public enum MatMulPrecision
{
    /// <summary>Float32 throughout (exact to float32 rounding).</summary>
    Float32,

    /// <summary>
    /// Operands rounded to bfloat16 and multiplied on tensor cores, sums kept in float32 (as PyTorch's bfloat16 autocast):
    /// several times faster on GPUs with bfloat16 tensor cores (NVIDIA compute capability 8.0 and
    /// newer). Tensors stay float32; devices without such hardware (and the CPU) keep computing in float32.
    /// On Vulkan devices with cooperative matrices, such products run on them where measured faster (bfloat16 operands,
    /// or 16-bit float ones scaled per row and column where the device has no bfloat16 matrices: within bfloat16's error).
    /// </summary>
    BFloat16,

    /// <summary>
    /// Operands quantized to FP8 (e4m3: 4 exponent, 3 mantissa bits) with one scale per row of the left operand and per
    /// column of the right one, multiplied on FP8 tensor cores (twice the bfloat16 rate; NVIDIA compute capability 8.9
    /// and newer) and summed in float32. Attention and devices without FP8 use bfloat16. Faster but
    /// coarser than <see cref="BFloat16"/>: compare training curves before relying on it.
    /// </summary>
    Float8,

    /// <summary>
    /// Operands rounded to IEEE 16-bit floats (scaled per row and column by powers of two, so their range is kept) and
    /// multiplied on matrix units, sums kept in float32 (plan 37, T2: a run computing in float16): on Vulkan devices with
    /// float16 cooperative matrices, where measured faster. Devices without float16 products (CUDA, whose tensor-core
    /// kernels take bfloat16, and the CPU) compute in float32.
    /// </summary>
    Float16,
}

/// <summary>
/// Chooses <see cref="MatMulPrecision"/> for matrix products: a scope on the current thread (a training step, a prompt's
/// pass), else <see cref="Chosen"/> for the whole process (IDRAK_MATMUL, or an app), else a model's own: the precision of
/// the element type its run computes in (plan 37, T2: <see cref="For"/>, which a loaded model's network applies through
/// <see cref="Prefer"/>), else float32. This is how the run's element type reaches the products: bfloat16 or float16
/// operands where the device computes them natively, float32 sums everywhere.
/// </summary>
/// <example>
/// <code>
/// using (MixedPrecision.BFloat16())
/// {
///     var loss = Losses.CrossEntropy(model.Forward(x), y);
///     loss.Backward();                  // the backward products use the same precision inside the scope
/// }
/// </code>
/// </example>
public static class MixedPrecision
{
    [ThreadStatic]
    private static MatMulPrecision? t_current;

    /// <summary>The variable naming the process's precision of matrix products over every model's own: "IDRAK_MATMUL".</summary>
    public const string EnvironmentVariable = "IDRAK_MATMUL";

    /// <summary>
    /// The precision chosen for the whole process (initially from IDRAK_MATMUL: bf16, fp16 or fp8, else null), over each
    /// model's own (<see cref="Prefer"/>); null leaves it to the model's run (its element type), else float32.
    /// </summary>
    public static MatMulPrecision? Chosen { get; set; } =
        Environment.GetEnvironmentVariable(EnvironmentVariable) switch
        {
            "bf16" or "bfloat16" => MatMulPrecision.BFloat16,
            "fp16" or "f16" or "float16" => MatMulPrecision.Float16,
            "fp8" or "float8" => MatMulPrecision.Float8,
            "fp32" or "f32" or "float32" => MatMulPrecision.Float32,
            _ => null,
        };

    /// <summary>The precision outside any scope: <see cref="Chosen"/>, else float32. Setting it sets <see cref="Chosen"/>.</summary>
    public static MatMulPrecision Default
    {
        get => Chosen ?? MatMulPrecision.Float32;
        set => Chosen = value;
    }

    /// <summary>The precision matrix products use on the current thread.</summary>
    public static MatMulPrecision Current => t_current ?? Default;

    /// <summary>Uses <paramref name="precision"/> on this thread until the returned scope is disposed.</summary>
    public static Scope Use(MatMulPrecision precision)
    {
        var previous = t_current;
        t_current = precision;
        return new Scope(previous);
    }

    /// <summary>
    /// Uses <paramref name="precision"/> (a model's own: the precision of its run's element type) on this thread until the
    /// returned scope is disposed, unless the caller chose one (a scope on this thread, or <see cref="Chosen"/>), which
    /// stays; null when it does not apply (nothing to dispose).
    /// </summary>
    public static Scope? Prefer(MatMulPrecision? precision) =>
        precision is { } p && t_current is null && Chosen is null ? Use(p) : null;

    /// <summary>
    /// The precision of products for a run computing in <paramref name="type"/> (plan 37, T2): bfloat16 →
    /// <see cref="MatMulPrecision.BFloat16"/>, float16 → <see cref="MatMulPrecision.Float16"/>, fp8 →
    /// <see cref="MatMulPrecision.Float8"/>, any other float32. Where the device has no such products they run in float32.
    /// </summary>
    public static MatMulPrecision For(ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        return ReferenceEquals(type, ElementType.BFloat16) ? MatMulPrecision.BFloat16
            : ReferenceEquals(type, ElementType.Float16) ? MatMulPrecision.Float16
            : ReferenceEquals(type, ElementType.Float8E4M3) || ReferenceEquals(type, ElementType.Float8E5M2) ? MatMulPrecision.Float8
            : MatMulPrecision.Float32;
    }

    /// <summary>
    /// Null when <paramref name="device"/> runs <see cref="MatMulPrecision.BFloat16"/> products on tensor cores; otherwise
    /// why not (they then run in float32). Loads the tensor-core kernels on first call.
    /// </summary>
    public static string? TensorCoresUnavailable(Device device) => device.Backend.TensorCoresUnavailable();

    /// <summary>
    /// True when matrix products run on bfloat16 or fp8 tensor cores where the device has them (<see cref="MatMulPrecision.BFloat16"/>
    /// or <see cref="MatMulPrecision.Float8"/>); float16 products take the cooperative-matrix path of their own.
    /// </summary>
    public static bool UsesTensorCores => Current is MatMulPrecision.BFloat16 or MatMulPrecision.Float8;

    /// <summary>Uses <see cref="MatMulPrecision.BFloat16"/> on this thread until the returned scope is disposed.</summary>
    public static Scope BFloat16() => Use(MatMulPrecision.BFloat16);

    /// <summary>Restores the previous precision when disposed.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly MatMulPrecision? _previous;

        internal Scope(MatMulPrecision? previous) => _previous = previous;

        /// <inheritdoc />
        public void Dispose() => t_current = _previous;
    }
}
