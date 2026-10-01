// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

/// <summary>The arithmetic used for large float32 matrix products.</summary>
public enum MatMulPrecision
{
    /// <summary>Float32 throughout (exact to float32 rounding).</summary>
    Float32,

    /// <summary>
    /// Operands rounded to bfloat16 and multiplied on tensor cores, sums kept in float32 (as PyTorch's bfloat16 autocast):
    /// several times faster on GPUs with bfloat16 tensor cores (NVIDIA compute capability 8.0 and newer: RTX 30xx and
    /// later, A100, H100). Tensors stay float32; devices without such hardware (and the CPU) keep computing in float32.
    /// </summary>
    BFloat16,

    /// <summary>
    /// Operands quantized to FP8 (e4m3: 4 exponent, 3 mantissa bits) with one scale per row of the left operand and per
    /// column of the right one, multiplied on FP8 tensor cores (twice the bfloat16 rate; NVIDIA compute capability 8.9
    /// and newer: RTX 40xx / 50xx, H100) and summed in float32. Attention and devices without FP8 use bfloat16. Faster but
    /// coarser than <see cref="BFloat16"/>: compare training curves before relying on it.
    /// </summary>
    Float8,
}

/// <summary>
/// Chooses <see cref="MatMulPrecision"/> for matrix products: <see cref="Default"/> for the whole process, or a scope on
/// the current thread for a training step.
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

    /// <summary>The precision outside any scope (initially from IDRAK_MATMUL=bf16 or fp8, else float32).</summary>
    public static MatMulPrecision Default { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_MATMUL") switch
        {
            "bf16" or "bfloat16" => MatMulPrecision.BFloat16,
            "fp8" or "float8" => MatMulPrecision.Float8,
            _ => MatMulPrecision.Float32,
        };

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
    /// Null when <paramref name="device"/> runs <see cref="MatMulPrecision.BFloat16"/> products on tensor cores; otherwise
    /// why not (they then run in float32). Loads the tensor-core kernels on first call.
    /// </summary>
    public static string? TensorCoresUnavailable(Device device) => device.Type == DeviceType.Cpu
        ? "the CPU computes matrix products in float32"
        : ((Backends.Cuda.CudaBackend)device.Backend).TensorCoresUnavailable();

    /// <summary>True when matrix products run on tensor cores (any precision other than float32).</summary>
    internal static bool UsesTensorCores => Current != MatMulPrecision.Float32;

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
