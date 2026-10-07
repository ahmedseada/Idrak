// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;
using Idrak.Layers.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// Device kernels of the plug-ins in this assembly, written as an outside package writes them: each operation is
/// declared with <see cref="PluginOperations.Register"/> (its kernel delegate and a default kernel that runs on every
/// device), and <see cref="Install"/> registers a kernel for the CPU and, for the packed format and the cache layout, a
/// SPIR-V kernel for Vulkan (<see cref="PluginShaders"/>, dispatched through <see cref="VulkanKernel.Dispatch"/>), a PTX
/// kernel for CUDA (<see cref="CudaKernel.Launch"/>) and a HIP C++ kernel for HIP (<see cref="HipKernel.Launch"/>), the
/// last two from <see cref="PluginGpuSources"/>. The
/// packed format (<see cref="RoundedWeight"/>) unpacks its weights with <see cref="Unpack"/>, the cache layout
/// (<see cref="ScaledLayout"/>) expands its rows with <see cref="Scale"/>, <see cref="PluginTests.Softplus"/> (an
/// <see cref="Autograd.Function"/>) runs <see cref="SoftplusOp"/>, and the graph operation <see cref="ScaledTanhGraphOp"/>
/// runs <see cref="ScaledTanhOp"/>.
/// </summary>
public static class PluginKernels
{
    /// <summary>values[i] = the bfloat16 half i % 2 (low first) of word i / 2 of packed, as float32, for i &lt; n.</summary>
    public delegate void UnpackPairs(Backend backend, Storage packed, Storage values, int n);

    /// <summary>output[i] = rows[i] · scales[i / width] for i &lt; n.</summary>
    public delegate void ScaleRows(Backend backend, Storage rows, Storage scales, Storage output, int n, int width);

    /// <summary>y[i] = log(1 + e^x[i]) for i &lt; n.</summary>
    public delegate void Softplus(Backend backend, Storage x, Storage y, int n);

    /// <summary>y[i] = scale · tanh(x[i]) for i &lt; n.</summary>
    public delegate void ScaledTanh(Backend backend, Storage x, Storage y, int n, float scale);

    /// <summary>The graph operation <see cref="ScaledTanhGraphOp"/> is registered as.</summary>
    public const string ScaledTanhName = "outside-scaled-tanh";

    private const int Width = 64;   // the shaders' local size

    private static readonly VulkanKernel UnpackShader = new(PluginShaders.UnpackPairs, bindings: 2, pushConstantBytes: 4, "outside_unpack_pairs", writes: 0b10);
    private static readonly VulkanKernel ScaleShader = new(PluginShaders.ScaleRows, bindings: 3, pushConstantBytes: 8, "outside_scale_rows", writes: 0b100);

    private const int Threads = 256;   // threads per block of the CUDA and HIP kernels

    // Two kernels of one PTX text and one HIP source: each device loads the module once for both.
    private static readonly CudaKernel UnpackPtx = new(PluginGpuSources.Ptx, "outside_unpack_pairs");
    private static readonly CudaKernel ScalePtx = new(PluginGpuSources.Ptx, "outside_scale_rows");
    private static readonly HipKernel UnpackHip = new(PluginGpuSources.Hip, "outside_unpack_pairs");
    private static readonly HipKernel ScaleHip = new(PluginGpuSources.Hip, "outside_scale_rows");

    /// <summary>The plug-in's CUDA kernels (the PTX test reads their parameters).</summary>
    internal static IReadOnlyList<CudaKernel> CudaKernels => [UnpackPtx, ScalePtx];

    /// <summary>The plug-in's HIP kernels.</summary>
    internal static IReadOnlyList<HipKernel> HipKernels => [UnpackHip, ScaleHip];

    /// <summary>The packed format's unpacking; its default reads the words to the host and writes the values back.</summary>
    public static PluginOperation<UnpackPairs> Unpack { get; } = PluginOperations.Register<UnpackPairs>("Outside.UnpackPairs", HostUnpack, KernelSource.Host);

    /// <summary>The cache layout's row scaling; its default composes the library's <c>GroupScaleShift</c>.</summary>
    public static PluginOperation<ScaleRows> Scale { get; } = PluginOperations.Register<ScaleRows>("Outside.ScaleRows",
        static (b, rows, scales, output, n, width) => b.GroupScaleShift(rows, scales, null, output, n, n / width, width, accumulate: false), KernelSource.Composed);

    /// <summary>Softplus's forward step; its default composes e^x, + 1 and the logarithm.</summary>
    public static PluginOperation<Softplus> SoftplusOp { get; } = PluginOperations.Register<Softplus>("Outside.Softplus", static (b, x, y, n) =>
    {
        b.Unary(UnaryOp.Exp, x, y, n);
        b.Affine(y, y, n, 1f, 1f);
        b.Unary(UnaryOp.Log, y, y, n);
    }, KernelSource.Composed);

    /// <summary>The graph operation's kernel; its default composes tanh and a scale.</summary>
    public static PluginOperation<ScaledTanh> ScaledTanhOp { get; } = PluginOperations.Register<ScaledTanh>("Outside.ScaledTanh", static (b, x, y, n, scale) =>
    {
        b.Unary(UnaryOp.Tanh, x, y, n);
        b.Affine(y, y, n, scale, 0f);
    }, KernelSource.Composed);

    /// <summary>
    /// The graph operation <see cref="ScaledTanhName"/>: scale · tanh(input 0) with the "scale" attribute, its forward step
    /// through <see cref="ScaledTanhOp"/> and its gradient scale · (1 - tanh²).
    /// </summary>
    public static GraphOp ScaledTanhGraphOp { get; } = c => ScaledTanhFunction(c.Float("scale", 1f)).Apply(c.Input(0));

    /// <summary>
    /// Registers the plug-in's device kernels: one per operation for the CPU (loops over the storages' host memory) and
    /// the packed format's and the cache layout's kernels for Vulkan (SPIR-V), CUDA (PTX) and HIP (HIP C++, where hipRTC
    /// is on the machine: <see cref="HipKernel.UnavailableReason"/>). Disposing the result removes them.
    /// </summary>
    public static IDisposable Install()
    {
        IDisposable[] handles =
        [
            Kernels.Register(Unpack, "cpu", static (_, packed, values, n) => UnpackOnHost(packed.HostMemory, values.HostMemory, n)),
            Kernels.Register(Unpack, "vulkan", static (b, packed, values, n) =>
                UnpackShader.Dispatch(b, Groups(n), 1, 1, [packed, values], MemoryMarshal.AsBytes<int>([n]))),
            Kernels.Register(Unpack, "cuda", static (b, packed, values, n) => UnpackPtx.Launch(b, Blocks(n), 1, 1, Threads, 1, 1, [packed, values, n])),
            Kernels.Register(Unpack, "hip", static (b, packed, values, n) => UnpackHip.Launch(b, Blocks(n), 1, 1, Threads, 1, 1, [packed, values, n]),
                static b => HipKernel.UnavailableReason(b) is null),
            Kernels.Register(Scale, "cpu", static (_, rows, scales, output, n, width) =>
            {
                Span<float> r = rows.HostMemory, s = scales.HostMemory, o = output.HostMemory;
                for (int i = 0; i < n; i++)
                {
                    o[i] = r[i] * s[i / width];
                }
            }),
            Kernels.Register(Scale, "vulkan", static (b, rows, scales, output, n, width) =>
                ScaleShader.Dispatch(b, Groups(n), 1, 1, [rows, scales, output], MemoryMarshal.AsBytes<int>([n, width]))),
            Kernels.Register(Scale, "cuda", static (b, rows, scales, output, n, width) =>
                ScalePtx.Launch(b, Blocks(n), 1, 1, Threads, 1, 1, [rows, scales, output, n, width])),
            Kernels.Register(Scale, "hip", static (b, rows, scales, output, n, width) =>
                ScaleHip.Launch(b, Blocks(n), 1, 1, Threads, 1, 1, [rows, scales, output, n, width]), static b => HipKernel.UnavailableReason(b) is null),
            Kernels.Register(SoftplusOp, "cpu", static (_, x, y, n) =>
            {
                Span<float> xs = x.HostMemory, ys = y.HostMemory;
                for (int i = 0; i < n; i++)
                {
                    ys[i] = MathF.Max(xs[i], 0f) + MathF.Log(1f + MathF.Exp(-MathF.Abs(xs[i])));   // stable for large |x|
                }
            }),
            Kernels.Register(ScaledTanhOp, "cpu", static (_, x, y, n, scale) =>
            {
                Span<float> xs = x.HostMemory, ys = y.HostMemory;
                for (int i = 0; i < n; i++)
                {
                    ys[i] = scale * MathF.Tanh(xs[i]);
                }
            }),
        ];
        return new Handles(handles);
    }

    /// <summary>softplus(x) through <see cref="SoftplusOp"/>, without gradients (the forward step of <see cref="PluginTests.Softplus"/>).</summary>
    public static Tensor RunSoftplus(Tensor x)
    {
        var y = Tensor.Empty(x.Shape, x.Device);
        SoftplusOp.KernelFor(x.Backend)(x.Backend, x.Storage, y.Storage, x.Size);
        return y;
    }

    /// <summary>scale · tanh(x) as a differentiable function whose forward step runs <see cref="ScaledTanhOp"/>.</summary>
    public static DifferentiableFunction ScaledTanhFunction(float scale) => Autograd.Function("outside-scaled-tanh", x =>
    {
        var y = Tensor.Empty(x[0].Shape, x[0].Device);
        ScaledTanhOp.KernelFor(x[0].Backend)(x[0].Backend, x[0].Storage, y.Storage, x[0].Size, scale);
        return y;
    }, (x, y, g) => [g * (1f - x[0].Tanh().Square()) * scale]);

    // The default kernel: the words to the host, unpacked there, the values back to the device.
    private static void HostUnpack(Backend backend, Storage packed, Storage values, int n)
    {
        var words = new float[(n + 1) / 2];
        var output = new float[n];
        backend.Download(packed, words);
        UnpackOnHost(words, output, n);
        backend.Upload(output, values);
    }

    private static void UnpackOnHost(ReadOnlySpan<float> words, Span<float> values, int n)
    {
        for (int i = 0; i < n; i++)
        {
            uint word = BitConverter.SingleToUInt32Bits(words[i >> 1]);
            values[i] = BitConverter.UInt32BitsToSingle((i & 1) == 0 ? word << 16 : word & 0xFFFF0000u);
        }
    }

    // Workgroups for n elements: one per 64, at most 65535 (the shaders loop with a grid stride past that).
    private static uint Groups(int n) => (uint)Math.Clamp((n + Width - 1) / Width, 1, 65535);

    // CUDA and HIP blocks for n elements: one per 256, at most 65535 (the kernels loop with a grid stride past that).
    private static uint Blocks(int n) => (uint)Math.Clamp((n + Threads - 1) / Threads, 1, 65535);

    private sealed class Handles(IDisposable[] handles) : IDisposable
    {
        public void Dispose()
        {
            foreach (var handle in handles)
            {
                handle.Dispose();
            }
        }
    }
}
