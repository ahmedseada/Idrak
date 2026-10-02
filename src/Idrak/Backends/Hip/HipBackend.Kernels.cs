// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Backends.Hip.HipRuntime;

namespace Idrak.Backends.Hip;

// The operations with a kernel of their own (HipKernels). The kernels are compiled the first time one is needed, for
// this device's architecture, block size and runtime, or read from the kernel cache (HipKernelCache); when they cannot
// be (no hipRTC, a compiler error, limits the runtime reports implausibly, IDRAK_HIP_KERNELS=0) every operation here
// takes the host fallback instead, and KernelsUnavailableReason says why.
internal sealed unsafe partial class HipBackend
{
    private readonly object _kernelGate = new();
    private Dictionary<string, IntPtr>? _kernels;
    private bool _kernelsTried;
    private int _blockSize;

    /// <summary>Why this device runs no kernels of its own (null when they loaded, or have not been needed yet).</summary>
    public string? KernelsUnavailableReason { get; private set; }

    /// <summary>Where the kernels came from: "compiled", "cached", or "" when they did not load or were not needed yet.</summary>
    public string KernelsOrigin { get; private set; } = "";

    /// <summary>Threads per block of this device's kernels (0 until they load).</summary>
    internal int BlockSize => _blockSize;

    /// <summary>The kernel cache this backend uses (tests point it elsewhere before the kernels load).</summary>
    internal HipKernelCache KernelCache { get; set; } = HipKernelCache.FromEnvironment();

    /// <summary>Whether this device's kernels loaded (compiling or reading them on first use).</summary>
    internal bool HasKernels => Kernels() is not null;

    private Dictionary<string, IntPtr>? Kernels()
    {
        lock (_kernelGate)
        {
            if (!_kernelsTried)
            {
                _kernelsTried = true;
                try
                {
                    _kernels = LoadKernels();
                }
                catch (HipException e)
                {
                    KernelsUnavailableReason = e.Message;
                }
            }

            return _kernels;
        }
    }

    private Dictionary<string, IntPtr>? LoadKernels()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_HIP_KERNELS") is "0" or "false")
        {
            KernelsUnavailableReason = "turned off by IDRAK_HIP_KERNELS=0";
            return null;
        }

        var limits = _device.Limits;
        if (limits.Problem() is { } problem)
        {
            KernelsUnavailableReason = $"{problem}; the kernels stay off";
            return null;
        }

        int block = HipKernels.BlockSizeFor(limits);
        string architecture = limits.Architecture;
        string[] options = HipRtc.Options(architecture, block);
        string versions = $"runtime {_device.RuntimeVersion}, driver {_device.DriverVersion}, compiler {HipRtc.Version}";
        string identity = $"{limits.Name}/{limits.Uuid}/{limits.PciAddress}";
        string key = HipKernelCache.Key(HipKernels.Source, options, architecture, identity, versions);

        MakeCurrent();
        if (KernelCache.Load(key, architecture) is { } cached)
        {
            if (TryLoadModule(cached, out var module, out _))
            {
                KernelsOrigin = "cached";
                _blockSize = block;
                return Functions(module);
            }

            KernelCache.Delete(key, architecture);                         // refused by this runtime: compile again
        }

        if (HipRtc.UnavailableReason is { Length: > 0 } missing)
        {
            KernelsUnavailableReason = missing;
            return null;
        }

        byte[] code = HipRtc.Compile(HipKernels.Source, "idrak_kernels.hip", options);
        if (!TryLoadModule(code, out var compiled, out string error))
        {
            throw new HipException($"the runtime refused the compiled kernels: {error}");
        }

        KernelCache.Store(key, architecture, code);
        KernelsOrigin = "compiled";
        _blockSize = block;
        return Functions(compiled);
    }

    private static bool TryLoadModule(byte[] code, out IntPtr module, out string error)
    {
        IntPtr loaded;
        fixed (byte* c = code)
        {
            int result = hipModuleLoadData(&loaded, c);
            module = loaded;
            error = result == Success ? "" : Describe(result);
            return result == Success;
        }
    }

    private Dictionary<string, IntPtr> Functions(IntPtr module)
    {
        var functions = new Dictionary<string, IntPtr>();
        foreach (string name in HipKernels.Parameters.Keys)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(name + "\0");
            IntPtr function;
            fixed (byte* n = bytes)
            {
                Check(hipModuleGetFunction(&function, module, n), $"hipModuleGetFunction({name})");
            }

            functions[name] = function;
        }

        return functions;
    }

    // The kernel `name`, or null when the kernels are unavailable.
    private IntPtr? Kernel(string name) => Kernels() is { } kernels ? kernels[name] : null;

    /// <summary>
    /// Launches kernel <paramref name="name"/> with blocks of <see cref="BlockSize"/> threads. Each argument sits in a
    /// 64-bit slot; the runtime copies each parameter's size from the start of its slot, the value itself on
    /// little-endian hosts (the kernel's parameter list is checked against the arguments first).
    /// </summary>
    private void Launch(string name, IntPtr function, uint gridX, uint gridY, params ReadOnlySpan<ulong> args)
    {
        if (HipKernels.Parameters[name].Length != args.Length)
        {
            throw new InvalidOperationException($"Kernel {name} declares {HipKernels.Parameters[name].Length} parameters but was launched with {args.Length} arguments.");
        }

        MakeCurrent();
        ulong* values = stackalloc ulong[args.Length];
        void** pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            values[i] = args[i];
            pointers[i] = &values[i];
        }

        Check(hipModuleLaunchKernel(function, gridX, gridY, 1, (uint)_blockSize, 1, 1, 0, _stream, pointers, null), $"hipModuleLaunchKernel({name})");
    }

    // An element-wise kernel over n values (grid-stride, so the grid stays within the reported limit). False: no kernels.
    private bool Elements(string name, long n, params ReadOnlySpan<ulong> args)
    {
        if (Kernel(name) is not { } function)
        {
            return false;
        }

        if (n > 0)
        {
            long blocks = Math.Min((n + _blockSize - 1) / _blockSize, _device.Limits.MaxGridDimX);
            Launch(name, function, (uint)blocks, 1, args);
        }

        return true;
    }

    // A kernel with one block per row. False: no kernels, or more rows than a grid holds.
    private bool Rows(string name, int rows, params ReadOnlySpan<ulong> args)
    {
        if (rows > _device.Limits.MaxGridDimX || Kernel(name) is not { } function)
        {
            return false;
        }

        if (rows > 0)
        {
            Launch(name, function, (uint)rows, 1, args);
        }

        return true;
    }

    private static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    private static ulong U(int value) => (uint)value;

    public override void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        string name = op switch
        {
            BinaryOp.Add => "add_f32",
            BinaryOp.Sub => "sub_f32",
            BinaryOp.Mul => "mul_f32",
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        if (!Elements(name, n, P(a), P(b), P(c), U(n)))
        {
            base.Binary(op, a, b, c, n);
        }
    }

    public override void Unary(UnaryOp op, Storage x, Storage y, int n)
    {
        string? name = op switch
        {
            UnaryOp.Relu => "relu_f32",
            UnaryOp.Sigmoid => "sigmoid_f32",
            UnaryOp.Tanh => "tanh_f32",
            UnaryOp.Square => "square_f32",
            UnaryOp.Abs => "abs_f32",
            UnaryOp.Exp => "exp_f32",
            UnaryOp.Log => "log_f32",
            _ => null,                                                    // GELU: the host fallback for now
        };
        if (name is null || !Elements(name, n, P(x), P(y), U(n)))
        {
            base.Unary(op, x, y, n);
        }
    }

    public override void Axpy(Storage x, Storage y, int n, float alpha)
    {
        if (!Elements("axpy_f32", n, P(x), P(y), U(n), F(alpha)))
        {
            base.Axpy(x, y, n, alpha);
        }
    }

    public override void Affine(Storage x, Storage y, int n, float alpha, float beta)
    {
        if (!Elements("affine_f32", n, P(x), P(y), U(n), F(alpha), F(beta)))
        {
            base.Affine(x, y, n, alpha, beta);
        }
    }

    public override void MulAdd(Storage a, Storage b, Storage c, int n)
    {
        if (!Elements("muladd_f32", n, P(a), P(b), P(c), U(n)))
        {
            base.MulAdd(a, b, c, n);
        }
    }

    public override void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols)
    {
        if (cols <= 0 || !Elements("add_rowvec_f32", (long)rows * cols, P(a), P(v), P(c), U(rows), U(cols)))
        {
            base.AddRowVector(a, v, c, rows, cols);
        }
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        if (!Rows("rms_norm_f32", rows, P(x), P(y), P(inv), U(rows), U(cols), F(eps)))
        {
            base.RmsNorm(x, y, inv, rows, cols, eps);
        }
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (!Rows("rms_norm_affine_f32", rows, P(x), P(gain), P(y), U(rows), U(cols), F(eps), F(offset)))
        {
            base.RmsNormAffine(x, gain, y, rows, cols, eps, offset);
        }
    }

    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        // One thread per output: column blocks along x, rows along y (within the reported grid limits).
        if (Kernel("int8_matmul_f32") is { } function)
        {
            long columnBlocks = ((long)n + _blockSize - 1) / _blockSize;
            if (m <= _device.Limits.MaxGridDimY && columnBlocks <= _device.Limits.MaxGridDimX)
            {
                if (m > 0 && n > 0)
                {
                    int stride = (n + 3) / 4 * 4;                             // bytes per weight row, as packed
                    Launch("int8_matmul_f32", function, (uint)columnBlocks, (uint)m, P(x), P(q), P(scales), P(y), U(m), U(n), U(k), U(stride));
                }

                return;
            }
        }

        base.Int8MatMul(x, q, scales, y, m, n, k);
    }
}
