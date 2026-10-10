// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Plug-ins' own kernels (CudaKernel): each PTX text JIT-compiled once on this device, each entry looked up once, then
// launched like the library's kernels (on the stream, through the stream gate, profiled under the kernel's name).
internal sealed unsafe partial class CudaBackend
{
    private readonly object _pluginGate = new();

    // Loaded modules by PTX text, and functions by kernel (reference: two kernels of one text share the module).
    private readonly Dictionary<string, IntPtr> _pluginModules = new(StringComparer.Ordinal);
    private readonly Dictionary<CudaKernel, IntPtr> _pluginFunctions = new(ReferenceEqualityComparer.Instance);

    /// <summary>Queues a plug-in's kernel (its arguments already checked against its parameters).</summary>
    internal void Launch(CudaKernel kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ,
        ReadOnlySpan<KernelArgument> arguments, uint sharedBytes)
    {
        long threads = (long)blockX * blockY * blockZ;
        if (threads > _limits.MaxThreadsPerBlock || (threads == 0 && gridX * (ulong)gridY * gridZ != 0))
        {
            throw new ArgumentOutOfRangeException(nameof(blockX),
                $"CUDA kernel '{kernel.Name}': blocks of {blockX} × {blockY} × {blockZ} threads; {Name} runs 1 to {_limits.MaxThreadsPerBlock} a block.");
        }

        if (gridX > int.MaxValue || gridY > (uint)_limits.MaxGridY || gridZ > (uint)_limits.MaxGridZ)
        {
            throw new ArgumentOutOfRangeException(nameof(gridX),
                $"CUDA kernel '{kernel.Name}': {gridX} × {gridY} × {gridZ} blocks exceed {Name}'s {int.MaxValue} × {_limits.MaxGridY} × {_limits.MaxGridZ}.");
        }

        int sharedLimit = Math.Max(_limits.SharedPerBlock, _limits.SharedPerBlockOptin);
        if (sharedBytes > sharedLimit)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedBytes), sharedBytes,
                $"CUDA kernel '{kernel.Name}': {sharedBytes} bytes of shared memory; {Name} allows {sharedLimit} a block.");
        }

        Span<ulong> values = arguments.Length <= StackArguments ? stackalloc ulong[arguments.Length] : new ulong[arguments.Length];
        for (int i = 0; i < arguments.Length; i++)
        {
            values[i] = arguments[i].Kind == KernelArgumentKind.Pointer ? PluginPointer(kernel, i, arguments[i]) : arguments[i].Bits;
        }

        if (gridX == 0 || gridY == 0 || gridZ == 0)
        {
            return;
        }

        IntPtr function = PluginFunction(kernel);
        if (sharedBytes > _limits.SharedPerBlock)
        {
            MakeCurrent();
            Check(cuFuncSetAttribute(function, FunctionAttributeMaxDynamicSharedSizeBytes, (int)sharedBytes), $"cuFuncSetAttribute({kernel.Name})");
        }

        if (_profile is not null)
        {
            _profileLabel = kernel.Name;                                          // consumed by the profiled launch
        }

        t_sharedBytes = sharedBytes;
        LaunchGrid(function, gridX, gridY, gridZ, blockX, blockY, blockZ, values);
    }

    // The device address argument i stands for: a live storage of this device, at its offset.
    private ulong PluginPointer(CudaKernel kernel, int i, KernelArgument argument)
    {
        if (argument.Storage is not CudaStorage storage || !ReferenceEquals(storage.Backend, this))
        {
            throw new ArgumentException($"CUDA kernel '{kernel.Name}': storage argument {i} is not on {Name}.", "arguments");
        }

        if (storage.Pointer == 0)
        {
            throw new InvalidOperationException($"CUDA kernel '{kernel.Name}': storage argument {i} was evicted (Backend.Evict); restore it first.");
        }

        return storage.Pointer + (ulong)argument.Offset * sizeof(float);
    }

    // The kernel's function on this device: its module loaded (once per PTX text) and its entry looked up, once.
    private IntPtr PluginFunction(CudaKernel kernel)
    {
        if (kernel.LastFunction is { } last && ReferenceEquals(last.Owner, this))
        {
            return last.Function;
        }

        lock (_pluginGate)
        {
            if (!_pluginFunctions.TryGetValue(kernel, out IntPtr function))
            {
                MakeCurrent();
                if (!_pluginModules.TryGetValue(kernel.Ptx, out IntPtr module))
                {
                    module = LoadModule(kernel.Ptx, $"the PTX of kernel '{kernel.Name}'");
                    _pluginModules[kernel.Ptx] = module;
                }

                byte[] entry = Encoding.ASCII.GetBytes(kernel.Entry + "\0");
                fixed (byte* p = entry)
                {
                    Check(cuModuleGetFunction(out function, module, p), $"cuModuleGetFunction({kernel.Entry})");
                }

                _pluginFunctions[kernel] = function;
            }

            kernel.LastFunction = new(this, function);
            return function;
        }
    }
}
