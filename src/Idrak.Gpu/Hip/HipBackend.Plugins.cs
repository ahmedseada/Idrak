// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// Plug-ins' own kernels (HipKernel): each source compiled once on this device with hipRTC (or read from the kernel
// cache, keyed like the library's own kernels), each function looked up once, then launched on the stream.
internal sealed unsafe partial class HipBackend
{
    private readonly object _pluginGate = new();

    // Loaded modules by cache key (source, options, architecture, device, versions), and functions by kernel.
    private readonly Dictionary<string, IntPtr> _pluginModules = new(StringComparer.Ordinal);
    private readonly Dictionary<HipKernel, IntPtr> _pluginFunctions = new(ReferenceEqualityComparer.Instance);

    /// <summary>Queues a plug-in's kernel (its arguments already checked against its parameters).</summary>
    internal void Launch(HipKernel kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ,
        ReadOnlySpan<KernelArgument> arguments, uint sharedBytes)
    {
        var limits = _device.Limits;
        long threads = (long)blockX * blockY * blockZ;
        if (threads > limits.MaxThreadsPerBlock || blockX > (uint)limits.MaxBlockDimX || (threads == 0 && gridX * (ulong)gridY * gridZ != 0))
        {
            throw new ArgumentOutOfRangeException(nameof(blockX),
                $"HIP kernel '{kernel.Name}': blocks of {blockX} × {blockY} × {blockZ} threads; {Name} runs 1 to {limits.MaxThreadsPerBlock} a block ({limits.MaxBlockDimX} along x).");
        }

        if (gridX > (uint)limits.MaxGridDimX || gridY > (uint)limits.MaxGridDimY)
        {
            throw new ArgumentOutOfRangeException(nameof(gridX),
                $"HIP kernel '{kernel.Name}': {gridX} × {gridY} × {gridZ} blocks exceed {Name}'s {limits.MaxGridDimX} × {limits.MaxGridDimY}.");
        }

        if (sharedBytes > limits.SharedMemoryPerBlock)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedBytes), sharedBytes,
                $"HIP kernel '{kernel.Name}': {sharedBytes} bytes of shared memory; {Name} allows {limits.SharedMemoryPerBlock} a block.");
        }

        // The arguments' slots and their addresses: on the stack, unless a plug-in declares more parameters than
        // StackArguments (its source decides the count).
        int count = Math.Max(arguments.Length, 1);
        Span<ulong> slots = count <= StackArguments ? stackalloc ulong[count] : new ulong[count];
        Span<nint> addresses = count <= StackArguments ? stackalloc nint[count] : new nint[count];
        fixed (ulong* values = slots)
        fixed (nint* pointers = addresses)
        {
            for (int i = 0; i < arguments.Length; i++)
            {
                values[i] = arguments[i].Kind == KernelArgumentKind.Pointer ? PluginPointer(kernel, i, arguments[i]) : arguments[i].Bits;
                pointers[i] = (nint)(values + i);
            }

            if (gridX == 0 || gridY == 0 || gridZ == 0)
            {
                return;
            }

            IntPtr function = PluginFunction(kernel);
            MakeCurrent();
            int result = hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedBytes, _stream, (void**)pointers, null);
            if (result != Success)
            {
                Check(result, $"hipModuleLaunchKernel({kernel.Name})");            // the message built only on failure
            }
        }
    }

    // Arguments a plug-in launch keeps on the stack (64 slots and 64 addresses: 1 KiB); more go to the heap.
    private const int StackArguments = 64;

    // The device address argument i stands for: a live storage of this device, at its offset.
    private ulong PluginPointer(HipKernel kernel, int i, KernelArgument argument)
    {
        if (argument.Storage is not HipStorage storage || !ReferenceEquals(storage.Backend, this))
        {
            throw new ArgumentException($"HIP kernel '{kernel.Name}': storage argument {i} is not on {Name}.", "arguments");
        }

        if (storage.Pointer == 0)
        {
            throw new InvalidOperationException($"HIP kernel '{kernel.Name}': storage argument {i} was evicted (its memory was given back); restore it first.");
        }

        return storage.Pointer + (ulong)argument.Offset * sizeof(float);
    }

    // The kernel's function on this device: its source compiled (once per source and options) and its function looked up, once.
    private IntPtr PluginFunction(HipKernel kernel)
    {
        if (kernel.LastFunction is { } last && ReferenceEquals(last.Owner, this))
        {
            return last.Function;
        }

        lock (_pluginGate)
        {
            if (!_pluginFunctions.TryGetValue(kernel, out IntPtr function))
            {
                IntPtr module = PluginModule(kernel);
                byte[] entry = Encoding.ASCII.GetBytes(kernel.Entry + "\0");
                fixed (byte* n = entry)
                {
                    Check(hipModuleGetFunction(&function, module, n), $"hipModuleGetFunction({kernel.Entry})");
                }

                _pluginFunctions[kernel] = function;
            }

            kernel.LastFunction = new(this, function);
            return function;
        }
    }

    // The module of the kernel's source on this device: loaded already, read from the kernel cache, or compiled.
    private IntPtr PluginModule(HipKernel kernel)
    {
        var limits = _device.Limits;
        string architecture = limits.Architecture;
        string[] options = string.IsNullOrEmpty(architecture) ? ["-O3", .. kernel.Options]
            : ["-O3", $"--gpu-architecture={architecture}", .. kernel.Options];     // as the library's own, without IDRAK_BLOCK
        string versions = $"runtime {_device.RuntimeVersion}, driver {_device.DriverVersion}, compiler {HipRtc.Version}";
        string identity = $"{limits.Name}/{limits.Uuid}/{limits.PciAddress}";
        string key = HipKernelCache.Key(kernel.Source, options, architecture, identity, versions);
        if (_pluginModules.TryGetValue(key, out IntPtr module))
        {
            return module;
        }

        MakeCurrent();
        if (KernelCache.Load(key, architecture) is { } cached)
        {
            if (TryLoadModule(cached, out module, out _))
            {
                return _pluginModules[key] = module;
            }

            KernelCache.Delete(key, architecture);                         // refused by this runtime: compile again
        }

        byte[] code = HipRtc.Compile(kernel.Source, kernel.Name + ".hip", options, $"HIP kernel '{kernel.Name}'");
        if (!TryLoadModule(code, out module, out string error))
        {
            throw new HipException($"the runtime refused the compiled HIP kernel '{kernel.Name}': {error}");
        }

        KernelCache.Store(key, architecture, code);
        return _pluginModules[key] = module;
    }
}
