// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// GCN code objects Idrak wrote, loaded and launched on this device (plan 35's gate): the code object is written for the
// target the device reports, in the formats IDRAK_HIP_CODE_OBJECT gives (5, then 4), each tried until the runtime takes
// one; a launch passes the kernel arguments as one buffer laid out as the code object's metadata says. Kept apart from
// the operations (HipBackend.Kernels.cs), which take the host fallback until the kernels are wired there.
internal sealed unsafe partial class HipBackend
{
    /// <summary>
    /// A GCN kernel loaded on this device: its module and function, the target and the code object format the runtime
    /// took, and the bytes of kernel arguments its metadata declares. <see cref="Dispose"/> unloads the module.
    /// </summary>
    internal sealed class GcnFunction : IDisposable
    {
        private readonly HipBackend _backend;
        private IntPtr _module;
        private GCHandle _image;                                                  // the loaded image, pinned while its module lives

        internal GcnFunction(HipBackend backend, IntPtr module, IntPtr handle, GCHandle image, string name, GcnIsaVersion target, int format, int argumentBytes, string refused)
        {
            _backend = backend;
            _module = module;
            _image = image;
            Handle = handle;
            Name = name;
            Target = target;
            Format = format;
            ArgumentBytes = argumentBytes;
            Refused = refused;
        }

        /// <summary>The kernel's name.</summary>
        public string Name { get; }

        /// <summary>The target the code object was written for (the device's reported one).</summary>
        public GcnIsaVersion Target { get; }

        /// <summary>The code object format the runtime loaded.</summary>
        public int Format { get; }

        /// <summary>Bytes of kernel arguments a launch passes.</summary>
        public int ArgumentBytes { get; }

        /// <summary>The formats the runtime refused before this one, with its errors; empty when the first one loaded.</summary>
        public string Refused { get; }

        internal IntPtr Handle { get; private set; }

        public void Dispose()
        {
            if (_module != IntPtr.Zero)
            {
                _backend.MakeCurrent();
                _backend.WaitStream();                                           // launches queued earlier still run its code
                if (hipModuleUnload != null)
                {
                    Check(hipModuleUnload(_module), nameof(hipModuleUnload));
                }

                _module = IntPtr.Zero;
                Handle = IntPtr.Zero;
                _image.Free();
            }
        }
    }

    /// <summary>The runtime's version as it reports it (major · 10⁷ + minor · 10⁵ + patch).</summary>
    internal int RuntimeVersion => _device.RuntimeVersion;

    /// <summary>The driver's version as the runtime reports it.</summary>
    internal int DriverVersion => _device.DriverVersion;

    /// <summary>
    /// Writes the kernel <paramref name="kernel"/> makes for this device's family into a code object for the target the
    /// device reports, and loads it: format by format as IDRAK_HIP_CODE_OBJECT gives them (newest first), the next when
    /// the runtime refuses one (to load it, or to find the kernel in it).
    /// </summary>
    /// <exception cref="HipException">
    /// The device reports no target Idrak reads, no registered family handles it, or the runtime refused every format; the
    /// message names the target, each format tried with the runtime's error, and the runtime.
    /// </exception>
    internal GcnFunction LoadGcn(Func<IGcnTarget, GcnKernelCode> kernel)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        string reported = DeviceLimits.Architecture;
        if (!GcnIsaVersion.TryParse(reported, out var isa))
        {
            throw new HipException($"{Name} reports the GCN target '{reported}' (gcnArchName), which is not a target string Idrak reads: no GCN code can be written for it.");
        }

        IGcnTarget target;
        try
        {
            target = GcnTargets.For(reported);
        }
        catch (NotRegisteredException e)
        {
            throw new HipException($"No GCN code can be written for {Name}: {e.Message}");
        }

        var code = kernel(target);
        int[] formats = GcnCodeObject.FormatsFromEnvironment();
        byte[] name = Encoding.UTF8.GetBytes(code.Name + "\0");
        var refused = new StringBuilder();
        MakeCurrent();
        foreach (int format in formats)
        {
            var image = GCHandle.Alloc(GcnCodeObject.Write(target, isa, format, [code]), GCHandleType.Pinned);
            IntPtr module = IntPtr.Zero, function = IntPtr.Zero;
            int result = hipModuleLoadData(&module, (void*)image.AddrOfPinnedObject());
            string call = nameof(hipModuleLoadData);
            if (result == Success)
            {
                fixed (byte* p = name)
                {
                    result = hipModuleGetFunction(&function, module, p);
                }

                call = nameof(hipModuleGetFunction);
            }

            if (result == Success)
            {
                return new GcnFunction(this, module, function, image, code.Name, isa, format, GcnCodeObject.KernargBytes(code), refused.ToString());
            }

            if (module != IntPtr.Zero && hipModuleUnload != null)
            {
                hipModuleUnload(module);
            }

            image.Free();
            refused.Append(refused.Length > 0 ? "; " : "").Append($"format {format}: {call} failed with {Describe(result)}");
        }

        throw new HipException($"The HIP runtime refused Idrak's GCN code object for {isa} ({code.Name}) in every format tried: {refused}. "
                               + $"HIP runtime {FormatVersion(RuntimeVersion)} (driver {DriverVersion}) from {(LoadedFrom.Length > 0 ? LoadedFrom : "the system's search")}; "
                               + "IDRAK_HIP_CODE_OBJECT chooses the formats tried (5, 4 or auto).");
    }

    /// <summary>
    /// Queues <paramref name="function"/> on this device's stream in <paramref name="blocks"/> blocks of
    /// <paramref name="blockSize"/> work-items, its kernel arguments packed in <paramref name="arguments"/> as its metadata lays
    /// them out (each at its offset, of its size). They go through hipModuleLaunchKernel's <c>extra</c>, as one buffer
    /// with its size: HIP's documented form for a module's kernel (the runtime reads each argument from the buffer at
    /// the offset the code object's metadata gives, and copies them before the call returns).
    /// </summary>
    internal void LaunchGcn(GcnFunction function, uint blocks, uint blockSize, ReadOnlySpan<byte> arguments)
    {
        ArgumentNullException.ThrowIfNull(function);
        if (function.Handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(function.Name);
        }

        if (arguments.Length != function.ArgumentBytes)
        {
            throw new ArgumentException($"{function.Name} takes {function.ArgumentBytes} bytes of kernel arguments, not {arguments.Length}.", nameof(arguments));
        }

        if (blocks == 0 || blocks > (uint)DeviceLimits.MaxGridDimX || blockSize == 0 || blockSize > (uint)DeviceLimits.MaxBlockDimX)
        {
            throw new ArgumentOutOfRangeException(nameof(blocks), $"{function.Name}: {blocks} blocks of {blockSize}; the device takes 1 to {DeviceLimits.MaxGridDimX} blocks of 1 to {DeviceLimits.MaxBlockDimX}.");
        }

        MakeCurrent();
        nuint size = (nuint)arguments.Length;
        fixed (byte* p = arguments)
        {
            void** extra = stackalloc void*[5];
            extra[0] = (void*)LaunchParamBufferPointer;
            extra[1] = p;
            extra[2] = (void*)LaunchParamBufferSize;
            extra[3] = &size;
            extra[4] = (void*)LaunchParamEnd;
            Check(hipModuleLaunchKernel(function.Handle, blocks, 1, 1, blockSize, 1, 1, 0, _stream, null, extra), nameof(hipModuleLaunchKernel));
        }
    }

    /// <summary>Fills the first <paramref name="n"/> floats of <paramref name="y"/> with <paramref name="value"/> through GCN's <c>fill_f32</c> (loaded by <see cref="LoadGcn"/> from <see cref="GcnKernels.Fill"/>).</summary>
    internal void FillGcn(GcnFunction fill, Storage y, int n, float value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(n, Math.Min(y.Length, GcnKernels.FillMostElements));
        if (n == 0)
        {
            return;
        }

        Span<byte> arguments = stackalloc byte[GcnKernels.FillArgumentBytes];
        GcnKernels.PackFill(arguments, P(y), value, (uint)n);
        LaunchGcn(fill, (uint)((n + GcnKernels.FillBlock - 1) / GcnKernels.FillBlock), GcnKernels.FillBlock, arguments);
    }
}
