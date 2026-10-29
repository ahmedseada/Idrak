// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// GCN code objects Idrak wrote, loaded and launched on this device (plan 35's gate): the code object is written for the
// target the device reports, in the formats IDRAK_HIP_CODE_OBJECT gives (5, then 4), each tried until the runtime takes
// one; a launch passes the kernel arguments as one buffer laid out as the code object's metadata says. The operations'
// kernels are one code object of every kernel (HipBackend.Kernels.cs), launched through the same LaunchGcn.
internal sealed unsafe partial class HipBackend
{
    /// <summary>
    /// A GCN kernel loaded on this device: its module and function, the target and the code object format the runtime
    /// took, and its arguments as the code object's metadata declares them (each at its offset, of its size, and the
    /// bytes a launch passes). <see cref="Dispose"/> unloads the module when the function owns it (<see cref="LoadGcn"/>);
    /// a function of the kernel set (<see cref="GcnKernelSet"/>) shares the set's module, which the set unloads.
    /// </summary>
    internal sealed class GcnFunction : IDisposable
    {
        private readonly HipBackend _backend;
        private IntPtr _module;
        private GCHandle _image;                                                  // the loaded image, pinned while its module lives

        internal GcnFunction(HipBackend backend, IntPtr module, IntPtr handle, GCHandle image, string name, GcnIsaVersion target, int format, int argumentBytes, string refused,
            IReadOnlyList<GcnArgument>? arguments = null, int block = 0)
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
            Arguments = arguments ?? [];
            Block = block;
        }

        /// <summary>A function of a module another owner unloads (the kernel set's), with its layout from the metadata.</summary>
        internal GcnFunction(HipBackend backend, IntPtr handle, GcnKernelLayout layout, GcnIsaVersion target, int format)
            : this(backend, IntPtr.Zero, handle, default, layout.Name, target, format, layout.ArgumentBytes, "", layout.Arguments, layout.Block)
        {
        }

        /// <summary>The block the kernel runs in, as its metadata declares it (0 when the function was loaded without it).</summary>
        public int Block { get; }

        /// <summary>The arguments the metadata declares, by offset (empty when the function was loaded without them).</summary>
        public IReadOnlyList<GcnArgument> Arguments { get; }

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
                return new GcnFunction(this, module, function, image, code.Name, isa, format, GcnCodeObject.KernargBytes(code), refused.ToString(), code.Arguments,
                    code.RequiredWorkgroupSize?.X ?? code.MaxFlatWorkgroupSize);
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
    internal void LaunchGcn(GcnFunction function, uint blocks, uint blockSize, ReadOnlySpan<byte> arguments) => LaunchGcn(function, blocks, 1, blockSize, arguments);

    /// <summary>
    /// <see cref="LaunchGcn(GcnFunction, uint, uint, ReadOnlySpan{byte})"/> over a grid of <paramref name="blocks"/> ×
    /// <paramref name="blocksY"/> blocks (a block is one high).
    /// </summary>
    internal void LaunchGcn(GcnFunction function, uint blocks, uint blocksY, uint blockSize, ReadOnlySpan<byte> arguments) =>
        LaunchGcn(function, blocks, blocksY, 1, blockSize, arguments);

    /// <summary>
    /// <see cref="LaunchGcn(GcnFunction, uint, uint, ReadOnlySpan{byte})"/> over a grid of <paramref name="blocks"/> ×
    /// <paramref name="blocksY"/> × <paramref name="blocksZ"/> blocks (a block is one high and one deep; z within what the
    /// device reports, one when it reports nothing).
    /// </summary>
    internal void LaunchGcn(GcnFunction function, uint blocks, uint blocksY, uint blocksZ, uint blockSize, ReadOnlySpan<byte> arguments)
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

        if (blocks == 0 || blocks > (uint)DeviceLimits.MaxGridDimX || blocksY == 0 || blocksY > (uint)DeviceLimits.MaxGridDimY || blockSize == 0 || blockSize > (uint)DeviceLimits.MaxBlockDimX)
        {
            throw new ArgumentOutOfRangeException(nameof(blocks), $"{function.Name}: {blocks} × {blocksY} blocks of {blockSize}; the device takes 1 to {DeviceLimits.MaxGridDimX} × "
                + $"{DeviceLimits.MaxGridDimY} blocks of 1 to {DeviceLimits.MaxBlockDimX}.");
        }

        if (blocksZ == 0 || blocksZ > (uint)Math.Max(1, DeviceLimits.MaxGridDimZ))
        {
            throw new ArgumentOutOfRangeException(nameof(blocksZ), $"{function.Name}: {blocksZ} blocks in z; the device takes 1 to {Math.Max(1, DeviceLimits.MaxGridDimZ)}.");
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
            Launch(function, blocks, blocksY, blocksZ, blockSize, extra);              // timed while a profile runs (HipBackend.Profile.cs)
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
