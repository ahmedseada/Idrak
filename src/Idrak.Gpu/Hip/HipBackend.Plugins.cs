// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// Plug-ins' own kernels (GcnKernel), as CUDA's CudaBackend.Plugins.cs runs plug-ins' PTX: each GCN text assembled once
// on this device by Idrak's assembler for the target the device reports, its code object kept in the kernel cache (the
// key: the text, the target, the device, the code object format and the runtime, driver and library versions, as the
// library's own set) and loaded once in the first format the runtime takes, then launched like the library's kernels: on
// the stream, its arguments packed at the offsets the code object's metadata gives, timed under the kernel's name while
// a profile runs, and a refused launch reported by the telemetry (LaunchFailed). Each load is reported once
// (KernelsLoaded: from the cache or assembled, its time and size). IDRAK_HIP_KERNELS turns off the library's kernels,
// not a plug-in's (CUDA's plug-in PTX always runs, and Vulkan's setting leaves a plug-in's SPIR-V on).
internal sealed unsafe partial class HipBackend : IKernelLauncher<GcnKernel>
{
    // Arguments a plug-in launch keeps on the stack (32 values: 256 bytes); more go to the heap (the text decides the count).
    private const int PluginStackArguments = 32;

    private readonly Lock _pluginLock = new();

    // Loaded functions by GCN text (two kernels of one text share it), and the modules they live in (kept while the device lives).
    private readonly Dictionary<string, GcnFunction> _pluginFunctions = new(StringComparer.Ordinal);
    private readonly List<GcnKernelSet> _pluginModules = [];

    /// <summary>
    /// Why plug-ins' GCN kernels cannot run on this device (null when they can): its reported target is not one Idrak
    /// reads, or no registered family handles it. IDRAK_HIP_KERNELS plays no part (it turns off the library's own kernels).
    /// </summary>
    internal string? PluginKernelsUnavailableReason => TargetUnavailable(Name, DeviceLimits.Architecture, out _, out _);

    // Queues a plug-in's kernel (its arguments and declared block already checked by GcnKernel.Launch).
    void IKernelLauncher<GcnKernel>.Launch(GcnKernel kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ,
        ReadOnlySpan<KernelArgument> arguments, uint sharedBytes)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        var limits = DeviceLimits;
        bool work = gridX != 0 && gridY != 0 && gridZ != 0;
        ulong items = (ulong)blockX * blockY * blockZ;
        if (items > (ulong)limits.MaxThreadsPerBlock || blockX > (uint)limits.MaxBlockDimX || (items == 0 && work))
        {
            throw new ArgumentOutOfRangeException(nameof(blockX), string.Create(CultureInfo.InvariantCulture,
                $"GCN kernel '{kernel.Name}': blocks of {blockX} × {blockY} × {blockZ} work-items; {Name} runs 1 to {limits.MaxThreadsPerBlock} a block ({limits.MaxBlockDimX} along x)."));
        }

        uint mostZ = (uint)Math.Max(1, limits.MaxGridDimZ);
        if (gridX > (uint)limits.MaxGridDimX || gridY > (uint)limits.MaxGridDimY || gridZ > mostZ)
        {
            throw new ArgumentOutOfRangeException(nameof(gridX), string.Create(CultureInfo.InvariantCulture,
                $"GCN kernel '{kernel.Name}': {gridX} × {gridY} × {gridZ} blocks exceed {Name}'s {limits.MaxGridDimX} × {limits.MaxGridDimY} × {mostZ}."));
        }

        // The dispatch counts work-items in 32 bits a dimension (the dispatch packet's grid size).
        if ((ulong)gridX * blockX > uint.MaxValue || (ulong)gridY * blockY > uint.MaxValue || (ulong)gridZ * blockZ > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(gridX), string.Create(CultureInfo.InvariantCulture,
                $"GCN kernel '{kernel.Name}': {gridX} × {gridY} × {gridZ} blocks of {blockX} × {blockY} × {blockZ} pass 2^32 work-items in a dimension, which a dispatch cannot count."));
        }

        long lds = kernel.Declarations.LdsBytes + (long)sharedBytes;
        if (lds > limits.SharedMemoryPerBlock)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedBytes), sharedBytes, string.Create(CultureInfo.InvariantCulture,
                $"GCN kernel '{kernel.Name}': {sharedBytes} bytes of dynamic shared memory after the {kernel.Declarations.LdsBytes} its text declares (.lds); {Name} allows {limits.SharedMemoryPerBlock} a block."));
        }

        Span<ulong> values = arguments.Length <= PluginStackArguments ? stackalloc ulong[PluginStackArguments] : new ulong[arguments.Length];
        values = values[..arguments.Length];
        for (int i = 0; i < arguments.Length; i++)
        {
            values[i] = arguments[i].Kind == KernelArgumentKind.Pointer ? PluginPointer(kernel, i, arguments[i]) : arguments[i].Bits;
        }

        if (!work)
        {
            return;
        }

        var function = PluginFunction(kernel);
        int bytes = function.ArgumentBytes;
        Span<byte> buffer = bytes <= MostArgumentBytes ? stackalloc byte[MostArgumentBytes] : new byte[bytes];
        buffer = buffer[..bytes];
        buffer.Clear();
        GcnKernels.PackArguments(function.Arguments, buffer, values);
        if (Profiling)
        {
            LabelNextLaunch(kernel.Name, 0);                                       // timed under the kernel's name, as CUDA's plug-ins
        }

        MakeCurrent();
        nuint size = (nuint)bytes;
        fixed (byte* p = buffer)
        {
            void** extra = stackalloc void*[5];
            extra[0] = (void*)LaunchParamBufferPointer;
            extra[1] = p;
            extra[2] = (void*)LaunchParamBufferSize;
            extra[3] = &size;
            extra[4] = (void*)LaunchParamEnd;
            Launch(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedBytes, bytes == 0 ? null : extra);
        }
    }

    // The device address argument i stands for: a live storage of this device, at its offset.
    private ulong PluginPointer(GcnKernel kernel, int i, KernelArgument argument)
    {
        if (argument.Storage is not HipStorage storage || !ReferenceEquals(storage.Backend, this))
        {
            throw new ArgumentException($"GCN kernel '{kernel.Name}': storage argument {i} ({kernel.ParameterNames[i]}) is not on {Name}.", "arguments");
        }

        if (storage.Pointer == 0)
        {
            throw new InvalidOperationException($"GCN kernel '{kernel.Name}': storage argument {i} ({kernel.ParameterNames[i]}) was evicted (its memory was given back); restore it first.");
        }

        return storage.Pointer + (ulong)argument.Offset * sizeof(float);
    }

    // The kernel's function on this device: its text assembled (or read from the cache) and loaded once a text.
    private GcnFunction PluginFunction(GcnKernel kernel)
    {
        if (kernel.LastFunction is { } last && ReferenceEquals(last.Owner, this))
        {
            return last.Function;
        }

        lock (_pluginLock)
        {
            if (!_pluginFunctions.TryGetValue(kernel.Gcn, out var function))
            {
                function = LoadPlugin(kernel, HipKernelCache.FromEnvironment());
                _pluginFunctions[kernel.Gcn] = function;
            }

            kernel.LastFunction = new(this, function);
            return function;
        }
    }

    /// <summary>
    /// <paramref name="kernel"/> loaded on this device through <paramref name="cache"/>: for each format IDRAK_HIP_CODE_OBJECT
    /// allows (newest first), the cached code object of its key, else the text assembled for the reported target (once
    /// for all formats) and written; its metadata read (the kernel declared as its text declares it); loaded by the
    /// runtime and the function looked up. A cached object that does not read or that the runtime refuses is deleted and
    /// assembled again; one assembled is kept. The module stays loaded while the device lives, and the load is reported
    /// once (KernelsLoaded).
    /// </summary>
    /// <exception cref="HipException">
    /// No GCN code can be written for the device (its target unread or not registered), the text does not assemble for it
    /// (the assembler's line and column), or the runtime refused the code object in every format.
    /// </exception>
    internal GcnFunction LoadPlugin(GcnKernel kernel, HipKernelCache cache)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(cache);
        long start = Stopwatch.GetTimestamp();
        string reported = DeviceLimits.Architecture;
        if (TargetUnavailable(Name, reported, out var isa, out var target) is { } unavailable)
        {
            throw new HipException($"GCN kernel '{kernel.Name}': {unavailable}.");
        }

        int[] formats;
        try
        {
            formats = GcnCodeObject.FormatsFromEnvironment();
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new HipException($"GCN kernel '{kernel.Name}': {e.Message}");
        }

        GcnKernelSource[] sources = [new(kernel.Entry, kernel.Text)];
        string device = $"{DeviceLimits.Name} {DeviceLimits.Uuid}";
        string versions = $"runtime {RuntimeVersion}, driver {DriverVersion}";
        GcnKernelCode? code = null;
        var assembly = TimeSpan.Zero;
        var refused = new StringBuilder();
        MakeCurrent();
        foreach (int format in formats)
        {
            string key = HipKernelCache.KeyOf(sources, [$"format={format}", "plug-in"], isa.ToString(), device, versions);
            byte[]? image = cache.Load(key, reported);
            bool cached = image is not null;
            while (true)
            {
                if (image is null)
                {
                    long assembling = Stopwatch.GetTimestamp();
                    try
                    {
                        code ??= kernel.Assemble(isa);
                        image = GcnCodeObject.Write(target!, isa, format, [code]);
                    }
                    catch (Exception e) when (e is GcnSyntaxException or GcnAllocationException or ArgumentException or NotSupportedException)
                    {
                        throw new HipException($"GCN kernel '{kernel.Name}' does not assemble for {isa} ({Name}): {e.Message}");
                    }

                    assembly += Stopwatch.GetElapsedTime(assembling);
                }

                string? problem = PluginLayout(image, kernel, out var layout);
                string? error = null;
                if (problem is null && TryLoadSet(image, new Dictionary<string, GcnKernelLayout>(StringComparer.Ordinal) { [layout!.Name] = layout }, isa, format,
                        out var module, out var pinned, out var functions, out error))
                {
                    if (!cached)
                    {
                        cache.Store(key, reported, image);
                    }

                    var set = new GcnKernelSet(this, module, pinned, functions, isa, format, image.Length, cached, assembly, Stopwatch.GetElapsedTime(start), refused.ToString());
                    _pluginModules.Add(set);
                    ReportPluginLoaded(kernel, set);
                    return functions[kernel.Entry];
                }

                if (cached)
                {
                    cache.Delete(key, reported);                                // a kept object that does not read or load: made again
                    (image, cached) = (null, false);
                    continue;
                }

                if (problem is not null)
                {
                    throw new HipException($"GCN kernel '{kernel.Name}': the code object Idrak wrote for {isa} does not declare it as its text does: {problem}");
                }

                refused.Append(refused.Length > 0 ? "; " : "").Append(CultureInfo.InvariantCulture, $"format {format}: {error}");
                break;
            }
        }

        throw new HipException($"The HIP runtime refused the code object of GCN kernel '{kernel.Name}' for {isa} in every format tried: {refused}. "
                               + $"HIP runtime {FormatVersion(RuntimeVersion)} (driver {DriverVersion}) from {(LoadedFrom.Length > 0 ? LoadedFrom : "the system's search")}; "
                               + "IDRAK_HIP_CODE_OBJECT chooses the formats tried (5, 4 or auto).");
    }

    /// <summary>
    /// The plug-in kernel's layout as <paramref name="image"/>'s metadata declares it: the one kernel, named as its text
    /// names it, with the arguments its text declares (offsets, sizes, buffers against values); the problem otherwise.
    /// </summary>
    internal static string? PluginLayout(byte[] image, GcnKernel kernel, out GcnKernelLayout? layout)
    {
        layout = null;
        IReadOnlyList<GcnKernelLayout> kernels;
        try
        {
            kernels = GcnCodeObject.ReadKernels(image);
        }
        catch (Exception e) when (e is InvalidDataException or OverflowException)
        {
            return e.Message;
        }

        if (kernels.Count != 1 || kernels[0].Name != kernel.Entry)
        {
            return $"it declares {string.Join(", ", kernels.Select(k => k.Name))}, not {kernel.Entry} alone";
        }

        var declared = kernel.Declarations.Arguments;
        var written = kernels[0].Arguments;
        if (written.Count != declared.Count)
        {
            return $"{written.Count} arguments, not the {declared.Count} its text declares";
        }

        for (int i = 0; i < declared.Count; i++)
        {
            if (written[i].Offset != declared[i].Offset || written[i].Size != declared[i].Size || written[i].Kind != declared[i].Kind)
            {
                return $"argument {i} ({declared[i].Name}) is {written[i].Size} bytes at {written[i].Offset} ({written[i].Kind}), not {declared[i].Size} at {declared[i].Offset} ({declared[i].Kind})";
            }
        }

        layout = kernels[0];
        return null;
    }

    // Reports a plug-in kernel's load (once a text and device), while someone listens: from the cache or assembled, its
    // time and size (rules 76, 78).
    private void ReportPluginLoaded(GcnKernel kernel, GcnKernelSet set)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            string assembled = set.FromCache ? "" : string.Create(CultureInfo.InvariantCulture, $", assembled in {set.Assembly.TotalMilliseconds:F1} ms");
            string detail = $"plug-in GCN kernel '{kernel.Name}', {set.Target}, code object format {set.Format}{assembled}{(set.Refused.Length > 0 ? $"; refused first: {set.Refused}" : "")}";
            Telemetry.KernelsLoaded(new KernelsLoaded($"hip:{Ordinal}", Kind, 1, set.FromCache, set.Duration, set.Bytes, detail));
        }
    }
}
