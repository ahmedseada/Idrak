// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// HIP's kernels. The library depends on no part of the HIP SDK, as it depends on no CUDA Toolkit, so HIP has no
// compiler: its kernels are Idrak's own GCN code (plan 35), written in C# (GcnKernels.All) and assembled by Idrak, as
// CUDA's are PTX. On first need, every kernel is assembled for the target the device reports into one code object, in
// the formats IDRAK_HIP_CODE_OBJECT allows, kept in the kernel cache (HipKernelCache: the key hashes every kernel's text,
// the target, the device and the runtime and library versions) and loaded once; each function is looked up by name with
// its arguments as the code object's metadata declares them (GcnCodeObject.ReadKernels), and every launch is checked
// against them, as CUDA's _signatures are. The load is reported once (Telemetry.KernelsLoaded): from the cache (warm)
// or assembled (cold), with its time and size (rules 76, 78).
//
// The kernels are unavailable, with the reason KernelsUnavailableReason gives and every operation through the host
// fallback, when IDRAK_HIP_KERNELS=0 turns them off, the device reports a target no registered family handles (or one
// Idrak does not read), its limits are implausible, or the runtime refuses the code object in every format.
internal sealed unsafe partial class HipBackend
{
    /// <summary>The setting that turns HIP's kernels off: 0 or false (every operation through the host fallback).</summary>
    internal const string KernelsVariable = "IDRAK_HIP_KERNELS";

    // The most bytes of kernel arguments a kernel of the set may declare: the launch's buffer is this, on the stack.
    private const int MostArgumentBytes = 256;

    private readonly Lock _kernelLock = new();
    private volatile GcnKernelSet? _kernelSet;
    private volatile string? _kernelsUnavailable;
    private volatile bool _kernelsTried;

    /// <summary>
    /// HIP's kernels loaded on a device: one module of every kernel in <see cref="GcnKernels.All"/>, its functions by name
    /// (each with its metadata's arguments), the target and format the runtime took, and how the load went (from the
    /// cache or assembled, its times and the code object's size). <see cref="Dispose"/> unloads the module.
    /// </summary>
    internal sealed class GcnKernelSet : IDisposable
    {
        private readonly HipBackend _backend;
        private IntPtr _module;
        private GCHandle _image;

        internal GcnKernelSet(HipBackend backend, IntPtr module, GCHandle image, Dictionary<string, GcnFunction> functions, GcnIsaVersion target, int format,
            int bytes, bool fromCache, TimeSpan assembly, TimeSpan duration, string refused)
        {
            _backend = backend;
            _module = module;
            _image = image;
            Functions = functions;
            Target = target;
            Format = format;
            Bytes = bytes;
            FromCache = fromCache;
            Assembly = assembly;
            Duration = duration;
            Refused = refused;
        }

        /// <summary>The kernels by name.</summary>
        public IReadOnlyDictionary<string, GcnFunction> Functions { get; }

        /// <summary>The target the code object was written for (the device's reported one).</summary>
        public GcnIsaVersion Target { get; }

        /// <summary>The code object format the runtime loaded.</summary>
        public int Format { get; }

        /// <summary>The code object's size in bytes.</summary>
        public int Bytes { get; }

        /// <summary>Whether the code object came from the kernel cache (warm) rather than the assembler (cold).</summary>
        public bool FromCache { get; }

        /// <summary>Assembling every kernel and writing the code object (zero when it came from the cache).</summary>
        public TimeSpan Assembly { get; }

        /// <summary>The whole load: the cache read or the assembly, the metadata read, the runtime's load and the lookups.</summary>
        public TimeSpan Duration { get; }

        /// <summary>The formats the runtime refused before this one, with its errors; empty when the first one loaded.</summary>
        public string Refused { get; }

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
                _image.Free();
            }
        }
    }

    /// <summary>Why this device runs no kernels of its own (loading them first when they were not tried yet); null when they run.</summary>
    public string? KernelsUnavailableReason => KernelSet is null ? _kernelsUnavailable : null;

    // Why the kernels did not load, for listings, diagnostics and the host-fallback event.
    string? IKernelAvailability.KernelsUnavailable() => KernelsUnavailableReason;

    /// <summary>The loaded kernels (loading them on first need), or null when they are unavailable.</summary>
    internal GcnKernelSet? KernelSet => _kernelsTried ? _kernelSet : LoadKernels();

    /// <summary>The reason IDRAK_HIP_KERNELS (<paramref name="value"/>) gives for the kernels being off, or null when it leaves them on.</summary>
    internal static string? KernelsSettingReason(string? value)
    {
        string text = value?.Trim() ?? "";
        return text.Equals("0", StringComparison.Ordinal) || text.Equals("false", StringComparison.OrdinalIgnoreCase) || text.Equals("off", StringComparison.OrdinalIgnoreCase)
            ? $"turned off by {KernelsVariable}={text}"
            : null;
    }

    /// <summary>
    /// Why no GCN code can be written for the target <paramref name="reported"/> by <paramref name="device"/> (null when a
    /// registered family handles it, given in <paramref name="target"/>): not a target string Idrak reads, or one no family
    /// handles ("not registered", naming it).
    /// </summary>
    internal static string? TargetUnavailable(string device, string reported, out GcnIsaVersion isa, out IGcnTarget? target)
    {
        target = null;
        if (!GcnIsaVersion.TryParse(reported, out isa))
        {
            return $"{device} reports the GCN target '{reported}' (gcnArchName), which is not a target string Idrak reads: no GCN code can be written for it";
        }

        try
        {
            target = GcnTargets.For(reported);
            return null;
        }
        catch (NotRegisteredException e)
        {
            return $"no GCN code can be written for {device}: {e.Message}";
        }
    }

    // Loads the kernels once (under the lock), keeping the set or the reason they are unavailable.
    private GcnKernelSet? LoadKernels()
    {
        lock (_kernelLock)
        {
            if (_kernelsTried)
            {
                return _kernelSet;
            }

            GcnKernelSet? set = null;
            string? why = KernelsSettingReason(Environment.GetEnvironmentVariable(KernelsVariable));
            if (why is null && DeviceLimits.Problem() is { } problem)
            {
                why = $"{Name} reports implausible limits ({problem}), so its kernels stay off";
            }

            if (why is null)
            {
                try
                {
                    set = LoadKernelSet(HipKernelCache.FromEnvironment(), out why);
                }
                catch (HipException e)
                {
                    why = e.Message;                                             // reported itself (DeviceFailed)
                }
            }

            if (set is not null)
            {
                ReportKernelsLoaded(set);
            }

            (_kernelSet, _kernelsUnavailable) = (set, why);
            _kernelsTried = true;
            return set;
        }
    }

    /// <summary>
    /// Every kernel of <see cref="GcnKernels.All"/> loaded on this device as one code object, through
    /// <paramref name="cache"/>: for each format IDRAK_HIP_CODE_OBJECT allows (newest first), the cached code object of its
    /// key, else the kernels assembled for the reported target (once for all formats) and written; its metadata read
    /// (every kernel present, its arguments within <see cref="MostArgumentBytes"/>); loaded by the runtime, each function
    /// looked up. A cached object that does not read or that the runtime refuses is deleted and assembled again; one
    /// assembled is kept. Returns null with <paramref name="unavailable"/> when no code can be written for the device (an
    /// unread or unregistered target, a format setting refused, kernels that do not assemble). Not installed: the caller
    /// owns it (the bench loads sets of its own to time cold and warm loads).
    /// </summary>
    /// <exception cref="HipException">The runtime refused the code object in every format; the message names the target,
    /// each format with the runtime's error, and the runtime.</exception>
    internal GcnKernelSet? LoadKernelSet(HipKernelCache cache, out string? unavailable)
    {
        ArgumentNullException.ThrowIfNull(cache);
        long start = Stopwatch.GetTimestamp();
        string reported = DeviceLimits.Architecture;
        unavailable = TargetUnavailable(Name, reported, out var isa, out var target);
        if (unavailable is not null)
        {
            return null;
        }

        int[] formats;
        try
        {
            formats = GcnCodeObject.FormatsFromEnvironment();
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            unavailable = e.Message;
            return null;
        }

        var all = GcnKernels.All;
        string device = $"{DeviceLimits.Name} {DeviceLimits.Uuid}";
        string versions = $"runtime {RuntimeVersion}, driver {DriverVersion}";
        GcnKernelCode[]? codes = null;
        var assembly = TimeSpan.Zero;
        var refused = new StringBuilder();
        MakeCurrent();
        foreach (int format in formats)
        {
            string key = HipKernelCache.KeyOf(all, [$"format={format}"], isa.ToString(), device, versions);
            byte[]? image = cache.Load(key, reported);
            bool cached = image is not null;
            while (true)
            {
                if (image is null)
                {
                    try
                    {
                        long assembling = Stopwatch.GetTimestamp();
                        codes ??= GcnKernels.AssembleAll(target!, isa);
                        image = GcnCodeObject.Write(target!, isa, format, codes);
                        assembly += Stopwatch.GetElapsedTime(assembling);
                    }
                    catch (Exception e) when (e is GcnSyntaxException or GcnAllocationException or ArgumentException)
                    {
                        unavailable = $"Idrak's GCN kernels do not assemble for {isa}: {e.Message}";
                        ReportKernelsFailed(unavailable);
                        return null;
                    }
                }

                string? problem = Layouts(image, all, out var layouts);
                string? error = null;
                if (problem is null && TryLoadSet(image, layouts, isa, format, out var module, out var pinned, out var functions, out error))
                {
                    if (!cached)
                    {
                        cache.Store(key, reported, image);
                    }

                    return new GcnKernelSet(this, module, pinned, functions, isa, format, image.Length, cached, assembly, Stopwatch.GetElapsedTime(start), refused.ToString());
                }

                if (cached)
                {
                    cache.Delete(key, reported);                                // a kept object that does not read or load: made again
                    (image, cached) = (null, false);
                    continue;
                }

                if (problem is not null)
                {
                    unavailable = $"Idrak's GCN code object for {isa} does not declare its kernels as written: {problem}";
                    ReportKernelsFailed(unavailable);
                    return null;
                }

                refused.Append(refused.Length > 0 ? "; " : "").Append(CultureInfo.InvariantCulture, $"format {format}: {error}");
                break;
            }
        }

        throw new HipException($"The HIP runtime refused Idrak's GCN code object of {all.Count} kernels for {isa} in every format tried: {refused}. "
                               + $"HIP runtime {FormatVersion(RuntimeVersion)} (driver {DriverVersion}) from {(LoadedFrom.Length > 0 ? LoadedFrom : "the system's search")}; "
                               + "IDRAK_HIP_CODE_OBJECT chooses the formats tried (5, 4 or auto).");
    }

    // The code object's kernels as its metadata declares them, each of `all` present once with arguments that fit the
    // launch's buffer; the problem otherwise.
    private static string? Layouts(byte[] image, IReadOnlyList<GcnKernelSource> all, out Dictionary<string, GcnKernelLayout> layouts)
    {
        layouts = new Dictionary<string, GcnKernelLayout>(StringComparer.Ordinal);
        try
        {
            foreach (var layout in GcnCodeObject.ReadKernels(image))
            {
                if (!layouts.TryAdd(layout.Name, layout))
                {
                    return $"{layout.Name} is declared twice";
                }
            }
        }
        catch (Exception e) when (e is InvalidDataException or OverflowException)
        {
            return e.Message;
        }

        foreach (var kernel in all)
        {
            if (!layouts.TryGetValue(kernel.Name, out var layout))
            {
                return $"{kernel.Name} is missing from its metadata";
            }

            if (layout.ArgumentBytes > MostArgumentBytes || layout.Block <= 0)
            {
                return $"{kernel.Name} declares {layout.ArgumentBytes} bytes of kernel arguments (at most {MostArgumentBytes}) and a block of {layout.Block}";
            }
        }

        return layouts.Count == all.Count ? null : $"its metadata declares {layouts.Count} kernels, not {all.Count}";
    }

    // The runtime's load of one code object and the lookup of each function; false with the runtime's error.
    private bool TryLoadSet(byte[] image, Dictionary<string, GcnKernelLayout> layouts, GcnIsaVersion isa, int format, out IntPtr module, out GCHandle pinned,
        out Dictionary<string, GcnFunction> functions, out string? error)
    {
        functions = new Dictionary<string, GcnFunction>(layouts.Count, StringComparer.Ordinal);
        pinned = GCHandle.Alloc(image, GCHandleType.Pinned);
        IntPtr loaded = IntPtr.Zero;
        int result = hipModuleLoadData(&loaded, (void*)pinned.AddrOfPinnedObject());
        module = loaded;
        error = null;
        if (result != Success)
        {
            error = $"{nameof(hipModuleLoadData)} failed with {Describe(result)}";
        }
        else
        {
            foreach (var layout in layouts.Values)
            {
                byte[] name = Encoding.UTF8.GetBytes(layout.Name + "\0");
                IntPtr function = IntPtr.Zero;
                fixed (byte* p = name)
                {
                    result = hipModuleGetFunction(&function, loaded, p);
                }

                if (result != Success)
                {
                    error = $"{nameof(hipModuleGetFunction)}({layout.Name}) failed with {Describe(result)}";
                    break;
                }

                functions[layout.Name] = new GcnFunction(this, function, layout, isa, format);
            }
        }

        if (error is null)
        {
            return true;
        }

        if (loaded != IntPtr.Zero && hipModuleUnload != null)
        {
            hipModuleUnload(loaded);
        }

        pinned.Free();
        module = IntPtr.Zero;
        functions.Clear();
        return false;
    }

    /// <summary>The kernel <paramref name="name"/> of the loaded set, or null when the kernels are unavailable.</summary>
    private GcnFunction? Gcn(string name) => KernelSet is { } set && set.Functions.TryGetValue(name, out var function) ? function : null;

    /// <summary>
    /// Launches <paramref name="function"/> over <paramref name="blocks"/> × <paramref name="blocksY"/> blocks of the block
    /// its metadata declares, its arguments packed from <paramref name="values"/> (one 64-bit value per argument, in the
    /// metadata's order: a pointer, a 32-bit value or a float's bits) at the offsets the metadata gives.
    /// </summary>
    /// <exception cref="InvalidOperationException">The values do not match the arguments the kernel declares (CUDA's signature check).</exception>
    private void LaunchKernel(GcnFunction function, uint blocks, uint blocksY, params ReadOnlySpan<ulong> values)
    {
        if (values.Length != function.Arguments.Count)
        {
            throw new InvalidOperationException($"Kernel {function.Name} declares {function.Arguments.Count} arguments but was launched with {values.Length}.");
        }

        Span<byte> buffer = stackalloc byte[MostArgumentBytes];
        var arguments = buffer[..function.ArgumentBytes];
        arguments.Clear();
        GcnKernels.PackArguments(function.Arguments, arguments, values);
        LaunchGcn(function, blocks, blocksY, (uint)function.Block, arguments);
    }

    // A count as a kernel's 32-bit argument, and a float's bits.
    private static ulong U(int value) => (uint)value;

    private static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    // Blocks for `work` items in blocks of `per` (rounded up), within the device's grid in x (rule 84: its reported limit).
    private uint BlocksFor(long work, int per) => (uint)Math.Clamp((work + per - 1) / per, 1, DeviceLimits.MaxGridDimX);
}
