// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using static Idrak.Gpu.Vulkan.VulkanDriver;

namespace Idrak.Gpu.Vulkan;

// A kernel the driver refuses never takes the device down. A driver may refuse to build a pipeline its device reports it
// can run (vkCreateComputePipelines: VK_ERROR_UNKNOWN for one product kernel on one phone's driver, while another driver
// of the same family built it). The ladder, once per kernel and backend (the refusal is kept, so the driver is asked once
// per process; every later dispatch finds it in one lookup):
//   1. the same SPIR-V without the required subgroup size and without whole subgroups, where the kernel allows it (its
//      reductions read the subgroup size at run time; a kernel built for one size, such as an emulated cooperative
//      matrix, keeps its request and is refused instead: rule 87);
//   2. the kernel's next variant, where it has variants (the packed products: the other words per row of k, the other
//      widths, then the smaller row blocks: VulkanBackend.Kernels.cs, PackedProduct), and a refused candidate of a
//      measurement is left out of it (VulkanBackend.KernelTuning.cs) and a refused width out of the width probe;
//   3. the operation's fallback through the dispatcher (Backend.BuildsKernelsWhenUsed): its host fallback, its composed
//      path, or the caller's own path.
// What was refused, how, and what it fell back to are in the device's facts ("refused kernels").
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>
    /// Tests and diagnostics: kernels whose pipelines backends started afterwards treat as refused (VK_ERROR_UNKNOWN)
    /// without asking the driver: a kernel name, or "name:sized" to refuse it only with its subgroup-size request (the
    /// retry without it then builds). IDRAK_VULKAN_REFUSE_KERNELS sets it for a process ("int8_gemv_8,softmax_rows:sized").
    /// </summary>
    internal static IReadOnlyList<string>? RefuseKernels { get; set; } = ParseRefusals(Environment.GetEnvironmentVariable("IDRAK_VULKAN_REFUSE_KERNELS"));

    /// <summary>The names of <see cref="RefuseKernels"/> from a comma list (null when empty).</summary>
    internal static IReadOnlyList<string>? ParseRefusals(string? list) =>
        list?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is { Length: > 0 } names ? names : null;

    // This backend's refused kernels (taken from RefuseKernels when it started), by kernel object (a name at a width and
    // reduction); guarded by itself. _anyRefused: whether any is (the paths that look them up skip the lookup otherwise).
    private readonly (string Name, bool SizedOnly)[] _injected = Injected();
    private readonly ConcurrentDictionary<VulkanKernel, KernelRefusal> _refused = new(ReferenceEqualityComparer.Instance);
    private readonly ConcurrentDictionary<VulkanKernel, string> _unsized = new(ReferenceEqualityComparer.Instance);
    private volatile bool _anyRefused;
    private int _pipelineAttempts;

    private static (string Name, bool SizedOnly)[] Injected() =>
        RefuseKernels is { } names ? [.. names.Select(n => n.EndsWith(":sized", StringComparison.Ordinal) ? (n[..^":sized".Length], true) : (n, false))] : [];

    /// <summary>A refused kernel: what was tried, the error every later dispatch of it gets, and what ran instead.</summary>
    private sealed class KernelRefusal(string kernel, int width, string tried, KernelUnavailableException error)
    {
        public string Kernel { get; } = kernel;

        public int Width { get; } = width;

        public string Tried { get; } = tried;

        public KernelUnavailableException Error { get; } = error;

        // What ran instead ("Int8MatMul: the host", "the packed products: int8_gemv_4"), in the order found; guarded by itself.
        public List<string> Instead { get; } = [];
    }

    /// <inheritdoc />
    protected override bool BuildsKernelsWhenUsed => true;

    /// <summary>Pipelines this backend asked the driver for (or the injection refused), each attempt counted (tests).</summary>
    internal int PipelineAttempts => Volatile.Read(ref _pipelineAttempts);

    /// <summary>The kernels refused on this backend, by name and width (tests and listings).</summary>
    internal IReadOnlyList<(string Kernel, int Width)> RefusedKernels => [.. _refused.Values.Select(r => (r.Kernel, r.Width)).Distinct()];

    /// <summary>Whether the kernel <paramref name="name"/> built for <paramref name="width"/> was refused here.</summary>
    internal bool IsRefused(string name, int width) => _anyRefused && _refused.Values.Any(r => r.Kernel == name && r.Width == width);

    // Whether the injection refuses this kernel (with its subgroup-size request when `sized`).
    private bool Injects(string name, bool sized)
    {
        foreach (var (kernel, sizedOnly) in _injected)
        {
            if (kernel == name && (sized || !sizedOnly))
            {
                return true;
            }
        }

        return false;
    }

    // Builds the pipeline of `kernel` from `module`: as asked (its required subgroup size and whole subgroups where
    // subgroup size control is on), then without that request where the kernel allows it; or throws its refusal.
    private ulong BuildPipeline(VulkanKernel kernel, ulong module, ulong layout)
    {
        bool sized = kernel.RequiredSubgroupSize > 0 && _sizeControl.Enabled;       // VulkanBackend.SubgroupSize.cs
        bool full = sized && _sizeControl.FullSubgroups;
        int result = CreatePipeline(kernel, module, layout, sized, full, out ulong handle);
        if (result == Success)
        {
            return handle;
        }

        var tried = new List<string>
        {
            sized ? string.Create(CultureInfo.InvariantCulture, $"with subgroups of {kernel.RequiredSubgroupSize}{(full ? ", whole" : "")}: {VulkanDriver.Describe(result)}") : VulkanDriver.Describe(result),
        };
        if (sized && !kernel.SubgroupSizeFixed)
        {
            result = CreatePipeline(kernel, module, layout, sized: false, full: false, out handle);
            if (result == Success)
            {
                _unsized[kernel] = string.Create(CultureInfo.InvariantCulture,
                    $"{kernel.Name} at width {kernel.LocalSizeX}: built without its request for subgroups of {kernel.RequiredSubgroupSize} ({tried[0]})");
                return handle;
            }

            tried.Add($"without the subgroup-size request: {VulkanDriver.Describe(result)}");
        }
        else if (sized)
        {
            tried.Add("not retried without the request: the kernel is written for that subgroup size");
        }

        throw Refuse(kernel, $"vkCreateComputePipelines {string.Join("; ", tried)}");
    }

    private int CreatePipeline(VulkanKernel kernel, ulong module, ulong layout, bool sized, bool full, out ulong handle)
    {
        Interlocked.Increment(ref _pipelineAttempts);
        handle = 0;
        if (_injected.Length > 0 && Injects(kernel.Name, sized))
        {
            return -13;                                                            // VK_ERROR_UNKNOWN, as a driver answered
        }

        byte* entry = stackalloc byte[] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
        var required = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
        {
            SType = StructurePipelineShaderStageRequiredSubgroupSizeCreateInfo,
            RequiredSubgroupSize = (uint)kernel.RequiredSubgroupSize,
        };
        var info = new VkComputePipelineCreateInfo
        {
            SType = StructureComputePipelineCreateInfo,
            Stage = new VkPipelineShaderStageCreateInfo
            {
                SType = StructurePipelineShaderStageCreateInfo,
                PNext = sized ? &required : null,
                Flags = full ? PipelineStageRequireFullSubgroups : 0,
                Stage = ShaderStageCompute,
                Module = module,
                Name = entry,
            },
            Layout = layout,
            BasePipelineIndex = -1,
        };
        ulong built;
        int result = vkCreateComputePipelines(_device, _pipelineCache, 1, &info, null, &built);
        handle = result == Success ? built : 0;
        return result;
    }

    // Keeps `kernel` refused here and returns the error its dispatches get: reported to telemetry once, here (the
    // dispatches' errors are not reported again).
    private KernelUnavailableException Refuse(VulkanKernel kernel, string tried)
    {
        string message = string.Create(CultureInfo.InvariantCulture,
            $"{Name} could not build Vulkan kernel '{kernel.Name}' at width {kernel.LocalSizeX} ({tried}); it is unavailable on this device, and what uses it runs another variant or its fallback");
        KernelUnavailableException error;
        using (DeviceException.Handled())
        {
            error = new KernelUnavailableException("vulkan", kernel.Name, message);
        }

        var refusal = _refused.GetOrAdd(kernel, _ => new KernelRefusal(kernel.Name, kernel.LocalSizeX, tried, error));
        _anyRefused = true;
        if (Telemetry.IsEnabled(TelemetryLevel.Devices) && ReferenceEquals(refusal.Error, error))
        {
            Telemetry.DeviceFailed(new DeviceFailed("vulkan", null, message, RetriedOnHost: false, Hint: null));
        }

        return refusal.Error;
    }

    // The kept refusal of `kernel`, thrown on its every dispatch after the first (no driver call).
    private void ThrowIfRefused(VulkanKernel kernel)
    {
        if (_anyRefused && _refused.TryGetValue(kernel, out var refusal))
        {
            throw refusal.Error;
        }
    }

    /// <inheritdoc />
    protected override void KernelFellBack(string operation, KernelUnavailableException refused, string fallback) =>
        NoteInstead(refused.Kernel, $"{operation}: {fallback}");

    // Records what ran instead of the refused kernel `name` (at every width it was refused at).
    private void NoteInstead(string name, string instead)
    {
        foreach (var refusal in _refused.Values)
        {
            if (refusal.Kernel == name)
            {
                lock (refusal.Instead)
                {
                    if (!refusal.Instead.Contains(instead))
                    {
                        refusal.Instead.Add(instead);
                    }
                }
            }
        }
    }

    /// <summary>The refused kernels for listings: "none", or each with what was tried and what ran instead.</summary>
    internal string DescribeRefusals()
    {
        if (!_anyRefused)
        {
            return "none";
        }

        // One entry per name and width (a kernel built with and without subgroup reductions is two objects).
        return string.Join("; ", _refused.Values.GroupBy(r => (r.Kernel, r.Width)).OrderBy(g => g.Key.Kernel, StringComparer.Ordinal).ThenBy(g => g.Key.Width).Select(g =>
        {
            var instead = new List<string>();
            foreach (var r in g)
            {
                lock (r.Instead)
                {
                    instead.AddRange(r.Instead.Where(i => !instead.Contains(i)));
                }
            }

            return string.Create(CultureInfo.InvariantCulture,
                $"{g.Key.Kernel} at width {g.Key.Width} ({g.First().Tried}); instead: {(instead.Count == 0 ? "nothing has needed it since" : string.Join(", ", instead))}");
        }));
    }

    /// <summary>The kernels built without their subgroup-size request, for listings ("none" when every request was kept).</summary>
    internal string DescribeUnsized() => _unsized.IsEmpty ? "none" : string.Join("; ", _unsized.Values.Order(StringComparer.Ordinal));
}
