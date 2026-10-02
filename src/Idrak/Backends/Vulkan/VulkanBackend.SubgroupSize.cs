// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// Subgroup size control (VK_EXT_subgroup_size_control, core in Vulkan 1.3). Where the device reports it (the
// subgroupSizeControl feature, and compute among requiredSubgroupSizeStages), the feature is enabled when the device is
// created, and a kernel that uses subgroup operations may have its pipeline require one of the sizes the device runs
// (minSubgroupSize to maxSubgroupSize; whole subgroups too where computeFullSubgroups is reported). Which size, or the
// device's default, is measured per kernel and width the first time the kernel runs (the existing tuning pattern: every
// candidate on the real inputs writing scratch copies of the outputs, the default kept unless another is clearly
// faster) and stored with the other kernel choices. The kernels' reductions read the subgroup size at run time, so a
// pipeline of any size computes the same thing (summed in a different order when the size differs). Devices without
// the feature, and IDRAK_VULKAN_SUBGROUP_SIZE=0, keep every pipeline as it was.
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>What a device reports of subgroup size control.</summary>
    /// <param name="Enabled">The feature is reported and compute is among requiredSubgroupSizeStages: enabled at device creation.</param>
    /// <param name="Extension">Enabled through VK_EXT_subgroup_size_control (else as core Vulkan 1.3).</param>
    /// <param name="FullSubgroups">computeFullSubgroups: sized pipelines also require whole subgroups.</param>
    internal readonly record struct SubgroupSizeSupport(bool Enabled, bool Extension, bool FullSubgroups);

    /// <summary>Tests: false makes backends created afterwards leave subgroup size control off;
    /// IDRAK_VULKAN_SUBGROUP_SIZE=0 sets it for a process.</summary>
    internal static bool? SubgroupSizeControlOverride { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_VULKAN_SUBGROUP_SIZE") is "0" or "false" ? false : null;

    private readonly SubgroupSizeSupport _sizeControl;

    // Kernels by the kernel they are a sized variant of (the chosen one, or the kernel itself for the default), and the
    // variants by (kernel, size). Guarded by themselves.
    private readonly Dictionary<VulkanKernel, VulkanKernel> _sized = [];
    private readonly Dictionary<(VulkanKernel, int), VulkanKernel> _sizeVariants = [];

    /// <summary>Whether subgroup size control is enabled on this device.</summary>
    internal bool SubgroupSizeControl => _sizeControl.Enabled;

    /// <summary>Whether sized pipelines also require whole subgroups (computeFullSubgroups).</summary>
    internal bool FullSubgroups => _sizeControl.FullSubgroups;

    // What the device reports, read before the device is created.
    private static SubgroupSizeSupport ProbeSubgroupSizeControl(PhysicalDevice physical)
    {
        if (SubgroupSizeControlOverride == false)
        {
            return default;
        }

        uint instance = MakeVersion(1, 0);
        if (Exports(nameof(vkEnumerateInstanceVersion)) && vkEnumerateInstanceVersion(out uint loader) == Success)
        {
            instance = Math.Min(loader & ~0xFFFu, MakeVersion(1, 3));                // as the instance was created
        }

        bool extension = HasExtension(physical.Handle, SubgroupSizeControlExtension);
        bool core = Math.Min(physical.Properties.ApiVersion, instance) >= MakeVersion(1, 3);
        if (!extension && !core)
        {
            return default;
        }

        var sizes = new VkPhysicalDeviceSubgroupSizeControlProperties { SType = StructurePhysicalDeviceSubgroupSizeControlProperties };
        var properties = new VkPhysicalDeviceProperties2 { SType = StructurePhysicalDeviceProperties2, PNext = &sizes };
        vkGetPhysicalDeviceProperties2(physical.Handle, &properties);
        var features = new VkPhysicalDeviceSubgroupSizeControlFeatures { SType = StructurePhysicalDeviceSubgroupSizeControlFeatures };
        var all = new VkPhysicalDeviceFeatures2 { SType = StructurePhysicalDeviceFeatures2, PNext = &features };
        vkGetPhysicalDeviceFeatures2(physical.Handle, &all);
        bool enabled = features.SubgroupSizeControl != 0 && (sizes.RequiredSubgroupSizeStages & ShaderStageCompute) != 0
            && sizes.MinSubgroupSize > 0 && sizes.MaxSubgroupSize >= sizes.MinSubgroupSize;
        return new SubgroupSizeSupport(enabled, enabled && extension, enabled && features.ComputeFullSubgroups != 0);
    }

    /// <summary>
    /// The subgroup sizes a kernel of workgroup width <paramref name="width"/> may require here: the powers of two from
    /// minSubgroupSize to maxSubgroupSize that divide the width and of which maxComputeWorkgroupSubgroups cover it (none
    /// without subgroup size control).
    /// </summary>
    internal int[] SubgroupSizeCandidates(int width)
    {
        if (!_sizeControl.Enabled || width <= 0)
        {
            return [];
        }

        var sizes = new List<int>();
        for (long s = Math.Max(Facts.MinSubgroupSize, 1); s <= Facts.MaxSubgroupSize; s *= 2)
        {
            if ((s & (s - 1)) == 0 && s <= width && width % s == 0
                && (Facts.MaxComputeWorkgroupSubgroups == 0 || width <= s * Facts.MaxComputeWorkgroupSubgroups))
            {
                sizes.Add((int)s);
            }
        }

        return [.. sizes];
    }

    /// <summary>The subgroup size chosen for kernel <paramref name="name"/> (0: the device's default), or null when none
    /// was chosen for it (for tests and diagnostics).</summary>
    internal int? ChosenSubgroupSize(string name)
    {
        lock (_sized)
        {
            foreach (var (kernel, chosen) in _sized)
            {
                if (kernel.Name == name)
                {
                    return chosen.RequiredSubgroupSize;
                }
            }
        }

        return null;
    }

    // Dispatches a generated kernel: its sized variant where subgroup size control is on and it uses subgroup
    // operations (the size measured on its first run here) unless it keeps the default size, else the kernel itself.
    private void DispatchKernel(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (_sizeControl.Enabled && kernel.UsesSubgroups && !kernel.DefaultSubgroupSize)
        {
            kernel = Sized(kernel, groupsX, groupsY, groupsZ, storages, push);
        }

        Dispatch(kernel, groupsX, groupsY, groupsZ, storages, push);
    }

    // The variant of `kernel` to run: known, stored, or measured now (the default while measuring is impossible).
    private VulkanKernel Sized(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (kernel.LastSized is SizedChoice last && ReferenceEquals(last.Owner, this))
        {
            return last.Kernel;                                                // looked up without a lock
        }

        lock (_sized)
        {
            if (_sized.TryGetValue(kernel, out var known))
            {
                return known;
            }
        }

        int width = kernel.LocalSizeX;
        int[] sizes = SubgroupSizeCandidates(width);
        if (sizes.Length == 0)
        {
            return Remember(kernel, 0);
        }

        var key = new VulkanTuneKey(VulkanTuneOp.SubgroupSize, StableId(kernel.Name), width, (int)Facts.MinSubgroupSize, (int)Facts.MaxSubgroupSize);
        if (TryTuned(key, out int choice) && (choice == 0 || Array.IndexOf(sizes, choice) >= 0))
        {
            return Remember(kernel, choice);
        }

        if (!Autotune || t_timing)
        {
            return kernel;                                                     // measured on a later run
        }

        // Candidates: the device's default (0), then each size it may require.
        var pushed = push.ToArray();
        (uint gx, uint gy, uint gz) = (groupsX, groupsY, groupsZ);
        choice = TuneWithScratch(key, [0, .. sizes], 0, storages.ToArray(), kernel.Writes,
            (size, bound) => Dispatch(Variant(kernel, size), gx, gy, gz, bound, pushed));
        return Remember(kernel, choice);
    }

    private VulkanKernel Remember(VulkanKernel kernel, int size)
    {
        var chosen = Variant(kernel, size);
        lock (_sized)
        {
            _sized[kernel] = chosen;
        }

        kernel.LastSized = new SizedChoice(this, chosen);
        return chosen;
    }

    // The variant a backend chose for a kernel (kept on the kernel for the backend that chose last).
    private sealed record SizedChoice(VulkanBackend Owner, VulkanKernel Kernel);

    // `kernel` with its pipeline requiring subgroups of `size` (the kernel itself for 0), made once.
    private VulkanKernel Variant(VulkanKernel kernel, int size)
    {
        if (size == 0)
        {
            return kernel;
        }

        lock (_sizeVariants)
        {
            if (!_sizeVariants.TryGetValue((kernel, size), out var variant))
            {
                variant = new VulkanKernel(kernel.Spirv, kernel.Bindings, kernel.PushConstantBytes, kernel.Name, kernel.Writes) { RequiredSubgroupSize = size };
                _sizeVariants[(kernel, size)] = variant;
            }

            return variant;
        }
    }
}
