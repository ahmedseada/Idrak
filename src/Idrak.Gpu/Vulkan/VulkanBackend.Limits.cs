// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Vulkan;

// What the kernels are shaped from: the limits the device reports (VulkanDeviceFacts; never its vendor or name). The
// workgroup width of every kernel follows from them (VulkanKernels.WidthFor); dispatch shapes read the workgroup
// counts; the choices no limit decides (split counts, kernel variants) are measured on the device
// (VulkanBackend.KernelTuning.cs).
internal sealed partial class VulkanBackend
{
    private DeviceLimits? _limits;

    /// <summary>This device's limits as the kernels use them, and the kernel width chosen from them.</summary>
    internal DeviceLimits Limits => _limits ??= DeviceLimits.Of(Facts, _physical.Properties);

    /// <summary>Invocations per workgroup of this device's kernels.</summary>
    internal int Width => Limits.Width;
}

/// <summary>
/// A device's compute limits as the kernels use them (from <see cref="VulkanDeviceFacts"/> and VkPhysicalDeviceLimits)
/// and the kernel width chosen from them.
/// </summary>
/// <param name="MaxInvocations">maxComputeWorkGroupInvocations.</param>
/// <param name="MaxSizeX">maxComputeWorkGroupSize[0].</param>
/// <param name="SharedBytes">maxComputeSharedMemorySize.</param>
/// <param name="MaxGroupsX">maxComputeWorkGroupCount[0].</param>
/// <param name="MaxGroupsY">maxComputeWorkGroupCount[1].</param>
/// <param name="MaxGroupsZ">maxComputeWorkGroupCount[2].</param>
/// <param name="SubgroupSize">subgroupSize (invocations that run in lockstep).</param>
/// <param name="SubgroupReductions">Whether compute shaders have subgroup shuffles (supportedStages has COMPUTE and
/// supportedOperations has BASIC and SHUFFLE): the kernels' workgroup reductions then finish in subgroups, as a butterfly
/// over the pairs workgroup memory's tree pairs (the same bits).</param>
/// <param name="Butterfly">The butterfly's invocations: the largest power of two no larger than the smallest subgroup the
/// device runs a kernel in (minSubgroupSize where reported, else subgroupSize), so a butterfly never leaves its subgroup.</param>
/// <param name="Width">Invocations per workgroup of the kernels (<see cref="VulkanKernels.WidthFor"/>, or
/// IDRAK_VULKAN_WIDTH within the limits).</param>
/// <param name="InfNanPreserve">Whether the kernels declare SignedZeroInfNanPreserve for 32-bit floats (where the device
/// reports it, <see cref="VulkanDeviceFacts.InfNanPreserve"/>, unless IDRAK_VULKAN_INF_NAN=0): the driver then keeps their
/// tests against ±∞ and NaN (masks, sentinels, the sampler's +∞ wins and NaN never drawn) as written. Without it a driver
/// may assume every float is finite and fold such a test either way, depending on the code around it.</param>
internal sealed record DeviceLimits(int MaxInvocations, int MaxSizeX, int SharedBytes, uint MaxGroupsX, uint MaxGroupsY, uint MaxGroupsZ,
    int SubgroupSize, bool SubgroupReductions, int Butterfly, int Width, bool InfNanPreserve)
{
    /// <summary>Tests: false makes backends created afterwards build their kernels without SignedZeroInfNanPreserve;
    /// IDRAK_VULKAN_INF_NAN=0 sets it for a process.</summary>
    internal static bool? InfNanOverride { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_VULKAN_INF_NAN") is "0" or "false" ? false : null;

    /// <summary>Tests: the kernel width every backend created afterwards uses (a power of two within the device's
    /// limits), instead of the formula's; IDRAK_VULKAN_WIDTH sets it for a process.</summary>
    internal static int? WidthOverride { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("IDRAK_VULKAN_WIDTH"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int width) ? width : null;

    /// <summary>Tests: false makes backends created afterwards reduce through workgroup memory alone, as on devices
    /// without subgroup arithmetic; IDRAK_VULKAN_SUBGROUPS=0 sets it for a process.</summary>
    internal static bool? SubgroupsOverride { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_VULKAN_SUBGROUPS") is "0" or "false" ? false : null;

    internal static DeviceLimits Of(VulkanDeviceFacts facts, in VkPhysicalDeviceProperties p)
    {
        int invocations = (int)Math.Min(facts.MaxWorkGroupInvocations, int.MaxValue);
        int sizeX = (int)Math.Min(facts.MaxWorkGroupSize.X, int.MaxValue);
        int shared = (int)Math.Min(facts.MaxSharedMemoryBytes, int.MaxValue);
        int smallest = (int)Math.Min(facts.MinSubgroupSize > 0 ? facts.MinSubgroupSize : facts.SubgroupSize, VulkanKernels.MaxWidth);
        int butterfly = smallest >= 2 ? (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)smallest + 1) / 2 : 0;
        bool reductions = butterfly >= 2 && facts.ComputeSubgroups(VulkanDeviceFacts.SubgroupBasic | VulkanDeviceFacts.SubgroupShuffle)
            && SubgroupsOverride != false;
        int chosen = VulkanKernels.WidthFor(invocations, sizeX, shared, (int)facts.SubgroupSize);
        if (WidthOverride is int forced && forced >= VulkanKernels.MinWidth && forced <= Math.Min(VulkanKernels.MaxWidth, Math.Min(invocations, sizeX))
            && (forced & (forced - 1)) == 0 && VulkanKernels.SharedBytesBound(forced) <= shared)
        {
            chosen = forced;
        }

        return new DeviceLimits(invocations, sizeX, shared, p.MaxComputeWorkGroupCountX, p.MaxComputeWorkGroupCountY, p.MaxComputeWorkGroupCountZ,
            (int)facts.SubgroupSize, reductions, reductions ? butterfly : 0, chosen, facts.InfNanPreserve && InfNanOverride != false);
    }

    /// <summary>The limits in one line, for benchmark headers.</summary>
    public override string ToString() =>
        $"width {Width}, {(SubgroupReductions ? $"subgroup (butterfly {Butterfly})" : "workgroup-memory")} reductions{(InfNanPreserve ? ", ±∞ and NaN kept" : "")}";
}
