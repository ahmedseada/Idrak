// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

/// <summary>
/// What a Vulkan device reports about itself that kernels and the runtime choose by (read once, through one
/// vkGetPhysicalDeviceProperties2 chain): its identity across APIs, its subgroups, its workgroup and shared-memory limits
/// and its allocation limits. Choices read these, never the device's or vendor's name. Core Vulkan reports no count of
/// compute units, so none is given.
/// </summary>
internal sealed record VulkanDeviceFacts
{
    // VkSubgroupFeatureFlagBits.
    public const uint SubgroupBasic = 0x1, SubgroupVote = 0x2, SubgroupArithmetic = 0x4, SubgroupBallot = 0x8,
        SubgroupShuffle = 0x10, SubgroupShuffleRelative = 0x20, SubgroupClustered = 0x40, SubgroupQuad = 0x80;

    /// <summary>VK_PHYSICAL_DEVICE_TYPE_* (integrated, discrete, virtual, CPU, other).</summary>
    public uint DeviceType { get; init; }

    /// <summary>deviceUUID: the same GPU reports the same UUID through every API (CUDA's cuDeviceGetUuid too).</summary>
    public Guid DeviceUuid { get; init; }

    /// <summary>The device's PCI address (domain, bus, device, function) when it reports one (VK_EXT_pci_bus_info), else null.</summary>
    public (uint Domain, uint Bus, uint Device, uint Function)? PciAddress { get; init; }

    /// <summary>driverUUID: changes with the driver build (measured choices are cached per device and driver).</summary>
    public Guid DriverUuid { get; init; }

    /// <summary>The driver's version number, as the driver encodes it.</summary>
    public uint DriverVersion { get; init; }

    /// <summary>The default subgroup size (lanes that run in lockstep).</summary>
    public uint SubgroupSize { get; init; }

    /// <summary>The smallest and largest subgroup sizes the device runs (both <see cref="SubgroupSize"/> when it does not say).</summary>
    public uint MinSubgroupSize { get; init; }

    /// <inheritdoc cref="MinSubgroupSize"/>
    public uint MaxSubgroupSize { get; init; }

    /// <summary>The most subgroups in one compute workgroup (0 when the device does not say).</summary>
    public uint MaxComputeWorkgroupSubgroups { get; init; }

    /// <summary>The shader stages with subgroup operations (VkShaderStageFlags).</summary>
    public uint SubgroupStages { get; init; }

    /// <summary>The subgroup operations supported (VkSubgroupFeatureFlags: <see cref="SubgroupBasic"/>, ...).</summary>
    public uint SubgroupOperations { get; init; }

    /// <summary>Whether quad operations work in every stage that has subgroup operations.</summary>
    public bool QuadOperationsInAllStages { get; init; }

    /// <summary>The most invocations in one compute workgroup.</summary>
    public uint MaxWorkGroupInvocations { get; init; }

    /// <summary>The largest compute workgroup in each dimension.</summary>
    public (uint X, uint Y, uint Z) MaxWorkGroupSize { get; init; }

    /// <summary>Shared (workgroup) memory per workgroup, in bytes.</summary>
    public uint MaxSharedMemoryBytes { get; init; }

    /// <summary>The largest storage buffer a kernel binds, in bytes.</summary>
    public uint MaxStorageBufferRange { get; init; }

    /// <summary>The largest single memory allocation, in bytes (maxMemoryAllocationSize).</summary>
    public ulong MaxMemoryAllocationSize { get; init; }

    /// <summary>The most descriptors in one descriptor set.</summary>
    public uint MaxPerSetDescriptors { get; init; }

    /// <summary>Whether compute shaders have subgroup operations of every kind in <paramref name="operations"/>.</summary>
    public bool ComputeSubgroups(uint operations) => (SubgroupStages & ShaderStageCompute) != 0 && (SubgroupOperations & operations) == operations;

    /// <summary>One line for benchmark headers and diagnostics.</summary>
    public string Describe()
    {
        string subgroups = MinSubgroupSize != MaxSubgroupSize ? $"{SubgroupSize} ({MinSubgroupSize}-{MaxSubgroupSize})" : $"{SubgroupSize}";
        var ops = new List<string>();
        foreach (var (bit, name) in new[] { (SubgroupBasic, "basic"), (SubgroupVote, "vote"), (SubgroupArithmetic, "arithmetic"),
                     (SubgroupBallot, "ballot"), (SubgroupShuffle, "shuffle"), (SubgroupShuffleRelative, "relative"),
                     (SubgroupClustered, "clustered"), (SubgroupQuad, "quad") })
        {
            if ((SubgroupOperations & bit) != 0)
            {
                ops.Add(name);
            }
        }

        string compute = (SubgroupStages & ShaderStageCompute) != 0 ? "" : ", not in compute";
        return $"subgroup {subgroups} [{string.Join(' ', ops)}{compute}]; workgroup ≤ {MaxWorkGroupInvocations} " +
               $"({MaxWorkGroupSize.X}×{MaxWorkGroupSize.Y}×{MaxWorkGroupSize.Z}); shared {MaxSharedMemoryBytes >> 10} KiB; " +
               $"storage ≤ {MaxStorageBufferRange >> 20} MiB; allocation ≤ {MaxMemoryAllocationSize >> 20} MiB";
    }
}

internal sealed unsafe partial class VulkanBackend
{
    /// <summary>What the device reports about itself (subgroups, workgroup and memory limits, UUIDs).</summary>
    internal VulkanDeviceFacts Facts => _physical.Facts;

    /// <summary>The device facts and the device's own one-line summary (for benchmark headers).</summary>
    internal string Describe() =>
        $"{Name}: storages {DescribeStorage()} on a {StorageHeapBytes >> 20:N0} MiB heap{(ReadsThroughStaging ? ", reads through staging" : "")}, " +
        $"{(PushDescriptors ? "pushed descriptors" : "descriptor sets")} ({PushDescriptorsChoice}), pages of {PageBytes >> 20} MiB; {Facts.Describe()}; {DescribeMatrixUnits()}";

    // The device's facts and its driver's name, from one vkGetPhysicalDeviceProperties2 chain (every structure in it
    // valid for the device's and the instance's Vulkan versions and its extensions).
    private static (VulkanDeviceFacts Facts, string Driver) QueryFacts(IntPtr physical, in VkPhysicalDeviceProperties properties, uint instanceVersion)
    {
        var id = new VkPhysicalDeviceIdProperties { SType = StructurePhysicalDeviceIdProperties };
        var subgroup = new VkPhysicalDeviceSubgroupProperties { SType = StructurePhysicalDeviceSubgroupProperties, PNext = &id };
        var maintenance = new VkPhysicalDeviceMaintenance3Properties { SType = StructurePhysicalDeviceMaintenance3Properties, PNext = &subgroup };
        var driver = new VkPhysicalDeviceDriverProperties { SType = StructurePhysicalDeviceDriverProperties };
        var sizes = new VkPhysicalDeviceSubgroupSizeControlProperties { SType = StructurePhysicalDeviceSubgroupSizeControlProperties };
        void* chain = &maintenance;
        uint version = Math.Min(properties.ApiVersion, instanceVersion);
        bool hasDriver = version >= MakeVersion(1, 2);
        if (hasDriver)
        {
            driver.PNext = chain;
            chain = &driver;
        }

        bool hasSizes = version >= MakeVersion(1, 3) || HasExtension(physical, SubgroupSizeControlExtension);
        if (hasSizes)
        {
            sizes.PNext = chain;
            chain = &sizes;
        }

        var pci = new VkPhysicalDevicePciBusInfoProperties { SType = StructurePhysicalDevicePciBusInfoProperties };
        bool hasPci = HasExtension(physical, PciBusInfoExtension);
        if (hasPci)
        {
            pci.PNext = chain;
            chain = &pci;
        }

        var all = new VkPhysicalDeviceProperties2 { SType = StructurePhysicalDeviceProperties2, PNext = chain };
        vkGetPhysicalDeviceProperties2(physical, &all);

        string name = hasDriver ? Utf8(driver.DriverName, 256) : "", info = hasDriver ? Utf8(driver.DriverInfo, 256) : "";
        string driverText = name.Length == 0 ? $"driver 0x{properties.DriverVersion:X}" : info.Length > 0 ? $"{name} {info}" : name;
        bool sized = hasSizes && sizes.MinSubgroupSize > 0 && sizes.MaxSubgroupSize >= sizes.MinSubgroupSize;
        var facts = new VulkanDeviceFacts
        {
            DeviceType = properties.DeviceType,
            DeviceUuid = new Guid(new ReadOnlySpan<byte>(id.DeviceUuid, 16)),
            PciAddress = hasPci ? (pci.PciDomain, pci.PciBus, pci.PciDevice, pci.PciFunction) : null,
            DriverUuid = new Guid(new ReadOnlySpan<byte>(id.DriverUuid, 16)),
            DriverVersion = properties.DriverVersion,
            SubgroupSize = subgroup.SubgroupSize,
            MinSubgroupSize = sized ? sizes.MinSubgroupSize : subgroup.SubgroupSize,
            MaxSubgroupSize = sized ? sizes.MaxSubgroupSize : subgroup.SubgroupSize,
            MaxComputeWorkgroupSubgroups = hasSizes ? sizes.MaxComputeWorkgroupSubgroups : 0,
            SubgroupStages = subgroup.SupportedStages,
            SubgroupOperations = subgroup.SupportedOperations,
            QuadOperationsInAllStages = subgroup.QuadOperationsInAllStages != 0,
            MaxWorkGroupInvocations = properties.MaxComputeWorkGroupInvocations,
            MaxWorkGroupSize = (properties.MaxComputeWorkGroupSizeX, properties.MaxComputeWorkGroupSizeY, properties.MaxComputeWorkGroupSizeZ),
            MaxSharedMemoryBytes = properties.MaxComputeSharedMemorySize,
            MaxStorageBufferRange = properties.MaxStorageBufferRange,
            MaxMemoryAllocationSize = maintenance.MaxMemoryAllocationSize,
            MaxPerSetDescriptors = maintenance.MaxPerSetDescriptors,
        };
        return (facts, driverText);
    }
}
