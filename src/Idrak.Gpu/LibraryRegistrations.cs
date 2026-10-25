// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;

namespace Idrak.Gpu;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its GPU devices, CUDA first,
/// then Vulkan, then HIP, in <see cref="DeviceProviders"/>; and how the CUDA and Vulkan backends hand their measured
/// choices between a cluster's ranks, in <see cref="MeasuredChoiceProviders"/> (HIP measures none). <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, after the core assembly's own
/// registrations when the application ships it.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = RegisterDevices,
        [typeof(MeasuredChoiceProviders)] = () =>
        {
            MeasuredChoiceProviders.Register(new CudaMeasuredChoices());   // the ranks of a cluster run rank 0's measured choices
            MeasuredChoiceProviders.Register(new VulkanMeasuredChoices());
        },
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }

    private static void RegisterDevices()
    {
        DeviceProviders.Register(new CudaProvider());
        DeviceProviders.Register(new VulkanProvider());
        DeviceProviders.Register(new HipProvider());
    }

    private sealed class CudaProvider : ISeparableDeviceProvider
    {
        public string Name => "cuda";

        public string Summary => "GPUs through the CUDA driver (PTX kernels generated in C#)";

        public string Display => "CUDA";

        public DeviceType Type => DeviceType.Cuda;

        public int Count => CudaBackend.DeviceCount;

        public string? UnavailableReason => CudaBackend.UnavailableReason;

        public Backend Create(int ordinal) => CudaBackend.Get(ordinal);

        public bool IsStarted(int ordinal) => CudaBackend.IsInitialized(ordinal);

        public Backend CreateSeparate(int ordinal, SeparateBackendOptions options) => CudaBackend.CreateSeparate(ordinal, options);

        public int? DefaultRank(int ordinal) => ordinal == 0 ? 100 : 99;   // the first GPU, as before

        public Guid? DeviceUuid(int ordinal) => CudaBackend.DeviceUuid(ordinal);
    }
}
