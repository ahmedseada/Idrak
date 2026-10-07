// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;

namespace Idrak.Gpu;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its GPU devices, CUDA first,
/// then Vulkan, then HIP, in <see cref="DeviceProviders"/>. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, after the core assembly's own
/// registrations when the application ships it.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = RegisterDevices,
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

    private sealed class CudaProvider : DeviceProvider
    {
        public override string Kind => "cuda";

        public override string Display => "CUDA";

        public override DeviceType Type => DeviceType.Cuda;

        public override int Count => CudaBackend.DeviceCount;

        public override string? UnavailableReason => CudaBackend.UnavailableReason;

        public override Backend Create(int ordinal) => CudaBackend.Get(ordinal);

        public override bool IsStarted(int ordinal) => CudaBackend.IsInitialized(ordinal);

        public override int? DefaultRank(int ordinal) => ordinal == 0 ? 100 : 99;   // the first GPU, as before

        public override Guid? DeviceUuid(int ordinal) => CudaBackend.DeviceUuid(ordinal);
    }
}
