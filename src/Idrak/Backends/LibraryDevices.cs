// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends.Cuda;
using Idrak.Backends.Hip;
using Idrak.Backends.Vulkan;

namespace Idrak.Backends;

/// <summary>
/// The GPU devices this assembly ships: CUDA first, then Vulkan, then HIP. <see cref="DeviceProviders"/> (in
/// Idrak.Abstraction, which cannot name them) asks for them by name the first time it is used.
/// </summary>
internal static class LibraryDevices
{
    private static IEnumerable<DeviceProvider> Providers() => [new CudaProvider(), new VulkanProvider(), new HipProvider()];

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
