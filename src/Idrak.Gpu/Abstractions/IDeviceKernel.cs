// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;
using Idrak.Gpu.Vulkan;

namespace Idrak.Gpu.Abstractions;

/// <summary>
/// A compute kernel a plug-in ships for one kind of device: PTX text (<see cref="Cuda.CudaKernel"/>) or SPIR-V words
/// (<see cref="Vulkan.VulkanKernel"/>), launched from a kernel the plug-in
/// registers for that kind (<see cref="Kernels.Register{TKernel}(Operation{TKernel}, string, TKernel, Func{Backend, bool})"/>).
/// It runs on any backend that launches kernels of its type (<see cref="IKernelLauncher{TKernel}"/>,
/// <see cref="IVulkanKernelDispatcher"/>): the library's backend of the kind implements it, and so may a backend
/// that drives the same hardware in its own way.
/// </summary>
public interface IDeviceKernel
{
    /// <summary>A name for messages and the profiler.</summary>
    string Name { get; }

    /// <summary>
    /// Queues the kernel on <paramref name="backend"/> over gridX × gridY × gridZ blocks (workgroups) of blockX × blockY ×
    /// blockZ threads, with <paramref name="arguments"/> in the order the kernel declares its parameters (storages of that
    /// device, integers, floats) and <paramref name="sharedBytes"/> of dynamic shared memory. It runs in order with the
    /// device's other work; reading a storage back waits for it. A grid with a zero dimension launches nothing.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <paramref name="backend"/> does not launch kernels of this type, a storage is not on it, or the arguments do not
    /// match the parameters.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">The grid, the block or the shared memory exceed the device's limits.</exception>
    void Launch(Backend backend, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, ReadOnlySpan<KernelArgument> arguments, uint sharedBytes = 0);
}

/// <summary>
/// A device that launches kernels of type <typeparamref name="TKernel"/> (the CUDA backend <see cref="Cuda.CudaKernel"/>s):
/// what a kernel's <see cref="IDeviceKernel.Launch"/> asks of the backend it
/// is given, instead of a check for one backend class.
/// </summary>
/// <typeparam name="TKernel">The kind of kernel the device launches.</typeparam>
public interface IKernelLauncher<in TKernel>
    where TKernel : class, IDeviceKernel
{
    /// <summary>
    /// Queues <paramref name="kernel"/>, its <paramref name="arguments"/> already checked against its parameters (see
    /// <see cref="IDeviceKernel.Launch"/>); the device checks the grid, the block and the shared memory against its limits.
    /// </summary>
    void Launch(TKernel kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, ReadOnlySpan<KernelArgument> arguments, uint sharedBytes);
}

/// <summary>
/// A device that dispatches <see cref="VulkanKernel"/>s (the Vulkan backend): what <see cref="VulkanKernel.Dispatch"/> asks of
/// the backend it is given, instead of a check for one backend class.
/// </summary>
public interface IVulkanKernelDispatcher
{
    /// <summary>
    /// Queues <paramref name="kernel"/> over groupsX × groupsY × groupsZ workgroups, binding i being
    /// <paramref name="storages"/>[i] and <paramref name="pushConstants"/> its push-constant block; the device checks both
    /// against the kernel.
    /// </summary>
    void Dispatch(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> pushConstants);
}
