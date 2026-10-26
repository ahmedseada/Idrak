// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What every device provides (plan 33): its kind and name, what its kernels can do, its memory, the copies to and from
/// it, and a point that waits for its work. <see cref="Backend"/> implements it; a new device derives from
/// <see cref="Backend"/>, which runs every operation through the host fallback until the device adds kernels of its own,
/// and adds the capability interfaces it has (<see cref="IGraphCaptureBackend"/>, <see cref="IKernelProfiler"/>,
/// <see cref="IFloat8Backend"/>, <see cref="IRowAttentionBackend"/>, <see cref="IMaskedCacheAttentionBackend"/>,
/// <see cref="ISegmentedAttentionBackend"/>,
/// <see cref="IFusedPackedKernels"/>, <see cref="IMatrixUnitBackend"/>, <see cref="IRecurrentKernels"/>,
/// <see cref="IOffloadingBackend"/>, <see cref="IPeerCopyBackend"/>, <see cref="IMemoryReport"/>,
/// <see cref="ILongLivedAllocations"/>), each read once when the backend starts.
/// </summary>
public interface IDeviceBackend
{
    /// <summary>The kind of device this drives ("cpu", "cuda", "vulkan", …): the kernels registered for it run here.</summary>
    string Kind { get; }

    /// <summary>A human-readable name of the device, such as the GPU model.</summary>
    string Name { get; }

    /// <summary>What this device's kernels can do (see <see cref="BackendCapabilities"/>).</summary>
    BackendCapabilities Capabilities { get; }

    /// <summary>
    /// A block of <paramref name="length"/> floats of device memory (from the device's pool where it keeps one); zeroed
    /// when <paramref name="zeroed"/>, else holding whatever was there.
    /// </summary>
    Storage Allocate(int length, bool zeroed);

    /// <summary>Takes back the memory of a storage whose last reference was released, or that an eviction empties.</summary>
    void Return(Storage storage);

    /// <summary>
    /// Points <paramref name="storage"/> at no memory after its memory went back with <see cref="Return"/> (an eviction);
    /// kernels given it fail until <see cref="Attach"/>. Only the device's own memory management calls it.
    /// </summary>
    void Detach(Storage storage);

    /// <summary>
    /// Points <paramref name="storage"/> (evicted) at the memory of <paramref name="fresh"/>, a block just allocated for
    /// it, which is not used on its own afterwards. Only the device's own memory management calls it.
    /// </summary>
    void Attach(Storage storage, Storage fresh);

    /// <summary>Copies <paramref name="source"/> into the first source.Length floats of <paramref name="destination"/>.</summary>
    void Upload(ReadOnlySpan<float> source, Storage destination);

    /// <summary>Copies the first destination.Length floats of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    void Download(Storage source, Span<float> destination);

    /// <summary>Copies destination.Length floats starting at element <paramref name="offset"/>.</summary>
    void DownloadRange(Storage source, int offset, Span<float> destination);

    /// <summary>The bytes in use, cached and offloaded, and the limit (<see cref="ComputeResources.GetMemoryUsage"/>).</summary>
    MemoryUsage GetMemoryUsage();

    /// <summary>Frees pooled blocks that no tensor is using.</summary>
    void ReleaseCachedMemory();

    /// <summary>Waits until every piece of work queued on the device has finished.</summary>
    void Synchronize();
}

/// <summary>
/// A storage whose values are in memory the process reads directly (the CPU device's): callers read and write them in
/// place instead of copying with <see cref="Backend.Download"/> and <see cref="Backend.Upload"/>, and a host fallback
/// (<c>HostCall</c>) takes them as they are, without a copy to the device and back.
/// </summary>
public interface IHostStorage
{
    /// <summary>The values: the storage's first <see cref="Storage.Length"/> floats.</summary>
    Span<float> HostMemory { get; }
}
