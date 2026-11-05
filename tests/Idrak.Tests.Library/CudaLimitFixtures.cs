// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Cuda;

// Test fixtures: CUDA device limits as NVIDIA documents them per compute capability, so the shape and tuning tests can
// judge what the kernels give on devices other than the one at hand. The library reads a device's limits from the device
// itself (CudaDeviceLimits.Read) and holds no such table (the owner's rule: no card or capability list in the library).
internal static class CudaLimitFixtures
{
    /// <summary>
    /// Limits as NVIDIA documents them per compute capability (CUDA C++ Programming Guide, "Technical Specifications per
    /// Compute Capability"): the multiprocessors, memory and L2 as given.
    /// </summary>
    internal static CudaDeviceLimits Documented(int major, int minor, int multiprocessors, long memoryBytes = 8L << 30, int l2Bytes = 0)
    {
        int cc = major * 10 + minor;
        int optinKiB = cc switch
        {
            >= 120 => 99,       // 12.x
            >= 100 => 227,      // 10.x
            >= 90 => 227,       // 9.0
            >= 89 => 99,        // 8.9
            >= 87 => 163,       // 8.7
            >= 86 => 99,        // 8.6
            >= 80 => 163,       // 8.0
            >= 75 => 64,        // 7.5
            >= 70 => 96,        // 7.x
            _ => 48,
        };
        int threadsPerSm = cc switch { 75 => 1024, 86 or 87 or 89 or >= 120 => 1536, _ => 2048 };
        return new CudaDeviceLimits
        {
            ComputeMajor = major,
            ComputeMinor = minor,
            Multiprocessors = multiprocessors,
            WarpSize = 32,
            MaxThreadsPerBlock = 1024,
            MaxThreadsPerMultiprocessor = threadsPerSm,
            SharedPerBlock = 48 * 1024,
            SharedPerBlockOptin = optinKiB * 1024,
            RegistersPerBlock = 65536,
            RegistersPerMultiprocessor = 65536,
            L2Bytes = l2Bytes,
            CooperativeLaunch = cc >= 60,                                                // compute 6.0 on
            MaxGridY = 65535,
            MaxGridZ = 65535,
            MemoryBytes = memoryBytes,
        };
    }
}
