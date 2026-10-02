// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using static Idrak.Backends.Hip.HipRuntime;

namespace Idrak.Backends.Hip;

/// <summary>
/// What a HIP device reports about itself, read once when the devices are probed. The kernels' block size and every
/// launch limit come from these values (<see cref="HipKernels.BlockSizeFor"/>), never from the device's name or
/// architecture: the architecture string only names the compiler's target.
/// </summary>
internal sealed record HipDeviceLimits
{
    /// <summary>The device's name as the runtime reports it.</summary>
    public required string Name { get; init; }

    /// <summary>The compiler target the runtime reports (gcnArchName, with its feature flags), or "" when not found.</summary>
    public required string Architecture { get; init; }

    public required long TotalMemory { get; init; }

    public required int ComputeUnits { get; init; }

    /// <summary>Lanes per wavefront: the hardware's SIMD width as reported (it differs between GPU generations).</summary>
    public required int WarpSize { get; init; }

    public required int MaxThreadsPerBlock { get; init; }

    public required int MaxBlockDimX { get; init; }

    public required int MaxGridDimX { get; init; }

    public required int MaxGridDimY { get; init; }

    /// <summary>Shared memory (LDS) a block may use, in bytes.</summary>
    public required int SharedMemoryPerBlock { get; init; }

    public required int L2CacheBytes { get; init; }

    public required int MemoryBusWidth { get; init; }

    public required int Major { get; init; }

    public required int Minor { get; init; }

    public required bool Integrated { get; init; }

    /// <summary>"domain:bus:device" as reported (for a stable identity in caches), or "" when not reported.</summary>
    public required string PciAddress { get; init; }

    public Guid? Uuid { get; init; }

    /// <summary>
    /// Why the kernels cannot use these limits (values no device reports, such as a runtime that numbers its attributes
    /// differently), or null when they are usable. The kernels then stay off and every operation takes the host fallback.
    /// </summary>
    public string? Problem()
    {
        if (WarpSize <= 0 || WarpSize > 1024 || !BitOperations.IsPow2(WarpSize))
        {
            return $"the runtime reports a wavefront of {WarpSize} lanes";
        }

        if (MaxThreadsPerBlock < WarpSize || MaxThreadsPerBlock > 65536 || MaxBlockDimX < WarpSize)
        {
            return $"the runtime reports {MaxThreadsPerBlock} threads per block ({MaxBlockDimX} along x) with {WarpSize}-lane wavefronts";
        }

        if (SharedMemoryPerBlock < WarpSize * sizeof(float))
        {
            return $"the runtime reports {SharedMemoryPerBlock} bytes of shared memory per block";
        }

        if (MaxGridDimX <= 0 || MaxGridDimY <= 0)
        {
            return $"the runtime reports a grid of {MaxGridDimX} × {MaxGridDimY} blocks";
        }

        return null;
    }

    /// <summary>Reads device <paramref name="ordinal"/>'s limits (the runtime is loaded and initialized).</summary>
    public static unsafe HipDeviceLimits Read(int ordinal)
    {
        int device;
        Check(hipDeviceGet(&device, ordinal), nameof(hipDeviceGet));
        byte* name = stackalloc byte[256];
        name[0] = 0;
        Check(hipDeviceGetName(name, 256, device), nameof(hipDeviceGetName));
        nuint memory;
        Check(hipDeviceTotalMem(&memory, device), nameof(hipDeviceTotalMem));

        int Attribute(int attribute)
        {
            int value = 0;
            return hipDeviceGetAttribute(&value, attribute, ordinal) == Success ? value : 0;
        }

        Guid? uuid = null;
        if (hipDeviceGetUuid != null)
        {
            byte* bytes = stackalloc byte[16];
            if (hipDeviceGetUuid(bytes, device) == Success)
            {
                var id = new Guid(new ReadOnlySpan<byte>(bytes, 16));
                uuid = id == Guid.Empty ? null : id;
            }
        }

        int domain = Attribute(AttributePciDomainId), bus = Attribute(AttributePciBusId), slot = Attribute(AttributePciDeviceId);
        return new HipDeviceLimits
        {
            Name = Marshal.PtrToStringUTF8((IntPtr)name) ?? "",
            Architecture = ReadArchitecture(ordinal),
            TotalMemory = (long)memory,
            ComputeUnits = Attribute(AttributeMultiprocessorCount),
            WarpSize = Attribute(AttributeWarpSize),
            MaxThreadsPerBlock = Attribute(AttributeMaxThreadsPerBlock),
            MaxBlockDimX = Attribute(AttributeMaxBlockDimX),
            MaxGridDimX = Attribute(AttributeMaxGridDimX),
            MaxGridDimY = Attribute(AttributeMaxGridDimY),
            SharedMemoryPerBlock = Attribute(AttributeMaxSharedMemoryPerBlock),
            L2CacheBytes = Attribute(AttributeL2CacheSize),
            MemoryBusWidth = Attribute(AttributeMemoryBusWidth),
            Major = Attribute(AttributeComputeCapabilityMajor),
            Minor = Attribute(AttributeComputeCapabilityMinor),
            Integrated = Attribute(AttributeIntegrated) != 0,
            PciAddress = (domain | bus | slot) == 0 ? "" : $"{domain:x}:{bus:x}:{slot:x}",
            Uuid = uuid,
        };
    }

    // The device-properties struct is read into a buffer far larger than any release's struct, and the architecture
    // string is found by its content (see ArchitectureFrom): its offset moved when the struct grew (6.0, R0600).
    private static unsafe string ReadArchitecture(int ordinal)
    {
        var call = hipGetDevicePropertiesR0600 != null ? hipGetDevicePropertiesR0600 : hipGetDeviceProperties;
        if (call == null)
        {
            return "";
        }

        byte[] buffer = new byte[PropertiesBufferBytes];
        fixed (byte* p = buffer)
        {
            return call(p, ordinal) == Success ? ArchitectureFrom(buffer) : "";
        }
    }

    /// <summary>Bytes given to the device-properties call: several times the largest struct of any release (under 2 KiB).</summary>
    internal const int PropertiesBufferBytes = 16 * 1024;

    /// <summary>
    /// The compiler target in a device-properties struct: the first NUL-terminated string after the device name (the
    /// struct's first 256 bytes) that starts with the LLVM target prefix "gfx" and a digit, with its feature flags
    /// ("gfx90a:sramecc+:xnack-"); "" when there is none (the backend then compiles for the current device).
    /// </summary>
    internal static string ArchitectureFrom(ReadOnlySpan<byte> properties)
    {
        ReadOnlySpan<byte> prefix = "gfx"u8;
        for (int i = 256; i + prefix.Length < properties.Length; i += 4)          // the field follows 4-byte members
        {
            if (!properties[i..].StartsWith(prefix) || !char.IsAsciiDigit((char)properties[i + prefix.Length]) || properties[i - 1] != 0)
            {
                continue;
            }

            int end = i;
            while (end < properties.Length && end - i < 256 && properties[end] != 0)
            {
                end++;
            }

            string text = Encoding.ASCII.GetString(properties[i..end]);
            if (text.All(c => char.IsAsciiLetterOrDigit(c) || c is ':' or '+' or '-' or '_'))
            {
                return text;
            }
        }

        return "";
    }
}
