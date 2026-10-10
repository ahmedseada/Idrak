// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

/// <summary>
/// What a CUDA device reports about itself (cuDeviceGetAttribute). Kernel shapes, the checks of what a kernel needs, and
/// every heuristic read these, never a card's name: a choice that cannot be derived from them is measured on the device
/// (CudaBackend.Tuning.cs). Values a driver does not report are 0.
/// </summary>
internal sealed record CudaDeviceLimits
{
    public required int ComputeMajor { get; init; }
    public required int ComputeMinor { get; init; }
    public required int Multiprocessors { get; init; }
    public required int WarpSize { get; init; }
    public required int MaxThreadsPerBlock { get; init; }
    public required int MaxThreadsPerMultiprocessor { get; init; }
    public int MaxBlocksPerMultiprocessor { get; init; }

    /// <summary>Shared memory a block may declare statically (bytes).</summary>
    public required int SharedPerBlock { get; init; }

    /// <summary>Shared memory a block may use when its kernel opts in (static + dynamic, bytes).</summary>
    public required int SharedPerBlockOptin { get; init; }

    public int SharedPerMultiprocessor { get; init; }
    public required int RegistersPerBlock { get; init; }
    public int RegistersPerMultiprocessor { get; init; }
    public int L2Bytes { get; init; }
    public int MemoryBusWidth { get; init; }
    public int ClockKhz { get; init; }
    public int MemoryClockKhz { get; init; }
    /// <summary>Whether the device runs cooperative launches (every block resident at once: grid barriers).</summary>
    public bool CooperativeLaunch { get; init; }

    public required int MaxGridY { get; init; }
    public required int MaxGridZ { get; init; }
    public long MemoryBytes { get; init; }

    /// <summary>Compute capability as one number (8.6 → 86).</summary>
    public int ComputeCapability => ComputeMajor * 10 + ComputeMinor;

    // CUdevice_attribute values (cuda.h).
    private const int MaxThreadsPerBlockAttribute = 1, MaxGridDimYAttribute = 6, MaxGridDimZAttribute = 7, MaxSharedPerBlockAttribute = 8,
        WarpSizeAttribute = 10, MaxRegistersPerBlockAttribute = 12, ClockRateAttribute = 13, MemoryClockRateAttribute = 36,
        BusWidthAttribute = 37, L2SizeAttribute = 38, MaxThreadsPerMultiprocessorAttribute = 39, MaxSharedPerMultiprocessorAttribute = 81,
        MaxRegistersPerMultiprocessorAttribute = 82, MaxSharedPerBlockOptinAttribute = 97, MaxBlocksPerMultiprocessorAttribute = 106, CooperativeLaunchAttribute = 95;

    /// <summary>Reads the limits of device <paramref name="device"/> (an attribute the driver does not know reads as 0).</summary>
    public static CudaDeviceLimits Read(int device, long memoryBytes)
    {
        int Get(int attribute) => cuDeviceGetAttribute(out int value, attribute, device) == 0 ? value : 0;
        int shared = Get(MaxSharedPerBlockAttribute);
        return new CudaDeviceLimits
        {
            ComputeMajor = Get(AttributeComputeCapabilityMajor),
            ComputeMinor = Get(AttributeComputeCapabilityMinor),
            Multiprocessors = Get(AttributeMultiprocessorCount),
            WarpSize = Get(WarpSizeAttribute),
            MaxThreadsPerBlock = Get(MaxThreadsPerBlockAttribute),
            MaxThreadsPerMultiprocessor = Get(MaxThreadsPerMultiprocessorAttribute),
            MaxBlocksPerMultiprocessor = Get(MaxBlocksPerMultiprocessorAttribute),
            SharedPerBlock = shared,
            SharedPerBlockOptin = Math.Max(shared, Get(MaxSharedPerBlockOptinAttribute)),
            SharedPerMultiprocessor = Get(MaxSharedPerMultiprocessorAttribute),
            RegistersPerBlock = Get(MaxRegistersPerBlockAttribute),
            RegistersPerMultiprocessor = Get(MaxRegistersPerMultiprocessorAttribute),
            L2Bytes = Get(L2SizeAttribute),
            MemoryBusWidth = Get(BusWidthAttribute),
            ClockKhz = Get(ClockRateAttribute),
            MemoryClockKhz = Get(MemoryClockRateAttribute),
            CooperativeLaunch = Get(CooperativeLaunchAttribute) != 0,
            MaxGridY = Get(MaxGridDimYAttribute),
            MaxGridZ = Get(MaxGridDimZAttribute),
            MemoryBytes = memoryBytes,
        };
    }

    /// <summary>
    /// Limits as NVIDIA documents them per compute capability (CUDA C++ Programming Guide, "Technical Specifications per
    /// Compute Capability"), for tests and for describing what the shapes give on a card without one at hand. A real
    /// device's limits come from <see cref="Read"/>.
    /// </summary>
    public static CudaDeviceLimits Documented(int major, int minor, int multiprocessors, long memoryBytes = 8L << 30, int l2Bytes = 0)
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

/// <summary>
/// The kernel shapes generated into the PTX for one device, derived from what it reports (<see cref="CudaDeviceLimits"/>).
/// Two kinds (plans/README.md, "Kernel shapes"):
/// <list type="bullet">
/// <item>Shapes the kernels are written to take any value of: block sizes of the one-dimensional and block-reduction
/// kernels and of the sampler, and the per-warp scratch arrays, all relative to the reported threads per block.</item>
/// <item>Geometry a kernel is written for (tiles, warps per tile, stages, rows in registers): fixed by the kernel's code,
/// so the device's limits judge whether it can run (<see cref="Derive"/>, <see cref="PtxKernels.Fits"/>); a kernel
/// family that does not fit is not loaded and its operations take another path.</item>
/// </list>
/// Equal records generate equal PTX, so devices with the same limits share one module text.
/// </summary>
internal sealed record KernelShapes
{
    /// <summary>
    /// Threads per block of the one-dimensional kernels and the block reductions (sum, group statistics): a quarter of the
    /// largest block, so at least four blocks stay resident per SM on every generation (an SM holds at least as many
    /// threads as one block). 1024 / 4 = 256 on every CUDA GPU so far (5070 Ti, 5050, 3060 alike). A power of two (the
    /// reductions halve it).
    /// </summary>
    public required int BlockSize { get; init; }

    /// <summary>Threads of the sampler's block per row: the largest block (1024 on every CUDA GPU so far).</summary>
    public required int SamplerThreads { get; init; }

    /// <summary>Warps in the largest block: the per-warp scratch arrays of the row kernels (32 on every CUDA GPU so far).</summary>
    public required int MaxWarpsPerBlock { get; init; }

    /// <summary>
    /// The shapes for a device, or null with the reason when the kernels cannot run on it: the warp is 32 lanes in every
    /// kernel (PTX's shuffles and lane masks are written for it), the largest fixed block is 1024 threads (gemv_nn_f32:
    /// 32 columns × 32 slices of k), and every main-module kernel's static shared memory, generated for these shapes,
    /// must fit what a block may declare.
    /// </summary>
    public static KernelShapes? Derive(CudaDeviceLimits limits, out string? unsupported)
    {
        unsupported = null;
        if (limits.WarpSize != 32)
        {
            unsupported = $"the kernels are written for 32-lane warps; the GPU reports {limits.WarpSize}";
            return null;
        }

        if (limits.MaxThreadsPerBlock < PtxKernels.LargestFixedBlock)
        {
            unsupported = $"the kernels need blocks of {PtxKernels.LargestFixedBlock} threads; the GPU allows {limits.MaxThreadsPerBlock}";
            return null;
        }

        int threads = FloorPowerOfTwo(limits.MaxThreadsPerBlock);
        var shapes = new KernelShapes
        {
            BlockSize = Math.Max(limits.WarpSize, threads / 4),
            SamplerThreads = threads,
            MaxWarpsPerBlock = threads / limits.WarpSize,
        };
        shapes = shapes == Default ? Default : shapes;
        if (PtxKernels.Fits(PtxKernels.SourceFor(shapes), limits) is { } reason)
        {
            unsupported = reason;
            return null;
        }

        return shapes;
    }

    /// <summary>
    /// The shapes on every CUDA GPU so far (1024 threads per block, 32-lane warps): what the RTX 5070 Ti, RTX 5050 and
    /// RTX 3060 (and compute 5.0 to 12.x as documented) all derive. <see cref="PtxKernels.Source"/> is built for these.
    /// </summary>
    public static KernelShapes Default { get; } = new() { BlockSize = 256, SamplerThreads = 1024, MaxWarpsPerBlock = 32 };

    private static int FloorPowerOfTwo(int value) => value <= 1 ? 1 : 1 << (31 - int.LeadingZeroCount(value));
}

internal static partial class PtxKernels
{
    /// <summary>The largest block any main-module kernel is written for (gemv_nn_f32's 32 × 32 threads).</summary>
    public const int LargestFixedBlock = GemvThreads;

    /// <summary>
    /// Static shared memory (bytes) of each kernel in <paramref name="ptx"/>, summed over its <c>.shared</c> arrays
    /// (".shared", an optional ".align n", ".type", the name and "[count]"; a dynamic array, "[]", counts nothing). The
    /// module text is read this way on every load: span searches, no regular expressions.
    /// </summary>
    public static IReadOnlyDictionary<string, int> StaticSharedBytes(string ptx)
    {
        var result = new Dictionary<string, int>();
        var entries = Entries(ptx);
        for (int i = 0; i < entries.Count; i++)
        {
            int start = entries[i].Start, end = i + 1 < entries.Count ? entries[i + 1].Start : ptx.Length;
            ReadOnlySpan<char> kernel = ptx.AsSpan(start, end - start);
            int bytes = 0, position = 0;
            while (kernel[position..].IndexOf(".shared", StringComparison.Ordinal) is >= 0 and var found)
            {
                int at = position + found + ".shared".Length;
                position = position + found + 1;
                if (SharedArray(kernel, at) is var (type, count, next))
                {
                    int element = kernel[type] switch { "b8" or "u8" or "s8" => 1, "b16" or "u16" or "s16" or "f16" => 2, "b64" or "u64" or "s64" or "f64" => 8, _ => 4 };
                    bytes += element * int.Parse(kernel[count], System.Globalization.CultureInfo.InvariantCulture);
                    position = next;
                }
            }

            result[entries[i].Name] = bytes;
        }

        return result;
    }

    // After ".shared" at `at`: white space, optionally ".align", white space, digits and white space, then the element
    // type (".type"), white space, the name and "[count]". The type's and the count's ranges and the position after "]",
    // or null when the text there is not such an array.
    private static (Range Type, Range Count, int Next)? SharedArray(ReadOnlySpan<char> text, int at)
    {
        int position = SkipWhiteSpace(text, at);
        if (position == at)
        {
            return null;
        }

        if (text[position..].StartsWith(".align", StringComparison.Ordinal))
        {
            int digits = SkipWhiteSpace(text, position + ".align".Length);
            int digitsEnd = SkipDigits(text, digits);
            int next = SkipWhiteSpace(text, digitsEnd);
            if (digits > position + ".align".Length && digitsEnd > digits && next > digitsEnd && SharedArrayAfterAlign(text, next) is { } aligned)
            {
                return aligned;
            }
        }

        return SharedArrayAfterAlign(text, position);
    }

    // ".type", white space, the name, "[count]" at `position`.
    private static (Range Type, Range Count, int Next)? SharedArrayAfterAlign(ReadOnlySpan<char> text, int position)
    {
        if (position >= text.Length || text[position] != '.')
        {
            return null;
        }

        int typeEnd = SkipWord(text, position + 1);
        int name = SkipWhiteSpace(text, typeEnd);
        int nameEnd = SkipWord(text, name);
        if (typeEnd == position + 1 || name == typeEnd || nameEnd == name || nameEnd >= text.Length || text[nameEnd] != '[')
        {
            return null;
        }

        int countEnd = SkipDigits(text, nameEnd + 1);
        if (countEnd == nameEnd + 1 || countEnd >= text.Length || text[countEnd] != ']')
        {
            return null;
        }

        return (new Range(position + 1, typeEnd), new Range(nameEnd + 1, countEnd), countEnd + 1);
    }

    // The first position at or after `position` that is not a decimal digit (a regular expression's \d).
    private static int SkipDigits(ReadOnlySpan<char> text, int position)
    {
        while (position < text.Length && char.IsDigit(text[position]))
        {
            position++;
        }

        return position;
    }

    /// <summary>Dynamic shared memory a tensor-core kernel asks for at launch (bytes; 0 for none).</summary>
    public static int DynamicSharedBytes(string kernel) =>
        kernel.StartsWith("flash_tc_bwd", StringComparison.Ordinal)
            ? kernel.Contains("_kv_", StringComparison.Ordinal)
                ? FlashTensorBackwardKvShared(kernel.EndsWith("d64", StringComparison.Ordinal) ? 64 : 128)
                : FlashTensorBackwardQShared(kernel.EndsWith("d64", StringComparison.Ordinal) ? 64 : 128)
            : kernel.StartsWith("gemm8_", StringComparison.Ordinal) ? EightBitShared : 0;

    /// <summary>
    /// Why the kernels of a module cannot run on a device (null when they can): each kernel's static shared memory must fit
    /// what a block may declare, and static plus dynamic what a block may opt in to.
    /// </summary>
    public static string? Fits(string moduleSource, CudaDeviceLimits limits)
    {
        foreach (var (kernel, staticBytes) in StaticSharedBytes(moduleSource))
        {
            int total = staticBytes + DynamicSharedBytes(kernel);
            if (staticBytes > limits.SharedPerBlock || total > limits.SharedPerBlockOptin)
            {
                return $"{kernel} needs {total} bytes of shared memory per block ({staticBytes} static); the GPU allows {limits.SharedPerBlock} static, {limits.SharedPerBlockOptin} in all";
            }
        }

        return null;
    }
}
