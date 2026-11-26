// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Storages larger than one binding. A kernel binds at most maxStorageBufferRange bytes of a storage (128 MiB on some
// devices), while a buffer and its memory may be larger (up to maxMemoryAllocationSize), so a large storage keeps its one
// buffer (uploads, downloads and fills copy without a range limit) and the operations that read large weights bind it in
// windows: ranges of whole rows of at most maxStorageBufferRange bytes, each starting at a multiple of
// minStorageBufferOffsetAlignment, one dispatch per window. The window size follows from those two reported limits only.
// Storages that fit one binding take the path they always did (bound whole, one dispatch). The embedding gathers bind
// windows of table rows; each dispatch writes the outputs whose row its window holds.
internal sealed partial class VulkanBackend
{
    /// <summary>Tests: the binding range every backend created afterwards uses when lower than the device's
    /// (maxStorageBufferRange), to reach the windowed paths on any device; IDRAK_VULKAN_MAX_STORAGE_BYTES sets it for a
    /// process.</summary>
    internal static long? StorageRangeOverride { get; set; } = BytesSetting("IDRAK_VULKAN_MAX_STORAGE_BYTES");

    // The binding range: the device's, or the override when lower; whole words.
    private static long StorageRange(uint device, long? setting)
    {
        long range = device;
        if (setting is long s and > 0)
        {
            range = Math.Min(range, s);
        }

        return Math.Max(range / sizeof(float) * sizeof(float), sizeof(float));
    }

    /// <summary>Tests: sets the binding range of this backend (null: the device's, or the override's); returns the one it had.</summary>
    internal long LimitStorageRange(long? bytes)
    {
        long previous = MaxStorageBytes;
        MaxStorageBytes = StorageRange(_physical.Properties.MaxStorageBufferRange, bytes ?? StorageRangeOverride);
        return previous;
    }

    /// <summary>Where a window may start, in bytes (minStorageBufferOffsetAlignment, at least a word).</summary>
    internal long StorageAlignment => Math.Max((long)_physical.Properties.MinStorageBufferOffsetAlignment, sizeof(float));

    // The bytes of a window of a storage of `bytes` starting at `offset`: the rest of the storage, at most the binding
    // range. Throws when the start is outside the storage or not aligned.
    private long WindowBytes(long bytes, long offset)
    {
        if (offset < 0 || offset >= bytes || offset % StorageAlignment != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"A window at byte {offset:N0} of a storage of {bytes:N0} bytes (windows start at multiples of {StorageAlignment}).");
        }

        return Math.Min(bytes - offset, MaxStorageBytes);
    }

    /// <summary>
    /// Rows per window of row-major storages bound together (row r of storage i starts at byte r · rowBytes[i]): the
    /// most rows, a multiple of <paramref name="multiple"/>, whose bytes fit <paramref name="range"/> in every storage
    /// and that keep every window's start in each storage a multiple of <paramref name="alignment"/> (a power of two);
    /// 0 when not even one such step of rows fits.
    /// </summary>
    internal static long WindowRows(long range, long alignment, long multiple, ReadOnlySpan<long> rowBytes)
    {
        long step = Math.Max(multiple, 1), most = long.MaxValue;
        alignment = Math.Max(alignment, 1);
        foreach (long bytes in rowBytes)
        {
            if (bytes <= 0)
            {
                return 0;
            }

            step = Lcm(step, alignment / Gcd(bytes, alignment));                 // rows between aligned starts
            most = Math.Min(most, range / bytes);
        }

        return most == long.MaxValue ? 0 : most / step * step;
    }

    // ------------------------------------------------------------------ packed products

    // ------------------------------------------------------------------ packed products

    private static long Gcd(long a, long b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return a;
    }

    private static long Lcm(long a, long b) => a / Gcd(a, b) * b;

    // ------------------------------------------------------------------ gathers

    // A gather from a table larger than a binding ([vocabulary, rows of `rowWords` words]): one dispatch per window of
    // table rows. False when the indices or the output do not fit a binding or no window of one row fits.
    private bool GatherWindows(string kernel, Storage table, Storage indices, Storage y, int count, int dim, int vocabulary, int rowWords)
    {
        long rowBytes = (long)rowWords * 4;
        long window = WindowRows(MaxStorageBytes, StorageAlignment, 1, [rowBytes]);
        if (!Fit(indices, y) || window <= 0 || window > int.MaxValue)
        {
            return false;
        }

        if (count <= 0 || dim <= 0)
        {
            return true;
        }

        var dispatched = Kernel(kernel);
        uint groups = GridGroups((long)count * dim);
        Span<byte> b = stackalloc byte[20];
        Span<long> at = stackalloc long[3];
        for (long first = 0; first < vocabulary && first * rowBytes < BlockBytes(table.Length); first += window)
        {
            int rows = (int)Math.Min(window, vocabulary - first);
            at[0] = first * rowBytes;
            Dispatch(dispatched, groups, 1, 1, [table, indices, y], new Push(b).I(count).I(dim).I(vocabulary).I((int)first).I(rows).Bytes, at);
        }

        return true;
    }
}
