// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Storages larger than one binding. A kernel binds at most maxStorageBufferRange bytes of a storage (128 MiB on some
// devices), while a buffer and its memory may be larger (up to maxMemoryAllocationSize), so a large storage keeps its one
// buffer (uploads, downloads and fills copy without a range limit) and the operations that read large weights bind it in
// windows: ranges of whole rows of at most maxStorageBufferRange bytes, each starting at a multiple of
// minStorageBufferOffsetAlignment, one dispatch per window. The window size follows from those two reported limits only.
// Storages that fit one binding take the path they always did (bound whole, one dispatch).
//
//   - the packed products (int8, int4, bfloat16): windows of rows of k, a whole number of splits each; each window's
//     dispatch writes its splits' partial sums, which gemv_reduce adds in split order (VulkanKernels.PackedGemv);
//   - the embedding gathers (float32, bfloat16): windows of table rows; each dispatch writes the outputs whose row its
//     window holds;
//   - the dequantizations: windows of rows of k of the packed weights and of the float32 output alike.
//
// Other operations on a storage larger than a binding take the host fallback.
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

    // Rows of k a split of a packed product's chunk is a multiple of when its weights are windowed: int4's scale groups
    // (32 rows), else any row.
    private static int WindowUnit(VulkanKernels.PackedFormat format) => format == VulkanKernels.PackedFormat.Int4 ? VulkanKernels.GemvChunkAlign : 1;

    // Rows of x per group of a windowed packed product whose x and y windows start at aligned offsets (rows of k and of
    // n floats): the step between such rows.
    private long RowStep(int n, int k) =>
        Lcm(StorageAlignment / Gcd((long)k * 4, StorageAlignment), StorageAlignment / Gcd((long)n * 4, StorageAlignment));

    // Rows of k per window of a packed product's weights (and int4's scales): 0 when they fit one binding, -1 when they
    // do not and cannot be windowed (no window of whole split units fits, or the partial sums of one split per window
    // would not fit a binding even for one aligned group of rows of x).
    private long PackedWindowRows(VulkanKernels.PackedFormat format, Storage weights, Storage? scales, int m, int n, int k)
    {
        bool int4 = format == VulkanKernels.PackedFormat.Int4;
        if (BlockBytes(weights.Length) <= MaxStorageBytes && !(int4 && scales is not null && BlockBytes(scales.Length) > MaxStorageBytes))
        {
            return 0;
        }

        long words = ((long)n + VulkanKernels.ColumnsPerWord(format) - 1) / VulkanKernels.ColumnsPerWord(format);
        // int4's scale rows (8 · words values per 32 rows of k) start at byte r · words for r a multiple of 32.
        long rows = int4 ? WindowRows(MaxStorageBytes, StorageAlignment, WindowUnit(format), [words * 4, words])
            : WindowRows(MaxStorageBytes, StorageAlignment, WindowUnit(format), [words * 4]);
        if (rows <= 0 || rows > int.MaxValue)
        {
            return -1;
        }

        long windows = (k + rows - 1) / rows;
        return windows == 1 || PartialRows(windows, m, n, k) > 0 ? rows : -1;
    }

    // Rows of x per group whose partial sums of `splits` splits fit a binding: all m, else a multiple of the aligned step
    // (0 when not even one step fits).
    private long PartialRows(long splits, int m, int n, int k)
    {
        long most = MaxStorageBytes / (splits * n * 4);
        most = Math.Min(most, int.MaxValue / Math.Max(1, splits * n));
        if (most >= m)
        {
            return m;
        }

        long step = RowStep(n, k);
        return most / step * step;
    }

    // A packed product over weights in windows of `window` rows of k: the chunk (rows per split, a multiple of the
    // format's unit) is made to divide the window, at most `chunk` and with at most maxComputeWorkGroupCount[1] splits
    // per window (one split per window when the partial sums of more would not fit a binding); then for each group of
    // rows of x whose partial sums fit a binding (all of them unless m is large), one dispatch per window writes its
    // splits' partial sums, and gemv_reduce adds them in split order.
    private void RunPackedWindows(VulkanKernels.PackedFormat format, string kernel, int width, uint columnBlocks, int blockRows, int chunk, int window,
        Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        int unit = WindowUnit(format), units = window / unit, want = Math.Max(1, chunk / unit), parts = 0;
        for (int d = Math.Min(want, units); d >= 1 && parts == 0; d--)
        {
            parts = units % d == 0 && units / d <= Limits.MaxGroupsY ? d : 0;
        }

        for (int d = want + 1; d <= units && parts == 0; d++)
        {
            parts = units % d == 0 && units / d <= Limits.MaxGroupsY ? d : 0;
        }

        chunk = Math.Max(parts, 1) * unit;
        int splits = (k + chunk - 1) / chunk;
        long group = splits == 1 ? m : PartialRows(splits, m, n, k);
        if (group <= 0 || group < m && PartialRows((k + window - 1) / window, m, n, k) > group)
        {
            chunk = window;                                                    // one split per window: fewer partial sums
            splits = (k + chunk - 1) / chunk;
            group = splits == 1 ? m : PartialRows(splits, m, n, k);            // PackedWindowRows checked one step fits
        }

        int perWindow = window / chunk, rowsPer = (int)group;
        long rowBytes = ((long)n + VulkanKernels.ColumnsPerWord(format) - 1) / VulkanKernels.ColumnsPerWord(format) * 4;
        var dispatched = width == Width ? Kernel(kernel) : KernelAt(kernel, width, Limits.SubgroupArithmetic);
        var output = splits == 1 ? null : Allocate(splits * rowsPer * n, zeroed: false);
        try
        {
            Span<byte> b = stackalloc byte[24];
            Span<long> at = stackalloc long[4];
            Span<byte> reduceBytes = stackalloc byte[12];
            for (int r0 = 0; r0 < m; r0 += rowsPer)
            {
                int mm = Math.Min(rowsPer, m - r0);
                uint rowBlocks = (uint)((mm + blockRows - 1) / blockRows);
                long xAt = (long)r0 * k * 4, yAt = (long)r0 * n * 4;
                var target = output ?? y;
                long targetAt = output is null ? yAt : 0;
                for (int first = 0; first < splits; first += perWindow)
                {
                    long row = (long)first * chunk;
                    var push = new Push(b).I(mm).I(n).I(k).I(chunk).I(first).I(splits).Bytes;
                    uint count = (uint)Math.Min(perWindow, splits - first);
                    at[0] = xAt;
                    at[1] = row * rowBytes;
                    if (scales is null)
                    {
                        at[2] = targetAt;
                        Dispatch(dispatched, columnBlocks, count, rowBlocks, [x, weights, target], push, at[..3]);
                    }
                    else
                    {
                        at[2] = format == VulkanKernels.PackedFormat.Int4 ? row * (rowBytes / 4) : 0;   // int4: 8 · words scales per 32 rows
                        at[3] = targetAt;
                        Dispatch(dispatched, columnBlocks, count, rowBlocks, [x, weights, scales, target], push, at);
                    }
                }

                if (output is not null)
                {
                    var reducePush = new Push(reduceBytes).I(mm * n).I(n).I(splits).Bytes;
                    uint groups = GridGroups((long)mm * n);
                    if (format == VulkanKernels.PackedFormat.Int8)
                    {
                        at[0] = 0;
                        at[1] = 0;
                        at[2] = yAt;
                        Dispatch(Kernel("int8_gemv_reduce"), groups, 1, 1, [output, scales!, y], reducePush, at[..3]);
                    }
                    else
                    {
                        at[0] = 0;
                        at[1] = yAt;
                        Dispatch(Kernel("gemv_reduce"), groups, 1, 1, [output, y], reducePush, at[..2]);
                    }
                }
            }
        }
        finally
        {
            output?.Release();                                                 // reused in queue order
        }
    }

    private static long Gcd(long a, long b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return a;
    }

    private static long Lcm(long a, long b) => a / Gcd(a, b) * b;

    // ------------------------------------------------------------------ gathers and dequantizations

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

    // A dequantization of packed weights [k, n] (or of an output) larger than a binding: one dispatch per window of rows
    // of k of the packed weights, the output (and int4's scales) alike. False when int8's scales do not fit a binding or
    // no window fits.
    private bool DequantizeWindows(VulkanKernels.PackedFormat format, Storage packed, Storage? scales, Storage w, int k, int n)
    {
        bool int4 = format == VulkanKernels.PackedFormat.Int4;
        long words = ((long)n + VulkanKernels.ColumnsPerWord(format) - 1) / VulkanKernels.ColumnsPerWord(format);
        long window = int4 ? WindowRows(MaxStorageBytes, StorageAlignment, VulkanKernels.GemvChunkAlign, [words * 4, (long)n * 4, words])
            : WindowRows(MaxStorageBytes, StorageAlignment, 1, [words * 4, (long)n * 4]);
        if (KernelsOff || window <= 0 || window > int.MaxValue || (format == VulkanKernels.PackedFormat.Int8 && !Fit(scales!)))
        {
            return false;
        }

        string name = format switch
        {
            VulkanKernels.PackedFormat.Int8 => "int8_dequantize",
            VulkanKernels.PackedFormat.Int4 => "int4_dequantize",
            _ => "bf16_dequantize",
        };
        var dispatched = Kernel(name);
        Span<byte> b = stackalloc byte[8];
        Span<long> at = stackalloc long[3];
        for (long row = 0; row < k; row += window)
        {
            int rows = (int)Math.Min(window, k - row);
            var push = new Push(b).I(rows).I(n).Bytes;
            uint groups = GridGroups((long)rows * n);
            if (scales is null)
            {
                at[0] = row * words * 4;
                at[1] = row * n * 4;
                Dispatch(dispatched, groups, 1, 1, [packed, w], push, at[..2]);
            }
            else
            {
                at[0] = row * words * 4;
                at[1] = int4 ? row * words : 0;
                at[2] = row * n * 4;
                Dispatch(dispatched, groups, 1, 1, [packed, scales, w], push, at);
            }
        }

        return true;
    }
}
