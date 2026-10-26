// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

// What a device's memory costs beyond what the library holds: on each device IDRAK_DEVICES names (default every GPU in
// use) that reports its driver's free count (Backend.FreeMemory), the memory held outside the library right after the
// device is opened and after its first kernels ran, then for blocks of several sizes what the driver takes per block
// against the bytes asked for (IDRAK_BENCH_ALLOC_MB: the most each size allocates in all, default 256).
internal abstract partial class Tests
{
    internal static int BenchAlloc()
    {
        long budget = (Environment.GetEnvironmentVariable("IDRAK_BENCH_ALLOC_MB") is { Length: > 0 } mb ? long.Parse(mb, CultureInfo.InvariantCulture) : 256) << 20;
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [.. Device.InUse().Where(d => d.Type != DeviceType.Cpu)];
        foreach (var device in devices)
        {
            var backend = device.Backend;
            if (backend.FreeMemory() is null || backend.Hardware.MemoryBytes is not { } total)
            {
                Console.WriteLine($"{device}: reports no free memory; nothing to measure");
                continue;
            }

            long Outside()
            {
                var usage = backend.GetMemoryUsage();
                return total - backend.FreeMemory()!.Value - usage.InUse - usage.Cached;
            }

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{device}: {total / 1048576.0:F0} MiB"));
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  held outside the library once opened:          {Outside() / 1048576.0,8:F1} MiB"));
            using (var warm = Tensor.From([1f, 2f, 3f], [3], device))
            using (var doubled = warm * 2f)
            {
                _ = doubled.ToArray();
            }

            ComputeResources.ReleaseCachedMemory(device);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  held outside the library after its first kernels: {Outside() / 1048576.0,5:F1} MiB"));
            foreach (bool longLived in (bool[])[false, true])
            {
                Console.WriteLine(longLived ? "  the same blocks as long-lived ones (Backend.LongLivedAllocations: shared chunks)" : "  each block its own allocation");
                Console.WriteLine("  block size     blocks   asked MiB   driver MiB   driver per block   overhead");
                foreach (long size in (long[])[4 << 10, 64 << 10, 1 << 20, (1 << 20) + (4 << 10), 2 << 20, (2 << 20) + (4 << 10), (10 << 20) + (4 << 10), (26 << 20) + (4 << 10)])
                {
                    int count = (int)Math.Clamp(budget / size, 4, 2048);
                    ComputeResources.ReleaseCachedMemory(device);
                    long before = backend.FreeMemory()!.Value;
                    var blocks = new List<Tensor>(count);
                    try
                    {
                        using (longLived ? backend.LongLivedAllocations() : null)
                        {
                            for (int i = 0; i < count; i++)
                            {
                                blocks.Add(Tensor.Zeros([(int)(size / 4)], device));
                            }
                        }

                        _ = blocks[^1].ToArray().Length;                                // the zeroing done
                        long taken = before - backend.FreeMemory()!.Value, asked = size * count;
                        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                            $"  {Size(size),-12} {count,8} {asked / 1048576.0,11:F1} {taken / 1048576.0,12:F1} {Size(taken / count),18} {100.0 * (taken - asked) / asked,9:F1}%"));
                    }
                    finally
                    {
                        blocks.ForEach(b => b.Dispose());
                        ComputeResources.ReleaseCachedMemory(device);
                    }
                }
            }
        }

        return 0;

        static string Size(long bytes) => bytes >= 1 << 20
            ? string.Create(CultureInfo.InvariantCulture, $"{bytes / 1048576.0:F2} MiB")
            : string.Create(CultureInfo.InvariantCulture, $"{bytes / 1024.0:F0} KiB");
    }
}
