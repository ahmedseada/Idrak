using Idrak.Backends.Cuda;

namespace Idrak.Diagnostics;

/// <summary>One kernel (or matrix-product shape) in a <see cref="GpuProfiler"/> report.</summary>
/// <param name="Name">The kernel; matrix products as "kernel_{a}{b} m x n x k".</param>
/// <param name="Calls">Launches.</param>
/// <param name="Milliseconds">Total GPU time.</param>
/// <param name="Tflops">Achieved TFLOPS for matrix products (null for other kernels).</param>
public sealed record GpuProfileEntry(string Name, long Calls, double Milliseconds, double? Tflops);

/// <summary>
/// Times every kernel a CUDA device runs between <see cref="Start"/> and <see cref="Stop"/>, by waiting for each one:
/// shows where a training step's GPU time goes. Much slower than normal running, so profile a step or two, not a run.
/// </summary>
public static class GpuProfiler
{
    /// <summary>Starts timing the kernels of <paramref name="device"/> (a CUDA device; ignored for the CPU).</summary>
    public static void Start(Device device)
    {
        if (device.Type == DeviceType.Cuda)
        {
            device.Synchronize();
            ((CudaBackend)device.Backend).StartProfile();
        }
    }

    /// <summary>Stops timing and returns the kernels, most time first.</summary>
    public static IReadOnlyList<GpuProfileEntry> Stop(Device device)
    {
        if (device.Type != DeviceType.Cuda)
        {
            return [];
        }

        device.Synchronize();
        return [.. ((CudaBackend)device.Backend).StopProfile()
            .Select(e =>
            {
                double ms = e.Value.Ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                return new GpuProfileEntry(e.Key, e.Value.Calls, ms, e.Value.Flops > 0 ? e.Value.Flops / (ms * 1e9) : null);
            })
            .OrderByDescending(e => e.Milliseconds)];
    }

    /// <summary>A table of <paramref name="entries"/> (the first <paramref name="rows"/>), with totals.</summary>
    public static string Format(IReadOnlyList<GpuProfileEntry> entries, int rows = 40)
    {
        double total = entries.Sum(e => e.Milliseconds);
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"{"kernel",-52} {"calls",7} {"ms",10} {"share",7} {"TFLOPS",7}");
        foreach (var e in entries.Take(rows))
        {
            sb.AppendLine($"{e.Name,-52} {e.Calls,7} {e.Milliseconds,10:F2} {e.Milliseconds / total,7:P1} {(e.Tflops is { } t ? t.ToString("F1") : ""),7}");
        }

        var products = entries.Where(e => e.Tflops is not null).ToList();
        double productMs = products.Sum(e => e.Milliseconds), flops = products.Sum(e => e.Tflops!.Value * e.Milliseconds * 1e9);
        sb.AppendLine($"total {total:F1} ms in {entries.Sum(e => e.Calls)} launches; matrix products {productMs:F1} ms "
                      + $"({(productMs > 0 ? flops / (productMs * 1e9) : 0):F1} TFLOPS), everything else {total - productMs:F1} ms");
        return sb.ToString();
    }
}
