// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;

namespace Idrak.Cluster;

/// <summary>
/// The built-in link profiles (plan 23), none of which carries a figure of some hardware: a name that states its rate
/// carries exactly that rate ("ethernet-1g": 1 Gb/s; any registered name takes a rate suffix, "direct-link-10g"); a link
/// this machine can reach is measured on it at first use ("same-device": a copy inside one device's memory;
/// "shared-memory": a copy in host memory; "pcie": a device's copies to and from the host); every other link
/// ("infiniband", "roce", "nvlink", "thunderbolt", "usb4", "wifi", "ethernet", "direct-link") has no rate until the
/// topology gives one in its name or a bench measures it, and the simulator refuses to guess. Latency is zero unless
/// measured; a calibration against a real run corrects both.
/// </summary>
public static class BuiltInLinks
{
    /// <summary>The bytes a measurement copies, or null for 32 MiB (at most an eighth of what the device has free); <c>IDRAK_CLUSTER_PROBE_BYTES</c>.</summary>
    public static long? ProbeBytes { get; set; } = Environment.GetEnvironmentVariable("IDRAK_CLUSTER_PROBE_BYTES") is { Length: > 0 } text
        ? ClusterTopology.ParseMemory(text)
        : null;

    internal static IEnumerable<ILinkProfile> All()
    {
        foreach (string rate in (string[])["1g", "2.5g", "10g", "100g"])
        {
            LinkProfiles.TryParseRate(rate, out double bytesPerSecond);
            yield return new LinkProfile($"ethernet-{rate}", bytesPerSecond, TimeSpan.Zero) { Summary = $"Ethernet at {rate[..^1]} Gb/s" };
        }

        foreach (var (name, summary) in new[]
                 {
                     ("ethernet", "Ethernet: give the rate (ethernet-25g)"),
                     ("direct-link", "a cable between two machines, no switch (crossover Ethernet, Thunderbolt or USB4 networking): give the rate (direct-link-10g)"),
                     ("thunderbolt", "Thunderbolt networking: give the rate (thunderbolt-40g)"),
                     ("usb4", "USB4 networking: give the rate (usb4-40g)"),
                     ("infiniband", "InfiniBand: give the rate (infiniband-200g)"),
                     ("roce", "RDMA over Converged Ethernet: give the rate (roce-100g)"),
                     ("nvlink", "NVLink between two GPUs of one machine: give the rate (nvlink-900g)"),
                     ("wifi", "Wi-Fi: give the rate (wifi-300m)"),
                 })
        {
            yield return new LinkProfile(name, 0, TimeSpan.Zero) { Summary = summary };
        }

        yield return new MeasuredLink(ClusterTopology.SameDevice, "two ranks on one device: a copy inside its memory, measured on the device", MeasureSameDevice);
        yield return new MeasuredLink("shared-memory", "two processes on one machine: a copy in host memory, measured here", _ => MeasureHost());
        yield return new MeasuredLink("pcie", "two devices of one machine through the host: a device's copies to and from host memory, measured on the device", MeasureHostStaging);
    }

    private static long Probe(Device device)
    {
        long bytes = ProbeBytes ?? 32L << 20;
        if (device.Backend.AvailableMemory() is { } free && free > 0)
        {
            bytes = Math.Min(bytes, free / 8);
        }

        return Math.Max(4096, bytes / sizeof(float) * sizeof(float));
    }

    // The median of three timed calls after one warm call (rule 76: the first pays for compiling).
    private static double Median(Action call)
    {
        call();
        Span<double> times = stackalloc double[3];
        for (int i = 0; i < times.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            call();
            times[i] = Stopwatch.GetElapsedTime(start).TotalSeconds;
        }

        times.Sort();
        return Math.Max(times[1], 1e-9);
    }

    private static (double Bandwidth, TimeSpan Latency) MeasureSameDevice(Device device)
    {
        var backend = device.Backend;
        int count = (int)(Probe(device) / sizeof(float));
        var x = backend.Allocate(count, zeroed: true);
        var y = backend.Allocate(count, zeroed: false);
        try
        {
            double seconds = Median(() =>
            {
                backend.Copy(x, y, count);
                backend.Synchronize();
            });
            double latency = Median(() =>
            {
                backend.Copy(x, y, 1);
                backend.Synchronize();
            });
            return (count * (double)sizeof(float) / seconds, TimeSpan.FromSeconds(latency));
        }
        finally
        {
            x.Release();
            y.Release();
        }
    }

    private static (double Bandwidth, TimeSpan Latency) MeasureHost()
    {
        int bytes = (int)Math.Min(int.MaxValue, Probe(Device.Cpu));
        var source = GC.AllocateUninitializedArray<byte>(bytes);
        var target = GC.AllocateUninitializedArray<byte>(bytes);
        double seconds = Median(() => source.AsSpan().CopyTo(target));
        double latency = Median(() => source.AsSpan(0, sizeof(float)).CopyTo(target));
        return (bytes / seconds, TimeSpan.FromSeconds(latency));
    }

    private static (double Bandwidth, TimeSpan Latency) MeasureHostStaging(Device device)
    {
        if (device.Type == DeviceType.Cpu)
        {
            return MeasureHost();
        }

        var backend = device.Backend;
        int count = (int)(Probe(device) / sizeof(float));
        var host = new float[count];
        var storage = backend.Allocate(count, zeroed: false);
        try
        {
            double down = Median(() =>
            {
                backend.Download(storage, host);
                backend.Synchronize();
            });
            double up = Median(() =>
            {
                backend.Upload(host, storage);
                backend.Synchronize();
            });
            double latency = Median(() =>
            {
                backend.Download(storage, host.AsSpan(0, 1));
                backend.Upload(host.AsSpan(0, 1), storage);
                backend.Synchronize();
            });
            return (count * (double)sizeof(float) / (down + up), TimeSpan.FromSeconds(latency));
        }
        finally
        {
            storage.Release();
        }
    }
}

/// <summary>
/// A link measured on this machine at first use, once per physical device (the simulator asks for the device a rank is
/// carved from). Unmeasured it has no bandwidth, so predicting from it refuses rather than guesses.
/// </summary>
internal sealed class MeasuredLink(string name, string summary, Func<Device, (double Bandwidth, TimeSpan Latency)> measure) : ILinkProfile
{
    private readonly ConcurrentDictionary<string, Lazy<ILinkProfile>> _measured = new(StringComparer.OrdinalIgnoreCase);

    public string Name { get; } = name;

    public string Summary { get; } = summary;

    public double BandwidthBytesPerSecond => 0;

    public TimeSpan Latency => TimeSpan.Zero;

    public TimeSpan Jitter => TimeSpan.Zero;

    public double LossRate => 0;

    public double FailureRate => 0;

    public TimeSpan Predict(long bytes, Random seeded) =>
        throw new InvalidOperationException($"Link '{Name}' is measured on a device: the simulator measures it for the device a rank is on.");

    public ILinkProfile With(double? bandwidthBytesPerSecond = null, TimeSpan? latency = null, TimeSpan? jitter = null, double? lossRate = null,
        double? failureRate = null, string? name = null) =>
        new LinkProfile(name ?? Name, bandwidthBytesPerSecond ?? 0, latency ?? TimeSpan.Zero) { Summary = Summary, Jitter = jitter ?? TimeSpan.Zero, LossRate = lossRate ?? 0, FailureRate = failureRate ?? 0 };

    // The profile as measured on `device` (measured once per device, on first use).
    public ILinkProfile On(Device device) => _measured.GetOrAdd(device.ToString(), _ => new Lazy<ILinkProfile>(() =>
    {
        var (bandwidth, latency) = measure(device);
        return new LinkProfile(Name, bandwidth, latency) { Summary = $"{Summary} ({device})" };
    })).Value;

    // `link` for a rank on `device` (a name: "cuda:0"): measured there when it is a measured link.
    public static ILinkProfile Resolve(ILinkProfile link, string device) => link is MeasuredLink measured ? measured.On(Device.Parse(device)) : link;
}
