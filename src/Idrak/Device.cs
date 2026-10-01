// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Backends;
using Idrak.Backends.Cpu;
using Idrak.Backends.Cuda;

namespace Idrak;

/// <summary>The kind of hardware a <see cref="Device"/> runs on.</summary>
public enum DeviceType
{
    /// <summary>The host CPU, using SIMD and multi-threading.</summary>
    Cpu,

    /// <summary>An NVIDIA GPU, driven through the CUDA driver API.</summary>
    Cuda,

    /// <summary>A GPU driven through Vulkan compute (Intel, AMD and others), with kernels generated as SPIR-V.</summary>
    Vulkan,

    /// <summary>A device whose backend was registered by name (see <see cref="Device.Get"/>).</summary>
    Other,
}

/// <summary>
/// A place where tensors live and where their math runs.
/// Use <see cref="Cpu"/>, <see cref="Cuda(int)"/>, <see cref="Get"/> / <see cref="Parse"/> ("vulkan:0"), or
/// <see cref="Default"/>, which picks the best GPU found and the CPU otherwise.
/// </summary>
public sealed class Device
{
    private static readonly ConcurrentDictionary<(string Kind, int Ordinal), Device> Devices = new();

    private readonly DeviceProvider? _provider;

    private Device(DeviceType type, int ordinal, DeviceProvider? provider)
    {
        Type = type;
        Ordinal = ordinal;
        _provider = provider;
    }

    /// <summary>The host CPU.</summary>
    public static Device Cpu { get; } = new(DeviceType.Cpu, 0, null);

    /// <summary>
    /// The device new tensors and layers use when none is given. Defaults to the best GPU found (a CUDA GPU first, then
    /// a Vulkan one), otherwise the CPU. Set it to change the default for the process.
    /// </summary>
    public static Device Default
    {
        get => field ??= Best();
        set => field = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>True when an NVIDIA driver is installed and at least one CUDA GPU was found.</summary>
    public static bool IsCudaAvailable => CudaBackend.DeviceCount > 0;

    /// <summary>The number of CUDA GPUs the driver reports (0 when there is no driver).</summary>
    public static int CudaDeviceCount => CudaBackend.DeviceCount;

    /// <summary>Returns the CUDA GPU with the given ordinal.</summary>
    /// <exception cref="InvalidOperationException">CUDA is not available or the ordinal is out of range.</exception>
    public static Device Cuda(int ordinal = 0) => Get("cuda", ordinal);

    /// <summary>
    /// The devices found: the CPU, then each GPU every backend lists (devices a backend can drive but does not list,
    /// such as a software Vulkan driver, are reached by name with <see cref="Get"/>).
    /// </summary>
    public static IReadOnlyList<Device> Available
    {
        get
        {
            var devices = new List<Device> { Cpu };
            foreach (var provider in DeviceProviders.All)
            {
                for (int i = 0; i < provider.Count; i++)
                {
                    if (provider.Listed(i))
                    {
                        devices.Add(Get(provider.Kind, i));
                    }
                }
            }

            return devices;
        }
    }

    /// <summary>The device of a kind ("cpu", "cuda", "vulkan", …) with the given ordinal.</summary>
    /// <exception cref="InvalidOperationException">No backend of that kind, or the ordinal is out of range.</exception>
    public static Device Get(string kind, int ordinal = 0)
    {
        ArgumentNullException.ThrowIfNull(kind);
        if (kind.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return ordinal == 0 ? Cpu : throw new InvalidOperationException("There is one CPU device, cpu:0.");
        }

        var provider = DeviceProviders.Find(kind) ?? throw new InvalidOperationException(
            $"No device kind '{kind}' (known: cpu, {string.Join(", ", DeviceProviders.All.Select(p => p.Kind))}).");
        if ((uint)ordinal >= (uint)provider.Count)
        {
            throw new InvalidOperationException(provider.Count == 0
                ? $"{provider.Display} is not available: {provider.UnavailableReason}"
                : $"{provider.Display} device {ordinal} does not exist; {provider.Count} device(s) found.");
        }

        return Devices.GetOrAdd((provider.Kind, ordinal), static (key, p) => new Device(p.Type, key.Ordinal, p), provider);
    }

    /// <summary>The device a name such as "cpu", "cuda", "cuda:1" or "vulkan:0" stands for (no ordinal: 0).</summary>
    public static Device Parse(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        int colon = name.IndexOf(':');
        return colon < 0 ? Get(name.Trim())
            : Get(name[..colon].Trim(), int.Parse(name.AsSpan(colon + 1), System.Globalization.CultureInfo.InvariantCulture));
    }

    /// <summary>Whether this is the CPU or a GPU.</summary>
    public DeviceType Type { get; }

    /// <summary>The device's index among those of its kind; always 0 for the CPU.</summary>
    public int Ordinal { get; }

    /// <summary>The kind of backend driving the device: "cpu", "cuda", "vulkan", ….</summary>
    public string Kind => _provider?.Kind ?? "cpu";

    /// <summary>A human-readable name, such as the GPU model.</summary>
    public string Name => Backend.Name;

    /// <summary>
    /// True for a GPU (a device with its own memory and kernels), false for the CPU. For questions such as "does a model
    /// on this device use GPU memory"; which kernels a device has is the backend's to answer.
    /// </summary>
    public bool IsGpu => Type != DeviceType.Cpu;

    /// <summary>The CPU and every GPU whose backend has been started (by a tensor or model placed on it).</summary>
    internal static IEnumerable<Device> InUse()
    {
        yield return Cpu;
        foreach (var provider in DeviceProviders.All)
        {
            for (int i = 0; i < provider.Count; i++)
            {
                if (provider.IsStarted(i))
                {
                    yield return Get(provider.Kind, i);
                }
            }
        }
    }

    internal Backend Backend => field ??= _provider is null ? CpuBackend.Instance : _provider.Create(Ordinal);

    /// <summary>Waits until all queued work on this device has finished.</summary>
    public void Synchronize() => Backend.Synchronize();

    /// <inheritdoc />
    public override string ToString() => _provider is null ? "cpu" : $"{_provider.Kind}:{Ordinal}";

    // The listed device ranked highest by its backend (CUDA before Vulkan), else the CPU.
    private static Device Best()
    {
        Device best = Cpu;
        int bestRank = 0;
        foreach (var provider in DeviceProviders.All)
        {
            for (int i = 0; i < provider.Count; i++)
            {
                if (provider.Listed(i) && provider.DefaultRank(i) is int rank && rank > bestRank)
                {
                    (best, bestRank) = (Get(provider.Kind, i), rank);
                }
            }
        }

        return best;
    }
}
