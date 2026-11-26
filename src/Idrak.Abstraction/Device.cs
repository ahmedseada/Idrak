// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction;

/// <summary>The kind of hardware a <see cref="Device"/> runs on.</summary>
public enum DeviceType
{
    /// <summary>The host CPU, using SIMD and multi-threading.</summary>
    Cpu,

    /// <summary>An NVIDIA GPU, driven through the CUDA driver API.</summary>
    Cuda,

    /// <summary>A GPU driven through Vulkan compute (Intel, AMD and others), with kernels generated as SPIR-V.</summary>
    Vulkan,

    /// <summary>Another kind of device.</summary>
    Other,

    /// <summary>An AMD GPU driven through the HIP runtime (the graphics driver's on Windows, ROCm's on Linux), with Idrak's own GCN kernels; no SDK.</summary>
    Hip,
}

/// <summary>
/// A place where tensors live and where their math runs.
/// Use <see cref="Cpu"/>, <see cref="Get"/> / <see cref="Parse"/> ("cpu"), or <see cref="Default"/> (the CPU unless set).
/// </summary>
public sealed class Device
{
    private Device(DeviceType type, int ordinal)
    {
        Type = type;
        Ordinal = ordinal;
    }

    /// <summary>The host CPU.</summary>
    public static Device Cpu { get; } = new(DeviceType.Cpu, 0);

    // The default a host set, or null.
    private static Device? s_setDefault;

    /// <summary>The device new tensors and layers use when none is given: the CPU unless set. Set it to change the default for the process.</summary>
    public static Device Default
    {
        get => s_setDefault ?? Cpu;
        set => s_setDefault = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>The devices found: the CPU.</summary>
    public static IReadOnlyList<Device> Available => [Cpu];

    /// <summary>The device of a kind ("cpu") with the given ordinal.</summary>
    /// <exception cref="InvalidOperationException">No device of that kind, or the ordinal is out of range.</exception>
    public static Device Get(string kind, int ordinal = 0)
    {
        ArgumentNullException.ThrowIfNull(kind);
        if (kind.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return ordinal == 0 ? Cpu : throw new InvalidOperationException("There is one CPU device, cpu:0.");
        }

        throw new InvalidOperationException($"No device kind '{kind}' (known: cpu).");
    }

    /// <summary>The device a name such as "cpu" or "cpu:0" stands for (no ordinal: 0).</summary>
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

    /// <summary>The kind of backend driving the device: "cpu".</summary>
    public string Kind => "cpu";

    /// <summary>A human-readable name, such as the GPU model.</summary>
    public string Name => Backend.Name;

    /// <summary>
    /// True for a GPU (a device with its own memory and kernels), false for the CPU. For questions such as "does a model
    /// on this device use GPU memory"; which kernels a device has is the backend's to answer.
    /// </summary>
    public bool IsGpu => Type != DeviceType.Cpu;

    /// <summary>The devices whose backend has been started: the CPU.</summary>
    internal static IEnumerable<Device> InUse()
    {
        yield return Cpu;
    }

    /// <summary>The device's backend (the device contract: memory, copies and kernels), started on first use.</summary>
    public Backend Backend => CpuBackend.Instance;

    /// <summary>
    /// A new backend of this device, apart from <see cref="Backend"/>: its own memory pool, streams and memory cap, sharing
    /// the hardware (plan 23: a simulated cluster's devices are carved from a physical one this way).
    /// </summary>
    /// <remarks>
    /// The caller owns it and disposes it (<see cref="Backend.Dispose()"/>), which gives its memory, kernels and streams back
    /// to the device; its tensors cannot be used after that. <see cref="Backend"/>, the device's own, lives for the process
    /// and is never disposed.
    /// </remarks>
    public Backend CreateSeparateBackend(SeparateBackendOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return CpuBackend.CreateSeparate(options);
    }

    /// <summary>Waits until all queued work on this device has finished.</summary>
    public void Synchronize() => Backend.Synchronize();

    /// <inheritdoc />
    public override string ToString() => "cpu";

}
