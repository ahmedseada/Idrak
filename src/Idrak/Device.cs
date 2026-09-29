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
}

/// <summary>
/// A place where tensors live and where their math runs.
/// Use <see cref="Cpu"/>, <see cref="Cuda(int)"/>, or <see cref="Default"/>, which picks
/// the first GPU when an NVIDIA driver is installed and the CPU otherwise.
/// </summary>
public sealed class Device
{
    private static readonly ConcurrentDictionary<int, Device> CudaDevices = new();

    private Device(DeviceType type, int ordinal)
    {
        Type = type;
        Ordinal = ordinal;
    }

    /// <summary>The host CPU.</summary>
    public static Device Cpu { get; } = new(DeviceType.Cpu, 0);

    /// <summary>
    /// The device new tensors and layers use when none is given. Defaults to the first
    /// CUDA GPU if one is available, otherwise the CPU. Set it to change the default for the process.
    /// </summary>
    public static Device Default
    {
        get => field ??= IsCudaAvailable ? Cuda(0) : Cpu;
        set => field = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>True when an NVIDIA driver is installed and at least one CUDA GPU was found.</summary>
    public static bool IsCudaAvailable => CudaBackend.DeviceCount > 0;

    /// <summary>The number of CUDA GPUs the driver reports (0 when there is no driver).</summary>
    public static int CudaDeviceCount => CudaBackend.DeviceCount;

    /// <summary>Returns the CUDA GPU with the given ordinal.</summary>
    /// <exception cref="InvalidOperationException">CUDA is not available or the ordinal is out of range.</exception>
    public static Device Cuda(int ordinal = 0)
    {
        if ((uint)ordinal >= (uint)CudaBackend.DeviceCount)
        {
            throw new InvalidOperationException(CudaBackend.DeviceCount == 0
                ? $"CUDA is not available: {CudaBackend.UnavailableReason}"
                : $"CUDA device {ordinal} does not exist; {CudaBackend.DeviceCount} device(s) found.");
        }

        return CudaDevices.GetOrAdd(ordinal, static o => new Device(DeviceType.Cuda, o));
    }

    /// <summary>Whether this is the CPU or a GPU.</summary>
    public DeviceType Type { get; }

    /// <summary>The GPU index for CUDA devices; always 0 for the CPU.</summary>
    public int Ordinal { get; }

    /// <summary>A human-readable name, such as the GPU model.</summary>
    public string Name => Type == DeviceType.Cpu
        ? $"CPU ({Environment.ProcessorCount} threads, {System.Numerics.Vector<float>.Count}-wide SIMD)"
        : ((CudaBackend)Backend).Name;

    internal Backend Backend => field ??= Type == DeviceType.Cpu ? CpuBackend.Instance : CudaBackend.Get(Ordinal);

    /// <summary>Waits until all queued work on this device has finished.</summary>
    public void Synchronize() => Backend.Synchronize();

    /// <inheritdoc />
    public override string ToString() => Type == DeviceType.Cpu ? "cpu" : $"cuda:{Ordinal}";
}
