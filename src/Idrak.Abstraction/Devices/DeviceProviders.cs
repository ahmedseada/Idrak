// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// One kind of device beyond the CPU (CUDA, Vulkan, HIP, …): how many there are, and the backend that drives each. Devices
/// come from the registered providers (<see cref="DeviceProviders"/>), registered by <see cref="Name"/>, so a new kind of
/// hardware is a provider and a <see cref="Backend"/>, with no change to <see cref="Device"/> or the layers. A kind that
/// can carve several backends out of one device implements <see cref="ISeparableDeviceProvider"/> too.
/// </summary>
public interface IDeviceProvider
{
    /// <summary>The kind's name in device names ("cuda" in "cuda:1"): the key it is registered and found by.</summary>
    string Name { get; }

    /// <summary>One line about the kind, for listings and help ("GPUs through the CUDA driver").</summary>
    string Summary { get; }

    /// <summary>The name in messages ("CUDA"); <see cref="Name"/> unless the provider says otherwise.</summary>
    string Display => Name;

    /// <summary>The <see cref="DeviceType"/> of its devices.</summary>
    DeviceType Type { get; }

    /// <summary>
    /// The <see cref="DeviceType"/> of device <paramref name="ordinal"/>: <see cref="Type"/> unless the provider's devices
    /// differ (a simulated cluster's devices are each the type of the device they are carved from).
    /// </summary>
    DeviceType TypeOf(int ordinal) => Type;

    /// <summary>The number of devices found (0 when the driver is missing).</summary>
    int Count { get; }

    /// <summary>Why there are no devices, when there are none; null when there is nothing to say.</summary>
    string? UnavailableReason => null;

    /// <summary>Starts (or returns) the backend of device <paramref name="ordinal"/>.</summary>
    Backend Create(int ordinal);

    /// <summary>Whether device <paramref name="ordinal"/>'s backend has been started.</summary>
    bool IsStarted(int ordinal);

    /// <summary>
    /// Whether <see cref="Device.Available"/> (and so the test runner) lists the device (true unless the provider says
    /// otherwise); false for devices only worth using when asked for by name (a software rasterizer, a GPU another backend
    /// already drives).
    /// </summary>
    bool Listed(int ordinal) => true;

    /// <summary>How strongly <see cref="Device.Default"/> prefers the device (the highest wins; null, the default: never by default).</summary>
    int? DefaultRank(int ordinal) => null;

    /// <summary>Why the device is not listed or not chosen by default, for device listings; null when there is nothing to say.</summary>
    string? Note(int ordinal) => null;

    /// <summary>
    /// The UUID the driver reports for device <paramref name="ordinal"/> (the same for one GPU across APIs: CUDA's
    /// cuDeviceGetUuid, Vulkan's deviceUUID), so a GPU two providers reach is recognized; null when unknown.
    /// </summary>
    Guid? DeviceUuid(int ordinal) => null;
}

/// <summary>
/// A device kind that makes separate backends of one device (<see cref="Device.CreateSeparateBackend"/>): a simulated
/// cluster carves several devices out of one this way (plan 23). A kind without it cannot.
/// </summary>
public interface ISeparableDeviceProvider : IDeviceProvider
{
    /// <summary>
    /// A new backend of device <paramref name="ordinal"/>, apart from the one <see cref="IDeviceProvider.Create"/> returns:
    /// its own memory pool, streams and memory cap (<see cref="SeparateBackendOptions"/>), sharing the hardware.
    /// </summary>
    Backend CreateSeparate(int ordinal, SeparateBackendOptions options);
}

/// <summary>
/// The device kinds beyond the CPU. The GPU devices that ship with the library (CUDA, then Vulkan, then HIP) live in the
/// Idrak.Gpu assembly, which registers them; they are registered first, before any other provider, as soon as the
/// registry is first used, so <see cref="Device.Available"/> lists them even when no type of Idrak has been touched yet.
/// </summary>
public static class DeviceProviders
{
    // A provider makes devices; a device never falls back to another (plan 10, decision 13).
    private static readonly SlotTable<string, IDeviceProvider> Table = new(nameof(DeviceProviders), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a provider makes devices, which hold memory and state, and a device never falls back to another");

    static DeviceProviders()
    {
        LibraryDefaults.Ensure(typeof(DeviceProviders));                  // Idrak.Gpu's devices, first
        Table.RegisterDefault(DeferredTensors.Kind, new DeferredProvider());   // tensors not read yet (plan 23, C8b): by name only
    }

    /// <summary>The registered providers, in registration order.</summary>
    public static IReadOnlyList<IDeviceProvider> All => Table.Values;

    /// <summary>
    /// Registers a provider under its <see cref="IDeviceProvider.Name"/>; one of a registered kind takes its place, and the
    /// library's provider of that kind comes back with <see cref="Unregister"/>. A device does not fall back to the
    /// library's provider when an app's one fails.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDeviceProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's provider of <paramref name="kind"/> (the library's comes back, if it has one); false when the app registered none.</summary>
    public static bool Unregister(string kind) => Table.Unregister(kind);

    /// <summary>The provider of <paramref name="kind"/>, or null.</summary>
    public static IDeviceProvider? Find(string kind) => Table.Find(kind);

    /// <summary>The library's provider of <paramref name="kind"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDeviceProvider? Default(string kind) => Table.Default(kind);

    /// <summary>Who registered the provider of <paramref name="kind"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string kind) => Table.Origin(kind);
}

/// <summary>What a separate backend of a device (<see cref="ISeparableDeviceProvider.CreateSeparate"/>, <see cref="Device.CreateSeparateBackend"/>) is given.</summary>
/// <param name="Name">Its name in messages and listings ("sim:2 (cuda:0, 8 GB cap)").</param>
/// <param name="MemoryLimit">Its hard cap in bytes, or null for the limit every device of its kind has.</param>
/// <param name="Offload">
/// Whether allocations past the cap go to pinned host memory where the kind can (CUDA), or null for
/// <see cref="ComputeResources.OffloadToHostMemory"/>. Where the kind cannot, the cap is hard: an allocation past it throws
/// <see cref="ResourceLimitExceededException"/>.
/// </param>
public sealed record SeparateBackendOptions(string Name, long? MemoryLimit, bool? Offload = null);
