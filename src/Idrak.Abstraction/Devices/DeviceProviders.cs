// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// One kind of device beyond the CPU (CUDA, Vulkan, HIP, …): how many there are, and the backend that drives each. Devices
/// come from the registered providers (<see cref="DeviceProviders"/>), so a new kind of hardware is a provider and a
/// <see cref="Backend"/>, with no change to <see cref="Device"/> or the layers.
/// </summary>
public abstract class DeviceProvider
{
    /// <summary>The name in device names ("cuda" in "cuda:1").</summary>
    public abstract string Kind { get; }

    /// <summary>The name in messages ("CUDA").</summary>
    public virtual string Display => Kind;

    /// <summary>The <see cref="DeviceType"/> of its devices.</summary>
    public abstract DeviceType Type { get; }

    /// <summary>The number of devices found (0 when the driver is missing).</summary>
    public abstract int Count { get; }

    /// <summary>Why there are no devices, when there are none.</summary>
    public virtual string? UnavailableReason => null;

    /// <summary>Starts (or returns) the backend of device <paramref name="ordinal"/>.</summary>
    public abstract Backend Create(int ordinal);

    /// <summary>Whether device <paramref name="ordinal"/>'s backend has been started.</summary>
    public abstract bool IsStarted(int ordinal);

    /// <summary>
    /// Whether <see cref="Device.Available"/> (and so the test runner) lists the device; false for devices only worth
    /// using when asked for by name (a software rasterizer, a GPU another backend already drives).
    /// </summary>
    public virtual bool Listed(int ordinal) => true;

    /// <summary>How strongly <see cref="Device.Default"/> prefers the device (the highest wins; null: never by default).</summary>
    public virtual int? DefaultRank(int ordinal) => null;

    /// <summary>Why the device is not listed or not chosen by default, for device listings; null when there is nothing to say.</summary>
    public virtual string? Note(int ordinal) => null;

    /// <summary>
    /// The UUID the driver reports for device <paramref name="ordinal"/> (the same for one GPU across APIs: CUDA's
    /// cuDeviceGetUuid, Vulkan's deviceUUID), so a GPU two providers reach is recognized; null when unknown.
    /// </summary>
    public virtual Guid? DeviceUuid(int ordinal) => null;
}

/// <summary>
/// The device kinds beyond the CPU. The GPU devices that ship with the library (CUDA, then Vulkan, then HIP) live in the
/// Idrak.Gpu assembly, which registers them; they are registered first, before any other provider, as soon as the
/// registry is first used, so <see cref="Device.Available"/> lists them even when no type of Idrak has been touched yet.
/// </summary>
public static class DeviceProviders
{
    // A provider makes devices; a device never falls back to another (plan 10, decision 13).
    private static readonly SlotTable<string, DeviceProvider> Table = new(nameof(DeviceProviders), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a provider makes devices, which hold memory and state, and a device never falls back to another");

    static DeviceProviders() => LibraryDefaults.Ensure(typeof(DeviceProviders));   // Idrak.Gpu's devices, first

    /// <summary>The registered providers, in registration order.</summary>
    public static IReadOnlyList<DeviceProvider> All => Table.Values;

    /// <summary>
    /// Registers a provider; one of a registered kind takes its place, and the library's provider of that kind comes back
    /// with <see cref="Unregister"/>. A device does not fall back to the library's provider when an app's one fails.
    /// </summary>
    public static void Register(DeviceProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        Table.Register(provider.Kind, provider);
    }

    /// <summary>Removes the app's provider of <paramref name="kind"/> (the library's comes back, if it has one); false when the app registered none.</summary>
    public static bool Unregister(string kind) => Table.Unregister(kind);

    /// <summary>The provider of <paramref name="kind"/>, or null.</summary>
    public static DeviceProvider? Find(string kind) => Table.Find(kind);

    /// <summary>The library's provider of <paramref name="kind"/>, whatever an app registered over it; null when the library has none.</summary>
    public static DeviceProvider? Default(string kind) => Table.Default(kind);

    /// <summary>Who registered the provider of <paramref name="kind"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string kind) => Table.Origin(kind);
}
