// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cluster;

/// <summary>
/// Devices carved from a physical one (plan 23): <c>sim:N</c>, each a separate backend of the physical device
/// (<see cref="Device.CreateSeparateBackend"/>) with its own memory pool and a hard cap, so out-of-memory, sharding and
/// offloading are tested for real on one card. Several carved devices on one card share its compute, as ranks on one
/// GPU would. A carved device is the type of its physical one (a CUDA card's are CUDA devices) and runs its kernels.
/// </summary>
/// <remarks>
/// Carving the same physical device with the same cap and slot gives the same device back, so a process that runs
/// several simulated clusters reuses their devices (and their pools, rule 77) instead of carving new ones each time.
/// </remarks>
public static class SimulatedDevices
{
    /// <summary>The device kind of carved devices ("sim:0").</summary>
    public const string Kind = "sim";

    private static readonly Lock Gate = new();
    private static readonly List<Carved> Devices = [];
    private static readonly Dictionary<(string Physical, long? Cap, int Slot, bool? Offload), int> Index = [];

    /// <summary>The number of devices carved so far in this process.</summary>
    public static int Count
    {
        get
        {
            lock (Gate)
            {
                return Devices.Count;
            }
        }
    }

    /// <summary>
    /// The device carved from <paramref name="physical"/> with a cap of <paramref name="memoryBytes"/> (null: the physical
    /// device's own limit) in slot <paramref name="slot"/> (a rank's index, so two ranks on one card get two devices).
    /// </summary>
    /// <param name="physical">The device on this machine it is carved from ("cuda:0", "vulkan:0", the CPU).</param>
    /// <param name="memoryBytes">Its hard cap in bytes, or null.</param>
    /// <param name="slot">Which of the devices carved alike this is.</param>
    /// <param name="offload">Whether allocations past the cap go to pinned host memory where the kind can, or null for the process's setting.</param>
    /// <exception cref="ArgumentException">The physical device is itself a carved one, or the cap is not positive.</exception>
    public static Device Carve(Device physical, long? memoryBytes, int slot, bool? offload = null)
    {
        ArgumentNullException.ThrowIfNull(physical);
        ArgumentOutOfRangeException.ThrowIfNegative(slot);
        if (physical.Kind == Kind)
        {
            throw new ArgumentException($"{physical} is a carved device; carve from the physical device it came from ({PhysicalOf(physical)}).", nameof(physical));
        }

        if (memoryBytes is <= 0)
        {
            throw new ArgumentException($"A carved device's cap must be more than 0 bytes (got {memoryBytes}).", nameof(memoryBytes));
        }

        int ordinal;
        lock (Gate)
        {
            var key = (physical.ToString(), memoryBytes, slot, offload);
            if (!Index.TryGetValue(key, out ordinal))
            {
                ordinal = Devices.Count;
                Devices.Add(new Carved(physical, memoryBytes, offload, ordinal));
                Index[key] = ordinal;
            }
        }

        return Device.Get(Kind, ordinal);
    }

    /// <summary>The physical device <paramref name="device"/> was carved from, or the device itself when it is not a carved one.</summary>
    public static Device PhysicalOf(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return device.Kind == Kind ? Get(device.Ordinal).Physical : device;
    }

    /// <summary>The cap of a carved device in bytes, or null (the physical device's own limit, or not a carved device).</summary>
    public static long? CapOf(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return device.Kind == Kind ? Get(device.Ordinal).Cap : null;
    }

    private static Carved Get(int ordinal)
    {
        lock (Gate)
        {
            return (uint)ordinal < (uint)Devices.Count ? Devices[ordinal] : throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, $"{Devices.Count} devices have been carved.");
        }
    }

    // One carved device: its backend is made on first use.
    private sealed class Carved(Device physical, long? cap, bool? offload, int ordinal)
    {
        private readonly Lock _gate = new();

        public Device Physical { get; } = physical;

        public long? Cap { get; } = cap;

        public Backend? Backend { get; private set; }

        public string Name => string.Create(CultureInfo.InvariantCulture,
            $"{Kind}:{ordinal} ({Physical}, {(Cap is { } c ? ClusterTopology.FormatMemory(c) + " cap" : "its own memory")})");

        public Backend Start()
        {
            lock (_gate)
            {
                return Backend ??= Physical.CreateSeparateBackend(new SeparateBackendOptions(Name, Cap, offload));
            }
        }
    }

    // The "sim" kind in DeviceProviders: one device per carving, never listed or chosen by default.
    internal sealed class Provider : DeviceProvider
    {
        public override string Kind => SimulatedDevices.Kind;

        public override string Display => "Simulated";

        public override DeviceType Type => DeviceType.Other;

        public override DeviceType TypeOf(int ordinal) => (uint)ordinal < (uint)Count ? Get(ordinal).Physical.Type : Type;

        public override int Count => SimulatedDevices.Count;

        public override string? UnavailableReason => "no device has been carved yet (SimulatedDevices.Carve, or a simulated cluster run)";

        public override Backend Create(int ordinal) => Get(ordinal).Start();

        public override bool IsStarted(int ordinal) => (uint)ordinal < (uint)Count && Get(ordinal).Backend is not null;

        public override bool Listed(int ordinal) => false;

        public override string? Note(int ordinal) => $"carved: {Get(ordinal).Name}";

        public override Guid? DeviceUuid(int ordinal)
        {
            var physical = Get(ordinal).Physical;
            return physical.Type == DeviceType.Cpu ? null : DeviceProviders.Find(physical.Kind)?.DeviceUuid(physical.Ordinal);
        }
    }
}
