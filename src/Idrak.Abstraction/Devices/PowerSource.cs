// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Whether the machine runs on mains power or on its battery, as the operating system reports it. Laptops clock their
/// GPUs and CPUs down on battery, so a choice measured there (a split count, a cut-over) can be wrong on mains power and
/// the other way round: the tuning caches of every backend keep one set of choices per power source. Read once per
/// process, so the choices a process measures and saves stay under one key. A machine without a battery, or one the
/// operating system says nothing about, counts as mains. <c>IDRAK_POWER_SOURCE=ac|battery</c> overrides it (tests).
/// </summary>
public static partial class PowerSource
{
    private static readonly Lazy<string> Reported = new(Read);

    /// <summary>"ac" or "battery".</summary>
    public static string Current => Environment.GetEnvironmentVariable("IDRAK_POWER_SOURCE") is { } forced && forced is "ac" or "battery" ? forced : Reported.Value;

    private static string Read()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                // ACLineStatus 0: offline (on battery); 1: online; 255: unknown.
                return GetSystemPowerStatus(out var status) && status.ACLineStatus == 0 ? "battery" : "ac";
            }

            if (OperatingSystem.IsLinux() || OperatingSystem.IsAndroid())
            {
                return Linux("/sys/class/power_supply");
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or DllNotFoundException or EntryPointNotFoundException)
        {
        }

        return "ac";
    }

    // On battery when a battery is discharging and no mains or USB supply is online.
    internal static string Linux(string root)
    {
        if (!Directory.Exists(root))
        {
            return "ac";
        }

        bool mains = false, discharging = false;
        foreach (var supply in Directory.EnumerateDirectories(root))
        {
            string type = ReadText(Path.Combine(supply, "type"));
            if (type is "Mains" or "USB" or "USB_C" or "USB_PD")
            {
                mains |= ReadText(Path.Combine(supply, "online")) == "1";
            }
            else if (type == "Battery")
            {
                discharging |= ReadText(Path.Combine(supply, "status")) == "Discharging";
            }
        }

        return discharging && !mains ? "battery" : "ac";
    }

    private static string ReadText(string path) => File.Exists(path) ? File.ReadAllText(path).Trim() : "";

    [StructLayout(LayoutKind.Sequential)]
    private struct SystemPowerStatus
    {
        public byte ACLineStatus;
        public byte BatteryFlag;
        public byte BatteryLifePercent;
        public byte SystemStatusFlag;
        public int BatteryLifeTime;
        public int BatteryFullLifeTime;
    }

    [LibraryImport("kernel32", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetSystemPowerStatus(out SystemPowerStatus status);
}
