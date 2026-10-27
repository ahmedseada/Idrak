// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui;

/// <summary>What the app runs on, read at run time (never a build switch), so the defaults that depend on it can be tested.</summary>
public static class IdrakPlatform
{
    /// <summary>
    /// Whether the app runs on a desktop (Windows, macOS, Linux): a mouse and a keyboard, a pointer that hovers, other
    /// programs to start. False on a phone or a tablet's mobile system (Android, iOS).
    /// </summary>
    public static bool IsDesktop { get; } = OperatingSystem.IsMacCatalyst()
        || (!OperatingSystem.IsAndroid() && !OperatingSystem.IsIOS() && (OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() || OperatingSystem.IsLinux()));
}
