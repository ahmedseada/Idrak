// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>How hard the system asks the app to give memory back, the least first.</summary>
public enum MemoryPressureLevel
{
    /// <summary>The app's window is hidden or it went to the background (Android: TRIM_MEMORY_UI_HIDDEN, BACKGROUND).</summary>
    Background,

    /// <summary>The system is running somewhat low while the app runs (Android: TRIM_MEMORY_RUNNING_MODERATE).</summary>
    Moderate,

    /// <summary>The system is running low while the app runs (Android: TRIM_MEMORY_RUNNING_LOW).</summary>
    Low,

    /// <summary>The system is about to end processes (Android: TRIM_MEMORY_RUNNING_CRITICAL, COMPLETE, onLowMemory).</summary>
    Critical,
}

/// <summary>One memory-pressure signal.</summary>
/// <param name="Level">How hard the system asks.</param>
/// <param name="Source">What the platform called it ("TRIM_MEMORY_RUNNING_LOW"), for logs.</param>
public sealed record MemoryPressureSignal(MemoryPressureLevel Level, string Source);

/// <summary>
/// The phone's signal to give memory back. On Android it is <c>IComponentCallbacks2.OnTrimMemory</c> and
/// <c>OnLowMemory</c>, registered on the application context; where a platform has no signal,
/// <c>Idrak.Maui.ManualMemoryPressure</c> raises it when asked (tests, a button).
/// </summary>
public interface IMemoryPressureSource
{
    /// <summary>Raised on the thread the platform signals on, once a signal.</summary>
    event EventHandler<MemoryPressureSignal>? Pressure;
}
