// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Android.Content;
using Android.Content.Res;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// Android's memory-pressure signal: an android.content.ComponentCallbacks2 (Android.Content.IComponentCallbacks2)
// registered on the application context (Context.registerComponentCallbacks), so it hears onTrimMemory(int level) and
// onLowMemory() for the whole app, whatever activity is shown. The levels (ComponentCallbacks2.TRIM_MEMORY_*,
// Android.Content.TrimMemory): RUNNING_MODERATE 5, RUNNING_LOW 10, RUNNING_CRITICAL 15 while the app runs; UI_HIDDEN 20
// when its window is hidden; BACKGROUND 40, MODERATE 60, COMPLETE 80 while it is cached in the background, COMPLETE
// the last before the system ends the process. Raised on the main thread, as Android calls these.
internal sealed class AndroidMemoryPressure : Java.Lang.Object, IComponentCallbacks2, IMemoryPressureSource
{
    public AndroidMemoryPressure() => global::Android.App.Application.Context.RegisterComponentCallbacks(this);

    public event EventHandler<MemoryPressureSignal>? Pressure;

    public void OnTrimMemory(TrimMemory level)
    {
        var (pressure, name) = level switch
        {
            TrimMemory.RunningModerate => (MemoryPressureLevel.Moderate, "TRIM_MEMORY_RUNNING_MODERATE"),
            TrimMemory.RunningLow => (MemoryPressureLevel.Low, "TRIM_MEMORY_RUNNING_LOW"),
            TrimMemory.RunningCritical => (MemoryPressureLevel.Critical, "TRIM_MEMORY_RUNNING_CRITICAL"),
            TrimMemory.UiHidden => (MemoryPressureLevel.Background, "TRIM_MEMORY_UI_HIDDEN"),
            TrimMemory.Background => (MemoryPressureLevel.Background, "TRIM_MEMORY_BACKGROUND"),
            TrimMemory.Moderate => (MemoryPressureLevel.Low, "TRIM_MEMORY_MODERATE"),
            TrimMemory.Complete => (MemoryPressureLevel.Critical, "TRIM_MEMORY_COMPLETE"),
            _ => ((int)level >= 80 ? MemoryPressureLevel.Critical : (int)level >= 15 ? MemoryPressureLevel.Low : MemoryPressureLevel.Moderate, $"TRIM_MEMORY {(int)level}"),
        };
        Pressure?.Invoke(this, new MemoryPressureSignal(pressure, name));
    }

    public void OnLowMemory() => Pressure?.Invoke(this, new MemoryPressureSignal(MemoryPressureLevel.Critical, "onLowMemory"));

    public void OnConfigurationChanged(Configuration newConfig)
    {
        // Configuration changes (rotation, theme) do not concern memory; the theme follows through MAUI.
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            global::Android.App.Application.Context.UnregisterComponentCallbacks(this);
        }

        base.Dispose(disposing);
    }
}
