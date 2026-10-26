// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Android.App;
using Android.Runtime;

namespace Idrak.Mobile;

/// <summary>The Android application: builds the MAUI app (<see cref="MauiProgram.CreateMauiApp"/>).</summary>
[Application]
public class MainApplication(IntPtr handle, JniHandleOwnership ownership) : MauiApplication(handle, ownership)
{
    /// <inheritdoc />
    protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
}
