// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Maui.Platforms.Windows;

// Whether the app's window is shown on a screen now, for the kept window place (KeptWindowPlacement): its WinUI window's
// handle (WinRT.Interop.WindowNative.GetWindowHandle) neither minimised (user32's IsIconic: a minimised window's place is
// the system's parking spot, never kept) nor off every display (MonitorFromWindow with MONITOR_DEFAULTTONULL gives none
// when the window meets no monitor, as after a display it was on is removed). A window not made yet is taken as shown.
internal static partial class WindowsScreens
{
    private const uint MonitorDefaultToNull = 0;

    public static bool IsShown(Window window)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (window.Handler?.PlatformView is not global::Microsoft.UI.Xaml.Window native)
        {
            return true;
        }

        nint handle = global::WinRT.Interop.WindowNative.GetWindowHandle(native);
        return handle != 0 && !IsIconic(handle) && MonitorFromWindow(handle, MonitorDefaultToNull) != 0;
    }

    [LibraryImport("user32")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool IsIconic(nint window);

    [LibraryImport("user32")]
    private static partial nint MonitorFromWindow(nint window, uint flags);
}
