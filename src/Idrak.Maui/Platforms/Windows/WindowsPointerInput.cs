// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Maui.Controls;
using FrameworkElement = Microsoft.UI.Xaml.FrameworkElement;
using PointerRoutedEventArgs = Microsoft.UI.Xaml.Input.PointerRoutedEventArgs;

namespace Idrak.Maui.Platforms.Windows;

// The mouse wheel on Windows: over a ZoomView (Generate's preview) a vertical wheel zooms about the pointer, a notch a step
// (ZoomView.ZoomByWheel), and the wheel is taken (nothing behind scrolls). A touchpad's pinch reaches a WinUI app as the
// same wheel with Ctrl, so it zooms too; a touch screen's pinch stays MAUI's pinch gesture. The view gets a transparent
// background when it has none, so the wheel is read over all of it, not only over the picture's pixels.
internal static class WindowsPointerInput
{
    private static readonly ConditionalWeakTable<FrameworkElement, object> Hooked = new();

    /// <summary>The wheel zooms <paramref name="view"/> from now on (hooked again when MAUI gives it a new platform view).</summary>
    public static void Watch(ZoomView view)
    {
        ArgumentNullException.ThrowIfNull(view);
        view.HandlerChanged += (_, _) => Hook(view);
        Hook(view);
    }

    private static void Hook(ZoomView view)
    {
        if (view.Handler?.PlatformView is not FrameworkElement element || Hooked.TryGetValue(element, out _))
        {
            return;
        }

        Hooked.Add(element, new object());
        if (view.Background is null && view.BackgroundColor is null)
        {
            view.BackgroundColor = Colors.Transparent;                           // hit-tested everywhere, drawn as before
        }

        element.PointerWheelChanged += (_, e) => OnWheel(view, element, e);
    }

    private static void OnWheel(ZoomView view, FrameworkElement element, PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(element);
        if (point.Properties.IsHorizontalMouseWheel || element.ActualWidth <= 0 || element.ActualHeight <= 0)
        {
            return;
        }

        view.ZoomByWheel(point.Properties.MouseWheelDelta / ZoomView.WheelNotch, point.Position.X / element.ActualWidth, point.Position.Y / element.ActualHeight);
        e.Handled = true;
    }
}
