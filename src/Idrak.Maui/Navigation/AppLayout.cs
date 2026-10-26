// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Navigation;

/// <summary>
/// The window's layout as measured: its width against <see cref="Breakpoint"/> (<see cref="IdrakMauiOptions.WideLayoutWidth"/>)
/// chooses the wide layout (the side rail, pages in columns) or the narrow one (the bottom bar, one column). The host page
/// sets <see cref="Width"/> each time the window's size changes (rotation, split screen, a resized window), and every view
/// that binds <see cref="IsWide"/> follows at once. Nothing here asks what device it runs on.
/// </summary>
public sealed class AppLayout : ViewModel
{
    private double _width;
    private double _breakpoint;
    private bool _wide;

    /// <summary>A layout that turns wide from <paramref name="breakpoint"/> units.</summary>
    public AppLayout(double breakpoint = 840)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(breakpoint);
        _breakpoint = breakpoint;
    }

    /// <summary>The window width from which the layout is wide.</summary>
    public double Breakpoint
    {
        get => _breakpoint;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            if (SetProperty(ref _breakpoint, value))
            {
                Measure();
            }
        }
    }

    /// <summary>The window's width as last measured (0 before any).</summary>
    public double Width
    {
        get => _width;
        set
        {
            if (SetProperty(ref _width, value))
            {
                Measure();
            }
        }
    }

    /// <summary>Whether the wide layout shows: the rail and the pages in columns.</summary>
    public bool IsWide
    {
        get => _wide;
        private set
        {
            if (SetProperty(ref _wide, value))
            {
                OnPropertyChanged(nameof(IsNarrow));
                LayoutChanged?.Invoke(this, value);
            }
        }
    }

    /// <summary>Whether the narrow layout shows: the bottom bar and one column.</summary>
    public bool IsNarrow => !IsWide;

    /// <summary>Raised when the layout turns wide (true) or narrow (false).</summary>
    public event EventHandler<bool>? LayoutChanged;

    /// <summary>Whether a window <paramref name="width"/> wide takes the wide layout from <paramref name="breakpoint"/>.</summary>
    public static bool IsWideAt(double width, double breakpoint) => double.IsFinite(width) && width >= breakpoint;

    private void Measure() => IsWide = IsWideAt(_width, _breakpoint);
}

/// <summary>The words the app uses for the device it runs on ("phone", "tablet", "device"), from the platform's idiom.</summary>
public static class DeviceWords
{
    /// <summary>"phone" on a phone, "tablet" on a tablet, "device" elsewhere or where the idiom cannot be read.</summary>
    public static string Noun
    {
        get
        {
            try
            {
                var idiom = Microsoft.Maui.Devices.DeviceInfo.Current.Idiom;
                return idiom == Microsoft.Maui.Devices.DeviceIdiom.Phone ? "phone" : idiom == Microsoft.Maui.Devices.DeviceIdiom.Tablet ? "tablet" : "device";
            }
            catch (Exception e) when (e is NotImplementedException or NotSupportedException or InvalidOperationException)
            {
                return "device";                                                // no platform (the tests): a word that fits any
            }
        }
    }

    /// <summary><paramref name="text"/> with "{device}" read as <paramref name="noun"/>.</summary>
    public static string Fill(string text, string noun) => (text ?? "").Replace("{device}", noun, StringComparison.Ordinal);
}
