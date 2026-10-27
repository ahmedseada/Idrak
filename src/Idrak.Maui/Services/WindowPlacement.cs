// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>A window's place on the desktop: its left and top edges and its size, in device-independent units.</summary>
/// <param name="X">The left edge.</param>
/// <param name="Y">The top edge.</param>
/// <param name="Width">The width.</param>
/// <param name="Height">The height.</param>
public sealed record WindowPlace(double X, double Y, double Width, double Height)
{
    /// <summary>Whether it is a place a window can be put at: every value a finite number, the size above zero.</summary>
    public bool IsReal => double.IsFinite(X) && double.IsFinite(Y) && double.IsFinite(Width) && double.IsFinite(Height) && Width > 0 && Height > 0;

    /// <summary>The place as it is kept: "x,y,width,height", invariant.</summary>
    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{X:R},{Y:R},{Width:R},{Height:R}");

    /// <summary>The place kept as <paramref name="text"/> ("x,y,width,height"), or null when it is not one (none kept, an older app's).</summary>
    public static WindowPlace? Parse(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return null;
        }

        Span<double> values = stackalloc double[4];
        int count = 0;
        foreach (var range in text.AsSpan().Split(','))
        {
            if (count == 4 || !double.TryParse(text.AsSpan(range), NumberStyles.Float, CultureInfo.InvariantCulture, out values[count]))
            {
                return null;
            }

            count++;
        }

        var place = count == 4 ? new WindowPlace(values[0], values[1], values[2], values[3]) : null;
        return place is { IsReal: true } ? place : null;
    }
}

/// <summary>
/// The main window on a desktop (<see cref="IWindowPlacement"/>): its least size from the options (and again when the
/// settings are saved), and, with <see cref="IdrakMauiOptions.KeepWindowPlace"/>, the size and place it had when the app
/// last ran, kept under <see cref="Key"/> in the settings store as the window moves and resizes. Each change is written off
/// the UI thread, one write at a time: changes that come while one is written make a single write after it, with the
/// latest place. A place the platform says is on no screen (a display since removed) is not kept, and a window restored
/// there is brought back to the main display.
/// </summary>
public sealed class KeptWindowPlacement : IWindowPlacement
{
    /// <summary>The settings store's key of the kept place.</summary>
    public const string Key = IdrakMauiSettings.Prefix + "window.place";

    /// <summary>The least <see cref="IdrakMauiOptions.WindowMinimumWidth"/> the settings take: the narrowest phone layout's width.</summary>
    public const double LeastWidth = 320;

    /// <summary>The least <see cref="IdrakMauiOptions.WindowMinimumHeight"/> the settings take.</summary>
    public const double LeastHeight = 320;

    private readonly ISettingsStore _store;
    private readonly IdrakMauiOptions _options;
    private readonly IdrakMauiSettings? _settings;
    private readonly Func<Window, bool>? _onScreen;
    private readonly Lock _gate = new();
    private WindowPlace? _pending;
    private bool _writing;
    private Task _write = Task.CompletedTask;

    /// <summary>
    /// The placement over <paramref name="store"/> and <paramref name="options"/>; <paramref name="settings"/>, when given,
    /// brings a saved least size to the window at once; <paramref name="onScreen"/> says whether a window is shown on a
    /// screen now (not minimised, on some display); null takes every place as on screen.
    /// </summary>
    public KeptWindowPlacement(ISettingsStore store, IdrakMauiOptions options, IdrakMauiSettings? settings = null, Func<Window, bool>? onScreen = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _settings = settings;
        _onScreen = onScreen;
    }

    /// <summary>The place kept, or null when none is.</summary>
    public WindowPlace? Kept => WindowPlace.Parse(_store.Get(Key));

    /// <summary>The write of the latest place in progress (complete when none is), for a caller that must wait for it.</summary>
    public Task Writing
    {
        get
        {
            lock (_gate)
            {
                return _write;
            }
        }
    }

    /// <inheritdoc />
    public void Attach(Window window)
    {
        ArgumentNullException.ThrowIfNull(window);
        Minimum(window, _options);
        bool restored = false;
        if (_options.KeepWindowPlace && Kept is { } place)
        {
            window.Width = Math.Max(place.Width, window.MinimumWidth);
            window.Height = Math.Max(place.Height, window.MinimumHeight);
            window.X = place.X;
            window.Y = place.Y;
            restored = true;
        }

        window.SizeChanged += (_, _) => Remember(window);
        window.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(Window.X) or nameof(Window.Y))
            {
                Remember(window);                                                // a move changes no size: SizeChanged is not raised for it
            }
        };
        window.Created += (_, _) =>
        {
            if (restored && _onScreen is { } check && !check(window))
            {
                BringBack(window);
            }
        };
        window.Destroying += (_, _) => Flush();
        if (_settings is not null)
        {
            _settings.Saved += (_, saved) => window.Dispatcher.Dispatch(() => Minimum(window, saved));
        }
    }

    /// <summary>Keeps <paramref name="window"/>'s place now (written off the calling thread); nothing when keeping is off or the place is not real or on no screen.</summary>
    public void Remember(Window window)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (!_options.KeepWindowPlace)
        {
            return;
        }

        var place = new WindowPlace(window.X, window.Y, window.Width, window.Height);
        if (!place.IsReal || (_onScreen is { } check && !check(window)))
        {
            return;                                                             // not shown yet, minimised, or on no screen
        }

        lock (_gate)
        {
            _pending = place;
            if (_writing)
            {
                return;                                                         // the write in progress takes it after its own
            }

            _writing = true;
            _write = Task.Run(Write);
        }
    }

    // Writes the latest place until none waits; a store that refuses (its file held by another write) keeps the place
    // waiting for the next change or the window's close.
    private void Write()
    {
        while (true)
        {
            WindowPlace? next;
            lock (_gate)
            {
                next = _pending;
                _pending = null;
                if (next is null)
                {
                    _writing = false;
                    return;
                }
            }

            try
            {
                _store.Set(Key, next.ToString());
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                lock (_gate)
                {
                    _pending ??= next;
                    _writing = false;
                }

                return;
            }
        }
    }

    // At the window's close: the write in progress finished, and a place still waiting written.
    private void Flush()
    {
        Writing.Wait();
        WindowPlace? last;
        lock (_gate)
        {
            last = _pending;
            _pending = null;
        }

        if (last is not null)
        {
            try
            {
                _store.Set(Key, last.ToString());
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // The place of an earlier change stays kept.
            }
        }
    }

    private static void Minimum(Window window, IdrakMauiOptions options)
    {
        window.MinimumWidth = Math.Max(options.WindowMinimumWidth, LeastWidth);
        window.MinimumHeight = Math.Max(options.WindowMinimumHeight, LeastHeight);
    }

    // A window restored on no screen: centred on the main display, no larger than it.
    private static void BringBack(Window window)
    {
        var display = DeviceDisplay.Current.MainDisplayInfo;
        double density = display.Density > 0 ? display.Density : 1;
        double width = display.Width / density, height = display.Height / density;
        if (width <= 0 || height <= 0)
        {
            return;
        }

        window.Width = Math.Min(window.Width, width);
        window.Height = Math.Min(window.Height, height);
        window.X = Math.Max(0, (width - window.Width) / 2);
        window.Y = Math.Max(0, (height - window.Height) / 2);
    }
}

/// <summary>A phone's window, which the system sizes and places: left as it is.</summary>
public sealed class SystemWindowPlacement : IWindowPlacement
{
    /// <inheritdoc />
    public void Attach(Window window) => ArgumentNullException.ThrowIfNull(window);
}
