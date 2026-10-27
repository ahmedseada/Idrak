// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using AndroidX.Core.View;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// Android's soft keyboard: after each layout of the shown activity's window (ViewTreeObserver.GlobalLayout), the root
// window insets say whether the input method is visible (WindowInsetsCompat.Type.Ime(): exact from API 30, AndroidX's
// reading of the bottom inset before). Only read: MAUI keeps its own insets listener, which pads the page for the
// keyboard (SafeAreaEdges); this one tells the host, which takes the bottom bar away meanwhile.
internal sealed class AndroidSoftKeyboard : ISoftKeyboard
{
    private global::Android.Views.View? _root;
    private bool _visible;

    public bool IsVisible => _visible;

    public event EventHandler<bool>? VisibilityChanged;

    public void Watch()
    {
        var root = Microsoft.Maui.ApplicationModel.Platform.CurrentActivity?.Window?.DecorView;
        if (root is null || ReferenceEquals(root, _root))
        {
            return;
        }

        if (_root?.ViewTreeObserver is { IsAlive: true } previous)
        {
            previous.GlobalLayout -= OnLayout;
        }

        _root = root;
        if (root.ViewTreeObserver is { } observer)
        {
            observer.GlobalLayout += OnLayout;
        }

        OnLayout(null, EventArgs.Empty);
    }

    private void OnLayout(object? sender, EventArgs e)
    {
        bool visible = _root is { } root && ViewCompat.GetRootWindowInsets(root) is { } insets && insets.IsVisible(WindowInsetsCompat.Type.Ime());
        if (visible != _visible)
        {
            _visible = visible;
            VisibilityChanged?.Invoke(this, visible);
        }
    }
}
