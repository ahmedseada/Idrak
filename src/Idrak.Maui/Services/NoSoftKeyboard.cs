// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>A platform without a soft keyboard to watch (the tests, a desktop): never up.</summary>
public sealed class NoSoftKeyboard : ISoftKeyboard
{
    /// <inheritdoc />
    public bool IsVisible => false;

    /// <inheritdoc />
    public event EventHandler<bool>? VisibilityChanged
    {
        add { }
        remove { }
    }

    /// <inheritdoc />
    public void Watch()
    {
    }
}
