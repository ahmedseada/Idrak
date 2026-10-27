// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// Whether the phone's soft keyboard is up, for the app's host: while it is, the bottom bar steps aside (the keyboard
/// covers where it was) and the page's bottom edge follows the keyboard (<c>SafeAreaEdges</c>), so a composer sits right
/// above it. On Android it reads the input method's visibility from the window's insets at each layout; where a platform
/// has none, <c>Idrak.Maui.NoSoftKeyboard</c> (never up).
/// </summary>
public interface ISoftKeyboard
{
    /// <summary>Whether the keyboard is up now.</summary>
    bool IsVisible { get; }

    /// <summary>Raised on the UI thread when the keyboard comes up (true) or goes (false).</summary>
    event EventHandler<bool>? VisibilityChanged;

    /// <summary>Starts watching the shown window (the host calls it when it appears; again changes nothing).</summary>
    void Watch();
}
