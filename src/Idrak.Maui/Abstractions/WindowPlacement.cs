// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// The app's main window on a desktop: the least size it may be made, and its size and place kept between runs
/// (<see cref="IdrakMauiOptions.WindowMinimumWidth"/>, <see cref="IdrakMauiOptions.WindowMinimumHeight"/>,
/// <see cref="IdrakMauiOptions.KeepWindowPlace"/>). <c>UseIdrak</c> registers <c>Idrak.Maui.KeptWindowPlacement</c> on
/// Windows; on a phone the system sizes the window and <c>Idrak.Maui.SystemWindowPlacement</c> leaves it as it is. The
/// window is attached where it is made (<c>AppHostPage.CreateWindow</c>).
/// </summary>
public interface IWindowPlacement
{
    /// <summary>Gives <paramref name="window"/> its least size and the size and place it was kept at, then keeps them as it changes.</summary>
    void Attach(Window window);
}
