// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Navigation;

/// <summary>The bottom bar of a narrow window, bound to the host's <see cref="AppHostViewModel"/>.</summary>
public partial class BottomNavigationBar : ContentView
{
    /// <summary>The bar (its binding context is the host's view model).</summary>
    public BottomNavigationBar() => InitializeComponent();
}
