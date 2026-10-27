// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Navigation;
using Idrak.Maui.Views;

namespace Idrak.Maui.Tests;

// The soft keyboard (plan 34, the owner's phone runs): what is typed into sits right above it, the bottom bar steps aside.
internal static partial class MauiTests
{
    // The host's bottom bar steps aside while the soft keyboard is up, and every page's bottom edge follows the keyboard.
    private static void HostKeyboard()
    {
        var layout = new AppLayout(840) { Width = 400 };
        var host = new AppHostViewModel(new IdrakMauiOptions(), layout);
        Check(host.ShowsBottomBar, "a narrow window shows the bar");
        host.IsKeyboardOpen = true;
        Check(!host.ShowsBottomBar && host.IsNarrow, "the keyboard up: no bar");
        host.IsKeyboardOpen = false;
        layout.Width = 1200;
        Check(!host.ShowsBottomBar, "a wide window: the rail, no bar");
        var page = new LabPage(new Label(), new object(), "lab");
        Check(page.SafeAreaEdges.Bottom == SafeAreaRegions.All, $"a pushed page's bottom edge: {page.SafeAreaEdges.Bottom}");
        var keyboard = new NoSoftKeyboard();
        keyboard.Watch();
        Check(!keyboard.IsVisible, "no platform: never up");
    }
}
