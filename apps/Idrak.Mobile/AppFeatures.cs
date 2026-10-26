// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Views;

namespace Idrak.Mobile;

/// <summary>
/// The app's own features, registered in <see cref="MobileFeatures"/> beside the library's. Self-test is the app's: it will
/// run the test library's groups (plan 34, M7b), which a MAUI library does not reference; until then it is a planned
/// feature with its real name and place, so its phase replaces only this registration's page and view model.
/// </summary>
public static class AppFeatures
{
    /// <summary>The self-test page.</summary>
    public const string SelfTest = "self-test";

    /// <summary>Its place in the menu: after the library's labs.</summary>
    public const int SelfTestOrder = 90;

    /// <summary>Registers the app's features.</summary>
    public static void Register() => MobileFeatures.Register(new PlannedFeature(SelfTest, "Self-test", "✓", SelfTestOrder,
        "The library's tests, run on this phone",
        "A test group by filter on cpu or vulkan:N with live progress; the failures file shared out",
        PhoneNeeds.None, ["Idrak.Tests.Library"], icon: "shield", tagline: "The library's own tests"));
}
