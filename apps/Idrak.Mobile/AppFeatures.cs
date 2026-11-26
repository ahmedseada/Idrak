// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Mobile;

/// <summary>
/// The app's own features. Self-test is the app's: it will run the test library's groups (plan 34, M7b), which a MAUI
/// library does not reference; until then it is a planned feature with its real name and place. Listing it beside the library's
/// features is not on the new design yet.
/// </summary>
public static class AppFeatures
{
    /// <summary>The self-test page.</summary>
    public const string SelfTest = "self-test";

    /// <summary>Its place in the menu: after the library's labs.</summary>
    public const int SelfTestOrder = 90;
}
