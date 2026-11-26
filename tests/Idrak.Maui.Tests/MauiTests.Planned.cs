// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// The planned features (plan 34's labs not built yet): opened on the one placeholder page, and the "planned" mark in both
// themes.
internal static partial class MauiTests
{
    // The app's Self-test as apps/Idrak.Mobile has it (AppFeatures; the app is built for Android only, so not here).
    private static PlannedFeature SelfTest() => new("self-test", "Self-test", "✓", 90, "The library's tests, run on this phone",
        "A test group by filter on cpu or vulkan:N with live progress; the failures file shared out", PhoneNeeds.None, ["Idrak.Tests.Library"]);

    private static void PlaceholderOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var lines = PlaceholderViewModel.Describe(PhoneNeeds.Network | PhoneNeeds.Model);
        Check(lines[0].Contains("network", StringComparison.OrdinalIgnoreCase) && lines[^1].Contains("model", StringComparison.Ordinal), $"search: {string.Join(" | ", lines)}");
        Check(PlaceholderViewModel.Describe(PhoneNeeds.Camera).Single().Contains("camera", StringComparison.Ordinal) && PlaceholderViewModel.Describe(PhoneNeeds.None).Count == 0,
            "the camera's line, and none for nothing");

        var selfTest = SelfTest();
        object model = selfTest.CreateViewModel(services);
        var selfPage = selfTest.CreatePage(services, model);
        Check(selfPage is PlaceholderPage && model is PlaceholderViewModel { NeedsNothing: true, IsPlanned: true, Title: "Self-test" }, "the app's self-test opens the placeholder");
        bool refused = false;
        try
        {
            selfTest.CreatePage(services, new object());
        }
        catch (ArgumentException)
        {
            refused = true;
        }

        Check(refused, "the placeholder page took another view model");
        Check(new PlaceholderViewModel(new DevicesFeature()) is { IsPlanned: false, StateText: "ready", HasComing: false }, "a ready feature on the placeholder");
    }

    // The contrast of two colours (WCAG 2: relative luminances, 1 to 21).
    private static double Contrast(Color a, Color b)
    {
        static double Luminance(Color c)
        {
            static double Channel(float v) => v <= 0.04045 ? v / 12.92 : Math.Pow((v + 0.055) / 1.055, 2.4);
            return 0.2126 * Channel(c.Red) + 0.7152 * Channel(c.Green) + 0.0722 * Channel(c.Blue);
        }

        double la = Luminance(a), lb = Luminance(b);
        return (Math.Max(la, lb) + 0.05) / (Math.Min(la, lb) + 0.05);
    }

    private static void PlannedMark()
    {
        var styles = new IdrakStyles();
        foreach (string key in (string[])["IdrakBadge", "IdrakBadgeLabel", "IdrakFontSizeBadge", "IdrakBadgePadding", "IdrakMenuHeaderPadding", "IdrakMenuItemPadding"])
        {
            Check(styles.ContainsKey(key), $"the styles have no {key}");
        }

        foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
        {
            var colors = IdrakThemes.Colors(theme);
            Color Get(string key) => colors.TryGetValue(key, out var value) && value is Color c ? c : throw new CheckException($"{theme}: no colour {key}");
            Color text = Get("IdrakBadgeForeground"), ground = Get("IdrakBadgeBackground");
            Check(Contrast(text, ground) >= 4.5, $"{theme}: the mark's text on its ground, contrast {Contrast(text, ground):F2}");
            foreach (string around in (string[])["IdrakFlyoutBackground", "IdrakCardBackground", "IdrakPageBackground"])
            {
                Check(Contrast(text, Get(around)) >= 4.5 && !ground.Equals(Get(around)), $"{theme}: the mark on {around}, contrast {Contrast(text, Get(around)):F2}");
            }

            Check(!Get("IdrakBadgeStroke").Equals(Get("IdrakFlyoutBackground")), $"{theme}: the mark's edge is the menu's ground");
        }

        // The placeholder (a planned lab) shows the mark; every page's sizes, spacing and colours are theme keys (ThemeKeysEverywhere).
        string xaml = File.ReadAllText(Path.Combine(RepositoryRoot(), "src", "Idrak.Maui", "Views", "PlaceholderView.xaml"));
        Check(xaml.Contains("IdrakBadge", StringComparison.Ordinal), "the placeholder view has no planned mark");
        ThemeKeysOnly(Path.Combine("src", "Idrak.Maui", "Views", "PlaceholderView.xaml"));
    }
}
