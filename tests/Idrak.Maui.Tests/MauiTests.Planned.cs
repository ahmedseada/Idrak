// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Text.RegularExpressions;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// The planned features (plan 34's labs not built yet): registered by their real names and places, opened on the one
// placeholder page, replaced by a ready feature in the same place, and the "planned" mark in both themes.
internal static partial class MauiTests
{
    // The library's planned labs in the plan's menu order (after Devices and Models).
    private static readonly (string Name, string Title, int Order, PhoneNeeds Needs)[] PlannedLabs =
    [
        (MobileFeatures.Search, "Search", 80, PhoneNeeds.Model | PhoneNeeds.Network),
    ];

    // The app's Self-test as apps/Idrak.Mobile registers it (AppFeatures; the app is built for Android only, so not here).
    private static PlannedFeature SelfTest() => new("self-test", "Self-test", "✓", 90, "The library's tests, run on this phone",
        "A test group by filter on cpu or vulkan:N with live progress; the failures file shared out", PhoneNeeds.None, ["Idrak.Tests.Library"]);

    private static string Menu() => string.Join(", ", MobileFeatures.All.Select(f => $"{f.Name}@{MobileFeatures.Place(f)}"));

    private static void PlannedRegistered()
    {
        var devices = MobileFeatures.Get(MobileFeatures.Devices);
        Check(devices.State == FeatureState.Ready && devices.Coming.Length == 0, "the devices feature is ready");
        foreach (var (name, title, order, needs) in PlannedLabs)
        {
            var feature = MobileFeatures.Get(name);
            Check(feature is PlannedFeature && MobileFeatures.Origin(name) == Overrides.Library && ReferenceEquals(MobileFeatures.Default(name), feature),
                $"{name}: {feature.GetType().Name} from {MobileFeatures.Origin(name)}");
            Check(feature.Name == name && feature.Title == title && feature.Order == order && MobileFeatures.Place(feature) == order && feature.State == FeatureState.Planned,
                $"{name}: title {feature.Title}, order {feature.Order}, state {feature.State}");
            Check(feature.Needs == needs && feature.Packages.Count > 0 && feature.Packages.All(p => p.StartsWith("Idrak", StringComparison.Ordinal)) && feature.ModelFamilies.Count == 0,
                $"{name}: needs {feature.Needs}, packages {string.Join(", ", feature.Packages)}, families {string.Join(", ", feature.ModelFamilies)}");
            Check(feature.Coming.Length > 0 && feature.Summary.Length > 0 && feature.Glyph.Length is > 0 and <= 2 && feature.Order > devices.Order,
                $"{name}: coming '{feature.Coming}', summary '{feature.Summary}', glyph '{feature.Glyph}'");
        }

        var plannedNames = PlannedLabs.Select(p => p.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var listed = MobileFeatures.All.Select(f => f.Name).Where(n => n == MobileFeatures.Devices || plannedNames.Contains(n)).ToList();
        Check(listed.SequenceEqual([MobileFeatures.Devices, .. PlannedLabs.Select(p => p.Name)]), $"menu: {Menu()}");

        var selfTest = SelfTest();
        try
        {
            MobileFeatures.Register(selfTest);
            Check(MobileFeatures.All[^1] == selfTest && MobileFeatures.Origin("self-test") == typeof(MauiTests).Assembly.GetName().Name && MobileFeatures.Default("self-test") is null,
                $"the app's self-test: {Menu()}");
            var options = new IdrakMauiOptions();
            options.DisabledFeatures.Add("SEARCH");
            Check(!MobileFeatures.Enabled(options).Any(f => f.Name == MobileFeatures.Search) && MobileFeatures.Enabled(options).Any(f => f.Name == MobileFeatures.Devices),
                "a planned feature is turned off like any other");
        }
        finally
        {
            MobileFeatures.Unregister("self-test");
        }

        foreach (var (empty, make) in new (string, Func<PlannedFeature>)[]
                 {
                     ("name", () => new PlannedFeature(" ", "T", "*", 1, "s", "c")),
                     ("title", () => new PlannedFeature("n", "", "*", 1, "s", "c")),
                     ("summary", () => new PlannedFeature("n", "T", "*", 1, "", "c")),
                     ("what is coming", () => new PlannedFeature("n", "T", "*", 1, "s", " ")),
                 })
        {
            bool refused = false;
            try
            {
                _ = make();
            }
            catch (ArgumentException)
            {
                refused = true;
            }

            Check(refused, $"a planned feature without its {empty} was made");
        }
    }

    private static void PlaceholderOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        foreach (var (name, title, _, needs) in PlannedLabs)
        {
            var feature = MobileFeatures.Get(name);
            var page = MobileFeatures.Open(name, services);
            Check(page is PlaceholderPage { BindingContext: PlaceholderViewModel } && page.Title == title, $"{name}: page {page.GetType().Name}, title {page.Title}");
            var vm = (PlaceholderViewModel)page.BindingContext;
            Check(vm.Feature == feature && vm.Title == title && vm.Glyph == feature.Glyph && vm.Summary == feature.Summary && vm.Coming == feature.Coming && vm.HasComing,
                $"{name}: the view model shows another feature");
            Check(vm.IsPlanned && vm.StateText == "planned" && vm.HasPackages && vm.Packages == string.Join(", ", feature.Packages) && !vm.HasModelFamilies,
                $"{name}: state {vm.StateText}, packages {vm.Packages}");
            Check(vm.NeedLines.Count == BitOperations.PopCount((uint)needs) && vm.NeedsNothing == (needs == PhoneNeeds.None), $"{name}: needs {string.Join(" | ", vm.NeedLines)}");
        }

        var search = (PlaceholderViewModel)MobileFeatures.Open(MobileFeatures.Search, services).BindingContext;
        Check(search.NeedLines.SequenceEqual(PlaceholderViewModel.Describe(PhoneNeeds.Network | PhoneNeeds.Model))
              && search.NeedLines[0].Contains("network", StringComparison.OrdinalIgnoreCase) && search.NeedLines[^1].Contains("model", StringComparison.Ordinal),
            $"search: {string.Join(" | ", search.NeedLines)}");
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
        Check(new PlaceholderViewModel(MobileFeatures.Get(MobileFeatures.Devices)) is { IsPlanned: false, StateText: "ready", HasComing: false }, "a ready feature on the placeholder");
    }

    // A feature that sets only what it must: no order, no state, nothing coming.
    private sealed class PlainFeature(string name, string title) : IMobileFeature
    {
        public string Name => name;

        public string Summary => "a plain test feature";

        public string Title => title;

        public string Glyph => "+";

        public PhoneNeeds Needs => PhoneNeeds.None;

        public IReadOnlyList<string> Packages => [];

        public IReadOnlyList<string> ModelFamilies => [];

        public object CreateViewModel(IServiceProvider services) => new object();

        public Page CreatePage(IServiceProvider services, object viewModel) => new ContentPage();
    }

    private static void PlannedReplaced()
    {
        var before = MobileFeatures.All.Select(f => f.Name).ToList();
        int at = before.IndexOf(MobileFeatures.Search);
        Check(at > 0, $"search is not in the menu: {Menu()}");
        IMobileFeature ready = new PlainFeature(MobileFeatures.Search, "Search");
        IMobileFeature ordered = new LabFeature(MobileFeatures.Search, "Search", order: 80);
        IMobileFeature zeta = new PlainFeature("zeta-lab", "Zeta"), alpha = new PlainFeature("alpha-lab", "Alpha");
        try
        {
            MobileFeatures.Register(ready);
            Check(MobileFeatures.Get(MobileFeatures.Search) == ready && ready.State == FeatureState.Ready && ready.Coming.Length == 0 && ready.Order == MobileFeatures.Unordered,
                $"the ready search: state {ready.State}, order {ready.Order}");
            Check(MobileFeatures.Place(ready) == 80 && MobileFeatures.All.Select(f => f.Name).SequenceEqual(before), $"a ready search without an order moved: {Menu()}");
            Check(MobileFeatures.Enabled(new IdrakMauiOptions()).Select(f => f.Name).SequenceEqual(before), "the enabled menu changed");

            MobileFeatures.Register(ordered);
            Check(MobileFeatures.Get(MobileFeatures.Search) == ordered && MobileFeatures.All.Select(f => f.Name).SequenceEqual(before), $"a ready search at its order moved: {Menu()}");

            MobileFeatures.Register(zeta);
            MobileFeatures.Register(alpha);
            var all = MobileFeatures.All;
            Check(all[^2] == alpha && all[^1] == zeta && MobileFeatures.Place(zeta) == MobileFeatures.Unordered && zeta.State == FeatureState.Ready,
                $"features without an order come last, by title: {Menu()}");
            Check(all.Take(all.Count - 2).All(f => MobileFeatures.Place(f) < MobileFeatures.Unordered), $"an ordered feature after the unordered: {Menu()}");
        }
        finally
        {
            MobileFeatures.Unregister(MobileFeatures.Search);
            MobileFeatures.Unregister("zeta-lab");
            MobileFeatures.Unregister("alpha-lab");
        }

        Check(MobileFeatures.Get(MobileFeatures.Search) is PlannedFeature && MobileFeatures.All.Select(f => f.Name).SequenceEqual(before), $"the planned search came back elsewhere: {Menu()}");
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

        // Sizes, spacing and colours on these pages come from theme keys ({DynamicResource ...}), never a literal.
        var literal = new Regex("""\b(TextColor|BackgroundColor|BorderColor|Stroke|StrokeThickness|FontSize|Padding|Margin|Spacing|ColumnSpacing|RowSpacing)="(?!\{)""");
        foreach (string file in (string[])[Path.Combine("apps", "Idrak.Mobile", "MenuPage.xaml"), Path.Combine("src", "Idrak.Maui", "Views", "PlaceholderPage.xaml")])
        {
            string xaml = File.ReadAllText(Path.Combine(RepositoryRoot(), file));
            var found = literal.Matches(xaml).Select(m => m.Value).ToList();
            Check(found.Count == 0, $"{file} sets {string.Join(", ", found)} without a theme key");
            Check(xaml.Contains("IdrakBadge", StringComparison.Ordinal), $"{file} has no planned mark");
        }
    }
}
