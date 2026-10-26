// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>
/// The placeholder page of a feature not built yet (<see cref="FeatureState.Planned"/>), or of any feature by its
/// registration alone: its title, glyph and summary, what it will do (<see cref="IMobileFeature.Coming"/>), what it will
/// need from the phone (<see cref="IMobileFeature.Needs"/>, one line each) and the library's packages and model families
/// it will use. Everything is read from the feature, so one page serves them all.
/// </summary>
public sealed class PlaceholderViewModel : ViewModel
{
    /// <summary>A view model of <paramref name="feature"/>.</summary>
    public PlaceholderViewModel(IMobileFeature feature)
    {
        Feature = feature ?? throw new ArgumentNullException(nameof(feature));
        NeedLines = Describe(feature.Needs);
    }

    /// <summary>The feature shown.</summary>
    public IMobileFeature Feature { get; }

    /// <summary>The page's title.</summary>
    public string Title => Feature.Title;

    /// <summary>The feature's mark.</summary>
    public string Glyph => Feature.Glyph;

    /// <summary>One line on what it is.</summary>
    public string Summary => Feature.Summary;

    /// <summary>What it will do once built ("" when the feature says nothing).</summary>
    public string Coming => Feature.Coming;

    /// <summary>Whether <see cref="Coming"/> has something to show.</summary>
    public bool HasComing => Coming.Length > 0;

    /// <summary>Whether the feature is planned (the page shows the "planned" mark).</summary>
    public bool IsPlanned => Feature.State == FeatureState.Planned;

    /// <summary>The mark's text.</summary>
    public string StateText => IsPlanned ? "planned" : "ready";

    /// <summary>What it will need from the phone, one line each; empty when nothing.</summary>
    public IReadOnlyList<string> NeedLines { get; }

    /// <summary>Whether it needs nothing from the phone.</summary>
    public bool NeedsNothing => NeedLines.Count == 0;

    /// <summary>The library's packages it uses, joined; "" when none are named.</summary>
    public string Packages => string.Join(", ", Feature.Packages);

    /// <summary>Whether <see cref="Packages"/> has something to show.</summary>
    public bool HasPackages => Feature.Packages.Count > 0;

    /// <summary>The model families it needs registered, joined; "" when none.</summary>
    public string ModelFamilies => string.Join(", ", Feature.ModelFamilies);

    /// <summary>Whether <see cref="ModelFamilies"/> has something to show.</summary>
    public bool HasModelFamilies => Feature.ModelFamilies.Count > 0;

    /// <summary>A line for each of <paramref name="needs"/>, in the flags' order.</summary>
    public static IReadOnlyList<string> Describe(PhoneNeeds needs)
    {
        var lines = new List<string>(4);
        if ((needs & PhoneNeeds.Microphone) != 0)
        {
            lines.Add("The microphone, asked for at first use");
        }

        if ((needs & PhoneNeeds.Camera) != 0)
        {
            lines.Add("The camera, asked for at first use");
        }

        if ((needs & PhoneNeeds.Network) != 0)
        {
            lines.Add("The network, to pull a model");
        }

        if ((needs & PhoneNeeds.Model) != 0)
        {
            lines.Add("A model from the model store, pulled once and then used offline");
        }

        return lines;
    }
}
