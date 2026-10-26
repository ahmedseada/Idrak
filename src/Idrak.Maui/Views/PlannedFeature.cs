// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// A feature not built yet (<see cref="FeatureState.Planned"/>): its name, title, glyph and place in the menu, what it will
/// need from the phone and the library, and what it will do (<see cref="Coming"/>), shown by the library's placeholder page
/// (<see cref="PlaceholderPage"/> over a <see cref="PlaceholderViewModel"/>). Its phase registers the built feature under
/// the same name and order, so the menu does not change. The library's are in <see cref="MobileFeatures"/>; an app makes
/// its own the same way.
/// </summary>
public sealed class PlannedFeature : IMobileFeature
{
    /// <summary>A planned feature.</summary>
    /// <param name="name">Its name in <see cref="MobileFeatures"/> ("chat"), the one its built feature will take.</param>
    /// <param name="title">The name in the menu and the page's title ("Chat").</param>
    /// <param name="glyph">A character or two drawn in the system font beside the title.</param>
    /// <param name="order">Its place in the menu (lower first).</param>
    /// <param name="summary">One line for the menu and listings.</param>
    /// <param name="coming">What it will do once built, one line.</param>
    /// <param name="needs">What it will need from the phone.</param>
    /// <param name="packages">The library's packages it will use.</param>
    /// <param name="modelFamilies">The model families it will need registered; none when null.</param>
    public PlannedFeature(string name, string title, string glyph, int order, string summary, string coming,
        PhoneNeeds needs = PhoneNeeds.None, IReadOnlyList<string>? packages = null, IReadOnlyList<string>? modelFamilies = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        ArgumentNullException.ThrowIfNull(glyph);
        ArgumentException.ThrowIfNullOrWhiteSpace(summary);
        ArgumentException.ThrowIfNullOrWhiteSpace(coming);
        Name = name;
        Title = title;
        Glyph = glyph;
        Order = order;
        Summary = summary;
        Coming = coming;
        Needs = needs;
        Packages = packages ?? [];
        ModelFamilies = modelFamilies ?? [];
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string Summary { get; }

    /// <inheritdoc />
    public string Title { get; }

    /// <inheritdoc />
    public string Glyph { get; }

    /// <inheritdoc />
    public int Order { get; }

    /// <inheritdoc />
    public FeatureState State => FeatureState.Planned;

    /// <inheritdoc />
    public string Coming { get; }

    /// <inheritdoc />
    public PhoneNeeds Needs { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; }

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services) => new PlaceholderViewModel(this);

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new PlaceholderPage(viewModel as PlaceholderViewModel ?? throw new ArgumentException($"The placeholder page takes a {nameof(PlaceholderViewModel)}.", nameof(viewModel)));
}

/// <summary>
/// The library's planned features (plan 34's labs, in its menu order), registered as defaults in <see cref="MobileFeatures"/>.
/// Orders leave room: Devices is 0, the Models lab 10, Chat 20 and Transcribe 30 (built, so no longer here); the Bench (held) and an app's own come after.
/// </summary>
internal static class PlannedFeatures
{
    public static readonly PlannedFeature[] Library =
    [
        new(MobileFeatures.Speak, "Speak", "▷", 40,
            "Text read aloud on the speaker",
            "Synthesis played on the speaker; the voices that need espeak-ng are listed with that reason",
            PhoneNeeds.Model | PhoneNeeds.Network, ["Idrak.Audio", "Idrak.Audio.Plugins"]),
        new(MobileFeatures.AudioProcess, "Audio process", "≈", 50,
            "A recording through a named pipeline",
            "Record, apply a named pipeline, play before and after, with its latency and delay",
            PhoneNeeds.Microphone, ["Idrak.Audio"]),
        new(MobileFeatures.Ocr, "OCR", "Aa", 60,
            "Text read from a photo or the camera",
            "A photo or the camera, decoded by the library's own codecs and read by a small reader; a vision-language reader only where the measured memory allows",
            PhoneNeeds.Camera | PhoneNeeds.Model | PhoneNeeds.Network, ["Idrak.Imaging", "Idrak.Vision", "Idrak.Vision.Plugins"]),
        new(MobileFeatures.Image, "Image", "◐", 70,
            "Transforms and scan profiles on a photo",
            "Named transforms and scan profiles on a photo, with their cold and warm times",
            PhoneNeeds.Camera, ["Idrak.Imaging"]),
        new(MobileFeatures.Search, "Search", "⌕", 80,
            "Retrieval over your own notes",
            "Embeddings of your own notes and retrieval over them, on the phone",
            PhoneNeeds.Model | PhoneNeeds.Network, ["Idrak.Text"]),
    ];

    public static void RegisterDefaults(SlotTable<string, IMobileFeature> table)
    {
        foreach (var feature in Library)
        {
            table.RegisterDefault(feature.Name, feature);
        }
    }
}
