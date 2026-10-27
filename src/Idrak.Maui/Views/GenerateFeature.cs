// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "generate" feature: text drawn as pages by the library's synthetic pages (what <c>idrak data synth</c>
/// does) in the phone's fonts or added ones, made to look scanned by effects that stack; a large preview with zoom and
/// before/after, the pages as a grid, saved to Pictures, shared, or written as a data set. Needs Idrak.Text and
/// Idrak.Imaging only (no model).
/// </summary>
public sealed class GenerateFeature : IMobileFeature
{
    /// <inheritdoc />
    public string Name => MobileFeatures.Generate;

    /// <inheritdoc />
    public string Summary => "Text drawn as pages in the phone's fonts or yours, made to look scanned by effects that stack; previewed, saved, shared or written as a data set";

    /// <inheritdoc />
    public string Title => "Generate";

    /// <inheritdoc />
    public string Glyph => "✎";

    /// <inheritdoc />
    public int Order => 75;

    /// <inheritdoc />
    public FeatureGroup Group => FeatureGroup.Vision;

    /// <inheritdoc />
    public string Tagline => "Text to scanned pages";

    /// <inheritdoc />
    public string Icon => "pen";

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.None;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Text", "Idrak.Imaging"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        string storage = services.GetService<IdrakStartupReport>()?.AppStorage ?? ComputeResources.CacheFolder;
        return new GenerateViewModel(
            services.GetService<IShareService>() ?? new MauiShareService(),
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            storage,
            services.GetService<IPictureSaver>(),
            services.GetService<ProgressCenter>(),
            services.GetService<GenerateLabOptions>(),
            services.GetService<IFilePicker>(),
            services.GetService<ISystemFonts>() ?? new PlatformSystemFonts());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new GeneratePage(viewModel as GenerateViewModel ?? throw new ArgumentException($"The generate page takes a {nameof(GenerateViewModel)}.", nameof(viewModel)), services);

    /// <inheritdoc />
    public View? CreateView(IServiceProvider services, object viewModel) =>
        viewModel is GenerateViewModel vm ? new GenerateView(vm, services) : throw new ArgumentException($"The {Name} view takes a {nameof(GenerateViewModel)}.", nameof(viewModel));
}
