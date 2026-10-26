// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Media;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "image" feature: a photo from the camera or the gallery, decoded by the library's own codecs and turned
/// upright; the registered image transforms as chips that build a pipeline, the scan profiles to draw steps from; the
/// pipeline applied with each step's time, cold and warm apart; the original and the result shown; the result saved and
/// shared as PNG or JPEG. Needs the library's Idrak.Imaging only (no model).
/// </summary>
public sealed class ImageFeature : IMobileFeature
{
    private const string FeatureName = MobileFeatures.Image;

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "Named transforms and scan profiles on a photo, decoded by the library's own codecs: each step's time, cold and warm apart; the result saved or shared";

    /// <inheritdoc />
    public string Title => "Image";

    /// <inheritdoc />
    public string Glyph => "◐";

    /// <inheritdoc />
    public int Order => 70;

    /// <inheritdoc />
    public FeatureGroup Group => FeatureGroup.Vision;

    /// <inheritdoc />
    public string Tagline => "Transforms, scans";

    /// <inheritdoc />
    public string Icon => "image";

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Camera;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Imaging"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        return new ImageViewModel(
            services.GetService<IPermissionService>() ?? new MauiPermissionService(),
            services.GetService<IShareService>() ?? new MauiShareService(),
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            services.GetService<ImageLabOptions>(),
            services.GetService<IMediaPicker>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new ImagePage(viewModel as ImageViewModel ?? throw new ArgumentException($"The image page takes an {nameof(ImageViewModel)}.", nameof(viewModel)));

    /// <inheritdoc />
    public View? CreateView(IServiceProvider services, object viewModel) =>
        viewModel is ImageViewModel vm ? new ImageView(vm, services) : throw new ArgumentException($"The {Name} view takes a {nameof(ImageViewModel)}.", nameof(viewModel));
}
