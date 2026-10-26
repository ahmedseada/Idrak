// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.ApplicationModel.DataTransfer;
using Microsoft.Maui.Media;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "ocr" feature: a photo from the camera or the gallery, decoded by the library's own codecs; an optional
/// preprocessing pipeline and the page shown as the model will see it; read by a vision-language model of the store, loaded
/// once after the measured check (refused with the limit named when it does not fit), the answer streamed with its tokens
/// per second, cold and warm apart; the text copied or shared. Any registered vision family reads: the library registers
/// none, the app registers the families it shows (each vision plug-in by its own Register call). Its view model is the
/// app's one (<c>UseIdrak</c> registers it once), so a read goes on while another page is open.
/// </summary>
public sealed class OcrFeature : IMobileFeature
{
    private const string FeatureName = MobileFeatures.Ocr;

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "Text read from a photo or the camera by a vision-language model of the store, streamed with its tokens per second; copied or shared";

    /// <inheritdoc />
    public string Title => "OCR";

    /// <inheritdoc />
    public string Glyph => "Aa";

    /// <inheritdoc />
    public int Order => 60;

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Camera | PhoneNeeds.Model;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Imaging", "Idrak.Text"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        if (services.GetService<OcrViewModel>() is { } shared)
        {
            return shared;
        }

        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var models = services.GetService<IModelService>() ?? new ModelService(services.GetService<IMemoryPressureSource>() ?? new ManualMemoryPressure(), options);
        return new OcrViewModel(
            services.GetService<IModelStore>() ?? new ModelStore(options),
            services.GetService<IVisionReaderService>() ?? new VisionReaderService(models),
            services.GetService<IPermissionService>() ?? new MauiPermissionService(),
            services.GetService<IShareService>() ?? new MauiShareService(),
            options,
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            services.GetService<OcrLabOptions>(),
            clipboard: services.GetService<IClipboard>(),
            picker: services.GetService<IMediaPicker>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new OcrPage(viewModel as OcrViewModel ?? throw new ArgumentException($"The OCR page takes an {nameof(OcrViewModel)}.", nameof(viewModel)), services);
}
