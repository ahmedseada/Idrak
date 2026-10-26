// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.ApplicationModel.DataTransfer;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "transcribe" feature: live transcription from the microphone (the library's <c>LiveTranscriber</c> over
/// the capture this phone has, aaudio on Android) with an optional cleaning pipeline, or an audio file decoded by the
/// library's codecs, read live or whole; the committed words and the tentative ones as they arrive, the level, the time and
/// how far behind the model runs; the text copied or shared. Any registered speech family reads: the library registers
/// none, the app registers the families it shows (<c>SpeechFamilies</c>, a plug-in's own Register call).
/// </summary>
public sealed class TranscribeFeature : IMobileFeature
{
    private const string FeatureName = MobileFeatures.Transcribe;

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "Live transcription from the microphone with an optional cleaning pipeline, or an audio file; the words as they are heard, copied or shared";

    /// <inheritdoc />
    public string Title => "Transcribe";

    /// <inheritdoc />
    public string Glyph => "🎙";

    /// <inheritdoc />
    public int Order => 30;

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Microphone | PhoneNeeds.Model;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Audio"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        return new TranscribeViewModel(
            services.GetService<IModelStore>() ?? new ModelStore(options),
            services.GetService<IModelService>() ?? new ModelService(services.GetService<IMemoryPressureSource>() ?? new ManualMemoryPressure(), options),
            services.GetService<IPermissionService>() ?? new MauiPermissionService(),
            services.GetService<IShareService>() ?? new MauiShareService(),
            options,
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            services.GetService<ProgressCenter>(),
            services.GetService<IClipboard>(),
            services.GetService<IFilePicker>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new TranscribePage(viewModel as TranscribeViewModel ?? throw new ArgumentException($"The transcribe page takes a {nameof(TranscribeViewModel)}.", nameof(viewModel)), services);
}
