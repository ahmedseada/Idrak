// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "speak" feature: text read aloud on the speaker by a synthesis model of the store (any registered
/// synthesis family; the library registers none, the app registers the ones it shows), played as the first sentence is
/// ready, with the time to the first sound and the real-time factor, cold and warm apart; the voices of the model and the
/// voice store; models and voices that need what is not registered here (the espeak-ng phonemizer, an outside program)
/// listed with that reason; the speech saved as WAV or MP3 and shared.
/// </summary>
public sealed class SpeakFeature : IMobileFeature
{
    private const string FeatureName = MobileFeatures.Speak;

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "Text read aloud on the speaker as each sentence is ready, in the model's voices or an installed one; the speech saved and shared";

    /// <inheritdoc />
    public string Title => "Speak";

    /// <inheritdoc />
    public string Glyph => "▷";

    /// <inheritdoc />
    public int Order => 40;

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Model;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Audio"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        return new SpeakViewModel(
            services.GetService<IModelStore>() ?? new ModelStore(options),
            services.GetService<IModelService>() ?? new ModelService(services.GetService<IMemoryPressureSource>() ?? new ManualMemoryPressure(), options),
            services.GetService<ISoundPlayer>() ?? new SoundPlayer(),
            services.GetService<IShareService>() ?? new MauiShareService(),
            options,
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            services.GetService<ProgressCenter>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new SpeakPage(viewModel as SpeakViewModel ?? throw new ArgumentException($"The speak page takes a {nameof(SpeakViewModel)}.", nameof(viewModel)), services);
}
