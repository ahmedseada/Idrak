// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "audio-process" feature: a recording from the microphone or a sound file decoded by the library's
/// codecs, run through a pipeline written as the library's text (the registered processors as chips), played before and
/// after on the speaker, with the pipeline's latency and delay, the time it took cold and warm and the levels before and
/// after; the result saved and shared; optional live monitoring through the pipeline's live steps.
/// </summary>
public sealed class AudioProcessFeature : IMobileFeature
{
    private const string FeatureName = MobileFeatures.AudioProcess;

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "A recording or a file through a pipeline of the library's processors, heard before and after, with its latency, delay and time";

    /// <inheritdoc />
    public string Title => "Studio";

    /// <inheritdoc />
    public string Glyph => "≈";

    /// <inheritdoc />
    public int Order => 50;

    /// <inheritdoc />
    public FeatureGroup Group => FeatureGroup.Voice;

    /// <inheritdoc />
    public string Tagline => "Clean a recording";

    /// <inheritdoc />
    public string Icon => "wave";

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Microphone;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Audio"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        return new AudioProcessViewModel(
            services.GetService<ISoundPlayer>() ?? new SoundPlayer(),
            services.GetService<IPermissionService>() ?? new MauiPermissionService(),
            services.GetService<IShareService>() ?? new MauiShareService(),
            options,
            services.GetService<ISettingsStore>() ?? new MemorySettingsStore(),
            services.GetService<ProgressCenter>(),
            services.GetService<IFilePicker>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new AudioProcessPage(viewModel as AudioProcessViewModel ?? throw new ArgumentException($"The audio process page takes a {nameof(AudioProcessViewModel)}.", nameof(viewModel)));

    /// <inheritdoc />
    public View? CreateView(IServiceProvider services, object viewModel) =>
        viewModel is AudioProcessViewModel vm ? new StudioView(vm, services) : throw new ArgumentException($"The {Name} view takes a {nameof(AudioProcessViewModel)}.", nameof(viewModel));
}
