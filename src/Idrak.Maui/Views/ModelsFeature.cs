// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "models" feature (<see cref="MobileFeatures.Models"/>): pull a Hugging Face model into the app's storage
/// with its progress, list the models stored with their sizes and folders, remove one, and the storage's free space. Its
/// view model is the app's one (<c>UseIdrak</c> registers it once), so a pull goes on while another page is open and shows
/// again when this one comes back.
/// </summary>
public sealed class ModelsFeature : IMobileFeature
{
    /// <inheritdoc />
    public string Name => MobileFeatures.Models;

    /// <inheritdoc />
    public string Summary => "Pull a Hugging Face model into the app's storage with its progress; the models stored, their sizes and the free space";

    /// <inheritdoc />
    public string Title => "Models";

    /// <inheritdoc />
    public string Glyph => "⇩";

    /// <inheritdoc />
    public int Order => 10;

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Network;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak.Data"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        if (services.GetService<ModelsViewModel>() is { } shared)
        {
            return shared;
        }

        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        return new ModelsViewModel(services.GetService<IModelStore>() ?? new ModelStore(options), options, services.GetService<ProgressCenter>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new ModelsPage(viewModel as ModelsViewModel ?? throw new ArgumentException($"The models page takes a {nameof(ModelsViewModel)}.", nameof(viewModel)));
}
