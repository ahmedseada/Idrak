// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "devices" feature (<see cref="MobileFeatures.Devices"/>): the devices found in the app's own process (on a
/// phone, its own Vulkan driver), with their measured choices, CPU features and memory.
/// </summary>
public sealed class DevicesFeature : IMobileFeature
{
    /// <inheritdoc />
    public string Name => MobileFeatures.Devices;

    /// <inheritdoc />
    public string Summary => "The devices found in the app's process: the Vulkan driver with its kernel width, subgroup sizes and memory budget, the CPU's features, memory in use";

    /// <inheritdoc />
    public string Title => "Devices";

    /// <inheritdoc />
    public string Glyph => "▣";

    /// <inheritdoc />
    public int Order => 0;

    /// <inheritdoc />
    public FeatureGroup Group => FeatureGroup.Device;

    /// <inheritdoc />
    public string Tagline => "CPU, GPU, memory";

    /// <inheritdoc />
    public string Icon => "cpu";

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.None;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak", "Idrak.Gpu"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services) =>
        new DevicesViewModel(services.GetService<IDeviceInformation>() ?? new DeviceInformation());

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new DevicesPage(viewModel as DevicesViewModel ?? throw new ArgumentException($"The devices page takes a {nameof(DevicesViewModel)}.", nameof(viewModel)));

    /// <inheritdoc />
    public View? CreateView(IServiceProvider services, object viewModel) =>
        viewModel is DevicesViewModel vm ? new DevicesView(vm, services) : throw new ArgumentException($"The {Name} view takes a {nameof(DevicesViewModel)}.", nameof(viewModel));
}
