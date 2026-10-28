// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui;

/// <summary>The library's start-up steps, registered as defaults in <see cref="StartupSteps"/>.</summary>
internal static class LibraryStartupSteps
{
    public static IEnumerable<IStartupStep> All() => [new CpuStartupStep(), new GpuStartupStep(), new SectionsStartupStep()];
}

/// <summary>The CPU described (<see cref="IDeviceInformation.Read"/> of "cpu"): its facts and memory, read once for the app.</summary>
public sealed class CpuStartupStep : IStartupStep
{
    /// <inheritdoc />
    public string Name => StartupSteps.Cpu;

    /// <inheritdoc />
    public string Summary => "The CPU described: its instruction sets, caches and memory";

    /// <inheritdoc />
    public string Title => "Reading the CPU";

    /// <inheritdoc />
    public int Order => 10;

    /// <inheritdoc />
    public StartupThread Thread => StartupThread.Worker;

    /// <inheritdoc />
    public bool IsWanted(StartupContext context) => context?.Services.GetService<IDeviceInformation>() is not null;

    /// <inheritdoc />
    public Task RunAsync(StartupContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.Services.GetRequiredService<IDeviceInformation>().Read([Device.Cpu.ToString()]);
        return Task.CompletedTask;
    }
}

/// <summary>
/// Every GPU backend started and its devices listed, with Home's summary of the device and the store's models
/// (<see cref="DeviceStatus.EnsureReadAsync"/>, the reading Home shares), so Home shows them as it appears.
/// </summary>
public sealed class GpuStartupStep : IStartupStep
{
    /// <inheritdoc />
    public string Name => StartupSteps.Gpu;

    /// <inheritdoc />
    public string Summary => "Every GPU backend started and listed, with Home's device summary and the store's models";

    /// <inheritdoc />
    public string Title => "Starting the GPU";

    /// <inheritdoc />
    public int Order => 20;

    /// <inheritdoc />
    public StartupThread Thread => StartupThread.Worker;

    /// <inheritdoc />
    public bool IsWanted(StartupContext context) => context?.Services.GetService<DeviceStatus>() is not null;

    /// <inheritdoc />
    public Task RunAsync(StartupContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        return context.Services.GetRequiredService<DeviceStatus>().EnsureReadAsync();
    }
}

/// <summary>
/// The tabs' views made ahead (<see cref="AppHostPage.PrebuildAsync"/>), one a turn of the UI thread, when
/// <see cref="IdrakMauiOptions.PrebuildSections"/> is on: a tab's first tap only shows it.
/// </summary>
public sealed class SectionsStartupStep : IStartupStep
{
    /// <inheritdoc />
    public string Name => StartupSteps.Sections;

    /// <inheritdoc />
    public string Summary => "The tabs' views made ahead, one a turn of the UI thread, so a first tap only shows a tab";

    /// <inheritdoc />
    public string Title => "Making the tabs";

    /// <inheritdoc />
    public int Order => 30;

    /// <inheritdoc />
    public StartupThread Thread => StartupThread.Ui;

    /// <inheritdoc />
    public bool IsWanted(StartupContext context) =>
        context?.Host is AppHostPage host && !host.IsPrebuilt && (context.Services.GetService<IdrakMauiOptions>()?.PrebuildSections ?? true);

    /// <inheritdoc />
    public Task RunAsync(StartupContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        return context.Host is AppHostPage host ? host.PrebuildAsync() : Task.CompletedTask;
    }
}
