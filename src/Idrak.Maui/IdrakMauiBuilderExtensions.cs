// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Idrak.Maui;

/// <summary><c>UseIdrak</c>: the library inside a MAUI app.</summary>
public static class IdrakMauiBuilderExtensions
{
    /// <summary>
    /// Sets the library up for the app, before the first device or download: the options (<paramref name="configure"/> gives
    /// the app's defaults; what the user saved in the settings page is applied over them), then the cache folder (the
    /// app's own storage by default), every <see cref="ComputeResources"/> limit, the device and the token
    /// (<see cref="IdrakMauiStartup.Apply"/>); the "maui" progress reporter; the services (<see cref="IModelService"/>,
    /// <see cref="IModelStore"/>, <see cref="IDeviceInformation"/>, <see cref="IPermissionService"/>,
    /// <see cref="IShareService"/>, <see cref="ISoundPlayer"/>, the settings and the memory-pressure signal); the library's features
    /// (<see cref="MobileFeatures"/>: Devices, Models, Chat with <see cref="IChatModelService"/>, Transcribe, Speak, Audio
    /// process, OCR with <see cref="IVisionReaderService"/>, and Image) and the settings page. The app registers its own
    /// features and model families after; it gives its own defaults for the OCR and Image labs' settings by registering an
    /// <see cref="OcrLabOptions"/> or <see cref="ImageLabOptions"/> (what the user saves is applied over them).
    /// </summary>
    public static MauiAppBuilder UseIdrak(this MauiAppBuilder builder, Action<IdrakMauiOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(builder);
        var options = new IdrakMauiOptions();
        configure?.Invoke(options);
        var store = options.SettingsStore ?? new MauiPreferencesStore();
        var secrets = options.SecretStore ?? new MauiSecureStore();
        var settings = new IdrakMauiSettings(store, secrets);
        var skipped = options.KeepSettings
            ? Task.Run(() => settings.LoadIntoAsync(options)).GetAwaiter().GetResult()       // off the UI thread: secure storage may post to it
            : [];
        var startup = IdrakMauiStartup.Apply(options, FileSystem.AppDataDirectory);
        if (skipped.Count > 0)
        {
            startup = startup with { Problems = [.. startup.Problems, .. skipped.Select(s => $"A saved setting was not read: {s}")] };
        }

        var progress = new ProgressCenter(options.ProgressInterval);
        ProgressReporters.Register(new MauiProgressReporterKind(progress));
        if (options.UseMauiProgress && string.IsNullOrEmpty(Environment.GetEnvironmentVariable(ProgressReporters.EnvironmentVariable)))
        {
            Environment.SetEnvironmentVariable(ProgressReporters.EnvironmentVariable, MauiProgressReporterKind.ReporterName);
        }

        _ = MobileFeatures.Names;                                                   // the library's features, registered first

        var services = builder.Services;
        services.AddSingleton(options);
        services.AddSingleton(startup);
        services.AddSingleton(settings);
        services.AddSingleton(store);
        services.AddSingleton(secrets);
        services.AddSingleton(progress);
#if ANDROID
        services.TryAddSingleton<IMemoryPressureSource, Idrak.Maui.Platforms.Android.AndroidMemoryPressure>();
#else
        services.TryAddSingleton<IMemoryPressureSource, ManualMemoryPressure>();
#endif
        services.TryAddSingleton<IModelService>(sp => new ModelService(sp.GetRequiredService<IMemoryPressureSource>(), options));
        services.TryAddSingleton<IModelStore>(_ => new ModelStore(options));
        services.TryAddSingleton<IChatModelService>(sp => new ChatModelService(sp.GetRequiredService<IModelService>()));
        services.TryAddSingleton<IVisionReaderService>(sp => new VisionReaderService(sp.GetRequiredService<IModelService>()));
        services.TryAddSingleton<IDeviceInformation, DeviceInformation>();
        services.TryAddSingleton<IPermissionService, MauiPermissionService>();
        services.TryAddSingleton<IShareService, MauiShareService>();
        services.TryAddSingleton<ISoundPlayer, SoundPlayer>();                    // one speaker for the app: one sound at a time
        services.AddTransient<DevicesViewModel>();
        services.TryAddSingleton<ModelsViewModel>();                                // one for the app: a pull outlives its page
        services.TryAddSingleton(sp => new ChatViewModel(sp.GetRequiredService<IModelStore>(), sp.GetRequiredService<IChatModelService>(), options));   // a chat outlives its page
        services.TryAddSingleton(sp => new OcrViewModel(sp.GetRequiredService<IModelStore>(), sp.GetRequiredService<IVisionReaderService>(),   // a read outlives its page
            sp.GetRequiredService<IPermissionService>(), sp.GetRequiredService<IShareService>(), options, store, sp.GetService<OcrLabOptions>()));
        services.AddTransient(sp => new SettingsViewModel(options, settings, startup, progress));
        services.AddTransient<SettingsPage>();
        services.TryAddEnumerable(ServiceDescriptor.Transient<IMauiInitializeService, IdrakInitializer>());
        return builder;
    }
}

// When the app is built: the memory-pressure signal hooked to the model service from the start, so the phone's first
// signal reaches it even before any model is loaded.
internal sealed class IdrakInitializer : IMauiInitializeService
{
    public void Initialize(IServiceProvider services) => _ = services.GetRequiredService<IModelService>();
}
