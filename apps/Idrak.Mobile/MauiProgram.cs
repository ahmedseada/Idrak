// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Mobile;

/// <summary>The app's start: the library first (<c>UseIdrak</c>, before any device or download), then its pages.</summary>
public static class MauiProgram
{
    /// <summary>Builds the app.</summary>
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder.UseMauiApp<App>()
            .UseIdrak(options =>
            {
                // The app's defaults; what the user saves in the settings page is applied over them.
                options.Theme = IdrakThemeChoice.System;

                // The small models the Models page offers to tap: the app's own list (the library suggests none).
                options.SuggestedModels = ["Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"];
            });

        // The model families the app shows are registered here, each plug-in by its own Register call, as their labs
        // arrive (plan 34, M5 on: the chat, speech and reading labs). The labs so far (Devices, Models, Settings) need
        // none; the espeak-ng phonemizer is never registered (it runs a program, which an app cannot).

        builder.Services.AddSingleton<AppFlyoutPage>();
        builder.Services.AddSingleton<MenuPage>();

        // The app's own features, after the library's (UseIdrak registered those).
        AppFeatures.Register();
        return builder.Build();
    }
}
