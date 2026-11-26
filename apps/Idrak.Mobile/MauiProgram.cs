// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Plugins.IdrakAudio;
using Idrak.Audio.Plugins.RnNoise;
using Idrak.Audio.Plugins.Vits;
using Idrak.Audio.Plugins.Whisper;
using Idrak.Maui;
using Idrak.Maui.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision.Plugins.Llava;
using Idrak.Vision.Plugins.Qwen;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Mobile;

/// <summary>
/// The app's start: its fonts, the library first (<c>UseIdrak</c>, before any device or download: the features, the
/// navigation host with the bottom bar or the rail, the theme's services), then the model families it shows and its own
/// Self-test feature.
/// </summary>
public static class MauiProgram
{
    /// <summary>Builds the app.</summary>
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder.UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
                // The design's fonts (Resources/Fonts, SIL OFL 1.1), under the aliases Idrak.Maui's styles name.
                fonts.AddFont("Inter-Regular.otf", "Inter");
                fonts.AddFont("Inter-Medium.otf", "InterMedium");
                fonts.AddFont("Inter-SemiBold.otf", "InterSemiBold");
                fonts.AddFont("Inter-Bold.otf", "InterBold");
                fonts.AddFont("Inter-ExtraBold.otf", "InterExtraBold");
                fonts.AddFont("JetBrainsMono-Regular.ttf", "JetBrainsMono");
                fonts.AddFont("JetBrainsMono-Bold.ttf", "JetBrainsMonoBold");
                fonts.AddFont("IBMPlexSansArabic-SemiBold.ttf", "IbmPlexArabicSemiBold");
            })
            .UseIdrak(options =>
            {
                // The app's defaults; what the user saves in the settings page is applied over them.
                options.Theme = IdrakThemeChoice.System;

                // The animated splash over the host page while the app starts (the library's is off unless an app asks: it
                // is Idrak's own mark and name); the settings' splash length chooses how much of it plays.
                options.ShowSplash = true;

                // The small models the Models page offers to tap: the app's own list (the library suggests none). The last
                // is the smallest Qwen-VL, a vision-language model the OCR lab reads with (the lab measures whether it fits
                // this phone before loading it, and refuses it with the limit named when it does not).
                options.SuggestedModels = ["Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", "Qwen/Qwen2-VL-2B-Instruct"];
            });

        // The model families the app shows (the Chat lab's Qwen, Llama, Gemma and Mistral, the Transcribe lab's Whisper and
        // RNNoise, the Speak lab's VITS and IdrakAudio, the OCR lab's Gemma 3 vision, Qwen2-VL and LLaVA) were registered
        // here, each plug-in by its own Register call: picking a family by its name is not on the new design yet.

        builder.Services.AddSingleton(new OcrLabOptions());

        // The Terminal page runs the idrak commands in the app's own process (Idrak.Cli.Commands): one runner for the app (one
        // command at a time), allowing what the Terminal's setting allows (other programs and a network listener on a desktop
        // by default, nothing on a phone), read at each command; never console input (the page is the shell).
        builder.Services.AddSingleton<ICommandRunner>(sp =>
        {
            var options = sp.GetRequiredService<IdrakMauiOptions>();
            return new IdrakCommandRunner(allows: () => options.TerminalAllowInEffect);
        });

        // The app's own features (Self-test, AppFeatures) were registered after the library's: registering a feature is not
        // on the new design yet.
        return builder.Build();
    }
}
