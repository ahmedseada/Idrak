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
            })
            .UseIdrak(options =>
            {
                // The app's defaults; what the user saves in the settings page is applied over them.
                options.Theme = IdrakThemeChoice.System;

                // The small models the Models page offers to tap: the app's own list (the library suggests none). The last
                // is the smallest Qwen-VL, a vision-language model the OCR lab reads with (the lab measures whether it fits
                // this phone before loading it, and refuses it with the limit named when it does not).
                options.SuggestedModels = ["Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", "Qwen/Qwen2-VL-2B-Instruct"];

                // The chat's weights as int8 by default (a quarter of float32's memory); the settings page changes it.
                options.ChatWeights = "int8";
            });

        // The model families the app shows are registered here, each plug-in by its own Register call (the library
        // registers none: a model of another family is refused as not registered). The espeak-ng phonemizer is never
        // registered (it runs a program, which an app cannot).
        // The Chat lab's text families: Qwen (Qwen2, Qwen3 and their mixtures of experts), Llama, Gemma and Mistral.
        QwenPlugin.Register();
        LlamaPlugin.Register();
        GemmaPlugin.Register();
        MistralPlugin.Register();
        // The Transcribe lab: Whisper reads the speech models, and RNNoise is a live cleaning step for its pipeline
        // setting (rnnoise model=PATH, its model file kept on the phone).
        WhisperPlugin.Register();
        RnNoisePlugin.Register();
        // The Speak lab's synthesis families: VITS (the MMS-TTS checkpoints, with uroman's Arabic romanizer) by its family
        // alone, without the espeak-ng phonemizer VitsPlugin.Register would bring (a program the app cannot run: a
        // checkpoint that reads phonemes is listed with that reason); IdrakAudio (the library's own speech models).
        VitsPlugin.RegisterFamily();
        IdrakAudioPlugin.Register();
        // The OCR lab's vision-language families (Idrak.Vision.Plugins), each with its text decoder: Gemma 3 vision,
        // Qwen2-VL and Qwen2.5-VL, and LLaVA.
        Gemma3VisionPlugin.Register();
        QwenVisionPlugin.Register();
        LlavaPlugin.Register();

        // The OCR lab reads with int8 weights by default (a quarter of float32's memory), as the chat does; the lab's
        // settings change it, and what the user keeps there is applied over this.
        builder.Services.AddSingleton(new OcrLabOptions { Weights = "int8" });

        // The Terminal page runs the idrak commands in the app's own process (Idrak.Cli.Commands): one runner for the app (one
        // command at a time), allowing what the Terminal's setting allows (other programs and a network listener on a desktop
        // by default, nothing on a phone), read at each command; never console input (the page is the shell).
        builder.Services.AddSingleton<ICommandRunner>(sp =>
        {
            var options = sp.GetRequiredService<IdrakMauiOptions>();
            return new IdrakCommandRunner(allows: () => options.TerminalAllowInEffect);
        });

        // The app's own features, after the library's (UseIdrak registered those).
        AppFeatures.Register();
        return builder.Build();
    }
}
