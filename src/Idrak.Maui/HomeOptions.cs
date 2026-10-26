// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// One slide of Home's carousel: a kicker, a title whose end is drawn in the accent, a line of text, and a call to action
/// that opens a section or a feature. "{device}" in its texts reads "phone", "tablet" or "device", as the app runs on.
/// </summary>
/// <param name="Kicker">The small capitals above the title ("What is Idrak").</param>
/// <param name="Title">The title's start ("AI that runs ").</param>
/// <param name="Emphasis">The title's end, in the accent ("on this {device}.").</param>
/// <param name="Text">One or two sentences.</param>
/// <param name="Action">The call to action's label ("Start a chat").</param>
/// <param name="Target">What the action opens: a section (<see cref="AppSections"/>) or a feature (<see cref="MobileFeatures"/>) by name.</param>
/// <param name="Caption">The slide's name under the carousel ("The library").</param>
/// <param name="Tone">Its ground: "Sky", "Amber", "Green" or "Violet" (the theme's IdrakBrushHero{Tone}).</param>
/// <param name="Art">Its art: "logo" for Idrak's mark, else the name of one of Idrak's icons.</param>
public sealed record HomeSlide(string Kicker, string Title, string Emphasis, string Text, string Action, string Target, string Caption, string Tone, string Art);

/// <summary>What Home says (<see cref="IdrakMauiOptions.Home"/>): the carousel's slides and the samples its tiles show before the labs have run.</summary>
public sealed class HomeOptions
{
    /// <summary>The library's four slides: what Idrak is, the device measured, privacy, and the plug-in.</summary>
    public static IReadOnlyList<HomeSlide> LibrarySlides { get; } =
    [
        new("What is Idrak", "AI that runs ", "on this {device}.", "Language, speech and vision models in pure .NET, on the CPU and this {device}'s own GPU.",
            "Start a chat", AppSections.Chat, "The library", "Sky", "logo"),
        new("Measured, not assumed", "Its own GPU and CPU, ", "measured here.",
            "The Vulkan driver loads in the app's own process; the kernels, the vectors and the memory are measured on this {device}, never looked up.",
            "See the devices", MobileFeatures.Devices, "The device", "Amber", "cpu"),
        new("Private by design", "Nothing leaves ", "this {device}.",
            "Models are pulled once; speech, photos and chats are read here, with no server in between.",
            "Try transcribing", MobileFeatures.Transcribe, "Privacy", "Green", "lock"),
        new("For any MAUI app", "One call: ", "UseIdrak.",
            "Idrak.Maui is a plug-in: an app adds the labs it wants, its own pages and the model families it shows.",
            "All labs", AppSections.Labs, "The plug-in", "Violet", "plug"),
    ];

    /// <summary>The carousel's slides, in order.</summary>
    public IReadOnlyList<HomeSlide> Slides { get; set; } = LibrarySlides;

    /// <summary>The question the chat tile shows before the first chat.</summary>
    public string ChatSampleQuestion { get; set; } = "Summarise this in Arabic";

    /// <summary>The answer the chat tile shows before the first chat.</summary>
    public string ChatSampleAnswer { get; set; } = "بالتأكيد! إليك الملخص…";

    /// <summary>The line the Speak tile shows before the first speech.</summary>
    public string SpeakSample { get; set; } = "“مرحبا بكم في إدراك”";
}
