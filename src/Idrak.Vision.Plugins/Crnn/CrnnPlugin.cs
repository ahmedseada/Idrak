// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision.Plugins.Crnn;

/// <summary>
/// The CRNN line recognizer (convolutions, a bidirectional LSTM, CTC) as a plug-in: <see cref="CrnnFamily"/> registered in
/// <see cref="ImageModelFamilies"/> under <see cref="Architecture"/>, so a recognizer's model folder loads through
/// <c>ImageModels.Load</c> and reads lines through the library's line-recognition task (<c>model.LineReader()</c>,
/// <c>idrak predict</c>, <c>idrak train --task line-recognition</c>). <see cref="CrnnRecognizer"/> makes, trains, saves
/// and reads one. <see cref="VisionPlugins.RegisterAll"/> registers it with the package's other families (the idrak tool
/// does at startup); an app may call <see cref="Register"/> alone.
/// </summary>
public static class CrnnPlugin
{
    /// <summary>The architecture name a recognizer's config.json gives: "CrnnForLineRecognition".</summary>
    public const string Architecture = "CrnnForLineRecognition";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("crnn"; listed in <see cref="VisionPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new CrnnRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="ImageModelFamilies"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class CrnnRegistration() : IdrakPlugin("crnn")
    {
        protected override void RegisterParts() => Part(() => ImageModelFamilies.Register(new CrnnFamily()), () => ImageModelFamilies.Unregister(Architecture));
    }
}
