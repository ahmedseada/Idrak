// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Audio.Plugins.EspeakNg;

/// <summary>
/// The espeak-ng program as a phonemizer: <see cref="EspeakNgPhonemizer"/> registered in <see cref="TextPhonemizers"/>
/// as <see cref="Phonemizer"/>. The library starts no outside program, so this plug-in is the only place one runs; the
/// program is the user's (<see cref="EspeakNgPhonemizer.ExecutableVariable"/>, else <c>espeak-ng</c> on the PATH). A
/// family whose checkpoints read espeak's phonemes registers it (the VITS family does).
/// </summary>
public static class EspeakNgPlugin
{
    /// <summary>The name the phonemizer is registered under in <see cref="TextPhonemizers"/>: "espeak-ng".</summary>
    public const string Phonemizer = "espeak-ng";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("espeak-ng"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new EspeakNgRegistration();

    /// <summary>Whether the phonemizer is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the phonemizer (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the phonemizer, leaving <see cref="TextPhonemizers"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class EspeakNgRegistration() : IdrakPlugin(Phonemizer)
    {
        protected override void RegisterParts() =>
            Part(() => TextPhonemizers.Register(new EspeakNgPhonemizer()), () => TextPhonemizers.Unregister(Phonemizer));
    }
}
