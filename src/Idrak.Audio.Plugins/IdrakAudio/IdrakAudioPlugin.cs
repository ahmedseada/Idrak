// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Audio.Plugins.IdrakAudio;

/// <summary>
/// The IdrakAudio families, made from a spec, trained by the library and read back through its contracts: the speech
/// family in <see cref="SpeechFamilies"/> under <see cref="IdrakAudioFamily.Architecture"/> (the "ctc" and "seq2seq"
/// recognizers), the synthesis family in <see cref="SpeechSynthesisFamilies"/> under the same name (the "tts"
/// synthesizers), and the audio-language family under <see cref="IdrakAudioLanguageFamily.Architecture"/> (its text
/// decoder in <c>PretrainedArchitectures</c>, its audio part in <see cref="AudioFamilies"/>).
/// </summary>
public static class IdrakAudioPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("idrak-audio"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new IdrakAudioRegistration();

    /// <summary>Whether the families are registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the families (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the families, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The four registrations, each a part.
    private sealed class IdrakAudioRegistration() : IdrakPlugin("idrak-audio")
    {
        protected override void RegisterParts()
        {
            Part(() => SpeechFamilies.Register(new IdrakAudioFamily()), () => SpeechFamilies.Unregister(IdrakAudioFamily.Architecture));
            Part(() => SpeechSynthesisFamilies.Register(new IdrakAudioSynthesisFamily()), () => SpeechSynthesisFamilies.Unregister(IdrakAudioFamily.Architecture));
            Part(() => PretrainedArchitectures.Register(IdrakAudioLanguageFamily.Architecture, IdrakAudioLanguageFamily.TextDecoder()),
                () => PretrainedArchitectures.Unregister(IdrakAudioLanguageFamily.Architecture));
            Part(() => AudioFamilies.Register(new IdrakAudioLanguageFamily()), () => AudioFamilies.Unregister(IdrakAudioLanguageFamily.Architecture));
        }
    }
}
