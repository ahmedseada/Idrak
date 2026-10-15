// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.IdrakAudio;

/// <summary>
/// Registers the IdrakAudio families: the speech family in <see cref="SpeechFamilies"/> under
/// <see cref="IdrakAudioFamily.Architecture"/> (the "ctc" and "seq2seq" recognizers), the synthesis family in
/// <see cref="SpeechSynthesisFamilies"/> under the same name (the "tts" synthesizers), and the audio-language family under
/// <see cref="IdrakAudioLanguageFamily.Architecture"/> (its text decoder in <c>PretrainedArchitectures</c>, its audio part
/// in <see cref="AudioFamilies"/>).
/// </summary>
public static class IdrakAudioPlugin
{
    /// <summary>Registers the families.</summary>
    public static void Register()
    {
        SpeechFamilies.Register(new IdrakAudioFamily());
        SpeechSynthesisFamilies.Register(new IdrakAudioSynthesisFamily());
        PretrainedArchitectures.Register(IdrakAudioLanguageFamily.Architecture, IdrakAudioLanguageFamily.TextDecoder());
        AudioFamilies.Register(new IdrakAudioLanguageFamily());
    }

    /// <summary>The entry the command line calls for <c>-P Idrak.IdrakAudio.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registrations <see cref="Register"/> made.</summary>
    public static void Unregister()
    {
        SpeechFamilies.Unregister(IdrakAudioFamily.Architecture);
        SpeechSynthesisFamilies.Unregister(IdrakAudioFamily.Architecture);
        PretrainedArchitectures.Unregister(IdrakAudioLanguageFamily.Architecture);
        AudioFamilies.Unregister(IdrakAudioLanguageFamily.Architecture);
    }
}
