// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.IdrakAudio;

/// <summary>Registers the IdrakAudio speech family in <see cref="SpeechFamilies"/> under <see cref="IdrakAudioFamily.Architecture"/>.</summary>
public static class IdrakAudioPlugin
{
    /// <summary>Registers the family.</summary>
    public static void Register() => SpeechFamilies.Register(new IdrakAudioFamily());

    /// <summary>The entry the command line calls for <c>-P Idrak.IdrakAudio.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registration <see cref="Register"/> made.</summary>
    public static void Unregister() => SpeechFamilies.Unregister(IdrakAudioFamily.Architecture);
}
