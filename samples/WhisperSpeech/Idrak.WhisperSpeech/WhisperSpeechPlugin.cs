// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.WhisperSpeech;

/// <summary>
/// Registers the Whisper-style family in <see cref="SpeechFamilies"/> under <see cref="WhisperFamily.Architecture"/>, so
/// <c>SpeechModels.Load</c> reads its checkpoints.
/// </summary>
public static class WhisperSpeechPlugin
{
    /// <summary>Registers the family.</summary>
    public static void Register() => SpeechFamilies.Register(new WhisperFamily());

    /// <summary>The entry the command line calls for <c>-P Idrak.WhisperSpeech.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registration <see cref="Register"/> made.</summary>
    public static void Unregister() => SpeechFamilies.Unregister(WhisperFamily.Architecture);
}
