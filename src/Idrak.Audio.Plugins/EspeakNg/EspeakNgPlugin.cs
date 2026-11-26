// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Plugins.EspeakNg;

/// <summary>
/// The espeak-ng program as a phonemizer: <see cref="EspeakNgPhonemizer"/>, named <see cref="Phonemizer"/>. The library
/// starts no outside program, so this plug-in is the only place one runs; the program is the user's
/// (<see cref="EspeakNgPhonemizer.ExecutableVariable"/>, else <c>espeak-ng</c> on the PATH).
/// </summary>
public static class EspeakNgPlugin
{
    /// <summary>The phonemizer's name: "espeak-ng".</summary>
    public const string Phonemizer = "espeak-ng";
}
