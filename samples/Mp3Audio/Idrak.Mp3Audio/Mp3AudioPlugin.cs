// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Mp3Audio;

/// <summary>
/// Registers <see cref="Mp3Codec"/> in <see cref="AudioCodecs"/> under "mp3" (.mp3, .mp2) and <see cref="Mp3Encoder"/>
/// in <see cref="AudioEncoders"/> under "mp3" (.mp3), so <c>AudioEncoders.Save("x.mp3", audio)</c> writes MP3.
/// </summary>
public static class Mp3AudioPlugin
{
    /// <summary>
    /// Registers the codec (with <paramref name="options"/>, the defaults when null) and the encoder (with
    /// <paramref name="encodeOptions"/>; when null, <see cref="Mp3EncodeOptions.FromEnvironment"/>: the IDRAK_MP3_*
    /// variables, the defaults where unset).
    /// </summary>
    public static void Register(Mp3Options? options = null, Mp3EncodeOptions? encodeOptions = null)
    {
        AudioCodecs.Register(new Mp3Codec(options));
        AudioEncoders.Register(new Mp3Encoder(encodeOptions ?? Mp3EncodeOptions.FromEnvironment()));
    }

    /// <summary>The entry the command line calls for <c>-P Idrak.Mp3Audio.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registrations <see cref="Register"/> made; false when there were none.</summary>
    public static bool Unregister() => AudioCodecs.Unregister("mp3") | AudioEncoders.Unregister("mp3");
}
