// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Mp3Audio;

/// <summary>Registers <see cref="Mp3Codec"/> in <see cref="AudioCodecs"/> under "mp3" (.mp3, .mp2).</summary>
public static class Mp3AudioPlugin
{
    /// <summary>Registers the codec (with <paramref name="options"/>, the defaults when null).</summary>
    public static void Register(Mp3Options? options = null) => AudioCodecs.Register(new Mp3Codec(options));

    /// <summary>The entry the command line calls for <c>-P Idrak.Mp3Audio.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registration <see cref="Register"/> made.</summary>
    public static bool Unregister() => AudioCodecs.Unregister("mp3");
}
