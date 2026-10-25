// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

[assembly: IdrakPluginPackage(typeof(Idrak.Audio.Plugins.AudioPluginsPackage))]

namespace Idrak.Audio.Plugins;

/// <summary>
/// The speech and voice model families (recognition, synthesis, voice conversion, phonemizers) and the learned audio
/// processors (RNNoise) of Idrak.Audio.Plugins.
/// Each family is a plug-in in a namespace of its own (<c>Idrak.Audio.Plugins.Family</c>), whose static
/// <c>FamilyPlugin</c> class registers it alone (<c>Register()</c>, <c>Unregister()</c>; see <see
/// cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every one of them through <see cref="IdrakPlugins"/>; no
/// model is loaded until one asks for its family, so registering costs only the names.
/// </summary>
public static class AudioPlugins
{
    /// <summary>
    /// The package, as <c>-P</c> finds it too (<c>[assembly: IdrakPluginPackage]</c>): its name ("Idrak.Audio.Plugins", as the
    /// library's "not registered" messages and the registries' origins name it) and <see cref="All"/>.
    /// </summary>
    public static IIdrakPluginPackage Package { get; } = new AudioPluginsPackage();

    /// <summary>
    /// Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: Whisper
    /// (<see cref="Whisper.WhisperPlugin"/>), the espeak-ng phonemizer (<see cref="EspeakNg.EspeakNgPlugin"/>, before the
    /// VITS family that needs it), VITS with uroman's Arabic romanizer (<see cref="Vits.VitsPlugin"/>), OpenVoice
    /// (<see cref="OpenVoice.OpenVoicePlugin"/>), IdrakAudio (<see cref="IdrakAudio.IdrakAudioPlugin"/>) and the RNNoise
    /// denoiser, the audio processor "rnnoise" (<see cref="RnNoise.RnNoisePlugin"/>).
    /// </summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } =
        [Whisper.WhisperPlugin.Plugin, EspeakNg.EspeakNgPlugin.Plugin, Vits.VitsPlugin.Plugin, OpenVoice.OpenVoicePlugin.Plugin, IdrakAudio.IdrakAudioPlugin.Plugin,
         RnNoise.RnNoisePlugin.Plugin];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll(IIdrakPluginPackage)"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(Package);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(Package);
}

/// <summary>
/// The package <c>-P</c> and <see cref="IdrakPlugins.Load"/> find through this assembly's
/// <see cref="IdrakPluginPackageAttribute"/>: <see cref="AudioPlugins"/>' name and plug-ins.
/// </summary>
internal sealed class AudioPluginsPackage : IIdrakPluginPackage
{
    /// <inheritdoc/>
    public string Name => "Idrak.Audio.Plugins";

    /// <inheritdoc/>
    public IReadOnlyList<IIdrakPlugin> All => AudioPlugins.All;
}
