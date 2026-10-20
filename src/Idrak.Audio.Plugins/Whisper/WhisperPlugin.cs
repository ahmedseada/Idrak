// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Audio.Plugins.Whisper;

/// <summary>
/// The Whisper-style speech recognition family: its checkpoints saved by transformers
/// (<c>WhisperForConditionalGeneration</c>: config.json, model.safetensors, tokenizer.json, preprocessor_config.json,
/// generation_config.json) registered in <see cref="SpeechFamilies"/> under <see cref="WhisperFamily.Architecture"/>, so
/// <c>SpeechModels.Load</c> reads them into a <see cref="WhisperRecognizer"/>.
/// </summary>
public static class WhisperPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("whisper"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new WhisperRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="SpeechFamilies"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class WhisperRegistration() : IdrakPlugin("whisper")
    {
        protected override void RegisterParts() =>
            Part(() => SpeechFamilies.Register(new WhisperFamily()), () => SpeechFamilies.Unregister(WhisperFamily.Architecture));
    }
}
