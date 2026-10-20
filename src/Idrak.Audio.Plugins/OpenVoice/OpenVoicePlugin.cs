// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Models;

namespace Idrak.Audio.Plugins.OpenVoice;

/// <summary>
/// OpenVoice's tone colour converter as a voice converter family: registered in <see cref="VoiceConverters"/> under
/// <see cref="OpenVoiceFamily.FamilyName"/>, so <c>VoiceConverters</c> reads its folders into an
/// <see cref="OpenVoiceConverter"/> (a reference encoder for a person's clips, and the converter that changes any speech
/// to that voice).
/// </summary>
public static class OpenVoicePlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("openvoice"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new OpenVoiceRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="VoiceConverters"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class OpenVoiceRegistration() : IdrakPlugin("openvoice")
    {
        protected override void RegisterParts() =>
            Part(() => VoiceConverters.Register(new OpenVoiceFamily()), () => VoiceConverters.Unregister(OpenVoiceFamily.FamilyName));
    }
}

/// <summary>
/// The family: reads an OpenVoice converter folder (OpenVoice's config.json and its tensors, safetensors or any
/// format <c>CheckpointFormats</c> opens) into an <see cref="OpenVoiceConverter"/>. Its config.json names no
/// architecture, so it is recognised by its shape (<see cref="OpenVoiceConfig.Recognizes"/>).
/// </summary>
public sealed class OpenVoiceFamily : IVoiceConverterFamily
{
    /// <summary>The name it is registered under, "openvoice".</summary>
    public const string FamilyName = "openvoice";

    /// <inheritdoc />
    public string Name => FamilyName;

    /// <inheritdoc />
    public bool Recognizes(JsonObject config) => OpenVoiceConfig.Recognizes(config);

    /// <inheritdoc />
    public IVoiceConverter Read(VoiceConverterCheckpoint checkpoint, VoiceConverterLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        var config = OpenVoiceConfig.FromJson(checkpoint.Config);
        var weights = new StoredLayers(OpenVoiceTensors.Over(checkpoint.Tensors), options.Device ?? Device.Default);
        return new OpenVoiceConverter(weights, config, checkpoint.Folder, checkpoint.Notes);
    }
}
