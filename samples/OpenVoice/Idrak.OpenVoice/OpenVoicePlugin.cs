// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.OpenVoice;

/// <summary>Registers OpenVoice's converter in <see cref="VoiceConverters"/> under <see cref="OpenVoiceFamily.FamilyName"/>.</summary>
public static class OpenVoicePlugin
{
    /// <summary>Registers the family.</summary>
    public static void Register() => VoiceConverters.Register(new OpenVoiceFamily());

    /// <summary>The entry the command line calls for <c>-P Idrak.OpenVoice.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registration <see cref="Register"/> made.</summary>
    public static void Unregister() => VoiceConverters.Unregister(OpenVoiceFamily.FamilyName);
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
