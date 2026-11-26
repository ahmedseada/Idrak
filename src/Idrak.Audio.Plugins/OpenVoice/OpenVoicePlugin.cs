// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio.Plugins.OpenVoice;

/// <summary>
/// The family: reads an OpenVoice converter folder (OpenVoice's config.json and its tensors) into an
/// <see cref="OpenVoiceConverter"/>. Its config.json names no
/// architecture, so it is recognised by its shape (<see cref="OpenVoiceConfig.Recognizes"/>).
/// </summary>
public sealed class OpenVoiceFamily : IVoiceConverterFamily
{
    /// <summary>The family's name, "openvoice".</summary>
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
