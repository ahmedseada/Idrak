// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Audio;

/// <summary>
/// The audio side of a loaded audio-language model, read beside it: core's <see cref="PretrainedModel"/> knows no audio
/// part, so this package reads it from the model's folder through the family registered for the model's architecture
/// (<see cref="AudioFamilies.For"/>), opening the checkpoint again as a vision family's checkpoint is opened.
/// </summary>
public static class AudioLanguage
{
    /// <summary>
    /// The audio part of <paramref name="model"/> as its family reads it: null for a model whose config.json has no
    /// <c>audio_config</c> (a text-only or vision-language checkpoint), else the family's reading of the checkpoint
    /// (null again when that checkpoint holds no audio weights).
    /// </summary>
    /// <param name="model">A loaded model (its folder holds config.json and the checkpoint).</param>
    /// <param name="architecture">The architecture name to match the family by; config.json's first "architectures" entry when null.</param>
    /// <param name="notes">Where the family appends what it approximated; a new list when null.</param>
    /// <exception cref="NotSupportedException">The checkpoint has an audio part and no family is registered under its name (the message names <see cref="AudioFamilies.Register"/>).</exception>
    /// <exception cref="InvalidDataException">The family found a tensor missing or of the wrong shape.</exception>
    public static PretrainedAudio? Attach(PretrainedModel model, string? architecture = null, List<string>? notes = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        string name = architecture ?? (string?)model.Config["architectures"]?[0]
            ?? throw new InvalidDataException($"{model.Folder}: config.json names no architecture; pass the architecture name.");
        var family = AudioFamilies.For(name, model.Config);
        if (family is null)
        {
            return null;
        }

        var format = CheckpointFormats.For(model.Folder);
        using var tensors = format.Open(model.Folder);
        return family.Read(new AudioCheckpoint(name, model.Config, tensors, model.Folder, () => format.Open(model.Folder), notes ?? []));
    }
}
