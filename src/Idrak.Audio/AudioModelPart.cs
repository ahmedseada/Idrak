// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Audio;

/// <summary>
/// The "audio" model part: an audio-language model's audio side, stored beside its decoder under an <c>audio_config</c>,
/// read, made and opened through the family of the model's architecture (found by its name, which is not on the new
/// design yet). Its source is a <see cref="ChatAudioParts"/>
/// that trains when the family's part does (<see cref="IAudioTuningPart"/>).
/// </summary>
internal sealed class AudioModelPart : IModelPart
{
    // The audio-language families were registered (AudioFamilies) and a checkpoint's found by its architecture name, which is
    // not on the new design yet.
    private static IAudioFamily? FindFamily(string architecture) =>
        throw new NotSupportedException($"The audio family '{architecture}' by name is not on the new design yet.");

    public string Kind => ChatParts.Audio;

    public bool In(JsonObject config) => AudioFamilies.HasAudio(config);

    // A checkpoint whose family is not registered loads as its decoder alone (a note says why); its audio tensors stay
    // unused until the family is registered and the part is opened.
    public IReadOnlySet<string> StoredTensors(ModelPartCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (FindFamily(checkpoint.Architecture) is null)
        {
            checkpoint.Notes.Add($"the audio part is not read: no audio family reads '{checkpoint.Architecture}'.");
            return new HashSet<string>();
        }

        return Read(checkpoint)?.StoredTensors ?? new HashSet<string>();
    }

    public CreatedModelPart Create(ModelPartRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var family = FindFamily(request.Architecture) as ICreatableAudioFamily
            ?? throw new NotSupportedException($"Audio family '{request.Architecture}' cannot make an audio part from a description (it reads checkpoints only).");
        var made = family.Create(request.Description, request.TextDim, request.Tokenizer, request.Seed);
        return new CreatedModelPart(made.Config, made.Tensors, made.Files);
    }

    public ITrainableSoftParts Open(ModelPartCheckpoint checkpoint, ModelPartOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var audio = Read(checkpoint) ?? throw new InvalidDataException($"The {checkpoint.Architecture} checkpoint holds no audio weights.");
        return new ChatAudioParts(audio, new AudioEncoderOptions { Device = options.Device, Trained = options.Trained }, options.Cache, ownsCache: true);
    }

    private static IPretrainedAudio? Read(ModelPartCheckpoint checkpoint)
    {
        var family = (AudioFamilies.HasAudio(checkpoint.Config) ? FindFamily(checkpoint.Architecture) : null)
            ?? throw new InvalidDataException($"The {checkpoint.Architecture} config.json has no audio_config.");
        return family.Read(new AudioCheckpoint(checkpoint.Architecture, checkpoint.Config, checkpoint.Tensors, checkpoint.Folder, checkpoint.Open, checkpoint.Notes));
    }
}
