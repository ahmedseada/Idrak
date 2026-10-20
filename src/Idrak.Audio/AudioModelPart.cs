// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Audio;

/// <summary>
/// The "audio" model part (<see cref="ModelParts"/>, a library default this package registers): an audio-language
/// model's audio side, stored beside its decoder under an <c>audio_config</c>, read, made and opened through the family
/// registered under the model's architecture in <see cref="AudioFamilies"/>. Its source is a <see cref="ChatAudioParts"/>
/// that trains when the family's part does (<see cref="IAudioTuningPart"/>).
/// </summary>
internal sealed class AudioModelPart : IModelPart
{
    public string Kind => ChatParts.Audio;

    public bool In(JsonObject config) => AudioFamilies.HasAudio(config);

    // A checkpoint whose family is not registered loads as its decoder alone (a note says why); its audio tensors stay
    // unused until the family is registered and the part is opened.
    public IReadOnlySet<string> StoredTensors(ModelPartCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (AudioFamilies.Find(checkpoint.Architecture) is null)
        {
            checkpoint.Notes.Add($"the audio part is not read: no audio family is registered as '{checkpoint.Architecture}' (AudioFamilies.Register; a plug-in package such as Idrak.Audio.Plugins brings one: AudioPlugins.RegisterAll()).");
            return new HashSet<string>();
        }

        return Read(checkpoint)?.StoredTensors ?? new HashSet<string>();
    }

    public CreatedModelPart Create(ModelPartRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var made = AudioFamilies.Get(request.Architecture).Create(request.Description, request.TextDim, request.Tokenizer, request.Seed);
        return new CreatedModelPart(made.Config, made.Tensors, made.Files);
    }

    public ITrainableSoftParts Open(ModelPartCheckpoint checkpoint, ModelPartOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var audio = Read(checkpoint) ?? throw new InvalidDataException($"The {checkpoint.Architecture} checkpoint holds no audio weights.");
        return new ChatAudioParts(audio, new AudioEncoderOptions { Device = options.Device, Trained = options.Trained }, options.Cache, ownsCache: true);
    }

    private static PretrainedAudio? Read(ModelPartCheckpoint checkpoint)
    {
        var family = AudioFamilies.For(checkpoint.Architecture, checkpoint.Config)
            ?? throw new InvalidDataException($"The {checkpoint.Architecture} config.json has no audio_config.");
        return family.Read(new AudioCheckpoint(checkpoint.Architecture, checkpoint.Config, checkpoint.Tensors, checkpoint.Folder, checkpoint.Open, checkpoint.Notes));
    }
}
