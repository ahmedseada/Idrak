// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Audio;

/// <summary>
/// Loads speech recognition models through their registered family (<see cref="SpeechFamilies"/>), as
/// <c>ImageModels.Load</c> loads image models: a model folder (config.json and safetensors weights, or any folder a
/// registered checkpoint format reads), its architecture from config.json's "architectures", the tensors opened one at a
/// time as the family makes its layers.
/// </summary>
/// <remarks>
/// <code>
/// SpeechFamilies.Register(new WhisperFamily());                       // a plug-in or the app brings the family
/// using var recognizer = SpeechModels.Load("models/whisper-tiny");    // config.json, model.safetensors, tokenizer.json, preprocessor_config.json
/// var text = recognizer.Transcribe(AudioCodecs.Decode("talk.wav")).Text;
/// </code>
/// </remarks>
public static class SpeechModels
{
    /// <summary>
    /// Reads the speech model at <paramref name="path"/>: a model folder, or its config.json or a .safetensors file in it.
    /// </summary>
    /// <exception cref="NotSupportedException">No family is registered under the architecture; the message names <see cref="SpeechFamilies"/>.</exception>
    /// <exception cref="InvalidDataException">The checkpoint names no architecture, or its family cannot read it.</exception>
    public static ISpeechRecognizer Load(string path, SpeechModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        options ??= new SpeechModelOptions();
        string target = File.Exists(path) && (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || Path.GetFileName(path) == "config.json")
            ? Path.GetDirectoryName(Path.GetFullPath(path))! : path;
        var format = CheckpointFormats.For(target);
        string folder = format.Prepare(target);
        string configFile = Path.Combine(folder, "config.json");
        var config = File.Exists(configFile)
            ? JsonNode.Parse(File.ReadAllText(configFile)) as JsonObject ?? throw new InvalidDataException($"{configFile} is not a JSON object.")
            : throw new InvalidDataException($"{folder} has no config.json (it names the speech model's architecture).");
        string architecture = options.Architecture
            ?? (config["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : (string?)config["architecture"])
            ?? throw new InvalidDataException($"{configFile} names no architecture (\"architectures\"); pass SpeechModelOptions.Architecture.");
        var family = SpeechFamilies.Get(architecture);                        // refused before any weight is opened
        var notes = new List<string>(format.Notes(folder));
        using var tensors = format.Open(folder);
        var checkpoint = new SpeechCheckpoint(architecture, config, path, folder, tensors, options, notes);
        return family.Read(checkpoint) ?? throw new InvalidOperationException($"Speech family '{architecture}' read {path} as nothing.");
    }
}
