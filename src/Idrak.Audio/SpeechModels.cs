// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio;

/// <summary>Settings for <see cref="SpeechModels.Create"/>.</summary>
public sealed record SpeechCreateOptions
{
    /// <summary>Where the created model is loaded (the default device when null); its weights are drawn and written first.</summary>
    public Device? Device { get; init; }

    /// <summary>The seed the weights are drawn from.</summary>
    public int Seed { get; init; }

    /// <summary>The architecture (a <see cref="SpeechFamilies"/> name); null: the spec's "architecture" (or its "architectures"[0]).</summary>
    public string? Architecture { get; init; }

    /// <summary>Write into a folder that already holds files (they are replaced); otherwise such a folder is refused.</summary>
    public bool Overwrite { get; init; }
}

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

    /// <summary>
    /// A speech model that did not exist: the family registered under the spec's architecture makes it from
    /// <paramref name="spec"/> (<see cref="ISpeechFamily.Create"/>: its tensors drawn from the seed, its config.json and
    /// other files), which is written into <paramref name="folder"/> (config.json with "architectures", model.safetensors
    /// in float32, the tokenizer as tokenizer.json when it is an <see cref="IPretrainedTokenizer"/> and the family wrote none, the
    /// family's files) and loaded from there, so it is a model like any other: it transcribes, trains and saves (plan 21, S6).
    /// </summary>
    /// <exception cref="NotSupportedException">No family is registered under the architecture, or it makes no models from a description.</exception>
    /// <exception cref="InvalidDataException">The spec names no architecture.</exception>
    /// <exception cref="IOException">The folder holds files and <see cref="SpeechCreateOptions.Overwrite"/> is off.</exception>
    public static ISpeechRecognizer Create(string folder, JsonObject spec, ITokenizer tokenizer, SpeechCreateOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options ??= new SpeechCreateOptions();
        string architecture = Architecture(spec, options);
        var family = SpeechFamilies.Get(architecture);
        CheckFolder(folder, options);
        Write(folder, architecture, family.Create(spec, tokenizer, options.Seed), tokenizer);
        return Load(folder, new SpeechModelOptions { Device = options.Device });
    }

    // The architecture a create spec names (the options' first).
    internal static string Architecture(JsonObject spec, SpeechCreateOptions options) =>
        options.Architecture ?? (string?)spec["architecture"] ?? (spec["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : null)
            ?? throw new InvalidDataException("The speech model's spec names no architecture (\"architecture\"); pass SpeechCreateOptions.Architecture.");

    // A folder that already holds files is refused unless the options overwrite it.
    internal static void CheckFolder(string folder, SpeechCreateOptions options)
    {
        if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any() && !options.Overwrite)
        {
            throw new IOException($"{folder} already holds files; choose an empty folder or overwrite.");
        }
    }

    // A created model written as a model folder: config.json with "architectures", model.safetensors in float32, the
    // family's files, and the tokenizer as tokenizer.json when it is a pretrained one and the family wrote none.
    internal static void Write(string folder, string architecture, CreatedSpeechModel created, ITokenizer tokenizer)
    {
        var config = (JsonObject)created.Config.DeepClone();
        config["architectures"] = new JsonArray((JsonNode)architecture);
        Directory.CreateDirectory(folder);
        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), created.Tensors, SafeTensorType.F32);
        foreach (var (name, text) in created.Files)
        {
            File.WriteAllText(Path.Combine(folder, name), text);
        }

        if (!created.Files.ContainsKey("tokenizer.json") && tokenizer is IPretrainedTokenizer pretrained)
        {
            pretrained.Save(Path.Combine(folder, "tokenizer.json"));
        }

        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }
}
