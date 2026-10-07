// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Nlp;

/// <summary>
/// How an adapter folder was made, written next to the adapter (<see cref="FileName"/>): the base model it was tuned from,
/// as the user named it (a Hugging Face id, folder, .gguf or store:name; PEFT's adapter_config.json often holds only a
/// cache folder's name), the system prompt added to the training conversations, and the longest training sequence. A
/// program can then load the folder, and prompt the model as it was trained, without being told how it was made.
/// </summary>
/// <example>
/// <code>
/// var manifest = TuningManifest.Read(folder)!;
/// using var model = manifest.LoadModel(folder, Device.Cuda());       // the base model with the adapter merged
/// </code>
/// </example>
public sealed record TuningManifest
{
    /// <summary>The file's name in the adapter folder.</summary>
    public const string FileName = "idrak-tuning.json";

    private const string Format = "idrak-tuning/1";

    /// <summary>The model the adapter was tuned from, as given to the tuner.</summary>
    public required string BaseModel { get; init; }

    /// <summary>The system prompt added to training conversations that had none (prompt the tuned model with it), or null.</summary>
    public string? System { get; init; }

    /// <summary>Longest training sequence in tokens.</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>Whether <paramref name="folder"/> has a manifest.</summary>
    public static bool Exists(string folder) => File.Exists(Path.Combine(folder, FileName));

    /// <summary>The manifest in <paramref name="folder"/>, or null when it has none.</summary>
    public static TuningManifest? Read(string folder)
    {
        string path = Path.Combine(folder, FileName);
        if (!File.Exists(path))
        {
            return null;
        }

        var json = JsonNode.Parse(File.ReadAllText(path))?.AsObject() ?? throw new InvalidDataException($"{path} is empty.");
        if ((string?)json["format"] != Format)
        {
            throw new InvalidDataException($"{path}: unknown format '{json["format"]}' (expected {Format}).");
        }

        return new TuningManifest
        {
            BaseModel = (string?)json["baseModel"] ?? throw new InvalidDataException($"{path} names no baseModel."),
            System = (string?)json["system"],
            MaxLength = (int?)json["maxLength"] ?? 2048,
        };
    }

    /// <summary>Writes the manifest into <paramref name="folder"/> (created if needed); a local base model is written as a full path.</summary>
    public void Save(string folder)
    {
        Directory.CreateDirectory(folder);
        var json = new JsonObject
        {
            ["format"] = Format,
            // A local folder or file as a full path, so the adapter loads from any working directory.
            ["baseModel"] = Directory.Exists(BaseModel) || File.Exists(BaseModel) ? Path.GetFullPath(BaseModel) : BaseModel,
            ["maxLength"] = MaxLength,
        };
        if (System is not null)
        {
            json["system"] = System;
        }

        File.WriteAllText(Path.Combine(folder, FileName), json.ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = true, Encoder = global::System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }));
    }

    /// <summary>
    /// The base model with the adapter in <paramref name="folder"/> merged (as fast as the base model), on
    /// <paramref name="device"/> (the CPU when null); bfloat16 weights on CUDA. <paramref name="options"/> overrides the rest.
    /// </summary>
    public PretrainedModel LoadModel(string folder, Device? device = null, PretrainedOptions? options = null)
    {
        device ??= options?.Device ?? Device.Cpu;
        var model = PretrainedModel.Load(ModelSource.Resolve(BaseModel), (options ?? new PretrainedOptions { BFloat16 = device.IsGpu }) with
        {
            Device = device, MergeAdapter = folder,
        });
        model.Network.Eval();
        return model;
    }
}
