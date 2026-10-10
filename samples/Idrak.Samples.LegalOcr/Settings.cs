// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.LegalOcr;

/// <summary>A model the page can switch to: a Gemma 3 vision checkpoint, its preprocessing, prompt and answer length.</summary>
public sealed record ReaderModel
{
    /// <summary>The switch's key ("bakrianoo", "ours").</summary>
    public required string Id { get; init; }

    /// <summary>The name the page shows.</summary>
    public required string Name { get; init; }

    /// <summary>The Hugging Face repository, downloaded once into the library's cache (unless <see cref="Folder"/> is given).</summary>
    public string? Repo { get; init; }

    /// <summary>A checkpoint folder already on disk (config.json, safetensors, tokenizer), used instead of <see cref="Repo"/>.</summary>
    public string? Folder { get; init; }

    /// <summary>A LoRA adapter folder merged into the weights as they load (ours); null for a full checkpoint.</summary>
    public string? Adapter { get; init; }

    /// <summary>One line about the model.</summary>
    public string? About { get; init; }

    /// <summary>The image transforms before the model's own processor (the library's pipeline syntax).</summary>
    public string Preprocessing { get; init; } = "";

    /// <summary>The instruction sent with the image.</summary>
    public string Prompt { get; init; } = "Extract details to JSON.";

    /// <summary>The most tokens the answer may take.</summary>
    public int MaxTokens { get; init; } = 2048;
}

/// <summary>The app's settings (the "LegalOcr" section of appsettings.json, or --LegalOcr:Key=value on the command line).</summary>
public sealed record LegalOcrSettings
{
    /// <summary>"auto" (the best device found), "cpu", "cuda", "cuda:N".</summary>
    public string Device { get; init; } = "auto";

    /// <summary>How the projections are stored: "bf16" (half of float32; exact for the bfloat16 checkpoints), "int8", "int4" or "f32".</summary>
    public string Weights { get; init; } = "bf16";

    /// <summary>The attention cache: "f32" or "bf16" (half the memory).</summary>
    public string KeyValueCache { get; init; } = "f32";

    /// <summary>The longest prompt and answer in tokens (the image's tokens included).</summary>
    public int Context { get; init; } = 8192;

    /// <summary>The models of the switch.</summary>
    public List<ReaderModel> Models { get; init; } = [];

    /// <summary>The evaluation pages: a data file of conversations (ShareGPT or messages) with their expected answers.</summary>
    public string? EvaluationData { get; init; }

    /// <summary>Where the data file's images are (a folder or a zip).</summary>
    public string? EvaluationImages { get; init; }

    /// <summary>A Hugging Face token for gated models (else HF_TOKEN or the saved login).</summary>
    public string? HuggingFaceToken { get; init; }
}
