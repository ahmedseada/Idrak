// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.LegalOcr;

/// <summary>A model the page can switch to: a Gemma 3 vision checkpoint, its preprocessing, prompt and answer length.</summary>
public sealed record ReaderModel
{
    /// <summary>The switch's key ("bakrianoo", "gemma"; a tuned adapter's is "adapter:" and its folder's name).</summary>
    public required string Id { get; init; }

    /// <summary>The name the page shows.</summary>
    public required string Name { get; init; }

    /// <summary>The Hugging Face repository, downloaded once into the library's cache (unless <see cref="Folder"/> is given).</summary>
    public string? Repo { get; init; }

    /// <summary>A checkpoint folder already on disk (config.json, safetensors, tokenizer), used instead of <see cref="Repo"/>.</summary>
    public string? Folder { get; init; }

    /// <summary>A LoRA adapter folder merged into the weights as they load; null for a full checkpoint.</summary>
    public string? Adapter { get; init; }

    /// <summary>One line about the model.</summary>
    public string? About { get; init; }

    /// <summary>The image transforms before the model's own processor (the library's pipeline syntax).</summary>
    public string Preprocessing { get; init; } = "";

    /// <summary>The instruction sent with the image.</summary>
    public string Prompt { get; init; } = "Extract details to JSON.";

    /// <summary>The most tokens the answer may take.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Gemma 3's pan and scan by default (an elongated page also read as crops); off, as transformers' processor has it.</summary>
    public bool PanAndScan { get; init; }
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

    /// <summary>The training conversations (ShareGPT or messages) the Fine-tune tab trains on.</summary>
    public string? TrainingData { get; init; }

    /// <summary>Where the training file's images are (a folder or a zip); null: <see cref="EvaluationImages"/>.</summary>
    public string? TrainingImages { get; init; }

    /// <summary>Where tuned adapters are written, one folder each (relative to the app's folder; listed in the switch).</summary>
    public string AdaptersFolder { get; init; } = "adapters";

    /// <summary>The Fine-tune tab's defaults.</summary>
    public TuningDefaults Tuning { get; init; } = new();
}

/// <summary>The Fine-tune tab's settings, each one the page can change for a run.</summary>
public sealed record TuningDefaults
{
    /// <summary>The model of the switch the adapter is tuned on (a full checkpoint, or an adapter to continue).</summary>
    public string Base { get; init; } = "gemma";

    /// <summary>How the base's projections are stored while tuning: "bf16", "int8" or "int4" (QLoRA: the least memory).</summary>
    public string Weights { get; init; } = "bf16";

    /// <summary>The image transforms every training page gets (saved with the adapter; reading it uses them too).</summary>
    public string Preprocessing { get; init; } = "grayscale,max_width=1024,contrast=1.5";

    /// <summary>Passes over the training data.</summary>
    public int Epochs { get; init; } = 1;

    /// <summary>The peak learning rate.</summary>
    public float LearningRate { get; init; } = 2e-4f;

    /// <summary>LoRA rank.</summary>
    public int Rank { get; init; } = 16;

    /// <summary>LoRA alpha.</summary>
    public float Alpha { get; init; } = 32;

    /// <summary>The longest training sequence in tokens (the image's tokens included).</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>Tokens per batch (several pages packed into one).</summary>
    public int BatchTokens { get; init; } = 4096;

    /// <summary>Batches whose gradients add up before each optimizer step.</summary>
    public int Accumulate { get; init; } = 1;

    /// <summary>Train the vision projector beside the adapters.</summary>
    public bool TrainProjector { get; init; }

    /// <summary>Evaluation pages read and scored by CER while training (0: no CER, the evaluation loss only).</summary>
    public int CerPages { get; init; } = 8;

    /// <summary>Score CER every this many steps (0: at the end of each epoch).</summary>
    public int CerEvery { get; init; }

    /// <summary>Score CER once before training too (the base model's score to compare with).</summary>
    public bool CerBefore { get; init; } = true;

    /// <summary>The most tokens an answer scored for CER may take.</summary>
    public int AnswerTokens { get; init; } = 1536;

    /// <summary>Train on the first this many pages only (0: all; a short run to try the settings).</summary>
    public int MaxPages { get; init; }

    /// <summary>Evaluate the loss every this many steps (0: at the end of each epoch).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>Save the adapter every this many steps (0: at the end only).</summary>
    public int SaveEvery { get; init; }
}
