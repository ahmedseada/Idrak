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

    /// <summary>
    /// The key of the answer's JSON holding the page's whole text, shown alone in the Full text tab ("full_text_extract"
    /// for bakrianoo's model); null: a key starting with "full_text", else the longest text in the answer.
    /// </summary>
    public string? FullText { get; init; }
}

/// <summary>The app's settings (the "LegalOcr" section of appsettings.json, or --LegalOcr:Key=value on the command line).</summary>
public sealed record LegalOcrSettings
{
    /// <summary>"auto" (the best device found), "cpu", "cuda", "cuda:N".</summary>
    public string Device { get; init; } = "auto";

    /// <summary>The model (an id of <see cref="Models"/>) loaded before the server starts, or null to pick one on the page.</summary>
    public string? LoadAtStart { get; init; }

    /// <summary>How the projections are stored: "bf16" (half of float32; exact for the bfloat16 checkpoints), "int8", "int4" or "f32".</summary>
    public string Weights { get; init; } = "bf16";

    /// <summary>
    /// How the output head is stored when the projections are int8 or int4: "bf16" (the checkpoint's own next-word scores
    /// and one vocabulary table: the most accurate, and less memory than an int8 or int4 head beside the table; a little
    /// slower per word) or "" (as the projections: the fastest).
    /// </summary>
    public string HeadWeights { get; init; } = "bf16";

    /// <summary>
    /// The vision encoder's products: "f32" (the checkpoint's own features) or "bf16" (tensor cores: a page's encoding
    /// several times faster on GPUs that have them, features within bfloat16's rounding; compare CER on the evaluation
    /// pages before reading with it).
    /// </summary>
    public string EncoderPrecision { get; init; } = "f32";

    /// <summary>
    /// The prompt's pass through the language model (the page's image tokens and the instruction): "f32" or "bf16" (tensor
    /// cores where the GPU has them; the answer's decoding unaffected; compare CER on the evaluation pages before reading with it).
    /// </summary>
    public string PrefillPrecision { get; init; } = "f32";

    /// <summary>The attention cache: "f32" or "bf16" (half the memory).</summary>
    public string KeyValueCache { get; init; } = "f32";

    /// <summary>The longest prompt and answer in tokens (the image's tokens included).</summary>
    public int Context { get; init; } = 8192;

    /// <summary>
    /// Pages the Evaluate tab reads together (ImageReader.ReadBatch: one pass a token for all of them, each answer the
    /// page's own): 1 reads them one by one. Each page in a batch holds its own attention cache; when the device has no
    /// room for a batch, its pages are read one by one.
    /// </summary>
    public int ReadBatch { get; init; } = 1;

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

    /// <summary>The Read tab's generation settings to start from (repetition penalties, the loop stop).</summary>
    public GenerationDefaults Generation { get; init; } = new();

    /// <summary>
    /// The values the Read tab's preprocessing steps start from, by transform: a step in the pipeline syntax
    /// (<c>"unsharp": "unsharp=2,percent=150,threshold=3"</c>). A transform without one starts empty.
    /// </summary>
    public Dictionary<string, string> PreprocessingDefaults { get; init; } = new(StringComparer.OrdinalIgnoreCase);
}

/// <summary>How a read generates its answer (greedy throughout); each one the page can change for a read.</summary>
public sealed record GenerationDefaults
{
    /// <summary>Divides the scores of tokens seen in the last <see cref="RepeatLastN"/> tokens (1 = off).</summary>
    public float RepeatPenalty { get; init; } = 1f;

    /// <summary>Subtracted from the score of every token seen recently (0 = off).</summary>
    public float PresencePenalty { get; init; }

    /// <summary>Subtracted from a token's score once per time it was seen recently (0 = off).</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary>How many recent tokens the penalties look at.</summary>
    public int RepeatLastN { get; init; } = 64;

    /// <summary>End the answer when one phrase repeats this many times in a row (0 = off).</summary>
    public int RepeatStop { get; init; } = 4;

    /// <summary>Sample the words (as transformers' generate does with the model's generation_config) instead of greedy reading.</summary>
    public bool Sample { get; init; }

    /// <summary>Sampling's temperature (1: the model's own probabilities).</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>Sample among the k most likely words (0 = all).</summary>
    public int TopK { get; init; } = 64;

    /// <summary>Sample among the most likely words holding this probability (1 = off).</summary>
    public float TopP { get; init; } = 0.95f;

    /// <summary>Drop words less than this fraction as likely as the best one (0 = off).</summary>
    public float MinP { get; init; }
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

    /// <summary>The evaluation loss over the first this many evaluation pages (0: all of them; each is a forward pass, twice a run).</summary>
    public int LossPages { get; init; }
}
