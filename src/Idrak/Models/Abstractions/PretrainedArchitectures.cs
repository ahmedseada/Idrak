// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How to read one family of pretrained models: its configuration (a Hugging Face <c>config.json</c>) becomes a
/// <see cref="DecoderSpec"/>, and each of Idrak's weight names (see <see cref="DecoderSpec"/>) is found in the
/// checkpoint. A family that does not fit <see cref="DecoderSpec"/> builds its own network with <see cref="Build"/>.
/// Register new families with <see cref="PretrainedArchitectures.Register"/>.
/// </summary>
public sealed class PretrainedArchitecture
{
    /// <summary>
    /// The model described by a configuration; append anything approximated to the notes. With <see cref="Build"/>, the
    /// spec still gives the model's sizes (vocabulary, width, layers, heads, context length) to the tokenizer, generation
    /// and tools, and is not built.
    /// </summary>
    public required Func<JsonObject, List<string>, DecoderSpec> Spec { get; init; }

    /// <summary>
    /// Builds the network itself instead of building the spec (Idrak's <c>spec.Build(weights, options)</c>), or null (the
    /// default) to build the spec. It returns the network's layers in order: together they take token ids [batch, time] and
    /// return logits [batch, time, vocabulary], and the last one is the output head (a linear layer, so the model can be
    /// fine-tuned); a network built from a spec (Idrak's <c>Sequential</c>) is taken as it is, other layers are put in one in
    /// this order. To work with the KV cache, the attention layers implement <see cref="ICachedModule"/>.
    /// </summary>
    public Func<PretrainedBuildContext, IEnumerable<Module>>? Build { get; init; }

    /// <summary>The checkpoint's name for one of Idrak's weight names (null when the checkpoint does not store it).</summary>
    public required Func<string, string?> TensorName { get; init; }

    /// <summary>
    /// For a family whose checkpoints come in several namings (Gemma 3's vision-language model is saved three ways): the
    /// architecture fitted to one checkpoint, from the names of the tensors it holds. Loading asks it before reading the
    /// spec and the weights, and keeps the one it returns (so adapters and exports use the checkpoint's naming). Null (the
    /// default) when <see cref="TensorName"/> fits every checkpoint of the family.
    /// </summary>
    public Func<IReadOnlySet<string>, PretrainedArchitecture>? ForCheckpoint { get; init; }

    /// <summary>
    /// Whether the checkpoint stores this weight as [out, in] (the PyTorch Linear layout), so it is transposed to
    /// Idrak's [in, out]. By default: every projection (attention, feed-forward, head) is transposed.
    /// </summary>
    public Func<string, bool> Transposed { get; init; } = name =>
        name.Contains(".attn.", StringComparison.Ordinal) && !name.Contains("_norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name.Contains(".mlp.", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name == "head.weight";
}

/// <summary>What <see cref="PretrainedArchitecture.Build"/> receives.</summary>
/// <param name="Config">The model's config.json.</param>
/// <param name="Spec">The spec <see cref="PretrainedArchitecture.Spec"/> read from it.</param>
/// <param name="Weights">
/// The checkpoint's tensors by Idrak's names, through <see cref="PretrainedArchitecture.TensorName"/> and
/// <see cref="PretrainedArchitecture.Transposed"/> (with a merged adapter's updates added), as building a spec reads them.
/// </param>
/// <param name="Checkpoint">The checkpoint's tensors by their stored names, as stored.</param>
/// <param name="Options">Device, packed weight format and context length chosen by the caller.</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record PretrainedBuildContext(JsonObject Config, DecoderSpec Spec, IWeightSource Weights, ITensorStore Checkpoint,
    DecoderBuildOptions Options, List<string> Notes);

/// <summary>
/// The model families pretrained models are loaded as, by the architecture name in <c>config.json</c>
/// ("architectures": [...]). Idrak registers Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma 2 and Gemma 3
/// (text, and the text decoder of its vision-language model: the vision part is a <see cref="VisionFamilies"/> registration,
/// which the library does not make), and the mixture-of-experts families Mixtral, Qwen2-MoE and Qwen3-MoE; add others with <see cref="Register"/>
/// (its <c>PretrainedFamilies.LlamaStyle</c> makes one for families that share the Llama naming).
/// </summary>
public static class PretrainedArchitectures
{
    private static readonly SlotTable<string, PretrainedArchitecture> Registry = new(nameof(PretrainedArchitectures), comparer: StringComparer.Ordinal,
        unguarded: "a family's functions (spec, tensor names, build) must agree with one another, so the app's and the library's cannot be mixed");

    static PretrainedArchitectures() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterPretrainedArchitectures);   // the built-in families, on first use

    /// <summary>Registers how to read the architecture <paramref name="name"/>; under a built-in name it overrides the library's until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, PretrainedArchitecture architecture)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(architecture);
        Registry.Register(name, architecture, System.Reflection.Assembly.GetCallingAssembly(), architecture.Spec);
    }

    /// <summary>Removes the app's architecture <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static PretrainedArchitecture Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No architecture '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with PretrainedArchitectures.Register.");

    /// <summary>The library's architecture <paramref name="name"/>, whatever an app registered over it (for an app's family to build on); null when the library has none.</summary>
    public static PretrainedArchitecture? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the architecture <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}

/// <summary>Settings for building a <see cref="DecoderSpec"/> into a network (Idrak's <c>DecoderBuilder.Build</c>).</summary>
public sealed record DecoderBuildOptions
{
    /// <summary>Where the layers are created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>
    /// Store every Linear layer (attention, feed-forward and output head) as int8: each weight is quantized on the host as
    /// it is read, so a large model never exists as float32 on the device.
    /// </summary>
    public bool Int8 { get; init; }

    /// <summary>
    /// Store the projections as bfloat16 (half the float32 memory, values to about 3 significant digits; exact for
    /// checkpoints stored in bfloat16). <see cref="Int8"/> takes precedence.
    /// </summary>
    public bool BFloat16 { get; init; }

    /// <summary>
    /// Store the projections as 4-bit weights (see <see cref="Int4Weight"/>): about 5 bits per weight, quantized on the host
    /// as they are read. <see cref="Int8"/> takes precedence; this over <see cref="BFloat16"/>.
    /// </summary>
    public bool Int4 { get; init; }

    /// <summary>
    /// Store the projections in the packed format registered under this name (see <see cref="PackedWeight.FormatNames"/>:
    /// "int8", "int4", "bfloat16", or a format of your own added with <see cref="PackedWeight.Register"/>), packed on the
    /// host as they are read. Takes precedence over <see cref="Int8"/>, <see cref="Int4"/> and <see cref="BFloat16"/>.
    /// </summary>
    public string? PackedFormatName { get; init; }

    /// <summary>
    /// Standard deviation of a normal initialization of every weight matrix and embedding table (biases zero, norm gains
    /// one), as GPT-2 and nanoGPT use (0.02). Null keeps the default (uniform Xavier weights, ±0.02 embeddings).
    /// Ignored when building from weights.
    /// </summary>
    public float? InitStd { get; init; }

    /// <summary>Longest sequence the model will see (the rotary tables' size); the spec's <see cref="DecoderSpec.MaxPositions"/> when null.</summary>
    public int? MaxPositions { get; init; }

    /// <summary>Seed for random initialization (when no weights are given).</summary>
    public int Seed { get; init; }
}
