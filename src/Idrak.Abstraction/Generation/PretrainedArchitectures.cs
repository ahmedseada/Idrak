// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

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
/// ("architectures": [...]). Idrak.LanguageModels registers Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma 2 and Gemma 3
/// (text), and the mixture-of-experts families Mixtral, Qwen2-MoE and Qwen3-MoE; add others with <see cref="Register"/>
/// (its <c>PretrainedFamilies.LlamaStyle</c> makes one for families that share the Llama naming).
/// </summary>
public static class PretrainedArchitectures
{
    private static readonly Dictionary<string, PretrainedArchitecture> Registry = new(StringComparer.Ordinal);

    static PretrainedArchitectures() => LibraryDefaults.Ensure();

    /// <summary>Registers (or replaces) how to read the architecture <paramref name="name"/>.</summary>
    public static void Register(string name, PretrainedArchitecture architecture)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(architecture);
        lock (Registry)
        {
            Registry[name] = architecture;
        }
    }

    /// <summary>Removes the architecture <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static PretrainedArchitecture Get(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var architecture) ? architecture
                : throw new NotSupportedException($"No architecture '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with PretrainedArchitectures.Register.");
        }
    }
}
