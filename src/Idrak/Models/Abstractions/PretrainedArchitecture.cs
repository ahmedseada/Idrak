// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How to read one family of pretrained models: its configuration (a Hugging Face <c>config.json</c>) becomes a
/// <see cref="DecoderSpec"/>, and each of Idrak's weight names (see <see cref="DecoderSpec"/>) is found in the
/// checkpoint. What a family can do beyond that
/// is an interface of its own, tested once when a model is loaded or made: <see cref="INetworkBuildingArchitecture"/> (it
/// builds its own network instead of the spec's), <see cref="IConfigWritingArchitecture"/> (it writes a spec as
/// config.json, so models of the family can be made from a spec) and <see cref="ICheckpointFittedArchitecture"/> (its
/// checkpoints come in several namings). <see cref="PretrainedArchitecture"/> is a base with the default layout rule.
/// </summary>
public interface IPretrainedArchitecture
{
    /// <summary>
    /// The model described by a configuration; append anything approximated to <paramref name="notes"/>. For a family that
    /// builds its own network, the spec still gives the model's sizes (vocabulary, width, layers, heads, context length) to
    /// the tokenizer, generation and tools, and is not built.
    /// </summary>
    DecoderSpec Spec(JsonObject config, List<string> notes);

    /// <summary>The checkpoint's name for one of Idrak's weight names (null when the checkpoint does not store it).</summary>
    string? TensorName(string name);

    /// <summary>
    /// Whether the checkpoint stores this weight as [out, in] (the PyTorch Linear layout), so it is transposed to Idrak's
    /// [in, out] (<see cref="PretrainedArchitecture.TransposedProjection"/>: every projection, attention, feed-forward and
    /// head, for most families).
    /// </summary>
    bool Transposed(string name);
}

/// <summary>
/// A family that builds the network itself instead of building its spec (Idrak's <c>spec.Build(weights, options)</c>).
/// </summary>
public interface INetworkBuildingArchitecture : IPretrainedArchitecture
{
    /// <summary>
    /// The network's layers in order: together they take token ids [batch, time] and return logits [batch, time,
    /// vocabulary], and the last one is the output head (a linear layer, so the model can be fine-tuned); a network built
    /// from a spec (Idrak's <c>Sequential</c>) is taken as it is, other layers are put in one in this order. To work with
    /// the KV cache, the attention layers implement <see cref="ICachedModule"/>.
    /// </summary>
    IEnumerable<Module> Build(PretrainedBuildContext context);
}

/// <summary>A family that writes a spec as config.json in its own keys, for models made from a spec (<c>PretrainedModel.Create</c>).</summary>
public interface IConfigWritingArchitecture : IPretrainedArchitecture
{
    /// <summary>
    /// The config.json of <paramref name="spec"/> in this family's keys (the inverse of <see cref="IPretrainedArchitecture.Spec"/>):
    /// its spec must be <paramref name="spec"/> again.
    /// </summary>
    /// <exception cref="NotSupportedException">A spec the family's keys cannot say.</exception>
    JsonObject Config(DecoderSpec spec);
}

/// <summary>
/// A family whose checkpoints come in several namings (Gemma 3's vision-language model is saved three ways): loading asks
/// it for the architecture fitted to one checkpoint before reading the spec and the weights, and keeps the one it returns
/// (so adapters and exports use the checkpoint's naming).
/// </summary>
public interface ICheckpointFittedArchitecture : IPretrainedArchitecture
{
    /// <summary>The architecture fitted to a checkpoint holding the tensors <paramref name="names"/>.</summary>
    IPretrainedArchitecture ForCheckpoint(IReadOnlySet<string> names);
}

/// <summary>
/// A base for a family (<see cref="IPretrainedArchitecture"/>): it reads its spec and names its tensors; every projection
/// is transposed unless it says otherwise.
/// </summary>
public abstract class PretrainedArchitecture : IPretrainedArchitecture
{
    /// <inheritdoc />
    public abstract DecoderSpec Spec(JsonObject config, List<string> notes);

    /// <inheritdoc />
    public abstract string? TensorName(string name);

    /// <inheritdoc />
    public virtual bool Transposed(string name) => TransposedProjection(name);

    /// <summary>
    /// The usual layout rule: every projection (attention, feed-forward, head) is stored as [out, in], the PyTorch Linear
    /// layout, and transposed to Idrak's [in, out].
    /// </summary>
    public static bool TransposedProjection(string name) =>
        name.Contains(".attn.", StringComparison.Ordinal) && !name.Contains("_norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name.Contains(".mlp.", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name == "head.weight";
}

/// <summary>What <see cref="INetworkBuildingArchitecture.Build"/> receives.</summary>
/// <param name="Config">The model's config.json.</param>
/// <param name="Spec">The spec <see cref="IPretrainedArchitecture.Spec"/> read from it.</param>
/// <param name="Weights">
/// The checkpoint's tensors by Idrak's names, through <see cref="IPretrainedArchitecture.TensorName"/> and
/// <see cref="IPretrainedArchitecture.Transposed"/> (with a merged adapter's updates added), as building a spec reads them.
/// </param>
/// <param name="Checkpoint">The checkpoint's tensors by their stored names, as stored.</param>
/// <param name="Options">Device, packed weight format and context length chosen by the caller.</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record PretrainedBuildContext(JsonObject Config, DecoderSpec Spec, IWeightSource Weights, ITensorStore Checkpoint,
    DecoderBuildOptions Options, List<string> Notes);

/// <summary>Settings for building a <see cref="DecoderSpec"/> into a network (Idrak's <c>DecoderBuilder.Build</c>).</summary>
public sealed record DecoderBuildOptions
{
    /// <summary>Where the layers are created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>
    /// How many decoder layers, the first ones, are made on <see cref="Device"/> (a weight placement's split,
    /// <see cref="WeightPlacements"/>): the others are made on <see cref="HostDevice"/> and computed there, their input
    /// moved there and the last one's output moved back for the final norm and head (only activations cross). Null (the
    /// default): every layer on <see cref="Device"/>.
    /// </summary>
    public int? DeviceLayers { get; init; }

    /// <summary>Where the layers past <see cref="DeviceLayers"/> are made and computed: the CPU when null.</summary>
    public Device? HostDevice { get; init; }

    /// <summary>
    /// Build each decoder layer on its first use (plan 37, W2): until then it holds no memory and its weights are not read,
    /// so the build's time does not grow with the layers. Needs weights to read from, which must stay readable until every
    /// layer is built. False (the default): every layer built now.
    /// </summary>
    public bool LazyLayers { get; init; }

    /// <summary>
    /// Where a mixture-of-experts model's experts live (plan 37, R1, <see cref="ExpertPlacements"/>). With
    /// <see cref="ExpertPlan.Offload"/> and a device other than the host, each mixture-of-experts layer keeps at most
    /// <see cref="ExpertPlan.DeviceExperts"/> experts on the device and makes the others on <see cref="HostDevice"/>
    /// (Idrak's <c>MixtureOfExperts.Cache</c>). Null (the default): every expert where its layer is.
    /// </summary>
    public ExpertPlan? Experts { get; init; }

    /// <summary>
    /// Read each expert of a mixture-of-experts layer the first time the router picks it (plan 37, R1 with W2's lazy
    /// loading): until then it is a placeholder holding no memory, so experts never picked are never read. Needs weights
    /// to read from, which must stay readable while the model is used. False (the default): every expert read with its layer.
    /// </summary>
    public bool LazyExperts { get; init; }

    // The placement the layers follow the memory with at step boundaries, with its request at load (PretrainedModel.Load
    // sets it when the placement may move layers), or null: they stay where they are built.
    internal (IWeightPlacement Placement, WeightPlacementRequest Request)? FollowMemory { get; init; }

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
