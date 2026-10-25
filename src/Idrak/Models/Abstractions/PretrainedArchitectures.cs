// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How to read one family of pretrained models: its configuration (a Hugging Face <c>config.json</c>) becomes a
/// <see cref="DecoderSpec"/>, and each of Idrak's weight names (see <see cref="DecoderSpec"/>) is found in the
/// checkpoint. Registered by architecture name in <see cref="PretrainedArchitectures"/>. What a family can do beyond that
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

/// <summary>
/// The model families pretrained models are loaded as, by the architecture name in <c>config.json</c>
/// ("architectures": [...]). The library registers none: the families are plug-ins (Idrak.Text.Plugins,
/// Idrak.Vision.Plugins), and a vision-language model's vision part is a <see cref="VisionFamilies"/> registration. Add
/// others with <see cref="Register"/> (its <c>PretrainedFamilies.CommonArchitecture</c> makes one for families that share
/// the common configuration keys and checkpoint names). An architecture nobody registered is refused, naming the plug-in
/// packages.
/// </summary>
public static class PretrainedArchitectures
{
    // The registry holds families by the architecture names checkpoints give (one family may be registered under several),
    // so it is a table of names, not of kinds named by themselves.
    private static readonly SlotTable<string, IPretrainedArchitecture> Registry = new(nameof(PretrainedArchitectures), comparer: StringComparer.Ordinal,
        unguarded: "a family's parts (spec, tensor names, build) must agree with one another, so the app's and the library's cannot be mixed");

    /// <summary>Registers how to read the architecture <paramref name="name"/>, until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, IPretrainedArchitecture architecture)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(architecture);
        Registry.Register(name, architecture, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the architecture <paramref name="name"/> an app or plug-in registered; false when none did.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static IPretrainedArchitecture Get(ReadOnlySpan<char> name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No architecture '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with PretrainedArchitectures.Register, or reference Idrak.Text.Plugins (text models) or Idrak.Vision.Plugins (vision-language models) and call its RegisterAll().");

    /// <summary>The library's architecture <paramref name="name"/>, whatever an app registered over it (for an app's family to build on); null when the library has none.</summary>
    public static IPretrainedArchitecture? Default(string name) => Registry.Default(name);

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
    /// Store the projections in the packed format registered under this name (see <see cref="PackedWeightFormats.Names"/>:
    /// "int8", "int4", "bfloat16", or a format of your own added with <see cref="PackedWeightFormats.Register"/>), packed on the
    /// host as they are read. Takes precedence over <see cref="Int8"/>, <see cref="Int4"/> and <see cref="BFloat16"/>.
    /// </summary>
    public string? PackedFormatName { get; init; }

    /// <summary>
    /// Store the output head in the packed format registered under this name ("bfloat16", "int8", "int4", or one's own)
    /// instead of the projections' format; null packs it as the projections. Only for builds whose projections are packed.
    /// A tied head packed as bfloat16 is the model's only vocabulary table (the embedding reads its columns): with int8 or
    /// int4 projections that keeps the checkpoint's own next-token scores and lookups (exact for a bfloat16 checkpoint) in
    /// less memory than a packed head beside a separate bfloat16 table, at the cost of reading twice (int8) or about three
    /// times (int4) the head's bytes per generated token.
    /// </summary>
    public string? HeadFormatName { get; init; }

    /// <summary>
    /// Where a frozen build's separate token table lives (one beside a head of another format; a bfloat16 head is the
    /// table itself and stays with the head) on a device that can keep it in system memory: null takes
    /// <see cref="ComputeResources.LookupTables"/> (IDRAK_LOOKUP_TABLES; automatic by default).
    /// </summary>
    public LookupTablePlacement? LookupTables { get; init; }

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
