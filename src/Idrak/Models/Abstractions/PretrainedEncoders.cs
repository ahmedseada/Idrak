// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Models.Abstractions;

/// <summary>
/// What an encoder or encoder-decoder family builds its network from (<see cref="IPretrainedEncoderFamily.Build"/>,
/// <see cref="IPretrainedSeq2SeqFamily.Build"/>).
/// </summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by (config.json's "architectures").</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while building (do not keep it).</param>
/// <param name="Folder">The model folder, or null.</param>
/// <param name="Device">Where the network is built.</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record PretrainedCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Device Device, List<string> Notes);

/// <summary>
/// An encoder family (a model of token ids to hidden states, such as a masked-language-model encoder): how its
/// checkpoints become an <see cref="ISequenceEncoder"/> (ids, each sequence's length and optional token types to hidden
/// states [batch, time, width]), with the heads the checkpoint holds and the pooler its sentence vectors use. Registered
/// in <see cref="PretrainedEncoders"/> under the architecture name its checkpoints give; Idrak's
/// <c>PretrainedEncoderModel.Load</c> reads one. The library registers none: a family is a plug-in's.
/// </summary>
public interface IPretrainedEncoderFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>The encoder of the checkpoint (Idrak's <c>SequenceEncoder</c> built from its tensors, or one of the family's own).</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ISequenceEncoder Build(PretrainedCheckpoint checkpoint);

    /// <summary>
    /// The heads the checkpoint holds beside the encoder (a classifier, a masked-language-model head; made with
    /// <see cref="EncoderHeads"/>), none by default.
    /// </summary>
    IReadOnlyList<EncoderHead> Heads(PretrainedCheckpoint checkpoint, ISequenceEncoder encoder) => [];

    /// <summary>The pooler (a name in <see cref="SequencePoolers"/>) the family's sequence vectors use: "first" by default.</summary>
    string Pooler(JsonObject config) => SequencePoolers.First;
}

/// <summary>
/// The encoder families, by architecture name (config.json's "architectures"). The library registers none: a family is
/// an application of the encoder contract, registered by the plug-in that brings it; a checkpoint whose name has no
/// family is refused, naming this registry. Register one with <see cref="Register"/>.
/// </summary>
public static class PretrainedEncoders
{
    private static readonly SlotTable<string, IPretrainedEncoderFamily> Registry = new(nameof(PretrainedEncoders), comparer: StringComparer.Ordinal,
        unguarded: "a family's encoder, heads and pooler must agree with one another and with its checkpoints, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IPretrainedEncoderFamily.Name"/>, until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPretrainedEncoderFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/>; false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static IPretrainedEncoderFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message says how to register one.</exception>
    public static IPretrainedEncoderFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>Who registered the family <paramref name="name"/>; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Encoder family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "reference the plug-in package that brings the family (for example Idrak.Text.Plugins) and call its registration, or register it with "
            + "PretrainedEncoders.Register (the library registers none).");
}

/// <summary>
/// A text encoder-decoder family: how its checkpoints become an <see cref="ISeq2SeqModel"/> (the source encoded once,
/// the target decoded token by token with cross-attention to it). Registered in <see cref="PretrainedSeq2Seq"/> under the
/// architecture name its checkpoints give; Idrak's <c>PretrainedSeq2SeqModel.Load</c> reads one and
/// <c>Seq2SeqGenerator</c> generates with it. The library registers none: a family is a plug-in's.
/// </summary>
public interface IPretrainedSeq2SeqFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>
    /// The model of the checkpoint: Idrak's <c>Seq2SeqNetwork</c> (an <see cref="ISequenceEncoder"/> and a decoder of
    /// <c>DecoderBlock</c>s with cross-attention) built from its tensors, or one of the family's own.
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ISeq2SeqModel Build(PretrainedCheckpoint checkpoint);

    /// <summary>
    /// Whether the family's sources end with its end token (the model was trained so): generation then appends it when the
    /// tokenizer's encoding does not end with it (a tokenizer read from a SentencePiece model file adds none). False by default.
    /// </summary>
    bool EndsSource(JsonObject config) => false;
}

/// <summary>
/// The text encoder-decoder families, by architecture name (config.json's "architectures"). The library registers none:
/// a family is registered by the plug-in that brings it; a checkpoint whose name has no family is refused, naming this
/// registry. Register one with <see cref="Register"/>.
/// </summary>
public static class PretrainedSeq2Seq
{
    private static readonly SlotTable<string, IPretrainedSeq2SeqFamily> Registry = new(nameof(PretrainedSeq2Seq), comparer: StringComparer.Ordinal,
        unguarded: "a family's encoder and decoder must agree with one another and with its checkpoints, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IPretrainedSeq2SeqFamily.Name"/>, until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPretrainedSeq2SeqFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/>; false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static IPretrainedSeq2SeqFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message says how to register one.</exception>
    public static IPretrainedSeq2SeqFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>Who registered the family <paramref name="name"/>; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Encoder-decoder family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "reference the plug-in package that brings the family (for example Idrak.Text.Plugins) and call its registration, or register it with "
            + "PretrainedSeq2Seq.Register (the library registers none).");
}

/// <summary>What <see cref="EncoderHeads.Create"/> gives a head kind to make a head from.</summary>
/// <param name="Width">The encoder's width (the head's input).</param>
/// <param name="Outputs">The head's outputs: the labels of a classifier, the vocabulary of a masked-language-model head.</param>
/// <param name="Settings">The kind's own settings (see <see cref="EncoderHeads"/>).</param>
/// <param name="Device">Where its parameters live (the default device when null).</param>
/// <param name="Seed">The seed of random initial values.</param>
/// <param name="Weights">
/// Reads the head's tensors by their names in the kind ("dense.weight", "out.bias", ...) and shapes in Idrak's layout
/// (a linear layer's weight [in, out]), or null for random values; it returns null for a tensor it does not hold (an
/// optional bias then starts at zero).
/// </param>
public sealed record EncoderHeadRequest(int Width, int Outputs, JsonObject Settings, Device? Device = null, int Seed = 0,
    Func<string, IReadOnlyList<int>, float[]?>? Weights = null)
{
    /// <summary>
    /// The token embedding a head's output layer reads in place, transposed (a tied output: one copy of the table, which
    /// gradients reach as the embedding's), instead of an "out.weight" of its own; null (the default) for none. The
    /// "masked-lm" kind takes it ([outputs, width]); its "out.bias" is still read.
    /// </summary>
    public Embedding? TiedTo { get; init; }
}

/// <summary>
/// A head kind an encoder's checkpoint may hold, registered by its <see cref="IKind.Name"/> in <see cref="EncoderHeads"/>:
/// the settings it reads (<see cref="IConfigurableKind.Keys"/>), the tensors it reads by name in its own naming
/// (<see cref="Weights"/>, through <see cref="EncoderHeadRequest.Weights"/>), and the head it makes for a request.
/// </summary>
public interface IEncoderHeadKind : IKind<EncoderHeadRequest, EncoderHead>
{
    /// <summary>The names of the tensors it reads ("dense.weight", "out.bias", …), in Idrak's layout.</summary>
    IReadOnlyList<string> Weights { get; }
}

/// <summary>
/// The head kinds an encoder's checkpoint may hold, by name. The library registers three generic ones:
/// <list type="bullet">
/// <item>"sequence-classification": the pooled vector [batch, width] → logits [batch, outputs]: an optional dense layer
/// ("dense": true; "dense.weight" [width, width], "dense.bias") with "activation" ("tanh", the default, "gelu", "relu"),
/// then "out.weight" [width, outputs] and "out.bias".</item>
/// <item>"masked-lm": the states [batch, time, width] → logits [batch, time, outputs]: an optional transform
/// ("transform": true; "dense.weight", "dense.bias", "activation" ("gelu", the default, "gelu-tanh", "relu", "tanh"),
/// then a layer norm "norm.weight", "norm.bias" with "norm_epsilon", default 1e-5), then "out.weight"
/// [width, outputs] and "out.bias"; a tied head reads the token embedding in place instead
/// (<see cref="EncoderHeadRequest.TiedTo"/>).</item>
/// <item>"pooler": the pooled vector [batch, width] → [batch, width]: "dense.weight" [width, width] and "dense.bias",
/// then "activation" ("tanh", the default, "gelu", "gelu-tanh", "relu"); its outputs are the width.</item>
/// </list>
/// A family maps its checkpoint's names onto these in its <see cref="EncoderHeadRequest.Weights"/>, or registers a kind of
/// its own. A head is trained with its encoder, so the registry has no fallback.
/// </summary>
public static class EncoderHeads
{
    /// <summary>The sequence classifier: "sequence-classification".</summary>
    public const string SequenceClassification = "sequence-classification";

    /// <summary>The masked-language-model head: "masked-lm".</summary>
    public const string MaskedLanguageModel = "masked-lm";

    /// <summary>The pooled vector's projection (a dense layer and an activation, the vector a classifier reads): "pooler".</summary>
    public const string Pooler = "pooler";

    private static readonly KindTable<IEncoderHeadKind> Registry = BuiltIn();

    private static KindTable<IEncoderHeadKind> BuiltIn()
    {
        var table = new KindTable<IEncoderHeadKind>(nameof(EncoderHeads), "Encoder head", "add it with EncoderHeads.Register", comparer: StringComparer.Ordinal,
            unguarded: "a head is trained with its encoder and read from its checkpoint, so an app's kind and the library's cannot be mixed");
        table.RegisterDefault(new BuiltInKind(SequenceClassification, "the pooled vector to logits: an optional dense layer and activation, then the output layer",
            new OptionKeys("dense", "activation"), ["dense.weight", "dense.bias", "out.weight", "out.bias"], request => EncoderHead.Classification(request)));
        table.RegisterDefault(new BuiltInKind(MaskedLanguageModel, "the states to logits: an optional transform (dense, activation, layer norm), then the output layer",
            new OptionKeys("transform", "activation", "norm_epsilon"), ["dense.weight", "dense.bias", "norm.weight", "norm.bias", "out.weight", "out.bias"],
            request => EncoderHead.MaskedLanguageModel(request)));
        table.RegisterDefault(new BuiltInKind(Pooler, "the pooled vector's projection: a dense layer and an activation", new OptionKeys("activation"), ["dense.weight", "dense.bias"],
            request => EncoderHead.Pooler(request)));
        return table;
    }

    /// <summary>Registers <paramref name="kind"/> under its <see cref="IKind.Name"/>; under a library name it takes the library's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IEncoderHeadKind kind) => Registry.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's kind <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered head kinds.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The head kind <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message lists the registered ones.</exception>
    public static IEncoderHeadKind Get(ReadOnlySpan<char> name) => Registry.Find(name)
        ?? throw new NotSupportedException($"Encoder head '{name}' is not registered (registered: {string.Join(", ", Registry.Names)}); add it with EncoderHeads.Register.");

    /// <summary>A head of the kind <paramref name="name"/> for <paramref name="request"/>.</summary>
    /// <exception cref="NotSupportedException">The kind is not registered.</exception>
    /// <exception cref="ArgumentException">A setting is invalid; the message names it.</exception>
    public static EncoderHead Create(ReadOnlySpan<char> name, EncoderHeadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Get(name).Create(request);
    }

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IEncoderHeadKind? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    // A library head: its settings' keys and its tensors' names.
    private sealed class BuiltInKind(string name, string summary, OptionKeys keys, string[] weights, Func<EncoderHeadRequest, EncoderHead> create) : IEncoderHeadKind
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => keys;

        public IReadOnlyList<string> Weights => weights;

        public EncoderHead Create(EncoderHeadRequest request) => create(request);
    }
}
