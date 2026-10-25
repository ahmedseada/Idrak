// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

/// <summary>
/// A linear layer that takes an adapter (<see cref="ILinearAdapter"/>): <c>AddLora</c> and <c>AddDora</c> put one on the
/// layers they choose, <c>MergeLora</c> folds it into the weight, <c>DisableAdapters</c> lifts it off for a while, and a
/// pretrained model saves and loads it in peft's layout (<see cref="IPeftAdapter"/>).
/// </summary>
public interface IAdaptableLinear : ILinearLayer
{
    /// <summary>The layer's name in its parent (q, k, v, o, gate, up, down, head): what adapter targets match.</summary>
    string? Name { get; }

    /// <summary>Where the layer's weights are: an adapter added to it makes its tensors there.</summary>
    Device Device { get; }

    /// <summary>The adapter on the layer, or null; its parameters follow the layer's own.</summary>
    ILinearAdapter? Adapter { get; set; }

    /// <summary>Folds the adapter into the weight and removes it (the outputs stay the same); nothing without one.</summary>
    void MergeAdapter();
}

/// <summary>
/// A linear layer whose weight can be held in the built-in packed formats (<see cref="PackedFormat"/>): <c>QuantizeInt8</c>,
/// <c>QuantizeInt4</c>, <c>ToBFloat16</c>, <c>ToFloat32</c> and <c>DequantizeInt8</c> convert the layers they choose, and
/// <c>ModuleFiles</c> saves them packed and makes room for them before a load, counting the layers in model order.
/// </summary>
public interface IPackableLinear : ILinearLayer
{
    /// <summary>The layer's name in its parent (q, k, v, o, gate, up, down, head): what conversion targets match.</summary>
    string? Name { get; }

    /// <summary>The packed weights the layer holds (a built-in format, <see cref="IKernelPackedWeight"/>, or one of one's own), else null (float weights).</summary>
    IPackedWeight? PackedWeight { get; }

    /// <summary>Holds the weight in <paramref name="format"/>, converted from what the layer holds now; nothing when it holds that format already.</summary>
    void Pack(PackedFormat format);

    /// <summary>Holds float32 weights again (the rounding stays), trainable or not; nothing when the weights are float already.</summary>
    void Unpack(bool trainable);

    /// <summary>
    /// Makes room for weights in <paramref name="format"/> that a load then fills (allocated, not computed from the
    /// weights the layer holds, which it lets go of); nothing when it holds that format already.
    /// </summary>
    void Reserve(PackedFormat format);
}

/// <summary>
/// A layer that names the weights a checkpoint stores for it: each one's name after the layer's path ("weight",
/// "bias") and its tensor, in the order a save writes them. A pretrained model's stored tensors are these, under the
/// architecture's checkpoint names; a layer holding no weights of its own (a tied head) names none.
/// </summary>
public interface INamedWeights
{
    /// <summary>The layer's stored weights: each one's name after the layer's path and its tensor, in order.</summary>
    IEnumerable<(string Name, Tensor Tensor)> NamedWeights();
}

/// <summary>
/// What of a layer is staged on its device while it computes when the device keeps tensors in system memory
/// (<see cref="IMemoryOffload"/>). A layer that does not implement it has its parameters staged.
/// </summary>
public interface IOffloadableLayer
{
    /// <summary>
    /// The parameters the layer reads whole while it computes: staged for its pass, and marked cold while frozen. None
    /// for a table read by rows (an embedding's lookup reads only the rows it needs, which system memory serves without
    /// copying the table).
    /// </summary>
    IEnumerable<Tensor> StagedParameters();

    /// <summary>The fixed tensors besides its parameters that it reads whole (a linear layer's packed weights): staged with them and always marked cold.</summary>
    IEnumerable<Tensor> StagedBuffers();
}

/// <summary>
/// A module whose forward pass is recomputed in the backward pass instead of keeping its activations when a training
/// loop or a pipeline stage checkpoints (a transformer block: many activations, of which only its output is kept).
/// </summary>
public interface ICheckpointable;

/// <summary>
/// One block of a decoder stack (attention and a feed-forward block over [batch, time, width]): images enter a decoder's
/// sequence before its first block, and training checkpoints block by block.
/// </summary>
public interface IDecoderBlock : ICheckpointable
{
    /// <summary>The width the block reads and writes.</summary>
    int Width { get; }
}

/// <summary>The most positions a layer takes (a learned position table's rows, an attention layer's maximum).</summary>
public interface IPositionLimit
{
    /// <summary>The most positions the layer takes.</summary>
    int MaxPositions { get; }
}

/// <summary>
/// A layer that runs only some of its parts for each position (a mixture of experts: the router, the experts a token is
/// routed to, a shared expert): what one position's pass holds at once is measured over these, not over every part.
/// </summary>
public interface IRoutedLayer
{
    /// <summary>The parts one position runs through, each with how many of its like run (the experts per token).</summary>
    IEnumerable<(Module Part, int Times)> PartsPerPosition();
}

/// <summary>
/// A layer whose forward pass goes beyond the device on every call (a mixture's routing read back to the host, a
/// pipeline boundary sending to other stages): a decoding step through it is not recorded and replayed.
/// </summary>
public interface IHostRoundTrip;

/// <summary>
/// An attention layer as generation and packed training see it: what its key/value cache holds (heads, head size, the
/// window it keeps) and which batched attention it runs, prompts of different lengths decoded together (per-row starts)
/// and packed sequences. The generator and the packing read it from every layer that implements it, found once per model.
/// </summary>
public interface IAttentionCacheInfo
{
    /// <summary>The key and value heads the cache holds a row of per position.</summary>
    int KvHeads { get; }

    /// <summary>The size of each head.</summary>
    int HeadDim { get; }

    /// <summary>The positions a query sees back to, its own included; null when it sees the whole sequence.</summary>
    int? SlidingWindow { get; }

    /// <summary>Whether <paramref name="backend"/> attends for this layer from each row's own start (prompts of different lengths decoded in one batch).</summary>
    bool SupportsRows(Backend backend);

    /// <summary>Whether the device the layer's weights are on runs it over packed sequences (each position attending within its own sequence).</summary>
    bool SupportsPacking();

    /// <summary>
    /// Whether the device the layer's weights are on runs it over packed sequences by key ranges (images attended by a
    /// rule that is not causal, an image's rows seeing their whole image).
    /// </summary>
    bool SupportsSpans();
}

/// <summary>
/// A layer that numbers positions itself, from the start of each row (a sinusoidal position encoding), unaware of the
/// positions packed sequences or per-row starts give: a model holding one runs one sequence a row.
/// </summary>
public interface IPositionAware;
