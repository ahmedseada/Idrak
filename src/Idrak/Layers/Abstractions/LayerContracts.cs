// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

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
/// One block of a decoder stack (attention and a feed-forward block over [batch, time, width]): images enter a decoder's
/// sequence before its first block, and training checkpoints block by block.
/// </summary>
public interface IDecoderBlock : ICheckpointable
{
    /// <summary>The width the block reads and writes.</summary>
    int Width { get; }
}

/// <summary>
/// An encoder whose token embedding table a head may read in place: a masked-language head tied to the word embeddings
/// (<c>EncoderHeadRequest.TiedTo</c>) reads this table instead of a decoder of its own.
/// </summary>
public interface IWordEmbeddings
{
    /// <summary>The token embedding table, [vocabulary, width].</summary>
    Embedding Words { get; }
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
/// A layer that numbers positions itself, from the start of each row (a sinusoidal position encoding), unaware of the
/// positions packed sequences or per-row starts give: a model holding one runs one sequence a row.
/// </summary>
public interface IPositionAware;
