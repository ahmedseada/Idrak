// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// An encoder of token sequences: token ids [batch, time] (with each sequence's length, its padding after it, and
/// optional token types) in, hidden states [batch, time, <see cref="Width"/>] out, every position seeing the whole of its
/// sequence and nothing of its padding. A sequence's states are the same alone and in a padded batch (within float
/// rounding); the states at padded positions are of no meaning. Idrak's <c>SequenceEncoder</c> is the library's encoder
/// built from layers; a pretrained family (Idrak's <c>PretrainedEncoders</c>) builds one from its checkpoint.
/// </summary>
public interface ISequenceEncoder : IDisposable
{
    /// <summary>The hidden states' width.</summary>
    int Width { get; }

    /// <summary>The token ids it embeds: [0, vocabulary).</summary>
    int Vocabulary { get; }

    /// <summary>The token types it embeds, [0, types); 0 when it takes none.</summary>
    int TypeVocabulary { get; }

    /// <summary>The longest sequence it encodes.</summary>
    int MaxPositions { get; }

    /// <summary>Where it runs.</summary>
    Device Device { get; }

    /// <summary>Its layers (parameters to train, save and move).</summary>
    Module Network { get; }

    /// <summary>
    /// The hidden states [batch, time, width] of <paramref name="ids"/> [batch, time]: sequence b is its first
    /// <paramref name="lengths"/>[b] positions (every position when null), and attends to them only. Gradients are
    /// recorded when autograd is on.
    /// </summary>
    /// <param name="ids">Token ids, [batch, time], on <see cref="Device"/>.</param>
    /// <param name="lengths">Each sequence's length (0 to time), or null when none is padded.</param>
    /// <param name="tokenTypes">Each token's type, [batch, time] (type 0 for every token when null), for an encoder with <see cref="TypeVocabulary"/> above 0.</param>
    /// <exception cref="ArgumentException">The shapes do not fit, a length is outside 0 to time, or token types are given to an encoder that takes none.</exception>
    Tensor Encode(Tensor ids, IReadOnlyList<int>? lengths = null, Tensor? tokenTypes = null);
}

/// <summary>
/// How a padded batch of hidden states [batch, time, width] becomes one vector per sequence [batch, width], looking at
/// each sequence's own positions only (registered by name in <see cref="SequencePoolers"/>: "first", "mean", "max"). An
/// empty sequence (length 0) pools its first position. Pooling is differentiable.
/// </summary>
public interface ISequencePooler
{
    /// <summary>Its name in <see cref="SequencePoolers"/>.</summary>
    string Name { get; }

    /// <summary>One vector per sequence, [batch, width], from <paramref name="hidden"/> [batch, time, width].</summary>
    /// <param name="hidden">The states, [batch, time, width].</param>
    /// <param name="lengths">Each sequence's length (0 to time), or null for every position.</param>
    /// <exception cref="ArgumentException">The states are not [batch, time, width], or the lengths do not fit them.</exception>
    Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths);
}

/// <summary>
/// The sequence poolers, by name. The library registers "first" (the first position's state), "mean" (the average of
/// each sequence's states: Idrak.Text's <c>TextEncoder</c> default) and "max" (each feature's largest value over the
/// sequence); a family or an app registers others. A pooled vector feeds a head or an index trained with it, so a pooler
/// is used whole: the registry has no fallback.
/// </summary>
public static class SequencePoolers
{
    /// <summary>The first position's state: "first".</summary>
    public const string First = "first";

    /// <summary>The average of the sequence's states: "mean".</summary>
    public const string Mean = "mean";

    /// <summary>Each feature's largest value over the sequence: "max".</summary>
    public const string Max = "max";

    private static readonly SlotTable<string, ISequencePooler> Registry = BuiltIn();

    private static SlotTable<string, ISequencePooler> BuiltIn()
    {
        var table = new SlotTable<string, ISequencePooler>(nameof(SequencePoolers), comparer: StringComparer.Ordinal,
            unguarded: "a pooled vector feeds a head or an index trained with it, so an app's pooler and the library's cannot be mixed");
        table.RegisterDefault(First, new FirstPooler());
        table.RegisterDefault(Mean, new MeanPooler());
        table.RegisterDefault(Max, new MaxPooler());
        return table;
    }

    /// <summary>Registers <paramref name="pooler"/> under its <see cref="ISequencePooler.Name"/>; under a library name it takes the library's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISequencePooler pooler)
    {
        ArgumentNullException.ThrowIfNull(pooler);
        ArgumentException.ThrowIfNullOrEmpty(pooler.Name);
        Registry.Register(pooler.Name, pooler, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's pooler <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered poolers.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The pooler <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message lists the registered ones.</exception>
    public static ISequencePooler Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(SequencePoolers), name, Registry.Keys,
            $"Sequence pooler '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with SequencePoolers.Register.");

    /// <summary>The library's pooler <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISequencePooler? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the pooler <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// The batch, time and width of <paramref name="hidden"/> [batch, time, width], checked against
    /// <paramref name="lengths"/> (one per sequence, 0 to time): what a pooler of one's own checks first.
    /// </summary>
    /// <exception cref="ArgumentException">They do not fit.</exception>
    public static (int Batch, int Time, int Width) Check(Tensor hidden, IReadOnlyList<int>? lengths)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        if (hidden.Rank != 3 || hidden.Shape[1] == 0)
        {
            throw new ArgumentException($"A pooler takes hidden states [batch, time, width] with at least one position, got {Tensor.FormatShape(hidden.Shape)}.", nameof(hidden));
        }

        int n = hidden.Shape[0], t = hidden.Shape[1];
        if (lengths is not null)
        {
            if (lengths.Count != n)
            {
                throw new ArgumentException($"{lengths.Count} lengths for {n} sequences.", nameof(lengths));
            }

            for (int b = 0; b < n; b++)
            {
                if (lengths[b] < 0 || lengths[b] > t)
                {
                    throw new ArgumentException($"Sequence {b}'s length {lengths[b]} is outside 0 to {t}, the states' positions.", nameof(lengths));
                }
            }
        }

        return (n, t, hidden.Shape[2]);
    }

    // The first position: [batch, width].
    private sealed class FirstPooler : ISequencePooler
    {
        public string Name => First;

        public Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths)
        {
            var (n, _, d) = Check(hidden, lengths);
            return hidden.Narrow(1, 0, 1).Reshape(n, d);
        }
    }

    // A weighted sum over time with each real position weighing 1 / length (an empty sequence: its first position), as one
    // batched product [batch, 1, time] · [batch, time, width]; the same arithmetic as TextEncoder's average always had.
    private sealed class MeanPooler : ISequencePooler
    {
        public string Name => Mean;

        public Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths)
        {
            var (n, t, d) = Check(hidden, lengths);
            float[] weights = ArrayPool<float>.Shared.Rent(n * t);
            Tensor mean;
            try
            {
                for (int b = 0; b < n; b++)
                {
                    int length = lengths?[b] ?? t;
                    var row = weights.AsSpan(b * t, t);
                    row.Clear();
                    row[..length].Fill(1f / Math.Max(length, 1));
                    if (length == 0)
                    {
                        row[0] = 1f;                                        // an empty sequence averages its first (padding) position
                    }
                }

                mean = Tensor.From(weights.AsSpan(0, n * t), [n, 1, t], hidden.Device);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(weights);
            }

            return mean.MatMul(hidden).Reshape(n, d);
        }
    }

    // Each feature's largest value over the sequence's positions: the padded rows pushed far below any state (a row-wise
    // affine, x · 0 - 1e30), then one max pool over time of [batch, width, 1, time].
    private sealed class MaxPooler : ISequencePooler
    {
        public string Name => Max;

        public Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths)
        {
            var (n, t, d) = Check(hidden, lengths);
            var x = hidden;
            if (lengths is not null && lengths.Any(l => l < t))
            {
                var scale = new float[n * t];
                var shift = new float[n * t];
                for (int b = 0; b < n; b++)
                {
                    int length = Math.Max(lengths[b], 1);                   // an empty sequence: its first position
                    scale.AsSpan(b * t, length).Fill(1f);
                    shift.AsSpan(b * t + length, t - length).Fill(-1e30f);
                }

                var scales = Tensor.From(scale, [n * t], hidden.Device);
                var shifts = Tensor.From(shift, [n * t], hidden.Device);
                x = hidden.GroupAffine(scales, shifts, n * t, d);
            }

            var columns = x.Permute(0, 2, 1).Reshape(n, d, 1, t);
            return columns.MaxPool(new Devices.ConvGeometry(n, d, 1, t, 1, t, 1, t, 0, 0)).Reshape(n, d);
        }
    }
}

/// <summary>
/// A text encoder-decoder (sequence to sequence) as generation drives it: the source encoded once
/// (<see cref="Encoder"/>), then the target decoded token by token with cross-attention to the encoder's states, its
/// self-attention through a key/value cache (<see cref="DecodingContext"/>) and its cross-attention over the memory
/// projected once per sequence (<see cref="SetMemory"/>). Idrak's <c>Seq2SeqNetwork</c> is the library's model built from
/// an encoder and a decoder of layers; a pretrained family (Idrak's <c>PretrainedSeq2Seq</c>) builds one from its
/// checkpoint, and Idrak's <c>Seq2SeqGenerator</c> generates text with any of them.
/// </summary>
public interface ISeq2SeqModel : IDisposable
{
    /// <summary>The source's encoder: ids [batch, time] to the memory [batch, time, width].</summary>
    ISequenceEncoder Encoder { get; }

    /// <summary>The decoder's vocabulary (the logits' last dimension).</summary>
    int Vocabulary { get; }

    /// <summary>The token decoding starts from.</summary>
    int DecoderStartToken { get; }

    /// <summary>The token that ends a target.</summary>
    int EndToken { get; }

    /// <summary>The longest target the decoder takes, the start token included.</summary>
    int MaxTargetPositions { get; }

    /// <summary>Where it runs.</summary>
    Device Device { get; }

    /// <summary>Its layers, the encoder's and the decoder's (parameters to train, save and move).</summary>
    Module Network { get; }

    /// <summary>
    /// The logits [batch, time, vocabulary] of target ids [batch, time] (the start token first, each position seeing the
    /// ones before it) over <paramref name="memory"/> [batch, sourceTime, width], each sequence attending to its first
    /// <paramref name="memoryLengths"/>[b] memory positions (all when null): the whole target at once, as training reads it.
    /// </summary>
    Tensor Decode(Tensor ids, Tensor memory, IReadOnlyList<int>? memoryLengths = null);

    /// <summary>
    /// Gives <paramref name="context"/> the memory for every cross-attention (projected once, read by every step); a memory
    /// of one sequence serves every row of the context.
    /// </summary>
    void SetMemory(Tensor memory, DecodingContext context, IReadOnlyList<int>? memoryLengths = null);

    /// <summary>
    /// The logits of the new target positions <paramref name="ids"/> [batch, steps] through the context's cache (one
    /// step: [batch, 1]); with <see cref="DecodingContext.LastPositionOnly"/>, of the last one only. The same values as
    /// <see cref="Decode"/> at those positions, within float rounding.
    /// </summary>
    Tensor DecodeCached(Tensor ids, DecodingContext context);
}
