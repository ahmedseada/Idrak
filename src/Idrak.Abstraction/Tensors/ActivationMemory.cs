// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// How training keeps activations in the decoder layers. Results that no backward step reads (projections before their
/// bias or rotation, head rearrangements, residual sums, the output projections) are released as soon as their last
/// forward use is done (<see cref="ReleaseUnused"/>, on by default: free memory, no extra work). With
/// <see cref="RecomputeFeedForward"/>, the feed-forward block's activation (act(gate) · up, the largest activation of a
/// block) is released after the forward pass too and recomputed from gate and up when the backward pass needs it: one
/// element-wise kernel per block for about a fifth less activation memory.
/// </summary>
/// <example>
/// <code>
/// using (ActivationMemory.Recompute())
/// {
///     loss = model.Forward(tokens) …;   // the feed-forward activations are recomputed in loss.Backward()
///     loss.Backward();
/// }
/// </code>
/// </example>
public static class ActivationMemory
{
    [ThreadStatic]
    private static int t_recompute;

    [ThreadStatic]
    private static int t_compress;

    /// <summary>Release results no backward step reads during the forward pass (default true; false keeps every result until the step ends).</summary>
    public static bool ReleaseUnused { get; set; } = true;

    /// <summary>Whether the feed-forward activations are recomputed in the backward pass on this thread (see <see cref="Recompute"/>).</summary>
    public static bool RecomputeFeedForward => t_recompute > 0;

    /// <summary>Recomputes the feed-forward activations in the backward pass on this thread until the returned scope is disposed.</summary>
    public static Scope Recompute()
    {
        t_recompute++;
        return new Scope(0);
    }

    /// <summary>Whether kept activations are held as bfloat16 until the backward pass on this thread (see <see cref="CompressToBFloat16"/>).</summary>
    public static bool BFloat16 => t_compress > 0;

    /// <summary>
    /// Holds the activations the backward pass reads as bfloat16 between the passes on this thread until the returned
    /// scope is disposed: half their memory, one pack and one unpack pass each. The decoder compresses the inputs of its
    /// tensor-core products and attention (queries, keys, values, the attention output, the normalized inputs of the
    /// projections, the feed-forward gate, up and activation), which those products round to bfloat16 anyway.
    /// </summary>
    public static Scope CompressToBFloat16()
    {
        t_compress++;
        return new Scope(1);
    }

    /// <summary>Holds the values as bfloat16 until the backward pass reads them, while <see cref="CompressToBFloat16"/> is in effect.</summary>
    internal static void Compress(params Tensor?[] tensors)
    {
        if (t_compress == 0 || !Autograd.IsEnabled)
        {
            return;
        }

        foreach (var tensor in tensors)
        {
            tensor?.CompressToBFloat16();
        }
    }

    /// <summary>Releases the values of results no backward step reads, while gradients are recorded.</summary>
    internal static void Release(params Tensor?[] tensors) => Release([], tensors);

    /// <summary>
    /// Releases the values of <paramref name="tensors"/> except where they share memory with <paramref name="kept"/>:
    /// a view (a reshape, or a permutation that only moves dimensions of size 1) of a result still needed shares its
    /// memory, which must stay.
    /// </summary>
    internal static void Release(ReadOnlySpan<Tensor> kept, params Tensor?[] tensors)
    {
        if (!ReleaseUnused || !Autograd.IsEnabled)
        {
            return;
        }

        foreach (var tensor in tensors)
        {
            if (tensor is null)
            {
                continue;
            }

            bool shared = false;
            foreach (var needed in kept)
            {
                shared |= ReferenceEquals(needed.Storage, tensor.Storage);
            }

            if (!shared)
            {
                tensor.DropValue();
            }
        }
    }

    /// <summary>Ends <see cref="Recompute"/> or <see cref="CompressToBFloat16"/>.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly int _kind;

        internal Scope(int kind) => _kind = kind;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_kind == 0)
            {
                t_recompute--;
            }
            else
            {
                t_compress--;
            }
        }
    }
}

/// <summary>Compresses a tensor (<see cref="ActivationMemory.CompressToBFloat16"/>) when disposed: at the end of the block that last reads it.</summary>
internal readonly struct CompressAfter(Tensor tensor) : IDisposable
{
    /// <inheritdoc />
    public void Dispose() => ActivationMemory.Compress(tensor);
}
