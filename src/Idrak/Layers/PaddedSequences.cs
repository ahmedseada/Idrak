// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// A batch of sequences padded to one length: which positions of <c>[B, C, T]</c> (or rows of <c>[B·T, …]</c>) belong to a
/// sequence, as the masks layers multiply by and the key bias attention adds. Each mask is made once per channel count
/// (or head count) and kept until the batch is disposed; the tensors belong to no scope, so a mask first asked for inside
/// a checkpointed block outlives it, and <see cref="Dispose"/> releases them once the pass's backward step has read them.
/// Where a layer's signature takes a null batch, it stands for one sequence with nothing padded, and no mask is applied.
/// </summary>
public sealed class PaddedSequences : IDisposable
{
    private readonly int[] _lengths;
    private readonly Device _device;
    private readonly Dictionary<int, Tensor> _byChannels = [];
    private readonly Dictionary<int, Tensor> _keys = [];
    private Tensor? _rows;

    /// <summary>The batch of <paramref name="lengths"/>, each at most <paramref name="length"/> (the padded length), on <paramref name="device"/>.</summary>
    public PaddedSequences(IReadOnlyList<int> lengths, int length, Device device)
    {
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentNullException.ThrowIfNull(device);
        foreach (int l in lengths)
        {
            if (l < 0 || l > length)
            {
                throw new ArgumentOutOfRangeException(nameof(lengths), l, $"A sequence's length is between 0 and the padded length {length}.");
            }
        }

        _lengths = [.. lengths];
        Length = length;
        _device = device;
        Positions = _lengths.Sum();
    }

    /// <summary>The sequences.</summary>
    public int Batch => _lengths.Length;

    /// <summary>The padded length.</summary>
    public int Length { get; }

    /// <summary>Each sequence's length.</summary>
    public IReadOnlyList<int> Lengths => _lengths;

    /// <summary>The positions that belong to a sequence, over the batch.</summary>
    public int Positions { get; }

    /// <summary>The mask [B, channels, T]: 1 within each sequence's length, 0 after.</summary>
    public Tensor Of(int channels)
    {
        if (!_byChannels.TryGetValue(channels, out var mask))
        {
            var values = new float[Batch * channels * Length];
            for (int b = 0; b < Batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    values.AsSpan((b * channels + c) * Length, _lengths[b]).Fill(1f);
                }
            }

            _byChannels[channels] = mask = TensorScope.Untrack(Tensor.From(values, [Batch, channels, Length], _device));
        }

        return mask;
    }

    /// <summary>x [B, C, T] with its padding zeroed.</summary>
    public Tensor Apply(Tensor x) => x * Of(x.Shape[1]);

    /// <summary>The mask of the rows [B·T]: 1 for a position within its sequence, 0 for padding.</summary>
    public Tensor Rows
    {
        get
        {
            if (_rows is null)
            {
                var values = new float[Batch * Length];
                for (int b = 0; b < Batch; b++)
                {
                    values.AsSpan(b * Length, _lengths[b]).Fill(1f);
                }

                _rows = TensorScope.Untrack(Tensor.From(values, [Batch * Length], _device));
            }

            return _rows;
        }
    }

    /// <summary>x [B·T, features] with its padded rows zeroed (one row-wise scale).</summary>
    public Tensor ApplyRows(Tensor x) => x.GroupAffine(Rows, null, Batch * Length, x.Shape[1]);

    /// <summary>The attention bias [B·heads, T, T]: 0 where the key is within its sequence, -1e4 where it is padding (a masked fill before the softmax).</summary>
    public Tensor Keys(int heads)
    {
        if (!_keys.TryGetValue(heads, out var keys))
        {
            var values = new float[Batch * heads * Length * Length];
            for (int b = 0; b < Batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int i = 0; i < Length; i++)
                    {
                        values.AsSpan(((b * heads + h) * Length + i) * Length + _lengths[b], Length - _lengths[b]).Fill(-1e4f);
                    }
                }
            }

            _keys[heads] = keys = TensorScope.Untrack(Tensor.From(values, [Batch * heads, Length, Length], _device));
        }

        return keys;
    }

    /// <summary>Releases the masks: after the pass's backward step, which reads them.</summary>
    public void Dispose()
    {
        foreach (var mask in _byChannels.Values)
        {
            mask.Dispose();
        }

        foreach (var keys in _keys.Values)
        {
            keys.Dispose();
        }

        _rows?.Dispose();
        _byChannels.Clear();
        _keys.Clear();
        _rows = null;
    }
}
