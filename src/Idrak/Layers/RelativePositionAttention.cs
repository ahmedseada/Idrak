// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Models;

namespace Idrak.Layers;

/// <summary>
/// Multi-head self-attention with a learned relative-position bias within a window (Shaw et al.'s relative position
/// representations as VITS's and Glow-TTS's text encoders use them, transformers' <c>VitsAttention</c>): the scores add
/// q · r(j − i), the output adds p · r'(j − i), offsets beyond ±window reading nothing. Rows [B · T, C] in and out, a
/// padded batch's mask keeping every padded key unattended. Speaking, the relative tables are sliced on the host per
/// length and cached; training, the embeddings are parameters gathered per length, so their gradients reach them.
/// </summary>
public sealed class RelativePositionAttention : Module
{
    private readonly int _heads, _window;
    private readonly Linear _q, _k, _v, _out;
    private readonly float[]? _relK, _relV;                                      // [2w + 1, d] each
    private readonly Tensor? _relKParameter, _relVParameter, _zeroRow;          // training: the relative embeddings [2w + 1, d] as parameters, and a zero row [1, d]
    private readonly ConcurrentDictionary<int, (float[] RelK, float[] RelV, float[] Offsets)> _byLength = new();

    /// <summary>
    /// Over the projections (each hidden → hidden), <paramref name="heads"/> heads, and the relative embeddings of
    /// <paramref name="window"/> (0: none): the keys' and values' [2 · window + 1, hidden / heads] as host values when the
    /// layer speaks, or as parameters (<paramref name="relativeKeys"/>, <paramref name="relativeValues"/>) when it trains.
    /// </summary>
    public RelativePositionAttention(Linear q, Linear k, Linear v, Linear output, int heads, int window, float[]? relativeKeyValues, float[]? relativeValueValues,
        Tensor? relativeKeys = null, Tensor? relativeValues = null)
    {
        _q = q ?? throw new ArgumentNullException(nameof(q));
        _k = k ?? throw new ArgumentNullException(nameof(k));
        _v = v ?? throw new ArgumentNullException(nameof(v));
        _out = output ?? throw new ArgumentNullException(nameof(output));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegative(window);
        int c = q.OutFeatures, d = c / heads;
        if (d * heads != c)
        {
            throw new ArgumentException($"The width {c} is not a multiple of {heads} heads.", nameof(heads));
        }

        if (window > 0)
        {
            int span = 2 * window + 1;
            if (relativeKeyValues is null || relativeValueValues is null || relativeKeyValues.Length != span * d || relativeValueValues.Length != span * d)
            {
                throw new ArgumentException($"A window of {window} takes relative keys and values of [{span}, {d}] each.", nameof(relativeKeyValues));
            }

            if ((relativeKeys is null) != (relativeValues is null))
            {
                throw new ArgumentException("The relative keys and values are both parameters, or neither.", nameof(relativeValues));
            }
        }

        _heads = heads;
        _window = window;
        _relK = window > 0 ? relativeKeyValues : null;
        _relV = window > 0 ? relativeValueValues : null;
        _relKParameter = window > 0 ? relativeKeys : null;
        _relVParameter = window > 0 ? relativeValues : null;
        _zeroRow = _relKParameter is null ? null : Tensor.PersistentZeros([1, d], _relKParameter.Device);
    }

    /// <summary>The heads.</summary>
    public int Heads => _heads;

    /// <summary>The relative-position window (0: none).</summary>
    public int Window => _window;

    /// <summary>The width.</summary>
    public int Hidden => _q.OutFeatures;

    /// <summary>
    /// From stored weights under <paramref name="prefix"/> (transformers' names): <c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>,
    /// <c>out_proj</c> (each [hidden, hidden] with a bias when stored), and for a <paramref name="window"/> above 0
    /// <c>emb_rel_k</c> and <c>emb_rel_v</c> [1, 2 · window + 1, hidden / heads] (parameters when <paramref name="stored"/> trains).
    /// </summary>
    public static RelativePositionAttention FromStored(StoredLayers stored, string prefix, int hidden, int heads, int window)
    {
        ArgumentNullException.ThrowIfNull(stored);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        int d = hidden / heads;
        if (d * heads != hidden)
        {
            throw new InvalidDataException($"hidden_size {hidden} is not a multiple of num_attention_heads {heads}.");
        }

        var q = stored.Linear($"{prefix}.q_proj", hidden, hidden);
        var k = stored.Linear($"{prefix}.k_proj", hidden, hidden);
        var v = stored.Linear($"{prefix}.v_proj", hidden, hidden);
        var o = stored.Linear($"{prefix}.out_proj", hidden, hidden);
        if (window == 0)
        {
            return new RelativePositionAttention(q, k, v, o, heads, 0, null, null);
        }

        int span = 2 * window + 1;
        var shape = stored.Shape($"{prefix}.emb_rel_k");
        if (shape.Length != 3 || shape[1] != span || shape[2] != d)
        {
            throw new InvalidDataException($"'{prefix}.emb_rel_k' is {Tensor.FormatShape(shape)}, the configuration says [1, {span}, {d}].");
        }

        var relK = stored.Read($"{prefix}.emb_rel_k");
        var relV = stored.Read($"{prefix}.emb_rel_v");
        if (!stored.Trainable)
        {
            return new RelativePositionAttention(q, k, v, o, heads, window, relK, relV);
        }

        // The relative embeddings as parameters, gathered per length (offsets beyond the window read the zero row).
        var relKParameter = stored.Parameter($"{prefix}.emb_rel_k", relK, [span, d], shape);
        var relVParameter = stored.Parameter($"{prefix}.emb_rel_v", relV, [span, d], stored.Shape($"{prefix}.emb_rel_v"));
        return new RelativePositionAttention(q, k, v, o, heads, window, relK, relV, relKParameter, relVParameter);
    }

    /// <summary>
    /// The attention over x [B · T, C], the rows of <paramref name="batch"/> sequences padded to <paramref name="t"/>
    /// positions (the mask null for one, unpadded): [B · T, C]. With a mask, no padded key is attended.
    /// </summary>
    public Tensor Forward(Tensor x, int batch, int t, PaddedSequences? mask)
    {
        ArgumentNullException.ThrowIfNull(x);
        int h = _heads, c = Hidden, d = c / h, bh = batch * h;
        if (x.Rank != 2 || x.Shape[0] != batch * t || x.Shape[1] != c)
        {
            throw new ArgumentException($"The attention takes [{batch} · {t}, {c}] rows, got {Tensor.FormatShape(x.Shape)}.", nameof(x));
        }

        float scaling = 1f / MathF.Sqrt(d);
        Tensor Heads(Tensor rows) => rows.Reshape(batch, t, h, d).Permute(0, 2, 1, 3).Reshape(bh, t, d);   // [B·h, T, d]
        var q = Heads(_q.Forward(x) * scaling);
        var k = Heads(_k.Forward(x));
        var v = Heads(_v.Forward(x));
        var scores = q.MatMul(k, transposeB: true);                             // [B·h, T, T]
        var tables = _relK is null ? default : _byLength.GetOrAdd(t, BuildTables);
        if (_relK is not null)
        {
            var relK = RelativeTable(_relKParameter, tables.RelK, tables.Offsets, t, d, x.Device);
            var relative = q.MatMul(relK, transposeB: true);                    // [B·h, T, 2T - 1]: query i against offset r - (T - 1)
            scores = scores + RelativeToAbsolute(relative, bh, t, x.Device);
        }

        if (mask is not null)
        {
            scores = scores + mask.Keys(h);                                     // no padded key is attended
        }

        var probs = scores.Softmax();
        var output = probs.MatMul(v);                                            // [B·h, T, d]
        if (_relV is not null)
        {
            output = output + AbsoluteToRelative(probs, bh, t, x.Device).MatMul(RelativeTable(_relVParameter, tables.RelV, tables.Offsets, t, d, x.Device));
        }

        return _out.Forward(output.Reshape(batch, h, t, d).Permute(0, 2, 1, 3).Reshape(batch * t, c));
    }

    // [h, T, 2T - 1] by offset (r = j - i + T - 1) to [h, T, T] by key, and back (zeros where a key falls outside), with
    // VITS's padding and reshapes: each value moves once, so the backward pass is slices and padding (no gather's
    // scatter over h·T² positions, and no index table sent to the device each call).
    private static Tensor RelativeToAbsolute(Tensor x, int h, int t, Device device)
    {
        if (t == 1)
        {
            return x;
        }

        var padded = Tensor.Concat([x, Tensor.Zeros([h, t, 1], device)], 2);                                    // [h, T, 2T]
        var flat = Tensor.Concat([padded.Reshape(h, 2 * t * t), Tensor.Zeros([h, t - 1], device)], 1);         // [h, 2T² + T - 1]
        return flat.Reshape(h, t + 1, 2 * t - 1).Narrow(1, 0, t).Narrow(2, t - 1, t);
    }

    private static Tensor AbsoluteToRelative(Tensor x, int h, int t, Device device)
    {
        if (t == 1)
        {
            return x;
        }

        var padded = Tensor.Concat([x, Tensor.Zeros([h, t, t - 1], device)], 2);                                // [h, T, 2T - 1]
        var flat = Tensor.Concat([Tensor.Zeros([h, t], device), padded.Reshape(h, t * (2 * t - 1))], 1);       // [h, 2T²]
        return flat.Reshape(h, t, 2 * t).Narrow(2, 1, 2 * t - 1);
    }

    // The relative embeddings [2T - 1, d] for T queries: the host's slices when speaking; when training, the parameters'
    // rows gathered (each offset's row, the zero row beyond the window), so their gradients reach the parameters.
    private Tensor RelativeTable(Tensor? parameter, float[] host, float[] offsets, int t, int d, Device device) =>
        parameter is null
            ? Tensor.From(host, [2 * t - 1, d], device)
            : Tensor.Concat([parameter, _zeroRow!], 0).EmbeddingLookup(Tensor.From(offsets, [offsets.Length], device));

    // The relative embeddings for T queries: offset r in [0, 2T - 2] is key j = i + r - (T - 1).
    private (float[] RelK, float[] RelV, float[] Offsets) BuildTables(int t)
    {
        int w = _window, d = Hidden / _heads, offsets = 2 * t - 1;
        var relK = new float[offsets * d];
        var relV = new float[offsets * d];
        var rows = new float[offsets];
        for (int r = 0; r < offsets; r++)
        {
            int offset = r - (t - 1);
            rows[r] = Math.Abs(offset) <= w ? offset + w : 2 * w + 1;
            if (Math.Abs(offset) <= w)
            {
                _relK!.AsSpan((offset + w) * d, d).CopyTo(relK.AsSpan(r * d, d));
                _relV!.AsSpan((offset + w) * d, d).CopyTo(relV.AsSpan(r * d, d));
            }
        }

        return (relK, relV, rows);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, 1, input.Shape[0], null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_q, _k, _v, _out];

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in base.Parameters())
        {
            yield return p;
        }

        if (_relKParameter is not null)
        {
            yield return _relKParameter;
            yield return _relVParameter!;
        }
    }

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _q.Dispose();
        _k.Dispose();
        _v.Dispose();
        _out.Dispose();
        _relKParameter?.Dispose();
        _relVParameter?.Dispose();
        _zeroRow?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"RelativePositionAttention({Hidden}, {_heads} heads, window {_window})";
}
