// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// Another store's tensors under the names a reader asks for (family-neutral: a vision part's names without its
/// checkpoint's prefix, one model's layer names read from another's checkpoint), and, when asked to, a weight stored
/// weight-normed read folded: a <c>….weight</c> the store lacks, held as gains and directions (PyTorch's
/// <c>weight_g</c> and <c>weight_v</c>, or <c>parametrizations.weight.original0</c> and <c>original1</c>), reads as
/// <see cref="StoredWeights.FoldWeightNorm"/> folds them. A tensor read under its stored name keeps everything the store
/// gives (its format, bfloat16 copies, transposed reads, read time and bytes); nothing is read until asked for.
/// </summary>
/// <example>
/// <code>
/// // Only the vision tower's tensors, without the "vision_tower." prefix:
/// var vision = new RenamedTensorStore(checkpoint, names.ToDictionary(n => n["vision_tower.".Length..], n => n));
/// // Every name through a function, weight-normed weights folded, the checkpoint left open:
/// var renamed = new RenamedTensorStore(checkpoint, name => name.Replace("decoder.", "dec."), foldWeightNorm: true, ownsInner: false);
/// </code>
/// </example>
public sealed class RenamedTensorStore : IBFloat16TensorStore
{
    private const string Weight = ".weight";
    private readonly ITensorStore _inner;
    private readonly IReadOnlyDictionary<string, string>? _names;
    private readonly Func<string, string>? _rename;
    private readonly bool _foldWeightNorm;
    private readonly bool _ownsInner;

    /// <summary>
    /// Shows the tensors <paramref name="names"/> lists (shown name → stored name) and only those.
    /// </summary>
    /// <param name="inner">The store the tensors are read from.</param>
    /// <param name="names">Each shown name with its name in <paramref name="inner"/>.</param>
    /// <param name="foldWeightNorm">Whether a <c>….weight</c> stored weight-normed reads folded.</param>
    /// <param name="ownsInner">Whether disposing this store disposes <paramref name="inner"/>.</param>
    public RenamedTensorStore(ITensorStore inner, IReadOnlyDictionary<string, string> names, bool foldWeightNorm = false, bool ownsInner = true)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(names);
        (_inner, _names, _foldWeightNorm, _ownsInner) = (inner, names, foldWeightNorm, ownsInner);
    }

    /// <summary>
    /// Shows every name through <paramref name="rename"/> (shown name → stored name; a name it leaves as it is passes
    /// through). <see cref="Names"/> are then the inner store's, since a function cannot be listed.
    /// </summary>
    /// <param name="inner">The store the tensors are read from.</param>
    /// <param name="rename">The stored name of a shown name.</param>
    /// <param name="foldWeightNorm">Whether a <c>….weight</c> stored weight-normed reads folded.</param>
    /// <param name="ownsInner">Whether disposing this store disposes <paramref name="inner"/>.</param>
    public RenamedTensorStore(ITensorStore inner, Func<string, string> rename, bool foldWeightNorm = false, bool ownsInner = true)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(rename);
        (_inner, _rename, _foldWeightNorm, _ownsInner) = (inner, rename, foldWeightNorm, ownsInner);
    }

    /// <summary>The store the tensors are read from.</summary>
    public ITensorStore Inner => _inner;

    /// <inheritdoc />
    public IEnumerable<string> Names => _names?.Keys ?? _inner.Names;

    /// <inheritdoc />
    public bool Contains(string name)
    {
        if (_names is not null)
        {
            return _names.ContainsKey(name);
        }

        string stored = _rename!(name);
        return _inner.Contains(stored) || Folded(name, stored) is not null;
    }

    /// <inheritdoc />
    public int[] ShapeOf(string name) => Resolve(name) switch
    {
        (_, { } norm) => _inner.ShapeOf(norm.Directions),
        var (stored, _) => _inner.ShapeOf(stored),
    };

    /// <inheritdoc />
    public float[] Read(string name) => Resolve(name) switch
    {
        (_, { } norm) => StoredWeights.FoldWeightNorm(_inner.Read(norm.Gains), _inner.Read(norm.Directions), _inner.ShapeOf(norm.Directions)),
        var (stored, _) => _inner.Read(stored),
    };

    /// <inheritdoc />
    /// <remarks>Two stored tensors are compared by the inner store (piece by piece where it can); a folded weight is computed first.</remarks>
    public bool SameValues(string first, string second) =>
        Resolve(first) is (var a, null) && Resolve(second) is (var b, null) ? _inner.SameValues(a, b)
        : ShapeOf(first).AsSpan().SequenceEqual(ShapeOf(second)) && Read(first).AsSpan().SequenceEqual(Read(second));

    /// <inheritdoc />
    public float[] ReadTransposed(string name)
    {
        var (stored, norm) = Resolve(name);
        if (norm is null)
        {
            return _inner.ReadTransposed(stored);
        }

        var shape = _inner.ShapeOf(norm.Value.Directions);
        return HostParallel.Transpose(Read(name), shape[0], shape.Length > 1 ? shape[1] : 1);
    }

    /// <inheritdoc />
    public WeightFormat? FormatOf(string name) => Resolve(name) is (var stored, null) ? _inner.FormatOf(stored) : null;   // a folded weight is computed in float32

    /// <inheritdoc />
    public bool ReadBFloat16(string name, bool transposed, Span<ushort> destination, int stride) =>
        _inner is IBFloat16TensorStore inner && Resolve(name) is (var stored, null) && inner.ReadBFloat16(stored, transposed, destination, stride);

    /// <inheritdoc />
    public TimeSpan? ReadTime => _inner.ReadTime;

    /// <inheritdoc />
    public long? BytesRead => _inner.BytesRead;

    /// <summary>Disposes the inner store when this store owns it.</summary>
    public void Dispose()
    {
        if (_ownsInner)
        {
            _inner.Dispose();
        }
    }

    // The stored name of a shown one.
    private string Stored(string name) =>
        _names is null ? _rename!(name)
        : _names.TryGetValue(name, out var stored) ? stored
        : throw new KeyNotFoundException($"No tensor '{name}' among the {_names.Count} this store shows.");

    // The stored name of a shown one, with the gains and directions to fold when it is a "….weight" the inner store lacks
    // but holds weight-normed. A name the function maps to nothing stored is refused here; one a map lists is left to the
    // inner store to refuse, as it would.
    private (string Stored, (string Gains, string Directions)? Norm) Resolve(string name)
    {
        string stored = Stored(name);
        if (_inner.Contains(stored))
        {
            return (stored, null);
        }

        var norm = Folded(name, stored);
        return norm is null && _names is null ? throw Missing(name, stored) : (stored, norm);
    }

    // The gains and directions of a "….weight" held weight-normed (when folding), else null.
    private (string Gains, string Directions)? Folded(string name, string stored) =>
        _foldWeightNorm && name.EndsWith(Weight, StringComparison.Ordinal) && stored.EndsWith(Weight, StringComparison.Ordinal)
            ? StoredWeights.WeightNormNames(_inner, stored[..^Weight.Length])
            : null;

    private static InvalidDataException Missing(string name, string stored) =>
        new($"The checkpoint has no tensor '{stored}' (read as '{name}').");
}
