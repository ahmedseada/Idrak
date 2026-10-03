// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>How token ids are packed in a <see cref="TokenFileSource"/> file (little-endian).</summary>
public enum TokenType
{
    /// <summary>16-bit unsigned ids (vocabularies up to 65,536), the usual packing of pretraining token files.</summary>
    UInt16,

    /// <summary>32-bit signed ids.</summary>
    Int32,

    /// <summary>32-bit unsigned ids.</summary>
    UInt32,
}

/// <summary>
/// Packed token ids for language-model pretraining, memory-mapped so files larger than memory work, cut into windows of
/// <see cref="Length"/> tokens: the features are tokens [s, s + length) and the targets the next tokens [s + 1, s + length
/// + 1), for window starts s = 0, stride, 2 x stride, .... Ids become floats (exact up to 16,777,216), as the
/// embedding layers and <see cref="Losses.SparseCrossEntropy(Tensor, Tensor, float)"/> take them. A .npy file of one
/// dimension is read with its own element type.
/// </summary>
/// <example>
/// <code>
/// using var tokens = new TokenFileSource("train.bin", length: 256);
/// var loader = new DataLoader(tokens, batchSize: 32, shuffle: true, seed: 1);
/// </code>
/// </example>
public sealed class TokenFileSource : ISampleSource, IDisposable
{
    private readonly MappedFile _file;
    private readonly ElementType _type;
    private readonly long _offset;
    private readonly int[] _shape;

    /// <summary>Maps <paramref name="path"/>.</summary>
    /// <param name="path">The token file: ids back to back, or a one-dimensional .npy array.</param>
    /// <param name="length">Tokens per window (the sequence length).</param>
    /// <param name="stride">Tokens between window starts; defaults to <paramref name="length"/> (windows do not overlap).</param>
    /// <param name="type">The packing of a raw file (a .npy file says its own).</param>
    public TokenFileSource(string path, int length, int? stride = null, TokenType type = TokenType.UInt16)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Stride = stride ?? length;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Stride, nameof(stride));
        _file = new MappedFile(path);
        try
        {
            if (NpyHeader.IsNpy(_file.Bytes(0, (int)Math.Min(_file.Length, 6))))
            {
                var header = NpyHeader.Read(_file);
                if (header.Shape.Length != 1 || header.Type is not (ElementType.UInt8 or ElementType.UInt16 or ElementType.Int16 or ElementType.Int32 or ElementType.UInt32 or ElementType.Int64))
                {
                    throw new InvalidDataException($"{path}: token ids need a one-dimensional integer array, not {header.Type} [{string.Join(", ", header.Shape)}].");
                }

                (_type, _offset, TokenCount) = (header.Type, header.DataOffset, header.Shape[0]);
            }
            else
            {
                _type = type switch { TokenType.Int32 => ElementType.Int32, TokenType.UInt32 => ElementType.UInt32, _ => ElementType.UInt16 };
                int size = Elements.Size(_type);
                if (_file.Length % size != 0)
                {
                    throw new InvalidDataException($"{path}: {_file.Length} bytes is not a whole number of {type} token ids.");
                }

                TokenCount = _file.Length / size;
            }

            if (TokenCount < length + 1)
            {
                throw new InvalidDataException($"{path} has {TokenCount} tokens; a window of {length} needs {length + 1}.");
            }
        }
        catch
        {
            _file.Dispose();
            throw;
        }

        Length = length;
        _shape = [length];
        Count = (int)Math.Min(int.MaxValue, (TokenCount - length - 1) / Stride + 1);
    }

    /// <summary>Tokens in the file.</summary>
    public long TokenCount { get; }

    /// <summary>Tokens per window.</summary>
    public int Length { get; }

    /// <summary>Tokens between window starts.</summary>
    public int Stride { get; }

    /// <summary>Windows: every start s with s + length + 1 tokens in the file.</summary>
    public int Count { get; }

    /// <summary>[length].</summary>
    public IReadOnlyList<int> FeatureShape => _shape;

    /// <summary>[length]: the next token at every position.</summary>
    public IReadOnlyList<int> TargetShape => _shape;

    /// <inheritdoc />
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        if ((uint)index >= (uint)Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        int size = Elements.Size(_type);
        var bytes = _file.Bytes(_offset + (long)index * Stride * size, (Length + 1) * size);
        Elements.ToFloat(bytes[..(Length * size)], _type, features[..Length]);
        Elements.ToFloat(bytes[size..], _type, targets[..Length]);
    }

    /// <summary>Unmaps the file.</summary>
    public void Dispose() => _file.Dispose();
}

/// <summary>
/// NumPy .npy arrays as samples, memory-mapped so large arrays are not loaded at once: the first dimension counts the
/// samples and the rest is each sample's shape (e.g. images [N, 28, 28] give features [28, 28]). The targets come from
/// a second file with as many samples: its values ([N] gives targets [1]), or one-hot rows when <c>classes</c> is given
/// and it holds class indices. Every little-endian number type and booleans are read, as float32.
/// </summary>
public sealed class NpySource : ISampleSource, IDisposable
{
    private readonly MappedFile _features;
    private readonly MappedFile? _targets;
    private readonly NpyHeader _featureHeader;
    private readonly NpyHeader? _targetHeader;
    private readonly int[] _featureShape, _targetShape;
    private readonly int _featureSize, _targetSize;
    private readonly int? _classes;

    /// <summary>Maps the arrays.</summary>
    /// <param name="features">The features: [N, ...].</param>
    /// <param name="targets">The targets: [N] or [N, ...]; null for none (targets of shape [0]).</param>
    /// <param name="classes">Read [N] targets as class indices in [0, classes) and give one-hot rows.</param>
    public NpySource(string features, string? targets = null, int? classes = null)
    {
        _features = new MappedFile(features);
        try
        {
            _featureHeader = NpyHeader.Read(_features);
            if (_featureHeader.Shape.Length == 0)
            {
                throw new InvalidDataException($"{features} holds a single value, not samples.");
            }

            Count = _featureHeader.Shape[0];
            _featureShape = _featureHeader.Shape.Length == 1 ? [1] : _featureHeader.Shape[1..];
            if (targets is not null)
            {
                _targets = new MappedFile(targets);
                _targetHeader = NpyHeader.Read(_targets);
                if (_targetHeader.Shape.Length == 0 || _targetHeader.Shape[0] != Count)
                {
                    throw new InvalidDataException($"{targets} has [{string.Join(", ", _targetHeader.Shape)}]: not {Count} samples, as {features} has.");
                }

                if (classes is { } c)
                {
                    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(c, nameof(classes));
                    if (_targetHeader.Shape.Length != 1)
                    {
                        throw new InvalidDataException($"{targets}: class indices need shape [N], not [{string.Join(", ", _targetHeader.Shape)}].");
                    }

                    _classes = c;
                }

                _targetShape = _classes is { } k ? [k] : _targetHeader.Shape.Length == 1 ? [1] : _targetHeader.Shape[1..];
            }
            else
            {
                _targetShape = [0];
            }
        }
        catch
        {
            Dispose();
            throw;
        }

        _featureSize = SampleSourceExtensions.Size(_featureShape);
        _targetSize = _classes is null ? SampleSourceExtensions.Size(_targetShape) : 1;
    }

    /// <summary>Samples: the first dimension of the arrays.</summary>
    public int Count { get; }

    /// <inheritdoc />
    public IReadOnlyList<int> FeatureShape => _featureShape;

    /// <inheritdoc />
    public IReadOnlyList<int> TargetShape => _targetShape;

    /// <inheritdoc />
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        if ((uint)index >= (uint)Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        int size = Elements.Size(_featureHeader.Type);
        Elements.ToFloat(_features.Bytes(_featureHeader.DataOffset + (long)index * _featureSize * size, _featureSize * size), _featureHeader.Type, features[.._featureSize]);
        if (_targets is null || _targetHeader is null)
        {
            return;
        }

        size = Elements.Size(_targetHeader.Type);
        var bytes = _targets.Bytes(_targetHeader.DataOffset + (long)index * _targetSize * size, _targetSize * size);
        if (_classes is not { } classes)
        {
            Elements.ToFloat(bytes, _targetHeader.Type, targets[.._targetSize]);
            return;
        }

        Span<float> label = stackalloc float[1];
        Elements.ToFloat(bytes, _targetHeader.Type, label);
        int c = (int)label[0];
        if (c != label[0] || (uint)c >= (uint)classes)
        {
            throw new InvalidDataException($"{_targets.Path}: target {label[0]} of sample {index} is not a class index in [0, {classes}).");
        }

        targets.Clear();
        targets[c] = 1f;
    }

    /// <summary>Unmaps the files.</summary>
    public void Dispose()
    {
        _features.Dispose();
        _targets?.Dispose();
    }
}
