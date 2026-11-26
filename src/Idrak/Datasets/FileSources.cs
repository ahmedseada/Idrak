// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Datasets;

/// <summary>
/// The record beside a token file (<c>tokens.bin.json</c>, format <c>idrak-tokens/1</c>) that says how its ids are stored:
/// <see cref="TokenFileWriter.Write"/> writes it, <see cref="TokenFileSource"/> reads it. Ids are stored as little-endian
/// float32 values (exact up to 16,777,216); a record naming any other stored type is refused (its type is not on the new
/// design yet).
/// </summary>
public static class TokenFileRecord
{
    /// <summary>The record's format name.</summary>
    public const string Format = "idrak-tokens/1";

    // The stored type the record names for float32 ids, the one read and written.
    internal const string Float32 = "float32";

    /// <summary>Where the record of <paramref name="tokenFile"/> is: beside it, its name with ".json" added.</summary>
    public static string PathOf(string tokenFile)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenFile);
        return tokenFile + ".json";
    }

    /// <summary>
    /// Whether the record beside <paramref name="tokenFile"/> says its ids are float32; false when there is no record (no
    /// file, or a .json of another format).
    /// </summary>
    /// <exception cref="InvalidDataException">The record is not JSON, names no stored type, or names one not on the new design yet.</exception>
    public static bool ReadType(string tokenFile)
    {
        string record = PathOf(tokenFile);
        if (!File.Exists(record))
        {
            return false;
        }

        JsonObject? json;
        try
        {
            json = JsonNode.Parse(File.ReadAllText(record)) as JsonObject;
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{record}: the token file's record is not JSON ({e.Message}).", e);
        }

        if (json?["format"] is not JsonValue format || format.GetValueKind() != JsonValueKind.String || format.GetValue<string>() != Format)
        {
            return false;
        }

        if (json["type"] is not JsonValue type || type.GetValueKind() != JsonValueKind.String)
        {
            throw new InvalidDataException($"{record}: the record names no \"type\".");
        }

        return type.GetValue<string>() == Float32 ? true
            : throw new InvalidDataException($"{record}: the ids are stored as {type.GetValue<string>()}: type not on the new design yet.");
    }
}

/// <summary>
/// Packed token ids for language-model pretraining, memory-mapped so files larger than memory work, cut into windows of
/// <see cref="Length"/> tokens: the features are tokens [s, s + length) and the targets the next tokens [s + 1, s + length
/// + 1), for window starts s = 0, stride, 2 x stride, .... The ids are float32 values (exact up to 16,777,216), as
/// <see cref="ISampleSource"/> gives samples: a one-dimensional float32 .npy file, or a raw file whose record says so
/// (<see cref="TokenFileRecord"/>, written beside it by <see cref="TokenFileWriter"/>).
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
    private readonly long _offset;
    private readonly int[] _shape;

    /// <summary>Maps <paramref name="path"/>.</summary>
    /// <param name="path">The token file: float32 ids back to back with their record, or a one-dimensional float32 .npy array.</param>
    /// <param name="length">Tokens per window (the sequence length).</param>
    /// <param name="stride">Tokens between window starts; defaults to <paramref name="length"/> (windows do not overlap).</param>
    /// <exception cref="InvalidDataException">
    /// A raw file with no record, or a file that is not a whole number of ids or shorter than one window.
    /// </exception>
    public TokenFileSource(string path, int length, int? stride = null)
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
                if (header.Shape.Length != 1)
                {
                    throw new InvalidDataException($"{path}: token ids need a one-dimensional array, not [{string.Join(", ", header.Shape)}].");
                }

                (_offset, TokenCount) = (header.DataOffset, header.Shape[0]);
            }
            else
            {
                if (!TokenFileRecord.ReadType(path))
                {
                    throw new InvalidDataException(
                        $"{path} has no record of how its ids are stored ({TokenFileRecord.PathOf(path)}, which idrak data tokens and TokenFileWriter write beside it): "
                        + "its ids are in a type not on the new design yet.");
                }

                if (_file.Length % sizeof(float) != 0)
                {
                    throw new InvalidDataException($"{path}: {_file.Length} bytes is not a whole number of float32 token ids.");
                }

                TokenCount = _file.Length / sizeof(float);
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

        long start = _offset + (long)index * Stride * sizeof(float);
        _file.Floats(start, features[..Length]);
        _file.Floats(start + sizeof(float), targets[..Length]);
    }

    /// <summary>Unmaps the file.</summary>
    public void Dispose() => _file.Dispose();
}

/// <summary>
/// NumPy .npy arrays as samples, memory-mapped so large arrays are not loaded at once: the first dimension counts the
/// samples and the rest is each sample's shape (e.g. images [N, 28, 28] give features [28, 28]). The targets come from
/// a second file with as many samples: its values ([N] gives targets [1]), or one-hot rows when <c>classes</c> is given
/// and it holds class indices. Little-endian float32 arrays are read; any other element type is refused (not on the
/// new design yet).
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

        _features.Floats(_featureHeader.DataOffset + (long)index * _featureSize * sizeof(float), features[.._featureSize]);
        if (_targets is null || _targetHeader is null)
        {
            return;
        }

        long at = _targetHeader.DataOffset + (long)index * _targetSize * sizeof(float);
        if (_classes is not { } classes)
        {
            _targets.Floats(at, targets[.._targetSize]);
            return;
        }

        Span<float> label = stackalloc float[1];
        _targets.Floats(at, label);
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
