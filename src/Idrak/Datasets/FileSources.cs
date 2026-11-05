// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Datasets;

/// <summary>
/// How token files store their ids, and the record beside a token file (<c>tokens.bin.json</c>, format
/// <c>idrak-tokens/1</c>) that says it: <see cref="TokenFileWriter.Write"/> writes it, <see cref="TokenFileSource"/> reads
/// it, and a file without one is read with the type it is given. Ids are stored in any registered integer type of whole
/// bytes (<see cref="Abstraction.ElementTypes.IdTypes"/>: uint8, uint16, int32, uint32, ... and any an app registers),
/// little-endian.
/// </summary>
public static class TokenFileRecord
{
    /// <summary>The record's format name.</summary>
    public const string Format = "idrak-tokens/1";

    /// <summary>Where the record of <paramref name="tokenFile"/> is: beside it, its name with ".json" added.</summary>
    public static string PathOf(string tokenFile)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenFile);
        return tokenFile + ".json";
    }

    /// <summary>Whether token files may store their ids in <paramref name="type"/>: an integer type of whole bytes, one value a block.</summary>
    public static bool IsIdType(Abstraction.ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        return type.IsInteger && type.BlockSize == 1 && type.BlockBits % 8 == 0;
    }

    /// <summary>
    /// The id type the record beside <paramref name="tokenFile"/> says, or null when there is none (no file, or a .json of
    /// another format).
    /// </summary>
    /// <exception cref="InvalidDataException">The record is not JSON, or names no id type or one that is not registered.</exception>
    public static Abstraction.ElementType? ReadType(string tokenFile)
    {
        string record = PathOf(tokenFile);
        if (!File.Exists(record))
        {
            return null;
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
            return null;
        }

        if (json["type"] is not JsonValue type || type.GetValueKind() != JsonValueKind.String)
        {
            throw new InvalidDataException($"{record}: the record names no \"type\" (one of {IdTypeNames}).");
        }

        try
        {
            return Parse(type.GetValue<string>());
        }
        catch (ArgumentException e)
        {
            throw new InvalidDataException($"{record}: {e.Message}", e);
        }
    }

    /// <summary>
    /// The id type <paramref name="text"/> names: a registered integer type of whole bytes by its name or short name
    /// (uint16, UInt16, u16; int32, i32), as the settings and the records name them.
    /// </summary>
    /// <exception cref="ArgumentException">No such type is registered, or it is not one ids are stored in (the message names those that are).</exception>
    public static Abstraction.ElementType Parse(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        var name = text.AsSpan().Trim();
        var type = Abstraction.ElementTypes.Find(name);
        if (type is null && name.Length > 1 && (name[0] is 'u' or 'U' or 'i' or 'I') && int.TryParse(name[1..], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int bits))
        {
            type = Abstraction.ElementTypes.Find((name[0] is 'u' or 'U' ? "uint" : "int") + bits.ToString(System.Globalization.CultureInfo.InvariantCulture));   // u16: uint16
        }

        return type is not null && IsIdType(type) ? type
            : throw new ArgumentException($"Token ids are stored in an integer type of whole bytes ({IdTypeNames}), not '{text}'.", nameof(text));
    }

    // The registered id types, smallest first, for messages.
    private static string IdTypeNames => string.Join(", ", Abstraction.ElementTypes.IdTypes.Select(t => t.Name));
}

/// <summary>
/// Packed token ids for language-model pretraining, memory-mapped so files larger than memory work, cut into windows of
/// <see cref="Length"/> tokens: the features are tokens [s, s + length) and the targets the next tokens [s + 1, s + length
/// + 1), for window starts s = 0, stride, 2 x stride, .... The ids are read as <see cref="ISampleSource"/> gives samples,
/// as floats (exact up to 16,777,216). A .npy file of one dimension is read with its own element type; a raw file with
/// the type its record says (<see cref="TokenFileRecord"/>, written beside it by <see cref="TokenFileWriter"/>), else
/// the type it is given: any registered integer type of whole bytes.
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
    private readonly Abstraction.ElementType _type;
    private readonly int _size;
    private readonly long _offset;
    private readonly int[] _shape;

    /// <summary>Maps <paramref name="path"/>.</summary>
    /// <param name="path">The token file: ids back to back, or a one-dimensional .npy array.</param>
    /// <param name="length">Tokens per window (the sequence length).</param>
    /// <param name="stride">Tokens between window starts; defaults to <paramref name="length"/> (windows do not overlap).</param>
    /// <param name="type">
    /// The ids' type in a raw file (<see cref="TokenFileRecord.IsIdType"/>): null to read it from the file's record
    /// (<see cref="TokenFileRecord"/>); given while the record says another, the file is refused naming both. A .npy file
    /// says its own.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="type"/> is not a type ids are stored in.</exception>
    /// <exception cref="InvalidDataException">
    /// A raw file with no record and no <paramref name="type"/>, a record and a <paramref name="type"/> that disagree, or a
    /// file that is not a whole number of ids or shorter than one window.
    /// </exception>
    public TokenFileSource(string path, int length, int? stride = null, Abstraction.ElementType? type = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Stride = stride ?? length;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Stride, nameof(stride));
        if (type is not null && !TokenFileRecord.IsIdType(type))
        {
            throw new ArgumentException($"Token ids are stored in an integer type of whole bytes, not {type.Name}.", nameof(type));
        }

        _file = new MappedFile(path);
        try
        {
            if (NpyHeader.IsNpy(_file.Bytes(0, (int)Math.Min(_file.Length, 6))))
            {
                var header = NpyHeader.Read(_file);
                var stored = Abstraction.ElementTypes.Find(header.Type.ToString());   // the registry's type of the array's (int32, uint16, ...)
                if (header.Shape.Length != 1 || stored is null || !TokenFileRecord.IsIdType(stored))
                {
                    throw new InvalidDataException($"{path}: token ids need a one-dimensional integer array, not {header.Type} [{string.Join(", ", header.Shape)}].");
                }

                (_type, _offset, TokenCount) = (stored, header.DataOffset, header.Shape[0]);
                _size = (int)_type.ByteCount(1);
            }
            else
            {
                var recorded = TokenFileRecord.ReadType(path);
                if (recorded is { } said && type is { } given && !string.Equals(said.Name, given.Name, StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidDataException($"{path}: its record ({TokenFileRecord.PathOf(path)}) says the ids are {said.Name}, the setting {given.Name}; give {said.Name} or no type.");
                }

                _type = recorded ?? type ?? throw new InvalidDataException(
                    $"{path} has no record of its id type ({TokenFileRecord.PathOf(path)}, which idrak data tokens and TokenFileWriter write beside it): "
                    + "give the type (an integer type such as uint16 or int32) in TokenFileSource's type, the tokens sample source's \"type\" option, "
                    + "idrak pretrain --token-type or a pretraining source's \"token_type\".");
                _size = (int)_type.ByteCount(1);
                if (_file.Length % _size != 0)
                {
                    throw new InvalidDataException($"{path}: {_file.Length} bytes is not a whole number of {_type.Name} token ids.");
                }

                TokenCount = _file.Length / _size;
            }

            if (_type.ToFloat32 is null)
            {
                throw new NotSupportedException($"{path}: the id type {_type.Name} has no conversion to the samples' float32 (ElementType.ToFloat32).");
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

        int size = _size;
        var bytes = _file.Bytes(_offset + (long)index * Stride * size, (Length + 1) * size);
        _type.Decode(bytes[..(Length * size)], features[..Length]);                  // the type's own conversion (vector loops for the built-ins)
        _type.Decode(bytes[size..], targets[..Length]);
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
