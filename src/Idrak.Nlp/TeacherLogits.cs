// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

/// <summary>
/// Writes a teacher logits file: for each training sequence, the teacher's k largest logits (and their token ids) at
/// each trained position, read back by <see cref="TeacherLogitsFile"/> and trained on with
/// <see cref="DistillationTeachers.FromFile"/>. Six bytes per kept token (a 32-bit id and the logit as a 16-bit float,
/// relative to the position's largest), so k = 16 costs about 100 bytes per trained token. Sequences are found again by
/// their content (tokens and trained positions), so the student must encode the data exactly as when the file was written
/// (same data, chat template, maximum length and system prompt).
/// </summary>
/// <remarks>
/// Layout (little-endian): "IDRAKTL1", version (int32), vocabulary (int32), k (int32), the vocabulary fingerprint (a
/// length-prefixed UTF-8 string); then one record per sequence: trained positions n (int32), the sequence's key (uint64),
/// n·k ids (int32) and n·k logits (float16), each position's largest first; then the index: records (int32), each key
/// (uint64) and offset (int64); last, the index's offset (int64) and "IDRAKEND".
/// </remarks>
public sealed class TeacherLogitsWriter : IDisposable
{
    internal static readonly byte[] Magic = "IDRAKTL1"u8.ToArray();
    internal static readonly byte[] EndMagic = "IDRAKEND"u8.ToArray();
    internal const int Version = 1;

    private readonly FileStream _stream;
    private readonly BinaryWriter _writer;
    private readonly List<(ulong Key, long Offset)> _index = [];
    private readonly HashSet<ulong> _keys = [];
    private bool _closed;

    /// <summary>
    /// Creates (or replaces) <paramref name="path"/> for distributions over <paramref name="vocabulary"/> tokens with
    /// <paramref name="topK"/> kept per position, recording the teacher's <paramref name="vocabularyFingerprint"/>
    /// (<see cref="DistillationTeacher.VocabularyFingerprint"/>) for the check of the student that reads it.
    /// </summary>
    public TeacherLogitsWriter(string path, int vocabulary, int topK, string? vocabularyFingerprint = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabulary);
        if (topK < 1 || topK > vocabulary)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), $"k is between 1 and the vocabulary ({vocabulary}).");
        }

        Vocabulary = vocabulary;
        TopK = topK;
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        _stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        _writer = new BinaryWriter(_stream, Encoding.UTF8, leaveOpen: true);
        _writer.Write(Magic);
        _writer.Write(Version);
        _writer.Write(vocabulary);
        _writer.Write(topK);
        _writer.Write(vocabularyFingerprint ?? "");
    }

    /// <summary>The width of the distributions.</summary>
    public int Vocabulary { get; }

    /// <summary>Tokens kept per position.</summary>
    public int TopK { get; }

    /// <summary>Sequences written so far.</summary>
    public int Count => _index.Count;

    /// <summary>
    /// Adds <paramref name="sequence"/>'s top-k logits: <paramref name="ids"/> and <paramref name="logits"/> hold
    /// <see cref="TopK"/> entries for each trained position in order (sequence.TrainedTokens · k each), in any order within
    /// a position. A sequence already written (the same content) is skipped; false then.
    /// </summary>
    public bool Add(TrainingSequence sequence, ReadOnlySpan<int> ids, ReadOnlySpan<float> logits)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        ObjectDisposedException.ThrowIf(_closed, this);
        int n = sequence.TrainedTokens, k = TopK;
        if (ids.Length != n * k || logits.Length != n * k)
        {
            throw new ArgumentException($"A sequence with {n} trained tokens needs {n * k} ids and logits, got {ids.Length} and {logits.Length}.");
        }

        ulong key = Hash.Of(sequence);
        if (!_keys.Add(key))
        {
            return false;
        }

        _writer.Flush();
        _index.Add((key, _stream.Position));
        _writer.Write(n);
        _writer.Write(key);
        var order = new int[k];
        var rowIds = new int[n * k];
        var rowValues = new Half[n * k];
        float[] values = logits.ToArray();
        for (int r = 0; r < n; r++)
        {
            for (int j = 0; j < k; j++)
            {
                order[j] = r * k + j;
            }

            Array.Sort(order, (a, b) => values[b].CompareTo(values[a]));
            float max = values[order[0]];
            for (int j = 0; j < k; j++)
            {
                int id = ids[order[j]];
                if ((uint)id >= (uint)Vocabulary)
                {
                    throw new ArgumentOutOfRangeException(nameof(ids), $"Token {id} is outside the vocabulary of {Vocabulary}.");
                }

                rowIds[r * k + j] = id;
                rowValues[r * k + j] = (Half)(values[order[j]] - max);
            }
        }

        foreach (int id in rowIds)
        {
            _writer.Write(id);
        }

        foreach (var value in rowValues)
        {
            _writer.Write(value);
        }

        return true;
    }

    /// <summary>
    /// Writes <paramref name="teacher"/>'s top <paramref name="topK"/> logits for every one of <paramref name="sequences"/>
    /// to <paramref name="path"/> (the teacher in inference mode, padded batches of at most <paramref name="batchTokens"/>
    /// positions on its own device); <paramref name="progress"/> receives the sequences done. Returns the sequences written
    /// (repeats are stored once).
    /// </summary>
    public static int Write(string path, PretrainedModel teacher, IReadOnlyList<TrainingSequence> sequences, int topK = 16, Action<int>? progress = null,
        int batchTokens = 4096)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(sequences);
        var source = new ModelTeacher(teacher, 0, batchTokens);
        int k = Math.Min(topK, source.Vocabulary);
        using var writer = new TeacherLogitsWriter(path, source.Vocabulary, k, source.Fingerprint);
        for (int first = 0; first < sequences.Count;)
        {
            // Groups of sequences up to batchTokens positions: one teacher pass each.
            int last = first;
            long positions = sequences[first].Tokens.Length;
            while (last + 1 < sequences.Count && positions + sequences[last + 1].Tokens.Length <= batchTokens)
            {
                positions += sequences[++last].Tokens.Length;
            }

            var group = sequences.Skip(first).Take(last - first + 1).ToList();
            var (ids, logits) = source.TopLogits(group, k);
            int at = 0;
            foreach (var sequence in group)
            {
                int n = sequence.TrainedTokens * k;
                writer.Add(sequence, ids.AsSpan(at, n), logits.AsSpan(at, n));
                at += n;
            }

            first = last + 1;
            progress?.Invoke(first);
        }

        return writer.Count;
    }

    /// <summary>Writes the index and closes the file.</summary>
    public void Dispose()
    {
        if (_closed)
        {
            return;
        }

        _closed = true;
        _writer.Flush();
        long indexOffset = _stream.Position;
        _writer.Write(_index.Count);
        foreach (var (key, offset) in _index)
        {
            _writer.Write(key);
            _writer.Write(offset);
        }

        _writer.Write(indexOffset);
        _writer.Write(EndMagic);
        _writer.Dispose();
        _stream.Dispose();
    }
}

/// <summary>
/// Reads a teacher logits file written by <see cref="TeacherLogitsWriter"/>: the stored top-k logits of a training
/// sequence, found by its content. Safe to read from several threads.
/// </summary>
public sealed class TeacherLogitsFile : IDisposable
{
    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private readonly Dictionary<ulong, long> _index;
    private readonly Lock _lock = new();

    private TeacherLogitsFile(string path, FileStream stream)
    {
        Path = path;
        _stream = stream;
        _reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (!_reader.ReadBytes(8).AsSpan().SequenceEqual(TeacherLogitsWriter.Magic))
        {
            throw new InvalidDataException($"{path} is not a teacher logits file.");
        }

        int version = _reader.ReadInt32();
        if (version != TeacherLogitsWriter.Version)
        {
            throw new InvalidDataException($"{path} is a teacher logits file of version {version}; this library reads version {TeacherLogitsWriter.Version}.");
        }

        Vocabulary = _reader.ReadInt32();
        TopK = _reader.ReadInt32();
        string fingerprint = _reader.ReadString();
        VocabularyFingerprint = fingerprint.Length > 0 ? fingerprint : null;
        if (Vocabulary < 1 || TopK < 1 || TopK > Vocabulary || stream.Length < 24)
        {
            throw new InvalidDataException($"{path}: a vocabulary of {Vocabulary} with k = {TopK} is not a valid teacher logits file.");
        }

        stream.Position = stream.Length - 16;
        long indexOffset = _reader.ReadInt64();
        if (!_reader.ReadBytes(8).AsSpan().SequenceEqual(TeacherLogitsWriter.EndMagic) || indexOffset < 0 || indexOffset > stream.Length - 20)
        {
            throw new InvalidDataException($"{path} is incomplete (its writer was not closed): write it again.");
        }

        stream.Position = indexOffset;
        int count = _reader.ReadInt32();
        _index = new Dictionary<ulong, long>(count);
        for (int i = 0; i < count; i++)
        {
            ulong key = _reader.ReadUInt64();
            _index[key] = _reader.ReadInt64();
        }
    }

    /// <summary>Opens <paramref name="path"/>; <see cref="InvalidDataException"/> when it is not a complete teacher logits file.</summary>
    public static TeacherLogitsFile Open(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            return new TeacherLogitsFile(path, stream);
        }
        catch (Exception ex) when (ex is EndOfStreamException or IOException)
        {
            stream.Dispose();
            throw new InvalidDataException($"{path} is not a complete teacher logits file.", ex);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>Whether <paramref name="path"/> is a file that starts as a teacher logits file does.</summary>
    public static bool IsTeacherLogits(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        using var stream = File.OpenRead(path);
        Span<byte> head = stackalloc byte[8];
        return stream.Read(head) == 8 && head.SequenceEqual(TeacherLogitsWriter.Magic);
    }

    /// <summary>The file's path.</summary>
    public string Path { get; }

    /// <summary>The width of the distributions.</summary>
    public int Vocabulary { get; }

    /// <summary>Tokens kept per position.</summary>
    public int TopK { get; }

    /// <summary>The teacher's vocabulary fingerprint, or null when none was recorded.</summary>
    public string? VocabularyFingerprint { get; }

    /// <summary>Sequences stored.</summary>
    public int Count => _index.Count;

    /// <summary>Whether <paramref name="sequence"/> (by content) is stored.</summary>
    public bool Contains(TrainingSequence sequence) => _index.ContainsKey(Hash.Of(sequence));

    /// <summary>
    /// The stored top-k of <paramref name="sequence"/>, or null when it is not in the file: k ids and logits per trained
    /// position, in order, each position's largest first, the logits relative to it (the largest is 0).
    /// </summary>
    public (int[] Ids, float[] Logits)? Read(TrainingSequence sequence)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        if (!_index.TryGetValue(Hash.Of(sequence), out long offset))
        {
            return null;
        }

        lock (_lock)
        {
            _stream.Position = offset;
            int n = _reader.ReadInt32();
            _reader.ReadUInt64();
            if (n != sequence.TrainedTokens)
            {
                throw new InvalidDataException($"{Path}: a stored sequence has {n} trained tokens where the data has {sequence.TrainedTokens}.");
            }

            int count = n * TopK;
            var ids = new int[count];
            var logits = new float[count];
            for (int i = 0; i < count; i++)
            {
                ids[i] = _reader.ReadInt32();
            }

            for (int i = 0; i < count; i++)
            {
                logits[i] = (float)_reader.ReadHalf();
            }

            return (ids, logits);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _reader.Dispose();
        _stream.Dispose();
    }
}
