// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Text;

namespace Idrak.Datasets;

/// <summary>How <see cref="TokenFileWriter.Write"/> cuts text files into documents (each encoded alone, the separator after it).</summary>
public enum TokenDocuments
{
    /// <summary>Documents between lines of <c>---</c>, as <c>idrak data text</c> writes them (a file without such lines is one document).</summary>
    Blocks,

    /// <summary>Each line is a document.</summary>
    Lines,

    /// <summary>Each file is one document.</summary>
    Files,
}

/// <summary>Settings for <see cref="TokenFileWriter.Write"/>.</summary>
public sealed record TokenFileWriterOptions
{
    /// <summary>
    /// The type ids are stored in: any registered integer type of whole bytes (<see cref="TokenFileRecord.IsIdType"/>), or
    /// null for <see cref="Abstraction.ElementTypes.IdEnvironmentVariable"/> when set, else the smallest that holds the
    /// tokenizer's ids and the separator (<see cref="Abstraction.ElementTypes.ChooseIds"/>, rule 74). The type is recorded
    /// beside the file (<see cref="TokenFileRecord"/>), where <see cref="TokenFileSource"/> reads it.
    /// </summary>
    public Abstraction.ElementType? Type { get; init; }

    /// <summary>An id written after every document (a document's end, "&lt;/s&gt;" for example), or null for none.</summary>
    public int? Separator { get; init; }

    /// <summary>How files are cut into documents.</summary>
    public TokenDocuments Documents { get; init; } = TokenDocuments.Blocks;

    /// <summary>Threads that encode documents (0: one per processor).</summary>
    public int Workers { get; init; }
}

/// <summary>What <see cref="TokenFileWriter.Write"/> wrote.</summary>
/// <param name="Path">The token file.</param>
/// <param name="Documents">Documents encoded.</param>
/// <param name="Tokens">Ids written (separators included).</param>
/// <param name="Bytes">The text's bytes read.</param>
/// <param name="Type">The type the ids are stored in (the one asked, else the smallest that holds them).</param>
/// <param name="Record">The record written beside the file (<see cref="TokenFileRecord"/>).</param>
public sealed record TokenFileResult(string Path, long Documents, long Tokens, long Bytes, Abstraction.ElementType Type, string Record);

/// <summary>
/// Encodes text files into a token file for pretraining (read back by <see cref="TokenFileSource"/>): each document's
/// ids, then the separator, little-endian in the integer type asked or else the smallest that holds the tokenizer's ids, in
/// the files' order, and the type in a record beside the file (<see cref="TokenFileRecord"/>). Documents are encoded on
/// every processor in batches and written in order; ids that do not fit the type asked are refused before anything is
/// written.
/// </summary>
public static class TokenFileWriter
{
    private const int Batch = 256;

    /// <summary>Writes the ids of <paramref name="inputs"/>' documents to <paramref name="output"/>.</summary>
    /// <exception cref="FileNotFoundException">An input is missing.</exception>
    /// <exception cref="ArgumentException">
    /// <see cref="TokenFileWriterOptions.Type"/> is not an integer type of whole bytes, or the tokenizer's ids (or the
    /// separator) do not fit it or the type <see cref="Abstraction.ElementTypes.IdEnvironmentVariable"/> names.
    /// </exception>
    public static TokenFileResult Write(string output, IReadOnlyList<string> inputs, ITokenizer tokenizer, TokenFileWriterOptions? options = null,
        IProgress<long>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(output);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options ??= new TokenFileWriterOptions();
        if (options.Separator is < 0)
        {
            throw new ArgumentException($"The separator id {options.Separator} is negative.", nameof(options));
        }

        long needed = Math.Max(tokenizer.VocabularySize - 1L, options.Separator ?? 0);
        Abstraction.ElementType type;
        if (options.Type is { } asked)
        {
            if (!TokenFileRecord.IsIdType(asked))
            {
                throw new ArgumentException($"Token ids are stored in an integer type of whole bytes, not {asked.Name}.", nameof(options));
            }

            if (!asked.Holds(0, needed))
            {
                string smallest = Abstraction.ElementTypes.IdTypes.FirstOrDefault(t => t.Holds(0, needed))?.Name ?? "none registered";
                throw new ArgumentException($"The tokenizer's ids (up to {needed}) do not fit {asked.Name}; ask {smallest} or no type (the smallest that holds them).", nameof(options));
            }

            type = asked;
        }
        else
        {
            type = Abstraction.ElementTypes.ChooseIds(needed, 0, source: "TokenFileWriterOptions.Type").Type;   // IDRAK_ID_DTYPE, else the smallest
        }

        foreach (string input in inputs)
        {
            if (!File.Exists(input))
            {
                throw new FileNotFoundException($"{input} not found.", input);
            }
        }

        int width = (int)type.ByteCount(1);
        long low = (long)Int128.Max(type.MinValue, long.MinValue), high = (long)Int128.Min(type.MaxValue, long.MaxValue);
        long documents = 0, tokens = 0, bytes = 0;
        var parallel = new ParallelOptions { MaxDegreeOfParallelism = options.Workers > 0 ? options.Workers : ComputeResources.MaxCpuThreads, CancellationToken = cancellationToken };
        string? folder = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(output));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(output, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
        var pending = new List<string>(Batch);
        var encoded = new IReadOnlyList<int>[Batch];
        void Flush()
        {
            int count = pending.Count;
            Parallel.For(0, count, parallel, i => encoded[i] = tokenizer.Encode(pending[i]));
            for (int i = 0; i < count; i++)
            {
                var ids = encoded[i];
                int n = ids.Count + (options.Separator is null ? 0 : 1);
                byte[] buffer = ArrayPool<byte>.Shared.Rent(n * width);
                int[] values = ArrayPool<int>.Shared.Rent(n);
                try
                {
                    for (int k = 0; k < ids.Count; k++)
                    {
                        values[k] = ids[k];
                    }

                    if (options.Separator is int separator)
                    {
                        values[ids.Count] = separator;
                    }

                    var span = buffer.AsSpan(0, n * width);
                    Store(values.AsSpan(0, n), type, low, high, span);
                    stream.Write(span);
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                    ArrayPool<int>.Shared.Return(values);
                }

                tokens += n;
                encoded[i] = [];
            }

            documents += count;
            pending.Clear();
            progress?.Report(bytes);
        }

        void Add(string document)
        {
            pending.Add(document);
            if (pending.Count == Batch)
            {
                Flush();
            }
        }

        foreach (string input in inputs)
        {
            bytes += new FileInfo(input).Length;
            if (options.Documents == TokenDocuments.Files)
            {
                Add(File.ReadAllText(input, Encoding.UTF8));
                continue;
            }

            using var reader = new StreamReader(input, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, 1 << 20);
            var document = new StringBuilder();
            bool any = false;
            while (reader.ReadLine() is { } line)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (options.Documents == TokenDocuments.Lines)
                {
                    Add(line);
                    continue;
                }

                if (line == "---")
                {
                    Add(document.ToString());
                    (document.Length, any) = (0, false);
                    continue;
                }

                if (any)
                {
                    document.Append('\n');
                }

                document.Append(line);
                any = true;
            }

            if (options.Documents == TokenDocuments.Blocks && (any || document.Length > 0))
            {
                Add(document.ToString());
            }
        }

        if (pending.Count > 0)
        {
            Flush();
        }

        stream.Flush();
        string record = TokenFileRecord.PathOf(System.IO.Path.GetFullPath(output));
        var described = new System.Text.Json.Nodes.JsonObject
        {
            ["format"] = TokenFileRecord.Format, ["type"] = type.Name, ["tokens"] = tokens, ["documents"] = documents, ["separator"] = options.Separator,
            ["vocabulary"] = tokenizer.VocabularySize,
        };
        File.WriteAllText(record, described.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
        return new TokenFileResult(System.IO.Path.GetFullPath(output), documents, tokens, bytes, type, record);
    }

    // The ids as `type` stores them: little-endian integers of its width (two's complement when signed), each checked
    // against its range [low, high] (a tokenizer id past its vocabulary is refused, naming it).
    private static void Store(ReadOnlySpan<int> ids, Abstraction.ElementType type, long low, long high, Span<byte> destination)
    {
        int width = destination.Length / Math.Max(1, ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            long v = ids[i];
            if (v < low || v > high)
            {
                throw new ArgumentException($"The tokenizer gave the id {v}, which {type.Name} does not hold ([{low}, {high}]).");
            }

            var at = destination.Slice(i * width, width);
            switch (width)
            {
                case 1:
                    at[0] = (byte)v;
                    break;
                case 2:
                    BinaryPrimitives.WriteUInt16LittleEndian(at, (ushort)v);
                    break;
                case 4:
                    BinaryPrimitives.WriteUInt32LittleEndian(at, (uint)v);
                    break;
                case 8:
                    BinaryPrimitives.WriteInt64LittleEndian(at, v);
                    break;
                default:
                    for (int b = 0; b < width; b++)
                    {
                        at[b] = b < 8 ? (byte)(v >> (8 * b)) : (byte)(v < 0 ? 0xFF : 0);
                    }

                    break;
            }
        }
    }
}
