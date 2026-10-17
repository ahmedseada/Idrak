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
    /// <summary>How ids are stored (the reader, <see cref="TokenFileSource"/>, is told the same).</summary>
    public TokenType Type { get; init; } = TokenType.UInt16;

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
public sealed record TokenFileResult(string Path, long Documents, long Tokens, long Bytes);

/// <summary>
/// Encodes text files into a token file for pretraining (read back by <see cref="TokenFileSource"/>): each document's
/// ids, then the separator, little-endian in the chosen width, in the files' order. Documents are encoded on every
/// processor in batches and written in order; a tokenizer whose ids do not fit the width is refused before anything
/// is written.
/// </summary>
public static class TokenFileWriter
{
    private const int Batch = 256;

    /// <summary>Writes the ids of <paramref name="inputs"/>' documents to <paramref name="output"/>.</summary>
    /// <exception cref="FileNotFoundException">An input is missing.</exception>
    /// <exception cref="ArgumentException">The tokenizer's ids (or the separator) do not fit <see cref="TokenFileWriterOptions.Type"/>.</exception>
    public static TokenFileResult Write(string output, IReadOnlyList<string> inputs, ITokenizer tokenizer, TokenFileWriterOptions? options = null,
        IProgress<long>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(output);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options ??= new TokenFileWriterOptions();
        long largest = options.Type switch { TokenType.UInt16 => ushort.MaxValue, TokenType.Int32 => int.MaxValue, _ => uint.MaxValue };
        if (tokenizer.VocabularySize - 1 > largest || options.Separator is int s && (s < 0 || s > largest))
        {
            throw new ArgumentException($"The tokenizer's {tokenizer.VocabularySize} ids do not fit {options.Type}; write {TokenType.UInt32} or {TokenType.Int32} ids.", nameof(options));
        }

        foreach (string input in inputs)
        {
            if (!File.Exists(input))
            {
                throw new FileNotFoundException($"{input} not found.", input);
            }
        }

        int width = options.Type == TokenType.UInt16 ? 2 : 4;
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
                try
                {
                    var span = buffer.AsSpan(0, n * width);
                    for (int k = 0; k < ids.Count; k++)
                    {
                        Put(span, k, ids[k], options.Type);
                    }

                    if (options.Separator is int separator)
                    {
                        Put(span, ids.Count, separator, options.Type);
                    }

                    stream.Write(span);
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(buffer);
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

        return new TokenFileResult(System.IO.Path.GetFullPath(output), documents, tokens, bytes);
    }

    private static void Put(Span<byte> span, int index, int id, TokenType type)
    {
        if (type == TokenType.UInt16)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(span[(index * 2)..], checked((ushort)id));
        }
        else if (type == TokenType.Int32)
        {
            BinaryPrimitives.WriteInt32LittleEndian(span[(index * 4)..], id);
        }
        else
        {
            BinaryPrimitives.WriteUInt32LittleEndian(span[(index * 4)..], checked((uint)id));
        }
    }
}
