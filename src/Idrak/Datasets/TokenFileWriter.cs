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
/// <param name="Record">The record written beside the file (<see cref="TokenFileRecord"/>).</param>
public sealed record TokenFileResult(string Path, long Documents, long Tokens, long Bytes, string Record);

/// <summary>
/// Encodes text files into a token file for pretraining (read back by <see cref="TokenFileSource"/>): each document's
/// ids, then the separator, as little-endian float32 values (exact up to 16,777,216), in the files' order, and a record
/// beside the file (<see cref="TokenFileRecord"/>). Documents are encoded on every processor in batches and written in
/// order.
/// </summary>
public static class TokenFileWriter
{
    private const int Batch = 256;

    /// <summary>Writes the ids of <paramref name="inputs"/>' documents to <paramref name="output"/>.</summary>
    /// <exception cref="FileNotFoundException">An input is missing.</exception>
    /// <exception cref="ArgumentException">The separator is negative.</exception>
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

        foreach (string input in inputs)
        {
            if (!File.Exists(input))
            {
                throw new FileNotFoundException($"{input} not found.", input);
            }
        }

        const int width = sizeof(float);
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
                float[] values = ArrayPool<float>.Shared.Rent(n);
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
                    Store(values.AsSpan(0, n), span);
                    stream.Write(span);
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                    ArrayPool<float>.Shared.Return(values);
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
            ["format"] = TokenFileRecord.Format, ["type"] = TokenFileRecord.Float32, ["tokens"] = tokens, ["documents"] = documents, ["separator"] = options.Separator,
            ["vocabulary"] = tokenizer.VocabularySize,
        };
        File.WriteAllText(record, described.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
        return new TokenFileResult(System.IO.Path.GetFullPath(output), documents, tokens, bytes, record);
    }

    // The ids as little-endian float32 values.
    private static void Store(ReadOnlySpan<float> ids, Span<byte> destination)
    {
        for (int i = 0; i < ids.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(destination[(i * sizeof(float))..], ids[i]);
        }
    }
}
