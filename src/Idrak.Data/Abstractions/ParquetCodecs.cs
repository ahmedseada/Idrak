// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Abstractions;

/// <summary>
/// Decompression of the pages of Parquet column chunks compressed with one codec, by its id in the Parquet format
/// (CompressionCodec: 0 uncompressed, 1 Snappy, 2 Gzip, 3 LZO, 4 Brotli, 5 LZ4, 6 Zstandard, 7 LZ4 raw).
/// </summary>
public interface IParquetCodec
{
    /// <summary>The codec's id in the Parquet format (the column chunk's "codec").</summary>
    int Id { get; }

    /// <summary>The codec's name, shown in messages.</summary>
    string Name { get; }

    /// <summary>The page's bytes, decompressed; <paramref name="uncompressedSize"/> is the size the page header gives.</summary>
    byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize);
}

