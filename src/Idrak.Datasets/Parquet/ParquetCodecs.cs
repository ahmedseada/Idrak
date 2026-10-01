// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Datasets.Parquet;

namespace Idrak.Datasets;

/// <summary>
/// Decompression of the pages of Parquet column chunks compressed with one codec, by its id in the Parquet format
/// (CompressionCodec: 0 uncompressed, 1 Snappy, 2 Gzip, 3 LZO, 4 Brotli, 5 LZ4, 6 Zstandard, 7 LZ4 raw). Register new
/// ones with <see cref="ParquetCodecs.Register"/>.
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

/// <summary>
/// The compression codecs <see cref="ParquetFile"/> reads, by Parquet codec id. Uncompressed, Snappy, Gzip, Brotli and
/// LZ4 (raw) are registered; add others (Zstandard, for example), or replace these, with <see cref="Register"/>.
/// The codec is looked up once per column chunk, so a registration applies to chunks read after it.
/// </summary>
public static class ParquetCodecs
{
    private static readonly Dictionary<int, IParquetCodec> Registry = new()
    {
        [0] = new BuiltIn(0, "Uncompressed", (input, _) => input.ToArray()),
        [1] = new BuiltIn(1, "Snappy", Codecs.Snappy),
        [2] = new BuiltIn(2, "Gzip", Codecs.Gzip),
        [4] = new BuiltIn(4, "Brotli", Codecs.Brotli),
        [7] = new BuiltIn(7, "Lz4Raw", Codecs.Lz4Block),
    };

    private delegate byte[] Decompressor(ReadOnlySpan<byte> input, int uncompressedSize);

    /// <summary>Registers (or replaces) the codec for its <see cref="IParquetCodec.Id"/>.</summary>
    public static void Register(IParquetCodec codec)
    {
        ArgumentNullException.ThrowIfNull(codec);
        lock (Registry)
        {
            Registry[codec.Id] = codec;
        }
    }

    /// <summary>Removes the codec registered for <paramref name="id"/>; returns whether there was one.</summary>
    public static bool Unregister(int id)
    {
        lock (Registry)
        {
            return Registry.Remove(id);
        }
    }

    /// <summary>The registered codec ids, in order.</summary>
    public static IReadOnlyCollection<int> Ids
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys.Order()];
            }
        }
    }

    /// <summary>The registered codec names, in id order.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.OrderBy(p => p.Key).Select(p => p.Value.Name)];
            }
        }
    }

    /// <summary>The codec registered for the Parquet codec id <paramref name="id"/>.</summary>
    public static IParquetCodec Get(int id)
    {
        lock (Registry)
        {
            if (Registry.TryGetValue(id, out var codec))
            {
                return codec;
            }

            string known = string.Join(", ", Registry.Where(p => p.Key != 0).OrderBy(p => p.Key).Select(p => p.Value.Name));
            throw new NotSupportedException(id == 6
                ? $"This Parquet file is compressed with Zstandard, which is not supported yet; {known} are (Hugging Face's own Parquet files use Snappy). "
                  + "Add a codec with ParquetCodecs.Register."
                : $"Parquet compression codec {id} is not supported ({known}); add it with ParquetCodecs.Register.");
        }
    }

    private sealed class BuiltIn(int id, string name, Decompressor decompress) : IParquetCodec
    {
        public int Id => id;

        public string Name => name;

        public byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize) => decompress(input, uncompressedSize);
    }
}
