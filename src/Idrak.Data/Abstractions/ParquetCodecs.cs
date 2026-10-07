// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Data.Abstractions;

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
/// The compression codecs the Parquet reader (<c>ParquetFile</c> in Idrak.Data) reads, by Parquet codec id. Idrak.Data
/// registers uncompressed, Snappy, Gzip, Brotli and LZ4 (raw); add others (Zstandard, for example), or replace these, with
/// <see cref="Register"/>. The codec is looked up once per column chunk, so a registration applies to chunks read after it.
/// </summary>
public static class ParquetCodecs
{
    private static readonly SlotTable<int, IParquetCodec> Registry = new(nameof(ParquetCodecs), (slot, app, library) => new GuardedCodec(slot, app, library));

    static ParquetCodecs() => Overrides.AsLibraryDefaults(Parquet.Codecs.RegisterAll);   // the built-in codecs, on first use

    /// <summary>
    /// Registers the codec for its <see cref="IParquetCodec.Id"/>; for a built-in id it overrides the library's, which
    /// stays behind it (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IParquetCodec codec)
    {
        ArgumentNullException.ThrowIfNull(codec);
        Registry.Register(codec.Id, codec, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's codec for <paramref name="id"/> (a built-in id gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(int id) => Registry.Unregister(id);

    /// <summary>The registered codec ids, in order.</summary>
    public static IReadOnlyCollection<int> Ids => [.. Registry.Keys.Order()];

    /// <summary>The registered codec names, in id order.</summary>
    public static IReadOnlyCollection<string> Names => [.. Registry.Keys.Order().Select(id => Registry.Find(id)!.Name)];

    /// <summary>The library's codec for <paramref name="id"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IParquetCodec? Default(int id) => Registry.Default(id);

    /// <summary>Who registered the codec for <paramref name="id"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(int id) => Registry.Origin(id);

    /// <summary>What happens when the app's codec for <paramref name="id"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's).</summary>
    public static void SetPolicy(int id, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(id, policy, shadowRate);

    /// <summary>The codec registered for the Parquet codec id <paramref name="id"/>.</summary>
    public static IParquetCodec Get(int id)
    {
        if (Registry.TryGet(id, out var codec))
        {
            return codec;
        }

        string known = string.Join(", ", Registry.Keys.Where(i => i != 0).Order().Select(i => Registry.Find(i)!.Name));
        throw new NotSupportedException(id == 6
            ? $"This Parquet file is compressed with Zstandard, which is not supported yet; {known} are (Hugging Face's own Parquet files use Snappy). "
              + "Add a codec with ParquetCodecs.Register."
            : $"Parquet compression codec {id} is not supported ({known}); add it with ParquetCodecs.Register.");
    }

    // An app's codec over a built-in: each page falls back on its own, or is compared with the built-in's output.
    private sealed class GuardedCodec(Slot slot, IParquetCodec app, IParquetCodec library) : IParquetCodec
    {
        public int Id => app.Id;

        public string Name => app.Name;

        public byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Decompress(input, uncompressedSize);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = app.Decompress(input, uncompressedSize);
                        run.Done(answer.AsSpan().SequenceEqual(other) ? null : $"{other.Length} bytes differ from the library's {answer.Length}");
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Decompress(input, uncompressedSize);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Decompress(input, uncompressedSize);
            }
        }
    }
}
