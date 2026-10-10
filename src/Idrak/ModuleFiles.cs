// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak;

/// <summary>
/// Weights files of modules (.ikw): <see cref="Save(Module, string)"/> writes every parameter and buffer, with the packed
/// weights of <see cref="Linear"/> and <see cref="Embedding"/> layers as they are; <see cref="Load(Module, string)"/> reads
/// them back into a module of the same architecture.
/// </summary>
public static class ModuleFiles
{
    private const uint FileMagic = 0x3257_4B49; // "IKW2": parameters followed by buffers (float32)
    private const uint FileMagic3 = 0x3357_4B49; // "IKW3": int8 layers, then tensors each with an element type
    private const uint FileMagic4 = 0x3457_4B49; // "IKW4": int8 layers, bfloat16 layers, then tensors as in IKW3
    private const uint FileMagic5 = 0x3557_4B49; // "IKW5": int8, bfloat16, int4 layers, bfloat16 embeddings, then tensors as in IKW3

    /// <summary>Writes all parameter values to a binary file (float32).</summary>
    public static void Save(this Module module, string path) => Save(module, path, WeightFormat.Float32);

    /// <summary>
    /// Writes all parameter values to a binary file in <paramref name="format"/>: Float16 and BFloat16 halve the file
    /// (values are rounded; they are float32 again after loading). Packed weights (int8, 4-bit, bfloat16 and formats of
    /// one's own) are always stored as they are.
    /// </summary>
    public static void Save(this Module module, string path, WeightFormat format)
    {
        using var stream = File.Create(path);
        module.Save(stream, format);
    }

    /// <summary>Writes all parameter values to <paramref name="stream"/> (the same format as <see cref="Save(Module, string)"/>); the stream stays open.</summary>
    public static void Save(this Module module, Stream stream) => Save(module, stream, WeightFormat.Float32);

    /// <summary>Writes all parameter values to <paramref name="stream"/> in <paramref name="format"/>; the stream stays open.</summary>
    public static void Save(this Module module, Stream stream, WeightFormat format)
    {
        // IKW3: the int8 layers (by position among the Linear layers), then each tensor with its element type.
        var linears = module.Descendants().OfType<Linear>().ToList();
        var int8 = linears.Select((l, i) => (l, i)).Where(p => p.l.Int8 is not null).ToList();
        var half = linears.Select((l, i) => (l, i)).Where(p => p.l.BFloat16 is not null).ToList();
        var int4 = linears.Select((l, i) => (l, i)).Where(p => p.l.Int4 is not null).ToList();
        var tables = module.Descendants().OfType<Embedding>().Select((e, i) => (e, i)).Where(p => p.e.BFloat16 is not null).ToList();
        var exact = int8.SelectMany(p => new[] { p.l.Int8!.Packed, p.l.Int8.Scales }).Concat(half.Select(p => p.l.BFloat16!.Packed))
            .Concat(int4.SelectMany(p => new[] { p.l.Int4!.Packed, p.l.Int4.Scales }))
            .Concat(tables.Select(p => p.e.BFloat16!.Packed))
            .Concat(linears.Where(l => l.PackedWeight is { Format: null }).SelectMany(l => l.PackedWeight!.Buffers()))   // formats of one's own
            .ToHashSet(ReferenceEqualityComparer.Instance);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var parameters = module.Parameters().Concat(module.Buffers()).ToList();
        bool five = int4.Count > 0 || tables.Count > 0;
        writer.Write(five ? FileMagic5 : half.Count > 0 ? FileMagic4 : FileMagic3);
        writer.Write(int8.Count);
        foreach (var (_, index) in int8)
        {
            writer.Write(index);
        }

        if (half.Count > 0 || five)
        {
            writer.Write(half.Count);
            foreach (var (_, index) in half)
            {
                writer.Write(index);
            }
        }

        if (five)
        {
            writer.Write(int4.Count);
            foreach (var (_, index) in int4)
            {
                writer.Write(index);
            }

            writer.Write(tables.Count);
            foreach (var (_, index) in tables)
            {
                writer.Write(index);
            }
        }

        writer.Write(parameters.Count);
        foreach (var p in parameters)
        {
            writer.Write(p.Rank);
            foreach (int d in p.Shape)
            {
                writer.Write(d);
            }

            var type = exact.Contains(p) ? WeightFormat.Float32 : format;
            writer.Write((byte)type);
            var values = p.ToArray();
            if (type == WeightFormat.Float32 && BitConverter.IsLittleEndian)
            {
                writer.Write(MemoryMarshal.AsBytes(values.AsSpan()));                  // the values are the stored bytes
            }
            else
            {
                writer.Write(Encode(values, type));
            }
        }
    }

    /// <summary>Reads parameter values written by <see cref="Save(Module, string)"/> into this module. The architecture must match.</summary>
    public static void Load(this Module module, string path)
    {
        using var stream = File.OpenRead(path);
        Load(module, stream, path);
    }

    /// <summary>Reads parameter values written by <see cref="Idrak.ModuleFiles.Save(Module, Stream)"/> from <paramref name="stream"/>; the stream stays open.</summary>
    public static void Load(this Module module, Stream stream) => Load(module, stream, "The stream");

    private static void Load(Module module, Stream stream, string source)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        uint magic = reader.ReadUInt32();
        if (magic is not (FileMagic or FileMagic3 or FileMagic4 or FileMagic5))
        {
            throw new InvalidDataException($"{source} is not a Idrak weights file.");
        }

        if (magic is FileMagic3 or FileMagic4 or FileMagic5)
        {
            // Layers stored as int8, int4 or bfloat16 get room for their packed weights first (allocated, not computed: the
            // file's values replace them).
            var linears = module.Descendants().OfType<Linear>().ToList();
            int quantized = reader.ReadInt32();
            for (int i = 0; i < quantized; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file quantizes Linear layer {index}, but the model has {linears.Count}.");
                }

                linears[index].LoadAsInt8();
            }

            int halves = magic is FileMagic4 or FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < halves; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file stores Linear layer {index} as bfloat16, but the model has {linears.Count}.");
                }

                linears[index].LoadAsBFloat16();
            }

            int nibbles = magic == FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < nibbles; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file stores Linear layer {index} as int4, but the model has {linears.Count}.");
                }

                linears[index].LoadAsInt4();
            }

            var embeddings = module.Descendants().OfType<Embedding>().ToList();
            int tables = magic == FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < tables; i++)
            {
                int index = reader.ReadInt32();
                if (index >= embeddings.Count)
                {
                    throw new InvalidDataException($"The file stores Embedding {index} as bfloat16, but the model has {embeddings.Count}.");
                }

                embeddings[index].LoadAsBFloat16();
            }
        }

        var parameters = module.Parameters().Concat(module.Buffers()).ToList();
        int count = reader.ReadInt32();
        if (count != parameters.Count)
        {
            throw new InvalidDataException($"The file has {count} parameter tensors but the model has {parameters.Count}.");
        }

        // One buffer for the stored bytes and one for decoded values, grown to the largest tensor, instead of new arrays
        // for every tensor.
        byte[] buffer = [];
        float[] decoded = [];
        foreach (var p in parameters)
        {
            var shape = new int[reader.ReadInt32()];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            if (!shape.AsSpan().SequenceEqual(p.Shape))
            {
                throw new InvalidDataException($"Shape mismatch: file has {Tensor.FormatShape(shape)}, model has {Tensor.FormatShape(p.Shape)}.");
            }

            var type = magic is FileMagic3 or FileMagic4 or FileMagic5 ? (WeightFormat)reader.ReadByte() : WeightFormat.Float32;
            var codec = WeightCodec.For(type);
            int length = p.Size * codec.BytesPerValue;
            if (buffer.Length < length)
            {
                buffer = GC.AllocateUninitializedArray<byte>(length);
            }

            var bytes = buffer.AsSpan(0, length);
            reader.BaseStream.ReadExactly(bytes);
            if (type == WeightFormat.Float32 && BitConverter.IsLittleEndian)
            {
                p.Load(MemoryMarshal.Cast<byte, float>(bytes));                       // the stored bytes are the values
            }
            else
            {
                if (decoded.Length < p.Size)
                {
                    decoded = GC.AllocateUninitializedArray<float>(p.Size);
                }

                var values = decoded.AsSpan(0, p.Size);
                codec.Decode(bytes, values);
                p.Load(values);
            }
        }
    }

    // Little-endian bytes of the values in `format` (see WeightCodec).
    private static byte[] Encode(float[] values, WeightFormat format)
    {
        var codec = WeightCodec.For(format);
        var bytes = new byte[values.Length * codec.BytesPerValue];
        codec.Encode(values, bytes);
        return bytes;
    }
}
