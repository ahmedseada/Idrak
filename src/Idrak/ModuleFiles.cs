// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak;

/// <summary>
/// Weights files of modules (.ikw): <see cref="Save(Module, string)"/> writes every parameter and buffer as float32;
/// <see cref="Load(Module, string)"/> reads them back into a module of the same architecture.
/// </summary>
public static class ModuleFiles
{
    private const uint FileMagic = 0x3257_4B49; // "IKW2": parameters followed by buffers (float32)
    private const uint FileMagic3 = 0x3357_4B49; // "IKW3": layers of other types, then tensors each with a stored type
    private const uint FileMagic4 = 0x3457_4B49; // "IKW4": as IKW3, with a second list of layers of other types
    private const uint FileMagic5 = 0x3557_4B49; // "IKW5": as IKW4, with two more lists of layers of other types

    /// <summary>Writes all parameter values to a binary file (float32).</summary>
    public static void Save(this Module module, string path)
    {
        using var stream = File.Create(path);
        module.Save(stream);
    }

    /// <summary>Writes all parameter values to <paramref name="stream"/> (the same format as <see cref="Save(Module, string)"/>); the stream stays open.</summary>
    public static void Save(this Module module, Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var parameters = module.Parameters().Concat(module.Buffers()).ToList();
        writer.Write(FileMagic);
        writer.Write(parameters.Count);
        foreach (var p in parameters)
        {
            writer.Write(p.Rank);
            foreach (int d in p.Shape)
            {
                writer.Write(d);
            }

            var values = p.ToArray();
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, uint>(values.AsSpan());
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            writer.Write(MemoryMarshal.AsBytes(values.AsSpan()));                  // the values are the stored bytes
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

        bool typed = magic is FileMagic3 or FileMagic4 or FileMagic5;
        if (typed)
        {
            // An older file lists its layers of other types first: only a file with none holds float32 values alone.
            int lists = magic == FileMagic3 ? 1 : magic == FileMagic4 ? 2 : 4;
            for (int list = 0; list < lists; list++)
            {
                if (reader.ReadInt32() != 0)
                {
                    throw new NotSupportedException($"{source} holds layers of a type other than float32: type not on the new design yet.");
                }
            }
        }

        var parameters = module.Parameters().Concat(module.Buffers()).ToList();
        int count = reader.ReadInt32();
        if (count != parameters.Count)
        {
            throw new InvalidDataException($"The file has {count} parameter tensors but the model has {parameters.Count}.");
        }

        // One buffer for the stored values, grown to the largest tensor, instead of new arrays for every tensor.
        float[] buffer = [];
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

            if (typed && reader.ReadByte() is var stored && stored != 0)
            {
                throw new NotSupportedException($"{source} stores a tensor of {Tensor.FormatShape(shape)} in stored type #{stored}: type not on the new design yet.");
            }

            if (buffer.Length < p.Size)
            {
                buffer = GC.AllocateUninitializedArray<float>(p.Size);
            }

            var values = buffer.AsSpan(0, p.Size);
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values));
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, uint>(values);
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            p.Load(values);
        }
    }
}
