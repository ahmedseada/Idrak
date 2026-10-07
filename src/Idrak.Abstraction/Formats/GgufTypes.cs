// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>Turns stored blocks into float32 values: <paramref name="raw"/> holds values.Length / block values whole blocks.</summary>
/// <param name="raw">The stored bytes (at least as many blocks as <paramref name="values"/> receives).</param>
/// <param name="values">Where the values go.</param>
public delegate void GgufDequantizer(ReadOnlySpan<byte> raw, Span<float> values);

/// <summary>
/// A ggml tensor type: its name, its block (values and bytes) and how blocks become float32. Register new ones with
/// <see cref="GgufTypes.Register"/>.
/// </summary>
public sealed class GgufType
{
    /// <summary>Its name (Q4_0, Q6_K, …), as in llama.cpp.</summary>
    public required string Name { get; init; }

    /// <summary>Values per block (1 for plain number types).</summary>
    public required int BlockValues { get; init; }

    /// <summary>Stored bytes per block.</summary>
    public required int BlockBytes { get; init; }

    /// <summary>
    /// Dequantizes whole blocks (called once per tensor or per chunk of rows, with many blocks; split the work across
    /// threads here, as <see cref="Blockwise"/> does).
    /// </summary>
    public required GgufDequantizer Dequantize { get; init; }

    /// <summary>
    /// A type whose blocks <paramref name="block"/> dequantizes one at a time (one block of bytes into
    /// <paramref name="blockValues"/> values); the blocks are split across threads.
    /// </summary>
    public static unsafe GgufType Blockwise(string name, int blockValues, int blockBytes, GgufDequantizer block) => new()
    {
        Name = name,
        BlockValues = blockValues,
        BlockBytes = blockBytes,
        Dequantize = (raw, values) =>
        {
            int blocks = values.Length / blockValues;
            // The workers cannot capture spans: they read and write the pinned memory through pointers.
            fixed (byte* input = raw)
            fixed (float* output = values)
            {
                nint source = (nint)input, target = (nint)output;
                Parallel.For(0, (blocks + 1023) / 1024, chunk =>
                {
                    int first = chunk * 1024, last = Math.Min(blocks, first + 1024);
                    for (int b = first; b < last; b++)
                    {
                        block(new ReadOnlySpan<byte>((byte*)source + (long)b * blockBytes, blockBytes),
                            new Span<float>((float*)target + (long)b * blockValues, blockValues));
                    }
                });
            }
        },
    };
}

/// <summary>
/// The ggml tensor types GGUF files are read with, by type id. Idrak.LanguageModels registers F32 (0), F16 (1), BF16 (30),
/// Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K–Q6_K (10–14), IQ4_NL (20) and IQ4_XS (23); add others with
/// <see cref="Register"/>. A tensor's type is looked up once per read, not per block.
/// </summary>
public static class GgufTypes
{
    private static readonly Dictionary<int, GgufType> Registry = [];

    static GgufTypes() => LibraryDefaults.Ensure();

    /// <summary>Registers (or replaces) the ggml type with id <paramref name="id"/>.</summary>
    public static void Register(int id, GgufType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (type.BlockValues <= 0 || type.BlockBytes <= 0)
        {
            throw new ArgumentException($"{type.Name}: a block needs at least one value and one byte.", nameof(type));
        }

        lock (Registry)
        {
            Registry[id] = type;
        }
    }

    /// <summary>Removes the type registered with id <paramref name="id"/>; false when there is none.</summary>
    public static bool Unregister(int id)
    {
        lock (Registry)
        {
            return Registry.Remove(id);
        }
    }

    /// <summary>The registered type ids.</summary>
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

    /// <summary>The type registered with id <paramref name="id"/>.</summary>
    public static GgufType Get(int id) => Find(id)
        ?? throw new NotSupportedException($"No ggml type #{id} is registered ({string.Join(", ", Ids.Select(i => $"{Find(i)?.Name} ({i})"))}); add it with GgufTypes.Register.");

    /// <summary>The type registered with id <paramref name="id"/>, or null when there is none.</summary>
    public static GgufType? Find(int id)
    {
        lock (Registry)
        {
            return Registry.GetValueOrDefault(id);
        }
    }
}
