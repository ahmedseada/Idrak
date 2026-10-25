// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>Turns stored blocks into float32 values: <paramref name="raw"/> holds values.Length / block values whole blocks.</summary>
/// <param name="raw">The stored bytes (at least as many blocks as <paramref name="values"/> receives).</param>
/// <param name="values">Where the values go.</param>
public delegate void GgufDequantizer(ReadOnlySpan<byte> raw, Span<float> values);

/// <summary>
/// A ggml tensor type: its name, its block (values and bytes) and how blocks become float32. Registered by type id in
/// <see cref="GgufTypes"/>; <see cref="GgufType.Blockwise"/> makes one from a function of one block.
/// </summary>
public interface IGgufType
{
    /// <summary>Its name (Q4_0, Q6_K, …), as in llama.cpp.</summary>
    string Name { get; }

    /// <summary>Values per block (1 for plain number types).</summary>
    int BlockValues { get; }

    /// <summary>Stored bytes per block.</summary>
    int BlockBytes { get; }

    /// <summary>
    /// Dequantizes whole blocks: <paramref name="raw"/> holds values.Length / <see cref="BlockValues"/> blocks. Called once
    /// per tensor or per chunk of rows, with many blocks: a type splits the work across threads here, and reads the bytes
    /// with vector instructions where it can (a scalar tail for what is left), giving the values a value-by-value read gives.
    /// </summary>
    void Dequantize(ReadOnlySpan<byte> raw, Span<float> values);
}

/// <summary>Ggml tensor types made from a function of one block (<see cref="Blockwise"/>).</summary>
public static class GgufType
{
    /// <summary>
    /// A type whose blocks <paramref name="block"/> dequantizes one at a time (one block of bytes into
    /// <paramref name="blockValues"/> values); the blocks are split across threads.
    /// </summary>
    public static IGgufType Blockwise(string name, int blockValues, int blockBytes, GgufDequantizer block)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(block);
        return new BlockwiseType(name, blockValues, blockBytes, block);
    }

    private sealed class BlockwiseType(string name, int blockValues, int blockBytes, GgufDequantizer block) : IGgufType
    {
        public string Name => name;

        public int BlockValues => blockValues;

        public int BlockBytes => blockBytes;

        public unsafe void Dequantize(ReadOnlySpan<byte> raw, Span<float> values)
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
        }

        public override string ToString() => name;
    }
}

/// <summary>
/// The ggml tensor types GGUF files are read with, by type id. Idrak registers F32 (0), F16 (1), BF16 (30),
/// Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K–Q6_K (10–14), IQ4_NL (20) and IQ4_XS (23); add others with
/// <see cref="Register"/>. A tensor's type is looked up once per read, not per block.
/// </summary>
public static class GgufTypes
{
    private static readonly SlotTable<int, IGgufType> Registry = new(nameof(GgufTypes), Guard);

    static GgufTypes() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterGgufTypes);   // the built-in types, on first use

    /// <summary>
    /// Registers the ggml type with id <paramref name="id"/>; under a built-in id it overrides the library's type, which
    /// stays behind it (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(int id, IGgufType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (type.BlockValues <= 0 || type.BlockBytes <= 0)
        {
            throw new ArgumentException($"{type.Name}: a block needs at least one value and one byte.", nameof(type));
        }

        Registry.Register(id, type, System.Reflection.Assembly.GetCallingAssembly(), type.Name);
    }

    /// <summary>Removes the app's type with id <paramref name="id"/> (a built-in id gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(int id) => Registry.Unregister(id);

    /// <summary>The registered type ids.</summary>
    public static IReadOnlyCollection<int> Ids => [.. Registry.Keys.Order()];

    /// <summary>The type registered with id <paramref name="id"/>.</summary>
    public static IGgufType Get(int id) => Find(id)
        ?? throw new NotRegisteredException(nameof(GgufTypes), Text(id), Ids.Select(Text),
            $"No ggml type #{id} is registered ({string.Join(", ", Ids.Select(i => $"{Find(i)?.Name} ({i})"))}); add it with GgufTypes.Register.");

    /// <summary>The type registered with id <paramref name="id"/>, or null when there is none.</summary>
    public static IGgufType? Find(int id) => Registry.Find(id);

    // An id as a refusal names it (NotRegisteredException.Name and Registered).
    internal static string Text(int id) => id.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>The library's type with id <paramref name="id"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IGgufType? Default(int id) => Registry.Default(id);

    /// <summary>Who registered the type with id <paramref name="id"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(int id) => Registry.Origin(id);

    /// <summary>
    /// What happens when the app's dequantizer of type <paramref name="id"/> fails (<see cref="SlotPolicy.Throw"/> unless set:
    /// the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's). It falls back per call, when the app's type has the library's block sizes.
    /// </summary>
    public static void SetPolicy(int id, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(id, policy, shadowRate);

    // A type with other block sizes reads other bytes: the library's cannot stand in for it.
    private static IGgufType Guard(Slot slot, IGgufType app, IGgufType library) =>
        app.BlockValues != library.BlockValues || app.BlockBytes != library.BlockBytes ? app : new GuardedType(slot, app, library);

    private sealed class GuardedType(Slot slot, IGgufType app, IGgufType library) : IGgufType
    {
        public string Name => app.Name;

        public int BlockValues => app.BlockValues;

        public int BlockBytes => app.BlockBytes;

        public void Dequantize(ReadOnlySpan<byte> raw, Span<float> values)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                library.Dequantize(raw, values);
                if (run is not null)
                {
                    run.Answered();
                    var other = new float[values.Length];
                    try
                    {
                        app.Dequantize(raw, other);
                        run.Done(Comparisons.Difference(values, other, 1e-5f));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return;
            }

            try
            {
                app.Dequantize(raw, values);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                library.Dequantize(raw, values);
            }
        }
    }
}
