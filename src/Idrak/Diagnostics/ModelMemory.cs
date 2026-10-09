// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Diagnostics;

/// <summary>One part of a model's memory, in bytes.</summary>
/// <param name="Name">What it is ("embedding", "head", "layers", "encoder", "kv cache", ...).</param>
/// <param name="Bytes">Its bytes on the device.</param>
/// <param name="Note">How it is held, when that explains the figure ("bfloat16", "the head's columns: no copy of its own").</param>
public sealed record MemoryPart(string Name, long Bytes, string? Note = null);

/// <summary>
/// Where a model's device memory goes, from its tensors and shapes (family-neutral: any module, any decoder built from a
/// <see cref="DecoderSpec"/>, any key/value layout). Each storage is counted once, so weights shared between layers (a
/// tied head and its embedding) are not counted twice. Compare with what the device itself reports
/// (<see cref="ComputeResources.GetMemoryUsage"/>: in use, and the peak since <see cref="ComputeResources.ResetPeakMemoryUsage"/>),
/// which adds activations, workspaces and the runtime's own blocks.
/// </summary>
public static class ModelMemory
{
    /// <summary>The bytes of <paramref name="module"/>'s parameters and buffers (packed weights as packed), each storage once.</summary>
    public static long TensorBytes(Module module)
    {
        ArgumentNullException.ThrowIfNull(module);
        return Bytes(module, new HashSet<Storage>(ReferenceEqualityComparer.Instance));
    }

    /// <summary>
    /// A decoder's weights by part: the token embedding (named <c>embed</c> by <see cref="DecoderBuilder"/>), the output
    /// head (<c>head</c>) and everything else (the layers and norms). A tied head reading the embedding's table, or an
    /// embedding reading the head's columns, counts 0 bytes with a note saying whose table it reads.
    /// </summary>
    public static IReadOnlyList<MemoryPart> Decoder(Module decoder)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        var counted = new HashSet<Storage>(ReferenceEqualityComparer.Instance);
        var children = decoder.Children().ToList();
        var embedding = children.OfType<Embedding>().FirstOrDefault();
        var head = children.OfType<Linear>().LastOrDefault(l => l.Name == "head") ?? (children.Count > 0 ? children[^1] as Linear : null);
        var parts = new List<MemoryPart>();
        if (embedding is not null)
        {
            long bytes = Bytes(embedding, counted);
            parts.Add(new MemoryPart("embedding", bytes, embedding.Head is not null ? "reads the head's bfloat16 columns: no table of its own"
                : embedding.BFloat16 is not null ? "bfloat16" : "float32"));
        }

        if (head is not null)
        {
            long bytes = Bytes(head, counted);
            string format = head.PackedWeight?.Name ?? "float32";
            parts.Add(new MemoryPart("head", bytes, head.TiedTo is not null ? "reads the embedding's table: no weights of its own"
                : head.SharedTable is not null ? $"{format}, also the embedding's table" : format));
        }

        long rest = children.Where(c => !ReferenceEquals(c, embedding) && !ReferenceEquals(c, head)).Sum(c => Bytes(c, counted));
        parts.Add(new MemoryPart("layers", rest));
        return parts;
    }

    /// <summary>
    /// The bytes of the key/value cache a decoder allocates for <paramref name="capacity"/> positions of
    /// <paramref name="batch"/> sequences in <paramref name="layout"/>: every layer holds keys and values of
    /// <see cref="DecoderSpec.KvHeads"/> heads for the whole capacity (windowed layers too), plus per-row scales when
    /// the layout has them.
    /// </summary>
    public static long KeyValueCacheBytes(DecoderSpec spec, int capacity, KeyValueLayout layout, int batch = 1)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(layout);
        long rows = (long)batch * spec.KvHeads * capacity;
        long perLayer = 2 * rows * (layout.RowWidth(spec.HeadDim) + (layout.HasScales ? 1 : 0)) * 4;
        return spec.Layers * perLayer;
    }

    /// <summary>
    /// What generating with a decoder holds: its weights by part (<see cref="Decoder"/>), the image encoder's weights when
    /// it has one (any <see cref="IVisionEncoder"/> that is a <see cref="Module"/>; one that is not counts 0 bytes, with a
    /// note), and the key/value cache of <paramref name="capacity"/> positions in <paramref name="layout"/>
    /// (<see cref="KeyValueCacheBytes"/>). Storage shared between the decoder and the encoder is counted once.
    /// </summary>
    public static IReadOnlyList<MemoryPart> Generation(Module decoder, DecoderSpec spec, int capacity, KeyValueLayout layout,
        IVisionEncoder? encoder = null, int batch = 1)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(layout);
        var parts = Decoder(decoder).ToList();
        if (encoder is not null)
        {
            var counted = new HashSet<Storage>(ReferenceEqualityComparer.Instance);
            Bytes(decoder, counted);
            parts.Add(encoder is Module module ? new MemoryPart("encoder", Bytes(module, counted))
                : new MemoryPart("encoder", 0, $"{encoder.GetType().Name} is not a module: its tensors are not counted"));
        }

        parts.Add(new MemoryPart("kv cache", KeyValueCacheBytes(spec, capacity, layout, batch), $"{layout.Name}, {capacity} positions"));
        return parts;
    }

    /// <summary>
    /// The device's peak beyond <paramref name="parts"/>: what activations, workspaces and temporary copies added at the
    /// most (<paramref name="usage"/>'s <see cref="MemoryUsage.Peak"/> less the parts, at least 0). Reset the peak
    /// (<see cref="ComputeResources.ResetPeakMemoryUsage"/>) before the work it should describe; the parts are assumed
    /// alive at the peak.
    /// </summary>
    public static MemoryPart Activations(IEnumerable<MemoryPart> parts, MemoryUsage usage)
    {
        ArgumentNullException.ThrowIfNull(parts);
        return new MemoryPart("activations", Math.Max(0, usage.Peak - parts.Sum(p => p.Bytes)), "the device's peak beyond the parts above");
    }

    private static long Bytes(Module module, HashSet<Storage> counted) =>
        module.Parameters().Concat(module.Buffers()).Select(t => t.Storage).Where(counted.Add).Sum(s => 4L * s.Length);
}
