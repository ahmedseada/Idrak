// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Models;

/// <summary>
/// The weights' format <see cref="PretrainedModel.Load"/> chose (<see cref="PretrainedOptions.AutoWeights"/>), measured
/// before any weight was read: the bytes the weights take in it, the working memory beside them (the key/value caches
/// for <see cref="Context"/> tokens, a step's scores, the headroom), and what the device reported free.
/// </summary>
/// <param name="Format">The format chosen: "float32", "bfloat16", "int8" or "int4" (<see cref="AutoWeights.Formats"/>).</param>
/// <param name="WeightBytes">The bytes the weights take in it.</param>
/// <param name="WorkingBytes">The working memory kept beside them.</param>
/// <param name="AvailableBytes">The bytes the device reported free (or the budget given), or null when it reports none.</param>
/// <param name="Context">The tokens the key/value caches were counted for.</param>
public sealed record WeightsChoice(string Format, long WeightBytes, long WorkingBytes, long? AvailableBytes, int Context)
{
    /// <summary>The weights and the working memory.</summary>
    public long NeededBytes => WeightBytes + WorkingBytes;

    /// <summary>Whether it fits the memory reported free (true when the device reports none).</summary>
    public bool Fits => AvailableBytes is not { } free || NeededBytes <= free;

    /// <summary>Whether the load goes on with part of the model in system memory (nothing fits, and offload is on).</summary>
    public bool Offload { get; init; }

    /// <summary>"int8 (auto): 8.1 GB of weights and 0.6 GB of working memory for 4096 tokens, of 14.2 GB free".</summary>
    public override string ToString() =>
        $"{Format} (auto): {AutoWeights.Gigabytes(WeightBytes)} of weights and {AutoWeights.Gigabytes(WorkingBytes)} of working memory for {Context} tokens"
        + (AvailableBytes is { } free ? $", of {AutoWeights.Gigabytes(free)} free" : ", the device reports no free memory")
        + (Offload ? "; it does not fit, so part of it goes to system memory (offload)" : "");
}

/// <summary>
/// The weights "auto" choice of <see cref="PretrainedModel.Load"/> (<see cref="PretrainedOptions.AutoWeights"/>): the
/// largest of <see cref="Formats"/> whose weights and working memory fit the memory the device reports free now
/// (<c>Backend.AvailableMemory</c>: its free memory and its pool's reusable blocks, less its reserve), counted from the
/// model's spec before any weight is read. Card-agnostic: every figure is measured or read from the model.
/// </summary>
public static class AutoWeights
{
    /// <summary>The formats tried, the largest first: float32, bfloat16, int8, int4.</summary>
    public static IReadOnlyList<string> Formats { get; } = ["float32", "bfloat16", "int8", "int4"];

    /// <summary>The context the caches are counted for when neither the options nor the load give one: 4096 tokens (<see cref="PretrainedOptions.AutoWeightsContext"/>).</summary>
    public const int DefaultContext = 4096;

    /// <summary>The share of the weights' bytes kept free beside the working memory when the options give none: 0.1 (<see cref="PretrainedOptions.AutoWeightsHeadroom"/>).</summary>
    public const double DefaultHeadroom = 0.1;

    /// <summary>
    /// The bytes <paramref name="spec"/>'s weights take in <paramref name="format"/>: float32 4 a weight; bfloat16 2; int8 1
    /// and its scales (1/16 more); int4 about 5 bits. With int8 and int4 the projections and the head are packed and the
    /// token table stays beside them in bfloat16.
    /// </summary>
    /// <exception cref="ArgumentException">A format not in <see cref="Formats"/>.</exception>
    public static long WeightBytes(DecoderSpec spec, string format)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(format);
        long parameters = spec.ParameterCount;
        long table = (long)spec.Vocabulary * spec.Dim;
        long projections = Math.Max(0, parameters - (spec.TieEmbeddings ? table : 2 * table));
        return format switch
        {
            "float32" => parameters * 4,
            "bfloat16" => parameters * 2,
            "int8" => (long)((projections + table) * 1.0625) + table * 2,
            "int4" => (long)((projections + table) * 0.625) + table * 2,
            _ => throw new ArgumentException($"The weights choice counts {string.Join(", ", Formats)}, not '{format}'.", nameof(format)),
        };
    }

    /// <summary>
    /// The working memory beside the weights: the key/value caches of <paramref name="context"/> tokens (every layer,
    /// float32), a step's scores over the vocabulary, and <paramref name="headroom"/> of <paramref name="weightBytes"/>.
    /// </summary>
    public static long WorkingBytes(DecoderSpec spec, int context, long weightBytes, double headroom)
    {
        ArgumentNullException.ThrowIfNull(spec);
        long caches = 2L * spec.Layers * spec.KvHeads * spec.HeadDim * Math.Max(1, context) * sizeof(float);
        long scores = 2L * spec.Vocabulary * sizeof(float);
        return caches + scores + (long)(Math.Max(0, headroom) * weightBytes);
    }

    /// <summary>
    /// The largest of <paramref name="formats"/> (<see cref="Formats"/> when null) whose weights and working memory for
    /// <paramref name="context"/> tokens fit <paramref name="available"/> bytes (the first when the device reports none).
    /// When none fits: the smallest with <see cref="WeightsChoice.Offload"/> when <paramref name="offload"/> is on, else
    /// an <see cref="InsufficientMemoryException"/> before anything is read, naming what is needed, what is free and what
    /// would fit (a shorter context, offload, a smaller model).
    /// </summary>
    public static WeightsChoice Choose(DecoderSpec spec, int context, long? available, bool offload, double headroom = DefaultHeadroom,
        IReadOnlyList<string>? formats = null, string? model = null)
    {
        ArgumentNullException.ThrowIfNull(spec);
        formats ??= Formats;
        if (formats.Count == 0)
        {
            throw new ArgumentException("The weights choice needs at least one format.", nameof(formats));
        }

        WeightsChoice Of(string format)
        {
            long weights = WeightBytes(spec, format);
            return new WeightsChoice(format, weights, WorkingBytes(spec, context, weights, headroom), available, context);
        }

        var tried = formats.Select(Of).ToList();
        if (tried.FirstOrDefault(c => c.Fits) is { } fits)
        {
            return fits;
        }

        var smallest = tried.MinBy(c => c.NeededBytes)!;
        if (offload)
        {
            return smallest with { Offload = true };
        }

        // What would fit: the longest context the smallest format's weights leave room for.
        long free = available!.Value;
        long perToken = 2L * spec.Layers * spec.KvHeads * spec.HeadDim * sizeof(float);
        long room = free - smallest.WeightBytes - (long)(Math.Max(0, headroom) * smallest.WeightBytes) - 2L * spec.Vocabulary * sizeof(float);
        int longest = perToken > 0 && room > 0 ? (int)Math.Min(int.MaxValue, room / perToken) : 0;
        string hint = longest >= 256
            ? $"{smallest.Format} fits with a context of up to {longest} tokens"
            : $"not even {smallest.Format}'s weights fit";
        throw new InsufficientMemoryException(
            $"{model ?? "The model"} does not fit the device: {smallest.Format}, the smallest format, needs {Gigabytes(smallest.NeededBytes)} "
            + $"({Gigabytes(smallest.WeightBytes)} of weights and {Gigabytes(smallest.WorkingBytes)} of working memory for {context} tokens) and {Gigabytes(free)} is free; "
            + $"{hint}. Give a shorter context, turn on offload (IDRAK_OFFLOAD=1, ComputeResources.OffloadToHostMemory) to keep part of it in system memory, "
            + "pick a smaller model, or free the device. Needed by format: "
            + string.Join(", ", tried.Select(c => $"{c.Format} {Gigabytes(c.NeededBytes)}")) + ".");
    }

    /// <summary>Bytes as gigabytes with one decimal ("8.1 GB", invariant culture).</summary>
    public static string Gigabytes(long bytes) => (bytes / 1e9).ToString("F1", CultureInfo.InvariantCulture) + " GB";
}
