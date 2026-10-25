// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Networking.Abstractions;

/// <summary>
/// One piece of a gradient bucket: values [<paramref name="Offset"/>, <paramref name="Offset"/> + <paramref name="Count"/>)
/// of parameter <paramref name="Parameter"/> (its index in the optimizer's list), at <paramref name="At"/> in the bucket.
/// </summary>
public readonly record struct GradientPiece(int Parameter, int Offset, int Count, int At);

/// <summary>
/// One rank's compression of gradient buckets on their way across the ranks (plan 23, phase C4): it sums a bucket of this
/// rank's gradient values over every rank, sending less than the values where it can. Made once per run by its
/// <see cref="IGradientCompression"/>; any state it keeps (top-k's error-feedback residual) is allocated once and lives
/// with it. <see cref="Reduce"/> is called one bucket at a time, in bucket order, the same buckets on every rank, from the
/// thread that reduces the step's buckets (not the training thread while the backward pass overlaps): it works on the
/// host values it is given and on the collectives' host values (<see cref="IHostValueCollectives"/>).
/// </summary>
public interface IGradientCompressor : IDisposable
{
    /// <summary>The compression's name.</summary>
    string Name { get; }

    /// <summary>
    /// Replaces <paramref name="values"/> (this rank's gradient values of a bucket, as <paramref name="pieces"/> lay them out)
    /// with the sum over every rank of each rank's values times its <paramref name="weight"/> (its rows in the step, or 1),
    /// as compressed: the same values on every rank afterwards.
    /// </summary>
    void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces);

    /// <summary>
    /// The pieces were summed again in full, uncompressed (their gradient changed after they were sent): whatever the
    /// compression kept of them this step is forgotten (top-k's residual).
    /// </summary>
    void Forget(ReadOnlySpan<GradientPiece> pieces)
    {
    }
}

/// <summary>One kind of gradient compression, registered by name in <see cref="GradientCompressions"/>.</summary>
public interface IGradientCompression
{
    /// <summary>The name ("none", "bf16", "top-k").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>A rank's compressor of the gradients of <paramref name="parameters"/> over <paramref name="collectives"/>.</summary>
    /// <exception cref="ArgumentException">A setting it needs is missing (top-k's share).</exception>
    IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options);
}

/// <summary>
/// The gradient compressions, by name (plan 23, phase C4): "none" (float32), "bf16" (bfloat16 on the wire, sums in
/// float32) and "top-k" (the largest share of each bucket's values, by magnitude, with the rest kept for the next step:
/// error feedback). A run names one with <see cref="ClusterOptions.Compression"/> (<c>IDRAK_CLUSTER_COMPRESSION</c>).
/// </summary>
public static class GradientCompressions
{
    private static readonly SlotTable<string, IGradientCompression> Table = BuiltIn();

    private static SlotTable<string, IGradientCompression> BuiltIn()
    {
        var table = new SlotTable<string, IGradientCompression>(nameof(GradientCompressions), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a compression's values are summed with every other rank's in the same form, so another cannot answer for one rank");
        table.RegisterDefault("none", new Compression.PlainCompression());
        table.RegisterDefault("bf16", new Compression.BFloat16Compression());
        table.RegisterDefault("top-k", new Compression.TopKCompression());
        return table;
    }

    /// <summary>The registered compressions, in registration order.</summary>
    public static IReadOnlyList<IGradientCompression> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="compression"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGradientCompression compression)
    {
        ArgumentNullException.ThrowIfNull(compression);
        ArgumentException.ThrowIfNullOrWhiteSpace(compression.Name);
        Table.Register(compression.Name, compression, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's compression of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The compression of that name, or null.</summary>
    public static IGradientCompression? Find(string name) => Table.Find(name);

    /// <summary>The compression of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IGradientCompression Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(GradientCompressions), name, Table.Keys,
            $"Gradient compression '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add one with GradientCompressions.Register (IDRAK_CLUSTER_COMPRESSION names it).");

    /// <summary>The library's compression of that name, or null.</summary>
    public static IGradientCompression? Default(string name) => Table.Default(name);

    /// <summary>Who registered the compression of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each compression as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(c => $"{c.Name}: {c.Summary}"));

    /// <summary>The compressor <paramref name="options"/> names (<see cref="ClusterOptions.CompressionName"/>).</summary>
    public static IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(options);
        return Get(options.CompressionName).Create(collectives, parameters, options);
    }
}
