// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Nlp.Abstractions;

/// <summary>
/// What a cached image feature depends on, all of it: the image's bytes (their SHA-256, <see cref="ChatImage.Hash"/>), the
/// transforms and the vision options it was prepared with (<see cref="TuningImages"/>), and the encoder that made it (its
/// family, a hash of its checkpoint, the precision of its weights), and which stage's output it is. Change any and the
/// cache misses: a cached feature is never one of another image, preparation or encoder.
/// </summary>
/// <param name="Image">The SHA-256 of the image's encoded bytes, as <see cref="ChatImage.Hash"/> gives it.</param>
/// <param name="Transforms">The image transforms' text (<see cref="ImageTransformPipeline.ToString"/>; "" for none).</param>
/// <param name="VisionOptions">The vision options' text (<see cref="VisionOptions.ToString"/>; "none" for none).</param>
/// <param name="Family">The vision family's name (the architecture name).</param>
/// <param name="Checkpoint">A hash identifying the checkpoint's weights (any text that changes when they do).</param>
/// <param name="DType">The precision the encoder computes and keeps its weights in ("float32", "bfloat16").</param>
/// <param name="Stage">Which output is cached: <see cref="TowerStage"/> (the frozen tower's) unless another is named.</param>
public sealed record FeatureCacheKey(string Image, string Transforms, string VisionOptions, string Family, string Checkpoint, string DType, string Stage = FeatureCacheKey.TowerStage)
{
    /// <summary>The vision tower's output (before the projector): what a frozen tower lets the tuner keep.</summary>
    public const string TowerStage = "tower";

    /// <summary>The key of <paramref name="image"/> prepared by <paramref name="images"/> and encoded by the encoder described.</summary>
    public static FeatureCacheKey For(ChatImage image, TuningImages images, string family, string checkpoint, string dtype, string stage = TowerStage)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(images);
        return new FeatureCacheKey(image.Hash, images.Pipeline.ToString(), images.VisionOptions.ToString(), family, checkpoint, dtype, stage);
    }

    /// <summary>
    /// The key as 64 lowercase hex digits (the SHA-256 of its fields written as a JSON array): a file name for a cache that
    /// keeps entries on disk; two keys with the same <see cref="Id"/> are equal.
    /// </summary>
    public string Id => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(ToJson().ToJsonString())));

    /// <summary>The fields as a JSON array, in the order of the constructor.</summary>
    public JsonArray ToJson() => [Image, Transforms, VisionOptions, Family, Checkpoint, DType, Stage];
}

/// <summary>How a cache of <see cref="FeatureCaches"/> is made.</summary>
public sealed record FeatureCacheOptions
{
    /// <summary>
    /// Where a cache that keeps entries on disk keeps them (<see cref="FeatureCaches.DefaultFolder"/> when null: under
    /// <c>IDRAK_CACHE</c>).
    /// </summary>
    public string? Folder { get; init; }

    /// <summary>
    /// The most bytes of values the cache keeps. Null: the cache decides from what the machine reports (the memory cache:
    /// half of the system memory available when it is made); the disk cache keeps what the disk takes.
    /// </summary>
    public long? Budget { get; init; }
}

/// <summary>
/// A store of image features (the frozen vision tower's output, so a fine-tune encodes each image once, not every epoch),
/// by <see cref="FeatureCacheKey"/>. Values go in as tensors on any device and come out as new tensors on the device asked
/// for, which the caller owns; the cache keeps its own copy (float32), so what was put can be disposed. A cache may forget
/// entries (a bounded one evicts the least recently used): a miss is never an error, and a hit is exactly what was put.
/// Thread-safe. Made by <see cref="FeatureCaches.Create"/>; the testing kit checks one (<c>Conformance.CheckFeatureCache</c>).
/// </summary>
public interface IFeatureCache : IDisposable
{
    /// <summary>The cache's kind, as registered ("memory", "disk").</summary>
    string Name { get; }

    /// <summary>The entries kept.</summary>
    int Count { get; }

    /// <summary>The bytes of values kept (float32: 4 per element).</summary>
    long Bytes { get; }

    /// <summary>Whether <paramref name="key"/> is kept.</summary>
    bool Contains(FeatureCacheKey key);

    /// <summary>The values kept under <paramref name="key"/>, as a new tensor on <paramref name="device"/> (the caller disposes it); false on a miss.</summary>
    bool TryGet(FeatureCacheKey key, Device device, [NotNullWhen(true)] out Tensor? value);

    /// <summary>
    /// Keeps a copy of <paramref name="value"/>'s values and shape under <paramref name="key"/>, replacing what was there.
    /// False when the cache does not keep it (larger than its budget, or the disk refused the write); never throws for that.
    /// </summary>
    bool Put(FeatureCacheKey key, Tensor value);

    /// <summary>Forgets every entry.</summary>
    void Clear();
}

/// <summary>
/// The kinds of feature cache, by name (ignoring case). The library's:
/// <list type="bullet">
/// <item><see cref="Memory"/>: in this process's memory, least recently used entries evicted to stay within a byte
/// budget (<see cref="FeatureCacheOptions.Budget"/>, or half of the system memory available when it is made).</item>
/// <item><see cref="Disk"/>: one safetensors file per entry (float32, the key in its metadata) in a folder
/// (<see cref="FeatureCacheOptions.Folder"/>, or <see cref="DefaultFolder"/>), written whole before it replaces anything,
/// so a cache survives the process and serves later runs and other processes.</item>
/// </list>
/// Register another with <see cref="Register"/>; one under a library name shadows the library's (<see cref="SetPolicy"/>:
/// under <see cref="SlotPolicy.FallBack"/> the library's is made when the app's cannot be; a cache holds entries across
/// calls, so it falls back when it is made, not per call, and under <see cref="SlotPolicy.Shadow"/> only the library's is
/// made, since making one may create its folder).
/// </summary>
public static class FeatureCaches
{
    /// <summary>The cache in memory.</summary>
    public const string Memory = "memory";

    /// <summary>The cache on disk.</summary>
    public const string Disk = "disk";

    private static readonly SlotTable<string, Func<FeatureCacheOptions, IFeatureCache>> Table = BuiltIn();

    private static SlotTable<string, Func<FeatureCacheOptions, IFeatureCache>> BuiltIn()
    {
        var table = new SlotTable<string, Func<FeatureCacheOptions, IFeatureCache>>(nameof(FeatureCaches),
            (slot, app, library) => options => slot.Call(() => app(options), () => library(options), effects: true), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(Memory, options => new MemoryFeatureCache(options));
        table.RegisterDefault(Disk, options => new DiskFeatureCache(options));
        return table;
    }

    /// <summary>
    /// The folder the disk cache uses unless told otherwise: <c>features</c> under the library's cache folder
    /// (<c>IDRAK_CACHE</c>, or <c>~/.cache/idrak</c>).
    /// </summary>
    public static string DefaultFolder => Path.Combine(
        Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } cache
            ? cache
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile, Environment.SpecialFolderOption.DoNotVerify), ".cache", "idrak"),
        "features");

    /// <summary>Registers the cache <paramref name="name"/>, made by <paramref name="create"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, Func<FeatureCacheOptions, IFeatureCache> create)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(create);
        Table.Register(name, create, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's cache <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>What makes the cache <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names the registered ones.</exception>
    public static Func<FeatureCacheOptions, IFeatureCache> Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Feature cache '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with FeatureCaches.Register.");

    /// <summary>A new cache of the kind <paramref name="name"/>, made with <paramref name="options"/> (the kind's defaults when null). Dispose it when done.</summary>
    public static IFeatureCache Create(string name, FeatureCacheOptions? options = null) => Get(name)(options ?? new FeatureCacheOptions());

    /// <summary>The library's cache <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static Func<FeatureCacheOptions, IFeatureCache>? Default(string name) => Table.Default(name);

    /// <summary>Who registered the cache <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's cache <paramref name="name"/> cannot be made (<see cref="SlotPolicy.Throw"/> unless set; <see cref="SlotPolicy.FallBack"/> makes the library's).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);
}
