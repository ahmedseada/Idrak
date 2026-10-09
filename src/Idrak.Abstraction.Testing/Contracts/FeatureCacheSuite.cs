// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The fields of a feature cache's key, as the cache checks see them (Idrak.Nlp's <c>FeatureCacheKey</c>, field for field):
/// the image's hash, its transforms and vision options, the encoder's family, checkpoint and precision, and the stage.
/// </summary>
/// <param name="Image">The SHA-256 of the image's bytes.</param>
/// <param name="Transforms">The image transforms' text.</param>
/// <param name="VisionOptions">The vision options' text.</param>
/// <param name="Family">The vision family.</param>
/// <param name="Checkpoint">The checkpoint's hash.</param>
/// <param name="DType">The encoder's precision.</param>
/// <param name="Stage">Which output is cached.</param>
public sealed record FeatureKeyFields(string Image, string Transforms, string VisionOptions, string Family, string Checkpoint, string DType, string Stage);

/// <summary>
/// A feature cache as the cache checks drive it. The contract lives with the package that uses it (<c>IFeatureCache</c>
/// in Idrak.Nlp) and this kit depends on Idrak.Abstraction alone, so a test wraps the cache it checks in these delegates
/// (its key made from <see cref="FeatureKeyFields"/>).
/// </summary>
/// <param name="put">Keeps a copy of a tensor under a key (<c>Put</c>); false when not kept.</param>
/// <param name="get">The tensor kept under a key, made on the device given, or null on a miss (<c>TryGet</c>).</param>
/// <param name="clear">Forgets every entry (<c>Clear</c>).</param>
/// <param name="owner">Disposed with this (the cache itself), or null.</param>
public sealed class FeatureCacheUnderTest(Func<FeatureKeyFields, Tensor, bool> put, Func<FeatureKeyFields, Device, Tensor?> get, Action clear, IDisposable? owner = null)
    : IDisposable
{
    /// <summary>Keeps a copy of <paramref name="value"/> under <paramref name="key"/>; false when not kept.</summary>
    public bool Put(FeatureKeyFields key, Tensor value) => put(key, value);

    /// <summary>The tensor kept under <paramref name="key"/>, on <paramref name="device"/> (the caller's), or null.</summary>
    public Tensor? Get(FeatureKeyFields key, Device device) => get(key, device);

    /// <summary>Forgets every entry.</summary>
    public void Clear() => clear();

    /// <summary>Disposes the cache.</summary>
    public void Dispose() => owner?.Dispose();
}

/// <summary>
/// The checks of a feature cache (Idrak.Nlp's <c>IFeatureCache</c>), wrapped as <see cref="FeatureCacheUnderTest"/>, on
/// small tensors of many shapes (each case clears the cache first):
/// <list type="bullet">
/// <item>an empty cache misses;</item>
/// <item>a small value is kept, and a hit gives back exactly its values and shape, on the device asked for;</item>
/// <item>the cache keeps its own copy (the tensor put can be disposed), and every hit is a new tensor of the caller's;</item>
/// <item>a key differing in any one field (image, transforms, vision options, family, checkpoint, precision, stage)
/// misses;</item>
/// <item>a second put under the same key replaces the first;</item>
/// <item>puts and gets from several threads at once: every hit is exactly what was put under its key;</item>
/// <item>after a clear, the key misses.</item>
/// </list>
/// </summary>
/// <param name="device">Where the values put live and hits are asked for (the CPU when null).</param>
public sealed class FeatureCacheSuite(Device? device = null) : ContractSuite<FeatureCacheUnderTest>
{
    private static readonly string[] FieldNames = ["image", "transforms", "vision options", "family", "checkpoint", "dtype", "stage"];

    /// <summary>Where the values put live and hits are asked for.</summary>
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <inheritdoc />
    public override string Name => "feature-cache";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one value", [1], 1, Fields("a", "", "none"), threads: 1);
        yield return Case("a row of features", [16], 2, Fields("b", "grayscale", "none"), threads: 1);
        yield return Case("tower output [1, 16, 8]", [1, 16, 8], 3, Fields("c", "grayscale,max_width=1024,contrast=1.5", "do_pan_and_scan=true"), threads: 2);
        yield return Case("three blocks [3, 4, 5]", [3, 4, 5], 4, Fields("d", "", "none"), threads: 4);
        yield return Case("Arabic and odd text in the key", [2, 3], 5, Fields("e", "contrast=1.5", "خيار=نعم, crops=4\n"), threads: 2);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int rank = random.Next(1, 4), limit = large ? 64 : 9;
        int[] shape = [.. Enumerable.Range(0, rank).Select(_ => random.Next(1, limit))];
        string Text() => new([.. Enumerable.Range(0, random.Next(0, 12)).Select(_ => (char)(random.Next(3) == 0 ? random.Next(0x600, 0x6FF) : random.Next(0x20, 0x7F)))]);
        var fields = new FeatureKeyFields(Convert.ToHexStringLower(BitConverter.GetBytes(random.NextInt64())), Text(), Text(), Text(), Text(), random.Next(2) == 0 ? "float32" : "bfloat16",
            random.Next(4) == 0 ? "features" : "tower");
        return Case($"random [{string.Join(", ", shape)}]", shape, random.Next(), fields, threads: random.Next(1, 5));
    }

    /// <inheritdoc />
    public override void Run(FeatureCacheUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int[] shape = [.. @case.Data["shape"]!.AsArray().Select(n => (int)n!)];
        int seed = (int)@case.Data["seed"]!, threads = (int)@case.Data["threads"]!;
        var key = ReadFields(@case.Data["fields"]!.AsObject());
        float[] values = Values(shape, seed);

        implementation.Clear();
        using (var missed = implementation.Get(key, Device))
        {
            checks.Check("an empty cache misses", missed is null, "a cleared cache gave a value");
        }

        var source = Tensor.From(values, shape, Device);
        bool kept = implementation.Put(key, source);
        source.Dispose();                                                      // the cache keeps its own copy
        checks.Check("keeps a small value", kept, $"a value of {values.Length} elements was not kept");
        if (!kept)
        {
            return;
        }

        var first = implementation.Get(key, Device);
        var second = implementation.Get(key, Device);
        try
        {
            checks.Check("hits after a put", first is not null && second is not null, "missed right after the put");
            if (first is null || second is null)
            {
                return;
            }

            checks.Check("a hit is on the device asked for", first.Device == Device, $"on {first.Device}, asked {Device}");
            checks.Compare("a hit gives back the shape put", Comparisons.Exact(string.Join("x", shape), string.Join("x", first.Shape.ToArray())));
            checks.Compare("a hit gives back exactly the values put", Comparisons.Difference(values, first.ToArray(), 0f));
            checks.Check("each hit is a new tensor", !ReferenceEquals(first, second), "two hits returned the same tensor object");
            first.Dispose();
            first = null;
            checks.Compare("a hit outlives another one's disposal", Comparisons.Difference(values, second.ToArray(), 0f));
        }
        finally
        {
            first?.Dispose();
            second?.Dispose();
        }

        for (int f = 0; f < FieldNames.Length; f++)
        {
            using var other = implementation.Get(Change(key, f), Device);
            checks.Check($"a key differing in its {FieldNames[f]} misses", other is null, $"a key with another {FieldNames[f]} hit");
        }

        float[] replaced = Values(shape, seed + 1);
        using (var again = Tensor.From(replaced, shape, Device))
        {
            implementation.Put(key, again);
        }

        using (var now = implementation.Get(key, Device))
        {
            checks.Check("a second put replaces the first", now is not null && Comparisons.Difference(replaced, now.ToArray(), 0f) is null, "the key gave the first value, or missed");
        }

        // Several threads, each its own key (the case's with another image) and value, putting and reading back.
        string? wrong = null;
        int hits = 0;
        Parallel.For(0, Math.Max(2, threads * 2), new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, threads) }, i =>
        {
            var mine = key with { Image = $"{key.Image}-{i}" };
            float[] own = Values(shape, seed * 31 + i);
            using (var put = Tensor.From(own, shape, Device))
            {
                implementation.Put(mine, put);
            }

            for (int round = 0; round < 3; round++)
            {
                using var got = implementation.Get(mine, Device);
                if (got is null)
                {
                    continue;
                }

                Interlocked.Increment(ref hits);
                if (Comparisons.Difference(own, got.ToArray(), 0f) is { } difference)
                {
                    Interlocked.CompareExchange(ref wrong, $"thread {i}: {difference}", null);
                }
            }
        });
        checks.Check("from several threads, every hit is what was put under its key", wrong is null, wrong ?? "");
        checks.Check("from several threads, values put are found", hits > 0, "no thread found its value");

        implementation.Clear();
        using (var cleared = implementation.Get(key, Device))
        {
            checks.Check("misses after a clear", cleared is null, "the key hit after Clear");
        }
    }

    private static FeatureKeyFields Fields(string image, string transforms, string options) =>
        new(Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(image))), transforms, options,
            "TinyFamilyForConditionalGeneration", "0123abcd", "float32", "tower");

    // The key with field `f` changed (a character added).
    private static FeatureKeyFields Change(FeatureKeyFields key, int f) => f switch
    {
        0 => key with { Image = key.Image + "0" },
        1 => key with { Transforms = key.Transforms + "," },
        2 => key with { VisionOptions = key.VisionOptions + " " },
        3 => key with { Family = key.Family + "X" },
        4 => key with { Checkpoint = key.Checkpoint + "1" },
        5 => key with { DType = key.DType == "float32" ? "bfloat16" : "float32" },
        _ => key with { Stage = key.Stage + "+" },
    };

    private static float[] Values(int[] shape, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(random.NextDouble() * 20 - 10))];
    }

    private static ContractCase Case(string name, int[] shape, int seed, FeatureKeyFields fields, int threads) => new(name, new JsonObject
    {
        ["shape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
        ["seed"] = seed,
        ["threads"] = threads,
        ["fields"] = new JsonObject
        {
            ["image"] = fields.Image, ["transforms"] = fields.Transforms, ["vision_options"] = fields.VisionOptions, ["family"] = fields.Family,
            ["checkpoint"] = fields.Checkpoint, ["dtype"] = fields.DType, ["stage"] = fields.Stage,
        },
    });

    private static FeatureKeyFields ReadFields(JsonObject json) =>
        new((string)json["image"]!, (string)json["transforms"]!, (string)json["vision_options"]!, (string)json["family"]!, (string)json["checkpoint"]!,
            (string)json["dtype"]!, (string)json["stage"]!);
}
