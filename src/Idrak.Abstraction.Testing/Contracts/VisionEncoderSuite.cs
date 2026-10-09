// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a vision encoder (<see cref="IVisionEncoder"/>, a vision family's image side) on images of many sizes:
/// <list type="bullet">
/// <item>deterministic: the same image gives the same features every time;</item>
/// <item>token counts match the blocks: one features per block <see cref="IVisionEncoder.Blocks"/> gives (most families: one
/// per image), each with as many rows as its layout says, the encoder's width, a grid holding that many tokens, and position
/// ids (when given) for every token;</item>
/// <item>batch equals single: an image encodes the same alone and among others;</item>
/// <item>device equals CPU: given the same family's encoder on the CPU, the same layouts and features within the tolerance.</item>
/// </list>
/// Each case is images described as data (sizes, channels and a seed for their pixels, or the index of a sample given).
/// With <c>options</c>, every call gives the family those options (a family that makes several views of an image checks
/// them that way: a tall and a wide image are among the cases).
/// </summary>
/// <param name="cpu">The same family's encoder on the CPU, to compare a device's with; null skips that check.</param>
/// <param name="samples">Images of your own to check first (real ones: the encoder's preprocessing sees what it is for).</param>
/// <param name="tolerance">The largest difference allowed between features that should agree (batch, device).</param>
/// <param name="options">The family's options given to every call (null: none).</param>
public sealed class VisionEncoderSuite(IVisionEncoder? cpu = null, IReadOnlyList<ImageData>? samples = null, float tolerance = 1e-4f, VisionOptions? options = null)
    : ContractSuite<IVisionEncoder>
{
    private readonly IReadOnlyList<ImageData> _samples = samples ?? [];

    /// <inheritdoc />
    public override string Name => "vision-encoder";

    /// <summary>The device the checked encoder runs on (set by <see cref="Conformance.CheckVisionEncoder"/>).</summary>
    public Device On { get; init; } = Device.Cpu;

    /// <inheritdoc />
    public override Device Device => On;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < _samples.Count; i++)
        {
            yield return Case($"sample {i + 1}", [Sample(i)]);
        }

        yield return Case("a colour image wider than tall", [Random(3, 29, 37, 1)]);
        yield return Case("a grey image", [Random(1, 16, 16, 2)]);
        yield return Case("a tiny image", [Random(3, 5, 9, 3)]);
        yield return Case("three images of different sizes", [Random(3, 40, 24, 4), Random(1, 12, 30, 5), Random(3, 64, 64, 6)]);
        yield return Case("a tall image and a wide one", [Random(3, 150, 47, 7), Random(3, 41, 133, 8)]);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int count = random.Next(1, 4), most = large ? 512 : 72;
        var images = Enumerable.Range(0, count).Select(_ => _samples.Count > 0 && random.Next(4) == 0 ? Sample(random.Next(_samples.Count))
            : Random(random.Next(3) == 0 ? 1 : 3, random.Next(4, most), random.Next(4, most), random.Next())).ToArray();
        return Case($"{count} random image{(count == 1 ? "" : "s")}", images);
    }

    /// <inheritdoc />
    public override void Run(IVisionEncoder implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var images = @case.Data["images"]!.AsArray().Select(n => Image(n!.AsObject())).ToList();
        var layouts = images.Select(i => implementation.Blocks(i, options)).ToList();
        var first = Values(implementation, images, checks, layouts);
        var again = Values(implementation, images, null, layouts);
        checks.Check("deterministic", first.Zip(again).All(p => p.First.AsSpan().SequenceEqual(p.Second)), "the same images gave different features");

        string? batch = null;
        for (int i = 0, at = 0; i < images.Count && batch is null; at += layouts[i].Count, i++)
        {
            var alone = Values(implementation, [images[i]], null, [layouts[i]]);
            for (int b = 0; b < alone.Length && batch is null; b++)
            {
                batch = Comparisons.Difference(alone[b], first[at + b], tolerance) is { } difference ? $"image {i} block {b} alone and in the batch: {difference}" : null;
            }
        }

        checks.Compare("batch equals single", batch);
        if (cpu is null)
        {
            checks.Skip("device equals CPU", "no CPU encoder given");
            return;
        }

        var cpuLayouts = images.Select(i => cpu.Blocks(i, options)).ToList();
        if (!cpuLayouts.Zip(layouts).All(p => p.First.SequenceEqual(p.Second)))
        {
            checks.Fail("device equals CPU", $"blocks {Describe(layouts)} on {implementation.Device}, {Describe(cpuLayouts)} on the CPU");
            return;
        }

        var reference = Values(cpu, images, null, cpuLayouts);
        string? device = null;
        for (int i = 0; i < reference.Length && device is null; i++)
        {
            device = Comparisons.Difference(reference[i], first[i], tolerance) is { } difference ? $"block {i}: {difference}" : null;
        }

        checks.Compare("device equals CPU", device);
    }

    private static string Describe(IEnumerable<IReadOnlyList<ImageTokenLayout>> layouts) => string.Join("; ", layouts.Select(l => string.Join(" + ", l)));

    // Each block's features as floats (the images' blocks in order); with checks, the layout rules are recorded.
    private float[][] Values(IVisionEncoder encoder, IReadOnlyList<ImageData> images, CaseChecks? checks, IReadOnlyList<IReadOnlyList<ImageTokenLayout>> layouts)
    {
        var features = encoder.Encode(images, options);
        try
        {
            if (checks is not null)
            {
                var blocks = layouts.SelectMany(l => l).ToList();
                string? problem = layouts.Any(l => l.Count == 0) ? "an image has no blocks"
                    : features.Count != blocks.Count ? $"{features.Count} blocks' features for {images.Count} images of {blocks.Count} blocks" : null;
                for (int i = 0; i < features.Count && problem is null; i++)
                {
                    var f = features[i];
                    var shape = f.Features.Shape;
                    problem = !f.Layout.Equals(blocks[i]) ? $"block {i}: features laid out as {f.Layout}, Blocks says {blocks[i]}"
                        : shape.Length != 2 || shape[0] != blocks[i].Tokens || shape[1] != encoder.Width ? $"block {i}: features {Tensor.FormatShape(shape)} for {blocks[i]} of width {encoder.Width}"
                        : f.Layout.Grid.Count > 0 && f.Layout.Grid.Aggregate(1, (a, v) => a * v) != f.Tokens ? $"block {i}: grid [{string.Join(", ", f.Layout.Grid)}] for {f.Tokens} tokens"
                        : f.Positions is { } p && (p.Rank != 2 || p.Shape[1] != f.Tokens) ? $"block {i}: position ids {Tensor.FormatShape(p.Shape)} for {f.Tokens} tokens"
                        : f.Features.Device != encoder.Device ? $"block {i}: features on {f.Features.Device}, the encoder on {encoder.Device}"
                        : null;
                }

                checks.Compare("token counts match the blocks", problem);
            }

            return [.. features.Select(f => f.Features.ToArray())];
        }
        finally
        {
            foreach (var f in features)
            {
                f.Dispose();
            }
        }
    }

    private static ContractCase Case(string name, JsonObject[] images) => new(name, new JsonObject { ["images"] = new JsonArray([.. images]) });

    private static JsonObject Random(int channels, int height, int width, int seed) =>
        new() { ["channels"] = channels, ["height"] = height, ["width"] = width, ["seed"] = seed };

    private static JsonObject Sample(int index) => new() { ["sample"] = index };

    // The image a case describes: a sample, or smooth gradients with noise (whole 255ths, as decoded 8-bit images are).
    private ImageData Image(JsonObject json)
    {
        if ((int?)json["sample"] is int index)
        {
            return _samples[index];
        }

        int c = (int)json["channels"]!, h = (int)json["height"]!, w = (int)json["width"]!;
        var random = new Random((int)json["seed"]!);
        var pixels = new float[c * h * w];
        for (int ch = 0; ch < c; ch++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    double v = 0.5 + 0.4 * Math.Sin((x + 1.7 * ch) / (2.0 + ch)) * Math.Cos(y / 3.0) + 0.1 * (random.NextDouble() - 0.5);
                    pixels[(ch * h + y) * w + x] = (float)(Math.Clamp(Math.Round(v * 255), 0, 255) / 255.0);
                }
            }
        }

        return new ImageData(pixels, c, h, w);
    }
}
