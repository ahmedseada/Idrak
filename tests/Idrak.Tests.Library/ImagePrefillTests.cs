// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Layers;
using Idrak.Models;

// Image tokens in Gemma 3's decoder (plan 11, phase 5): phase 0's projected image features put into the prompt give
// transformers' logits and greedy tokens, with and without the KV cache; several images, batches and rows of different
// lengths agree with single runs.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ImagePrefillGroup =
    [
        ("image prefill: the tiny Gemma 3 with phase 0's image features gives transformers' prompt logits and 20 greedy tokens (cached and not, every layout); without the image-block mask the logits change from the first image token on", ImagePrefillMatchesReference),
    ];

    internal static (int[] Ids, float[] Features) ImagePromptReference()
    {
        int[] ids = [.. ReadNpyInt64(TestData("vlm/reference/prompt-image-input_ids.npy")).Select(i => (int)i)];
        return (ids, ReadNpyFloat32(TestData("vlm/reference/image_features.npy")));
    }

    internal static Tensor Ids(IReadOnlyList<int> ids, Device device, int rows = 1) => Tensor.From([.. ids.Select(i => (float)i)], [rows, ids.Count / rows], device);

    internal static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }


    internal static void ImagePrefillMatchesReference(Device device)
    {
        RegisterGemma3Vision();
        var (ids, featureValues) = ImagePromptReference();
        float[] expected = ReadNpyFloat32(TestData("vlm/reference/prompt-image-logits.npy"));
        float[] expectedSteps = ReadNpyFloat32(TestData("vlm/reference/prompt-image-generate-logits.npy"));
        var facts = JsonNode.Parse(File.ReadAllText(TestData("vlm/manifest.json")))!["facts"]!;
        int[] newTokens = [.. facts["generate"]!["new_tokens"]!.AsArray().Select(n => (int)n!)];
        float unmaskedChange = (float)facts["image_prompt"]!["max_abs_logit_change_without_token_type_ids"]!;
        const int vocabulary = 366;
        using var noGrad = Autograd.NoGrad();
        foreach (string folder in new[] { "tiny-gemma3-pre452", "tiny-gemma3", "tiny-gemma3-v5" })
        {
            using var model = PretrainedModel.Load(TestData($"vlm/{folder}"), new PretrainedOptions { Device = device });
            var decoder = model.Network;
            var tokens = ((Gemma3Vision)model.Vision!).ImageTokens;
            var rule = model.Vision!.Attention;
            using var features = Tensor.From(featureValues, [1, 4, 24], device);
            var images = ImagePrefill.Locate(ids, tokens.ImageToken, [features]);
            Check(images is [{ Position: 14, Tokens: 4, Sequence: 0 }], $"{folder}: the image's tokens at {string.Join(", ", images.Select(i => i.Position))}");
            using var input = Ids(ids, device);

            // One pass without the cache.
            var whole = decoder.Forward(input, images, rule).ToArray();
            float wholeDifference = MaxDifference(expected, whole);
            AssertClose(expected, whole, 2e-5f, $"{folder}: prompt logits without the cache");

            // The cached prefill, then 20 greedy steps over the cache.
            using var context = new DecodingContext(device, batch: 1, capacity: 64);
            var prefill = decoder.ForwardCached(input, images, rule, context).ToArray();
            float prefillDifference = MaxDifference(expected, prefill);
            AssertClose(expected, prefill, 2e-5f, $"{folder}: prompt logits of the cached prefill");
            var generated = new List<int>();
            var stepLogits = new List<float>();
            float[] last = prefill[^vocabulary..];
            for (int step = 0; step < newTokens.Length; step++)
            {
                stepLogits.AddRange(last);
                int token = ArgMax(last);
                generated.Add(token);
                using var next = Ids([token], device);
                last = decoder.ForwardCached(next, context).ToArray();
            }

            Check(generated.SequenceEqual(newTokens), $"{folder}: greedy tokens {string.Join(" ", generated)}, expected {string.Join(" ", newTokens)}");
            float stepDifference = MaxDifference(expectedSteps, [.. stepLogits]);
            AssertClose(expectedSteps, [.. stepLogits], 2e-5f, $"{folder}: the greedy steps' logits");

            // A pass without autograd copies each window's key ranges (and the composed path's mask) to the device once, not
            // once per attention layer, and frees the copies when it ends; with autograd each layer gets tensors of its own.
            var blocks = new ImageBlocks(1, ids.Length, 24, images, rule, segments: null) { KeepOnDevice = true };
            var (starts, ends) = blocks.Spans(0, 0, null, device);
            var mask = blocks.Mask(0, 0, null, ids.Length, device);
            Check(ReferenceEquals(starts, blocks.Spans(0, 0, null, device).Starts) && ReferenceEquals(ends, blocks.Spans(0, 0, null, device).Ends)
                  && ReferenceEquals(mask, blocks.Mask(0, 0, null, ids.Length, device)), $"{folder}: the next layer reads the same device copies");
            Check(!ReferenceEquals(starts, blocks.Spans(0, 1, null, device).Starts), $"{folder}: another offset is another table");
            blocks.ReleaseDevice();
            Check(starts.IsDisposed && ends.IsDisposed && mask!.IsDisposed, $"{folder}: the copies are freed with the pass");
            blocks.KeepOnDevice = false;
            Check(!ReferenceEquals(blocks.Spans(0, 0, null, device).Starts, blocks.Spans(0, 0, null, device).Starts), $"{folder}: kept only while asked");

            // One pass without the cache over the prompt and the generated tokens: the same logits.
            using var full = Ids([.. ids, .. generated[..^1]], device);
            var fullLogits = decoder.Forward(full, images, rule).ToArray();
            AssertClose(expectedSteps, fullLogits[((ids.Length - 1) * vocabulary)..], 2e-5f, $"{folder}: one pass over prompt and answer");

            // Without the image-block mask (each image token a block of its own: causal), the logits change from the first
            // image token on, by what transformers measured without token_type_ids, and not before.
            var causal = Enumerable.Range(0, 4).Select(i => new PromptImage(14 + i, features.Reshape(4, 24).Narrow(0, i, 1))).ToList();
            var unmasked = decoder.Forward(input, causal, rule).ToArray();
            float before = MaxDifference(expected[..(14 * vocabulary)], unmasked[..(14 * vocabulary)]);
            float after = MaxDifference(expected[(14 * vocabulary)..(15 * vocabulary)], unmasked[(14 * vocabulary)..(15 * vocabulary)]);
            float change = MaxDifference(expected, unmasked);
            Check(before < 2e-5f && after > 1e-2f && MathF.Abs(change - unmaskedChange) < 1e-3f,
                $"{folder}: without the mask, {before:G3} before the image, {after:G3} at its first token, {change:G4} at most (transformers: {unmaskedChange:G4})");
            Console.WriteLine($"    {folder} on {device}: largest logit difference {wholeDifference:G3} (one pass), {prefillDifference:G3} (cached prefill), {stepDifference:G3} (20 greedy steps); tokens {string.Join(" ", generated)}; without the mask {change:G4}");
        }
    }

}
