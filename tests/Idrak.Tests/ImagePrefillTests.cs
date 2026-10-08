// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Models;

// Image tokens in Gemma 3's decoder (plan 11, phase 5): phase 0's projected image features put into the prompt give
// transformers' logits and greedy tokens, with and without the KV cache; several images, batches and rows of different
// lengths agree with single runs.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ImagePrefillGroup =
    [
        ("image prefill: the tiny Gemma 3 with phase 0's image features gives transformers' prompt logits and 20 greedy tokens (cached and not, every layout); without the image-block mask the logits change from the first image token on", ImagePrefillMatchesReference),
        ("image prefill: two images in one prompt, a continued prefill, a batch and rows of different lengths agree with single runs; other cache formats; text prompts unchanged", ImagePrefillConsistency),
    ];

    private static (int[] Ids, float[] Features) ImagePromptReference()
    {
        int[] ids = [.. ReadNpyInt64(TestData("vlm/reference/prompt-image-input_ids.npy")).Select(i => (int)i)];
        return (ids, ReadNpyFloat32(TestData("vlm/reference/image_features.npy")));
    }

    private static Tensor Ids(IReadOnlyList<int> ids, Device device, int rows = 1) => Tensor.From([.. ids.Select(i => (float)i)], [rows, ids.Count / rows], device);

    private static int ArgMax(ReadOnlySpan<float> values)
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

    private static float MaxDifference(float[] a, float[] b) => a.Zip(b, (x, y) => MathF.Abs(x - y)).Max();

    private static void ImagePrefillMatchesReference(Device device)
    {
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
            var tokens = model.Vision!.ImageTokens;
            using var features = Tensor.From(featureValues, [1, 4, 24], device);
            var images = ImagePrefill.Locate(ids, tokens.ImageToken, [features]);
            Check(images is [{ Position: 14, Tokens: 4, Sequence: 0 }], $"{folder}: the image's tokens at {string.Join(", ", images.Select(i => i.Position))}");
            using var input = Ids(ids, device);

            // One pass without the cache.
            var whole = decoder.Forward(input, images).ToArray();
            float wholeDifference = MaxDifference(expected, whole);
            AssertClose(expected, whole, 2e-5f, $"{folder}: prompt logits without the cache");

            // The cached prefill, then 20 greedy steps over the cache.
            using var context = new DecodingContext(device, batch: 1, capacity: 64);
            var prefill = decoder.ForwardCached(input, images, context).ToArray();
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

            // One pass without the cache over the prompt and the generated tokens: the same logits.
            using var full = Ids([.. ids, .. generated[..^1]], device);
            var fullLogits = decoder.Forward(full, images).ToArray();
            AssertClose(expectedSteps, fullLogits[((ids.Length - 1) * vocabulary)..], 2e-5f, $"{folder}: one pass over prompt and answer");

            // Without the image-block mask (each image token a block of its own: causal), the logits change from the first
            // image token on, by what transformers measured without token_type_ids, and not before.
            var causal = Enumerable.Range(0, 4).Select(i => new PromptImage(14 + i, features.Reshape(4, 24).Narrow(0, i, 1))).ToList();
            var unmasked = decoder.Forward(input, causal).ToArray();
            float before = MaxDifference(expected[..(14 * vocabulary)], unmasked[..(14 * vocabulary)]);
            float after = MaxDifference(expected[(14 * vocabulary)..(15 * vocabulary)], unmasked[(14 * vocabulary)..(15 * vocabulary)]);
            float change = MaxDifference(expected, unmasked);
            Check(before < 2e-5f && after > 1e-2f && MathF.Abs(change - unmaskedChange) < 1e-3f,
                $"{folder}: without the mask, {before:G3} before the image, {after:G3} at its first token, {change:G4} at most (transformers: {unmaskedChange:G4})");
            Console.WriteLine($"    {folder} on {device}: largest logit difference {wholeDifference:G3} (one pass), {prefillDifference:G3} (cached prefill), {stepDifference:G3} (20 greedy steps); tokens {string.Join(" ", generated)}; without the mask {change:G4}");
        }
    }

    private static void ImagePrefillConsistency(Device device)
    {
        var (ids, featureValues) = ImagePromptReference();
        const int vocabulary = 366;
        using var noGrad = Autograd.NoGrad();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        var decoder = model.Network;
        int imageToken = model.Vision!.ImageTokens.ImageToken;
        using var first = Tensor.From(featureValues, [1, 4, 24], device);
        using var second = first * -0.5f;

        // The prompt with a second image after the first ("\n\n<boi>" + 4 image tokens + "<eoi>\n\n" again), 47 tokens.
        int[] two = [.. ids[..20], .. ids[11..20], .. ids[20..]];
        Check(two.Length == 47 && two.Count(i => i == imageToken) == 8, $"two-image prompt: {string.Join(" ", two)}");
        var images = ImagePrefill.Locate(two, imageToken, [first, second]);
        Check(images.Select(i => i.Position).SequenceEqual([14, 23]), $"two images at {string.Join(", ", images.Select(i => i.Position))}");
        using var input = Ids(two, device);
        var whole = decoder.Forward(input, images).ToArray();

        // Cached, in one step and continued after a cached prefix (the second image in the second step).
        using (var context = new DecodingContext(device, 1, 64))
        {
            AssertClose(whole, decoder.ForwardCached(input, images, context).ToArray(), 2e-5f, "two images: cached prefill against one pass");
        }

        using (var context = new DecodingContext(device, 1, 64))
        {
            using var head = Ids(two[..20], device);
            using var tail = Ids(two[20..], device);
            var a = decoder.ForwardCached(head, [images[0]], context).ToArray();
            var b = decoder.ForwardCached(tail, [images[1] with { Position = 3 }], context).ToArray();
            AssertClose(whole, [.. a, .. b], 2e-5f, "two images: a prefill continuing a cached prefix");
        }

        // The first image's rows do not depend on what follows (rows before the second image equal the one-image prompt's).
        using (var single = Ids(ids, device))
        {
            var one = decoder.Forward(single, ImagePrefill.Locate(ids, imageToken, [first])).ToArray();
            AssertClose(one[..(20 * vocabulary)], whole[..(20 * vocabulary)], 2e-5f, "two images: the prefix as with one image");
            var same = decoder.Forward(input, ImagePrefill.Locate(two, imageToken, [first, first])).ToArray();
            AssertClose(same[..(23 * vocabulary)], whole[..(23 * vocabulary)], 2e-5f, "two images: rows before the second image do not read it");
            Check(MaxDifference(same[(23 * vocabulary)..], whole[(23 * vocabulary)..]) > 1e-3f, "two images: the second image's features reach what follows");
        }

        // A batch: the same prompt with the images swapped in its second row.
        var swapped = ImagePrefill.Locate(two, imageToken, [second, first], sequence: 1);
        var alone = decoder.Forward(input, ImagePrefill.Locate(two, imageToken, [second, first])).ToArray();
        using (var batch = Ids([.. two, .. two], device, rows: 2))
        {
            var both = decoder.Forward(batch, [.. images, .. swapped]).ToArray();
            AssertClose([.. whole, .. alone], both, 2e-5f, "a batch of two prompts without the cache");
            using var context = new DecodingContext(device, 2, 64);
            AssertClose([.. whole, .. alone], decoder.ForwardCached(batch, [.. images, .. swapped], context).ToArray(), 2e-5f, "a batch of two prompts, cached");
        }

        // Rows of different lengths: the one-image prompt left-padded beside the two-image one.
        using (var single = Ids(ids, device))
        using (var padded = Ids([.. two, .. new int[two.Length - ids.Length], .. ids], device, rows: 2))
        using (var context = new DecodingContext(device, 2, 64))
        {
            int pad = two.Length - ids.Length;
            var one = decoder.Forward(single, ImagePrefill.Locate(ids, imageToken, [first])).ToArray();
            context.SetRowStarts([0, pad]);
            var rows = decoder.ForwardCached(padded, [.. images, new PromptImage(14 + pad, first) { Sequence = 1 }], context).ToArray();
            AssertClose(whole, rows[..whole.Length], 2e-5f, "rows of different lengths: the full row");
            AssertClose(one, rows[(whole.Length + pad * vocabulary)..], 2e-5f, "rows of different lengths: the padded row");
        }

        // bfloat16 and int8 caches: as close to the float32 cache as the plain prefill of a text prompt is with them (the
        // tiny model's heads of 8 values round coarsely in int8).
        long[] textIds = ReadNpyInt64(TestData("vlm/reference/prompt-text-input_ids.npy"));
        using var textInput = Ids([.. textIds.Select(i => (int)i)], device);
        var textExact = decoder.Forward(textInput).ToArray();
        // A one-token "image" holding a text token's own scaled embedding takes the image path with a plainly causal mask:
        // the same logits as the plain prefill in every cache format.
        var embedded = decoder[1].Forward(decoder[0].Forward(textInput));
        var asImage = new[] { new PromptImage(5, embedded.Narrow(1, 5, 1).Reshape(1, 24)) };
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16, KeyValueFormat.Int8 })
        {
            using var context = new DecodingContext(device, 1, 64, format);
            float difference = MaxDifference(whole, decoder.ForwardCached(input, images, context).ToArray());
            using var textContext = new DecodingContext(device, 1, 64, format);
            var plain = decoder.ForwardCached(textInput, textContext).ToArray();
            float textDifference = MaxDifference(textExact, plain);
            using var spansContext = new DecodingContext(device, 1, 64, format);
            var spans = decoder.ForwardCached(textInput, asImage, spansContext).ToArray();
            Console.WriteLine($"    {format} cache on {device}: {difference:G3} from float32 with images, {textDifference:G3} for the text prompt, "
                + $"image path against plain prefill {MaxDifference(plain, spans):G3}");
            AssertClose(plain, spans, 2e-5f, $"{format} cache: the image path with a causal mask against the plain prefill");
        }

        // A text-only prompt through ImagePrefill with no images is the plain prefill.
        {
            using var context = new DecodingContext(device, 1, 64);
            AssertClose(ReadNpyFloat32(TestData("vlm/reference/prompt-text-logits.npy")), decoder.ForwardCached(textInput, [], context).ToArray(), 2e-5f, "a text prompt");
        }

        // Refused: a block past the prompt, overlapping blocks, features of the wrong width.
        foreach (var (bad, what) in new (PromptImage[], string)[]
                 {
                     ([new PromptImage(44, first)], "past the prompt"),
                     ([new PromptImage(14, first), new PromptImage(16, second)], "overlapping"),
                     ([new PromptImage(14, first.Reshape(4, 24).Narrow(1, 0, 12))], "the wrong width"),
                 })
        {
            try
            {
                decoder.Forward(input, bad).Dispose();
                Check(false, $"an image {what} is refused");
            }
            catch (ArgumentException)
            {
            }
        }
    }
}
