// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Audio.Plugins.Whisper;

// Plan 21, phase S4: speech recognition. Whisper's compression ratio, the core's CrossAttention and BeamSearch.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioAsrGroup =
    [
        ("audio asr: BeamSearch.Sample draws the first token by softmax(logits / T) (6000 draws within 0.025), rows sample apart and end at the end token, best first, suppressed tokens never drawn, a temperature of 0 refused", AudioAsrSampling),
        ("audio asr: Whisper's guards (OpenAI's transcribe): the compression ratio of a loop above 2.4, of a sentence under it", AudioAsrGuards),
        ("audio asr: CrossAttention equals a loop over the heads (softmax over every memory position), the cached path (SetMemory, ForwardCached) equals Forward, one memory serves several rows, a padded batch of memories (per-sequence lengths) equals each sequence alone and its padding gets no gradient; gradients to the input, the memory and every projection by finite differences on the CPU", AudioAsrCrossAttention),
    ];

    // The fixture's log-mel features (transformers'), [mels, frames], on the device.
    internal static Tensor FixtureMel(Device device) => Tensor.From(ReadNpyFloat32(AudioData("features/log-mel.npy")), [80, 3000], device);

    internal static float Largest(float[] values) => Math.Max(1f, values.Max(MathF.Abs));

    internal static void AudioAsrGuards(Device device)
    {
        _ = device;

        // The compression ratio: a loop compresses far better than a sentence.
        double loop = WhisperRecognizer.CompressionRatio(string.Concat(Enumerable.Repeat("شكرا لكم ", 40)));
        double sentence = WhisperRecognizer.CompressionRatio("أعيش في مدينة كبيرة وأحب أن أمشي في شوارعها القديمة في المساء");
        Check(loop > 2.4 && sentence < 2.4 && WhisperRecognizer.CompressionRatio("") == 0, $"compression ratios: a loop {loop:F2}, a sentence {sentence:F2}");
    }

    internal static void AudioAsrSampling(Device device)
    {
        if (device != Device.Cpu)
        {
            return;                                                             // host-side sampling over a fixed step: once
        }

        // A step that gives the same logits on every row and position: token 2 ends; the first draws follow softmax(logits / T).
        float[] first = [1f, 0.5f, -1f, 0.2f];
        Tensor Step(Tensor ids, DecodingContext context) => Tensor.From([.. Enumerable.Range(0, ids.Shape[0]).SelectMany(_ => first)], [ids.Shape[0], first.Length], Device.Cpu);
        var options = new BeamOptions { Beams = 1, MaxTokens = 1, EndToken = 2 };
        using var context = new DecodingContext(Device.Cpu, 1, 8);
        var random = new Random(11);
        const float T = 0.8f;
        var counts = new int[first.Length];
        const int Draws = 6000;
        for (int i = 0; i < Draws; i++)
        {
            var h = BeamSearch.Sample(Step, context, [0], options, T, random);
            counts[h.Count == 0 ? 2 : h[0].Tokens.Count == 0 ? 2 : h[0].Tokens[0]]++;
        }

        double z = first.Sum(x => Math.Exp(x / T));
        var expected = first.Select(x => Math.Exp(x / T) / z).ToArray();
        double worst = counts.Select((c, i) => Math.Abs((double)c / Draws - expected[i])).Max();
        Check(worst < 0.025, $"the first draws' frequencies [{string.Join(" ", counts.Select(c => ((double)c / Draws).ToString("F3", System.Globalization.CultureInfo.InvariantCulture)))}] against softmax(logits / {T}) [{string.Join(" ", expected.Select(e => e.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)))}]");

        // Rows sample apart, end at the end token, are ranked by mean log-probability; a forbidden token is never drawn.
        using var rows = new DecodingContext(Device.Cpu, 4, 32);
        var many = BeamSearch.Sample(Step, rows, [0], new BeamOptions { Beams = 4, MaxTokens = 20, EndToken = 2, SuppressTokens = [3] }, 1f, new Random(5));
        Check(many.Count == 4 && many.All(h => !h.Tokens.Contains(3) && !h.Tokens.Contains(2)) && many.Zip(many.Skip(1)).All(p => p.First.Score >= p.Second.Score),
            $"four rows: {string.Join(" | ", many.Select(h => string.Join(",", h.Tokens) + $" {h.Score:F3}"))}");
        Throws<ArgumentException>(() => BeamSearch.Sample(Step, rows, [0], new BeamOptions { Beams = 4, MaxTokens = 2, EndToken = 2 }, 0f, new Random(1)), "a temperature of 0");
    }

    internal static void AudioAsrCrossAttention(Device device)
    {
        const int Dim = 16, Heads = 4, MemoryDim = 12, Batch = 2, Rows = 3, Memory = 5;
        var random = new Random(5);
        using var cross = new CrossAttention(Dim, Heads, MemoryDim, device: device, random: random);
        var x = RandomArray(random, Batch * Rows * Dim);
        var m = RandomArray(random, Batch * Memory * MemoryDim);
        using var input = Tensor.From(x, [Batch, Rows, Dim], device);
        using var memory = Tensor.From(m, [Batch, Memory, MemoryDim], device);
        float[] output;
        using (Autograd.NoGrad())
        {
            output = cross.Forward(input, memory).ToArray();

            // The loop: each head's softmax over every memory position, the projections by the layer's own linears.
            var q = cross.Query.Forward(input).ToArray();
            var k = cross.Key.Forward(memory).ToArray();
            var v = cross.Value.Forward(memory).ToArray();
            int d = Dim / Heads;
            var merged = new float[Batch * Rows * Dim];
            for (int b = 0; b < Batch; b++)
            {
                for (int t = 0; t < Rows; t++)
                {
                    for (int h = 0; h < Heads; h++)
                    {
                        var scores = new double[Memory];
                        for (int j = 0; j < Memory; j++)
                        {
                            double dot = 0;
                            for (int e = 0; e < d; e++)
                            {
                                dot += q[(b * Rows + t) * Dim + h * d + e] * k[(b * Memory + j) * Dim + h * d + e];
                            }

                            scores[j] = dot / Math.Sqrt(d);
                        }

                        double max = scores.Max(), sum = scores.Sum(s => Math.Exp(s - max));
                        for (int e = 0; e < d; e++)
                        {
                            double value = 0;
                            for (int j = 0; j < Memory; j++)
                            {
                                value += Math.Exp(scores[j] - max) / sum * v[(b * Memory + j) * Dim + h * d + e];
                            }

                            merged[(b * Rows + t) * Dim + h * d + e] = (float)value;
                        }
                    }
                }
            }

            using var mergedTensor = Tensor.From(merged, [Batch, Rows, Dim], device);
            AssertClose(cross.Output.Forward(mergedTensor).ToArray(), output, 1e-5f, $"CrossAttention against the loop on {device}");

            // The cached path: the memory projected once into the context.
            using var context = new DecodingContext(device, Batch, 8);
            cross.SetMemory(memory, context);
            AssertClose(output, cross.ForwardCached(input, context).ToArray(), 1e-6f, $"ForwardCached against Forward on {device}");

            // One memory for three rows (a beam search's beams): each row as if alone with that memory.
            using var one = memory.Narrow(0, 0, 1);
            using var rows = Tensor.From(RandomArray(random, 3 * 2 * Dim), [3, 2, Dim], device);
            using var shared = new DecodingContext(device, 3, 8);
            cross.SetMemory(one, shared);
            var together = cross.ForwardCached(rows, shared).ToArray();
            for (int r = 0; r < 3; r++)
            {
                using var row = rows.Narrow(0, r, 1);
                AssertClose(cross.Forward(row, one).ToArray(), together[(r * 2 * Dim)..((r + 1) * 2 * Dim)], 1e-6f, $"row {r} of a shared memory on {device}");
            }

            Throws<ArgumentException>(() => cross.SetMemory(memory, new DecodingContext(device, 3, 8)), "a memory of two sequences for three rows");
        }

        // A padded batch of memories: sequence b attends to its first lengths[b] positions, as if alone with them.
        int[] lengths = [3, 4];
        float[] padded;
        using (Autograd.NoGrad())
        {
            padded = cross.Forward(input, memory, lengths).ToArray();
            for (int b = 0; b < Batch; b++)
            {
                using var alone = Tensor.From(x[(b * Rows * Dim)..((b + 1) * Rows * Dim)], [1, Rows, Dim], device);
                using var own = Tensor.From(m[(b * Memory * MemoryDim)..((b * Memory + lengths[b]) * MemoryDim)], [1, lengths[b], MemoryDim], device);
                AssertClose(cross.Forward(alone, own).ToArray(), padded[(b * Rows * Dim)..((b + 1) * Rows * Dim)], 1e-5f,
                    $"sequence {b} of a padded batch (memory length {lengths[b]} of {Memory}) against it alone on {device}");
            }

            using (EncoderMemory.Use(memory, lengths))
            {
                AssertClose(padded, cross.Forward(input).ToArray(), 0f, $"an EncoderMemory scope with lengths against Forward(input, memory, lengths) on {device}");
            }

            AssertClose(output, cross.Forward(input, memory, [Memory, Memory]).ToArray(), 0f, $"full lengths against no lengths on {device}");
            Throws<ArgumentException>(() => cross.Forward(input, memory, [0, Memory]), "a memory length of 0");
            Throws<ArgumentException>(() => cross.Forward(input, memory, [Memory + 1, Memory]), "a memory length past the memory");
            Throws<ArgumentException>(() => cross.Forward(input, memory, [Memory]), "one length for two sequences");
        }

        // The padding gets no gradient.
        using (var tracked = Tensor.From(m, [Batch, Memory, MemoryDim], device, requiresGrad: true))
        {
            using var padWeighting = Tensor.From(RandomArray(new Random(9), Batch * Rows * Dim), [Batch, Rows, Dim], device);
            (cross.Forward(input, tracked, lengths) * padWeighting).Sum().Backward();
            var grad = tracked.Grad!.ToArray();
            for (int b = 0; b < Batch; b++)
            {
                var pad = grad.AsSpan((b * Memory + lengths[b]) * MemoryDim, (Memory - lengths[b]) * MemoryDim);
                Check(!pad.ContainsAnyExcept(0f), $"sequence {b}'s padding gets no gradient on {device} (largest {pad.ToArray().Max(Math.Abs)})");
                Check(grad.AsSpan(b * Memory * MemoryDim, lengths[b] * MemoryDim).ContainsAnyExcept(0f), $"sequence {b}'s own positions get a gradient on {device}");
            }

            foreach (var parameter in cross.Parameters())
            {
                parameter.ZeroGrad();
            }
        }

        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // gradients by finite differences: once, on the host
        }

        var weights = RandomArray(new Random(8), Batch * Rows * Dim);
        using var weighting = Tensor.From(weights, [Batch, Rows, Dim], device);
        GradCheck(device, [Batch, Rows, Dim], t => (cross.Forward(t, memory) * weighting).Sum());
        GradCheck(device, [Batch, Memory, MemoryDim], t => (cross.Forward(input, t) * weighting).Sum());
        using (EncoderMemory.Use(memory))
        {
            ParameterGradCheck(cross, input, y => (y * weighting).Sum());
        }
    }
}
