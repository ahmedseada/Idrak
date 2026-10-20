// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.WhisperSpeech;

// Plan 21, phase S4: speech recognition. The Whisper plug-in on the fixture tests/Idrak.Tests/data/audio/tiny-whisper (a
// seeded tiny model whose references came from transformers: tools/audio/make_tiny_audio.py), the core's CrossAttention
// and BeamSearch, and the testing kit's speech recognizer suite.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioAsrGroup =
    [
        ("audio asr: the Whisper plug-in loads tiny-whisper: the encoder output and the first step's logits within 1e-4 of transformers', the language detected from one step (ar, with its logits), the prompts with and without timestamps; an unknown language, and the architecture without the plug-in, refused naming the registry", AudioAsrFixture),
        ("audio asr: greedy and beam-4 decoding give transformers' ids exactly (12 tokens, the end token suppressed first), and with timestamps (the timestamp rules) greedy and beam-4 do too; the int8 and bfloat16 caches run the same beam search", AudioAsrDecoding),
        ("audio asr: Transcribe on pcm16.wav gives the greedy reference's tokens and text, the language detected; chunks of 0.75 s equal two calls on the halves joined, TranscribeStream equals the chunked call, stereo and 44.1 kHz sound mixed and resampled; the timestamps' segment; the stats and progress; device bytes the same after every call from the second; cold and warm times", AudioAsrTranscribe),
        ("audio asr: BeamSearch.Sample draws the first token by softmax(logits / T) (6000 draws within 0.025), rows sample apart and end at the end token, best first, suppressed tokens never drawn, a temperature of 0 refused", AudioAsrSampling),
        ("audio asr: Whisper's guards (OpenAI's transcribe): the no-speech check leaves out a segment taken for silence (threshold 1 keeps all, 0 with an unsure decoding drops all) from the step after <|startoftranscript|>; the temperature fallback decodes an unsure segment again at the next temperature; the compression ratio of a loop above 2.4", AudioAsrGuards),
        ("audio asr: three segments through the encoder at once give each segment's output alone within 1e-5", AudioAsrBatch),
        ("audio asr: CrossAttention equals a loop over the heads (softmax over every memory position), the cached path (SetMemory, ForwardCached) equals Forward, one memory serves several rows, a padded batch of memories (per-sequence lengths) equals each sequence alone and its padding gets no gradient; gradients to the input, the memory and every projection by finite differences on the CPU", AudioAsrCrossAttention),
        ("audio asr: BeamSearch on the tiny Llama gives a reference beam search's hypotheses and scores (the whole sequences re-run, no cache): greedy, an end token, length penalties 0, 1 and 2, the least length, no repeated bigrams, without early stopping; the kit's speech recognizer suite passes on the Whisper plug-in", AudioAsrBeamSearch),
    ];

    private static WhisperRecognizer TinyWhisper(Device device, KeyValueFormat cache = KeyValueFormat.Float32)
    {
        WhisperSpeechPlugin.Register();
        return (WhisperRecognizer)SpeechModels.Load(AudioData("tiny-whisper"), new SpeechModelOptions { Device = device, Cache = cache });
    }

    private static (int[] Prompt, int[] Ids) WhisperReference(string name)
    {
        var json = JsonNode.Parse(File.ReadAllText(AudioData($"tiny-whisper/reference/{name}.json")))!.AsObject();
        return ([.. json["prompt"]!.AsArray().Select(n => (int)n!)], [.. json["ids"]!.AsArray().Select(n => (int)n!)]);
    }

    // The fixture's log-mel features (transformers'), [mels, frames], on the device.
    private static Tensor FixtureMel(Device device) => Tensor.From(ReadNpyFloat32(AudioData("features/log-mel.npy")), [80, 3000], device);

    private static float Largest(float[] values) => Math.Max(1f, values.Max(MathF.Abs));

    private static void AudioAsrFixture(Device device)
    {
        using var recognizer = TinyWhisper(device);
        Check(recognizer.Device == device && recognizer.Languages.SetEquals(["en", "ar"]), $"loaded on {recognizer.Device}, languages {string.Join(", ", recognizer.Languages)}");
        using var mel = FixtureMel(device);
        using var encoded = recognizer.Encode(mel);
        var expected = ReadNpyFloat32(AudioData("tiny-whisper/reference/encoder-output.npy"));
        Check(encoded.Shape.SequenceEqual([1, 1500, 32]), $"the encoder gives {Tensor.FormatShape(encoded.Shape)}");
        AssertClose(expected, encoded.ToArray(), 1e-4f * Largest(expected), $"the encoder output on {device} against transformers'");

        var (prompt, _) = WhisperReference("greedy-ids");
        var (stampedPrompt, _) = WhisperReference("timestamps-ids");
        Check(recognizer.Prompt("ar").SequenceEqual(prompt) && recognizer.Prompt("ar", timestamps: true).SequenceEqual(stampedPrompt),
            $"the prompts [{string.Join(" ", recognizer.Prompt("ar"))}] and [{string.Join(" ", recognizer.Prompt("ar", timestamps: true))}]");
        var logits = ReadNpyFloat32(AudioData("tiny-whisper/reference/logits-step1.npy"));
        AssertClose(logits, recognizer.Logits(encoded, prompt), 1e-4f * Largest(logits), $"the first step's logits on {device} against transformers'");

        var language = JsonNode.Parse(File.ReadAllText(AudioData("tiny-whisper/reference/language.json")))!.AsObject();
        var start = recognizer.Logits(encoded, [prompt[0]]);
        var special = JsonNode.Parse(File.ReadAllText(AudioData("tiny-whisper/special_tokens.json")))!.AsObject();
        foreach (string code in new[] { "en", "ar" })
        {
            float want = (float)language["logits"]![code]!, got = start[(int)special[$"<|{code}|>"]!];
            Check(MathF.Abs(want - got) <= 1e-4f * Math.Max(1f, MathF.Abs(want)), $"the <|{code}|> logit after <|startoftranscript|>: {got}, transformers {want}");
        }

        Check(recognizer.DetectLanguage(encoded) == (string)language["detected"]!, $"detected {recognizer.DetectLanguage(encoded)}, transformers' {language["detected"]}");
        ThrowsNaming<NotSupportedException>(() => recognizer.Prompt("fr"), "an unknown language", "'fr'", "ar, en");

        WhisperSpeechPlugin.Unregister();
        ThrowsNaming<NotSupportedException>(() => SpeechModels.Load(AudioData("tiny-whisper"), new SpeechModelOptions { Device = device }),
            "the family without the plug-in", "WhisperForConditionalGeneration", "not registered", "SpeechFamilies.Register", "Idrak.Audio.Plugins", "AudioPlugins.RegisterAll()");
    }

    private static void AudioAsrDecoding(Device device)
    {
        using var recognizer = TinyWhisper(device);
        using var mel = FixtureMel(device);
        using var encoded = recognizer.Encode(mel);
        foreach (var (name, beams) in new[] { ("greedy-ids", 1), ("beam4-ids", 4), ("timestamps-ids", 1), ("timestamps-beam4-ids", 4) })
        {
            var (prompt, ids) = WhisperReference(name);
            var hypotheses = recognizer.Decode(encoded, prompt, new TranscriptionOptions { Beams = beams, MaxTokens = ids.Length });
            Check(hypotheses.Count > 0 && hypotheses[0].Tokens.SequenceEqual(ids), $"{name} on {device}: [{string.Join(" ", hypotheses.FirstOrDefault()?.Tokens ?? [])}], transformers [{string.Join(" ", ids)}]");
            Check(hypotheses.Count <= beams && hypotheses.All(h => float.IsFinite(h.Score) && float.IsFinite(h.LogProb))
                  && hypotheses.Zip(hypotheses.Skip(1)).All(p => p.First.Score >= p.Second.Score), $"{name}: {hypotheses.Count} hypotheses, best first");
        }

        var (beamPrompt, beamIds) = WhisperReference("beam4-ids");
        foreach (var format in new[] { KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            using var other = TinyWhisper(device, format);
            var hypotheses = other.Decode(encoded, beamPrompt, new TranscriptionOptions { Beams = 4, MaxTokens = beamIds.Length });
            var again = other.Decode(encoded, beamPrompt, new TranscriptionOptions { Beams = 4, MaxTokens = beamIds.Length });
            Check(hypotheses.Count > 0 && hypotheses[0].Tokens.Count == beamIds.Length && hypotheses.All(h => float.IsFinite(h.Score))
                  && hypotheses.Zip(again).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens)),
                $"the {format} cache's beam search on {device}: [{string.Join(" ", hypotheses.FirstOrDefault()?.Tokens ?? [])}] (float32 [{string.Join(" ", beamIds)}]), the same twice");
        }

        // Sampling: BestOf samples, best first, the same seed the same samples, and a near-zero temperature the greedy ids.
        var sampled = recognizer.Decode(encoded, beamPrompt, new TranscriptionOptions { Temperature = 0.7f, BestOf = 3, MaxTokens = beamIds.Length, Seed = 4 });
        var resampled = recognizer.Decode(encoded, beamPrompt, new TranscriptionOptions { Temperature = 0.7f, BestOf = 3, MaxTokens = beamIds.Length, Seed = 4 });
        var (greedyPrompt, greedyIds) = WhisperReference("greedy-ids");
        var cold = recognizer.Decode(encoded, greedyPrompt, new TranscriptionOptions { Temperature = 1e-4f, BestOf = 2, MaxTokens = greedyIds.Length });
        Check(sampled.Count == 3 && sampled.Zip(resampled).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens)) && sampled.Zip(sampled.Skip(1)).All(p => p.First.Score >= p.Second.Score)
              && cold.All(h => h.Tokens.SequenceEqual(greedyIds)),
            $"sampling on {device}: 3 samples best first, the same with the same seed, at temperature 1e-4 the greedy ids [{string.Join(" ", cold.FirstOrDefault()?.Tokens ?? [])}] for [{string.Join(" ", greedyIds)}]");
    }

    private static void AudioAsrGuards(Device device)
    {
        using var recognizer = TinyWhisper(device);
        var audio = AudioCodecs.Decode(AudioData("wav/pcm16.wav"));
        var plain = recognizer.Transcribe(audio);

        // The no-speech check: a threshold of 1 leaves everything, one of 0 with every decoding "unsure" leaves out every segment.
        var kept = recognizer.Transcribe(audio, new TranscriptionOptions { NoSpeechThreshold = 1f });
        var dropped = recognizer.Transcribe(audio, new TranscriptionOptions { NoSpeechThreshold = 0f, LogProbThreshold = float.MaxValue });
        Check(kept.Text == plain.Text && dropped.Segments.Count == 0 && dropped.Text.Length == 0,
            $"the no-speech check on {device}: threshold 1 '{kept.Text}' (plain '{plain.Text}'), threshold 0 and unsure {dropped.Segments.Count} segments");

        // The no-speech probability: the softmax of the step after <|startoftranscript|> at the no-speech token.
        using (var noGrad = Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var features = recognizer.Features.Extract(audio, device).Transpose();
            using var encoded = recognizer.Encode(features);
            var prompt = recognizer.Prompt("ar");
            var logits = recognizer.Logits(encoded, [recognizer.Config.StartToken]);
            double max = logits.Max(), sum = logits.Sum(x => Math.Exp(x - max));
            double expected = Math.Exp(logits[recognizer.Config.NoSpeechToken] - max) / sum;
            float p = recognizer.NoSpeechProbability(encoded, prompt);
            Check(Math.Abs(p - expected) < 1e-6 && p is >= 0 and <= 1 && recognizer.Config.NoSpeechToken == recognizer.Config.NoTimestampsToken - 1,
                $"the no-speech probability on {device}: {p}, the step's softmax {expected}");
        }

        // The fallback: a decoding always "unsure" goes on to the next temperature; one temperature on, the sampled text.
        var fallback = recognizer.Transcribe(audio, new TranscriptionOptions { TemperatureFallback = [0.7f], LogProbThreshold = float.MaxValue, Seed = 2 });
        var hot = recognizer.Transcribe(audio, new TranscriptionOptions { Temperature = 0.7f, Seed = 2 });
        var sure = recognizer.Transcribe(audio, new TranscriptionOptions { TemperatureFallback = [0.7f], LogProbThreshold = float.MinValue, CompressionRatioThreshold = float.MaxValue });
        Check(fallback.Text == hot.Text && sure.Text == plain.Text, $"the fallback on {device}: '{fallback.Text}' (sampled at 0.7 '{hot.Text}'); not needed, the greedy text '{sure.Text}'");

        // The compression ratio: a loop compresses far better than a sentence.
        double loop = WhisperRecognizer.CompressionRatio(string.Concat(Enumerable.Repeat("شكرا لكم ", 40)));
        double sentence = WhisperRecognizer.CompressionRatio("أعيش في مدينة كبيرة وأحب أن أمشي في شوارعها القديمة في المساء");
        Check(loop > 2.4 && sentence < 2.4 && WhisperRecognizer.CompressionRatio("") == 0, $"compression ratios: a loop {loop:F2}, a sentence {sentence:F2}");
    }

    private static void AudioAsrSampling(Device device)
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

    private static void AudioAsrTranscribe(Device device)
    {
        using var recognizer = TinyWhisper(device);
        var audio = AudioCodecs.Decode(AudioData("wav/pcm16.wav"));
        var (_, greedy) = WhisperReference("greedy-ids");
        var options = new TranscriptionOptions { MaxTokens = greedy.Length };
        var progress = new List<TranscriptionProgress>();
        var watch = Stopwatch.StartNew();
        var result = recognizer.Transcribe(audio, options, new SyncProgress<TranscriptionProgress>(progress.Add));
        double cold = watch.Elapsed.TotalMilliseconds;
        // The text is the tokens before the specials (a timestamp token the untrained model gives without timestamps is not text).
        string text = recognizer.Tokenizer.Decode([.. greedy.Where(t => t < recognizer.Config.EndToken && !recognizer.Tokenizer.IsSpecial(t))]).Trim();
        Check(result.Language == "ar" && result.Segments is [{ } only] && only.Tokens.SequenceEqual(greedy) && only.Start == 0 && Math.Abs(only.End - 1.5) < 1e-9 && result.Text == text,
            $"Transcribe: {result.Language}, {string.Join(" | ", result.Segments.Select(s => $"{s.Start:F2}-{s.End:F2} [{string.Join(" ", s.Tokens)}]"))}, \"{result.Text}\" (the greedy reference reads \"{text}\")");
        Check(result.Stats.Segments == 1 && result.Stats.Batch == 1 && Math.Abs(result.Stats.AudioDuration.TotalSeconds - 1.5) < 1e-9 && result.Stats.Encoder > TimeSpan.Zero && result.Stats.Decoder > TimeSpan.Zero,
            $"the stats: {result.Stats}");
        Check(progress.Count >= 4 && progress[^1].Stage == "done" && progress.Select(p => p.Stage).Take(3).SequenceEqual(["features", "encoder", "decoder"]),
            $"progress: {string.Join(", ", progress.Select(p => p.Stage))}");

        // With timestamps: the timestamp reference's tokens, one segment from its opening timestamp to the chunk's end.
        var (_, stamped) = WhisperReference("timestamps-ids");
        var timed = recognizer.Transcribe(audio, new TranscriptionOptions { Language = "ar", Timestamps = true, MaxTokens = stamped.Length });
        int begin = recognizer.Config.TimestampBegin;
        Check(timed.Segments.Count >= 1 && timed.Segments.SelectMany(s => s.Tokens).SequenceEqual(stamped) && Math.Abs(timed.Segments[0].Start - (stamped[0] - begin) * recognizer.Config.TimePrecision) < 1e-9,
            $"timestamps: {string.Join(" | ", timed.Segments.Select(s => $"{s.Start:F2}-{s.End:F2} [{string.Join(" ", s.Tokens)}]"))}, transformers [{string.Join(" ", stamped)}]");

        // Chunks of 0.75 s without overlap or carried text: the two halves transcribed alone, joined.
        var chunked = options with { Language = "ar", ChunkSeconds = 0.75, OverlapSeconds = 0, CarryText = false };
        var halves = recognizer.Transcribe(audio, chunked);
        var first = recognizer.Transcribe(audio.Slice(0, 12000), options with { Language = "ar" });
        var second = recognizer.Transcribe(audio.Slice(12000, 12000), options with { Language = "ar" });
        Check(halves.Segments.Count == 2 && halves.Text == $"{first.Text} {second.Text}".Trim() && halves.Segments[0].Start == 0 && Math.Abs(halves.Segments[1].Start - 0.75) < 1e-9,
            $"chunks: \"{halves.Text}\" ({halves.Segments.Count} segments), the halves \"{first.Text}\" + \"{second.Text}\"");

        // TranscribeStream: the sound in thirds, gathered into the same 0.75 s chunks.
        async IAsyncEnumerable<AudioData> Thirds()
        {
            for (int i = 0; i < 3; i++)
            {
                await Task.Yield();
                yield return audio.Slice(i * 8000, 8000);
            }
        }

        var streamed = Task.Run(async () =>
        {
            var segments = new List<TranscriptSegment>();
            await foreach (var segment in recognizer.TranscribeStream(Thirds(), chunked))
            {
                segments.Add(segment);
            }

            return segments;
        }).GetAwaiter().GetResult();
        Check(streamed.Count == halves.Segments.Count && streamed.Zip(halves.Segments).All(p => p.First.Text == p.Second.Text && Math.Abs(p.First.Start - p.Second.Start) < 1e-9),
            $"TranscribeStream: {string.Join(" | ", streamed.Select(s => $"{s.Start:F2} {s.Text}"))}; chunked {string.Join(" | ", halves.Segments.Select(s => $"{s.Start:F2} {s.Text}"))}");

        // Stereo and 44.1 kHz: mixed to one channel and resampled.
        var stereo = AudioCodecs.Decode(AudioData("wav/stereo16.wav"));
        var mono = new float[stereo.Length];
        stereo.ToMono(mono);
        Check(recognizer.Transcribe(stereo, options with { Language = "ar" }).Text == recognizer.Transcribe(new AudioData(mono, 1, 16000), options with { Language = "ar" }).Text, "stereo is its channels' mean");
        var fast = AudioCodecs.Decode(AudioData("wav/rate44k.wav"));
        Check(recognizer.Transcribe(fast, options with { Language = "ar" }).Text == recognizer.Transcribe(Resampler.Resample(fast, 16000), options with { Language = "ar" }).Text, "44.1 kHz is resampled to the model's rate");

        // Device memory: the same bytes after every call from the second (the caches made once, nothing kept per call).
        var held = new List<long>();
        var times = new List<double>();
        for (int i = 0; i < 3; i++)
        {
            watch.Restart();
            recognizer.Transcribe(audio, options);
            times.Add(watch.Elapsed.TotalMilliseconds);
            held.Add(device.Backend.GetMemoryUsage().InUse);
        }

        Check(held[1] == held[2], $"device bytes after each call: {string.Join(", ", held)}");
        Console.WriteLine($"    Transcribe 1.5 s on {device}: cold {cold:F1} ms, warm {times.Skip(1).Average():F1} ms (features {result.Stats.Features.TotalMilliseconds:F1}, encoder {result.Stats.Encoder.TotalMilliseconds:F1}, decoder {result.Stats.Decoder.TotalMilliseconds:F1} ms cold)");
    }

    private static void AudioAsrBatch(Device device)
    {
        using var recognizer = TinyWhisper(device);
        var sounds = new[]
        {
            AudioCodecs.Decode(AudioData("wav/pcm16.wav")),
            Resampler.Resample(AudioCodecs.Decode(AudioData("wav/rate8k.wav")), 16000),
            AudioCodecs.Decode(AudioData("wav/float32.wav")).Slice(4000, 12000),
        };
        var singles = new List<float[]>();
        var mels = new List<Tensor>();
        try
        {
            foreach (var sound in sounds)
            {
                using var features = recognizer.Features.Extract(sound, device);    // [frames, mels]
                var mel = features.Transpose();                       // [mels, frames]
                mels.Add(mel);
                using var alone = recognizer.Encode(mel);
                singles.Add(alone.ToArray());
            }

            using var stacked = Tensor.Stack(mels);
            using var batch = recognizer.Encode(stacked);
            var values = batch.ToArray();
            Check(batch.Shape.SequenceEqual([3, 1500, 32]), $"the batch gives {Tensor.FormatShape(batch.Shape)}");
            for (int i = 0; i < 3; i++)
            {
                AssertClose(singles[i], values[(i * singles[i].Length)..((i + 1) * singles[i].Length)], 1e-5f * Largest(singles[i]), $"segment {i} in the batch on {device}");
            }
        }
        finally
        {
            foreach (var mel in mels)
            {
                mel.Dispose();
            }
        }
    }

    private static void AudioAsrCrossAttention(Device device)
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

    private static void AudioAsrBeamSearch(Device device)
    {
        using (var model = PretrainedModel.Load(PretrainData("tiny-llama"), new PretrainedOptions { Device = device }))
        {
            var network = model.Network;
            network.Eval();
            int[] prompt = [.. model.Tokenizer!.Encode("the quick brown")];
            var greedy = SearchBeams(network, device, prompt, new BeamOptions { Beams = 1, MaxTokens = 8 });
            var greedyReference = ReferenceBeams(network, device, prompt, new BeamOptions { Beams = 1, MaxTokens = 8 });
            Check(greedy.Count == 1 && greedy[0].Tokens.SequenceEqual(greedyReference[0].Tokens), $"a beam of 1 is greedy: [{string.Join(" ", greedy[0].Tokens)}] against [{string.Join(" ", greedyReference[0].Tokens)}]");
            int end = greedy[0].Tokens[2];                                       // an end the search meets early
            var settings = new (string Name, BeamOptions Options)[]
            {
                ("3 beams", new BeamOptions { Beams = 3, MaxTokens = 8 }),
                ("3 beams, an end token", new BeamOptions { Beams = 3, MaxTokens = 8, EndToken = end }),
                ("4 beams, penalty 0", new BeamOptions { Beams = 4, MaxTokens = 7, EndToken = end, LengthPenalty = 0 }),
                ("2 beams, penalty 2, at least 4", new BeamOptions { Beams = 2, MaxTokens = 9, EndToken = end, LengthPenalty = 2, MinTokens = 4 }),
                ("3 beams, no repeated bigram", new BeamOptions { Beams = 3, MaxTokens = 8, NoRepeatNgram = 2 }),
                ("3 beams, without early stopping", new BeamOptions { Beams = 3, MaxTokens = 8, EndToken = end, EarlyStopping = false }),
                ("2 beams, suppressed tokens", new BeamOptions { Beams = 2, MaxTokens = 6, SuppressTokens = [greedy[0].Tokens[0]], BeginSuppressTokens = [greedy[0].Tokens[1]] }),
            };
            foreach (var (name, options) in settings)
            {
                var found = SearchBeams(network, device, prompt, options);
                var reference = ReferenceBeams(network, device, prompt, options);
                Check(found.Count == reference.Count && found.Zip(reference).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens) && Math.Abs(p.First.Score - p.Second.Score) <= 1e-4 * Math.Max(1, Math.Abs(p.Second.Score))),
                    $"{name} on {device}: {string.Join(" | ", found.Select(h => $"[{string.Join(" ", h.Tokens)}] {h.Score:F4}"))}; the reference {string.Join(" | ", reference.Select(h => $"[{string.Join(" ", h.Tokens)}] {h.Score:F4}"))}");
            }
        }

        // The kit's suite on the Whisper plug-in (an untrained model: no bound on what silence gives).
        using var recognizer = TinyWhisper(device);
        var report = Conformance.CheckSpeechRecognizer(new SpeechRecognizerUnderTest(16000, samples =>
        {
            var t = recognizer.Transcribe(new AudioData(samples, 1, 16000), new TranscriptionOptions { Language = "ar", MaxTokens = 6 });
            return new RecognizedSpeech(t.Text, [.. t.Segments.Select(s => new RecognizedSegment(s.Start, s.End, s.Text))]);
        }));
        Check(report.Passed, $"the speech recognizer suite on {device}: {report}");
    }

    private static List<BeamHypothesis> SearchBeams(Sequential network, Device device, int[] prompt, BeamOptions options)
    {
        using var context = new DecodingContext(device, options.Beams, prompt.Length + options.MaxTokens + 1) { LastPositionOnly = true };
        return [.. BeamSearch.Decode((ids, ctx) => network.ForwardCached(ids, ctx), context, prompt, options)];
    }

    // The beam search written again for the test: every beam's whole sequence run through the network each step (no cache,
    // no reordered rows), log-softmax then the rules, the 2·beams best candidates, an end only from the first beams, the
    // finished kept by their length-normalized score.
    private static List<BeamHypothesis> ReferenceBeams(Sequential network, Device device, int[] prompt, BeamOptions o)
    {
        var live = new List<(List<int> Tokens, double Score)> { ([], 0) };
        var finished = new List<(List<int> Tokens, double Score, double LogProb)>();
        void Finish(List<int> tokens, double logProb, int length)
        {
            double score = logProb / Math.Pow(Math.Max(length, 1), o.LengthPenalty);
            finished.Add((tokens, score, logProb));
            if (finished.Count > o.Beams)
            {
                finished.RemoveAt(finished.IndexOf(finished.MinBy(f => f.Score)));
            }
        }

        for (int s = 0; s < o.MaxTokens; s++)
        {
            var candidates = new List<(double Score, int Beam, int Token)>();
            for (int b = 0; b < live.Count; b++)
            {
                int[] sequence = [.. prompt, .. live[b].Tokens];
                float[] logits;
                using (Autograd.NoGrad())
                using (var scope = new TensorScope())
                {
                    var all = network.Predict(Tensor.From([.. sequence.Select(i => (float)i)], [1, sequence.Length], device)).ToArray();
                    logits = all[^(all.Length / sequence.Length)..];
                }

                double max = logits.Max(), sum = logits.Sum(l => Math.Exp(l - max));
                var lp = logits.Select(l => l - max - Math.Log(sum)).ToArray();
                foreach (int t in o.SuppressTokens)
                {
                    lp[t] = double.NegativeInfinity;
                }

                if (s == 0)
                {
                    foreach (int t in o.BeginSuppressTokens)
                    {
                        lp[t] = double.NegativeInfinity;
                    }
                }

                if (s < o.MinTokens && o.EndToken >= 0)
                {
                    lp[o.EndToken] = double.NegativeInfinity;
                }

                if (o.NoRepeatNgram == 2 && sequence.Length >= 1)
                {
                    for (int i = 0; i + 1 < sequence.Length; i++)
                    {
                        if (sequence[i] == sequence[^1])
                        {
                            lp[sequence[i + 1]] = double.NegativeInfinity;
                        }
                    }
                }

                for (int t = 0; t < lp.Length; t++)
                {
                    if (!double.IsNegativeInfinity(lp[t]))
                    {
                        candidates.Add((live[b].Score + lp[t], b, t));
                    }
                }
            }

            var top = candidates.OrderByDescending(c => c.Score).ThenBy(c => c.Beam).ThenBy(c => c.Token).Take(2 * o.Beams).ToList();
            var next = new List<(List<int> Tokens, double Score, int Rank)>();
            for (int rank = 0; rank < top.Count && next.Count < o.Beams; rank++)
            {
                var (score, beam, token) = top[rank];
                if (token == o.EndToken)
                {
                    if (rank < o.Beams)
                    {
                        Finish([.. live[beam].Tokens], score, s + 1);
                    }

                    continue;
                }

                next.Add(([.. live[beam].Tokens, token], score, rank));
            }

            if (next.Count == 0)
            {
                break;
            }

            double best = top[0].Score;
            bool done = finished.Count >= o.Beams
                && (o.EarlyStopping || finished.Min(f => f.Score) >= (o.LengthPenalty > 0 ? best / Math.Pow(s + 1, o.LengthPenalty) : best));
            if (done || s + 1 == o.MaxTokens)
            {
                if (!done)
                {
                    foreach (var (tokens, score, rank) in next.Where(n => n.Rank < o.Beams))
                    {
                        Finish(tokens, score, s + 1);
                    }
                }

                break;
            }

            live = [.. next.Select(n => (n.Tokens, n.Score))];
        }

        return [.. finished.OrderByDescending(f => f.Score).Select(f => new BeamHypothesis(f.Tokens, (float)f.Score, (float)f.LogProb))];
    }
}
