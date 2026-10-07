// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp;
using Idrak;

// GGUF: dequantization checked against the gguf package's reference, and models read from GGUF files against the same
// models in the Hugging Face layout (tools/gguf/make_fixtures.py makes both).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] GgufGroup =
    [
        ("gguf: metadata, and F16, BF16, Q4_0–Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS dequantized as llama.cpp's reference does", GgufQuants),
        ("evaluation: final numbers (#### n), exact, contains and F1 scores of answers", EvaluationMetrics),
        ("gguf: Llama (q/k row order) and Qwen3 (Q8_0) models read from GGUF match their Hugging Face copies: config, tokens, chat, loss", GgufModels),
    ];

    private static void EvaluationMetrics(Device device)
    {
        _ = device;
        Check(ChatEvaluation.FinalNumber("so 4 * 3 = 12\n#### 1,200") == 1200m && ChatEvaluation.FinalNumber("The answer is $8.") == 8m
              && ChatEvaluation.FinalNumber("-3.5 then 7") == 7m && ChatEvaluation.FinalNumber("none") is null, "final numbers");
        Check(ChatEvaluation.Score(AnswerMetric.Number, "Tom gets 20 - 12 = $8 back.", "20-12=8\n#### 8") == 1
              && ChatEvaluation.Score(AnswerMetric.Number, "#### 9", "#### 8") == 0, "number");
        Check(ChatEvaluation.Score(AnswerMetric.Exact, "  Paris. ", "paris") == 1 && ChatEvaluation.Score(AnswerMetric.Contains, "It is Paris, France", "paris") == 1, "exact, contains");
        Check(Math.Abs(ChatEvaluation.Score(AnswerMetric.F1, "the cat sat", "the cat ran") - 2.0 / 3) < 1e-9 && ChatEvaluation.Score(AnswerMetric.F1, "dog", "cat") == 0, "f1");
        Check(ChatEvaluation.Score(AnswerMetric.Auto, "so it is 14", "#### 14") == 1, "auto picks the number for #### references");
    }

    private static void GgufQuants(Device device)
    {
        _ = device;
        using var file = GgufFile.Open(TestData("gguf/quants.gguf"));
        Check(file.Version == 3 && file.Get("test.string", "") == "héllo" && file.Get<long[]>("test.numbers", []).SequenceEqual([1L, 2L, 3L]), "metadata");
        foreach (var type in new[] { "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS" })
        {
            var ours = file.Read($"q.{type}");
            var expected = file.Read($"expected.{type}");
            Check(ours.Length == expected.Length, $"{type}: {ours.Length} values, expected {expected.Length}");
            float worst = 0;
            for (int i = 0; i < ours.Length; i++)
            {
                worst = MathF.Max(worst, MathF.Abs(ours[i] - expected[i]) / MathF.Max(1f, MathF.Abs(expected[i])));
            }

            Check(worst < 1e-5f, $"{type}: largest difference {worst:G3}");
        }
    }

    private static void GgufModels(Device device)
    {
        string cache = Path.Combine(Path.GetTempPath(), "ns-gguf-" + Guid.NewGuid().ToString("N"));
        try
        {
            foreach (var (name, arch) in new[] { ("tiny-llama", "LlamaForCausalLM"), ("tiny-qwen3-q8", "Qwen3ForCausalLM") })
            {
                string folder = GgufModel.Prepare(TestData($"gguf/{name}.gguf"), cache);
                Check(GgufModel.Prepare(TestData($"gguf/{name}.gguf"), cache) == folder && GgufModel.IsPrepared(folder), $"{name}: prepared once");
                using var fromGguf = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                using var fromHf = PretrainedModel.Load(TestData($"gguf/{name}-hf"), new PretrainedOptions { Device = device });
                Check((string?)fromGguf.Config["architectures"]?[0] == arch && fromGguf.Spec.ToJson().ToJsonString() == fromHf.Spec.ToJson().ToJsonString(),
                    $"{name}: same model spec\n  gguf {fromGguf.Spec.ToJson().ToJsonString()}\n  hf   {fromHf.Spec.ToJson().ToJsonString()}");
                Check(fromGguf.Notes.Any(n => n.Contains(".gguf", StringComparison.Ordinal)), $"{name}: notes say where the weights come from");

                foreach (var text in new[] { "the answer is in the thin rain", "Hello, World!  123456 tabs\tand\nnew lines", "<|im_start|>user\nhi<|im_end|>" })
                {
                    var a = fromGguf.Tokenizer!.Encode(text);
                    var b = fromHf.Tokenizer!.Encode(text);
                    Check(a.SequenceEqual(b), $"{name}: tokens of '{text}': [{string.Join(",", a)}] vs [{string.Join(",", b)}]");
                    Check(fromGguf.Tokenizer.Decode(a) == text, $"{name}: decode round trip");
                }

                var chat = new[] { new ChatMessage("user", "hi there") };
                Check(fromGguf.ChatTemplate!.Render(chat, [], null) == fromHf.ChatTemplate!.Render(chat, [], null)
                      && fromGguf.ChatTemplate.StopSequences.Contains("<|im_end|>"), $"{name}: chat template and stop");

                int[] ids = [.. fromHf.Tokenizer!.Encode("the answer is in the thin rain, and then another answer")];
                TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
                float lossGguf = FineTuner.Evaluate(fromGguf, probe), lossHf = FineTuner.Evaluate(fromHf, probe);
                Check(MathF.Abs(lossGguf - lossHf) < 1e-4f, $"{name}: loss {lossGguf} from GGUF, {lossHf} from the Hugging Face copy");
            }

            // The local model store (its folder and manifests as that store writes them): the manifest of name:tag names the
            // blob holding the GGUF file.
            string store = Path.Combine(cache, "store");
            string? oldStore = Environment.GetEnvironmentVariable("OLLAMA_MODELS"), oldCache = Environment.GetEnvironmentVariable("IDRAK_CACHE");
            Directory.CreateDirectory(Path.Combine(store, "manifests", "registry.ollama.ai", "library", "tiny"));
            Directory.CreateDirectory(Path.Combine(store, "blobs"));
            File.Copy(TestData("gguf/tiny-llama.gguf"), Path.Combine(store, "blobs", "sha256-abc123"));
            File.WriteAllText(Path.Combine(store, "manifests", "registry.ollama.ai", "library", "tiny", "q8"),
                "{\"layers\":[{\"mediaType\":\"application/vnd.ollama.image.template\",\"digest\":\"sha256:t\"},{\"mediaType\":\"application/vnd.ollama.image.model\",\"digest\":\"sha256:abc123\"}]}");
            try
            {
                Environment.SetEnvironmentVariable("OLLAMA_MODELS", store);
                Environment.SetEnvironmentVariable("IDRAK_CACHE", cache);
                Check(ModelSource.LocalStoreModel("tiny:q8") == Path.Combine(store, "blobs", "sha256-abc123"), "store name → blob");
                string prepared = ModelSource.Resolve("store:tiny:q8");
                using var stored = PretrainedModel.Load(prepared, new PretrainedOptions { Device = device });
                Check(stored.Spec.Layers == 2 && prepared.StartsWith(Path.Combine(cache, "gguf"), StringComparison.Ordinal), "a model of the local store loads");
                try
                {
                    ModelSource.LocalStoreModel("missing");
                    Check(false, "a model missing from the store should fail");
                }
                catch (FileNotFoundException ex)
                {
                    Check(ex.Message.Contains("tiny:q8", StringComparison.Ordinal), $"lists the models there: {ex.Message}");
                }

                // A .gguf path loads directly (prepared under IDRAK_CACHE).
                using var direct = PretrainedModel.Load(TestData("gguf/tiny-qwen3-q8.gguf"), new PretrainedOptions { Device = device });
                Check(direct.Spec.QkNorm && GgufModel.IsPrepared(ModelSource.Resolve(TestData("gguf/tiny-qwen3-q8.gguf"))), "a .gguf path loads directly");
            }
            finally
            {
                Environment.SetEnvironmentVariable("OLLAMA_MODELS", oldStore);
                Environment.SetEnvironmentVariable("IDRAK_CACHE", oldCache);
            }

        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }
}
