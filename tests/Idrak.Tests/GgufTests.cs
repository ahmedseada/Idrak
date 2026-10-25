// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins;
using Idrak.Text.Plugins.Llama;

// GGUF: dequantization checked against the gguf package's reference, and models read from GGUF files against the same
// models in the Hugging Face layout (tools/gguf/make_fixtures.py makes both; tools/gguf/make_gemma.py the Gemma files, with
// llama.cpp's own converter).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] GgufGroup =
    [
        ("gguf: metadata, and F16, BF16, Q4_0–Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS dequantized as llama.cpp's reference does", GgufQuants),
        ("evaluation: final numbers (#### n), exact, contains and F1 scores of answers", EvaluationMetrics),
        ("gguf: Llama (q/k row order) and Qwen3 (Q8_0) models read from GGUF match their Hugging Face copies: config, tokens, chat, loss", GgufModels),
        ("gguf: gemma: a Gemma file written by llama.cpp's converter (f16, its norm gains stored with the 1 added) reads as its Hugging Face folder: spec, SentencePiece tokens and the chat prompt's, chat template, logits, greedy tokens", device => GgufGemma(device, "tiny-gemma", "GemmaForCausalLM")),
        ("gguf: gemma2: a Gemma 2 file written by llama.cpp's converter reads as its Hugging Face folder (post-norms under llama.cpp's names, GELU, soft-capping and window from the file, the query scalar llama.cpp takes): spec, tokens, chat template, logits over more positions than the window, greedy tokens", device => GgufGemma(device, "tiny-gemma2", "Gemma2ForCausalLM")),
        ("gguf: gemma3: a Gemma 3 file written by llama.cpp's converter reads as its Hugging Face folder (q/k norms, five windowed layers in six as llama.cpp assumes, the windowed layers' RoPE base from the file and the global one scaled): spec, tokens, chat template, logits over more positions than the window, greedy tokens", device => GgufGemma(device, "tiny-gemma3", "Gemma3ForCausalLM")),
        ("gguf: chat templates: a file without one gets the first registered fallback for its tokens (the library's ChatML, then Llama 3's from Idrak.Text.Plugins), an app's too; with none for its tokens it has no template, as a folder without one, and the note names GgufChatTemplates; a name nobody registered is refused as not registered", GgufChatTemplateFallbacks),
        ("gguf: F16 widened whole vectors at a time (IGgufType.Dequantize, through HalfFloats) gives .NET's own conversion bit for bit for every half (zeros, subnormals, infinities) and a NaN with its payload kept as the devices' float16 kernels keep it (not quieted), at a vector boundary and in the scalar tail", GgufHalfWidening),
    ];

    private static void GgufHalfWidening(Device device)
    {
        _ = device;
        var bytes = new byte[65536 * 2];
        for (int h = 0; h < 65536; h++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(h * 2), (ushort)h);
        }

        var f16 = GgufTypes.Get(1);
        foreach (int start in new[] { 0, 1, 3 })
        {
            int count = 65536 - start - start;
            var values = new float[count];
            f16.Dequantize(bytes.AsSpan(start * 2, count * 2), values);
            for (int i = 0; i < count; i++)
            {
                ushort h = (ushort)(start + i);
                uint expected = (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0 ? ((uint)(h & 0x8000) << 16) | 0x7F800000u | ((uint)(h & 0x3FF) << 13)
                    : BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(h));
                uint actual = BitConverter.SingleToUInt32Bits(values[i]);
                if (expected != actual)
                {
                    Check(false, $"half 0x{h:X4} from {start}: 0x{actual:X8}, expected 0x{expected:X8}");
                    return;
                }
            }
        }
    }

    private static void EvaluationMetrics(Device device)
    {
        _ = device;
        Check(TuningMetrics.FinalNumber("so 4 * 3 = 12\n#### 1,200") == 1200m && TuningMetrics.FinalNumber("The answer is $8.") == 8m
              && TuningMetrics.FinalNumber("-3.5 then 7") == 7m && TuningMetrics.FinalNumber("none") is null, "final numbers");
        Check(ChatEvaluation.Score(TuningMetrics.NumberMatch, "Tom gets 20 - 12 = $8 back.", "20-12=8\n#### 8") == 1
              && ChatEvaluation.Score(TuningMetrics.NumberMatch, "#### 9", "#### 8") == 0, "number");
        Check(ChatEvaluation.Score(TuningMetrics.ExactMatch, "  Paris. ", "paris") == 1 && ChatEvaluation.Score(TuningMetrics.ContainsMatch, "It is Paris, France", "paris") == 1, "exact, contains");
        Check(Math.Abs(ChatEvaluation.Score(TuningMetrics.WordF1, "the cat sat", "the cat ran") - 2.0 / 3) < 1e-9 && ChatEvaluation.Score(TuningMetrics.WordF1, "dog", "cat") == 0, "f1");
        Check(ChatEvaluation.Score(ChatEvaluation.Auto, "so it is 14", "#### 14") == 1, "auto picks the number for #### references");
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

    // A Gemma GGUF file of tools/gguf/make_gemma.py against the Hugging Face folder it was converted from.
    private static void GgufGemma(Device device, string name, string architecture)
    {
        string cache = Path.Combine(Path.GetTempPath(), "ns-gguf-" + Guid.NewGuid().ToString("N"));
        try
        {
            string folder = GgufModel.Prepare(TestData($"gguf/{name}.gguf"), cache);
            using var fromGguf = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            using var fromHf = PretrainedModel.Load(TestData($"gguf/{name}-hf"), new PretrainedOptions { Device = device });
            var (specGguf, specHf) = (fromGguf.Spec.ToJson(), fromHf.Spec.ToJson());
            Check((string?)fromGguf.Config["architectures"]?[0] == architecture && JsonNode.DeepEquals(specGguf, specHf),
                $"{name}: same model spec\n  gguf {specGguf.ToJsonString()}\n  hf   {specHf.ToJsonString()}");
            Check(fromGguf.Spec is { NormOffset: 1f, Activation: FeedForwardActivation.Gelu } && fromGguf.Notes.Any(n => n.Contains(".gguf", StringComparison.Ordinal)),
                $"{name}: Gemma's (1 + gain) norms and GELU; notes say where the weights come from: {string.Join("; ", fromGguf.Notes)}");

            // The file's SentencePiece tokenizer (tokenizer.ggml.model "llama") encodes as the folder's tokenizer.json, the
            // chat template's turn markers as their special tokens.
            foreach (string text in new[] { "the image is a scan of a page with lines of text", "Write one line about the moon and the sun in the night.",
                         "what is in this image ?\n\nread the scan", "  two  spaces and\ttabs ", "العلم نور" })
            {
                var ids = fromGguf.Tokenizer!.Encode(text);
                Check(ids.SequenceEqual(fromHf.Tokenizer!.Encode(text)) && fromGguf.Tokenizer.Decode(ids) == fromHf.Tokenizer.Decode(ids),
                    $"{name}: tokens of '{text}': [{string.Join(",", ids)}] vs [{string.Join(",", fromHf.Tokenizer!.Encode(text))}]");
            }

            var chat = new[] { new ChatMessage("user", "Write one line about the moon.") };
            string prompt = fromHf.ChatTemplate!.Render(chat, [], null);
            Check(fromGguf.ChatTemplate!.Render(chat, [], null) == prompt && fromGguf.Tokenizer!.Encode(prompt).SequenceEqual(fromHf.Tokenizer!.Encode(prompt))
                  && fromGguf.Tokenizer.Encode(prompt).Contains(((IPretrainedTokenizer)fromHf.Tokenizer).IdOf("<start_of_turn>")!.Value),
                $"{name}: the chat prompt and its tokens: [{string.Join(",", fromGguf.Tokenizer!.Encode(prompt))}] vs [{string.Join(",", fromHf.Tokenizer!.Encode(prompt))}]");

            // Logits over more positions than the window (8), and greedy tokens.
            int[] tokens = [.. fromHf.Tokenizer.Encode("the image is a scan of a page with lines of text and a box in the corner")];
            Check(tokens.Length > 12, $"{name}: {tokens.Length} tokens");
            using var input = Tensor.From([.. tokens.Select(t => (float)t)], [1, tokens.Length], device);
            AssertClose(fromHf.Network.Predict(input).ToArray(), fromGguf.Network.Predict(input).ToArray(), 1e-4f, $"{name}: logits from GGUF against the Hugging Face folder");
            var greedy = new GenerationOptions { Temperature = 0, RepeatPenalty = 1, NumPredict = 16, Seed = 1 };
            string textGguf = fromGguf.CreateGenerator().Generate("the moon and the sun", greedy).Text, textHf = fromHf.CreateGenerator().Generate("the moon and the sun", greedy).Text;
            Check(textGguf == textHf && fromHf.Tokenizer.Encode(textHf).Count > 4, $"{name}: greedy text '{textGguf}' from GGUF, '{textHf}' from the Hugging Face folder");
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    private static void GgufChatTemplateFallbacks(Device device)
    {
        _ = device;
        string library = Overrides.Library;
        Check(GgufChatTemplates.Names.First() == "chatml" && GgufChatTemplates.Origin("chatml") == library && GgufChatTemplates.Default("chatml") is not null
              && GgufChatTemplates.Origin(LlamaPlugin.ChatTemplateName) == TextPlugins.Package.Name && GgufChatTemplates.Default(LlamaPlugin.ChatTemplateName) is null,
            $"the library's ChatML first, Llama 3's the plug-in's: {string.Join(", ", GgufChatTemplates.Names.Select(n => $"{n}={GgufChatTemplates.Origin(n)}"))}");
        string refused = Failure<NotSupportedException>(() => GgufChatTemplates.Get("imaginary")).Message;
        Check(refused.Contains("GGUF chat template 'imaginary' is not registered", StringComparison.Ordinal) && refused.Contains("GgufChatTemplates.Register", StringComparison.Ordinal)
              && refused.Contains("Idrak.Text.Plugins", StringComparison.Ordinal), $"a name nobody registered: {refused}");

        var root = Directory.CreateTempSubdirectory("idrak-gguf-template-");
        try
        {
            // A GGUF tokenizer of these tokens (and template), prepared: the chat template written, and the notes.
            int count = 0;
            (string? Template, List<string> Notes, string Folder) Prepared(string[] tokens, string? template = null)
            {
                var notes = new List<string>();
                List<KeyValuePair<string, object>> metadata = [new("tokenizer.ggml.model", "gpt2"), new("tokenizer.ggml.tokens", tokens)];
                if (template is not null)
                {
                    metadata.Add(new("tokenizer.chat_template", template));
                }

                string folder = PrepareGgufTokenizer(root.FullName, $"t{count++}", metadata, notes);
                return ((string?)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "tokenizer_config.json")))!["chat_template"], notes, folder);
            }

            string[] chatMl = ["a", "b", "<|im_start|>", "<|im_end|>"], llama3 = ["a", "b", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"];
            var (written, notes, _) = Prepared(chatMl);
            Check(written == GgufChatTemplates.Default("chatml")!.Template && written.Contains("<|im_start|>", StringComparison.Ordinal)
                  && notes.Any(n => n.Contains("'chatml'", StringComparison.Ordinal) && n.Contains("GgufChatTemplates", StringComparison.Ordinal)), $"ChatML's tokens: {string.Join("; ", notes)}");
            (written, notes, _) = Prepared(llama3);
            Check(written == LlamaArchitectures.ChatTemplateText && notes.Any(n => n.Contains("'llama3'", StringComparison.Ordinal)), $"Llama 3's tokens: {string.Join("; ", notes)}");
            (written, _, _) = Prepared([.. chatMl, .. llama3[2..]]);
            Check(written == GgufChatTemplates.Default("chatml")!.Template, "both: ChatML, the library's asked first (as before the registry)");
            (written, notes, _) = Prepared(chatMl, "{{ 'own' }}");
            Check(written == "{{ 'own' }}" && notes.Count == 0, "a file's own template is kept, no fallback asked");

            // An app's template for its tokens, and over the library's; unregistered, the library's comes back.
            GgufChatTemplates.Register("test-turns", new GgufChatTemplate { Matches = (_, tokens) => tokens.Contains("<|user|>"), Template = "{{ 'test' }}" });
            GgufChatTemplates.Register("chatml", new GgufChatTemplate { Matches = (_, tokens) => tokens.Contains("<|im_start|>"), Template = "{{ 'app chatml' }}" });
            try
            {
                Check(GgufChatTemplates.Origin("test-turns") == "Idrak.Tests" && GgufChatTemplates.Names.Last() == "test-turns" && GgufChatTemplates.Names.First() == "chatml",
                    "an app's new name is asked last; its template over the library's keeps the library's place");
                Check(Prepared(["a", "<|user|>"]).Template == "{{ 'test' }}" && Prepared(chatMl).Template == "{{ 'app chatml' }}", "the app's templates are used");
            }
            finally
            {
                Check(GgufChatTemplates.Unregister("test-turns") && GgufChatTemplates.Unregister("chatml") && !GgufChatTemplates.Unregister("chatml"), "unregistered");
            }

            Check(GgufChatTemplates.Find("test-turns") is null && Prepared(chatMl).Template == GgufChatTemplates.Default("chatml")!.Template, "the library's ChatML is back");

            // Without the families, Llama 3's tokens find no template: none is written, as a folder without one has none.
            WithoutTextFamilies(() =>
            {
                Check(GgufChatTemplates.Find(LlamaPlugin.ChatTemplateName) is null && GgufChatTemplates.Names.SequenceEqual(["chatml"]), "only the library's ChatML is left");
                var (none, why, folder) = Prepared(llama3);
                Check(none is null && Idrak.Abstraction.Generation.ChatTemplates.Load(folder) is null, "no chat template, as a model folder without one");
                Check(why.Any(n => n.Contains("no fallback registered in GgufChatTemplates (chatml)", StringComparison.Ordinal) && n.Contains("chat is not available", StringComparison.Ordinal)
                      && n.Contains("GgufChatTemplates.Register", StringComparison.Ordinal) && n.Contains("Idrak.Text.Plugins", StringComparison.Ordinal)), $"the note names the registry: {string.Join("; ", why)}");
            });
            Check(GgufChatTemplates.Origin(LlamaPlugin.ChatTemplateName) == TextPlugins.Package.Name && Prepared(llama3).Template == LlamaArchitectures.ChatTemplateText,
                "registered again, Llama 3's template is back");
        }
        finally
        {
            root.Delete(recursive: true);
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
                      && fromGguf.ChatTemplate!.StopSequences.Contains("<|im_end|>"), $"{name}: chat template and stop");

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
