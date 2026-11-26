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
using Idrak.Text.Plugins.Llama;

// GGUF: files loaded by their own names through the families handed over; and the answer metrics' scores.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GgufGroup =
    [
        ("evaluation: final numbers (#### n), exact, contains and F1 scores of answers", EvaluationMetrics),
        ("gguf: a Qwen3 file loads by its own names (its architecture, tokenizer model and pre-tokenizer), its weights as stored, as its Hugging Face twin reads; its own chat template read by the reader handed over", GgufQwen3Loads),
    ];

    // The tiny Qwen3 GGUF fixture (q8_0 weights) prepared and loaded through the built-in families, each part found by the
    // file's own names: its projections kept as q8_0, their values the Hugging Face twin's, its tokenizer the twin's ids,
    // generating; written as q4_k too.
    internal static void GgufQwen3Loads(Device device)
    {
        CpuOnly(device);
        string cache = TempFolder();
        try
        {
            var families = Idrak.Text.Plugins.BuiltInFamilies.All;
            string folder = GgufModel.Prepare(Path.Combine(DataFolder(), "gguf", "tiny-qwen3-q8.gguf"), cache, families);
            string twin = Path.Combine(DataFolder(), "gguf", "tiny-qwen3-q8-hf");
            using var gguf = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Families = families, ChatTemplateReader = JinjaChatTemplate.Reader });
            using var hf = PretrainedModel.Load(twin, new PretrainedOptions { Device = device, Families = families });
            Check(gguf.Architecture.Names.Contains("Qwen3ForCausalLM") && !gguf.Notes.Any(n => n.Contains("pre-tokenizer", StringComparison.Ordinal)),
                $"the family and the pre-tokenizer found by the file's own names: {string.Join("; ", gguf.Notes)}");
            // Its chat template the file's own (tokenizer.chat_template), read by the reader handed over; none without one.
            using (var file = GgufFile.Open(Path.Combine(DataFolder(), "gguf", "tiny-qwen3-q8.gguf")))
            {
                string? own = file.Get<string?>("tokenizer.chat_template", null);
                Check(own is not null && gguf.ChatTemplate is JinjaChatTemplate template && template.Source == own && hf.ChatTemplate is null,
                    "the file's own chat template read by the reader handed over (and none read without one)");
            }

            var projections = Linears(gguf.Network).Where(l => l.TiedTo is null).ToList();
            Check(projections.Count > 0 && projections.All(l => l.Stored?.Type == Idrak.Types.Q8_0.Type),
                $"every projection kept as stored, q8_0: {string.Join(", ", projections.Select(l => l.Stored?.Type.Name ?? "float32"))}");
            using var reader = SafeTensorsReader.Open(Path.Combine(twin, "model.safetensors"));
            var q = projections[0].Stored!.ToArray();
            var expected = reader.Read("model.layers.0.self_attn.q_proj.weight");
            double worst = 0;
            for (int i = 0; i < q.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(q[i] - expected[i]));
            }

            Check(q.Length == expected.Length && worst < 1e-6, $"the first projection's values are the twin's: worst difference {worst}");
            const string text = "hello world, the tokenizer reads 123 numbers";
            Check(gguf.Tokenizer!.Encode(text).SequenceEqual(hf.Tokenizer!.Encode(text)), "the file's tokenizer gives the twin's ids");
            var options = new Idrak.Abstraction.Generation.GenerationOptions { TopK = 1, NumPredict = 6, Seed = 1 };
            _ = new TextGenerator(gguf.Network, gguf.Tokenizer, 64).Generate("hello", options);

            // The other fixtures, each by its own names: its query and key projections and norm gains as its twin's, its
            // tokenizer's ids, generating (Gemma 3's RoPE scaling, linear, read by the method's own name).
            foreach (string name in new[] { "tiny-llama", "tiny-gemma", "tiny-gemma2", "tiny-gemma3" })
            {
                string other = GgufModel.Prepare(Path.Combine(DataFolder(), "gguf", name + ".gguf"), cache, families);
                string otherTwin = Path.Combine(DataFolder(), "gguf", name + "-hf");
                using var store = GgufModel.OpenTensors(other, families);
                using var twinStore = SafeTensorsReader.Open(Directory.GetFiles(otherTwin, "*.safetensors").Single());
                foreach (string tensor in new[] { "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.input_layernorm.weight" })
                {
                    var a = store.Read(tensor);
                    var b = twinStore.Read(tensor);
                    double most = 0;
                    for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
                    {
                        most = Math.Max(most, Math.Abs(a[i] - b[i]));
                    }

                    Check(a.Length == b.Length && most < 1e-3, $"{name}: {tensor} differs from the twin's by {most}");
                }

                using var otherModel = PretrainedModel.Load(other, new PretrainedOptions { Device = device, Families = families });
                using var otherHf = PretrainedModel.Load(otherTwin, new PretrainedOptions { Device = device, Families = families });
                Check(otherModel.Tokenizer!.Encode(text).SequenceEqual(otherHf.Tokenizer!.Encode(text)), $"{name}: the file's tokenizer gives the twin's ids");
                _ = new TextGenerator(otherModel.Network, otherModel.Tokenizer, 64).Generate("hello", options);
                Console.WriteLine($"  {name}.gguf: a {otherModel.Architecture.Names[0]} by its own names, its tensors and token ids the twin's, generating");
            }

            using var written = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Families = families, Weights = Idrak.Types.Q4_K.Type });
            var rewritten = Linears(written.Network).Where(l => l.TiedTo is null).ToList();
            Check(rewritten.All(l => l.Stored is { } s && (s.Type == Idrak.Types.Q4_K.Type || s.Shape[1] % 256 != 0 && s.Type == Idrak.Types.Q8_0.Type)),
                $"written as q4_k where its rows are whole blocks: {string.Join(", ", rewritten.Select(l => l.Stored?.Type.Name ?? "float32"))}");
            Console.WriteLine($"  tiny-qwen3-q8.gguf: {projections.Count} projections as q8_0, the twin's values and token ids, generating; written as {string.Join("/", rewritten.Select(l => l.Stored!.Type.Name).Distinct())}");
        }
        finally
        {
            Directory.Delete(cache, recursive: true);
        }
    }

    internal static void EvaluationMetrics(Device device)
    {
        _ = device;
        Check(TuningMetrics.FinalNumber("so 4 * 3 = 12\n#### 1,200") == 1200m && TuningMetrics.FinalNumber("The answer is $8.") == 8m
              && TuningMetrics.FinalNumber("-3.5 then 7") == 7m && TuningMetrics.FinalNumber("none") is null, "final numbers");
        static double Score(string metric, string answer, string reference) => new AnswerMatchMetric(metric).Score(answer, reference).Value;
        Check(Score(TuningMetrics.NumberMatch, "Tom gets 20 - 12 = $8 back.", "20-12=8\n#### 8") == 1
              && Score(TuningMetrics.NumberMatch, "#### 9", "#### 8") == 0, "number");
        Check(Score(TuningMetrics.ExactMatch, "  Paris. ", "paris") == 1 && Score(TuningMetrics.ContainsMatch, "It is Paris, France", "paris") == 1, "exact, contains");
        Check(Math.Abs(Score(TuningMetrics.WordF1, "the cat sat", "the cat ran") - 2.0 / 3) < 1e-9 && Score(TuningMetrics.WordF1, "dog", "cat") == 0, "f1");
    }
}
