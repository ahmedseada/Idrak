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

// GGUF: a file's own chat template as the tokenizer's folder is prepared; and the answer metrics' scores.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GgufGroup =
    [
        ("evaluation: final numbers (#### n), exact, contains and F1 scores of answers", EvaluationMetrics),
        ("gguf: chat templates: a file's own chat template is written as it is, with no note", GgufChatTemplateFallbacks),
    ];

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

    internal static void GgufChatTemplateFallbacks(Device device)
    {
        _ = device;
        var root = Directory.CreateTempSubdirectory("idrak-gguf-template-");
        try
        {
            // A GGUF tokenizer of these tokens and its own template, prepared: the chat template written, and the notes.
            var notes = new List<string>();
            List<KeyValuePair<string, object>> metadata =
                [new("tokenizer.ggml.model", "gpt2"), new("tokenizer.ggml.tokens", new[] { "a", "b", "<|im_start|>", "<|im_end|>" }), new("tokenizer.chat_template", "{{ 'own' }}")];
            string folder = PrepareGgufTokenizer(root.FullName, "t0", metadata, notes);
            string? written = (string?)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "tokenizer_config.json")))!["chat_template"];
            Check(written == "{{ 'own' }}" && notes.Count == 0, "a file's own template is kept, no fallback asked");
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }
}
