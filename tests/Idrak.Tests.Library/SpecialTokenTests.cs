// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Models;
using Idrak.Text;

// A special token sampled mid-reply (a model's begin-of-sentence token in its code) is left out of the text, as
// transformers' skip_special_tokens=True, while the end-of-turn marker still ends the reply and the ids stay as sampled.
internal abstract partial class Tests
{
    // Gives the same scripted ids on every row, step by step (the last one again once the script runs out).
    internal sealed class ScriptedSampler(SamplerRequest request, int[] script) : ITokenSampler
    {
        private readonly List<int[]> _steps = [];

        public int Rows => request.Rows;

        public int Vocabulary => request.Vocabulary;

        public Tensor Ids { get; } = Tensor.Persistent(new float[request.Rows], [request.Rows], request.Device, requiresGrad: false);

        public bool Recordable => false;

        public void Sample(Tensor logits)
        {
            int id = script[Math.Min(_steps.Count, script.Length - 1)];
            var ids = Enumerable.Repeat(id, Rows).ToArray();
            Ids.Load([.. ids.Select(i => (float)i)]);
            _steps.Add(ids);
        }

        public void SetHistory(IReadOnlyList<int> tokens)
        {
        }

        public void Reset() => _steps.Clear();

        public SampledToken[][] Read(int fromStep, int toStep) =>
            [.. _steps.Skip(fromStep).Take(toStep - fromStep).Select(ids => ids.Select(id => new SampledToken(id, 1f, 0f, [(id, 1f), (-1, 0f), (-1, 0f), (-1, 0f), (-1, 0f)])).ToArray())];

        public void Dispose() => Ids.Dispose();
    }

    internal static void SpecialTokensLeftOutOfText(Device device)
    {
        if (device.Type != DeviceType.Cpu || !FirstRun(nameof(SpecialTokensLeftOutOfText)))
        {
            return;                                                             // the text of ids: one device shows it
        }

        string folder = TempFolder("special-tokens");
        PretrainedModel.Create(folder, TinyLlamaSpec(), "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 4 }).Dispose();
        using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
        var tokenizer = (IPretrainedTokenizer)model.Tokenizer!;
        int bos = tokenizer.IdOf("<s>")!.Value, end = tokenizer.IdOf("<|im_end|>")!.Value;
        int[] words = [.. tokenizer.Encode("print hello").Where(id => !tokenizer.IsSpecial(id))];
        Check(tokenizer.IsSpecial(bos) && tokenizer.IsSpecial(end) && words.Length >= 2, $"the tokenizer marks <s> and <|im_end|> special ({bos}, {end}; {words.Length} word ids)");
        int[] script = [words[0], bos, words[1], end, words[0]];
        string expected = tokenizer.Decode(words[..2]);
        var options = new GenerationOptions { Stop = ["<|im_end|>"], NumPredict = 8, ReturnTokens = true, UseGraph = false };

        TextGenerator Generator(bool skip) => new(model.Network, tokenizer, 64) { KeepCache = false, SkipSpecialTokens = skip, CreateSampler = request => new ScriptedSampler(request, script) };
        var chunks = Generator(skip: true).Stream("x", options).ToList();
        string text = string.Concat(chunks.Select(c => c.Text));
        Check(text == expected && chunks[^1].DoneReason == "stop", $"the <s> sampled mid-reply is left out, the reply ends at <|im_end|>: '{text}' (expected '{expected}', {chunks[^1].DoneReason})");
        Check(chunks[^1].Tokens!.Reply.SequenceEqual(script[..4]), $"the ids are kept as sampled: {string.Join(", ", chunks[^1].Tokens!.Reply)}");

        var batch = Generator(skip: true).GenerateBatch(["x", "y"], options);
        Check(batch.All(r => r.Text == expected && r.DoneReason == "stop"), $"a batch leaves it out too: {string.Join(" | ", batch.Select(r => $"'{r.Text}' {r.DoneReason}"))}");

        string written = string.Concat(Generator(skip: false).Stream("x", options).Select(c => c.Text));
        Check(written.Contains("<s>", StringComparison.Ordinal), $"SkipSpecialTokens false writes it: '{written}'");
    }
}
