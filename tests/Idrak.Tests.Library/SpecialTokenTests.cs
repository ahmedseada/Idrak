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
}
