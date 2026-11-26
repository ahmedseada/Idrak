// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

// Plan 27, R3: what the language-model policy in Idrak.Text trains with: the policy losses through FineTuningLosses.Policy
// (with a test-only wrapper of a library loss that records what it was given) and the teacher divergences.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ReinforcementTextGroup =
    [
        ("reinforcement text: a policy loss through FineTuningLosses.Policy gets the token values, behaviour and reference log-probabilities a step's sequences carry (the reference only when the loss reads it: IPolicyLoss.NeedsReference), and the input gives them all or none", PolicyLossAdapter),
        ("reinforcement text: on-policy distillation: forward, reverse and jsd match a direct computation on fixed logits (values and gradients through the chunked head); the library's forward is the distillation's bit for bit", DivergenceKinds),
    ];

    internal const string ChatMlTemplate =
        "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}";

    internal static FineTuningOptions SessionOptions(float rate = 5e-3f) => new()
    {
        Rank = 4, Alpha = 8, LearningRate = rate, BatchTokens = 512, Seed = 7, CudaGraphs = false, Checkpointing = false, RecomputeFeedForward = false, Scheduler = (optimizer, _) => new LambdaSchedule(optimizer, (_, rate) => rate),
    };

    internal static float[] TrainedValues(PretrainedModel model) => [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())];

    internal static bool SameBits(float[] a, float[] b) => a.Length == b.Length && a.Zip(b).All(p => BitConverter.SingleToInt32Bits(p.First) == BitConverter.SingleToInt32Bits(p.Second));

    internal static ActionRecord Turn(int[] prompt, int[] reply) => new(reply, [.. reply.Select(_ => -1f)], float.NaN, prompt);

    // An environment of two turns: a question, then the same question again after the first reply; rewarded at the end.
    internal sealed class TwoTurns : IEnvironment<ChatObservation, ChatMessage>
    {
        private List<ChatMessage> _messages = [];
        private int _turn = -1;

        public Space ObservationSpace => Space.Conversation;

        public Space ActionSpace => Space.Conversation;

        public ChatObservation Reset(int seed)
        {
            _messages = [new ChatMessage("user", "Q")];
            _turn = 0;
            return new([.. _messages]);
        }

        public ValueTask<EnvironmentStep<ChatObservation>> StepAsync(ChatMessage action, CancellationToken cancellationToken = default)
        {
            if (_turn < 0)
            {
                throw new InvalidOperationException("reset first");
            }

            _messages.Add(action);
            if (++_turn == 1)
            {
                _messages.Add(new ChatMessage("user", "Q"));
                return ValueTask.FromResult(new EnvironmentStep<ChatObservation>(new([.. _messages]), 0f, false, false));
            }

            _turn = -1;
            return ValueTask.FromResult(new EnvironmentStep<ChatObservation>(new([.. _messages]), action.Content == "yes" ? 1f : 0f, true, false));
        }

        public void Dispose()
        {
        }
    }

    // A library policy loss under a test name, recording what it was given: the largest |log p − behaviour| of its first
    // call, and whether any call had the reference's log-probabilities. It reads the reference when the library's does.
    internal sealed class RecordingLoss(IPolicyLoss inner) : IPolicyLoss
    {
        public float? FirstGap { get; private set; }

        public bool SawReference { get; private set; }

        public bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => inner.NeedsReference(options);

        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
        {
            FirstGap ??= input.LogProbabilities.ToArray().Zip(input.BehaviorLogProbabilities.ToArray()).Max(p => Math.Abs(p.First - p.Second));
            SawReference |= input.ReferenceLogProbabilities is not null;
            return inner.Loss(input, options);
        }
    }

    // A loss that does not say whether it reads the reference: it is given it (IPolicyLoss.NeedsReference's default).
    internal sealed class SilentLoss : IPolicyLoss
    {
        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null) => input.LogProbabilities.Sum();
    }

    internal static void PolicyLossAdapter(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the losses run on the CPU on every device
        }

        int[] counts = [3, 2, 4];
        var random = new Random(3);
        float[] Values(int n, float scale, float shift) => [.. Enumerable.Range(0, n).Select(_ => shift + scale * (random.NextSingle() - 0.5f))];
        float[] logp = Values(9, 1f, -1.5f), behavior = Values(9, 1f, -1.5f), reference = Values(9, 1f, -1.5f), advantages = Values(9, 2f, 0f);
        var grpo = new ClippedRatioLoss("grpo");
        var weighted = new ClippedRatioLoss("grpo", kl: 0.1f);

        // Whether a loss reads the reference: grpo and ppo when their KL is weighted, reinforce never, a loss that does not say always.
        Check(weighted.NeedsReference() && !grpo.NeedsReference() && !new ClippedRatioLoss("grpo", kl: 0f).NeedsReference()
              && new ClippedRatioLoss("ppo", kl: 0.04f).NeedsReference() && !new ClippedRatioLoss("ppo").NeedsReference()
              && !new ReinforceLoss().NeedsReference(), "NeedsReference: the KL's weight decides");
        Check(((IPolicyLoss)new SilentLoss()).NeedsReference(), "a loss that does not answer is given the reference");

        // The adapter's loss and gradient equal the policy loss's on the same values.
        var leaf = Tensor.From(logp, [9], Device.Cpu, requiresGrad: true);
        var input = new FineTuningLossInput(leaf, counts, Tensor.From(reference, [9], Device.Cpu), stepTokens: 12, tokenValues: Tensor.From(advantages, [9], Device.Cpu),
            behaviorLogProbabilities: Tensor.From(behavior, [9], Device.Cpu));
        var adapted = FineTuningLosses.Policy(weighted).Compute(input);
        adapted.Backward();
        var direct = Tensor.From(logp, [9], Device.Cpu, requiresGrad: true);
        var expected = weighted.Loss(new PolicyLossInput(direct, counts, Tensor.From(advantages, [9], Device.Cpu), Tensor.From(behavior, [9], Device.Cpu),
            Tensor.From(reference, [9], Device.Cpu), stepUnits: 12));
        expected.Backward();
        AssertClose([expected.Item()], [adapted.Item()], 1e-6f, "the policy loss through the adapter");
        AssertClose(direct.Grad!.ToArray(), leaf.Grad!.ToArray(), 1e-6f, "its gradient");

        // A loss that does not read the reference (grpo without a KL) is not given it, though the input has one.
        var recorder = new RecordingLoss(grpo);
        var noReference = FineTuningLosses.Policy(recorder).Compute(new FineTuningLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(reference, [9], Device.Cpu),
            tokenValues: Tensor.From(advantages, [9], Device.Cpu)));
        var expectedNoReference = grpo.Loss(new PolicyLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(advantages, [9], Device.Cpu)));
        AssertClose([expectedNoReference.Item()], [noReference.Item()], 1e-6f, "on-policy, no reference");
        Check(!recorder.SawReference, "the reference is left out for a loss that does not read it");
        var reads = new RecordingLoss(weighted);
        _ = FineTuningLosses.Policy(reads).Compute(new FineTuningLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(reference, [9], Device.Cpu),
            tokenValues: Tensor.From(advantages, [9], Device.Cpu)));
        Check(reads.SawReference, "and given to one that reads it");

        // From the sequences: a value each (repeated over its tokens) or token values, behaviour log-probabilities all or none.
        TrainingSequence Sequence(int trained, float? value = null, float[]? tokens = null, float[]? behaviour = null) =>
            new([.. Enumerable.Range(0, trained + 1)], [false, .. Enumerable.Repeat(true, trained)]) { Value = value, TokenValues = tokens, BehaviorLogProbabilities = behaviour };
        var values = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 0.5f), Sequence(2, tokens: [1f, 2f])], null, false, 0, 0);
        Check(values.TokenValues.ToArray().SequenceEqual([0.5f, 0.5f, 0.5f, 1f, 2f]) && !values.HasBehavior, "a value repeated, token values as given, no behaviour");
        ThrowsNaming<InvalidOperationException>(() => _ = values.SequenceValues, "a sequence without a value", "TrainingSequence.Value");
        var mixed = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 1f, behaviour: [-1f, -1f, -1f]), Sequence(2, 1f)], null, false, 0, 0);
        ThrowsNaming<InvalidOperationException>(() => _ = mixed.BehaviorLogProbabilities, "behaviour for some sequences only", "every sequence or none");
        var both = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 1f, behaviour: [-1f, -2f, -3f]), Sequence(2, 2f, behaviour: [-4f, -5f])], null, false, 0, 0);
        Check(both.BehaviorLogProbabilities!.ToArray().SequenceEqual([-1f, -2f, -3f, -4f, -5f]) && both.SequenceValues.ToArray().SequenceEqual([1f, 2f]), "behaviour and sequence values joined in order");
    }

    // One prompt, its sequences for answer `answer` (the prompt as the template renders it, the answer then the end of turn).
    internal static TrainingSequence AnswerSequence(PretrainedModel model, IReadOnlyList<ChatMessage> prompt, string answer)
    {
        int[] p = [.. model.Tokenizer!.Encode(model.ChatTemplate!.Render(prompt, [], null))], a = [.. model.Tokenizer.Encode(answer + "<|im_end|>")];
        return new TrainingSequence([.. p, .. a], [.. p.Select(_ => false), .. a.Select(_ => true)]);
    }

    internal static void DivergenceKinds(Device device)
    {
        const int Positions = 7, Dim = 10, Vocabulary = 17;
        var random = new Random(51);
        float[] Values(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2f - 1f) * scale)];
        float[] h = Values(Positions * Dim, 1f), w = Values(Dim * Vocabulary, 0.8f);
        int[] rows = [0, 2, 3, 5, 6, 2];
        float[] teacherLogits = Values(rows.Length * Vocabulary, 3f), weights = Values(rows.Length, 1f);
        const float Temperature = 1.5f;
        var q = new double[rows.Length * Vocabulary];
        for (int r = 0; r < rows.Length; r++)
        {
            double max = Enumerable.Range(0, Vocabulary).Max(v => teacherLogits[r * Vocabulary + v] / Temperature);
            var e = Enumerable.Range(0, Vocabulary).Select(v => Math.Exp(teacherLogits[r * Vocabulary + v] / Temperature - max)).ToArray();
            q[r * Vocabulary + 3] = 0;                                         // a zero of q, read as 1e-30 inside the logarithms
            for (int v = 0; v < Vocabulary; v++)
            {
                q[r * Vocabulary + v] = v == 3 ? 0 : e[v] / (e.Sum() - e[3]);
            }
        }

        // The divergence of one row from its logits z, in double precision, as the divergences' documentation writes it.
        static double Divergence(string kind, double[] z, double[] qr, double temperature, double beta)
        {
            double max = z.Max() / temperature, sum = z.Sum(x => Math.Exp(x / temperature - max));
            double[] logp = [.. z.Select(x => x / temperature - max - Math.Log(sum))];
            double Log(double x) => Math.Log(Math.Max(x, 1e-30));
            double d = 0;
            for (int v = 0; v < z.Length; v++)
            {
                double p = Math.Exp(logp[v]), m = beta * qr[v] + (1 - beta) * p;
                d += kind switch
                {
                    "forward" => qr[v] * (Log(qr[v]) - logp[v]),
                    "reverse" => p * (logp[v] - Log(qr[v])),
                    _ => beta * qr[v] * (Log(qr[v]) - Log(m)) + (1 - beta) * p * (logp[v] - Log(m)),
                };
            }

            return d;
        }

        foreach (var (kind, divergence, beta) in new (string, ITeacherDivergence, float)[] { ("forward", new TeacherDivergences.ForwardDivergence(), 0f), ("reverse", new TeacherDivergences.ReverseDivergence(), 0f), ("jsd", new TeacherDivergences.JensenShannonDivergence(), 0.3f) })
        {
            var expected = new float[rows.Length];
            var dz = new double[rows.Length * Vocabulary];
            for (int r = 0; r < rows.Length; r++)
            {
                var z = new double[Vocabulary];
                for (int v = 0; v < Vocabulary; v++)
                {
                    for (int d = 0; d < Dim; d++)
                    {
                        z[v] += h[rows[r] * Dim + d] * w[d * Vocabulary + v];
                    }
                }

                double[] qr = q[(r * Vocabulary)..((r + 1) * Vocabulary)];
                expected[r] = (float)Divergence(kind, z, qr, Temperature, beta);
                for (int v = 0; v < Vocabulary; v++)
                {
                    // Central differences: the gradient checked against the divergence itself, not against its formula.
                    double saved = z[v];
                    z[v] = saved + 1e-5;
                    double up = Divergence(kind, z, qr, Temperature, beta);
                    z[v] = saved - 1e-5;
                    double down = Divergence(kind, z, qr, Temperature, beta);
                    z[v] = saved;
                    dz[r * Vocabulary + v] = weights[r] * (up - down) / 2e-5;
                }
            }

            var expectedHidden = new float[Positions * Dim];
            for (int r = 0; r < rows.Length; r++)
            {
                for (int d = 0; d < Dim; d++)
                {
                    for (int v = 0; v < Vocabulary; v++)
                    {
                        expectedHidden[rows[r] * Dim + d] += (float)(dz[r * Vocabulary + v] * w[d * Vocabulary + v]);
                    }
                }
            }

            using var scope = new TensorScope();
            var hidden = Tensor.From(h, [Positions, Dim], device, requiresGrad: true);
            var weight = Tensor.From(w, [Dim, Vocabulary], device);
            float[] qf = [.. q.Select(v => (float)v)];
            Tensor Teacher(int start, int count) => Tensor.From(qf.AsSpan(start * Vocabulary, count * Vocabulary), [count, Vocabulary], device);
            TokenDivergenceRule rule = (s, t, g) => divergence.Rows(s, t, beta, g);
            var values = Tensor.TokenDivergences(hidden, x => x.MatMul(weight), rows, Teacher, Temperature, 4, rule);
            AssertClose(expected, values, 1e-4f, $"{kind}: the divergences");
            var lossTensor = Tensor.TokenDivergenceRows(hidden, x => x.MatMul(weight), rows, weights, Teacher, Temperature, 4, rule);
            AssertClose([(float)expected.Select((v, i) => (double)v * weights[i]).Sum()], [lossTensor.Item()], 1e-4f, $"{kind}: the weighted sum");
            lossTensor.Backward();
            CloseByNorm(expectedHidden, hidden.Grad!.ToArray(), 1e-3f, $"{kind}: the hidden gradient against central differences");
            if (kind == "forward")
            {
                var built = Tensor.TokenDivergences(hidden, x => x.MatMul(weight), rows, Teacher, Temperature, 4);
                Check(SameBits(built, values), "the library's forward KL is the distillation's own, bit for bit");
            }
        }
    }
}
