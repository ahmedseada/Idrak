// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;
using Idrak.Training;

// The teacher pattern (knowledge distillation): the per-token divergence and its gradient, the classifier loss, padded
// vocabularies and the prompts a teacher answers.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DistillationGroup =
    [
        ("distillation: the token divergence (full and top-k teachers, chunked, repeated rows, a temperature) and its gradient match a dense computation", DivergenceFormula),
        ("distillation: the classifier loss matches its formula and gradient, alone and mixed with one-hot or index labels; WithTeacher puts the teacher's logits before the targets", ClassifierDistillation),
        ("distillation: padded vocabularies match, their fingerprints alike; the teacher's prompts are read from the data's rows", VocabularyCheck),
    ];

    private static void DivergenceFormula(Device device)
    {
        const int Positions = 9, Dim = 12, Vocabulary = 23;
        var random = new Random(41);
        float[] Values(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2f - 1f) * scale)];
        float[] h = Values(Positions * Dim, 1f), w = Values(Dim * Vocabulary, 0.8f);
        int[] rows = [0, 2, 3, 5, 6, 2, 8];
        float[] teacherLogits = Values(rows.Length * Vocabulary, 3f), weights = Values(rows.Length, 1f);
        foreach (var (temperature, topK) in new[] { (1f, 0), (2f, 0), (1.5f, 5) })
        {
            // The teacher's probabilities on the host: a softmax at the temperature, or over each row's top k (renormalized).
            var q = new double[rows.Length * Vocabulary];
            for (int r = 0; r < rows.Length; r++)
            {
                var row = teacherLogits.AsSpan(r * Vocabulary, Vocabulary).ToArray();
                var kept = topK == 0 ? Enumerable.Range(0, Vocabulary).ToArray() : [.. Enumerable.Range(0, Vocabulary).OrderByDescending(v => row[v]).Take(topK)];
                double max = kept.Max(v => row[v]), sum = kept.Sum(v => Math.Exp((row[v] - max) / temperature));
                foreach (int v in kept)
                {
                    q[r * Vocabulary + v] = Math.Exp((row[v] - max) / temperature) / sum;
                }
            }

            // The student's logits z = h·W at the listed rows, the divergences and d(Σ w·KL)/dz = w (p − q) / T.
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

                double max = z.Max() / temperature, sum = z.Sum(x => Math.Exp(x / temperature - max));
                double kl = 0;
                for (int v = 0; v < Vocabulary; v++)
                {
                    double logp = z[v] / temperature - max - Math.Log(sum), qv = q[r * Vocabulary + v];
                    kl += qv > 0 ? qv * (Math.Log(qv) - logp) : 0;
                    dz[r * Vocabulary + v] = weights[r] * (Math.Exp(logp) - qv) / temperature;
                }

                expected[r] = (float)kl;
            }

            var expectedHidden = new float[Positions * Dim];
            var expectedWeight = new float[Dim * Vocabulary];
            for (int r = 0; r < rows.Length; r++)
            {
                for (int d = 0; d < Dim; d++)
                {
                    for (int v = 0; v < Vocabulary; v++)
                    {
                        expectedHidden[rows[r] * Dim + d] += (float)(dz[r * Vocabulary + v] * w[d * Vocabulary + v]);
                        expectedWeight[d * Vocabulary + v] += (float)(h[rows[r] * Dim + d] * dz[r * Vocabulary + v]);
                    }
                }
            }

            using var scope = new TensorScope();
            var hidden = Tensor.From(h, [Positions, Dim], device, requiresGrad: true);
            var weight = Tensor.From(w, [Dim, Vocabulary], device, requiresGrad: true);
            float[] qf = [.. q.Select(v => (float)v)];
            Tensor Teacher(int start, int count) => Tensor.From(qf.AsSpan(start * Vocabulary, count * Vocabulary), [count, Vocabulary], device);
            string what = $"T = {temperature}, top-k {topK}";
            var values = Tensor.TokenDivergences(hidden, x => x.MatMul(weight), rows, Teacher, temperature, chunkRows: 3);
            AssertClose(expected, values, 1e-4f, $"divergences, {what}");
            var loss = Tensor.TokenDivergenceRows(hidden, x => x.MatMul(weight), rows, weights, Teacher, temperature, chunkRows: 4);
            AssertClose([(float)expected.Select((v, i) => (double)v * weights[i]).Sum()], [loss.Item()], 1e-4f, $"weighted sum, {what}");
            loss.Backward();
            CloseByNorm(expectedHidden, hidden.Grad!.ToArray(), 1e-4f, $"hidden gradient, {what}");
            CloseByNorm(expectedWeight, weight.Grad!.ToArray(), 1e-4f, $"head gradient, {what}");
        }
    }

    private static void ClassifierDistillation(Device device)
    {
        const int Rows = 6, Classes = 4;
        var random = new Random(43);
        float[] student = [.. Enumerable.Range(0, Rows * Classes).Select(_ => 2f * random.NextSingle() - 1f)];
        float[] teacher = [.. Enumerable.Range(0, Rows * Classes).Select(_ => 4f * random.NextSingle() - 2f)];
        int[] labels = [.. Enumerable.Range(0, Rows).Select(_ => random.Next(Classes))];
        const float T = 2.5f, Alpha = 0.3f;

        // The formula on the host: T² · mean KL(q_T ‖ p_T), its gradient T · (p_T − q_T) / rows; the label term's (p − y) / rows.
        double soft = 0, hard = 0;
        var softGradient = new float[Rows * Classes];
        var hardGradient = new float[Rows * Classes];
        double[] Softmax(float[] x, int r, float t)
        {
            double max = Enumerable.Range(0, Classes).Max(c => x[r * Classes + c]) / t;
            var e = Enumerable.Range(0, Classes).Select(c => Math.Exp(x[r * Classes + c] / t - max)).ToArray();
            return [.. e.Select(v => v / e.Sum())];
        }

        for (int r = 0; r < Rows; r++)
        {
            double[] p = Softmax(student, r, T), q = Softmax(teacher, r, T), p1 = Softmax(student, r, 1f);
            for (int c = 0; c < Classes; c++)
            {
                soft += q[c] * (Math.Log(q[c]) - Math.Log(p[c]));
                softGradient[r * Classes + c] = (float)(T * (p[c] - q[c]) / Rows);
                hardGradient[r * Classes + c] = (float)((p1[c] - (labels[r] == c ? 1 : 0)) / Rows);
            }

            hard -= Math.Log(p1[labels[r]]);
        }

        soft *= T * T / Rows;
        hard /= Rows;
        using (var scope = new TensorScope())
        {
            var s = Tensor.From(student, [Rows, Classes], device, requiresGrad: true);
            var value = Losses.Distillation(s, Tensor.From(teacher, [Rows, Classes], device), T);
            value.Backward();
            AssertClose([(float)soft], [value.Item()], 1e-5f, "distillation loss against its formula");
            AssertClose(softGradient, s.Grad!.ToArray(), 1e-5f, "distillation gradient");
        }

        // Mixed with labels, given as one-hot rows or as class indices after the teacher's logits.
        foreach (bool oneHot in new[] { true, false })
        {
            int columns = Classes + (oneHot ? Classes : 1);
            var target = new float[Rows * columns];
            for (int r = 0; r < Rows; r++)
            {
                teacher.AsSpan(r * Classes, Classes).CopyTo(target.AsSpan(r * columns));
                if (oneHot)
                {
                    target[r * columns + Classes + labels[r]] = 1f;
                }
                else
                {
                    target[r * columns + Classes] = labels[r];
                }
            }

            using var scope = new TensorScope();
            var s = Tensor.From(student, [Rows, Classes], device, requiresGrad: true);
            var value = Losses.Distillation(T, Alpha)(s, Tensor.From(target, [Rows, columns], device));
            value.Backward();
            string labelsAs = oneHot ? "one-hot labels" : "index labels";
            AssertClose([(float)(Alpha * soft + (1 - Alpha) * hard)], [value.Item()], 1e-5f, $"mixed loss, {labelsAs}");
            AssertClose([.. softGradient.Zip(hardGradient, (a, b) => Alpha * a + (1 - Alpha) * b)], s.Grad!.ToArray(), 1e-5f, $"mixed gradient, {labelsAs}");
        }

        Check(Refused(() => Losses.Distillation(T, Alpha)(Tensor.Zeros([2, 3], device), Tensor.Zeros([2, 5], device))), "targets without the teacher's logits and labels are refused");

        // WithTeacher: the teacher's logits first, then the labels; a student trained on them learns the teacher's outputs.
        var features = new float[64, 5];
        var classes = new int[64];
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                features[i, j] = 2f * random.NextSingle() - 1f;
            }

            classes[i] = features[i, 0] + features[i, 1] > 0 ? 1 : features[i, 2] > 0 ? 2 : 0;
        }

        var data = Dataset.FromClassLabels(features, classes, 3);
        using var teacherModel = new Sequential { new Linear(5, 3, device: device, random: new Random(1)) };
        var withTeacher = data.WithTeacher(teacherModel, batchSize: 10);
        var expectedLogits = teacherModel.Predict(features);
        Check(withTeacher.TargetCount == 6 && withTeacher.TargetNames[0] == "teacher0" && withTeacher.Count == 64, $"targets: {string.Join(", ", withTeacher.TargetNames)}");
        for (int i = 0; i < 64; i++)
        {
            AssertClose([.. Enumerable.Range(0, 3).Select(c => expectedLogits[i, c]), .. data.GetTargets(i).ToArray()], withTeacher.GetTargets(i).ToArray(), 1e-5f, $"sample {i}'s targets");
        }

        using var studentModel = new Sequential { new Linear(5, 3, device: device, random: new Random(2)) };
        using var trainer = new Trainer(studentModel, Losses.Distillation(2f, 1f), p => new Adam(p, 0.05f));
        var history = trainer.Fit(new DataLoader(withTeacher, 16, shuffle: true, device: device, seed: 1), epochs: 40);
        Check(history.Epochs[^1].Loss < 0.2 * history.Epochs[0].Loss, $"the student learns the teacher: {history.Epochs[0].Loss:F4} → {history.Epochs[^1].Loss:F4}");
    }

    // A tiny teacher for the student of TinyChatSpec: the same vocabulary, wider and with other weights.
    private static DecoderSpec TinyTeacherSpec() => TinyChatSpec() with { Dim = 48, Heads = 3, KvHeads = 1, HeadDim = 16, FfDim = 96 };

    private static void VocabularyCheck(Device device)
    {
        _ = device;
        string studentFolder = WriteChatModel(TinyChatSpec());
        try
        {
            // The same tokenizer padded to two widths matches; the fingerprints agree.
            var a = BpeTokenizer.Load(Path.Combine(studentFolder, "tokenizer.json"));
            var b = BpeTokenizer.Load(Path.Combine(studentFolder, "tokenizer.json"));
            b.PadVocabulary(320);
            DistillationTeacher.CheckVocabulary(a, b);
            Check(DistillationTeacher.VocabularyFingerprint(a) == DistillationTeacher.VocabularyFingerprint(b), "padding leaves the fingerprint");

            // The teacher's prompts: up to the last user message of a row.
            var prompts = new[]
            {
                new JsonObject { ["question"] = "What is two and two?", ["answer"] = "four" },
                new JsonObject { ["prompt"] = "Say hello." },
                JsonNode.Parse("""{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "and now?"}, {"role": "assistant", "content": "bye"}]}""")!.AsObject(),
                new JsonObject { ["text"] = "no prompt here" },
            };
            Check(TeacherData.Prompt(prompts[0])!["messages"]!.AsArray().Count == 1 && TeacherData.Prompt(prompts[2])!["messages"]!.AsArray().Count == 3
                  && TeacherData.Prompt(prompts[3]) is null && (string?)TeacherData.Prompt(prompts[1], "Be brief.")!["messages"]![0]!["role"] == "system",
                "prompts: up to the last user message, a prompt column, nothing from plain text, a system message added");
        }
        finally
        {
            Directory.Delete(studentFolder, true);
        }
    }
}
