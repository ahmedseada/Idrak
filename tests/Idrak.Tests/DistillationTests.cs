// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak;

// The teacher pattern (knowledge distillation): the per-token divergence and its gradient, the classifier loss, the
// fine-tuning loss against both models' full logits, stored top-k logits, the vocabulary check, teacher-written data,
// and a tiny student that ends closer to its teacher by distillation than by its labels.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DistillationGroup =
    [
        ("distillation: the token divergence (full and top-k teachers, chunked, repeated rows, a temperature) and its gradient match a dense computation", DivergenceFormula),
        ("distillation: the classifier loss matches its formula and gradient, alone and mixed with one-hot or index labels; WithTeacher puts the teacher's logits before the targets", ClassifierDistillation),
        ("distillation: the fine-tuning loss and its gradients through the adapters match both models' full logits (on-the-fly teacher, a temperature, mixed with the labels)", DistillationFineTuning),
        ("distillation: teacher logits stored as top-k round-trip, train as the top-k teacher does (renormalized), and refuse missing sequences and incomplete files", StoredTeacherLogits),
        ("distillation: a teacher with another vocabulary is refused with a clear message; padded vocabularies match; teacher-written data trains across vocabularies", VocabularyCheck),
        ("distillation: a tiny student distilled from a tiny teacher ends closer to it (KL on held-out text) than the same student trained on labels alone", DistillationBeatsLabels),
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

    // The model's logits at every position of a sequence but the last, [length − 1, vocabulary] (no gradients).
    private static float[] SequenceLogits(PretrainedModel model, TrainingSequence sequence)
    {
        using var scope = new TensorScope();
        using var noGrad = Autograd.NoGrad();
        model.Network.Eval();
        var tokens = Tensor.From([.. sequence.Tokens[..^1].Select(t => (float)t)], [1, sequence.Tokens.Length - 1], model.Device);
        return model.Network.Forward(tokens).ToArray();
    }

    private static double[] LogSoftmax(ReadOnlySpan<float> row, double temperature)
    {
        double max = double.NegativeInfinity;
        foreach (float v in row)
        {
            max = Math.Max(max, v / temperature);
        }

        double sum = 0;
        foreach (float v in row)
        {
            sum += Math.Exp(v / temperature - max);
        }

        var result = new double[row.Length];
        for (int i = 0; i < row.Length; i++)
        {
            result[i] = row[i] / temperature - max - Math.Log(sum);
        }

        return result;
    }

    // The mean over the trained tokens of α T² KL(q_T ‖ p_T) − (1 − α) log p(token), from both models' full logits; with
    // `teacherRows` (top-k per trained token) the teacher's distribution is the renormalized softmax over those entries.
    private static double DistillationLoss(PretrainedModel student, PretrainedModel? teacher, IReadOnlyList<TrainingSequence> sequences, double temperature, double alpha,
        Func<TrainingSequence, (int[] Ids, float[] Logits)>? teacherRows = null, int k = 0)
    {
        double total = 0;
        int count = 0;
        foreach (var sequence in sequences)
        {
            float[] s = SequenceLogits(student, sequence);
            float[]? t = teacher is null ? null : SequenceLogits(teacher, sequence);
            var stored = teacherRows?.Invoke(sequence);
            int vocabulary = s.Length / (sequence.Tokens.Length - 1), row = 0;
            for (int i = 0; i + 1 < sequence.Tokens.Length; i++)
            {
                if (!sequence.Trained[i + 1])
                {
                    continue;
                }

                var logp = LogSoftmax(s.AsSpan(i * vocabulary, vocabulary), temperature);
                var q = new double[vocabulary];
                if (stored is { } top)
                {
                    float max = top.Logits.AsSpan(row * k, k).ToArray().Max();
                    double sum = Enumerable.Range(0, k).Sum(j => Math.Exp((top.Logits[row * k + j] - max) / temperature));
                    for (int j = 0; j < k; j++)
                    {
                        q[top.Ids[row * k + j]] += Math.Exp((top.Logits[row * k + j] - max) / temperature) / sum;
                    }
                }
                else
                {
                    var logq = LogSoftmax(t!.AsSpan(i * vocabulary, vocabulary), temperature);
                    q = [.. logq.Select(Math.Exp)];
                }

                double kl = 0;
                for (int v = 0; v < vocabulary; v++)
                {
                    kl += q[v] > 0 ? q[v] * (Math.Log(q[v]) - logp[v]) : 0;
                }

                double label = LogSoftmax(s.AsSpan(i * vocabulary, vocabulary), 1.0)[sequence.Tokens[i + 1]];
                total += alpha * temperature * temperature * kl - (1 - alpha) * label;
                count++;
                row++;
            }
        }

        return total / count;
    }

    private static void DistillationFineTuning(Device device)
    {
        string studentFolder = WriteChatModel(TinyChatSpec()), teacherFolder = WriteChatModel(TinyTeacherSpec());
        try
        {
            var sequences = RandomSequences(10, 81);
            const float T = 2f, Alpha = 0.6f, Rate = 0.5f;
            string[] targets = ["q", "v", "down"];
            using var teacher = PretrainedModel.Load(teacherFolder, new PretrainedOptions { Device = device });

            // The reference: the loss written densely with autograd over the full logits, and its gradients.
            using var reference = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            reference.AddAdapters(4, 8, targets, seed: 6);
            double expectedLoss = DistillationLoss(reference, teacher, sequences, T, Alpha);
            var parameters = reference.Network.TrainableParameters().ToList();
            float[][] before = [.. parameters.Select(p => p.ToArray())];
            int trained = sequences.Sum(s => s.TrainedTokens);
            reference.Network.Train();
            foreach (var sequence in sequences)
            {
                using var scope = new TensorScope();
                int n = sequence.Tokens.Length - 1, vocabulary = reference.Spec.Vocabulary;
                float[] t = SequenceLogits(teacher, sequence);
                var q = new float[n * vocabulary];
                var logq = new float[n * vocabulary];
                var label = new float[n * vocabulary];
                for (int i = 0; i < n; i++)
                {
                    if (sequence.Trained[i + 1])
                    {
                        var lq = LogSoftmax(t.AsSpan(i * vocabulary, vocabulary), T);
                        for (int v = 0; v < vocabulary; v++)
                        {
                            logq[i * vocabulary + v] = (float)lq[v];
                            q[i * vocabulary + v] = (float)Math.Exp(lq[v]);
                        }

                        label[i * vocabulary + sequence.Tokens[i + 1]] = 1f;
                    }
                }

                reference.Network.Train();
                var tokens = Tensor.From([.. sequence.Tokens[..^1].Select(x => (float)x)], [1, n], device);
                var logits = reference.Network.Forward(tokens).Reshape(n, vocabulary);
                var kl = (Tensor.From(q, [n, vocabulary], device) * (Tensor.From(logq, [n, vocabulary], device) - (logits * (1f / T)).LogSoftmax())).Sum();
                var nll = -(Tensor.From(label, [n, vocabulary], device) * logits.LogSoftmax()).Sum();
                (kl * (Alpha * T * T / trained) + nll * ((1f - Alpha) / trained)).Backward();
            }

            float[][] gradients = [.. parameters.Select(p => p.Grad?.ToArray() ?? new float[p.Size])];

            // One plain SGD step of the fine-tuner (no momentum, no clipping): the adapters move by −rate · gradient.
            using var student = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            using var distilled = DistillationTeachers.FromModel(teacher);
            var losses = new List<float>();
            FineTuner.Train(student, sequences, null, new FineTuningOptions
            {
                Rank = 4, Alpha = 8, Targets = targets, Seed = 6, BatchTokens = 4096, Epochs = 1, MaxGradientNorm = 1e9f,
                Optimizer = ps => new Sgd(ps, Rate), Scheduler = FineTuningSchedules.Constant(0f),
                Teacher = distilled, Loss = FineTuningLosses.Distillation(T, Alpha),
            }, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)));
            Check(losses.Count == 1, $"one step, {losses.Count}");
            float tolerance = device.Type == DeviceType.Cpu ? 1e-3f : 1e-2f;
            AssertClose([(float)expectedLoss], [losses[0]], tolerance, "the step's distillation loss against both models' full logits");
            var after = student.Network.TrainableParameters().ToList();
            Check(after.Count == parameters.Count, "the same adapters");
            CloseByNorm([.. gradients.SelectMany(g => g)], [.. before.Zip(after, (b, a) => b.Zip(a.ToArray(), (x, y) => (x - y) / Rate)).SelectMany(g => g)],
                device.Type == DeviceType.Cpu ? 2e-3f : 2e-2f, "the adapters' gradients through the distillation loss");

            // Without a loss the teacher implies FineTuningLosses.Distillation with its defaults (T = 1, α = 0.5).
            using var plain = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            var defaultLosses = new List<float>();
            FineTuner.Train(plain, sequences, null, new FineTuningOptions { Rank = 4, Alpha = 8, Targets = targets, Seed = 6, Epochs = 1, Teacher = distilled },
                progress: new SynchronousProgress<FineTuningProgress>(p => defaultLosses.Add(p.Loss)));
            using var fresh = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            AssertClose([(float)DistillationLoss(fresh, teacher, sequences, 1, 0.5)], [defaultLosses[0]], tolerance, "a teacher without a loss: the default distillation");

            // The teacher on another device than the student (here the CPU): the same step.
            if (device.Type != DeviceType.Cpu)
            {
                using var cpuTeacher = PretrainedModel.Load(teacherFolder, new PretrainedOptions { Device = Device.Cpu });
                using var elsewhere = DistillationTeachers.FromModel(cpuTeacher);
                using var other = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
                var otherLosses = new List<float>();
                FineTuner.Train(other, sequences, null, new FineTuningOptions
                {
                    Rank = 4, Alpha = 8, Targets = targets, Seed = 6, Epochs = 1, Teacher = elsewhere, Loss = FineTuningLosses.Distillation(T, Alpha),
                }, progress: new SynchronousProgress<FineTuningProgress>(p => otherLosses.Add(p.Loss)));
                AssertClose([(float)expectedLoss], [otherLosses[0]], tolerance, "a teacher on the CPU for a student on the device");
            }
        }
        finally
        {
            Directory.Delete(studentFolder, true);
            Directory.Delete(teacherFolder, true);
        }
    }

    private static void StoredTeacherLogits(Device device)
    {
        string studentFolder = WriteChatModel(TinyChatSpec()), teacherFolder = WriteChatModel(TinyTeacherSpec());
        string path = Path.Combine(TempFolder(), "teacher.topk");
        try
        {
            var sequences = RandomSequences(8, 83);
            const int K = 8;
            using var teacher = PretrainedModel.Load(teacherFolder, new PretrainedOptions { Device = device });
            int written = TeacherLogitsWriter.Write(path, teacher, [.. sequences, sequences[0]], K, batchTokens: 160);
            Check(written == sequences.Count && TeacherLogitsFile.IsTeacherLogits(path), $"each sequence stored once: {written}");
            long bytes = new FileInfo(path).Length;
            Check(bytes < 80 + sequences.Sum(s => 12 + 6L * K * s.TrainedTokens + 16), $"about 6 bytes per kept token: {bytes} bytes");

            // The stored entries are the teacher's top k, largest first, relative to the largest (16-bit floats).
            using (var file = TeacherLogitsFile.Open(path))
            {
                Check(file.Count == sequences.Count && file.TopK == K && file.Vocabulary == teacher.Spec.Vocabulary
                      && file.VocabularyFingerprint == DistillationTeacher.VocabularyFingerprint(teacher.Tokenizer!), "the file's header");
                foreach (var sequence in sequences)
                {
                    var (ids, logits) = file.Read(sequence) ?? throw new Exception("a stored sequence is missing");
                    float[] full = SequenceLogits(teacher, sequence);
                    int vocabulary = teacher.Spec.Vocabulary, row = 0;
                    for (int i = 0; i + 1 < sequence.Tokens.Length; i++)
                    {
                        if (!sequence.Trained[i + 1])
                        {
                            continue;
                        }

                        var values = full.AsSpan(i * vocabulary, vocabulary).ToArray();
                        int[] top = [.. Enumerable.Range(0, vocabulary).OrderByDescending(v => values[v]).Take(K)];
                        Check(ids.AsSpan(row * K, K).SequenceEqual(top), $"top-{K} ids at position {i}");
                        AssertClose([.. top.Select(v => values[v] - values[top[0]])], logits.AsSpan(row * K, K).ToArray(), 2e-3f, $"stored logits at position {i}");
                        row++;
                    }
                }

                Check(file.Read(RandomSequences(1, 84)[0]) is null, "a sequence not written is not found");
            }

            // Training from the file: the loss of the renormalized top k, as the top-k model teacher gives.
            using var student = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            using var reader = TeacherLogitsFile.Open(path);
            double expected = DistillationLoss(student, null, sequences, 1.5, 0.7, s => reader.Read(s)!.Value, K);
            float First(DistillationTeacher source)
            {
                using var model = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
                var losses = new List<float>();
                FineTuner.Train(model, sequences, null, new FineTuningOptions
                {
                    Rank = 4, Alpha = 8, Epochs = 1, BatchTokens = 4096, Teacher = source, Loss = FineTuningLosses.Distillation(1.5f, 0.7f),
                }, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)));
                return losses[0];
            }

            float tolerance = device.Type == DeviceType.Cpu ? 1e-3f : 1e-2f;
            using (var stored = DistillationTeachers.FromFile(path))
            {
                Check(stored.TopK == K && stored.Vocabulary == teacher.Spec.Vocabulary, "the stored teacher's sizes");
                AssertClose([(float)expected], [First(stored)], tolerance, "the stored teacher's loss against the renormalized top-k formula");
            }

            using (var topK = DistillationTeachers.FromModel(teacher, topK: K))
            {
                AssertClose([(float)expected], [First(topK)], Math.Max(tolerance, 3e-3f), "the top-k model teacher gives what its stored logits give");
            }

            // A sequence the file lacks, and a file cut short, are refused with their reasons.
            using (var stored = DistillationTeachers.FromFile(path))
            using (var model = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device }))
            {
                try
                {
                    FineTuner.Train(model, RandomSequences(2, 85), null, new FineTuningOptions { Rank = 4, Alpha = 8, Epochs = 1, Teacher = stored });
                    Check(false, "unknown sequences trained");
                }
                catch (InvalidOperationException ex)
                {
                    Check(ex.Message.Contains("not in the teacher logits file", StringComparison.Ordinal), ex.Message);
                }
            }

            string cut = path + ".cut";
            File.WriteAllBytes(cut, File.ReadAllBytes(path)[..^20]);
            Check(Refused(() => TeacherLogitsFile.Open(cut).Dispose()), "an incomplete file is refused");
            Check(!TeacherLogitsFile.IsTeacherLogits(studentFolder + "/config.json"), "a JSON file is not teacher logits");
        }
        finally
        {
            Directory.Delete(studentFolder, true);
            Directory.Delete(teacherFolder, true);
            Directory.Delete(Path.GetDirectoryName(path)!, true);
        }
    }

    private static void VocabularyCheck(Device device)
    {
        string studentFolder = WriteChatModel(TinyChatSpec()), teacherFolder = WriteChatModel(TinyTeacherSpec());
        string folder = TempFolder();
        try
        {
            // The teacher's tokenizer with two tokens swapped: the same size, another vocabulary.
            string tokenizerPath = Path.Combine(teacherFolder, "tokenizer.json");
            var json = JsonNode.Parse(File.ReadAllText(tokenizerPath))!.AsObject();
            var vocab = json["model"]!["vocab"]!.AsObject();
            (vocab["a"], vocab["b"]) = ((int)vocab["b"]!, (int)vocab["a"]!);
            File.WriteAllText(tokenizerPath, json.ToJsonString());

            var sequences = RandomSequences(4, 86);
            using var teacher = PretrainedModel.Load(teacherFolder, new PretrainedOptions { Device = device });
            using var student = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device });
            using (var source = DistillationTeachers.FromModel(teacher))
            {
                try
                {
                    FineTuner.Train(student, sequences, null, new FineTuningOptions { Rank = 4, Alpha = 8, Epochs = 1, Teacher = source });
                    Check(false, "a teacher with another vocabulary trained");
                }
                catch (InvalidOperationException ex)
                {
                    Check(ex.Message.Contains("do not share a vocabulary", StringComparison.Ordinal) && ex.Message.Contains("token 97 is \"b\" for the teacher and \"a\" for the student", StringComparison.Ordinal)
                          && ex.Message.Contains("TeacherData", StringComparison.Ordinal), ex.Message);
                }
            }

            // Stored logits carry the teacher's fingerprint: refused too.
            string path = Path.Combine(folder, "teacher.topk");
            TeacherLogitsWriter.Write(path, teacher, sequences, 4);
            using (var stored = DistillationTeachers.FromFile(path))
            {
                Check(Refused(() => stored.Check(student.Tokenizer!)), "stored logits of another vocabulary are refused");
            }

            // The same tokenizer padded to two widths matches; the fingerprints agree.
            var a = BpeTokenizer.Load(Path.Combine(studentFolder, "tokenizer.json"));
            var b = BpeTokenizer.Load(Path.Combine(studentFolder, "tokenizer.json"));
            b.PadVocabulary(320);
            DistillationTeacher.CheckVocabulary(a, b);
            Check(DistillationTeacher.VocabularyFingerprint(a) == DistillationTeacher.VocabularyFingerprint(b), "padding leaves the fingerprint");
            Check(DistillationTeacher.VocabularyFingerprint(a) != DistillationTeacher.VocabularyFingerprint(teacher.Tokenizer!), "another vocabulary, another fingerprint");

            // Sequence-level distillation needs no shared vocabulary: the teacher writes answers, the student trains on them.
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
            var rows = TeacherData.Generate(teacher.CreateChat(), prompts, new TeacherDataOptions { MaxNewTokens = 6, KeepCutOff = true, BatchSize = 2 }).ToList();
            Check(rows.Count == 3 && rows.All(r => (string?)r["messages"]!.AsArray()[^1]!["role"] == "assistant"), $"teacher rows: {string.Join("\n", rows.Select(r => r.ToJsonString()))}");
            Check(TeacherData.Generate(teacher.CreateChat(), prompts, new TeacherDataOptions { MaxNewTokens = 2 }).All(r => false), "answers cut off are left out unless kept");
            var encoder = new ChatTranscriptEncoder(student.JinjaTemplate!, student.Tokenizer!);
            var encoded = rows.SelectMany(r => encoder.EncodeRow(r, 200)).ToList();
            Check(encoded.Count == 3 && encoded.All(s => s.TrainedTokens > 0), "the teacher's answers encode for the student");
            var losses = FineTuner.Train(student, encoded, encoded, new FineTuningOptions { Rank = 4, Alpha = 8, Epochs = 2, LearningRate = 1e-2f, EvaluateEvery = 1 });
            Check(losses.Count > 0 && losses.All(float.IsFinite), "the student trains on the teacher's answers");
        }
        finally
        {
            Directory.Delete(studentFolder, true);
            Directory.Delete(teacherFolder, true);
            Directory.Delete(folder, true);
        }
    }

    // A tiny teacher made confident (its final norm's gain scaled up, so its distributions are peaked), with text sampled
    // from it; the student learns that text from the labels alone, or from the teacher's distributions.
    private static void DistillationBeatsLabels(Device device)
    {
        string studentFolder = WriteChatModel(TinyChatSpec()), teacherFolder = WriteChatModel(TinyTeacherSpec());
        try
        {
            static PretrainedModel Teacher(string folder, Device device)
            {
                var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var norm = (RMSNorm)model.Network.ToList()[^2];
                norm.Gain.Scale(3f);
                return model;
            }

            // Text sampled from the teacher (on the CPU, the same on every device): 48 sequences, 32 held out.
            List<TrainingSequence> train, held;
            using (var sampler = Teacher(teacherFolder, Device.Cpu))
            {
                var random = new Random(87);
                var sampled = new List<TrainingSequence>();
                for (int s = 0; s < 80; s++)
                {
                    var tokens = new List<int> { random.Next(256) };
                    while (tokens.Count < 40)
                    {
                        float[] logits = SequenceLogits(sampler, new TrainingSequence([.. tokens, 0], new bool[tokens.Count + 1]));
                        var last = LogSoftmax(logits.AsSpan(logits.Length - 260, 260), 1.0);
                        double u = random.NextDouble(), acc = 0;
                        int next = 0;
                        for (; next < 259; next++)
                        {
                            acc += Math.Exp(last[next]);
                            if (u < acc)
                            {
                                break;
                            }
                        }

                        tokens.Add(next);
                    }

                    sampled.Add(new TrainingSequence([.. tokens], [false, .. Enumerable.Repeat(true, tokens.Count - 1)]));
                }

                train = sampled[..48];
                held = sampled[48..];
            }

            using var teacher = Teacher(teacherFolder, device);
            double Divergence(PretrainedModel student) => DistillationLoss(student, teacher, held, 1, 1);
            var options = new FineTuningOptions { Rank = 8, Alpha = 16, LearningRate = 2e-2f, Epochs = 16, BatchTokens = 512, WarmupFraction = 0f, Seed = 5, Targets = ["q", "k", "v", "o", "gate", "up", "down"] };
            double before, labels, distilled;
            using (var student = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device }))
            {
                before = Divergence(student);
                FineTuner.Train(student, train, null, options);
                labels = Divergence(student);
            }

            using (var student = PretrainedModel.Load(studentFolder, new PretrainedOptions { Device = device }))
            using (var source = DistillationTeachers.FromModel(teacher))
            {
                FineTuner.Train(student, train, null, options with { Teacher = source, Loss = FineTuningLosses.Distillation(1f, 1f) });
                distilled = Divergence(student);
            }

            Console.WriteLine($"    held-out KL(teacher ‖ student): {before:F4} before, {labels:F4} trained on labels, {distilled:F4} distilled");
            Check(labels < before, $"labels sampled from the teacher bring the student closer: {before:F4} → {labels:F4}");
            Check(distilled < labels && before - distilled >= 1.25 * (before - labels),
                $"distillation brings it closer still: {distilled:F4} against {labels:F4} from labels alone (gains {before - distilled:F4} and {before - labels:F4})");
        }
        finally
        {
            Directory.Delete(studentFolder, true);
            Directory.Delete(teacherFolder, true);
        }
    }

    // idrak distill, in-process: the three ways, and the refusals.
    private static void CliDistill(Device device)
    {
        string student = WriteChatModel(TinyChatSpec()), teacher = WriteChatModel(TinyTeacherSpec()), other = WriteChatModel(TinyTeacherSpec());
        string folder = TempFolder();
        try
        {
            string tokenizerPath = Path.Combine(other, "tokenizer.json");
            var tokenizer = JsonNode.Parse(File.ReadAllText(tokenizerPath))!.AsObject();
            var vocab = tokenizer["model"]!["vocab"]!.AsObject();
            (vocab["a"], vocab["b"]) = ((int)vocab["b"]!, (int)vocab["a"]!);
            File.WriteAllText(tokenizerPath, tokenizer.ToJsonString());

            string d = device.ToString();
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, Enumerable.Range(0, 10).Select(i => new JsonObject { ["question"] = $"What is {i} plus {i}?", ["answer"] = $"{i + i}" }.ToJsonString()));
            string[] tune = ["--rank", "4", "--lora-alpha", "8", "--max-length", "64", "--batch-tokens", "256", "--epochs", "1", "--eval-fraction", "0", "-d", d, "--cache", folder];

            // On the fly: both models loaded, the teacher's full distributions.
            string live = Path.Combine(folder, "live");
            var distilled = TrainCliJson(["distill", "--teacher", teacher, "--student", student, "--data", data, "-o", live, "--temperature", "2", "--alpha", "0.7", .. tune]);
            Check((string?)distilled["mode"] == "logits" && (int?)distilled["exitCode"] == 0 && (double?)distilled["alpha"] == 0.7 && (int?)distilled["topK"] == 0
                  && File.Exists(Path.Combine(live, "idrak-tuning.json")) && File.Exists(Path.Combine(live, "adapter_model.safetensors")), $"distill on the fly: {distilled}");
            Check(distilled["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("distilling: temperature 2, alpha 0.7, the teacher's full distributions", StringComparison.Ordinal)),
                $"distill says how: {distilled}");

            // Precomputed top-k logits, then training from the file (the teacher not loaded).
            string logits = Path.Combine(folder, "teacher.topk");
            var written = TrainCliJson(["distill", "--teacher", teacher, "--student", student, "--data", data, "--precompute", logits, "--top-k", "8", .. tune]);
            Check((string?)written["mode"] == "precompute" && (int?)written["sequences"] == 10 && TeacherLogitsFile.IsTeacherLogits(logits), $"distill --precompute: {written}");
            string fromFile = Path.Combine(folder, "stored");
            var stored = TrainCliJson(["distill", "--teacher", logits, "--student", student, "--data", data, "-o", fromFile, .. tune]);
            Check((string?)stored["mode"] == "stored logits" && (int?)stored["topK"] == 8 && File.Exists(Path.Combine(fromFile, "idrak-tuning.json")), $"distill from stored logits: {stored}");

            // Another vocabulary: refused for logits, fine for teacher-written answers.
            var refused = TrainCli(["distill", "--teacher", other, "--student", student, "--data", data, "-o", Path.Combine(folder, "refused"), .. tune]);
            Check(refused.Code == 1 && refused.Err.Contains("do not share a vocabulary", StringComparison.Ordinal), $"a vocabulary mismatch is refused: {refused.Code}\n{refused.Out}\n{refused.Err}");
            string answers = Path.Combine(folder, "answers");
            var generated = TrainCliJson(["distill", "--teacher", other, "--student", student, "--data", data, "-o", answers, "--generate", "--max-new", "6", "--keep-cut-off", .. tune]);
            Check((string?)generated["mode"] == "generate" && (int?)generated["answers"] == 10 && File.ReadAllLines(Path.Combine(answers, "teacher-data.jsonl")).Length == 10
                  && File.Exists(Path.Combine(answers, "idrak-tuning.json")), $"distill --generate: {generated}");

            // Usage errors.
            Check(TrainCli("distill", "--teacher", teacher, "--data", data, "-o", live).Code == 2, "no student: a usage error");
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--generate", "--precompute", logits).Code == 2, "--generate with --precompute: a usage error");
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "-o", live, "--alpha", "2").Code == 2, "--alpha above 1: a usage error");
            Check(TrainCli("distill", "--teacher", logits, "--student", student, "--data", data, "--generate", "-o", live).Code == 2, "stored logits cannot write answers");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
            Directory.Delete(other, true);
        }
    }
}
