// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

/// <summary>
/// The built-in teachers of knowledge distillation (<see cref="DistillationTeacher"/>, Idrak.Abstraction) for
/// <see cref="FineTuner"/>: the distribution over the next token at every trained position, which
/// <see cref="FineTuningLossInput.TeacherDivergence"/> compares with the student's (and
/// <see cref="FineTuningLosses.Distillation"/> trains on). Set one as <see cref="FineTuningOptions.Teacher"/>. Two kinds:
/// <list type="bullet">
/// <item><see cref="FromModel"/>: a loaded model computes its distributions on the fly, in inference mode, batch by batch
/// (any weight format, on any device: the student's or another; only its hidden states at the trained positions are kept
/// between the passes of a step, and the head runs on a chunk of rows at a time).</item>
/// <item><see cref="FromFile"/>: the top-k logits per token stored by <see cref="TeacherLogitsWriter"/>, so the teacher need
/// not be loaded while the student trains. The distribution is the softmax over the stored k tokens alone, renormalized
/// (the tokens outside the top k get probability 0): an approximation that keeps nearly all of the mass for usual k
/// (16 to 64) and a teacher that is confident, and drops the tail the student would otherwise learn.</item>
/// </list>
/// Teacher and student must share a vocabulary (the same token for every id): see
/// <see cref="DistillationTeacher.CheckVocabulary"/>. When they do not, distil through data the teacher writes instead
/// (<see cref="TeacherData"/>).
/// </summary>
public static class DistillationTeachers
{
    /// <summary>
    /// A teacher that runs <paramref name="teacher"/> on each batch's sequences: in inference mode (dropout off, no
    /// gradients), on its own device, padded batches of at most <paramref name="batchTokens"/> positions. With
    /// <paramref name="topK"/> &gt; 0 only the k most likely tokens of each position are kept and renormalized (as stored
    /// logits are); 0 uses the full distribution. The model is not disposed with the teacher.
    /// </summary>
    public static DistillationTeacher FromModel(PretrainedModel teacher, int topK = 0, int batchTokens = 4096)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentOutOfRangeException.ThrowIfNegative(topK);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchTokens);
        return new ModelTeacher(teacher, topK, batchTokens);
    }

    /// <summary>A teacher that reads the top-k logits stored in <paramref name="path"/> (see <see cref="TeacherLogitsWriter"/>); disposing it closes the file.</summary>
    public static DistillationTeacher FromFile(string path) => new StoredTeacher(TeacherLogitsFile.Open(path));
}

// Top-k logits as dense distributions, for the built-in teachers.
internal static class TopKLogits
{
    // Dense probabilities from top-k entries (ids and logits, k per row): the softmax at the temperature over each row's
    // entries below `vocabulary`.
    internal static Tensor Dense(int[] ids, float[] logits, int k, int start, int count, float temperature, int vocabulary, Device device)
    {
        var values = new float[count * vocabulary];
        for (int r = 0; r < count; r++)
        {
            int at = (start + r) * k;
            float max = float.NegativeInfinity;
            for (int j = 0; j < k; j++)
            {
                if (ids[at + j] < vocabulary && logits[at + j] > max)
                {
                    max = logits[at + j];
                }
            }

            double sum = 0;
            for (int j = 0; j < k; j++)
            {
                if (ids[at + j] < vocabulary)
                {
                    sum += Math.Exp((logits[at + j] - max) / temperature);
                }
            }

            for (int j = 0; j < k; j++)
            {
                if (ids[at + j] < vocabulary)
                {
                    values[r * vocabulary + ids[at + j]] += (float)(Math.Exp((logits[at + j] - max) / temperature) / sum);
                }
            }
        }

        return Tensor.From(values, [count, vocabulary], device);
    }

    // The k largest values of a row, largest first, with their indices.
    internal static void Top(ReadOnlySpan<float> row, int k, Span<int> ids, Span<float> values)
    {
        int kept = 0;
        for (int v = 0; v < row.Length; v++)
        {
            float x = row[v];
            if (kept == k && !(x > values[k - 1]))
            {
                continue;
            }

            int at = kept < k ? kept++ : k - 1;
            while (at > 0 && values[at - 1] < x)
            {
                values[at] = values[at - 1];
                ids[at] = ids[at - 1];
                at--;
            }

            values[at] = x;
            ids[at] = v;
        }
    }
}

// A loaded model as the teacher: its hidden states at the batch's trained positions, the head run per chunk on demand.
internal sealed class ModelTeacher(PretrainedModel model, int topK, int batchTokens) : DistillationTeacher
{
    private string? _fingerprint;

    public override int Vocabulary => model.Spec.Vocabulary;

    public override int TopK => topK;

    public override string? Fingerprint => model.Tokenizer is { } t ? _fingerprint ??= VocabularyFingerprint(t) : null;

    public override ITokenizer? Tokenizer => model.Tokenizer;

    public override TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences) => new Batch(this, Hidden(sequences));

    // The teacher's final hidden states at the trained positions of the sequences, in order: [trained tokens, dim] on its
    // device. Padded batches of sequences in their order (right padding: a causal model's positions before it are
    // unchanged), run without packing even inside a packed step of the student.
    private Tensor Hidden(IReadOnlyList<TrainingSequence> sequences)
    {
        var modules = model.Network.ToList();
        if (modules[^1] is not Linear)
        {
            throw new InvalidOperationException("The teacher's network does not end with its output head (a Linear layer).");
        }

        var parts = new List<Tensor>();
        var network = model.Network;
        bool wasTraining = network.IsTraining;
        using var packing = PackedSequences.Suspend();
        using var noGrad = Autograd.NoGrad();
        network.Eval();
        try
        {
            for (int first = 0; first < sequences.Count;)
            {
                int last = first, longest = Math.Max(1, sequences[first].Tokens.Length - 1);
                while (last + 1 < sequences.Count && (last + 2 - first) * Math.Max(longest, sequences[last + 1].Tokens.Length - 1) <= batchTokens)
                {
                    longest = Math.Max(longest, sequences[++last].Tokens.Length - 1);
                }

                if (longest > model.MaxPositions)
                {
                    throw new InvalidOperationException($"A sequence of {longest + 1} tokens is longer than the teacher's context of {model.MaxPositions} positions.");
                }

                int count = last - first + 1;
                var inputs = new float[count * longest];
                var rows = new List<int>();
                for (int s = 0; s < count; s++)
                {
                    var sequence = sequences[first + s];
                    for (int t = 0; t + 1 < sequence.Tokens.Length; t++)
                    {
                        inputs[s * longest + t] = sequence.Tokens[t];
                        if (sequence.Trained[t + 1])
                        {
                            rows.Add(s * longest + t);
                        }
                    }
                }

                if (rows.Count > 0)
                {
                    using var scope = new TensorScope();
                    var hidden = Tensor.From(inputs, [count, longest], model.Device);
                    for (int i = 0; i < modules.Count - 1; i++)
                    {
                        // Each module's intermediate results, and the previous activation, are freed at once.
                        Tensor next;
                        using (var inner = new TensorScope())
                        {
                            next = inner.Keep(modules[i].Forward(hidden));
                        }

                        if (!ReferenceEquals(next, hidden))
                        {
                            hidden.Dispose();
                        }

                        hidden = next;
                    }

                    parts.Add(scope.Keep(Tensor.GatherRows(hidden.Reshape(count * longest, hidden.Shape[^1]), [.. rows])));
                }

                first = last + 1;
            }
        }
        finally
        {
            network.Train(wasTraining);
        }

        if (parts.Count == 0)
        {
            return TensorScope.Untrack(Tensor.Zeros([0, model.Spec.Dim], model.Device));
        }

        if (parts.Count == 1)
        {
            return TensorScope.Untrack(parts[0]);
        }

        var all = Tensor.Concat(parts, 0);
        parts.ForEach(p => p.Dispose());
        return TensorScope.Untrack(all);
    }

    // The head's logits of trained positions start … start + count − 1, [count, teacher vocabulary] on the teacher's device.
    private Tensor Logits(Tensor hidden, int start, int count)
    {
        using var noGrad = Autograd.NoGrad();
        var head = (Linear)model.Network.ToList()[^1];
        return head.Forward(hidden.Narrow(0, start, count));
    }

    // The k largest logits (and their ids) of every trained position, for storing.
    internal (int[] Ids, float[] Logits) TopLogits(IReadOnlyList<TrainingSequence> sequences, int k)
    {
        using var scope = new TensorScope();
        using var hidden = Hidden(sequences);
        int rows = hidden.Shape[0], vocabulary = Vocabulary;
        k = Math.Min(k, vocabulary);
        var ids = new int[rows * k];
        var logits = new float[rows * k];
        int chunk = Math.Max(1, (1 << 22) / Math.Max(1, vocabulary));       // about 16 MB of logits at a time
        for (int r0 = 0; r0 < rows; r0 += chunk)
        {
            int n = Math.Min(chunk, rows - r0);
            using var inner = new TensorScope();
            var values = Logits(hidden, r0, n).ToArray();
            int width = values.Length / n;
            for (int r = 0; r < n; r++)
            {
                TopKLogits.Top(values.AsSpan(r * width, width), k, ids.AsSpan((r0 + r) * k, k), logits.AsSpan((r0 + r) * k, k));
            }
        }

        return (ids, logits);
    }

    private sealed class Batch(ModelTeacher teacher, Tensor hidden) : TeacherDistributions
    {
        public override Tensor Probabilities(int start, int count, float temperature, int vocabulary, Device device)
        {
            Tensor result;
            using (var scope = new TensorScope())
            {
                var logits = teacher.Logits(hidden, start, count);
                int width = logits.Shape[^1];
                if (teacher.TopK > 0 && teacher.TopK < width)
                {
                    var values = logits.ToArray();
                    int k = teacher.TopK;
                    var ids = new int[count * k];
                    var top = new float[count * k];
                    for (int r = 0; r < count; r++)
                    {
                        TopKLogits.Top(values.AsSpan(r * width, width), k, ids.AsSpan(r * k, k), top.AsSpan(r * k, k));
                    }

                    return scope.Keep(TopKLogits.Dense(ids, top, k, 0, count, temperature, vocabulary, device));
                }

                using var noGrad = Autograd.NoGrad();
                var scaled = logits * (1f / temperature);
                if (width > vocabulary)
                {
                    scaled = scaled.Narrow(1, 0, vocabulary);
                }

                var probabilities = scaled.Softmax();
                if (width < vocabulary)
                {
                    probabilities = Tensor.Concat([probabilities, Tensor.Zeros([count, vocabulary - width], probabilities.Device)], 1);
                }

                result = scope.Keep(probabilities.Device == device ? probabilities : probabilities.To(device));
            }

            return result;
        }

        public override void Dispose() => hidden.Dispose();
    }
}

// Stored top-k logits as the teacher.
internal sealed class StoredTeacher(TeacherLogitsFile file) : DistillationTeacher
{
    public override int Vocabulary => file.Vocabulary;

    public override int TopK => file.TopK;

    public override string? Fingerprint => file.VocabularyFingerprint;

    public override TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences)
    {
        int k = file.TopK, total = sequences.Sum(s => s.TrainedTokens);
        var ids = new int[total * k];
        var logits = new float[total * k];
        int at = 0;
        foreach (var sequence in sequences)
        {
            var (i, l) = file.Read(sequence) ?? throw new InvalidOperationException(
                $"A sequence of {sequence.Tokens.Length} tokens is not in the teacher logits file {file.Path}: write it from the same data, maximum length and system prompt as the training.");
            i.CopyTo(ids, at * k);
            l.CopyTo(logits, at * k);
            at += sequence.TrainedTokens;
        }

        return new Batch(ids, logits, k);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            file.Dispose();
        }
    }

    private sealed class Batch(int[] ids, float[] logits, int k) : TeacherDistributions
    {
        public override Tensor Probabilities(int start, int count, float temperature, int vocabulary, Device device) =>
            TopKLogits.Dense(ids, logits, k, start, count, temperature, vocabulary, device);
    }
}

// FNV-1a, 64 bits: the keys of stored sequences (not for security).
internal static class Hash
{
    public const ulong Start = 14695981039346656037UL;

    public static ulong Add(ulong hash, ReadOnlySpan<byte> bytes)
    {
        foreach (byte b in bytes)
        {
            hash = (hash ^ b) * 1099511628211UL;
        }

        return hash;
    }

    // A sequence's key: its tokens and which of them train.
    public static ulong Of(TrainingSequence sequence)
    {
        ulong hash = Add(Start, BitConverter.GetBytes(sequence.Tokens.Length));
        hash = Add(hash, System.Runtime.InteropServices.MemoryMarshal.AsBytes(sequence.Tokens.AsSpan()));
        foreach (bool trained in sequence.Trained)
        {
            hash = (hash ^ (trained ? 1UL : 2UL)) * 1099511628211UL;
        }

        return hash;
    }
}
