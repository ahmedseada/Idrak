// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.LanguageModels;

/// <summary>
/// The teacher of knowledge distillation for <see cref="FineTuner"/>: the distribution over the next token at every
/// trained position, which <see cref="FineTuningLossInput.TeacherDivergence"/> compares with the student's (and
/// <see cref="FineTuningLosses.Distillation"/> trains on). Set it as <see cref="FineTuningOptions.Teacher"/>. Two kinds:
/// <list type="bullet">
/// <item><see cref="FromModel"/>: a loaded model computes its distributions on the fly, in inference mode, batch by batch
/// (any weight format, on any device: the student's or another; only its hidden states at the trained positions are kept
/// between the passes of a step, and the head runs on a chunk of rows at a time).</item>
/// <item><see cref="FromFile"/>: the top-k logits per token stored by <see cref="TeacherLogitsWriter"/>, so the teacher need
/// not be loaded while the student trains. The distribution is the softmax over the stored k tokens alone, renormalized
/// (the tokens outside the top k get probability 0): an approximation that keeps nearly all of the mass for usual k
/// (16 to 64) and a teacher that is confident, and drops the tail the student would otherwise learn.</item>
/// </list>
/// Teacher and student must share a vocabulary (the same token for every id): see <see cref="CheckVocabulary"/>. When they
/// do not, distil through data the teacher writes instead (<see cref="TeacherData"/>).
/// </summary>
public abstract class DistillationTeacher : IDisposable
{
    private protected DistillationTeacher()
    {
    }

    /// <summary>The width of the teacher's distributions (its output head; for stored logits, the width recorded).</summary>
    public abstract int Vocabulary { get; }

    /// <summary>Tokens kept per position (0: the full distribution).</summary>
    public abstract int TopK { get; }

    /// <summary>
    /// A fingerprint of the teacher's vocabulary (<see cref="VocabularyFingerprint"/>), or null when unknown; a student
    /// whose own fingerprint differs does not share it.
    /// </summary>
    public abstract string? Fingerprint { get; }

    /// <summary>The teacher's tokenizer, when it has one at hand (a model's; not stored logits).</summary>
    public virtual ITokenizer? Tokenizer => null;

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

    /// <summary>
    /// Checks that <paramref name="student"/> can learn from this teacher: the same token for every id (see
    /// <see cref="CheckVocabulary"/>, or the fingerprints of stored logits), and an output head at least as wide as the
    /// tokens the two share. Throws <see cref="InvalidOperationException"/> naming the difference when not.
    /// </summary>
    public void Check(PretrainedModel student)
    {
        ArgumentNullException.ThrowIfNull(student);
        var tokenizer = student.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer: its vocabulary cannot be compared with the teacher's.");
        if (Tokenizer is { } own)
        {
            CheckVocabulary(own, tokenizer);
        }
        else if (Fingerprint is { } fingerprint && fingerprint != VocabularyFingerprint(tokenizer))
        {
            throw new InvalidOperationException(Mismatch("the stored teacher logits were written for another vocabulary (their fingerprint differs from the student's)"));
        }
    }

    /// <summary>
    /// Checks that <paramref name="teacher"/> and <paramref name="student"/> share a vocabulary: every id is the same
    /// token in both (ids past a tokenizer's end, such as the padding rows models add for alignment, count as empty, so
    /// the same tokenizer padded to two widths matches). Logit distillation compares two distributions entry by entry,
    /// so it needs this; throws <see cref="InvalidOperationException"/> naming the first id that differs otherwise.
    /// </summary>
    public static void CheckVocabulary(ITokenizer teacher, ITokenizer student)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(student);
        int count = Math.Max(teacher.VocabularySize, student.VocabularySize);
        for (int id = 0; id < count; id++)
        {
            string a = TokenText(teacher, id), b = TokenText(student, id);
            if (!string.Equals(a, b, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(Mismatch(
                    $"token {id} is {Quote(a)} for the teacher and {Quote(b)} for the student ({Tokens(teacher):N0} and {Tokens(student):N0} tokens)"));
            }
        }
    }

    /// <summary>
    /// A short fingerprint of <paramref name="tokenizer"/>'s vocabulary (a 64-bit hash of every token's text, in id order,
    /// trailing empty ids left out): equal for two tokenizers <see cref="CheckVocabulary"/> accepts. Stored with teacher
    /// logits to check the student that reads them.
    /// </summary>
    public static string VocabularyFingerprint(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        int count = Tokens(tokenizer);
        ulong hash = Hash.Start;
        for (int id = 0; id < count; id++)
        {
            hash = Hash.Add(hash, Encoding.UTF8.GetBytes(TokenText(tokenizer, id)));
            hash = Hash.Add(hash, [0xFF]);                                      // a separator no UTF-8 text contains
        }

        return $"{count}-{hash:x16}";
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases what the teacher holds (an open file).</summary>
    protected virtual void Dispose(bool disposing)
    {
    }

    /// <summary>The distributions at the trained positions of <paramref name="sequences"/>, in order (the batch's).</summary>
    internal abstract TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences);

    private static string Mismatch(string detail) =>
        $"The teacher and the student do not share a vocabulary: {detail}. Logit distillation compares their distributions token by token, "
        + "so it needs the same tokenizer; distil through answers the teacher writes instead (sequence-level distillation, TeacherData), which works across vocabularies.";

    private static string Quote(string text) => text.Length == 0 ? "nothing" : $"\"{text}\"";

    // A token's text: the vocabulary entry where the tokenizer exposes it, else its decoded text.
    private static string TokenText(ITokenizer tokenizer, int id) => (uint)id >= (uint)tokenizer.VocabularySize ? ""
        : tokenizer switch
        {
            BpeTokenizer bpe => bpe.TokenOf(id),
            _ => tokenizer.Decode([id]),
        };

    // The tokens up to the last non-empty one (padding ids at the end left out).
    private static int Tokens(ITokenizer tokenizer)
    {
        int count = tokenizer.VocabularySize;
        while (count > 0 && TokenText(tokenizer, count - 1).Length == 0)
        {
            count--;
        }

        return count;
    }
}

/// <summary>The teacher's distributions at the trained positions of one batch.</summary>
internal abstract class TeacherDistributions : IDisposable
{
    /// <summary>
    /// The probabilities at temperature <paramref name="temperature"/> of trained positions start … start + count − 1 as
    /// [count, vocabulary] on <paramref name="device"/>: a teacher wider than <paramref name="vocabulary"/> is cut to it (and
    /// renormalized), a narrower one gets zeros past its end.
    /// </summary>
    public abstract Tensor Probabilities(int start, int count, float temperature, int vocabulary, Device device);

    /// <inheritdoc />
    public virtual void Dispose()
    {
    }

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

    internal override TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences) => new Batch(this, Hidden(sequences));

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
                TeacherDistributions.Top(values.AsSpan(r * width, width), k, ids.AsSpan((r0 + r) * k, k), logits.AsSpan((r0 + r) * k, k));
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
                        Top(values.AsSpan(r * width, width), k, ids.AsSpan(r * k, k), top.AsSpan(r * k, k));
                    }

                    return scope.Keep(Dense(ids, top, k, 0, count, temperature, vocabulary, device));
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

    internal override TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences)
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
            Dense(ids, logits, k, start, count, temperature, vocabulary, device);
    }
}

// FNV-1a, 64 bits: fingerprints and the keys of stored sequences (not for security).
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
