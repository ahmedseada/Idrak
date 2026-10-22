// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text;
using Idrak.Datasets.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

/// <summary>
/// The built-in teachers of knowledge distillation (<see cref="DistillationTeacher"/>, Idrak.Abstraction) for
/// <see cref="FineTuner"/>: the distribution over the next token at every trained position, which
/// <see cref="FineTuningLossInput.TeacherDivergence(float)"/> compares with the student's (and
/// <see cref="FineTuningLosses.Distillation"/> trains on). Set one as <see cref="FineTuningOptions.Teacher"/>. Two kinds:
/// <list type="bullet">
/// <item><see cref="FromModel(PretrainedModel, int, int)"/>: a loaded model computes its distributions on the fly, in inference mode, batch by batch
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
    public static DistillationTeacher FromModel(PretrainedModel teacher, int topK = 0, int batchTokens = 4096) => FromModel(teacher, topK, batchTokens, TuningImages.None);

    /// <summary>
    /// <see cref="FromModel(PretrainedModel, int, int)"/> for sequences with images too: a vision-language teacher reads
    /// each image itself, prepared as <paramref name="images"/> says (the student's preparation), through its own vision
    /// part (made on first use, disposed with the teacher). Each of the student's image blocks becomes the teacher's run of
    /// image tokens for that image (the text around it, its begin and end tokens included, as the student has it), and the
    /// trained positions follow; a teacher that cuts an image into another number of blocks is refused.
    /// </summary>
    public static DistillationTeacher FromModel(PretrainedModel teacher, int topK, int batchTokens, TuningImages images)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentOutOfRangeException.ThrowIfNegative(topK);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchTokens);
        return new ModelTeacher(teacher, topK, batchTokens, images);
    }

    /// <summary>A teacher that reads the top-k logits stored in <paramref name="path"/> (see <see cref="TeacherLogitsWriter"/>); disposing it closes the file.</summary>
    public static DistillationTeacher FromFile(string path) => new StoredTeacher(TeacherLogitsFile.Open(path));

    /// <summary>
    /// Checks that <paramref name="teacher"/> can teach <paramref name="student"/> on-policy
    /// (<see cref="FineTuningLosses.OnPolicyDistillation"/>): it must compute its distributions on sequences it has never
    /// seen (the student's own samples), so a live teacher (<see cref="FromModel(PretrainedModel, int, int)"/>, or one of
    /// one's own), not stored logits (<see cref="FromFile"/>, which hold only the sequences they were written for); and it
    /// must share the student's vocabulary (<see cref="DistillationTeacher.Check"/>).
    /// </summary>
    /// <exception cref="NotSupportedException">A teacher file.</exception>
    /// <exception cref="InvalidOperationException">The vocabularies differ: the message names <c>idrak distill --generate</c> (answers the teacher writes, which cross tokenizers).</exception>
    public static void CheckOnPolicy(DistillationTeacher teacher, ITokenizer student)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(student);
        if (teacher is StoredTeacher)
        {
            throw new NotSupportedException("On-policy distillation needs a live teacher (DistillationTeachers.FromModel): a teacher file holds the logits of the "
                                            + "sequences it was written for, and on-policy distillation trains on the student's own new samples, which no file covers.");
        }

        try
        {
            teacher.Check(student);
        }
        catch (InvalidOperationException e)
        {
            throw new InvalidOperationException($"On-policy distillation compares the teacher's and the student's distributions token by token on the student's samples, "
                                                + $"so they must share a vocabulary: {e.Message} Across tokenizers, train on answers the teacher writes (idrak distill --generate).", e);
        }
    }
}

// Top-k logits as dense distributions, for the built-in teachers.
internal static class TopKLogits
{
    // Dense probabilities from top-k entries (ids and logits, k per row): the softmax at the temperature over each row's
    // entries below `vocabulary`.
    internal static Tensor Dense(int[] ids, float[] logits, int k, int start, int count, float temperature, int vocabulary, Device device)
    {
        // The rows are built in a pooled buffer (cleared: every entry not in a row's top k is 0), uploaded, and returned in finally.
        int size = count * vocabulary;
        float[] values = ArrayPool<float>.Shared.Rent(Math.Max(size, 1));
        try
        {
            values.AsSpan(0, size).Clear();
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

            return Tensor.From(values.AsSpan(0, size), [count, vocabulary], device);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
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
internal sealed class ModelTeacher(PretrainedModel model, int topK, int batchTokens, TuningImages images) : DistillationTeacher
{
    private string? _fingerprint;
    private Linear? _head;
    private TuningVision? _vision;
    private readonly Dictionary<string, ISoftPartSource> _parts = new(StringComparer.Ordinal);

    // The teacher's own vision side, made when the first sequence with images comes.
    private TuningVision Vision() => _vision ??= model.Vision is null
        ? throw new NotSupportedException("The sequences hold images and the teacher reads none (it has no vision part): distil from a vision-language teacher.")
        : TuningVision.Create(model, images);

    // The teacher's own source of a kind of soft part (its audio part), opened when the first sequence with one comes.
    private ISoftPartSource Part(string kind)
    {
        if (!_parts.TryGetValue(kind, out var source))
        {
            source = model.Parts.Contains(kind)
                ? model.OpenPart(kind)
                : throw new NotSupportedException($"The sequences hold '{kind}' parts and the teacher reads none (it has no such part): distil from a teacher that has one.");
            _parts[kind] = source;
        }

        return source;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _vision?.Dispose();
            _vision = null;
            foreach (var source in _parts.Values)
            {
                (source as IDisposable)?.Dispose();
            }

            _parts.Clear();
        }

        base.Dispose(disposing);
    }

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
                if (sequences[first].Images.Count > 0 || sequences[first].Parts.Count > 0)
                {
                    // A sequence with images (or other soft parts) alone: the teacher's soft tokens in place of the student's, its features in them.
                    if (WithImages(sequences[first], modules) is { } rowsOfImages)
                    {
                        parts.Add(rowsOfImages);
                    }

                    first++;
                    continue;
                }

                int last = first, longest = Math.Max(1, sequences[first].Tokens.Length - 1);
                while (last + 1 < sequences.Count && sequences[last + 1].Images.Count == 0 && sequences[last + 1].Parts.Count == 0
                       && (last + 2 - first) * Math.Max(longest, sequences[last + 1].Tokens.Length - 1) <= batchTokens)
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

    // The teacher's final hidden states at the trained positions of one sequence with images or other soft parts
    // (untracked), or null when none of its positions trains. Each run of the student's soft tokens becomes the teacher's
    // own (its soft token, as many as its family reads the image or part as), its features in them.
    private Tensor? WithImages(TrainingSequence sequence, List<Module> modules)
    {
        var vision = sequence.Images.Count > 0 ? Vision() : null;
        var tokens = new List<int>(sequence.Tokens.Length);
        var trained = new List<bool>(sequence.Tokens.Length);
        var teacherImages = new List<TrainingImage>();
        // Runs of soft tokens in prompt order: an image's block (Part -1) or a soft part's (Part its index).
        var runs = sequence.Images.SelectMany((image, k) => image.Blocks.Select((b, j) => (Image: k, Part: -1, Block: j, b.Position, b.Tokens)))
            .Concat(sequence.Parts.SelectMany((part, k) => part.Blocks.Select((b, j) => (Image: -1, Part: k, Block: j, b.Position, b.Tokens))))
            .OrderBy(r => r.Position).ToList();
        var layouts = sequence.Images.Select(image => vision!.Blocks(image.Image, vision.Images)).ToList();
        for (int k = 0; k < sequence.Images.Count; k++)
        {
            if (layouts[k].Count != sequence.Images[k].Blocks.Count)
            {
                throw new NotSupportedException($"The teacher cuts an image into {layouts[k].Count} blocks, the student into {sequence.Images[k].Blocks.Count}: "
                    + "distil between families that read an image as the same number of blocks.");
            }
        }

        var partSources = sequence.Parts.Select(p => Part(p.Kind)).ToList();
        var partLayouts = sequence.Parts.Select((p, k) => partSources[k].Blocks(p.Part)).ToList();
        for (int k = 0; k < sequence.Parts.Count; k++)
        {
            if (partLayouts[k].Count != sequence.Parts[k].Blocks.Count)
            {
                throw new NotSupportedException($"The teacher reads a '{sequence.Parts[k].Kind}' part as {partLayouts[k].Count} blocks, the student as {sequence.Parts[k].Blocks.Count}: "
                    + "distil between families that read it as the same number of blocks.");
            }
        }

        var blocks = sequence.Images.Select(_ => new List<(int, int)>()).ToList();
        var partBlocks = sequence.Parts.Select(_ => new List<(int Position, int Tokens)>()).ToList();
        int cursor = 0;
        foreach (var run in runs)
        {
            for (; cursor < run.Position; cursor++)
            {
                tokens.Add(sequence.Tokens[cursor]);
                trained.Add(sequence.Trained[cursor]);
            }

            int count, soft;
            if (run.Image >= 0)
            {
                count = layouts[run.Image][run.Block].Tokens;
                soft = vision!.Format.ImageToken;
                blocks[run.Image].Add((tokens.Count, count));
            }
            else
            {
                count = partLayouts[run.Part][run.Block].Tokens;
                soft = partSources[run.Part].Format.ImageToken;
                partBlocks[run.Part].Add((tokens.Count, count));
            }

            for (int i = 0; i < count; i++)
            {
                tokens.Add(soft);
                trained.Add(false);
            }

            cursor = run.Position + run.Tokens;
        }

        for (; cursor < sequence.Tokens.Length; cursor++)
        {
            tokens.Add(sequence.Tokens[cursor]);
            trained.Add(sequence.Trained[cursor]);
        }

        if (tokens.Count - 1 > model.MaxPositions)
        {
            throw new InvalidOperationException($"A sequence of {tokens.Count} tokens with the teacher's image tokens is longer than its context of {model.MaxPositions} positions.");
        }

        for (int k = 0; k < sequence.Images.Count; k++)
        {
            teacherImages.Add(new TrainingImage(sequence.Images[k].Image, vision!.Images, blocks[k])
            {
                AxisPositions = layouts[k].Select(l => vision.Encoder.AxisPositions(l)).ToList() is var axes && axes.Any(a => a is not null) ? axes : null,
            });
        }

        var rows = Enumerable.Range(0, tokens.Count - 1).Where(t => trained[t + 1]).ToArray();
        if (rows.Length == 0)
        {
            return null;
        }

        using var scope = new TensorScope();
        var features = vision is null ? [] : vision.Features(teacherImages);
        var prompt = new List<PromptImage>();
        IImageAttentionRule? attention = vision?.Attention;
        for (int k = 0; k < sequence.Parts.Count; k++)
        {
            var source = partSources[k];
            if (attention is not null && attention.Name != source.Attention.Name)
            {
                throw new NotSupportedException($"The teacher's '{source.Kind}' parts attend by '{source.Attention.Name}' and its other soft tokens by '{attention.Name}': one rule per sequence.");
            }

            attention ??= source.Attention;
            var encoded = source.Encode([sequence.Parts[k].Part]);
            for (int b = 0; b < partBlocks[k].Count; b++)
            {
                prompt.Add(new PromptImage(partBlocks[k][b].Position, encoded[b].Features));
            }
        }

        for (int k = 0; k < teacherImages.Count; k++)
        {
            int at = 0, b = 0;
            foreach (var (position, count) in teacherImages[k].Blocks)
            {
                var block = at == 0 && count == features[k].Shape[0] ? features[k] : features[k].Narrow(0, at, count);
                var axes = teacherImages[k].AxisPositions?[b];
                prompt.Add(new PromptImage(position, block) { Positions = axes is null ? null : Tensor.From(axes, [axes.Length / count, count], Device.Cpu) });
                at += count;
                b++;
            }
        }

        prompt.Sort((x, y) => x.Position.CompareTo(y.Position));
        var ids = Tensor.From(tokens.Take(tokens.Count - 1).Select(t => (float)t).ToArray(), [1, tokens.Count - 1], model.Device);
        using var images = model.Network.Begin(ids, prompt, attention!);
        var hidden = ids;
        for (int i = 0; i < modules.Count - 1; i++)
        {
            if (i == images.FirstBlock)
            {
                hidden = images.Substitute(hidden);
            }

            hidden = modules[i].Forward(hidden);
        }

        return TensorScope.Untrack(scope.Keep(Tensor.GatherRows(hidden.Reshape(tokens.Count - 1, hidden.Shape[^1]), rows)));
    }

    // The head's logits of trained positions start … start + count − 1, [count, teacher vocabulary] on the teacher's device.
    private Tensor Logits(Tensor hidden, int start, int count)
    {
        using var noGrad = Autograd.NoGrad();
        _head ??= (Linear)model.Network.ToList()[^1];                           // found once, not per chunk
        return _head.Forward(hidden.Narrow(0, start, count));
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
            var chunkLogits = Logits(hidden, r0, n);
            int size = chunkLogits.Size, width = size / n;
            float[] values = ArrayPool<float>.Shared.Rent(size);                  // one chunk's logits, returned in finally
            try
            {
                chunkLogits.CopyTo(values.AsSpan(0, size));
                for (int r = 0; r < n; r++)
                {
                    TopKLogits.Top(values.AsSpan(r * width, width), k, ids.AsSpan((r0 + r) * k, k), logits.AsSpan((r0 + r) * k, k));
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(values);
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
                    int k = teacher.TopK, size = logits.Size;
                    var ids = new int[count * k];
                    var top = new float[count * k];
                    float[] values = ArrayPool<float>.Shared.Rent(size);          // the logits, returned in finally
                    try
                    {
                        logits.CopyTo(values.AsSpan(0, size));
                        for (int r = 0; r < count; r++)
                        {
                            TopKLogits.Top(values.AsSpan(r * width, width), k, ids.AsSpan(r * k, k), top.AsSpan(r * k, k));
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(values);
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

    // A sequence's key: its tokens and which of them train. The integers are hashed as little-endian bytes (the file's
    // byte order), so a key is the same on every machine.
    public static ulong Of(TrainingSequence sequence)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(bytes, sequence.Tokens.Length);
        ulong hash = Add(Start, bytes);
        if (BitConverter.IsLittleEndian)
        {
            hash = Add(hash, System.Runtime.InteropServices.MemoryMarshal.AsBytes(sequence.Tokens.AsSpan()));
        }
        else
        {
            foreach (int token in sequence.Tokens)
            {
                System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(bytes, token);
                hash = Add(hash, bytes);
            }
        }

        foreach (bool trained in sequence.Trained)
        {
            hash = (hash ^ (trained ? 1UL : 2UL)) * 1099511628211UL;
        }

        // The images too (their content hashes and preparation): the same text over other pages is another record. A
        // sequence without images keeps the key it always had.
        foreach (var image in sequence.Images)
        {
            hash = Add(hash, System.Text.Encoding.UTF8.GetBytes(image.Image.Hash));
            hash = Add(hash, System.Text.Encoding.UTF8.GetBytes(image.Preparation.ToJson().ToJsonString()));
        }

        return hash;
    }
}
