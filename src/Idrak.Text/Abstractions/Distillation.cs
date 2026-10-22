// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;

namespace Idrak.Text.Abstractions;

/// <summary>
/// One tokenized training sequence: its token ids and, for each token, whether the model learns to produce it (the
/// assistant's turns: reasoning, text, tool calls and the end-of-turn marker; not the system, user or tool messages).
/// </summary>
public sealed record TrainingSequence(int[] Tokens, bool[] Trained)
{
    private (bool[]? Of, int Count) _trained;

    /// <summary>Tokens the loss is computed on (a token is predicted from the ones before it, so the first never is).</summary>
    public int TrainedTokens
    {
        get
        {
            // Counted once per Trained array (batching asks for it many times per step).
            var cached = _trained;
            if (!ReferenceEquals(cached.Of, Trained))
            {
                int count = 0;
                for (int i = 1; i < Trained.Length; i++)
                {
                    count += Trained[i] ? 1 : 0;
                }

                _trained = cached = (Trained, count);
            }

            return cached.Count;
        }
    }

    /// <summary>
    /// The images in the sequence's prompt (a vision-language fine-tune, <see cref="Idrak.Text.ChatTranscriptEncoder.Vision"/>),
    /// in order: each image as received, how it is prepared, and where its blocks of image tokens lie in
    /// <see cref="Tokens"/>. Empty (the default) for text: such sequences batch and train as before.
    /// </summary>
    public IReadOnlyList<TrainingImage> Images { get; init; } = [];

    /// <summary>
    /// The parts of other kinds in the sequence's prompt (an audio part, through <see cref="Idrak.Text.ChatTranscriptEncoder.Parts"/>),
    /// in order: each part as received and where its blocks of soft tokens lie in <see cref="Tokens"/>. Empty (the
    /// default) for text and images.
    /// </summary>
    public IReadOnlyList<TrainingPart> Parts { get; init; } = [];

    /// <summary>
    /// One value for the whole sequence that a loss of one's own reads (<c>FineTuningLossInput.SequenceValues</c>): a policy
    /// loss's advantage, a reward. Null (the default): none.
    /// </summary>
    public float? Value { get; init; }

    /// <summary>
    /// A value per trained token, in their order (<c>FineTuningLossInput.TokenValues</c>): an episode whose turns have
    /// advantages of their own; null (the default): <see cref="Value"/> for every trained token.
    /// </summary>
    public float[]? TokenValues { get; init; }

    /// <summary>
    /// log p of each trained token under the policy that sampled the sequence, in their order
    /// (<c>FineTuningLossInput.BehaviorLogProbabilities</c>, a policy loss's ratio); null (the default): none.
    /// </summary>
    public float[]? BehaviorLogProbabilities { get; init; }

    /// <summary>
    /// log p of each trained token under the reference, computed beforehand (a session's <c>LogProbabilities</c> with the
    /// adapters off), in their order; null (the default): <c>FineTuningLossInput.ReferenceLogProbabilities</c> computes
    /// them when a loss asks.
    /// </summary>
    public float[]? ReferenceLogProbabilities { get; init; }

    /// <summary>
    /// Equal when both hold the same token, trained, image and part lists (the cached count, and the values and
    /// log-probabilities a loss reads, are not compared).
    /// </summary>
    public bool Equals(TrainingSequence? other) =>
        other is not null && ReferenceEquals(Tokens, other.Tokens) && ReferenceEquals(Trained, other.Trained) && ReferenceEquals(Images, other.Images)
        && ReferenceEquals(Parts, other.Parts);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Tokens, Trained);
}

/// <summary>
/// An image of a <see cref="TrainingSequence"/>: where its features come from (the image's encoded bytes and the
/// preparation it goes through, which with the encoder's identity key its cached features, <see cref="FeatureCacheKey"/>)
/// and where they go: each of the image's blocks of image tokens (as the family's encoder gives them, in order) with its
/// first position in the sequence's tokens and its token count. The image tokens and the markers around them are never
/// trained.
/// </summary>
/// <param name="Image">The image as received (its bytes; decoded only when its features are made).</param>
/// <param name="Preparation">The transforms and vision options it is read with.</param>
/// <param name="Blocks">Each block's first position in the tokens and its token count, in the order the encoder gives the blocks.</param>
public sealed record TrainingImage(ChatImage Image, Idrak.Text.TuningImages Preparation, IReadOnlyList<(int Position, int Tokens)> Blocks)
{
    /// <summary>The image tokens of all its blocks.</summary>
    public int Tokens => Blocks.Sum(b => b.Tokens);

    /// <summary>
    /// Each block's axis positions for a multi-axis rotary embedding ([axes · tokens], <see cref="IVisionEncoder.AxisPositions"/>),
    /// in the order of <see cref="Blocks"/>; null for a family whose image tokens take the next positions as text does.
    /// </summary>
    public IReadOnlyList<float[]?>? AxisPositions { get; init; }
}

/// <summary>
/// A part of another kind (<see cref="ChatPart.Kind"/>: "audio", say) of a <see cref="TrainingSequence"/>, the shape of
/// <see cref="TrainingImage"/> without the preparation: the part as received (its bytes; its features made by the
/// <see cref="ISoftPartSource"/> of its kind when a step needs them) and each of its blocks of soft tokens with its first
/// position in the sequence's tokens and its token count. The soft tokens and the markers around them are never trained.
/// </summary>
/// <param name="Part">The part as received.</param>
/// <param name="Blocks">Each block's first position in the tokens and its token count, in the order the source gives the blocks.</param>
public sealed record TrainingPart(ChatPart Part, IReadOnlyList<(int Position, int Tokens)> Blocks)
{
    /// <summary>The part's kind.</summary>
    public string Kind => Part.Kind;

    /// <summary>The soft tokens of all its blocks.</summary>
    public int Tokens => Blocks.Sum(b => b.Tokens);
}

/// <summary>
/// The teacher of knowledge distillation: the distribution over the next token at every trained position of a batch,
/// which a fine-tuning loss compares with the student's (Idrak.Text's <c>FineTuningLossInput.TeacherDivergence</c>
/// and <c>FineTuningLosses.Distillation</c>; set the teacher as <c>FineTuningOptions.Teacher</c>). Idrak.Text
/// makes the two built-in kinds (<c>DistillationTeachers.FromModel</c>: a loaded model computing its distributions on the
/// fly; <c>DistillationTeachers.FromFile</c>: stored top-k logits); derive from this class for another source of
/// distributions (a teacher served elsewhere, another stored format), returning each batch's distributions from
/// <see cref="Distributions"/>.
/// <para>
/// Teacher and student must share a vocabulary (the same token for every id): see <see cref="CheckVocabulary"/>. When they
/// do not, distil through data the teacher writes instead (sequence-level distillation).
/// </para>
/// </summary>
public abstract class DistillationTeacher : IDisposable
{
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
    /// The distributions at the trained positions of <paramref name="sequences"/>, in order (the batch's): computed now or
    /// on demand, and released when the result is disposed (after the step).
    /// </summary>
    public abstract TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences);

    /// <summary>
    /// Checks that a student with the tokenizer <paramref name="student"/> can learn from this teacher: the same token for
    /// every id (see <see cref="CheckVocabulary"/>, or the fingerprints of stored logits). Throws
    /// <see cref="InvalidOperationException"/> naming the difference when not.
    /// </summary>
    public void Check(ITokenizer student)
    {
        ArgumentNullException.ThrowIfNull(student);
        if (Tokenizer is { } own)
        {
            CheckVocabulary(own, student);
        }
        else if (Fingerprint is { } fingerprint && fingerprint != VocabularyFingerprint(student))
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
        ulong hash = 14695981039346656037UL;                                    // FNV-1a, 64 bits
        byte[] utf8 = ArrayPool<byte>.Shared.Rent(256);                         // each token's UTF-8, in one reused buffer
        try
        {
            for (int id = 0; id < count; id++)
            {
                string text = TokenText(tokenizer, id);
                int most = Encoding.UTF8.GetMaxByteCount(text.Length);
                if (most > utf8.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(most);
                    ArrayPool<byte>.Shared.Return(utf8);
                    utf8 = larger;
                }

                foreach (byte b in utf8.AsSpan(0, Encoding.UTF8.GetBytes(text, utf8)))
                {
                    hash = (hash ^ b) * 1099511628211UL;
                }

                hash = (hash ^ 0xFF) * 1099511628211UL;                         // a separator no UTF-8 text contains
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(utf8);
        }

        return string.Create(CultureInfo.InvariantCulture, $"{count}-{hash:x16}");
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

    private static string Mismatch(string detail) =>
        $"The teacher and the student do not share a vocabulary: {detail}. Logit distillation compares their distributions token by token, "
        + "so it needs the same tokenizer; distil through answers the teacher writes instead (sequence-level distillation, TeacherData), which works across vocabularies.";

    private static string Quote(string text) => text.Length == 0 ? "nothing" : $"\"{text}\"";

    // A token's text: its vocabulary entry (see ITokenizer.TokenOf); "" past the tokenizer's end.
    private static string TokenText(ITokenizer tokenizer, int id) => (uint)id >= (uint)tokenizer.VocabularySize ? "" : tokenizer.TokenOf(id);

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

/// <summary>The teacher's distributions at the trained positions of one batch (see <see cref="DistillationTeacher.Distributions"/>).</summary>
public abstract class TeacherDistributions : IDisposable
{
    /// <summary>
    /// The probabilities at temperature <paramref name="temperature"/> of trained positions start … start + count − 1 as
    /// [count, vocabulary] on <paramref name="device"/>: a teacher wider than <paramref name="vocabulary"/> is cut to it (and
    /// renormalized), a narrower one gets zeros past its end.
    /// </summary>
    public abstract Tensor Probabilities(int start, int count, float temperature, int vocabulary, Device device);

    /// <summary>Releases what the distributions hold (hidden states kept on a device, for example).</summary>
    public virtual void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}
