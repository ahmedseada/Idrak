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
/// fly; <c>DistillationTeachers.FromFile</c>: stored top-k logits); implement this interface for another source of
/// distributions (a teacher served elsewhere, another stored format), returning each batch's distributions from
/// <see cref="Distributions"/>.
/// <para>
/// Teacher and student must share a vocabulary (the same token for every id): every teacher says what its vocabulary is
/// (<see cref="Identity"/>), and training checks it against the student's first (<see cref="Check"/>). When they do not
/// share one, distil through data the teacher writes instead (sequence-level distillation).
/// </para>
/// </summary>
public interface IDistillationTeacher : IDisposable
{
    /// <summary>The width of the teacher's distributions (its output head; for stored logits, the width recorded).</summary>
    int Vocabulary { get; }

    /// <summary>Tokens kept per position (0: the full distribution).</summary>
    int TopK { get; }

    /// <summary>
    /// What the teacher's vocabulary is: its tokenizer, a fingerprint of it, or a teacher's own statement that it cannot
    /// say (<see cref="TeacherVocabulary.Unknown"/>, with the reason). Required: a teacher cannot leave it out.
    /// </summary>
    TeacherVocabulary Identity { get; }

    /// <summary>
    /// The distributions at the trained positions of <paramref name="sequences"/>, in order (the batch's): computed now or
    /// on demand, and released when the result is disposed (after the step).
    /// </summary>
    ITeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences);

    /// <summary>
    /// Checks that a student with the tokenizer <paramref name="student"/> can learn from this teacher (see
    /// <see cref="TeacherVocabulary.Check"/>). Throws <see cref="InvalidOperationException"/> naming the difference when not.
    /// </summary>
    void Check(ITokenizer student) => Identity.Check(student);
}

/// <summary>The teacher's distributions at the trained positions of one batch (see <see cref="IDistillationTeacher.Distributions"/>).</summary>
public interface ITeacherDistributions : IDisposable
{
    /// <summary>
    /// The probabilities at temperature <paramref name="temperature"/> of trained positions start … start + count − 1 as
    /// [count, vocabulary] on <paramref name="device"/>: a teacher wider than <paramref name="vocabulary"/> is cut to it (and
    /// renormalized), a narrower one gets zeros past its end.
    /// </summary>
    Tensor Probabilities(int start, int count, float temperature, int vocabulary, Device device);
}

/// <summary>
/// What a teacher's vocabulary is (<see cref="IDistillationTeacher.Identity"/>), so a student can be checked against it:
/// the teacher's tokenizer (<see cref="Of(ITokenizer)"/>: every id compared), the fingerprint stored with its logits
/// (<see cref="Fingerprinted"/>), or a statement that it is unknown (<see cref="Unknown"/>: a model without a tokenizer,
/// logits written before fingerprints were stored), whose reason a check names instead of passing silently.
/// </summary>
public sealed class TeacherVocabulary
{
    private string? _fingerprint;

    private TeacherVocabulary(ITokenizer? tokenizer, string? fingerprint, string? unknown)
    {
        Tokenizer = tokenizer;
        _fingerprint = fingerprint;
        UnknownReason = unknown;
    }

    /// <summary>The vocabulary of <paramref name="tokenizer"/>: a student's is compared id by id (<see cref="DistillationTeacher.CheckVocabulary"/>).</summary>
    public static TeacherVocabulary Of(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return new TeacherVocabulary(tokenizer, null, null);
    }

    /// <summary>The vocabulary whose fingerprint is <paramref name="fingerprint"/> (<see cref="DistillationTeacher.VocabularyFingerprint"/>): a student's is compared by its fingerprint.</summary>
    public static TeacherVocabulary Fingerprinted(string fingerprint)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fingerprint);
        return new TeacherVocabulary(null, fingerprint, null);
    }

    /// <summary>
    /// A vocabulary the teacher cannot state, for <paramref name="reason"/> ("the stored logits were written without a
    /// fingerprint"): a check cannot compare it and passes, on the teacher's word; the reason says why.
    /// </summary>
    public static TeacherVocabulary Unknown(string reason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reason);
        return new TeacherVocabulary(null, null, reason);
    }

    /// <summary>The teacher's tokenizer, when the identity is one (a model's; not stored logits).</summary>
    public ITokenizer? Tokenizer { get; }

    /// <summary>The vocabulary's fingerprint: stored, or computed once from <see cref="Tokenizer"/>; null when unknown.</summary>
    public string? Fingerprint => _fingerprint ??= Tokenizer is { } tokenizer ? DistillationTeacher.VocabularyFingerprint(tokenizer) : null;

    /// <summary>Why the vocabulary is unknown (<see cref="Unknown"/>); null when it is known.</summary>
    public string? UnknownReason { get; }

    /// <summary>Whether the vocabulary is known (a tokenizer or a fingerprint): a check then compares it.</summary>
    public bool IsKnown => UnknownReason is null;

    /// <summary>
    /// Checks that a student with the tokenizer <paramref name="student"/> shares this vocabulary: the same token for every
    /// id (a tokenizer, <see cref="DistillationTeacher.CheckVocabulary"/>) or the same fingerprint (stored logits); an
    /// <see cref="Unknown"/> vocabulary is taken on the teacher's word. Throws <see cref="InvalidOperationException"/>
    /// naming the difference when not.
    /// </summary>
    public void Check(ITokenizer student)
    {
        ArgumentNullException.ThrowIfNull(student);
        if (Tokenizer is { } own)
        {
            DistillationTeacher.CheckVocabulary(own, student);
        }
        else if (_fingerprint is { } fingerprint && fingerprint != DistillationTeacher.VocabularyFingerprint(student))
        {
            throw new InvalidOperationException(DistillationTeacher.Mismatch("the stored teacher logits were written for another vocabulary (their fingerprint differs from the student's)"));
        }
    }

    /// <inheritdoc />
    public override string ToString() => Tokenizer is not null ? "a tokenizer" : _fingerprint is { } f ? $"fingerprint {f}" : $"unknown ({UnknownReason})";
}

/// <summary>
/// What every distillation teacher's vocabulary is checked with (<see cref="IDistillationTeacher"/>,
/// <see cref="TeacherVocabulary"/>): the comparison of two tokenizers id by id, and the fingerprint stored with teacher
/// logits.
/// </summary>
public static class DistillationTeacher
{
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

    internal static string Mismatch(string detail) =>
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
