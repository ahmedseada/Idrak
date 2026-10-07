// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Abstraction.Training;

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

    /// <summary>Equal when both hold the same token and trained arrays (the cached count is not compared).</summary>
    public bool Equals(TrainingSequence? other) =>
        other is not null && ReferenceEquals(Tokens, other.Tokens) && ReferenceEquals(Trained, other.Trained);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Tokens, Trained);
}

/// <summary>
/// The teacher of knowledge distillation: the distribution over the next token at every trained position of a batch,
/// which a fine-tuning loss compares with the student's (Idrak.LanguageModels' <c>FineTuningLossInput.TeacherDivergence</c>
/// and <c>FineTuningLosses.Distillation</c>; set the teacher as <c>FineTuningOptions.Teacher</c>). Idrak.LanguageModels
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
        for (int id = 0; id < count; id++)
        {
            foreach (byte b in Encoding.UTF8.GetBytes(TokenText(tokenizer, id)))
            {
                hash = (hash ^ b) * 1099511628211UL;
            }

            hash = (hash ^ 0xFF) * 1099511628211UL;                             // a separator no UTF-8 text contains
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
