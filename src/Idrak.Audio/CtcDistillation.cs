// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// Frame-level distillation of a CTC student from a CTC teacher of the same classes (plan 21, S6;
/// <see cref="SpeechTrainingOptions.Teacher"/>): (1 − α) · the student's CTC loss (<see cref="CtcObjective.Loss(ICtcSpeechModel, Tensor, IReadOnlyList{int}, IReadOnlyList{int[]}, SpeechForwardOptions?)"/>)
/// + α · <see cref="Losses.Distillation(Tensor, Tensor, float)"/> over every step the utterances own (their padding left
/// out): T² · the mean over those steps of KL(softmax(teacher / T) ‖ softmax(student / T)). The teacher's posteriors are
/// constants, taken with gradients off on its own device. With α = 0 it is the CTC loss alone.
/// </summary>
public static class CtcDistillation
{
    /// <summary>The step's loss for a batch of features [batch, frames, mels] on the student's device.</summary>
    /// <exception cref="ArgumentException">The teacher gives an utterance other steps than the student, or other classes.</exception>
    public static Tensor Loss(ICtcSpeechModel student, ICtcSpeechModel teacher, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, float alpha,
        float temperature, SpeechForwardOptions? forward = null)
    {
        ArgumentNullException.ThrowIfNull(student);
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(frames);
        var logits = student.Logits(features, frames, forward);
        var ctc = Losses.Ctc(logits.LogSoftmax(), labels, [.. frames.Select(student.StepsFor)], student.Blank, LossReduction.Mean, zeroInfinity: true, batchFirst: true);
        if (alpha <= 0f)
        {
            return ctc;
        }

        Tensor taught;
        using (Autograd.NoGrad())
        {
            var input = teacher.Device == student.Device ? features : features.To(teacher.Device);
            taught = teacher.Logits(input, frames);
            if (teacher.Device != student.Device)
            {
                taught = taught.To(student.Device);
            }
        }

        if (taught.Shape[^1] != logits.Shape[^1])
        {
            throw new ArgumentException($"The teacher gives {taught.Shape[^1]} classes and the student {logits.Shape[^1]}: a CTC teacher has the student's classes.", nameof(teacher));
        }

        // Each utterance's own steps, as one [steps, classes] block of every utterance's rows.
        var student_ = new List<Tensor>(frames.Count);
        var teacher_ = new List<Tensor>(frames.Count);
        for (int i = 0; i < frames.Count; i++)
        {
            int steps = student.StepsFor(frames[i]);
            if (teacher.StepsFor(frames[i]) != steps)
            {
                throw new ArgumentException($"The teacher gives {teacher.StepsFor(frames[i])} steps for {frames[i]} frames and the student {steps}: a CTC teacher subsamples as the student does.", nameof(teacher));
            }

            if (steps > 0)
            {
                student_.Add(logits.Narrow(0, i, 1).Narrow(1, 0, steps).Reshape(steps, logits.Shape[^1]));
                teacher_.Add(taught.Narrow(0, i, 1).Narrow(1, 0, steps).Reshape(steps, taught.Shape[^1]));
            }
        }

        if (student_.Count == 0)
        {
            return ctc;
        }

        var soft = Losses.Distillation(student_.Count == 1 ? student_[0] : Tensor.Concat(student_), teacher_.Count == 1 ? teacher_[0] : Tensor.Concat(teacher_), temperature);
        return alpha >= 1f ? soft : ctc * (1f - alpha) + soft * alpha;
    }

    /// <summary>
    /// Refuses a teacher that cannot teach <paramref name="student"/> frame by frame: other classes, another blank, or
    /// other feature options (the teacher reads the student's features).
    /// </summary>
    /// <exception cref="ArgumentException">Why the teacher does not fit.</exception>
    public static void Check(ICtcSpeechModel student, ICtcSpeechModel teacher)
    {
        ArgumentNullException.ThrowIfNull(student);
        ArgumentNullException.ThrowIfNull(teacher);
        if (teacher.Classes != student.Classes || teacher.Blank != student.Blank)
        {
            throw new ArgumentException($"The CTC teacher has {teacher.Classes} classes (blank {teacher.Blank}) and the student {student.Classes} (blank {student.Blank}): "
                + "frame-level distillation needs the same labels (create the student with the teacher's tokenizer).", nameof(teacher));
        }

        string mine = student.Features.Options.ToJson().ToJsonString(), theirs = teacher.Features.Options.ToJson().ToJsonString();
        if (mine != theirs)
        {
            throw new ArgumentException($"The CTC teacher reads other features ({theirs}) than the student ({mine}): frame-level distillation gives both the same features.", nameof(teacher));
        }
    }
}
