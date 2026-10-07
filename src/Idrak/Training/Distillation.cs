// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Layers;

namespace Idrak.Training;

/// <summary>
/// Knowledge distillation for models trained with <see cref="Trainer"/> (classifiers, per sample): a teacher's logits
/// are computed once and stored with the targets (<see cref="WithTeacher"/>), and <see cref="Losses.Distillation(float, float)"/>
/// trains the student on them, alone or mixed with the labels. Language models distil per token through the
/// fine-tuning loss hook instead (Idrak.Nlp: <c>FineTuningLosses.Distillation</c>).
/// </summary>
/// <example>
/// <code>
/// var train = data.WithTeacher(teacher);                                     // targets: teacher logits, then the labels
/// using var trainer = new Trainer(student, Losses.Distillation(temperature: 2f, alpha: 0.7f), p => new AdamW(p, 3e-3f));
/// trainer.Fit(new DataLoader(train, 64, shuffle: true), epochs: 20);
/// </code>
/// </example>
public static class Distillation
{
    /// <summary>
    /// <paramref name="data"/> with <paramref name="teacher"/>'s logits for each sample put before its targets: targets
    /// [samples, classes + the original target columns], with target names teacher0, teacher1 … then the original ones.
    /// The teacher runs in inference mode (dropout off, no gradients) on its own device, <paramref name="batchSize"/>
    /// samples at a time; the features are unchanged.
    /// </summary>
    /// <param name="data">The training (or validation) data; its targets are the labels (one-hot rows or class indices).</param>
    /// <param name="teacher">The trained teacher: raw scores (logits) out, one row of classes per sample.</param>
    /// <param name="batchSize">Samples per teacher pass.</param>
    public static Dataset WithTeacher(this Dataset data, Module teacher, int batchSize = 256)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        var device = teacher.WeightsDevice ?? Device.Default;
        float[]? logits = null;
        int classes = 0;
        foreach (var batch in new DataLoader(data, batchSize, device: device))
        {
            using var _ = batch;
            using var output = teacher.Predict(batch.Features);
            var values = output.ToArray();
            if (logits is null)
            {
                classes = output.Size / batch.Size;
                logits = new float[data.Count * classes];
            }

            if (output.Size != batch.Size * classes)
            {
                throw new InvalidOperationException($"The teacher gave {output.Size / batch.Size} outputs per sample, then {classes}.");
            }

            values.CopyTo(logits, batch.Index * batchSize * classes);
        }

        logits ??= [];
        int targets = data.TargetCount, columns = classes + targets;
        var combined = new float[data.Count * columns];
        for (int i = 0; i < data.Count; i++)
        {
            logits.AsSpan(i * classes, classes).CopyTo(combined.AsSpan(i * columns));
            data.GetTargets(i).CopyTo(combined.AsSpan(i * columns + classes));
        }

        var names = Enumerable.Range(0, classes).Select(c => $"teacher{c}").Concat(data.TargetNames).ToList();
        return Dataset.FromFlat(data.Features.ToArray(), combined, data.Count, data.FeatureNames, names).WithFeatureShape([.. data.FeatureShape]);
    }
}
