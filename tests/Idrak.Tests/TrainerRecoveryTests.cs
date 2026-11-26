// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 36, G-1b (C8): the generic Trainer's out-of-memory recovery (OutOfMemoryPolicies) and its resumable checkpoints.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] TrainerRecoveryGroup =
    [
        ("trainer recovery: a run stopped after epoch 2 of 4 and resumed from its CheckpointFolder checkpoint (weights, optimizer, step, early-stopping state, seeded dropout's draws) ends with the uninterrupted run's weights bit for bit; only the newest checkpoint is kept, and each is written whole", d => { if (d == Device.Cpu) TrainerResumesBitForBit(); }),
        ("trainer recovery: an older checkpoint still loads: the Checkpoint callback's weights-only folder and an .ikw file give their weights from the first step, and a first-format train-state (a stop's, without the early-stopping state or the draws) resumes at its step", d => { if (d == Device.Cpu) TrainerLoadsOldCheckpoints(); }),
    ];

    // The model of these tests (the same weights from the same seed), with dropout when asked.
    private static Sequential RecoveryModel(bool dropout) => dropout
        ? Network.Input(3).OnDevice(Device.Cpu).Seed(11).Linear(16).ReLU().Dropout(0.2f).Linear(1).Build()
        : Network.Input(3).OnDevice(Device.Cpu).Seed(11).Linear(16).ReLU().Linear(1).Build();

    private static float[][] RecoveryWeights(Module model) => [.. model.Parameters().Select(p => p.ToArray())];

    private static void TrainerResumesBitForBit()
    {
        string folder = Path.Combine(Path.GetTempPath(), "idrak-trainer-resume-" + Guid.NewGuid().ToString("N"));
        try
        {
            Trainer Make(Module model, string? checkpoints = null, string? resume = null, Recorder? recorder = null)
            {
                var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f))
                {
                    EarlyStoppingPatience = 50, RestoreBestWeights = true, CheckpointFolder = checkpoints, ResumeFrom = resume,
                };
                if (recorder is not null)
                {
                    trainer.Callbacks.Add(recorder);
                }

                return trainer;
            }

            var (train, validation) = CallbackData(Device.Cpu);
            using var uninterrupted = RecoveryModel(dropout: true);
            using (var trainer = Make(uninterrupted))
            {
                trainer.Fit(train, 4, validation);
            }

            (train, validation) = CallbackData(Device.Cpu);
            using var stopped = RecoveryModel(dropout: true);
            using (var trainer = Make(stopped, checkpoints: folder, recorder: new Recorder { StopAfterEpoch = (_, e) => e.Epoch == 2 }))
            {
                Check(trainer.Fit(train, 4, validation).Epochs.Count == 2, "the run stops after epoch 2");
            }

            var written = Directory.GetDirectories(folder).Select(Path.GetFileName).ToList();
            int steps = train.BatchCount;
            Check(written.SequenceEqual([$"checkpoint-{2 * steps}"]) && !Directory.EnumerateDirectories(folder, "*.partial").Any(),
                $"only the newest checkpoint is kept, written whole: {string.Join(", ", written)}");
            string checkpoint = Path.Combine(folder, $"checkpoint-{2 * steps}");
            string state = File.ReadAllText(Path.Combine(checkpoint, "train-state.json"));
            Check(state.Contains("\"format\":\"idrak-train-state/2\"", StringComparison.Ordinal) && state.Contains("\"dropout_draws\":[", StringComparison.Ordinal)
                  && state.Contains("\"best_loss\"", StringComparison.Ordinal) && File.Exists(Path.Combine(checkpoint, "optimizer.safetensors"))
                  && File.Exists(Path.Combine(checkpoint, "best-weights.safetensors")),
                $"the checkpoint holds the weights, the optimizer, the step, the early-stopping state and the draws: {state}");

            (train, validation) = CallbackData(Device.Cpu);
            using var resumed = RecoveryModel(dropout: true);
            TrainingHistory history;
            using (var trainer = Make(resumed, resume: checkpoint))
            {
                history = trainer.Fit(train, 4, validation);
            }

            var expected = RecoveryWeights(uninterrupted);
            var actual = RecoveryWeights(resumed);
            Check(expected.Zip(actual).All(pair => pair.First.AsSpan().SequenceEqual(pair.Second)),
                "the resumed run ends with the uninterrupted run's weights bit for bit");
            Check(history.Epochs.Select(e => e.Epoch).SequenceEqual([3, 4]), $"the resumed run trains epochs 3 and 4: {string.Join(", ", history.Epochs.Select(e => e.Epoch))}");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    // Records the weights the run starts from.
    private sealed class StartWeights : ITrainerCallback
    {
        public float[][]? Weights { get; private set; }

        public void OnTrainBegin(TrainerContext context) => Weights = [.. context.Model.Parameters().Select(p => p.ToArray())];
    }

    private static void TrainerLoadsOldCheckpoints()
    {
        string folder = Path.Combine(Path.GetTempPath(), "idrak-trainer-old-" + Guid.NewGuid().ToString("N"));
        try
        {
            var (train, validation) = CallbackData(Device.Cpu);
            using var saved = RecoveryModel(dropout: false);
            using (var trainer = new Trainer(saved, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Callbacks = { new Checkpoint(folder, keepBest: false) } })
            {
                trainer.Fit(train, 1, validation);
            }

            var savedWeights = RecoveryWeights(saved);
            foreach (string from in new[] { folder, Path.Combine(folder, "last.ikw") })
            {
                (train, _) = CallbackData(Device.Cpu);
                using var model = RecoveryModel(dropout: false);
                var start = new StartWeights();
                using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { ResumeFrom = from, Callbacks = { start } };
                var history = trainer.Fit(train, 1);
                Check(start.Weights!.Zip(savedWeights).All(pair => pair.First.AsSpan().SequenceEqual(pair.Second)) && history.Epochs.Count == 1,
                    $"{Path.GetFileName(from)}: a weights-only checkpoint gives its weights, and the run starts at its first step");
            }

            // A first-format train-state (a stop's before the early-stopping state and the draws were kept): it resumes at its step.
            string checkpoints = Path.Combine(folder, "checkpoints");
            (train, validation) = CallbackData(Device.Cpu);
            using (var model = RecoveryModel(dropout: false))
            using (var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { CheckpointFolder = checkpoints })
            {
                trainer.Fit(train, 1, validation);
            }

            string checkpoint = Directory.GetDirectories(checkpoints).Single();
            long step = train.BatchCount;
            File.WriteAllText(Path.Combine(checkpoint, "train-state.json"), $"{{\"format\":\"idrak-train-state/1\",\"step\":{step}}}\n");
            File.Delete(Path.Combine(checkpoint, "best-weights.safetensors"));
            (train, validation) = CallbackData(Device.Cpu);
            using (var model = RecoveryModel(dropout: false))
            using (var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { ResumeFrom = checkpoint })
            {
                var history = trainer.Fit(train, 2, validation);
                Check(history.Epochs.Select(e => e.Epoch).SequenceEqual([2]), $"a first-format checkpoint resumes at its step: {string.Join(", ", history.Epochs.Select(e => e.Epoch))}");
            }

            ThrowsNaming<FileNotFoundException>(() =>
            {
                using var model = RecoveryModel(dropout: false);
                using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { ResumeFrom = Path.Combine(folder, "nothing-here") };
                trainer.Fit(train, 1);
            }, "a folder with no checkpoint", "train-state.json", "last.ikw");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }
}
