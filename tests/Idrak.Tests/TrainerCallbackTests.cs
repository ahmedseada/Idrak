// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// Trainer callbacks: when they run, stopping and cancelling, and the built-in EarlyStopping, Checkpoint and CsvLog.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Callbacks =
    [
        ("callbacks: begin, batch, epoch and end run in order with the right epoch and step; no batch loss unless asked", CallbacksOrder),
        ("callbacks: Stop() from OnEpochEnd and OnBatchEnd ends training; cancellation ends it after the current batch", CallbacksStop),
        ("callbacks: EarlyStopping gives the same history and weights as EarlyStoppingPatience (Trainer and TrainingRun)", CallbacksEarlyStopping),
        ("callbacks: Checkpoint writes last and best weights that load back; CsvLog writes one row per epoch", CallbacksCheckpointAndCsv),
    ];

    private sealed class Recorder(bool needsLoss = false) : ITrainerCallback
    {
        public List<string> Calls { get; } = [];
        public List<BatchCompleted> Batches { get; } = [];
        public Func<TrainerContext, BatchCompleted, bool>? StopAfterBatch { get; init; }
        public Func<TrainerContext, EpochCompleted, bool>? StopAfterEpoch { get; init; }
        public Action<TrainerContext, BatchCompleted>? AfterBatch { get; init; }
        public TrainingHistory? Ended { get; private set; }

        public bool NeedsBatchLoss => needsLoss;

        public void OnTrainBegin(TrainerContext context) => Calls.Add($"begin {context.Epoch}/{context.Step}");

        public void OnBatchEnd(TrainerContext context, BatchCompleted batch)
        {
            Calls.Add($"batch {context.Epoch}.{batch.Batch} step {context.Step}");
            Batches.Add(batch);
            AfterBatch?.Invoke(context, batch);
            if (StopAfterBatch?.Invoke(context, batch) == true)
            {
                context.Stop();
            }
        }

        public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
        {
            Calls.Add($"epoch {epoch.Epoch} history {context.History.Epochs.Count}");
            if (StopAfterEpoch?.Invoke(context, epoch) == true)
            {
                context.Stop();
            }
        }

        public void OnTrainEnd(TrainerContext context, TrainingHistory history)
        {
            Calls.Add($"end {history.Epochs.Count}");
            Ended = history;
        }
    }

    private static (DataLoader Train, DataLoader Validation) CallbackData(Device device)
    {
        var split = SmallRegression(120, 5).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        return (split.Train.Batches(16, shuffle: true, device: device, seed: 3), split.Test.Batches(64, device: device));
    }

    private static void CallbacksOrder(Device device)
    {
        var (train, validation) = CallbackData(device);
        int batches = train.BatchCount;
        using var model = BuiltMlpFor3(device);
        var recorder = new Recorder();
        using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Callbacks = { recorder } };
        var history = trainer.Fit(train, 3, validation);

        var expected = new List<string> { "begin 0/0" };
        for (int e = 1; e <= 3; e++)
        {
            for (int b = 1; b <= batches; b++)
            {
                expected.Add($"batch {e}.{b} step {(e - 1) * batches + b}");
            }

            expected.Add($"epoch {e} history {e}");
        }

        expected.Add("end 3");
        Check(recorder.Calls.SequenceEqual(expected), $"order: {string.Join(" | ", recorder.Calls.Take(12))}");
        Check(ReferenceEquals(recorder.Ended, history), "OnTrainEnd receives the returned history");
        Check(recorder.Batches.All(b => double.IsNaN(b.Loss)), "no batch loss read when no callback needs it");

        using var model2 = BuiltMlpFor3(device);
        var withLoss = new Recorder(needsLoss: true);
        using var trainer2 = new Trainer(model2, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Callbacks = { withLoss } };
        var history2 = trainer2.Fit(train, 1);
        double mean = withLoss.Batches.Sum(b => b.Loss * b.BatchSize) / withLoss.Batches.Sum(b => b.BatchSize);
        Check(withLoss.Batches.All(b => double.IsFinite(b.Loss)) && Math.Abs(mean - history2.Epochs[0].Loss) < 1e-4,
            $"NeedsBatchLoss: batch losses average to the epoch loss ({mean} vs {history2.Epochs[0].Loss})");
    }

    private static void CallbacksStop(Device device)
    {
        var (train, validation) = CallbackData(device);
        int batches = train.BatchCount;

        using (var model = BuiltMlpFor3(device))
        {
            var stopper = new Recorder { StopAfterEpoch = (_, e) => e.Epoch == 2 };
            using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Callbacks = { stopper } };
            var history = trainer.Fit(train, 10, validation);
            Check(history.Epochs.Count == 2 && !history.StoppedEarly && stopper.Calls[^1] == "end 2", $"Stop() in OnEpochEnd: {history.Epochs.Count} epochs");
        }

        (train, validation) = CallbackData(device);
        using (var model = BuiltMlpFor3(device))
        {
            var stopper = new Recorder { StopAfterBatch = (c, _) => c.Step == batches + 2 };
            var run = new TrainingRun
            {
                Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
                Train = train, Validation = validation, Epochs = 10, Callbacks = [stopper],
            };
            var history = run.Fit();
            Check(history.Epochs.Count == 2 && stopper.Batches.Count == batches + 2, $"Stop() in OnBatchEnd: {history.Epochs.Count} epochs, {stopper.Batches.Count} batches");
            Check(history.Epochs[1].ValidationLoss is not null && stopper.Calls[^2] == "epoch 2 history 2", "the partial epoch is summarized and validated");
        }

        (train, validation) = CallbackData(device);
        using (var model = BuiltMlpFor3(device))
        {
            using var source = new CancellationTokenSource();
            var watcher = new Recorder { AfterBatch = (c, _) => { if (c.Step == batches + 2) source.Cancel(); } };
            using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Callbacks = { watcher } };
            bool threw = false;
            try
            {
                trainer.Fit(train, 10, validation, source.Token);
            }
            catch (OperationCanceledException)
            {
                threw = true;
            }

            Check(threw && watcher.Batches.Count == batches + 2, $"cancelled after the current batch ({watcher.Batches.Count} batches)");
            Check(watcher.Ended?.Epochs.Count == 1 && watcher.Calls[^1] == "end 1", "OnTrainEnd runs; the unfinished epoch is not added");
        }
    }

    private static float[] Weights(Module model) => [.. model.Parameters().SelectMany(p => p.ToArray())];

    private static void CallbacksEarlyStopping(Device device)
    {
        // Fresh loaders for each run, so every run sees the same shuffled order.
        var (train, validation) = CallbackData(device);
        using var builtIn = BuiltMlpFor3(device);
        using var trainer = new Trainer(builtIn, Losses.MeanSquaredError, p => new Adam(p, 0.08f)) { EarlyStoppingPatience = 2 };
        var expected = trainer.Fit(train, 200, validation);
        Check(expected.StoppedEarly && expected.Epochs.Count < 200, $"the built-in stops early ({expected.Epochs.Count} epochs)");

        (train, validation) = CallbackData(device);
        using var viaCallback = BuiltMlpFor3(device);
        var early = new EarlyStopping(2);
        using var callbackTrainer = new Trainer(viaCallback, Losses.MeanSquaredError, p => new Adam(p, 0.08f)) { Callbacks = { early } };
        var history = callbackTrainer.Fit(train, 200, validation);

        (train, validation) = CallbackData(device);
        using var viaRun = BuiltMlpFor3(device);
        var runHistory = new TrainingRun
        {
            Model = viaRun, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.08f),
            Train = train, Validation = validation, Epochs = 200, Callbacks = [new EarlyStopping(2)],
        }.Fit();

        foreach (var (h, name) in new[] { (history, "callback"), (runHistory, "TrainingRun") })
        {
            Check(h.StoppedEarly && h.Epochs.Count == expected.Epochs.Count && h.BestEpoch == expected.BestEpoch && h.BestLoss == expected.BestLoss,
                $"{name}: {h.Epochs.Count} epochs, best {h.BestEpoch} vs {expected.Epochs.Count}, best {expected.BestEpoch}");
            for (int i = 0; i < h.Epochs.Count; i++)
            {
                Check(h.Epochs[i].Loss == expected.Epochs[i].Loss && h.Epochs[i].ValidationLoss == expected.Epochs[i].ValidationLoss, $"{name}: epoch {i + 1}");
            }
        }

        Check(early.BestEpoch == expected.BestEpoch, "EarlyStopping.BestEpoch");
        Check(Weights(viaCallback).SequenceEqual(Weights(builtIn)) && Weights(viaRun).SequenceEqual(Weights(builtIn)), "the best weights are restored");
    }

    private sealed class BestSnapshot : ITrainerCallback
    {
        public float[]? Best { get; private set; }

        public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
        {
            if (epoch.IsBest)
            {
                Best = Weights(context.Model);
            }
        }
    }

    private static void CallbacksCheckpointAndCsv(Device device)
    {
        var (train, validation) = CallbackData(device);
        string folder = Path.Combine(Path.GetTempPath(), "idrak-checkpoint-" + Guid.NewGuid().ToString("N"));
        string csv = Path.Combine(folder, "log.csv");
        try
        {
            using var model = BuiltMlpFor3(device);
            var snapshot = new BestSnapshot();
            var checkpoint = new Checkpoint(folder);
            using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.08f))
            {
                Metrics = { Metric.MeanAbsoluteError },
                Callbacks = { snapshot, checkpoint, new CsvLog(csv) },
            };
            var history = trainer.Fit(train, 6, validation);

            using var last = BuiltMlpFor3(device);
            last.Load(checkpoint.LastPath);
            Check(Weights(last).SequenceEqual(Weights(model)), "last.ikw holds the final weights");
            using var best = BuiltMlpFor3(device);
            best.Load(checkpoint.BestPath);
            Check(Weights(best).SequenceEqual(snapshot.Best!), "best.ikw holds the best epoch's weights");
            Check(Directory.GetFiles(folder, "*.tmp").Length == 0, "no temporary files left");

            var lines = File.ReadAllLines(csv);
            Check(lines[0] == "epoch,loss,mae,val_loss,val_mae,learning_rate", $"header: {lines[0]}");
            Check(lines.Length == 1 + history.Epochs.Count, $"one row per epoch ({lines.Length - 1})");
            var inv = CultureInfo.InvariantCulture;
            for (int i = 0; i < history.Epochs.Count; i++)
            {
                var e = history.Epochs[i];
                string row = string.Join(',', e.Epoch.ToString(inv), e.Loss.ToString("R", inv), e.Metrics["mae"].ToString("R", inv),
                    e.ValidationLoss!.Value.ToString("R", inv), e.ValidationMetrics!["mae"].ToString("R", inv), e.LearningRate.ToString("R", inv));
                Check(lines[i + 1] == row, $"row {i + 1}: {lines[i + 1]}");
            }

            using var every = BuiltMlpFor3(device);
            var sparse = new Checkpoint(Path.Combine(folder, "sparse"), everyEpochs: 4, keepBest: false);
            using var t2 = new Trainer(every, Losses.MeanSquaredError, p => new Adam(p, 0.08f)) { Callbacks = { sparse } };
            t2.Fit(train, 3);
            Check(!File.Exists(sparse.LastPath) && !File.Exists(sparse.BestPath), "everyEpochs: 4 and keepBest: false write nothing in 3 epochs");
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
