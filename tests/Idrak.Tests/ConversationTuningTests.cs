// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// The library's conversation tuning (ConversationTuning: what idrak tune and the legal OCR app train with) on the tiny
// Gemma 3 and LlamaFactory's ShareGPT records: the data and its images read, a held-out share, the image side, the
// scorer and the tuning settings prepared; a run trains, reports each step and the answers as they are scored, and
// writes the adapter, its image preparation and manifest; a stopped run saves what it trained so far.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ConversationTuningGroup =
    [
        ("conversation tuning: prepare reads ShareGPT with images from a zip, holds out a seeded share, limits the conversations, builds the image side and the cer scorer; nothing to train on and a metric without evaluation data are refused", ConversationTuningPrepares),
        ("conversation tuning: train reports every step and each scored answer, writes the adapter, tuning_images.json and idrak-tuning.json; a stopped run saves the adapter so far", ConversationTuningTrains),
    ];

    private static ConversationTuningOptions TinyConversationOptions(string zip) => new()
    {
        Images = zip, Preparation = TuningImages.Parse("grayscale"), MaxNewTokens = 4,
        Tuning = new FineTuningOptions { Rank = 2, Alpha = 4, Targets = ["q", "v"], LearningRate = 0.05f, BatchTokens = 64, MaxLength = 128, Packing = false },
    };

    private static void ConversationTuningPrepares(Device device)
    {
        RegisterGemma3Vision();
        string folder = TempFolder();
        try
        {
            var (_, zip) = WriteTuningImages(folder);
            string train = WriteShareGpt(folder, "gemma3", "train.json");
            using var model = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = device });
            var lines = new List<string>();
            var stages = new List<TuningStage>();
            using (var run = ConversationTuning.Prepare(model, [train], TinyConversationOptions(zip) with { EvaluationFraction = 0.34, Seed = 3, Metric = TuningMetrics.CharacterErrorRate },
                       new Reports<TuningStage>(stages.Add), lines.Add))
            {
                Check(run.TrainingConversations.Count == 2 && run.EvaluationConversations!.Count == 1 && run.TrainingSequences.Count == 2 && run.EvaluationSequences!.Count == 1,
                    $"{run.TrainingConversations.Count} + {run.EvaluationConversations?.Count} conversations");
                Check(run.Vision is not null && run.Tuning.Vision == run.Vision && run.Tuning.Answers == run.Scorer && run.Scorer!.Count == 1, "the image side and the scorer in the tuning");
                Check(run.Vision!.Images.Grayscale || run.Vision.Images.Pipeline.ToString().Contains("grayscale", StringComparison.Ordinal), run.Vision.Images.ToString());
                Check(run.TrainingSequences.All(s => s.Images.Count > 0), "the training sequences hold their images");
                Check(lines.Any(l => l.StartsWith("training: 3 conversations (sharegpt)", StringComparison.Ordinal)) && lines.Any(l => l.StartsWith("evaluation: 1 conversations held out", StringComparison.Ordinal)),
                    string.Join(" | ", lines));
                Check(stages.Count > 0 && stages[^1].Done == stages[^1].Total, $"{stages.Count} stages");
                Check(run.EvaluationLoss() is > 0, "an evaluation loss");
            }

            using (var limited = ConversationTuning.Prepare(model, [train], TinyConversationOptions(zip) with { MaxConversations = 1 }))
            {
                Check(limited.TrainingSequences.Count == 1 && limited.EvaluationSequences is null && limited.Scorer is null, $"{limited.TrainingSequences.Count} sequences");
            }

            try
            {
                ConversationTuning.Prepare(model, [train], TinyConversationOptions(zip) with { Metric = TuningMetrics.CharacterErrorRate }).Dispose();
                Check(false, "a metric without evaluation conversations must be refused");
            }
            catch (ArgumentException ex)
            {
                Check(ex.Message.Contains("evaluation", StringComparison.Ordinal), ex.Message);
            }

            string empty = Path.Combine(folder, "empty.json");
            File.WriteAllText(empty, """[{"messages": [{"role": "user", "content": "hello"}]}]""");
            try
            {
                ConversationTuning.Prepare(model, [empty], TinyConversationOptions(zip)).Dispose();
                Check(false, "nothing to train on must be refused");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message.Contains("Nothing to train on", StringComparison.Ordinal), ex.Message);
            }
        }
        finally
        {
            DeleteFolder(folder);
        }
    }

    private static void ConversationTuningTrains(Device device)
    {
        RegisterGemma3Vision();
        string folder = TempFolder();
        try
        {
            var (_, zip) = WriteTuningImages(folder);
            string train = WriteShareGpt(folder, "gemma3", "train.json");
            string validation = WriteShareGpt(folder, "gemma3", "val.json", records: 2);
            var options = TinyConversationOptions(zip) with { Evaluation = validation, Metric = TuningMetrics.CharacterErrorRate, MetricEvery = 2 };
            options = options with { Tuning = options.Tuning with { Epochs = 2 } };

            using (var model = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = device }))
            using (var run = ConversationTuning.Prepare(model, [train], options))
            {
                var steps = new List<FineTuningProgress>();
                int answered = 0;
                run.Scorer!.Progress = new Reports<TuningAnswer>(_ => answered++);
                string adapter = Path.Combine(folder, "adapter");
                var result = run.Train(adapter, VlmModel, new Reports<FineTuningProgress>(steps.Add));
                Check(!result.Stopped && steps.Count > 0 && steps[^1].Step == steps[^1].TotalSteps && result.Last == steps[^1], $"{steps.Count} steps, stopped {result.Stopped}");
                Check(result.EvaluationLosses.Count == 2 && result.Answers is { Answers.Count: 2 } && answered == 2 * (steps[^1].TotalSteps / 2),
                    $"{result.EvaluationLosses.Count} evaluations, {answered} answers scored");
                Check(File.Exists(Path.Combine(adapter, "adapter_model.safetensors")) && TuningImages.Read(adapter) is { } images && images.Family is not null
                      && TuningManifest.Read(adapter)!.BaseModel == Path.GetFullPath(VlmModel) && result.Folder == Path.GetFullPath(adapter),
                    "the adapter, its preparation and manifest");
            }

            // Stopped after the first step: the adapter so far is saved.
            using (var model = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = device }))
            using (var run = ConversationTuning.Prepare(model, [train], TinyConversationOptions(zip) with { Tuning = TinyConversationOptions(zip).Tuning with { Epochs = 20 } }))
            {
                using var cancel = new CancellationTokenSource();
                string adapter = Path.Combine(folder, "stopped");
                var result = run.Train(adapter, VlmModel, new Reports<FineTuningProgress>(_ => cancel.Cancel()), cancel.Token);
                Check(result.Stopped && result.Last is { Step: >= 1 } last && last.Step < last.TotalSteps, $"stopped {result.Stopped} at {result.Last?.Step} of {result.Last?.TotalSteps}");
                Check(File.Exists(Path.Combine(adapter, "adapter_model.safetensors")) && TuningManifest.Exists(adapter) && TuningImages.Exists(adapter), "the adapter so far");
            }
        }
        finally
        {
            DeleteFolder(folder);
        }
    }

    private static void DeleteFolder(string folder)
    {
        try
        {
            Directory.Delete(folder, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private sealed class Reports<T>(Action<T> record) : IProgress<T>
    {
        public void Report(T value) => record(value);
    }
}
