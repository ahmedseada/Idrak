// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text;
using Idrak.Text.Abstractions;

// Plan 19, phase B3: the pretraining loop with images: a stage's options over the scratch model's image records.
internal abstract partial class Tests
{
    internal static PretrainingOptions VisionStage(string name, IReadOnlyList<string> trained, int steps) => new()
    {
        Stages = [new PretrainingStage(name)
        {
            Trained = trained, Steps = steps, LearningRate = 0.05f, Schedule = "constant", WarmupFraction = 0,
            Sources = [new PretrainingSource(PretrainingSources.Records, [ScratchData("records.jsonl")]) { Format = "messages" }],
        }],
        MaxLength = 128, BatchTokens = 384, SequencesPerStep = 3, Optimizer = "sgd", Momentum = 0, WeightDecay = 0, MaxGradientNorm = 0,
        CudaGraphs = false, KeepCheckpoints = 0,
    };
}
