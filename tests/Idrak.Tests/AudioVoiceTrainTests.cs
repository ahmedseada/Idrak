// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Vits;

// Plan 21, phase S9 route C: a voice trained by fine-tuning a VITS checkpoint against discriminators. The fixture of
// tools/audio/make_tiny_audio.py --sections vits-train (vits-train/: a tiny VitsModel) loaded for training, and the options
// of a short run on tone speech.
internal abstract partial class DesktopTests : Tests
{
    private static string VitsTrainData(string relative) => AudioData(Path.Combine("vits-train", relative));

    private static VitsSynthesizer LoadVitsTrain(Device device) =>
        (VitsSynthesizer)SpeechSynthesisModels.Load(VitsTrainData("model"), new SpeechSynthesisLoadOptions { Device = device, Training = true });

    // A short run of the tiny VITS model: AdamW at a fine-tuning rate, a constant schedule, no clipping (VITS's), segments of 8 frames, 16 mel bands;
    // the feature cache the default one on disk, in `cache`.
    private static SpeechTrainingOptions VitsToneTraining(string cache) => new()
    {
        Steps = 4, BatchSeconds = 2.5, LearningRate = 2e-4f, Schedule = "constant", WarmupFraction = 0, MaxGradientNorm = 0, Seed = 5,
        FeatureCacheFolder = cache, Adversarial = new AdversarialSynthesisOptions { SegmentFrames = 8, Mels = 16 },
    };
}
