// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.VitsSpeech;

// Plan 21, phase S9 route C: a voice trained on a person's recordings by fine-tuning a VITS checkpoint against
// discriminators (AdversarialSynthesisTrainer, the VITS plug-in's training, the "checkpoint" voice kind and idrak voice
// train). The fixture comes from tools/audio/make_tiny_audio.py --sections vits-train (vits-train/: a tiny VitsModel with
// a weight-normed decoder and tiny weight-normed discriminators in torch, VITS's step on two utterances with its draws
// saved: both losses and every gradient of step 1, the alignment search's scores and paths, three SGD steps).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioVoiceTrainGroup =
    [
        ("audio voice train: a tiny VITS model loaded for training gives torch's step-1 discriminator loss and generator terms (mel, KL, duration, adversarial, feature matching) within 1e-4, every discriminator and generator gradient (weight-normed and transposed convolutions included) within 1e-4 of its tensor's largest (of a thousandth of the largest of any, for the key biases' zero gradients), the alignment search's durations exactly, and three SGD steps' losses and both models' weights within 1e-4", AudioVoiceTrainFixture),
        ("audio voice train: the training rows of a clip are VITS's spectrogram (within 1e-4 of torch's) and its samples, a frame a hop; the differentiable log-mel of a segment's sound equals the log-mel of its spectrogram rows within 1e-4; the generator's pass without kept graphs gives the kept pass's losses exactly; a speaking-only model, a multi-speaker one and discriminators unlike the folder's refused", AudioVoiceTrainUnits),
        ("audio voice train: AdversarialSynthesisTrainer fine-tunes the tiny VITS model on tone speech: a run stopped after step 2 and resumed from checkpoint-2 gives the whole run's losses, generator and discriminators bit for bit (its state naming the adversarial objective and options); a run that does not keep the generator's pass gives the same losses; the checkpoint speaks as a VITS model; a feature augmentation refused", AudioVoiceTrainResume),
        ("audio voice train: a short CPU run of the tiny VITS model on tone speech lowers its mel L1 (the mean of the last 5 steps under 0.8 of the first 5; step time printed)", AudioVoiceTrainSmoke),
    ];

    private static string VitsTrainData(string relative) => AudioData(Path.Combine("vits-train", relative));

    private static JsonObject VitsTrainReference => JsonNode.Parse(File.ReadAllText(VitsTrainData("reference/steps.json")))!.AsObject();

    private static AdversarialSynthesisOptions VitsTrainOptions(JsonObject reference) => new()
    {
        SegmentFrames = (int)reference["segment_frames"]!, MelWeight = (float)reference["mel_weight"]!, KlWeight = (float)reference["kl_weight"]!,
        DurationWeight = (float)reference["duration_weight"]!, AdversarialWeight = (float)reference["adversarial_weight"]!,
        FeatureMatchingWeight = (float)reference["feature_matching_weight"]!, Mels = (int)reference["mels"]!,
    };

    // The fixture's two utterances as the trainer hands them over: [2, longest, bins + hop] rows (torch's spectrogram, then each frame's samples).
    private static (Tensor Features, int[] Frames, int[][] Ids) VitsTrainRows(JsonObject reference, Device device)
    {
        int[] frames = [.. reference["frames"]!.AsArray().Select(v => (int)v!)];
        int[][] ids = [.. reference["ids"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
        int hop = (int)reference["hop"]!, bins = (int)reference["n_fft"]! / 2 + 1, width = bins + hop, longest = frames.Max();
        var rows = new float[frames.Length * longest * width];
        for (int u = 0; u < frames.Length; u++)
        {
            var spectrogram = ReadNpyFloat32(VitsTrainData($"reference/spectrogram-{u}.npy"));   // [bins, frames]
            var audio = ReadNpyFloat32(VitsTrainData($"audio-{u}.npy"));
            for (int f = 0; f < frames[u]; f++)
            {
                int at = (u * longest + f) * width;
                for (int k = 0; k < bins; k++)
                {
                    rows[at + k] = spectrogram[k * frames[u] + f];
                }

                audio.AsSpan(f * hop, hop).CopyTo(rows.AsSpan(at + bins, hop));
            }
        }

        return (Tensor.From(rows, [frames.Length, longest, width], device), frames, ids);
    }

    // The torch draws of `step` for the fixture's utterances.
    private static Func<int, AdversarialSynthesisBatch, VitsDraws> VitsTrainDraws(JsonObject reference)
    {
        var draws = ReadSafeTensors(VitsTrainData("reference/draws.safetensors"));
        var starts = reference["starts"]!.AsArray();
        return (i, batch) => new VitsDraws(draws[$"step{batch.Step}.posterior.{batch.Utterances[i]}"].Values, draws[$"step{batch.Step}.duration.{batch.Utterances[i]}"].Values,
            (int)starts[batch.Step - 1]![batch.Utterances[i]]!);
    }

    private static VitsSynthesizer LoadVitsTrain(Device device) =>
        (VitsSynthesizer)SpeechSynthesisModels.Load(VitsTrainData("model"), new SpeechSynthesisLoadOptions { Device = device, Training = true });

    private static void AudioVoiceTrainFixture(Device device)
    {
        VitsSpeechPlugin.Register();
        try
        {
            var reference = VitsTrainReference;
            var options = VitsTrainOptions(reference);
            using var model = LoadVitsTrain(device);
            Check(model.Trainable && model.Architecture == VitsSpeechFamily.Architecture, "loaded for training");
            var (features, frames, ids) = VitsTrainRows(reference, device);
            using var featuresScope = features;
            model.Training.Draws = VitsTrainDraws(reference);
            AdversarialSynthesisBatch Batch(int step) => new(features, frames, ids, [0, 1], step, 0, 2, frames.Sum(), ids.Sum(l => l.Length));

            // The alignment search: exact on torch's scores, and on the scores the model computes itself.
            int[][] torchDurations = [.. reference["durations"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
            for (int u = 0; u < 2; u++)
            {
                var path = MonotonicAlignment.Search(ReadNpyFloat32(VitsTrainData($"reference/neg-cent-{u}.npy")), frames[u], ids[u].Length);
                Check(MonotonicAlignment.Durations(path, ids[u].Length).SequenceEqual(torchDurations[u]), $"utterance {u}: the search on torch's scores gives {string.Join(" ", MonotonicAlignment.Durations(path, ids[u].Length))}, torch {string.Join(" ", torchDurations[u])}");
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var pass = model.Training.Pass(Batch(1), u, options);
                Check(pass.Durations.SequenceEqual(torchDurations[u]), $"utterance {u}: the model's own search gives {string.Join(" ", pass.Durations)}, torch {string.Join(" ", torchDurations[u])}");
            }

            ThrowsNaming<ArgumentException>(() => MonotonicAlignment.Search(new float[6], 2, 3), "fewer frames than tokens", "at least as many frames as tokens");

            var discriminators = model.Discriminators(options, 0);
            var named = model.Training.Opened!.Module.Named;
            var generator = (VitsParameterSet)model.Generator;
            using var sgdD = new Sgd(discriminators.Parameters(), (float)reference["discriminator_lr"]!);
            using var sgdG = new Sgd(model.Generator.Parameters(), (float)reference["generator_lr"]!);
            var steps = reference["steps"]!.AsArray();
            for (int s = 1; s <= 3; s++)
            {
                using var scope = new TensorScope();
                var expected = steps[s - 1]!;
                using var step = model.Begin(Batch(s), options, keep: true);
                sgdD.ZeroGrad();
                float lossD = step.Discriminate();
                Check(MathF.Abs(lossD - (float)expected["discriminator"]!) <= 1e-4f * MathF.Abs((float)expected["discriminator"]!), $"step {s}: discriminator loss {lossD}, torch {expected["discriminator"]}");
                if (s == 1)
                {
                    GradientsClose(ReadSafeTensors(VitsTrainData("reference/grads-discriminator-step1.safetensors")),
                        named.Select(p => (p.Name, p.Values(p.Tensor.Grad ?? throw new Exception($"{p.Name} has no gradient")))), 1e-4f, "discriminator step-1 gradient");
                }

                sgdD.Step();
                sgdG.ZeroGrad();
                var terms = step.Generate();
                foreach (var (key, name) in new[] { ("generator", "generator") })
                {
                    float torch = (float)expected[name]!;
                    Check(MathF.Abs(terms[key] - torch) <= 1e-4f * MathF.Abs(torch), $"step {s}: generator loss {terms[key]}, torch {torch}");
                }

                foreach (var (name, value) in expected["terms"]!.AsObject())
                {
                    float torch = (float)value!;
                    Check(MathF.Abs(terms[name] - torch) <= 1e-4f * Math.Max(MathF.Abs(torch), 1e-3f), $"step {s}: {name} {terms[name]}, torch {torch}");
                }

                if (s == 1)
                {
                    GradientsClose(ReadSafeTensors(VitsTrainData("reference/grads-generator-step1.safetensors")),
                        generator.Named.Select(p => (p.Name, p.Values(p.Tensor.Grad ?? throw new Exception($"{p.Name} has no gradient")))), 1e-4f, "generator step-1 gradient");
                }

                sgdG.Step();
            }

            TensorsClose(ReadSafeTensors(VitsTrainData("reference/weights-step3.safetensors")), generator.Named.Select(p => (p.Name, p.Values())), 1e-4f, "the generator after 3 SGD steps");
            TensorsClose(ReadSafeTensors(VitsTrainData("reference/discriminator-step3.safetensors")), named.Select(p => (p.Name, p.Values())), 1e-4f, "the discriminators after 3 SGD steps");
        }
        finally
        {
            VitsSpeechPlugin.Unregister();
        }
    }

    private static void AudioVoiceTrainUnits(Device device)
    {
        VitsSpeechPlugin.Register();
        try
        {
            var reference = VitsTrainReference;
            var options = VitsTrainOptions(reference);
            using var model = LoadVitsTrain(device);
            var extractor = ((IAdversarialSpeechSynthesizer)model).Features;
            int bins = (int)reference["n_fft"]! / 2 + 1, hop = (int)reference["hop"]!;
            Check(extractor.Name == "vits-frames" && model.FeatureWidth == bins + hop && extractor.Options.HopSize == hop, $"the rows: {extractor.Name}, {model.FeatureWidth} wide");
            var (features, frames, ids) = VitsTrainRows(reference, device);
            using var featuresScope = features;
            for (int u = 0; u < 2; u++)
            {
                var audio = ReadNpyFloat32(VitsTrainData($"audio-{u}.npy"));
                using var rows = extractor.Extract(new AudioData(audio, 1, 16000), Device.Cpu);
                Check(rows.Shape[0] == frames[u] && extractor.FramesFor(audio.Length) == frames[u], $"utterance {u}: {rows.Shape[0]} frames, torch {frames[u]}");
                var values = rows.ToArray();
                var spectrogram = ReadNpyFloat32(VitsTrainData($"reference/spectrogram-{u}.npy"));
                var mine = new float[bins * frames[u]];
                for (int f = 0; f < frames[u]; f++)
                {
                    for (int k = 0; k < bins; k++)
                    {
                        mine[k * frames[u] + f] = values[f * (bins + hop) + k];
                    }

                    Check(values.AsSpan(f * (bins + hop) + bins, hop).SequenceEqual(audio.AsSpan(f * hop, hop)), $"utterance {u}, frame {f}: its samples");
                }

                CloseRelative(spectrogram, mine, 1e-4f, $"utterance {u}'s spectrogram against torch's");
            }

            // The mel loss's two sides agree on real sound: the log-mel of the samples (padded, the DFT as a convolution) and of the spectrogram rows.
            using (var scope = new TensorScope())
            {
                var (filters, basis) = model.Training.Filters(options);
                // A 20-frame segment: its frames whose window lies inside it (the edges are reflect-padded within the segment, as VITS's mel loss pads them).
                const int Segment = 20;
                int size = (bins - 1) * 2, pad = (size - hop) / 2, from = (pad + hop - 1) / hop, to = (Segment * hop + pad - size) / hop, mels = options.Mels;
                var rows = features.Narrow(0, 1, 1).Reshape(features.Shape[1], bins + hop).Narrow(0, 4, Segment);
                var fromRows = rows.Narrow(1, 0, bins).MatMul(filters).Clamp(1e-5f, float.PositiveInfinity).Log().ToArray();
                var fromSound = model.Training.LogMel(rows.Narrow(1, bins, hop).Reshape(1, Segment * hop), basis, filters).ToArray();
                Check(fromSound.Length == Segment * mels && to - from >= 8, $"{fromSound.Length / mels} frames, {to - from + 1} inside");
                CloseRelative(fromRows[(from * mels)..((to + 1) * mels)], fromSound[(from * mels)..((to + 1) * mels)], 1e-4f, "the segment's log-mel from its sound against from its spectrogram");
            }

            // The pass run again (not kept) gives the same losses as the kept one.
            model.Training.Draws = VitsTrainDraws(reference);
            var batch = new AdversarialSynthesisBatch(features, frames, ids, [0, 1], 1, 0, 2, frames.Sum(), ids.Sum(l => l.Length));
            float[] Run(bool keep)
            {
                using var scope = new TensorScope();
                using var step = model.Begin(batch, options, keep);
                float d = step.Discriminate();
                var terms = step.Generate();
                foreach (var p in model.Generator.Parameters().Concat(model.Discriminators(options, 0).Parameters()))
                {
                    p.ZeroGrad();
                }

                return [d, .. terms.OrderBy(t => t.Key, StringComparer.Ordinal).Select(t => t.Value)];
            }

            var kept = Run(true);
            var again = Run(false);
            Check(kept.SequenceEqual(again), $"the pass run again: {string.Join(", ", again)}, kept {string.Join(", ", kept)}");

            // Refusals: discriminators unlike the folder's, a model loaded to speak, a multi-speaker checkpoint.
            using (var fresh = LoadVitsTrain(device))
            {
                ThrowsNaming<ArgumentException>(() => fresh.Discriminators(options with { Discriminator = new JsonObject { ["periods"] = new JsonArray(2) } }, 0),
                    "discriminators unlike the folder's", "holds discriminators of", "AdversarialSynthesisOptions.Discriminator");
            }

            using (var speaking = (VitsSynthesizer)SpeechSynthesisModels.Load(VitsTrainData("model"), new SpeechSynthesisLoadOptions { Device = device }))
            {
                ThrowsNaming<InvalidOperationException>(() => _ = speaking.Generator, "a model loaded to speak", "SpeechSynthesisLoadOptions.Training");
            }

            string root = TempFolder("vits-speakers");
            try
            {
                foreach (string file in Directory.GetFiles(VitsTrainData("model")))
                {
                    File.Copy(file, Path.Combine(root, Path.GetFileName(file)));
                }

                var config = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "config.json")))!.AsObject();
                config["num_speakers"] = 3;
                config["speaker_embedding_size"] = 4;
                File.WriteAllText(Path.Combine(root, "config.json"), config.ToJsonString());
                ThrowsNaming<NotSupportedException>(() => SpeechSynthesisModels.Load(root, new SpeechSynthesisLoadOptions { Device = device, Training = true }).Dispose(),
                    "a multi-speaker checkpoint", "single-speaker", "3 speakers");
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }
        finally
        {
            VitsSpeechPlugin.Unregister();
        }
    }

    // A short run of the tiny VITS model: AdamW at a fine-tuning rate, a constant schedule, no clipping (VITS's), segments of 8 frames, 16 mel bands.
    private static SpeechTrainingOptions VitsToneTraining(string cache) => new()
    {
        Steps = 4, BatchSeconds = 2.5, LearningRate = 2e-4f, Schedule = "constant", WarmupFraction = 0, MaxGradientNorm = 0, Seed = 5,
        FeatureCache = FeatureCaches.Disk, FeatureCacheFolder = cache, Adversarial = new AdversarialSynthesisOptions { SegmentFrames = 8, Mels = 16 },
    };

    private static void AudioVoiceTrainResume(Device device)
    {
        string root = TempFolder("vits-resume");
        VitsSpeechPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 8, 31, maxLetters: 4);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = VitsToneTraining(Path.Combine(root, "cache")) with { SaveEvery = 2 };
            SpeechTrainingResult whole;
            using (var model = LoadVitsTrain(device))
            {
                ThrowsNaming<ArgumentException>(() => AdversarialSynthesisTrainer.Train(model, train, null, options with { Augmentations = ["spec-mask"] }, Path.Combine(root, "none")),
                    "a feature augmentation", "'spec-mask'", "sound augmentations only");
                whole = AdversarialSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run-a"));
            }

            Check(whole.Steps == 4 && whole.Losses.All(float.IsFinite) && whole.Utterances == 8, $"the run: {whole.Steps} steps, {whole.Utterances} utterances, losses {string.Join(", ", whole.Losses)}");
            var reports = new List<SpeechTrainingProgress>();
            using (var model = LoadVitsTrain(device))
            using (var stop = new CancellationTokenSource())
            {
                var stopper = new Progress(reports, r =>
                {
                    if (r.Stage == "step" && r.Step == 2)
                    {
                        stop.Cancel();
                    }
                });
                ThrowsNaming<OperationCanceledException>(() => AdversarialSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run-b"), progress: stopper,
                    cancellationToken: stop.Token), "the run cancelled after step 2");
            }

            var terms = reports.First(r => r.Stage == "step").Terms;
            Check(terms is not null && new[] { "mel", "kl", "duration", "adversarial", "feature_matching", "generator", "discriminator" }.All(terms.ContainsKey),
                $"the progress carries the terms: {(terms is null ? "none" : string.Join(", ", terms.Keys))}");
            string checkpoint = Path.Combine(root, "run-b", "checkpoint-2");
            var state = JsonNode.Parse(File.ReadAllText(Path.Combine(checkpoint, SpeechTrainer.StateFile)))!;
            Check(state["objective"]?.GetValue<string>() == AdversarialSynthesisTrainer.Objective && (int?)state["options"]!["adversarial"]!["segment_frames"] == 8
                && AdversarialSynthesisTrainer.StateFiles.All(f => File.Exists(Path.Combine(checkpoint, f))) && File.Exists(Path.Combine(checkpoint, "discriminator.safetensors"))
                && File.Exists(Path.Combine(checkpoint, "vocab.json")), "the state names the adversarial objective and its options; both optimizers' state, the discriminators and the tokenizer are saved");
            var resumed = AdversarialSynthesisTrainer.Resume(checkpoint, device: device);
            Check(resumed.Losses.SequenceEqual(whole.Losses), $"resumed losses {string.Join(", ", resumed.Losses)} are not the whole run's {string.Join(", ", whole.Losses)}");
            foreach (string file in new[] { "model.safetensors", "discriminator.safetensors" })
            {
                Check(File.ReadAllBytes(Path.Combine(root, "run-b", "checkpoint-4", file)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(root, "run-a", "checkpoint-4", file))),
                    $"the resumed run's {file} is the whole run's, bit for bit");
            }

            using (var model = LoadVitsTrain(device))
            {
                var again = AdversarialSynthesisTrainer.Train(model, train, null, options with { Adversarial = options.Adversarial! with { KeepGeneratorPass = false } }, Path.Combine(root, "run-c"));
                Check(again.Losses.SequenceEqual(whole.Losses), $"without the kept pass: {string.Join(", ", again.Losses)}, kept {string.Join(", ", whole.Losses)}");
            }

            using var trained = SpeechSynthesisModels.Load(Path.Combine(root, "run-a", "checkpoint-4"), new SpeechSynthesisLoadOptions { Device = device });
            var spoken = trained.Speak("abc", new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0 }).Audio;
            Check(trained is VitsSynthesizer && spoken.Length > 0 && spoken.Samples.All(float.IsFinite), "the checkpoint speaks as a VITS model");
        }
        finally
        {
            VitsSpeechPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceTrainSmoke(Device device)
    {
        string root = TempFolder("vits-smoke");
        VitsSpeechPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 16, 37, maxLetters: 4);
            var options = VitsToneTraining(Path.Combine(root, "cache")) with { Steps = 60, LearningRate = 2e-3f, BatchSeconds = 3 };
            var reports = new List<SpeechTrainingProgress>();
            using var model = LoadVitsTrain(device);
            var started = System.Diagnostics.Stopwatch.GetTimestamp();
            var result = AdversarialSynthesisTrainer.Train(model, new SpeechSource(SpeechSources.Transcripts, [transcripts]), null, options, Path.Combine(root, "run"),
                progress: new Progress(reports));
            var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(started);
            var mel = reports.Where(r => r.Stage == "step").Select(r => r.Terms!["mel"]).ToArray();
            double first = mel.Take(5).Average(), last = mel.TakeLast(5).Average();
            Console.WriteLine($"    {result.Steps} steps in {elapsed.TotalSeconds:F1} s ({elapsed.TotalMilliseconds / result.Steps:F0} ms a step): mel L1 term {first:F3} -> {last:F3}");
            Check(mel.Length == 60 && last < 0.8 * first, $"the mel L1 term falls: the first 5 steps {first:F3}, the last 5 {last:F3} ({string.Join(", ", mel.Select(m => m.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)))})");
        }
        finally
        {
            VitsSpeechPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
