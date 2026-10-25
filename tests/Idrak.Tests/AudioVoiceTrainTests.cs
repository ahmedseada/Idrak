// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli;
using Idrak.Models;
using Idrak.Audio.Plugins.Vits;

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
        ("audio voice train: a trained folder installs as a voice of kind checkpoint (consent required, a folder without config.json refused, the training files left out, an installed name kept unless overwritten); VoicedSynthesizer speaks it with its own model, the samples the folder's own; export and import give it back, a checkpoint voice without its model refused", AudioVoiceTrainInstall),
        ("audio voice train: idrak voice train on the CPU fine-tunes the tiny VITS model on tone speech and installs the voice (refused without consent or a trainable base), voice show names its base and steps, speak --voice gives the checkpoint's samples; --resume and idrak resume add steps and install again; --time-steps times steps and installs nothing; its help has examples, every option and lines under 120 characters", AudioVoiceTrainCli),
        ("audio voice train: the stochastic duration predictor's flows forward (training) then in reverse (speaking) give the log-durations back within 1e-4: VITS's order, no flip after the element-wise affine (transformers' forward flips there, and a model fine-tuned that way speaks with durations it was not trained on)", AudioVoiceTrainDurationFlows),
        ("audio voice train: Frozen [\"decoder\"] keeps the decoder's tensors bit for bit while the other parts train, takes them out of the generator's parameters and saves them in the run's options; a part VITS does not have is refused naming its parts", AudioVoiceTrainFrozen),
        ("audio voice train: DiscriminatorWeights starts a run from a file's discriminators when the base ships none: a checkpoint's discriminator.safetensors (its discriminator.json beside it) read bit for bit, a file under VITS's own weight_g and weight_v names (a release's D_*.pth) the same, a base folder's own discriminators kept whatever it names, a missing file refused naming the option; the run saves them in its state", AudioVoiceTrainDiscriminatorWeights),
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
        VitsPlugin.Register();
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
            var generator = (ParameterSet)model.Generator;
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
                var terms = step.Generate().ByName(0f);
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
            VitsPlugin.Unregister();
        }
    }

    private static void AudioVoiceTrainUnits(Device device)
    {
        VitsPlugin.Register();
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
                var terms = step.Generate().ByName(0f);
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
            VitsPlugin.Unregister();
        }
    }

    private static void AudioVoiceTrainDurationFlows(Device device)
    {
        VitsPlugin.Register();
        try
        {
            using var model = LoadVitsTrain(device);
            var predictor = (VitsStochasticDurationPredictor)model.StochasticDurations;
            var random = new Random(7);
            int c = model.Config.HiddenSize, t = 9;
            float[] Uniform(int n, float bound) => [.. Enumerable.Range(0, n).Select(_ => (float)((random.NextDouble() * 2 - 1) * bound))];
            using var scope = new TensorScope();
            var x = predictor.Conditioning(Tensor.From(Uniform(c * t, 1f), [c, t], device), null);
            float[] logDurations = Uniform(t, 2f), pair = Uniform(t, 2f);
            var prior = predictor.Prior(Tensor.From([.. logDurations, .. pair], [1, 2, t], device), x.Reshape(1, c, t), null, out _);
            var back = predictor.Reverse(prior.ToArray(), x);
            float worst = logDurations.Zip(back, (a, b) => MathF.Abs(a - b)).Max();
            Check(worst < 1e-4f, $"the duration flows forward then in reverse give log-durations {string.Join(" ", back.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))} "
                + $"for {string.Join(" ", logDurations.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))} (off by {worst:G3}): training's flow order is not the inverse of speaking's");
        }
        finally
        {
            VitsPlugin.Unregister();
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
        VitsPlugin.Register();
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
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceTrainDiscriminatorWeights(Device device)
    {
        string root = TempFolder("vits-disc-weights");
        VitsPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 6, 37, maxLetters: 4);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = VitsToneTraining(Path.Combine(root, "cache")) with { Steps = 1 };
            using (var model = LoadVitsTrain(device))
            {
                AdversarialSynthesisTrainer.Train(model, train, null, options, Path.Combine(root, "run"));
            }

            // The tiny model's folder ships discriminators; a copy without them starts from the file the options name.
            string bare = Path.Combine(root, "bare");
            Directory.CreateDirectory(bare);
            foreach (string file in Directory.GetFiles(VitsTrainData("model")).Where(f => !Path.GetFileName(f).StartsWith("discriminator.", StringComparison.Ordinal)))
            {
                File.Copy(file, Path.Combine(bare, Path.GetFileName(file)));
            }

            string trained = Path.Combine(root, "run", "checkpoint-1", "discriminator.safetensors");
            using var reader = Idrak.Models.SafeTensorsReader.Open(trained);
            var names = ((Idrak.Abstraction.Formats.ITensorStore)reader).Names.ToList();
            string released = Path.Combine(root, "release", "D.safetensors");
            Directory.CreateDirectory(Path.GetDirectoryName(released)!);
            File.Copy(Path.Combine(root, "run", "checkpoint-1", "discriminator.json"), Path.Combine(root, "release", "discriminator.json"));
            Idrak.Models.SafeTensorsWriter.Write(released, names.Select(n => (n.Replace(".parametrizations.weight.original0", ".weight_g", StringComparison.Ordinal)
                .Replace(".parametrizations.weight.original1", ".weight_v", StringComparison.Ordinal), ((Idrak.Abstraction.Formats.ITensorStore)reader).ShapeOf(n), reader.Read(n))), Idrak.Models.SafeTensorType.F32);

            List<(string Name, float[] Values)> Loaded(string folder, string? file)
            {
                using var synthesizer = (VitsSynthesizer)SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = device, Training = true });
                synthesizer.Discriminators(options.Adversarial! with { DiscriminatorWeights = file }, 0);
                return [.. synthesizer.Training.Opened!.Module.Named.Select(p => (p.Name, p.Values()))];
            }

            var fromCheckpoint = Loaded(bare, trained);
            Check(fromCheckpoint.Count == names.Count && fromCheckpoint.All(p => names.Contains(p.Name) && p.Values.AsSpan().SequenceEqual(reader.Read(p.Name))),
                "the checkpoint's discriminators read bit for bit under their names");
            var fromRelease = Loaded(bare, released);
            Check(fromRelease.Count == fromCheckpoint.Count && fromRelease.Any(p => p.Name.EndsWith(".weight_g", StringComparison.Ordinal))
                && fromRelease.Zip(fromCheckpoint).All(pair => pair.First.Values.AsSpan().SequenceEqual(pair.Second.Values)),
                "a file under weight_g and weight_v names gives the same discriminators");
            var fromFolder = Loaded(VitsTrainData("model"), released);
            Check(fromFolder.All(p => p.Name.Contains(".parametrizations.", StringComparison.Ordinal) || p.Name.EndsWith(".bias", StringComparison.Ordinal))
                && fromFolder.Zip(fromCheckpoint).Any(pair => !pair.First.Values.AsSpan().SequenceEqual(pair.Second.Values)),
                "a folder's own discriminators are the ones trained, whatever the options name");
            using (var synthesizer = (VitsSynthesizer)SpeechSynthesisModels.Load(bare, new SpeechSynthesisLoadOptions { Device = device, Training = true }))
            {
                ThrowsNaming<FileNotFoundException>(() => synthesizer.Discriminators(options.Adversarial! with { DiscriminatorWeights = Path.Combine(root, "none.pth") }, 0),
                    "a missing discriminators file", "none.pth", "DiscriminatorWeights");
            }

            // A run from the bare copy and the released file: its checkpoint holds them, trained, and its state names the file.
            using (var synthesizer = (VitsSynthesizer)SpeechSynthesisModels.Load(bare, new SpeechSynthesisLoadOptions { Device = device, Training = true }))
            {
                var result = AdversarialSynthesisTrainer.Train(synthesizer, train, null, options with { Adversarial = options.Adversarial! with { DiscriminatorWeights = released } }, Path.Combine(root, "run2"));
                Check(result.Steps == 1 && File.Exists(Path.Combine(root, "run2", "checkpoint-1", "discriminator.safetensors")), "the run saves the discriminators it started from");
            }

            string state = File.ReadAllText(Path.Combine(root, "run2", "checkpoint-1", SpeechTrainer.StateFile));
            Check(state.Contains("\"discriminator_weights\":", StringComparison.Ordinal) && state.Contains("D.safetensors", StringComparison.Ordinal), "the run's options name the file");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceTrainFrozen(Device device)
    {
        string root = TempFolder("vits-frozen");
        VitsPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 6, 37, maxLetters: 4);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = VitsToneTraining(Path.Combine(root, "cache")) with { Steps = 2 };
            using (var model = LoadVitsTrain(device))
            {
                Check(model.Parts.Order().SequenceEqual(["decoder", "duration_predictor", "flow", "posterior_encoder", "text_encoder"]),
                    $"VITS names its parts: {string.Join(", ", model.Parts)}");
                ThrowsNaming<ArgumentException>(() => AdversarialSynthesisTrainer.Train(model, train, null,
                    options with { Adversarial = options.Adversarial! with { Frozen = ["vocoder"] } }, Path.Combine(root, "none")), "no part 'vocoder'", "decoder", "flow");
            }

            using (var model = LoadVitsTrain(device))
            {
                int all = model.Generator.Parameters().Count();
                var result = AdversarialSynthesisTrainer.Train(model, train, null, options with { Adversarial = options.Adversarial! with { Frozen = ["decoder"] } }, Path.Combine(root, "run"));
                int trained = model.Generator.Parameters().Count();
                Check(result.Steps == 2 && result.Losses.All(float.IsFinite) && trained < all, $"{result.Steps} steps; {trained} of {all} tensors trained");
            }

            using var before = Idrak.Models.SafeTensorsReader.Open(Path.Combine(VitsTrainData("model"), "model.safetensors"));
            using var after = Idrak.Models.SafeTensorsReader.Open(Path.Combine(root, "run", "checkpoint-2", "model.safetensors"));
            int kept = 0, moved = 0;
            foreach (string name in ((Idrak.Abstraction.Formats.ITensorStore)before).Names)
            {
                bool same = before.Read(name).AsSpan().SequenceEqual(after.Read(name));
                if (name.StartsWith("decoder.", StringComparison.Ordinal))
                {
                    Check(same, $"the frozen decoder's {name} changed");
                    kept++;
                }
                else if (!same)
                {
                    moved++;
                }
            }

            Check(kept > 0 && moved > 0, $"{kept} decoder tensors kept, {moved} other tensors trained");
            var state = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "run", "checkpoint-2", SpeechTrainer.StateFile)))!.ToJsonString();
            Check(state.Contains("\"frozen\":[\"decoder\"]", StringComparison.Ordinal), "the run's options keep the frozen parts (a resume freezes them too)");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceTrainSmoke(Device device)
    {
        string root = TempFolder("vits-smoke");
        VitsPlugin.Register();
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
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioVoiceTrainInstall(Device device)
    {
        string root = TempFolder("vits-voice");
        VitsPlugin.Register();
        try
        {
            var store = new FolderVoiceStore(Path.Combine(root, "voices"), "test-checkpoint-store");
            string model = VitsTrainData("model");
            var consent = new VoiceConsent("A tester", "I agree that this test voice is trained and installed.", DateTimeOffset.UtcNow);
            var options = new VoiceCheckpointOptions
            {
                Consent = consent, Languages = ["en"], Base = "vits-train/model", Training = new JsonObject { ["steps"] = 3, ["utterances"] = 2 },
                Exclude = [.. AdversarialSynthesisTrainer.StateFiles, "discriminator.safetensors", "discriminator.json"],
            };
            ThrowsNaming<ArgumentException>(() => store.AddCheckpoint("tester", model, options with { Consent = null }), "no consent", "has no consent", "VoiceCheckpointOptions.Consent");
            ThrowsNaming<FileNotFoundException>(() => store.AddCheckpoint("tester", root, options), "a folder without config.json", "holds no config.json");
            var voice = store.AddCheckpoint("tester", model, options);
            Check(voice.Kind == VoiceKinds.Checkpoint && voice.Base == "vits-train/model" && voice.Consent?.Speaker == "A tester" && voice.Languages.SequenceEqual(["en"])
                && File.Exists(Path.Combine(voice.CheckpointFolder!, "model.safetensors")) && File.Exists(Path.Combine(voice.CheckpointFolder!, "vocab.json"))
                && !File.Exists(Path.Combine(voice.CheckpointFolder!, "discriminator.safetensors")) && (int?)voice.Manifest["training"]!["steps"] == 3,
                $"the checkpoint voice: {voice.Manifest.ToJsonString()}");
            ThrowsNaming<IOException>(() => store.AddCheckpoint("tester", model, options), "a second voice of the name", "installed already", "Overwrite");
            Check(store.AddCheckpoint("tester", model, options with { Overwrite = true }).Kind == VoiceKinds.Checkpoint && store.List().Count == 1, "overwritten");
            Check(((IVoiceStore)new NoModelStore()).Name == "no-models", "a store of its own");
            Check((IVoiceStore)new NoModelStore() is not ICheckpointVoiceStore && store is ICheckpointVoiceStore, "a store that keeps no models is no ICheckpointVoiceStore");

            // Spoken through VoicedSynthesizer: the voice's own model speaks, whatever the synthesizer given.
            var speak = new SpeechSynthesisOptions { Voice = "tester", NoiseScale = 0, DurationNoiseScale = 0 };
            using var other = SpeechSynthesisModels.Load(AudioData("tiny-vits"), new SpeechSynthesisLoadOptions { Device = device });
            using var voiced = new VoicedSynthesizer(other, store, ownsSynthesizer: false);
            var (speaker, inner, converter, _) = voiced.Resolve(speak);
            Check(speaker is VitsSynthesizer && !ReferenceEquals(speaker, other) && converter is null && inner.Voice is null, "the voice resolves to its own model");
            var spoken = voiced.Speak("abc ab", speak).Audio;
            using (var own = SpeechSynthesisModels.Load(store.Find("tester")!.CheckpointFolder!, new SpeechSynthesisLoadOptions { Device = device }))
            {
                var expected = own.Speak("abc ab", speak with { Voice = null }).Audio;
                Check(spoken.Length > 0 && spoken.Samples.AsSpan().SequenceEqual(expected.Samples), $"VoicedSynthesizer's {spoken.Length} samples are the checkpoint's own {expected.Length}");
            }

            string zip = Path.Combine(root, "tester.zip");
            store.Export("tester", zip);
            var elsewhere = new FolderVoiceStore(Path.Combine(root, "elsewhere"), "test-elsewhere");
            var imported = elsewhere.Import(zip);
            Check(imported.Kind == VoiceKinds.Checkpoint && File.ReadAllBytes(Path.Combine(imported.CheckpointFolder!, "model.safetensors")).AsSpan()
                .SequenceEqual(File.ReadAllBytes(Path.Combine(voice.CheckpointFolder!, "model.safetensors"))), "export and import give the voice and its model back");
            string broken = Path.Combine(root, "broken");
            Directory.CreateDirectory(broken);
            File.Copy(Path.Combine(imported.Folder, FolderVoiceStore.ManifestFile), Path.Combine(broken, FolderVoiceStore.ManifestFile));
            ThrowsNaming<InvalidDataException>(() => elsewhere.Import(broken, overwrite: true), "a checkpoint voice without its model", "holds no model");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    // A store that keeps no model folders: it is no ICheckpointVoiceStore.
    private sealed class NoModelStore : IVoiceStore
    {
        public string Name => "no-models";

        public string Location => "nowhere";

        public IReadOnlyList<VoiceProfile> List() => [];

        public VoiceProfile? Find(string name) => null;

        public VoiceProfile Add(string name, IReadOnlyList<string> clips, ISpeakerEncoder encoder, VoiceAddOptions options, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public float[] ReadEmbedding(VoiceProfile voice) => throw new NotSupportedException();

        public VoiceProfile Import(string folderOrZip, bool overwrite = false) => throw new NotSupportedException();

        public void Export(string name, string zip) => throw new NotSupportedException();

        public bool Remove(string name) => false;
    }

    private static void AudioVoiceTrainCli(Device device)
    {
        string root = TempFolder("vits-cli"), d = device.ToString();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 6, 41, maxLetters: 4);
            string store = Path.Combine(root, "voices"), run = Path.Combine(root, "run");
            const string Consent = "I agree that this test voice is trained and installed.";
            string[] train = ["voice", "train", "tester", "--from", transcripts, "--base", VitsTrainData("model"), "-P", VitsPluginPath, "--steps", "2", "--batch-seconds", "2",
                "--segment-frames", "8", "--mels", "16", "--save-every", "1", "--log-every", "1", "--feature-cache", "none", "-o", run, "--voices", store, "-d", d];
            var noConsent = AudioCli(train);
            Check(noConsent.Code == 2 && noConsent.Err.Contains("--consent", StringComparison.Ordinal), $"no consent: {noConsent.Err}");
            var notTrainable = AudioCli([.. train.Select(a => a == VitsTrainData("model") ? AudioData("tiny-whisper") : a), "--consent", Consent]);
            Check(notTrainable.Code != 0, $"a base that is not a synthesizer: {notTrainable.Err}");
            var printed = AudioCli([.. train, "--consent", Consent]);
            Check(printed.Code == 0 && printed.Out.Contains("step 2/2", StringComparison.Ordinal) && printed.Out.Contains(" mel ", StringComparison.Ordinal)
                && printed.Out.Contains(" left", StringComparison.Ordinal) && printed.Out.Contains("installed 'tester'", StringComparison.Ordinal), $"voice train: {printed.Out}\n{printed.Err}");
            var shown = AudioCliJson("voice", "show", "tester", "--voices", store);
            Check((string?)shown["kind"] == VoiceKinds.Checkpoint && (int?)shown["training"]!["steps"] == 2 && (string?)shown["base"] == VitsTrainData("model")
                && (string?)shown["consent"]!["statement"] == Consent, $"voice show: {shown}");
            var runShown = AudioCli("runs", "show", run);
            Check(runShown.Code == 0 && runShown.Out.Contains("Settings  voice tester", StringComparison.Ordinal) && runShown.Out.Contains("2 steps", StringComparison.Ordinal)
                && !runShown.Out.Contains("task auto", StringComparison.Ordinal), $"runs show of a voice run names its voice, base and steps: {runShown.Out}");

            string wav = Path.Combine(root, "tester.wav");
            var spoken = AudioCli("speak", "-P", VitsPluginPath, AudioData("tiny-vits"), "abc ab", "--voice", "tester", "--voices", store, "--noise", "0", "--duration-noise", "0", "-o", wav, "-d", d);
            Check(spoken.Code == 0, $"speak --voice tester: {spoken.Err}");
            VitsPlugin.Register();
            try
            {
                using var own = SpeechSynthesisModels.Load(Path.Combine(store, "tester", "model"), new SpeechSynthesisLoadOptions { Device = device });
                var expected = own.Speak("abc ab", new SpeechSynthesisOptions { NoiseScale = 0, DurationNoiseScale = 0 }).Audio;
                var written = AudioCodecs.Decode(wav);
                Check(written.Length == expected.Length && written.SampleRate == expected.SampleRate, $"the checkpoint voice spoke {written.Length} samples, the checkpoint {expected.Length}");
            }
            finally
            {
                VitsPlugin.Unregister();
            }

            var resumed = AudioCliJson("voice", "train", "--resume", run, "--steps", "1", "-P", VitsPluginPath, "--voices", store, "-d", d);
            Check((int?)resumed["steps"] == 3 && (string?)resumed["voice"]!["name"] == "tester" && (int?)resumed["voice"]!["training"]!["steps"] == 3, $"--resume: {resumed}");
            var again = AudioCliJson("resume", run, "--steps", "1", "-P", VitsPluginPath, "--voices", store, "-d", d);
            Check((int?)again["steps"] == 4 && Directory.Exists(Path.Combine(run, "checkpoint-4")), $"idrak resume: {again}");

            // --time-steps times steps instead of training (kernel times only where the device times them) and installs nothing.
            var profiled = AudioCli([.. train.Select(a => a == "tester" ? "profiled" : a), "--consent", Consent, "--time-steps"]);
            Check(profiled.Code == 0 && profiled.Out.Contains("per step: ", StringComparison.Ordinal) && profiled.Out.Contains("real time", StringComparison.Ordinal)
                && !Directory.Exists(Path.Combine(store, "profiled")), $"voice train --time-steps: {profiled.Out}\n{profiled.Err}");

            var (code, help, _) = AudioCli("voice", "train", "--help");
            var command = CommandTable.All.First(c => c.Name == "voice train");
            Check(code == 0 && help.Contains("\nExamples:\n", StringComparison.Ordinal) && help.Contains("idrak voice train", StringComparison.Ordinal)
                && help.TrimEnd().Split('\n').All(l => l.Length <= 120) && command.ValueOptions.All(o => help.Contains(o, StringComparison.Ordinal))
                && Idrak.Cli.Shared.Messages.Arabic.ContainsKey(command.Summary), $"idrak voice train --help: {help}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
