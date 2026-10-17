// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;
using Idrak.Audio;
using Idrak.Audio.Layers;
using Idrak.Layers;
using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>
/// The training rows of a VITS model (an <see cref="IAudioFeatureExtractor"/> named "vits-frames"): per frame of the
/// sound at the model's rate, its linear magnitude spectrum as VITS's <c>spectrogram_torch</c> takes it (reflect-padded by
/// (n_fft - hop) / 2, frames not centred, a periodic Hann window, √(power + 1e-6)) and the frame's hop of samples:
/// [frames, bins + hop]. The posterior encoder reads the spectra, the mel loss's target is their mel spectrum, and the
/// discriminators read the samples of the decoder's segment. Taken once and cached by the trainer.
/// </summary>
internal sealed class VitsTrainingFeatures : IAudioFeatureExtractor
{
    public VitsTrainingFeatures(VitsConfig config)
    {
        Bins = config.SpectrogramBins;
        Hop = config.HopLength;
        Options = new AudioFeatureOptions
        {
            SampleRate = config.SamplingRate, WindowSize = (Bins - 1) * 2, HopSize = Hop, Window = "hann", Center = false, PadSeconds = null, Normalize = null,
            DynamicRange = null, Path = "host",
        };
    }

    public int Bins { get; }

    public int Hop { get; }

    public int Width => Bins + Hop;

    public string Name => "vits-frames";

    public AudioFeatureOptions Options { get; }

    public int FramesFor(long samples) => samples < Options.WindowSize / 2 ? 0 : (int)(samples / Hop);

    public Tensor Extract(AudioData audio, Device device)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var samples = audio.Channels == 1 ? audio.Channel(0).ToArray() : Mono(audio);
        int frames = FramesFor(samples.Length);
        if (frames == 0)
        {
            throw new ArgumentException($"{samples.Length} samples are too few for a frame of {Options.WindowSize} (with its reflect padding).", nameof(audio));
        }

        int pad = (Options.WindowSize - Hop) / 2;
        var padded = new float[samples.Length + 2 * pad];
        samples.CopyTo(padded.AsSpan(pad));
        for (int i = 1; i <= pad; i++)
        {
            padded[pad - i] = samples[i];
            padded[pad + samples.Length - 1 + i] = samples[samples.Length - 1 - i];
        }

        var power = new float[Stft.FramesFor(padded.Length, Options) * Bins];
        Stft.PowerOnHost(padded, Options, power);
        var rows = new float[frames * Width];
        for (int f = 0; f < frames; f++)
        {
            var row = rows.AsSpan(f * Width, Width);
            for (int k = 0; k < Bins; k++)
            {
                row[k] = MathF.Sqrt(power[f * Bins + k] + 1e-6f);
            }

            samples.AsSpan(f * Hop, Hop).CopyTo(row[Bins..]);
        }

        return Tensor.From(rows, [frames, Width], device);
    }

    private static float[] Mono(AudioData audio)
    {
        var mono = new float[audio.Length];
        audio.ToMono(mono);
        return mono;
    }
}

/// <summary>What one utterance's generator pass draws: the posterior's noise [F, frames], the duration posterior's noise [2, tokens], the decoder segment's first frame.</summary>
internal sealed record VitsDraws(float[] Posterior, float[] Duration, int Start);

/// <summary>
/// The training of a <see cref="VitsSynthesizer"/> loaded with <see cref="SpeechSynthesisLoadOptions.Training"/> (plan 21, S9
/// route C), VITS's step on a micro-batch padded and masked as VITS trains it: the text encoder's prior, the posterior
/// encoder on the linear spectrograms, their latents drawn and taken through the flow, the monotonic alignment search on
/// the host for each utterance's durations, the KL of the posterior from the prior expanded to the frames, the stochastic
/// duration predictor's loss, and the decoder on a random segment of each utterance's latents, whose sound is scored by
/// the mel loss and the discriminators, all the segments as one batch.
/// </summary>
internal sealed class VitsTraining : IDisposable
{
    private readonly VitsSynthesizer _model;
    private readonly ConcurrentDictionary<(Device, int, float, float), (Tensor Mel, Tensor Basis)> _filters = new();
    private HifiGanDiscriminators? _discriminators;

    private readonly StoredLayers _weights;

    public VitsTraining(VitsSynthesizer model, StoredLayers weights)
    {
        _model = model;
        _weights = weights;
        Generator = new ParameterSet("VITS generator", weights.Trained);
        Features = new VitsTrainingFeatures(model.Config);
    }

    public ParameterSet Generator { get; }

    public VitsTrainingFeatures Features { get; }

    /// <summary>Tests: the draws of each utterance (its position in the batch, the batch) instead of the seeded ones.</summary>
    internal Func<int, AdversarialSynthesisBatch, VitsDraws>? Draws { get; set; }

    public HifiGanDiscriminators? Opened => _discriminators;

    public HifiGanDiscriminators Discriminators(AdversarialSynthesisOptions options, int seed)
    {
        if (_discriminators is not null)
        {
            return _discriminators;
        }

        var asked = options.Discriminator is { } description ? HifiGanDiscriminatorConfig.FromJson(description) : null;
        string? folder = _model.Folder;
        if (folder is not null && File.Exists(Path.Combine(folder, HifiGanDiscriminators.WeightsFile)))
        {
            string configPath = Path.Combine(folder, HifiGanDiscriminators.ConfigFile);
            var saved = File.Exists(configPath) ? HifiGanDiscriminatorConfig.FromJson(JsonNode.Parse(File.ReadAllText(configPath))!.AsObject()) : new HifiGanDiscriminatorConfig();
            if (asked is not null && asked.ToJson().ToJsonString() != saved.ToJson().ToJsonString())
            {
                throw new ArgumentException($"{folder} holds discriminators of {saved.ToJson().ToJsonString()}, and the options describe {asked.ToJson().ToJsonString()}: "
                    + "train them as they are (no AdversarialSynthesisOptions.Discriminator), or from a folder without them.");
            }

            using var reader = SafeTensorsReader.Open(Path.Combine(folder, HifiGanDiscriminators.WeightsFile));
            _discriminators = new HifiGanDiscriminators(saved, reader, _model.Device);
        }
        else if (options.DiscriminatorWeights is { } file)
        {
            // Trained discriminators to start from: Idrak's discriminator.safetensors (its discriminator.json beside it), or
            // the original VITS and MMS releases' D_*.pth under their weight_g and weight_v names (VITS's own shapes).
            if (!File.Exists(file))
            {
                throw new FileNotFoundException($"No discriminators file {file} (AdversarialSynthesisOptions.DiscriminatorWeights).", file);
            }

            string configPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(file))!, HifiGanDiscriminators.ConfigFile);
            var config = asked ?? (File.Exists(configPath) ? HifiGanDiscriminatorConfig.FromJson(JsonNode.Parse(File.ReadAllText(configPath))!.AsObject()) : new HifiGanDiscriminatorConfig());
            using ITensorStore store = PyTorchCheckpointReader.IsCheckpoint(file) ? PyTorchCheckpointReader.Open(file) : SafeTensorsReader.Open(file);
            try
            {
                _discriminators = new HifiGanDiscriminators(config, store, _model.Device);
            }
            catch (InvalidDataException e)
            {
                throw new InvalidDataException($"{file} does not hold discriminators of {config.ToJson().ToJsonString()}: {e.Message}", e);
            }
        }
        else
        {
            _discriminators = HifiGanDiscriminators.Create(asked ?? new HifiGanDiscriminatorConfig(), _model.Device, seed);
        }

        return _discriminators;
    }

    public bool Accepts(int frames, int[] labels, AdversarialSynthesisOptions options) =>
        labels.Length > 0 && frames >= labels.Length && frames >= options.SegmentFrames;

    public IAdversarialSynthesisStep Begin(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, bool keep)
    {
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        return new Step(this, batch, options, keep, Discriminators(options, batch.Seed));
    }

    public float MelError(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options)
    {
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        if (batch.Frames.Count == 0)
        {
            return 0;
        }

        using var pass = PassAll(batch, options);
        return pass.Mel.Item();
    }

    /// <summary>
    /// Writes the model as the folder it was read from looks: its files (config.json, vocab.json, the tokenizer's) copied,
    /// model.safetensors with the trained tensors under their stored names and layouts and every other tensor of the
    /// checkpoint as it was, and the discriminators (when opened) as discriminator.safetensors and discriminator.json.
    /// </summary>
    public void Save(string folder)
    {
        Directory.CreateDirectory(folder);
        string full = Path.GetFullPath(folder);
        if (_model.Folder is { } source && !string.Equals(Path.GetFullPath(source), full, StringComparison.Ordinal))
        {
            var skipped = new HashSet<string>(AdversarialSynthesisTrainer.StateFiles.Concat(TrainingFiles), StringComparer.OrdinalIgnoreCase);
            string[] weights = [".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5", ".msgpack"];
            foreach (string file in Directory.EnumerateFiles(source))
            {
                string name = Path.GetFileName(file);
                if (!skipped.Contains(name) && !weights.Contains(Path.GetExtension(name), StringComparer.OrdinalIgnoreCase) && name != "tensors.json")
                {
                    File.Copy(file, Path.Combine(folder, name), overwrite: true);
                }
            }
        }

        var trained = Generator.Named;
        var names = new HashSet<string>(trained.Select(p => p.Name), StringComparer.Ordinal);
        var rows = new List<(string, int[], float[])>(trained.Count);
        rows.AddRange(trained.Select(p => (p.Name, p.Stored, p.Values())));
        if (_model.Open is { } open)
        {
            using var store = open();
            foreach (string name in store.Names.Where(n => !names.Contains(n)).Order(StringComparer.Ordinal))
            {
                rows.Add((name, store.ShapeOf(name), store.Read(name)));       // what training does not touch, as the checkpoint stores it
            }
        }

        string target = Path.Combine(folder, "model.safetensors");
        string staging = target + ".writing";
        SafeTensorsWriter.Write(staging, rows, SafeTensorType.F32, new Dictionary<string, string> { ["format"] = "pt" });
        File.Move(staging, target, overwrite: true);
        _discriminators?.Save(folder);
    }

    public static IReadOnlyList<string> TrainingFiles { get; } = [HifiGanDiscriminators.WeightsFile, HifiGanDiscriminators.ConfigFile];

    public void Dispose()
    {
        _discriminators?.Dispose();
        foreach (var (mel, basis) in _filters.Values)
        {
            mel.Dispose();
            basis.Dispose();
        }
    }

    // One utterance's generator pass (tests): the terms before their weights, the segment's sound made [1, L] and its real sound.
    internal sealed record UtterancePass(Tensor Kl, Tensor Nll, Tensor Mel, Tensor Made, Tensor Real, int[] Durations);

    // A micro-batch's generator pass: the KL and duration terms summed over its utterances, and, for all of them as one
    // batch (equal segments), the sound made [B, L], the real sound [B, L] and the mel loss (the mean over the batch, which
    // is the mean of the utterances' means), with each utterance's durations.
    internal sealed record BatchPass(Tensor Kl, Tensor Nll, Tensor Mel, Tensor Made, Tensor Real, IReadOnlyList<int[]> Durations, PaddedSequences? TokenMask, PaddedSequences? FrameMask) : IDisposable
    {
        /// <summary>Releases the batch's masks (their tensors belong to no scope): once the pass's backward step has run.</summary>
        public void Dispose()
        {
            TokenMask?.Dispose();
            FrameMask?.Dispose();
        }
    }

    // Utterance i of the batch through the generator on its own (gradients as Autograd says).
    internal UtterancePass Pass(AdversarialSynthesisBatch batch, int i, AdversarialSynthesisOptions options)
    {
        using var pass = PassAll(batch, options, [i]);                            // one utterance: no masks to release
        return new UtterancePass(pass.Kl, pass.Nll, pass.Mel, pass.Made, pass.Real, pass.Durations[0]);
    }

    // The utterances (all of the batch, or the ones given) through the generator as one padded batch, VITS's way: the
    // text encoder, the posterior encoder and the flow with masks over the padding, the alignment search per utterance
    // on the host, the KL and duration terms as masked sums, then every utterance's segment through the decoder, the mel
    // loss and the discriminators together; every layer runs once a micro-batch. One utterance runs unpadded, unmasked.
    internal BatchPass PassAll(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, IReadOnlyList<int>? utterances = null)
    {
        using var checkpoints = batch.Checkpointing ? ActivationMemory.CheckpointBlocks() : (ActivationMemory.Scope?)null;
        utterances ??= [.. Enumerable.Range(0, batch.Frames.Count)];
        var config = _model.Config;
        var device = _model.Device;
        int count = utterances.Count, bins = Features.Bins, hop = Features.Hop, seg = options.SegmentFrames, flow = config.FlowSize, longest = batch.Features.Shape[1];
        var labels = new int[count][];
        var frames = new int[count];
        var draws = new VitsDraws[count];
        for (int b = 0; b < count; b++)
        {
            int i = utterances[b];
            labels[b] = batch.Labels[i];
            frames[b] = batch.Frames[i];
            draws[b] = Draws?.Invoke(i, batch) ?? Draw(batch, i, seg);
        }

        int tokens = labels.Max(l => l.Length), length = frames.Max();
        var tokenMask = count == 1 ? null : new PaddedSequences([.. labels.Select(l => l.Length)], tokens, device);
        var frameMask = count == 1 ? null : new PaddedSequences(frames, length, device);

        // Each utterance's training rows: its spectrogram columns into [B, bins, length] (zero after its frames), its rows kept for the segment.
        var rows = new Tensor[count];
        var spectrograms = new Tensor[count];
        for (int b = 0; b < count; b++)
        {
            var all = batch.Features.Narrow(0, utterances[b], 1).Reshape(longest, Features.Width);
            rows[b] = all.Narrow(0, 0, frames[b]);                               // [frames, bins + hop]
            spectrograms[b] = (count == 1 ? rows[b] : all.Narrow(0, 0, length)).Narrow(1, 0, bins).Reshape(1, length, bins);
        }

        var spectrogram = (count == 1 ? spectrograms[0] : Tensor.Concat(spectrograms, 0)).Permute(0, 2, 1);   // [B, bins, length]
        var (hidden, means, logStddevs) = _model.Encoder.Forward(labels, tokenMask);          // [B·T, C], [B·T, F], [B·T, F]
        var (posteriorMeans, posteriorLogs) = _model.Posterior.Stats(spectrogram, null, frameMask);   // [B, F, length]
        var posteriorNoise = new float[count * flow * length];                     // each utterance's draws over its frames, zero after
        for (int b = 0; b < count; b++)
        {
            for (int f = 0; f < flow; f++)
            {
                draws[b].Posterior.AsSpan(f * frames[b], frames[b]).CopyTo(posteriorNoise.AsSpan((b * flow + f) * length, frames[b]));
            }
        }

        var latents = posteriorMeans + Tensor.From(posteriorNoise, [count, flow, length], device) * posteriorLogs.Exp();
        var prior = _model.Flow.Forward(latents, null, frameMask);                  // [B, F, length]

        // The alignment: each utterance's frames' log-likelihood under each of its tokens' prior, the best monotonic path on the host.
        var paths = new int[count][];
        var durations = new int[count][];
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            for (int b = 0; b < count; b++)
            {
                int t = labels[b].Length;
                var meansB = means.Narrow(0, b * tokens, t);                        // [T, F]
                var logsB = logStddevs.Narrow(0, b * tokens, t);
                var priorT = prior.Narrow(0, b, 1).Reshape(flow, length).Narrow(1, 0, frames[b]).Transpose();   // [frames, F]
                var inverse = (logsB * -2f).Exp();                                  // [T, F]
                var perToken = (logsB * -1f - 0.5f * MathF.Log(2 * MathF.PI)).Sum(1) + (meansB.Square() * inverse * -0.5f).Sum(1);   // [T]
                var scores = (priorT.Square() * -0.5f).MatMul(inverse, transposeB: true) + priorT.MatMul(meansB * inverse, transposeB: true) + perToken;   // [frames, T]
                paths[b] = MonotonicAlignment.Search(scores.ToArray(), frames[b], t);
                durations[b] = MonotonicAlignment.Durations(paths[b], t);
            }
        }

        // The KL of the posterior from the prior expanded to the frames, over the frames within the utterances.
        var index = new int[count * length];
        for (int b = 0; b < count; b++)
        {
            for (int n = 0; n < length; n++)
            {
                index[b * length + n] = b * tokens + (n < frames[b] ? paths[b][n] : 0);   // the padding reads any row: it is masked
            }
        }

        var gather = VitsOps.Indices(index, device);
        var expandedMeans = means.EmbeddingLookup(gather).Reshape(count, length, flow).Permute(0, 2, 1);   // [B, F, length]
        var expandedLogs = logStddevs.EmbeddingLookup(gather).Reshape(count, length, flow).Permute(0, 2, 1);
        var klTerms = expandedLogs - posteriorLogs - 0.5f + (prior - expandedMeans).Square() * 0.5f * (expandedLogs * -2f).Exp();
        var kl = (frameMask is null ? klTerms : frameMask.Apply(klTerms)).Sum();

        // The duration predictor's loss over the tokens within the utterances (its durations padded with 1, its noise with 0).
        var durationValues = new float[count * tokens];
        var durationNoise = new float[count * 2 * tokens];
        for (int b = 0; b < count; b++)
        {
            int t = labels[b].Length;
            for (int n = 0; n < tokens; n++)
            {
                durationValues[b * tokens + n] = n < t ? durations[b][n] : 1f;
            }

            for (int c = 0; c < 2; c++)
            {
                draws[b].Duration.AsSpan(c * t, t).CopyTo(durationNoise.AsSpan((b * 2 + c) * tokens, t));
            }
        }

        var nll = _model.StochasticDurations.Loss(hidden.Reshape(count, tokens, config.HiddenSize).Permute(0, 2, 1).Detach(),
            Tensor.From(durationValues, [count, 1, tokens], device), Tensor.From(durationNoise, [count, 2, tokens], device), tokenMask);

        // Every utterance's decoder segment: its latents [F, seg], its real samples and its mel target, then all as one batch.
        var (melFilters, basis) = Filters(options);
        var segments = new Tensor[count];
        var reals = new Tensor[count];
        var targets = new Tensor[count];
        for (int b = 0; b < count; b++)
        {
            int start = draws[b].Start;
            segments[b] = (count == 1 ? latents.Reshape(flow, length) : latents.Narrow(0, b, 1).Reshape(flow, length)).Narrow(1, start, seg).Reshape(1, flow, seg);
            var segment = rows[b].Narrow(0, start, seg);
            reals[b] = segment.Narrow(1, bins, hop).Reshape(1, seg * hop);
            targets[b] = segment.Narrow(1, 0, bins).MatMul(melFilters).Clamp(1e-5f, float.PositiveInfinity).Log();   // [seg, mels]
        }

        var stacked = count == 1 ? segments[0] : Tensor.Concat(segments, 0);     // [B, F, seg]
        var made = _model.Decoder.Forward(stacked, null);                        // [B, seg · hop]
        var real = count == 1 ? reals[0] : Tensor.Concat(reals, 0);              // [B, seg · hop]
        var target = count == 1 ? targets[0] : Tensor.Concat(targets, 0);        // [B · seg, mels]
        var mel = (target - LogMel(made, basis, melFilters)).Abs().Mean();
        return new BatchPass(kl, nll, mel, made, real, durations, tokenMask, frameMask);
    }

    // The log-mel spectra [B · frames, mels] of the sound [B, L] as VITS's mel_spectrogram_torch takes it (the library's differentiable log-mel).
    internal Tensor LogMel(Tensor sound, Tensor basis, Tensor melFilters) => DifferentiableLogMel.Compute(sound, basis, melFilters, (Features.Bins - 1) * 2, Features.Hop);

    // The mel filterbank [bins, mels] (librosa's, slaney scale and norm) and the DFT filters [2 · bins, n_fft] (cosines,
    // then minus sines, of a periodic Hann window), once per device and mel setting.
    internal (Tensor Mel, Tensor Basis) Filters(AdversarialSynthesisOptions options)
    {
        var device = _model.Device;
        return _filters.GetOrAdd((device, options.Mels, options.MelMinFrequency, options.MelMaxFrequency ?? -1f), key =>
        {
            var mel = MelFilterbank.Build(Features.Options with { Mels = options.Mels, MinFrequency = options.MelMinFrequency, MaxFrequency = options.MelMaxFrequency });
            int size = Features.Options.WindowSize, bins = Features.Bins;
            var basis = DifferentiableLogMel.Basis(size, Stft.Window("hann", size));
            return (Tensor.Persistent(mel, [bins, options.Mels], device), Tensor.Persistent(basis, [2 * bins, size], device));
        });
    }

    // The seeded draws of an utterance: one generator per (seed, step, utterance), so a step's draws do not depend on how it is split.
    private VitsDraws Draw(AdversarialSynthesisBatch batch, int i, int seg)
    {
        int frames = batch.Frames[i], tokens = batch.Labels[i].Length;
        ulong hash = 14695981039346656037UL;
        foreach (int v in (int[])[batch.Seed, batch.Step, batch.Utterances[i]])
        {
            hash = (hash ^ (uint)v) * 1099511628211UL;
        }

        var random = new Random(unchecked((int)(hash ^ (hash >> 32))));
        var posterior = new float[_model.Config.FlowSize * frames];
        for (int k = 0; k < posterior.Length; k++)
        {
            posterior[k] = (float)random.NextGaussian();
        }

        var duration = new float[2 * tokens];
        for (int k = 0; k < duration.Length; k++)
        {
            duration[k] = (float)random.NextGaussian();
        }

        return new VitsDraws(posterior, duration, random.Next(frames - seg + 1));
    }

    // A micro-batch's step: the discriminators' half, then the generator's (VITS's losses, each utterance's share of the step's).
    private sealed class Step(VitsTraining owner, AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, bool keep, HifiGanDiscriminators discriminators)
        : IAdversarialSynthesisStep
    {
        private BatchPass? _kept;

        // The weight-normed filters are made once a half (WeightNorm): the discriminators' for their half (gradients
        // gathered, then handed to g and v), the generator's for its passes (kept from the discriminators' half to the
        // generator's when the passes are kept, else made again there), each dropped at its half's end.
        public float Discriminate()
        {
            WeightNorm.Bind(owner._weights.Normed);
            WeightNorm.Bind(discriminators.Normed);
            try
            {
                float total = Discriminators();
                WeightNorm.Flush(discriminators.Normed);
                return total;
            }
            finally
            {
                WeightNorm.Unbind(discriminators.Normed);
                if (!keep)
                {
                    WeightNorm.Unbind(owner._weights.Normed);
                }
            }
        }

        // The micro-batch's utterances as one batch of segments: each discriminator's mean over the batch is the mean of
        // the utterances' means (equal segments), so the batch's share of the step's loss is the utterances' shares summed.
        private float Discriminators()
        {
            using var scope = keep ? null : new TensorScope();
            BatchPass pass;
            if (keep)
            {
                pass = _kept = owner.PassAll(batch, options);
            }
            else
            {
                using (Autograd.NoGrad())
                {
                    pass = owner.PassAll(batch, options);
                }
            }

            var (real, _) = discriminators.Forward(pass.Real);
            var (made, _) = discriminators.Forward(pass.Made.Detach());
            var share = AdversarialLosses.LeastSquaresDiscriminator(real, made) * ((float)batch.Frames.Count / batch.StepUtterances);
            float total = share.Item();
            share.Backward();
            if (!keep)
            {
                pass.Dispose();                                                  // the kept pass's masks go after the generator's half
            }

            return total;
        }

        public IReadOnlyDictionary<string, float> Generate()
        {
            var terms = new Dictionary<string, float>(StringComparer.Ordinal)
            {
                ["mel"] = 0, ["kl"] = 0, ["duration"] = 0, ["adversarial"] = 0, ["feature_matching"] = 0, ["generator"] = 0,
            };
            var parameters = discriminators.Module.Parameters().ToList();
            foreach (var p in parameters)
            {
                p.RequiresGrad = false;                                          // the generator's half gives the discriminators no gradient
            }

            if (!keep || owner._weights.Normed.Any(n => !n.Bound))
            {
                WeightNorm.Bind(owner._weights.Normed);
            }

            WeightNorm.Bind(discriminators.Normed);                       // constants this half
            try
            {
                using var scope = keep ? null : new TensorScope();
                using var pass = _kept ?? owner.PassAll(batch, options);         // its masks go after the backward step below
                _kept = null;
                List<List<Tensor>> realMaps;
                using (Autograd.NoGrad())
                {
                    realMaps = discriminators.Forward(pass.Real).Maps;
                }

                var (scores, madeMaps) = discriminators.Forward(pass.Made);
                var adversarial = AdversarialLosses.LeastSquaresGenerator(scores);
                var matching = AdversarialLosses.FeatureMatching(realMaps, madeMaps);

                // The batch's means are the utterances' means averaged (equal segments): the batch's share of the step is
                // the utterances' shares summed; the KL and duration terms are each utterance's own, summed.
                float share = (float)batch.Frames.Count / batch.StepUtterances;
                var mel = pass.Mel * (options.MelWeight * share);
                var kl = pass.Kl * (options.KlWeight / batch.StepFrames);
                var duration = pass.Nll * (options.DurationWeight / batch.StepLabels);
                var adv = adversarial * (options.AdversarialWeight * share);
                var fm = matching * (options.FeatureMatchingWeight * share);
                var loss = mel + kl + duration + adv + fm;
                terms["mel"] += mel.Item();
                terms["kl"] += kl.Item();
                terms["duration"] += duration.Item();
                terms["adversarial"] += adv.Item();
                terms["feature_matching"] += fm.Item();
                terms["generator"] += loss.Item();
                loss.Backward();
            }
            finally
            {
                WeightNorm.Unbind(discriminators.Normed);
                foreach (var p in parameters)
                {
                    p.RequiresGrad = true;
                }
            }

            WeightNorm.Flush(owner._weights.Normed);
            WeightNorm.Unbind(owner._weights.Normed);

            return terms;
        }

        public void Dispose()
        {
            _kept?.Dispose();
            _kept = null;
            WeightNorm.Unbind(owner._weights.Normed);
            WeightNorm.Unbind(discriminators.Normed);
        }
    }
}
