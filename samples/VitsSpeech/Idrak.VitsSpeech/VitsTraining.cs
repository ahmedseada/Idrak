// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;
using Idrak.Models;
using Idrak.Audio;

namespace Idrak.VitsSpeech;

/// <summary>
/// VITS's monotonic alignment search (its <c>monotonic_align</c>): the path through an utterance's [frames, tokens]
/// scores (each frame's log-likelihood under each token's prior) that visits every token in order, at least a frame each,
/// with the largest sum; on the host, in float as the Cython code adds, so its paths are VITS's exactly.
/// </summary>
internal static class MonotonicAlignment
{
    /// <summary>The token of each frame along the best path of <paramref name="scores"/> [frames, tokens] (row-major); frames are at least tokens.</summary>
    /// <exception cref="ArgumentException">Fewer frames than tokens, or the scores are not frames × tokens.</exception>
    public static int[] Search(ReadOnlySpan<float> scores, int frames, int tokens)
    {
        if (tokens < 1 || frames < tokens || scores.Length != frames * tokens)
        {
            throw new ArgumentException($"The alignment search takes [frames, tokens] scores with at least as many frames as tokens; got {scores.Length} values for {frames} x {tokens}.");
        }

        var value = scores.ToArray();
        const float Never = -1e9f;
        for (int y = 0; y < frames; y++)
        {
            int last = Math.Min(tokens, y + 1);
            for (int x = Math.Max(0, tokens + y - frames); x < last; x++)
            {
                float current = x == y ? Never : value[(y - 1) * tokens + x];
                float previous = x == 0 ? y == 0 ? 0f : Never : value[(y - 1) * tokens + x - 1];
                value[y * tokens + x] += Math.Max(previous, current);
            }
        }

        var path = new int[frames];
        int index = tokens - 1;
        for (int y = frames - 1; y >= 0; y--)
        {
            path[y] = index;
            if (index != 0 && (index == y || value[(y - 1) * tokens + index] < value[(y - 1) * tokens + index - 1]))
            {
                index--;
            }
        }

        return path;
    }

    /// <summary>Each token's frames along <paramref name="path"/> (a frame's token).</summary>
    public static int[] Durations(ReadOnlySpan<int> path, int tokens)
    {
        var durations = new int[tokens];
        foreach (int token in path)
        {
            durations[token]++;
        }

        return durations;
    }
}

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
/// route C), VITS's step on each utterance alone (no padding): the text encoder's prior, the posterior encoder on the
/// linear spectrogram, its latents drawn and taken through the flow, the monotonic alignment search on the host for the
/// durations, the KL of the posterior from the prior expanded to the frames, the stochastic duration predictor's loss,
/// and the decoder on a random segment of the latents, whose sound is scored by the mel loss and the discriminators.
/// </summary>
internal sealed class VitsTraining : IDisposable
{
    private readonly VitsSynthesizer _model;
    private readonly ConcurrentDictionary<(Device, int, float, float), (Tensor Mel, Tensor Basis)> _filters = new();
    private VitsDiscriminators? _discriminators;

    private readonly VitsWeights _weights;

    public VitsTraining(VitsSynthesizer model, VitsWeights weights)
    {
        _model = model;
        _weights = weights;
        Generator = new VitsParameterSet("VITS generator", weights.Trained);
        Features = new VitsTrainingFeatures(model.Config);
    }

    public VitsParameterSet Generator { get; }

    public VitsTrainingFeatures Features { get; }

    /// <summary>Tests: the draws of each utterance (its position in the batch, the batch) instead of the seeded ones.</summary>
    internal Func<int, AdversarialSynthesisBatch, VitsDraws>? Draws { get; set; }

    public VitsDiscriminators? Opened => _discriminators;

    public VitsDiscriminators Discriminators(AdversarialSynthesisOptions options, int seed)
    {
        if (_discriminators is not null)
        {
            return _discriminators;
        }

        var asked = options.Discriminator is { } description ? VitsDiscriminatorConfig.FromJson(description) : null;
        string? folder = _model.Folder;
        if (folder is not null && File.Exists(Path.Combine(folder, VitsDiscriminators.WeightsFile)))
        {
            string configPath = Path.Combine(folder, VitsDiscriminators.ConfigFile);
            var saved = File.Exists(configPath) ? VitsDiscriminatorConfig.FromJson(JsonNode.Parse(File.ReadAllText(configPath))!.AsObject()) : new VitsDiscriminatorConfig();
            if (asked is not null && asked.ToJson().ToJsonString() != saved.ToJson().ToJsonString())
            {
                throw new ArgumentException($"{folder} holds discriminators of {saved.ToJson().ToJsonString()}, and the options describe {asked.ToJson().ToJsonString()}: "
                    + "train them as they are (no AdversarialSynthesisOptions.Discriminator), or from a folder without them.");
            }

            using var reader = SafeTensorsReader.Open(Path.Combine(folder, VitsDiscriminators.WeightsFile));
            _discriminators = new VitsDiscriminators(saved, reader, _model.Device);
        }
        else if (options.DiscriminatorWeights is { } file)
        {
            // Trained discriminators to start from: Idrak's discriminator.safetensors (its discriminator.json beside it), or
            // the original VITS and MMS releases' D_*.pth under their weight_g and weight_v names (VITS's own shapes).
            if (!File.Exists(file))
            {
                throw new FileNotFoundException($"No discriminators file {file} (AdversarialSynthesisOptions.DiscriminatorWeights).", file);
            }

            string configPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(file))!, VitsDiscriminators.ConfigFile);
            var config = asked ?? (File.Exists(configPath) ? VitsDiscriminatorConfig.FromJson(JsonNode.Parse(File.ReadAllText(configPath))!.AsObject()) : new VitsDiscriminatorConfig());
            using ITensorStore store = PyTorchCheckpointReader.IsCheckpoint(file) ? PyTorchCheckpointReader.Open(file) : SafeTensorsReader.Open(file);
            try
            {
                _discriminators = new VitsDiscriminators(config, store, _model.Device);
            }
            catch (InvalidDataException e)
            {
                throw new InvalidDataException($"{file} does not hold discriminators of {config.ToJson().ToJsonString()}: {e.Message}", e);
            }
        }
        else
        {
            _discriminators = VitsDiscriminators.Create(asked ?? new VitsDiscriminatorConfig(), _model.Device, seed);
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
        double sum = 0;
        for (int i = 0; i < batch.Frames.Count; i++)
        {
            using var scope = new TensorScope();
            sum += Pass(batch, i, options).Mel.Item();
        }

        return (float)(sum / Math.Max(1, batch.Frames.Count));
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

    public static IReadOnlyList<string> TrainingFiles { get; } = [VitsDiscriminators.WeightsFile, VitsDiscriminators.ConfigFile];

    public void Dispose()
    {
        _discriminators?.Dispose();
        foreach (var (mel, basis) in _filters.Values)
        {
            mel.Dispose();
            basis.Dispose();
        }
    }

    // One utterance's generator pass: the terms before their weights, the segment's sound made and its real sound.
    internal sealed record UtterancePass(Tensor Kl, Tensor Nll, Tensor Mel, Tensor Made, Tensor Real, int[] Durations);

    // Utterance i of the batch through the generator (gradients as Autograd says).
    internal UtterancePass Pass(AdversarialSynthesisBatch batch, int i, AdversarialSynthesisOptions options)
    {
        var config = _model.Config;
        int frames = batch.Frames[i], bins = Features.Bins, hop = Features.Hop, seg = options.SegmentFrames, longest = batch.Features.Shape[1];
        var labels = batch.Labels[i];
        var draws = Draws?.Invoke(i, batch) ?? Draw(batch, i, seg);
        var rows = batch.Features.Narrow(0, i, 1).Reshape(longest, Features.Width).Narrow(0, 0, frames);   // [frames, bins + hop]
        var spectrogram = rows.Narrow(1, 0, bins).Transpose();                   // [bins, frames]
        var device = _model.Device;

        var (hidden, means, logStddevs) = _model.Encoder.Forward(labels);        // [T, C], [T, F], [T, F]
        var (posteriorMeans, posteriorLogs) = _model.Posterior.Stats(spectrogram, null);   // [F, frames]
        var latents = posteriorMeans + Tensor.From(draws.Posterior, [config.FlowSize, frames], device) * posteriorLogs.Exp();
        var prior = _model.Flow.Forward(latents, null);                         // [F, frames]

        // The alignment: each frame's log-likelihood under each token's prior, the best monotonic path on the host.
        int[] path;
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var inverse = (logStddevs * -2f).Exp();                              // [T, F]
            var perToken = (logStddevs * -1f - 0.5f * MathF.Log(2 * MathF.PI)).Sum(1) + (means.Square() * inverse * -0.5f).Sum(1);   // [T]
            var priorT = prior.Transpose();                                       // [frames, F]
            var scores = (priorT.Square() * -0.5f).MatMul(inverse, transposeB: true) + priorT.MatMul(means * inverse, transposeB: true) + perToken;   // [frames, T]
            path = MonotonicAlignment.Search(scores.ToArray(), frames, labels.Length);
        }

        var durations = MonotonicAlignment.Durations(path, labels.Length);
        var index = VitsOps.Indices(path, device);
        var expandedMeans = means.EmbeddingLookup(index).Transpose();            // [F, frames]
        var expandedLogs = logStddevs.EmbeddingLookup(index).Transpose();
        var kl = (expandedLogs - posteriorLogs - 0.5f + (prior - expandedMeans).Square() * 0.5f * (expandedLogs * -2f).Exp()).Sum();
        var nll = _model.StochasticDurations.Loss(hidden.Transpose().Detach(), Tensor.From([.. durations.Select(d => (float)d)], [1, labels.Length], device),
            Tensor.From(draws.Duration, [2, labels.Length], device));

        // The decoder on the segment, its log-mel against the real one's.
        var made = _model.Decoder.Forward(latents.Narrow(1, draws.Start, seg), null);   // [1, seg · hop]
        var real = rows.Narrow(0, draws.Start, seg).Narrow(1, bins, hop).Reshape(1, seg * hop);
        var (melFilters, basis) = Filters(options);
        var target = rows.Narrow(0, draws.Start, seg).Narrow(1, 0, bins).MatMul(melFilters).Clamp(1e-5f, float.PositiveInfinity).Log();   // [seg, mels]
        var mel = (target - LogMel(made, basis, melFilters)).Abs().Mean();
        return new UtterancePass(kl, nll, mel, made, real, durations);
    }

    // The log-mel spectrum [frames, mels] of the sound [1, L] as VITS's mel_spectrogram_torch takes it, differentiable:
    // reflect-padded (a gather), the windowed DFT as a convolution with its cosine and sine filters, √(re² + im² + 1e-6),
    // the filterbank, log of at least 1e-5.
    internal Tensor LogMel(Tensor sound, Tensor basis, Tensor melFilters)
    {
        int size = (Features.Bins - 1) * 2, bins = Features.Bins, hop = Features.Hop;
        var padded = VitsOps.Reflect(sound, (size - hop) / 2);
        int length = padded.Shape[1], frames = (length - size) / hop + 1;
        var spectrum = padded.Reshape(1, 1, 1, length).Convolution(basis, null, new ConvGeometry(1, 1, 1, length, 1, size, 1, hop, 0, 0)).Reshape(2 * bins, frames);
        var magnitude = (spectrum.Narrow(0, 0, bins).Square() + spectrum.Narrow(0, bins, bins).Square() + 1e-6f).Sqrt();   // [bins, frames]
        return magnitude.Transpose().MatMul(melFilters).Clamp(1e-5f, float.PositiveInfinity).Log();
    }

    // The mel filterbank [bins, mels] (librosa's, slaney scale and norm) and the DFT filters [2 · bins, n_fft] (cosines,
    // then minus sines, of a periodic Hann window), once per device and mel setting.
    internal (Tensor Mel, Tensor Basis) Filters(AdversarialSynthesisOptions options)
    {
        var device = _model.Device;
        return _filters.GetOrAdd((device, options.Mels, options.MelMinFrequency, options.MelMaxFrequency ?? -1f), key =>
        {
            var mel = MelFilterbank.Build(Features.Options with { Mels = options.Mels, MinFrequency = options.MelMinFrequency, MaxFrequency = options.MelMaxFrequency });
            int size = Features.Options.WindowSize, bins = Features.Bins;
            var window = Stft.Window("hann", size);
            var basis = new float[2 * bins * size];
            for (int k = 0; k < bins; k++)
            {
                for (int n = 0; n < size; n++)
                {
                    double angle = 2 * Math.PI * n * k / size;
                    basis[k * size + n] = (float)(window[n] * Math.Cos(angle));
                    basis[(bins + k) * size + n] = (float)(-window[n] * Math.Sin(angle));
                }
            }

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
            posterior[k] = (float)VitsNoise.Normal(random);
        }

        var duration = new float[2 * tokens];
        for (int k = 0; k < duration.Length; k++)
        {
            duration[k] = (float)VitsNoise.Normal(random);
        }

        return new VitsDraws(posterior, duration, random.Next(frames - seg + 1));
    }

    // A micro-batch's step: the discriminators' half, then the generator's (VITS's losses, each utterance's share of the step's).
    private sealed class Step(VitsTraining owner, AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, bool keep, VitsDiscriminators discriminators)
        : IAdversarialSynthesisStep
    {
        private readonly UtterancePass?[] _kept = new UtterancePass?[batch.Frames.Count];

        // The weight-normed filters are made once a half (WeightNormed): the discriminators' for their half (gradients
        // gathered, then handed to g and v), the generator's for its passes (kept from the discriminators' half to the
        // generator's when the passes are kept, else made again there), each dropped at its half's end.
        public float Discriminate()
        {
            WeightNormedLayers.Bind(owner._weights.Normed);
            WeightNormedLayers.Bind(discriminators.Normed);
            try
            {
                float total = Discriminators();
                WeightNormedLayers.Flush(discriminators.Normed);
                return total;
            }
            finally
            {
                WeightNormedLayers.Unbind(discriminators.Normed);
                if (!keep)
                {
                    WeightNormedLayers.Unbind(owner._weights.Normed);
                }
            }
        }

        private float Discriminators()
        {
            float total = 0;
            for (int i = 0; i < _kept.Length; i++)
            {
                using var scope = keep ? null : new TensorScope();
                UtterancePass pass;
                if (keep)
                {
                    pass = _kept[i] = owner.Pass(batch, i, options);
                }
                else
                {
                    using (Autograd.NoGrad())
                    {
                        pass = owner.Pass(batch, i, options);
                    }
                }

                var (real, _) = discriminators.Forward(pass.Real);
                var (made, _) = discriminators.Forward(pass.Made.Detach());
                Tensor? loss = null;
                for (int d = 0; d < real.Count; d++)
                {
                    var term = (real[d] * -1f + 1f).Square().Mean() + made[d].Square().Mean();
                    loss = loss is null ? term : loss + term;
                }

                var share = loss! * (1f / batch.StepUtterances);
                total += share.Item();
                share.Backward();
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
                WeightNormedLayers.Bind(owner._weights.Normed);
            }

            WeightNormedLayers.Bind(discriminators.Normed);                       // constants this half
            try
            {
                for (int i = 0; i < _kept.Length; i++)
                {
                    using var scope = keep ? null : new TensorScope();
                    var pass = _kept[i] ?? owner.Pass(batch, i, options);
                    _kept[i] = null;
                    List<List<Tensor>> realMaps;
                    using (Autograd.NoGrad())
                    {
                        realMaps = discriminators.Forward(pass.Real).Maps;
                    }

                    var (scores, madeMaps) = discriminators.Forward(pass.Made);
                    Tensor? adversarial = null, matching = null;
                    for (int d = 0; d < scores.Count; d++)
                    {
                        var term = (scores[d] * -1f + 1f).Square().Mean();
                        adversarial = adversarial is null ? term : adversarial + term;
                        for (int l = 0; l < madeMaps[d].Count; l++)
                        {
                            var difference = (realMaps[d][l] - madeMaps[d][l]).Abs().Mean();
                            matching = matching is null ? difference : matching + difference;
                        }
                    }

                    float perUtterance = 1f / batch.StepUtterances;
                    var mel = pass.Mel * (options.MelWeight * perUtterance);
                    var kl = pass.Kl * (options.KlWeight / batch.StepFrames);
                    var duration = pass.Nll * (options.DurationWeight / batch.StepLabels);
                    var adv = adversarial! * (options.AdversarialWeight * perUtterance);
                    var fm = matching! * (options.FeatureMatchingWeight * perUtterance);
                    var loss = mel + kl + duration + adv + fm;
                    terms["mel"] += mel.Item();
                    terms["kl"] += kl.Item();
                    terms["duration"] += duration.Item();
                    terms["adversarial"] += adv.Item();
                    terms["feature_matching"] += fm.Item();
                    terms["generator"] += loss.Item();
                    loss.Backward();
                }
            }
            finally
            {
                WeightNormedLayers.Unbind(discriminators.Normed);
                foreach (var p in parameters)
                {
                    p.RequiresGrad = true;
                }
            }

            WeightNormedLayers.Flush(owner._weights.Normed);
            WeightNormedLayers.Unbind(owner._weights.Normed);

            return terms;
        }

        public void Dispose()
        {
            Array.Clear(_kept);
            WeightNormedLayers.Unbind(owner._weights.Normed);
            WeightNormedLayers.Unbind(discriminators.Normed);
        }
    }
}
