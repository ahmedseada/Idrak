// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using Idrak.Data;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

/// <summary>
/// The image side of a vision-language fine-tune, all of it from the model's vision family (<c>PretrainedModel.Vision</c>,
/// never a family of its own): the family's encoder on the model's device, how the prompt holds an image
/// (<see cref="Format"/>) and how image tokens attend (<see cref="Attention"/>), how images are prepared
/// (<see cref="Images"/>), which vision parts train (<see cref="Parts"/>, through the family's
/// <see cref="IVisionTuningPart"/>) and where image features are kept between steps and epochs (<see cref="Cache"/>).
/// <see cref="ChatTranscriptEncoder.Vision"/> encodes transcripts with images through it, and
/// <see cref="FineTuningOptions.Vision"/> trains on them.
/// <para>
/// The vision tower is frozen: each distinct image (its bytes, preparation and this encoder) runs through it once, without
/// gradients and with its intermediate results freed at once, and its output is kept in the cache: the features
/// themselves when the projector is frozen too, the tower's output when the projector trains (the projector then runs in
/// every step, with gradients). A step's features are released with the step.
/// </para>
/// </summary>
/// <example>
/// <code>
/// using var vision = TuningVision.Create(model, TuningImages.Parse("max_width=1024"), parts: ["projector"]);
/// var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Vision = vision };
/// var train = transcripts.Select(t => encoder.Encode(t, 2048)).OfType&lt;TrainingSequence&gt;().ToList();
/// FineTuner.Train(model, train, null, new FineTuningOptions { Vision = vision }, "adapter");   // adapters, projector, tuning_images.json
/// </code>
/// </example>
public sealed class TuningVision : IDisposable
{
    private readonly ConcurrentDictionary<(string Image, TuningImages Preparation), IReadOnlyList<ImageTokenLayout>> _blocks = new();
    private readonly bool _ownsCache;
    private readonly IVisionTuningPart? _projector;              // runs each step on the cached tower output, when set
    private long _hits, _encoded;

    private TuningVision(PretrainedModel model, IVisionEncoder encoder, TuningImages images, IReadOnlyList<string> parts, IVisionTuningPart? tuning,
        IReadOnlyList<Tensor> parameters, IFeatureCache cache, bool ownsCache, string checkpoint)
    {
        (Model, Vision, Encoder, Images, Parts, Tuning, Parameters, Cache, _ownsCache, Checkpoint) =
            (model, model.Vision!, encoder, images, parts, tuning, parameters, cache, ownsCache, checkpoint);
        _projector = tuning ?? (model.TrainedVisionTensors.Count > 0 ? Vision as IVisionTuningPart : null);
    }

    /// <summary>
    /// The image side of fine-tuning <paramref name="model"/>: its family checked first (registered, the vision options
    /// it takes, the parts it offers), then its encoder built on the model's device (with the model's trained vision
    /// tensors, <see cref="PretrainedModel.CreateVisionEncoder"/>; float32 weights when a part trains), the trained parts'
    /// parameters marked to receive gradients.
    /// </summary>
    /// <param name="model">A vision-language model (its family registered).</param>
    /// <param name="images">How every image is prepared (<see cref="TuningImages.None"/> when null).</param>
    /// <param name="parts">The vision parts that train beside the language model's adapters (<see cref="VisionTuningParts.Projector"/>); none when null.</param>
    /// <param name="cache">Where image features are kept (<see cref="FeatureCaches"/>; not disposed by this); a new "memory" one when null (disposed with this).</param>
    /// <exception cref="InvalidOperationException">The model has no vision part.</exception>
    /// <exception cref="NotSupportedException">
    /// The model's vision family is not registered (the registry's message), it does not offer a part asked for, or the
    /// tower is asked for (it trains in the step: plan 12, phase 6).
    /// </exception>
    /// <exception cref="ArgumentException">A vision option the family does not take.</exception>
    public static TuningVision Create(PretrainedModel model, TuningImages? images = null, IEnumerable<string>? parts = null, IFeatureCache? cache = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        var vision = model.Vision ?? throw new InvalidOperationException(
            "The model has no vision part (a text-only checkpoint): it cannot fine-tune on images. Use a vision-language checkpoint, or data without images.");
        images ??= TuningImages.None;
        var asked = (parts ?? []).Select(p => p?.Trim().ToLowerInvariant() ?? "").Where(p => p.Length > 0).Distinct(StringComparer.Ordinal).ToList();

        // The encoder first: a family no longer registered fails here, with the registry's message.
        var encoder = model.CreateVisionEncoder(new VisionEncoderOptions { Device = model.Device, Weights = asked.Count > 0 ? EncoderWeights.Float32 : EncoderWeights.AsStored });
        try
        {
            images.ThrowIfUnknown(vision.Family, vision.VisionOptionKeys);
            var tuning = VisionTuningParts.For(vision, asked);
            if (asked.FirstOrDefault(p => p != VisionTuningParts.Projector) is { } other)
            {
                throw new NotSupportedException($"Training the vision {other} needs it in every step, with no cached features (plan 12, phase 6, not built yet); "
                    + $"train the {VisionTuningParts.Projector} or the language model only.");
            }

            if ((tuning is not null || model.TrainedVisionTensors.Count > 0) && encoder is not IVisionEncoderStages)
            {
                throw new NotSupportedException($"The vision family {vision.Family}'s encoder does not show its stages (IVisionEncoderStages): the tower's output cannot be kept "
                    + "apart from the projector, so the projector cannot train with a cached tower.");
            }

            var parameters = new List<Tensor>();
            foreach (string part in asked)
            {
                foreach (var parameter in tuning!.Parameters(encoder, part))
                {
                    parameter.RequiresGrad = true;
                    parameters.Add(parameter);
                }
            }

            bool owns = cache is null;
            return new TuningVision(model, encoder, images, asked, tuning, parameters, cache ?? FeatureCaches.Create(FeatureCaches.Memory), owns, CheckpointIdentity(model));
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>The model.</summary>
    public PretrainedModel Model { get; }

    /// <summary>The model's vision part (its family's registration).</summary>
    public PretrainedVision Vision { get; }

    /// <summary>The family's encoder, on the model's device (owned: disposed with this).</summary>
    public IVisionEncoder Encoder { get; }

    /// <summary>How the rendered prompt's image markers expand (the family's).</summary>
    public IImagePromptFormat Format => Vision.PromptFormat;

    /// <summary>How image tokens attend (the family's rule).</summary>
    public IImageAttentionRule Attention => Vision.Attention;

    /// <summary>How images are prepared, for the transcripts encoded through this (saved next to the adapters).</summary>
    public TuningImages Images { get; }

    /// <summary>The vision parts that train (lowercase names; empty: the language model trains alone).</summary>
    public IReadOnlyList<string> Parts { get; }

    /// <summary>The family's trainable parts, when <see cref="Parts"/> is not empty.</summary>
    public IVisionTuningPart? Tuning { get; }

    /// <summary>The trained parts' parameters (marked to receive gradients), for the optimizer beside the adapters.</summary>
    public IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>Where image features are kept between steps and epochs.</summary>
    public IFeatureCache Cache { get; }

    /// <summary>What identifies the checkpoint's vision weights in the cache's keys (a hash of the weight files and any trained vision tensors).</summary>
    public string Checkpoint { get; }

    /// <summary>Image features taken from <see cref="Cache"/> so far.</summary>
    public long CacheHits => Interlocked.Read(ref _hits);

    /// <summary>Images run through the tower so far (each distinct image once while the cache keeps it).</summary>
    public long Encoded => Interlocked.Read(ref _encoded);

    /// <summary>
    /// <paramref name="image"/>'s pixels as the encoder reads them: decoded (<see cref="ChatImageDecoder.Decode(ChatImage)"/>,
    /// EXIF orientation applied), then through <paramref name="preparation"/>'s transforms (<see cref="Images"/>' when null).
    /// </summary>
    public ImageData Read(ChatImage image, TuningImages? preparation = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var pipeline = (preparation ?? Images).Pipeline;
        var decoded = ChatImageDecoder.Decode(image);
        return pipeline.IsEmpty ? decoded : pipeline.Apply(decoded);
    }

    /// <summary>
    /// The blocks of image tokens <paramref name="image"/> becomes under <paramref name="preparation"/> (<see cref="Images"/>
    /// when null): the encoder's <see cref="IVisionEncoder.Blocks"/> of the prepared pixels, with the preparation's vision
    /// options. Kept per image and preparation (an image repeated across records is decoded once); the pixels are not kept.
    /// </summary>
    public IReadOnlyList<ImageTokenLayout> Blocks(ChatImage image, TuningImages? preparation = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var prepared = preparation ?? Images;
        return _blocks.GetOrAdd((image.Hash, prepared), _ => Encoder.Blocks(Read(image, prepared), Options(prepared)));
    }

    /// <summary>The cache key of <paramref name="image"/>'s output at <paramref name="stage"/> under <paramref name="preparation"/>, for this encoder.</summary>
    public FeatureCacheKey Key(ChatImage image, TuningImages preparation, string stage) =>
        FeatureCacheKey.For(image, preparation, Vision.Family, Checkpoint, "float32", stage);

    /// <summary>
    /// The stage the cache keeps: the tower's output when the projector trains or the model carries trained vision tensors
    /// (the projector then runs in every step), else the features.
    /// </summary>
    public string CachedStage => _projector is not null ? FeatureCacheKey.TowerStage : FeatureCacheKey.FeaturesStage;

    // The image's features for a step, [its tokens (all blocks, in order), width] on the encoder's device: the cached stage
    // (computed once, without gradients), then the projector with gradients when it trains. Made in the caller's tensor
    // scope (released with the step).
    internal Tensor Features(TrainingImage image, out bool hit)
    {
        var prepared = image.Preparation;
        var key = Key(image.Image, prepared, CachedStage);
        Tensor kept;
        if (Cache.TryGet(key, Encoder.Device, out var cached))
        {
            (hit, kept) = (true, cached);
            Interlocked.Increment(ref _hits);
        }
        else
        {
            hit = false;
            Interlocked.Increment(ref _encoded);
            kept = Compute(image.Image, prepared);
            Cache.Put(key, kept);
        }

        Tensor features = kept;
        if (_projector is not null)
        {
            var projected = _projector.Features(Encoder, kept);                          // [blocks, tokens, width], gradients reach the projector
            features = projected.Reshape(projected.Shape[0] * projected.Shape[1], projected.Shape[2]);
        }

        if (features.Rank != 2 || features.Shape[0] < image.Tokens || features.Shape[1] != Vision.Width)
        {
            throw new InvalidOperationException($"The features of {image.Image} are {Tensor.FormatShape(features.Shape)}; its blocks take {image.Tokens} tokens of width {Vision.Width}.");
        }

        return features;
    }

    // The cached stage of one image: the tower's output [blocks, ...] (projector trained) or the features [tokens, width],
    // without gradients; every intermediate result (pixels, the tower's activations) freed at once.
    private Tensor Compute(ChatImage image, TuningImages preparation)
    {
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var pixels = Read(image, preparation);
        var options = Options(preparation);
        if (_projector is not null)
        {
            return scope.Keep(VisionTuningParts.FrozenTower(Encoder, pixels, options)!);
        }

        var blocks = Encoder.Encode([pixels], options);
        if (blocks.Any(b => b.Positions is not null))
        {
            throw new NotSupportedException($"The vision family {Vision.Family} places image tokens by several axes (M-RoPE), which the tuner's decoder does not.");
        }

        return scope.Keep(blocks.Count == 1 ? blocks[0].Features : Tensor.Concat([.. blocks.Select(b => b.Features)], 0));
    }

    // Writes the trained parts (as the model's trained vision tensors) and the preparation beside the adapters.
    internal void Save(string folder, bool images)
    {
        if (Parts.Count > 0)
        {
            Model.KeepTrainedVision(Encoder, Parts);
        }

        Model.SaveAdapter(folder);
        if (images)
        {
            (Images with { Family = Vision.Family }).Save(folder);              // whose vision options they are: run checks the model's family
        }
    }

    private static VisionOptions? Options(TuningImages preparation) => preparation.VisionOptions.Count == 0 ? null : preparation.VisionOptions;

    // The checkpoint's identity for cache keys: the vision family, the weight files (names, sizes, times) and config.json,
    // and the names of the model's trained vision tensors.
    private static string CheckpointIdentity(PretrainedModel model)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        void Add(string text) => hash.AppendData(Encoding.UTF8.GetBytes(text + "\n"));
        Add(model.Vision!.Family);
        if (Directory.Exists(model.Folder))
        {
            foreach (var file in new DirectoryInfo(model.Folder).GetFiles().Where(f => f.Extension is ".safetensors" or ".gguf" or ".bin" or ".json")
                         .OrderBy(f => f.Name, StringComparer.Ordinal))
            {
                Add($"{file.Name} {file.Length} {file.LastWriteTimeUtc.Ticks}");
            }
        }
        else if (File.Exists(model.Folder))
        {
            var file = new FileInfo(model.Folder);
            Add($"{file.FullName} {file.Length} {file.LastWriteTimeUtc.Ticks}");
        }

        // Trained vision tensors (a fine-tuned projector) are not in the files: with them, the cache keeps the tower's output,
        // which they do not change (CachedStage); their names still tell such a run's keys apart.
        foreach (string name in model.TrainedVisionTensors.Order(StringComparer.Ordinal))
        {
            Add("trained " + name);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>Disposes the encoder, and the cache when this made it.</summary>
    public void Dispose()
    {
        Encoder.Dispose();
        if (_ownsCache)
        {
            Cache.Dispose();
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"{Vision.Family}: {Images}; {(Parts.Count == 0 ? "the language model trains alone" : $"trains {string.Join(", ", Parts)}")}; {Cache.Name} feature cache ({CachedStage})";
}
