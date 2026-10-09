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
/// A frozen vision tower: each distinct image (its bytes, preparation and this encoder) runs through it once, without
/// gradients and with its intermediate results freed at once, and its output is kept in the cache: the features
/// themselves when the projector is frozen too, the tower's output when the projector trains (the projector then runs in
/// every step, with gradients). A trained tower (<see cref="VisionTuningParts.Tower"/>, full weights through the family's
/// <see cref="IVisionTuningPart.Tower"/>) runs in every step with gradients, its blocks checkpointed when the tuner
/// checkpoints; the cache then keeps the pixel values. A step's distinct images go through the tower (on cache misses)
/// and the projector together, one pass each where their shapes allow (each image's features as it gets alone; one at
/// a time when the device runs out of memory for the frozen pass). A step's features are released with the step.
/// </para>
/// <para>
/// Memory: only the trained parts are built with float32 weights (<see cref="VisionEncoderOptions.TrainedParts"/>); a
/// frozen tower keeps the checkpoint's precision.
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
    /// tensors, <see cref="PretrainedModel.CreateVisionEncoder"/>; the trained parts with float32 weights, the rest as the
    /// checkpoint stores them), every encoder parameter frozen but the trained parts', which are marked to receive gradients.
    /// </summary>
    /// <param name="model">A vision-language model (its family registered).</param>
    /// <param name="images">How every image is prepared (<see cref="TuningImages.None"/> when null).</param>
    /// <param name="parts">The vision parts that train beside the language model's adapters (<see cref="VisionTuningParts.Projector"/>, <see cref="VisionTuningParts.Tower"/>); none when null.</param>
    /// <param name="cache">Where image features are kept (<see cref="FeatureCaches"/>; not disposed by this); a new "memory" one when null (disposed with this).</param>
    /// <exception cref="InvalidOperationException">The model has no vision part.</exception>
    /// <exception cref="NotSupportedException">
    /// The model's vision family is not registered (the registry's message), or it does not offer a part asked for.
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
        var encoder = model.CreateVisionEncoder(new VisionEncoderOptions { Device = model.Device, Weights = EncoderWeights.AsStored, TrainedParts = asked });
        try
        {
            images.ThrowIfUnknown(vision.Family, vision.VisionOptionKeys);
            var tuning = VisionTuningParts.For(vision, asked);
            if ((tuning is not null || model.TrainedVisionTensors.Count > 0) && encoder is not IVisionEncoderStages)
            {
                throw new NotSupportedException($"The vision family {vision.Family}'s encoder does not show its stages (IVisionEncoderStages): the tower's output cannot be kept "
                    + "apart from the projector, so the projector cannot train with a cached tower, nor the tower from pixel values.");
            }

            // Nothing of the encoder records gradients but the trained parts (a frozen projector between a trained tower and
            // the decoder passes gradients through without keeping its own).
            if (encoder is Module module)
            {
                foreach (var parameter in module.Parameters())
                {
                    parameter.RequiresGrad = false;
                }
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

    /// <summary>Whether the vision tower trains (<see cref="VisionTuningParts.Tower"/> among <see cref="Parts"/>): it then runs in every step.</summary>
    public bool TrainsTower => Parts.Contains(VisionTuningParts.Tower);

    /// <summary>
    /// The stage the cache keeps: the pixel values when the tower trains (it runs in every step); the tower's output when
    /// the projector trains or the model carries trained vision tensors (the projector then runs in every step); else the
    /// features.
    /// </summary>
    public string CachedStage => TrainsTower ? FeatureCacheKey.PixelsStage : _projector is not null ? FeatureCacheKey.TowerStage : FeatureCacheKey.FeaturesStage;

    /// <summary>
    /// The time the last batch's images took (their cached stage from the cache or computed, then the trained or carried
    /// parts), and how many went through one pass together; for the tuner's trace.
    /// </summary>
    internal (double Milliseconds, int Images, int Passes) LastFeatures { get; private set; }

    // The features of a batch's distinct images for a step, each [its tokens (all blocks, in order), width] on the
    // encoder's device: the cached stage (from the cache, else computed without gradients, the misses together), then the
    // tower (when it trains, its blocks checkpointed with `checkpointing`) and the projector (when it trains or carries
    // trained values) over all the images joined, with gradients when a part trains. Made in the caller's tensor scope
    // (released with the step).
    internal IReadOnlyList<Tensor> Features(IReadOnlyList<TrainingImage> images, bool checkpointing = false)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var kept = new Tensor[images.Count];
        var missing = new List<int>();
        for (int i = 0; i < images.Count; i++)
        {
            if (Cache.TryGet(Key(images[i].Image, images[i].Preparation, CachedStage), Encoder.Device, out var cached))
            {
                kept[i] = cached;
                Interlocked.Increment(ref _hits);
            }
            else
            {
                missing.Add(i);
            }
        }

        int passes = 0;
        if (missing.Count > 0)
        {
            Interlocked.Add(ref _encoded, missing.Count);
            var computed = Compute([.. missing.Select(i => images[i])], ref passes);
            for (int k = 0; k < missing.Count; k++)
            {
                kept[missing[k]] = computed[k];
                Cache.Put(Key(images[missing[k]].Image, images[missing[k]].Preparation, CachedStage), computed[k]);
            }
        }

        IReadOnlyList<Tensor> features = kept;
        using (Parameters.Count == 0 ? Autograd.NoGrad() : (IDisposable?)null)            // nothing trains on the image side
        {
            if (TrainsTower)
            {
                using var blocks = checkpointing ? ActivationMemory.CheckpointBlocks() : (ActivationMemory.Scope?)null;
                features = Joined(kept, x => Tuning!.Tower(Encoder, x), ref passes);
            }

            if (_projector is not null)
            {
                features = Joined(features, x => _projector.Features(Encoder, x), ref passes);   // [blocks, tokens, width] each
            }
        }

        var result = new Tensor[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            var f = features[i];
            if (f.Rank == 3)
            {
                f = f.Reshape(f.Shape[0] * f.Shape[1], f.Shape[2]);
            }

            if (f.Rank != 2 || f.Shape[0] < images[i].Tokens || f.Shape[1] != Vision.Width)
            {
                throw new InvalidOperationException($"The features of {images[i].Image} are {Tensor.FormatShape(f.Shape)}; its blocks take {images[i].Tokens} tokens of width {Vision.Width}.");
            }

            result[i] = f;
        }

        LastFeatures = (watch.Elapsed.TotalMilliseconds, images.Count, passes);
        return result;
    }

    // `pass` over the tensors joined along their first dimension (one pass for all, when their other dimensions agree and
    // there are several), split back into each tensor's rows; one pass each otherwise. Gradients flow through the join.
    private static IReadOnlyList<Tensor> Joined(IReadOnlyList<Tensor> inputs, Func<Tensor, Tensor> pass, ref int passes)
    {
        if (inputs.Count > 1 && inputs.All(x => x.Rank == inputs[0].Rank && x.Shape[1..].SequenceEqual(inputs[0].Shape[1..])))
        {
            var output = pass(Tensor.Concat(inputs, 0));
            passes++;
            var parts = new Tensor[inputs.Count];
            int at = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                parts[i] = output.Narrow(0, at, inputs[i].Shape[0]);
                at += inputs[i].Shape[0];
            }

            if (at != output.Shape[0])
            {
                throw new InvalidOperationException($"A joined pass over {at} rows gave {output.Shape[0]}.");
            }

            return parts;
        }

        passes += inputs.Count;
        return [.. inputs.Select(pass)];
    }

    // The cached stage of images, without gradients (every intermediate result freed at once): the pixel values (tower
    // trained), the tower's output [blocks, ...] (projector run in the step), or the features [tokens, width]. The images'
    // pixel values go through the family's stages in one pass when their shapes agree; one at a time when the device runs
    // out of memory for that pass (or the encoder shows no stages).
    private List<Tensor> Compute(IReadOnlyList<TrainingImage> images, ref int passes)
    {
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var stages = Encoder as IVisionEncoderStages;
        if (stages is null || !_positionsChecked)
        {
            // The first image through the encoder's own Encode: a family that places image tokens by several axes (M-RoPE)
            // says so there, and the tuner's decoder does not.
            var first = images[0];
            var blocks = Encoder.Encode([Read(first.Image, first.Preparation)], Options(first.Preparation));
            if (blocks.Any(b => b.Positions is not null))
            {
                throw new NotSupportedException($"The vision family {Vision.Family} places image tokens by several axes (M-RoPE), which the tuner's decoder does not.");
            }

            _positionsChecked = true;
            if (stages is null)
            {
                // No stages: each image through Encode (only the features can be kept).
                var all = new List<Tensor> { blocks.Count == 1 ? blocks[0].Features : Tensor.Concat([.. blocks.Select(b => b.Features)], 0) };
                foreach (var image in images.Skip(1))
                {
                    var more = Encoder.Encode([Read(image.Image, image.Preparation)], Options(image.Preparation));
                    all.Add(more.Count == 1 ? more[0].Features : Tensor.Concat([.. more.Select(b => b.Features)], 0));
                }

                passes += images.Count;
                return [.. all.Select(scope.Keep)];
            }
        }

        var pixels = images.Select(i => OnDevice(stages.PixelValues(Read(i.Image, i.Preparation), Options(i.Preparation)))).ToList();
        if (TrainsTower)
        {
            return [.. pixels.Select(scope.Keep)];
        }

        Func<Tensor, Tensor> pass = _projector is not null ? stages.Tower : stages.Features;
        IReadOnlyList<Tensor> outputs;
        int counted = passes;
        try
        {
            outputs = Joined(pixels, pass, ref counted);
        }
        catch (ResourceLimitExceededException) when (pixels.Count > 1)
        {
            counted = passes;
            outputs = Joined([pixels[0]], pass, ref counted);           // the device's memory decides: one image at a time
            outputs = [.. outputs, .. pixels.Skip(1).Select(p => Joined([p], pass, ref counted)[0])];
        }

        passes = counted;
        return [.. outputs.Select(o => scope.Keep(_projector is not null || o.Rank != 3 ? o : o.Reshape(o.Shape[0] * o.Shape[1], o.Shape[2])))];
    }

    private bool _positionsChecked;

    // Pixel values on the encoder's device (a family may make them on the host).
    private Tensor OnDevice(Tensor values) => values.Device == Encoder.Device ? values : Tensor.From(values.ToArray(), values.Shape.ToArray(), Encoder.Device);

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
            Images.Save(folder);
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
