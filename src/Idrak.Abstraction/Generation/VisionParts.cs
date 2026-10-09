// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// The vision part of a vision-language checkpoint as its family reads it (core's <c>IVisionFamily.Read</c>): everything
/// the rest of the library needs to put images into the prompt, and nothing about how the family builds its encoder. A
/// family derives from it and adds what is its own (Gemma 3's <c>Gemma3Vision</c>: SigLIP's configuration, the projector's
/// pooling, the image token ids). The loaded model carries it (<c>PretrainedModel.Vision</c>); generation and fine-tuning
/// read it.
/// </summary>
public abstract class PretrainedVision
{
    /// <summary>The family it was read by: its name in core's <c>VisionFamilies</c> (the architecture name).</summary>
    public abstract string Family { get; }

    /// <summary>The width of the image features: the text decoder's (one embedding per image token).</summary>
    public abstract int Width { get; }

    /// <summary>How the rendered prompt's image markers expand to the image tokens, with the family's token ids.</summary>
    public abstract IImagePromptFormat PromptFormat { get; }

    /// <summary>How the image tokens attend in the prompt (a rule of <see cref="ImageAttentionRules"/>, the family's choice).</summary>
    public abstract IImageAttentionRule Attention { get; }

    /// <summary>The checkpoint tensors (by their stored names) the vision part reads: they do not count as unused.</summary>
    public abstract IReadOnlyCollection<string> StoredTensors { get; }

    /// <summary>
    /// Builds the family's vision encoder (with its own preprocessing) from the checkpoint: decoded images in, each
    /// image's features out (<see cref="IVisionEncoder"/>). Dispose it when done.
    /// </summary>
    public abstract IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null);

    /// <summary>
    /// The keys of the <see cref="VisionOptions"/> this family's encoder takes (at creation,
    /// <see cref="VisionEncoderOptions.VisionOptions"/>, and per request); none unless the family says. Callers check
    /// options against them early (<see cref="VisionOptions.ThrowIfUnknown"/>); the encoder checks the values.
    /// </summary>
    public virtual IReadOnlyCollection<string> VisionOptionKeys => [];

    /// <summary>One line for <c>idrak show</c> and messages: the family's encoder and image tokens.</summary>
    public virtual string Describe() => $"{Family}, {PromptFormat.Name}, image tokens attend: {Attention.Name}";

    /// <inheritdoc />
    public override string ToString() => Describe();
}

/// <summary>How <see cref="PretrainedVision.CreateEncoder"/> builds an encoder.</summary>
public sealed record VisionEncoderOptions
{
    /// <summary>Where it runs (<see cref="Device.Default"/> when null; pass the language model's device).</summary>
    public Device? Device { get; init; }

    /// <summary>Turn images to grayscale before the family's preprocessing (some fine-tunes ask for it).</summary>
    public bool Grayscale { get; init; }

    /// <summary>
    /// The family's own options for every image the encoder reads (its keys, <see cref="PretrainedVision.VisionOptionKeys"/>),
    /// over the family's defaults from the checkpoint's files; a request's options go over these per key. Null: none.
    /// </summary>
    /// <remarks>The family refuses a key it does not take, naming those it does, when it builds the encoder.</remarks>
    public VisionOptions? VisionOptions { get; init; }

    /// <summary>
    /// How the encoder holds its projection weights: as the checkpoint stores them (the default: a bfloat16 checkpoint's
    /// weights stay bfloat16, half the memory of float32 for the same values, while inputs, outputs and sums stay
    /// float32), all float32, or bfloat16 (rounded when stored wider). A family builds its layers with core's
    /// <c>StoredWeights</c> to honour it.
    /// </summary>
    public EncoderWeights Weights { get; init; }

    /// <summary>
    /// The vision parts (<see cref="VisionTuningParts"/> names) that train, or take trained values, in this encoder: a
    /// family builds them with float32 weights whatever <see cref="Weights"/> says, and the rest as <see cref="Weights"/>
    /// says, so a trained projector over a bfloat16 checkpoint keeps its tower in bfloat16 (half the tower's weight
    /// memory). Empty (the default): every part as <see cref="Weights"/> says. A family that cannot build its parts apart
    /// may build the whole encoder in float32 when any is named.
    /// </summary>
    public IReadOnlyCollection<string> TrainedParts { get; init; } = [];

    /// <summary>Whether <see cref="TrainedParts"/> names <paramref name="part"/> (ignoring case).</summary>
    public bool Trains(string part) => TrainedParts.Contains(part, StringComparer.OrdinalIgnoreCase);
}

/// <summary>How an encoder built from a checkpoint holds its projection weights (<see cref="VisionEncoderOptions.Weights"/>).</summary>
public enum EncoderWeights
{
    /// <summary>
    /// As the checkpoint stores them (core's <c>ITensorStore.FormatOf</c>): bfloat16 weights as bfloat16, anything else
    /// as float32. The computation is the float32 one either way (the same weight values, float32 activations and sums).
    /// </summary>
    AsStored,

    /// <summary>Every weight in float32 (twice the memory of a bfloat16 checkpoint's, the same values).</summary>
    Float32,

    /// <summary>Projection weights in bfloat16, rounded to nearest when the checkpoint stores them wider.</summary>
    BFloat16,
}

/// <summary>
/// The parts of a vision-language family's image side that may train in a fine-tune, for a family that allows it: a
/// <see cref="PretrainedVision"/> implements it to offer them. A family that does not trains the language model only (its
/// tower and projector stay as loaded), and asking it for a part is an error naming that
/// (<see cref="VisionTuningParts.For"/>). The parts are named <see cref="VisionTuningParts.Projector"/> (what maps the
/// tower's output to the language model's width) and <see cref="VisionTuningParts.Tower"/> (the vision encoder itself);
/// a family offers the ones it can train.
/// <para>
/// Every method takes an encoder the family built (<see cref="PretrainedVision.CreateEncoder"/>), whose modules hold the
/// parameters, and refuses another. The forward passes record gradients (when autograd is on) and give the same values as
/// the encoder's stages (<see cref="IVisionEncoderStages"/>): <see cref="Tower"/> as <c>Tower</c>, and
/// <see cref="Features"/> of the tower's output as <c>Features</c> of the pixel values. A tuner with a frozen tower keeps
/// the tower's output (a feature cache) and runs only <see cref="Features"/> each step. A trained part is saved beside the
/// adapters by the checkpoint's own names and layout (<see cref="Export"/>, as PEFT's <c>modules_to_save</c>) and put
/// back into a new encoder by <see cref="Import"/>.
/// </para>
/// </summary>
public interface IVisionTuningPart
{
    /// <summary>The parts this family lets train, by name (<see cref="VisionTuningParts.Projector"/>, <see cref="VisionTuningParts.Tower"/>).</summary>
    IReadOnlyList<string> TrainableParts { get; }

    /// <summary>
    /// The parameters of the part <paramref name="part"/> in <paramref name="encoder"/>: leaf tensors the caller marks to
    /// receive gradients (<see cref="Tensor.RequiresGrad"/>) and gives its optimizer.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built.</exception>
    /// <exception cref="NotSupportedException">The family does not offer <paramref name="part"/> (the message names the parts it offers).</exception>
    /// <exception cref="InvalidOperationException">The part's weights are held fixed in this encoder (packed or bfloat16); the message says how to build one that trains.</exception>
    IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part);

    /// <summary>
    /// The features [images, tokens, width] of the tower's output <paramref name="towerOutput"/> (as
    /// <see cref="IVisionEncoderStages.Tower"/> or <see cref="Tower"/> gives it, on the encoder's device), through the
    /// family's projection, recording gradients.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built, or the shape is not the tower's output.</exception>
    Tensor Features(IVisionEncoder encoder, Tensor towerOutput);

    /// <summary>
    /// The tower's output for pixel values [images, channels, height, width] (as <see cref="IVisionEncoderStages.PixelValues"/>
    /// gives them, several images' values joined along the first dimension giving each image's output as alone), recording
    /// gradients: for training <see cref="VisionTuningParts.Tower"/>. While <see cref="ActivationMemory.CheckpointsBlocks"/>
    /// is on (the tuner checkpoints), the family runs the tower's blocks checkpointed (each block's output kept, the block
    /// run again in the backward pass), with the same gradients.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built.</exception>
    /// <exception cref="NotSupportedException">The family does not offer <see cref="VisionTuningParts.Tower"/>.</exception>
    Tensor Tower(IVisionEncoder encoder, Tensor pixelValues);

    /// <summary>
    /// The values of the part <paramref name="part"/> in <paramref name="encoder"/> as the checkpoint stores them: each
    /// tensor under its name in the checkpoint (one of <see cref="PretrainedVision.StoredTensors"/>) and in the
    /// checkpoint's layout, as new tensors on the CPU that the caller disposes. What a fine-tune saves beside its adapters
    /// (PEFT's <c>modules_to_save</c>), so transformers reads the trained part as it reads the checkpoint's.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built.</exception>
    /// <exception cref="NotSupportedException">The family does not offer <paramref name="part"/>, or cannot write it yet (the message says).</exception>
    /// <exception cref="InvalidOperationException">The part's weights are held packed in this encoder.</exception>
    IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part);

    /// <summary>
    /// Puts <paramref name="tensors"/> (named and laid out as <see cref="Export"/> gives them, on any device) into
    /// <paramref name="encoder"/>'s trainable parts, and returns the names it took; a name of no part it offers is left
    /// for the caller to report.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built, or a tensor's shape is not the part's.</exception>
    /// <exception cref="InvalidOperationException">The part's weights are held packed in this encoder (build it with <see cref="EncoderWeights.Float32"/>).</exception>
    IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors);

    /// <summary>
    /// The parts whose tensors <paramref name="names"/> (checkpoint names, as <see cref="Export"/> gives them) are: the
    /// parts an encoder must build with float32 weights to take those values (<see cref="VisionEncoderOptions.TrainedParts"/>).
    /// By default every part the family offers (<see cref="TrainableParts"/>); a family says which, so the others keep the
    /// checkpoint's precision.
    /// </summary>
    IReadOnlyList<string> PartsOf(IEnumerable<string> names) => TrainableParts;
}

/// <summary>
/// The names of the trainable parts of a vision family (<see cref="IVisionTuningPart"/>), and the check a tuner makes
/// before it trains any: <see cref="For"/>.
/// </summary>
public static class VisionTuningParts
{
    /// <summary>The projector: the tower's output to the language model's width (Gemma 3's soft-embedding norm and projection, LLaVA's MLP).</summary>
    public const string Projector = "projector";

    /// <summary>The vision tower (the encoder before the projector).</summary>
    public const string Tower = "tower";

    /// <summary>
    /// The frozen vision tower's output for <paramref name="image"/> under <paramref name="options"/>, as a tuner with a
    /// frozen tower keeps it (the input of <see cref="IVisionTuningPart.Features"/>): the encoder's pixel values (one per
    /// block, moved to the encoder's device when made elsewhere) through its tower (<see cref="IVisionEncoderStages"/>),
    /// without gradients, every intermediate result freed at once. Null when the encoder does not show its stages.
    /// </summary>
    /// <exception cref="ArgumentException">An option is not one the family takes, or its value is not valid.</exception>
    public static Tensor? FrozenTower(IVisionEncoder encoder, ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(image);
        if (encoder is not IVisionEncoderStages stages)
        {
            return null;
        }

        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var values = stages.PixelValues(image, options);
        if (values.Device != encoder.Device)
        {
            values = Tensor.From(values.ToArray(), values.Shape.ToArray(), encoder.Device);
        }

        return scope.Keep(stages.Tower(values));
    }

    /// <summary>The parts <paramref name="vision"/>'s family lets train; none for a family that does not implement <see cref="IVisionTuningPart"/>.</summary>
    public static IReadOnlyList<string> Offered(PretrainedVision vision)
    {
        ArgumentNullException.ThrowIfNull(vision);
        return vision is IVisionTuningPart tuning ? tuning.TrainableParts : [];
    }

    /// <summary>
    /// The family's trainable parts when <paramref name="parts"/> asks for some it offers; null when it asks for none (the
    /// language model trains alone).
    /// </summary>
    /// <param name="vision">The model's vision part.</param>
    /// <param name="parts">The parts asked for (<see cref="Projector"/>, <see cref="Tower"/>; names compare ignoring case).</param>
    /// <exception cref="NotSupportedException">
    /// A part the family does not offer: the message names the family and the parts it offers (none for a family without
    /// <see cref="IVisionTuningPart"/>, which trains the language model only).
    /// </exception>
    public static IVisionTuningPart? For(PretrainedVision vision, IEnumerable<string> parts)
    {
        ArgumentNullException.ThrowIfNull(vision);
        ArgumentNullException.ThrowIfNull(parts);
        var asked = parts.Select(p => p?.Trim() ?? "").Where(p => p.Length > 0).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        if (asked.Count == 0)
        {
            return null;
        }

        var offered = Offered(vision);
        var missing = asked.Where(p => !offered.Contains(p, StringComparer.OrdinalIgnoreCase)).ToList();
        if (missing.Count > 0)
        {
            string which = string.Join(", ", missing.Select(p => $"'{p}'"));
            throw new NotSupportedException(offered.Count == 0
                ? $"The vision family {vision.Family} offers no vision parts to train (it trains the language model only); asked for {which}."
                : $"The vision family {vision.Family} does not offer {which} for training; it offers: {string.Join(", ", offered)}.");
        }

        return (IVisionTuningPart)vision;
    }
}
