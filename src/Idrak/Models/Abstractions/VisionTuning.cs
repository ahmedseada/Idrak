// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

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
/// the tower's output (a feature cache) and runs only <see cref="Features"/> each step.
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
    /// gives them), recording gradients: for training <see cref="VisionTuningParts.Tower"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The encoder is not one this family built.</exception>
    /// <exception cref="NotSupportedException">The family does not offer <see cref="VisionTuningParts.Tower"/>.</exception>
    Tensor Tower(IVisionEncoder encoder, Tensor pixelValues);
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
