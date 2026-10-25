// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Vision.Abstractions;

/// <summary>
/// The settings of a detection or segmentation loss (<see cref="VisionLosses"/>); each loss reads those it has and
/// ignores the rest.
/// </summary>
public sealed record VisionLossOptions
{
    /// <summary>How the losses become the value returned (default the mean).</summary>
    public LossReduction Reduction { get; init; } = LossReduction.Mean;

    /// <summary>Focal loss: the positives' weight α (default 0.25; negative for no weighting).</summary>
    public float Alpha { get; init; } = 0.25f;

    /// <summary>Focal loss: the focusing exponent γ (default 2).</summary>
    public float Gamma { get; init; } = 2f;

    /// <summary>Smooth L1: where the loss turns from squared to linear (default 1; 0 is the L1 loss).</summary>
    public float Beta { get; init; } = 1f;

    /// <summary>Dice: added to the numerator and the denominator (default 1), so empty classes score 1, not 0/0.</summary>
    public float Smooth { get; init; } = 1f;

    /// <summary>Dice: sum over the whole batch per class (true), or per sample and class (false, the default).</summary>
    public bool Batch { get; init; }

    /// <summary>Box overlap losses: keeps the divisions finite (default torchvision's 1e-7).</summary>
    public float Epsilon { get; init; } = 1e-7f;
}

/// <summary>What a vision loss scores (<see cref="IVisionLoss.Target"/>): what its predictions and targets hold.</summary>
public enum VisionLossTarget
{
    /// <summary>Corner boxes [..., 4] against corner boxes: a detector's boxes.</summary>
    Boxes,

    /// <summary>Logits against 0/1 targets of one shape: classes (a detector's, or each pixel's), scored through a sigmoid.</summary>
    Classes,

    /// <summary>Probabilities against one-hot targets [N, classes, ...]: masks (a segmenter's pixels), the logits softmaxed first.</summary>
    Masks,
}

/// <summary>
/// A detection or segmentation loss, registered in <see cref="VisionLosses"/> under its <see cref="IKind.Name"/>:
/// predictions against targets, with gradients to the predictions. What the tensors hold is its <see cref="Target"/>:
/// corner boxes [..., 4] for the box losses, logits and 0/1 targets of one shape for the focal loss, probabilities and
/// one-hot targets [N, classes, ...] for dice.
/// </summary>
public interface IVisionLoss : IKind
{
    /// <summary>What it scores, so a trainer gives it the predictions it reads (a segmenter's logits softmaxed for <see cref="VisionLossTarget.Masks"/>).</summary>
    VisionLossTarget Target { get; }

    /// <summary>The loss of <paramref name="predicted"/> against <paramref name="target"/>.</summary>
    Tensor Compute(Tensor predicted, Tensor target, VisionLossOptions options);
}

/// <summary>
/// The detection and segmentation losses, by name. The library's: "iou", "giou", "diou", "ciou" (box overlap losses on
/// corner boxes, torchvision's formulas), "l1" and "smooth-l1" (box coordinates), "focal" (sigmoid focal loss on logits)
/// and "dice" (soft dice per class on probabilities). Register another under a new name, or under a library name to
/// replace it: the library's stays behind it until <see cref="Unregister"/>.
/// </summary>
public static class VisionLosses
{
    private static readonly KindTable<IVisionLoss> Registry = BuiltIn();

    private static KindTable<IVisionLoss> BuiltIn()
    {
        var table = new KindTable<IVisionLoss>(nameof(VisionLosses), "Vision loss", "add it with VisionLosses.Register", Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new Loss("iou", "1 − IoU of corner boxes", VisionLossTarget.Boxes, (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.IoU, o.Reduction, o.Epsilon)));
        table.RegisterDefault(new Loss("giou", "1 − generalized IoU of corner boxes", VisionLossTarget.Boxes,
            (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.GIoU, o.Reduction, o.Epsilon)));
        table.RegisterDefault(new Loss("diou", "1 − distance IoU of corner boxes", VisionLossTarget.Boxes,
            (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.DIoU, o.Reduction, o.Epsilon)));
        table.RegisterDefault(new Loss("ciou", "1 − complete IoU of corner boxes", VisionLossTarget.Boxes,
            (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.CIoU, o.Reduction, o.Epsilon)));
        table.RegisterDefault(new Loss("l1", "the L1 distance of box coordinates", VisionLossTarget.Boxes, (p, t, o) => DetectionLosses.SmoothL1(p, t, 0f, o.Reduction)));
        table.RegisterDefault(new Loss("smooth-l1", "the smooth L1 distance of box coordinates (beta)", VisionLossTarget.Boxes,
            (p, t, o) => DetectionLosses.SmoothL1(p, t, o.Beta, o.Reduction)));
        table.RegisterDefault(new Loss("focal", "sigmoid focal loss on logits (alpha, gamma)", VisionLossTarget.Classes,
            (p, t, o) => DetectionLosses.Focal(p, t, o.Alpha, o.Gamma, o.Reduction)));
        table.RegisterDefault(new Loss("dice", "soft dice per class on probabilities (smooth, batch)", VisionLossTarget.Masks,
            (p, t, o) => SegmentationLosses.Dice(p, t, o.Smooth, o.Batch, o.Reduction)));
        return table;
    }

    /// <summary>Registers <paramref name="loss"/> under its name; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVisionLoss loss) => Registry.Register(loss, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's loss <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered loss names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The loss <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names those that are.</exception>
    public static IVisionLoss Get(ReadOnlySpan<char> name) => Registry.TryGet(name, out var loss) ? loss
        : throw new NotSupportedException($"No vision loss '{name}' is registered ({string.Join(", ", Registry.Names)}); add it with VisionLosses.Register.");

    /// <summary>The library's loss <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IVisionLoss? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the loss <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's loss <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the values are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The loss <paramref name="name"/> of <paramref name="predicted"/> against <paramref name="target"/>.</summary>
    public static Tensor Compute(string name, Tensor predicted, Tensor target, VisionLossOptions? options = null) => Get(name).Compute(predicted, target, options ?? new());

    // An app's loss under its policy; Shadow compares the values (the library's tensor answers, so its graph is the one trained).
    private static IVisionLoss Guard(Slot slot, IVisionLoss app, IVisionLoss library) => new Loss(app.Name, app.Summary, app.Target, (predicted, target, options) =>
        slot.Call(() => app.Compute(predicted, target, options), () => library.Compute(predicted, target, options), (a, b) =>
        {
            using (Autograd.NoGrad())
            {
                return Comparisons.Tensors(a, b, 1e-4f);
            }
        }));

    // A loss as a function and what it scores.
    private sealed class Loss(string name, string summary, VisionLossTarget scores, Func<Tensor, Tensor, VisionLossOptions, Tensor> compute) : IVisionLoss
    {
        public string Name => name;

        public string Summary => summary;

        public VisionLossTarget Target => scores;

        public Tensor Compute(Tensor predicted, Tensor target, VisionLossOptions options) => compute(predicted, target, options);
    }
}
