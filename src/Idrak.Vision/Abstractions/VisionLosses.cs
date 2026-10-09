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

/// <summary>
/// A detection or segmentation loss: predictions against targets, with gradients to the predictions. What the tensors hold
/// depends on the loss: corner boxes [..., 4] for the box losses, logits and 0/1 targets of one shape for the focal loss,
/// probabilities and one-hot targets [N, classes, ...] for dice.
/// </summary>
public delegate Tensor VisionLoss(Tensor predicted, Tensor target, VisionLossOptions options);

/// <summary>
/// The detection and segmentation losses, by name. The library's: "iou", "giou", "diou", "ciou" (box overlap losses on
/// corner boxes, torchvision's formulas), "l1" and "smooth-l1" (box coordinates), "focal" (sigmoid focal loss on logits)
/// and "dice" (soft dice per class on probabilities). Register another under a new name, or under a library name to
/// replace it: the library's stays behind it until <see cref="Unregister"/>.
/// </summary>
public static class VisionLosses
{
    private static readonly SlotTable<string, VisionLoss> Registry = BuiltIn();

    private static SlotTable<string, VisionLoss> BuiltIn()
    {
        var table = new SlotTable<string, VisionLoss>(nameof(VisionLosses), Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault("iou", (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.IoU, o.Reduction, o.Epsilon));
        table.RegisterDefault("giou", (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.GIoU, o.Reduction, o.Epsilon));
        table.RegisterDefault("diou", (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.DIoU, o.Reduction, o.Epsilon));
        table.RegisterDefault("ciou", (p, t, o) => DetectionLosses.BoxIou(p, t, BoxOverlap.CIoU, o.Reduction, o.Epsilon));
        table.RegisterDefault("l1", (p, t, o) => DetectionLosses.SmoothL1(p, t, 0f, o.Reduction));
        table.RegisterDefault("smooth-l1", (p, t, o) => DetectionLosses.SmoothL1(p, t, o.Beta, o.Reduction));
        table.RegisterDefault("focal", (p, t, o) => DetectionLosses.Focal(p, t, o.Alpha, o.Gamma, o.Reduction));
        table.RegisterDefault("dice", (p, t, o) => SegmentationLosses.Dice(p, t, o.Smooth, o.Batch, o.Reduction));
        return table;
    }

    /// <summary>Registers the loss <paramref name="name"/>; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, VisionLoss loss)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(loss);
        Registry.Register(name, loss, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's loss <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered loss names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The loss <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names those that are.</exception>
    public static VisionLoss Get(string name) => Registry.TryGet(name, out var loss) ? loss
        : throw new NotSupportedException($"No vision loss '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with VisionLosses.Register.");

    /// <summary>The library's loss <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static VisionLoss? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the loss <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's loss <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the values are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The loss <paramref name="name"/> of <paramref name="predicted"/> against <paramref name="target"/>.</summary>
    public static Tensor Compute(string name, Tensor predicted, Tensor target, VisionLossOptions? options = null) => Get(name)(predicted, target, options ?? new());

    // An app's loss under its policy; Shadow compares the values (the library's tensor answers, so its graph is the one trained).
    private static VisionLoss Guard(Slot slot, VisionLoss app, VisionLoss library) => (predicted, target, options) =>
        slot.Call(() => app(predicted, target, options), () => library(predicted, target, options), (a, b) =>
        {
            using (Autograd.NoGrad())
            {
                return Comparisons.Tensors(a, b, 1e-4f);
            }
        });
}
