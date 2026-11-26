// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>The settings of a box matcher (<see cref="BoxMatchers"/>); each matcher reads those it has.</summary>
public sealed record BoxMatchOptions
{
    /// <summary>Threshold matching: a prediction whose best quality is at least this is matched to that truth (default 0.5).</summary>
    public float PositiveThreshold { get; init; } = 0.5f;

    /// <summary>
    /// Threshold matching: a prediction whose best quality is below this is a negative (default 0.4); between the two
    /// thresholds it is ignored (<see cref="BoxMatchers.Ignored"/>), neither trained as an object nor as background.
    /// </summary>
    public float NegativeThreshold { get; init; } = 0.4f;

    /// <summary>
    /// Threshold matching: every truth also keeps the predictions of its own best quality, even below the thresholds
    /// (torchvision's <c>allow_low_quality_matches</c>), so no object goes without a prediction (default false).
    /// </summary>
    public bool AllowLowQuality { get; init; }
}

/// <summary>
/// Assigns predictions (anchors, or a set predictor's outputs) to the true objects from a quality matrix
/// [predictions, truths] (row-major, higher is better: an IoU, or a negated cost), registered in <see cref="BoxMatchers"/>
/// under its <c>IKind.Name</c>. The result holds, for each prediction, the index of its truth,
/// <see cref="BoxMatchers.Negative"/> or <see cref="BoxMatchers.Ignored"/>. A quality of -∞ forbids the pair.
/// </summary>
public interface IBoxMatcher
{
    /// <summary>The assignment of each of <paramref name="predictions"/> predictions to one of <paramref name="truths"/> truths (or none).</summary>
    int[] Match(ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions options);
}

/// <summary>
/// The box matchers, by name. The library's: "iou-threshold" (each prediction to its best truth when the quality passes
/// <see cref="BoxMatchOptions.PositiveThreshold"/>, a negative below <see cref="BoxMatchOptions.NegativeThreshold"/>,
/// ignored between; torchvision's <c>Matcher</c>) and "hungarian" (one to one, the largest total quality, by the
/// Hungarian algorithm in O(n³): set prediction). Build quality matrices with <c>BoxMatching</c> (Idrak.Vision). Register
/// another under a new name, or under a library name to replace it: the library's stays behind it until <c>Unregister</c>.
/// </summary>
public static class BoxMatchers
{
    /// <summary>A prediction matched to no object: background.</summary>
    public const int Negative = -1;

    /// <summary>A prediction left out of training (between the thresholds).</summary>
    public const int Ignored = -2;

    /// <summary>"iou-threshold": threshold matching.</summary>
    public const string Threshold = "iou-threshold";

    /// <summary>"hungarian": optimal one-to-one matching.</summary>
    public const string Hungarian = "hungarian";
}
