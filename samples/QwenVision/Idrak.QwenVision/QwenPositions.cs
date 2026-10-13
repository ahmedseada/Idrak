// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's image positions (transformers' <c>get_rope_index</c>), the <c>grid</c> rule of the position-rule contract
/// plan 16 adds to the library (step 1), fed to its multi-axis rotary positions (step 2): text tokens take the same
/// position on every axis; an image's tokens take, from the position after the preceding text, the frame on the time
/// axis and their row and column in the merged grid on the height and width axes; the text after an image continues
/// from the largest position used plus one, and <c>rope_deltas</c> (that largest position + 1 − the token count)
/// shifts every decoded token's position afterwards. The library's <c>sequential</c> rule keeps every existing family
/// bit for bit; this file registers <c>grid</c> under <see cref="QwenVisionPlugin.GridPositions"/> once the contract
/// exists. Until step 1 lands there is no library type to implement, so this is the rule's description and its
/// arithmetic's home.
/// </summary>
public static class QwenPositions
{
    /// <summary>The axes of Qwen-VL's rotary positions, in the order the rotary sections take them.</summary>
    public static readonly IReadOnlyList<string> Axes = ["time", "height", "width"];

    /// <summary>
    /// The position ids of a prompt, [3, tokens] (one row per axis), and its rope delta, from the prompt's token ids and
    /// each image's merged grid in order of appearance.
    /// </summary>
    public static (int[,] Positions, int RopeDelta) Of(ReadOnlySpan<int> tokens, QwenImageTokens imageTokens, IReadOnlyList<QwenImageGrid> grids, int mergeSize) =>
        throw QwenVisionPlugin.NotBuilt("the grid position rule (the library's position-rule contract comes first)", 1);
}
