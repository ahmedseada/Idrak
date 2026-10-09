// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>An object found in an image: its box, its class (index and name, when known) and the detector's score.</summary>
public sealed record Detection(BoundingBox Box, int Class, float Score, string? Label = null);

/// <summary>Finds objects in images. Implement it over any detection network; Idrak.Vision's <c>ModelDetector</c> is one.</summary>
public interface IObjectDetector
{
    /// <summary>The objects in <paramref name="image"/>, in its pixel coordinates.</summary>
    IReadOnlyList<Detection> Detect(ImageData image);
}

/// <summary>
/// Proposes the regions of an image to classify. <c>ComponentProposer</c> takes its connected regions;
/// implement it for other layouts (text lines split into characters, a grid of form fields, a detector's boxes).
/// </summary>
public interface IRegionProposer
{
    /// <summary>The regions of <paramref name="image"/>, in the order they should be reported.</summary>
    IReadOnlyList<PixelBox> Propose(ForegroundImage image);
}
