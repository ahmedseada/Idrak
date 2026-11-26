// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Abstractions;

/// <summary>How an annotation format reads a dataset (<see cref="IAnnotationFormat.Read"/>); each format reads what it needs.</summary>
public sealed record AnnotationReadOptions
{
    /// <summary>The class names, in index order, when the files do not name them (YOLO) or to fix their order; null to take them from the files.</summary>
    public IReadOnlyList<string>? Classes { get; init; }

    /// <summary>The folder the images are in, when not the format's usual place.</summary>
    public string? ImagesRoot { get; init; }
}

/// <summary>
/// A detection or segmentation dataset format (COCO, YOLO text, Pascal VOC, ...): reads its annotation files into an
/// <see cref="AnnotatedDataset"/> (images named, not decoded) and writes one back, so a dataset written and read again is
/// the same. The library's: COCO, YOLO and Pascal VOC (Idrak.Data).
/// </summary>
public interface IAnnotationFormat
{
    /// <summary>The format's name (the library's: "coco", "yolo", "voc").</summary>
    string Name { get; }

    /// <summary>What the format reads (a file or a folder, and its layout), for listings and error messages.</summary>
    string Summary { get; }

    /// <summary>
    /// What a dataset written and read back keeps (the class names, flags, areas, ids, masks it stores, and how far a box
    /// may move): what a caller may count on, and what the testing kit's annotation format suite checks.
    /// </summary>
    AnnotationFormatTraits Keeps { get; }

    /// <summary>Whether <paramref name="path"/> looks like a dataset of this format (by its name and first bytes; nothing is read in full).</summary>
    bool CanRead(string path);

    /// <summary>The dataset at <paramref name="path"/>; a malformed file is an <see cref="InvalidDataException"/> naming it.</summary>
    AnnotatedDataset Read(string path, AnnotationReadOptions? options = null);

    /// <summary>Writes <paramref name="dataset"/>'s annotations at <paramref name="path"/> (the images are not copied).</summary>
    void Write(AnnotatedDataset dataset, string path);
}

