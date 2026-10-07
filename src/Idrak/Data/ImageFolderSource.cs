// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// Images as samples, decoded when read (with the registered <see cref="ImageCodecs"/>) and resized to one
/// [channels, height, width] shape with values in [0, 1]. The targets are one-hot rows over <see cref="Classes"/>
/// (none for unlabelled images, to predict). The usual layout is a folder per class, read by the folder constructor:
/// <c>folder/cat/*.png</c>, <c>folder/dog/*.png</c>.
/// </summary>
/// <example>
/// <code>
/// var images = new ImageFolderSource("shapes", channels: 1, height: 28, width: 28);
/// var (train, test) = images.Split(0.8, seed: 1);
/// var loader = new DataLoader(train, 64, shuffle: true, seed: 2) { Transforms = [new RandomShift(2), new RandomRotation(10)] };
/// </code>
/// </example>
public sealed class ImageFolderSource : ISampleSource
{
    private readonly string[] _files;
    private readonly int[]? _labels;
    private readonly int[] _featureShape, _targetShape;

    /// <summary>
    /// Every sub-folder of <paramref name="folder"/> is a class, in ordinal name order, and its images (sub-folders
    /// included, ordinal path order) are its samples; files without a registered codec's extension are skipped.
    /// </summary>
    /// <exception cref="DirectoryNotFoundException">The folder does not exist.</exception>
    public ImageFolderSource(string folder, int channels, int height, int width)
        : this(Scan(folder, out var classes), classes, channels, height, width)
    {
        Folder = folder;
    }

    /// <summary>Images listed by path with their class indices into <paramref name="classes"/>.</summary>
    public ImageFolderSource(IReadOnlyList<(string Path, int Label)> files, IReadOnlyList<string> classes, int channels, int height, int width)
        : this([.. files.Select(f => f.Path)], [.. files.Select(f => f.Label)], classes, channels, height, width)
    {
    }

    /// <summary>Unlabelled images (no targets), e.g. to predict.</summary>
    public ImageFolderSource(IReadOnlyList<string> files, int channels, int height, int width)
        : this([.. files], null, [], channels, height, width)
    {
    }

    private ImageFolderSource(string[] files, int[]? labels, IReadOnlyList<string> classes, int channels, int height, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        for (int i = 0; labels is not null && i < labels.Length; i++)
        {
            if ((uint)labels[i] >= (uint)classes.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(labels), $"{files[i]}: class {labels[i]} is outside the {classes.Count} classes.");
            }
        }

        _files = files;
        _labels = labels;
        Classes = [.. classes];
        _featureShape = [channels, height, width];
        _targetShape = [labels is null ? 0 : classes.Count];
    }

    /// <summary>
    /// The folder constructor with each size that is not given taken from the first image: its height and width, and
    /// 1 channel for grey or 3 for colour.
    /// </summary>
    public static ImageFolderSource Open(string folder, int? channels = null, int? height = null, int? width = null)
    {
        var files = Scan(folder, out var classes);
        if (channels is null || height is null || width is null)
        {
            var first = files.Count > 0 ? ImageCodecs.ReadInfo(files[0].Path) : null;
            if (first is not { } info)
            {
                throw new InvalidDataException(files.Count == 0
                    ? $"{folder} has no images in class folders ({string.Join(", ", ImageCodecs.Extensions)})."
                    : $"{files[0].Path}: no registered image codec reads it ({string.Join(", ", ImageCodecs.Names)}).");
            }

            channels ??= info.Channels;
            height ??= info.Height;
            width ??= info.Width;
        }

        return new ImageFolderSource(files, classes, channels.Value, height.Value, width.Value) { Folder = folder };
    }

    /// <summary>The folder read by the folder constructor, or null.</summary>
    public string? Folder { get; private init; }

    /// <summary>The class names (the folder names), in label order; empty for unlabelled images.</summary>
    public IReadOnlyList<string> Classes { get; }

    /// <summary>The image files, in sample order.</summary>
    public IReadOnlyList<string> Files => _files;

    /// <summary>The class index of each sample, or null for unlabelled images.</summary>
    public IReadOnlyList<int>? Labels => _labels;

    /// <inheritdoc />
    public int Count => _files.Length;

    /// <summary>[channels, height, width].</summary>
    public IReadOnlyList<int> FeatureShape => _featureShape;

    /// <summary>[classes]: one-hot rows ([0] for unlabelled images).</summary>
    public IReadOnlyList<int> TargetShape => _targetShape;

    /// <summary>Decodes and resizes image <paramref name="index"/>, and writes its one-hot class.</summary>
    /// <exception cref="InvalidDataException">The file is not an image a registered codec reads; the message names it.</exception>
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        ImageCodecs.Decode(_files[index]).Resize(_featureShape[0], _featureShape[1], _featureShape[2], features);
        if (_labels is not null)
        {
            targets.Clear();
            targets[_labels[index]] = 1f;
        }
    }

    // Class folders in ordinal order and their image files (recursively, ordinal path order), labelled by folder.
    private static List<(string Path, int Label)> Scan(string folder, out List<string> classes)
    {
        if (!Directory.Exists(folder))
        {
            throw new DirectoryNotFoundException($"{folder} is not a folder.");
        }

        var folders = Directory.GetDirectories(folder).Order(StringComparer.Ordinal).ToList();
        classes = [.. folders.Select(Path.GetFileName).OfType<string>()];
        return [.. folders.SelectMany((d, label) => Directory.EnumerateFiles(d, "*", SearchOption.AllDirectories)
            .Where(ImageCodecs.CanDecode).Order(StringComparer.Ordinal).Select(f => (f, label)))];
    }
}
