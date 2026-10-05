// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>Which neighbours connect pixels into one region.</summary>
public enum Connectivity
{
    /// <summary>Left, right, up and down: pixels touching only at a corner are separate.</summary>
    Four = 4,

    /// <summary>The eight around a pixel: diagonal neighbours join too.</summary>
    Eight = 8,
}

/// <summary>A connected region of foreground: its label in <see cref="RegionMap.Labels"/>, box, pixel count and centre of mass.</summary>
public sealed record Region(int Label, PixelBox Box, int Area, float CenterX, float CenterY);

/// <summary>
/// The connected regions of an image's foreground: <see cref="Regions"/> in raster order (by their first pixel, top to
/// bottom, left to right) and a label per pixel (0 for background, otherwise the region's <see cref="Region.Label"/>).
/// </summary>
public sealed class RegionMap
{
    private readonly int[] _labels;

    internal RegionMap(int width, int height, int[] labels, IReadOnlyList<Region> regions)
    {
        Width = width;
        Height = height;
        _labels = labels;
        Regions = regions;
    }

    /// <summary>The image's width.</summary>
    public int Width { get; }

    /// <summary>The image's height.</summary>
    public int Height { get; }

    /// <summary>The regions, labelled 1 to Count in raster order.</summary>
    public IReadOnlyList<Region> Regions { get; }

    /// <summary>Each pixel's region label, row by row (0: background, or a region smaller than the minimum area).</summary>
    public ReadOnlySpan<int> Labels => _labels;

    /// <summary>The label of the pixel at (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public int LabelAt(int x, int y) => _labels[y * Width + x];
}

/// <summary>
/// Labels the connected regions of an image's foreground: blob counting, objects on a plain background, form fields,
/// touching characters a projection cannot separate. Two passes over the pixels with union-find; one int per pixel.
/// </summary>
public static class ConnectedComponents
{
    /// <summary>
    /// The regions of <paramref name="image"/>'s foreground (pixels above its threshold). Regions of fewer than
    /// <paramref name="minArea"/> pixels are left out (their pixels labelled 0).
    /// </summary>
    public static RegionMap Find(ForegroundImage image, Connectivity connectivity = Connectivity.Eight, int minArea = 1)
    {
        ArgumentNullException.ThrowIfNull(image);
        int width = image.Width, height = image.Height;
        var values = image.Data;
        float threshold = image.Threshold;
        var labels = new int[width * height];
        var parent = new List<int> { 0 };   // union-find over provisional labels; 0 is the background

        // Pass 1: provisional labels from the neighbours already seen (left, and the row above).
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                if (values[i] <= threshold)
                {
                    continue;
                }

                int label = 0;
                void Join(int neighbour)
                {
                    if (neighbour == 0)
                    {
                        return;
                    }

                    if (label == 0)
                    {
                        label = neighbour;
                    }
                    else if (neighbour != label)
                    {
                        Union(parent, label, neighbour);
                    }
                }

                if (x > 0)
                {
                    Join(labels[i - 1]);
                }

                if (y > 0)
                {
                    Join(labels[i - width]);
                    if (connectivity == Connectivity.Eight)
                    {
                        if (x > 0)
                        {
                            Join(labels[i - width - 1]);
                        }

                        if (x < width - 1)
                        {
                            Join(labels[i - width + 1]);
                        }
                    }
                }

                if (label == 0)
                {
                    label = parent.Count;
                    parent.Add(label);
                }

                labels[i] = label;
            }
        }

        // Pass 2: each pixel takes its set's final label (1, 2, ... in raster order of the sets' first pixels) and adds to its region's statistics.
        var final = new int[parent.Count];
        var stats = new List<(int MinX, int MinY, int MaxX, int MaxY, long SumX, long SumY, int Area)>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] == 0)
            {
                continue;
            }

            int root = Find(parent, labels[i]);
            if (final[root] == 0)
            {
                stats.Add((int.MaxValue, int.MaxValue, -1, -1, 0, 0, 0));
                final[root] = stats.Count;
            }

            int label = final[root], x = i % width, y = i / width;
            labels[i] = label;
            var s = stats[label - 1];
            stats[label - 1] = (Math.Min(s.MinX, x), Math.Min(s.MinY, y), Math.Max(s.MaxX, x), Math.Max(s.MaxY, y), s.SumX + x, s.SumY + y, s.Area + 1);
        }

        // Small regions out; the others renumbered 1..n.
        var renumber = new int[stats.Count + 1];
        var regions = new List<Region>();
        for (int k = 0; k < stats.Count; k++)
        {
            var s = stats[k];
            if (s.Area < minArea)
            {
                continue;
            }

            renumber[k + 1] = regions.Count + 1;
            regions.Add(new Region(regions.Count + 1, new PixelBox(s.MinX, s.MinY, s.MaxX - s.MinX + 1, s.MaxY - s.MinY + 1), s.Area,
                (float)s.SumX / s.Area + 0.5f, (float)s.SumY / s.Area + 0.5f));
        }

        if (regions.Count != stats.Count)
        {
            for (int i = 0; i < labels.Length; i++)
            {
                labels[i] = renumber[labels[i]];
            }
        }

        return new RegionMap(width, height, labels, regions);
    }

    private static int Find(List<int> parent, int label)
    {
        int root = label;
        while (parent[root] != root)
        {
            root = parent[root];
        }

        while (parent[label] != root)
        {
            (parent[label], label) = (root, parent[label]);   // path compression
        }

        return root;
    }

    private static void Union(List<int> parent, int a, int b)
    {
        int ra = Find(parent, a), rb = Find(parent, b);
        if (ra != rb)
        {
            parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
        }
    }
}
