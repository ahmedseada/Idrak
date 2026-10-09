// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using Idrak.Data.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// A batch of detection or segmentation samples: the images as one tensor, each image's objects (boxes in the batch's
/// pixels, classes, masks) and, when the samples have them, the pixel classes as one tensor. The pixels are held once, in
/// the tensors. Dispose it after the step.
/// </summary>
public sealed class DetectionBatch : IDisposable
{
    internal DetectionBatch(Tensor images, BoundingBox[][] boxes, int[][] labels, byte[][]?[] masks, Tensor? pixelClasses, int index)
    {
        Images = images;
        Boxes = boxes;
        Labels = labels;
        Masks = masks;
        PixelClasses = pixelClasses;
        Index = index;
    }

    /// <summary>The images, [N, channels, height, width], values in [0, 1].</summary>
    public Tensor Images { get; }

    /// <summary>Each image's boxes, in the batch's pixels.</summary>
    public IReadOnlyList<IReadOnlyList<BoundingBox>> Boxes { get; }

    /// <summary>Each image's objects' classes.</summary>
    public IReadOnlyList<IReadOnlyList<int>> Labels { get; }

    /// <summary>Each image's objects' masks (width x height, row by row, non-zero inside), null for an image whose sample has none.</summary>
    public IReadOnlyList<IReadOnlyList<byte[]>?> Masks { get; }

    /// <summary>The class of every pixel, [N, height, width] (as <c>Losses.PixelCrossEntropy</c> takes them), or null when the samples have none.</summary>
    public Tensor? PixelClasses { get; }

    /// <summary>The batch's position in the epoch.</summary>
    public int Index { get; }

    /// <summary>The number of images.</summary>
    public int Count => Boxes.Count;

    /// <summary>Image <paramref name="image"/>'s boxes as corners [objects, 4] (x1, y1, x2, y2) on the batch's device, as the box losses take them.</summary>
    public Tensor Corners(int image)
    {
        var boxes = Boxes[image];
        var values = new float[boxes.Count * 4];
        for (int i = 0; i < boxes.Count; i++)
        {
            (values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3]) = (boxes[i].X, boxes[i].Y, boxes[i].Right, boxes[i].Bottom);
        }

        return Tensor.From(values, [boxes.Count, 4], Images.Device);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Images.Dispose();
        PixelClasses?.Dispose();
    }
}

/// <summary>
/// Batches detection and segmentation samples with their augmentations, off the training thread: while the model trains
/// on a batch, worker threads read, decode and augment the next ones. Every sample ends at <see cref="Width"/> x
/// <see cref="Height"/> (letterboxed by default) with <see cref="Channels"/> channels. A sample's random numbers are seeded
/// from the loader's seed, the epoch and the sample, so a run repeats exactly whatever the number of workers.
/// </summary>
/// <remarks>
/// The workers default to the CPU threads the machine reports (<see cref="ComputeResources.MaxCpuThreads"/>) less the
/// training thread; the batches gathered ahead default to as many as fit in a quarter of the memory the machine reports
/// free, at most one per worker. Both can be set. Each sample's pixels are copied by its worker into the batch's pooled
/// host buffer, so a batch is held once on the host until it is uploaded, and once in its tensors.
/// </remarks>
public sealed class AugmentedImageLoader : IEnumerable<DetectionBatch>
{
    private readonly Func<int, AnnotatedImage> _read;
    private readonly Random _random;
    private readonly int _seed;
    private int _epoch;

    /// <summary>A loader over <paramref name="count"/> samples, each read by <paramref name="read"/> (from any thread).</summary>
    /// <param name="count">The number of samples.</param>
    /// <param name="read">Reads sample i (decoded, before augmentation); called from worker threads.</param>
    /// <param name="width">The batch's image width.</param>
    /// <param name="height">The batch's image height.</param>
    /// <param name="batchSize">Samples per batch.</param>
    /// <param name="shuffle">Reorder the samples every epoch.</param>
    /// <param name="seed">The seed of the order and of the augmentations.</param>
    /// <param name="device">Where the batches go (<see cref="Device.Default"/> when null).</param>
    public AugmentedImageLoader(int count, Func<int, AnnotatedImage> read, int width, int height, int batchSize = 16, bool shuffle = false, int? seed = null, Device? device = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentNullException.ThrowIfNull(read);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        Count = count;
        _read = read;
        Width = width;
        Height = height;
        BatchSize = batchSize;
        Shuffle = shuffle;
        _seed = seed ?? Random.Shared.Next();
        _random = new Random(_seed);
        Device = device ?? Device.Default;
    }

    /// <summary>
    /// A loader over an annotated dataset's images (decoded with <see cref="ImageCodecs"/> when read): each image's boxes and
    /// classes and, with <paramref name="masks"/>, its objects' masks (crowd and difficult objects left out).
    /// </summary>
    public static AugmentedImageLoader FromDataset(AnnotatedDataset dataset, int width, int height, int batchSize = 16, bool shuffle = false, int? seed = null,
        Device? device = null, bool masks = false)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        return new AugmentedImageLoader(dataset.Images.Count, i => dataset.Images[i].ToSample(ImageCodecs.Decode(dataset.PathOf(i)), masks), width, height, batchSize, shuffle,
            seed, device);
    }

    /// <summary>The number of samples.</summary>
    public int Count { get; }

    /// <summary>The batch's image width.</summary>
    public int Width { get; }

    /// <summary>The batch's image height.</summary>
    public int Height { get; }

    /// <summary>Samples per batch.</summary>
    public int BatchSize { get; }

    /// <summary>Whether the samples are reordered every epoch.</summary>
    public bool Shuffle { get; }

    /// <summary>Where the batches go.</summary>
    public Device Device { get; }

    /// <summary>The augmentations, in order (none by default); <see cref="Augmentations.Parse"/> reads them from text.</summary>
    public IReadOnlyList<IAugmentation> Augmentations { get; init; } = [];

    /// <summary>The batch's channels: 3 (colour) by default, or 1 (grey); images of the other kind are converted.</summary>
    public int Channels { get; init; } = 3;

    /// <summary>Whether each sample is letterboxed to the batch's size (aspect ratio kept, the rest padded; the default) or stretched.</summary>
    public bool KeepRatio { get; init; } = true;

    /// <summary>The value of letterbox padding.</summary>
    public float Fill { get; init; } = 0.5f;

    /// <summary>Skip the last, smaller batch.</summary>
    public bool DropLast { get; init; }

    /// <summary>The threads that read and augment samples: by default the CPU threads the machine reports less the training thread (at least 1).</summary>
    public int Workers
    {
        get => field > 0 ? field : Math.Max(1, ComputeResources.MaxCpuThreads - 1);
        init => field = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }

    /// <summary>
    /// The batches gathered ahead of training: by default as many as fit in a quarter of the memory the machine reports
    /// free (images, pixel classes and masks of a batch), at most one per worker, at least one.
    /// </summary>
    public int Prefetch
    {
        get => field > 0 ? field : MeasuredPrefetch();
        init => field = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }

    /// <summary>The number of batches an epoch has.</summary>
    public int BatchCount => DropLast ? Count / BatchSize : (Count + BatchSize - 1) / BatchSize;

    private int MeasuredPrefetch()
    {
        long batchBytes = (long)BatchSize * Width * Height * (Channels * sizeof(float) + sizeof(float) + sizeof(byte));
        long free = Device.Cpu.Backend.AvailableMemory() ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return (int)Math.Clamp(free / 4 / Math.Max(1, batchBytes), 1, Workers);
    }

    /// <summary>One sample as a batch holds it: read, augmented and brought to the batch's size and channels.</summary>
    /// <param name="index">The sample.</param>
    /// <param name="epoch">The epoch (its random numbers differ from epoch to epoch).</param>
    public AnnotatedImage Prepare(int index, int epoch = 0)
    {
        var random = new Random(SampleSeed(_seed, epoch, index));
        var context = new AugmentationContext(random, r => _read(r.Next(Count)));
        var sample = Idrak.Abstraction.Data.Augmentations.Apply(Augmentations, _read(index), context);
        sample = new SampleResize(Width, Height, KeepRatio, Fill).Apply(sample, context);
        if (sample.Image.Channels != Channels)
        {
            sample = new AnnotatedImage(new ImageData(sample.Image.Resize(Channels, Height, Width), Channels, Height, Width), [.. sample.Boxes], [.. sample.Labels],
                sample.Masks?.ToArray(), sample.PixelClasses);
        }

        return sample;
    }

    /// <inheritdoc />
    public IEnumerator<DetectionBatch> GetEnumerator()
    {
        int epoch = Interlocked.Increment(ref _epoch) - 1;
        int[] order = [.. Enumerable.Range(0, Count)];
        if (Shuffle)
        {
            lock (_random)
            {
                _random.Shuffle(order);
            }
        }

        return Batches(order, epoch);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // A batch gathered on the host by the workers: its pixels (and pixel classes) in pooled buffers, its objects.
    private sealed record HostBatch(int Size, float[] Pixels, float[]? Classes, BoundingBox[][] Boxes, int[][] Labels, byte[][]?[] Masks);

    private IEnumerator<DetectionBatch> Batches(int[] order, int epoch)
    {
        int batches = BatchCount, workers = Workers;
        if (batches == 0)
        {
            yield break;
        }

        using var ready = new BlockingCollection<(HostBatch? Batch, Exception? Error)>(Prefetch);
        using var stop = new CancellationTokenSource();
        var options = new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = stop.Token };
        var producer = Task.Run(() =>
        {
            try
            {
                for (int b = 0; b < batches && !stop.IsCancellationRequested; b++)
                {
                    int start = b * BatchSize;
                    ready.Add((Gather(order, start, Math.Min(BatchSize, Count - start), epoch, options), null), stop.Token);
                }
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested)
            {
            }
            catch (Exception e)
            {
                try
                {
                    ready.Add((null, e), stop.Token);
                }
                catch (OperationCanceledException)
                {
                }
            }
            finally
            {
                ready.CompleteAdding();
            }
        });

        try
        {
            int index = 0;
            foreach (var (batch, error) in ready.GetConsumingEnumerable())
            {
                if (error is not null)
                {
                    throw new InvalidOperationException($"Preparing batch {index} failed: {error.Message}", error);
                }

                yield return Upload(batch!, index++);
            }
        }
        finally
        {
            stop.Cancel();
            try
            {
                producer.Wait();
            }
            catch (AggregateException)
            {
            }

            foreach (var (batch, _) in ready.GetConsumingEnumerable())     // gathered ahead and not taken: their buffers go back
            {
                Return(batch);
            }
        }
    }

    // Prepares the samples of one batch on the workers, each writing its pixels into the batch's pooled buffer; a sample's
    // own pixels are garbage as soon as they are copied.
    private HostBatch Gather(int[] order, int start, int size, int epoch, ParallelOptions options)
    {
        int plane = Width * Height, per = Channels * plane;
        var pixels = ArrayPool<float>.Shared.Rent(size * per);
        var boxes = new BoundingBox[size][];
        var labels = new int[size][];
        byte[][]?[] masks = new byte[size][][];
        var classes = new int[size][];
        try
        {
            Parallel.For(0, size, options, i =>
            {
                var sample = Prepare(order[start + i], epoch);
                sample.Image.Pixels.AsSpan(0, per).CopyTo(pixels.AsSpan(i * per, per));
                (boxes[i], labels[i], masks[i], classes[i]) = ([.. sample.Boxes], [.. sample.Labels], sample.Masks?.ToArray(), sample.PixelClasses!);
            });
        }
        catch
        {
            ArrayPool<float>.Shared.Return(pixels);
            throw;
        }

        float[]? map = null;
        if (classes.Any(c => c is not null))
        {
            map = ArrayPool<float>.Shared.Rent(size * plane);
            for (int i = 0; i < size; i++)
            {
                var span = map.AsSpan(i * plane, plane);
                span.Clear();
                if (classes[i] is { } given)
                {
                    for (int k = 0; k < plane; k++)
                    {
                        span[k] = given[k];
                    }
                }
            }
        }

        return new HostBatch(size, pixels, map, boxes, labels, masks);
    }

    // The batch's tensors on the device (one copy from the pooled buffers, which go back to the pool).
    private DetectionBatch Upload(HostBatch batch, int index)
    {
        try
        {
            int plane = Width * Height;
            var images = Tensor.From(batch.Pixels.AsSpan(0, batch.Size * Channels * plane), [batch.Size, Channels, Height, Width], Device);
            var pixelClasses = batch.Classes is null ? null : Tensor.From(batch.Classes.AsSpan(0, batch.Size * plane), [batch.Size, Height, Width], Device);
            return new DetectionBatch(images, batch.Boxes, batch.Labels, batch.Masks, pixelClasses, index);
        }
        finally
        {
            Return(batch);
        }
    }

    private static void Return(HostBatch? batch)
    {
        if (batch is null)
        {
            return;
        }

        ArrayPool<float>.Shared.Return(batch.Pixels);
        if (batch.Classes is not null)
        {
            ArrayPool<float>.Shared.Return(batch.Classes);
        }
    }

    // The seed of one sample's augmentations: the loader's seed, the epoch and the sample mixed (SplitMix64's finalizer), as DataLoader's.
    private static int SampleSeed(int seed, int epoch, int sample)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)epoch) * 0x9E3779B97F4A7C15UL ^ (ulong)(uint)sample * 0xD6E8FEB86659FD93UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return (int)((z ^ (z >> 31)) & int.MaxValue);
    }
}
