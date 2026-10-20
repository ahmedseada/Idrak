// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Vision.Plugins.Qwen;

/// <summary>
/// Qwen-VL's image side: images become the embeddings of their image tokens, which replace the
/// <c>&lt;|image_pad|&gt;</c> rows of the prompt. Each image is one block whose layout is its merged grid (rows ×
/// columns of tokens, a size of its own), and each token carries its axis positions (frame 0, its row, its column in that
/// grid: <see cref="AxisPositions"/>), which the library places in the prompt by <see cref="ImagePositionRules.Grid"/>.
/// The pixels go through <see cref="Processor"/>, <see cref="VisionTower"/> and <see cref="Merger"/>; the images of one
/// call run as one pass of the tower (each attending within itself). Every <c>Encode</c> runs without recording gradients
/// and returns tensors the caller disposes. As stages (<see cref="IVisionEncoderStages"/>), pixel values are
/// [images, grid rows, grid columns, patch values]: images join a pass only when their grids agree.
/// </summary>
public sealed class QwenImageEncoder : Module, IVisionEncoder, IVisionEncoderStages
{
    internal QwenImageEncoder(QwenVision vision, QwenVisionTower tower, QwenPatchMerger merger, QwenImageProcessor processor, Device device) =>
        (Vision, VisionTower, Merger, Processor, Device) = (vision, tower, merger, processor, device);

    /// <summary>The vision part this encoder was built from.</summary>
    public QwenVision Vision { get; }

    /// <summary>The vision transformer: pixel patches to patch outputs.</summary>
    public QwenVisionTower VisionTower { get; }

    /// <summary>The merger: patch outputs to image-token embeddings.</summary>
    public QwenPatchMerger Merger { get; }

    /// <summary>How images become pixel patches (the smart resize within the pixel budget, rescale, normalize, patches).</summary>
    public QwenImageProcessor Processor { get; }

    /// <inheritdoc />
    public int Width => Vision.TextDim;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        return [Processor.With(options).Grid(image).Layout(Vision.Config.MergeSize)];
    }

    /// <summary>A token's position on each axis (time 0, its row, its column in the merged grid), [3 · tokens], row-major as the merger gives the tokens.</summary>
    /// <exception cref="ArgumentException">The layout has no two-sided grid.</exception>
    public float[] AxisPositions(ImageTokenLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        if (layout.Grid.Count != 2 || layout.Grid[0] * layout.Grid[1] != layout.Tokens)
        {
            throw new ArgumentException($"A Qwen-VL block's layout is a grid of rows x columns tokens, got {layout}.", nameof(layout));
        }

        int rows = layout.Grid[0], columns = layout.Grid[1], tokens = layout.Tokens;
        var positions = new float[3 * tokens];                                          // time stays 0 for a still image
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                positions[tokens + r * columns + c] = r;
                positions[2 * tokens + r * columns + c] = c;
            }
        }

        return positions;
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return [];
        }

        var processor = Processor.With(options);
        int values = Vision.Config.PatchValues, merge = Vision.Config.MergeSize;
        var grids = images.Select(processor.Grid).ToList();
        int patches = grids.Sum(g => g.Patches);
        float[] pixels = ArrayPool<float>.Shared.Rent(patches * values);              // written, copied into the tensor, given back
        Tensor features;
        try
        {
            int at = 0;
            for (int i = 0; i < images.Count; i++)
            {
                processor.Patches(images[i], pixels.AsSpan(at * values, grids[i].Patches * values));
                at += grids[i].Patches;
            }

            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            var input = Tensor.From(pixels.AsSpan(0, patches * values), [patches, values], Device);
            features = scope.Keep(Merger.Forward(VisionTower.Forward(input, grids)));                // [tokens of all, text width]
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pixels);
        }

        var result = new List<ImageFeatures>(images.Count);
        try
        {
            int token = 0;
            foreach (var grid in grids)
            {
                var layout = grid.Layout(merge);
                var mine = images.Count == 1 ? features : features.Narrow(0, token, layout.Tokens);
                var positions = AxisPositions(layout);
                result.Add(new ImageFeatures(mine, layout, Tensor.Persistent(positions, [3, layout.Tokens], Device.Cpu)));
                token += layout.Tokens;
            }

            if (images.Count > 1)
            {
                features.Dispose();
            }
        }
        catch
        {
            result.ForEach(f => f.Dispose());
            throw;
        }

        return result;
    }

    /// <summary>The pixel patches of one image as [1, grid rows, grid columns, patch values], on the host.</summary>
    public Tensor PixelValues(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var processor = Processor.With(options);
        var grid = processor.Grid(image);
        int count = grid.Patches * Vision.Config.PatchValues;
        float[] values = ArrayPool<float>.Shared.Rent(count);
        try
        {
            processor.Patches(image, values.AsSpan(0, count));
            return Tensor.From(values.AsSpan(0, count), [1, grid.Rows, grid.Columns, Vision.Config.PatchValues], Device.Cpu);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    /// <summary>The tower's output [images, patches, tower width] of pixel values [images, grid rows, grid columns, patch values], without gradients.</summary>
    public Tensor Tower(Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        using var noGrad = Autograd.NoGrad();
        return VisionTower.Forward(OnDevice(pixelValues));
    }

    /// <summary>The image tokens' embeddings [images, tokens, text width] of pixel values as <see cref="Tower"/> takes them.</summary>
    public Tensor Features(Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        using var noGrad = Autograd.NoGrad();
        return Merger.Forward(VisionTower.Forward(OnDevice(pixelValues)));
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Merger.Forward(VisionTower.Forward(input));

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [VisionTower, Merger];

    /// <inheritdoc />
    public override void Dispose()
    {
        VisionTower.Dispose();
        Merger.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"QwenImageEncoder({VisionTower}, {Merger}, {Processor})";

    private Tensor OnDevice(Tensor values) => values.Device == Device ? values : Tensor.From(values.ToArray(), values.Shape, Device);
}
