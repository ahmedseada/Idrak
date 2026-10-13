// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.QwenVision;

/// <summary>
/// The Qwen-VL vision transformer (Qwen2_5_VisionTransformerPretrainedModel / Qwen2VisionTransformerPretrainedModel),
/// without its merger: pixel patches [patches, channels · temporal · patch · patch] through the patch embedding (a 3D
/// convolution whose kernel is the whole patch: one product, no bias), then <see cref="QwenVisionConfig.Depth"/> blocks
/// (x + attention(norm1(x)), x + mlp(norm2(x))) whose queries and keys rotate by each patch's row and column in its grid
/// (half the head's pairs by the row, half by the column; theta <see cref="QwenVisionConfig.RopeTheta"/>). Every image
/// attends within itself only, through the library's span attention (<see cref="Tensor.AttentionSpans"/>, no score
/// matrix stored), so several images of different sizes run as one pass. Qwen2.5-VL reorders the patches into windows of
/// <see cref="QwenVisionConfig.WindowTokens"/>² merged tokens and attends within each window except in
/// <see cref="QwenVisionConfig.FullAttentionBlocks"/>, with RMSNorm and a SwiGLU feed-forward; Qwen2-VL attends over the
/// whole image in every block, with LayerNorm and a quick-GELU feed-forward. The output is in the patches' own order.
/// As a module its forward pass takes pixel values [images, grid rows, grid columns, patch values] (one grid for all) and
/// gives [images, patches, width].
/// </summary>
public sealed class QwenVisionTower : Module
{
    private readonly Linear _patches;
    private readonly QwenVisionBlock[] _blocks;

    private QwenVisionTower(QwenVisionConfig config, QwenVlVariant variant, Linear patches, QwenVisionBlock[] blocks)
    {
        (Config, Variant, _patches, _blocks) = (config, variant, patches, blocks);
    }

    /// <summary>The tower's configuration.</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>Which family's tower.</summary>
    public QwenVlVariant Variant { get; }

    /// <summary>
    /// The tower from the vision part's tensors (<see cref="QwenVision.OpenTensors"/>: <c>patch_embed.proj.weight</c>,
    /// <c>blocks.N.…</c>), read one at a time, on <paramref name="device"/>. Projections keep the precision
    /// <paramref name="weights"/> asks for (as stored by default); the patch embedding, norms and biases are float32.
    /// </summary>
    public static QwenVisionTower FromTensors(QwenVisionConfig config, QwenVlVariant variant, ITensorStore tensors, Device device, EncoderWeights weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        var built = new List<IDisposable>();
        try
        {
            // The patch embedding's [dim, channels, temporal, patch, patch] kernel read as [dim, patch values]: a product
            // over each patch's values in the processor's order.
            var embed = Linear.FromWeights(Tensor.Persistent(tensors.ReadTransposed2D("patch_embed.proj.weight", config.Dim, config.PatchValues),
                [config.PatchValues, config.Dim], device, requiresGrad: true));
            built.Add(embed);
            var blocks = new QwenVisionBlock[config.Depth];
            for (int i = 0; i < blocks.Length; i++)
            {
                blocks[i] = QwenVisionBlock.FromTensors(config, variant, tensors, $"blocks.{i}.", device, weights);
                built.Add(blocks[i]);
            }

            return new QwenVisionTower(config, variant, embed, blocks);
        }
        catch
        {
            built.ForEach(b => b.Dispose());
            throw;
        }
    }

    /// <summary>The patch outputs [patches, width] of one or several images' patches (concatenated, each image's in its own order) with their grids.</summary>
    /// <exception cref="ArgumentException">The patches and the grids disagree.</exception>
    public Tensor Forward(Tensor patches, IReadOnlyList<QwenImageGrid> grids)
    {
        ArgumentNullException.ThrowIfNull(patches);
        ArgumentNullException.ThrowIfNull(grids);
        int count = grids.Sum(g => g.Patches);
        if (patches.Rank != 2 || patches.Shape[0] != count || patches.Shape[1] != Config.PatchValues)
        {
            throw new ArgumentException($"The tower takes [{count}, {Config.PatchValues}] patches for these grids, got {Tensor.FormatShape(patches.Shape)}.", nameof(patches));
        }

        var device = patches.Device;
        var layout = QwenTowerLayout.For(Config, grids);
        var cos = Tensor.From(layout.Cos, [count, Config.HeadDim / 2], device);
        var sin = Tensor.From(layout.Sin, [count, Config.HeadDim / 2], device);
        var index = Tensor.From(layout.Index, [count], device);
        var full = (Tensor.From(layout.FullStarts, [count], device), Tensor.From(layout.FullEnds, [count], device));
        var windows = layout.WindowStarts is null ? full : (Tensor.From(layout.WindowStarts, [count], device), Tensor.From(layout.WindowEnds!, [count], device));

        // Without autograd (and not recording a graph), each step runs in a scope of its own and the step before's output
        // is freed once used, as Sequential runs its layers: a pass holds one block's activations, not all of the tower's
        // (a page of 5,000 patches would otherwise keep about 400 MB per block until the read ends).
        bool freeing = !Autograd.IsEnabled && !ComputeGraph.IsCapturing;
        bool owned = false;
        Tensor Step(Tensor x, Func<Tensor, Tensor> step)
        {
            if (!freeing)
            {
                return step(x);
            }

            Tensor next;
            bool created;
            using (var scope = new TensorScope())
            {
                next = step(x);
                created = scope.Owns(next);
                scope.Keep(next);
            }

            if (!ReferenceEquals(next, x))
            {
                if (owned)
                {
                    x.Dispose();
                }

                owned = created;
            }

            return next;
        }

        var x = Step(patches, _patches.Forward);                                                // [patches, dim]
        if (layout.Order is { } order)
        {
            x = Step(x, h => h.EmbeddingLookup(Tensor.From(order, [order.Length], device)));    // into window order
        }

        for (int i = 0; i < _blocks.Length; i++)
        {
            var block = _blocks[i];
            var spans = Variant == QwenVlVariant.Qwen2Vl || Config.FullAttentionBlocks.Contains(i) ? full : windows;
            x = Step(x, h => block.Forward(h, cos, sin, index, spans));
        }

        if (layout.Reverse is { } reverse)
        {
            x = Step(x, h => h.EmbeddingLookup(Tensor.From(reverse, [reverse.Length], device)));   // back into the patches' order
        }

        return x;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4 || input.Shape[3] != Config.PatchValues)
        {
            throw new ArgumentException($"The tower takes pixel values [images, grid rows, grid columns, {Config.PatchValues}], got {Tensor.FormatShape(input.Shape)}.", nameof(input));
        }

        int images = input.Shape[0], rows = input.Shape[1], columns = input.Shape[2], patches = rows * columns;
        var grid = new QwenImageGrid(1, rows, columns);
        return Forward(input.Reshape(images * patches, Config.PatchValues), [.. Enumerable.Repeat(grid, images)]).Reshape(images, patches, Config.Dim);
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_patches, .. _blocks];

    /// <inheritdoc />
    public override void Dispose()
    {
        _patches.Dispose();
        foreach (var block in _blocks)
        {
            block.Dispose();
        }

        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"QwenVisionTower({Variant}, {Config})";

    // Every trainable tensor under its name in the vision part, and whether it is held transposed ([in, out] of [out, in]).
    internal Dictionary<string, (Tensor Tensor, bool Transposed)> Values()
    {
        var values = new Dictionary<string, (Tensor, bool)>(StringComparer.Ordinal) { ["patch_embed.proj.weight"] = (_patches.Weight, true) };
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (var (name, value) in _blocks[i].Values())
            {
                values[$"blocks.{i}.{name}"] = value;
            }
        }

        return values;
    }
}

// One block of the tower: the norms (RMSNorm or LayerNorm), the q, k, v projection and its output projection, the
// feed-forward (SwiGLU: gate, up, down; or fc1, quick-GELU, fc2).
internal sealed class QwenVisionBlock : Module
{
    private readonly Module _norm1, _norm2;
    private readonly Linear _qkv, _proj;
    private readonly Linear[] _mlp;
    private readonly QwenVisionConfig _config;
    private readonly QwenVlVariant _variant;

    private QwenVisionBlock(QwenVisionConfig config, QwenVlVariant variant, Module norm1, Linear qkv, Linear proj, Module norm2, Linear[] mlp) =>
        (_config, _variant, _norm1, _qkv, _proj, _norm2, _mlp) = (config, variant, norm1, qkv, proj, norm2, mlp);

    public static QwenVisionBlock FromTensors(QwenVisionConfig config, QwenVlVariant variant, ITensorStore tensors, string prefix, Device device, EncoderWeights weights)
    {
        var built = new List<IDisposable>();
        try
        {
            T Add<T>(T item) where T : IDisposable
            {
                built.Add(item);
                return item;
            }

            Linear Dense(string name) => Add(StoredWeights.Linear(tensors, prefix + name, weights, device));
            var norm1 = Add(QwenNorm.Read(tensors, prefix + "norm1", variant, config.Dim, device));
            var norm2 = Add(QwenNorm.Read(tensors, prefix + "norm2", variant, config.Dim, device));
            Linear[] mlp = variant == QwenVlVariant.Qwen25Vl
                ? [Dense("mlp.gate_proj"), Dense("mlp.up_proj"), Dense("mlp.down_proj")]
                : [Dense("mlp.fc1"), Dense("mlp.fc2")];
            return new QwenVisionBlock(config, variant, norm1, Dense("attn.qkv"), Dense("attn.proj"), norm2, mlp);
        }
        catch
        {
            built.ForEach(b => b.Dispose());
            throw;
        }
    }

    // x [patches, dim] (in the tower's order) through the block; each row attends to the keys of its range.
    public Tensor Forward(Tensor x, Tensor cos, Tensor sin, Tensor index, (Tensor Starts, Tensor Ends) spans)
    {
        int n = x.Shape[0], heads = _config.Heads, d = _config.HeadDim;
        var qkv = _qkv.Forward(_norm1.Forward(x));                                             // [patches, 3 · dim]: q, k, v
        Tensor Head(int part) => qkv.Narrow(1, part * _config.Dim, _config.Dim).Reshape(1, n, heads, d);
        var q = Head(0).Rope(cos, sin, index, d / 2, interleaved: false).Reshape(n, heads, d).Permute(1, 0, 2);   // [heads, patches, d]
        var k = Head(1).Rope(cos, sin, index, d / 2, interleaved: false).Reshape(n, heads, d).Permute(1, 0, 2);
        var v = Head(2).Reshape(n, heads, d).Permute(1, 0, 2);
        var attended = Tensor.AttentionSpans(q, k, v, spans.Starts, spans.Ends, 1f / MathF.Sqrt(d));
        x = x + _proj.Forward(attended.Permute(1, 0, 2).Reshape(n, _config.Dim));
        var h = _norm2.Forward(x);
        var hidden = _variant == QwenVlVariant.Qwen25Vl
            ? _mlp[0].Forward(h).Silu() * _mlp[1].Forward(h)
            : QuickGelu(_mlp[0].Forward(h));
        return x + _mlp[^1].Forward(hidden);
    }

    private static Tensor QuickGelu(Tensor x) => x * (x * 1.702f).Sigmoid();

    protected override Tensor ForwardCore(Tensor input) =>
        throw new NotSupportedException("A tower block runs inside the tower (QwenVisionTower.Forward), which gives it its rotary rows and key ranges.");

    public override IEnumerable<Module> Children() => [_norm1, _qkv, _proj, _norm2, .. _mlp];

    public override void Dispose()
    {
        foreach (var child in Children())
        {
            child.Dispose();
        }

        GC.SuppressFinalize(this);
    }

    // The block's trainable tensors under their names in the vision part.
    public IEnumerable<(string Name, (Tensor Tensor, bool Transposed) Value)> Values()
    {
        foreach (var (norm, name) in new[] { (_norm1, "norm1"), (_norm2, "norm2") })
        {
            foreach (var value in QwenNorm.Values(norm, name))
            {
                yield return value;
            }
        }

        string[] names = _variant == QwenVlVariant.Qwen25Vl ? ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] : ["mlp.fc1", "mlp.fc2"];
        foreach (var (linear, name) in new[] { (_qkv, "attn.qkv"), (_proj, "attn.proj") }.Concat(_mlp.Zip(names)))
        {
            yield return ($"{name}.weight", (linear.Weight, true));
            yield return ($"{name}.bias", (linear.Bias!, false));
        }
    }
}

// The norms of Qwen-VL: RMSNorm (y · w, eps 1e-6) in Qwen2.5-VL, LayerNorm (eps 1e-6) in Qwen2-VL.
internal static class QwenNorm
{
    public static Module Read(ITensorStore tensors, string name, QwenVlVariant variant, int dim, Device device)
    {
        var gain = Tensor.Persistent(tensors.Read(name + ".weight"), [dim], device, requiresGrad: true);
        if (variant == QwenVlVariant.Qwen25Vl)
        {
            return RMSNorm.FromWeights(gain, 1e-6f);
        }

        var bias = Tensor.Persistent(tensors.Read(name + ".bias"), [dim], device, requiresGrad: true);
        return LayerNorm.FromWeights(gain, bias, 1e-6f);
    }

    public static IEnumerable<(string Name, (Tensor Tensor, bool Transposed) Value)> Values(Module norm, string name) => norm switch
    {
        RMSNorm rms => [($"{name}.weight", (rms.Gain, false))],
        LayerNorm layer => [($"{name}.weight", (layer.Gamma, false)), ($"{name}.bias", (layer.Beta, false))],
        _ => [],
    };
}

// Where each patch goes in a pass of the tower over images with these grids: the window order (Qwen2.5-VL) and its
// reverse, each patch's rotary rows (by its row and column), and each row's key ranges (its image; its window).
internal sealed class QwenTowerLayout
{
    public required float[] Cos { get; init; }

    public required float[] Sin { get; init; }

    public required float[] Index { get; init; }

    public float[]? Order { get; init; }

    public float[]? Reverse { get; init; }

    public required float[] FullStarts { get; init; }

    public required float[] FullEnds { get; init; }

    public float[]? WindowStarts { get; init; }

    public float[]? WindowEnds { get; init; }

    // transformers' get_vision_window_index: the merged tokens of each image cut into windows of `w` x `w` tokens (the
    // last ones partial), window by window, row-major within each; each token's merge² patches stay together.
    public static QwenTowerLayout For(QwenVisionConfig config, IReadOnlyList<QwenImageGrid> grids)
    {
        int count = grids.Sum(g => g.Patches), m = config.MergeSize, unit = m * m, half = config.HeadDim / 2, quarter = half / 2;
        bool windowed = config.WindowSize > 0;
        var order = new int[count];
        var fullStarts = new float[count];
        var fullEnds = new float[count];
        var windowStarts = windowed ? new float[count] : null;
        var windowEnds = windowed ? new float[count] : null;
        int image = 0, at = 0;
        foreach (var grid in grids)
        {
            int rows = grid.Rows / m, columns = grid.Columns / m, w = Math.Max(1, config.WindowTokens);
            if (grid.Rows % m != 0 || grid.Columns % m != 0)
            {
                throw new ArgumentException($"A grid of {grid.Rows} x {grid.Columns} patches does not merge by {m}.", nameof(grids));
            }

            for (int frame = 0; frame < grid.Frames; frame++)
            {
                int frameStart = image + frame * rows * columns * unit;
                for (int wy = 0; wy < (windowed ? rows : 1); wy += windowed ? w : 1)
                {
                    for (int wx = 0; wx < (windowed ? columns : 1); wx += windowed ? w : 1)
                    {
                        int from = at;
                        int y1 = windowed ? Math.Min(rows, wy + w) : rows, x1 = windowed ? Math.Min(columns, wx + w) : columns;
                        for (int ty = wy; ty < y1; ty++)
                        {
                            for (int tx = wx; tx < x1; tx++)
                            {
                                int token = frameStart + (ty * columns + tx) * unit;
                                for (int u = 0; u < unit; u++)
                                {
                                    order[at++] = token + u;
                                }
                            }
                        }

                        if (windowed)
                        {
                            windowStarts.AsSpan(from, at - from).Fill(from);
                            windowEnds.AsSpan(from, at - from).Fill(at);
                        }
                    }
                }
            }

            // Every patch of the image sees the image (each frame its own segment, as transformers' cu_seqlens).
            for (int frame = 0; frame < grid.Frames; frame++)
            {
                int from = image + frame * grid.Rows * grid.Columns, to = from + grid.Rows * grid.Columns;
                fullStarts.AsSpan(from, to - from).Fill(from);
                fullEnds.AsSpan(from, to - from).Fill(to);
            }

            image += grid.Patches;
        }

        // Each patch's row and column in its grid (block-major order, as the processor lays them out), by its place in the
        // pass; rotary pair i < quarter by the row, the rest by the column, frequencies 1 / theta^(2j / half).
        var row = new int[count];
        var column = new int[count];
        image = 0;
        foreach (var grid in grids)
        {
            int columns = grid.Columns / m, i = 0;
            for (int frame = 0; frame < grid.Frames; frame++)
            {
                for (int by = 0; by < grid.Rows / m; by++)
                {
                    for (int bx = 0; bx < columns; bx++)
                    {
                        for (int my = 0; my < m; my++)
                        {
                            for (int mx = 0; mx < m; mx++, i++)
                            {
                                row[image + i] = by * m + my;
                                column[image + i] = bx * m + mx;
                            }
                        }
                    }
                }
            }

            image += grid.Patches;
        }

        var frequencies = new double[quarter];
        for (int j = 0; j < quarter; j++)
        {
            frequencies[j] = 1.0 / Math.Pow(config.RopeTheta, 2.0 * j / half);
        }

        var cos = new float[count * half];
        var sin = new float[count * half];
        for (int r = 0; r < count; r++)
        {
            int patch = windowed ? order[r] : r;
            for (int j = 0; j < half; j++)
            {
                double angle = (j < quarter ? row[patch] : column[patch]) * frequencies[j % quarter];
                cos[r * half + j] = (float)Math.Cos(angle);
                sin[r * half + j] = (float)Math.Sin(angle);
            }
        }

        float[]? forward = null, reverse = null;
        if (windowed)
        {
            forward = new float[count];
            reverse = new float[count];
            for (int r = 0; r < count; r++)
            {
                forward[r] = order[r];
                reverse[order[r]] = r;
            }
        }

        return new QwenTowerLayout
        {
            Cos = cos, Sin = sin, Index = [.. Enumerable.Range(0, count).Select(i => (float)i)], Order = forward, Reverse = reverse,
            FullStarts = Reorder(fullStarts, fullEnds, order, windowed).Starts, FullEnds = Reorder(fullStarts, fullEnds, order, windowed).Ends,
            WindowStarts = windowStarts, WindowEnds = windowEnds,
        };
    }

    // The image ranges by row of the pass: in window order the rows move, the ranges (whole images) stay.
    private static (float[] Starts, float[] Ends) Reorder(float[] starts, float[] ends, int[] order, bool windowed)
    {
        if (!windowed)
        {
            return (starts, ends);
        }

        var s = new float[starts.Length];
        var e = new float[ends.Length];
        for (int r = 0; r < order.Length; r++)
        {
            s[r] = starts[order[r]];
            e[r] = ends[order[r]];
        }

        return (s, e);
    }
}

// ReadTransposed of a tensor stored with more than two dimensions, as [rows, columns] → [columns, rows].
internal static class TensorStoreExtensions
{
    public static float[] ReadTransposed2D(this ITensorStore store, string name, int rows, int columns)
    {
        var shape = store.ShapeOf(name);
        if (shape.Aggregate(1L, (a, b) => a * b) != (long)rows * columns)
        {
            throw new InvalidDataException($"{name}: shape {Tensor.FormatShape(shape)} does not hold {rows} x {columns} values.");
        }

        var values = store.Read(name);
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }
}
