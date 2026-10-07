// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Layers;

/// <summary>
/// Batch normalization for [N, C] features or [N, C, ...] images/sequences: normalizes each channel with
/// the batch mean and variance during training (and running averages in evaluation), then applies a
/// learned per-channel scale (gamma) and shift (beta). Speeds up and stabilizes training of deep networks.
/// </summary>
public sealed class BatchNorm : Module
{
    /// <summary>Creates the layer.</summary>
    /// <param name="channels">Size of dimension 1 (features for [N, C], channels for [N, C, H, W]).</param>
    /// <param name="momentum">Weight of the current batch in the running statistics.</param>
    /// <param name="epsilon">Added to the variance for numerical stability.</param>
    /// <param name="device">Where the parameters live.</param>
    public BatchNorm(int channels, float momentum = 0.1f, float epsilon = 1e-5f, Device? device = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        device ??= Device.Default;
        Channels = channels;
        Momentum = momentum;
        Epsilon = epsilon;
        Gamma = CreateParameter(Enumerable.Repeat(1f, channels).ToArray(), [channels], device);
        Beta = CreateParameter(new float[channels], [channels], device);
        RunningMean = CreateBuffer(new float[channels], [channels], device);
        RunningVariance = CreateBuffer(Enumerable.Repeat(1f, channels).ToArray(), [channels], device);
    }

    /// <summary>Number of normalized channels.</summary>
    public int Channels { get; }

    /// <summary>Weight of each new batch in the running statistics.</summary>
    public float Momentum { get; }

    /// <summary>Variance epsilon.</summary>
    public float Epsilon { get; }

    /// <summary>Learned per-channel scale.</summary>
    public Tensor Gamma { get; private set; }

    /// <summary>Learned per-channel shift.</summary>
    public Tensor Beta { get; private set; }

    /// <summary>Running mean used in evaluation mode.</summary>
    public Tensor RunningMean { get; private set; }

    /// <summary>Running (unbiased) variance used in evaluation mode.</summary>
    public Tensor RunningVariance { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank < 2 || input.Shape[1] != Channels)
        {
            throw new ArgumentException($"BatchNorm({Channels}) expects [N, {Channels}, ...], got {Tensor.FormatShape(input.Shape)}.");
        }

        int outer = input.Shape[0], inner = input.Size / (outer * Channels);
        Tensor normalized;
        if (IsTraining)
        {
            normalized = input.Normalize(outer, Channels, inner, Epsilon, out var mean, out var variance);
            int m = outer * inner;
            var backend = RunningMean.Backend;
            backend.Affine(RunningMean.Storage, RunningMean.Storage, Channels, 1f - Momentum, 0f);
            backend.Axpy(mean.Storage, RunningMean.Storage, Channels, Momentum);
            backend.Affine(RunningVariance.Storage, RunningVariance.Storage, Channels, 1f - Momentum, 0f);
            backend.Axpy(variance.Storage, RunningVariance.Storage, Channels, Momentum * m / Math.Max(m - 1, 1));
            mean.Dispose();
            variance.Dispose();
        }
        else
        {
            var invStd = Tensor.Empty([Channels], input.Device);
            input.Backend.InvSqrt(RunningVariance.Storage, invStd.Storage, Channels, Epsilon);
            normalized = input.NormalizeWith(RunningMean, invStd, outer, Channels, inner);
            if (!normalized.RequiresGrad)
            {
                invStd.Dispose();
            }
        }

        return normalized.GroupAffine(Gamma, Beta, Channels, inner);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Gamma, Beta];

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [RunningMean, RunningVariance];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Gamma = MoveTensor(Gamma, device);
        Beta = MoveTensor(Beta, device);
        RunningMean = MoveTensor(RunningMean, device);
        RunningVariance = MoveTensor(RunningVariance, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"BatchNorm({Channels})";
}

/// <summary>
/// Fixed per-channel normalization: (x - mean[c]) / std[c] over [N, C, ...] (the channels of images, the features of
/// [N, F]). Image networks expect inputs normalized by their training data's statistics (ImageNet's, for example);
/// as the first layer of the network, the normalization runs on the device, is saved with the model, and every caller
/// (predictors, packages, the CLI) feeds plain [0, 1] images. Nothing is trained; mean and std are kept as buffers.
/// </summary>
public sealed class ChannelNormalize : Module
{
    /// <summary>Creates the layer.</summary>
    /// <param name="mean">Each channel's mean.</param>
    /// <param name="std">Each channel's standard deviation (positive).</param>
    /// <param name="device">Where the values live.</param>
    public ChannelNormalize(IReadOnlyList<float> mean, IReadOnlyList<float> std, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(std);
        if (mean.Count == 0 || mean.Count != std.Count)
        {
            throw new ArgumentException($"Give one mean and one std per channel ({mean.Count} means, {std.Count} stds).");
        }

        if (std.Any(s => !(s > 0)))
        {
            throw new ArgumentOutOfRangeException(nameof(std), "Every std must be positive.");
        }

        device ??= Device.Default;
        Mean = [.. mean];
        Std = [.. std];
        Scale = CreateBuffer([.. std.Select(s => 1f / s)], [std.Count], device);
        Shift = CreateBuffer([.. mean.Select((m, c) => -m / std[c])], [std.Count], device);
    }

    /// <summary>Each channel's mean.</summary>
    public IReadOnlyList<float> Mean { get; }

    /// <summary>Each channel's standard deviation.</summary>
    public IReadOnlyList<float> Std { get; }

    /// <summary>The number of channels.</summary>
    public int Channels => Mean.Count;

    private Tensor Scale { get; set; }

    private Tensor Shift { get; set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank < 2 || input.Shape[1] != Channels)
        {
            throw new ArgumentException($"ChannelNormalize expects [N, {Channels}, ...], got {Tensor.FormatShape(input.Shape)}.");
        }

        int inner = input.Size / (input.Shape[0] * Channels);
        return input.GroupAffine(Scale, Shift, Channels, inner);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Scale, Shift];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Scale = MoveTensor(Scale, device);
        Shift = MoveTensor(Shift, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"ChannelNormalize({Channels})";
}

/// <summary>
/// Layer normalization over the last dimension: each sample (or token) is normalized on its own, then scaled
/// and shifted per feature. The standard normalization of transformers; independent of batch size.
/// </summary>
public sealed class LayerNorm : Module
{
    /// <summary>Creates the layer for inputs whose last dimension is <paramref name="features"/>.</summary>
    public LayerNorm(int features, float epsilon = 1e-5f, Device? device = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(features);
        device ??= Device.Default;
        Features = features;
        Epsilon = epsilon;
        Gamma = CreateParameter(Enumerable.Repeat(1f, features).ToArray(), [features], device);
        Beta = CreateParameter(new float[features], [features], device);
    }

    private LayerNorm(Tensor gamma, Tensor beta, float epsilon)
    {
        Features = gamma.Size;
        Epsilon = epsilon;
        Gamma = gamma;
        Beta = beta;
    }

    /// <summary>A layer around existing gain and shift vectors [features]; the layer takes ownership.</summary>
    public static LayerNorm FromWeights(Tensor gamma, Tensor beta, float epsilon) => new(gamma, beta, epsilon);

    /// <summary>Size of the normalized (last) dimension.</summary>
    public int Features { get; }

    /// <summary>Variance epsilon.</summary>
    public float Epsilon { get; }

    /// <summary>Learned per-feature scale.</summary>
    public Tensor Gamma { get; private set; }

    /// <summary>Learned per-feature shift.</summary>
    public Tensor Beta { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank < 1 || input.Shape[^1] != Features)
        {
            throw new ArgumentException($"LayerNorm({Features}) expects [..., {Features}], got {Tensor.FormatShape(input.Shape)}.");
        }

        if (!Autograd.IsEnabled)
        {
            return input.LayerNormFused(Gamma, Beta, Epsilon);   // inference: one kernel instead of three
        }

        return input.LayerNormTrain(Gamma, Beta, Epsilon);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Gamma, Beta];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Gamma = MoveTensor(Gamma, device);
        Beta = MoveTensor(Beta, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"LayerNorm({Features})";
}
