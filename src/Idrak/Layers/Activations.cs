// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>Logistic sigmoid activation; squashes values into (0, 1).</summary>
public sealed class Sigmoid : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Sigmoid();

    /// <inheritdoc />
    public override string ToString() => "Sigmoid";
}

/// <summary>Hyperbolic tangent activation; squashes values into (-1, 1).</summary>
public sealed class Tanh : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Tanh();

    /// <inheritdoc />
    public override string ToString() => "Tanh";
}

/// <summary>Rectified linear unit activation: max(x, 0).</summary>
public sealed class ReLU : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Relu();

    /// <inheritdoc />
    public override string ToString() => "ReLU";
}

/// <summary>
/// Randomly zeroes a fraction <see cref="Probability"/> of the inputs during training (and scales the
/// rest by 1 / (1 - p)) to reduce overfitting. In evaluation mode it passes inputs through unchanged.
/// </summary>
public sealed class Dropout : Module
{
    private readonly Random _random;

    /// <summary>Creates the layer.</summary>
    /// <param name="probability">Fraction of inputs to drop, in [0, 1).</param>
    /// <param name="random">Seed source for the masks; pass a seeded <see cref="System.Random"/> for reproducible runs.</param>
    public Dropout(float probability = 0.5f, Random? random = null)
    {
        if (probability is < 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(probability), probability, "Dropout probability must be in [0, 1).");
        }

        Probability = probability;
        _random = random ?? new Random();
    }

    /// <summary>Fraction of inputs dropped during training.</summary>
    public float Probability { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        IsTraining && Probability > 0f ? input.Dropout(Probability, DropoutSeeds.Next(_random)) : input;

    /// <summary>residual + this(x) in one pass (the seed drawn as <see cref="ForwardCore"/> draws it).</summary>
    internal Tensor AddTo(Tensor residual, Tensor x) =>
        IsTraining && Probability > 0f ? Tensor.AddDropout(residual, x, Probability, DropoutSeeds.Next(_random)) : residual + x;

    /// <inheritdoc />
    public override string ToString() => $"Dropout(p={Probability})";
}

/// <summary>Gaussian error linear unit, the standard activation in transformers.</summary>
public sealed class GELU : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Gelu();

    /// <inheritdoc />
    public override string ToString() => "GELU";
}

/// <summary>
/// GELU without the tanh approximation: 0.5 · x · (1 + erf(x / √2)), PyTorch's <c>gelu</c> (transformers' "gelu").
/// erf is computed from the library's element-wise operations (Abramowitz and Stegun 7.1.26: within 1.5e-7 of erf), so it
/// runs and is differentiated on every device without a kernel of its own; <see cref="GELU"/> (the tanh form) is the
/// fused one.
/// </summary>
public sealed class ExactGELU : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Apply(input);

    /// <summary>0.5 · x · (1 + erf(x / √2)) of <paramref name="x"/>.</summary>
    public static Tensor Apply(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        return x * (Erf(x * (1f / MathF.Sqrt(2f))) * 0.5f + 0.5f);
    }

    /// <summary>The error function of <paramref name="x"/>, element-wise, within 1.5e-7 (Abramowitz and Stegun 7.1.26).</summary>
    public static Tensor Erf(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
        var a = x.Abs();
        var t = (a * p + 1f).Pow(-1f);
        var poly = ((((t * a5 + a4) * t + a3) * t + a2) * t + a1) * t;
        return x.Sign() * (1f - poly * (a.Square() * -1f).Exp());
    }

    /// <inheritdoc />
    public override string ToString() => "ExactGELU";
}

/// <summary>Quick GELU, x · sigmoid(1.702 · x): the activation of OpenAI's CLIP (transformers' "quick_gelu").</summary>
public sealed class QuickGELU : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input * (input * 1.702f).Sigmoid();

    /// <inheritdoc />
    public override string ToString() => "QuickGELU";
}

/// <summary>
/// Softmax over the last dimension, turning scores into probabilities. Use it for inference output only:
/// train with raw scores and <see cref="Losses.CrossEntropy(Tensor, Tensor)"/>, which applies log-softmax itself.
/// </summary>
public sealed class Softmax : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Softmax();

    /// <inheritdoc />
    public override string ToString() => "Softmax";
}

/// <summary>Wraps any tensor function as a layer, e.g. <c>new Lambda(x =&gt; x.Mean(1), "MeanOverTime")</c>.</summary>
/// <param name="function">The computation; it may use any tensor operation and is differentiated automatically.</param>
/// <param name="name">Display name for summaries.</param>
public sealed class Lambda(Func<Tensor, Tensor> function, string name = "Lambda") : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => function(input);

    /// <inheritdoc />
    public override string ToString() => name;
}

/// <summary>
/// The seeds dropout masks are drawn with. An activation checkpoint records the seeds its first pass draws and replays
/// them when it recomputes the pass for the backward pass, so both passes drop the same elements.
/// </summary>
internal static class DropoutSeeds
{
    [ThreadStatic]
    private static List<uint>? t_record;

    [ThreadStatic]
    private static Queue<uint>? t_replay;

    public static uint Next(Random random)
    {
        if (t_replay is { Count: > 0 } replay)
        {
            return replay.Dequeue();
        }

        uint seed = (uint)random.Next();
        t_record?.Add(seed);
        return seed;
    }

    /// <summary>Records the seeds drawn until disposed into <paramref name="seeds"/>.</summary>
    public static Restore Record(List<uint> seeds)
    {
        var restore = new Restore(t_record, t_replay);
        t_record = seeds;
        t_replay = null;
        return restore;
    }

    /// <summary>Hands out <paramref name="seeds"/> in order until disposed.</summary>
    public static Restore Replay(List<uint> seeds)
    {
        var restore = new Restore(t_record, t_replay);
        t_record = null;
        t_replay = new Queue<uint>(seeds);
        return restore;
    }

    public readonly struct Restore(List<uint>? record, Queue<uint>? replay) : IDisposable
    {
        public void Dispose() => (t_record, t_replay) = (record, replay);
    }
}
