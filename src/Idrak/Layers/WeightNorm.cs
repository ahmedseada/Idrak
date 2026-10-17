// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// Weight normalization (Salimans and Kingma, 2016; PyTorch's <c>weight_norm</c> with dim 0): a layer's weight kept as a
/// gain g per output row and a direction v, the weight g · v / ‖v‖ made from them (<see cref="Tensor.WeightNorm"/>, one
/// kernel each way). Unbound, <see cref="Weight"/> makes the weight each time it is read, with its graph, so gradients
/// reach g and v through it. Bound (<see cref="Bind()"/>), the weight is made once and shared by every pass until
/// <see cref="Unbind()"/>: a leaf whose gradient the passes add up, handed to g and v in closed form by <see cref="Flush()"/>
/// (the same gradient as per pass, with one copy of the weight instead of one per pass).
/// </summary>
public sealed class WeightNorm : IDisposable
{
    private Tensor? _bound;

    /// <summary>
    /// A parametrization over <paramref name="gain"/> (one value per row of the direction, in any shape) and
    /// <paramref name="direction"/> (the weight's shape; its first dimension is the rows). Neither is owned: the layer
    /// holding them disposes them.
    /// </summary>
    public WeightNorm(Tensor gain, Tensor direction)
    {
        ArgumentNullException.ThrowIfNull(gain);
        ArgumentNullException.ThrowIfNull(direction);
        if (direction.Shape.Length == 0 || gain.Size != direction.Shape[0])
        {
            throw new ArgumentException($"A weight norm's gains hold {gain.Size} values for {(direction.Shape.Length == 0 ? 0 : direction.Shape[0])} rows of {Tensor.FormatShape(direction.Shape)}.", nameof(gain));
        }

        Gain = gain;
        Direction = direction;
    }

    /// <summary>The gains g, one per row.</summary>
    public Tensor Gain { get; }

    /// <summary>The directions v, the weight's shape.</summary>
    public Tensor Direction { get; }

    /// <summary>Whether the weight is bound for the current step (<see cref="Bind()"/>).</summary>
    public bool Bound => _bound is not null;

    /// <summary>The weight g · v / ‖v‖: the bound one, else made now with its graph.</summary>
    public Tensor Weight => _bound ?? Tensor.WeightNorm(Gain, Direction);

    /// <summary>
    /// Makes the weight once, outside any scope and without a graph (<see cref="Flush()"/> works the gradient out itself),
    /// as a leaf that takes gradients when g or v does: a frozen layer's weight is a constant.
    /// </summary>
    public void Bind()
    {
        Unbind();
        Tensor weight;
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            weight = TensorScope.Untrack(Tensor.WeightNorm(Gain, Direction));
        }

        weight.RequiresGrad = Gain.RequiresGrad || Direction.RequiresGrad;
        _bound = weight;
    }

    /// <summary>
    /// Gives the bound weight's gathered gradient ∂w to g and v in closed form (<see cref="Backend.WeightNormBackward"/>):
    /// with w = g · v / ‖v‖ per row, ∂g = ⟨∂w, v⟩ / ‖v‖ and ∂v = (g / ‖v‖) · (∂w − v · ⟨∂w, v⟩ / ‖v‖²). Nothing when the
    /// weight is not bound, took no gradient, or neither g nor v takes one.
    /// </summary>
    public void Flush()
    {
        if (_bound?.Grad is not { } gradient || !(Gain.RequiresGrad || Direction.RequiresGrad))
        {
            return;
        }

        int rows = Direction.Shape[0], columns = Direction.Size / rows;
        Direction.Backend.WeightNormBackward(Gain.Storage, Direction.Storage, gradient.Storage,
            Gain.RequiresGrad ? Gain.GradStorage() : null, Direction.RequiresGrad ? Direction.GradStorage() : null, rows, columns);
    }

    /// <summary>Drops the bound weight (passes make it again).</summary>
    public void Unbind()
    {
        _bound?.Dispose();
        _bound = null;
    }

    /// <summary>Binds every parametrization (<see cref="Bind()"/>).</summary>
    public static void Bind(IEnumerable<WeightNorm> norms)
    {
        foreach (var norm in norms)
        {
            norm.Bind();
        }
    }

    /// <summary>Flushes every parametrization's gathered gradient (<see cref="Flush()"/>).</summary>
    public static void Flush(IEnumerable<WeightNorm> norms)
    {
        foreach (var norm in norms)
        {
            norm.Flush();
        }
    }

    /// <summary>Unbinds every parametrization (<see cref="Unbind()"/>).</summary>
    public static void Unbind(IEnumerable<WeightNorm> norms)
    {
        foreach (var norm in norms)
        {
            norm.Unbind();
        }
    }

    /// <summary>Drops the bound weight; g and v stay with the layer that owns them.</summary>
    public void Dispose() => Unbind();
}
