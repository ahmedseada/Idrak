// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

/// <summary>
/// An operation with a forward and a backward step of one's own, both written with public tensor operations (no device
/// code): autograd records it as one operation and calls the backward step to pass gradients to its inputs, so it
/// composes with every other operation and layer. Create one with <see cref="Autograd.Function"/>.
/// </summary>
/// <example>
/// <code>
/// // softplus(x) = log(1 + e^x), whose derivative is sigmoid(x)
/// var softplus = Autograd.Function("softplus",
///     forward: x => (x[0].Exp() + 1f).Log(),
///     backward: (x, y, g) => [g * x[0].Sigmoid()]);
/// var loss = softplus.Apply(logits).Mean();
/// loss.Backward();
/// </code>
/// </example>
public sealed class DifferentiableFunction
{
    private readonly Func<IReadOnlyList<Tensor>, Tensor> _forward;
    private readonly Func<IReadOnlyList<Tensor>, Tensor, Tensor, IReadOnlyList<Tensor?>> _backward;

    internal DifferentiableFunction(string name, Func<IReadOnlyList<Tensor>, Tensor> forward,
        Func<IReadOnlyList<Tensor>, Tensor, Tensor, IReadOnlyList<Tensor?>> backward)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(backward);
        Name = name;
        _forward = forward;
        _backward = backward;
    }

    /// <summary>The operation's name, as telemetry and error messages show it.</summary>
    public string Name { get; }

    /// <summary>
    /// Runs the forward step on <paramref name="inputs"/> and returns its result, recorded for <see cref="Tensor.Backward()"/>
    /// when an input requires gradients. The forward step runs without recording (its own operations are not part of the
    /// graph); the tensors it creates besides the result are disposed.
    /// </summary>
    public Tensor Apply(params ReadOnlySpan<Tensor> inputs)
    {
        if (inputs.IsEmpty)
        {
            throw new ArgumentException($"{Name} was given no inputs.", nameof(inputs));
        }

        return Tensor.Apply(this, inputs.ToArray());
    }

    internal Tensor Forward(IReadOnlyList<Tensor> inputs) =>
        _forward(inputs) ?? throw new InvalidOperationException($"The forward step of {Name} returned null.");

    internal IReadOnlyList<Tensor?> Backward(IReadOnlyList<Tensor> inputs, Tensor output, Tensor gradient)
    {
        var gradients = _backward(inputs, output, gradient) ?? throw new InvalidOperationException($"The backward step of {Name} returned null.");
        if (gradients.Count != inputs.Count)
        {
            throw new InvalidOperationException($"The backward step of {Name} returned {gradients.Count} gradients for {inputs.Count} inputs (null for an input without one).");
        }

        return gradients;
    }
}
