// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>Global switches for automatic differentiation.</summary>
public static class Autograd
{
    [ThreadStatic]
    private static int t_disabledDepth;

    /// <summary>Whether operations on the current thread record the graph needed for <see cref="Tensor.Backward()"/>.</summary>
    public static bool IsEnabled => t_disabledDepth == 0;

    /// <summary>
    /// Turns gradient recording off on this thread until the returned scope is disposed.
    /// Use it for inference and evaluation: it is faster and allocates less.
    /// </summary>
    /// <example><c>using (Autograd.NoGrad()) { var prediction = model.Forward(x); }</c></example>
    public static NoGradScope NoGrad()
    {
        t_disabledDepth++;
        return new NoGradScope(true);
    }

    /// <summary>
    /// Defines an operation with its own forward and backward steps, written with public tensor operations (see
    /// <see cref="DifferentiableFunction"/>). Call <see cref="DifferentiableFunction.Apply"/> to run it.
    /// </summary>
    /// <param name="name">The operation's name (for telemetry and error messages).</param>
    /// <param name="forward">Computes the result from the inputs. It runs without recording gradients.</param>
    /// <param name="backward">
    /// Given the inputs, the forward result and the gradient of the loss with respect to that result, returns the gradient
    /// with respect to each input, in order and of the input's shape (null for an input that needs none). It runs without
    /// recording gradients; the tensors it creates are disposed after the gradients are added to the inputs'.
    /// </param>
    /// <example>
    /// <code>
    /// var softplus = Autograd.Function("softplus", x => (x[0].Exp() + 1f).Log(), (x, y, g) => [g * x[0].Sigmoid()]);
    /// </code>
    /// </example>
    public static DifferentiableFunction Function(string name, Func<IReadOnlyList<Tensor>, Tensor> forward,
        Func<IReadOnlyList<Tensor>, Tensor, Tensor, IReadOnlyList<Tensor?>> backward) => new(name, forward, backward);

    /// <summary>Restores gradient recording when disposed.</summary>
    public readonly struct NoGradScope : IDisposable
    {
        private readonly bool _active;

        internal NoGradScope(bool active) => _active = active;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_active)
            {
                t_disabledDepth--;
            }
        }
    }
}
