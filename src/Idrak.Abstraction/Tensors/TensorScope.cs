// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Disposes every tensor created on the current thread while the scope is active, so a training
/// loop can recycle its intermediate results without a <c>using</c> on each one.
/// Model parameters, gradients and optimizer state are never captured by a scope.
/// </summary>
/// <example>
/// <code>
/// for (int epoch = 0; epoch &lt; 1000; epoch++)
/// {
///     using var scope = new TensorScope();
///     var loss = Losses.MeanSquaredError(model.Forward(x), y);
///     optimizer.ZeroGrad();
///     loss.Backward();
///     optimizer.Step();
/// }   // every tensor created in the iteration is released here
/// </code>
/// </example>
public sealed class TensorScope : IDisposable
{
    [ThreadStatic]
    private static TensorScope? t_current;

    private readonly TensorScope? _parent;
    private readonly List<Tensor> _tensors = [];
    private bool _disposed;

    /// <summary>Starts a scope on the current thread; scopes nest.</summary>
    public TensorScope()
    {
        _parent = t_current;
        t_current = this;
    }

    internal static void Track(Tensor tensor) => t_current?._tensors.Add(tensor);

    /// <summary>Whether <paramref name="tensor"/> was created in this scope (and is still to be disposed by it).</summary>
    internal bool Owns(Tensor tensor) => _tensors.FindLastIndex(t => ReferenceEquals(t, tensor)) >= 0;

    /// <summary>Keeps <paramref name="tensor"/> alive past this scope (it moves to the enclosing scope, if any).</summary>
    public Tensor Keep(Tensor tensor)
    {
        int index = _tensors.FindLastIndex(t => ReferenceEquals(t, tensor));
        if (index >= 0)
        {
            _tensors.RemoveAt(index);
            _parent?._tensors.Add(tensor);
        }

        return tensor;
    }

    /// <summary>
    /// Takes <paramref name="tensor"/> out of every scope on this thread: the caller disposes it (for tensors that outlive
    /// the scopes they happen to be made in, such as a distillation teacher's states for one training batch).
    /// </summary>
    internal static Tensor Untrack(Tensor tensor)
    {
        for (var scope = t_current; scope is not null; scope = scope._parent)
        {
            int index = scope._tensors.FindLastIndex(t => ReferenceEquals(t, tensor));
            if (index >= 0)
            {
                scope._tensors.RemoveAt(index);
            }
        }

        return tensor;
    }

    /// <summary>
    /// Closes the scope without disposing its tensors and hands them to the caller, who becomes responsible for them
    /// (used by <see cref="ComputeGraph"/>, whose recorded work keeps using those tensors).
    /// </summary>
    internal List<Tensor> Detach()
    {
        if (t_current != this)
        {
            throw new InvalidOperationException("Only the innermost TensorScope can be detached.");
        }

        _disposed = true;
        t_current = _parent;
        var tensors = new List<Tensor>(_tensors);
        _tensors.Clear();
        return tensors;
    }

    /// <summary>Disposes the tensors created in this scope and restores the enclosing scope.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (t_current != this)
        {
            throw new InvalidOperationException("TensorScopes must be disposed in the reverse order they were created, on the thread that created them.");
        }

        t_current = _parent;
        foreach (var tensor in _tensors)
        {
            tensor.Dispose();
        }

        _tensors.Clear();
    }
}
