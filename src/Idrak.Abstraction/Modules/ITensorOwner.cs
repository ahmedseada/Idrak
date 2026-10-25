// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Modules;

/// <summary>
/// What holds the tensors a model is made of: its trainable parameters and its buffers (statistics that are saved and
/// moved but not optimized), which move to a device together and are freed together (<see cref="IDisposable.Dispose"/>
/// frees what <see cref="Parameters"/> and <see cref="Buffers"/> list). <see cref="Module"/> implements it over its
/// children; a module that holds tensors of its own implements all three for them, so an optimizer, a checkpoint, a move
/// and a dispose see the same tensors.
/// </summary>
public interface ITensorOwner : IDisposable
{
    /// <summary>The trainable tensors, in a stable order (an optimizer's state and a checkpoint follow it).</summary>
    IEnumerable<Tensor> Parameters();

    /// <summary>The tensors that are saved and moved but not optimized (batch-norm running statistics).</summary>
    IEnumerable<Tensor> Buffers();

    /// <summary>Moves every tensor of <see cref="Parameters"/> and <see cref="Buffers"/> to <paramref name="device"/>.</summary>
    void MoveTo(Device device);
}
