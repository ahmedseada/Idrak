// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Operations;

/// <summary>
/// An operation of the device contract (plan 9): its name, its index in each device's kernel slots, and the delegate type
/// its kernels have (<see cref="OperationKernels"/>). The descriptors are in <see cref="Ops"/>, generated from
/// <c>Backend</c> by tools/operations/generate.py.
/// </summary>
internal sealed class Operation(string name, int index, Type kernelType)
{
    /// <summary>The operation's name (<c>Backend</c>'s method; overloads are numbered: MaxPoolBackward2).</summary>
    public string Name { get; } = name;

    /// <summary>The operation's slot in a device's kernel table.</summary>
    public int Index { get; } = index;

    /// <summary>The delegate type a kernel for it must have: the device's backend, then the operation's arguments.</summary>
    public Type KernelType { get; } = kernelType;

    /// <inheritdoc />
    public override string ToString() => Name;
}
