// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// An operation of the device contract (plan 9): its name, its index in each device's kernel slots, the delegate type its
/// kernels have and what runs when a device has no kernel for it. The library's operations are in <see cref="Ops"/>,
/// generated from <see cref="Backend"/> by tools/operations/generate.py, with their delegates in
/// <see cref="OperationKernels"/>. Each is an <see cref="Operation{TKernel}"/> of its kernel's delegate type.
/// </summary>
public class Operation
{
    private protected Operation(string name, int index, Type kernelType, KernelSource fallback)
    {
        Name = name;
        Index = index;
        KernelType = kernelType;
        Fallback = fallback;
    }

    /// <summary>
    /// The operation's name: its method on <see cref="Backend"/> (overloads are numbered: MaxPoolBackward2).
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// The operation's row in <see cref="Kernels.Chain"/>: from 0 to <see cref="Ops.All"/>.Count - 1.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// The delegate type a kernel for it must have (in <see cref="OperationKernels"/> for the library's operations): the
    /// device's backend, then the operation's arguments.
    /// </summary>
    public Type KernelType { get; }

    /// <summary>
    /// What runs on a device with no kernel of its own: <see cref="KernelSource.Host"/>,
    /// <see cref="KernelSource.Composed"/> or <see cref="KernelSource.None"/>.
    /// </summary>
    public KernelSource Fallback { get; }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>An operation whose kernels are <typeparamref name="TKernel"/>s (<see cref="Operation.KernelType"/>).</summary>
/// <typeparam name="TKernel">The kernel's delegate type: the device's backend first, then the operation's arguments.</typeparam>
public class Operation<TKernel> : Operation
    where TKernel : Delegate
{
    internal Operation(string name, int index, KernelSource fallback)
        : base(name, index, typeof(TKernel), fallback)
    {
    }
}

/// <summary>Which kernel runs an operation on a device, in the order the dispatcher looks for one (<see cref="Kernels.Chain"/>).</summary>
public enum KernelSource
{
    /// <summary>
    /// The device's own kernel: its backend overrides the operation's <c>NameKernel</c>.
    /// </summary>
    Device,

    /// <summary>Other operations on the same device (the contract's default body composes them).</summary>
    Composed,

    /// <summary>
    /// The host fallback: the operands are copied to system memory, the CPU runs the operation, and the results go back.
    /// </summary>
    Host,

    /// <summary>No kernel: the operation returns false (or zero) and the caller takes another path (unfused products, say).</summary>
    None,
}
