// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// An operation of the device contract (plan 9): its name, its index in each device's kernel slots, the delegate type its
/// kernels have (<see cref="OperationKernels"/>) and what runs when a device has no kernel for it. The descriptors are in
/// <see cref="Ops"/>, generated from <see cref="Backend"/> by tools/operations/generate.py; new operations come with new
/// versions of the library, never from outside it.
/// </summary>
public sealed class Operation
{
    internal Operation(string name, string kernelMethod, string signature, int index, Type kernelType, KernelSource fallback)
    {
        Name = name;
        KernelMethod = kernelMethod;
        Signature = signature;
        Index = index;
        KernelType = kernelType;
        Fallback = fallback;
    }

    /// <summary>The operation's name: its method on <see cref="Backend"/> (overloads are numbered: MaxPoolBackward2).</summary>
    public string Name { get; }

    /// <summary>The operation's slot in a device's kernel table, from 0 to <see cref="Ops.All"/>.Count - 1.</summary>
    public int Index { get; }

    /// <summary>The delegate type a kernel for it must have (in <see cref="OperationKernels"/>): the device's backend, then the operation's arguments.</summary>
    public Type KernelType { get; }

    /// <summary>
    /// What runs on a device with no kernel of its own and none registered: <see cref="KernelSource.Host"/>,
    /// <see cref="KernelSource.Composed"/> or <see cref="KernelSource.None"/>.
    /// </summary>
    public KernelSource Fallback { get; }

    /// <summary>The device's own kernel on <see cref="Backend"/> (MaxPoolBackwardKernel).</summary>
    internal string KernelMethod { get; }

    /// <summary>The kernel method's parameter types by .NET name (Storage,Int32,ConvGeometry&amp;), which tell overloads apart.</summary>
    internal string Signature { get; }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>Which kernel runs an operation on a device, in the order the dispatcher looks for one (<see cref="Kernels.Chain"/>).</summary>
public enum KernelSource
{
    /// <summary>A kernel registered for the device's kind with <see cref="Kernels.Register"/> whose requirement holds.</summary>
    Registered,

    /// <summary>The device's own kernel: its backend overrides the operation's <c>NameKernel</c>.</summary>
    Device,

    /// <summary>Other operations on the same device (the contract's default body composes them).</summary>
    Composed,

    /// <summary>The host fallback: the operands are copied to system memory, the CPU runs the operation, and the results go back.</summary>
    Host,

    /// <summary>No kernel: the operation returns false (or zero) and the caller takes another path (unfused products, say).</summary>
    None,
}
