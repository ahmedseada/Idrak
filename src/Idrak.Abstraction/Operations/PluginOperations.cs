// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// An operation a plug-in declares (<see cref="PluginOperations.Register"/>): a packed weight format, a key/value cache
/// layout, a graph operation or an <see cref="Autograd.Function"/> names the kernel it needs, with a default kernel that
/// runs on every device, and kernels for some kinds of device are registered for it with <see cref="Kernels.Register"/>
/// like for the library's operations. <see cref="KernelFor"/> returns the kernel to run on a device: the one registered
/// last for the device's kind whose requirement holds, else <see cref="DefaultKernel"/>. It shows in
/// <see cref="Kernels.Chain"/> and <c>idrak kernels</c> after the library's operations, and <see cref="Kernels.Trace"/>
/// counts its calls.
/// </summary>
/// <typeparam name="TKernel">
/// The kernel's delegate type: the device's backend first (<see cref="Backend"/>), then the operation's arguments
/// (storages, sizes, scalars).
/// </typeparam>
/// <example>
/// <code>
/// public delegate void ScaleRows(Backend backend, Storage rows, Storage scales, Storage output, int n, int width);
///
/// public static readonly PluginOperation&lt;ScaleRows&gt; Scale = PluginOperations.Register&lt;ScaleRows&gt;("MyLayout.ScaleRows",
///     (b, rows, scales, output, n, width) =&gt; b.GroupScaleShift(rows, scales, null, output, n, n / width, width, false),
///     KernelSource.Composed);
///
/// Kernels.Register(Scale, "cpu", (b, rows, scales, output, n, width) =&gt; { /* loops over HostMemory */ });
/// Scale.KernelFor(backend)(backend, rows, scales, output, n, width);
/// </code>
/// </example>
public sealed class PluginOperation<TKernel> : Operation
    where TKernel : Delegate
{
    private readonly int _slot;

    internal PluginOperation(string name, int index, TKernel defaultKernel, KernelSource fallback)
        : base(name, "", "", index, typeof(TKernel), fallback)
    {
        _slot = index - OperationIndex.Count;
        DefaultKernel = defaultKernel;
    }

    /// <summary>
    /// The kernel that runs where none is registered: a host implementation (<see cref="KernelSource.Host"/>: it reads
    /// its operands with <see cref="Backend.Download"/> and writes them back with <see cref="Backend.Upload"/>), a
    /// composition of the library's operations (<see cref="KernelSource.Composed"/>), or one that reports "no kernel"
    /// (<see cref="KernelSource.None"/>), as <see cref="Operation.Fallback"/> says.
    /// </summary>
    public TKernel DefaultKernel { get; }

    /// <inheritdoc />
    internal override Delegate Default => DefaultKernel;

    /// <summary>
    /// The kernel to run on <paramref name="backend"/>: the one registered last for its kind whose requirement holds, else
    /// <see cref="DefaultKernel"/>. Call it with the same backend: <c>op.KernelFor(backend)(backend, ...)</c>. Resolved
    /// once per device and cached until the registered kernels change, so a call costs an array read.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TKernel KernelFor(Backend backend)
    {
        var kernels = backend.PluginKernels;
        int slot = _slot;
        return (uint)slot < (uint)kernels.Length && kernels[slot] is { } kernel ? Unsafe.As<TKernel>(kernel) : Resolve(backend);
    }

    // Off the fast path: the device resolves its slots (first call, after a change), is traced, or runs the host default.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private TKernel Resolve(Backend backend)
    {
        var kernels = backend.PluginChoices(_slot);
        backend.PluginCalled(Index);
        if (kernels[_slot] is { } kernel)
        {
            return Unsafe.As<TKernel>(kernel);
        }

        backend.HostCalled(Name);
        return DefaultKernel;
    }
}

/// <summary>
/// The operations plug-ins declare (<see cref="PluginOperation{TKernel}"/>): a name, a kernel delegate type and a default
/// kernel each, after the library's operations (<see cref="Ops"/>). A declaration lasts as long as the process; declare
/// each operation once, in a static field of the plug-in.
/// </summary>
public static class PluginOperations
{
    private static readonly List<Operation> Registry = [];

    private static Operation[] s_all = [];

    /// <summary>The plug-in operations declared so far, in order (their <see cref="Operation.Index"/> follows <see cref="Ops.All"/>).</summary>
    public static IReadOnlyList<Operation> All => Volatile.Read(ref s_all);

    /// <summary>
    /// Declares the operation <paramref name="name"/> with kernels of type <typeparamref name="TKernel"/> and
    /// <paramref name="defaultKernel"/> where none is registered (what it does: <paramref name="fallback"/>). Register a
    /// kernel per kind of device with <see cref="Kernels.Register{TKernel}(PluginOperation{TKernel}, string, TKernel, Func{Backend, bool})"/>.
    /// </summary>
    /// <param name="name">A name no other operation has (the library's included); prefix it with the plug-in's ("MyFormat.Unpack").</param>
    /// <param name="defaultKernel">The kernel that runs on every device where none is registered.</param>
    /// <param name="fallback">What the default kernel is: <see cref="KernelSource.Host"/>, <see cref="KernelSource.Composed"/> or <see cref="KernelSource.None"/>.</param>
    /// <exception cref="ArgumentException">
    /// The name is taken, the fallback is not one of the three, or <typeparamref name="TKernel"/> is not the default kernel's
    /// own delegate type.
    /// </exception>
    public static PluginOperation<TKernel> Register<TKernel>(string name, TKernel defaultKernel, KernelSource fallback = KernelSource.Host)
        where TKernel : Delegate
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(defaultKernel);
        if (fallback is not (KernelSource.Host or KernelSource.Composed or KernelSource.None))
        {
            throw new ArgumentException($"A plug-in operation's default kernel is a host, composed or none fallback, not {fallback}.", nameof(fallback));
        }

        if (defaultKernel.GetType() != typeof(TKernel))
        {
            throw new ArgumentException($"The kernels of {name} are {typeof(TKernel).Name}s, a delegate type of their own; the default is a {defaultKernel.GetType().Name}.", nameof(defaultKernel));
        }

        lock (Registry)
        {
            if (Ops.All.Any(o => o.Name.Equals(name, StringComparison.OrdinalIgnoreCase)) || Registry.Any(o => o.Name.Equals(name, StringComparison.OrdinalIgnoreCase)))
            {
                throw new ArgumentException($"An operation named '{name}' exists already; prefix a plug-in's operations with its name.", nameof(name));
            }

            var operation = new PluginOperation<TKernel>(name, OperationIndex.Count + Registry.Count, defaultKernel, fallback);
            Registry.Add(operation);
            Volatile.Write(ref s_all, [.. Registry]);
            return operation;
        }
    }

    /// <summary>The plug-in operation declared as <paramref name="name"/> (any case), or null.</summary>
    public static Operation? TryGet(string name) => All.FirstOrDefault(o => o.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
}
