// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

// The kernels of plug-in operations on this device (PluginOperation.KernelFor).
public abstract partial class Backend
{
    // What PluginOperation.KernelFor reads, by plug-in slot: the kernel to run, or null where the call takes the slow path
    // (the host default, counted); shorter than the slot when the slots must be resolved, and empty while traced.
    internal Delegate?[] PluginKernels = Kernels.Unresolved();

    // Every plug-in operation's kernel as resolved (null: the host default), or empty when it must be resolved again.
    private Delegate?[] _pluginChoices = Kernels.Unresolved();

    /// <summary>The plug-in kernels resolved for this device, holding <paramref name="slot"/> (resolved again when they do not).</summary>
    internal Delegate?[] PluginChoices(int slot)
    {
        var choices = Volatile.Read(ref _pluginChoices);
        if (slot < choices.Length)
        {
            return choices;
        }

        var fast = Volatile.Read(ref PluginKernels);
        var resolved = Kernels.ResolvePlugins(this, out _);
        if (ReferenceEquals(Interlocked.CompareExchange(ref _pluginChoices, resolved, choices), choices) && Volatile.Read(ref _trace) is null)
        {
            Interlocked.CompareExchange(ref PluginKernels, resolved, fast);   // unless Kernels changed meanwhile
        }

        return resolved;
    }

    /// <summary>Counts a call of the plug-in operation <paramref name="index"/> when the device is traced.</summary>
    internal void PluginCalled(int index) => Volatile.Read(ref _trace)?.Called(index);

    /// <summary>Marks the plug-in kernels as changed: the next call resolves them again.</summary>
    private void PluginKernelsChanged()
    {
        Volatile.Write(ref PluginKernels, Kernels.Unresolved());
        Volatile.Write(ref _pluginChoices, Kernels.Unresolved());
    }
}
