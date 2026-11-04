// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// Disposing a separate backend (CreateSeparate: a simulated rank carved from one GPU), as CUDA's: its kernels' modules,
// every block it took from the runtime and its stream go back, so creating and disposing backends does not grow the
// device's memory (rule 77). The device's own backend lives for the process (LivesForProcess): its Dispose does nothing.
internal sealed unsafe partial class HipBackend
{
    // Set once the dispose destroyed the stream: kernels given back afterwards do not wait for it.
    private volatile bool _streamDestroyed;

    protected override void Dispose(bool disposing)
    {
        int failure = Success;
        string failed = "";
        void Note(int result, string call)
        {
            if (result != Success && failure == Success)
            {
                (failure, failed) = (result, call);
            }
        }

        Bind();
        Note(hipStreamSynchronize(_stream), nameof(hipStreamSynchronize));    // queued work may still use its memory and kernels

        // The kernels' modules: the set's, the plug-ins' and the kernels loaded alone.
        var modules = new List<IDisposable>();
        lock (_pluginLock)
        {
            modules.AddRange(_pluginModules);
            _pluginModules.Clear();
            _pluginFunctions.Clear();
        }

        lock (_ownedFunctions)
        {
            modules.AddRange(_ownedFunctions);
            _ownedFunctions.Clear();
        }

        if (_kernelSet is { } set)
        {
            modules.Add(set);
            _kernelSet = null;
        }

        foreach (var module in modules)
        {
            try
            {
                module.Dispose();
            }
            catch (HipException e) when (failure == Success)
            {
                (failure, failed) = (-1, e.Message);
            }
        }

        // Every block it took (in use by a tensor that outlives the backend, or cached).
        lock (_pool)
        {
            foreach (ulong pointer in _deviceBlocks)
            {
                Note(hipFree(pointer), nameof(hipFree));
            }

            _deviceBlocks.Clear();
            _pool.Clear();
            _poolSizes.Clear();
        }

        if (hipEventDestroy != null)
        {
            foreach (var (start, end) in new[] { _profileEvents, _tuneEvents })
            {
                if (start != IntPtr.Zero)
                {
                    Note(hipEventDestroy(start), nameof(hipEventDestroy));
                    Note(hipEventDestroy(end), nameof(hipEventDestroy));
                }
            }

            (_profileEvents, _tuneEvents) = (default, default);
        }

        if (hipStreamDestroy != null)
        {
            Note(hipStreamDestroy(_stream), nameof(hipStreamDestroy));
        }

        _streamDestroyed = true;
        if (failure != Success)
        {
            throw new HipException($"Disposing {Name}: {(failure == -1 ? failed : $"{failed} failed with {Describe(failure)}")}; the rest was given back.");
        }
    }
}
