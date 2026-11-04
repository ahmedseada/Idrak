// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Disposing a separate backend (CreateSeparate: a simulated rank carved from one GPU): everything it took from the driver
// goes back, so creating and disposing backends does not grow the device's memory (rule 77). The device's own backend
// lives for the process (LivesForProcess): its Dispose does nothing.
internal sealed unsafe partial class CudaBackend
{
    protected override void Dispose(bool disposing)
    {
        int failure = 0;
        string failed = "";
        void Note(int result, string call)
        {
            if (result != 0 && failure == 0)
            {
                (failure, failed) = (result, call);
            }
        }

        Bind();
        _streamGate.EnterReadLock();                                       // another backend's recording of the device ends first
        try
        {
            // Work queued on this backend's streams may still use its memory and kernels.
            Note(cuStreamSynchronize(_stream), nameof(cuStreamSynchronize));
            if (_copyStream != IntPtr.Zero)
            {
                Note(cuStreamSynchronize(_copyStream), nameof(cuStreamSynchronize));
            }

            lock (_graphHandles)
            {
                foreach (var (executable, graph) in _graphHandles)
                {
                    Note(cuGraphExecDestroy(executable), nameof(cuGraphExecDestroy));
                    Note(cuGraphDestroy(graph), nameof(cuGraphDestroy));
                    GraphDestroyed();
                }

                _graphHandles.Clear();
            }

            // Every block: those it took alone (in use by a tensor that outlives the backend, or cached), the chunks (with the
            // blocks placed in them) and the pinned system blocks.
            lock (_pool)
            {
                foreach (ulong pointer in _deviceBlocks)
                {
                    Note(cuMemFree(pointer), nameof(cuMemFree));
                }

                foreach (var chunk in _chunks)
                {
                    Note(cuMemFree(chunk.Address), nameof(cuMemFree));
                }

                foreach (var host in _hostBlocks.Values)
                {
                    Note(cuMemFreeHost(host), nameof(cuMemFreeHost));
                }

                _deviceBlocks.Clear();
                _chunks.Clear();
                _chunkBlocks.Clear();
                _hostBlocks.Clear();
                _hostPool.Clear();
                _pool.Clear();
                _poolSizes.Clear();
            }

            lock (_staging)
            {
                if (_stagingHost != IntPtr.Zero)
                {
                    Note(cuMemFreeHost(_stagingHost), nameof(cuMemFreeHost));
                    _stagingHost = IntPtr.Zero;
                }

                while (_staged.Count > 0)
                {
                    Note(cuEventDestroy(_staged.Dequeue().Done), nameof(cuEventDestroy));
                }

                while (_spareEvents.Count > 0)
                {
                    Note(cuEventDestroy(_spareEvents.Pop()), nameof(cuEventDestroy));
                }

                _stagingFailed = true;                                     // no new ring for a disposed backend
            }

            foreach (var (start, end) in new[] { _profileEvents, _tuneEvents })
            {
                if (start != IntPtr.Zero)
                {
                    Note(cuEventDestroy(start), nameof(cuEventDestroy));
                    Note(cuEventDestroy(end), nameof(cuEventDestroy));
                }
            }

            (_profileEvents, _tuneEvents) = (default, default);
            if (_gridBarrier != 0)
            {
                Note(cuMemFree(_gridBarrier), nameof(cuMemFree));
                _gridBarrier = 0;
            }

            lock (_modules)
            {
                foreach (IntPtr module in _modules)
                {
                    Note(cuModuleUnload(module), nameof(cuModuleUnload));
                }

                _modules.Clear();
            }

            Note(cuStreamDestroy(_stream), nameof(cuStreamDestroy));
            if (_copyStream != IntPtr.Zero)
            {
                Note(cuStreamDestroy(_copyStream), nameof(cuStreamDestroy));
                _copyStream = IntPtr.Zero;
            }
        }
        finally
        {
            _streamGate.ExitReadLock();
        }

        // This backend's retain of the device's primary context: the device's own backend and other separate ones keep theirs.
        Note(cuDevicePrimaryCtxRelease(_deviceHandle), nameof(cuDevicePrimaryCtxRelease));
        if (failure != 0)
        {
            throw new CudaException($"Disposing {Name}: {failed} failed ({CudaDriver.Describe(failure)}); the rest was given back.");
        }
    }
}
