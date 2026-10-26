// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Gpu;

/// <summary>
/// Device memory that one kind of work writes and reads within its own queued work (the split partial sums of a few-row
/// packed product, added by its last block or by a reduction queued right after it): one block kept and reused in queue
/// order, as the pool would hand the same block back on the next call anyway, so a decoding step allocates no storage for
/// it. A larger piece replaces the block and the one outgrown stays alive, unreleased: work recorded into a graph, and
/// work other threads have queued, may still use it (shapes repeat, so it grows a few times and then stays).
/// </summary>
internal sealed class KeptScratch
{
    private readonly Lock _gate = new();
    private Storage? _current;
    private List<Storage>? _outgrown;

    /// <summary>A block of at least <paramref name="length"/> floats on <paramref name="backend"/>, kept for the next call.</summary>
    public Storage Get(Backend backend, int length)
    {
        var current = Volatile.Read(ref _current);
        if (current is not null && current.Length >= length)
        {
            return current;
        }

        lock (_gate)
        {
            current = _current;
            if (current is not null && current.Length >= length)
            {
                return current;
            }

            var grown = backend.Allocate(length, zeroed: false);
            if (current is not null)
            {
                (_outgrown ??= []).Add(current);
            }

            Volatile.Write(ref _current, grown);
            return grown;
        }
    }
}
