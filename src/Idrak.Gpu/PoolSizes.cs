// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu;

/// <summary>
/// The sizes a GPU memory pool holds cached blocks of, in order, for its best-fit reuse (the smallest cached size in a
/// range). A sorted list searched by halves: a size that empties and fills again (an output freed and taken back on every
/// decoding step) allocates nothing once the list has grown, where a <c>SortedSet</c> allocates a node on every add, and
/// the range query reads no view. Not thread-safe: the pool calls it under its lock.
/// </summary>
internal sealed class PoolSizes
{
    private readonly List<int> _sizes = [];

    /// <summary>How many sizes are held.</summary>
    public int Count => _sizes.Count;

    /// <summary>Adds <paramref name="size"/> unless it is held already.</summary>
    public void Add(int size)
    {
        int at = _sizes.BinarySearch(size);
        if (at < 0)
        {
            _sizes.Insert(~at, size);
        }
    }

    /// <summary>Removes <paramref name="size"/>; false when it was not held.</summary>
    public bool Remove(int size)
    {
        int at = _sizes.BinarySearch(size);
        if (at < 0)
        {
            return false;
        }

        _sizes.RemoveAt(at);
        return true;
    }

    /// <summary>Removes every size.</summary>
    public void Clear() => _sizes.Clear();

    /// <summary>The smallest size held in [<paramref name="low"/>, <paramref name="high"/>], or 0 when none is.</summary>
    public int Smallest(int low, int high)
    {
        int at = _sizes.BinarySearch(low);
        if (at < 0)
        {
            at = ~at;
        }

        return at < _sizes.Count && _sizes[at] <= high ? _sizes[at] : 0;
    }
}
