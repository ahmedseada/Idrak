// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// Storages of described values: the words that hold values of a type its description reads (TypeDescription), marked
// with it, so a device's described lane reads them by the description alone.
public abstract partial class Backend
{
    /// <summary>
    /// A storage of <paramref name="count"/> values described by <paramref name="description"/>: the 4-byte words that
    /// hold its whole groups of blocks (<see cref="TypeDescription.UnitValues"/> values in
    /// <see cref="TypeDescription.UnitBytes"/> bytes each), zeroed when <paramref name="zeroed"/>, marked with it
    /// (<see cref="Storage.Description"/>).
    /// </summary>
    /// <exception cref="ArgumentException">The words would pass what one storage holds (2³¹ - 1).</exception>
    public Storage Allocate(TypeDescription description, int count, bool zeroed)
    {
        ArgumentNullException.ThrowIfNull(description);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        long bytes = (count + (long)description.UnitValues - 1) / description.UnitValues * description.UnitBytes;
        long words = (bytes + sizeof(float) - 1) / sizeof(float);
        var storage = Allocate(words <= int.MaxValue ? (int)words : throw new ArgumentException($"{count} described values take {words} words: past what one storage holds."),
            zeroed);
        storage.SetDescription(description, count);
        return storage;
    }
}
