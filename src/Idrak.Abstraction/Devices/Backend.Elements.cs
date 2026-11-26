// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// Storages of typed values: the words that hold values of a type as the type stores them (IElementType), marked with it,
// so a device reads them through the type (its described lane reads its description).
public abstract partial class Backend
{
    /// <summary>
    /// A storage of <paramref name="count"/> values of <paramref name="type"/>: the 4-byte words that hold the
    /// <see cref="IElementType.ByteCount"/> bytes they take as the type stores them, zeroed when <paramref name="zeroed"/>,
    /// marked with it (<see cref="Storage.Type"/>).
    /// </summary>
    /// <param name="type">The values' type.</param>
    /// <param name="count">The values.</param>
    /// <param name="zeroed">Whether its words are cleared.</param>
    /// <exception cref="ArgumentException">The words would pass what one storage holds (2³¹ - 1), or a tensor's values (2³¹ - 1).</exception>
    public Storage Allocate(IElementType type, long count, bool zeroed)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        long words = (type.ByteCount(count) + sizeof(float) - 1) / sizeof(float);
        if (words > int.MaxValue || count > int.MaxValue)
        {
            throw new ArgumentException($"{count} values of {type.Name} take {words} words: past what one storage holds.", nameof(count));
        }

        var storage = Allocate((int)words, zeroed);
        storage.SetType(type, count);
        return storage;
    }
}
