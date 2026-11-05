// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

// Element types (plan 37, T0 and T1): storages allocated with a type, the conversion between types, and the dispatcher's
// conversion step: an operation's index storages (Operation.IndexParameters) in another type than the device's kernel
// takes (BackendCapabilities.IndexType) are converted once for the call, by the device's own conversion, and counted
// (ElementTypes.Conversions, telemetry's ElementsConverted). tools/operations/generate.py reads this file after
// Backend.cs, Backend.Routing.cs and Backend.Layout.cs: its operations come after theirs in the slot order.
public abstract partial class Backend
{
    /// <summary>
    /// Converts every element of <paramref name="source"/> to <paramref name="destination"/>'s element type (their
    /// <see cref="Storage.ElementType"/>, as many elements each): integers to and from float32 and between int32 and int64
    /// exactly, or refused on the CPU (an index that is not a whole number, or past what the other type holds exactly:
    /// float32 holds integers up to 2²⁴); any other pair through float32 with the types' own conversions
    /// (<see cref="ElementType.Decode"/>, <see cref="ElementType.Encode"/>).
    /// </summary>
    public virtual void ConvertElementsKernel(Storage source, Storage destination)
    {
        using var h = new HostCall(this, nameof(ConvertElements));
        CpuBackend.Instance.ConvertElements(h[source], h[destination]);
    }

    /// <summary>
    /// A storage of <paramref name="count"/> elements of <paramref name="type"/> (plan 37, T0): the 4-byte words that hold
    /// them (<see cref="ElementType.StorageWords"/>), zeroed when <paramref name="zeroed"/>, marked with the type
    /// (<see cref="Storage.ElementType"/>). Float32 is <see cref="Allocate(int, bool)"/>'s.
    /// </summary>
    /// <exception cref="ArgumentException">The words would pass what one storage holds (2³¹ - 1).</exception>
    public Storage Allocate(ElementType type, int count, bool zeroed)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (ReferenceEquals(type, ElementType.Float32))
        {
            return Allocate(count, zeroed);
        }

        long words = type.StorageWords(count);
        var storage = Allocate(words <= int.MaxValue ? (int)words : throw new ArgumentException($"{count} {type.Name} elements take {words} words: past what one storage holds."),
            zeroed);
        storage.SetType(type, count);
        return storage;
    }

    /// <summary>The element type <paramref name="operation"/>'s kernel on this device reads its indices in (<see cref="BackendCapabilities.IndexType"/>, or its own entry).</summary>
    public ElementType IndexTypeOf(Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return operation.IsPlugin ? Capabilities.IndexType : IndexTypeAt(operation.Index);
    }

    // The index type of each library operation on this device, read once from its capabilities.
    private ElementType[]? _indexTypes;

    // The index type of the operation at `index` (OperationIndex), for the dispatcher's check.
    internal ElementType IndexTypeAt(int index) => (_indexTypes ?? IndexTypesNow())[index];

    private ElementType[] IndexTypesNow()
    {
        var capabilities = Capabilities;
        var types = new ElementType[OperationIndex.Count];
        foreach (var operation in Ops.All)
        {
            types[operation.Index] = capabilities.IndexTypes.TryGetValue(operation, out var own) ? own : capabilities.IndexType;
        }

        return _indexTypes = types;
    }

    /// <summary>
    /// The storage <paramref name="operation"/>'s kernel reads in place of <paramref name="given"/> (an index storage of
    /// another type than the kernel's, the operation's slot <paramref name="index"/>): a new storage of the kernel's type,
    /// holding <paramref name="given"/>'s elements converted when the kernel reads them (<paramref name="read"/>), counted.
    /// <see cref="IndicesBack"/> gives it back.
    /// </summary>
    internal Storage IndicesAs(Operation operation, Storage given, int index, bool read)
    {
        var type = IndexTypeAt(index);
        if (ReferenceEquals(given.ElementType, type))
        {
            return given;
        }

        long count = given.ElementCount;
        var used = Allocate(type, count <= int.MaxValue ? (int)count : throw new ArgumentException($"{count} indices: past what one storage holds."), zeroed: !read);
        if (read)
        {
            try
            {
                ConvertElements(given, used);
            }
            catch
            {
                used.Release();
                throw;
            }

            ElementTypes.Converted(Kind, operation.Name, given.ElementType, type, count);
        }

        return used;
    }

    /// <summary>
    /// Gives back a storage <see cref="IndicesAs"/> made for <paramref name="given"/>: its elements converted back into
    /// <paramref name="given"/> first when the kernel wrote them (<paramref name="write"/>), counted.
    /// </summary>
    internal void IndicesBack(Operation operation, Storage given, Storage used, bool write)
    {
        if (ReferenceEquals(given, used))
        {
            return;
        }

        try
        {
            if (write)
            {
                ConvertElements(used, given);
                ElementTypes.Converted(Kind, operation.Name, used.ElementType, given.ElementType, used.ElementCount);
            }
        }
        finally
        {
            used.Release();
        }
    }
}
