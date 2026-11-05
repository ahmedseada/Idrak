// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

// Element types (plan 37, T0 and T1): storages allocated with a type, the conversion between types, and the dispatcher's
// conversion steps: an operation's index storages (Operation.IndexParameters) in another type than the device's kernel
// takes (BackendCapabilities.IndexType), and its value storages in a type the kernel does not take
// (BackendCapabilities.ValueTypes; decision 8: ValueConversions), are converted once for the call, by the device's own
// conversion, and counted (ElementTypes.Conversions, telemetry's ElementsConverted): a gap stays visible, never a silent
// misread. tools/operations/generate.py reads this file after Backend.cs, Backend.Routing.cs and Backend.Layout.cs: its
// operations come after theirs in the slot order.
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
    /// The value types <paramref name="operation"/>'s kernel on this device reads and writes without a conversion
    /// (<see cref="BackendCapabilities.ValueTypesOf"/>); float32 where the device reports none, what every kernel took
    /// before the reports (decision 8). The dispatcher converts a value storage of another type into the first.
    /// </summary>
    public IReadOnlyList<ElementType> ValueTypesOf(Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return operation.IsPlugin ? ValueTypesFrom(Capabilities.ValueTypesOf(operation)) : ValueTypesAt(operation.Index);
    }

    // The value types of each library operation on this device, read once from its capabilities.
    private IReadOnlyList<ElementType>[]? _valueTypes;

    internal IReadOnlyList<ElementType> ValueTypesAt(int index) => (_valueTypes ?? ValueTypesNow())[index];

    private IReadOnlyList<ElementType>[] ValueTypesNow()
    {
        var capabilities = Capabilities;
        var types = new IReadOnlyList<ElementType>[OperationIndex.Count];
        foreach (var operation in Ops.All)
        {
            types[operation.Index] = ValueTypesFrom(capabilities.ValueTypesOf(operation));
        }

        return _valueTypes = types;
    }

    private static IReadOnlyList<ElementType> ValueTypesFrom(IReadOnlyList<ElementType> reported) => reported.Count > 0 ? reported : s_float32Values;

    private static readonly IReadOnlyList<ElementType> s_float32Values = [ElementType.Float32];

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

/// <summary>
/// The value storages of one call converted for its kernel (decision 8): each storage of a type the kernel does not take
/// (neither in <see cref="Backend.ValueTypesOf"/> nor naming the operation in its <see cref="ElementType.Kernels"/>) is
/// converted once into the kernel's first value type, counted; after the call each converted one the kernel changed is
/// converted back into the caller's storage (counted), so an output, or an input written in place, keeps its own type,
/// and one the kernel only read is left as it was (its values never pass through another type and back). A storage given
/// twice is converted once. Disposing gives the converted storages back.
/// </summary>
internal sealed class ValueConversions(Backend backend, Operation operation, int index) : IDisposable
{
    private List<(Storage Given, Storage Used, float[] Before)>? _converted;

    /// <summary>The storage the kernel reads in place of <paramref name="given"/>: itself when the kernel takes its type.</summary>
    public Storage In(Storage given)
    {
        var type = given.ElementType;
        var accepted = backend.ValueTypesAt(index);
        if (Contains(accepted, type) || Reads(type))
        {
            return given;
        }

        if (_converted is not null)
        {
            foreach (var (g, used, _) in _converted)
            {
                if (ReferenceEquals(g, given))
                {
                    return used;
                }
            }
        }

        var target = accepted[0];
        long count = given.ElementCount;
        var converted = backend.Allocate(target, count <= int.MaxValue ? (int)count : throw new ArgumentException($"{count} {type.Name} values: past what one storage holds."), zeroed: false);
        float[]? before = null;
        try
        {
            backend.ConvertElements(given, converted);
            before = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, converted.Length));
            backend.Download(converted, before.AsSpan(0, converted.Length));
        }
        catch
        {
            converted.Release();
            if (before is not null)
            {
                System.Buffers.ArrayPool<float>.Shared.Return(before);
            }

            throw;
        }

        ElementTypes.Converted(backend.Kind, operation.Name, type, target, count);
        (_converted ??= []).Add((given, converted, before));
        return converted;
    }

    /// <summary><see cref="In"/> for an optional storage.</summary>
    public Storage? Maybe(Storage? given) => given is null ? null : In(given);

    /// <summary>After the call: each converted storage the kernel changed converted back into the caller's, counted.</summary>
    public void Done()
    {
        if (_converted is null)
        {
            return;
        }

        foreach (var (given, used, before) in _converted)
        {
            float[] after = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, used.Length));
            try
            {
                var now = after.AsSpan(0, used.Length);
                backend.Download(used, now);
                if (!System.Runtime.InteropServices.MemoryMarshal.AsBytes(now).SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(before.AsSpan(0, used.Length))))
                {
                    backend.ConvertElements(used, given);
                    ElementTypes.Converted(backend.Kind, operation.Name, used.ElementType, given.ElementType, used.ElementCount);
                }
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(after);
            }
        }
    }

    public void Dispose()
    {
        if (_converted is null)
        {
            return;
        }

        foreach (var (_, used, before) in _converted)
        {
            used.Release();
            System.Buffers.ArrayPool<float>.Shared.Return(before);
        }

        _converted = null;
    }

    private static bool Contains(IReadOnlyList<ElementType> types, ElementType type)
    {
        for (int i = 0; i < types.Count; i++)
        {
            if (ReferenceEquals(types[i], type))
            {
                return true;
            }
        }

        return false;
    }

    // Whether the type names the operation (or every one) among the kernels that read it natively.
    private bool Reads(ElementType type)
    {
        foreach (string name in type.Kernels)
        {
            if (name == "*" || string.Equals(name, operation.Name, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
