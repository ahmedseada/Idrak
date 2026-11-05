// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// What a weight mapping is asked for one of a decoder's projections as a build makes it (plan 37, W3): the device it is
/// made on, whether its layer is one a split computes on the CPU (<paramref name="CpuLayer"/>: a weight placement's layers
/// past the device's, W1) and whether the layer is built on its first use (<paramref name="Lazy"/>, W2).
/// </summary>
/// <param name="Device">Where the weight is made.</param>
/// <param name="CpuLayer">Whether its layer is one of a split's CPU layers.</param>
/// <param name="Lazy">Whether its layer is built on its first use (a lazy load).</param>
public sealed record WeightMappingRequest(Device Device, bool CpuLayer, bool Lazy);

/// <summary>
/// A weight mapping (<see cref="WeightMappings"/>): whether a projection's weights are read in place from the checkpoint's
/// file mapping (where the source can: <see cref="IMappedTensorStore"/>, a device that reads a mapping in place) or copied
/// into memory.
/// </summary>
public interface IWeightMapping : IKind
{
    /// <summary>Whether the weights <paramref name="request"/> describes are read in place from the file's mapping where they can be.</summary>
    bool Maps(WeightMappingRequest request);
}

/// <summary>
/// The weight mappings, by name (ignoring case): whether a decoder's weights are read in place from the checkpoint's file
/// mapping or copied into memory (plan 37, W3). The library registers <see cref="Map"/> (the default: a split model's CPU
/// layers and lazily built layers read their weights where the file's pages lie, on a device that reads a mapping in place
/// (the CPU), where the file stores them as they are computed, float32, or as the load keeps them, IDRAK_KEEP_PACKED; the
/// operating system's page cache holds them once, with the same products bit for bit) and <see cref="Copy"/> (every weight
/// copied into memory, as before). A load takes the mapping from its options, else <see cref="EnvironmentVariable"/>
/// (IDRAK_WEIGHT_MAPPING), else <see cref="Map"/>; a name nobody registered is refused with "not registered". What the
/// mappings read in place, and every copy taken instead or after, is counted in <see cref="MappedWeights"/>.
/// </summary>
public static class WeightMappings
{
    /// <summary>A split's CPU layers and lazily built layers read in place from the file's mapping.</summary>
    public const string Map = "map";

    /// <summary>Every weight copied into memory, as before plan 37's W3.</summary>
    public const string Copy = "copy";

    /// <summary>The variable naming the mapping when an option does not: "IDRAK_WEIGHT_MAPPING".</summary>
    public const string EnvironmentVariable = "IDRAK_WEIGHT_MAPPING";

    private static readonly SlotTable<string, IWeightMapping> Table = new(nameof(WeightMappings), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a mapping decides whether a model's weights are read from its file or copied: falling back to another would hold them otherwise than asked");

    static WeightMappings() => Overrides.AsLibraryDefaults(() =>
    {
        Register(new MapInPlace());
        Register(new CopyIntoMemory());
    });

    /// <summary>The mapping named by <see cref="EnvironmentVariable"/>, else <see cref="Map"/>.</summary>
    public static string DefaultName => Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Map;

    /// <summary>Registers <paramref name="mapping"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IWeightMapping mapping)
    {
        ArgumentNullException.ThrowIfNull(mapping);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapping.Name);
        Table.Register(mapping.Name, mapping, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's mapping <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The mapping registered as <paramref name="name"/>, or null.</summary>
    public static IWeightMapping? Find(string name) => Table.Find(name);

    /// <summary>The mapping registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IWeightMapping Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(WeightMappings), name, Table.Keys,
            $"Weight mapping '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with WeightMappings.Register.");

    /// <summary>The library's mapping <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IWeightMapping? Default(string name) => Table.Default(name);

    /// <summary>Who registered the mapping <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered mapping as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(m => $"{m.Name}: {m.Summary}"));

    private sealed class MapInPlace : IWeightMapping
    {
        public string Name => Map;

        public string Summary => "a split model's CPU layers and lazily built layers read their weights in place from the checkpoint's file mapping (the CPU; float32 and kept formats), no second copy in memory";

        public bool Maps(WeightMappingRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            return (request.CpuLayer || request.Lazy) && request.Device.Backend is IMappedStorageBackend;
        }
    }

    private sealed class CopyIntoMemory : IWeightMapping
    {
        public string Name => Copy;

        public string Summary => "every weight copied from the file into memory, as before (the same results)";

        public bool Maps(WeightMappingRequest request) => false;
    }
}
