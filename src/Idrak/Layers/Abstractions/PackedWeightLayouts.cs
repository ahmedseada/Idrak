// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// How the weights of one packed format are kept and read back (a load's packed-weight cache,
/// <see cref="ComputeResources.PackedWeightCache"/>, which <c>DecoderSpec.Build</c> reads and writes): the format's weights are their <see cref="IKernelPackedWeight.PackedValues"/>
/// and <see cref="IKernelPackedWeight.ScaleValues"/>, kept as they are, and made again from them with no packing. Registered
/// under the format's name (<see cref="PackedWeightLayouts"/>); a format with no layout is packed on every load.
/// </summary>
public interface IPackedWeightLayout : IKind
{
    /// <summary>
    /// The version of the bits the format packs: raised whenever the same values would pack to other bits (a changed
    /// quantizer), so the weights kept by an earlier version are made again.
    /// </summary>
    int Version { get; }

    /// <summary>Whether <paramref name="weight"/> is one this layout reads back (the format's own weights).</summary>
    /// <param name="weight">The weights packed.</param>
    bool Keeps(IKernelPackedWeight weight);

    /// <summary>
    /// The weights [<paramref name="rows"/>, <paramref name="columns"/>] on <paramref name="device"/> from the values their
    /// <see cref="IKernelPackedWeight.PackedValues"/> and <see cref="IKernelPackedWeight.ScaleValues"/> held, copied once to
    /// the device; null when the lengths are not this format's for that shape (the kept weights are then made again).
    /// </summary>
    /// <param name="packed">The packed values, as kept.</param>
    /// <param name="scales">The scales, as kept (empty for a format without).</param>
    /// <param name="rows">Input features.</param>
    /// <param name="columns">Output features.</param>
    /// <param name="device">Where the weights are made.</param>
    IKernelPackedWeight? FromStored(ReadOnlySpan<float> packed, ReadOnlySpan<float> scales, int rows, int columns, Device device);
}

/// <summary>
/// The layouts packed weights are kept in, by their format's name (ignoring case): "int8" and "int4" are the library's. A
/// format of one's own (<see cref="PackedWeightFormats.Register"/>) is kept between loads once its layout is registered
/// here under the same name by the same assembly; a layout an app registers over a library name stands for that app's
/// format only.
/// </summary>
public static class PackedWeightLayouts
{
    private static readonly KindTable<IPackedWeightLayout> Table = Layouts();

    private static KindTable<IPackedWeightLayout> Layouts()
    {
        var table = new KindTable<IPackedWeightLayout>(nameof(PackedWeightLayouts), "Packed weight layout", "add it with PackedWeightLayouts.Register",
            comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a layout reads back only what its own format kept: another layout cannot read those bits in its place");
        table.RegisterDefault(new Int8Layout());
        table.RegisterDefault(new Int4Layout());
        return table;
    }

    /// <summary>Registers <paramref name="layout"/> under its <see cref="IKind.Name"/> (its format's name), over the library's of that name.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPackedWeightLayout layout) => Table.Register(layout, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's layout <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered layout names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The layout of the format <paramref name="name"/>, or null when it has none (its weights are not kept).</summary>
    public static IPackedWeightLayout? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The library's layout <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPackedWeightLayout? Default(string name) => Table.Default(name);

    /// <summary>Who registered the layout <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered layout as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    private sealed class Int8Layout : IPackedWeightLayout
    {
        public string Name => "int8";

        public string Summary => "int8 bytes and a scale a column, as packed";

        public int Version => Int8Weight.PackingVersion;

        public bool Keeps(IKernelPackedWeight weight) => weight is Int8Weight;

        public IKernelPackedWeight? FromStored(ReadOnlySpan<float> packed, ReadOnlySpan<float> scales, int rows, int columns, Device device) =>
            packed.Length == Int8Weight.PackedLength(rows, columns) && scales.Length == columns ? Int8Weight.FromStored(packed, scales, rows, columns, device) : null;
    }

    private sealed class Int4Layout : IPackedWeightLayout
    {
        public string Name => "int4";

        public string Summary => "4-bit nibbles and a scale per 32 rows of a column, as packed";

        public int Version => Int4Weight.PackingVersion;

        public bool Keeps(IKernelPackedWeight weight) => weight is Int4Weight;

        public IKernelPackedWeight? FromStored(ReadOnlySpan<float> packed, ReadOnlySpan<float> scales, int rows, int columns, Device device) =>
            packed.Length == Int4Weight.PackedLength(rows, columns) && scales.Length == Int4Weight.ScalesLength(rows, columns)
                ? Int4Weight.FromStored(packed, scales, rows, columns, device) : null;
    }
}
