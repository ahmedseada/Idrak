// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The order a line's characters are met in along its columns, and back. A text is stored in logical order (the order it
/// is typed and read in); on the page a line is drawn in visual order, where a right-to-left line holds left-to-right runs
/// (numbers, Latin words) drawn the other way. A recognizer that reads a line's columns in its reading direction (a
/// right-to-left line flipped first) meets the characters in column order: <see cref="ToColumns"/> gives that order for a
/// transcription (the labels a line recognizer trains on) and <see cref="FromColumns"/> turns a reading back to logical
/// order. Characters are kept as they are (no mirrored glyphs).
/// </summary>
public interface IReadingOrder
{
    /// <summary>Its name in <see cref="ReadingOrders"/>.</summary>
    string Name { get; }

    /// <summary>
    /// <paramref name="logical"/> in the order its line's columns show it, read in the line's direction (right to left
    /// when <paramref name="rightToLeft"/>).
    /// </summary>
    string ToColumns(string logical, bool rightToLeft);

    /// <summary>A text in column order (as <see cref="ToColumns"/> gives it) back in logical order.</summary>
    string FromColumns(string columns, bool rightToLeft);
}

/// <summary>
/// The reading orders, by name (ignoring case). Built in (Idrak.Imaging): <see cref="Bidi"/> (the Unicode bidirectional
/// algorithm, UAX #9: levels resolved for a paragraph of the line's direction, reordered by rules L1 and L2) and
/// <see cref="Runs"/> (in a right-to-left line, each run of digits and left-to-right letters, with the spaces and
/// separators inside it, reversed; its own inverse). Register another with <see cref="Register"/>; one under a library
/// name takes its place, which stays behind it (see <see cref="SetPolicy"/>).
/// </summary>
public static class ReadingOrders
{
    /// <summary>"bidi": the Unicode bidirectional algorithm (UAX #9).</summary>
    public const string Bidi = "bidi";

    /// <summary>"runs": left-to-right runs reversed in a right-to-left line (right-to-left runs in a left-to-right one).</summary>
    public const string Runs = "runs";

    private static readonly SlotTable<string, IReadingOrder> Registry =
        new(nameof(ReadingOrders), (slot, app, library) => new GuardedOrder(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static ReadingOrders() => LibraryDefaults.Ensure(typeof(ReadingOrders));

    /// <summary>Registers <paramref name="order"/> under its <see cref="IReadingOrder.Name"/>; under a library name it takes that order's place.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IReadingOrder order)
    {
        ArgumentNullException.ThrowIfNull(order);
        ArgumentException.ThrowIfNullOrWhiteSpace(order.Name);
        Registry.Register(order.Name, order, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's order <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The order <paramref name="name"/> (an app's guarded by its policy), or null.</summary>
    public static IReadingOrder? Find(string name) => Registry.TryGet(name, out var order) ? order : null;

    /// <summary>The order <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IReadingOrder Get(string name) => Find(name)
        ?? throw new NotRegisteredException(nameof(ReadingOrders), name, Registry.Keys,
            $"Reading order '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with ReadingOrders.Register.");

    /// <summary>The library's order <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IReadingOrder? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the order <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>Each registered order as "name: what it does", one per line.</summary>
    public static string Describe() => string.Join("\n", Registry.Keys.Select(k => $"{k}: {Registry.Find(k)}"));

    /// <summary>
    /// What happens when the app's order <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> orders with the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's text is used, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    // An app's order over the library's: each call falls back on its own, or is compared with the library's text.
    private sealed class GuardedOrder(Slot slot, IReadingOrder app, IReadingOrder library) : IReadingOrder
    {
        public string Name => app.Name;

        public string ToColumns(string logical, bool rightToLeft) =>
            slot.Call(() => app.ToColumns(logical, rightToLeft), () => library.ToColumns(logical, rightToLeft), (a, b) => Comparisons.Exact(a, b));

        public string FromColumns(string columns, bool rightToLeft) =>
            slot.Call(() => app.FromColumns(columns, rightToLeft), () => library.FromColumns(columns, rightToLeft), (a, b) => Comparisons.Exact(a, b));

        public override string ToString() => app.ToString() ?? app.Name;
    }
}
