// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A text binding's settings, read once per page from its options (<see cref="ITextBinding.Read"/>): the options as
/// given, and whatever a binding parses them into (a binding's own settings derive from this class).
/// </summary>
public class TextBindingSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public TextBindingSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a binding's defaults.</summary>
    public static TextBindingSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>Text put into regions: each region's text (empty for a region left empty), and the label, the drawn text one line a group in reading order.</summary>
/// <param name="Texts">Each region's text, in the order the regions were given ("" for one that draws nothing).</param>
/// <param name="Label">The drawn text: one line per group that has text, in reading order; the units of each line as its regions hold them.</param>
public sealed record BoundText(IReadOnlyList<string> Texts, IReadOnlyList<string> Label);

/// <summary>
/// Puts a page's text into the regions its shapes make (<see cref="ShapeRegion"/>): which units (whitespace-separated
/// tokens) of which line go into which region. Its settings are read once per page (<see cref="Read"/>). The contract: a
/// region left empty draws nothing and its units are absent from the label; units placed nowhere are absent from the
/// label; the label is exactly the drawn text, one line per group that has text, in reading order (groups in the order
/// of their numbers, a group's regions in the order of their indexes). The same input and seed give the same binding.
/// Register one with <see cref="TextBindings.Register"/>.
/// </summary>
public interface ITextBinding : IConfigurableKind
{
    /// <summary>The binding's settings from <paramref name="options"/> (its own keys, <see cref="IConfigurableKind.Keys"/>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad value.</exception>
    TextBindingSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>Puts <paramref name="lines"/> into <paramref name="regions"/> with <paramref name="settings"/>, its choices drawn from <paramref name="seed"/>.</summary>
    BoundText Bind(IReadOnlyList<string> lines, IReadOnlyList<ShapeRegion> regions, TextBindingSettings settings, int seed);
}

/// <summary>
/// The text bindings, by name (ignoring case). Built in (Idrak.Imaging): <c>sequence</c> (each line fills one group of
/// regions in reading order, a unit a region unless <c>take</c> says otherwise; <c>blank</c> and <c>blank-at</c> leave
/// regions empty). Register another with <see cref="Register"/>; one under a library name shadows the library's, which
/// <see cref="Unregister"/> brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class TextBindings
{
    /// <summary>The binding that fills a group of regions per line, in reading order.</summary>
    public const string Sequence = "sequence";

    private static readonly KindTable<ITextBinding> Table = new(nameof(TextBindings), "Text binding", "add it with TextBindings.Register",
        (slot, app, library) => new GuardedBinding(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static TextBindings() => LibraryDefaults.Ensure(typeof(TextBindings));

    /// <summary>Registers <paramref name="binding"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextBinding binding) => Table.Register(binding, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's binding <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The binding registered as <paramref name="name"/>, or null.</summary>
    public static ITextBinding? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The binding registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITextBinding Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's binding <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITextBinding? Default(string name) => Table.Default(name);

    /// <summary>Who registered the binding <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's binding <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> binds with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's bind and
    /// compares the app's on a sample of pages).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered binding as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The keys of every registered binding, each once, in registration order.</summary>
    public static IReadOnlyList<string> AllKeys() => [.. Table.Kinds.SelectMany(b => b.Keys).Distinct(StringComparer.OrdinalIgnoreCase)];

    // The app's settings and the library's, read from the same options (the library's null when it refuses them).
    private sealed class GuardedSettings(TextBindingSettings app, TextBindingSettings? library, IReadOnlyDictionary<string, string> options) : TextBindingSettings(options)
    {
        public TextBindingSettings App => app;

        public TextBindingSettings? Library => library;
    }

    private sealed class GuardedBinding(Slot slot, ITextBinding app, ITextBinding library) : ITextBinding
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public TextBindingSettings Read(IReadOnlyDictionary<string, string> options)
        {
            var own = app.Read(options);
            TextBindingSettings? theirs;
            try
            {
                theirs = library.Read(PageOptions.Only(options, library.Keys));
            }
            catch (ArgumentException)
            {
                theirs = null;
            }

            return new GuardedSettings(own, theirs, options);
        }

        public BoundText Bind(IReadOnlyList<string> lines, IReadOnlyList<ShapeRegion> regions, TextBindingSettings settings, int seed)
        {
            var (own, theirs) = settings is GuardedSettings g ? (g.App, g.Library) : (settings, settings);
            return theirs is null
                ? app.Bind(lines, regions, own, seed)
                : slot.Call(() => app.Bind(lines, regions, own, seed), () => library.Bind(lines, regions, theirs, seed),
                    (a, b) => Comparisons.Sequences(a.Texts, b.Texts) ?? Comparisons.Sequences(a.Label, b.Label));
        }
    }
}
