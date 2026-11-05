// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction;

/// <summary>
/// A rule for the type an operation's result takes from its inputs' types (plan 37, T4; decision 6: types are data),
/// registered in <see cref="TypePromotions"/>: asked in turn, the app's before the library's, until one decides.
/// </summary>
public interface ITypePromotion : IKind
{
    /// <summary>
    /// The type of <paramref name="operation"/>'s result (any operation when null) over inputs of <paramref name="left"/>
    /// and <paramref name="right"/>, or null when this rule does not decide it (the next rule is asked).
    /// </summary>
    /// <param name="operation">The operation, or null for any.</param>
    /// <param name="left">The first input's type.</param>
    /// <param name="right">The second input's type.</param>
    ElementType? Promote(Operations.Operation? operation, ElementType left, ElementType right);
}

/// <summary>
/// The rules that give an operation's result type from its inputs' types (plan 37, T4), by name: the library's
/// <see cref="Standard"/>, and any an app registers (asked first, newest first). No type is fixed here: the result
/// follows from the inputs' types alone, as data (bits, float or integer, signed or not, block format or not), so a type
/// an app registers takes part without a rule of its own. A pair no rule decides is refused, naming it.
/// </summary>
public static class TypePromotions
{
    /// <summary>The library's rule: "standard".</summary>
    public const string Standard = "standard";

    private static readonly KindTable<ITypePromotion> Table = Rules();

    private static KindTable<ITypePromotion> Rules()
    {
        var table = new KindTable<ITypePromotion>(nameof(TypePromotions), "Type promotion", "add it with TypePromotions.Register",
            comparer: StringComparer.OrdinalIgnoreCase, newestFirst: true,
            unguarded: "a rule decides the type results are made in: another rule cannot stand in for it once results exist");
        table.RegisterDefault(new StandardPromotion());
        return table;
    }

    /// <summary>Registers <paramref name="rule"/> under its name; it is asked before the rules registered earlier (the library's last).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITypePromotion rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        Table.Register(rule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's rule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered rules' names, in the order they are asked.</summary>
    public static IReadOnlyList<string> Names => Table.Names;

    /// <summary>The rule registered as <paramref name="name"/>, or null.</summary>
    public static ITypePromotion? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The library's rule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITypePromotion? Default(string name) => Table.Default(name);

    /// <summary>Who registered the rule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered rule as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The type of <paramref name="operation"/>'s result over inputs of <paramref name="left"/> and <paramref name="right"/>: the first rule that decides.</summary>
    /// <exception cref="NotSupportedException">No rule decides the pair (the message names it and the rules).</exception>
    public static ElementType Result(ElementType left, ElementType right, Operations.Operation? operation = null)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        foreach (var rule in Table.Kinds)
        {
            if (rule.Promote(operation, left, right) is { } type)
            {
                return type;
            }
        }

        throw new NotSupportedException($"No type promotion decides {left.Name} with {right.Name}{(operation is null ? "" : $" in {operation.Name}")} "
            + $"(rules: {string.Join(", ", Table.Names)}): convert one input first (Tensor.ConvertTo), or register a rule with TypePromotions.Register.");
    }

    /// <summary>The type of <paramref name="operation"/>'s result over inputs of <paramref name="inputs"/>' types, promoted pairwise in order.</summary>
    /// <exception cref="ArgumentException">No input.</exception>
    /// <exception cref="NotSupportedException">No rule decides a pair.</exception>
    public static ElementType Result(ReadOnlySpan<ElementType> inputs, Operations.Operation? operation = null)
    {
        if (inputs.IsEmpty)
        {
            throw new ArgumentException("A result's type needs at least one input's.", nameof(inputs));
        }

        var type = inputs[0];
        for (int i = 1; i < inputs.Length; i++)
        {
            type = Result(type, inputs[i], operation);
        }

        return type;
    }

    // The library's rule, from the types' data alone: the same type stays; a block format (a storage format) takes the
    // other input's type; floats take the one with more bits, and two of the same bits the narrowest registered float
    // that holds every value of both exactly; a float with an integer or a truth value takes the float; integers take the
    // narrowest registered integer that holds both ranges; a truth value with an integer takes the integer.
    private sealed class StandardPromotion : ITypePromotion
    {
        private static readonly ConcurrentDictionary<(ElementType, ElementType), bool> s_holds = new();

        public string Name => Standard;

        public string Summary => "from the types' data: floats over integers, more bits over fewer, the narrowest type holding both exactly; a block format takes the other input's type";

        public ElementType? Promote(Operations.Operation? operation, ElementType left, ElementType right)
        {
            if (ReferenceEquals(left, right))
            {
                return left;
            }

            if (left.IsBlock != right.IsBlock)
            {
                return left.IsBlock ? right : left;
            }

            if (left.IsBlock)
            {
                return null;                                                // two storage formats: the caller widens one first
            }

            if (left.IsFloat && right.IsFloat)
            {
                return left.BlockBits != right.BlockBits && Holds(Wider(left, right), Narrower(left, right))
                    ? Wider(left, right)
                    : ElementTypes.All.Where(t => t.IsFloat && !t.IsBlock && t.BlockBits > Math.Max(left.BlockBits, right.BlockBits) && Holds(t, left) && Holds(t, right))
                        .OrderBy(t => t.BlockBits).FirstOrDefault();
            }

            if (left.IsFloat || right.IsFloat)
            {
                return left.IsFloat ? left : right;
            }

            if (!left.IsInteger || !right.IsInteger)
            {
                return left.IsInteger ? left : right.IsInteger ? right : null;   // a truth value takes the integer
            }

            var low = Int128.Min(left.MinValue, right.MinValue);
            var high = Int128.Max(left.MaxValue, right.MaxValue);
            return ElementTypes.All.Where(t => t.IsInteger && !t.IsBlock && t.MinValue <= low && t.MaxValue >= high)
                .OrderBy(t => t.BlockBits).ThenBy(t => t.IsSigned).FirstOrDefault();
        }

        private static ElementType Wider(ElementType a, ElementType b) => a.BlockBits >= b.BlockBits ? a : b;

        private static ElementType Narrower(ElementType a, ElementType b) => a.BlockBits >= b.BlockBits ? b : a;

        // Whether `wide` holds every value of `narrow` exactly: every code of a type of up to 16 bits stored in `wide` and
        // read back (NaNs aside); a wider `narrow` is held when `wide` has more bits.
        private static bool Holds(ElementType wide, ElementType narrow) => s_holds.GetOrAdd((wide, narrow), static pair =>
        {
            var (w, n) = pair;
            if (ReferenceEquals(w, n))
            {
                return true;
            }

            if (n.BlockBits > 16 || n.ToFloat32 is null || w.ToFloat32 is null || w.FromFloat32 is null)
            {
                return w.BlockBits > n.BlockBits;
            }

            int codes = 1 << n.BlockBits;
            var bytes = new byte[n.ByteCount(codes)];
            for (int c = 0; c < codes; c++)
            {
                if (n.BlockBits == 16)
                {
                    bytes[2 * c] = (byte)c;
                    bytes[2 * c + 1] = (byte)(c >> 8);
                }
                else
                {
                    bytes[c] = (byte)c;
                }
            }

            var values = new float[codes];
            n.Decode(bytes, values);
            var stored = new byte[w.ByteCount(codes)];
            w.Encode(values, stored);
            var back = new float[codes];
            w.Decode(stored, back);
            for (int c = 0; c < codes; c++)
            {
                if (!float.IsNaN(values[c]) && BitConverter.SingleToInt32Bits(values[c]) != BitConverter.SingleToInt32Bits(back[c]))
                {
                    return false;
                }
            }

            return true;
        });
    }
}
