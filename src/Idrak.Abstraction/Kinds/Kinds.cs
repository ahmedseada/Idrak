// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// What every registered kind is: listed, described and found by name. A registry of kinds keeps them in a
/// <see cref="KindTable{TKind}"/>, which registers each under its <see cref="Name"/>, so a name never drifts from what
/// it registers, and lists names and summaries alike for every <c>--list</c>, help page and "not registered" error.
/// </summary>
public interface IKind
{
    /// <summary>The kind's name in its registry: what a command line, a settings file or code asks for.</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }
}

/// <summary>
/// A kind that reads options of its own, by key: the keys it takes, so any other is refused naming these. The set is
/// made once, an <see cref="OptionKeys"/> (a <see cref="System.Collections.Frozen.FrozenSet{T}"/> that also keeps the
/// declared order for listings), so a key read from text is checked by span without building a list per call.
/// </summary>
public interface IConfigurableKind : IKind
{
    /// <summary>The option keys the kind takes; any other is refused, naming these.</summary>
    IReadOnlySet<string> Keys { get; }
}

/// <summary>
/// A kind that makes something from one request record: a field added to the request later never breaks an
/// implementation. The request may be a ref struct (a span-carrying request read from input).
/// </summary>
/// <typeparam name="TRequest">What the kind is asked with: its options and whatever else it needs.</typeparam>
/// <typeparam name="T">What it makes.</typeparam>
public interface IKind<in TRequest, out T> : IConfigurableKind where TRequest : allows ref struct
{
    /// <summary>A new <typeparamref name="T"/> made for <paramref name="request"/>.</summary>
    /// <exception cref="ArgumentException">An option the kind does not take (the message names it and <see cref="IConfigurableKind.Keys"/>), or a bad value.</exception>
    T Create(TRequest request);
}
