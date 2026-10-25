// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// A registry's refusal of a name nothing is registered as: which registry, the name asked for, the names it holds,
/// and the message the registry words ("Speech family 'x' is not registered (registered: a, b); register it with ...").
/// It is a <see cref="NotSupportedException"/>, as every registry's refusal has been, so code that catches that keeps
/// working; code that must tell a refusal from another failure (a command line that adds "load its plug-in with -P")
/// catches this type, never the message's text.
/// </summary>
public sealed class NotRegisteredException : NotSupportedException
{
    /// <summary>The refusal of <paramref name="name"/> by <paramref name="registry"/>, with the registry's own message.</summary>
    /// <param name="registry">The registry's name, as its reports name it ("SpeechFamilies").</param>
    /// <param name="name">The name nothing is registered as.</param>
    /// <param name="registered">The names the registry holds, in its order (copied).</param>
    /// <param name="message">The registry's message, which names the registered ones and what to do.</param>
    public NotRegisteredException(string registry, string name, IEnumerable<string> registered, string message)
        : base(message)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(registry);
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(registered);
        Registry = registry;
        Name = name;
        Registered = [.. registered];
    }

    /// <summary>The registry's name, as its reports name it ("SpeechFamilies").</summary>
    public string Registry { get; }

    /// <summary>The name nothing is registered as.</summary>
    public string Name { get; }

    /// <summary>The names the registry held when it refused, in its order.</summary>
    public IReadOnlyList<string> Registered { get; }
}
