// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Reinforcement;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the advantage estimators
/// (<c>AdvantageEstimators</c>) and policy losses (<c>PolicyLosses</c>) of the general algorithms, none yet.
/// <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once per registry, before that
/// registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = [];

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
