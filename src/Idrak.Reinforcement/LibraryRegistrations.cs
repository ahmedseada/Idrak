// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the advantage estimators gae,
/// group, leave-one-out and returns (<see cref="AdvantageEstimators"/>) and the policy losses reinforce, ppo and grpo
/// (<see cref="PolicyLosses"/>). <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once
/// per registry, before that registry is first used, so they are the slots' library defaults.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(AdvantageEstimators)] = Estimators.Register,
        [typeof(PolicyLosses)] = PolicyLossTerms.Register,
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
