// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The classic control environments as a plug-in: <c>bandit</c> (<see cref="BanditEnvironment"/>), <c>cartpole</c>
/// (<see cref="CartPoleEnvironment"/>, Gymnasium's CartPole-v1), <c>pendulum</c> (<see cref="PendulumEnvironment"/>,
/// Pendulum-v1, a box action) and <c>mountain-car</c> (<see cref="MountainCarEnvironment"/>, MountainCar-v0) in
/// <c>EnvironmentKinds</c>. They observe float arrays, so <c>ModulePolicy</c> (Idrak.Reinforcement) trains in them
/// on-policy, and <c>OffPolicyTrainer</c> (DQN) in the discrete ones. <c>ReinforcementPlugins.RegisterAll</c>
/// registers the plug-in with the package's other families (the idrak tool does at startup); an app may call
/// <c>Register</c> alone, or make an environment directly (<c>new CartPoleEnvironment()</c>).
/// </summary>
public static class ClassicControlPlugin
{
    /// <summary>The <c>bandit</c> kind (its default options; an environment of others is made directly).</summary>
    public static IEnvironmentKind BanditKind { get; } = new ClassicKind(BanditEnvironment.KindName, typeof(int), () => new BanditEnvironment());

    /// <summary>The <c>cartpole</c> kind (Gymnasium's CartPole-v1) (its default options; an environment of others is made directly).</summary>
    public static IEnvironmentKind CartPoleKind { get; } = new ClassicKind(CartPoleEnvironment.KindName, typeof(int), () => new CartPoleEnvironment());

    /// <summary>The <c>pendulum</c> kind (Gymnasium's Pendulum-v1, a box action) (its default options; an environment of others is made directly).</summary>
    public static IEnvironmentKind PendulumKind { get; } = new ClassicKind(PendulumEnvironment.KindName, typeof(float[]), () => new PendulumEnvironment());

    /// <summary>The <c>mountain-car</c> kind (Gymnasium's MountainCar-v0) (its default options; an environment of others is made directly).</summary>
    public static IEnvironmentKind MountainCarKind { get; } = new ClassicKind(MountainCarEnvironment.KindName, typeof(int), () => new MountainCarEnvironment());

    /// <summary>The four kinds.</summary>
    public static IReadOnlyList<IEnvironmentKind> Kinds { get; } = [BanditKind, CartPoleKind, PendulumKind, MountainCarKind];

    // A classic kind: its environments observe float arrays, draw nothing when they are made (every one of a kind with the
    // same options is the same environment) and keep no shared state, so creating one costs its arrays.
    private sealed class ClassicKind(string name, Type actionType, Func<IEnvironment> create) : IEnvironmentKind
    {
        public string Name => name;

        public Type ObservationType => typeof(float[]);

        public Type ActionType => actionType;

        // The kinds' options by key were read through their option schema, which went with the registry: a kind makes its
        // environments with their default options, and an environment of other options is made directly.
        public IEnvironment Create(EnvironmentRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            if (request.Options is { Count: > 0 } options)
            {
                throw new ArgumentException($"The {name} environment kind takes no options by key (given '{string.Join("', '", options.Keys)}'): make the environment with its options directly.", nameof(request));
            }

            return create();
        }
    }
}
