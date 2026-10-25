// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The classic control environments as a plug-in: <c>bandit</c> (<see cref="BanditEnvironment"/>), <c>cartpole</c>
/// (<see cref="CartPoleEnvironment"/>, Gymnasium's CartPole-v1), <c>pendulum</c> (<see cref="PendulumEnvironment"/>,
/// Pendulum-v1, a box action) and <c>mountain-car</c> (<see cref="MountainCarEnvironment"/>, MountainCar-v0) in
/// <see cref="EnvironmentKinds"/>. They observe float arrays, so <c>ModulePolicy</c> (Idrak.Reinforcement) trains in them
/// on-policy, and <c>OffPolicyTrainer</c> (DQN) in the discrete ones. <see cref="ReinforcementPlugins.RegisterAll"/>
/// registers the plug-in with the package's other families (the idrak tool does at startup); an app may call
/// <see cref="Register"/> alone, or make an environment directly (<c>new CartPoleEnvironment()</c>).
/// </summary>
public static class ClassicControlPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("classic"; listed in <see cref="ReinforcementPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new ClassicRegistration();

    /// <summary>The <c>bandit</c> kind: options <see cref="BanditOptions.Keys"/>.</summary>
    public static IEnvironmentKind BanditKind { get; } = new ClassicKind(BanditEnvironment.KindName,
        "a multi-armed bandit: arms, means and spreads as options (Sutton and Barto's 10-armed testbed by default)", BanditOptions.Schema, typeof(int),
        options => new BanditEnvironment(BanditOptions.FromOptions(options)));

    /// <summary>The <c>cartpole</c> kind (Gymnasium's CartPole-v1): options <see cref="CartPoleOptions.Keys"/>.</summary>
    public static IEnvironmentKind CartPoleKind { get; } = new ClassicKind(CartPoleEnvironment.KindName,
        "Gymnasium's CartPole-v1: balance a pole on a cart, 1 a step, 500 steps at most", CartPoleOptions.Schema, typeof(int),
        options => new CartPoleEnvironment(CartPoleOptions.FromOptions(options)));

    /// <summary>The <c>pendulum</c> kind (Gymnasium's Pendulum-v1, a box action): options <see cref="PendulumOptions.Keys"/>.</summary>
    public static IEnvironmentKind PendulumKind { get; } = new ClassicKind(PendulumEnvironment.KindName,
        "Gymnasium's Pendulum-v1: swing a pendulum up with a torque in [-2, 2] (a box action), 200 steps", PendulumOptions.Schema, typeof(float[]),
        options => new PendulumEnvironment(PendulumOptions.FromOptions(options)));

    /// <summary>The <c>mountain-car</c> kind (Gymnasium's MountainCar-v0): options <see cref="MountainCarOptions.Keys"/>.</summary>
    public static IEnvironmentKind MountainCarKind { get; } = new ClassicKind(MountainCarEnvironment.KindName,
        "Gymnasium's MountainCar-v0: rock an under-powered car up to the flag, -1 a step, 200 steps at most", MountainCarOptions.Schema, typeof(int),
        options => new MountainCarEnvironment(MountainCarOptions.FromOptions(options)));

    /// <summary>The four kinds, in the order the plug-in registers them.</summary>
    public static IReadOnlyList<IEnvironmentKind> Kinds { get; } = [BanditKind, CartPoleKind, PendulumKind, MountainCarKind];

    /// <summary>Whether the plug-in is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the four kinds (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters them, leaving <see cref="EnvironmentKinds"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class ClassicRegistration() : IdrakPlugin("classic")
    {
        protected override void RegisterParts()
        {
            foreach (var kind in Kinds)
            {
                Part(() => EnvironmentKinds.Register(kind), () => EnvironmentKinds.Unregister(kind.Name));
            }
        }
    }

    // A classic kind: its environments observe float arrays, draw nothing when they are made (every one of a kind with the
    // same options is the same environment) and keep no shared state, so creating one costs its arrays.
    private sealed class ClassicKind(string name, string summary, OptionSchema options, Type actionType,
        Func<IReadOnlyDictionary<string, string>?, IEnvironment> create) : IEnvironmentKind, IOptionSchema
    {
        public string Name => name;

        public string Summary => summary;

        public OptionSchema Options => options;

        public IReadOnlySet<string> Keys => options.Keys;

        public Type ObservationType => typeof(float[]);

        public Type ActionType => actionType;

        public IEnvironment Create(EnvironmentRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            return create(request.Options);
        }
    }
}
