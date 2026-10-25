// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Layers;

namespace Idrak.Reinforcement;

/// <summary>
/// Networks and policies made from settings (<see cref="ModulePolicyOptions"/>: the hidden widths, the activation, a value
/// head) for an environment kind's spaces, for a command or an app that trains in an environment of numbers without writing
/// a network itself (<c>idrak reinforce --env cartpole</c>): a <see cref="ModulePolicy{TAction}"/> for the on-policy trainer
/// (categorical over a discrete action space, Gaussian over a box) and a Q-network with its target for the off-policy one
/// (<see cref="OffPolicyTrainer"/>, discrete actions only). Every weight is drawn from the seed given, so the same seed and
/// settings make the same networks. An app that needs another architecture builds its modules and passes them to
/// <see cref="ModulePolicy"/>'s Categorical or Gaussian, or to the off-policy trainer.
/// </summary>
public static class PolicyNetworks
{
    /// <summary>tanh, the default activation (as Stable-Baselines3's policies have it).</summary>
    public const string Tanh = "tanh";

    // The streams the weights are drawn from (Seeds.Derive), apart from the trainers'.
    private const int ActorStream = 41, CriticStream = 42, QStream = 43, TargetStream = 44;

    /// <summary>The activations a network made here can have between its layers: tanh, relu, gelu, sigmoid (the library's layers).</summary>
    public static IReadOnlyList<string> Activations { get; } = [Tanh, "relu", "gelu", "sigmoid"];

    /// <summary>
    /// The observation and action spaces of <paramref name="environmentKind"/>'s environments with
    /// <paramref name="environmentOptions"/>: one environment made and disposed (the options checked as the kind checks them).
    /// </summary>
    /// <exception cref="ArgumentException">An option the kind does not take, or a bad value (the kind's message).</exception>
    public static (Space Observations, Space Actions) Spaces(IEnvironmentKind environmentKind, IReadOnlyDictionary<string, string>? environmentOptions = null)
    {
        ArgumentNullException.ThrowIfNull(environmentKind);
        using var probe = environmentKind.Create(new EnvironmentRequest(environmentOptions));
        return (probe.ObservationSpace, probe.ActionSpace);
    }

    /// <summary>
    /// A fully connected network: <paramref name="inputs"/> to each of <see cref="ModulePolicyOptions.Hidden"/> in turn (each
    /// followed by <see cref="ModulePolicyOptions.Activation"/>) to <paramref name="outputs"/>, its weights drawn from
    /// <paramref name="random"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An activation that is not one of <see cref="Activations"/>, or a width below 1.</exception>
    public static Sequential Network(int inputs, int outputs, ModulePolicyOptions options, Device? device = null, Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputs);
        _ = Activation(options.Activation);                                              // an unknown name fails before any weight is drawn
        var network = new Sequential();
        int width = inputs;
        foreach (int size in options.Hidden)
        {
            if (size < 1)
            {
                throw new ArgumentException($"A hidden width is at least 1, not {size} (ModulePolicyOptions.Hidden).", nameof(options));
            }

            network.Add(new Linear(width, size, device: device, random: random));
            network.Add(Activation(options.Activation));
            width = size;
        }

        network.Add(new Linear(width, outputs, device: device, random: random));
        return network;
    }

    /// <summary>
    /// A categorical policy for a discrete action space: an actor from the observations to a logit per action and, with
    /// <paramref name="valueHead"/>, a critic to one value, both <see cref="Network"/>s of <paramref name="options"/>; with
    /// <paramref name="reference"/>, a frozen copy of the actor's starting weights (a KL penalty's reference).
    /// </summary>
    /// <param name="observationSpace">The environment's observations: a box.</param>
    /// <param name="actionSpace">Its actions: discrete.</param>
    /// <param name="options">The networks' shape and the optimizer.</param>
    /// <param name="valueHead">Whether the policy has a critic (<see cref="ModulePolicyOptions.ValueHeadFor"/> decides it for a run).</param>
    /// <param name="device">Where the weights live; null for the default device.</param>
    /// <param name="seed">The weights' draws: the same seed makes the same policy.</param>
    /// <param name="reference">Whether to keep a frozen reference of the actor.</param>
    /// <exception cref="ArgumentException">An observation space that is not a box, an action space that is not discrete, or a bad network setting.</exception>
    public static ModulePolicy<int> Categorical(Space observationSpace, Space actionSpace, ModulePolicyOptions options, bool valueHead, Device? device = null, int seed = 0,
        bool reference = false)
    {
        ArgumentNullException.ThrowIfNull(actionSpace);
        if (actionSpace.Kind != SpaceKind.Discrete)
        {
            throw new ArgumentException($"A categorical policy chooses among discrete actions, not {actionSpace}; use PolicyNetworks.Gaussian for a box.", nameof(actionSpace));
        }

        var (actor, critic, frozen) = Modules(observationSpace, actionSpace.Count, options, valueHead, device, seed, reference);
        return ModulePolicy.Categorical(observationSpace, actionSpace, actor, critic, frozen, options);
    }

    /// <summary>
    /// A Gaussian policy for a box of actions: an actor from the observations to a mean per action element (each drawn with a
    /// learned spread from <see cref="ModulePolicyOptions.InitialLogStd"/>) and, with <paramref name="valueHead"/>, a critic,
    /// both <see cref="Network"/>s of <paramref name="options"/>.
    /// </summary>
    /// <param name="observationSpace">The environment's observations: a box.</param>
    /// <param name="actionSpace">Its actions: a box.</param>
    /// <param name="options">The networks' shape and the optimizer.</param>
    /// <param name="valueHead">Whether the policy has a critic.</param>
    /// <param name="device">Where the weights live; null for the default device.</param>
    /// <param name="seed">The weights' draws: the same seed makes the same policy.</param>
    /// <param name="reference">Whether to keep a frozen reference of the actor.</param>
    /// <exception cref="ArgumentException">An observation or action space that is not a box, or a bad network setting.</exception>
    public static ModulePolicy<float[]> Gaussian(Space observationSpace, Space actionSpace, ModulePolicyOptions options, bool valueHead, Device? device = null, int seed = 0,
        bool reference = false)
    {
        ArgumentNullException.ThrowIfNull(actionSpace);
        if (actionSpace.Kind != SpaceKind.Box)
        {
            throw new ArgumentException($"A Gaussian policy acts in a box, not {actionSpace}; use PolicyNetworks.Categorical for discrete actions.", nameof(actionSpace));
        }

        var (actor, critic, frozen) = Modules(observationSpace, actionSpace.Size, options, valueHead, device, seed, reference);
        return ModulePolicy.Gaussian(observationSpace, actionSpace, actor, critic, frozen, options);
    }

    /// <summary>
    /// The online Q-network and its target for <see cref="OffPolicyTrainer"/>: two <see cref="Network"/>s of
    /// <paramref name="options"/> from the observations to a value per action (the target takes the online one's weights
    /// when the trainer is made).
    /// </summary>
    /// <param name="observationSpace">The environment's observations: a box.</param>
    /// <param name="actionSpace">Its actions: discrete (a Q-network values each choice).</param>
    /// <param name="options">The networks' shape (the optimizer is the trainer's, <see cref="OffPolicyOptions"/>).</param>
    /// <param name="device">Where the weights live; null for the default device.</param>
    /// <param name="seed">The weights' draws: the same seed makes the same networks.</param>
    /// <exception cref="ArgumentException">An observation space that is not a box, or an action space that is not discrete (the message names it).</exception>
    public static (Sequential Network, Sequential Target) QNetworks(Space observationSpace, Space actionSpace, ModulePolicyOptions options, Device? device = null, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(observationSpace);
        ArgumentNullException.ThrowIfNull(actionSpace);
        ArgumentNullException.ThrowIfNull(options);
        if (observationSpace.Kind != SpaceKind.Box)
        {
            throw new ArgumentException($"A Q-network reads float observations (a box), not {observationSpace}.", nameof(observationSpace));
        }

        if (actionSpace.Kind != SpaceKind.Discrete)
        {
            throw new ArgumentException($"A Q-network values each of a discrete set of actions; {actionSpace} is not one: train a box of actions on-policy (a Gaussian policy).",
                nameof(actionSpace));
        }

        var network = Network(observationSpace.Size, actionSpace.Count, options, device, new Random(Seeds.Derive(seed, QStream)));
        var target = Network(observationSpace.Size, actionSpace.Count, options, device, new Random(Seeds.Derive(seed, TargetStream)));
        return (network, target);
    }

    // The actor (to `outputs`), the critic (to one value) when asked, and a frozen reference of the actor's shape when asked.
    private static (Sequential Actor, Sequential? Critic, Sequential? Reference) Modules(Space observationSpace, int outputs, ModulePolicyOptions options, bool valueHead,
        Device? device, int seed, bool reference)
    {
        ArgumentNullException.ThrowIfNull(observationSpace);
        ArgumentNullException.ThrowIfNull(options);
        if (observationSpace.Kind != SpaceKind.Box)
        {
            throw new ArgumentException($"A module policy reads float observations (a box), not {observationSpace}.", nameof(observationSpace));
        }

        int inputs = observationSpace.Size;
        var actor = Network(inputs, outputs, options, device, new Random(Seeds.Derive(seed, ActorStream)));
        var critic = valueHead ? Network(inputs, 1, options, device, new Random(Seeds.Derive(seed, CriticStream))) : null;
        var frozen = reference ? Network(inputs, outputs, options, device, new Random(Seeds.Derive(seed, ActorStream))) : null;   // takes the actor's weights
        return (actor, critic, frozen);
    }

    private static Module Activation(string name) => name.ToLowerInvariant() switch
    {
        Tanh => new Layers.Tanh(),
        "relu" => new ReLU(),
        "gelu" => new GELU(),
        "sigmoid" => new Sigmoid(),
        _ => throw new ArgumentException($"The activation '{name}' is not one of {string.Join(", ", Activations)} (ModulePolicyOptions.Activation)."),
    };
}
