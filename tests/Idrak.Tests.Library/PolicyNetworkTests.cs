// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Layers;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins.Classic;

// Plan 27, R9: the builders idrak reinforce uses for an environment of numbers (PolicyNetworks: a module policy and a
// Q-network from ModulePolicyOptions' hidden widths, activation and value head) and the options' variables.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] PolicyNetworkGroup =
    [
        ("reinforcement: policy networks: ModulePolicyOptions' hidden widths, activation and value head have defaults, read IDRAK_RL_HIDDEN, IDRAK_RL_ACTIVATION and IDRAK_RL_VALUE_HEAD (bad values refused naming them), and the automatic value head follows the run's algorithm and estimator", d => { if (d == Device.Cpu) PolicyNetworkOptions(); }),
        ("reinforcement: policy networks: PolicyNetworks builds a network of the options' shape, a categorical or Gaussian policy for an environment kind's spaces (a critic and a reference when asked) and DQN's Q-network and target, the same weights from the same seed; a box refused for a Q-network naming it", PolicyNetworkBuilders),
    ];

    internal static void PolicyNetworkOptions()
    {
        string[] names = ["IDRAK_RL_HIDDEN", "IDRAK_RL_ACTIVATION", "IDRAK_RL_VALUE_HEAD"];
        var defaults = new ModulePolicyOptions();
        Check(defaults.Hidden.SequenceEqual([64, 64]) && defaults.Activation == "tanh" && defaults.ValueHead is null, $"the defaults: {string.Join(",", defaults.Hidden)} {defaults.Activation}");
        Check(ModulePolicyOptions.ParseHidden("32, 16").SequenceEqual([32, 16]) && ModulePolicyOptions.ParseHidden("none").Count == 0 && ModulePolicyOptions.ParseHidden(" ").Count == 0,
            "ParseHidden: widths, none, empty");
        ThrowsNaming<ArgumentException>(() => ModulePolicyOptions.ParseHidden("64,0"), "a width of 0", "64,0", "none");
        try
        {
            WithEnvironment(names.ToDictionary(n => n, _ => (string?)null), () =>
                Check(ModulePolicyOptions.FromEnvironment() == new ModulePolicyOptions(), "with no variable set, the defaults (equal records)"));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_HIDDEN"] = "32,8", ["IDRAK_RL_ACTIVATION"] = "ReLU", ["IDRAK_RL_VALUE_HEAD"] = "off" }, () =>
            {
                var read = ModulePolicyOptions.FromEnvironment();
                Check(read.Hidden.SequenceEqual([32, 8]) && read.Activation == "relu" && read.ValueHead == false, $"the variables: {string.Join(",", read.Hidden)} {read.Activation} {read.ValueHead}");
            });
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_HIDDEN"] = "none", ["IDRAK_RL_VALUE_HEAD"] = "auto" }, () =>
                Check(ModulePolicyOptions.FromEnvironment(new ModulePolicyOptions { ValueHead = true }) is { Hidden.Count: 0, ValueHead: null }, "none and auto"));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_HIDDEN"] = "wide" }, () =>
                ThrowsNaming<ArgumentException>(() => ModulePolicyOptions.FromEnvironment(), "a bad width", "IDRAK_RL_HIDDEN=wide"));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_ACTIVATION"] = "swish" }, () =>
                ThrowsNaming<ArgumentException>(() => ModulePolicyOptions.FromEnvironment(), "a bad activation", "IDRAK_RL_ACTIVATION=swish", "tanh, relu, gelu, sigmoid"));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_VALUE_HEAD"] = "maybe" }, () =>
                ThrowsNaming<ArgumentException>(() => ModulePolicyOptions.FromEnvironment(), "a bad switch", "IDRAK_RL_VALUE_HEAD=maybe"));
        }
        finally
        {
            foreach (string name in names)
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        // The automatic value head: where the run's advantages use values.
        var auto = new ModulePolicyOptions();
        Check(auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "ppo" }) && !auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "grpo" })
              && !auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "reinforce" }) && auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "reinforce", Estimator = "gae" })
              && !auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "ppo", Estimator = "returns" })
              && auto.ValueHeadFor(new ReinforcementOptions { Algorithm = "reinforce", Estimator = "returns", EstimatorOptions = new Dictionary<string, string> { ["baseline"] = "value" } }),
            "the automatic value head: ppo, gae and a value baseline have one; grpo, reinforce and plain returns do not");
        Check(!new ModulePolicyOptions { ValueHead = false }.ValueHeadFor(new ReinforcementOptions { Algorithm = "ppo" })
              && new ModulePolicyOptions { ValueHead = true }.ValueHeadFor(new ReinforcementOptions { Algorithm = "grpo" }), "a value head set wins");
    }

    internal static void PolicyNetworkBuilders(Device device)
    {
        static float[] Weights(Module module) => [.. module.Parameters().SelectMany(p => p.ToArray())];

        // A network of the options' shape.
        var shape = new ModulePolicyOptions { Hidden = [8, 6], Activation = "relu" };
        using (var network = PolicyNetworks.Network(4, 3, shape, device, new Random(1)))
        using (var again = PolicyNetworks.Network(4, 3, shape, device, new Random(1)))
        {
            var layers = network.Children().ToList();
            Check(layers.Count == 5 && layers[0] is Linear { InFeatures: 4, OutFeatures: 8 } && layers[1] is ReLU && layers[2] is Linear { InFeatures: 8, OutFeatures: 6 }
                  && layers[3] is ReLU && layers[4] is Linear { InFeatures: 6, OutFeatures: 3 }, $"the layers: {string.Join(", ", layers.Select(l => l.GetType().Name))}");
            Check(Weights(network).SequenceEqual(Weights(again)), "the same draws give the same weights");
        }

        using (var linear = PolicyNetworks.Network(1, 4, new ModulePolicyOptions { Hidden = [] }, device, new Random(1)))
        {
            Check(linear.Children().Single() is Linear { InFeatures: 1, OutFeatures: 4 }, "no hidden width: one linear layer");
        }

        ThrowsNaming<ArgumentException>(() => PolicyNetworks.Network(4, 2, new ModulePolicyOptions { Activation = "swish" }, device), "an unknown activation", "swish", "tanh, relu, gelu, sigmoid");

        // The spaces of a kind, and a policy for them.
        var (cartObservations, cartActions) = PolicyNetworks.Spaces(ClassicControlPlugin.CartPoleKind);
        Check(cartObservations.Kind == SpaceKind.Box && cartObservations.Size == 4 && cartActions.Equals(Space.Discrete(2)), $"cartpole's spaces: {cartObservations}, {cartActions}");
        ThrowsNaming<ArgumentException>(() => PolicyNetworks.Spaces(ClassicControlPlugin.CartPoleKind, new Dictionary<string, string> { ["nope"] = "1" }), "an option the kind does not take", "nope");
        var options = new ModulePolicyOptions { Hidden = [16] };
        using (var policy = PolicyNetworks.Categorical(cartObservations, cartActions, options, valueHead: true, device, seed: 5))
        using (var same = PolicyNetworks.Categorical(cartObservations, cartActions, options, valueHead: true, device, seed: 5))
        using (var other = PolicyNetworks.Categorical(cartObservations, cartActions, options, valueHead: false, device, seed: 6, reference: true))
        {
            Check(policy.Critic is not null && other.Critic is null && other.HasReference && !policy.HasReference, "a critic and a reference when asked");
            Check(Weights(policy.Actor).SequenceEqual(Weights(same.Actor)) && Weights(policy.Critic!).SequenceEqual(Weights(same.Critic!)), "the same seed makes the same policy");
            Check(!Weights(policy.Actor).SequenceEqual(Weights(other.Actor)) && !Weights(policy.Actor).SequenceEqual(Weights(policy.Critic!).Take(Weights(policy.Actor).Length)),
                "another seed, and the critic, draw other weights");
            var acted = policy.ActAsync([new float[4], [0.01f, 0.02f, 0.03f, 0.04f]], new ActionSampling { Seed = 1 }).AsTask().GetAwaiter().GetResult();
            Check(acted.Actions.Count == 2 && acted.Actions.All(a => a is 0 or 1) && acted.Records.All(r => float.IsFinite(r.Value)), "the policy acts with a value per row");
        }

        var (pendulumObservations, pendulumActions) = PolicyNetworks.Spaces(ClassicControlPlugin.PendulumKind);
        using (var gaussian = PolicyNetworks.Gaussian(pendulumObservations, pendulumActions, options, valueHead: false, device, seed: 2))
        {
            var acted = gaussian.ActAsync([new float[3]], new ActionSampling { Seed = 1 }).AsTask().GetAwaiter().GetResult();
            Check(gaussian.Critic is null && acted.Actions[0].Length == 1 && pendulumActions.Contains(acted.Actions[0]), $"a Gaussian policy for the pendulum's box: {acted.Actions[0][0]}");
        }

        ThrowsNaming<ArgumentException>(() => PolicyNetworks.Categorical(pendulumObservations, pendulumActions, options, false, device), "a categorical policy for a box", "box(");
        ThrowsNaming<ArgumentException>(() => PolicyNetworks.Gaussian(cartObservations, cartActions, options, false, device), "a Gaussian policy for discrete actions", "discrete(2)");

        // DQN's networks: one value per action, the target of the same architecture; a box refused naming it.
        var (online, target) = PolicyNetworks.QNetworks(cartObservations, cartActions, options, device, seed: 3);
        var (online2, target2) = PolicyNetworks.QNetworks(cartObservations, cartActions, options, device, seed: 3);
        Check(Weights(online).SequenceEqual(Weights(online2)) && !Weights(online).SequenceEqual(Weights(target)), "the same seed makes the same Q-network; the target draws its own");
        online2.Dispose();
        target2.Dispose();
        using (var trainer = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, online, target, new OffPolicyOptions { Steps = 10, ReplayCapacity = 16 }))
        {
            Check(trainer.Values(new float[4]).Length == 2 && Weights(trainer.Target).SequenceEqual(Weights(trainer.Network)), "the off-policy trainer takes them; the target starts as the online network");
        }

        ThrowsNaming<ArgumentException>(() => PolicyNetworks.QNetworks(pendulumObservations, pendulumActions, options, device), "a Q-network for a box", "box(", "discrete");
    }
}
