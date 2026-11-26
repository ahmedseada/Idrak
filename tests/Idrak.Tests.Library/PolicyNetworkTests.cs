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
        ("reinforcement: policy networks: ModulePolicyOptions' hidden widths, activation and value head have defaults, read IDRAK_RL_HIDDEN, IDRAK_RL_ACTIVATION and IDRAK_RL_VALUE_HEAD (bad values refused naming them), and the automatic value head follows the run's algorithm and estimator", d => { CpuOnly(d); PolicyNetworkOptions(); }),
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
}
