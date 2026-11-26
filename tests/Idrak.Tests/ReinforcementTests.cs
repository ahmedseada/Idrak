// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Reinforcement;

// Plan 27, phase R2: Idrak.Reinforcement, the general part: the run's options and their variables.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ReinforcementGroup =
    [
        ("reinforcement: options read their IDRAK_RL_* variables and refuse bad values naming the variable", ReinforcementOptionVariables),
    ];

    private static void ReinforcementOptionVariables(Device device)
    {
        _ = device;
        string[] names = ["IDRAK_RL_ALGORITHM", "IDRAK_RL_GROUP", "IDRAK_RL_PARALLEL_STEPS", "IDRAK_RL_CLIP", "IDRAK_RL_WARMUP", "IDRAK_RL_GROUP_SCALE", "IDRAK_RL_EPISODES",
            "IDRAK_RL_LEARNING_RATE", "IDRAK_RL_OPTIMIZER", "IDRAK_RL_RESUME"];
        try
        {
            var defaults = ReinforcementOptions.FromEnvironment();
            Check(defaults == new ReinforcementOptions(), "with no variable set, the defaults");
            Environment.SetEnvironmentVariable("IDRAK_RL_ALGORITHM", "grpo");
            Environment.SetEnvironmentVariable("IDRAK_RL_GROUP", "4");
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "off");
            Environment.SetEnvironmentVariable("IDRAK_RL_CLIP", "0.3");
            Environment.SetEnvironmentVariable("IDRAK_RL_WARMUP", "0.1");
            Environment.SetEnvironmentVariable("IDRAK_RL_GROUP_SCALE", "false");
            Environment.SetEnvironmentVariable("IDRAK_RL_RESUME", "runs/rl");
            Environment.SetEnvironmentVariable("IDRAK_RL_LEARNING_RATE", "0.01");
            Environment.SetEnvironmentVariable("IDRAK_RL_OPTIMIZER", "sgd");
            var read = ReinforcementOptions.FromEnvironment(new ReinforcementOptions { Iterations = 7 });
            Check(read.Algorithm == "grpo" && read.GroupSize == 4 && read.ParallelSteps == false && read.Clip == 0.3f && read.ScheduleSettings.WarmupFraction == 0.1f
                  && read.GroupScale == false && read.Resume == "runs/rl" && read.Iterations == 7, $"the variables set their options, the others keep the start's: {read}");
            var policyOptions = ModulePolicyOptions.FromEnvironment();
            Check(policyOptions.LearningRate == 0.01f && policyOptions.Optimizer == "sgd", $"the module policy's variables: {policyOptions}");
            Environment.SetEnvironmentVariable("IDRAK_RL_EPISODES", "zero");
            ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a bad value", "IDRAK_RL_EPISODES=zero");
            Environment.SetEnvironmentVariable("IDRAK_RL_EPISODES", null);
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "sometimes");
            ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a bad switch", "IDRAK_RL_PARALLEL_STEPS=sometimes", "on, off or auto");
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "auto");
            Check(ReinforcementOptions.FromEnvironment(new ReinforcementOptions { ParallelSteps = true }).ParallelSteps is null, "auto measures");
        }
        finally
        {
            foreach (string name in names)
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        var table = Idrak.Cli.Shared.EnvironmentVariables.All.Where(v => v.Name.StartsWith("IDRAK_RL_", StringComparison.Ordinal)).Select(v => v.Name).ToHashSet(StringComparer.Ordinal);
        Check(names.All(table.Contains) && table.Count >= 33, $"the IDRAK_RL_* variables are in the tool's table: {table.Count}");
    }
}
