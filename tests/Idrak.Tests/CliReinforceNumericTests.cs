// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Plan 27, R9: idrak reinforce in an environment of numbers (the classic plug-in's kinds), with no model: the help shows
// the path.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CliReinforceNumericGroup =
    [
        ("cli reinforce: an environment of numbers: the help shows the path", d => { CpuOnly(d); CliReinforceNumericUsage(); }),
    ];

    private static void CliReinforceNumericUsage()
    {
        var help = TrainCli("help", "reinforce").Out;
        Check(help.Contains("idrak reinforce --env cartpole --algorithm dqn --steps 50000 -o runs/cartpole-dqn", StringComparison.Ordinal)
              && help.Contains("--hidden W,W", StringComparison.Ordinal) && help.Contains("IDRAK_RL_DQN_STEPS", StringComparison.Ordinal), "help reinforce shows the numeric path");
        Check(TrainCli("help", "reinforcement").Out.Contains("idrak reinforce --env cartpole --algorithm ppo --iterations 50 -o runs/cartpole", StringComparison.Ordinal),
            "help reinforcement shows the numeric path");
    }
}
