// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cluster;
using Idrak.Networking.Training;
using Idrak.Reinforcement;

// Plan 27, R11: the cluster dashboard steers reinforce. What runs in one process: the pass rate's setting and a launcher's
// control file, which holds a run at a boundary while paused and stops it there.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterControlGroup =
    [
        ("cluster control: reinforce: the pass rate's setting; in one process the control-file steering pauses, goes on and stops at a boundary (read at most every poll, a cancellation ending a pause), rank 0's state file saying so", ControlRegistry),
    ];

    private static bool ControlState(string file, string state, long step) => ClusterControlFile.ReadState(file) is { } said && said.State == state && said.Step == step;

    private static void ControlRegistry(Device device)
    {
        if (!FirstRun(nameof(ControlRegistry)))
        {
            return;
        }

        // The pass rate's setting.
        Check(new ReinforcementOptions().PassReturn == 1f, "an episode passes at a return of 1 by default");
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "0.5" }, () => Check(ReinforcementOptions.FromEnvironment().PassReturn == 0.5f, "IDRAK_RL_PASS_RETURN=0.5"));
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "none" }, () => Check(ReinforcementOptions.FromEnvironment().PassReturn is null, "IDRAK_RL_PASS_RETURN=none: no pass rate"));
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "most" },
            () => ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a pass return that is not a number", "IDRAK_RL_PASS_RETURN=most"));

        // One process: the control file.
        string folder = TempFolder("cluster-control");
        try
        {
            string file = Path.Combine(folder, "control");
            ClusterOptions Options(double poll) => new() { Simulation = new SimulationOptions { Control = file, ControlPoll = TimeSpan.FromSeconds(poll) } };
            var steering = new ControlFileSteeringProvider().Create(null, Options(0.01))!;
            Check(steering.Name == "control-file" && steering.Boundary(0) == ClusterControlCommand.Run && ClusterControlFile.ReadState(file) is null,
                "no file yet: the run goes on, nothing reported");
            ClusterControlFile.Write(file, ClusterControlCommand.Pause);
            Thread.Sleep(30);
            var resume = Task.Run(() =>
            {
                bool held = UiWait(() => ControlState(file, "paused", 1), 30);
                ClusterControlFile.Write(file, ClusterControlCommand.Run);
                return held;
            });
            Check(steering.Boundary(1) == ClusterControlCommand.Run && resume.Result && ControlState(file, "running", 1), "paused at boundary 1, rank 0 reports it, then goes on");
            ClusterControlFile.Write(file, ClusterControlCommand.Stop);
            Thread.Sleep(30);
            Check(steering.Boundary(2) == ClusterControlCommand.Stop && ControlState(file, "stopped", 2), "stopped at boundary 2, reported");
            Thread.Sleep(30);
            Check(steering.Stops(3, null) && !((IRunSteering?)null).Stops(3, null), "in one process a stop is the trainer's at once; nothing steering: no stop");

            // Read at most every poll: a stop written right after a read waits for the next.
            ClusterControlFile.Write(file, ClusterControlCommand.Run);
            var slow = new ControlFileSteeringProvider().Create(null, Options(5))!;
            Check(slow.Boundary(0) == ClusterControlCommand.Run, "the first boundary reads the file");
            ClusterControlFile.Write(file, ClusterControlCommand.Stop);
            Check(slow.Boundary(1) == ClusterControlCommand.Run, "within the poll the file is not read again");

            // A cancellation while paused ends the pause as a stop.
            ClusterControlFile.Write(file, ClusterControlCommand.Pause);
            using var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
            Check(new ControlFileSteeringProvider().Create(null, Options(0.05))!.Boundary(4, cancel.Token) == ClusterControlCommand.Stop && ControlState(file, "stopped", 4),
                "Ctrl+C while paused stops the run there");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
