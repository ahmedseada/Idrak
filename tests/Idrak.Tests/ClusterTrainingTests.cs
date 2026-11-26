// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;

// Plan 23, phase C2: data parallel: the split of a step's draws between the ranks, and idrak cluster sim's usage.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterTrainingGroup =
    [
        ("cluster: a step's draws go round the ranks while their shares are equal and in proportion once measured", ClusterDrawShares),
        ("cluster: cli cluster sim: another command and a missing run folder are usage errors", ClusterCliSim),
    ];

    private static void ClusterDrawShares(Device device)
    {
        if (!FirstRun(nameof(ClusterDrawShares)))
        {
            return;
        }

        long[] sizes = [5, 9, 5, 2, 5, 5, 7];
        Check(DrawShares.Owners(sizes, DrawShares.Even(3)).SequenceEqual([0, 1, 2, 0, 1, 2, 0]), "equal shares: rank r takes draws k·W + r whatever their sizes");
        var owners = DrawShares.Owners([.. Enumerable.Repeat(10L, 100)], [0.75, 0.25]);
        Check(owners.Count(o => o == 0) == 75, $"shares of 3:1 give rank 0 three draws in four: {owners.Count(o => o == 0)}");
        Check(DrawShares.Owners(sizes, [0.6, 0.4]).SequenceEqual(DrawShares.Owners(sizes, [0.6, 0.4])), "the same shares give the same owners");
    }

    private static void ClusterCliSim(Device device)
    {
        if (!FirstRun(nameof(ClusterCliSim)))
        {
            return;
        }

        string folder = TempFolder("cluster-cli-sim");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            string run = Path.Combine(folder, "run");
            (int Code, string Output, string Error) Idrak(params string[] args)
            {
                using var output = new StringWriter(CultureInfo.InvariantCulture);
                using var error = new StringWriter(CultureInfo.InvariantCulture);
                int code = CommandLine.Run(args, output, error);
                return (code, output.ToString(), error.ToString());
            }

            var (other, _, otherError) = Idrak("cluster", "sim", "--topology", topology, "--", "chat", TinyLlama, "-o", run);
            Check(other == ExitCodes.Usage && otherError.Contains("pretrain", StringComparison.Ordinal), $"another command is a usage error naming the one it runs: {otherError}");
            var (missing, _, missingError) = Idrak("cluster", "sim", "--topology", topology, "--", "pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"));
            Check(missing == ExitCodes.Usage && missingError.Contains("-o RUN", StringComparison.Ordinal), $"a missing run folder is a usage error naming -o: {missingError}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
