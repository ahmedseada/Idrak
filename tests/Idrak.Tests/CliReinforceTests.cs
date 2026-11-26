// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli;

// Plan 27, R5: idrak reinforce and idrak distill --on-policy (R3's OnPolicyDistiller): the help topic and the usage errors
// cluster sim and distill give before a model loads. R3c: idrak reinforce under idrak cluster sim (an in-process cluster:
// a thread per rank, no child process).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliReinforceGroup =
    [
        ("cli reinforce: help reinforcement names idrak reinforce", d => { CpuOnly(d); CliReinforceHelp(); }),
        ("cli reinforce: cluster sim refuses reinforce without -o and distill without --on-policy, naming why (on-policy distillation runs there since R10: distillation cluster:)", CliReinforceCluster),
        ("cli distill: --on-policy's usage errors exit with 2: with --generate, a --mix above 1", CliDistillOnPolicy),
    ];

    private static void CliReinforceHelp() =>
        Check(TrainCli("help", "reinforcement").Out.Contains("idrak reinforce", StringComparison.Ordinal), "help reinforcement");

    // A tiny model and a prompts file of question-and-answer rows (read as idrak tune reads them).
    private static (string Model, string Data) ReinforceFixture(string folder)
    {
        string model = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        string data = Path.Combine(folder, "prompts.jsonl");
        File.WriteAllLines(data, ["{\"question\": \"Q\", \"answer\": \"yes\"}", "{\"question\": \"R\", \"answer\": \"no\"}"]);
        return (model, data);
    }

    private static string[] TinySession(Device device, string cache) =>
        ["--rank", "4", "--alpha", "8", "--lr", "1e-2", "--batch-tokens", "256", "--max-length", "64", "--max-new", "4", "-d", device.ToString(), "--cache", cache];

    private static void CliReinforceCluster(Device device)
    {
        string folder = TempFolder();
        var (model, data) = ReinforceFixture(folder);
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(device, 2).ToJson());
            string alone = Path.Combine(folder, "alone");

            // Usage errors: no output folder, distill's offline ways on a cluster (R10: --on-policy and reinforce --algorithm distill run there).
            var noOut = ClusterCli("cluster", "sim", "--topology", topology, "--", "reinforce", model, "--data", data);
            Check(noOut.Code == ExitCodes.Usage && noOut.Error.Contains("-o DIR", StringComparison.Ordinal), $"reinforce without -o names it: {noOut.Error}");
            var distill = ClusterCli("cluster", "sim", "--topology", topology, "--", "distill", "--teacher", model, "--student", model, "--data", data, "-o", alone);
            Check(distill.Code == ExitCodes.Usage && distill.Error.Contains("--on-policy only", StringComparison.Ordinal), $"cluster sim refuses distill without --on-policy naming why: {distill.Error}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    private static void CliDistillOnPolicy(Device device)
    {
        _ = device;
        string folder = TempFolder();
        string student = WriteChatModel(TinyChatSpec()), teacher = WriteChatModel(TinyTeacherSpec());
        try
        {
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, Enumerable.Range(0, 6).Select(i => new JsonObject { ["question"] = $"What is {i} plus {i}?", ["answer"] = $"{i + i}" }.ToJsonString()));
            string output = Path.Combine(folder, "on-policy");

            // Usage errors.
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--generate", "-o", output).Code == 2, "--on-policy with --generate");
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--mix", "2", "-o", output).Code == 2, "--mix above 1");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
        }
    }
}
