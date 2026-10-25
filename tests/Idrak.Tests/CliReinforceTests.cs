// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli;
using Idrak.Models;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text;
using Idrak.Text.Abstractions;

// Plan 27, R5: idrak reinforce (the library's trainer over Idrak.Text's language-model policy, in the prompts environment
// and the coding plug-in's) and idrak distill --on-policy (R3's OnPolicyDistiller), run in-process on the tests' tiny
// decoders; no download. R3c: idrak reinforce under idrak cluster sim (an in-process cluster: a thread per rank, no child
// process).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliReinforceGroup =
    [
        ("cli reinforce: --list names every registry's entries (environments, rewards, algorithms and distill, estimators, sandboxes, divergences) in text and JSON; usage errors (no model, no --out, an unregistered environment, algorithm or estimator, episodes that do not fill the groups, a coding run without a suite, nothing to resume, a teacher weight without a teacher, distill without a teacher or outside prompts) exit with 2 before a model loads", d => { if (d == Device.Cpu) CliReinforceListAndUsage(); }),
        ("cli reinforce: a seeded grpo iteration on the prompts environment of a tiny model (the command line over the IDRAK_RL_* variables): its adapter, checkpoint and JSON log line are written, the same seed gives the same returns, and --resume goes on from the checkpoint", CliReinforcePrompts),
        ("cli reinforce: grpo on a coding suite (the coding plug-in's environment, verified by git grep in the process sandbox) with the tiny model: every episode verified and rewarded, the adapter written", CliReinforceCoding),
        ("cli reinforce: cluster sim runs a seeded grpo on 2 ranks of the prompts environment (4 shards from --shards, against one process with IDRAK_RL_SHARDS=4): every rank ends, rank 0 alone prints and writes (the output folder holds what one process's holds: one adapter, its manifest, the checkpoints; the log a line per iteration; the session's SaveOnce writes nothing on rank 1), the first rollout's rewards are one process's and the adapters within R3b's 1e-4, and --resume goes on on the cluster as in one process; distill without --on-policy refused on a cluster with the reason (on-policy distillation runs there since R10: distillation cluster:)", CliReinforceCluster),
        ("cli distill: --on-policy: the student answers the prompts itself and learns the teacher's distributions on its samples (R3's OnPolicyDistiller), with a mix of data answers; the adapter is written; reinforce --algorithm distill runs the same training; its usage errors exit with 2", CliDistillOnPolicy),
    ];

    private static void CliReinforceListAndUsage()
    {
        // --list: each registry's names, as registered (the CLI registers the four plug-in packages).
        var listed = TrainCliJson("reinforce", "--list");
        HashSet<string> Names(string section) => [.. listed[section]!.AsArray().Select(e => (string)e!["name"]!)];
        Check(Names("environments").SetEquals(EnvironmentKinds.Names) && Names("environments").IsSupersetOf(["prompts", "coding"]), $"environments: {listed["environments"]}");
        Check(Names("rewards").SetEquals(RewardFunctions.Names) && Names("rewards").IsSupersetOf(["metric", "tests-passed"]), $"rewards: {listed["rewards"]}");
        Check(Names("algorithms").SetEquals(PolicyLosses.Names.Append("distill")) && Names("algorithms").IsSupersetOf(["grpo", "ppo", "reinforce"]), $"algorithms: {listed["algorithms"]}");
        Check(Names("estimators").SetEquals(AdvantageEstimators.Names) && Names("estimators").Contains("group"), $"estimators: {listed["estimators"]}");
        Check(Names("sandboxes").SetEquals(CommandSandboxes.Names) && Names("sandboxes").IsSupersetOf(["process", "container"]), $"sandboxes: {listed["sandboxes"]}");
        Check(Names("divergences").SetEquals(TeacherDivergences.Names) && Names("divergences").IsSupersetOf(["forward", "reverse", "jsd"]), $"divergences: {listed["divergences"]}");
        var text = TrainCli("reinforce", "--list");
        Check(text.Code == 0 && new[] { "environments (EnvironmentKinds):", "rewards (RewardFunctions):", "algorithms (PolicyLosses):", "estimators (AdvantageEstimators):",
            "sandboxes (CommandSandboxes):", "divergences (TeacherDivergences):" }.All(h => text.Out.Contains(h, StringComparison.Ordinal)), $"--list in text: {text.Out}");
        Check(TrainCli("help", "reinforcement").Out.Contains("idrak reinforce", StringComparison.Ordinal), "help reinforcement");

        // Usage errors, each before a model loads (the model named does not exist).
        string folder = TempFolder();
        try
        {
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllText(data, "{\"question\": \"Q\", \"answer\": \"yes\"}\n");
            string[] model = ["no-such-model-folder", "-o", Path.Combine(folder, "out")];
            void Usage(string what, string expected, params string[] args)
            {
                var (code, output, error) = TrainCli(["reinforce", .. args]);
                Check(code == 2 && error.Contains(expected, StringComparison.Ordinal), $"{what}: exit {code}, {error}{output}");
            }

            Usage("no model", "Give the model", "--env", "prompts", "--data", data, "-o", folder);
            Usage("no --out", "--out DIR", "no-such-model-folder", "--data", data);
            Usage("an unregistered environment", "not registered", [.. model, "--env", "nowhere"]);
            Usage("an unregistered algorithm", "--algorithm nope", [.. model, "--data", data, "--algorithm", "nope"]);
            Usage("an unregistered estimator", "--estimator nope", [.. model, "--data", data, "--estimator", "nope"]);
            Usage("episodes that do not fill the groups", "not a multiple of --group", [.. model, "--data", data, "--episodes", "6", "--group", "4"]);
            Usage("prompts without data", "--data FILE", [.. model, "--env", "prompts"]);
            Usage("coding without a suite", "--suite DIR", [.. model, "--env", "coding"]);
            Usage("an option the environment does not take", "does not take 'suite'", [.. model, "--data", data, "--suite", folder]);
            Usage("an unregistered sandbox", "--sandbox nowhere", [.. model, "--env", "coding", "--suite", folder, "--sandbox", "nowhere"]);
            Usage("nothing to resume", "Nothing to resume", [.. model, "--data", data, "--resume"]);
            Usage("a teacher weight without a teacher", "needs a teacher", [.. model, "--data", data, "--teacher-weight", "1"]);
            Usage("distill without a teacher", "give --teacher", [.. model, "--data", data, "--algorithm", "distill"]);
            Usage("distill outside prompts", "--env prompts", [.. model, "--env", "coding", "--suite", folder, "--algorithm", "distill", "--teacher", "t"]);
            Usage("an unregistered divergence", "--divergence nope", [.. model, "--data", data, "--divergence", "nope"]);
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_ALGORITHM"] = "nope" }, () =>
                Usage("an algorithm from its variable", "IDRAK_RL_ALGORITHM=nope", [.. model, "--data", data]));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_GROUP"] = "x" }, () =>
                Usage("a variable that does not parse", "IDRAK_RL_GROUP=x", [.. model, "--data", data]));
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

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

    private static void CliReinforcePrompts(Device device)
    {
        string folder = TempFolder();
        var (model, data) = ReinforceFixture(folder);
        try
        {
            string first = Path.Combine(folder, "first"), again = Path.Combine(folder, "again"), log = Path.Combine(folder, "log.jsonl");
            string[] run = ["reinforce", model, "--env", "prompts", "--data", data, "--reward", "exact", "--algorithm", "grpo", "--group", "4", "--episodes", "4", "--seed", "3",
                .. TinySession(device, folder)];
            JsonObject one = null!, two = null!;
            // The command line wins over the variables (5 iterations, groups of 2); the variables it does not set apply.
            WithEnvironment(new Dictionary<string, string?>
            {
                ["IDRAK_RL_ITERATIONS"] = "5", ["IDRAK_RL_GROUP"] = "2", ["IDRAK_RL_LOG"] = log, ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off",
            }, () =>
            {
                one = TrainCliJson([.. run, "--iterations", "1", "-o", first]);
                two = TrainCliJson([.. run, "--iterations", "1", "-o", again]);
            });
            var history = one["history"]!.AsArray();
            Check((int?)one["exitCode"] == 0 && (int?)one["iterations"] == 1 && history.Count == 1 && (int?)history[0]!["episodes"] == 4 && (string?)one["environment"] == "prompts",
                $"one grpo iteration: {one}");
            Check(File.Exists(Path.Combine(first, "adapter_model.safetensors")) && File.Exists(Path.Combine(first, "idrak-tuning.json")), "the adapter and its manifest are written");
            Check(Directory.Exists(Path.Combine(first, "checkpoints", "iteration-000001")) && (string?)one["checkpoint"] == Path.Combine(first, "checkpoints", "iteration-000001"),
                $"the checkpoint is written into the output: {one["checkpoint"]}");
            Check(File.ReadAllLines(log).Length == 2 && JsonNode.Parse(File.ReadAllLines(log)[0]) is JsonObject, "IDRAK_RL_LOG: a JSON line per iteration (two runs of one)");
            Check(history[0]!["returns"]!.ToJsonString() == two["history"]!.AsArray()[0]!["returns"]!.ToJsonString()
                  && history[0]!["loss"]!.ToJsonString() == two["history"]!.AsArray()[0]!["loss"]!.ToJsonString(), $"the same seed gives the same returns and loss: {history} | {two["history"]}");
            Check(one["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("iteration 1/1: reward", StringComparison.Ordinal)), $"a line per iteration: {one["output"]}");

            // --resume: the second iteration from the first's checkpoint.
            var resumed = TrainCliJson([.. run, "--iterations", "2", "-o", first, "--resume"]);
            var later = resumed["history"]!.AsArray();
            Check((int?)resumed["iterations"] == 2 && later.Count == 1 && (int?)later[0]!["iteration"] == 2
                  && Directory.Exists(Path.Combine(first, "checkpoints", "iteration-000002")), $"--resume goes on at iteration 2: {resumed}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    private static void CliReinforceCoding(Device device)
    {
        string folder = TempFolder();
        string model = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            string suite = WriteQuickSuite(folder), output = Path.Combine(folder, "coding");
            var result = TrainCliJson(["reinforce", model, "--env", "coding", "--suite", suite, "--sandbox", "process", "--rounds", "2", "--env-option", "system=Fix it.",
                "--env-option", $"work={Path.Combine(folder, "work")}", "--algorithm", "grpo", "--group", "2", "--episodes", "2", "--iterations", "1", "--environments", "1", "--seed", "5",
                "-o", output, .. TinySession(device, folder)]);
            var history = result["history"]!.AsArray();
            var returns = history.Count == 1 ? history[0]!["returns"]!.AsArray().Select(r => (float)r!).ToList() : [];
            Check((int?)result["exitCode"] == 0 && (string?)result["environment"] == CodingEnvironmentKind.KindName && returns.Count == 2 && returns.All(r => r is 0f or 1f),
                $"grpo on the coding suite: {result}");
            Check(File.Exists(Path.Combine(output, "adapter_model.safetensors")) && Directory.Exists(Path.Combine(output, "checkpoints", "iteration-000001")), "the adapter and the checkpoint");
            Check(!Directory.Exists(Path.Combine(folder, "work")) || !Directory.EnumerateDirectories(Path.Combine(folder, "work")).Any(),
                "the episodes' workspaces are deleted when the run ends");
        }
        finally
        {
            CodingWorkspace.Delete(folder);
            Directory.Delete(model, true);
        }
    }

    private static void CliReinforceCluster(Device device)
    {
        string folder = TempFolder();
        var (model, data) = ReinforceFixture(folder);
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(device, 2).ToJson());
            string alone = Path.Combine(folder, "alone"), ranks = Path.Combine(folder, "ranks");
            string aloneLog = Path.Combine(folder, "alone.jsonl"), ranksLog = Path.Combine(folder, "ranks.jsonl");
            // A reward with a spread (1 less the character error rate against the reference), so the updates move the adapters.
            string[] Reinforce(int iterations, string output, params string[] more) => ["reinforce", model, "--env", "prompts", "--data", data, "--reward", "cer",
                "--algorithm", "grpo", "--group", "4", "--episodes", "8", "--iterations", iterations.ToString(CultureInfo.InvariantCulture), "--seed", "3", "-o", output,
                .. TinySession(device, folder), .. more];
            Dictionary<string, string?> Variables(string log, string? shards) => new()
            {
                ["IDRAK_RL_LOG"] = log, ["IDRAK_RL_SHARDS"] = shards, ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off",
            };

            (int Code, string Out, string Err) one = (0, "", ""), two = (0, "", "");
            WithEnvironment(Variables(aloneLog, "4"), () => one = TrainCli(Reinforce(2, alone)));
            WithEnvironment(Variables(ranksLog, null), () => two = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(2, ranks, "--shards", "4")]));
            Check(one.Code == 0, $"one process, 4 shards: exit {one.Code}\n{one.Out}\n{one.Err}");
            Check(two.Code == 0 && two.Out.Contains("network time on the virtual clock: rank 0", StringComparison.Ordinal) && two.Out.Contains("rank 1", StringComparison.Ordinal),
                $"cluster sim runs reinforce to the end on both ranks: exit {two.Code}\n{two.Out}\n{two.Err}");
            int Count(string text, string part) => text.Split('\n').Count(l => l.StartsWith(part, StringComparison.Ordinal));
            Check(Count(two.Out, "iteration 1/2: reward") == 1 && Count(two.Out, "iteration 2/2: reward") == 1 && Count(two.Out, "trained 2 iterations") == 1
                  && two.Out.Contains("adapter written to", StringComparison.Ordinal), $"rank 0 alone prints its lines (once each): {two.Out}");

            // The output folder: what one process writes, once (no event log, nothing half-written).
            static string[] Entries(string path) => [.. Directory.EnumerateFileSystemEntries(path).Select(e => Path.GetFileName(e)).Order(StringComparer.Ordinal)];
            Check(Entries(ranks).SequenceEqual(Entries(alone)) && Entries(ranks).Contains("adapter_model.safetensors") && Entries(ranks).Contains("idrak-tuning.json")
                  && Entries(ranks).Contains("checkpoints") && !Entries(ranks).Contains("cluster-events.idrklog"),
                $"the cluster's output holds what one process's holds: {string.Join(", ", Entries(ranks))} | {string.Join(", ", Entries(alone))}");
            Check(Entries(Path.Combine(ranks, "checkpoints")).SequenceEqual(["iteration-000002"]), $"one checkpoint, moved into place: {string.Join(", ", Entries(Path.Combine(ranks, "checkpoints")))}");

            // The log: a line per iteration (rank 0's alone); the first rollout is one process's (the same weights, 4 shards).
            JsonObject[] Lines(string log) => [.. File.ReadAllLines(log).Select(l => (JsonObject)JsonNode.Parse(l)!)];
            var aloneLines = Lines(aloneLog);
            var ranksLines = Lines(ranksLog);
            Check(aloneLines.Length == 2 && ranksLines.Length == 2 && ranksLines.Select(l => (int)l["iteration"]!).SequenceEqual([1, 2]), $"the log: a line per iteration: {File.ReadAllText(ranksLog)}");
            string Rewards(JsonObject line) => string.Join(" ", new[] { "episodes", "reward_mean", "reward_spread", "reward_min", "reward_max" }.Select(k => line[k]!.ToJsonString()));
            Check(Rewards(ranksLines[0]) == Rewards(aloneLines[0]), $"the first rollout's rewards are one process's: {Rewards(ranksLines[0])} | {Rewards(aloneLines[0])}");
            Check((double)aloneLines[0]["reward_spread"]! > 0, $"the rewards differ, so the updates move the adapters: {aloneLines[0]}");

            // The adapters: within R3b's tolerance of one process's (a step's gradients are summed per rank, then across the ranks).
            static (string[] Names, float[] Values) Adapter(string path)
            {
                using var reader = SafeTensorsReader.Open(Path.Combine(path, "adapter_model.safetensors"));
                string[] names = [.. reader.Tensors.Keys.Order(StringComparer.Ordinal)];
                return (names, [.. names.SelectMany(n => reader.Read(n))]);
            }

            void SameAdapter(string what)
            {
                var (aloneNames, aloneValues) = Adapter(alone);
                var (ranksNames, ranksValues) = Adapter(ranks);
                Check(ranksNames.SequenceEqual(aloneNames) && aloneNames.Length > 0, $"{what}: the same tensors: {string.Join(", ", ranksNames)}");
                if (Gap(aloneValues, ranksValues) > 1e-4f)
                {
                    // Where they part: each tensor's gap, and both runs' lines per iteration (a rollout that went another way
                    // shows in its rewards, a gradient that differs only in the adapters).
                    var gaps = new List<string>();
                    using (var a = SafeTensorsReader.Open(Path.Combine(alone, "adapter_model.safetensors")))
                    using (var b = SafeTensorsReader.Open(Path.Combine(ranks, "adapter_model.safetensors")))
                    {
                        foreach (string name in aloneNames)
                        {
                            float[] x = a.Read(name), y = b.Read(name);
                            int differ = x.Zip(y).Count(p => MathF.Abs(p.First - p.Second) > 1e-4f);
                            if (differ > 0)
                            {
                                gaps.Add($"{name}: {differ} of {x.Length} differ, by up to {Gap(x, y):E2}");
                            }
                        }
                    }

                    AssertClose(aloneValues, ranksValues, 1e-4f, $"{what}: the 2-rank adapter against one process's ({string.Join("; ", gaps)})\n"
                        + $"one process's log:\n{File.ReadAllText(aloneLog).TrimEnd()}\nthe 2 ranks' log:\n{File.ReadAllText(ranksLog).TrimEnd()}");
                }

                Console.WriteLine($"    {what}: the adapter's largest gap to one process {Gap(aloneValues, ranksValues):E2}");
            }

            SameAdapter("after 2 iterations");

            // --resume on the cluster: iteration 3 from the checkpoint, as one process goes on.
            WithEnvironment(Variables(aloneLog, "4"), () => one = TrainCli([.. Reinforce(3, alone), "--resume"]));
            WithEnvironment(Variables(ranksLog, null), () => two = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(3, ranks, "--shards", "4", "--resume")]));
            Check(one.Code == 0 && two.Code == 0 && Count(two.Out, "iteration 3/3: reward") == 1 && !two.Out.Contains("iteration 1/3", StringComparison.Ordinal),
                $"--resume goes on at iteration 3 on the cluster: exit {two.Code}\n{two.Out}\n{two.Err}");
            ranksLines = Lines(ranksLog);
            Check(ranksLines.Length == 3 && (int)ranksLines[2]["iteration"]! == 3 && Entries(Path.Combine(ranks, "checkpoints")).SequenceEqual(["iteration-000002", "iteration-000003"]),
                $"the resumed run's line and checkpoint: {File.ReadAllText(ranksLog)}");
            SameAdapter("resumed to 3 iterations");

            // The session's collective save: every rank takes part, rank 0 alone writes (each rank names a folder of its own here).
            string[] targets = [Path.Combine(folder, "save-0"), Path.Combine(folder, "save-1")];
            var wrote = DpRun(DpTopology(device, 2), DpOptions(), rank =>
            {
                using var loaded = PretrainedModel.Load(model, new PretrainedOptions { Device = rank.Device });
                using var session = new FineTuningSession(loaded, SessionOptions() with { Cluster = rank.Collectives, ClusterOptions = DpOptions() }, totalSteps: 1);
                return session.SaveOnce(targets[rank.Rank]);
            });
            Check(wrote.SequenceEqual([true, false]) && File.Exists(Path.Combine(targets[0], "adapter_model.safetensors")) && !Directory.Exists(targets[1]),
                $"SaveOnce: rank 0 wrote, rank 1 wrote nothing: {string.Join(", ", wrote)}");

            // Usage errors: no output folder, distill's offline ways on a cluster (R10: --on-policy and reinforce --algorithm distill run there), a bad split.
            var noOut = ClusterCli("cluster", "sim", "--topology", topology, "--", "reinforce", model, "--data", data);
            Check(noOut.Code == ExitCodes.Usage && noOut.Error.Contains("-o DIR", StringComparison.Ordinal), $"reinforce without -o names it: {noOut.Error}");
            var distill = ClusterCli("cluster", "sim", "--topology", topology, "--", "distill", "--teacher", model, "--student", model, "--data", data, "-o", alone);
            Check(distill.Code == ExitCodes.Usage && distill.Error.Contains("--on-policy only", StringComparison.Ordinal), $"cluster sim refuses distill without --on-policy naming why: {distill.Error}");
            var spread = TrainCli([.. Reinforce(1, Path.Combine(folder, "s"), "--spread", "diagonal")]);
            Check(spread.Code == 2 && spread.Err.Contains("--spread diagonal", StringComparison.Ordinal), $"a bad --spread is named: {spread.Err}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    private static void CliDistillOnPolicy(Device device)
    {
        string folder = TempFolder();
        string student = WriteChatModel(TinyChatSpec()), teacher = WriteChatModel(TinyTeacherSpec());
        try
        {
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, Enumerable.Range(0, 6).Select(i => new JsonObject { ["question"] = $"What is {i} plus {i}?", ["answer"] = $"{i + i}" }.ToJsonString()));
            string[] tune = ["--rank", "4", "--lora-alpha", "8", "--max-length", "64", "--batch-tokens", "256", "-d", device.ToString(), "--cache", folder];

            // distill --on-policy: two steps of four prompts, a quarter of them from the data's answers.
            string output = Path.Combine(folder, "on-policy");
            var distilled = TrainCliJson(["distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--mix", "0.25", "--steps", "2", "--batch-size", "4",
                "--max-new", "4", "--divergence", "jsd", "--beta", "0.3", "-o", output, .. tune]);
            var losses = distilled["losses"]!.AsArray().Select(l => (float)l!).ToList();
            Check((string?)distilled["mode"] == "on-policy" && (int?)distilled["exitCode"] == 0 && losses.Count == 2 && losses.All(float.IsFinite)
                  && (string?)distilled["divergence"] == "jsd" && (double?)distilled["mix"] == 0.25, $"distill --on-policy: {distilled}");
            Check(File.Exists(Path.Combine(output, "adapter_model.safetensors")) && File.Exists(Path.Combine(output, "idrak-tuning.json")), "the on-policy adapter is written");
            Check(distilled["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("on-policy distillation: 2 steps of 4 prompts", StringComparison.Ordinal)), $"distill says how: {distilled}");

            // reinforce --algorithm distill: the same training, its steps the iterations and its prompts the episodes.
            var alone = TrainCliJson(["reinforce", student, "--env", "prompts", "--data", data, "--algorithm", "distill", "--teacher", teacher, "--iterations", "2", "--episodes", "4",
                "--max-new", "4", "-o", Path.Combine(folder, "alone"), "--rank", "4", "--alpha", "8", "--max-length", "64", "--batch-tokens", "256", "-d", device.ToString(), "--cache", folder]);
            Check((int?)alone["exitCode"] == 0 && alone["losses"]!.AsArray().Count == 2 && File.Exists(Path.Combine(folder, "alone", "adapter_model.safetensors")),
                $"reinforce --algorithm distill: {alone}");

            // Usage errors.
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--generate", "-o", output).Code == 2, "--on-policy with --generate");
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--mix", "2", "-o", output).Code == 2, "--mix above 1");
            Check(TrainCli("distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--divergence", "nope", "-o", output).Err
                .Contains("--divergence nope", StringComparison.Ordinal), "an unregistered divergence is named");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
        }
    }
}
