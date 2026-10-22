// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text.Abstractions;

// Plan 27, R5: idrak reinforce (the library's trainer over Idrak.Text's language-model policy, in the prompts environment
// and the coding plug-in's) and idrak distill --on-policy (R3's OnPolicyDistiller), run in-process on the tests' tiny
// decoders; no download.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliReinforceGroup =
    [
        ("cli reinforce: --list names every registry's entries (environments, rewards, algorithms and distill, estimators, sandboxes, divergences) in text and JSON; usage errors (no model, no --out, an unregistered environment, algorithm or estimator, episodes that do not fill the groups, a coding run without a suite, nothing to resume, a teacher weight without a teacher, distill without a teacher or outside prompts) exit with 2 before a model loads", d => { if (d == Device.Cpu) CliReinforceListAndUsage(); }),
        ("cli reinforce: a seeded grpo iteration on the prompts environment of a tiny model (the command line over the IDRAK_RL_* variables): its adapter, checkpoint and JSON log line are written, the same seed gives the same returns, and --resume goes on from the checkpoint", CliReinforcePrompts),
        ("cli reinforce: grpo on a coding suite (the coding plug-in's environment, verified by git grep in the process sandbox) with the tiny model: every episode verified and rewarded, the adapter written", CliReinforceCoding),
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
