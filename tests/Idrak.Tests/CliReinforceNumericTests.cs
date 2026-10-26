// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Abstractions;

// Plan 27, R9: idrak reinforce in an environment of numbers (the classic plug-in's kinds), with no model: PPO on a module
// policy built from options (R2's trainer) and DQN on a Q-network built the same way (R6's off-policy trainer), run in-process.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CliReinforceNumericGroup =
    [
        ("cli reinforce: an environment of numbers: --list names the classic kinds and dqn (off-policy, QLosses); the help shows the path; usage errors exit with 2 before anything trains (an --env-option key the kind does not take, also with IDRAK_RL_ENV; dqn on the pendulum naming its box; a model given; a language model's option; the other trainer's options; an algorithm in neither registry; nothing to resume; a bad --hidden, --activation or IDRAK_RL_HIDDEN)", d => { if (d == Device.Cpu) CliReinforceNumericUsage(); }),
        ("cli reinforce: a seeded ppo run on cartpole (no model; a categorical policy with a value head built from --hidden over IDRAK_RL_HIDDEN, the command line over IDRAK_RL_ITERATIONS) writes the policy, idrak-reinforce.json, its checkpoints and a log line per iteration; --resume from the iteration-2 checkpoint continues bit for bit (iterations 3 and 4's returns and losses, the policy's files)", CliReinforceNumericPpo),
        ("cli reinforce: a seeded dqn run on the bandit (a linear Q-network from --hidden none, IDRAK_RL_DQN_REPLAY underneath, --steps over IDRAK_RL_DQN_STEPS) finds the best arm and writes network.ikw, idrak-reinforce.json, its checkpoints and the log; --resume from the step-1500 checkpoint continues bit for bit (the later reports, the network, the replay buffer)", CliReinforceNumericDqn),
    ];

    private static void CliReinforceNumericUsage()
    {
        // --list: the classic kinds among the environments, dqn among the off-policy algorithms.
        var listed = TrainCliJson("reinforce", "--list");
        HashSet<string> Names(string section) => [.. listed[section]!.AsArray().Select(e => (string)e!["name"]!)];
        Check(Names("environments").IsSupersetOf(["bandit", "cartpole", "pendulum", "mountain-car"]), $"the classic kinds are listed: {listed["environments"]}");
        Check(Names("off-policy").SetEquals(QLosses.Names) && Names("off-policy").Contains("dqn"), $"dqn is listed: {listed["off-policy"]}");
        var text = TrainCli("reinforce", "--list");
        Check(text.Code == 0 && text.Out.Contains("off-policy (QLosses):", StringComparison.Ordinal) && text.Out.Contains("  dqn ", StringComparison.Ordinal)
              && text.Out.Contains("  cartpole ", StringComparison.Ordinal), $"--list in text: {text.Out}");
        var help = TrainCli("help", "reinforce").Out;
        Check(help.Contains("idrak reinforce --env cartpole --algorithm dqn --steps 50000 -o runs/cartpole-dqn", StringComparison.Ordinal)
              && help.Contains("--hidden W,W", StringComparison.Ordinal) && help.Contains("IDRAK_RL_DQN_STEPS", StringComparison.Ordinal), "help reinforce shows the numeric path");
        Check(TrainCli("help", "reinforcement").Out.Contains("idrak reinforce --env cartpole --algorithm ppo --iterations 50 -o runs/cartpole", StringComparison.Ordinal),
            "help reinforcement shows the numeric path");

        string folder = TempFolder();
        try
        {
            string output = Path.Combine(folder, "out");
            void Usage(string what, string[] expected, params string[] args)
            {
                var (code, stdout, error) = TrainCli(["reinforce", .. args]);
                Check(code == 2 && expected.All(e => error.Contains(e, StringComparison.Ordinal)), $"{what}: exit {code}, {error}{stdout}");
                Check(!Directory.Exists(output), $"{what}: nothing was written before the refusal");
            }

            Usage("an option the kind does not take", ["cartpole environment does not take 'nope'", "max-steps"], "--env", "cartpole", "--env-option", "nope=1", "-o", output);
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_ENV"] = "cartpole" }, () =>
                Usage("the kind from IDRAK_RL_ENV", ["cartpole environment does not take 'nope'"], "--env-option", "nope=1", "-o", output));
            Usage("dqn on the pendulum", ["--algorithm dqn", "pendulum", "box(", "on-policy"], "--env", "pendulum", "--algorithm", "dqn", "-o", output);
            Usage("a model given", ["no model is loaded", "'Qwen/Qwen3-0.6B'"], "Qwen/Qwen3-0.6B", "--env", "cartpole", "-o", output);
            Usage("a language model's option", ["--data", "--max-new", "not a language model"], "--env", "cartpole", "--data", "x.jsonl", "--max-new", "8", "-o", output);
            Usage("an on-policy option for dqn", ["--iterations", "off-policy"], "--env", "cartpole", "--algorithm", "dqn", "--iterations", "3", "-o", output);
            Usage("an off-policy option for ppo", ["--steps", "on-policy"], "--env", "cartpole", "--algorithm", "ppo", "--steps", "10", "-o", output);
            Usage("an algorithm in neither registry", ["--algorithm nope", "PolicyLosses", "QLosses", "dqn"], "--env", "cartpole", "--algorithm", "nope", "-o", output);
            Usage("distill outside a language model", ["--algorithm distill", "a language model's"], "--env", "cartpole", "--algorithm", "distill", "-o", output);
            Usage("no output", ["--out DIR"], "--env", "cartpole");
            Usage("nothing to resume (ppo)", ["Nothing to resume"], "--env", "cartpole", "--resume", "-o", output);
            Usage("nothing to resume (dqn)", ["Nothing to resume"], "--env", "bandit", "--algorithm", "dqn", "--resume", "-o", output);
            Usage("a bad --hidden", ["--hidden", "64,0"], "--env", "cartpole", "--hidden", "64,0", "-o", output);
            Usage("a bad --activation", ["--activation swish", "tanh, relu, gelu, sigmoid"], "--env", "cartpole", "--activation", "swish", "-o", output);
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_HIDDEN"] = "wide" }, () =>
                Usage("a bad IDRAK_RL_HIDDEN", ["IDRAK_RL_HIDDEN=wide"], "--env", "cartpole", "-o", output));
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_DQN_BATCH"] = "zero" }, () =>
                Usage("a bad IDRAK_RL_DQN_BATCH", ["IDRAK_RL_DQN_BATCH=zero"], "--env", "bandit", "--algorithm", "dqn", "-o", output));
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CopyFolder(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (string file in Directory.EnumerateFiles(from))
        {
            File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
        }

        foreach (string folder in Directory.EnumerateDirectories(from))
        {
            CopyFolder(folder, Path.Combine(to, Path.GetFileName(folder)));
        }
    }

    private static bool SameBytes(string a, string b) => File.Exists(a) && File.Exists(b) && File.ReadAllBytes(a).AsSpan().SequenceEqual(File.ReadAllBytes(b));

    private static void CliReinforceNumericPpo(Device device)
    {
        string folder = TempFolder();
        try
        {
            string first = Path.Combine(folder, "first"), resumed = Path.Combine(folder, "resumed"), log = Path.Combine(folder, "ppo.jsonl");
            string[] Ppo(string output, params string[] more) => ["reinforce", "--env", "cartpole", "--algorithm", "ppo", "--iterations", "4", "--episodes", "8",
                "--minibatch", "4", "--hidden", "16,16", "--lr", "0.003", "--save-every", "2", "--seed", "7", "-d", device.ToString(), "-o", output, .. more];
            var variables = new Dictionary<string, string?>
            {
                ["IDRAK_RL_ITERATIONS"] = "9", ["IDRAK_RL_HIDDEN"] = "8", ["IDRAK_RL_LOG"] = log, ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off",
            };
            JsonObject one = null!;
            WithEnvironment(variables, () => one = TrainCliJson(Ppo(first)));
            var history = one["history"]!.AsArray();
            Check((int?)one["exitCode"] == 0 && (int?)one["iterations"] == 4 && history.Count == 4 && (string?)one["environment"] == "cartpole" && (string?)one["algorithm"] == "ppo"
                  && (string?)one["actions"] == "discrete(2)" && history.All(h => (int)h!["episodes"]! == 8), $"four ppo iterations on cartpole: {one}");
            Check(one["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("iteration 4/4: reward", StringComparison.Ordinal))
                  && one["output"]!.AsArray().Any(l => ((string?)l)!.Contains("a categorical policy of 16, 16 tanh with a value head", StringComparison.Ordinal)), $"the lines: {one["output"]}");
            foreach (string file in new[] { "actor.ikw", "critic.ikw", "optimizer.safetensors", "policy.json", "idrak-reinforce.json" })
            {
                Check(File.Exists(Path.Combine(first, file)), $"{file} is written into --out");
            }

            var settings = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(first, "idrak-reinforce.json")))!;
            Check((string?)settings["policy"] == "categorical" && settings["hidden"]!.ToJsonString() == "[16,16]" && (bool?)settings["valueHead"] == true
                  && (int?)settings["iterations"] == 4 && (int?)settings["seed"] == 7, $"the settings file: {settings}");
            Check(Directory.Exists(Path.Combine(first, "checkpoints", "iteration-000002")) && Directory.Exists(Path.Combine(first, "checkpoints", "iteration-000004")),
                "a checkpoint every 2 iterations");
            Check(File.ReadAllLines(log).Length == 4, "IDRAK_RL_LOG: a JSON line per iteration");

            // --resume from the iteration-2 checkpoint (in another folder): iterations 3 and 4 as the run without a stop made them.
            CopyFolder(Path.Combine(first, "checkpoints", "iteration-000002"), Path.Combine(resumed, "checkpoints", "iteration-000002"));
            JsonObject two = null!;
            WithEnvironment(variables, () => two = TrainCliJson(Ppo(resumed, "--resume")));
            var later = two["history"]!.AsArray();
            Check((int?)two["iterations"] == 4 && later.Count == 2 && (int?)later[0]!["iteration"] == 3, $"--resume goes on at iteration 3: {two}");
            for (int i = 0; i < 2; i++)
            {
                Check(later[i]!["returns"]!.ToJsonString() == history[i + 2]!["returns"]!.ToJsonString() && later[i]!["loss"]!.ToJsonString() == history[i + 2]!["loss"]!.ToJsonString(),
                    $"iteration {i + 3}: the same returns and loss bit for bit: {later[i]} | {history[i + 2]}");
            }

            foreach (string file in new[] { "actor.ikw", "critic.ikw", "optimizer.safetensors" })
            {
                Check(SameBytes(Path.Combine(first, file), Path.Combine(resumed, file)), $"the resumed {file} is the uninterrupted run's byte for byte");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliReinforceNumericDqn(Device device)
    {
        string folder = TempFolder();
        try
        {
            string first = Path.Combine(folder, "first"), resumed = Path.Combine(folder, "resumed"), log = Path.Combine(folder, "dqn.jsonl");
            string[] Dqn(string output, params string[] more) => ["reinforce", "--env", "bandit", "--env-option", "means=0.2,0.5,1.0,0.8", "--env-option", "spreads=1",
                "--algorithm", "dqn", "--steps", "3000", "--hidden", "none", "--lr", "0.01", "--batch", "32", "--learning-starts", "100", "--train-every", "1",
                "--target-every", "100", "--epsilon-fraction", "0.5", "--report-every", "500", "--save-every", "1500", "--evaluate-every", "3000",
                "--evaluation-episodes", "400", "--seed", "3", "-d", device.ToString(), "-o", output, .. more];
            var variables = new Dictionary<string, string?> { ["IDRAK_RL_DQN_STEPS"] = "10", ["IDRAK_RL_DQN_REPLAY"] = "2000", ["IDRAK_RL_LOG"] = log };
            JsonObject one = null!;
            WithEnvironment(variables, () => one = TrainCliJson(Dqn(first)));
            var history = one["history"]!.AsArray();
            Check((int?)one["exitCode"] == 0 && (long?)one["steps"] == 3000 && history.Count == 6 && (string?)one["algorithm"] == "dqn" && (string?)one["environment"] == "bandit",
                $"3000 dqn steps on the bandit, a report every 500: {one}");
            Check(one["output"]!.AsArray().Any(l => ((string?)l)!.Contains("a Q-network of one linear layer, 3000 steps, 1 environment side by side, a replay of 2000 transitions", StringComparison.Ordinal))
                  && one["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("step 3000/3000: return", StringComparison.Ordinal)), $"the lines: {one["output"]}");
            double evaluation = (double)history[^1]!["evaluation"]!;
            Check(evaluation > 0.9, $"400 greedy pulls pay {evaluation:0.###}: the best arm (mean 1.0)");
            Console.WriteLine($"    dqn on the bandit through the command: 400 greedy pulls pay {evaluation:0.###}");
            var settings = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(first, "idrak-reinforce.json")))!;
            Check(File.Exists(Path.Combine(first, "network.ikw")) && (string?)settings["policy"] == "q-network" && settings["hidden"]!.ToJsonString() == "[]"
                  && (long?)settings["steps"] == 3000 && (string?)settings["environmentOptions"]!["means"] == "0.2,0.5,1.0,0.8", $"the network and its settings: {settings}");
            string[] checkpoints = [.. Directory.EnumerateDirectories(Path.Combine(first, "checkpoints")).Select(d => Path.GetFileName(d)).Order(StringComparer.Ordinal)];
            Check(checkpoints.SequenceEqual(["step-000000001500", "step-000000003000"]), $"a checkpoint every 1500 steps: {string.Join(", ", checkpoints)}");
            Check(File.ReadAllLines(log).Length == 6, "IDRAK_RL_LOG: a JSON line per report");

            // --resume from the step-1500 checkpoint (in another folder): the later reports as the run without a stop made them.
            CopyFolder(Path.Combine(first, "checkpoints", "step-000000001500"), Path.Combine(resumed, "checkpoints", "step-000000001500"));
            JsonObject two = null!;
            WithEnvironment(variables, () => two = TrainCliJson(Dqn(resumed, "--resume")));
            var later = two["history"]!.AsArray();
            Check((long?)two["steps"] == 3000 && later.Count == 3 && (long?)later[0]!["step"] == 2000, $"--resume goes on at step 1500: {two}");
            for (int i = 0; i < 3; i++)
            {
                Check(later[i]!.ToJsonString() == history[i + 3]!.ToJsonString(), $"the report at step {(i + 4) * 500}: the same bit for bit: {later[i]} | {history[i + 3]}");
            }

            Check(SameBytes(Path.Combine(first, "network.ikw"), Path.Combine(resumed, "network.ikw")), "the resumed network is the uninterrupted run's byte for byte");
            foreach (string file in new[] { "network.ikw", "target.ikw", "optimizer.safetensors", "replay.bin" })
            {
                Check(SameBytes(Path.Combine(first, "checkpoints", "step-000000003000", file), Path.Combine(resumed, "checkpoints", "step-000000003000", file)),
                    $"the last checkpoint's {file} byte for byte");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
