// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Classic;

// Plan 27, phase R6: Idrak.Reinforcement.Plugins.Classic, the classic control environments: registration, Gymnasium's
// trajectories (tests/Idrak.Tests/data/reinforcement, written by tools/reinforcement/make_fixtures.py), R1's environment
// suite, the reset distributions and the bandit's statistics, then PPO (R2's trainer) and DQN (the off-policy trainer)
// solving them within seeded budgets.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClassicControlGroup =
    [
        ("plugins: classic: ClassicControlPlugin is the plug-in \"classic\" of ReinforcementPlugins.All; Register twice registers bandit, cartpole, pendulum and mountain-car once, from Idrak.Reinforcement.Plugins; Unregister leaves EnvironmentKinds as found; options are refused naming the keys", d => { if (d == Device.Cpu) ClassicPluginRegisters(); }),
        ("plugins: classic: CartPole, Pendulum and MountainCar match Gymnasium's trajectories from the same start states and actions (observations and rewards within 1e-5, terminated and truncated exactly; the fixtures' hashes are the manifest's)", d => { if (d == Device.Cpu) ClassicMatchesGymnasium(); }),
        ("plugins: classic: the four kinds pass the environment suite (with options too), start from Gymnasium's reset distributions drawn from the episode's seed, and the bandit pays its arms' means and spreads", d => { if (d == Device.Cpu) ClassicSuitesAndDraws(); }),
        ("plugins: classic: PPO (R2's trainer) solves CartPole: 32 greedy evaluation episodes average at least 490 within 150 iterations, then 100 new greedy episodes average at least 475, the rest cut at Gymnasium's 500 steps", ClassicPpoSolvesCartPole),
        ("plugins: classic: DQN (the off-policy trainer) solves the bandit (its greedy arm the best one, with and without double Q-learning) and CartPole (16 greedy evaluation episodes average at least 490 within 60,000 steps, then 100 new greedy episodes at least 475)", ClassicDqnSolves),
    ];

    internal static void ClassicPluginRegisters()
    {
        var plugin = ClassicControlPlugin.Plugin;
        Check(plugin.Name == "classic" && ReinforcementPlugins.All.Contains(plugin) && plugin.GetType().Namespace == "Idrak.Reinforcement.Plugins.Classic",
            "the plug-in \"classic\" of ReinforcementPlugins.All, in Idrak.Reinforcement.Plugins.Classic");
        string[] kinds = ["bandit", "cartpole", "pendulum", "mountain-car"];
        Check(ClassicControlPlugin.Kinds.Select(k => k.Name).SequenceEqual(kinds), $"the kinds: {string.Join(", ", ClassicControlPlugin.Kinds.Select(k => k.Name))}");
        string State() => $"{string.Join(",", EnvironmentKinds.Names)} | {string.Join(",", IdrakPlugins.Names)}";
        bool before = ClassicControlPlugin.IsRegistered;
        try
        {
            ClassicControlPlugin.Unregister();
            string found = State();
            Check(kinds.All(k => EnvironmentKinds.Find(k) is null), $"unregistered, no classic kind: {found}");
            ThrowsNaming<NotSupportedException>(() => EnvironmentKinds.Get("cartpole"), "an unregistered kind", "not registered", "Idrak.Reinforcement.Plugins");
            ClassicControlPlugin.Register();
            ClassicControlPlugin.Register();
            Check(kinds.All(k => EnvironmentKinds.Names.Count(n => n == k) == 1 && EnvironmentKinds.Origin(k) == "Idrak.Reinforcement.Plugins"), $"the four kinds once, from Idrak.Reinforcement.Plugins: {State()}");
            Check(EnvironmentKinds.Get("cartpole").ObservationType == typeof(float[]) && EnvironmentKinds.Get("pendulum").ActionType == typeof(float[])
                  && EnvironmentKinds.Get("mountain-car").ActionType == typeof(int), "float observations; a box action for the pendulum, choices for the others");
            using (var cart = EnvironmentKinds.Create<float[], int>("cartpole"))
            {
                Check(cart is CartPoleEnvironment { Options.MaxSteps: 500 } && cart.ActionSpace.Equals(Space.Discrete(2)) && cart.ObservationSpace.Size == 4,
                    $"a cartpole of Gymnasium's CartPole-v1: {cart.ObservationSpace}, {cart.ActionSpace}");
            }

            ClassicControlPlugin.Unregister();
            Check(State() == found, $"Unregister leaves the registries as found:\n{found}\nnow:\n{State()}");
        }
        finally
        {
            if (before)
            {
                ClassicControlPlugin.Register();
            }
        }

        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.CartPoleKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["gravity"] = "9" })), "an unknown option",
            "'gravity'", "max-steps", "sutton-barto", "reset-low", "reset-high");
        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.PendulumKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["max-steps"] = "-1" })), "a bad value", "'max-steps'", "-1");
        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.MountainCarKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["reset-low"] = "-0.3", ["reset-high"] = "-0.5" })),
            "bounds the wrong way round", "reset-low", "reset-high");
        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.BanditKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["arms"] = "3", ["means"] = "1,2" })), "means for other arms", "means", "3 arms");
        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.BanditKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["spreads"] = "1,2" })), "spreads for other arms", "spreads");
        using var pendulum = new PendulumEnvironment();
        pendulum.Reset(1);
        ThrowsNaming<ArgumentException>(() => pendulum.Step([float.NaN]), "a NaN torque", "finite torque");
        ThrowsNaming<ArgumentException>(() => new CartPoleEnvironment().ResetTo([0, 0, 0]), "a state of 3", "4 finite values");
    }

    // ---------------------------------------------------------------- Gymnasium's trajectories

    internal static void ClassicMatchesGymnasium()
    {
        // The fixtures are the script's: their hashes are the manifest's.
        var manifest = JsonNode.Parse(File.ReadAllText(TestData("reinforcement/manifest.json")))!.AsObject();
        foreach (var (file, entry) in manifest["files"]!.AsObject())
        {
            string hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(TestData($"reinforcement/{file}"))));
            Check(hash == (string?)entry!["sha256"], $"{file}'s SHA-256 is the manifest's (the fixtures are tools/reinforcement/make_fixtures.py's): {hash}");
        }

        const double Tolerance = 1e-5;
        var reports = new List<string>();
        foreach (string kind in new[] { "cartpole", "pendulum", "mountain-car" })
        {
            var document = JsonNode.Parse(File.ReadAllText(TestData($"reinforcement/{kind}.json")))!.AsObject();
            Check((string?)document["gymnasium"] == "1.2.1", $"{kind}: the fixtures of Gymnasium 1.2.1 (tools/reinforcement/requirements.txt): {document["gymnasium"]}");
            int episodes = 0, steps = 0, exact = 0;
            double worst = 0;
            foreach (var episode in document["episodes"]!.AsArray())
            {
                string name = $"{kind}, {(string?)episode!["name"]}";
                var options = episode["options"]!.AsObject().ToDictionary(p => p.Key, p => (string)p.Value!, StringComparer.Ordinal);
                using var environment = ClassicControlPlugin.Kinds.Single(k => k.Name == kind).Create(new EnvironmentRequest(options));
                double[] start = [.. episode["start"]!.AsArray().Select(v => (double)v!)];
                float[] observation = environment switch
                {
                    CartPoleEnvironment cart => cart.ResetTo(start),
                    PendulumEnvironment pendulum => pendulum.ResetTo(start),
                    MountainCarEnvironment car => car.ResetTo(start),
                    _ => throw new InvalidOperationException(kind),
                };
                var expected = episode["observations"]!.AsArray();
                var actions = episode["actions"]!.AsArray();
                var rewards = episode["rewards"]!.AsArray();
                var terminated = episode["terminated"]!.AsArray();
                var truncated = episode["truncated"]!.AsArray();
                double Compare(float[] actual, JsonNode? reference, string what)
                {
                    var values = reference!.AsArray();
                    Check(values.Count == actual.Length, $"{name}, {what}: {actual.Length} values, {values.Count} expected");
                    double most = 0;
                    for (int i = 0; i < actual.Length; i++)
                    {
                        double x = (double)values[i]!;
                        double difference = Math.Abs(actual[i] - x) / Math.Max(1, Math.Abs(x));
                        Check(difference <= Tolerance, string.Create(CultureInfo.InvariantCulture, $"{name}, {what}, element {i}: {actual[i]:R}, Gymnasium's {x:R}"));
                        most = Math.Max(most, difference);
                    }

                    exact += most == 0 ? 1 : 0;
                    return most;
                }

                worst = Math.Max(worst, Compare(observation, expected[0], "the start's observation"));
                for (int t = 0; t < actions.Count; t++)
                {
                    var step = environment switch
                    {
                        CartPoleEnvironment cart => cart.Step((int)actions[t]!),
                        MountainCarEnvironment car => car.Step((int)actions[t]!),
                        PendulumEnvironment pendulum => pendulum.Step([(float)(double)actions[t]!]),
                        _ => throw new InvalidOperationException(kind),
                    };
                    worst = Math.Max(worst, Compare(step.Observation, expected[t + 1], $"step {t + 1}'s observation"));
                    double reward = (double)rewards[t]!;
                    Check(Math.Abs(step.Reward - reward) <= Tolerance * Math.Max(1, Math.Abs(reward)), string.Create(CultureInfo.InvariantCulture, $"{name}, step {t + 1}'s reward: {step.Reward:R}, Gymnasium's {reward:R}"));
                    Check(step.Terminated == (bool)terminated[t]! && step.Truncated == (bool)truncated[t]!,
                        $"{name}, step {t + 1}: terminated {step.Terminated}, truncated {step.Truncated}; Gymnasium's {terminated[t]}, {truncated[t]}");
                    steps++;
                }

                int played = environment switch { CartPoleEnvironment cart => cart.Steps, PendulumEnvironment pendulum => pendulum.Steps, MountainCarEnvironment car => car.Steps, _ => -1 };
                Check(played == actions.Count, $"{name}: {played} steps counted, {actions.Count} played");
                ThrowsNaming<InvalidOperationException>(() => _ = environment switch
                {
                    CartPoleEnvironment cart => cart.Step(0),
                    MountainCarEnvironment car => car.Step(1),
                    PendulumEnvironment pendulum => pendulum.Step([0f]),
                    _ => throw new InvalidOperationException(kind),
                }, $"{name}: a step after the end", "reset it first");
                episodes++;
            }

            reports.Add(string.Create(CultureInfo.InvariantCulture, $"{kind} {episodes} episodes, {steps} steps, largest difference {worst:G3} ({exact} of {steps + episodes} observations bit for bit)"));
        }

        Console.WriteLine($"    Gymnasium: {string.Join("; ", reports)}");
    }

    // ---------------------------------------------------------------- the suites and the draws

    internal static void ClassicSuitesAndDraws()
    {
        // R1's environment suite, with the defaults and with options.
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.CartPoleKind, (r, _) => r.Next(2), maxSteps: 600).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.CartPoleKind, (r, _) => r.Next(2),
            new Dictionary<string, string> { ["max-steps"] = "20", ["sutton-barto"] = "true", ["reset-low"] = "-0.1", ["reset-high"] = "0.1" }).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], float[]>(ClassicControlPlugin.PendulumKind, (r, _) => [(float)(r.NextDouble() * 4 - 2)], maxSteps: 300).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], float[]>(ClassicControlPlugin.PendulumKind, (r, _) => [(float)(r.NextDouble() * 4 - 2)],
            new Dictionary<string, string> { ["g"] = "9.81", ["max-steps"] = "30", ["reset-angle"] = "0.5", ["reset-speed"] = "0" }).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.MountainCarKind, (r, _) => r.Next(3), maxSteps: 300).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.MountainCarKind, (r, _) => r.Next(3),
            new Dictionary<string, string> { ["goal-velocity"] = "0.01", ["max-steps"] = "50" }).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.BanditKind, (r, _) => r.Next(10)).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<float[], int>(ClassicControlPlugin.BanditKind, (r, _) => r.Next(3),
            new Dictionary<string, string> { ["means"] = "0,1,2", ["spreads"] = "0.5,1,0", ["pulls"] = "5" }).ThrowIfFailed();

        // Gymnasium's start distributions, from the episode's seed.
        const int Resets = 4000;
        using var cart = new CartPoleEnvironment();
        using var pendulum = new PendulumEnvironment();
        using var car = new MountainCarEnvironment();
        var cartStates = Enumerable.Range(0, Resets).Select(s => { cart.Reset(s); return cart.State; }).ToArray();
        var pendulumStates = Enumerable.Range(0, Resets).Select(s => { pendulum.Reset(s); return pendulum.State; }).ToArray();
        var carStates = Enumerable.Range(0, Resets).Select(s => { car.Reset(s); return car.State; }).ToArray();
        void Uniform(IEnumerable<double> values, double low, double high, string what)
        {
            var list = values.ToList();
            double mean = list.Average(), variance = list.Average(v => (v - mean) * (v - mean)), width = high - low;
            Check(list.All(v => v >= low && v < high), $"{what}: every draw in [{low}, {high})");
            Check(Math.Abs(mean - (low + high) / 2) < 0.05 * width && Math.Abs(variance - width * width / 12) < 0.1 * width * width / 12,
                string.Create(CultureInfo.InvariantCulture, $"{what}: uniform (mean {mean:G4}, variance {variance:G4}; expected {(low + high) / 2:G4}, {width * width / 12:G4})"));
        }

        for (int i = 0; i < 4; i++)
        {
            Uniform(cartStates.Select(s => s[i]), -0.05, 0.05, $"cartpole's start element {i}");
        }

        Uniform(pendulumStates.Select(s => s[0]), -Math.PI, Math.PI, "pendulum's start angle");
        Uniform(pendulumStates.Select(s => s[1]), -1, 1, "pendulum's start speed");
        Uniform(carStates.Select(s => s[0]), -0.6, -0.4, "mountain car's start position");
        Check(carStates.All(s => s[1] == 0), "mountain car starts still");
        cart.Reset(7);
        double[] first = cart.State;
        cart.Reset(8);
        cart.Reset(7);
        Check(first.SequenceEqual(cart.State) && !first.SequenceEqual(cartStates[8]), "the same seed starts the same state, another seed another");

        // The bandit: its arms' means and spreads; the testbed's means are the same in every environment of a kind.
        using var bandit = new BanditEnvironment(BanditOptions.FromOptions(new Dictionary<string, string> { ["means"] = "0.2,0.5,1.0,0.8", ["spreads"] = "1,1,0.5,0", ["pulls"] = "100" }));
        Check(bandit.BestArm == 2 && bandit.ActionSpace.Equals(Space.Discrete(4)), $"the best arm: {bandit.BestArm}");
        foreach (var (arm, mean, spread) in new[] { (2, 1.0, 0.5), (0, 0.2, 1.0), (3, 0.8, 0.0) })
        {
            var pulls = new List<double>();
            for (int episode = 0; episode < 60; episode++)
            {
                bandit.Reset(episode);
                for (int p = 0; p < 100; p++)
                {
                    var step = bandit.Step(arm);
                    pulls.Add(step.Reward);
                    Check(step.Terminated == (p == 99) && !step.Truncated, "the last pull of an episode terminates it");
                }
            }

            double average = pulls.Average(), deviation = Math.Sqrt(pulls.Average(v => (v - average) * (v - average)));
            Check(Math.Abs(average - mean) < 0.05 && Math.Abs(deviation - spread) < 0.05,
                string.Create(CultureInfo.InvariantCulture, $"arm {arm} pays mean {average:G4} and spread {deviation:G4} over {pulls.Count} pulls; its {mean} and {spread}"));
        }

        using var testbed = new BanditEnvironment();
        using var again = (BanditEnvironment)ClassicControlPlugin.BanditKind.Create(new EnvironmentRequest(Seed: 5));
        using var other = new BanditEnvironment(new BanditOptions { MeansSeed = 1 });
        Check(testbed.Means.Count == 10 && testbed.Means.SequenceEqual(again.Means) && !testbed.Means.SequenceEqual(other.Means) && testbed.Spreads.All(s => s == 1),
            $"the 10-armed testbed: its means from means-seed (the same in every environment of the kind): {string.Join(", ", testbed.Means.Select(m => m.ToString("0.##", CultureInfo.InvariantCulture)))}");
    }

    // ---------------------------------------------------------------- PPO and DQN solve them

    internal static void ClassicPpoSolvesCartPole(Device device)
    {
        var kind = ClassicControlPlugin.CartPoleKind;
        var (observations, actions) = Spaces(kind);
        var random = new Random(1);
        using var policy = ModulePolicy.Categorical(observations, actions, ModulePolicy.Perceptron(4, 2, [64, 64], device, random), ModulePolicy.Perceptron(4, 1, [64, 64], device, random),
            options: new ModulePolicyOptions { LearningRate = 1e-3f });
        const int Budget = 150;
        var options = new ReinforcementOptions
        {
            Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 16, MiniBatch = 4, Epochs = 4, Iterations = Budget, Seed = 1, NormalizeAdvantages = true,
            EvaluateEvery = 1, EvaluationEpisodes = 32, Environments = 1, ParallelSteps = false,
        };
        long steps = 0;
        var (result, at) = TrainUntil(kind, policy, options, s =>
        {
            steps += s.Steps;
            return s.EvaluationReward >= 490;
        });
        Check(at > 0 && at <= Budget, $"32 greedy episodes average at least 490 within {Budget} iterations (after {result.Iterations}: {result.History[^1].EvaluationReward:0.#})");
        using var collector = new RolloutCollector<float[], int>(kind, environments: 1);
        using var final = collector.CollectAsync(policy, [.. Enumerable.Range(0, 100).Select(k => new EpisodeStart(k, 1_000_000 + k, k))], new ActionSampling { Greedy = true }).AsTask().GetAwaiter().GetResult();
        double mean = final.Trajectories.Average(t => t.Return);
        Check(mean >= 475, $"100 new greedy episodes average {mean:0.#} (Gymnasium's bar for CartPole-v1: 475)");
        // An episode ends terminated before 500 steps or is cut at the 500th (truncated; one that also fails on that step is
        // both in Gymnasium, and the collector keeps "terminated", so nothing is bootstrapped from it).
        Check(final.Trajectories.All(t => t.Steps <= 500 && (t.Terminated || (t.Truncated && t.Steps == 500))) && final.Trajectories.Any(t => t.Steps == 500 && t.Truncated),
            $"episodes end terminated, or truncated at Gymnasium's 500 steps: {string.Join(", ", final.Trajectories.Select(t => $"{t.Steps}{(t.Terminated ? "t" : "")}{(t.Truncated ? "c" : "")}"))}");
        Console.WriteLine($"    ppo: cartpole solved after {at} iterations ({steps} steps); 100 new greedy episodes average {mean:0.#}");
    }

    internal static void ClassicDqnSolves(Device device)
    {
        // The bandit: one pull an episode, a linear Q-network (its bias the arms' values); solved when its greedy arm is the best.
        var reached = new List<string>();
        foreach (bool doubleQ in new[] { false, true })
        {
            var banditOptions = new OffPolicyOptions
            {
                Steps = 3_000, Seed = 3, LearningRate = 0.01f, BatchSize = 32, LearningStarts = 100, TrainEvery = 1, GradientSteps = 1, TargetEvery = 100, EpsilonFraction = 0.5f,
                ReportEvery = 100, DoubleQ = doubleQ, EnvironmentOptions = new Dictionary<string, string> { ["means"] = "0.2,0.5,1.0,0.8", ["spreads"] = "1" },
            };
            OffPolicyTrainer? running = null;
            long solved = 0;
            using var bandit = new OffPolicyTrainer(ClassicControlPlugin.BanditKind, ModulePolicy.Perceptron(1, 4, [], device, new Random(1)), ModulePolicy.Perceptron(1, 4, [], device, new Random(2)), banditOptions)
            {
                OnReport = s => solved = running!.Act([0f]) == 2 ? (solved == 0 ? s.Step : solved) : 0,   // the first report from which the greedy arm stays the best
            };
            running = bandit;
            bandit.Train();
            float[] values = bandit.Values([0f]);
            Check(DqnArgMax(values) == 2 && Math.Abs(values[2] - 1.0) < 0.15 && solved > 0,
                $"double {doubleQ}: the greedy arm is the best one (2, mean 1.0) from step {solved}: Q = {string.Join(", ", values.Select(v => v.ToString("0.##", CultureInfo.InvariantCulture)))}");
            double final = bandit.EvaluateAsync(400, 777).AsTask().GetAwaiter().GetResult();
            Check(final > 0.9, $"double {doubleQ}: 400 greedy pulls pay {final:0.###} (the best arm's mean 1.0)");
            reached.Add(string.Create(CultureInfo.InvariantCulture, $"bandit{(doubleQ ? " (double)" : "")} the best arm from step {solved} of 3000, Q {string.Join("/", values.Select(v => v.ToString("0.##", CultureInfo.InvariantCulture)))}"));
        }

        // CartPole: Stable-Baselines3 zoo's DQN settings for CartPole-v1 on a 64 x 64 network.
        const long Budget = 60_000;
        var cartOptions = new OffPolicyOptions
        {
            Steps = Budget, Seed = 2, LearningRate = 2.3e-3f, BatchSize = 64, ReplayCapacity = 100_000, LearningStarts = 1_000, TrainEvery = 256, GradientSteps = 128, TargetEvery = 10,
            EpsilonFraction = 0.16f, EpsilonEnd = 0.04f, ReportEvery = 1_000, EvaluateEvery = 1_000, EvaluationEpisodes = 16,
        };
        using var stop = new CancellationTokenSource();
        long at = 0;
        using var cartpole = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [64, 64], device, new Random(2)),
            ModulePolicy.Perceptron(4, 2, [64, 64], device, new Random(0)), cartOptions)
        {
            OnReport = s =>
            {
                if (at == 0 && s.EvaluationReward >= 490)
                {
                    at = s.Step;
                    stop.Cancel();
                }
            },
        };
        var result = cartpole.Train(stop.Token);
        Check(at > 0 && at <= Budget, $"16 greedy episodes average at least 490 within {Budget} steps (after {result.Steps}: {result.History[^1].EvaluationReward:0.#})");
        double mean = cartpole.EvaluateAsync(100, 777).AsTask().GetAwaiter().GetResult();
        Check(mean >= 475, $"100 new greedy episodes average {mean:0.#} (Gymnasium's bar for CartPole-v1: 475)");
        Check(cartpole.Replay.Count == at && cartpole.Replay.Bytes > 0, $"every step in the replay buffer: {cartpole.Replay.Count}");
        reached.Add(string.Create(CultureInfo.InvariantCulture, $"cartpole after {at} steps ({result.Updates} updates), 100 new greedy episodes average {mean:0.#}"));
        Console.WriteLine($"    dqn: {string.Join("; ", reached)}");
    }

    internal static int DqnArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }
}
