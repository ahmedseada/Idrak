// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Plugins.Classic;

// Plan 27, phase R6: Idrak.Reinforcement.Plugins.Classic, the classic control environments: their kinds, Gymnasium's
// trajectories (tests/Idrak.Tests/data/reinforcement, written by tools/reinforcement/make_fixtures.py), the reset
// distributions and the bandit's statistics.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClassicControlGroup =
    [
        ("plugins: classic: ClassicControlPlugin's kinds are bandit, cartpole, pendulum and mountain-car (float observations, a box action for the pendulum); a kind refuses options by key, naming them; a NaN torque and a short state are refused", d => { CpuOnly(d); ClassicPluginRegisters(); }),
        ("plugins: classic: CartPole, Pendulum and MountainCar match Gymnasium's trajectories made with their default options from the same start states and actions (observations and rewards within 1e-5, terminated and truncated exactly; the fixtures' hashes are the manifest's)", d => { CpuOnly(d); ClassicMatchesGymnasium(); }),
        ("plugins: classic: the environments start from Gymnasium's reset distributions drawn from the episode's seed, and the bandit pays its arms' means and spreads", d => { CpuOnly(d); ClassicSuitesAndDraws(); }),
    ];

    internal static void ClassicPluginRegisters()
    {
        string[] kinds = ["bandit", "cartpole", "pendulum", "mountain-car"];
        Check(ClassicControlPlugin.Kinds.Select(k => k.Name).SequenceEqual(kinds), $"the kinds: {string.Join(", ", ClassicControlPlugin.Kinds.Select(k => k.Name))}");
        Check(ClassicControlPlugin.CartPoleKind.ObservationType == typeof(float[]) && ClassicControlPlugin.PendulumKind.ActionType == typeof(float[])
              && ClassicControlPlugin.MountainCarKind.ActionType == typeof(int), "float observations; a box action for the pendulum, choices for the others");
        using (var cart = (IEnvironment<float[], int>)ClassicControlPlugin.CartPoleKind.Create(new EnvironmentRequest()))
        {
            Check(cart is CartPoleEnvironment { Options.MaxSteps: 500 } && cart.ActionSpace.Equals(Space.Discrete(2)) && cart.ObservationSpace.Size == 4,
                $"a cartpole of Gymnasium's CartPole-v1: {cart.ObservationSpace}, {cart.ActionSpace}");
        }

        ThrowsNaming<ArgumentException>(() => ClassicControlPlugin.CartPoleKind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["gravity"] = "9" })), "an option by key",
            "'gravity'");
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
                if (options.Count > 0)
                {
                    continue;                                                    // an environment's options by key are not on the new design yet
                }

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
        using var bandit = new BanditEnvironment(new BanditOptions { Arms = 4, Means = [0.2, 0.5, 1.0, 0.8], Spreads = [1, 1, 0.5, 0], Pulls = 100 });
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
