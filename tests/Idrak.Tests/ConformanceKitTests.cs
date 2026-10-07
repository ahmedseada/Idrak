// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Generation.Abstractions;

// Plan 10, phases 5 and 6c: the testing kit (Idrak.Abstraction.Testing) is the one source of the per-device operation
// checks. The device check runs on every device of the run; the contract checks, stress runs and saved cases once.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ConformanceKitGroup =
    [
        ("conformance kit: every operation on the device agrees with the CPU, call by call, and the CPU with plain loops (Conformance.Check)", DeviceConformant),
        ("conformance kit: a device with memory and copies only passes the device check; a wrong kernel registered for it fails it, and the saved case replays the failure until the kernel is removed", KitFindsWrongKernel),
        ("conformance kit: a stress run replays the operations from several threads with no failure and no memory growth; a cancelled run stops early and leaves no memory behind", DeviceStress),
        ("conformance kit: the library's token sampler, tokenizers and RoPE scalings pass their contract suites; a sampler and a scaling that differ fail them", ContractsConformant),
        ("conformance kit: contract stress runs (threads, large inputs) pass; a failing case saved to a file replays the same failure", ContractStressAndRegression),
    ];

    private static void DeviceConformant(Device device)
    {
        var report = Conformance.Check(device);
        if (Environment.GetEnvironmentVariable("IDRAK_TRACE") == "1")
        {
            Console.WriteLine(report);
        }

        report.ThrowIfFailed();
        int checkedOperations = report.Entries.Count(e => e.Status == CheckStatus.Passed);
        Check(checkedOperations >= (device.Type == DeviceType.Cpu ? 90 : 80), $"only {checkedOperations} operations checked:\n{report}");
        Check(report.Entries.Count == Ops.All.Count, "one entry per operation");
    }

    private static void KitFindsWrongKernel(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(KitFindsWrongKernel)))
        {
            return;
        }

        var minimal = MinimalBackend.Instance;
        var options = new DeviceCheckOptions { Filter = name => name.StartsWith("softmax", StringComparison.Ordinal) || name.StartsWith("element-wise", StringComparison.Ordinal) };
        var report = Conformance.Check(minimal, options);
        report.ThrowIfFailed();
        Check(report.Entries.Single(e => e.Name == "Softmax").Detail.StartsWith("the host fallback", StringComparison.Ordinal), $"Softmax: {report.Entries.Single(e => e.Name == "Softmax")}");
        Check(report.Skipped.Any(e => e.Name == "Im2Col" && e.Detail.Contains("no case", StringComparison.Ordinal)), "a filtered-out operation is reported as not reached");

        // Softmax off by a little on the minimal device: found, named, saved and replayed.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-kit-{Guid.NewGuid():N}");
        try
        {
            OperationKernels.Softmax wrong = (backend, x, y, rows, cols, log) =>
            {
                backend.SoftmaxKernel(x, y, rows, cols, log);
                backend.Affine(y, y, rows * cols, 1.01f, 0f);
            };
            using (Kernels.Register(Ops.Softmax, "minimal", wrong))
            {
                var failing = Conformance.Check(minimal, options);
                Check(!failing.Passed && failing.Failures.All(f => f.Check == "Softmax") && failing.Entries.Single(e => e.Name == "Softmax").Status == CheckStatus.Failed,
                    $"the wrong softmax is found:\n{failing}");
                Check(failing.Failures[0].Message.Contains("element", StringComparison.Ordinal) && failing.Failures[0].Message.Contains("y:", StringComparison.Ordinal),
                    $"the failure names the argument and element: {failing.Failures[0]}");
                Check(failing.Entries.Single(e => e.Name == "Softmax").Detail.StartsWith("a registered kernel", StringComparison.Ordinal), "the chain names the registered kernel");
                var saved = failing.SaveFailures(folder);
                Check(saved.Count == failing.Failures.Count && saved.All(File.Exists), $"saved {saved.Count} cases");
                Check(!Regression.Replay(folder, minimal).Passed, "the saved cases fail while the wrong kernel is registered");
            }

            var replay = Regression.Replay(folder, minimal);
            Check(replay.Passed && replay.Cases == Directory.GetFiles(folder).Length, $"the saved cases pass once it is removed:\n{replay}");
            Check(Regression.Replay(folder, Device.Cpu).Passed, "and on the CPU");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    private static void DeviceStress(Device device)
    {
        var report = Stress.Run(device, new StressOptions { Iterations = 400, Threads = 2, Large = false });
        report.ThrowIfFailed();
        Check(report.Iterations == 400 && !report.Cancelled && report.Slowest.Count > 0, report.ToString());

        if (FirstRun(nameof(DeviceStress)))
        {
            using var cancel = new CancellationTokenSource();
            cancel.Cancel();
            var cancelled = Stress.Run(device, new StressOptions { Iterations = 1_000_000, Threads = 2, Large = false }, cancel.Token);
            Check(cancelled.Cancelled && cancelled.Iterations < 1_000_000 && cancelled.Passed, cancelled.ToString());
            var timed = Stress.Run(Device.Cpu, new StressOptions { Duration = TimeSpan.FromMilliseconds(300), Threads = 3, Large = false });
            Check(timed.Passed && timed.Iterations > 0 && timed.Elapsed < TimeSpan.FromSeconds(30), timed.ToString());
        }
    }

    private static void ContractsConformant(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ContractsConformant)))
        {
            return;
        }

        var samplers = new TokenSamplerSuite(Drive(TokenSampler.Create));
        var sampler = Conformance.Check(Drive(TokenSampler.Create), samplers);
        sampler.ThrowIfFailed();
        Check(sampler.Entries.Any(e => e.Name == "greedy tokens are the reference's" && e.Status == CheckStatus.Passed)
              && sampler.Entries.Any(e => e.Name == "tokens the settings allow" && e.Status == CheckStatus.Passed), sampler.ToString());
        Conformance.Check(Drive(TokenSampler.Create), new TokenSamplerSuite(Drive(TokenSampler.Create)) { Exact = true }).ThrowIfFailed();

        // A sampler that ignores the settings and always takes the last token fails the allowed-tokens and greedy checks.
        var last = Conformance.Check(Drive(request => new LastTokenSampler(request)), samplers, new ContractCheckOptions { RandomCases = 4 });
        Check(!last.Passed && last.Failures.Any(f => f.Check == "greedy tokens are the reference's"), last.ToString());

        Conformance.Check(new CharTokenizer("abcdefghijklmnopqrstuvwxyz ,.!?"), new CharTokenizer("abcdefghijklmnopqrstuvwxyz ,.!?")).ThrowIfFailed();
        var words = WordTokenizer.FromTexts(["hello world, the model is learning.", "the data and the token"], ["<unk>", "<s>"]);
        Conformance.Check(words, words).ThrowIfFailed();
        var exact = Conformance.Check(new CharTokenizer("ab"), options: null);
        exact.ThrowIfFailed();
        Check(!Conformance.Check(new CharTokenizer("ab"), new TokenizerSuite { ExactRoundTrip = true }).Passed, "dropping characters breaks an exact round trip");

        foreach (string type in new[] { "linear", "llama3", "yarn", "dynamic" })
        {
            Conformance.CheckRopeScaling(type).ThrowIfFailed();
        }

        // A YaRN that forgets its attention factor differs from the library's.
        RopeScalingMethod noFactor = input => RopeScalings.Default("yarn")!(input) with { AttentionFactor = 1.0 };
        var yarn = Conformance.CheckRopeScaling("yarn", noFactor);
        Check(!yarn.Passed && yarn.Failures.All(f => f.Check == "agrees with the library's method"), yarn.ToString());
    }

    private static void ContractStressAndRegression(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ContractStressAndRegression)))
        {
            return;
        }

        var options = new StressOptions { Iterations = 60, Threads = 4 };
        Stress.Run(new CharTokenizer("abcdefghijklmnopqrstuvwxyz "), new TokenizerSuite(), options).ThrowIfFailed();
        Stress.Run(Drive(TokenSampler.Create), new TokenSamplerSuite(Drive(TokenSampler.Create)), options with { Iterations = 30 }).ThrowIfFailed();
        Stress.Run(RopeScalings.Get("dynamic"), new RopeScalingSuite("dynamic"), options).ThrowIfFailed();

        // A scaling of an app's own type, checked with its parameters; a broken one's failing case saved and replayed.
        var suite = new RopeScalingSuite("app-scaled", [new JsonObject { ["factor"] = 2.0 }]);
        RopeScalingMethod good = input => new RopeScalingResult([.. input.Frequencies.Select(f => f / 2)]);
        RopeScalingMethod broken = input => new RopeScalingResult([.. input.Frequencies.Select(f => input.RotaryDim > 100 ? double.NaN : f / 2)]);
        Conformance.Check(good, suite).ThrowIfFailed();
        var failing = Conformance.Check(broken, suite, new ContractCheckOptions { RandomCases = 40, Large = true });
        Check(!failing.Passed && failing.Failures.All(f => f.Check == "frequencies finite and positive" && f.Repro is not null), failing.ToString());
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-kit-{Guid.NewGuid():N}");
        try
        {
            failing.SaveFailures(folder);
            var replayed = Regression.Replay(folder, broken, suite);
            Check(!replayed.Passed && replayed.Cases == failing.Failures.Count, replayed.ToString());
            Check(Regression.Replay(folder, good, suite).Passed, "the fixed method passes the saved cases");
            Check(Regression.LoadAll(folder).All(c => c.Kind == "rope-scaling"), "saved as rope-scaling cases");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // An ITokenSampler factory (the token-sampler contract of Idrak.Nlp) as the kit's sampler checks drive it.
    internal static Func<SamplerSettings, SamplerUnderTest> Drive(Func<SamplerRequest, ITokenSampler> create) => settings =>
    {
        var sampler = create(new SamplerRequest(settings.Device, settings.Rows, settings.Vocabulary, settings.Steps, settings.HistoryCapacity, new GenerationOptions
        {
            Temperature = settings.Temperature, TopK = settings.TopK, TopP = settings.TopP, MinP = settings.MinP, RepeatPenalty = settings.RepeatPenalty,
            RepeatLastN = settings.RepeatLastN, PresencePenalty = settings.PresencePenalty, FrequencyPenalty = settings.FrequencyPenalty, Seed = settings.Seed,
        }));
        return new SamplerUnderTest(sampler.Rows, sampler.Vocabulary, sampler.SetHistory, logits =>
        {
            sampler.Sample(logits);
            return sampler.Ids.ToArray();
        }, sampler.Reset, (from, to) => [.. sampler.Read(from, to).Select(step => step.Select(t => new SampledStatistics(t.Id, t.Probability, t.Entropy)).ToArray())], sampler);
    };

    // Takes the last token of the vocabulary every time.
    private sealed class LastTokenSampler(SamplerRequest request) : ITokenSampler
    {
        private readonly List<SampledToken[]> _steps = [];

        public int Rows => request.Rows;

        public int Vocabulary => request.Vocabulary;

        public Tensor Ids { get; } = Tensor.Persistent(new float[request.Rows], [request.Rows], request.Device);

        public bool Recordable => false;

        public void Sample(Tensor logits)
        {
            Ids.Load([.. Enumerable.Repeat((float)(Vocabulary - 1), Rows)]);
            _steps.Add([.. Enumerable.Repeat(new SampledToken(Vocabulary - 1, 1f, 0f, []), Rows)]);
        }

        public void SetHistory(IReadOnlyList<int> tokens)
        {
        }

        public void Reset() => _steps.Clear();

        public SampledToken[][] Read(int fromStep, int toStep) => [.. _steps.Skip(fromStep).Take(toStep - fromStep)];

        public void Dispose() => Ids.Dispose();
    }
}
