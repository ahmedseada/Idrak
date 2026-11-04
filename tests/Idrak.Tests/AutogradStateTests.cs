// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Training;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;

// A second backward pass, the optimizer's step count and the tuning caches' settings (plan 36, G-1 3b).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] AutogradStateGroup =
    [
        ("autograd: a second Backward through a result an earlier pass freed throws naming it (its gradient is never dropped silently); Backward(retainGraph: true) keeps the graph and both passes add up", AutogradSecondPass),
        ("optimizer: Adam and AdamW take PyTorch's update, lr·m̂/(√v̂ + ε) with the bias-corrected moments (not ε added to the raw √v), through Step and the fused ClipAndStep, within 1e-5 of a double-precision reference where the two forms differ by far more", AdamMatchesPyTorch),
        ("optimizer state: the step count is saved as an exact integer past 2^24 (the low 24 bits and the rest), an earlier one-float state still loads, and a step that is not a whole number is refused naming it", d => { CpuOnly(d); OptimizerStepState(); }),
        ("tuning cache: 0, false, off and no turn tuning and its caches off for every backend; a save waits for another process's lock and keeps both processes' choices; threads saving the CPU's tuning file at once keep every line", d => { CpuOnly(d); TuningCacheSettings(); }),
    ];

    private static void AutogradSecondPass(Device device)
    {
        // Two losses read one activation h = x·x; the first pass frees h's step.
        var x = Tensor.From([1f, 2f, 3f], [3], device, requiresGrad: true);
        var h = x * x;
        var first = h.Sum();
        var second = (h * 3f).Sum();
        first.Backward();
        float[] once = x.Grad!.ToArray();
        Check(once.SequenceEqual([2f, 4f, 6f]), $"the first pass: dx = 2x, [{string.Join(", ", once)}]");

        string? refusal = null;
        try
        {
            second.Backward();
        }
        catch (InvalidOperationException e)
        {
            refusal = e.Message;
        }

        Check(refusal is not null && refusal.Contains("shape [3]", StringComparison.Ordinal) && refusal.Contains("Backward(retainGraph: true)", StringComparison.Ordinal)
              && refusal.Contains("dropped", StringComparison.Ordinal), $"a second pass through the freed activation throws naming it and the option: {refusal ?? "it ran"}");
        Check(x.Grad!.ToArray().SequenceEqual(once), "the refused pass added nothing to the leaf's gradient");

        // The graph kept: both passes reach x, as PyTorch's retain_graph does (2x + 3·2x = 8x).
        var y = Tensor.From([1f, 2f, 3f], [3], device, requiresGrad: true);
        var g = y * y;
        var a = g.Sum();
        var b = (g * 3f).Sum();
        a.Backward(retainGraph: true);
        b.Backward();
        float[] both = y.Grad!.ToArray();
        Check(both.SequenceEqual([8f, 16f, 24f]), $"with retainGraph both passes add up: dy = 8y, [{string.Join(", ", both)}]");

        bool again = false;
        try
        {
            a.Backward();                                                       // b's pass freed the graph (no retainGraph)
        }
        catch (InvalidOperationException)
        {
            again = true;
        }

        Check(again, "a pass after the last one that kept the graph throws again");
    }

    // PyTorch's Adam (torch.optim.Adam / AdamW, single-tensor), in double: the reference the update must equal.
    private static double[] TorchAdam(double[] start, double[][] gradients, double lr, double beta1, double beta2, double eps, double weightDecay, bool decoupled, bool torchEpsilon)
    {
        var p = (double[])start.Clone();
        var m = new double[p.Length];
        var v = new double[p.Length];
        for (int t = 1; t <= gradients.Length; t++)
        {
            double bias1 = 1 - Math.Pow(beta1, t), bias2 = 1 - Math.Pow(beta2, t);
            for (int i = 0; i < p.Length; i++)
            {
                double g = gradients[t - 1][i];
                if (decoupled)
                {
                    p[i] *= 1 - lr * weightDecay;
                }
                else
                {
                    g += weightDecay * p[i];
                }

                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                p[i] -= torchEpsilon
                    ? lr * (m[i] / bias1) / (Math.Sqrt(v[i] / bias2) + eps)                         // torch: eps added to sqrt(v̂)
                    : lr * Math.Sqrt(bias2) / bias1 * m[i] / (Math.Sqrt(v[i]) + eps);             // the earlier form: eps added to sqrt(v)
            }
        }

        return p;
    }

    private static void AdamMatchesPyTorch(Device device)
    {
        double[] start = [0.5, -1.0, 2.0, 1e-3];
        double[][] gradients = [[1e-3, -2e-3, 5e-4, 1e-3], [2e-3, -1e-3, 1e-3, -5e-4], [-1e-3, 3e-3, 2e-3, 1e-3]];      // near ε: the forms differ
        const float Lr = 1e-2f, Beta1 = 0.9f, Beta2 = 0.999f, Eps = 1e-3f, Decay = 0.1f;
        foreach (bool decoupled in new[] { false, true })
        {
            double[] torch = TorchAdam(start, gradients, Lr, Beta1, Beta2, Eps, decoupled ? Decay : 0, decoupled, torchEpsilon: true);
            double[] earlier = TorchAdam(start, gradients, Lr, Beta1, Beta2, Eps, decoupled ? Decay : 0, decoupled, torchEpsilon: false);
            double apart = torch.Zip(earlier).Max(x => Math.Abs(x.First - x.Second));
            Check(apart > 1e-4, $"the reference tells the two forms apart ({apart:G3})");
            foreach (bool fused in new[] { false, true })
            {
                var p = Tensor.From([.. start.Select(x => (float)x)], [start.Length], device, requiresGrad: true);
                using Optimizer optimizer = decoupled ? new AdamW([p], Lr, Beta1, Beta2, Eps, Decay) : new Adam([p], Lr, Beta1, Beta2, Eps);
                foreach (var g in gradients)
                {
                    optimizer.ZeroGrad();
                    using var weights = Tensor.From([.. g.Select(x => (float)x)], [g.Length], device);
                    (p * weights).Sum().Backward();                                    // dp = g
                    if (fused)
                    {
                        optimizer.ClipAndStep(1e6f);                                   // a norm no gradient reaches: the fused pass, no clipping
                    }
                    else
                    {
                        optimizer.Step();
                    }
                }

                float[] got = p.ToArray();
                double gap = got.Zip(torch).Max(x => Math.Abs(x.First - x.Second));
                Check(gap <= 1e-5, $"{(decoupled ? "AdamW" : "Adam")}, {(fused ? "ClipAndStep" : "Step")}: within 1e-5 of PyTorch's update ({gap:G3}; the earlier form is {apart:G3} away): "
                    + $"[{string.Join(", ", got)}] vs [{string.Join(", ", torch.Select(x => x.ToString("G7", System.Globalization.CultureInfo.InvariantCulture)))}]");
            }
        }
    }

    private static void OptimizerStepState()
    {
        using var p = Tensor.From([1f, 2f], [2], requiresGrad: true);
        using var adam = new Adam([p]);
        IReadOnlyDictionary<string, OptimizerStateValue> Step(params float[] values) =>
            new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal) { ["step"] = new([values.Length], values) };

        const int Large = (1 << 24) + 3;                                       // 16,777,219: one float rounds it to ...220
        adam.LoadState(Step(3f, 1f));                                           // the integer form: low 24 bits, then the rest
        var saved = adam.SaveState()["step"];
        Check(saved.Shape.SequenceEqual([2]) && saved.Values.SequenceEqual([3f, 1f]), $"step {Large} saved exactly as [low, high]: [{string.Join(", ", saved.Values)}]");
        Check((int)saved.Values[0] + ((int)saved.Values[1] << 24) == Large, "the two parts give the step back");

        adam.LoadState(Step(12345f));                                           // a checkpoint written before: one float
        Check(adam.SaveState()["step"].Values.SequenceEqual([12345f, 0f]), "an earlier one-float step loads, and saves in the integer form");

        string? refusal = null;
        try
        {
            adam.LoadState(Step(1.5f));
        }
        catch (ArgumentException e)
        {
            refusal = e.Message;
        }

        Check(refusal is not null && refusal.Contains("'step'", StringComparison.Ordinal), $"a step that is not a whole number is refused naming it: {refusal ?? "loaded"}");
    }

    private static void TuningCacheSettings()
    {
        foreach (string? setting in new[] { "0", "false", "FALSE", "off", "Off", "no", " off " })
        {
            Check(TuningFiles.IsOff(setting), $"'{setting}' turns a tuning setting off");
        }

        foreach (string? setting in new[] { null, "", "1", "true", "on", "/tmp/somewhere" })
        {
            Check(!TuningFiles.IsOff(setting), $"'{setting}' does not turn it off");
        }

        foreach (string off in new[] { "off", "no", "False" })
        {
            string? Environment(string name) => name == "IDRAK_TUNING_CACHE" || name == "IDRAK_HIP_KERNEL_CACHE" ? off : null;
            Check(TuningCache.Folder(Environment) is null && TuningCache.Folder(Environment, "hip") is null && HipKernelCache.DefaultFolder(Environment) is null,
                $"IDRAK_TUNING_CACHE / IDRAK_HIP_KERNEL_CACHE={off}: CUDA's, HIP's tuning and HIP's kernel caches are off");
        }

        string folder = TempFolder();
        string?[] saved = [.. new[] { "IDRAK_VULKAN_TUNING_CACHE", "IDRAK_TUNING_CACHE" }.Select(System.Environment.GetEnvironmentVariable)];
        try
        {
            System.Environment.SetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE", "off");
            System.Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", "no");
            Check(VulkanBackend.DefaultTuningCacheFile() is null && CpuTuning.DefaultCacheFile() is null,
                "IDRAK_VULKAN_TUNING_CACHE=off and IDRAK_TUNING_CACHE=no turn Vulkan's and the CPU's tuning files off (they took 0 and false only)");
            System.Environment.SetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE", saved[0]);
            System.Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", saved[1]);

            // A save waits while another process holds the file's lock, and keeps what that process wrote meanwhile.
            string path = Path.Combine(folder, "device.txt");
            var identity = new TuningIdentity("a GPU", "a driver", "a build");
            var ours = new TuneKey(TuneOp.GemvSplits, 0, 1, 2048, 1024);
            var theirs = new TuneKey(TuneOp.GemvSplits, 0, 1, 4096, 1024);
            Task<bool> saving;
            using (TuningFiles.Lock(path))
            {
                saving = Task.Run(() => TuningCache.Save(path, identity, new Dictionary<TuneKey, int> { [ours] = 16 }));
                Thread.Sleep(300);
                Check(!saving.IsCompleted, "the save waits for the lock another process holds");
                File.WriteAllText(path, TuningCache.Serialize(identity, new Dictionary<TuneKey, int> { [theirs] = 8 }));     // that process's merge
            }

            Check(saving.Wait(TimeSpan.FromSeconds(30)) && saving.Result, "the save goes on once the lock is free");
            var kept = TuningCache.Load(path, identity);
            Check(kept.GetValueOrDefault(ours) == 16 && kept.GetValueOrDefault(theirs) == 8, $"both processes' choices are kept: {kept.Count} entries");

            // Threads stand in for processes (the lock is per open file): every save of the CPU's file keeps every line.
            string cpuFile = Path.Combine(folder, "cpu", "tuning.tsv");
            CpuTuning.CacheFile = cpuFile;
            try
            {
                Parallel.For(0, 24, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i => CpuTuning.Save($"machine-{i}", new CpuTuning.Cutovers(1000 + i, 2000 + i, "measured")));
            }
            finally
            {
                CpuTuning.UseDefaultCacheFile();
            }

            var lines = File.ReadAllLines(cpuFile);
            Check(lines.Length == 24 && Enumerable.Range(0, 24).All(i => lines.Any(l => l.StartsWith($"machine-{i}\t", StringComparison.Ordinal))),
                $"24 saves at once keep 24 lines, not {lines.Length}");
        }
        finally
        {
            System.Environment.SetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE", saved[0]);
            System.Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", saved[1]);
            Directory.Delete(folder, recursive: true);
        }
    }
}
