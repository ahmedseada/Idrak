// Self-contained test runner (no test framework dependency). Runs on the CPU and every CUDA GPU;
//   IDRAK_DEVICES=cpu,cuda:0,cuda:2 …   only these devices
//   dotnet run --project tests/Idrak.Tests                  run every test on every available device
//   dotnet run --project tests/Idrak.Tests -- --dump-ptx f  write the generated CUDA kernels to f
//   … -- --bench-cpu                                                             time the CPU decoding kernels (bfloat16 products, cached attention, sampling)
//   … -- --bench-gemm / --bench-gemv / --bench-fp8                               time large products / decoding-sized packed products
//   IDRAK_FILTER=retrieval dotnet run --project tests/Idrak.Tests   only tests whose name contains the text
//   IDRAK_TIMEOUT=60 …    a test still running after this many seconds (default 300) is reported as HANG and the run stops

using System.Diagnostics;
using Idrak;
using Idrak.Backends.Cuda;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

if (args is ["--bench-gemm"])
{
    return Tests.BenchGemm();
}

if (args is ["--bench-gemv"])
{
    return Tests.BenchGemv();
}

if (args is ["--bench-cpu"])
{
    return Tests.BenchCpu();
}

if (args is ["--bench-fp8"])
{
    return Tests.BenchFp8Quantizers();
}

if (args is ["--dump-ptx", var ptxPath])
{
    File.WriteAllText(ptxPath, PtxKernels.Source);
    File.WriteAllText(Path.ChangeExtension(ptxPath, ".tensorcore.ptx"), PtxKernels.TensorCoreSource);
    Console.WriteLine($"Wrote {PtxKernels.Source.Length} characters of PTX to {ptxPath}");
    return 0;
}

// What the run is on, so a pasted result says where it passed (each GPU's line adds its compute capability and driver).
Console.WriteLine($"Idrak tests: {System.Runtime.InteropServices.RuntimeInformation.OSDescription} "
                  + $"({System.Runtime.InteropServices.RuntimeInformation.OSArchitecture}), {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}");

// Every device: the CPU and each CUDA GPU (IDRAK_DEVICES=cpu,cuda:1 … to choose), since the library runs on any of them.
var devices = new List<Device> { Device.Cpu };
if (Device.IsCudaAvailable)
{
    for (int i = 0; i < Device.CudaDeviceCount; i++)
    {
        devices.Add(Device.Cuda(i));
    }

    if (Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen)
    {
        var names = chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        devices = [.. devices.Where(d => names.Contains(d.ToString(), StringComparer.OrdinalIgnoreCase)
                                         || d.Type == DeviceType.Cuda && d.Ordinal == 0 && names.Contains("cuda", StringComparer.OrdinalIgnoreCase))];
    }
}
else
{
    Console.WriteLine($"CUDA not available ({CudaBackend.UnavailableReason}); testing the CPU only.");
}

int failed = 0, passed = 0;
string? running = null;
var runningSince = Stopwatch.StartNew();
int timeout = int.TryParse(Environment.GetEnvironmentVariable("IDRAK_TIMEOUT"), out int t) ? t : 300;
var watchdog = new Thread(() =>
{
    while (true)
    {
        Thread.Sleep(1000);
        if (Volatile.Read(ref running) is { } current && runningSince.Elapsed.TotalSeconds > timeout)
        {
            Console.WriteLine($"  HANG {current}: still running after {timeout} s");
            Console.WriteLine($"{passed} passed, {failed} failed, 1 hung");
            Environment.Exit(3);
        }
    }
}) { IsBackground = true };
watchdog.Start();
foreach (var device in devices)
{
    Console.WriteLine($"== {device}: {device.Name}");
    string? lostIn = null;
    foreach (var (name, test) in Tests.All.Where(t => Environment.GetEnvironmentVariable("IDRAK_FILTER") is not { } f || t.Name.Contains(f, StringComparison.OrdinalIgnoreCase)))
    {
        if (lostIn is not null)
        {
            failed++;
            continue;                                                            // counted, not run: the device is unusable
        }

        var sw = Stopwatch.StartNew();
        runningSince.Restart();
        Volatile.Write(ref running, $"[{device}] {name}");
        try
        {
            using (new TensorScope())
            {
                test(device);
            }

            passed++;
            Console.WriteLine($"  PASS {name} ({sw.ElapsedMilliseconds} ms)");
        }
        catch (Exception ex)
        {
            failed++;
            Console.WriteLine($"  FAIL {name}: {(Environment.GetEnvironmentVariable("IDRAK_TRACE") == "1" ? ex.ToString() : ex.Message)}");

            // A kernel that faults (illegal or misaligned address, illegal instruction, launch failure) ruins the CUDA
            // context: every later call on it fails the same way. Name the test and skip the rest on this device.
            if (device.Type == DeviceType.Cuda && ex.Message is var message
                && (message.Contains("ILLEGAL", StringComparison.Ordinal) || message.Contains("MISALIGNED", StringComparison.Ordinal)
                    || message.Contains("LAUNCH_FAILED", StringComparison.Ordinal) || message.Contains("HARDWARE_STACK_ERROR", StringComparison.Ordinal)))
            {
                lostIn = name;
                Console.WriteLine($"  the CUDA context was lost in this test: the remaining {device} tests are skipped and counted as failed.");
                Console.WriteLine("  (the fault is reported where the device is next used; IDRAK_CUDA_DEBUG=1 names the kernel that faulted)");
            }
        }
    }
}

Volatile.Write(ref running, null);
Console.WriteLine($"{passed} passed, {failed} failed");
return failed == 0 ? 0 : 1;

internal static partial class Tests
{
    public static (string Name, Action<Device> Run)[] All => [.. Basic, .. Advanced, .. Decoding, .. Generation, .. Simplified, .. AspNetCore, .. Retrieval, .. Onnx, .. Quantization, .. Decoder, .. Pretrained, .. ChatTemplates, .. FineTuning, .. MixedPrecisionGroup, .. CodingToolsGroup, .. DatasetsGroup, .. GgufGroup, .. NamingGroup, .. StreamingText];

    private static readonly (string Name, Action<Device> Run)[] Basic =
    [
        ("matmul matches reference (all transposes, beta 0 and 1)", MatMulReference),
        ("element-wise ops match reference", ElementWiseReference),
        ("large tensors (parallel / multi-block paths)", LargeTensors),
        ("gradient: sigmoid, sum", d => GradCheck(d, [3, 4], x => x.Sigmoid().Sum())),
        ("gradient: tanh, mean", d => GradCheck(d, [5, 3], x => x.Tanh().Mean())),
        ("gradient: relu, square", d => GradCheck(d, [4, 4], x => (x.Relu() + x.Square()).Sum(), avoidZero: true)),
        ("gradient: mul, sub, scalar ops", d => GradCheck(d, [2, 6], x => ((x * x) - (2f * x) + 1f - (x / 4f)).Mean() * 3f)),
        ("gradient: matmul left and right", GradMatMul),
        ("gradient: bias broadcast, reshape", GradBias),
        ("gradient: mean squared error", GradMse),
        ("backward accumulates and ZeroGrad resets", Accumulation),
        ("optimizers minimise a quadratic", Optimizers),
        ("save and load round-trip", SaveLoad),
        ("tensor scope releases intermediates", Scope),
        ("gradient: abs", d => GradCheck(d, [3, 5], x => x.Abs().Mean(), avoidZero: true)),
        ("gradient: dropout (fixed mask)", d => GradCheck(d, [4, 8], x => x.Dropout(0.3f, 1234).Tanh().Sum())),
        ("dropout drops the right fraction and matches the CPU mask", DropoutBehaviour),
        ("csv parsing, split and scalers", CsvAndScalers),
        ("data loader covers every sample (with prefetch)", LoaderCoverage),
        ("trainer fits a noisy linear function", TrainerLearns),
        ("telemetry hooks receive events only when subscribed", TelemetryHooks),
        ("memory limit is enforced", MemoryLimit),
        ("single-threaded CPU matches multi-threaded", ThreadBudget),
        ("prediction does not leak memory", PredictionMemory),
    ];

    private static void DropoutBehaviour(Device device)
    {
        const int N = 100_000;
        using var ones = Tensor.Ones([N], device);
        var y = ones.Dropout(0.25f, 99).ToArray();
        int kept = y.Count(v => v != 0f);
        Check(Math.Abs(kept / (double)N - 0.75) < 0.01, $"kept fraction {kept / (double)N:F4}, expected 0.75");
        Check(y.Where(v => v != 0f).All(v => MathF.Abs(v - 1f / 0.75f) < 1e-5f), "survivors must be scaled by 1/(1-p)");
        using var cpuOnes = Tensor.Ones([N], Device.Cpu);
        var cpu = cpuOnes.Dropout(0.25f, 99).ToArray();
        Check(cpu.SequenceEqual(y), "the dropout mask must be identical on every device");
        for (int i = 0; i < N; i++)
        {
            // The vectorized kernels must reproduce the scalar reference hash exactly.
            Check((y[i] != 0f) == Idrak.Backends.DropoutMask.Keep(99, (uint)i, 0.25f), $"dropout mask differs from the reference at {i}");
        }

        var layer = new Dropout(0.5f);
        layer.Eval();
        using var x = Tensor.From([1f, 2f, 3f], device);
        Check(ReferenceEquals(layer.Forward(x), x), "dropout must be the identity in evaluation mode");
    }

    private static void CsvAndScalers(Device device)
    {
        const string csv = """
            id,size,rooms,price
            1,100,2,200.5
            2,150,3,"310"

            3,80,1,150
            4,120,2,260
            """;
        var data = Dataset.ParseCsv(csv, new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
        Check(data.Count == 4 && data.FeatureCount == 2 && data.TargetCount == 1, $"parsed {data}");
        Check(data.FeatureNames.SequenceEqual(["size", "rooms"]), "feature names");
        AssertClose([150f, 3f], data.GetFeatures(1).ToArray(), 0, "row 2 features");
        AssertClose([310f], data.GetTargets(1).ToArray(), 0, "quoted target");

        var (train, test) = data.Split(0.5, seed: 3);
        Check(train.Count == 2 && test.Count == 2, "split sizes");

        var scaler = StandardScaler.FitFeatures(data);
        var scaled = data.Scale(scaler);
        var column0 = Enumerable.Range(0, 4).Select(i => scaled.GetFeatures(i)[0]).ToArray();
        AssertClose([0f], [column0.Average()], 1e-5f, "scaled mean");
        var restored = scaled.Features.ToArray();
        scaler.InverseTransform(restored, 2);
        AssertClose(data.Features.ToArray(), restored, 1e-5f, "inverse transform");

        try
        {
            Dataset.ParseCsv("a,b\n1,x\n", new CsvOptions { TargetColumns = ["b"] });
            throw new Exception("bad number accepted");
        }
        catch (FormatException ex) when (ex.Message.Contains("line 2"))
        {
        }
    }

    private static void LoaderCoverage(Device device)
    {
        // 3,000 rows x 20 columns at batch 1,000 is large enough to use the background prefetch path.
        const int Rows = 3_001, Cols = 20;
        var features = new float[Rows, Cols];
        var targets = new float[Rows, 1];
        for (int i = 0; i < Rows; i++)
        {
            features[i, 0] = i;
            targets[i, 0] = i;
        }

        var loader = new DataLoader(Dataset.FromArrays(features, targets), batchSize: 1_000, shuffle: true, device: device, seed: 5);
        Check(loader.BatchCount == 4, "batch count");
        for (int epoch = 0; epoch < 2; epoch++)
        {
            var seen = new List<float>();
            foreach (var batch in loader)
            {
                using (batch)
                {
                    var x = batch.Features.ToArray();
                    var y = batch.Targets.ToArray();
                    for (int r = 0; r < batch.Size; r++)
                    {
                        Check(x[r * Cols] == y[r], "features and targets must stay aligned");
                        seen.Add(y[r]);
                    }
                }
            }

            seen.Sort();
            Check(seen.SequenceEqual(Enumerable.Range(0, Rows).Select(i => (float)i)), "every sample exactly once per epoch");
        }
    }

    private static void TrainerLearns(Device device)
    {
        var random = new Random(8);
        const int N = 512;
        var x = new float[N, 3];
        var y = new float[N, 1];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                x[i, j] = random.NextSingle() * 2 - 1;
            }

            y[i, 0] = 2 * x[i, 0] - 3 * x[i, 1] + 0.5f * x[i, 2] + 1 + (random.NextSingle() - 0.5f) * 0.01f;
        }

        var (train, test) = Dataset.FromArrays(x, y).Split(0.8, seed: 1);
        using var model = new Sequential { new Linear(3, 1, device: device, random: random) };
        using var optimizer = new Adam(model.Parameters(), 0.05f);
        var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError) { Metrics = { Metric.RootMeanSquaredError } };
        var history = trainer.Fit(new DataLoader(train, 32, shuffle: true, device: device, seed: 2), epochs: 60,
            validation: new DataLoader(test, 128, device: device));
        var result = trainer.Evaluate(new DataLoader(test, 128, device: device));
        Check(result.Metrics["rmse"] < 0.02, $"test rmse {result.Metrics["rmse"]:F4}");
        Check(history.Epochs.Count == 60 && history.BestEpoch > 0, "history");
        var predictions = trainer.Predict(test);
        var report = RegressionReport.Compute(predictions.Cast<float>().ToArray(), test.Targets);
        Check(report.RSquared > 0.999, $"R² {report.RSquared:F5}");
    }

    private sealed class CountingHook(TelemetryLevel levels) : ITelemetryHook
    {
        public TelemetryLevel Levels => levels;
        public int Started, Batches, Epochs, Completed, Layers, Operations, Inferences;
        public double LastEpochLoss;

        public void OnTrainingStarted(in TrainingStarted e) => Started++;
        public void OnBatchCompleted(in BatchCompleted e) => Batches++;
        public void OnEpochCompleted(in EpochCompleted e)
        {
            Epochs++;
            LastEpochLoss = e.Loss;
        }

        public void OnTrainingCompleted(in TrainingCompleted e) => Completed++;
        public void OnLayerForward(in LayerForward e) => Layers++;
        public void OnOperation(in OperationCompleted e) => Operations++;
        public void OnInference(in InferenceCompleted e) => Inferences++;
    }

    private static void TelemetryHooks(Device device)
    {
        var data = Dataset.FromArrays(new float[,] { { 0 }, { 1 }, { 2 }, { 3 } }, new float[,] { { 0 }, { 2 }, { 4 }, { 6 } });
        using var model = new Sequential { new Linear(1, 4, device: device), new Tanh(), new Linear(4, 1, device: device) };
        using var optimizer = new Sgd(model.Parameters(), 0.01f);
        var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError);
        var loader = new DataLoader(data, 2, device: device);

        var hook = new CountingHook(TelemetryLevel.All);
        using (Telemetry.Subscribe(hook))
        {
            Check(Telemetry.IsEnabled(TelemetryLevel.Operations), "levels should be active while subscribed");
            trainer.Fit(loader, epochs: 3);
            using var input = Tensor.From(new float[,] { { 1 } }, device);
            using var output = model.Predict(input);
        }

        Check(hook.Started == 1 && hook.Completed == 1, "start/complete events");
        Check(hook.Epochs == 3 && hook.Batches == 6, $"epochs {hook.Epochs}, batches {hook.Batches}");
        Check(hook.Layers >= 3 * 2 * 4, $"layer events {hook.Layers}");
        Check(hook.Operations > 0 && hook.Inferences == 1, "operation and inference events");
        Check(double.IsFinite(hook.LastEpochLoss), "epoch loss");

        int before = hook.Epochs;
        Check(Telemetry.ActiveLevels == TelemetryLevel.None, "no levels after unsubscribing");
        trainer.Fit(loader, epochs: 1);
        Check(hook.Epochs == before, "no events after unsubscribing");
    }

    private static void MemoryLimit(Device device)
    {
        bool gpu = device.Type == DeviceType.Cuda;
        long? old = gpu ? ComputeResources.GpuMemoryLimit : ComputeResources.CpuMemoryLimit;
        long inUse = ComputeResources.GetMemoryUsage(device).InUse;
        try
        {
            long limit = inUse + 1_000_000;
            if (gpu) ComputeResources.GpuMemoryLimit = limit; else ComputeResources.CpuMemoryLimit = limit;
            using (var small = Tensor.Zeros([1000], device))
            {
            }

            try
            {
                using var big = Tensor.Zeros([1_000_000], device);
                throw new Exception("allocation over the limit succeeded");
            }
            catch (ResourceLimitExceededException)
            {
            }
        }
        finally
        {
            if (gpu) ComputeResources.GpuMemoryLimit = old; else ComputeResources.CpuMemoryLimit = old;
        }
    }

    private static void ThreadBudget(Device device)
    {
        var random = new Random(9);
        using var a = Tensor.From(RandomArray(random, 300 * 200), [300, 200], device);
        using var b = Tensor.From(RandomArray(random, 200 * 100), [200, 100], device);
        using var big = Tensor.From(RandomArray(random, 200_000), device);
        var parallelProduct = a.MatMul(b).ToArray();
        var parallelTanh = big.Tanh().ToArray();
        int old = ComputeResources.MaxCpuThreads;
        try
        {
            ComputeResources.MaxCpuThreads = 1;
            AssertClose(parallelProduct, a.MatMul(b).ToArray(), 1e-6f, "matmul with 1 thread");
            AssertClose(parallelTanh, big.Tanh().ToArray(), 0, "tanh with 1 thread");
        }
        finally
        {
            ComputeResources.MaxCpuThreads = old;
        }
    }

    private static void PredictionMemory(Device device)
    {
        using var model = new Sequential { new Linear(8, 64, device: device), new ReLU(), new Linear(64, 1, device: device) };
        using var x = Tensor.Ones([128, 8], device);
        long before = ComputeResources.GetMemoryUsage(device).InUse;
        for (int i = 0; i < 5; i++)
        {
            model.Predict(x).Dispose();
        }

        long after = ComputeResources.GetMemoryUsage(device).InUse;
        Check(after == before, $"in-use memory grew from {before} to {after} bytes");
    }

    private static void MatMulReference(Device device)
    {
        var random = new Random(1);
        foreach (var (m, n, k) in new[] { (1, 1, 1), (3, 5, 2), (17, 33, 9), (64, 48, 70), (130, 67, 91), (1100, 1030, 21), (1, 700, 300), (8, 300, 77), (16, 129, 65) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    foreach (float beta in new[] { 0f, 1f })
                    {
                        var a = RandomArray(random, m * k);
                        var b = RandomArray(random, k * n);
                        var c = RandomArray(random, m * n);
                        var expected = new float[m * n];
                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                double acc = 0;
                                for (int p = 0; p < k; p++)
                                {
                                    acc += (double)a[ta ? p * m + i : i * k + p] * b[tb ? j * k + p : p * n + j];
                                }

                                expected[i * n + j] = (float)(acc + beta * c[i * n + j]);
                            }
                        }

                        var backend = device.Backend;
                        using var ta_ = Tensor.From(a, device);
                        using var tb_ = Tensor.From(b, device);
                        using var tc = Tensor.From(c, device);
                        backend.MatMul(ta_.Storage, tb_.Storage, tc.Storage, m, n, k, ta, tb, beta);
                        AssertClose(expected, tc.ToArray(), 1e-4f, $"matmul m={m} n={n} k={k} transA={ta} transB={tb} beta={beta}");
                    }
                }
            }
        }
    }

    private static void ElementWiseReference(Device device)
    {
        var random = new Random(2);
        var x = RandomArray(random, 37, 4f);
        var y = RandomArray(random, 37, 4f);
        using var tx = Tensor.From(x, device);
        using var ty = Tensor.From(y, device);
        AssertClose(x.Select(v => 1f / (1f + MathF.Exp(-v))).ToArray(), tx.Sigmoid().ToArray(), 1e-5f, "sigmoid");
        AssertClose(x.Select(MathF.Tanh).ToArray(), tx.Tanh().ToArray(), 1e-5f, "tanh");
        AssertClose(x.Select(v => MathF.Max(v, 0)).ToArray(), tx.Relu().ToArray(), 0, "relu");
        AssertClose(x.Select(v => v * v).ToArray(), tx.Square().ToArray(), 1e-6f, "square");
        AssertClose(x.Zip(y, (a, b) => a + b).ToArray(), (tx + ty).ToArray(), 1e-6f, "add");
        AssertClose(x.Zip(y, (a, b) => a - b).ToArray(), (tx - ty).ToArray(), 1e-6f, "sub");
        AssertClose(x.Zip(y, (a, b) => a * b).ToArray(), (tx * ty).ToArray(), 1e-6f, "mul");
        AssertClose(x.Select(v => 3f * v - 2f).ToArray(), (3f * tx - 2f).ToArray(), 1e-5f, "affine");
        AssertClose([x.Sum()], [tx.Sum().Item()], 1e-4f, "sum");
        AssertClose([x.Average()], [tx.Mean().Item()], 1e-5f, "mean");
        using var saturated = Tensor.From([-100f, 100f, -1000f, 1000f], device);
        AssertClose([0f, 1f, 0f, 1f], saturated.Sigmoid().ToArray(), 1e-6f, "sigmoid saturation");
        AssertClose([-1f, 1f, -1f, 1f], saturated.Tanh().ToArray(), 1e-6f, "tanh saturation");
    }

    private static void LargeTensors(Device device)
    {
        const int N = 1_000_003;
        var random = new Random(3);
        var x = RandomArray(random, N);
        using var tx = Tensor.From(x, device);
        double expectedSum = x.Sum(v => (double)v);
        AssertClose([(float)expectedSum], [tx.Sum().Item()], 1e-2f, "large sum");
        var sig = tx.Sigmoid().ToArray();
        for (int i = 0; i < N; i += 9973)
        {
            AssertClose([1f / (1f + MathF.Exp(-x[i]))], [sig[i]], 1e-5f, $"large sigmoid[{i}]");
        }

        using var matrix = Tensor.From(x.AsSpan(0, 1000 * 1000), [1000, 1000], device);
        using var ones = Tensor.Ones([1000], device);
        var rowSums = matrix.MatMul(ones.Reshape(1000, 1)).ToArray();
        for (int r = 0; r < 1000; r += 97)
        {
            double expected = 0;
            for (int c = 0; c < 1000; c++)
            {
                expected += x[r * 1000 + c];
            }

            AssertClose([(float)expected], [rowSums[r]], 1e-3f, $"large matmul row {r}");
        }
    }

    private static void GradMatMul(Device device)
    {
        var random = new Random(4);
        using var w = Tensor.From(RandomArray(random, 4 * 3), [4, 3], device);
        using var v = Tensor.From(RandomArray(random, 5 * 4), [5, 4], device);
        GradCheck(device, [5, 4], x => x.MatMul(w).Tanh().Sum());
        GradCheck(device, [4, 3], x => v.MatMul(x).Sigmoid().Mean());
    }

    private static void GradBias(Device device)
    {
        var random = new Random(5);
        using var input = Tensor.From(RandomArray(random, 6 * 3), [6, 3], device);
        GradCheck(device, [3], b => (input + b).Tanh().Sum());
        GradCheck(device, [6, 3], x => x.Reshape(2, 9).Square().Sum());
    }

    private static void GradMse(Device device)
    {
        var random = new Random(6);
        using var target = Tensor.From(RandomArray(random, 8), [4, 2], device);
        GradCheck(device, [4, 2], x => Losses.MeanSquaredError(x.Sigmoid(), target));
    }

    private static void Accumulation(Device device)
    {
        using var x = Tensor.From([1f, 2f, 3f], device, requiresGrad: true);
        (x * x).Sum().Backward();
        (x * x).Sum().Backward();
        AssertClose([4f, 8f, 12f], x.Grad!.ToArray(), 1e-6f, "accumulated gradient");
        x.ZeroGrad();
        AssertClose([0f, 0f, 0f], x.Grad!.ToArray(), 0, "zeroed gradient");
        using (Autograd.NoGrad())
        {
            Check(!(x * x).RequiresGrad, "NoGrad should stop recording");
        }
    }

    private static void Optimizers(Device device)
    {
        foreach (var make in new Func<IEnumerable<Tensor>, Optimizer>[]
        {
            p => new Sgd(p, 0.1f),
            p => new Sgd(p, 0.05f, momentum: 0.9f),
            p => new Adam(p, 0.1f),
        })
        {
            using var x = Tensor.From([5f, -3f, 2f, 7f, -1f, 4f, 0.5f, -6f, 9f], device, requiresGrad: true);
            using var optimizer = make([x]);
            for (int i = 0; i < 300; i++)
            {
                using var scope = new TensorScope();
                optimizer.ZeroGrad();
                (x - 1f).Square().Sum().Backward();
                optimizer.Step();
            }

            AssertClose(Enumerable.Repeat(1f, 9).ToArray(), x.ToArray(), 1e-2f, optimizer.GetType().Name);
        }
    }

    private static void SaveLoad(Device device)
    {
        string path = Path.GetTempFileName();
        try
        {
            using var a = new Sequential { new Linear(3, 5, device: device), new Tanh(), new Linear(5, 2, device: device) };
            using var b = new Sequential { new Linear(3, 5, device: device), new Tanh(), new Linear(5, 2, device: device) };
            a.Save(path);
            b.Load(path);
            foreach (var (pa, pb) in a.Parameters().Zip(b.Parameters()))
            {
                AssertClose(pa.ToArray(), pb.ToArray(), 0, "loaded parameter");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void Scope(Device device)
    {
        Tensor inner;
        using (var scope = new TensorScope())
        {
            inner = Tensor.Ones([4], device) * 2f;
            var kept = scope.Keep(Tensor.Ones([4], device));
            kept.Dispose();
        }

        try
        {
            inner.ToArray();
            throw new Exception("tensor created in a disposed scope is still usable");
        }
        catch (ObjectDisposedException)
        {
        }
    }

    /// <summary>Compares autograd's gradient of f at a random point with central finite differences.</summary>
    internal static void GradCheck(Device device, int[] shape, Func<Tensor, Tensor> f, bool avoidZero = false, float tolerance = 2e-2f, float scale = 1f)
    {
        var random = new Random(shape.Sum() * 31 + shape.Length);
        var x0 = RandomArray(random, shape.Aggregate(1, (a, b) => a * b), scale);
        if (avoidZero)
        {
            for (int i = 0; i < x0.Length; i++)
            {
                if (MathF.Abs(x0[i]) < 0.1f)
                {
                    x0[i] += 0.3f;
                }
            }
        }

        using var x = Tensor.From(x0, shape, device, requiresGrad: true);
        f(x).Backward();
        var analytic = x.Grad!.ToArray();

        const float h = 1e-2f;
        var numeric = new float[x0.Length];
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < x0.Length; i++)
            {
                var plus = (float[])x0.Clone();
                var minus = (float[])x0.Clone();
                plus[i] += h;
                minus[i] -= h;
                using var tp = Tensor.From(plus, shape, device);
                using var tm = Tensor.From(minus, shape, device);
                numeric[i] = (f(tp).Item() - f(tm).Item()) / (2 * h);
            }
        }

        AssertClose(numeric, analytic, tolerance, "gradient");
    }

    internal static float[] RandomArray(Random random, int n, float scale = 1f)
    {
        var values = new float[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (random.NextSingle() * 2f - 1f) * scale;
        }

        return values;
    }

    internal static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            float allowed = tolerance * Math.Max(1f, MathF.Abs(expected[i]));
            if (!(MathF.Abs(expected[i] - actual[i]) <= allowed))
            {
                throw new Exception($"{what}: element {i} is {actual[i]}, expected {expected[i]}");
            }
        }
    }

    internal static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new Exception(message);
        }
    }
}
