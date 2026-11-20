// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text;
using Idrak.Adad.Abstractions;
using Idrak.Adad.Devices.Cuda;
using Idrak.Adad.Measure;
using Idrak.Adad.Types;

namespace Idrak.Adad.Host;

// adad: the entry point Adad runs directly from (its name settled in N2). For now:
//   adad kernels [--models LIST] [--m LIST] [--type NAME] [--samples N] [--out DIR]
//       the kernel bench of the prompt product for a super-block type: each product of each model at each m, every lane
//       (control, A, B) at its own measured shape and split, warm and cold, each kernel of the lane on its own, and the
//       measured choice; writes DIR/kernels.txt, DIR/choices.txt (the store), DIR/measured.txt (every measurement),
//       DIR/compiled.txt (each kernel's registers and spills).
//   adad profile --choices FILE [--models LIST] [--m LIST] [--type NAME] [--lanes all|chosen]
//       runs each lane's stored variant once a shape, nothing measured (for a kernel profiler: one launch a kernel a shape).
internal static class Program
{
    private sealed record Product(string Name, int K, int N);

    private sealed record Model(string Name, Product[] Products);

    // The models' products (y = x · Wᵀ, W n rows of k): from their config.json (hidden, intermediate, heads × head_dim,
    // key/value heads × head_dim).
    private static readonly Model[] Models =
    [
        new("qwen3-1.7b", Products(hidden: 2048, intermediate: 6144, q: 16 * 128, kv: 8 * 128)),
        new("llama-3.2-3b", Products(hidden: 3072, intermediate: 8192, q: 24 * 128, kv: 8 * 128)),
    ];

    private static Product[] Products(int hidden, int intermediate, int q, int kv) =>
    [
        new("q", hidden, q), new("k", hidden, kv), new("v", hidden, kv), new("o", q, hidden),
        new("gate", hidden, intermediate), new("up", hidden, intermediate), new("down", intermediate, hidden),
    ];

    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            Console.WriteLine("adad kernels [--models qwen3-1.7b,llama-3.2-3b] [--m 128,512,2048,4096] [--type q4_k] [--samples 9] [--out DIR]");
            Console.WriteLine("adad profile --choices FILE [--models ...] [--m ...] [--type q4_k] [--lanes all|chosen] [--repeat N]");
            Console.WriteLine("adad ptx [--type q4_k] [--out DIR]   every lane's modules as PTX files (for ptxas and cuobjdump)");
            return 0;
        }

        var options = Options(args.AsSpan(1));
        string Opt(string name, string fallback) => options.TryGetValue(name, out var v) ? v : fallback;
        var models = Opt("models", "qwen3-1.7b,llama-3.2-3b").Split(',').Select(n => Models.Single(m => m.Name == n)).ToArray();
        int[] ms = [.. Opt("m", "128,512,2048,4096").Split(',').Select(v => int.Parse(v, CultureInfo.InvariantCulture))];
        var type = SuperBlocks.ByName[Opt("type", "q4_k")];
        using var device = CudaDevice.TryOpen(0, out string reason) ?? throw new InvalidOperationException(reason);
        return args[0] switch
        {
            "kernels" => Kernels(device, type, models, ms, int.Parse(Opt("samples", "9"), CultureInfo.InvariantCulture), Opt("out", Path.Combine("test-output", "adad", "kernels"))),
            "profile" => Profile(device, type, models, ms, Opt("choices", ""), Opt("lanes", "all"), int.Parse(Opt("repeat", "1"), CultureInfo.InvariantCulture)),
            "ptx" => Ptx(device, type, Opt("out", Path.Combine("test-output", "adad", "ptx"))),
            _ => throw new ArgumentException($"unknown command {args[0]}"),
        };
    }

    private static Dictionary<string, string> Options(ReadOnlySpan<string> args)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            options[args[i].TrimStart('-')] = args[i + 1];
        }

        return options;
    }

    private static string Header(CudaDevice device) => string.Create(CultureInfo.InvariantCulture,
        $"device: {device.Name}; compute {device.ComputeMajor}.{device.ComputeMinor}; {device.Multiprocessors} SMs; {device.TotalMemory >> 20} MiB; L2 {device.L2Bytes >> 10} KiB; shared a block {device.SharedPerBlockOptin} B\n"
        + $"driver: CUDA {CudaDevice.DriverVersion}\n"
        + $"adad: {typeof(SuperProduct).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion} ({File.GetLastWriteTimeUtc(typeof(SuperProduct).Assembly.Location):yyyy-MM-dd HH:mm:ss} UTC)\n");

    // Random valid blocks of the description: random bytes, d and dmin random small finite halves (timing data).
    internal static byte[] RandomBlocks(SuperBlock s, int rows, int k, int seed)
    {
        var random = new Random(seed);
        var bytes = new byte[(long)rows * (k / s.Values) * s.Bytes];
        random.NextBytes(bytes);
        for (long at = 0; at < bytes.Length; at += s.Bytes)
        {
            BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.ScaleOffset), (Half)(random.NextSingle() * 0.002f));
            if (s.MinOffset >= 0)
            {
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.MinOffset), (Half)(random.NextSingle() * 0.002f));
            }
        }

        return bytes;
    }

    internal static float[] RandomRows(int rows, int k, int seed)
    {
        var random = new Random(seed);
        var x = new float[(long)rows * k];
        for (long i = 0; i < x.Length; i++)
        {
            x[i] = (float)(random.NextDouble() + random.NextDouble() + random.NextDouble() + random.NextDouble() - 2.0);
        }

        return x;
    }

    private static int Kernels(CudaDevice device, SuperBlock type, Model[] models, int[] ms, int samples, string outDir)
    {
        Directory.CreateDirectory(outDir);
        var tuner = new Tuner(device);
        var product = new SuperProduct(device, tuner);
        var report = new StringBuilder();
        var choiceTable = new StringBuilder();
        void Say(string line)
        {
            Console.WriteLine(line);
            report.AppendLine(line);
        }

        Say("adad kernels: the prompt product y = x · Wᵀ for a super-block weight, every lane at its measured shape and split");
        Say(Header(device).TrimEnd());
        Say(string.Create(CultureInfo.InvariantCulture, $"type: {string.Join(", ", SuperBlocks.ByName.Where(e => e.Value == type).Select(e => e.Key))} (description id {type.Id}, {type.Code})"));
        Say(string.Create(CultureInfo.InvariantCulture, $"samples a timing: {samples} (warm: back-to-back runs, about 1 ms, at most 256, a sample; cold: one run after a read of twice the L2 cache); the lanes in turns; each kernel alone, warm and cold"));
        var clock = Stopwatch.StartNew();
        var lanes = new SuperLane?[SuperProduct.LaneNames.Length];
        for (int l = 0; l < lanes.Length; l++)
        {
            lanes[l] = product.Lane(type, l, out string why);
            Say($"lane {SuperProduct.LaneNames[l]}: {(lanes[l] is null ? "no candidate: " + why : "ready")}");
        }

        File.WriteAllLines(Path.Combine(outDir, "compiled.txt"), lanes.Where(l => l is not null).SelectMany(l => l!.Compiled));
        Say(LaunchCost(device, lanes[0]!));
        choiceTable.AppendLine("| model | product | k | n | m | control µs | A µs | B µs | measured choice | lane variants (shape, splits) |");
        choiceTable.AppendLine("|---|---|---|---|---|---|---|---|---|---|");
        foreach (var model in models)
        {
            foreach (int m in ms)
            {
                Say("");
                Say(string.Create(CultureInfo.InvariantCulture, $"## {model.Name}, m = {m}"));
                Say("| product | k | n | lane | shape, splits | warm µs (min-max) | spread | TFLOP/s | cold µs | kernels (warm / cold µs; GB/s or TFLOP/s warm / cold) |");
                Say("|---|---|---|---|---|---|---|---|---|---|");
                var x = device.Allocate((long)m * model.Products.Max(p => p.K) * 4);
                var y = device.Allocate((long)m * model.Products.Max(p => p.N) * 4);
                try
                {
                    foreach (var p in model.Products)
                    {
                        var w = device.Allocate((long)p.N * (p.K / type.Values) * type.Bytes);
                        try
                        {
                            device.Upload(x, RandomRows(m, p.K, 1000 + p.K).AsSpan());
                            device.Upload(w, RandomBlocks(type, p.N, p.K, p.N * 31 + p.K).AsSpan());
                            var op = new ProductOperands(x.Pointer, w.Pointer, y.Pointer, m, p.N, p.K);
                            product.Run(type, op);                         // the measured choice (lanes' variants, then lanes)
                            int chosen = product.ChosenLane(type, m, p.N, p.K);
                            var rows = BenchShape(device, tuner, product, type, lanes, op, samples);
                            double flops = 2.0 * m * p.N * p.K;
                            var cells = new string[lanes.Length];
                            var variants = new List<string>();
                            foreach (var r in rows)
                            {
                                Say(string.Create(CultureInfo.InvariantCulture,
                                    $"| {p.Name} | {p.K} | {p.N} | {SuperProduct.LaneNames[r.Lane]}{(r.Lane == chosen ? " (chosen)" : "")} | {r.Shape} | {r.Warm * 1000:F1} ({r.WarmMin * 1000:F1}-{r.WarmMax * 1000:F1}) | {(r.WarmMax - r.WarmMin) / r.Warm * 100:F1}% | {flops / (r.Warm * 1e-3) / 1e12:F1} | {r.Cold * 1000:F1} | {r.Kernels} |"));
                                cells[r.Lane] = string.Create(CultureInfo.InvariantCulture, $"{r.Warm * 1000:F1}");
                                variants.Add($"{SuperProduct.LaneNames[r.Lane]} {r.Shape}");
                            }

                            choiceTable.AppendLine(string.Create(CultureInfo.InvariantCulture,
                                $"| {model.Name} | {p.Name} | {p.K} | {p.N} | {m} | {cells[0] ?? "-"} | {cells[1] ?? "-"} | {cells[2] ?? "-"} | {(chosen >= 0 ? SuperProduct.LaneNames[chosen] : "-")} | {string.Join("; ", variants)} |"));
                        }
                        finally
                        {
                            device.Release(w);
                        }
                    }
                }
                finally
                {
                    device.Release(x);
                    device.Release(y);
                }

                Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"[{clock.Elapsed.TotalSeconds:F0} s] {model.Name} m = {m} done"));
            }
        }

        Say("");
        Say("## The measured choice per shape (warm µs at each lane's own measured shape and split)");
        Say(choiceTable.ToString().TrimEnd());
        Say("");
        Say(string.Create(CultureInfo.InvariantCulture, $"device memory: the pool held {device.PeakPoolBytes >> 20} MiB at most ({device.DriverAllocations} driver allocations); launches {device.Launches}; {clock.Elapsed.TotalSeconds:F0} s"));
        File.WriteAllText(Path.Combine(outDir, "kernels.txt"), report.ToString());
        File.WriteAllText(Path.Combine(outDir, "choices.txt"), Header(device) + product.SaveChoices());
        File.WriteAllLines(Path.Combine(outDir, "measured.txt"), tuner.Log.Select(Tuner.Describe));
        return 0;
    }

    private sealed record Row(int Lane, string Shape, double Warm, double WarmMin, double WarmMax, double Cold, string Kernels);

    // The host's time a launch: the splits' sum over no values, 20 batches of 500 launches, each batch timed by the host's
    // clock from its first launch to its last enqueued (the device drained between batches, so the queue never fills).
    private static string LaunchCost(CudaDevice device, SuperLane lane)
    {
        var times = new double[20];
        for (int batch = 0; batch < times.Length; batch++)
        {
            device.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 500; i++)
            {
                lane.LaunchEmpty();
            }

            times[batch] = Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000 / 500;
        }

        device.Synchronize();
        Array.Sort(times);
        return string.Create(CultureInfo.InvariantCulture, $"host: {times[times.Length / 2]:F2} µs a launch (median of 20 batches of 500 empty launches; {times[0]:F2}-{times[^1]:F2})");
    }

    // The bench's warm sample: back-to-back runs adding up to about 1 ms (at most 256), so a short product's sample is not
    // a handful of runs (the measured choice keeps the proof's 0.2 ms, at most 16).
    private static int SampleRepeats(float onceMs) => Math.Clamp((int)MathF.Ceiling(1f / Math.Max(onceMs, 1e-3f)), 1, 256);

    // Each lane at its variant: warmed together, then timed in turns (so all see the same clocks), warm then cold, then each
    // of its kernels warm on its own.
    private static List<Row> BenchShape(CudaDevice device, Tuner tuner, SuperProduct product, SuperBlock type, SuperLane?[] lanes, ProductOperands op, int samples)
    {
        var present = Enumerable.Range(0, lanes.Length).Where(l => lanes[l] is not null).ToArray();
        var variant = present.ToDictionary(l => l, l => product.VariantOf(type, l, op));
        Action<int> run = l => lanes[l]!.Run(variant[l], op);
        TuneTiming.Warm(() =>
        {
            device.TimeStart();
            foreach (int l in present)
            {
                run(l);
            }

            return device.TimeEnd();
        });
        var repeats = present.ToDictionary(l => l, l => SampleRepeats(tuner.TimeRuns(l, 1, run, cold: false)));
        var warm = present.ToDictionary(l => l, _ => new List<float>());
        var cold = present.ToDictionary(l => l, _ => new List<float>());
        for (int s = 0; s < samples; s++)
        {
            foreach (int l in s % 2 == 0 ? present : present.Reverse())
            {
                warm[l].Add(tuner.TimeRuns(l, repeats[l], run, cold: false) / repeats[l]);
            }
        }

        for (int s = 0; s < samples; s++)
        {
            foreach (int l in s % 2 == 0 ? present : present.Reverse())
            {
                cold[l].Add(tuner.TimeRuns(l, 1, run, cold: true));
            }
        }

        var rows = new List<Row>();
        foreach (int l in present)
        {
            var lane = lanes[l]!;
            int v = variant[l];
            var (cfg, splits) = lane.Describe(v);
            var scratch = lane.Rent(v, op);
            var kernels = new List<string>();
            try
            {
                for (int piece = 0; piece < lane.PieceCount(v); piece++)
                {
                    for (int before = 0; before < piece; before++)
                    {
                        lane.Launch(before, v, op, scratch);               // its inputs written
                    }

                    var info = lane.Piece(piece, v, op);
                    int p = piece;
                    Action<int> one = _ => lane.Launch(p, v, op, scratch);
                    int r = SampleRepeats(tuner.TimeRuns(0, 1, one, cold: false));
                    var times = new float[samples];
                    var colds = new float[samples];
                    for (int s = 0; s < samples; s++)
                    {
                        times[s] = tuner.TimeRuns(0, r, one, cold: false) / r;
                        colds[s] = tuner.TimeRuns(0, 1, one, cold: true);
                    }

                    float t = TuneTiming.Median(times), tc = TuneTiming.Median(colds);
                    string Rate(float ms) => info.Flops > 0 ? string.Create(CultureInfo.InvariantCulture, $"{info.Flops / (ms * 1e-3) / 1e12:F1} TFLOP/s")
                        : string.Create(CultureInfo.InvariantCulture, $"{info.Bytes / (ms * 1e-3) / 1e9:F0} GB/s");
                    kernels.Add(string.Create(CultureInfo.InvariantCulture, $"{info.Kernel} {t * 1000:F1} / {tc * 1000:F1} ({Rate(t)} / {Rate(tc)})"));
                }
            }
            finally
            {
                device.Synchronize();
                lane.Return(scratch);
            }

            var w = warm[l].ToArray();
            float median = TuneTiming.Median(w);
            var c = cold[l].ToArray();
            rows.Add(new Row(l, string.Create(CultureInfo.InvariantCulture, $"{cfg}, {splits}"), median, w.Min(), w.Max(), TuneTiming.Median(c), string.Join("; ", kernels)));
        }

        return rows;
    }

    private static int Ptx(CudaDevice device, SuperBlock type, string outDir)
    {
        Directory.CreateDirectory(outDir);
        var product = new SuperProduct(device, new Tuner(device)) { Measuring = false };
        int written = 0;
        for (int l = 0; l < SuperProduct.LaneNames.Length; l++)
        {
            if (product.Lane(type, l, out _) is not { } lane)
            {
                continue;
            }

            foreach (var (entry, module) in lane.Modules())
            {
                File.WriteAllText(Path.Combine(outDir, entry + ".ptx"), module);
                written++;
            }
        }

        Console.WriteLine($"{written} modules written to {outDir}");
        return 0;
    }

    private static int Profile(CudaDevice device, SuperBlock type, Model[] models, int[] ms, string choices, string which, int repeat)
    {
        var tuner = new Tuner(device);
        var product = new SuperProduct(device, tuner) { Measuring = false };
        int read = File.Exists(choices) ? product.LoadChoices(File.ReadAllText(choices)) : 0;
        Console.WriteLine(Header(device).TrimEnd());
        Console.WriteLine($"choices read: {read} from {choices}");
        var lanes = new SuperLane?[SuperProduct.LaneNames.Length];
        for (int l = 0; l < lanes.Length; l++)
        {
            lanes[l] = product.Lane(type, l, out _);
        }

        int order = 0;
        foreach (var model in models)
        {
            foreach (int m in ms)
            {
                var x = device.Allocate((long)m * model.Products.Max(p => p.K) * 4);
                var y = device.Allocate((long)m * model.Products.Max(p => p.N) * 4);
                foreach (var p in model.Products.DistinctBy(p => (p.K, p.N)))
                {
                    var w = device.Allocate((long)p.N * (p.K / type.Values) * type.Bytes);
                    device.Upload(x, RandomRows(m, p.K, 1000 + p.K).AsSpan());
                    device.Upload(w, RandomBlocks(type, p.N, p.K, p.N * 31 + p.K).AsSpan());
                    var op = new ProductOperands(x.Pointer, w.Pointer, y.Pointer, m, p.N, p.K);
                    int chosen = product.ChosenLane(type, m, p.N, p.K);
                    for (int l = 0; l < lanes.Length; l++)
                    {
                        if (lanes[l] is null || which == "chosen" && l != chosen)
                        {
                            continue;
                        }

                        int v = product.VariantOf(type, l, op);
                        var (cfg, splits) = lanes[l]!.Describe(v);
                        for (int piece = 0; piece < lanes[l]!.PieceCount(v); piece++)
                        {
                            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                                $"launch {order++}: {model.Name} {p.Name} m={m} k={p.K} n={p.N} lane {SuperProduct.LaneNames[l]}{(l == chosen ? " (chosen)" : "")} {cfg}, {splits}: {lanes[l]!.Piece(piece, v, op).Kernel}"));
                        }

                        for (int r = 0; r < repeat; r++)
                        {
                            lanes[l]!.Run(v, op);
                        }

                        device.Synchronize();
                    }

                    device.Release(w);
                }

                device.Release(x);
                device.Release(y);
            }
        }

        return 0;
    }
}
