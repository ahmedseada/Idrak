// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Abstraction.Devices;

namespace Idrak.Benches;

/// <summary>
/// The host's cost a launch: thousands of calls on a few elements queued without a sync, the time before the sync divided
/// by the calls (what a step of many small kernels pays per launch, on the thread that issues them), at each layer: the
/// backend's operation on storages that exist (the kernel's dispatch and the driver's launch), the tensor operation
/// (an allocation from the pool, the scope, telemetry), the tensor operation with its gradient recorded, a binary
/// operation, an affine one, and a small product. The time with the sync says whether the device kept up.
/// </summary>
internal static class LaunchBench
{
    private const string Section = "launch";

    public static void Run(Device device, Settings settings, Report report)
    {
        const int Elements = 32;
        int calls = Math.Max(1000, settings.Calls * 250);
        var backend = device.Backend;
        var values = Enumerable.Range(0, Elements).Select(i => (i % 7) * 0.1f - 0.3f).ToArray();
        using var x = Tensor.From(values, device);
        using var y = Tensor.From(values, device);
        using var z = Tensor.From(values, device);
        using var xg = Tensor.From(values, device, requiresGrad: true);
        using var a = Tensor.From([.. Enumerable.Range(0, Elements * Elements).Select(i => (i % 11) * 0.01f)], [Elements, Elements], device);
        using var b = Tensor.From([.. Enumerable.Range(0, Elements * Elements).Select(i => (i % 13) * 0.01f)], [Elements, Elements], device);
        Console.WriteLine($"   {calls} calls on {Elements} elements a round, {settings.Repeats} rounds; µs a call before the sync (the host's cost), and with it");
        (string Name, Action Call)[] cases =
        [
            ("backend unary on existing storages", () => backend.Unary(UnaryOp.Exp, x.Storage, y.Storage, Elements)),
            ("backend binary on existing storages", () => backend.Binary(BinaryOp.Add, x.Storage, y.Storage, z.Storage, Elements)),
            ("tensor unary (exp)", () => { using var r = x.Exp(); }),
            ("tensor unary, gradient recorded", () => { using var r = xg.Exp(); }),
            ("tensor binary (add)", () => { using var r = x + y; }),
            ("tensor affine (x * 2 + 1)", () => { using var r = x * 2f + 1f; }),
            ("tensor matmul 32 x 32", () => { using var r = a.MatMul(b); }),
        ];
        foreach (var (name, call) in cases)
        {
            double bestQueued = double.MaxValue, bestSynced = double.MaxValue;
            for (int round = 0; round <= settings.Repeats; round++)                  // the first round warms (kernels loaded, the pool filled)
            {
                using var scope = new TensorScope();
                backend.Synchronize();
                var clock = Stopwatch.StartNew();
                for (int i = 0; i < calls; i++)
                {
                    call();
                }

                double queued = clock.Elapsed.TotalMilliseconds * 1000 / calls;
                backend.Synchronize();
                double synced = clock.Elapsed.TotalMilliseconds * 1000 / calls;
                if (round > 0)
                {
                    bestQueued = Math.Min(bestQueued, queued);
                    bestSynced = Math.Min(bestSynced, synced);
                }
            }

            Console.WriteLine(Bench.Line(name, ("µs a call queued", Bench.F(bestQueued, "0.00")), ("µs with the sync", Bench.F(bestSynced, "0.00"))));
            report.Add(Section, name, device, "queued", bestQueued, "µs");
            report.Add(Section, name, device, "synced", bestSynced, "µs");
        }
    }
}
