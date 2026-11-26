// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

// Plan 9 / plan 10 phase 1b: operations as data. Every operation of the device contract goes through one dispatcher to
// the device's own kernel.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] OperationGroup =
    [
        ("operations: every compute method of Backend is an operation (a NameKernel, virtual or forwarding to its capability interface, the dispatching Name with the same parameters, a descriptor with the matching kernel delegate); the rest is device plumbing (python3 tools/operations/generate.py)", OperationsMatchBackend),
        ("operations: Kernels.Chain names the kernel each operation runs (the device's own, composed, host fallback or none): the CPU runs its own, a device with memory and copies only falls back", ChainNamesKernels),
        ("operations: a trace counts the calls of each operation and the host fallbacks by operation, read from the dispatcher; it ends when disposed", TraceCountsCalls),
        ("operations: a failing device kernel throws a DeviceException, reported to telemetry once (a console logger prints it)", DeviceFailureReported),
        ("operations: on a device that builds its kernels when first used, a refused kernel (KernelUnavailableException) falls back from the first call: the host (the CPU's bits) or the caller's path, told to the device once; other device errors still reach the caller", RefusedKernelFallsBack),
        ("operations: NormStats keeps its precision for one-element groups and large values with a small spread (variance from two passes), also over a group of many chunks (merged by count, mean and sum of squares)", NormStatsPrecision),
    ];

    // Backend's public virtual members that are not operations: memory, copies and the questions a caller asks before
    // choosing a path (plan 9: Backend keeps these). Graph capture, profiling, FP8, peer copies and the memory report are
    // capability interfaces (plan 33), not virtual members.
    internal static readonly string[] DevicePlumbing =
    [
        "CreateHostStaging", "PrefersComposedAttention", "Copy", "Copy2D",
        "ResetPeakMemoryUsage", "LiveStorages", "UploadRange", "Dispose", "StoredMatMulManyAvailable", "StoredEpilogueFused", "MergesAttention",
    ];

    internal static void OperationsMatchBackend(Device device)
    {
        _ = device;
        const BindingFlags Declared = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly;
        var methods = typeof(Backend).GetMethods(Declared).Where(m => !m.IsSpecialName).ToList();
        var kernels = methods.Where(m => !m.IsAbstract && m.Name.EndsWith("Kernel", StringComparison.Ordinal)).ToList();
        var other = methods.Where(m => m.IsVirtual && !m.IsAbstract && !kernels.Contains(m) && !DevicePlumbing.Contains(m.Name)).Select(m => m.Name).Distinct().ToList();
        Check(other.Count == 0, $"public virtual methods of Backend that are neither operations (NameKernel) nor device plumbing: {string.Join(", ", other)}");
        Check(kernels.Count == OperationIndex.Count && Ops.All.Count == OperationIndex.Count,
            $"{kernels.Count} NameKernel methods, {Ops.All.Count} descriptors, {OperationIndex.Count} slots: run python3 tools/operations/generate.py");

        // Matched by name, not by place: Backend's operations are declared over several partial files (Backend.cs, then
        // Backend.Routing.cs: the generator's order), whose members reflection may list in another file order. Overloads
        // share a file, so their numbers follow their order there.
        static Type[] Parameters(MethodInfo m) => [.. m.GetParameters().Select(p => p.ParameterType)];
        var seen = new Dictionary<string, int>(StringComparer.Ordinal);
        var matched = new HashSet<int>();
        for (int i = 0; i < kernels.Count; i++)
        {
            var kernel = kernels[i];
            string name = kernel.Name[..^"Kernel".Length];
            int overload = seen[name] = seen.GetValueOrDefault(name) + 1;
            string expected = overload == 1 ? name : name + overload;
            var operation = Ops.All.FirstOrDefault(o => o.Name == expected);
            Check(operation is not null && Ops.All[operation.Index] == operation && matched.Add(operation.Index), $"Backend declares {kernel.Name}: no descriptor {expected} (or one met twice)");

            var dispatch = methods.SingleOrDefault(m => m.Name == name && !m.IsVirtual && Parameters(m).SequenceEqual(Parameters(kernel)));
            Check(dispatch is not null && dispatch.ReturnType == kernel.ReturnType, $"no dispatching {name} with the parameters of {kernel.Name}");

            var invoke = operation!.KernelType.GetMethod("Invoke")!;
            Check(invoke.ReturnType == kernel.ReturnType && Parameters(invoke).SequenceEqual([typeof(Backend), .. Parameters(kernel)]),
                $"the delegate of {operation.Name} does not match {kernel.Name}");
        }
    }

    internal static void ChainNamesKernels(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ChainNamesKernels)))
        {
            return;
        }

        var cpu = Kernels.Chain(Device.Cpu.Backend);
        Check(cpu.Count == Ops.All.Count && cpu.Select((c, i) => c.Operation.Index == i).All(x => x), "one choice per operation, in slot order");
        var notOwn = cpu.Where(c => c.Source != KernelSource.Device).ToList();
        Check(notOwn.All(c => c.Source is KernelSource.Composed or KernelSource.None) && cpu.Count(c => c.Source == KernelSource.Device) > 80,
            $"the CPU runs its own kernels (or composes, or has none): {string.Join(", ", notOwn.Select(c => $"{c.Operation} {c.Source}"))}");
        Check(cpu[Ops.Softmax.Index].Source == KernelSource.Device && Ops.Softmax.Fallback == KernelSource.Host
              && Ops.MatMulMany.Fallback == KernelSource.Composed, "fallbacks of Softmax and MatMulMany");

        var minimal = MinimalBackend.Instance;
        Check(Kernels.Chain(minimal).All(c => c.Source == c.Operation.Fallback), "a device with memory and copies only takes each operation's fallback");
        Check(Ops.All.Select(o => o.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() == Ops.All.Count
              && Ops.All.Single(o => o.Name.Equals("softmax", StringComparison.OrdinalIgnoreCase)) == Ops.Softmax
              && Ops.All.Single(o => o.Name.Equals("MaxPoolBackward2", StringComparison.OrdinalIgnoreCase)) == Ops.MaxPoolBackward2,
            "each operation states its own name, no two alike (a text matched by walking Ops.All where it comes in)");
    }

    internal static void NormStatsPrecision(Device device)
    {
        var b = device.Backend;
        const float eps = 1e-5f;
        float[] values = [2.7182817f, -1.4142135f];                               // two groups of one value (squares inexact in float)
        var x = b.Allocate(values.Length, zeroed: false);
        var stats = new[] { b.Allocate(2, zeroed: true), b.Allocate(2, zeroed: true), b.Allocate(2, zeroed: true) };
        try
        {
            b.Upload(values, x);
            b.NormStats(x, stats[0], stats[1], stats[2], 1, 2, 1, eps);              // [outer 1, groups 2, inner 1]: one value each
            var invStd = new float[2];
            b.Download(stats[2], invStd);
            float expected = 1f / MathF.Sqrt(eps);
            Check(Math.Abs(invStd[0] - expected) < 1e-3f * expected && Math.Abs(invStd[1] - expected) < 1e-3f * expected,
                $"one-element groups: invStd {invStd[0]}, {invStd[1]}, expected {expected}");

            // 64 values: long enough for the vectorized sums; 3 · 8,192 + 5: a group a device may split across blocks (its last
            // chunk nearly empty), whose chunks' statistics are merged.
            foreach (int run in new[] { 64, 3 * 8192 + 5 })
            {
                var tail = b.Allocate(run, zeroed: false);
                var near = new float[run];
                for (int i = 0; i < run; i++)
                {
                    near[i] = 1000f + 0.25f * MathF.Sin(i * 1.7f);                     // 1000 ± 0.25, irregular
                }

                double mu = near.Average(v => (double)v), exact = near.Average(v => (v - mu) * (v - mu));

                try
                {
                    b.Upload(near, tail);
                    b.NormStats(tail, stats[0], stats[1], stats[2], 1, 1, run, eps);
                    float[] mean = new float[1], variance = new float[1];
                    b.Download(stats[0], mean);
                    b.Download(stats[1], variance);
                    Check(Math.Abs(mean[0] - mu) < 1e-5 * mu && Math.Abs(variance[0] - exact) < 1e-3 * exact,
                        $"1000 ± 0.25, {run} values: mean {mean[0]} (expected {mu}), variance {variance[0]} (expected {exact})");
                }
                finally
                {
                    tail.Release();
                }
            }
        }
        finally
        {
            x.Release();
            foreach (var st in stats)
            {
                st.Release();
            }
        }
    }

    internal static void TraceCountsCalls(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(TraceCountsCalls)))
        {
            return;
        }

        var minimal = MinimalBackend.Instance;
        var y = minimal.Allocate(8, zeroed: true);
        try
        {
            long before = Kernels.HostCalls(minimal);
            using (var trace = Kernels.Trace(minimal))
            {
                minimal.Fill(y, 8, 1f);
                minimal.Fill(y, 8, 2f);
                minimal.Unary(UnaryOp.Exp, y, y, 8);
                minimal.AddDropout(y, y, y, 8, 0f, 1);                       // composed: Dropout and Axpy, each through the host
                Check(trace.Calls(Ops.Fill) == 2 && trace.Calls(Ops.Unary) == 1 && trace.Calls(Ops.AddDropout) == 1 && trace.Calls(Ops.Dropout) == 1
                      && trace.Calls(Ops.Softmax) == 0, $"calls: Fill {trace.Calls(Ops.Fill)}, Unary {trace.Calls(Ops.Unary)}, Dropout {trace.Calls(Ops.Dropout)}");
                var host = trace.HostCallsByOperation;
                Check(host.GetValueOrDefault("Fill") == 2 && host.GetValueOrDefault("Unary") == 1 && host.GetValueOrDefault("Dropout") == 1
                      && host.GetValueOrDefault("Axpy") == 1 && !host.ContainsKey("AddDropout") && trace.HostCalls == 5,
                    $"host fallbacks: {string.Join(", ", host.Select(h => $"{h.Key} x{h.Value}"))}");
                Check(Kernels.HostCalls(minimal) - before == 5, "the device's host fallbacks, read from the dispatcher");

                bool refused = false;
                try
                {
                    Kernels.Trace(minimal).Dispose();
                }
                catch (InvalidOperationException)
                {
                    refused = true;
                }

                Check(refused, "a second trace of one device is refused");
            }

            using var after = Kernels.Trace(minimal);
            after.Dispose();
            minimal.Fill(y, 8, 3f);
            Check(after.Calls(Ops.Fill) == 0 && after.HostCalls == 0, "a disposed trace counts nothing");
        }
        finally
        {
            minimal.Return(y);
        }
    }

    // For the tests that list a device's host fallbacks by operation: counts them into `counts` from now on (null stops and
    // adds what the trace counted).
    internal static readonly ConcurrentDictionary<Backend, (KernelTrace Trace, ConcurrentDictionary<string, long> Counts)> HostCallCounts = new();

    internal static void CountHostCalls(Backend backend, ConcurrentDictionary<string, long>? counts)
    {
        if (HostCallCounts.TryRemove(backend, out var running))
        {
            running.Trace.Dispose();
            foreach (var (operation, calls) in running.Trace.HostCallsByOperation)
            {
                running.Counts.AddOrUpdate(operation, calls, (_, n) => n + calls);
            }
        }

        if (counts is not null)
        {
            HostCallCounts[backend] = (Kernels.Trace(backend), counts);
        }
    }

    // A million tiny operations (a 16-float fill) through the dispatcher and straight to the device's kernel, on the CPU:
    // alternating rounds, the best of each.
    internal static int BenchDispatch()
    {
        var backend = Device.Cpu.Backend;
        var y = backend.Allocate(16, zeroed: true);
        const int Calls = 1_000_000;
        double direct = double.MaxValue, dispatched = double.MaxValue;
        for (int round = 0; round < 40; round++)
        {
            var sw = Stopwatch.StartNew();
            DirectFills(backend, y, Calls);
            direct = Math.Min(direct, sw.Elapsed.TotalMilliseconds);
            sw.Restart();
            DispatchedFills(backend, y, Calls);
            dispatched = Math.Min(dispatched, sw.Elapsed.TotalMilliseconds);
        }

        backend.Return(y);
        Console.WriteLine($"{Calls:N0} fills of 16 floats on the CPU (best of 40): device kernel {direct:F2} ms, through the dispatcher {dispatched:F2} ms ({(dispatched / direct - 1) * 100:+0.0;-0.0}%)");
        return 0;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    internal static void DirectFills(Backend backend, Storage y, int calls)
    {
        for (int i = 0; i < calls; i++)
        {
            backend.FillKernel(y, 16, i);
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    internal static void DispatchedFills(Backend backend, Storage y, int calls)
    {
        for (int i = 0; i < calls; i++)
        {
            backend.Fill(y, 16, i);
        }
    }
}
