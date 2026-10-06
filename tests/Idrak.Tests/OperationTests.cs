// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Reflection;
using Idrak.Abstraction.Operations;

// Plan 9 / plan 10 phase 1b: operations as data. Every operation of the device contract goes through one dispatcher:
// the kernel registered for the device kind (Kernels.Register) where its requirement holds, else the device's own.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OperationGroup =
    [
        ("operations: every compute method of Backend is an operation (a virtual NameKernel, the dispatching Name with the same parameters, a descriptor with the matching kernel delegate); the rest is device plumbing (python3 tools/operations/generate.py)", OperationsMatchBackend),
        ("operations: a registered kernel runs instead of the device's own on its device kind where its requirement holds, not elsewhere; removing it restores the device's kernel; a kernel of the wrong delegate is refused", RegisteredKernelRuns),
    ];

    // Backend's public virtual members that are not operations: memory, copies, graph capture, profiling and the
    // questions a caller asks before choosing a path (plan 9: Backend keeps these).
    private static readonly string[] DevicePlumbing =
    [
        "AbortCapture", "BeginCapture", "EndCapture", "DestroyGraph", "ReplayGraph", "CreateHostStaging", "StartProfile", "StopProfile",
        "TensorCoresUnavailable", "ReuseQuantizedOperands", "Float8PaddedK", "SupportsSegmentedAttention", "PrefersPackedMatMul", "Copy", "Copy2D",
    ];

    private static void OperationsMatchBackend(Device device)
    {
        _ = device;
        const BindingFlags Declared = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly;
        var methods = typeof(Backend).GetMethods(Declared).Where(m => !m.IsSpecialName).ToList();
        var kernels = methods.Where(m => m.IsVirtual && !m.IsAbstract && m.Name.EndsWith("Kernel", StringComparison.Ordinal)).ToList();
        var other = methods.Where(m => m.IsVirtual && !m.IsAbstract && !kernels.Contains(m) && !DevicePlumbing.Contains(m.Name)).Select(m => m.Name).Distinct().ToList();
        Check(other.Count == 0, $"public virtual methods of Backend that are neither operations (NameKernel) nor device plumbing: {string.Join(", ", other)}");
        Check(kernels.Count == OperationIndex.Count && Ops.All.Count == OperationIndex.Count,
            $"{kernels.Count} NameKernel methods, {Ops.All.Count} descriptors, {OperationIndex.Count} slots: run python3 tools/operations/generate.py");

        static Type[] Parameters(MethodInfo m) => [.. m.GetParameters().Select(p => p.ParameterType)];
        var seen = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < kernels.Count; i++)
        {
            var kernel = kernels[i];
            string name = kernel.Name[..^"Kernel".Length];
            int overload = seen[name] = seen.GetValueOrDefault(name) + 1;
            var operation = Ops.All[i];
            Check(operation.Index == i && operation.Name == (overload == 1 ? name : name + overload), $"descriptor {i} is {operation.Name}, Backend declares {kernel.Name} there");

            var dispatch = methods.SingleOrDefault(m => m.Name == name && !m.IsVirtual && Parameters(m).SequenceEqual(Parameters(kernel)));
            Check(dispatch is not null && dispatch.ReturnType == kernel.ReturnType, $"no dispatching {name} with the parameters of {kernel.Name}");

            var invoke = operation.KernelType.GetMethod("Invoke")!;
            Check(invoke.ReturnType == kernel.ReturnType && Parameters(invoke).SequenceEqual([typeof(Backend), .. Parameters(kernel)]),
                $"the delegate of {operation.Name} does not match {kernel.Name}");
        }
    }

    private static void RegisteredKernelRuns(Device device)
    {
        var backend = device.Backend;
        var y = backend.Allocate(4, zeroed: true);
        float[] Values()
        {
            var values = new float[4];
            backend.Download(y, values);
            return values;
        }

        try
        {
            int calls = 0;
            OperationKernels.Fill plusOne = (b, s, n, value) =>
            {
                calls++;
                b.FillKernel(s, n, value + 1f);   // the device's own kernel, with another value
            };

            using (Kernels.Register(Ops.Fill, backend.Kind.ToUpperInvariant(), plusOne))
            {
                backend.Fill(y, 4, 2f);
                Check(calls == 1 && Values().All(v => v == 3f), $"registered kernel: {calls} calls, {string.Join(", ", Values())}");

                using (Kernels.Register(Ops.Fill, backend.Kind, plusOne, requirement: _ => false))
                {
                    backend.Fill(y, 4, 5f);   // the later registration does not apply here: the earlier one still does
                    Check(calls == 2 && Values().All(v => v == 6f), $"requirement false: {calls} calls, {string.Join(", ", Values())}");
                }
            }

            using (Kernels.Register(Ops.Fill, "no-such-device", plusOne))
            {
                backend.Fill(y, 4, 7f);
                Check(calls == 2 && Values().All(v => v == 7f), $"another device kind: {calls} calls, {string.Join(", ", Values())}");
            }

            backend.Fill(y, 4, 1f);
            Check(calls == 2 && Values().All(v => v == 1f), $"after removal: {calls} calls, {string.Join(", ", Values())}");

            bool refused = false;
            try
            {
                Kernels.Register(Ops.Fill, backend.Kind, (OperationKernels.Unary)((_, _, _, _, _) => { })).Dispose();
            }
            catch (ArgumentException)
            {
                refused = true;
            }

            Check(refused, "a Unary kernel registered for Fill");
        }
        finally
        {
            backend.Return(y);
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
    private static void DirectFills(Backend backend, Storage y, int calls)
    {
        for (int i = 0; i < calls; i++)
        {
            backend.FillKernel(y, 16, i);
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void DispatchedFills(Backend backend, Storage y, int calls)
    {
        for (int i = 0; i < calls; i++)
        {
            backend.Fill(y, 16, i);
        }
    }
}
