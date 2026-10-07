// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing.Devices;

/// <summary>
/// The device check: the cases run on the CPU with every operation call recorded, then each call runs again on the
/// device under test, through its dispatcher (<c>Backend.Name(...)</c>), on copies of the same values.
/// </summary>
internal static class DeviceConformance
{
    // The recording registers kernels for the CPU's kind: one recording at a time.
    private static readonly Lock Gate = new();

    // What happened to the calls of one operation.
    private sealed class Tally
    {
        public int Compared, Failed, Unsupported, NoReference;
        public readonly HashSet<string> Cases = new(StringComparer.Ordinal);
    }

    /// <summary>Records the calls of the cases <paramref name="options"/> selects; a case the CPU fails is reported and dropped.</summary>
    public static List<RecordedCall> Record(DeviceCheckOptions options, List<ConformanceFailure> failures, out int cases)
    {
        if (OperationCalls.Count != Ops.All.Count)
        {
            throw new InvalidOperationException($"The testing kit knows {OperationCalls.Count} operations and Idrak.Abstraction has {Ops.All.Count}: "
                                                + "use the testing kit of the same version as Idrak.Abstraction.");
        }

        var runs = new List<(DeviceCase Case, int Seed, string Name)>();
        for (int i = 0; i < options.Cases.Count; i++)
        {
            var @case = options.Cases[i];
            if (options.Filter is { } filter && !filter(@case.Name))
            {
                continue;
            }

            int count = @case.Random ? Math.Max(1, options.RandomRuns) : 1;
            for (int run = 0; run < count; run++)
            {
                int seed = HashCode.Combine(options.Seed, i, run) & int.MaxValue;
                runs.Add((@case, seed, @case.Random ? $"{@case.Name} (seed {seed})" : @case.Name));
            }
        }

        cases = runs.Count;
        var recorder = new CallRecorder();
        lock (Gate)
        {
            var handles = OperationCalls.Intercept(Device.Cpu.Backend.Kind, recorder);
            try
            {
                foreach (var (@case, seed, name) in runs)
                {
                    int before = recorder.Calls.Count;
                    using var context = new DeviceCaseContext(seed, options.Large, recorder);
                    try
                    {
                        using (recorder.Start(name))
                        using (new TensorScope())
                        {
                            @case.Run(context);
                        }
                    }
                    catch (Exception ex)
                    {
                        recorder.Truncate(before);
                        failures.Add(new ConformanceFailure($"case {@case.Name}", name, $"the CPU (the library default) fails it: {ex.Message}"));
                    }
                }
            }
            finally
            {
                foreach (var handle in handles)
                {
                    handle.Dispose();
                }
            }
        }

        return [.. recorder.Calls];
    }

    /// <summary>Checks <paramref name="backend"/> against the CPU on every case of <paramref name="options"/>.</summary>
    public static ConformanceReport Check(Backend backend, string subject, DeviceCheckOptions options)
    {
        var failures = new List<ConformanceFailure>();
        var calls = Record(options, failures, out int cases);
        return Replay(backend, subject, calls, options, failures, cases);
    }

    /// <summary>Runs recorded calls on <paramref name="backend"/> and reports by operation.</summary>
    public static ConformanceReport Replay(Backend backend, string subject, IReadOnlyList<RecordedCall> calls, DeviceCheckOptions options,
        List<ConformanceFailure> failures, int cases)
    {
        var tallies = new Tally[Ops.All.Count];
        for (int i = 0; i < tallies.Length; i++)
        {
            tallies[i] = new Tally();
        }

        foreach (var call in calls)
        {
            var tally = tallies[call.Operation.Index];
            tally.Cases.Add(call.Case);
            if (call.Result is false)
            {
                tally.NoReference++;                                     // the CPU has no kernel for these arguments
                continue;
            }

            string? problem = Run(backend, call, options.ToleranceFor(call.Operation), out bool unsupported);
            if (unsupported)
            {
                tally.Unsupported++;
            }
            else if (problem is null)
            {
                tally.Compared++;
            }
            else if (++tally.Failed <= options.MaxFailures)
            {
                failures.Add(new ConformanceFailure(call.Operation.Name, call.Case, problem, OperationCases.ToCase(call, backend.Kind, problem)));
            }
        }

        var chain = Kernels.Chain(backend);
        var entries = new List<CheckEntry>(Ops.All.Count);
        foreach (var operation in Ops.All)
        {
            var tally = tallies[operation.Index];
            var source = chain[operation.Index].Source;
            string kernel = source switch
            {
                KernelSource.Registered => "a registered kernel",
                KernelSource.Device => "the device's own kernel",
                KernelSource.Composed => "composed of other operations",
                KernelSource.Host => "the host fallback",
                _ => "no kernel",
            };

            string seen = $"{tally.Cases.Count} case{(tally.Cases.Count == 1 ? "" : "s")}";
            if (tally.Failed > 0)
            {
                entries.Add(new CheckEntry(operation.Name, CheckStatus.Failed, $"{kernel}: {tally.Failed} of {tally.Failed + tally.Compared} calls differ ({seen})"));
            }
            else if (tally.Compared > 0)
            {
                string partly = tally.Unsupported > 0 ? $"; {tally.Unsupported} calls had no kernel on the device" : "";
                entries.Add(new CheckEntry(operation.Name, CheckStatus.Passed, $"{kernel}: {tally.Compared} calls agree ({seen}){partly}"));
            }
            else if (tally.Unsupported > 0 || source == KernelSource.None)
            {
                entries.Add(new CheckEntry(operation.Name, CheckStatus.Skipped, $"unsupported: {kernel} on this device for the cases' arguments (callers take another path)"));
            }
            else if (tally.NoReference > 0)
            {
                entries.Add(new CheckEntry(operation.Name, CheckStatus.Skipped, "the CPU has no kernel for the cases' arguments (nothing to compare with)"));
            }
            else
            {
                entries.Add(new CheckEntry(operation.Name, CheckStatus.Skipped, $"no case reaches it ({kernel} on this device)"));
            }
        }

        return new ConformanceReport(subject, entries, failures, cases);
    }

    /// <summary>
    /// Runs one recorded call on <paramref name="backend"/> with copies of its values; null when everything it writes
    /// agrees with the CPU, else what differs. <paramref name="unsupported"/> is set when the device returned false
    /// (no kernel for these arguments) where the CPU ran one.
    /// </summary>
    public static string? Run(Backend backend, RecordedCall call, float tolerance, out bool unsupported)
    {
        unsupported = false;
        var storages = new Storage?[call.Before.Length];
        try
        {
            for (int i = 0; i < storages.Length; i++)
            {
                storages[i] = backend.Allocate(call.Before[i].Length, zeroed: false);
                backend.Upload(call.Before[i], storages[i]!);
            }

            var result = OperationCalls.Invoke(backend, call.Operation, RecordedCall.Bind(call.Arguments, storages!));
            if (call.Result is true && result is false)
            {
                unsupported = true;
                return null;
            }

            var problems = new List<string>();
            if (!Equals(result, call.Result))
            {
                problems.Add($"returned {result ?? "nothing"}, the CPU {call.Result ?? "nothing"}");
            }

            for (int i = 0; i < storages.Length; i++)
            {
                var values = new float[call.After[i].Length];
                backend.Download(storages[i]!, values);
                if (Comparisons.Difference(call.After[i], values, tolerance) is { } difference)
                {
                    problems.Add($"{ArgumentName(call, i)}: {difference}");
                }
            }

            return problems.Count == 0 ? null : string.Join("; ", problems);
        }
        catch (Exception ex)
        {
            return $"threw {ex.GetType().Name}: {ex.Message}";
        }
        finally
        {
            foreach (var storage in storages)
            {
                storage?.Release();
            }
        }
    }

    // The parameter that holds slot `index` (with the tuple and item within it for a span of tuples).
    internal static string ArgumentName(RecordedCall call, int index)
    {
        var names = OperationCalls.Parameters[call.Operation.Index];
        for (int i = 0; i < call.Arguments.Length; i++)
        {
            string name = i < names.Length ? names[i] : $"argument {i + 1}";
            switch (call.Arguments[i])
            {
                case Slot s when s.Index == index:
                    return name;
                case object?[][] tuples:
                    for (int t = 0; t < tuples.Length; t++)
                    {
                        int j = Array.FindIndex(tuples[t], v => v is Slot s && s.Index == index);
                        if (j >= 0)
                        {
                            return $"{name}[{t}].Item{j + 1}";
                        }
                    }

                    break;
            }
        }

        return $"storage {index}";
    }
}
