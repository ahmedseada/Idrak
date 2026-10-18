// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A transport as the transport checks drive it: a function that connects a group of ranks (one <see cref="ITransport"/>
/// each, rank i at index i) whose receives wait at most the timeout given. Each case connects a group of its own and
/// disposes it.
/// </summary>
/// <param name="name">The transport's name in reports.</param>
/// <param name="connect">Connects a group of that many ranks with that timeout.</param>
public sealed class TransportUnderTest(string name, Func<int, TimeSpan, IReadOnlyList<ITransport>> connect)
{
    /// <summary>The transport's name in reports.</summary>
    public string Name { get; } = name;

    /// <summary>A connected group of <paramref name="worldSize"/> ranks.</summary>
    public IReadOnlyList<ITransport> Connect(int worldSize, TimeSpan timeout) => connect(worldSize, timeout);
}

/// <summary>
/// A collective algorithm as the collective checks drive it: a function that connects a group of ranks (one
/// <see cref="ICollectives"/> each, rank i at index i), over whatever transport the caller chose.
/// </summary>
/// <param name="name">The algorithm's name in reports.</param>
/// <param name="connect">Connects a group of that many ranks with that timeout.</param>
public sealed class CollectivesUnderTest(string name, Func<int, TimeSpan, IReadOnlyList<ICollectives>> connect)
{
    /// <summary>The algorithm's name in reports.</summary>
    public string Name { get; } = name;

    /// <summary>A connected group of <paramref name="worldSize"/> ranks.</summary>
    public IReadOnlyList<ICollectives> Connect(int worldSize, TimeSpan timeout) => connect(worldSize, timeout);
}

// Runs each rank's work on a thread of its own (a rank blocks on its peers, so a pool thread is not enough) and reports
// each rank's exception, or a hang past the limit.
internal static class RankThreads
{
    public static Exception?[] Run(int worldSize, Action<int> work, TimeSpan limit)
    {
        var errors = new Exception?[worldSize];
        var threads = new Thread[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            int rank = r;
            threads[r] = new Thread(() =>
            {
                try
                {
                    work(rank);
                }
                catch (Exception e)
                {
                    errors[rank] = e;
                }
            })
            { IsBackground = true, Name = $"rank {rank}" };
            threads[r].Start();
        }

        long start = Stopwatch.GetTimestamp();
        for (int r = 0; r < worldSize; r++)
        {
            var left = limit - Stopwatch.GetElapsedTime(start);
            if (!threads[r].Join(left > TimeSpan.Zero ? left : TimeSpan.Zero))
            {
                errors[r] ??= new TimeoutException($"rank {r} did not finish within {limit.TotalSeconds:F0} s (a rank waiting on a peer that never sends?)");
            }
        }

        return errors;
    }

    public static string? Describe(Exception?[] errors)
    {
        var failed = errors.Select((e, r) => e is null ? null : $"rank {r}: {e.GetType().Name}: {e.Message}").Where(m => m is not null).ToList();
        return failed.Count == 0 ? null : string.Join("; ", failed);
    }
}

/// <summary>
/// The checks of a transport (<see cref="ITransport"/>, plan 23), on groups of 2 to 8 ranks connected afresh for each case:
/// <list type="bullet">
/// <item>each end reports its own rank and the world size;</item>
/// <item>every message arrives whole, of every size from 0 bytes up, from every rank to every other, and messages from one
/// rank under one tag arrive in the order they were sent (both the blocking and the awaiting calls);</item>
/// <item>messages under different tags do not mix: a later tag can be received first;</item>
/// <item>a barrier holds every rank until the last arrives;</item>
/// <item>sending to itself or outside the group is refused, and a receive buffer too small for the message is refused;</item>
/// <item>a receive from a rank that never sends times out, naming that rank.</item>
/// </list>
/// </summary>
public sealed class TransportSuite : ContractSuite<TransportUnderTest>
{
    private static readonly TimeSpan CaseTimeout = TimeSpan.FromSeconds(30);

    /// <inheritdoc />
    public override string Name => "transport";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("two ranks, 0 to 65,537 bytes", "exchange", 2, [0, 1, 7, 65_537], 1);
        yield return Case("three ranks", "exchange", 3, [1, 1_023, 4_096], 2);
        yield return Case("four ranks", "exchange", 4, [7, 65_537], 3);
        yield return Case("eight ranks", "exchange", 8, [0, 300], 4);
        yield return Case("refusals", "refusals", 2, [16], 5);
        yield return Case("a rank that never sends", "timeout", 2, [], 6);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        int world = random.Next(2, large ? 9 : 6);
        var sizes = Enumerable.Range(0, random.Next(1, 5)).Select(_ => random.Next(0, large ? 1 << 20 : 70_000)).ToArray();
        return Case($"random {world} ranks", "exchange", world, sizes, random.Next());
    }

    /// <inheritdoc />
    protected override string Describe(TransportUnderTest implementation) => implementation.Name;

    /// <inheritdoc />
    public override void Run(TransportUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        int world = (int)@case.Data["world"]!;
        int seed = (int)@case.Data["seed"]!;
        int[] sizes = [.. @case.Data["sizes"]!.AsArray().Select(n => (int)n!)];
        switch ((string)@case.Data["kind"]!)
        {
            case "refusals":
                Refusals(implementation, checks);
                return;
            case "timeout":
                Timeout(implementation, checks);
                return;
        }

        var group = implementation.Connect(world, CaseTimeout);
        try
        {
            checks.Check("each end reports its rank and the world size",
                group.Count == world && group.Select((t, r) => t.Rank == r && t.WorldSize == world).All(x => x),
                $"connected {group.Count} ends reporting ({string.Join(", ", group.Select(t => $"{t.Rank}/{t.WorldSize}"))}) for {world} ranks");

            var problems = new string?[world];
            long lastEntered = 0;
            var exited = new long[world];
            var errors = RankThreads.Run(world, r =>
            {
                problems[r] = Exchange(group[r], world, sizes, seed);
                if (r == world - 1)
                {
                    Thread.Sleep(30);
                    Interlocked.Exchange(ref lastEntered, Stopwatch.GetTimestamp());
                }

                group[r].Barrier();
                exited[r] = Stopwatch.GetTimestamp();
            }, CaseTimeout * 2);
            string? failure = RankThreads.Describe(errors);
            checks.Check("every message arrives whole and in order, under its tag", failure is null && problems.All(p => p is null),
                failure ?? string.Join("; ", problems.Where(p => p is not null)));
            if (failure is null)
            {
                checks.Check("a barrier holds every rank until the last arrives", exited.All(t => t >= lastEntered),
                    $"ranks {string.Join(", ", exited.Select((t, r) => (t, r)).Where(x => x.t < lastEntered).Select(x => x.r))} left the barrier before rank {world - 1} reached it");
            }
        }
        finally
        {
            foreach (var t in group)
            {
                t.Dispose();
            }
        }
    }

    // Every rank sends every other rank one message per size, under tags 0, 1, 2 in turn; the first half by awaited sends
    // received by blocking receives, the second half the other way round. Each source's tags are received last tag first.
    private static string? Exchange(ITransport transport, int world, int[] sizes, int seed)
    {
        int me = transport.Rank;
        var sends = new List<Task>();
        for (int to = 0; to < world; to++)
        {
            if (to == me)
            {
                continue;
            }

            for (int k = 0; k < sizes.Length; k++)
            {
                var payload = Payload(seed, me, to, k, sizes[k]);
                if (k % 2 == 0)
                {
                    sends.Add(transport.SendAsync(to, payload, k % 3).AsTask());
                }
                else
                {
                    int target = to, tag = k % 3;
                    sends.Add(Task.Run(() => transport.Send(target, payload, tag)));
                }
            }
        }

        int largest = sizes.Length == 0 ? 0 : sizes.Max();
        var buffer = new byte[largest + 16];
        for (int from = 0; from < world; from++)
        {
            if (from == me)
            {
                continue;
            }

            for (int tag = 2; tag >= 0; tag--)
            {
                for (int k = 0; k < sizes.Length; k++)
                {
                    if (k % 3 != tag)
                    {
                        continue;
                    }

                    int length = k % 2 == 0
                        ? transport.Receive(from, buffer, tag)
                        : transport.ReceiveAsync(from, buffer.AsMemory(), tag).AsTask().GetAwaiter().GetResult();
                    var expected = Payload(seed, from, me, k, sizes[k]);
                    if (length != expected.Length || !buffer.AsSpan(0, length).SequenceEqual(expected))
                    {
                        return $"rank {me} got {length} bytes from rank {from} under tag {tag} where message {k} of {expected.Length} bytes was due";
                    }
                }
            }
        }

        Task.WaitAll([.. sends]);
        return null;
    }

    private static byte[] Payload(int seed, int from, int to, int index, int length)
    {
        var bytes = new byte[length];
        for (int j = 0; j < length; j++)
        {
            bytes[j] = (byte)((seed * 7) + (from * 31) + (to * 17) + (index * 13) + (j * 5) + (j >> 8));
        }

        return bytes;
    }

    private static void Refusals(TransportUnderTest implementation, CaseChecks checks)
    {
        var group = implementation.Connect(2, CaseTimeout);
        try
        {
            checks.Check("sending to itself is refused", Throws<ArgumentOutOfRangeException>(() => group[0].Send(0, [1, 2], 0)), "rank 0 sent to itself without an ArgumentOutOfRangeException");
            checks.Check("sending outside the group is refused", Throws<ArgumentOutOfRangeException>(() => group[0].Send(2, [1, 2], 0)), "rank 0 sent to rank 2 of 2 without an ArgumentOutOfRangeException");
            var errors = RankThreads.Run(2, r =>
            {
                if (r == 0)
                {
                    group[0].Send(1, new byte[16], 4);
                }
                else if (!Throws<InvalidDataException>(() => group[1].Receive(0, new byte[8], 4)))
                {
                    throw new InvalidOperationException("a 16-byte message went into an 8-byte buffer without an InvalidDataException");
                }
            }, CaseTimeout * 2);
            checks.Check("a receive buffer too small for the message is refused", errors.All(e => e is null), RankThreads.Describe(errors) ?? "");
        }
        finally
        {
            foreach (var t in group)
            {
                t.Dispose();
            }
        }
    }

    private static void Timeout(TransportUnderTest implementation, CaseChecks checks)
    {
        var group = implementation.Connect(2, TimeSpan.FromSeconds(1));
        try
        {
            long start = Stopwatch.GetTimestamp();
            Exception? thrown = null;
            try
            {
                group[0].Receive(1, new byte[8], 9);
            }
            catch (Exception e)
            {
                thrown = e;
            }

            var took = Stopwatch.GetElapsedTime(start);
            checks.Check("a receive from a rank that never sends times out naming the rank",
                thrown is TimeoutException && thrown.Message.Contains("rank 1", StringComparison.OrdinalIgnoreCase) && took < CaseTimeout,
                thrown is null ? "the receive returned" : $"after {took.TotalSeconds:F1} s: {thrown.GetType().Name}: {thrown.Message}");
        }
        finally
        {
            foreach (var t in group)
            {
                t.Dispose();
            }
        }
    }

    private static bool Throws<TException>(Action action)
        where TException : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (TException)
        {
            return true;
        }
    }

    private static ContractCase Case(string name, string kind, int world, int[] sizes, int seed) =>
        new(name, new JsonObject { ["kind"] = kind, ["world"] = world, ["sizes"] = new JsonArray([.. sizes.Select(s => (JsonNode)s)]), ["seed"] = seed });
}

/// <summary>
/// The checks of a collective algorithm (<see cref="ICollectives"/>, plan 23), on groups of 2 to 8 ranks connected afresh
/// for each case, every operation in turn on buffers with guard values past the count:
/// <list type="bullet">
/// <item>each rank reports its rank and the world size;</item>
/// <item>all-reduce gives the sum (within float rounding of the sum in rank order; exactly for whole numbers, also on a
/// bfloat16 wire), the largest and the smallest value, and every rank ends with identical bytes;</item>
/// <item>all-gather puts each rank's values in rank order; reduce-scatter leaves each rank its slice of the sum; broadcast
/// copies the root's values;</item>
/// <item>values past the count are not touched;</item>
/// <item>a rerun on a new group gives identical bytes (the order of each sum is fixed by the sizes, not by timing).</item>
/// </list>
/// </summary>
public sealed class CollectivesSuite : ContractSuite<CollectivesUnderTest>
{
    private const float Guard = 12345.5f;
    private static readonly TimeSpan CaseTimeout = TimeSpan.FromSeconds(60);

    /// <inheritdoc />
    public override string Name => "collectives";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        int seed = 1;
        foreach (int world in (int[])[2, 3, 4, 8])
        {
            yield return Case($"{world} ranks, 1 whole value", world, 1, "whole", "f32", seed++);
            yield return Case($"{world} ranks, 7 whole values on a bfloat16 wire", world, 7, "whole", "bf16", seed++);
            yield return Case($"{world} ranks, 1,023 values", world, 1_023, "random", "f32", seed++);
            yield return Case($"{world} ranks, 65,537 values", world, 65_537, "random", world == 8 ? "bf16" : "f32", seed++);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large) =>
        Case("random", random.Next(2, large ? 9 : 6), random.Next(1, large ? 300_000 : 5_000), random.Next(2) == 0 ? "whole" : "random",
            random.Next(3) == 0 ? "bf16" : "f32", random.Next());

    /// <inheritdoc />
    protected override string Describe(CollectivesUnderTest implementation) => implementation.Name;

    /// <inheritdoc />
    public override void Run(CollectivesUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        int world = (int)@case.Data["world"]!, count = (int)@case.Data["count"]!, seed = (int)@case.Data["seed"]!;
        bool whole = (string)@case.Data["values"]! == "whole";
        var wire = (string)@case.Data["wire"]! == "bf16" ? WireType.BFloat16 : WireType.Float32;
        int root = seed % world;
        var inputs = Enumerable.Range(0, world).Select(r => Values(seed, r, count, whole)).ToArray();
        var scattered = Enumerable.Range(0, world).Select(r => Values(seed + 101, r, count * world, whole)).ToArray();

        var first = RunGroup(implementation, world, count, wire, root, inputs, scattered, checks, report: true);
        if (first is null)
        {
            return;
        }

        var second = RunGroup(implementation, world, count, wire, root, inputs, scattered, checks, report: false);
        if (second is not null)
        {
            bool same = first.Zip(second).All(p => p.First.Zip(p.Second).All(q => q.First.AsSpan().SequenceEqual(q.Second)));
            checks.Check("a rerun gives identical bytes", same, "a second group given the same values ended with different bytes");
        }

        // Every rank's results, in order: sum, max, min, gathered, scattered, broadcast.
        float[] Reference(Func<int, int, float> value, int length, ReduceOp op)
        {
            var result = new float[length];
            for (int i = 0; i < length; i++)
            {
                double acc = value(0, i);
                for (int r = 1; r < world; r++)
                {
                    float v = value(r, i);
                    acc = op switch { ReduceOp.Max => Math.Max(acc, v), ReduceOp.Min => Math.Min(acc, v), _ => acc + v };
                }

                result[i] = (float)acc;
            }

            return result;
        }

        float[] Magnitude(Func<int, int, float> value, int length)
        {
            var result = new float[length];
            for (int i = 0; i < length; i++)
            {
                double acc = 0;
                for (int r = 0; r < world; r++)
                {
                    acc += Math.Abs(value(r, i));
                }

                result[i] = (float)acc;
            }

            return result;
        }

        var sum = Reference((r, i) => inputs[r][i], count, ReduceOp.Sum);
        var size = Magnitude((r, i) => inputs[r][i], count);
        var max = Reference((r, i) => inputs[r][i], count, ReduceOp.Max);
        var min = Reference((r, i) => inputs[r][i], count, ReduceOp.Min);
        double epsilon = wire == WireType.BFloat16 ? 1.0 / 256 : 1.2e-7;
        string? sumProblem = null, gatherProblem = null, scatterProblem = null, broadcastProblem = null, maxProblem = null;
        for (int r = 0; r < world && sumProblem is null; r++)
        {
            sumProblem = Close(first[r][0], sum, size, whole ? 0 : (world + 1) * epsilon, $"rank {r}'s all-reduce sum");
            maxProblem ??= Close(first[r][1], max, size, 0, $"rank {r}'s all-reduce max") ?? Close(first[r][2], min, size, 0, $"rank {r}'s all-reduce min");
            var gathered = inputs.SelectMany(v => v).ToArray();
            gatherProblem ??= Close(first[r][3], gathered, gathered, 0, $"rank {r}'s all-gather");
            var slice = Reference((q, i) => scattered[q][(r * count) + i], count, ReduceOp.Sum);
            var sliceSize = Magnitude((q, i) => scattered[q][(r * count) + i], count);
            scatterProblem ??= Close(first[r][4], slice, sliceSize, whole ? 0 : (world + 1) * 1.2e-7, $"rank {r}'s reduce-scatter slice");
            broadcastProblem ??= Close(first[r][5], inputs[root], inputs[root], 0, $"rank {r}'s broadcast from rank {root}");
        }

        checks.Check(whole ? "all-reduce of whole numbers is exact" : "all-reduce sums within float rounding", sumProblem is null, sumProblem ?? "");
        checks.Check("all-reduce max and min are exact", maxProblem is null, maxProblem ?? "");
        checks.Check("all-gather puts each rank's values in rank order", gatherProblem is null, gatherProblem ?? "");
        checks.Check("reduce-scatter leaves each rank its slice of the sum", scatterProblem is null, scatterProblem ?? "");
        checks.Check("broadcast copies the root's values", broadcastProblem is null, broadcastProblem ?? "");
        bool identical = Enumerable.Range(1, world - 1).All(r => first[r][0].AsSpan().SequenceEqual(first[0][0]) && first[r][3].AsSpan().SequenceEqual(first[0][3]));
        checks.Check("every rank ends with identical bytes", identical, "two ranks hold different all-reduce or all-gather results");
    }

    // One group: every operation on every rank; null when a rank failed (reported).
    private static float[][][]? RunGroup(CollectivesUnderTest implementation, int world, int count, WireType wire, int root, float[][] inputs,
        float[][] scattered, CaseChecks checks, bool report)
    {
        var group = implementation.Connect(world, CaseTimeout);
        try
        {
            if (report)
            {
                checks.Check("each rank reports its rank and the world size",
                    group.Count == world && group.Select((c, r) => c.Rank == r && c.WorldSize == world).All(x => x),
                    $"connected {group.Count} ranks reporting ({string.Join(", ", group.Select(c => $"{c.Rank}/{c.WorldSize}"))}) for {world} ranks");
            }

            var results = new float[world][][];
            bool untouched = true;
            var errors = RankThreads.Run(world, r =>
            {
                var c = group[r];
                var backend = c.Device.Backend;
                float[] Op(float[] input, int length, int outputLength, Action<Storage, Storage> op)
                {
                    var buffer = Upload(backend, input, length);
                    var output = outputLength == 0 ? buffer : Upload(backend, Enumerable.Repeat(Guard, outputLength + 3).ToArray(), outputLength);
                    try
                    {
                        op(buffer, output);
                        var all = new float[output.Length];
                        backend.Download(output, all);
                        if (all.AsSpan(outputLength == 0 ? length : outputLength).IndexOfAnyExcept(Guard) >= 0)
                        {
                            untouched = false;
                        }

                        return all[..(outputLength == 0 ? length : outputLength)];
                    }
                    finally
                    {
                        buffer.Release();
                        if (!ReferenceEquals(output, buffer))
                        {
                            output.Release();
                        }
                    }
                }

                results[r] =
                [
                    Op(inputs[r], count, 0, (b, _) => c.AllReduce(b, count, ReduceOp.Sum, wire)),
                    Op(inputs[r], count, 0, (b, _) => c.AllReduce(b, count, ReduceOp.Max)),
                    Op(inputs[r], count, 0, (b, _) => c.AllReduce(b, count, ReduceOp.Min)),
                    Op(inputs[r], count, world * count, (b, o) => c.AllGather(b, count, o)),
                    Op(scattered[r], world * count, count, (b, o) => c.ReduceScatter(b, count, o, ReduceOp.Sum)),
                    Op(inputs[r], count, 0, (b, _) => c.Broadcast(b, count, root)),
                ];
                c.Barrier();
            }, CaseTimeout * 2);
            string? failure = RankThreads.Describe(errors);
            if (report)
            {
                checks.Check("every operation completes on every rank", failure is null, failure ?? "");
                if (failure is null)
                {
                    checks.Check("values past the count are not touched", untouched, "an operation wrote past the count it was given");
                }
            }

            return failure is null ? results : null;
        }
        finally
        {
            foreach (var c in group)
            {
                c.Dispose();
            }
        }
    }

    private static Storage Upload(Backend backend, float[] values, int length)
    {
        var storage = backend.Allocate(length + 3, zeroed: false);
        var padded = new float[length + 3];
        values.AsSpan(0, Math.Min(values.Length, length)).CopyTo(padded);
        padded.AsSpan(length).Fill(Guard);
        backend.Upload(padded, storage);
        return storage;
    }

    private static float[] Values(int seed, int rank, int count, bool whole)
    {
        var random = new Random(HashCode.Combine(seed, rank));
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = whole ? random.Next(-8, 9) : (float)((random.NextDouble() - 0.5) * (1 + rank) * Math.Pow(10, random.Next(-3, 4)));
        }

        return values;
    }

    // Null when every value is within tolerance · magnitude of the reference (exact at 0); else the first difference.
    private static string? Close(float[] got, float[] expected, float[] magnitude, double tolerance, string what)
    {
        if (got.Length != expected.Length)
        {
            return $"{what} has {got.Length} values where {expected.Length} were due";
        }

        for (int i = 0; i < got.Length; i++)
        {
            double allowed = tolerance * magnitude[i];
            if (tolerance == 0 ? got[i] != expected[i] : Math.Abs(got[i] - expected[i]) > allowed + 1e-30)
            {
                return $"{what}: value {i} is {got[i]:R} where {expected[i]:R} was due (allowed difference {allowed:G3})";
            }
        }

        return null;
    }

    private static ContractCase Case(string name, int world, int count, string values, string wire, int seed) =>
        new(name, new JsonObject { ["world"] = world, ["count"] = count, ["values"] = values, ["wire"] = wire, ["seed"] = seed });
}

/// <summary>
/// The checks of a link profile (<see cref="ILinkProfile"/>, plan 23) on transfers of many sizes:
/// <list type="bullet">
/// <item>its figures are in range;</item>
/// <item>a transfer of no bytes takes the latency, plus at most the jitter;</item>
/// <item>the time grows with the bytes, and the same seed gives the same time;</item>
/// <item>a copy with one figure changed keeps the others;</item>
/// <item>a profile with no known bandwidth refuses to predict instead of guessing.</item>
/// </list>
/// </summary>
public sealed class LinkProfileSuite : ContractSuite<ILinkProfile>
{
    /// <inheritdoc />
    public override string Name => "link-profile";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("small transfers", [0, 1, 64, 4_096], 1);
        yield return Case("gradient-sized transfers", [1 << 20, 25 << 20, 1L << 30], 2);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large) =>
        Case("random", [.. Enumerable.Range(0, 4).Select(_ => random.NextInt64(0, large ? 1L << 34 : 1L << 26)).Order()], random.Next());

    /// <inheritdoc />
    protected override string Describe(ILinkProfile implementation) => implementation.Name;

    /// <inheritdoc />
    public override void Run(ILinkProfile implementation, ContractCase @case, CaseChecks checks)
    {
        string? invalid = null;
        try
        {
            LinkProfile.Validate(implementation);
        }
        catch (ArgumentException e)
        {
            invalid = e.Message;
        }

        checks.Check("its figures are in range", invalid is null, invalid ?? "");
        int seed = (int)@case.Data["seed"]!;
        long[] sizes = [.. @case.Data["bytes"]!.AsArray().Select(n => (long)n!)];
        if (!(implementation.BandwidthBytesPerSecond > 0))
        {
            bool refused;
            try
            {
                implementation.Predict(1, new Random(seed));
                refused = false;
            }
            catch (InvalidOperationException)
            {
                refused = true;
            }

            checks.Check("a profile with no known bandwidth refuses to predict", refused, "it predicted a time without a bandwidth");
            return;
        }

        var zero = implementation.Predict(0, new Random(seed));
        checks.Check("no bytes take the latency, plus at most the jitter", zero >= implementation.Latency && zero <= implementation.Latency + implementation.Jitter + TimeSpan.FromTicks(1),
            $"0 bytes took {zero.TotalMicroseconds:F3} us with latency {implementation.Latency.TotalMicroseconds:F3} us and jitter {implementation.Jitter.TotalMicroseconds:F3} us");
        var times = sizes.Select(b => implementation.Predict(b, new Random(seed))).ToArray();
        bool grows = times.Zip(times.Skip(1)).All(p => p.Second >= p.First);
        checks.Check("the time grows with the bytes", grows, $"times {string.Join(", ", times.Select(t => t.TotalMilliseconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)))} ms for {string.Join(", ", sizes)} bytes");
        bool repeats = sizes.All(b => implementation.Predict(b, new Random(seed)) == implementation.Predict(b, new Random(seed)));
        checks.Check("the same seed gives the same time", repeats, "two predictions with the same seed differ");
        var copy = implementation.With(bandwidthBytesPerSecond: implementation.BandwidthBytesPerSecond * 2);
        checks.Check("a copy with one figure changed keeps the others",
            copy.BandwidthBytesPerSecond == implementation.BandwidthBytesPerSecond * 2 && copy.Latency == implementation.Latency && copy.Jitter == implementation.Jitter
            && copy.LossRate == implementation.LossRate && copy.FailureRate == implementation.FailureRate && copy.Name == implementation.Name,
            "With(bandwidth) changed another figure or the name");
    }

    private static ContractCase Case(string name, long[] bytes, int seed) =>
        new(name, new JsonObject { ["bytes"] = new JsonArray([.. bytes.Select(b => (JsonNode)b)]), ["seed"] = seed });
}

/// <summary>
/// The checks of a topology (<see cref="IClusterTopology"/>, plan 23):
/// <list type="bullet">
/// <item>the world size is the device count, and every rank is placed once, in file order;</item>
/// <item>every pair of different ranks has a link, the same both ways, with figures in range; two ranks on one device of
/// one node are joined by <c>same-device</c>;</item>
/// <item>a node is found by its name ignoring case, and an unknown name is not;</item>
/// <item>a rank outside the group is refused;</item>
/// <item>a <see cref="ClusterTopology"/> written to JSON reads back to the same topology.</item>
/// </list>
/// </summary>
public sealed class TopologySuite : ContractSuite<IClusterTopology>
{
    /// <inheritdoc />
    public override string Name => "topology";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return new ContractCase("every pair", new JsonObject { ["pairs"] = 0, ["seed"] = 1 });
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large) =>
        new("sampled pairs", new JsonObject { ["pairs"] = large ? 4_096 : 256, ["seed"] = random.Next() });

    /// <inheritdoc />
    public override void Run(IClusterTopology implementation, ContractCase @case, CaseChecks checks)
    {
        var t = implementation;
        int devices = t.Nodes.Sum(n => n.Devices.Count);
        checks.Check("the world size is the device count", t.WorldSize == devices, $"world size {t.WorldSize} for {devices} devices");
        var expected = t.Nodes.SelectMany((n, i) => n.Devices.Select((_, local) => (i, local))).ToArray();
        bool placed = expected.Length == t.WorldSize && expected.Select((p, r) => t.Place(r) == p).All(x => x)
                      && Enumerable.Range(0, t.Nodes.Count).All(n => t.Ranks(n).SequenceEqual(Enumerable.Range(0, t.WorldSize).Where(r => t.Place(r).Node == n)));
        checks.Check("every rank is placed once, in file order", placed, "Place or Ranks disagree with the nodes' devices in file order");

        int pairs = (int)@case.Data["pairs"]!;
        var random = new Random((int)@case.Data["seed"]!);
        var sample = pairs == 0 || (long)t.WorldSize * t.WorldSize <= pairs
            ? Enumerable.Range(0, t.WorldSize).SelectMany(a => Enumerable.Range(0, t.WorldSize).Where(b => b != a).Select(b => (a, b)))
            : Enumerable.Range(0, pairs).Select(_ => (random.Next(t.WorldSize), random.Next(t.WorldSize))).Where(p => p.Item1 != p.Item2);
        string? linkProblem = null;
        foreach (var (a, b) in sample)
        {
            var ab = t.Link(a, b);
            var ba = t.Link(b, a);
            var (na, la) = t.Place(a);
            var (nb, lb) = t.Place(b);
            bool sameDevice = na == nb && string.Equals(t.Nodes[na].Devices[la].Device, t.Nodes[nb].Devices[lb].Device, StringComparison.OrdinalIgnoreCase);
            try
            {
                LinkProfile.Validate(ab);
            }
            catch (ArgumentException e)
            {
                linkProblem ??= $"ranks {a} and {b}: {e.Message}";
            }

            if (ab.Name != ba.Name || ab.BandwidthBytesPerSecond != ba.BandwidthBytesPerSecond || ab.Latency != ba.Latency)
            {
                linkProblem ??= $"ranks {a} and {b} are joined by {ab.Name} one way and {ba.Name} the other";
            }

            if (sameDevice && !string.Equals(ab.Name, ClusterTopology.SameDevice, StringComparison.OrdinalIgnoreCase))
            {
                linkProblem ??= $"ranks {a} and {b} share {t.Nodes[na].Devices[la].Device} of node {t.Nodes[na].Name} but are joined by {ab.Name}";
            }
        }

        checks.Check("every pair has a link, the same both ways", linkProblem is null, linkProblem ?? "");
        bool found = t.Nodes.Select((n, i) => t.NodeOf(n.Name.ToUpperInvariant()) == i && t.NodeOf(n.Name.ToLowerInvariant()) == i).All(x => x)
                     && t.NodeOf("no-such-node-" + Guid.NewGuid().ToString("N")) is null;
        checks.Check("a node is found by its name ignoring case", found, "NodeOf missed a node's name in another case, or found an unknown name");
        bool refused;
        try
        {
            t.Place(t.WorldSize);
            refused = false;
        }
        catch (ArgumentOutOfRangeException)
        {
            refused = true;
        }

        checks.Check("a rank outside the group is refused", refused, $"Place({t.WorldSize}) answered for a group of {t.WorldSize}");
        if (t is ClusterTopology file)
        {
            var back = ClusterTopology.Parse(file.ToJson());
            checks.Check("written to JSON it reads back the same", back.ToJson() == file.ToJson() && back.WorldSize == file.WorldSize,
                "the JSON read back differs from the topology written");
        }
        else
        {
            checks.Skip("written to JSON it reads back the same", "not a ClusterTopology");
        }
    }
}
