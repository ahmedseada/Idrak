// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Training;
using Idrak.Networking.Transports;
using Idrak.Text;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 23, phase C6: receive flow control on the real transports (C5's open item), made directly and run in one process.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterShardingGroup =
    [
        ("cluster: receive flow control bounds a slow receiver's memory on tcp, shared-memory and pcie-host (credits per peer and tag), an awaited send beyond the credit waits without a thread, and every rank sending before it receives still completes", ClusterReceiveCredit),
    ];

    // `world` ranks of one machine joined by `provider`, every rank connecting on a thread of its own, with `options`.
    private static ITransport[] CreditGroup(ITransportProvider provider, int world, ClusterOptions options)
    {
        var topology = RealTopology([world]);
        return ConnectAll(world, r => provider.Connect(new ClusterContext(topology, r, options)));
    }

    private static void ClusterReceiveCredit(Device device)
    {
        if (!FirstRun(nameof(ClusterReceiveCredit)))
        {
            return;
        }

        const int Size = 16 << 10, Count = 120;
        foreach (var provider in new ITransportProvider[] { new TcpProvider(), new SharedMemoryProvider(), new PciHostProvider() })
        {
            long Peak(int inFlight, long? budget)
            {
                var group = CreditGroup(provider, 2, RealOptions(TimeSpan.FromSeconds(30)) with { InFlight = inFlight, ReceiveBytes = budget });
                try
                {
                    var sending = Task.Run(() =>
                    {
                        for (int i = 0; i < Count; i++)
                        {
                            group[0].Send(1, Enumerable.Repeat((byte)i, Size).ToArray(), 5);
                        }
                    });
                    var buffer = new byte[Size];
                    for (int i = 0; i < Count; i++)
                    {
                        Thread.Sleep(1);                                     // a slow receiver
                        int n = group[1].Receive(0, buffer, 5);
                        Check(n == Size && buffer[0] == (byte)i && buffer[^1] == (byte)i, $"{provider.Name}: message {i} arrived whole and in order");
                    }

                    Check(sending.Wait(TimeSpan.FromSeconds(30)), $"{provider.Name}: the sender finished");
                    return ((FramedTransport)group[1]).PeakQueuedBytes;
                }
                finally
                {
                    foreach (var t in group)
                    {
                        t.Dispose();
                    }
                }
            }

            long bounded = Peak(2, 24 << 10), loose = Peak(10_000, null);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {provider.Name}: {Count} messages of {Size / 1024} KiB to a slow receiver: at most {bounded / 1024.0:F0} KiB held with 2 in flight and a 24 KiB budget, {loose / 1024.0:F0} KiB without a bound"));
            Check(bounded <= Size * 2 && bounded <= (24 << 10) + Size, $"{provider.Name}: the receiver held at most the credit: {bounded} B");
            Check(loose > bounded * 4, $"{provider.Name}: without the bound a fast sender fills the receiver: {loose} B against {bounded} B");
        }

        // Every rank sends a message far larger than the budget to every other rank, under two tags, before it receives
        // anything: a message is always let go when none of its peer and tag is outstanding, so nobody waits in a circle.
        foreach (var provider in new ITransportProvider[] { new TcpProvider(), new SharedMemoryProvider(), new PciHostProvider() })
        {
            var group = CreditGroup(provider, 3, RealOptions(TimeSpan.FromSeconds(30)) with { InFlight = 1, ReceiveBytes = 4 << 10, SharedMemoryBytes = 64 << 10 });
            try
            {
                const int Large = 1 << 20;
                var ranks = Enumerable.Range(0, 3).Select(me => Task.Factory.StartNew(() =>
                {
                    var t = group[me];
                    for (int to = 0; to < 3; to++)
                    {
                        if (to != me)
                        {
                            t.Send(to, Enumerable.Repeat((byte)(me + 1), Large).ToArray(), 6);
                            t.Send(to, [(byte)me], 7);
                        }
                    }

                    var buffer = new byte[Large];
                    for (int from = 0; from < 3; from++)
                    {
                        if (from != me)
                        {
                            bool small = t.Receive(from, buffer, 7) == 1 && buffer[0] == from;
                            bool large = t.Receive(from, buffer, 6) == Large && buffer[Large - 1] == from + 1;
                            if (!small || !large)
                            {
                                throw new InvalidDataException($"rank {me} got the wrong bytes from rank {from}");
                            }
                        }
                    }
                }, TaskCreationOptions.LongRunning)).ToArray();
                Check(Task.WaitAll(ranks, TimeSpan.FromSeconds(60)), $"{provider.Name}: every rank sent 1 MiB to every other before receiving (a 4 KiB budget, 1 in flight) and all arrived");
                foreach (var task in ranks.Where(t => t.IsFaulted))
                {
                    Check(false, $"{provider.Name}: {task.Exception!.InnerException!.Message}");
                }

                // An awaited send beyond the credit waits without a thread, and goes once the peer takes the one before.
                var first = group[0].SendAsync(1, new byte[8], 8);
                var second = group[0].SendAsync(1, new byte[8], 8);
                Check(first.IsCompleted && !second.IsCompleted, $"{provider.Name}: the second awaited send waits for its credit ({first.IsCompleted}, {second.IsCompleted})");
                group[1].Receive(0, new byte[8], 8);
                Check(second.AsTask().Wait(TimeSpan.FromSeconds(10)), $"{provider.Name}: it went once the first was taken");
                group[1].Receive(0, new byte[8], 8);
            }
            finally
            {
                foreach (var t in group)
                {
                    t.Dispose();
                }
            }
        }
    }
}
