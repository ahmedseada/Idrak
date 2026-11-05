// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;

// Plan 37, K0: a paged key/value cache with shared prompt prefixes. A context paged in a KeyValuePages pool gives the
// unpaged cache's logits bit for bit when fed alike (every built-in format); a context that reads another's published
// pages gives what an unpaged context holding the same rows gives; a shared page is copied before it is written; N
// contexts sharing a prompt hold about one prompt's pages and their own; the generator shares its generations' pages.
internal abstract partial class Tests
{
    // The tiny model's token file as ids: `count` of them from `start`.
    private static int[] PagedIds(int start, int count)
    {
        var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
        return [.. Enumerable.Range(start, count).Select(i => (int)BitConverter.ToUInt16(tokens, i * 2))];
    }

    // The logits of feeding `ids` [from, to) through `network` with `context`: in one pass, or one position a step.
    private static float[] PagedFeed(Sequential network, DecodingContext context, int[] ids, int from, int to, bool steps, Device device)
    {
        using var noGrad = Autograd.NoGrad();
        if (!steps)
        {
            using var scope = new TensorScope();
            return network.ForwardCached(Tensor.From([.. ids[from..to].Select(i => (float)i)], [1, to - from], device), context).ToArray();
        }

        var all = new List<float>();
        for (int i = from; i < to; i++)
        {
            using var scope = new TensorScope();
            all.AddRange(network.ForwardCached(Tensor.From([(float)ids[i]], [1, 1], device), context).ToArray());
        }

        return [.. all];
    }

    private static string PagedGap(float[] a, float[] b) =>
        a.Length != b.Length ? $"lengths {a.Length} and {b.Length}" : string.Create(CultureInfo.InvariantCulture, $"largest gap {Gap(a, b):G3}, {a.Zip(b).Count(p => p.First != p.Second)} differ");

    internal static void PagedCacheMatchesUnpaged(Device device)
    {
        using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
        var network = model.Network;
        int[] a = PagedIds(0, 48), b = [.. a[..24], .. PagedIds(200, 14)];
        const int Capacity = 128;
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            var layout = KeyValueLayouts.For(format);
            using var pool = new KeyValuePages(device, layout, pageTokens: 8);

            // A: its prompt in one pass, then 8 steps; unpaged and paged alike.
            using var referenceA = new DecodingContext(device, 1, Capacity, layout);
            float[] wantA = [.. PagedFeed(network, referenceA, a, 0, 40, false, device), .. PagedFeed(network, referenceA, a, 40, 48, true, device)];
            using var pagedA = pool.Open(Capacity, a, 39);
            Check(pagedA.Length == 0 && pagedA.Pages == pool && !pagedA.Recordable, $"{format}: an empty pool maps nothing, and a paged context's steps are not recorded");
            float[] gotA = [.. PagedFeed(network, pagedA, a, 0, 40, false, device), .. PagedFeed(network, pagedA, a, 40, 48, true, device)];
            Check(gotA.SequenceEqual(wantA), $"{format}: the paged context gives the unpaged cache's logits bit for bit ({PagedGap(wantA, gotA)})");
            pool.Publish(pagedA, a);
            Check(pagedA.PageCount == 6 && pool.Stats is { Pages: 6, InUse: 6, Shared: 0, Idle: 0 }, $"{format}: A holds 6 pages of 8 positions: {pool.Stats}");

            // B shares A's first 24 positions (3 whole pages), then its own 14 tokens.
            using var referenceB = new DecodingContext(device, 1, Capacity, layout);
            PagedFeed(network, referenceB, a, 0, 40, false, device);                 // the rows A's pass computed
            referenceB.Truncate(24);
            float[] wantB = [.. PagedFeed(network, referenceB, b, 24, 30, false, device), .. PagedFeed(network, referenceB, b, 30, 38, true, device)];
            using var pagedB = pool.Open(Capacity, b, b.Length - 1);
            Check(pagedB.Length == 24 && pool.Stats is { Shared: 3 } shared && shared.SavedBytes > 0, $"{format}: B reads A's 3 whole pages: Length {pagedB.Length}, {pool.Stats}");
            float[] gotB = [.. PagedFeed(network, pagedB, b, 24, 30, false, device), .. PagedFeed(network, pagedB, b, 30, 38, true, device)];
            Check(gotB.SequenceEqual(wantB), $"{format}: B over A's pages gives the unpaged cache's logits bit for bit ({PagedGap(wantB, gotB)})");

            // B truncated into a page A reads, and fed again: it copies the page; A's next steps are unchanged.
            pagedB.Truncate(20);
            referenceB.Truncate(20);
            float[] wantAgain = PagedFeed(network, referenceB, b, 20, 30, false, device);
            float[] gotAgain = PagedFeed(network, pagedB, b, 20, 30, false, device);
            Check(gotAgain.SequenceEqual(wantAgain), $"{format}: B truncated into a shared page and fed again ({PagedGap(wantAgain, gotAgain)})");
            int[] more = PagedIds(400, 4);
            float[] wantMore = PagedFeed(network, referenceA, more, 0, 4, true, device), gotMore = PagedFeed(network, pagedA, more, 0, 4, true, device);
            Check(gotMore.SequenceEqual(wantMore) && pool.Stats.Shared == 2,
                $"{format}: A's pages untouched by B's write (copied on write; 2 pages still shared: {pool.Stats}); A's next steps {PagedGap(wantMore, gotMore)}");
        }

        // A context opened from another pool, and what a paged context refuses.
        using var other = new KeyValuePages(device, KeyValueLayouts.For(KeyValueFormat.Float32), pageTokens: 8);
        using var stranger = new DecodingContext(device, 1, Capacity);
        ThrowsNaming<ArgumentException>(() => other.Publish(stranger, a), "publishing an unpaged context", "KeyValuePages.Open");
        using var paged = other.Open(Capacity, a, 0);
        ThrowsNaming<NotSupportedException>(() => paged.ReorderRows([0]), "reordering a paged context's rows", "IDRAK_KV_SHARE_PREFIX");
        ThrowsNaming<ArgumentException>(() => KeyValuePages.ParsePageTokens("0", KeyValuePages.PageTokensVariable), "a page of no positions", "IDRAK_KV_PAGE_TOKENS", "1 or more");
        Check(KeyValuePages.ParsePageTokens(" 32 ", "x") == 32 && KeyValuePages.ParsePageTokens(null, "x") is null, "page sizes parse");
    }

    internal static void PagedCacheSharedMemory(Device device)
    {
        using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
        var network = model.Network;
        const int Prompt = 48, Own = 8, Requests = 4, Page = 8;
        int capacity = model.MaxPositions;                                   // the rotary tables' positions
        int[] prompt = PagedIds(0, Prompt);
        int[][] ids = [.. Enumerable.Range(0, Requests).Select(r => prompt.Concat(PagedIds(500 + (r * 50), Own)).ToArray())];
        var layout = KeyValueLayouts.For(KeyValueFormat.Float32);

        // Unpaged: every request its own cache of the prompt and its tokens (the prompt's rows as one pass computed them).
        var backend = device.Backend;
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long before = backend.GetMemoryUsage().InUse;
        var unpaged = new List<DecodingContext>();
        var want = new float[Requests][];
        try
        {
            for (int r = 0; r < Requests; r++)
            {
                var context = new DecodingContext(device, 1, capacity, layout);
                unpaged.Add(context);
                PagedFeed(network, context, ids[r], 0, Prompt, false, device);
                want[r] = PagedFeed(network, context, ids[r], Prompt, Prompt + Own, true, device);
            }

            long unpagedBytes = backend.GetMemoryUsage().InUse - before;
            long unpagedCaches = unpaged.Sum(c => c.CacheBytes);
            foreach (var context in unpaged)
            {
                context.Dispose();
            }

            unpaged.Clear();

            // Paged: the first computes the prompt and publishes it; the others read its pages, all open at once.
            before = backend.GetMemoryUsage().InUse;
            using var pool = new KeyValuePages(device, layout, Page);
            var paged = new List<DecodingContext>();
            try
            {
                var got = new float[Requests][];
                for (int r = 0; r < Requests; r++)
                {
                    var context = pool.Open(capacity, ids[r], Prompt);
                    paged.Add(context);
                    if (r == 0)
                    {
                        PagedFeed(network, context, ids[r], 0, Prompt, false, device);
                        pool.Publish(context, ids[r]);
                    }

                    Check(context.Length == Prompt, $"request {r} starts after the prompt: {context.Length}");
                    got[r] = PagedFeed(network, context, ids[r], Prompt, Prompt + Own, true, device);
                }

                long pagedBytes = backend.GetMemoryUsage().InUse - before;
                var stats = pool.Stats;
                long page = stats.Bytes / stats.Pages;
                for (int r = 0; r < Requests; r++)
                {
                    Check(got[r].SequenceEqual(want[r]), $"request {r}: the shared prompt's pages give its own cache's logits bit for bit ({PagedGap(want[r], got[r])})");
                }

                // One prompt's pages (8) and each request's own (1 each): 12 pages, the prompt's shared by all 4.
                Check(stats is { Pages: (Prompt / Page) + Requests, Shared: Prompt / Page } && stats.SavedBytes == (Requests - 1) * (Prompt / Page) * page,
                    $"4 requests sharing a prompt of {Prompt} positions hold its {Prompt / Page} pages once and one page each: {stats}");
                Check(pagedBytes <= stats.Bytes + (64L << 10) && pagedBytes < unpagedBytes / 2,
                    $"measured: the paged requests hold {pagedBytes:N0} B on the device (their pages {stats.Bytes:N0} B), the unpaged {unpagedBytes:N0} B (caches {unpagedCaches:N0} B)");
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"    measured: {Requests} requests of a {Prompt}-position prompt and {Own} own tokens: paged {pagedBytes:N0} B in use ({stats.Pages} pages of {page:N0} B, {stats.SavedBytes:N0} B saved), unpaged {unpagedBytes:N0} B"));
            }
            finally
            {
                foreach (var context in paged)
                {
                    context.Dispose();
                }
            }

            Check(pool.Stats is { InUse: 0, Idle: > 0 } idle && idle.Idle <= (Prompt / Page) + 1, $"closed, the published pages stay idle up to one context's worth: {pool.Stats}");
            pool.ReleaseIdle();
            Check(pool.Stats is { Pages: 0, Bytes: 0 }, $"ReleaseIdle frees them: {pool.Stats}");
        }
        finally
        {
            foreach (var context in unpaged)
            {
                context.Dispose();
            }
        }
    }

    internal static void PagedCacheGenerator(Device device)
    {
        using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 8, ChunkSize = 1, Seed = 3 };
        string system = "You are a careful assistant who answers briefly and plainly. ";
        string first = system + "What is two and two?", second = system + "Name a colour.";

        // Unpaged, one after another: the second reuses the first's kept cache for their shared start.
        var keep = new TextGenerator(model.Network, model.Tokenizer!, model.MaxPositions) { KeepCache = true, SharePrefix = false };
        string wantFirst = keep.Generate(first, options).Text, wantSecond = keep.Generate(second, options).Text;
        keep.ReleaseCache();

        // Paged with pages of one position (the shared start as the kept cache's), the two generations interleaved: the
        // second opens while the first decodes, reading the first prompt's pages.
        var shared = new TextGenerator(model.Network, model.Tokenizer!, model.MaxPositions) { SharePrefix = true, PageTokens = 1 };
        var reports = new List<KeyValuePagesReported>();
        var hook = new PagesHook(reports);
        string gotFirst, gotSecond;
        KeyValuePageStats during;
        using (Telemetry.Subscribe(hook))
        {
            using var a = shared.Stream(first, options).GetEnumerator();
            var textA = new System.Text.StringBuilder();
            Check(a.MoveNext(), "the first generation gives a chunk");
            textA.Append(a.Current.Text);
            using var b = shared.Stream(second, options).GetEnumerator();
            var textB = new System.Text.StringBuilder();
            Check(b.MoveNext(), "the second generation gives a chunk");
            textB.Append(b.Current.Text);
            during = shared.PageStats!.Value;
            while (a.MoveNext())
            {
                textA.Append(a.Current.Text);
            }

            while (b.MoveNext())
            {
                textB.Append(b.Current.Text);
            }

            (gotFirst, gotSecond) = (textA.ToString(), textB.ToString());
        }

        Check(gotFirst == wantFirst && gotSecond == wantSecond, $"the paged generations' replies are the kept cache's: [{gotFirst}] [{gotSecond}] against [{wantFirst}] [{wantSecond}]");
        Check(during is { Shared: > 0, SavedBytes: > 0 }, $"while both ran they shared the system prompt's pages: {during}");
        Check(reports.Count == 2 && reports[0].MappedTokens == 0 && reports[1].MappedTokens > 0 && reports[1].Shared > 0 && reports[1].SavedBytes > 0,
            $"the telemetry reports each opening's pages in use, shared and bytes saved: {string.Join(" | ", reports)}");
        Check(shared.PageStats is { InUse: 0, Idle: > 0 }, $"after both, their pages stay idle for the next prompt: {shared.PageStats}");
        shared.ReleaseCache();
        Check(shared.PageStats is { Pages: 0 }, $"ReleaseCache frees the idle pages: {shared.PageStats}");
    }

    // Keeps every KeyValuePagesReported published.
    private sealed class PagesHook(List<KeyValuePagesReported> reports) : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Inference;

        public void OnKeyValuePages(in KeyValuePagesReported e)
        {
            lock (reports)
            {
                reports.Add(e);
            }
        }
    }
}
