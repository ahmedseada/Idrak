// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction.Generation;

/// <summary>What a <see cref="KeyValuePages"/> pool holds now.</summary>
/// <param name="PageTokens">The positions a page holds.</param>
/// <param name="Pages">The pages alive: in use by a context, or idle and kept for a later prompt.</param>
/// <param name="InUse">The pages one or more contexts read.</param>
/// <param name="Shared">The pages two or more contexts read at once.</param>
/// <param name="Idle">The published pages no context reads, kept for a later prompt that starts alike.</param>
/// <param name="Bytes">The device memory of every page alive.</param>
/// <param name="SavedBytes">What the shared pages would take again in contexts of their own: each one's bytes once for every reader past the first.</param>
public readonly record struct KeyValuePageStats(int PageTokens, int Pages, int InUse, int Shared, int Idle, long Bytes, long SavedBytes);

/// <summary>
/// A paged key/value cache with shared prompt prefixes (plan 37, K0), as vLLM's: the keys and values of every attention
/// layer are kept in pages of <see cref="PageTokens"/> positions, and contexts opened from the pool
/// (<see cref="Open"/>) read the pages of a prompt that starts as theirs instead of computing and holding them again. A
/// page holds every layer's rows for its positions; once a context has computed it whole it is published under its
/// tokens (<see cref="Publish"/>), found by later prompts by those exact tokens after the same earlier pages, and never
/// written again: a context that would write into a page others read copies it first (copy on write). A paged context
/// assembles one layer's working cache from its pages while that layer attends and copies the step's new rows back to
/// its own pages, so attention runs the same kernels on the same values, bit for bit, on every device and in every
/// format (<see cref="IKeyValueLayout"/>); its steps are not recorded (<see cref="DecodingContext.Recordable"/>).
/// Published pages no context reads stay for a later prompt, up to <see cref="IdlePageLimit"/> (least recently used go
/// first), so the pool's memory does not grow with use. Thread-safe: concurrent generations share one pool.
/// </summary>
public sealed class KeyValuePages : IDisposable
{
    /// <summary>The variable giving the positions a page holds: "IDRAK_KV_PAGE_TOKENS".</summary>
    public const string PageTokensVariable = "IDRAK_KV_PAGE_TOKENS";

    /// <summary>The variable that makes generations share their prompts' pages when "1", "true", "on" or "yes": "IDRAK_KV_SHARE_PREFIX".</summary>
    public const string SharePrefixVariable = "IDRAK_KV_SHARE_PREFIX";

    /// <summary>The positions a page holds when neither an option nor <see cref="PageTokensVariable"/> says: 64.</summary>
    public const int StandardPageTokens = 64;

    private readonly Lock _gate = new();
    private readonly Dictionary<(int Parent, ulong Hash), List<Page>> _index = [];
    private readonly HashSet<Page> _live = new(ReferenceEqualityComparer.Instance);
    private long _clock;
    private int _nextId;
    private int _largest;
    private bool _disposed;

    /// <summary>A pool of pages on <paramref name="device"/> stored by <paramref name="layout"/>.</summary>
    /// <param name="device">Where the pages live (the model's device).</param>
    /// <param name="layout">How keys and values are stored.</param>
    /// <param name="pageTokens">The positions a page holds, or null for <see cref="DefaultPageTokens"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">A page of fewer than one position.</exception>
    public KeyValuePages(Device device, IKeyValueLayout layout, int? pageTokens = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(layout);
        int tokens = pageTokens ?? DefaultPageTokens;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokens, nameof(pageTokens));
        Device = device;
        Layout = layout;
        PageTokens = tokens;
    }

    /// <summary>
    /// The positions a page holds by default: <see cref="PageTokensVariable"/> (IDRAK_KV_PAGE_TOKENS) when set, else
    /// <see cref="StandardPageTokens"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The variable is not a whole number, 1 or more (the message names it).</exception>
    public static int DefaultPageTokens => ParsePageTokens(Environment.GetEnvironmentVariable(PageTokensVariable), PageTokensVariable) ?? StandardPageTokens;

    /// <summary>Whether generations share their prompts' pages by default: <see cref="SharePrefixVariable"/> (IDRAK_KV_SHARE_PREFIX) is "1", "true", "on" or "yes".</summary>
    public static bool DefaultSharePrefix => Environment.GetEnvironmentVariable(SharePrefixVariable)?.Trim().ToLowerInvariant() is "1" or "true" or "on" or "yes";

    /// <summary>A page's positions from <paramref name="value"/> (null or empty: none given), named <paramref name="where"/> in the error.</summary>
    /// <exception cref="ArgumentException">Not a whole number, 1 or more.</exception>
    public static int? ParsePageTokens(string? value, string where)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return int.TryParse(value.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int tokens) && tokens >= 1
            ? tokens
            : throw new ArgumentException($"{where}={value} is not a whole number of positions, 1 or more.");
    }

    /// <summary>Where the pages live.</summary>
    public Device Device { get; }

    /// <summary>How keys and values are stored.</summary>
    public IKeyValueLayout Layout { get; }

    /// <summary>The positions a page holds.</summary>
    public int PageTokens { get; }

    /// <summary>
    /// The most published pages kept while no context reads them, or null (the default): as many as the largest context
    /// opened from the pool held, so the pool keeps about one conversation for its next turn, as a kept cache does.
    /// </summary>
    public int? IdlePageLimit
    {
        get;
        set
        {
            if (value is < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "The idle pages kept are 0 or more.");
            }

            field = value;
            lock (_gate)
            {
                Evict();
            }
        }
    }

    /// <summary>What the pool holds now.</summary>
    public KeyValuePageStats Stats
    {
        get
        {
            lock (_gate)
            {
                return StatsLocked();
            }
        }
    }

    /// <summary>
    /// A new decoding context of one row and <paramref name="capacity"/> positions whose caches are paged in this pool,
    /// reading the published pages of the longest prompt that starts as <paramref name="ids"/> does, whole pages only and
    /// at most <paramref name="limit"/> positions: its <see cref="DecodingContext.Length"/> is the positions it starts
    /// with (a multiple of <see cref="PageTokens"/>, 0 when no page matches), and the caller feeds the rest.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The pool was disposed.</exception>
    public DecodingContext Open(int capacity, IReadOnlyList<int> ids, int limit)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var context = new DecodingContext(Device, 1, capacity, Layout);
        var table = new List<Page>();
        try
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                int most = Math.Min(Math.Min(limit, ids.Count), capacity) / PageTokens;
                Page? parent = null;
                for (int p = 0; p < most; p++)
                {
                    var page = FindLocked(parent, ids, p * PageTokens);
                    if (page is null)
                    {
                        break;
                    }

                    page.Refs++;
                    page.Used = ++_clock;
                    table.Add(page);
                    parent = page;
                }
            }

            context.AttachPages(this, table);
        }
        catch
        {
            Release(table);
            context.Dispose();
            throw;
        }

        Report(table.Count * PageTokens);
        return context;
    }

    /// <summary>
    /// Publishes the whole pages <paramref name="context"/> (opened from this pool) holds of <paramref name="ids"/> (the
    /// tokens of its positions, from the first): later prompts that start alike read them. A page another context
    /// published with the same tokens first is left as it is, and the pages after it are not published.
    /// </summary>
    /// <exception cref="ArgumentException">The context was not opened from this pool.</exception>
    public void Publish(DecodingContext context, IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(ids);
        var table = context.PagesOf(this);
        int whole = Math.Min(context.Length, ids.Count) / PageTokens;
        lock (_gate)
        {
            Page? parent = null;
            for (int p = 0; p < whole && p < table.Count; p++)
            {
                var page = table[p];
                if (!page.Published)
                {
                    if (FindLocked(parent, ids, p * PageTokens) is not null || parent is { Published: false })
                    {
                        break;                                               // published by another first: the chain ends here
                    }

                    page.Tokens = Slice(ids, p * PageTokens, PageTokens);
                    page.Parent = parent;
                    if (parent is not null)
                    {
                        (parent.Children ??= []).Add(page);
                    }

                    Bucket(parent, page.Tokens).Add(page);
                }

                page.Used = ++_clock;
                parent = page;
            }

            _largest = Math.Max(_largest, table.Count);
        }
    }

    /// <summary>Frees every published page no context reads (contexts keep theirs until disposed).</summary>
    public void ReleaseIdle()
    {
        lock (_gate)
        {
            foreach (var page in _live.Where(p => p.Refs == 0).ToList())
            {
                Forget(page);
            }
        }
    }

    /// <summary>Frees the idle pages and opens no more contexts; contexts opened before keep their pages until disposed.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
        }

        ReleaseIdle();
    }

    // A new page of this pool, read by one context.
    internal Page NewPage()
    {
        lock (_gate)
        {
            var page = new Page(_nextId++) { Refs = 1, Used = ++_clock };
            _live.Add(page);
            return page;
        }
    }

    // A context's own copy of `page` (read by others, or published), which it then writes: every layer's block copied
    // as it is; `page` loses that reader.
    internal Page CopyForWrite(Page page)
    {
        var copy = NewPage();
        try
        {
            foreach (var (owner, block) in page.Blocks)
            {
                var own = new KeyValueCache(block.Keys.Shape[0], PageTokens, block.HeadDim, Device, Layout);
                copy.Blocks[owner] = own;
                CopyRows(block, 0, own, 0, PageTokens);
            }
        }
        catch
        {
            Release([copy]);
            throw;
        }

        Release([page]);
        return copy;
    }

    // Whether a context that reads `page` must copy it before writing: others read it, or it is published.
    internal bool Shared(Page page)
    {
        lock (_gate)
        {
            return page.Published || page.Refs > 1;
        }
    }

    // A context stops reading `pages`: a page no context reads is freed, unless published (then idle, kept for a later prompt).
    internal void Release(IEnumerable<Page> pages)
    {
        lock (_gate)
        {
            foreach (var page in pages)
            {
                if (--page.Refs > 0)
                {
                    continue;
                }

                if (!page.Published || _disposed)
                {
                    Forget(page);
                }
            }

            Evict();
        }
    }

    // Rows [from, from + count) of every tensor of `source` to rows [to, to + count) of `target` (same rows and row width).
    internal static void CopyRows(KeyValueCache source, int from, KeyValueCache target, int to, int count)
    {
        Copy(source.Keys, target.Keys);
        Copy(source.Values, target.Values);
        if (source.KeyScales is not null && target.KeyScales is not null)
        {
            Copy(source.KeyScales, target.KeyScales);
            Copy(source.ValueScales!, target.ValueScales!);
        }

        void Copy(Tensor a, Tensor b)
        {
            int rows = a.Shape[0], aSlots = a.Shape[1], bSlots = b.Shape[1], inner = a.Size / (rows * aSlots);
            a.Backend.Copy2D(a.Storage, from * inner, aSlots * inner, b.Storage, to * inner, bSlots * inner, rows, count * inner, accumulate: false);
        }
    }

    // Publishes what the pool holds after a context opened (Telemetry.KeyValuePages, at the inference level).
    internal void Report(int mapped)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Inference))
        {
            var stats = Stats;
            Telemetry.KeyValuePages(new KeyValuePagesReported(Device.ToString(), stats.PageTokens, stats.Pages, stats.InUse, stats.Shared, stats.Idle, stats.Bytes,
                stats.SavedBytes, mapped));
        }
    }

    private KeyValuePageStats StatsLocked()
    {
        int inUse = 0, shared = 0, idle = 0;
        long bytes = 0, saved = 0;
        foreach (var page in _live)
        {
            long pageBytes = page.Bytes;
            bytes += pageBytes;
            if (page.Refs > 0)
            {
                inUse++;
            }
            else
            {
                idle++;
            }

            if (page.Refs > 1)
            {
                shared++;
                saved += (page.Refs - 1) * pageBytes;
            }
        }

        return new KeyValuePageStats(PageTokens, _live.Count, inUse, shared, idle, bytes, saved);
    }

    // The published page after `parent` holding ids[start, start + PageTokens), or null.
    private Page? FindLocked(Page? parent, IReadOnlyList<int> ids, int start)
    {
        if (start + PageTokens > ids.Count || !_index.TryGetValue((parent?.Id ?? -1, Hash(ids, start, PageTokens)), out var candidates))
        {
            return null;
        }

        foreach (var page in candidates)
        {
            if (ReferenceEquals(page.Parent, parent) && Same(page.Tokens!, ids, start))
            {
                return page;
            }
        }

        return null;
    }

    private List<Page> Bucket(Page? parent, int[] tokens)
    {
        var key = (parent?.Id ?? -1, Hash(tokens, 0, tokens.Length));
        if (!_index.TryGetValue(key, out var list))
        {
            _index[key] = list = [];
        }

        return list;
    }

    // Frees idle pages past the limit, least recently used first, leaves only (a page's later pages are found through it).
    private void Evict()
    {
        int limit = IdlePageLimit ?? _largest;
        int idle = _live.Count(p => p.Refs == 0);
        while (idle > limit)
        {
            Page? oldest = null;
            foreach (var page in _live)
            {
                if (page.Refs == 0 && page.Children is not { Count: > 0 } && (oldest is null || page.Used < oldest.Used))
                {
                    oldest = page;
                }
            }

            if (oldest is null)
            {
                return;
            }

            Forget(oldest);
            idle--;
        }
    }

    // Frees a page no context reads: out of the index and its parent's children, its blocks disposed.
    private void Forget(Page page)
    {
        if (!_live.Remove(page))
        {
            return;
        }

        if (page.Tokens is { } tokens && _index.TryGetValue((page.Parent?.Id ?? -1, Hash(tokens, 0, tokens.Length)), out var list))
        {
            list.Remove(page);
            if (list.Count == 0)
            {
                _index.Remove((page.Parent?.Id ?? -1, Hash(tokens, 0, tokens.Length)));
            }
        }

        page.Parent?.Children?.Remove(page);
        foreach (var block in page.Blocks.Values)
        {
            block.Dispose();
        }

        page.Blocks.Clear();
    }

    // ids[start, start + count).
    private static int[] Slice(IReadOnlyList<int> ids, int start, int count)
    {
        var slice = new int[count];
        for (int i = 0; i < count; i++)
        {
            slice[i] = ids[start + i];
        }

        return slice;
    }

    private static bool Same(int[] tokens, IReadOnlyList<int> ids, int start)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            if (tokens[i] != ids[start + i])
            {
                return false;
            }
        }

        return true;
    }

    // FNV-1a over the tokens (a key for the index only: candidates are compared token by token).
    private static ulong Hash(IReadOnlyList<int> ids, int start, int count)
    {
        ulong hash = 14695981039346656037UL;
        for (int i = start; i < start + count; i++)
        {
            hash = (hash ^ (uint)ids[i]) * 1099511628211UL;
        }

        return hash;
    }

    // One page: every layer's block of PageTokens positions, by the layer that owns the cache.
    internal sealed class Page(int id)
    {
        public int Id { get; } = id;

        public Dictionary<object, KeyValueCache> Blocks { get; } = new(ReferenceEqualityComparer.Instance);

        public int Refs { get; set; }

        public long Used { get; set; }

        public Page? Parent { get; set; }

        public List<Page>? Children { get; set; }

        // Its tokens once published (then never written again), else null.
        public int[]? Tokens { get; set; }

        public bool Published => Tokens is not null;

        public long Bytes => Blocks.Values.Sum(b => b.Bytes);
    }
}
