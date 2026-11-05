// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

// A decoding context whose caches are paged in a KeyValuePages pool (plan 37, K0): its positions' keys and values live in
// pages, some of them read by other contexts too. While a step runs, the layer that attends gets a working cache of the
// context's slots, its pages' rows copied in (the same values the unpaged cache holds, so attention runs the same kernels
// on the same values); when the next layer asks for its cache, or the step ends, the step's new rows go back to this
// context's own pages (a page others read is copied first) and the working cache is freed: one layer's working cache at
// a time.
public sealed partial class DecodingContext
{
    private KeyValuePages? _pagePool;
    private List<KeyValuePages.Page>? _pageTable;

    /// <summary>The pool this context's caches are paged in (<see cref="KeyValuePages.Open"/>), or null for an unpaged context.</summary>
    public KeyValuePages? Pages => _pagePool;

    /// <summary>The device memory of the pages this context reads (shared ones counted whole), 0 for an unpaged context.</summary>
    public long PageBytes => _pageTable?.Sum(p => p.Bytes) ?? 0;

    /// <summary>The pages this context reads, in position order (0 for an unpaged context).</summary>
    public int PageCount => _pageTable?.Count ?? 0;

    // Pages the context in `pool` with `table` read: it starts at their positions.
    internal void AttachPages(KeyValuePages pool, List<KeyValuePages.Page> table)
    {
        if (_pagePool is not null || Length > 0 || _caches.Count > 0 || Batch != 1)
        {
            throw new InvalidOperationException("Pages are given to a new context of one row, before its first step.");
        }

        (_pagePool, _pageTable) = (pool, table);
        Length = table.Count * pool.PageTokens;
        Position.FillInPlace(Length);
    }

    // The pages this context reads in `pool`.
    internal List<KeyValuePages.Page> PagesOf(KeyValuePages pool) =>
        ReferenceEquals(_pagePool, pool) ? _pageTable! : throw new ArgumentException("The context was not opened from this pool (KeyValuePages.Open).", nameof(pool));

    // The working cache of `owner` for this step: its pages' rows for the positions so far, at the context's slots.
    private KeyValueCache PagedCacheFor(object owner, int rows, int headDim)
    {
        if (_caches.TryGetValue(owner, out var current))
        {
            return current;
        }

        FlushPages(InStep ? _steps : 0);                                 // the layer before: its rows to the pages
        if (_slots == 0)
        {
            _slots = SlotsFor(Length + Math.Max(1, _steps));
        }

        int size = _pagePool!.PageTokens;
        var work = new KeyValueCache(rows, _slots, headDim, Device, Layout);
        try
        {
            for (int p = 0; p * size < Length; p++)
            {
                var block = _pageTable![p].Blocks.GetValueOrDefault(owner)
                    ?? throw new InvalidOperationException($"Page {p} of the context holds no rows for this layer: its pages come from another model's contexts.");
                KeyValuePages.CopyRows(block, 0, work, p * size, Math.Min(size, Length - p * size));
            }
        }
        catch
        {
            work.Dispose();
            throw;
        }

        _caches[owner] = work;
        return work;
    }

    // Each working cache's rows of the step's `steps` positions to this context's pages (copied first where others read
    // them), then the working caches freed.
    private void FlushPages(int steps)
    {
        if (_pagePool is not { } pool || _caches.Count == 0)
        {
            return;
        }

        try
        {
            foreach (var (owner, work) in _caches)
            {
                int size = pool.PageTokens, end = Math.Min(Length + steps, Capacity);
                for (int position = Length; position < end;)
                {
                    int p = position / size, at = position % size, count = Math.Min(size - at, end - position);
                    while (_pageTable!.Count <= p)
                    {
                        _pageTable.Add(pool.NewPage());
                    }

                    var page = _pageTable[p];
                    if (pool.Shared(page))
                    {
                        _pageTable[p] = page = pool.CopyForWrite(page);
                    }

                    if (!page.Blocks.TryGetValue(owner, out var block))
                    {
                        using (Device.Backend.LongLivedAllocations())
                        {
                            page.Blocks[owner] = block = new KeyValueCache(work.Keys.Shape[0], size, work.HeadDim, Device, Layout);
                        }
                    }

                    KeyValuePages.CopyRows(work, position, block, at, count);
                    position += count;
                }
            }
        }
        finally
        {
            DropWorkingCaches();
        }
    }

    // Frees the working caches without keeping their rows (a step withdrawn, the context reset or disposed).
    private void DropWorkingCaches()
    {
        if (_pagePool is null)
        {
            return;
        }

        foreach (var work in _caches.Values)
        {
            work.Dispose();
        }

        _caches.Clear();
    }

    // Stops reading the pages from position `length` on (whole pages past it), after a truncation or a reset.
    private void ReleasePagesFrom(int length)
    {
        if (_pagePool is not { } pool)
        {
            return;
        }

        DropWorkingCaches();
        int keep = (length + pool.PageTokens - 1) / pool.PageTokens;
        if (_pageTable!.Count > keep)
        {
            var released = _pageTable.GetRange(keep, _pageTable.Count - keep);
            _pageTable.RemoveRange(keep, released.Count);
            pool.Release(released);
        }
    }

    // A paged context refuses what its pages cannot follow.
    private void RefusePaged(string what)
    {
        if (_pagePool is not null)
        {
            throw new NotSupportedException($"{what} on a context paged in a shared pool (KeyValuePages, {KeyValuePages.SharePrefixVariable}): "
                + "open an unpaged context for it (IDRAK_KV_SHARE_PREFIX=0, TextGenerator.SharePrefix false).");
        }
    }
}
