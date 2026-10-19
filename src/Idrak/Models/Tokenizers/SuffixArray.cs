// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// Suffix arrays in linear time (SA-IS: Nong, Zhang and Chan's induced sorting, as Yuta Mori's sais implements it) and the
/// internal nodes of a text's suffix tree from it (esaxx's enhanced suffix array, the one SentencePiece's unigram trainer
/// seeds its vocabulary from). The text is an array of symbols 0 to <c>k - 1</c>; callers rank a text's code points
/// first, so <c>k</c> is the number of distinct symbols rather than the whole Unicode range (the order is kept, so the
/// suffix array is the same).
/// </summary>
internal static class SuffixArray
{
    /// <summary>The suffix array of <paramref name="text"/> (symbols 0 to <paramref name="k"/> - 1) into <paramref name="sa"/> (its length or more).</summary>
    public static void Build(int[] text, int n, int[] sa, int k)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        ArgumentOutOfRangeException.ThrowIfLessThan(k, 1);
        if (n <= 1)
        {
            if (n == 1)
            {
                sa[0] = 0;
            }

            return;
        }

        SuffixSort(text, 0, sa, 0, n, k);
    }

    /// <summary>
    /// The internal nodes of the suffix tree of <paramref name="text"/> from its suffix array (esaxx's <c>suffixtree</c>):
    /// node i covers the suffixes <c>sa[left[i]] .. sa[right[i] - 1]</c>, which share their first <c>depth[i]</c> symbols.
    /// <paramref name="left"/>, <paramref name="right"/> and <paramref name="depth"/> have the text's length; returns the
    /// number of nodes. The nodes come in esaxx's order, so a reader that stops early stops where SentencePiece would.
    /// </summary>
    public static int InternalNodes(int[] text, int n, int[] sa, int[] left, int[] right, int[] depth)
    {
        if (n == 0)
        {
            return 0;
        }

        // Psi (in left), then the permuted longest-common-prefix array (in right), then the LCP array (in left again).
        int[] psi = left;
        psi[sa[0]] = sa[n - 1];
        for (int i = 1; i < n; i++)
        {
            psi[sa[i]] = sa[i - 1];
        }

        int[] plcp = right;
        int h = 0;
        for (int i = 0; i < n; i++)
        {
            int j = psi[i];
            while (i + h < n && j + h < n && text[i + h] == text[j + h])
            {
                h++;
            }

            plcp[i] = h;
            if (h > 0)
            {
                h--;
            }
        }

        int[] lcp = left;
        for (int i = 0; i < n; i++)
        {
            lcp[i] = plcp[sa[i]];
        }

        lcp[0] = -1;
        var stack = new Stack<(int Start, int Height)>();
        stack.Push((-1, -1));
        int nodes = 0;
        for (int i = 0; ; i++)
        {
            var current = (Start: i, Height: i == n ? -1 : lcp[i]);
            var candidate = stack.Peek();
            while (candidate.Height > current.Height)
            {
                if (i - candidate.Start > 1)
                {
                    left[nodes] = candidate.Start;
                    right[nodes] = i;
                    depth[nodes] = candidate.Height;
                    nodes++;
                }

                current.Start = candidate.Start;
                stack.Pop();
                candidate = stack.Peek();
            }

            if (candidate.Height < current.Height)
            {
                stack.Push(current);
            }

            if (i == n)
            {
                break;
            }

            stack.Push((i, n - sa[i] + 1));
        }

        return nodes;
    }

    // The counts of each symbol.
    private static void Counts(int[] t, int to, int n, int[] c, int co, int k)
    {
        Array.Clear(c, co, k);
        for (int i = 0; i < n; i++)
        {
            c[co + t[to + i]]++;
        }
    }

    // The start (or end) of each symbol's bucket.
    private static void Buckets(int[] c, int co, int[] b, int bo, int k, bool end)
    {
        int sum = 0;
        if (end)
        {
            for (int i = 0; i < k; i++)
            {
                sum += c[co + i];
                b[bo + i] = sum;
            }
        }
        else
        {
            for (int i = 0; i < k; i++)
            {
                sum += c[co + i];
                b[bo + i] = sum - c[co + i];
            }
        }
    }

    // Induces the order of the L-type, then the S-type suffixes (sais' induceSA). C and B may be one array: the counts are
    // then taken again before each pass.
    private static void Induce(int[] t, int to, int[] sa, int[] c, int co, int[] b, int bo, int n, int k)
    {
        bool shared = ReferenceEquals(c, b) && co == bo;
        if (shared)
        {
            Counts(t, to, n, c, co, k);
        }

        Buckets(c, co, b, bo, k, end: false);
        int j = n - 1;
        int c1 = t[to + j];
        int at = b[bo + c1];
        sa[at++] = (0 < j && t[to + j - 1] < c1) ? ~j : j;
        for (int i = 0; i < n; i++)
        {
            j = sa[i];
            sa[i] = ~j;
            if (0 < j)
            {
                int c0 = t[to + --j];
                if (c0 != c1)
                {
                    b[bo + c1] = at;
                    at = b[bo + (c1 = c0)];
                }

                sa[at++] = (0 < j && t[to + j - 1] < c1) ? ~j : j;
            }
        }

        if (shared)
        {
            Counts(t, to, n, c, co, k);
        }

        Buckets(c, co, b, bo, k, end: true);
        c1 = 0;
        at = b[bo + c1];
        for (int i = n - 1; 0 <= i; i--)
        {
            if (0 < (j = sa[i]))
            {
                int c0 = t[to + --j];
                if (c0 != c1)
                {
                    b[bo + c1] = at;
                    at = b[bo + (c1 = c0)];
                }

                sa[--at] = (j == 0 || t[to + j - 1] > c1) ? ~j : j;
            }
            else
            {
                sa[i] = ~j;
            }
        }
    }

    // sais' suffixsort: the text is t[to .. to + n); the suffix array goes to sa[0 .. n), with fs free entries after it.
    private static void SuffixSort(int[] t, int to, int[] sa, int fs, int n, int k)
    {
        int[] cArray, bArray;
        int co, bo;

        // Stage 1: sort the LMS substrings.
        if (fs < k)
        {
            cArray = bArray = new int[k];
            co = bo = 0;
        }
        else
        {
            cArray = sa;
            co = n;
            bArray = sa;
            bo = k <= fs - k ? n + k : n;
        }

        Counts(t, to, n, cArray, co, k);
        Buckets(cArray, co, bArray, bo, k, end: true);
        Array.Clear(sa, 0, n);
        int c = 0, c1 = t[to + n - 1];
        for (int i = n - 2; 0 <= i; i--, c1 = t[to + i + 1])
        {
            int c0 = t[to + i];
            if (c0 < c1 + c)
            {
                c = 1;
            }
            else if (c != 0)
            {
                sa[--bArray[bo + c1]] = i + 1;
                c = 0;
            }
        }

        Induce(t, to, sa, cArray, co, bArray, bo, n, k);

        // Compact the sorted LMS substrings into the first m entries.
        int m = 0;
        for (int i = 0; i < n; i++)
        {
            int p = sa[i];
            if (0 < p && t[to + p - 1] > t[to + p])
            {
                int c0 = t[to + p];
                int j = p + 1;
                while (j < n && c0 == t[to + j])
                {
                    j++;
                }

                if (j < n && c0 < t[to + j])
                {
                    sa[m++] = p;
                }
            }
        }

        Array.Clear(sa, m, n >> 1);

        // The length of every LMS substring.
        {
            int j = n;
            c = 0;
            c1 = t[to + n - 1];
            for (int i = n - 2; 0 <= i; i--, c1 = t[to + i + 1])
            {
                int c0 = t[to + i];
                if (c0 < c1 + c)
                {
                    c = 1;
                }
                else if (c != 0)
                {
                    sa[m + ((i + 1) >> 1)] = j - i - 1;
                    j = i + 1;
                    c = 0;
                }
            }
        }

        // Name the LMS substrings.
        int name = 0;
        {
            int q = n, qlength = 0;
            for (int i = 0; i < m; i++)
            {
                int p = sa[i], plength = sa[m + (p >> 1)];
                bool differ = true;
                if (plength == qlength)
                {
                    int j = 0;
                    while (j < plength && t[to + p + j] == t[to + q + j])
                    {
                        j++;
                    }

                    if (j == plength)
                    {
                        differ = false;
                    }
                }

                if (differ)
                {
                    name++;
                    q = p;
                    qlength = plength;
                }

                sa[m + (p >> 1)] = name;
            }
        }

        // Stage 2: solve the reduced problem, recursing while names repeat.
        if (name < m)
        {
            int ra = n + fs - m;                                                 // the reduced text, at the end of sa
            for (int i = m + (n >> 1) - 1, j = m - 1; m <= i; i--)
            {
                if (sa[i] != 0)
                {
                    sa[ra + j--] = sa[i] - 1;
                }
            }

            SuffixSort(sa, ra, sa, fs + n - (m * 2), m, name);
            {
                int j = m - 1;
                c = 0;
                c1 = t[to + n - 1];
                for (int i = n - 2; 0 <= i; i--, c1 = t[to + i + 1])
                {
                    int c0 = t[to + i];
                    if (c0 < c1 + c)
                    {
                        c = 1;
                    }
                    else if (c != 0)
                    {
                        sa[ra + j--] = i + 1;
                        c = 0;
                    }
                }
            }

            for (int i = 0; i < m; i++)
            {
                sa[i] = sa[ra + sa[i]];
            }
        }

        // Stage 3: induce the result from the sorted LMS suffixes.
        if (fs < k)
        {
            cArray = bArray = new int[k];
            co = bo = 0;
        }
        else
        {
            cArray = sa;
            co = n;
            bArray = sa;
            bo = k <= fs - k ? n + k : n;
        }

        Counts(t, to, n, cArray, co, k);
        Buckets(cArray, co, bArray, bo, k, end: true);
        Array.Clear(sa, m, n - m);
        for (int i = m - 1; 0 <= i; i--)
        {
            int j = sa[i];
            sa[i] = 0;
            sa[--bArray[bo + t[to + j]]] = j;
        }

        Induce(t, to, sa, cArray, co, bArray, bo, n, k);
    }
}
