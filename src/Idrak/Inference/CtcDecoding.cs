// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Inference.Abstractions;

namespace Idrak.Inference;

/// <summary>The library's CTC decoders, registered in <see cref="CtcDecoders"/> as "greedy" and "beam".</summary>
internal static class CtcDecoding
{
    /// <summary>Best path: the most likely class at each step (the first on a tie), repeats collapsed and blanks dropped.</summary>
    public static IReadOnlyList<CtcHypothesis> Greedy(ReadOnlyMemory<float> logProbs, int steps, int classes, CtcDecodeOptions options)
    {
        var values = logProbs.Span;
        var labels = new List<int>();
        double score = 0;
        int previous = -1;
        for (int t = 0; t < steps; t++)
        {
            var row = values.Slice(t * classes, classes);
            int best = 0;
            for (int c = 1; c < classes; c++)
            {
                if (row[c] > row[best])
                {
                    best = c;
                }
            }

            score += row[best];
            if (best != options.Blank && best != previous)
            {
                labels.Add(best);
            }

            previous = best;
        }

        return [new CtcHypothesis([.. labels], (float)score)];
    }

    // A prefix is a node of a trie (its parent prefix and last label), so extending one is O(1) and prefixes share memory.
    private readonly record struct Node(int Parent, int Label);

    // A prefix's probability in log space split by how its paths end: on a blank, or on its last label.
    private struct Score(double blank, double label)
    {
        public double Blank = blank;
        public double Label = label;

        public readonly double Total => LogAdd(Blank, Label);
    }

    /// <summary>
    /// Prefix beam search: after each step the <see cref="CtcDecodeOptions.BeamWidth"/> most likely prefixes are kept, each
    /// scored over all the paths that read as it (ending in a blank or not, so "a a" and "a - a" stay apart); a step's
    /// classes are pruned to <see cref="CtcDecodeOptions.TopClasses"/> and <see cref="CtcDecodeOptions.MinLogProbability"/>.
    /// A step's candidates are keyed by (prefix, label) and only the kept ones become trie nodes, so memory grows with
    /// the beam, not with beam × classes.
    /// </summary>
    public static IReadOnlyList<CtcHypothesis> BeamSearch(ReadOnlyMemory<float> logProbs, int steps, int classes, CtcDecodeOptions options)
    {
        var values = logProbs.Span;
        int blank = options.Blank;
        var nodes = new List<Node> { new(-1, -1) };                          // node 0: the empty prefix
        var children = new Dictionary<(int Parent, int Label), int>();
        var beam = new List<(int Node, Score Score)> { (0, new Score(0, double.NegativeInfinity)) };
        var next = new Dictionary<(int Parent, int Label), Score>();
        var candidates = new List<int>(classes);

        void Add((int Parent, int Label) key, double blankPart, double labelPart)
        {
            ref var score = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(next, key, out bool found);
            if (!found)
            {
                score = new Score(double.NegativeInfinity, double.NegativeInfinity);
            }

            score.Blank = LogAdd(score.Blank, blankPart);
            score.Label = LogAdd(score.Label, labelPart);
        }

        for (int t = 0; t < steps; t++)
        {
            var row = values.Slice(t * classes, classes);
            Candidates(row, options, candidates);
            next.Clear();
            foreach (var (prefix, score) in beam)
            {
                double total = score.Total;
                var self = (nodes[prefix].Parent, nodes[prefix].Label);
                int last = self.Label;
                foreach (int c in candidates)
                {
                    double p = row[c];
                    if (c == blank)
                    {
                        Add(self, total + p, double.NegativeInfinity);
                    }
                    else if (c == last)
                    {
                        Add(self, double.NegativeInfinity, score.Label + p);                       // the label continues: "a" stays "a"
                        Add((prefix, c), double.NegativeInfinity, score.Blank + p);               // after a blank: "a - a" reads "aa"
                    }
                    else
                    {
                        Add((prefix, c), double.NegativeInfinity, total + p);
                    }
                }
            }

            beam.Clear();
            foreach (var (key, score) in Best(next, options.BeamWidth))
            {
                int node = key.Parent < 0 ? 0 : children.TryGetValue(key, out int known) ? known : -1;
                if (node < 0)
                {
                    node = nodes.Count;
                    nodes.Add(new Node(key.Parent, key.Label));
                    children[key] = node;
                }

                beam.Add((node, score));
            }
        }

        var results = new List<CtcHypothesis>();
        foreach (var (prefix, score) in beam.Take(options.Results))
        {
            var labels = new List<int>();
            for (int node = prefix; node > 0; node = nodes[node].Parent)
            {
                labels.Add(nodes[node].Label);
            }

            labels.Reverse();
            results.Add(new CtcHypothesis([.. labels], (float)score.Total));
        }

        return results;
    }

    // The classes of a step that extend prefixes: every class, or the most likely ones (the best always among them).
    private static void Candidates(ReadOnlySpan<float> row, CtcDecodeOptions options, List<int> candidates)
    {
        candidates.Clear();
        int best = 0;
        for (int c = 1; c < row.Length; c++)
        {
            if (row[c] > row[best])
            {
                best = c;
            }
        }

        for (int c = 0; c < row.Length; c++)
        {
            if (c == best || options.MinLogProbability is not { } floor || row[c] >= floor)
            {
                candidates.Add(c);
            }
        }

        if (options.TopClasses is { } top && candidates.Count > top)
        {
            var scores = row.ToArray();
            candidates.Sort((a, b) => scores[b].CompareTo(scores[a]) is var order && order != 0 ? order : a.CompareTo(b));
            candidates.RemoveRange(top, candidates.Count - top);
        }
    }

    // The `count` most likely prefixes, the most likely first (on a tie the one with the older parent, then the smaller
    // label, so results are stable): a heap of the best `count` seen, O(n log count).
    private static List<KeyValuePair<(int Parent, int Label), Score>> Best(Dictionary<(int Parent, int Label), Score> prefixes, int count)
    {
        var heap = new PriorityQueue<KeyValuePair<(int Parent, int Label), Score>, (double Total, int Parent, int Label)>(count + 1, Worse);
        foreach (var entry in prefixes)
        {
            heap.Enqueue(entry, (entry.Value.Total, entry.Key.Parent, entry.Key.Label));
            if (heap.Count > count)
            {
                heap.Dequeue();                                                       // the worst kept so far
            }
        }

        var best = new List<KeyValuePair<(int Parent, int Label), Score>>(heap.Count);
        while (heap.Count > 0)
        {
            best.Add(heap.Dequeue());
        }

        best.Reverse();
        return best;
    }

    // Orders worse first: a lower total, then a younger parent, then a larger label.
    private static readonly Comparer<(double Total, int Parent, int Label)> Worse = Comparer<(double Total, int Parent, int Label)>.Create((a, b) =>
        a.Total != b.Total ? a.Total.CompareTo(b.Total) : a.Parent != b.Parent ? b.Parent.CompareTo(a.Parent) : b.Label.CompareTo(a.Label));

    private static double LogAdd(double a, double b)
    {
        if (double.IsNegativeInfinity(a))
        {
            return b;
        }

        if (double.IsNegativeInfinity(b))
        {
            return a;
        }

        return a > b ? a + Math.Log(1 + Math.Exp(b - a)) : b + Math.Log(1 + Math.Exp(a - b));
    }
}
