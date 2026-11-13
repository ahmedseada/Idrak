// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// "sequence": each text line fills one group of regions (a grid's row) in reading order, its units its
/// whitespace-separated tokens: <c>take</c> says how many units each region index takes (a list, <c>1:rest</c>: one
/// region may take whatever the others leave, the others served from both ends; 1 each by default), <c>blank</c> is the
/// chance a region is left empty and <c>blank-at</c> which indexes may be (from 1: <c>3..18</c>, <c>1:4</c>; all by
/// default). A group of one region (a box's) takes the whole line. Lines beyond the groups, and groups beyond the lines,
/// draw nothing.
/// </summary>
internal sealed class SequenceBinding : ITextBinding
{
    private static readonly OptionKeys Own = new("take", "blank", "blank-at");

    public string Name => TextBindings.Sequence;

    public string Summary => "each line fills one group of regions in reading order, a unit a region (take: units per region, rest for one that takes what the others leave; blank: the chance a region is left empty; blank-at: which, from 1)";

    public IReadOnlySet<string> Keys => Own;

    public TextBindingSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        int[] take = [];
        int rest = -1;
        float blank = 0;
        IReadOnlySet<int>? blankAt = null;
        foreach (var (raw, value) in options)
        {
            string key = raw.ToLowerInvariant();
            string what = $"Page setting {key}";
            switch (key)
            {
                case "take":
                {
                    var items = PageOptions.List(what, value);
                    take = new int[items.Count];
                    rest = -1;
                    for (int i = 0; i < items.Count; i++)
                    {
                        if (items[i].Equals("rest", StringComparison.OrdinalIgnoreCase))
                        {
                            rest = rest < 0 ? i : throw new ArgumentException($"{what} has rest once at most, not '{value}'.");
                            take[i] = 0;
                        }
                        else
                        {
                            take[i] = PageOptions.Integer(what, items[i], 1, 100_000);
                        }
                    }

                    break;
                }

                case "blank":
                    blank = PageOptions.Chance(what, value);
                    break;
                case "blank-at":
                    blankAt = PageOptions.Indexes(what, value, 1, 100_000);
                    break;
                default:
                    throw new ArgumentException($"The text binding sequence does not take '{raw}'; it takes: {Own}.");
            }
        }

        return new Settings(options, take, rest, blank, blankAt);
    }

    public BoundText Bind(IReadOnlyList<string> lines, IReadOnlyList<ShapeRegion> regions, TextBindingSettings settings, int seed)
    {
        ArgumentNullException.ThrowIfNull(lines);
        ArgumentNullException.ThrowIfNull(regions);
        var s = settings as Settings ?? (Settings)Read(settings?.Options ?? PageOptions.None);
        var texts = new string[regions.Count];
        Array.Fill(texts, "");
        var label = new List<string>();

        // The groups in the order of their numbers, each group's regions in the order of their indexes.
        var groups = new SortedDictionary<int, List<int>>();
        for (int i = 0; i < regions.Count; i++)
        {
            if (!groups.TryGetValue(regions[i].Group, out var members))
            {
                groups[regions[i].Group] = members = [];
            }

            members.Add(i);
        }

        var random = new Random(seed);
        int line = 0;
        foreach (var (_, members) in groups)
        {
            if (line >= lines.Count)
            {
                break;
            }

            members.Sort((a, b) => regions[a].Index != regions[b].Index ? regions[a].Index.CompareTo(regions[b].Index) : a.CompareTo(b));
            string[] units = lines[line++].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var assigned = Assign(units, members.Count, s);
            var drawn = new List<string>(members.Count);
            for (int k = 0; k < members.Count; k++)
            {
                // Every region draws its number whether it may be left empty or not, so one does not move another.
                bool empty = random.NextDouble() < s.Blank && (s.BlankAt is null || s.BlankAt.Contains(k + 1));
                string text = empty ? "" : assigned[k];
                texts[members[k]] = text;
                if (text.Length > 0)
                {
                    drawn.Add(text);
                }
            }

            if (drawn.Count > 0)
            {
                label.Add(string.Join(" ", drawn));
            }
        }

        return new BoundText(texts, label);
    }

    // The units each of `count` regions takes, in reading order: from the start, those after a rest from the end.
    private static string[] Assign(string[] units, int count, Settings s)
    {
        var assigned = new string[count];
        Array.Fill(assigned, "");
        if (count == 1)
        {
            assigned[0] = string.Join(" ", units);
            return assigned;
        }

        int Count(int index) => index < s.Take.Length ? s.Take[index] : 1;
        int from = 0, to = units.Length;
        int rest = s.Rest < count ? s.Rest : -1;
        for (int k = 0; k < count && k != rest; k++)
        {
            int n = Math.Min(Count(k), to - from);
            assigned[k] = string.Join(" ", units, from, n);
            from += n;
        }

        if (rest >= 0)
        {
            for (int k = count - 1; k > rest; k--)
            {
                int n = Math.Min(Count(k), to - from);
                assigned[k] = string.Join(" ", units, to - n, n);
                to -= n;
            }

            assigned[rest] = string.Join(" ", units, from, to - from);
        }

        return assigned;
    }

    private sealed class Settings(IReadOnlyDictionary<string, string> given, int[] take, int rest, float blank, IReadOnlySet<int>? blankAt) : TextBindingSettings(given)
    {
        public int[] Take { get; } = take;

        public int Rest { get; } = rest;

        public float Blank { get; } = blank;

        public IReadOnlySet<int>? BlankAt { get; } = blankAt;
    }
}
