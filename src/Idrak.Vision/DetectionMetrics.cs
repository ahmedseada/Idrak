// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// COCO's mean average precision for boxes ("coco" in <see cref="VisionMetrics"/>), as pycocotools' <c>COCOeval</c>
/// computes it: per image and class the detections (at most <see cref="VisionMetricOptions.MaxDetections"/> an image, the
/// best first) are matched greedily to the unmatched truths of highest IoU at each threshold 0.5, 0.55, ..., 0.95; crowd
/// and difficult truths are matched last and make their detections count neither way (a crowd's IoU is the share of the
/// detection inside it). Precision is made monotone and read at 101 recall points. Each image is matched as it is added
/// (its overlaps computed once for every threshold and area range); only its detections' scores and outcomes are kept,
/// eight bytes a detection and area range.
/// </summary>
/// <remarks>
/// Values: <c>map</c> (IoU 0.5:0.95), <c>map50</c>, <c>map75</c>, <c>map_small</c>, <c>map_medium</c>, <c>map_large</c>
/// (objects under 32², up to 96², above; the area is the truth's <see cref="ObjectAnnotation.Area"/>, else its box's),
/// <c>mar100</c> (recall at IoU 0.5:0.95), then <c>ap/CLASS</c> per class. A class with no truth that counts is left out of
/// the means; a mean over nothing is NaN (pycocotools' -1).
/// </remarks>
public sealed class CocoAveragePrecision : IDetectionMetric
{
    private const int ThresholdCount = 10;
    private static readonly double[] Thresholds = [.. Enumerable.Range(0, ThresholdCount).Select(i => 0.5 + 0.05 * i)];
    private static readonly (double Low, double High)[] Areas = [(0, 1e10), (0, 32 * 32), (32 * 32, 96 * 96), (96 * 96, 1e10)];

    // numpy's spacing(1) and finfo(float64).eps, which pycocotools and the VOC devkit add to divisions.
    internal const double MachineEpsilon = 2.220446049250313e-16;

    private readonly VisionMetricOptions _options;
    private readonly Dictionary<int, ClassRecord> _classes = [];

    // Per class and area range: each kept detection's score (in the order pycocotools concatenates them) and its outcome
    // per threshold as bits (bit t: matched at threshold t; bit 16 + t: ignored at threshold t), and the number of truths
    // that count. Eight bytes a detection and area range.
    private sealed class ClassRecord
    {
        public readonly List<float>[] Scores = [.. Areas.Select(_ => new List<float>())];
        public readonly List<int>[] Outcomes = [.. Areas.Select(_ => new List<int>())];
        public readonly int[] Truths = new int[Areas.Length];
    }

    /// <summary>A COCO metric with <paramref name="options"/> (<see cref="VisionMetricOptions.MaxDetections"/>, <see cref="VisionMetricOptions.ClassNames"/>).</summary>
    public CocoAveragePrecision(VisionMetricOptions? options = null)
    {
        _options = options ?? new();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_options.MaxDetections);
    }

    /// <inheritdoc />
    public string Name => "coco";

    /// <inheritdoc />
    public void Reset() => _classes.Clear();

    /// <inheritdoc />
    public void Add(IReadOnlyList<Detection> predictions, IReadOnlyList<ObjectAnnotation> truths)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        ArgumentNullException.ThrowIfNull(truths);
        var detectionsByClass = new Dictionary<int, List<Detection>>();
        for (int i = 0; i < predictions.Count; i++)
        {
            (detectionsByClass.TryGetValue(predictions[i].Class, out var list) ? list : detectionsByClass[predictions[i].Class] = []).Add(predictions[i]);
        }

        var truthsByClass = new Dictionary<int, List<ObjectAnnotation>>();
        foreach (var t in truths)
        {
            (truthsByClass.TryGetValue(t.Class, out var list) ? list : truthsByClass[t.Class] = []).Add(t);
        }

        foreach (int c in detectionsByClass.Keys.Union(truthsByClass.Keys))
        {
            var record = _classes.TryGetValue(c, out var r) ? r : _classes[c] = new ClassRecord();
            var dts = detectionsByClass.GetValueOrDefault(c) ?? [];
            if (dts.Count > 1)
            {
                // The best first, equal scores in the order given (numpy's stable mergesort), at most MaxDetections.
                int[] order = [.. Enumerable.Range(0, dts.Count)];
                Array.Sort(order, (a, b) => dts[b].Score.CompareTo(dts[a].Score) is var byScore and not 0 ? byScore : a.CompareTo(b));
                dts = [.. order.Take(_options.MaxDetections).Select(i => dts[i])];
            }

            Evaluate(record, dts, truthsByClass.GetValueOrDefault(c) ?? []);
        }
    }

    // pycocotools' evaluateImg for one image and class, every area range, the overlaps computed once.
    private static void Evaluate(ClassRecord record, List<Detection> dts, List<ObjectAnnotation> gts)
    {
        int d = dts.Count, g = gts.Count;
        var iou = new double[d * g];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < g; j++)
            {
                iou[i * g + j] = Overlap(dts[i].Box, gts[j].Box, gts[j].Crowd);
            }
        }

        var order = new int[g];
        var ignoredTruth = new bool[g];
        var taken = new bool[g];
        for (int a = 0; a < Areas.Length; a++)
        {
            var (low, high) = Areas[a];
            bool Ignore(ObjectAnnotation t) => t.Crowd || t.Difficult || AreaOf(t) < low || AreaOf(t) > high;

            // Truths that count first, the others after, each in the order given (numpy's stable mergesort).
            int k = 0;
            for (int j = 0; j < g; j++)
            {
                if (!Ignore(gts[j]))
                {
                    order[k++] = j;
                }
            }

            record.Truths[a] += k;
            for (int j = 0; j < g; j++)
            {
                if (Ignore(gts[j]))
                {
                    order[k++] = j;
                }
            }

            for (int j = 0; j < g; j++)
            {
                ignoredTruth[j] = Ignore(gts[order[j]]);
            }

            var outcomes = new int[d];
            for (int t = 0; t < ThresholdCount; t++)
            {
                Array.Clear(taken);
                for (int i = 0; i < d; i++)
                {
                    double best = Math.Min(Thresholds[t], 1 - 1e-10);
                    int m = -1;
                    for (int j = 0; j < g; j++)
                    {
                        if (taken[j] && !gts[order[j]].Crowd)
                        {
                            continue;
                        }

                        if (m > -1 && !ignoredTruth[m] && ignoredTruth[j])
                        {
                            break;
                        }

                        double overlap = iou[i * g + order[j]];
                        if (overlap < best)
                        {
                            continue;
                        }

                        (best, m) = (overlap, j);
                    }

                    if (m == -1)
                    {
                        double area = dts[i].Box.Width * (double)dts[i].Box.Height;
                        outcomes[i] |= area < low || area > high ? 1 << (16 + t) : 0;
                        continue;
                    }

                    outcomes[i] |= 1 << t | (ignoredTruth[m] ? 1 << (16 + t) : 0);
                    taken[m] = true;
                }
            }

            for (int i = 0; i < d; i++)
            {
                record.Scores[a].Add(dts[i].Score);
                record.Outcomes[a].Add(outcomes[i]);
            }
        }
    }

    private static double AreaOf(ObjectAnnotation g) => g.Area ?? g.Box.Width * (double)g.Box.Height;

    // IoU, or for a crowd the share of the detection inside it (pycocotools' iou with iscrowd), in double precision.
    private static double Overlap(BoundingBox d, BoundingBox g, bool crowd)
    {
        double iw = Math.Min((double)d.X + d.Width, (double)g.X + g.Width) - Math.Max(d.X, g.X);
        double ih = Math.Min((double)d.Y + d.Height, (double)g.Y + g.Height) - Math.Max(d.Y, g.Y);
        if (iw <= 0 || ih <= 0)
        {
            return 0;
        }

        double inter = iw * ih, areaD = (double)d.Width * d.Height, union = crowd ? areaD : areaD + (double)g.Width * g.Height - inter;
        return inter / union;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, double> Compute()
    {
        var classes = _classes.Keys.Order().ToList();
        // [area][class][threshold]: precision at 101 recall points averaged (AP), and the final recall; NaN when no truth counts.
        var ap = new double[Areas.Length][][];
        var recall = new double[Areas.Length][][];
        for (int a = 0; a < Areas.Length; a++)
        {
            ap[a] = new double[classes.Count][];
            recall[a] = new double[classes.Count][];
            for (int k = 0; k < classes.Count; k++)
            {
                (ap[a][k], recall[a][k]) = Accumulate(_classes[classes[k]], a);
            }
        }

        double Mean(IEnumerable<double> values)
        {
            var valid = values.Where(v => !double.IsNaN(v)).ToList();
            return valid.Count == 0 ? double.NaN : valid.Average();
        }

        var result = new Dictionary<string, double>
        {
            ["map"] = Mean(ap[0].SelectMany(t => t)),
            ["map50"] = Mean(ap[0].Select(t => t[0])),
            ["map75"] = Mean(ap[0].Select(t => t[5])),
            ["map_small"] = Mean(ap[1].SelectMany(t => t)),
            ["map_medium"] = Mean(ap[2].SelectMany(t => t)),
            ["map_large"] = Mean(ap[3].SelectMany(t => t)),
            ["mar100"] = Mean(recall[0].SelectMany(t => t)),
        };
        for (int k = 0; k < classes.Count; k++)
        {
            result["ap/" + ClassName(_options.ClassNames, classes[k])] = Mean(ap[0][k]);
        }

        return result;
    }

    internal static string ClassName(IReadOnlyList<string>? names, int c) =>
        names is not null && (uint)c < (uint)names.Count ? names[c] : c.ToString(CultureInfo.InvariantCulture);

    // pycocotools' accumulate for one class and area range: per threshold, the AP over 101 recall points and the last recall.
    private static (double[] Ap, double[] Recall) Accumulate(ClassRecord record, int a)
    {
        var ap = new double[ThresholdCount];
        var recall = new double[ThresholdCount];
        int truths = record.Truths[a];
        if (truths == 0)
        {
            Array.Fill(ap, double.NaN);
            Array.Fill(recall, double.NaN);
            return (ap, recall);
        }

        var scores = record.Scores[a];
        var outcomes = record.Outcomes[a];
        int n = scores.Count;
        int[] order = [.. Enumerable.Range(0, n)];
        Array.Sort(order, (x, y) => scores[y].CompareTo(scores[x]) is var byScore and not 0 ? byScore : x.CompareTo(y));
        var rc = new double[n];
        var pr = new double[n];
        for (int t = 0; t < ThresholdCount; t++)
        {
            int count = 0;
            double tp = 0, fp = 0;
            foreach (int i in order)
            {
                int outcome = outcomes[i];
                if ((outcome >> (16 + t) & 1) != 0)
                {
                    continue;                                                  // ignored: neither right nor wrong
                }

                if ((outcome >> t & 1) != 0)
                {
                    tp++;
                }
                else
                {
                    fp++;
                }

                rc[count] = tp / truths;
                pr[count++] = tp / (fp + tp + MachineEpsilon);
            }

            recall[t] = count > 0 ? rc[count - 1] : 0;
            for (int i = count - 1; i > 0; i--)
            {
                pr[i - 1] = Math.Max(pr[i - 1], pr[i]);
            }

            // numpy's searchsorted(rc, r / 100, side="left") for r = 0 .. 100, walking up the recalls once.
            double sum = 0;
            int at = 0;
            for (int r = 0; r <= 100; r++)
            {
                double threshold = r == 100 ? 1.0 : r * 0.01;                 // numpy's linspace(0, 1, 101)
                while (at < count && rc[at] < threshold)
                {
                    at++;
                }

                sum += at < count ? pr[at] : 0;
            }

            ap[t] = sum / 101;
        }

        return (ap, recall);
    }
}

/// <summary>
/// Pascal VOC's average precision at IoU above 0.5 ("voc" and "voc07" in <see cref="VisionMetrics"/>), as the devkit's
/// <c>voc_eval</c>: per class every detection, most confident first, is matched to the truth of its image it overlaps most;
/// a match above 0.5 is right when that truth is not yet taken, wrong when it is, and counts neither way when the truth is
/// difficult (or a crowd). AP over every recall point (VOC 2010 to 2012), or 11 points (VOC 2007).
/// </summary>
/// <remarks>Values: <c>map</c> (the mean over classes with a truth that counts), then <c>ap/CLASS</c> per class.</remarks>
public sealed class VocAveragePrecision : IDetectionMetric
{
    private readonly VisionMetricOptions _options;
    private readonly bool _elevenPoints;
    private readonly Dictionary<int, (List<(double Score, int Outcome)> Detections, int Truths)> _classes = [];

    /// <summary>A VOC metric with <paramref name="options"/>; <paramref name="elevenPoints"/> for VOC 2007's AP.</summary>
    public VocAveragePrecision(VisionMetricOptions? options = null, bool elevenPoints = false)
    {
        _options = options ?? new();
        _elevenPoints = elevenPoints;
    }

    /// <inheritdoc />
    public string Name => _elevenPoints ? "voc07" : "voc";

    /// <inheritdoc />
    public void Reset() => _classes.Clear();

    /// <inheritdoc />
    /// <remarks>
    /// Matching is per image, so it is done here (in the order of the scores, as the devkit's global order visits an
    /// image's detections); only each detection's score and outcome (right 1, wrong 0, neither -1) is kept.
    /// </remarks>
    public void Add(IReadOnlyList<Detection> predictions, IReadOnlyList<ObjectAnnotation> truths)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        ArgumentNullException.ThrowIfNull(truths);
        foreach (int c in predictions.Select(p => p.Class).Concat(truths.Select(t => t.Class)).Distinct())
        {
            var gts = truths.Where(t => t.Class == c).ToList();
            var taken = new bool[gts.Count];
            var entry = _classes.TryGetValue(c, out var e) ? e : ([], 0);
            entry.Truths += gts.Count(g => !g.Difficult && !g.Crowd);
            foreach (var d in predictions.Where(p => p.Class == c).Select((p, i) => (p, i)).OrderByDescending(x => x.p.Score).ThenBy(x => x.i).Select(x => x.p))
            {
                double best = double.NegativeInfinity;
                int at = -1;
                for (int g = 0; g < gts.Count; g++)
                {
                    double overlap = d.Box.IntersectionOverUnion(gts[g].Box);
                    if (overlap > best)
                    {
                        (best, at) = (overlap, g);
                    }
                }

                int outcome = 0;
                if (best > 0.5)
                {
                    outcome = gts[at].Difficult || gts[at].Crowd ? -1 : !taken[at] ? 1 : 0;
                    taken[at] |= outcome == 1;
                }

                entry.Detections.Add((d.Score, outcome));
            }

            _classes[c] = entry;
        }
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, double> Compute()
    {
        var perClass = new SortedDictionary<int, double>();
        foreach (var (c, (detections, truths)) in _classes)
        {
            perClass[c] = truths == 0 ? double.NaN : AveragePrecision(detections, truths);
        }

        var valid = perClass.Values.Where(v => !double.IsNaN(v)).ToList();
        var result = new Dictionary<string, double> { ["map"] = valid.Count == 0 ? double.NaN : valid.Average() };
        foreach (var (c, ap) in perClass)
        {
            result["ap/" + CocoAveragePrecision.ClassName(_options.ClassNames, c)] = ap;
        }

        return result;
    }

    private double AveragePrecision(List<(double Score, int Outcome)> detections, int truths)
    {
        double tp = 0, fp = 0;
        var recall = new List<double>();
        var precision = new List<double>();
        foreach (var (_, outcome) in detections.Select((d, i) => (d, i)).OrderByDescending(x => x.d.Score).ThenBy(x => x.i).Select(x => x.d))
        {
            tp += outcome == 1 ? 1 : 0;
            fp += outcome == 0 ? 1 : 0;
            recall.Add(tp / truths);
            precision.Add(tp / Math.Max(tp + fp, CocoAveragePrecision.MachineEpsilon));
        }

        if (_elevenPoints)
        {
            double ap = 0;
            for (int i = 0; i <= 10; i++)
            {
                double t = i / 10.0, p = 0;
                for (int k = 0; k < recall.Count; k++)
                {
                    p = recall[k] >= t ? Math.Max(p, precision[k]) : p;
                }

                ap += p / 11;
            }

            return ap;
        }

        double[] mrec = [0, .. recall, 1], mpre = [0, .. precision, 0];
        for (int i = mpre.Length - 1; i > 0; i--)
        {
            mpre[i - 1] = Math.Max(mpre[i - 1], mpre[i]);
        }

        double sum = 0;
        for (int i = 1; i < mrec.Length; i++)
        {
            if (mrec[i] != mrec[i - 1])
            {
                sum += (mrec[i] - mrec[i - 1]) * mpre[i];
            }
        }

        return sum;
    }
}
