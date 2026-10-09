// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 13, step 4: training detection and segmentation. Box losses (IoU, GIoU, DIoU, CIoU), smooth L1, focal and dice
// losses with gradients; IoU-threshold and Hungarian matching; COCO and VOC mean average precision and mean IoU. Checked
// against torchvision, PyTorch, SciPy and pycocotools (tests/Idrak.Tests/data/vision-detection, written by
// tools/pytorch/vision_detection_reference.py), finite differences and the testing kit's suites.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionDetectionGroup =
    [
        ("vision detection: IoU, GIoU, DIoU and CIoU box losses match torchvision's (losses and gradients, the same box, boxes apart, one inside the other), through VisionLosses with every reduction", BoxLossesMatchTorchvision),
        ("vision detection: smooth L1 and L1, sigmoid focal (α 0.25 γ 2, no α γ 0.5) and soft dice match PyTorch, gradients included", DenseLossesMatchPyTorch),
        ("vision detection: gradient: every registered loss against finite differences; the testing kit's loss suite passes for each and fails a wrong one", VisionLossGradients),
        ("vision detection: Hungarian matching matches SciPy's linear_sum_assignment, threshold matching torchvision's Matcher (with and without low-quality matches); the kit's matcher suites pass", MatchingMatchesReferences),
        ("vision detection: COCO mAP (0.5:0.95, 50, 75, small, medium, large, AR100) matches pycocotools, VOC AP (all points and 11) the devkit's voc_eval, mean IoU numpy's confusion matrix; the kit's metric suites pass", MetricsMatchReferences),
        ("vision detection: registries: VisionLosses, BoxMatchers and VisionMetrics name their built-ins and refuse unknown names; an app's loss falls back or is shadowed; RLE counts match pycocotools'", VisionDetectionRegistries),
    ];

    private static readonly Lazy<JsonObject> DetectionReference = new(() => JsonNode.Parse(File.ReadAllText(
        Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vision-detection", "reference.json")))!.AsObject());

    private static float[] FloatArray(JsonNode? node) => [.. node!.AsArray().Select(v => (float)v!)];

    private static void BoxLossesMatchTorchvision(Device device)
    {
        var c = DetectionReference.Value["box_losses"]!;
        float[] predicted = FloatArray(c["predicted"]), target = FloatArray(c["target"]);
        int n = predicted.Length / 4;
        foreach (string name in new[] { "iou", "giou", "diou", "ciou" })
        {
            using var scope = new TensorScope();
            var p = Tensor.From(predicted, [n, 4], device, requiresGrad: true);
            var t = Tensor.From(target, [n, 4], device);
            var losses = VisionLosses.Compute(name, p, t, new VisionLossOptions { Reduction = LossReduction.None });
            AssertClose(FloatArray(c[name]!["losses"]), losses.ToArray(), 1e-5f, $"{name} losses");
            losses.Sum().Backward();
            AssertClose(FloatArray(c[name]!["grad"]), p.Grad!.ToArray(), 1e-4f, $"{name} gradient");
            float sum = FloatArray(c[name]!["losses"]).Sum();
            Check(MathF.Abs(VisionLosses.Compute(name, p, t, new VisionLossOptions { Reduction = LossReduction.Sum }).Item() - sum) <= 1e-4f * Math.Max(1, sum)
                  && MathF.Abs(VisionLosses.Compute(name, p, t).Item() - sum / n) <= 1e-5f, $"{name}: sum and mean");
        }

        // [..., 4] boxes keep their leading shape; the losses are [...] with gradients there too.
        using (var scope = new TensorScope())
        {
            var p = Tensor.From(predicted, [2, n / 2, 4], device);
            var losses = DetectionLosses.BoxIou(p, Tensor.From(target, [2, n / 2, 4], device), BoxOverlap.GIoU, LossReduction.None);
            Check(losses.Shape.SequenceEqual([2, n / 2]), $"[2, {n / 2}, 4] boxes give losses {Tensor.FormatShape(losses.Shape)}");
        }
    }

    private static void DenseLossesMatchPyTorch(Device device)
    {
        var r = DetectionReference.Value;
        using var scope = new TensorScope();
        var s = r["smooth_l1"]!;
        var a = Tensor.From(FloatArray(s["predicted"]), [6, 4], device, requiresGrad: true);
        var b = Tensor.From(FloatArray(s["target"]), [6, 4], device);
        var smooth = DetectionLosses.SmoothL1(a, b, 1f, LossReduction.None);
        AssertClose(FloatArray(s["beta1"]), smooth.ToArray(), 1e-6f, "smooth L1, β 1");
        smooth.Sum().Backward();
        AssertClose(FloatArray(s["grad_beta1"]), a.Grad!.ToArray(), 1e-6f, "smooth L1 gradient");
        AssertClose(FloatArray(s["beta05"]), VisionLosses.Compute("smooth-l1", a, b, new() { Beta = 0.5f, Reduction = LossReduction.None }).ToArray(), 1e-6f, "smooth L1, β 0.5");
        AssertClose(FloatArray(s["l1"]), VisionLosses.Compute("l1", a, b, new() { Reduction = LossReduction.None }).ToArray(), 1e-6f, "L1");

        var f = r["focal"]!;
        float[] logits = FloatArray(f["logits"]), targets = FloatArray(f["targets"]);
        foreach (string key in new[] { "a025_g2", "none_g05" })
        {
            var x = Tensor.From(logits, [5, 7], device, requiresGrad: true);
            var options = new VisionLossOptions { Alpha = (float)f[key]!["alpha"]!, Gamma = (float)f[key]!["gamma"]!, Reduction = LossReduction.None };
            var losses = VisionLosses.Compute("focal", x, Tensor.From(targets, [5, 7], device), options);
            AssertClose(FloatArray(f[key]!["losses"]), losses.ToArray(), 1e-5f, $"focal {key}");
            losses.Mean().Backward();
            AssertClose(FloatArray(f[key]!["grad_mean"]), x.Grad!.ToArray(), 1e-5f, $"focal {key} gradient");
        }

        var d = r["dice"]!;
        var z = Tensor.From(FloatArray(d["logits"]), [2, 3, 4, 5], device, requiresGrad: true);
        var probabilities = z.Permute(0, 2, 3, 1).Softmax().Permute(0, 3, 1, 2);
        var dice = SegmentationLosses.Dice(probabilities, Tensor.From(FloatArray(d["targets"]), [2, 3, 4, 5], device), 1f, false, LossReduction.None);
        AssertClose(FloatArray(d["losses"]), dice.ToArray(), 1e-5f, "dice losses [N, classes]");
        dice.Mean().Backward();
        AssertClose(FloatArray(d["grad_logits_mean"]), z.Grad!.ToArray(), 1e-5f, "dice gradient through the softmax");
        var batch = VisionLosses.Compute("dice", probabilities, Tensor.From(FloatArray(d["targets"]), [2, 3, 4, 5], device), new() { Batch = true, Reduction = LossReduction.None });
        Check(batch.Shape.SequenceEqual([3]), $"batch dice per class: {Tensor.FormatShape(batch.Shape)}");
    }

    private static void VisionLossGradients(Device device)
    {
        float[] Boxes(Random random, int n)
        {
            var b = new float[n * 4];
            for (int i = 0; i < n; i++)
            {
                float x = random.NextSingle() * 10, y = random.NextSingle() * 10;
                (b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]) = (x, y, x + 1 + random.NextSingle() * 8, y + 1 + random.NextSingle() * 8);
            }

            return b;
        }

        var random = new Random(5);
        float[] target = Boxes(random, 6), start = Boxes(random, 6);
        foreach (string name in new[] { "iou", "giou", "diou", "ciou", "l1", "smooth-l1" })
        {
            GradCheckAt(device, start, [6, 4], x => VisionLosses.Compute(name, x, Tensor.From(target, [6, 4], device)), $"{name} gradient");
        }

        var labels = Enumerable.Range(0, 24).Select(i => i % 5 == 0 ? 1f : 0f).ToArray();
        GradCheckAt(device, [.. Enumerable.Range(0, 24).Select(i => MathF.Sin(i) * 3)], [4, 6], x => VisionLosses.Compute("focal", x, Tensor.From(labels, [4, 6], device)), "focal gradient");
        var onehot = Enumerable.Range(0, 2 * 3 * 4).Select(i => i / 4 % 3 == i % 4 % 3 ? 1f : 0f).ToArray();
        GradCheckAt(device, [.. Enumerable.Range(0, 24).Select(i => MathF.Cos(i))], [2, 3, 2, 2],
            x => VisionLosses.Compute("dice", x.Permute(0, 2, 3, 1).Softmax().Permute(0, 3, 1, 2), Tensor.From(onehot, [2, 3, 2, 2], device)), "dice gradient");

        // The kit's suite: every library loss passes it against itself as the reference; a loss that is off fails.
        foreach (var (name, input) in new[] { ("iou", VisionLossInput.Boxes), ("giou", VisionLossInput.Boxes), ("diou", VisionLossInput.Boxes), ("ciou", VisionLossInput.Boxes),
                     ("l1", VisionLossInput.Boxes), ("smooth-l1", VisionLossInput.Boxes), ("focal", VisionLossInput.Logits), ("dice", VisionLossInput.Probabilities) })
        {
            Func<Tensor, Tensor, Tensor> loss = (p, t) => VisionLosses.Get(name)(p, t, new() { Reduction = LossReduction.None });
            Func<Tensor, Tensor, Tensor> library = (p, t) => VisionLosses.Default(name)!(p, t, new() { Reduction = LossReduction.None });
            Conformance.Check(loss, new VisionLossSuite(input, library), new ContractCheckOptions { RandomCases = 6 }).ThrowIfFailed();
        }

        Func<Tensor, Tensor, Tensor> off = (p, t) => VisionLosses.Get("giou")(p, t, new() { Reduction = LossReduction.None }) * 1.01f;
        var report = Conformance.Check(off, new VisionLossSuite(VisionLossInput.Boxes, (p, t) => VisionLosses.Get("giou")(p, t, new() { Reduction = LossReduction.None })),
            new ContractCheckOptions { RandomCases = 2 });
        Check(!report.Passed && report.Failures.Any(f => f.Check.Contains("reference", StringComparison.Ordinal)), $"a loss 1% off passes:\n{report}");
    }

    // Autograd's gradient of f at `values` against central differences in double precision sums.
    private static void GradCheckAt(Device device, float[] values, int[] shape, Func<Tensor, Tensor> f, string what)
    {
        using var scope = new TensorScope();
        var x = Tensor.From(values, shape, device, requiresGrad: true);
        f(x).Backward();
        var analytic = x.Grad!.ToArray();
        const float H = 1e-3f;
        var numeric = new float[values.Length];
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < values.Length; i++)
            {
                var plus = (float[])values.Clone();
                var minus = (float[])values.Clone();
                plus[i] += H;
                minus[i] -= H;
                numeric[i] = (float)(((double)f(Tensor.From(plus, shape, device)).Item() - f(Tensor.From(minus, shape, device)).Item()) / (2 * H));
            }
        }

        AssertClose(numeric, analytic, 2e-2f, what);
    }

    private static void MatchingMatchesReferences(Device device)
    {
        _ = device;
        foreach (var c in DetectionReference.Value["hungarian"]!.AsArray())
        {
            int rows = (int)c!["rows"]!, columns = (int)c["columns"]!;
            double[] cost = [.. c["cost"]!.AsArray().Select(v => (double)v!)];
            float[] quality = [.. cost.Select(v => (float)-v)];
            var matches = BoxMatchers.Match(BoxMatchers.Hungarian, quality, rows, columns);
            int[] rowInd = [.. c["row_ind"]!.AsArray().Select(v => (int)v!)], colInd = [.. c["col_ind"]!.AsArray().Select(v => (int)v!)];
            var expected = Enumerable.Repeat(BoxMatchers.Negative, rows).ToArray();
            for (int k = 0; k < rowInd.Length; k++)
            {
                expected[rowInd[k]] = colInd[k];
            }

            Check(matches.SequenceEqual(expected), $"Hungarian {rows} x {columns}: [{string.Join(", ", matches)}], SciPy [{string.Join(", ", expected)}]");
            double total = Enumerable.Range(0, rows).Where(i => matches[i] >= 0).Sum(i => cost[i * columns + matches[i]]);
            Check(Math.Abs(total - (double)c["total"]!) < 1e-5, $"Hungarian {rows} x {columns}: total {total}, SciPy {(double)c["total"]!}");
        }

        var m = DetectionReference.Value["matcher"]!;
        int predictions = (int)m["predictions"]!, truths = (int)m["truths"]!;
        float[] iou = FloatArray(m["quality"]);
        foreach (var (key, low) in new[] { ("plain", false), ("low_quality", true) })
        {
            var matches = BoxMatchers.Match(BoxMatchers.Threshold, iou, predictions, truths, new BoxMatchOptions { AllowLowQuality = low });
            int[] expected = [.. m[key]!.AsArray().Select(v => (int)v!)];
            Check(matches.SequenceEqual(expected), $"threshold matching ({key}): [{string.Join(", ", matches)}], torchvision [{string.Join(", ", expected)}]");
        }

        // Matrices from boxes: IoU agrees with the boxes' own, GIoU is within [-1, 1] and below the IoU.
        var a = new[] { new BoundingBox(0, 0, 10, 10), new BoundingBox(20, 20, 5, 5) };
        var b = new[] { new BoundingBox(5, 0, 10, 10), new BoundingBox(0, 0, 10, 10), new BoundingBox(100, 100, 1, 1) };
        var plainIou = BoxMatching.IouMatrix(a, b);
        var giou = BoxMatching.GeneralizedIouMatrix(a, b);
        Check(plainIou.Length == 6 && Math.Abs(plainIou[0] - 50f / 150) < 1e-6 && plainIou[1] == 1 && plainIou[2] == 0 && giou.All(v => v is >= -1 and <= 1)
              && giou.Zip(plainIou).All(p => p.First <= p.Second + 1e-6), $"IoU [{string.Join(", ", plainIou)}], GIoU [{string.Join(", ", giou)}]");
        var probabilities = new float[] { 0.9f, 0.1f, 0.2f, 0.8f };
        var set = BoxMatching.SetPredictionQuality(probabilities, 2, [new BoundingBox(0, 0, 10, 10), new BoundingBox(50, 50, 10, 10)], [1, 0],
            [new BoundingBox(50, 50, 10, 10), new BoundingBox(0, 0, 10, 10)], scale: 100);
        Check(BoxMatchers.Match(BoxMatchers.Hungarian, set, 2, 2).SequenceEqual([1, 0]), "set prediction quality pairs each output with its object");
        Check(BoxMatchers.Match(BoxMatchers.Hungarian, new[] { float.NegativeInfinity, 0.5f, float.NegativeInfinity, float.NegativeInfinity }, 2, 2).SequenceEqual([1, -1]),
            "forbidden pairs stay unmatched");

        Conformance.Check<Func<ReadOnlyMemory<float>, int, int, int[]>>((q, p, g) => BoxMatchers.Get(BoxMatchers.Hungarian)(q, p, g, new()),
            new BoxMatcherSuite(BoxMatcherExpectation.Optimal)).ThrowIfFailed();
        foreach (bool low in new[] { false, true })
        {
            Conformance.Check<Func<ReadOnlyMemory<float>, int, int, int[]>>((q, p, g) => BoxMatchers.Get(BoxMatchers.Threshold)(q, p, g, new() { AllowLowQuality = low }),
                new BoxMatcherSuite(new BoxMatcherExpectation(false, 0.5f, 0.4f, low))).ThrowIfFailed();
        }

        // A greedy one-to-one matcher is not optimal: the suite says so.
        Func<ReadOnlyMemory<float>, int, int, int[]> greedy = (q, p, g) =>
        {
            var result = Enumerable.Repeat(-1, p).ToArray();
            var taken = new bool[g];
            for (int i = 0; i < p; i++)
            {
                int best = -1;
                for (int j = 0; j < g; j++)
                {
                    if (!taken[j] && !float.IsNegativeInfinity(q.Span[i * g + j]) && (best < 0 || q.Span[i * g + j] > q.Span[i * g + best]))
                    {
                        best = j;
                    }
                }

                if (best >= 0)
                {
                    (result[i], taken[best]) = (best, true);
                }
            }

            return result;
        };
        Check(!Conformance.Check(greedy, new BoxMatcherSuite(BoxMatcherExpectation.Optimal), new ContractCheckOptions { RandomCases = 30 }).Passed, "a greedy matcher passes as optimal");
    }

    private static void MetricsMatchReferences(Device device)
    {
        _ = device;
        // COCO: the same scenes pycocotools evaluated.
        var coco = DetectionReference.Value["coco"]!;
        var gt = coco["ground_truth"]!;
        var categories = gt["categories"]!.AsArray().Select(c => (int)c!["id"]!).ToList();
        var truthsByImage = gt["annotations"]!.AsArray().GroupBy(a => (int)a!["image_id"]!).ToDictionary(g => g.Key, g => g.Select(a =>
        {
            var b = FloatArray(a!["bbox"]);
            return new ObjectAnnotation(new BoundingBox(b[0], b[1], b[2], b[3]), categories.IndexOf((int)a["category_id"]!))
            {
                Area = (float)a["area"]!, Crowd = (int)a["iscrowd"]! != 0,
            };
        }).ToList());
        var detectionsByImage = coco["detections"]!.AsArray().GroupBy(d => (int)d!["image_id"]!).ToDictionary(g => g.Key, g => g.Select(d =>
        {
            var b = FloatArray(d!["bbox"]);
            return new Detection(new BoundingBox(b[0], b[1], b[2], b[3]), categories.IndexOf((int)d["category_id"]!), (float)d["score"]!);
        }).ToList());
        var metric = VisionMetrics.Detection("coco");
        foreach (int image in gt["images"]!.AsArray().Select(i => (int)i!["id"]!))
        {
            metric.Add(detectionsByImage.GetValueOrDefault(image, []), truthsByImage.GetValueOrDefault(image, []));
        }

        var values = metric.Compute();
        var stats = coco["stats"]!.AsArray().Select(v => (double)v!).ToArray();
        var names = coco["stats_names"]!.AsArray().Select(v => (string)v!).ToArray();
        foreach (string key in new[] { "map", "map50", "map75", "map_small", "map_medium", "map_large", "mar100" })
        {
            double expected = stats[Array.IndexOf(names, key)];
            Check(Math.Abs(values[key] - expected) < 1e-9, $"COCO {key}: {values[key]:R}, pycocotools {expected:R}");
        }

        Check(values.Keys.First() == "map" && values.ContainsKey("ap/2"), $"COCO values: {string.Join(", ", values.Keys)}");

        // VOC: the devkit's voc_eval per class, every recall point and 11 points.
        var voc = DetectionReference.Value["voc"]!;
        foreach (var (name, key) in new[] { ("voc", "all_points"), ("voc07", "eleven_points") })
        {
            var m = VisionMetrics.Detection(name);
            foreach (var image in voc["images"]!.AsArray())
            {
                var truths = image!["objects"]!.AsArray().Select(o =>
                {
                    var b = FloatArray(o!["box"]);
                    return new ObjectAnnotation(BoundingBox.FromCorners(b[0], b[1], b[2], b[3]), (int)o["class"]!) { Difficult = (bool)o["difficult"]! };
                }).ToList();
                var detections = image["detections"]!.AsArray().Select(d =>
                {
                    var b = FloatArray(d!["box"]);
                    return new Detection(BoundingBox.FromCorners(b[0], b[1], b[2], b[3]), (int)d["class"]!, (float)d["score"]!);
                }).ToList();
                m.Add(detections, truths);
            }

            var result = m.Compute();
            double[] expected = [.. voc[key]!.AsArray().Select(v => (double)v!)];
            for (int c = 0; c < expected.Length; c++)
            {
                Check(Math.Abs(result[$"ap/{c}"] - expected[c]) < 1e-6, $"{name} AP of class {c}: {result[$"ap/{c}"]:R}, voc_eval {expected[c]:R}");
            }

            Check(Math.Abs(result["map"] - expected.Average()) < 1e-6, $"{name} mAP {result["map"]}");
        }

        // Mean IoU with an ignored class.
        var s = DetectionReference.Value["miou"]!;
        int classes = (int)s["classes"]!, w = (int)s["width"]!, h = (int)s["height"]!;
        int[] truth = [.. s["truth"]!.AsArray().Select(v => (int)v!)], predicted = [.. s["predicted"]!.AsArray().Select(v => (int)v!)];
        var miou = VisionMetrics.Segmentation("miou", new VisionMetricOptions { Classes = classes, IgnoreClass = 255 });
        for (int i = 0; i < truth.Length / (w * h); i++)
        {
            // The masks' classes are counted by the metric's own; 255 is outside them, so the mask is given a larger class count.
            miou.Add(new SegmentationMask(predicted[(i * w * h)..((i + 1) * w * h)], w, h, 256), new SegmentationMask(truth[(i * w * h)..((i + 1) * w * h)], w, h, 256));
        }

        var scores = miou.Compute();
        foreach (string key in new[] { "miou", "pixel_accuracy", "mean_accuracy", "fwiou" })
        {
            Check(Math.Abs(scores[key] - (double)s[key]!) < 1e-12, $"segmentation {key}: {scores[key]:R}, numpy {(double)s[key]!:R}");
        }

        double[] classIou = [.. s["iou"]!.AsArray().Select(v => (double)v!)];
        Check(Enumerable.Range(0, classes).All(c => Math.Abs(scores[$"iou/{c}"] - classIou[c]) < 1e-12), "per-class IoU");

        // The kit's suites, each metric against itself as the reference (through the registry).
        foreach (string name in new[] { "coco", "voc", "voc07" })
        {
            double Run(IReadOnlyList<DetectionImage> images, VisionMetricFactory create)
            {
                var metric = (IDetectionMetric)create(new VisionMetricOptions());
                foreach (var image in images)
                {
                    metric.Add([.. image.Predictions.Select(p => new Detection(p.Box, p.Class, p.Score))], image.Truths);
                }

                return metric.Compute()["map"];
            }

            Conformance.Check<Func<IReadOnlyList<DetectionImage>, double>>(images => Run(images, VisionMetrics.Get(name)),
                new DetectionMetricSuite(images => Run(images, VisionMetrics.Default(name)!)), new ContractCheckOptions { RandomCases = 8 }).ThrowIfFailed();
        }

        Conformance.Check<Func<IReadOnlyList<SegmentationImage>, double>>(images =>
        {
            var metric = VisionMetrics.Segmentation("miou", new VisionMetricOptions { Classes = images[0].Classes });
            foreach (var image in images)
            {
                metric.Add(new SegmentationMask(image.Predicted, image.Width, image.Height, image.Classes), new SegmentationMask(image.Expected, image.Width, image.Height, image.Classes));
            }

            return metric.Compute()["miou"];
        }, new SegmentationMetricSuite()).ThrowIfFailed();

        // A metric that ignores scores (counts matches only) fails the ranking check.
        Func<IReadOnlyList<DetectionImage>, double> recall = images =>
        {
            int found = images.Sum(i => i.Truths.Count(t => !t.Difficult && i.Predictions.Any(p => p.Class == t.Class && p.Box.IntersectionOverUnion(t.Box) > 0.5f)));
            int total = images.Sum(i => i.Truths.Count(t => !t.Difficult));
            return total == 0 ? 0 : (double)found / total;
        };
        Check(!Conformance.Check(recall, new DetectionMetricSuite()).Passed, "a recall passes as an average precision");
    }

    private static void VisionDetectionRegistries(Device device)
    {
        Check(new[] { "iou", "giou", "diou", "ciou", "l1", "smooth-l1", "focal", "dice" }.All(VisionLosses.Names.Contains), $"losses: {string.Join(", ", VisionLosses.Names)}");
        Check(BoxMatchers.Names.Contains("iou-threshold") && BoxMatchers.Names.Contains("hungarian"), $"matchers: {string.Join(", ", BoxMatchers.Names)}");
        Check(new[] { "coco", "voc", "voc07", "miou" }.All(VisionMetrics.Names.Contains), $"metrics: {string.Join(", ", VisionMetrics.Names)}");
        foreach (var (what, act) in new (string, Action)[]
                 {
                     ("loss", () => VisionLosses.Get("yolo-box")), ("matcher", () => BoxMatchers.Get("sinkhorn")), ("metric", () => VisionMetrics.Create("lvis")),
                 })
        {
            try
            {
                act();
                throw new Exception($"an unknown {what} was found");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains("is registered", StringComparison.Ordinal), $"unknown {what}: {e.Message}");
            }
        }

        Check(VisionMetrics.Detection("coco") is CocoAveragePrecision && VisionMetrics.Segmentation("miou", new() { Classes = 3 }) is SegmentationMetrics, "metric kinds");
        try
        {
            VisionMetrics.Segmentation("coco", new() { Classes = 3 });
            throw new Exception("coco measured masks");
        }
        catch (ArgumentException)
        {
        }

        // An app's loss over a library name: Throw by default, FallBack answers with the library's, Shadow compares.
        using var scope = new TensorScope();
        var p = Tensor.From(new float[] { 0, 0, 4, 4 }, [1, 4], device);
        var t = Tensor.From(new float[] { 1, 1, 5, 5 }, [1, 4], device);
        float expected = VisionLosses.Compute("giou", p, t).Item();
        VisionLosses.Register("giou", (_, _, _) => throw new InvalidOperationException("app loss broke"));
        try
        {
            Check(VisionLosses.Origin("giou") != Overrides.Library, "the app's loss shadows the library's");
            try
            {
                VisionLosses.Compute("giou", p, t);
                throw new Exception("a failing app loss did not throw");
            }
            catch (InvalidOperationException)
            {
            }

            VisionLosses.SetPolicy("giou", SlotPolicy.FallBack);
            Check(MathF.Abs(VisionLosses.Compute("giou", p, t).Item() - expected) < 1e-6f, "FallBack answers with the library's loss");
            VisionLosses.SetPolicy("giou", SlotPolicy.Shadow, 1);
            Check(MathF.Abs(VisionLosses.Compute("giou", p, t).Item() - expected) < 1e-6f, "Shadow answers with the library's loss");
        }
        finally
        {
            VisionLosses.SetPolicy("giou", SlotPolicy.Throw);
            VisionLosses.Unregister("giou");
        }

        Check(VisionLosses.Origin("giou") == Overrides.Library && MathF.Abs(VisionLosses.Compute("giou", p, t).Item() - expected) < 1e-6f, "Unregister brings the library's loss back");

        // COCO's run-length codec.
        foreach (var c in DetectionReference.Value["rle"]!.AsArray())
        {
            int h = (int)c!["height"]!, w = (int)c["width"]!;
            byte[] pixels = [.. c["mask"]!.AsArray().Select(v => (byte)(int)v!)];
            var mask = ObjectMask.FromPixels(pixels, w, h);
            int[] runs = [.. c["runs"]!.AsArray().Select(v => (int)v!)];
            Check(mask.Counts!.SequenceEqual(runs), $"{w} x {h}: runs [{string.Join(", ", mask.Counts!)}], pycocotools [{string.Join(", ", runs)}]");
            Check(ObjectMask.EncodeCounts(runs) == (string)c["counts"]!, $"{w} x {h}: '{ObjectMask.EncodeCounts(runs)}', pycocotools '{(string)c["counts"]!}'");
            Check(ObjectMask.DecodeCounts((string)c["counts"]!).SequenceEqual(runs), $"{w} x {h}: decoding pycocotools' counts");
            Check(mask.Rasterize(w, h).SequenceEqual(pixels) && mask.Rasterize(w, h).Count(v => v != 0) == (int)c["area"]!, $"{w} x {h}: rasterized");
        }
    }
}
