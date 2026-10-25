// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Vision.Abstractions;

/// <summary>
/// A measure of a detector or a segmenter over a test set: add its answers image by image, then <see cref="Compute"/>.
/// <see cref="IDetectionMetric"/> takes detections, <see cref="ISegmentationMetric"/> masks.
/// </summary>
public interface IVisionMetric
{
    /// <summary>The name it is registered under (<c>coco</c>).</summary>
    string Name { get; }

    /// <summary>Forgets every image added.</summary>
    void Reset();

    /// <summary>
    /// The values over every image added so far, by name, the metric's headline value first (<c>map</c> for detection,
    /// <c>miou</c> for segmentation). A value that has no images to stand on is NaN.
    /// </summary>
    IReadOnlyDictionary<string, double> Compute();
}

/// <summary>A detection metric: the detections of each image against its true objects.</summary>
public interface IDetectionMetric : IVisionMetric
{
    /// <summary>
    /// Adds one image: the detector's <paramref name="predictions"/> and the <paramref name="truths"/> (crowd and difficult
    /// objects count neither way, as COCO and Pascal VOC have it).
    /// </summary>
    void Add(IReadOnlyList<Detection> predictions, IReadOnlyList<ObjectAnnotation> truths);
}

/// <summary>A segmentation metric: each image's predicted mask against its true one.</summary>
public interface ISegmentationMetric : IVisionMetric
{
    /// <summary>Adds one image's predicted and true masks (the same size).</summary>
    void Add(SegmentationMask predicted, SegmentationMask expected);
}

/// <summary>What a metric is created with (<see cref="IVisionMetricKind"/>, <see cref="VisionMetrics.Create"/>); each reads what it needs.</summary>
public sealed record VisionMetricOptions
{
    /// <summary>The number of classes (segmentation: required; detection: 0 to take the classes from the images).</summary>
    public int Classes { get; init; }

    /// <summary>Segmentation: pixels of this true class are left out (as a "void" label), or null for none.</summary>
    public int? IgnoreClass { get; init; }

    /// <summary>Detection: the most detections of an image that count, the best first (COCO's 100).</summary>
    public int MaxDetections { get; init; } = 100;

    /// <summary>The class names, for the per-class values' keys (<c>ap/cat</c>); the indices when null.</summary>
    public IReadOnlyList<string>? ClassNames { get; init; }
}

/// <summary>
/// A kind of vision metric, registered in <see cref="VisionMetrics"/> under its <see cref="IKind.Name"/>: what it measures
/// (<see cref="Task"/>, known without making one) and a new metric for its options
/// (<see cref="IKind{TRequest, T}.Create"/>: an <see cref="IDetectionMetric"/> for <see cref="ImageTask.Detection"/>, an
/// <see cref="ISegmentationMetric"/> for <see cref="ImageTask.Segmentation"/>).
/// </summary>
public interface IVisionMetricKind : IKind<VisionMetricOptions, IVisionMetric>
{
    /// <summary>What its metrics measure: <see cref="ImageTask.Detection"/> or <see cref="ImageTask.Segmentation"/>.</summary>
    ImageTask Task { get; }
}

/// <summary>
/// The detection and segmentation metrics, by name. The library's: "coco" (COCO's mean average precision: 101 recall
/// points, IoU 0.5 to 0.95 by 0.05, with AP50, AP75, small / medium / large objects and AR100, as pycocotools computes
/// them), "voc" (Pascal VOC's AP at IoU above 0.5, every recall point, as the VOC 2010 to 2012 devkit), "voc07" (the same
/// with VOC 2007's 11 recall points) and "miou" (segmentation: each class's intersection over union, their mean, pixel
/// accuracy, mean class accuracy and frequency-weighted IoU). Register another under a new name, or under a library name
/// to replace it: the library's stays behind it until <see cref="Unregister"/>.
/// </summary>
public static class VisionMetrics
{
    private static readonly KindTable<IVisionMetricKind> Registry = BuiltIn();

    private static KindTable<IVisionMetricKind> BuiltIn()
    {
        var table = new KindTable<IVisionMetricKind>(nameof(VisionMetrics), "Vision metric", "add it with VisionMetrics.Register", Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new Kind("coco", "COCO's mean average precision (IoU 0.5 to 0.95), AP50, AP75, by size, AR100", ImageTask.Detection, o => new CocoAveragePrecision(o)));
        table.RegisterDefault(new Kind("voc", "Pascal VOC's AP at IoU 0.5, every recall point", ImageTask.Detection, o => new VocAveragePrecision(o, elevenPoints: false)));
        table.RegisterDefault(new Kind("voc07", "Pascal VOC 2007's AP at IoU 0.5, 11 recall points", ImageTask.Detection, o => new VocAveragePrecision(o, elevenPoints: true)));
        table.RegisterDefault(new Kind("miou", "each class's IoU and their mean, pixel and class accuracy, frequency-weighted IoU", ImageTask.Segmentation,
            o => new SegmentationMetrics(o.Classes > 0 ? o.Classes : throw new ArgumentException("A segmentation metric needs the number of classes (VisionMetricOptions.Classes)."),
                o.IgnoreClass, o.ClassNames)));
        return table;
    }

    /// <summary>Registers <paramref name="kind"/> under its name; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVisionMetricKind kind) => Registry.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's metric <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered metric names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The metric kind <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names those that are.</exception>
    public static IVisionMetricKind Get(ReadOnlySpan<char> name) => Registry.TryGet(name, out var kind) ? kind
        : throw new NotSupportedException($"No vision metric '{name}' is registered ({string.Join(", ", Registry.Names)}); add it with VisionMetrics.Register.");

    /// <summary>The library's metric kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IVisionMetricKind? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the metric <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's metric <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/>: the library's, fed the same images, answers from then on), or whether it only runs
    /// beside the library's (<see cref="SlotPolicy.Shadow"/>: the library's answers, the values are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>A new metric <paramref name="name"/>.</summary>
    public static IVisionMetric Create(string name, VisionMetricOptions? options = null) => Get(name).Create(options ?? new());

    /// <summary>A new detection metric <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">The metric is not a detection metric (its kind says so before one is made).</exception>
    public static IDetectionMetric Detection(string name, VisionMetricOptions? options = null)
    {
        var kind = Get(name);
        return kind.Task == ImageTask.Detection && kind.Create(options ?? new()) is IDetectionMetric metric ? metric
            : throw new ArgumentException($"The vision metric '{name}' does not measure detections.");
    }

    /// <summary>A new segmentation metric <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">The metric is not a segmentation metric (its kind says so before one is made).</exception>
    public static ISegmentationMetric Segmentation(string name, VisionMetricOptions options)
    {
        var kind = Get(name);
        return kind.Task == ImageTask.Segmentation && kind.Create(options) is ISegmentationMetric metric ? metric
            : throw new ArgumentException($"The vision metric '{name}' does not measure segmentation masks.");
    }

    // An app's metric under its policy: the library's is fed the same images, so it can answer when the app's fails
    // (FallBack) or be compared with it (Shadow).
    private static IVisionMetricKind Guard(Slot slot, IVisionMetricKind app, IVisionMetricKind library) => new Kind(app.Name, app.Summary, app.Task, options =>
    {
        var mine = app.Create(options);
        return mine switch
        {
            IDetectionMetric detection => new GuardedDetection(slot, detection, (IDetectionMetric)library.Create(options)),
            ISegmentationMetric segmentation => new GuardedSegmentation(slot, segmentation, (ISegmentationMetric)library.Create(options)),
            _ => mine,
        };
    });

    // A metric kind as a function and its task.
    private sealed class Kind(string name, string summary, ImageTask task, Func<VisionMetricOptions, IVisionMetric> create) : IVisionMetricKind
    {
        public string Name => name;

        public string Summary => summary;

        public ImageTask Task => task;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IVisionMetric Create(VisionMetricOptions request) => create(request);
    }

    private abstract class Guarded(Slot slot, IVisionMetric app, IVisionMetric library) : IVisionMetric
    {
        private bool _broken;

        public string Name => app.Name;

        public void Reset()
        {
            library.Reset();
            app.Reset();
            _broken = false;
        }

        public IReadOnlyDictionary<string, double> Compute() => _broken ? library.Compute() : slot.Call(app.Compute, library.Compute, (a, b) =>
            a.Count == b.Count && a.All(p => b.TryGetValue(p.Key, out double v) && (Math.Abs(v - p.Value) <= 1e-6 || double.IsNaN(v) && double.IsNaN(p.Value))) ? null
                : $"values {string.Join(", ", a.Select(p => $"{p.Key}={p.Value:G6}"))} and {string.Join(", ", b.Select(p => $"{p.Key}={p.Value:G6}"))}");

        protected void Both(Action toApp, Action toLibrary)
        {
            toLibrary();
            if (_broken)
            {
                return;
            }

            try
            {
                toApp();
            }
            catch (Exception e) when (slot.Policy == SlotPolicy.Shadow && e is not OperationCanceledException || slot.Failed(e))
            {
                _broken = true;
            }
        }
    }

    private sealed class GuardedDetection(Slot slot, IDetectionMetric app, IDetectionMetric library) : Guarded(slot, app, library), IDetectionMetric
    {
        public void Add(IReadOnlyList<Detection> predictions, IReadOnlyList<ObjectAnnotation> truths) => Both(() => app.Add(predictions, truths), () => library.Add(predictions, truths));
    }

    private sealed class GuardedSegmentation(Slot slot, ISegmentationMetric app, ISegmentationMetric library) : Guarded(slot, app, library), ISegmentationMetric
    {
        public void Add(SegmentationMask predicted, SegmentationMask expected) => Both(() => app.Add(predicted, expected), () => library.Add(predicted, expected));
    }
}
