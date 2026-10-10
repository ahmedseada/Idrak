// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Optimizers;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// What <c>idrak train</c> keeps of an image run besides <see cref="RunSettings"/> (run.json's "image"): the model it
/// fine-tunes (or none: the builder network), the task, how the data is read, and the registry names of its
/// augmentations, loss, matcher, metric and decoder.
/// </summary>
internal sealed record ImageRunOptions
{
    /// <summary>The image model checkpoint fine-tuned through its family; null when the builder network is trained.</summary>
    public string? Model { get; init; }

    /// <summary>classification, segmentation or detection; null: from the model.</summary>
    public string? Task { get; init; }

    /// <summary>auto, folder (class folders), masks (images/ and masks/), or an AnnotationFormats name.</summary>
    public string DataFormat { get; init; } = "auto";

    /// <summary>The augmentation pipeline (Augmentations' text), or null.</summary>
    public string? Augment { get; init; }

    /// <summary>A VisionLosses name (or cross-entropy), or null for the task's default.</summary>
    public string? Loss { get; init; }

    /// <summary>A BoxMatchers name, or null for iou-threshold.</summary>
    public string? Matcher { get; init; }

    /// <summary>A VisionMetrics name (or accuracy), or null for the task's default.</summary>
    public string? Metric { get; init; }

    /// <summary>The evaluation data (read as the training data), or null to hold out --validation of it.</summary>
    public string? Eval { get; init; }

    /// <summary>The decoder of a detector (DetectionDecoders and DetectionHeads); null: the model's.</summary>
    public string? Decoder { get; init; }

    public JsonObject ToJson() => new()
    {
        ["model"] = Model,
        ["task"] = Task,
        ["data_format"] = DataFormat,
        ["augment"] = Augment,
        ["loss"] = Loss,
        ["matcher"] = Matcher,
        ["metric"] = Metric,
        ["eval"] = Eval,
        ["decoder"] = Decoder,
    };

    public static ImageRunOptions FromJson(JsonObject json) => new()
    {
        Model = (string?)json["model"],
        Task = (string?)json["task"],
        DataFormat = (string?)json["data_format"] ?? "auto",
        Augment = (string?)json["augment"],
        Loss = (string?)json["loss"],
        Matcher = (string?)json["matcher"],
        Metric = (string?)json["metric"],
        Eval = (string?)json["eval"],
        Decoder = (string?)json["decoder"],
    };

    /// <summary>The task named by --task (classify, segment, detect and their nouns), or null.</summary>
    public static ImageTask? ParseTask(string? text) => text?.ToLowerInvariant() switch
    {
        null => null,
        "classify" or "classification" => ImageTask.Classification,
        "segment" or "segmentation" => ImageTask.Segmentation,
        "detect" or "detection" => ImageTask.Detection,
        _ => throw new UsageException($"--task {text}: for images use classify, segment or detect."),
    };
}

/// <summary>
/// An image training run of <c>idrak train</c>: a builder network or an image model read through its family
/// (fine-tuned), on class folders (classification), image and mask folders or an annotated dataset (segmentation) or an
/// annotated dataset read by an <see cref="AnnotationFormats">annotation format</see> (detection). Samples are read,
/// augmented (<see cref="Augmentations"/>) and stretched to the network's input off the training thread
/// (<see cref="AugmentedImageLoader"/>); the loss is the task's (or a <see cref="VisionLosses"/> entry), a detector's
/// candidates come from its decoder's <see cref="DetectionHeads">head</see> and are matched by a
/// <see cref="BoxMatchers">matcher</see>; the evaluation data is scored by a <see cref="VisionMetrics">metric</see>. A
/// batch the device has no room for is split into micro-batches whose gradients add up (halved again as needed). The run
/// folder holds run.json, the checkpoints and log.jsonl as for any run; a builder network becomes a package (.ikm), a
/// fine-tuned model its weights (.ikw, read by <c>idrak predict --weights</c>).
/// </summary>
internal static class ImageTraining
{
    // The data of a run: images with their objects, classes or masks, read (and decoded) when the loader asks.
    private sealed class ImageSet
    {
        public required string Source { get; init; }

        public required string Layout { get; init; }

        public required IReadOnlyList<string> Files { get; init; }

        public required IReadOnlyList<string> Classes { get; set; }

        public required Func<int, AnnotatedImage> Read { get; set; }

        public int[]? ImageClasses { get; set; }

        public int Count => Files.Count;

        public ImageSet Subset(int[] indices)
        {
            var read = Read;
            var classes = ImageClasses;
            return new ImageSet
            {
                Source = Source, Layout = Layout, Classes = Classes, Files = [.. indices.Select(i => Files[i])], Read = i => read(indices[i]),
                ImageClasses = classes is null ? null : [.. indices.Select(i => classes[i])],
            };
        }
    }

    /// <summary>Runs <paramref name="settings"/> in <paramref name="folder"/>; continues from <paramref name="resume"/> weights after <paramref name="epochsDone"/> epochs.</summary>
    public static int Run(CommandContext context, RunSettings settings, string folder, string? resume = null, int epochsDone = 0, int? epochs = null)
    {
        try
        {
            return Train(context, settings, folder, resume, epochsDone, epochs);
        }
        catch (Exception e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
    }

    private static int Train(CommandContext context, RunSettings settings, string folder, string? resume, int epochsDone, int? epochs)
    {
        var options = ImageRunOptions.FromJson(settings.Image ?? throw new InvalidOperationException("Not an image run."));
        var device = context.Device;
        Directory.CreateDirectory(folder);

        // ---------------------------------------------------------------- the model
        ImageModel? family = null;
        NetworkBuilder? builder = null;
        Module network;
        if (options.Model is { } checkpoint)
        {
            family = ImageModelFiles.Load(context, checkpoint, null, trainable: true);
            network = family.Network;
        }
        else
        {
            builder = Network.FromJson(settings.Network);
            if (builder.InputKind != InputKind.Image)
            {
                throw new UsageException($"The network takes {builder.InputKind.ToString().ToLowerInvariant()} input; image training needs an image input (\"input\": \"Image\", \"shape\": [channels, height, width]).");
            }

            if (settings.Network["seed"] is null)
            {
                builder.Seed(settings.Seed);
            }

            network = builder.OnDevice(device).Build();
        }

        try
        {
            return Train(context, settings, options, folder, family, builder, network, resume, epochsDone, epochs);
        }
        finally
        {
            if (family is not null)
            {
                family.Dispose();
            }
            else
            {
                network.Dispose();
            }
        }
    }

    private static int Train(CommandContext context, RunSettings settings, ImageRunOptions options, string folder, ImageModel? family, NetworkBuilder? builder,
        Module network, string? resume, int epochsDone, int? epochs)
    {
        var device = context.Device;
        var task = ImageRunOptions.ParseTask(options.Task)
                   ?? family?.Task
                   ?? (options.Decoder is not null ? ImageTask.Detection : builder!.CurrentShape.Count == 3 ? ImageTask.Segmentation : ImageTask.Classification);
        if (family is not null && family.Task != task)
        {
            throw new UsageException($"{options.Model} is a {ImagePredict.Task(family.Task)} model; --task {options.Task} does not train it.");
        }

        if (task == ImageTask.Features)
        {
            throw new UsageException($"{options.Model} is a backbone (features): it has no head to train; fine-tune a classifier, detector or segmenter.");
        }

        int channels = family?.Channels ?? builder!.InputShape[0];
        var shape = family is null ? [.. builder!.InputShape] : family.InputShape
            ?? throw new UsageException($"{options.Model}'s preprocessing follows each image's size (a resize by its shortest edge); training batches one size, which this family does not give.");
        int height = shape[1], width = shape[2];

        // The task's outputs: the classes a builder network has (its last layer), or the family's labels.
        int? outputs = builder is null ? family!.Labels?.Count
            : task == ImageTask.Classification ? builder.CurrentShape.Aggregate(1, (a, b) => a * b)
            : task == ImageTask.Segmentation ? builder.CurrentShape[0]
            : null;

        // ---------------------------------------------------------------- the data
        var watch = Stopwatch.StartNew();
        var train = ReadData(settings.Data, options.DataFormat, task, outputs, "--data");
        var evaluation = options.Eval is { } evalPath ? ReadData(Path.GetFullPath(evalPath), options.DataFormat, task, outputs, "--eval") : null;
        IReadOnlyList<string> labels = family?.Labels is { } named ? named : train.Classes;
        Reconcile(train, labels, outputs, task, "--data");
        if (evaluation is not null)
        {
            Reconcile(evaluation, labels, outputs, task, "--eval");
        }

        context.Write($"Data      {train.Count:N0} images from {settings.Data} ({train.Layout}){(evaluation is null ? "" : $" · {evaluation.Count:N0} for evaluation from {options.Eval}")} ({watch.ElapsedMilliseconds} ms)");
        context.Write($"Task      {ImagePredict.Task(task)}, {labels.Count} classes ({string.Join(", ", labels.Take(8))}{(labels.Count > 8 ? ", ..." : "")}) · input [{channels}, {height}, {width}]");
        if (train.Count < 2)
        {
            throw new InvalidDataException($"{settings.Data} has {train.Count} images; training needs more.");
        }

        if (evaluation is null && settings.Validation > 0)
        {
            var order = Enumerable.Range(0, train.Count).ToArray();
            new Random(settings.Seed).Shuffle(order);
            int held = Math.Clamp((int)Math.Round(train.Count * settings.Validation), 1, train.Count - 1);
            evaluation = train.Subset(order[..held]);
            train = train.Subset(order[held..]);
        }

        // ---------------------------------------------------------------- what the steps use
        IReadOnlyList<IAugmentation> pipeline;
        try
        {
            pipeline = Augmentations.Parse(options.Augment);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new UsageException($"--augment: {e.Message} (registered: {string.Join(", ", Augmentations.Names)})");
        }

        if (task == ImageTask.Classification && pipeline.FirstOrDefault(a => a.Name is "mosaic" or "mixup") is { } combining)
        {
            throw new UsageException($"--augment {combining.Name} combines several images' objects; a classifier keeps one class per image.");
        }

        var lossName = CheckName(options.Loss, "--loss", task == ImageTask.Detection ? VisionLosses.Names : [CrossEntropy, .. VisionLosses.Names]);
        var matcherName = CheckName(options.Matcher, "--matcher", BoxMatchers.Names);
        if (matcherName is not null && task != ImageTask.Detection)
        {
            throw new UsageException("--matcher is for detectors (it assigns candidates to true boxes).");
        }

        string metricName = CheckName(options.Metric, "--metric", task == ImageTask.Classification ? [Accuracy] : VisionMetrics.Names)
                            ?? (task switch { ImageTask.Classification => Accuracy, ImageTask.Segmentation => "miou", _ => "coco" });
        var (scale, shift) = Normalization(family?.Preprocessor, channels);
        string? decoderName = task != ImageTask.Detection ? null
            : options.Decoder ?? family?.Decoder ?? throw new UsageException("Give --decoder NAME: the detector's decoder (registered in DetectionDecoders, its head in DetectionHeads).");
        var decoderSettings = family?.DecoderSettings ?? [];
        var decoderContext = new DetectionDecoderContext(height, width, labels, decoderSettings);
        DetectionHead? head = decoderName is null ? null : DetectionHeads.Create(decoderName, decoderContext);
        DetectionDecoder? decoder = decoderName is null ? null : DetectionDecoders.Create(decoderName, decoderContext);
        var objective = new DetectionObjectiveOptions { Matcher = matcherName ?? BoxMatchers.Threshold, BoxLoss = lossName ?? "giou", Scale = Math.Max(height, width) };

        Tensor Loss(Tensor x, DetectionBatch batch, int start, int count, int[] classes)
        {
            var output = network.Forward(x);
            switch (task)
            {
                case ImageTask.Classification:
                {
                    var logits = output.Reshape(count, -1);
                    var onehot = new float[count * labels.Count];
                    for (int i = 0; i < count; i++)
                    {
                        onehot[i * labels.Count + classes[start + i]] = 1f;
                    }

                    var target = Tensor.From(onehot, [count, labels.Count], device);
                    return lossName is null or CrossEntropy ? Losses.CrossEntropy(logits, target) : VisionLosses.Compute(lossName, logits, target);
                }

                case ImageTask.Segmentation:
                {
                    var pixels = batch.PixelClasses ?? throw new InvalidDataException("The segmentation samples have no pixel classes.");
                    var target = count == batch.Count ? pixels : pixels.Narrow(0, start, count);
                    var logits = output.Shape[2] == height && output.Shape[3] == width ? output : output.Interpolate((height, width), InterpolationMode.Bilinear);
                    if (lossName is null or CrossEntropy)
                    {
                        return Losses.PixelCrossEntropy(logits, target);
                    }

                    var onehot = Tensor.OneHot(target, labels.Count).Permute(0, 3, 1, 2);
                    var predicted = lossName == "dice" ? logits.Permute(0, 2, 3, 1).Softmax().Permute(0, 3, 1, 2) : logits;
                    return VisionLosses.Compute(lossName, predicted, onehot);
                }

                default:
                {
                    var candidates = head!(output);
                    return DetectionObjective.Loss(candidates, [.. batch.Boxes.Skip(start).Take(count)], [.. batch.Labels.Skip(start).Take(count)], objective);
                }
            }
        }

        // ---------------------------------------------------------------- loaders, optimizer, run folder
        var trainOrder = Enumerable.Range(0, train.Count).ToArray();
        var shuffle = new Random(settings.Seed + epochsDone);
        var trainLoader = Loader(train, trainOrder, pipeline, settings.Batch, settings.Seed + epochsDone);
        var evalOrder = evaluation is null ? null : Enumerable.Range(0, evaluation.Count).ToArray();
        var evalLoader = evaluation is null ? null : Loader(evaluation, evalOrder!, [], settings.Batch, settings.Seed);
        AugmentedImageLoader Loader(ImageSet set, int[] order, IReadOnlyList<IAugmentation> augmentations, int batch, int seed) =>
            new(set.Count, i => set.Read(order[i]), width, height, batch, shuffle: false, seed, device)
            {
                Augmentations = augmentations, Channels = channels, KeepRatio = false, Fill = 0f,
            };

        if (resume is not null)
        {
            network.Load(resume);
        }

        network.Train();
        string optimizerName = settings.Optimizer.ToLowerInvariant();
        var parameters = network.Parameters().ToList();
        if (parameters.Count == 0)
        {
            throw new UsageException($"{options.Model ?? "The network"} has no trainable parameters.");
        }

        using Optimizer optimizer = optimizerName switch
        {
            "adamw" => new AdamW(parameters, settings.LearningRate, weightDecay: settings.WeightDecay ?? 1e-4f),
            "adam" => new Adam(parameters, settings.LearningRate, weightDecay: settings.WeightDecay ?? 0f),
            "sgd" => new Sgd(parameters, settings.LearningRate, momentum: 0.9f, weightDecay: settings.WeightDecay ?? 0f),
            _ => throw new UsageException($"--optimizer {settings.Optimizer}: use adamw, adam or sgd."),
        };
        int total = epochs ?? Math.Max(0, settings.Epochs - epochsDone);
        if (total == 0)
        {
            context.Error($"The run has done its {settings.Epochs} epochs; give --epochs N to train N more.");
            return ExitCodes.Failed;
        }

        File.WriteAllText(Path.Combine(folder, "run.json"), settings.ToJson().ToJsonString(CommandContext.JsonOutput));
        if (builder is not null)
        {
            File.WriteAllText(Path.Combine(folder, "network.json"), settings.Network.ToJsonString(CommandContext.JsonOutput));
        }

        context.Write($"Split     {train.Count:N0} training / {evaluation?.Count ?? 0:N0} evaluation images (seed {settings.Seed})"
                      + (pipeline.Count > 0 ? $" · augment {string.Join(", ", pipeline.Select(a => a.Name))}" : "")
                      + (scale is null ? "" : " · the family's normalization"));
        context.Write($"Training  {network.ParameterCount:N0} parameters on {device} · {optimizerName} {settings.LearningRate.ToString("G", CultureInfo.InvariantCulture)} · batch {settings.Batch} · "
                      + $"loss {lossName ?? (task == ImageTask.Detection ? "giou" : CrossEntropy)}{(task == ImageTask.Detection ? $" · matcher {objective.Matcher} · head {decoderName}" : "")} · "
                      + $"{(resume is null ? "" : $"from epoch {epochsDone + 1}, ")}up to {total} epochs");

        var started = new TrainingStarted(network.Name ?? options.Model ?? "network", optimizerName, device, total + epochsDone, train.Count, evaluation?.Count,
            settings.Batch, trainLoader.BatchCount, network.ParameterCount, settings.LearningRate, ComputeResources.MaxCpuThreads);
        var writer = new RunLogWriter(Path.Combine(folder, "log.jsonl"), started, epochsDone);
        writer.Append(started);

        // ---------------------------------------------------------------- the loop
        int? patience = evaluation is null ? null : settings.Patience ?? 20;
        int every = context.Verbose ? 1 : Math.Max(1, total / 10);
        int micro = settings.Batch;
        double best = double.PositiveInfinity;
        int bestEpoch = 0, ran = 0, sinceBest = 0;
        bool stoppedEarly = false;
        var clock = Stopwatch.StartNew();
        string last = Path.Combine(folder, "last.ikw"), bestFile = Path.Combine(folder, "best.ikw");
        double finalLoss = double.NaN;
        using (var progress = new ProgressLine(context, "training", (long)total * trainLoader.BatchCount))
        using (var interrupt = new Interrupt(context))
        {
            for (int epoch = 1; epoch <= total && !interrupt.Requested; epoch++)
            {
                var epochClock = Stopwatch.StartNew();
                shuffle.Shuffle(trainOrder);
                network.Train();
                double sum = 0;
                int seen = 0;
                progress.Label = $"epoch {epoch + epochsDone}/{total + epochsDone}";
                foreach (var batch in trainLoader)
                {
                    using (batch)
                    {
                        int[] classes = Classes(train, trainOrder, batch, settings.Batch);
                        var x = Normalized(batch.Images, scale, shift, channels, height * width);
                        using var normalized = ReferenceEquals(x, batch.Images) ? null : x;
                        while (true)
                        {
                            try
                            {
                                optimizer.ZeroGrad();
                                double batchLoss = 0;
                                for (int start = 0; start < batch.Count; start += micro)
                                {
                                    int count = Math.Min(micro, batch.Count - start);
                                    using var scope = new TensorScope();
                                    var part = count == batch.Count ? x : x.Narrow(0, start, count);
                                    var loss = Loss(part, batch, start, count, classes);
                                    var scaled = count == batch.Count ? loss : loss * (count / (float)batch.Count);
                                    scaled.Backward();
                                    batchLoss += loss.Item() * count;
                                }

                                optimizer.Step();
                                sum += batchLoss;
                                seen += batch.Count;
                                break;
                            }
                            catch (ResourceLimitExceededException e) when (micro > 1)
                            {
                                micro = (micro + 1) / 2;
                                progress.Erase();
                                context.Write($"  out of device memory ({e.Message.Split(':')[0]}): micro-batches of {micro} from this step, their gradients added up");
                            }
                        }
                    }

                    progress.Advance();
                    if (interrupt.Requested)
                    {
                        break;
                    }
                }

                if (interrupt.Requested && seen < train.Count)
                {
                    break;                                                       // the unfinished epoch is not summarized
                }

                double trainLoss = sum / Math.Max(1, seen);
                double? validationLoss = evalLoader is null ? null : Evaluate(evalLoader, batch => Classes(evaluation!, evalOrder!, batch, settings.Batch));
                double monitored = validationLoss ?? trainLoss;
                bool isBest = monitored < best;
                network.Save(last);
                if (isBest)
                {
                    (best, bestEpoch, sinceBest) = (monitored, epoch, 0);
                    network.Save(bestFile);
                }
                else
                {
                    sinceBest++;
                }

                ran = epoch;
                finalLoss = trainLoss;
                epochClock.Stop();
                writer.Append(new EpochCompleted(epoch + epochsDone, total + epochsDone, trainLoss, new Dictionary<string, double>(), validationLoss,
                    validationLoss is null ? null : new Dictionary<string, double>(), optimizer.LearningRate, epochClock.Elapsed,
                    seen / Math.Max(epochClock.Elapsed.TotalSeconds, 1e-9), ComputeResources.GetMemoryUsage(device), isBest));
                if (epoch % every == 0 || epoch == 1 || epoch == total)
                {
                    progress.Erase();
                    context.Write($"  epoch {epoch + epochsDone,4}  loss {RunLog.Format(trainLoss)}" + (validationLoss is { } v ? $"  val_loss {RunLog.Format(v)}" : ""));
                }

                if (patience is > 0 && sinceBest >= patience)
                {
                    stoppedEarly = true;
                    break;
                }
            }

            progress.Finish();
            if (interrupt.Requested)
            {
                context.Write(interrupt.TimedOut ? $"Stopped   --timeout ran out; 'idrak resume {Path.GetFileName(folder)}' continues the run" : $"Stopped   'idrak resume {Path.GetFileName(folder)}' continues the run");
            }
        }

        double seconds = clock.Elapsed.TotalSeconds;
        writer.Append(new TrainingCompleted(ran, clock.Elapsed, finalLoss, bestEpoch + epochsDone, best, stoppedEarly, false));
        if (evaluation is not null && File.Exists(bestFile))
        {
            network.Load(bestFile);                                              // the best epoch's weights are the ones kept
        }

        // ---------------------------------------------------------------- the metric on the evaluation data
        var report = Score(evalLoader ?? Loader(train, Enumerable.Range(0, train.Count).ToArray(), [], settings.Batch, settings.Seed), evaluation ?? train, evalOrder ?? Enumerable.Range(0, train.Count).ToArray());
        string headline = report.Count == 0 ? "" : string.Join(" · ", report.Take(task == ImageTask.Detection ? 3 : 2).Select(p => $"{p.Key} {RunLog.Format(p.Value)}"));
        context.Write($"Result    {(evaluation is null ? "training" : "evaluation")} {metricName}: {headline}");

        // ---------------------------------------------------------------- what is written
        var metrics = new JsonObject();
        foreach (var (key, value) in report)
        {
            metrics[key] = double.IsFinite(value) ? Math.Round(value, 6) : null;
        }

        var meta = new JsonObject
        {
            ["format"] = "idrak-train/1",
            ["task"] = ImagePredict.Task(task),
            ["input"] = "images",
            ["classes"] = new JsonArray([.. labels.Select(n => (JsonNode)n)]),
            ["inputShape"] = new JsonArray(channels, height, width),
            ["data"] = settings.Data,
            ["epochs"] = epochsDone + ran,
            ["metric"] = metricName,
            ["metrics"] = metrics,
            ["augment"] = options.Augment,
            ["decoder"] = decoderName,
            ["decoderSettings"] = decoderName is null ? null : decoderSettings.DeepClone(),
            ["model"] = options.Model,
        };
        File.WriteAllText(Path.Combine(folder, "training.json"), meta.ToJsonString(CommandContext.JsonOutput));
        if (builder is not null)
        {
            TrainSession.Package(network, settings.Network, null, null, meta, settings.Output);
        }
        else
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(settings.Output))!);
            network.Save(settings.Output);
        }

        context.Write($"Best      epoch {bestEpoch + epochsDone} (loss {RunLog.Format(best)}){(stoppedEarly ? ", stopped early" : "")} · {seconds:F1} s");
        context.Write($"Wrote     {settings.Output} · run {folder}");
        context.Write(builder is not null
            ? $"Next      idrak predict {settings.Output} NEW.png   ·   idrak runs show {folder}"
            : $"Next      idrak predict -P PLUGIN {options.Model} NEW.png --weights {settings.Output}   ·   idrak runs show {folder}");
        context.WriteJson(new JsonObject
        {
            ["model"] = Path.GetFullPath(settings.Output),
            ["run"] = Path.GetFullPath(folder),
            ["task"] = ImagePredict.Task(task),
            ["classes"] = meta["classes"]!.DeepClone(),
            ["images"] = train.Count + (options.Eval is null ? evaluation?.Count ?? 0 : 0),
            ["trainingImages"] = train.Count,
            ["evaluationImages"] = evaluation?.Count ?? 0,
            ["parameters"] = network.ParameterCount,
            ["device"] = device.ToString(),
            ["epochs"] = epochsDone + ran,
            ["bestEpoch"] = bestEpoch + epochsDone,
            ["bestLoss"] = double.IsFinite(best) ? Math.Round(best, 6) : null,
            ["finalLoss"] = double.IsFinite(finalLoss) ? Math.Round(finalLoss, 6) : null,
            ["stoppedEarly"] = stoppedEarly,
            ["microBatch"] = micro,
            ["seconds"] = Math.Round(seconds, 3),
            ["metric"] = metricName,
            ["metrics"] = metrics.DeepClone(),
        });
        return ExitCodes.Ok;

        // The evaluation loss (no gradients, evaluation mode).
        double Evaluate(AugmentedImageLoader loader, Func<DetectionBatch, int[]> classesOf)
        {
            network.Eval();
            double sum = 0;
            int seen = 0;
            using (Autograd.NoGrad())
            {
                foreach (var batch in loader)
                {
                    using (batch)
                    {
                        using var scope = new TensorScope();
                        var x = Normalized(batch.Images, scale, shift, channels, height * width);
                        sum += Loss(x, batch, 0, batch.Count, classesOf(batch)).Item() * batch.Count;
                        seen += batch.Count;
                    }
                }
            }

            network.Train();
            return sum / Math.Max(1, seen);
        }

        // The metric over the evaluation data: accuracy, a segmentation metric over the masks, or a detection metric over
        // the decoded and suppressed boxes, all at the network's input size.
        IReadOnlyDictionary<string, double> Score(AugmentedImageLoader loader, ImageSet set, int[] order)
        {
            network.Eval();
            var metricOptions = new VisionMetricOptions { Classes = labels.Count, ClassNames = labels };
            IVisionMetric? metric = task == ImageTask.Classification ? null : VisionMetrics.Create(metricName, metricOptions);
            int right = 0, count = 0;
            using (Autograd.NoGrad())
            {
                foreach (var batch in loader)
                {
                    using (batch)
                    {
                        using var scope = new TensorScope();
                        var x = Normalized(batch.Images, scale, shift, channels, height * width);
                        var output = network.Forward(x);
                        switch (task)
                        {
                            case ImageTask.Classification:
                                var classes = Classes(set, order, batch, settings.Batch);
                                var values = output.ToArray();
                                int per = values.Length / batch.Count;
                                for (int i = 0; i < batch.Count; i++)
                                {
                                    int guess = 0;
                                    for (int k = 1; k < per; k++)
                                    {
                                        guess = values[i * per + k] > values[i * per + guess] ? k : guess;
                                    }

                                    right += guess == classes[i] ? 1 : 0;
                                }

                                break;
                            case ImageTask.Segmentation:
                                var logits = output.Shape[2] == height && output.Shape[3] == width ? output : output.Interpolate((height, width), InterpolationMode.Bilinear);
                                var masks = SegmentationMask.FromLogits(logits);
                                var truth = batch.PixelClasses!.ToArray();
                                for (int i = 0; i < batch.Count; i++)
                                {
                                    var expected = new SegmentationMask([.. truth.AsSpan(i * height * width, height * width).ToArray().Select(v => (int)v)], width, height, labels.Count);
                                    ((ISegmentationMetric)metric!).Add(masks[i], expected);
                                }

                                break;
                            default:
                                var raw = output.ToArray();
                                int[] rawShape = [.. output.Shape[1..]];
                                int size = raw.Length / batch.Count;
                                for (int i = 0; i < batch.Count; i++)
                                {
                                    var found = decoder!(raw.AsSpan(i * size, size), rawShape).Select(d => d with { Box = d.Box.Clip(width, height) }).Where(d => d.Box.Area > 0).ToList();
                                    var kept = NonMaxSuppression.Apply(found, 0.5f, 0.001f, perClass: true, metricOptions.MaxDetections);
                                    ((IDetectionMetric)metric!).Add(kept, [.. batch.Boxes[i].Select((b, k) => new ObjectAnnotation(b, batch.Labels[i][k]))]);
                                }

                                break;
                        }

                        count += batch.Count;
                    }
                }
            }

            network.Train();
            return metric?.Compute() ?? new Dictionary<string, double> { [Accuracy] = right / (double)Math.Max(1, count) };
        }
    }

    private const string CrossEntropy = "cross-entropy", Accuracy = "accuracy";

    // ---------------------------------------------------------------- reading the data

    // The images of --data or --eval for the task: class folders, image and mask folders, or a dataset an annotation format reads.
    private static ImageSet ReadData(string path, string format, ImageTask task, int? outputs, string option)
    {
        format = format.ToLowerInvariant();
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            throw new UsageException($"{option} {path}: no such file or folder.");
        }

        bool maskFolders = Directory.Exists(Path.Combine(path, "images")) && Directory.Exists(Path.Combine(path, "masks"));
        if (format == "masks" || format == "auto" && task == ImageTask.Segmentation && maskFolders)
        {
            return task == ImageTask.Segmentation ? MaskFolders(path, outputs, option)
                : throw new UsageException($"{option}: image and mask folders are a segmenter's data, not a {ImagePredict.Task(task)} model's.");
        }

        if (format == "folder" || format == "auto" && task == ImageTask.Classification)
        {
            return task == ImageTask.Classification ? ClassFolders(path, option)
                : throw new UsageException($"{option}: class folders are a classifier's data, not a {ImagePredict.Task(task)} model's.");
        }

        if (task == ImageTask.Classification)
        {
            throw new UsageException($"{option}: a classifier reads a folder with a folder of images per class (--data-format folder), not {format} annotations.");
        }

        AnnotatedDataset dataset;
        string layout;
        try
        {
            var reader = format == "auto" ? AnnotationFormats.Find(path) ?? throw new NotSupportedException($"no registered annotation format reads it ({string.Join(", ", AnnotationFormats.Names)})")
                : AnnotationFormats.Get(format);
            layout = reader.Name;
            dataset = reader.Read(path);
        }
        catch (NotSupportedException e)
        {
            throw new UsageException($"{option} {path}: {e.Message}; --data-format takes auto, folder, masks or an annotation format ({string.Join(", ", AnnotationFormats.Names)}).");
        }

        bool segment = task == ImageTask.Segmentation;
        return new ImageSet
        {
            Source = path,
            Layout = layout,
            Files = [.. Enumerable.Range(0, dataset.Images.Count).Select(dataset.PathOf)],
            Classes = segment ? ["background", .. dataset.Classes] : dataset.Classes,
            Read = i =>
            {
                var sample = dataset.Images[i].ToSample(ImageCodecs.Decode(dataset.PathOf(i)), masks: segment);
                if (!segment)
                {
                    return sample;
                }

                // The objects' masks (or their boxes) painted as a class per pixel: background 0, the object's class + 1.
                int plane = sample.Width * sample.Height;
                var pixels = new int[plane];
                for (int k = 0; k < sample.Count; k++)
                {
                    var mask = sample.Masks?[k] ?? AnnotatedImage.Fill(sample.Boxes[k], sample.Width, sample.Height);
                    for (int j = 0; j < plane; j++)
                    {
                        if (mask[j] != 0)
                        {
                            pixels[j] = sample.Labels[k] + 1;
                        }
                    }
                }

                return new AnnotatedImage(sample.Image, [.. sample.Boxes], [.. sample.Labels], sample.Masks?.ToArray(), pixels);
            },
        };
    }

    // folder/CLASS/*.png: a class per subfolder, in name order (as the image folder source has them).
    private static ImageSet ClassFolders(string path, string option)
    {
        if (!Directory.Exists(path))
        {
            throw new UsageException($"{option} {path}: a classifier reads a folder with a folder of images per class.");
        }

        var folders = Directory.GetDirectories(path).Order(StringComparer.Ordinal).ToList();
        var files = new List<string>();
        var classes = new List<int>();
        for (int c = 0; c < folders.Count; c++)
        {
            foreach (string file in Directory.EnumerateFiles(folders[c], "*", SearchOption.AllDirectories).Where(ImageCodecs.CanDecode).Order(StringComparer.Ordinal))
            {
                files.Add(file);
                classes.Add(c);
            }
        }

        if (folders.Count < 2 || files.Count == 0)
        {
            throw new InvalidDataException($"{path} needs a folder per class with images in them ({string.Join(", ", ImageFiles.DecodedExtensions)}).");
        }

        return new ImageSet
        {
            Source = path, Layout = "class folders", Files = files, Classes = [.. folders.Select(Path.GetFileName).OfType<string>()],
            Read = i => new AnnotatedImage(ImageCodecs.Decode(files[i])), ImageClasses = [.. classes],
        };
    }

    // folder/images/NAME.* with folder/masks/NAME.* (a grey image whose level is each pixel's class); the class names from
    // folder/classes.txt (one a line), else the network's outputs numbered.
    private static ImageSet MaskFolders(string path, int? outputs, string option)
    {
        var images = Directory.EnumerateFiles(Path.Combine(path, "images")).Where(ImageCodecs.CanDecode).Order(StringComparer.Ordinal).ToList();
        var masks = Directory.EnumerateFiles(Path.Combine(path, "masks")).Where(ImageCodecs.CanDecode)
            .GroupBy(f => Path.GetFileNameWithoutExtension(f), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.OrderBy(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? 0 : 1).First(), StringComparer.Ordinal);
        string? missing = images.FirstOrDefault(f => !masks.ContainsKey(Path.GetFileNameWithoutExtension(f)));
        if (missing is not null)
        {
            throw new InvalidDataException($"{missing} has no mask in {Path.Combine(path, "masks")} (the same name, any image extension).");
        }

        string names = Path.Combine(path, "classes.txt");
        IReadOnlyList<string> classes = File.Exists(names) ? [.. File.ReadAllLines(names).Select(l => l.Trim()).Where(l => l.Length > 0)]
            : outputs is { } count ? [.. Enumerable.Range(0, count).Select(i => i.ToString(CultureInfo.InvariantCulture))]
            : throw new UsageException($"{option} {path}: give the class names in classes.txt (one a line, the mask level's order).");
        if (images.Count == 0)
        {
            throw new InvalidDataException($"{Path.Combine(path, "images")} has no images ({string.Join(", ", ImageFiles.DecodedExtensions)}).");
        }

        return new ImageSet
        {
            Source = path, Layout = "image and mask folders", Files = images, Classes = classes,
            Read = i =>
            {
                var image = ImageCodecs.Decode(images[i]);
                string maskFile = masks[Path.GetFileNameWithoutExtension(images[i])];
                var mask = ImageCodecs.Decode(maskFile);
                if (mask.Width != image.Width || mask.Height != image.Height)
                {
                    throw new InvalidDataException($"{maskFile} is {mask.Width} x {mask.Height}; its image is {image.Width} x {image.Height}.");
                }

                var pixels = new int[image.Width * image.Height];
                for (int j = 0; j < pixels.Length; j++)
                {
                    int level = (int)MathF.Round(mask.Pixels[j] * 255f);
                    pixels[j] = level < classes.Count ? level
                        : throw new InvalidDataException($"{maskFile} has level {level}, past the {classes.Count} classes (a mask's level is its pixel's class).");
                }

                return new AnnotatedImage(image, pixelClasses: pixels);
            },
        };
    }

    // The data's classes as the model's: the same names (in the model's order, read by name), or as many classes by index.
    private static void Reconcile(ImageSet set, IReadOnlyList<string> labels, int? outputs, ImageTask task, string option)
    {
        if (outputs is { } count && labels.Count != count)
        {
            throw new InvalidDataException($"{set.Source} has {labels.Count} classes ({string.Join(", ", labels.Take(8))}{(labels.Count > 8 ? ", ..." : "")}) but the network has "
                                           + $"{count} {(task == ImageTask.Segmentation ? "output channels" : "outputs")}; make its last layer {labels.Count} wide.");
        }

        if (set.Classes.SequenceEqual(labels, StringComparer.Ordinal))
        {
            return;
        }

        int[] map;
        if (set.Classes.All(c => labels.Contains(c, StringComparer.Ordinal)))
        {
            map = [.. set.Classes.Select(c => labels.ToList().IndexOf(c))];
        }
        else if (set.Classes.Count == labels.Count)
        {
            map = [.. Enumerable.Range(0, labels.Count)];                         // by index
        }
        else
        {
            throw new InvalidDataException($"{option} {set.Source}: its classes ({string.Join(", ", set.Classes.Take(8))}) are not the model's ({string.Join(", ", labels.Take(8))}).");
        }

        var read = set.Read;
        set.Read = i =>
        {
            var sample = read(i);
            return new AnnotatedImage(sample.Image, [.. sample.Boxes], [.. sample.Labels.Select(l => map[l])], sample.Masks?.ToArray(),
                sample.PixelClasses is { } pixels ? [.. pixels.Select(p => map[p])] : null);
        };
        if (set.ImageClasses is { } classes)
        {
            set.ImageClasses = [.. classes.Select(c => map[c])];
        }

        set.Classes = labels;
    }

    // A registry name given on the command line, checked against the names there are (case as registered).
    private static string? CheckName(string? name, string option, IEnumerable<string> names)
    {
        if (name is null)
        {
            return null;
        }

        var known = names.ToList();
        return known.FirstOrDefault(n => n.Equals(name, StringComparison.OrdinalIgnoreCase))
            ?? throw new UsageException($"{option} {name}: use {string.Join(", ", known)}.");
    }

    // The classes of a batch's images (classification): batch b of a loader of batches of `size` holds the samples at
    // positions b·size onward of the order the loader reads.
    private static int[] Classes(ImageSet set, int[] order, DetectionBatch batch, int size) =>
        set.ImageClasses is not { } classes ? [] : [.. Enumerable.Range(0, batch.Count).Select(i => classes[order[batch.Index * size + i]])];

    // The family's rescale and normalization as one per-channel affine of the loader's [0, 1] values; null when none.
    private static (float[]? Scale, float[]? Shift) Normalization(ImagePreprocessor? preprocessor, int channels)
    {
        if (preprocessor is null)
        {
            return (null, null);
        }

        var scale = new float[channels];
        var shift = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float std = preprocessor.Normalize ? preprocessor.Std[Math.Min(c, preprocessor.Std.Count - 1)] : 1f;
            float mean = preprocessor.Normalize ? preprocessor.Mean[Math.Min(c, preprocessor.Mean.Count - 1)] : 0f;
            scale[c] = (float)((preprocessor.Rescale ? preprocessor.RescaleFactor * 255 : 255) / std);
            shift[c] = -mean / std;
        }

        return scale.All(s => MathF.Abs(s - 1f) < 1e-7f) && shift.All(s => s == 0) ? (null, null) : (scale, shift);
    }

    // The images as the network takes them: the loader's tensor itself without a normalization, else a new one.
    private static Tensor Normalized(Tensor images, float[]? scale, float[]? shift, int channels, int plane)
    {
        if (scale is null)
        {
            return images;
        }

        using var s = Tensor.From(scale, [channels], images.Device);
        using var b = Tensor.From(shift!, [channels], images.Device);
        return images.GroupAffine(s, b, channels, plane);
    }
}
