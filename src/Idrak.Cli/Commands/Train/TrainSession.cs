// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using System.Text;
using Idrak.Cli.Shared;
using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training.Abstractions;
using Idrak.Training;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// The settings of a training run: the network (builder JSON), the data and how it is read, and the training loop's
/// options. Written to the run folder's <c>run.json</c>, so <c>idrak resume</c> rebuilds the same run.
/// </summary>
internal sealed record RunSettings
{
    public required JsonObject Network { get; init; }

    public required string Data { get; init; }

    public required string Output { get; init; }

    public IReadOnlyList<string> Targets { get; init; } = [];

    public IReadOnlyList<string> Ignore { get; init; } = [];

    public string? Task { get; init; }

    public int Epochs { get; init; } = 100;

    public float LearningRate { get; init; } = 1e-3f;

    public int Batch { get; init; } = 32;

    public int? Patience { get; init; }

    public double Validation { get; init; } = 0.2;

    public int Seed { get; init; } = 1;

    public string Optimizer { get; init; } = "adamw";

    public float? WeightDecay { get; init; }

    public bool Scale { get; init; } = true;

    public JsonObject ToJson() => new()
    {
        ["format"] = "idrak-run/1",
        ["network"] = Network.DeepClone(),
        ["data"] = Data,
        ["out"] = Output,
        ["targets"] = new JsonArray([.. Targets.Select(t => (JsonNode)t)]),
        ["ignore"] = new JsonArray([.. Ignore.Select(t => (JsonNode)t)]),
        ["task"] = Task,
        ["epochs"] = Epochs,
        ["learning_rate"] = LearningRate,
        ["batch"] = Batch,
        ["patience"] = Patience,
        ["validation"] = Validation,
        ["seed"] = Seed,
        ["optimizer"] = Optimizer,
        ["weight_decay"] = WeightDecay,
        ["scale"] = Scale,
    };

    public static RunSettings FromJson(JsonObject json) => new()
    {
        Network = (JsonObject)(json["network"] ?? throw new InvalidDataException("run.json has no network.")).DeepClone(),
        Data = (string?)json["data"] ?? throw new InvalidDataException("run.json has no data."),
        Output = (string?)json["out"] ?? "model.ikm",
        Targets = json["targets"] is JsonArray t ? [.. t.Select(n => (string)n!)] : [],
        Ignore = json["ignore"] is JsonArray i ? [.. i.Select(n => (string)n!)] : [],
        Task = (string?)json["task"],
        Epochs = (int?)json["epochs"] ?? 100,
        LearningRate = (float?)json["learning_rate"] ?? 1e-3f,
        Batch = (int?)json["batch"] ?? 32,
        Patience = (int?)json["patience"],
        Validation = (double?)json["validation"] ?? 0.2,
        Seed = (int?)json["seed"] ?? 1,
        Optimizer = (string?)json["optimizer"] ?? "adamw",
        WeightDecay = (float?)json["weight_decay"],
        Scale = (bool?)json["scale"] ?? true,
    };
}

/// <summary>
/// One training run of a builder network on a CSV file (through the library's <see cref="Dataset"/>) or an image folder
/// (class subfolders): split, scale, train with validation, early stopping, checkpoints and a JSON Lines log in the run
/// folder, then a model package (.ikm) with the architecture, weights, scalers and what <c>idrak predict</c> needs.
/// </summary>
internal static class TrainSession
{
    /// <summary>Runs <paramref name="settings"/> in <paramref name="folder"/>; continues from <paramref name="resume"/> weights after <paramref name="epochsDone"/> epochs.</summary>
    public static int Run(CommandContext context, RunSettings settings, string folder, string? resume = null, int epochsDone = 0, int? epochs = null)
    {
        var device = context.Device;
        var builder = Network.FromJson(settings.Network);
        int[] input = [.. builder.InputShape];
        int outputs = builder.CurrentShape.Aggregate(1, (a, b) => a * b);
        Directory.CreateDirectory(folder);

        // ---------------------------------------------------------------- data
        var watch = Stopwatch.StartNew();
        var loaded = Directory.Exists(settings.Data) ? LoadImages(settings.Data, builder) : LoadTable(settings, builder, outputs);
        var (data, task, classes) = loaded;
        context.Write($"Data      {data.Count:N0} rows from {settings.Data}: {data.FeatureCount} features [{Shape(data.FeatureShape)}] -> {string.Join(", ", data.TargetNames)}"
                      + $" ({watch.ElapsedMilliseconds} ms)");
        context.Write($"Task      {task}{(classes is null ? "" : $", {classes.Count} classes ({string.Join(", ", classes.Take(8))}{(classes.Count > 8 ? ", ..." : "")})")}");
        if (data.Count < 2)
        {
            throw new InvalidDataException($"{settings.Data} has {data.Count} rows; training needs more.");
        }

        var (train, validation) = settings.Validation > 0 ? data.Split(1 - settings.Validation, settings.Seed) : (data, null);
        bool scale = settings.Scale && builder.InputKind != InputKind.Tokens && !Directory.Exists(settings.Data);
        string scalerFile = Path.Combine(folder, "features.txt"), targetFile = Path.Combine(folder, "targets.txt");
        StandardScaler? featureScaler = !scale ? null : resume is not null && File.Exists(scalerFile) ? StandardScaler.Load(scalerFile) : StandardScaler.FitFeatures(train);
        StandardScaler? targetScaler = task != "regression" || !settings.Scale ? null
            : resume is not null && File.Exists(targetFile) ? StandardScaler.Load(targetFile) : StandardScaler.FitTargets(train);
        Dataset Prepare(Dataset d)
        {
            var scaled = featureScaler is null && targetScaler is null ? d : d.Scale(featureScaler, targetScaler);
            if (task == "classification" && classes is not null && scaled.TargetCount == 1 && outputs > 1)
            {
                scaled = scaled.ToOneHot(outputs);
            }

            return builder.InputKind == InputKind.Features ? scaled : scaled.WithFeatureShape(input);
        }

        var trainSet = Prepare(train);
        var validationSet = validation is { Count: > 0 } ? Prepare(validation) : null;
        featureScaler?.Save(scalerFile);
        targetScaler?.Save(targetFile);
        File.WriteAllText(Path.Combine(folder, "network.json"), settings.Network.ToJsonString(CommandContext.JsonOutput));
        File.WriteAllText(Path.Combine(folder, "run.json"), settings.ToJson().ToJsonString(CommandContext.JsonOutput));
        context.Write($"Split     {trainSet.Count:N0} training / {validationSet?.Count ?? 0:N0} validation rows (seed {settings.Seed})"
                      + (scale ? " · features standard-scaled" : "") + (targetScaler is not null ? " · target standard-scaled" : ""));

        // ---------------------------------------------------------------- model and loop
        if (settings.Network["seed"] is null)
        {
            builder.Seed(settings.Seed);
        }

        using var model = builder.OnDevice(device).Build();
        if (resume is not null)
        {
            model.Load(resume);
        }

        var (loss, metric) = task switch
        {
            "classification" when outputs == 1 => ((Func<Tensor, Tensor, Tensor>)Losses.BinaryCrossEntropyWithLogits, Metric.BinaryAccuracy(0f)),
            "classification" => (Losses.CrossEntropy, Metric.Accuracy),
            _ => (Losses.MeanSquaredError, Metric.MeanAbsoluteError),
        };
        string optimizerName = settings.Optimizer.ToLowerInvariant();
        Func<IEnumerable<Tensor>, Optimizer> optimizer = optimizerName switch
        {
            "adamw" => p => new AdamW(p, settings.LearningRate, weightDecay: settings.WeightDecay ?? 1e-4f),
            "adam" => p => new Adam(p, settings.LearningRate, weightDecay: settings.WeightDecay ?? 0f),
            "sgd" => p => new Sgd(p, settings.LearningRate, momentum: 0.9f, weightDecay: settings.WeightDecay ?? 0f),
            _ => throw new UsageException($"--optimizer {settings.Optimizer}: use adamw, adam or sgd."),
        };
        int total = epochs ?? Math.Max(0, settings.Epochs - epochsDone);
        if (total == 0)
        {
            context.Error($"The run has done its {settings.Epochs} epochs; give --epochs N to train N more.");
            return ExitCodes.Failed;
        }

        int? patience = validationSet is null ? null : settings.Patience ?? 20;
        int every = context.Verbose ? 1 : Math.Max(1, total / 10);
        var clock = Stopwatch.StartNew();
        using var progress = new ProgressLine(context, "training", unit: ProgressUnit.Items);
        using var trainer = new Trainer(model, loss, optimizer)
        {
            Metrics = { metric },
            EarlyStoppingPatience = patience is > 0 ? patience : null,
            OnEpoch = e =>
            {
                if (e.Epoch % every == 0 || e.Epoch == 1 || e.Epoch == total)
                {
                    progress.Erase();
                    context.Write($"  epoch {e.Epoch + epochsDone,4}  loss {RunLog.Format(e.Loss)}"
                                  + (e.ValidationLoss is { } v ? $"  val_loss {RunLog.Format(v)}" : "")
                                  + string.Concat((e.ValidationMetrics ?? e.Metrics).Select(m => $"  {(e.ValidationMetrics is null ? "" : "val_")}{m.Key} {RunLog.Format(m.Value)}")));
                }
            },
        };
        var trainLoader = new DataLoader(trainSet, settings.Batch, shuffle: true, device: device, seed: settings.Seed + epochsDone);
        var validationLoader = validationSet is null ? null : new DataLoader(validationSet, Math.Max(settings.Batch, 256), device: device);
        string log = Path.Combine(folder, "log.jsonl");
        trainer.Callbacks.Add(new Checkpoint(folder));
        trainer.Callbacks.Add(new TrainingProgress(progress, total, epochsDone));
        trainer.Callbacks.Add(new RunLogWriter(log, new TrainingStarted(model.Name ?? "network", optimizerName, device, total + epochsDone, trainSet.Count,
            validationSet?.Count, settings.Batch, (trainSet.Count + settings.Batch - 1) / settings.Batch, model.ParameterCount, settings.LearningRate,
            ComputeResources.MaxCpuThreads), epochsDone));
        context.Write($"Training  {model.ParameterCount:N0} parameters on {device} · {optimizerName} {settings.LearningRate.ToString("G", CultureInfo.InvariantCulture)} · batch {settings.Batch} · "
                      + $"{(resume is null ? "" : $"from epoch {epochsDone + 1}, ")}up to {total} epochs" + (trainer.EarlyStoppingPatience is { } p ? $", early stop after {p}" : ""));
        // Ctrl+C or --timeout: training stops after the current batch, keeping the checkpoints (idrak resume continues).
        TrainingHistory history;
        using (var interrupt = new Interrupt(context))
        {
            history = trainer.Fit(trainLoader, total, validationLoader, interrupt.Token);
            if (interrupt.Requested)
            {
                context.Write(interrupt.TimedOut ? $"Stopped   --timeout ran out; 'idrak resume {Path.GetFileName(folder)}' continues the run" : $"Stopped   'idrak resume {Path.GetFileName(folder)}' continues the run");
            }
        }

        // ---------------------------------------------------------------- evaluate in the data's units
        var report = new JsonObject();
        var check = validation is { Count: > 0 } ? validation : train;
        var checkSet = Prepare(check);
        float[] predicted = [.. trainer.Predict(checkSet).Cast<float>()];
        if (task == "regression")
        {
            targetScaler?.InverseTransform(predicted, check.TargetCount);
            var r = RegressionReport.Compute(predicted, check.Targets);
            report["mae"] = Math.Round(r.MeanAbsoluteError, 6);
            report["rmse"] = Math.Round(r.RootMeanSquaredError, 6);
            report["r2"] = Math.Round(r.RSquared, 6);
            context.Write($"Result    {(validation is null ? "training" : "validation")} MAE {RunLog.Format(r.MeanAbsoluteError)} · RMSE {RunLog.Format(r.RootMeanSquaredError)} · R² {r.RSquared:F4}");
        }
        else
        {
            var expected = checkSet.Targets;
            int right = 0;
            for (int i = 0; i < check.Count; i++)
            {
                int guess = outputs == 1 ? (predicted[i] > 0 ? 1 : 0) : ArgMax(predicted.AsSpan(i * outputs, outputs));
                int truth = outputs == 1 ? (expected[i] > 0.5f ? 1 : 0) : ArgMax(expected.Slice(i * outputs, outputs));
                right += guess == truth ? 1 : 0;
            }

            report["accuracy"] = Math.Round(right / (double)Math.Max(1, check.Count), 6);
            context.Write($"Result    {(validation is null ? "training" : "validation")} accuracy {right / (double)Math.Max(1, check.Count):P1} ({right} of {check.Count})");
        }

        // ---------------------------------------------------------------- package
        var meta = new JsonObject
        {
            ["format"] = "idrak-train/1",
            ["task"] = task,
            ["input"] = Directory.Exists(settings.Data) ? "images" : "table",
            ["features"] = new JsonArray([.. data.FeatureNames.Select(n => (JsonNode)n)]),
            ["targets"] = new JsonArray([.. (classes is not null && data.TargetCount != 1 ? ["class"] : data.TargetNames).Select(n => (JsonNode)n)]),
            ["classes"] = classes is null ? null : new JsonArray([.. classes.Select(n => (JsonNode)n)]),
            ["inputShape"] = new JsonArray([.. input.Select(n => (JsonNode)n)]),
            ["data"] = Path.GetFullPath(settings.Data),
            ["epochs"] = epochsDone + history.Epochs.Count,
            ["metrics"] = report,
        };
        File.WriteAllText(Path.Combine(folder, "training.json"), meta.ToJsonString(CommandContext.JsonOutput));
        Package(model, settings.Network, featureScaler, targetScaler, meta, settings.Output);
        double seconds = clock.Elapsed.TotalSeconds;
        context.Write($"Best      epoch {history.BestEpoch + epochsDone} (loss {RunLog.Format(history.BestLoss)}){(history.StoppedEarly ? ", stopped early and restored its weights" : "")} · {seconds:F1} s");
        context.Write($"Wrote     {settings.Output} · run {folder}");
        context.Write($"Next      idrak predict {settings.Output} -i NEW.{(Directory.Exists(settings.Data) ? "png" : "csv")}   ·   idrak runs show {folder}");
        context.WriteJson(new JsonObject
        {
            ["model"] = Path.GetFullPath(settings.Output),
            ["run"] = Path.GetFullPath(folder),
            ["task"] = task,
            ["classes"] = meta["classes"]?.DeepClone(),
            ["rows"] = data.Count,
            ["trainingRows"] = trainSet.Count,
            ["validationRows"] = validationSet?.Count ?? 0,
            ["parameters"] = model.ParameterCount,
            ["device"] = device.ToString(),
            ["epochs"] = epochsDone + history.Epochs.Count,
            ["bestEpoch"] = history.BestEpoch + epochsDone,
            ["bestLoss"] = double.IsFinite(history.BestLoss) ? Math.Round(history.BestLoss, 6) : null,
            ["finalLoss"] = history.Epochs.Count > 0 && double.IsFinite(history.Epochs[^1].Loss) ? Math.Round(history.Epochs[^1].Loss, 6) : null,
            ["stoppedEarly"] = history.StoppedEarly,
            ["seconds"] = Math.Round(seconds, 3),
            ["metrics"] = report.DeepClone(),
        });
        return ExitCodes.Ok;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            best = values[i] > values[best] ? i : best;
        }

        return best;
    }

    /// <summary>Writes a model package: the architecture, the weights, the scalers and the training metadata.</summary>
    public static void Package(Module model, JsonObject network, IScaler? features, IScaler? targets, JsonObject? meta, string path, Idrak.Abstraction.Generation.ITokenizer? tokenizer = null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var writer = ModelPackage.Create(path).Architecture(ModelPackage.DefaultModelName, network).Weights(model);
        if (features is not null)
        {
            writer.Scaler("features", features);
        }

        if (targets is not null)
        {
            writer.Scaler("targets", targets);
        }

        if (meta is not null)
        {
            writer.Json("training", meta);
        }

        if (tokenizer is not null)
        {
            writer.Tokenizer("tokenizer", tokenizer);
        }

        writer.Save();
    }

    private static string Shape(IReadOnlyList<int> shape) => string.Join("x", shape);

    // ---------------------------------------------------------------- tables

    // A CSV through the library's Dataset; a class column of names (not numbers) becomes class indices first.
    private static (Dataset Data, string Task, IReadOnlyList<string>? Classes) LoadTable(RunSettings settings, NetworkBuilder builder, int outputs)
    {
        if (!File.Exists(settings.Data))
        {
            throw new UsageException($"--data {settings.Data}: no such file or folder (a CSV file, or a folder of class folders of images).");
        }

        string text = File.ReadAllText(settings.Data);
        var lines = text.Split('\n').Select(l => l.TrimEnd('\r')).ToList();
        string[] header = lines.FirstOrDefault(l => l.Trim().Length > 0)?.Split(',').Select(h => h.Trim().Trim('"')).ToArray()
            ?? throw new InvalidDataException($"{settings.Data} is empty.");
        var targets = settings.Targets.Count > 0 ? settings.Targets : [header[^1]];
        foreach (string column in targets.Concat(settings.Ignore).Where(c => !header.Contains(c, StringComparer.OrdinalIgnoreCase)))
        {
            throw new UsageException($"Column '{column}' is not in {settings.Data} (columns: {string.Join(", ", header)}).");
        }

        string task = settings.Task switch
        {
            null => outputs > 1 ? "classification" : "regression",
            "regression" or "regress" => "regression",
            "classify" or "classification" => "classification",
            var other => throw new UsageException($"--task {other}: use regression or classify."),
        };

        // Class names: the distinct values of a single class column when they are not all whole numbers.
        IReadOnlyList<string>? classes = null;
        if (task == "classification" && targets.Count == 1)
        {
            int column = Array.FindIndex(header, h => h.Equals(targets[0], StringComparison.OrdinalIgnoreCase));
            int first = lines.FindIndex(l => l.Trim().Length > 0);
            var values = lines.Skip(first + 1).Where(l => l.Trim().Length > 0).Select(l => Cell(l, column)).ToList();
            bool numbers = values.All(v => int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n >= 0);
            if (numbers)
            {
                int count = values.Count == 0 ? 0 : values.Max(v => int.Parse(v, CultureInfo.InvariantCulture)) + 1;
                classes = [.. Enumerable.Range(0, Math.Max(count, outputs)).Select(i => i.ToString(CultureInfo.InvariantCulture))];
            }
            else if (values.All(v => double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out _)))
            {
                throw new InvalidDataException($"'{targets[0]}' holds numbers that are not class indices (0, 1, 2, ...), but the network has {outputs} outputs and so classifies; "
                                               + "for regression make its last layer 1 wide (or give --task regression with one output per target).");
            }
            else
            {
                classes = [.. values.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)];
                var index = classes.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i.ToString(CultureInfo.InvariantCulture), StringComparer.Ordinal);
                var rebuilt = new StringBuilder();
                for (int i = 0; i < lines.Count; i++)
                {
                    if (i <= first || lines[i].Trim().Length == 0)
                    {
                        rebuilt.Append(lines[i]).Append('\n');
                        continue;
                    }

                    var cells = lines[i].Split(',');
                    cells[column] = index[cells[column].Trim().Trim('"')];
                    rebuilt.Append(string.Join(',', cells)).Append('\n');
                }

                text = rebuilt.ToString();
            }

            if (outputs == 1 && classes.Count > 2)
            {
                throw new InvalidDataException($"{settings.Data} has {classes.Count} classes in '{targets[0]}' but the network has one output (two classes); make its last layer {classes.Count} wide.");
            }

            if (outputs > 1 && classes.Count > outputs)
            {
                throw new InvalidDataException($"{settings.Data} has {classes.Count} classes in '{targets[0]}' but the network has {outputs} outputs; make its last layer {classes.Count} wide.");
            }
        }

        var data = Dataset.ParseCsv(text, new CsvOptions { TargetColumns = targets, IgnoreColumns = settings.Ignore });
        int expected = builder.InputShape.Aggregate(1, (a, b) => a * b);
        if (data.FeatureCount != expected)
        {
            throw new InvalidDataException($"The network takes {expected} input values [{Shape([.. builder.InputShape])}] but {settings.Data} has {data.FeatureCount} feature columns "
                                           + $"({string.Join(", ", data.FeatureNames.Take(12))}{(data.FeatureCount > 12 ? ", ..." : "")}); change the network's input or the columns (--target, --ignore).");
        }

        if (task == "regression" && outputs != data.TargetCount)
        {
            throw new InvalidDataException($"The network has {outputs} outputs but there are {data.TargetCount} target columns ({string.Join(", ", data.TargetNames)}).");
        }

        return (data, task, classes);
    }

    private static string Cell(string line, int column)
    {
        var cells = line.Split(',');
        return column < cells.Length ? cells[column].Trim().Trim('"') : "";
    }

    // ---------------------------------------------------------------- image folders

    // Class folders of images (folder/cat/*.png, folder/dog/*.png), fitted to the network's image input.
    private static (Dataset Data, string Task, IReadOnlyList<string>? Classes) LoadImages(string folder, NetworkBuilder builder)
    {
        if (builder.InputKind != InputKind.Image)
        {
            throw new UsageException($"{folder} is a folder of images, but the network takes {builder.InputKind.ToString().ToLowerInvariant()} input; start it with an image input (\"input\": \"Image\", \"shape\": [channels, height, width]).");
        }

        var (channels, height, width) = (builder.InputShape[0], builder.InputShape[1], builder.InputShape[2]);
        var images = new ImageFolderSource(folder, channels, height, width);
        if (images.Classes.Count < 2 || images.Count == 0)
        {
            throw new InvalidDataException($"{folder} needs a folder per class with images in them ({string.Join(", ", ImageFiles.DecodedExtensions)}).");
        }

        int outputs = builder.CurrentShape.Aggregate(1, (a, b) => a * b);
        if (outputs != images.Classes.Count)
        {
            throw new InvalidDataException($"{folder} has {images.Classes.Count} classes but the network has {outputs} outputs; make its last layer {images.Classes.Count} wide.");
        }

        var names = images.Classes;
        var data = Dataset.FromSource(images, [.. Enumerable.Range(0, channels * height * width).Select(i => $"x{i}")], names);
        return (data, "classification", names);
    }
}
