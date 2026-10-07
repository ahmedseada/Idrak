// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;

namespace Idrak.Cli.Commands.Train;

/// <summary><c>idrak train SPEC.json --data FILE</c>: trains a builder network on a CSV or an image folder and writes a model package.</summary>
internal sealed class TrainCommand : Command
{
    public override string Name => "train";

    public override string Summary => "Train a network from a builder JSON on a CSV or an image folder; writes a model package (.ikm)";

    public override string Usage => """
        SPEC.json --data FILE|FOLDER [-t COL]... [-o MODEL.ikm] [options]

        Arguments:
          SPEC.json  the network as the builder writes it (NetworkBuilder.ToJson, idrak suggest's network.json):
                     {"format": "idrak-network/1", "input": "Features", "shape": [9], "steps": [...]}

        Options:
              --data FILE        a CSV with a header (numbers; a class column may hold names), or a folder with a folder of
                                 images per class (PNG, BMP, PGM, PPM, or a --plugin's codec; fitted to the image input)
          -t, --target COL       the target column (repeatable; default: the last column)
              --ignore A,B       columns that are not features (an id, ...)
              --task T           regression or classify (default: classify when the network has several outputs)
              --epochs N         at most N epochs (default 100); --patience N: stop after N epochs without a better
                                 validation loss (default 20, 0: never)
              --lr F, --batch N  learning rate (default 0.001) and batch size (default 32)
              --optimizer NAME   adamw (default), adam or sgd; --weight-decay F
              --validation F     the fraction held out for validation (default 0.2, 0: none); --seed N (default 1)
              --no-scale         keep the features (and a regression target) unscaled
          -o, --out FILE         the model package (default: the network's name, or SPEC's, .ikm)
              --run DIR          the run folder (default CACHE/runs/TIME-NAME): run.json, network.json, scalers, the
                                 checkpoints last.ikw and best.ikw, and log.jsonl (idrak runs, idrak resume read them)
          -C, --config FILE      a train.json (from idrak suggest): the options above without the dashes
                                 ("target", "epochs", "lr", "batch", ...); options on the command line win

        Examples:
          idrak train network.json --data houses.csv -t price -o houses.ikm
          idrak train cnn.json --data ./shapes --epochs 40 -d vulkan:0
          idrak train network.json --data houses.csv --config train.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--data", "--target", "--ignore", "--task", "--epochs", "--lr", "--batch", "--patience", "--validation", "--optimizer", "--weight-decay", "--out", "--run"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-scale"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-t"] = "--target", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string specPath = context.Argument(0, "SPEC.json (the network, as the builder's JSON)");
        var network = JsonNode.Parse(File.ReadAllText(specPath), documentOptions: CliConfig.ReadOptions) as JsonObject
            ?? throw new UsageException($"{specPath} is not a JSON object.");
        if ((string?)network["format"] != "idrak-network/1" && network["network"] is JsonObject inner)
        {
            network = (JsonObject)inner.DeepClone();                              // a file that holds the network under "network"
        }

        var file = context.Option("--config") is { } configPath
            ? JsonNode.Parse(File.ReadAllText(configPath), documentOptions: CliConfig.ReadOptions) as JsonObject : null;
        string? Value(string option, params string[] keys)
        {
            if (context.Option(option) is { } given)
            {
                return given;
            }

            foreach (string key in keys.Prepend(option[2..]))
            {
                if (file?[key] is JsonValue v)
                {
                    return v.TryGetValue(out string? s) ? s : v.ToJsonString();
                }
            }

            return null;
        }

        IReadOnlyList<string> List(string option, params string[] keys)
        {
            var given = context.Options(option).SelectMany(v => v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
            if (given.Count > 0)
            {
                return given;
            }

            foreach (string key in keys.Prepend(option[2..]))
            {
                switch (file?[key])
                {
                    case JsonArray a:
                        return [.. a.Select(n => n?.ToString()).OfType<string>()];
                    case JsonValue v when v.TryGetValue(out string? s):
                        return s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                }
            }

            return [];
        }

        int Int(string option, int fallback, params string[] keys) => Value(option, keys) is not { } text ? fallback
            : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : throw new UsageException($"{option} needs a whole number, not '{text}'.");
        double Real(string option, double fallback, params string[] keys) => Value(option, keys) is not { } text ? fallback
            : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : throw new UsageException($"{option} needs a number, not '{text}'.");

        string data = Value("--data", "dataset") ?? throw new UsageException("Give --data FILE (a CSV) or --data FOLDER (a folder of class folders of images).");
        string name = (string?)network["name"] ?? Path.GetFileNameWithoutExtension(specPath);
        double validation = Real("--validation", 0.2, "validation_fraction", "validationFraction");
        if (validation is < 0 or >= 1)
        {
            throw new UsageException("--validation needs a fraction from 0 to 1.");
        }

        var settings = new RunSettings
        {
            Network = network,
            Data = Path.GetFullPath(data),
            Output = Path.GetFullPath(Value("--out") ?? $"{name}.ikm"),
            Targets = List("--target", "targets"),
            Ignore = List("--ignore"),
            Task = Value("--task"),
            Epochs = Int("--epochs", 100),
            LearningRate = (float)Real("--lr", 1e-3, "learning_rate", "learningRate"),
            Batch = Int("--batch", 32, "batch_size", "batchSize"),
            Patience = Value("--patience", "early_stop", "earlyStop") is null ? null : Int("--patience", 20, "early_stop", "earlyStop"),
            Validation = validation,
            Seed = Int("--seed", 1),
            Optimizer = Value("--optimizer") ?? "adamw",
            WeightDecay = Value("--weight-decay", "weight_decay", "weightDecay") is null ? null : (float)Real("--weight-decay", 0, "weight_decay", "weightDecay"),
            Scale = !context.Flag("--no-scale") && (bool?)file?["scale"] != false,
        };
        if (settings.Epochs <= 0 || settings.Batch <= 0)
        {
            throw new UsageException("--epochs and --batch need positive numbers.");
        }

        string folder = context.Option("--run") ?? NewRunFolder(context, name);
        return TrainSession.Run(context, settings, folder);
    }

    private static string NewRunFolder(CommandContext context, string name)
    {
        string safe = new([.. name.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '-')]);
        string folder = Path.Combine(RunLog.RunsFolder(context), $"{DateTime.Now:yyyyMMdd-HHmmss}-{safe}");
        for (int i = 2; Directory.Exists(folder); i++)
        {
            folder = Path.Combine(RunLog.RunsFolder(context), $"{DateTime.Now:yyyyMMdd-HHmmss}-{safe}-{i}");
        }

        return folder;
    }
}

/// <summary><c>idrak resume RUN</c>: continues a run from its last checkpoint.</summary>
internal sealed class ResumeCommand : Command
{
    public override string Name => "resume";

    public override string Summary => "Continue a training run from its last checkpoint (the remaining epochs, or --epochs N more)";

    public override string Usage => """
        RUN [--epochs N] [-o MODEL.ikm]

        Arguments:
          RUN  a run folder, its log.jsonl, or a name from idrak runs list

        Options:
              --epochs N  train N more epochs (default: what is left of the run's epochs)
          -o, --out FILE  the model package (default: the run's)

        The run's data, split, scalers and settings are reused and its log continues. The checkpoint holds the weights
        only: the optimizer's moments and the learning-rate schedule start again (the library does not save them).

        Examples:
          idrak resume 20261003-101500-houses
          idrak resume ./runs/houses --epochs 50 -d cuda:0
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--epochs", "--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        var run = RunLog.Find(context, context.Argument(0, "RUN (a run folder or a name from idrak runs list)"));
        if (run.Folder is null || run.Settings is null)
        {
            context.Error($"{run.Path} is a log without its run folder (run.json); only runs written by idrak train can be resumed.");
            return ExitCodes.Failed;
        }

        string checkpoint = Path.Combine(run.Folder, "last.ikw");
        if (!File.Exists(checkpoint))
        {
            context.Error($"{run.Folder} has no checkpoint (last.ikw): the run stopped before its first epoch ended. Train it again with idrak train.");
            return ExitCodes.Failed;
        }

        var settings = RunSettings.FromJson(run.Settings);
        if (context.Option("--out") is { } output)
        {
            settings = settings with { Output = output };
        }

        int done = run.Epochs.Count == 0 ? 0 : run.Epochs.Max(e => e.Epoch);
        int? more = context.Option("--epochs") is null ? null : context.IntOption("--epochs", 0);
        if (more is <= 0)
        {
            throw new UsageException("--epochs needs a positive number.");
        }

        if (more is { } extra)
        {
            settings = settings with { Epochs = done + extra };
        }

        context.Write($"Resuming  {run.Name} after epoch {done}");
        return TrainSession.Run(context, settings, run.Folder, checkpoint, done, more);
    }
}

/// <summary><c>idrak runs list</c>: the training runs under the runs folder.</summary>
internal sealed class RunsListCommand : Command
{
    public override string Name => "runs list";

    public override IReadOnlyCollection<string> Aliases { get; } = ["runs"];

    public override string Summary => "Training runs (from their JSON Lines logs): epochs, best epoch and loss, time, status";

    public override string Usage => """
        [FOLDER]

        Arguments:
          FOLDER  where the runs are (default CACHE/runs, where idrak train writes them); any folder of run folders
                  or of JSON Lines telemetry logs

        Examples:
          idrak runs list
          idrak runs list ./experiments --json
        """;

    public override int Run(CommandContext context)
    {
        string folder = context.Positional.Count > 0 ? context.Positional[0] : RunLog.RunsFolder(context);
        var runs = RunLog.All(folder);
        if (runs.Count == 0)
        {
            context.Write($"No runs in {folder} (idrak train writes them there).");
        }
        else
        {
            context.Table(["Run", "Started", "Epochs", "Best", "Best loss", "Final loss", "Time", "Status"], runs.Select(r => (IReadOnlyList<string>)
            [
                r.Name, r.Time?.ToLocalTime().ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture) ?? "", r.Epochs.Count.ToString(CultureInfo.InvariantCulture),
                r.Best?.Epoch.ToString(CultureInfo.InvariantCulture) ?? "-", RunLog.Format(r.Best is { } b ? b.ValidationLoss ?? b.Loss : null),
                RunLog.Format(r.Epochs.Count > 0 ? r.Epochs[^1].Loss : null), $"{r.TotalSeconds:F1} s", r.Status,
            ]));
        }

        context.WriteJson(new JsonObject { ["folder"] = folder, ["runs"] = new JsonArray([.. runs.Select(r => (JsonNode)r.ToJson(epochs: false))]) });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak runs show RUN</c>: one run's settings, loss curves and best epoch.</summary>
internal sealed class RunsShowCommand : Command
{
    public override string Name => "runs show";

    public override string Summary => "One training run: settings, loss curves (text plot), the epochs, best epoch and time";

    public override string Usage => """
        RUN

        Arguments:
          RUN  a run folder, a JSON Lines telemetry log, or a name from idrak runs list; -v lists every epoch

        Examples:
          idrak runs show 20261003-101500-houses
          idrak runs show training.jsonl --json
        """;

    public override int Run(CommandContext context)
    {
        var run = RunLog.Find(context, context.Argument(0, "RUN (a run folder, a log file or a name from idrak runs list)"));
        var best = run.Best;
        context.Write($"Run       {run.Name} ({run.Status}){(run.Time is { } t ? $", started {t.ToLocalTime():yyyy-MM-dd HH:mm}" : "")}");
        var started = run.Started;
        if (run.Settings is { } s)
        {
            context.Write($"Settings  data {s["data"]} · task {(string?)s["task"] ?? "auto"} · {s["optimizer"]} {s["learning_rate"]} · batch {s["batch"]} · epochs {s["epochs"]} · validation {s["validation"]} · seed {s["seed"]}");
        }

        if (started is not null)
        {
            context.Write($"Model     {started["model"]} · {started["parameters"]:N0} parameters · {started["optimizer"]} on {started["device"]} · {started["training_samples"]} training"
                          + (started["validation_samples"] is { } v ? $" / {v} validation samples" : " samples"));
        }

        context.Write($"Epochs    {run.Epochs.Count} in {run.TotalSeconds:F1} s · best epoch {best?.Epoch.ToString(CultureInfo.InvariantCulture) ?? "-"} "
                      + $"(loss {RunLog.Format(best?.Loss)}{(best?.ValidationLoss is { } bv ? $", val_loss {RunLog.Format(bv)}" : "")})");
        if (run.Epochs.Count > 0)
        {
            var series = new List<(string, IReadOnlyList<double?>)> { ("loss", [.. run.Epochs.Select(e => e.Loss)]) };
            if (run.Epochs.Any(e => e.ValidationLoss is not null))
            {
                series.Add(("val_loss", [.. run.Epochs.Select(e => e.ValidationLoss)]));
            }

            context.Write("");
            foreach (string line in TextPlot.Plot(series).TrimEnd('\n').Split('\n'))
            {
                context.Write(line);
            }

            context.Write("");
            var metrics = run.Epochs.SelectMany(e => e.ValidationMetrics.Keys.Select(k => "val_" + k).Concat(e.Metrics.Keys)).Distinct().ToList();
            var shown = context.Verbose || run.Epochs.Count <= 12 ? run.Epochs : [.. run.Epochs.Take(5), .. run.Epochs.TakeLast(5)];
            context.Table(["Epoch", "Loss", "Validation loss", .. metrics, "LR", "ms", ""], shown.Select(e => (IReadOnlyList<string>)
            [
                e.Epoch.ToString(CultureInfo.InvariantCulture), RunLog.Format(e.Loss), RunLog.Format(e.ValidationLoss),
                .. metrics.Select(m => RunLog.Format(m.StartsWith("val_", StringComparison.Ordinal) && e.ValidationMetrics.TryGetValue(m[4..], out double vm) ? vm
                    : e.Metrics.TryGetValue(m, out double tm) ? tm : null)),
                e.LearningRate.ToString("G3", CultureInfo.InvariantCulture), e.Milliseconds.ToString("F0", CultureInfo.InvariantCulture), e == best ? "best" : "",
            ]));
            if (shown.Count < run.Epochs.Count)
            {
                context.Write($"({run.Epochs.Count - shown.Count} epochs not shown; -v shows every one)");
            }
        }

        context.WriteJson(run.ToJson(epochs: true));
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak runs compare RUN RUN...</c>: runs side by side.</summary>
internal sealed class RunsCompareCommand : Command
{
    public override string Name => "runs compare";

    public override string Summary => "Training runs side by side: settings, best epoch and loss, time, and their validation curves";

    public override string Usage => """
        RUN RUN [RUN...]

        Arguments:
          RUN  run folders, JSON Lines telemetry logs, or names from idrak runs list

        Examples:
          idrak runs compare 20261003-101500-houses 20261003-103000-houses
          idrak runs compare a.jsonl b.jsonl --json
        """;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count < 2)
        {
            throw new UsageException("Give two or more runs to compare.");
        }

        var runs = context.Positional.Select(r => RunLog.Find(context, r)).ToList();
        string Setting(RunLog r, string key) => r.Settings?[key]?.ToString() ?? r.Started?[key]?.ToString() ?? "-";
        context.Table(["Run", "Epochs", "Best", "Best loss", "Final loss", "LR", "Batch", "Optimizer", "Time"], runs.Select(r => (IReadOnlyList<string>)
        [
            r.Name, r.Epochs.Count.ToString(CultureInfo.InvariantCulture), r.Best?.Epoch.ToString(CultureInfo.InvariantCulture) ?? "-",
            RunLog.Format(r.Best is { } b ? b.ValidationLoss ?? b.Loss : null), RunLog.Format(r.Epochs.Count > 0 ? r.Epochs[^1].Loss : null),
            Setting(r, "learning_rate"), r.Settings?["batch"]?.ToString() ?? Setting(r, "batch_size"), Setting(r, "optimizer"), $"{r.TotalSeconds:F1} s",
        ]));
        var winner = runs.Where(r => r.Best is not null).MinBy(r => r.Best!.ValidationLoss ?? r.Best.Loss);
        if (winner is not null)
        {
            context.Write($"\nLowest    {winner.Name} (epoch {winner.Best!.Epoch}, {RunLog.Format(winner.Best.ValidationLoss ?? winner.Best.Loss)})");
        }

        var series = runs.Select(r => (r.Name, (IReadOnlyList<double?>)[.. r.Epochs.Select(e => e.ValidationLoss ?? e.Loss)])).ToList();
        if (series.Any(s => s.Item2.Count > 0))
        {
            context.Write("");
            foreach (string line in TextPlot.Plot(series).TrimEnd('\n').Split('\n'))
            {
                context.Write(line);
            }
        }

        context.WriteJson(new JsonObject
        {
            ["runs"] = new JsonArray([.. runs.Select(r => (JsonNode)r.ToJson(epochs: true))]),
            ["lowest"] = winner?.Name,
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak predict MODEL.ikm --input FILE</c>: runs a trained package on new rows or images.</summary>
internal sealed class PredictCommand : Command
{
    public override string Name => "predict";

    public override string Summary => "Run a model package (.ikm) on new rows (CSV, JSON Lines, Parquet) or images and write the predictions";

    public override string Usage => """
        MODEL.ikm -i FILE|FOLDER [-o OUT]

        Options:
          -i, --input FILE  rows with the training's feature columns (by name; a target column is ignored), or an image
                            or a folder of images for an image model
          -o, --out FILE    write the rows with the predictions added (.csv, .jsonl, .json); without it they are shown
              --top N       classification: also give the N most likely classes (default 1)

        Examples:
          idrak predict houses.ikm -i new-houses.csv
          idrak predict shapes.ikm -i ./unlabelled -o predictions.csv --top 3
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--input", "--out", "--top"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-i"] = "--input", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string modelPath = context.Argument(0, "MODEL.ikm (a package written by idrak train or Predictor.Save)");
        string input = context.Option("--input") ?? (context.Positional.Count > 1 ? context.Positional[1] : throw new UsageException("Give -i, --input FILE with the rows to predict."));
        int top = context.IntOption("--top", 1);
        if (!File.Exists(modelPath))
        {
            throw new UsageException($"{modelPath} not found.");
        }

        using var package = ModelPackage.Open(modelPath);
        var meta = package.Contains(PackageEntryKind.Json, "training") ? package.Json("training") as JsonObject : null;
        // A package saved by Predictor.Save has no training entry; its predictor entry names the classes (and its
        // softmax setting matches what the classification output below applies).
        var predictor = meta is null && package.Contains(PackageEntryKind.Json, "predictor") ? package.Json("predictor") as JsonObject : null;
        var builder = package.Network();
        int[] shape = [.. builder.InputShape];
        int size = shape.Aggregate(1, (a, b) => a * b);
        string task = (string?)meta?["task"] ?? (predictor?["classes"] is JsonArray ? "classification" : "regression");
        var classes = (meta?["classes"] ?? predictor?["classes"]) is JsonArray c ? c.Select(n => (string)n!).ToList() : null;
        var targets = meta?["targets"] is JsonArray t ? t.Select(n => (string)n!).ToList() : ["prediction"];

        // The inputs: rows by feature name, or images.
        List<JsonObject> rows;
        float[] features;
        if ((string?)meta?["input"] == "images" || Directory.Exists(input) || ImageFiles.IsDecoded(input))
        {
            var files = Directory.Exists(input) ? Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories).Where(ImageFiles.IsDecoded).Order(StringComparer.Ordinal).ToList() : [input];
            if (builder.InputKind != InputKind.Image)
            {
                throw new UsageException($"{modelPath} takes {builder.InputKind.ToString().ToLowerInvariant()} input, not images; give a CSV or JSON Lines file.");
            }

            rows = [.. files.Select(f => new JsonObject { ["file"] = f })];
            features = [.. Dataset.FromSource(new ImageFolderSource(files, shape[0], shape[1], shape[2])).Features];
        }
        else
        {
            rows = [.. RowFiles.Read(input)];
            var names = meta?["features"] is JsonArray f ? f.Select(n => (string)n!).ToList() : null;
            features = new float[rows.Count * size];
            for (int r = 0; r < rows.Count; r++)
            {
                var values = names is not null
                    ? names.Select(n => RowFiles.Number(rows[r][n] ?? rows[r].FirstOrDefault(p => p.Key.Equals(n, StringComparison.OrdinalIgnoreCase)).Value)
                        ?? throw new InvalidDataException($"Row {r + 1} of {input} has no number in '{n}' (the model reads {string.Join(", ", names)}).")).ToList()
                    : rows[r].Select(p => RowFiles.Number(p.Value) ?? 0).Take(size).ToList();
                if (values.Count != size)
                {
                    throw new InvalidDataException($"Row {r + 1} of {input} has {values.Count} values; the model takes {size}.");
                }

                for (int i = 0; i < size; i++)
                {
                    features[r * size + i] = (float)values[i];
                }
            }

            if (package.Contains(PackageEntryKind.StandardScaler, "features") || package.Contains(PackageEntryKind.MinMaxScaler, "features"))
            {
                package.Scaler("features").Transform(features, size);
            }
        }

        if (rows.Count == 0)
        {
            context.Write($"{input} has no rows.");
            context.WriteJson(new JsonObject { ["model"] = modelPath, ["rows"] = 0, ["predictions"] = new JsonArray() });
            return ExitCodes.Ok;
        }

        // The network on the device, in batches.
        using var model = package.BuildNetwork(device: context.Device);
        int outputs = builder.CurrentShape.Aggregate(1, (a, b) => a * b);
        var output = new float[rows.Count * outputs];
        int batch = (int?)predictor?["batchSize"] ?? 512;
        for (int start = 0; start < rows.Count; start += batch)
        {
            int n = Math.Min(batch, rows.Count - start);
            using var x = Tensor.From(features.AsSpan(start * size, n * size).ToArray(), [n, .. shape], context.Device);
            using var y = model.Predict(x);
            y.ToArray().AsSpan(0, n * outputs).CopyTo(output.AsSpan(start * outputs));
        }

        if (task == "regression" && (package.Contains(PackageEntryKind.StandardScaler, "targets") || package.Contains(PackageEntryKind.MinMaxScaler, "targets")))
        {
            package.Scaler("targets").InverseTransform(output, outputs);
        }

        // The predictions added to each row.
        var results = new List<JsonObject>();
        for (int r = 0; r < rows.Count; r++)
        {
            var row = (JsonObject)rows[r].DeepClone();
            var values = output.AsSpan(r * outputs, outputs);
            if (task == "classification")
            {
                var probabilities = outputs == 1 ? [1 - Sigmoid(values[0]), Sigmoid(values[0])] : Softmax(values);
                var ranked = probabilities.Select((p, i) => (p, i)).OrderByDescending(p => p.p).ToList();
                string Label(int i) => classes is not null && i < classes.Count ? classes[i] : i.ToString(CultureInfo.InvariantCulture);
                row["prediction"] = Label(ranked[0].i);
                row["probability"] = Math.Round(ranked[0].p, 6);
                if (top > 1)
                {
                    row["top"] = new JsonArray([.. ranked.Take(top).Select(p => (JsonNode)new JsonObject { ["class"] = Label(p.i), ["probability"] = Math.Round(p.p, 6) })]);
                }
            }
            else
            {
                for (int k = 0; k < outputs; k++)
                {
                    row[outputs == 1 ? "prediction" : $"{(k < targets.Count ? targets[k] : $"output{k}")}_predicted"] = Math.Round(values[k], 6);
                }
            }

            results.Add(row);
        }

        if (context.Option("--out") is { } outFile)
        {
            RowFiles.Write(results, outFile);
            context.Write($"{results.Count:N0} predictions written to {outFile}");
        }
        else
        {
            var columns = results[0].Select(p => p.Key).Where(k => k != "top").ToList();
            // Image files relative to the input folder, shortened from the front, so the table shows which file is which.
            string Shown(JsonObject r, string k) => k == "file" && (string?)r[k] is { } file
                ? RowFiles.CellEnd(Directory.Exists(input) ? Path.GetRelativePath(input, file) : Path.GetFileName(file))
                : RowFiles.Cell(r[k], 16);
            context.Table(columns, results.Take(context.Verbose ? int.MaxValue : 20).Select(r => (IReadOnlyList<string>)[.. columns.Select(k => Shown(r, k))]));
            if (results.Count > 20 && !context.Verbose)
            {
                context.Write($"({results.Count - 20} more rows; -o FILE writes them all, -v shows them)");
            }
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = modelPath,
            ["task"] = task,
            ["rows"] = results.Count,
            ["output"] = context.Option("--out"),
            ["predictions"] = new JsonArray([.. results.Select(r => (JsonNode)r.DeepClone())]),
        });
        return ExitCodes.Ok;
    }

    private static double Sigmoid(float x) => 1 / (1 + Math.Exp(-x));

    private static double[] Softmax(ReadOnlySpan<float> values)
    {
        float max = float.NegativeInfinity;
        foreach (float v in values)
        {
            max = Math.Max(max, v);
        }

        var result = new double[values.Length];
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += result[i] = Math.Exp(values[i] - max);
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }
}

/// <summary><c>idrak package --model DIR --out MODEL.ikm</c>: bundles a network, its weights, scalers and tokenizer into one package.</summary>
internal sealed class PackageCommand : Command
{
    public override string Name => "package";

    public override string Summary => "Bundle a network, its weights, scalers and tokenizer from a folder into one model package (.ikm)";

    public override string Usage => """
        --model DIR -o MODEL.ikm [--checkpoint best|last|FILE]

        Options:
          -m, --model DIR     a folder with network.json and weights (.ikw): a run folder of idrak train, or your own;
                              also taken when present: features.txt and targets.txt (scalers), tokenizer.json (a
                              character or word tokenizer), training.json (what idrak predict reads)
              --checkpoint W  which weights: best (default when present), last, or a .ikw file
          -o, --out FILE      the package to write

        Examples:
          idrak package --model ~/.cache/idrak/runs/20261003-101500-houses -o houses-last.ikm --checkpoint last
          idrak package --model ./my-model -o my-model.ikm
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--model", "--out", "--checkpoint"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-m"] = "--model", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string folder = context.Option("--model") ?? (context.Positional.Count > 0 ? context.Positional[0] : throw new UsageException("Give --model DIR, the folder to package."));
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out MODEL.ikm.");
        if (!Directory.Exists(folder))
        {
            throw new UsageException($"{folder} is not a folder.");
        }

        string networkFile = Path.Combine(folder, "network.json");
        if (!File.Exists(networkFile))
        {
            context.Error($"{folder} has no network.json (the network as the builder's JSON).");
            return ExitCodes.Failed;
        }

        string choice = context.Option("--checkpoint") ?? "best";
        string weights = choice switch
        {
            "best" when File.Exists(Path.Combine(folder, "best.ikw")) => Path.Combine(folder, "best.ikw"),
            "best" or "last" when File.Exists(Path.Combine(folder, "last.ikw")) => Path.Combine(folder, "last.ikw"),
            "best" or "last" => Directory.GetFiles(folder, "*.ikw").Order(StringComparer.Ordinal).FirstOrDefault()
                ?? throw new InvalidDataException($"{folder} has no weights (.ikw)."),
            var file => File.Exists(file) ? file : Path.Combine(folder, file) is var inside && File.Exists(inside) ? inside
                : throw new UsageException($"--checkpoint {file}: no such weights file."),
        };

        var network = JsonNode.Parse(File.ReadAllText(networkFile)) as JsonObject ?? throw new InvalidDataException($"{networkFile} is not a JSON object.");
        using var model = Network.FromJson(network).Build();
        model.Load(weights);
        var scalers = new Dictionary<string, IScaler>();
        foreach (string name in new[] { "features", "targets" })
        {
            string file = Path.Combine(folder, name + ".txt");
            if (File.Exists(file))
            {
                scalers[name] = Scaler(file);
            }
        }

        string metaFile = Path.Combine(folder, "training.json");
        var meta = File.Exists(metaFile) ? JsonNode.Parse(File.ReadAllText(metaFile)) as JsonObject : null;
        string tokenizerFile = Path.Combine(folder, "tokenizer.json");
        var tokenizer = File.Exists(tokenizerFile) ? Tokenizers.Load(tokenizerFile) : null;
        TrainSession.Package(model, network, scalers.GetValueOrDefault("features"), scalers.GetValueOrDefault("targets"), meta, output, tokenizer);
        List<string> included = ["network.json", Path.GetFileName(weights), .. scalers.Keys.Select(k => k + ".txt")];
        if (meta is not null)
        {
            included.Add("training.json");
        }

        if (tokenizer is not null)
        {
            included.Add("tokenizer.json");
        }

        context.Write($"Wrote {output}: {string.Join(", ", included)} ({model.ParameterCount:N0} parameters)");
        context.WriteJson(new JsonObject
        {
            ["package"] = Path.GetFullPath(output),
            ["weights"] = weights,
            ["included"] = new JsonArray([.. included.Select(i => (JsonNode)i)]),
            ["parameters"] = model.ParameterCount,
        });
        return ExitCodes.Ok;
    }

    private static IScaler Scaler(string file)
    {
        try
        {
            return StandardScaler.Load(file);
        }
        catch (Exception e) when (e is FormatException or InvalidDataException or IndexOutOfRangeException)
        {
            return MinMaxScaler.Load(file);
        }
    }
}
