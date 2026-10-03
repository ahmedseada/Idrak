// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Cli.Shared;

/// <summary>What the person asked of <c>idrak suggest</c>.</summary>
internal sealed record DesignOptions(string? Target, string? TextColumn, string? Task, string? Budget, long? MaxParams, Device Device, BaseModelInfo? Base);

/// <summary>
/// The sizes a network is made from, so the search can vary them: the width (first hidden width of an MLP, filters
/// of the first convolution, embedding width of a text or language model), the depth (hidden layers, convolution
/// blocks, transformer layers; 0 for a text model is the bag-of-embeddings model), dropout and learning rate.
/// </summary>
internal sealed record Variant(int Width, int Depth, float Dropout, float LearningRate)
{
    public override string ToString() => FormattableString.Invariant($"width {Width}, depth {Depth}, dropout {Dropout:0.##}, learning rate {LearningRate:0.####}");
}

/// <summary>One decision and the rule that made it (printed by <c>--explain</c>).</summary>
internal sealed record Reason(string Area, string Choice, string Rule);

/// <summary>A proposed setup: the files <c>idrak suggest</c> writes and the lines it prints.</summary>
internal sealed class DesignPlan
{
    public required string Task { get; init; }

    public List<(string Label, string Text)> Lines { get; } = [];

    public List<Reason> Reasons { get; } = [];

    public List<string> Warnings { get; } = [];

    /// <summary>The prepared data's description (prep.json), or null when nothing is prepared (a base-model setup).</summary>
    public JsonObject? Prep { get; set; }

    /// <summary>The training setup (train.json), for networks.</summary>
    public JsonObject? Train { get; set; }

    /// <summary>The fine-tuning setup (tune.json), with a base model.</summary>
    public JsonObject? Tune { get; set; }

    /// <summary>The first guess.</summary>
    public Variant? Variant { get; set; }

    /// <summary>Makes the network of a variant (null for a base-model setup).</summary>
    public Func<Variant, NetworkBuilder>? Make { get; init; }

    /// <summary>A variant's network as the builder's JSON.</summary>
    public JsonObject? Network(Variant v) => Make?.Invoke(v).ToJson();

    /// <summary>The parameter cap the rules sized the network to.</summary>
    public long ParameterCap { get; set; }

    public void Why(string area, string choice, string rule) => Reasons.Add(new Reason(area, choice, rule));
}

/// <summary>
/// Steps 2 and 3 of <c>idrak suggest</c> (plans/idrak-cli.md, "Design a model"): picks the task and proposes a network,
/// its data preparation and its training from fixed rules sized by the data and the device. The rules are
/// deterministic (the same data and options give the same files) and each choice records its rule. The numbers are
/// rules of thumb, documented where they are applied; <c>--search</c> measures around them.
/// </summary>
internal static class DesignRules
{
    // Target column names tried, in order, when --target is not given (else the last column).
    private static readonly string[] TargetNames = ["target", "label", "labels", "class", "y", "output", "price", "score", "rating"];

    /// <summary>Proposes a setup for <paramref name="profile"/>.</summary>
    public static DesignPlan Propose(DataProfile profile, DesignOptions options)
    {
        var design = profile.Kind switch
        {
            DataKind.Images => Images(profile, options),
            DataKind.Chat when options.Base is not null => BaseTune(profile, options, "chat"),
            DataKind.Chat => LanguageModel(profile, options),
            DataKind.Preference when options.Base is not null => BaseTune(profile, options, "preference"),
            DataKind.Preference => throw new UsageException("Preference pairs train a base model (DPO): add --base MODEL."),
            _ => Table(profile, options),
        };
        if (design.Variant is { } v && design.Make is not null)
        {
            CheckMemory(design, NetworkAnalysis.Of(design.Network(v)!), options.Device, (int?)design.Train?["batchSize"] ?? 32);
        }

        if (profile.Truncated)
        {
            design.Warnings.Add($"read the first {profile.Rows.Count:N0} of {profile.TotalRows:N0} rows; the statistics come from them");
        }

        return design;
    }

    /// <summary>
    /// The parameter cap: a number of parameters per training example by task (tables 50, images 100, text 200,
    /// language models 1 per character), clamped to a range, then divided by 4 for --budget small or multiplied by 4 for
    /// large; --max-params replaces it. Small data gets a small network, so it cannot simply memorize the rows.
    /// </summary>
    public static long Cap(DesignOptions options, double examples, double perExample, long min, long max)
    {
        if (options.MaxParams is { } given)
        {
            return given;
        }

        long cap = (long)Math.Clamp(examples * perExample, min, max);
        return options.Budget switch
        {
            "small" => Math.Max(1_000, cap / 4),
            "large" => cap * 4,
            _ => cap,
        };
    }

    /// <summary>The number of candidates <c>--search N</c> tries: the first guess, then fixed variations in a fixed order.</summary>
    public static List<Variant> Candidates(DesignPlan design, int count)
    {
        var v = design.Variant!;
        int minDepth = design.Task is "sequence" ? 0 : 1;
        var list = new List<Variant>
        {
            v,
            v with { Width = v.Width * 2 },
            v with { Dropout = v.Dropout > 0 ? 0 : 0.2f },
            v with { Depth = v.Depth + 1 },
            v with { Width = Math.Max(8, v.Width / 2) },
            v with { LearningRate = v.LearningRate * 3 },
            v with { Depth = Math.Max(minDepth, v.Depth - 1) },
            v with { LearningRate = v.LearningRate / 3 },
            v with { Width = v.Width * 2, Depth = v.Depth + 1 },
            v with { Width = v.Width * 2, Dropout = v.Dropout > 0 ? 0 : 0.2f },
        };
        var valid = new List<Variant>();
        foreach (var candidate in list.Distinct())
        {
            try
            {
                design.Make!(candidate).ToJson();                                       // shapes check: e.g. too many pools for the image
                valid.Add(candidate);
            }
            catch (InvalidOperationException)
            {
            }
        }

        return [.. valid.Take(Math.Max(1, count))];
    }

    // ------------------------------------------------------------------ tables and text columns

    private static DesignPlan Table(DataProfile profile, DesignOptions options)
    {
        var warnings = new List<string>();
        var target = TargetColumn(profile, options, warnings);
        var targetSpec = TargetSpec(profile, target, options.Task, out string task, out string targetRule);
        ColumnProfile? text = options.TextColumn is { } textName
            ? profile.Column(textName) ?? throw new UsageException($"--text {textName}: no such column (columns: {string.Join(", ", profile.Columns.Select(c => c.Name))}).")
            : options.Task == "sequence" ? profile.Columns.Where(c => c != target && c.Type == "text").MaxBy(c => c.MeanWords) : null;
        if (options.Task == "sequence" && text is null)
        {
            throw new UsageException("--task sequence needs a text column: name it with --text COL.");
        }

        int rows = profile.Rows.Count(r => r[target.Name] is not null);
        var design = text is not null
            ? Text(profile, options, target, targetSpec, task, text)
            : Tabular(profile, options, target, targetSpec, task);
        design.Warnings.AddRange(warnings);
        design.Why("task", design.Lines.First(l => l.Label == "Task").Text, targetRule);
        Balance(design, profile, target, targetSpec);
        if (rows < 50)
        {
            design.Warnings.Add($"only {rows} rows with a target: any network will be unreliable; collect more data or use a simple model");
        }
        else if (rows < 200)
        {
            design.Warnings.Add($"few rows ({rows}): strong regularization chosen (dropout, weight decay, a small network)");
        }

        return design;
    }

    private static ColumnProfile TargetColumn(DataProfile profile, DesignOptions options, List<string> warnings)
    {
        if (options.Target is { } name)
        {
            return profile.Column(name) ?? throw new UsageException($"--target {name}: no such column (columns: {string.Join(", ", profile.Columns.Select(c => c.Name))}).");
        }

        var guess = TargetNames.Select(n => profile.Column(n)).FirstOrDefault(c => c is not null) ?? profile.Columns[^1];
        warnings.Add($"no --target given: guessed '{guess.Name}' (a column named like a target, else the last column)");
        return guess;
    }

    // The target: a number (regression) or classes. Rule: a category or boolean column is classes; a numeric column
    // with at most 20 distinct whole values, each value seen 5 times on average, is classes; any other number is a
    // regression target, log-transformed when every value is positive and the skew is above 1.
    private static JsonObject TargetSpec(DataProfile profile, ColumnProfile target, string? taskOption, out string task, out string rule)
    {
        if (target.Type is "text" or "other" && taskOption is not "classify")
        {
            throw new UsageException($"The target '{target.Name}' holds {(target.Type == "text" ? "free text" : "lists or objects")}; choose another --target.");
        }

        bool numeric = target.Type == "number";
        bool fewValues = target.Distinct.Count <= 20 && !target.ManyDistinct && target.Integers && target.Present >= target.Distinct.Count * 5;
        bool classes = taskOption switch
        {
            "regression" => false,
            "classify" => true,
            _ => !numeric || fewValues || target.Booleans > 0,
        };
        if (classes)
        {
            var keys = profile.Rows.Select(r => DataPreparation.Key(r[target.Name])).OfType<string>().Distinct().ToList();
            bool allNumbers = keys.All(k => double.TryParse(k, NumberStyles.Float, CultureInfo.InvariantCulture, out _));
            keys = allNumbers ? [.. keys.OrderBy(k => double.Parse(k, CultureInfo.InvariantCulture))] : [.. keys.Order(StringComparer.Ordinal)];
            if (keys.Count < 2)
            {
                throw new UsageException($"The target '{target.Name}' has a single value; nothing to learn.");
            }

            task = "classify";
            rule = taskOption == "classify" ? "--task classify" : numeric
                ? $"a numeric target with at most 20 distinct whole values (here {keys.Count}) seen 5 times each on average is classes"
                : "a target of words or booleans is classes";
            return new JsonObject { ["column"] = target.Name, ["type"] = "classes", ["classes"] = new JsonArray([.. keys.Select(k => (JsonNode)k)]) };
        }

        if (!numeric)
        {
            throw new UsageException($"--task regression needs a numeric target; '{target.Name}' holds words.");
        }

        var values = profile.Rows.Select(r => DataPreparation.Number(r[target.Name])).OfType<double>().ToList();
        bool log = values.Count > 0 && values.Min() > 0 && target.Skew > 1;
        var scaled = log ? values.Select(v => Math.Log(v)).ToList() : values;
        double mean = scaled.Average(), std = Math.Sqrt(scaled.Sum(v => (v - mean) * (v - mean)) / Math.Max(1, scaled.Count - 1));
        task = "regression";
        rule = taskOption == "regression" ? "--task regression"
            : $"a numeric target with more than 20 distinct values (here {(target.ManyDistinct ? "10,000+" : target.Distinct.Count.ToString("N0", CultureInfo.InvariantCulture))}) or fractions is a number";
        return new JsonObject
        {
            ["column"] = target.Name, ["type"] = "number", ["log"] = log, ["mean"] = Round(mean), ["std"] = Round(std == 0 ? 1 : std),
            ["skew"] = Round(target.Skew),
        };
    }

    private static DesignPlan Tabular(DataProfile profile, DesignOptions options, ColumnProfile target, JsonObject targetSpec, string task)
    {
        var design = new DesignPlan { Task = task, Make = null! };
        var features = new JsonArray();
        var dropped = new JsonArray();
        int numbers = 0, categories = 0, width = 0;
        foreach (var column in profile.Columns.Where(c => c != target))
        {
            string? reason = DropReason(column, profile.Rows.Count);
            if (reason is not null)
            {
                dropped.Add(new JsonObject { ["column"] = column.Name, ["reason"] = reason });
                if (column.Type == "text")
                {
                    design.Warnings.Add($"'{column.Name}' holds free text and is not used; --text {column.Name} designs a text model");
                }

                continue;
            }

            if (column.Type == "number")
            {
                numbers++;
                width++;
                var values = profile.Rows.Select(r => DataPreparation.Number(r[column.Name])).OfType<double>().ToList();
                double median = ColumnProfile.Quantile(values, 0.5), mean = values.Count == 0 ? 0 : values.Average();
                double std = values.Count < 2 ? 1 : Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1));
                features.Add(new JsonObject { ["column"] = column.Name, ["type"] = "number", ["fill"] = Round(median), ["mean"] = Round(mean), ["std"] = Round(std == 0 ? 1 : std) });
            }
            else
            {
                // One-hot of the 32 most frequent values (ties by name); rarer values share the all-zeros row.
                categories++;
                var top = column.Distinct.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal).Take(32).Select(p => p.Key).ToList();
                width += top.Count;
                features.Add(new JsonObject { ["column"] = column.Name, ["type"] = "onehot", ["values"] = new JsonArray([.. top.Select(t => (JsonNode)t)]) });
            }
        }

        if (width == 0)
        {
            throw new UsageException($"No usable feature columns besides the target '{target.Name}' ({dropped.Count} dropped: {string.Join(", ", dropped.Select(d => $"{d!["column"]} ({d["reason"]})"))}).");
        }

        int rows = profile.Rows.Count(r => r[target.Name] is not null);
        double train = rows * 0.8;
        int outputs = (string?)targetSpec["type"] == "classes" ? targetSpec["classes"]!.AsArray().Count : 1;
        design.Prep = new JsonObject
        {
            ["format"] = "idrak-prep/1", ["kind"] = "table", ["source"] = profile.Path, ["target"] = targetSpec, ["features"] = features, ["dropped"] = dropped,
            ["split"] = new JsonObject { ["validation"] = 0.2, ["seed"] = 1 },
        };

        // Width: the next power of two at or above twice the features, between 16 and 256. Depth: one hidden layer under
        // 500 rows, two under 50,000, three above. Dropout: 0.2 under 1,000 rows, 0.1 under 10,000, none above.
        int first = Math.Clamp(NextPowerOfTwo(2 * width), 16, 256);
        int depth = rows < 500 ? 1 : rows < 50_000 ? 2 : 3;
        float dropout = rows < 1_000 ? 0.2f : rows < 10_000 ? 0.1f : 0f;
        Func<Variant, NetworkBuilder> make = v => Mlp(width, outputs, v);
        long cap = Cap(options, train, 50, 2_000, 5_000_000);
        var variant = FitToCap(new Variant(first, depth, dropout, 1e-3f), make, cap, minWidth: 8);
        design = Copy(design, make);
        design.Variant = variant;
        design.ParameterCap = cap;
        var analysis = NetworkAnalysis.Of(make(variant).ToJson());
        Training(design, profile, task, train, rows, targetSpec, isImage: false);
        var hidden = Enumerable.Range(0, variant.Depth).Select(i => Math.Max(8, variant.Width >> i));

        string dataLine = $"{profile.TotalRows:N0} rows · {profile.Columns.Count} columns ({profile.Columns.Count(c => c.Type == "number")} numeric, {profile.Columns.Count(c => c.Type == "category")} categories"
            + (profile.Columns.Any(c => c.Type == "text") ? $", {profile.Columns.Count(c => c.Type == "text")} text" : "") + $") · {profile.MissingShare * 100:0.#}% missing";
        design.Lines.Add(("Data", dataLine));
        design.Lines.Add(("Task", task == "classify" ? $"classification, {outputs} classes (target '{target.Name}')" : $"regression (target '{target.Name}' is a number)"));
        string prepLine = $"standard-scale {numbers} numbers · one-hot {categories} categories (-> {width} features) · fill missing with medians"
            + ((bool?)targetSpec["log"] == true ? $" · log-transform the target (skew {(double)targetSpec["skew"]!:0.0})" : "")
            + $" · {profile.Duplicates} duplicates · 80/20 split, seed 1" + (dropped.Count > 0 ? $" · dropped {string.Join(", ", dropped.Select(d => (string)d!["column"]!))}" : "");
        design.Lines.Add(("Prep", prepLine));
        design.Lines.Add(("Network", $"MLP {width} -> {string.Join(" -> ", hidden)} -> {outputs}, ReLU" + (variant.Dropout > 0 ? $", dropout {variant.Dropout:0.##}" : "")
            + $" (about {DeviceMemory.Count(analysis.Parameters)} parameters; cap {DeviceMemory.Count(cap)} for {rows:N0} rows)"));
        design.Why("prep", "standard-scale numbers, one-hot categories", "numbers are centred and scaled so no feature dominates; a category becomes one 0/1 feature per value (its 32 most frequent values)");
        design.Why("prep", "fill missing numbers with the median", "the median is not pulled by outliers; a missing category is all zeros");
        if ((bool?)targetSpec["log"] == true)
        {
            design.Why("prep", "log-transform the target", "every target value is positive and the skew is above 1: the logarithm makes errors relative");
        }

        design.Why("network", $"width {variant.Width}", "the next power of two at or above twice the features, between 16 and 256, halved until the parameters fit the cap");
        design.Why("network", $"{variant.Depth} hidden layer{(variant.Depth == 1 ? "" : "s")}", "one under 500 rows, two under 50,000, three above");
        design.Why("network", variant.Dropout > 0 ? $"dropout {variant.Dropout:0.##}" : "no dropout", "0.2 under 1,000 rows, 0.1 under 10,000, none above");
        design.Why("network", $"cap {DeviceMemory.Count(cap)} parameters", "50 per training row, between 2k and 5M (--budget small ÷4, large ×4, --max-params replaces it)");
        foreach (var d in dropped)
        {
            design.Why("prep", $"drop '{(string)d!["column"]!}'", (string)d["reason"]!);
        }

        return design;
    }

    // Why a column is not a feature, or null when it is one.
    private static string? DropReason(ColumnProfile column, int rows)
    {
        if (column.Present == 0)
        {
            return "empty";
        }

        if (column.Type is "other")
        {
            return "holds lists or objects";
        }

        if (column.Type is "text")
        {
            return "free text (use --text)";
        }

        if (column.Missing * 2 > rows)
        {
            return $"mostly missing ({column.Missing * 100 / Math.Max(1, rows)}%)";
        }

        if (column.Distinct.Count <= 1 && !column.ManyDistinct)
        {
            return "a single value";
        }

        bool unique = column.ManyDistinct || column.Distinct.Count * 20 >= column.Present * 19;
        bool named = column.Name.Equals("id", StringComparison.OrdinalIgnoreCase) || column.Name.EndsWith("_id", StringComparison.OrdinalIgnoreCase)
                     || column.Name.EndsWith("Id", StringComparison.Ordinal) || column.Name.Equals("index", StringComparison.OrdinalIgnoreCase);
        if (column.Present >= 20 && (column.Type == "number" ? unique && named && column.Integers : column.Distinct.Count * 2 >= column.Present || column.ManyDistinct))
        {
            return "looks like an identifier (a different value in almost every row)";
        }

        return null;
    }

    private static DesignPlan Text(DataProfile profile, DesignOptions options, ColumnProfile target, JsonObject targetSpec, string task, ColumnProfile text)
    {
        // Words of two or more occurrences, most frequent first (ties by spelling), at most 20,000 (5,000 small, 50,000
        // large); every word when fewer than 50 occur twice. Length: the 95th percentile of words per row, 4 to 256.
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var lengths = new List<double>();
        foreach (var row in profile.Rows)
        {
            var words = DataPreparation.Words(DataPreparation.Key(row[text.Name]) ?? "").ToList();
            lengths.Add(words.Count);
            foreach (var w in words)
            {
                counts[w] = counts.GetValueOrDefault(w) + 1;
            }
        }

        int maxVocabulary = options.Budget switch { "small" => 5_000, "large" => 50_000, _ => 20_000 };
        int minCount = counts.Count(p => p.Value >= 2) >= 50 ? 2 : 1;
        var vocabulary = counts.Where(p => p.Value >= minCount).OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal)
            .Take(maxVocabulary).Select(p => p.Key).ToList();
        int length = Math.Clamp((int)ColumnProfile.Quantile(lengths, 0.95), 4, 256);
        int v = vocabulary.Count + 2;
        int rows = profile.Rows.Count(r => r[target.Name] is not null);
        int outputs = (string?)targetSpec["type"] == "classes" ? targetSpec["classes"]!.AsArray().Count : 1;
        Func<Variant, NetworkBuilder> make = variant => TextNetwork(v, length, outputs, variant);

        // Under 5,000 rows: mean of word embeddings (a bag of words, robust on little data), width 32 under 1,000 rows
        // else 64. From 5,000 rows: two transformer layers of width 64 (128 from 50,000 rows), learning rate 5e-4.
        var first = rows < 5_000
            ? new Variant(rows < 1_000 ? 32 : 64, 0, rows < 1_000 ? 0.2f : 0.1f, 1e-3f)
            : new Variant(rows < 50_000 ? 64 : 128, 2, 0.1f, 5e-4f);
        long cap = Cap(options, rows * 0.8, 200, 20_000, 20_000_000);
        var variant = FitToCap(first, make, cap, minWidth: 16);
        var design = new DesignPlan { Task = "sequence", Make = make, Variant = variant, ParameterCap = cap };
        design.Prep = new JsonObject
        {
            ["format"] = "idrak-prep/1", ["kind"] = "text", ["source"] = profile.Path, ["column"] = text.Name, ["target"] = targetSpec,
            ["tokenizer"] = new JsonObject
            {
                ["type"] = "words", ["lowercase"] = true, ["length"] = length,
                ["vocabulary"] = new JsonArray([.. new[] { "<pad>", "<unk>" }.Concat(vocabulary).Select(w => (JsonNode)w)]),
            },
            ["split"] = new JsonObject { ["validation"] = 0.2, ["seed"] = 1 },
        };
        var analysis = NetworkAnalysis.Of(make(variant).ToJson());
        Training(design, profile, task, rows * 0.8, rows, targetSpec, isImage: false);
        design.Lines.Add(("Data", $"{profile.TotalRows:N0} rows · text column '{text.Name}': {text.MeanWords:0} words on average, {text.MaxWords:N0} longest · {counts.Count:N0} distinct words"));
        design.Lines.Add(("Task", $"text {(task == "classify" ? $"classification, {outputs} classes" : "regression")} (target '{target.Name}', text '{text.Name}')"));
        design.Lines.Add(("Prep", $"lower-case words · vocabulary {v:N0} (words seen {minCount}+ times) · {length} words per row (95th percentile), cut or padded · 80/20 split, seed 1"));
        design.Lines.Add(("Network", (variant.Depth == 0
            ? $"embedding {v:N0} x {variant.Width} -> mean over words -> linear {variant.Width} -> ReLU -> linear {outputs}"
            : $"embedding {v:N0} x {variant.Width} -> positions -> transformer x {variant.Depth} ({Heads(variant.Width)} heads) -> layer norm -> mean -> linear {outputs}")
            + $" (about {DeviceMemory.Count(analysis.Parameters)} parameters)"));
        design.Why("prep", $"{length} words per row", "the 95th percentile of words per row, between 4 and 256: longer rows are cut");
        design.Why("prep", $"vocabulary of {v:N0}", "words seen at least twice (every word when fewer than 50 are), at most 20,000 (--budget small 5,000, large 50,000), plus padding and unknown");
        design.Why("network", variant.Depth == 0 ? "mean of word embeddings" : $"{variant.Depth} transformer layers",
            "under 5,000 rows the mean of word embeddings (few parameters besides the table); from 5,000 rows two transformer layers");
        design.Why("network", $"width {variant.Width}", "32 under 1,000 rows, 64 under 50,000, 128 above; halved until the parameters fit the cap (200 per training row)");
        return design;
    }

    // ------------------------------------------------------------------ images

    private static DesignPlan Images(DataProfile profile, DesignOptions options)
    {
        int classes = profile.ClassNames.Count;
        if (classes < 2)
        {
            throw new UsageException($"{profile.Path} has one class folder; image classification needs two or more (one folder per class).");
        }

        // Size: the median width and height; scaled down so the longer side is at most 64 (128 with --budget large).
        // Channels: the most common count, alpha dropped (gray stays gray, colour stays colour).
        var widths = profile.Images.Select(i => (double)i.Info.Width).ToList();
        var heights = profile.Images.Select(i => (double)i.Info.Height).ToList();
        int w = (int)ColumnProfile.Quantile(widths, 0.5), h = (int)ColumnProfile.Quantile(heights, 0.5);
        int limit = options.Budget == "large" ? 128 : 64;
        double scale = Math.Min(1, limit / (double)Math.Max(w, h));
        int width = Math.Max(4, (int)Math.Round(w * scale)), height = Math.Max(4, (int)Math.Round(h * scale));
        int channels = profile.Images.GroupBy(i => i.Info.Channels is 2 ? 1 : i.Info.Channels >= 3 ? 3 : 1).OrderByDescending(g => g.Count()).ThenBy(g => g.Key).First().Key;
        bool sameSize = widths.Distinct().Count() == 1 && heights.Distinct().Count() == 1;
        int n = profile.Images.Count;
        var perClass = Enumerable.Range(0, classes).Select(c => profile.Images.Count(i => i.Class == c)).ToArray();
        double balance = (perClass.Max() - perClass.Min()) / (double)perClass.Average();

        // Blocks: one per halving until the side is about 4 pixels (log2(min side / 4)), between 1 and 4. Filters: 16
        // under 5,000 images, else 32, doubling per block up to 256. Dropout 0.2 under 10,000 images, else 0.
        int blocks = Math.Clamp((int)Math.Floor(Math.Log2(Math.Min(width, height) / 4.0)), 1, 4);
        var first = new Variant(n < 5_000 ? 16 : 32, blocks, n < 10_000 ? 0.2f : 0f, 3e-3f);
        Func<Variant, NetworkBuilder> make = v => Cnn(channels, height, width, classes, v);
        long cap = Cap(options, n * 0.8, 100, 20_000, 20_000_000);
        var variant = FitToCap(first, make, cap, minWidth: 8);
        var design = new DesignPlan { Task = "image", Make = make, Variant = variant, ParameterCap = cap };
        var augment = n < 10_000 ? new JsonArray("flip", "rotate 10", "shift 2") : new JsonArray();
        design.Prep = new JsonObject
        {
            ["format"] = "idrak-prep/1", ["kind"] = "images", ["source"] = profile.Path, ["channels"] = channels, ["height"] = height, ["width"] = width,
            ["scale"] = "0-1", ["classes"] = new JsonArray([.. profile.ClassNames.Select(c => (JsonNode)c)]), ["augment"] = augment,
            ["split"] = new JsonObject { ["validation"] = 0.2, ["seed"] = 1 },
        };
        var spec = new JsonObject { ["type"] = "classes", ["classes"] = design.Prep["classes"]!.DeepClone() };
        Training(design, profile, "image", n * 0.8, n, spec, isImage: true);
        var analysis = NetworkAnalysis.Of(make(variant).ToJson());
        string size = sameSize ? $"{w}x{h}" : $"{widths.Min()}x{heights.Min()} to {widths.Max()}x{heights.Max()} (median {w}x{h})";
        design.Lines.Add(("Data", $"{n:N0} images in {classes} class folders · {size} {(channels == 1 ? "grayscale" : "colour")} · "
            + (balance <= 0.1 ? $"balanced (±{balance * 50:0}%)" : $"unbalanced ({perClass.Min()} to {perClass.Max()} per class)")));
        design.Lines.Add(("Task", $"image classification, {classes} classes"));
        design.Lines.Add(("Prep", $"{(sameSize && scale == 1 ? "" : $"resize to {width}x{height} · ")}scale to [0, 1]" + (augment.Count > 0 ? " · augment: flips, ±10° rotations, ±2 px shifts (small set)" : "")));
        design.Lines.Add(("Network", $"[conv {variant.Width} -> batch norm -> ReLU -> pool] x {variant.Depth} -> global average pool -> {(variant.Dropout > 0 ? $"dropout {variant.Dropout:0.##} -> " : "")}linear {classes}"
            + $" (about {DeviceMemory.Count(analysis.Parameters)} parameters)"));
        int undecodable = profile.Images.Count(i => !i.Info.Decodable);
        if (undecodable > 0)
        {
            design.Warnings.Add($"{undecodable} images cannot be decoded here ({string.Join(", ", profile.Images.Where(i => !i.Info.Decodable).Select(i => i.Info.Format).Distinct())}); --search and the trainer read PNG, BMP, PGM and PPM");
        }

        if (perClass.Min() < 20)
        {
            design.Warnings.Add($"a class has only {perClass.Min()} images: expect weak results for it; augmentation helps");
        }

        design.Why("task", "image classification", "a folder of class folders holding images is image classification (one class per folder)");
        design.Why("prep", $"{width}x{height}, {channels} channel{(channels == 1 ? "" : "s")}", "the median size, scaled so the longer side is at most 64 pixels (128 with --budget large); the most common channel count, alpha dropped");
        design.Why("network", $"{variant.Depth} convolution blocks", "one block per halving of the shorter side down to about 4 pixels, between 1 and 4");
        design.Why("network", $"{variant.Width} filters first", "16 under 5,000 images, else 32, doubling per block (up to 256); halved until the parameters fit the cap (100 per training image)");
        design.Why("prep", augment.Count > 0 ? "augmentation" : "no augmentation", "under 10,000 images, flips, small rotations and shifts make the set look larger");
        return design;
    }

    // ------------------------------------------------------------------ language data

    private static DesignPlan LanguageModel(DataProfile profile, DesignOptions options)
    {
        // Characters, most frequent first, at most 254 plus padding and unknown. Context: the next power of two at or
        // above the median conversation length, between 32 and 256 characters.
        var texts = profile.Conversations.Select(DataProfile.Render).ToList();
        var counts = texts.SelectMany(t => t).GroupBy(c => c).Select(g => (Symbol: g.Key.ToString(), Count: g.Count()))
            .OrderByDescending(p => p.Count).ThenBy(p => p.Symbol, StringComparer.Ordinal).Take(254).Select(p => p.Symbol).ToList();
        long characters = texts.Sum(t => (long)t.Length);
        int length = Math.Clamp(NextPowerOfTwo((int)ColumnProfile.Quantile([.. texts.Select(t => (double)t.Length)], 0.5)), 32, 256);
        int vocabulary = counts.Count + 2;

        // Width 64 and 2 layers; 128 and 4 layers from 5 million characters. Learning rate 1e-3 (3e-4 when larger).
        bool large = characters >= 5_000_000;
        var first = new Variant(large ? 128 : 64, large ? 4 : 2, 0.1f, large ? 3e-4f : 1e-3f);
        Func<Variant, NetworkBuilder> make = v => Architectures.Gpt(vocabulary, length, v.Width, Heads(v.Width), v.Depth, 4 * v.Width, v.Dropout).Named("suggested-language-model").Seed(1);
        long cap = Cap(options, characters, 1, 20_000, 50_000_000);
        var variant = FitToCap(first, make, cap, minWidth: 16);
        var design = new DesignPlan { Task = "chat", Make = make, Variant = variant, ParameterCap = cap };
        design.Prep = new JsonObject
        {
            ["format"] = "idrak-prep/1", ["kind"] = "language-model", ["source"] = profile.Path, ["render"] = "role: content lines",
            ["tokenizer"] = new JsonObject
            {
                ["type"] = "characters", ["length"] = length,
                ["vocabulary"] = new JsonArray([.. new[] { "<pad>", "<unk>" }.Concat(counts).Select(s => (JsonNode)s)]),
            },
            ["split"] = new JsonObject { ["validation"] = 0.1, ["seed"] = 1 },
        };
        var analysis = NetworkAnalysis.Of(make(variant).ToJson());
        int windows = (int)Math.Max(1, characters / length);
        Training(design, profile, "chat", windows * 0.9, windows, new JsonObject { ["type"] = "tokens" }, isImage: false);
        design.Lines.Add(("Data", $"{profile.Conversations.Count:N0} {(profile.Layout is "text" ? "texts" : "conversations")} ({profile.Layout ?? "rows"}) · {characters:N0} characters · {length} characters per window"));
        design.Lines.Add(("Task", "language model from scratch (next character), as no --base model was given"));
        design.Lines.Add(("Prep", $"render as 'role: content' lines · characters as tokens ({vocabulary} symbols) · windows of {length} · 90/10 split, seed 1"));
        design.Lines.Add(("Network", $"GPT: embedding {vocabulary} x {variant.Width} -> positions -> causal transformer x {variant.Depth} ({Heads(variant.Width)} heads) -> layer norm -> linear {vocabulary}"
            + $" (about {DeviceMemory.Count(analysis.Parameters)} parameters)"));
        design.Warnings.Add("no --base model: a small model trained from scratch learns the style of this data, not to answer; --base MODEL sets up a LoRA fine-tune instead");
        design.Why("task", "language model from scratch", "chat rows are a chat fine-tune of a base model; without --base, the rows train a small next-character model");
        design.Why("prep", "characters as tokens", "no tokenizer without a base model; characters need no vocabulary file and have no unknown words");
        design.Why("network", $"width {variant.Width}, {variant.Depth} layers", "64 wide and 2 layers, 128 and 4 from 5 million characters; halved until the parameters fit one per training character");
        return design;
    }

    private static DesignPlan BaseTune(DataProfile profile, DesignOptions options, string kind)
    {
        var model = options.Base!;
        var texts = profile.Conversations.Select(DataProfile.Render).ToList();
        bool counted = model.Tokenizer is not null;
        var tokens = texts.Select(t => counted ? (double)model.Tokenizer!.Encode(t).Count + 8 : Math.Ceiling(ColumnProfile.Words(t) * 1.3) + 8).ToList();
        double median = ColumnProfile.Quantile([.. tokens], 0.5), p99 = ColumnProfile.Quantile([.. tokens], 0.99), longest = tokens.Max();
        int rows = profile.Conversations.Count;

        // Rank 8 under 1,000 rows, 16 under 20,000, 32 above; alpha twice the rank. Epochs 3 under 1,000 rows, 2 under
        // 10,000, else 1. Length: the next power of two at or above the 99th percentile, 256 to the model's context (or
        // 4,096). Learning rate 2e-4 for supervised fine-tuning, 5e-5 for DPO.
        int rank = rows < 1_000 ? 8 : rows < 20_000 ? 16 : 32;
        int epochs = rows < 1_000 ? 3 : rows < 10_000 ? 2 : 1;
        int maxLength = Math.Clamp(NextPowerOfTwo((int)p99), 256, model.Context is { } c && c > 256 ? c : 4_096);
        int batchTokens = Math.Max(4_096, 2 * maxLength);
        bool dpo = kind == "preference";
        float lr = dpo ? 5e-5f : 2e-4f;
        var design = new DesignPlan { Task = kind };
        design.Lines.Add(("Data", $"{rows:N0} {(dpo ? "preference pairs" : profile.Layout is "text" ? "texts" : "conversations")} ({profile.Layout ?? "rows"}) · "
            + $"{median:N0} tokens median, {longest:N0} longest" + (counted ? "" : " (estimated from words)")));
        design.Lines.Add(("Task", dpo ? "preference training (DPO) of the base model" : "chat fine-tune (loss on the assistant's tokens only)"));

        // Memory: the base in bfloat16 (2 bytes per parameter) or 4 bits (about 0.6), plus a fifth for the adapters and
        // optimizer, plus the activations of one batch with checkpointing (batch tokens × width × layers × 4 bytes).
        long? memory = DeviceMemory.Total(options.Device);
        string precision = "bf16";
        if (model.Parameters is { } p)
        {
            double activations = batchTokens * (double)(model.Hidden ?? 1024) * (model.Layers ?? 24) * 4;
            double bf16 = p * 2 * 1.2 + activations, int4 = p * 0.6 * 1.2 + activations;
            if (memory is { } m)
            {
                precision = bf16 <= 0.8 * m ? "bf16" : "int4";
                string fits = bf16 <= 0.8 * m ? "bfloat16 base with LoRA fits; QLoRA not needed"
                    : int4 <= 0.8 * m ? "bfloat16 does not fit; 4-bit base (QLoRA)" : "even a 4-bit base may not fit: add --offload to the tune run";
                design.Lines.Add(("Memory", $"{options.Device} offers {DeviceMemory.Format(m)}; the setup needs about {DeviceMemory.Format(precision == "bf16" ? bf16 : int4)} -> {fits}"));
                if (int4 > 0.8 * m)
                {
                    design.Warnings.Add("the model may not fit this device's memory even at 4 bits");
                }
            }
            else
            {
                design.Lines.Add(("Memory", $"{options.Device}'s memory is not reported; a bfloat16 base needs about {DeviceMemory.Format(bf16)}, a 4-bit one {DeviceMemory.Format(int4)}"));
                design.Warnings.Add($"the memory of {options.Device} is unknown: check the bfloat16 / 4-bit choice against the card");
            }
        }
        else
        {
            design.Warnings.Add($"'{model.Name}' is not in the local caches: sized without its config and tokenizer (idrak pull {model.Name} first for exact numbers)");
        }

        string targets = "q,k,v,o,gate,up,down";
        design.Tune = new JsonObject
        {
            ["format"] = "idrak-tune/1", ["model"] = model.Name, ["data"] = profile.Path, ["loss"] = dpo ? "dpo" : "sft", ["adapter-type"] = "lora",
            ["rank"] = rank, ["alpha"] = 2 * rank, ["targets"] = targets, ["lr"] = lr, ["schedule"] = "cosine", ["warmup"] = 0.03, ["epochs"] = epochs,
            ["max-length"] = maxLength, ["batch-tokens"] = batchTokens, ["eval-fraction"] = rows >= 200 ? 0.05 : 0.0, ["seed"] = 1,
        };
        if (precision == "int4")
        {
            design.Tune["int4"] = true;
        }
        else
        {
            design.Tune["bf16"] = true;
        }

        if (dpo)
        {
            design.Tune["beta"] = 0.1;
        }

        design.Lines.Add(("Setup", $"LoRA rank {rank}, alpha {2 * rank} on {targets.Replace(",", "/", StringComparison.Ordinal)} · {(precision == "int4" ? "4-bit" : "bfloat16")} base · packing at {maxLength:N0} tokens"));
        design.Lines.Add(("Training", $"AdamW {lr:0e0} · cosine with 3% warm-up · {epochs} epoch{(epochs == 1 ? "" : "s")} · {batchTokens:N0} tokens per batch" + (dpo ? " · DPO beta 0.1" : "")));
        design.Why("setup", $"LoRA rank {rank}", "rank 8 under 1,000 rows, 16 under 20,000, 32 above; alpha twice the rank");
        design.Why("setup", $"{maxLength:N0} tokens", "the next power of two at or above the 99th percentile of tokens per row, between 256 and the model's context");
        design.Why("setup", precision == "int4" ? "4-bit base (QLoRA)" : "bfloat16 base", "bfloat16 when the base (2 bytes per parameter), adapters and one batch fit 80% of the device's memory, else 4 bits");
        design.Why("training", $"{epochs} epochs, learning rate {lr:0e0}", "3 epochs under 1,000 rows, 2 under 10,000, else 1; 2e-4 for fine-tuning, 5e-5 for DPO");
        return design;
    }

    /// <summary>Text classification with a base model: the label as the assistant's answer, among fixed choices.</summary>
    public static DesignPlan BaseClassify(DataProfile profile, DesignOptions options, DesignPlan network)
    {
        var prep = network.Prep!;
        var target = prep["target"]!.AsObject();
        var conversations = profile.Rows.Where(r => r[(string)target["column"]!] is not null)
            .Select(r => new JsonObject
            {
                ["messages"] = new JsonArray(
                    new JsonObject { ["role"] = "user", ["content"] = DataPreparation.Key(r[(string)prep["column"]!]) ?? "" },
                    new JsonObject { ["role"] = "assistant", ["content"] = DataPreparation.Key(r[(string)target["column"]!]) ?? "" }),
            }).ToList();
        var chat = DataProfileForChat(conversations);
        var design = BaseTune(chat, options, "chat");
        var result = new DesignPlan { Task = network.Task };
        result.Lines.Add(network.Lines.First(l => l.Label == "Data"));
        result.Lines.Add(("Task", network.Lines.First(l => l.Label == "Task").Text + $", as answers of {options.Base!.Name}"));
        result.Lines.AddRange(design.Lines.Where(l => l.Label is not ("Data" or "Task")));
        result.Warnings.AddRange(network.Warnings.Concat(design.Warnings));
        result.Reasons.AddRange(network.Reasons.Where(r => r.Area == "task").Concat(design.Reasons));
        result.Tune = design.Tune;
        result.Tune!["data"] = profile.Path;
        if ((string?)target["type"] == "classes")
        {
            result.Tune!["choices"] = string.Join(",", target["classes"]!.AsArray().Select(c => (string)c!));
        }

        result.Prep = new JsonObject
        {
            ["format"] = "idrak-prep/1", ["kind"] = "chat-mapping", ["source"] = profile.Path,
            ["user"] = "{" + (string)prep["column"]! + "}", ["assistant"] = "{" + (string)target["column"]! + "}",
        };
        result.Why("task", "the label as the base model's answer", "--base with a text column fine-tunes the model to answer each text with its label, among the label values");
        return result;
    }

    // The label rows as conversations, profiled as chat data (through a temporary JSON Lines file).
    private static DataProfile DataProfileForChat(List<JsonObject> conversations)
    {
        var path = Path.Combine(Path.GetTempPath(), $"idrak-suggest-{Environment.ProcessId}-{Guid.NewGuid():N}.jsonl");
        try
        {
            File.WriteAllLines(path, conversations.Select(c => c.ToJsonString()));
            return DataProfile.Read(path, null, "chat");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------------ shared rules

    // Optimizer, schedule, batch, epochs, loss and metric for a network.
    private static void Training(DesignPlan design, DataProfile profile, string task, double train, int rows, JsonObject targetSpec, bool isImage)
    {
        var v = design.Variant!;

        // Batch: the power of two at or below a 32nd of the training rows, between 8 and 256 (128 for images, 32 for
        // language windows). Epochs: enough for about 3,000 steps, between 20 and 300 (images 15 to 100, language
        // models 5 to 50); early stopping after a tenth of them without improvement, 5 to 20.
        int maxBatch = isImage ? 128 : task == "chat" ? 32 : 256;
        int batch = Math.Clamp(PreviousPowerOfTwo((int)Math.Max(1, train / 32)), 8, maxBatch);
        double steps = Math.Max(1, train / batch);
        var (minEpochs, maxEpochs) = isImage ? (15, 100) : task == "chat" ? (5, 50) : (20, 300);
        int epochs = Math.Clamp((int)Math.Ceiling(3_000 / steps), minEpochs, maxEpochs);
        int patience = Math.Clamp(epochs / 10, 5, 20);
        float decay = rows < 1_000 ? 1e-3f : 1e-4f;
        bool classes = (string?)targetSpec["type"] == "classes";
        string loss = task == "chat" ? "token-cross-entropy" : classes ? "cross-entropy" : "mse";
        string metric = task == "chat" ? "loss" : classes ? "accuracy" : "rmse";
        int warmup = isImage || task == "chat" ? Math.Min(2, epochs / 10) : 0;
        design.Train = new JsonObject
        {
            ["format"] = "idrak-train/1", ["task"] = task, ["target"] = (string?)targetSpec["column"], ["optimizer"] = "adamw",
            ["learningRate"] = v.LearningRate, ["weightDecay"] = decay, ["batchSize"] = batch, ["epochs"] = epochs, ["earlyStopping"] = patience,
            ["schedule"] = "cosine", ["warmupEpochs"] = warmup, ["loss"] = loss, ["metric"] = metric, ["validationFraction"] = (double?)design.Prep?["split"]?["validation"] ?? 0.2,
            ["seed"] = 1,
        };
        design.Lines.Add(("Training", $"AdamW {v.LearningRate:0e0}, weight decay {decay:0e0} · {(warmup > 0 ? $"cosine with {warmup} warm-up epochs" : "cosine")} · batch {batch} · up to {epochs} epochs, early stop after {patience} · {LossName(loss)}"));
        design.Why("training", $"batch {batch}", $"the power of two at or below a 32nd of the training rows, between 8 and {maxBatch}");
        design.Why("training", $"up to {epochs} epochs", $"about 3,000 steps, between {minEpochs} and {maxEpochs}; early stopping after a tenth of them (5 to 20) keeps the best weights");
        design.Why("training", $"weight decay {decay:0e0}", "1e-3 under 1,000 rows (stronger regularization), else 1e-4");
        design.Why("training", LossName(loss), task == "chat" ? "next-token prediction" : classes ? "classes: softmax cross-entropy over one output per class" : "a number: mean squared error on the scaled target");
        _ = profile;
    }

    private static string LossName(string loss) => loss switch
    {
        "mse" => "mean squared error",
        "cross-entropy" => "cross-entropy",
        _ => "next-token cross-entropy",
    };

    // Unbalanced classes (the largest class over 3 times the smallest): oversample the rare ones in training.
    private static void Balance(DesignPlan design, DataProfile profile, ColumnProfile target, JsonObject targetSpec)
    {
        if ((string?)targetSpec["type"] != "classes" || design.Prep is null)
        {
            return;
        }

        var classes = targetSpec["classes"]!.AsArray();
        var counts = classes.Select(c => profile.Rows.Count(r => DataPreparation.Key(r[target.Name]) == (string?)c)).ToArray();
        if (counts.Min() * 3 < counts.Max())
        {
            design.Prep["balance"] = "oversample";
            design.Warnings.Add($"unbalanced classes ({counts.Min()} to {counts.Max()} rows per class): rare classes are oversampled in training");
            design.Why("prep", "oversample rare classes", "the largest class has over 3 times the rows of the smallest");
        }
    }

    // Warns when training the network at its batch would not fit the device's memory.
    private static void CheckMemory(DesignPlan design, NetworkAnalysis analysis, Device device, int batch)
    {
        long need = analysis.TrainingBytes(batch);
        if (DeviceMemory.Total(device) is { } memory && need > memory * 0.8)
        {
            design.Warnings.Add($"training at batch {batch} needs about {DeviceMemory.Format(need)}, more than 80% of {device}'s {DeviceMemory.Format(memory)}: lower the batch or --budget");
        }
    }

    // Halves the width (not below minWidth), then lowers the depth, until the network has at most `cap` parameters.
    private static Variant FitToCap(Variant v, Func<Variant, NetworkBuilder> make, long cap, int minWidth)
    {
        long Count(Variant x) => NetworkAnalysis.Of(make(x).ToJson()).Parameters;
        while (Count(v) > cap && v.Width / 2 >= minWidth)
        {
            v = v with { Width = v.Width / 2 };
        }

        while (Count(v) > cap && v.Depth > 1)
        {
            v = v with { Depth = v.Depth - 1 };
        }

        return v;
    }

    /// <summary>The MLP: Input(features) → depth × [Linear(width / 2^i, at least 8), ReLU, Dropout] → Linear(outputs).</summary>
    public static NetworkBuilder Mlp(int features, int outputs, Variant v)
    {
        var b = Network.Input(features);
        for (int i = 0; i < v.Depth; i++)
        {
            b.Linear(Math.Max(8, v.Width >> i)).ReLU();
            if (v.Dropout > 0)
            {
                b.Dropout(v.Dropout);
            }
        }

        return b.Linear(outputs).Named("suggested-mlp").Seed(1);
    }

    /// <summary>The CNN: depth × [Conv2d 3×3 (width, doubling, at most 256), BatchNorm, ReLU, MaxPool2d(2)] → global average pool → Dropout → Linear.</summary>
    public static NetworkBuilder Cnn(int channels, int height, int width, int classes, Variant v)
    {
        var b = Network.Image(channels, height, width);
        int filters = v.Width;
        for (int i = 0; i < v.Depth; i++)
        {
            b.Conv2d(filters, 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2);
            filters = Math.Min(256, filters * 2);
        }

        b.GlobalAveragePool2d();
        if (v.Dropout > 0)
        {
            b.Dropout(v.Dropout);
        }

        return b.Linear(classes).Named("suggested-cnn").Seed(1);
    }

    /// <summary>The text model: depth 0 is the mean of word embeddings with one hidden layer; otherwise transformer layers.</summary>
    public static NetworkBuilder TextNetwork(int vocabulary, int length, int outputs, Variant v)
    {
        var b = Network.Tokens(length).Embedding(vocabulary, v.Width);
        if (v.Depth == 0)
        {
            b.MeanOverTime().Linear(v.Width).ReLU();
            if (v.Dropout > 0)
            {
                b.Dropout(v.Dropout);
            }
        }
        else
        {
            b.PositionalEncoding().Repeat(v.Depth, x => x.TransformerEncoderLayer(Heads(v.Width), 2 * v.Width, v.Dropout)).LayerNorm().MeanOverTime();
        }

        return b.Linear(outputs).Named("suggested-text").Seed(1);
    }

    // Attention heads: one per 32 of width, 1 to 8 (the width is a power of two, so they divide it).
    private static int Heads(int width) => Math.Clamp(width / 32, 1, 8);

    private static int NextPowerOfTwo(int value) => value <= 1 ? 1 : (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)value);

    private static int PreviousPowerOfTwo(int value) => value <= 1 ? 1 : 1 << System.Numerics.BitOperations.Log2((uint)value);

    private static double Round(double value) => Math.Round(value, 6);

    private static DesignPlan Copy(DesignPlan design, Func<Variant, NetworkBuilder> make)
    {
        var copy = new DesignPlan { Task = design.Task, Make = make, Prep = design.Prep };
        copy.Warnings.AddRange(design.Warnings);
        return copy;
    }
}
