// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Nlp;

namespace Idrak.Cli.Commands.Design;

/// <summary>
/// <c>idrak suggest DATA</c> (<c>sg</c>): reads the data, picks the task and writes a network and its training setup
/// (plans/idrak-cli.md, "Design a model"). The rules live in <see cref="DesignRules"/>, the optional measuring in
/// <see cref="DesignSearch"/>.
/// </summary>
internal sealed class SuggestCommand : Command
{
    private static readonly string[] Tasks = ["regression", "classify", "image", "sequence", "chat", "preference"];

    public override string Name => "suggest";

    public override IReadOnlyCollection<string> Aliases => ["sg"];

    public override string Summary => "Design a network and its training setup for a data set";

    public override string Usage =>
        "DATA [options]\n\n" +
        "DATA is a CSV, TSV, JSON Lines, JSON or Parquet file (or a folder of them), a folder of class folders of images,\n" +
        "chat rows (conversations or text) or preference rows (chosen and rejected answers). Writes network.json (the\n" +
        "network builder's JSON: Network.FromJson and idrak train read it), train.json and prep.json; with --base, tune.json.\n\n" +
        "Options:\n" +
        "  -t, --target COL        the column to predict (default: a column named like a target, else the last one)\n" +
        "      --text COL          a text column: designs a text model (words as tokens)\n" +
        "      --task NAME         regression, classify, image, sequence, chat or preference (default: from the data)\n" +
        "      --budget SIZE       small, medium (default) or large: the parameter cap ÷4, ×1 or ×4\n" +
        "      --max-params N      the parameter cap itself (e.g. 50000, 50k, 2M)\n" +
        "  -n, --search N          train N candidates briefly on the device and keep the best on held-out data\n" +
        "  -e, --explain           print the rule behind each choice\n" +
        "  -b, --base MODEL        a base language model: a LoRA setup (tune.json) sized to the device for chat, preference\n" +
        "                          and text data; looked up in the local caches only (idrak pull fetches it)\n" +
        "  -o, --out DIR           where the files go (default: the current folder)\n" +
        "      --assist MODEL      a local chat model explains the choices in plain words (it never changes them)\n" +
        "  -w, --weights FORMAT, -k, --kv FORMAT, --context N: how --assist loads its model\n\n" +
        "Examples:\n" +
        "  idrak sg houses.csv -t SalePrice -e            design a network for a CSV, with the reasons\n" +
        "  idrak sg ./shapes -n 6 -d vulkan:0             try 6 CNN variants on the GPU, keep the best\n" +
        "  idrak sg reviews.jsonl -t label --text review  a text classifier\n" +
        "  idrak sg chats.jsonl -b owner/model -o ./run    LoRA setup sized to the device\n\n" +
        "Limits: 12-bit and arithmetic-coded JPEG files are not decoded;\n" +
        "a GPU's memory is known only through a configured limit (the library does not report it).";

    public override IReadOnlyCollection<string> ValueOptions =>
        ["--target", "--text", "--task", "--budget", "--max-params", "--search", "--base", "--out", "--assist", .. ModelChoices.ValueOptions.Where(o => o != "--adapter")];

    public override IReadOnlyCollection<string> Flags => ["--explain"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-t"] = "--target", ["-n"] = "--search", ["-e"] = "--explain", ["-b"] = "--base", ["-o"] = "--out", ["-w"] = "--weights", ["-k"] = "--kv",
    };

    public override int Run(CommandContext context)
    {
        string data = context.Argument(0, "DATA (a data file or folder)");
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"One DATA argument, not {context.Positional.Count}.");
        }

        string? task = context.Option("--task")?.ToLowerInvariant();
        if (task is not null && !Tasks.Contains(task))
        {
            throw new UsageException($"--task {task}: use {string.Join(", ", Tasks)}.");
        }

        string? budget = context.Option("--budget")?.ToLowerInvariant();
        if (budget is not null and not ("small" or "medium" or "large"))
        {
            throw new UsageException($"--budget {budget}: use small, medium or large.");
        }

        long? maxParams = context.Option("--max-params") is { } mp ? ParseCount(mp) : null;
        int search = context.IntOption("--search", 1);
        if (search < 1)
        {
            throw new UsageException("--search needs 1 or more candidates.");
        }

        string outDir = Path.GetFullPath(context.Option("--out") ?? ".");
        var baseModel = context.Option("--base") is { } b ? BaseModelInfo.Inspect(context, ModelChoices.Choose(context, b).Model) : null;
        var profile = DataProfile.Read(data, context.Option("--target"), task);
        if (task == "image" && profile.Kind != DataKind.Images)
        {
            throw new UsageException($"--task image needs a folder of class folders of images; {data} is not one.");
        }

        var options = new DesignOptions(context.Option("--target"), context.Option("--text"), task, budget, maxParams, context.Device, baseModel, context.Seed ?? 1);
        var design = DesignRules.Propose(profile, options);
        if (baseModel is not null && design.Make is not null)
        {
            if (design.Task == "sequence")
            {
                design = DesignRules.BaseClassify(profile, options, design);
            }
            else if (design.Task != "chat")
            {
                design.Warnings.Add("--base applies to text, chat and preference data; this design does not use it");
            }
        }

        // --search: measure the candidates on the device and keep the best.
        JsonArray? searchJson = null;
        if (search > 1 && design.Make is not null)
        {
            searchJson = Search(context, design, profile, search);
        }
        else if (search > 1)
        {
            design.Warnings.Add("--search measures networks; a base-model setup has none to compare");
        }

        // Write the files.
        Directory.CreateDirectory(outDir);
        var written = new List<string>();
        void Save(string name, JsonNode? node)
        {
            if (node is not null)
            {
                File.WriteAllText(Path.Combine(outDir, name), node.ToJsonString(CommandContext.JsonOutput) + "\n");
                written.Add(name);
            }
        }

        var network = design.Variant is { } chosen ? design.Network(chosen) : null;
        Save("network.json", network);
        Save("train.json", design.Train);
        Save("prep.json", design.Prep);
        Save("tune.json", design.Tune);
        string shown = Path.GetRelativePath(Environment.CurrentDirectory, outDir) is var relative && !relative.StartsWith("..", StringComparison.Ordinal) ? relative : outDir;
        string Rel(string file) => shown == "." ? file : Path.Combine(shown, file);
        string next = design.Tune is not null
            ? $"idrak tune --config {Rel("tune.json")}"
            : $"idrak train {Rel("network.json")} --data {data} --config {Rel("train.json")}";
        design.Lines.Add(("Wrote", string.Join(" · ", written) + (shown == "." ? "" : $" in {shown}")));
        design.Lines.Add(("Next", next));

        // Lines and reasons in reading order: the data, the task, then preparation, network, search and training.
        string[] order = ["Data", "Task", "Memory", "Prep", "Network", "Search", "Setup", "Training", "Wrote", "Next"];
        string[] areas = ["task", "prep", "network", "search", "setup", "training"];
        var lines = design.Lines.OrderBy(l => Array.IndexOf(order, l.Label) is var i and >= 0 ? i : order.Length).ToList();
        var reasons = design.Reasons.OrderBy(r => Array.IndexOf(areas, r.Area) is var i and >= 0 ? i : areas.Length).ToList();
        design.Lines.Clear();
        design.Lines.AddRange(lines);
        design.Reasons.Clear();
        design.Reasons.AddRange(reasons);
        foreach (var (label, text) in design.Lines)
        {
            context.Write($"{label,-9} {text}");
        }

        foreach (string warning in design.Warnings)
        {
            context.Write($"{"Warning",-9} {warning}");
        }

        string? assist = context.Option("--assist") is { } assistModel ? Assist(context, assistModel, design) : null;
        if (context.Flag("--explain"))
        {
            context.Write("");
            context.Write("Why:");
            foreach (var r in design.Reasons)
            {
                context.Write($"  {r.Area,-9} {r.Choice}: {r.Rule}");
            }
        }

        if (assist is not null)
        {
            context.Write("");
            context.Write($"{"Assist",-9} {assist}");
        }

        context.WriteJson(new JsonObject
        {
            ["data"] = DataJson(profile),
            ["task"] = design.Task,
            ["lines"] = new JsonObject(design.Lines.Select(l => KeyValuePair.Create(l.Label.ToLowerInvariant(), (JsonNode?)l.Text))),
            ["variant"] = design.Variant is { } v ? VariantJson(v) : null,
            ["parameters"] = network is null ? null : NetworkAnalysis.Of(network).Parameters,
            ["parameterCap"] = design.Make is null ? null : design.ParameterCap,
            ["search"] = searchJson,
            ["out"] = outDir,
            ["files"] = new JsonArray([.. written.Select(w => (JsonNode)w)]),
            ["warnings"] = new JsonArray([.. design.Warnings.Select(w => (JsonNode)w)]),
            ["reasons"] = new JsonArray([.. design.Reasons.Select(r => (JsonNode)new JsonObject { ["area"] = r.Area, ["choice"] = r.Choice, ["rule"] = r.Rule })]),
            ["assist"] = assist,
            ["next"] = next,
        });
        return ExitCodes.Ok;
    }

    /// <summary>A count with an optional k, M or G suffix ("50k" is 50,000).</summary>
    internal static long ParseCount(string text)
    {
        string t = text.Trim().Replace("_", "", StringComparison.Ordinal).Replace(",", "", StringComparison.Ordinal);
        double factor = t.Length > 0 ? char.ToLowerInvariant(t[^1]) switch { 'k' => 1e3, 'm' => 1e6, 'g' => 1e9, _ => 1 } : 1;
        if (factor > 1)
        {
            t = t[..^1];
        }

        return double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && value * factor >= 1
            ? (long)(value * factor)
            : throw new UsageException($"--max-params needs a positive count such as 50000, 50k or 2M, not '{text}'.");
    }

    private static JsonArray Search(CommandContext context, DesignPlan design, DataProfile profile, int count)
    {
        var prepared = DataPreparation.Apply(design.Prep!, profile);
        var candidates = DesignRules.Candidates(design, count);
        int epochs = DesignSearch.Epochs(design, prepared);
        context.Detail($"search: {candidates.Count} candidates x {epochs} epochs on {context.Device} ({prepared.Train.Count} training, {prepared.Validation.Count} held-out rows)");
        var results = DesignSearch.Run(design, prepared, candidates, context.Device,
            r => context.Detail($"  {r.Index}. {r.Variant}: " + (r.Error ?? string.Create(CultureInfo.InvariantCulture, $"{r.Metric} {Score(r)} in {r.Seconds:F1} s"))));
        var best = DesignSearch.Best(results);
        var first = results[0];
        double seconds = results.Sum(r => r.Seconds);
        string widths = string.Join("/", candidates.Select(c => c.Width).Distinct());
        string depths = string.Join("/", candidates.Select(c => c.Depth).Distinct());
        string dropouts = string.Join("/", candidates.Select(c => c.Dropout.ToString("0.##", CultureInfo.InvariantCulture)).Distinct());
        var line = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{candidates.Count} candidates x {epochs} epochs on {context.Device} ({seconds:F0} s): width {widths} · depth {depths} · dropout {dropouts}"));
        if (best is not null)
        {
            line.Append($"\n{"",-9} best: {best.Variant} -> validation {best.Metric} {Score(best)}");
            if (best.Index != 1 && first.Error is null)
            {
                line.Append($" (first guess {Score(first)})");
            }

            design.Variant = best.Variant;
            design.Train!["learningRate"] = best.Variant.LearningRate;
        }
        else
        {
            design.Warnings.Add("no candidate trained: " + string.Join("; ", results.Select(r => r.Error).Distinct()));
        }

        int at = design.Lines.FindIndex(l => l.Label == "Training");
        design.Lines.Insert(at < 0 ? design.Lines.Count : at, ("Search", line.ToString()));
        if (best is not null && best.Index != 1)
        {
            var analysis = NetworkAnalysis.Of(design.Network(best.Variant)!);
            int network = design.Lines.FindIndex(l => l.Label == "Network");
            design.Lines[network] = ("Network", design.Lines[network].Text + $"; searched: {best.Variant} ({Units.Short(analysis.Parameters)} parameters)");
        }

        design.Why("search", best is null ? "first guess kept" : $"candidate {best.Index}", DesignSearch.HigherIsBetter(first.Metric)
            ? "the highest validation accuracy wins; ties go to the lower validation loss, then the earlier candidate"
            : "the lowest validation error wins; ties go to the earlier candidate");
        return new JsonArray([.. results.Select(r => (JsonNode)new JsonObject
        {
            ["index"] = r.Index, ["variant"] = VariantJson(r.Variant), ["parameters"] = r.Parameters, ["metric"] = r.Metric,
            ["score"] = double.IsFinite(r.Score) ? r.Score : null, ["validationLoss"] = double.IsFinite(r.ValidationLoss) ? r.ValidationLoss : null,
            ["seconds"] = Math.Round(r.Seconds, 3), ["error"] = r.Error, ["best"] = best?.Index == r.Index,
        })]);
    }

    private static string Score(SearchResult r) => r.Metric == "accuracy" ? (r.Score * 100).ToString("0.0", CultureInfo.InvariantCulture) + "%" : r.Score.ToString("0.####", CultureInfo.InvariantCulture);

    // A local chat model puts the choices into plain words; its answer is printed, the files stay as the rules made them.
    private static string Assist(CommandContext context, string name, DesignPlan design)
    {
        var choice = ModelChoices.Choose(context, name);
        using var model = ModelChoices.Load(context, choice);
        var chat = model.CreateChat(contextLength: choice.Context ?? 4096);
        var facts = new StringBuilder();
        foreach (var (label, text) in design.Lines.Where(l => l.Label is not ("Wrote" or "Next")))
        {
            facts.Append(label).Append(": ").Append(text).Append('\n');
        }

        foreach (var r in design.Reasons)
        {
            facts.Append("Rule (").Append(r.Area).Append(") ").Append(r.Choice).Append(": ").Append(r.Rule).Append('\n');
        }

        var request = new ChatRequest(
        [
            new ChatMessage("system", "You explain machine-learning setups to a developer in plain words. Explain only the choices given; do not change them."),
            new ChatMessage("user", "Explain this setup and why each choice fits the data, in a short paragraph:\n" + facts),
        ], Think: false, Options: new GenerationOptions { Temperature = 0.3f, NumPredict = 300, Seed = context.Seed ?? 1 });
        return chat.Chat(request).Message?.Content.Trim() ?? "";
    }

    private static JsonObject VariantJson(Variant v) => new()
    {
        ["width"] = v.Width, ["depth"] = v.Depth, ["dropout"] = v.Dropout, ["learningRate"] = v.LearningRate,
    };

    private static JsonObject DataJson(DataProfile p)
    {
        var json = new JsonObject
        {
            ["path"] = p.Path, ["kind"] = p.Kind.ToString().ToLowerInvariant(), ["format"] = p.Format, ["rows"] = p.TotalRows,
            ["profiledRows"] = p.Kind == DataKind.Images ? p.Images.Count : p.Rows.Count, ["duplicates"] = p.Duplicates,
        };
        if (p.Kind == DataKind.Table)
        {
            json["missing"] = Math.Round(p.MissingShare, 6);
            json["columns"] = new JsonArray([.. p.Columns.Select(c => (JsonNode)new JsonObject
            {
                ["name"] = c.Name, ["type"] = c.Type, ["missing"] = c.Missing, ["distinct"] = c.ManyDistinct ? null : c.Distinct.Count,
            })]);
        }
        else if (p.Kind == DataKind.Images)
        {
            json["classes"] = new JsonArray([.. p.ClassNames.Select((n, i) => (JsonNode)new JsonObject { ["name"] = n, ["images"] = p.Images.Count(x => x.Class == i) })]);
        }
        else
        {
            json["layout"] = p.Layout;
            json["conversations"] = p.Conversations.Count;
        }

        return json;
    }
}
