// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp;

namespace Idrak.Cli.Commands.Data;

/// <summary><c>idrak data preview FILE</c>: the first rows, the columns and their types.</summary>
internal sealed class DataPreviewCommand : Command
{
    public override string Name => "data preview";

    public override string Summary => "First rows, columns and their types of a data file (Parquet, JSON Lines, CSV)";

    public override string Usage => """
        FILE [-n ROWS]

        Options:
          -n, --rows N  rows to show (default 5); the column types come from the first 1,000 rows

        Examples:
          idrak data preview houses.csv
          idrak data preview train.parquet -n 10 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--rows"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-n"] = "--rows" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE (a CSV, JSON Lines, JSON or Parquet file)");
        int show = context.IntOption("--rows", 5);
        var data = RowFiles.Read(path);
        var sample = new List<JsonObject>();
        long count = 0;
        foreach (var row in data)
        {
            if (count++ < 1000)
            {
                sample.Add(row);
            }
        }

        var columns = RowFiles.Columns(sample);
        context.Write($"{path}: {RowFiles.FormatName(path)}, {count:N0} rows, {columns.Count} columns");
        context.Table(["Column", "Type", "Missing"], columns.Select(c => (IReadOnlyList<string>)[c.Name, c.Type, c.Missing == 0 ? "" : $"{c.Missing:N0}"]));
        if (show > 0 && sample.Count > 0)
        {
            context.Write("");
            context.Table([.. columns.Select(c => c.Name)], sample.Take(show).Select(r => (IReadOnlyList<string>)[.. columns.Select(c => RowFiles.Cell(r[c.Name]))]));
        }

        context.WriteJson(new JsonObject
        {
            ["file"] = path,
            ["format"] = RowFiles.FormatName(path),
            ["rows"] = count,
            ["columns"] = new JsonArray([.. columns.Select(c => (JsonNode)new JsonObject { ["name"] = c.Name, ["type"] = c.Type, ["missing"] = c.Missing })]),
            ["preview"] = new JsonArray([.. sample.Take(show).Select(r => (JsonNode)r.DeepClone())]),
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak data validate FILE --as chat|preference|table</c>: rows checked against the shape a command reads.</summary>
internal sealed class DataValidateCommand : Command
{
    public override string Name => "data validate";

    public override string Summary => "Check rows against the shape a command expects (chat, preference or table), with the first bad rows";

    public override string Usage => """
        FILE --as chat|preference|table [-t COL] [--show N]

        Options:
              --as KIND   chat: conversations with an assistant turn (as idrak tune reads them, any common layout);
                          preference: prompt, chosen and rejected (dpo, orpo, simpo); table: the same columns in every row,
                          scalar values, numbers where the column holds numbers (as idrak train reads a CSV)
          -t, --target C  table: this column must be present and filled
              --show N    bad rows to show (default 5)

        Exit code 1 when a row is bad.

        Examples:
          idrak data validate chats.jsonl --as chat
          idrak data validate houses.csv --as table -t price
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--as", "--target", "--show"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-t"] = "--target" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE");
        string kind = context.Option("--as") ?? throw new UsageException("Give --as chat, --as preference or --as table.");
        if (kind is not ("chat" or "preference" or "table"))
        {
            throw new UsageException($"--as {kind}: use chat, preference or table.");
        }

        string? target = context.Option("--target");
        int show = context.IntOption("--show", 5);
        var rows = RowFiles.Read(path).ToList();
        var columns = RowFiles.Columns(rows.Take(1000).ToList());
        // A column holds numbers when most of its filled values are numbers; the others are then bad rows.
        var numeric = columns.Select(c => c.Name).Where(name =>
        {
            var filled = rows.Take(1000).Select(r => r[name]).Where(v => v is not null).ToList();
            return filled.Count > 0 && filled.Count(v => RowFiles.Number(v) is not null) * 2 > filled.Count;
        }).ToHashSet(StringComparer.Ordinal);
        var expected = rows.Count > 0 ? rows[0].Select(p => p.Key).ToList() : [];
        var expectedSet = expected.ToHashSet(StringComparer.Ordinal);
        if (kind == "table" && target is not null && rows.Count > 0 && !columns.Any(c => c.Name == target))
        {
            context.Error($"{path} has no column '{target}' (columns: {string.Join(", ", columns.Select(c => c.Name))}).");
            return ExitCodes.Failed;
        }

        var bad = new List<(long Line, string Reason, JsonObject Row)>();
        long badCount = 0;
        for (int i = 0; i < rows.Count; i++)
        {
            string? reason = kind switch
            {
                "chat" => Chat(rows[i]),
                "preference" => Preference(rows[i]),
                _ => Table(rows[i], expected, expectedSet, numeric, target),
            };
            if (reason is not null)
            {
                badCount++;
                if (bad.Count < show)
                {
                    bad.Add((i + 1, reason, rows[i]));
                }
            }
        }

        context.Write($"{path}: {rows.Count:N0} rows as {kind}: {rows.Count - badCount:N0} valid, {badCount:N0} invalid");
        foreach (var (line, reason, row) in bad)
        {
            context.Write($"  row {line}: {reason}");
            context.Write($"    {DataTool.Short(row)}");
        }

        if (badCount > bad.Count)
        {
            context.Write($"  ... and {badCount - bad.Count:N0} more (--show N for more)");
        }

        context.WriteJson(new JsonObject
        {
            ["file"] = path,
            ["as"] = kind,
            ["rows"] = rows.Count,
            ["valid"] = rows.Count - badCount,
            ["invalid"] = badCount,
            ["bad"] = new JsonArray([.. bad.Select(b => (JsonNode)new JsonObject { ["row"] = b.Line, ["reason"] = b.Reason, ["data"] = b.Row.DeepClone() })]),
        });
        return badCount == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    private static readonly string[] Roles = ["system", "user", "assistant", "tool"];

    private static string? Chat(JsonObject row)
    {
        if (ChatRows.Normalize((JsonObject)row.DeepClone(), RowKind.Chat) is not { } chat || chat["messages"] is not JsonArray messages)
        {
            return "not a conversation (expected messages, or a known layout such as question/answer, instruction/output, ShareGPT turns)";
        }

        foreach (var message in messages)
        {
            string? role = (string?)message?["role"];
            if (role is null || !Roles.Contains(role))
            {
                return $"a message has role '{role}' (use system, user, assistant or tool)";
            }

            if (message?["content"] is not JsonValue && message?["tool_calls"] is null)
            {
                return $"a {role} message has no text content";
            }
        }

        return messages.Any(m => (string?)m?["role"] == "assistant") ? null : "no assistant turn to train on";
    }

    private static string? Preference(JsonObject row)
    {
        if (ChatRows.Preference((JsonObject)row.DeepClone()) is not { } pair)
        {
            return "not a preference pair (expected prompt, chosen and rejected)";
        }

        return JsonNode.DeepEquals(pair["chosen"], pair["rejected"]) ? "chosen and rejected are the same" : null;
    }

    private static string? Table(JsonObject row, List<string> expected, HashSet<string> expectedSet, HashSet<string> numeric, string? target)
    {
        if (expected.FirstOrDefault(c => !row.ContainsKey(c)) is { } missing)
        {
            return $"column '{missing}' is missing";
        }

        if (row.Select(p => p.Key).FirstOrDefault(c => !expectedSet.Contains(c)) is { } extra)
        {
            return $"column '{extra}' is not in the first row";
        }

        if (row.FirstOrDefault(p => p.Value is JsonObject or JsonArray) is { Key: { } nested })
        {
            return $"'{nested}' holds a list or an object, not a value";
        }

        if (target is not null && row[target] is null)
        {
            return $"the target '{target}' is empty";
        }

        foreach (var (key, value) in row)
        {
            if (numeric.Contains(key) && value is not null && RowFiles.Number(value) is null)
            {
                return $"'{key}' is {RowFiles.Cell(value)}, not a number";
            }
        }

        return null;
    }
}

/// <summary><c>idrak data stats FILE [-m MODEL]</c>: lengths in characters, words and, with a model, tokens.</summary>
internal sealed class DataStatsCommand : Command
{
    public override string Name => "data stats";

    public override string Summary => "Row lengths in characters, words and tokens, a length histogram, and the rows over a context length";

    public override string Usage => """
        FILE [-m MODEL] [--context N] [--column COL]

        Options:
          -m, --model MODEL  count tokens with this model's tokenizer (a Hugging Face id, a folder, a .gguf file or an
                             alias); conversations are rendered with its chat template first
              --context N    count the rows longer than N tokens (default: the model's context length when known)
              --column COL   the text column (default: conversations as rendered, else every text value of a row)

        Examples:
          idrak data stats chats.jsonl -m qwen
          idrak data stats reviews.csv --column review
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--model", "--context", "--column"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-m"] = "--model" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE");
        string? column = context.Option("--column");
        int? contextLength = context.Option("--context") is null ? null : context.IntOption("--context", 0);
        var rows = RowFiles.Read(path).ToList();
        ITokenizer? tokenizer = null;
        JinjaChatTemplate? template = null;
        PretrainedModel? loaded = null;
        try
        {
            if (context.Option("--model") is { } name)
            {
                var choice = ModelChoices.Choose(context, name);
                string resolved = ModelChoices.Resolve(context, choice.Model);
                if (Directory.Exists(resolved) && File.Exists(Path.Combine(resolved, "tokenizer.json")))
                {
                    var bpe = BpeTokenizer.Load(resolved);
                    (tokenizer, template) = (bpe, JinjaChatTemplate.Load(resolved, bpe));
                    contextLength ??= JsonNode.Parse(File.Exists(Path.Combine(resolved, "config.json")) ? File.ReadAllText(Path.Combine(resolved, "config.json")) : "{}")?["max_position_embeddings"]?.GetValue<int>();
                }
                else
                {
                    loaded = ModelChoices.Load(context, choice);
                    (tokenizer, template) = (loaded.Tokenizer, loaded.JinjaTemplate);
                    contextLength ??= loaded.MaxPositions;
                }

                if (tokenizer is null)
                {
                    context.Error($"{name} has no tokenizer to count tokens with.");
                    return ExitCodes.Failed;
                }
            }

            var characters = new List<int>();
            var words = new List<int>();
            var tokens = new List<int>();
            int conversations = 0;
            foreach (var row in rows)
            {
                string text;
                if (column is not null)
                {
                    text = RowFiles.Raw(row[column]);
                }
                else if (ChatRows.Normalize((JsonObject)row.DeepClone(), RowKind.Chat) is { } chat)
                {
                    conversations++;
                    var transcript = ChatTranscript.FromJson(chat);
                    text = template is not null ? template.Render(transcript.Messages, transcript.Tools, null, addGenerationPrompt: false)
                        : string.Join("\n", transcript.Messages.Select(m => m.Content));
                }
                else
                {
                    text = string.Join("\n", row.Select(p => p.Value).OfType<JsonValue>().Where(v => v.TryGetValue(out string? _)).Select(v => v.GetValue<string>()));
                }

                characters.Add(text.Length);
                words.Add(Words(text));
                if (tokenizer is not null)
                {
                    tokens.Add(tokenizer.Encode(text).Count);
                }
            }

            // Sorted once, in place: the table, the JSON and the histogram read them in any order.
            characters.Sort();
            words.Sort();
            tokens.Sort();
            var lengths = tokens.Count > 0 ? tokens : characters;
            string unit = tokens.Count > 0 ? "tokens" : "characters";
            long over = contextLength is { } limit && tokens.Count > 0 ? tokens.Count(t => t > limit) : 0;
            context.Write($"{path}: {rows.Count:N0} rows{(conversations > 0 ? $" ({conversations:N0} conversations)" : "")}");
            context.Table(["", "Mean", "Median", "p95", "Max", "Total"],
            [
                Row("characters", characters),
                Row("words", words),
                .. tokens.Count > 0 ? [Row("tokens", tokens)] : Array.Empty<IReadOnlyList<string>>(),
            ]);
            var histogram = Histogram(lengths);
            if (histogram.Count > 0)
            {
                context.Write($"\nlength in {unit}:");
                int most = histogram.Max(h => h.Count);
                foreach (var (from, to, count) in histogram)
                {
                    context.Write($"  {from,8:N0} - {to,-8:N0} {new string('#', (int)Math.Ceiling(30.0 * count / Math.Max(1, most))),-30} {count:N0}");
                }
            }

            if (contextLength is { } length && tokens.Count > 0)
            {
                context.Write($"\n{over:N0} rows ({over / (double)Math.Max(1, rows.Count):P1}) are longer than {length:N0} tokens");
            }

            context.WriteJson(new JsonObject
            {
                ["file"] = path,
                ["rows"] = rows.Count,
                ["conversations"] = conversations,
                ["characters"] = Json(characters),
                ["words"] = Json(words),
                ["tokens"] = tokens.Count > 0 ? Json(tokens) : null,
                ["context"] = contextLength,
                ["overContext"] = tokens.Count > 0 && contextLength is not null ? over : null,
                ["histogram"] = new JsonArray([.. histogram.Select(h => (JsonNode)new JsonObject { ["from"] = h.From, ["to"] = h.To, ["rows"] = h.Count })]),
            });
            return ExitCodes.Ok;
        }
        finally
        {
            loaded?.Dispose();
        }
    }

    // Words as string.Split(null, RemoveEmptyEntries) counts them (runs of characters that are not white space),
    // without the array of words per row.
    private static int Words(ReadOnlySpan<char> text)
    {
        int count = 0;
        bool word = false;
        foreach (char c in text)
        {
            bool white = char.IsWhiteSpace(c);
            if (!white && !word)
            {
                count++;
            }

            word = !white;
        }

        return count;
    }

    // values sorted.
    private static IReadOnlyList<string> Row(string name, List<int> sorted)
    {
        if (sorted.Count == 0)
        {
            return [name, "", "", "", "", ""];
        }

        string F(double v) => v.ToString("N0", CultureInfo.InvariantCulture);
        return [name, F(sorted.Average()), F(sorted[sorted.Count / 2]), F(sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * 0.95))]), F(sorted[^1]), F(sorted.Sum(v => (long)v))];
    }

    // values sorted.
    private static JsonObject Json(List<int> sorted) => sorted.Count == 0 ? [] : new JsonObject
    {
        ["mean"] = Math.Round(sorted.Average(), 2), ["median"] = sorted[sorted.Count / 2], ["max"] = sorted[^1], ["total"] = sorted.Sum(v => (long)v),
    };

    // Ten equal buckets from the shortest to the longest (values sorted), counted in one pass.
    private static List<(int From, int To, int Count)> Histogram(List<int> sorted)
    {
        if (sorted.Count == 0)
        {
            return [];
        }

        int min = sorted[0], max = sorted[^1];
        int buckets = Math.Min(10, max - min + 1), width = (int)Math.Ceiling((max - min + 1) / (double)buckets);
        var counts = new int[buckets];
        foreach (int v in sorted)
        {
            counts[(v - min) / width]++;
        }

        return [.. Enumerable.Range(0, buckets).Select(b => (min + b * width, min + (b + 1) * width - 1, counts[b]))];
    }
}

/// <summary><c>idrak data convert IN OUT</c>: between formats, and from common chat layouts to chat rows.</summary>
internal sealed class DataConvertCommand : Command
{
    public override string Name => "data convert";

    public override string Summary => "Convert rows between CSV, JSON Lines and JSON (Parquet is read), and chat layouts to chat rows";

    public override string Usage => """
        IN OUT [--as chat|preference|text] [-s SYSTEM] [-f]

        Arguments:
          IN   a CSV, TSV, JSON Lines, JSON or Parquet file
          OUT  the file to write; its extension picks the format: .jsonl, .json, .csv or .tsv (writing Parquet is
               not in the library yet)

        Options:
              --as chat        rows as conversations {"messages": [...]} (from messages, ShareGPT, Alpaca, question/answer, ...)
              --as preference  rows as {"prompt", "chosen", "rejected"}; --as text: rows as {"text"}
          -s, --system S       a system message for conversations without one
          -f, --force          overwrite OUT

        Examples:
          idrak data convert train.parquet train.jsonl
          idrak data convert alpaca.json chats.jsonl --as chat -s "Answer briefly."
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--as", "--system"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-s"] = "--system", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "IN (the file to convert)"), output = context.Argument(1, "OUT (the file to write)");
        string? kind = context.Option("--as");
        if (kind is not (null or "chat" or "preference" or "text"))
        {
            throw new UsageException($"--as {kind}: use chat, preference or text.");
        }

        if (Path.GetFullPath(input) == Path.GetFullPath(output))
        {
            throw new UsageException("IN and OUT are the same file; write to another name.");
        }

        if (File.Exists(output) && !context.Flag("--force"))
        {
            context.Error($"{output} exists; give -f, --force to overwrite it.");
            return ExitCodes.Failed;
        }

        string? system = context.Option("--system");
        long read = 0, dropped = 0;
        var rows = RowFiles.Read(input).Select(row =>
        {
            read++;
            var converted = kind switch
            {
                "chat" => ChatRows.Normalize(row, RowKind.Chat, null, system),
                "preference" => ChatRows.Preference(row, system),
                "text" => ChatRows.Normalize(row, RowKind.Text, null, system),
                _ => row,
            };
            if (converted is null)
            {
                dropped++;
            }

            return converted;
        }).OfType<JsonObject>();
        long written = RowFiles.Write(rows, output);
        context.Write($"{written:N0} rows written to {output} ({RowFiles.FormatName(input)} to {RowFiles.FormatName(output)}{(kind is null ? "" : $", as {kind}")})");
        if (dropped > 0)
        {
            context.Write($"{dropped:N0} of {read:N0} rows dropped: not readable as {kind} (idrak data validate {input} --as {kind} shows why)");
        }

        context.WriteJson(new JsonObject { ["input"] = input, ["output"] = output, ["read"] = read, ["written"] = written, ["dropped"] = dropped });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak data dedupe FILE</c>: removes repeated rows, exact or near (text compared without case, spacing and punctuation).</summary>
internal sealed class DataDedupeCommand : Command
{
    public override string Name => "data dedupe";

    public override string Summary => "Remove duplicate rows (exact, or near: text compared without case, spacing and punctuation)";

    public override string Usage => """
        FILE [-o OUT] [--columns A,B] [--near] [--dry-run] [-y]

        Options:
          -o, --out FILE     where to write the rows kept (default: FILE itself, which needs -y, --yes)
              --columns A,B  compare only these columns (default: every column)
              --near         text values are compared lower-cased, with letters and digits only, spaces collapsed
              --dry-run      count the duplicates without writing
          -y, --yes          overwrite FILE without asking

        Examples:
          idrak data dedupe chats.jsonl -o chats.unique.jsonl --near
          idrak data dedupe houses.csv --dry-run
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--columns"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--near", "--dry-run", "--yes"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-y"] = "--yes" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE");
        string output = context.Option("--out") ?? path;
        bool dryRun = Terminal.DryRun(context), near = context.Flag("--near");
        if (!dryRun && Path.GetFullPath(output) == Path.GetFullPath(path) && !Terminal.Confirm(context, $"Overwrite {path} with the rows kept (-o FILE writes elsewhere)?"))
        {
            context.Error($"{path} is left as it was.");
            return ExitCodes.Failed;
        }

        var columns = context.Option("--columns")?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var rows = RowFiles.Read(path).ToList();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var key = new StringBuilder();
        var kept = rows.Where(row => seen.Add(Key(row, columns, near, key))).ToList();
        int removed = rows.Count - kept.Count;
        if (!dryRun)
        {
            RowFiles.Write(kept, output);
        }

        context.Write($"{path}: {rows.Count:N0} rows, {removed:N0} {(near ? "near " : "")}duplicates{(dryRun ? " (dry run: nothing written)" : $" removed, {kept.Count:N0} written to {output}")}");
        context.WriteJson(new JsonObject { ["file"] = path, ["rows"] = rows.Count, ["duplicates"] = removed, ["kept"] = kept.Count, ["output"] = dryRun ? null : output, ["near"] = near });
        return ExitCodes.Ok;
    }

    // The row's key, built in key (one builder reused for every row).
    private static string Key(JsonObject row, string[]? columns, bool near, StringBuilder key)
    {
        key.Clear();
        foreach (var (name, value) in row)
        {
            if (columns is not null && !columns.Contains(name))
            {
                continue;
            }

            key.Append(name).Append('\u0001');
            string text = RowFiles.Raw(value);
            if (near)
            {
                AppendNear(key, text);
            }
            else
            {
                key.Append(text);
            }

            key.Append('\u0002');
        }

        return key.ToString();
    }

    // The text lower-cased with letters and digits only, each run of other characters one space, none at the ends: the
    // words of text.ToLowerInvariant() with the others as spaces, split on spaces and joined by one, in one pass.
    private static void AppendNear(StringBuilder key, string text)
    {
        bool word = false, any = false;
        foreach (char c in text)
        {
            char lower = char.ToLowerInvariant(c);
            if (!char.IsLetterOrDigit(lower))
            {
                word = false;
                continue;
            }

            if (!word && any)
            {
                key.Append(' ');
            }

            key.Append(lower);
            word = any = true;
        }
    }
}

/// <summary><c>idrak data split FILE</c>: train, validation and test files, shuffled with a seed, optionally stratified.</summary>
internal sealed class DataSplitCommand : Command
{
    public override string Name => "data split";

    public override string Summary => "Split rows into train, validation and test files with a seed (stratified by a column with -t)";

    public override string Usage => """
        FILE [--validation F] [--test F] [--seed N] [-t COL] [-o PREFIX]

        Options:
              --validation F  the validation fraction (default 0.1); --test F: the test fraction (default 0.1); 0 skips a part
              --seed N        the shuffle's seed (default 1)
          -t, --target C      keep this column's class balance in every part
          -o, --out P         the files' prefix (default FILE without its extension): P.train.EXT, P.validation.EXT, P.test.EXT

        Examples:
          idrak data split houses.csv --test 0.2 --validation 0
          idrak data split reviews.jsonl -t label -o splits/reviews
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--validation", "--test", "--target", "--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-t"] = "--target", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE");
        double validation = Fraction(context, "--validation", 0.1), test = Fraction(context, "--test", 0.1);
        if (validation + test >= 1)
        {
            throw new UsageException($"--validation {validation} and --test {test} leave nothing to train on.");
        }

        int seed = context.Seed ?? 1;
        string? target = context.Option("--target");
        var rows = RowFiles.Read(path).ToList();
        string format = RowFiles.FormatName(path);
        string extension = format is "csv" or "tsv" or "jsonl" or "json" ? format : "jsonl";
        string prefix = context.Option("--out") ?? Path.Combine(Path.GetDirectoryName(path) ?? "", Path.GetFileNameWithoutExtension(path));
        var parts = new Dictionary<string, List<JsonObject>> { ["train"] = [], ["validation"] = [], ["test"] = [] };
        var random = new Random(seed);
        foreach (var group in Groups(rows, target))
        {
            var shuffled = group.OrderBy(_ => random.Next()).ToList();
            int testCount = (int)Math.Round(shuffled.Count * test), validationCount = (int)Math.Round(shuffled.Count * validation);
            parts["test"].AddRange(shuffled.Take(testCount));
            parts["validation"].AddRange(shuffled.Skip(testCount).Take(validationCount));
            parts["train"].AddRange(shuffled.Skip(testCount + validationCount));
        }

        var written = new JsonObject();
        foreach (var (name, part) in parts)
        {
            if (name != "train" && (name == "test" ? test : validation) == 0)
            {
                continue;
            }

            string file = $"{prefix}.{name}.{extension}";
            RowFiles.Write(part, file);
            written[name] = new JsonObject { ["file"] = file, ["rows"] = part.Count };
            context.Write($"{name,-10} {part.Count,8:N0} rows  {file}");
        }

        context.WriteJson(new JsonObject { ["file"] = path, ["rows"] = rows.Count, ["seed"] = seed, ["stratifiedBy"] = target, ["parts"] = written });
        return ExitCodes.Ok;
    }

    internal static double Fraction(CommandContext context, string option, double fallback) =>
        context.Option(option) is not { } text ? fallback
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v is >= 0 and < 1 ? v
        : throw new UsageException($"{option} needs a fraction from 0 to 1, not '{text}'.");

    /// <summary>The rows grouped by <paramref name="target"/>'s value (one group without a target).</summary>
    internal static IEnumerable<List<JsonObject>> Groups(List<JsonObject> rows, string? target)
    {
        if (target is null)
        {
            return [rows];
        }

        if (rows.Count > 0 && rows.All(r => !r.ContainsKey(target)))
        {
            throw new UsageException($"No row has the column '{target}' (columns: {string.Join(", ", rows[0].Select(p => p.Key))}).");
        }

        return rows.GroupBy(r => RowFiles.Raw(r[target]), StringComparer.Ordinal).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => g.ToList());
    }
}

/// <summary><c>idrak data sample FILE -n N</c>: a random or stratified sample of rows.</summary>
internal sealed class DataSampleCommand : Command
{
    public override string Name => "data sample";

    public override string Summary => "A random (or, with -t, stratified) sample of rows";

    public override string Usage => """
        FILE -n N [--seed N] [-t COL] [-o OUT]

        Options:
          -n, --rows N    how many rows
              --seed N    the seed (default 1)
          -t, --target C  keep this column's class proportions
          -o, --out FILE  write the sample (by extension); without it the rows are printed as JSON Lines

        Examples:
          idrak data sample chats.jsonl -n 100 -o small.jsonl
          idrak data sample reviews.csv -n 500 -t label -o reviews.sample.csv
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--rows", "--target", "--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-n"] = "--rows", ["-t"] = "--target", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE");
        int count = context.Option("--rows") is null ? throw new UsageException("Give -n N, the number of rows to sample.") : context.IntOption("--rows", 0);
        if (count <= 0)
        {
            throw new UsageException("-n needs a positive number of rows.");
        }

        var random = new Random(context.Seed ?? 1);
        var rows = RowFiles.Read(path).ToList();
        var groups = DataSplitCommand.Groups(rows, context.Option("--target")).ToList();
        int wanted = Math.Min(count, rows.Count);

        // Each group's share by largest remainder, so the shares add up to the sample size.
        var shares = groups.Select(g => wanted * (double)g.Count / Math.Max(1, rows.Count)).ToList();
        var take = shares.Select(s => (int)Math.Floor(s)).ToArray();
        foreach (int i in shares.Select((s, i) => (s - Math.Floor(s), i)).OrderByDescending(p => p.Item1).Select(p => p.i).Take(wanted - take.Sum()))
        {
            take[i]++;
        }

        var sample = groups.SelectMany((g, i) => g.OrderBy(_ => random.Next()).Take(take[i])).OrderBy(_ => random.Next()).ToList();
        if (context.Option("--out") is { } output)
        {
            RowFiles.Write(sample, output);
            context.Write($"{sample.Count:N0} of {rows.Count:N0} rows written to {output}");
        }
        else if (!context.Json && !context.Quiet)
        {
            foreach (var row in sample)
            {
                context.Output.WriteLine(row.ToJsonString(CommandContext.JsonLine));
            }
        }

        context.WriteJson(new JsonObject
        {
            ["file"] = path,
            ["rows"] = rows.Count,
            ["sampled"] = sample.Count,
            ["output"] = context.Option("--out"),
            ["sample"] = context.Option("--out") is null ? new JsonArray([.. sample.Select(r => (JsonNode)r.DeepClone())]) : null,
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak data mix RECIPE.json</c>: a training set assembled from several sources with weights.</summary>
internal sealed class DataMixCommand : Command
{
    public override string Name => "data mix";

    public override string Summary => "Assemble a training set from several sources with weights (a dataset recipe)";

    public override string Usage => """
        RECIPE.json -o OUT [--eval FILE] [--eval-fraction F] [--seed N]

        Arguments:
          RECIPE.json  {"sources": [{"source": "hf:...", "weight": 2, "take": 4000}, "local.jsonl?weight=0.5"],
                       "system": "...", "seed": 1, "min_chars": 20, "eval_fraction": 0.02} (idrak help data has more)

        Options:
          -o, --out FILE                   the training rows (.jsonl, .json or .csv)
              --eval FILE                  the held-out rows (default OUT.eval.jsonl) when the recipe or --eval-fraction holds some out
              --eval-fraction F, --seed N  override the recipe's

        Examples:
          idrak data mix recipe.json -o train.jsonl
          idrak data mix recipe.json -o train.jsonl --eval-fraction 0.05 --seed 2
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--eval", "--eval-fraction"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string recipePath = context.Argument(0, "RECIPE.json");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the training rows.");
        var recipe = DatasetRecipe.Load(recipePath);
        if (context.Option("--eval-fraction") is not null)
        {
            recipe = recipe with { EvaluationFraction = DataSplitCommand.Fraction(context, "--eval-fraction", 0) };
        }

        if (context.Seed is { } seed)
        {
            recipe = recipe with { Seed = seed };
        }

        var downloads = Http.Downloader(context);
        var counts = new RecipeCounts();
        var (train, evaluation) = recipe.Build(downloads, counts);
        long written = RowFiles.Write(train, output);
        context.Write($"{written:N0} rows from {recipe.Sources.Count} source{(recipe.Sources.Count == 1 ? "" : "s")} written to {output}"
                      + (recipe.Deduplicate ? $" ({counts.Duplicates:N0} repeats dropped)" : ""));
        string? evalFile = null;
        long held = 0;
        if (evaluation is not null)
        {
            evalFile = context.Option("--eval") ?? Path.ChangeExtension(output, null) + ".eval.jsonl";
            held = RowFiles.Write(evaluation, evalFile);
            context.Write($"{held:N0} evaluation rows written to {evalFile}");
        }

        context.WriteJson(new JsonObject
        {
            ["recipe"] = recipePath,
            ["sources"] = new JsonArray([.. recipe.Sources.Select(s => (JsonNode)s.ToString())]),
            ["output"] = output,
            ["rows"] = written,
            ["duplicates"] = counts.Duplicates,
            ["evalOutput"] = evalFile,
            ["evalRows"] = held,
        });
        return ExitCodes.Ok;
    }
}
