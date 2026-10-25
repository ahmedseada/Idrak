// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Imaging;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

/// <summary>
/// The parts a synthetic page adds around its record's text: header lines at the start, centre and end, a title, a
/// subject, a signature and a footer, each drawn per page from its alternatives at its chance, their slots
/// (<c>{digits:6}</c>, <c>{hijri}</c>, ...; see <see cref="PagePartFillers"/>) filled from the page's seed. Written as
/// JSON (<see cref="Read"/>): <c>{"headerStart": [["line", "line"], ...], "headerCenter", "headerEnd", "titles": ["..."],
/// "subjects", "signatures", "footers", "chances": {"headerStart": 0.9, ...}, "digits": "٠١٢٣٤٥٦٧٨٩", "nativeDigits":
/// 0.3, "language": "ar"}</c>; an alternative may be one line as a string. The library has no parts of its own: a page
/// with none is its record's text alone.
/// </summary>
public sealed record SyntheticPageParts
{
    /// <summary>No parts: the record's text alone.</summary>
    public static SyntheticPageParts Empty { get; } = new();

    /// <summary>Alternatives for the header's start side (each a list of lines).</summary>
    public IReadOnlyList<IReadOnlyList<string>> HeaderStart { get; init; } = [];

    /// <summary>Alternatives for the header's centre.</summary>
    public IReadOnlyList<IReadOnlyList<string>> HeaderCenter { get; init; } = [];

    /// <summary>Alternatives for the header's end side (number, date, attachments).</summary>
    public IReadOnlyList<IReadOnlyList<string>> HeaderEnd { get; init; } = [];

    /// <summary>Alternatives for a title.</summary>
    public IReadOnlyList<string> Titles { get; init; } = [];

    /// <summary>Alternatives for a subject line.</summary>
    public IReadOnlyList<string> Subjects { get; init; } = [];

    /// <summary>Alternatives for the signature block.</summary>
    public IReadOnlyList<IReadOnlyList<string>> Signatures { get; init; } = [];

    /// <summary>Alternatives for the footer.</summary>
    public IReadOnlyList<IReadOnlyList<string>> Footers { get; init; } = [];

    /// <summary>How often each part is drawn (0 to 1), by its key ("headerStart", "headerCenter", "headerEnd", "title", "subject", "signature", "footer"); 1 when not given.</summary>
    public IReadOnlyDictionary<string, double> Chances { get; init; } = new Dictionary<string, double>();

    /// <summary>Ten digits 0 to 9 of the language's own (<c>٠١٢٣٤٥٦٧٨٩</c>), written into the parts instead of 0-9 on <see cref="NativeDigits"/> of the pages; null for none.</summary>
    public string? Digits { get; init; }

    /// <summary>The share of pages whose parts use <see cref="Digits"/> (0 to 1).</summary>
    public double NativeDigits { get; init; }

    /// <summary>The pages' language (BCP 47, "ar"), for the shapers; null for none.</summary>
    public string? Language { get; init; }

    /// <summary>The folder <c>{line:FILE}</c> paths are relative to (the parts file's).</summary>
    public string? Folder { get; init; }

    private static readonly string[] Keys = ["headerStart", "headerCenter", "headerEnd", "titles", "subjects", "signatures", "footers", "chances", "digits", "nativeDigits", "language"];

    internal static readonly string[] PartKeys = ["headerStart", "headerCenter", "headerEnd", "title", "subject", "signature", "footer"];

    /// <summary>The parts in the JSON file at <paramref name="path"/> (see the class).</summary>
    /// <exception cref="InvalidDataException">A key or a slot that is not known, or a value of the wrong kind; the message names it.</exception>
    public static SyntheticPageParts Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = JsonNode.Parse(File.ReadAllBytes(path), documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true }) as JsonObject
            ?? throw new InvalidDataException($"{path}: page parts are a JSON object.");
        return FromJson(json, Path.GetDirectoryName(Path.GetFullPath(path)), path);
    }

    /// <summary>The parts in <paramref name="json"/> (see the class); <paramref name="folder"/> for <c>{line:FILE}</c>.</summary>
    public static SyntheticPageParts FromJson(JsonObject json, string? folder = null, string where = "page parts")
    {
        ArgumentNullException.ThrowIfNull(json);
        foreach (var (key, _) in json)
        {
            if (!Keys.Contains(key, StringComparer.Ordinal))
            {
                throw new InvalidDataException($"{where}: unknown key '{key}' (known: {string.Join(", ", Keys)}).");
            }
        }

        IReadOnlyList<IReadOnlyList<string>> Blocks(string key) => json[key] switch
        {
            null => [],
            JsonArray a => [.. a.Select(item => item switch
            {
                JsonValue v when v.GetValueKind() == JsonValueKind.String => (IReadOnlyList<string>)[v.GetValue<string>()],
                JsonArray lines => [.. lines.Select(l => l is JsonValue lv && lv.GetValueKind() == JsonValueKind.String ? lv.GetValue<string>()
                    : throw new InvalidDataException($"{where}: {key}'s lines are text, not {l?.ToJsonString()}."))],
                _ => throw new InvalidDataException($"{where}: {key} holds lines (a list of text) or one line (text), not {item?.ToJsonString()}."),
            })],
            var other => throw new InvalidDataException($"{where}: {key} is a list of alternatives, not {other.ToJsonString()}."),
        };

        IReadOnlyList<string> Lines(string key) => [.. Blocks(key).Select(b => b.Count == 1 ? b[0]
            : throw new InvalidDataException($"{where}: each of {key} is one line."))];

        var chances = new Dictionary<string, double>(StringComparer.Ordinal);
        if (json["chances"] is JsonObject c)
        {
            foreach (var (key, value) in c)
            {
                if (!PartKeys.Contains(key, StringComparer.Ordinal))
                {
                    throw new InvalidDataException($"{where}: unknown part '{key}' in chances (parts: {string.Join(", ", PartKeys)}).");
                }

                double chance = value is JsonValue v && v.TryGetValue(out double d) ? d : double.NaN;
                chances[key] = chance is >= 0 and <= 1 ? chance : throw new InvalidDataException($"{where}: the chance of {key} is from 0 to 1, not {value?.ToJsonString()}.");
            }
        }
        else if (json["chances"] is { } bad)
        {
            throw new InvalidDataException($"{where}: chances is an object of part: chance, not {bad.ToJsonString()}.");
        }

        string? digits = (string?)json["digits"];
        if (digits is not null && digits.EnumerateRunes().Count() != 10)
        {
            throw new InvalidDataException($"{where}: digits are the ten digits 0 to 9 of a language, not '{digits}'.");
        }

        double native = json["nativeDigits"] is JsonValue n && n.TryGetValue(out double nd) ? nd : 0;
        if (native is < 0 or > 1 || (native > 0 && digits is null))
        {
            throw new InvalidDataException($"{where}: nativeDigits is a share from 0 to 1, with digits given.");
        }

        var parts = new SyntheticPageParts
        {
            HeaderStart = Blocks("headerStart"), HeaderCenter = Blocks("headerCenter"), HeaderEnd = Blocks("headerEnd"),
            Titles = Lines("titles"), Subjects = Lines("subjects"), Signatures = Blocks("signatures"), Footers = Blocks("footers"),
            Chances = chances, Digits = digits, NativeDigits = native, Language = (string?)json["language"], Folder = folder,
        };

        // Every slot names a registered filler, found now rather than on the thousandth page.
        foreach (string line in parts.AllLines())
        {
            foreach (var (name, _) in PagePartTemplate.Slots(line, where))
            {
                if (PagePartFillers.Find(name) is null)
                {
                    throw new InvalidDataException($"{where}: no slot {{{name}}} is registered, in \"{line}\" (registered: {string.Join(", ", PagePartFillers.Names)}); add it with PagePartFillers.Register.");
                }
            }
        }

        return parts;
    }

    internal IEnumerable<string> AllLines() =>
        HeaderStart.Concat(HeaderCenter).Concat(HeaderEnd).Concat(Signatures).Concat(Footers).SelectMany(b => b).Concat(Titles).Concat(Subjects);

    internal double Chance(string part) => Chances.TryGetValue(part, out double c) ? c : 1;
}

/// <summary>How synthetic pages are made (see <see cref="SyntheticPages"/>).</summary>
public sealed record SyntheticPageOptions
{
    /// <summary>The fonts (those that have a page's characters are drawn from), sizes, spacing and chances of the page layouts.</summary>
    public required PageStyle Style { get; init; }

    /// <summary>The parts around each record's text (header, subject, signature, footer); none unless given.</summary>
    public SyntheticPageParts Parts { get; init; } = SyntheticPageParts.Empty;

    /// <summary>Pages made from each record.</summary>
    public int Variants { get; init; } = 3;

    /// <summary>The data set's seed: the same seed, records and options give the same pages, byte for byte.</summary>
    public int Seed { get; init; }

    /// <summary>The page layouts (<see cref="PageLayouts"/>), one drawn per page; a name given twice is drawn twice as often.</summary>
    public IReadOnlyList<string> Layouts { get; init; } = ["letter", "plain"];

    /// <summary>
    /// The scan profiles (<see cref="ScanProfiles"/>: a name, or a profile written as text; "none" for a page as drawn),
    /// one drawn per page; a name given twice is drawn twice as often. By default every built-in look, "mixed" (every
    /// effect at a small chance of its own) three times, so pages shuffle through all of them.
    /// </summary>
    public IReadOnlyList<string> Scans { get; init; } = ["none", "clean", "office", "old", "pale", "worn", "fax", "handled", "marked", "damaged", "scanner", "aged", "copied", "photo", "wrecked", "mixed", "mixed", "mixed"];

    /// <summary>The share of pages drawn in grey (else as the style says).</summary>
    public double Grey { get; init; } = 0.3;

    /// <summary>The record format (<see cref="SyntheticRecordFormats"/>); null to find it from the input.</summary>
    public string? Format { get; init; }

    /// <summary>The answer's JSON key holding the page's text; null for the first key starting with "full_text", else the longest text.</summary>
    public string? FullTextKey { get; init; }

    /// <summary>The question written into records made from plain text.</summary>
    public string Prompt { get; init; } = "Extract the text.";

    /// <summary>The JSON key a plain text's answer is written under; null for the text itself.</summary>
    public string? AnswerKey { get; init; }

    /// <summary>The pages' image format: "jpeg" (at <see cref="Quality"/>) or a registered encoder (<see cref="ImageEncoders"/>, "png").</summary>
    public string ImageFormat { get; init; } = "jpeg";

    /// <summary>The JPEG quality (1 to 100).</summary>
    public int Quality { get; init; } = 90;

    /// <summary>Write the images into a .zip beside the records instead of a folder.</summary>
    public bool Zip { get; init; }

    /// <summary>Pages made at once; 0 for one per processor.</summary>
    public int Workers { get; init; }

    /// <summary>The earliest date a page's date slots are drawn from.</summary>
    public DateTime DateFrom { get; init; } = new(2000, 1, 1);

    /// <summary>The latest.</summary>
    public DateTime DateTo { get; init; } = new(2030, 12, 31);

    /// <summary>Records read at most (0 for all), for a trial.</summary>
    public int Limit { get; init; }

    internal void Check()
    {
        ArgumentNullException.ThrowIfNull(Style);
        if (Style.Fonts.Count == 0)
        {
            throw new ArgumentException("Synthetic pages need at least one font (SyntheticPageOptions.Style.Fonts).");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(Variants, 1, nameof(Variants));
        ArgumentOutOfRangeException.ThrowIfNegative(Workers, nameof(Workers));
        ArgumentOutOfRangeException.ThrowIfNegative(Limit, nameof(Limit));
        if (Layouts.Count == 0 || Scans.Count == 0)
        {
            throw new ArgumentException("Synthetic pages need at least one layout and one scan profile (\"none\" for pages as drawn).");
        }

        foreach (string layout in Layouts)
        {
            _ = PageLayouts.Get(layout);
        }

        foreach (string scan in Scans)
        {
            _ = SyntheticPages.Profile(scan);
        }

        if (Style.Crop is < 0 or > 1 || !(Style.CropPadding.Min >= 0 && Style.CropPadding.Min <= Style.CropPadding.Max))
        {
            throw new ArgumentOutOfRangeException(nameof(Style), $"Crop is a chance from 0 to 1 ({Style.Crop}) and its padding a range of pixels from 0 ({Style.CropPadding}).");
        }

        if (Grey is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(Grey), Grey, "Grey is a share from 0 to 1.");
        }

        if (Quality is < 1 or > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(Quality), Quality, "The JPEG quality is 1 to 100.");
        }

        if (!ImageFormat.Equals("jpeg", StringComparison.OrdinalIgnoreCase))
        {
            _ = ImageEncoders.Get(ImageFormat);
        }

        if (DateFrom > DateTo)
        {
            throw new ArgumentException($"DateFrom ({DateFrom:yyyy-MM-dd}) is after DateTo ({DateTo:yyyy-MM-dd}).");
        }
    }
}

/// <summary>One synthetic page: its image after its scan, its text in reading order, its record's new answer, and how it was made.</summary>
/// <param name="Image">The page as scanned.</param>
/// <param name="Text">The drawn text in reading order.</param>
/// <param name="Answer">The record's answer with its page text replaced by <paramref name="Text"/>.</param>
/// <param name="Seed">The page's seed.</param>
/// <param name="Layout">The layout's name.</param>
/// <param name="Scan">The scan's steps ("" for none).</param>
/// <param name="Profile">The scan profile drawn for it ("none", "clean", ..., or a profile's text).</param>
/// <param name="Description">What the layout chose (font, sizes, ornaments).</param>
public sealed record SyntheticPage(ImageData Image, string Text, string Answer, int Seed, string Layout, string Scan, string Description, string Profile);

/// <summary>How far making synthetic pages has come.</summary>
/// <param name="Pages">Pages made.</param>
/// <param name="Total">Pages to make.</param>
/// <param name="Skipped">Records left out (no text, or no font with their characters).</param>
/// <param name="Elapsed">Time so far.</param>
/// <param name="LastSkip">Why the latest record was left out, if one was.</param>
public sealed record SyntheticPagesProgress(int Pages, int Total, int Skipped, TimeSpan Elapsed, string? LastSkip)
{
    /// <summary>Pages a second so far.</summary>
    public double PagesPerSecond => Elapsed.TotalSeconds > 0 ? Pages / Elapsed.TotalSeconds : 0;

    /// <summary>The time left at the speed so far.</summary>
    public TimeSpan Remaining => PagesPerSecond > 0 ? TimeSpan.FromSeconds((Total - Pages) / PagesPerSecond) : TimeSpan.Zero;
}

/// <summary>What making synthetic pages wrote.</summary>
/// <param name="Records">Records read.</param>
/// <param name="Pages">Pages (records written).</param>
/// <param name="Skips">Records left out, with why (the first 50).</param>
/// <param name="Data">The written records (.jsonl).</param>
/// <param name="Images">The images' folder or zip.</param>
/// <param name="Manifest">How each page was made (.jsonl).</param>
/// <param name="Elapsed">The time it took.</param>
public sealed record SyntheticPagesResult(int Records, int Pages, IReadOnlyList<string> Skips, string Data, string Images, string Manifest, TimeSpan Elapsed);

/// <summary>
/// Tuning data from text (plan 18): each record's answer text drawn as pages (a layout from <see cref="PageLayouts"/>,
/// header, subject, signature and footer parts around it), made to look scanned (a profile from
/// <see cref="ScanProfiles"/>), the record written again with the page as its image and the drawn text, in reading order,
/// as its answer's text. Every choice comes from the data set's seed, the record and the variant: the same inputs give
/// the same bytes. Pages are made on every processor and written in order.
/// </summary>
public static class SyntheticPages
{
    /// <summary>The scan profile <paramref name="nameOrText"/> names or writes; null for "none".</summary>
    public static IScanProfile? Profile(string nameOrText) =>
        nameOrText.Trim().Equals("none", StringComparison.OrdinalIgnoreCase) ? null : ScanProfiles.Resolve(nameOrText);

    /// <summary>The keys <see cref="Style"/> reads.</summary>
    public static IReadOnlyList<string> StyleKeys { get; } =
        ["width", "height", "size", "spacing", "margin", "ink", "justify", "color", "rule", "emblem", "stamp", "blue-signature", "header-font", "crop", "crop-padding"];

    /// <summary>
    /// <paramref name="style"/> with the settings of <paramref name="settings"/> over it, written
    /// <c>KEY=VALUE,...</c>: width and height (pixels), size (the body's pixels per em), spacing (line height
    /// multiples), margin (a share of the width), ink (darkness, 0 to 1) and crop-padding (pixels) as ranges
    /// <c>MIN..MAX</c> or one value; justify, rule, emblem, stamp, blue-signature, header-font and crop (a page cut to
    /// its drawn parts, <see cref="PageStyle.Crop"/>) as chances (0 to 1); color as true or false.
    /// </summary>
    /// <exception cref="ArgumentException">A key not known, or a value out of its range; the message names them.</exception>
    public static PageStyle Style(PageStyle style, string? settings)
    {
        ArgumentNullException.ThrowIfNull(style);
        foreach (string raw in (settings ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = raw.IndexOf('=');
            string key = (eq < 0 ? raw : raw[..eq]).Trim().ToLowerInvariant(), value = eq < 0 ? "" : raw[(eq + 1)..].Trim();
            (float Min, float Max) Range(float least, float most)
            {
                int dots = value.IndexOf("..", StringComparison.Ordinal);
                string a = dots < 0 ? value : value[..dots], b = dots < 0 ? value : value[(dots + 2)..];
                return float.TryParse(a, NumberStyles.Float, CultureInfo.InvariantCulture, out float min) && float.TryParse(b, NumberStyles.Float, CultureInfo.InvariantCulture, out float max)
                    && min >= least && max <= most && min <= max
                    ? (min, max)
                    : throw new ArgumentException($"Page setting {key} is a value or a range MIN..MAX from {least.ToString(CultureInfo.InvariantCulture)} to {most.ToString(CultureInfo.InvariantCulture)}, not '{value}'.");
            }

            float Chance() => Range(0, 1).Min;
            int Pixels() => (int)Range(64, 20000).Min;
            style = key switch
            {
                "width" => style with { Width = Pixels() },
                "height" => style with { Height = Pixels() },
                "size" => style with { BodySize = Range(4, 400) },
                "spacing" => style with { LineSpacing = Range(0.5f, 4) },
                "margin" => style with { Margin = Range(0, 0.4f) },
                "ink" => style with { Ink = Range(0, 1) },
                "justify" => style with { Justify = Chance() },
                "rule" => style with { Rule = Chance() },
                "emblem" => style with { Emblem = Chance() },
                "stamp" => style with { Stamp = Chance() },
                "blue-signature" => style with { BlueSignature = Chance() },
                "header-font" => style with { OtherHeaderFont = Chance() },
                "crop" => style with { Crop = Chance() },
                "crop-padding" => style with { CropPadding = Range(0, 2000) },
                "color" => style with
                {
                    Color = bool.TryParse(value, out bool color) ? color : throw new ArgumentException($"Page setting color is true or false, not '{value}'."),
                },
                _ => throw new ArgumentException($"Unknown page setting '{key}' (known: {string.Join(", ", StyleKeys)})."),
            };
        }

        return style;
    }

    /// <summary>The fonts in <paramref name="folder"/> (.ttf, .otf, .ttc; every face of a collection).</summary>
    public static IReadOnlyList<IFontFace> Fonts(string folder) => FontFormats.ReadFolder(folder);

    /// <summary>The page a record's answer makes as variant <paramref name="variant"/> of record <paramref name="record"/>.</summary>
    /// <exception cref="InvalidDataException">The answer has no page text, or no font has its characters.</exception>
    public static SyntheticPage Make(string answer, SyntheticPageOptions options, int record = 0, int variant = 0, string where = "the answer")
    {
        ArgumentNullException.ThrowIfNull(answer);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        return Draw(SyntheticAnswer.Parse(answer, options.FullTextKey, where), options, record, variant, where);
    }

    /// <summary>The records <paramref name="input"/> holds (read once to count them, for progress).</summary>
    public static int Count(string input, SyntheticPageOptions options)
    {
        var format = options.Format is null ? SyntheticRecordFormats.Detect(input) : SyntheticRecordFormats.Get(options.Format);
        int count = format.Read(input, options.Prompt, options.AnswerKey).Count();
        return options.Limit > 0 ? Math.Min(count, options.Limit) : count;
    }

    /// <summary>
    /// Makes <see cref="SyntheticPageOptions.Variants"/> pages from each record of <paramref name="input"/> and writes
    /// them: the records to <paramref name="output"/> (.jsonl, in the input's shape), the images to the folder of its
    /// name beside it (or its .zip, <see cref="SyntheticPageOptions.Zip"/>), as the records name them
    /// (<c>{name}/{record}-{variant}.jpg</c>), and how each page was made to <c>{name}.manifest.jsonl</c>.
    /// </summary>
    /// <exception cref="InvalidDataException">Every record was left out; the message says why the first was.</exception>
    public static SyntheticPagesResult Write(string input, string output, SyntheticPageOptions options, IProgress<SyntheticPagesProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(input);
        ArgumentException.ThrowIfNullOrEmpty(output);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        var format = options.Format is null ? SyntheticRecordFormats.Detect(input) : SyntheticRecordFormats.Get(options.Format);
        int total = Count(input, options) * options.Variants;

        string data = Path.GetFullPath(output.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase) ? output : output + ".jsonl");
        string folder = Path.GetDirectoryName(data)!, stem = Path.GetFileNameWithoutExtension(data);
        Directory.CreateDirectory(folder);
        string images = options.Zip ? Path.Combine(folder, stem + ".zip") : Path.Combine(folder, stem);
        string manifest = Path.Combine(folder, stem + ".manifest.jsonl");
        bool jpeg = options.ImageFormat.Equals("jpeg", StringComparison.OrdinalIgnoreCase);
        var encoder = jpeg ? null : ImageEncoders.Get(options.ImageFormat);
        string extension = jpeg ? ".jpg" : encoder!.Extensions.First();
        int workers = options.Workers > 0 ? options.Workers : ComputeResources.MaxCpuThreads;

        var watch = Stopwatch.StartNew();
        var skips = new List<string>();
        int skipped = 0, pages = 0, records = 0;
        string? firstSkip = null;
        // "\n" between records on every machine, so the same inputs give the same bytes everywhere (rule 70).
        using var dataWriter = new StreamWriter(data, append: false, new UTF8Encoding(false)) { NewLine = "\n" };
        using var manifestWriter = new StreamWriter(manifest, append: false, new UTF8Encoding(false)) { NewLine = "\n" };
        using var zip = options.Zip ? new ZipArchive(new FileStream(images, FileMode.Create, FileAccess.Write), ZipArchiveMode.Create) : null;
        if (!options.Zip)
        {
            Directory.CreateDirectory(images);
        }

        var source = format.Read(input, options.Prompt, options.AnswerKey);
        if (options.Limit > 0)
        {
            source = source.Take(options.Limit);
        }

        // Records in chunks; each chunk's pages made at once, then written in order.
        foreach (var chunk in source.Select((r, i) => (Record: r, Index: i)).Chunk(Math.Max(1, workers * 2 / options.Variants + 1)))
        {
            cancellationToken.ThrowIfCancellationRequested();
            records += chunk.Length;
            var jobs = chunk.SelectMany(c => Enumerable.Range(0, options.Variants).Select(v => (c.Record, c.Index, Variant: v))).ToArray();
            var made = new (byte[] Bytes, string Record, string Manifest)?[jobs.Length];
            var failed = new string?[jobs.Length];
            try
            {
                Parallel.For(0, jobs.Length, new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, j =>
                {
                    var (record, index, variant) = jobs[j];
                    try
                    {
                        var page = Draw(SyntheticAnswer.Parse(record.Answer, options.FullTextKey, record.Where), options, index, variant, record.Where);
                        byte[] bytes = jpeg ? JpegEncoder.Encode(page.Image, options.Quality, optimize: true) : encoder!.Encode(page.Image);
                        string name = $"{stem}/{index:D6}-{variant:D2}{extension}";
                        string line = Json(record.Write(page.Answer, name));
                        string about = Json(new JsonObject
                        {
                            ["image"] = name, ["record"] = index, ["variant"] = variant, ["where"] = record.Where, ["seed"] = page.Seed,
                            ["layout"] = page.Layout, ["profile"] = page.Profile, ["scan"] = page.Scan, ["page"] = page.Description,
                            ["width"] = page.Image.Width, ["height"] = page.Image.Height, ["channels"] = page.Image.Channels,
                        });
                        made[j] = (bytes, line, about);
                    }
                    catch (InvalidDataException e)
                    {
                        failed[j] = e.Message;
                    }
                });
            }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                // The first page's own error, not the loop's wrapper.
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }

            for (int j = 0; j < jobs.Length; j++)
            {
                if (made[j] is not { } page)
                {
                    // A record whose page cannot be drawn is left out once, not once a variant.
                    if (jobs[j].Variant == 0)
                    {
                        skipped++;
                        firstSkip ??= failed[j];
                        if (skips.Count < 50)
                        {
                            skips.Add(failed[j]!);
                        }
                    }

                    continue;
                }

                string name = $"{stem}/{jobs[j].Index:D6}-{jobs[j].Variant:D2}{extension}";
                if (zip is not null)
                {
                    using var entry = zip.CreateEntry(name, CompressionLevel.NoCompression).Open();
                    entry.Write(page.Bytes);
                }
                else
                {
                    File.WriteAllBytes(Path.Combine(folder, name), page.Bytes);
                }

                dataWriter.WriteLine(page.Record);
                manifestWriter.WriteLine(page.Manifest);
                pages++;
            }

            dataWriter.Flush();
            manifestWriter.Flush();
            progress?.Report(new SyntheticPagesProgress(pages, total, skipped, watch.Elapsed, skips.Count > 0 ? skips[^1] : null));
        }

        if (pages == 0 && skipped > 0)
        {
            throw new InvalidDataException($"No page could be made from {input}: every record was left out ({skipped}); the first: {firstSkip}");
        }

        return new SyntheticPagesResult(records, pages, skips, data, images, manifest, watch.Elapsed);
    }

    private static readonly JsonWriterOptions LineOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    internal static string Json(JsonNode node, bool indented = false, int indentSize = 2, char indentCharacter = ' ', string newLine = "\n")
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer, LineOptions with { Indented = indented, IndentSize = indentSize, IndentCharacter = indentCharacter, NewLine = newLine }))
        {
            node.WriteTo(writer);
        }

        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    // A page's seed from the data set's, the record's and the variant's (SplitMix64; the same in every process).
    internal static int PageSeed(int seed, int record, int variant)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)record) * 31 + (uint)variant + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return (int)((z ^ (z >> 31)) & int.MaxValue);
    }

    private static SyntheticPage Draw(SyntheticAnswer answer, SyntheticPageOptions options, int record, int variant, string where)
    {
        int seed = PageSeed(options.Seed, record, variant);
        var random = new Random(seed);
        string layoutName = options.Layouts[random.Next(options.Layouts.Count)];
        string scanName = options.Scans[random.Next(options.Scans.Count)];
        bool grey = random.NextDouble() < options.Grey;
        int layoutSeed = random.Next(), scanSeed = random.Next();

        var parts = options.Parts;
        var context = new PagePartContext
        {
            Random = random, Fields = answer.Fields, Folder = parts.Folder,
            Date = options.DateFrom.AddDays(Math.Floor(random.NextDouble() * ((options.DateTo - options.DateFrom).TotalDays + 1))),
        };
        bool native = parts.Digits is not null && random.NextDouble() < parts.NativeDigits;
        IReadOnlyList<string> Block(IReadOnlyList<IReadOnlyList<string>> alternatives, string part)
        {
            // Every part draws the same numbers whether it is taken or not, so one part's chance does not move another's.
            double take = random.NextDouble();
            int pick = random.Next(Math.Max(1, alternatives.Count));
            return alternatives.Count == 0 || take >= parts.Chance(part) ? [] : Fill(alternatives[pick], context, native ? parts.Digits : null);
        }

        string? Line(IReadOnlyList<string> alternatives, string part) =>
            Block([.. alternatives.Select(a => (IReadOnlyList<string>)[a])], part) is [var line] ? line : null;

        var body = answer.PageText.Replace("\r", "", StringComparison.Ordinal).Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToArray();
        if (body.Length == 0)
        {
            throw new InvalidDataException($"{where}: the answer's page text is empty.");
        }

        var content = new PageContent
        {
            HeaderStart = Block(parts.HeaderStart, "headerStart"), HeaderCenter = Block(parts.HeaderCenter, "headerCenter"),
            HeaderEnd = Block(parts.HeaderEnd, "headerEnd"), Title = Line(parts.Titles, "title"), Subject = Line(parts.Subjects, "subject"),
            Body = body, Signature = Block(parts.Signatures, "signature"), Footer = Block(parts.Footers, "footer"), Language = parts.Language,
        };

        var style = options.Style;
        var fonts = FontsFor(style.Fonts, content.ReadingOrder(), where);
        var pageStyle = style with { Fonts = fonts, Fallbacks = [.. style.Fallbacks, .. style.Fonts], Color = style.Color && !grey };
        var rendered = PageLayouts.Get(layoutName).Render(content, pageStyle, layoutSeed);
        var drawn = rendered.Image;
        string description = rendered.Description;
        // Cut to the drawn parts (PageStyle.Crop), before the scan so its look lands on the cut page. Drawn last and only
        // when asked, so every other choice, and a page that is not cut, stays as it was.
        if (style.Crop > 0f && random.NextDouble() < style.Crop && Bounds(rendered, style.CropPadding, random) is { } box)
        {
            drawn = Cut(drawn, box);
            description += string.Create(CultureInfo.InvariantCulture, $", cut to {box.Width} x {box.Height}");
        }

        var steps = Profile(scanName)?.Sample(scanSeed) ?? ImageTransformPipeline.Empty;
        var image = steps.IsEmpty ? drawn : steps.Apply(drawn);
        return new SyntheticPage(image, rendered.Text, answer.With(rendered.Text), seed, layoutName, steps.ToString(), description, scanName);
    }

    // The box around a page's drawn parts with padding drawn from its range, within the page; null when nothing was drawn.
    private static (int X, int Y, int Width, int Height)? Bounds(RenderedPage page, (float Min, float Max) padding, Random random)
    {
        float left = float.MaxValue, top = float.MaxValue, right = float.MinValue, bottom = float.MinValue;
        foreach (var block in page.Blocks)
        {
            if (block.Width > 0 && block.Height > 0)
            {
                (left, top) = (Math.Min(left, block.X), Math.Min(top, block.Y));
                (right, bottom) = (Math.Max(right, block.X + block.Width), Math.Max(bottom, block.Y + block.Height));
            }
        }

        if (left > right)
        {
            return null;
        }

        // A part's box spans its column (a short line's box is as wide as the page's text), so the box is narrowed to the
        // ink inside it: the pixels that differ from the paper (the page's corner) in a channel, rows read in order.
        var image = page.Image;
        int bx0 = Math.Clamp((int)MathF.Floor(left), 0, image.Width - 1), by0 = Math.Clamp((int)MathF.Floor(top), 0, image.Height - 1);
        int bx1 = Math.Clamp((int)MathF.Ceiling(right), bx0 + 1, image.Width), by1 = Math.Clamp((int)MathF.Ceiling(bottom), by0 + 1, image.Height);
        var pixels = image.Pixels;
        int plane = image.Height * image.Width;
        int inkLeft = int.MaxValue, inkTop = int.MaxValue, inkRight = -1, inkBottom = -1;
        for (int y = by0; y < by1; y++)
        {
            for (int x = bx0; x < bx1; x++)
            {
                bool ink = false;
                for (int c = 0; c < image.Channels && !ink; c++)
                {
                    ink = MathF.Abs(pixels[c * plane + y * image.Width + x] - pixels[c * plane]) > InkContrast;
                }

                if (ink)
                {
                    (inkLeft, inkRight) = (Math.Min(inkLeft, x), Math.Max(inkRight, x));
                    (inkTop, inkBottom) = (Math.Min(inkTop, y), y);
                }
            }
        }

        if (inkRight >= 0)
        {
            (bx0, by0, bx1, by1) = (inkLeft, inkTop, inkRight + 1, inkBottom + 1);
        }

        int pad = (int)MathF.Round(padding.Min + (float)random.NextDouble() * (padding.Max - padding.Min));
        int x0 = Math.Max(0, bx0 - pad), y0 = Math.Max(0, by0 - pad);
        int x1 = Math.Min(image.Width, bx1 + pad), y1 = Math.Min(image.Height, by1 + pad);
        return (x0, y0, x1 - x0, y1 - y0);
    }

    // How far a pixel's value is from the paper's for a cut page to count it as ink (values in [0, 1]).
    private const float InkContrast = 0.1f;

    // The pixels of `box`, each channel's rows copied in memory order.
    private static ImageData Cut(ImageData image, (int X, int Y, int Width, int Height) box)
    {
        var source = image.Pixels;
        var pixels = new float[image.Channels * box.Height * box.Width];
        for (int c = 0; c < image.Channels; c++)
        {
            for (int row = 0; row < box.Height; row++)
            {
                Array.Copy(source, ((long)c * image.Height + box.Y + row) * image.Width + box.X, pixels, ((long)c * box.Height + row) * box.Width, box.Width);
            }
        }

        return new ImageData(pixels, image.Channels, box.Height, box.Width);
    }

    private static IReadOnlyList<string> Fill(IReadOnlyList<string> lines, PagePartContext context, string? digits)
    {
        var filled = new List<string>(lines.Count);
        foreach (string line in lines)
        {
            if (PagePartTemplate.Fill(line, context) is { Length: > 0 } text)
            {
                filled.Add(digits is null ? text : NativeDigits(text, digits));
            }
        }

        return filled;
    }

    private static string NativeDigits(string text, string digits)
    {
        var runes = digits.EnumerateRunes().Select(r => r.ToString()).ToArray();
        var sb = new StringBuilder(text.Length);
        foreach (char ch in text)
        {
            sb.Append(ch is >= '0' and <= '9' ? runes[ch - '0'] : ch.ToString());
        }

        return sb.ToString();
    }

    // The fonts that have every character the page draws; else those that, with the other fonts behind them, do.
    private static IReadOnlyList<IFontFace> FontsFor(IReadOnlyList<IFontFace> fonts, string text, string where)
    {
        var needed = new HashSet<int>();
        foreach (var rune in text.EnumerateRunes())
        {
            var category = Rune.GetUnicodeCategory(rune);
            if (!Rune.IsWhiteSpace(rune) && !Rune.IsControl(rune) && category is not (UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark
                or UnicodeCategory.EnclosingMark or UnicodeCategory.Format))
            {
                needed.Add(rune.Value);
            }
        }

        var whole = fonts.Where(f => needed.All(cp => f.Glyph(cp) != 0)).ToList();
        if (whole.Count > 0)
        {
            return whole;
        }

        int missing = needed.FirstOrDefault(cp => fonts.All(f => f.Glyph(cp) == 0), -1);
        if (missing >= 0)
        {
            throw new InvalidDataException($"{where}: no font has U+{missing:X4} ({char.ConvertFromUtf32(missing)}); add a font that has it.");
        }

        return fonts;
    }
}

// An answer: JSON (in a code fence or not) whose page text is one of its keys, or plain text.
internal sealed class SyntheticAnswer
{
    private string _fenceOpen = "", _fenceClose = "";
    private JsonObject? _json;
    private string? _key;
    private bool _indented;
    private int _indentSize = 2;
    private char _indentCharacter = ' ';
    private string _newLine = "\n";              // the answer's own, never the machine's (rule 70)
    private string _comma = ",", _colon = ":";

    public string PageText { get; private set; } = "";

    public IReadOnlyDictionary<string, string> Fields { get; private set; } = new Dictionary<string, string>();

    public static SyntheticAnswer Parse(string answer, string? fullTextKey, string where)
    {
        var parsed = new SyntheticAnswer();
        string body = answer.Trim();
        if (body.StartsWith("```", StringComparison.Ordinal))
        {
            int newline = body.IndexOf('\n');
            int close = body.LastIndexOf("```", StringComparison.Ordinal);
            if (newline > 0 && close > newline)
            {
                parsed._fenceOpen = body[..(newline + 1)];
                parsed._fenceClose = "\n```";
                body = body[(newline + 1)..close].Trim();
            }
        }

        JsonObject? json = null;
        if (body.StartsWith('{'))
        {
            try
            {
                json = JsonNode.Parse(body) as JsonObject;
            }
            catch (JsonException)
            {
                json = null;
            }
        }

        if (json is null)
        {
            if (fullTextKey is not null)
            {
                throw new InvalidDataException($"{where}: the answer is not JSON, so it has no '{fullTextKey}'.");
            }

            parsed._fenceOpen = parsed._fenceClose = "";
            parsed.PageText = answer;
            return parsed;
        }

        var texts = json.Where(p => p.Value is JsonValue v && v.GetValueKind() == JsonValueKind.String).Select(p => (p.Key, Text: p.Value!.GetValue<string>())).ToList();
        string? key = fullTextKey ?? texts.FirstOrDefault(t => t.Key.StartsWith("full_text", StringComparison.OrdinalIgnoreCase)).Key
            ?? texts.OrderByDescending(t => t.Text.Length).FirstOrDefault().Key;
        if (key is null || json[key] is not JsonValue value || value.GetValueKind() != JsonValueKind.String)
        {
            throw new InvalidDataException($"{where}: the answer has no text under '{key ?? fullTextKey ?? "full_text"}' (its keys: {string.Join(", ", json.Select(p => p.Key))}).");
        }

        parsed._json = json;
        parsed._key = key;
        parsed.Layout(body);
        parsed.PageText = value.GetValue<string>();
        parsed.Fields = json.Where(p => p.Value is JsonValue v && v.GetValueKind() is JsonValueKind.String or JsonValueKind.Number)
            .ToDictionary(p => p.Key, p => p.Value is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : p.Value!.ToJsonString(), StringComparer.Ordinal);
        return parsed;
    }

    // How the JSON was laid out: indented (its indent's size and character), or on one line with or without a space
    // after its commas and colons (Python's json.dumps writes ", " and ": "), found outside its strings.
    private void Layout(string json)
    {
        int newline = json.IndexOf('\n');
        _indented = newline >= 0;
        if (_indented)
        {
            _newLine = newline > 0 && json[newline - 1] == '\r' ? "\r\n" : "\n";
            int at = newline + 1, size = 0;
            while (at + size < json.Length && json[at + size] is ' ' or '\t')
            {
                size++;
            }

            if (size > 0)
            {
                _indentSize = Math.Min(size, 127);
                _indentCharacter = json[at];
            }

            return;
        }

        bool inString = false, comma = false, colon = false;
        for (int i = 0; i < json.Length - 1 && !(comma && colon); i++)
        {
            char c = json[i];
            if (inString)
            {
                if (c == '\\')
                {
                    i++;
                }
                else if (c == '"')
                {
                    inString = false;
                }

                continue;
            }

            if (c == '"')
            {
                inString = true;
            }
            else if (c == ',' && !comma)
            {
                comma = true;
                _comma = json[i + 1] == ' ' ? ", " : ",";
            }
            else if (c == ':' && !colon)
            {
                colon = true;
                _colon = json[i + 1] == ' ' ? ": " : ":";
            }
        }
    }

    // The answer with its page text replaced, written as it was (fence, layout, key order).
    public string With(string pageText)
    {
        if (_json is null)
        {
            return pageText;
        }

        var copy = (JsonObject)_json.DeepClone();
        copy[_key!] = pageText;
        string json;
        if (_indented)
        {
            json = SyntheticPages.Json(copy, indented: true, _indentSize, _indentCharacter, _newLine);
        }
        else
        {
            var sb = new StringBuilder();
            OneLine(sb, copy);
            json = sb.ToString();
        }

        return _fenceOpen + json + _fenceClose;
    }

    private void OneLine(StringBuilder sb, JsonNode? node)
    {
        switch (node)
        {
            case JsonObject o:
                sb.Append('{');
                bool first = true;
                foreach (var (key, value) in o)
                {
                    sb.Append(first ? "" : _comma).Append(SyntheticPages.Json(JsonValue.Create(key))).Append(_colon);
                    OneLine(sb, value);
                    first = false;
                }

                sb.Append('}');
                break;
            case JsonArray a:
                sb.Append('[');
                for (int i = 0; i < a.Count; i++)
                {
                    sb.Append(i == 0 ? "" : _comma);
                    OneLine(sb, a[i]);
                }

                sb.Append(']');
                break;
            case null:
                sb.Append("null");
                break;
            default:
                sb.Append(SyntheticPages.Json(node));
                break;
        }
    }
}

// A page part's text with {NAME} and {NAME:ARGUMENT} slots ("{{" and "}}" for braces).
internal static class PagePartTemplate
{
    public static IEnumerable<(string Name, string? Argument)> Slots(string text, string where)
    {
        foreach (var (_, name, argument, _) in Pieces(text, where))
        {
            if (name is not null)
            {
                yield return (name, argument);
            }
        }
    }

    // The text with its slots filled; null when a slot has nothing (the line is left out).
    public static string? Fill(string text, PagePartContext context)
    {
        var sb = new StringBuilder(text.Length + 16);
        foreach (var (literal, name, argument, _) in Pieces(text, "a page part"))
        {
            if (name is null)
            {
                sb.Append(literal);
                continue;
            }

            if (PagePartFillers.Get(name).Fill(argument, context) is not { } value)
            {
                return null;
            }

            sb.Append(value);
        }

        return sb.ToString().Trim();
    }

    private static List<(string Literal, string? Name, string? Argument, int At)> Pieces(string text, string where)
    {
        var pieces = new List<(string, string?, string?, int)>();
        var literal = new StringBuilder();
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if ((c == '{' || c == '}') && i + 1 < text.Length && text[i + 1] == c)
            {
                literal.Append(c);
                i++;
                continue;
            }

            if (c == '}')
            {
                throw new InvalidDataException($"{where}: a '}}' with no '{{' in \"{text}\" (write }}}} for a brace).");
            }

            if (c != '{')
            {
                literal.Append(c);
                continue;
            }

            int close = text.IndexOf('}', i + 1);
            if (close < 0)
            {
                throw new InvalidDataException($"{where}: a '{{' with no '}}' in \"{text}\" (write {{{{ for a brace).");
            }

            if (literal.Length > 0)
            {
                pieces.Add((literal.ToString(), null, null, i));
                literal.Clear();
            }

            string slot = text[(i + 1)..close];
            int colon = slot.IndexOf(':');
            pieces.Add(("", (colon < 0 ? slot : slot[..colon]).Trim(), colon < 0 ? null : slot[(colon + 1)..], i));
            i = close;
        }

        if (literal.Length > 0)
        {
            pieces.Add((literal.ToString(), null, null, text.Length));
        }

        return pieces;
    }
}

// The built-in slots of page parts.
internal static class LibraryPagePartFillers
{
    public static IPagePartFiller[] All =>
    [
        new Filler("digits", "{digits:N}: N random digits (6 unless given; the first not 0)", (a, c) =>
        {
            int n = a is null ? 6 : int.TryParse(a, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v is >= 1 and <= 40 ? v
                : throw new ArgumentException($"{{digits:N}} takes N from 1 to 40, not '{a}'.");
            var sb = new StringBuilder(n);
            sb.Append((char)('1' + c.Random.Next(9)));
            for (int i = 1; i < n; i++)
            {
                sb.Append((char)('0' + c.Random.Next(10)));
            }

            return sb.ToString();
        }),
        new Filler("number", "{number:MIN-MAX}: a whole number from MIN to MAX", (a, c) =>
        {
            int dash = a?.IndexOf('-', 1) ?? -1;
            if (dash < 0 || !long.TryParse(a![..dash], NumberStyles.Integer, CultureInfo.InvariantCulture, out long min)
                || !long.TryParse(a[(dash + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out long max) || min > max)
            {
                throw new ArgumentException($"{{number:MIN-MAX}} takes two whole numbers, the first not above the second, not '{a}'.");
            }

            return c.Random.NextInt64(min, max + 1).ToString(CultureInfo.InvariantCulture);
        }),
        new Filler("pick", "{pick:a|b|c}: one of the choices", (a, c) =>
        {
            var choices = (a ?? "").Split('|');
            return a is null or "" ? throw new ArgumentException("{pick:a|b|c} needs choices.") : choices[c.Random.Next(choices.Length)];
        }),
        new Filler("field", "{field:KEY}: the record answer's field KEY (the line is left out when it has none)", (a, c) =>
            a is null or "" ? throw new ArgumentException("{field:KEY} needs a key.") : c.Fields.TryGetValue(a, out var v) && v.Length > 0 ? v : null),
        new Filler("line", "{line:FILE}: a random line of a UTF-8 text file (relative to the parts file)", (a, c) =>
        {
            if (a is null or "")
            {
                throw new ArgumentException("{line:FILE} needs a file.");
            }

            var lines = Lines.GetOrAdd(Path.GetFullPath(Path.Combine(c.Folder ?? ".", a)), path => File.Exists(path)
                ? [.. File.ReadAllLines(path, Encoding.UTF8).Select(l => l.Trim()).Where(l => l.Length > 0)]
                : throw new FileNotFoundException($"{{line:{a}}}: {path} is not there.", path));
            return lines.Length == 0 ? null : lines[c.Random.Next(lines.Length)];
        }),
        new Filler("date", "{date:FORMAT}: the page's date, Gregorian (FORMAT of yyyy, yy, MM, M, dd, d and others as written; yyyy/MM/dd unless given)", (a, c) =>
            Format(a, c.Date.Year, c.Date.Month, c.Date.Day)),
        new Filler("hijri", "{hijri:FORMAT}: the page's date in the Umm al-Qura calendar (FORMAT as date's)", (a, c) =>
        {
            var calendar = new UmAlQuraCalendar();
            var date = c.Date < calendar.MinSupportedDateTime ? calendar.MinSupportedDateTime : c.Date > calendar.MaxSupportedDateTime ? calendar.MaxSupportedDateTime : c.Date;
            return Format(a, calendar.GetYear(date), calendar.GetMonth(date), calendar.GetDayOfMonth(date));
        }),
    ];

    private static readonly ConcurrentDictionary<string, string[]> Lines = new(StringComparer.Ordinal);

    // yyyy, yy, MM, M, dd and d replaced; everything else as written.
    private static string Format(string? format, int year, int month, int day)
    {
        format = string.IsNullOrEmpty(format) ? "yyyy/MM/dd" : format;
        var sb = new StringBuilder();
        for (int i = 0; i < format.Length;)
        {
            int run = 1;
            while (i + run < format.Length && format[i + run] == format[i])
            {
                run++;
            }

            string piece = format.Substring(i, run);
            sb.Append(piece switch
            {
                "yyyy" => year.ToString("D4", CultureInfo.InvariantCulture),
                "yy" => (year % 100).ToString("D2", CultureInfo.InvariantCulture),
                "MM" => month.ToString("D2", CultureInfo.InvariantCulture),
                "M" => month.ToString(CultureInfo.InvariantCulture),
                "dd" => day.ToString("D2", CultureInfo.InvariantCulture),
                "d" => day.ToString(CultureInfo.InvariantCulture),
                _ => piece,
            });
            i += run;
        }

        return sb.ToString();
    }

    private sealed class Filler(string name, string summary, Func<string?, PagePartContext, string?> fill) : IPagePartFiller
    {
        public string Name => name;

        public string Summary => summary;

        public string? Fill(string? argument, PagePartContext context) => fill(argument, context);
    }
}

// ShareGPT records: the last "gpt" turn is the answer.
internal sealed class ShareGptSyntheticRecords : ISyntheticRecordFormat
{
    public string Name => "sharegpt";

    public string Summary => "LlamaFactory's {\"conversations\": [{\"from\", \"value\"}], \"images\"} (JSON Lines or an array); the last gpt turn is the answer";

    public bool Recognizes(string path) => SyntheticRecordFiles.First(path) is { } first && first["conversations"] is JsonArray;

    public IEnumerable<ISyntheticRecord> Read(string path, string prompt, string? answerKey)
    {
        foreach (var (record, where) in TuningRecords.Read(path))
        {
            var turns = record["conversations"] as JsonArray ?? throw new InvalidDataException($"{where}: a ShareGPT record needs \"conversations\".");
            int last = SyntheticRecordFiles.LastIndex(turns, t => (string?)t?["from"] is "gpt" or "assistant");
            if (last < 0 || turns[last]!["value"] is not JsonValue v || v.GetValueKind() != JsonValueKind.String)
            {
                throw new InvalidDataException($"{where}: the record has no gpt turn with text to draw.");
            }

            yield return new SyntheticRecord
            {
                Answer = v.GetValue<string>(), Where = where,
                Writer = (answer, image) =>
                {
                    var copy = (JsonObject)record.DeepClone();
                    var copyTurns = (JsonArray)copy["conversations"]!;
                    copyTurns[last]!["value"] = answer;
                    SyntheticRecordFiles.OneImage(copyTurns, "from", "human", "value");
                    copy["images"] = new JsonArray(image);
                    return copy;
                },
            };
        }
    }
}

// Chat messages records: the last assistant message is the answer.
internal sealed class MessagesSyntheticRecords : ISyntheticRecordFormat
{
    public string Name => "messages";

    public string Summary => "{\"messages\": [{\"role\", \"content\"}], \"images\"} (JSON Lines or an array); the last assistant message is the answer";

    public bool Recognizes(string path) => SyntheticRecordFiles.First(path) is { } first && first["messages"] is JsonArray;

    public IEnumerable<ISyntheticRecord> Read(string path, string prompt, string? answerKey)
    {
        foreach (var (record, where) in TuningRecords.Read(path))
        {
            var messages = record["messages"] as JsonArray ?? throw new InvalidDataException($"{where}: a messages record needs \"messages\".");
            int last = SyntheticRecordFiles.LastIndex(messages, m => (string?)m?["role"] == "assistant");
            string? answer = last < 0 ? null : messages[last]!["content"] switch
            {
                JsonValue v when v.GetValueKind() == JsonValueKind.String => v.GetValue<string>(),
                JsonArray parts => string.Concat(parts.Where(p => (string?)p?["type"] == "text").Select(p => (string?)p!["text"])),
                _ => null,
            };
            if (answer is null)
            {
                throw new InvalidDataException($"{where}: the record has no assistant message with text to draw.");
            }

            yield return new SyntheticRecord
            {
                Answer = answer, Where = where,
                Writer = (text, image) =>
                {
                    var copy = (JsonObject)record.DeepClone();
                    var copyMessages = (JsonArray)copy["messages"]!;
                    copyMessages[last]!["content"] = text;

                    // Image parts and <image> markers out, then one bare image part (or marker) in the first user message.
                    foreach (var message in copyMessages)
                    {
                        if (message?["content"] is JsonArray parts)
                        {
                            for (int i = parts.Count - 1; i >= 0; i--)
                            {
                                if ((string?)parts[i]?["type"] is "image" or "image_url")
                                {
                                    parts.RemoveAt(i);
                                }
                            }
                        }
                    }

                    SyntheticRecordFiles.OneImage(copyMessages, "role", "user", "content");
                    copy["images"] = new JsonArray(image);
                    return copy;
                },
            };
        }
    }
}

// Plain text: a .txt of documents between lines of "---", or a folder of .txt files; written as ShareGPT.
internal sealed class TextSyntheticRecords : ISyntheticRecordFormat
{
    public string Name => "text";

    public string Summary => "plain text: a .txt file of documents separated by lines of ---, or a folder of .txt files (one document each); written as ShareGPT with the prompt";

    public bool Recognizes(string path) =>
        Directory.Exists(path) ? Directory.EnumerateFiles(path, "*.txt").Any() : File.Exists(path) && Path.GetExtension(path).Equals(".txt", StringComparison.OrdinalIgnoreCase);

    public IEnumerable<ISyntheticRecord> Read(string path, string prompt, string? answerKey)
    {
        IEnumerable<(string Text, string Where)> documents = Directory.Exists(path)
            ? Directory.EnumerateFiles(path, "*.txt").Order(StringComparer.Ordinal).Select(f => (File.ReadAllText(f, Encoding.UTF8), Path.GetFileName(f)))
            : Split(path);
        foreach (var (text, where) in documents)
        {
            string trimmed = text.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }

            string answer = answerKey is null ? trimmed : SyntheticPages.Json(new JsonObject { [answerKey] = trimmed });
            yield return new SyntheticRecord
            {
                Answer = answer, Where = where,
                Writer = (written, image) => new JsonObject
                {
                    ["conversations"] = new JsonArray(
                        new JsonObject { ["from"] = "human", ["value"] = "<image>" + prompt },
                        new JsonObject { ["from"] = "gpt", ["value"] = written }),
                    ["images"] = new JsonArray(image),
                },
            };
        }
    }

    private static IEnumerable<(string, string)> Split(string path)
    {
        var current = new StringBuilder();
        int document = 1, line = 0, start = 1;
        foreach (string text in File.ReadLines(path, Encoding.UTF8))
        {
            line++;
            if (text.Trim() == "---")
            {
                yield return (current.ToString(), $"{Path.GetFileName(path)} document {document++} (line {start})");
                current.Clear();
                start = line + 1;
                continue;
            }

            current.Append(text).Append('\n');
        }

        yield return (current.ToString(), $"{Path.GetFileName(path)} document {document} (line {start})");
    }
}

internal static class SyntheticRecordFiles
{
    // The file's first record, or null when it is not a JSON records file.
    public static JsonObject? First(string path)
    {
        if (!File.Exists(path) || Path.GetExtension(path).ToLowerInvariant() is not (".json" or ".jsonl"))
        {
            return null;
        }

        try
        {
            return TuningRecords.Read(path).Select(r => r.Record).FirstOrDefault();
        }
        catch (Exception e) when (e is InvalidDataException or JsonException or IOException)
        {
            return null;
        }
    }

    public static int LastIndex(JsonArray items, Func<JsonNode?, bool> match)
    {
        for (int i = items.Count - 1; i >= 0; i--)
        {
            if (match(items[i]))
            {
                return i;
            }
        }

        return -1;
    }

    // Every "<image>" marker out of the turns' texts, then one at the start of the first turn of `role`.
    public static void OneImage(JsonArray turns, string roleKey, string role, string textKey)
    {
        const string Marker = "<image>";
        int first = -1;
        for (int i = 0; i < turns.Count; i++)
        {
            if (turns[i] is not JsonObject turn)
            {
                continue;
            }

            if (first < 0 && (string?)turn[roleKey] == role)
            {
                first = i;
            }

            if (turn[textKey] is JsonValue v && v.GetValueKind() == JsonValueKind.String)
            {
                turn[textKey] = v.GetValue<string>().Replace(Marker, "", StringComparison.Ordinal);
            }
            else if (turn[textKey] is JsonArray parts)
            {
                foreach (var part in parts)
                {
                    if (part?["text"] is JsonValue t && t.GetValueKind() == JsonValueKind.String)
                    {
                        part["text"] = t.GetValue<string>().Replace(Marker, "", StringComparison.Ordinal);
                    }
                }
            }
        }

        if (first < 0)
        {
            turns.Insert(0, new JsonObject { [roleKey] = role, [textKey] = Marker });
            return;
        }

        var target = (JsonObject)turns[first]!;
        if (target[textKey] is JsonArray content)
        {
            content.Insert(0, new JsonObject { ["type"] = "image" });
        }
        else
        {
            target[textKey] = Marker + ((string?)target[textKey] ?? "");
        }
    }
}
