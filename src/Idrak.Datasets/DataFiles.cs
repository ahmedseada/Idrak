using System.Formats.Tar;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Datasets;

/// <summary>File formats <see cref="Dataset"/> reads.</summary>
public enum DataFormat
{
    /// <summary>One JSON object per line (.jsonl, .ndjson).</summary>
    JsonLines,

    /// <summary>A JSON document (.json): an array of objects, an object holding such an array, or columns of equal length.</summary>
    Json,

    /// <summary>Comma-separated values with a header row (.csv).</summary>
    Csv,

    /// <summary>Tab-separated values with a header row (.tsv).</summary>
    Tsv,

    /// <summary>Apache Parquet (.parquet), the format of most Hugging Face datasets.</summary>
    Parquet,

    /// <summary>Plain text (.txt, .md): see <see cref="ReadOptions.Text"/>.</summary>
    Text,

    /// <summary>A source file, as one row {"text", "path", "language"}.</summary>
    Code,
}

/// <summary>How text files become rows.</summary>
public enum TextRows
{
    /// <summary>Each non-empty line is a row {"text"} (as Hugging Face's text loader).</summary>
    Lines,

    /// <summary>Each block of text between blank lines is a row.</summary>
    Paragraphs,

    /// <summary>The whole file is one row {"text", "path"}.</summary>
    Document,
}

/// <summary>How <see cref="Dataset"/> reads files.</summary>
public sealed record ReadOptions
{
    /// <summary>The defaults.</summary>
    public static ReadOptions Default { get; } = new();

    /// <summary>The format, instead of choosing it by extension.</summary>
    public DataFormat? Format { get; init; }

    /// <summary>Text files: a row per line (default), paragraph or file.</summary>
    public TextRows Text { get; init; } = TextRows.Lines;

    /// <summary>CSV / TSV: turn numbers and true/false into JSON numbers and booleans, empty cells into null.</summary>
    public bool InferTypes { get; init; } = true;

    /// <summary>JSON documents: the property holding the rows (default: the only array of objects, if there is one).</summary>
    public string? JsonProperty { get; init; }

    /// <summary>Only files (and archive entries) whose path matches this glob, e.g. <c>*.cs</c> or <c>data/train-*.parquet</c>.</summary>
    public string? Pattern { get; init; }

    /// <summary>Add a "_file" column with the path each row came from.</summary>
    public bool IncludeFile { get; init; }

    /// <summary>Also read source code files (as <see cref="DataFormat.Code"/>) in folders and archives.</summary>
    public bool IncludeCode { get; init; }

    /// <summary>
    /// Every text file (data files such as .json and .csv included) is one row {"text", "path", "language"}: a repository
    /// or folder read as documents rather than as data. Implies <see cref="IncludeCode"/>.
    /// </summary>
    public bool Documents { get; init; }

    /// <summary>Largest source or text file read as one document, in bytes (larger ones, often generated, are skipped).</summary>
    public long MaxDocumentBytes { get; init; } = 1 << 20;

    /// <summary>The folder paths are shown relative to (set by <see cref="Dataset.FromFolder"/>).</summary>
    public string? Root { get; init; }
}

/// <summary>Reads data files into rows.</summary>
public static class DataFiles
{
    private static readonly Dictionary<string, string> CodeLanguages = new(StringComparer.OrdinalIgnoreCase)
    {
        [".cs"] = "csharp", [".csx"] = "csharp", [".razor"] = "razor", [".cshtml"] = "razor", [".xaml"] = "xml", [".csproj"] = "xml", [".props"] = "xml",
        [".ts"] = "typescript", [".tsx"] = "typescript", [".mts"] = "typescript", [".cts"] = "typescript",
        [".js"] = "javascript", [".jsx"] = "javascript", [".mjs"] = "javascript", [".cjs"] = "javascript",
        [".html"] = "html", [".htm"] = "html", [".css"] = "css", [".scss"] = "scss", [".sass"] = "sass", [".less"] = "less",
        [".py"] = "python", [".java"] = "java", [".kt"] = "kotlin", [".go"] = "go", [".rs"] = "rust", [".rb"] = "ruby", [".php"] = "php",
        [".c"] = "c", [".h"] = "c", [".cpp"] = "cpp", [".cc"] = "cpp", [".hpp"] = "cpp", [".swift"] = "swift", [".scala"] = "scala",
        [".fs"] = "fsharp", [".vb"] = "vb", [".sql"] = "sql", [".sh"] = "shell", [".ps1"] = "powershell", [".yml"] = "yaml", [".yaml"] = "yaml",
        [".xml"] = "xml", [".toml"] = "toml", [".dockerfile"] = "dockerfile", [".vue"] = "vue", [".svelte"] = "svelte", [".dart"] = "dart", [".lua"] = "lua",
    };

    private static readonly HashSet<string> SkippedFolders = new(["bin", "obj", "node_modules", ".git", ".vs", ".idea", "dist", ".angular", "__pycache__", ".venv"],
        StringComparer.OrdinalIgnoreCase);

    /// <summary>The format of <paramref name="path"/> by its extension (.gz stripped), or null for an unknown one.</summary>
    public static DataFormat? FormatOf(string path, bool includeCode = true)
    {
        string name = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? path[..^3] : path;
        string extension = Path.GetExtension(name).ToLowerInvariant();
        return extension switch
        {
            ".jsonl" or ".ndjson" => DataFormat.JsonLines,
            ".json" => DataFormat.Json,
            ".csv" => DataFormat.Csv,
            ".tsv" => DataFormat.Tsv,
            ".parquet" => DataFormat.Parquet,
            ".txt" or ".text" or ".md" or ".markdown" or ".rst" => DataFormat.Text,
            _ when includeCode && CodeLanguages.ContainsKey(extension) => DataFormat.Code,
            _ => null,
        };
    }

    /// <summary>Whether <paramref name="path"/> is an archive whose entries are read (.zip, .tar, .tar.gz, .tgz).</summary>
    public static bool IsArchive(string path) =>
        path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase);

    /// <summary>The rows of one file (or of the data files inside an archive).</summary>
    public static IEnumerable<JsonObject> Read(string path, ReadOptions options)
    {
        string shown = options.Root is { } root ? Path.GetRelativePath(root, path).Replace('\\', '/') : Path.GetFileName(path);
        if (IsArchive(path) && options.Format is null)
        {
            return ReadArchive(path, options);
        }

        var format = options.Format ?? DocumentFormat(FormatOf(path, includeCode: true), options)
                     ?? throw new NotSupportedException($"'{path}': unknown data format (set ReadOptions.Format).");
        return ReadStream(() => Open(path), shown, format, options);
    }

    /// <summary>The rows of a stream holding a file of <paramref name="format"/> (<paramref name="path"/> is shown in rows and errors).</summary>
    public static IEnumerable<JsonObject> ReadStream(Func<Stream> open, string path, DataFormat format, ReadOptions options)
    {
        var rows = format switch
        {
            DataFormat.JsonLines => JsonLines(open, path),
            DataFormat.Json => Json(open, path, options),
            DataFormat.Csv => Delimited(open, ',', options),
            DataFormat.Tsv => Delimited(open, '\t', options),
            DataFormat.Parquet => ParquetRows(open),
            DataFormat.Text => Text(open, path, options),
            DataFormat.Code => Code(open, path, options),
            _ => throw new NotSupportedException(format.ToString()),
        };
        return options.IncludeFile ? rows.Select(r => { r["_file"] = path; return r; }) : rows;
    }

    private static DataFormat? DocumentFormat(DataFormat? format, ReadOptions options) =>
        options.Documents && format is not null and not DataFormat.Parquet ? DataFormat.Code : format;

    internal static IEnumerable<string> InFolder(string root, string? pattern, ReadOptions options)
    {
        var glob = pattern ?? options.Pattern;
        var match = glob is null ? null : Glob(glob);
        var pending = new Stack<string>();
        pending.Push(root);
        var files = new List<string>();
        while (pending.Count > 0)
        {
            string folder = pending.Pop();
            foreach (var sub in Directory.EnumerateDirectories(folder))
            {
                if (!SkippedFolders.Contains(Path.GetFileName(sub)))
                {
                    pending.Push(sub);
                }
            }

            foreach (var file in Directory.EnumerateFiles(folder))
            {
                string relative = Path.GetRelativePath(root, file).Replace('\\', '/');
                bool known = options.Format is not null || IsArchive(file) || FormatOf(file, options.IncludeCode || options.Documents) is not null;
                if (known && (match is null || match.IsMatch(relative) || match.IsMatch(Path.GetFileName(file))))
                {
                    files.Add(file);
                }
            }
        }

        files.Sort(StringComparer.Ordinal);
        return files;
    }

    /// <summary>A regular expression for a glob: <c>*</c> within a path segment, <c>**</c> across segments, <c>?</c> one character.</summary>
    public static Regex Glob(string glob)
    {
        var sb = new StringBuilder("^");
        for (int i = 0; i < glob.Length; i++)
        {
            char c = glob[i];
            if (c == '*' && i + 1 < glob.Length && glob[i + 1] == '*')
            {
                sb.Append(".*");
                i++;
                if (i + 1 < glob.Length && glob[i + 1] == '/')
                {
                    i++;
                    sb.Append("/?");
                }
            }
            else
            {
                sb.Append(c switch { '*' => "[^/]*", '?' => "[^/]", _ => Regex.Escape(c.ToString()) });
            }
        }

        return new Regex(sb.Append('$').ToString(), RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    private static Stream Open(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        return path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(stream, CompressionMode.Decompress) : stream;
    }

    private static IEnumerable<JsonObject> ReadArchive(string path, ReadOptions options)
    {
        var match = options.Pattern is null ? null : Glob(options.Pattern);
        bool Wanted(string name)
        {
            if (match is not null && !match.IsMatch(name) && !match.IsMatch(Path.GetFileName(name)) || name.Split('/').Any(SkippedFolders.Contains))
            {
                return false;
            }

            var format = DocumentFormat(FormatOf(name, options.IncludeCode || options.Documents), options);
            return format is not null && !(options.Documents && format == DataFormat.Parquet);
        }

        if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            using var zip = ZipFile.OpenRead(path);
            foreach (var entry in zip.Entries.Where(e => e.Length > 0 && Wanted(e.FullName)).OrderBy(e => e.FullName, StringComparer.Ordinal))
            {
                foreach (var row in ReadStream(() => Buffered(entry.Open(), entry.Length, entry.FullName), entry.FullName, DocumentFormat(FormatOf(entry.FullName), options)!.Value, options))
                {
                    yield return row;
                }
            }

            yield break;
        }

        using var file = File.OpenRead(path);
        using Stream tarStream = path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase) ? file : new GZipStream(file, CompressionMode.Decompress);
        using var tar = new TarReader(tarStream);
        while (tar.GetNextEntry() is { } entry)
        {
            // GitHub's tarballs put everything under "<owner>-<repo>-<sha>/": strip that top folder from the shown path.
            string name = entry.Name.Replace('\\', '/');
            if (entry.EntryType is not (TarEntryType.RegularFile or TarEntryType.V7RegularFile) || entry.DataStream is null || !Wanted(name))
            {
                continue;
            }

            var bytes = new MemoryStream();
            entry.DataStream.CopyTo(bytes);
            var data = bytes.ToArray();
            foreach (var row in ReadStream(() => Decompressed(new MemoryStream(data), name), name, DocumentFormat(FormatOf(name), options)!.Value, options))
            {
                yield return row;
            }
        }
    }

    private static Stream Decompressed(Stream stream, string name) =>
        name.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(stream, CompressionMode.Decompress) : stream;

    // Zip entries are not seekable; Parquet needs seeking, so entries are read into memory.
    private static Stream Buffered(Stream entry, long length, string name)
    {
        var memory = new MemoryStream(length > 0 && length < int.MaxValue ? (int)length : 0);
        using (entry)
        {
            entry.CopyTo(memory);
        }

        memory.Position = 0;
        return Decompressed(memory, name);
    }

    private static IEnumerable<JsonObject> JsonLines(Func<Stream> open, string path)
    {
        using var reader = new StreamReader(open(), Encoding.UTF8, true, 1 << 16);
        int number = 0;
        while (reader.ReadLine() is { } line)
        {
            number++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            JsonNode? node;
            try
            {
                node = JsonNode.Parse(line);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
            }

            yield return node as JsonObject ?? new JsonObject { ["value"] = node };
        }
    }

    private static IEnumerable<JsonObject> Json(Func<Stream> open, string path, ReadOptions options)
    {
        JsonNode? document;
        using (var stream = open())
        {
            try
            {
                document = JsonNode.Parse(stream, documentOptions: new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
            }
            catch (JsonException)
            {
                document = null;
            }
        }

        if (document is null)
        {
            // Often a .json file is JSON Lines.
            foreach (var row in JsonLines(open, path))
            {
                yield return row;
            }

            yield break;
        }

        foreach (var row in RowsOf(document, options.JsonProperty, path))
        {
            yield return row;
        }
    }

    private static IEnumerable<JsonObject> RowsOf(JsonNode document, string? property, string path)
    {
        if (property is not null)
        {
            document = document[property] ?? throw new InvalidDataException($"{path} has no property '{property}'.");
        }

        switch (document)
        {
            case JsonArray array:
                // Detach all rows at once: removing them one by one from the front shifts the whole array each time (O(n²)).
                var items = array.ToArray();
                array.Clear();
                for (int i = 0; i < items.Length; i++)
                {
                    var item = items[i];
                    items[i] = null;   // a row the caller is done with can be collected
                    yield return item as JsonObject ?? new JsonObject { ["value"] = item };
                }

                break;
            case JsonObject o when o.Count > 0 && o.All(p => p.Value is JsonArray a && a.Count == ((JsonArray)o.First().Value!).Count)
                                   && o.Count(p => ((JsonArray)p.Value!).All(x => x is JsonObject)) is var objectColumns:
                if (o.Count == 1 || objectColumns == 1 && property is null && o.Count(p => ((JsonArray)p.Value!).Count > 0) == 1)
                {
                    // {"data": [ {...}, ... ]}: the rows are in the one array.
                    var inner = o.First(p => ((JsonArray)p.Value!).All(x => x is JsonObject)).Value!;
                    foreach (var row in RowsOf(inner, null, path))
                    {
                        yield return row;
                    }

                    break;
                }

                // Columns: {"a": [1, 2], "b": ["x", "y"]} → {"a": 1, "b": "x"}, {"a": 2, "b": "y"}.
                int length = ((JsonArray)o.First().Value!).Count;
                for (int i = 0; i < length; i++)
                {
                    var row = new JsonObject();
                    foreach (var (name, column) in o)
                    {
                        row[name] = ((JsonArray)column!)[i]?.DeepClone();
                    }

                    yield return row;
                }

                break;
            case JsonObject o when property is null && o.Where(p => p.Value is JsonArray a && a.Count > 0 && a.All(x => x is JsonObject)).ToList() is [var only]:
                foreach (var row in RowsOf(only.Value!, null, path))
                {
                    yield return row;
                }

                break;
            case JsonObject o:
                yield return o;
                break;
            default:
                yield return new JsonObject { ["value"] = document.DeepClone() };
                break;
        }
    }

    private static IEnumerable<JsonObject> Delimited(Func<Stream> open, char delimiter, ReadOptions options)
    {
        using var reader = new StreamReader(open(), Encoding.UTF8, true, 1 << 16);
        string[]? header = null;
        foreach (var record in Records(reader, delimiter))
        {
            if (header is null)
            {
                header = [.. record.Select((h, i) => h.Length > 0 ? h : $"column{i + 1}")];
                continue;
            }

            if (record.Count == 1 && record[0].Length == 0)
            {
                continue;                                           // blank line
            }

            var row = new JsonObject();
            for (int i = 0; i < header.Length; i++)
            {
                string cell = i < record.Count ? record[i] : "";
                row[header[i]] = options.InferTypes ? Infer(cell) : cell;
            }

            yield return row;
        }
    }

    // RFC 4180 records: quoted fields may hold delimiters, quotes ("") and line breaks.
    private static IEnumerable<List<string>> Records(TextReader reader, char delimiter)
    {
        var fields = new List<string>();
        var field = new StringBuilder();
        bool quoted = false, any = false;
        int c;
        while ((c = reader.Read()) >= 0)
        {
            char ch = (char)c;
            any = true;
            if (quoted)
            {
                if (ch == '"')
                {
                    if (reader.Peek() == '"')
                    {
                        field.Append('"');
                        reader.Read();
                    }
                    else
                    {
                        quoted = false;
                    }
                }
                else
                {
                    field.Append(ch);
                }
            }
            else if (ch == '"' && field.Length == 0)
            {
                quoted = true;
            }
            else if (ch == delimiter)
            {
                fields.Add(field.ToString());
                field.Clear();
            }
            else if (ch is '\n' or '\r')
            {
                if (ch == '\r' && reader.Peek() == '\n')
                {
                    reader.Read();
                }

                fields.Add(field.ToString());
                field.Clear();
                yield return fields;
                fields = [];
                any = false;
            }
            else
            {
                field.Append(ch);
            }
        }

        if (any)
        {
            fields.Add(field.ToString());
            yield return fields;
        }
    }

    private static JsonNode? Infer(string cell)
    {
        if (cell.Length == 0)
        {
            return null;
        }

        // Codes such as zip codes and ids keep their leading zeros as text.
        string digits = cell.TrimStart('-', '+');
        if (digits.Length > 1 && digits[0] == '0' && char.IsAsciiDigit(digits[1]))
        {
            return cell;
        }

        if (long.TryParse(cell, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long whole))
        {
            return whole;
        }

        if (double.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) && char.IsAsciiDigit(cell[^1]))
        {
            return number;
        }

        return cell switch
        {
            "true" or "True" or "TRUE" => true,
            "false" or "False" or "FALSE" => false,
            _ => cell,
        };
    }

    private static IEnumerable<JsonObject> Text(Func<Stream> open, string path, ReadOptions options)
    {
        using var reader = new StreamReader(open(), Encoding.UTF8, true, 1 << 16);
        switch (options.Text)
        {
            case TextRows.Lines:
                while (reader.ReadLine() is { } line)
                {
                    if (line.Length > 0)
                    {
                        yield return new JsonObject { ["text"] = line };
                    }
                }

                break;
            case TextRows.Paragraphs:
                var paragraph = new StringBuilder();
                while (true)
                {
                    string? line = reader.ReadLine();
                    if (line is null || line.Trim().Length == 0)
                    {
                        if (paragraph.Length > 0)
                        {
                            yield return new JsonObject { ["text"] = paragraph.ToString().TrimEnd('\n') };
                            paragraph.Clear();
                        }

                        if (line is null)
                        {
                            break;
                        }

                        continue;
                    }

                    paragraph.Append(line).Append('\n');
                }

                break;
            default:
                yield return new JsonObject { ["text"] = reader.ReadToEnd(), ["path"] = path };
                break;
        }
    }

    private static IEnumerable<JsonObject> Code(Func<Stream> open, string path, ReadOptions options)
    {
        using var stream = open();
        var bytes = new MemoryStream();
        var buffer = new byte[1 << 16];
        int read;
        while ((read = stream.Read(buffer)) > 0)
        {
            bytes.Write(buffer, 0, read);
            if (bytes.Length > options.MaxDocumentBytes)
            {
                yield break;                                        // too large: generated or data, not code
            }
        }

        var data = bytes.GetBuffer().AsSpan(0, (int)bytes.Length);
        if (data[..Math.Min(data.Length, 8000)].Contains((byte)0))
        {
            yield break;                                            // binary
        }

        string text = Encoding.UTF8.GetString(data.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]) ? data[3..] : data);
        yield return new JsonObject
        {
            ["text"] = text,
            ["path"] = path,
            ["language"] = CodeLanguages.TryGetValue(Path.GetExtension(path), out var language) ? language : "text",
        };
    }

    private static IEnumerable<JsonObject> ParquetRows(Func<Stream> open)
    {
        using var stream = open();
        if (stream.CanSeek)
        {
            foreach (var row in ParquetFile.ReadRows(stream))
            {
                yield return row;
            }

            yield break;
        }

        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        memory.Position = 0;
        foreach (var row in ParquetFile.ReadRows(memory))
        {
            yield return row;
        }
    }
}
