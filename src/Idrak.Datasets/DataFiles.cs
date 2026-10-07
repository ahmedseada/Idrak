// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Formats.Tar;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Datasets;

/// <summary>Reads data files into rows.</summary>
public static class DataFiles
{
    internal static readonly Dictionary<string, string> CodeLanguages = new(StringComparer.OrdinalIgnoreCase)
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

    /// <summary>
    /// The format of <paramref name="path"/> by its extension (.gz stripped), or null for an unknown one. Formats added with
    /// <see cref="DataFileFormats.Register"/> have no <see cref="DataFormat"/> (null here): see <see cref="DataFileFormats.Find"/>.
    /// </summary>
    public static DataFormat? FormatOf(string path, bool includeCode = true) =>
        DataFileFormats.Find(path, includeCode) is { } format && Enum.TryParse<DataFormat>(format.Name, ignoreCase: true, out var builtIn) ? builtIn : null;

    /// <summary>Whether <paramref name="path"/> is an archive whose entries are read (.zip, .tar, .tar.gz, .tgz).</summary>
    public static bool IsArchive(string path) =>
        path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase);

    /// <summary>The rows of one file (or of the data files inside an archive).</summary>
    public static IEnumerable<JsonObject> Read(string path, ReadOptions options)
    {
        string shown = options.Root is { } root ? Path.GetRelativePath(root, path).Replace('\\', '/') : Path.GetFileName(path);
        if (IsArchive(path) && options.Format is null && options.FileFormat is null)
        {
            return ReadArchive(path, options);
        }

        var format = Chosen(options) ?? DocumentFormat(DataFileFormats.Find(path, includeCode: true), options)
                     ?? throw new NotSupportedException($"'{path}': unknown data format (set ReadOptions.Format).");
        return ReadStream(() => Open(path), shown, format, options);
    }

    /// <summary>The rows of a stream holding a file of <paramref name="format"/> (<paramref name="path"/> is shown in rows and errors).</summary>
    public static IEnumerable<JsonObject> ReadStream(Func<Stream> open, string path, DataFormat format, ReadOptions options) =>
        ReadStream(open, path, DataFileFormats.Get(format.ToString()), options);

    /// <summary>The rows of a stream holding a file of <paramref name="format"/> (<paramref name="path"/> is shown in rows and errors).</summary>
    public static IEnumerable<JsonObject> ReadStream(Func<Stream> open, string path, IDataFileFormat format, ReadOptions options)
    {
        var rows = format.Read(open, path, options);
        return options.IncludeFile ? rows.Select(r => { r["_file"] = path; return r; }) : rows;
    }

    // The format the options set, if any (a format object first, then a built-in by its enum value).
    private static IDataFileFormat? Chosen(ReadOptions options) =>
        options.FileFormat ?? (options.Format is { } format ? DataFileFormats.Get(format.ToString()) : null);

    // Documents: every format but Parquet is read as source code (one row per file).
    private static IDataFileFormat? DocumentFormat(IDataFileFormat? format, ReadOptions options) =>
        options.Documents && format is not null && !Is(format, DataFormat.Parquet) ? DataFileFormats.Get(nameof(DataFormat.Code)) : format;

    // Whether the format has the name of a built-in (it may be a replacement registered under that name).
    private static bool Is(IDataFileFormat format, DataFormat builtIn) =>
        string.Equals(format.Name, builtIn.ToString(), StringComparison.OrdinalIgnoreCase);

    // The built-in formats, registered in DataFileFormats by LibraryRegistrations (once, before it is first used).
    internal static void RegisterAll()
    {
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.JsonLines), [".jsonl", ".ndjson"], (open, path, _) => JsonLines(open, path)));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Json), [".json"], Json));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Csv), [".csv"], (open, _, options) => Delimited(open, ',', options)));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Tsv), [".tsv"], (open, _, options) => Delimited(open, '\t', options)));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Parquet), [".parquet"], (open, _, _) => ParquetRows(open)));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Text), [".txt", ".text", ".md", ".markdown", ".rst"], Text));
        DataFileFormats.Register(new BuiltIn(nameof(DataFormat.Code), [.. CodeLanguages.Keys], Code));
    }

    private sealed class BuiltIn(string name, string[] extensions, Func<Func<Stream>, string, ReadOptions, IEnumerable<JsonObject>> read) : IDataFileFormat
    {
        public string Name => name;

        public IReadOnlyCollection<string> Extensions => extensions;

        public IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options) => read(open, path, options);
    }

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
                bool known = options.Format is not null || options.FileFormat is not null || IsArchive(file)
                             || DataFileFormats.Find(file, options.IncludeCode || options.Documents) is not null;
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

            var format = DocumentFormat(DataFileFormats.Find(name, options.IncludeCode || options.Documents), options);
            return format is not null && !(options.Documents && Is(format, DataFormat.Parquet));
        }

        if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            using var zip = ZipFile.OpenRead(path);
            foreach (var entry in zip.Entries.Where(e => e.Length > 0 && Wanted(e.FullName)).OrderBy(e => e.FullName, StringComparer.Ordinal))
            {
                foreach (var row in ReadStream(() => Buffered(entry.Open(), entry.Length, entry.FullName), entry.FullName, DocumentFormat(DataFileFormats.Find(entry.FullName), options)!, options))
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
            foreach (var row in ReadStream(() => Decompressed(new MemoryStream(data), name), name, DocumentFormat(DataFileFormats.Find(name), options)!, options))
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

    // Lines are split and parsed as UTF-8 bytes, not decoded to text and encoded back. Lines end at \n, \r or \r\n (as ReadLine).
    internal static IEnumerable<JsonObject> JsonLines(Func<Stream> open, string path)
    {
        var stream = open();
        try
        {
            var buffer = new byte[1 << 16];
            int start = 0, length = 0, number = 0;
            bool end = false;
            while (length < 4 && !end)
            {
                int read = stream.Read(buffer, length, buffer.Length - length);
                length += read;
                end = read == 0;
            }

            var head = buffer.AsSpan(0, length);
            if (head.StartsWith((ReadOnlySpan<byte>)[0xFE, 0xFF]) || head.StartsWith((ReadOnlySpan<byte>)[0xFF, 0xFE]) || head.StartsWith((ReadOnlySpan<byte>)[0, 0, 0xFE, 0xFF]))
            {
                // A UTF-16 or UTF-32 byte order mark: read as text, as StreamReader detects it.
                stream = new PrefixedStream(buffer[..length], stream);
                foreach (var row in JsonLinesText(stream, path))
                {
                    yield return row;
                }

                yield break;
            }

            if (head.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
            {
                start = 3;
            }

            while (true)
            {
                int found = buffer.AsSpan(start, length - start).IndexOfAny((byte)'\n', (byte)'\r');
                if (found < 0 && !end)
                {
                    // The line goes on past the buffer: move it to the front (growing the buffer if it fills it) and read more.
                    if (start == 0 && length == buffer.Length)
                    {
                        Array.Resize(ref buffer, buffer.Length * 2);
                    }
                    else if (start > 0)
                    {
                        buffer.AsSpan(start, length - start).CopyTo(buffer);
                        length -= start;
                        start = 0;
                    }

                    int read = stream.Read(buffer, length, buffer.Length - length);
                    length += read;
                    end = read == 0;
                    continue;
                }

                int lineEnd = found < 0 ? length : start + found;
                if (found < 0 && lineEnd == start)
                {
                    break;
                }

                // A \r at the end of the data read so far may be the start of \r\n: read on before deciding.
                if (found >= 0 && buffer[lineEnd] == '\r' && lineEnd + 1 == length && !end)
                {
                    if (start > 0)
                    {
                        buffer.AsSpan(start, length - start).CopyTo(buffer);
                        length -= start;
                        start = 0;
                    }
                    else if (length == buffer.Length)
                    {
                        Array.Resize(ref buffer, buffer.Length * 2);
                    }

                    int read = stream.Read(buffer, length, buffer.Length - length);
                    length += read;
                    end = read == 0;
                    continue;
                }

                number++;
                var node = ParseLine(buffer, start, lineEnd - start, path, number, out bool blank);
                start = found < 0 ? length : lineEnd + 1;
                if (found >= 0 && buffer[lineEnd] == '\r' && start < length && buffer[start] == '\n')
                {
                    start++;
                }

                if (!blank)
                {
                    yield return node as JsonObject ?? new JsonObject { ["value"] = node };
                }
            }
        }
        finally
        {
            stream.Dispose();
        }
    }

    private static readonly SearchValues<byte> AsciiSpaces = SearchValues.Create(" \t\v\f"u8);

    // A line of JSON Lines; blank (all white space, as string.IsNullOrWhiteSpace) lines are skipped. Invalid UTF-8 is decoded
    // to text first, its bad bytes replaced as StreamReader replaces them.
    private static JsonNode? ParseLine(byte[] buffer, int start, int length, string path, int number, out bool blank)
    {
        var line = buffer.AsSpan(start, length);
        int text = line.IndexOfAnyExcept(AsciiSpaces);
        blank = text < 0;
        if (blank)
        {
            return null;
        }

        string? decoded = null;
        if (line[text] >= 0x80 || !System.Text.Unicode.Utf8.IsValid(line))
        {
            decoded = Encoding.UTF8.GetString(line);
            if (string.IsNullOrWhiteSpace(decoded))
            {
                blank = true;
                return null;
            }
        }

        try
        {
            return decoded is null ? JsonNode.Parse(line) : JsonNode.Parse(decoded);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
        }
    }

    private static IEnumerable<JsonObject> JsonLinesText(Stream stream, string path)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8, true, 1 << 16);
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

    // The bytes already read from a stream, then the rest of it.
    private sealed class PrefixedStream(byte[] prefix, Stream rest) : Stream
    {
        private int _pos;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            if (_pos < prefix.Length)
            {
                int n = Math.Min(buffer.Length, prefix.Length - _pos);
                prefix.AsSpan(_pos, n).CopyTo(buffer);
                _pos += n;
                return n;
            }

            return rest.Read(buffer);
        }

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                rest.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    internal static IEnumerable<JsonObject> Json(Func<Stream> open, string path, ReadOptions options)
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

    internal static IEnumerable<JsonObject> Delimited(Func<Stream> open, char delimiter, ReadOptions options)
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
    // The list is reused: each record is read before the next is asked for.
    private static IEnumerable<List<string>> Records(TextReader reader, char delimiter)
    {
        var records = new CsvReader(reader, delimiter);
        var fields = new List<string>();
        while (records.Next(fields))
        {
            yield return fields;
            fields.Clear();
        }
    }

    // Reads blocks of characters and jumps between delimiters, quotes and line breaks rather than reading a character at a time;
    // an unquoted field within one block is made straight from it.
    private sealed class CsvReader(TextReader reader, char delimiter)
    {
        private static readonly SearchValues<char> Commas = SearchValues.Create(",\r\n"), Tabs = SearchValues.Create("\t\r\n");

        // A quote opens quoting only as a field's first character, so after it only these end an unquoted run.
        private readonly SearchValues<char> _ends = delimiter switch
        {
            ',' => Commas,
            '\t' => Tabs,
            _ => SearchValues.Create([delimiter, '\r', '\n']),
        };

        private readonly char[] _buffer = new char[1 << 14];
        private readonly StringBuilder _field = new();
        private int _pos, _length;

        // Adds the next record's fields; false at the end of the text.
        public bool Next(List<string> fields)
        {
            if (_pos >= _length && !Fill())
            {
                return false;
            }

            while (true)
            {
                if (_pos >= _length && !Fill())
                {
                    fields.Add("");                                 // the text ends after a delimiter
                    return true;
                }

                if (_buffer[_pos] == '"')
                {
                    _pos++;
                    ReadQuoted();
                }

                while (true)
                {
                    int start = _pos;
                    int found = _buffer.AsSpan(start, _length - start).IndexOfAny(_ends);
                    if (found < 0)
                    {
                        _field.Append(_buffer, start, _length - start);
                        if (!Fill())
                        {
                            fields.Add(Take(0, 0));
                            return true;
                        }

                        continue;
                    }

                    int end = start + found;
                    char ch = _buffer[end];
                    fields.Add(Take(start, end));
                    _pos = end + 1;
                    if (ch == delimiter)
                    {
                        break;
                    }

                    if (ch == '\r' && (_pos < _length || Fill()) && _buffer[_pos] == '\n')
                    {
                        _pos++;
                    }

                    return true;
                }
            }
        }

        // A quoted field's text into _field, up to its closing quote (or the end of the text); "" is read as ".
        private void ReadQuoted()
        {
            while (true)
            {
                int start = _pos;
                int found = _buffer.AsSpan(start, _length - start).IndexOf('"');
                if (found < 0)
                {
                    _field.Append(_buffer, start, _length - start);
                    if (!Fill())
                    {
                        return;
                    }

                    continue;
                }

                int quote = start + found;
                _field.Append(_buffer, start, quote - start);
                _pos = quote + 1;
                if ((_pos < _length || Fill()) && _buffer[_pos] == '"')
                {
                    _field.Append('"');
                    _pos++;
                    continue;
                }

                return;
            }
        }

        // The field: the text held in _field (from earlier blocks or quotes), then _buffer[start..end].
        private string Take(int start, int end)
        {
            if (_field.Length == 0)
            {
                return new string(_buffer, start, end - start);
            }

            string text = _field.Append(_buffer, start, end - start).ToString();
            _field.Clear();
            return text;
        }

        private bool Fill()
        {
            _pos = 0;
            _length = reader.Read(_buffer, 0, _buffer.Length);
            return _length > 0;
        }
    }

    private static JsonNode? Infer(string cell)
    {
        if (cell.Length == 0)
        {
            return null;
        }

        // Codes such as zip codes and ids keep their leading zeros as text.
        var digits = cell.AsSpan().TrimStart("-+");
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

    internal static IEnumerable<JsonObject> Text(Func<Stream> open, string path, ReadOptions options)
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
                    if (line is null || line.AsSpan().IsWhiteSpace())
                    {
                        if (paragraph.Length > 0)
                        {
                            yield return new JsonObject { ["text"] = paragraph.ToString(0, paragraph.Length - 1) };   // without the last line's \n
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

    internal static IEnumerable<JsonObject> Code(Func<Stream> open, string path, ReadOptions options)
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

    internal static IEnumerable<JsonObject> ParquetRows(Func<Stream> open)
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
