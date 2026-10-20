// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

// The records of a data file: JSON Lines (one object per non-empty line) or a JSON array of objects (LlamaFactory's
// train.json), streamed one record at a time either way, each with where it is for messages ("train.json, record 12").
internal static class TuningRecords
{
    // JsonNode's own converter, with no reflection (the package is trimmed and AOT-compiled).
    private static readonly JsonTypeInfo<JsonNode> NodeInfo = JsonMetadataServices.CreateValueInfo<JsonNode>(
        new JsonSerializerOptions { TypeInfoResolver = JsonTypeInfoResolver.Combine() }, JsonMetadataServices.JsonNodeConverter);

    public static IEnumerable<(JsonObject Record, string Where)> Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Tuning data file not found: {path}", path);
        }

        return IsArray(path) ? FromArray(path) : FromLines(path);
    }

    // A record's place for messages, with its "id" when it has one.
    public static string Where(string at, JsonObject record) =>
        record["id"] is JsonValue id && id.GetValueKind() is JsonValueKind.String or JsonValueKind.Number ? $"{at} (id {id.ToJsonString().Trim('"')})" : at;

    // Whether the file's first character that is not white space (after a byte order mark) opens an array.
    private static bool IsArray(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096);
        using var reader = new StreamReader(stream, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        int c;
        while ((c = reader.Read()) >= 0 && char.IsWhiteSpace((char)c))
        {
        }

        return c == '[';
    }

    private static IEnumerable<(JsonObject, string)> FromLines(string path)
    {
        foreach (var (node, number) in JsonLines.Read(path))
        {
            if (node is not JsonObject record)
            {
                var notObject = new InvalidDataException("the line is not a JSON object");
                throw new InvalidDataException($"{path}, line {number}: {notObject.Message}", notObject);
            }

            yield return (record, Where($"{path}, line {number}", record));
        }
    }

    private static IEnumerable<(JsonObject, string)> FromArray(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        using var items = JsonSerializer.DeserializeAsyncEnumerable(stream, NodeInfo).ToBlockingEnumerable().GetEnumerator();
        for (int number = 1; ; number++)
        {
            JsonNode? item;
            try
            {
                if (!items.MoveNext())
                {
                    yield break;
                }

                item = items.Current;
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"{path}, record {number}: not valid JSON ({ex.Message})", ex);
            }

            if (item is not JsonObject record)
            {
                throw new InvalidDataException($"{path}, record {number}: a record is a JSON object, not {item?.ToJsonString() ?? "null"}.");
            }

            yield return (record, Where($"{path}, record {number}", record));
        }
    }
}

// The values of a JSON Lines file, each with its line number: the file read as UTF-8 bytes in blocks into one pooled
// buffer, each line parsed from its bytes (no string per line). Lines are numbered and blank ones skipped as
// File.ReadLines and string.IsNullOrWhiteSpace do (a line ends at "\n", "\r\n" or "\r"; a UTF-8 byte order mark is
// skipped); a line that is not valid UTF-8, or starts with white space beyond ASCII's, is decoded as File.ReadLines
// decodes it, and a file with a UTF-16 or UTF-32 byte order mark is read through File.ReadLines. A line that is not
// valid JSON throws InvalidDataException("{path}, line {number}: ...") over the JsonException.
internal static class JsonLines
{
    private static readonly SearchValues<byte> LineEnds = SearchValues.Create("\r\n"u8);
    private static readonly SearchValues<byte> AsciiWhiteSpace = SearchValues.Create("\t\n\v\f\r "u8);

    public static IEnumerable<(JsonNode? Node, int Line)> Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
        int skip = ByteOrderMark(stream);
        if (skip < 0)
        {
            foreach (var line in FromText(path))
            {
                yield return line;
            }

            yield break;
        }

        stream.Position = skip;
        byte[] buffer = ArrayPool<byte>.Shared.Rent(1 << 16);
        int start = 0, filled = 0, number = 0;
        bool end = false;
        try
        {
            while (true)
            {
                if (NextLine(buffer.AsSpan(start, filled - start), end, out int length, out int next))
                {
                    number++;
                    var node = Parse(buffer.AsSpan(start, length), path, number, out bool blank);
                    start += next;
                    if (!blank)
                    {
                        yield return (node, number);
                    }

                    continue;
                }

                if (end)
                {
                    yield break;
                }

                // The unfinished line moves to the front (into a larger buffer when it fills this one), then more is read.
                if (start == 0 && filled == buffer.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(buffer.Length * 2);
                    buffer.AsSpan(0, filled).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(buffer);
                    buffer = larger;
                }
                else if (start > 0)
                {
                    buffer.AsSpan(start, filled - start).CopyTo(buffer);
                    filled -= start;
                    start = 0;
                }

                int read = stream.Read(buffer, filled, buffer.Length - filled);
                filled += read;
                end = read == 0;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The bytes of a UTF-8 byte order mark to skip (3 or 0), or -1 for a UTF-16 or UTF-32 one (as StreamReader detects them).
    private static int ByteOrderMark(FileStream stream)
    {
        Span<byte> head = stackalloc byte[4];
        ReadOnlySpan<byte> bytes = head[..stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false)];
        return bytes.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]) ? 3
            : bytes.StartsWith((ReadOnlySpan<byte>)[0xFE, 0xFF]) || bytes.StartsWith((ReadOnlySpan<byte>)[0xFF, 0xFE])
              || bytes.StartsWith((ReadOnlySpan<byte>)[0x00, 0x00, 0xFE, 0xFF]) ? -1
            : 0;
    }

    // The first line of `data`: its length without the line end, and the bytes it takes with it. Not found while more
    // can come (no line end yet, or a "\r" last whose "\n" may follow); at the end, the rest is the last line.
    private static bool NextLine(ReadOnlySpan<byte> data, bool end, out int length, out int consumed)
    {
        int at = data.IndexOfAny(LineEnds);
        if (at >= 0 && (data[at] == (byte)'\n' || at + 1 < data.Length || end))
        {
            length = at;
            consumed = at + (data[at] == (byte)'\r' && at + 1 < data.Length && data[at + 1] == (byte)'\n' ? 2 : 1);
            return true;
        }

        length = consumed = data.Length;
        return end && data.Length > 0;
    }

    // A line's value; blank (and null) for a line of white space only.
    private static JsonNode? Parse(ReadOnlySpan<byte> line, string path, int number, out bool blank)
    {
        int first = line.IndexOfAnyExcept(AsciiWhiteSpace);
        blank = first < 0;
        if (blank)
        {
            return null;
        }

        try
        {
            if (line[first] < 0x80 && System.Text.Unicode.Utf8.IsValid(line))
            {
                return JsonNode.Parse(line);
            }

            // Other white space, or bytes that are not UTF-8: the text File.ReadLines reads (invalid bytes as U+FFFD).
            string text = Encoding.UTF8.GetString(line);
            blank = string.IsNullOrWhiteSpace(text);
            return blank ? null : JsonNode.Parse(text);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
        }
    }

    // A file in another encoding than UTF-8, read as text.
    private static IEnumerable<(JsonNode? Node, int Line)> FromText(string path)
    {
        int number = 0;
        foreach (string line in File.ReadLines(path))
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

            yield return (node, number);
        }
    }
}

// Where a data file's images come from: a folder (the data file's, or the one given) or a zip opened read-only, of
// which only the image entries the data names are read. Absolute paths are read from disk (one that is not there, a path
// from the machine the data was made on, by its longest trailing part under the root); data URLs are decoded; remote
// URLs are refused.
internal sealed class TuningImageSource : IDisposable
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".pnm",
    };

    private readonly string _folder;
    private readonly string? _zipPath;
    private readonly ZipArchive? _zip;
    private readonly Dictionary<string, ZipArchiveEntry>? _entries;
    private readonly string? _top;        // the zip's one top-level folder, when everything is in one
    private readonly string? _stem;       // the zip's file name without its extension
    private readonly Dictionary<string, ZipArchiveEntry?>? _endings;   // every trailing part of every entry's path; null: several entries end so

    /// <summary>Throws when <paramref name="images"/> is given and is neither a folder nor a .zip file (named as the <paramref name="what"/> root).</summary>
    public static void CheckRoot(string? images, string what = "images")
    {
        if (images is not null && !Directory.Exists(images) && !(File.Exists(images) && string.Equals(Path.GetExtension(images), ".zip", StringComparison.OrdinalIgnoreCase)))
        {
            throw new DirectoryNotFoundException($"The {what} root {images} is neither a folder nor a .zip file.");
        }
    }

    public TuningImageSource(string dataPath, string? images, string what = "images")
    {
        CheckRoot(images, what);
        string root = images ?? Path.GetDirectoryName(Path.GetFullPath(dataPath)) ?? ".";
        if (images is not null && File.Exists(images) && string.Equals(Path.GetExtension(images), ".zip", StringComparison.OrdinalIgnoreCase))
        {
            _zipPath = images;
            _folder = Path.GetDirectoryName(Path.GetFullPath(images)) ?? ".";
            _zip = new ZipArchive(new FileStream(images, FileMode.Open, FileAccess.Read, FileShare.Read), ZipArchiveMode.Read, leaveOpen: false);
            _entries = new Dictionary<string, ZipArchiveEntry>(StringComparer.Ordinal);
            foreach (var entry in _zip.Entries.Where(e => e.FullName.Length > 0 && !e.FullName.EndsWith('/')))
            {
                _entries.TryAdd(Normalize(entry.FullName), entry);
            }

            // Each entry under its trailing parts too (a/b/c.png as b/c.png and c.png), made once for every lookup.
            _endings = new Dictionary<string, ZipArchiveEntry?>(StringComparer.Ordinal);
            foreach (var (key, entry) in _entries)
            {
                for (int at = key.IndexOf('/', StringComparison.Ordinal); at >= 0; at = key.IndexOf('/', at + 1))
                {
                    string ending = key[(at + 1)..];
                    _endings[ending] = _endings.ContainsKey(ending) ? null : entry;
                }
            }

            var tops = _entries.Keys.Select(k => k.IndexOf('/', StringComparison.Ordinal) is var at and > 0 ? k[..at] : null).Distinct().ToList();
            _top = tops is [{ } only] ? only : null;
            _stem = Path.GetFileNameWithoutExtension(images);
        }
        else
        {
            _folder = root;
        }
    }

    /// <summary>The image <paramref name="reference"/> names, for the record at <paramref name="where"/>.</summary>
    public ChatImage Load(string reference, string where)
    {
        if (string.IsNullOrWhiteSpace(reference))
        {
            throw new InvalidDataException($"{where}: an image entry is empty.");
        }

        string text = reference.Trim();
        if (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                return ChatImage.FromDataUrl(text);
            }
            catch (FormatException ex)
            {
                throw new InvalidDataException($"{where}: {ex.Message}", ex);
            }
        }

        if (Uri.TryCreate(text, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https" or "ftp" or "s3" or "gs")
        {
            throw new InvalidDataException($"{where}: the image {text} is remote; Idrak reads local images only (download them, then give their folder or zip as the images root).");
        }

        if (Path.IsPathFullyQualified(text) && (File.Exists(text) || _zip is null && !Trailing(Normalize(text)).Any(p => File.Exists(Path.Combine(_folder, p)))))
        {
            return FromFile(text, where);
        }

        // A path from the machine the data was made on (/workspace/pdf_images/0012/page_013.jpg) is found by its longest
        // trailing part under the images root (pdf_images/0012/page_013.jpg, else 0012/page_013.jpg, ...).
        return _zip is null ? FromFolder(text, where) : FromZip(text, where);
    }

    public void Dispose() => _zip?.Dispose();

    /// <summary>
    /// The sound <paramref name="reference"/> names (a path under the root or in the zip, or a data URL), as a part of
    /// the "audio" kind registered in <see cref="ChatParts"/> (Idrak.Audio's), for the record at <paramref name="where"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">No "audio" kind is registered, or the reference is empty, remote or not a sound.</exception>
    public ChatPart LoadAudio(string reference, string where)
    {
        if (ChatParts.Find(ChatParts.Audio) is not IBinaryPartKind audio)
        {
            throw new InvalidDataException($"{where}: the record names a sound ({reference}) and no 'audio' part kind is registered in ChatParts "
                + "(ship Idrak.Audio, or register one with ChatParts.Register).");
        }

        if (string.IsNullOrWhiteSpace(reference))
        {
            throw new InvalidDataException($"{where}: an audio entry is empty.");
        }

        string text = reference.Trim();
        if (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            int comma = text.IndexOf(',', StringComparison.Ordinal);
            string[] header = comma < 0 ? [] : text[5..comma].Split(';', StringSplitOptions.TrimEntries);
            if (comma < 0 || !header.Skip(1).Contains("base64", StringComparer.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"{where}: an audio data URL must be base64 (data:audio/wav;base64,...).");
            }

            try
            {
                return audio.FromBytes(Convert.FromBase64String(text[(comma + 1)..]), header[0].Length == 0 ? null : header[0]);
            }
            catch (Exception ex) when (ex is FormatException or ArgumentException)
            {
                throw new InvalidDataException($"{where}: {ex.Message}", ex);
            }
        }

        if (Uri.TryCreate(text, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https" or "ftp" or "s3" or "gs")
        {
            throw new InvalidDataException($"{where}: the sound {text} is remote; Idrak reads local files only (download them, then give their folder or zip as the audios root).");
        }

        if (_zip is not null && !(Path.IsPathFullyQualified(text) && File.Exists(text)))
        {
            string name = Normalize(text);
            var entry = Find(name) ?? throw new FileNotFoundException($"{where}: the sound {reference} is not in {_zipPath} (looked for '{name}' and for an entry ending in each trailing part of it).", reference);
            if (entry.Length == 0 || entry.Length > Array.MaxLength)
            {
                throw new InvalidDataException($"{where}: the sound {entry.FullName} in {_zipPath} is {(entry.Length == 0 ? "empty" : $"{entry.Length:N0} bytes, too large for one sound")}.");
            }

            return ReadAudio(audio, entry.Open(), (int)entry.Length, where, entry.FullName);
        }

        string path = Path.IsPathFullyQualified(text) && File.Exists(text) ? text : FolderPath(text);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{where}: audio file not found: {path}", path);
        }

        long size = new FileInfo(path).Length;
        if (size == 0 || size > Array.MaxLength)
        {
            throw new InvalidDataException($"{where}: the audio file {path} is {(size == 0 ? "empty" : $"{size:N0} bytes, too large for one sound")}.");
        }

        return ReadAudio(audio, File.OpenRead(path), (int)size, where, path);
    }

    // The sound's bytes read through a pooled block into the part's own array (one allocation per sound, rule 10).
    private static ChatPart ReadAudio(IBinaryPartKind audio, Stream stream, int length, string where, string name)
    {
        byte[] block = ArrayPool<byte>.Shared.Rent(length);
        try
        {
            using (stream)
            {
                stream.ReadExactly(block.AsSpan(0, length));
            }

            return audio.FromBytes(block.AsSpan(0, length), null);
        }
        catch (ArgumentException ex)
        {
            throw new InvalidDataException($"{where}: {name}: {ex.Message}", ex);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(block);
        }
    }

    // A relative path under the folder, else its longest trailing part that is a file there; the direct path when none is.
    private string FolderPath(string reference)
    {
        string direct = Path.Combine(_folder, reference);
        if (File.Exists(direct))
        {
            return direct;
        }

        foreach (string part in Trailing(Normalize(reference)).Skip(1))
        {
            string path = Path.Combine(_folder, part);
            if (File.Exists(path))
            {
                return path;
            }
        }

        return direct;                                                          // names the path as given
    }

    private static ChatImage FromFile(string path, string where)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{where}: image file not found: {path}", path);
        }

        var image = ChatImage.FromFile(path);
        return ImageExtensions.Contains(Path.GetExtension(path)) || image.MediaType is not null ? image
            : throw new InvalidDataException($"{where}: {path} is not an image (no image file extension and no PNG, JPEG, GIF, BMP, WebP or TIFF signature).");
    }

    // A relative path under the folder; else its longest trailing part that is a file there.
    private ChatImage FromFolder(string reference, string where) => FromFile(FolderPath(reference), where);

    // The path and its trailing parts, longest first: a/b/c.png, b/c.png, c.png.
    private static IEnumerable<string> Trailing(string path)
    {
        for (int at = -1; ; at = path.IndexOf('/', at + 1))
        {
            yield return path[(at + 1)..];
            if (path.IndexOf('/', at + 1) < 0)
            {
                yield break;
            }
        }
    }

    private ChatImage FromZip(string reference, string where)
    {
        string name = Normalize(reference);
        var entry = Find(name) ?? throw new FileNotFoundException(
            $"{where}: the image {reference} is not in {_zipPath} (looked for '{name}'{(_top is null ? "" : $" and '{_top}/{name}'")}, and for an entry ending in "
            + $"each trailing part of it: {string.Join(", ", Trailing(name).Select(p => $"'/{p}'" + (_endings!.TryGetValue(p, out var e) && e is null ? " (in several entries)" : "")))}).", reference);
        if (!ImageExtensions.Contains(Path.GetExtension(entry.FullName)))
        {
            throw new InvalidDataException($"{where}: the entry {entry.FullName} of {_zipPath} is not an image file ({string.Join(", ", ImageExtensions.Order(StringComparer.Ordinal))}); only images are read from the zip.");
        }

        if (entry.Length > Array.MaxLength)
        {
            throw new InvalidDataException($"{where}: the entry {entry.FullName} of {_zipPath} is {entry.Length:N0} bytes, too large for one image.");
        }

        if (entry.Length == 0)
        {
            throw new InvalidDataException($"{where}: the image {entry.FullName} in {_zipPath} is empty.");
        }

        // Inflated straight into the image's own array (a page is a large-object allocation: one, not a buffer and a copy).
        using var stream = entry.Open();
        return ChatImage.FromStream(stream, (int)entry.Length);
    }

    // The entry a relative path names: exactly; under the zip's one top folder; without a first folder named as the zip
    // (images/a.png in images.zip holding a.png); or the one entry ending in the path's longest trailing part that some
    // entry ends in (/workspace/pdf_images/1/p.jpg finds downloaded_images/pdf_images/1/p.jpg; a part several entries end
    // in finds none).
    private ZipArchiveEntry? Find(string name)
    {
        if (_entries!.TryGetValue(name, out var entry)
            || (_top is not null && _entries.TryGetValue($"{_top}/{name}", out entry))
            || (_stem is not null && name.StartsWith(_stem + "/", StringComparison.Ordinal) && _entries.TryGetValue(name[(_stem.Length + 1)..], out entry)))
        {
            return entry;
        }

        foreach (string part in Trailing(name))
        {
            if (_entries.TryGetValue(part, out entry))
            {
                return entry;
            }

            if (_endings!.TryGetValue(part, out var ending))
            {
                return ending;                                                  // null when several entries end so
            }
        }

        return null;
    }

    private static string Normalize(string path)
    {
        string name = path.Replace('\\', '/');
        if (name.Length >= 2 && name[1] == ':' && char.IsAsciiLetter(name[0]))
        {
            name = name[2..];                                                   // a drive (C:/data/a.png) is the other machine's
        }

        while (name.StartsWith("./", StringComparison.Ordinal))
        {
            name = name[2..];
        }

        return name.TrimStart('/');
    }
}

// An image part not read yet: the index of the record's "images" entry it stands for, or the reference a part gave
// itself (a path or data URL in the part: Index unused). The formats read a record's text first, check the count of its
// images, then read them when the record is taken.
internal sealed record PendingImage(int Index, string? Reference = null) : ChatPart
{
    public override string Kind => ChatParts.Image;
}

// An audio part not read yet, as PendingImage: the index of the record's "audios" entry, or the part's own reference.
internal sealed record PendingAudio(int Index, string? Reference = null) : ChatPart
{
    public override string Kind => ChatParts.Audio;
}

// A record read and checked, its images and sounds not read yet: the transcript with a PendingImage (PendingAudio) for
// each image (audio) part (but those that carry their bytes), the "images" ("audios") entries they take, how many image
// and audio parts it has and where the record is.
internal sealed record PendingRecord(ChatTranscript Transcript, IReadOnlyList<string> References, int Images, string Where)
{
    public IReadOnlyList<string> AudioReferences { get; init; } = [];

    public int Audios { get; init; }
}

// The placeholders counted while a record's messages are read: image parts and <image> markers, audio parts and <audio>
// markers, each numbered from 0 in its own list.
internal struct PendingCounts
{
    public int Images, Audios;
}

// "transcripts": {"audio", "text"} per line or as a JSON array, the keys as Idrak.Audio's "transcripts" speech source reads
// them (audio, audio_filepath, path or file; text, transcript or sentence), so NeMo manifests read too: each row a
// conversation of the user's sound and the prompt (TuningDataOptions.Prompt), answered by the text.
internal sealed class TranscriptsDataFormat : ITuningDataFormat
{
    private static readonly string[] AudioKeys = ["audio", "audio_filepath", "path", "file"], TextKeys = ["text", "transcript", "sentence"];

    public string Name => TuningDataFormats.Transcripts;

    public string Summary => "{\"audio\", \"text\"} per line or a JSON array (also NeMo's audio_filepath): the sound and the prompt (TuningDataOptions.Prompt, \"Transcribe.\"), answered by the text";

    public bool Recognizes(JsonObject record) =>
        record["messages"] is null && record["conversations"] is null && AudioKeys.Any(k => record[k] is JsonValue) && TextKeys.Any(k => record[k] is JsonValue);

    public IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options = null) => TuningRecordImages.Read(path, options, (r, w) => Pending(r, w, options));

    public PendingConversations ReadPending(string path, TuningDataOptions? options = null) => TuningRecordImages.ReadPending(path, options, (r, w) => Pending(r, w, options));

    private static PendingRecord Pending(JsonObject record, string where, TuningDataOptions? options)
    {
        string? sound = AudioKeys.Select(k => (string?)record[k]).FirstOrDefault(v => !string.IsNullOrWhiteSpace(v));
        string? text = TextKeys.Select(k => (string?)record[k]).FirstOrDefault(v => v is not null);
        if (sound is null || text is null)
        {
            throw new InvalidDataException($"{where}: a transcript row needs a sound ({string.Join(", ", AudioKeys)}) and its text ({string.Join(", ", TextKeys)}).");
        }

        string prompt = options?.Prompt ?? "Transcribe.";
        ChatPart[] user = prompt.Length == 0 ? [new PendingAudio(-1, sound)] : [new PendingAudio(-1, sound), new ChatText(prompt)];
        var transcript = new ChatTranscript([new ChatMessage("user", user), new ChatMessage("assistant", text)], []);
        return TuningRecordImages.Check(transcript, new PendingCounts(), [], $"{TuningRecordImages.Marker} markers", where);
    }
}

// What the two library formats share: placeholders found while a record's messages are read, checked against its
// "images" list, then filled in place. Each format reads a record into a transcript whose images are placeholders
// (`pending`); Read fills each record's images as the record is read (streamed), ReadPending keeps every record and reads
// a record's images only when it is taken.
internal static class TuningRecordImages
{
    public static IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options, Func<JsonObject, string, PendingRecord> pending)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var records = TuningRecords.Read(path);
        TuningImageSource.CheckRoot(options?.Images);
        return Iterate();

        IEnumerable<ChatTranscript> Iterate()
        {
            using var source = new TuningImageSource(path, options?.Images);
            using var audios = AudioSource(path, options, source);
            foreach (var (record, where) in records)
            {
                yield return Fill(pending(record, where), source, audios);
            }
        }
    }

    // Where sounds come from: their own root when given, else where the images come from.
    private static TuningImageSource AudioSource(string path, TuningDataOptions? options, TuningImageSource images) =>
        options?.Audios is { } root ? new TuningImageSource(path, root, "audios") : images;

    public static PendingConversations ReadPending(string path, TuningDataOptions? options, Func<JsonObject, string, PendingRecord> pending)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var records = TuningRecords.Read(path);
        TuningImageSource.CheckRoot(options?.Images);
        var read = new List<PendingRecord>();
        int images = 0;
        foreach (var (record, where) in records)
        {
            var one = pending(record, where);
            read.Add(one);
            images += one.Images;
        }

        var source = new TuningImageSource(path, options?.Images);             // opened once every record has read
        var audios = AudioSource(path, options, source);
        return new PendingConversations(read.Count, images, i => Fill(read[i], source, audios), ReferenceEquals(audios, source) ? source : new Both(source, audios));
    }

    private sealed class Both(IDisposable a, IDisposable b) : IDisposable
    {
        public void Dispose()
        {
            a.Dispose();
            b.Dispose();
        }
    }

    public const string Marker = "<image>";

    public const string AudioMarker = "<audio>";

    // Whether a failure while reading the record at `where` is rethrown as an InvalidDataException naming it (one that
    // already names it, or a missing file, is not).
    public static bool Rewrap(Exception ex, string where) =>
        ex is FormatException or NotSupportedException or InvalidOperationException or ArgumentException or JsonException
        || (ex is InvalidDataException && !ex.Message.StartsWith(where, StringComparison.Ordinal));

    // The parts of a text with each <image> marker a pending image and each <audio> marker a pending sound, numbered
    // from `next`'s counts (counted up).
    public static IEnumerable<ChatPart> Split(string text, ref PendingCounts next, bool images = true, bool audios = true)
    {
        var parts = new List<ChatPart>();
        int from = 0;
        while (true)
        {
            int image = images ? text.IndexOf(Marker, from, StringComparison.Ordinal) : -1, audio = audios ? text.IndexOf(AudioMarker, from, StringComparison.Ordinal) : -1;
            if (image < 0 && audio < 0)
            {
                break;
            }

            bool isImage = audio < 0 || (image >= 0 && image < audio);
            int at = isImage ? image : audio;
            if (at > from)
            {
                parts.Add(new ChatText(text[from..at]));
            }

            parts.Add(isImage ? new PendingImage(next.Images++) : new PendingAudio(next.Audios++));
            from = at + (isImage ? Marker.Length : AudioMarker.Length);
        }

        if (from < text.Length)
        {
            parts.Add(new ChatText(from == 0 ? text : text[from..]));
        }

        return parts;
    }

    // The record's "images" entries as references (strings, or objects with a "path"); null when it has none.
    public static List<string>? References(JsonObject record, string where) => References(record, "images", where);

    // The record's `key` entries ("images", "audios") as references (strings, or objects with a "path"); null when it has none.
    public static List<string>? References(JsonObject record, string key, string where)
    {
        if (record[key] is not { } node)
        {
            return null;
        }

        if (node is JsonValue single && single.GetValueKind() == JsonValueKind.String)
        {
            return [single.GetValue<string>()];
        }

        string what = key == "images" ? "image" : "audio";
        if (node is not JsonArray list)
        {
            throw new InvalidDataException($"{where}: \"{key}\" is a list of {what} paths, not {node.ToJsonString()}.");
        }

        var references = new List<string>(list.Count);
        foreach (var item in list)
        {
            references.Add(item switch
            {
                JsonValue v when v.GetValueKind() == JsonValueKind.String => v.GetValue<string>(),
                JsonObject o when o["path"] is JsonValue p && p.GetValueKind() == JsonValueKind.String => p.GetValue<string>(),
                _ => throw new InvalidDataException($"{where}: an entry of \"{key}\" is a path (or an object with a \"path\"), not {item?.ToJsonString() ?? "null"}."),
            });
        }

        return references;
    }

    // The record, after checking there is one entry of `references` per image placeholder (`placeholders.Images` of them:
    // bare parts and markers; parts that name their own image are pending too, and counted apart) and one entry of the
    // record's "audios" per audio placeholder.
    public static PendingRecord Check(ChatTranscript transcript, PendingCounts placeholders, IReadOnlyList<string> references, string what, string where, JsonObject? record = null)
    {
        if (placeholders.Images != references.Count)
        {
            throw new InvalidDataException($"{where}: {placeholders.Images} {what} for {references.Count} entr{(references.Count == 1 ? "y" : "ies")} of \"images\"; "
                + "each image needs its own placeholder, in the order of the list.");
        }

        var audioReferences = record is null ? null : References(record, "audios", where);
        if (placeholders.Audios != (audioReferences?.Count ?? 0))
        {
            throw new InvalidDataException($"{where}: {placeholders.Audios} audio placeholder{(placeholders.Audios == 1 ? "" : "s")} (bare audio parts and {AudioMarker} markers) for "
                + $"{audioReferences?.Count ?? 0} entr{(audioReferences?.Count == 1 ? "y" : "ies")} of \"audios\"; each sound needs its own placeholder, in the order of the list.");
        }

        int images = 0, audios = 0;
        foreach (var message in transcript.Messages)
        {
            images += message.Parts.Count(p => p is PendingImage or ChatImage);     // the parts that carry their bytes are read already
            audios += message.Parts.Count(p => p is PendingAudio || (p is not ChatText and not ChatImage && p.Kind == ChatParts.Audio));
        }

        return new PendingRecord(transcript, references, images, where) { AudioReferences = audioReferences ?? [], Audios = audios };
    }

    // The record's transcript with each pending image and sound read (an "images" or "audios" entry once, however many
    // placeholders take it).
    public static ChatTranscript Fill(PendingRecord record, TuningImageSource source, TuningImageSource? audios = null)
    {
        if (record.Images == 0 && record.Audios == 0)
        {
            return record.Transcript;
        }

        string where = record.Where;
        audios ??= source;
        try
        {
            var images = new ChatImage?[record.References.Count];
            var sounds = new ChatPart?[record.AudioReferences.Count];
            var messages = record.Transcript.Messages.Select(m => m.Parts.Any(p => p is PendingImage or PendingAudio)
                ? m with
                {
                    Parts = [.. m.Parts.Select(p => p switch
                    {
                        PendingImage pending => pending.Reference is { } own ? source.Load(own, where) : images[pending.Index] ??= source.Load(record.References[pending.Index], where),
                        PendingAudio pending => pending.Reference is { } own ? audios.LoadAudio(own, where) : sounds[pending.Index] ??= audios.LoadAudio(record.AudioReferences[pending.Index], where),
                        _ => p,
                    })],
                }
                : m).ToList();
            return record.Transcript with { Messages = messages };
        }
        catch (Exception ex) when (Rewrap(ex, where))
        {
            throw new InvalidDataException($"{where}: {ex.Message}", ex);
        }
    }
}

// "messages": JSON Lines (or an array) of {"messages": [...], "tools": [...]}, image parts from the chat JSON, a path, a
// data URL, or the record's "images" list (bare image parts and, with that list, <image> markers in text).
internal sealed class MessagesDataFormat : ITuningDataFormat
{
    public string Name => TuningDataFormats.Messages;

    public string Summary => "{\"messages\": [{\"role\", \"content\"}], \"tools\"} per line (or a JSON array); image (audio) parts by path, data URL, base64 or the record's \"images\" (\"audios\") list";

    public bool Recognizes(JsonObject record) => record["messages"] is JsonArray;

    public IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options = null) => TuningRecordImages.Read(path, options, Pending);

    public PendingConversations ReadPending(string path, TuningDataOptions? options = null) => TuningRecordImages.ReadPending(path, options, Pending);

    // The record at `where` with its images as placeholders, their count checked against its "images".
    private static PendingRecord Pending(JsonObject record, string where)
    {
        try
        {
            var references = TuningRecordImages.References(record, where);
            bool audios = record["audios"] is not null;
            var pending = new PendingCounts();
            var transcript = ChatTranscript.Parse(record, content => Content(content, references is not null, audios, ref pending));
            if (references is null && pending.Images > 0)
            {
                throw new InvalidDataException($"{where}: {pending.Images} image part{(pending.Images == 1 ? "" : "s")} without data or a path, and no \"images\" list to take them from.");
            }

            return TuningRecordImages.Check(transcript, pending, references ?? [], "image placeholders (bare image parts and <image> markers)", where, record);
        }
        catch (Exception ex) when (TuningRecordImages.Rewrap(ex, where))
        {
            throw new InvalidDataException($"{where}: {ex.Message}", ex);
        }
    }

    // Markers: <image> markers take the record's "images" when it has that list, <audio> markers its "audios"; a text
    // without the list for a marker keeps it as text.
    private static IReadOnlyList<ChatPart> Content(JsonNode? content, bool markers, bool audioMarkers, ref PendingCounts pending)
    {
        switch (content)
        {
            case null:
                return [];
            case JsonValue v when v.GetValueKind() == JsonValueKind.String:
                string text = v.GetValue<string>();
                return text.Length == 0 ? [] : [.. Split(text, markers, audioMarkers, ref pending)];
            case JsonArray items:
                var parts = new List<ChatPart>(items.Count);
                foreach (var item in items)
                {
                    if (item is JsonObject o && (string?)o["type"] is (ChatParts.Image or "image_url") and var type)
                    {
                        var (reference, bare) = Image(o, type);
                        if (bare)
                        {
                            parts.Add(new PendingImage(pending.Images++));
                        }
                        else if (reference is not null)
                        {
                            parts.Add(new PendingImage(-1, reference));               // read when the record is taken
                        }
                        else
                        {
                            parts.Add(ChatParts.FromJson(o));
                        }
                    }
                    else if (item is JsonObject a && (string?)a["type"] == ChatParts.Audio)
                    {
                        if (a["data"] is not null)
                        {
                            parts.Add(ChatParts.FromJson(a));                         // the registered kind reads its bytes
                        }
                        else if (Reference(a, "path", "audio", "url") is { } reference)
                        {
                            parts.Add(new PendingAudio(-1, reference));
                        }
                        else
                        {
                            parts.Add(new PendingAudio(pending.Audios++));
                        }
                    }
                    else if ((markers || audioMarkers) && item is JsonObject t && (string?)t["type"] == ChatParts.Text && (string?)t["text"] is { } inner)
                    {
                        parts.AddRange(Split(inner, markers, audioMarkers, ref pending));
                    }
                    else
                    {
                        parts.Add(ChatParts.FromJson(item));
                    }
                }

                return parts;
            default:
                return ChatParts.ContentFromJson(content);
        }
    }

    // The text's parts: its markers of the kinds whose lists the record has become pending parts; the others stay text.
    private static IEnumerable<ChatPart> Split(string text, bool markers, bool audioMarkers, ref PendingCounts pending) =>
        markers || audioMarkers ? TuningRecordImages.Split(text, ref pending, markers, audioMarkers) : [new ChatText(text)];

    private static string? Reference(JsonObject part, params ReadOnlySpan<string> keys)
    {
        foreach (string key in keys)
        {
            if (part[key] is JsonValue value && value.GetValueKind() == JsonValueKind.String)
            {
                return value.GetValue<string>();
            }
        }

        return null;
    }

    // An image part's reference: a path or URL (null when it carries its bytes as base64 "data"), or bare (no source at
    // all: the next entry of "images").
    private static (string? Reference, bool Bare) Image(JsonObject part, string type)
    {
        if (type == "image_url")
        {
            return part["image_url"] switch
            {
                JsonObject url when (string?)url["url"] is { } u => (u, false),
                JsonValue url when url.GetValueKind() == JsonValueKind.String => (url.GetValue<string>(), false),
                _ => throw new FormatException($"An image_url part needs a \"url\": {part.ToJsonString()}"),
            };
        }

        if (part["data"] is not null)
        {
            return (null, false);
        }

        foreach (string key in (ReadOnlySpan<string>)["path", "image", "url"])
        {
            if (part[key] is JsonValue value && value.GetValueKind() == JsonValueKind.String)
            {
                return (value.GetValue<string>(), false);
            }
        }

        return (null, true);
    }
}

// "sharegpt": {"conversations": [{"from", "value"}], "system", "tools", "images"} per line or as a JSON array, as
// LlamaFactory reads it: each <image> in a turn's text is the next entry of "images", and the counts must agree.
internal sealed class ShareGptDataFormat : ITuningDataFormat
{
    public string Name => TuningDataFormats.ShareGpt;

    public string Summary => "{\"conversations\": [{\"from\", \"value\"}], \"system\", \"images\", \"audios\"} per line or a JSON array (LlamaFactory); each <image> (<audio>) in a turn takes the next entry of \"images\" (\"audios\")";

    public bool Recognizes(JsonObject record) => record["conversations"] is JsonArray;

    public IEnumerable<ChatTranscript> Read(string path, TuningDataOptions? options = null) => TuningRecordImages.Read(path, options, Pending);

    public PendingConversations ReadPending(string path, TuningDataOptions? options = null) => TuningRecordImages.ReadPending(path, options, Pending);

    // The record at `where` with its images as placeholders, their count checked against its "images".
    private static PendingRecord Pending(JsonObject record, string where)
    {
        try
        {
            if (record["conversations"] is not JsonArray)
            {
                throw new InvalidDataException($"{where}: a ShareGPT record needs \"conversations\".");
            }

            // LlamaFactory writes tools as a JSON string; the transcript reads a list.
            if (record["tools"] is JsonValue tools && tools.GetValueKind() == JsonValueKind.String)
            {
                record = (JsonObject)record.DeepClone();
                string text = tools.GetValue<string>();
                record["tools"] = string.IsNullOrWhiteSpace(text) ? null : JsonNode.Parse(text);
            }

            var pending = new PendingCounts();
            var transcript = ChatTranscript.Parse(record, value => value switch
            {
                null => [],
                JsonValue v when v.GetValueKind() == JsonValueKind.String => [.. TuningRecordImages.Split(v.GetValue<string>(), ref pending)],
                _ => throw new InvalidDataException($"a turn's \"value\" is text, not {value.ToJsonString()}"),
            });
            if ((string?)record["system"] is { Length: > 0 } system && transcript.Messages is not [{ Role: "system" }, ..])
            {
                transcript = transcript with { Messages = [new ChatMessage("system", system), .. transcript.Messages] };
            }

            return TuningRecordImages.Check(transcript, pending, TuningRecordImages.References(record, where) ?? [], $"{TuningRecordImages.Marker} markers", where, record);
        }
        catch (Exception ex) when (TuningRecordImages.Rewrap(ex, where))
        {
            throw new InvalidDataException($"{where}: {ex.Message}", ex);
        }
    }
}
