// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace Idrak.Audio;

// "transcripts": JSONL rows of an audio path and its text (the keys of idrak data audio's manifest.jsonl, and NeMo's
// audio_filepath and duration), the paths relative to the transcript file, its zip, or the source's Audio folder or zip.
internal sealed class TranscriptSpeechSource : ISpeechSource
{
    public string Name => SpeechSources.Transcripts;

    public string Summary => "JSONL rows of {\"audio\", \"text\"} (or audio_filepath, path, file, name; transcript, sentence; seconds, duration), "
        + "audio paths relative to the transcript file, its zip, or the source's audio folder or zip";

    public ISpeechReader Open(SpeechSource source)
    {
        var reader = new SpeechReader(source);
        try
        {
            foreach (string path in source.Paths)
            {
                if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
                {
                    var archive = reader.Archive(path);
                    var entries = archive.Entries.Where(e => e.FullName.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)).OrderBy(e => e.FullName, StringComparer.Ordinal).ToList();
                    if (entries.Count == 0)
                    {
                        throw new InvalidDataException($"{path} holds no .jsonl transcript file.");
                    }

                    foreach (var entry in entries)
                    {
                        string folder = entry.FullName.Contains('/', StringComparison.Ordinal) ? entry.FullName[..entry.FullName.LastIndexOf('/')] : "";
                        Read($"{path}!{entry.FullName}", reader.EntryBytes(path, entry.FullName), relative => reader.Resolve(relative, path, folder), reader);
                    }
                }
                else
                {
                    if (!File.Exists(path))
                    {
                        throw new FileNotFoundException($"The transcript file {path} was not found.", path);
                    }

                    string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
                    Read(path, File.ReadAllBytes(path), relative => reader.Resolve(relative, null, folder), reader);
                }
            }

            reader.Finish();
            return reader;
        }
        catch
        {
            reader.Dispose();
            throw;
        }
    }

    // The rows of one transcript file, line by line from its bytes (rule 74).
    private static void Read(string name, byte[] bytes, Func<string, string> resolve, SpeechReader reader)
    {
        var data = bytes.AsSpan();
        if (data.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
        {
            data = data[3..];
        }

        int line = 0;
        while (data.Length > 0)
        {
            line++;
            int end = data.IndexOf((byte)'\n');
            var row = (end < 0 ? data : data[..end]).TrimEnd((byte)'\r');
            data = end < 0 ? default : data[(end + 1)..];
            if (row.Trim((byte)' ').IsEmpty)
            {
                continue;
            }

            var (audio, text, seconds) = Row(row, name, line);
            reader.Add($"{name}:{line}", resolve(audio), text, seconds);
        }
    }

    private static (string Audio, string Text, double? Seconds) Row(ReadOnlySpan<byte> row, string name, int line)
    {
        string? audio = null, text = null;
        double? seconds = null;
        try
        {
            var json = new Utf8JsonReader(row);
            if (!json.Read() || json.TokenType != JsonTokenType.StartObject)
            {
                throw new InvalidDataException($"{name}, line {line}: a transcript row is a JSON object.");
            }

            while (json.Read() && json.TokenType == JsonTokenType.PropertyName)
            {
                var key = json.ValueSpan;
                json.Read();
                if (json.TokenType == JsonTokenType.String && (key.SequenceEqual("audio"u8) || key.SequenceEqual("audio_filepath"u8) || key.SequenceEqual("path"u8)
                        || key.SequenceEqual("file"u8) || key.SequenceEqual("name"u8)))
                {
                    audio ??= json.GetString();
                }
                else if (json.TokenType == JsonTokenType.String && (key.SequenceEqual("text"u8) || key.SequenceEqual("transcript"u8) || key.SequenceEqual("sentence"u8)))
                {
                    text ??= json.GetString();
                }
                else if (json.TokenType == JsonTokenType.Number && (key.SequenceEqual("seconds"u8) || key.SequenceEqual("duration"u8)))
                {
                    seconds ??= json.GetDouble();
                }
                else
                {
                    json.Skip();
                }
            }
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{name}, line {line}: not a JSON object ({e.Message}).", e);
        }

        return (audio ?? throw new InvalidDataException($"{name}, line {line}: the row names no audio (\"audio\")."),
            text ?? throw new InvalidDataException($"{name}, line {line}: the row has no \"text\"."), seconds);
    }
}

// "audio-folder": every sound file a registered codec reads, in a folder (and its subfolders) or a zip, with its text in
// the .txt file of the same name beside it.
internal sealed class FolderSpeechSource : ISpeechSource
{
    public string Name => SpeechSources.AudioFolder;

    public string Summary => "every sound file of a folder or zip (and its subfolders) with its transcript beside it: name.wav and name.txt";

    public ISpeechReader Open(SpeechSource source)
    {
        var reader = new SpeechReader(source);
        try
        {
            int untranscribed = 0;
            foreach (string path in source.Paths)
            {
                if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
                {
                    var archive = reader.Archive(path);
                    var names = archive.Entries.Select(e => e.FullName).ToHashSet(StringComparer.Ordinal);
                    foreach (string entry in names.Where(AudioCodecs.CanDecode).Order(StringComparer.Ordinal))
                    {
                        string transcript = Path.ChangeExtension(entry, ".txt");
                        if (!names.Contains(transcript))
                        {
                            untranscribed++;
                            continue;
                        }

                        reader.Add($"{path}!{entry}", $"{path}!{entry}", Text(reader.EntryBytes(path, transcript)), null);
                    }
                }
                else
                {
                    if (!Directory.Exists(path))
                    {
                        throw new DirectoryNotFoundException($"The audio folder {path} was not found.");
                    }

                    foreach (string file in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories).Where(AudioCodecs.CanDecode).Order(StringComparer.Ordinal))
                    {
                        string transcript = Path.ChangeExtension(file, ".txt");
                        if (!File.Exists(transcript))
                        {
                            untranscribed++;
                            continue;
                        }

                        reader.Add(file, file, Text(File.ReadAllBytes(transcript)), null);
                    }
                }
            }

            if (untranscribed > 0)
            {
                reader.Note($"{untranscribed} sound files have no .txt transcript beside them and are left out");
            }

            reader.Finish();
            return reader;
        }
        catch
        {
            reader.Dispose();
            throw;
        }
    }

    private static string Text(byte[] bytes)
    {
        var span = bytes.AsSpan();
        if (span.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
        {
            span = span[3..];
        }

        return Encoding.UTF8.GetString(span).Trim();
    }
}

// The utterances of an opened source and where their bytes are: files, or entries of zips opened once and read under a
// lock (an archive is not safe to read from several threads).
internal sealed class SpeechReader(SpeechSource source) : ISpeechReader
{
    private readonly List<SpeechUtterance> _all = [];
    private readonly List<double?> _seconds = [];
    private readonly List<string> _notes = [];
    private readonly Dictionary<string, ZipArchive> _archives = new(StringComparer.Ordinal);
    private SpeechUtterance[] _kept = [];
    private int _missing;

    public int Count => _kept.Length;

    public IReadOnlyList<string> Notes => _notes;

    public SpeechUtterance this[int index] => _kept[index];

    public ZipArchive Archive(string path)
    {
        string full = Path.GetFullPath(path);
        if (!_archives.TryGetValue(full, out var archive))
        {
            if (!File.Exists(full))
            {
                throw new FileNotFoundException($"The archive {path} was not found.", path);
            }

            archive = ZipFile.OpenRead(full);
            _archives[full] = archive;
        }

        return archive;
    }

    // A relative audio path from a transcript: under the source's Audio folder or zip when given, else under the transcript's
    // own folder (in its zip when it is in one). Rooted paths stay as they are.
    public string Resolve(string relative, string? zip, string folder)
    {
        if (Path.IsPathRooted(relative))
        {
            return relative;
        }

        if (source.Audio is { } audio)
        {
            return audio.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ? $"{audio}!{Entry(relative)}" : Path.GetFullPath(Path.Combine(audio, relative));
        }

        return zip is null ? Path.GetFullPath(Path.Combine(folder, relative)) : $"{zip}!{Entry(folder.Length == 0 ? relative : folder + "/" + relative)}";
    }

    private static string Entry(string path)
    {
        var parts = new List<string>();
        foreach (string part in path.Replace('\\', '/').Split('/'))
        {
            if (part is "" or ".")
            {
                continue;
            }

            if (part == ".." && parts.Count > 0)
            {
                parts.RemoveAt(parts.Count - 1);
            }
            else
            {
                parts.Add(part);
            }
        }

        return string.Join('/', parts);
    }

    public void Add(string name, string audio, string text, double? seconds)
    {
        if (!Exists(audio))
        {
            if (!source.SkipMissing)
            {
                throw new FileNotFoundException($"{name}: the audio {audio} was not found (SpeechSource.SkipMissing leaves such rows out).", audio);
            }

            _missing++;
            return;
        }

        _all.Add(new SpeechUtterance(name, audio, text, double.NaN));
        _seconds.Add(seconds);
    }

    public void Note(string note) => _notes.Add(note);

    // Every utterance's length (the row's, else its header's, read in parallel for files), then the source's limits.
    public void Finish()
    {
        var seconds = new double[_all.Count];
        Parallel.For(0, _all.Count, ComputeResources.ParallelOptions, i =>
        {
            seconds[i] = _seconds[i] ?? (Split(_all[i].Audio).Zip is null ? Seconds(_all[i]) : double.NaN);
        });
        for (int i = 0; i < _all.Count; i++)
        {
            if (double.IsNaN(seconds[i]))
            {
                seconds[i] = Seconds(_all[i]);                                          // zip entries: one at a time
            }
        }

        var kept = new List<SpeechUtterance>(_all.Count);
        int shorter = 0, longer = 0;
        for (int i = 0; i < _all.Count; i++)
        {
            if (seconds[i] < source.MinSeconds)
            {
                shorter++;
            }
            else if (source.MaxSeconds > 0 && seconds[i] > source.MaxSeconds)
            {
                longer++;
            }
            else
            {
                kept.Add(_all[i] with { Seconds = seconds[i] });
            }
        }

        if (_missing > 0)
        {
            _notes.Add($"{_missing} rows name audio that was not found and are left out");
        }

        if (shorter > 0)
        {
            _notes.Add(string.Create(CultureInfo.InvariantCulture, $"{shorter} utterances shorter than {source.MinSeconds} s are left out"));
        }

        if (longer > 0)
        {
            _notes.Add(string.Create(CultureInfo.InvariantCulture, $"{longer} utterances longer than {source.MaxSeconds} s are left out"));
        }

        _kept = [.. kept];
        _all.Clear();
        _seconds.Clear();
    }

    private double Seconds(SpeechUtterance utterance)
    {
        var (zip, entry) = Split(utterance.Audio);
        AudioInfo? info;
        if (zip is null)
        {
            info = AudioCodecs.ReadInfo(utterance.Audio);
        }
        else
        {
            byte[] head = ArrayPool<byte>.Shared.Rent(AudioCodecs.HeaderBytes);
            try
            {
                int length;
                lock (_archives)
                {
                    using var stream = Archive(zip).GetEntry(entry)!.Open();
                    length = stream.ReadAtLeast(head.AsSpan(0, AudioCodecs.HeaderBytes), AudioCodecs.HeaderBytes, throwOnEndOfStream: false);
                }

                info = AudioCodecs.ReadInfo(head.AsSpan(0, length));
                if (info is { Samples: < 0 })
                {
                    info = AudioCodecs.ReadInfo(EntryBytes(zip, entry));
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(head);
            }
        }

        if (info is not { Samples: >= 0 } known)
        {
            throw new InvalidDataException($"{utterance.Name}: {utterance.Audio} is not a sound file a registered codec reads ({string.Join(", ", AudioCodecs.Names)}).");
        }

        return known.Samples / (double)known.SampleRate;
    }

    private bool Exists(string audio)
    {
        var (zip, entry) = Split(audio);
        if (zip is null)
        {
            return File.Exists(audio);
        }

        lock (_archives)
        {
            return Archive(zip).GetEntry(entry) is not null;
        }
    }

    private static (string? Zip, string Entry) Split(string audio)
    {
        int bang = audio.IndexOf(".zip!", StringComparison.OrdinalIgnoreCase);
        return bang < 0 ? (null, audio) : (audio[..(bang + 4)], audio[(bang + 5)..]);
    }

    public byte[] EntryBytes(string zip, string entry)
    {
        lock (_archives)
        {
            var found = Archive(zip).GetEntry(entry) ?? throw new FileNotFoundException($"{zip} has no entry {entry}.");
            var bytes = new byte[found.Length];
            using var stream = found.Open();
            stream.ReadExactly(bytes);
            return bytes;
        }
    }

    public byte[] ReadBytes(int index)
    {
        var (zip, entry) = Split(_kept[index].Audio);
        return zip is null ? File.ReadAllBytes(entry) : EntryBytes(zip, entry);
    }

    public void Dispose()
    {
        lock (_archives)
        {
            foreach (var archive in _archives.Values)
            {
                archive.Dispose();
            }

            _archives.Clear();
        }
    }
}
