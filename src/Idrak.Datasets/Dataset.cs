using System.Collections;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Datasets;

/// <summary>When <see cref="Dataset.Mix"/> stops.</summary>
public enum MixStop
{
    /// <summary>When the first source runs out (the proportions hold throughout).</summary>
    FirstExhausted,

    /// <summary>When every source has run out (sources that run out drop out and the others share their weight).</summary>
    AllExhausted,
}

/// <summary>
/// A dataset: rows (JSON objects, column name → value) read lazily and streamed, so datasets larger than memory work.
/// Each enumeration reads the source again; operations (<see cref="Select"/>, <see cref="Where"/>, <see cref="Shuffle"/>,
/// <see cref="Deduplicate"/>, <see cref="Split"/>, <see cref="Mix"/> …) return new datasets without reading anything.
/// Create one with <see cref="FromFile"/>, <see cref="FromFolder"/>, <see cref="FromUrl"/> or
/// the Hugging Face, GitHub, Kaggle and Zenodo sources.
/// </summary>
public sealed class Dataset : IEnumerable<JsonObject>
{
    private readonly Func<IEnumerable<JsonObject>> _rows;

    /// <summary>A dataset whose rows <paramref name="rows"/> produces (called again for each enumeration).</summary>
    public Dataset(Func<IEnumerable<JsonObject>> rows, string name = "dataset")
    {
        _rows = rows ?? throw new ArgumentNullException(nameof(rows));
        Name = name;
    }

    /// <summary>A name for messages and reports (the source it was read from).</summary>
    public string Name { get; }

    // The files a source reads (downloading remote ones first), when the dataset is a source rather than derived from one.
    internal Func<IReadOnlyList<string>>? Files { get; init; }

    /// <summary>
    /// The local files this dataset reads, downloading remote ones into the cache first (without reading rows). Only for
    /// datasets created directly from a source (files, folders, URLs, Hugging Face, GitHub, Kaggle, Zenodo).
    /// </summary>
    public IReadOnlyList<string> Download() =>
        Files?.Invoke() ?? throw new InvalidOperationException($"{Name} is derived from another dataset; download its source instead.");

    /// <summary>Rows kept in memory.</summary>
    public static Dataset FromRows(IEnumerable<JsonObject> rows, string name = "rows")
    {
        var list = rows.ToList();
        return new Dataset(() => list.Select(r => (JsonObject)r.DeepClone()), name);
    }

    /// <summary>
    /// A file: JSON Lines, JSON, CSV, TSV, Parquet, text or source code (chosen by extension unless
    /// <see cref="ReadOptions.Format"/> is set), also inside .gz, .zip, .tar and .tar.gz.
    /// </summary>
    public static Dataset FromFile(string path, ReadOptions? options = null)
    {
        string full = Path.GetFullPath(path);
        if (!File.Exists(full))
        {
            throw new FileNotFoundException($"'{full}' does not exist.", full);
        }

        return new Dataset(() => DataFiles.Read(full, options ?? ReadOptions.Default), Path.GetFileName(full)) { Files = () => [full] };
    }

    /// <summary>Several files, one after another.</summary>
    public static Dataset FromFiles(IEnumerable<string> paths, ReadOptions? options = null, string? name = null)
    {
        var list = paths.Select(Path.GetFullPath).ToList();
        foreach (var file in list.Where(f => !File.Exists(f)))
        {
            throw new FileNotFoundException($"'{file}' does not exist.", file);
        }

        return new Dataset(() => list.SelectMany(f => DataFiles.Read(f, options ?? ReadOptions.Default)), name ?? $"{list.Count} files") { Files = () => list };
    }

    /// <summary>
    /// The data files in <paramref name="folder"/> and below whose names match <paramref name="pattern"/> (default: every
    /// file of a known format), in path order. Build output and dependency folders are skipped.
    /// </summary>
    public static Dataset FromFolder(string folder, string? pattern = null, ReadOptions? options = null)
    {
        string root = Path.GetFullPath(folder);
        if (!Directory.Exists(root))
        {
            throw new DirectoryNotFoundException($"'{root}' does not exist.");
        }

        var effective = (options ?? ReadOptions.Default) with { Root = (options ?? ReadOptions.Default).Root ?? root };
        return new Dataset(() => DataFiles.InFolder(root, pattern, effective).SelectMany(f => DataFiles.Read(f, effective)), Path.GetFileName(root))
        {
            Files = () => [.. DataFiles.InFolder(root, pattern, effective)],
        };
    }

    /// <summary>A file downloaded from <paramref name="url"/> (once: it is cached, see <see cref="Downloader"/>).</summary>
    public static Dataset FromUrl(string url, ReadOptions? options = null, Downloader? downloader = null, IReadOnlyDictionary<string, string>? headers = null)
    {
        _ = new Uri(url);                                           // a bad URL fails here, not on first use
        string Fetch() => (downloader ?? Downloader.Shared).Download(url, headers);
        return new Dataset(() => DataFiles.Read(Fetch(), options ?? ReadOptions.Default), url) { Files = () => [Fetch()] };
    }

    /// <summary>Datasets one after another.</summary>
    public static Dataset Concat(params Dataset[] datasets) =>
        new(() => datasets.SelectMany(d => d), string.Join(" + ", datasets.Select(d => d.Name)));

    /// <summary>
    /// Interleaves datasets at random, each row drawn from a source with probability proportional to its weight (as
    /// Hugging Face's interleave_datasets): <c>Mix([(chat, 0.6), (code, 0.3), (tools, 0.1)], seed: 1)</c>.
    /// </summary>
    public static Dataset Mix(IReadOnlyList<(Dataset Data, double Weight)> sources, int seed = 0, MixStop stop = MixStop.FirstExhausted)
    {
        if (sources.Count == 0 || sources.Any(s => s.Weight <= 0 || double.IsNaN(s.Weight)))
        {
            throw new ArgumentException("Mix needs at least one source, each with a positive weight.", nameof(sources));
        }

        return new Dataset(() => MixRows(sources, seed, stop), string.Join(" | ", sources.Select(s => $"{s.Data.Name}×{s.Weight:G3}")));
    }

    /// <summary>Each row changed by <paramref name="map"/>; rows mapped to null are dropped.</summary>
    public Dataset Select(Func<JsonObject, JsonObject?> map) => new(() => _rows().Select(map).OfType<JsonObject>(), Name);

    /// <summary>The rows <paramref name="keep"/> accepts.</summary>
    public Dataset Where(Func<JsonObject, bool> keep) => new(() => _rows().Where(keep), Name);

    /// <summary>The first <paramref name="count"/> rows.</summary>
    public Dataset Take(long count) => new(() => TakeRows(_rows(), count), Name);

    /// <summary>All rows after the first <paramref name="count"/>.</summary>
    public Dataset Skip(long count) => new(() => SkipRows(_rows(), count), Name);

    /// <summary>
    /// Rows in random order, drawn from a buffer of <paramref name="buffer"/> rows (a full shuffle when the dataset fits in
    /// the buffer; otherwise a local one, as streaming shuffles are). The same seed gives the same order.
    /// </summary>
    public Dataset Shuffle(int seed = 0, int buffer = 100_000) => new(() => ShuffleRows(_rows(), seed, buffer), Name);

    /// <summary>
    /// Rows without repeats: two rows are the same when <paramref name="columns"/> (default: all) hold exactly the same
    /// values. Keeps the first; remembers 16 bytes per distinct row. To count texts that differ only in form (case,
    /// spacing, letter forms) as the same, <see cref="Normalize(ITextNormalizer, string, string)"/> them into a key
    /// column first and deduplicate on that: <c>data.Normalize(rules, "text", into: "key").Deduplicate(["key", "label"])</c>.
    /// </summary>
    public Dataset Deduplicate(IReadOnlyList<string>? columns = null) =>
        new(() => DistinctRows(_rows(), columns), Name);

    /// <summary>
    /// The text values of <paramref name="columns"/> rewritten by <paramref name="normalizer"/>, in place (other values
    /// and missing columns are left as they are). A separate step from <see cref="Deduplicate"/> and the rest: the rules are
    /// the consumer's (<see cref="ITextNormalizer"/>).
    /// </summary>
    public Dataset Normalize(ITextNormalizer normalizer, params string[] columns)
    {
        ArgumentNullException.ThrowIfNull(normalizer);
        return Select(row =>
        {
            foreach (string column in columns)
            {
                if (row[column] is JsonValue value && value.TryGetValue<string>(out var text))
                {
                    row[column] = normalizer.Normalize(text);
                }
            }

            return row;
        });
    }

    /// <summary>
    /// <paramref name="column"/>'s text normalized into <paramref name="into"/>, keeping the original: a key to compare
    /// or group rows by (<see cref="Deduplicate"/>, a split key) while the text itself stays as written. Rows whose
    /// column is missing or not text get no key.
    /// </summary>
    public Dataset Normalize(ITextNormalizer normalizer, string column, string into)
    {
        ArgumentNullException.ThrowIfNull(normalizer);
        return Select(row =>
        {
            if (row[column] is JsonValue value && value.TryGetValue<string>(out var text))
            {
                row[into] = normalizer.Normalize(text);
            }

            return row;
        });
    }

    /// <summary>
    /// Splits into training and evaluation rows by a hash of each row's content (or of <paramref name="key"/>): the same
    /// row always lands on the same side, whatever the order, and no row is held in memory.
    /// </summary>
    public (Dataset Train, Dataset Evaluation) Split(double evaluationFraction, int seed = 0, Func<JsonObject, string>? key = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(evaluationFraction);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(evaluationFraction, 1);
        bool Evaluation(JsonObject row) => (key is null ? Fraction(row, seed) : Fraction(key(row), seed)) < evaluationFraction;
        return (new Dataset(() => _rows().Where(r => !Evaluation(r)), Name + " (train)"), new Dataset(() => _rows().Where(Evaluation), Name + " (evaluation)"));
    }

    /// <summary>The rows with only <paramref name="columns"/>.</summary>
    public Dataset SelectColumns(params string[] columns) => Select(row =>
    {
        var kept = new JsonObject();
        foreach (var column in columns)
        {
            if (row.TryGetPropertyValue(column, out var value))
            {
                row.Remove(column);
                kept[column] = value;
            }
        }

        return kept;
    });

    /// <summary>The rows without <paramref name="columns"/>.</summary>
    public Dataset RemoveColumns(params string[] columns) => Select(row =>
    {
        foreach (var column in columns)
        {
            row.Remove(column);
        }

        return row;
    });

    /// <summary>The rows with column <paramref name="from"/> renamed to <paramref name="to"/>.</summary>
    public Dataset RenameColumn(string from, string to) => Select(row =>
    {
        if (row.TryGetPropertyValue(from, out var value))
        {
            row.Remove(from);
            row[to] = value;
        }

        return row;
    });

    /// <summary>Counts the rows (reads the whole dataset).</summary>
    public long Count()
    {
        long count = 0;
        foreach (var _ in _rows())
        {
            count++;
        }

        return count;
    }

    /// <summary>The column names seen in the first <paramref name="sample"/> rows, in first-seen order.</summary>
    public IReadOnlyList<string> Columns(int sample = 100)
    {
        var columns = new List<string>();
        foreach (var row in _rows().Take(sample))
        {
            columns.AddRange(row.Select(p => p.Key).Where(k => !columns.Contains(k)));
        }

        return columns;
    }

    /// <summary>Writes the rows as JSON Lines (UTF-8, one object per line); returns the number written.</summary>
    public long WriteJsonLines(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        long count = 0;
        using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
        var options = new JsonWriterOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        using var buffer = new MemoryStream();
        foreach (var row in _rows())
        {
            buffer.SetLength(0);
            using (var json = new Utf8JsonWriter(buffer, options))
            {
                row.WriteTo(json);
            }

            writer.Write(Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length));
            writer.Write('\n');
            count++;
        }

        return count;
    }

    /// <inheritdoc />
    public IEnumerator<JsonObject> GetEnumerator() => _rows().GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public override string ToString() => Name;

    // A number in [0, 1) from a string, the same on every machine and run.
    // A row's place in [0, 1) for a seed: the 128-bit hash of the seed and the text (or the whole row), no copy of it.
    internal static double Fraction(string text, int seed)
    {
        var hash = Seeded(seed);
        hash.Add(1, text);
        return ((ulong)hash.Value >> 11) * (1.0 / (1UL << 53));
    }

    private static double Fraction(JsonObject row, int seed)
    {
        var hash = Seeded(seed);
        hash.AddRow(row);
        return ((ulong)hash.Value >> 11) * (1.0 / (1UL << 53));
    }

    private static RowHash Seeded(int seed)
    {
        var hash = new RowHash();
        hash.Add(3, [(char)(seed & 0xFFFF), (char)((uint)seed >> 16)]);
        return hash;
    }

    private static IEnumerable<JsonObject> TakeRows(IEnumerable<JsonObject> rows, long count)
    {
        if (count <= 0)
        {
            yield break;
        }

        long taken = 0;
        foreach (var row in rows)
        {
            yield return row;
            if (++taken >= count)
            {
                yield break;
            }
        }
    }

    private static IEnumerable<JsonObject> SkipRows(IEnumerable<JsonObject> rows, long count)
    {
        long seen = 0;
        foreach (var row in rows)
        {
            if (seen++ >= count)
            {
                yield return row;
            }
        }
    }

    private static IEnumerable<JsonObject> ShuffleRows(IEnumerable<JsonObject> rows, int seed, int buffer)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(buffer);
        var random = new Random(seed);
        var pool = new List<JsonObject>();
        foreach (var row in rows)
        {
            if (pool.Count < buffer)
            {
                pool.Add(row);
                continue;
            }

            int pick = random.Next(pool.Count);
            yield return pool[pick];
            pool[pick] = row;
        }

        // What is left: a Fisher-Yates shuffle.
        for (int i = pool.Count - 1; i >= 0; i--)
        {
            int j = random.Next(i + 1);
            yield return pool[j];
            pool[j] = pool[i];
        }
    }

    // One 128-bit hash per row (16 bytes per distinct row kept), taken straight from the values' characters: no joined
    // string, no UTF-8 copy. A text value hashes its characters; anything else its JSON; a missing column its own tag.
    private static IEnumerable<JsonObject> DistinctRows(IEnumerable<JsonObject> rows, IReadOnlyList<string>? columns)
    {
        var seen = new HashSet<UInt128>();
        foreach (var row in rows)
        {
            var hash = new RowHash();
            if (columns is null)
            {
                hash.AddRow(row);
            }
            else
            {
                for (int i = 0; i < columns.Count; i++)
                {
                    hash.AddValue(row[columns[i]]);
                }
            }

            if (seen.Add(hash.Value))
            {
                yield return row;
            }
        }
    }

    // Two independent 64-bit lanes over 8 bytes at a time, each value closed by its kind and length (so "ab","c" and
    // "a","bc" differ), then a full avalanche of both lanes. Not cryptographic: accidental collisions only (about one
    // chance in 2^64 among 2^32 distinct rows).
    private struct RowHash()
    {
        private const ulong P1 = 0x9E3779B185EBCA87UL, P2 = 0xC2B2AE3D27D4EB4FUL, P3 = 0x165667B19E3779F9UL, P4 = 0xD6E8FEB86659FD93UL;
        private ulong _a = 0x243F6A8885A308D3UL, _b = 0x13198A2E03707344UL;

        public void Add(byte kind, ReadOnlySpan<char> text)
        {
            var bytes = MemoryMarshal.AsBytes(text);
            int i = 0;
            for (; i + 8 <= bytes.Length; i += 8)
            {
                Mix(MemoryMarshal.Read<ulong>(bytes[i..]));
            }

            ulong tail = 0;
            for (int shift = 0; i < bytes.Length; i++, shift += 8)
            {
                tail |= (ulong)bytes[i] << shift;
            }

            Mix(tail);
            Mix(((ulong)kind << 56) | (uint)text.Length);
        }

        // Every property in order: its name, then its value (the same rows as their JSON text would give, without it).
        public void AddRow(JsonObject row)
        {
            foreach (var (name, node) in row)
            {
                Add(4, name);
                AddValue(node);
            }
        }

        // A text value hashes its characters; a missing or null value its own tag; anything else its JSON.
        public void AddValue(JsonNode? node)
        {
            if (node is null)
            {
                Add(0, "");
            }
            else if (node is JsonValue value && value.TryGetValue<string>(out var text))
            {
                Add(1, text);
            }
            else
            {
                Add(2, node.ToJsonString());
            }
        }

        public readonly UInt128 Value
        {
            get
            {
                ulong a = Avalanche(_a ^ BitOperations.RotateLeft(_b, 29)), b = Avalanche(_b ^ BitOperations.RotateLeft(_a, 43) ^ P3);
                return new UInt128(a, b);
            }
        }

        private void Mix(ulong v)
        {
            _a = BitOperations.RotateLeft(_a ^ (v * P1), 31) * P2;
            _b = BitOperations.RotateLeft(_b + (v * P3), 27) * P4 + _a;
        }

        private static ulong Avalanche(ulong h)
        {
            h ^= h >> 33;
            h *= 0xFF51AFD7ED558CCDUL;
            h ^= h >> 33;
            h *= 0xC4CEB9FE1A85EC53UL;
            return h ^ (h >> 33);
        }
    }

    private static IEnumerable<JsonObject> MixRows(IReadOnlyList<(Dataset Data, double Weight)> sources, int seed, MixStop stop)
    {
        var random = new Random(seed);
        var enumerators = sources.Select(s => s.Data.GetEnumerator()).ToList();
        var weights = sources.Select(s => s.Weight).ToArray();
        try
        {
            while (true)
            {
                double total = weights.Sum();
                if (total <= 0)
                {
                    yield break;
                }

                double pick = random.NextDouble() * total;
                int index = -1;
                for (int i = 0; i < weights.Length; i++)
                {
                    if (weights[i] <= 0)
                    {
                        continue;
                    }

                    index = i;                                      // the last live source if rounding leaves pick ≥ 0
                    if ((pick -= weights[i]) < 0)
                    {
                        break;
                    }
                }

                if (enumerators[index].MoveNext())
                {
                    yield return enumerators[index].Current;
                    continue;
                }

                if (stop == MixStop.FirstExhausted)
                {
                    yield break;
                }

                weights[index] = 0;
            }
        }
        finally
        {
            foreach (var enumerator in enumerators)
            {
                enumerator.Dispose();
            }
        }
    }
}
