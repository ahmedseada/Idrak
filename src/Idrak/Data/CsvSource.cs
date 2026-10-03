// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Idrak.Data;

/// <summary>
/// A numeric CSV file read lazily, for files too large to load: opening it reads the header, the first use of
/// <see cref="Count"/> or <see cref="Read"/> scans the file once for where each row starts (12 bytes per row are kept),
/// and every read parses one row from the file. Columns, blank lines and quoting follow <see cref="Dataset.LoadCsv"/>,
/// so the samples are those of the loaded dataset. <see cref="AsStream"/> reads the rows in order without the scan.
/// </summary>
/// <example>
/// <code>
/// using var rows = CsvSource.Open("big.csv", new CsvOptions { TargetColumns = ["price"] });
/// var loader = new DataLoader(rows, batchSize: 256, shuffle: true);
/// </code>
/// </example>
public sealed class CsvSource : ISampleSource, IDisposable
{
    private readonly SafeFileHandle _file;
    private readonly CsvOptions _options;
    private readonly string[] _header;
    private readonly int[] _featureColumns, _targetColumns;
    private readonly bool[] _parsed;
    private readonly long _bodyStart;
    private readonly int[] _featureShape, _targetShape;
    private readonly Lazy<(long[] Starts, int[] Lengths)> _rows;

    private CsvSource(string path, CsvOptions options)
    {
        Path = path;
        _options = options;
        _file = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        try
        {
            // The header: the first line that is not blank.
            long position = Bom();
            string? first = null;
            while (first is null && NextLine(ref position, out long start, out int length))
            {
                string line = Text(start, length);
                if (!string.IsNullOrWhiteSpace(line))
                {
                    first = line;
                    _bodyStart = options.HasHeader ? position : start;
                }
            }

            if (first is null)
            {
                throw new FormatException($"{path} is empty.");
            }

            string[] header = [.. first.Split(options.Delimiter).Select(s => s.Trim().Trim('"'))];
            if (!options.HasHeader)
            {
                header = [.. Enumerable.Range(0, header.Length).Select(i => i.ToString(CultureInfo.InvariantCulture))];
            }

            int Resolve(string column)
            {
                int index = Array.FindIndex(header, h => string.Equals(h, column, StringComparison.OrdinalIgnoreCase));
                if (index < 0 && int.TryParse(column, out int numeric) && numeric >= 0 && numeric < header.Length)
                {
                    index = numeric;
                }

                return index >= 0 ? index : throw new ArgumentException($"Column '{column}' not found in {path}. Columns: {string.Join(", ", header)}.");
            }

            _targetColumns = [.. options.TargetColumns.Select(Resolve)];
            if (_targetColumns.Length == 0)
            {
                throw new ArgumentException("At least one target column is required.", nameof(options));
            }

            var ignored = options.IgnoreColumns.Select(Resolve).Concat(_targetColumns).ToHashSet();
            _featureColumns = [.. Enumerable.Range(0, header.Length).Where(i => !ignored.Contains(i))];
            _parsed = new bool[header.Length];
            foreach (int i in _featureColumns.Concat(_targetColumns))
            {
                _parsed[i] = true;
            }

            _header = header;
            FeatureNames = [.. _featureColumns.Select(i => header[i])];
            TargetNames = [.. _targetColumns.Select(i => header[i])];
            _featureShape = [_featureColumns.Length];
            _targetShape = [_targetColumns.Length];
            _rows = new Lazy<(long[], int[])>(Scan, LazyThreadSafetyMode.ExecutionAndPublication);
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    /// <summary>Opens <paramref name="path"/> and reads its header.</summary>
    /// <exception cref="ArgumentException">A named column is not in the header; the message lists the columns.</exception>
    public static CsvSource Open(string path, CsvOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new CsvSource(path, options);
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>Input column names.</summary>
    public IReadOnlyList<string> FeatureNames { get; }

    /// <summary>Target column names.</summary>
    public IReadOnlyList<string> TargetNames { get; }

    /// <summary>Rows (the first use scans the file).</summary>
    public int Count => _rows.Value.Starts.Length;

    /// <summary>[feature columns].</summary>
    public IReadOnlyList<int> FeatureShape => _featureShape;

    /// <summary>[target columns].</summary>
    public IReadOnlyList<int> TargetShape => _targetShape;

    /// <summary>Parses row <paramref name="index"/> (0 is the first row after the header).</summary>
    /// <exception cref="FormatException">A value is not a number, or the row has another number of fields.</exception>
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        var (starts, lengths) = _rows.Value;
        if ((uint)index >= (uint)starts.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        Parse(Text(starts[index], lengths[index]), $"{Path} row {index + 1}", features, targets);
    }

    /// <summary>The rows in file order, read front to back on every pass (no scan, nothing kept).</summary>
    public ISampleStream AsStream() => new Stream(this);

    /// <summary>Closes the file.</summary>
    public void Dispose() => _file.Dispose();

    private void Parse(ReadOnlySpan<char> line, string where, Span<float> features, Span<float> targets)
    {
        Span<float> values = _header.Length <= 256 ? stackalloc float[_header.Length] : new float[_header.Length];
        int column = 0;
        foreach (var range in line.Split(_options.Delimiter))
        {
            if (column >= _header.Length)
            {
                throw new FormatException($"{where}: more than {_header.Length} fields.");
            }

            var field = line[range].Trim().Trim('"');
            if (_parsed[column] && !float.TryParse(field, NumberStyles.Float, _options.Culture, out values[column]))
            {
                throw new FormatException($"{where}, column '{_header[column]}': '{field}' is not a number.");
            }

            column++;
        }

        if (column != _header.Length)
        {
            throw new FormatException($"{where}: expected {_header.Length} fields, found {column}.");
        }

        for (int j = 0; j < _featureColumns.Length; j++)
        {
            features[j] = values[_featureColumns[j]];
        }

        for (int j = 0; j < _targetColumns.Length; j++)
        {
            targets[j] = values[_targetColumns[j]];
        }
    }

    // Where every row that is not blank starts, and its length, after the header: one sequential pass over the file.
    private (long[], int[]) Scan()
    {
        var starts = new List<long>();
        var lengths = new List<int>();
        using var stream = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
        stream.Position = _bodyStart;
        var buffer = new byte[1 << 20];
        long offset = _bodyStart, lineStart = _bodyStart;
        bool blank = true, high = false, afterReturn = false;
        void End(long end)
        {
            // A line of ASCII spaces is blank; one with other bytes too is decoded to check (char.IsWhiteSpace, as LoadCsv).
            if (!blank || high && !IsBlank(lineStart, checked((int)(end - lineStart))))
            {
                starts.Add(lineStart);
                lengths.Add(checked((int)(end - lineStart)));
            }

            blank = true;
            high = false;
        }

        int read;
        while ((read = stream.Read(buffer)) > 0)
        {
            int i = 0;
            while (i < read)
            {
                if (afterReturn)
                {
                    afterReturn = false;
                    if (buffer[i] == '\n')
                    {
                        lineStart = offset + ++i;
                        continue;
                    }
                }

                if (!blank)
                {
                    int found = buffer.AsSpan(i, read - i).IndexOfAny((byte)'\r', (byte)'\n');
                    if (found < 0)
                    {
                        break;
                    }

                    i += found;
                }

                byte b = buffer[i];
                if (b is (byte)'\r' or (byte)'\n')
                {
                    End(offset + i);
                    afterReturn = b == '\r';
                    lineStart = offset + ++i;
                    continue;
                }

                if (b >= 0x80)
                {
                    high = true;
                }
                else if (b is not ((byte)' ' or (byte)'\t' or 0x0B or 0x0C))
                {
                    blank = false;
                }

                i++;
            }

            offset += read;
        }

        if (offset > lineStart)
        {
            End(offset);
        }

        return ([.. starts], [.. lengths]);
    }

    // The 3 bytes of a UTF-8 byte order mark at the start, or 0.
    private long Bom()
    {
        Span<byte> head = stackalloc byte[3];
        int read = RandomAccess.Read(_file, head, 0);
        return read == 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF ? 3 : 0;
    }

    // The line at position: it ends at \r, \n or \r\n (as File.ReadAllLines splits); position moves past the break.
    private bool NextLine(ref long position, out long start, out int length)
    {
        long size = RandomAccess.GetLength(_file);
        start = position;
        length = 0;
        if (position >= size)
        {
            return false;
        }

        Span<byte> buffer = stackalloc byte[4096];
        while (true)
        {
            int read = RandomAccess.Read(_file, buffer, position);
            if (read == 0)
            {
                length = checked((int)(position - start));
                position = size;
                return true;
            }

            int found = buffer[..read].IndexOfAny((byte)'\r', (byte)'\n');
            if (found < 0)
            {
                position += read;
                continue;
            }

            long end = position + found;
            length = checked((int)(end - start));
            position = end + 1;
            if (buffer[found] == '\r')
            {
                Span<byte> next = stackalloc byte[1];
                if (RandomAccess.Read(_file, next, position) == 1 && next[0] == '\n')
                {
                    position++;
                }
            }

            return true;
        }
    }

    private bool IsBlank(long start, int length) => length == 0 || string.IsNullOrWhiteSpace(Text(start, length));

    private string Text(long start, int length)
    {
        if (length == 0)
        {
            return "";
        }

        var bytes = length <= 1024 ? stackalloc byte[length] : new byte[length];
        int read = 0;
        while (read < length)
        {
            int n = RandomAccess.Read(_file, bytes[read..], start + read);
            if (n == 0)
            {
                throw new EndOfStreamException($"{Path} ended while reading a row.");
            }

            read += n;
        }

        return Encoding.UTF8.GetString(bytes);
    }

    private sealed class Stream(CsvSource csv) : ISampleStream
    {
        public IReadOnlyList<int> FeatureShape => csv.FeatureShape;

        public IReadOnlyList<int> TargetShape => csv.TargetShape;

        public ISampleReader Open() => new Reader(csv);
    }

    private sealed class Reader : ISampleReader
    {
        private readonly CsvSource _csv;
        private readonly StreamReader _reader;
        private int _row;

        public Reader(CsvSource csv)
        {
            _csv = csv;
            _reader = new StreamReader(new FileStream(csv.Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan), Encoding.UTF8);
            _reader.BaseStream.Position = csv._bodyStart;
            _reader.DiscardBufferedData();
        }

        public bool Read(Span<float> features, Span<float> targets)
        {
            while (_reader.ReadLine() is { } line)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    _csv.Parse(line, $"{_csv.Path} row {++_row}", features, targets);
                    return true;
                }
            }

            return false;
        }

        public void Dispose() => _reader.Dispose();
    }
}
