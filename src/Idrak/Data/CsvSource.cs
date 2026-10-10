// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
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
                if (index < 0 && int.TryParse(column, NumberStyles.Integer, CultureInfo.InvariantCulture, out int numeric) && numeric >= 0 && numeric < header.Length)
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

        int length = lengths[index];
        byte[]? rented = length <= 1024 ? null : ArrayPool<byte>.Shared.Rent(length);
        try
        {
            Span<byte> bytes = rented is null ? stackalloc byte[length] : rented.AsSpan(0, length);
            ReadAt(starts[index], bytes);
            Parse(bytes, index + 1, features, targets);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    /// <summary>The rows in file order, read front to back on every pass (no scan, nothing kept).</summary>
    public ISampleStream AsStream() => new Stream(this);

    /// <summary>Closes the file.</summary>
    public void Dispose() => _file.Dispose();

    // Row `row` (counted from 1) parsed from its UTF-8 bytes; the error messages name it.
    private void Parse(ReadOnlySpan<byte> line, int row, Span<float> features, Span<float> targets)
    {
        float[]? rented = _header.Length <= 256 ? null : ArrayPool<float>.Shared.Rent(_header.Length);
        try
        {
            Span<float> values = rented is null ? stackalloc float[_header.Length] : rented.AsSpan(0, _header.Length);
            CsvRow.Parse(line, _options.Delimiter, _options.Culture, _parsed, _header, values, Path, "row", row);
            for (int j = 0; j < _featureColumns.Length; j++)
            {
                features[j] = values[_featureColumns[j]];
            }

            for (int j = 0; j < _targetColumns.Length; j++)
            {
                targets[j] = values[_targetColumns[j]];
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
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

        Span<byte> buffer = stackalloc byte[1024];
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
                var next = buffer[..1];                                            // the line is found: the buffer is free
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
        ReadAt(start, bytes);
        return Encoding.UTF8.GetString(bytes);
    }

    // Fills `bytes` from the file at `start`.
    private void ReadAt(long start, Span<byte> bytes)
    {
        int read = 0;
        while (read < bytes.Length)
        {
            int n = RandomAccess.Read(_file, bytes[read..], start + read);
            if (n == 0)
            {
                throw new EndOfStreamException($"{Path} ended while reading a row.");
            }

            read += n;
        }
    }

    private sealed class Stream(CsvSource csv) : ISampleStream
    {
        public IReadOnlyList<int> FeatureShape => csv.FeatureShape;

        public IReadOnlyList<int> TargetShape => csv.TargetShape;

        public ISampleReader Open() => new Reader(csv);
    }

    // The rows front to back as UTF-8 bytes, read in blocks into a pooled buffer (grown for a row longer than it).
    private sealed class Reader : ISampleReader
    {
        private readonly CsvSource _csv;
        private readonly FileStream _file;
        private byte[]? _buffer = ArrayPool<byte>.Shared.Rent(1 << 16);
        private int _start, _end, _row;
        private bool _ended, _afterReturn;

        public Reader(CsvSource csv)
        {
            _csv = csv;
            _file = new FileStream(csv.Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
            _file.Position = csv._bodyStart;
        }

        public bool Read(Span<float> features, Span<float> targets)
        {
            while (NextLine(out int start, out int length))
            {
                var line = _buffer.AsSpan(start, length);
                if (!CsvRow.IsBlank(line))
                {
                    _csv.Parse(line, ++_row, features, targets);
                    return true;
                }
            }

            return false;
        }

        public void Dispose()
        {
            _file.Dispose();
            if (_buffer is { } buffer)
            {
                _buffer = null;
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        // The next line in the buffer: it ends at \r, \n or \r\n, as StreamReader.ReadLine splits them (the last one may
        // end at the end of the file); more of the file is read as needed.
        private bool NextLine(out int start, out int length)
        {
            ObjectDisposedException.ThrowIf(_buffer is null, this);
            int searched = 0;                                                      // bytes after _start known to hold no break
            while (true)
            {
                if (_afterReturn)
                {
                    if (_start == _end && !_ended)
                    {
                        Fill();
                        continue;
                    }

                    _afterReturn = false;
                    if (_start < _end && _buffer[_start] == '\n')
                    {
                        _start++;
                    }
                }

                int found = _buffer.AsSpan(_start + searched, _end - _start - searched).IndexOfAny((byte)'\r', (byte)'\n');
                if (found >= 0)
                {
                    start = _start;
                    length = searched + found;
                    _afterReturn = _buffer[start + length] == '\r';
                    _start = start + length + 1;
                    return true;
                }

                searched = _end - _start;
                if (_ended)
                {
                    start = _start;
                    length = _end - _start;
                    _start = _end;
                    return length > 0;
                }

                Fill();
            }
        }

        // Moves the unfinished line to the front of the buffer (a larger one when it fills the buffer) and reads more.
        private void Fill()
        {
            int kept = _end - _start;
            if (_start > 0)
            {
                _buffer.AsSpan(_start, kept).CopyTo(_buffer);
            }
            else if (kept == _buffer!.Length)
            {
                var larger = ArrayPool<byte>.Shared.Rent(2 * _buffer.Length);
                _buffer.AsSpan(0, kept).CopyTo(larger);
                ArrayPool<byte>.Shared.Return(_buffer);
                _buffer = larger;
            }

            _start = 0;
            _end = kept;
            int read = _file.Read(_buffer!, _end, _buffer!.Length - _end);
            if (read == 0)
            {
                _ended = true;
            }
            else
            {
                _end += read;
            }
        }
    }
}

// One CSV row split at the delimiter and parsed, for Dataset.LoadCsv and CsvSource alike: each field trimmed of white
// space and then of quotes, the parsed columns read as numbers in the options' culture. A row of ASCII bytes is parsed
// from its bytes; any other row is decoded first, so both read what the row's text says.
internal static class CsvRow
{
    /// <summary>Parses <paramref name="line"/> into <paramref name="values"/> (one per column); errors name "{source} {unit} {number}".</summary>
    public static void Parse(ReadOnlySpan<char> line, char delimiter, IFormatProvider culture, bool[] parsed, string[] header, Span<float> values,
        string source, string unit, int number)
    {
        int column = 0;
        foreach (var range in line.Split(delimiter))
        {
            if (column >= values.Length)
            {
                throw TooMany(source, unit, number, values.Length);
            }

            var field = line[range].Trim().Trim('"');
            if (parsed[column] && !float.TryParse(field, NumberStyles.Float, culture, out values[column]))
            {
                throw NotANumber(source, unit, number, header[column], field.ToString());
            }

            column++;
        }

        if (column != values.Length)
        {
            throw Fields(source, unit, number, values.Length, column);
        }
    }

    /// <summary>The same for a row of UTF-8 bytes.</summary>
    public static void Parse(ReadOnlySpan<byte> line, char delimiter, IFormatProvider culture, bool[] parsed, string[] header, Span<float> values,
        string source, string unit, int number)
    {
        if (delimiter > 0x7F || !Ascii.IsValid(line))
        {
            char[] text = ArrayPool<char>.Shared.Rent(Encoding.UTF8.GetMaxCharCount(line.Length));
            try
            {
                int length = Encoding.UTF8.GetChars(line, text);
                Parse(text.AsSpan(0, length), delimiter, culture, parsed, header, values, source, unit, number);
            }
            finally
            {
                ArrayPool<char>.Shared.Return(text);
            }

            return;
        }

        int column = 0;
        foreach (var range in line.Split((byte)delimiter))
        {
            if (column >= values.Length)
            {
                throw TooMany(source, unit, number, values.Length);
            }

            var field = TrimWhiteSpace(line[range]).Trim((byte)'"');
            if (parsed[column] && !float.TryParse(field, NumberStyles.Float, culture, out values[column]))
            {
                throw NotANumber(source, unit, number, header[column], Encoding.ASCII.GetString(field));
            }

            column++;
        }

        if (column != values.Length)
        {
            throw Fields(source, unit, number, values.Length, column);
        }
    }

    /// <summary>Whether a row of UTF-8 bytes is empty or white space only (as <see cref="MemoryExtensions.IsWhiteSpace"/> on its text).</summary>
    public static bool IsBlank(ReadOnlySpan<byte> line)
    {
        int i = 0;
        while (i < line.Length && line[i] < 0x80 && char.IsWhiteSpace((char)line[i]))
        {
            i++;
        }

        if (i == line.Length)
        {
            return true;
        }

        if (line[i] < 0x80)
        {
            return false;                                                          // an ASCII character that is not white space
        }

        char[] text = ArrayPool<char>.Shared.Rent(Encoding.UTF8.GetMaxCharCount(line.Length - i));
        try
        {
            return text.AsSpan(0, Encoding.UTF8.GetChars(line[i..], text)).IsWhiteSpace();
        }
        finally
        {
            ArrayPool<char>.Shared.Return(text);
        }
    }

    // An ASCII field without the white space at either end (char.IsWhiteSpace, as string.Trim).
    private static ReadOnlySpan<byte> TrimWhiteSpace(ReadOnlySpan<byte> field)
    {
        int start = 0, end = field.Length;
        while (start < end && char.IsWhiteSpace((char)field[start]))
        {
            start++;
        }

        while (end > start && char.IsWhiteSpace((char)field[end - 1]))
        {
            end--;
        }

        return field[start..end];
    }

    private static FormatException TooMany(string source, string unit, int number, int columns) =>
        new($"{source} {unit} {number}: more than {columns} fields.");

    private static FormatException NotANumber(string source, string unit, int number, string column, string field) =>
        new($"{source} {unit} {number}, column '{column}': '{field}' is not a number.");

    private static FormatException Fields(string source, string unit, int number, int columns, int found) =>
        new($"{source} {unit} {number}: expected {columns} fields, found {found}.");
}
