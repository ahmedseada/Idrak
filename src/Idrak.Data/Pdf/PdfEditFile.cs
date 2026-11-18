// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// The file writer's update side (ISO 32000-2, 7.5.6): an incremental update appended to a file whose bytes the destination
/// already holds. Its offsets count on past them; new objects are numbered from the file's <c>/Size</c>; the file's own
/// objects are written again only when <see cref="Reuse"/> names them (under their number and generation); and the update
/// ends with one cross-reference section listing only what it wrote (a classic table, or a stream over object streams),
/// pointing back to the file's last section by <c>/Prev</c>, or, for a file whose cross-reference was rebuilt by scanning,
/// listing every object so that it stands alone.
/// </summary>
internal sealed partial class PdfFileWriter
{
    // An object of the file before the update that the update does not write again.
    private const long Untouched = long.MinValue + 1;

    // The first number the update gives (the objects below it are the file's).
    private int _first;

    private PdfFileWriter(Stream destination, int level, bool packing, long prefix, long shift, int first)
    {
        (_destination, _level, _written, _shift, _first) = (destination, level, prefix, shift, first);
        _packed = packing ? new PdfSyntax(Block + 4096) : null;
        _entries = new long[Math.Max(64, first + 16)];
        _entries.AsSpan(0, first).Fill(Untouched);
        _count = first;
    }

    /// <summary>
    /// A writer of an incremental update to a file of <paramref name="prefix"/> bytes the destination already holds: offsets
    /// counted past them, less <paramref name="shift"/> (the bytes before the file's header, when its offsets count from
    /// it); new objects numbered from <paramref name="first"/> (the file's <c>/Size</c>); streams deflated at
    /// <paramref name="level"/>; small objects packed into object streams when <paramref name="packing"/>.
    /// </summary>
    public static PdfFileWriter Append(Stream destination, long prefix, long shift, int first, int level, bool packing) =>
        new(destination, level, packing, prefix, shift, Math.Max(1, first));

    /// <summary>Marks the file's object <paramref name="number"/> to be written again by the update, with its <paramref name="generation"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The number is not one of the file's.</exception>
    public void Reuse(int number, int generation)
    {
        if (number <= 0 || number >= _first)
        {
            throw new ArgumentOutOfRangeException(nameof(number), number, $"Object {number} is not one of the file's (1 to {_first - 1}).");
        }

        if (_entries[number] == Untouched)
        {
            _entries[number] = NotWritten;
        }

        if (generation != 0)
        {
            (_generations ??= [])[number] = generation;
        }
    }

    /// <summary>
    /// Ends the update: its cross-reference section (a classic table when <paramref name="classic"/>, else a stream) lists
    /// every object it wrote, and <paramref name="carried"/> those it did not (a section standing alone: object 0 is listed
    /// too); the trailer has <c>/Size</c> (<paramref name="size"/> at least), <c>/Prev</c> when <paramref name="previous"/>
    /// is 0 or more, and what <paramref name="trailer"/> writes (<c>/Root</c>, <c>/Info</c>, <c>/Encrypt</c>, <c>/ID</c>);
    /// then <c>startxref</c> and <c>%%EOF</c>.
    /// </summary>
    /// <exception cref="InvalidOperationException">An object named or numbered was never written (a fault of the editor), or a classic table would list an object in an object stream.</exception>
    public void FinishUpdate(bool classic, long previous, long size, IReadOnlyList<(int Number, PdfEntry Entry)> carried, Action<PdfSyntax> trailer)
    {
        FlushPacked();
        var rows = new List<(int Number, byte Type, long Second, int Third)>();
        for (int n = 1; n < _count; n++)
        {
            long e = _entries[n];
            if (e == Untouched)
            {
                continue;
            }

            if (e == NotWritten)
            {
                throw new InvalidOperationException($"The PDF editor named object {n} for its update and never wrote it.");
            }

            rows.Add(e >= 0 ? (n, (byte)1, e, Generation(n)) : (n, (byte)2, (-e - 1) / 65536, (int)((-e - 1) % 65536)));
        }

        if (carried.Count > 0)
        {
            rows.Add((0, 0, 0, 65535));                                       // the head of the free list, in a section that stands alone
            foreach (var (number, entry) in carried)
            {
                if (number > 0 && number < _count && _entries[number] == Untouched && entry.Kind is PdfEntryKind.InUse or PdfEntryKind.Compressed)
                {
                    rows.Add((number, (byte)(entry.Kind == PdfEntryKind.InUse ? 1 : 2), entry.Offset, entry.Generation));
                }
            }
        }

        if (classic)
        {
            if (rows.Exists(r => r.Type == 2))
            {
                throw new InvalidOperationException("A classic cross-reference table cannot list objects held in object streams.");
            }

            rows.Sort((a, b) => a.Number.CompareTo(b.Number));
            long at = Position;
            _pending.Raw("xref\n"u8);
            Span<byte> entry = stackalloc byte[20];
            for (int i = 0; i < rows.Count;)
            {
                int run = Run(rows, i);
                _pending.Integer(rows[i].Number).Space().Integer(run).Line();
                for (int k = i; k < i + run; k++)
                {
                    var r = rows[k];
                    r.Second.TryFormat(entry, out _, "D10", CultureInfo.InvariantCulture);
                    entry[10] = (byte)' ';
                    r.Third.TryFormat(entry[11..], out _, "D5", CultureInfo.InvariantCulture);
                    entry[16] = (byte)' ';
                    entry[17] = r.Type == 0 ? (byte)'f' : (byte)'n';
                    (entry[18], entry[19]) = ((byte)'\r', (byte)'\n');
                    _pending.Raw(entry);
                    if (_pending.Length >= Block)
                    {
                        Flush();
                    }
                }

                i += run;
            }

            _pending.Raw("trailer\n<</Size "u8).Integer(Math.Max(size, _count));
            if (previous >= 0)
            {
                _pending.Raw("/Prev "u8).Integer(previous);
            }

            trailer(_pending);
            _pending.Raw(">>\nstartxref\n"u8).Integer(at - _shift).Raw("\n%%EOF\n"u8);
            Flush();
            return;
        }

        int self = Allocate();
        long start = Position;
        _entries[self] = start - _shift;
        rows.Add((self, 1, start - _shift, 0));
        rows.Sort((a, b) => a.Number.CompareTo(b.Number));
        long largest = 0;
        foreach (var r in rows)
        {
            largest = Math.Max(largest, r.Second);
        }

        int width = 1;
        while (width < 8 && largest >> (8 * width) != 0)
        {
            width++;
        }

        int row = 1 + width + 2;
        using var data = new PdfSyntax(row * rows.Count);
        var bytes = data.Buffer.Reserve(row * rows.Count);
        Span<byte> field = stackalloc byte[8];
        for (int k = 0; k < rows.Count; k++)
        {
            var r = bytes.Slice(k * row, row);
            r[0] = rows[k].Type;
            BinaryPrimitives.WriteInt64BigEndian(field, rows[k].Second);
            field[(8 - width)..].CopyTo(r[1..]);
            BinaryPrimitives.WriteUInt16BigEndian(r[(1 + width)..], (ushort)rows[k].Third);
        }

        data.Buffer.Advance(row * rows.Count);
        _xrefNumber = self;                                                    // a cross-reference stream is never encrypted
        var s = BeginStream(self);
        s.Raw("/Type/XRef/W[1 "u8).Integer(width).Raw(" 2]/Index["u8);
        for (int i = 0; i < rows.Count;)
        {
            int run = Run(rows, i);
            if (i > 0)
            {
                s.Space();
            }

            s.Integer(rows[i].Number).Space().Integer(run);
            i += run;
        }

        s.Raw("]/Size "u8).Integer(Math.Max(size, _count));
        if (previous >= 0)
        {
            s.Raw("/Prev "u8).Integer(previous);
        }

        trailer(s);
        EndStream(data.Span, deflate: true);
        _pending.Raw("startxref\n"u8).Integer(start - _shift).Raw("\n%%EOF\n"u8);
        Flush();
    }

    // How many rows from `i` list consecutive numbers (one subsection).
    private static int Run(List<(int Number, byte Type, long Second, int Third)> rows, int i)
    {
        int run = 1;
        while (i + run < rows.Count && rows[i + run].Number == rows[i].Number + run)
        {
            run++;
        }

        return run;
    }
}
