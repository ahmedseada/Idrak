// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Data.Pdf;

/// <summary>A page's place in a tagged file's structure: the tag of each marked-content id (<c>/MCID</c>) on it.</summary>
internal readonly struct PdfMarks(Dictionary<(int Page, int Mcid), DocumentTag> structure, int page)
{
    /// <summary>The tag of marked-content id <paramref name="mcid"/> on the page, or null.</summary>
    public DocumentTag? Find(int mcid) => structure.TryGetValue((page, mcid), out var tag) ? tag : null;
}

/// <summary>
/// The content interpreter's part for tagged files (ISO 32000-2, 14.6-14.8): each run's place in
/// the document's structure, from the marked content it stands in (<c>BMC</c>, <c>BDC</c> with an <c>/MCID</c>, inline or
/// named in the resources' <c>/Properties</c>; <c>/Artifact</c> marks what is not content). An annotation's appearance is
/// read through <see cref="ReadForm"/>, as a form's widgets are.
/// </summary>
internal sealed partial class PdfContentReader
{
    // What an inline dictionary's bytes hold that the parser reads (an escape, a comment, a string, an array, an inner
    // dictionary): without them, a flat dictionary's numbers are read by the lexer.
    private static readonly SearchValues<byte> NotPlain = SearchValues.Create("#%()<>[]{}"u8);

    private readonly List<(int Depth, DocumentTag? Tag)> _marks = [];
    private PdfPooledList<DocumentTag?> _tags;                             // each run's tag, pooled (a field of its own, never readonly)
    private DocumentTag? _tag;

    /// <summary>A tagged file's structure for this page: the tag of a marked-content id; null when the file is not tagged.</summary>
    public PdfMarks? Marks { get; set; }

    // The tags at a page's end: the runs' go back to the pool, the marked content in force and the structure are let go.
    private void EndTags()
    {
        _marks.Clear();
        _tags.Return();
        _tag = null;
        Marks = null;
    }

    // BMC or BDC: the marked content's tag (an artifact, or the structure's place of its /MCID), kept until its EMC.
    private void Tagging(ReadOnlySpan<byte> data, bool properties, PdfDictionary? resources)
    {
        var tag = _tag;
        int at = _operandCount - (properties ? 2 : 1);
        if (at >= 0 && _operands[at].Kind == PdfToken.Name && tag is not { IsArtifact: true })
        {
            Span<byte> buffer = stackalloc byte[128];
            var name = NameBytes(data.Slice(_operands[at].Start, _operands[at].Length), buffer);
            if (name.SequenceEqual("Artifact"u8))
            {
                tag = DocumentTag.Artifact;
            }
            else if (properties && Marks is { } marks && Mcid(data, _operands[_operandCount - 1], resources) is { } mcid && marks.Find(mcid) is { } found)
            {
                tag = found;
            }
        }

        _marks.Add((_marked, tag));
        _tag = tag;
    }

    // EMC: the marked content that ends gives back the tag in force before it.
    private void Untag()
    {
        if (_marks.Count > 0 && _marks[^1].Depth == _marked)
        {
            _marks.RemoveAt(_marks.Count - 1);
            _tag = _marks.Count > 0 ? _marks[^1].Tag : null;
        }
    }

    // A BDC's /MCID: from its properties written inline, or named in the resources' /Properties.
    private int? Mcid(ReadOnlySpan<byte> data, Operand properties, PdfDictionary? resources)
    {
        try
        {
            PdfDictionary? dictionary = null;
            if (properties.Kind == PdfToken.Dictionary)
            {
                var raw = data.Slice(properties.Start, properties.Length);
                if (raw.IndexOf("MCID"u8) >= 0)
                {
                    if (PlainNumber(raw, "MCID"u8, out double number))
                    {
                        // As the parser's integer would be: a whole number from 0 to int.MaxValue (NaN: none, or not a number).
                        return number >= 0 && number <= int.MaxValue && number == Math.Floor(number) ? (int)number : null;
                    }

                    var parser = new PdfParser(raw, _file.MaxDepth, complete: true);
                    dictionary = parser.ReadValue().AsDictionary;
                }
            }
            else if (properties.Kind == PdfToken.Name)
            {
                Span<byte> buffer = stackalloc byte[128];
                dictionary = _file.Dictionary(Lookup(resources, "Properties"u8, NameBytes(data.Slice(properties.Start, properties.Length), buffer)));
            }

            return dictionary is not null && _file.Resolve(dictionary["MCID"u8]).TryInteger(out long mcid) && mcid is >= 0 and <= int.MaxValue ? (int)mcid : null;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: a marked-content span's properties: {e.Message}");
            return null;
        }
    }

    // The value of `key` in a dictionary written in the content (`<< ... >>`, as the lexer gives it) when the dictionary is
    // flat and plain, read by the lexer with nothing made: only names, numbers, true, false and null (no reference, string,
    // array, inner dictionary, comment or escape, which the parser reads, its errors the page's issues), each key a name
    // with one value; a key given twice keeps its last value, as the parser keeps it. `number` is NaN when the key's
    // value is not a number or the key is not there. False when the dictionary is not that plain: the parser reads it.
    private static bool PlainNumber(ReadOnlySpan<byte> raw, ReadOnlySpan<byte> key, out double number)
    {
        number = double.NaN;
        if (raw.Length < 4 || !raw.StartsWith("<<"u8) || !raw.EndsWith(">>"u8))
        {
            return false;
        }

        var inside = raw[2..^2];
        if (inside.IndexOfAny(NotPlain) >= 0)
        {
            return false;
        }

        var lexer = new PdfContentLexer(inside);
        bool atKey = true, wanted = false;
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out double value);
            if (token == PdfToken.End)
            {
                return atKey;                                                  // a key without its value: the parser's to say
            }

            var word = inside.Slice(start, length);
            if (atKey)
            {
                if (token != PdfToken.Name)
                {
                    return false;
                }

                wanted = word.SequenceEqual(key);
            }
            else if (token == PdfToken.Number)
            {
                number = wanted ? value : number;
            }
            else if (token == PdfToken.Name || (token == PdfToken.Keyword && (word.SequenceEqual("true"u8) || word.SequenceEqual("false"u8) || word.SequenceEqual("null"u8))))
            {
                number = wanted ? double.NaN : number;
            }
            else
            {
                return false;                                                  // a reference's R, or a word the parser refuses
            }

            atKey = !atKey;
        }
    }
}
