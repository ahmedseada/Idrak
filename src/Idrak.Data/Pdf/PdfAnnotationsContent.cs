// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// The content interpreter's part for tagged files (ISO 32000-2, 14.6-14.8): each run's place in
/// the document's structure, from the marked content it stands in (<c>BMC</c>, <c>BDC</c> with an <c>/MCID</c>, inline or
/// named in the resources' <c>/Properties</c>; <c>/Artifact</c> marks what is not content). An annotation's appearance is
/// read through <see cref="ReadForm"/>, as a form's widgets are.
/// </summary>
internal sealed partial class PdfContentReader
{
    private readonly List<(int Depth, DocumentTag? Tag)> _marks = [];
    private readonly List<DocumentTag?> _tags = [];
    private DocumentTag? _tag;

    /// <summary>A tagged file's structure for this page: the tag of a marked-content id; null when the file is not tagged.</summary>
    public Func<int, DocumentTag?>? Marks { get; set; }

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
            else if (properties && Marks is { } marks && Mcid(data, _operands[_operandCount - 1], resources) is { } mcid && marks(mcid) is { } found)
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
}
