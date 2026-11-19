// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Idrak.Data.Pdf;

/// <summary>
/// One signature of a PDF document as read (ISO 32000-2, 12.8): its field, its signature dictionary, the byte range checked
/// against the file, what <c>/Contents</c> holds (CMS, a timestamp token, or the legacy PKCS #1 signature), its timestamp,
/// its permission and what changed after it. What <see cref="PdfSignatureCheck"/> checks; immutable.
/// </summary>
internal sealed class PdfSignatureEntry
{
    public required PdfSignatureContext Context { get; init; }

    public required string Field { get; init; }

    public required PdfDictionary Dictionary { get; init; }

    public required string SubFilter { get; init; }

    public DocumentSignatureKind Kind { get; set; }

    /// <summary>The signature's bytes (the reserved zeros after it included).</summary>
    public required byte[] Contents { get; init; }

    /// <summary>The byte range as given (offset and length pairs).</summary>
    public required long[] ByteRange { get; init; }

    /// <summary>Why the byte range is not a signature's (it does not start at 0, its hole is not the <c>/Contents</c> string, ...), or null.</summary>
    public string? RangeProblem { get; init; }

    /// <summary>Where the signed bytes end (the end of the revision signed).</summary>
    public long SignedEnd { get; init; }

    public bool Whole { get; init; }

    /// <summary>The CMS it holds (null for the legacy PKCS #1 form, or when it cannot be read: <see cref="ContentsProblem"/>).</summary>
    public CmsSignedData? Cms { get; init; }

    public CmsSignerInfo? Signer { get; init; }

    public CmsCertificate? SignerCertificate { get; init; }

    /// <summary>The certificates the legacy PKCS #1 form gives in <c>/Cert</c>.</summary>
    public IReadOnlyList<CmsCertificate> CertEntry { get; init; } = [];

    /// <summary>Why <c>/Contents</c> cannot be read, or null.</summary>
    public string? ContentsProblem { get; init; }

    /// <summary>Its timestamp (the signature's own, or the document timestamp it is), or null; why it cannot be read.</summary>
    public CmsTimestamp? Timestamp { get; init; }

    public string? TimestampProblem { get; init; }

    /// <summary>The DocMDP permission that applies to it (1 no changes, 2 form filling, 3 annotations), or null for none.</summary>
    public int? Permission { get; init; }

    /// <summary>Whether it is the document's certification signature.</summary>
    public bool Certification { get; init; }

    /// <summary>The fields its FieldMDP lock names (all of them when <see cref="LocksAll"/>), and those it excludes.</summary>
    public IReadOnlyList<string> Locked { get; init; } = [];

    public bool LocksAll { get; init; }

    public IReadOnlyList<string> LockExcluded { get; init; } = [];

    /// <summary>What changed after it (found at reading).</summary>
    public PdfChangeReport Changes { get; set; } = PdfChangeReport.Nothing;

    public bool IsDocumentTimestamp => SubFilter == "ETSI.RFC3161";
}

/// <summary>
/// What every signature of a document shares: the document, its validation data (the catalog's <c>/DSS</c>: certificates,
/// revocation lists and OCSP responses, read once when a check first needs them).
/// </summary>
internal sealed class PdfSignatureContext
{
    private readonly Lazy<PdfValidationData> _dss;

    public PdfSignatureContext(PdfDocument document)
    {
        Document = document;
        _dss = new(() => PdfValidationData.Read(document), LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public PdfDocument Document { get; }

    public PdfFile File => Document.File;

    public PdfValidationData Dss => _dss.Value;
}

/// <summary>The document's validation data (PAdES: the DSS), each item parsed once; what cannot be read is listed apart.</summary>
internal sealed class PdfValidationData
{
    public IReadOnlyList<CmsCertificate> Certificates { get; init; } = [];

    public IReadOnlyList<CmsCrl> Crls { get; init; } = [];

    public IReadOnlyList<CmsOcsp> Ocsps { get; init; } = [];

    /// <summary>The VRI keys (upper-case hex SHA-1 of the signatures they validate).</summary>
    public IReadOnlySet<string> Vri { get; init; } = new HashSet<string>();

    public IReadOnlyList<string> Problems { get; init; } = [];

    public bool Present { get; init; }

    public static PdfValidationData Read(PdfDocument document)
    {
        var file = document.File;
        var certificates = new List<CmsCertificate>();
        var crls = new List<CmsCrl>();
        var ocsps = new List<CmsOcsp>();
        var vri = new HashSet<string>(StringComparer.Ordinal);
        var problems = new List<string>();
        var seen = new HashSet<int>();
        PdfDictionary? dss;
        try
        {
            dss = file.Dictionary(file.Dictionary(file.Trailer["Root"u8])?["DSS"u8] ?? PdfValue.Null);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return new PdfValidationData { Problems = [$"the DSS cannot be read ({e.Message})"] };
        }

        if (dss is null)
        {
            return new PdfValidationData();
        }

        void Streams(PdfDictionary from, ReadOnlySpan<byte> key, Action<byte[]> add)
        {
            PdfArray? items;
            try
            {
                items = file.Resolve(from[key]).AsArray;
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                problems.Add($"the DSS's /{System.Text.Encoding.ASCII.GetString(key)} cannot be read ({e.Message})");
                return;
            }

            foreach (var item in items?.Items ?? [])
            {
                if (item.IsReference && !seen.Add(item.AsReference.Number))
                {
                    continue;                                                  // the same stream listed again (in a VRI entry)
                }

                try
                {
                    if (file.Resolve(item).AsStream is { } stream)
                    {
                        using var decoded = file.Decode(stream);
                        add(decoded.Span.ToArray());
                    }
                }
                catch (InvalidDataException e) when (!PdfLimit.Is(e))
                {
                    problems.Add(e.Message);
                }
            }
        }

        void Add(byte[] data, string what, Action<byte[]> parse)
        {
            try
            {
                parse(data);
            }
            catch (InvalidDataException e)
            {
                problems.Add($"a {what} in the DSS cannot be read ({e.Message})");
            }
        }

        void All(PdfDictionary from, bool vri)
        {
            Streams(from, vri ? "Cert"u8 : "Certs"u8, d => Add(d, "certificate", x => certificates.Add(CmsCertificate.Parse(x))));
            Streams(from, vri ? "CRL"u8 : "CRLs"u8, d => Add(d, "revocation list", x => crls.Add(CmsCrl.Parse(x))));
            Streams(from, vri ? "OCSP"u8 : "OCSPs"u8, d => Add(d, "OCSP response", x => ocsps.Add(CmsOcsp.Parse(x))));
        }

        All(dss, vri: false);
        if (file.Dictionary(dss["VRI"u8]) is { } entries)
        {
            for (int i = 0; i < entries.Count; i++)
            {
                vri.Add(entries.KeyAt(i).ToString().ToUpperInvariant());
                if (file.Dictionary(entries.ValueAt(i)) is { } entry)
                {
                    All(entry, vri: true);
                }
            }
        }

        return new PdfValidationData { Certificates = certificates, Crls = crls, Ocsps = ocsps, Vri = vri, Problems = problems, Present = true };
    }
}

/// <summary>
/// The signatures of a PDF document: every signature field of the AcroForm whose value is a signature dictionary, read and
/// made a <see cref="DocumentSignature"/> (in the order they were made: by where their signed bytes end). Nothing is trusted
/// here: a signature's check is <see cref="PdfSignatureCheck"/>'s, against the trust the app gives.
/// </summary>
internal static class PdfSignatures
{
    public static IReadOnlyList<DocumentSignature> Read(PdfDocument document)
    {
        var context = new PdfSignatureContext(document);
        var file = document.File;
        var catalog = file.Dictionary(file.Trailer["Root"u8]);
        var form = catalog is null ? null : file.Dictionary(catalog["AcroForm"u8]);
        if (form is null)
        {
            return [];
        }

        var found = new List<(string Name, PdfDictionary Field, PdfDictionary Signature, PdfValue Value)>();
        PdfFormFields.Walk(file, form, (name, _, field, type) =>
        {
            if (type == "Sig" && !field["V"u8].IsNull && file.Dictionary(field["V"u8]) is { } signature)
            {
                found.Add((name, field, signature, field["V"u8]));
            }
        });

        // The certification signature (the catalog's /Perms /DocMDP) and its permission apply to every later signature.
        PdfValue certification = PdfValue.Null;
        if (catalog is not null && file.Dictionary(catalog["Perms"u8]) is { } perms)
        {
            certification = perms["DocMDP"u8];
        }

        var entries = new List<PdfSignatureEntry>();
        foreach (var (name, field, signature, value) in found)
        {
            bool certifies = certification.IsReference && value.IsReference && certification.AsReference == value.AsReference;
            entries.Add(Entry(context, name, field, signature, value, certifies));
        }

        entries.Sort((a, b) => a.SignedEnd.CompareTo(b.SignedEnd));
        var certifying = entries.FirstOrDefault(e => e.Certification);
        var result = new List<DocumentSignature>(entries.Count);
        for (int k = 0; k < entries.Count; k++)
        {
            var entry = entries[k];
            int? permission = entry.Permission;
            if (permission is null && certifying is not null && certifying.SignedEnd <= entry.SignedEnd)
            {
                permission = certifying.Permission;
            }

            entry.Kind = Kind(entry, entries, context);
            entry.Changes = PdfSignatureChanges.Analyse(entry, permission);
            var e = entry;
            result.Add(new DocumentSignature(entry.Field, entry.Kind, options => PdfSignatureCheck.Run(e, options))
            {
                KindName = entry.SubFilter,
                Signer = Certificate(entry.SignerCertificate),
                SignedAt = file.Resolve(entry.Dictionary["M"u8]).AsString is { } m ? PdfText.Date(m.Bytes) ?? entry.Signer?.SigningTime() : entry.Signer?.SigningTime(),
                TimestampedAt = entry.Timestamp?.Time,
                Reason = Text(file, entry.Dictionary["Reason"u8]),
                Location = Text(file, entry.Dictionary["Location"u8]),
                ByteRange = entry.ByteRange,
                Changes = entry.Changes.Changes,
            });
        }

        return result;
    }

    private static X509Certificate2? Certificate(CmsCertificate? certificate)
    {
        try
        {
            return certificate?.Certificate;
        }
        catch (CryptographicException)
        {
            return null;                                                       // the base library does not read it (the check reads it here)
        }
    }

    private static string? Text(PdfFile file, PdfValue value) => file.Resolve(value).AsString is { } text ? PdfText.Decode(text.Bytes) : null;

    // The form a signature takes: by its /SubFilter, and for PAdES by what it holds (a timestamp), what the document adds
    // for it (validation data) and after it (a document timestamp).
    private static DocumentSignatureKind Kind(PdfSignatureEntry entry, List<PdfSignatureEntry> all, PdfSignatureContext context)
    {
        switch (entry.SubFilter)
        {
            case "adbe.pkcs7.detached":
                return DocumentSignatureKind.Cms;
            case "adbe.pkcs7.sha1":
                return DocumentSignatureKind.LegacyPkcs7Sha1;
            case "adbe.x509.rsa_sha1":
                return DocumentSignatureKind.LegacyPkcs1;
            case "ETSI.CAdES.detached":
                if (entry.Signer?.Unsigned(CmsOids.SignatureTimeStampToken) is null)
                {
                    return DocumentSignatureKind.BaselineB;
                }

                var data = context.Dss;
                bool validated = data.Present && (data.Vri.Contains(Convert.ToHexString(SHA1.HashData(Trimmed(entry.Contents))))
                                                  || (data.Vri.Count == 0 && (data.Crls.Count > 0 || data.Ocsps.Count > 0)));
                if (!validated)
                {
                    return DocumentSignatureKind.BaselineT;
                }

                return all.Exists(o => o.IsDocumentTimestamp && o.SignedEnd > entry.SignedEnd) ? DocumentSignatureKind.BaselineLta : DocumentSignatureKind.BaselineLt;
            default:
                return DocumentSignatureKind.Other;
        }
    }

    /// <summary>A signature's bytes without the reserved zeros after its DER (what a VRI key is the SHA-1 of).</summary>
    public static ReadOnlySpan<byte> Trimmed(byte[] contents)
    {
        try
        {
            var reader = CmsAsn.Reader(contents);
            int length = reader.ReadEncodedValue().Length;
            return contents.AsSpan(0, length);
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return contents;
        }
    }

    private static PdfSignatureEntry Entry(PdfSignatureContext context, string name, PdfDictionary field, PdfDictionary signature, PdfValue value, bool certifies)
    {
        var file = context.File;
        string subFilter = file.Resolve(signature["SubFilter"u8]).AsName?.ToString() ?? "";
        byte[] contents = file.Resolve(signature["Contents"u8]).AsString?.Bytes ?? [];
        long[] range = file.Resolve(signature["ByteRange"u8]).AsArray?.Items.Select(v => file.Resolve(v).TryInteger(out long n) ? n : -1).ToArray() ?? [];
        string? rangeProblem = CheckRange(file, value, range, contents, out long signedEnd, out bool whole);

        CmsSignedData? cms = null;
        CmsSignerInfo? signer = null;
        CmsCertificate? certificate = null;
        var certEntry = new List<CmsCertificate>();
        string? contentsProblem = null;
        CmsTimestamp? timestamp = null;
        string? timestampProblem = null;
        try
        {
            if (contents.Length == 0)
            {
                contentsProblem = "the signature dictionary has no /Contents";
            }
            else if (subFilter == "adbe.x509.rsa_sha1")
            {
                // /Contents: a DER OCTET STRING holding the PKCS #1 signature; /Cert: the signer's certificate (first), then others.
                var cert = file.Resolve(signature["Cert"u8]);
                PdfValue[] items = cert.AsArray?.Items ?? (cert.AsString is not null ? new[] { cert } : []);
                foreach (var item in items)
                {
                    if (file.Resolve(item).AsString is { } der && CmsCertificate.TryParse(der.Bytes) is { } read)
                    {
                        certEntry.Add(read);
                    }
                }

                certificate = certEntry.Count > 0 ? certEntry[0] : null;
            }
            else if (subFilter == "ETSI.RFC3161")
            {
                timestamp = CmsTimestamp.Parse(contents);
                cms = timestamp.Token;
                signer = cms.Signers.Count == 1 ? cms.Signers[0] : null;
                certificate = timestamp.Signer;
            }
            else
            {
                cms = CmsSignedData.Parse(contents);
                signer = cms.Signers.Count == 1 ? cms.Signers[0] : null;
                if (signer is null)
                {
                    contentsProblem = $"the CMS has {cms.Signers.Count} signers (one is expected)";
                }
                else
                {
                    certificate = signer.FindCertificate(cms.Certificates);
                    if (signer.Unsigned(CmsOids.SignatureTimeStampToken) is { Values.Length: >= 1 } token)
                    {
                        try
                        {
                            timestamp = CmsTimestamp.Parse(token.Values[0]);
                        }
                        catch (InvalidDataException e)
                        {
                            timestampProblem = e.Message;
                        }
                    }
                }
            }
        }
        catch (InvalidDataException e)
        {
            contentsProblem = e.Message;
        }

        // The permission: the signature's own DocMDP (a certification), else its field's /Lock; and the fields it locks.
        int? permission = null;
        var locked = new List<string>();
        var excluded = new List<string>();
        bool all = false;
        foreach (var reference in file.Resolve(signature["Reference"u8]).AsArray?.Items ?? [])
        {
            if (file.Dictionary(reference) is not { } sigRef)
            {
                continue;
            }

            var parameters = file.Dictionary(sigRef["TransformParams"u8]);
            var method = file.Resolve(sigRef["TransformMethod"u8]);
            if (method.IsName("DocMDP"u8))
            {
                permission = parameters is not null && file.Resolve(parameters["P"u8]).TryInteger(out long p) && p is >= 1 and <= 3 ? (int)p : 2;
            }
            else if (method.IsName("FieldMDP"u8) && parameters is not null)
            {
                Lock(file, parameters, locked, excluded, ref all);
            }
        }

        if (file.Dictionary(field["Lock"u8]) is { } fieldLock)
        {
            if (permission is null && file.Resolve(fieldLock["P"u8]).TryInteger(out long p) && p is >= 1 and <= 3)
            {
                permission = (int)p;
            }

            Lock(file, fieldLock, locked, excluded, ref all);
        }

        return new PdfSignatureEntry
        {
            Context = context, Field = name, Dictionary = signature, SubFilter = subFilter, Contents = contents, ByteRange = range, RangeProblem = rangeProblem,
            SignedEnd = signedEnd, Whole = whole, Cms = cms, Signer = signer, SignerCertificate = certificate, CertEntry = certEntry, ContentsProblem = contentsProblem,
            Timestamp = timestamp, TimestampProblem = timestampProblem, Permission = permission, Certification = certifies, Locked = locked, LocksAll = all,
            LockExcluded = excluded,
        };
    }

    // A FieldMDP lock: /Action All, Include (/Fields) or Exclude (/Fields).
    private static void Lock(PdfFile file, PdfDictionary parameters, List<string> locked, List<string> excluded, ref bool all)
    {
        var action = file.Resolve(parameters["Action"u8]);
        var fields = (file.Resolve(parameters["Fields"u8]).AsArray?.Items ?? []).Select(f => Text(file, f)).OfType<string>().ToList();
        if (action.IsName("All"u8))
        {
            all = true;
        }
        else if (action.IsName("Include"u8))
        {
            locked.AddRange(fields);
        }
        else if (action.IsName("Exclude"u8))
        {
            all = true;
            excluded.AddRange(fields);
        }
    }

    // The byte range against the file: two pairs, from the file's start, whose hole is exactly this signature's /Contents hex
    // string inside its own dictionary, ending at the file's end or at a revision's end (%%EOF); null when it is so.
    private static string? CheckRange(PdfFile file, PdfValue value, long[] range, byte[] contents, out long signedEnd, out bool whole)
    {
        long length = file.Source.Length;
        signedEnd = 0;
        whole = false;
        if (range.Length != 4 || Array.Exists(range, v => v < 0))
        {
            return $"a byte range of {range.Length} numbers ({string.Join(" ", range)}): a signature's is two pairs, offset and length";
        }

        long a = range[0], b = range[1], c = range[2], d = range[3];
        signedEnd = c + d;
        whole = signedEnd == length;
        if (a != 0)
        {
            return $"a byte range that starts at byte {a}, not at the file's start";
        }

        if (b <= 0 || c < b + 2 || signedEnd > length)
        {
            return $"a byte range [{a} {b} {c} {d}] that does not fit a file of {length:N0} bytes";
        }

        long hole = c - b;
        if (hole - 2 != 2L * contents.Length)
        {
            return $"a byte range whose hole ({hole:N0} bytes) is not the signature's /Contents ({contents.Length:N0} bytes, {2L * contents.Length + 2:N0} as hex)";
        }

        using (var bracket = PdfWindow.Read(file.Source, b, 1))
        using (var close = PdfWindow.Read(file.Source, c - 1, 1))
        {
            if (bracket.Span is not [(byte)'<'] || close.Span is not [(byte)'>'])
            {
                return "a byte range whose hole is not a hex string";
            }
        }

        // The hole's digits are the /Contents read: the hole is this dictionary's own string.
        using (var digits = PdfWindow.Read(file.Source, b + 1, hole - 2))
        {
            var span = digits.Span;
            for (int i = 0; i < contents.Length; i++)
            {
                if (span.Length < 2 * i + 2 || Hex(span[2 * i]) * 16 + Hex(span[2 * i + 1]) != contents[i])
                {
                    return "a byte range whose hole holds other bytes than the signature's /Contents";
                }
            }
        }

        if (value.IsReference && file.Xref[value.AsReference.Number] is { Kind: PdfEntryKind.InUse } entry)
        {
            long start = entry.Offset + file.Xref.Shift;
            if (start >= b || b - start > 1 << 20)
            {
                return "a byte range whose hole is not inside the signature's dictionary";
            }

            using var head = PdfWindow.Read(file.Source, start, b - start);
            if (head.Span.IndexOf("endobj"u8) >= 0)
            {
                return "a byte range whose hole is not inside the signature's dictionary";
            }
        }
        else if (value.IsReference)
        {
            return "a signature dictionary inside an object stream, whose /Contents a byte range cannot leave out";
        }

        if (!whole)
        {
            using var end = PdfWindow.Read(file.Source, Math.Max(0, signedEnd - 32), Math.Min(32, signedEnd));
            if (!end.Span.TrimEnd(" \r\n\t\f\0"u8).EndsWith("%%EOF"u8))
            {
                return $"a byte range that ends at byte {signedEnd:N0}, inside a revision (not after its %%EOF): what follows cannot be told apart from what was signed";
            }
        }

        return null;
    }

    private static int Hex(byte b) => b switch
    {
        >= (byte)'0' and <= (byte)'9' => b - '0',
        >= (byte)'a' and <= (byte)'f' => b - 'a' + 10,
        >= (byte)'A' and <= (byte)'F' => b - 'A' + 10,
        _ => -1000,
    };
}

/// <summary>The AcroForm's fields walked (ISO 32000-2, 12.7.4): each terminal field with its full name and its type (inherited).</summary>
internal static class PdfFormFields
{
    /// <summary>
    /// Calls <paramref name="visit"/> for every field of <paramref name="form"/> that has no kids that are fields (a field
    /// whose kids are widgets alone is terminal): its full name (the partial names joined by dots), its node (a reference,
    /// or the dictionary itself), its dictionary and its /FT (inherited; null for none). A node met again is passed over;
    /// the depth is the file's limit.
    /// </summary>
    public static void Walk(PdfFile file, PdfDictionary form, Action<string, PdfValue, PdfDictionary, string?> visit)
    {
        var seen = new HashSet<int>();
        var stack = new Stack<(PdfValue Node, string Prefix, string? Type, int Depth)>();
        var fields = file.Resolve(form["Fields"u8]).AsArray?.Items ?? [];
        for (int i = fields.Length - 1; i >= 0; i--)
        {
            stack.Push((fields[i], "", null, 1));
        }

        while (stack.Count > 0)
        {
            var (node, prefix, inherited, depth) = stack.Pop();
            if ((node.IsReference && !seen.Add(node.AsReference.Number)) || depth > file.MaxDepth)
            {
                continue;
            }

            PdfDictionary? field;
            try
            {
                field = file.Dictionary(node);
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                continue;
            }

            if (field is null)
            {
                continue;
            }

            string? own = file.Resolve(field["T"u8]).AsString is { } t ? PdfText.Decode(t.Bytes) : null;
            string name = own is null ? prefix : prefix.Length == 0 ? own : prefix + "." + own;
            string? type = file.Resolve(field["FT"u8]).AsName?.ToString() ?? inherited;
            var kids = file.Resolve(field["Kids"u8]).AsArray?.Items ?? [];
            bool fieldKids = false;
            foreach (var kid in kids)
            {
                if (file.Dictionary(kid) is { } k && k.Contains("T"u8))
                {
                    fieldKids = true;
                    break;
                }
            }

            if (!fieldKids)
            {
                visit(name, node, field, type);
                continue;
            }

            for (int i = kids.Length - 1; i >= 0; i--)
            {
                stack.Push((kids[i], name, type, depth + 1));
            }
        }
    }
}
