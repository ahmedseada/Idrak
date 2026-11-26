// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>A signature an editor or a builder was asked for: the field and how to sign.</summary>
internal readonly record struct PdfSignRequest(string Field, SigningOptions Options);

/// <summary>
/// PDF signing (contract C25, PAdES baseline, ETSI EN 319 142-1): each signature is an incremental update of its own on
/// what came before (so an earlier signature's bytes stay as they were): its field (an empty signature field of the form,
/// or a new one with its widget on the page asked, invisible without a box), an appearance (text in the standard
/// Helvetica, an image, or none), the signature dictionary (<c>ETSI.CAdES.detached</c>) with its byte range and its
/// <c>/Contents</c> reserved, then filled: the bytes hashed around the hole, a CMS SignedData written on the DER (content
/// type, message digest and ESS signing-certificate-v2 signed; no signing time, the claimed time being <c>/M</c>), signed
/// with the app's key; B-T adds the timestamp of the signature's value from the authority the app names (through
/// <c>SignatureServices</c>; nothing else is asked), and B-LT then one more update adding the validation data to the
/// DSS (the chain's and the authority's certificates, the revocation data the app gives or, when it turns it on, fetched),
/// with its VRI entry. A file's encryption is kept (strings and streams encrypted by its own key; <c>/Contents</c> never).
/// </summary>
internal static class PdfSigning
{
    private const int TimestampRoom = 16 * 1024, Window = 1 << 20;

    // The standard Helvetica's index among the standard 14 fonts, and the characters WinAnsiEncoding codes (what an
    // appearance's text is written in).
    private static readonly int Helvetica = PdfFontTables.StandardFontIndex("Helvetica");
    private static readonly Lazy<(Dictionary<char, byte> Codes, char[] Characters)> WinAnsi = new(WinAnsiCodes);

    /// <summary>Checks a <c>Sign</c> call's arguments, so a mistake is refused where it is made, not at the save.</summary>
    public static void Validate(string field, SigningOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Certificate is null || !options.Certificate.HasPrivateKey)
        {
            throw new ArgumentException($"The signing certificate '{options.Certificate?.Subject}' has no private key (SigningOptions.Certificate: a certificate with its key).", nameof(options));
        }

        if (options.Digest == HashAlgorithmName.SHA1 || !(options.Digest == HashAlgorithmName.SHA256 || options.Digest == HashAlgorithmName.SHA384 || options.Digest == HashAlgorithmName.SHA512))
        {
            throw new ArgumentException($"A signature is made with SHA-256, SHA-384 or SHA-512, not {options.Digest.Name} (SigningOptions.Digest; SHA-1 is read only).", nameof(options));
        }

        if (options.Level is not (DocumentSignatureLevel.BaselineB or DocumentSignatureLevel.BaselineT or DocumentSignatureLevel.BaselineLt))
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Level, "Not a signature level (SigningOptions.Level).");
        }

        if (options.Level != DocumentSignatureLevel.BaselineB && options.TimestampAuthority is null)
        {
            throw new ArgumentException($"PAdES {options.Level} needs a timestamp from an authority the app names (SigningOptions.TimestampAuthority); nothing is asked of one it does not.", nameof(options));
        }

        if (options.Level == DocumentSignatureLevel.BaselineLt && options.Revocation.Count == 0 && !options.Online)
        {
            throw new ArgumentException(
                "PAdES B-LT adds the chain's revocation data to the document: give it (SigningOptions.Revocation: revocation lists and OCSP responses) or let it be fetched (SigningOptions.Online, or IDRAK_DOCUMENT_SIGNATURE_ONLINE).",
                nameof(options));
        }

        if (options.Box is { } box && !(float.IsFinite(box.X) && float.IsFinite(box.Y) && box.Width >= 0 && box.Height >= 0 && float.IsFinite(box.Width) && float.IsFinite(box.Height)))
        {
            throw new ArgumentOutOfRangeException(nameof(options), box, "A signature field's box is finite, with a width and a height of 0 or more (SigningOptions.Box).");
        }

        if (options.Timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Timeout, "A service's timeout is more than zero (SigningOptions.Timeout).");
        }
    }

    /// <summary>
    /// Writes <paramref name="source"/> to <paramref name="destination"/> signed: its bytes as they are, then each signature
    /// of <paramref name="requests"/> as an update of its own on what came before (B-LT's validation data in one more).
    /// The updates are kept in memory (each a few KiB with its reserved signature); the source is read where it is.
    /// </summary>
    /// <exception cref="ArgumentException">A field that is not a signature field, a page out of range, a revocation item that is neither a list nor a response.</exception>
    /// <exception cref="InvalidOperationException">The field is signed already, or the signature does not fit the room reserved for it twice.</exception>
    /// <exception cref="IOException">The timestamp authority (or a revocation address) could not be reached or answered wrongly.</exception>
    public static void WriteSigned(PdfDocument source, IReadOnlyList<PdfSignRequest> requests, Stream destination, DocumentWriteOptions writeOptions)
    {
        var updates = new List<byte[]>();
        bool newline = NeedsNewline(source.File.Source);
        var opened = new List<IDisposable>();
        try
        {
            var current = source;
            foreach (var (field, options) in requests)
            {
                var signed = Sign(current, field, options, writeOptions);
                updates.Add(signed.Update);
                current = Reopen(source, newline, updates, opened);
                if (options.Level == DocumentSignatureLevel.BaselineLt)
                {
                    updates.Add(ValidationData(current, signed, options, writeOptions));
                    current = Reopen(source, newline, updates, opened);
                }
            }

            Copy(source.File, destination);
            if (newline)
            {
                destination.WriteByte((byte)'\n');
            }

            foreach (byte[] update in updates)
            {
                destination.Write(update);
            }

            destination.Flush();
        }
        finally
        {
            for (int i = opened.Count - 1; i >= 0; i--)
            {
                opened[i].Dispose();
            }
        }
    }

    // The document as it is after the updates so far: the source's bytes (and the line end it lacked) and the updates.
    private static PdfDocument Reopen(PdfDocument source, bool newline, List<byte[]> updates, List<IDisposable> opened)
    {
        var stream = new PdfJoinedStream(source.File.Source, newline, [.. updates]);
        opened.Add(stream);
        var document = PdfDocument.Open(stream, source.File.Name, source.Options);
        opened.Add(document);
        return document;
    }

    /// <summary>Whether a file's bytes end without a line end (an update then starts on a line of its own).</summary>
    public static bool NeedsNewline(PdfSource source)
    {
        Span<byte> last = stackalloc byte[1];
        return source.Length > 0 && source.Read(source.Length - 1, last) == 1 && last[0] is not ((byte)'\n' or (byte)'\r');
    }

    /// <summary>A file's bytes copied to <paramref name="destination"/>, in windows.</summary>
    public static void Copy(PdfFile file, Stream destination)
    {
        var source = file.Source;
        if (source.InMemory)
        {
            destination.Write(source.Memory.Span);
            return;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(Window);
        try
        {
            for (long at = 0; at < source.Length;)
            {
                int read = source.Read(at, buffer.AsSpan(0, (int)Math.Min(buffer.Length, source.Length - at)));
                if (read <= 0)
                {
                    throw new IOException($"{file.Name}: the file ended at byte {at:N0} of {source.Length:N0} while it was copied.");
                }

                destination.Write(buffer, 0, read);
                at += read;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // What a signature update gives: its bytes, the CMS it holds, and the certificates its validation data covers.
    private sealed record Signed(byte[] Update, byte[] Cms, IReadOnlyList<CmsCertificate> Certificates);

    // Where the signature goes: an empty signature field of the form (its node, and its widget's when separate), or a new
    // field on a page; and where its appearance is drawn.
    private sealed class Target
    {
        public required string Name { get; init; }

        public PdfValue Field { get; init; }

        public PdfDictionary? FieldDictionary { get; init; }

        public PdfValue Widget { get; init; }

        public PdfDictionary? WidgetDictionary { get; init; }

        public PdfValue Form { get; init; }

        public PdfDictionary? FormDictionary { get; init; }

        public int Page { get; init; } = -1;

        public PdfValue PageNode { get; init; }

        public double[] Rect { get; init; } = [0, 0, 0, 0];

        /// <summary>The appearance's size and its matrix (the form's space onto the page's user space), when one is drawn.</summary>
        public (double Width, double Height, PdfMatrix Matrix)? Appearance { get; init; }

        public bool New => Field.IsNull;
    }

    private static Target Find(PdfDocument document, string name, SigningOptions options)
    {
        var file = document.File;
        var catalog = file.Dictionary(file.Trailer["Root"u8]) ?? throw new InvalidDataException($"{file.Name}: no document catalog to sign.");
        var formValue = catalog["AcroForm"u8];
        var form = file.Dictionary(formValue);
        (PdfValue Node, PdfDictionary Field, string? Type)? found = null;
        if (form is not null)
        {
            PdfFormFields.Walk(file, form, (fullName, node, field, type) =>
            {
                if (found is null && fullName == name)
                {
                    found = (node, field, type);
                }
            });
        }

        bool drawn = options.AppearanceText is not null || options.AppearanceImage is not null;
        if (found is { } existing)
        {
            if (existing.Type != "Sig")
            {
                throw new ArgumentException($"The field '{name}' is a {existing.Type ?? "typeless"} field, not a signature field: a signature goes into a signature field (or a new one under another name).", "field");
            }

            if (!existing.Field["V"u8].IsNull)
            {
                throw new InvalidOperationException($"The signature field '{name}' is signed already: a new signature goes into an empty or a new field.");
            }

            if (!existing.Node.IsReference)
            {
                throw new InvalidDataException($"{file.Name}: the signature field '{name}' is not an object of its own, so it cannot be updated: save the document as a rewrite first.");
            }

            PdfValue widget = PdfValue.Null;
            PdfDictionary? widgetDictionary = null;
            if (!existing.Field["Subtype"u8].IsName("Widget"u8))
            {
                foreach (var kid in file.Resolve(existing.Field["Kids"u8]).AsArray?.Items ?? [])
                {
                    if (file.Dictionary(kid) is { } k && k["Subtype"u8].IsName("Widget"u8) && kid.IsReference)
                    {
                        (widget, widgetDictionary) = (kid, k);
                        break;
                    }
                }
            }

            var annotation = widgetDictionary ?? existing.Field;
            double[] rect = Rect(file, annotation["Rect"u8]);
            (double, double, PdfMatrix)? appearance = null;
            if (drawn && rect[2] - rect[0] > 0 && rect[3] - rect[1] > 0)
            {
                int page = PageOf(document, annotation["P"u8]);
                appearance = page >= 0 ? Place(document, page, Viewer(document, page, rect)).Appearance
                    : (rect[2] - rect[0], rect[3] - rect[1], PdfMatrix.Identity);
            }

            return new Target
            {
                Name = name, Field = existing.Node, FieldDictionary = existing.Field, Widget = widget, WidgetDictionary = widgetDictionary, Form = formValue,
                FormDictionary = form, Rect = rect, Appearance = appearance,
            };
        }

        if (name.Contains('.', StringComparison.Ordinal))
        {
            throw new ArgumentException($"A new signature field is named without periods ('{name}'): a period separates the names of a field's parents, and no field '{name}' exists.", "field");
        }

        if ((uint)options.Page >= (uint)document.PageCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Page, $"The document has {document.PageCount} pages (SigningOptions.Page, from 0).");
        }

        var node = document.PageObject(options.Page).Node;
        if (!node.IsReference)
        {
            throw new InvalidDataException($"{file.Name}: page {options.Page + 1} is not an object of its own, so a signature field cannot be added to it: save the document as a rewrite first.");
        }

        if (options.Box is { IsEmpty: false } box && box.Width > 0 && box.Height > 0)
        {
            var (rect, appearance) = Place(document, options.Page, box);
            return new Target
            {
                Name = name, Form = formValue, FormDictionary = form, Page = options.Page, PageNode = node, Rect = rect, Appearance = drawn ? appearance : null,
            };
        }

        return new Target { Name = name, Form = formValue, FormDictionary = form, Page = options.Page, PageNode = node };
    }

    // A rectangle [llx lly urx ury] with its corners in order, or zeros.
    private static double[] Rect(PdfFile file, PdfValue value)
    {
        double[] rect = [0, 0, 0, 0];
        if (file.Resolve(value).AsArray is { Count: >= 4 } array)
        {
            for (int i = 0; i < 4; i++)
            {
                rect[i] = file.Resolve(array[i]).TryNumber(out double n) && double.IsFinite(n) ? n : 0;
            }
        }

        return [Math.Min(rect[0], rect[2]), Math.Min(rect[1], rect[3]), Math.Max(rect[0], rect[2]), Math.Max(rect[1], rect[3])];
    }

    // The page a widget's /P names (its index), or -1.
    private static int PageOf(PdfDocument document, PdfValue page)
    {
        if (!page.IsReference)
        {
            return -1;
        }

        for (int i = 0; i < document.PageCount; i++)
        {
            var node = document.PageObject(i).Node;
            if (node.IsReference && node.AsReference.Number == page.AsReference.Number)
            {
                return i;
            }
        }

        return -1;
    }

    // A rectangle of the page's user space as shown (points from the top-left corner).
    private static DocumentBox Viewer(PdfDocument document, int page, double[] rect)
    {
        var display = document.Frame(page).Display;
        var (x0, y0, x1, y1) = Bounds(display, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
        return new DocumentBox((float)x0, (float)y0, (float)(x1 - x0), (float)(y1 - y0));
    }

    // A box on the page as shown, placed: its rectangle in the page's user space, and the appearance's size and matrix (its
    // form's space, origin at the box's lower left as shown, onto the user space: upright on a turned page).
    private static (double[] Rect, (double Width, double Height, PdfMatrix Matrix) Appearance) Place(PdfDocument document, int page, DocumentBox box)
    {
        var display = document.Frame(page).Display;
        var toPage = new PdfMatrix(1, 0, 0, -1, box.X, box.Y + box.Height).Then(PdfEditPages.Inverse(display));
        var (x0, y0, x1, y1) = Bounds(toPage, 0, 0, box.Width, box.Height);
        return ([x0, y0, x1, y1], (box.Width, box.Height, toPage with { E = 0, F = 0 }));
    }

    private static (double, double, double, double) Bounds(PdfMatrix m, double x, double y, double width, double height)
    {
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var (u, v) in (ReadOnlySpan<(double, double)>)[(x, y), (x + width, y), (x, y + height), (x + width, y + height)])
        {
            double px = m.A * u + m.C * v + m.E, py = m.B * u + m.D * v + m.F;
            (minX, minY, maxX, maxY) = (Math.Min(minX, px), Math.Min(minY, py), Math.Max(maxX, px), Math.Max(maxY, py));
        }

        return (minX, minY, maxX, maxY);
    }

    // One signature's update, signed: written with /Contents reserved, the byte range filled, the bytes around the hole
    // hashed, the CMS made (and timestamped) and written into the hole; once more with the room it needed when it did not fit.
    private static Signed Sign(PdfDocument document, string field, SigningOptions options, DocumentWriteOptions writeOptions)
    {
        var file = document.File;
        var signer = CmsCertificate.From(options.Certificate);
        var chain = options.Chain.Select(CmsCertificate.From).ToList();
        var target = Find(document, field, options);
        var time = options.SigningTime ?? DateTimeOffset.UtcNow;
        int reserve = signer.Raw.Length + chain.Sum(c => c.Raw.Length) + CmsKeys.MostSignatureBytes(options.Certificate) + 2 * signer.Issuer.Length + 2048
                      + (options.Level == DocumentSignatureLevel.BaselineB ? 0 : TimestampRoom);

        // The source's bytes hashed once; each attempt hashes its update from a copy of that state.
        using var head = IncrementalHash.CreateHash(options.Digest);
        HashSource(head, file);
        bool newline = NeedsNewline(file.Source);
        if (newline)
        {
            head.AppendData("\n"u8);
        }

        long prefix = file.Source.Length + (newline ? 1 : 0);
        for (int attempt = 0; ; attempt++)
        {
            var (update, rangeAt, rangeWidth, holeStart, holeEnd) = Write(document, target, options, writeOptions, reserve, prefix, time);
            long total = prefix + update.Length;
            long b = prefix + holeStart, c = prefix + holeEnd;
            byte[] range = Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"[0 {b} {c} {total - c}]"));
            if (range.Length > rangeWidth)
            {
                throw new InvalidOperationException($"The byte range {Encoding.ASCII.GetString(range)} does not fit the {rangeWidth} bytes reserved for it (a fault of the writer).");
            }

            range.CopyTo(update.AsSpan(rangeAt));
            update.AsSpan(rangeAt + range.Length, rangeWidth - range.Length).Fill((byte)' ');
            using var hash = head.Clone();
            hash.AppendData(update, 0, holeStart);
            hash.AppendData(update, holeEnd, update.Length - holeEnd);
            byte[] digest = hash.GetHashAndReset();

            byte[] attributes = CmsWriter.SignedAttributes(CmsOids.Data, digest, signer, options.Digest);
            var (value, scheme) = CmsKeys.Sign(options.Certificate, options.Digest, options.RsaPadding, CmsDigest.Of(options.Digest, attributes));
            List<(string, byte[])>? unsigned = null;
            IReadOnlyList<CmsCertificate> stamped = [];
            if (options.Level != DocumentSignatureLevel.BaselineB)
            {
                var (token, read) = Timestamp(value, options, file.MaxStreamBytes);
                unsigned = [(CmsOids.SignatureTimeStampToken, token)];
                stamped = read.Token.Certificates;
            }

            byte[] cms = CmsWriter.SignedData(CmsOids.Data, null, signer, options.Digest, attributes, scheme, value, [signer, .. chain], unsigned);
            int room = (holeEnd - holeStart - 2) / 2;
            if (cms.Length > room)
            {
                if (attempt > 0)
                {
                    throw new InvalidOperationException($"The signature ({cms.Length:N0} bytes) does not fit the {room:N0} bytes reserved for it the second time either.");
                }

                reserve = cms.Length + 4096;                                   // the timestamp was larger than its room: once more
                continue;
            }

            var hex = update.AsSpan(holeStart + 1, 2 * cms.Length);
            for (int i = 0; i < cms.Length; i++)
            {
                hex[2 * i] = Digit(cms[i] >> 4);
                hex[2 * i + 1] = Digit(cms[i] & 0xF);
            }

            return new Signed(update, cms, [signer, .. chain, .. stamped]);
        }
    }

    private static byte Digit(int value) => (byte)(value < 10 ? '0' + value : 'A' + value - 10);

    private static void HashSource(IncrementalHash hash, PdfFile file)
    {
        var source = file.Source;
        if (source.InMemory)
        {
            hash.AppendData(source.Memory.Span);
            return;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(Window);
        try
        {
            for (long at = 0; at < source.Length;)
            {
                int read = source.Read(at, buffer.AsSpan(0, (int)Math.Min(buffer.Length, source.Length - at)));
                if (read <= 0)
                {
                    throw new IOException($"{file.Name}: the file ended at byte {at:N0} of {source.Length:N0} while it was hashed.");
                }

                hash.AppendData(buffer, 0, read);
                at += read;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The signature services were registered by their scheme (SignatureServices) and asked through it: not on the new design yet.
    private static byte[] Send(SignatureServiceRequest request) =>
        throw new NotSupportedException("A timestamp or revocation request through the signature services by scheme is not on the new design yet.");

    // The signature's value timestamped by the authority the app named.
    private static (byte[] Token, CmsTimestamp Read) Timestamp(byte[] value, SigningOptions options, long maxBytes)
    {
        var authority = options.TimestampAuthority!;
        byte[] imprint = CmsDigest.Of(options.Digest, value);
        var nonce = new BigInteger(RandomNumberGenerator.GetBytes(8), isUnsigned: true);
        byte[] response = Send(new SignatureServiceRequest(authority, SignatureServiceKind.Timestamp)
        {
            Body = CmsTimestamp.Request(options.Digest, imprint, nonce), Timeout = options.Timeout, MaxResponseBytes = maxBytes,
        });
        var (token, read) = CmsTimestamp.FromResponse(response, imprint, nonce, authority.ToString());
        if (read.CheckToken() is { } broken)
        {
            throw new IOException($"The timestamp authority {authority} answered with a token that does not verify: {broken}.");
        }

        return (token, read);
    }

    // The update's bytes with the signature's room reserved: where the byte range's placeholder is and how wide, where the
    // hole is (its '<' and past its '>').
    private static (byte[] Update, int RangeAt, int RangeWidth, int HoleStart, int HoleEnd) Write(PdfDocument document, Target target, SigningOptions options,
        DocumentWriteOptions writeOptions, int reserve, long prefix, DateTimeOffset time)
    {
        var file = document.File;
        var xref = file.Xref;
        var (standsAlone, carried, classic, size) = Layout(xref, file);
        using var output = new MemoryStream(4096 + 2 * reserve);
        int rangeAt, rangeWidth, holeStart, holeEnd;
        using (var writer = PdfFileWriter.Append(output, prefix, xref.Shift, size, writeOptions.Compression, packing: !classic && writeOptions.ObjectStreams))
        {
            writer.Security = file.Security;
            using var copier = new PdfEditCopier(writer, writeOptions.Compression, recompress: false);
            var own = copier.For(document, identity: true);
            int signature = writer.Allocate();
            int appearance = 0;
            if (target.Appearance is { } placed)
            {
                int image = 0;
                if (options.AppearanceImage is { } picture)
                {
                    image = writer.Allocate();
                    PdfWriterImages.Write(writer, image, picture, DocumentImageEncoding.Auto, writeOptions);
                }

                appearance = writer.Allocate();
                var s = writer.BeginStream(appearance).Raw("/Type/XObject/Subtype/Form/BBox[0 0 "u8).Number(placed.Width, 4).Space().Number(placed.Height, 4)
                    .Raw("]/Matrix["u8).Number(placed.Matrix.A, 6).Space().Number(placed.Matrix.B, 6).Space().Number(placed.Matrix.C, 6).Space()
                    .Number(placed.Matrix.D, 6).Raw(" 0 0]/Resources<<"u8);
                if (options.AppearanceText is not null)
                {
                    s.Raw("/Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>>>"u8);
                }

                if (image > 0)
                {
                    s.Raw("/XObject<</Im1 "u8).Reference(image).Raw(">>"u8);
                }

                s.Raw(">>"u8);
                writer.EndStream(Content(options, placed.Width, placed.Height, image > 0), deflate: true);
            }

            if (target.New)
            {
                int field = writer.Allocate();
                var (page, pageGeneration) = (target.PageNode.AsReference.Number, target.PageNode.AsReference.Generation);
                copier.WriteObject(field, s =>
                {
                    s.Raw("<</FT/Sig/T"u8).Text(target.Name).Raw("/V "u8).Reference(signature).Raw("/Type/Annot/Subtype/Widget/F 132/Rect["u8);
                    for (int i = 0; i < 4; i++)
                    {
                        if (i > 0)
                        {
                            s.Space();
                        }

                        s.Number(target.Rect[i], 4);
                    }

                    s.Raw("]/P "u8).Integer(page).Space().Integer(pageGeneration).Raw(" R"u8);
                    if (appearance > 0)
                    {
                        s.Raw("/AP<</N "u8).Reference(appearance).Raw(">>"u8);
                    }

                    s.Raw(">>"u8);
                });

                AddToPage(file, writer, copier, own, target.PageNode, field);
                AddToForm(file, writer, copier, own, target, field);
            }
            else
            {
                bool merged = target.Widget.IsNull;
                Rewrite(writer, copier, own, target.Field, target.FieldDictionary!, key => key.Is("V"u8) || (merged && appearance > 0 && key.Is("AP"u8)), s =>
                {
                    s.Raw("/V "u8).Reference(signature);
                    if (merged && appearance > 0)
                    {
                        s.Raw("/AP<</N "u8).Reference(appearance).Raw(">>"u8);
                    }
                });
                if (!merged && appearance > 0)
                {
                    Rewrite(writer, copier, own, target.Widget, target.WidgetDictionary!, key => key.Is("AP"u8), s => s.Raw("/AP<</N "u8).Reference(appearance).Raw(">>"u8));
                }

                if (target.FormDictionary is { } form && !(file.Resolve(form["SigFlags"u8]).TryInteger(out long flags) && (flags & 3) == 3))
                {
                    AddToForm(file, writer, copier, own, target, 0);
                }
            }

            // The signature dictionary, on its own (never packed): the byte range's placeholder and /Contents reserved.
            int width = prefix + (1L << 24) < 10_000_000_000L ? 10 : 19;
            string zeros = new('0', width);
            byte[] placeholder = Encoding.ASCII.GetBytes($"[0 {zeros} {zeros} {zeros}]");
            var sig = writer.BeginObject(signature);
            sig.Raw("<</Type/Sig/Filter/Adobe.PPKLite/SubFilter/ETSI.CAdES.detached/ByteRange"u8);
            rangeAt = (int)(writer.Position - prefix);
            rangeWidth = placeholder.Length;
            sig.Raw(placeholder).Raw("/Contents"u8);
            holeStart = (int)(writer.Position - prefix);
            var room = sig.Buffer.Reserve(2 * reserve + 2);
            room.Fill((byte)'0');
            room[0] = (byte)'<';
            room[^1] = (byte)'>';
            sig.Buffer.Advance(2 * reserve + 2);
            holeEnd = (int)(writer.Position - prefix);
            sig.Raw("/M"u8).Text(PdfDocumentBuilder.Date(time.ToUniversalTime()));
            if (options.Reason is { } reason)
            {
                sig.Raw("/Reason"u8).Text(reason);
            }

            if (options.Location is { } location)
            {
                sig.Raw("/Location"u8).Text(location);
            }

            sig.Raw(">>"u8);
            writer.EndObject();
            copier.Drain();
            writer.FinishUpdate(classic, standsAlone ? -1 : xref.Sections[0].Offset, size, carried, s => Trailer(s, copier, own, file));
        }

        return (output.ToArray(), rangeAt, rangeWidth, holeStart, holeEnd);
    }

    // How an update of `file` is written: standing alone when its cross-reference was rebuilt (every object listed), a
    // table after a table, a stream after a stream; objects numbered from its /Size.
    private static (bool StandsAlone, List<(int Number, PdfEntry Entry)> Carried, bool Classic, int Size) Layout(PdfXref xref, PdfFile file)
    {
        bool standsAlone = xref.Repaired || xref.Sections.Count == 0;
        var carried = new List<(int Number, PdfEntry Entry)>();
        if (standsAlone)
        {
            for (int n = 1; n < xref.Count; n++)
            {
                if (xref.Entries[n].Kind is PdfEntryKind.InUse or PdfEntryKind.Compressed)
                {
                    carried.Add((n, xref.Entries[n]));
                }
            }
        }

        bool classic = standsAlone ? !carried.Exists(c => c.Entry.Kind == PdfEntryKind.Compressed) : xref.Sections[0].Kind is "table" or "hybrid";
        long size = Math.Max(xref.Count, file.Trailer["Size"u8].TryInteger(out long stated) ? stated : 0);
        return (standsAlone, carried, classic, (int)Math.Min(size, int.MaxValue - 1));
    }

    // The trailer an update writes: the source's /Root, /Info, /Encrypt and /ID kept.
    private static void Trailer(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own, PdfFile file)
    {
        foreach (var key in (ReadOnlySpan<string>)["Root", "Info", "Encrypt", "ID"])
        {
            byte[] name = Encoding.ASCII.GetBytes(key);
            if (!file.Trailer[name].IsNull)
            {
                copier.Entry(s, own, name, file.Trailer[name]);
            }
        }
    }

    // An object of the file written again: its entries but those `skip` names, then what `more` writes.
    private static void Rewrite(PdfFileWriter writer, PdfEditCopier copier, PdfEditCopier.Source own, PdfValue node, PdfDictionary dictionary, Func<PdfName, bool> skip,
        Action<PdfSyntax> more)
    {
        var reference = node.AsReference;
        writer.Reuse(reference.Number, reference.Generation);
        copier.WriteObject(reference.Number, s =>
        {
            s.Raw("<<"u8);
            for (int i = 0; i < dictionary.Count; i++)
            {
                if (!skip(dictionary.KeyAt(i)))
                {
                    copier.Entry(s, own, dictionary.KeyAt(i).Bytes, dictionary.ValueAt(i));
                }
            }

            more(s);
            s.Raw(">>"u8);
        });
    }

    // The new widget added to its page's /Annots (the array written again when it is an object of its own, else the page).
    private static void AddToPage(PdfFile file, PdfFileWriter writer, PdfEditCopier copier, PdfEditCopier.Source own, PdfValue pageNode, int field)
    {
        var page = file.Dictionary(pageNode) ?? throw new InvalidDataException($"{file.Name}: the page a signature field goes on cannot be read.");
        var annots = page["Annots"u8];
        var items = file.Resolve(annots).AsArray?.Items ?? [];
        void List(PdfSyntax s)
        {
            s.Byte((byte)'[');
            foreach (var item in items)
            {
                copier.Value(s, own, item);
                s.Space();
            }

            s.Reference(field).Byte((byte)']');
        }

        if (annots.IsReference && file.Resolve(annots).AsArray is not null)
        {
            writer.Reuse(annots.AsReference.Number, annots.AsReference.Generation);
            copier.WriteObject(annots.AsReference.Number, List);
            return;
        }

        Rewrite(writer, copier, own, pageNode, page, key => key.Is("Annots"u8), s =>
        {
            s.Raw("/Annots"u8);
            List(s);
        });
    }

    // The form's /Fields given the new field (when `field` is above 0) and /SigFlags 3 (signatures exist, append only): the
    // form written again when it is an object of its own, else the catalog with the form inside it (or a form made).
    private static void AddToForm(PdfFile file, PdfFileWriter writer, PdfEditCopier copier, PdfEditCopier.Source own, Target target, int field)
    {
        void Entries(PdfSyntax s, PdfDictionary? form)
        {
            var fields = form is null ? [] : file.Resolve(form["Fields"u8]).AsArray?.Items ?? [];
            for (int i = 0; i < (form?.Count ?? 0); i++)
            {
                var key = form!.KeyAt(i);
                if (!key.Is("Fields"u8) && !key.Is("SigFlags"u8))
                {
                    copier.Entry(s, own, key.Bytes, form.ValueAt(i));
                }
            }

            s.Raw("/Fields["u8);
            foreach (var item in fields)
            {
                copier.Value(s, own, item);
                s.Space();
            }

            if (field > 0)
            {
                s.Reference(field);
            }

            s.Raw("]/SigFlags 3"u8);
        }

        if (target.Form.IsReference && target.FormDictionary is { } formDictionary)
        {
            writer.Reuse(target.Form.AsReference.Number, target.Form.AsReference.Generation);
            copier.WriteObject(target.Form.AsReference.Number, s =>
            {
                s.Raw("<<"u8);
                Entries(s, formDictionary);
                s.Raw(">>"u8);
            });
            return;
        }

        var catalogValue = file.Trailer["Root"u8];
        if (!catalogValue.IsReference)
        {
            throw new InvalidDataException($"{file.Name}: a catalog that is not an object of its own cannot be updated: save the document as a rewrite first.");
        }

        var catalog = file.Dictionary(catalogValue)!;
        if (target.FormDictionary is { } direct)
        {
            // A form inside the catalog: the catalog written again with it.
            Rewrite(writer, copier, own, catalogValue, catalog, key => key.Is("AcroForm"u8), s =>
            {
                s.Raw("/AcroForm<<"u8);
                Entries(s, direct);
                s.Raw(">>"u8);
            });
            return;
        }

        // No form yet: one made, an object of its own (a later signature then writes the form alone again).
        int made = writer.Allocate();
        copier.WriteObject(made, s =>
        {
            s.Raw("<<"u8);
            Entries(s, null);
            s.Raw(">>"u8);
        });
        Rewrite(writer, copier, own, catalogValue, catalog, key => key.Is("AcroForm"u8), s => s.Raw("/AcroForm "u8).Reference(made));
    }

    // The appearance's content: the image (fitted, on the left when there is text) and the text's lines in Helvetica, the
    // largest size that fits, centred in the height.
    private static byte[] Content(SigningOptions options, double width, double height, bool image)
    {
        using var s = new PdfSyntax(1024);
        double pad = Math.Min(width, height) * 0.06 + 1, left = pad;
        if (image && options.AppearanceImage is { } picture && picture.Width > 0 && picture.Height > 0)
        {
            double area = options.AppearanceText is null ? width - 2 * pad : Math.Min(height - 2 * pad, (width - 2 * pad) / 2);
            double scale = Math.Max(0, Math.Min(area / picture.Width, (height - 2 * pad) / picture.Height));
            double w = picture.Width * scale, h = picture.Height * scale;
            s.Raw("q "u8).Number(w, 4).Raw(" 0 0 "u8).Number(h, 4).Space().Number(pad + (area - w) / 2, 4).Space().Number((height - h) / 2, 4).Raw(" cm /Im1 Do Q\n"u8);
            left = pad + area + pad;
        }

        if (options.AppearanceText is { } text)
        {
            var lines = text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
            var codes = lines.Select(Codes).ToArray();
            double widest = codes.Max(line => line.Sum(c => PdfFontTables.StandardWidth(Helvetica, Character(c)))) / 1000.0;
            double size = Math.Min((height - 2 * pad) / (lines.Length * 1.2), widest > 0 ? (width - left - pad) / widest : height);
            size = Math.Clamp(size, 1, 72);
            double top = (height + lines.Length * size * 1.2) / 2;
            s.Raw("BT /F1 "u8).Number(size, 3).Raw(" Tf 0 g\n"u8);
            for (int i = 0; i < codes.Length; i++)
            {
                double baseline = top - size * (1.2 * i + 0.95);
                s.Raw("1 0 0 1 "u8).Number(left, 3).Space().Number(baseline, 3).Raw(" Tm "u8).Literal(codes[i]).Raw(" Tj\n"u8);
            }

            s.Raw("ET\n"u8);
        }

        return s.Span.ToArray();
    }

    // A line's WinAnsi codes: '?' for a character the encoding does not hold (an image draws any script).
    private static byte[] Codes(string line)
    {
        var map = WinAnsi.Value.Codes;
        var codes = new byte[line.Length];
        for (int i = 0; i < line.Length; i++)
        {
            codes[i] = map.TryGetValue(line[i], out byte code) ? code : (byte)'?';
        }

        return codes;
    }

    // The character a WinAnsi code stands for (its width is the standard font's for it).
    private static int Character(byte code) => WinAnsi.Value.Characters[code];

    private static (Dictionary<char, byte>, char[]) WinAnsiCodes()
    {
        var map = new Dictionary<char, byte>();
        char[] characters = new char[256];
        characters.AsSpan().Fill('?');
        Span<char> text = stackalloc char[8];
        for (int code = 32; code < 256; code++)
        {
            var name = PdfFontTables.EncodingName(PdfFontTables.WinAnsiEncoding, code);
            if (!name.IsEmpty && PdfFontTables.GlyphText(name, false, text) == 1)
            {
                map.TryAdd(text[0], (byte)code);
                characters[code] = text[0];
            }
        }

        return (map, characters);
    }

    // B-LT: one more update adding the validation data to the DSS (certificates, revocation lists, OCSP responses; existing
    // ones kept, the same bytes not written twice) and a VRI entry for the signature.
    private static byte[] ValidationData(PdfDocument document, Signed signed, SigningOptions options, DocumentWriteOptions writeOptions)
    {
        var file = document.File;
        var certificates = new List<CmsCertificate>();
        foreach (var certificate in signed.Certificates)
        {
            if (!certificates.Exists(c => c.Same(certificate)))
            {
                certificates.Add(certificate);
            }
        }

        var crls = new List<byte[]>();
        var ocsps = new List<byte[]>();
        foreach (var item in options.Revocation)
        {
            if (Try(() => CmsOcsp.Parse(item)) is not null)
            {
                ocsps.Add(item.ToArray());
            }
            else if (Try(() => CmsCrl.Parse(item)) is not null)
            {
                crls.Add(item.ToArray());
            }
            else
            {
                throw new ArgumentException("SigningOptions.Revocation holds an item that is neither a revocation list nor an OCSP response (DER).", nameof(options));
            }
        }

        if (options.Online)
        {
            Fetch(certificates, crls, ocsps, options, file.MaxStreamBytes);
        }

        var catalogValue = file.Trailer["Root"u8];
        if (!catalogValue.IsReference)
        {
            throw new InvalidDataException($"{file.Name}: a catalog that is not an object of its own cannot be updated.");
        }

        var catalog = file.Dictionary(catalogValue)!;
        var dss = file.Dictionary(catalog["DSS"u8]);
        var xref = file.Xref;
        var (standsAlone, carried, classic, size) = Layout(xref, file);
        bool newline = NeedsNewline(file.Source);
        long prefix = file.Source.Length + (newline ? 1 : 0);
        using var output = new MemoryStream();
        using (var writer = PdfFileWriter.Append(output, prefix, xref.Shift, size, writeOptions.Compression, packing: !classic && writeOptions.ObjectStreams))
        {
            writer.Security = file.Security;
            using var copier = new PdfEditCopier(writer, writeOptions.Compression, recompress: false);
            var own = copier.For(document, identity: true);

            // Each kind's items: the DSS's own (their references and bytes), then the new ones written as streams.
            List<(PdfValue Value, byte[] Bytes)> Existing(ReadOnlySpan<byte> key)
            {
                var list = new List<(PdfValue, byte[])>();
                foreach (var item in dss is null ? [] : file.Resolve(dss[key]).AsArray?.Items ?? [])
                {
                    try
                    {
                        if (file.Resolve(item).AsStream is { } stream)
                        {
                            using var decoded = file.Decode(stream);
                            list.Add((item, decoded.Span.ToArray()));
                        }
                    }
                    catch (InvalidDataException e) when (!PdfLimit.Is(e))
                    {
                        // A damaged item of an earlier DSS is kept as it is, and compared with nothing.
                    }
                }

                return list;
            }

            (List<PdfValue> All, List<PdfValue> Mine) Merge(List<(PdfValue Value, byte[] Bytes)> existing, IEnumerable<byte[]> wanted)
            {
                var all = existing.Select(e => e.Value).ToList();
                var mine = new List<PdfValue>();
                foreach (byte[] data in wanted)
                {
                    int at = existing.FindIndex(e => e.Bytes.AsSpan().SequenceEqual(data));
                    if (at >= 0)
                    {
                        mine.Add(existing[at].Value);
                        continue;
                    }

                    int number = writer.Allocate();
                    writer.BeginStream(number);
                    writer.EndStream(data, deflate: true);
                    var reference = PdfValue.Reference(number, 0);
                    all.Add(reference);
                    mine.Add(reference);
                    existing.Add((reference, data));
                }

                return (all, mine);
            }

            var certs = Merge(Existing("Certs"u8), certificates.Select(c => c.Raw.ToArray()));
            var lists = Merge(Existing("CRLs"u8), crls);
            var answers = Merge(Existing("OCSPs"u8), ocsps);
            void Items(PdfSyntax s, List<PdfValue> items)
            {
                s.Byte((byte)'[');
                foreach (var item in items)
                {
                    copier.Value(s, own, item);
                    s.Space();
                }

                s.Byte((byte)']');
            }

            int vri = writer.Allocate();
            copier.WriteObject(vri, s =>
            {
                s.Raw("<</Cert"u8);
                Items(s, certs.Mine);
                s.Raw("/CRL"u8);
                Items(s, lists.Mine);
                s.Raw("/OCSP"u8);
                Items(s, answers.Mine);
                s.Raw(">>"u8);
            });

            string key = Convert.ToHexString(SHA1.HashData(signed.Cms));
            int dictionary = catalog["DSS"u8].IsReference ? catalog["DSS"u8].AsReference.Number : writer.Allocate();
            if (catalog["DSS"u8].IsReference)
            {
                writer.Reuse(dictionary, catalog["DSS"u8].AsReference.Generation);
            }

            var vris = dss is null ? null : file.Dictionary(dss["VRI"u8]);
            copier.WriteObject(dictionary, s =>
            {
                s.Raw("<</Type/DSS/Certs"u8);
                Items(s, certs.All);
                s.Raw("/CRLs"u8);
                Items(s, lists.All);
                s.Raw("/OCSPs"u8);
                Items(s, answers.All);
                s.Raw("/VRI<<"u8);
                for (int i = 0; i < (vris?.Count ?? 0); i++)
                {
                    if (!vris!.KeyAt(i).ToString().Equals(key, StringComparison.OrdinalIgnoreCase))
                    {
                        copier.Entry(s, own, vris.KeyAt(i).Bytes, vris.ValueAt(i));
                    }
                }

                s.Name(key).Space().Reference(vri).Raw(">>>>"u8);
            });

            if (!catalog["DSS"u8].IsReference)
            {
                Rewrite(writer, copier, own, catalogValue, catalog, k => k.Is("DSS"u8), s => s.Raw("/DSS "u8).Reference(dictionary));
            }

            copier.Drain();
            writer.FinishUpdate(classic, standsAlone ? -1 : xref.Sections[0].Offset, size, carried, s => Trailer(s, copier, own, file));
        }

        return output.ToArray();
    }

    private static T? Try<T>(Func<T> parse) where T : class
    {
        try
        {
            return parse();
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    // Revocation data the app's items lack, fetched (B-LT with Online): for each certificate an issuer of the set issued,
    // its OCSP responder first, then its lists.
    private static void Fetch(List<CmsCertificate> certificates, List<byte[]> crls, List<byte[]> ocsps, SigningOptions options, long maxBytes)
    {
        foreach (var certificate in certificates)
        {
            if (certificate.SelfIssued || certificates.Find(certificate.NamesIssuer) is not { } issuer)
            {
                continue;
            }

            bool known = crls.Exists(c => Try(() => CmsCrl.Parse(c)) is { } crl && crl.Issuer.Span.SequenceEqual(certificate.Issuer.Span))
                         || ocsps.Exists(o => Try(() => CmsOcsp.Parse(o)) is { } ocsp && ocsp.Responses.Any(r => r.Serial == certificate.Serial));
            if (known)
            {
                continue;
            }

            bool fetched = false;
            foreach (string address in certificate.OcspAddresses)
            {
                try
                {
                    byte[] answer = Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.Ocsp)
                    {
                        Body = CmsRevocation.Request(certificate, issuer), Timeout = options.Timeout, MaxResponseBytes = maxBytes,
                    });
                    if (CmsOcsp.Parse(answer).Failure is null)
                    {
                        ocsps.Add(answer);
                        fetched = true;
                        break;
                    }
                }
                catch (Exception e) when (e is IOException or InvalidDataException or UriFormatException or NotSupportedException or OperationCanceledException or TimeoutException)
                {
                    // The next address, then the lists.
                }
            }

            foreach (string address in fetched ? [] : certificate.CrlAddresses)
            {
                try
                {
                    byte[] list = Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.RevocationList)
                    {
                        Timeout = options.Timeout, MaxResponseBytes = maxBytes,
                    });
                    CmsCrl.Parse(list);
                    crls.Add(list);
                    break;
                }
                catch (Exception e) when (e is IOException or InvalidDataException or UriFormatException or NotSupportedException or OperationCanceledException or TimeoutException)
                {
                    // The next address.
                }
            }
        }

        if (crls.Count == 0 && ocsps.Count == 0)
        {
            throw new IOException("PAdES B-LT: no revocation data was given (SigningOptions.Revocation) and none could be fetched from the addresses the certificates name.");
        }
    }
}

/// <summary>
/// A document's bytes after updates, as one read-only seekable stream: its source's bytes, the line end it lacked, then each
/// update; what a document is opened from between signatures. The source stays its document's.
/// </summary>
internal sealed class PdfJoinedStream : Stream
{
    private readonly PdfSource _source;
    private readonly byte[][] _updates;
    private readonly long _head, _length;
    private long _position;

    public PdfJoinedStream(PdfSource source, bool newline, byte[][] updates)
    {
        (_source, _updates) = (source, updates);
        _head = source.Length + (newline ? 1 : 0);
        _length = _head + updates.Sum(u => (long)u.Length);
    }

    public override bool CanRead => true;

    public override bool CanSeek => true;

    public override bool CanWrite => false;

    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set => _position = Math.Clamp(value, 0, _length);
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        int total = 0;
        while (total < buffer.Length && _position < _length)
        {
            int read;
            if (_position < _source.Length)
            {
                read = _source.Read(_position, buffer[total..(int)Math.Min(buffer.Length, total + _source.Length - _position)]);
                if (read <= 0)
                {
                    break;
                }
            }
            else if (_position < _head)
            {
                buffer[total] = (byte)'\n';
                read = 1;
            }
            else
            {
                long at = _position - _head;
                int k = 0;
                while (at >= _updates[k].Length)
                {
                    at -= _updates[k].Length;
                    k++;
                }

                read = Math.Min(buffer.Length - total, _updates[k].Length - (int)at);
                _updates[k].AsSpan((int)at, read).CopyTo(buffer[total..]);
            }

            total += read;
            _position += read;
        }

        return total;
    }

    public override long Seek(long offset, SeekOrigin origin) => Position = origin switch
    {
        SeekOrigin.Begin => offset,
        SeekOrigin.Current => _position + offset,
        _ => _length + offset,
    };

    public override void Flush()
    {
    }

    public override void SetLength(long value) => throw new NotSupportedException("A document's bytes are read, not written.");

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("A document's bytes are read, not written.");
}

/// <summary>
/// Where a builder that signs writes its file before the signatures are added: the destination it was given, until
/// <see cref="Spill"/> (called before anything is written) sends every byte to a temporary file (in the system's temporary
/// folder, deleted when closed) that is opened and signed onto the destination at the end.
/// </summary>
internal sealed class PdfSpillStream(Stream destination) : Stream
{
    private Stream _target = destination;
    private long _written;

    /// <summary>The temporary file the bytes go to, once spilled.</summary>
    public FileStream? Spilled { get; private set; }

    /// <summary>Sends every byte from now on to a temporary file.</summary>
    /// <exception cref="InvalidOperationException">Bytes were written to the destination already.</exception>
    public void Spill()
    {
        if (Spilled is not null)
        {
            return;
        }

        if (_written > 0)
        {
            throw new InvalidOperationException("Sign is called before the first page or attached file: what is written before it would be outside the signed file's first form.");
        }

        Spilled = Create();
        _target = Spilled;
    }

    /// <summary>A temporary file for a document's bytes, deleted when it is closed.</summary>
    public static FileStream Create() =>
        new(Path.Combine(Path.GetTempPath(), "idrak-" + Path.GetRandomFileName() + ".pdf"), FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 1 << 16, FileOptions.DeleteOnClose);

    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => true;

    public override long Length => _written;

    public override long Position
    {
        get => _written;
        set => throw new NotSupportedException("A document is written in order.");
    }

    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

    public override void Write(ReadOnlySpan<byte> buffer)
    {
        _target.Write(buffer);
        _written += buffer.Length;
    }

    public override void Flush() => _target.Flush();

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException("A document is written, not read.");

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException("A document is written in order.");

    public override void SetLength(long value) => throw new NotSupportedException("A document is written in order.");

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Spilled?.Dispose();
            Spilled = null;
        }

        base.Dispose(disposing);
    }
}
