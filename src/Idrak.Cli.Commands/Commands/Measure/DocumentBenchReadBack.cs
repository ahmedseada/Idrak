// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// What an encrypted output reads back to, hashed, for <c>idrak bench documents --hash</c> (plan 28, D13; the owner's
/// decision 3a). An encrypted save never repeats byte for byte (a fresh random IV, as ISO 32000-2 asks: D11's decision 1),
/// so such an output is proven unchanged by what it reads back to, opened with the options (and so the password) its case
/// opened the source with: the document's information and how it is encrypted (<see cref="DocumentInfo.Security"/>: the
/// handler, its revision, the key's length, the ciphers, the permissions), every page's label, size, rotation, text, runs
/// (text, box, font, size, colours, glyphs, order), images (each one's decoded pixels and mask), paths, clips, issues and
/// annotations (an appearance's content too), the fields with their values, the outline, the named destinations, the
/// attached files' contents, and each signature's check (whether each step passed). Left out: what the standard makes
/// random (the IVs, the file key, the salts in <c>/O</c>, <c>/U</c>, <c>/OE</c>, <c>/UE</c> and <c>/Perms</c>, the file's
/// identifier), where objects lie in the file (a signature's byte range: an encrypted string's escapes move them), and a
/// check's times and wording. Written in a fixed order: numbers by their bits, little-endian; text as UTF-8 after its
/// length; a dictionary by its sorted keys; trees in pre-order with each node's children counted. The same document hashes
/// the same on every run, culture and machine.
/// </summary>
internal static class DocumentBenchReadBack
{
    /// <summary>
    /// The hash (SHA-256, hex) of <paramref name="output"/>, a case's whole output for <paramref name="file"/>, and whether
    /// it was read back: an encrypted output by what it reads back to (<see cref="Write"/>), one that no format opens by
    /// why (the same text every run, so a writer that starts writing unreadable files is seen); one that opens unencrypted
    /// by its bytes, as every other output is hashed.
    /// </summary>
    public static (string Output, bool ReadBack) Hash(ReadOnlyMemory<byte> output, string file, DocumentOptions read)
    {
        using var canon = new Canon();
        IDocument document;
        try
        {
            document = DocumentFormats.Open(output, Path.GetFileName(file), read);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
        {
            canon.Text("unreadable");
            canon.Text(e.GetType().Name + ": " + e.Message);
            return (canon.Finish(), true);
        }

        using (document)
        {
            if (!document.Info.Encrypted)
            {
                return (Convert.ToHexStringLower(SHA256.HashData(output.Span)), false);
            }

            try
            {
                Write(canon, document);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
            {
                canon.Text("failed");      // read so far, then why the rest was not
                canon.Text(e.GetType().Name + ": " + e.Message);
            }

            return (canon.Finish(), true);
        }
    }

    // The document, in a fixed order.
    private static void Write(Canon c, IDocument document)
    {
        c.Text(document.Format);
        Info(c, document.Info);
        c.Int(document.PageCount);
        for (int i = 0; i < document.PageCount; i++)
        {
            c.Text(document.PageLabel(i));
            var page = document.ReadPage(i, DocumentParts.All);
            Content(c, page);
            Annotations(c, page.Annotations);
        }

        Fields(c, document.Fields);
        Outline(c, document.Outline);
        c.Int(document.Destinations.Count);
        foreach (var destination in document.Destinations)
        {
            c.Destination(destination);
        }

        c.Int(document.Attachments.Count);
        foreach (var attachment in document.Attachments)
        {
            c.Attachment(attachment);
        }

        Signatures(c, document.Signatures);
    }

    private static void Info(Canon c, DocumentInfo info)
    {
        c.Text(info.Title);
        c.Text(info.Author);
        c.Text(info.Subject);
        c.Text(info.Keywords);
        c.Text(info.Creator);
        c.Text(info.Producer);
        c.Date(info.Created);
        c.Date(info.Modified);
        c.Text(info.Version);
        c.Bool(info.Encrypted);
        c.Bool(info.Repaired);
        c.Int(info.Entries.Count);
        foreach (string key in info.Entries.Keys.Order(StringComparer.Ordinal))
        {
            c.Text(key);
            c.Text(info.Entries[key]);
        }

        // How it is encrypted: only what repeats (no key, salt, IV or identifier is in it).
        c.Bool(info.Security is not null);
        if (info.Security is { } security)
        {
            c.Text(security.Handler);
            c.Int(security.Revision);
            c.Int(security.KeyBits);
            c.Text(security.Strings);
            c.Text(security.Streams);
            c.Text(security.EmbeddedFiles);
            c.Int((int)security.Permissions);
            c.Bool(security.MetadataEncrypted);
        }
    }

    // A page's content: its size and turn, its text, runs, images, paths, clips and issues (not its annotations).
    private static void Content(Canon c, DocumentPage page)
    {
        c.Int(page.Index);
        c.Float(page.Width);
        c.Float(page.Height);
        c.Int(page.Rotation);
        c.Box(page.Box);
        c.Text(page.Text);
        c.Int(page.Lines.Count);
        c.Int(page.Runs.Count);
        foreach (var run in page.Runs)
        {
            c.Int(run.Order);
            c.Text(run.Text);
            c.Box(run.Box);
            c.Float(run.FontSize);
            c.Text(run.Font);
            c.Int((int)run.Mode);
            c.Colour(run.Colour);
            c.Colour(run.StrokeColour);
            c.Float(run.StrokeWidth);
            c.Float(run.Baseline);
            c.Matrix(run.GlyphMatrix);
            c.Int(run.Clip);
            c.Int(run.Group);
            c.Int(run.SoftMask);
            c.Int((int)run.Blend);
            c.Bool(run.Logical);
            c.Glyphs(run.Glyphs.Span);
        }

        c.Int(page.Images.Count);
        foreach (var image in page.Images)
        {
            c.Int(image.Order);
            c.Text(image.Encoding);
            c.Int(image.PixelWidth);
            c.Int(image.PixelHeight);
            c.Box(image.Box);
            c.Matrix(image.Matrix);
            c.Int(image.Clip);
            c.Int(image.Group);
            c.Int(image.SoftMask);
            c.Int((int)image.Blend);
            c.Float(image.Alpha);
            c.Colour(image.Stencil);
            try
            {
                c.Image(image.Decode());
                var mask = image.Mask();
                c.Bool(mask is not null);
                if (mask is not null)
                {
                    c.Image(mask);
                }
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException)
            {
                c.Text("not decoded: " + e.Message);
            }
        }

        Paths(c, page.Paths);
        Paths(c, page.Clips);
        c.Int(page.Issues.Count);
        foreach (var issue in page.Issues)
        {
            c.Text(issue.Kind);
            c.Text(issue.Detail);
            c.Int(issue.Count);
        }

        c.Int(page.Groups.Count);
        c.Int(page.Shadings.Count);
        c.Int(page.SoftMasks.Count);
    }

    private static void Paths(Canon c, IReadOnlyList<DocumentPath> paths)
    {
        c.Int(paths.Count);
        foreach (var path in paths)
        {
            c.Int(path.Order);
            c.Bytes(MemoryMarshal.AsBytes(path.Verbs.Span));     // one byte a verb
            c.Floats(path.Points.Span);
            c.Int((int)path.Fill);
            c.Colour(path.FillColour);
            c.Float(path.Stroke);
            c.Colour(path.StrokeColour);
            c.Int((int)path.Cap);
            c.Int((int)path.Join);
            c.Floats(path.Dash.Span);
            c.Float(path.DashPhase);
            c.Float(path.MiterLimit);
            c.Int(path.Clip);
            c.Int(path.ClipRun);
            c.Int(path.ClipRunCount);
            c.Int(path.Group);
            c.Int(path.SoftMask);
            c.Int((int)path.Blend);
            c.Pattern(path.FillPattern);
            c.Pattern(path.StrokePattern);
        }
    }

    // A page's annotations and their replies, in pre-order (each with its depth and its replies counted), on a stack of
    // our own: a long chain of replies takes no stack.
    private static void Annotations(Canon c, IReadOnlyList<DocumentAnnotation> annotations)
    {
        c.Int(annotations.Count);
        var stack = new Stack<(DocumentAnnotation Annotation, int Depth)>();
        for (int i = annotations.Count - 1; i >= 0; i--)
        {
            stack.Push((annotations[i], 0));
        }

        while (stack.TryPop(out var top))
        {
            var a = top.Annotation;
            c.Int(top.Depth);
            c.Text(a.Kind);
            c.Int(a.Page);
            c.Box(a.Box);
            c.Int((int)a.Flags);
            c.Text(a.Contents);
            c.Text(a.Uri);
            c.Text(a.Author);
            c.Text(a.Id);
            c.Text(a.InReplyTo);
            c.Text(a.Parent);
            c.Text(a.Popup);
            c.Text(a.State);
            c.Text(a.StateModel);
            c.Text(a.ReplyType);
            c.Text(a.Icon);
            c.Text(a.OverlayText);
            c.Bool(a.Open);
            c.Float(a.Opacity);
            c.Float(a.BorderWidth);
            c.Colour(a.Colour);
            c.Colour(a.InteriorColour);
            c.Destination(a.Destination);
            c.Date(a.Created);
            c.Date(a.Modified);
            c.Floats(a.Points.Span);
            c.Int(a.LineEndings.Count);
            foreach (string ending in a.LineEndings)
            {
                c.Text(ending);
            }

            Paths(c, a.Ink);
            c.Bool(a.Attachment is not null);
            if (a.Attachment is { } attachment)
            {
                c.Attachment(attachment);
            }

            c.Bool(a.Appearance is not null);
            if (a.Appearance is { } appearance)
            {
                Content(c, appearance);
            }

            c.Int(a.Replies.Count);
            for (int i = a.Replies.Count - 1; i >= 0; i--)
            {
                stack.Push((a.Replies[i], top.Depth + 1));
            }
        }
    }

    private static void Fields(Canon c, IReadOnlyList<DocumentField> fields)
    {
        c.Int(fields.Count);
        foreach (var field in fields)
        {
            c.Text(field.Name);
            c.Int((int)field.Kind);
            c.Text(field.Value);
            c.Int(field.Values.Count);
            foreach (string value in field.Values)
            {
                c.Text(value);
            }

            c.Text(field.DefaultValue);
            c.Int((int)field.Flags);
            c.Bool(field.MaxLength.HasValue);
            c.Int(field.MaxLength ?? 0);
            c.Int(field.Options.Count);
            foreach (var option in field.Options)
            {
                c.Text(option.Text);
                c.Text(option.Export);
            }

            c.Int(field.Widgets.Count);
            foreach (var widget in field.Widgets)
            {
                c.Int(widget.Page);
                c.Box(widget.Box);
                c.Text(widget.State);
                c.Text(widget.OnState);
            }

            c.Text(field.Caption);
            c.Text(field.Tooltip);
            c.Text(field.Font);
            c.Float(field.FontSize);
            c.Int((int)field.Alignment);
            c.Int((int)field.Mark);
            c.Colour(field.TextColour);
            c.Colour(field.Background);
            c.Colour(field.Border);
        }
    }

    // The outline in pre-order, each item with its depth and its children counted, on a stack of our own (a deep outline
    // takes no stack).
    private static void Outline(Canon c, IReadOnlyList<DocumentOutline> outline)
    {
        c.Int(outline.Count);
        var stack = new Stack<(DocumentOutline Item, int Depth)>();
        for (int i = outline.Count - 1; i >= 0; i--)
        {
            stack.Push((outline[i], 0));
        }

        while (stack.TryPop(out var top))
        {
            var item = top.Item;
            c.Int(top.Depth);
            c.Text(item.Title);
            c.Bool(item.Open);
            c.Int((int)item.Style);
            c.Text(item.Uri);
            c.Colour(item.Colour);
            c.Destination(item.Destination);
            c.Int(item.Children.Count);
            for (int i = item.Children.Count - 1; i >= 0; i--)
            {
                stack.Push((item.Children[i], top.Depth + 1));
            }
        }
    }

    // Each signature and its check: whether each step passed, not its details or times. Checked at the time the signer
    // states (else the library's own choice), trusting no root and fetching nothing, so the check is the same in every run
    // whatever the bench was given: the chain step fails alike everywhere, and the digest, the signature and the changes
    // after it say whether the signed file reads back the same.
    private static void Signatures(Canon c, IReadOnlyList<DocumentSignature> signatures)
    {
        c.Int(signatures.Count);
        foreach (var signature in signatures)
        {
            c.Text(signature.Field);
            c.Int((int)signature.Kind);
            c.Text(signature.KindName);
            c.Text(signature.Reason);
            c.Text(signature.Location);
            c.Int((int)signature.Changes);
            c.Bool(signature.CoversWholeFile);
            c.Text(signature.Signer?.Thumbprint);
            try
            {
                var check = signature.Check(new SignatureCheckOptions { Trusted = [], UseMachineStore = false, Online = false, At = signature.SignedAt });
                c.Bool(check.Valid);
                c.Int(check.Steps.Count);
                foreach (var step in check.Steps)
                {
                    c.Text(step.Name);
                    c.Bool(step.Passed);
                }
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or CryptographicException)
            {
                c.Text("not checked: " + e.Message);
            }
        }
    }

    // The canonical writer: every value as fixed bytes into one SHA-256.
    private sealed class Canon : IDisposable
    {
        private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        public void Dispose() => _hash.Dispose();

        public string Finish() => Convert.ToHexStringLower(_hash.GetHashAndReset());

        public void Int(long value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(long)];
            BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
            _hash.AppendData(bytes);
        }

        public void Bool(bool value) => Int(value ? 1 : 0);

        public void Float(float value) => Int(BitConverter.SingleToInt32Bits(value));

        public void Float(float? value)
        {
            Bool(value.HasValue);
            Float(value ?? 0f);
        }

        // Text as UTF-8 after its length in bytes (-1: none), so no two texts run together.
        public void Text(string? text)
        {
            if (text is null)
            {
                Int(-1);
                return;
            }

            int most = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[]? rented = null;
            Span<byte> bytes = most <= 256 ? stackalloc byte[256] : (rented = ArrayPool<byte>.Shared.Rent(most));
            try
            {
                int count = Encoding.UTF8.GetBytes(text, bytes);
                Int(count);
                _hash.AppendData(bytes[..count]);
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }

        public void Bytes(ReadOnlySpan<byte> bytes)
        {
            Int(bytes.Length);
            _hash.AppendData(bytes);
        }

        // Floats by their bits, little-endian on every machine.
        public void Floats(ReadOnlySpan<float> values)
        {
            Int(values.Length);
            if (BitConverter.IsLittleEndian)
            {
                _hash.AppendData(MemoryMarshal.AsBytes(values));
                return;
            }

            Span<byte> bytes = stackalloc byte[sizeof(float)];
            foreach (float value in values)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
                _hash.AppendData(bytes);
            }
        }

        public void Date(DateTimeOffset? date)
        {
            Bool(date.HasValue);
            Int(date?.UtcTicks ?? 0);
            Int((long)(date?.Offset.TotalMinutes ?? 0));
        }

        public void Box(DocumentBox box)
        {
            Float(box.X);
            Float(box.Y);
            Float(box.Width);
            Float(box.Height);
        }

        public void Matrix(DocumentMatrix? matrix)
        {
            Bool(matrix.HasValue);
            if (matrix is { } m)
            {
                Float(m.A);
                Float(m.B);
                Float(m.C);
                Float(m.D);
                Float(m.E);
                Float(m.F);
            }
        }

        public void Colour(DocumentColour? colour)
        {
            Bool(colour.HasValue);
            if (colour is { } value)
            {
                Int((int)value.Space);
                Int(value.Count);
                for (int i = 0; i < value.Count; i++)
                {
                    Float(value[i]);
                }

                Float(value.Alpha);
            }
        }

        public void Destination(DocumentDestination? destination)
        {
            Bool(destination is not null);
            if (destination is not null)
            {
                Int(destination.Page);
                Int((int)destination.Fit);
                Float(destination.X);
                Float(destination.Y);
                Float(destination.Right);
                Float(destination.Bottom);
                Float(destination.Zoom);
                Text(destination.Name);
            }
        }

        // A pattern's place and steps (its cell's content is drawing, not what the path reads back to).
        public void Pattern(DocumentPattern? pattern)
        {
            Bool(pattern is not null);
            if (pattern is not null)
            {
                Box(pattern.Bounds);
                Matrix(pattern.Matrix);
                Float(pattern.XStep);
                Float(pattern.YStep);
                Bool(pattern.Cell is not null);
                Bool(pattern.Shading is not null);
            }
        }

        public void Glyphs(ReadOnlySpan<DocumentGlyph> glyphs)
        {
            Int(glyphs.Length);
            Span<byte> bytes = stackalloc byte[16];
            foreach (var glyph in glyphs)
            {
                BinaryPrimitives.WriteInt32LittleEndian(bytes, glyph.Glyph);
                BinaryPrimitives.WriteSingleLittleEndian(bytes[4..], glyph.X);
                BinaryPrimitives.WriteSingleLittleEndian(bytes[8..], glyph.Y);
                BinaryPrimitives.WriteSingleLittleEndian(bytes[12..], glyph.Advance);
                _hash.AppendData(bytes);
            }
        }

        // An image's decoded values: its 8-bit planes and their levels when it holds them, else its values.
        public void Image(ImageData image)
        {
            Int(image.Channels);
            Int(image.Height);
            Int(image.Width);
            if (image.TryGetPlanes(out var planes, out var levels))
            {
                Int(planes.Length);
                foreach (var plane in planes)
                {
                    Bytes(plane.Span);
                }

                Floats(levels.Span);
            }
            else
            {
                Int(-1);
                Floats(image.Pixels);
            }
        }

        // An attached file: its name, description, type and size, and its contents' own SHA-256.
        public void Attachment(DocumentAttachment attachment)
        {
            Text(attachment.Name);
            Text(attachment.Description);
            Text(attachment.MediaType);
            Int(attachment.Size);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(81920);
            try
            {
                using var contents = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                using (var stream = attachment.Read())
                {
                    int read;
                    while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        contents.AppendData(buffer, 0, read);
                    }
                }

                Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
                contents.GetHashAndReset(digest);
                Bytes(digest);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
            {
                Text("unreadable: " + e.Message);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }
}
