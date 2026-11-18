// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A document file a format is checked on (<see cref="DocumentFormatSuite"/>): one it reads, with what it must give when
/// known, or one it must refuse.
/// </summary>
/// <param name="Path">The file.</param>
/// <param name="Pages">The pages it holds; null to check the format's own consistency only.</param>
/// <param name="Texts">Each page's text as the layout gives it (with the default options); null when not known.</param>
/// <param name="Error">
/// When set, the file is not valid: opening it and reading its pages must fail with an <see cref="InvalidDataException"/> or
/// a <see cref="NotSupportedException"/> whose message contains this text.
/// </param>
public sealed record DocumentSample(string Path, int? Pages = null, IReadOnlyList<string>? Texts = null, string? Error = null);

/// <summary>
/// The checks of a document format (<see cref="IDocumentFormat"/>, registered in <see cref="DocumentFormats"/>) on files of
/// its own. For each valid sample:
/// <list type="bullet">
/// <item>the format accepts the file's first bytes and refuses other formats' (<see cref="Foreign"/>);</item>
/// <item>it opens, its pages count as expected, each page reads with its index and a size that is not negative, every
/// run's box inside its page, the expected text, every image decoding to its size or refused naming its encoding, and the
/// same page read again is the same;</item>
/// <item>pages read in parallel give the same as one by one (a document is read from several threads);</item>
/// <item>its information and attachments read (each attachment its size), a page past the last is refused;</item>
/// <item>with tight limits, reading either passes or fails with an <see cref="InvalidDataException"/> naming the limit.</item>
/// </list>
/// An invalid sample fails saying what is wrong. The random cases damage a sample (cut at a random length, or random bytes
/// changed): opening and reading it then passes or fails with an <see cref="InvalidDataException"/>, never another
/// exception (an image's decoding may also say its encoding is not read).
/// </summary>
public sealed class DocumentFormatSuite : ContractSuite<IDocumentFormat>
{
    private static readonly string[] LimitNames = [nameof(DocumentOptions.MaxStreamBytes), nameof(DocumentOptions.MaxObjects), nameof(DocumentOptions.MaxDepth)];

    private readonly IReadOnlyList<DocumentSample> _samples;
    private readonly DocumentOptions _options;

    /// <summary>The checks on <paramref name="samples"/>.</summary>
    /// <param name="samples">Files in the format checked (at least one valid one).</param>
    /// <param name="foreign">File starts of other formats the format must refuse; null for <see cref="DefaultForeign"/>.</param>
    /// <param name="options">The options documents are opened with; null for the defaults.</param>
    public DocumentFormatSuite(IReadOnlyList<DocumentSample> samples, IReadOnlyList<byte[]>? foreign = null, DocumentOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (!samples.Any(s => s.Error is null))
        {
            throw new ArgumentException("A document format is checked on files of its own: give at least one valid sample.", nameof(samples));
        }

        _samples = samples;
        Foreign = foreign ?? DefaultForeign;
        _options = options ?? new DocumentOptions();
    }

    /// <summary>The starts of other formats' files a document format refuses by default: empty, PNG, JPEG, GIF, WAVE and a line of plain text.</summary>
    public static IReadOnlyList<byte[]> DefaultForeign { get; } =
    [
        [],
        [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, (byte)'I', (byte)'H', (byte)'D', (byte)'R'],
        [0xFF, 0xD8, 0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0],
        "GIF89a"u8.ToArray(),
        "RIFF$\0\0\0WAVEfmt "u8.ToArray(),
        "An ordinary letter: thank you for your note of last week.\n"u8.ToArray(),
    ];

    /// <summary>The file starts the format must refuse.</summary>
    public IReadOnlyList<byte[]> Foreign { get; }

    /// <inheritdoc />
    public override string Name => "document-format";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < _samples.Count; i++)
        {
            yield return Case($"{System.IO.Path.GetFileName(_samples[i].Path)}, whole", i, "whole", 0);
        }

        int first = Enumerable.Range(0, _samples.Count).First(i => _samples[i].Error is null);
        yield return Case($"{System.IO.Path.GetFileName(_samples[first].Path)}, with tight limits", first, "limits", 0);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int[] valid = [.. Enumerable.Range(0, _samples.Count).Where(i => _samples[i].Error is null)];
        int index = valid[random.Next(valid.Length)];
        bool cut = random.Next(2) == 0;
        return Case($"{System.IO.Path.GetFileName(_samples[index].Path)}, {(cut ? "cut" : "with bytes changed")}", index, cut ? "cut" : "flip", random.Next());
    }

    /// <inheritdoc />
    public override void Run(IDocumentFormat implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        string mode = (string)@case.Data["mode"]!;
        byte[] bytes = File.ReadAllBytes(sample.Path);
        string file = System.IO.Path.GetFileName(sample.Path);
        switch (mode)
        {
            case "whole" when sample.Error is { } error:
                RunInvalid(implementation, file, bytes, error, checks);
                break;
            case "whole":
                RunWhole(implementation, sample, file, bytes, checks);
                break;
            case "limits":
                RunLimits(implementation, file, bytes, checks);
                break;
            default:
                RunDamaged(implementation, file, Damage(bytes, mode == "cut", new Random((int)@case.Data["seed"]!)), checks);
                break;
        }
    }

    private void RunWhole(IDocumentFormat implementation, DocumentSample sample, string file, byte[] bytes, CaseChecks checks)
    {
        checks.Check("accepts its own files", implementation.Accepts(bytes.AsSpan(0, Math.Min(bytes.Length, DocumentFormats.HeaderBytes))), $"{file}: its first bytes are refused");
        int accepted = Foreign.Count(f => implementation.Accepts(f));
        checks.Check("refuses other formats' files", accepted == 0, $"{accepted} of {Foreign.Count} other formats' starts are accepted");

        using var stream = new FileStream(sample.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var document = implementation.Open(stream, file, _options);
        checks.Check("the document names its format", string.Equals(document.Format, implementation.Name, StringComparison.OrdinalIgnoreCase),
            $"{file}: format \"{document.Format}\", the format is \"{implementation.Name}\"");
        int count = document.PageCount;
        checks.Check("the page count", count >= 0 && (sample.Pages is null || sample.Pages == count), $"{file}: {count} pages, expected {sample.Pages}");
        checks.Check("the information reads", document.Info is not null, $"{file}: no information");

        var pages = new string[count];
        for (int i = 0; i < count; i++)
        {
            var page = document.ReadPage(i, DocumentParts.All);
            pages[i] = DocumentChecks.Signature(page);
            checks.Check("each page is its index, of a size that is not negative", page.Index == i && page.Width >= 0 && page.Height >= 0,
                $"{file}: page {i} reads as page {page.Index}, {page.Width} x {page.Height}");
            var outside = page.Width > 0 && page.Height > 0 ? page.Runs.FirstOrDefault(r => !r.Box.IsEmpty && !r.Box.Within(page.Box, 0.5f)) : null;
            checks.Check("every run's box is inside its page", outside is null, $"{file}: page {i}: \"{outside?.Text}\" at {outside?.Box} on a page of {page.Width} x {page.Height}");
            if (sample.Texts is { } texts && i < texts.Count)
            {
                checks.Compare("the expected text", Comparisons.Difference(texts[i], page.Text));
            }

            foreach (var image in page.Images)
            {
                try
                {
                    var pixels = image.Decode();
                    checks.Check("every image decodes to its size", pixels.Width == image.PixelWidth && pixels.Height == image.PixelHeight,
                        $"{file}: page {i}: an image of {image.PixelWidth} x {image.PixelHeight} decodes to {pixels.Width} x {pixels.Height}");
                }
                catch (NotSupportedException e)
                {
                    checks.Check("an image no codec reads names its encoding", e.Message.Contains(image.Encoding, StringComparison.OrdinalIgnoreCase),
                        $"{file}: page {i}: \"{e.Message}\" does not name the encoding {image.Encoding}");
                }
            }

            checks.Compare("the same page read again", Comparisons.Difference(pages[i], DocumentChecks.Signature(document.ReadPage(i, DocumentParts.All))));
        }

        var parallel = new string[count];
        Parallel.For(0, count, i => parallel[i] = DocumentChecks.Signature(document.ReadPage(i, DocumentParts.All)));
        checks.Compare("pages read in parallel give the same", Comparisons.Sequences(pages, parallel));

        foreach (var attachment in document.Attachments)
        {
            using var content = attachment.Read();
            using var copy = new MemoryStream();
            content.CopyTo(copy);
            checks.Check("every attachment reads its size", attachment.Size < 0 || attachment.Size == copy.Length, $"{file}: {attachment.Name} reads {copy.Length} bytes of {attachment.Size}");
        }

        try
        {
            document.ReadPage(count, DocumentParts.Text);
            checks.Fail("a page past the last is refused", $"{file}: page {count} of {count} read");
        }
        catch (ArgumentOutOfRangeException)
        {
            checks.Pass("a page past the last is refused");
        }
    }

    private void RunInvalid(IDocumentFormat implementation, string file, byte[] bytes, string error, CaseChecks checks)
    {
        try
        {
            using var document = implementation.Open(new MemoryStream(bytes, writable: false), file, _options);
            ReadEverything(document);
            checks.Fail("an invalid file fails saying what is wrong", $"{file} read without an error (expected one about \"{error}\")");
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            checks.Check("an invalid file fails saying what is wrong", e.Message.Contains(error, StringComparison.Ordinal), $"\"{e.Message}\" does not say \"{error}\"");
        }
    }

    private void RunLimits(IDocumentFormat implementation, string file, byte[] bytes, CaseChecks checks)
    {
        var tight = _options with { MaxStreamBytes = 1, MaxObjects = 1, MaxDepth = 1 };
        try
        {
            using var document = implementation.Open(new MemoryStream(bytes, writable: false), file, tight);
            ReadEverything(document);
            checks.Pass("a limit passed fails naming it");
        }
        catch (InvalidDataException e)
        {
            checks.Check("a limit passed fails naming it", LimitNames.Any(n => e.Message.Contains(n, StringComparison.Ordinal)),
                $"{file}: \"{e.Message}\" names none of {string.Join(", ", LimitNames)}");
        }
        catch (Exception e)
        {
            checks.Fail("a limit passed fails naming it", $"{file}: {e.GetType().Name}: {e.Message}");
        }
    }

    private void RunDamaged(IDocumentFormat implementation, string file, byte[] damaged, CaseChecks checks)
    {
        try
        {
            implementation.Accepts(damaged.AsSpan(0, Math.Min(damaged.Length, DocumentFormats.HeaderBytes)));
            using var document = implementation.Open(new MemoryStream(damaged, writable: false), file, _options);
            ReadEverything(document);
            checks.Pass("a damaged file reads or fails as damaged");
        }
        catch (InvalidDataException)
        {
            checks.Pass("a damaged file reads or fails as damaged");
        }
        catch (Exception e)
        {
            checks.Fail("a damaged file reads or fails as damaged", $"{file} damaged ({damaged.Length} bytes): {e.GetType().Name}: {e.Message}");
        }
    }

    // Every page with every part, every image decoded (an encoding no codec reads may say so), every attachment read.
    private static void ReadEverything(IDocument document)
    {
        _ = document.Info;
        for (int i = 0; i < document.PageCount; i++)
        {
            var page = document.ReadPage(i, DocumentParts.All);
            foreach (var image in page.Images)
            {
                try
                {
                    _ = image.Decode();
                }
                catch (NotSupportedException) when (!string.Equals(image.Encoding, DocumentImage.Raw, StringComparison.OrdinalIgnoreCase))
                {
                    // An encoding no registered codec reads: refused as such, which is not damage.
                }
            }
        }

        foreach (var attachment in document.Attachments)
        {
            using var content = attachment.Read();
            content.CopyTo(Stream.Null);
        }
    }

    // The sample cut at a random length, or with a few random bytes changed.
    private static byte[] Damage(byte[] bytes, bool cut, Random random)
    {
        if (cut)
        {
            return bytes[..random.Next(0, bytes.Length)];
        }

        byte[] changed = (byte[])bytes.Clone();
        for (int k = 0, n = 1 + random.Next(8); k < n && changed.Length > 0; k++)
        {
            changed[random.Next(changed.Length)] = (byte)random.Next(256);
        }

        return changed;
    }

    private static ContractCase Case(string name, int sample, string mode, int seed) =>
        new(name, new JsonObject { ["sample"] = sample, ["mode"] = mode, ["seed"] = seed });
}

// What the document suites compare pages and images by.
internal static class DocumentChecks
{
    // A page as text: its size, runs (text, box, size, mode, order), text, images (encoding, size, box), paths (verbs,
    // points, fill, stroke) and issue kinds; boxes and points rounded to a thousandth of a point.
    public static string Signature(DocumentPage page)
    {
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"page {page.Index} {R(page.Width)} x {R(page.Height)} turned {page.Rotation}\n");
        foreach (var run in page.Runs)
        {
            text.Append(CultureInfo.InvariantCulture, $"run \"{run.Text}\" {Box(run.Box)} {R(run.FontSize)} {run.Mode} {run.Order} glyphs {run.Glyphs.Length}\n");
        }

        text.Append("text \"").Append(page.Text).Append("\"\n");
        foreach (var image in page.Images)
        {
            text.Append(CultureInfo.InvariantCulture, $"image {image.Encoding} {image.PixelWidth} x {image.PixelHeight} {Box(image.Box)} {image.Order}\n");
        }

        foreach (var path in page.Paths)
        {
            text.Append(CultureInfo.InvariantCulture, $"path {path.Verbs.Length} verbs {string.Join(" ", path.Points.ToArray().Select(R))} {path.Fill} {path.Stroke?.ToString(CultureInfo.InvariantCulture) ?? "-"} {path.Order}\n");
        }

        foreach (var issue in page.Issues)
        {
            text.Append(CultureInfo.InvariantCulture, $"issue {issue.Kind} {issue.Count}\n");
        }

        return text.ToString();
    }

    public static string Box(DocumentBox box) => $"({R(box.X)}, {R(box.Y)}, {R(box.Width)}, {R(box.Height)})";

    public static string R(float value) => MathF.Round(value, 3).ToString("0.###", CultureInfo.InvariantCulture);

    // The value of pixel (x, y) of channel c (clamped to the image).
    public static float Pixel(ImageData image, int channel, int x, int y)
    {
        x = Math.Clamp(x, 0, image.Width - 1);
        y = Math.Clamp(y, 0, image.Height - 1);
        return image.Pixels[(channel * image.Height + y) * image.Width + x];
    }

    // A pixel's grey level: its one channel, or the luma of its three.
    public static float Grey(ImageData image, int x, int y) =>
        image.Channels == 1 ? Pixel(image, 0, x, y) : 0.299f * Pixel(image, 0, x, y) + 0.587f * Pixel(image, 1, x, y) + 0.114f * Pixel(image, 2, x, y);
}
