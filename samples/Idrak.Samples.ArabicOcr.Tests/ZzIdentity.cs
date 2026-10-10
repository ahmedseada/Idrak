// TEMPORARY identity harness (not committed): hashes of the app's outputs, compared before and after a change.
using System.Security.Cryptography;
using System.Text;
using Idrak;
using Idrak.Data.Abstractions;

namespace Idrak.Samples.ArabicOcr.Tests;

internal static class ZzIdentity
{
    public static void Run()
    {
        Console.WriteLine("    segment   " + Segments());
        Console.WriteLine("    align     " + Alignments());
        Console.WriteLine("    order     " + Orders());
        Console.WriteLine("    encode    " + Encodes());
        Console.WriteLine("    prepare   " + Prepares());
        if (Environment.GetEnvironmentVariable("ZZ_TIME") is "1")
        {
            Time();
        }
    }

    private static void Time()
    {
        var r = new Random(9);
        var (page, _) = Synthetic.Page([.. Enumerable.Range(0, 40).Select(_ => Synthetic.Text(r, 50, 60))], r, 1.5);
        int n = page.Width * page.Height;
        var colour = new ImageData([.. page.Pixels, .. page.Pixels, .. page.Pixels], 3, page.Height, page.Width);
        foreach (var (name, image) in new[] { ("grey", page), ("rgb", colour) })
        {
            for (int i = 0; i < 3; i++) { LineSegmenter.Find(image); }
            Array.Clear(LineSegmenter.Zz);
            long bytes = GC.GetTotalAllocatedBytes(true);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            const int Runs = 10;
            for (int i = 0; i < Runs; i++) { LineSegmenter.Find(image); }
            Console.WriteLine("    split " + string.Join(" ", LineSegmenter.Zz.Select(z => (z / Runs / 1).ToString("F1")))); Array.Clear(LineSegmenter.Zz);
            Console.WriteLine($"    find {name} {page.Width}x{page.Height}: {clock.Elapsed.TotalMilliseconds / Runs:F1} ms, {(GC.GetTotalAllocatedBytes(true) - bytes) / Runs / 1024} KB");
        }
    }

    private static readonly string[] Letters = ["ا", "ب", "ت", "ث", "د", "ر", "س", "م"];

    private static string[] Texts(Random r, int n) => [.. Enumerable.Range(0, n).Select(_ => Synthetic.Text(r, 3, 12))];

    private static IEnumerable<ImageData> Pages()
    {
        double[] skews = [0, 2, -1.7, 2.9, 0.6, -2.4];
        for (int i = 0; i < skews.Length; i++)
        {
            var r = new Random(100 + i);
            var (page, _) = Synthetic.Page(Texts(r, 3 + i), r, skews[i]);
            yield return page;
            // light ink on dark, and a 3-channel page
            var inverted = page.Pixels.Select(v => 1f - v).ToArray();
            yield return new ImageData(inverted, 1, page.Height, page.Width);
            int n = page.Width * page.Height;
            var colour = new float[3 * n];
            for (int c = 0; c < 3; c++)
            {
                for (int k = 0; k < n; k++)
                {
                    colour[c * n + k] = Math.Clamp(page.Pixels[k] + 0.01f * c, 0f, 1f);
                }
            }

            yield return new ImageData(colour, 3, page.Height, page.Width);
        }

        yield return new ImageData(new float[50 * 40], 1, 40, 50);                       // blank
    }

    private static string Segments()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var page in Pages())
        {
            foreach (var settings in new[] { new SegmentationSettings(), new SegmentationSettings { MaxSkew = 0 }, new SegmentationSettings { MaxSkew = 1, Padding = 0.5 } })
            {
                var (lines, skew) = LineSegmenter.Find(page, settings);
                Add(hash, $"{skew:R}|{lines.Count}");
                foreach (var l in lines)
                {
                    Add(hash, $"{l.Index},{l.X},{l.Y},{l.Width},{l.Height},{l.Image.Channels},{l.Image.Height},{l.Image.Width}");
                    hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(l.Image.Pixels.AsSpan()));
                }
            }
        }

        return Convert.ToHexString(hash.GetHashAndReset())[..16];
    }

    private static string Alignments()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var r = new Random(7);
        for (int t = 0; t < 40; t++)
        {
            var lines = Texts(r, 2 + t % 6).Select(s => s + (t % 5 == 0 ? " 𝒜b12" : "")).ToArray();
            string page = string.Join(" ", lines.Select(l => l + " " + Letters[r.Next(Letters.Length)]));
            var drafts = lines.Select(l => Mutate(l, r)).ToList();
            if (t % 7 == 0)
            {
                drafts.Insert(1, "zzzz zzzz");
            }

            foreach (var a in TextAlignment.Align(drafts, page, t % 3 == 0 ? 0.3 : 0.5))
            {
                Add(hash, a is { } v ? $"{v.Text}|{v.Match:R}" : "null");
            }

            for (int from = 0; from < page.Length; from += 7)
            {
                if (!char.IsLowSurrogate(page[from])) { Add(hash, TextAlignment.Best(drafts[0], page, from)?.ToString() ?? "null"); }
            }
        }

        Add(hash, TextAlignment.Best("", "abc", 0)?.ToString() ?? "null");
        Add(hash, TextAlignment.Best("a", "abc", 3)?.ToString() ?? "null");
        Add(hash, TextAlignment.Best("abc", "x", 0)?.ToString() ?? "null");
        return Convert.ToHexString(hash.GetHashAndReset())[..16];
    }

    private static string Mutate(string s, Random r)
    {
        var b = new StringBuilder(s);
        for (int k = 0; k < 2 && b.Length > 1; k++)
        {
            int at = r.Next(b.Length);
            if (char.IsSurrogate(b[at]))
            {
                continue;
            }

            switch (r.Next(3))
            {
                case 0: b.Remove(at, 1); break;
                case 1: b.Insert(at, Letters[r.Next(Letters.Length)]); break;
                default: b[at] = Letters[r.Next(Letters.Length)][0]; break;
            }
        }

        return b.ToString();
    }

    private static readonly string[] Samples =
    [
        "", "abc", "بب 123 س abc de م ١٢", "𝒜𝒝 x 12.5-3 بت", "a, b: c/d ب", " 12 ", "ب12ب", "١٢٣ ب ٤٥٦", "é ت 9", "x 😀 y ب",
    ];

    private static string Orders()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var r = new Random(3);
        foreach (string s in Samples.Concat(Enumerable.Range(0, 50).Select(_ => Synthetic.Text(r, 1, 20) + " " + r.Next(0, 99999) + " ab")))
        {
            Add(hash, ReadingOrder.ToColumns(s) + "|" + ReadingOrder.FromColumns(s));
        }

        return Convert.ToHexString(hash.GetHashAndReset())[..16];
    }

    private static string Encodes()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (bool rtl in new[] { true, false })
        {
            var settings = new RecognizerSettings { Alphabet = [" ", "1", "2", "a", "b", "𝒜", .. Letters], Height = 16, Channels = [4, 4, 4], Hidden = 4, Layers = 1, RightToLeft = rtl };
            using var recognizer = Recognizer.Create(settings, Device.Cpu, 1);
            foreach (string s in Samples)
            {
                var (labels, unknown) = recognizer.Encode(s);
                Add(hash, string.Join(",", labels) + "|" + string.Join(",", unknown) + "|" + recognizer.Text(labels));
            }
        }

        return Convert.ToHexString(hash.GetHashAndReset())[..16];
    }

    private static string Prepares()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var r = new Random(5);
        for (int i = 0; i < 20; i++)
        {
            var line = Synthetic.LineImage(Synthetic.Text(r, 2, 9), r);
            if (i % 3 == 1)
            {
                line = new ImageData([.. line.Pixels.Select(v => 1f - v)], 1, line.Height, line.Width);
            }
            else if (i % 3 == 2)
            {
                line = new ImageData([.. line.Pixels, .. line.Pixels.Select(v => v * 0.5f), .. line.Pixels], 3, line.Height, line.Width);
            }

            foreach (bool rtl in new[] { true, false })
            {
                var p = PreparedLine.From(line, 16 + 4 * (i % 3), rtl);
                Add(hash, $"{p.Width}|{p.Steps}");
                hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(p.Pixels.AsSpan()));
            }
        }

        var flat = PreparedLine.From(new ImageData(Enumerable.Repeat(0.7f, 30).ToArray(), 1, 5, 6), 8, true);
        hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(flat.Pixels.AsSpan()));
        return Convert.ToHexString(hash.GetHashAndReset())[..16];
    }

    private static void Add(IncrementalHash hash, string s) => hash.AppendData(Encoding.UTF8.GetBytes(s + "\n"));
}
