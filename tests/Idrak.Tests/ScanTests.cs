// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction.Data;

// Plan 18, phase 4: scan degradations as image transforms, mixed by scan profiles drawn from a seed.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ScanGroup =
    [
        ("scans: rotate turns about the centre (90 degrees exactly, the right side to the top) and fills what it uncovers with the paper; perspective keeps the size and fills its corners", ScanGeometry),
        ("scans: noise adds grain of the asked strength, the same in every channel; speckle adds specks in proportion to their count; the same seed gives the same pixels, another seed others", ScanGrain),
        ("scans: fade lightens ink toward the paper by its share and leaves the paper; uneven varies it over the page; tint multiplies toward its colour; stain and bleed only darken; lowres keeps the size and softens", ScanTone),
        ("scans: every scan transform refuses values out of its range and leaves its input as it was", ScanRefusals),
        ("scans: a profile draws each step at its chance, its values within their ranges (whole where the transform takes whole numbers), the same steps for the same seed, and reads back from its own text; bad profiles are refused", ScanProfileSampling),
        ("scans: the built-in profiles (clean, office, old, fax) degrade a drawn page and keep its size and its ink; an app's profile replaces the library's of its name only", ScanProfilesOnPages),
    ];

    private static ImageData Bytes8(int channels, int height, int width, Func<int, int, int, byte> value)
    {
        var planes = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            planes[c] = new byte[height * width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    planes[c][y * width + x] = value(c, y, x);
                }
            }
        }

        return new ImageData(planes, height, width, [.. Enumerable.Range(0, 256).Select(v => v / 255f)]);
    }

    private static ImageData Scan(ImageData image, string pipeline) => ImageTransformPipeline.Parse(pipeline).Apply(image);

    private static bool SamePixels(ImageData a, ImageData b) =>
        a.Channels == b.Channels && a.Height == b.Height && a.Width == b.Width && Planes(a).Zip(Planes(b)).All(p => p.First.SequenceEqual(p.Second));

    private static void ScanGeometry(Device device)
    {
        int n = 41;
        var random = new Random(3);
        var noise = Bytes8(1, n, n, (_, _, _) => (byte)random.Next(256));
        var source = Planes(noise)[0];
        var turned = Planes(Scan(noise, "rotate=90"))[0];
        int worst = 0;
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++)
            {
                worst = Math.Max(worst, Math.Abs(turned[y * n + x] - source[x * n + (n - 1 - y)]));
            }
        }

        Check(worst <= 1, $"rotate=90: each pixel from the one a quarter turn away, within {worst} of a level");

        // A white page with a black bar at its right edge: a small turn uncovers corners in the paper's grey, not black.
        var page = Bytes8(3, 200, 300, (c, y, x) => x >= 280 ? (byte)0 : (byte)(230 + c * 5));
        var small = Scan(page, "rotate=3");
        var planes = Planes(small);
        Check(planes[0][0] is >= 228 and <= 232 && planes[2][0] is >= 238 and <= 242, $"rotate=3: the uncovered corner is the paper ({planes[0][0]}, {planes[1][0]}, {planes[2][0]})");
        Check(Planes(Scan(page, "rotate=3,fill=0"))[0][0] == 0, "fill=0: the corner black");
        Check(SamePixels(Scan(page, "rotate=0"), page), "rotate=0 changes nothing");

        var warped = Scan(page, "perspective=0.05,seed=4");
        Check(warped.Height == page.Height && warped.Width == page.Width, "perspective keeps the size");
        Check(Planes(warped)[0][0] is >= 228 and <= 232, "and fills its uncovered corners with the paper");
        Check(!SamePixels(warped, Scan(page, "perspective=0.05,seed=5")) && SamePixels(warped, Scan(page, "perspective=0.05,seed=4")), "its corners from its seed");
    }

    private static void ScanGrain(Device device)
    {
        var grey = Bytes8(3, 300, 400, (_, _, _) => 128);
        var noisy = Scan(grey, "noise=6,seed=11");
        var planes = Planes(noisy);
        double mean = planes[0].Average(v => (double)v), sd = Math.Sqrt(planes[0].Average(v => (v - mean) * (v - mean)));
        Check(Math.Abs(mean - 128) < 0.3 && Math.Abs(sd - 6) < 0.3, $"noise=6: mean {mean:F2}, deviation {sd:F2}");
        Check(planes[0].SequenceEqual(planes[1]) && planes[1].SequenceEqual(planes[2]), "the same grain in every channel");
        Check(SamePixels(noisy, Scan(grey, "noise=6,seed=11")) && !SamePixels(noisy, Scan(grey, "noise=6,seed=12")), "the same seed the same grain, another seed another");

        long pinned = Planes(Scan(Bytes8(1, 48, 64, (_, _, _) => 128), "noise=6,seed=11"))[0].Sum(v => (long)v);
        Check(pinned == 392807, $"noise=6,seed=11 on 64x48 of grey 128 sums to {pinned}: the same in every process and on every machine (392807)");

        var white = Bytes8(1, 1000, 1000, (_, _, _) => 250);
        int Dark(ImageData image) => Planes(image)[0].Count(v => v < 200);
        int few = Dark(Scan(white, "speckle=100,seed=2")), many = Dark(Scan(white, "speckle=400,seed=2"));
        Check(few > 0 && many > 2.5 * few && many < 6 * few, $"speckle: {few} dark pixels for 100 specks a megapixel, {many} for 400");
        Check(SamePixels(Scan(white, "speckle=0"), white), "speckle=0 changes nothing");
    }

    private static void ScanTone(Device device)
    {
        // Black text (columns 10-19) on paper 240.
        var page = Bytes8(1, 100, 200, (_, _, x) => x is >= 10 and < 20 ? (byte)0 : (byte)240);
        var faded = Planes(Scan(page, "fade=0.5"))[0];
        Check(faded[10] is >= 119 and <= 121 && faded[50] == 240, $"fade=0.5: ink 0 to {faded[10]} (half way to the paper), paper stays {faded[50]}");
        var uneven = Planes(Scan(page, "fade=0.5,uneven=1,seed=3"))[0];
        var inks = Enumerable.Range(0, 100).Select(y => uneven[y * 200 + 15]).ToArray();
        Check(inks.Max() - inks.Min() > 10 && inks.All(v => v <= 121), $"uneven=1: the ink from {inks.Min()} to {inks.Max()}, never fainter than the even fade");

        var white = Bytes8(3, 10, 10, (_, _, _) => 255);
        var tinted = Planes(Scan(white, "tint=1,color=f2e6c4"));
        Check(tinted[0][0] == 0xf2 && tinted[1][0] == 0xe6 && tinted[2][0] == 0xc4, "tint=1: white becomes the colour");
        var half = Planes(Scan(white, "tint=0.5,color=000000"));
        Check(half[0][0] is 127 or 128, $"tint=0.5 toward black: {half[0][0]}");

        var paper = Bytes8(3, 400, 400, (c, _, x) => x < 200 && x % 20 < 4 ? (byte)20 : (byte)235);
        var before = Planes(paper);
        foreach (string step in new[] { "stain=3,seed=8,opacity=0.5", "bleed=0.4" })
        {
            var after = Planes(Scan(paper, step));
            bool darker = Enumerable.Range(0, 3).All(c => after[c].Zip(before[c]).All(p => p.First <= p.Second + 1));
            int changed = after[0].Zip(before[0]).Count(p => p.First < p.Second - 2);
            Check(darker && changed > 100, $"{step}: only darker, {changed} pixels darkened");
        }

        var bled = Planes(Scan(paper, "bleed=0.4"))[0];
        Check(bled[398] < 230 && bled[247] >= 233, "bleed: the left half's ink shows through on the right, mirrored");

        var checker = Bytes8(1, 64, 64, (_, y, x) => (x + y) % 2 == 0 ? (byte)0 : (byte)255);
        var soft = Scan(checker, "lowres=0.5");
        var values = Planes(soft)[0];
        Check(soft.Width == 64 && soft.Height == 64 && values.Max() - values.Min() < 100, $"lowres=0.5: the size kept, a one-pixel checker softened to {values.Min()}-{values.Max()}");
        Check(SamePixels(Scan(checker, "lowres=1"), checker), "lowres=1 changes nothing");
    }

    private static void ScanRefusals(Device device)
    {
        foreach (string bad in new[]
        {
            "rotate=200", "rotate=3,fill=300", "perspective=0.5", "noise=-1", "noise=3,seed=x", "speckle=200000", "speckle=10,size=50",
            "fade=1", "fade=0.3,uneven=2", "tint=2", "tint=0.5,color=zz0000", "stain=60", "stain=2,opacity=3", "lowres=0.05", "lowres=2",
            "bleed=1.5", "bleed=0.2,spread=100",
        })
        {
            Throws<ArgumentException>(() => ImageTransformPipeline.Parse(bad), bad);
        }

        var page = Bytes8(3, 50, 60, (c, y, x) => (byte)((x * 7 + y * 3 + c * 50) % 256));
        var copy = Planes(page);
        foreach (string step in new[] { "rotate=2", "perspective=0.02,seed=1", "noise=5,seed=1", "speckle=5000,seed=1", "fade=0.5", "tint=0.5", "stain=2,seed=1", "lowres=0.5", "bleed=0.3" })
        {
            var result = Scan(page, step);
            Check(Planes(page).Zip(copy).All(p => p.First.SequenceEqual(p.Second)) && result.Width == 60 && result.Height == 50, $"{step}: the input as it was, the size kept");
        }

        foreach (string name in new[] { "rotate", "perspective", "noise", "speckle", "fade", "tint", "stain", "lowres", "bleed" })
        {
            var transform = ImageTransforms.Get(name);
            Check(transform.Parameters.Count > 0 && transform.Parameters[0].Key == "" && transform.Label != name, $"{name} describes its inputs ({transform.Label})");
        }
    }

    private static void ScanProfileSampling(Device device)
    {
        var profile = new ScanProfile("test", "rotate=-2.5..2.5@0.5,jpeg=40..90@0.5,stain=1..4@0.5,opacity=0.1..0.3,noise=3");
        Check(new ScanProfile("again", profile.Text).Text == profile.Text, $"the text reads back: {profile.Text}");
        Check(profile.Sample(5).ToString() == profile.Sample(5).ToString(), "the same seed, the same steps");
        var samples = Enumerable.Range(0, 400).Select(profile.Sample).ToList();
        Check(samples.Select(s => s.ToString()).Distinct().Count() > 300, "other seeds, other steps");
        foreach (string name in new[] { "rotate", "jpeg", "stain" })
        {
            int taken = samples.Count(s => s.Contains(name));
            Check(taken is > 150 and < 250, $"{name} at chance 0.5: taken {taken} times in 400");
        }

        Check(samples.All(s => s.Contains("noise")), "a step with no chance is always taken");
        var rotations = samples.SelectMany(s => s.Steps).Where(s => s.Name == "rotate").Select(s => s.Number()).ToList();
        Check(rotations.All(r => r is >= -2.5 and <= 2.5) && rotations.Any(r => r != Math.Round(r)) && rotations.Max() - rotations.Min() > 4,
            $"rotations within -2.5..2.5, fractional ({rotations.Min():F2} to {rotations.Max():F2})");
        var qualities = samples.SelectMany(s => s.Steps).Where(s => s.Name == "jpeg").Select(s => s.Value!).ToList();
        Check(qualities.All(q => int.TryParse(q, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v is >= 40 and <= 90), "JPEG qualities whole, within 40..90");
        var stains = samples.SelectMany(s => s.Steps).Where(s => s.Name == "stain").ToList();
        Check(stains.All(s => s.Option("seed") is not null && double.Parse(s.Option("opacity")!, CultureInfo.InvariantCulture) is >= 0.1 and <= 0.3),
            "a seeded step gets a seed, an option its value from its range");
        Check(samples.All(s => ImageTransformPipeline.Parse(s.ToString()).Equals(s)), "every drawn pipeline reads back as itself");

        // A step's chance does not move the others: with the jpeg step never taken, the rotations stay.
        var without = new ScanProfile("test", "rotate=-2.5..2.5@0.5,jpeg=40..90@0,stain=1..4@0.5,opacity=0.1..0.3,noise=3");
        Check(Enumerable.Range(0, 50).All(s => profile.Sample(s).Steps.FirstOrDefault(x => x.Name == "rotate")?.ToString() == without.Sample(s).Steps.FirstOrDefault(x => x.Name == "rotate")?.ToString()),
            "a step's chance leaves the other steps' draws alone");

        foreach (string bad in new[] { "nope=1", "rotate=1@2", "rotate=5..1", "rotate=-200..0", "jpeg=50,opacity=1", "stain=2,opacity=0.2@0.5", "=3" })
        {
            Throws<ArgumentException>(() => _ = new ScanProfile("bad", bad), bad);
        }

        Check(ScanProfiles.Resolve("office").Name == "office" && ScanProfiles.Resolve("rotate=-1.0..1.0").Name == "custom", "a name or a profile's text");
        Throws<NotSupportedException>(() => ScanProfiles.Resolve("nope"), "an unknown profile name");
    }

    private static void ScanProfilesOnPages(Device device)
    {
        Check(new[] { "clean", "office", "old", "fax" }.All(ScanProfiles.Names.Contains), $"profiles {string.Join(", ", ScanProfiles.Names)}");
        var page = PageLayouts.Get("letter").Render(ArabicLetter, TestPageStyle(), 3).Image;
        int ink = Planes(page)[0].Count(v => v < 100);
        var watch = Stopwatch.StartNew();
        int count = 0;
        foreach (string name in ScanProfiles.Names)
        {
            for (int seed = 1; seed <= 3; seed++)
            {
                var steps = ScanProfiles.Get(name).Sample(seed);
                var scanned = steps.Apply(page);
                count++;
                var plane = Planes(scanned)[0];
                int paper = plane.Order().ElementAt(plane.Length * 9 / 10);
                int left = plane.Count(v => v < paper - 40);
                Check(scanned.Width == page.Width && scanned.Height == page.Height && left > ink / 6, $"{name}, seed {seed} ({steps}): {left} dark pixels of {ink}");
            }
        }

        Console.WriteLine($"    {count} scans, {watch.Elapsed.TotalMilliseconds / count:F0} ms a page");

        ScanProfiles.Register(new ScanProfile("office", "rotate=1.0"));
        try
        {
            Check(ScanProfiles.Get("office").Sample(1).ToString() == "rotate=1" && ScanProfiles.Get("old").Sample(1).Steps.Count > 0, "the app's office, the library's old");
        }
        finally
        {
            ScanProfiles.Unregister("office");
        }

        Check(ScanProfiles.Get("office").Summary.StartsWith("an office scanner", StringComparison.Ordinal), "unregistered: the library's office again");
    }
}
