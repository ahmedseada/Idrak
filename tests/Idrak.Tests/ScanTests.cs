// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction.Data;
using Idrak.Nlp;

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
        ("scans: the paper effects (wrinkle, fold, smudge, blot, strike, underline, cross, scribble, highlight, tear, holes, scratch, taped, shadow, streak, light, curl; water, burn, foxing, staple, dogear, clip, residue, wormholes, sunfade, piece, whiteout, sticky, burnhole, cutoff; photocopy, carbon, lowtoner, banding, ghost, feather, misregister; desk, glare, vignette, motion, moire, handshadow, fingerprint; lined, grid, guilloche, recycled) are seeded, keep the size and the input, refuse bad values; what removes paper (tears, holes, pieces, burns, cut-off, dog-ears, sticky notes, staples, clips) stays out of the text and correction fluid leaves the ink, pen marks and highlighter only darken, scratches only lighten; a page ripped and taped shows the background along its tear, tape over it, and keeps its ink", PaperEffects),
        ("scans: an app's own effect (an IImageTransform registered by name) is used in a profile's text, sampled with its seed, and drawn on synthetic pages, as the library's are", AppEffect),
        ("scans: dropout breaks ink in patches (only ink lightened, the paper left, seeded); the default mix shuffles every look, turns about one page in six, and has straight clear pages, faint ink, broken strokes, pen marks and handled or damaged paper; looks joined with + take each one's effects on a page, a turn or the JPEG once and last", ScanMix),
        ("scans: the built-in profiles (clean, office, old, pale, worn, fax, handled, marked, damaged, scanner, aged, copied, photo, wrecked, mixed) degrade a drawn page and keep its size and its ink; an app's profile replaces the library's of its name only", ScanProfilesOnPages),
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
            "bleed=1.5", "bleed=0.2,spread=100", "dropout=0.9", "dropout=0.1,strength=2", "dropout=0.1,size=100",
        })
        {
            Throws<ArgumentException>(() => ImageTransformPipeline.Parse(bad), bad);
        }

        var page = Bytes8(3, 50, 60, (c, y, x) => (byte)((x * 7 + y * 3 + c * 50) % 256));
        var copy = Planes(page);
        foreach (string step in new[] { "rotate=2", "dropout=0.2,seed=1", "perspective=0.02,seed=1", "noise=5,seed=1", "speckle=5000,seed=1", "fade=0.5", "tint=0.5", "stain=2,seed=1", "lowres=0.5", "bleed=0.3" })
        {
            var result = Scan(page, step);
            Check(Planes(page).Zip(copy).All(p => p.First.SequenceEqual(p.Second)) && result.Width == 60 && result.Height == 50, $"{step}: the input as it was, the size kept");
        }

        foreach (string name in new[] { "rotate", "perspective", "noise", "speckle", "dropout", "fade", "tint", "stain", "lowres", "bleed" })
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
        Check(new[] { "clean", "office", "old", "pale", "worn", "fax" }.All(ScanProfiles.Names.Contains), $"profiles {string.Join(", ", ScanProfiles.Names)}");
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

    private static void ScanMix(Device device)
    {
        // Dropout: black strokes (columns 10-29) on paper 240.
        var page = Bytes8(1, 200, 300, (_, _, x) => x is >= 10 and < 30 ? (byte)0 : (byte)240);
        var worn = Planes(Scan(page, "dropout=0.3,size=6,seed=4"))[0];
        var before = Planes(page)[0];
        int lightened = 0;
        bool paperKept = true, onlyLighter = true;
        for (int i = 0; i < before.Length; i++)
        {
            paperKept &= before[i] != 240 || worn[i] == 240;
            onlyLighter &= worn[i] >= before[i];
            lightened += worn[i] > before[i] ? 1 : 0;
        }

        int ink = before.Count(v => v == 0);
        Check(paperKept && onlyLighter && lightened > ink / 10 && lightened < ink, $"dropout: {lightened} of {ink} ink pixels lightened, the paper untouched");
        Check(SamePixels(Scan(page, "dropout=0.3,size=6,seed=4"), Scan(page, "dropout=0.3,size=6,seed=4")) && !SamePixels(Scan(page, "dropout=0.3,size=6,seed=5"), Scan(page, "dropout=0.3,size=6,seed=4")),
            "seeded");
        Check(SamePixels(Scan(page, "dropout=0"), page), "dropout=0 changes nothing");

        // The default mix: one look a page, as SyntheticPages draws it.
        var looks = new SyntheticPageOptions { Style = new PageStyle { Fonts = [SubsetFace()] } }.Scans;
        Check(looks.SequenceEqual(["none", "clean", "office", "old", "pale", "worn", "fax", "handled", "marked", "damaged", "scanner", "aged", "copied", "photo", "wrecked", "mixed", "mixed", "mixed"]), string.Join(",", looks));
        int pages = 0, turned = 0, faint = 0, broken = 0, untouched = 0, marked = 0, damaged = 0;
        foreach (string look in looks)
        {
            for (int seed = 0; seed < 300; seed++)
            {
                var steps = SyntheticPages.Profile(look)?.Sample(seed) ?? ImageTransformPipeline.Empty;
                pages++;
                turned += steps.Contains("rotate") || steps.Contains("perspective") ? 1 : 0;
                faint += steps.Steps.Any(x => x.Name == "fade" && x.Number() >= 0.45) ? 1 : 0;
                broken += steps.Contains("dropout") || steps.Contains("binarize") ? 1 : 0;
                marked += new[] { "strike", "underline", "cross", "scribble", "highlight" }.Any(steps.Contains) ? 1 : 0;
                damaged += new[] { "tear", "holes", "scratch", "wrinkle", "fold", "smudge" }.Any(steps.Contains) ? 1 : 0;
                untouched += steps.IsEmpty ? 1 : 0;
            }
        }

        // Looks joined with '+': both looks' effects on a page, the turn and the JPEG once, and the JPEG last.
        var joined = ScanProfiles.Resolve("damaged + marked");
        int both = 0;
        for (int seed = 0; seed < 300; seed++)
        {
            var steps = joined.Sample(seed);
            var names = steps.Steps.Select(x => x.Name).ToList();
            both += new[] { "tear", "holes", "scratch", "taped" }.Any(steps.Contains) && new[] { "strike", "underline", "cross", "scribble", "highlight" }.Any(steps.Contains) ? 1 : 0;
            Check(names.Count(n => n == "rotate") <= 1 && names.Count(n => n == "jpeg") <= 1 && (!names.Contains("jpeg") || names[^1] == "jpeg"), $"seed {seed}: {steps}");
            Check(joined.Sample(seed).Equals(steps), $"seed {seed}: seeded");
        }

        Check(joined.Name == "damaged+marked" && both > 100, $"{both} of 300 pages damaged and marked");
        Check(SyntheticPages.Profile("handled+scanner+marked") is { Name: "handled+scanner+marked" }, "three looks");
        Throws<NotSupportedException>(() => ScanProfiles.Resolve("damaged+nope"), "an unknown look");
        Throws<ArgumentException>(() => ScanProfiles.Resolve("damaged+"), "a look missing");

        double Share(int n) => (double)n / pages;
        Check(Share(turned) is > 0.10 and < 0.25, $"pages turned: {Share(turned):P0}");
        Check(Share(faint) > 0.08 && Share(broken) > 0.12 && Share(untouched) > 0.07 && Share(marked) > 0.1 && Share(damaged) > 0.2,
            $"faint ink {Share(faint):P0}, broken strokes {Share(broken):P0}, as drawn {Share(untouched):P0}, pen marks {Share(marked):P0}, handled or damaged paper {Share(damaged):P0}");
    }

    private static void PaperEffects(Device device)
    {
        var page = PageLayouts.Get("letter").Render(ArabicLetter, TestPageStyle() with { Rule = 0, Emblem = 0, Stamp = 0 }, 3).Image;
        var before = Planes(page);
        string[] effects =
        [
            "wrinkle=0.7", "fold=2", "smudge=3", "blot=3", "strike=2", "underline=2", "cross=1", "scribble=3", "highlight=2", "tear=0.04", "holes=3",
            "scratch=8", "taped=2", "taped=1,tape=clear", "water=2", "burn=0.04", "foxing=150", "staple=1", "dogear=0.08", "clip=1", "residue=2", "wormholes=8",
            "sunfade=0.5", "piece=2", "whiteout=3", "sticky=1", "burnhole=2", "cutoff=0.04", "photocopy=2", "carbon=0.8", "lowtoner=0.6", "banding=0.3",
            "ghost=0.3", "feather=0.8", "misregister=3", "desk=0.1", "glare=0.5", "vignette=0.5", "motion=4", "moire=0.15", "handshadow=0.4", "fingerprint=2",
            "lined=0.8", "grid=0.8", "guilloche=0.8", "recycled=0.8", "shadow=0.2", "streak=4", "light=0.4", "curl=0.7",
        ];
        foreach (string effect in effects)
        {
            string name = effect.Split('=')[0];
            var transform = ImageTransforms.Get(name);
            Check(transform.Parameters.Count > 1 && transform.Label != name, $"{name} describes its inputs ({transform.Label})");
            var once = Scan(page, effect + ",seed=5");
            Check(once.Width == page.Width && once.Height == page.Height && !SamePixels(once, page), $"{effect}: the size kept, the page changed");
            Check(SamePixels(once, Scan(page, effect + ",seed=5")), $"{effect}: the same seed, the same page");
            Check(Planes(page).Zip(before).All(p => p.First.SequenceEqual(p.Second)), $"{effect}: the input as it was");
        }

        // Tears and holes stay in the margins: the text's area (the page's middle) is as it was.
        var blocks = PageLayouts.Get("letter").Render(ArabicLetter, TestPageStyle() with { Rule = 0, Emblem = 0, Stamp = 0 }, 3).Blocks.Where(b => b.Text.Length > 0).ToList();
        int left = (int)blocks.Min(b => b.X), right = (int)Math.Ceiling(blocks.Max(b => b.X + b.Width)), top = (int)blocks.Min(b => b.Y), bottom = (int)Math.Ceiling(blocks.Max(b => b.Y + b.Height));
        string[] margins = ["holes=4", "burn=0.05", "piece=3", "wormholes=20", "burnhole=4", "cutoff=0.05", "sticky=2", "staple=2", "clip=2", "dogear=0.15"];
        foreach (string effect in Enumerable.Range(1, 10).Select(k => $"tear=0.05,seed={k}").Concat(margins.SelectMany(m => Enumerable.Range(1, 4).Select(k => $"{m},seed={k}"))))
        {
            var after = Planes(Scan(page, effect));
            bool kept = true;
            for (int y = top; y < bottom; y++)
            {
                for (int x = left; x < right; x++)
                {
                    kept &= after[0][y * page.Width + x] == before[0][y * page.Width + x];
                }
            }

            Check(kept, $"{effect}: the text's area untouched");
        }

        // Ripped and taped: on a blank page the background (black here) shows along the whole tear, and tape lies over it;
        // on the letter, its ink stays (a few pixels lost along the tear at most).
        var blank = Bytes8(3, 400, 300, (_, _, _) => 240);
        foreach (int seed in new[] { 1, 2, 3, 4 })
        {
            var ripped = Planes(Scan(blank, $"taped=1,fill=0,gap=3,shift=2,seed={seed}"))[0];
            int gap = ripped.Count(v => v < 60), taped = ripped.Count(v => v is >= 60 and not 240);
            Check(gap >= 300 * 0.8 && gap < 300 * 6 && taped > 200, $"taped, seed {seed}: {gap} background pixels along the tear, {taped} under the tape or on its torn edge");
        }

        int inkBefore = before[0].Count(v => v < 100);
        foreach (int seed in new[] { 1, 2, 3 })
        {
            int inkAfter = Planes(Scan(page, $"taped=2,fill=255,seed={seed}"))[0].Count(v => v < 100);
            Check(inkAfter > inkBefore * 0.9, $"taped=2, seed {seed}: {inkAfter} of {inkBefore} ink pixels left");
        }

        // Correction fluid goes on blank paper only: every ink pixel as it was.
        foreach (int seed in new[] { 1, 2, 3, 4 })
        {
            var after = Planes(Scan(page, $"whiteout=8,seed={seed}"));
            Check(Enumerable.Range(0, before[0].Length).All(i => before[0][i] >= 200 || after[0][i] == before[0][i]) && !SamePixels(Scan(page, $"whiteout=8,seed={seed}"), page),
                $"whiteout, seed {seed}: the ink untouched");
        }

        // Pen and highlighter only darken; scratches that lighten only lighten.
        foreach (string effect in new[] { "strike=3,seed=1", "underline=3,seed=1", "cross=1,seed=1", "scribble=4,seed=1", "highlight=3,seed=1" })
        {
            var after = Planes(Scan(page, effect));
            Check(Enumerable.Range(0, 3).All(c => after[c].Zip(before[c]).All(p => p.First <= p.Second)), $"{effect}: only darker");
        }

        foreach (string bad in new[] { "wrinkle=2", "fold=1.5", "fold=9", "strike=2,color=purple", "tear=0.2", "holes=7", "shadow=0.5", "light=1", "curl=2", "smudge=2,length=900", "blot=2,size=99", "taped=4", "taped=1.5", "taped=1,gap=40", "taped=1,shift=-1", "taped=1,tape=pink",
            "water=9", "burn=0.2", "foxing=-1", "dogear=0.5", "photocopy=4", "photocopy=1.5", "carbon=2", "glare=0.9", "motion=20", "desk=0.5", "lined=2", "misregister=9" })
        {
            Throws<ArgumentException>(() => ImageTransformPipeline.Parse(bad), bad);
        }
    }

    private static void AppEffect(Device device)
    {
        ImageTransforms.Register(new CoffeeRing());
        try
        {
            var profile = new ScanProfile("cafe", "coffee=1..3@1,seed=0..1000,rotate=-0.5..0.5@0", "rings from a cup");
            var steps = profile.Sample(4);
            Check(steps.Steps[0].Name == "coffee" && steps.Steps[0].Integer() is >= 1 and <= 3 && steps.Steps[0].Option("seed") is not null, steps.ToString());
            ScanProfiles.Register(profile);
            var options = new SyntheticPageOptions { Style = new PageStyle { Fonts = [SubsetFace()], Width = 400, Height = 300, BodySize = (10, 12) }, Scans = ["cafe", "coffee=2@1"] };
            var plain = SyntheticPages.Make("نص الصفحة", options with { Scans = ["none"] });
            foreach (int variant in new[] { 0, 1, 2, 3 })
            {
                var page = SyntheticPages.Make("نص الصفحة", options, variant: variant);
                Check(page.Scan.StartsWith("coffee=", StringComparison.Ordinal) && page.Profile is "cafe" or "coffee=2@1" && page.Text == plain.Text,
                    $"variant {variant}: {page.Profile}, {page.Scan}");
                Check(Planes(page.Image)[0].Count(v => v < 200) > Planes(plain.Image)[0].Count(v => v < 200), "the rings drawn");
            }
        }
        finally
        {
            ScanProfiles.Unregister("cafe");
            ImageTransforms.Unregister("coffee");
        }

        Throws<ArgumentException>(() => _ = new ScanProfile("cafe", "coffee=1"), "unregistered: the profile is refused, naming the transforms there are");
    }

    // An app's effect: N dark rings (a cup's), placed by its seed.
    private sealed class CoffeeRing : IImageTransform
    {
        public string Name => "coffee";

        public string Summary => "coffee=N: N rings from a cup; seed=S";

        public IReadOnlyCollection<string> Keys => ["seed"];

        public void Check(ImageTransformStep step)
        {
            if (step.Integer() is < 0 or > 5)
            {
                throw new ArgumentException("coffee is 0 to 5 rings.");
            }
        }

        public ImageData Apply(ImageData image, ImageTransformStep step)
        {
            image.TryGetPlanes(out var held, out var levels);
            var planes = held.Select(p => p.ToArray()).ToArray();
            var random = new Random(int.Parse(step.Option("seed") ?? "0", System.Globalization.CultureInfo.InvariantCulture));
            for (int k = 0; k < step.Integer(); k++)
            {
                double cx = random.NextDouble() * image.Width, cy = random.NextDouble() * image.Height, r = 20 + random.NextDouble() * 30;
                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        double d = Math.Abs(Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - r);
                        if (d < 2)
                        {
                            foreach (var plane in planes)
                            {
                                plane[y * image.Width + x] = (byte)(plane[y * image.Width + x] * 0.6);
                            }
                        }
                    }
                }
            }

            return new ImageData(planes, image.Height, image.Width, levels.ToArray());
        }
    }
}
