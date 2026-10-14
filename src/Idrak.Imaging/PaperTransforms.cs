// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using static Idrak.Imaging.ScanTransforms;

namespace Idrak.Imaging;

/// <summary>
/// What happens to a paper between the printer and the scanner, as image transforms (plan 18): it is handled
/// (<c>wrinkle</c>, <c>fold</c>, <c>smudge</c>, <c>blot</c>), marked with a pen (<c>strike</c>, <c>underline</c>,
/// <c>cross</c>, <c>highlight</c>, <c>scribble</c>), damaged (<c>tear</c>, <c>holes</c>, <c>scratch</c>, <c>taped</c>)
/// and scanned (<c>shadow</c>, <c>streak</c>, <c>light</c>, <c>curl</c>); PaperEffects.cs adds age, damage, copiers,
/// cameras and paper kinds. Every one takes a <c>seed</c>, and none takes
/// away text a page's answer holds: tears and holes stay in the margins, pen marks are thin or see-through, smudges and
/// blots are small, a page ripped and taped back loses no more than a few pixels along its tear, so a page's text can
/// still be read. Tears and holes measure the page's blank margins (where its first ink is) and stay inside them.
/// </summary>
internal static class PaperTransforms
{
    internal static void RegisterDefaults()
    {
        foreach (var transform in (IImageTransform[])
        [
            Make("wrinkle", "wrinkle=A: crumpled paper: soft creases across the page shading it and shifting the ink a little, A from 0 to 1; seed=N",
                "Wrinkles", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Wrinkle(i, v, Seed(s))),
            Make("fold", "fold=N: N fold creases (a letter folded in thirds or halves): a sharp line, a highlight beside it and a soft shade; seed=N",
                "Folds", "Creases", 0, 6, 1, ["seed"], (i, s, v) => PaperImageOps.Fold(i, (int)v, Seed(s)), whole: true),
            Make("smudge", "smudge=N: N smears of ink dragged over the text by a hand; length=L pixels (30); seed=N",
                "Smudges", "Smudges", 0, 30, 1, ["seed", "length"], (i, s, v) => PaperImageOps.Smudge(i, (int)v, Option(s, "length"), Seed(s)), whole: true),
            Make("blot", "blot=N: N small ink blots and their splatter (up to size=R pixels, 5); seed=N",
                "Ink blots", "Blots", 0, 30, 1, ["seed", "size"], (i, s, v) => PaperImageOps.Blot(i, (int)v, Option(s, "size"), Seed(s)), whole: true),
            Make("strike", "strike=N: N pen lines through text lines (blue, black or red; color=blue|black|red|green|any); seed=N",
                "Struck through", "Lines", 0, 20, 1, ["seed", "color"], (i, s, v) => PaperImageOps.Pen(i, PaperImageOps.PenMark.Strike, (int)v, PenColor(s), Seed(s)), whole: true),
            Make("underline", "underline=N: N pen lines under text lines; color as strike; seed=N",
                "Underlined", "Lines", 0, 20, 1, ["seed", "color"], (i, s, v) => PaperImageOps.Pen(i, PaperImageOps.PenMark.Underline, (int)v, PenColor(s), Seed(s)), whole: true),
            Make("cross", "cross=N: N blocks of one to three text lines crossed out with an X; color as strike; seed=N",
                "Crossed out", "Crosses", 0, 6, 1, ["seed", "color"], (i, s, v) => PaperImageOps.Pen(i, PaperImageOps.PenMark.Cross, (int)v, PenColor(s), Seed(s)), whole: true),
            Make("scribble", "scribble=N: N pen squiggles and notes in the margins, now and then a ring around words; color as strike; seed=N",
                "Scribbles", "Scribbles", 0, 20, 1, ["seed", "color"], (i, s, v) => PaperImageOps.Pen(i, PaperImageOps.PenMark.Scribble, (int)v, PenColor(s), Seed(s)), whole: true),
            Make("highlight", "highlight=N: N text lines (or parts) marked with a see-through highlighter (yellow, green, pink or blue); seed=N",
                "Highlighted", "Lines", 0, 20, 1, ["seed"], (i, s, v) => PaperImageOps.Highlight(i, (int)v, Seed(s)), whole: true),
            Make("tear", "tear=D: torn edges, as deep as D of the page's shorter side at most (0 to 0.05, inside the margins); fill=V the scanner's background (dark or white unless given); seed=N",
                "Torn edges", "Depth", 0, 0.05, 0.005, ["seed", "fill"], (i, s, v) => PaperImageOps.Tear(i, v, Fill(s), Seed(s))),
            Make("holes", "holes=N: N punch holes in the side margin; fill=V the background seen through them; seed=N",
                "Punch holes", "Holes", 0, 4, 1, ["seed", "fill"], (i, s, v) => PaperImageOps.Holes(i, (int)v, Fill(s), Seed(s)), whole: true),
            Make("scratch", "scratch=N: N thin scratches, light or dark, a long way across the page; seed=N",
                "Scratches", "Scratches", 0, 40, 1, ["seed"], (i, s, v) => PaperImageOps.Scratch(i, (int)v, Seed(s)), whole: true),
            Make("taped", "taped=N: the page ripped right across N times (1 to 3) and taped back together: a jagged tear through the text, its pieces up to gap=G pixels apart (3) and shifted up to shift=S pixels along it (3), the background showing through, strips of tape over it, tape=clear|yellowed|matte|any (any: clear the most, transparent with a sheen and visible edges); letters on the tear are split but readable; fill=V the background; seed=N",
                "Ripped and taped", "Tears", 0, 3, 1, ["seed", "fill", "gap", "shift", "tape"],
                (i, s, v) => PaperImageOps.Taped(i, (int)v, Option(s, "gap"), Option(s, "shift"), TapeKind(s), Fill(s), Seed(s)), whole: true),
            Make("water", "water=N: N water marks: a wavy tide line, the paper inside stained and its ink run a little (still readable); seed=N",
                "Water marks", "Marks", 0, 4, 1, ["seed"], (i, s, v) => PaperImageOps.Water(i, (int)v, Seed(s)), whole: true),
            Make("burn", "burn=D: scorched edges burnt away up to D of the page's shorter side (0 to 0.05), charred and browned, inside the margins; fill=V the background; seed=N",
                "Burnt edges", "Depth", 0, 0.05, 0.005, ["seed", "fill"], (i, s, v) => PaperImageOps.Burn(i, v, Fill(s), Seed(s))),
            Make("foxing", "foxing=N: N rust-brown spots of old paper (0 to 600), alone or in clusters; seed=N",
                "Foxing", "Spots", 0, 600, 1, ["seed"], (i, s, v) => PaperImageOps.Foxing(i, (int)v, Seed(s)), whole: true),
            Make("staple", "staple=N: N staples in a top corner's margin (0 to 2), or the holes and rust a removed one leaves; fill=V; seed=N",
                "Staples", "Staples", 0, 2, 1, ["seed", "fill"], (i, s, v) => PaperImageOps.Staple(i, (int)v, Fill(s), Seed(s)), whole: true),
            Make("dogear", "dogear=S: a corner folded over, S of the page's shorter side (0 to 0.15, as the margins allow); fill=V; seed=N",
                "Dog-ear", "Size", 0, 0.15, 0.01, ["seed", "fill"], (i, s, v) => PaperImageOps.DogEar(i, v, Fill(s), Seed(s))),
            Make("clip", "clip=N: N paper clips over the top edge (0 to 2), or the dents a removed one leaves; in the top margin; seed=N",
                "Paper clips", "Clips", 0, 2, 1, ["seed"], (i, s, v) => PaperImageOps.Clip(i, (int)v, Seed(s)), whole: true),
            Make("residue", "residue=N: N patches of old tape's yellowed glue where tape was pulled off (0 to 6), see-through; seed=N",
                "Tape residue", "Patches", 0, 6, 1, ["seed"], (i, s, v) => PaperImageOps.Residue(i, (int)v, Seed(s)), whole: true),
            Make("wormholes", "wormholes=N: N small ragged holes (0 to 30), some in runs, in the margins; fill=V; seed=N",
                "Worm holes", "Holes", 0, 30, 1, ["seed", "fill"], (i, s, v) => PaperImageOps.Wormholes(i, (int)v, Fill(s), Seed(s)), whole: true),
            Make("sunfade", "sunfade=A: a band along one edge faded by the sun, its ink paler and its paper yellower, by up to A (0 to 0.6); seed=N",
                "Sun-faded", "Strength", 0, 0.6, 0.05, ["seed"], (i, s, v) => PaperImageOps.Sunfade(i, v, Seed(s))),
            Make("piece", "piece=N: N pieces torn out of the page's edge (0 to 3), deep ragged bites inside the margins; fill=V; seed=N",
                "Torn-out pieces", "Pieces", 0, 3, 1, ["seed", "fill"], (i, s, v) => PaperImageOps.Piece(i, (int)v, Fill(s), Seed(s)), whole: true),
            Make("whiteout", "whiteout=N: N dried patches of correction fluid (0 to 8), only on blank paper; seed=N",
                "Correction fluid", "Patches", 0, 8, 1, ["seed"], (i, s, v) => PaperImageOps.Whiteout(i, (int)v, Seed(s)), whole: true),
            Make("sticky", "sticky=N: N sticky notes stuck on an edge (0 to 2), the part on the page in the margin; seed=N",
                "Sticky notes", "Notes", 0, 2, 1, ["seed"], (i, s, v) => PaperImageOps.Sticky(i, (int)v, Seed(s)), whole: true),
            Make("burnhole", "burnhole=N: N burn holes with a charred ring and a scorch (0 to 4), in the margins; fill=V; seed=N",
                "Burn holes", "Holes", 0, 4, 1, ["seed", "fill"], (i, s, v) => PaperImageOps.BurnHole(i, (int)v, Fill(s), Seed(s)), whole: true),
            Make("cutoff", "cutoff=D: the page cut off by the scanner along one or two sides, up to D of its shorter side (0 to 0.05, inside the margins); fill=V the scanner bed; seed=N",
                "Cut off", "Depth", 0, 0.05, 0.005, ["seed", "fill"], (i, s, v) => PaperImageOps.Cutoff(i, v, Fill(s), Seed(s))),
            Make("photocopy", "photocopy=N: a photocopy N times over (0 to 3): grey, harder contrast, bolder strokes, toner dust; seed=N",
                "Photocopy", "Generations", 0, 3, 1, ["seed"], (i, s, v) => PaperImageOps.Photocopy(i, (int)v, Seed(s)), whole: true),
            Make("carbon", "carbon=A: a carbon copy: the ink turned carbon blue or violet by A (0 to 1), soft, its edge doubled faintly; seed=N",
                "Carbon copy", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Carbon(i, v, Seed(s))),
            Make("lowtoner", "lowtoner=A: low toner: the ink faded in streaks down the page by up to A (0 to 0.8); seed=N",
                "Low toner", "Strength", 0, 0.8, 0.05, ["seed"], (i, s, v) => PaperImageOps.LowToner(i, v, Seed(s))),
            Make("banding", "banding=A: printer banding: the ink lighter in bands across the page, by up to A (0 to 0.4); seed=N",
                "Printer banding", "Strength", 0, 0.4, 0.02, ["seed"], (i, s, v) => PaperImageOps.Banding(i, v, Seed(s))),
            Make("ghost", "ghost=A: a faint copy of the page's ink a little way off (a double image), A as dark (0 to 0.4); seed=N",
                "Ghost image", "Strength", 0, 0.4, 0.02, ["seed"], (i, s, v) => PaperImageOps.Ghost(i, v, Seed(s))),
            Make("feather", "feather=A: ink spread into soft paper, strokes fuzzier and bolder by A (0 to 1); seed=N",
                "Ink feathering", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Feather(i, v, Seed(s))),
            Make("misregister", "misregister=P: a colour print's red and blue shifted up to P pixels apart (0 to 4; a grey page unchanged); seed=N",
                "Misregistration", "Pixels", 0, 4, 0.5, ["seed"], (i, s, v) => PaperImageOps.Misregister(i, v, Seed(s))),
            Make("desk", "desk=S: a page photographed on a desk: up to S smaller (0 to 0.2), on wood, a grey table or a dark cloth, with its shadow; seed=N",
                "On a desk", "Shrink", 0, 0.2, 0.01, ["seed"], (i, s, v) => PaperImageOps.Desk(i, v, Seed(s))),
            Make("glare", "glare=A: a soft bright reflection washing part of the page toward white by up to A (0 to 0.6); seed=N",
                "Glare", "Strength", 0, 0.6, 0.05, ["seed"], (i, s, v) => PaperImageOps.Glare(i, v, Seed(s))),
            Make("vignette", "vignette=A: the corners darker than the middle by up to A (0 to 0.6), as a camera lens; seed=N",
                "Vignetting", "Strength", 0, 0.6, 0.05, ["seed"], (i, s, v) => PaperImageOps.Vignette(i, v, Seed(s))),
            Make("motion", "motion=L: a camera moving L pixels in one direction while taking the photo (0 to 6); seed=N",
                "Motion blur", "Pixels", 0, 6, 0.5, ["seed"], (i, s, v) => PaperImageOps.Motion(i, v, Seed(s))),
            Make("moire", "moire=A: faint interference waves of a page photographed off a screen or a halftone, A strong (0 to 0.25); seed=N",
                "Moire", "Strength", 0, 0.25, 0.01, ["seed"], (i, s, v) => PaperImageOps.Moire(i, v, Seed(s))),
            Make("handshadow", "handshadow=A: a hand's or a phone's soft shadow reaching in from an edge, up to A darker (0 to 0.5); seed=N",
                "Hand shadow", "Strength", 0, 0.5, 0.05, ["seed"], (i, s, v) => PaperImageOps.HandShadow(i, v, Seed(s))),
            Make("fingerprint", "fingerprint=N: N faint greasy fingerprints (0 to 4); seed=N",
                "Fingerprints", "Prints", 0, 4, 1, ["seed"], (i, s, v) => PaperImageOps.Fingerprint(i, (int)v, Seed(s)), whole: true),
            Make("lined", "lined=A: ruled paper: faint blue lines (and now and then a red margin) under the ink, A strong (0 to 1); seed=N",
                "Ruled paper", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Lined(i, v, Seed(s))),
            Make("grid", "grid=A: graph paper: a faint grid under the ink, A strong (0 to 1); seed=N",
                "Graph paper", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Grid(i, v, Seed(s))),
            Make("guilloche", "guilloche=A: a security background of fine wavy lines in a light colour, A strong (0 to 1); seed=N",
                "Security pattern", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Guilloche(i, v, Seed(s))),
            Make("recycled", "recycled=A: recycled paper: fibres and a fine grain, A strong (0 to 1); seed=N",
                "Recycled paper", "Strength", 0, 1, 0.05, ["seed"], (i, s, v) => PaperImageOps.Recycled(i, v, Seed(s))),
            Make("shadow", "shadow=S: a shadow along one or two edges (a scanner lid or a book's gutter), S of the page wide (0 to 0.3); seed=N",
                "Edge shadow", "Width", 0, 0.3, 0.01, ["seed"], (i, s, v) => PaperImageOps.Shadow(i, v, Seed(s))),
            Make("streak", "streak=N: N streaks down the page from a dirty scanner glass or roller; seed=N",
                "Scanner streaks", "Streaks", 0, 20, 1, ["seed"], (i, s, v) => PaperImageOps.Streak(i, (int)v, Seed(s)), whole: true),
            Make("light", "light=A: uneven lighting, up to A darker in places (0 to 0.6); seed=N",
                "Uneven light", "Strength", 0, 0.6, 0.05, ["seed"], (i, s, v) => PaperImageOps.Light(i, v, Seed(s))),
            Make("curl", "curl=A: one edge of the page curled up toward the camera or the scanner lid (0 to 1); fill=V; seed=N",
                "Page curl", "Strength", 0, 1, 0.05, ["seed", "fill"], (i, s, v) => PaperImageOps.Curl(i, v, Fill(s), Seed(s))),
        ])
        {
            ImageTransforms.Register(transform);
        }
    }

    // A transform whose value is a number in [min, max] (a whole one when `whole`), with the options `keys`.
    private static IImageTransform Make(string name, string summary, string label, string valueLabel, double min, double max, double step, string[] keys,
        Func<ImageData, ImageTransformStep, double, ImageData> apply, bool whole = false)
    {
        var parameters = new List<ImageTransformParameter>
        {
            new("", valueLabel, whole ? ImageTransformInput.Integer : ImageTransformInput.Number) { Min = min, Max = max, Step = step },
        };
        if (keys.Contains("seed"))
        {
            parameters.Add(SeedInput);
        }

        if (keys.Contains("fill"))
        {
            parameters.Add(FillInput);
        }

        if (keys.Contains("tape"))
        {
            parameters.Add(new ImageTransformParameter("tape", "Tape", ImageTransformInput.Choice) { Choices = ["any", "clear", "yellowed", "matte"], Optional = true });
        }

        if (keys.Contains("color"))
        {
            parameters.Add(new ImageTransformParameter("color", "Pen", ImageTransformInput.Choice) { Choices = ["any", "blue", "black", "red", "green"], Optional = true });
        }

        return new LibraryImageTransforms.Transform(name, summary, keys,
            s =>
            {
                double v = Check(s, min, max, keys.Contains("fill") ? "fill" : null);
                if (whole && v != Math.Floor(v))
                {
                    throw new ArgumentException($"{name} is a whole number, not {v.ToString(CultureInfo.InvariantCulture)}.");
                }

                if (keys.Contains("color"))
                {
                    PenColor(s);
                }

                if (keys.Contains("tape"))
                {
                    TapeKind(s);
                }

                foreach (string key in keys.Where(Options.ContainsKey))
                {
                    _ = Option(s, key);
                }
            },
            (i, s) => apply(i, s, s.Number()))
        {
            Label = label, Parameters = parameters,
        };
    }

    // The number options the paper effects take: what each is when not given, and its range.
    private static readonly Dictionary<string, (double Otherwise, double Min, double Max)> Options = new()
    {
        ["length"] = (30, 2, 200), ["size"] = (5, 1, 30), ["gap"] = (3, 0, 12), ["shift"] = (3, 0, 12),
    };

    private static double Option(ImageTransformStep step, string key)
    {
        var (otherwise, min, max) = Options[key];
        return OptionNumber(step, key, otherwise, min, max);
    }

    private static string TapeKind(ImageTransformStep step)
    {
        string kind = (step.Option("tape") ?? "any").ToLowerInvariant();
        return kind is "any" or "clear" or "yellowed" or "matte" ? kind : throw new ArgumentException($"{step.Name}'s tape is any, clear, yellowed or matte, not '{kind}'.");
    }

    private static string PenColor(ImageTransformStep step)
    {
        string color = (step.Option("color") ?? "any").ToLowerInvariant();
        return color is "any" or "blue" or "black" or "red" or "green" ? color : throw new ArgumentException($"{step.Name}'s color is any, blue, black, red or green, not '{color}'.");
    }

    /// <summary>The library's paper scan profiles: handled, marked, damaged, scanner, aged, copied, photo, wrecked, and mixed (every effect at a small chance).</summary>
    internal static void RegisterProfiles()
    {
        ScanProfiles.Register(new ScanProfile("handled",
            "wrinkle=0.2..0.7@0.6,fold=1..3@0.5,dogear=0.03..0.08@0.25,staple=1..1@0.25,clip=1..1@0.2,smudge=1..4@0.5,blot=1..3@0.3,fingerprint=1..2@0.25,"
            + "residue=1..2@0.15,whiteout=1..2@0.15,sticky=1..1@0.15,rotate=-1.0..1.0@0.15,blur=0.3..0.8@0.3,noise=2..6@0.4,jpeg=50..90@0.5",
            "a paper that went through hands: wrinkles, folds, a dog-ear, staples, clips, smudges, ink blots, fingerprints, correction fluid, a sticky note"));
        ScanProfiles.Register(new ScanProfile("marked",
            "strike=1..3@0.5,underline=1..4@0.5,cross=1..1@0.25,highlight=1..4@0.5,scribble=1..4@0.4,rotate=-0.8..0.8@0.1,jpeg=60..95@0.5",
            "a page someone worked on: lines struck through, underlined, crossed out, highlighted, notes in the margins"));
        ScanProfiles.Register(new ScanProfile("damaged",
            "stain=1..2@0.3,water=1..2@0.2,dropout=0.02..0.1@0.3,fold=1..2@0.4,tear=0.01..0.04@0.5,piece=1..1@0.25,holes=2..3@0.3,wormholes=2..10@0.15,"
            + "burnhole=1..2@0.1,burn=0.01..0.03@0.1,scratch=2..12@0.5,taped=1..2@0.35,rotate=-1.2..1.2@0.2,jpeg=50..90@0.5",
            "a damaged paper: torn edges, torn-out pieces, punch holes, worm and burn holes, scorched edges, scratches, ripped and taped back, water, stains, worn ink"));
        ScanProfiles.Register(new ScanProfile("scanner",
            "light=0.1..0.4@0.6,curl=0.2..0.8@0.3,shadow=0.04..0.2@0.6,streak=1..6@0.4,cutoff=0.005..0.03@0.2,rotate=-1.0..1.0@0.2,noise=2..6@0.5,jpeg=45..90@0.6",
            "a hasty scan: an edge shadow, uneven light, a curled edge, streaks from the glass, the page cut off"));
        ScanProfiles.Register(new ScanProfile("aged",
            "tint=0.2..0.55@0.8,recycled=0.2..0.7@0.3,sunfade=0.2..0.5@0.4,foxing=20..200@0.6,water=1..2@0.3,residue=1..3@0.3,staple=1..1@0.3,dogear=0.03..0.1@0.3,"
            + "wormholes=2..12@0.25,fade=0.1..0.4@0.4,dropout=0.02..0.1@0.3,rotate=-1.0..1.0@0.15,jpeg=50..90@0.5",
            "an old paper: yellowed, sun-faded, foxed, water-marked, tape glue, an old staple, a dog-ear, worm holes, faded ink"));
        ScanProfiles.Register(new ScanProfile("copied",
            "carbon=0.5..1@0.2,lowtoner=0.2..0.6@0.3,banding=0.05..0.25@0.3,ghost=0.05..0.25@0.2,feather=0.3..0.9@0.25,misregister=1..3@0.15,photocopy=1..3@0.7,"
            + "speckle=50..300@0.5,cutoff=0.005..0.03@0.25,streak=1..3@0.3,rotate=-1.0..1.0@0.2,jpeg=50..90@0.5",
            "a copy or a poor print: photocopied once or more, carbon, low toner, banding, a ghost image, feathered ink, colours out of register"));
        ScanProfiles.Register(new ScanProfile("photo",
            "desk=0.03..0.12@0.4,perspective=0..0.02@0.3,glare=0.1..0.45@0.35,handshadow=0.1..0.4@0.3,fingerprint=1..2@0.1,vignette=0.1..0.45@0.5,light=0.1..0.3@0.4,"
            + "motion=1..3@0.2,blur=0.3..1.0@0.4,moire=0.03..0.12@0.1,noise=3..8@0.6,rotate=-1.5..1.5@0.15,jpeg=45..85@0.7",
            "a page photographed with a phone: on a desk, at an angle, glare, a hand's shadow, darker corners, a little motion, moire"));
        ScanProfiles.Register(new ScanProfile("wrecked",
            "foxing=20..150@0.4,water=1..3@0.5,stain=1..3@0.5,wrinkle=0.3..0.8@0.6,fold=1..3@0.6,dogear=0.03..0.1@0.3,smudge=1..4@0.4,dropout=0.05..0.15@0.5,"
            + "tear=0.01..0.04@0.5,piece=1..2@0.5,burn=0.01..0.04@0.4,holes=2..3@0.3,burnhole=1..3@0.3,wormholes=2..10@0.3,scratch=4..16@0.6,taped=1..2@0.5,"
            + "rotate=-1.0..1.0@0.2,jpeg=50..90@0.6",
            "a wrecked paper: many kinds of damage at once, the text still readable"));
        ScanProfiles.Register(new ScanProfile("mixed",
            "lined=0.3..0.8@0.04,grid=0.3..0.8@0.03,guilloche=0.2..0.6@0.04,recycled=0.2..0.7@0.06,tint=0.1..0.5@0.2,sunfade=0.2..0.5@0.06,foxing=20..150@0.06,"
            + "carbon=0.5..1@0.03,lowtoner=0.2..0.6@0.05,banding=0.05..0.25@0.05,ghost=0.05..0.25@0.04,feather=0.3..0.9@0.05,misregister=1..3@0.03,"
            + "stain=1..3@0.1,water=1..2@0.06,strike=1..2@0.12,underline=1..3@0.15,cross=1..1@0.06,highlight=1..3@0.15,scribble=1..3@0.12,"
            + "fade=0.2..0.7@0.2,uneven=0..0.7,dropout=0.02..0.15@0.15,smudge=1..3@0.15,blot=1..2@0.1,fingerprint=1..2@0.05,bleed=0.05..0.2@0.1,"
            + "wrinkle=0.1..0.6@0.2,fold=1..2@0.15,dogear=0.03..0.08@0.05,staple=1..1@0.05,clip=1..1@0.04,residue=1..2@0.04,whiteout=1..2@0.04,sticky=1..1@0.03,"
            + "tear=0.01..0.035@0.12,piece=1..1@0.04,holes=2..3@0.1,wormholes=2..8@0.04,burnhole=1..2@0.02,burn=0.01..0.03@0.02,scratch=1..8@0.15,taped=1..1@0.06,"
            + "photocopy=1..2@0.05,curl=0.2..0.6@0.08,shadow=0.04..0.15@0.15,light=0.1..0.35@0.2,streak=1..4@0.1,cutoff=0.005..0.03@0.04,"
            + "desk=0.03..0.1@0.03,glare=0.1..0.4@0.04,handshadow=0.1..0.35@0.03,vignette=0.1..0.4@0.05,motion=1..2.5@0.03,moire=0.03..0.1@0.02,"
            + "speckle=20..300@0.25,rotate=-1.5..1.5@0.15,perspective=0..0.012@0.05,blur=0.3..1.0@0.3,noise=2..7@0.3,lowres=0.5..0.9@0.2,jpeg=40..92@0.5",
            "a shuffle of everything: each paper, pen, damage, copier, camera and scanner effect at a small chance of its own, so pages get every mix"));
    }
}

/// <summary>The paper effects on 8-bit planes; each returns a new image and leaves its input alone.</summary>
internal static partial class PaperImageOps
{
    internal enum PenMark
    {
        Strike,
        Underline,
        Cross,
        Scribble,
    }

    // A text line found on the page: its rows and the columns its ink spans.
    private readonly record struct TextBand(int Top, int Bottom, int Left, int Right);

    // The page's text lines, from the rows holding ink (darker than the paper by a third): runs of such rows, each with
    // the span of its ink.
    private static List<TextBand> Bands(byte[][] planes, int w, int h, byte[] paper)
    {
        var ink = InkMask(planes, w, h, paper);
        var bands = new List<TextBand>();
        int top = -1, left = w, right = -1;
        for (int y = 0; y <= h; y++)
        {
            int count = 0, l = w, r = -1;
            if (y < h)
            {
                var row = ink.AsSpan(y * w, w);
                for (int x = 0; x < w; x++)
                {
                    if (row[x])
                    {
                        count++;
                        l = Math.Min(l, x);
                        r = x;
                    }
                }
            }

            if (count > Math.Max(2, w / 200))
            {
                top = top < 0 ? y : top;
                left = Math.Min(left, l);
                right = Math.Max(right, r);
            }
            else if (top >= 0)
            {
                if (y - top >= 6 && right - left > w / 20)
                {
                    bands.Add(new TextBand(top, y - 1, left, right));
                }

                (top, left, right) = (-1, w, -1);
            }
        }

        return bands;
    }

    // The page's ink: the pixels darker than the paper by a third, less the dark regions that touch the page's border (the
    // background through a tear or a missing piece, a burnt edge, a scanner bed, a desk), which are not text.
    private static bool[] InkMask(byte[][] planes, int w, int h, byte[] paper)
    {
        var grey = planes.Length == 1 ? planes[0] : PillowImageOps.Grey(planes);
        int dark = (int)(paper.Average(p => p) * 0.66);
        var ink = new bool[w * h];
        for (int i = 0; i < ink.Length; i++)
        {
            ink[i] = grey[i] < dark;
        }

        var stack = new Stack<int>();
        void Clear(int i)
        {
            if (ink[i])
            {
                ink[i] = false;
                stack.Push(i);
            }
        }

        for (int x = 0; x < w; x++)
        {
            Clear(x);
            Clear((h - 1) * w + x);
        }

        for (int y = 0; y < h; y++)
        {
            Clear(y * w);
            Clear(y * w + w - 1);
        }

        while (stack.Count > 0)
        {
            int i = stack.Pop(), x = i % w;
            if (x > 0)
            {
                Clear(i - 1);
            }

            if (x < w - 1)
            {
                Clear(i + 1);
            }

            if (i >= w)
            {
                Clear(i - w);
            }

            if (i < ink.Length - w)
            {
                Clear(i + w);
            }
        }

        return ink;
    }

    // The box holding the page's ink (darker than the paper by a third): its left, top, right and bottom; the whole page
    // when there is none. Tears and holes stay outside it, so no text a page's answer holds is cut.
    private static (int Left, int Top, int Right, int Bottom) InkBox(byte[][] planes, int w, int h)
    {
        var ink = InkMask(planes, w, h, ScanImageOps.PaperLevels(planes, w * h));
        int left = w, top = h, right = -1, bottom = -1;
        for (int y = 0; y < h; y++)
        {
            var row = ink.AsSpan(y * w, w);
            for (int x = 0; x < w; x++)
            {
                if (row[x])
                {
                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                    top = Math.Min(top, y);
                    bottom = y;
                }
            }
        }

        return right < 0 ? (0, 0, w - 1, h - 1) : (left, top, right, bottom);
    }

    private static (float R, float G, float B) Pen(string color, Random random)
    {
        if (color == "any")
        {
            double p = random.NextDouble();
            color = p < 0.55 ? "blue" : p < 0.85 ? "black" : p < 0.95 ? "red" : "green";
        }

        return color switch
        {
            "black" => (25, 25, 30),
            "red" => (175, 30, 35),
            "green" => (20, 110, 55),
            _ => ((float)(15 + random.Next(30)), (float)(35 + random.Next(30)), (float)(120 + random.Next(50))),
        };
    }

    private static float[] Channels((float R, float G, float B) color, int channels) =>
        channels == 3 ? [color.R, color.G, color.B] : [0.299f * color.R + 0.587f * color.G + 0.114f * color.B];

    // A pen stroke along `points`, `thickness` pixels wide with soft edges, laid on with `opacity`.
    private static void Stroke(byte[][] planes, int w, int h, IReadOnlyList<(double X, double Y)> points, double thickness, float[] color, float opacity,
        bool lighten = false)
    {
        double half = thickness / 2;
        for (int k = 1; k < points.Count; k++)
        {
            var (ax, ay) = points[k - 1];
            var (bx, by) = points[k];
            int x0 = Math.Max(0, (int)(Math.Min(ax, bx) - half - 1)), x1 = Math.Min(w - 1, (int)(Math.Max(ax, bx) + half + 1));
            int y0 = Math.Max(0, (int)(Math.Min(ay, by) - half - 1)), y1 = Math.Min(h - 1, (int)(Math.Max(ay, by) + half + 1));
            double dx = bx - ax, dy = by - ay, length2 = Math.Max(1e-9, dx * dx + dy * dy);
            for (int y = y0; y <= y1; y++)
            {
                for (int x = x0; x <= x1; x++)
                {
                    double px = x + 0.5 - ax, py = y + 0.5 - ay;
                    double t = Math.Clamp((px * dx + py * dy) / length2, 0, 1);
                    double ex = px - t * dx, ey = py - t * dy;
                    float cover = (float)Math.Clamp(half + 0.5 - Math.Sqrt(ex * ex + ey * ey), 0, 1) * opacity;
                    if (cover <= 0)
                    {
                        continue;
                    }

                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        // Ink on ink stays as dark as the darker; a lightening stroke (a scratch) only lightens.
                        float target = lighten ? Math.Max(v, color[c]) : Math.Min(v, color[c]);
                        v = (byte)(v + (target - v) * cover + 0.5f);
                    }
                }
            }
        }
    }

    // A hand-drawn line from (x0, y0) to (x1, y1): a slight wave and a slight tremor, about every 6 pixels.
    private static List<(double X, double Y)> HandLine(double x0, double y0, double x1, double y1, Random random, double wave)
    {
        double length = Math.Sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        int steps = Math.Max(2, (int)(length / 6));
        double phase = random.NextDouble() * 6.3, period = 80 + random.NextDouble() * 160;
        double nx = -(y1 - y0) / Math.Max(1, length), ny = (x1 - x0) / Math.Max(1, length);
        var points = new List<(double, double)>(steps + 1);
        for (int k = 0; k <= steps; k++)
        {
            double t = (double)k / steps, along = t * length;
            double off = wave * Math.Sin(phase + along / period * 6.28) + (random.NextDouble() - 0.5) * 0.6;
            points.Add((x0 + (x1 - x0) * t + nx * off, y0 + (y1 - y0) * t + ny * off));
        }

        return points;
    }

    /// <summary>Pen marks over the text (struck through, underlined, crossed out) or squiggles in its margins.</summary>
    public static ImageData Pen(ImageData image, PenMark mark, int count, string color, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var bands = Bands(planes, w, h, ScanImageOps.PaperLevels(planes, w * h));
        for (int k = 0; k < count; k++)
        {
            var ink = Channels(Pen(color, random), planes.Length);
            double thickness = 1.2 + random.NextDouble() * 1.6;
            float opacity = (float)(0.7 + random.NextDouble() * 0.25);
            if (mark == PenMark.Scribble || bands.Count == 0)
            {
                Scribble(planes, w, h, bands, random, ink, thickness, opacity);
                continue;
            }

            var band = bands[random.Next(bands.Count)];
            int span = band.Right - band.Left;
            double from = band.Left + span * random.NextDouble() * 0.4, to = band.Right - span * random.NextDouble() * 0.4;
            double slope = (random.NextDouble() - 0.5) * 0.01 * (to - from);
            switch (mark)
            {
                case PenMark.Strike:
                {
                    double y = (band.Top + band.Bottom) / 2.0 + (random.NextDouble() - 0.5) * 0.3 * (band.Bottom - band.Top);
                    Stroke(planes, w, h, HandLine(from, y, to, y + slope, random, 0.8), thickness, ink, opacity);
                    break;
                }

                case PenMark.Underline:
                {
                    double y = band.Bottom + 1.5 + random.NextDouble() * 3;
                    Stroke(planes, w, h, HandLine(from, y, to, y + slope, random, 0.6), thickness, ink, opacity);
                    break;
                }

                default:
                {
                    // An X over two to four lines from this one down, across at least a third of their width.
                    int index = bands.IndexOf(band), last = Math.Min(bands.Count - 1, index + 1 + random.Next(3));
                    int left = bands.Skip(index).Take(last - index + 1).Min(b => b.Left), right = bands.Skip(index).Take(last - index + 1).Max(b => b.Right);
                    double width = (right - left) * (0.35 + 0.6 * random.NextDouble());
                    double x0 = left + (right - left - width) * random.NextDouble(), x1 = x0 + width;
                    double top = band.Top - 3, bottom = Math.Max(bands[last].Bottom + 3, top + 40);
                    Stroke(planes, w, h, HandLine(x0, top, x1, bottom, random, 1.5), thickness, ink, opacity * 0.9f);
                    Stroke(planes, w, h, HandLine(x0, bottom, x1, top, random, 1.5), thickness, ink, opacity * 0.9f);
                    break;
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    // A squiggle or a short note-like line in a margin, or now and then a ring around part of a text line.
    private static void Scribble(byte[][] planes, int w, int h, List<TextBand> bands, Random random, float[] ink, double thickness, float opacity)
    {
        if (bands.Count > 0 && random.NextDouble() < 0.3)
        {
            var band = bands[random.Next(bands.Count)];
            double cx = band.Left + (band.Right - band.Left) * (0.2 + 0.6 * random.NextDouble()), cy = (band.Top + band.Bottom) / 2.0;
            double rx = (band.Right - band.Left) * (0.05 + 0.12 * random.NextDouble()) + 20, ry = (band.Bottom - band.Top) * 0.9 + 6;
            var ring = new List<(double, double)>();
            double start = random.NextDouble() * 6.3, turn = 6.28 * (1.05 + 0.1 * random.NextDouble());
            for (int k = 0; k <= 60; k++)
            {
                double a = start + turn * k / 60, wobble = 1 + 0.05 * Math.Sin(a * 3 + start);
                ring.Add((cx + rx * wobble * Math.Cos(a), cy + ry * wobble * Math.Sin(a)));
            }

            Stroke(planes, w, h, ring, thickness, ink, opacity);
            return;
        }

        // In a margin: the left or right tenth, or the top or bottom twentieth.
        int side = random.Next(4);
        double x = side switch { 0 => w * 0.01 + random.NextDouble() * w * 0.07, 1 => w * 0.92 + random.NextDouble() * w * 0.07, _ => w * (0.1 + 0.8 * random.NextDouble()) };
        double y = side switch { 2 => h * 0.005 + random.NextDouble() * h * 0.04, 3 => h * 0.955 + random.NextDouble() * h * 0.04, _ => h * (0.1 + 0.8 * random.NextDouble()) };
        var points = new List<(double, double)> { (x, y) };
        double heading = side < 2 ? Math.PI / 2 : 0, speed = 3 + random.NextDouble() * 3;
        int steps = 15 + random.Next(50);
        for (int k = 0; k < steps; k++)
        {
            heading += (random.NextDouble() - 0.5) * 1.4;
            x = Math.Clamp(x + Math.Cos(heading) * speed, 1, w - 2);
            y = Math.Clamp(y + Math.Sin(heading) * speed, 1, h - 2);
            points.Add((x, y));
        }

        Stroke(planes, w, h, points, thickness, ink, opacity);
    }

    /// <summary>See-through highlighter bands over text lines (multiplied in, so the ink stays dark).</summary>
    public static ImageData Highlight(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var bands = Bands(planes, w, h, ScanImageOps.PaperLevels(planes, w * h));
        (float, float, float)[] markers = [(250, 240, 90), (170, 240, 140), (250, 170, 210), (160, 215, 250)];
        for (int k = 0; k < count && bands.Count > 0; k++)
        {
            var band = bands[random.Next(bands.Count)];
            var tint = Channels(markers[random.Next(markers.Length)], planes.Length);
            float strength = (float)(0.55 + random.NextDouble() * 0.35);
            int span = band.Right - band.Left;
            int x0 = band.Left + (int)(span * random.NextDouble() * 0.5) - 3, x1 = band.Right - (int)(span * random.NextDouble() * 0.4) + 3;
            int pad = Math.Max(1, (band.Bottom - band.Top) / 6);
            for (int y = Math.Max(0, band.Top - pad); y <= Math.Min(h - 1, band.Bottom + pad); y++)
            {
                // Ragged ends, as a marker leaves them.
                int start = Math.Max(0, x0 + random.Next(-3, 4)), end = Math.Min(w - 1, x1 + random.Next(-3, 4));
                for (int x = start; x <= end; x++)
                {
                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        v = (byte)(v * (1 - strength * (1 - tint[c] / 255f)) + 0.5f);
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Crumpled paper: soft random creases shading the page and moving the ink by a pixel or two.</summary>
    public static ImageData Wrinkle(ImageData image, double strength, int seed)
    {
        if (strength == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        int creases = 4 + (int)(strength * 10);
        var lines = new (double Nx, double Ny, double C, double Width, double Amplitude)[creases];
        for (int k = 0; k < creases; k++)
        {
            double angle = random.NextDouble() * Math.PI;
            double nx = Math.Cos(angle), ny = Math.Sin(angle);
            double c = nx * random.NextDouble() * w + ny * random.NextDouble() * h;
            lines[k] = (nx, ny, c, 15 + random.NextDouble() * 90, (random.NextDouble() < 0.5 ? -1 : 1) * (0.4 + 0.6 * random.NextDouble()));
        }

        double lightX = Math.Cos(0.8), lightY = Math.Sin(0.8), shift = 2.5 * strength;
        var shade = new float[w * h];
        var warped = ScanImageOps.WarpRows(image, (y, sxs, sys) =>
        {
            for (int x = 0; x < sxs.Length; x++)
            {
                double gx = 0, gy = 0, dx = 0, dy = 0;
                foreach (var (nx, ny, c, width, amplitude) in lines)
                {
                    double d = (nx * x + ny * y - c) / width;
                    if (d is > -3 and < 3)
                    {
                        double e = Math.Exp(-d * d);
                        double slope = amplitude * -2 * d * e;   // the crease's slope across it
                        gx += slope * nx;
                        gy += slope * ny;
                        dx += amplitude * e * nx;
                        dy += amplitude * e * ny;
                    }
                }

                shade[y * sxs.Length + x] = (float)Math.Clamp(1 + strength * 0.12 * (gx * lightX + gy * lightY), 0.75, 1.08);
                sxs[x] = x + shift * dx;
                sys[x] = y + shift * dy;
            }
        }, null);
        return Multiply(warped, shade);
    }

    // Each pixel times its factor (the paper may brighten a little, to white at most).
    private static ImageData Multiply(ImageData image, float[] factor)
    {
        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
        {
            foreach (var plane in planes)
            {
                for (int i = first * w; i < last * w; i++)
                {
                    plane[i] = (byte)Math.Clamp(plane[i] * factor[i] + 0.5f, 0, 255);
                }
            }
        });

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Fold creases: a letter folded in thirds or halves, each fold a dark line, a light line beside it and a soft shade.</summary>
    public static ImageData Fold(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var factor = new float[w * h];
        Array.Fill(factor, 1f);
        bool across = random.NextDouble() < 0.75;   // most letters fold across their height
        for (int k = 0; k < count; k++)
        {
            double share = count == 1 ? 0.5 : (k + 1.0) / (count + 1);
            double angle = (random.NextDouble() - 0.5) * 0.03;
            double nx = across ? -Math.Sin(angle) : Math.Cos(angle), ny = across ? Math.Cos(angle) : Math.Sin(angle);
            double c = (across ? h : w) * (share + (random.NextDouble() - 0.5) * 0.04);
            float dark = (float)(0.72 + random.NextDouble() * 0.18), soft = (float)(0.03 + random.NextDouble() * 0.05);
            double reach = 30 + random.NextDouble() * 90;
            int side = random.NextDouble() < 0.5 ? -1 : 1;
            HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
            {
                for (int y = first; y < last; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        double d = nx * x + ny * y - c;
                        float f = 1;
                        if (Math.Abs(d) < 1.2)
                        {
                            f = dark + (1 - dark) * (float)(Math.Abs(d) / 1.2);
                        }
                        else if (d * side > 0 && d * side < 4)
                        {
                            f = 1.04f;
                        }

                        if (d * side < 0 && -d * side < reach)
                        {
                            f *= 1 - soft * (float)(1 + d * side / reach);
                        }

                        factor[y * w + x] *= f;
                    }
                }
            });
        }

        return Multiply(image, factor);
    }

    /// <summary>Smears: ink dragged a few tens of pixels by a hand, darkening and spreading it under small ellipses.</summary>
    public static ImageData Smudge(ImageData image, int count, double length, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        var source = planes.Select(p => (byte[])p.Clone()).ToArray();
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var bands = Bands(planes, w, h, ScanImageOps.PaperLevels(planes, w * h));
        for (int k = 0; k < count; k++)
        {
            double cx, cy;
            if (bands.Count > 0)
            {
                var band = bands[random.Next(bands.Count)];
                cx = band.Left + random.NextDouble() * (band.Right - band.Left);
                cy = (band.Top + band.Bottom) / 2.0;
            }
            else
            {
                (cx, cy) = (random.NextDouble() * w, random.NextDouble() * h);
            }

            double rx = 15 + random.NextDouble() * 45, ry = 6 + random.NextDouble() * 18, angle = random.NextDouble() * Math.PI;
            double ca = Math.Cos(angle), sa = Math.Sin(angle), drag = length * (0.5 + 0.5 * random.NextDouble());
            float strength = (float)(0.35 + random.NextDouble() * 0.35);
            int reach = (int)Math.Ceiling(Math.Max(rx, ry) + drag);
            for (int y = Math.Max(0, (int)cy - reach); y <= Math.Min(h - 1, (int)cy + reach); y++)
            {
                for (int x = Math.Max(0, (int)cx - reach); x <= Math.Min(w - 1, (int)cx + reach); x++)
                {
                    double dx = x - cx, dy = y - cy;
                    double u = (dx * ca + dy * sa) / rx, v = (-dx * sa + dy * ca) / ry;
                    double d = u * u + v * v;
                    if (d >= 1)
                    {
                        continue;
                    }

                    float falloff = (float)(1 - d) * strength;
                    for (int c = 0; c < planes.Length; c++)
                    {
                        // The darkest ink behind this pixel along the drag, as a finger leaves it.
                        int darkest = source[c][y * w + x];
                        for (int s = 1; s <= 6; s++)
                        {
                            int sx = (int)(x - ca * drag * s / 6), sy = (int)(y - sa * drag * s / 6);
                            if ((uint)sx < (uint)w && (uint)sy < (uint)h)
                            {
                                darkest = Math.Min(darkest, source[c][sy * w + sx] + 18 * s);
                            }
                        }

                        ref byte p = ref planes[c][y * w + x];
                        p = (byte)Math.Clamp(p + (Math.Min(p, darkest) - p) * falloff + 0.5f, 0, 255);
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Ink blots: a few overlapping drops and some splatter, small enough to leave letters readable.</summary>
    public static ImageData Blot(ImageData image, int count, double size, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        for (int k = 0; k < count; k++)
        {
            double cx = random.NextDouble() * w, cy = random.NextDouble() * h;
            var ink = random.NextDouble() < 0.6 ? Channels(Pen("blue", random), planes.Length) : [.. paper.Select(p => p * 0.1f)];
            float opacity = (float)(0.6 + random.NextDouble() * 0.3);
            int drops = 3 + random.Next(4), splatter = random.Next(8);
            for (int d = 0; d < drops + splatter; d++)
            {
                bool drop = d < drops;
                double r = drop ? size * (0.4 + 0.6 * random.NextDouble()) : 0.6 + random.NextDouble() * 1.2;
                double spread = drop ? size * 0.8 : size * 3;
                double x = cx + (random.NextDouble() - 0.5) * 2 * spread, y = cy + (random.NextDouble() - 0.5) * 2 * spread;
                Stroke(planes, w, h, [(x, y), (x + 0.01, y)], 2 * r, ink, opacity);
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Torn edges: one to three sides torn along a jagged line inside the margin, and now and then a corner off.</summary>
    public static ImageData Tear(ImageData image, double depth, int? fill, int seed)
    {
        if (depth == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        double most = depth * side;
        var ink = InkBox(planes, w, h);
        double[] room = [ink.Top - 6, w - 1 - ink.Right - 6, h - 1 - ink.Bottom - 6, ink.Left - 6];   // the blank margin of each side
        var cut = new double[4][];
        int torn = 1 + random.Next(3);
        var sides = Enumerable.Range(0, 4).OrderBy(_ => random.Next()).Take(torn).ToHashSet();
        for (int s = 0; s < 4; s++)
        {
            int length = s % 2 == 0 ? w : h;
            cut[s] = new double[length];
            if (!sides.Contains(s))
            {
                continue;
            }

            // A jagged line: a random walk, deeper in a few torn bites.
            double at = most * 0.2 * random.NextDouble();
            int bites = 1 + random.Next(3);
            var biteAt = Enumerable.Range(0, bites).Select(_ => (Center: random.NextDouble() * length, Width: length * (0.05 + 0.2 * random.NextDouble()))).ToArray();
            for (int i = 0; i < length; i++)
            {
                at = Math.Clamp(at + (random.NextDouble() - 0.5) * 1.6, 0, most * 0.3);
                double bite = biteAt.Max(b => Math.Max(0, 1 - Math.Abs(i - b.Center) / b.Width));
                cut[s][i] = Math.Max(0, Math.Min(Math.Min(most, room[s]), at + most * Math.Pow(bite, 1.5)));
            }
        }

        // A corner folded or torn off, now and then.
        int corner = random.NextDouble() < 0.4 ? random.Next(4) : -1;
        double cornerSize = most * (1.2 + random.NextDouble());
        if (corner >= 0)
        {
            // Every ink pixel's distances to the corner's two sides add up to at least those of the ink box's corner.
            double nearest = (corner is 1 or 2 ? room[1] : room[3]) + (corner is 2 or 3 ? room[2] : room[0]);
            cornerSize = Math.Min(cornerSize, nearest);
            corner = cornerSize > 4 ? corner : -1;
        }
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double inside = Math.Min(Math.Min(y - cut[0][x], h - 1 - y - cut[2][x]), Math.Min(x - cut[3][y], w - 1 - x - cut[1][y]));
                if (corner >= 0)
                {
                    double cx = corner is 1 or 2 ? w - 1 - x : x, cy = corner is 2 or 3 ? h - 1 - y : y;
                    inside = Math.Min(inside, (cx + cy - cornerSize) / 1.414);
                }

                if (inside >= 1.5)
                {
                    continue;
                }

                // Off the paper: the background; at the torn edge, the paper's fibres a little lighter.
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = inside < 0 ? background[c] : (byte)Math.Min(255, v + 10);
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>
    /// A page ripped right across and taped back together: each tear a jagged line through the page (its text too), the
    /// piece beyond it moved up to <paramref name="gap"/> pixels away and <paramref name="shift"/> along it, the
    /// background through the gap (a light one grey, the edges' shadow), the torn edges a little lighter (the paper's fibres), then two to four strips of
    /// tape over each tear, all clear, yellowed or matte, the text under them still seen.
    /// </summary>
    public static ImageData Taped(ImageData image, int count, double gap, double shift, string tape, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        // Through the gap: the background, a light one seen grey (the shadow of the paper's edges), so the tear shows.
        var background = Background(planes.Length, fill, random);
        double shade = background[0] > 200 ? 0.55 + 0.2 * random.NextDouble() : 1;
        background = [.. background.Select(b => (byte)(b * shade))];
        var tears = new List<(bool Across, double[] Line, int Apart)>(count);
        for (int k = 0; k < count; k++)
        {
            bool across = random.NextDouble() < 0.7;   // most pages are ripped across their height
            int length = across ? w : h, extent = across ? h : w;

            // A jagged line: a slow bend, a faster wave, a random walk and fibre-sized jitter.
            double start = extent * (0.2 + 0.6 * random.NextDouble()), drift = extent * (random.NextDouble() - 0.5) * 0.12;
            double a1 = side * (0.004 + 0.012 * random.NextDouble()), p1 = length * (0.25 + 0.5 * random.NextDouble()), f1 = random.NextDouble() * 6.3;
            double a2 = side * (0.002 + 0.005 * random.NextDouble()), p2 = length * (0.06 + 0.1 * random.NextDouble()), f2 = random.NextDouble() * 6.3;
            var line = new double[length];
            double walk = 0;
            for (int u = 0; u < length; u++)
            {
                walk = Math.Clamp(walk + (random.NextDouble() - 0.5) * 1.2, -3, 3);
                line[u] = start + drift * u / length + a1 * Math.Sin(f1 + u * 6.283 / p1) + a2 * Math.Sin(f2 + u * 6.283 / p2) + walk + (random.NextDouble() - 0.5) * 0.8;
            }

            int along = (int)Math.Round((random.NextDouble() * 2 - 1) * shift), apart = (int)Math.Round(gap * (0.35 + 0.65 * random.NextDouble()));
            var source = planes.Select(p => (byte[])p.Clone()).ToArray();
            HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
            {
                for (int y = first; y < last; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int u = across ? x : y, v = across ? y : x, i = y * w + x;
                        double rim = line[u] - v;   // the first piece: on or before the line
                        int from = i;
                        if (rim < 0)
                        {
                            // The second piece, moved: where it was, unless that is off the page or on the first piece's side.
                            int su = u - along, sv = v - apart;
                            rim = su >= 0 && su < length && sv >= 0 ? sv - line[su] : -1;
                            if (rim <= 0)
                            {
                                for (int c = 0; c < planes.Length; c++)
                                {
                                    planes[c][i] = background[c];
                                }

                                continue;
                            }

                            from = across ? sv * w + su : su * w + sv;
                        }

                        for (int c = 0; c < planes.Length; c++)
                        {
                            byte value = source[c][from];
                            planes[c][i] = rim < 1.5 ? (byte)Math.Min(255, value + 14) : value;
                        }
                    }
                }
            });
            tears.Add((across, line, apart));
        }

        // The tape: one kind on the page (clear the most when any); strips across each tear, now and then one along it.
        double pick = random.NextDouble();
        string kind = tape != "any" ? tape : pick < 0.6 ? "clear" : pick < 0.85 ? "yellowed" : "matte";
        var sheen = new Random(random.Next());
        foreach (var (across, line, apart) in tears)
        {
            int strips = 2 + random.Next(3);
            for (int s = 0; s < strips; s++)
            {
                double uc = line.Length * (s + 0.5 + (random.NextDouble() - 0.5) * 0.5) / strips;
                double vc = line[Math.Clamp((int)uc, 0, line.Length - 1)] + apart / 2.0;
                bool lengthwise = random.NextDouble() < 0.15;
                double angle = (lengthwise ? 0 : Math.PI / 2) + (random.NextDouble() - 0.5) * 0.5;   // from the tear's direction
                double tapeLength = side * (lengthwise ? 0.18 + 0.15 * random.NextDouble() : 0.07 + 0.07 * random.NextDouble());
                double tapeWidth = side * (0.022 + 0.015 * random.NextDouble());
                double du = Math.Cos(angle), dv = Math.Sin(angle);
                Tape(planes, w, h, across ? uc : vc, across ? vc : uc, across ? du : dv, across ? dv : du, tapeLength, tapeWidth, kind, sheen);
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    // A strip of tape centred on (cx, cy) along (ax, ay), its ends cut in teeth as a dispenser leaves them. Clear tape is
    // transparent: a faint haze, a glossy streak along it, its edges seen as a dark line with a light one inside, now and
    // then a trapped bubble. Yellowed tape tints what is under it; matte tape frosts it lighter. The text stays readable.
    private static void Tape(byte[][] planes, int w, int h, double cx, double cy, double ax, double ay, double length, double width, string kind, Random random)
    {
        float[] tint = kind == "yellowed" ? Tint(planes.Length, 0.98f, 0.93f, 0.78f) : [.. Enumerable.Repeat(1f, planes.Length)];
        float lift = kind switch { "clear" => 0.04f, "yellowed" => 0.02f, _ => 0.16f };   // toward white: matte tape the most
        double streak = (random.NextDouble() - 0.5) * 1.2, streakWidth = 0.15 + 0.2 * random.NextDouble();
        float gloss = kind == "clear" ? (float)(0.12 + 0.12 * random.NextDouble()) : kind == "yellowed" ? 0.05f : 0f;
        var bubbles = Enumerable.Range(0, kind == "clear" && random.NextDouble() < 0.4 ? 1 + random.Next(3) : 0)
            .Select(_ => (A: (random.NextDouble() - 0.5) * 1.6, B: (random.NextDouble() - 0.5) * 1.4, R: 0.08 + 0.12 * random.NextDouble())).ToArray();
        Strip(w, h, cx, cy, ax, ay, length, width, true, (x, y, a, b, edge) =>
        {
            float shine = gloss * (float)Math.Max(0, 1 - Math.Abs(b - streak) / streakWidth);
            bool rim = false;
            foreach (var bubble in bubbles)
            {
                double d = Math.Sqrt((a - bubble.A) * (a - bubble.A) * 9 + (b - bubble.B) * (b - bubble.B));
                rim |= Math.Abs(d - bubble.R) < 0.04;
                shine += d < bubble.R ? 0.15f : 0f;
            }

            for (int c = 0; c < planes.Length; c++)
            {
                ref byte v = ref planes[c][y * w + x];
                float value = (v + (255 - v) * (lift + shine)) * tint[c];
                value *= edge ? 0.86f : rim ? 0.9f : 1f;
                v = (byte)Math.Clamp(value + 0.5f, 0, 255);
            }
        });
    }

    // What is seen through a hole or past a torn edge: the given grey, else a dark or a white scanner lid.
    private static byte[] Background(int channels, int? fill, Random random)
    {
        byte level = fill is { } f ? (byte)f : random.NextDouble() < 0.6 ? (byte)(30 + random.Next(60)) : (byte)(235 + random.Next(20));
        return [.. Enumerable.Repeat(level, channels)];
    }

    /// <summary>Punch holes in the side margin (a binder's two or three, or four), with the background through them and a faint rim.</summary>
    public static ImageData Holes(ImageData image, int count, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        double radius = Math.Max(4, Math.Min(w, h) * 0.009);
        bool left = random.NextDouble() < 0.7;
        var ink = InkBox(planes, w, h);
        double room = left ? ink.Left : w - 1 - ink.Right;   // the blank side margin
        radius = Math.Min(radius, (room - 8) / 2.5);
        if (radius < 3)
        {
            return image;   // no room beside the text for a hole
        }

        double x = left ? Math.Min(w * 0.025, room - radius - 5) : Math.Max(w * 0.975, w - room + radius + 5), spacing = h * (0.22 + 0.1 * random.NextDouble()), middle = h * (0.45 + 0.1 * random.NextDouble());
        for (int k = 0; k < count; k++)
        {
            double cy = middle + (k - (count - 1) / 2.0) * spacing, cx = x;
            for (int y = Math.Max(0, (int)(cy - radius - 3)); y <= Math.Min(h - 1, (int)(cy + radius + 3)); y++)
            {
                for (int px = Math.Max(0, (int)(cx - radius - 3)); px <= Math.Min(w - 1, (int)(cx + radius + 3)); px++)
                {
                    double d = Math.Sqrt((px + 0.5 - cx) * (px + 0.5 - cx) + (y + 0.5 - cy) * (y + 0.5 - cy));
                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + px];
                        if (d < radius)
                        {
                            v = background[c];
                        }
                        else if (d < radius + 2)
                        {
                            v = (byte)(v * 0.85f);
                        }
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Scratches: thin, long, slightly bent lines, most lighter than the paper (the scanner glass), some dark.</summary>
    public static ImageData Scratch(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        for (int k = 0; k < count; k++)
        {
            bool light = random.NextDouble() < 0.7;
            float[] color = light ? [.. paper.Select(p => Math.Min(255f, p + 25))] : [.. paper.Select(p => p * 0.45f)];
            double length = Math.Max(w, h) * (0.2 + 0.7 * random.NextDouble()), angle = random.NextDouble() * Math.PI;
            double x0 = random.NextDouble() * w, y0 = random.NextDouble() * h;
            var line = HandLine(x0, y0, x0 + Math.Cos(angle) * length, y0 + Math.Sin(angle) * length, random, 3);
            if (light)
            {
                Stroke(planes, w, h, line, 0.6 + random.NextDouble() * 0.9, color, (float)(0.4 + random.NextDouble() * 0.4), lighten: true);
            }
            else
            {
                Stroke(planes, w, h, line, 0.6 + random.NextDouble() * 0.8, color, (float)(0.3 + random.NextDouble() * 0.4));
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A shadow along one edge or two (a scanner lid left open, a book's gutter): darkest at the edge, fading in.</summary>
    public static ImageData Shadow(ImageData image, double width, int seed)
    {
        if (width == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var factor = new float[w * h];
        Array.Fill(factor, 1f);
        int edges = random.NextDouble() < 0.7 ? 1 : 2;
        foreach (int edge in Enumerable.Range(0, 4).OrderBy(_ => random.Next()).Take(edges))
        {
            float darkness = (float)(0.25 + random.NextDouble() * 0.45);
            double reach = width * (edge % 2 == 0 ? h : w);
            HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
            {
                for (int y = first; y < last; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        double d = edge switch { 0 => y, 1 => w - 1 - x, 2 => h - 1 - y, _ => x };
                        if (d < reach)
                        {
                            double t = 1 - d / reach;
                            factor[y * w + x] *= 1 - darkness * (float)(t * t);
                        }
                    }
                }
            });
        }

        return Multiply(image, factor);
    }

    /// <summary>Streaks down the page: a dirty scanner glass or roller, dark or light, stronger in places.</summary>
    public static ImageData Streak(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var factor = new float[w * h];
        Array.Fill(factor, 1f);
        for (int k = 0; k < count; k++)
        {
            double x0 = random.NextDouble() * w, width = 0.8 + random.NextDouble() * 3.5;
            float amount = (float)((random.NextDouble() < 0.65 ? -1 : 0.5) * (0.04 + random.NextDouble() * 0.14));
            double phase = random.NextDouble() * 6.3, period = h * (0.2 + random.NextDouble());
            for (int y = 0; y < h; y++)
            {
                float along = (float)(0.6 + 0.4 * Math.Sin(phase + y / period * 6.28));
                for (int x = Math.Max(0, (int)(x0 - width)); x <= Math.Min(w - 1, (int)(x0 + width)); x++)
                {
                    float cover = (float)Math.Clamp(width / 2 + 0.5 - Math.Abs(x + 0.5 - x0), 0, 1);
                    factor[y * w + x] *= 1 + amount * along * cover;
                }
            }
        }

        return Multiply(image, factor);
    }

    /// <summary>Uneven lighting: a smooth field over the page, up to <paramref name="amount"/> darker, and a slight vignette.</summary>
    public static ImageData Light(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var grid = ScanImageOps.SmoothGrid(seed, out int cells);
        var factor = new float[w * h];
        HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
        {
            for (int y = first; y < last; y++)
            {
                float v = (y + 0.5f) / h;
                for (int x = 0; x < w; x++)
                {
                    float u = (x + 0.5f) / w;
                    float edge = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
                    factor[y * w + x] = 1 - (float)amount * (0.7f * ScanImageOps.Smooth(grid, cells, u, v) + 0.6f * edge);
                }
            }
        });

        return Multiply(image, factor);
    }

    /// <summary>One edge curled up: the page bent and darkened toward that edge, the background where it lifted.</summary>
    public static ImageData Curl(ImageData image, double amount, int? fill, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        int edge = random.Next(4);
        double band = 0.15 + 0.2 * random.NextDouble(), lift = amount * 0.04;
        var shade = new float[w * h];
        var bent = ScanImageOps.WarpRows(image, (y, sxs, sys) =>
        {
            for (int x = 0; x < sxs.Length; x++)
            {
                // t: how far into the curling band, 0 where it starts to 1 at the edge.
                double along = edge switch { 0 => 1 - (double)y / h, 1 => (double)x / w, 2 => (double)y / h, _ => 1 - (double)x / w };
                double t = Math.Max(0, (along - (1 - band)) / band);
                double pull = lift * t * t;   // the edge drawn in toward the page's middle
                shade[y * sxs.Length + x] = 1 - (float)(0.3 * amount * t * t);
                sxs[x] = edge switch { 1 => x + pull * w * 1.0, 3 => x - pull * w, _ => x };
                sys[x] = edge switch { 0 => y - pull * h, 2 => y + pull * h, _ => y };
            }
        }, fill ?? (random.NextDouble() < 0.6 ? 30 + random.Next(60) : 240));
        return Multiply(bent, shade);
    }
}
