// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Testing;
using Idrak.Abstraction.Text;
using Idrak.Text;

// Handwriting habits on synthetic pages: an option, off by default (every page as before); at a chance, the body is
// written with the habits and the label says what is drawn, or keeps the source's spelling.
internal abstract partial class DesktopTests
{
    private static void SyntheticHabits(Device device)
    {
        var habits = TextHabits.Get("arabic-handwriting");
        Check(TextHabits.Origin("arabic-handwriting") == Overrides.Library && habits.Chances.Values.All(c => c == 0), $"habits: {TextHabits.Describe()}");

        // At chance 1 every habit where it holds: hamza and madda off alef, final ya and ta marbuta, dhah, the marks.
        string source = "أحمد في المدرسة إلى آخر ظلٍّ يوم";
        string all = habits.Apply(source, new Random(1), _ => 1);
        Check(all == "احمد فى المدرسه الى اخر ضل يوم", $"at chance 1: {all}");
        Check(ReferenceEquals(habits.Apply(source, new Random(1)), source) && habits.Apply("بيت كبير", new Random(1), _ => 1) == "بيت كبير",
            "at the habits' own chance (0) the text is the same; a text they do not hold for too");
        Check(habits.Apply(source, new Random(1), n => n == "final-ya" ? 1 : 0) == "أحمد فى المدرسة إلى آخر ظلٍّ يوم", "one habit alone; a ya not at a word's end is kept");

        Conformance.CheckTextHabits("arabic-handwriting", [source, "مدرسة جميلة", "أنا في البيت"], "بيت كبير").ThrowIfFailed();

        // Pages: no habits (the default), habits=none and habits at chance 0 draw the same bytes and labels.
        var amiri = FontFormats.Read(FontData("arabic/amiri.ttf")).Single();
        var options = new SyntheticPageOptions { Style = new PageStyle { Fonts = [amiri], Width = 620, Height = 877 }, Layouts = ["plain"], Scans = ["none"], Seed = 5 };
        var plain = SyntheticPages.Make(source, options);
        foreach (string settings in (string[])["habits=none", "habits=arabic-handwriting", "habits=arabic-handwriting,habit-chance=0,label=source"])
        {
            var page = SyntheticPages.Make(source, options with { Style = SyntheticPages.Style(options.Style, settings) });
            Check(page.Image.Pixels.AsSpan().SequenceEqual(plain.Image.Pixels) && page.Text == plain.Text && page.Answer == plain.Answer, $"{settings}: the page as before");
        }

        // At chance 1, label=drawn (the default): the label is the drawn text; label=source: the input's, the same image.
        var drawn = SyntheticPages.Make(source, options with { Style = SyntheticPages.Style(options.Style, "habits=arabic-handwriting,habit-chance=1") });
        var kept = SyntheticPages.Make(source, options with { Style = SyntheticPages.Style(options.Style, "habits=arabic-handwriting,habit-chance=1,label=source") });
        Check(drawn.Text == all && !drawn.Image.Pixels.AsSpan().SequenceEqual(plain.Image.Pixels), $"label=drawn: {drawn.Text}");
        Check(kept.Text == source && kept.Image.Pixels.AsSpan().SequenceEqual(drawn.Image.Pixels), $"label=source: {kept.Text}");
        var one = SyntheticPages.Make(source, options with { Style = SyntheticPages.Style(options.Style, "habits=arabic-handwriting,habit.alef-hamza-above=1") });
        Check(one.Text == "احمد في المدرسة إلى آخر ظلٍّ يوم", $"habit.NAME: {one.Text}");

        // A rules file; bad settings and files refused naming the key.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-habits-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string file = Path.Combine(folder, "mine.json");
            File.WriteAllText(file, """{"name": "mine", "rules": [{"name": "o", "from": "ب", "to": "ث", "where": "initial", "chance": 1}]}""", new UTF8Encoding(false));
            var mine = SyntheticPages.Make("باب بيت", options with { Style = SyntheticPages.Style(options.Style, $"habits={file}") });
            Check(mine.Text == "ثاب ثيت", $"a rules file: {mine.Text}");
            foreach (var (json, key) in new[]
            {
                ("""{"name": "x", "rules": [{"name": "o", "from": "a", "to": "b", "where": "end"}]}""", "rules[0].where"),
                ("""{"name": "x", "rules": [{"name": "o", "from": "a", "to": "b", "chance": 2}]}""", "rules[0].chance"),
                ("""{"name": "x", "rules": [{"name": "o", "from": "", "to": "b"}]}""", "rules[0].from"),
                ("""{"name": "x", "rules": [], "colour": 1}""", "colour"),
            })
            {
                File.WriteAllText(file, json);
                try
                {
                    RuleTextHabits.Read(file);
                    Check(false, $"{key}: not refused");
                }
                catch (InvalidDataException e)
                {
                    Check(e.Message.StartsWith(file + ":", StringComparison.Ordinal) && e.Message.Contains($"'{key}'", StringComparison.Ordinal), e.Message);
                }
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        foreach (string bad in (string[])["habits=nope", "label=both", "habit.final-ya=2", "habit-chance=-1"])
        {
            try
            {
                SyntheticPages.Style(options.Style, bad);
                Check(false, $"{bad}: not refused");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains(bad.Split('=')[0], StringComparison.Ordinal), $"{bad}: {e.Message}");
            }
        }
    }
}
