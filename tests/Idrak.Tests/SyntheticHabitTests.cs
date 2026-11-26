// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Text;
using Idrak.Imaging;
using Idrak.Text;

// Handwriting habits on synthetic pages: an option, off by default (every page as before); at a chance, the body is
// written with the habits and the label says what is drawn, or keeps the source's spelling.
internal abstract partial class DesktopTests
{
    private static void SyntheticHabits(Device device)
    {
        // Pages: no habits (the default) and habits=none draw the same bytes and labels.
        string source = "أحمد في المدرسة إلى آخر ظلٍّ يوم";
        var amiri = new OpenTypeFormat().Read(File.ReadAllBytes(FontData("arabic/amiri.ttf")), "amiri").Single();
        var options = new SyntheticPageOptions { Style = new PageStyle { Fonts = [amiri], Width = 620, Height = 877 }, Layouts = ["plain"], Scans = ["none"], Seed = 5 };
        var plain = SyntheticPages.Make(source, options);
        var none = SyntheticPages.Make(source, options with { Style = SyntheticPages.Style(options.Style, "habits=none") });
        Check(none.Image.Pixels.AsSpan().SequenceEqual(plain.Image.Pixels) && none.Text == plain.Text && none.Answer == plain.Answer, "habits=none: the page as before");

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

        foreach (string bad in (string[])["label=both", "habit.final-ya=2", "habit-chance=-1"])
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
