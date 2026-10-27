// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Text;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// The Generate lab: text drawn as scanned pages by the library's synthetic pages, as idrak data synth draws them.
internal static partial class MauiTests
{
    private const string GenerateText = "بسم الله الرحمن الرحيم\nHello from Idrak, 2026";

    private sealed class FakePictures : IPictureSaver
    {
        public List<(byte[] Bytes, string Name, string Type, string Album)> Saved { get; } = [];

        public bool CanSave => true;

        public string Unavailable => "";

        public Task<string> SaveAsync(byte[] bytes, string fileName, string mediaType, string album, CancellationToken cancellationToken = default)
        {
            Saved.Add((bytes, fileName, mediaType, album));
            return Task.FromResult($"Pictures/{album}/{fileName}");
        }
    }

    // A lab over the repository's test fonts as the phone's, its added fonts and data sets in a folder of its own.
    private static GenerateViewModel GenerateLab(string name, out string storage, IPictureSaver? pictures = null, int count = 2)
    {
        storage = Temp(name);
        // The phone's fonts: the repository's Arabic test fonts (no Latin) and the app's mono font (Latin), so a page of both needs two.
        string phone = Path.Combine(storage, "phone-fonts");
        Directory.CreateDirectory(phone);
        foreach (string font in Directory.EnumerateFiles(Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "fonts", "arabic"), "*.ttf")
                     .Append(Path.Combine(RepositoryRoot(), "apps", "Idrak.Mobile", "Resources", "Fonts", "JetBrainsMono-Regular.ttf")))
        {
            File.Copy(font, Path.Combine(phone, Path.GetFileName(font)), overwrite: true);
        }

        var defaults = new GenerateLabOptions { SystemFonts = phone, Count = count };
        return new GenerateViewModel(new FakeShare(), new MemorySettingsStore(), storage, pictures, defaults: defaults) { Text = GenerateText };
    }

    private static void GenerateMakesPages()
    {
        var pictures = new FakePictures();
        var lab = GenerateLab("generate-pages", out string storage, pictures);
        lab.ListFontsAsync().GetAwaiter().GetResult();
        Check(lab.AllFonts.Count >= 3 && lab.AllFonts.All(f => !f.IsAdded) && lab.ChosenFonts.Count == 0, $"{lab.AllFonts.Count} fonts listed, none read");
        Check(lab.Effects.Select(e => e.Name).SequenceEqual(ScanProfiles.Names) && lab.Layouts.Count(l => l.IsSelected) == 2 && lab.Combination == "none",
            $"effects {string.Join(", ", lab.Effects.Select(e => e.Name))}; layouts {string.Join(", ", lab.Layouts.Where(l => l.IsSelected).Select(l => l.Name))}");

        // Effects stack into one look, in the order tapped; a second tap takes one out.
        lab.Effects.Single(e => e.Name == "marked").ToggleCommand.Execute(null);
        lab.Effects.Single(e => e.Name == "aged").ToggleCommand.Execute(null);
        lab.Effects.Single(e => e.Name == "photo").ToggleCommand.Execute(null);
        lab.ToggleEffect("photo");
        Check(lab.Combination == "marked+aged" && lab.Effects.Where(e => e.IsSelected).Select(e => e.Name).Order().SequenceEqual(["aged", "marked"]) && lab.LookText.Contains("marked+aged", StringComparison.Ordinal),
            lab.Combination);
        lab.SeedText = "7";
        var clock = Stopwatch.StartNew();
        Check(lab.GenerateAsync().GetAwaiter().GetResult(), lab.Error);
        Console.WriteLine($"    2 pages ({lab.Combination}) in {clock.Elapsed.TotalMilliseconds:F0} ms, fonts picked: {string.Join(", ", lab.ChosenFonts.Select(Path.GetFileNameWithoutExtension))}");
        Check(lab.Variants.Count == 2 && lab.Variants.All(v => v.Page.Profile == "marked+aged" && v.Page.Image.Width > 0) && ReferenceEquals(lab.Selected, lab.Variants[0])
              && lab.PreviewSource is not null && lab.Variants[0].IsSelected && lab.ChosenFonts.Count > 0 && !lab.IsBusy && lab.Work is null,
            $"{lab.Variants.Count} pages; {string.Join(", ", lab.Variants.Select(v => v.Page.Profile))}");
        Check(lab.Variants[0].Page.Text.Contains("Hello", StringComparison.Ordinal) && lab.Variants[0].Page.Seed != lab.Variants[1].Page.Seed, "the text drawn, each page its own");

        // The library's own page for the same text and options (rule 71), and the same pages for the same seed.
        var fonts = lab.ChosenFonts.Order(StringComparer.Ordinal).SelectMany(p => FontFormats.Read(p)).ToList();
        var options = lab.PageOptions(fonts);
        Check(options.Scans.SequenceEqual(["marked+aged"]) && options.Seed == 7 && options.Variants == 2 && options.Layouts.SequenceEqual(["letter", "plain"]), string.Join(", ", options.Scans));
        var library = SyntheticPages.Make(GenerateText, options, 0, 1, "the text");
        Check(library.Text == lab.Variants[1].Page.Text && library.Scan == lab.Variants[1].Page.Scan && library.Description == lab.Variants[1].Page.Description,
            $"the library's page: {library.Scan} vs {lab.Variants[1].Page.Scan}");
        string firstScan = lab.Variants[0].Page.Scan;
        Check(lab.GenerateAsync().GetAwaiter().GetResult() && lab.Variants[0].Page.Scan == firstScan, "the same seed, the same pages");

        // Before and after: the same drawing without its scan.
        var shown = lab.Variants[0];
        Check(lab.ShowCleanAsync(true).GetAwaiter().GetResult() && lab.ShowClean && shown.Clean is { Profile: "none", Scan: "" } clean && clean.Text == shown.Page.Text
              && clean.Layout == shown.Page.Layout && clean.Description == shown.Page.Description && lab.CompareText == "Show scanned", "the clean page");
        Check(lab.ShowCleanAsync(false).GetAwaiter().GetResult() && !lab.ShowClean, "scanned again");

        // A random look, every look, a clean page.
        string random = lab.RandomEffects(seed: 5);
        var parts = random.Split('+');
        Check(parts.Length is >= 1 and <= 3 && parts.All(p => ScanProfiles.Names.Contains(p)) && lab.Effects.Count(e => e.IsSelected) == parts.Length, random);
        lab.EveryLookCommand.Execute(null);
        Check(lab.EveryLook && lab.PageOptions(fonts).Scans.SequenceEqual(new SyntheticPageOptions { Style = new PageStyle { Fonts = [] } }.Scans), "every look: the CLI's default");
        lab.ToggleEffect("fax");
        Check(!lab.EveryLook && lab.Combination.EndsWith("fax", StringComparison.Ordinal), "a chip goes back to the combination");
        lab.ClearEffectsCommand.Execute(null);
        Check(lab.Combination == "none" && lab.PageOptions(fonts).Scans.SequenceEqual(["none"]), "clean");

        // Saved to Pictures (JPEG at the library's quality), shared.
        Check(lab.SaveToPicturesAsync().GetAwaiter().GetResult() is { } where && where.StartsWith("Pictures/Idrak/page-7-1", StringComparison.Ordinal)
              && pictures.Saved.Single() is { Type: "image/jpeg" } saved && saved.Bytes[0] == 0xFF && saved.Bytes[1] == 0xD8, lab.Error);
        Check(lab.ShareAsync().GetAwaiter().GetResult(), lab.Error);
        var none = GenerateLab("generate-nopictures", out _);
        Check(none.SaveToPicturesAsync().GetAwaiter().GetResult() is null && none.Error.Contains("Pictures", StringComparison.Ordinal) && !none.CanSaveToPictures, none.Error);

        // A data set, as the CLI writes one.
        lab.Combination = "office";
        Check(lab.SaveDatasetAsync().GetAwaiter().GetResult() is { } data && File.ReadAllLines(data).Length == 2
              && Directory.EnumerateFiles(Path.Combine(Path.GetDirectoryName(data)!, "pages")).Count() == 2 && File.Exists(Path.ChangeExtension(data, ".manifest.jsonl"))
              && data.StartsWith(Path.Combine(storage, "datasets"), StringComparison.Ordinal), lab.Error + lab.Status);
    }

    private static void GenerateCancels()
    {
        var lab = GenerateLab("generate-cancel", out _, count: 40);
        lab.Combination = "clean";
        var run = lab.GenerateAsync();
        var clock = Stopwatch.StartNew();
        while (lab.Variants.Count == 0 && !run.IsCompleted && clock.Elapsed < TimeSpan.FromSeconds(60))
        {
            Thread.Sleep(5);
        }

        lab.CancelCommand.Execute(null);
        bool done = run.GetAwaiter().GetResult();
        Check(!done && lab.Variants.Count is > 0 and < 40 && lab.Status.StartsWith("Stopped", StringComparison.Ordinal) && !lab.IsBusy, $"{lab.Variants.Count} pages, '{lab.Status}' {lab.Error}");
        lab.Text = " ";
        Check(!lab.GenerateAsync().GetAwaiter().GetResult() && lab.Error.Contains("text", StringComparison.Ordinal) && !lab.CanGenerate, "no text: nothing made");
    }

    private static void GenerateFontsAndSettings()
    {
        string storage = Temp("generate-fonts");
        var lab = new GenerateViewModel(new FakeShare(), new MemorySettingsStore(), storage, defaults: new GenerateLabOptions { SystemFonts = Path.Combine(storage, "none") });
        byte[] cairo = File.ReadAllBytes(Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "fonts", "arabic", "cairo.ttf"));
        Check(lab.AddFontsAsync([("cairo.ttf", cairo), ("broken.ttf", [1, 2, 3, 4]), ("notes.txt", [65])]).GetAwaiter().GetResult() == 1, lab.Error);
        Check(File.Exists(Path.Combine(storage, "fonts", "cairo.ttf")) && lab.AllFonts.Single() is { IsAdded: true, IsChosen: true, Name: "cairo" }
              && lab.Error.Contains("broken.ttf", StringComparison.Ordinal) && lab.Error.Contains("notes.txt", StringComparison.Ordinal), lab.Error);
        lab.FontQuery = "zzz";
        Check(lab.Fonts.Count == 0 && lab.AllFonts.Count == 1, "the search hides it");
        lab.FontQuery = "CAI";
        Check(lab.Fonts.Count == 1, "found by its name");
        lab.ClearFontsCommand.Execute(null);
        Check(lab.ChosenFonts.Count == 0 && !lab.AllFonts[0].IsChosen && lab.FontsText.StartsWith("No font chosen", StringComparison.Ordinal), lab.FontsText);
        lab.Text = "";
        Check(lab.PickFontsForTextAsync().GetAwaiter().GetResult() == 0 && lab.Error.Contains("text", StringComparison.Ordinal), lab.Error);

        // The settings: every option the CLI takes, checked, kept and read back.
        Check(GenerateSettings.Definitions.Select(s => s.Key).Distinct().Count() == GenerateSettings.Definitions.Count
              && new[] { "count", "grey", "page", "parts", "image_format", "quality", "prompt", "answer_key", "zip", "workers" }.All(k => GenerateSettings.Definitions.Any(s => s.Key == k)),
            string.Join(", ", GenerateSettings.Definitions.Select(s => s.Key)));
        foreach (var (key, text) in new[] { ("count", "0"), ("grey", "1.5"), ("page", "colour=yes"), ("quality", "101"), ("parts", "relative.json"), ("image_format", "gif9") })
        {
            var entry = lab.Settings.Single(s => s.Key == key);
            entry.Text = text;
            Check(!lab.ApplySettings() && entry.HasError, $"{key}={text} was taken");
            entry.Text = "";
        }

        lab.Settings.Single(s => s.Key == "count").Text = "5";
        lab.Settings.Single(s => s.Key == "page").Text = "size=20..28,stamp=0.5";
        Check(lab.ApplySettings() && lab.Options.Count == 5 && lab.Options.Page == "size=20..28,stamp=0.5", lab.SettingsMessage);
        var store = new MemorySettingsStore();
        GenerateSettings.Save(store, lab.Options);
        var read = GenerateSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Count == 5 && read.Page == lab.Options.Page, "kept and read back");

        // The feature: in the Vision group and tab, after Read text and Image; its view binds in both themes.
        var feature = MobileFeatures.Get(MobileFeatures.Generate);
        Check(feature is { Title: "Generate", Tagline: "Text to scanned pages", Group: FeatureGroup.Vision, Icon: "pen" }
              && AppSections.Get(AppSections.Vision).Features.SequenceEqual([MobileFeatures.Ocr, MobileFeatures.Image, MobileFeatures.Generate]), feature.Title);
        using var services = AppServices();
        foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
        {
            var view = FeatureViews.Create(services, MobileFeatures.Generate);
            Themed(view, theme);
            Check(view is GenerateView && CheckLabels(view, theme, "generate") > 10, theme.ToString());
            ((GenerateView)view).Apply(true);
            ((GenerateView)view).Apply(false);
            Check(CheckLabels(Themed(new FontsView((GenerateViewModel)view.BindingContext), theme), theme, "fonts") > 1, "the fonts page");
        }
    }
}
