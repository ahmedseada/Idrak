// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Media;

namespace Idrak.Maui.Tests;

// The Image lab (plan 34, M6) on the CPU: photos made here with the library's encoders (and the repository's EXIF
// fixture), a codec and a transform registered over the library's to count what runs, a media picker standing in for
// the camera and the gallery.
internal static partial class MauiTests
{
    // A colour page of `width` x `height`: bands and a gradient, so every transform changes something.
    private static ImageData TestPicture(int width, int height)
    {
        var pixels = new float[3 * height * width];
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    pixels[(c * height + y) * width + x] = ((x * (c + 1) + y * 3 + (y / 7 % 2) * 90) % 256) / 255f;
                }
            }
        }

        return new ImageData(pixels, 3, height, width);
    }

    private static ChatImage Png(ImageData image) => ChatImage.FromBytes(ImageEncoders.Get("png").Encode(image), "image/png");

    // The pixels as bytes, for exact comparisons (8-bit images).
    private static byte[] Levels(ImageData image) => [.. image.Pixels.Select(v => (byte)Math.Clamp((int)MathF.Round(v * 255f), 0, 255))];

    // A codec over the library's PNG and JPEG that counts the files it decodes.
    private sealed class CountingCodec(IImageCodec library) : IImageCodec
    {
        private int _decodes;

        public int Decodes => Volatile.Read(ref _decodes);

        public string Name => library.Name;

        public IReadOnlyCollection<string> Extensions => library.Extensions;

        public ImageInfo? ReadInfo(ReadOnlySpan<byte> header) => library.ReadInfo(header);

        public ImageData Decode(ReadOnlySpan<byte> file)
        {
            Interlocked.Increment(ref _decodes);
            return library.Decode(file);
        }
    }

    // A transform that returns its image and counts its runs (a step whose first run in the process is cold).
    private sealed class CountingTransform(string name) : IImageTransform
    {
        private int _runs;

        public int Runs => Volatile.Read(ref _runs);

        public string Name => name;

        public string Summary => "counts its runs (a test's)";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public void Check(ImageTransformStep step) => step.ThrowIfValue();

        public ImageData Apply(ImageData image, ImageTransformStep step)
        {
            Interlocked.Increment(ref _runs);
            return image;
        }
    }

    // A media picker: the camera and the gallery give `result` (null: the user cancelled), counting the calls.
    private sealed class FakePicker(bool camera = true, FileResult? result = null) : IMediaPicker
    {
        public int Captures { get; private set; }

        public int Picks { get; private set; }

        public bool IsCaptureSupported => camera;

        public Task<FileResult?> PickPhotoAsync(MediaPickerOptions? options = null)
        {
            Picks++;
            return Task.FromResult(result);
        }

        public Task<List<FileResult>> PickPhotosAsync(MediaPickerOptions? options = null)
        {
            Picks++;
            return Task.FromResult(result is null ? new List<FileResult>() : [result]);
        }

        public Task<FileResult?> CapturePhotoAsync(MediaPickerOptions? options = null)
        {
            Captures++;
            return Task.FromResult(result);
        }

        public Task<FileResult?> PickVideoAsync(MediaPickerOptions? options = null) => throw new NotSupportedException();

        public Task<List<FileResult>> PickVideosAsync(MediaPickerOptions? options = null) => throw new NotSupportedException();

        public Task<FileResult?> CaptureVideoAsync(MediaPickerOptions? options = null) => throw new NotSupportedException();
    }

    // Runs `test` with counting codecs over the library's PNG and JPEG, and the library's back after.
    private static void WithCountingCodecs(Action<CountingCodec, CountingCodec> test)
    {
        var png = new CountingCodec(ImageCodecs.Default("png")!);
        var jpeg = new CountingCodec(ImageCodecs.Default("jpeg")!);
        ImageCodecs.Register(png);
        ImageCodecs.Register(jpeg);
        try
        {
            test(png, jpeg);
        }
        finally
        {
            ImageCodecs.Unregister("png");
            ImageCodecs.Unregister("jpeg");
        }
    }

    private static ImageViewModel ImageLab(MemorySettingsStore? settings = null, ImageLabOptions? defaults = null, FakePermissions? permissions = null, FakeShare? share = null,
        FakePicker? picker = null) =>
        new(permissions ?? new FakePermissions(true), share ?? new FakeShare(), settings ?? new MemorySettingsStore(), defaults, picker ?? new FakePicker());

    private static void ImageOpens()
    {
        WithCountingCodecs((png, jpeg) =>
        {
            var picture = TestPicture(97, 61);
            var lab = ImageLab(defaults: new ImageLabOptions { PreviewPixels = 40 });

            // A PNG made here: decoded once, its pixels the library's own, its picture shrunk to the setting's size and made once.
            var file = Png(picture);
            var expected = Levels(ChatImageDecoder.Decode(file.Data.Span));     // the test's own decode
            int before = png.Decodes;
            Check(lab.OpenAsync(file, "page.png").GetAwaiter().GetResult(), $"not opened: {lab.Error}");
            Check(png.Decodes == before + 1 && lab.Photo is { Format: "png", Orientation: 1 } photo && photo.Pixels.Width == 97 && photo.Pixels.Height == 61
                  && Levels(photo.Pixels).SequenceEqual(expected), $"{png.Decodes - before} decodes; {lab.PhotoText}");
            Check(lab.HasPhoto && lab.OriginalPreview is { Width: 40 } preview && preview.Height == (int)Math.Round(61 * 40 / 97.0)
                  && ImageCodecs.ReadInfo(preview.Png) is { Format: "png", Width: 40 } && lab.OriginalSource is StreamImageSource,
                $"the picture: {lab.OriginalPreview?.Width} x {lab.OriginalPreview?.Height}");
            Check(lab.PhotoText.Contains("page.png: png, 97 x 61, colour", StringComparison.Ordinal) && lab.PhotoText.Contains("read in", StringComparison.Ordinal), lab.PhotoText);

            // No size set and no screen to measure (no platform here): the picture keeps the image's size.
            var whole = ImageLab();
            Check(whole.OpenAsync(file, "page.png").GetAwaiter().GetResult() && whole.OriginalPreview is { Width: 97, Height: 61 }, "the picture at the image's own size");

            // A JPEG made here by the library's encoder.
            int jpegs = jpeg.Decodes;
            var made = ChatImage.FromBytes(JpegEncoder.Encode(picture, 90), "image/jpeg");
            Check(lab.OpenAsync(made, "page.jpg").GetAwaiter().GetResult() && jpeg.Decodes == jpegs + 1 && lab.Photo is { Format: "jpeg" }, $"the JPEG: {lab.Error}");

            // A phone's photo with EXIF orientation 6: turned upright, its sides swapped.
            var exif = ChatImage.FromFile(Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vlm", "exif", "exif-6.jpg"));
            var stored = ImageCodecs.ReadInfo(exif.Data.Span)!.Value;
            Check(lab.OpenAsync(exif, "IMG_0001.jpg").GetAwaiter().GetResult() && lab.Photo is { Orientation: 6 } turned
                  && turned.Pixels.Width == stored.Height && turned.Pixels.Height == stored.Width && lab.PhotoText.Contains("turned upright (EXIF 6)", StringComparison.Ordinal),
                $"orientation: {lab.PhotoText}");
            Check(lab.Photo!.Pixels.TryGetPlanes(out _, out _), "the upright photo stayed 8-bit");
        });

        // HEIC (a phone's own format) and an unknown file are refused, naming what the library reads.
        var lab2 = ImageLab();
        byte[] heic = [0, 0, 0, 24, (byte)'f', (byte)'t', (byte)'y', (byte)'p', (byte)'h', (byte)'e', (byte)'i', (byte)'c', 0, 0, 0, 0, (byte)'m', (byte)'i', (byte)'f', (byte)'1'];
        Check(!lab2.OpenAsync(ChatImage.FromBytes(heic), "IMG_0002.HEIC").GetAwaiter().GetResult() && lab2.Error.Contains("HEIC", StringComparison.Ordinal)
              && lab2.Error.Contains("jpeg", StringComparison.Ordinal) && lab2.Error.Contains("png", StringComparison.Ordinal) && !lab2.HasPhoto, lab2.Error);
        Check(Photos.HeifBrand(heic) == "heic" && Photos.HeifBrand(new byte[20]) is null, "the HEIF brand");
        Check(!lab2.OpenAsync(ChatImage.FromBytes("not an image"u8), "notes.txt").GetAwaiter().GetResult() && lab2.Error.Contains("not an image the library's codecs read", StringComparison.Ordinal)
              && lab2.Formats.Contains("jpeg", StringComparison.Ordinal), lab2.Error);
    }

    private static void ImageChips()
    {
        var lab = ImageLab();
        Check(lab.Transforms.Select(c => c.Name).SequenceEqual(ImageTransforms.Names) && lab.Transforms.Count >= 19, $"{lab.Transforms.Count} chips");
        foreach (var chip in lab.Transforms)
        {
            var transform = ImageTransforms.Get(chip.Name);
            Check(chip.Label == transform.Label && chip.Summary == transform.Summary && chip.HasParameters == transform.Parameters.Count > 0, $"{chip.Name}: {chip.Label}, '{chip.Parameters}'");
            _ = ImageTransformPipeline.Parse(chip.Step);                        // every chip's step parses
        }

        var contrast = lab.Transforms.Single(c => c.Name == "contrast");
        var jpeg = lab.Transforms.Single(c => c.Name == "jpeg");
        Check(jpeg.Parameters.Contains("value: whole number 1 to 100", StringComparison.Ordinal) && jpeg.Parameters.Contains("subsampling?: 4:2:0", StringComparison.Ordinal)
              && jpeg.Step is "jpeg=50" or "jpeg=51", $"jpeg: '{jpeg.Parameters}', step {jpeg.Step}");
        lab.Pipeline = "";
        lab.Transforms.Single(c => c.Name == "grayscale").AddCommand.Execute(null);
        contrast.AddCommand.Execute(null);
        jpeg.AddCommand.Execute(null);
        Check(lab.Pipeline == $"grayscale,{contrast.Step},{jpeg.Step}", $"pipeline '{lab.Pipeline}'");
        lab.AddStep("max_width=64,resample=bicubic");
        Check(ImageTransformPipeline.Parse(lab.Pipeline).Steps.Count == 4, $"pipeline '{lab.Pipeline}'");

        // An unknown step is refused at Apply, naming the registered transforms.
        lab.OpenAsync(Png(TestPicture(20, 10)), "small.png").GetAwaiter().GetResult();
        lab.Pipeline = "grayscale,emboss";
        Check(!lab.ApplyAsync().GetAwaiter().GetResult() && lab.Error.Contains("Unknown image transform 'emboss'", StringComparison.Ordinal)
              && lab.Error.Contains("contrast", StringComparison.Ordinal), lab.Error);

        // A scan profile draws its steps: the same seed, the same steps; none picked is said.
        Check(lab.Profiles.Count == ScanProfiles.Names.Count && lab.Profiles.Contains("office"), $"profiles: {string.Join(", ", lab.Profiles)}");
        lab.SelectedProfile = null;
        Check(!lab.DrawProfile() && lab.Error.Contains("Pick a scan profile", StringComparison.Ordinal), lab.Error);
        lab.SelectedProfile = "office";
        Check(lab.ProfileText == ScanProfiles.Get("office").Summary, $"profile text '{lab.ProfileText}'");
        lab.Settings.Single(e => e.Key == "seed").Text = "1234";
        Check(lab.ApplySettings() && lab.DrawProfile(), $"{lab.SettingsMessage} {lab.Error}");
        string drawn = lab.Pipeline;
        Check(drawn == ScanProfiles.Get("office").Sample(1234).ToString() && lab.Status.Contains("seed 1234", StringComparison.Ordinal), $"'{drawn}', '{lab.Status}'");
        lab.Pipeline = "";
        lab.DrawProfile();
        Check(lab.Pipeline == drawn && lab.SelectedProfile == "office", "the same seed drew other steps");
    }

    private static void ImageApplied()
    {
        string cache = Temp("image-cache");
        var probe = new CountingTransform("maui-image-probe");
        ImageTransforms.Register(probe);
        ComputeResources.CacheFolder = cache;
        try
        {
            WithCountingCodecs((png, _) =>
            {
                var share = new FakeShare();
                var settings = new MemorySettingsStore();
                var lab = ImageLab(settings, new ImageLabOptions { PreviewPixels = 32 }, share: share);
                var file = Png(TestPicture(90, 50));
                var decoded = ChatImageDecoder.Decode(file.Data.Span);         // the test's own decode, for the expected result
                int before = png.Decodes;
                Check(lab.OpenAsync(file, "scan.png").GetAwaiter().GetResult(), lab.Error);
                Check(!lab.HasResult && lab.SaveAsync().GetAwaiter().GetResult() is { } plain && File.Exists(plain), $"the photo itself is saved before any run: {lab.Error}");

                // The pipeline on the decoded photo: the library's own result, step by step with times; cold, then warm.
                const string text = "grayscale,maui-image-probe,max_width=45,contrast=1.5";
                lab.Pipeline = text;
                Check(lab.ApplyAsync().GetAwaiter().GetResult(), lab.Error);
                var expected = ImageTransformPipeline.Parse(text).Apply(decoded);
                Check(lab.Result is { Channels: 1, Width: 45 } result && result.Height == expected.Height && result.Pixels.AsSpan().SequenceEqual(expected.Pixels),
                    $"result {lab.Result?.Channels} x {lab.Result?.Height} x {lab.Result?.Width}: not the library's");
                Check(lab.Steps.Count == 4 && lab.Steps.Select(s => s.Step).SequenceEqual(ImageTransformPipeline.Parse(text).Steps.Select(s => s.ToString()))
                      && lab.Steps.Single(s => s.Step == "maui-image-probe").Cold && lab.StepsText.Split('\n').Length == 4, lab.StepsText);
                Check(lab.HasCold && lab.ColdText.Contains("maui-image-probe", StringComparison.Ordinal) && !lab.HasWarm, $"'{lab.ColdText}', '{lab.WarmText}'");
                Check(lab.Steps.Single(s => s.Step == "maui-image-probe").Text.EndsWith("(cold)", StringComparison.Ordinal), "the step's cold mark");
                Check(lab.ApplyAsync().GetAwaiter().GetResult() && lab.HasWarm && lab.WarmText.Contains("mean over 1 run", StringComparison.Ordinal)
                      && !lab.Steps.Any(s => s.Cold) && probe.Runs == 3, $"'{lab.WarmText}', {probe.Runs} probe runs (two the lab's, one the test's own expected result)");
                Check(lab.ApplyAsync().GetAwaiter().GetResult() && lab.WarmText.Contains("mean over 2 runs", StringComparison.Ordinal), lab.WarmText);
                Check(png.Decodes == before + 1, $"the photo was decoded {png.Decodes - before} times for three runs");
                Check(lab.ResultPreview is { Width: 32 } && lab.HasResult && lab.ResultText.StartsWith("45 x ", StringComparison.Ordinal), $"{lab.ResultText}");
                Check(settings.Get(ImageSettings.Prefix + "pipeline") == ImageTransformPipeline.Parse(text).ToString(), "the pipeline kept in the settings");

                // Saved as PNG (the default): the result's own pixels at its size; saved once, again only when it changed.
                string? saved = lab.SaveAsync().GetAwaiter().GetResult();
                Check(saved is not null && saved.StartsWith(Path.Combine(cache, "images"), StringComparison.Ordinal) && saved.EndsWith(".png", StringComparison.Ordinal), $"saved {saved}: {lab.Error}");
                var back = ImageCodecs.Decode(saved!);
                Check(back.Width == 45 && back.Channels == 1 && Levels(back).SequenceEqual(Levels(lab.Result!)), "the saved PNG is not the result");
                Check(lab.SaveAsync().GetAwaiter().GetResult() == saved && lab.Status.StartsWith("Saved already", StringComparison.Ordinal), lab.Status);

                // As JPEG at the settings' quality: a smaller file at a lower quality, the result's size.
                lab.Settings.Single(e => e.Key == "save_format").SelectedChoice = "jpeg";
                lab.Settings.Single(e => e.Key == "jpeg_quality").Text = "95";
                Check(lab.ApplySettings(), lab.SettingsMessage);
                string high = lab.SaveAsync().GetAwaiter().GetResult()!;
                Thread.Sleep(1100);                                             // a file name a second later
                lab.Settings.Single(e => e.Key == "jpeg_quality").Text = "20";
                Check(lab.ApplySettings(), lab.SettingsMessage);
                lab.ApplyAsync().GetAwaiter().GetResult();
                string low = lab.SaveAsync().GetAwaiter().GetResult()!;
                Check(high.EndsWith(".jpg", StringComparison.Ordinal) && low != high && new FileInfo(low).Length < new FileInfo(high).Length
                      && ImageCodecs.ReadInfo(low) is { Format: "jpeg", Width: 45 }, $"{high} ({new FileInfo(high).Length} bytes), {low} ({new FileInfo(low).Length} bytes)");
                Check(lab.ShareAsync().GetAwaiter().GetResult() && share.Shared == low, $"shared {share.Shared}");
            });
        }
        finally
        {
            ImageTransforms.Unregister(probe.Name);
            ComputeResources.CacheFolder = null;
        }
    }

    private static void ImageSettingsMapped()
    {
        // Nothing set: png, the library's JPEG quality (JpegEncoder's own), its resampling, the screen's size.
        var empty = new ImageLabOptions();
        var picture = TestPicture(24, 16);
        Check(empty.Format == "png" && empty.Extension == ".png" && empty.MediaType == "image/png" && empty.Resampling is null && empty.ToPipeline().IsEmpty,
            "the defaults");
        Check(empty.Encode(picture).AsSpan().SequenceEqual(ImageEncoders.Get("png").Encode(picture)), "png: the library's encoder");
        Check(new ImageLabOptions { SaveFormat = "jpeg" }.Encode(picture).AsSpan().SequenceEqual(JpegEncoder.Encode(picture)), "jpeg without a quality: the library's");
        Check(ImageSettings.Formats.Take(2).SequenceEqual(["png", "jpeg"]) && ImageSettings.Formats.Contains("netpbm"), string.Join(", ", ImageSettings.Formats));

        var texts = new Dictionary<string, string>
        {
            ["profile"] = "damaged+marked", ["seed"] = "77", ["save_format"] = "JPG", ["jpeg_quality"] = "88", ["preview_px"] = "720", ["preview_resample"] = "bicubic",
        };
        Check(texts.Keys.Order().SequenceEqual(ImageSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new ImageLabOptions { Pipeline = "grayscale" };
        foreach (var setting in ImageSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        Check(set is { Profile: "damaged+marked", Seed: 77, SaveFormat: "jpeg", JpegQuality: 88, PreviewPixels: 720, Resampling: ImageResampling.Bicubic } && set.Extension == ".jpg",
            $"mapped: {set.Profile}, {set.Seed}, {set.SaveFormat}, {set.JpegQuality}, {set.PreviewPixels}, {set.Resampling}");
        Check(set.Encode(picture).AsSpan().SequenceEqual(JpegEncoder.Encode(picture, 88)), "jpeg at the setting's quality");

        var store = new MemorySettingsStore();
        ImageSettings.Save(store, set);
        Check(store.Keys.All(k => k.StartsWith(ImageSettings.Prefix, StringComparison.Ordinal)), string.Join(", ", store.Keys));
        var read = ImageSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Pipeline == "grayscale" && ImageSettings.Definitions.All(s => s.Get(read) == s.Get(set)), string.Join("; ", skipped));

        // The app's defaults under the saved values; a saved value that no longer reads is skipped, named.
        var app = new ImageLabOptions { JpegQuality = 60, PreviewPixels = 300 };
        store.Set(ImageSettings.Prefix + "preview_px", null);
        store.Set(ImageSettings.Prefix + "seed", "lots");
        var over = ImageSettings.Load(store, out skipped, app);
        Check(over.JpegQuality == 88 && over.PreviewPixels == 300 && app.JpegQuality == 60 && skipped.Count == 1 && skipped[0].StartsWith("Profile seed", StringComparison.Ordinal),
            $"{over.JpegQuality}, {over.PreviewPixels}; skipped {string.Join("; ", skipped)}");

        foreach (var (key, value, says) in new[]
                 {
                     ("profile", "nosuchprofile", "nosuchprofile"), ("save_format", "gif", "png, jpeg"), ("jpeg_quality", "0", "1 to 100"), ("jpeg_quality", "101", "1 to 100"),
                     ("preview_px", "4", "16 or more"), ("preview_resample", "nearest", "lanczos"), ("seed", "-1", "0 or more"),
                 })
        {
            try
            {
                ImageSettings.Definitions.Single(s => s.Key == key).Set(new ImageLabOptions(), value);
                Check(false, $"{key}={value} was taken");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{key}={value}: {e.Message}");
            }
        }

        // The page: a refused value keeps nothing and says why.
        var lab = ImageLab(store);
        lab.Settings.Single(e => e.Key == "jpeg_quality").Text = "500";
        Check(!lab.ApplySettings() && lab.Settings.Single(e => e.Key == "jpeg_quality").HasError && lab.Options.JpegQuality == 88, lab.SettingsMessage);
    }

    private static void ImagePageOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var page = MobileFeatures.Open(MobileFeatures.Image, services);
        Check(page is ImagePage { BindingContext: ImageViewModel } && page.Title == "Image", $"page {page.GetType().Name}, title {page.Title}");
        Check(MobileFeatures.Origin(MobileFeatures.Image) == Overrides.Library
              && MobileFeatures.Get(MobileFeatures.Image) is ImageFeature { Order: 70, Needs: PhoneNeeds.Camera } feature && ((IMobileFeature)feature).State == FeatureState.Ready
              && MobileFeatures.Place(feature) == 70 && feature.Packages.SequenceEqual(["Idrak.Imaging"]), "the library's image feature");

        // The camera asked first: refused, nothing is taken and the page says how to allow it.
        var refused = new FakePermissions(false);
        var picker = new FakePicker();
        var lab = ImageLab(permissions: refused, picker: picker);
        Check(!lab.TakePhotoAsync(camera: true).GetAwaiter().GetResult() && lab.IsPermissionRefused && picker.Captures == 0 && refused.Requests == 1
              && lab.Error.Contains("Allow it in the app's settings", StringComparison.Ordinal), lab.Error);
        lab.OpenPermissionSettingsCommand.Execute(null);
        Check(refused.SettingsOpened == 1, "the app's settings did not open");

        // Allowed: the camera opens; a photo not taken (cancelled) opens nothing. No camera at all: said.
        var allowed = new FakePicker();
        var lab2 = ImageLab(picker: allowed);
        Check(!lab2.TakePhotoAsync(camera: true).GetAwaiter().GetResult() && allowed.Captures == 1 && !lab2.HasError && !lab2.HasPhoto, lab2.Error);
        Check(!lab2.TakePhotoAsync(camera: false).GetAwaiter().GetResult() && allowed.Picks == 1, "the gallery");
        var none = ImageLab(picker: new FakePicker(camera: false));
        Check(!none.TakePhotoAsync(camera: true).GetAwaiter().GetResult() && none.Error.Contains("no camera", StringComparison.Ordinal) && !none.IsPermissionRefused, none.Error);
    }
}
