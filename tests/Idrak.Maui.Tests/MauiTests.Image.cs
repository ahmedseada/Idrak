// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Media;

namespace Idrak.Maui.Tests;

// The Image lab (plan 34, M6) on the CPU: its settings, and a media picker standing in for the camera and the gallery.
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

    private static ImageViewModel ImageLab(MemorySettingsStore? settings = null, ImageLabOptions? defaults = null, FakePermissions? permissions = null, FakeShare? share = null,
        FakePicker? picker = null) =>
        new(permissions ?? new FakePermissions(true), share ?? new FakeShare(), settings ?? new MemorySettingsStore(), defaults, picker ?? new FakePicker());

    private static void ImageSettingsMapped()
    {
        // Nothing set: png, the library's JPEG quality (JpegEncoder's own), its resampling, the screen's size.
        var empty = new ImageLabOptions();
        var picture = TestPicture(24, 16);
        Check(empty.Format == "png" && empty.MediaType == "image/png" && empty.Resampling is null, "the defaults");
        Check(new ImageLabOptions { SaveFormat = "jpeg" }.Encode(picture).AsSpan().SequenceEqual(JpegEncoder.Encode(picture)), "jpeg without a quality: the library's");
        Check(ImageSettings.Formats.Take(2).SequenceEqual(["png", "jpeg"]), string.Join(", ", ImageSettings.Formats));

        var texts = new Dictionary<string, string>
        {
            ["profile"] = "", ["seed"] = "77", ["save_format"] = "JPG", ["jpeg_quality"] = "88", ["preview_px"] = "720", ["preview_resample"] = "bicubic",
        };
        Check(texts.Keys.Order().SequenceEqual(ImageSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new ImageLabOptions();
        foreach (var setting in ImageSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        Check(set is { Seed: 77, SaveFormat: "jpeg", JpegQuality: 88, PreviewPixels: 720, Resampling: ImageResampling.Bicubic } && set.Extension == ".jpg",
            $"mapped: {set.Profile}, {set.Seed}, {set.SaveFormat}, {set.JpegQuality}, {set.PreviewPixels}, {set.Resampling}");
        Check(set.Encode(picture).AsSpan().SequenceEqual(JpegEncoder.Encode(picture, 88)), "jpeg at the setting's quality");

        var store = new MemorySettingsStore();
        ImageSettings.Save(store, set);
        Check(store.Keys.All(k => k.StartsWith(ImageSettings.Prefix, StringComparison.Ordinal)), string.Join(", ", store.Keys));
        var read = ImageSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && ImageSettings.Definitions.All(s => s.Get(read) == s.Get(set)), string.Join("; ", skipped));

        // The app's defaults under the saved values; a saved value that no longer reads is skipped, named.
        var app = new ImageLabOptions { JpegQuality = 60, PreviewPixels = 300 };
        store.Set(ImageSettings.Prefix + "preview_px", null);
        store.Set(ImageSettings.Prefix + "seed", "lots");
        var over = ImageSettings.Load(store, out skipped, app);
        Check(over.JpegQuality == 88 && over.PreviewPixels == 300 && app.JpegQuality == 60 && skipped.Count == 1 && skipped[0].StartsWith("Profile seed", StringComparison.Ordinal),
            $"{over.JpegQuality}, {over.PreviewPixels}; skipped {string.Join("; ", skipped)}");

        foreach (var (key, value, says) in new[]
                 {
                     ("jpeg_quality", "0", "1 to 100"), ("jpeg_quality", "101", "1 to 100"),
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
        var feature = new ImageFeature();
        var page = feature.CreatePage(services, feature.CreateViewModel(services));
        Check(page is ImagePage { BindingContext: ImageViewModel } && page.Title == "Image", $"page {page.GetType().Name}, title {page.Title}");
        Check(feature is { Name: MobileFeatures.Image, Order: 70, Needs: PhoneNeeds.Camera } && ((IMobileFeature)feature).State == FeatureState.Ready
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
