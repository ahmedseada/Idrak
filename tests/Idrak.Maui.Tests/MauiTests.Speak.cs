// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Speak lab (plan 34, M6): its settings, and its page over the app's services.
internal static partial class MauiTests
{
    // Whether `action` throws `TException` with `says` in its message.
    private static bool Throws<TException>(Action action, string says) where TException : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (TException e)
        {
            return e.Message.Contains(says, StringComparison.Ordinal);
        }
    }

    private static void SpeakSettingsMapped()
    {
        var empty = new SpeakLabOptions();
        var speech = empty.ToSynthesis();
        Check(speech == new SpeechSynthesisOptions() with { MaxCharacters = SpeakSettings.DefaultMaxCharacters }, $"the library's synthesis options but the lab's pieces: {speech}");
        Check(empty.ToPlayback() is { Playback: null, Label: "speech" } play && play.Output == new AudioPlaybackOptions(), "the library's playback");
        Check(empty.ToEncode() == new AudioEncodeOptions(), "16-bit WAV");
        Check(SpeakSettings.Definitions.Single(s => s.Key == "rate").Placeholder == "1" && SpeakSettings.Definitions.Single(s => s.Key == "pieces").Placeholder == "160"
              && SpeakSettings.Definitions.Single(s => s.Key == "buffer_s").Placeholder == "0.5", "the defaults shown");

        var texts = new Dictionary<string, string>
        {
            ["rate"] = "1,5", ["noise"] = "0", ["duration_noise"] = "0.2", ["seed"] = "7", ["pieces"] = "40", ["decoder_s"] = "2", ["sample_rate"] = "22050",
            ["vocoder"] = "", ["voice_conversion"] = "off", ["tau"] = "0.1", ["voice_store"] = "", ["playback"] = "",
            ["output_device"] = "hw:0,0", ["buffer_s"] = "0.25", ["block_s"] = "0.01", ["performance"] = "LowLatency", ["format"] = "", ["bits"] = "24",
        };
        Check(texts.Keys.Order().SequenceEqual(SpeakSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new SpeakLabOptions { Model = "a/b@c", Voice = "alice" };
        foreach (var setting in SpeakSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        var mapped = set.ToSynthesis();
        Check(mapped.Rate == 1.5f && mapped.NoiseScale == 0 && mapped.DurationNoiseScale == 0.2f && mapped.Seed == 7 && mapped.MaxCharacters == 40 && mapped.DecoderSeconds == 2
              && mapped.SampleRate == 22050 && mapped.VoiceConversion == "off" && mapped.Conversion is { Tau: 0.1f, Seed: 7 } && mapped.Voice == "alice", $"synthesis {mapped}");
        Check(set.ToPlayback().Output is { Device: "hw:0,0", BufferSeconds: 0.25, BlockSeconds: 0.01, Performance: AudioPerformance.LowLatency }, $"playback {set.ToPlayback()}");
        Check(set.ToEncode().BitsPerSample == 24, "saved as");

        var store = new MemorySettingsStore();
        SpeakSettings.Save(store, set);
        Check(store.Keys.Count > 0 && store.Keys.All(k => k.StartsWith(SpeakSettings.Prefix, StringComparison.Ordinal)), $"keys: {string.Join(", ", store.Keys)}");
        var read = SpeakSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Model == "a/b@c" && read.Voice == "alice", string.Join("; ", skipped));
        foreach (var setting in SpeakSettings.Definitions)
        {
            Check(setting.Get(read) == setting.Get(set), $"{setting.Key}: read {setting.Get(read)}, kept {setting.Get(set)}");
        }

        store.Set(SpeakSettings.Prefix + "rate", "quick");
        SpeakSettings.Load(store, out skipped);
        Check(skipped.Count == 1 && skipped[0].StartsWith("Speaking rate", StringComparison.Ordinal), $"skipped: {string.Join("; ", skipped)}");
        foreach (var (key, text, says) in new[]
                 {
                     ("rate", "0", "above 0"), ("noise", "-1", "0 or more"), ("pieces", "-3", "0 or more"), ("performance", "fast", "LowLatency"), ("bits", "12", "one of"),
                 })
        {
            try
            {
                SpeakSettings.Definitions.Single(s => s.Key == key).Set(new SpeakLabOptions(), text);
                Check(false, $"{key}={text} was taken");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{key}={text}: {e.Message}");
            }
        }
    }

    private static void SpeakPageOpens()
    {
        IMobileFeature feature = new SpeakFeature();
        Check(feature is SpeakFeature { Name: MobileFeatures.Speak, Order: 40 } && feature.State == FeatureState.Ready && (feature.Needs & PhoneNeeds.Model) != 0
              && MobileFeatures.Place(feature) == 40, $"the speak feature: {feature.GetType().Name}");
        var options = new IdrakMauiOptions { Device = "cpu" };
        using var models = new ModelService(new ManualMemoryPressure(), options);
        using var services = new ServiceCollection().AddSingleton<IModelStore>(new FakeStore()).AddSingleton<IModelService>(models).AddSingleton(options)
            .AddSingleton<ISettingsStore>(new MemorySettingsStore()).AddSingleton<ISoundPlayer>(new SoundPlayer()).AddSingleton<IShareService>(new FakeShare())
            .BuildServiceProvider();
        var page = feature.CreatePage(services, feature.CreateViewModel(services));
        Check(page is SpeakPage { BindingContext: SpeakViewModel { OpenModels: not null } } && page.Title == "Speak", $"page {page.GetType().Name}, title {page.Title}");
        ThemeKeysOnly(Path.Combine("src", "Idrak.Maui", "Views", "SpeakView.xaml"));
    }

    // A page's sizes, spacing and colours come from theme keys ({DynamicResource ...}) or its own styles, never a literal on an element.
    private static void ThemeKeysOnly(string file)
    {
        var literal = new System.Text.RegularExpressions.Regex("""<(?!Setter)\w[^>]*\b(TextColor|BackgroundColor|BorderColor|Stroke|StrokeThickness|FontSize|Padding|Margin|Spacing|ColumnSpacing|RowSpacing)="(?!\{)""");
        string xaml = File.ReadAllText(Path.Combine(RepositoryRoot(), file));
        var found = literal.Matches(xaml).Select(m => m.Groups[1].Value).ToList();
        Check(found.Count == 0, $"{file} sets {string.Join(", ", found)} without a theme key");
    }
}
