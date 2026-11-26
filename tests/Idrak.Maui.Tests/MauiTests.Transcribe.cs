// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.ApplicationModel.DataTransfer;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Transcribe lab's settings (plan 34, M5).
internal static partial class MauiTests
{
    private static void TranscribeSettingsMapped()
    {
        // Nothing set: every value is the library's own.
        var empty = new TranscribeLabOptions();
        var live = empty.ToLive();
        var library = new LiveTranscriptionOptions();
        foreach (var property in typeof(LiveTranscriptionOptions).GetProperties().Where(p => p.Name != nameof(LiveTranscriptionOptions.Transcription)))
        {
            Check(Equals(property.GetValue(live), property.GetValue(library)), $"{property.Name}: {property.GetValue(live)}, the library's {property.GetValue(library)}");
        }

        var request = new TranscriptionOptions();
        Check(live.Transcription.Language is null && live.Transcription.Task == request.Task && live.Transcription.Beams == request.Beams && live.Transcription.Prompt is null,
            "the transcription's defaults");
        Check(empty.ToCapture() == new AudioCaptureOptions(), $"capture {empty.ToCapture()}");
        Check(!empty.ToLive(microphone: false).KeepUp && empty.ToPipeline().IsEmpty && empty.FileMode == TranscribeFileMode.Live, "a file at its own pace, no pipeline, read live");
        var step = TranscribeSettings.Definitions.Single(s => s.Key == "step_s");
        Check(step.Placeholder == "1" && TranscribeSettings.Definitions.Single(s => s.Key == "window_s").Placeholder == "20", $"the library's defaults shown: {step.Placeholder}");

        // Each setting to its option.
        var texts = new Dictionary<string, string>
        {
            ["step_s"] = "0,5", ["window_s"] = "12", ["silence_s"] = "0.8", ["silence_db"] = "30", ["silence_floor_db"] = "-45", ["min_speech_s"] = "0.2",
            ["pre_roll_s"] = "0.1", ["policy"] = "", ["agreement"] = "3", ["prompt_chars"] = "50", ["keep_up"] = "off", ["beams"] = "2",
            ["prompt"] = "Idrak", ["pipeline"] = "", ["capture"] = "", ["input_device"] = "hw:1,0", ["sample_rate"] = "48000",
            ["block_s"] = "0.02", ["file_mode"] = "whole", ["file_block_s"] = "0.25", ["file_keep_up"] = "on",
        };
        Check(texts.Keys.Order().SequenceEqual(TranscribeSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new TranscribeLabOptions { Language = "ar", Task = "translate" };
        foreach (var setting in TranscribeSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        var mapped = set.ToLive();
        Check(mapped.StepSeconds == 0.5 && mapped.WindowSeconds == 12 && mapped.SilenceSeconds == 0.8 && mapped.SilenceDecibels == 30 && mapped.SilenceFloorDecibels == -45
              && mapped.MinSpeechSeconds == 0.2 && mapped.PreRollSeconds == 0.1 && mapped.Agreement == 3 && mapped.PromptCharacters == 50 && !mapped.KeepUp,
            $"live {mapped}");
        Check(mapped.Transcription.Language == "ar" && mapped.Transcription.Task == "translate" && mapped.Transcription.Beams == 2 && mapped.Transcription.Prompt == "Idrak",
            $"transcription {mapped.Transcription}");
        Check(set.ToLive(microphone: false).KeepUp && set.FileMode == TranscribeFileMode.Whole && set.FileBlockSeconds == 0.25, "the file's");
        Check(set.ToCapture() is { SampleRate: 48000, Device: "hw:1,0", BlockSeconds: 0.02, Channels: 1 }, $"capture {set.ToCapture()}");

        // Kept and read back; a bad saved value skipped by name.
        var store = new MemorySettingsStore();
        TranscribeSettings.Save(store, set);
        Check(store.Keys.Count > 0 && store.Keys.All(k => k.StartsWith(TranscribeSettings.Prefix, StringComparison.Ordinal)), $"keys: {string.Join(", ", store.Keys)}");
        var read = TranscribeSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Language == "ar" && read.Task == "translate", string.Join("; ", skipped));
        foreach (var setting in TranscribeSettings.Definitions)
        {
            Check(setting.Get(read) == setting.Get(set), $"{setting.Key}: read {setting.Get(read)}, kept {setting.Get(set)}");
        }

        store.Set(TranscribeSettings.Prefix + "window_s", "wide");
        TranscribeSettings.Load(store, out skipped);
        Check(skipped.Count == 1 && skipped[0].StartsWith("Window", StringComparison.Ordinal), $"skipped: {string.Join("; ", skipped)}");

        // Refused values say what is taken.
        foreach (var (key, text, says) in new[]
                 {
                     ("step_s", "0", "above 0"), ("agreement", "0", "1 or more"), ("silence_floor_db", "3", "under 0"), ("file_mode", "sometimes", "live or whole"),
                 })
        {
            try
            {
                TranscribeSettings.Definitions.Single(s => s.Key == key).Set(new TranscribeLabOptions(), text);
                Check(false, $"{key}={text} was taken");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{key}={text}: {e.Message}");
            }
        }
    }

    private sealed class FakeStore(params StoredModel[] models) : IModelStore
    {
        public string Folder => Path.GetTempPath();

        public bool HasToken => false;

        public Task<StoredModel> PullAsync(string model, string? revision = null, IProgressReporter? progress = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("no pulls here");

        public IReadOnlyList<StoredModel> List() => models;

        public bool Remove(string id) => false;

        public StorageSpace? FreeSpace() => null;
    }

    private sealed class FakePermissions(bool granted) : IPermissionService
    {
        public int Requests { get; private set; }

        public int SettingsOpened { get; private set; }

        public Task<PermissionOutcome> CheckAsync(PhoneNeeds needs) => Task.FromResult(Outcome(needs));

        public Task<PermissionOutcome> RequestAsync(PhoneNeeds needs)
        {
            Requests++;
            return Task.FromResult(Outcome(needs));
        }

        public void OpenSettings() => SettingsOpened++;

        private PermissionOutcome Outcome(PhoneNeeds needs) => granted ? new PermissionOutcome(needs, true, "")
            : new PermissionOutcome(needs, false, "The app may not use the microphone. Allow it in the app's settings (Permissions), then try again.");
    }

    private sealed class FakeShare : IShareService
    {
        public string? Shared { get; private set; }

        public Task ShareFileAsync(string path, string? title = null, string? contentType = null)
        {
            Shared = File.Exists(path) ? path : throw new FileNotFoundException(path);
            return Task.CompletedTask;
        }
    }

    private sealed class FakeClipboard : IClipboard
    {
        public string? Text { get; private set; }

        public bool HasText => Text is not null;

        public event EventHandler<EventArgs>? ClipboardContentChanged;

        public Task SetTextAsync(string? text)
        {
            Text = text;
            ClipboardContentChanged?.Invoke(this, EventArgs.Empty);
            return Task.CompletedTask;
        }

        public Task<string?> GetTextAsync() => Task.FromResult(Text);
    }
}
