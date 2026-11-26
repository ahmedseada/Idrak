// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

namespace Idrak.Maui.Tests;

// The Audio process lab's settings (plan 34, M6).
internal static partial class MauiTests
{
    private static void AudioProcessSettingsMapped()
    {
        var empty = new AudioProcessLabOptions();
        Check(empty.ToCapture() == new AudioCaptureOptions() && empty.ToPipeline().IsEmpty, $"the library's capture: {empty.ToCapture()}");
        Check(empty.ToPlayback("original") is { Label: "original", Playback: null } play && play.Output == new AudioPlaybackOptions(), "the library's playback");
        Check(empty.ToMonitorCapture() is { BlockSeconds: AudioProcessSettings.DefaultMonitorBlockSeconds, Performance: AudioPerformance.LowLatency }
              && empty.ToMonitorPlayback().Output is { BufferSeconds: AudioProcessSettings.DefaultMonitorBufferSeconds, Performance: AudioPerformance.LowLatency } monitor
              && monitor.BlockSeconds <= monitor.BufferSeconds, $"monitoring's short blocks and buffer: {empty.ToMonitorCapture()}, {empty.ToMonitorPlayback().Output}");
        var texts = new Dictionary<string, string>
        {
            ["process_block_s"] = "0,1", ["warm_run"] = "off", ["capture"] = "", ["input_device"] = "hw:1,0", ["sample_rate"] = "48000", ["channels"] = "2",
            ["capture_block_s"] = "0.02", ["max_record_s"] = "30", ["playback"] = "", ["output_device"] = "hw:0,0", ["buffer_s"] = "0.3",
            ["monitor_block_s"] = "0.005", ["monitor_buffer_s"] = "0.04", ["monitor_performance"] = "PowerSaving", ["format"] = "", ["bits"] = "float",
        };
        Check(texts.Keys.Order().SequenceEqual(AudioProcessSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new AudioProcessLabOptions();
        foreach (var setting in AudioProcessSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        Check(set.ProcessBlockSeconds == 0.1 && set.WarmRun == false && set.MaxRecordSeconds == 30 && set.ToEncode() is { Float: true, BitsPerSample: 32 }, "the lab's own");
        Check(set.ToCapture() is { Device: "hw:1,0", SampleRate: 48000, Channels: 2, BlockSeconds: 0.02 }, $"capture {set.ToCapture()}");
        Check(set.ToMonitorCapture() is { BlockSeconds: 0.005, Performance: AudioPerformance.PowerSaving }, $"monitor capture {set.ToMonitorCapture()}");
        Check(set.ToPlayback("x").Output is { Device: "hw:0,0", BufferSeconds: 0.3 }
              && set.ToMonitorPlayback().Output is { BufferSeconds: 0.04, Performance: AudioPerformance.PowerSaving }, $"playback {set.ToPlayback("x").Output}");

        var store = new MemorySettingsStore();
        AudioProcessSettings.Save(store, set);
        Check(store.Keys.All(k => k.StartsWith(AudioProcessSettings.Prefix, StringComparison.Ordinal)), $"keys: {string.Join(", ", store.Keys)}");
        var read = AudioProcessSettings.Load(store, out var skipped);
        Check(skipped.Count == 0, string.Join("; ", skipped));
        foreach (var setting in AudioProcessSettings.Definitions)
        {
            Check(setting.Get(read) == setting.Get(set), $"{setting.Key}: read {setting.Get(read)}, kept {setting.Get(set)}");
        }

        foreach (var (key, text, says) in new[]
                 {
                     ("process_block_s", "-1", "0 or more"), ("channels", "0", "from 1 to 32"), ("max_record_s", "0", "above 0"), ("warm_run", "maybe", "on, off"),
                 })
        {
            try
            {
                AudioProcessSettings.Definitions.Single(s => s.Key == key).Set(new AudioProcessLabOptions(), text);
                Check(false, $"{key}={text} was taken");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{key}={text}: {e.Message}");
            }
        }
    }
}
