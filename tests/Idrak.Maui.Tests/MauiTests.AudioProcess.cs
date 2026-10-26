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

// The Audio process lab (plan 34, M6) on the CPU: a capture registered in AudioCaptures feeding a generated tone, a
// playback registered in AudioPlaybacks recording what it is given, a WAV made here; the results against the library's own
// pipeline on the same sound (rule 71).
internal static partial class MauiTests
{
    // One Audio process page over fakes: the capture and the speaker registered (an app's: ForThisMachine picks them first).
    private sealed class ProcessLab : IDisposable
    {
        public ProcessLab(FakeCapture? capture = null, bool granted = true, IDispatcher? dispatcher = null, TimeProvider? time = null, TimeSpan? interval = null)
        {
            Capture = capture;
            Folder = Temp("process-lab");
            if (capture is not null)
            {
                AudioCaptures.Register(capture);
            }

            AudioPlaybacks.Register(Playback);
            Center = new ProgressCenter(interval ?? TimeSpan.FromMilliseconds(250), dispatcher, time);
            Player = new SoundPlayer(time);
            Permissions = new FakePermissions(granted);
            Vm = new AudioProcessViewModel(Player, Permissions, Share, new IdrakMauiOptions { Device = "cpu" }, Settings, Center);
        }

        public FakeCapture? Capture { get; }

        public FakePlayback Playback { get; } = new();

        public string Folder { get; }

        public ProgressCenter Center { get; }

        public SoundPlayer Player { get; }

        public FakePermissions Permissions { get; }

        public FakeShare Share { get; } = new();

        public MemorySettingsStore Settings { get; } = new();

        public AudioProcessViewModel Vm { get; }

        public void Set(string key, string text)
        {
            Vm.Settings.Single(e => e.Setting.Key == key).Text = text;
            Check(Vm.ApplySettings(), $"{key}={text}: {Vm.SettingsMessage} {string.Join("; ", Vm.Settings.Where(e => e.HasError).Select(e => e.Error))}");
        }

        // A WAV of a loud tone, a pause and a soft one, written here and opened in the lab.
        public AudioData OpenTone()
        {
            var signal = Signal((1, 0.5f), (0.5, 0), (1, 0.2f));
            for (int i = 0; i < signal.Length; i++)
            {
                signal[i] += 0.05f;                                              // an offset the high-pass takes off
            }

            string wav = Path.Combine(Folder, "tone.wav");
            AudioEncoders.Save(wav, new AudioData(signal, 1, Rate));
            Check(Vm.OpenFileAsync(wav).GetAwaiter().GetResult(), $"opened: {Vm.Error}");
            return AudioCodecs.Decode(wav);
        }

        public void Dispose()
        {
            Vm.Dispose();
            Player.Dispose();
            AudioPlaybacks.Unregister(Playback.Name);
            if (Capture is not null)
            {
                AudioCaptures.Unregister(Capture.Name);
            }
        }
    }

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
            ["process_block_s"] = "0,1", ["warm_run"] = "off", ["capture"] = AudioCaptures.Alsa, ["input_device"] = "hw:1,0", ["sample_rate"] = "48000", ["channels"] = "2",
            ["capture_block_s"] = "0.02", ["max_record_s"] = "30", ["playback"] = AudioPlaybacks.Alsa, ["output_device"] = "hw:0,0", ["buffer_s"] = "0.3",
            ["monitor_block_s"] = "0.005", ["monitor_buffer_s"] = "0.04", ["monitor_performance"] = "PowerSaving", ["format"] = "mp3", ["bits"] = "float",
        };
        Check(texts.Keys.Order().SequenceEqual(AudioProcessSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new AudioProcessLabOptions { Pipeline = "highpass=80" };
        foreach (var setting in AudioProcessSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        Check(set.ProcessBlockSeconds == 0.1 && set.WarmRun == false && set.MaxRecordSeconds == 30 && set.SaveFormat == "mp3" && set.ToEncode() is { Float: true, BitsPerSample: 32 },
            "the lab's own");
        Check(set.ToCapture() is { Device: "hw:1,0", SampleRate: 48000, Channels: 2, BlockSeconds: 0.02 } && set.Capture == AudioCaptures.Alsa, $"capture {set.ToCapture()}");
        Check(set.ToMonitorCapture() is { BlockSeconds: 0.005, Performance: AudioPerformance.PowerSaving }, $"monitor capture {set.ToMonitorCapture()}");
        Check(set.ToPlayback("x") is { Playback: AudioPlaybacks.Alsa } p && p.Output is { Device: "hw:0,0", BufferSeconds: 0.3 }
              && set.ToMonitorPlayback().Output is { BufferSeconds: 0.04, Performance: AudioPerformance.PowerSaving }, $"playback {set.ToPlayback("x").Output}");

        var store = new MemorySettingsStore();
        AudioProcessSettings.Save(store, set);
        Check(store.Keys.All(k => k.StartsWith(AudioProcessSettings.Prefix, StringComparison.Ordinal)) && store.Get(AudioProcessSettings.Prefix + "pipeline") == "highpass=80",
            $"keys: {string.Join(", ", store.Keys)}");
        var read = AudioProcessSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Pipeline == "highpass=80", string.Join("; ", skipped));
        foreach (var setting in AudioProcessSettings.Definitions)
        {
            Check(setting.Get(read) == setting.Get(set), $"{setting.Key}: read {setting.Get(read)}, kept {setting.Get(set)}");
        }

        foreach (var (key, text, says) in new[]
                 {
                     ("process_block_s", "-1", "0 or more"), ("channels", "0", "from 1 to 32"), ("capture", "nosuch", "registered"), ("max_record_s", "0", "above 0"),
                     ("warm_run", "maybe", "on, off"), ("format", "ogg", "registered"),
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

    private static void AudioProcessChips()
    {
        using var lab = new ProcessLab();
        var vm = lab.Vm;
        Check(vm.Processors.Select(c => c.Name).SequenceEqual(AudioProcessors.Names), $"chips: {string.Join(", ", vm.Processors.Select(c => c.Name))}");
        var refused = new List<string>();
        foreach (var chip in vm.Processors)
        {
            try
            {
                _ = AudioPipeline.Parse(chip.Step);
            }
            catch (ArgumentException e)
            {
                refused.Add($"{chip.Step}: {e.Message}");
            }

            Check(chip.Summary.Length > 0 && chip.Label.Length > 0 && chip.ParametersText.Length > 0 && chip.Live == AudioProcessors.Get(chip.Name).Live, $"{chip.Name}: its description");
        }

        Check(refused.Count == 0, $"a chip's step does not read: {string.Join(" | ", refused)}");
        var normalize = vm.Processors.Single(c => c.Name == AudioProcessors.Normalize);
        var highpass = vm.Processors.Single(c => c.Name == AudioProcessors.HighPass);
        Check(!normalize.Live && normalize.LiveText.Contains("agc", StringComparison.Ordinal) && highpass.LiveText == "live" && highpass.ParametersText.Contains("Hz", StringComparison.Ordinal),
            $"normalize: {normalize.LiveText}; highpass: {highpass.LiveText}, {highpass.ParametersText}");

        // Tapping appends a step after a comma, keeps the pipeline, and the check says what it is.
        Check(vm.PipelineCheck.StartsWith("No steps", StringComparison.Ordinal) && !vm.HasPipelineError, $"empty: {vm.PipelineCheck}");
        highpass.AddCommand.Execute(null);
        Check(vm.PipelineText == highpass.Step && vm.PipelineCheck.Contains("all live", StringComparison.Ordinal), $"one step: {vm.PipelineText}; {vm.PipelineCheck}");
        vm.AddStepCommand.Execute(normalize);
        Check(vm.PipelineText == $"{highpass.Step}, {normalize.Step}" && vm.PipelineCheck.StartsWith("2 steps", StringComparison.Ordinal)
              && vm.PipelineCheck.Contains("normalize needs the whole recording", StringComparison.Ordinal), $"two steps: {vm.PipelineText}; {vm.PipelineCheck}");
        Check(lab.Settings.Get(AudioProcessSettings.Prefix + "pipeline") == vm.PipelineText, "kept");
        vm.PipelineText = "highpass=80, nosuchstep";
        Check(vm.HasPipelineError && vm.PipelineError.Contains("nosuchstep", StringComparison.Ordinal) && vm.PipelineError.Contains("registered", StringComparison.Ordinal)
              && vm.PipelineCheck.Length == 0, $"an unknown step: {vm.PipelineError}");
        lab.OpenTone();
        Check(!vm.ApplyAsync().GetAwaiter().GetResult() && vm.Error.Contains("does not read", StringComparison.Ordinal) && !vm.HasProcessed, $"refused: {vm.Error}");
        vm.PipelineText = "highpass=-5";
        Check(vm.HasPipelineError, $"a value out of range: {vm.PipelineCheck}");
        vm.ClearPipelineCommand.Execute(null);
        Check(vm.PipelineText.Length == 0 && !vm.HasPipelineError, "cleared");
    }

    private static void AudioProcessApplied()
    {
        string cache = Temp("process-cache");
        ComputeResources.CacheFolder = cache;
        try
        {
            // The peak meter: the largest magnitude in decibels of full scale, its vector loop and scalar tail alike.
            var tone = Signal((0.1, 0.5f));
            foreach (int length in (int[])[tone.Length, 1001, 7, 1])
            {
                double expectedPeak = 20 * Math.Log10(Math.Max(tone.AsSpan(0, length).ToArray().Max(MathF.Abs), 1e-6f));
                Check(Math.Abs(LevelMeter.PeakDecibels(tone.AsSpan(0, length)) - expectedPeak) < 1e-4, $"{length} samples: peak {LevelMeter.PeakDecibels(tone.AsSpan(0, length))}, expected {expectedPeak}");
            }

            Check(Math.Abs(LevelMeter.PeakDecibels(tone) - 20 * Math.Log10(0.5)) < 0.01 && LevelMeter.PeakDecibels(new float[800]) == -120 && double.IsNegativeInfinity(LevelMeter.PeakDecibels([])),
                $"a tone at half scale peaks at {LevelMeter.PeakDecibels(tone)} dB; silence and nothing");

            using var lab = new ProcessLab();
            var vm = lab.Vm;
            Check(!vm.ApplyAsync().GetAwaiter().GetResult() && vm.Error.Contains("Record or pick", StringComparison.Ordinal), $"nothing to apply: {vm.Error}");
            var audio = lab.OpenTone();
            Check(vm.HasOriginal && vm.OriginalText.Contains("16000 Hz", StringComparison.Ordinal) && vm.BeforeText.StartsWith("RMS", StringComparison.Ordinal), $"{vm.OriginalText}; {vm.BeforeText}");

            // The library's own result on the same sound, bit for bit; its latency and delay; cold, then warm.
            const string Text = "highpass=80, limit, normalize=-20";
            vm.PipelineText = Text;
            Check(vm.ApplyAsync().GetAwaiter().GetResult(), $"applied: {vm.Error}");
            var pipeline = AudioPipeline.Parse(Text);
            var expected = pipeline.Apply(audio);
            Check(vm.Processed is { } processed && processed.Samples.AsSpan().SequenceEqual(expected.Samples) && processed.SampleRate == expected.SampleRate,
                "the processed sound is the library's");
            int latency, delay;
            using (var stage = pipeline.Open(audio.SampleRate, audio.Channels))
            {
                (latency, delay) = (stage.Latency, stage.Delay);
            }

            Check(vm.Latency == latency && vm.Delay == delay && vm.LatencyText.Contains($"Latency {latency} samples", StringComparison.Ordinal)
                  && vm.LatencyText.Contains($"Delay {delay} samples", StringComparison.Ordinal), $"latency {vm.Latency} ({latency}), delay {vm.Delay} ({delay}): {vm.LatencyText}");
            Check(vm.ColdTime is { } cold && vm.WarmTime is { } warm && cold > TimeSpan.Zero && warm > TimeSpan.Zero && vm.ColdText.StartsWith("Cold", StringComparison.Ordinal)
                  && vm.WarmText.StartsWith("Warm", StringComparison.Ordinal) && vm.ColdText.Contains("real-time factor", StringComparison.Ordinal), $"{vm.ColdText}; {vm.WarmText}");
            Check(vm.AfterText.StartsWith("RMS", StringComparison.Ordinal) && vm.AfterText != vm.BeforeText, $"levels: {vm.BeforeText} / {vm.AfterText}");
            double rms = LevelMeter.Decibels(expected.Samples), peak = LevelMeter.PeakDecibels(expected.Samples);
            Check(vm.AfterText == $"RMS {AudioTextFor(rms)}, peak {AudioTextFor(peak)}" && peak <= 0, $"the processed level: {vm.AfterText} (RMS {rms:0.00}, peak {peak:0.00})");
            var coldBefore = vm.ColdTime;
            Check(vm.ApplyAsync().GetAwaiter().GetResult() && vm.ColdTime == coldBefore, "the cold time is the pipeline's first apply's");

            // In blocks, as live sound goes: the same samples.
            lab.Set("process_block_s", "0.1");
            Check(vm.ApplyAsync().GetAwaiter().GetResult(), $"in blocks: {vm.Error}");
            var blocks = vm.Processed!;
            Check(blocks.Length == expected.Length && MaxDifference(blocks.Samples, expected.Samples) < 1e-5f && vm.ProcessedText.Contains("in blocks", StringComparison.Ordinal),
                $"in blocks: {blocks.Length} samples for {expected.Length}, largest difference {MaxDifference(blocks.Samples, expected.Samples)}");

            // Warm off: a new pipeline is timed cold only.
            lab.Set("warm_run", "off");
            lab.Set("process_block_s", "");
            vm.PipelineText = "highpass=120";
            Check(vm.ApplyAsync().GetAwaiter().GetResult() && vm.WarmTime is null && vm.WarmText.Contains("off", StringComparison.Ordinal), $"warm off: {vm.WarmText}");

            // Saved and shared.
            string? wav = vm.SaveAsync().GetAwaiter().GetResult();
            Check(wav is not null && wav.StartsWith(cache, StringComparison.Ordinal) && AudioCodecs.Decode(wav).Length == vm.Processed!.Length, $"saved: {wav} {vm.Error}");
            string? mp3 = vm.SaveAsync("mp3").GetAwaiter().GetResult();
            Check(mp3 is not null && mp3.EndsWith(".mp3", StringComparison.Ordinal) && new FileInfo(mp3).Length > 0, $"mp3: {mp3} {vm.Error}");
            Check(vm.ShareAsync().GetAwaiter().GetResult() && lab.Share.Shared is { } shared && shared.EndsWith(".wav", StringComparison.Ordinal), $"shared: {lab.Share.Shared}");

            // A file that is not sound.
            string bad = Path.Combine(lab.Folder, "notes.wav");
            File.WriteAllText(bad, "not a sound");
            Check(!vm.OpenFileAsync(bad).GetAwaiter().GetResult() && vm.Error.StartsWith("notes.wav could not be read", StringComparison.Ordinal), $"error: {vm.Error}");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
        }
    }

    private static string AudioTextFor(double decibels) => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{decibels:0.0} dB");

    private static float MaxDifference(float[] a, float[] b)
    {
        float worst = 0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }

        return worst;
    }

    private static void AudioProcessRecorded()
    {
        var signal = Signal((1.5, 0.5f), (0.5, 0.2f));
        using (var lab = new ProcessLab(new FakeCapture(signal)))
        {
            Check(lab.Vm.RecordAsync().GetAwaiter().GetResult(), $"recorded: {lab.Vm.Error}");
            Check(lab.Permissions.Requests == 1 && lab.Capture!.Opened == 1 && lab.Vm.Original is { } original && original.Samples.AsSpan().SequenceEqual(signal),
                $"the recording is the microphone's: {lab.Vm.Original?.Length} samples for {signal.Length}");
            Check(lab.Vm.OriginalText.Contains("recorded from fake-microphone", StringComparison.Ordinal) && !lab.Vm.IsRecording && lab.Vm.IsIdle, lab.Vm.OriginalText);
            Check(lab.Vm.Level > 0 && lab.Vm.LevelText.EndsWith("dB", StringComparison.Ordinal), $"the level shown: {lab.Vm.Level} {lab.Vm.LevelText}");

            // The longest recording stops it.
            lab.Set("max_record_s", "1");
            Check(lab.Vm.RecordAsync().GetAwaiter().GetResult() && lab.Vm.Original!.Length == Rate && lab.Vm.OriginalText.Contains("longest recording", StringComparison.Ordinal),
                $"stopped at the longest: {lab.Vm.Original?.Length}; {lab.Vm.OriginalText}");
        }

        // A recording held open stops when asked, what was heard kept.
        using (var held = new ProcessLab(new FakeCapture(signal) { HoldOpen = true }))
        {
            var run = held.Vm.RecordAsync();
            WaitFor(() => held.Capture!.Blocks * 0.05 >= 2, "the signal to be heard");
            Check(held.Vm.IsRecording && held.Vm.RecordText == "Stop", "recording");
            held.Vm.RecordCommand.Execute(null);
            Check(run.Wait(TimeSpan.FromSeconds(20)) && run.Result && held.Vm.Original!.Samples.AsSpan().SequenceEqual(signal), $"stopped and kept: {held.Vm.Error}");
        }

        // Refused: nothing opened, how to allow it said.
        using (var refused = new ProcessLab(new FakeCapture(signal), granted: false))
        {
            Check(!refused.Vm.RecordAsync().GetAwaiter().GetResult() && refused.Capture!.Opened == 0 && refused.Vm.IsPermissionRefused
                  && refused.Vm.Error.Contains("Allow it in the app's settings", StringComparison.Ordinal), $"refused: {refused.Vm.Error}");
            Check(!refused.Vm.MonitorAsync().GetAwaiter().GetResult() && refused.Capture!.Opened == 0, "monitoring refused too");
            refused.Vm.OpenPermissionSettingsCommand.Execute(null);
            Check(refused.Permissions.SettingsOpened == 1, "the app's settings opened");
        }

        // Hundreds of blocks inside one progress interval: at most one UI dispatch; the last values shown at the end.
        var dispatcher = new QueuedDispatcher();
        using var batched = new ProcessLab(new FakeCapture(Signal((10, 0.5f))), dispatcher: dispatcher, time: new SharedTime(), interval: TimeSpan.FromHours(1));
        int before = dispatcher.Dispatched;
        Check(batched.Vm.RecordAsync().GetAwaiter().GetResult(), $"recorded: {batched.Vm.Error}");
        int dispatched = dispatcher.Dispatched - before;
        Check(batched.Capture!.Blocks >= 200 && dispatched <= 2 && batched.Vm.Level > 0, $"{batched.Capture.Blocks} blocks made {dispatched} UI dispatches");
        Console.WriteLine($"    {batched.Capture.Blocks} blocks: {dispatched} UI dispatch(es)");
    }

    private static void AudioProcessPlayed()
    {
        using var lab = new ProcessLab();
        var vm = lab.Vm;
        Check(!vm.PlayAsync(original: true).GetAwaiter().GetResult() && vm.Status.Contains("Record or pick", StringComparison.Ordinal), "nothing to play");
        var audio = lab.OpenTone();
        vm.PipelineText = "highpass=80, normalize=-20";
        Check(vm.ApplyAsync().GetAwaiter().GetResult(), vm.Error);
        Check(vm.PlayAsync(original: true).GetAwaiter().GetResult() && lab.Playback.Samples().AsSpan().SequenceEqual(audio.Samples), "A: the original played");
        lab.Playback.Written.Clear();
        Check(vm.PlayAsync(original: false).GetAwaiter().GetResult() && lab.Playback.Samples().AsSpan().SequenceEqual(vm.Processed!.Samples), "B: the processed played");
        Check(!vm.IsPlaying && vm.PlayingText.Length == 0 && vm.Status.StartsWith("Played the processed", StringComparison.Ordinal), vm.Status);

        // B while A plays: A stops, B plays to its end, one output at a time.
        var held = new TaskCompletionSource();
        lab.Playback.OnWrite = (_, token) => held.Task.WaitAsync(token);
        var a = vm.PlayAsync(original: true);
        WaitFor(() => lab.Player.Playing == "original", "the original to play");
        Check(vm.IsPlaying && vm.PlayingText.Contains("original", StringComparison.Ordinal), vm.PlayingText);
        lab.Playback.OnWrite = null;
        Check(vm.PlayAsync(original: false).GetAwaiter().GetResult() && a.Wait(TimeSpan.FromSeconds(10)) && !a.Result && lab.Playback.MostOpen == 1,
            $"A stopped by B: most open {lab.Playback.MostOpen}");
        Check(!vm.IsPlaying && vm.Status.StartsWith("Played the processed", StringComparison.Ordinal), $"B's status kept: {vm.Status}");

        // Stop stops it.
        lab.Playback.OnWrite = (_, token) => held.Task.WaitAsync(token);
        var c = vm.PlayAsync(original: false);
        WaitFor(() => lab.Player.IsPlaying, "the processed to play");
        vm.StopCommand.Execute(null);
        Check(c.Wait(TimeSpan.FromSeconds(10)) && !c.Result && !vm.IsPlaying, "stopped");
    }

    private static void AudioProcessMonitored()
    {
        var signal = Signal((1, 0.5f), (0.5, 0.2f));
        using var lab = new ProcessLab(new FakeCapture(signal));
        var vm = lab.Vm;
        vm.PipelineText = "highpass=80, trim";
        Check(!vm.MonitorAsync().GetAwaiter().GetResult() && vm.Error.Contains("live steps only", StringComparison.Ordinal) && vm.Error.Contains("trim", StringComparison.Ordinal)
              && lab.Capture!.Opened == 0, $"a whole-recording step refused by name: {vm.Error}");
        const string Text = "highpass=80, limit";
        vm.PipelineText = Text;
        Check(vm.MonitorAsync().GetAwaiter().GetResult(), $"monitored: {vm.Error}");
        Check(lab.Capture!.Opened == 1 && lab.Playback.LastOptions is { BufferSeconds: AudioProcessSettings.DefaultMonitorBufferSeconds, Performance: AudioPerformance.LowLatency },
            $"the monitor's playback: {lab.Playback.LastOptions}");

        // What played is the pipeline's stage over the microphone's blocks, bit for bit, and the recording's apply shifted by the latency.
        var pipeline = AudioPipeline.Parse(Text);
        int block = (int)Math.Round(AudioProcessSettings.DefaultMonitorBlockSeconds * Rate);
        var expected = new List<float>();
        int latency, delay;
        using (var stage = pipeline.Open(Rate, 1))
        {
            (latency, delay) = (stage.Latency, stage.Delay);
            for (int start = 0; start < signal.Length; start += block)
            {
                expected.AddRange(stage.Process(new AudioData(signal.AsSpan(start, Math.Min(block, signal.Length - start)).ToArray(), 1, Rate)).Samples);
            }
        }

        float[] played = lab.Playback.Samples();
        Check(played.AsSpan().SequenceEqual(expected.ToArray()), $"{played.Length} samples played, {expected.Count} from the stage");
        var whole = pipeline.Apply(new AudioData(signal, 1, Rate));
        Check(latency > 0 && played.Length >= signal.Length - latency && MaxDifference(played[latency..], whole.Samples) < 1e-5f,
            $"the apply shifted by the latency ({latency}): largest difference {MaxDifference(played[latency..], whole.Samples)}");
        double expectedDelay = (double)delay / Rate + AudioProcessSettings.DefaultMonitorBlockSeconds + AudioProcessSettings.DefaultMonitorBufferSeconds;
        Check(Math.Abs(vm.MonitorDelaySeconds - expectedDelay) < 1e-9 && vm.MonitorText.StartsWith("Delay about", StringComparison.Ordinal) && !vm.IsMonitoring
              && vm.MonitorWarning.Contains("headphones", StringComparison.Ordinal), $"delay {vm.MonitorDelaySeconds} ({expectedDelay}): {vm.MonitorText}");

        // Held open, it stops when asked.
        using var held = new ProcessLab(new FakeCapture(signal) { HoldOpen = true });
        held.Vm.PipelineText = "highpass=80";
        var run = held.Vm.MonitorAsync();
        WaitFor(() => held.Vm.IsMonitoring && held.Playback.Written.Count > 10, "monitoring to play");
        Check(held.Vm.MonitorButtonText == "Stop monitoring", held.Vm.MonitorButtonText);
        held.Vm.MonitorCommand.Execute(null);
        Check(run.Wait(TimeSpan.FromSeconds(20)) && run.Result && !held.Vm.IsMonitoring && held.Playback.Disposed == held.Playback.Opened, $"stopped: {held.Vm.Error}");
    }

    private static void AudioProcessPageOpens()
    {
        var feature = MobileFeatures.Get(MobileFeatures.AudioProcess);
        Check(feature is AudioProcessFeature { Order: 50 } && feature.State == FeatureState.Ready && (feature.Needs & PhoneNeeds.Microphone) != 0
              && MobileFeatures.Origin(MobileFeatures.AudioProcess) == Overrides.Library && MobileFeatures.Place(feature) == 50, $"the audio process feature: {feature.GetType().Name}");
        using var services = new ServiceCollection().AddSingleton(new IdrakMauiOptions()).AddSingleton<ISettingsStore>(new MemorySettingsStore())
            .AddSingleton<ISoundPlayer>(new SoundPlayer()).AddSingleton<IPermissionService>(new FakePermissions(true)).AddSingleton<IShareService>(new FakeShare())
            .BuildServiceProvider();
        var page = MobileFeatures.Open(MobileFeatures.AudioProcess, services);
        Check(page is AudioProcessPage { BindingContext: AudioProcessViewModel { Processors.Count: > 0 } } && page.Title == "Audio process", $"page {page.GetType().Name}, title {page.Title}");
        ((AudioProcessViewModel)page.BindingContext).Dispose();
        ThemeKeysOnly(Path.Combine("src", "Idrak.Maui", "Views", "AudioProcessPage.xaml"));
    }
}
