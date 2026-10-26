// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Whisper;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.ApplicationModel.DataTransfer;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Transcribe lab (plan 34, M5) on the CPU: a capture registered in AudioCaptures feeding a generated tone, a scripted
// recognizer kept in the model service's "speech" host, a store of model folders written here, a WAV made here; and the
// repository's tiny-whisper end to end through the Whisper plug-in.
internal static partial class MauiTests
{
    private const int Rate = 16000;
    private const string FakeArchitecture = "FakeSpeechForMauiTests";

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
            ["pre_roll_s"] = "0.1", ["policy"] = LiveCommitPolicies.Agreement, ["agreement"] = "3", ["prompt_chars"] = "50", ["keep_up"] = "off", ["beams"] = "2",
            ["prompt"] = "Idrak", ["pipeline"] = "highpass=80", ["capture"] = AudioCaptures.Alsa, ["input_device"] = "hw:1,0", ["sample_rate"] = "48000",
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
              && mapped.MinSpeechSeconds == 0.2 && mapped.PreRollSeconds == 0.1 && mapped.Agreement == 3 && mapped.PromptCharacters == 50 && !mapped.KeepUp
              && mapped.Policy == LiveCommitPolicies.Agreement, $"live {mapped}");
        Check(mapped.Transcription.Language == "ar" && mapped.Transcription.Task == "translate" && mapped.Transcription.Beams == 2 && mapped.Transcription.Prompt == "Idrak",
            $"transcription {mapped.Transcription}");
        Check(set.ToLive(microphone: false).KeepUp && set.FileMode == TranscribeFileMode.Whole && set.FileBlockSeconds == 0.25, "the file's");
        Check(set.ToCapture() is { SampleRate: 48000, Device: "hw:1,0", BlockSeconds: 0.02, Channels: 1 } && set.Capture == AudioCaptures.Alsa, $"capture {set.ToCapture()}");
        Check(set.ToPipeline().ToString() == "highpass=80", $"pipeline {set.ToPipeline()}");

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
                     ("step_s", "0", "above 0"), ("agreement", "0", "1 or more"), ("silence_floor_db", "3", "under 0"), ("capture", "nosuch", "registered"),
                     ("pipeline", "normalize", "whole recording"), ("pipeline", "nosuchstep", "Unknown audio processor"), ("file_mode", "sometimes", "live or whole"),
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

    // A tone of `amplitude` (0: silence) for each part, at 16 kHz.
    private static float[] Signal(params (double Seconds, float Amplitude)[] parts)
    {
        var samples = new List<float>();
        foreach (var (seconds, amplitude) in parts)
        {
            int count = (int)Math.Round(seconds * Rate);
            for (int i = 0; i < count; i++)
            {
                samples.Add(amplitude * MathF.Sin(2 * MathF.PI * 440 * samples.Count / Rate));
            }
        }

        return [.. samples];
    }

    // A recognizer whose words follow the sound: WordsPerSecond of the sound's length, from a vocabulary chosen by its peak
    // (a loud tone "alpha beta …", a soft one "one two …"); each call can move the test's clock on.
    private sealed class ScriptedRecognizer : ISpeechRecognizer
    {
        private static readonly string[] Loud = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];
        private static readonly string[] Soft = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
        private int _calls;

        public double WordsPerSecond { get; init; } = 2;

        public Action? OnCall { get; init; }

        public int Calls => Volatile.Read(ref _calls);

        public List<(int Samples, int Rate, string? Language)> Heard { get; } = [];

        public Device Device => Device.Cpu;

        public IReadOnlySet<string> Languages { get; } = new HashSet<string>(["en", "ar"]);

        public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            lock (Heard)
            {
                Heard.Add((audio.Length, audio.SampleRate, options?.Language));
            }

            Interlocked.Increment(ref _calls);
            OnCall?.Invoke();
            float peak = 0;
            foreach (float x in audio.Samples)
            {
                peak = Math.Max(peak, Math.Abs(x));
            }

            var words = peak > 0.35f ? Loud : Soft;
            int n = Math.Min(words.Length, (int)(audio.Duration.TotalSeconds * WordsPerSecond));
            string text = string.Join(' ', words.Take(n));
            progress?.Report(new TranscriptionProgress(1, 1, "decoder", TimeSpan.Zero));
            var stats = new TranscriptionStats(audio.Duration, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, 0, 1, 1);
            return new Transcription(text, options?.Language, [new TranscriptSegment(0, audio.Duration.TotalSeconds, text, 0f, [])], stats);
        }

        public IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("not in these tests");

        public void Dispose()
        {
        }
    }

    // A microphone registered in AudioCaptures: the signal in blocks of the asked length, then (HoldOpen) nothing until the
    // reading is cancelled, as a quiet room; cancelling ends it quietly, as the library's captures do.
    private sealed class FakeCapture(float[] signal) : IAudioCapture
    {
        private int _opened, _blocks;

        public string Name => "fake-microphone";

        public bool HoldOpen { get; init; }

        // Called with each block's seconds as it is given (a test moves its clock on with the sound, as time passes for a microphone).
        public Action<double>? OnBlock { get; init; }

        public int Opened => Volatile.Read(ref _opened);

        public int Blocks => Volatile.Read(ref _blocks);

        public bool IsAvailable(out string reason)
        {
            reason = "";
            return true;
        }

        public IReadOnlyList<AudioInputDevice> Devices() => [new AudioInputDevice("0", "a test tone", true)];

        public IAudioInput Open(AudioCaptureOptions options)
        {
            options.Check();
            Interlocked.Increment(ref _opened);
            return new Input(this, signal, options);
        }

        private sealed class Input(FakeCapture capture, float[] signal, AudioCaptureOptions options) : IAudioInput
        {
            public string Description => "a test tone";

            public int SampleRate => options.SampleRate;

            public int Channels => 1;

            public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                int block = Math.Max(1, (int)Math.Round(options.BlockSeconds * options.SampleRate));
                for (int start = 0; start < signal.Length && !cancellationToken.IsCancellationRequested; start += block)
                {
                    Interlocked.Increment(ref capture._blocks);
                    capture.OnBlock?.Invoke((double)Math.Min(block, signal.Length - start) / options.SampleRate);
                    yield return new AudioData(signal.AsSpan(start, Math.Min(block, signal.Length - start)).ToArray(), 1, options.SampleRate);
                    await Task.Yield();
                }

                if (capture.HoldOpen)
                {
                    try
                    {
                        await Task.Delay(Timeout.Infinite, cancellationToken);
                    }
                    catch (OperationCanceledException)
                    {
                        // Stopped: the input ends quietly.
                    }
                }
            }

            public void Dispose()
            {
            }
        }
    }

    private sealed class FakeFamily(string name) : ISpeechFamily
    {
        public string Name => name;

        public ISpeechRecognizer Read(SpeechCheckpoint checkpoint) => throw new NotSupportedException("the tests keep their recognizer in the host");
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

    // A clock moved by hand, safely from several threads.
    private sealed class SharedTime : TimeProvider
    {
        private long _ticks = 1_000_000;

        public override long TimestampFrequency => TimeSpan.TicksPerSecond;

        public override long GetTimestamp() => Interlocked.Read(ref _ticks);

        public override DateTimeOffset GetUtcNow() => new DateTimeOffset(2026, 10, 26, 0, 0, 0, TimeSpan.Zero).AddTicks(GetTimestamp());

        public void Advance(TimeSpan time) => Interlocked.Add(ref _ticks, time.Ticks);
    }

    // One Transcribe page over fakes: a store with one speech model of the fake family (registered), the recognizer kept in
    // the service's "speech" host, the capture registered (an app's: ForThisMachine picks it first).
    private sealed class Lab : IDisposable
    {
        public Lab(ScriptedRecognizer recognizer, FakeCapture? capture, bool granted = true, IDispatcher? dispatcher = null, TimeProvider? time = null,
            TimeSpan? interval = null, IFilePicker? picker = null)
        {
            Recognizer = recognizer;
            Capture = capture;
            Folder = Temp("transcribe-lab");
            string model = Path.Combine(Folder, "models", "test", "speech", "abc123");
            Directory.CreateDirectory(model);
            File.WriteAllText(Path.Combine(model, "config.json"), $$"""{"architectures": ["{{FakeArchitecture}}"]}""");
            SpeechFamilies.Register(new FakeFamily(FakeArchitecture));
            if (capture is not null)
            {
                AudioCaptures.Register(capture);
            }

            Options = new IdrakMauiOptions { Device = "cpu" };
            Models = new ModelService(new ManualMemoryPressure(), Options);
            Models.Host<ISpeechRecognizer>(TranscribeViewModel.ModelHostName, _ => recognizer);
            Center = new ProgressCenter(interval ?? TimeSpan.FromMilliseconds(250), dispatcher, time);
            Permissions = new FakePermissions(granted);
            Store = new FakeStore(new StoredModel("test/speech", "abc123", model, 10, 1, true, DateTimeOffset.UnixEpoch));
            Vm = new TranscribeViewModel(Store, Models, Permissions, Share, Options, Settings, Center, Clipboard, picker);
            Vm.RefreshAsync().GetAwaiter().GetResult();
            Check(Vm.SpeechModels.Count == 1 && Vm.SelectedModel is { Architecture: FakeArchitecture }, $"the fake speech model listed: {Vm.ModelsText}");
            Check(Vm.LoadModelAsync().GetAwaiter().GetResult(), $"loaded: {Vm.Error}");
        }

        public ScriptedRecognizer Recognizer { get; }

        public FakeCapture? Capture { get; }

        public string Folder { get; }

        public IdrakMauiOptions Options { get; }

        public ModelService Models { get; }

        public ProgressCenter Center { get; }

        public FakePermissions Permissions { get; }

        public FakeShare Share { get; } = new();

        public FakeClipboard Clipboard { get; } = new();

        public MemorySettingsStore Settings { get; } = new();

        public FakeStore Store { get; }

        public TranscribeViewModel Vm { get; }

        public void Set(string key, string text)
        {
            Vm.Settings.Single(e => e.Setting.Key == key).Text = text;
            Check(Vm.ApplySettings(), $"{key}={text}: {Vm.SettingsMessage} {string.Join("; ", Vm.Settings.Where(e => e.HasError).Select(e => e.Error))}");
        }

        public void Dispose()
        {
            Vm.Dispose();
            Models.Dispose();
            SpeechFamilies.Unregister(FakeArchitecture);
            if (Capture is not null)
            {
                AudioCaptures.Unregister(Capture.Name);
            }
        }
    }

    private static void WaitFor(Func<bool> condition, string what)
    {
        var until = DateTime.UtcNow + TimeSpan.FromSeconds(20);
        while (!condition())
        {
            Check(DateTime.UtcNow < until, $"timed out waiting for {what}");
            Thread.Sleep(5);
        }
    }

    private static void TranscribeLive()
    {
        // Each reading moves the clock on a second (a showing is due after it) and takes 0.3 s, so a showing the blocks
        // took first is made by the page's timer before the next reading.
        var time = new SharedTime();
        var recognizer = new ScriptedRecognizer
        {
            OnCall = () =>
            {
                time.Advance(TimeSpan.FromSeconds(1));
                Thread.Sleep(300);
            },
        };
        var capture = new FakeCapture(Signal((2, 0.5f), (1, 0), (2, 0.2f), (1, 0))) { OnBlock = seconds => time.Advance(TimeSpan.FromSeconds(seconds)) };
        using var lab = new Lab(recognizer, capture, time: time);
        lab.Set("keep_up", "off");                                              // a step every second of sound: the readings come in order

        // The meter: a block's root mean square in decibels of full scale, its vector loop and scalar tail alike.
        var tone = Signal((0.1, 0.5f));
        double rms = Math.Sqrt(tone.Sum(x => (double)x * x) / tone.Length);
        foreach (int length in (int[])[tone.Length, 1001, 7])
        {
            double expected = 10 * Math.Log10(tone.AsSpan(0, length).ToArray().Sum(x => (double)x * x) / length);
            Check(Math.Abs(LevelMeter.Decibels(tone.AsSpan(0, length)) - expected) < 1e-3, $"{length} samples: {LevelMeter.Decibels(tone.AsSpan(0, length))}, expected {expected}");
        }

        Check(Math.Abs(LevelMeter.Decibels(tone) - 20 * Math.Log10(rms)) < 1e-3 && Math.Abs(LevelMeter.Decibels(tone) + 9.03) < 0.05, $"a tone at half scale: {LevelMeter.Decibels(tone)} dB");
        Check(LevelMeter.Decibels(new float[800]) == -120 && double.IsNegativeInfinity(LevelMeter.Decibels([])), "silence and nothing");
        var seen = new List<(string Committed, string Tentative, double Level, string Behind)>();
        lab.Vm.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(TranscribeViewModel.TentativeText) or nameof(TranscribeViewModel.Level))
            {
                lock (seen)
                {
                    seen.Add((lab.Vm.CommittedText, lab.Vm.TentativeText, lab.Vm.Level, lab.Vm.BehindText));
                }
            }
        };

        Check(lab.Vm.RecordAsync().GetAwaiter().GetResult(), $"recorded: {lab.Vm.Error}");
        Check(lab.Permissions.Requests == 1 && capture.Opened == 1, $"asked {lab.Permissions.Requests}, opened {capture.Opened}");
        lock (seen)
        {
            Check(seen.Any(s => s.Tentative.Length > 0), "no tentative words reached the view model");
            Check(seen.Any(s => s.Committed.StartsWith("alpha", StringComparison.Ordinal) && s.Tentative.Length > 0 && !s.Committed.Contains('\n', StringComparison.Ordinal)),
                $"no committed words before the first segment ended: {string.Join(" | ", seen.Select(s => $"'{s.Committed}' ~'{s.Tentative}'"))}");
            Check(seen.Any(s => s.Level > 0.5), "the level of the loud tone");
            Check(seen.Any(s => s.Behind.Length > 0), "how far behind");
        }

        string[] lines = lab.Vm.CommittedText.Split('\n');
        Check(lines.Length == 2 && lines[0].StartsWith("alpha beta", StringComparison.Ordinal) && lines[1].StartsWith("one two", StringComparison.Ordinal),
            $"the text: '{lab.Vm.CommittedText}'");
        Check(lab.Vm.TentativeText.Length == 0 && lab.Vm.Text == lab.Vm.CommittedText, $"tentative at the end: '{lab.Vm.TentativeText}'");
        var segments = lab.Vm.Segments;
        Check(segments.Count == 2 && segments[0].Text == lines[0] && segments[1].Text == lines[1] && segments[1].Start > segments[0].End - 0.01, $"segments: {segments.Count}");
        Check(lab.Vm.ElapsedText != "00:00" && lab.Vm.HeardText.Length > 0 && double.IsFinite(lab.Vm.LevelDecibels) && !lab.Vm.IsRunning && !lab.Vm.IsRecording,
            $"elapsed {lab.Vm.ElapsedText}, heard {lab.Vm.HeardText}, level {lab.Vm.LevelDecibels}");
        Check(recognizer.Heard.All(h => h.Rate == Rate), "the capture's rate");
    }

    private static void TranscribeBatched()
    {
        var dispatcher = new QueuedDispatcher();
        var time = new SharedTime();                                           // frozen: every report falls inside one interval
        var recognizer = new ScriptedRecognizer();
        var capture = new FakeCapture(Signal((3, 0.5f), (1, 0), (3, 0.2f)));
        using var lab = new Lab(recognizer, capture, dispatcher: dispatcher, time: time, interval: TimeSpan.FromHours(1));
        lab.Set("keep_up", "off");
        dispatcher.Run();                                                       // the model's load showed on its item
        int before = dispatcher.Dispatched;
        int changes = 0;
        lab.Vm.PropertyChanged += (_, _) => Interlocked.Increment(ref changes);
        Check(lab.Vm.RecordAsync().GetAwaiter().GetResult(), $"recorded: {lab.Vm.Error}");
        int dispatched = dispatcher.Dispatched - before;
        Check(capture.Blocks >= 140 && recognizer.Calls >= 4, $"{capture.Blocks} blocks, {recognizer.Calls} readings");
        Check(dispatched <= 2, $"{capture.Blocks} blocks and {recognizer.Calls} readings made {dispatched} UI dispatches");
        string text = lab.Vm.CommittedText;
        Check(text.Split('\n').Length == 2 && lab.Vm.Segments.Count == 2, $"the last values shown at the end: '{text}'");
        dispatcher.Run();                                                       // the showing that waited is older: nothing goes back
        Check(lab.Vm.CommittedText == text && lab.Vm.TentativeText.Length == 0, $"an older showing replaced the last: '{lab.Vm.CommittedText}'");
        Console.WriteLine($"    {capture.Blocks} blocks, {recognizer.Calls} readings: {dispatched} UI dispatch(es), {changes} property changes");
    }

    private static void TranscribeStop()
    {
        var time = new SharedTime();
        var recognizer = new ScriptedRecognizer { OnCall = () => time.Advance(TimeSpan.FromSeconds(1)) };
        var capture = new FakeCapture(Signal((2, 0.5f))) { HoldOpen = true };
        using var lab = new Lab(recognizer, capture, time: time);
        var run = lab.Vm.RecordAsync();
        WaitFor(() => lab.Vm.TentativeText.Length > 0, "the first reading's words");
        Check(lab.Vm.IsRecording && lab.Vm.RecordText == "Stop" && lab.Vm.Segments.Count == 0, "recording, nothing finished yet");
        lab.Vm.StopCommand.Execute(null);
        Check(run.Wait(TimeSpan.FromSeconds(20)) && run.Result, $"the run ended: {lab.Vm.Error}");
        Check(lab.Vm.Segments.Count == 1 && lab.Vm.CommittedText == lab.Vm.Segments[0].Text && lab.Vm.CommittedText.StartsWith("alpha beta", StringComparison.Ordinal)
              && lab.Vm.TentativeText.Length == 0, $"the last segment committed: '{lab.Vm.CommittedText}' ~'{lab.Vm.TentativeText}'");
        Check(!lab.Vm.IsRecording && lab.Vm.Status.StartsWith("Stopped", StringComparison.Ordinal), $"status {lab.Vm.Status}");

        // Pressing record again adds below; the record button stops it.
        int calls = recognizer.Calls;
        var again = lab.Vm.RecordAsync();
        WaitFor(() => recognizer.Calls > calls, "the second recording's first reading");
        lab.Vm.RecordCommand.Execute(null);
        Check(again.Wait(TimeSpan.FromSeconds(20)) && lab.Vm.Segments.Count == 2 && lab.Vm.CommittedText.Split('\n').Length == 2, $"two recordings: '{lab.Vm.CommittedText}'");
    }

    private static void TranscribePermission()
    {
        var capture = new FakeCapture(Signal((1, 0.5f)));
        using var lab = new Lab(new ScriptedRecognizer(), capture, granted: false);
        Check(!lab.Vm.RecordAsync().GetAwaiter().GetResult(), "recorded without the permission");
        Check(lab.Permissions.Requests == 1 && capture.Opened == 0 && lab.Recognizer.Calls == 0, $"asked {lab.Permissions.Requests}, opened {capture.Opened}");
        Check(lab.Vm.IsPermissionRefused && lab.Vm.Error.Contains("Allow it in the app's settings", StringComparison.Ordinal), $"error: {lab.Vm.Error}");
        lab.Vm.OpenPermissionSettingsCommand.Execute(null);
        Check(lab.Permissions.SettingsOpened == 1, "the app's settings opened");
    }

    private static void TranscribePipeline()
    {
        var capture = new FakeCapture(Signal((2, 0.5f), (1, 0)));
        using var lab = new Lab(new ScriptedRecognizer(), capture);
        var entry = lab.Vm.Settings.Single(e => e.Setting.Key == "pipeline");
        foreach (var (text, says) in new[] { ("normalize", "use agc instead"), ("trim", "whole recording"), ("nosuchstep", "registered") })
        {
            entry.Text = text;
            Check(!lab.Vm.ApplySettings() && entry.HasError && entry.Error.Contains(says, StringComparison.Ordinal) && lab.Vm.Options.Pipeline is null,
                $"{text}: {entry.Error}");
        }

        // Past the setting (an older save), a whole-recording step is still refused before the microphone opens.
        lab.Vm.Options.Pipeline = "normalize";
        Check(!lab.Vm.RecordAsync().GetAwaiter().GetResult() && lab.Vm.Error.Contains("whole recording", StringComparison.Ordinal) && capture.Opened == 0, $"error: {lab.Vm.Error}");

        var statuses = new List<string>();
        lab.Vm.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(TranscribeViewModel.Status))
            {
                lock (statuses)
                {
                    statuses.Add(lab.Vm.Status);
                }
            }
        };
        lab.Set("pipeline", "highpass=80, agc");
        Check(lab.Settings.Get(TranscribeSettings.Prefix + "pipeline") == "highpass=80, agc", "kept");
        Check(lab.Vm.RecordAsync().GetAwaiter().GetResult() && capture.Opened == 1 && lab.Vm.Segments.Count == 1, $"recorded through the pipeline: {lab.Vm.Error}");
        lock (statuses)
        {
            Check(statuses.Any(s => s.Contains("cleaned by highpass=80, agc", StringComparison.Ordinal)), $"statuses: {string.Join(" | ", statuses)}");
        }
    }

    private static void TranscribeFile()
    {
        string cache = Temp("transcribe-cache");
        try
        {
            ComputeResources.CacheFolder = cache;
            using var lab = new Lab(new ScriptedRecognizer(), capture: null);
            var signal = Signal((2, 0.5f), (1, 0), (1.5, 0.2f), (0.5, 0));
            string wav = Path.Combine(lab.Folder, "talk.wav");
            AudioEncoders.Save(wav, new AudioData(signal, 1, Rate));

            // Through the live path, in blocks, at its own pace.
            Check(lab.Vm.TranscribeFileAsync(wav).GetAwaiter().GetResult(), $"read live: {lab.Vm.Error}");
            Check(lab.Vm.Segments.Count == 2 && lab.Vm.CommittedText.StartsWith("alpha", StringComparison.Ordinal) && lab.Recognizer.Heard.All(h => h.Rate == Rate),
                $"live: '{lab.Vm.CommittedText}'");
            Check(lab.Permissions.Requests == 0, "a file asked for the microphone");

            // Whole: one reading of the decoded file.
            lab.Set("file_mode", "whole");
            lab.Vm.Clear();
            Check(!lab.Vm.HasText, "cleared");
            int calls = lab.Recognizer.Calls;
            Check(lab.Vm.TranscribeFileAsync(wav).GetAwaiter().GetResult(), $"read whole: {lab.Vm.Error}");
            var decoded = AudioCodecs.Decode(wav);
            var heard = lab.Recognizer.Heard[^1];
            Check(lab.Recognizer.Calls == calls + 1 && heard.Samples == signal.Length && heard.Samples == decoded.Length,
                $"whole: {lab.Recognizer.Calls - calls} readings of {heard.Samples} samples ({signal.Length} written)");
            string expected = new ScriptedRecognizer().Transcribe(decoded).Text;
            Check(lab.Vm.CommittedText == expected && lab.Vm.Segments.Count == 1, $"whole: '{lab.Vm.CommittedText}', the recognizer's '{expected}'");

            // Copy and share.
            Check(lab.Vm.CopyAsync().GetAwaiter().GetResult() && lab.Clipboard.Text == expected, $"copied '{lab.Clipboard.Text}'");
            Check(lab.Vm.ShareAsync().GetAwaiter().GetResult() && lab.Share.Shared is { } shared && shared.StartsWith(cache, StringComparison.Ordinal)
                  && File.ReadAllText(shared) == expected + "\n", $"shared {lab.Share.Shared}: {lab.Vm.Error}");

            // A file that is not sound.
            string bad = Path.Combine(lab.Folder, "notes.wav");
            File.WriteAllText(bad, "not a sound");
            Check(!lab.Vm.TranscribeFileAsync(bad).GetAwaiter().GetResult() && lab.Vm.Error.StartsWith("notes.wav could not be read", StringComparison.Ordinal), $"error: {lab.Vm.Error}");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
        }
    }

    private static void TranscribeModels()
    {
        string root = Temp("transcribe-models");
        StoredModel Model(string id, string? architecture, bool complete = true)
        {
            string folder = Path.Combine(root, id.Replace('/', '_'));
            Directory.CreateDirectory(folder);
            if (architecture is not null)
            {
                File.WriteAllText(Path.Combine(folder, "config.json"), $$"""{"architectures": ["{{architecture}}"]}""");
            }

            return new StoredModel(id, "rev", folder, 1, 1, complete, DateTimeOffset.UnixEpoch);
        }

        var store = new FakeStore(Model("a/speech", FakeArchitecture), Model("b/unfinished", FakeArchitecture, complete: false), Model("c/chat", "LlamaForCausalLM"),
            Model("d/none", null));
        var options = new IdrakMauiOptions { Device = "cpu" };
        using var models = new ModelService(new ManualMemoryPressure(), options);
        models.Host<ISpeechRecognizer>(TranscribeViewModel.ModelHostName, _ => new ScriptedRecognizer());
        var settings = new MemorySettingsStore();
        SpeechFamilies.Register(new FakeFamily(FakeArchitecture));
        try
        {
            using var vm = new TranscribeViewModel(store, models, new FakePermissions(true), new FakeShare(), options, settings);
            vm.RefreshAsync().GetAwaiter().GetResult();
            Check(vm.SpeechModels.Select(m => m.Title).SequenceEqual(["a/speech"]) && vm.HasSpeechModels && vm.ModelsText.StartsWith("1 speech model", StringComparison.Ordinal),
                $"listed: {string.Join(", ", vm.SpeechModels.Select(m => m.Title))}; {vm.ModelsText}");
            Check(vm.SelectedModel?.Key == "a/speech@rev" && settings.Get(TranscribeSettings.Prefix + "model") == "a/speech@rev", "picked and kept");
            Check(vm.LoadModelAsync().GetAwaiter().GetResult() && vm.HasLanguages && vm.Languages.SequenceEqual(["auto", "ar", "en"]), $"languages: {string.Join(", ", vm.Languages)}");
            vm.SelectedLanguage = "ar";
            vm.SelectedTask = "translate";
            var kept = TranscribeSettings.Load(settings, out _);
            Check(kept.Language == "ar" && kept.Task == "translate" && kept.Model == "a/speech@rev", "the language and task kept");
            Check(models.Loaded.Any(m => m.Host == TranscribeViewModel.ModelHostName && m.Model.Name == "cpu|" + store.List()[0].Folder), $"loaded once in the host: {string.Join(", ", models.Loaded.Select(m => m.Model.Name))}");
        }
        finally
        {
            SpeechFamilies.Unregister(FakeArchitecture);
        }

        using (var none = new TranscribeViewModel(store, models, new FakePermissions(true), new FakeShare(), options, new MemorySettingsStore()))
        {
            none.RefreshAsync().GetAwaiter().GetResult();
            Check(none.HasNoSpeechModels && none.ModelsText.StartsWith("No speech", StringComparison.Ordinal), $"no family for them: {none.ModelsText}");
            Check(!none.RecordAsync().GetAwaiter().GetResult() && none.Error.Contains("pull one in Models", StringComparison.Ordinal), $"error: {none.Error}");
        }

        // The feature: a library default after Devices and Models, needing the microphone; its page bound to its view model.
        Check(MobileFeatures.Get("transcribe") is TranscribeFeature { Order: 30 } feature && (feature.Needs & PhoneNeeds.Microphone) != 0
              && MobileFeatures.Origin("transcribe") == Overrides.Library, "the transcribe feature");
        using var services = new ServiceCollection().AddSingleton<IModelStore>(store).AddSingleton<IModelService>(models).AddSingleton(options)
            .AddSingleton<ISettingsStore>(new MemorySettingsStore()).AddSingleton<IPermissionService>(new FakePermissions(true)).AddSingleton<IShareService>(new FakeShare())
            .BuildServiceProvider();
        var page = MobileFeatures.Open("transcribe", services);
        Check(page is TranscribePage { BindingContext: TranscribeViewModel { OpenModels: not null } } && page.Title == "Transcribe", $"page {page.GetType().Name}, title {page.Title}");
    }

    // tiny-whisper (tests/Idrak.Tests/data/audio) through the lab on the CPU gives the library's own transcription of a file:
    // whole, and through the live path as one segment (as the library's live test reads it).
    private static void TranscribeWhisper()
    {
        string fixture = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "audio", "tiny-whisper");
        string wav = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "audio", "wav", "pcm16.wav");
        WhisperPlugin.Register();
        try
        {
            string expected;
            using (var direct = SpeechModels.Load(fixture, new SpeechModelOptions { Device = Device.Cpu }))
            {
                expected = direct.Transcribe(AudioCodecs.Decode(wav), new TranscriptionOptions { Language = "ar" }).Text.Trim();
            }

            var options = new IdrakMauiOptions { Device = "cpu" };
            using var models = new ModelService(new ManualMemoryPressure(), options);
            var store = new FakeStore(new StoredModel("test/tiny-whisper", "fixture", fixture, 1, 1, true, DateTimeOffset.UnixEpoch));
            using var vm = new TranscribeViewModel(store, models, new FakePermissions(true), new FakeShare(), options, new MemorySettingsStore());
            vm.RefreshAsync().GetAwaiter().GetResult();
            Check(vm.SpeechModels.Count == 1 && vm.LoadModelAsync().GetAwaiter().GetResult() && vm.Languages.Contains("ar"), $"tiny-whisper loaded: {vm.Error}; {vm.ModelsText}");
            vm.SelectedLanguage = "ar";

            vm.Options.FileMode = TranscribeFileMode.Whole;
            Check(vm.TranscribeFileAsync(wav).GetAwaiter().GetResult(), $"whole: {vm.Error}");
            Check(vm.CommittedText == expected, $"whole: '{vm.CommittedText}', the library's '{expected}'");

            vm.Clear();
            vm.Options.FileMode = TranscribeFileMode.Live;
            vm.Options.StepSeconds = 0.3;
            vm.Options.PreRollSeconds = 30;
            vm.Options.SilenceSeconds = 30;
            Check(vm.TranscribeFileAsync(wav).GetAwaiter().GetResult(), $"live: {vm.Error}");
            Check(vm.Segments.Count == 1 && vm.CommittedText == expected && vm.Segments[0].Start == 0, $"live: '{vm.CommittedText}' in {vm.Segments.Count} segment(s), the library's '{expected}'");
            Check(models.Loaded.Count == 1, $"one load: {string.Join(", ", models.Loaded.Select(m => m.Model.Name))}");
            Console.WriteLine($"    tiny-whisper through the lab: '{expected}', whole and live");
        }
        finally
        {
            WhisperPlugin.Unregister();
        }
    }
}
