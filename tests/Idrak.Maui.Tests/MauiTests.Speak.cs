// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.EspeakNg;
using Idrak.Audio.Plugins.Vits;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Speak lab and the app's speaker (plan 34, M6) on the CPU: a playback registered in AudioPlaybacks recording what it
// is given, a scripted synthesizer kept in the model service's "synthesis" host, a store of model folders written here;
// the espeak-ng reason through the library's TextNeeds; and the repository's tiny-vits end to end through the VITS family.
internal static partial class MauiTests
{
    private const string FakeSynthesis = "FakeSynthesisForMauiTests";

    // A speaker registered in AudioPlaybacks (an app's: ForThisMachine picks it first): every block written kept, the
    // outputs open at once counted; a write can wait on OnWrite (a slow device, or one held until the test lets go).
    private sealed class FakePlayback(string name = "fake-speaker") : IAudioPlayback
    {
        private int _open, _opened, _mostOpen;

        public string Name => name;

        public bool Available { get; init; } = true;

        public Func<AudioData, CancellationToken, Task>? OnWrite { get; set; }

        public List<string> Log { get; } = [];

        public List<AudioData> Written { get; } = [];

        public int Opened => Volatile.Read(ref _opened);

        public int MostOpen => Volatile.Read(ref _mostOpen);

        public int Drained { get; private set; }

        public int Disposed { get; private set; }

        public AudioPlaybackOptions? LastOptions { get; private set; }

        public float[] Samples()
        {
            lock (Written)
            {
                return [.. Written.SelectMany(b => b.Samples)];
            }
        }

        public bool IsAvailable(out string reason)
        {
            reason = Available ? "" : "a test speaker that is switched off";
            return Available;
        }

        public IReadOnlyList<AudioOutputDevice> Devices() => [new AudioOutputDevice("0", "a test speaker", true)];

        public IAudioOutput Open(AudioPlaybackOptions options)
        {
            options.Check();
            LastOptions = options;
            Interlocked.Increment(ref _opened);
            int open = Interlocked.Increment(ref _open);
            int most;
            while (open > (most = Volatile.Read(ref _mostOpen)) && Interlocked.CompareExchange(ref _mostOpen, open, most) != most)
            {
            }

            return new Output(this, options);
        }

        private void Note(string what)
        {
            lock (Log)
            {
                Log.Add(what);
            }
        }

        private sealed class Output(FakePlayback playback, AudioPlaybackOptions options) : IAudioOutput
        {
            private int _writes, _disposed;

            public string Description => $"{playback.Name} (test)";

            public int SampleRate => options.SampleRate ?? AudioPlaybackOptions.FallbackSampleRate;

            public int Channels => options.Channels ?? 1;

            public async ValueTask WriteAsync(AudioData block, CancellationToken cancellationToken = default)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                int n = ++_writes;
                playback.Note($"write {n} begins");
                if (playback.OnWrite is { } wait)
                {
                    await wait(block, cancellationToken).ConfigureAwait(false);
                }

                lock (playback.Written)
                {
                    playback.Written.Add(new AudioData(block.Samples.AsSpan().ToArray(), block.Channels, block.SampleRate));
                }

                playback.Note($"write {n} ends");
            }

            public Task DrainAsync(CancellationToken cancellationToken = default)
            {
                playback.Drained++;
                return Task.CompletedTask;
            }

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                {
                    playback.Disposed++;
                    Interlocked.Decrement(ref playback._open);
                }
            }
        }
    }

    // A synthesizer whose speech follows the text: each piece (SpeechSynthesisText.Pieces by the options' MaxCharacters)
    // a tone of 0.01 s a character at 16 kHz, louder for each piece; each piece can move the test's clock on.
    private sealed class ScriptedSynthesizer : ISpeechSynthesizer
    {
        private int _pieces;

        public Action? OnPiece { get; init; }

        public IReadOnlyList<string> Speakers { get; init; } = [];

        public int Pieces => Volatile.Read(ref _pieces);

        public List<SpeechSynthesisOptions> Asked { get; } = [];

        public Device Device => Device.Cpu;

        public int SampleRate => Rate;

        public IReadOnlyList<string> Voices => Speakers;

        public static AudioData Piece(string text, int index)
        {
            var samples = new float[Math.Max(1, text.Length) * Rate / 100];
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] = 0.1f * (index % 5 + 1) * MathF.Sin(2 * MathF.PI * 220 * i / Rate);
            }

            return new AudioData(samples, 1, Rate);
        }

        public (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        {
            var pieces = SpeechSynthesisText.Pieces(text, options?.MaxCharacters ?? 0).Select(Piece).ToList();
            return (new AudioData([.. pieces.SelectMany(p => p.Samples)], 1, Rate), new SynthesisStats(TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, 0, 0));
        }

        public async IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            lock (Asked)
            {
                Asked.Add(options ?? new SpeechSynthesisOptions());
            }

            int index = 0;
            foreach (string piece in SpeechSynthesisText.Pieces(text, options?.MaxCharacters ?? 0))
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
                OnPiece?.Invoke();
                Interlocked.Increment(ref _pieces);
                yield return Piece(piece, index++);
            }
        }

        public void Dispose()
        {
        }
    }

    private sealed class FakeSynthesisFamily(string name) : ISpeechSynthesisFamily
    {
        public string Name => name;

        public ISpeechSynthesizer Read(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options) =>
            throw new NotSupportedException("the tests keep their synthesizer in the host");
    }

    // One Speak page over fakes: a store with one synthesis model of the fake family (registered), the synthesizer kept in
    // the service's "synthesis" host, the speaker registered (an app's: ForThisMachine picks it first).
    private sealed class SpeakLab : IDisposable
    {
        public SpeakLab(ScriptedSynthesizer synthesizer, FakePlayback? playback = null, IDispatcher? dispatcher = null, TimeProvider? time = null, TimeSpan? interval = null)
        {
            Synthesizer = synthesizer;
            Playback = playback ?? new FakePlayback();
            Folder = Temp("speak-lab");
            string model = Path.Combine(Folder, "models", "test", "voice", "abc123");
            Directory.CreateDirectory(model);
            File.WriteAllText(Path.Combine(model, "config.json"), $$"""{"architectures": ["{{FakeSynthesis}}"]}""");
            SpeechSynthesisFamilies.Register(new FakeSynthesisFamily(FakeSynthesis));
            AudioPlaybacks.Register(Playback);
            Options = new IdrakMauiOptions { Device = "cpu" };
            Models = new ModelService(new ManualMemoryPressure(), Options);
            Models.Host<ISpeechSynthesizer>(SpeakViewModel.ModelHostName, _ => synthesizer);
            Center = new ProgressCenter(interval ?? TimeSpan.FromMilliseconds(250), dispatcher, time);
            Player = new SoundPlayer(time);
            Store = new FakeStore(new StoredModel("test/voice", "abc123", model, 10, 1, true, DateTimeOffset.UnixEpoch));
            Vm = new SpeakViewModel(Store, Models, Player, Share, Options, Settings, Center);
            Vm.RefreshAsync().GetAwaiter().GetResult();
            Check(Vm.SynthesisModels.Count == 1 && Vm.SelectedModel is { Architecture: FakeSynthesis, CanSpeak: true }, $"the fake synthesis model listed: {Vm.ModelsText}");
            Check(Vm.LoadModelAsync().GetAwaiter().GetResult(), $"loaded: {Vm.Error}");
        }

        public ScriptedSynthesizer Synthesizer { get; }

        public FakePlayback Playback { get; }

        public string Folder { get; }

        public IdrakMauiOptions Options { get; }

        public ModelService Models { get; }

        public ProgressCenter Center { get; }

        public SoundPlayer Player { get; }

        public FakeShare Share { get; } = new();

        public MemorySettingsStore Settings { get; } = new();

        public FakeStore Store { get; }

        public SpeakViewModel Vm { get; }

        public void Set(string key, string text)
        {
            Vm.Settings.Single(e => e.Setting.Key == key).Text = text;
            Check(Vm.ApplySettings(), $"{key}={text}: {Vm.SettingsMessage} {string.Join("; ", Vm.Settings.Where(e => e.HasError).Select(e => e.Error))}");
        }

        public void Dispose()
        {
            Vm.Dispose();
            Player.Dispose();
            Models.Dispose();
            SpeechSynthesisFamilies.Unregister(FakeSynthesis);
            AudioPlaybacks.Unregister(Playback.Name);
        }
    }

    // The blocks of `pieces`, each asked for noted in `log` as it is asked.
    private static async IAsyncEnumerable<AudioData> Logged(int pieces, List<string> log, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (int i = 1; i <= pieces; i++)
        {
            lock (log)
            {
                log.Add($"asked {i}");
            }

            await Task.Delay(5, cancellationToken);
            yield return ScriptedSynthesizer.Piece("abcdefghij", i);
        }
    }

    private static void SoundPlayed()
    {
        var playback = new FakePlayback();
        AudioPlaybacks.Register(playback);
        try
        {
            using var player = new SoundPlayer();
            var tone = ScriptedSynthesizer.Piece(new string('a', 50), 0);
            var played = player.PlayAsync(tone, new SoundPlayRequest { Label = "tone" }).GetAwaiter().GetResult();
            Check(played is { Completed: true, Blocks: 1, Label: "tone" } && played.FirstSound is { } first && first <= played.Elapsed && played.Output == "fake-speaker (test)"
                  && Math.Abs(played.Seconds - tone.Duration.TotalSeconds) < 1e-9, $"one recording: {played}");
            Check(playback.Drained == 1 && playback.Disposed == 1 && playback.Samples().AsSpan().SequenceEqual(tone.Samples) && !player.IsPlaying, "written, drained and closed");
            Check(playback.LastOptions == new AudioPlaybackOptions(), $"the library's output options: {playback.LastOptions}");

            // Blocks as they come: the next is asked for while one is written (a sentence made while the last plays).
            playback.Log.Clear();
            var log = playback.Log;
            playback.OnWrite = (_, token) => Task.Delay(30, token);
            var blocks = player.PlayAsync(Logged(3, log), new SoundPlayRequest { Output = new AudioPlaybackOptions { BufferSeconds = 0.2 } }).GetAwaiter().GetResult();
            List<string> order;
            lock (log)
            {
                order = [.. log];
            }

            Check(blocks is { Completed: true, Blocks: 3 } && order.IndexOf("asked 2") < order.IndexOf("write 1 ends") && order.IndexOf("asked 3") < order.IndexOf("write 2 ends"),
                $"the next block asked for while one is written: {string.Join(", ", order)}");
            Check(playback.LastOptions?.BufferSeconds == 0.2, "the request's output options");

            // One sound at a time: a play stops the one before (A/B); Stop stops it at once, the output closed.
            var held = new TaskCompletionSource();
            playback.OnWrite = (_, token) => held.Task.WaitAsync(token);
            var a = player.PlayAsync(tone, new SoundPlayRequest { Label = "a" });
            WaitFor(() => player.Playing == "a" && playback.Opened == 3, "a to play");
            playback.OnWrite = null;
            var b = player.PlayAsync(tone, new SoundPlayRequest { Label = "b" }).GetAwaiter().GetResult();
            Check(a.Wait(TimeSpan.FromSeconds(10)) && !a.Result.Completed && b.Completed && playback.MostOpen == 1, $"a stopped by b: {a.Result}, {b}; most open {playback.MostOpen}");
            playback.OnWrite = (_, token) => held.Task.WaitAsync(token);
            var c = player.PlayAsync(tone, new SoundPlayRequest { Label = "c" });
            WaitFor(() => player.Playing == "c", "c to play");
            Check(player.Stop() && c.Wait(TimeSpan.FromSeconds(10)) && !c.Result.Completed && !player.IsPlaying && playback.Disposed == playback.Opened, $"stopped: {c.Result}");
            Check(!player.Stop(), "nothing to stop");
            playback.OnWrite = null;

            // Refusals: a playback that cannot play here (its reason), one not registered, options out of range.
            var off = new FakePlayback("fake-off") { Available = false };
            AudioPlaybacks.Register(off);
            try
            {
                Check(Throws<NotSupportedException>(() => player.PlayAsync(tone, new SoundPlayRequest { Playback = "fake-off" }).GetAwaiter().GetResult(), "switched off"),
                    "a playback that cannot play here says why");
            }
            finally
            {
                AudioPlaybacks.Unregister("fake-off");
            }

            Check(Throws<NotRegisteredException>(() => player.PlayAsync(tone, new SoundPlayRequest { Playback = "no-such-speaker" }).GetAwaiter().GetResult(), "not registered"),
                "a playback not registered");
            Check(Throws<ArgumentException>(() => player.PlayAsync(tone, new SoundPlayRequest { Output = new AudioPlaybackOptions { BufferSeconds = 0 } }).GetAwaiter().GetResult(), "BufferSeconds"),
                "options out of range");

            // UseIdrak registers the one speaker.
            Check(AudioPlaybacks.ForThisMachine() == playback, "the app's playback first");
        }
        finally
        {
            AudioPlaybacks.Unregister(playback.Name);
        }
    }

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
            ["vocoder"] = Vocoders.DefaultName, ["voice_conversion"] = "off", ["tau"] = "0.1", ["voice_store"] = VoiceStores.FolderName, ["playback"] = AudioPlaybacks.Alsa,
            ["output_device"] = "hw:0,0", ["buffer_s"] = "0.25", ["block_s"] = "0.01", ["performance"] = "LowLatency", ["format"] = "mp3", ["bits"] = "24",
        };
        Check(texts.Keys.Order().SequenceEqual(SpeakSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new SpeakLabOptions { Model = "a/b@c", Voice = "alice" };
        foreach (var setting in SpeakSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        var mapped = set.ToSynthesis();
        Check(mapped.Rate == 1.5f && mapped.NoiseScale == 0 && mapped.DurationNoiseScale == 0.2f && mapped.Seed == 7 && mapped.MaxCharacters == 40 && mapped.DecoderSeconds == 2
              && mapped.SampleRate == 22050 && mapped.Vocoder == Vocoders.DefaultName && mapped.VoiceConversion == "off" && mapped.Conversion is { Tau: 0.1f, Seed: 7 }
              && mapped.VoiceStore == VoiceStores.FolderName && mapped.Voice == "alice", $"synthesis {mapped}");
        Check(set.ToPlayback() is { Playback: AudioPlaybacks.Alsa } p && p.Output is { Device: "hw:0,0", BufferSeconds: 0.25, BlockSeconds: 0.01, Performance: AudioPerformance.LowLatency },
            $"playback {set.ToPlayback()}");
        Check(set.SaveFormat == "mp3" && set.ToEncode().BitsPerSample == 24, "saved as");

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
                     ("rate", "0", "above 0"), ("noise", "-1", "0 or more"), ("pieces", "-3", "0 or more"), ("vocoder", "nosuch", "registered"),
                     ("voice_conversion", "nosuch", "auto, off"), ("playback", "nosuch", "registered"), ("performance", "fast", "LowLatency"), ("bits", "12", "one of"),
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

    private const string ThreeSentences = "The first sentence is here. A second one follows it! And is there a third?";

    private static void SpeakStreams()
    {
        string cache = Temp("speak-cache");
        ComputeResources.CacheFolder = cache;
        try
        {
            using var lab = new SpeakLab(new ScriptedSynthesizer());
            lab.Set("pieces", "30");
            lab.Vm.Text = ThreeSentences;
            var pieces = SpeechSynthesisText.Pieces(ThreeSentences, 30);
            Check(pieces.Count == 3, $"three pieces: {string.Join(" | ", pieces)}");
            Check(lab.Vm.SpeakAsync().GetAwaiter().GetResult(), $"spoken: {lab.Vm.Error}");
            var expected = pieces.Select(ScriptedSynthesizer.Piece).ToList();
            Check(lab.Playback.Written.Count == 3 && lab.Playback.Samples().AsSpan().SequenceEqual(expected.SelectMany(p => p.Samples).ToArray()) && lab.Playback.Drained == 1,
                $"{lab.Playback.Written.Count} pieces played");
            Check(lab.Synthesizer.Asked[^1].MaxCharacters == 30, "the settings' pieces asked for");
            Check(lab.Vm.Speech is { } speech && speech.Samples.AsSpan().SequenceEqual(lab.Playback.Samples()) && lab.Vm.HasSpeech, "the speech kept is what played");
            Check(lab.Vm.HasCold && lab.Vm.ColdText.StartsWith("Cold", StringComparison.Ordinal) && !lab.Vm.HasWarm && lab.Vm.FirstSound is not null && lab.Vm.RealTimeFactor > 0,
                $"cold: {lab.Vm.ColdText}; warm: {lab.Vm.WarmText}");
            Check(lab.Vm.Status.StartsWith("Done", StringComparison.Ordinal) && lab.Vm.ProgressLine.StartsWith("3 pieces", StringComparison.Ordinal), $"status {lab.Vm.Status}; {lab.Vm.ProgressLine}");

            Check(lab.Vm.SpeakAsync().GetAwaiter().GetResult() && lab.Vm.HasWarm && lab.Vm.WarmText.StartsWith("Warm", StringComparison.Ordinal) && lab.Vm.HasCold,
                $"warm after cold: {lab.Vm.WarmText}");

            // Played again, saved as WAV and MP3, shared.
            int written = lab.Playback.Written.Count;
            Check(lab.Vm.PlayAgainAsync().GetAwaiter().GetResult() && lab.Playback.Written.Count == written + 1, $"played again: {lab.Vm.Error}");
            string? wav = lab.Vm.SaveAsync("wav").GetAwaiter().GetResult();
            Check(wav is not null && wav.StartsWith(cache, StringComparison.Ordinal) && wav.EndsWith(".wav", StringComparison.Ordinal)
                  && AudioCodecs.Decode(wav) is { SampleRate: Rate, Channels: 1 } decoded && decoded.Length == lab.Vm.Speech!.Length, $"wav: {wav} {lab.Vm.Error}");
            Check(lab.Vm.CanSaveMp3, "the library writes MP3");
            string? mp3 = lab.Vm.SaveAsync("mp3").GetAwaiter().GetResult();
            Check(mp3 is not null && mp3.EndsWith(".mp3", StringComparison.Ordinal) && new FileInfo(mp3).Length > 0 && AudioCodecs.Decode(mp3).Duration.TotalSeconds > 0.5,
                $"mp3: {mp3} {lab.Vm.Error}");
            Check(lab.Vm.ShareAsync().GetAwaiter().GetResult() && lab.Share.Shared is { } shared && shared.EndsWith(".wav", StringComparison.Ordinal), $"shared {lab.Share.Shared}: {lab.Vm.Error}");

            // Nothing to say, and the voices of a model with speakers.
            lab.Vm.Text = "  ";
            Check(!lab.Vm.SpeakAsync().GetAwaiter().GetResult() && lab.Vm.Error.Contains("text to speak", StringComparison.Ordinal), $"empty: {lab.Vm.Error}");
            var voices = SpeakViewModel.ListVoices(new ScriptedSynthesizer { Speakers = ["p225", "p226"] }, new SpeechSynthesisOptions());
            Check(voices.Select(v => v.Name).SequenceEqual(["", "p225", "p226"]) && voices[0].Title.Contains("p225", StringComparison.Ordinal) && voices.All(v => v.CanSpeak),
                $"voices: {string.Join(", ", voices.Select(v => v.Title))}");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
        }
    }

    private static void SpeakBatched()
    {
        var dispatcher = new QueuedDispatcher();
        var time = new SharedTime();                                           // frozen: every piece falls inside one interval
        using var lab = new SpeakLab(new ScriptedSynthesizer(), dispatcher: dispatcher, time: time, interval: TimeSpan.FromHours(1));
        lab.Set("pieces", "12");
        lab.Vm.Text = string.Join(" ", Enumerable.Range(1, 60).Select(i => $"Piece {i}."));
        dispatcher.Run();
        int before = dispatcher.Dispatched, showings = lab.Vm.Showings;
        Check(lab.Vm.SpeakAsync().GetAwaiter().GetResult(), $"spoken: {lab.Vm.Error}");
        int dispatched = dispatcher.Dispatched - before;
        Check(lab.Synthesizer.Pieces >= 60 && lab.Playback.Written.Count == lab.Synthesizer.Pieces, $"{lab.Synthesizer.Pieces} pieces made, {lab.Playback.Written.Count} played");
        Check(dispatched <= 2 && lab.Vm.Showings - showings <= 2, $"{lab.Synthesizer.Pieces} pieces made {dispatched} UI dispatches ({lab.Vm.Showings - showings} showings)");
        Check(lab.Vm.ProgressLine.StartsWith($"{lab.Synthesizer.Pieces} pieces", StringComparison.Ordinal), $"the last values at the end: {lab.Vm.ProgressLine}");
        Console.WriteLine($"    {lab.Synthesizer.Pieces} pieces: {dispatched} UI dispatch(es)");
    }

    private static void SpeakStopped()
    {
        using var lab = new SpeakLab(new ScriptedSynthesizer());
        lab.Set("pieces", "30");
        var held = new TaskCompletionSource();
        int writes = 0;
        lab.Playback.OnWrite = (_, token) => Interlocked.Increment(ref writes) == 1 ? Task.CompletedTask : held.Task.WaitAsync(token);
        lab.Vm.Text = ThreeSentences;
        var run = lab.Vm.SpeakAsync();
        WaitFor(() => Volatile.Read(ref writes) >= 2, "the second piece to be written");
        Check(lab.Vm.IsSpeaking && lab.Vm.SpeakText == "Stop", "speaking");
        lab.Vm.SpeakCommand.Execute(null);                                      // the button stops it
        Check(run.Wait(TimeSpan.FromSeconds(20)) && !run.Result, $"stopped: {lab.Vm.Error}");
        Check(!lab.Vm.IsSpeaking && lab.Vm.Status.StartsWith("Stopped", StringComparison.Ordinal) && lab.Playback.Disposed == lab.Playback.Opened && lab.Playback.Drained == 0,
            $"status {lab.Vm.Status}; outputs {lab.Playback.Opened} opened, {lab.Playback.Disposed} closed");
        Check(lab.Vm.Speech is { } kept && kept.Length >= ScriptedSynthesizer.Piece("x", 0).Length && !lab.Vm.HasCold, $"what was made kept ({lab.Vm.Speech?.Length}); no measure of a stopped speech");
    }

    // Copies tiny-vits's config.json and vocab.json (and its weights when `weights`) into `folder`, with a tokenizer_config.json when given.
    private static string TinyVitsCopy(string folder, string? tokenizerConfig, bool weights = false)
    {
        string fixture = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "audio", "tiny-vits");
        Directory.CreateDirectory(folder);
        foreach (string file in weights ? (string[])["config.json", "vocab.json", "model.safetensors"] : ["config.json", "vocab.json"])
        {
            File.Copy(Path.Combine(fixture, file), Path.Combine(folder, file), overwrite: true);
        }

        if (tokenizerConfig is not null)
        {
            File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), tokenizerConfig);
        }

        return folder;
    }

    private static void SpeakEspeakReason()
    {
        Check(!EspeakNgPlugin.IsRegistered && TextPhonemizers.Names.Count == 0, $"nothing registers a phonemizer here: {string.Join(", ", TextPhonemizers.Names)}");
        VitsPlugin.RegisterFamily();
        string root = Temp("speak-espeak");
        var playback = new FakePlayback();
        AudioPlaybacks.Register(playback);
        var voices = new FolderVoiceStore(Path.Combine(root, "voices"), "maui-test-voices");
        VoiceStores.Register(voices);
        try
        {
            Check(!EspeakNgPlugin.IsRegistered && TextPhonemizers.Names.Count == 0, "the VITS family alone brings no espeak-ng");
            string plain = TinyVitsCopy(Path.Combine(root, "plain"), null, weights: true);
            string phonemes = TinyVitsCopy(Path.Combine(root, "phonemes"), """{"phonemize": true, "language": "en-us"}""", weights: true);
            var store = new FakeStore(new StoredModel("test/plain", "r1", plain, 1, 1, true, DateTimeOffset.UnixEpoch),
                new StoredModel("test/phonemes", "r1", phonemes, 1, 1, true, DateTimeOffset.UnixEpoch));
            var (listed, others) = SpeakViewModel.ListSynthesisModels(store);
            Check(listed.Count == 2 && others == 0, $"both listed, not hidden: {listed.Count}, {others} others");
            var reasoned = listed.Single(m => m.Title == "test/phonemes");
            Check(!reasoned.CanSpeak && reasoned.Reason is { } reason && reason.Contains("phonemizer", StringComparison.Ordinal) && reason.Contains("espeak-ng", StringComparison.Ordinal)
                  && reasoned.TextNeeds is { Phonemes: true, PhonemeLanguage: "en-us" } && reasoned.ToString().Contains("cannot speak here", StringComparison.Ordinal)
                  && reasoned.Details.Contains("reads phonemes (en-us)", StringComparison.Ordinal), $"the phonemize model: {reasoned.Reason}; {reasoned.Details}");
            Check(listed.Single(m => m.Title == "test/plain").CanSpeak, "the plain model speaks");

            // Installed voices: an embedding one made by a converter not registered, a checkpoint one that reads phonemes.
            Directory.CreateDirectory(Path.Combine(root, "voices", "ahmed"));
            File.WriteAllText(Path.Combine(root, "voices", "ahmed", "voice.json"), """{"kind": "embedding", "encoder": {"family": "openvoice"}, "languages": ["ar"]}""");
            TinyVitsCopy(Path.Combine(root, "voices", "tuned", "model"), """{"phonemize": true}""");
            File.WriteAllText(Path.Combine(root, "voices", "tuned", "voice.json"), """{"kind": "checkpoint", "checkpoint": "model"}""");
            var options = new IdrakMauiOptions { Device = "cpu" };
            using var models = new ModelService(new ManualMemoryPressure(), options);
            using var player = new SoundPlayer();
            var settings = new MemorySettingsStore();
            using var vm = new SpeakViewModel(store, models, player, new FakeShare(), options, settings);
            vm.Settings.Single(e => e.Setting.Key == "voice_store").Text = "maui-test-voices";
            Check(vm.ApplySettings(), vm.SettingsMessage);
            vm.RefreshAsync().GetAwaiter().GetResult();
            Check(vm.SynthesisModels.Count == 2 && vm.SelectedModel?.Title == "test/plain" && vm.ModelsText.Contains("1 of them cannot speak here", StringComparison.Ordinal),
                $"the one that speaks picked first: {vm.SelectedModel?.Title}; {vm.ModelsText}");
            Check(vm.LoadModelAsync().GetAwaiter().GetResult(), $"loaded: {vm.Error}");
            var embedding = vm.Voices.Single(v => v.Name == "ahmed");
            var checkpoint = vm.Voices.Single(v => v.Name == "tuned");
            Check(!embedding.CanSpeak && embedding.Reason!.Contains("'openvoice'", StringComparison.Ordinal) && embedding.Reason.Contains("VoiceConverters", StringComparison.Ordinal),
                $"the embedding voice: {embedding.Reason}");
            Check(!checkpoint.CanSpeak && checkpoint.Reason!.Contains("phonemizer", StringComparison.Ordinal), $"the checkpoint voice: {checkpoint.Reason}");
            vm.SelectedVoice = embedding;
            vm.Text = "abc";
            Check(vm.HasVoiceReason && !vm.SpeakAsync().GetAwaiter().GetResult() && vm.Error.Contains("openvoice", StringComparison.Ordinal) && playback.Opened == 0,
                $"the embedding voice refused before speaking: {vm.Error}");
            vm.SelectedVoice = vm.Voices[0];

            // The phonemize model picked: its reason shown, nothing loaded, speaking refused with the reason.
            vm.SelectedModel = vm.SynthesisModels.Single(m => m.Title == "test/phonemes");
            Check(vm.HasModelReason && vm.ModelReason.Contains("espeak-ng", StringComparison.Ordinal) && !vm.HasVoices, $"reason: {vm.ModelReason}");
            Check(!vm.SpeakAsync().GetAwaiter().GetResult() && vm.Error.Contains("phonemizer", StringComparison.Ordinal) && playback.Opened == 0, $"refused: {vm.Error}");
            Check(models.Loaded.All(m => !m.Model.Name.EndsWith(phonemes, StringComparison.Ordinal)), "the phonemize model was not loaded");
        }
        finally
        {
            VoiceStores.Unregister("maui-test-voices");
            AudioPlaybacks.Unregister(playback.Name);
            VitsPlugin.UnregisterFamily();
        }
    }

    // tiny-vits (tests/Idrak.Tests/data/audio) through the lab on the CPU: the speech played is the library's own for the
    // same text and options, piece by piece; the speech saved as WAV reads back the same.
    private static void SpeakTinyVits()
    {
        string fixture = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "audio", "tiny-vits");
        string text = (string)JsonNode.Parse(File.ReadAllText(Path.Combine(fixture, "reference", "input.json")))!["text"]!;
        string cache = Temp("speak-vits-cache");
        VitsPlugin.RegisterFamily();
        var playback = new FakePlayback();
        AudioPlaybacks.Register(playback);
        ComputeResources.CacheFolder = cache;
        try
        {
            var options = new IdrakMauiOptions { Device = "cpu" };
            using var models = new ModelService(new ManualMemoryPressure(), options);
            using var player = new SoundPlayer();
            var store = new FakeStore(new StoredModel("test/tiny-vits", "fixture", fixture, 1, 1, true, DateTimeOffset.UnixEpoch));
            var share = new FakeShare();
            using var vm = new SpeakViewModel(store, models, player, share, options, new MemorySettingsStore());
            vm.RefreshAsync().GetAwaiter().GetResult();
            Check(vm.SynthesisModels.Count == 1 && vm.SelectedModel is { CanSpeak: true, Architecture: VitsSpeechFamily.Architecture }, $"tiny-vits listed: {vm.ModelsText}");
            Check(vm.LoadModelAsync().GetAwaiter().GetResult() && vm.ModelStatus.Contains("16000 Hz", StringComparison.Ordinal), $"loaded: {vm.Error} {vm.ModelStatus}");
            foreach (var (key, value) in new[] { ("noise", "0"), ("duration_noise", "0"), ("pieces", "3") })
            {
                vm.Settings.Single(e => e.Setting.Key == key).Text = value;
            }

            Check(vm.ApplySettings(), vm.SettingsMessage);
            vm.Text = text;
            Check(vm.SpeakAsync().GetAwaiter().GetResult(), $"spoken: {vm.Error}");
            float[] expected;
            using (var direct = SpeechSynthesisModels.Load(fixture, new SpeechSynthesisLoadOptions { Device = Device.Cpu }))
            {
                expected = [.. direct.SpeakStream(text, vm.Options.ToSynthesis()).ToBlockingEnumerable().SelectMany(p => p.Samples)];
            }

            Check(expected.Length > 0 && playback.Samples().AsSpan().SequenceEqual(expected) && vm.Speech!.Samples.AsSpan().SequenceEqual(expected),
                $"the lab's speech is the library's: {playback.Samples().Length} samples played, {expected.Length} expected");
            Check(vm.HasCold && vm.RealTimeFactor > 0 && vm.FirstSound is not null, $"measured: {vm.ColdText}");
            string? wav = vm.SaveAsync("wav").GetAwaiter().GetResult();
            Check(wav is not null && AudioCodecs.Decode(wav).Length == expected.Length, $"saved: {wav} {vm.Error}");
            Check(vm.ShareAsync().GetAwaiter().GetResult() && share.Shared is not null, $"shared: {vm.Error}");
            Console.WriteLine($"    tiny-vits through the lab: {expected.Length} samples, {vm.ColdText}");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
            AudioPlaybacks.Unregister(playback.Name);
            VitsPlugin.UnregisterFamily();
        }
    }

    private static void SpeakPageOpens()
    {
        var feature = MobileFeatures.Get(MobileFeatures.Speak);
        Check(feature is SpeakFeature { Order: 40 } && feature.State == FeatureState.Ready && (feature.Needs & PhoneNeeds.Model) != 0 && MobileFeatures.Origin(MobileFeatures.Speak) == Overrides.Library
              && MobileFeatures.Place(feature) == 40, $"the speak feature: {feature.GetType().Name}");
        var options = new IdrakMauiOptions { Device = "cpu" };
        using var models = new ModelService(new ManualMemoryPressure(), options);
        using var services = new ServiceCollection().AddSingleton<IModelStore>(new FakeStore()).AddSingleton<IModelService>(models).AddSingleton(options)
            .AddSingleton<ISettingsStore>(new MemorySettingsStore()).AddSingleton<ISoundPlayer>(new SoundPlayer()).AddSingleton<IShareService>(new FakeShare())
            .BuildServiceProvider();
        var page = MobileFeatures.Open(MobileFeatures.Speak, services);
        Check(page is SpeakPage { BindingContext: SpeakViewModel { OpenModels: not null } } && page.Title == "Speak", $"page {page.GetType().Name}, title {page.Title}");
        using var none = (SpeakViewModel)feature.CreateViewModel(services);       // one not bound to a page (no UI thread here)
        none.RefreshAsync().GetAwaiter().GetResult();
        Check(none.HasNoModels && (none.ModelsText.StartsWith("No", StringComparison.Ordinal)), $"none: {none.ModelsText}");
        none.Text = "hello";
        Check(!none.SpeakAsync().GetAwaiter().GetResult() && none.Error.Contains("pull one in Models", StringComparison.Ordinal), $"error: {none.Error}");
        ThemeKeysOnly(Path.Combine("src", "Idrak.Maui", "Views", "SpeakPage.xaml"));
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
