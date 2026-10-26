// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 34, phase M1: sound out. The playback registry beside the captures', a test speaker built on the library's own
// playback queue (the conversion proven exact against Resampler, the back-pressure, the drain, a failure and a stop),
// ALSA's null device played through the driver where libasound is installed, and AAudio's answers off Android.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioPlaybackGroup =
    [
        ("audio playback: the registries: winmm, alsa and aaudio playbacks and the aaudio capture registered, each unavailable off its own system with the reason; this machine's chosen; an app's playback chosen before the library's, then unregistered; an unknown name not registered", AudioPlaybackRegistry),
        ("audio playback: conversion is exact: 22.05 kHz stereo blocks of uneven sizes to 48 kHz mono are Resampler.Resample of the blocks joined and folded as ToMono, rounded to 16 bits; 24 kHz mono to 48 kHz stereo repeats it; a rate change ends the stream; 16-bit sound at the output's rate is queued as its bytes; three channels fold onto two in float; the vector and scalar 16-bit rounding agree", AudioPlaybackConversion),
        ("audio playback: a test speaker through AudioPlaybacks: a synthesis-like stream written block by block and drained comes out whole and in order; a write larger than the buffer waits while the device is held (back-pressure) and a drain waits for the device's own play-out; a second write at once is refused; a failed device's message reaches the writer; dispose ends a waiting write; the playing thread allocates nothing per block", AudioPlaybackSemantics),
        ("audio playback: ALSA's null device played through the driver (a 22.05 kHz tone to 48 kHz stereo in three blocks, drained, disposed) where libasound is installed; its outputs listed with the default first", AudioPlaybackAlsa),
        ("audio playback: aaudio off Android: capture and playback unavailable with the reason, Open refused naming it, no devices; a device is \"default\" or a number (a name refused); the microphone's failure names RECORD_AUDIO and AAudio.h's result", AudioPlaybackAaudio),
    ];

    // A speaker of the tests: the library's PlaybackQueue over a device that keeps what it is given. `Running` reset holds
    // the device (it takes nothing); a drain is logged before it is answered; FailAfter bytes the device fails.
    internal sealed class TestSpeaker(string name) : IAudioPlayback
    {
        public PcmSampleFormat Format { get; init; } = PcmSampleFormat.S16LE;

        public ManualResetEventSlim Running { get; } = new(true);

        public MemoryStream Played { get; } = new();

        public List<string> Events { get; } = [];

        public long FailAfter { get; set; } = long.MaxValue;

        // Bytes the playing thread allocated in its takes after the first few (the steady per-block path).
        public long TakeAllocated { get; private set; }

        public int Takes { get; private set; }

        public string Name => name;

        public bool IsAvailable(out string reason)
        {
            reason = "";
            return true;
        }

        public IReadOnlyList<AudioOutputDevice> Devices() => [new("default", "the test speaker", true)];

        public IAudioOutput Open(AudioPlaybackOptions options)
        {
            options.Check();
            int rate = options.SampleRate ?? AudioPlaybackOptions.FallbackSampleRate, channels = options.Channels ?? AudioPlaybackOptions.FallbackChannels;
            int frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * rate));
            var queue = new PlaybackQueue($"speaker (test, {name})", rate, channels, Format, options.BufferSeconds, options.Resample, q => Play(q, frames));
            queue.Start();
            return queue;
        }

        private void Play(PlaybackQueue queue, int frames)
        {
            var buffer = new byte[frames * queue.FrameBytes];
            while (!queue.Stopping)
            {
                if (!Running.Wait(20))
                {
                    continue;
                }

                long before = GC.GetAllocatedBytesForCurrentThread();
                int got = queue.Take(buffer, 100);
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                if (got == 0)
                {
                    if (queue.DrainPending && !queue.Stopping)
                    {
                        lock (Events)
                        {
                            Events.Add("drain");
                        }

                        Thread.Sleep(30);                                     // the device plays out what it holds
                        queue.Drained();
                    }

                    continue;
                }

                if (++Takes > 5)
                {
                    TakeAllocated += allocated;
                }

                lock (Played)
                {
                    if (Played.Length + got > FailAfter)
                    {
                        throw new InvalidOperationException("the test speaker was unplugged");
                    }

                    Played.Write(buffer, 0, got);
                }

                lock (Events)
                {
                    Events.Add("play");
                }
            }
        }

        public byte[] Bytes()
        {
            lock (Played)
            {
                return Played.ToArray();
            }
        }
    }

    internal static void Finish(Task task, string what, int seconds = 30) =>
        Check(task.Wait(TimeSpan.FromSeconds(seconds)), $"{what} did not finish within {seconds} s");

    internal static void FailsWith<T>(Func<Task> action, string what, string contains) where T : Exception
    {
        try
        {
            Finish(action(), what);
            Check(false, $"{what}: expected {typeof(T).Name}");
        }
        catch (Exception e) when ((e is AggregateException a ? a.InnerException : e) is T inner)
        {
            Check(inner.Message.Contains(contains, StringComparison.Ordinal), $"{what}: the message names '{contains}': {inner.Message}");
        }
    }

    internal static void WriteAll(IAudioOutput output, IEnumerable<AudioData> blocks)
    {
        Finish(Task.Run(async () =>
        {
            foreach (var block in blocks)
            {
                await output.WriteAsync(block);
            }

            await output.DrainAsync();
        }), $"writing and draining {output.Description}");
    }

    // Planar [channels, frames] floats as interleaved little-endian 16-bit samples, by the scalar rounding.
    internal static byte[] Interleaved16(float[] planar, int channels)
    {
        int frames = planar.Length / channels;
        var bytes = new byte[planar.Length * 2];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < channels; c++)
            {
                BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan((f * channels + c) * 2), PlaybackConverter.ToInt16(planar[c * frames + f]));
            }
        }

        return bytes;
    }

    // The first frame where two interleaved 16-bit streams differ, for messages.
    internal static string FirstDifference(byte[] played, byte[] expected, int channels)
    {
        int n = Math.Min(played.Length, expected.Length) / 2;
        for (int i = 0; i < n; i++)
        {
            if (played[2 * i] != expected[2 * i] || played[2 * i + 1] != expected[2 * i + 1])
            {
                return $"frame {i / channels} channel {i % channels}: {BinaryPrimitives.ReadInt16LittleEndian(played.AsSpan(2 * i))} played, "
                    + $"{BinaryPrimitives.ReadInt16LittleEndian(expected.AsSpan(2 * i))} expected";
            }
        }

        return $"{played.Length} bytes played, {expected.Length} expected";
    }

    // A tone with a second, quieter one, at `rate` over `channels` (channel c a little shifted), planar.
    internal static float[] Tones(int frames, int rate, int channels)
    {
        var planar = new float[frames * channels];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < frames; i++)
            {
                double t = (double)i / rate;
                planar[c * frames + i] = (float)(0.6 * Math.Sin(2 * Math.PI * 440 * t + c) + 0.3 * Math.Sin(2 * Math.PI * 3100 * t));
            }
        }

        return planar;
    }

    internal static List<AudioData> Blocks(AudioData whole, params int[] sizes)
    {
        var blocks = new List<AudioData>();
        int at = 0;
        foreach (int size in sizes)
        {
            blocks.Add(whole.Slice(at, size));
            at += size;
        }

        Check(at == whole.Length, $"the test's blocks cover {at} of {whole.Length} frames");
        return blocks;
    }

    internal static void AudioPlaybackRegistry(Device device)
    {
        _ = device;
        Check(AudioPlaybacks.Names.Contains(AudioPlaybacks.Winmm) && AudioPlaybacks.Names.Contains(AudioPlaybacks.Alsa) && AudioPlaybacks.Names.Contains(AudioPlaybacks.Aaudio),
            $"playbacks: {string.Join(", ", AudioPlaybacks.Names)}");
        Check(AudioCaptures.Names.Contains(AudioCaptures.Aaudio) && AudioCaptures.Origin(AudioCaptures.Aaudio) == Overrides.Library,
            $"captures: {string.Join(", ", AudioCaptures.Names)}; aaudio's origin {AudioCaptures.Origin(AudioCaptures.Aaudio)}");
        Check(AudioPlaybacks.Get("WINMM").IsAvailable(out string winmm) == OperatingSystem.IsWindows() && (OperatingSystem.IsWindows() || winmm.Contains("Windows", StringComparison.Ordinal)),
            $"winmm: {winmm}");
        bool alsaHere = AudioPlaybacks.Get("alsa").IsAvailable(out string alsa);
        Check(alsaHere || !OperatingSystem.IsLinux() || alsa.Contains("libasound", StringComparison.Ordinal), $"alsa: {alsa}");
        Check(AudioPlaybacks.Get("aaudio").IsAvailable(out string aaudio) == OperatingSystem.IsAndroid() && (OperatingSystem.IsAndroid() || aaudio.Contains("Android", StringComparison.Ordinal)),
            $"aaudio: {aaudio}");
        string? expected = OperatingSystem.IsWindows() ? "winmm" : OperatingSystem.IsAndroid() ? "aaudio" : alsaHere ? "alsa" : null;
        if (expected is not null)
        {
            var chosen = AudioPlaybacks.ForThisMachine();
            Check(chosen.Name == expected && chosen.Devices().Count > 0 && chosen.Devices()[0].IsDefault,
                $"this machine's playback: {chosen.Name} (expected {expected}), outputs {string.Join(", ", chosen.Devices().Select(d => d.Id))}");
        }
        else
        {
            ThrowsNaming<NotSupportedException>(() => AudioPlaybacks.ForThisMachine(), "no playback here", "No audio playback", "aaudio:");
        }

        if (OperatingSystem.IsAndroid())
        {
            Check(AudioCaptures.ForThisMachine().Name == "aaudio", $"Android's capture: {AudioCaptures.ForThisMachine().Name}");
        }

        var app = new TestSpeaker("test-speaker");
        AudioPlaybacks.Register(app);
        try
        {
            Check(AudioPlaybacks.ForThisMachine() == app && AudioPlaybacks.Origin("test-speaker") is { } origin && origin != Overrides.Library && AudioPlaybacks.Default("test-speaker") is null,
                $"the app's playback first: {AudioPlaybacks.ForThisMachine().Name}, origin {AudioPlaybacks.Origin("test-speaker")}");
        }
        finally
        {
            Check(AudioPlaybacks.Unregister("test-speaker") && AudioPlaybacks.Find("test-speaker") is null && !AudioPlaybacks.Unregister("alsa"),
                "the app's playback unregistered; the library's stays");
        }

        ThrowsNaming<NotSupportedException>(() => AudioPlaybacks.Get("pulse-direct"), "an unknown playback", "'pulse-direct' is not registered", "winmm");
        ThrowsNaming<ArgumentException>(() => new AudioPlaybackOptions { BufferSeconds = 0.01, BlockSeconds = 0.02 }.Check(), "a buffer under a block", "BufferSeconds");
        Console.WriteLine($"    playbacks {string.Join(", ", AudioPlaybacks.Names)}; alsa here: {alsaHere}; aaudio: {aaudio}");
    }

    internal static void AudioPlaybackConversion(Device device)
    {
        _ = device;

        // 22.05 kHz stereo to 48 kHz mono, in blocks of uneven sizes (one longer than a conversion chunk).
        var stereo = new AudioData(Tones(10_000, 22050, 2), 2, 22050);
        var speaker = new TestSpeaker("convert");
        using (var output = speaker.Open(new AudioPlaybackOptions { SampleRate = 48000, Channels = 1, BufferSeconds = 0.05 }))
        {
            WriteAll(output, Blocks(stereo, 1000, PlaybackConverter.ChunkFrames + 1, 3, stereo.Length - 1000 - (PlaybackConverter.ChunkFrames + 1) - 3));
        }

        var mono = new float[stereo.Length];
        stereo.ToMono(mono);
        var expected = Interleaved16(Resampler.Resample(new AudioData(mono, 1, 22050), 48000).Samples, 1);
        var played = speaker.Bytes();
        Check(played.AsSpan().SequenceEqual(expected), $"22.05 kHz stereo to 48 kHz mono: {FirstDifference(played, expected, 1)}");
        Check(speaker.Events[^1] == "drain", $"the drain came last: {string.Join(" ", speaker.Events.TakeLast(3))}");

        // 24 kHz mono to 48 kHz stereo: the resampled channel on both; then a block at 48 kHz ends that stream.
        var voice = new AudioData(Tones(7_000, 24000, 1), 1, 24000);
        var direct = new AudioData(Tones(500, 48000, 1), 1, 48000);
        var both = new TestSpeaker("repeat");
        using (var output = both.Open(new AudioPlaybackOptions { SampleRate = 48000, Channels = 2 }))
        {
            WriteAll(output, [.. Blocks(voice, 2500, 4500), direct]);
        }

        var resampled = Resampler.Resample(voice, 48000).Samples;
        float[] joined = [.. resampled, .. direct.Samples];
        var expectedStereo = Interleaved16([.. joined, .. joined], 2);
        played = both.Bytes();
        Check(played.AsSpan().SequenceEqual(expectedStereo), $"24 kHz mono to 48 kHz stereo, then 48 kHz: {FirstDifference(played, expectedStereo, 2)}");

        // 16-bit sound at the output's rate and channels: its bytes as they are.
        var pcm = new byte[2 * 2 * 3000];
        for (int i = 0; i < pcm.Length / 2; i++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(pcm.AsSpan(2 * i), (short)((i * 7919 % 65536) - 32768));
        }

        var bits = new TestSpeaker("bits");
        using (var output = bits.Open(new AudioPlaybackOptions { SampleRate = 44100, Channels = 2 }))
        {
            WriteAll(output, [new AudioData(pcm, 2, 44100)]);
        }

        Check(bits.Bytes().AsSpan().SequenceEqual(pcm), $"16-bit at the output's rate: {FirstDifference(bits.Bytes(), pcm, 2)}");

        // Three channels onto two, in float: channel 0 the mean of 0 and 2, channel 1 itself.
        var three = new AudioData(Tones(1500, 16000, 3), 3, 16000);
        var floats = new TestSpeaker("float") { Format = PcmSampleFormat.F32LE };
        using (var output = floats.Open(new AudioPlaybackOptions { SampleRate = 16000, Channels = 2 }))
        {
            WriteAll(output, Blocks(three, 700, 800));
        }

        var values = MemoryMarshal.Cast<byte, float>(floats.Bytes()).ToArray();
        Check(values.Length == 2 * three.Length, $"{values.Length} float samples for {three.Length} frames of two channels");
        for (int f = 0; f < three.Length; f++)
        {
            float left = (three.Channel(0)[f] + three.Channel(2)[f]) * (1f / 2), right = three.Channel(1)[f];
            Check(values[2 * f] == left && values[2 * f + 1] == right, $"frame {f}: {values[2 * f]}, {values[2 * f + 1]}; expected {left}, {right}");
        }

        // The vector and scalar roundings: halves to even, the range clamped.
        var samples = new float[1003];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = i % 3 == 0 ? (i - 500 + 0.5f) / 32768f : i % 3 == 1 ? (i - 500) / 300f : MathF.Sin(i);
        }

        var vector = new short[samples.Length];
        PlaybackConverter.ToInt16(samples, vector);
        for (int i = 0; i < samples.Length; i++)
        {
            short scalar = (short)Math.Clamp(MathF.Round(samples[i] * 32768f, MidpointRounding.ToEven), -32768f, 32767f);
            Check(vector[i] == scalar && PlaybackConverter.ToInt16(samples[i]) == scalar, $"sample {i} ({samples[i]}): {vector[i]} by vectors, {scalar} expected");
        }

        Check(PlaybackConverter.ToInt16(1.5f) == 32767 && PlaybackConverter.ToInt16(-1.5f) == -32768 && PlaybackConverter.ToInt16(-1f) == -32768
            && PlaybackConverter.ToInt16(12.5f / 32768f) == 12, "clamping and halves to even");
        Console.WriteLine($"    {played.Length / 4} stereo frames from 24 kHz exact; {expected.Length / 2} mono frames from 22.05 kHz stereo exact; 16-bit passed through; 3 -> 2 channels in float");
    }

    // A synthesizer-like stream: 0.1 s blocks of a 24 kHz voice, given as they are made.
    internal static async IAsyncEnumerable<AudioData> SpokenBlocks(AudioData voice, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (int start = 0; start < voice.Length && !cancellationToken.IsCancellationRequested; start += 2400)
        {
            await Task.Yield();
            yield return voice.Slice(start, Math.Min(2400, voice.Length - start));
        }
    }

    internal static void AudioPlaybackSemantics(Device device)
    {
        _ = device;

        // Through the registry, from a stream of blocks as synthesis gives them.
        var speaker = new TestSpeaker("test-pipeline");
        AudioPlaybacks.Register(speaker);
        var voice = new AudioData(Tones(24000, 24000, 1), 1, 24000);
        try
        {
            using var output = AudioPlaybacks.ForThisMachine().Open(new AudioPlaybackOptions { SampleRate = 48000, Channels = 1, BufferSeconds = 0.1 });
            Finish(Task.Run(async () =>
            {
                await foreach (var block in SpokenBlocks(voice))
                {
                    await output.WriteAsync(block);
                }

                await output.DrainAsync();
            }), "the stream through the test speaker");
            var expected = Interleaved16(Resampler.Resample(voice, 48000).Samples, 1);
            Check(speaker.Bytes().AsSpan().SequenceEqual(expected) && output.SampleRate == 48000 && output.Channels == 1 && output.Description.Contains("test-pipeline", StringComparison.Ordinal),
                $"a second of a 24 kHz voice at 48 kHz: {FirstDifference(speaker.Bytes(), expected, 1)}; {output.Description}");
            Check(speaker.Takes > 10 && speaker.TakeAllocated == 0, $"the playing thread allocated {speaker.TakeAllocated} bytes over {speaker.Takes - 5} steady takes");
        }
        finally
        {
            AudioPlaybacks.Unregister("test-pipeline");
        }

        // Back-pressure: 0.5 s into a 0.05 s buffer waits while the device is held, and goes on once it plays.
        var held = new TestSpeaker("held");
        held.Running.Reset();
        using (var output = held.Open(new AudioPlaybackOptions { SampleRate = 8000, Channels = 1, BlockSeconds = 0.01, BufferSeconds = 0.05 }))
        {
            var tone = new AudioData(Tones(4000, 8000, 1), 1, 8000);
            var write = output.WriteAsync(tone).AsTask();
            Check(!write.Wait(300) && held.Bytes().Length == 0, $"a write of 0.5 s into a 0.05 s buffer finished while the device was held: {write.Status}");
            FailsWith<InvalidOperationException>(() => output.WriteAsync(tone).AsTask(), "a second write while one waits", "await each before the next");
            held.Running.Set();
            Finish(write, "the held write once the device plays");

            // A drain waits for the device's own play-out, and the device is held again first.
            held.Running.Reset();
            Finish(output.WriteAsync(tone.Slice(0, 200)).AsTask(), "a small write");
            var drain = output.DrainAsync();
            Check(!drain.Wait(300), $"a drain finished while the device was held: {drain.Status}");
            held.Running.Set();
            Finish(drain, "the drain once the device plays");
            Check(held.Bytes().AsSpan().SequenceEqual(Interleaved16([.. tone.Samples, .. tone.Samples.AsSpan(0, 200)], 1)) && held.Events[^1] == "drain",
                $"the held speaker's sound: {held.Bytes().Length} bytes, last event {held.Events[^1]}");
        }

        // A device that fails: its message reaches the waiting writer, then every call.
        var failing = new TestSpeaker("failing") { FailAfter = 1000 };
        using (var output = failing.Open(new AudioPlaybackOptions { SampleRate = 8000, Channels = 1, BlockSeconds = 0.01, BufferSeconds = 0.05 }))
        {
            var tone = new AudioData(Tones(4000, 8000, 1), 1, 8000);
            FailsWith<InvalidOperationException>(() => output.WriteAsync(tone).AsTask(), "a write to a failing device", "the test speaker was unplugged");
            FailsWith<InvalidOperationException>(() => output.DrainAsync(), "a drain after the failure", "the test speaker was unplugged");
        }

        // Dispose ends a write that waits for room.
        var stopped = new TestSpeaker("stopped");
        stopped.Running.Reset();
        var stopping = stopped.Open(new AudioPlaybackOptions { SampleRate = 8000, Channels = 1, BlockSeconds = 0.01, BufferSeconds = 0.05 });
        var waiting = stopping.WriteAsync(new AudioData(Tones(4000, 8000, 1), 1, 8000)).AsTask();
        Check(!waiting.Wait(100), "the write to a held device finished");
        stopping.Dispose();
        FailsWith<ObjectDisposedException>(() => waiting, "a write waiting when the output is disposed", "stopped");
        FailsWith<ObjectDisposedException>(() => stopping.WriteAsync(new AudioData(new float[10], 1, 8000)).AsTask(), "a write after dispose", "speaker (test, stopped)");
        Console.WriteLine($"    a second of 24 kHz speech exact at 48 kHz through AudioPlaybacks ({speaker.Takes} takes, nothing allocated); back-pressure, drain, failure and dispose as described");
    }

    internal static void AudioPlaybackAlsa(Device device)
    {
        _ = device;
        var alsa = AudioPlaybacks.Get("alsa");
        if (!alsa.IsAvailable(out string reason))
        {
            Console.WriteLine($"    alsa not here ({reason}): its null device not played");
            return;
        }

        var outputs = alsa.Devices();
        Check(outputs.Count > 0 && outputs[0].Id == "default" && outputs[0].IsDefault && outputs.Skip(1).All(d => !d.IsDefault), $"outputs: {string.Join(", ", outputs.Select(d => d.Id))}");
        var tone = new AudioData(Tones(6615, 22050, 1), 1, 22050);
        using (var output = alsa.Open(new AudioPlaybackOptions { Device = "null", SampleRate = 48000, Channels = 2 }))
        {
            Check(output.SampleRate == 48000 && output.Channels == 2 && output.Description.Contains("null", StringComparison.Ordinal), $"{output.Description}: {output.SampleRate} Hz, {output.Channels} channel(s)");
            WriteAll(output, Blocks(tone, 2000, 2000, 2615));
        }

        ThrowsNaming<InvalidOperationException>(() => alsa.Open(new AudioPlaybackOptions { Device = "no-such-alsa-device" }).Dispose(), "an unknown ALSA name", "ALSA could not open 'no-such-alsa-device'");
        Console.WriteLine($"    alsa's null device: 0.3 s at 22.05 kHz played as 48 kHz stereo and drained; outputs {string.Join(", ", outputs.Take(6).Select(d => d.Id))}");
    }

    internal static void AudioPlaybackAaudio(Device device)
    {
        _ = device;
        var playback = AudioPlaybacks.Get("aaudio");
        var capture = AudioCaptures.Get("aaudio");
        Check(AAudioLibrary.DeviceId(null) == 0 && AAudioLibrary.DeviceId("default") == 0 && AAudioLibrary.DeviceId("12") == 12, "device ids: default is AAUDIO_UNSPECIFIED, a number itself");
        ThrowsNaming<ArgumentException>(() => AAudioLibrary.DeviceId("speaker"), "a device by name", "device id", "'speaker'");
        Check(AAudioLibrary.ResultName(-899) == "AAUDIO_ERROR_DISCONNECTED" && AAudioLibrary.ResultName(-885) == "AAUDIO_ERROR_TIMEOUT" && AAudioLibrary.ResultName(-880) == "AAUDIO_ERROR_INVALID_RATE"
            && AAudioLibrary.ResultName(-897) is null, "AAudio.h's results by number");
        string microphone = AAudioLibrary.Message("open the microphone", -896, input: true), speaker = AAudioLibrary.Message("open the speaker", -889, input: false);
        Check(microphone.Contains("AAUDIO_ERROR_INTERNAL (-896)", StringComparison.Ordinal) && microphone.Contains("RECORD_AUDIO", StringComparison.Ordinal)
            && speaker.Contains("AAUDIO_ERROR_UNAVAILABLE (-889)", StringComparison.Ordinal) && !speaker.Contains("RECORD_AUDIO", StringComparison.Ordinal),
            $"messages: '{microphone}'; '{speaker}'");
        if (OperatingSystem.IsAndroid())
        {
            Check(playback.IsAvailable(out string here) && capture.IsAvailable(out _), $"aaudio on Android: {here}");
            Console.WriteLine($"    aaudio here: outputs {string.Join(", ", playback.Devices().Select(d => d.Id))}");
            return;
        }

        Check(!playback.IsAvailable(out string reason) && reason.Contains("Android", StringComparison.Ordinal) && !capture.IsAvailable(out string why) && why == reason,
            $"aaudio off Android: '{reason}'");
        Check(playback.Devices().Count == 0 && capture.Devices().Count == 0, "aaudio lists no devices off Android");
        ThrowsNaming<NotSupportedException>(() => playback.Open(new AudioPlaybackOptions()), "aaudio's playback off Android", "aaudio playback cannot play here", "Android");
        ThrowsNaming<NotSupportedException>(() => capture.Open(new AudioCaptureOptions()), "aaudio's capture off Android", "aaudio capture cannot record here", "Android");
        Console.WriteLine($"    aaudio: {reason}; '{microphone}'");
    }
}
