// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 30: live transcription. The stream input, the library's LiveTranscriber on a scripted recognizer and on
// tiny-whisper, the audio captures (ALSA's null device where libasound is installed) and idrak transcribe --live.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioLiveGroup =
    [
        ("audio live: PcmStreamInput reads raw s16le stereo and f32le, a WAV header of unknown size with a chunk before its data, a stream shorter than a header, all through reads of three bytes at a time; frames split between reads wait for their rest", AudioLiveStream),
        ("audio live: LiveTranscriber on a scripted recognizer: quiet before speech never transcribed, words committed once two readings agree, a segment ended by a pause and the last by the end, the first segment's text the second's prompt; a long tone cut by the window", AudioLiveScripted),
        ("audio live: tiny-whisper over pcm16.wav given in 0.1 s blocks: the steps' updates and one finished segment whose text is the whole file's transcription", AudioLiveWhisper),
        ("audio live: the captures: winmm and alsa registered with their reasons, this machine's chosen; ALSA's null device recorded through the driver (16 kHz mono blocks, stopped by the token) where libasound is installed", AudioLiveCaptures),
        ("audio live: idrak transcribe --live FILE prints the segment with its times and writes -o SRT; --list-inputs lists the captures; a second source refused", AudioLiveCli),
    ];

    // A stream that gives at most `step` bytes a read.
    private sealed class ThreeByteStream(byte[] bytes, int step) : MemoryStream(bytes)
    {
        public override int Read(byte[] buffer, int offset, int count) => base.Read(buffer, offset, Math.Min(count, step));

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            base.ReadAsync(buffer[..Math.Min(buffer.Length, step)], cancellationToken);
    }

    private static List<AudioData> ReadAll(IAudioInput input)
    {
        var blocks = new List<AudioData>();
        var reading = Task.Run(async () =>
        {
            await foreach (var block in input.ReadAsync())
            {
                blocks.Add(block);
            }
        });
        Check(reading.Wait(TimeSpan.FromSeconds(30)), "the stream was not read within 30 s");
        return blocks;
    }

    // The blocks' samples of channel c, joined.
    private static float[] Joined(List<AudioData> blocks, int channel) => [.. blocks.SelectMany(b => b.Channel(channel).ToArray())];

    private static void AudioLiveStream(Device device)
    {
        _ = device;
        // Raw 16-bit stereo: 1000 frames, the right channel the left negated.
        var raw = new byte[1000 * 4];
        for (int f = 0; f < 1000; f++)
        {
            short left = (short)((f * 37 % 2000) - 1000);
            BinaryPrimitives.WriteInt16LittleEndian(raw.AsSpan(f * 4), left);
            BinaryPrimitives.WriteInt16LittleEndian(raw.AsSpan(f * 4 + 2), (short)-left);
        }

        using (var input = new PcmStreamInput(new ThreeByteStream(raw, 3), "raw", PcmSampleFormat.S16LE, 8000, 2, blockSeconds: 0.01))
        {
            var blocks = ReadAll(input);
            var leftOut = Joined(blocks, 0);
            var rightOut = Joined(blocks, 1);
            Check(leftOut.Length == 1000 && blocks.All(b => b.SampleRate == 8000 && b.Channels == 2) && input.Format == "s16le",
                $"{leftOut.Length} frames in {blocks.Count} blocks at {blocks[0].SampleRate} Hz, {input.Format}");
            for (int f = 0; f < 1000; f++)
            {
                float expected = ((f * 37 % 2000) - 1000) / 32768f;
                Check(leftOut[f] == expected && rightOut[f] == -expected, $"frame {f}: {leftOut[f]}, {rightOut[f]}; expected {expected}");
            }

            Check(blocks.Count == 13 && blocks[0].Length == 80, $"blocks of 80 frames: {string.Join(", ", blocks.Select(b => b.Length))}");
        }

        // Raw float32 mono.
        var floats = new byte[300 * 4];
        for (int i = 0; i < 300; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(floats.AsSpan(i * 4), MathF.Sin(i * 0.1f));
        }

        using (var input = new PcmStreamInput(new ThreeByteStream(floats, 3), "floats", PcmSampleFormat.F32LE, 16000, 1))
        {
            var samples = Joined(ReadAll(input), 0);
            Check(samples.Length == 300 && samples.Select((x, i) => x == MathF.Sin(i * 0.1f)).All(b => b) && input.Format == "f32le", $"{samples.Length} float samples");
        }

        // A WAV header with unknown sizes, a LIST chunk before the data, 22.05 kHz 16-bit mono.
        var wav = new List<byte>();
        wav.AddRange("RIFF"u8.ToArray());
        wav.AddRange(BitConverter.GetBytes(uint.MaxValue));
        wav.AddRange("WAVEfmt "u8.ToArray());
        wav.AddRange(BitConverter.GetBytes(16));
        wav.AddRange(BitConverter.GetBytes((ushort)1));
        wav.AddRange(BitConverter.GetBytes((ushort)1));
        wav.AddRange(BitConverter.GetBytes(22050));
        wav.AddRange(BitConverter.GetBytes(44100));
        wav.AddRange(BitConverter.GetBytes((ushort)2));
        wav.AddRange(BitConverter.GetBytes((ushort)16));
        wav.AddRange("LIST"u8.ToArray());
        wav.AddRange(BitConverter.GetBytes(5));
        wav.AddRange("INFOx\0"u8.ToArray());                                   // 5 bytes and the pad byte
        wav.AddRange("data"u8.ToArray());
        wav.AddRange(BitConverter.GetBytes(uint.MaxValue));
        for (short i = 0; i < 500; i++)
        {
            wav.AddRange(BitConverter.GetBytes((short)(i * 10)));
        }

        using (var input = new PcmStreamInput(new ThreeByteStream([.. wav], 3), "wav", PcmSampleFormat.F32LE, 16000, 2))
        {
            var samples = Joined(ReadAll(input), 0);
            Check(input.SampleRate == 22050 && input.Channels == 1 && input.Format == "s16le" && samples.Length == 500 && samples[499] == 4990 / 32768f,
                $"the WAV stream: {input.SampleRate} Hz, {input.Channels} channel(s), {input.Format}, {samples.Length} samples");
        }

        // Shorter than a header: the bytes are the sound.
        using (var input = new PcmStreamInput(new MemoryStream([1, 0, 2, 0]), "short"))
        {
            var samples = Joined(ReadAll(input), 0);
            Check(samples.SequenceEqual([1 / 32768f, 2 / 32768f]), $"four bytes give [{string.Join(", ", samples)}]");
        }
    }

    // Sound of tones and quiet: (seconds, loud) pieces at 16 kHz, given in 0.1 s blocks.
    private static async IAsyncEnumerable<AudioData> Pieces((double Seconds, bool Loud)[] pieces, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        const int Rate = 16000;
        var all = new List<float>();
        foreach (var (seconds, loud) in pieces)
        {
            int n = (int)(seconds * Rate);
            for (int i = 0; i < n; i++)
            {
                all.Add(loud ? 0.5f * MathF.Sin(2 * MathF.PI * 440 * i / Rate) : 0.0001f * ((i * 7919 % 13) - 6) / 6f);
            }
        }

        for (int start = 0; start < all.Count && !cancellationToken.IsCancellationRequested; start += Rate / 10)
        {
            yield return new AudioData([.. all.Skip(start).Take(Math.Min(Rate / 10, all.Count - start))], 1, Rate);
            await Task.Yield();
        }
    }

    // A recognizer whose text is one word per whole half second of loud sound it is given ("w1 w2 ..."), the last word
    // changing with each call (so only the words before it ever agree); it records each call's loud share and prompt.
    private sealed class ScriptedRecognizer : ISpeechRecognizer
    {
        public List<(double Seconds, double LoudSeconds, string? Prompt)> Calls { get; } = [];

        public Device Device => Device.Cpu;

        public IReadOnlySet<string> Languages { get; } = new HashSet<string>();

        public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            var samples = audio.Channel(0);
            int loud = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                loud += MathF.Abs(samples[i]) > 0.1f ? 1 : 0;
            }

            double loudSeconds = loud / (0.872 * audio.SampleRate);                // a 0.5 sine is above 0.1 for 87.2 % of its samples
            Calls.Add((audio.Duration.TotalSeconds, loudSeconds, options?.Prompt));
            int words = (int)(loudSeconds / 0.5 + 0.01);
            var text = string.Join(' ', Enumerable.Range(1, words).Select(w => w == words ? $"w{w}#{Calls.Count}" : $"w{w}"));
            return new Transcription(text, null, [], new TranscriptionStats(audio.Duration, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, 0, 1, 1));
        }

        public IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public void Dispose()
        {
        }
    }

    private static List<LiveTranscript> Updates(ISpeechRecognizer recognizer, IAsyncEnumerable<AudioData> sound, LiveTranscriptionOptions options)
    {
        var updates = new List<LiveTranscript>();
        var run = Task.Run(async () =>
        {
            await foreach (var update in LiveTranscriber.TranscribeAsync(recognizer, sound, options))
            {
                updates.Add(update);
            }
        });
        Check(run.Wait(TimeSpan.FromSeconds(120)), "the live transcription did not end within 120 s");
        return updates;
    }

    private static void AudioLiveScripted(Device device)
    {
        _ = device;
        var recognizer = new ScriptedRecognizer();
        var options = new LiveTranscriptionOptions { StepSeconds = 0.5, SilenceSeconds = 0.6, KeepUp = false, Transcription = new TranscriptionOptions { Prompt = "start" } };
        var updates = Updates(recognizer, Pieces([(1.5, false), (3, true), (1.2, false), (2, true)]), options);
        var segments = updates.Where(u => u.SegmentEnded).ToList();
        Check(recognizer.Calls.All(c => c.LoudSeconds > 0), $"a call on quiet only: {string.Join(", ", recognizer.Calls.Select(c => $"{c.Seconds:0.0}/{c.LoudSeconds:0.0}"))}");
        Check(segments.Count == 2, $"segments: {string.Join(" | ", segments.Select(s => $"{s.Start:0.0}-{s.End:0.0} '{s.Segment}'"))}");
        Check(segments[0].Segment!.StartsWith("w1 w2 w3 w4 w5", StringComparison.Ordinal) && segments[1].Segment!.StartsWith("w1 w2 w3", StringComparison.Ordinal),
            $"the segments' texts: '{segments[0].Segment}', '{segments[1].Segment}'");
        Check(segments[0].Start >= 1.0 && segments[0].Start <= 1.3 && segments[1].Start >= 4.4, $"the segments start at {segments[0].Start:0.00} and {segments[1].Start:0.00} s");
        var committed = updates.TakeWhile(u => !u.SegmentEnded).Select(u => u.Committed).Where(c => c.Length > 0).ToList();
        Check(committed.Count > 0 && string.Join(' ', committed).Split(' ').All(w => !w.Contains('#', StringComparison.Ordinal)),
            $"words committed before the first pause: [{string.Join(" | ", committed)}]");
        Check(string.Join(' ', committed.Append(segments[0].Committed).Where(c => c.Length > 0)) == segments[0].Segment,
            $"the committed words and the segment's rest make its text: [{string.Join(" | ", committed)}] + '{segments[0].Committed}' against '{segments[0].Segment}'");
        Check(recognizer.Calls[0].Prompt == "start" && recognizer.Calls[^1].Prompt!.EndsWith(segments[0].Segment!, StringComparison.Ordinal),
            $"prompts: first '{recognizer.Calls[0].Prompt}', last '{recognizer.Calls[^1].Prompt}'");

        var windowed = Updates(new ScriptedRecognizer(), Pieces([(6, true)]), options with { WindowSeconds = 2.5 });
        int cut = windowed.Count(u => u.SegmentEnded);
        Check(cut >= 3 && windowed.Where(u => u.SegmentEnded).All(u => u.End - u.Start <= 3.01), $"a 6 s tone in a 2.5 s window: {cut} segments");
        Console.WriteLine($"    {recognizer.Calls.Count} readings, {segments.Count} segments: '{segments[0].Segment}', '{segments[1].Segment}'; the window cut a 6 s tone into {cut}");
    }

    private static void AudioLiveWhisper(Device device)
    {
        using var recognizer = TinyWhisper(device);
        var audio = AudioCodecs.Decode(AudioData("wav/pcm16.wav"));
        var request = new TranscriptionOptions { Language = "ar" };
        string whole = recognizer.Transcribe(audio, request).Text.Trim();

        async IAsyncEnumerable<AudioData> Blocks()
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            int block = audio.SampleRate / 10;
            for (int start = 0; start < mono.Length; start += block)
            {
                yield return new AudioData(mono.AsSpan(start, Math.Min(block, mono.Length - start)).ToArray(), 1, audio.SampleRate);
                await Task.Yield();
            }
        }

        var updates = Updates(recognizer, Blocks(), new LiveTranscriptionOptions { Transcription = request, StepSeconds = 0.3, PreRollSeconds = 30, SilenceSeconds = 30, KeepUp = false });
        var segments = updates.Where(u => u.SegmentEnded).ToList();
        Check(segments.Count == 1 && segments[0].Segment == whole && segments[0].Start == 0, $"segments: {string.Join(" | ", segments.Select(s => $"{s.Start}-{s.End} '{s.Segment}'"))}; the whole file reads '{whole}'");
        Console.WriteLine($"    {updates.Count} updates; the segment '{segments[0].Segment}' as the whole file's");
    }

    private static void AudioLiveCaptures(Device device)
    {
        _ = device;
        Check(AudioCaptures.Names.Contains(AudioCaptures.Winmm) && AudioCaptures.Names.Contains(AudioCaptures.Alsa), $"captures: {string.Join(", ", AudioCaptures.Names)}");
        Check(AudioCaptures.Get("winmm").IsAvailable(out string winmm) == OperatingSystem.IsWindows(), $"winmm: {winmm}");
        var alsa = AudioCaptures.Get("alsa");
        bool alsaHere = alsa.IsAvailable(out string reason);
        Check(alsaHere || !OperatingSystem.IsLinux() || reason.Contains("libasound", StringComparison.Ordinal), $"alsa: {reason}");
        if (OperatingSystem.IsWindows() || alsaHere)
        {
            var chosen = AudioCaptures.ForThisMachine();
            Check(chosen.Name == (OperatingSystem.IsWindows() ? "winmm" : "alsa") && chosen.Devices().Count > 0 && chosen.Devices()[0].IsDefault,
                $"this machine's capture: {chosen.Name}, inputs {string.Join(", ", chosen.Devices().Select(d => d.Id))}");
        }
        else
        {
            ThrowsNaming<NotSupportedException>(() => AudioCaptures.ForThisMachine(), "no capture here", "No audio capture");
        }

        if (!alsaHere)
        {
            Console.WriteLine($"    alsa not here ({reason}): its null device not recorded");
            return;
        }

        using var input = alsa.Open(new AudioCaptureOptions { Device = "null", SampleRate = 16000, Channels = 1, BlockSeconds = 0.05 });
        using var stop = new CancellationTokenSource();
        var blocks = new List<AudioData>();
        var reading = Task.Run(async () =>
        {
            await foreach (var block in input.ReadAsync(stop.Token))
            {
                blocks.Add(block);
                if (blocks.Count == 5)
                {
                    stop.Cancel();
                }
            }
        });
        Check(reading.Wait(TimeSpan.FromSeconds(30)), "the null device was not read and stopped within 30 s");
        Check(blocks.Count >= 5 && blocks.All(b => b.SampleRate == 16000 && b.Channels == 1 && b.Length > 0), $"{blocks.Count} blocks: {string.Join(", ", blocks.Take(6).Select(b => $"{b.Length}@{b.SampleRate}"))}");
        Console.WriteLine($"    alsa's null device: {blocks.Count} blocks of {blocks[0].Length} samples, then stopped; inputs {string.Join(", ", alsa.Devices().Take(6).Select(d => d.Id))}");
    }

    private static void AudioLiveCli(Device device)
    {
        string root = TempFolder("live-cli");
        try
        {
            string srt = Path.Combine(root, "live.srt");
            var (code, output, error) = AudioCli("transcribe", AudioData("tiny-whisper"), "--live", AudioData("wav/pcm16.wav"), "-l", "ar", "--timestamps", "--step", "0.3",
                "-o", srt, "-d", device.ToString());
            Check(code == 0 && output.Contains("[00:00", StringComparison.Ordinal) && File.Exists(srt) && File.ReadAllText(srt).Contains("-->", StringComparison.Ordinal),
                $"transcribe --live: exit {code}\n{output}\n{error}");
            var listed = AudioCli("transcribe", "--list-inputs");
            Check(listed.Code == 0 && listed.Out.Contains("winmm", StringComparison.Ordinal) && listed.Out.Contains("alsa", StringComparison.Ordinal), $"--list-inputs: exit {listed.Code}\n{listed.Out}\n{listed.Err}");
            var two = AudioCli("transcribe", AudioData("tiny-whisper"), "--live", "a.wav", "b.wav");
            Check(two.Code != 0 && two.Err.Contains("one source", StringComparison.Ordinal), $"two sources: exit {two.Code}\n{two.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
