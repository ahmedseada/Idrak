// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Mp3Audio;

// The MP3 plug-in's encoder (samples/Mp3Audio, Mp3Encoder): an IAudioEncoder registered in AudioEncoders. Its streams are
// decoded with the plug-in's own decoder; mp3enc/ holds files it wrote that ffmpeg decoded without an error when they
// were made (tools/audio/make_mp3_encoder_fixture.cs), with ffmpeg's lengths, SNRs and block RMS, so ffmpeg is not needed
// here. A decoder gives the sound Mp3Encoder.Delay (1057) samples late: the tests align by it.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioMp3EncodeGroup =
    [
        ("audio codecs: mp3 encode: tones, chirps and noise at every rate (MPEG-1 32, 44.1, 48 kHz; MPEG-2 16, 22.05, 24 kHz), mono and stereo, at the lowest, the default and the highest bit rate decode through the plug-in to whole frames covering the sound and the 1057-sample delay, constant frames of the bit rate, and a tone's SNR after the delay above its floor (128 kbit/s 44.1 kHz stereo and 64 kbit/s 16 kHz mono: 40 dB); the WAV fixture's signals too", d => { if (d == Device.Cpu) Mp3EncodeRoundTrip(); }),
        ("audio codecs: mp3 encode: the ffmpeg-checked fixture (mp3enc: all six rates, joint, plain and mixed-down stereo, 32 to 320 kbit/s, the three qualities) is what the encoder writes now, byte for byte, and the plug-in decodes it to ffmpeg's lengths, SNR and block RMS", d => { if (d == Device.Cpu) Mp3EncodeFixture(); }),
        ("audio codecs: mp3 encode: AudioEncoders.Save writes .mp3 once the plug-in is registered and refuses it, not registered, before and after; bit rates, sample rates, channels and the lowpass are checked against what MPEG takes (the message names the allowed ones); defaults, resampling, the mono mix and the IDRAK_MP3_* variables", d => { if (d == Device.Cpu) Mp3EncodeRegistryAndOptions(); }),
        ("audio codecs: mp3 encode: the same input gives the same bytes, in parallel and in order for every frames-per-task; three minutes of 44.1 kHz stereo and a minute of 16 kHz mono encode at the speed reported with little memory", d => { if (d == Device.Cpu) Mp3EncodeDeterminismAndSpeed(); }),
    ];

    private static readonly int[] Mp3Rates = [32000, 44100, 48000, 16000, 22050, 24000];

    private static AudioData Mp3Tone(int rate, int channels, int samples, float amplitude = 0.5f)
    {
        var s = new float[samples * channels];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < samples; i++)
            {
                s[c * samples + i] = amplitude * (float)Math.Sin(2 * Math.PI * (1000 + 500 * c) * i / rate);
            }
        }

        return new AudioData(s, channels, rate);
    }

    // 100 Hz up to 30% of the rate (inside the automatic lowpass at the default bit rates).
    private static AudioData Mp3Chirp(int rate, int channels, int samples)
    {
        var s = new float[samples * channels];
        double top = 0.3 * rate, seconds = (double)samples / rate;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < samples; i++)
            {
                double t = (double)i / rate;
                s[c * samples + i] = 0.5f * (float)Math.Sin(2 * Math.PI * (100 * t + 0.5 * (top - 100) / seconds * t * t) + c);
            }
        }

        return new AudioData(s, channels, rate);
    }

    private static AudioData Mp3Noise(int rate, int channels, int samples, float amplitude = 0.25f)
    {
        var s = new float[samples * channels];
        uint x = 2463534242;
        for (int i = 0; i < s.Length; i++)
        {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s[i] = amplitude * ((x >> 8) / 8388608f - 1f);
        }

        return new AudioData(s, channels, rate);
    }

    // The SNR of `decoded` against `source` from the encoder's delay on.
    private static double Mp3Snr(AudioData source, AudioData decoded)
    {
        double signal = 0, error = 0;
        int n = source.Length;
        for (int c = 0; c < source.Channels; c++)
        {
            for (int i = 0; i < n; i++)
            {
                double x = source.Samples[c * n + i], y = decoded.Samples[c * decoded.Length + i + Mp3Encoder.Delay];
                signal += x * x;
                error += (x - y) * (x - y);
            }
        }

        return 10 * Math.Log10(signal / Math.Max(error, 1e-30));
    }

    // Every frame of a plain stream (no tags): its header's version, rate, bit rate, mode and size, and that the sizes
    // follow the bit rate (the padding bit adds the one byte); the problem found, or null.
    private static string? Mp3FrameProblem(byte[] bytes, int rate, int channels, int kbps, out int frames, out int midSide)
    {
        int[] mpeg1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320], mpeg2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
        int[] rates1 = [44100, 48000, 32000], rates2 = [22050, 24000, 16000];
        frames = 0;
        midSide = 0;
        for (int at = 0; at < bytes.Length; frames++)
        {
            if (at + 4 > bytes.Length || bytes[at] != 0xFF || (bytes[at + 1] & 0xE6) != 0xE2)
            {
                return $"frame {frames} at byte {at}: no Layer III header";
            }

            bool isMpeg1 = (bytes[at + 1] & 0x18) == 0x18;
            int frameRate = (isMpeg1 ? rates1 : rates2)[(bytes[at + 2] >> 2) & 3], frameKbps = (isMpeg1 ? mpeg1 : mpeg2)[bytes[at + 2] >> 4];
            int padding = (bytes[at + 2] >> 1) & 1, mode = bytes[at + 3] >> 6;
            if (frameRate != rate || frameKbps != kbps || (mode == 3) != (channels == 1) || (bytes[at + 1] & 1) == 0)
            {
                return $"frame {frames}: {frameRate} Hz, {frameKbps} kbit/s, mode {mode}, protection bit {bytes[at + 1] & 1}; expected {rate} Hz, {kbps} kbit/s, {channels} channels, no CRC";
            }

            midSide += mode == 1 && (bytes[at + 3] & 0x20) != 0 ? 1 : 0;
            at += (int)((isMpeg1 ? 144000L : 72000L) * kbps / rate) + padding;
        }

        return null;
    }

    private static void Mp3EncodeRoundTrip()
    {
        var codec = new Mp3Codec();
        var report = new List<string>();
        foreach (int rate in Mp3Rates)
        {
            bool mpeg1 = rate >= 32000;
            int perFrame = mpeg1 ? 1152 : 576;
            foreach (int channels in new[] { 1, 2 })
            {
                int standard = Mp3EncodeOptions.DefaultBitrate(rate, channels);
                int[] bitrates = mpeg1 ? [32, standard, 320] : [8, standard, 160];
                // The tone's SNR floors at the lowest, the default and the highest rate (the encoder reached 31-82 dB,
                // 48-58 dB at the defaults; 8 kbit/s MPEG-2 stereo has a few bytes a frame for both channels: no floor).
                double[] floors = (mpeg1, channels) switch
                {
                    (true, 1) => [35, 40, 60],
                    (true, _) => [25, 40, 50],
                    (false, 1) => [20, 40, 55],
                    _ => [double.NegativeInfinity, 40, 40],
                };
                for (int k = 0; k < 3; k++)
                {
                    int kbps = bitrates[k];
                    var encoder = new Mp3Encoder(new Mp3EncodeOptions { Bitrate = kbps });
                    foreach (var (name, signal) in new[] { ("tone", Mp3Tone(rate, channels, rate)), ("chirp", Mp3Chirp(rate, channels, rate)), ("noise", Mp3Noise(rate, channels, rate)) })
                    {
                        string what = $"{name} at {rate} Hz, {channels} channel(s), {kbps} kbit/s";
                        byte[] bytes = encoder.Encode(signal);
                        string? problem = Mp3FrameProblem(bytes, rate, channels, kbps, out int frames, out _);
                        Check(problem is null, $"{what}: {problem}");
                        int expectedFrames = (signal.Length + Mp3Encoder.Delay + perFrame - 1) / perFrame;
                        double kbit = bytes.Length * 8.0 * rate / ((double)frames * perFrame) / 1000;
                        Check(frames == expectedFrames && Math.Abs(kbit - kbps) <= 0.01 * kbps + 0.05,
                            $"{what}: {frames} frames ({expectedFrames} cover the sound and the delay), {kbit:F2} kbit/s on average");
                        var decoded = codec.Decode(bytes);
                        Check(decoded.SampleRate == rate && decoded.Channels == channels && decoded.Length == frames * perFrame,
                            $"{what}: decoded {decoded.Channels} x {decoded.Length} at {decoded.SampleRate} Hz, expected {channels} x {frames * perFrame}");
                        double snr = Mp3Snr(signal, decoded);
                        if (name == "tone")
                        {
                            Check(snr >= floors[k], $"{what}: the tone's SNR is {snr:F1} dB, below its floor of {floors[k]} dB");
                            report.Add($"{rate / 1000.0:0.##}k {channels}ch {kbps}k {snr:F0}");
                        }
                        else if (k > 0)
                        {
                            double floor = name == "chirp" ? 30 : 3;                // the encoder reached 42-75 (chirp) and 3.9-34 dB (noise)
                            Check(snr >= floor, $"{what}: SNR {snr:F1} dB, below {floor} dB");
                        }
                    }
                }
            }
        }

        Console.WriteLine($"    tone SNR (dB): {string.Join(", ", report)}");

        // The WAV fixture's signals (a sweeping tone with a noise burst) at the default bit rates; rate8k is resampled to
        // 16 kHz, the nearest rate MP3 has (the floors are 6 to 10 dB under what the encoder reached).
        foreach (var (wav, floor) in new[] { ("pcm16", 25.0), ("stereo16", 30.0), ("rate44k", 14.0), ("rate8k", 30.0), ("float32", 25.0), ("extensible16", 30.0) })
        {
            var source = AudioCodecs.Decode(AudioData($"wav/{wav}.wav"));
            var decoded = codec.Decode(new Mp3Encoder().Encode(source));
            var expected = decoded.SampleRate == source.SampleRate ? source : Resampler.Resample(source, decoded.SampleRate);
            double snr = Mp3Snr(expected, decoded);
            Check(decoded.Channels == source.Channels && decoded.SampleRate == (source.SampleRate == 8000 ? 16000 : source.SampleRate) && snr >= floor,
                $"wav/{wav}: {decoded.Channels} x {decoded.Length} at {decoded.SampleRate} Hz, SNR {snr:F1} dB (floor {floor})");
        }
    }

    // The fixture's signal for an entry: a WAV's samples, with a second channel of them reversed and halved, labelled with
    // another rate, its first samples only (all exact), as tools/audio/make_mp3_encoder_fixture.cs takes them.
    private static AudioData Mp3FixtureSignal(JsonObject entry)
    {
        var audio = AudioCodecs.Decode(AudioData($"wav/{(string)entry["wav"]!}.wav"));
        int take = (int)entry["take"]!, rate = (int)entry["rate"]!;
        if (take > 0)
        {
            audio = audio.Slice(0, take);
        }

        var samples = audio.Samples;
        int n = audio.Length;
        if ((bool)entry["stereo"]!)
        {
            var both = new float[2 * n];
            samples.AsSpan(0, n).CopyTo(both);
            for (int i = 0; i < n; i++)
            {
                both[n + i] = 0.5f * samples[n - 1 - i];
            }

            return new AudioData(both, 2, rate);
        }

        return new AudioData([.. samples], audio.Channels, rate);
    }

    private static void Mp3EncodeFixture()
    {
        var files = JsonNode.Parse(File.ReadAllText(AudioData("mp3enc/files.json")))!.AsObject();
        Check(files.Count >= 10, $"the fixture lists {files.Count} files");
        var codec = new Mp3Codec();
        var rates = new HashSet<int>();
        foreach (var (name, node) in files)
        {
            var entry = node!.AsObject();
            var signal = Mp3FixtureSignal(entry);
            var options = new Mp3EncodeOptions
            {
                Bitrate = (int?)entry["bitrate"],
                ChannelMode = Enum.Parse<Mp3ChannelMode>((string)entry["channel_mode"]!),
                Quality = Enum.Parse<Mp3Quality>((string)entry["quality"]!),
            };
            byte[] stored = File.ReadAllBytes(AudioData($"mp3enc/{(string)entry["file"]!}"));
            byte[] encoded = new Mp3Encoder(options).Encode(signal);
            Check(encoded.Length == (int)entry["bytes"]! && Convert.ToHexStringLower(SHA256.HashData(encoded)) == (string)entry["sha256"]! && encoded.AsSpan().SequenceEqual(stored),
                $"{name}: the encoder writes {encoded.Length} bytes that differ from the ffmpeg-checked {(int)entry["bytes"]!} (first difference at {encoded.AsSpan().CommonPrefixLength(stored)}); "
                + "if the change is meant, regenerate with: dotnet run tools/audio/make_mp3_encoder_fixture.cs -c Release --no-cache");

            var decoded = codec.Decode(stored);
            int channels = (int)entry["ffmpeg_channels"]!;
            Check(decoded.Channels == channels && decoded.SampleRate == (int)entry["ffmpeg_rate"]! && decoded.Length == (int)entry["ffmpeg_samples"]!,
                $"{name}: the plug-in decodes {decoded.Channels} x {decoded.Length} at {decoded.SampleRate} Hz, ffmpeg {channels} x {(int)entry["ffmpeg_samples"]!} at {(int)entry["ffmpeg_rate"]!}");
            var expected = signal.Channels > channels ? MonoOf(signal) : signal;
            double snr = Mp3Snr(expected, decoded), ffmpegSnr = (double)entry["ffmpeg_snr_db"]!;
            Check(Math.Abs(snr - ffmpegSnr) < 0.01, $"{name}: the plug-in's decoding has an SNR of {snr:F3} dB, ffmpeg's {ffmpegSnr:F3}");
            var rms = entry["ffmpeg_block_rms"]!.AsArray();
            for (int c = 0; c < channels; c++)
            {
                var blocks = rms[c]!.AsArray();
                for (int b = 0; b < blocks.Count; b++)
                {
                    int from = b * 576, to = Math.Min(decoded.Length, from + 576);
                    double sum = 0;
                    for (int i = from; i < to; i++)
                    {
                        double v = decoded.Samples[c * decoded.Length + i];
                        sum += v * v;
                    }

                    double mine = Math.Sqrt(sum / (to - from)), theirs = (double)blocks[b]!;
                    Check(Math.Abs(mine - theirs) <= 1e-4, $"{name}: channel {c}'s block {b} has an RMS of {mine:F7}, ffmpeg's {theirs:F7}");
                }
            }

            rates.Add(decoded.SampleRate);
        }

        Check(rates.SetEquals(Mp3Rates), $"the fixture covers {string.Join(", ", rates.Order())} Hz");
    }

    private static AudioData MonoOf(AudioData audio)
    {
        var mono = new float[audio.Length];
        audio.ToMono(mono);
        return new AudioData(mono, 1, audio.SampleRate);
    }

    private static void Mp3EncodeRegistryAndOptions()
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-mp3enc-{Environment.ProcessId}");
        var tone = Mp3Tone(44100, 2, 44100);
        string path = Path.Combine(folder, "tone.mp3");
        var before = AudioEncoders.Names.ToArray();
        try
        {
            ThrowsNaming<NotSupportedException>(() => AudioEncoders.Save(path, tone), "an .mp3 without the plug-in", "'.mp3'", "not registered", "AudioEncoders.Register");
            Mp3AudioPlugin.Register(encodeOptions: new Mp3EncodeOptions());
            try
            {
                Check(AudioEncoders.For(path) is Mp3Encoder && AudioEncoders.Get("MP3") is Mp3Encoder && AudioEncoders.Origin("mp3") == typeof(Mp3Encoder).Assembly.GetName().Name
                      && AudioEncoders.Extensions.Contains(".mp3"), $"the plug-in's encoder writes .mp3: {string.Join(", ", AudioEncoders.Names)} ({AudioEncoders.Origin("mp3")})");
                AudioEncoders.Save(path, tone, new AudioEncodeOptions { BitsPerSample = 24 });   // a WAV's options: MP3 has no sample format and ignores them
                byte[] saved = File.ReadAllBytes(path);
                Check(saved.AsSpan().SequenceEqual(new Mp3Encoder().Encode(tone)), "AudioEncoders.Save writes the encoder's bytes");
                var read = AudioCodecs.Decode(path);
                Check(read.Channels == 2 && read.SampleRate == 44100 && Mp3Snr(tone, read) > 40, $"the saved file reads back through AudioCodecs: {read.Channels} x {read.Length}, SNR {Mp3Snr(tone, read):F1} dB");
                Check(Mp3FrameProblem(saved, 44100, 2, 128, out _, out _) is null, "44.1 kHz stereo is 128 kbit/s unless set");
            }
            finally
            {
                Check(Mp3AudioPlugin.Unregister(), "Unregister removes the registrations");
            }

            Check(AudioEncoders.Names.SequenceEqual(before), $"Unregister brings the library's encoders back: {string.Join(", ", AudioEncoders.Names)}");
            ThrowsNaming<NotSupportedException>(() => AudioEncoders.Save(path, tone), "an .mp3 after Unregister", "not registered");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        // Options checked against what MPEG takes, the message naming what it does take.
        var mono16 = Mp3Tone(16000, 1, 16000);
        ThrowsNaming<ArgumentException>(() => new Mp3Encoder(new Mp3EncodeOptions { Bitrate = 192 }).Encode(mono16), "192 kbit/s at 16 kHz",
            "8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160 kbit/s", "not 192", "MPEG-2");
        ThrowsNaming<ArgumentException>(() => new Mp3Encoder(new Mp3EncodeOptions { Bitrate = 8 }).Encode(tone), "8 kbit/s at 44.1 kHz",
            "32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbit/s", "not 8", "MPEG-1");
        ThrowsNaming<ArgumentException>(() => new Mp3Encoder(new Mp3EncodeOptions { Bitrate = 100 }).Encode(tone), "a bit rate MPEG has not", "not 100");
        ThrowsNaming<ArgumentException>(() => new Mp3Encoder(new Mp3EncodeOptions { SampleRate = 11025 }).Encode(tone), "an MPEG-2.5 rate", "16000, 22050, 24000, 32000, 44100, 48000", "not 11025");
        ThrowsNaming<ArgumentException>(() => new Mp3Encoder(new Mp3EncodeOptions { LowpassHz = -1 }).Encode(tone), "a negative lowpass", "LowpassHz");
        var three = new AudioData(new float[3 * 1000], 3, 44100);
        ThrowsNaming<NotSupportedException>(() => new Mp3Encoder().Encode(three), "three channels", "one or two channels", "Mono");
        Check(new Mp3Codec().Decode(new Mp3Encoder(new Mp3EncodeOptions { ChannelMode = Mp3ChannelMode.Mono }).Encode(three)).Channels == 1, "three channels mixed down with Mono");
        ThrowsNaming<ArgumentException>(() => Mp3EncodeOptions.Bitrates(8000), "the bit rates of 8 kHz", "not 8000");

        // Defaults: 64 kbit/s for 16 kHz mono, 96 for MPEG-2 stereo; the nearest rate MP3 has; the rate asked for.
        Check(Mp3FrameProblem(new Mp3Encoder().Encode(mono16), 16000, 1, 64, out _, out _) is null, "16 kHz mono is 64 kbit/s unless set");
        Check(Mp3FrameProblem(new Mp3Encoder().Encode(Mp3Tone(24000, 2, 24000)), 24000, 2, 96, out _, out _) is null, "24 kHz stereo is 96 kbit/s unless set");
        var codec = new Mp3Codec();
        foreach (var (from, to) in new[] { (8000, 16000), (11025, 16000), (96000, 48000), (40000, 44100), (30000, 32000) })
        {
            var decoded = codec.Decode(new Mp3Encoder().Encode(Mp3Tone(from, 1, from / 2)));
            Check(decoded.SampleRate == to, $"{from} Hz is written at the nearest rate MP3 has, {to}: {decoded.SampleRate}");
        }

        var resampled = codec.Decode(new Mp3Encoder(new Mp3EncodeOptions { SampleRate = 22050 }).Encode(tone));
        Check(resampled.SampleRate == 22050 && Mp3Snr(Resampler.Resample(tone, 22050), resampled) > 35, $"44.1 kHz written at 22.05 kHz: {resampled.SampleRate} Hz");

        // Channel modes: the mono mix is the mean of the channels; plain stereo never mid/side; joint stereo takes mid/side
        // where the channels are alike.
        var stereo = Mp3Chirp(44100, 2, 44100);
        Check(new Mp3Encoder(new Mp3EncodeOptions { ChannelMode = Mp3ChannelMode.Mono }).Encode(stereo).AsSpan().SequenceEqual(new Mp3Encoder().Encode(MonoOf(stereo))),
            "ChannelMode.Mono writes the mean of the channels");
        var alike = new float[2 * 44100];
        Mp3Chirp(44100, 1, 44100).Samples.CopyTo(alike, 0);
        Mp3Chirp(44100, 1, 44100).Samples.CopyTo(alike, 44100);
        alike[44100 + 5000] += 0.01f;
        var same = new AudioData(alike, 2, 44100);
        Check(Mp3FrameProblem(new Mp3Encoder().Encode(same), 44100, 2, 128, out int frames, out int midSide) is null && midSide > frames / 2,
            $"joint stereo codes two alike channels as mid/side: {midSide} of {frames} frames");
        Check(Mp3FrameProblem(new Mp3Encoder(new Mp3EncodeOptions { ChannelMode = Mp3ChannelMode.Stereo }).Encode(same), 44100, 2, 128, out _, out int none) is null && none == 0,
            "plain stereo codes left and right");
        var lowpassed = codec.Decode(new Mp3Encoder(new Mp3EncodeOptions { LowpassHz = 4000 }).Encode(Mp3Tone(44100, 1, 44100)));
        var open = codec.Decode(new Mp3Encoder(new Mp3EncodeOptions { LowpassHz = 0 }).Encode(Mp3Tone(44100, 1, 44100)));
        Check(Mp3Snr(Mp3Tone(44100, 1, 44100), lowpassed) > 30 && Mp3Snr(Mp3Tone(44100, 1, 44100), open) > 40, "a 1 kHz tone passes a 4 kHz lowpass and none");
        var cut = codec.Decode(new Mp3Encoder(new Mp3EncodeOptions { LowpassHz = 800 }).Encode(Mp3Tone(44100, 1, 44100)));
        Check(Mp3Snr(Mp3Tone(44100, 1, 44100), cut) < 1, "an 800 Hz lowpass takes the 1 kHz tone out");

        // The environment's settings (what Register uses when given none), and a value it does not take named.
        string[] names = ["IDRAK_MP3_BITRATE", "IDRAK_MP3_RATE", "IDRAK_MP3_CHANNELS", "IDRAK_MP3_QUALITY", "IDRAK_MP3_LOWPASS", "IDRAK_MP3_FRAMES_PER_TASK"];
        var saved2 = names.ToDictionary(n => n, Environment.GetEnvironmentVariable);
        try
        {
            foreach (string n in names)
            {
                Environment.SetEnvironmentVariable(n, null);
            }

            Check(Mp3EncodeOptions.FromEnvironment() == new Mp3EncodeOptions(), "no variables: the defaults");
            Environment.SetEnvironmentVariable("IDRAK_MP3_BITRATE", "96");
            Environment.SetEnvironmentVariable("IDRAK_MP3_RATE", "22050");
            Environment.SetEnvironmentVariable("IDRAK_MP3_CHANNELS", "mono");
            Environment.SetEnvironmentVariable("IDRAK_MP3_QUALITY", "high");
            Environment.SetEnvironmentVariable("IDRAK_MP3_LOWPASS", "0");
            Environment.SetEnvironmentVariable("IDRAK_MP3_FRAMES_PER_TASK", "4");
            var env = Mp3EncodeOptions.FromEnvironment();
            Check(env == new Mp3EncodeOptions { Bitrate = 96, SampleRate = 22050, ChannelMode = Mp3ChannelMode.Mono, Quality = Mp3Quality.High, LowpassHz = 0, FramesPerTask = 4 },
                $"the IDRAK_MP3_* variables: {env}");
            Environment.SetEnvironmentVariable("IDRAK_MP3_CHANNELS", "surround");
            ThrowsNaming<ArgumentException>(() => Mp3EncodeOptions.FromEnvironment(), "a channel mode it does not take", "IDRAK_MP3_CHANNELS", "surround");
            Environment.SetEnvironmentVariable("IDRAK_MP3_CHANNELS", null);
            Environment.SetEnvironmentVariable("IDRAK_MP3_BITRATE", "fast");
            ThrowsNaming<ArgumentException>(() => Mp3EncodeOptions.FromEnvironment(), "a bit rate that is not a number", "IDRAK_MP3_BITRATE", "fast");
        }
        finally
        {
            foreach (var (n, value) in saved2)
            {
                Environment.SetEnvironmentVariable(n, value);
            }
        }
    }

    private static void Mp3EncodeDeterminismAndSpeed()
    {
        // The same bytes twice, and for every chunking (in order on one thread, 1, 3, 32 and 1000 frames a task).
        foreach (var signal in new[] { Mp3Chirp(44100, 2, 5 * 44100), Mp3Noise(16000, 1, 5 * 16000), Mp3Noise(24000, 2, 5 * 24000) })
        {
            byte[] first = new Mp3Encoder().Encode(signal);
            Check(first.AsSpan().SequenceEqual(new Mp3Encoder().Encode(signal)), $"{signal.SampleRate} Hz x {signal.Channels}: the same bytes twice");
            foreach (int perTask in new[] { 0, 1, 3, 32, 1000 })
            {
                byte[] other = new Mp3Encoder(new Mp3EncodeOptions { FramesPerTask = perTask }).Encode(signal);
                Check(first.AsSpan().SequenceEqual(other), $"{signal.SampleRate} Hz x {signal.Channels}, {perTask} frames a task: the same bytes as 32 (first difference at {first.AsSpan().CommonPrefixLength(other)})");
            }

            foreach (var quality in new[] { Mp3Quality.Fast, Mp3Quality.High })
            {
                var decoded = new Mp3Codec().Decode(new Mp3Encoder(new Mp3EncodeOptions { Quality = quality }).Encode(signal));
                Check(decoded.Channels == signal.Channels && Mp3Snr(signal, decoded) > 3, $"{signal.SampleRate} Hz x {signal.Channels} at {quality}: SNR {Mp3Snr(signal, decoded):F1} dB");
            }
        }

        // Three minutes of 44.1 kHz stereo (a chirp with noise under it) and a minute of 16 kHz mono (the speech models'
        // rate), written to a stream: the time, and the bytes allocated beyond the output.
        var music = Mp3Chirp(44100, 2, 180 * 44100);
        var mix = music.Samples;
        uint x = 2463534242;
        for (int i = 0; i < mix.Length; i++)
        {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            mix[i] = 0.8f * mix[i] + 0.05f * ((x >> 8) / 8388608f - 1f);
        }

        var speech = Mp3Chirp(16000, 1, 60 * 16000);
        var encoder = new Mp3Encoder();
        var watch = Stopwatch.StartNew();
        encoder.Encode(speech, Stream.Null);
        double cold = watch.Elapsed.TotalMilliseconds;
        watch.Restart();
        encoder.Encode(speech, Stream.Null);
        double warm = watch.Elapsed.TotalMilliseconds;
        long before = GC.GetTotalAllocatedBytes(precise: true);
        watch.Restart();
        encoder.Encode(music, Stream.Null);
        double parallel = watch.Elapsed.TotalSeconds;
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        watch.Restart();
        new Mp3Encoder(new Mp3EncodeOptions { FramesPerTask = 0 }).Encode(music, Stream.Null);
        double serial = watch.Elapsed.TotalSeconds;
        Check(allocated < 16 << 20, $"three minutes allocated {allocated:N0} bytes");
        Console.WriteLine($"    180 s of 44.1 kHz stereo MP3 (128 kbit/s): {parallel * 1000:F0} ms on {ComputeResources.MaxCpuThreads} threads ({180 / parallel:F0}x real time), {serial * 1000:F0} ms in order; "
            + $"{allocated:N0} bytes allocated; 60 s of 16 kHz mono (64 kbit/s): {cold:F0} ms the first time, {warm:F0} ms after");
    }
}
