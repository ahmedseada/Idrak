// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.PluginTests;

// Plan 21, phase S1: AudioData and the sound codecs. The files come from tools/audio/make_tiny_audio.py (wav/).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioCodecGroup =
    [
        ("audio codecs: every WAV of the fixture (PCM 8 to 32 bits, float 32 and 64, stereo, extensible, a LIST chunk before the data, 8 and 44.1 kHz) decodes to its reference samples, its header is read from the first bytes, a data chunk larger than the file is cut to it, and damaged files are refused naming the file", AudioWavDecodes),
        ("audio codecs: the WAV encoder round-trips every bit depth and float within the depth's step, writes a 16-bit file's samples through unchanged, and extensible past two channels or 16 bits; AudioEncoders.Save writes by extension", AudioWavEncodes),
        ("audio codecs: a ten-minute file decodes into one block (no allocation per call beyond the samples), the registry names wav, and its contract suite passes on the fixture", AudioWavSuite),
        ("audio codecs: a codec registered by an outside assembly is asked first, decodes its own format through AudioCodecs, and Unregister brings the library's order back", AudioOutsideCodec),
        ("audio data: the vectorised 16-bit conversion equals the scalar one for mono and stereo, the 16-bit view ends on the first float read, ToMono is the channels' mean, Pad fills with silence and Slice cuts every channel", AudioDataConversions),
    ];

    private static string AudioData(string relative) => TestData(Path.Combine("audio", relative));

    private static (string Name, JsonObject Entry)[] WavFixtureFiles()
    {
        var files = JsonNode.Parse(File.ReadAllText(AudioData("wav/files.json")))!.AsObject();
        return [.. files.Select(p => (p.Key, p.Value!.AsObject()))];
    }

    private static void AudioWavDecodes(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host only: once
        }

        foreach (var (name, entry) in WavFixtureFiles())
        {
            string path = AudioData($"wav/{name}.wav");
            int rate = (int)entry["rate"]!, channels = (int)entry["channels"]!, samples = (int)entry["samples"]!;
            var info = AudioCodecs.ReadInfo(path);
            long claimed = name == "oversized16" ? 2L * samples : samples;        // its header claims twice what the file holds
            Check(info is { } i && i.SampleRate == rate && i.Channels == channels && i.Samples == claimed && i.Format == "wav",
                $"{name}: the header from the first bytes says {info}, the fixture {channels} x {claimed} at {rate} Hz");
            var audio = AudioCodecs.Decode(path);
            Check(audio.SampleRate == rate && audio.Channels == channels && audio.Length == samples, $"{name}: decoded {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz");
            var expected = ReadNpyFloat32(AudioData($"wav/{name}.npy"));
            bool isFloat = name.StartsWith("float", StringComparison.Ordinal);
            if (isFloat)
            {
                AssertClose(expected, audio.Samples, 1e-6f, $"{name}: the samples");
            }
            else
            {
                Check(expected.AsSpan().SequenceEqual(audio.Samples), $"{name}: PCM samples decode exactly (first difference at {expected.AsSpan().CommonPrefixLength(audio.Samples)})");
            }

            Check(AudioCodecs.CanDecode(path) && AudioCodecs.ReadInfo(File.ReadAllBytes(path)) == info, $"{name}: CanDecode and the in-memory header");
        }

        var oversized = AudioCodecs.ReadInfo(AudioData("wav/oversized16.wav"));
        Check(oversized is { Samples: 48000 } && AudioCodecs.Decode(AudioData("wav/oversized16.wav")).Length == 24000,
            $"a data chunk claiming twice the file: the header says so ({oversized}), the decoder reads what is there");

        foreach (var (file, text) in new[] { ("bad-truncated.wav", "fmt"), ("bad-no-fmt.wav", "fmt") })
        {
            string path = AudioData("wav/" + file);
            try
            {
                AudioCodecs.Decode(path);
                Check(false, $"{file} was decoded");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains(file, StringComparison.Ordinal) && e.Message.Contains(text, StringComparison.Ordinal), $"{file}: {e.Message}");
            }
        }

        Throws<InvalidDataException>(() => AudioCodecs.Decode("not a wav"u8.ToArray()), "bytes of no registered format are refused");
        Check(AudioCodecs.ReadInfo("RIFF"u8.ToArray()) is null, "a file shorter than a header is not a wav");
        Throws<NotSupportedException>(() => AudioCodecs.Get("ogg"), "an unregistered codec name");
    }

    private static void AudioWavEncodes(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var random = new Random(21);
        const int Length = 1001;
        var planar = new float[2 * Length];
        for (int i = 0; i < planar.Length; i++)
        {
            planar[i] = (float)(random.NextDouble() * 2 - 1);
        }

        planar[0] = 1f;
        planar[1] = -1f;
        var audio = new AudioData(planar, 2, 22050);
        var encoder = AudioEncoders.Get("wav");
        foreach (var (bits, isFloat) in new[] { (8, false), (16, false), (24, false), (32, false), (32, true) })
        {
            var bytes = encoder.Encode(audio, new AudioEncodeOptions { BitsPerSample = bits, Float = isFloat });
            var back = AudioCodecs.Decode(bytes);
            Check(back.SampleRate == 22050 && back.Channels == 2 && back.Length == Length, $"{bits}-bit{(isFloat ? " float" : "")}: {back.Channels} x {back.Length} at {back.SampleRate} Hz");
            float step = isFloat ? 0f : 1f / (1L << (bits - 1));
            float worst = 0f;
            for (int i = 0; i < planar.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(planar[i] - back.Samples[i]));
            }

            Check(worst <= step, $"{bits}-bit{(isFloat ? " float" : "")}: the round trip is within the step {step} (worst {worst})");
            ushort tag = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(20));
            Check(tag == (bits > 16 ? 0xFFFE : isFloat ? 3 : 1), $"{bits}-bit{(isFloat ? " float" : "")}: format tag {tag:X}");
        }

        var three = new AudioData(new float[3 * 10], 3, 8000);
        Check(BinaryPrimitives.ReadUInt16LittleEndian(encoder.Encode(three).AsSpan(20)) == 0xFFFE, "three channels are written extensible");

        // A 16-bit file's samples go through the encoder as they are: the data chunk of the copy is the original's.
        string source = AudioData("wav/stereo16.wav");
        var held = AudioCodecs.Decode(source);
        Check(held.TryGetSamples16(out var raw, out bool bigEndian) && !bigEndian && raw.Length == 2 * 2 * held.Length, "a 16-bit file is held as its 16-bit samples");
        var copy = encoder.Encode(held);
        var original = File.ReadAllBytes(source);
        Check(copy.AsSpan(copy.Length - raw.Length).SequenceEqual(original.AsSpan(original.Length - raw.Length)), "the copy's data chunk is the original's bytes");
        Check(AudioCodecs.Decode(copy).Samples.AsSpan().SequenceEqual(held.Samples), "and decodes to the same samples");

        string folder = TempFolder("audio-encode");
        try
        {
            string path = Path.Combine(folder, "sub", "out.wav");
            AudioEncoders.Save(path, audio, new AudioEncodeOptions { BitsPerSample = 24 });
            Check(AudioCodecs.ReadInfo(path) is { BitsPerSample: 24, Channels: 2 }, "Save writes by the extension, with the options");
            Throws<NotSupportedException>(() => AudioEncoders.Save(Path.Combine(folder, "out.ogg"), audio), "an extension no encoder writes");
            Throws<ArgumentException>(() => encoder.Encode(audio, new AudioEncodeOptions { BitsPerSample = 12 }), "12 bits are refused");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void AudioWavSuite(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        Check(AudioCodecs.Names.Contains("wav") && AudioCodecs.Extensions.Contains(".wav") && AudioCodecs.Origin("wav") == Overrides.Library && AudioEncoders.Origin("wav") == Overrides.Library,
            $"the registries name wav from the library: {string.Join(", ", AudioCodecs.Names)}; {string.Join(", ", AudioEncoders.Names)}");

        // Ten minutes of 16-bit mono: decoded as one block of its bytes, the file's own buffer pooled.
        const int Rate = 16000, Minutes = 10;
        int frames = Rate * 60 * Minutes;
        string folder = TempFolder("audio-long");
        try
        {
            string path = Path.Combine(folder, "long.wav");
            var pcm = new byte[frames * 2];
            var random = new Random(3);
            random.NextBytes(pcm);
            AudioEncoders.Save(path, new AudioData(pcm, 1, Rate));
            AudioCodecs.Decode(path);                                                // warm: the pool holds the file's block
            long before = GC.GetTotalAllocatedBytes(precise: true);
            var audio = AudioCodecs.Decode(path);
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            Check(audio.Length == frames && audio.TryGetSamples16(out var held, out _) && held.Span.SequenceEqual(pcm), "the long file decodes to its bytes");
            Check(allocated < pcm.Length + (1 << 16), $"decoding allocated {allocated:N0} bytes for {pcm.Length:N0} bytes of samples (the file's buffer is pooled)");
            Console.WriteLine($"    ten minutes of 16-bit mono: {allocated:N0} bytes allocated for {pcm.Length:N0} bytes of samples");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        var codec = AudioCodecs.Get("wav");
        var samples = new List<AudioSample>();
        foreach (var (name, entry) in WavFixtureFiles())
        {
            var expected = new DecodedAudio((int)entry["rate"]!, (int)entry["channels"]!, ReadNpyFloat32(AudioData($"wav/{name}.npy")));
            samples.Add(new AudioSample(AudioData($"wav/{name}.wav"), expected, name.StartsWith("float", StringComparison.Ordinal) ? 1e-6f : 0f));
        }

        samples.Add(new AudioSample(AudioData("wav/bad-truncated.wav"), Error: "fmt"));
        samples.Add(new AudioSample(AudioData("wav/bad-no-fmt.wav"), Error: "fmt"));
        var under = new AudioCodecUnderTest(
            bytes => codec.ReadInfo(bytes.Span) is { } info ? new AudioHeader(info.SampleRate, info.Channels, info.Samples) : null,
            bytes => { var a = codec.Decode(bytes.Span); return new DecodedAudio(a.SampleRate, a.Channels, a.Samples); });
        var report = Conformance.CheckAudioCodec(under, samples);
        Check(report.Passed, $"the audio codec suite on wav: {report}");
    }

    private static void AudioOutsideCodec(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var before = AudioCodecs.Names.ToArray();
        AudioPlugin.RegisterIdrakPlugin();
        try
        {
            Check(AudioCodecs.Names.First() == OutsideRawCodec.FormatName && AudioCodecs.Origin(OutsideRawCodec.FormatName) == typeof(AudioPlugin).Assembly.GetName().Name,
                $"the outside codec is asked first, from its assembly: {string.Join(", ", AudioCodecs.Names)} ({AudioCodecs.Origin(OutsideRawCodec.FormatName)})");
            float[] planar = [0.5f, -0.25f, 0.125f, 0f, 1f, -1f];
            var bytes = OutsideRawCodec.Encode(planar, 2, 48000);
            var decoded = AudioCodecs.Decode(bytes);
            Check(decoded.SampleRate == 48000 && decoded.Channels == 2 && decoded.Samples.AsSpan().SequenceEqual(planar), "its format decodes through AudioCodecs");
            Check(AudioCodecs.ReadInfo(bytes) is { Format: OutsideRawCodec.FormatName, Samples: 3 } && AudioCodecs.CanDecode("a.oraw"), "its header and extension are known");
            Check(AudioCodecs.Decode(AudioData("wav/pcm16.wav")).Length == 24000, "the library's wav still decodes behind it");
        }
        finally
        {
            AudioPlugin.Unregister();
        }

        Check(AudioCodecs.Names.SequenceEqual(before) && !AudioCodecs.CanDecode("a.oraw") && !AudioCodecs.Unregister(OutsideRawCodec.FormatName), "Unregister brings the order back");
    }

    private static void AudioDataConversions(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var random = new Random(9);
        foreach (int channels in new[] { 1, 2 })
        {
            const int Length = 2039;                                           // not a multiple of any vector width
            var pcm = new byte[Length * channels * 2];
            random.NextBytes(pcm);
            var expected = new float[channels * Length];
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < Length; i++)
                {
                    expected[c * Length + i] = BinaryPrimitives.ReadInt16LittleEndian(pcm.AsSpan((i * channels + c) * 2)) / 32768f;
                }
            }

            var audio = new AudioData(pcm, channels, 16000);
            Check(audio.TryGetSamples16(out _, out _) && audio.Length == Length && audio.Duration.TotalSeconds > 0.12, $"{channels} channel(s): held as 16-bit before the floats are read");
            Check(audio.Samples.AsSpan().SequenceEqual(expected), $"{channels} channel(s): the vectorised conversion equals the scalar one");
            Check(!audio.TryGetSamples16(out _, out _), $"{channels} channel(s): the 16-bit view ends on the first float read");

            var mono = new float[Length];
            audio.ToMono(mono);
            for (int i = 0; i < Length; i++)
            {
                float mean = 0f;
                for (int c = 0; c < channels; c++)
                {
                    mean += expected[c * Length + i];
                }

                mean *= 1f / channels;
                if (mean != mono[i])
                {
                    Check(false, $"{channels} channel(s): ToMono at {i}: {mono[i]}, expected {mean}");
                    break;
                }
            }

            var padded = new float[Length + 100];
            audio.Pad(channels - 1, padded);
            Check(padded.AsSpan(0, Length).SequenceEqual(audio.Channel(channels - 1)) && padded.AsSpan(Length).IndexOfAnyExcept(0f) < 0, $"{channels} channel(s): Pad copies the channel and fills with silence");
            var cut = audio.Slice(100, 50);
            Check(cut.Channels == channels && cut.Length == 50 && cut.Channel(0).SequenceEqual(audio.Channel(0).Slice(100, 50)) && cut.Channel(channels - 1).SequenceEqual(audio.Channel(channels - 1).Slice(100, 50)),
                $"{channels} channel(s): Slice cuts every channel");
        }

        var big = new AudioData([0x12, 0x34, 0x56, 0x78], 1, 8000, bigEndian: true);
        Check(big.Samples.SequenceEqual([0x1234 / 32768f, 0x5678 / 32768f]), "big-endian 16-bit samples read in their order");
        Throws<ArgumentException>(() => new AudioData(new byte[3], 1, 8000), "odd bytes are not 16-bit frames");
        Throws<ArgumentException>(() => new AudioData(new float[5], 2, 8000), "five samples are not stereo frames");
        Throws<ArgumentException>(() => AudioCodecs.Load(AudioData("wav/stereo16.wav"), 3), "channels are not invented");
        Check(AudioCodecs.Load(AudioData("wav/stereo16.wav"), 1).Channels == 1 && AudioCodecs.Load(AudioData("wav/stereo16.wav"), 2).Channels == 2, "Load mixes down to mono or keeps the file's channels");
    }
}
