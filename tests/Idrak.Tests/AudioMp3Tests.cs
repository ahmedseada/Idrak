// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Mp3Audio;

// The MP3 plug-in (samples/Mp3Audio): an IAudioCodec registered in AudioCodecs. The files come from
// tools/audio/make_tiny_audio.py (mp3/: LAME, shine and ffmpeg's Layer II encoder; the references are ffmpeg's decoding,
// checked there against mpg123's).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioMp3Group =
    [
        ("audio codecs: mp3: every MP3 and MP2 of the fixture (LAME and shine; MPEG-1, 2 and 2.5; mono, stereo and joint stereo; CBR and VBR; Layer II) decodes to ffmpeg's samples within 1e-4 and its exact length (a LAME or Lavc tag's delay and padding trimmed); its header is read from the first bytes", d => { if (d == Device.Cpu) AudioMp3Fixture(); }),
        ("audio codecs: mp3: ID3v2 and ID3v1 tags are skipped, a file cut inside a frame gives its whole frames, Gapless off gives every decoded sample, junk before the stream and a lost sync are passed over, Layer I and bytes of no format are refused, and the parallel decode equals decoding in order bit for bit", d => { if (d == Device.Cpu) AudioMp3Edges(); }),
        ("audio codecs: mp3: the plug-in registers mp3 in AudioCodecs (asked before the library's codecs; ChatAudio and AudioCodecs.Load read .mp3 and .mp2) and Unregister removes it; the codec suite passes on the fixture; minutes of sound decode into one block at the speed reported", d => { if (d == Device.Cpu) AudioMp3PluginAndSuite(); }),
    ];

    // ffmpeg's samples, clamped to [-1, 1] as AudioData's are (an MP3 overshoots full scale where the sound is loud).
    private static float[] Mp3Reference(string name) => [.. ReadNpyFloat32(AudioData($"mp3/{name}.npy")).Select(v => Math.Clamp(v, -1f, 1f))];

    private static (string Name, JsonObject Entry)[] Mp3FixtureFiles()
    {
        var files = JsonNode.Parse(File.ReadAllText(AudioData("mp3/files.json")))!.AsObject();
        return [.. files.Select(p => (p.Key, p.Value!.AsObject()))];
    }

    private static void AudioMp3Fixture()
    {
        var codec = new Mp3Codec();
        foreach (var (name, entry) in Mp3FixtureFiles())
        {
            string path = AudioData($"mp3/{(string)entry["file"]!}");
            int rate = (int)entry["rate"]!, channels = (int)entry["channels"]!, samples = (int)entry["samples"]!;
            byte[] bytes = File.ReadAllBytes(path);
            var info = codec.ReadInfo(bytes.AsSpan(0, Math.Min(bytes.Length, AudioCodecs.HeaderBytes)));
            Check(info is { } i && i.SampleRate == rate && i.Channels == channels && i.Format == "mp3" && (i.Samples == samples || i.Samples == -1),
                $"{name}: the header says {info}, the fixture {channels} x {samples} at {rate} Hz");
            Check(codec.ReadInfo(bytes) is { } whole && whole.Samples == samples, $"{name}: the whole file's header says {codec.ReadInfo(bytes)}, the fixture {samples} samples");
            var audio = codec.Decode(bytes);
            Check(audio.SampleRate == rate && audio.Channels == channels && audio.Length == samples, $"{name}: decoded {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz, ffmpeg {channels} x {samples}");
            var expected = Mp3Reference(name);
            AssertClose(expected, audio.Samples, 1e-4f, $"{name}: the samples against ffmpeg's (clamped to [-1, 1] as AudioData's are)");
        }
    }

    private static void AudioMp3Edges()
    {
        var codec = new Mp3Codec();
        byte[] joint = File.ReadAllBytes(AudioData("mp3/lame-joint44k.mp3"));
        var reference = codec.Decode(joint);

        // The tagged file is the same encoding with ID3v2 before it and ID3v1 after it.
        var tagged = codec.Decode(File.ReadAllBytes(AudioData("mp3/tagged44k.mp3")));
        Check(tagged.Samples.AsSpan().SequenceEqual(reference.Samples), "ID3v2 and ID3v1 tags are skipped: the tagged file decodes as the untagged");

        // Junk before the stream, and an ID3v2 tag of 70 000 bytes (past the 64 KiB the registry reads first).
        var junk = new byte[333];
        new Random(4).NextBytes(junk);
        junk[0] = 0;                                                             // not an ID3 start
        Check(codec.Decode([.. junk, .. joint]).Samples.AsSpan().SequenceEqual(reference.Samples), "junk before the first frame is passed over");
        var id3 = new byte[10 + 70_000];
        "ID3"u8.CopyTo(id3);
        id3[3] = 3;
        int body = 70_000;
        (id3[6], id3[7], id3[8], id3[9]) = ((byte)(body >> 21 & 0x7F), (byte)(body >> 14 & 0x7F), (byte)(body >> 7 & 0x7F), (byte)(body & 0x7F));
        byte[] bigTag = [.. id3, .. joint];
        Check(codec.ReadInfo(bigTag.AsSpan(0, AudioCodecs.HeaderBytes)) is { Samples: -1 } && codec.Decode(bigTag).Samples.AsSpan().SequenceEqual(reference.Samples),
            "an ID3v2 tag past the first 64 KiB: the header asks for the whole file, which decodes as the untagged");

        // Cut inside a frame: the whole frames before the cut, as the whole file gives them.
        var cut = codec.Decode(File.ReadAllBytes(AudioData("mp3/bad-truncated.mp3")));
        Check(cut.Length > 0 && cut.Length < reference.Length && cut.Channels == 2
              && reference.Samples.AsSpan(0, cut.Length).SequenceEqual(cut.Samples.AsSpan(0, cut.Length))
              && reference.Samples.AsSpan(reference.Length, cut.Length).SequenceEqual(cut.Samples.AsSpan(cut.Length, cut.Length)),
            $"a cut file gives its first {cut.Length} of {reference.Length} samples, as the whole file has them");

        // Gapless off: the tag's delay and padding kept (the encoder's frames whole).
        var all = new Mp3Codec(new Mp3Options { Gapless = false }).Decode(joint);
        Check(all.Length > reference.Length && all.Length % 1152 == 0, $"Gapless off gives every decoded sample: {all.Length} against {reference.Length}");

        // A lost sync: junk in the middle of a stream without a tag; the decoder finds the next frame again.
        byte[] notag = File.ReadAllBytes(AudioData("mp3/lame-notag44k.mp3"));
        var plain = codec.Decode(notag);
        int middle = FrameStart(notag, notag.Length / 2);
        byte[] broken = [.. notag.AsSpan(0, middle), .. junk, .. notag.AsSpan(middle)];
        var resynced = codec.Decode(broken);
        // The frame just before the junk has no next header to confirm it, so it goes (as minimp3 drops it); the rest is found again.
        Check(resynced.Channels == 2 && resynced.Length == plain.Length - 1152 && plain.Samples.AsSpan(0, 1152 * 4).SequenceEqual(resynced.Samples.AsSpan(0, 1152 * 4)),
            $"junk inside the stream: {resynced.Length} samples ({plain.Length} without it, one frame less), the frames before it unchanged");

        // Layer I, and bytes that are no MPEG audio.
        const int Layer1Bytes = 32;                                              // 12 · 32 kbps / 44.1 kHz slots of 4 bytes
        var layer1 = new byte[12 * Layer1Bytes];
        for (int f = 0; f < 12; f++)
        {
            layer1[f * Layer1Bytes] = 0xFF;
            layer1[f * Layer1Bytes + 1] = 0xFF;                                  // MPEG-1, Layer I, no CRC
            layer1[f * Layer1Bytes + 2] = 0x10;                                  // 32 kbps, 44.1 kHz
            layer1[f * Layer1Bytes + 3] = 0xC0;                                  // mono
        }

        ThrowsNaming<NotSupportedException>(() => codec.Decode(layer1), "Layer I", "Layer I");
        var noise = new byte[5000];
        new Random(8).NextBytes(noise);
        Check(codec.ReadInfo(noise) is null && codec.ReadInfo(File.ReadAllBytes(AudioData("wav/pcm16.wav"))) is null && codec.ReadInfo(File.ReadAllBytes(AudioData("flac/pcm16.flac"))) is null,
            "noise, a WAV and a FLAC are not MPEG audio");
        Throws<InvalidDataException>(() => codec.Decode(noise), "bytes of no MPEG frame are refused");

        // In parallel (a few frames per task, each task warming up from earlier frames) and in order: the same bits.
        foreach (var (name, entry) in Mp3FixtureFiles())
        {
            byte[] bytes = File.ReadAllBytes(AudioData($"mp3/{(string)entry["file"]!}"));
            var inOrder = new Mp3Codec(new Mp3Options { FramesPerTask = 0 }).Decode(bytes);
            var parallel = new Mp3Codec(new Mp3Options { FramesPerTask = 3 }).Decode(bytes);
            Check(inOrder.Samples.AsSpan().SequenceEqual(parallel.Samples), $"{name}: three frames per task and in order agree (first difference at {inOrder.Samples.AsSpan().CommonPrefixLength(parallel.Samples)})");
        }
    }

    // The offset of the first frame header at or after `from` that the next frame's header confirms.
    private static int FrameStart(byte[] bytes, int from)
    {
        for (int i = from; i + 4 < bytes.Length; i++)
        {
            if (bytes[i] == 0xFF && (bytes[i + 1] & 0xE0) == 0xE0 && new Mp3Codec().ReadInfo(bytes.AsSpan(i)) is not null)
            {
                return i;
            }
        }

        throw new InvalidOperationException("no frame found");
    }

    private static void AudioMp3PluginAndSuite()
    {
        var before = AudioCodecs.Names.ToArray();
        Throws<InvalidDataException>(() => AudioCodecs.Decode(AudioData("mp3/lame-joint44k.mp3")), "without the plug-in an MP3 is no registered format");
        Mp3AudioPlugin.Register();
        try
        {
            Check(AudioCodecs.Names.First() == "mp3" && AudioCodecs.Origin("mp3") == typeof(Mp3Codec).Assembly.GetName().Name && AudioCodecs.CanDecode("talk.mp3") && AudioCodecs.CanDecode("old.mp2"),
                $"the plug-in's mp3 is asked first: {string.Join(", ", AudioCodecs.Names)} ({AudioCodecs.Origin("mp3")})");
            var expected = Mp3Reference("lame-joint44k");
            AssertClose(expected, AudioCodecs.Decode(AudioData("mp3/lame-joint44k.mp3")).Samples, 1e-4f, "AudioCodecs.Decode reads an MP3 through the plug-in");
            Check(AudioCodecs.Load(AudioData("mp3/mp2-44k.mp2"), channels: 1).Channels == 1, "AudioCodecs.Load reads an MP2 and mixes it to one channel");
            var part = ChatAudio.FromFile(AudioData("mp3/lame-16k-mono.mp3"));
            Check(part.MediaType == "audio/mpeg" && part.Decode().Length == 24000, $"a chat part of an MP3 is audio/mpeg and decodes: {part}");
            Check(AudioCodecs.Decode(AudioData("wav/pcm16.wav")).Length == 24000 && AudioCodecs.Decode(AudioData("flac/pcm16.flac")).Length == 24000, "WAV and FLAC still decode with mp3 asked first");

            // The codec suite on the fixture.
            var codec = AudioCodecs.Get("mp3");
            var samples = Mp3FixtureFiles().Select(f => new AudioSample(AudioData($"mp3/{(string)f.Entry["file"]!}"),
                new DecodedAudio((int)f.Entry["rate"]!, (int)f.Entry["channels"]!, Mp3Reference(f.Name)), 1e-4f)).ToList();
            var report = Conformance.CheckAudioCodec(new AudioCodecUnderTest(
                bytes => codec.ReadInfo(bytes.Span) is { } i ? new AudioHeader(i.SampleRate, i.Channels, i.Samples) : null,
                bytes => { var a = codec.Decode(bytes.Span); return new DecodedAudio(a.SampleRate, a.Channels, a.Samples); }), samples);
            Check(report.Passed, $"the audio codec suite on mp3: {report}");
        }
        finally
        {
            Mp3AudioPlugin.Unregister();
        }

        Check(AudioCodecs.Names.SequenceEqual(before) && !AudioCodecs.CanDecode("talk.mp3"), "Unregister brings the library's codecs back");

        // Three minutes of 44.1 kHz stereo (the untagged file's frames over and over, its ID3v2 tag left out), decoded warm into one block.
        byte[] file = File.ReadAllBytes(AudioData("mp3/lame-notag44k.mp3"));
        byte[] one = file[FrameStart(file, 0)..];
        int copies = 117;
        var big = new byte[one.Length * copies];
        for (int c = 0; c < copies; c++)
        {
            one.CopyTo(big, c * one.Length);
        }

        var mp3 = new Mp3Codec();
        mp3.Decode(big);
        long before2 = GC.GetTotalAllocatedBytes(precise: true);
        var watch = Stopwatch.StartNew();
        var audio = mp3.Decode(big);
        double seconds = watch.Elapsed.TotalSeconds;
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before2;
        long floats = 4L * audio.Channels * audio.Length;
        var inOrder = new Mp3Codec(new Mp3Options { FramesPerTask = 0 });
        watch.Restart();
        var sequential = inOrder.Decode(big);
        double sequentialSeconds = watch.Elapsed.TotalSeconds;
        Check(audio.Length == copies * 67968 && audio.Samples.AsSpan().SequenceEqual(sequential.Samples), $"{audio.Length} samples, the same in order");
        Check(allocated < floats + big.Length + (16 << 20), $"decoding allocated {allocated:N0} bytes for {floats:N0} bytes of floats");
        Console.WriteLine($"    {audio.Duration.TotalSeconds:F0} s of 44.1 kHz stereo MP3: {seconds * 1000:F0} ms in parallel ({audio.Duration.TotalSeconds / seconds:F0}x real time), "
            + $"{sequentialSeconds * 1000:F0} ms in order; {allocated:N0} bytes allocated for {floats:N0} bytes of floats");
    }
}
