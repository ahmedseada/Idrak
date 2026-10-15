// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 21, phase S1: the FLAC codec. The files come from tools/audio/make_tiny_audio.py (flac/, written by libFLAC through
// soundfile); the round trips use a small FLAC writer of the test's own (fixed predictors, Rice residuals, every channel
// pairing), so the decoder is checked against more than one encoder.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioFlacGroup =
    [
        ("audio codecs: flac: every FLAC of the fixture (libFLAC's levels 0, 5 and 8, 8, 16 and 24 bits, stereo pairs, 44.1 kHz, constant and verbatim subframes, wasted bits) decodes to its integers exactly; its header is read from the first bytes", d => { if (d == Device.Cpu) AudioFlacFixture(); }),
        ("audio codecs: flac: a file cut inside a frame keeps its whole frames (refused with KeepCutEnd off), a flipped byte is refused naming the CRC and the file, an ID3v2 tag before the stream is skipped, a stream that does not say its length decodes the same, and one frame per task gives the samples of many", d => { if (d == Device.Cpu) AudioFlacDamaged(); }),
        ("audio codecs: flac: the test's writer (fixed orders 0 to 4, independent, left/side, side/right and mid/side channels, fixed and variable block sizes, 8 to 32 bits, a short last frame) round-trips exactly through the decoder", d => { if (d == Device.Cpu) AudioFlacRoundTrips(); }),
        ("audio codecs: flac: ten minutes decode into one block (the samples' bytes allocated, no more), the registry names flac from the library, and the codec suite passes on the fixture; the decode's speed", d => { if (d == Device.Cpu) AudioFlacLongAndSuite(); }),
    ];

    private static (string Name, JsonObject Entry)[] FlacFixtureFiles()
    {
        var files = JsonNode.Parse(File.ReadAllText(AudioData("flac/files.json")))!.AsObject();
        return [.. files.Select(p => (p.Key, p.Value!.AsObject()))];
    }

    private static void AudioFlacFixture()
    {
        foreach (var (name, entry) in FlacFixtureFiles())
        {
            string path = AudioData($"flac/{name}.flac");
            int rate = (int)entry["rate"]!, channels = (int)entry["channels"]!, samples = (int)entry["samples"]!, bits = (int)entry["bits"]!;
            var info = AudioCodecs.ReadInfo(path);
            Check(info is { } i && i.SampleRate == rate && i.Channels == channels && i.Samples == samples && i.BitsPerSample == bits && !i.IsFloat && i.Format == "flac",
                $"{name}: the header says {info}, the fixture {channels} x {samples} at {rate} Hz, {bits} bits");
            var audio = AudioCodecs.Decode(path);
            Check(audio.SampleRate == rate && audio.Channels == channels && audio.Length == samples, $"{name}: decoded {audio.Channels} x {audio.Length} at {audio.SampleRate} Hz");
            var expected = ReadNpyFloat32(AudioData($"flac/{name}.npy"));
            Check(expected.AsSpan().SequenceEqual(audio.Samples), $"{name}: the samples decode exactly (first difference at {expected.AsSpan().CommonPrefixLength(audio.Samples)} of {expected.Length})");
            Check(AudioCodecs.CanDecode(path) && AudioCodecs.ReadInfo(File.ReadAllBytes(path)) == info, $"{name}: CanDecode and the in-memory header");
        }

        // The same signal as the WAV fixture's 16-bit file: the two formats give the same samples.
        Check(AudioCodecs.Decode(AudioData("flac/pcm16.flac")).Samples.AsSpan().SequenceEqual(AudioCodecs.Decode(AudioData("wav/pcm16.wav")).Samples), "pcm16.flac and pcm16.wav hold the same samples");
        Check(AudioCodecs.Load(AudioData("flac/stereo16.flac"), channels: 1).Channels == 1, "Load mixes a FLAC to one channel");
    }

    private static void AudioFlacDamaged()
    {
        var whole = ReadNpyFloat32(AudioData("flac/pcm16.npy"));

        // Cut inside a frame: the whole frames before the cut, or refused when asked.
        var cut = AudioCodecs.Decode(AudioData("flac/bad-truncated.flac"));
        Check(cut.Length > 0 && cut.Length < whole.Length && whole.AsSpan(0, cut.Length).SequenceEqual(cut.Samples),
            $"a cut file gives its first {cut.Length} of {whole.Length} samples, as the whole file has them");
        var strict = new FlacCodec(new FlacOptions { KeepCutEnd = false });
        ThrowsNaming<InvalidDataException>(() => strict.Decode(File.ReadAllBytes(AudioData("flac/bad-truncated.flac"))), "a cut file with KeepCutEnd off", "cut");

        // A flipped byte in the first frame's data: refused naming its CRC and the file.
        ThrowsNaming<InvalidDataException>(() => AudioCodecs.Decode(AudioData("flac/bad-crc.flac")), "a damaged frame", "CRC-16", "bad-crc.flac", "frame 0");

        // An ID3v2 tag (with a 300-byte body) before the stream.
        byte[] file = File.ReadAllBytes(AudioData("flac/pcm16.flac"));
        var tagged = new byte[10 + 300 + file.Length];
        "ID3"u8.CopyTo(tagged);
        tagged[3] = 4;
        tagged[8] = 300 >> 7;
        tagged[9] = 300 & 0x7F;
        file.CopyTo(tagged, 310);
        Check(AudioCodecs.ReadInfo(tagged) is { Format: "flac", Samples: 24000 } && AudioCodecs.Decode(tagged).Samples.AsSpan().SequenceEqual(whole), "an ID3v2 tag before the stream is skipped");

        // The stream info's sample count zeroed (a stream written before its length was known).
        var unknown = (byte[])file.Clone();
        unknown[21] &= 0xF0;
        unknown.AsSpan(22, 4).Clear();
        Check(AudioCodecs.ReadInfo(unknown) is { Samples: -1 } && AudioCodecs.Decode(unknown).Samples.AsSpan().SequenceEqual(whole), "a stream that does not say its length decodes the same");

        // One frame per worker task and all frames in one task give the same samples.
        foreach (string name in new[] { "stereo16", "mixed16", "rate44k" })
        {
            byte[] bytes = File.ReadAllBytes(AudioData($"flac/{name}.flac"));
            var one = new FlacCodec(new FlacOptions { FramesPerTask = 1 }).Decode(bytes);
            var all = new FlacCodec(new FlacOptions { FramesPerTask = 1 << 20 }).Decode(bytes);
            Check(one.Samples.AsSpan().SequenceEqual(all.Samples), $"{name}: one frame per task and every frame in one task agree");
        }

        Throws<InvalidDataException>(() => new FlacCodec().Decode("fLaX0000"u8.ToArray()), "bytes without the fLaC marker");
        Check(new FlacCodec().ReadInfo("RIFF0000WAVE"u8) is null && new FlacCodec().ReadInfo("ID3"u8) is null, "a WAV and a short ID3 start are not FLAC");
    }

    private static void AudioFlacRoundTrips()
    {
        var random = new Random(47);
        var codec = new FlacCodec();
        int cases = 0;
        foreach (int bits in new[] { 8, 12, 16, 20, 24, 32 })
        {
            foreach (int assignment in new[] { 0, 8, 9, 10 })
            {
                foreach (int order in new[] { 0, 1, 2, 3, 4 })
                {
                    bool variable = (cases & 1) == 1;
                    int channels = assignment == 0 ? 1 + cases % 3 : 2;
                    int length = 1000 + random.Next(9000), block = new[] { 192, 1000, 4096, 1152 }[cases % 4];
                    var ints = new long[channels][];
                    long full = 1L << (bits - 1);
                    for (int c = 0; c < channels; c++)
                    {
                        ints[c] = new long[length];
                        double phase = random.NextDouble();
                        for (int i = 0; i < length; i++)
                        {
                            double v = 0.7 * Math.Sin(2 * Math.PI * (0.01 + 0.002 * c) * i + phase) + 0.05 * (random.NextDouble() * 2 - 1);
                            ints[c][i] = Math.Clamp((long)Math.Round(v * full), -full, full - 1);
                        }
                    }

                    ints[0][0] = -full;                                         // the extremes
                    ints[0][1] = full - 1;
                    byte[] file = TestFlacWriter.Write(ints, 16000 + cases, bits, block, order, assignment, variable);
                    var audio = codec.Decode(file);
                    float scale = 1f / full;
                    var expected = new float[channels * length];
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < length; i++)
                        {
                            expected[c * length + i] = ints[c][i] * scale;
                        }
                    }

                    string what = $"{bits} bits, assignment {assignment}, order {order}, {channels} channel(s), blocks of {block}{(variable ? " (variable)" : "")}, {length} samples";
                    Check(audio.Channels == channels && audio.SampleRate == 16000 + cases && audio.Length == length, $"{what}: {audio.Channels} x {audio.Length} at {audio.SampleRate}");
                    Check(expected.AsSpan().SequenceEqual(audio.Samples), $"{what}: first difference at {expected.AsSpan().CommonPrefixLength(audio.Samples)}");
                    cases++;
                }
            }
        }

        Check(cases == 120, $"{cases} round trips");
    }

    private static void AudioFlacLongAndSuite()
    {
        Check(AudioCodecs.Names.Contains("flac") && AudioCodecs.Extensions.Contains(".flac") && AudioCodecs.Origin("flac") == Overrides.Library && AudioCodecs.Get("flac") is FlacCodec,
            $"the registry names flac from the library: {string.Join(", ", AudioCodecs.Names)}");

        // Ten minutes of 16-bit mono (a tone and noise) through the test's writer, decoded warm.
        const int Rate = 16000;
        int length = Rate * 60 * 10;
        var ints = new long[length];
        var random = new Random(9);
        for (int i = 0; i < length; i++)
        {
            ints[i] = (long)Math.Round((0.5 * Math.Sin(2 * Math.PI * 440 * i / Rate) + 0.1 * (random.NextDouble() * 2 - 1)) * 32767);
        }

        byte[] file = TestFlacWriter.Write([ints], Rate, 16, 4096, 2, 0, variable: false);
        var codec = AudioCodecs.Get("flac");
        codec.Decode(file);                                                     // warm: the pooled copy and the scratch blocks
        long before = GC.GetTotalAllocatedBytes(precise: true);
        var watch = Stopwatch.StartNew();
        var audio = codec.Decode(file);
        double seconds = watch.Elapsed.TotalSeconds;
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        long floats = 4L * length;
        Check(audio.Length == length && audio.Samples[12345] == ints[12345] / 32768f, "the long file decodes to its samples");
        Check(allocated < floats + (1 << 20), $"decoding allocated {allocated:N0} bytes for {floats:N0} bytes of floats");
        Console.WriteLine($"    ten minutes of 16-bit FLAC ({file.Length:N0} bytes): {seconds * 1000:F0} ms ({600 / seconds:F0}x real time, {file.Length / seconds / 1e6:F0} MB/s), {allocated:N0} bytes allocated for {floats:N0} bytes of floats");

        // The codec suite on the fixture: every file's samples, the damaged one refused naming its CRC, cut copies.
        var samples = new List<AudioSample>();
        foreach (var (name, entry) in FlacFixtureFiles())
        {
            samples.Add(new AudioSample(AudioData($"flac/{name}.flac"), new DecodedAudio((int)entry["rate"]!, (int)entry["channels"]!, ReadNpyFloat32(AudioData($"flac/{name}.npy")))));
        }

        samples.Add(new AudioSample(AudioData("flac/bad-crc.flac"), Error: "CRC-16"));
        var under = new AudioCodecUnderTest(
            bytes => codec.ReadInfo(bytes.Span) is { } info ? new AudioHeader(info.SampleRate, info.Channels, info.Samples) : null,
            bytes => { var a = codec.Decode(bytes.Span); return new DecodedAudio(a.SampleRate, a.Channels, a.Samples); });
        var report = Conformance.CheckAudioCodec(under, samples);
        Check(report.Passed, $"the audio codec suite on flac: {report}");
    }

    // A small FLAC writer for the tests: one fixed predictor of the order given for every subframe, one Rice partition with a
    // parameter from the residual's mean, the channel pairing given, block sizes in the header (16 bits), the stream info
    // with the length and no MD5.
    private static class TestFlacWriter
    {
        public static byte[] Write(long[][] channels, int rate, int bits, int block, int order, int assignment, bool variable)
        {
            int count = channels.Length, length = channels[0].Length;
            var output = new List<byte>();
            output.AddRange("fLaC"u8.ToArray());
            var info = new BitWriter();
            info.Write(16, (ulong)block);
            info.Write(16, (ulong)block);
            info.Write(24, 0);
            info.Write(24, 0);
            info.Write(20, (ulong)rate);
            info.Write(3, (ulong)(count - 1));
            info.Write(5, (ulong)(bits - 1));
            info.Write(36, (ulong)length);
            info.Write(64, 0);
            info.Write(64, 0);
            output.Add(0x80);                                                   // the last metadata block: stream info
            output.AddRange([0, 0, 34]);
            output.AddRange(info.ToArray());
            for (int start = 0, frame = 0; start < length; start += block, frame++)
            {
                int n = Math.Min(block, length - start);
                var header = new BitWriter();
                header.Write(15, 0b111111111111100);
                header.Write(1, variable ? 1UL : 0);
                header.Write(4, 7);                                             // the block size in 16 bits after the number
                header.Write(4, 0);                                             // the rate from the stream info
                header.Write(4, (ulong)(assignment == 0 ? count - 1 : assignment));   // independent: the channels less one
                header.Write(3, 0);                                             // the bits from the stream info
                header.Write(1, 0);
                foreach (byte b in Utf8Number(variable ? start : frame))
                {
                    header.Write(8, b);
                }

                header.Write(16, (ulong)(n - 1));
                var bytes = header.ToArray().ToList();
                bytes.Add(Crc8(bytes));
                var body = new BitWriter();
                body.Append(bytes);
                for (int c = 0; c < count; c++)
                {
                    var values = Channel(channels, c, start, n, assignment);
                    int width = bits + (assignment switch { 8 or 10 => c == 1 ? 1 : 0, 9 => c == 0 ? 1 : 0, _ => 0 });
                    Subframe(body, values, width, Math.Min(order, n));
                }

                body.Align();
                var frameBytes = body.ToArray().ToList();
                ushort crc = Crc16(frameBytes);
                frameBytes.Add((byte)(crc >> 8));
                frameBytes.Add((byte)crc);
                output.AddRange(frameBytes);
            }

            return [.. output];
        }

        private static long[] Channel(long[][] channels, int c, int start, int n, int assignment)
        {
            var values = new long[n];
            for (int i = 0; i < n; i++)
            {
                long left = channels[0][start + i], right = channels.Length > 1 ? channels[1][start + i] : 0;
                values[i] = assignment switch
                {
                    8 => c == 0 ? left : left - right,                          // left, side
                    9 => c == 0 ? left - right : right,                         // side, right
                    10 => c == 0 ? (left + right) >> 1 : left - right,          // mid, side
                    _ => channels[c][start + i],
                };
            }

            return values;
        }

        private static void Subframe(BitWriter w, long[] s, int width, int order)
        {
            w.Write(1, 0);
            w.Write(6, (ulong)(8 + order));
            w.Write(1, 0);
            for (int i = 0; i < order; i++)
            {
                w.Write(width, (ulong)s[i] & Mask(width));
            }

            var residual = new long[s.Length - order];
            for (int i = order; i < s.Length; i++)
            {
                long p = order switch
                {
                    0 => 0,
                    1 => s[i - 1],
                    2 => 2 * s[i - 1] - s[i - 2],
                    3 => 3 * (s[i - 1] - s[i - 2]) + s[i - 3],
                    _ => 4 * (s[i - 1] + s[i - 3]) - 6 * s[i - 2] - s[i - 4],
                };
                residual[i - order] = s[i] - p;
            }

            double mean = residual.Length == 0 ? 0 : residual.Average(r => (double)Math.Abs(r));
            int k = mean < 1 ? 0 : Math.Min(30, (int)Math.Floor(Math.Log2(mean)));
            w.Write(2, 1);                                                      // 5-bit parameters
            w.Write(4, 0);                                                      // one partition
            w.Write(5, (ulong)k);
            foreach (long r in residual)
            {
                ulong u = (ulong)((r << 1) ^ (r >> 63));
                for (ulong q = u >> k; q > 0;)
                {
                    int zeros = (int)Math.Min(q, 32UL);
                    w.Write(zeros, 0);
                    q -= (ulong)zeros;
                }

                w.Write(1, 1);
                w.Write(k, u & Mask(k));
            }
        }

        private static ulong Mask(int bits) => bits >= 64 ? ulong.MaxValue : (1UL << bits) - 1;

        private static byte[] Utf8Number(long v)
        {
            if (v < 0x80)
            {
                return [(byte)v];
            }

            int length = 2;
            while (v >= 1L << (5 * length + 1))
            {
                length++;
            }

            var bytes = new byte[length];
            for (int i = length - 1; i > 0; i--)
            {
                bytes[i] = (byte)(0x80 | (v & 0x3F));
                v >>= 6;
            }

            bytes[0] = (byte)((0xFF00 >> length) & 0xFF | (int)v);
            return bytes;
        }

        private static byte Crc8(List<byte> bytes)
        {
            int crc = 0;
            foreach (byte b in bytes)
            {
                crc ^= b;
                for (int k = 0; k < 8; k++)
                {
                    crc = (crc & 0x80) != 0 ? ((crc << 1) ^ 0x07) & 0xFF : (crc << 1) & 0xFF;
                }
            }

            return (byte)crc;
        }

        private static ushort Crc16(List<byte> bytes)
        {
            int crc = 0;
            foreach (byte b in bytes)
            {
                crc ^= b << 8;
                for (int k = 0; k < 8; k++)
                {
                    crc = (crc & 0x8000) != 0 ? ((crc << 1) ^ 0x8005) & 0xFFFF : (crc << 1) & 0xFFFF;
                }
            }

            return (ushort)crc;
        }

        private sealed class BitWriter
        {
            private readonly List<byte> _bytes = [];
            private int _current, _used;

            public void Write(int count, ulong value)
            {
                while (count > 0)
                {
                    int take = Math.Min(count, 8 - _used);
                    _current = (_current << take) | (int)((value >> (count - take)) & ((1UL << take) - 1));
                    _used += take;
                    count -= take;
                    if (_used == 8)
                    {
                        _bytes.Add((byte)_current);
                        (_current, _used) = (0, 0);
                    }
                }
            }

            public void Append(IEnumerable<byte> bytes)
            {
                foreach (byte b in bytes)
                {
                    Write(8, b);
                }
            }

            public void Align()
            {
                if (_used > 0)
                {
                    Write(8 - _used, 0);
                }
            }

            public byte[] ToArray() => [.. _bytes];
        }
    }
}
