// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Idrak.Audio;

/// <summary>How <see cref="FlacCodec"/> decodes.</summary>
public sealed record FlacOptions
{
    /// <summary>
    /// Whether a file cut inside its last frame (a download that stopped) gives the samples of its whole frames, as a WAV
    /// whose data chunk claims more than the file holds does; false refuses it as cut. On by default.
    /// </summary>
    public bool KeepCutEnd { get; init; } = true;

    /// <summary>Whether every frame's header CRC-8 and data CRC-16 are checked (a damaged frame refused by number). On by default.</summary>
    public bool CheckCrc { get; init; } = true;

    /// <summary>Frames a worker decodes at a time when the frames are decoded in parallel (1 or more).</summary>
    public int FramesPerTask { get; init; } = 8;
}

/// <summary>
/// FLAC files (decode only): the stream info, the frames' constant, verbatim, fixed and LPC subframes with Rice-coded
/// residuals, wasted bits, the left/side, side/right and mid/side channel pairs, 4 to 32 bits per sample; an ID3v2 tag
/// before the stream is skipped. The frames are found by their sync code (each candidate's header CRC-8 and frame number
/// checked) and decoded in parallel into one planar block; each frame's CRC-16 and length are checked, and a stream whose
/// frames cannot be told apart that way is decoded again frame after frame, which names the damaged frame. The samples'
/// MD5 in the stream info is not checked (the CRCs cover every frame).
/// </summary>
/// <param name="options">How it decodes (the defaults when null).</param>
public sealed class FlacCodec(FlacOptions? options = null) : IAudioCodec
{
    private const int StreamInfoBytes = 34;

    private readonly FlacOptions _options = options ?? new FlacOptions();

    /// <summary>The stream info block's values.</summary>
    private readonly record struct StreamInfo(int MinBlock, int MaxBlock, int MinFrame, int SampleRate, int Channels, int Bits, long Samples, int FramesStart);

    /// <summary>A frame header's values; <see cref="Length"/> is the header's bytes, CRC-8 included.</summary>
    private readonly record struct FrameHeader(bool Variable, int BlockSize, int SampleRate, int Assignment, int Channels, int Bits, long Number, int Length);

    /// <inheritdoc />
    public string Name => "flac";

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".flac"];

    /// <summary>The options it decodes with.</summary>
    public FlacOptions Options => _options;

    /// <inheritdoc />
    public AudioInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        int start = SkipId3(header);
        if (start < 0)
        {
            return new AudioInfo(0, 0, -1, 0, false, Name);                     // an ID3v2 tag longer than the bytes given: ask with all
        }

        if (header.Length < start + 4 || !header.Slice(start, 4).SequenceEqual("fLaC"u8))
        {
            return null;
        }

        if (header.Length < start + 8 + StreamInfoBytes)
        {
            return new AudioInfo(0, 0, -1, 0, false, Name);
        }

        if (!TryStreamInfo(header, start, out var info, out _))
        {
            return new AudioInfo(0, 0, -1, 0, false, Name);                     // a damaged header the decoder refuses by name
        }

        return new AudioInfo(info.SampleRate, info.Channels, info.Samples > 0 ? info.Samples : -1, info.Bits, false, Name);
    }

    /// <inheritdoc />
    public AudioData Decode(ReadOnlySpan<byte> file)
    {
        int start = SkipId3(file);
        if (start < 0 || file.Length < start + 4 || !file.Slice(start, 4).SequenceEqual("fLaC"u8))
        {
            throw new InvalidDataException(start < 0 ? "an ID3v2 tag longer than the file (a cut or damaged file)." : "not a FLAC stream (no fLaC marker).");
        }

        if (!TryStreamInfo(file, start, out var info, out string? problem))
        {
            throw new InvalidDataException(problem!);
        }

        var frames = Locate(file, info, out bool cut);
        if (cut && !_options.KeepCutEnd)
        {
            throw new InvalidDataException("the stream is cut inside a frame (FlacOptions.KeepCutEnd is off).");
        }

        long total = 0;
        foreach (var (_, header) in frames)
        {
            total += header.BlockSize;
        }

        if (info.Samples > 0 && total > info.Samples)
        {
            throw new InvalidDataException($"the frames hold {total} samples, the stream info {info.Samples}.");
        }

        if ((long)info.Channels * total > Array.MaxLength)
        {
            throw new NotSupportedException($"{info.Channels} x {total} samples are more than one block holds.");
        }

        var planar = GC.AllocateUninitializedArray<float>((int)(info.Channels * total));
        var offsets = new long[frames.Count];
        for (int f = 1; f < frames.Count; f++)
        {
            offsets[f] = offsets[f - 1] + frames[f - 1].Header.BlockSize;
        }

        if (!DecodeParallel(file, info, frames, offsets, planar, total))
        {
            DecodeInOrder(file, info, frames, planar, total);                   // names the frame the parallel pass could not place
        }

        return new AudioData(planar, info.Channels, info.SampleRate);
    }

    // The byte after an ID3v2 tag at the start (0 without one), or -1 when the tag runs past the bytes given.
    private static int SkipId3(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 10 || !bytes[..3].SequenceEqual("ID3"u8))
        {
            return 0;
        }

        int size = (bytes[6] & 0x7F) << 21 | (bytes[7] & 0x7F) << 14 | (bytes[8] & 0x7F) << 7 | (bytes[9] & 0x7F);
        int end = 10 + size + ((bytes[5] & 0x10) != 0 ? 10 : 0);                 // a footer when flagged
        return end <= bytes.Length ? end : -1;
    }

    // The stream info (the first metadata block) and where the frames start (after the last metadata block).
    private static bool TryStreamInfo(ReadOnlySpan<byte> file, int start, out StreamInfo info, out string? problem)
    {
        info = default;
        int at = start + 4;
        if (file.Length < at + 4 || (file[at] & 0x7F) != 0 || ReadUInt24(file[(at + 1)..]) < StreamInfoBytes)
        {
            problem = "the first metadata block is not a stream info block.";
            return false;
        }

        var s = file.Slice(at + 4, StreamInfoBytes);
        int minBlock = BinaryPrimitives.ReadUInt16BigEndian(s), maxBlock = BinaryPrimitives.ReadUInt16BigEndian(s[2..]);
        int minFrame = ReadUInt24(s[4..]);
        ulong packed = BinaryPrimitives.ReadUInt64BigEndian(s[10..]);        // rate 20, channels - 1 3, bits - 1 5, samples 36
        int rate = (int)(packed >> 44), channels = (int)((packed >> 41) & 7) + 1, bits = (int)((packed >> 36) & 31) + 1;
        long samples = (long)(packed & 0xF_FFFF_FFFF);
        if (rate == 0 || bits < 4 || maxBlock < 16 || minBlock > maxBlock)
        {
            problem = $"a stream info of {rate} Hz, {bits} bits, blocks of {minBlock} to {maxBlock} samples.";
            return false;
        }

        // The other metadata blocks (seek table, comments, pictures, padding) are skipped by their lengths.
        for (bool last = (file[at] & 0x80) != 0; ;)
        {
            at += 4 + ReadUInt24(file[(at + 1)..]);
            if (last)
            {
                break;
            }

            if (file.Length < at + 4)
            {
                problem = "the metadata blocks run past the file (a cut or damaged file).";
                return false;
            }

            last = (file[at] & 0x80) != 0;
        }

        if (at > file.Length)
        {
            problem = "the metadata blocks run past the file (a cut or damaged file).";
            return false;
        }

        info = new StreamInfo(minBlock, maxBlock, minFrame, rate, channels, bits, samples, at);
        problem = null;
        return true;
    }

    private static int ReadUInt24(ReadOnlySpan<byte> b) => b[0] << 16 | b[1] << 8 | b[2];

    // The frames' starts and headers: from the first, each next one is the first sync code after it whose header parses,
    // passes its CRC-8, agrees with the stream info and carries the next number. `cut` when the last frame runs past the
    // file (it is dropped).
    private List<(int Start, FrameHeader Header)> Locate(ReadOnlySpan<byte> file, in StreamInfo info, out bool cut)
    {
        var frames = new List<(int, FrameHeader)>(info.Samples > 0 ? (int)Math.Min(1 << 20, info.Samples / Math.Max(1, info.MaxBlock) + 1) : 64);
        cut = false;
        int at = info.FramesStart;
        if (at >= file.Length)
        {
            return frames;
        }

        if (!TryHeader(file, at, info, out var header, out string? problem))
        {
            throw new InvalidDataException($"the first frame's header: {problem}");
        }

        long nextNumber = 0;
        while (true)
        {
            if (header.Number != nextNumber)
            {
                throw new InvalidDataException($"frame {frames.Count} is numbered {header.Number}, expected {nextNumber}.");
            }

            frames.Add((at, header));
            nextNumber = header.Variable ? header.Number + header.BlockSize : header.Number + 1;
            int from = at + header.Length + Math.Max(1, info.MinFrame - header.Length - 2);   // no frame is shorter than the stream info's least
            int next = -1;
            FrameHeader found = default;
            while (from < file.Length - 1)
            {
                int ff = file[from..].IndexOf((byte)0xFF);
                if (ff < 0)
                {
                    break;
                }

                int candidate = from + ff;
                if (candidate + 1 < file.Length && (file[candidate + 1] & 0xFE) == 0xF8
                    && TryHeader(file, candidate, info, out found, out _) && found.Number == nextNumber && found.Variable == header.Variable)
                {
                    next = candidate;
                    break;
                }

                from = candidate + 1;
            }

            if (next < 0)
            {
                // The last frame: whole when the stream info's count is reached; short of it (or with no count), whole when it
                // decodes within the file, else cut and dropped. A stream short of its count is cut either way.
                long have = frames.Sum(f => (long)f.Item2.BlockSize);
                if (info.Samples == 0 || have < info.Samples)
                {
                    if (!FrameFits(file, at, info, header))
                    {
                        frames.RemoveAt(frames.Count - 1);
                        cut = true;
                    }
                    else
                    {
                        cut = info.Samples > 0;
                    }
                }

                return frames;
            }

            at = next;
            header = found;
        }
    }

    // Whether the frame at `at` decodes within the file (for a stream that does not say its length).
    private bool FrameFits(ReadOnlySpan<byte> file, int at, in StreamInfo info, in FrameHeader header)
    {
        var scratch = ArrayPool<long>.Shared.Rent(header.BlockSize * header.Channels);
        try
        {
            return TryFrame(file, at, info, header, scratch, out _, out _);
        }
        finally
        {
            ArrayPool<long>.Shared.Return(scratch);
        }
    }

    private bool DecodeParallel(ReadOnlySpan<byte> file, in StreamInfo info, List<(int Start, FrameHeader Header)> frames, long[] offsets, float[] planar, long total)
    {
        if (frames.Count == 0)
        {
            return true;
        }

        // The span cannot cross into the workers: they read a pooled copy of the compressed bytes (a fraction of the samples').
        int length = file.Length;
        byte[] copy = ArrayPool<byte>.Shared.Rent(length);
        try
        {
            file.CopyTo(copy);
            var streamInfo = info;
            bool ok = true;
            HostParallel.For(frames.Count, Math.Max(1, _options.FramesPerTask), (first, last) =>
            {
                var bytes = copy.AsSpan(0, length);
                var scratch = ArrayPool<long>.Shared.Rent(streamInfo.MaxBlock * streamInfo.Channels);
                try
                {
                    for (int f = first; f < last && Volatile.Read(ref ok); f++)
                    {
                        var (start, header) = frames[f];
                        int end = f + 1 < frames.Count ? frames[f + 1].Start : -1;
                        bool known = end >= 0 && _options.CheckCrc;                // the extent known: its CRC first, garbage never decoded
                        if ((known && !CrcMatches(bytes, start, end))
                            || !TryFrame(bytes, start, streamInfo, header, scratch, out int frameEnd, out _, crcChecked: known) || (end >= 0 && frameEnd != end))
                        {
                            Volatile.Write(ref ok, false);
                            return;
                        }

                        Store(scratch, header, planar, offsets[f], total, streamInfo.Bits);
                    }
                }
                finally
                {
                    ArrayPool<long>.Shared.Return(scratch);
                }
            });
            return ok;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(copy);
        }
    }

    // Frame after frame, each one's end found by decoding it: the exact error for the frame that does not decode.
    private void DecodeInOrder(ReadOnlySpan<byte> file, in StreamInfo info, List<(int Start, FrameHeader Header)> frames, float[] planar, long total)
    {
        int count = frames.Count;
        var scratch = ArrayPool<long>.Shared.Rent(info.MaxBlock * info.Channels);
        try
        {
            int at = info.FramesStart;
            long offset = 0;
            for (int f = 0; f < count; f++)
            {
                if (!TryHeader(file, at, info, out var header, out string? problem))
                {
                    throw new InvalidDataException($"frame {f} at byte {at}: {problem}");
                }

                if (!TryFrame(file, at, info, header, scratch, out int end, out problem))
                {
                    // Where the frame was located with a next one after it, its CRC-16 over that extent says whether it is damaged.
                    bool damaged = _options.CheckCrc && f + 1 < count && frames[f].Start == at && !CrcMatches(file, at, frames[f + 1].Start);
                    throw new InvalidDataException(damaged
                        ? $"frame {f} at byte {at}: the frame's CRC-16 does not match (damaged data; decoding it: {problem})"
                        : $"frame {f} at byte {at}: {problem}");
                }

                Store(scratch, header, planar, offset, total, info.Bits);
                offset += header.BlockSize;
                at = end;
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(scratch);
        }
    }

    // A frame's integers (channel after channel, block size each) as floats at their place in the planar block.
    private static void Store(long[] samples, in FrameHeader header, float[] planar, long offset, long total, int bits)
    {
        float scale = 1f / (1L << (bits - 1));
        int n = header.BlockSize;
        for (int c = 0; c < header.Channels; c++)
        {
            var source = samples.AsSpan(c * n, n);
            var target = planar.AsSpan((int)(c * total + offset), n);
            for (int i = 0; i < n; i++)
            {
                target[i] = source[i] * scale;
            }
        }
    }

    private bool TryHeader(ReadOnlySpan<byte> file, int at, in StreamInfo info, out FrameHeader header, out string? problem)
    {
        header = default;
        if (at + 6 > file.Length || file[at] != 0xFF || (file[at + 1] & 0xFE) != 0xF8)
        {
            problem = "no frame sync code.";
            return false;
        }

        bool variable = (file[at + 1] & 1) != 0;
        int sizeCode = file[at + 2] >> 4, rateCode = file[at + 2] & 15, assignment = file[at + 3] >> 4, bitsCode = (file[at + 3] >> 1) & 7;
        if ((file[at + 3] & 1) != 0 || sizeCode == 0 || rateCode == 15 || assignment > 10 || bitsCode == 3)
        {
            problem = "a reserved value in the frame header.";
            return false;
        }

        // The frame or sample number, coded as UTF-8 extended to 7 bytes (36 bits).
        int p = at + 4;
        int lead = file[p];
        int extra = lead < 0x80 ? 0 : lead >= 0xFE ? 6 : BitOperations.LeadingZeroCount((uint)~lead << 24) - 1;
        if (lead is >= 0x80 and < 0xC0 || p + 1 + extra > file.Length)
        {
            problem = "a damaged frame number.";
            return false;
        }

        long number = extra == 0 ? lead : lead & (0x3F >> extra);
        for (int i = 1; i <= extra; i++)
        {
            int b = file[p + i];
            if ((b & 0xC0) != 0x80)
            {
                problem = "a damaged frame number.";
                return false;
            }

            number = number << 6 | (uint)(b & 0x3F);
        }

        p += 1 + extra;
        int blockSize = sizeCode switch
        {
            1 => 192,
            <= 5 => 576 << (sizeCode - 2),
            6 => p < file.Length ? file[p++] + 1 : -1,
            7 => p + 1 < file.Length ? ((file[p++] << 8) | file[p++]) + 1 : -1,
            _ => 256 << (sizeCode - 8),
        };
        int rate = rateCode switch
        {
            0 => info.SampleRate,
            1 => 88200, 2 => 176400, 3 => 192000, 4 => 8000, 5 => 16000, 6 => 22050, 7 => 24000, 8 => 32000, 9 => 44100, 10 => 48000, 11 => 96000,
            12 => p < file.Length ? file[p++] * 1000 : -1,
            13 => p + 1 < file.Length ? (file[p++] << 8) | file[p++] : -1,
            _ => p + 1 < file.Length ? ((file[p++] << 8) | file[p++]) * 10 : -1,
        };
        int bits = bitsCode switch { 0 => info.Bits, 1 => 8, 2 => 12, 4 => 16, 5 => 20, 6 => 24, _ => 32 };
        int channels = assignment < 8 ? assignment + 1 : 2;
        if (blockSize < 0 || rate < 0 || p >= file.Length)
        {
            problem = "the frame header runs past the file.";
            return false;
        }

        if (_options.CheckCrc && Crc8(file[at..p]) != file[p])
        {
            problem = "the frame header's CRC-8 does not match.";
            return false;
        }

        if (channels != info.Channels || bits != info.Bits || blockSize > info.MaxBlock)
        {
            problem = $"a frame of {channels} channels, {bits} bits and {blockSize} samples in a stream of {info.Channels}, {info.Bits} and at most {info.MaxBlock}.";
            return false;
        }

        header = new FrameHeader(variable, blockSize, rate, assignment, channels, bits, number, p + 1 - at);
        problem = null;
        return true;
    }

    // The frame's samples into `samples` (channel after channel), the channels restored from their pair; `end` the byte
    // after its CRC-16.
    private bool TryFrame(ReadOnlySpan<byte> file, int at, in StreamInfo info, in FrameHeader header, long[] samples, out int end, out string? problem, bool crcChecked = false)
    {
        end = -1;
        var reader = new BitReader(file, at + header.Length);
        int n = header.BlockSize;
        for (int c = 0; c < header.Channels; c++)
        {
            int bits = header.Bits + (header.Assignment switch { 8 or 10 => c == 1 ? 1 : 0, 9 => c == 0 ? 1 : 0, _ => 0 });   // the side channel has one bit more
            if (!TrySubframe(ref reader, samples.AsSpan(c * n, n), bits, out problem))
            {
                problem = $"channel {c}: {problem}";
                return false;
            }
        }

        reader.AlignToByte();
        int crcAt = reader.BytePosition;
        if (crcAt + 2 > file.Length)
        {
            problem = "the frame runs past the file (a cut file).";
            return false;
        }

        if (_options.CheckCrc && !crcChecked && Crc16(file[at..crcAt]) != BinaryPrimitives.ReadUInt16BigEndian(file[crcAt..]))
        {
            problem = "the frame's CRC-16 does not match (damaged data).";
            return false;
        }

        var left = samples.AsSpan(0, n);
        var right = header.Channels > 1 ? samples.AsSpan(n, n) : default;
        switch (header.Assignment)
        {
            case 8:                                                             // left, side: right = left - side
                for (int i = 0; i < n; i++)
                {
                    right[i] = left[i] - right[i];
                }

                break;
            case 9:                                                             // side, right: left = side + right
                for (int i = 0; i < n; i++)
                {
                    left[i] += right[i];
                }

                break;
            case 10:                                                            // mid, side
                for (int i = 0; i < n; i++)
                {
                    long side = right[i], mid = left[i] << 1 | (side & 1);
                    left[i] = (mid + side) >> 1;
                    right[i] = (mid - side) >> 1;
                }

                break;
        }

        end = crcAt + 2;
        problem = null;
        return true;
    }

    private static bool TrySubframe(ref BitReader reader, Span<long> output, int bits, out string? problem)
    {
        problem = null;
        if (!reader.TryRead(8, out ulong head) || (head & 0x80) != 0)
        {
            problem = reader.Ended ? "the frame runs past the file (a cut file)." : "a subframe's padding bit is set.";
            return false;
        }

        int type = (int)(head >> 1) & 0x3F;
        int wasted = 0;
        if ((head & 1) != 0)
        {
            if (!reader.TryUnary(out int zeros))
            {
                problem = "the frame runs past the file (a cut file).";
                return false;
            }

            wasted = zeros + 1;
            bits -= wasted;
            if (bits <= 0)
            {
                problem = $"{wasted} wasted bits of {bits + wasted}.";
                return false;
            }
        }

        bool ok;
        if (type == 0)
        {
            ok = reader.TryReadSigned(bits, out long value);
            output.Fill(value);
        }
        else if (type == 1)
        {
            ok = true;
            for (int i = 0; i < output.Length && ok; i++)
            {
                ok = reader.TryReadSigned(bits, out output[i]);
            }
        }
        else if (type is >= 8 and <= 12)
        {
            ok = TryFixed(ref reader, output, bits, type - 8, out problem);
        }
        else if (type >= 32)
        {
            ok = TryLpc(ref reader, output, bits, type - 31, out problem);
        }
        else
        {
            problem = $"a reserved subframe type {type}.";
            return false;
        }

        if (!ok)
        {
            problem ??= "the frame runs past the file (a cut file).";
            return false;
        }

        if (wasted > 0)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] <<= wasted;
            }
        }

        return true;
    }

    private static bool TryFixed(ref BitReader reader, Span<long> s, int bits, int order, out string? problem)
    {
        problem = null;
        if (order > s.Length)
        {
            problem = $"a fixed predictor of order {order} for {s.Length} samples.";
            return false;
        }

        for (int i = 0; i < order; i++)
        {
            if (!reader.TryReadSigned(bits, out s[i]))
            {
                return false;
            }
        }

        if (!TryResidual(ref reader, s, order, out problem))
        {
            return false;
        }

        switch (order)
        {
            case 1:
                for (int i = 1; i < s.Length; i++)
                {
                    s[i] += s[i - 1];
                }

                break;
            case 2:
                for (int i = 2; i < s.Length; i++)
                {
                    s[i] += 2 * s[i - 1] - s[i - 2];
                }

                break;
            case 3:
                for (int i = 3; i < s.Length; i++)
                {
                    s[i] += 3 * (s[i - 1] - s[i - 2]) + s[i - 3];
                }

                break;
            case 4:
                for (int i = 4; i < s.Length; i++)
                {
                    s[i] += 4 * (s[i - 1] + s[i - 3]) - 6 * s[i - 2] - s[i - 4];
                }

                break;
        }

        return true;
    }

    private static bool TryLpc(ref BitReader reader, Span<long> s, int bits, int order, out string? problem)
    {
        problem = null;
        if (order > s.Length)
        {
            problem = $"an LPC predictor of order {order} for {s.Length} samples.";
            return false;
        }

        for (int i = 0; i < order; i++)
        {
            if (!reader.TryReadSigned(bits, out s[i]))
            {
                return false;
            }
        }

        if (!reader.TryRead(4, out ulong precisionCode) || !reader.TryReadSigned(5, out long shift))
        {
            return false;
        }

        if (precisionCode == 15 || shift < 0)
        {
            problem = precisionCode == 15 ? "a reserved LPC precision." : $"a negative LPC shift {shift}.";
            return false;
        }

        int precision = (int)precisionCode + 1;
        Span<long> coefficients = stackalloc long[32];
        for (int j = 0; j < order; j++)
        {
            if (!reader.TryReadSigned(precision, out coefficients[j]))
            {
                return false;
            }
        }

        if (!TryResidual(ref reader, s, order, out problem))
        {
            return false;
        }

        var c = coefficients[..order];
        int q = (int)shift;
        for (int i = order; i < s.Length; i++)
        {
            long sum = 0;
            var history = s.Slice(i - order, order);                            // s[i - order] .. s[i - 1]
            for (int j = 0; j < order; j++)
            {
                sum += c[j] * history[order - 1 - j];                           // c[0] weighs s[i - 1]
            }

            s[i] += sum >> q;
        }

        return true;
    }

    // The residual after the warm-up samples, into s[order..]: Rice-coded partitions (4- or 5-bit parameters, an escape
    // to raw values).
    private static bool TryResidual(ref BitReader reader, Span<long> s, int order, out string? problem)
    {
        problem = null;
        if (!reader.TryRead(2, out ulong method) || !reader.TryRead(4, out ulong partitionOrder))
        {
            return false;
        }

        if (method > 1)
        {
            problem = $"a reserved residual coding {method}.";
            return false;
        }

        int parameterBits = method == 0 ? 4 : 5, escape = (1 << parameterBits) - 1;
        int partitions = 1 << (int)partitionOrder;
        int perPartition = s.Length >> (int)partitionOrder;
        if (perPartition << (int)partitionOrder != s.Length || perPartition < order)
        {
            problem = $"{partitions} residual partitions of a block of {s.Length} after {order} warm-up samples.";
            return false;
        }

        int i = order;
        for (int p = 0; p < partitions; p++)
        {
            int count = p == 0 ? perPartition - order : perPartition;
            if (!reader.TryRead(parameterBits, out ulong parameter))
            {
                return false;
            }

            if ((int)parameter == escape)
            {
                if (!reader.TryRead(5, out ulong raw))
                {
                    return false;
                }

                for (int k = 0; k < count; k++, i++)
                {
                    if (raw == 0)
                    {
                        s[i] = 0;
                    }
                    else if (!reader.TryReadSigned((int)raw, out s[i]))
                    {
                        return false;
                    }
                }

                continue;
            }

            if (!reader.TryRice((int)parameter, s.Slice(i, count)))
            {
                return false;
            }

            i += count;
        }

        return true;
    }

    // Whether the frame from `start` to `end` (its CRC-16 the last two bytes) is whole.
    private static bool CrcMatches(ReadOnlySpan<byte> file, int start, int end) =>
        end - start > 2 && Crc16(file[start..(end - 2)]) == BinaryPrimitives.ReadUInt16BigEndian(file[(end - 2)..]);

    private static byte Crc8(ReadOnlySpan<byte> bytes)
    {
        int crc = 0;
        foreach (byte b in bytes)
        {
            crc = Crc8Table[crc ^ b];
        }

        return (byte)crc;
    }

    private static ushort Crc16(ReadOnlySpan<byte> bytes)
    {
        var table = Crc16Table;
        int crc = 0;
        foreach (byte b in bytes)
        {
            crc = ((crc << 8) ^ table[(crc >> 8) ^ b]) & 0xFFFF;
        }

        return (ushort)crc;
    }

    private static readonly byte[] Crc8Table = MakeCrc8();
    private static readonly ushort[] Crc16Table = MakeCrc16();

    private static byte[] MakeCrc8()
    {
        var table = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            int crc = i;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 0x80) != 0 ? (crc << 1) ^ 0x07 : crc << 1;
            }

            table[i] = (byte)crc;
        }

        return table;
    }

    private static ushort[] MakeCrc16()
    {
        var table = new ushort[256];
        for (int i = 0; i < 256; i++)
        {
            int crc = i << 8;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 0x8000) != 0 ? (crc << 1) ^ 0x8005 : crc << 1;
            }

            table[i] = (ushort)crc;
        }

        return table;
    }

    // Bits read most significant first through a 64-bit cache, eight bytes loaded at a time where the file has them.
    private ref struct BitReader(ReadOnlySpan<byte> data, int start)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _next = start;                                               // the next byte to load
        private ulong _cache;                                                    // valid bits from the top
        private int _bits;

        public bool Ended { get; private set; }

        public readonly int BytePosition => _next - _bits / 8;                   // after AlignToByte: the next byte

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Refill()
        {
            if (_next + 8 <= _data.Length)
            {
                // The bits below the whole bytes taken are the next byte's: a later load ORs the same bits in again.
                _cache |= BinaryPrimitives.ReadUInt64BigEndian(_data[_next..]) >> _bits;
                int take = (63 - _bits) >> 3;
                _next += take;
                _bits += take * 8;
                return;
            }

            while (_bits <= 56 && _next < _data.Length)
            {
                _cache |= (ulong)_data[_next++] << (56 - _bits);
                _bits += 8;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryRead(int count, out ulong value)
        {
            if (count == 0)
            {
                value = 0;
                return true;
            }

            if (_bits < count)
            {
                Refill();
                if (_bits < count)
                {
                    Ended = true;
                    value = 0;
                    return false;
                }
            }

            value = _cache >> (64 - count);
            _cache <<= count;
            _bits -= count;
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryReadSigned(int count, out long value)
        {
            if (!TryRead(count, out ulong raw))
            {
                value = 0;
                return false;
            }

            value = count == 0 ? 0 : (long)(raw << (64 - count)) >> (64 - count);
            return true;
        }

        // The zeros before the next 1 (the 1 consumed).
        public bool TryUnary(out int zeros)
        {
            zeros = 0;
            while (true)
            {
                if (_bits == 0)
                {
                    Refill();
                    if (_bits == 0)
                    {
                        Ended = true;
                        return false;
                    }
                }

                int lead = BitOperations.LeadingZeroCount(_cache);
                if (lead < _bits)
                {
                    zeros += lead;
                    _cache <<= lead + 1;
                    _bits -= lead + 1;
                    return true;
                }

                zeros += _bits;
                _cache = 0;
                _bits = 0;
            }
        }

        // Rice-coded values with parameter k into `output` (each zig-zag decoded).
        public bool TryRice(int k, Span<long> output)
        {
            for (int i = 0; i < output.Length; i++)
            {
                if (!TryUnary(out int quotient) || !TryRead(k, out ulong low))
                {
                    return false;
                }

                ulong u = ((ulong)quotient << k) | low;
                output[i] = (long)(u >> 1) ^ -(long)(u & 1);
            }

            return true;
        }

        public void AlignToByte()
        {
            int drop = _bits & 7;
            _cache <<= drop;
            _bits -= drop;
        }
    }
}
