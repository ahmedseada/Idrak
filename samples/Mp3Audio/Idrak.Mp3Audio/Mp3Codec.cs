// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;

namespace Idrak.Mp3Audio;

/// <summary>How <see cref="Mp3Codec"/> decodes.</summary>
public sealed record Mp3Options
{
    /// <summary>
    /// Trim the encoder's delay and padding that a LAME or Lavc tag in the first frame names (with the decoder's 529
    /// samples), so the sound starts and ends where the encoded one did, as ffmpeg and minimp3 do. On by default; off
    /// gives every decoded sample.
    /// </summary>
    public bool Gapless { get; init; } = true;

    /// <summary>
    /// Frames a worker decodes at a time. Each worker starts a few frames earlier (enough for the bit reservoir and the
    /// filters' state) and drops what those give, so the samples are the same as decoding in order. 0 or less decodes
    /// in order on one thread.
    /// </summary>
    public int FramesPerTask { get; init; } = 512;

    /// <summary>
    /// When bytes that are not MPEG audio come before the first frame header found (not an ID3 tag: junk, or the samples
    /// of another format), the headers that must follow it in a row, each where the one before ends, before the bytes
    /// are taken as MPEG audio by <see cref="Mp3Codec.ReadInfo"/>: a frame sync found by chance (quiet 16-bit samples
    /// hold many FF Fx pairs) does not chain. A stream shorter than that after its junk is refused.
    /// </summary>
    public int SyncFramesAfterJunk { get; init; } = 4;
}

/// <summary>
/// MPEG audio files (decode only): MPEG-1, MPEG-2 and MPEG-2.5 Layer III (.mp3) and Layer II (.mp2), mono, stereo, joint
/// (mid/side and intensity) stereo, constant and variable bit rates, free format; an ID3v2 tag before the stream and
/// ID3v1, Lyrics or APE tags after it are skipped; a stream that loses sync is found again (as minimp3 finds it); a
/// Xing, Info or LAME tag gives the length and, with <see cref="Mp3Options.Gapless"/>, the encoder's delay and padding.
/// The decoding is minimp3's (https://github.com/lieff/minimp3, CC0) ported to C#, frames decoded in parallel in
/// chunks with the same samples as in order. Layer I is refused by name (no fixture checks it yet).
/// </summary>
/// <param name="options">How it decodes (the defaults when null).</param>
public sealed class Mp3Codec(Mp3Options? options = null) : IAudioCodec
{
    private const int DecoderDelay = 528 + 1;

    private readonly Mp3Options _options = options ?? new Mp3Options();

    /// <summary>One frame of the stream: where it lies and whether the decoder starts over there.</summary>
    private readonly record struct Frame(int Offset, int Size, int Samples, int Channels, int Layer, bool Reset, int MainBytes);

    /// <summary>What a Xing, Info or LAME tag says.</summary>
    private readonly record struct VbrTag(long Frames, int Delay, int Padding);

    /// <inheritdoc />
    public string Name => "mp3";

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".mp3", ".mp2"];

    /// <summary>The options it decodes with.</summary>
    public Mp3Options Options => _options;

    /// <inheritdoc />
    public AudioInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        byte[] bytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, header.Length));
        try
        {
            header.CopyTo(bytes);
            var (start, end, id3Past) = Bounds(bytes, header.Length);
            if (id3Past)
            {
                return new AudioInfo(0, 0, -1, 32, true, Name);                 // an ID3v2 tag past the bytes given: ask with all of them
            }

            int freeFormat = 0;
            int skip = Mp3Decoder.FindFrame(bytes, start, end - start, ref freeFormat, out int frameSize);
            if (frameSize == 0)
            {
                return null;
            }

            int at = start + skip;
            if (OtherContainer(bytes, header.Length) || (skip > 0 && !Chains(bytes, at, end, freeFormat, _options.SyncFramesAfterJunk)))
            {
                return null;                                                     // another format's file, or a sync found by chance in its bytes
            }

            int rate = Mp3Decoder.SampleRateHz(bytes, at), channels = Mp3Decoder.IsMono(bytes, at) ? 1 : 2;
            if (ReadTag(bytes, at, frameSize) is { } tag)
            {
                long samples = tag.Frames * Mp3Decoder.FrameSamples(bytes, at);
                return new AudioInfo(rate, channels, Trimmed(samples, tag), 32, true, Name);
            }

            // No tag: the frames counted when the bytes given end on a whole frame; unknown (-1) when they end inside one (the
            // first bytes of a longer file, or a cut file).
            var frames = Index(bytes, start, end, out bool cut);
            return new AudioInfo(rate, channels, cut ? -1 : frames.Sum(f => (long)f.Samples), 32, true, Name);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    // A RIFF (WAV), FLAC, Ogg or MP4 file: never a stream of MPEG frames, whatever its samples hold (an MPEG stream inside
    // a WAV file is the WAV codec's to read or refuse).
    private static bool OtherContainer(byte[] bytes, int length) =>
        length >= 4 && (bytes.AsSpan(0, 4).SequenceEqual("RIFF"u8) || bytes.AsSpan(0, 4).SequenceEqual("fLaC"u8) || bytes.AsSpan(0, 4).SequenceEqual("OggS"u8)
                        || (length >= 8 && bytes.AsSpan(4, 4).SequenceEqual("ftyp"u8)));

    // Whether `count` frame headers follow the one at `at` in a row, each where the one before ends, within [at, end).
    private static bool Chains(byte[] bytes, int at, int end, int freeFormat, int count)
    {
        int o = at;
        for (int k = 0; k < count; k++)
        {
            int size = Mp3Decoder.FrameBytes(bytes, o, freeFormat) + Mp3Decoder.Padding(bytes, o);
            if (size <= 0 || o + size + 4 > end || !Mp3Decoder.HeaderCompare(bytes, at, bytes, o + size))
            {
                return false;
            }

            o += size;
        }

        return true;
    }

    /// <inheritdoc />
    public AudioData Decode(ReadOnlySpan<byte> file)
    {
        byte[] bytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, file.Length));
        try
        {
            file.CopyTo(bytes);
            return Decode(bytes, file.Length);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    private AudioData Decode(byte[] bytes, int length)
    {
        var (start, end, id3Past) = Bounds(bytes, length);
        if (id3Past)
        {
            throw new InvalidDataException("an ID3v2 tag longer than the file (a cut or damaged file).");
        }

        var frames = Index(bytes, start, end, out _);
        if (frames.Count == 0)
        {
            throw new InvalidDataException("no MPEG audio frame (not an MP3 file, or a damaged one).");
        }

        if (frames.Any(f => f.Layer == 1))
        {
            throw new NotSupportedException("MPEG Layer I frames are not read (Layer II and Layer III are).");
        }

        var first = frames[0];
        int rate = Mp3Decoder.SampleRateHz(bytes, first.Offset), channels = first.Channels;
        VbrTag? tag = first.Layer == 3 ? ReadTag(bytes, first.Offset, first.Size) : null;
        if (tag is not null)
        {
            frames.RemoveAt(0);                                                  // the tag's frame holds no sound
            if (frames.Count > 0 && !frames[0].Reset)
            {
                frames[0] = frames[0] with { Reset = true };                     // minimp3 starts over after the tag frame
            }
        }

        var offsets = new long[frames.Count + 1];
        for (int f = 0; f < frames.Count; f++)
        {
            offsets[f + 1] = offsets[f] + frames[f].Samples;
        }

        long decoded = offsets[^1];
        long skip = 0, keep = decoded;
        if (tag is { } t && _options.Gapless)
        {
            skip = Math.Min(decoded, t.Delay);
            keep = Math.Min(decoded - skip, Math.Max(0, Trimmed(t.Frames * first.Samples, t)));
        }

        if (keep * channels > Array.MaxLength)
        {
            throw new NotSupportedException($"{channels} x {keep} samples are more than one block holds.");
        }

        var planar = GC.AllocateUninitializedArray<float>((int)(keep * channels));
        int perTask = _options.FramesPerTask;
        int chunks = perTask <= 0 ? 1 : (frames.Count + perTask - 1) / perTask;
        HostParallel.For(chunks, 1, (firstChunk, lastChunk) =>
        {
            var decoder = new Mp3Decoder();
            var pcm = new float[1152 * 2];
            for (int c = firstChunk; c < lastChunk; c++)
            {
                int from = perTask <= 0 ? 0 : c * perTask, to = perTask <= 0 ? frames.Count : Math.Min(frames.Count, from + perTask);
                int warm = WarmupStart(frames, from);
                for (int f = warm; f < to; f++)
                {
                    var frame = frames[f];
                    int n = decoder.DecodeFrameAt(bytes, frame.Offset, frame.Size, frame.Reset || f == warm, pcm);
                    if (f < from)
                    {
                        continue;                                                // warming up: the state, not the samples
                    }

                    Store(pcm, n == 0 ? 0 : frame.Channels, frame.Samples, offsets[f], skip, keep, planar, channels);
                }
            }
        });

        return new AudioData(planar, channels, rate);
    }

    // The frame a chunk starting at `from` decodes from, so that its state is as decoding in order makes it: the frame
    // before it decoded exactly (the IMDCT overlap and the synthesis history come from that one frame), which needs the up
    // to 511 bytes of main data its bit reservoir reaches back into, so frames before that one until they hold 511 bytes;
    // or a frame where the state starts over anyway.
    private static int WarmupStart(List<Frame> frames, int from)
    {
        if (from == 0 || frames[from].Reset)
        {
            return from;
        }

        int at = from - 1, bytes = 0;
        while (at > 0 && !frames[at].Reset && bytes < 511)
        {
            at--;
            bytes += frames[at].MainBytes;
        }

        return at;
    }

    // A frame's interleaved samples (silence when it gave none: its reservoir was not there) into the planar block, the
    // part within [skip, skip + keep) of the stream; a frame of another channel count mixed or copied to the stream's.
    private static void Store(float[] pcm, int frameChannels, int samples, long at, long skip, long keep, float[] planar, int channels)
    {
        long from = Math.Max(at, skip), to = Math.Min(at + samples, skip + keep);
        if (to <= from)
        {
            return;
        }

        int first = (int)(from - at), count = (int)(to - from), target = (int)(from - skip);
        for (int c = 0; c < channels; c++)
        {
            var output = planar.AsSpan(c * (int)keep + target, count);
            if (frameChannels == 0)
            {
                output.Clear();                                                  // its reservoir was not there: silence
            }
            else if (frameChannels == channels)
            {
                for (int i = 0; i < count; i++)
                {
                    output[i] = Math.Clamp(pcm[(first + i) * channels + c], -1f, 1f);
                }
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int k = first + i;
                    output[i] = Math.Clamp(frameChannels == 1 ? pcm[k] : 0.5f * (pcm[2 * k] + pcm[2 * k + 1]), -1f, 1f);
                }
            }
        }
    }

    private long Trimmed(long samples, VbrTag tag) => _options.Gapless ? Math.Max(0, samples - tag.Delay - Math.Max(0, tag.Padding)) : samples;

    // The stream's bytes: after an ID3v2 tag at the start, before ID3v1 (and its TAG+), Lyrics3 and APE tags at the end.
    private static (int Start, int End, bool Id3Past) Bounds(byte[] b, int length)
    {
        int start = 0, end = length;
        if (length >= 10 && b[0] == 'I' && b[1] == 'D' && b[2] == '3' && (b[5] & 15) == 0 && ((b[6] | b[7] | b[8] | b[9]) & 0x80) == 0)
        {
            int size = (b[6] << 21 | b[7] << 14 | b[8] << 7 | b[9]) + 10 + ((b[5] & 16) != 0 ? 10 : 0);
            if (size >= length)
            {
                return (length, length, true);
            }

            start = size;
        }

        if (end - start > 128 && b[end - 128] == 'T' && b[end - 127] == 'A' && b[end - 126] == 'G')
        {
            end -= 128;
            if (end - start > 227 && b.AsSpan(end - 227, 4).SequenceEqual("TAG+"u8))
            {
                end -= 227;
            }
        }

        if (end - start > 32 && b.AsSpan(end - 32, 8).SequenceEqual("APETAGEX"u8))
        {
            int apeSize = BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(end - 20));     // the footer's tag size (items and footer)
            bool hasHeader = (b[end - 9] & 0x80) != 0;
            int total = apeSize + (hasHeader ? 32 : 0);
            end -= total > 0 && total <= end - start ? total : 32;
        }

        return (start, end, false);
    }

    // Every frame from `start`, taken as the decoder takes them; `cut` when the last one runs past `end`.
    private static List<Frame> Index(byte[] bytes, int start, int end, out bool cut)
    {
        var frames = new List<Frame>(Math.Max(16, (end - start) / 200));
        var header = new byte[4];
        int freeFormat = 0, at = start;
        cut = false;
        while (at < end)
        {
            var (skip, size, reset) = Mp3Decoder.NextFrame(bytes, at, end - at, header, ref freeFormat);
            if (size == 0)
            {
                cut = at + skip < end - 4;                                       // bytes left that hold no whole frame
                break;
            }

            int o = at + skip;
            int samples = Mp3Decoder.FrameSamples(bytes, o);
            int layer = 4 - Mp3Decoder.LayerBits(bytes, o);
            int channels = Mp3Decoder.IsMono(bytes, o) ? 1 : 2;
            int side = !Mp3Decoder.TestMpeg1(bytes, o) ? (channels == 1 ? 9 : 17) : channels == 1 ? 17 : 32;
            int main = Math.Max(0, size - 4 - side - (Mp3Decoder.IsCrc(bytes, o) ? 2 : 0));
            frames.Add(new Frame(o, size, samples, channels, layer, reset, main));
            at = o + size;
        }

        return frames;
    }

    // A Xing or Info tag in the first frame (after its side information): the frame count and, from a LAME (or Lavc)
    // extension, the encoder's delay and padding, the decoder's 529 samples added to the delay and taken from the padding.
    private static VbrTag? ReadTag(byte[] b, int frame, int frameSize)
    {
        if (4 - Mp3Decoder.LayerBits(b, frame) != 3)
        {
            return null;
        }

        int channels = Mp3Decoder.IsMono(b, frame) ? 1 : 2;
        int side = !Mp3Decoder.TestMpeg1(b, frame) ? (channels == 1 ? 9 : 17) : channels == 1 ? 17 : 32;
        int tag = frame + 4 + (Mp3Decoder.IsCrc(b, frame) ? 2 : 0) + side;
        if (tag + 16 > frame + frameSize)
        {
            return null;
        }

        var id = b.AsSpan(tag, 4);
        if (!id.SequenceEqual("Xing"u8) && !id.SequenceEqual("Info"u8))
        {
            return null;
        }

        int flags = b[tag + 7];
        if ((flags & 1) == 0)
        {
            return null;
        }

        int p = tag + 8;
        long frames = BinaryPrimitives.ReadUInt32BigEndian(b.AsSpan(p));
        p += 4;
        if ((flags & 2) != 0)
        {
            p += 4;
        }

        if ((flags & 4) != 0)
        {
            p += 100;
        }

        if ((flags & 8) != 0)
        {
            p += 4;
        }

        int delay = 0, padding = 0;
        if (p < frame + frameSize && b[p] != 0)
        {
            p += 21;
            if (p - frame + 14 >= frameSize)
            {
                return null;
            }

            delay = ((b[p] << 4) | (b[p + 1] >> 4)) + DecoderDelay;
            padding = (((b[p + 1] & 0xF) << 8) | b[p + 2]) - DecoderDelay;
        }

        return new VbrTag(frames, delay, padding);
    }
}
