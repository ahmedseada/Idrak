// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Mp3Audio.Mp3Tables;

namespace Idrak.Mp3Audio;

/// <summary>What one decoded frame was.</summary>
/// <param name="FrameBytes">The bytes it took from the input, the bytes skipped before it included (0: no frame found).</param>
/// <param name="Offset">The bytes skipped before the frame (junk, or a lost sync).</param>
/// <param name="Channels">1 or 2.</param>
/// <param name="SampleRate">Samples a second.</param>
/// <param name="Layer">1, 2 or 3.</param>
internal readonly record struct Mp3FrameInfo(int FrameBytes, int Offset, int Channels, int SampleRate, int Layer);

/// <summary>
/// An MPEG-1, MPEG-2 and MPEG-2.5 Layer II and Layer III frame decoder: a port of minimp3's scalar paths
/// (https://github.com/lieff/minimp3, CC0, public domain), its float arithmetic in the same order. One decoder holds one
/// stream's state (the bit reservoir, the IMDCT overlap, the synthesis filter's history); frames go through
/// <see cref="DecodeFrame"/> in order.
/// </summary>
internal sealed class Mp3Decoder
{
    private const int MaxFreeFormatFrameSize = 2304, MaxFrameSyncMatches = 10, MaxBitReservoirBytes = 511, HeaderSize = 4;
    private const int ShortBlockType = 2, StopBlockType = 3, ModeMono = 3, ModeJointStereo = 1;
    private const int MaxScfi = (255 - 4 - 210 + 3) & ~3;                  // BITS_DEQUANTIZER_OUT -1: MAX_SCF 41, rounded up to 44

    // The stream's state.
    private readonly float[] _mdctOverlap = new float[2 * 9 * 32];
    private readonly float[] _qmfState = new float[15 * 2 * 32];
    private readonly byte[] _reservBuf = new byte[MaxBitReservoirBytes];
    private readonly byte[] _header = new byte[4];
    private int _reserv, _freeFormatBytes;

    // A frame's scratch: the main data (its reservoir bytes and its own; padded for the Huffman reader's look-ahead).
    private readonly byte[] _maindata = new byte[MaxBitReservoirBytes + MaxFreeFormatFrameSize + 4096];
    private readonly GranuleInfo[] _grInfo = [new(), new(), new(), new()];
    private readonly float[] _grbuf = new float[2 * 576];
    private readonly float[] _scf = new float[40];
    private readonly float[] _syn = new float[33 * 64];
    private readonly byte[] _istPos = new byte[2 * 39];
    private readonly BitStream _frameBits = new(), _mainBits = new();

    private sealed class GranuleInfo
    {
        public byte[] Sfb = ScfLong;
        public int SfbAt, Part23Length, BigValues, ScalefacCompress;
        public int GlobalGain, BlockType, MixedBlockFlag, NLongSfb, NShortSfb;
        public readonly int[] TableSelect = new int[3], RegionCount = new int[3], SubblockGain = new int[3];
        public int Preflag, ScalefacScale, Count1Table, Scfsi;
    }

    private sealed class BitStream
    {
        public byte[] Buf = [];
        public int Offset, Pos, Limit;

        public void Init(byte[] buf, int offset, int bytes)
        {
            Buf = buf;
            Offset = offset;
            Pos = 0;
            Limit = bytes * 8;
        }

        public uint GetBits(int n)
        {
            uint cache = 0, s = (uint)(Pos & 7);
            int shl = n + (int)s;
            int p = Offset + (Pos >> 3);
            if ((Pos += n) > Limit)
            {
                return 0;
            }

            uint next = (uint)(Buf[p++] & (255 >> (int)s));
            while ((shl -= 8) > 0)
            {
                cache |= next << shl;
                next = Buf[p++];
            }

            return cache | (next >> -shl);
        }
    }

    /// <summary>Forgets the stream's state (as at its start).</summary>
    public void Reset()
    {
        Array.Clear(_mdctOverlap);
        Array.Clear(_qmfState);
        Array.Clear(_reservBuf);
        Array.Clear(_header);
        _reserv = 0;
        _freeFormatBytes = 0;
    }

    // ---- the frame header

    internal static bool IsMono(byte[] h, int o) => (h[o + 3] & 0xC0) == 0xC0;

    internal static bool IsMsStereo(byte[] h, int o) => (h[o + 3] & 0xE0) == 0x60;

    internal static bool IsFreeFormat(byte[] h, int o) => (h[o + 2] & 0xF0) == 0;

    internal static bool IsCrc(byte[] h, int o) => (h[o + 1] & 1) == 0;

    internal static bool TestPadding(byte[] h, int o) => (h[o + 2] & 0x2) != 0;

    internal static bool TestMpeg1(byte[] h, int o) => (h[o + 1] & 0x8) != 0;

    internal static bool TestNotMpeg25(byte[] h, int o) => (h[o + 1] & 0x10) != 0;

    internal static bool TestIStereo(byte[] h, int o) => (h[o + 3] & 0x10) != 0;

    internal static bool TestMsStereo(byte[] h, int o) => (h[o + 3] & 0x20) != 0;

    internal static int StereoMode(byte[] h, int o) => (h[o + 3] >> 6) & 3;

    internal static int StereoModeExt(byte[] h, int o) => (h[o + 3] >> 4) & 3;

    internal static int LayerBits(byte[] h, int o) => (h[o + 1] >> 1) & 3;

    internal static int BitrateIndex(byte[] h, int o) => h[o + 2] >> 4;

    internal static int SampleRateIndex(byte[] h, int o) => (h[o + 2] >> 2) & 3;

    internal static int MySampleRate(byte[] h, int o) => SampleRateIndex(h, o) + (((h[o + 1] >> 3) & 1) + ((h[o + 1] >> 4) & 1)) * 3;

    internal static bool IsFrame576(byte[] h, int o) => (h[o + 1] & 14) == 2;

    internal static bool IsLayer1(byte[] h, int o) => (h[o + 1] & 6) == 6;

    internal static bool HeaderValid(byte[] h, int o) =>
        h[o] == 0xff && ((h[o + 1] & 0xF0) == 0xf0 || (h[o + 1] & 0xFE) == 0xe2) && LayerBits(h, o) != 0 && BitrateIndex(h, o) != 15 && SampleRateIndex(h, o) != 3;

    internal static bool HeaderCompare(byte[] h1, int o1, byte[] h2, int o2) =>
        HeaderValid(h2, o2) && ((h1[o1 + 1] ^ h2[o2 + 1]) & 0xFE) == 0 && ((h1[o1 + 2] ^ h2[o2 + 2]) & 0x0C) == 0 && !(IsFreeFormat(h1, o1) ^ IsFreeFormat(h2, o2));

    internal static int BitrateKbps(byte[] h, int o) => 2 * HalfRate[(TestMpeg1(h, o) ? 1 : 0) * 45 + (LayerBits(h, o) - 1) * 15 + BitrateIndex(h, o)];

    internal static int SampleRateHz(byte[] h, int o)
    {
        int hz = SampleRateIndex(h, o) switch { 0 => 44100, 1 => 48000, _ => 32000 };
        return hz >> (TestMpeg1(h, o) ? 0 : 1) >> (TestNotMpeg25(h, o) ? 0 : 1);
    }

    internal static int FrameSamples(byte[] h, int o) => IsLayer1(h, o) ? 384 : 1152 >> (IsFrame576(h, o) ? 1 : 0);

    internal static int FrameBytes(byte[] h, int o, int freeFormatSize)
    {
        int bytes = FrameSamples(h, o) * BitrateKbps(h, o) * 125 / SampleRateHz(h, o);
        if (IsLayer1(h, o))
        {
            bytes &= ~3;                                                         // slot align
        }

        return bytes != 0 ? bytes : freeFormatSize;
    }

    internal static int Padding(byte[] h, int o) => TestPadding(h, o) ? (IsLayer1(h, o) ? 4 : 1) : 0;

    // Whether the frames after the one at `o` keep its header (up to MaxFrameSyncMatches of them, or to the end).
    private static bool MatchFrame(byte[] buf, int o, int bytes, int frameBytes)
    {
        int i = 0;
        for (int matches = 0; matches < MaxFrameSyncMatches; matches++)
        {
            i += FrameBytes(buf, o + i, frameBytes) + Padding(buf, o + i);
            if (i + HeaderSize > bytes)
            {
                return matches > 0;
            }

            if (!HeaderCompare(buf, o, buf, o + i))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>The offset of the first frame in buf[o..o+bytes) (bytes when none), its size with padding in <paramref name="frameSize"/> (0: none).</summary>
    internal static int FindFrame(byte[] buf, int o, int bytes, ref int freeFormatBytes, out int frameSize)
    {
        for (int i = 0; i < bytes - HeaderSize; i++)
        {
            int at = o + i;
            if (!HeaderValid(buf, at))
            {
                continue;
            }

            int frameBytes = FrameBytes(buf, at, freeFormatBytes);
            int frameAndPadding = frameBytes + Padding(buf, at);
            for (int k = HeaderSize; frameBytes == 0 && k < MaxFreeFormatFrameSize && i + 2 * k < bytes - HeaderSize; k++)
            {
                if (HeaderCompare(buf, at, buf, at + k))
                {
                    int fb = k - Padding(buf, at);
                    int nextfb = fb + Padding(buf, at + k);
                    if (i + k + nextfb + HeaderSize > bytes || !HeaderCompare(buf, at, buf, at + k + nextfb))
                    {
                        continue;
                    }

                    frameAndPadding = k;
                    frameBytes = fb;
                    freeFormatBytes = fb;
                }
            }

            if ((frameBytes != 0 && i + frameAndPadding <= bytes && MatchFrame(buf, at, bytes - i, frameBytes)) || (i == 0 && frameAndPadding == bytes))
            {
                frameSize = frameAndPadding;
                return i;
            }

            freeFormatBytes = 0;
        }

        frameSize = 0;
        return bytes;
    }

    /// <summary>
    /// The next frame as <see cref="DecodeFrame"/> would take it, without decoding it: its offset from o, its size with
    /// padding (0: none), and whether the decoder's state starts over there (a lost sync found again, or the first frame).
    /// <paramref name="header"/> and <paramref name="freeFormatBytes"/> carry the walk's state as the decoder carries its own.
    /// </summary>
    internal static (int Skip, int Size, bool Reset) NextFrame(byte[] buf, int o, int bytes, byte[] header, ref int freeFormatBytes)
    {
        int frameSize = 0, i = 0;
        bool reset = false;
        if (bytes > 4 && header[0] == 0xff && HeaderCompare(header, 0, buf, o))
        {
            frameSize = FrameBytes(buf, o, freeFormatBytes) + Padding(buf, o);
            if (frameSize != bytes && (frameSize + HeaderSize > bytes || !HeaderCompare(buf, o, buf, o + frameSize)))
            {
                frameSize = 0;
            }
        }

        if (frameSize == 0)
        {
            reset = true;
            Array.Clear(header);
            freeFormatBytes = 0;
            i = FindFrame(buf, o, bytes, ref freeFormatBytes, out frameSize);
            if (frameSize == 0 || i + frameSize > bytes)
            {
                return (i, 0, true);
            }
        }

        Array.Copy(buf, o + i, header, 0, HeaderSize);
        return (i, frameSize, reset);
    }

    /// <summary>
    /// Decodes the frame at buf[at..at+size) (found by <see cref="NextFrame"/>) into <paramref name="pcm"/>, the state
    /// reset first when <paramref name="reset"/> (as the walk said): the samples per channel it gave (0 when its bit
    /// reservoir is not there yet).
    /// </summary>
    public int DecodeFrameAt(byte[] buf, int at, int size, bool reset, float[] pcm)
    {
        if (reset)
        {
            Reset();
        }

        return DecodeFound(buf, at, size, pcm);
    }

    /// <summary>
    /// Decodes the next frame of buf[o..o+bytes) into <paramref name="pcm"/> (interleaved floats in [-1, 1] for a full
    /// scale, at least 1152 × 2): the samples per channel it gave (0 when the frame's bit reservoir is not there yet, or
    /// no frame was found: <see cref="Mp3FrameInfo.FrameBytes"/> 0).
    /// </summary>
    public int DecodeFrame(byte[] buf, int o, int bytes, float[] pcm, out Mp3FrameInfo info)
    {
        int i = 0, frameSize = 0;
        if (bytes > 4 && _header[0] == 0xff && HeaderCompare(_header, 0, buf, o))
        {
            frameSize = FrameBytes(buf, o, _freeFormatBytes) + Padding(buf, o);
            if (frameSize != bytes && (frameSize + HeaderSize > bytes || !HeaderCompare(buf, o, buf, o + frameSize)))
            {
                frameSize = 0;
            }
        }

        if (frameSize == 0)
        {
            Reset();
            i = FindFrame(buf, o, bytes, ref _freeFormatBytes, out frameSize);
            if (frameSize == 0 || i + frameSize > bytes)
            {
                info = new Mp3FrameInfo(i, i, 0, 0, 0);
                return 0;
            }
        }

        int h = o + i;
        info = new Mp3FrameInfo(i + frameSize, i, IsMono(buf, h) ? 1 : 2, SampleRateHz(buf, h), 4 - LayerBits(buf, h));
        return DecodeFound(buf, h, frameSize, pcm);
    }

    private int DecodeFound(byte[] buf, int h, int frameSize, float[] pcm)
    {
        Array.Copy(buf, h, _header, 0, HeaderSize);
        int channels = IsMono(buf, h) ? 1 : 2;
        int layer = 4 - LayerBits(buf, h);
        var bs = _frameBits;
        bs.Init(buf, h + HeaderSize, frameSize - HeaderSize);
        if (IsCrc(buf, h))
        {
            bs.GetBits(16);
        }

        bool success = true;
        if (layer == 3)
        {
            int mainDataBegin = ReadSideInfo(bs, 0, buf, h);
            if (mainDataBegin < 0 || bs.Pos > bs.Limit)
            {
                Reset();
                return 0;
            }

            success = RestoreReservoir(bs, mainDataBegin);
            if (success)
            {
                int granules = TestMpeg1(buf, h) ? 2 : 1;
                for (int igr = 0; igr < granules; igr++)
                {
                    Array.Clear(_grbuf);
                    DecodeGranule(igr * channels, channels);
                    SynthGranule(18, channels, pcm, 576 * channels * igr);
                }
            }

            SaveReservoir();
        }
        else
        {
            var sci = new ScaleInfo();
            ReadScaleInfo(buf, h, bs, sci);
            Array.Clear(_grbuf);
            int at = 0, outAt = 0;
            for (int igr = 0; igr < 3; igr++)
            {
                if (12 == (at += DequantizeGranule(at, bs, sci, layer | 1)))
                {
                    at = 0;
                    ApplyScf384(sci, igr);
                    SynthGranule(12, channels, pcm, outAt);
                    Array.Clear(_grbuf);
                    outAt += 384 * channels;
                }

                if (bs.Pos > bs.Limit)
                {
                    Reset();
                    return 0;
                }
            }
        }

        return success ? FrameSamples(_header, 0) : 0;
    }

    // ---- Layer III side information and scale factors

    private int ReadSideInfo(BitStream bs, int grBase, byte[] h, int ho)
    {
        int srIdx = MySampleRate(h, ho);
        srIdx -= srIdx != 0 ? 1 : 0;
        int grCount = IsMono(h, ho) ? 1 : 2;
        uint scfsi = 0;
        int mainDataBegin, part23Sum = 0;
        bool mpeg1 = TestMpeg1(h, ho);
        if (mpeg1)
        {
            grCount *= 2;
            mainDataBegin = (int)bs.GetBits(9);
            scfsi = bs.GetBits(7 + grCount);
        }
        else
        {
            mainDataBegin = (int)(bs.GetBits(8 + grCount) >> grCount);
        }

        int g = grBase;
        do
        {
            var gr = _grInfo[g];
            if (IsMono(h, ho))
            {
                scfsi <<= 4;
            }

            gr.Part23Length = (int)bs.GetBits(12);
            part23Sum += gr.Part23Length;
            gr.BigValues = (int)bs.GetBits(9);
            if (gr.BigValues > 288)
            {
                return -1;
            }

            gr.GlobalGain = (int)bs.GetBits(8);
            gr.ScalefacCompress = (int)bs.GetBits(mpeg1 ? 4 : 9);
            gr.Sfb = ScfLong;
            gr.SfbAt = srIdx * 23;
            gr.NLongSfb = 22;
            gr.NShortSfb = 0;
            uint tables;
            if (bs.GetBits(1) != 0)
            {
                gr.BlockType = (int)bs.GetBits(2);
                if (gr.BlockType == 0)
                {
                    return -1;
                }

                gr.MixedBlockFlag = (int)bs.GetBits(1);
                gr.RegionCount[0] = 7;
                gr.RegionCount[1] = 255;
                if (gr.BlockType == ShortBlockType)
                {
                    scfsi &= 0x0F0F;
                    if (gr.MixedBlockFlag == 0)
                    {
                        gr.RegionCount[0] = 8;
                        gr.Sfb = ScfShort;
                        gr.SfbAt = srIdx * 40;
                        gr.NLongSfb = 0;
                        gr.NShortSfb = 39;
                    }
                    else
                    {
                        gr.Sfb = ScfMixed;
                        gr.SfbAt = srIdx * 40;
                        gr.NLongSfb = mpeg1 ? 8 : 6;
                        gr.NShortSfb = 30;
                    }
                }

                tables = bs.GetBits(10);
                tables <<= 5;
                gr.SubblockGain[0] = (int)bs.GetBits(3);
                gr.SubblockGain[1] = (int)bs.GetBits(3);
                gr.SubblockGain[2] = (int)bs.GetBits(3);
            }
            else
            {
                gr.BlockType = 0;
                gr.MixedBlockFlag = 0;
                tables = bs.GetBits(15);
                gr.RegionCount[0] = (int)bs.GetBits(4);
                gr.RegionCount[1] = (int)bs.GetBits(3);
                gr.RegionCount[2] = 255;
            }

            gr.TableSelect[0] = (int)(tables >> 10);
            gr.TableSelect[1] = (int)((tables >> 5) & 31);
            gr.TableSelect[2] = (int)(tables & 31);
            gr.Preflag = mpeg1 ? (int)bs.GetBits(1) : gr.ScalefacCompress >= 500 ? 1 : 0;
            gr.ScalefacScale = (int)bs.GetBits(1);
            gr.Count1Table = (int)bs.GetBits(1);
            gr.Scfsi = (int)((scfsi >> 12) & 15);
            scfsi <<= 4;
            g++;
        }
        while (--grCount > 0);

        if (part23Sum + bs.Pos > bs.Limit + mainDataBegin * 8)
        {
            return -1;
        }

        return mainDataBegin;
    }

    private static void ReadScalefactors(Span<byte> scf, byte[] istPos, int istAt, ReadOnlySpan<byte> scfSize, byte[] counts, int countAt, BitStream bs, int scfsi)
    {
        int s0 = 0;
        for (int i = 0; i < 4 && counts[countAt + i] != 0; i++, scfsi *= 2)
        {
            int cnt = counts[countAt + i];
            if ((scfsi & 8) != 0)
            {
                istPos.AsSpan(istAt, cnt).CopyTo(scf[s0..]);
            }
            else
            {
                int bits = scfSize[i];
                if (bits == 0)
                {
                    scf.Slice(s0, cnt).Clear();
                    istPos.AsSpan(istAt, cnt).Clear();
                }
                else
                {
                    int maxScf = scfsi < 0 ? (1 << bits) - 1 : -1;
                    for (int k = 0; k < cnt; k++)
                    {
                        int s = (int)bs.GetBits(bits);
                        istPos[istAt + k] = (byte)(s == maxScf ? -1 : s);
                        scf[s0 + k] = (byte)s;
                    }
                }
            }

            istAt += cnt;
            s0 += cnt;
        }

        scf[s0] = scf[s0 + 1] = scf[s0 + 2] = 0;
    }

    private static float LdexpQ2(float y, int expQ2)
    {
        int e;
        do
        {
            e = Math.Min(30 * 4, expQ2);
            y *= ExpFrac[e & 3] * (1 << 30 >> (e >> 2));
        }
        while ((expQ2 -= e) > 0);
        return y;
    }

    private void DecodeScalefactors(byte[] h, int ho, int istAt, BitStream bs, GranuleInfo gr, int ch)
    {
        int partitionAt = ((gr.NShortSfb != 0 ? 1 : 0) + (gr.NLongSfb == 0 ? 1 : 0)) * 28;
        Span<byte> scfSize = stackalloc byte[4];
        Span<byte> iscf = stackalloc byte[40];
        iscf.Clear();
        int scfShift = gr.ScalefacScale + 1, scfsi = gr.Scfsi;
        if (TestMpeg1(h, ho))
        {
            int part = ScfcDecode[gr.ScalefacCompress];
            scfSize[1] = scfSize[0] = (byte)(part >> 2);
            scfSize[3] = scfSize[2] = (byte)(part & 3);
        }
        else
        {
            int ist = TestIStereo(h, ho) && ch != 0 ? 1 : 0;
            int sfc = gr.ScalefacCompress >> ist, k, modprod;
            for (k = ist * 3 * 4; sfc >= 0; sfc -= modprod, k += 4)
            {
                modprod = 1;
                for (int i = 3; i >= 0; i--)
                {
                    scfSize[i] = (byte)(sfc / modprod % Mod[k + i]);
                    modprod *= Mod[k + i];
                }
            }

            partitionAt += k;
            scfsi = -16;
        }

        ReadScalefactors(iscf, _istPos, istAt, scfSize, ScfPartitions, partitionAt, bs, scfsi);
        if (gr.NShortSfb != 0)
        {
            int sh = 3 - scfShift;
            for (int i = 0; i < gr.NShortSfb; i += 3)
            {
                iscf[gr.NLongSfb + i + 0] = (byte)(iscf[gr.NLongSfb + i + 0] + (gr.SubblockGain[0] << sh));
                iscf[gr.NLongSfb + i + 1] = (byte)(iscf[gr.NLongSfb + i + 1] + (gr.SubblockGain[1] << sh));
                iscf[gr.NLongSfb + i + 2] = (byte)(iscf[gr.NLongSfb + i + 2] + (gr.SubblockGain[2] << sh));
            }
        }
        else if (gr.Preflag != 0)
        {
            for (int i = 0; i < 10; i++)
            {
                iscf[11 + i] = (byte)(iscf[11 + i] + Preamp[i]);
            }
        }

        int gainExp = gr.GlobalGain - 4 - 210 - (IsMsStereo(h, ho) ? 2 : 0);
        float gain = LdexpQ2(1 << (MaxScfi / 4), MaxScfi - gainExp);
        for (int i = 0; i < gr.NLongSfb + gr.NShortSfb; i++)
        {
            _scf[i] = LdexpQ2(gain, iscf[i] << scfShift);
        }
    }

    private static float Pow43Of(int x)
    {
        int mult = 256;
        if (x < 129)
        {
            return Pow43[16 + x];
        }

        if (x < 1024)
        {
            mult = 16;
            x <<= 3;
        }

        int sign = 2 * x & 64;
        float frac = (float)((x & 63) - sign) / ((x & ~63) + sign);
        return Pow43[16 + ((x + sign) >> 6)] * (1f + frac * ((4f / 3) + frac * (2f / 9))) * mult;
    }

    // ---- Layer III Huffman decoding (minimp3's L3_huffman, its bit cache as locals)

    private void Huffman(int dst, BitStream bs, GranuleInfo gr, int layer3grLimit)
    {
        var grbuf = _grbuf;
        float one = 0f;
        int ireg = 0, bigValCnt = gr.BigValues, np;
        byte[] sfbTable = gr.Sfb;
        int sfb = gr.SfbAt, scfAt = 0;
        byte[] buf = bs.Buf;
        int next = bs.Offset + bs.Pos / 8;
        uint cache = ((((buf[next] * 256u) + buf[next + 1]) * 256u + buf[next + 2]) * 256u + buf[next + 3]) << (bs.Pos & 7);
        int sh = (bs.Pos & 7) - 8;
        next += 4;

        while (bigValCnt > 0)
        {
            int tabNum = gr.TableSelect[ireg];
            int sfbCnt = gr.RegionCount[ireg++];
            int codebook = TabIndex[tabNum];
            int linbits = Linbits[tabNum];
            do
            {
                np = sfbTable[sfb++] / 2;
                int pairs = Math.Min(bigValCnt, np);
                one = _scf[scfAt++];
                do
                {
                    int w = 5;
                    int leaf = Tabs[codebook + (int)(cache >> (32 - w))];
                    while (leaf < 0)
                    {
                        cache <<= w;
                        sh += w;
                        w = leaf & 7;
                        leaf = Tabs[codebook + (int)Peek(cache, w) - (leaf >> 3)];
                    }

                    int bits = leaf >> 8;
                    cache <<= bits;
                    sh += bits;
                    for (int j = 0; j < 2; j++, dst++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        if (lsb == 15 && linbits != 0)
                        {
                            lsb += (int)Peek(cache, linbits);
                            cache <<= linbits;
                            sh += linbits;
                            while (sh >= 0)
                            {
                                cache |= (uint)buf[next++] << sh;
                                sh -= 8;
                            }

                            grbuf[dst] = one * Pow43Of(lsb) * ((int)cache < 0 ? -1 : 1);
                        }
                        else
                        {
                            grbuf[dst] = Pow43[16 + lsb - 16 * (int)(cache >> 31)] * one;
                        }

                        if (lsb != 0)
                        {
                            cache <<= 1;
                            sh += 1;
                        }
                    }

                    while (sh >= 0)
                    {
                        cache |= (uint)buf[next++] << sh;
                        sh -= 8;
                    }
                }
                while (--pairs > 0);
            }
            while ((bigValCnt -= np) > 0 && --sfbCnt >= 0);
        }

        for (np = 1 - bigValCnt; ; dst += 4)
        {
            byte[] count1 = gr.Count1Table != 0 ? Tab33 : Tab32;
            int leaf = count1[(int)(cache >> 28)];
            if ((leaf & 8) == 0)
            {
                leaf = count1[(leaf >> 3) + (int)Peek(cache << 4, leaf & 3)];
            }

            cache <<= leaf & 7;
            sh += leaf & 7;
            if ((next - bs.Offset) * 8 - 24 + sh > layer3grLimit)
            {
                break;
            }

            if (--np == 0)
            {
                np = sfbTable[sfb++] / 2;
                if (np == 0)
                {
                    break;
                }

                one = _scf[scfAt++];
            }

            if ((leaf & 128) != 0)
            {
                grbuf[dst + 0] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }

            if ((leaf & 64) != 0)
            {
                grbuf[dst + 1] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }

            if (--np == 0)
            {
                np = sfbTable[sfb++] / 2;
                if (np == 0)
                {
                    break;
                }

                one = _scf[scfAt++];
            }

            if ((leaf & 32) != 0)
            {
                grbuf[dst + 2] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }

            if ((leaf & 16) != 0)
            {
                grbuf[dst + 3] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }

            while (sh >= 0)
            {
                cache |= (uint)buf[next++] << sh;
                sh -= 8;
            }
        }

        bs.Pos = layer3grLimit;
    }

    // The top n bits of the cache (0 for none: C# would shift by 32 as by 0).
    private static uint Peek(uint cache, int n) => n == 0 ? 0 : cache >> (32 - n);

    // ---- Layer III stereo

    private void MidsideStereo(int left, int n)
    {
        var g = _grbuf;
        int right = left + 576;
        for (int i = 0; i < n; i++)
        {
            float a = g[left + i], b = g[right + i];
            g[left + i] = a + b;
            g[right + i] = a - b;
        }
    }

    private void IntensityStereoBand(int left, int n, float kl, float kr)
    {
        var g = _grbuf;
        for (int i = 0; i < n; i++)
        {
            g[left + i + 576] = g[left + i] * kr;
            g[left + i] = g[left + i] * kl;
        }
    }

    private void StereoTopBand(int right, byte[] sfb, int sfbAt, int nbands, Span<int> maxBand)
    {
        maxBand[0] = maxBand[1] = maxBand[2] = -1;
        for (int i = 0; i < nbands; i++)
        {
            for (int k = 0; k < sfb[sfbAt + i]; k += 2)
            {
                if (_grbuf[right + k] != 0 || _grbuf[right + k + 1] != 0)
                {
                    maxBand[i % 3] = i;
                    break;
                }
            }

            right += sfb[sfbAt + i];
        }
    }

    private void StereoProcess(int left, int istAt, byte[] sfb, int sfbAt, byte[] h, int ho, ReadOnlySpan<int> maxBand, int mpeg2Sh)
    {
        uint maxPos = TestMpeg1(h, ho) ? 7u : 64u;
        for (int i = 0; sfb[sfbAt + i] != 0; i++)
        {
            uint ipos = _istPos[istAt + i];
            if (i > maxBand[i % 3] && ipos < maxPos)
            {
                float kl, kr, s = TestMsStereo(h, ho) ? 1.41421356f : 1;
                if (TestMpeg1(h, ho))
                {
                    kl = Pan[2 * ipos];
                    kr = Pan[2 * ipos + 1];
                }
                else
                {
                    kl = 1;
                    kr = LdexpQ2(1, (int)((ipos + 1) >> 1 << mpeg2Sh));
                    if ((ipos & 1) != 0)
                    {
                        kl = kr;
                        kr = 1;
                    }
                }

                IntensityStereoBand(left, sfb[sfbAt + i], kl * s, kr * s);
            }
            else if (TestMsStereo(h, ho))
            {
                MidsideStereo(left, sfb[sfbAt + i]);
            }

            left += sfb[sfbAt + i];
        }
    }

    private void IntensityStereo(int istAt, int g, byte[] h, int ho)
    {
        var gr = _grInfo[g];
        Span<int> maxBand = stackalloc int[3];
        int nSfb = gr.NLongSfb + gr.NShortSfb;
        int maxBlocks = gr.NShortSfb != 0 ? 3 : 1;
        StereoTopBand(576, gr.Sfb, gr.SfbAt, nSfb, maxBand);
        if (gr.NLongSfb != 0)
        {
            maxBand[0] = maxBand[1] = maxBand[2] = Math.Max(Math.Max(maxBand[0], maxBand[1]), maxBand[2]);
        }

        for (int i = 0; i < maxBlocks; i++)
        {
            int defaultPos = TestMpeg1(h, ho) ? 3 : 0;
            int itop = nSfb - maxBlocks + i;
            int prev = itop - maxBlocks;
            _istPos[istAt + itop] = (byte)(maxBand[i] >= prev ? defaultPos : _istPos[istAt + prev]);
        }

        StereoProcess(0, istAt, gr.Sfb, gr.SfbAt, h, ho, maxBand, _grInfo[g + 1].ScalefacCompress & 1);
    }

    // ---- Layer III reorder, alias reduction, IMDCT

    private void Reorder(int grbufAt, byte[] sfb, int sfbAt)
    {
        var g = _grbuf;
        var scratch = _syn;
        int src = grbufAt, dst = 0, len;
        for (; (len = sfb[sfbAt]) != 0; sfbAt += 3, src += 2 * len)
        {
            for (int i = 0; i < len; i++, src++)
            {
                scratch[dst++] = g[src];
                scratch[dst++] = g[src + len];
                scratch[dst++] = g[src + 2 * len];
            }
        }

        Array.Copy(scratch, 0, g, grbufAt, dst);
    }

    private void Antialias(int at, int nbands)
    {
        var g = _grbuf;
        for (; nbands > 0; nbands--, at += 18)
        {
            for (int i = 0; i < 8; i++)
            {
                float u = g[at + 18 + i], d = g[at + 17 - i];
                g[at + 18 + i] = u * Aa[i] - d * Aa[8 + i];
                g[at + 17 - i] = u * Aa[8 + i] + d * Aa[i];
            }
        }
    }

    private static void Dct3x9(Span<float> y)
    {
        float s0 = y[0], s2 = y[2], s4 = y[4], s6 = y[6], s8 = y[8];
        float t0 = s0 + s6 * 0.5f;
        s0 -= s6;
        float t4 = (s4 + s2) * 0.93969262f;
        float t2 = (s8 + s2) * 0.76604444f;
        s6 = (s4 - s8) * 0.17364818f;
        s4 += s8 - s2;

        s2 = s0 - s4 * 0.5f;
        y[4] = s4 + s0;
        s8 = t0 - t2 + s6;
        s0 = t0 - t4 + t2;
        s4 = t0 + t4 - s6;

        float s1 = y[1], s3 = y[3], s5 = y[5], s7 = y[7];

        s3 *= 0.86602540f;
        t0 = (s5 + s1) * 0.98480775f;
        t4 = (s5 - s7) * 0.34202014f;
        t2 = (s1 + s7) * 0.64278761f;
        s1 = (s1 - s5 - s7) * 0.86602540f;

        s5 = t0 - s3 - t2;
        s7 = t4 - s3 - t0;
        s3 = t4 + s3 - t2;

        y[0] = s4 - s7;
        y[1] = s2 + s1;
        y[2] = s0 - s3;
        y[3] = s8 + s5;
        y[5] = s8 - s5;
        y[6] = s0 + s3;
        y[7] = s2 - s1;
        y[8] = s4 + s7;
    }

    private void Imdct36(int at, int overlapAt, int windowAt, int nbands)
    {
        var g = _grbuf;
        var overlap = _mdctOverlap;
        Span<float> co = stackalloc float[9];
        Span<float> si = stackalloc float[9];
        for (int j = 0; j < nbands; j++, at += 18, overlapAt += 9)
        {
            co[0] = -g[at];
            si[0] = g[at + 17];
            for (int i = 0; i < 4; i++)
            {
                si[8 - 2 * i] = g[at + 4 * i + 1] - g[at + 4 * i + 2];
                co[1 + 2 * i] = g[at + 4 * i + 1] + g[at + 4 * i + 2];
                si[7 - 2 * i] = g[at + 4 * i + 4] - g[at + 4 * i + 3];
                co[2 + 2 * i] = -(g[at + 4 * i + 3] + g[at + 4 * i + 4]);
            }

            Dct3x9(co);
            Dct3x9(si);
            si[1] = -si[1];
            si[3] = -si[3];
            si[5] = -si[5];
            si[7] = -si[7];
            for (int i = 0; i < 9; i++)
            {
                float ovl = overlap[overlapAt + i];
                float sum = co[i] * Twid9[9 + i] + si[i] * Twid9[0 + i];
                overlap[overlapAt + i] = co[i] * Twid9[0 + i] - si[i] * Twid9[9 + i];
                g[at + i] = ovl * MdctWindow[windowAt + i] - sum * MdctWindow[windowAt + 9 + i];
                g[at + 17 - i] = ovl * MdctWindow[windowAt + 9 + i] + sum * MdctWindow[windowAt + i];
            }
        }
    }

    private static void Idct3(float x0, float x1, float x2, Span<float> dst)
    {
        float m1 = x1 * 0.86602540f;
        float a1 = x0 - x2 * 0.5f;
        dst[1] = x0 + x2;
        dst[0] = a1 + m1;
        dst[2] = a1 - m1;
    }

    // x: 16 values at stride 1 from xAt (read at 0, 3, ..., 15); dst and overlap as offsets into their arrays.
    private static void Imdct12(ReadOnlySpan<float> x, float[] dst, int dstAt, float[] overlap, int overlapAt)
    {
        Span<float> co = stackalloc float[3];
        Span<float> si = stackalloc float[3];
        Idct3(-x[0], x[6] + x[3], x[12] + x[9], co);
        Idct3(x[15], x[12] - x[9], x[6] - x[3], si);
        si[1] = -si[1];
        for (int i = 0; i < 3; i++)
        {
            float ovl = overlap[overlapAt + i];
            float sum = co[i] * Twid3[3 + i] + si[i] * Twid3[0 + i];
            overlap[overlapAt + i] = co[i] * Twid3[0 + i] - si[i] * Twid3[3 + i];
            dst[dstAt + i] = ovl * Twid3[2 - i] - sum * Twid3[5 - i];
            dst[dstAt + 5 - i] = ovl * Twid3[5 - i] + sum * Twid3[2 - i];
        }
    }

    private void ImdctShort(int at, int overlapAt, int nbands)
    {
        var g = _grbuf;
        var overlap = _mdctOverlap;
        Span<float> tmp = stackalloc float[18];
        for (; nbands > 0; nbands--, overlapAt += 9, at += 18)
        {
            g.AsSpan(at, 18).CopyTo(tmp);
            Array.Copy(overlap, overlapAt, g, at, 6);
            Imdct12(tmp, g, at + 6, overlap, overlapAt + 6);
            Imdct12(tmp[1..], g, at + 12, overlap, overlapAt + 6);
            Imdct12(tmp[2..], overlap, overlapAt, overlap, overlapAt + 6);
        }
    }

    private void ChangeSign(int at)
    {
        var g = _grbuf;
        at += 18;
        for (int b = 0; b < 32; b += 2, at += 36)
        {
            for (int i = 1; i < 18; i += 2)
            {
                g[at + i] = -g[at + i];
            }
        }
    }

    private void ImdctGranule(int at, int overlapAt, int blockType, int nLongBands)
    {
        if (nLongBands != 0)
        {
            Imdct36(at, overlapAt, 0, nLongBands);
            at += 18 * nLongBands;
            overlapAt += 9 * nLongBands;
        }

        if (blockType == ShortBlockType)
        {
            ImdctShort(at, overlapAt, 32 - nLongBands);
        }
        else
        {
            Imdct36(at, overlapAt, blockType == StopBlockType ? 18 : 0, 32 - nLongBands);
        }
    }

    // ---- Layer III bit reservoir

    private void SaveReservoir()
    {
        var bs = _mainBits;
        int pos = (bs.Pos + 7) / 8;
        int remains = bs.Limit / 8 - pos;
        if (remains > MaxBitReservoirBytes)
        {
            pos += remains - MaxBitReservoirBytes;
            remains = MaxBitReservoirBytes;
        }

        if (remains > 0)
        {
            Array.Copy(_maindata, pos, _reservBuf, 0, remains);
        }

        _reserv = Math.Max(0, remains);
    }

    private bool RestoreReservoir(BitStream bs, int mainDataBegin)
    {
        int frameBytes = (bs.Limit - bs.Pos) / 8;
        int bytesHave = Math.Min(_reserv, mainDataBegin);
        Array.Copy(_reservBuf, Math.Max(0, _reserv - mainDataBegin), _maindata, 0, bytesHave);
        Array.Copy(bs.Buf, bs.Offset + bs.Pos / 8, _maindata, bytesHave, frameBytes);
        Array.Clear(_maindata, bytesHave + frameBytes, 8);                    // the Huffman reader's look-ahead reads zeros
        _mainBits.Init(_maindata, 0, bytesHave + frameBytes);
        return _reserv >= mainDataBegin;
    }

    private void DecodeGranule(int g, int nch)
    {
        var h = _header;
        for (int ch = 0; ch < nch; ch++)
        {
            int limit = _mainBits.Pos + _grInfo[g + ch].Part23Length;
            DecodeScalefactors(h, 0, ch * 39, _mainBits, _grInfo[g + ch], ch);
            Huffman(576 * ch, _mainBits, _grInfo[g + ch], limit);
        }

        if (TestIStereo(h, 0))
        {
            IntensityStereo(39, g, h, 0);
        }
        else if (IsMsStereo(h, 0))
        {
            MidsideStereo(0, 576);
        }

        for (int ch = 0; ch < nch; ch++)
        {
            var gr = _grInfo[g + ch];
            int aaBands = 31;
            int nLongBands = (gr.MixedBlockFlag != 0 ? 2 : 0) << (MySampleRate(h, 0) == 2 ? 1 : 0);
            if (gr.NShortSfb != 0)
            {
                aaBands = nLongBands - 1;
                Reorder(576 * ch + nLongBands * 18, gr.Sfb, gr.SfbAt + gr.NLongSfb);
            }

            Antialias(576 * ch, aaBands);
            ImdctGranule(576 * ch, 9 * 32 * ch, gr.BlockType, nLongBands);
            ChangeSign(576 * ch);
        }
    }

    // ---- the polyphase synthesis (both layers)

    private void DctII(int at, int n)
    {
        var g = _grbuf;
        Span<float> t = stackalloc float[32];
        for (int k = 0; k < n; k++)
        {
            int y = at + k;
            for (int i = 0; i < 8; i++)
            {
                float x0 = g[y + i * 18];
                float x1 = g[y + (15 - i) * 18];
                float x2 = g[y + (16 + i) * 18];
                float x3 = g[y + (31 - i) * 18];
                float t0 = x0 + x3;
                float t1 = x1 + x2;
                float t2 = (x1 - x2) * Sec[3 * i + 0];
                float t3 = (x0 - x3) * Sec[3 * i + 1];
                t[i] = t0 + t1;
                t[8 + i] = (t0 - t1) * Sec[3 * i + 2];
                t[16 + i] = t3 + t2;
                t[24 + i] = (t3 - t2) * Sec[3 * i + 2];
            }

            for (int r = 0; r < 4; r++)
            {
                var x = t.Slice(8 * r, 8);
                float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7], xt;
                xt = x0 - x7; x0 += x7;
                x7 = x1 - x6; x1 += x6;
                x6 = x2 - x5; x2 += x5;
                x5 = x3 - x4; x3 += x4;
                x4 = x0 - x3; x0 += x3;
                x3 = x1 - x2; x1 += x2;
                x[0] = x0 + x1;
                x[4] = (x0 - x1) * 0.70710677f;
                x5 = x5 + x6;
                x6 = (x6 + x7) * 0.70710677f;
                x7 = x7 + xt;
                x3 = (x3 + x4) * 0.70710677f;
                x5 -= x7 * 0.198912367f;                                         // rotate by PI/8
                x7 += x5 * 0.382683432f;
                x5 -= x7 * 0.198912367f;
                x0 = xt - x6; xt += x6;
                x[1] = (xt + x7) * 0.50979561f;
                x[2] = (x4 + x3) * 0.54119611f;
                x[3] = (x0 - x5) * 0.60134488f;
                x[5] = (x0 + x5) * 0.89997619f;
                x[6] = (x4 - x3) * 1.30656302f;
                x[7] = (xt - x7) * 2.56291556f;
            }

            for (int i = 0; i < 7; i++, y += 4 * 18)
            {
                g[y + 0 * 18] = t[i];
                g[y + 1 * 18] = t[16 + i] + t[24 + i] + t[24 + i + 1];
                g[y + 2 * 18] = t[8 + i] + t[8 + i + 1];
                g[y + 3 * 18] = t[16 + i + 1] + t[24 + i] + t[24 + i + 1];
            }

            g[y + 0 * 18] = t[7];
            g[y + 1 * 18] = t[16 + 7] + t[24 + 7];
            g[y + 2 * 18] = t[8 + 7];
            g[y + 3 * 18] = t[24 + 7];
        }
    }

    private const float PcmScale = 1f / 32768f;

    private static void SynthPair(float[] pcm, int dst, int nch, float[] z, int zAt)
    {
        float a;
        a = (z[zAt + 14 * 64] - z[zAt + 0]) * 29;
        a += (z[zAt + 1 * 64] + z[zAt + 13 * 64]) * 213;
        a += (z[zAt + 12 * 64] - z[zAt + 2 * 64]) * 459;
        a += (z[zAt + 3 * 64] + z[zAt + 11 * 64]) * 2037;
        a += (z[zAt + 10 * 64] - z[zAt + 4 * 64]) * 5153;
        a += (z[zAt + 5 * 64] + z[zAt + 9 * 64]) * 6574;
        a += (z[zAt + 8 * 64] - z[zAt + 6 * 64]) * 37489;
        a += z[zAt + 7 * 64] * 75038;
        pcm[dst] = a * PcmScale;

        zAt += 2;
        a = z[zAt + 14 * 64] * 104;
        a += z[zAt + 12 * 64] * 1567;
        a += z[zAt + 10 * 64] * 9727;
        a += z[zAt + 8 * 64] * 64019;
        a += z[zAt + 6 * 64] * -9975;
        a += z[zAt + 4 * 64] * -45;
        a += z[zAt + 2 * 64] * 146;
        a += z[zAt + 0 * 64] * -5;
        pcm[dst + 16 * nch] = a * PcmScale;
    }

    // One pair of subband sample rows (x at xl, the right channel 576 further when stereo) into 64 output samples.
    private void Synth(int xl, float[] pcm, int dstl, int nch, int lins)
    {
        var g = _grbuf;
        var z = _syn;
        int xr = xl + 576 * (nch - 1);
        int dstr = dstl + (nch - 1);
        int zlin = lins + 15 * 64;
        int w = 0;

        z[zlin + 4 * 15] = g[xl + 18 * 16];
        z[zlin + 4 * 15 + 1] = g[xr + 18 * 16];
        z[zlin + 4 * 15 + 2] = g[xl + 0];
        z[zlin + 4 * 15 + 3] = g[xr + 0];

        z[zlin + 4 * 31] = g[xl + 1 + 18 * 16];
        z[zlin + 4 * 31 + 1] = g[xr + 1 + 18 * 16];
        z[zlin + 4 * 31 + 2] = g[xl + 1];
        z[zlin + 4 * 31 + 3] = g[xr + 1];

        SynthPair(pcm, dstr, nch, z, lins + 4 * 15 + 1);
        SynthPair(pcm, dstr + 32 * nch, nch, z, lins + 4 * 15 + 64 + 1);
        SynthPair(pcm, dstl, nch, z, lins + 4 * 15);
        SynthPair(pcm, dstl + 32 * nch, nch, z, lins + 4 * 15 + 64);

        Span<float> a = stackalloc float[4];
        Span<float> b = stackalloc float[4];
        for (int i = 14; i >= 0; i--)
        {
            z[zlin + 4 * i] = g[xl + 18 * (31 - i)];
            z[zlin + 4 * i + 1] = g[xr + 18 * (31 - i)];
            z[zlin + 4 * i + 2] = g[xl + 1 + 18 * (31 - i)];
            z[zlin + 4 * i + 3] = g[xr + 1 + 18 * (31 - i)];
            z[zlin + 4 * (i + 16)] = g[xl + 1 + 18 * (1 + i)];
            z[zlin + 4 * (i + 16) + 1] = g[xr + 1 + 18 * (1 + i)];
            z[zlin + 4 * (i - 16) + 2] = g[xl + 18 * (1 + i)];
            z[zlin + 4 * (i - 16) + 3] = g[xr + 18 * (1 + i)];

            for (int k = 0; k < 8; k++)
            {
                float w0 = Win[w++], w1 = Win[w++];
                int vz = zlin + 4 * i - k * 64;
                int vy = zlin + 4 * i - (15 - k) * 64;
                for (int j = 0; j < 4; j++)
                {
                    if (k == 0)
                    {
                        b[j] = z[vz + j] * w1 + z[vy + j] * w0;
                        a[j] = z[vz + j] * w0 - z[vy + j] * w1;
                    }
                    else if ((k & 1) == 0)
                    {
                        b[j] += z[vz + j] * w1 + z[vy + j] * w0;
                        a[j] += z[vz + j] * w0 - z[vy + j] * w1;
                    }
                    else
                    {
                        b[j] += z[vz + j] * w1 + z[vy + j] * w0;
                        a[j] += z[vy + j] * w1 - z[vz + j] * w0;
                    }
                }
            }

            pcm[dstr + (15 - i) * nch] = a[1] * PcmScale;
            pcm[dstr + (17 + i) * nch] = b[1] * PcmScale;
            pcm[dstl + (15 - i) * nch] = a[0] * PcmScale;
            pcm[dstl + (17 + i) * nch] = b[0] * PcmScale;
            pcm[dstr + (47 - i) * nch] = a[3] * PcmScale;
            pcm[dstr + (49 + i) * nch] = b[3] * PcmScale;
            pcm[dstl + (47 - i) * nch] = a[2] * PcmScale;
            pcm[dstl + (49 + i) * nch] = b[2] * PcmScale;
        }
    }

    private void SynthGranule(int nbands, int nch, float[] pcm, int pcmAt)
    {
        for (int i = 0; i < nch; i++)
        {
            DctII(576 * i, nbands);
        }

        Array.Copy(_qmfState, 0, _syn, 0, 15 * 64);
        for (int i = 0; i < nbands; i += 2)
        {
            Synth(i, pcm, pcmAt + 32 * nch * i, nch, i * 64);
        }

        if (nch == 1)
        {
            for (int i = 0; i < 15 * 64; i += 2)
            {
                _qmfState[i] = _syn[nbands * 64 + i];
            }
        }
        else
        {
            Array.Copy(_syn, nbands * 64, _qmfState, 0, 15 * 64);
        }
    }

    // ---- Layer II (and I) scale information and dequantisation

    private sealed class ScaleInfo
    {
        public readonly float[] Scf = new float[3 * 64];
        public int TotalBands, StereoBands;
        public readonly byte[] Bitalloc = new byte[64], Scfcod = new byte[64];
    }

    private readonly record struct SubbandAlloc(int TabOffset, int CodeTabWidth, int BandCount);

    private static readonly SubbandAlloc[] AllocL1 = [new(76, 4, 32)];
    private static readonly SubbandAlloc[] AllocL2M2 = [new(60, 4, 4), new(44, 3, 7), new(44, 2, 19)];
    private static readonly SubbandAlloc[] AllocL2M1 = [new(0, 4, 3), new(16, 4, 8), new(32, 3, 12), new(40, 2, 7)];
    private static readonly SubbandAlloc[] AllocL2M1LowRate = [new(44, 4, 2), new(44, 3, 10)];

    private static readonly float[] DeqL12 = MakeDeqL12();

    private static float[] MakeDeqL12()
    {
        int[] divisors = [3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 3, 5, 9];
        var values = new float[18 * 3];
        for (int i = 0; i < divisors.Length; i++)
        {
            values[3 * i] = 9.53674316e-07f / divisors[i];
            values[3 * i + 1] = 7.56931807e-07f / divisors[i];
            values[3 * i + 2] = 6.00777173e-07f / divisors[i];
        }

        return values;
    }

    private static SubbandAlloc[] SubbandAllocTable(byte[] h, int ho, ScaleInfo sci)
    {
        SubbandAlloc[] alloc;
        int mode = StereoMode(h, ho);
        int nbands, stereoBands = mode == ModeMono ? 0 : mode == ModeJointStereo ? (StereoModeExt(h, ho) << 2) + 4 : 32;
        if (IsLayer1(h, ho))
        {
            alloc = AllocL1;
            nbands = 32;
        }
        else if (!TestMpeg1(h, ho))
        {
            alloc = AllocL2M2;
            nbands = 30;
        }
        else
        {
            alloc = AllocL2M1;
            int sampleRateIdx = SampleRateIndex(h, ho);
            int kbps = BitrateKbps(h, ho) >> (mode != ModeMono ? 1 : 0);
            if (kbps == 0)
            {
                kbps = 192;                                                      // free format
            }

            nbands = 27;
            if (kbps < 56)
            {
                alloc = AllocL2M1LowRate;
                nbands = sampleRateIdx == 2 ? 12 : 8;
            }
            else if (kbps >= 96 && sampleRateIdx != 1)
            {
                nbands = 30;
            }
        }

        sci.TotalBands = nbands;
        sci.StereoBands = Math.Min(stereoBands, nbands);
        return alloc;
    }

    private static void ReadL12Scalefactors(BitStream bs, byte[] bitalloc, byte[] scfcod, int bands, float[] scf)
    {
        int at = 0;
        for (int i = 0; i < bands; i++)
        {
            float s = 0;
            int ba = bitalloc[i];
            int mask = ba != 0 ? 4 + ((19 >> scfcod[i]) & 3) : 0;
            for (int m = 4; m != 0; m >>= 1)
            {
                if ((mask & m) != 0)
                {
                    int b = (int)bs.GetBits(6);
                    s = DeqL12[ba * 3 - 6 + b % 3] * (1 << 21 >> b / 3);
                }

                scf[at++] = s;
            }
        }
    }

    private static void ReadScaleInfo(byte[] h, int ho, BitStream bs, ScaleInfo sci)
    {
        var allocs = SubbandAllocTable(h, ho, sci);
        int next = 0, k = 0, baBits = 0, codeTab = 0;
        for (int i = 0; i < sci.TotalBands; i++)
        {
            if (i == k)
            {
                var a = allocs[next++];
                k += a.BandCount;
                baBits = a.CodeTabWidth;
                codeTab = a.TabOffset;
            }

            byte ba = BitallocCodeTab[codeTab + (int)bs.GetBits(baBits)];
            sci.Bitalloc[2 * i] = ba;
            if (i < sci.StereoBands)
            {
                ba = BitallocCodeTab[codeTab + (int)bs.GetBits(baBits)];
            }

            sci.Bitalloc[2 * i + 1] = sci.StereoBands != 0 ? ba : (byte)0;
        }

        for (int i = 0; i < 2 * sci.TotalBands; i++)
        {
            sci.Scfcod[i] = sci.Bitalloc[i] != 0 ? IsLayer1(h, ho) ? (byte)2 : (byte)bs.GetBits(2) : (byte)6;
        }

        ReadL12Scalefactors(bs, sci.Bitalloc, sci.Scfcod, sci.TotalBands * 2, sci.Scf);
        for (int i = sci.StereoBands; i < sci.TotalBands; i++)
        {
            sci.Bitalloc[2 * i + 1] = 0;
        }
    }

    private int DequantizeGranule(int at, BitStream bs, ScaleInfo sci, int groupSize)
    {
        var g = _grbuf;
        int choff = 576;
        for (int j = 0; j < 4; j++)
        {
            int dst = at + groupSize * j;
            for (int i = 0; i < 2 * sci.TotalBands; i++)
            {
                int ba = sci.Bitalloc[i];
                if (ba != 0)
                {
                    if (ba < 17)
                    {
                        int half = (1 << (ba - 1)) - 1;
                        for (int k = 0; k < groupSize; k++)
                        {
                            g[dst + k] = (int)bs.GetBits(ba) - half;
                        }
                    }
                    else
                    {
                        uint mod = (2u << (ba - 17)) + 1;                           // 3, 5, 9
                        uint code = bs.GetBits((int)(mod + 2 - (mod >> 3)));       // 5, 7, 10
                        for (int k = 0; k < groupSize; k++, code /= mod)
                        {
                            g[dst + k] = (int)(code % mod - mod / 2);
                        }
                    }
                }

                dst += choff;
                choff = 18 - choff;
            }
        }

        return groupSize * 4;
    }

    private void ApplyScf384(ScaleInfo sci, int scfAt)
    {
        var g = _grbuf;
        Array.Copy(g, sci.StereoBands * 18, g, 576 + sci.StereoBands * 18, (sci.TotalBands - sci.StereoBands) * 18);
        int dst = 0;
        for (int i = 0; i < sci.TotalBands; i++, dst += 18, scfAt += 6)
        {
            for (int k = 0; k < 12; k++)
            {
                g[dst + k] *= sci.Scf[scfAt];
                g[dst + k + 576] *= sci.Scf[scfAt + 3];
            }
        }
    }
}
