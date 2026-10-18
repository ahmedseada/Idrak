// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Audio;

/// <summary>
/// Writes MP3 files (an <see cref="IAudioEncoder"/> "mp3", the library's, registered in <see cref="AudioEncoders"/>): MPEG-1
/// Layer III at 32, 44.1 and 48 kHz and MPEG-2 Layer III at 16, 22.05 and 24 kHz, one or two channels (left/right or
/// mid/side joint stereo), constant bit rate with the bit reservoir, long blocks. Written from ISO/IEC 11172-3 and
/// 13818-3: the polyphase analysis filterbank and MDCT, a masking model, the rate and distortion loops, the standard's
/// Huffman tables. The stream is plain frames (no ID3, Xing or LAME tag): a decoder gives <see cref="Delay"/> samples of
/// silence before the sound and silence after it to the end of the last frame. <see cref="AudioEncodeOptions"/> are
/// ignored (MP3 has no sample format); the encoder's own are <see cref="Mp3EncodeOptions"/>.
/// </summary>
/// <param name="options">How it encodes; null: the IDRAK_MP3_* variables, the defaults where unset.</param>
public sealed class Mp3Encoder(Mp3EncodeOptions? options = null) : IAudioEncoder
{
    /// <summary>
    /// The samples (at the output's rate) the decoded sound starts later than the encoded one: 481 of the polyphase
    /// filterbanks and 576 of the MDCT's overlap, the same in every decoder that follows the standard.
    /// </summary>
    public const int Delay = 1057;

    private readonly Mp3EncodeOptions? _given = options;
    private Mp3EncodeOptions? _environment;

    /// <inheritdoc />
    public string Name => "mp3";

    /// <inheritdoc />
    public string Extension => ".mp3";

    /// <summary>The options it encodes with: those given, else the environment's (<see cref="Mp3EncodeOptions.FromEnvironment"/>, read at the first use).</summary>
    public Mp3EncodeOptions Options => _given ?? (_environment ??= Mp3EncodeOptions.FromEnvironment());

    /// <inheritdoc />
    /// <exception cref="ArgumentException">An option MP3 does not take (a bit rate or sample rate; the message names those it does).</exception>
    /// <exception cref="NotSupportedException">More than two channels without <see cref="Mp3ChannelMode.Mono"/>.</exception>
    public void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(output);
        var format = Resolve(audio.SampleRate, audio.Channels);
        var source = format.Rate == audio.SampleRate ? audio : Resampler.Resample(audio, format.Rate);
        int length = source.Length;
        float[]? mono = null;
        try
        {
            var planes = new ReadOnlyMemory<float>[format.Channels];
            if (format.Channels == 1 && source.Channels > 1)
            {
                mono = ArrayPool<float>.Shared.Rent(Math.Max(1, length));
                source.ToMono(mono);
                planes[0] = mono.AsMemory(0, length);
            }
            else
            {
                var samples = source.Samples;
                for (int c = 0; c < format.Channels; c++)
                {
                    planes[c] = samples.AsMemory(c * length, length);
                }
            }

            new Mp3Stream(format, Options.FramesPerTask).Write(planes, length, output);
        }
        finally
        {
            if (mono is not null)
            {
                ArrayPool<float>.Shared.Return(mono);
            }
        }
    }

    /// <summary>The stream's format for input of <paramref name="rate"/> Hz and <paramref name="channels"/> channels, the options checked.</summary>
    internal Mp3Format Resolve(int rate, int channels)
    {
        var o = Options;
        if (channels > 2 && o.ChannelMode != Mp3ChannelMode.Mono)
        {
            throw new NotSupportedException($"MP3 holds one or two channels, not {channels}; encode with Mp3ChannelMode.Mono to mix them down.");
        }

        int outRate = o.SampleRate ?? Nearest(rate);
        if (!Mp3EncodeOptions.SampleRates.Contains(outRate))
        {
            throw new ArgumentException($"MP3 is written at {string.Join(", ", Mp3EncodeOptions.SampleRates)} Hz, not {outRate} (Mp3EncodeOptions.SampleRate).");
        }

        int outChannels = o.ChannelMode == Mp3ChannelMode.Mono || channels == 1 ? 1 : 2;
        var bitrates = Mp3EncodeOptions.Bitrates(outRate);
        int bitrate = o.Bitrate ?? Mp3EncodeOptions.DefaultBitrate(outRate, outChannels);
        if (!bitrates.Contains(bitrate))
        {
            throw new ArgumentException($"MP3 at {outRate} Hz ({(outRate >= 32000 ? "MPEG-1" : "MPEG-2")} Layer III) is written at {string.Join(", ", bitrates)} kbit/s, not {bitrate} (Mp3EncodeOptions.Bitrate).");
        }

        if (o.LowpassHz < 0)
        {
            throw new ArgumentException($"Mp3EncodeOptions.LowpassHz is a frequency in Hz (0 for none, null for automatic), not {o.LowpassHz}.");
        }

        int lines = o.LowpassHz switch
        {
            null => AutoLines(outRate, outChannels, bitrate),
            0 => 576,
            int hz => (int)Math.Min(576, ((long)hz * 1152 + outRate - 1) / outRate),
        };
        return new Mp3Format(outRate, outChannels, bitrate, outChannels == 2 && o.ChannelMode == Mp3ChannelMode.JointStereo, lines, o.Quality);
    }

    // The supported rate nearest to `rate` (the higher of two as near).
    private static int Nearest(int rate)
    {
        int best = 0;
        foreach (int r in Mp3EncodeOptions.SampleRates)
        {
            if (best == 0 || Math.Abs(r - rate) <= Math.Abs(best - rate))
            {
                best = r;
            }
        }

        return best;
    }

    // The lines kept by the automatic lowpass: a fraction of the band that grows with the bits a sample and a channel
    // gets (all of it from 2.5 bits), interpolated between measured points.
    private static int AutoLines(int rate, int channels, int bitrate)
    {
        ReadOnlySpan<double> bits = [0.3, 0.5, 0.73, 1.0, 1.45, 2.0, 2.5];
        ReadOnlySpan<double> fraction = [0.25, 0.38, 0.5, 0.65, 0.77, 0.9, 1.0];
        double perSample = bitrate * 1000.0 / ((double)rate * channels);
        double kept = fraction[0];
        if (perSample >= bits[^1])
        {
            kept = 1;
        }
        else
        {
            for (int i = 1; i < bits.Length; i++)
            {
                if (perSample < bits[i])
                {
                    double t = Math.Max(0, (perSample - bits[i - 1]) / (bits[i] - bits[i - 1]));
                    kept = fraction[i - 1] + t * (fraction[i] - fraction[i - 1]);
                    break;
                }
            }
        }

        return Math.Clamp((int)(kept * 576), 1, 576);
    }
}

/// <summary>What one encode writes: its rate, channels, bit rate and the tables that go with them.</summary>
internal sealed class Mp3Format
{
    private static readonly int[] PsyRates = [44100, 48000, 32000, 22050, 24000, 16000];
    private static readonly int[] ScfRows = [5, 6, 7, 2, 3, 4];

    public Mp3Format(int rate, int channels, int bitrate, bool joint, int lines, Mp3Quality quality)
    {
        Rate = rate;
        Channels = channels;
        Bitrate = bitrate;
        Joint = joint;
        Lines = lines;
        Quality = quality;
        Mpeg1 = rate >= 32000;
        Granules = Mpeg1 ? 2 : 1;
        int psy = Array.IndexOf(PsyRates, rate);
        RateIndex = psy % 3;
        BitrateIndex = Mp3EncodeOptions.Bitrates(rate).ToList().IndexOf(bitrate) + 1;
        SideBytes = Mpeg1 ? (channels == 1 ? 17 : 32) : (channels == 1 ? 9 : 17);
        MaxBegin = Mpeg1 ? 511 : 255;
        FrameNumerator = (Mpeg1 ? 144000L : 72000L) * bitrate;
        BandStart = new int[23];
        for (int b = 0; b < 22; b++)
        {
            BandStart[b + 1] = BandStart[b] + Mp3Tables.ScfLong[ScfRows[psy] * 23 + b];
        }

        Spread = Mp3EncoderTables.PsySpread.AsMemory(psy * 22 * 22, 22 * 22);
        Ath = Mp3EncoderTables.PsyAth.AsMemory(psy * 22, 22);
    }

    public int Rate { get; }

    public int Channels { get; }

    public int Bitrate { get; }

    public bool Joint { get; }

    /// <summary>The frequency lines coded (the rest are the lowpass's).</summary>
    public int Lines { get; }

    public Mp3Quality Quality { get; }

    public bool Mpeg1 { get; }

    public int Granules { get; }

    public int RateIndex { get; }

    public int BitrateIndex { get; }

    public int SideBytes { get; }

    /// <summary>The bytes main_data_begin reaches back (the bit reservoir's size).</summary>
    public int MaxBegin { get; }

    /// <summary>A frame's bytes times the rate: 144000 (MPEG-1) or 72000 (MPEG-2) times the kbit/s.</summary>
    public long FrameNumerator { get; }

    public int[] BandStart { get; }

    public ReadOnlyMemory<float> Spread { get; }

    public ReadOnlyMemory<float> Ath { get; }
}

/// <summary>
/// One encode: chunks of frames analysed in parallel (each worker primes its filters on the granule before its first),
/// their bits planned in order (the bit reservoir), their granules quantized in parallel, the frames packed in order.
/// Every granule's lines come from the input alone and every plan from the frames before it, so the bytes do not depend
/// on the chunking or the threads. The hot loops are compiled optimized at once (AggressiveOptimization): an encode is
/// often the only one in its process, and tiered compilation made the first one three to five times slower.
/// </summary>
internal sealed class Mp3Stream(Mp3Format format, int framesPerTask)
{
    /// <summary>The bytes a granule's main data may take (4095 bits), and a little room.</summary>
    public const int UnitBytes = 520;

    private const int Overhead = 60;                                      // a granule's bits beyond its entropy (the scalefactors, the tables' slack)

    private readonly ConcurrentBag<Mp3Analysis> _analyses = [];
    private readonly ConcurrentBag<Mp3Quantizer> _quantizers = [];
    private long _reservoir, _paddingRest;

    public void Write(ReadOnlyMemory<float>[] planes, int length, Stream output)
    {
        var f = format;
        int perFrame = f.Granules * f.Channels;
        long granules = ((long)length + Mp3Encoder.Delay + 575) / 576;
        long frames = (granules + f.Granules - 1) / f.Granules;
        bool parallel = framesPerTask > 0;
        int chunk = (int)Math.Max(1, Math.Min(frames, parallel ? (long)framesPerTask * Math.Max(1, ComputeResources.MaxCpuThreads) : 32));
        int units = chunk * perFrame;
        var xr = ArrayPool<float>.Shared.Rent(units * 576);
        var xmin = ArrayPool<float>.Shared.Rent(units * 22);
        var entropy = ArrayPool<float>.Shared.Rent(units);
        var silent = ArrayPool<bool>.Shared.Rent(units);
        var midSide = ArrayPool<bool>.Shared.Rent(chunk);
        var slots = ArrayPool<int>.Shared.Rent(chunk);
        var paddings = ArrayPool<int>.Shared.Rent(chunk);
        var targets = ArrayPool<int>.Shared.Rent(units);
        var used = ArrayPool<int>.Shared.Rent(units);
        var sides = ArrayPool<Mp3Granule>.Shared.Rent(units);
        var bits = ArrayPool<byte>.Shared.Rent(units * UnitBytes);
        var packer = new Mp3Packer(f, output);
        try
        {
            for (long first = 0; first < frames; first += chunk)
            {
                int n = (int)Math.Min(chunk, frames - first);
                long start = first;
                void Analyse(int a, int b) => AnalyseFrames(planes, start, a, b, xr, xmin, entropy, silent, midSide);
                void Quantize(int a, int b) => QuantizeUnits(a, b, xr, xmin, targets, bits, used, sides);
                if (parallel)
                {
                    HostParallel.For(n, framesPerTask, Analyse);
                }
                else
                {
                    Analyse(0, n);
                }

                for (int i = 0; i < n; i++)
                {
                    Plan(i, entropy, silent, targets, slots, paddings);
                }

                if (parallel)
                {
                    HostParallel.For(n * perFrame, 4, Quantize);
                }
                else
                {
                    Quantize(0, n * perFrame);
                }

                for (int i = 0; i < n; i++)
                {
                    packer.Frame(paddings[i], slots[i], midSide[i], sides.AsSpan(i * perFrame, perFrame), bits.AsSpan(i * perFrame * UnitBytes), used.AsSpan(i * perFrame, perFrame));
                }
            }

            packer.Finish();
        }
        finally
        {
            packer.Dispose();
            ArrayPool<float>.Shared.Return(xr);
            ArrayPool<float>.Shared.Return(xmin);
            ArrayPool<float>.Shared.Return(entropy);
            ArrayPool<bool>.Shared.Return(silent);
            ArrayPool<bool>.Shared.Return(midSide);
            ArrayPool<int>.Shared.Return(slots);
            ArrayPool<int>.Shared.Return(paddings);
            ArrayPool<int>.Shared.Return(targets);
            ArrayPool<int>.Shared.Return(used);
            ArrayPool<Mp3Granule>.Shared.Return(sides);
            ArrayPool<byte>.Shared.Return(bits);
        }
    }

    // Frames [a, b) of the chunk starting at frame `first`: each granule's lines (the lowpass's cleared), its masking
    // thresholds and entropy, and per frame whether mid/side takes fewer bits than left/right.
    private void AnalyseFrames(ReadOnlyMemory<float>[] planes, long first, int a, int b, float[] xr, float[] xmin, float[] entropy, bool[] silent, bool[] midSide)
    {
        var f = format;
        int nch = f.Channels, perFrame = f.Granules * nch;
        var analysis = _analyses.TryTake(out var taken) ? taken : new Mp3Analysis(nch);
        Span<float> energy = stackalloc float[22];
        try
        {
            for (int c = 0; c < nch; c++)
            {
                analysis.Prime(c, planes[c].Span, (first + a) * f.Granules);
            }

            var spread = f.Spread.Span;
            var ath = f.Ath.Span;
            for (int i = a; i < b; i++)
            {
                for (int gr = 0; gr < f.Granules; gr++)
                {
                    for (int c = 0; c < nch; c++)
                    {
                        int u = i * perFrame + gr * nch + c;
                        var lines = xr.AsSpan(u * 576, 576);
                        analysis.Granule(c, planes[c].Span, (first + i) * f.Granules + gr, lines);
                        lines[f.Lines..].Clear();
                        entropy[u] = Mp3Analysis.Masking(lines, f.BandStart, spread, ath, energy, xmin.AsSpan(u * 22, 22));
                        silent[u] = Sum(energy) == 0;
                    }
                }

                midSide[i] = f.Joint && MidSide(i * perFrame, xr, xmin, entropy, silent);
            }
        }
        finally
        {
            _analyses.Add(analysis);
        }
    }

    private static float Sum(ReadOnlySpan<float> values)
    {
        float sum = 0;
        foreach (float v in values)
        {
            sum += v;
        }

        return sum;
    }

    // Whether a frame's granules take fewer bits as mid and side ((L + R) / sqrt 2, (L - R) / sqrt 2, both held to the
    // lower of the two channels' thresholds, so no noise is unmasked) than as left and right; if so they become so.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private bool MidSide(int firstUnit, float[] xr, float[] xmin, float[] entropy, bool[] silent)
    {
        var f = format;
        const float Half = 0.70710677f;
        Span<float> eMid = stackalloc float[22], eSide = stackalloc float[22], limit = stackalloc float[22];
        float leftRight = 0, midSide = 0;
        for (int gr = 0; gr < f.Granules; gr++)
        {
            int u = firstUnit + 2 * gr;
            leftRight += entropy[u] + entropy[u + 1];
            var l = xr.AsSpan(u * 576, 576);
            var r = xr.AsSpan((u + 1) * 576, 576);
            MidSideEnergies(l, r, f.BandStart, eMid, eSide);
            for (int b = 0; b < 22; b++)
            {
                limit[b] = Math.Min(xmin[u * 22 + b], xmin[(u + 1) * 22 + b]);
            }

            midSide += Mp3Analysis.Entropy(eMid, limit, f.BandStart) + Mp3Analysis.Entropy(eSide, limit, f.BandStart);
        }

        if (midSide >= leftRight)
        {
            return false;
        }

        for (int gr = 0; gr < f.Granules; gr++)
        {
            int u = firstUnit + 2 * gr;
            var l = xr.AsSpan(u * 576, 576);
            var r = xr.AsSpan((u + 1) * 576, 576);
            for (int i = 0; i < 576; i++)
            {
                float m = (l[i] + r[i]) * Half, s = (l[i] - r[i]) * Half;
                l[i] = m;
                r[i] = s;
            }

            for (int b = 0; b < 22; b++)
            {
                float low = Math.Min(xmin[u * 22 + b], xmin[(u + 1) * 22 + b]);
                xmin[u * 22 + b] = low;
                xmin[(u + 1) * 22 + b] = low;
            }

            float mid = Mp3Analysis.Energies(l, f.BandStart, eMid), side = Mp3Analysis.Energies(r, f.BandStart, eSide);
            entropy[u] = Mp3Analysis.Entropy(eMid, xmin.AsSpan(u * 22, 22), f.BandStart);
            entropy[u + 1] = Mp3Analysis.Entropy(eSide, xmin.AsSpan(u * 22, 22), f.BandStart);
            silent[u] = mid == 0;
            silent[u + 1] = side == 0;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MidSideEnergies(ReadOnlySpan<float> l, ReadOnlySpan<float> r, ReadOnlySpan<int> bandStart, Span<float> mid, Span<float> side)
    {
        const float Half = 0.70710677f;
        for (int b = 0; b < 22; b++)
        {
            float em = 0, es = 0;
            for (int i = bandStart[b]; i < bandStart[b + 1]; i++)
            {
                float m = (l[i] + r[i]) * Half, s = (l[i] - r[i]) * Half;
                em += m * m;
                es += s * s;
            }

            mid[b] = em;
            side[b] = es;
        }
    }

    // A frame's bits, in order: its size (the padding spreads the fraction of a byte a frame has over the frames), what
    // its granules ask for (their entropy) against the bits it has, the reservoir lent to a frame that asks for more
    // and saved from one that asks for less (never past its size: what would overflow it is spent), then split over the
    // granules by what each asks for. The reservoir is planned as if every granule used all its bits; a granule that
    // uses fewer leaves more in the stream, never less, so every plan fits.
    private void Plan(int frame, float[] entropy, bool[] silent, int[] targets, int[] slots, int[] paddings)
    {
        var f = format;
        int perFrame = f.Granules * f.Channels, first = frame * perFrame;
        int padding = 0;
        _paddingRest += f.FrameNumerator % f.Rate;
        if (_paddingRest >= f.Rate)
        {
            _paddingRest -= f.Rate;
            padding = 1;
        }

        int slot = (int)(f.FrameNumerator / f.Rate) + padding - 4 - f.SideBytes;
        slots[frame] = slot;
        paddings[frame] = padding;
        long mean = 8L * slot, most = 8L * f.MaxBegin, available = Math.Min(_reservoir, most);
        long demand = 0;
        int active = 0;
        for (int u = first; u < first + perFrame; u++)
        {
            if (!silent[u])
            {
                demand += (long)entropy[u] + Overhead;
                active++;
            }
        }

        long total = 0;
        if (active > 0)
        {
            total = demand > mean ? mean + Math.Min(demand - mean, available * 3 / 5) : Math.Max(demand, mean * 3 / 4);
            total = Math.Max(total, mean - (most - available));
            total = Math.Min(Math.Min(total, mean + available), active * 4095L) & ~7L;
        }

        long weights = 0, floor = mean / (perFrame * 20);
        for (int u = first; u < first + perFrame; u++)
        {
            weights += silent[u] ? 0 : (long)entropy[u] + Overhead + floor;
        }

        long planned = 0;
        for (int u = first; u < first + perFrame; u++)
        {
            long weight = silent[u] ? 0 : (long)entropy[u] + Overhead + floor;
            int target = weights == 0 ? 0 : (int)Math.Min(4095, total * weight / weights);
            targets[u] = target;
            planned += target;
        }

        for (int u = first; u < first + perFrame && planned < total; u++)
        {
            if (!silent[u])
            {
                int more = (int)Math.Min(4095 - targets[u], total - planned);         // what rounding and the cap left, to the first that can take it
                targets[u] += more;
                planned += more;
            }
        }

        _reservoir = Math.Min(most, available + mean - ((planned + 7) & ~7L));
    }

    private void QuantizeUnits(int a, int b, float[] xr, float[] xmin, int[] targets, byte[] bits, int[] used, Mp3Granule[] sides)
    {
        var f = format;
        var quantizer = _quantizers.TryTake(out var taken) ? taken : new Mp3Quantizer(f.BandStart, !f.Mpeg1, f.Quality);
        try
        {
            for (int u = a; u < b; u++)
            {
                var buffer = bits.AsSpan(u * UnitBytes, UnitBytes);
                buffer.Clear();
                used[u] = quantizer.Encode(xr.AsSpan(u * 576, 576), xmin.AsSpan(u * 22, 21), targets[u], buffer, out sides[u]);
            }
        }
        finally
        {
            _quantizers.Add(quantizer);
        }
    }
}

/// <summary>
/// Lays frames out: each frame's header and side information, then its slot of main data. A frame's main data starts
/// as far back in the slots before it as main_data_begin reaches (the bit reservoir) and as the data before it ends;
/// the bytes no frame's data takes stay 0. A frame is written once no later frame's data can reach into its slot.
/// </summary>
internal sealed class Mp3Packer(Mp3Format format, Stream output) : IDisposable
{
    private byte[] _slots = Zeroed(1 << 14);                              // the slot bytes from _base on
    private readonly byte[] _staged = ArrayPool<byte>.Shared.Rent(1 << 16);
    private readonly Queue<(byte[] Head, long Start, int Bytes)> _pending = new();
    private readonly Stack<byte[]> _heads = new();                       // headers written out, to be reused
    private long _base, _next, _end;
    private int _stagedBytes;

    private static byte[] Zeroed(int bytes)
    {
        var array = ArrayPool<byte>.Shared.Rent(bytes);
        Array.Clear(array);
        return array;
    }

    public void Frame(int padding, int slot, bool midSide, ReadOnlySpan<Mp3Granule> sides, ReadOnlySpan<byte> bits, ReadOnlySpan<int> used)
    {
        var f = format;
        long start = Math.Max(_end, _next - f.MaxBegin);
        int total = 0;
        foreach (int u in used)
        {
            total += u;
        }

        long end = start + (total + 7) / 8;
        if (end > _next + slot)
        {
            throw new InvalidOperationException($"a frame's {total} bits do not fit its {slot} bytes and the {_next - start} it may borrow (the bit plan is wrong).");
        }

        Reserve(_next + slot);
        long bit = (start - _base) * 8;
        for (int u = 0; u < used.Length; u++)
        {
            Mp3BitWriter.Append(_slots, bit, bits.Slice(u * Mp3Stream.UnitBytes, Mp3Stream.UnitBytes), used[u]);
            bit += used[u];
        }

        var head = _heads.Count > 0 ? _heads.Pop() : new byte[4 + f.SideBytes];
        head[0] = 0xFF;
        head[1] = f.Mpeg1 ? (byte)0xFB : (byte)0xF3;                    // sync, MPEG-1 or MPEG-2, Layer III, no CRC
        head[2] = (byte)((f.BitrateIndex << 4) | (f.RateIndex << 2) | (padding << 1));
        int mode = f.Channels == 1 ? 3 : f.Joint ? 1 : 0;
        head[3] = (byte)((mode << 6) | ((midSide ? 2 : 0) << 4));
        SideInformation(head.AsSpan(4), (int)(_next - start), sides);
        _pending.Enqueue((head, _next, slot));
        _end = end;
        _next += slot;
        Emit(Math.Max(_end, _next - f.MaxBegin));
    }

    public void Finish()
    {
        Emit(long.MaxValue);
        output.Write(_staged, 0, _stagedBytes);
        _stagedBytes = 0;
    }

    public void Dispose()
    {
        ArrayPool<byte>.Shared.Return(_slots);
        ArrayPool<byte>.Shared.Return(_staged);
    }

    // 11172-3 2.4.1.7 and 13818-3 2.4.1.7: main_data_begin, the private bits, scfsi (MPEG-1, all 0), then per granule
    // and channel its part2_3_length, big_values, global_gain, scalefac_compress, a long block's table_select and region
    // counts, preflag (MPEG-1), scalefac_scale and count1table_select.
    private void SideInformation(Span<byte> target, int mainDataBegin, ReadOnlySpan<Mp3Granule> sides)
    {
        var f = format;
        var w = new Mp3BitWriter(target);
        int nch = f.Channels;
        if (f.Mpeg1)
        {
            w.Write((uint)mainDataBegin, 9);
            w.Write(0, nch == 1 ? 5 : 3);
            w.Write(0, 4 * nch);
        }
        else
        {
            w.Write((uint)mainDataBegin, 8);
            w.Write(0, nch == 1 ? 1 : 2);
        }

        foreach (var g in sides)
        {
            w.Write((uint)g.Part23Length, 12);
            w.Write((uint)g.BigValues, 9);
            w.Write((uint)g.GlobalGain, 8);
            w.Write((uint)g.ScalefacCompress, f.Mpeg1 ? 4 : 9);
            w.Write(0, 1);                                               // window_switching_flag: a long block
            w.Write((uint)g.Table0, 5);
            w.Write((uint)g.Table1, 5);
            w.Write((uint)g.Table2, 5);
            w.Write((uint)g.Region0Count, 4);
            w.Write((uint)g.Region1Count, 3);
            if (f.Mpeg1)
            {
                w.Write(0, 1);                                           // preflag
            }

            w.Write((uint)g.ScalefacScale, 1);
            w.Write((uint)g.Count1Table, 1);
        }

        w.Flush();
        if (w.Bits != 8 * f.SideBytes)
        {
            throw new InvalidOperationException($"the side information took {w.Bits} bits, not {8 * f.SideBytes}");
        }
    }

    // The slots kept from _base cover up to `end`.
    private void Reserve(long end)
    {
        long need = end - _base;
        if (need <= _slots.Length)
        {
            return;
        }

        var larger = Zeroed((int)Math.Max(need, 2L * _slots.Length));
        _slots.AsSpan(0, (int)(_next - _base)).CopyTo(larger);
        ArrayPool<byte>.Shared.Return(_slots);
        _slots = larger;
    }

    // Writes the frames whose slots end at or before `final` (no data after it reaches them), then drops their bytes.
    private void Emit(long final)
    {
        while (_pending.Count > 0 && _pending.Peek().Start + _pending.Peek().Bytes <= final)
        {
            var (head, start, bytes) = _pending.Dequeue();
            Stage(head);
            Stage(_slots.AsSpan((int)(start - _base), bytes));
            _heads.Push(head);
        }

        long keepFrom = _pending.Count > 0 ? _pending.Peek().Start : _next;
        int drop = (int)(keepFrom - _base), kept = (int)(_next - keepFrom);
        if (drop > 0)
        {
            _slots.AsSpan(drop, kept).CopyTo(_slots);
            _slots.AsSpan(kept, drop).Clear();
            _base = keepFrom;
        }
    }

    private void Stage(ReadOnlySpan<byte> bytes)
    {
        if (_stagedBytes + bytes.Length > _staged.Length)
        {
            output.Write(_staged, 0, _stagedBytes);
            _stagedBytes = 0;
        }

        if (bytes.Length > _staged.Length)
        {
            output.Write(bytes);
            return;
        }

        bytes.CopyTo(_staged.AsSpan(_stagedBytes));
        _stagedBytes += bytes.Length;
    }
}
