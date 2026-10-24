// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace Idrak.Audio;

/// <summary>How raw PCM samples are stored (<see cref="PcmStreamInput"/>).</summary>
public enum PcmSampleFormat
{
    /// <summary>16-bit signed integers, little-endian (ffmpeg's s16le, arecord's S16_LE).</summary>
    S16LE,

    /// <summary>32-bit floats, little-endian (ffmpeg's f32le).</summary>
    F32LE,
}

/// <summary>
/// Live sound from a stream as it arrives (standard input from a recorder, a socket): a WAV header when the stream starts
/// with one (its rate, channels and format taken from it, its size ignored, as a pipe's is unknown), else raw PCM in the
/// format, rate and channels given. Blocks of <see cref="AudioCaptureOptions.BlockSeconds"/> are given as soon as they
/// are filled; the last, shorter one at the stream's end.
/// </summary>
public sealed class PcmStreamInput : IAudioInput
{
    private readonly Stream _stream;
    private readonly bool _leaveOpen;
    private readonly double _blockSeconds;
    private int _bits;
    private bool _float;
    private byte[] _start = [];
    private bool _read;

    /// <summary>A stream of raw PCM in <paramref name="format"/> (or a WAV stream, told from its first bytes).</summary>
    /// <exception cref="ArgumentException">The rate, the channels or the block is not positive.</exception>
    public PcmStreamInput(Stream stream, string description, PcmSampleFormat format = PcmSampleFormat.S16LE, int sampleRate = 16000, int channels = 1,
        double blockSeconds = 0.05, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        new AudioCaptureOptions { SampleRate = sampleRate, Channels = channels, BlockSeconds = blockSeconds }.Check();
        _stream = stream;
        _leaveOpen = leaveOpen;
        _blockSeconds = blockSeconds;
        (_bits, _float) = format == PcmSampleFormat.F32LE ? (32, true) : (16, false);
        Description = description;
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <inheritdoc />
    public string Description { get; }

    /// <inheritdoc />
    /// <remarks>The one given, or a WAV header's once the first block was read.</remarks>
    public int SampleRate { get; private set; }

    /// <inheritdoc />
    /// <remarks>The ones given, or a WAV header's once the first block was read.</remarks>
    public int Channels { get; private set; }

    /// <summary>The samples' format: the one given, or a WAV header's (16-bit, 24-bit and 32-bit integers, 32-bit floats).</summary>
    public string Format => _float ? "f32le" : $"s{_bits}le";

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">Read a second time.</exception>
    /// <exception cref="InvalidDataException">A WAV header this reader does not take (a compressed format, 8-bit samples).</exception>
    public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_read)
        {
            throw new InvalidOperationException("A stream input is read once.");
        }

        _read = true;
        await ReadHeaderAsync(cancellationToken).ConfigureAwait(false);
        int frameBytes = Channels * _bits / 8;
        int frames = Math.Max(1, (int)Math.Round(_blockSeconds * SampleRate));
        var block = new byte[frames * frameBytes];
        int filled = _start.Length;
        _start.CopyTo(block, 0);                                               // what the header check read of the sound
        while (true)
        {
            bool ended = false;
            while (filled < block.Length)
            {
                int n;
                try
                {
                    n = await _stream.ReadAsync(block.AsMemory(filled), cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    n = 0;                                                     // stopped: what was read is given, then the end
                }

                if (n == 0)
                {
                    ended = true;
                    break;
                }

                filled += n;
            }

            int whole = filled / frameBytes * frameBytes;
            if (whole > 0)
            {
                yield return Block(block.AsSpan(0, whole));
                block.AsSpan(whole, filled - whole).CopyTo(block);           // a frame split between reads waits for its rest
                filled -= whole;
            }

            if (ended)
            {
                yield break;
            }
        }
    }

    // One block of whole frames as AudioData: 16-bit kept as its bytes (floats made when read), others planar floats.
    private AudioData Block(ReadOnlySpan<byte> bytes)
    {
        int frames = bytes.Length / (Channels * _bits / 8);
        if (!_float && _bits == 16)
        {
            return new AudioData(bytes.ToArray(), Channels, SampleRate);
        }

        var planar = GC.AllocateUninitializedArray<float>(frames * Channels);
        int size = _bits / 8;
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < Channels; c++)
            {
                var sample = bytes.Slice((f * Channels + c) * size, size);
                planar[c * frames + f] = _float ? BinaryPrimitives.ReadSingleLittleEndian(sample)
                    : _bits == 24 ? ((sample[0] << 8 | sample[1] << 16 | sample[2] << 24) >> 8) / 8388608f
                    : BinaryPrimitives.ReadInt32LittleEndian(sample) / 2147483648f;
            }
        }

        return new AudioData(planar, Channels, SampleRate);
    }

    // A WAV header when the stream starts with "RIFF....WAVE": its format chunk read, the chunks before "data" skipped. Else
    // the bytes read stay the sound's first.
    private async Task ReadHeaderAsync(CancellationToken cancellationToken)
    {
        var head = new byte[12];
        int got = await FillAsync(head, cancellationToken).ConfigureAwait(false);
        if (got < 12 || !head.AsSpan(0, 4).SequenceEqual("RIFF"u8) || !head.AsSpan(8, 4).SequenceEqual("WAVE"u8))
        {
            _start = head[..got];
            return;
        }

        var chunk = new byte[8];
        bool format = false;
        while (await FillAsync(chunk, cancellationToken).ConfigureAwait(false) == 8)
        {
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(chunk.AsSpan(4));
            if (chunk.AsSpan(0, 4).SequenceEqual("data"u8))
            {
                if (!format)
                {
                    throw new InvalidDataException($"{Description}: a WAV stream's data before its format chunk.");
                }

                return;                                                        // the size is ignored: a pipe's is unknown
            }

            if (size > 1 << 20)
            {
                throw new InvalidDataException($"{Description}: a WAV chunk of {size} bytes before the data.");
            }

            var body = new byte[size + (size & 1)];
            if (await FillAsync(body, cancellationToken).ConfigureAwait(false) < body.Length)
            {
                break;
            }

            if (chunk.AsSpan(0, 4).SequenceEqual("fmt "u8))
            {
                ReadFormat(body);
                format = true;
            }
        }

        throw new InvalidDataException($"{Description}: a WAV stream that ends before its data chunk.");
    }

    private void ReadFormat(ReadOnlySpan<byte> body)
    {
        if (body.Length < 16)
        {
            throw new InvalidDataException($"{Description}: a WAV format chunk of {body.Length} bytes.");
        }

        int tag = BinaryPrimitives.ReadUInt16LittleEndian(body);
        if (tag == 0xFFFE && body.Length >= 26)
        {
            tag = BinaryPrimitives.ReadUInt16LittleEndian(body[24..]);        // WAVE_FORMAT_EXTENSIBLE: the sub-format's tag
        }

        int channels = BinaryPrimitives.ReadUInt16LittleEndian(body[2..]);
        int rate = (int)BinaryPrimitives.ReadUInt32LittleEndian(body[4..]);
        int bits = BinaryPrimitives.ReadUInt16LittleEndian(body[14..]);
        bool isFloat = tag == 3;
        if (!(tag == 1 && bits is 16 or 24 or 32 || isFloat && bits == 32) || channels < 1 || rate < 1)
        {
            throw new InvalidDataException($"{Description}: a WAV stream of format {tag}, {bits}-bit, {channels} channel(s) at {rate} Hz; live input takes 16, 24 or 32-bit integer or 32-bit float PCM.");
        }

        (Channels, SampleRate, _bits, _float) = (channels, rate, bits, isFloat);
    }

    // Reads until `buffer` is full or the stream ends; the bytes read.
    private async Task<int> FillAsync(byte[] buffer, CancellationToken cancellationToken)
    {
        int filled = 0;
        while (filled < buffer.Length)
        {
            int n = await _stream.ReadAsync(buffer.AsMemory(filled), cancellationToken).ConfigureAwait(false);
            if (n == 0)
            {
                break;
            }

            filled += n;
        }

        return filled;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_leaveOpen)
        {
            _stream.Dispose();
        }
    }
}
