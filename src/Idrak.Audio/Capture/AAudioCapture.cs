// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// The "aaudio" capture (<see cref="AudioCaptures.Aaudio"/>): Android's AAudio through libaaudio.so (Android 8.0, API 26,
/// and later), loaded when first used. 16-bit PCM asked at the rate and channels given (AAudio converts; the blocks carry
/// what the stream actually gave, and a stream opened in float is read as float), from the default microphone or a device
/// id (the NDK lists no devices: an id comes from Android's AudioManager.getDevices). Recorded on a thread of its own by
/// AAudio's blocking read, a block at a time with a 100 ms timeout so a stop is seen at once. The app needs the
/// RECORD_AUDIO permission; a refusal is named in the error.
/// </summary>
public sealed unsafe class AAudioCapture : IAudioCapture
{
    /// <inheritdoc />
    public string Name => AudioCaptures.Aaudio;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => AAudioLibrary.Available(out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioInputDevice> Devices() =>
        IsAvailable(out _) ? [new AudioInputDevice("default", "the system's default input (a device id from AudioManager.getDevices chooses another)", true)] : [];

    /// <inheritdoc />
    public IAudioInput Open(AudioCaptureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (!IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The aaudio capture cannot record here: {reason}.");
        }

        return new AAudioInput(AAudioLibrary.Api!, AAudioLibrary.DeviceId(options.Device), options).Queue;
    }

    // One recording: AAudioStream_read of a block with a timeout, delivered as 16-bit or float blocks.
    private sealed class AAudioInput
    {
        private readonly AAudioLibrary _api;
        private readonly nint _stream;
        private readonly int _frames;
        private readonly int _channels;
        private readonly bool _float;

        public AAudioInput(AAudioLibrary api, int deviceId, AudioCaptureOptions options)
        {
            _api = api;
            int asked = Math.Max(1, (int)Math.Round(options.BlockSeconds * options.SampleRate));
            _stream = api.OpenStreamWith(AAudioLibrary.DirectionInput, deviceId, options.SampleRate, options.Channels, options.Performance, options.Exclusive,
                capacityFrames: 4 * asked);                                   // the device's buffer: four blocks, as ALSA's
            int format = api.CheckedFormat(_stream, input: true);
            _float = format == AAudioLibrary.FormatFloat;
            int rate = api.GetSampleRate(_stream), channels = api.GetChannelCount(_stream);
            if (rate < 1 || channels < 1)
            {
                api.Close(_stream);
                throw new InvalidOperationException($"AAudio opened the microphone with {rate} Hz and {channels} channel(s).");
            }

            _channels = channels;
            _frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * rate));
            int id = api.GetDeviceId(_stream);
            Queue = new CaptureQueue($"microphone (device {id}, aaudio)", rate, channels, Record);
            Queue.Start();
        }

        // What the reader reads; its thread runs Record.
        public CaptureQueue Queue { get; }

        private void Record(CaptureQueue queue)
        {
            int sampleBytes = _float ? sizeof(float) : sizeof(short);
            var buffer = new byte[_frames * _channels * sampleBytes];
            bool started = false;
            try
            {
                AAudioLibrary.Check(_api.RequestStart(_stream), "start the microphone", input: true);
                started = true;
                fixed (byte* data = buffer)
                {
                    while (!queue.Stopping)
                    {
                        int got = _api.Read(_stream, data, _frames, AAudioLibrary.Timeout);   // fewer frames (or none) on the timeout
                        AAudioLibrary.Check(got, "read the microphone", input: true);
                        if (got == 0)
                        {
                            continue;
                        }

                        if (_float)
                        {
                            queue.DeliverFloat(MemoryMarshal.Cast<byte, float>(buffer.AsSpan(0, got * _channels * sizeof(float))));
                        }
                        else
                        {
                            queue.Deliver(buffer.AsSpan(0, got * _channels * sizeof(short)));
                        }
                    }
                }
            }
            finally
            {
                if (started)
                {
                    _api.RequestStop(_stream);
                }

                _api.Close(_stream);
            }
        }
    }
}
