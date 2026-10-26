// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Audio;

/// <summary>
/// The "aaudio" playback (<see cref="AudioPlaybacks.Aaudio"/>): Android's AAudio through libaaudio.so (Android 8.0, API
/// 26, and later), loaded when first used. 16-bit PCM asked at the rate and channels given, or at the device's own when
/// they are not set (no resampling twice); the output carries what the stream actually gave, and a stream opened in
/// float is written float. On the default speaker or a device id (the NDK lists no devices: an id comes from Android's
/// AudioManager.getDevices). Played on a thread of its own by AAudio's blocking write, with a 100 ms timeout so a stop is
/// seen at once. A drain waits until the device has read every frame written, then stops the stream (AAudio plays out
/// what it holds before it stops); the next write starts it again.
/// </summary>
public sealed unsafe class AAudioPlayback : IAudioPlayback
{
    /// <inheritdoc />
    public string Name => AudioPlaybacks.Aaudio;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => AAudioLibrary.Available(out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioOutputDevice> Devices() =>
        IsAvailable(out _) ? [new AudioOutputDevice("default", "the system's default output (a device id from AudioManager.getDevices chooses another)", true)] : [];

    /// <inheritdoc />
    public IAudioOutput Open(AudioPlaybackOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (!IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The aaudio playback cannot play here: {reason}.");
        }

        return new AAudioOutput(AAudioLibrary.Api!, AAudioLibrary.DeviceId(options.Device), options).Queue;
    }

    private sealed class AAudioOutput
    {
        private readonly AAudioLibrary _api;
        private readonly nint _stream;
        private readonly int _frames;
        private readonly int _rate;

        public AAudioOutput(AAudioLibrary api, int deviceId, AudioPlaybackOptions options)
        {
            _api = api;
            int capacity = options.SampleRate is { } asked ? 4 * Math.Max(1, (int)Math.Round(options.BlockSeconds * asked)) : AAudioLibrary.Unspecified;
            _stream = api.OpenStreamWith(AAudioLibrary.DirectionOutput, deviceId, options.SampleRate ?? AAudioLibrary.Unspecified, options.Channels ?? AAudioLibrary.Unspecified,
                options.Performance, options.Exclusive, capacity);              // the device's buffer: four blocks, as ALSA's
            int format = api.CheckedFormat(_stream, input: false);
            int rate = api.GetSampleRate(_stream), channels = api.GetChannelCount(_stream);
            if (rate < 1 || channels < 1)
            {
                api.Close(_stream);
                throw new InvalidOperationException($"AAudio opened the speaker with {rate} Hz and {channels} channel(s).");
            }

            _rate = rate;
            _frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * rate));
            int id = api.GetDeviceId(_stream);
            Queue = new PlaybackQueue($"speaker (device {id}, aaudio)", rate, channels, format == AAudioLibrary.FormatFloat ? PcmSampleFormat.F32LE : PcmSampleFormat.S16LE,
                options.BufferSeconds, options.Resample, Play);
            Queue.Start();
        }

        // What the writer writes; its thread runs Play.
        public PlaybackQueue Queue { get; }

        private void Play(PlaybackQueue queue)
        {
            int frameBytes = queue.FrameBytes;
            var buffer = new byte[_frames * frameBytes];
            bool started = false;
            try
            {
                fixed (byte* data = buffer)
                {
                    while (!queue.Stopping)
                    {
                        int got = queue.Take(buffer, 100) / frameBytes;
                        if (got == 0)
                        {
                            if (queue.DrainPending && !queue.Stopping)
                            {
                                if (started)
                                {
                                    PlayOut(queue);
                                    started = false;
                                }

                                queue.Drained();
                            }

                            continue;
                        }

                        if (!started)
                        {
                            AAudioLibrary.Check(_api.RequestStart(_stream), "start the speaker", input: false);
                            started = true;
                        }

                        for (int done = 0; done < got && !queue.Stopping;)
                        {
                            int put = _api.Write(_stream, data + done * frameBytes, got - done, AAudioLibrary.Timeout);   // fewer frames (or none) on the timeout
                            AAudioLibrary.Check(put, "write to the speaker", input: false);
                            done += put;
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

        // Waits until the device has read every frame written (with room for the buffer's length and a second more), then
        // stops the stream, which plays out what the device itself holds, and waits until it has stopped.
        private void PlayOut(PlaybackQueue queue)
        {
            long written = _api.GetFramesWritten(_stream);
            long left = written - _api.GetFramesRead(_stream);
            var limit = TimeSpan.FromSeconds(Math.Max(0, left) / (double)_rate + 1);
            int pause = Math.Max(1, (int)(1000L * Math.Max(1, _api.GetFramesPerBurst(_stream)) / _rate));
            long start = Stopwatch.GetTimestamp();
            while (!queue.Stopping && _api.GetFramesRead(_stream) < written && Stopwatch.GetElapsedTime(start) < limit)
            {
                Thread.Sleep(pause);
            }

            AAudioLibrary.Check(_api.RequestStop(_stream), "stop the speaker after its last sound", input: false);
            int state = AAudioLibrary.StateStopping;
            _api.WaitForStateChange(_stream, AAudioLibrary.StateStopping, &state, 1_000_000_000L);
        }
    }
}
