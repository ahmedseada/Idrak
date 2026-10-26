// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// The "alsa" playback (<see cref="AudioPlaybacks.Alsa"/>): Linux's ALSA through libasound.so.2, loaded when first used.
/// 16-bit PCM at the rate and channels asked (ALSA's plug layer converts; <see cref="AudioPlaybackOptions.FallbackSampleRate"/>
/// and <see cref="AudioPlaybackOptions.FallbackChannels"/> when not set), on <c>default</c> or any ALSA name
/// (<c>hw:0,0</c>, <c>pulse</c>, <c>pipewire</c>; <see cref="Devices"/> lists its playback names). Played on a thread of
/// its own, a block at a time, waiting at most 100 ms for room so a stop is seen at once; an underrun is recovered (the
/// device plays silence until the next block).
/// </summary>
public sealed unsafe class AlsaPlayback : IAudioPlayback
{
    /// <inheritdoc />
    public string Name => AudioPlaybacks.Alsa;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => AlsaLibrary.Available(out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioOutputDevice> Devices()
    {
        if (!IsAvailable(out _))
        {
            return [];
        }

        var devices = new List<AudioOutputDevice> { new("default", "the system's default output", true) };
        foreach (var (id, name) in AlsaLibrary.Api!.Devices(capture: false))
        {
            devices.Add(new AudioOutputDevice(id, name, false));
        }

        return devices;
    }

    /// <inheritdoc />
    public IAudioOutput Open(AudioPlaybackOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (!IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The alsa playback cannot play here: {reason}.");
        }

        return new AlsaOutput(AlsaLibrary.Api!, string.IsNullOrEmpty(options.Device) ? "default" : options.Device, options).Queue;
    }

    // One playback: a block taken from the queue, snd_pcm_wait for room, then snd_pcm_writei until all of it is written;
    // a drain is snd_pcm_drain (which starts a stream still waiting for its start threshold), then snd_pcm_prepare.
    private sealed class AlsaOutput
    {
        private readonly AlsaLibrary _api;
        private readonly nint _pcm;
        private readonly int _frames;

        public AlsaOutput(AlsaLibrary api, string device, AudioPlaybackOptions options)
        {
            _api = api;
            int rate = options.SampleRate ?? AudioPlaybackOptions.FallbackSampleRate, channels = options.Channels ?? AudioPlaybackOptions.FallbackChannels;
            _pcm = api.OpenPcm(device, AlsaLibrary.StreamPlayback);
            _frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * rate));
            try
            {
                uint latency = (uint)Math.Clamp(4 * options.BlockSeconds * 1e6, 20_000, 500_000);   // the device's buffer: four blocks
                api.Check(api.SetParams(_pcm, AlsaLibrary.FormatS16LE, AlsaLibrary.AccessInterleaved, (uint)channels, (uint)rate, 1, latency),
                    $"play 16-bit {channels} channel(s) at {rate} Hz on '{device}'");
            }
            catch
            {
                api.Close(_pcm);
                throw;
            }

            Queue = new PlaybackQueue($"speaker ({device}, alsa)", rate, channels, PcmSampleFormat.S16LE, options.BufferSeconds, options.Resample, Play);
            Queue.Start();
        }

        // What the writer writes; its thread runs Play.
        public PlaybackQueue Queue { get; }

        private void Play(PlaybackQueue queue)
        {
            int frameBytes = queue.FrameBytes;
            var buffer = new byte[_frames * frameBytes];
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
                                int drained = _api.Drain(_pcm);              // blocks until the device has played all it holds
                                _api.Check(drained == 0 ? _api.Prepare(_pcm) : _api.Recover(_pcm, drained, 1), "play out what it holds");
                                queue.Drained();
                            }

                            continue;
                        }

                        for (int done = 0; done < got && !queue.Stopping;)
                        {
                            int ready = _api.Wait(_pcm, 100);
                            if (ready == 0)
                            {
                                continue;                                      // no room yet: look at the stop flag again
                            }

                            nint put = ready < 0 ? ready : _api.WriteInterleaved(_pcm, data + done * frameBytes, (nuint)(got - done));
                            if (put < 0)
                            {
                                _api.Check(_api.Recover(_pcm, (int)put, 1), "recover from an underrun");   // the stream restarts
                                continue;
                            }

                            done += (int)put;
                        }
                    }
                }
            }
            finally
            {
                _api.Drop(_pcm);
                _api.Close(_pcm);
            }
        }
    }
}
