// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// The "alsa" capture (<see cref="AudioCaptures.Alsa"/>): Linux's ALSA through libasound.so.2, loaded when first used.
/// 16-bit PCM at the rate and channels asked (ALSA's plug layer converts), from <c>default</c> or any ALSA name
/// (<c>hw:1,0</c>, <c>pulse</c>, <c>pipewire</c>; <see cref="Devices"/> lists its capture names). Recorded on a thread of
/// its own, a block at a time, waiting at most 100 ms so a stop is seen at once.
/// </summary>
public sealed unsafe class AlsaCapture : IAudioCapture
{
    /// <inheritdoc />
    public string Name => AudioCaptures.Alsa;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => AlsaLibrary.Available(out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioInputDevice> Devices()
    {
        if (!IsAvailable(out _))
        {
            return [];
        }

        var devices = new List<AudioInputDevice> { new("default", "the system's default input", true) };
        foreach (var (id, name) in AlsaLibrary.Api!.Devices(capture: true))
        {
            devices.Add(new AudioInputDevice(id, name, false));
        }

        return devices;
    }

    /// <inheritdoc />
    public IAudioInput Open(AudioCaptureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (!IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The alsa capture cannot record here: {reason}.");
        }

        return new AlsaInput(AlsaLibrary.Api!, string.IsNullOrEmpty(options.Device) ? "default" : options.Device, options).Queue;
    }

    // One recording: snd_pcm_wait then snd_pcm_readi of a block, an overrun recovered (the sound lost is not made up).
    private sealed class AlsaInput
    {
        private readonly AlsaLibrary _api;
        private readonly nint _pcm;
        private readonly int _frames;
        private readonly int _channels;

        public AlsaInput(AlsaLibrary api, string device, AudioCaptureOptions options)
        {
            _api = api;
            _channels = options.Channels;
            _pcm = api.OpenPcm(device, AlsaLibrary.StreamCapture);
            _frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * options.SampleRate));
            try
            {
                uint latency = (uint)Math.Clamp(4 * options.BlockSeconds * 1e6, 20_000, 500_000);   // the device's buffer: four blocks
                api.Check(api.SetParams(_pcm, AlsaLibrary.FormatS16LE, AlsaLibrary.AccessInterleaved, (uint)options.Channels, (uint)options.SampleRate, 1, latency),
                    $"record 16-bit {options.Channels} channel(s) at {options.SampleRate} Hz from '{device}'");
            }
            catch
            {
                api.Close(_pcm);
                throw;
            }

            Queue = new CaptureQueue($"microphone ({device}, alsa)", options.SampleRate, options.Channels, Record);
            Queue.Start();
        }

        // What the reader reads; its thread runs Record.
        public CaptureQueue Queue { get; }

        private void Record(CaptureQueue queue)
        {
            var buffer = new byte[_frames * 2 * _channels];
            try
            {
                fixed (byte* data = buffer)
                {
                    while (!queue.Stopping)
                    {
                        int ready = _api.Wait(_pcm, 100);
                        if (ready == 0)
                        {
                            continue;                                          // nothing yet: look at the stop flag again
                        }

                        nint got = ready < 0 ? ready : _api.ReadInterleaved(_pcm, data, (nuint)_frames);
                        if (got < 0)
                        {
                            _api.Check(_api.Recover(_pcm, (int)got, 1), "recover from a read error");   // an overrun: the stream restarts
                            continue;
                        }

                        queue.Deliver(buffer.AsSpan(0, (int)got * 2 * _channels));
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
