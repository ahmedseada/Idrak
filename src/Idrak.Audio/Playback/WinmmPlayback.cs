// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// The "winmm" playback (<see cref="AudioPlaybacks.Winmm"/>): Windows' waveOut through winmm.dll, loaded when first used.
/// 16-bit PCM at the rate and channels asked (Windows' sound mapper converts to the device's own;
/// <see cref="AudioPlaybackOptions.FallbackSampleRate"/> and <see cref="AudioPlaybackOptions.FallbackChannels"/> when not
/// set), on the system's default output or one by number (<see cref="Devices"/>) or by a part of its name. Played on a
/// thread of its own through buffers the system plays in turn, each filled again as soon as it is done.
/// </summary>
public sealed unsafe class WinmmPlayback : IAudioPlayback
{
    /// <inheritdoc />
    public string Name => AudioPlaybacks.Winmm;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => WinmmLibrary.Available("waveOut", out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioOutputDevice> Devices()
    {
        if (!IsAvailable(out _))
        {
            return [];
        }

        var api = WinmmLibrary.Api!;
        var devices = new List<AudioOutputDevice> { new("default", "the system's default output", true) };
        uint count = api.OutGetNumDevs();
        for (uint i = 0; i < count; i++)
        {
            WinmmLibrary.WaveOutCaps caps;
            if (api.OutGetDevCaps(i, &caps, (uint)sizeof(WinmmLibrary.WaveOutCaps)) == 0)
            {
                devices.Add(new AudioOutputDevice(i.ToString(CultureInfo.InvariantCulture), new string(caps.Name, 0, WinmmLibrary.Length(caps.Name, 32)), false));
            }
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
            throw new NotSupportedException($"The winmm playback cannot play here: {reason}.");
        }

        var (id, name) = Choose(options.Device);
        return new WinmmOutput(WinmmLibrary.Api!, id, name, options).Queue;
    }

    // The device: the default for null or "default", a number, or the first whose name holds the text (ignoring case).
    private (uint Id, string Name) Choose(string? device)
    {
        if (device is null or "" || device.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return (WinmmLibrary.WaveMapper, "the default output");
        }

        var devices = Devices().Skip(1).ToList();
        if (uint.TryParse(device, NumberStyles.None, CultureInfo.InvariantCulture, out uint number))
        {
            return devices.FirstOrDefault(d => d.Id == device) is { } found ? (number, found.Name)
                : throw new ArgumentException($"There is no output device {number} (the devices: {List(devices)}).", nameof(device));
        }

        return devices.FirstOrDefault(d => d.Name.Contains(device, StringComparison.OrdinalIgnoreCase)) is { } named
            ? (uint.Parse(named.Id, CultureInfo.InvariantCulture), named.Name)
            : throw new ArgumentException($"No output device's name holds '{device}' (the devices: {List(devices)}).", nameof(device));
    }

    private static string List(List<AudioOutputDevice> devices) => devices.Count == 0 ? "none" : string.Join(", ", devices.Select(d => $"{d.Id}: {d.Name}"));

    // One playback: buffers enough for 0.2 s (at least four), each filled from the queue and written when the system is
    // done with it; a drain waits until none is queued with the system.
    private sealed class WinmmOutput
    {
        private readonly WinmmLibrary _api;
        private readonly nint _handle;
        private readonly EventWaitHandle _signal = new(false, EventResetMode.AutoReset);
        private readonly WinmmLibrary.WaveHeader* _headers;
        private readonly int _count;
        private readonly int _bytes;

        public WinmmOutput(WinmmLibrary api, uint device, string name, AudioPlaybackOptions options)
        {
            _api = api;
            int rate = options.SampleRate ?? AudioPlaybackOptions.FallbackSampleRate, channels = options.Channels ?? AudioPlaybackOptions.FallbackChannels;
            var format = WinmmLibrary.Pcm16(channels, rate);
            nint handle;
            try
            {
                Check(api.OutOpen(&handle, device, &format, _signal.SafeWaitHandle.DangerousGetHandle(), 0, WinmmLibrary.CallbackEvent),
                    $"open {name} for 16-bit {channels} channel(s) at {rate} Hz");
            }
            catch
            {
                _signal.Dispose();
                throw;
            }

            _handle = handle;
            int frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * rate));
            _bytes = frames * 2 * channels;
            _count = Math.Max(4, (int)Math.Ceiling(0.2 / options.BlockSeconds));
            _headers = (WinmmLibrary.WaveHeader*)NativeMemory.AllocZeroed((nuint)(_count * sizeof(WinmmLibrary.WaveHeader)));
            for (int i = 0; i < _count; i++)
            {
                _headers[i].Data = (nint)NativeMemory.AllocZeroed((nuint)_bytes);
                _headers[i].BufferLength = (uint)_bytes;
            }

            Queue = new PlaybackQueue($"speaker ({name}, winmm)", rate, channels, PcmSampleFormat.S16LE, options.BufferSeconds, options.Resample, Play);
            Queue.Start();
        }

        // What the writer writes; its thread runs Play.
        public PlaybackQueue Queue { get; }

        private void Play(PlaybackQueue queue)
        {
            uint size = (uint)sizeof(WinmmLibrary.WaveHeader);
            var prepared = new bool[_count];
            try
            {
                int next = 0;
                while (!queue.Stopping)
                {
                    var header = &_headers[next];
                    if ((Volatile.Read(ref header->Flags) & WinmmLibrary.InQueue) != 0)
                    {
                        _signal.WaitOne(100);                                // the system plays it still: wait for a buffer done
                        continue;
                    }

                    int got = queue.Take(new Span<byte>((void*)header->Data, _bytes), 100);
                    if (got == 0)
                    {
                        if (queue.DrainPending && !queue.Stopping)
                        {
                            while (!queue.Stopping && Playing())
                            {
                                _signal.WaitOne(100);
                            }

                            queue.Drained();
                        }

                        continue;
                    }

                    if (!prepared[next] || header->BufferLength != (uint)got)
                    {
                        if (prepared[next])
                        {
                            Check(_api.OutUnprepare(_handle, header, size), "release a buffer");
                        }

                        header->BufferLength = (uint)got;                    // a buffer is prepared for its length
                        header->Flags = 0;
                        Check(_api.OutPrepare(_handle, header, size), "prepare a buffer");
                        prepared[next] = true;
                    }

                    Check(_api.OutWrite(_handle, header, size), "queue a buffer");
                    next = (next + 1) % _count;
                }
            }
            finally
            {
                _api.OutReset(_handle);                                      // stops at once; every queued buffer comes back done
                for (int i = 0; i < _count; i++)
                {
                    if (prepared[i])
                    {
                        _api.OutUnprepare(_handle, &_headers[i], size);
                    }
                }

                _api.OutClose(_handle);
                for (int i = 0; i < _count; i++)
                {
                    NativeMemory.Free((void*)_headers[i].Data);
                }

                NativeMemory.Free(_headers);
                _signal.Dispose();
            }
        }

        // Whether any buffer is still queued with the system.
        private bool Playing()
        {
            for (int i = 0; i < _count; i++)
            {
                if ((Volatile.Read(ref _headers[i].Flags) & WinmmLibrary.InQueue) != 0)
                {
                    return true;
                }
            }

            return false;
        }

        private void Check(uint result, string what)
        {
            if (result != 0)
            {
                throw new InvalidOperationException($"winmm could not {what}: {_api.OutErrorText(result)} (MMRESULT {result}).");
            }
        }
    }
}
