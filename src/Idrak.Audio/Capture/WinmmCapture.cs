// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// The "winmm" capture (<see cref="AudioCaptures.Winmm"/>): Windows' waveIn through winmm.dll, loaded when first used.
/// 16-bit PCM at the rate and channels asked (Windows' sound mapper converts from the device's own), the system's default
/// input or one by number (<see cref="Devices"/>) or by a part of its name. Recorded on a thread of its own into buffers
/// the system fills in turn, each handed on as soon as it is full.
/// </summary>
public sealed unsafe class WinmmCapture : IAudioCapture
{
    /// <inheritdoc />
    public string Name => AudioCaptures.Winmm;

    /// <inheritdoc />
    public bool IsAvailable(out string reason) => WinmmLibrary.Available("waveIn", out reason);

    /// <inheritdoc />
    public IReadOnlyList<AudioInputDevice> Devices()
    {
        if (!IsAvailable(out _))
        {
            return [];
        }

        var api = WinmmLibrary.Api!;
        var devices = new List<AudioInputDevice> { new("default", "the system's default input", true) };
        uint count = api.InGetNumDevs();
        for (uint i = 0; i < count; i++)
        {
            WinmmLibrary.WaveInCaps caps;
            if (api.InGetDevCaps(i, &caps, (uint)sizeof(WinmmLibrary.WaveInCaps)) == 0)
            {
                devices.Add(new AudioInputDevice(i.ToString(System.Globalization.CultureInfo.InvariantCulture), new string(caps.Name, 0, WinmmLibrary.Length(caps.Name, 32)), false));
            }
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
            throw new NotSupportedException($"The winmm capture cannot record here: {reason}.");
        }

        var (id, name) = Choose(options.Device);
        return new WinmmInput(WinmmLibrary.Api!, id, name, options).Queue;
    }

    // The device: the default for null or "default", a number, or the first whose name holds the text (ignoring case).
    private (uint Id, string Name) Choose(string? device)
    {
        if (device is null or "" || device.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return (WinmmLibrary.WaveMapper, "the default input");
        }

        var devices = Devices().Skip(1).ToList();
        if (uint.TryParse(device, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out uint number))
        {
            return devices.FirstOrDefault(d => d.Id == device) is { } found ? (number, found.Name)
                : throw new ArgumentException($"There is no input device {number} (the devices: {List(devices)}).", nameof(device));
        }

        return devices.FirstOrDefault(d => d.Name.Contains(device, StringComparison.OrdinalIgnoreCase)) is { } named
            ? (uint.Parse(named.Id, System.Globalization.CultureInfo.InvariantCulture), named.Name)
            : throw new ArgumentException($"No input device's name holds '{device}' (the devices: {List(devices)}).", nameof(device));
    }

    private static string List(List<AudioInputDevice> devices) => devices.Count == 0 ? "none" : string.Join(", ", devices.Select(d => $"{d.Id}: {d.Name}"));

    // One recording: buffers enough for half a second queued with the system, each handed on and queued again when full.
    private sealed class WinmmInput
    {
        private readonly WinmmLibrary _api;
        private readonly nint _handle;
        private readonly EventWaitHandle _signal = new(false, EventResetMode.AutoReset);
        private readonly WinmmLibrary.WaveHeader* _headers;
        private readonly int _count;
        private readonly int _bytes;

        public WinmmInput(WinmmLibrary api, uint device, string name, AudioCaptureOptions options)
        {
            _api = api;
            var format = WinmmLibrary.Pcm16(options.Channels, options.SampleRate);
            nint handle;
            try
            {
                Check(api.InOpen(&handle, device, &format, _signal.SafeWaitHandle.DangerousGetHandle(), 0, WinmmLibrary.CallbackEvent), $"open {name}");
            }
            catch
            {
                _signal.Dispose();
                throw;
            }

            _handle = handle;
            int frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * options.SampleRate));
            _bytes = frames * 2 * options.Channels;
            _count = Math.Max(4, (int)Math.Ceiling(0.5 / options.BlockSeconds));
            _headers = (WinmmLibrary.WaveHeader*)NativeMemory.AllocZeroed((nuint)(_count * sizeof(WinmmLibrary.WaveHeader)));
            for (int i = 0; i < _count; i++)
            {
                _headers[i].Data = (nint)NativeMemory.AllocZeroed((nuint)_bytes);
                _headers[i].BufferLength = (uint)_bytes;
            }

            Queue = new CaptureQueue($"microphone ({name}, winmm)", options.SampleRate, options.Channels, Record);
            Queue.Start();
        }

        // What the reader reads; its thread runs Record.
        public CaptureQueue Queue { get; }

        private void Record(CaptureQueue queue)
        {
            int prepared = 0;
            try
            {
                for (; prepared < _count; prepared++)
                {
                    Check(_api.InPrepare(_handle, &_headers[prepared], (uint)sizeof(WinmmLibrary.WaveHeader)), "prepare a buffer");
                    Check(_api.InAddBuffer(_handle, &_headers[prepared], (uint)sizeof(WinmmLibrary.WaveHeader)), "queue a buffer");
                }

                Check(_api.InStart(_handle), "start recording");
                int next = 0;
                while (!queue.Stopping)
                {
                    _signal.WaitOne(100);
                    while (!queue.Stopping && (_headers[next].Flags & WinmmLibrary.Done) != 0)
                    {
                        queue.Deliver(new ReadOnlySpan<byte>((void*)_headers[next].Data, (int)_headers[next].BytesRecorded));
                        _headers[next].BytesRecorded = 0;
                        Check(_api.InAddBuffer(_handle, &_headers[next], (uint)sizeof(WinmmLibrary.WaveHeader)), "queue a buffer again");
                        next = (next + 1) % _count;
                    }
                }

                _api.InReset(_handle);                                         // stops; every queued buffer comes back done
                for (int i = 0; i < _count; i++, next = (next + 1) % _count)
                {
                    if ((_headers[next].Flags & WinmmLibrary.Done) != 0 && _headers[next].BytesRecorded > 0)
                    {
                        queue.Deliver(new ReadOnlySpan<byte>((void*)_headers[next].Data, (int)_headers[next].BytesRecorded));   // what was heard last
                    }
                }
            }
            finally
            {
                _api.InReset(_handle);
                for (int i = 0; i < prepared; i++)
                {
                    _api.InUnprepare(_handle, &_headers[i], (uint)sizeof(WinmmLibrary.WaveHeader));
                }

                _api.InClose(_handle);
                for (int i = 0; i < _count; i++)
                {
                    NativeMemory.Free((void*)_headers[i].Data);
                }

                NativeMemory.Free(_headers);
                _signal.Dispose();
            }
        }

        private void Check(uint result, string what)
        {
            if (result != 0)
            {
                throw new InvalidOperationException($"winmm could not {what}: {_api.InErrorText(result)} (MMRESULT {result}).");
            }
        }
    }
}
