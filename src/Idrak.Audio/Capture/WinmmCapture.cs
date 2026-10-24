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
    private const uint WaveMapper = uint.MaxValue;

    /// <inheritdoc />
    public string Name => AudioCaptures.Winmm;

    /// <inheritdoc />
    public bool IsAvailable(out string reason)
    {
        if (!OperatingSystem.IsWindows())
        {
            reason = "waveIn is Windows'";
            return false;
        }

        if (Winmm.Api is null)
        {
            reason = $"winmm.dll could not be loaded ({Winmm.Failure})";
            return false;
        }

        reason = "";
        return true;
    }

    /// <inheritdoc />
    public IReadOnlyList<AudioInputDevice> Devices()
    {
        if (!IsAvailable(out _))
        {
            return [];
        }

        var api = Winmm.Api!;
        var devices = new List<AudioInputDevice> { new("default", "the system's default input", true) };
        uint count = api.GetNumDevs();
        for (uint i = 0; i < count; i++)
        {
            WaveInCaps caps;
            if (api.GetDevCaps(i, &caps, (uint)sizeof(WaveInCaps)) == 0)
            {
                devices.Add(new AudioInputDevice(i.ToString(System.Globalization.CultureInfo.InvariantCulture), new string(caps.Name, 0, Length(caps.Name, 32)), false));
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
        return new WinmmInput(Winmm.Api!, id, name, options).Queue;
    }

    // The device: the default for null or "default", a number, or the first whose name holds the text (ignoring case).
    private (uint Id, string Name) Choose(string? device)
    {
        if (device is null or "" || device.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return (WaveMapper, "the default input");
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

    private static int Length(char* text, int max)
    {
        int n = 0;
        while (n < max && text[n] != '\0')
        {
            n++;
        }

        return n;
    }

    // One recording: buffers enough for half a second queued with the system, each handed on and queued again when full.
    private sealed class WinmmInput
    {
        private const uint CallbackEvent = 0x00050000;
        private const uint Done = 0x00000001;

        private readonly Winmm _api;
        private readonly nint _handle;
        private readonly EventWaitHandle _signal = new(false, EventResetMode.AutoReset);
        private readonly WaveHeader* _headers;
        private readonly int _count;
        private readonly int _bytes;

        public WinmmInput(Winmm api, uint device, string name, AudioCaptureOptions options)
        {
            _api = api;
            var format = new WaveFormat
            {
                Tag = 1, Channels = (ushort)options.Channels, Rate = (uint)options.SampleRate, BitsPerSample = 16,
                BlockAlign = (ushort)(2 * options.Channels), BytesPerSecond = (uint)(2 * options.Channels * options.SampleRate),
            };
            nint handle;
            try
            {
                Check(api.Open(&handle, device, &format, _signal.SafeWaitHandle.DangerousGetHandle(), 0, CallbackEvent), $"open {name}");
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
            _headers = (WaveHeader*)NativeMemory.AllocZeroed((nuint)(_count * sizeof(WaveHeader)));
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
                    Check(_api.Prepare(_handle, &_headers[prepared], (uint)sizeof(WaveHeader)), "prepare a buffer");
                    Check(_api.AddBuffer(_handle, &_headers[prepared], (uint)sizeof(WaveHeader)), "queue a buffer");
                }

                Check(_api.Start(_handle), "start recording");
                int next = 0;
                while (!queue.Stopping)
                {
                    _signal.WaitOne(100);
                    while (!queue.Stopping && (_headers[next].Flags & Done) != 0)
                    {
                        queue.Deliver(new ReadOnlySpan<byte>((void*)_headers[next].Data, (int)_headers[next].BytesRecorded));
                        _headers[next].BytesRecorded = 0;
                        Check(_api.AddBuffer(_handle, &_headers[next], (uint)sizeof(WaveHeader)), "queue a buffer again");
                        next = (next + 1) % _count;
                    }
                }

                _api.Reset(_handle);                                         // stops; every queued buffer comes back done
                for (int i = 0; i < _count; i++, next = (next + 1) % _count)
                {
                    if ((_headers[next].Flags & Done) != 0 && _headers[next].BytesRecorded > 0)
                    {
                        queue.Deliver(new ReadOnlySpan<byte>((void*)_headers[next].Data, (int)_headers[next].BytesRecorded));   // what was heard last
                    }
                }
            }
            finally
            {
                _api.Reset(_handle);
                for (int i = 0; i < prepared; i++)
                {
                    _api.Unprepare(_handle, &_headers[i], (uint)sizeof(WaveHeader));
                }

                _api.Close(_handle);
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
                throw new InvalidOperationException($"winmm could not {what}: {_api.ErrorText(result)} (MMRESULT {result}).");
            }
        }
    }

    // winmm.dll's waveIn functions, loaded once; null where it cannot be loaded (Failure says why).
    private sealed class Winmm
    {
        private static readonly Lazy<(Winmm? Api, string Failure)> Loaded = new(Load);

        public static Winmm? Api => Loaded.Value.Api;

        public static string Failure => Loaded.Value.Failure;

        public delegate* unmanaged[Stdcall]<uint> GetNumDevs;
        public delegate* unmanaged[Stdcall]<nuint, WaveInCaps*, uint, uint> GetDevCaps;
        public delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint> Open;
        public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> Prepare;
        public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> Unprepare;
        public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> AddBuffer;
        public delegate* unmanaged[Stdcall]<nint, uint> Start;
        public delegate* unmanaged[Stdcall]<nint, uint> Reset;
        public delegate* unmanaged[Stdcall]<nint, uint> Close;
        public delegate* unmanaged[Stdcall]<uint, char*, uint, uint> GetErrorText;

        public string ErrorText(uint result)
        {
            char* text = stackalloc char[256];
            return GetErrorText(result, text, 256) == 0 ? new string(text, 0, Length(text, 256)) : "no message";
        }

        private static (Winmm?, string) Load()
        {
            if (!OperatingSystem.IsWindows())
            {
                return (null, "not Windows");
            }

            if (!NativeLibrary.TryLoad("winmm.dll", out nint library))
            {
                return (null, "winmm.dll not found");
            }

            try
            {
                return (new Winmm
                {
                    GetNumDevs = (delegate* unmanaged[Stdcall]<uint>)NativeLibrary.GetExport(library, "waveInGetNumDevs"),
                    GetDevCaps = (delegate* unmanaged[Stdcall]<nuint, WaveInCaps*, uint, uint>)NativeLibrary.GetExport(library, "waveInGetDevCapsW"),
                    Open = (delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint>)NativeLibrary.GetExport(library, "waveInOpen"),
                    Prepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)NativeLibrary.GetExport(library, "waveInPrepareHeader"),
                    Unprepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)NativeLibrary.GetExport(library, "waveInUnprepareHeader"),
                    AddBuffer = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)NativeLibrary.GetExport(library, "waveInAddBuffer"),
                    Start = (delegate* unmanaged[Stdcall]<nint, uint>)NativeLibrary.GetExport(library, "waveInStart"),
                    Reset = (delegate* unmanaged[Stdcall]<nint, uint>)NativeLibrary.GetExport(library, "waveInReset"),
                    Close = (delegate* unmanaged[Stdcall]<nint, uint>)NativeLibrary.GetExport(library, "waveInClose"),
                    GetErrorText = (delegate* unmanaged[Stdcall]<uint, char*, uint, uint>)NativeLibrary.GetExport(library, "waveInGetErrorTextW"),
                }, "");
            }
            catch (EntryPointNotFoundException e)
            {
                return (null, e.Message);
            }
        }
    }

    // WAVEFORMATEX (packed to 2 bytes: 18 bytes).
    [StructLayout(LayoutKind.Sequential, Pack = 2)]
    private struct WaveFormat
    {
        public ushort Tag;
        public ushort Channels;
        public uint Rate;
        public uint BytesPerSecond;
        public ushort BlockAlign;
        public ushort BitsPerSample;
        public ushort Size;
    }

    // WAVEHDR.
    [StructLayout(LayoutKind.Sequential)]
    private struct WaveHeader
    {
        public nint Data;
        public uint BufferLength;
        public uint BytesRecorded;
        public nint User;
        public uint Flags;
        public uint Loops;
        public nint Next;
        public nint Reserved;
    }

    // WAVEINCAPSW.
    [StructLayout(LayoutKind.Sequential)]
    private struct WaveInCaps
    {
        public ushort Manufacturer;
        public ushort Product;
        public uint DriverVersion;
        public fixed char Name[32];
        public uint Formats;
        public ushort Channels;
        public ushort Reserved;
    }
}
