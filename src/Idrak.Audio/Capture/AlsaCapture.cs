// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;

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
    public bool IsAvailable(out string reason)
    {
        if (!OperatingSystem.IsLinux())
        {
            reason = "ALSA is Linux's";
            return false;
        }

        if (Alsa.Api is null)
        {
            reason = $"libasound.so.2 could not be loaded ({Alsa.Failure}; install alsa-lib, libasound2 on Debian and Ubuntu)";
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

        var api = Alsa.Api!;
        var devices = new List<AudioInputDevice> { new("default", "the system's default input", true) };
        void** hints;
        fixed (byte* pcm = "pcm\0"u8)
        {
            if (api.NameHint(-1, pcm, &hints) != 0)
            {
                return devices;
            }
        }

        try
        {
            for (void** hint = hints; *hint != null; hint++)
            {
                string? name = Hint(api, *hint, "NAME\0"u8), description = Hint(api, *hint, "DESC\0"u8), direction = Hint(api, *hint, "IOID\0"u8);
                if (name is null or "default" || direction == "Output")
                {
                    continue;                                                 // playback only, or the default listed first
                }

                devices.Add(new AudioInputDevice(name, (description ?? name).ReplaceLineEndings(" "), false));
            }
        }
        finally
        {
            api.FreeHint(hints);
        }

        return devices;
    }

    // One field of a name hint (`id` ends with its terminating zero; ALSA's value is malloc'd and freed here), or null.
    private static string? Hint(Alsa api, void* hint, ReadOnlySpan<byte> id)
    {
        fixed (byte* field = id)
        {
            byte* value = api.GetHint(hint, field);
            if (value == null)
            {
                return null;
            }

            try
            {
                return Marshal.PtrToStringUTF8((nint)value);
            }
            finally
            {
                NativeMemory.Free(value);
            }
        }
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

        return new AlsaInput(Alsa.Api!, string.IsNullOrEmpty(options.Device) ? "default" : options.Device, options).Queue;
    }

    // One recording: snd_pcm_wait then snd_pcm_readi of a block, an overrun recovered (the sound lost is not made up).
    private sealed class AlsaInput
    {
        private const int Capture = 1, FormatS16LE = 2, AccessInterleaved = 3;

        private readonly Alsa _api;
        private readonly nint _pcm;
        private readonly int _frames;
        private readonly int _channels;

        public AlsaInput(Alsa api, string device, AudioCaptureOptions options)
        {
            _api = api;
            _channels = options.Channels;
            nint pcm;
            fixed (byte* name = Encoding.UTF8.GetBytes(device + "\0"))
            {
                Check(api.Open(&pcm, name, Capture, 0), $"open '{device}'");
            }

            _pcm = pcm;
            _frames = Math.Max(1, (int)Math.Round(options.BlockSeconds * options.SampleRate));
            try
            {
                uint latency = (uint)Math.Clamp(4 * options.BlockSeconds * 1e6, 20_000, 500_000);   // the device's buffer: four blocks
                Check(api.SetParams(_pcm, FormatS16LE, AccessInterleaved, (uint)options.Channels, (uint)options.SampleRate, 1, latency),
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
                            Check(_api.Recover(_pcm, (int)got, 1), "recover from a read error");   // an overrun: the stream restarts
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

        private void Check(int result, string what)
        {
            if (result < 0)
            {
                throw new InvalidOperationException($"ALSA could not {what}: {Marshal.PtrToStringUTF8((nint)_api.Error(result))} ({result}).");
            }
        }
    }

    // libasound.so.2's PCM functions, loaded once; null where it cannot be loaded (Failure says why).
    private sealed class Alsa
    {
        private static readonly Lazy<(Alsa? Api, string Failure)> Loaded = new(Load);

        public static Alsa? Api => Loaded.Value.Api;

        public static string Failure => Loaded.Value.Failure;

        public delegate* unmanaged<nint*, byte*, int, int, int> Open;
        public delegate* unmanaged<nint, int, int, uint, uint, int, uint, int> SetParams;
        public delegate* unmanaged<nint, int, int> Wait;
        public delegate* unmanaged<nint, byte*, nuint, nint> ReadInterleaved;
        public delegate* unmanaged<nint, int, int, int> Recover;
        public delegate* unmanaged<nint, int> Drop;
        public delegate* unmanaged<nint, int> Close;
        public delegate* unmanaged<int, byte*> Error;
        public delegate* unmanaged<int, byte*, void***, int> NameHint;
        public delegate* unmanaged<void*, byte*, byte*> GetHint;
        public delegate* unmanaged<void**, int> FreeHint;

        private static (Alsa?, string) Load()
        {
            if (!OperatingSystem.IsLinux())
            {
                return (null, "not Linux");
            }

            if (!NativeLibrary.TryLoad("libasound.so.2", out nint library) && !NativeLibrary.TryLoad("libasound.so", out library))
            {
                return (null, "libasound.so.2 not found");
            }

            try
            {
                nint F(string name) => NativeLibrary.GetExport(library, name);
                return (new Alsa
                {
                    Open = (delegate* unmanaged<nint*, byte*, int, int, int>)F("snd_pcm_open"),
                    SetParams = (delegate* unmanaged<nint, int, int, uint, uint, int, uint, int>)F("snd_pcm_set_params"),
                    Wait = (delegate* unmanaged<nint, int, int>)F("snd_pcm_wait"),
                    ReadInterleaved = (delegate* unmanaged<nint, byte*, nuint, nint>)F("snd_pcm_readi"),
                    Recover = (delegate* unmanaged<nint, int, int, int>)F("snd_pcm_recover"),
                    Drop = (delegate* unmanaged<nint, int>)F("snd_pcm_drop"),
                    Close = (delegate* unmanaged<nint, int>)F("snd_pcm_close"),
                    Error = (delegate* unmanaged<int, byte*>)F("snd_strerror"),
                    NameHint = (delegate* unmanaged<int, byte*, void***, int>)F("snd_device_name_hint"),
                    GetHint = (delegate* unmanaged<void*, byte*, byte*>)F("snd_device_name_get_hint"),
                    FreeHint = (delegate* unmanaged<void**, int>)F("snd_device_name_free_hint"),
                }, "");
            }
            catch (EntryPointNotFoundException e)
            {
                return (null, e.Message);
            }
        }
    }
}
