// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Audio;

/// <summary>
/// libasound.so.2's PCM functions, loaded once at run time (nothing native shipped); null where it cannot be loaded
/// (<see cref="Failure"/> says why). The "alsa" capture and playback share it. The constants are alsa-lib's
/// (include/pcm.h).
/// </summary>
internal sealed unsafe class AlsaLibrary
{
    /// <summary>SND_PCM_STREAM_PLAYBACK and SND_PCM_STREAM_CAPTURE (snd_pcm_stream_t).</summary>
    public const int StreamPlayback = 0, StreamCapture = 1;

    /// <summary>SND_PCM_FORMAT_S16_LE (snd_pcm_format_t).</summary>
    public const int FormatS16LE = 2;

    /// <summary>SND_PCM_ACCESS_RW_INTERLEAVED (snd_pcm_access_t).</summary>
    public const int AccessInterleaved = 3;

    private static readonly Lazy<(AlsaLibrary? Api, string Failure)> Loaded = new(Load);

    public static AlsaLibrary? Api => Loaded.Value.Api;

    public static string Failure => Loaded.Value.Failure;

    public delegate* unmanaged<nint*, byte*, int, int, int> Open;
    public delegate* unmanaged<nint, int, int, uint, uint, int, uint, int> SetParams;
    public delegate* unmanaged<nint, int, int> Wait;
    public delegate* unmanaged<nint, byte*, nuint, nint> ReadInterleaved;
    public delegate* unmanaged<nint, byte*, nuint, nint> WriteInterleaved;
    public delegate* unmanaged<nint, int, int, int> Recover;
    public delegate* unmanaged<nint, int> Drain;
    public delegate* unmanaged<nint, int> Prepare;
    public delegate* unmanaged<nint, int> Drop;
    public delegate* unmanaged<nint, int> Close;
    public delegate* unmanaged<int, byte*> Error;
    public delegate* unmanaged<int, byte*, void***, int> NameHint;
    public delegate* unmanaged<void*, byte*, byte*> GetHint;
    public delegate* unmanaged<void**, int> FreeHint;

    /// <summary>Whether ALSA can run here; <paramref name="reason"/> says why not.</summary>
    public static bool Available(out string reason)
    {
        if (!OperatingSystem.IsLinux())
        {
            reason = "ALSA is Linux's";
            return false;
        }

        if (Api is null)
        {
            reason = $"libasound.so.2 could not be loaded ({Failure}; install alsa-lib, libasound2 on Debian and Ubuntu)";
            return false;
        }

        reason = "";
        return true;
    }

    /// <summary>Opens <paramref name="device"/> for <paramref name="stream"/> (blocking mode).</summary>
    public nint OpenPcm(string device, int stream)
    {
        nint pcm;
        fixed (byte* name = Encoding.UTF8.GetBytes(device + "\0"))
        {
            Check(Open(&pcm, name, stream, 0), $"open '{device}' for {(stream == StreamCapture ? "recording" : "playing")}");
        }

        return pcm;
    }

    /// <summary>Throws the ALSA message of a negative <paramref name="result"/>.</summary>
    public void Check(int result, string what)
    {
        if (result < 0)
        {
            throw new InvalidOperationException($"ALSA could not {what}: {Marshal.PtrToStringUTF8((nint)Error(result))} ({result}).");
        }
    }

    /// <summary>
    /// The PCM names of the devices that record (<paramref name="capture"/>) or play, from ALSA's name hints, with their
    /// descriptions ("default" left out: the callers list it first).
    /// </summary>
    public List<(string Id, string Name)> Devices(bool capture)
    {
        var devices = new List<(string, string)>();
        void** hints;
        fixed (byte* pcm = "pcm\0"u8)
        {
            if (NameHint(-1, pcm, &hints) != 0)
            {
                return devices;
            }
        }

        try
        {
            string other = capture ? "Output" : "Input";
            for (void** hint = hints; *hint != null; hint++)
            {
                string? name = Hint(*hint, "NAME\0"u8), description = Hint(*hint, "DESC\0"u8), direction = Hint(*hint, "IOID\0"u8);
                if (name is null or "default" || direction == other)
                {
                    continue;                                                 // the other direction only, or the default listed first
                }

                devices.Add((name, (description ?? name).ReplaceLineEndings(" ")));
            }
        }
        finally
        {
            FreeHint(hints);
        }

        return devices;
    }

    // One field of a name hint (`id` ends with its terminating zero; ALSA's value is malloc'd and freed here), or null.
    private string? Hint(void* hint, ReadOnlySpan<byte> id)
    {
        fixed (byte* field = id)
        {
            byte* value = GetHint(hint, field);
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

    private static (AlsaLibrary?, string) Load()
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
            return (new AlsaLibrary
            {
                Open = (delegate* unmanaged<nint*, byte*, int, int, int>)F("snd_pcm_open"),
                SetParams = (delegate* unmanaged<nint, int, int, uint, uint, int, uint, int>)F("snd_pcm_set_params"),
                Wait = (delegate* unmanaged<nint, int, int>)F("snd_pcm_wait"),
                ReadInterleaved = (delegate* unmanaged<nint, byte*, nuint, nint>)F("snd_pcm_readi"),
                WriteInterleaved = (delegate* unmanaged<nint, byte*, nuint, nint>)F("snd_pcm_writei"),
                Recover = (delegate* unmanaged<nint, int, int, int>)F("snd_pcm_recover"),
                Drain = (delegate* unmanaged<nint, int>)F("snd_pcm_drain"),
                Prepare = (delegate* unmanaged<nint, int>)F("snd_pcm_prepare"),
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
