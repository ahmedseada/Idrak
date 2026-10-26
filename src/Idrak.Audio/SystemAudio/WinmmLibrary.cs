// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// winmm.dll's waveIn and waveOut functions, loaded once at run time; null where it cannot be loaded (<see cref="Failure"/>
/// says why). The "winmm" capture and playback share it. The constants are mmsystem.h's (mmeapi.h).
/// </summary>
internal sealed unsafe class WinmmLibrary
{
    /// <summary>WAVE_MAPPER, (UINT)-1: the system's default device, converting from the device's own format.</summary>
    public const uint WaveMapper = uint.MaxValue;

    /// <summary>CALLBACK_EVENT: the callback is an event handle, signalled when a buffer is done.</summary>
    public const uint CallbackEvent = 0x00050000;

    /// <summary>WHDR_DONE: the device has finished with the buffer.</summary>
    public const uint Done = 0x00000001;

    /// <summary>WHDR_INQUEUE: the buffer is queued with the device.</summary>
    public const uint InQueue = 0x00000010;

    private static readonly Lazy<(WinmmLibrary? Api, string Failure)> Loaded = new(Load);

    public static WinmmLibrary? Api => Loaded.Value.Api;

    public static string Failure => Loaded.Value.Failure;

    // waveIn
    public delegate* unmanaged[Stdcall]<uint> InGetNumDevs;
    public delegate* unmanaged[Stdcall]<nuint, WaveInCaps*, uint, uint> InGetDevCaps;
    public delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint> InOpen;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> InPrepare;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> InUnprepare;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> InAddBuffer;
    public delegate* unmanaged[Stdcall]<nint, uint> InStart;
    public delegate* unmanaged[Stdcall]<nint, uint> InReset;
    public delegate* unmanaged[Stdcall]<nint, uint> InClose;
    public delegate* unmanaged[Stdcall]<uint, char*, uint, uint> InGetErrorText;

    // waveOut
    public delegate* unmanaged[Stdcall]<uint> OutGetNumDevs;
    public delegate* unmanaged[Stdcall]<nuint, WaveOutCaps*, uint, uint> OutGetDevCaps;
    public delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint> OutOpen;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> OutPrepare;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> OutUnprepare;
    public delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint> OutWrite;
    public delegate* unmanaged[Stdcall]<nint, uint> OutReset;
    public delegate* unmanaged[Stdcall]<nint, uint> OutClose;
    public delegate* unmanaged[Stdcall]<uint, char*, uint, uint> OutGetErrorText;

    /// <summary>Whether winmm can run here; <paramref name="reason"/> says why not.</summary>
    public static bool Available(string what, out string reason)
    {
        if (!OperatingSystem.IsWindows())
        {
            reason = $"{what} is Windows'";
            return false;
        }

        if (Api is null)
        {
            reason = $"winmm.dll could not be loaded ({Failure})";
            return false;
        }

        reason = "";
        return true;
    }

    /// <summary>The system's text for a waveIn result.</summary>
    public string InErrorText(uint result) => Text(InGetErrorText, result);

    /// <summary>The system's text for a waveOut result.</summary>
    public string OutErrorText(uint result) => Text(OutGetErrorText, result);

    private static string Text(delegate* unmanaged[Stdcall]<uint, char*, uint, uint> getText, uint result)
    {
        char* text = stackalloc char[256];
        return getText(result, text, 256) == 0 ? new string(text, 0, Length(text, 256)) : "no message";
    }

    /// <summary>The characters of a zero-terminated text of at most <paramref name="max"/>.</summary>
    public static int Length(char* text, int max)
    {
        int n = 0;
        while (n < max && text[n] != '\0')
        {
            n++;
        }

        return n;
    }

    private static (WinmmLibrary?, string) Load()
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
            nint F(string name) => NativeLibrary.GetExport(library, name);
            return (new WinmmLibrary
            {
                InGetNumDevs = (delegate* unmanaged[Stdcall]<uint>)F("waveInGetNumDevs"),
                InGetDevCaps = (delegate* unmanaged[Stdcall]<nuint, WaveInCaps*, uint, uint>)F("waveInGetDevCapsW"),
                InOpen = (delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint>)F("waveInOpen"),
                InPrepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveInPrepareHeader"),
                InUnprepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveInUnprepareHeader"),
                InAddBuffer = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveInAddBuffer"),
                InStart = (delegate* unmanaged[Stdcall]<nint, uint>)F("waveInStart"),
                InReset = (delegate* unmanaged[Stdcall]<nint, uint>)F("waveInReset"),
                InClose = (delegate* unmanaged[Stdcall]<nint, uint>)F("waveInClose"),
                InGetErrorText = (delegate* unmanaged[Stdcall]<uint, char*, uint, uint>)F("waveInGetErrorTextW"),
                OutGetNumDevs = (delegate* unmanaged[Stdcall]<uint>)F("waveOutGetNumDevs"),
                OutGetDevCaps = (delegate* unmanaged[Stdcall]<nuint, WaveOutCaps*, uint, uint>)F("waveOutGetDevCapsW"),
                OutOpen = (delegate* unmanaged[Stdcall]<nint*, uint, WaveFormat*, nint, nint, uint, uint>)F("waveOutOpen"),
                OutPrepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveOutPrepareHeader"),
                OutUnprepare = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveOutUnprepareHeader"),
                OutWrite = (delegate* unmanaged[Stdcall]<nint, WaveHeader*, uint, uint>)F("waveOutWrite"),
                OutReset = (delegate* unmanaged[Stdcall]<nint, uint>)F("waveOutReset"),
                OutClose = (delegate* unmanaged[Stdcall]<nint, uint>)F("waveOutClose"),
                OutGetErrorText = (delegate* unmanaged[Stdcall]<uint, char*, uint, uint>)F("waveOutGetErrorTextW"),
            }, "");
        }
        catch (EntryPointNotFoundException e)
        {
            return (null, e.Message);
        }
    }

    /// <summary>A 16-bit PCM WAVEFORMATEX of <paramref name="channels"/> at <paramref name="rate"/>.</summary>
    public static WaveFormat Pcm16(int channels, int rate) => new()
    {
        Tag = 1, Channels = (ushort)channels, Rate = (uint)rate, BitsPerSample = 16,
        BlockAlign = (ushort)(2 * channels), BytesPerSecond = (uint)(2 * channels * rate),
    };

    // WAVEFORMATEX (packed to 2 bytes: 18 bytes).
    [StructLayout(LayoutKind.Sequential, Pack = 2)]
    public struct WaveFormat
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
    public struct WaveHeader
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
    public struct WaveInCaps
    {
        public ushort Manufacturer;
        public ushort Product;
        public uint DriverVersion;
        public fixed char Name[32];
        public uint Formats;
        public ushort Channels;
        public ushort Reserved;
    }

    // WAVEOUTCAPSW: WAVEINCAPSW's fields and dwSupport.
    [StructLayout(LayoutKind.Sequential)]
    public struct WaveOutCaps
    {
        public ushort Manufacturer;
        public ushort Product;
        public uint DriverVersion;
        public fixed char Name[32];
        public uint Formats;
        public ushort Channels;
        public ushort Reserved;
        public uint Support;
    }
}
