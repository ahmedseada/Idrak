// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;

namespace Idrak.Audio;

/// <summary>
/// Android's AAudio, libaaudio.so (Android 8.0, API 26, and later), loaded once at run time through its C functions
/// (no Java binding, nothing shipped); null off Android or where it cannot be loaded (<see cref="Failure"/> says why).
/// The "aaudio" capture and playback share it and its blocking calls (read and write with a timeout, on a thread of
/// their own). Every number below is the NDK's, from <c>aaudio/AAudio.h</c>; the types are its typedefs:
/// <c>aaudio_result_t</c>, <c>aaudio_direction_t</c>, <c>aaudio_format_t</c>, <c>aaudio_sharing_mode_t</c>,
/// <c>aaudio_performance_mode_t</c> and <c>aaudio_stream_state_t</c> are all <c>int32_t</c>; the builder and the stream
/// are opaque pointers.
/// </summary>
internal sealed unsafe class AAudioLibrary
{
    /// <summary>AAUDIO_UNSPECIFIED (0): the device's own value (a device id, a rate, the channels, a capacity).</summary>
    public const int Unspecified = 0;

    /// <summary>AAUDIO_DIRECTION_OUTPUT (0) and AAUDIO_DIRECTION_INPUT (1).</summary>
    public const int DirectionOutput = 0, DirectionInput = 1;

    /// <summary>AAUDIO_FORMAT_PCM_I16 (1) and AAUDIO_FORMAT_PCM_FLOAT (2); AAUDIO_FORMAT_UNSPECIFIED is 0, AAUDIO_FORMAT_INVALID -1.</summary>
    public const int FormatI16 = 1, FormatFloat = 2;

    /// <summary>AAUDIO_SHARING_MODE_EXCLUSIVE (0) and AAUDIO_SHARING_MODE_SHARED (1).</summary>
    public const int SharingExclusive = 0, SharingShared = 1;

    /// <summary>AAUDIO_PERFORMANCE_MODE_NONE (10), _POWER_SAVING (11) and _LOW_LATENCY (12).</summary>
    public const int PerformanceNone = 10, PerformancePowerSaving = 11, PerformanceLowLatency = 12;

    /// <summary>AAUDIO_STREAM_STATE_STOPPING (9) and AAUDIO_STREAM_STATE_STOPPED (10).</summary>
    public const int StateStopping = 9, StateStopped = 10;

    /// <summary>The timeout of each blocking read or write, in nanoseconds (the stop flag is looked at between them).</summary>
    public const long Timeout = 100_000_000;

    private static readonly Lazy<(AAudioLibrary? Api, string Failure)> Loaded = new(Load);

    public static AAudioLibrary? Api => Loaded.Value.Api;

    public static string Failure => Loaded.Value.Failure;

    // aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** builder)
    public delegate* unmanaged<nint*, int> CreateStreamBuilder;

    // void AAudioStreamBuilder_setX(AAudioStreamBuilder* builder, int32_t value): setDirection, setDeviceId,
    // setSampleRate, setChannelCount, setFormat, setPerformanceMode, setSharingMode, setBufferCapacityInFrames
    public delegate* unmanaged<nint, int, void> SetDirection;
    public delegate* unmanaged<nint, int, void> SetDeviceId;
    public delegate* unmanaged<nint, int, void> SetSampleRate;
    public delegate* unmanaged<nint, int, void> SetChannelCount;
    public delegate* unmanaged<nint, int, void> SetFormat;
    public delegate* unmanaged<nint, int, void> SetPerformanceMode;
    public delegate* unmanaged<nint, int, void> SetSharingMode;
    public delegate* unmanaged<nint, int, void> SetBufferCapacityInFrames;

    // aaudio_result_t AAudioStreamBuilder_openStream(AAudioStreamBuilder* builder, AAudioStream** stream)
    public delegate* unmanaged<nint, nint*, int> OpenStream;

    // aaudio_result_t AAudioStreamBuilder_delete(AAudioStreamBuilder* builder)
    public delegate* unmanaged<nint, int> DeleteBuilder;

    // aaudio_result_t AAudioStream_requestStart / requestStop / close(AAudioStream* stream)
    public delegate* unmanaged<nint, int> RequestStart;
    public delegate* unmanaged<nint, int> RequestStop;
    public delegate* unmanaged<nint, int> Close;

    // aaudio_result_t AAudioStream_read(AAudioStream* stream, void* buffer, int32_t numFrames, int64_t timeoutNanoseconds)
    // and AAudioStream_write(AAudioStream* stream, const void* buffer, int32_t numFrames, int64_t timeoutNanoseconds):
    // the frames moved (fewer on a timeout), or a negative error
    public delegate* unmanaged<nint, void*, int, long, int> Read;
    public delegate* unmanaged<nint, void*, int, long, int> Write;

    // int32_t AAudioStream_getSampleRate / getChannelCount / getDeviceId / getFramesPerBurst(AAudioStream* stream),
    // aaudio_format_t AAudioStream_getFormat(AAudioStream* stream)
    public delegate* unmanaged<nint, int> GetSampleRate;
    public delegate* unmanaged<nint, int> GetChannelCount;
    public delegate* unmanaged<nint, int> GetDeviceId;
    public delegate* unmanaged<nint, int> GetFramesPerBurst;
    public delegate* unmanaged<nint, int> GetFormat;

    // int64_t AAudioStream_getFramesWritten / getFramesRead(AAudioStream* stream)
    public delegate* unmanaged<nint, long> GetFramesWritten;
    public delegate* unmanaged<nint, long> GetFramesRead;

    // aaudio_result_t AAudioStream_waitForStateChange(AAudioStream* stream, aaudio_stream_state_t inputState,
    //     aaudio_stream_state_t* nextState, int64_t timeoutNanoseconds)
    public delegate* unmanaged<nint, int, int*, long, int> WaitForStateChange;

    // const char* AAudio_convertResultToText(aaudio_result_t returnCode)
    public delegate* unmanaged<int, byte*> ConvertResultToText;

    /// <summary>Whether AAudio can run here; <paramref name="reason"/> says why not.</summary>
    public static bool Available(out string reason)
    {
        if (!OperatingSystem.IsAndroid())
        {
            reason = "AAudio is Android's (8.0, API 26, and later)";
            return false;
        }

        if (Api is null)
        {
            reason = $"libaaudio.so could not be loaded ({Failure}; Android 8.0, API 26, and later have it)";
            return false;
        }

        reason = "";
        return true;
    }

    /// <summary>
    /// The device id <paramref name="device"/> names: AAUDIO_UNSPECIFIED (the default) for null, empty or "default",
    /// else a number (an id from Android's AudioManager.getDevices, AudioDeviceInfo.getId).
    /// </summary>
    /// <exception cref="ArgumentException">It is neither.</exception>
    public static int DeviceId(string? device)
    {
        if (device is null or "" || device.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return Unspecified;
        }

        return int.TryParse(device, NumberStyles.None, CultureInfo.InvariantCulture, out int id) && id > 0 ? id
            : throw new ArgumentException($"AAudio takes \"default\" or a device id (a number from Android's AudioManager.getDevices), not '{device}': the NDK lists no devices by name.", nameof(device));
    }

    /// <summary>AAudio's performance mode for <paramref name="performance"/>.</summary>
    public static int PerformanceMode(AudioPerformance performance) => performance switch
    {
        AudioPerformance.PowerSaving => PerformancePowerSaving,
        AudioPerformance.LowLatency => PerformanceLowLatency,
        _ => PerformanceNone,
    };

    /// <summary>The name of an aaudio_result_t from AAudio.h, or null for one it does not define.</summary>
    public static string? ResultName(int result) => result switch
    {
        0 => "AAUDIO_OK",
        -899 => "AAUDIO_ERROR_DISCONNECTED",
        -898 => "AAUDIO_ERROR_ILLEGAL_ARGUMENT",
        -896 => "AAUDIO_ERROR_INTERNAL",
        -895 => "AAUDIO_ERROR_INVALID_STATE",
        -892 => "AAUDIO_ERROR_INVALID_HANDLE",
        -890 => "AAUDIO_ERROR_UNIMPLEMENTED",
        -889 => "AAUDIO_ERROR_UNAVAILABLE",
        -888 => "AAUDIO_ERROR_NO_FREE_HANDLES",
        -887 => "AAUDIO_ERROR_NO_MEMORY",
        -886 => "AAUDIO_ERROR_NULL",
        -885 => "AAUDIO_ERROR_TIMEOUT",
        -884 => "AAUDIO_ERROR_WOULD_BLOCK",
        -883 => "AAUDIO_ERROR_INVALID_FORMAT",
        -882 => "AAUDIO_ERROR_OUT_OF_RANGE",
        -881 => "AAUDIO_ERROR_NO_SERVICE",
        -880 => "AAUDIO_ERROR_INVALID_RATE",
        _ => null,
    };

    /// <summary>
    /// The message of a failed call: what was tried, AAudio's own text (when the library is loaded), the result's name
    /// and number; for the microphone, the permission Android needs, as a refusal looks like any other failure here.
    /// </summary>
    public static string Message(string what, int result, bool input)
    {
        string? text = Api is { } api && api.ConvertResultToText != null ? Marshal.PtrToStringUTF8((nint)api.ConvertResultToText(result)) : null;
        string name = ResultName(result) ?? "an unknown result";
        string message = $"AAudio could not {what}: {(string.IsNullOrEmpty(text) || text == name ? "" : text + ", ")}{name} ({result}).";
        return input
            ? message + " Android refuses the microphone to an app without the RECORD_AUDIO permission: the app declares it in its manifest and asks for it "
                + "at run time (MAUI: Permissions.RequestAsync<Permissions.Microphone>()) before opening the capture."
            : message;
    }

    /// <summary>Throws <see cref="Message"/> for a negative <paramref name="result"/>.</summary>
    public static void Check(int result, string what, bool input)
    {
        if (result < 0)
        {
            throw new InvalidOperationException(Message(what, result, input));
        }
    }

    /// <summary>
    /// Opens a stream through a builder (deleted before returning, opened or not): the direction, the device, the rate
    /// and channels asked (or the device's own), 16-bit samples, the performance and sharing modes, and the buffer's
    /// capacity in frames (AAUDIO_UNSPECIFIED leaves it to the device).
    /// </summary>
    public nint OpenStreamWith(int direction, int deviceId, int sampleRate, int channels, AudioPerformance performance, bool exclusive, int capacityFrames)
    {
        bool input = direction == DirectionInput;
        nint builder;
        Check(CreateStreamBuilder(&builder), "create a stream builder", input: false);
        try
        {
            SetDirection(builder, direction);
            SetDeviceId(builder, deviceId);
            SetSampleRate(builder, sampleRate);
            SetChannelCount(builder, channels);
            SetFormat(builder, FormatI16);                                   // AAudio converts to the device's own format
            SetPerformanceMode(builder, PerformanceMode(performance));
            SetSharingMode(builder, exclusive ? SharingExclusive : SharingShared);
            SetBufferCapacityInFrames(builder, capacityFrames);
            nint stream;
            Check(OpenStream(builder, &stream), $"open the {(input ? "microphone" : "speaker")} ({Describe(deviceId, sampleRate, channels)})", input);
            return stream;
        }
        finally
        {
            DeleteBuilder(builder);
        }
    }

    /// <summary>The stream's format, checked to be one the library reads and writes.</summary>
    public int CheckedFormat(nint stream, bool input)
    {
        int format = GetFormat(stream);
        if (format is not (FormatI16 or FormatFloat))
        {
            Close(stream);
            throw new InvalidOperationException($"AAudio opened the {(input ? "microphone" : "speaker")} with sample format {format}; the library reads and writes "
                + $"AAUDIO_FORMAT_PCM_I16 ({FormatI16}) and AAUDIO_FORMAT_PCM_FLOAT ({FormatFloat}).");
        }

        return format;
    }

    private static string Describe(int deviceId, int sampleRate, int channels)
    {
        var c = CultureInfo.InvariantCulture;
        return (deviceId == Unspecified ? "the default device" : "device " + deviceId.ToString(c)) + ", "
            + (sampleRate == Unspecified ? "its own rate" : sampleRate.ToString(c) + " Hz") + ", "
            + (channels == Unspecified ? "its own channels" : channels.ToString(c) + " channel(s)");
    }

    private static (AAudioLibrary?, string) Load()
    {
        if (!OperatingSystem.IsAndroid())
        {
            return (null, "not Android");
        }

        if (!NativeLibrary.TryLoad("libaaudio.so", out nint library))
        {
            return (null, "libaaudio.so not found");
        }

        try
        {
            nint F(string name) => NativeLibrary.GetExport(library, name);
            return (new AAudioLibrary
            {
                CreateStreamBuilder = (delegate* unmanaged<nint*, int>)F("AAudio_createStreamBuilder"),
                SetDirection = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setDirection"),
                SetDeviceId = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setDeviceId"),
                SetSampleRate = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setSampleRate"),
                SetChannelCount = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setChannelCount"),
                SetFormat = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setFormat"),
                SetPerformanceMode = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setPerformanceMode"),
                SetSharingMode = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setSharingMode"),
                SetBufferCapacityInFrames = (delegate* unmanaged<nint, int, void>)F("AAudioStreamBuilder_setBufferCapacityInFrames"),
                OpenStream = (delegate* unmanaged<nint, nint*, int>)F("AAudioStreamBuilder_openStream"),
                DeleteBuilder = (delegate* unmanaged<nint, int>)F("AAudioStreamBuilder_delete"),
                RequestStart = (delegate* unmanaged<nint, int>)F("AAudioStream_requestStart"),
                RequestStop = (delegate* unmanaged<nint, int>)F("AAudioStream_requestStop"),
                Close = (delegate* unmanaged<nint, int>)F("AAudioStream_close"),
                Read = (delegate* unmanaged<nint, void*, int, long, int>)F("AAudioStream_read"),
                Write = (delegate* unmanaged<nint, void*, int, long, int>)F("AAudioStream_write"),
                GetSampleRate = (delegate* unmanaged<nint, int>)F("AAudioStream_getSampleRate"),
                GetChannelCount = (delegate* unmanaged<nint, int>)F("AAudioStream_getChannelCount"),
                GetDeviceId = (delegate* unmanaged<nint, int>)F("AAudioStream_getDeviceId"),
                GetFramesPerBurst = (delegate* unmanaged<nint, int>)F("AAudioStream_getFramesPerBurst"),
                GetFormat = (delegate* unmanaged<nint, int>)F("AAudioStream_getFormat"),
                GetFramesWritten = (delegate* unmanaged<nint, long>)F("AAudioStream_getFramesWritten"),
                GetFramesRead = (delegate* unmanaged<nint, long>)F("AAudioStream_getFramesRead"),
                WaitForStateChange = (delegate* unmanaged<nint, int, int*, long, int>)F("AAudioStream_waitForStateChange"),
                ConvertResultToText = (delegate* unmanaged<int, byte*>)F("AAudio_convertResultToText"),
            }, "");
        }
        catch (EntryPointNotFoundException e)
        {
            return (null, e.Message);
        }
    }
}
