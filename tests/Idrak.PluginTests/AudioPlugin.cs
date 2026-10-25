// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using Idrak.Abstraction.Plugins;
using Idrak.Audio.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// This assembly as a plug-in for sound formats (plan 21, S1): <see cref="Plugin"/> ("outside-audio", in this assembly's
/// <see cref="PluginTestsPackage"/>) registers the "outside-raw" codec (<see cref="OutsideRawCodec"/>), a format of this
/// assembly's own that the library does not read, so the tests see an app's codec asked before the library's and
/// removed again.
/// </summary>
public static class AudioPlugin
{
    /// <summary>The plug-in: the codec (what <c>-P</c> registers through the package).</summary>
    public static IIdrakPlugin Plugin { get; } = new AudioRegistration();

    /// <summary>Whether the plug-in is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the codec (through <see cref="IdrakPlugins"/>); once.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Removes what <see cref="Register"/> registered (tests leave the registries as they found them).</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class AudioRegistration() : IdrakPlugin("outside-audio")
    {
        protected override void RegisterParts() => Part(() => AudioCodecs.Register(new OutsideRawCodec()), () => AudioCodecs.Unregister(OutsideRawCodec.FormatName));
    }
}

/// <summary>
/// A sound format of the tests' own: "ORAW", then the rate, channels and samples per channel as little-endian int32,
/// then the planar float32 samples.
/// </summary>
public sealed class OutsideRawCodec : IAudioCodec
{
    /// <summary>The format's name.</summary>
    public const string FormatName = "outside-raw";

    /// <inheritdoc />
    public string Name => FormatName;

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".oraw"];

    /// <summary>The file's bytes for planar samples.</summary>
    public static byte[] Encode(float[] planar, int channels, int sampleRate)
    {
        var bytes = new byte[16 + planar.Length * 4];
        "ORAW"u8.CopyTo(bytes);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8), channels);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12), planar.Length / channels);
        for (int i = 0; i < planar.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(16 + i * 4), planar[i]);
        }

        return bytes;
    }

    /// <inheritdoc />
    public AudioInfo? ReadInfo(ReadOnlySpan<byte> header) =>
        header.Length >= 16 && header[..4].SequenceEqual("ORAW"u8)
            ? new AudioInfo(BinaryPrimitives.ReadInt32LittleEndian(header[4..]), BinaryPrimitives.ReadInt32LittleEndian(header[8..]),
                BinaryPrimitives.ReadInt32LittleEndian(header[12..]), 32, true, FormatName)
            : null;

    /// <inheritdoc />
    public AudioData Decode(ReadOnlySpan<byte> file)
    {
        var info = ReadInfo(file) ?? throw new InvalidDataException("not an ORAW file.");
        int count = checked(info.Channels * (int)info.Samples);
        if (file.Length < 16 + count * 4)
        {
            throw new InvalidDataException($"{file.Length} bytes hold fewer than the {info.Samples} samples the header says.");
        }

        var planar = new float[count];
        for (int i = 0; i < count; i++)
        {
            planar[i] = BinaryPrimitives.ReadSingleLittleEndian(file.Slice(16 + i * 4, 4));
        }

        return new AudioData(planar, info.Channels, info.SampleRate);
    }
}
