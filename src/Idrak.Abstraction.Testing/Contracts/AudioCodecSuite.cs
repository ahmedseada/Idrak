// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>What a sound file's header says, as a codec under test reports it (Idrak.Audio's <c>AudioInfo</c> without the format's name).</summary>
/// <param name="SampleRate">Samples a second.</param>
/// <param name="Channels">Channels.</param>
/// <param name="Samples">Samples per channel, or -1 when the bytes given do not reach the sound's data.</param>
public readonly record struct AudioHeader(int SampleRate, int Channels, long Samples);

/// <summary>Decoded sound as a codec under test gives it: planar float32 samples <c>[Channels, Samples]</c> in [-1, 1].</summary>
/// <param name="SampleRate">Samples a second.</param>
/// <param name="Channels">Channels.</param>
/// <param name="Samples">The planar samples.</param>
public readonly record struct DecodedAudio(int SampleRate, int Channels, float[] Samples);

/// <summary>
/// A sound codec under test (Idrak.Audio's <c>IAudioCodec</c>, given as two delegates since this kit depends on
/// Idrak.Abstraction alone): the header from a file's first bytes (null when not its format), and the decoded sound.
/// </summary>
/// <param name="ReadInfo">The codec's <c>ReadInfo</c> over a file's start (or the whole file).</param>
/// <param name="Decode">The codec's <c>Decode</c> over a whole file.</param>
public sealed record AudioCodecUnderTest(Func<ReadOnlyMemory<byte>, AudioHeader?> ReadInfo, Func<ReadOnlyMemory<byte>, DecodedAudio> Decode);

/// <summary>
/// A sound file a codec is checked on (<see cref="AudioCodecSuite"/>): one it decodes, with the samples it must give
/// when known, or one it must refuse.
/// </summary>
/// <param name="Path">The sound file.</param>
/// <param name="Expected">The planar samples the file decodes to, with its rate and channels; null to check the codec's own consistency only.</param>
/// <param name="Tolerance">How far a decoded sample may be from the expected one (0: exact).</param>
/// <param name="Error">
/// When set, the file is not valid: decoding it must fail with an <see cref="InvalidDataException"/> or a
/// <see cref="NotSupportedException"/> whose message contains this text.
/// </param>
public sealed record AudioSample(string Path, DecodedAudio? Expected = null, float Tolerance = 0f, string? Error = null);

/// <summary>
/// The checks of a sound codec on files of its own. For each sample, whole:
/// <list type="bullet">
/// <item>the header is read from the file's start (its first 64 KiB) or, failing that, from the whole file, and never throws;</item>
/// <item>the decoded sound has the header's rate and channels, at most the header's sample count when it gave one
/// (less when the file holds less than its header claims), a whole number of frames, and finite values in [-1, 1];</item>
/// <item>decoding again, and from two threads at once, gives the same samples;</item>
/// <item>with <see cref="AudioSample.Expected"/>, those samples within the tolerance;</item>
/// <item>an invalid file (<see cref="AudioSample.Error"/>) fails with an error saying what is wrong.</item>
/// </list>
/// For a file cut at a random length (the random cases): the header read never throws and is null, "not reached yet"
/// (-1 samples) or the whole file's; decoding the cut gives a sound no longer than the whole (the data cut to the bytes)
/// or fails with one of the two errors, never anything else.
/// </summary>
/// <param name="samples">The sound files, in the format checked.</param>
public sealed class AudioCodecSuite(IReadOnlyList<AudioSample> samples) : ContractSuite<AudioCodecUnderTest>
{
    private const int HeaderBytes = 64 * 1024;

    private readonly IReadOnlyList<AudioSample> _samples = samples is { Count: > 0 } ? samples
        : throw new ArgumentException("A sound codec is checked on files of its own: give at least one sample.", nameof(samples));

    /// <inheritdoc />
    public override string Name => "audio-codec";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases() => _samples.Select((s, i) => Case(i, cut: -1));

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int index = random.Next(_samples.Count);
        long size = new FileInfo(_samples[index].Path).Length;
        return Case(index, cut: (int)Math.Min(size, random.Next(0, large ? 1 << 20 : 4096)));
    }

    /// <inheritdoc />
    public override void Run(AudioCodecUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        int cut = (int)@case.Data["cut"]!;
        string file = System.IO.Path.GetFileName(sample.Path);
        byte[] whole = File.ReadAllBytes(sample.Path);
        if (cut >= 0)
        {
            RunCut(implementation, sample, whole, cut, checks, file);
            return;
        }

        AudioHeader? header = null;
        try
        {
            header = implementation.ReadInfo(whole.AsMemory(0, Math.Min(whole.Length, HeaderBytes)));
            if (header is { Samples: < 0 } && whole.Length > HeaderBytes)
            {
                header = implementation.ReadInfo(whole);
            }

            checks.Check("the header is read without throwing", true, "");
        }
        catch (Exception e)
        {
            checks.Fail("the header is read without throwing", $"{file}: {e.GetType().Name}: {e.Message}");
        }

        checks.Check("the file is this codec's format", header is not null, $"{file}: ReadInfo gave null");
        if (sample.Error is { } error)
        {
            try
            {
                _ = implementation.Decode(whole);
                checks.Fail("refuses an invalid file", $"decoded {file} without an error (expected one about \"{error}\")");
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                checks.Check("an invalid file's error says what is wrong", e.Message.Contains(error, StringComparison.Ordinal), $"\"{e.Message}\" does not say \"{error}\"");
            }

            return;
        }

        var decoded = implementation.Decode(whole);
        checks.Check("the sound has channels and a rate", decoded.Channels > 0 && decoded.SampleRate > 0, $"{decoded.Channels} channels at {decoded.SampleRate} Hz");
        checks.Check("the samples are whole frames", decoded.Channels > 0 && decoded.Samples.Length % decoded.Channels == 0, $"{decoded.Samples.Length} values over {decoded.Channels} channels");
        if (header is { } h)
        {
            checks.Check("the header's rate and channels", h.SampleRate == decoded.SampleRate && h.Channels == decoded.Channels,
                $"header {h.Channels} x {h.SampleRate} Hz, decoded {decoded.Channels} x {decoded.SampleRate} Hz");
            if (h.Samples >= 0)
            {
                // A header may claim more than the file holds (never fixed after recording): the sound is what is there.
                long frames = decoded.Channels > 0 ? decoded.Samples.Length / decoded.Channels : -1;
                checks.Check("the decoded length is the header's, or less when the file is shorter", frames <= h.Samples, $"header {h.Samples} samples, decoded {frames}");
            }
        }

        int bad = -1;
        for (int i = 0; i < decoded.Samples.Length; i++)
        {
            float v = decoded.Samples[i];
            if (!float.IsFinite(v) || v < -1f || v > 1f)
            {
                bad = i;
                break;
            }
        }

        checks.Check("every sample is finite and in [-1, 1]", bad < 0, bad < 0 ? "" : $"sample {bad} is {decoded.Samples[bad]}");
        checks.Compare("the same samples on a second decoding", Comparisons.Difference(decoded.Samples, implementation.Decode(whole).Samples, 0f));
        var parallel = new DecodedAudio[2];
        Parallel.For(0, 2, i => parallel[i] = implementation.Decode(whole));
        checks.Compare("the same samples from two decodings at once",
            Comparisons.Difference(decoded.Samples, parallel[0].Samples, 0f) ?? Comparisons.Difference(decoded.Samples, parallel[1].Samples, 0f));
        if (sample.Expected is { } expected)
        {
            checks.Compare("the expected samples", Comparisons.Exact((expected.SampleRate, expected.Channels, expected.Samples.Length), (decoded.SampleRate, decoded.Channels, decoded.Samples.Length))
                ?? Comparisons.Difference(expected.Samples, decoded.Samples, sample.Tolerance));
        }
        else
        {
            checks.Skip("the expected samples", "the sample gives none");
        }
    }

    private static void RunCut(AudioCodecUnderTest implementation, AudioSample sample, byte[] whole, int cut, CaseChecks checks, string file)
    {
        var part = whole.AsMemory(0, cut);
        AudioHeader? wholeHeader = null;
        try
        {
            wholeHeader = implementation.ReadInfo(whole);
            var header = implementation.ReadInfo(part);
            checks.Check("a cut file's header never throws", true, "");
            bool fine = header is null || header.Value.Samples < 0 || header == wholeHeader
                        || (wholeHeader is { } w && header.Value.SampleRate == w.SampleRate && header.Value.Channels == w.Channels && header.Value.Samples <= w.Samples);
            checks.Check("a cut file's header is null, not reached, or the whole file's (its data cut)", fine, $"{file} cut at {cut}: {header}, whole {wholeHeader}");
        }
        catch (Exception e)
        {
            checks.Fail("a cut file's header never throws", $"{file} cut at {cut}: {e.GetType().Name}: {e.Message}");
        }

        try
        {
            var decoded = implementation.Decode(part);
            var all = implementation.Decode(whole);
            checks.Check("a cut file decodes no more than the whole", decoded.Samples.Length <= all.Samples.Length && decoded.Channels == all.Channels && decoded.SampleRate == all.SampleRate,
                $"{file} cut at {cut}: {decoded.Samples.Length} samples, whole {all.Samples.Length}");
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            checks.Check("a cut file decodes or is refused as damaged", true, "");
        }
        catch (Exception e)
        {
            checks.Fail("a cut file decodes or is refused as damaged", $"{file} cut at {cut}: {e.GetType().Name}: {e.Message}");
        }
    }

    private ContractCase Case(int sample, int cut) =>
        new($"{System.IO.Path.GetFileName(_samples[sample].Path)}{(cut < 0 ? (_samples[sample].Error is null ? "" : ", refused") : $", cut at {cut} bytes")}",
            new JsonObject { ["sample"] = sample, ["cut"] = cut });
}
