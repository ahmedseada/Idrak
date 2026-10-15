// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>A piece of a transcription as the speech recognizer checks see it.</summary>
/// <param name="Start">Its start, in seconds from the sound's start.</param>
/// <param name="End">Its end, in seconds.</param>
/// <param name="Text">Its text.</param>
public readonly record struct RecognizedSegment(double Start, double End, string Text);

/// <summary>A transcription as the speech recognizer checks see it: the whole text and its segments in order.</summary>
/// <param name="Text">The text.</param>
/// <param name="Segments">The segments, in the order they were heard.</param>
public sealed record RecognizedSpeech(string Text, IReadOnlyList<RecognizedSegment> Segments);

/// <summary>
/// A speech recognizer under test (Idrak.Audio's <c>ISpeechRecognizer</c>, or one of your own), given as what the suite
/// needs to drive it without depending on the package: the rate it takes mono samples at, and its transcription of them.
/// </summary>
/// <param name="SampleRate">The rate of the samples the suite makes.</param>
/// <param name="Transcribe">The transcription of mono samples in [-1, 1] at <paramref name="SampleRate"/>.</param>
public sealed record SpeechRecognizerUnderTest(int SampleRate, Func<float[], RecognizedSpeech> Transcribe);

/// <summary>
/// The checks of a speech recognizer on sounds the suite makes (silence, a tone, seeded noise, a sound of a few
/// milliseconds and an empty one, at fixed and random lengths), checks that hold for any recognizer, trained or not:
/// <list type="bullet">
/// <item>it transcribes without throwing, every segment starting within the sound and ending no earlier than it starts,
/// the segments in order;</item>
/// <item>the text holds each segment's text, in order;</item>
/// <item>the same text on a second call, and from two calls at once (a recognizer shared by threads);</item>
/// <item>with <paramref name="silenceMaxChars"/> given (a trained model), silence gives at most that many characters.</item>
/// </list>
/// </summary>
/// <param name="silenceMaxChars">The most characters silence may give, or null to skip that check (an untrained model says anything).</param>
public sealed class SpeechRecognizerSuite(int? silenceMaxChars = null) : ContractSuite<SpeechRecognizerUnderTest>
{
    /// <inheritdoc />
    public override string Name => "speech-recognizer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("silence", 0, 1.0, 0);
        yield return Case("tone", 1, 1.0, 3);
        yield return Case("noise", 2, 0.5, 7);
        yield return Case("short", 2, 0.01, 11);
        yield return Case("empty", 0, 0, 0);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int kind = random.Next(3);
        double seconds = Math.Round(0.05 + random.NextDouble() * (large ? 40 : 2), 2);
        return Case(kind switch { 0 => "silence", 1 => "tone", _ => "noise" }, kind, seconds, random.Next(1, 1000));
    }

    /// <inheritdoc />
    public override void Run(SpeechRecognizerUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int kind = (int)@case.Data["kind"]!, seed = (int)@case.Data["seed"]!;
        double seconds = (double)@case.Data["seconds"]!;
        int rate = implementation.SampleRate;
        var samples = new float[(int)Math.Round(seconds * rate)];
        var random = new Random(seed);
        for (int n = 0; n < samples.Length; n++)
        {
            samples[n] = kind switch
            {
                1 => 0.5f * MathF.Sin(2 * MathF.PI * (200 + seed) * n / rate),
                2 => (float)(random.NextDouble() * 2 - 1) * 0.5f,
                _ => 0f,
            };
        }

        RecognizedSpeech speech;
        try
        {
            speech = implementation.Transcribe(samples);
        }
        catch (Exception e)
        {
            checks.Check("transcribes without throwing", false, $"{e.GetType().Name}: {e.Message}");
            return;
        }

        checks.Check("transcribes without throwing", speech is not null, "the transcription is null");
        if (speech is null)
        {
            return;
        }

        double duration = (double)samples.Length / rate;
        int bad = -1;
        for (int i = 0; i < speech.Segments.Count && bad < 0; i++)
        {
            var s = speech.Segments[i];
            if (s.Start < 0 || s.Start > duration + 1e-6 || s.End < s.Start || (i > 0 && s.Start < speech.Segments[i - 1].Start))
            {
                bad = i;
            }
        }

        checks.Check("segments in order, each starting within the sound and ending no earlier than it starts", bad < 0,
            bad < 0 ? "" : $"segment {bad} runs {speech.Segments[bad].Start:F3} to {speech.Segments[bad].End:F3} s in a sound of {duration:F3} s");
        int at = 0, missing = -1;
        for (int i = 0; i < speech.Segments.Count && missing < 0; i++)
        {
            string piece = speech.Segments[i].Text.Trim();
            if (piece.Length == 0)
            {
                continue;
            }

            int found = speech.Text.IndexOf(piece, at, StringComparison.Ordinal);
            if (found < 0)
            {
                missing = i;
            }
            else
            {
                at = found + piece.Length;
            }
        }

        checks.Check("the text holds each segment's text, in order", missing < 0, missing < 0 ? "" : $"segment {missing}'s \"{speech.Segments[missing].Text}\" is not in \"{speech.Text}\" after position {at}");
        checks.Compare("the same text on a second call", Comparisons.Difference(speech.Text, implementation.Transcribe(samples).Text));
        var both = new RecognizedSpeech?[2];
        Parallel.For(0, 2, i => both[i] = implementation.Transcribe(samples));
        checks.Compare("the same text from two calls at once", Comparisons.Difference(speech.Text + "\n" + speech.Text, both[0]!.Text + "\n" + both[1]!.Text));
        if (kind == 0 && silenceMaxChars is { } most)
        {
            checks.Check("silence gives little text", speech.Text.Trim().Length <= most, $"silence gave \"{speech.Text}\" ({speech.Text.Trim().Length} characters, at most {most})");
        }
        else
        {
            checks.Skip("silence gives little text", silenceMaxChars is null ? "no bound given (an untrained model says anything)" : "not a silent case");
        }
    }

    private static ContractCase Case(string name, int kind, double seconds, int seed) =>
        new($"{name}, {seconds} s{(kind > 0 ? ", seed " + seed : "")}", new JsonObject { ["kind"] = kind, ["seconds"] = seconds, ["seed"] = seed });
}
