// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>How live sound is transcribed (<c>LiveTranscriber</c>): when it is read again, when a segment ends, what is quiet.</summary>
public sealed record LiveTranscriptionOptions
{
    /// <summary>How each step transcribes (language, task, beams, the guards); its <see cref="TranscriptionOptions.Prompt"/> starts the text carried between segments.</summary>
    public TranscriptionOptions Transcription { get; init; } = new();

    /// <summary>Seconds of new sound before the segment is transcribed again: the delay of a word on the screen, at the cost of one transcription each.</summary>
    public double StepSeconds { get; init; } = 1.0;

    /// <summary>The longest segment in seconds: one this long is ended even while the speaker goes on.</summary>
    public double WindowSeconds { get; init; } = 20;

    /// <summary>Seconds of quiet after speech that end a segment.</summary>
    public double SilenceSeconds { get; init; } = 0.6;

    /// <summary>A 20 ms frame is quiet this many decibels under the loudest frame heard so far.</summary>
    public double SilenceDecibels { get; init; } = 35;

    /// <summary>A frame under this level (decibels of full scale) is quiet whatever was heard before (a room before anyone speaks).</summary>
    public double SilenceFloorDecibels { get; init; } = -50;

    /// <summary>
    /// Seconds of loud 20 ms frames a segment needs before it is transcribed: a click, a breath or a keypress under it is
    /// left out unread (a recognizer given almost no sound echoes its prompt, the text before).
    /// </summary>
    public double MinSpeechSeconds { get; init; } = 0.3;

    /// <summary>Seconds of the quiet before speech kept with it (the start of a word under the threshold).</summary>
    public double PreRollSeconds { get; init; } = 0.3;

    /// <summary>The commit policy (<see cref="LiveCommitPolicies"/>): which words of a segment are final before it ends.</summary>
    public string Policy { get; init; } = LiveCommitPolicies.Agreement;

    /// <summary>How many transcriptions in a row a word must keep (the "agreement" policy's count; 2, Whisper-streaming's local agreement).</summary>
    public int Agreement { get; init; } = 2;

    /// <summary>
    /// True (live sound, the default): every block waiting is read before a step, so after a step that took longer than
    /// the step the next reads all that arrived meanwhile and the transcription keeps up. False (a recording given faster
    /// than it plays): a step every <see cref="StepSeconds"/> of sound, as if it came at its own pace.
    /// </summary>
    public bool KeepUp { get; init; } = true;

    /// <summary>The characters of the finished text given to the next segment as its prompt (0: none).</summary>
    public int PromptCharacters { get; init; } = 200;

    /// <summary>Throws when an option is out of range.</summary>
    /// <exception cref="ArgumentException">A length or count out of range.</exception>
    public void Check()
    {
        if (!(StepSeconds > 0) || !(WindowSeconds >= StepSeconds) || !(SilenceSeconds > 0) || !(SilenceDecibels > 0) || !(SilenceFloorDecibels < 0)
            || !(PreRollSeconds >= 0) || !(MinSpeechSeconds >= 0) || Agreement < 1 || PromptCharacters < 0 || string.IsNullOrWhiteSpace(Policy))
        {
            throw new ArgumentException($"Live transcription: StepSeconds ({StepSeconds}), SilenceSeconds ({SilenceSeconds}) and SilenceDecibels ({SilenceDecibels}) are positive, "
                + $"WindowSeconds ({WindowSeconds}) at least StepSeconds, SilenceFloorDecibels ({SilenceFloorDecibels}) under 0, PreRollSeconds ({PreRollSeconds}), "
                + $"MinSpeechSeconds ({MinSpeechSeconds}) and PromptCharacters ({PromptCharacters}) not negative, Agreement ({Agreement}) at least 1, a Policy named.");
        }
    }
}

/// <summary>
/// One update of a live transcription: the words made final by this step, the words after them that may still change,
/// and, when a segment ended, its whole text.
/// </summary>
/// <param name="Committed">Words final from this step on (append them to what came before); empty when none.</param>
/// <param name="Tentative">The words heard after the committed ones, which the next step may change; empty at a segment's end.</param>
/// <param name="Segment">The segment's whole text when it ended with this step (quiet, the window, the end of the sound), else null.</param>
/// <param name="Start">Seconds from the sound's start to the segment's start.</param>
/// <param name="End">Seconds from the sound's start to the end of what this step read.</param>
/// <param name="Elapsed">How long this step's transcription took.</param>
/// <param name="Behind">Seconds of sound that arrived during the step and wait for the next: how far behind the speaker it runs.</param>
public sealed record LiveTranscript(string Committed, string Tentative, string? Segment, double Start, double End, TimeSpan Elapsed, double Behind)
{
    /// <summary>Whether a segment ended with this update.</summary>
    public bool SegmentEnded => Segment is not null;
}

/// <summary>
/// Decides which leading words of a segment are final while it is still being heard, from its transcriptions so far
/// (registered in <see cref="LiveCommitPolicies"/>).
/// </summary>
public interface ILiveCommitPolicy
{
    /// <summary>The name it is registered under ("agreement").</summary>
    string Name { get; }

    /// <summary>
    /// How many leading words of the latest transcription (the last of <paramref name="transcriptions"/>, oldest first, each
    /// its words) are final; <paramref name="options"/> gives the policy's settings.
    /// </summary>
    int Stable(IReadOnlyList<IReadOnlyList<string>> transcriptions, LiveTranscriptionOptions options);
}

/// <summary>
/// The live commit policies, by name (ignoring case). Built in (Idrak.Audio): "agreement" (the words the last
/// <see cref="LiveTranscriptionOptions.Agreement"/> transcriptions agree on from the start: Whisper-streaming's local
/// agreement). Register another with <see cref="Register"/>.
/// </summary>
public static class LiveCommitPolicies
{
    /// <summary>"agreement".</summary>
    public const string Agreement = "agreement";

    private static readonly SlotTable<string, ILiveCommitPolicy> Table = BuiltIn();

    private static SlotTable<string, ILiveCommitPolicy> BuiltIn()
    {
        var table = new SlotTable<string, ILiveCommitPolicy>(nameof(LiveCommitPolicies), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a policy's choice of final words is its own, so another cannot stand in for it");
        table.RegisterDefault(Agreement, new Idrak.Audio.AgreementPolicy());
        return table;
    }

    /// <summary>Registers <paramref name="policy"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ILiveCommitPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentException.ThrowIfNullOrWhiteSpace(policy.Name);
        Table.Register(policy.Name, policy, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's policy of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The policy of that name, or null.</summary>
    public static ILiveCommitPolicy? Find(string name) => Table.Find(name);

    /// <summary>The policy of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ILiveCommitPolicy Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(LiveCommitPolicies), name, Table.Keys,
            $"Live commit policy '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with LiveCommitPolicies.Register.");

    /// <summary>The library's policy of that name, or null.</summary>
    public static ILiveCommitPolicy? Default(string name) => Table.Default(name);

    /// <summary>Who registered the policy of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);
}
