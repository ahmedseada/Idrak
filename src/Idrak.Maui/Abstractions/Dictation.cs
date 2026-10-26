// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>How a dictation ended, or why it did not start.</summary>
/// <param name="Text">What was heard, the whole text ("" when nothing was).</param>
/// <param name="Completed">Whether it ran and ended by <see cref="IDictation.Stop"/> or the sound's end, its last words read.</param>
/// <param name="Problem">Why it did not start or stopped early, or "".</param>
/// <param name="NeedsModel">Whether a speech model is needed first (the caller offers Models).</param>
/// <param name="PermissionRefused">Whether the microphone's permission was refused (the caller offers the app's settings).</param>
public sealed record DictationOutcome(string Text, bool Completed, string Problem, bool NeedsModel = false, bool PermissionRefused = false)
{
    /// <summary>Whether there is a problem to show.</summary>
    public bool HasProblem => Problem.Length > 0;
}

/// <summary>
/// Speech typed into a text box: the microphone transcribed live, the whole text heard so far handed on as it grows (the
/// finished words and the tentative ones after them), until <see cref="Stop"/> finishes the last words. The library's
/// dictation is the Transcribe lab's (its speech model, language, cleaning and live settings; <c>UseIdrak</c> registers
/// it); an app registers its own instead.
/// </summary>
public interface IDictation
{
    /// <summary>Whether a dictation runs.</summary>
    bool IsDictating { get; }

    /// <summary>
    /// Dictates until <see cref="Stop"/> (its last words then read) or <paramref name="cancellationToken"/> (at once):
    /// <paramref name="heard"/> gets the whole text so far each time it changes, from a worker thread. Returns at its end, or
    /// at once with why it cannot start (no speech model ready, the microphone refused, a dictation already running).
    /// </summary>
    Task<DictationOutcome> DictateAsync(Action<string> heard, CancellationToken cancellationToken = default);

    /// <summary>Stops the microphone; the words heard last are read, then <see cref="DictateAsync"/> returns.</summary>
    void Stop();
}
