// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui;

/// <summary>
/// The library's <see cref="IDictation"/>: the Transcribe lab's services (<see cref="TranscribeViewModel"/>, the app's one):
/// its speech model (the one picked there, loaded once in the model service's "speech" host and shared with the lab), its
/// language, cleaning pipeline and live settings, the microphone's permission. The lab's own transcript is left as it is,
/// and the lab does not record or read a file while a dictation runs (one reading of the model at a time).
/// </summary>
public sealed class TranscribeDictation(TranscribeViewModel transcribe) : IDictation
{
    private readonly TranscribeViewModel _transcribe = transcribe ?? throw new ArgumentNullException(nameof(transcribe));

    /// <inheritdoc />
    public bool IsDictating => _transcribe.IsDictating;

    /// <inheritdoc />
    public Task<DictationOutcome> DictateAsync(Action<string> heard, CancellationToken cancellationToken = default) => _transcribe.DictateAsync(heard, cancellationToken);

    /// <inheritdoc />
    public void Stop() => _transcribe.StopDictation();
}
