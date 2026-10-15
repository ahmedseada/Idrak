// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Models;

namespace Idrak.Cli.Shared;

/// <summary>
/// Sounds given to an audio-language model on the command line (<c>run --audio</c>, <c>chat --audio</c>), beside
/// <see cref="ImageInputs"/>: the files as chat audio parts (<see cref="ChatAudio"/>, the bytes as they are: the model's
/// audio family decodes them), and how a chat generator reads them (<see cref="ChatAudioParts"/> over the model's audio
/// part, which its family reads: <see cref="AudioLanguage.Attach"/>, a plug-in given with <c>-P</c>).
/// </summary>
internal static class AudioInputs
{
    /// <summary>The files as chat audio parts; a usage error names a file that does not exist.</summary>
    public static IReadOnlyList<ChatAudio> Read(IEnumerable<string> paths)
    {
        var sounds = new List<ChatAudio>();
        foreach (string path in paths)
        {
            sounds.Add(File.Exists(path) ? ChatAudio.FromFile(path) : throw new UsageException($"Sound file not found: {path}"));
        }

        return sounds;
    }

    /// <summary>
    /// How a chat generator reads <paramref name="model"/>'s sounds (a source for <c>ChatGenerator.Parts</c>, disposed by
    /// the caller), or null with the reason when it reads none: a model without <c>audio_config</c> (no reason), or one
    /// whose audio family is not registered (the registry's message and the <c>-P</c> remedy).
    /// </summary>
    public static ChatAudioParts? For(PretrainedModel model, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(model);
        reason = null;
        try
        {
            return AudioLanguage.Attach(model) is { } audio ? new ChatAudioParts(audio) : null;
        }
        catch (NotSupportedException e)
        {
            reason = Messages.T("{0} Load its plug-in with -P (--plugin).", e.Message);
            return null;
        }
    }
}
