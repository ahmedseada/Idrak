// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;

namespace Idrak.Audio;

/// <summary>
/// The espeak-ng program as a phonemizer: <c>espeak-ng -q --ipa -v LANGUAGE</c> on each stretch of text between
/// punctuation (<see cref="TextPhonemizers.KeepingPunctuation"/>), the text on its standard input. The program is the
/// user's: <see cref="Executable"/> (the <c>IDRAK_ESPEAK</c> variable, else <c>espeak-ng</c> on the PATH); the voice
/// is the checkpoint's language unless <c>IDRAK_ESPEAK_VOICE</c> names another. Each stretch starts the program once.
/// </summary>
public sealed class EspeakNgPhonemizer : ITextPhonemizer
{
    /// <summary>The variable naming the program (a full path), read when it runs.</summary>
    public const string ExecutableVariable = "IDRAK_ESPEAK";

    /// <summary>The variable naming the espeak voice to use instead of the checkpoint's language.</summary>
    public const string VoiceVariable = "IDRAK_ESPEAK_VOICE";

    /// <inheritdoc />
    public string Name => TextPhonemizers.EspeakNg;

    /// <summary>The program: <see cref="ExecutableVariable"/>'s value, else <c>espeak-ng</c> found on the PATH.</summary>
    public string Executable { get; init; } = "";

    /// <summary>How long one run may take before it is stopped.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <inheritdoc />
    public string Phonemize(string text, string language)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentException.ThrowIfNullOrWhiteSpace(language);
        string program = Executable.Length > 0 ? Executable : Environment.GetEnvironmentVariable(ExecutableVariable) is { Length: > 0 } given ? given : "espeak-ng";
        string voice = Environment.GetEnvironmentVariable(VoiceVariable) is { Length: > 0 } other ? other : language;
        return TextPhonemizers.KeepingPunctuation(text, stretch => Run(program, voice, stretch));
    }

    private string Run(string program, string voice, string text)
    {
        var start = new ProcessStartInfo(program)
        {
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            StandardInputEncoding = new UTF8Encoding(false), StandardOutputEncoding = Encoding.UTF8,
        };
        foreach (string argument in (string[])["-q", "--ipa", "-b", "1", "-v", voice])
        {
            start.ArgumentList.Add(argument);
        }

        Process process;
        try
        {
            process = Process.Start(start) ?? throw new NotSupportedException($"{program} did not start.");
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            throw new NotSupportedException($"This checkpoint reads phonemes, made here by the espeak-ng program, which was not found ('{program}': {e.Message}). "
                + $"Install espeak-ng and put it on the PATH or set {ExecutableVariable} to its full path, or register another phonemizer with TextPhonemizers.Register.", e);
        }

        using (process)
        {
            process.StandardInput.Write(text);
            process.StandardInput.Close();
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(Timeout))
            {
                process.Kill(entireProcessTree: true);
                throw new TimeoutException($"{program} did not finish within {Timeout.TotalSeconds:0} s.");
            }

            if (process.ExitCode != 0)
            {
                throw new NotSupportedException($"{program} -v {voice} failed ({process.ExitCode}): {error.Result.Trim()} (an unknown voice? {VoiceVariable} names another).");
            }

            return TextPhonemizers.Collapse(output.Result.Replace('\n', ' ').Replace('\r', ' '));
        }
    }
}
