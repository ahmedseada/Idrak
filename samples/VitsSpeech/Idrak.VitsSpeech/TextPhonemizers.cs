// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.VitsSpeech;

/// <summary>
/// Turns text into the phonemes a checkpoint was trained on: VITS checkpoints that set <c>phonemize</c> in
/// tokenizer_config.json (VCTK's, LJSpeech's) read IPA from espeak, as transformers' <c>VitsTokenizer</c> gets it from
/// the <c>phonemizer</c> package, so <see cref="VitsTokenizer"/> phonemizes such text before tokenizing. Registered in
/// <see cref="TextPhonemizers"/>.
/// </summary>
public interface ITextPhonemizer
{
    /// <summary>The name it is registered under ("espeak-ng").</summary>
    string Name { get; }

    /// <summary>
    /// The phonemes of <paramref name="text"/> in <paramref name="language"/> (an espeak voice name such as "en-us"):
    /// IPA with stress marks, words separated by a space, punctuation kept where it was.
    /// </summary>
    /// <exception cref="NotSupportedException">The phonemizer cannot run here (the message says what it needs).</exception>
    string Phonemize(string text, string language);
}

/// <summary>
/// The phonemizers, by name (ordinal). Built in: <see cref="EspeakNg"/> (the espeak-ng program, which the user
/// installs: the plug-in runs it, the library carries none). Register another (a pure .NET G2P, a service) with
/// <see cref="Register"/>; one under a built-in name shadows it.
/// </summary>
public static class TextPhonemizers
{
    /// <summary>The espeak-ng program (<see cref="EspeakNgPhonemizer"/>).</summary>
    public const string EspeakNg = "espeak-ng";

    /// <summary>The punctuation a phonemizer keeps in place (the phonemizer package's default marks).</summary>
    public const string Marks = ";:,.!?¡¿—…\"«»“”(){}[]";

    private static readonly SlotTable<string, ITextPhonemizer> Table = BuiltIn();

    private static SlotTable<string, ITextPhonemizer> BuiltIn()
    {
        var table = new SlotTable<string, ITextPhonemizer>(nameof(TextPhonemizers), comparer: StringComparer.Ordinal, newestFirst: true);
        table.RegisterDefault(EspeakNg, new EspeakNgPhonemizer());
        return table;
    }

    /// <summary>Registers <paramref name="phonemizer"/> under its <see cref="ITextPhonemizer.Name"/> (asked before the earlier ones by <see cref="First"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextPhonemizer phonemizer)
    {
        ArgumentNullException.ThrowIfNull(phonemizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(phonemizer.Name);
        Table.Register(phonemizer.Name, phonemizer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's phonemizer <paramref name="name"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, the most recently registered first.</summary>
    public static IReadOnlyList<string> Names => Table.Keys;

    /// <summary>The phonemizer <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names the registered ones.</exception>
    public static ITextPhonemizer Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Phonemizer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TextPhonemizers.Register.");

    /// <summary>The most recently registered phonemizer (the built-in espeak-ng when an app registered none).</summary>
    public static ITextPhonemizer First() => Get(Table.Keys[0]);

    /// <summary>Who registered the phonemizer <paramref name="name"/>: <see cref="Overrides.Library"/> for the built-in or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// <paramref name="text"/> phonemized a stretch at a time by <paramref name="phonemize"/> between the
    /// <paramref name="marks"/> (punctuation), the marks kept in place and followed by a space where the text had one,
    /// as the phonemizer package's <c>preserve_punctuation</c> does; runs of white space become one space, and the ends
    /// are trimmed.
    /// </summary>
    public static string KeepingPunctuation(string text, Func<string, string> phonemize, string marks = Marks)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(phonemize);
        var result = new StringBuilder(text.Length * 2);
        int start = 0;
        for (int i = 0; i <= text.Length; i++)
        {
            if (i < text.Length && marks.IndexOf(text[i], StringComparison.Ordinal) < 0)
            {
                continue;
            }

            string stretch = text[start..i].Trim();
            if (stretch.Length > 0)
            {
                result.Append(phonemize(stretch).Trim());
            }

            if (i < text.Length)
            {
                if (result.Length > 0 && result[^1] == ' ')
                {
                    result.Length--;
                }

                result.Append(text[i]);
                if (i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))
                {
                    result.Append(' ');
                }
            }

            start = i + 1;
        }

        return Collapse(result.ToString());
    }

    // Runs of white space as one space, the ends trimmed.
    internal static string Collapse(string text)
    {
        var result = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = result.Length > 0;
                continue;
            }

            if (space)
            {
                result.Append(' ');
                space = false;
            }

            result.Append(c);
        }

        return result.ToString();
    }
}

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
