// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>How a synthesizer cuts a text into the pieces it speaks one by one (<see cref="SpeechSynthesisOptions.MaxCharacters"/>).</summary>
public static class SpeechSynthesisText
{
    /// <summary>
    /// The pieces <paramref name="text"/> is spoken in: the whole text when <paramref name="maxCharacters"/> is 0,
    /// otherwise sentences (split at . ! ? ؟ ۔ 。 and line breaks) joined up to that many characters; a sentence longer
    /// than the limit is cut at its last space before it.
    /// </summary>
    public static IReadOnlyList<string> Pieces(string text, int maxCharacters)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (maxCharacters <= 0)
        {
            return [text];
        }

        var sentences = new List<string>();
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] is '.' or '!' or '?' or '؟' or '۔' or '。' or '\n')
            {
                string sentence = text[start..(i + 1)].Trim();
                if (sentence.Length > 0)
                {
                    sentences.Add(sentence);
                }

                start = i + 1;
            }
        }

        if (start < text.Length && text[start..].Trim() is { Length: > 0 } rest)
        {
            sentences.Add(rest);
        }

        var pieces = new List<string>();
        var current = "";
        foreach (var sentence in sentences)
        {
            foreach (var part in CutLong(sentence, maxCharacters))
            {
                if (current.Length == 0)
                {
                    current = part;
                }
                else if (current.Length + 1 + part.Length <= maxCharacters)
                {
                    current = current + " " + part;
                }
                else
                {
                    pieces.Add(current);
                    current = part;
                }
            }
        }

        if (current.Length > 0)
        {
            pieces.Add(current);
        }

        return pieces;
    }

    private static IEnumerable<string> CutLong(string sentence, int maxCharacters)
    {
        while (sentence.Length > maxCharacters)
        {
            int cut = sentence.LastIndexOf(' ', maxCharacters - 1);
            if (cut <= 0)
            {
                cut = maxCharacters;
            }

            yield return sentence[..cut].Trim();
            sentence = sentence[cut..].Trim();
        }

        if (sentence.Length > 0)
        {
            yield return sentence;
        }
    }
}
