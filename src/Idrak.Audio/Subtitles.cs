// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Audio;

/// <summary>A transcription's segments as subtitles: SubRip (.srt) or WebVTT (.vtt); segments without text are left out.</summary>
public static class Subtitles
{
    /// <summary>SubRip: numbered cues, times as HH:MM:SS,mmm.</summary>
    public static string Srt(Transcription transcription) => Render(transcription, vtt: false);

    /// <summary>WebVTT: a WEBVTT header, times as HH:MM:SS.mmm.</summary>
    public static string Vtt(Transcription transcription) => Render(transcription, vtt: true);

    /// <summary>The segments as SubRip (numbered cues, HH:MM:SS,mmm) or WebVTT (a WEBVTT header, HH:MM:SS.mmm); segments without text are left out.</summary>
    public static string Render(Transcription transcription, bool vtt)
    {
        ArgumentNullException.ThrowIfNull(transcription);
        var text = new StringBuilder(vtt ? "WEBVTT\n\n" : "");
        int cue = 0;
        foreach (var segment in transcription.Segments)
        {
            string words = segment.Text.Trim();
            if (words.Length == 0)
            {
                continue;
            }

            if (!vtt)
            {
                text.Append(++cue).Append('\n');
            }

            text.Append(Timestamp(segment.Start, vtt)).Append(" --> ").Append(Timestamp(Math.Max(segment.Start, segment.End), vtt)).Append('\n').Append(words).Append("\n\n");
        }

        return text.ToString();
    }

    /// <summary><paramref name="seconds"/> as HH:MM:SS,mmm (SubRip) or HH:MM:SS.mmm (WebVTT); never below zero.</summary>
    public static string Timestamp(double seconds, bool vtt)
    {
        long ms = (long)Math.Round(Math.Max(0, seconds) * 1000);
        return string.Create(CultureInfo.InvariantCulture, $"{ms / 3_600_000:00}:{ms / 60_000 % 60:00}:{ms / 1000 % 60:00}{(vtt ? '.' : ',')}{ms % 1000:000}");
    }
}
