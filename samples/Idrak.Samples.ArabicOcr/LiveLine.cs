// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// One status line redrawn in place on the console's error stream (four times a second at most), cleared before an
/// ordinary line is printed. When the error stream goes to a file, the status is a plain line once a minute instead;
/// when the app writes somewhere other than the console (its tests), nothing is shown.
/// </summary>
internal sealed class LiveLine(TextWriter error)
{
    private readonly bool _console = ReferenceEquals(error, Console.Error);
    private readonly bool _interactive = ReferenceEquals(error, Console.Error) && !Console.IsErrorRedirected;
    private readonly Stopwatch _sinceDraw = Stopwatch.StartNew();
    private bool _drawn;
    private int _shown;

    /// <summary>Redraws the line (when it is time to, or always with <paramref name="now"/>).</summary>
    public void Show(Func<string> text, bool now = false)
    {
        if (!_console || (!now && _drawn && _sinceDraw.ElapsedMilliseconds < (_interactive ? 250 : 60_000)))
        {
            return;
        }

        _sinceDraw.Restart();
        _drawn = true;
        string line = text();
        if (!_interactive)
        {
            error.WriteLine(line);
            error.Flush();
            return;
        }

        int width = Width();
        if (line.Length > width)
        {
            line = line[..width];
        }

        error.Write("\r" + line.PadRight(_shown));
        error.Flush();
        _shown = line.Length;
    }

    /// <summary>Erases the line, so the next output starts on a clean one.</summary>
    public void Clear()
    {
        if (_interactive && _shown > 0)
        {
            error.Write("\r" + new string(' ', _shown) + "\r");
            error.Flush();
            _shown = 0;
        }

        _drawn = false;
    }

    /// <summary>A bar of <paramref name="cells"/> cells, filled to <paramref name="fraction"/>.</summary>
    public static string Bar(double fraction, int cells = 20)
    {
        int filled = (int)Math.Round(Math.Clamp(fraction, 0, 1) * cells);
        return "[" + new string('#', filled) + new string('-', cells - filled) + "]";
    }

    /// <summary>A duration as h:mm:ss (or m:ss under an hour).</summary>
    public static string Time(TimeSpan time) =>
        time.TotalHours >= 1 ? $"{(int)time.TotalHours}:{time.Minutes:D2}:{time.Seconds:D2}" : $"{time.Minutes}:{time.Seconds:D2}";

    /// <summary>Bytes as KB, MB or GB.</summary>
    public static string Bytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):F0} MB",
        _ => $"{bytes / 1024.0:F0} KB",
    };

    private static int Width()
    {
        try
        {
            return Math.Max(40, Console.WindowWidth - 1);
        }
        catch (IOException)
        {
            return 159;
        }
    }
}
