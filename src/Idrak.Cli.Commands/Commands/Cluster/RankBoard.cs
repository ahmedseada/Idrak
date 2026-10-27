// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// The live view of a run of rank processes (plan 23, phase C5; CLAUDE.md: long work shows what it does, how far, how fast
/// and the time left): in a terminal, one line per rank redrawn in place on the error output (its device, whether it runs,
/// its time, how many lines it wrote and how fast, and its last line, which carries the command's own progress and time
/// left) under a heading with the ranks running and the time so far; lines the command passes on (rank 0's output) print
/// above the board. Without a terminal (a pipe, a log, <c>--plain</c>), every line prints as it comes, "rank N: …".
/// </summary>
internal sealed class RankBoard : IDisposable
{
    private readonly CommandContext _context;
    private readonly string _what;
    private readonly int[] _ranks;
    private readonly string[] _devices;
    private readonly long[] _lines;
    private readonly string?[] _last;
    private readonly int?[] _ended;
    private readonly Func<LocalhostLine, bool> _passOn;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private readonly Lock _gate = new();
    private readonly Timer? _timer;
    private int _drawn;
    private bool _finished;

    /// <param name="context">The command.</param>
    /// <param name="what">What runs ("pretrain on node desk").</param>
    /// <param name="ranks">The ranks shown, in order.</param>
    /// <param name="devices">Each shown rank's device, in the same order.</param>
    /// <param name="passOn">Lines printed in full (above the board in a terminal) rather than only shown as a rank's last line.</param>
    public RankBoard(CommandContext context, string what, IReadOnlyList<int> ranks, IReadOnlyList<string> devices, Func<LocalhostLine, bool> passOn)
    {
        _context = context;
        _what = what;
        _ranks = [.. ranks];
        _devices = [.. devices];
        _lines = new long[_ranks.Length];
        _last = new string?[_ranks.Length];
        _ended = new int?[_ranks.Length];
        _passOn = passOn;
        Live = !context.Quiet && !context.Json && !context.Plain && Terminal.IsTerminal(context.ErrorOutput);
        if (Live)
        {
            _timer = new Timer(_ => Draw(), null, TimeSpan.Zero, TimeSpan.FromMilliseconds(250));
        }
    }

    /// <summary>Whether the board is drawn in place (a terminal) rather than printed line by line.</summary>
    public bool Live { get; }

    /// <summary>A line from a rank (called from the processes' reading threads).</summary>
    public void Line(LocalhostLine line)
    {
        int at = Array.IndexOf(_ranks, line.Rank);
        lock (_gate)
        {
            if (at >= 0)
            {
                _lines[at]++;
                _last[at] = line.Text;
            }

            if (_finished)
            {
                return;
            }

            bool passOn = _passOn(line);
            if (Live)
            {
                if (passOn)
                {
                    Erase();
                    (line.IsError ? _context.ErrorOutput : _context.Output).WriteLine(line.Text);
                    Paint();
                }

                return;
            }

            if (passOn)
            {
                (line.IsError ? _context.ErrorOutput : _context.Output).WriteLine(line.Text);
            }
            else if (!_context.Quiet && !_context.Json)
            {
                (line.IsError ? _context.ErrorOutput : _context.Output).WriteLine($"rank {line.Rank}: {line.Text}");
            }
        }
    }

    /// <summary>Marks the ranks' ends and takes the board down, leaving the last state when it was live.</summary>
    public void Finish(IEnumerable<LocalhostRankResult> results)
    {
        lock (_gate)
        {
            foreach (var result in results)
            {
                int at = Array.IndexOf(_ranks, result.Rank);
                if (at >= 0)
                {
                    _ended[at] = result.ExitCode ?? -1;
                }
            }

            if (Live && !_finished)
            {
                Erase();
                Paint();
            }

            _finished = true;
        }
    }

    public void Dispose()
    {
        _timer?.Dispose();
        lock (_gate)
        {
            _finished = true;
        }
    }

    private void Draw()
    {
        lock (_gate)
        {
            if (!_finished)
            {
                Erase();
                Paint();
            }
        }
    }

    // Moves back over the drawn board and clears it.
    private void Erase()
    {
        if (_drawn == 0)
        {
            return;
        }

        var error = _context.ErrorOutput;
        error.Write($"\u001b[{_drawn}A\r\u001b[J");
        _drawn = 0;
    }

    private void Paint()
    {
        var error = _context.ErrorOutput;
        int width = Width();
        var elapsed = _watch.Elapsed;
        int running = _ended.Count(e => e is null);
        error.WriteLine(Fit($"{_what}: {running} of {_ranks.Length} rank(s) running, {Duration(elapsed)}", width));
        for (int i = 0; i < _ranks.Length; i++)
        {
            string state = _ended[i] switch { null => "running", 0 => "done", -1 => "ended", { } code => $"exit {code}" };
            double rate = _lines[i] / Math.Max(elapsed.TotalSeconds, 1e-3);
            string text = string.Create(CultureInfo.InvariantCulture, $"  rank {_ranks[i]} {_devices[i]}  {state}  {_lines[i]} lines ({rate:F1}/s)  {_last[i] ?? "starting"}");
            error.WriteLine(Fit(text, width));
        }

        error.Flush();
        _drawn = _ranks.Length + 1;
    }

    private static string Fit(string text, int width) => text.Length < width ? text : text[..Math.Max(1, width - 1)];

    private static int Width()
    {
        try
        {
            return Console.WindowWidth is > 20 and var w ? w : 120;
        }
        catch (IOException)
        {
            return 120;
        }
    }

    private static string Duration(TimeSpan time) => time.TotalHours >= 1
        ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours}h {time.Minutes}m")
        : time.TotalMinutes >= 1 ? string.Create(CultureInfo.InvariantCulture, $"{time.Minutes}m {time.Seconds}s")
        : string.Create(CultureInfo.InvariantCulture, $"{time.TotalSeconds:F1} s");
}
