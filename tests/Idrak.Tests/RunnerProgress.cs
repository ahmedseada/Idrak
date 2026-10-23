// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Diagnostics;

// The runner's --progress and --failures-only: the library's progress line (ProgressReporters) with a bar, and the tests'
// own lines kept off it.
internal static partial class Tests
{
    private const int BarWidth = 24;

    // A runner switch from its variable: 1 or true on, 0, false or unset off; anything else is refused naming it.
    internal static bool RunnerFlag(string variable) => Environment.GetEnvironmentVariable(variable) switch
    {
        null or "" or "0" or "false" or "False" => false,
        "1" or "true" or "True" => true,
        var other => throw new InvalidOperationException($"{variable}={other} is not 1, true, 0 or false."),
    };

    // "tests [#########...............]": the share done as a bar (the reporter adds the share, the count, the rate and the time left).
    internal static string ProgressLabel(long done, long total)
    {
        int filled = total > 0 ? (int)Math.Min(BarWidth, done * BarWidth / total) : 0;
        return $"tests [{new string('#', filled)}{new string('.', BarWidth - filled)}]";
    }

    // The tests' own lines while the progress line shows: a line starting takes the progress line off the screen first
    // (under the lock the reporter draws under), and a line ended shows it again, so neither lands inside the other.
    internal sealed class ProgressAwareWriter(TextWriter inner, TextWriter screen, IProgressReporter bar) : TextWriter
    {
        private bool _lineStart = true;

        public override Encoding Encoding => inner.Encoding;

        public override void Write(char value) => Write(value.ToString());

        public override void Write(char[] buffer, int index, int count) => Write(new string(buffer, index, count));

        public override void Write(string? value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return;
            }

            lock (screen)
            {
                if (_lineStart)
                {
                    bar.Erase();
                }

                inner.Write(value);
                _lineStart = value[^1] == '\n';
                if (_lineStart)
                {
                    inner.Flush();
                    bar.Advance(0);
                }
            }
        }

        public override void WriteLine() => Write(NewLine);

        public override void WriteLine(string? value) => Write(value + NewLine);

        public override void Flush() => inner.Flush();
    }
}
