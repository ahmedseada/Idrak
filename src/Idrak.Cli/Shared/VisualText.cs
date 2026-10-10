// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Cli.Shared;

/// <summary>How right-to-left text reaches the terminal (<c>--lang-render</c>).</summary>
internal enum TextRendering
{
    /// <summary>Visual on a terminal that does not reorder text itself, logical otherwise (<see cref="VisualText.Detect"/>).</summary>
    Auto,

    /// <summary>Shaped and reordered by the tool, for terminals that draw characters in the order they arrive.</summary>
    Visual,

    /// <summary>As stored (logical order, unshaped), for terminals that apply the bidirectional algorithm, and for files and pipes.</summary>
    Logical,
}

/// <summary>
/// Right-to-left text for terminals that show characters left to right in the order they arrive (most of them,
/// Windows' console host and Windows Terminal included): each line is shaped (<see cref="ArabicShaping"/>) and
/// reordered by the Unicode Bidirectional Algorithm (<see cref="Bidi"/>), so Arabic reads right to left with joined
/// letters while embedded English words, numbers and paths keep their order.
/// </summary>
/// <remarks>
/// A line is split at its column gaps (two or more spaces, as tables, name-value lines and help items are laid out),
/// and each part is a paragraph of its own whose direction is that of its first strong letter (rules P2 and P3): an
/// Arabic description beside an option name reads right to left in its own column, and the columns keep their places
/// (a part that got shorter, from a ligature or a combining mark, is padded so the next column starts where it did).
/// Leading indentation stays on the left. Colour escapes keep colouring the characters they coloured. Combining marks
/// follow their base after reordering (rule L3), brackets are mirrored in right-to-left runs (rule L4), and the
/// direction marks and controls (LRM, RLM, ALM, embeddings, isolates) are dropped from the output since they have
/// done their work.
/// </remarks>
internal static class VisualText
{
    /// <summary>The code points of <paramref name="text"/> (a lone surrogate is kept as its own value).</summary>
    public static int[] CodePoints(string text)
    {
        int count = 0;
        for (int i = 0; i < text.Length; i++, count++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
        }

        var points = new int[count];
        for (int i = 0, k = 0; i < text.Length; i++, k++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                points[k] = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }
            else
            {
                points[k] = text[i];
            }
        }

        return points;
    }

    /// <summary>Appends one code point (a lone surrogate as its own char).</summary>
    public static void Append(StringBuilder text, int codePoint)
    {
        if (codePoint < 0x10000)
        {
            text.Append((char)codePoint);
        }
        else
        {
            text.Append(char.ConvertFromUtf32(codePoint));
        }
    }

    /// <summary>Whether <paramref name="text"/> has anything this class would change: right-to-left text or Arabic letters.</summary>
    public static bool NeedsRendering(string text)
    {
        // Below U+0590 nothing needs it (surrogates are above it): a line of Latin text is one vectorized search.
        int first = text.AsSpan().IndexOfAnyInRange('֐', '￿');
        if (first < 0)
        {
            return false;
        }

        for (int i = first; i < text.Length; i++)
        {
            int cp = text[i];
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                cp = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }

            if (cp < 0x0590)
            {
                continue;
            }

            var type = Bidi.ClassOf(cp);
            if (type is BidiClass.R or BidiClass.AL or BidiClass.AN or BidiClass.RLE or BidiClass.RLO or BidiClass.RLI or BidiClass.FSI
                || cp is 0x200E or 0x200F)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// One line (no line break) in visual order. With <paramref name="rightAlignWidth"/> above 0, a line that is one
    /// right-to-left paragraph (no column gaps) is padded on the left to end at that column. A line without column gaps
    /// takes its direction from <paramref name="paragraphLevel"/> when that is 0 or 1 (a piece of a wrapped paragraph).
    /// </summary>
    public static string Line(string line, int rightAlignWidth = 0, int paragraphLevel = -1)
    {
        if (!NeedsRendering(line))
        {
            return line;
        }

        if (line.Contains('\r', StringComparison.Ordinal))
        {
            return string.Join('\r', line.Split('\r').Select(part => Line(part, rightAlignWidth, paragraphLevel)));
        }

        // Escape sequences out: colours (SGR) are remembered per character, other controls go first on the line.
        var plain = new StringBuilder(line.Length);
        var styles = new List<string>(line.Length);
        var controls = new StringBuilder();
        string style = "";
        bool styled = false;
        for (int i = 0; i < line.Length; i++)
        {
            if (line[i] == '\u001b' && i + 1 < line.Length && line[i + 1] is '[' or ']')
            {
                int end = EscapeEnd(line, i);
                string escape = line[i..end];
                if (escape[1] == '[' && escape[^1] == 'm')
                {
                    style = escape is "\u001b[0m" or "\u001b[m" ? "" : style + escape;
                    styled = true;
                }
                else
                {
                    controls.Append(escape);
                }

                i = end - 1;
                continue;
            }

            plain.Append(line[i]);
            styles.Add(style);
            if (char.IsHighSurrogate(line[i]) && i + 1 < line.Length && char.IsLowSurrogate(line[i + 1]))
            {
                plain.Append(line[++i]);
                styles.Add(style);
            }
        }

        string text = plain.ToString();
        var output = new StringBuilder(line.Length + 16).Append(controls);
        string current = "";
        int column = 0;
        void Emit(int codePoint, string charStyle)
        {
            if (charStyle != current)
            {
                output.Append("\u001b[0m").Append(charStyle);
                current = charStyle;
            }

            Append(output, codePoint);
            column += Width(codePoint);
        }

        // Indentation, then the parts between column gaps, each starting at the column it had. One right-to-left
        // paragraph with a width to align to ends at that width instead, its indentation dropped.
        var found = Parts(text);
        var parts = found.Select(p => (p.Start, p.Column, Shaped: Reorder(text.Substring(p.Start, p.Length), found.Count == 1 ? paragraphLevel : -1))).ToList();
        if (rightAlignWidth > 0 && parts.Count == 1 && parts[0].Shaped.Direction == 1)
        {
            int width = parts[0].Shaped.Points.Sum(Width);
            parts[0] = (parts[0].Start, Math.Max(0, rightAlignWidth - width), parts[0].Shaped);
        }

        foreach (var (start, target, (points, from, _)) in parts)
        {
            while (column < target)
            {
                Emit(' ', start > 0 ? styles[start - 1] : "");
            }

            for (int k = 0; k < points.Length; k++)
            {
                Emit(points[k], styles[start + from[k]]);
            }
        }

        if (styled)
        {
            output.Append("\u001b[0m");                                         // also ends a colour begun before the line
        }

        return output.ToString();
    }

    // The display width of a code point: none for marks and format characters, one otherwise.
    private static int Width(int codePoint) =>
        CharUnicodeInfo.GetUnicodeCategory(codePoint) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark or UnicodeCategory.Format ? 0 : 1;

    // The end (exclusive) of the escape sequence starting at index: CSI (ESC [ ... final byte) or OSC (ESC ] ... BEL or ST).
    private static int EscapeEnd(string line, int index)
    {
        if (line[index + 1] == '[')
        {
            for (int i = index + 2; i < line.Length; i++)
            {
                if (line[i] is >= '@' and <= '~')
                {
                    return i + 1;
                }
            }

            return line.Length;
        }

        for (int i = index + 2; i < line.Length; i++)
        {
            if (line[i] == '\u0007')
            {
                return i + 1;
            }

            if (line[i] == '\u001b' && i + 1 < line.Length && line[i + 1] == '\\')
            {
                return i + 2;
            }
        }

        return line.Length;
    }

    // The parts of a line between its indentation and column gaps (two or more spaces): start, length (UTF-16) and the
    // column the part starts at (its code point index in the line).
    private static List<(int Start, int Length, int Column)> Parts(string text)
    {
        var parts = new List<(int, int, int)>();
        int i = 0, column = 0;
        while (i < text.Length)
        {
            if (text[i] == ' ' && (parts.Count == 0 && i == column || i + 1 < text.Length && text[i + 1] == ' ' || i + 1 == text.Length))
            {
                // A gap (or the indentation, or trailing spaces): skipped over; the next part starts after it.
                while (i < text.Length && text[i] == ' ')
                {
                    i++;
                    column++;
                }

                continue;
            }

            int start = i, startColumn = column;
            while (i < text.Length && !(text[i] == ' ' && (i + 1 == text.Length || text[i + 1] == ' ')))
            {
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    i++;
                }

                i++;
                column++;
            }

            parts.Add((start, i - start, startColumn));
        }

        return parts;
    }

    /// <summary>
    /// One paragraph shaped and in visual order: its code points, for each the index (UTF-16) it came from in
    /// <paramref name="text"/>, and the paragraph's direction (0 left to right, 1 right to left). The direction is
    /// found from the text (-1) unless <paramref name="paragraphLevel"/> gives it.
    /// </summary>
    internal static (int[] Points, int[] From, int Direction) Reorder(string text, int paragraphLevel = -1)
    {
        var source = CodePoints(text);
        var offsets = new int[source.Length];
        for (int k = 1; k < source.Length; k++)
        {
            offsets[k] = offsets[k - 1] + (source[k - 1] >= 0x10000 ? 2 : 1);
        }

        var shaped = ArabicShaping.Shape(source, out var shapedFrom);
        var from = Array.ConvertAll(shapedFrom, k => offsets[k]);
        var types = Array.ConvertAll(shaped, Bidi.ClassOf);
        var levels = Bidi.Levels(types, shaped, paragraphLevel, out int paragraph);
        var order = Bidi.VisualOrder(levels);
        var points = new List<int>(order.Length);
        var origins = new List<int>(order.Length);
        for (int v = 0; v < order.Length; v++)
        {
            int index = order[v];
            // L3: marks of a right-to-left run come before their base after reversing; put them back after it.
            if (types[index] == BidiClass.NSM && (levels[index] & 1) == 1)
            {
                int end = v;
                while (end + 1 < order.Length && types[order[end + 1]] == BidiClass.NSM && (levels[order[end + 1]] & 1) == 1)
                {
                    end++;
                }

                if (end + 1 < order.Length)
                {
                    end++;
                    Add(order[end]);
                    for (int m = end - 1; m >= v; m--)
                    {
                        Add(order[m]);
                    }

                    v = end;
                    continue;
                }
            }

            Add(index);
        }

        return ([.. points], [.. origins], paragraph);

        void Add(int index)
        {
            int cp = shaped[index];
            if (cp is 0x200E or 0x200F or 0x061C)
            {
                return;                                                          // direction marks: done their work
            }

            points.Add((levels[index] & 1) == 1 ? Bidi.Mirror(cp) : cp);
            origins.Add(from[index]);
        }
    }

    /// <summary>
    /// Whether a terminal shows right-to-left text correctly by itself, from what it says about itself in the
    /// environment; null when nothing says so (then the tool reorders: most terminals do not).
    /// </summary>
    /// <remarks>
    /// Terminals that apply the bidirectional algorithm and join Arabic letters themselves: GNOME Terminal and the
    /// other terminals built on VTE from 0.58 (<c>VTE_VERSION</c> 5800 and above), Konsole (<c>KONSOLE_VERSION</c>),
    /// mlterm (<c>MLTERM</c> or a <c>TERM</c> starting with mlterm), mintty, which Git for Windows and Cygwin use
    /// (<c>TERM_PROGRAM=mintty</c>), and macOS Terminal (<c>TERM_PROGRAM=Apple_Terminal</c>). Windows' console host,
    /// Windows Terminal (<c>WT_SESSION</c>), VS Code's terminal (<c>TERM_PROGRAM=vscode</c>), xterm, kitty, Alacritty,
    /// WezTerm (its bidi setting is off by default), foot and iTerm2 draw cells in arrival order.
    /// </remarks>
    public static bool? TerminalReordersItself(Func<string, string?> environment)
    {
        if (environment("VTE_VERSION") is { } vte && int.TryParse(vte, NumberStyles.Integer, CultureInfo.InvariantCulture, out int version))
        {
            return version >= 5800;
        }

        if (environment("KONSOLE_VERSION") is { Length: > 0 } || environment("MLTERM") is { Length: > 0 }
            || environment("TERM") is { } term && term.StartsWith("mlterm", StringComparison.Ordinal))
        {
            return true;
        }

        return environment("TERM_PROGRAM") switch
        {
            "mintty" or "Apple_Terminal" => true,
            "vscode" or "iTerm.app" or "WezTerm" => false,
            _ => environment("WT_SESSION") is { Length: > 0 } ? false : null,
        };
    }

    /// <summary>
    /// The rendering <paramref name="requested"/> resolves to: <see cref="TextRendering.Auto"/> is logical when the output
    /// is not a terminal (files and pipes keep the text as stored) or the terminal reorders text itself, visual otherwise.
    /// </summary>
    public static TextRendering Resolve(TextRendering requested, bool terminal, Func<string, string?> environment) =>
        requested != TextRendering.Auto ? requested
        : !terminal || TerminalReordersItself(environment) == true ? TextRendering.Logical
        : TextRendering.Visual;

    /// <summary>Parses a <c>--lang-render</c> value: auto, visual or logical (visual-right: visual, right-aligned).</summary>
    public static (TextRendering Rendering, bool RightAlign) Parse(string text, string source) => text switch
    {
        "auto" => (TextRendering.Auto, false),
        "visual" => (TextRendering.Visual, false),
        "visual-right" => (TextRendering.Visual, true),
        "logical" => (TextRendering.Logical, false),
        _ => throw new UsageException($"{source} takes auto, visual, visual-right or logical, not '{text}'."),
    };
}

/// <summary>
/// A writer that passes text on in visual order (<see cref="VisualText.Line"/>). Text goes straight through until a
/// line's first right-to-left character; from there the rest of the line is held and rendered when its line break
/// arrives, so a streamed answer appears line by line and a prompt such as "> " at once. A held line wider than the
/// terminal is broken at spaces before it is reordered, as the bidirectional algorithm reorders each displayed line on
/// its own (otherwise the terminal would wrap the reordered text and the rows would read in the wrong order).
/// <see cref="Flush"/> keeps a held line held (streams flush after every token); <see cref="RenderPending"/> renders it
/// (a question waiting for its answer, and the end of the run).
/// </summary>
internal sealed class VisualWriter(TextWriter inner, int width = 0, bool rightAlign = false) : TextWriter
{
    private readonly StringBuilder _held = new();
    private int _column;
    private int _escape;                                                          // 0 none, 1 after ESC, 2 in CSI, 3 in OSC

    public override Encoding Encoding => inner.Encoding;

    /// <summary>The writer the text goes to.</summary>
    public TextWriter Inner => inner;

    public override void Write(char value)
    {
        if (value == '\n')
        {
            bool cr = _held.Length > 0 && _held[^1] == '\r';
            if (cr)
            {
                _held.Length--;
            }

            RenderPending();
            inner.Write(cr ? "\r\n" : "\n");
            _column = 0;
            return;
        }

        if (_held.Length > 0)
        {
            // Leading spaces (right alignment) or a high surrogate were held to see what follows them.
            bool spaces = _held[0] == ' ' && HeldIsSpaces();
            bool pair = _held.Length == 1 && char.IsHighSurrogate(_held[0]) && char.IsLowSurrogate(value);
            if (spaces && value != ' ' && !StartsHolding(value) || pair && !StartsHolding(char.ConvertToUtf32(_held[0], value)))
            {
                string pending = _held.ToString();
                _held.Clear();
                foreach (char c in pending)
                {
                    Pass(c);
                }

                Pass(value);
                return;
            }

            _held.Append(value);
            return;
        }

        if (_escape == 0 && (StartsHolding(value) || char.IsHighSurrogate(value) || value == ' ' && _column == 0 && rightAlign))
        {
            _held.Append(value);
            return;
        }

        Pass(value);
    }

    // Whether the held text is only spaces (right alignment), read in place: it is asked for every character of a held line.
    private bool HeldIsSpaces()
    {
        foreach (var chunk in _held.GetChunks())
        {
            if (chunk.Span.ContainsAnyExcept(' '))
            {
                return false;
            }
        }

        return true;
    }

    // Whether a character begins the held part of a line: a right-to-left letter or number, or a direction control.
    private static bool StartsHolding(int codePoint) =>
        codePoint >= 0x0590 && (codePoint is 0x200E or 0x200F
            || Bidi.ClassOf(codePoint) is BidiClass.R or BidiClass.AL or BidiClass.AN or BidiClass.RLE or BidiClass.RLO or BidiClass.RLI or BidiClass.FSI);

    // Writes a character through, keeping count of the columns it takes (escape sequences take none).
    private void Pass(char value)
    {
        inner.Write(value);
        int before = _escape;
        _escape = (_escape, value) switch
        {
            (0, '\u001b') => 1,
            (1, '[') => 2,
            (1, ']') => 3,
            (1, _) => 0,
            (2, >= '@' and <= '~') => 0,
            (3, '\u0007') => 0,
            (3, '\u001b') => 1,
            _ => _escape,
        };
        if (before == 0 && _escape == 0 && !char.IsLowSurrogate(value))
        {
            _column = value == '\r' ? 0 : _column + (char.IsControl(value) ? 0 : 1);
        }
    }

    public override void Write(string? value)
    {
        foreach (char c in value ?? "")
        {
            Write(c);
        }
    }

    public override void WriteLine(string? value)
    {
        Write(value);
        Write('\n');
    }

    public override void WriteLine() => Write('\n');

    /// <summary>Flushes what went through; a held line stays held until its line break.</summary>
    public override void Flush() => inner.Flush();

    /// <summary>Renders the held part of the line now (before a question waits for its answer, and at the end).</summary>
    public void RenderPending()
    {
        if (_held.Length > 0)
        {
            string text = _held.ToString();
            _held.Clear();
            inner.Write(Render(text));
        }

        inner.Flush();
    }

    // The held text in visual order, broken into rows that fit the terminal.
    private string Render(string text)
    {
        int available = width - _column;
        if (width <= 0 || available <= 0 || text.Length <= available || text.Contains('\u001b', StringComparison.Ordinal))
        {
            return VisualText.Line(text, rightAlign && _column == 0 ? width : 0);
        }

        // Rows after the first start under the first (a description column stays a column) unless that is past the middle.
        int hang = _column < width / 2 ? _column : 0;
        int level = Bidi.ParagraphLevel(Array.ConvertAll(VisualText.CodePoints(text), Bidi.ClassOf));
        var rows = new List<string>();
        int start = 0;
        while (start < text.Length)
        {
            int room = rows.Count == 0 ? available : width - hang;
            if (text.Length - start <= room)
            {
                rows.Add(text[start..]);
                break;
            }

            int cut = text.LastIndexOf(' ', start + room, room);
            cut = cut <= start ? start + room : cut;
            rows.Add(text[start..cut]);
            start = cut < text.Length && text[cut] == ' ' ? cut + 1 : cut;
        }

        return string.Join('\n', rows.Select((row, i) => (i > 0 ? new string(' ', hang) : "")
            + VisualText.Line(row, rightAlign && (i > 0 && hang == 0 || _column == 0) ? width : 0, level)));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            RenderPending();
        }

        base.Dispose(disposing);
    }
}
