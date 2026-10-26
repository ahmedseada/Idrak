// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Text;
using System.Windows.Input;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Markdown;

/// <summary>How a run of text is set.</summary>
[Flags]
public enum MarkdownRunStyle
{
    /// <summary>Plain text.</summary>
    Plain = 0,

    /// <summary>**Bold**.</summary>
    Bold = 1,

    /// <summary>*Italic*.</summary>
    Italic = 2,

    /// <summary>`Code`.</summary>
    Code = 4,
}

/// <summary>A run of text in one style.</summary>
/// <param name="Text">The text, as written (nothing in it changed, right-to-left text included).</param>
/// <param name="Style">How it is set.</param>
public readonly record struct MarkdownRun(string Text, MarkdownRunStyle Style);

/// <summary>One block of a Markdown answer: a paragraph (or heading), a list or a code block.</summary>
public abstract class MarkdownBlock : ViewModel
{
    /// <summary>Whether <paramref name="other"/> shows the same as this block.</summary>
    public abstract bool SameAs(MarkdownBlock other);

    /// <summary>Takes <paramref name="other"/>'s content (a block of the same kind), notifying what changed.</summary>
    internal abstract void Take(MarkdownBlock other);

    /// <summary>Whether <paramref name="text"/> reads right to left: its first strong letter is Arabic, Hebrew or another right-to-left script's.</summary>
    public static bool IsRightToLeft(ReadOnlySpan<char> text)
    {
        foreach (char c in text)
        {
            if (!char.IsLetter(c))
            {
                continue;
            }

            return c is >= '֐' and <= 'ࣿ' or >= 'יִ' and <= '﷿' or >= 'ﹰ' and <= '﻿';
        }

        return false;
    }

    /// <summary>The runs of an inline text: **bold** and __bold__, *italic*, `code`; a mark left open (an answer still coming) holds to the end.</summary>
    public static IReadOnlyList<MarkdownRun> Inline(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var runs = new List<MarkdownRun>();
        var current = new StringBuilder();
        var style = MarkdownRunStyle.Plain;
        void Flush()
        {
            if (current.Length > 0)
            {
                runs.Add(new MarkdownRun(current.ToString(), style));
                current.Clear();
            }
        }

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\\' && i + 1 < text.Length && text[i + 1] is '*' or '_' or '`' or '\\')
            {
                current.Append(text[++i]);
                continue;
            }

            if (c == '`')
            {
                Flush();
                int end = text.IndexOf('`', i + 1);
                string code = end < 0 ? text[(i + 1)..] : text[(i + 1)..end];
                if (code.Length > 0)
                {
                    runs.Add(new MarkdownRun(code, style | MarkdownRunStyle.Code));
                }

                i = end < 0 ? text.Length : end;
                continue;
            }

            if ((c == '*' || c == '_') && i + 1 < text.Length && text[i + 1] == c)
            {
                bool closing = (style & MarkdownRunStyle.Bold) != 0;
                if (closing || i + 2 < text.Length && !char.IsWhiteSpace(text[i + 2]))
                {
                    Flush();
                    style ^= MarkdownRunStyle.Bold;
                    i++;
                    continue;
                }
            }

            if (c == '*' && i + 1 < text.Length)
            {
                bool closing = (style & MarkdownRunStyle.Italic) != 0;
                bool opening = !closing && !char.IsWhiteSpace(text[i + 1]) && text.IndexOf('*', i + 1) > i + 1;
                if (closing && i > 0 && !char.IsWhiteSpace(text[i - 1]) || opening)
                {
                    Flush();
                    style ^= MarkdownRunStyle.Italic;
                    continue;
                }
            }
            else if (c == '*' && (style & MarkdownRunStyle.Italic) != 0)
            {
                Flush();
                style ^= MarkdownRunStyle.Italic;
                continue;
            }

            current.Append(c);
        }

        Flush();
        return runs;
    }

    internal static bool SameRuns(IReadOnlyList<MarkdownRun> a, IReadOnlyList<MarkdownRun> b)
    {
        if (a.Count != b.Count)
        {
            return false;
        }

        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>A paragraph, or a heading (<see cref="Level"/> 1 to 6).</summary>
public sealed class MarkdownParagraph : MarkdownBlock
{
    private IReadOnlyList<MarkdownRun> _runs;
    private string _text;
    private bool _rightToLeft;

    /// <summary>A paragraph of <paramref name="text"/> (its inline marks read) at heading <paramref name="level"/> (0: none).</summary>
    public MarkdownParagraph(string text, int level = 0)
    {
        ArgumentNullException.ThrowIfNull(text);
        Level = level;
        _text = text;
        _runs = Inline(text);
        _rightToLeft = IsRightToLeft(text);
    }

    /// <summary>The heading level, 1 to 6, or 0 for a paragraph.</summary>
    public int Level { get; }

    /// <summary>Whether it is a heading.</summary>
    public bool IsHeading => Level > 0;

    /// <summary>The text as written (the inline marks in it).</summary>
    public string Text => _text;

    /// <summary>Its runs.</summary>
    public IReadOnlyList<MarkdownRun> Runs => _runs;

    /// <summary>Whether it reads right to left.</summary>
    public bool RightToLeft => _rightToLeft;

    /// <inheritdoc />
    public override bool SameAs(MarkdownBlock other) => other is MarkdownParagraph p && p.Level == Level && p._text == _text;

    internal override void Take(MarkdownBlock other)
    {
        var p = (MarkdownParagraph)other;
        _text = p._text;
        if (!SameRuns(_runs, p._runs))
        {
            _runs = p._runs;
            OnPropertyChanged(nameof(Runs));
        }

        if (_rightToLeft != p._rightToLeft)
        {
            _rightToLeft = p._rightToLeft;
            OnPropertyChanged(nameof(RightToLeft));
        }
    }
}

/// <summary>One item of a list: its marker ("1." or "•") and its runs.</summary>
/// <param name="Marker">"1.", "2." for an ordered list, "•" for a bulleted one.</param>
/// <param name="Text">Its text as written.</param>
/// <param name="Runs">Its runs.</param>
/// <param name="RightToLeft">Whether it reads right to left.</param>
public sealed record MarkdownListItem(string Marker, string Text, IReadOnlyList<MarkdownRun> Runs, bool RightToLeft)
{
    /// <summary>An item of <paramref name="text"/>.</summary>
    public static MarkdownListItem Of(string marker, string text) => new(marker, text, MarkdownBlock.Inline(text), MarkdownBlock.IsRightToLeft(text));

    /// <summary>Equal when the marker and the text are.</summary>
    public bool Equals(MarkdownListItem? other) => other is not null && other.Marker == Marker && other.Text == Text;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Marker, Text);
}

/// <summary>A bulleted or numbered list.</summary>
public sealed class MarkdownList : MarkdownBlock
{
    private IReadOnlyList<MarkdownListItem> _items;

    /// <summary>A list of <paramref name="items"/>.</summary>
    public MarkdownList(bool ordered, IReadOnlyList<MarkdownListItem> items)
    {
        Ordered = ordered;
        _items = items ?? throw new ArgumentNullException(nameof(items));
    }

    /// <summary>Whether it is numbered.</summary>
    public bool Ordered { get; }

    /// <summary>Its items.</summary>
    public IReadOnlyList<MarkdownListItem> Items => _items;

    /// <inheritdoc />
    public override bool SameAs(MarkdownBlock other) => other is MarkdownList l && l.Ordered == Ordered && l._items.SequenceEqual(_items);

    internal override void Take(MarkdownBlock other)
    {
        _items = ((MarkdownList)other)._items;
        OnPropertyChanged(nameof(Items));
    }
}

/// <summary>A fenced code block, with its language and a copy command.</summary>
public sealed class MarkdownCode : MarkdownBlock
{
    private string _code;
    private bool _open;

    /// <summary>A code block of <paramref name="code"/> in <paramref name="language"/> ("" when none is named).</summary>
    public MarkdownCode(string language, string code, bool open = false, ICommand? copy = null)
    {
        Language = language ?? "";
        _code = code ?? "";
        _open = open;
        CopyCommand = copy;
    }

    /// <summary>The language named after the fence ("csharp"), or "".</summary>
    public string Language { get; }

    /// <summary>The language as the block's header shows it ("C#", "Python"; the name itself otherwise; "Code" when none).</summary>
    public string LanguageTitle => Language.ToLowerInvariant() switch
    {
        "" => "Code",
        "cs" or "csharp" or "c#" => "C#",
        "py" or "python" => "Python",
        "js" or "javascript" => "JavaScript",
        "ts" or "typescript" => "TypeScript",
        "sh" or "bash" or "shell" => "Shell",
        "json" => "JSON",
        "xml" or "xaml" => Language.ToUpperInvariant(),
        _ => Language,
    };

    /// <summary>The code, as written.</summary>
    public string Code => _code;

    /// <summary>Whether its closing fence has not come yet (an answer still streaming).</summary>
    public bool IsOpen => _open;

    /// <summary>Copies <see cref="Code"/> (given by the document; null when it cannot).</summary>
    public ICommand? CopyCommand { get; internal set; }

    /// <inheritdoc />
    public override bool SameAs(MarkdownBlock other) => other is MarkdownCode c && c.Language == Language && c._code == _code && c._open == _open;

    internal override void Take(MarkdownBlock other)
    {
        var c = (MarkdownCode)other;
        if (_code != c._code)
        {
            _code = c._code;
            OnPropertyChanged(nameof(Code));
        }

        if (_open != c._open)
        {
            _open = c._open;
            OnPropertyChanged(nameof(IsOpen));
        }
    }
}

/// <summary>
/// A Markdown answer as blocks, kept up to date as the text grows: <see cref="Update"/> parses the whole text again (a
/// few kilobytes, once a progress interval at most) and changes only the blocks that differ, in place where the kind is the
/// same, so a streaming answer redraws its last block and nothing above it. What it reads: paragraphs, headings (#),
/// bulleted (-, *, +) and numbered (1. or 1)) lists, fenced code blocks (```lang), **bold**, *italic* and `code`. Nothing
/// else is interpreted, and no text is changed (right-to-left text is kept as written; a block says its direction).
/// </summary>
public sealed class MarkdownDocument
{
    private readonly ICommand? _copy;
    private string _text = "";

    /// <summary>An empty document; <paramref name="copy"/> copies a code block's text (null: code blocks have no copy).</summary>
    public MarkdownDocument(Func<string, Task>? copy = null)
    {
        _copy = copy is null ? null : new Command<string>(async code => await copy(code ?? ""));
    }

    /// <summary>The blocks, in order.</summary>
    public ObservableCollection<MarkdownBlock> Blocks { get; } = [];

    /// <summary>The text last shown.</summary>
    public string Text => _text;

    /// <summary>How many blocks were added, replaced or changed in place since the start (each a redraw).</summary>
    public int Changes { get; private set; }

    /// <summary>Shows <paramref name="text"/>: the blocks that differ from the ones shown are changed, the rest kept.</summary>
    public void Update(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text == _text)
        {
            return;
        }

        _text = text;
        var next = Parse(text);
        for (int i = 0; i < next.Count; i++)
        {
            if (next[i] is MarkdownCode code)
            {
                code.CopyCommand = _copy;
            }

            if (i >= Blocks.Count)
            {
                Blocks.Add(next[i]);
                Changes++;
            }
            else if (!Blocks[i].SameAs(next[i]))
            {
                if (Blocks[i].GetType() == next[i].GetType() && Kind(Blocks[i]) == Kind(next[i]))
                {
                    Blocks[i].Take(next[i]);
                }
                else
                {
                    Blocks[i] = next[i];
                }

                Changes++;
            }
        }

        while (Blocks.Count > next.Count)
        {
            Blocks.RemoveAt(Blocks.Count - 1);
            Changes++;
        }
    }

    // The part of a block that a change in place cannot alter: a heading's level, a list's numbering, a code's language.
    private static string Kind(MarkdownBlock block) => block switch
    {
        MarkdownParagraph p => "p" + p.Level,
        MarkdownList l => l.Ordered ? "ol" : "ul",
        MarkdownCode c => "code:" + c.Language,
        _ => "",
    };

    /// <summary>The blocks of <paramref name="text"/>.</summary>
    public static IReadOnlyList<MarkdownBlock> Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var blocks = new List<MarkdownBlock>();
        var paragraph = new StringBuilder();
        List<MarkdownListItem>? items = null;
        bool ordered = false;
        string? marker = null;
        var item = new StringBuilder();
        void FlushParagraph()
        {
            if (paragraph.Length > 0)
            {
                blocks.Add(new MarkdownParagraph(paragraph.ToString()));
                paragraph.Clear();
            }
        }

        void FlushItem()
        {
            if (marker is not null)
            {
                items!.Add(MarkdownListItem.Of(marker, item.ToString()));
                item.Clear();
                marker = null;
            }
        }

        void FlushList()
        {
            FlushItem();
            if (items is { Count: > 0 })
            {
                blocks.Add(new MarkdownList(ordered, items));
            }

            items = null;
        }

        string[] lines = text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            string trimmed = line.TrimStart();
            int indent = line.Length - trimmed.Length;
            if (indent <= 3 && (trimmed.StartsWith("```", StringComparison.Ordinal) || trimmed.StartsWith("~~~", StringComparison.Ordinal)))
            {
                FlushParagraph();
                FlushList();
                string fence = trimmed[..3];
                string language = trimmed[3..].Trim();
                var code = new StringBuilder();
                bool closed = false;
                for (i++; i < lines.Length; i++)
                {
                    if (lines[i].TrimStart().StartsWith(fence, StringComparison.Ordinal))
                    {
                        closed = true;
                        break;
                    }

                    if (code.Length > 0)
                    {
                        code.Append('\n');
                    }

                    code.Append(lines[i]);
                }

                blocks.Add(new MarkdownCode(language, code.ToString(), open: !closed));
                continue;
            }

            if (trimmed.Length == 0)
            {
                FlushParagraph();
                FlushItem();
                continue;
            }

            if (indent <= 3 && Heading(trimmed) is var (level, title) && level > 0)
            {
                FlushParagraph();
                FlushList();
                blocks.Add(new MarkdownParagraph(title, level));
                continue;
            }

            if (ListItem(trimmed) is var (isOrdered, mark, rest) && mark is not null)
            {
                FlushParagraph();
                if (items is not null && isOrdered != ordered)
                {
                    FlushList();
                }

                FlushItem();
                items ??= [];
                ordered = isOrdered;
                marker = isOrdered ? mark : "•";
                item.Append(rest);
                continue;
            }

            if (items is not null)
            {
                if (marker is not null && (indent > 0 || lines[i - 1].Trim().Length > 0))
                {
                    item.Append('\n').Append(trimmed);                             // the item goes on (indented, or with no blank line before)
                    continue;
                }

                FlushList();
            }

            if (paragraph.Length > 0)
            {
                paragraph.Append('\n');
            }

            paragraph.Append(line.TrimEnd());
        }

        FlushParagraph();
        FlushList();
        return blocks;
    }

    // "## Title" → (2, "Title"); otherwise (0, "").
    private static (int Level, string Title) Heading(string line)
    {
        int level = 0;
        while (level < line.Length && level < 7 && line[level] == '#')
        {
            level++;
        }

        return level is >= 1 and <= 6 && level < line.Length && line[level] == ' ' ? (level, line[(level + 1)..].Trim().TrimEnd('#').TrimEnd()) : (0, "");
    }

    // "- text" → (false, "•", "text"); "12. text" or "12) text" → (true, "12.", "text"); otherwise (false, null, "").
    private static (bool Ordered, string? Marker, string Text) ListItem(string line)
    {
        if (line.Length >= 2 && line[0] is '-' or '*' or '+' && line[1] == ' ')
        {
            return (false, "•", line[2..].TrimStart());
        }

        int digits = 0;
        while (digits < line.Length && digits < 9 && char.IsAsciiDigit(line[digits]))
        {
            digits++;
        }

        return digits > 0 && digits + 1 < line.Length && line[digits] is '.' or ')' && line[digits + 1] == ' '
            ? (true, line[..digits] + ".", line[(digits + 2)..].TrimStart())
            : (false, null, "");
    }
}
