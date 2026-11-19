// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Data;

/// <summary>What kind of form field a <see cref="DocumentField"/> is.</summary>
public enum DocumentFieldKind : byte
{
    /// <summary>A text box.</summary>
    Text,

    /// <summary>A check box (on or off).</summary>
    CheckBox,

    /// <summary>One of a group of radio buttons.</summary>
    Radio,

    /// <summary>A list to choose from (one or several choices).</summary>
    List,

    /// <summary>A drop-down list, which may also take typed text.</summary>
    Combo,

    /// <summary>A push button (no value).</summary>
    PushButton,

    /// <summary>A place for a signature.</summary>
    Signature,
}

/// <summary>How a form field behaves.</summary>
[Flags]
public enum DocumentFieldFlags
{
    /// <summary>Nothing set.</summary>
    None = 0,

    /// <summary>It cannot be changed.</summary>
    ReadOnly = 1,

    /// <summary>It must have a value before the form is sent.</summary>
    Required = 2,

    /// <summary>Its value is not sent with the form.</summary>
    NoExport = 4,

    /// <summary>A text box of several lines.</summary>
    Multiline = 8,

    /// <summary>A text box that hides what is typed.</summary>
    Password = 16,

    /// <summary>A text box split into <see cref="DocumentField.MaxLength"/> equal cells, one character each.</summary>
    Comb = 32,

    /// <summary>A text box that does not scroll past its box.</summary>
    DoNotScroll = 64,

    /// <summary>A text box not spell-checked.</summary>
    DoNotSpellCheck = 128,

    /// <summary>A text box whose value has formatting (rich text).</summary>
    RichText = 256,

    /// <summary>A list that takes several choices.</summary>
    MultiSelect = 512,

    /// <summary>A drop-down list that takes typed text too.</summary>
    Editable = 1024,

    /// <summary>A list whose choices are shown sorted.</summary>
    Sort = 2048,

    /// <summary>Radio buttons of the same value turn on and off together.</summary>
    RadiosInUnison = 4096,

    /// <summary>A check box or radio button that cannot be turned off once on.</summary>
    NoToggleToOff = 8192,
}

/// <summary>Where a field's text sits across its box.</summary>
public enum DocumentFieldAlignment : byte
{
    /// <summary>Against the left edge.</summary>
    Left,

    /// <summary>Centred.</summary>
    Centre,

    /// <summary>Against the right edge.</summary>
    Right,
}

/// <summary>The mark a check box or radio button shows when it is on.</summary>
public enum DocumentFieldMark : byte
{
    /// <summary>A check mark.</summary>
    Check,

    /// <summary>A filled circle.</summary>
    Circle,

    /// <summary>A cross.</summary>
    Cross,

    /// <summary>A diamond.</summary>
    Diamond,

    /// <summary>A filled square.</summary>
    Square,

    /// <summary>A five-pointed star.</summary>
    Star,
}

/// <summary>A choice of a list or a drop-down list.</summary>
/// <param name="Text">What is shown.</param>
/// <param name="Export">The value the form sends (the shown text when the document gives none).</param>
public readonly record struct DocumentFieldOption(string Text, string Export);

/// <summary>Where a field is shown: a page, a box and, for a check box or radio button, the state it shows.</summary>
/// <param name="Page">The page's index.</param>
/// <param name="Box">The box on the page.</param>
/// <param name="State">The appearance state shown ("Off", or the on state's name), or null.</param>
public readonly record struct DocumentWidget(int Page, DocumentBox Box, string? State = null)
{
    /// <summary>
    /// The state that shows a check box or radio button on (its on state's name: the value the field takes when this one
    /// is turned on), or null for other fields. A writer adding a field names it here (a check box's "Yes" unless given).
    /// </summary>
    public string? OnState { get; init; }
}

/// <summary>
/// A form field: its full name (its parents' names and its own, joined by dots), kind, value and choices, how its text is
/// shown, and where it is shown. Fields of a dynamic form's data are read as fields too; their layout is not drawn.
/// </summary>
/// <param name="Name">The full, dotted name.</param>
/// <param name="Kind">What kind of field it is.</param>
public sealed record DocumentField(string Name, DocumentFieldKind Kind)
{
    /// <summary>Its value (the text, the chosen export value, the on state of a check box), or null when it has none.</summary>
    public string? Value { get; init; }

    /// <summary>Its values, for a list that takes several (else the one value, or none).</summary>
    public IReadOnlyList<string> Values { get; init; } = [];

    /// <summary>The choices of a list or drop-down list, in order.</summary>
    public IReadOnlyList<DocumentFieldOption> Options { get; init; } = [];

    /// <summary>How it behaves.</summary>
    public DocumentFieldFlags Flags { get; init; }

    /// <summary>The most characters a text box takes, or null when there is no limit.</summary>
    public int? MaxLength { get; init; }

    /// <summary>Where it is shown (one place for most fields; a radio group's each button).</summary>
    public IReadOnlyList<DocumentWidget> Widgets { get; init; } = [];

    /// <summary>The text shown when pointing at it, or null.</summary>
    public string? Tooltip { get; init; }

    /// <summary>The value it takes when the form is reset, or null.</summary>
    public string? DefaultValue { get; init; }

    /// <summary>Where its text sits across its box (a text box, a list, a drop-down list).</summary>
    public DocumentFieldAlignment Alignment { get; init; }

    /// <summary>
    /// The font its text is shown in, by the name the document gives it (its default appearance's: a font resource such as
    /// "Helv", or a font's name), or null. A writer adding a field takes a standard font's name ("Helvetica") or a face's
    /// family from <see cref="DocumentFormOptions"/>; null lets it choose one that shows the text.
    /// </summary>
    public string? Font { get; init; }

    /// <summary>The size its text is shown at, in points; 0 sizes it to fit its box (<see cref="DocumentFormOptions"/>).</summary>
    public float FontSize { get; init; }

    /// <summary>The colour of its text and of a check box's or radio button's mark (black).</summary>
    public DocumentColour TextColour { get; init; }

    /// <summary>The colour its box is filled with, or null for none.</summary>
    public DocumentColour? Background { get; init; }

    /// <summary>The colour of its box's border, or null for none.</summary>
    public DocumentColour? Border { get; init; }

    /// <summary>The caption a push button shows, or null.</summary>
    public string? Caption { get; init; }

    /// <summary>The mark a check box or radio button shows when on (a check box's check mark, a radio button's circle, unless given).</summary>
    public DocumentFieldMark Mark { get; init; }
}

/// <summary>
/// How a writer makes the look of form fields (each widget's appearance, shown without a reader making its own): the fonts
/// a field's text is laid out in when its own font cannot show it, the sizes text fitted to its box may take, and the
/// spacing. A field's text is set through <see cref="DocumentTextSetters"/> (the shapers and the reading order: every
/// script), in the field's own font when that shows every character, else in a face given here that does.
/// </summary>
public sealed record DocumentFormOptions
{
    private readonly float _minFontSize = 4f, _maxFontSize = 12f, _padding = 2f, _lineSpacing = 1.15f, _borderWidth = 1f;
    private readonly IReadOnlyList<IFontFace> _faces = [];
    private readonly string _textSetter = DocumentTextSetters.Shaped;

    /// <summary>
    /// A folder of fonts a field's text may be laid out in when its own font does not show it, matched by name and then by
    /// the characters they cover (<see cref="DocumentDrawOptions.FontsVariable"/>, the folder drawing takes; null: only
    /// <see cref="Faces"/>).
    /// </summary>
    public string? Fonts { get; init; } = DocumentSettings.Text(DocumentDrawOptions.FontsVariable);

    /// <summary>Faces a field's text may be laid out in, asked before <see cref="Fonts"/>' (none unless given).</summary>
    public IReadOnlyList<IFontFace> Faces
    {
        get => _faces;
        init => _faces = value ?? throw new ArgumentNullException(nameof(Faces));
    }

    /// <summary>The smallest size text fitted to its box (a field of size 0) takes, in points (4).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float MinFontSize
    {
        get => _minFontSize;
        init => _minFontSize = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(MinFontSize), value, "A size is a finite number of points above 0.");
    }

    /// <summary>
    /// The size fitted text of several lines (and a list's choices) starts from and shrinks below until it fits, in points
    /// (12); a single line is sized to fit its box's height and width.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float MaxFontSize
    {
        get => _maxFontSize;
        init => _maxFontSize = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(MaxFontSize), value, "A size is a finite number of points above 0.");
    }

    /// <summary>The space between a box's edge and its text, in points (2).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative or not finite.</exception>
    public float Padding
    {
        get => _padding;
        init => _padding = value >= 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(Padding), value, "Padding is a finite number of points, 0 or more.");
    }

    /// <summary>The distance between the baselines of a field's lines, as a multiple of its text's size (1.15).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float LineSpacing
    {
        get => _lineSpacing;
        init => _lineSpacing = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(LineSpacing), value, "Line spacing is a finite multiple above 0.");
    }

    /// <summary>The width of a border a field's appearance draws when its widget states none, in points (1).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative or not finite.</exception>
    public float BorderWidth
    {
        get => _borderWidth;
        init => _borderWidth = value >= 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(BorderWidth), value, "A border's width is a finite number of points, 0 or more.");
    }

    /// <summary>The colour behind a list's chosen rows (a light blue).</summary>
    public DocumentColour Highlight { get; init; } = DocumentColour.Rgb(0.6f, 0.75f, 0.95f);

    /// <summary>The text setter a field's text is laid out by: a <see cref="DocumentTextSetters"/> name (<see cref="DocumentTextSetters.Shaped"/>).</summary>
    public string TextSetter
    {
        get => _textSetter;
        init => _textSetter = string.IsNullOrWhiteSpace(value) ? throw new ArgumentException("A text setter is named.", nameof(TextSetter)) : value;
    }
}

/// <summary>The kinds of <see cref="DocumentAnnotation"/> the library names (a document's other kinds keep their own names).</summary>
public static class DocumentAnnotationKinds
{
    /// <summary>A note (an icon that opens text).</summary>
    public const string Note = "note";

    /// <summary>Text shown on the page.</summary>
    public const string FreeText = "free-text";

    /// <summary>A straight line (its ends in <see cref="DocumentAnnotation.Points"/>).</summary>
    public const string Line = "line";

    /// <summary>A rectangle.</summary>
    public const string Square = "square";

    /// <summary>An ellipse.</summary>
    public const string Circle = "circle";

    /// <summary>A closed polygon (its vertices in <see cref="DocumentAnnotation.Points"/>).</summary>
    public const string Polygon = "polygon";

    /// <summary>An open polyline (its vertices in <see cref="DocumentAnnotation.Points"/>).</summary>
    public const string Polyline = "polyline";

    /// <summary>Highlighted text (the quads in <see cref="DocumentAnnotation.Points"/>).</summary>
    public const string Highlight = "highlight";

    /// <summary>Underlined text.</summary>
    public const string Underline = "underline";

    /// <summary>Text underlined with a wavy line.</summary>
    public const string Squiggly = "squiggly";

    /// <summary>Struck-out text.</summary>
    public const string StrikeOut = "strike-out";

    /// <summary>Strokes written on the page (in <see cref="DocumentAnnotation.Ink"/>).</summary>
    public const string Ink = "ink";

    /// <summary>A stamp (its name in <see cref="DocumentAnnotation.Icon"/>).</summary>
    public const string Stamp = "stamp";

    /// <summary>A caret marking where text goes.</summary>
    public const string Caret = "caret";

    /// <summary>A link (its target in <see cref="DocumentAnnotation.Uri"/> or <see cref="DocumentAnnotation.Destination"/>).</summary>
    public const string Link = "link";

    /// <summary>An attached file (in <see cref="DocumentAnnotation.Attachment"/>).</summary>
    public const string FileAttachment = "file-attachment";

    /// <summary>The window another annotation's text opens in.</summary>
    public const string Popup = "popup";

    /// <summary>A mark of content to remove (applied on request: what is under it is removed, not covered).</summary>
    public const string Redaction = "redaction";

    /// <summary>A form field's place on the page.</summary>
    public const string Widget = "widget";
}

/// <summary>
/// An annotation on a page: a note, a mark on the text (highlight, underline, strike-out), a shape, ink written on the page,
/// a stamp, a link, an attached file, a redaction mark, a form field's widget or any other kind by its name.
/// </summary>
/// <param name="Kind">What kind it is: one of <see cref="DocumentAnnotationKinds"/>, or another kind by the name the document gives it.</param>
/// <param name="Page">The page's index.</param>
/// <param name="Box">Its box on the page.</param>
public sealed record DocumentAnnotation(string Kind, int Page, DocumentBox Box)
{
    /// <summary>The document's own key for it (to remove or reply to it), or null for one not read from a document.</summary>
    public string? Id { get; init; }

    /// <summary>Its text, or null.</summary>
    public string? Contents { get; init; }

    /// <summary>Who wrote it, or null.</summary>
    public string? Author { get; init; }

    /// <summary>When it was made, or null.</summary>
    public DateTimeOffset? Created { get; init; }

    /// <summary>When it was last changed, or null.</summary>
    public DateTimeOffset? Modified { get; init; }

    /// <summary>Its colour, or null when it has none.</summary>
    public DocumentColour? Colour { get; init; }

    /// <summary>Its opacity, 0 to 1 (1).</summary>
    public float Opacity { get; init; } = 1f;

    /// <summary>Its points, x then y in page points: a mark's quads (four corners each), a line's ends, a polygon's vertices.</summary>
    public ReadOnlyMemory<float> Points { get; init; }

    /// <summary>The strokes of an ink annotation, in the order written.</summary>
    public IReadOnlyList<DocumentPath> Ink { get; init; } = [];

    /// <summary>A link's web address, or null.</summary>
    public string? Uri { get; init; }

    /// <summary>A link's place in the document, or null.</summary>
    public DocumentDestination? Destination { get; init; }

    /// <summary>A stamp's or a note's icon name, or null.</summary>
    public string? Icon { get; init; }

    /// <summary>The file of a file attachment, or null.</summary>
    public DocumentAttachment? Attachment { get; init; }

    /// <summary>The replies to it, in order (its thread).</summary>
    public IReadOnlyList<DocumentAnnotation> Replies { get; init; } = [];

    /// <summary>
    /// How it looks, as the document draws it: its parts placed on the page (drawn when annotations are asked for), or null
    /// when it has none. A reader fills its parts as the page's own were asked (<see cref="DocumentParts"/>): asked for
    /// annotations alone, an appearance holds the page's size and nothing more.
    /// </summary>
    public DocumentPage? Appearance { get; init; }

    /// <summary>Its inside's colour (a square's, a circle's, a polygon's, a line's ends', a redaction mark's fill), or null when it has none.</summary>
    public DocumentColour? InteriorColour { get; init; }

    /// <summary>The width of its border, line or strokes in points, or null when the document gives none.</summary>
    public float? BorderWidth { get; init; }

    /// <summary>How a line's (or polyline's) first and last ends are drawn, by the names the document gives ("None", "OpenArrow", "ClosedArrow", "Circle", "Square", "Diamond", "Butt", ...); empty when none are given.</summary>
    public IReadOnlyList<string> LineEndings { get; init; } = [];

    /// <summary>How it behaves: hidden, printed, locked, ...</summary>
    public DocumentAnnotationFlags Flags { get; init; }

    /// <summary>The <see cref="Id"/> of the annotation it replies to, or null.</summary>
    public string? InReplyTo { get; init; }

    /// <summary>How it stands to <see cref="InReplyTo"/>: "reply" (in its thread) or "group" (one with it), or null.</summary>
    public string? ReplyType { get; init; }

    /// <summary>The state a reply sets ("Accepted", "Rejected", "Completed", "Marked", ...), or null.</summary>
    public string? State { get; init; }

    /// <summary>The model of <see cref="State"/> ("Review" or "Marked"), or null.</summary>
    public string? StateModel { get; init; }

    /// <summary>The <see cref="Id"/> of the popup its text opens in, or null.</summary>
    public string? Popup { get; init; }

    /// <summary>A popup's annotation (its <see cref="Id"/>), or null.</summary>
    public string? Parent { get; init; }

    /// <summary>Whether a note or a popup is shown open.</summary>
    public bool Open { get; init; }

    /// <summary>A redaction mark's text, shown over its area once it is applied, or null.</summary>
    public string? OverlayText { get; init; }
}

/// <summary>A place in a document: a page with a position and zoom, or a destination's name the document resolves.</summary>
/// <param name="Page">The page's index (-1 when only <see cref="Name"/> is given).</param>
public sealed record DocumentDestination(int Page)
{
    /// <summary>The left edge shown, in page points, or null to keep the viewer's.</summary>
    public float? X { get; init; }

    /// <summary>The top edge shown, in page points, or null to keep the viewer's.</summary>
    public float? Y { get; init; }

    /// <summary>The zoom (1 = 100%), or null to keep the viewer's.</summary>
    public float? Zoom { get; init; }

    /// <summary>The destination's name, for a named destination, or null.</summary>
    public string? Name { get; init; }

    /// <summary>
    /// How the page is shown: at <see cref="X"/>, <see cref="Y"/> and <see cref="Zoom"/>
    /// (<see cref="DocumentDestinationFit.Position"/>, the default), or fitted to the window (the whole page, its width at
    /// <see cref="Y"/>, its height at <see cref="X"/>, a box, or its content's bounds).
    /// </summary>
    public DocumentDestinationFit Fit { get; init; }

    /// <summary>The right edge of the box shown (<see cref="DocumentDestinationFit.Box"/>), in page points, or null.</summary>
    public float? Right { get; init; }

    /// <summary>The bottom edge of the box shown (<see cref="DocumentDestinationFit.Box"/>), in page points, or null.</summary>
    public float? Bottom { get; init; }
}

/// <summary>A link to write on a page: its box and its target (a web address or a place in the document).</summary>
/// <param name="Box">Its box on the page.</param>
/// <param name="Uri">A web address, or null.</param>
/// <param name="Destination">A place in the document, or null.</param>
public sealed record DocumentLink(DocumentBox Box, string? Uri = null, DocumentDestination? Destination = null);

/// <summary>How a bookmark's title is shown.</summary>
[Flags]
public enum DocumentOutlineStyle : byte
{
    /// <summary>Plain.</summary>
    None = 0,

    /// <summary>In italics.</summary>
    Italic = 1,

    /// <summary>In bold.</summary>
    Bold = 2,
}

/// <summary>A bookmark: its title, where it leads, and the bookmarks under it.</summary>
/// <param name="Title">The title shown.</param>
public sealed record DocumentOutline(string Title)
{
    /// <summary>Where it leads, or null.</summary>
    public DocumentDestination? Destination { get; init; }

    /// <summary>A web address it opens instead, or null.</summary>
    public string? Uri { get; init; }

    /// <summary>Whether its children are shown open.</summary>
    public bool Open { get; init; }

    /// <summary>The title's colour, or null.</summary>
    public DocumentColour? Colour { get; init; }

    /// <summary>The title's style.</summary>
    public DocumentOutlineStyle Style { get; init; }

    /// <summary>The bookmarks under it, in order.</summary>
    public IReadOnlyList<DocumentOutline> Children { get; init; } = [];
}

/// <summary>How a range of pages is numbered.</summary>
public enum DocumentPageNumbering : byte
{
    /// <summary>No number: the prefix alone.</summary>
    None,

    /// <summary>1, 2, 3.</summary>
    Decimal,

    /// <summary>I, II, III.</summary>
    UpperRoman,

    /// <summary>i, ii, iii.</summary>
    LowerRoman,

    /// <summary>A to Z, then AA to ZZ, ...</summary>
    UpperLetters,

    /// <summary>a to z, then aa to zz, ...</summary>
    LowerLetters,
}

/// <summary>
/// The labels of a range of pages, from <see cref="StartPage"/> to the next range's start: a prefix and a number in a style
/// ("iv", "A-3").
/// </summary>
/// <param name="StartPage">The range's first page (index).</param>
/// <param name="Numbering">How its pages are numbered.</param>
public sealed record DocumentPageLabel(int StartPage, DocumentPageNumbering Numbering)
{
    /// <summary>The text before the number ("A-"), or null.</summary>
    public string? Prefix { get; init; }

    /// <summary>The number of the range's first page (1).</summary>
    public int FirstNumber { get; init; } = 1;

    /// <summary>The label of page <paramref name="index"/>, counted from this range's start.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The page is before the range.</exception>
    public string Label(int index)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(index, StartPage);
        int number = FirstNumber + (index - StartPage);
        string digits = Numbering switch
        {
            DocumentPageNumbering.Decimal => number.ToString(CultureInfo.InvariantCulture),
            DocumentPageNumbering.UpperRoman => Roman(number),
            DocumentPageNumbering.LowerRoman => Roman(number).ToLowerInvariant(),
            DocumentPageNumbering.UpperLetters => Letters(number),
            DocumentPageNumbering.LowerLetters => Letters(number).ToLowerInvariant(),
            _ => "",
        };
        return (Prefix ?? "") + digits;
    }

    /// <summary>The label of page <paramref name="index"/> by the range of <paramref name="labels"/> it falls in, or null when it falls in none.</summary>
    public static string? Find(IReadOnlyList<DocumentPageLabel> labels, int index)
    {
        ArgumentNullException.ThrowIfNull(labels);
        DocumentPageLabel? range = null;
        foreach (var label in labels)
        {
            if (label.StartPage <= index && (range is null || label.StartPage >= range.StartPage))
            {
                range = label;
            }
        }

        return range?.Label(index);
    }

    private static string Roman(int number)
    {
        if (number <= 0)
        {
            return number.ToString(CultureInfo.InvariantCulture);
        }

        ReadOnlySpan<int> values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
        string[] symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"];
        var text = new System.Text.StringBuilder();
        for (int i = 0; i < values.Length; i++)
        {
            while (number >= values[i])
            {
                text.Append(symbols[i]);
                number -= values[i];
            }
        }

        return text.ToString();
    }

    // A, B, ... Z, AA, BB, ... ZZ, AAA, ...: the letter repeated once per pass through the alphabet.
    private static string Letters(int number) =>
        number <= 0 ? number.ToString(CultureInfo.InvariantCulture) : new string((char)('A' + (number - 1) % 26), (number - 1) / 26 + 1);
}
