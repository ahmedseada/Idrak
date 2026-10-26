// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Microsoft.Maui.Graphics;

namespace Idrak.Maui.Controls;

/// <summary>
/// A view drawn on a canvas whose colours are theme keys: each colour property is set from the theme with
/// <c>SetDynamicResource</c>, so a theme change redraws it in the new colours. Drawing costs nothing when not shown.
/// </summary>
public abstract class DrawnView : GraphicsView, IDrawable
{
    /// <summary>A view that draws itself.</summary>
    protected DrawnView()
    {
        Drawable = this;
        BackgroundColor = Colors.Transparent;
        InputTransparent = true;
    }

    /// <summary>Makes a colour property that redraws the view when it changes.</summary>
    protected static BindableProperty ColorProperty<T>(string name) where T : DrawnView =>
        BindableProperty.Create(name, typeof(Color), typeof(T), Colors.Transparent, propertyChanged: (b, _, _) => ((DrawnView)b).Invalidate());

    /// <summary>Makes a property that redraws the view when it changes.</summary>
    protected static BindableProperty DrawnProperty<T, TValue>(string name, TValue defaultValue) where T : DrawnView =>
        BindableProperty.Create(name, typeof(TValue), typeof(T), defaultValue, propertyChanged: (b, _, _) => ((DrawnView)b).Invalidate());

    /// <inheritdoc />
    public abstract void Draw(ICanvas canvas, RectF dirtyRect);
}

/// <summary>
/// The live waveform (the design's recording card): one bar a level heard, newest on the right, the latest few in the head
/// colour and a head line after them; bars not heard yet in the idle colour. <see cref="Push"/> adds a level (0 to 1);
/// a bound <see cref="Level"/> pushes each new value while <see cref="IsLive"/> is set.
/// </summary>
public sealed class WaveformView : DrawnView
{
    /// <summary>The latest level, pushed while live.</summary>
    public static readonly BindableProperty LevelProperty =
        BindableProperty.Create(nameof(Level), typeof(double), typeof(WaveformView), 0d, propertyChanged: (b, _, n) => ((WaveformView)b).OnLevel((double)n));

    /// <summary>Whether a bound level is pushed.</summary>
    public static readonly BindableProperty IsLiveProperty = DrawnProperty<WaveformView, bool>(nameof(IsLive), false);

    /// <summary>The bars heard.</summary>
    public static readonly BindableProperty BarColorProperty = ColorProperty<WaveformView>(nameof(BarColor));

    /// <summary>The newest bars and the head line.</summary>
    public static readonly BindableProperty HeadColorProperty = ColorProperty<WaveformView>(nameof(HeadColor));

    /// <summary>The bars not heard yet.</summary>
    public static readonly BindableProperty IdleColorProperty = ColorProperty<WaveformView>(nameof(IdleColor));

    private readonly double[] _levels;
    private int _count;
    private int _next;

    /// <summary>A waveform of <paramref name="bars"/> bars in the theme's violet, light sky and idle grey.</summary>
    public WaveformView(int bars)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(bars, 8);
        _levels = new double[bars];
        SetDynamicResource(BarColorProperty, "IdrakViolet");
        SetDynamicResource(HeadColorProperty, "IdrakWaveHead");
        SetDynamicResource(IdleColorProperty, "IdrakWaveIdle");
    }

    /// <summary>A waveform of 58 bars (the design's).</summary>
    public WaveformView()
        : this(58)
    {
    }

    /// <summary>The latest level (0 to 1).</summary>
    public double Level
    {
        get => (double)GetValue(LevelProperty);
        set => SetValue(LevelProperty, value);
    }

    /// <summary>Whether a bound level is pushed.</summary>
    public bool IsLive
    {
        get => (bool)GetValue(IsLiveProperty);
        set => SetValue(IsLiveProperty, value);
    }

    /// <summary>The bars heard.</summary>
    public Color BarColor
    {
        get => (Color)GetValue(BarColorProperty);
        set => SetValue(BarColorProperty, value);
    }

    /// <summary>The newest bars and the head line.</summary>
    public Color HeadColor
    {
        get => (Color)GetValue(HeadColorProperty);
        set => SetValue(HeadColorProperty, value);
    }

    /// <summary>The bars not heard yet.</summary>
    public Color IdleColor
    {
        get => (Color)GetValue(IdleColorProperty);
        set => SetValue(IdleColorProperty, value);
    }

    /// <summary>How many levels are held (at most the bars).</summary>
    public int Count => _count;

    /// <summary>The levels held, oldest first.</summary>
    public double[] Levels()
    {
        var levels = new double[_count];
        int start = (_next - _count + _levels.Length) % _levels.Length;
        for (int i = 0; i < _count; i++)
        {
            levels[i] = _levels[(start + i) % _levels.Length];
        }

        return levels;
    }

    /// <summary>Adds a level heard (clamped to 0 to 1) and redraws.</summary>
    public void Push(double level)
    {
        _levels[_next] = double.IsFinite(level) ? Math.Clamp(level, 0, 1) : 0;
        _next = (_next + 1) % _levels.Length;
        _count = Math.Min(_count + 1, _levels.Length);
        Invalidate();
    }

    /// <summary>Forgets the levels.</summary>
    public void Clear()
    {
        _count = 0;
        _next = 0;
        Invalidate();
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        int bars = _levels.Length;
        float step = dirtyRect.Width / bars, width = Math.Max(2f, step * 0.55f), middle = dirtyRect.Center.Y, half = dirtyRect.Height / 2;
        int start = (_next - _count + bars) % bars;
        const int head = 8;
        for (int i = 0; i < bars; i++)
        {
            bool heard = i < _count;
            double level = heard ? _levels[(start + i) % bars] : 0.25 + 0.2 * Math.Sin(i * 0.9);
            float h = Math.Max(width, (float)(0.12 + 0.88 * level) * half * 2 * 0.92f);
            canvas.FillColor = !heard ? IdleColor : i >= _count - head ? HeadColor : BarColor;
            canvas.FillRoundedRectangle(dirtyRect.Left + i * step + (step - width) / 2, middle - h / 2, width, h, width / 2);
        }

        if (_count > 0 && _count < bars)
        {
            float x = dirtyRect.Left + _count * step;
            canvas.StrokeColor = HeadColor;
            canvas.StrokeSize = 2;
            canvas.DrawLine(x, dirtyRect.Top + 4, x, dirtyRect.Bottom - 4);
        }
    }

    private void OnLevel(double level)
    {
        if (IsLive)
        {
            Push(level);
        }
    }
}

/// <summary>A small line chart of recent values (the chat's live tokens per second), the area under it tinted.</summary>
public sealed class SparklineView : DrawnView
{
    /// <summary>The values, oldest first.</summary>
    public static readonly BindableProperty ValuesProperty = DrawnProperty<SparklineView, IReadOnlyList<double>?>(nameof(Values), null);

    /// <summary>The line's colour.</summary>
    public static readonly BindableProperty LineColorProperty = ColorProperty<SparklineView>(nameof(LineColor));

    /// <summary>A sparkline in the theme's green.</summary>
    public SparklineView()
    {
        SetDynamicResource(LineColorProperty, "IdrakGreen");
        HeightRequest = 58;
    }

    /// <summary>The values, oldest first.</summary>
    public IReadOnlyList<double>? Values
    {
        get => (IReadOnlyList<double>?)GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }

    /// <summary>The line's colour.</summary>
    public Color LineColor
    {
        get => (Color)GetValue(LineColorProperty);
        set => SetValue(LineColorProperty, value);
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        if (Values is not { Count: > 1 } values)
        {
            return;
        }

        double max = 0;
        foreach (double v in values)
        {
            max = Math.Max(max, v);
        }

        max = max <= 0 ? 1 : max * 1.15;
        var line = new PathF();
        var area = new PathF();
        float step = dirtyRect.Width / (values.Count - 1);
        area.MoveTo(dirtyRect.Left, dirtyRect.Bottom);
        for (int i = 0; i < values.Count; i++)
        {
            float x = dirtyRect.Left + i * step, y = dirtyRect.Bottom - (float)(values[i] / max) * dirtyRect.Height;
            if (i == 0)
            {
                line.MoveTo(x, y);
            }
            else
            {
                line.LineTo(x, y);
            }

            area.LineTo(x, y);
        }

        area.LineTo(dirtyRect.Right, dirtyRect.Bottom);
        area.Close();
        canvas.SetFillPaint(new LinearGradientPaint([new PaintGradientStop(0, LineColor.WithAlpha(0.35f)), new PaintGradientStop(1, LineColor.WithAlpha(0f))],
            new Point(0, 0), new Point(0, 1)), dirtyRect);
        canvas.FillPath(area);
        canvas.StrokeColor = LineColor;
        canvas.StrokeSize = 2.2f;
        canvas.StrokeLineJoin = LineJoin.Round;
        canvas.DrawPath(line);
    }
}

/// <summary>A ring that fills clockwise from the top (a pull's progress); indeterminate when <see cref="Progress"/> is below 0.</summary>
public sealed class ProgressRing : DrawnView
{
    /// <summary>The share done, 0 to 1 (below 0: unknown).</summary>
    public static readonly BindableProperty ProgressProperty = DrawnProperty<ProgressRing, double>(nameof(Progress), 0d);

    /// <summary>The track.</summary>
    public static readonly BindableProperty TrackColorProperty = ColorProperty<ProgressRing>(nameof(TrackColor));

    /// <summary>The ring.</summary>
    public static readonly BindableProperty RingColorProperty = ColorProperty<ProgressRing>(nameof(RingColor));

    /// <summary>A ring of 44 units in the theme's accent.</summary>
    public ProgressRing()
    {
        SetDynamicResource(TrackColorProperty, "IdrakLine2");
        SetDynamicResource(RingColorProperty, "IdrakAccent");
        WidthRequest = 44;
        HeightRequest = 44;
    }

    /// <summary>The share done, 0 to 1.</summary>
    public double Progress
    {
        get => (double)GetValue(ProgressProperty);
        set => SetValue(ProgressProperty, value);
    }

    /// <summary>The track.</summary>
    public Color TrackColor
    {
        get => (Color)GetValue(TrackColorProperty);
        set => SetValue(TrackColorProperty, value);
    }

    /// <summary>The ring.</summary>
    public Color RingColor
    {
        get => (Color)GetValue(RingColorProperty);
        set => SetValue(RingColorProperty, value);
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        const float stroke = 4;
        var box = dirtyRect.Inflate(-stroke, -stroke);
        canvas.StrokeSize = stroke;
        canvas.StrokeLineCap = LineCap.Round;
        canvas.StrokeColor = TrackColor;
        canvas.DrawEllipse(box);
        double progress = Progress;
        float sweep = progress < 0 ? 90 : (float)(360 * Math.Clamp(progress, 0, 1));
        if (sweep > 0.5f)
        {
            canvas.StrokeColor = RingColor;
            canvas.DrawArc(box, 90, 90 - sweep, clockwise: true, closed: false);
        }
    }
}

/// <summary>A thin rounded bar filled to <see cref="Progress"/> (memory used, a level), in the theme's green-to-sky gradient.</summary>
public sealed class MeterBar : DrawnView
{
    /// <summary>The share filled, 0 to 1.</summary>
    public static readonly BindableProperty ProgressProperty = DrawnProperty<MeterBar, double>(nameof(Progress), 0d);

    /// <summary>A second share drawn after the first (the chat's key/value cache after the weights); 0 for none.</summary>
    public static readonly BindableProperty SecondProperty = DrawnProperty<MeterBar, double>(nameof(Second), 0d);

    /// <summary>The track.</summary>
    public static readonly BindableProperty TrackColorProperty = ColorProperty<MeterBar>(nameof(TrackColor));

    /// <summary>The fill's start.</summary>
    public static readonly BindableProperty StartColorProperty = ColorProperty<MeterBar>(nameof(StartColor));

    /// <summary>The fill's end (and the second share).</summary>
    public static readonly BindableProperty EndColorProperty = ColorProperty<MeterBar>(nameof(EndColor));

    /// <summary>A meter in the theme's green and sky.</summary>
    public MeterBar()
    {
        SetDynamicResource(TrackColorProperty, "IdrakLine2");
        SetDynamicResource(StartColorProperty, "IdrakGreen");
        SetDynamicResource(EndColorProperty, "IdrakAccent");
        HeightRequest = 4;
    }

    /// <summary>The share filled.</summary>
    public double Progress
    {
        get => (double)GetValue(ProgressProperty);
        set => SetValue(ProgressProperty, value);
    }

    /// <summary>A second share after the first.</summary>
    public double Second
    {
        get => (double)GetValue(SecondProperty);
        set => SetValue(SecondProperty, value);
    }

    /// <summary>The track.</summary>
    public Color TrackColor
    {
        get => (Color)GetValue(TrackColorProperty);
        set => SetValue(TrackColorProperty, value);
    }

    /// <summary>The fill's start.</summary>
    public Color StartColor
    {
        get => (Color)GetValue(StartColorProperty);
        set => SetValue(StartColorProperty, value);
    }

    /// <summary>The fill's end.</summary>
    public Color EndColor
    {
        get => (Color)GetValue(EndColorProperty);
        set => SetValue(EndColorProperty, value);
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        float radius = dirtyRect.Height / 2;
        canvas.FillColor = TrackColor;
        canvas.FillRoundedRectangle(dirtyRect, radius);
        float first = (float)Math.Clamp(double.IsFinite(Progress) ? Progress : 0, 0, 1) * dirtyRect.Width;
        float second = (float)Math.Clamp(double.IsFinite(Second) ? Second : 0, 0, 1) * dirtyRect.Width;
        if (first > 0)
        {
            var fill = new RectF(dirtyRect.Left, dirtyRect.Top, first, dirtyRect.Height);
            if (second > 0)
            {
                canvas.FillColor = StartColor;
            }
            else
            {
                canvas.SetFillPaint(new LinearGradientPaint([new PaintGradientStop(0, StartColor), new PaintGradientStop(1, EndColor)], new Point(0, 0), new Point(1, 0)), fill);
            }

            canvas.FillRoundedRectangle(fill, radius);
        }

        if (second > 0)
        {
            canvas.FillColor = EndColor;
            canvas.FillRoundedRectangle(new RectF(dirtyRect.Left + first, dirtyRect.Top, Math.Min(second, dirtyRect.Width - first), dirtyRect.Height), radius);
        }
    }
}

/// <summary>Idrak's mark (the library's icon without its square): the graph in the icon's sky, amber and white.</summary>
public sealed class LogoMark : DrawnView
{
    /// <summary>The edges.</summary>
    public static readonly BindableProperty EdgeColorProperty = ColorProperty<LogoMark>(nameof(EdgeColor));

    /// <summary>The corner nodes.</summary>
    public static readonly BindableProperty NodeColorProperty = ColorProperty<LogoMark>(nameof(NodeColor));

    /// <summary>The middle node.</summary>
    public static readonly BindableProperty CenterColorProperty = ColorProperty<LogoMark>(nameof(CenterColor));

    /// <summary>Whether the art's larger form is drawn (the carousel's: thinner edges and the middle nodes of the bars).</summary>
    public static readonly BindableProperty IsArtProperty = DrawnProperty<LogoMark, bool>(nameof(IsArt), false);

    /// <summary>The mark in the icon's colours.</summary>
    public LogoMark()
    {
        SetDynamicResource(EdgeColorProperty, "IdrakSky");
        SetDynamicResource(NodeColorProperty, "IdrakAmber");
        SetDynamicResource(CenterColorProperty, "IdrakWhite");
    }

    /// <summary>The edges.</summary>
    public Color EdgeColor
    {
        get => (Color)GetValue(EdgeColorProperty);
        set => SetValue(EdgeColorProperty, value);
    }

    /// <summary>The corner nodes.</summary>
    public Color NodeColor
    {
        get => (Color)GetValue(NodeColorProperty);
        set => SetValue(NodeColorProperty, value);
    }

    /// <summary>The middle node.</summary>
    public Color CenterColor
    {
        get => (Color)GetValue(CenterColorProperty);
        set => SetValue(CenterColorProperty, value);
    }

    /// <summary>Whether the art's larger form is drawn.</summary>
    public bool IsArt
    {
        get => (bool)GetValue(IsArtProperty);
        set => SetValue(IsArtProperty, value);
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        // The icon's 512-unit drawing (assets/icon.svg), scaled into the view.
        float s = Math.Min(dirtyRect.Width, dirtyRect.Height) / 512f, ox = dirtyRect.Center.X - 256 * s, oy = dirtyRect.Center.Y - 256 * s;
        PointF P(float x, float y) => new(ox + x * s, oy + y * s);
        canvas.StrokeColor = EdgeColor;
        canvas.StrokeSize = (IsArt ? 14 : 28) * s;
        canvas.StrokeLineCap = LineCap.Round;
        foreach (var (a, b) in Edges)
        {
            canvas.DrawLine(P(a.X, a.Y), P(b.X, b.Y));
        }

        canvas.FillColor = NodeColor;
        float node = (IsArt ? 30 : 44) * s;
        foreach (var corner in Corners)
        {
            var c = P(corner.X, corner.Y);
            canvas.FillCircle(c.X, c.Y, node);
        }

        canvas.FillColor = CenterColor;
        var middle = P(256, 256);
        canvas.FillCircle(middle.X, middle.Y, (IsArt ? 38 : 54) * s);
        if (IsArt)
        {
            foreach (float y in (float[])[140, 372])
            {
                var m = P(256, y);
                canvas.FillCircle(m.X, m.Y, 24 * s);
            }
        }
    }

    private static readonly (PointF A, PointF B)[] Edges =
    [
        (new(160, 140), new(352, 140)), (new(256, 140), new(256, 372)), (new(160, 372), new(352, 372)), (new(160, 140), new(256, 256)),
        (new(352, 140), new(256, 256)), (new(160, 372), new(256, 256)), (new(352, 372), new(256, 256)),
    ];

    private static readonly PointF[] Corners = [new(160, 140), new(352, 140), new(160, 372), new(352, 372)];
}

/// <summary>
/// A carousel's place as the approved design draws it: a dot a page, the page shown an elongated pill in the accent. Bound
/// to the carousel's <see cref="Position"/> and its page <see cref="Count"/>; the pill slides to the new page when it
/// changes (on screen only), its size from <see cref="DotSize"/>, <see cref="ActiveWidth"/> and <see cref="Spacing"/>.
/// </summary>
public sealed class PageIndicator : DrawnView
{
    /// <summary>The pages.</summary>
    public static readonly BindableProperty CountProperty =
        BindableProperty.Create(nameof(Count), typeof(int), typeof(PageIndicator), 0, propertyChanged: (b, _, _) => ((PageIndicator)b).Resize());

    /// <summary>The page shown, from 0.</summary>
    public static readonly BindableProperty PositionProperty =
        BindableProperty.Create(nameof(Position), typeof(int), typeof(PageIndicator), 0, propertyChanged: (b, o, n) => ((PageIndicator)b).Move((int)o, (int)n));

    /// <summary>A dot's size (its height, and a dot's width).</summary>
    public static readonly BindableProperty DotSizeProperty =
        BindableProperty.Create(nameof(DotSize), typeof(double), typeof(PageIndicator), 6d, propertyChanged: (b, _, _) => ((PageIndicator)b).Resize());

    /// <summary>The pill's width.</summary>
    public static readonly BindableProperty ActiveWidthProperty =
        BindableProperty.Create(nameof(ActiveWidth), typeof(double), typeof(PageIndicator), 22d, propertyChanged: (b, _, _) => ((PageIndicator)b).Resize());

    /// <summary>The space between two dots.</summary>
    public static readonly BindableProperty SpacingProperty =
        BindableProperty.Create(nameof(Spacing), typeof(double), typeof(PageIndicator), 6d, propertyChanged: (b, _, _) => ((PageIndicator)b).Resize());

    /// <summary>The dots' colour.</summary>
    public static readonly BindableProperty DotColorProperty = ColorProperty<PageIndicator>(nameof(DotColor));

    /// <summary>The pill's colour.</summary>
    public static readonly BindableProperty ActiveColorProperty = ColorProperty<PageIndicator>(nameof(ActiveColor));

    private const string SlideAnimation = "IdrakPageIndicator";
    private double _shown;

    /// <summary>An indicator in the theme's line and accent colours.</summary>
    public PageIndicator()
    {
        SetDynamicResource(DotColorProperty, "IdrakLine2");
        SetDynamicResource(ActiveColorProperty, "IdrakAccent");
        Resize();
    }

    /// <summary>The pages.</summary>
    public int Count
    {
        get => (int)GetValue(CountProperty);
        set => SetValue(CountProperty, value);
    }

    /// <summary>The page shown.</summary>
    public int Position
    {
        get => (int)GetValue(PositionProperty);
        set => SetValue(PositionProperty, value);
    }

    /// <summary>A dot's size.</summary>
    public double DotSize
    {
        get => (double)GetValue(DotSizeProperty);
        set => SetValue(DotSizeProperty, value);
    }

    /// <summary>The pill's width.</summary>
    public double ActiveWidth
    {
        get => (double)GetValue(ActiveWidthProperty);
        set => SetValue(ActiveWidthProperty, value);
    }

    /// <summary>The space between two dots.</summary>
    public double Spacing
    {
        get => (double)GetValue(SpacingProperty);
        set => SetValue(SpacingProperty, value);
    }

    /// <summary>The dots' colour.</summary>
    public Color DotColor
    {
        get => (Color)GetValue(DotColorProperty);
        set => SetValue(DotColorProperty, value);
    }

    /// <summary>The pill's colour.</summary>
    public Color ActiveColor
    {
        get => (Color)GetValue(ActiveColorProperty);
        set => SetValue(ActiveColorProperty, value);
    }

    /// <summary>Where the pill is drawn now, in pages (between two pages while it slides).</summary>
    public double Shown => _shown;

    /// <summary>The width of the dot of page <paramref name="index"/> when the pill is at <paramref name="shown"/>: the pill's at it, a dot's a page away or more, between them in between.</summary>
    public double WidthOf(int index, double shown) => DotSize + (ActiveWidth - DotSize) * Math.Max(0, 1 - Math.Abs(index - shown));

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        int count = Count;
        if (count <= 0)
        {
            return;
        }

        float size = (float)DotSize, radius = size / 2, top = dirtyRect.Center.Y - radius, x = dirtyRect.Left;
        for (int i = 0; i < count; i++)
        {
            double width = WidthOf(i, _shown);
            double share = Math.Max(0, 1 - Math.Abs(i - _shown));
            canvas.FillColor = share <= 0 ? DotColor : share >= 1 ? ActiveColor : Blend(DotColor, ActiveColor, (float)share);
            canvas.FillRoundedRectangle(x, top, (float)width, size, radius);
            x += (float)(width + Spacing);
        }
    }

    private static Color Blend(Color from, Color to, float share) =>
        new(from.Red + (to.Red - from.Red) * share, from.Green + (to.Green - from.Green) * share, from.Blue + (to.Blue - from.Blue) * share, from.Alpha + (to.Alpha - from.Alpha) * share);

    // Its width: every dot, the pill and the spaces between.
    private void Resize()
    {
        int count = Math.Max(0, Count);
        WidthRequest = count == 0 ? 0 : (count - 1) * (DotSize + Spacing) + ActiveWidth;
        HeightRequest = DotSize;
        Invalidate();
    }

    // The pill slides from the page it was at to the new one on screen; off screen it is there at once.
    private void Move(int from, int to)
    {
        this.AbortAnimation(SlideAnimation);
        if (!Motion.CanAnimate(this) || from == to)
        {
            _shown = to;
            Invalidate();
            return;
        }

        double start = _shown;
        this.Animate(SlideAnimation, t =>
        {
            _shown = start + (to - start) * t;
            Invalidate();
        }, length: 220, easing: Easing.CubicOut, finished: (_, _) =>
        {
            _shown = to;
            Invalidate();
        });
    }
}
