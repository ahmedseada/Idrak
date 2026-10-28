// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Microsoft.Maui.Graphics;

namespace Idrak.Maui.Controls;

/// <summary>
/// The splash's motion at one moment of the approved design (plan 34, phase S; its keyframes are a six-second loop, a
/// keyframe at p% falling at p × 0.06 s, played once here): the four signals along the edges into the core, the amber
/// pulse, the core's swell with its bloom and two rings, the signal up and down the axis, the mark's lift (a glide to the
/// left on a wide window), then the name, the second name and the line under them, and the foot (the status line and the
/// bar). Made by <see cref="At"/> from the design's time alone: no allocation, the same values for the same time.
/// </summary>
public readonly struct SplashFrame
{
    /// <summary>The design's loop, in seconds: a keyframe at p% is at p × <see cref="Loop"/> / 100.</summary>
    public const double Loop = 6;

    /// <summary>The design's time of a keyframe at <paramref name="percent"/>% of the loop, in seconds.</summary>
    public static double Time(double percent) => percent * Loop / 100;

    /// <summary>When the foot (the status line and the bar) starts to show: 49% (2.94 s).</summary>
    public static double FootStart => Time(49);

    /// <summary>When the last of the name settles (57%, 3.42 s): the animation's end; past it nothing moves but the status line.</summary>
    public static double IntroEnd => Time(57);

    private static readonly CubicBezier EaseOut = new(0, 0, 0.58, 1);
    private static readonly CubicBezier Glide = new(0.2, 0.8, 0.2, 1);
    private static readonly CubicBezier Spread = new(0.1, 0.6, 0.3, 1);

    /// <summary>The top pair's signal: its dash offset along the edge (16: not yet out of the node, -100: into the core), in the edge's 100 units.</summary>
    public double SignalTop { get; init; }

    /// <summary>The bottom pair's signal, as <see cref="SignalTop"/>.</summary>
    public double SignalBottom { get; init; }

    /// <summary>The axis' signal (out of the core up and down to the tips), as <see cref="SignalTop"/>.</summary>
    public double SignalAxis { get; init; }

    /// <summary>The amber nodes' scale (their pulse: 1, 1.2, 1).</summary>
    public double NodeScale { get; init; }

    /// <summary>The core's scale (its swell: 1, 1.18, 1).</summary>
    public double CoreScale { get; init; }

    /// <summary>The bloom's opacity (0 to 0.6).</summary>
    public double BloomOpacity { get; init; }

    /// <summary>The bloom's scale (1 to 2.8).</summary>
    public double BloomScale { get; init; }

    /// <summary>How far the tips are lit in the flash colour (0 to 1).</summary>
    public double Flash { get; init; }

    /// <summary>The first ring's opacity, at the ring colour's own (0 to 0.75).</summary>
    public double Ring1Opacity { get; init; }

    /// <summary>The first ring's scale of the mark's size.</summary>
    public double Ring1Scale { get; init; }

    /// <summary>The second ring's opacity (0 to 0.5).</summary>
    public double Ring2Opacity { get; init; }

    /// <summary>The second ring's scale of the mark's size.</summary>
    public double Ring2Scale { get; init; }

    /// <summary>How far the mark has lifted (a narrow window) or glided left (a wide one), 0 to 1.</summary>
    public double Lift { get; init; }

    /// <summary>The name's opacity.</summary>
    public double NameOpacity { get; init; }

    /// <summary>The name's shift into place, in the design's units (down on a narrow window, left on a wide one; 0 settled).</summary>
    public double NameShift { get; init; }

    /// <summary>The second name's and the line's opacity.</summary>
    public double SubOpacity { get; init; }

    /// <summary>The second name's and the line's shift into place, as <see cref="NameShift"/>.</summary>
    public double SubShift { get; init; }

    /// <summary>The foot's opacity (the status line and the bar).</summary>
    public double FootOpacity { get; init; }

    /// <summary>The motion at the design's time <paramref name="design"/> (seconds) on a narrow (stacked) or <paramref name="wide"/> (side by side) window.</summary>
    public static SplashFrame At(double design, bool wide)
    {
        double d = double.IsFinite(design) ? design : 0;
        return new SplashFrame
        {
            SignalTop = Key(d, 10, 23, 16, -100, null),
            SignalBottom = Key(d, 13, 26, 16, -100, null),
            SignalAxis = Key(d, 32, 43, 16, -100, null),
            NodeScale = d < Time(12) ? Key(d, 9, 12, 1, 1.2, EaseOut) : Key(d, 12, 17, 1.2, 1, EaseOut),
            CoreScale = d < Time(29) ? Key(d, 25, 29, 1, 1.18, EaseOut) : Key(d, 29, 36, 1.18, 1, EaseOut),
            BloomOpacity = d < Time(30) ? Key(d, 24, 30, 0, 0.6, EaseOut) : Key(d, 30, 44, 0.6, 0, EaseOut),
            BloomScale = d < Time(30) ? Key(d, 24, 30, 1, 1.7, EaseOut) : Key(d, 30, 44, 1.7, 2.8, EaseOut),
            Flash = d < Time(43) ? Key(d, 40, 43, 0, 1, EaseOut) : Key(d, 43, 50, 1, 0, EaseOut),
            Ring1Opacity = d < Time(29) ? Key(d, 27, 29, 0, 0.75, Spread) : Key(d, 29, 48, 0.75, 0, Spread),
            Ring1Scale = Key(d, 27, 48, 0.45, wide ? 3.2 : 2.9, Spread),
            Ring2Opacity = d < Time(34) ? Key(d, 32, 34, 0, 0.5, Spread) : Key(d, 34, 54, 0.5, 0, Spread),
            Ring2Scale = Key(d, 32, 54, 0.45, wide ? 3.8 : 3.4, Spread),
            Lift = Key(d, 34, 46, 0, 1, Glide),
            NameOpacity = wide ? Key(d, 43, 52, 0, 1, Glide) : Key(d, 40, 50, 0, 1, Glide),
            NameShift = wide ? Key(d, 43, 52, -16, 0, Glide) : Key(d, 40, 50, 14, 0, Glide),
            SubOpacity = wide ? Key(d, 47, 57, 0, 1, Glide) : Key(d, 45, 56, 0, 1, Glide),
            SubShift = wide ? Key(d, 47, 57, -12, 0, Glide) : Key(d, 45, 56, 10, 0, Glide),
            FootOpacity = Key(d, 49, 54, 0, 1, EaseOut),
        };
    }

    // One keyframe segment: from at the keyframe "from"% to "to" at the keyframe to%, eased (null: linear), held outside it.
    private static double Key(double d, double fromPercent, double toPercent, double from, double to, CubicBezier? ease)
    {
        double t0 = Time(fromPercent), t1 = Time(toPercent);
        if (d <= t0)
        {
            return from;
        }

        if (d >= t1)
        {
            return to;
        }

        double x = (d - t0) / (t1 - t0);
        return from + (to - from) * (ease is { } e ? e.Ease(x) : x);
    }

    // CSS's cubic-bezier timing: the curve's y where its x is the time's share (Newton's steps, then halving where they stall).
    private readonly struct CubicBezier(double x1, double y1, double x2, double y2)
    {
        public double Ease(double x)
        {
            double t = x;
            for (int i = 0; i < 8; i++)
            {
                double error = Curve(t, x1, x2) - x;
                if (Math.Abs(error) < 1e-7)
                {
                    return Curve(t, y1, y2);
                }

                double slope = Slope(t, x1, x2);
                if (Math.Abs(slope) < 1e-6)
                {
                    break;
                }

                t -= error / slope;
            }

            double low = 0, high = 1;
            t = x;
            for (int i = 0; i < 40; i++)
            {
                double value = Curve(t, x1, x2);
                if (Math.Abs(value - x) < 1e-7)
                {
                    break;
                }

                if (value < x)
                {
                    low = t;
                }
                else
                {
                    high = t;
                }

                t = (low + high) / 2;
            }

            return Curve(t, y1, y2);
        }

        private static double Curve(double t, double p1, double p2)
        {
            double u = 1 - t;
            return 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t;
        }

        private static double Slope(double t, double p1, double p2)
        {
            double u = 1 - t;
            return 3 * u * u * p1 + 6 * u * t * (p2 - p1) + 3 * t * t * (1 - p2);
        }
    }
}

/// <summary>
/// The splash's time, its status line and its hand-over, from the timestamps it is given (one a display frame) and the
/// app's <see cref="StartupRun"/>; it allocates nothing after construction. The design's time (<see cref="Design"/>) runs
/// with the clock up to the animation's end (<see cref="SplashFrame.IntroEnd"/>). The status line shows the start-up
/// steps in the order they began, each at least <see cref="IdrakMauiOptions.SplashStepTime"/> once the foot shows (a step
/// that began and ended before is not shown after the fact), then the Ready line; the bar is the share of the steps ended.
/// The hand-over: <see cref="SplashLength.Short"/> starts it the moment the start-up is done, running ahead to the name over
/// <see cref="IdrakMauiOptions.SplashCutAhead"/> while the stage fades over <see cref="IdrakMauiOptions.SplashFade"/>;
/// <see cref="SplashLength.Full"/> waits for the animation's end and the Ready line's least time, then fades. Past
/// <see cref="IdrakMauiOptions.SplashLongest"/> it hands over whatever still runs.
/// </summary>
public sealed class SplashTimeline
{
    /// <summary>How long a new status line takes to rise in (the design's 3% of the loop).</summary>
    public const double LineInSeconds = 0.18;

    /// <summary>How long the line it replaces takes to leave (the design's 2%).</summary>
    public const double LineOutSeconds = 0.12;

    // The bar follows the share of the steps ended with this time constant.
    private const double FillSeconds = 0.12;

    private readonly StartupRun _run;
    private readonly SplashLength _length;
    private readonly double _frequency, _step, _fade, _cut, _longest, _cross;
    private bool _started;
    private long _start, _last, _shownAt, _since, _handStart;
    private double _handFrom;
    private int _position = -1, _previous = -1;
    private bool _over;                                                         // the line's step ended before the foot showed: not shown after the fact

    /// <summary>A timeline over <paramref name="run"/> as <paramref name="options"/> time it, its timestamps <paramref name="frequency"/> a second.</summary>
    public SplashTimeline(StartupRun run, IdrakMauiOptions options, long frequency)
    {
        _run = run ?? throw new ArgumentNullException(nameof(run));
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frequency);
        _length = options.Splash;
        _frequency = frequency;
        _step = Seconds(options.SplashStepTime);
        _fade = Seconds(options.SplashFade);
        _cut = Seconds(options.SplashCutAhead);
        _longest = options.SplashLongest > TimeSpan.Zero ? options.SplashLongest.TotalSeconds : double.PositiveInfinity;
        _cross = Seconds(options.SplashThemeCross);
    }

    /// <summary>The run it shows.</summary>
    public StartupRun Run => _run;

    /// <summary>How much of the animation plays.</summary>
    public SplashLength Length => _length;

    /// <summary>Whether the first frame was taken.</summary>
    public bool IsStarted => _started;

    /// <summary>Seconds since the first frame.</summary>
    public double Elapsed { get; private set; }

    /// <summary>The design's time now, in seconds (it stops at <see cref="SplashFrame.IntroEnd"/>).</summary>
    public double Design { get; private set; }

    /// <summary>The stage's opacity: 1 until the hand-over's fade, then down to 0.</summary>
    public double StageAlpha { get; private set; } = 1;

    /// <summary>How far the ground has crossed from the system's theme to the app's (1: the app's), where they differ.</summary>
    public double ThemeCross { get; private set; } = 1;

    /// <summary>Whether the hand-over (the cut ahead and the fade into Home) has begun.</summary>
    public bool IsHandingOver { get; private set; }

    /// <summary>Whether the splash is over: faded out, the host's page shown.</summary>
    public bool IsFinished { get; private set; }

    /// <summary>The run's entry the status line shows (see <see cref="StartupRun.Title"/>), or -1 (none, or the Ready line).</summary>
    public int LineEntry => _position >= 0 && !IsReadyShown && !_over ? _run.StartedEntry(_position) : -1;

    /// <summary>The entry leaving the status line (rising out as the next comes in), or -1.</summary>
    public int PreviousEntry => _previous >= 0 ? _run.StartedEntry(_previous) : -1;

    /// <summary>Whether the status line is the Ready line.</summary>
    public bool IsReadyShown { get; private set; }

    /// <summary>The share of the start-up steps ended, as the bar draws it (it follows the share smoothly).</summary>
    public double Fill { get; private set; }

    /// <summary>How far the status line's newest line has risen in, 0 to 1.</summary>
    public double LineIn { get; private set; } = 1;

    /// <summary>How far the line it replaced has left, 0 to 1 (1: gone).</summary>
    public double LineOut { get; private set; } = 1;

    /// <summary>How many lines the status line has shown so far (the steps and Ready), for the tests and the telemetry of a run.</summary>
    public int LinesShown { get; private set; }

    /// <summary>Takes the first frame at <paramref name="now"/> (a timestamp); <see cref="Update"/> does when it was not.</summary>
    public void Start(long now)
    {
        _started = true;
        _start = now;
        _last = now;
        _shownAt = now;
        _since = now;
    }

    /// <summary>Moves the splash to <paramref name="now"/> (a timestamp, one a display frame); false once it is finished.</summary>
    public bool Update(long now)
    {
        if (!_started)
        {
            Start(now);
        }

        if (IsFinished)
        {
            return false;
        }

        double dt = Math.Max(0, (now - _last) / _frequency);
        _last = now;
        Elapsed = Math.Max(0, (now - _start) / _frequency);
        ThemeCross = _cross > 0 ? Math.Clamp(Elapsed / _cross, 0, 1) : 1;
        bool done = _run.IsDone;
        if (!IsHandingOver)
        {
            Design = Math.Min(Elapsed, SplashFrame.IntroEnd);
            if ((_length == SplashLength.Short && done) || Elapsed >= _longest)
            {
                HandOver(now);
            }
        }

        if (IsHandingOver)
        {
            double h = Math.Max(0, (now - _handStart) / _frequency);
            double target = Math.Max(_handFrom, SplashFrame.IntroEnd);
            Design = _cut > 0 ? _handFrom + (target - _handFrom) * Math.Min(1, h / _cut) : target;
            StageAlpha = _fade > 0 ? Math.Clamp(1 - h / _fade, 0, 1) : 0;
        }

        UpdateLine(now, done);
        if (!IsHandingOver && _length == SplashLength.Full && done && Design >= SplashFrame.IntroEnd && IsReadyShown && (now - _shownAt) / _frequency >= _step)
        {
            HandOver(now);
            StageAlpha = _fade > 0 ? 1 : 0;
        }

        UpdateFill(dt, done);
        if (IsHandingOver && StageAlpha <= 0 && Design >= Math.Max(_handFrom, SplashFrame.IntroEnd))
        {
            IsFinished = true;
        }

        return !IsFinished;
    }

    private void HandOver(long now)
    {
        IsHandingOver = true;
        _handStart = now;
        _handFrom = Design;
    }

    // The status line: while the foot is not shown it tracks the newest step at once (nothing is on screen to pace); once
    // shown, each step that began stays its least time before the next, then Ready (at once when the short splash hands over).
    private void UpdateLine(long now, bool done)
    {
        int latest = _run.Started - 1;
        if (Design < SplashFrame.FootStart)
        {
            _position = latest;
            _over = latest >= 0 && _run.State(_run.StartedEntry(latest)) is StartupStepState.Done or StartupStepState.Failed;
            _previous = -1;
            IsReadyShown = done;
            _shownAt = now;
            _since = long.MinValue;
            LineIn = 1;
            LineOut = 1;
            return;
        }

        if (_since == long.MinValue)
        {
            _since = now;                                                     // the foot came: its line counts from now
            _shownAt = now;
            LinesShown += (_position >= 0 && !_over) || IsReadyShown ? 1 : 0;
        }

        if (!IsReadyShown)
        {
            bool held = _position < 0 || _over || (now - _shownAt) / _frequency >= _step;
            if (IsHandingOver && done)
            {
                Change(now, ready: true);
            }
            else if (held && _position < latest)
            {
                Change(now, ready: false);
            }
            else if (held && done)
            {
                Change(now, ready: true);
            }
        }

        double since = Math.Max(0, (now - _since) / _frequency);
        LineIn = Math.Clamp(since / LineInSeconds, 0, 1);
        LineOut = Math.Clamp(since / LineOutSeconds, 0, 1);
        if (LineOut >= 1)
        {
            _previous = -1;
        }
    }

    private void Change(long now, bool ready)
    {
        _previous = _over ? -1 : _position;
        _over = false;
        if (ready)
        {
            IsReadyShown = true;
        }
        else
        {
            _position++;
        }

        _shownAt = now;
        _since = now;
        LinesShown++;
    }

    private void UpdateFill(double dt, bool done)
    {
        int count = _run.Count;
        double target = count == 0 ? (done ? 1 : 0) : (double)_run.Finished / count;
        if (Design < SplashFrame.FootStart)
        {
            Fill = 0;
            return;
        }

        Fill += (target - Fill) * (1 - Math.Exp(-dt / FillSeconds));
        if (Math.Abs(target - Fill) < 1e-4)
        {
            Fill = target;
        }
    }

    private static double Seconds(TimeSpan time) => time > TimeSpan.Zero ? time.TotalSeconds : 0;
}

/// <summary>
/// The animated splash (plan 34, phase S): Idrak's mark drawn from the icon's own geometry (<c>assets/icon.svg</c>, its
/// 512-unit drawing), the name block, then the app's real start-up steps over a thin bar (<see cref="StartupRun"/>), timed
/// by <see cref="SplashTimeline"/> and drawn once a display frame (<see cref="FrameClocks.Current"/>) while on screen. A
/// narrow window stacks it (the mark lifts, the names under it, the foot at the bottom); a wide one (from
/// <see cref="Breakpoint"/>) sets the names beside the mark as it glides left. Dark: the bare mark on its ground; light: the
/// icon's tile with a soft shadow on the light ground; every colour and font is a theme key. A frame allocates nothing:
/// the colours and fonts are kept until the theme changes them, and the text widths are measured once a line.
/// <see cref="Finished"/> is raised once, when it has faded out.
/// </summary>
public sealed class SplashView : DrawnView
{
    /// <summary>The ground (the system splash's).</summary>
    public static readonly BindableProperty GroundColorProperty = Colour(nameof(GroundColor));

    /// <summary>The icon's tile (clear: none, the bare mark).</summary>
    public static readonly BindableProperty TileColorProperty = Colour(nameof(TileColor));

    /// <summary>The tile's stroke.</summary>
    public static readonly BindableProperty TileStrokeColorProperty = Colour(nameof(TileStrokeColor));

    /// <summary>The tile's shadow.</summary>
    public static readonly BindableProperty ShadowColorProperty = Colour(nameof(ShadowColor));

    /// <summary>The edges and the bloom (the icon's sky).</summary>
    public static readonly BindableProperty EdgeColorProperty = Colour(nameof(EdgeColor));

    /// <summary>The corner nodes (the icon's amber).</summary>
    public static readonly BindableProperty NodeColorProperty = Colour(nameof(NodeColor));

    /// <summary>The core and the tips (the icon's white).</summary>
    public static readonly BindableProperty CoreColorProperty = Colour(nameof(CoreColor));

    /// <summary>The signals' light.</summary>
    public static readonly BindableProperty SignalColorProperty = Colour(nameof(SignalColor));

    /// <summary>The tips' flash.</summary>
    public static readonly BindableProperty FlashColorProperty = Colour(nameof(FlashColor));

    /// <summary>The rings.</summary>
    public static readonly BindableProperty RingColorProperty = Colour(nameof(RingColor));

    /// <summary>The name.</summary>
    public static readonly BindableProperty NameColorProperty = Colour(nameof(NameColor));

    /// <summary>The second name.</summary>
    public static readonly BindableProperty NativeColorProperty = Colour(nameof(NativeColor));

    /// <summary>The line under the names and the status line.</summary>
    public static readonly BindableProperty MutedColorProperty = Colour(nameof(MutedColor));

    /// <summary>A running step's dot and the bar's fill.</summary>
    public static readonly BindableProperty AccentColorProperty = Colour(nameof(AccentColor));

    /// <summary>The Ready line's dot.</summary>
    public static readonly BindableProperty ReadyColorProperty = Colour(nameof(ReadyColor));

    /// <summary>The bar's track.</summary>
    public static readonly BindableProperty TrackColorProperty = Colour(nameof(TrackColor));

    /// <summary>The name's font alias.</summary>
    public static readonly BindableProperty NameFontProperty = FontAlias(nameof(NameFont));

    /// <summary>The second name's font alias (the Arabic script's; the system font where the app registers none).</summary>
    public static readonly BindableProperty NativeFontProperty = FontAlias(nameof(NativeFont));

    /// <summary>The line's font alias.</summary>
    public static readonly BindableProperty TaglineFontProperty = FontAlias(nameof(TaglineFont));

    /// <summary>The status line's font alias.</summary>
    public static readonly BindableProperty StatusFontProperty = FontAlias(nameof(StatusFont));

    // The icon's 512-unit geometry (assets/icon.svg): the edges, the signals' paths, the nodes and the tile.
    private static readonly float[] EdgeLines =
    [
        160, 140, 256, 140, 256, 140, 352, 140, 256, 140, 256, 372, 160, 372, 256, 372, 256, 372, 352, 372,
        160, 140, 256, 256, 352, 140, 256, 256, 160, 372, 256, 256, 352, 372, 256, 256,
    ];

    private static readonly float[] TopSignals = [160, 140, 256, 256, 352, 140, 256, 256];
    private static readonly float[] BottomSignals = [160, 372, 256, 256, 352, 372, 256, 256];
    private static readonly float[] AxisSignals = [256, 256, 256, 140, 256, 256, 256, 372];
    private static readonly float[] Corners = [160, 140, 352, 140, 160, 372, 352, 372];
    private const float EdgeWidth = 16, SignalWidth = 20, SignalLength = 16, CornerRadius = 34, TipRadius = 30, CoreRadius = 44;
    private const float TileInset = 6, TileSize = 500, TileRound = 108, TileStroke = 12, EdgeOpacity = 0.9f;

    private readonly TimeProvider _time;
    private readonly IdrakMauiOptions _options;
    private Color? _edgeOnGround;
    private Color? _fromGround, _fromTile, _fromTileStroke, _fromShadow;
    private IFont _nameFont = Microsoft.Maui.Graphics.Font.Default, _nativeFont = Microsoft.Maui.Graphics.Font.Default;
    private IFont _taglineFont = Microsoft.Maui.Graphics.Font.Default, _statusFont = Microsoft.Maui.Graphics.Font.Default;
    private bool _coloursStale = true, _fontsStale = true, _systemFonts, _raised, _cleared;
    private float[] _lineWidths = [];
    private float _readyWidth = -1, _nativeWidth = -1, _measuredAt = -1;
    private PointF _centreShift;

    /// <summary>A splash showing <paramref name="run"/>, timed by <paramref name="options"/>, its clock <paramref name="time"/> (the system's by default).</summary>
    public SplashView(StartupRun run, IdrakMauiOptions options, TimeProvider? time = null)
    {
        ArgumentNullException.ThrowIfNull(run);
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _time = time ?? TimeProvider.System;
        Timeline = new SplashTimeline(run, options, _time.TimestampFrequency);
        Words = options.SplashWords ?? new SplashWords();
        Breakpoint = options.WideLayoutWidth;
        InputTransparent = false;                                               // the page under it takes no tap meanwhile
        SetDynamicResource(BackgroundColorProperty, "IdrakSplashGround");        // the ground from the first frame, before the canvas draws
        SetDynamicResource(GroundColorProperty, "IdrakSplashGround");
        SetDynamicResource(TileColorProperty, "IdrakSplashTile");
        SetDynamicResource(TileStrokeColorProperty, "IdrakSplashTileStroke");
        SetDynamicResource(ShadowColorProperty, "IdrakSplashShadow");
        SetDynamicResource(EdgeColorProperty, "IdrakSky");
        SetDynamicResource(NodeColorProperty, "IdrakAmber");
        SetDynamicResource(CoreColorProperty, "IdrakWhite");
        SetDynamicResource(SignalColorProperty, "IdrakSplashSignal");
        SetDynamicResource(FlashColorProperty, "IdrakSplashFlash");
        SetDynamicResource(RingColorProperty, "IdrakSplashRing");
        SetDynamicResource(NameColorProperty, "IdrakText");
        SetDynamicResource(NativeColorProperty, "IdrakAccentText");
        SetDynamicResource(MutedColorProperty, "IdrakTextMuted");
        SetDynamicResource(AccentColorProperty, "IdrakAccent");
        SetDynamicResource(ReadyColorProperty, "IdrakSuccess");
        SetDynamicResource(TrackColorProperty, "IdrakSplashTrack");
        SetDynamicResource(NameFontProperty, "IdrakFontExtraBold");
        SetDynamicResource(NativeFontProperty, "IdrakFontArabicSemiBold");
        SetDynamicResource(TaglineFontProperty, "IdrakFontMedium");
        SetDynamicResource(StatusFontProperty, "IdrakFontMono");
        SemanticProperties.SetDescription(this, Words.Name);
        HandlerChanged += (_, _) => _fontsStale = true;                       // the platform's fonts are reached through the handler
        SizeChanged += (_, _) => MeasureCentre();
    }

    // Where the window's middle is from this view's: the system's splash centres its picture in the whole window, this view
    // sits inside the page (under the status bar's inset). Measured when its size changes, not a frame.
    private void MeasureCentre()
    {
#if ANDROID
        if (Handler?.PlatformView is global::Android.Views.View view && view.RootView is { Width: > 0, Height: > 0 } root && view.Width > 0)
        {
            int[] at = new int[2];
            view.GetLocationInWindow(at);
            float density = view.Resources?.DisplayMetrics?.Density ?? 1;
            _centreShift = new PointF((root.Width / 2f - (at[0] + view.Width / 2f)) / density, (root.Height / 2f - (at[1] + view.Height / 2f)) / density);
        }
#endif
    }

    /// <summary>Raised once, when the splash has faded out (the host takes it away).</summary>
    public event EventHandler? Finished;

    /// <summary>Raised once, when the hand-over begins (the host gives its page its own ground back).</summary>
    public event EventHandler? HandingOver;

    /// <summary>The time, the status line and the hand-over.</summary>
    public SplashTimeline Timeline { get; }

    /// <summary>The words it shows.</summary>
    public SplashWords Words { get; }

    /// <summary>The window width from which the names sit beside the mark (the app's wide layout, <see cref="IdrakMauiOptions.WideLayoutWidth"/>).</summary>
    public double Breakpoint { get; set; }

    /// <summary>Whether the last frame drawn was the wide (side by side) layout.</summary>
    public bool DrewWide { get; private set; }

    /// <summary>
    /// The size, in device-independent units, of the mark's 512-unit box at the first frame: the system splash's picture as
    /// it drew the mark (so the two meet without a jump); it settles to the design's size as the mark lifts. 0: the design's.
    /// </summary>
    public double StartSize { get; set; }

    /// <summary>How far the window's middle is from this view's (where the system's splash centred the mark), in units; it eases away as the mark lifts.</summary>
    public PointF CentreShift
    {
        get => _centreShift;
        set => _centreShift = value;
    }

    /// <summary>How many frames it has drawn.</summary>
    public long FramesDrawn { get; private set; }

    /// <summary>The ground.</summary>
    public Color GroundColor { get => (Color)GetValue(GroundColorProperty); set => SetValue(GroundColorProperty, value); }

    /// <summary>The icon's tile.</summary>
    public Color TileColor { get => (Color)GetValue(TileColorProperty); set => SetValue(TileColorProperty, value); }

    /// <summary>The tile's stroke.</summary>
    public Color TileStrokeColor { get => (Color)GetValue(TileStrokeColorProperty); set => SetValue(TileStrokeColorProperty, value); }

    /// <summary>The tile's shadow.</summary>
    public Color ShadowColor { get => (Color)GetValue(ShadowColorProperty); set => SetValue(ShadowColorProperty, value); }

    /// <summary>The edges and the bloom.</summary>
    public Color EdgeColor { get => (Color)GetValue(EdgeColorProperty); set => SetValue(EdgeColorProperty, value); }

    /// <summary>The corner nodes.</summary>
    public Color NodeColor { get => (Color)GetValue(NodeColorProperty); set => SetValue(NodeColorProperty, value); }

    /// <summary>The core and the tips.</summary>
    public Color CoreColor { get => (Color)GetValue(CoreColorProperty); set => SetValue(CoreColorProperty, value); }

    /// <summary>The signals' light.</summary>
    public Color SignalColor { get => (Color)GetValue(SignalColorProperty); set => SetValue(SignalColorProperty, value); }

    /// <summary>The tips' flash.</summary>
    public Color FlashColor { get => (Color)GetValue(FlashColorProperty); set => SetValue(FlashColorProperty, value); }

    /// <summary>The rings.</summary>
    public Color RingColor { get => (Color)GetValue(RingColorProperty); set => SetValue(RingColorProperty, value); }

    /// <summary>The name.</summary>
    public Color NameColor { get => (Color)GetValue(NameColorProperty); set => SetValue(NameColorProperty, value); }

    /// <summary>The second name.</summary>
    public Color NativeColor { get => (Color)GetValue(NativeColorProperty); set => SetValue(NativeColorProperty, value); }

    /// <summary>The line under the names and the status line.</summary>
    public Color MutedColor { get => (Color)GetValue(MutedColorProperty); set => SetValue(MutedColorProperty, value); }

    /// <summary>A running step's dot and the bar's fill.</summary>
    public Color AccentColor { get => (Color)GetValue(AccentColorProperty); set => SetValue(AccentColorProperty, value); }

    /// <summary>The Ready line's dot.</summary>
    public Color ReadyColor { get => (Color)GetValue(ReadyColorProperty); set => SetValue(ReadyColorProperty, value); }

    /// <summary>The bar's track.</summary>
    public Color TrackColor { get => (Color)GetValue(TrackColorProperty); set => SetValue(TrackColorProperty, value); }

    /// <summary>The name's font alias.</summary>
    public string NameFont { get => (string)GetValue(NameFontProperty); set => SetValue(NameFontProperty, value); }

    /// <summary>The second name's font alias.</summary>
    public string NativeFont { get => (string)GetValue(NativeFontProperty); set => SetValue(NativeFontProperty, value); }

    /// <summary>The line's font alias.</summary>
    public string TaglineFont { get => (string)GetValue(TaglineFontProperty); set => SetValue(TaglineFontProperty, value); }

    /// <summary>The status line's font alias.</summary>
    public string StatusFont { get => (string)GetValue(StatusFontProperty); set => SetValue(StatusFontProperty, value); }

    /// <summary>
    /// Starts on the system theme's ground (and its tile) and crosses to this view's own over
    /// <see cref="IdrakMauiOptions.SplashThemeCross"/>: where the app's theme setting forces the other one, the system's
    /// splash (drawn in the system's theme) meets this one without a jump.
    /// </summary>
    public void CrossFrom(Color ground, Color tile, Color tileStroke, Color shadow)
    {
        _fromGround = ground ?? throw new ArgumentNullException(nameof(ground));
        _fromTile = tile;
        _fromTileStroke = tileStroke;
        _fromShadow = shadow;
        BackgroundColor = ground;                                               // the first frame's ground, before the canvas draws
    }

    /// <inheritdoc />
    protected override bool WantsFrames => !_raised;

    /// <inheritdoc />
    protected override void OnFrame() => NextFrame();

    /// <summary>One display frame: the timeline moved to now and the view redrawn; the events raised as the hand-over begins and ends.</summary>
    public void NextFrame() => Advance(_time.GetTimestamp());

    /// <summary>Moves the splash to <paramref name="timestamp"/> (of its clock) and redraws it (what <see cref="NextFrame"/> does at now).</summary>
    public void Advance(long timestamp)
    {
        if (_raised)
        {
            return;
        }

        bool running = Timeline.Update(timestamp);
        if (Timeline.IsHandingOver && !_cleared)
        {
            _cleared = true;
            BackgroundColor = Colors.Transparent;                               // once: the canvas draws the fading ground from here
            HandingOver?.Invoke(this, EventArgs.Empty);
        }

        if (!running)
        {
            _raised = true;
            UpdateTicking();
            Finished?.Invoke(this, EventArgs.Empty);
            return;
        }

        Invalidate();
    }

    /// <inheritdoc />
    public override void Draw(ICanvas canvas, RectF dirtyRect)
    {
        ArgumentNullException.ThrowIfNull(canvas);
        try
        {
            DrawFrame(canvas, dirtyRect);
        }
        catch (Exception e) when (e is not OutOfMemoryException && !_systemFonts)
        {
            // A font the platform's canvas refuses (Win2D given a font address an unpackaged Windows app cannot load) would end
            // the app from inside the drawing: the system's fonts from now on, this one frame dropped.
            _systemFonts = true;
            _fontsStale = true;
        }
    }

    private void DrawFrame(ICanvas canvas, RectF dirtyRect)
    {
        var timeline = Timeline;
        if (!timeline.IsStarted || timeline.IsFinished)
        {
            return;
        }

        FramesDrawn++;
        Refresh();
        float width = dirtyRect.Width, height = dirtyRect.Height;
        if (width <= 0 || height <= 0)
        {
            return;
        }

        bool wide = width >= Breakpoint;
        DrewWide = wide;
        var f = SplashFrame.At(timeline.Design, wide);
        float stage = (float)timeline.StageAlpha, cross = (float)timeline.ThemeCross;

        // The composition's scale: the design's sizes, shrunk where the window is smaller than they need.
        float k = wide ? MathF.Min(1, MathF.Min(width / 600f, height / 300f)) : MathF.Min(1, MathF.Min(width / 320f, height / 520f));
        float cx = dirtyRect.Left + width / 2, cy = dirtyRect.Top + height / 2;
        if (MathF.Abs(k - _measuredAt) > 1e-4f || _fontsStale)
        {
            Remeasure(k);
        }

        // The ground: the app's, with the system's over it while the theme crosses.
        Fill(canvas, GroundColor, stage, dirtyRect);
        if (_fromGround is not null && cross < 1)
        {
            Fill(canvas, _fromGround, stage * (1 - cross), dirtyRect);
        }

        // The mark starts where the system's splash left it (its size, the window's middle) and settles into the design's
        // place as it lifts; the rings spread from where it began.
        float lift = (float)f.Lift;
        float baseSize = (wide ? 200 : 160) * k, endSize = baseSize * (1 - (wide ? 0.2f : 0.22f));
        float startSize = StartSize > 0 ? (float)StartSize : baseSize;
        float size = startSize + (endSize - startSize) * lift;
        float shiftX = _centreShift.X * (1 - lift), shiftY = _centreShift.Y * (1 - lift);
        Ring(canvas, cx + shiftX, cy + shiftY, baseSize, (float)f.Ring1Scale, 1.5f * k, stage * (float)f.Ring1Opacity);
        Ring(canvas, cx + shiftX, cy + shiftY, baseSize, (float)f.Ring2Scale, 1f * k, stage * (float)f.Ring2Opacity);
        float mx = (wide ? cx - 200 * k * lift : cx) + shiftX, my = (wide ? cy : cy - 92 * k * lift) + shiftY;
        float s = size / 512, ox = mx - 256 * s, oy = my - 256 * s;

        // The tile (light) with its shadow: this theme's, with the system's over it while the theme crosses.
        Tile(canvas, ox, oy, s, TileColor, TileStrokeColor, ShadowColor, stage * (_fromGround is not null ? cross : 1));
        if (_fromGround is not null && cross < 1 && _fromTile is not null)
        {
            Tile(canvas, ox, oy, s, _fromTile, _fromTileStroke, _fromShadow, stage * (1 - cross));
        }

        // The bloom under the edges, the edges, the signals, the nodes, the tips and the core.
        canvas.Alpha = stage * (float)f.BloomOpacity;
        canvas.FillColor = EdgeColor;
        canvas.FillCircle(ox + 256 * s, oy + 256 * s, CoreRadius * (float)f.BloomScale * s);
        canvas.Alpha = stage;
        canvas.StrokeLineCap = LineCap.Round;
        canvas.StrokeColor = _edgeOnGround ?? EdgeColor;
        canvas.StrokeSize = EdgeWidth * s;
        for (int i = 0; i < EdgeLines.Length; i += 4)
        {
            canvas.DrawLine(ox + EdgeLines[i] * s, oy + EdgeLines[i + 1] * s, ox + EdgeLines[i + 2] * s, oy + EdgeLines[i + 3] * s);
        }

        canvas.StrokeColor = SignalColor;
        canvas.StrokeSize = SignalWidth * s;
        Signals(canvas, TopSignals, (float)f.SignalTop, ox, oy, s);
        Signals(canvas, BottomSignals, (float)f.SignalBottom, ox, oy, s);
        Signals(canvas, AxisSignals, (float)f.SignalAxis, ox, oy, s);
        canvas.FillColor = NodeColor;
        float corner = CornerRadius * (float)f.NodeScale * s;
        for (int i = 0; i < Corners.Length; i += 2)
        {
            canvas.FillCircle(ox + Corners[i] * s, oy + Corners[i + 1] * s, corner);
        }

        canvas.FillColor = CoreColor;
        canvas.FillCircle(ox + 256 * s, oy + 140 * s, TipRadius * s);
        canvas.FillCircle(ox + 256 * s, oy + 372 * s, TipRadius * s);
        if (f.Flash > 0)
        {
            canvas.Alpha = stage * (float)f.Flash;
            canvas.FillColor = FlashColor;
            canvas.FillCircle(ox + 256 * s, oy + 140 * s, TipRadius * s);
            canvas.FillCircle(ox + 256 * s, oy + 372 * s, TipRadius * s);
            canvas.Alpha = stage;
            canvas.FillColor = CoreColor;
        }

        canvas.FillCircle(ox + 256 * s, oy + 256 * s, CoreRadius * (float)f.CoreScale * s);
        if (wide)
        {
            DrawWideText(canvas, f, stage, cx, cy, k);
        }
        else
        {
            DrawStackedText(canvas, f, stage, cx, cy, k, dirtyRect.Bottom);
        }

        canvas.Alpha = 1;
    }

    // A narrow window (Main and PhoneLight): the names centred under the window's middle, the foot 92 units from the bottom.
    private void DrawStackedText(ICanvas canvas, in SplashFrame f, float stage, float cx, float cy, float k, float bottom)
    {
        float top = cy + 10 * k;
        float nameShift = (float)f.NameShift * k, subShift = (float)f.SubShift * k;
        Text(canvas, Words.Name, _nameFont, 46 * k, NameColor, stage * (float)f.NameOpacity, cx, top + Baseline(46, 1, 0.969f, 0.242f) * k + nameShift, HorizontalAlignment.Center);
        float sub = top + (46 + 6) * k;
        float subAlpha = stage * (float)f.SubOpacity;
        if (Words.NativeName.Length > 0)
        {
            Text(canvas, Words.NativeName, _nativeFont, 22 * k, NativeColor, subAlpha, cx, sub + Baseline(22, 1.2f, 1.0f, 0.45f) * k + subShift, HorizontalAlignment.Center);
            sub += (22 * 1.2f + 10) * k;
        }

        if (Words.Tagline.Length > 0)
        {
            Text(canvas, Words.Tagline, _taglineFont, 14 * k, MutedColor, subAlpha, cx, sub + 0.969f * 14 * k + subShift, HorizontalAlignment.Center);
        }

        float foot = stage * (float)f.FootOpacity;
        if (foot <= 0)
        {
            return;
        }

        float barTop = bottom - 95 * k, lineMiddle = barTop - (14 + 9) * k;
        StatusLines(canvas, foot, cx, lineMiddle, 12 * k, k, centred: true);
        Bar(canvas, cx - 70 * k, barTop, 140 * k, 3 * k, foot);
    }

    // A wide window (Desktop): the names left-aligned beside the mark (from 84 units left of the middle, 78 above it), the
    // second name and the line on one baseline, the foot under them.
    private void DrawWideText(ICanvas canvas, in SplashFrame f, float stage, float cx, float cy, float k)
    {
        float left = cx - 84 * k, top = cy - 78 * k;
        float nameShift = (float)f.NameShift * k, subShift = (float)f.SubShift * k;
        Text(canvas, Words.Name, _nameFont, 68 * k, NameColor, stage * (float)f.NameOpacity, left + nameShift, top + Baseline(68, 1, 0.969f, 0.242f) * k, HorizontalAlignment.Left);
        float row = top + (68 + 8) * k;
        float baseline = row + 24.5f * k;
        float subAlpha = stage * (float)f.SubOpacity;
        float x = left + subShift;
        if (Words.NativeName.Length > 0)
        {
            Text(canvas, Words.NativeName, _nativeFont, 28 * k, NativeColor, subAlpha, x, baseline, HorizontalAlignment.Left);
            if (_nativeWidth < 0)
            {
                _nativeWidth = Measure(canvas, Words.NativeName, _nativeFont, 28 * k);       // once a size
            }

            x += _nativeWidth + 14 * k;
        }

        if (Words.Tagline.Length > 0)
        {
            Text(canvas, Words.Tagline, _taglineFont, 17 * k, MutedColor, subAlpha, x, baseline, HorizontalAlignment.Left);
        }

        float foot = stage * (float)f.FootOpacity;
        if (foot <= 0)
        {
            return;
        }

        float lineTop = row + 33.6f * k + (8 + 22) * k;
        StatusLines(canvas, foot, left, lineTop + 9 * k, 13 * k, k, centred: false);
        Bar(canvas, left, lineTop + (18 + 12) * k, 220 * k, 3 * k, foot);
    }

    // The status line: the newest line rising in, the one it replaces rising out; a dot before each (green at Ready).
    private void StatusLines(ICanvas canvas, float alpha, float x, float middle, float size, float k, bool centred)
    {
        var timeline = Timeline;
        float lineIn = (float)timeline.LineIn, lineOut = (float)timeline.LineOut;
        int previous = timeline.PreviousEntry;
        if (previous >= 0 && lineOut < 1)
        {
            StatusLine(canvas, timeline.Run.Title(previous), LineWidth(canvas, previous, size), false, alpha * (1 - lineOut), x, middle - 6 * k * lineOut, size, k, centred);
        }

        if (timeline.IsReadyShown)
        {
            StatusLine(canvas, Words.Ready, _readyWidth, true, alpha * lineIn, x, middle + 6 * k * (1 - lineIn), size, k, centred);
        }
        else if (timeline.LineEntry is var entry and >= 0)
        {
            StatusLine(canvas, timeline.Run.Title(entry), LineWidth(canvas, entry, size), false, alpha * lineIn, x, middle + 6 * k * (1 - lineIn), size, k, centred);
        }
    }

    // A step's line width, measured once a line (and again only when the fonts or the scale change).
    private float LineWidth(ICanvas canvas, int entry, float size)
    {
        if (entry >= _lineWidths.Length)
        {
            int was = _lineWidths.Length;
            Array.Resize(ref _lineWidths, Math.Max(entry + 1, Math.Max(8, was * 2)));      // the run grew: once
            Array.Fill(_lineWidths, -1, was, _lineWidths.Length - was);
        }

        if (_lineWidths[entry] < 0)
        {
            _lineWidths[entry] = Measure(canvas, Timeline.Run.Title(entry), _statusFont, size);
        }

        return _lineWidths[entry];
    }

    private void StatusLine(ICanvas canvas, string text, float textWidth, bool ready, float alpha, float x, float middle, float size, float k, bool centred)
    {
        if (alpha <= 0)
        {
            return;
        }

        if (ready && textWidth < 0)
        {
            _readyWidth = textWidth = Measure(canvas, text, _statusFont, size);
        }

        float dot = 6 * k, gap = 8 * k;
        float start = centred ? x - (dot + gap + MathF.Max(0, textWidth)) / 2 : x;
        canvas.Alpha = alpha;
        canvas.FillColor = ready ? ReadyColor : AccentColor;
        canvas.FillCircle(start + dot / 2, middle, dot / 2);
        canvas.FontColor = MutedColor;
        canvas.Font = _statusFont;
        canvas.FontSize = size;
        canvas.DrawString(text, start + dot + gap, middle + 0.36f * size, HorizontalAlignment.Left);
    }

    private void Bar(ICanvas canvas, float left, float top, float width, float height, float alpha)
    {
        canvas.Alpha = alpha;
        canvas.FillColor = TrackColor;
        canvas.FillRoundedRectangle(left, top, width, height, height * 2 / 3);
        float fill = width * (float)Math.Clamp(Timeline.Fill, 0, 1);
        if (fill > 0.5f)
        {
            canvas.FillColor = AccentColor;
            canvas.FillRoundedRectangle(left, top, fill, height, MathF.Min(height * 2 / 3, fill / 2));
        }
    }

    // The baseline in a line box of lineHeight × size for a font of that ascent and descent (in em): CSS's half-leading.
    private static float Baseline(float size, float lineHeight, float ascent, float descent) => ((lineHeight - (ascent + descent)) / 2 + ascent) * size;

    private static void Text(ICanvas canvas, string text, IFont font, float size, Color color, float alpha, float x, float baseline, HorizontalAlignment align)
    {
        if (alpha <= 0 || text.Length == 0)
        {
            return;
        }

        canvas.Alpha = alpha;
        canvas.FontColor = color;
        canvas.Font = font;
        canvas.FontSize = size;
        canvas.DrawString(text, x, baseline, align);
    }

    private static void Fill(ICanvas canvas, Color color, float alpha, RectF rect)
    {
        if (alpha <= 0)
        {
            return;
        }

        canvas.Alpha = alpha;
        canvas.FillColor = color;
        canvas.FillRectangle(rect);
    }

    private void Ring(ICanvas canvas, float cx, float cy, float size, float scale, float border, float alpha)
    {
        if (alpha <= 0)
        {
            return;
        }

        canvas.Alpha = alpha;
        canvas.StrokeColor = RingColor;
        canvas.StrokeSize = border * scale;
        canvas.DrawCircle(cx, cy, (size / 2 - border / 2) * scale);              // the border inside the box, as CSS draws it
    }

    private static void Tile(ICanvas canvas, float ox, float oy, float s, Color? tile, Color? stroke, Color? shadow, float alpha)
    {
        if (tile is null || tile.Alpha <= 0 || alpha <= 0)
        {
            return;
        }

        // The design's shadow: 18 units down, blurred 40 (as a blur radius, about 34), on the 160-unit box (512 / 160 here).
        canvas.Alpha = alpha;
        if (shadow is not null && shadow.Alpha > 0)
        {
            canvas.SetShadow(new SizeF(0, 18 * 3.2f * s), 34 * 3.2f * s, shadow);
        }

        canvas.FillColor = tile;
        canvas.FillRoundedRectangle(ox + TileInset * s, oy + TileInset * s, TileSize * s, TileSize * s, TileRound * s);
        canvas.SetShadow(SizeF.Zero, 0, null!);
        if (stroke is not null && stroke.Alpha > 0)
        {
            canvas.StrokeColor = stroke;
            canvas.StrokeSize = TileStroke * s;
            canvas.DrawRoundedRectangle(ox + TileInset * s, oy + TileInset * s, TileSize * s, TileSize * s, TileRound * s);
        }
    }

    // A signal on each of its edges: a dash of 16 of the edge's 100 units, at the offset the keyframes give (CSS's dash
    // offset: 16 not yet out, -100 arrived), drawn as the part of the edge it covers.
    private static void Signals(ICanvas canvas, float[] lines, float offset, float ox, float oy, float s)
    {
        float from = MathF.Max(0, -offset) / 100, to = MathF.Min(100, SignalLength - offset) / 100;
        if (to <= from)
        {
            return;
        }

        for (int i = 0; i < lines.Length; i += 4)
        {
            float ax = lines[i], ay = lines[i + 1], bx = lines[i + 2], by = lines[i + 3];
            canvas.DrawLine(ox + (ax + (bx - ax) * from) * s, oy + (ay + (by - ay) * from) * s, ox + (ax + (bx - ax) * to) * s, oy + (ay + (by - ay) * to) * s);
        }
    }

    // The colours and fonts the theme gave, taken again only when it changes them (a frame takes the kept ones).
    private void Refresh()
    {
        if (_coloursStale)
        {
            _coloursStale = false;
            var under = TileColor is { Alpha: > 0 } tile ? tile : GroundColor;
            var edge = EdgeColor;
            _edgeOnGround = under is null || edge is null ? edge
                : new Color(under.Red + (edge.Red - under.Red) * EdgeOpacity, under.Green + (edge.Green - under.Green) * EdgeOpacity,
                    under.Blue + (edge.Blue - under.Blue) * EdgeOpacity, edge.Alpha);       // the edges' 0.9 over the mark's own ground
        }

        if (_fontsStale)
        {
            var services = Handler?.MauiContext?.Services;
            _nameFont = DrawnFonts.Resolve(_systemFonts ? null : NameFont, services, 800);
            _nativeFont = DrawnFonts.Resolve(_systemFonts ? null : NativeFont, services, 600);
            _taglineFont = DrawnFonts.Resolve(_systemFonts ? null : TaglineFont, services, 500);
            _statusFont = DrawnFonts.Resolve(_systemFonts ? null : StatusFont, services, 400);
        }
    }

    private void Remeasure(float k)
    {
        _fontsStale = false;
        _measuredAt = k;
        _readyWidth = -1;
        _nativeWidth = -1;
        if (_lineWidths.Length < Timeline.Run.Count)
        {
            _lineWidths = new float[Math.Max(8, Timeline.Run.Count * 2)];        // room for the run's lines, made once
        }

        Array.Fill(_lineWidths, -1);
    }

    private static float Measure(ICanvas canvas, string text, IFont font, float size)
    {
        try
        {
            return canvas.GetStringSize(text, font, size).Width;
        }
        catch (Exception e) when (e is NotImplementedException or NotSupportedException or InvalidOperationException)
        {
            return text.Length * size * 0.55f;                                  // a canvas that cannot measure: an estimate
        }
    }

    /// <inheritdoc />
    protected override void OnPropertyChanged(string? propertyName = null)
    {
        base.OnPropertyChanged(propertyName);
        if (propertyName is nameof(TileColor) or nameof(GroundColor) or nameof(EdgeColor))
        {
            _coloursStale = true;
        }
        else if (propertyName is nameof(NameFont) or nameof(NativeFont) or nameof(TaglineFont) or nameof(StatusFont))
        {
            _fontsStale = true;
        }
    }

    private static BindableProperty Colour(string name) =>
        BindableProperty.Create(name, typeof(Color), typeof(SplashView), Colors.Transparent, propertyChanged: (b, _, _) => ((SplashView)b).Invalidate());

    private static BindableProperty FontAlias(string name) =>
        BindableProperty.Create(name, typeof(string), typeof(SplashView), "", propertyChanged: (b, _, _) => ((SplashView)b).Invalidate());
}

/// <summary>
/// A font alias the app registered (<c>ConfigureFonts</c>) as the canvas of this platform finds it: on Android the font's
/// file among the app's assets, on Windows the family MAUI's font manager makes of it; an alias the app did not register is
/// the system's font at the weight asked for.
/// </summary>
internal static class DrawnFonts
{
    public static IFont Resolve(string? alias, IServiceProvider? services, int fallbackWeight)
    {
        var system = new Microsoft.Maui.Graphics.Font(null!, fallbackWeight);
        if (string.IsNullOrWhiteSpace(alias))
        {
            return system;
        }

        if (services?.GetService(typeof(IFontRegistrar)) is not IFontRegistrar registrar)
        {
            return new Microsoft.Maui.Graphics.Font(alias);                      // no platform's fonts (the tests): the alias as it is
        }

        string? file;
        try
        {
            file = registrar.GetFont(alias);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidOperationException or ArgumentException)
        {
            file = null;
        }

        if (file is null)
        {
            return system;                                                      // not registered: the system's font
        }
#if WINDOWS
        if (services.GetService(typeof(IFontManager)) is IFontManager manager)
        {
            var family = manager.GetFontFamily(Microsoft.Maui.Font.OfSize(alias, 12));
            return family?.Source is { Length: > 0 } source ? new Microsoft.Maui.Graphics.Font(source) : system;
        }

        return system;
#elif ANDROID
        return Path.IsPathRooted(file) ? system : new Microsoft.Maui.Graphics.Font(file);   // the asset's file name, which the canvas loads
#else
        return new Microsoft.Maui.Graphics.Font(alias);
#endif
    }
}
