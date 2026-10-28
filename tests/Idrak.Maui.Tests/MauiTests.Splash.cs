// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Numerics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Graphics;

using Font = Microsoft.Maui.Graphics.Font;
using IImage = Microsoft.Maui.Graphics.IImage;

namespace Idrak.Maui.Tests;

// Plan 34, phase S: the animated splash. The start-up run and its steps, the design's motion by its time, the three
// lengths with a clock driven by hand, the host's hand-over, both themes and both layouts drawn into a canvas that keeps
// count, no allocation a frame, the settings.
internal static partial class MauiTests
{
    // A clock moved by hand: a microsecond a tick.
    private sealed class HandClock : TimeProvider
    {
        public long Now { get; set; }

        public override long GetTimestamp() => Now;

        public override long TimestampFrequency => 1_000_000;

        public long At(double seconds) => Now = (long)Math.Round(seconds * 1_000_000);
    }

    // A canvas that draws nothing and keeps count of what it was asked, allocating nothing.
    private sealed class CountingCanvas : ICanvas
    {
        public readonly string[] Texts = new string[256];
        public int TextCount;
        public int Lines, Circles, RoundedFills, RoundedStrokes, Rectangles, Shadows;
        public Color? FirstRectangle, LastFill, TileFill;
        private Color? _fill;

        public void Reset()
        {
            TextCount = Lines = Circles = RoundedFills = RoundedStrokes = Rectangles = Shadows = 0;
            FirstRectangle = null;
            TileFill = null;
        }

        public bool Drew(string text)
        {
            for (int i = 0; i < Math.Min(TextCount, Texts.Length); i++)
            {
                if (Texts[i] == text)
                {
                    return true;
                }
            }

            return false;
        }

        public float DisplayScale { get; set; } = 1;
        public float StrokeSize { set { } }
        public float MiterLimit { set { } }
        public Color StrokeColor { set { } }
        public LineCap StrokeLineCap { set { } }
        public LineJoin StrokeLineJoin { set { } }
        public float[] StrokeDashPattern { set { } }
        public float StrokeDashOffset { set { } }
        public Color FillColor { set => _fill = value; }
        public Color FontColor { set { } }
        public IFont Font { set { } }
        public float FontSize { set { } }
        public float Alpha { set { } }
        public bool Antialias { set { } }
        public BlendMode BlendMode { set { } }

        public void DrawPath(PathF path) { }
        public void FillPath(PathF path, WindingMode windingMode) { }
        public void SubtractFromClip(float x, float y, float width, float height) { }
        public void ClipPath(PathF path, WindingMode windingMode = WindingMode.NonZero) { }
        public void ClipRectangle(float x, float y, float width, float height) { }
        public void DrawLine(float x1, float y1, float x2, float y2) => Lines++;
        public void DrawArc(float x, float y, float width, float height, float startAngle, float endAngle, bool clockwise, bool closed) { }
        public void FillArc(float x, float y, float width, float height, float startAngle, float endAngle, bool clockwise) { }
        public void DrawRectangle(float x, float y, float width, float height) { }
        public void FillRectangle(float x, float y, float width, float height)
        {
            Rectangles++;
            FirstRectangle ??= _fill;
        }

        public void DrawRoundedRectangle(float x, float y, float width, float height, float cornerRadius) => RoundedStrokes++;
        public void FillRoundedRectangle(float x, float y, float width, float height, float cornerRadius)
        {
            RoundedFills++;
            LastFill = _fill;
            TileFill ??= _fill;                                                 // the first a frame: the tile, where one is drawn
        }

        public void DrawEllipse(float x, float y, float width, float height) => Circles++;
        public void FillEllipse(float x, float y, float width, float height) => Circles++;
        public void DrawString(string value, float x, float y, HorizontalAlignment horizontalAlignment) => Texts[TextCount++ % Texts.Length] = value;
        public void DrawString(string value, float x, float y, float width, float height, HorizontalAlignment horizontalAlignment, VerticalAlignment verticalAlignment,
            TextFlow textFlow = TextFlow.ClipBounds, float lineSpacingAdjustment = 0) => Texts[TextCount++ % Texts.Length] = value;
        public void DrawText(Microsoft.Maui.Graphics.Text.IAttributedText value, float x, float y, float width, float height) { }
        public void Rotate(float degrees, float x, float y) { }
        public void Rotate(float degrees) { }
        public void Scale(float sx, float sy) { }
        public void Translate(float tx, float ty) { }
        public void ConcatenateTransform(Matrix3x2 transform) { }
        public void SaveState() { }
        public bool RestoreState() => true;
        public void ResetState() { }
        public void SetShadow(SizeF offset, float blur, Color color) => Shadows += color is null ? 0 : 1;
        public void SetFillPaint(Paint paint, RectF rectangle) { }
        public void DrawImage(IImage image, float x, float y, float width, float height) { }
        public SizeF GetStringSize(string value, IFont font, float fontSize) => new(value.Length * fontSize * 0.5f, fontSize);
        public SizeF GetStringSize(string value, IFont font, float fontSize, HorizontalAlignment horizontalAlignment, VerticalAlignment verticalAlignment) =>
            new(value.Length * fontSize * 0.5f, fontSize);
    }

    // A step that records when it ran and on which kind of thread, and fails when asked.
    private sealed class RecordedStep(string name, int order, StartupThread thread, ConcurrentQueue<string> ran, bool fails = false, bool wanted = true) : IStartupStep
    {
        public string Name => name;
        public string Summary => "a test's step";
        public string Title => "Doing " + name;
        public int Order => order;
        public StartupThread Thread => thread;
        public bool IsWanted(StartupContext context) => wanted;

        public async Task RunAsync(StartupContext context, CancellationToken cancellationToken)
        {
            await Task.Delay(5, cancellationToken);
            ran.Enqueue(name);
            if (fails)
            {
                throw new InvalidOperationException("this step fails\nand says more");
            }
        }
    }

    private static IdrakMauiOptions SplashOptions(SplashLength length) => new() { Device = "cpu", ShowSplash = true, Splash = length };

    // Moves the timeline from one time to another a display frame (60 a second) at a time, calling back each frame.
    private static void Frames(SplashTimeline timeline, HandClock clock, double from, double to, Action<double>? each = null)
    {
        for (double t = from; t <= to + 1e-9; t += 1.0 / 60)
        {
            timeline.Update(clock.At(t));
            each?.Invoke(t);
            if (timeline.IsFinished)
            {
                return;
            }
        }
    }

    private static void SplashStartupRun()
    {
        Check(StartupSteps.Names.SequenceEqual([StartupSteps.Cpu, StartupSteps.Gpu, StartupSteps.Sections])
              && StartupSteps.All.Select(s => s.Name).SequenceEqual([StartupSteps.Cpu, StartupSteps.Gpu, StartupSteps.Sections])
              && StartupSteps.Get("CPU").Title == "Reading the CPU" && StartupSteps.Get(StartupSteps.Gpu).Thread == StartupThread.Worker
              && StartupSteps.Get(StartupSteps.Sections).Thread == StartupThread.Ui,
            $"the library's steps: {string.Join(", ", StartupSteps.All.Select(s => $"{s.Name} ({s.Title}, {s.Thread})"))}");

        var clock = new HandClock();
        var run = new StartupRun(clock);
        clock.At(1);
        run.Record("settings", "Reading the settings", TimeSpan.FromMilliseconds(40));
        Check(run.Count == 1 && run.Started == 1 && run.Finished == 1 && !run.IsDone && run.State(0) == StartupStepState.Done
              && Math.Abs(run.Took(0).TotalMilliseconds - 40) < 0.01, $"recorded: {run.Count} {run.Finished} {run.State(0)} {run.Took(0)}");

        var ran = new ConcurrentQueue<string>();
        IStartupStep[] steps =
        [
            new RecordedStep("w1", 1, StartupThread.Worker, ran), new RecordedStep("w2", 2, StartupThread.Worker, ran, fails: true),
            new RecordedStep("w3", 3, StartupThread.Worker, ran), new RecordedStep("u1", 4, StartupThread.Ui, ran),
            new RecordedStep("u2", 5, StartupThread.Ui, ran), new RecordedStep("skipped", 6, StartupThread.Worker, ran, wanted: false),
        ];
        var task = run.RunAsync(new StartupContext(new ServiceCollection().BuildServiceProvider(), null), steps);
        Check(ReferenceEquals(task, run.RunAsync(new StartupContext(new ServiceCollection().BuildServiceProvider(), null), steps)), "a second run is the first");
        Check(run.Count == 6, $"planned at once: {run.Count} entries (the recorded one and five wanted)");
        Check(task.Wait(TimeSpan.FromSeconds(10)) && run.Completion.Wait(TimeSpan.FromSeconds(10)) && run.IsDone, $"done: {run.IsDone}");
        var order = ran.ToArray();
        Check(order.Where(n => n.StartsWith('w')).SequenceEqual(["w1", "w2", "w3"]) && order.Where(n => n.StartsWith('u')).SequenceEqual(["u1", "u2"]) && !order.Contains("skipped"),
            $"ran {string.Join(", ", order)}: the worker ones in order, the UI ones in order, the unwanted not");
        var names = Enumerable.Range(0, run.Count).Select(run.Name).ToList();
        int failed = names.IndexOf("w2");
        Check(run.State(failed) == StartupStepState.Failed && run.Error(failed) == "this step fails"
              && Enumerable.Range(0, run.Count).Where(i => i != failed).All(i => run.State(i) == StartupStepState.Done) && run.Finished == run.Count && run.Started == run.Count,
            $"states {string.Join(", ", Enumerable.Range(0, run.Count).Select(i => $"{run.Name(i)} {run.State(i)} {run.Error(i)}"))}");
        try
        {
            run.Begin("late", "Too late");
            Check(false, "an entry begun after the run ended");
        }
        catch (InvalidOperationException e)
        {
            Check(e.Message.Contains("late", StringComparison.Ordinal), e.Message);
        }

        // By hand: done only once every begun entry ended and nothing more comes.
        var hand = new StartupRun(clock);
        int a = hand.Begin("a", "A");
        hand.Complete();
        Check(!hand.IsDone && hand.Started == 1 && hand.Finished == 0, "a running entry keeps the run going");
        hand.End(a, new IOException("no\nmore"));
        Check(hand.IsDone && hand.State(a) == StartupStepState.Failed && hand.Error(a) == "no", "ended: done");

        // An app's step joins in order, and the library's come back without it; an unknown name is not registered.
        var mine = new RecordedStep("models-warm", 15, StartupThread.Worker, ran);
        try
        {
            StartupSteps.Register(mine);
            Check(StartupSteps.All.Select(s => s.Name).SequenceEqual([StartupSteps.Cpu, "models-warm", StartupSteps.Gpu, StartupSteps.Sections])
                  && StartupSteps.Origin("models-warm") == typeof(MauiTests).Assembly.GetName().Name, string.Join(", ", StartupSteps.All.Select(s => s.Name)));
        }
        finally
        {
            StartupSteps.Unregister("models-warm");
        }

        Check(StartupSteps.Find("models-warm") is null, "unregistered");
        try
        {
            StartupSteps.Get("nowhere");
            Check(false, "an unknown step was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Message.Contains("cpu", StringComparison.Ordinal), e.Message);
        }
    }

    private static void SplashMotion()
    {
        foreach (bool wide in (bool[])[false, true])
        {
            var first = SplashFrame.At(0, wide);
            Check(first is { Lift: 0, NodeScale: 1, CoreScale: 1, BloomOpacity: 0, Ring1Opacity: 0, Ring2Opacity: 0, NameOpacity: 0, SubOpacity: 0, FootOpacity: 0, Flash: 0 }
                  && first.SignalTop == 16 && first.SignalAxis == 16, $"{(wide ? "wide" : "narrow")} at 0: the system splash's frame (the mark still, nothing else)");
            var signals = SplashFrame.At(1.0, wide);
            Check(signals.SignalTop is < 16 and > -100 && signals.SignalBottom is < 16 and > -100 && signals.SignalAxis == 16 && signals.Lift == 0,
                $"at 1 s the signals travel in: top {signals.SignalTop:F1}, bottom {signals.SignalBottom:F1}, axis {signals.SignalAxis:F1}");
            var pulse = SplashFrame.At(SplashFrame.Time(12), wide);
            Check(Math.Abs(pulse.NodeScale - 1.2) < 1e-9, $"the amber pulse peaks at 12%: {pulse.NodeScale}");
            var swell = SplashFrame.At(SplashFrame.Time(29), wide);
            Check(Math.Abs(swell.CoreScale - 1.18) < 1e-9 && Math.Abs(swell.Ring1Opacity - 0.75) < 1e-9 && swell.BloomOpacity > 0.3 && swell.Ring1Scale > 0.45,
                $"the core swells at 29%: {swell.CoreScale}, ring {swell.Ring1Opacity}, bloom {swell.BloomOpacity:F2}");
            var axis = SplashFrame.At(SplashFrame.Time(38), wide);
            Check(axis.SignalAxis is < 16 and > -100 && axis.SignalTop == -100 && axis.Lift is > 0 and < 1, $"at 38% the axis' signal runs and the mark lifts: {axis.SignalAxis:F1}, lift {axis.Lift:F2}");
            var end = SplashFrame.At(SplashFrame.IntroEnd, wide);
            Check(end is { Lift: 1, NameOpacity: 1, SubOpacity: 1, FootOpacity: 1, NameShift: 0, SubShift: 0, Ring1Opacity: 0, Ring2Opacity: 0, BloomOpacity: 0, Flash: 0 },
                $"{(wide ? "wide" : "narrow")} at the end: the name settled, the rings gone");
            Check(end.Ring1Scale == (wide ? 3.2 : 2.9) && end.Ring2Scale == (wide ? 3.8 : 3.4) && SplashFrame.At(9, wide).Equals(end) && SplashFrame.At(1.234, wide).Equals(SplashFrame.At(1.234, wide)),
                "the rings' last scale by the layout; the same time, the same frame");
            for (double t = 0; t <= SplashFrame.IntroEnd; t += 0.01)
            {
                var f = SplashFrame.At(t, wide);
                Check(f.Lift is >= 0 and <= 1 && f.NameOpacity is >= 0 and <= 1 && f.FootOpacity is >= 0 and <= 1 && f.BloomOpacity is >= 0 and <= 0.6 + 1e-9
                      && double.IsFinite(f.Ring2Scale), $"at {t:F2} s a value out of its range");
            }
        }

        Check(Math.Abs(SplashFrame.IntroEnd - 3.42) < 1e-9 && Math.Abs(SplashFrame.FootStart - 2.94) < 1e-9, "the design's 57% and 49% of six seconds");
    }

    private static void SplashShort()
    {
        // Ready at 0.5 s: the cut ahead to the name and the fade together, within the fade's length.
        var clock = new HandClock();
        var run = new StartupRun(clock);
        int cpu = run.Begin("cpu", "Reading the CPU");
        var options = SplashOptions(SplashLength.Short);
        Check(new IdrakMauiOptions().Splash == SplashLength.Short, "short is the default");
        var timeline = new SplashTimeline(run, options, clock.TimestampFrequency);
        Frames(timeline, clock, 0, 0.5);
        Check(!timeline.IsHandingOver && timeline.StageAlpha == 1 && timeline.Design > 0.45, $"while the start-up runs it plays: {timeline.Design:F2}");
        run.End(cpu);
        run.Complete();
        double ready = 0.5 + 1.0 / 60, finished = -1, reachedName = -1;
        Frames(timeline, clock, ready, 3, t =>
        {
            if (reachedName < 0 && timeline.Design >= SplashFrame.IntroEnd)
            {
                reachedName = t;
            }

            if (timeline.IsFinished && finished < 0)
            {
                finished = t;
            }
        });
        Check(timeline.IsHandingOver && timeline.IsReadyShown && reachedName > 0 && reachedName - ready <= options.SplashCutAhead.TotalSeconds + 0.02,
            $"ready: it cut ahead to the name in {reachedName - ready:F3} s");
        Check(finished > 0 && finished - ready <= options.SplashFade.TotalSeconds + 0.02 && finished - ready >= options.SplashFade.TotalSeconds - 0.02 && timeline.StageAlpha == 0,
            $"faded into Home {finished - ready:F3} s after Ready (the fade is {options.SplashFade.TotalMilliseconds} ms)");

        // A start-up longer than the animation: the animation holds at its end and the status line keeps going.
        clock = new HandClock();
        run = new StartupRun(clock);
        timeline = new SplashTimeline(run, options, clock.TimestampFrequency);
        int gpu = -1;
        Frames(timeline, clock, 0, 6, t =>
        {
            if (gpu < 0 && t >= 0.2)
            {
                gpu = run.Begin("gpu", "Starting the GPU");
            }
        });
        Check(!timeline.IsHandingOver && timeline.Design == SplashFrame.IntroEnd && timeline.LineEntry == gpu && run.Title(timeline.LineEntry) == "Starting the GPU" && timeline.StageAlpha == 1,
            $"at 6 s still starting: design {timeline.Design}, line {timeline.LineEntry}");
        run.End(gpu);
        run.Complete();
        Frames(timeline, clock, 6 + 1.0 / 60, 7);
        Check(timeline.IsFinished, "it ends at Ready");

        // Past the longest, it hands over while a step still runs.
        clock = new HandClock();
        run = new StartupRun(clock);
        run.Begin("stuck", "A step that does not end");
        var capped = SplashOptions(SplashLength.Short);
        capped.SplashLongest = TimeSpan.FromSeconds(4);
        timeline = new SplashTimeline(run, capped, clock.TimestampFrequency);
        Frames(timeline, clock, 0, 5);
        Check(timeline.IsFinished && !run.IsDone, $"the longest: finished {timeline.IsFinished} with the start-up still running");
    }

    private static void SplashFull()
    {
        // Ready at 0.5 s: the whole animation still plays, then Ready its least time, then the fade.
        var clock = new HandClock();
        var run = new StartupRun(clock);
        run.Record("settings", "Reading the settings", TimeSpan.FromMilliseconds(10));
        int cpu = run.Begin("cpu", "Reading the CPU");
        var options = SplashOptions(SplashLength.Full);
        var timeline = new SplashTimeline(run, options, clock.TimestampFrequency);
        double handing = -1, finished = -1;
        Frames(timeline, clock, 0, 6, t =>
        {
            if (t >= 0.5 && !run.IsDone)
            {
                run.End(cpu);
                run.Complete();
            }

            if (handing < 0 && timeline.IsHandingOver)
            {
                handing = t;
            }

            if (finished < 0 && timeline.IsFinished)
            {
                finished = t;
            }
        });
        double readyShown = SplashFrame.FootStart;
        Check(handing >= SplashFrame.IntroEnd && Math.Abs(handing - (readyShown + options.SplashStepTime.TotalSeconds)) < 0.04,
            $"the hand-over began at {handing:F2} s: after the animation's end ({SplashFrame.IntroEnd}) and Ready's least time");
        Check(finished > 0 && Math.Abs(finished - handing - options.SplashFade.TotalSeconds) < 0.03 && finished < 4.2, $"finished at {finished:F2} s (about three seconds and the fade)");

        // The steps shown in the order they began, each at least its least time, then Ready; those before the foot are not
        // shown after the fact.
        clock = new HandClock();
        run = new StartupRun(clock);
        timeline = new SplashTimeline(run, options, clock.TimestampFrequency);
        var shown = new List<(string Line, double At)>();
        int early = -1, a = -1, b = -1, c = -1;
        Frames(timeline, clock, 0, 8, t =>
        {
            if (early < 0)
            {
                early = run.Begin("early", "Done before the foot");
            }

            if (t >= 1 && run.State(early) == StartupStepState.Running)
            {
                run.End(early);
            }

            if (a < 0 && t >= 3.0)
            {
                a = run.Begin("a", "Step A");
            }

            if (b < 0 && t >= 3.05)
            {
                run.End(a);
                b = run.Begin("b", "Step B");
            }

            if (c < 0 && t >= 3.1)
            {
                run.End(b);
                c = run.Begin("c", "Step C");
            }

            if (c >= 0 && t >= 3.15 && !run.IsDone)
            {
                run.End(c);
                run.Complete();
            }

            string line = timeline.IsReadyShown ? "Ready" : timeline.LineEntry >= 0 ? run.Title(timeline.LineEntry) : "";
            if (t >= SplashFrame.FootStart && (shown.Count == 0 || shown[^1].Line != line))
            {
                shown.Add((line, t));
            }
        });
        var lines = shown.Select(s => s.Line).ToList();
        Check(lines.SequenceEqual(["", "Step A", "Step B", "Step C", "Ready"]),
            $"lines (a step over before the foot showed is not shown after): {string.Join(" → ", shown.Select(s => $"'{s.Line}' at {s.At:F2}"))}");
        for (int i = 2; i < shown.Count; i++)
        {
            Check(shown[i].At - shown[i - 1].At >= options.SplashStepTime.TotalSeconds - 0.02, $"{shown[i - 1].Line} stayed {shown[i].At - shown[i - 1].At:F2} s");
        }

        Check(timeline.IsFinished && timeline.Fill == 1, $"finished {timeline.IsFinished}, the bar full ({timeline.Fill:F3})");
    }

    private static void SplashHandOver()
    {
        // Off, and an app that does not ask: no splash.
        foreach (var options in new[] { SplashOptions(SplashLength.Off), new IdrakMauiOptions { Device = "cpu" } })
        {
            using var none = AppServices(options, new StartupRun());
            Check(new AppHostPage(none).Splash is null, $"show {options.ShowSplash}, {options.Splash}: no splash");
        }

        var clock = new HandClock();
        var run = new StartupRun(clock);
        var shortly = SplashOptions(SplashLength.Short);
        using var services = AppServices(shortly, run);
        var host = new AppHostPage(services);
        var splash = host.Splash;
        var root = host.FindByName<Grid>("Root");
        Check(splash is not null && root.Children.Contains(splash) && Grid.GetColumnSpan(splash) == 2 && !splash.InputTransparent && host.Current is HomeView,
            "the splash over both columns, taking the taps, Home made under it");
        Check(host.SendBackButtonPressed(), "back while the splash shows: taken, nothing closes");

        // The start-up under it: the device read, the tabs made ahead; its steps are the run's.
        var startup = host.BeginStartup();
        Check(startup is not null && startup.Wait(TimeSpan.FromSeconds(30)) && run.IsDone, $"the start-up ran: done {run.IsDone}");
        var names = Enumerable.Range(0, run.Count).Select(run.Name).Order(StringComparer.Ordinal).ToList();
        Check(names.SequenceEqual([StartupSteps.Cpu, StartupSteps.Gpu, StartupSteps.Sections]) && Enumerable.Range(0, run.Count).All(i => run.State(i) == StartupStepState.Done),
            $"steps {string.Join(", ", Enumerable.Range(0, run.Count).Select(i => $"{run.Name(i)} {run.State(i)} {run.Error(i)}"))}");
        Check(host.IsPrebuilt && host.Views.Count >= 4 && services.GetRequiredService<DeviceStatus>().IsRead, $"made ahead: {string.Join(", ", host.Views.Keys)}; read {services.GetRequiredService<DeviceStatus>().IsRead}");
        var dispatcher = (InlineDispatcher)host.Dispatcher;
        int dispatched = dispatcher.Dispatched;
        new GpuStartupStep().RunAsync(new StartupContext(services, host), CancellationToken.None).GetAwaiter().GetResult();
        Check(dispatcher.Dispatched > dispatched, "the GPU step asks for the reading on the host's UI thread (Home's summary and models are set where it asks)");

        // Done: it hands over and is taken away; back is Home's again.
        bool handing = false, finished = false;
        splash!.HandingOver += (_, _) => handing = true;
        splash.Finished += (_, _) => finished = true;
        long start = TimeProvider.System.GetTimestamp();
        for (double t = 0; t < 2 && !finished; t += 1.0 / 60)
        {
            splash.Advance(start + (long)(t * TimeProvider.System.TimestampFrequency));
        }

        Check(handing && finished && host.Splash is null && !root.Children.Contains(splash) && host.Current is HomeView, $"handed over {handing}, finished {finished}, removed {host.Splash is null}");
        Check(!host.SendBackButtonPressed(), "back on Home after the splash: the app's own (nothing to close)");
    }

    private static void SplashDraws()
    {
        EnsureDispatcher();
        foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
        {
            var colors = IdrakThemes.Colors(theme);
            foreach (var (width, height, wide) in new[] { (390f, 844f, false), (1280f, 800f, true), (360f, 480f, false), (900f, 500f, true) })
            {
                var clock = new HandClock();
                var run = new StartupRun(clock);
                run.Begin("gpu", "Starting the GPU");
                var view = new SplashView(run, SplashOptions(SplashLength.Short), clock);
                _ = Themed(view, theme);
                Check(view.GroundColor.Equals(colors["IdrakSplashGround"]) && view.EdgeColor.Equals(colors["IdrakSky"]) && view.RingColor.Equals(colors["IdrakSplashRing"])
                      && view.NativeFont == "IbmPlexArabicSemiBold" && view.NameFont == "InterExtraBold", $"{theme}: the theme's keys");
                var canvas = new CountingCanvas();
                var rect = new RectF(0, 0, width, height);
                foreach (double t in (double[])[0, 1.0, 1.8, 2.5, 3.0, 4.0])
                {
                    view.Advance(clock.At(t));
                    canvas.Reset();
                    view.Draw(canvas, rect);
                    Check(view.DrewWide == wide && canvas.FirstRectangle is { } ground && ground.Equals(colors["IdrakSplashGround"]) && canvas.Lines >= 9 && canvas.Circles >= 7,
                        $"{theme} {width}x{height} at {t} s: wide {view.DrewWide}, ground {canvas.FirstRectangle?.ToArgbHex()}, {canvas.Lines} lines, {canvas.Circles} circles");
                    bool tiled = theme == AppTheme.Light;
                    Check(tiled ? canvas.Shadows == 1 && canvas.RoundedStrokes >= 1 && canvas.TileFill is not null && canvas.TileFill.Equals(colors["IdrakSplashTile"])
                                : canvas.Shadows == 0 && canvas.RoundedStrokes == 0,
                        $"{theme} at {t} s: {(tiled ? "the tile with its shadow" : "the bare mark")} ({canvas.Shadows} shadows, {canvas.RoundedStrokes} strokes)");
                    if (t >= 4)
                    {
                        Check(canvas.Drew("Idrak") && canvas.Drew("إدراك") && canvas.Drew("Perception, on this device.") && canvas.Drew("Starting the GPU") && canvas.RoundedFills >= 1,
                            $"{theme} {width}x{height} at {t} s: the names and the status line ({string.Join(" | ", canvas.Texts.Take(canvas.TextCount))})");
                    }
                    else if (t == 0)
                    {
                        Check(canvas.TextCount == 0, $"{theme} at 0: only the mark (as the system splash left it)");
                    }
                }
            }
        }

        // The first frame starts at the system picture's size and crosses from the system's ground.
        var crossClock = new HandClock();
        var crossing = new SplashView(new StartupRun(crossClock), SplashOptions(SplashLength.Short), crossClock) { StartSize = 192 };
        _ = Themed(crossing, AppTheme.Dark);
        var light = IdrakThemes.Colors(AppTheme.Light);
        crossing.CrossFrom((Color)light["IdrakSplashGround"], (Color)light["IdrakSplashTile"], (Color)light["IdrakSplashTileStroke"], (Color)light["IdrakSplashShadow"]);
        var c = new CountingCanvas();
        crossing.Advance(crossClock.At(0));
        crossing.Draw(c, new RectF(0, 0, 390, 844));
        Check(c.Rectangles == 2 && c.Shadows == 1 && crossing.Timeline.ThemeCross == 0, $"crossing at 0: both grounds ({c.Rectangles}), the system's tile ({c.Shadows} shadow)");
        crossing.Advance(crossClock.At(0.5));
        c.Reset();
        crossing.Draw(c, new RectF(0, 0, 390, 844));
        Check(c.Rectangles == 1 && c.Shadows == 0 && crossing.Timeline.ThemeCross == 1, $"crossed by 0.5 s: the app's ground only ({c.Rectangles})");
    }

    private static void SplashNoAllocation()
    {
        EnsureDispatcher();
        foreach (var (width, height) in new[] { (390f, 844f), (1280f, 800f) })
        {
            foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
            {
                // The start-up's own changes (a step begun, ended) are made between frames and not counted: only the frames are.
                (long Bytes, int Frames, int Lines) Pass()
                {
                    var clock = new HandClock();
                    var run = new StartupRun(clock);
                    int a = run.Begin("a", "Reading the CPU"), b = -1, c = -1;
                    var view = new SplashView(run, SplashOptions(SplashLength.Full), clock);
                    _ = Themed(view, theme);
                    var canvas = new CountingCanvas();
                    var rect = new RectF(0, 0, width, height);
                    view.Advance(clock.At(0));
                    view.Draw(canvas, rect);                                      // the first frame keeps the colours, fonts and widths
                    long bytes = 0;
                    int frames = 0;
                    for (double t = 1.0 / 60; t < 5.3 && !view.Timeline.IsHandingOver; t += 1.0 / 60)
                    {
                        if (b < 0 && t >= 3.0)
                        {
                            run.End(a);
                            b = run.Begin("b", "Starting the GPU");
                        }

                        if (c < 0 && t >= 3.1)
                        {
                            run.End(b);
                            c = run.Begin("c", "Making the tabs");
                        }

                        if (c >= 0 && t >= 4.2 && !run.IsDone)
                        {
                            run.End(c);
                            run.Complete();
                        }

                        long before = GC.GetAllocatedBytesForCurrentThread();
                        view.Advance(clock.At(t));
                        canvas.Reset();
                        view.Draw(canvas, rect);
                        bytes += GC.GetAllocatedBytesForCurrentThread() - before;
                        frames++;
                    }

                    return (bytes, frames, view.Timeline.LinesShown);
                }

                Pass();                                                          // the code paths compiled first
                var (allocated, count, lines) = Pass();
                Check(lines >= 4 && count > 300, $"{lines} lines over {count} frames");
                Check(allocated == 0, $"{theme} {width}x{height}: {allocated} bytes over {count} frames ({(double)allocated / count:F1} a frame); "
                    + "run again with DOTNET_TieredCompilation=0 to tell unoptimized code's boxing from the frame's own");
            }
        }
    }

    private static void SplashSettings()
    {
        string[] keys = ["splash", "splash_step_ms", "splash_fade_ms", "splash_cut_ms", "splash_theme_ms", "splash_longest_s", "splash_picture_dp", "splash_tile_scale"];
        var settings = keys.Select(k => IdrakMauiSettings.Definitions.Single(s => s.Key == k)).ToList();
        Check(settings.All(s => s.Group == "Layout and motion" && s.NeedsRestart), string.Join(", ", settings.Select(s => $"{s.Key} {s.Group}")));
        var options = new IdrakMauiOptions();
        Check(options is { Splash: SplashLength.Short, ShowSplash: false, SplashPictureSize: null, SplashTileScale: 0.8, SplashSystemIconSize: 192, SplashBaseSize: 160 }
              && options.SplashStepTime == TimeSpan.FromMilliseconds(600) && options.SplashFade == TimeSpan.FromMilliseconds(420) && options.SplashCutAhead == TimeSpan.FromMilliseconds(200),
            "the design's values by default; the library's splash off unless the app asks");
        var length = settings[0];
        Check(length.Choices.SequenceEqual(["short", "full", "off"]) && length.Get(options) == "short", $"choices {string.Join(", ", length.Choices)}");
        foreach (var (setting, text) in new[] { (length, "long"), (settings[1], "-5"), (settings[2], "slow"), (settings[5], "0"), (settings[7], "1.5") })
        {
            bool refused = false;
            try
            {
                setting.Set(options, text);
            }
            catch (FormatException)
            {
                refused = true;
            }

            Check(refused, $"{setting.Key} took '{text}'");
        }

        length.Set(options, "FULL");
        settings[1].Set(options, "250");
        settings[2].Set(options, "300,5");
        settings[6].Set(options, "180");
        settings[7].Set(options, "0.75");
        Check(options.Splash == SplashLength.Full && options.SplashStepTime == TimeSpan.FromMilliseconds(250) && options.SplashFade == TimeSpan.FromMilliseconds(300.5)
              && options.SplashPictureSize == 180 && options.SplashTileScale == 0.75, "set");
        var store = new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
        store.SaveAsync(options).GetAwaiter().GetResult();
        var read = new IdrakMauiOptions();
        store.LoadIntoAsync(read).GetAwaiter().GetResult();
        Check(read.Splash == SplashLength.Full && read.SplashStepTime == TimeSpan.FromMilliseconds(250) && read.SplashFade == TimeSpan.FromMilliseconds(300.5) && read.SplashPictureSize == 180,
            $"read back: {read.Splash}, {read.SplashStepTime}, {read.SplashFade}, {read.SplashPictureSize}");
        length.Set(read, "");
        settings[6].Set(read, "");
        Check(read.Splash == SplashLength.Short && read.SplashPictureSize is null, "empty: the defaults");
        var copy = new IdrakMauiOptions();
        copy.CopyFrom(new IdrakMauiOptions { ShowSplash = true, SplashWords = new SplashWords("Mine") });
        Check(copy.ShowSplash && copy.SplashWords.Name == "Mine", "the app's own values are copied with the rest");

        var page = new SettingsViewModel(new IdrakMauiOptions(), new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore()),
            IdrakMauiStartup.Apply(new IdrakMauiOptions(), Temp("splash-settings")));
        page.Query = "splash";
        var shown = page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).ToList();
        Check(keys.All(shown.Contains), string.Join(", ", shown));
        page.Query = "splash least";
        Check(page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).SequenceEqual(["splash_step_ms"]), string.Join(", ", page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key)));
    }
}
