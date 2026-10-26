// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Microsoft.Maui.Controls.Shapes;

namespace Idrak.Maui.Controls;

/// <summary>
/// The app's light motion: a cross-fade, a caret's blink, a record button's pulse. An animation runs only while its view is
/// on screen (it has a handler) and is shown, and stops when it is hidden or leaves the screen; nothing runs off screen or
/// in tests (no handler).
/// </summary>
public static class Motion
{
    /// <summary>Whether <paramref name="view"/> can animate now (it is on screen and shown).</summary>
    public static bool CanAnimate(VisualElement view) => view is { Handler: not null, IsVisible: true };

    /// <summary>Fades <paramref name="view"/> in (from clear) when it can animate; else shows it at once.</summary>
    public static void FadeIn(VisualElement view, uint milliseconds = 160)
    {
        ArgumentNullException.ThrowIfNull(view);
        view.AbortAnimation("IdrakFade");
        if (!CanAnimate(view))
        {
            view.Opacity = 1;
            return;
        }

        view.Opacity = 0;
        view.Animate("IdrakFade", new Animation(v => view.Opacity = v, 0, 1), length: milliseconds, easing: Easing.CubicOut);
    }
}

/// <summary>The streaming caret: a small bar in the caret colour that blinks while <see cref="IsBlinking"/> and on screen.</summary>
public sealed class CaretView : BoxView
{
    /// <summary>Whether it blinks (it is hidden when not).</summary>
    public static readonly BindableProperty IsBlinkingProperty =
        BindableProperty.Create(nameof(IsBlinking), typeof(bool), typeof(CaretView), false, propertyChanged: (b, _, _) => ((CaretView)b).Update());

    private const string Blink = "IdrakCaret";

    /// <summary>A caret 8 by 16.</summary>
    public CaretView()
    {
        WidthRequest = 8;
        HeightRequest = 16;
        CornerRadius = 2;
        IsVisible = false;
        HorizontalOptions = LayoutOptions.Start;
        VerticalOptions = LayoutOptions.End;
        SetDynamicResource(ColorProperty, "IdrakCaret");
        HandlerChanged += (_, _) => Update();
    }

    /// <summary>Whether it blinks.</summary>
    public bool IsBlinking
    {
        get => (bool)GetValue(IsBlinkingProperty);
        set => SetValue(IsBlinkingProperty, value);
    }

    private void Update()
    {
        IsVisible = IsBlinking;
        this.AbortAnimation(Blink);
        Opacity = 1;
        if (IsBlinking && Motion.CanAnimate(this))
        {
            this.Animate(Blink, new Animation(v => Opacity = v < 0.5 ? 1 : 0.15, 0, 1), length: 1000, repeat: () => IsBlinking && Handler is not null);
        }
    }
}

/// <summary>
/// The big record button (the design's red disc): a white square while recording, a white dot when idle; its halo pulses
/// while recording and on screen. Tapping runs <see cref="Command"/>.
/// </summary>
public sealed class RecordButton : Grid
{
    /// <summary>Whether it records (the square, and the pulse).</summary>
    public static readonly BindableProperty IsRecordingProperty =
        BindableProperty.Create(nameof(IsRecording), typeof(bool), typeof(RecordButton), false, propertyChanged: (b, _, _) => ((RecordButton)b).Update());

    /// <summary>What a tap runs.</summary>
    public static readonly BindableProperty CommandProperty = BindableProperty.Create(nameof(Command), typeof(System.Windows.Input.ICommand), typeof(RecordButton));

    private const string Pulse = "IdrakPulse";
    private readonly Border _halo = new();
    private readonly Border _mark = new() { StrokeThickness = 0, HorizontalOptions = LayoutOptions.Center, VerticalOptions = LayoutOptions.Center };

    /// <summary>A record button.</summary>
    public RecordButton()
    {
        WidthRequest = 122;
        HeightRequest = 122;
        _halo.SetDynamicResource(StyleProperty, "IdrakRecordHalo");
        var disc = new Border { HorizontalOptions = LayoutOptions.Center, VerticalOptions = LayoutOptions.Center, Content = _mark };
        disc.SetDynamicResource(StyleProperty, "IdrakRecordButton");
        _mark.SetDynamicResource(BackgroundColorProperty, "IdrakWhite");
        Children.Add(_halo);
        Children.Add(disc);
        var tap = new TapGestureRecognizer();
        tap.Tapped += (_, _) =>
        {
            if (Command is { } command && command.CanExecute(null))
            {
                command.Execute(null);
            }
        };
        GestureRecognizers.Add(tap);
        SemanticProperties.SetDescription(this, "Record");
        HandlerChanged += (_, _) => Update();
        Update();
    }

    /// <summary>Whether it records.</summary>
    public bool IsRecording
    {
        get => (bool)GetValue(IsRecordingProperty);
        set => SetValue(IsRecordingProperty, value);
    }

    /// <summary>What a tap runs.</summary>
    public System.Windows.Input.ICommand? Command
    {
        get => (System.Windows.Input.ICommand?)GetValue(CommandProperty);
        set => SetValue(CommandProperty, value);
    }

    private void Update()
    {
        bool on = IsRecording;
        _mark.WidthRequest = on ? 26 : 22;
        _mark.HeightRequest = on ? 26 : 22;
        _mark.StrokeShape = new RoundRectangle { CornerRadius = on ? 7 : 11 };
        SemanticProperties.SetDescription(this, on ? "Stop" : "Record");
        this.AbortAnimation(Pulse);
        _halo.Scale = 1;
        _halo.Opacity = on ? 1 : 0.5;
        if (on && Motion.CanAnimate(this))
        {
            this.Animate(Pulse, new Animation(v =>
            {
                _halo.Scale = 0.86 + 0.14 * v;
                _halo.Opacity = 1 - 0.55 * v;
            }, 0, 1), length: 1200, easing: Easing.SinInOut, repeat: () => IsRecording && Handler is not null);
        }
    }
}
