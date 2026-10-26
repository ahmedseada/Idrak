// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Theme;
using Microsoft.Maui.Controls.Shapes;

namespace Idrak.Maui.Controls;

/// <summary>
/// One of Idrak's icons (<see cref="IdrakIcons"/>) drawn as a stroke at <see cref="Size"/> (the design's 24-unit view box
/// scaled), in <see cref="Stroke"/> (the theme's text colour unless set), with round caps and joins and no fill.
/// </summary>
public class IconView : ContentView
{
    /// <summary>The icon's name ("home").</summary>
    public static readonly BindableProperty GlyphProperty =
        BindableProperty.Create(nameof(Glyph), typeof(string), typeof(IconView), "", propertyChanged: (b, _, n) => ((IconView)b)._path.Data = IdrakIcons.Get(n as string));

    /// <summary>The icon's width and height.</summary>
    public static readonly BindableProperty SizeProperty =
        BindableProperty.Create(nameof(Size), typeof(double), typeof(IconView), 20d, propertyChanged: (b, _, n) => ((IconView)b).Resize((double)n));

    /// <summary>The stroke's brush.</summary>
    public static readonly BindableProperty StrokeProperty =
        BindableProperty.Create(nameof(Stroke), typeof(Brush), typeof(IconView), propertyChanged: (b, _, n) => ((IconView)b)._path.Stroke = n as Brush ?? Brush.Transparent);

    private readonly Microsoft.Maui.Controls.Shapes.Path _path = new()
    {
        Aspect = Stretch.None,
        StrokeThickness = 1.8,
        StrokeLineCap = PenLineCap.Round,
        StrokeLineJoin = PenLineJoin.Round,
        InputTransparent = true,
    };

    /// <summary>An icon of 20 units in the theme's text colour.</summary>
    public IconView()
    {
        HorizontalOptions = LayoutOptions.Center;
        VerticalOptions = LayoutOptions.Center;
        InputTransparent = true;
        Content = _path;
        SetDynamicResource(StrokeProperty, "IdrakText");
        Resize(Size);
    }

    /// <summary>The icon's name ("home").</summary>
    public string Glyph
    {
        get => (string)GetValue(GlyphProperty);
        set => SetValue(GlyphProperty, value);
    }

    /// <summary>The icon's width and height.</summary>
    public double Size
    {
        get => (double)GetValue(SizeProperty);
        set => SetValue(SizeProperty, value);
    }

    /// <summary>The stroke's brush (a theme colour set through DynamicResource).</summary>
    public Brush? Stroke
    {
        get => (Brush?)GetValue(StrokeProperty);
        set => SetValue(StrokeProperty, value);
    }

    /// <summary>The icon's geometry, or null for an unknown name.</summary>
    public Geometry? Geometry => _path.Data;

    /// <summary>The scale from the 24-unit view box to <see cref="Size"/>.</summary>
    public Transform? ViewBoxScale => _path.RenderTransform;

    private void Resize(double size)
    {
        WidthRequest = size;
        HeightRequest = size;
        _path.WidthRequest = size;
        _path.HeightRequest = size;
        _path.RenderTransform = new ScaleTransform(size / 24, size / 24);
    }
}

/// <summary>
/// An icon on a tinted tile (the design's lab badges): <see cref="Tone"/> names the pair of theme colours used,
/// "IdrakTone{Tone}" for the icon and "IdrakTone{Tone}Background" for the tile ("Indigo", "Violet", "Green", "Rose",
/// "Amber", "Sky" or "Neutral").
/// </summary>
public class IconBadge : Border
{
    /// <summary>The icon's name.</summary>
    public static readonly BindableProperty GlyphProperty =
        BindableProperty.Create(nameof(Glyph), typeof(string), typeof(IconBadge), "", propertyChanged: (b, _, n) => ((IconBadge)b)._icon.Glyph = n as string ?? "");

    /// <summary>The tone.</summary>
    public static readonly BindableProperty ToneProperty =
        BindableProperty.Create(nameof(Tone), typeof(string), typeof(IconBadge), "Neutral", propertyChanged: (b, _, n) => ((IconBadge)b).ApplyTone(n as string));

    /// <summary>The tile's width and height (the icon is a little over half of it).</summary>
    public static readonly BindableProperty BadgeSizeProperty =
        BindableProperty.Create(nameof(BadgeSize), typeof(double), typeof(IconBadge), 34d, propertyChanged: (b, _, n) => ((IconBadge)b).Resize((double)n));

    private readonly IconView _icon = new();

    /// <summary>A neutral badge of 34 units.</summary>
    public IconBadge()
    {
        StrokeThickness = 0;
        Padding = 0;
        HorizontalOptions = LayoutOptions.Start;
        VerticalOptions = LayoutOptions.Center;
        Content = _icon;
        ApplyTone(Tone);
        Resize(BadgeSize);
    }

    /// <summary>The icon's name.</summary>
    public string Glyph
    {
        get => (string)GetValue(GlyphProperty);
        set => SetValue(GlyphProperty, value);
    }

    /// <summary>The tone ("Indigo", "Violet", "Green", "Rose", "Amber", "Sky", "Neutral").</summary>
    public string Tone
    {
        get => (string)GetValue(ToneProperty);
        set => SetValue(ToneProperty, value);
    }

    /// <summary>The tile's width and height.</summary>
    public double BadgeSize
    {
        get => (double)GetValue(BadgeSizeProperty);
        set => SetValue(BadgeSizeProperty, value);
    }

    private void ApplyTone(string? tone)
    {
        string name = string.IsNullOrEmpty(tone) ? "Neutral" : tone;
        SetDynamicResource(BackgroundColorProperty, $"IdrakTone{name}Background");
        _icon.SetDynamicResource(IconView.StrokeProperty, $"IdrakTone{name}");
    }

    private void Resize(double size)
    {
        WidthRequest = size;
        HeightRequest = size;
        StrokeShape = new RoundRectangle { CornerRadius = new CornerRadius(Math.Round(size / 3.1)) };
        _icon.Size = Math.Round(size * 0.56);
    }
}
