// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Controls;

/// <summary>
/// Its content zoomed by a pinch (about the pinch's point, from 1 to <see cref="MaximumScale"/>), moved by a drag while
/// zoomed (held inside its bounds), and back to the whole on a double tap. The content is scaled and moved as it draws
/// (Scale and Translation): nothing is laid out again while it zooms.
/// </summary>
public sealed class ZoomView : ContentView
{
    /// <summary>The most it zooms.</summary>
    public static readonly BindableProperty MaximumScaleProperty = BindableProperty.Create(nameof(MaximumScale), typeof(double), typeof(ZoomView), 6d);

    private double _startX, _startY;

    /// <summary>A zoom view (its content clipped to it).</summary>
    public ZoomView()
    {
        IsClippedToBounds = true;
        var pinch = new PinchGestureRecognizer();
        pinch.PinchUpdated += OnPinch;
        var pan = new PanGestureRecognizer();
        pan.PanUpdated += OnPan;
        var twice = new TapGestureRecognizer { NumberOfTapsRequired = 2 };
        twice.Tapped += (_, _) => Reset();
        GestureRecognizers.Add(pinch);
        GestureRecognizers.Add(pan);
        GestureRecognizers.Add(twice);
    }

    /// <summary>The most it zooms.</summary>
    public double MaximumScale
    {
        get => (double)GetValue(MaximumScaleProperty);
        set => SetValue(MaximumScaleProperty, value);
    }

    /// <summary>How far it is zoomed (1: the whole).</summary>
    public double Zoom => Content?.Scale ?? 1;

    /// <summary>Back to the whole.</summary>
    public void Reset()
    {
        if (Content is { } content)
        {
            content.Scale = 1;
            content.TranslationX = 0;
            content.TranslationY = 0;
        }
    }

    /// <summary>Zooms to <paramref name="scale"/> about the point at shares <paramref name="originX"/>, <paramref name="originY"/> of the view (0 to 1).</summary>
    public void ZoomTo(double scale, double originX = 0.5, double originY = 0.5)
    {
        if (Content is not { } content || Width <= 0 || Height <= 0)
        {
            return;
        }

        double next = Math.Clamp(scale, 1, Math.Max(1, MaximumScale));
        double before = content.Scale;
        // The point under the fingers stays under them: move by how far it would drift as the scale changes.
        double x = content.TranslationX - (originX - 0.5) * Width * (next - before);
        double y = content.TranslationY - (originY - 0.5) * Height * (next - before);
        content.Scale = next;
        (content.TranslationX, content.TranslationY) = Hold(x, y, next);
    }

    private void OnPinch(object? sender, PinchGestureUpdatedEventArgs e)
    {
        if (Content is not { } content)
        {
            return;
        }

        if (e.Status == GestureStatus.Running)
        {
            ZoomTo(content.Scale * e.Scale, e.ScaleOrigin.X, e.ScaleOrigin.Y);
        }
    }

    private void OnPan(object? sender, PanUpdatedEventArgs e)
    {
        if (Content is not { } content || content.Scale <= 1)
        {
            return;
        }

        switch (e.StatusType)
        {
            case GestureStatus.Started:
                (_startX, _startY) = (content.TranslationX, content.TranslationY);
                break;
            case GestureStatus.Running:
                (content.TranslationX, content.TranslationY) = Hold(_startX + e.TotalX, _startY + e.TotalY, content.Scale);
                break;
        }
    }

    // A move that keeps the zoomed content covering the view.
    private (double X, double Y) Hold(double x, double y, double scale)
    {
        double spareX = Width * (scale - 1) / 2, spareY = Height * (scale - 1) / 2;
        return (Math.Clamp(x, -spareX, spareX), Math.Clamp(y, -spareY, spareY));
    }
}
