// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Controls;

/// <summary>
/// A view the app's navigation hosts (<see cref="IHostedView"/>) that lays itself out for the window: wide (columns,
/// panes) or narrow (one column) as the shared <see cref="AppLayout"/> measures the window, or, without one (a page of its
/// own), as its own width compares with the layout's breakpoint. <see cref="Arrange"/> moves its parts when that changes.
/// </summary>
public abstract class AdaptiveView : ContentView, IHostedView
{
    private readonly AppLayout? _layout;
    private readonly double _breakpoint;
    private bool _wide;
    private bool _arranged;

    /// <summary>A view laid out by the app's layout (from <paramref name="services"/>), or by its own width.</summary>
    protected AdaptiveView(IServiceProvider? services)
    {
        _layout = services?.GetService<AppLayout>();
        _breakpoint = _layout?.Breakpoint ?? (services?.GetService<IdrakMauiOptions>()?.WideLayoutWidth ?? 840);
        _wide = _layout?.IsWide ?? false;
        Loaded += (_, _) =>
        {
            if (_layout is not null)
            {
                _layout.LayoutChanged += OnLayoutChanged;
                Apply(_layout.IsWide);
            }
        };
        Unloaded += (_, _) =>
        {
            if (_layout is not null)
            {
                _layout.LayoutChanged -= OnLayoutChanged;
            }
        };
        SizeChanged += (_, _) =>
        {
            if (_layout is null && Width > 0)
            {
                Apply(AppLayout.IsWideAt(Width, _breakpoint));
            }
        };
    }

    /// <summary>Whether the wide layout shows.</summary>
    public bool IsWide => _wide;

    /// <summary>Lays the view out for <paramref name="wide"/> now (the host calls it as the window changes; tests call it).</summary>
    public void Apply(bool wide)
    {
        if (_arranged && wide == _wide)
        {
            return;
        }

        _wide = wide;
        _arranged = true;
        Arrange(wide);
    }

    /// <inheritdoc />
    public virtual void Shown()
    {
        if (!_arranged)
        {
            Apply(_layout?.IsWide ?? _wide);
        }
    }

    /// <inheritdoc />
    public virtual void Hidden()
    {
    }

    /// <inheritdoc />
    public virtual bool ShowFeature(string feature) => false;

    /// <inheritdoc />
    public virtual bool CloseSheet() => false;

    /// <summary>Moves the view's parts for the wide or the narrow layout.</summary>
    protected abstract void Arrange(bool wide);

    /// <summary>Places <paramref name="view"/> in a grid's cell.</summary>
    protected static void Place(View view, int row, int column, int columnSpan = 1, int rowSpan = 1)
    {
        Grid.SetRow(view, row);
        Grid.SetColumn(view, column);
        Grid.SetColumnSpan(view, columnSpan);
        Grid.SetRowSpan(view, rowSpan);
    }

    private void OnLayoutChanged(object? sender, bool wide) => Dispatcher.Dispatch(() => Apply(wide));
}
