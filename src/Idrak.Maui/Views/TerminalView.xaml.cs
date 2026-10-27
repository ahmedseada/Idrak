// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.ComponentModel;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Terminal's view: while a command runs and the view is shown, it drains what the command wrote once a display frame
/// (<see cref="FrameClocks.Current"/>) into the list, one batch a frame, and follows the end unless the list was scrolled
/// up. Wide: the commands pane beside the output; narrow: in its place when asked.
/// </summary>
public partial class TerminalView : AdaptiveView
{
    private IDisposable? _ticking;
    private bool _onScreen, _follow = true;

    /// <summary>A view bound to <paramref name="viewModel"/>.</summary>
    public TerminalView(TerminalViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        viewModel.Copy ??= text => Clipboard.Default.SetTextAsync(text);
        InitializeComponent();
        viewModel.PropertyChanged += OnViewModel;
        viewModel.LinesAdded += (_, _) => FollowEnd();
        Output.Scrolled += (_, e) => _follow = e.LastVisibleItemIndex >= ViewModel.Lines.Count - 2;
    }

    /// <summary>The view model.</summary>
    public TerminalViewModel ViewModel { get; }

    /// <summary>Whether it is joined to the frame clock (a command runs and it is shown).</summary>
    public bool IsTicking => _ticking is not null;

    /// <inheritdoc />
    public override void Shown()
    {
        base.Shown();
        _onScreen = true;
        ViewModel.Drain();                                                       // what came while it was hidden
        UpdateTicking();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        _onScreen = false;
        UpdateTicking();
        base.Hidden();
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Root.ColumnDefinitions = wide ? [new ColumnDefinition(GridLength.Star), new ColumnDefinition(new GridLength(0.55, GridUnitType.Star))] : [new ColumnDefinition(GridLength.Star)];
        Place(CommandsPane, wide ? 0 : 1, wide ? 1 : 0, 1, wide ? 4 : 1);
        CommandsButton.IsVisible = !wide;
        ShowPanes();
    }

    private void OnViewModel(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(TerminalViewModel.IsRunning))
        {
            _follow = _follow || ViewModel.IsRunning;
            UpdateTicking();
        }
        else if (e.PropertyName == nameof(TerminalViewModel.ShowCommands))
        {
            ShowPanes();
        }
    }

    private void ShowPanes()
    {
        CommandsPane.IsVisible = IsWide || ViewModel.ShowCommands;
        OutputCard.IsVisible = IsWide || !ViewModel.ShowCommands;
    }

    // The frame clock while a command runs on screen; nothing ticks otherwise.
    private void UpdateTicking()
    {
        bool run = _onScreen && ViewModel.IsRunning && Handler is not null;
        if (run && _ticking is null)
        {
            _ticking = FrameClocks.Current.Subscribe(() => ViewModel.Drain(), FrameClocks.FramesPerSecond);
        }
        else if (!run && _ticking is not null)
        {
            _ticking.Dispose();
            _ticking = null;
        }
    }

    private void FollowEnd()
    {
        if (_follow && ViewModel.Lines.Count > 0 && Output.Handler is not null)
        {
            Output.ScrollTo(ViewModel.Lines.Count - 1, position: ScrollToPosition.End, animate: false);
        }
    }

    private void OnCompleted(object? sender, EventArgs e) => ViewModel.RunCommand.Execute(null);

    private void OnHistory(object? sender, EventArgs e)
    {
        if (sender is Button { Text: { Length: > 0 } line })
        {
            ViewModel.Input = line;
            ViewModel.ShowCommands = false;
            Line.Focus();
        }
    }
}
