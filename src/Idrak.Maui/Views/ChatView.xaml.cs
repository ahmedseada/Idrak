// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Chat's view: reads the store's models and the kept chats each time it is shown, keeps the composer above the keyboard
/// (the host's bottom edge follows it: <c>AppHostPage.KeyboardAware</c>) with the conversation still at its end as the
/// keyboard takes its room, follows the streaming answer to its end, copies with the clipboard, opens Models (+); the
/// microphone dictates into the message (<see cref="ChatViewModel.DictateCommand"/>); the sliders open the chosen model's
/// generation settings (a sheet from the bottom; on a wide window, the inspector's column). Wide: the conversations on the
/// left and the inspector on the right.
/// </summary>
public partial class ChatView : AdaptiveView
{
    private readonly IAppNavigator _navigator;
    private double _listHeight;
    private bool _follow = true, _scrollQueued;
    private long _scrolledAt;                                                   // when this view last scrolled the list itself (ms)

    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> open Models.</summary>
    public ChatView(ChatViewModel viewModel, IServiceProvider services)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        ArgumentNullException.ThrowIfNull(services);
        _navigator = Dialogs.Navigator(services);
        BindingContext = viewModel;
        viewModel.Copy = text => Clipboard.Default.SetTextAsync(text);
        viewModel.OpenModels = () => _navigator.OpenFeatureAsync(MobileFeatures.Models);
        InitializeComponent();
        MessageList.SizeChanged += OnListSize;
        MessageList.Scrolled += OnListScrolled;
        viewModel.Messages.CollectionChanged += (_, e) => _follow |= e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Add;   // a message sent: follow it
        Root.SizeChanged += (_, _) => SizeSheet();
    }

    /// <summary>The view model.</summary>
    public ChatViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        ViewModel.ScrollToEndRequested -= OnScrollToEnd;
        ViewModel.ScrollToEndRequested += OnScrollToEnd;
        var conversations = ViewModel.ReloadConversationsAsync();               // read on a worker thread, listed when read
        await ViewModel.RefreshAsync();
        await conversations;
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        ViewModel.ScrollToEndRequested -= OnScrollToEnd;
        ViewModel.IsTuning = false;
        base.Hidden();
    }

    /// <inheritdoc />
    /// <remarks>The tune sheet (narrow) or the inspector's settings (wide).</remarks>
    public override bool CloseSheet()
    {
        if (!ViewModel.IsTuning)
        {
            return false;
        }

        ViewModel.IsTuning = false;
        return true;
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Root.ColumnDefinitions = wide
            ? [new ColumnDefinition(new GridLength(290)), new ColumnDefinition(GridLength.Star), new ColumnDefinition(new GridLength(300))]
            : [new ColumnDefinition(GridLength.Star)];
        ListPane.IsVisible = wide;
        Inspector.IsVisible = wide;
        NewChat.IsVisible = !wide;
        Grid.SetColumn(Main, wide ? 1 : 0);
        Grid.SetColumn(Inspector, wide ? 2 : 0);
        MessageList.SetDynamicResource(MarginProperty, wide ? "IdrakWideFeedPadding" : "IdrakFeedPadding");
        Top.SetDynamicResource(PaddingProperty, wide ? "IdrakWideChatTopPadding" : "IdrakChatTopPadding");

        // The tune sheet: from the bottom over the chat (narrow), or in the inspector's column (wide).
        Grid.SetColumn(TuningLayer, wide ? 2 : 0);
        TuningScrim.IsVisible = !wide;
        TuningHandle.IsVisible = !wide;
        TuningPanel.VerticalOptions = wide ? LayoutOptions.Fill : LayoutOptions.End;
        if (wide)
        {
            TuningPanel.Style = null;
            TuningPanel.StrokeThickness = 0;
            TuningPanel.SetDynamicResource(Border.BackgroundColorProperty, "IdrakRailBackground");
            TuningPanel.SetDynamicResource(PaddingProperty, "IdrakPanePadding");
        }
        else
        {
            TuningPanel.SetDynamicResource(StyleProperty, "IdrakSheet");
        }

        SizeSheet();
    }

    // The narrow window's sheet takes most of the view's height (what the keyboard leaves of it); the wide pane all of it.
    private void SizeSheet() => TuningPanel.HeightRequest = IsWide || Root.Height <= 0 ? -1 : Math.Round(Root.Height * 0.86);

    // The conversation keeps its end in view when its room shrinks (the keyboard comes up, the composer grows).
    private void OnListSize(object? sender, EventArgs e)
    {
        double height = MessageList.Height;
        if (height > 0 && height < _listHeight)
        {
            OnScrollToEnd(sender, e);
        }

        _listHeight = height;
    }

    // The end kept in view while it is followed: scrolled once the layout has measured what changed (a reply's bubble grows
    // after its text is set, so scrolling at once would reach its old end), at most once a turn of the UI thread.
    private void OnScrollToEnd(object? sender, EventArgs e)
    {
        if (!_follow || _scrollQueued)
        {
            return;
        }

        _scrollQueued = true;
        if (Dispatcher?.Dispatch(ScrollNow) != true)
        {
            ScrollNow();
        }
    }

    private void ScrollNow()
    {
        _scrollQueued = false;
        if (_follow && ViewModel.Messages.Count > 0)
        {
            _scrolledAt = Environment.TickCount64;
            ListEnd.ScrollTo(MessageList, ViewModel.Messages.Count - 1);
        }
    }

    // The last reply grew (a streamed piece laid out): its new end brought into view while the chat is followed.
    private void OnBubbleSize(object? sender, EventArgs e)
    {
        if (sender is BindableObject { BindingContext: ChatBubble bubble } && ViewModel.Messages.Count > 0 && ReferenceEquals(bubble, ViewModel.Messages[^1]))
        {
            OnScrollToEnd(sender, e);
        }
    }

    // Followed while the list shows its end: scrolled away from it (up, to read) it is no longer; back at its end it is again.
    // The view's own scrolling and its settling do not count.
    private void OnListScrolled(object? sender, ItemsViewScrolledEventArgs e)
    {
        if (Environment.TickCount64 - _scrolledAt < 300)
        {
            return;
        }

        bool atEnd = ListEnd.IsAtEnd(MessageList) ?? e.LastVisibleItemIndex >= ViewModel.Messages.Count - 1;
        if (atEnd)
        {
            _follow = true;
        }
        else if (e.VerticalDelta < 0)
        {
            _follow = false;
        }
    }

    private void OnSuggestion(object? sender, TappedEventArgs e)
    {
        if (e.Parameter is string text)
        {
            ViewModel.SuggestCommand.Execute(text);
        }
    }
}

/// <summary>The chat's two bubbles: the user's and the model's.</summary>
public sealed class ChatBubbleTemplates : DataTemplateSelector
{
    /// <summary>The user's message.</summary>
    public DataTemplate? User { get; set; }

    /// <summary>The model's answer.</summary>
    public DataTemplate? Assistant { get; set; }

    /// <inheritdoc />
    protected override DataTemplate OnSelectTemplate(object item, BindableObject container) =>
        (item is ChatBubble { IsUser: true } ? User : Assistant) ?? throw new InvalidOperationException("The chat's bubble templates are not set.");
}

/// <summary>The tune sheet's settings, one template a kind: a number with a slider, a field, lines, a choice.</summary>
public sealed class ChatReplyTemplates : DataTemplateSelector
{
    /// <summary>A number with a range: a slider and its field.</summary>
    public DataTemplate? Slider { get; set; }

    /// <summary>A number without a range, or a short text: a field.</summary>
    public DataTemplate? Field { get; set; }

    /// <summary>Several lines (the system prompt, the stop texts).</summary>
    public DataTemplate? Lines { get; set; }

    /// <summary>Default, on or off.</summary>
    public DataTemplate? Choice { get; set; }

    /// <inheritdoc />
    protected override DataTemplate OnSelectTemplate(object item, BindableObject container) => (item switch
    {
        ChatReplyEntry { IsChoice: true } => Choice,
        ChatReplyEntry { IsLines: true } => Lines,
        ChatReplyEntry { HasSlider: true } => Slider,
        _ => Field,
    }) ?? throw new InvalidOperationException("The tune sheet's templates are not set.");
}
