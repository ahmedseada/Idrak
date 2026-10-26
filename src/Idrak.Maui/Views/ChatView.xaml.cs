// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;
using AndroidPlatform = Microsoft.Maui.Controls.PlatformConfiguration.Android;   // "Android" alone is the platform's namespace in the Android build
using Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific;

namespace Idrak.Maui.Views;

/// <summary>
/// Chat's view: reads the store's models and the kept chats each time it is shown, keeps the composer above the keyboard
/// (the window resizes while it is shown), follows the streaming answer to its end, copies with the clipboard, opens Models
/// (+); the microphone dictates into the message (<see cref="ChatViewModel.DictateCommand"/>). Wide: the conversations on the
/// left and the inspector on the right.
/// </summary>
public partial class ChatView : AdaptiveView
{
    private readonly IAppNavigator _navigator;
    private WindowSoftInputModeAdjust? _keyboard;

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
    }

    /// <summary>The view model.</summary>
    public ChatViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        if (Microsoft.Maui.Controls.Application.Current is { } app)
        {
            var android = app.On<AndroidPlatform>();
            _keyboard ??= android.GetWindowSoftInputModeAdjust();
            android.UseWindowSoftInputModeAdjust(WindowSoftInputModeAdjust.Resize);
        }

        ViewModel.ScrollToEndRequested -= OnScrollToEnd;
        ViewModel.ScrollToEndRequested += OnScrollToEnd;
        ViewModel.ListConversations();
        await ViewModel.RefreshAsync();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        ViewModel.ScrollToEndRequested -= OnScrollToEnd;
        if (Microsoft.Maui.Controls.Application.Current is { } app && _keyboard is { } keyboard)
        {
            app.On<AndroidPlatform>().UseWindowSoftInputModeAdjust(keyboard);
        }

        base.Hidden();
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
    }

    private void OnScrollToEnd(object? sender, EventArgs e)
    {
        if (ViewModel.Messages.Count > 0 && MessageList.Handler is not null)
        {
            MessageList.ScrollTo(ViewModel.Messages.Count - 1, position: ScrollToPosition.End, animate: false);
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
