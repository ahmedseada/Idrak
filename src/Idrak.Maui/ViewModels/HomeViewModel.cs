// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;

namespace Idrak.Maui.ViewModels;

/// <summary>One slide of Home's carousel, its "{device}" read for the device the app runs on.</summary>
public sealed class SlideItem(HomeSlide slide, string noun, ICommand open)
{
    /// <summary>The slide as given.</summary>
    public HomeSlide Slide { get; } = slide;

    /// <summary>The small capitals above the title.</summary>
    public string Kicker { get; } = DeviceWords.Fill(slide.Kicker, noun);

    /// <summary>The title's start.</summary>
    public string Title { get; } = DeviceWords.Fill(slide.Title, noun);

    /// <summary>The title's end, in the accent.</summary>
    public string Emphasis { get; } = DeviceWords.Fill(slide.Emphasis, noun);

    /// <summary>Its text.</summary>
    public string Text { get; } = DeviceWords.Fill(slide.Text, noun);

    /// <summary>The call to action's label.</summary>
    public string Action { get; } = slide.Action;

    /// <summary>Its ground's tone ("Sky", "Amber", "Green", "Violet").</summary>
    public string Tone { get; } = slide.Tone;

    /// <summary>Whether its art is Idrak's mark.</summary>
    public bool IsLogo => Slide.Art == "logo";

    /// <summary>Whether its art is one of Idrak's icons.</summary>
    public bool IsIcon => !IsLogo && Slide.Art.Length > 0;

    /// <summary>The icon's name when <see cref="IsIcon"/>.</summary>
    public string Art => Slide.Art;

    /// <summary>Opens what the slide's action names.</summary>
    public ICommand OpenCommand { get; } = open;
}

/// <summary>
/// Home (the approved design's landing): the source commit the app was built from, the device's status chip, the carousel
/// of the app's slides (<see cref="HomeOptions"/>) moving by itself every <see cref="IdrakMauiOptions.CarouselInterval"/>
/// while shown, this device measured (<see cref="DeviceStatus"/>), the labs to try with live previews (the chat's last
/// exchange and its speed), and on a wide window the models in the store.
/// </summary>
public sealed class HomeViewModel : ViewModel
{
    private readonly IdrakMauiOptions _options;
    private readonly IAppNavigator? _navigator;
    private readonly ChatViewModel? _chat;
    private int _position;
    private string _chatQuestion, _chatAnswer, _chatRate = "", _chatModel = "";

    /// <summary>Home over the app's options and device status; <paramref name="chat"/> gives the live chat preview, <paramref name="navigator"/> opens the labs.</summary>
    public HomeViewModel(IdrakMauiOptions options, DeviceStatus status, ChatViewModel? chat = null, IAppNavigator? navigator = null, SourceInfo? source = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Status = status ?? throw new ArgumentNullException(nameof(status));
        _chat = chat;
        _navigator = navigator;
        Source = source ?? SourceInfo.Library;
        Noun = DeviceWords.Noun;
        var open = new Command<string>(async target => await OpenAsync(target ?? ""));
        OpenCommand = open;
        foreach (var slide in options.Home.Slides)
        {
            Slides.Add(new SlideItem(slide, Noun, new Command(async () => await OpenAsync(slide.Target))));
        }

        _chatQuestion = options.Home.ChatSampleQuestion;
        _chatAnswer = options.Home.ChatSampleAnswer;
        SpeakLine = options.Home.SpeakSample;
        NextSlideCommand = new Command(NextSlide);
    }

    /// <summary>The device's status (shared with All labs).</summary>
    public DeviceStatus Status { get; }

    /// <summary>The source the app was built from.</summary>
    public SourceInfo Source { get; }

    /// <summary>"main · 628b6bb".</summary>
    public string SourceText => Source.Text;

    /// <summary>"phone", "tablet" or "device".</summary>
    public string Noun { get; }

    /// <summary>"This phone, measured".</summary>
    public string MeasuredTitle => $"This {Noun}, measured";

    /// <summary>"Models on this phone".</summary>
    public string ModelsTitle => $"Models on this {Noun}";

    /// <summary>The slides.</summary>
    public ObservableCollection<SlideItem> Slides { get; } = [];

    /// <summary>The slide shown.</summary>
    public int Position
    {
        get => _position;
        set
        {
            if (SetProperty(ref _position, Slides.Count == 0 ? 0 : Math.Clamp(value, 0, Slides.Count - 1)))
            {
                OnPropertyChanged(nameof(SlideCaption));
            }
        }
    }

    /// <summary>"1 of 4 · The library".</summary>
    public string SlideCaption => Slides.Count == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"{Position + 1} of {Slides.Count} · {Slides[Position].Slide.Caption}");

    /// <summary>How long a slide shows before the next (zero: it stays).</summary>
    public TimeSpan SlideInterval => _options.CarouselInterval is { } interval && interval > TimeSpan.Zero ? interval : TimeSpan.Zero;

    /// <summary>The chat tile's question.</summary>
    public string ChatQuestion
    {
        get => _chatQuestion;
        private set => SetProperty(ref _chatQuestion, value);
    }

    /// <summary>The chat tile's answer.</summary>
    public string ChatAnswer
    {
        get => _chatAnswer;
        private set => SetProperty(ref _chatAnswer, value);
    }

    /// <summary>The chat's last speed ("17.4 tok/s on vulkan:0"), or "".</summary>
    public string ChatRate
    {
        get => _chatRate;
        private set
        {
            if (SetProperty(ref _chatRate, value))
            {
                OnPropertyChanged(nameof(HasChatRate));
            }
        }
    }

    /// <summary>Whether the chat has a speed to show.</summary>
    public bool HasChatRate => ChatRate.Length > 0;

    /// <summary>The chat's model ("Qwen3-0.6B · int8"), or what to do.</summary>
    public string ChatModel
    {
        get => _chatModel.Length > 0 ? _chatModel : "Stream a model";
        private set => SetProperty(ref _chatModel, value);
    }

    /// <summary>The Speak tile's line.</summary>
    public string SpeakLine { get; }

    /// <summary>Opens a section or a feature by name (the tiles, the links).</summary>
    public ICommand OpenCommand { get; }

    /// <summary>Shows the next slide (the first after the last).</summary>
    public ICommand NextSlideCommand { get; }

    /// <summary>Shows the next slide, the first after the last.</summary>
    public void NextSlide() => Position = Slides.Count == 0 ? 0 : (Position + 1) % Slides.Count;

    /// <summary>Reads the device once and the chat's last exchange again (Home shown).</summary>
    public async Task RefreshAsync()
    {
        ReadChat();
        await Status.EnsureReadAsync().ConfigureAwait(true);
        await Status.RefreshModelsAsync().ConfigureAwait(true);
    }

    /// <summary>Opens <paramref name="target"/>: the section of that name, else the feature, else nothing.</summary>
    public async Task OpenAsync(string target)
    {
        if (_navigator is null || target.Length == 0)
        {
            return;
        }

        if (string.Equals(target, AppSections.Labs, StringComparison.OrdinalIgnoreCase))
        {
            await _navigator.OpenLabsAsync().ConfigureAwait(true);
        }
        else if (string.Equals(target, AppSections.Settings, StringComparison.OrdinalIgnoreCase))
        {
            await _navigator.OpenSettingsAsync().ConfigureAwait(true);
        }
        else if (AppSections.Find(target) is not null && await _navigator.OpenSectionAsync(target).ConfigureAwait(true))
        {
            return;
        }
        else if (MobileFeatures.Find(target) is not null)
        {
            await _navigator.OpenFeatureAsync(target).ConfigureAwait(true);
        }
    }

    // The chat's last question and answer (their first line) and its speed, when it has run.
    private void ReadChat()
    {
        if (_chat is null)
        {
            return;
        }

        var user = _chat.Messages.LastOrDefault(m => m.IsUser);
        var answer = _chat.Messages.LastOrDefault(m => m.IsAssistant && m.Text.Length > 0);
        if (user is not null && answer is not null)
        {
            ChatQuestion = Line(user.Text, 40);
            ChatAnswer = Line(answer.Text, 48);
        }

        ChatRate = _chat.LastRateText;
        if (_chat.Model is { } model)
        {
            ChatModel = $"{model.Model[(model.Model.LastIndexOf('/') + 1)..]} · {model.Weights}";
        }
    }

    private static string Line(string text, int most)
    {
        string line = text.Trim().Split('\n')[0].Trim();
        return line.Length <= most ? line : line[..(most - 1)].TrimEnd() + "…";
    }
}
