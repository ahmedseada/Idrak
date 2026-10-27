// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows.Input;

namespace Idrak.Maui.ViewModels;

/// <summary>
/// One setting on the chat's tune sheet (<see cref="ChatReplySetting"/>): its value as text (empty: the default, shown
/// beside it), a slider for a number with a range, a choice of default, on and off for a switch, and why a value was
/// refused. A value that reads is kept for the model at once (<see cref="ChatViewModel"/> listens).
/// </summary>
public sealed class ChatReplyEntry : ViewModel
{
    private string _text;
    private string _default;
    private string _error = "";

    internal ChatReplyEntry(ChatReplySetting setting, string text, string defaultText, Action<ChatReplyEntry> changed)
    {
        Definition = setting;
        _text = text;
        _default = defaultText;
        Choices = setting.Setting.Choices;
        Changed = changed;
        ResetCommand = new Command(() => Text = "");
    }

    /// <summary>The setting.</summary>
    public ChatReplySetting Definition { get; }

    internal Action<ChatReplyEntry> Changed { get; }

    /// <summary>Its key.</summary>
    public string Key => Definition.Setting.Key;

    /// <summary>Its title.</summary>
    public string Title => Definition.Setting.Title;

    /// <summary>Its help.</summary>
    public string Summary => Definition.Setting.Summary;

    /// <summary>The default (the app's setting, else the library's), as the field's placeholder shows it.</summary>
    public string Default
    {
        get => _default;
        internal set
        {
            if (SetProperty(ref _default, value))
            {
                OnPropertyChanged(nameof(DefaultText));
                OnPropertyChanged(nameof(SliderValue));
            }
        }
    }

    /// <summary>"default 0.8".</summary>
    public string DefaultText => "default " + Default;

    /// <summary>Edited over several lines (the system prompt, the stop texts).</summary>
    public bool IsLines => Definition.Lines;

    /// <summary>Edited as a choice (default, on, off).</summary>
    public bool IsChoice => Definition.Setting.Kind == LabSettingKind.Choice;

    /// <summary>The keyboard for the field.</summary>
    public Keyboard Keyboard => Definition.Setting.Kind == LabSettingKind.Number ? Keyboard.Numeric : Keyboard.Plain;

    /// <summary>The values to choose from, the first standing for the default.</summary>
    public IReadOnlyList<string> Choices { get; }

    /// <summary>Whether a slider shows (a number with a range).</summary>
    public bool HasSlider => Definition.Range is not null;

    /// <summary>The slider's least value.</summary>
    public double Minimum => Definition.Range?.Minimum ?? 0;

    /// <summary>The slider's most value.</summary>
    public double Maximum => Definition.Range?.Maximum ?? 1;

    /// <summary>Whether a value of its own is set (the reset shows).</summary>
    public bool IsSet => Text.Length > 0;

    /// <summary>Whether a change loads the model again (the context).</summary>
    public bool Reloads => Definition.Reloads;

    /// <summary>The value as text ("" for the default).</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, (value ?? "").Trim()))
            {
                OnPropertyChanged(nameof(SelectedChoice));
                OnPropertyChanged(nameof(SliderValue));
                OnPropertyChanged(nameof(IsSet));
                Changed(this);
            }
        }
    }

    /// <summary>The value as a choice (the first choice standing for "").</summary>
    public string? SelectedChoice
    {
        get => Text.Length == 0 && Choices.Count > 0 ? Choices[0] : Text;
        set => Text = value is null || Choices.Count > 0 && value == Choices[0] ? "" : value;
    }

    /// <summary>The slider's position: the value, else the default; moving it sets the value, rounded to the slider's step.</summary>
    public double SliderValue
    {
        get => Read(Text.Length > 0 ? Text : Default) is { } value ? Math.Clamp(value, Minimum, Maximum) : Minimum;
        set
        {
            double step = Definition.Range?.Step ?? 0;
            double rounded = step > 0 ? Math.Round(Math.Round(value / step) * step, 6) : value;
            string text = rounded.ToString("R", CultureInfo.InvariantCulture);
            if (Read(Text.Length > 0 ? Text : Default) is not { } current || Math.Abs(current - rounded) > 1e-9)
            {
                Text = text;
            }
        }
    }

    /// <summary>Why the value was refused, or "".</summary>
    public string Error
    {
        get => _error;
        internal set
        {
            if (SetProperty(ref _error, value ?? ""))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>Back to the default.</summary>
    public ICommand ResetCommand { get; }

    private static double? Read(string text) =>
        double.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) ? value : null;
}

/// <summary>The tune sheet's settings of one group ("Sampling").</summary>
public sealed class ChatReplyGroup(string title, IEnumerable<ChatReplyEntry> entries) : ObservableCollection<ChatReplyEntry>(entries)
{
    /// <summary>Its heading.</summary>
    public string Title { get; } = title;
}
