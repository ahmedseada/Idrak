// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Markdown;
using Microsoft.Maui.Dispatching;

namespace Idrak.Maui.ViewModels;

/// <summary>A model of the store the chat can load: a finished pull with a config.json, or a .gguf file.</summary>
public sealed class ChatModelChoice
{
    internal ChatModelChoice(StoredModel model, string path, string title)
    {
        Model = model;
        Path = path;
        Title = title;
    }

    /// <summary>The stored model.</summary>
    public StoredModel Model { get; }

    /// <summary>What the library loads: the revision folder, or the .gguf file in it.</summary>
    public string Path { get; }

    /// <summary>Its name in the picker ("Qwen/Qwen3-0.6B", with the revision or the file when one id has several).</summary>
    public string Title { get; }

    /// <inheritdoc />
    public override string ToString() => Title;

    /// <summary>
    /// The choices <paramref name="models"/> give: the finished ones (<see cref="StoredModel.Complete"/>) with a folder the
    /// library loads, by id; an id with several revisions or files is named with its revision and file.
    /// </summary>
    public static IReadOnlyList<ChatModelChoice> From(IEnumerable<StoredModel> models)
    {
        ArgumentNullException.ThrowIfNull(models);
        var found = new List<(StoredModel Model, string Path)>();
        foreach (var model in models.Where(m => m.Complete && Directory.Exists(m.Folder)))
        {
            if (File.Exists(System.IO.Path.Combine(model.Folder, "config.json")))
            {
                found.Add((model, model.Folder));
                continue;
            }

            foreach (string gguf in Directory.EnumerateFiles(model.Folder, "*.gguf", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            {
                found.Add((model, gguf));
            }
        }

        return [.. found.Select(f => new ChatModelChoice(f.Model, f.Path, found.Count(o => o.Model.Id == f.Model.Id) == 1 ? f.Model.Id
            : $"{f.Model.Id}@{f.Model.Revision}" + (f.Path == f.Model.Folder ? "" : "/" + System.IO.Path.GetFileName(f.Path))))];
    }
}

/// <summary>One message of the chat as its bubble shows it: changed on the UI thread only.</summary>
public sealed class ChatBubble : ViewModel
{
    private string _text = "";
    private string _thinking = "";
    private string _footer = "";
    private string _rate = "", _tokens = "", _first = "", _live = "";
    private double _thinkingSeconds;
    private bool _streaming, _expanded, _canRegenerate;

    internal ChatBubble(string role, string text, Func<string, Task> copy, string model = "", ICommand? regenerate = null, ICommand? stop = null)
    {
        StopCommand = stop ?? new Command(() => { });
        Role = role;
        _text = text;
        ModelName = model;
        CopyCommand = new Command(async () => await copy(Text));
        RegenerateCommand = regenerate ?? new Command(() => { });
        ToggleThinkingCommand = new Command(() => IsThinkingExpanded = !IsThinkingExpanded);
        Document = new MarkdownDocument(copy);
        if (Role != "user" && text.Length > 0)
        {
            Document.Update(text);
        }
    }

    /// <summary>The answer as Markdown blocks (paragraphs, lists, code), kept up to date as it streams.</summary>
    public MarkdownDocument Document { get; }

    /// <summary>The model that answered ("Qwen3-0.6B"), or "".</summary>
    public string ModelName { get; }

    /// <summary>The letter on the model's avatar.</summary>
    public string ModelInitial => ModelName.Length > 0 ? char.ToUpperInvariant(ModelName[0]).ToString() : "AI";

    /// <summary>Whether the message reads right to left (its first strong letter is Arabic, Hebrew…).</summary>
    public bool IsRightToLeft => MarkdownBlock.IsRightToLeft(Text);

    /// <summary>How long the model reasoned before it answered, in seconds (0 when it did not).</summary>
    public double ThinkingSeconds
    {
        get => _thinkingSeconds;
        internal set
        {
            if (SetProperty(ref _thinkingSeconds, value))
            {
                OnPropertyChanged(nameof(ThinkingTitle));
            }
        }
    }

    /// <summary>"Thought for 1.4 s", or "Thinking…" while it reasons.</summary>
    public string ThinkingTitle => IsStreaming && Text.Length == 0 ? "Thinking…"
        : string.Create(CultureInfo.InvariantCulture, $"Thought for {ThinkingSeconds:0.0} s");

    /// <summary>Whether the reasoning is shown.</summary>
    public bool IsThinkingExpanded
    {
        get => _expanded;
        set => SetProperty(ref _expanded, value);
    }

    /// <summary>Shows or hides the reasoning.</summary>
    public ICommand ToggleThinkingCommand { get; }

    /// <summary>"18.1 tok/s", or "".</summary>
    public string RateText
    {
        get => _rate;
        internal set
        {
            if (SetProperty(ref _rate, value))
            {
                OnPropertyChanged(nameof(HasMeta));
            }
        }
    }

    /// <summary>"212 tokens", or "".</summary>
    public string TokensText
    {
        get => _tokens;
        internal set => SetProperty(ref _tokens, value);
    }

    /// <summary>"first in 0.4 s", or "".</summary>
    public string FirstTokenText
    {
        get => _first;
        internal set => SetProperty(ref _first, value);
    }

    /// <summary>Whether the finished answer has its measures to show.</summary>
    public bool HasMeta => !IsStreaming && IsAssistant && (RateText.Length > 0 || TokensText.Length > 0);

    /// <summary>While it streams: "17.4 tok/s · 31 tokens".</summary>
    public string LiveText
    {
        get => _live;
        internal set => SetProperty(ref _live, value);
    }

    /// <summary>Whether this answer can be asked again (the last one, when nothing streams).</summary>
    public bool CanRegenerate
    {
        get => _canRegenerate;
        internal set => SetProperty(ref _canRegenerate, value);
    }

    /// <summary>Asks the last question again, in place of this answer.</summary>
    public ICommand RegenerateCommand { get; }

    /// <summary>Stops the answer streaming (the chat's Stop, beside the live rate).</summary>
    public ICommand StopCommand { get; }

    /// <summary>"user" or "assistant".</summary>
    public string Role { get; }

    /// <summary>Whether the user wrote it (the bubble on the right).</summary>
    public bool IsUser => Role == "user";

    /// <summary>Whether the model wrote it (the bubble on the left).</summary>
    public bool IsAssistant => !IsUser;

    /// <summary>The answer (or the user's message) so far.</summary>
    public string Text
    {
        get => _text;
        private set => SetProperty(ref _text, value);
    }

    /// <summary>The model's reasoning so far, or "".</summary>
    public string Thinking
    {
        get => _thinking;
        private set
        {
            if (SetProperty(ref _thinking, value))
            {
                OnPropertyChanged(nameof(HasThinking));
            }
        }
    }

    /// <summary>Whether there is reasoning to show.</summary>
    public bool HasThinking => Thinking.Length > 0;

    /// <summary>Whether the answer is still coming.</summary>
    public bool IsStreaming
    {
        get => _streaming;
        internal set
        {
            if (SetProperty(ref _streaming, value))
            {
                OnPropertyChanged(nameof(HasMeta));
                OnPropertyChanged(nameof(ThinkingTitle));
            }
        }
    }

    /// <summary>What it took ("85 tokens · 14.2 tokens/s · warm"), or "".</summary>
    public string Footer
    {
        get => _footer;
        internal set
        {
            if (SetProperty(ref _footer, value))
            {
                OnPropertyChanged(nameof(HasFooter));
            }
        }
    }

    /// <summary>Whether <see cref="Footer"/> has something to show.</summary>
    public bool HasFooter => Footer.Length > 0;

    /// <summary>How many times its text changed (each a UI update).</summary>
    public int Updates { get; private set; }

    /// <summary>Copies <see cref="Text"/> to the clipboard.</summary>
    public ICommand CopyCommand { get; }

    internal void Append(string text, string thinking)
    {
        if (text.Length > 0)
        {
            bool first = Text.Length == 0;
            Text += text;
            Document.Update(Text);
            if (first)
            {
                OnPropertyChanged(nameof(IsRightToLeft));
                OnPropertyChanged(nameof(ThinkingTitle));
            }
        }

        if (thinking.Length > 0)
        {
            Thinking += thinking;
        }

        Updates++;
    }
}

/// <summary>
/// The Chat lab: a model from the store (<see cref="IModelStore"/>, its finished pulls) loaded once through the
/// <see cref="IChatModelService"/> on the device and in the formats of the settings, refused with the limit named when it
/// does not fit; each answer streamed through the library's chat contract (<see cref="IChatModel"/>, the model's own chat
/// template) with its token count and tokens per second, the text added to its bubble at most once a progress interval
/// (<see cref="IdrakMauiOptions.ProgressInterval"/>) whatever the number of tokens; Stop cancels mid-answer; a new chat
/// clears the history. The system prompt, temperature, top-p, top-k, max new tokens, context, seed and thinking are the
/// settings' (<see cref="Options(IdrakMauiOptions)"/>), the library's own where none is set. The first answer after a load
/// (cold: compiling the methods and kernels) is reported apart from the warm ones (rule 76).
/// </summary>
public sealed class ChatViewModel : ViewModel
{
    private readonly IModelStore _store;
    private readonly IChatModelService _chats;
    private readonly IdrakMauiOptions _options;
    private readonly IDispatcher? _dispatcher;
    private readonly TimeProvider _time;
    private readonly Command _send;
    private readonly Command _stop;
    private readonly Command _newChat;
    private readonly List<ChatMessage> _history = [];
    private readonly List<double> _warmRates = [];
    private readonly Lock _pending = new();
    private readonly StringBuilder _pendingText = new();
    private readonly StringBuilder _pendingThinking = new();
    private CancellationTokenSource? _cancellation;
    private ChatModelChoice? _selected;
    private ChatModelInfo? _info;
    private int _answersSinceLoad;
    private bool _flushQueued;
    private long _lastFlush;
    private int _generated;
    private string _draft = "";
    private bool _loading;
    private bool _streaming;
    private bool _thinking;
    private string _modelText = "";
    private string _coldText = "";
    private string _warmText = "";
    private string _liveText = "";
    private string _status = "";
    private string _error = "";
    private readonly IConversationStore? _conversations;
    private readonly IDictation? _dictation;
    private bool _dictating, _dictationNeedsModel;
    private string _dictationNote = "";
    private readonly List<double> _rates = [];
    private string _conversationId;
    private string _conversationQuery = "";
    private string _lastRateText = "";
    private string _firstTokenText = "";
    private double _liveRate;
    private int _liveTokens;

    /// <summary>A chat over <paramref name="store"/>'s models, loaded by <paramref name="chats"/>, with <paramref name="options"/>' settings.</summary>
    /// <param name="store">The model store.</param>
    /// <param name="chats">Loads the chat models.</param>
    /// <param name="options">The app's options (read at each load and answer: a setting saved meanwhile applies to the next).</param>
    /// <param name="dispatcher">The UI thread's dispatcher; null: the app's (<c>Application.Current.Dispatcher</c>) when there is one, else the streaming thread itself.</param>
    /// <param name="time">The clock the batching reads (tests give their own).</param>
    /// <param name="conversations">Where the chats are kept between runs; null: none kept.</param>
    /// <param name="dictation">What the composer's microphone dictates with (the Transcribe lab's, <c>UseIdrak</c>'s); null: the microphone says there is none.</param>
    public ChatViewModel(IModelStore store, IChatModelService chats, IdrakMauiOptions options, IDispatcher? dispatcher = null, TimeProvider? time = null,
        IConversationStore? conversations = null, IDictation? dictation = null)
    {
        _conversations = conversations;
        _dictation = dictation;
        DictateCommand = new Command(async () => await ToggleDictationAsync());
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _chats = chats ?? throw new ArgumentNullException(nameof(chats));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _dispatcher = dispatcher;
        _time = time ?? TimeProvider.System;
        _send = new Command(async () => await SendAsync(), () => CanSend);
        _stop = new Command(Stop, () => IsStreaming);
        _newChat = new Command(NewChat, () => !IsStreaming);
        RefreshCommand = new Command(async () => await RefreshAsync());
        GoToModelsCommand = new Command(async () => await (OpenModels?.Invoke() ?? Task.CompletedTask));
        _conversationId = ConversationText.NewId(_time.GetUtcNow());
        RegenerateCommand = new Command(async () => await RegenerateAsync());
        SuggestCommand = new Command<string>(async text =>
        {
            Draft = text ?? "";
            await SendAsync();
        });
        OpenConversationCommand = new Command<string>(id => OpenConversation(id ?? ""));
        DeleteConversationCommand = new Command<string>(id => DeleteConversation(id ?? ""));
        foreach (string suggestion in options.ChatSuggestions)
        {
            Suggestions.Add(suggestion);
        }

        Messages.CollectionChanged += (_, _) => OnPropertyChanged(nameof(ShowSuggestions));
        ListConversations();
    }

    /// <summary>The model chosen, by its name without the owner ("Qwen3-0.6B"), or what to do.</summary>
    public string ModelName => SelectedModel is { } model ? ShortName(model.Title) : HasModels ? "Choose a model" : "No model yet";

    /// <summary>The letter on the model's avatar.</summary>
    public string ModelInitial => SelectedModel is { } model && ShortName(model.Title) is { Length: > 0 } name ? char.ToUpperInvariant(name[0]).ToString() : "?";

    /// <summary>"int8 · vulkan:0 · loaded in 3.1 s", or what is happening.</summary>
    public string ModelMeta => Model is { } m ? string.Create(CultureInfo.InvariantCulture, $"{m.Weights} · {m.Device} · loaded in {m.LoadTime.TotalSeconds:F1} s")
        : IsLoading ? "Loading…" : SelectedModel is null ? (HasModels ? "Tap to choose one of the store's models" : "Pull one in Models") : "Not loaded";

    /// <summary>Whether a model is loaded (the green dot).</summary>
    public bool IsModelLoaded => Model is not null;

    /// <summary>Questions to ask while the chat is empty.</summary>
    public ObservableCollection<string> Suggestions { get; } = [];

    /// <summary>Whether the suggestions show: nothing streams and nothing has been asked.</summary>
    public bool ShowSuggestions => IsIdle && Messages.Count == 0 && Suggestions.Count > 0;

    /// <summary>Asks a suggestion.</summary>
    public ICommand SuggestCommand { get; }

    /// <summary>Asks the last question again.</summary>
    public ICommand RegenerateCommand { get; }

    /// <summary>The live tokens per second of the answer streaming (the last answer's when none streams).</summary>
    public double LiveRate
    {
        get => _liveRate;
        private set
        {
            if (SetProperty(ref _liveRate, value))
            {
                OnPropertyChanged(nameof(LiveRateText));
            }
        }
    }

    /// <summary>"17.4".</summary>
    public string LiveRateText => LiveRate > 0 ? LiveRate.ToString("F1", CultureInfo.InvariantCulture) : "–";

    /// <summary>The tokens of the answer streaming (the last answer's when none streams).</summary>
    public int LiveTokens
    {
        get => _liveTokens;
        private set => SetProperty(ref _liveTokens, value);
    }

    /// <summary>The recent rates of the answer streaming, one a showing (the inspector's sparkline).</summary>
    public IReadOnlyList<double> RateHistory { get; private set; } = [];

    /// <summary>"0.38 s · warm", or "".</summary>
    public string FirstTokenText
    {
        get => _firstTokenText;
        private set => SetProperty(ref _firstTokenText, value);
    }

    /// <summary>The last answer's speed and device ("17.4 tok/s on vulkan:0"), or "" before any.</summary>
    public string LastRateText
    {
        get => _lastRateText;
        private set => SetProperty(ref _lastRateText, value);
    }

    /// <summary>The device the model is on, or "–".</summary>
    public string DeviceText => Model?.Device ?? "–";

    /// <summary>"0.91 GB of 5.6 GB": the model's weights and cache against the memory measured before loading, or "".</summary>
    public string MemoryText => Model?.Estimate is { } e
        ? $"{ProgressText.Bytes(e.NeededBytes)}" + (e.AvailableBytes is { } a ? $" of {ProgressText.Bytes(a)}" : "") : "";

    /// <summary>The weights' share of the memory measured (0 when unknown).</summary>
    public double WeightsShare => Model?.Estimate is { AvailableBytes: > 0 } e ? (double)e.WeightBytes / e.AvailableBytes.Value : 0;

    /// <summary>The cache's share of the memory measured (0 when unknown).</summary>
    public double CacheShare => Model?.Estimate is { AvailableBytes: > 0 } e ? (double)e.CacheBytes / e.AvailableBytes.Value : 0;

    /// <summary>"weights 734 MB".</summary>
    public string WeightsText => Model?.Estimate is { } e ? $"weights {ProgressText.Bytes(e.WeightBytes)}" : "";

    /// <summary>"cache 212 MB".</summary>
    public string CacheText => Model?.Estimate is { } e ? $"cache {ProgressText.Bytes(e.CacheBytes)}" : "";

    /// <summary>The sampling temperature of the next answers (the settings' value, changed here for this run).</summary>
    public double Temperature
    {
        get => _options.ChatTemperature ?? new GenerationOptions().Temperature;
        set
        {
            double rounded = Math.Round(Math.Clamp(value, 0, 2), 2);
            if (_options.ChatTemperature != rounded)
            {
                _options.ChatTemperature = rounded;
                OnPropertyChanged();
                OnPropertyChanged(nameof(TemperatureText));
            }
        }
    }

    /// <summary>"0.7".</summary>
    public string TemperatureText => Temperature.ToString("0.0#", CultureInfo.InvariantCulture);

    /// <summary>The most tokens of the next answers, as text ("" for until the turn ends).</summary>
    public string MaxNewTokensText
    {
        get => _options.ChatMaxNewTokens?.ToString(CultureInfo.InvariantCulture) ?? "";
        set
        {
            int? next = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0 ? n : null;
            if (_options.ChatMaxNewTokens != next)
            {
                _options.ChatMaxNewTokens = next;
                OnPropertyChanged();
            }
        }
    }

    /// <summary>The kept chats, grouped by day (Today, Yesterday, This week, Earlier), filtered by <see cref="ConversationQuery"/>.</summary>
    public ObservableCollection<ConversationGroup> Conversations { get; } = [];

    /// <summary>The search over the kept chats.</summary>
    public string ConversationQuery
    {
        get => _conversationQuery;
        set
        {
            if (SetProperty(ref _conversationQuery, (value ?? "").Trim()))
            {
                ListConversations();
            }
        }
    }

    /// <summary>The id of the chat shown.</summary>
    public string ConversationId => _conversationId;

    /// <summary>Opens a kept chat (by id).</summary>
    public ICommand OpenConversationCommand { get; }

    /// <summary>Forgets a kept chat (by id).</summary>
    public ICommand DeleteConversationCommand { get; }

    /// <summary>Lists the kept chats again (matching <see cref="ConversationQuery"/>), the one shown marked.</summary>
    public void ListConversations()
    {
        Conversations.Clear();
        if (_conversations is null)
        {
            return;
        }

        IReadOnlyList<ConversationSummary> kept;
        try
        {
            kept = _conversations.List();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The kept chats could not be read: {e.Message}";
            return;
        }

        var now = _time.GetUtcNow();
        foreach (var group in kept.Where(c => _conversationQuery.Length == 0 || c.Title.Contains(_conversationQuery, StringComparison.CurrentCultureIgnoreCase)
                     || c.Preview.Contains(_conversationQuery, StringComparison.CurrentCultureIgnoreCase))
                     .GroupBy(c => ConversationText.Group(c.Updated, now)))
        {
            Conversations.Add(new ConversationGroup(group.Key, group.Select(c => new ConversationEntry(c, ConversationText.When(c.Updated, now), c.Id == _conversationId,
                OpenConversationCommand))));
        }
    }

    /// <summary>Shows the kept chat <paramref name="id"/> (its messages; the model chosen stays); false when it is not kept or an answer streams.</summary>
    public bool OpenConversation(string id)
    {
        if (IsStreaming || _conversations?.Load(id) is not { } chat)
        {
            return false;
        }

        _history.Clear();
        Messages.Clear();
        foreach (var message in chat.Messages.Where(m => m.Role is "user" or "assistant"))
        {
            _history.Add(message);
            var bubble = new ChatBubble(message.Role, message.Content, CopyAsync, ShortName(chat.Model ?? ""), RegenerateCommand);
            if (message.Thinking is { Length: > 0 } thinking)
            {
                bubble.Append("", thinking);
            }

            Messages.Add(bubble);
        }

        _conversationId = chat.Id;
        MarkLast();
        LiveText = "";
        Error = "";
        Status = "";
        ListConversations();
        return true;
    }

    /// <summary>Forgets the kept chat <paramref name="id"/> (the chat shown starts afresh when it is that one).</summary>
    public bool DeleteConversation(string id)
    {
        if (_conversations is null || !_conversations.Delete(id))
        {
            return false;
        }

        if (id == _conversationId)
        {
            NewChat();
        }

        ListConversations();
        return true;
    }

    /// <summary>
    /// Asks the last question again: the last answer and its question leave the chat, and the question is sent anew. False
    /// while an answer streams or when there is no answer to replace.
    /// </summary>
    public async Task<bool> RegenerateAsync()
    {
        if (IsStreaming || Messages.Count < 2 || Messages[^1].IsUser || _history.Count < 2 || _history[^2].Role != "user")
        {
            return false;
        }

        string question = _history[^2].Content;
        _history.RemoveRange(_history.Count - 2, 2);
        Messages.RemoveAt(Messages.Count - 1);
        Messages.RemoveAt(Messages.Count - 1);
        Draft = question;
        return await SendAsync().ConfigureAwait(true);
    }

    private static string ShortName(string id) => id.Length == 0 ? "" : id[(id.TrimEnd('/').LastIndexOf('/') + 1)..];

    // Only the last answer can be asked again.
    private void MarkLast()
    {
        for (int i = 0; i < Messages.Count; i++)
        {
            Messages[i].CanRegenerate = i == Messages.Count - 1 && Messages[i].IsAssistant && !IsStreaming;
        }
    }

    // Keeps the chat shown (after each answer), titled by its first question.
    private void Keep()
    {
        if (_conversations is null || _history.Count == 0)
        {
            return;
        }

        try
        {
            string first = _history.FirstOrDefault(m => m.Role == "user")?.Content ?? "Chat";
            _conversations.Save(new ChatConversation(_conversationId, ConversationText.Title(first), _time.GetUtcNow(), SelectedModel?.Model.Id, [.. _history]));
            ListConversations();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The chat was not kept: {e.Message}";
        }
    }

    /// <summary>The page's title.</summary>
    public string Title => "Chat";

    /// <summary>The models the chat can load (the store's finished ones).</summary>
    public ObservableCollection<ChatModelChoice> Models { get; } = [];

    /// <summary>Whether there is a model to choose.</summary>
    public bool HasModels => Models.Count > 0;

    /// <summary>Whether the store has no model the chat can load.</summary>
    public bool HasNoModels => !HasModels;

    /// <summary>The model chosen; choosing one loads it (<see cref="Loading"/>).</summary>
    public ChatModelChoice? SelectedModel
    {
        get => _selected;
        set
        {
            if (SetProperty(ref _selected, value))
            {
                _send.ChangeCanExecute();
                OnPropertyChanged(nameof(CanSend));
                OnPropertyChanged(nameof(ModelName));
                OnPropertyChanged(nameof(ModelInitial));
                OnPropertyChanged(nameof(ModelMeta));
                if (value is not null)
                {
                    Loading = LoadAsync(value);
                }
            }
        }
    }

    /// <summary>The load the last choice started, or null.</summary>
    public Task<bool>? Loading { get; private set; }

    /// <summary>The model loaded, as the service reported it, or null.</summary>
    public ChatModelInfo? Model
    {
        get => _info;
        private set
        {
            if (SetProperty(ref _info, value))
            {
                OnPropertyChanged(nameof(HasThinkingSwitch));
                OnPropertyChanged(nameof(ModelMeta));
                OnPropertyChanged(nameof(IsModelLoaded));
                OnPropertyChanged(nameof(DeviceText));
                OnPropertyChanged(nameof(MemoryText));
                OnPropertyChanged(nameof(WeightsShare));
                OnPropertyChanged(nameof(CacheShare));
                OnPropertyChanged(nameof(WeightsText));
                OnPropertyChanged(nameof(CacheText));
            }
        }
    }

    /// <summary>Whether the model's template has a reasoning switch (the page shows <see cref="Thinking"/>).</summary>
    public bool HasThinkingSwitch => Model?.HasThinking == true;

    /// <summary>Whether the next answers reason (where the template has a switch): the setting's, else the model's own default.</summary>
    public bool Thinking
    {
        get => _thinking;
        set => SetProperty(ref _thinking, value);
    }

    /// <summary>The message being written.</summary>
    public string Draft
    {
        get => _draft;
        set
        {
            if (SetProperty(ref _draft, value ?? ""))
            {
                _send.ChangeCanExecute();
                OnPropertyChanged(nameof(CanSend));
            }
        }
    }

    /// <summary>The conversation's bubbles, oldest first.</summary>
    public ObservableCollection<ChatBubble> Messages { get; } = [];

    /// <summary>Whether a model is loading.</summary>
    public bool IsLoading
    {
        get => _loading;
        private set
        {
            if (SetProperty(ref _loading, value))
            {
                OnPropertyChanged(nameof(ModelMeta));
            }
        }
    }

    /// <summary>Whether an answer is streaming.</summary>
    public bool IsStreaming
    {
        get => _streaming;
        private set
        {
            if (SetProperty(ref _streaming, value))
            {
                OnPropertyChanged(nameof(IsIdle));
                OnPropertyChanged(nameof(ShowSuggestions));
                OnPropertyChanged(nameof(CanSend));
                _send.ChangeCanExecute();
                _stop.ChangeCanExecute();
                _newChat.ChangeCanExecute();
            }
        }
    }

    /// <summary>Whether no answer is streaming (the page shows Send, else Stop).</summary>
    public bool IsIdle => !IsStreaming;

    /// <summary>Whether a message can be sent now.</summary>
    public bool CanSend => !IsStreaming && SelectedModel is not null && Draft.Trim().Length > 0;

    /// <summary>The model loaded: its device, formats, context and load time ("Qwen/Qwen3-0.6B on cpu · weights int8 · …"), or what to do.</summary>
    public string ModelText
    {
        get => _modelText;
        private set => SetProperty(ref _modelText, value);
    }

    /// <summary>The first answer after the load (cold), or "".</summary>
    public string ColdText
    {
        get => _coldText;
        private set
        {
            if (SetProperty(ref _coldText, value))
            {
                OnPropertyChanged(nameof(HasCold));
            }
        }
    }

    /// <summary>Whether <see cref="ColdText"/> has something to show.</summary>
    public bool HasCold => ColdText.Length > 0;

    /// <summary>The warm answers since the load (the last and the mean), or "".</summary>
    public string WarmText
    {
        get => _warmText;
        private set
        {
            if (SetProperty(ref _warmText, value))
            {
                OnPropertyChanged(nameof(HasWarm));
            }
        }
    }

    /// <summary>Whether <see cref="WarmText"/> has something to show.</summary>
    public bool HasWarm => WarmText.Length > 0;

    /// <summary>The answer streaming: its tokens so far and their rate ("38 tokens · 12.4 tokens/s"), or "".</summary>
    public string LiveText
    {
        get => _liveText;
        private set => SetProperty(ref _liveText, value);
    }

    /// <summary>What the last action did, or "".</summary>
    public string Status
    {
        get => _status;
        private set
        {
            if (SetProperty(ref _status, value))
            {
                OnPropertyChanged(nameof(HasStatus));
            }
        }
    }

    /// <summary>Whether <see cref="Status"/> has something to show.</summary>
    public bool HasStatus => Status.Length > 0;

    /// <summary>Why the last action failed, with its cause, or "".</summary>
    public string Error
    {
        get => _error;
        private set
        {
            if (SetProperty(ref _error, value))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>How many times the streaming answers were handed to the UI thread (each a UI update), for the batching's measure.</summary>
    public int UiUpdates { get; private set; }

    /// <summary>Opens the Models page (the page sets it); without it the button does nothing.</summary>
    public Func<Task>? OpenModels { get; set; }

    /// <summary>Puts text on the clipboard (the page sets it: MAUI's Clipboard); without it copying says so.</summary>
    public Func<string, Task>? Copy { get; set; }

    /// <summary>Raised on the UI thread after the streaming answer grew, so the page scrolls to its end.</summary>
    public event EventHandler? ScrollToEndRequested;

    /// <summary>Sends <see cref="Draft"/>.</summary>
    public ICommand SendCommand => _send;

    /// <summary>Stops the answer streaming.</summary>
    public ICommand StopCommand => _stop;

    /// <summary>Clears the history.</summary>
    public ICommand NewChatCommand => _newChat;

    /// <summary>Reads the store's models again.</summary>
    public ICommand RefreshCommand { get; }

    /// <summary>Opens the Models page (<see cref="OpenModels"/>).</summary>
    public ICommand GoToModelsCommand { get; }

    /// <summary>
    /// The composer's microphone: dictates into <see cref="Draft"/> (the words heard live after what was typed, through
    /// the <see cref="IDictation"/> given: the Transcribe lab's model and settings), and stops it on the next tap.
    /// </summary>
    public ICommand DictateCommand { get; }

    /// <summary>Whether the microphone dictates into the draft now.</summary>
    public bool IsDictating
    {
        get => _dictating;
        private set
        {
            if (SetProperty(ref _dictating, value))
            {
                OnPropertyChanged(nameof(IsNotDictating));
                OnPropertyChanged(nameof(MicrophoneDescription));
            }
        }
    }

    /// <summary>Whether the microphone is idle.</summary>
    public bool IsNotDictating => !IsDictating;

    /// <summary>What the microphone does on a tap ("Dictate" or "Stop dictating").</summary>
    public string MicrophoneDescription => IsDictating ? "Stop dictating" : "Dictate";

    /// <summary>What the last dictation said (why it did not start, that it listens), or "".</summary>
    public string DictationNote
    {
        get => _dictationNote;
        private set
        {
            if (SetProperty(ref _dictationNote, value))
            {
                OnPropertyChanged(nameof(HasDictationNote));
            }
        }
    }

    /// <summary>Whether <see cref="DictationNote"/> has something to show.</summary>
    public bool HasDictationNote => DictationNote.Length > 0;

    /// <summary>Whether the dictation needs a speech model first (the note offers Models).</summary>
    public bool DictationNeedsModel
    {
        get => _dictationNeedsModel;
        private set => SetProperty(ref _dictationNeedsModel, value);
    }

    /// <summary>
    /// Starts dictating into the draft (the words heard after what is typed, live), or stops the dictation running; true
    /// when a dictation ran and ended with its last words read.
    /// </summary>
    public async Task<bool> ToggleDictationAsync()
    {
        if (_dictation is null)
        {
            DictationNote = "No dictation here: the app registers none.";
            return false;
        }

        if (IsDictating)
        {
            DictationNote = "Finishing the last words…";
            _dictation.Stop();
            return false;
        }

        string typed = Draft.TrimEnd();
        DictationNeedsModel = false;
        DictationNote = "Listening… tap the microphone again to stop.";
        IsDictating = true;
        try
        {
            var outcome = await _dictation.DictateAsync(heard => Post(() => Draft = Joined(typed, heard))).ConfigureAwait(true);
            if (outcome.Text.Length > 0)
            {
                Draft = Joined(typed, outcome.Text);
            }

            DictationNeedsModel = outcome.NeedsModel;
            DictationNote = outcome.Problem;
            return outcome.Completed;
        }
        finally
        {
            IsDictating = false;
        }
    }

    // What was typed, then what was heard.
    private static string Joined(string typed, string heard) => typed.Length == 0 ? heard : heard.Length == 0 ? typed : typed + " " + heard;

    /// <summary>
    /// The generation options the settings give: <see cref="IdrakMauiOptions.ChatTemperature"/>, <see cref="IdrakMauiOptions.ChatTopP"/>,
    /// <see cref="IdrakMauiOptions.ChatTopK"/>, <see cref="IdrakMauiOptions.ChatMaxNewTokens"/>, <see cref="IdrakMauiOptions.ChatContext"/>
    /// and <see cref="IdrakMauiOptions.ChatSeed"/> over the library's <see cref="GenerationOptions"/> (each one not set keeps the library's).
    /// </summary>
    public static GenerationOptions Options(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var library = new GenerationOptions();
        return library with
        {
            Temperature = options.ChatTemperature is { } temperature ? (float)temperature : library.Temperature,
            TopP = options.ChatTopP is { } topP ? (float)topP : library.TopP,
            TopK = options.ChatTopK ?? library.TopK,
            NumPredict = options.ChatMaxNewTokens ?? library.NumPredict,
            NumCtx = options.ChatContext ?? library.NumCtx,
            Seed = options.ChatSeed ?? library.Seed,
        };
    }

    /// <summary>What <paramref name="model"/> is loaded as with <paramref name="options"/>: its path, the device, the formats and the context.</summary>
    public static ChatLoadRequest LoadRequest(ChatModelChoice model, IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        return new ChatLoadRequest(model.Model.Id, model.Path, options.Device, options.ChatWeights, options.ChatKeyValueFormat, Options(options).NumCtx);
    }

    /// <summary>The messages a request sends: the system prompt of the settings (when set), then the history.</summary>
    public IReadOnlyList<ChatMessage> RequestMessages() =>
        _options.ChatSystemPrompt is { Length: > 0 } system ? [new ChatMessage("system", system), .. _history] : [.. _history];

    /// <summary>The conversation so far as the model sees it (user and assistant messages, without the system prompt).</summary>
    public IReadOnlyList<ChatMessage> History => _history;

    /// <summary>Reads the store's models on a worker thread and lists the ones the chat can load (keeping the one chosen).</summary>
    public async Task RefreshAsync()
    {
        try
        {
            var models = await Task.Run(() => ChatModelChoice.From(_store.List())).ConfigureAwait(true);
            string? chosen = SelectedModel?.Path;
            Models.Clear();
            foreach (var model in models)
            {
                Models.Add(model);
            }

            OnPropertyChanged(nameof(HasModels));
            OnPropertyChanged(nameof(HasNoModels));
            OnPropertyChanged(nameof(ModelName));
            OnPropertyChanged(nameof(ModelMeta));
            if (chosen is not null && Models.FirstOrDefault(m => m.Path == chosen) is { } same)
            {
                _selected = same;                                               // the same model: not loaded again
                OnPropertyChanged(nameof(SelectedModel));
            }

            if (Model is null && ModelText.Length == 0 || !HasModels)
            {
                ModelText = HasModels ? "Choose a model to load it." : "No finished model in the store: pull one in Models.";
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The models could not be read from {_store.Folder}: {e.Message}";
        }
    }

    /// <summary>
    /// Loads <paramref name="model"/> (once: one loaded already is shared) on a worker thread and shows its device, formats,
    /// context and load time; a model that does not fit, or whose family is not registered, is refused and named
    /// (<see cref="Error"/>). True when it is loaded.
    /// </summary>
    public async Task<bool> LoadAsync(ChatModelChoice model)
    {
        ArgumentNullException.ThrowIfNull(model);
        var request = LoadRequest(model, _options);
        IsLoading = true;
        Error = "";
        Model = null;
        ModelText = $"Loading {model.Title} on {request.Device}…";
        try
        {
            using var lease = await _chats.AcquireAsync(request).ConfigureAwait(true);
            Show(lease.Info, lease.Loaded);
            return true;
        }
        catch (Exception e)
        {
            ModelText = $"{model.Title} is not loaded.";
            Error = Explain(e, model.Title);
            return false;
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>
    /// Sends <see cref="Draft"/> and streams the answer into a new bubble (on a worker thread, the bubble updated at most once
    /// a progress interval); true when it ended (or was stopped) with an answer. The model is acquired for the answer only,
    /// so between answers memory pressure may take it (the next answer loads it again and is reported cold).
    /// </summary>
    public async Task<bool> SendAsync()
    {
        string text = Draft.Trim();
        if (IsStreaming || text.Length == 0)
        {
            return false;
        }

        if (SelectedModel is not { } model)
        {
            Error = "Choose a model first, or pull one in Models.";
            return false;
        }

        Draft = "";
        Error = "";
        Status = "";
        Messages.Add(new ChatBubble("user", text, CopyAsync));
        var bubble = new ChatBubble("assistant", "", CopyAsync, ShortName(model.Title), RegenerateCommand, StopCommand) { IsStreaming = true };
        Messages.Add(bubble);
        MarkLast();
        lock (_pending)
        {
            _rates.Clear();
        }

        RateHistory = [];
        OnPropertyChanged(nameof(RateHistory));
        _history.Add(new ChatMessage("user", text));
        var request = new ChatRequest(RequestMessages(), Think: HasThinkingSwitch ? Thinking : _options.ChatThinking, Options: Options(_options));
        var load = LoadRequest(model, _options);
        using var cancellation = new CancellationTokenSource();
        _cancellation = cancellation;
        IsStreaming = true;
        LiveText = "";
        ScrollToEndRequested?.Invoke(this, EventArgs.Empty);
        try
        {
            var answer = await Task.Run(() => StreamAsync(load, request, bubble, cancellation.Token), CancellationToken.None).ConfigureAwait(true);
            if (answer.Info is { } loaded)
            {
                Show(loaded, loaded: true);                                     // memory pressure had taken it: loaded again for this answer,
                _answersSinceLoad = 1;                                          // which is the cold one
            }

            _history.Add(answer.Message);
            Report(answer, bubble);
            Keep();
            return true;
        }
        catch (Exception e)
        {
            Error = Explain(e, model.Title);
            _history.RemoveAt(_history.Count - 1);                              // no answer: the history does not keep the question unanswered

            Post(() =>
            {
                bubble.IsStreaming = false;
                bubble.Footer = "no answer";
            });
            return false;
        }
        finally
        {
            _cancellation = null;
            IsStreaming = false;
            MarkLast();
        }
    }

    /// <summary>Stops the answer streaming: what came so far is kept as the answer.</summary>
    public void Stop()
    {
        try
        {
            _cancellation?.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // The answer ended meanwhile.
        }
    }

    /// <summary>Clears the history and the bubbles (the model stays loaded).</summary>
    public void NewChat()
    {
        if (IsStreaming)
        {
            return;
        }

        _history.Clear();
        Messages.Clear();
        LiveText = "";
        Error = "";
        Status = "A new chat.";
        _conversationId = ConversationText.NewId(_time.GetUtcNow());
        ListConversations();
    }

    /// <summary>
    /// What a failed load or answer of <paramref name="model"/> tells the user, with its cause: a model that does not fit
    /// (the limit named), a family that is not registered (the registry and what it holds), files that are gone, or the
    /// error's own message.
    /// </summary>
    public static string Explain(Exception error, string model)
    {
        ArgumentNullException.ThrowIfNull(error);
        string first = error.Message.Split('\n')[0].Trim();
        return error switch
        {
            ModelDoesNotFitException => first,
            NotRegisteredException missing => $"{model}: '{missing.Name}' is not registered in {missing.Registry} (registered: "
                + $"{(missing.Registered.Count == 0 ? "none" : string.Join(", ", missing.Registered))}). The app registers the families it shows, each plug-in by its own Register call. {first}",
            OperationCanceledException => "Stopped.",
            FileNotFoundException or DirectoryNotFoundException => $"{model}'s files are not there any more ({first}); pull it again in Models.",
            _ => $"{model}: {first}",
        };
    }

    private async Task CopyAsync(string text)
    {
        if (Copy is not { } copy)
        {
            Error = "Copying needs the page's clipboard, and this page has none.";
            return;
        }

        await copy(text).ConfigureAwait(true);
        Status = "Copied.";
    }

    private void Show(ChatModelInfo info, bool loaded)
    {
        Model = info;
        if (loaded)
        {
            _answersSinceLoad = 0;
            _warmRates.Clear();
            ColdText = "";
            WarmText = "";
        }

        if (info.ThinkingDefault is { } thinks)
        {
            Thinking = _options.ChatThinking ?? thinks;
        }

        ModelText = string.Create(CultureInfo.InvariantCulture,
            $"{info.Model} on {info.Device} · weights {info.Weights} · cache {info.KeyValueFormat} · {info.Context:N0} tokens · loaded in {info.LoadTime.TotalSeconds:F1} s")
            + (info.Estimate is { } e ? $"\nMeasured before loading: {e}" : "");
    }

    // On a worker thread: the answer, its pieces gathered and handed to the UI thread at most once a progress interval.
    private async Task<Answer> StreamAsync(ChatLoadRequest load, ChatRequest request, ChatBubble bubble, CancellationToken cancellationToken)
    {
        using var lease = await _chats.AcquireAsync(load, cancellationToken).ConfigureAwait(false);
        bool cold = lease.Loaded || _answersSinceLoad == 0;
        lock (_pending)
        {
            _pendingText.Clear();
            _pendingThinking.Clear();
            _flushQueued = false;
            _lastFlush = 0;
            _generated = 0;
        }

        long start = _time.GetTimestamp(), first = 0, thinkStart = 0, thinkEnd = 0;
        int firstGenerated = 0;
        ChatChunk? final = null;
        var content = new StringBuilder();
        bool stopped = false;
        try
        {
            await foreach (var chunk in lease.Chat.StreamAsync(request, cancellationToken).ConfigureAwait(false))
            {
                content.Append(chunk.Delta.Content);
                lock (_pending)
                {
                    _pendingText.Append(chunk.Delta.Content);
                    _pendingThinking.Append(chunk.Delta.Thinking);
                    _generated = Math.Max(_generated, chunk.Generated);
                }

                if (first == 0)
                {
                    (first, firstGenerated) = (_time.GetTimestamp(), chunk.Generated);
                }

                if (thinkStart == 0 && chunk.Delta.Thinking is { Length: > 0 })
                {
                    thinkStart = _time.GetTimestamp();
                }
                else if (thinkStart != 0 && thinkEnd == 0 && chunk.Delta.Content.Length > 0)
                {
                    thinkEnd = _time.GetTimestamp();
                }

                if (chunk.Done)
                {
                    final = chunk;
                }
                else
                {
                    Flush(bubble, first, firstGenerated, final: false);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            stopped = true;
        }

        stopped |= cancellationToken.IsCancellationRequested;
        Flush(bubble, first, firstGenerated, final: true);
        var message = final?.Message is { } reply ? reply : new ChatMessage("assistant", content.ToString());
        _answersSinceLoad++;
        double thinking = thinkStart == 0 ? 0 : _time.GetElapsedTime(thinkStart, thinkEnd == 0 ? _time.GetTimestamp() : thinkEnd).TotalSeconds;
        var firstToken = first == 0 ? TimeSpan.Zero : _time.GetElapsedTime(start, first);
        return new Answer(message, final?.Stats, final?.Generated ?? _generated, cold, stopped, _time.GetElapsedTime(start), lease.Loaded ? lease.Info : null, thinking, firstToken);
    }

    // Hands what came since the last showing to the UI thread: at once for the first piece and the last, else once the
    // progress interval has passed and the previous showing was taken (a later one carries all that came meanwhile).
    private void Flush(ChatBubble bubble, long first, int firstGenerated, bool final)
    {
        long now = _time.GetTimestamp();
        lock (_pending)
        {
            if (!final && (_flushQueued || _lastFlush != 0 && _time.GetElapsedTime(_lastFlush, now) < _options.ProgressInterval))
            {
                return;
            }

            _flushQueued = true;
            _lastFlush = now;
        }

        UiUpdates++;
        Post(() =>
        {
            string text, thinking;
            int generated;
            lock (_pending)
            {
                (text, thinking, generated) = (_pendingText.ToString(), _pendingThinking.ToString(), _generated);
                _pendingText.Clear();
                _pendingThinking.Clear();
                _flushQueued = false;
            }

            bubble.Append(text, thinking);
            double seconds = first == 0 ? 0 : _time.GetElapsedTime(first, now).TotalSeconds;
            double rate = seconds > 0 && generated > firstGenerated ? (generated - firstGenerated) / seconds : 0;
            LiveText = generated == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"{generated:N0} token{(generated == 1 ? "" : "s")}")
                + (rate > 0 ? string.Create(CultureInfo.InvariantCulture, $" · {rate:F1} tokens/s") : "");
            bubble.LiveText = string.Create(CultureInfo.InvariantCulture, $"{rate:F1} tok/s · {generated:N0} token{(generated == 1 ? "" : "s")}");
            LiveTokens = generated;
            if (rate > 0)
            {
                LiveRate = rate;
                _rates.Add(rate);
                if (_rates.Count > 48)
                {
                    _rates.RemoveAt(0);
                }

                RateHistory = [.. _rates];
                OnPropertyChanged(nameof(RateHistory));
            }

            ScrollToEndRequested?.Invoke(this, EventArgs.Empty);
        });
    }

    // After the answer: its bubble's footer, the cold or warm line, the live line's last word.
    private void Report(Answer answer, ChatBubble bubble)
    {
        var stats = answer.Stats;
        int tokens = stats?.GeneratedTokens ?? answer.Generated;
        double rate = stats?.TokensPerSecond ?? 0;
        string speed = string.Create(CultureInfo.InvariantCulture, $"{tokens:N0} token{(tokens == 1 ? "" : "s")}")
            + (rate > 0 ? string.Create(CultureInfo.InvariantCulture, $" · {rate:F1} tokens/s") : "")
            + (stats is { } s ? string.Create(CultureInfo.InvariantCulture, $" · prompt {s.PromptTokens:N0} tokens in {s.PromptDuration.TotalSeconds:F2} s") : "");
        string footer = speed + (answer.Cold ? " · cold" : " · warm") + (answer.Stopped ? " · stopped" : "");
        string firstToken = answer.FirstToken > TimeSpan.Zero ? string.Create(CultureInfo.InvariantCulture, $"{answer.FirstToken.TotalSeconds:0.0#} s") : "";
        Post(() =>
        {
            bubble.IsStreaming = false;
            bubble.Footer = footer;
            bubble.RateText = rate > 0 ? string.Create(CultureInfo.InvariantCulture, $"{rate:F1} tok/s") : "";
            bubble.TokensText = string.Create(CultureInfo.InvariantCulture, $"{tokens:N0} token{(tokens == 1 ? "" : "s")}");
            bubble.FirstTokenText = firstToken.Length > 0 ? "first in " + firstToken : "";
            bubble.ThinkingSeconds = answer.ThinkingSeconds;
            LiveText = answer.Stopped ? "Stopped: " + speed : speed;
            LiveTokens = tokens;
            if (rate > 0)
            {
                LiveRate = rate;
                LastRateText = string.Create(CultureInfo.InvariantCulture, $"{rate:F1} tok/s on {Model?.Device ?? "the device"}");
            }

            FirstTokenText = firstToken.Length > 0 ? firstToken + (answer.Cold ? " · cold" : " · warm") : "";
        });
        if (answer.Stopped)
        {
            Status = "Stopped: what came so far is kept as the answer.";
        }

        if (answer.Cold)
        {
            ColdText = "First answer after loading (cold: compiling methods and kernels): " + speed;
        }
        else if (rate > 0)
        {
            _warmRates.Add(rate);
            WarmText = string.Create(CultureInfo.InvariantCulture,
                $"Warm: {rate:F1} tokens/s last, {_warmRates.Average():F1} mean over {_warmRates.Count} answer{(_warmRates.Count == 1 ? "" : "s")}");
        }
    }

    private void Post(Action action)
    {
        var dispatcher = _dispatcher ?? Application.Current?.Dispatcher;
        if (dispatcher is null || !dispatcher.IsDispatchRequired)
        {
            action();
        }
        else
        {
            dispatcher.Dispatch(action);
        }
    }

    private sealed record Answer(ChatMessage Message, GenerationStats? Stats, int Generated, bool Cold, bool Stopped, TimeSpan Elapsed, ChatModelInfo? Info,
        double ThinkingSeconds, TimeSpan FirstToken);
}

/// <summary>A kept chat as the conversations list shows it.</summary>
public sealed class ConversationEntry(ConversationSummary summary, string when, bool current, ICommand open)
{
    /// <summary>Its id.</summary>
    public string Id => summary.Id;

    /// <summary>Its title.</summary>
    public string Title => summary.Title;

    /// <summary>The start of its last message.</summary>
    public string Preview => summary.Preview;

    /// <summary>When it last changed ("9:38", "Sun", "3 Oct").</summary>
    public string When { get; } = when;

    /// <summary>Whether it is the chat shown.</summary>
    public bool IsCurrent { get; } = current;

    /// <summary>Whether its preview reads right to left.</summary>
    public bool IsRightToLeft => MarkdownBlock.IsRightToLeft(Preview);

    /// <summary>Opens it (its id is the parameter).</summary>
    public ICommand OpenCommand { get; } = open;
}

/// <summary>The kept chats of one day group ("Today").</summary>
public sealed class ConversationGroup(string title, IEnumerable<ConversationEntry> chats) : ObservableCollection<ConversationEntry>(chats)
{
    /// <summary>Its heading.</summary>
    public string Title { get; } = title;
}
