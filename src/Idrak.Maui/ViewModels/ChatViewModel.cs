// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
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
    private bool _streaming;

    internal ChatBubble(string role, string text, Func<string, Task> copy)
    {
        Role = role;
        _text = text;
        CopyCommand = new Command(async () => await copy(Text));
    }

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
        internal set => SetProperty(ref _streaming, value);
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
            Text += text;
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

    /// <summary>A chat over <paramref name="store"/>'s models, loaded by <paramref name="chats"/>, with <paramref name="options"/>' settings.</summary>
    /// <param name="store">The model store.</param>
    /// <param name="chats">Loads the chat models.</param>
    /// <param name="options">The app's options (read at each load and answer: a setting saved meanwhile applies to the next).</param>
    /// <param name="dispatcher">The UI thread's dispatcher; null: the app's (<c>Application.Current.Dispatcher</c>) when there is one, else the streaming thread itself.</param>
    /// <param name="time">The clock the batching reads (tests give their own).</param>
    public ChatViewModel(IModelStore store, IChatModelService chats, IdrakMauiOptions options, IDispatcher? dispatcher = null, TimeProvider? time = null)
    {
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
        private set => SetProperty(ref _loading, value);
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
        var bubble = new ChatBubble("assistant", "", CopyAsync) { IsStreaming = true };
        Messages.Add(bubble);
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

        long start = _time.GetTimestamp(), first = 0;
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
        return new Answer(message, final?.Stats, final?.Generated ?? _generated, cold, stopped, _time.GetElapsedTime(start), lease.Loaded ? lease.Info : null);
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
            LiveText = generated == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"{generated:N0} token{(generated == 1 ? "" : "s")}")
                + (seconds > 0 && generated > firstGenerated ? string.Create(CultureInfo.InvariantCulture, $" · {(generated - firstGenerated) / seconds:F1} tokens/s") : "");
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
        Post(() =>
        {
            bubble.IsStreaming = false;
            bubble.Footer = footer;
            LiveText = answer.Stopped ? "Stopped: " + speed : speed;
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

    private sealed record Answer(ChatMessage Message, GenerationStats? Stats, int Generated, bool Cold, bool Stopped, TimeSpan Elapsed, ChatModelInfo? Info);
}
