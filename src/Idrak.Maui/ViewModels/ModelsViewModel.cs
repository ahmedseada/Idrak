// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Net;
using System.Windows.Input;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>A model on the Models page: the stored model and what the page shows of it.</summary>
public sealed class ModelEntry
{
    internal ModelEntry(StoredModel model, ICommand remove)
    {
        Model = model;
        RemoveCommand = remove;
    }

    /// <summary>The stored model.</summary>
    public StoredModel Model { get; }

    /// <summary>Its id.</summary>
    public string Title => Model.Id;

    /// <summary>Its size, files and revision ("1.2 GB in 9 files, revision c1899de289a0").</summary>
    public string Details => $"{ProgressText.Bytes(Model.Bytes)} in {Model.Files} file{(Model.Files == 1 ? "" : "s")}, revision {Model.Revision}";

    /// <summary>Where it is.</summary>
    public string Folder => Model.Folder;

    /// <summary>When a file of it last changed, in the phone's own format.</summary>
    public string Modified => Model.Modified == DateTimeOffset.MinValue ? "" : "changed " + Model.Modified.ToLocalTime().ToString("g", CultureInfo.CurrentCulture);

    /// <summary>Whether every file is in.</summary>
    public bool IsComplete => Model.Complete;

    /// <summary>Whether a pull of it is unfinished.</summary>
    public bool IsUnfinished => !Model.Complete;

    /// <summary>Its state as text.</summary>
    public string StateText => Model.Complete ? "complete" : "unfinished: pull it again to resume";

    /// <summary>Removes it (after a confirmation).</summary>
    public ICommand RemoveCommand { get; }
}

/// <summary>A suggested model to tap: it fills the id field.</summary>
/// <param name="Model">The id (or an id and a .gguf file in it).</param>
/// <param name="Command">Fills the id field with it.</param>
public sealed record SuggestionChip(string Model, ICommand Command);

/// <summary>
/// The Models page over the <see cref="IModelStore"/>: pull a Hugging Face model by its id and revision into the app's
/// storage, with its progress (bytes, rate, time left) on an item of the app's <see cref="ProgressCenter"/> (updated on the
/// UI thread at most once a progress interval), cancellable and resumed by pulling again; the models stored with their
/// sizes and folders, read with the storage's free space on a worker thread (one walk over the folders); removing one
/// after a confirmation (<see cref="Confirm"/>); every failure shown with its cause (<see cref="Explain"/>).
/// </summary>
public sealed class ModelsViewModel : ViewModel
{
    private readonly IModelStore _store;
    private readonly ProgressCenter? _progress;
    private readonly Command _pull;
    private readonly Command _cancel;
    private readonly Command _refresh;
    private CancellationTokenSource? _cancellation;
    private string _modelId = "";
    private string _revision = "";
    private string _pulling = "";
    private bool _busy;
    private bool _isPulling;
    private ProgressItem? _item;
    private string _status = "";
    private string _error = "";
    private string _storage = "";
    private string _summary = "";
    private string _token = "";

    /// <summary>A page over <paramref name="store"/>, suggesting <paramref name="options"/>' models, showing progress on <paramref name="progress"/>.</summary>
    public ModelsViewModel(IModelStore store, IdrakMauiOptions options, ProgressCenter? progress = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        ArgumentNullException.ThrowIfNull(options);
        _progress = progress;
        _pull = new Command(async () => await PullAsync(), () => !IsPulling);
        _cancel = new Command(Cancel, () => IsPulling);
        _refresh = new Command(async () => await RefreshAsync(), () => !IsBusy);
        foreach (string model in options.SuggestedModels.Where(m => !string.IsNullOrWhiteSpace(m)).Select(m => m.Trim()).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            Suggestions.Add(new SuggestionChip(model, new Command(() => ModelId = model)));
        }
    }

    /// <summary>The page's title.</summary>
    public string Title => "Models";

    /// <summary>Where pulled models are kept.</summary>
    public string Folder => _store.Folder;

    /// <summary>The model to pull: "owner/name", or "owner/name/file.gguf" for one file.</summary>
    public string ModelId
    {
        get => _modelId;
        set => SetProperty(ref _modelId, value ?? "");
    }

    /// <summary>The revision to pull (a branch, a tag or a commit); empty: "main".</summary>
    public string Revision
    {
        get => _revision;
        set => SetProperty(ref _revision, value ?? "");
    }

    /// <summary>The app's suggested models (<see cref="IdrakMauiOptions.SuggestedModels"/>).</summary>
    public ObservableCollection<SuggestionChip> Suggestions { get; } = [];

    /// <summary>Whether there are suggestions to show.</summary>
    public bool HasSuggestions => Suggestions.Count > 0;

    /// <summary>The models stored, by id.</summary>
    public ObservableCollection<ModelEntry> Models { get; } = [];

    /// <summary>The pull in progress, or the last one (its amounts, share, rate and time left), or null.</summary>
    public ProgressItem? Pull
    {
        get => _item;
        private set
        {
            if (SetProperty(ref _item, value))
            {
                OnPropertyChanged(nameof(HasPull));
            }
        }
    }

    /// <summary>Whether <see cref="Pull"/> has something to show.</summary>
    public bool HasPull => Pull is not null;

    /// <summary>Whether a pull is running.</summary>
    public bool IsPulling
    {
        get => _isPulling;
        private set
        {
            if (SetProperty(ref _isPulling, value))
            {
                _pull.ChangeCanExecute();
                _cancel.ChangeCanExecute();
            }
        }
    }

    /// <summary>Whether the list is being read.</summary>
    public bool IsBusy
    {
        get => _busy;
        set
        {
            if (SetProperty(ref _busy, value))
            {
                _refresh.ChangeCanExecute();
            }
        }
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

    /// <summary>The storage's free space ("12.3 GB free of 64.0 GB"), or why it is not known.</summary>
    public string StorageText
    {
        get => _storage;
        private set => SetProperty(ref _storage, value);
    }

    /// <summary>How many models are stored and their size.</summary>
    public string ModelsText
    {
        get => _summary;
        private set => SetProperty(ref _summary, value);
    }

    /// <summary>Whether a Hugging Face token is set, and what it is for.</summary>
    public string TokenText
    {
        get => _token;
        private set => SetProperty(ref _token, value);
    }

    /// <summary>
    /// Asks the user to confirm (a title and a message; true to go on). The page sets it (an alert); without it nothing is
    /// removed.
    /// </summary>
    public Func<string, string, Task<bool>>? Confirm { get; set; }

    /// <summary>Pulls <see cref="ModelId"/> at <see cref="Revision"/>.</summary>
    public ICommand PullCommand => _pull;

    /// <summary>Cancels the pull running.</summary>
    public ICommand CancelCommand => _cancel;

    /// <summary>Reads the models and the free space again.</summary>
    public ICommand RefreshCommand => _refresh;

    /// <summary>
    /// Pulls <see cref="ModelId"/> at <see cref="Revision"/> on a worker thread, its progress on <see cref="Pull"/>, then
    /// reads the list again; true when it is in. A failure or a cancellation is shown (<see cref="Error"/>,
    /// <see cref="Status"/>) and leaves what was downloaded, listed as unfinished.
    /// </summary>
    public async Task<bool> PullAsync()
    {
        string model = ModelId.Trim();
        if (IsPulling)
        {
            return false;
        }

        if (model.Length == 0)
        {
            Error = "Type a Hugging Face id (owner/name), or an id and a .gguf file in it (owner/name/file.gguf).";
            return false;
        }

        string revision = Revision.Trim();
        string name = revision.Length == 0 ? model : $"{model}@{revision}";
        using var cancellation = new CancellationTokenSource();
        _cancellation = cancellation;
        _pulling = model;
        Error = "";
        Status = $"Pulling {name}…";
        IsPulling = true;
        var reporter = _progress?.Start($"pull {name}", null, ProgressUnit.Bytes);
        Pull = reporter is null ? null : _progress!.ItemOf(reporter);
        try
        {
            var stored = await Task.Run(() => _store.PullAsync(model, revision.Length == 0 ? null : revision, reporter, cancellation.Token), CancellationToken.None)
                .ConfigureAwait(true);
            Status = $"{name} is in: {ProgressText.Bytes(stored.Bytes)} in {stored.Files} file{(stored.Files == 1 ? "" : "s")}.";
            return true;
        }
        catch (Exception e)
        {
            string text = Explain(e, name, _store.HasToken);
            if (e is OperationCanceledException && cancellation.IsCancellationRequested)
            {
                Status = text;
            }
            else
            {
                Status = "";
                Error = text;
            }

            return false;
        }
        finally
        {
            _cancellation = null;
            _pulling = "";
            IsPulling = false;
            await RefreshAsync().ConfigureAwait(true);
        }
    }

    /// <summary>Cancels the pull running (what it downloaded is kept; pulling again resumes it).</summary>
    public void Cancel()
    {
        try
        {
            _cancellation?.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // The pull ended meanwhile.
        }
    }

    /// <summary>Reads the stored models, the free space and whether a token is set, on a worker thread, and shows them.</summary>
    public async Task RefreshAsync()
    {
        if (IsBusy)
        {
            return;
        }

        IsBusy = true;
        try
        {
            var (models, space, token) = await Task.Run(() => (_store.List(), _store.FreeSpace(), _store.HasToken)).ConfigureAwait(true);
            Models.Clear();
            foreach (var model in models)
            {
                Models.Add(new ModelEntry(model, new Command(async () => await RemoveAsync(model))));
            }

            long bytes = models.Sum(m => m.Bytes);
            ModelsText = models.Count == 0 ? "No models pulled yet."
                : $"{models.Count} model{(models.Count == 1 ? "" : "s")}, {ProgressText.Bytes(bytes)}"
                  + (models.Count(m => !m.Complete) is var unfinished and > 0 ? $" ({unfinished} unfinished)" : "");
            StorageText = space is null ? "The free space of this storage could not be measured."
                : $"{ProgressText.Bytes(space.FreeBytes)} free of {ProgressText.Bytes(space.TotalBytes)}";
            TokenText = token ? "A Hugging Face token is set: gated and private models can be pulled."
                : "No Hugging Face token: gated and private models need one (Settings, Hugging Face token).";
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The models could not be read from {Folder}: {e.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>Removes <paramref name="model"/> once <see cref="Confirm"/> agrees; true when it was removed.</summary>
    public async Task<bool> RemoveAsync(StoredModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        string name = $"{model.Id}@{model.Revision}";
        if (IsPulling && string.Equals(StoreId(_pulling), model.Id, StringComparison.OrdinalIgnoreCase))
        {
            Error = $"{model.Id} is being pulled: cancel the pull first.";
            return false;
        }

        if (Confirm is not { } confirm)
        {
            Error = "Removing a model needs a confirmation, and this page has none to ask.";
            return false;
        }

        if (!await confirm("Remove the model", $"Remove {model.Id} (revision {model.Revision}, {ProgressText.Bytes(model.Bytes)}) from this phone? "
                                               + "Its files are deleted; pulling it again downloads them again.").ConfigureAwait(true))
        {
            return false;
        }

        Error = "";
        try
        {
            bool removed = await Task.Run(() => _store.Remove(name)).ConfigureAwait(true);
            Status = removed ? $"Removed {name}: {ProgressText.Bytes(model.Bytes)} freed." : $"{name} was not there any more.";
            await RefreshAsync().ConfigureAwait(true);
            return removed;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException)
        {
            Error = $"{name} was not removed: {e.Message}";
            return false;
        }
    }

    /// <summary>
    /// What a failed pull of <paramref name="model"/> tells the user, with its cause: cancelled (kept, resumed by pulling
    /// again), not enough space, a token needed (401) or refused, a gated model (403), not found (404), no network, a full
    /// storage while writing, or the error's own message.
    /// </summary>
    /// <param name="error">What the pull threw.</param>
    /// <param name="model">The model asked for, as the user typed it.</param>
    /// <param name="tokenSet">Whether a token was set (<see cref="IModelStore.HasToken"/>).</param>
    public static string Explain(Exception error, string model, bool tokenSet)
    {
        ArgumentNullException.ThrowIfNull(error);
        string first = error.Message.Split('\n')[0].Trim();
        return error switch
        {
            OperationCanceledException => $"Cancelled: what {model} downloaded so far is kept; pull it again to resume.",
            NotEnoughSpaceException => first,
            HttpRequestException { StatusCode: HttpStatusCode.Unauthorized } => tokenSet
                ? $"{model}: the Hugging Face token was refused (401). Check it in Settings (Hugging Face token), then pull again."
                : $"{model} needs a Hugging Face token (401): it is gated or private, or the id is not right. Set the token in Settings (Hugging Face token), or check the id.",
            HttpRequestException { StatusCode: HttpStatusCode.Forbidden } => tokenSet
                ? $"No access to {model} (403): it is gated. Accept its terms on its Hugging Face page with the token's account, then pull again."
                : $"No access to {model} (403): it is gated. Set a Hugging Face token in Settings, of an account that accepted its terms, then pull again.",
            HttpRequestException { StatusCode: HttpStatusCode.NotFound } =>
                $"{model} was not found on Hugging Face (404): check the id and the revision{(tokenSet ? "" : ", or set a token if it is private")}.",
            HttpRequestException { StatusCode: null } =>
                $"No network: Hugging Face could not be reached ({first}). What was downloaded is kept; pull again once connected.",
            HttpRequestException { StatusCode: { } status } => $"{model}: Hugging Face answered {(int)status} {status}. Pull again later ({first}).",
            IOException when IsStorageFull(error) => $"The storage is full: {model} could not be written ({first}). Remove a model to free space, then pull again to resume.",
            FileNotFoundException or ArgumentException or InvalidDataException => first,
            _ => $"{model} was not pulled: {first}",
        };
    }

    // ERROR_DISK_FULL and ERROR_HANDLE_DISK_FULL on Windows (in the HRESULT's low word); ENOSPC (28) elsewhere, the errno itself.
    private static bool IsStorageFull(Exception error) => OperatingSystem.IsWindows() ? (error.HResult & 0xFFFF) is 0x70 or 0x27 : error.HResult == 28;

    // "owner/name" of what was typed ("owner/name/file.gguf" is kept under owner/name).
    private static string StoreId(string model) => model.Split('/') is { Length: >= 2 } parts ? $"{parts[0]}/{parts[1]}" : model;
}
