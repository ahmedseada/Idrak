// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Chat lab: its view model streaming from a model behind the library's chat contract (IChatModel), the batching of
// its UI updates, Stop, the speeds and the settings as generation options.
internal static partial class MauiTests
{
    // A model store holding the given models.
    private sealed class ListedStore(params StoredModel[] models) : IModelStore
    {
        public string Folder => Path.GetTempPath();

        public bool HasToken => false;

        public Task<StoredModel> PullAsync(string model, string? revision = null, IProgressReporter? progress = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public IReadOnlyList<StoredModel> List() => models;

        public bool Remove(string id) => false;

        public StorageSpace? FreeSpace() => null;
    }

    // A chat model answering with `pieces` pieces of `piece`, each counting its token, then a final piece with the
    // statistics given; moves the test's clock by `step` a piece; Gate (when set) holds it after its first piece until
    // the caller cancels.
    private sealed class ScriptedChat(int pieces, string piece, GenerationStats stats, ManualTime? time = null, TimeSpan step = default) : IChatModel
    {
        public List<ChatRequest> Requests { get; } = [];

        public TaskCompletionSource FirstPiece { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool Gate { get; init; }

        public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Requests.Add(request);
            await Task.Yield();
            for (int i = 0; i < pieces; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                time?.Advance(step);
                yield return new ChatChunk(new ChatDelta(piece, "", [])) { Generated = i + 1 };
                FirstPiece.TrySetResult();
                if (Gate)
                {
                    await Task.Delay(Timeout.Infinite, cancellationToken);
                }
            }

            yield return new ChatChunk(new ChatDelta("", "", []), true, "stop", new ChatMessage("assistant", string.Concat(Enumerable.Repeat(piece, pieces))), stats)
            {
                Generated = pieces,
            };
        }
    }

    // A chat model service handing out one model, counting the leases and saying whether each loaded it.
    private sealed class OneChat(IChatModel chat, bool? thinking = null) : IChatModelService
    {
        public List<ChatLoadRequest> Requests { get; } = [];

        public bool Unloaded { get; set; } = true;

        public ChatMemoryEstimate Estimate(ChatLoadRequest request) => new(request.Model, "cpu", 1000, 100, 1 << 30, "the memory cpu reports free now");

        public Task<ChatLease> AcquireAsync(ChatLoadRequest request, CancellationToken cancellationToken = default)
        {
            Requests.Add(request);
            bool load = Unloaded;
            Unloaded = false;
            var info = new ChatModelInfo(request.Model, "cpu", request.KeyValueFormat ?? "float32", request.Context, TimeSpan.FromSeconds(1.5), thinking, Estimate(request));
            return Task.FromResult(new ChatLease(chat, info, load ? TimeSpan.FromSeconds(1.5) : TimeSpan.Zero));
        }
    }

    // A dispatcher that runs at once on the calling thread (the streaming thread stands in for the UI thread).
    private sealed class InlineDispatcher : IDispatcher
    {
        public int Dispatched { get; private set; }

        public bool IsDispatchRequired => true;

        public bool Dispatch(Action action)
        {
            Dispatched++;
            action();
            return true;
        }

        public bool DispatchDelayed(TimeSpan delay, Action action) => Dispatch(action);

        public IDispatcherTimer CreateTimer() => throw new NotSupportedException();
    }

    private static StoredModel Stored(string folder, string id = "lab/chat") =>
        new(id, "0123456789ab", folder, 1000, 3, Complete: true, DateTimeOffset.UtcNow);

    // A folder the chat can load (a config.json in it); the scripted models never read it.
    private static string ChatFolder(string name)
    {
        string folder = Temp(name);
        File.WriteAllText(Path.Combine(folder, "config.json"), "{}");
        return folder;
    }

    private static GenerationStats Stats(int tokens, double seconds, int promptTokens = 5) =>
        new(promptTokens, TimeSpan.FromSeconds(0.5), tokens, TimeSpan.FromSeconds(seconds), TimeSpan.FromSeconds(seconds + 0.5), 0);

    private static void ChatStreams()
    {
        string folder = ChatFolder("chat-stream");
        var stats = Stats(1000, 20);
        var model = new ScriptedChat(1000, "ab", stats);
        var chats = new OneChat(model, thinking: true);
        var dispatcher = new QueuedDispatcher();
        var options = new IdrakMauiOptions { ChatSystemPrompt = "Be brief.", ChatThinking = false };
        var page = new ChatViewModel(new ListedStore(Stored(folder), Stored(folder, "lab/unfinished") with { Complete = false }), chats, options, dispatcher, new ManualTime());
        page.RefreshAsync().GetAwaiter().GetResult();
        Check(page.Models.Count == 1 && page.Models[0].Title == "lab/chat" && page.Models[0].Path == folder, $"the finished models: {string.Join(", ", page.Models)}");
        page.SelectedModel = page.Models[0];
        Check(page.Loading!.GetAwaiter().GetResult() && page.ModelText.Contains("lab/chat on cpu", StringComparison.Ordinal)
              && page.ModelText.Contains("loaded in 1.5 s", StringComparison.Ordinal), page.ModelText);
        Check(page.HasThinkingSwitch && !page.Thinking, "the setting's thinking over the model's default");

        // A thousand pieces at one time make one showing and the final one; the bubble gets them all when the UI thread runs.
        page.Draft = "  hello  ";
        Check(page.CanSend && page.SendAsync().GetAwaiter().GetResult(), $"not answered: {page.Error}");
        Check(page.UiUpdates == 2 && dispatcher.Dispatched <= 3, $"{page.UiUpdates} showings and {dispatcher.Dispatched} dispatches for 1,000 pieces");
        dispatcher.Run();
        var bubble = page.Messages[^1];
        Check(page.Messages.Count == 2 && page.Messages[0].IsUser && page.Messages[0].Text == "hello" && bubble.IsAssistant, "the user's bubble, then the answer's");
        Check(bubble.Text == string.Concat(Enumerable.Repeat("ab", 1000)) && bubble.Updates == 2 && !bubble.IsStreaming, $"{bubble.Text.Length} characters in {bubble.Updates} updates");
        Check(bubble.Footer.StartsWith("1,000 tokens · 50.0 tokens/s · prompt 5 tokens in 0.50 s", StringComparison.Ordinal) && bubble.Footer.EndsWith("cold", StringComparison.Ordinal),
            bubble.Footer);
        Check(page.LiveText.StartsWith("1,000 tokens · 50.0 tokens/s", StringComparison.Ordinal) && page.HasCold && page.ColdText.Contains("50.0 tokens/s", StringComparison.Ordinal) && !page.HasWarm,
            $"'{page.LiveText}', '{page.ColdText}', '{page.WarmText}'");

        // The request: the system prompt first, the history, the thinking switch, the settings' options.
        var request = model.Requests.Single();
        Check(request.Messages.Select(m => $"{m.Role}:{m.Content}").SequenceEqual(["system:Be brief.", "user:hello"]) && request.Think == false
              && request.Options == ChatViewModel.Options(options), $"request: {string.Join(" | ", request.Messages.Select(m => $"{m.Role}:{m.Content}"))}, think {request.Think}");
        Check(page.History.Count == 2 && page.History[1].Content == bubble.Text, "the answer joins the history");

        // A second answer is warm, its request carries the history; a new chat clears it.
        page.Draft = "again";
        Check(page.SendAsync().GetAwaiter().GetResult(), page.Error);
        dispatcher.Run();
        Check(page.Messages[^1].Footer.EndsWith("warm", StringComparison.Ordinal) && page.WarmText.StartsWith("Warm: 50.0 tokens/s last, 50.0 mean over 1 answer", StringComparison.Ordinal)
              && model.Requests[1].Messages.Count == 4 && chats.Requests.Count == 3, $"'{page.WarmText}', {model.Requests[1].Messages.Count} messages, {chats.Requests.Count} leases");
        page.NewChat();
        Check(page.Messages.Count == 0 && page.History.Count == 0 && page.RequestMessages().Single().Role == "system", "a new chat keeps only the system prompt");

        // Copying goes through the page's clipboard.
        string? copied = null;
        page.Copy = text =>
        {
            copied = text;
            return Task.CompletedTask;
        };
        page.Draft = "copy me";
        page.SendAsync().GetAwaiter().GetResult();
        dispatcher.Run();
        page.Messages[^1].CopyCommand.Execute(null);
        Check(copied == page.Messages[^1].Text && page.Status == "Copied.", $"copied '{copied?.Length}' characters, status '{page.Status}'");
    }

    private static void ChatBatched()
    {
        // 1,000 pieces a millisecond apart (a second in all) with a 250 ms interval: a showing at the first piece, about one a
        // quarter second after, and the last; the bubble's text is every piece whatever the showings.
        string folder = ChatFolder("chat-batched");
        var time = new ManualTime();
        var model = new ScriptedChat(1000, "x", Stats(1000, 1), time, TimeSpan.FromMilliseconds(1));
        var dispatcher = new InlineDispatcher();
        var page = new ChatViewModel(new ListedStore(Stored(folder)), new OneChat(model), new IdrakMauiOptions { ProgressInterval = TimeSpan.FromMilliseconds(250) },
            dispatcher, time);
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models[0];
        page.Loading!.GetAwaiter().GetResult();
        int scrolled = 0;
        page.ScrollToEndRequested += (_, _) => scrolled++;
        page.Draft = "count";
        Check(page.SendAsync().GetAwaiter().GetResult(), page.Error);
        var bubble = page.Messages[^1];
        Check(bubble.Text.Length == 1000 && page.UiUpdates is >= 4 and <= 7 && bubble.Updates == page.UiUpdates, $"{page.UiUpdates} showings, {bubble.Updates} bubble updates for 1,000 pieces");
        Check(dispatcher.Dispatched <= page.UiUpdates + 2 && scrolled >= page.UiUpdates, $"{dispatcher.Dispatched} dispatches, {scrolled} scrolls");
    }

    private static void ChatStopped()
    {
        string folder = ChatFolder("chat-stop");
        var model = new ScriptedChat(10, "partial ", Stats(10, 1)) { Gate = true };
        var page = new ChatViewModel(new ListedStore(Stored(folder)), new OneChat(model), new IdrakMauiOptions(), new InlineDispatcher(), new ManualTime());
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models[0];
        page.Loading!.GetAwaiter().GetResult();
        page.Draft = "go on";
        var answering = page.SendAsync();
        Check(model.FirstPiece.Task.Wait(TimeSpan.FromSeconds(30)), "the first piece never came");
        Check(page.IsStreaming && page.StopCommand.CanExecute(null) && !page.SendCommand.CanExecute(null) && !page.NewChatCommand.CanExecute(null), "streaming: Stop only");
        page.Stop();
        Check(answering.Wait(TimeSpan.FromSeconds(30)) && answering.Result, $"stopping did not end the answer: {page.Error}");
        var bubble = page.Messages[^1];
        Check(!page.IsStreaming && bubble.Text == "partial " && bubble.Footer.EndsWith("stopped", StringComparison.Ordinal) && page.Status.StartsWith("Stopped", StringComparison.Ordinal),
            $"'{bubble.Text}', '{bubble.Footer}', '{page.Status}'");
        Check(page.History.Count == 2 && page.History[1].Content == "partial " && page.SendCommand.CanExecute(null) == false && page.NewChatCommand.CanExecute(null),
            "what came is kept as the answer");
    }

    private static void ChatPageOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var feature = new ChatFeature();
        var page = feature.CreatePage(services, feature.CreateViewModel(services));
        Check(page is ChatPage { BindingContext: ChatViewModel { Copy: not null, OpenModels: not null } } && page.Title == "Chat", $"page {page.GetType().Name}, title {page.Title}");
        Check(feature is { Name: MobileFeatures.Chat, Order: 20, Needs: PhoneNeeds.Model }, "the library's chat feature");
    }

    // A dictation that hears its words at once, then waits for Stop; or one with no speech model.
    private sealed class ScriptedDictation(params string[] heard) : IDictation
    {
        private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _stops;

        public bool NoModel { get; init; }

        public bool IsDictating { get; private set; }

        public int Stops => Volatile.Read(ref _stops);

        public async Task<DictationOutcome> DictateAsync(Action<string> onHeard, CancellationToken cancellationToken = default)
        {
            if (NoModel)
            {
                return new DictationOutcome("", false, "No speech model is ready to dictate with: pull one in Models.", NeedsModel: true);
            }

            IsDictating = true;
            foreach (string words in heard)
            {
                onHeard(words);
            }

            await _stopped.Task.ConfigureAwait(false);
            IsDictating = false;
            return new DictationOutcome(heard.Length > 0 ? heard[^1] : "", true, "");
        }

        public void Stop()
        {
            Interlocked.Increment(ref _stops);
            _stopped.TrySetResult();
        }
    }

    private static void ChatDictates()
    {
        string folder = ChatFolder("chat-dictate");
        var chats = new OneChat(new ScriptedChat(1, "a", Stats(1, 1)));
        var dictation = new ScriptedDictation("hello", "hello there");
        var page = new ChatViewModel(new ListedStore(Stored(folder)), chats, new IdrakMauiOptions(), new InlineDispatcher(), new ManualTime(), dictation: dictation);
        page.Draft = "Typed first: ";
        var run = page.ToggleDictationAsync();
        Check(page.IsDictating && !page.IsNotDictating && page.MicrophoneDescription == "Stop dictating" && page.Draft == "Typed first: hello there"
              && page.DictationNote.StartsWith("Listening", StringComparison.Ordinal) && !run.IsCompleted, $"dictating: '{page.Draft}', '{page.DictationNote}'");
        page.DictateCommand.Execute(null);                                     // the microphone again: it stops
        Check(run.Wait(TimeSpan.FromSeconds(20)) && run.Result && dictation.Stops == 1 && !page.IsDictating && page.MicrophoneDescription == "Dictate",
            $"stopped: {dictation.Stops} stop(s), '{page.DictationNote}'");
        Check(page.Draft == "Typed first: hello there" && !page.HasDictationNote && !page.DictationNeedsModel, $"the draft: '{page.Draft}'; '{page.DictationNote}'");

        // No speech model: said, with Models offered; nothing typed.
        var none = new ChatViewModel(new ListedStore(Stored(folder)), chats, new IdrakMauiOptions(), new InlineDispatcher(), new ManualTime(), dictation: new ScriptedDictation { NoModel = true });
        Check(!none.ToggleDictationAsync().GetAwaiter().GetResult() && none.DictationNeedsModel && none.DictationNote.Contains("Models", StringComparison.Ordinal) && none.Draft.Length == 0
              && !none.IsDictating, $"no model: '{none.DictationNote}'");

        // No dictation registered: said.
        var without = new ChatViewModel(new ListedStore(Stored(folder)), chats, new IdrakMauiOptions(), new InlineDispatcher(), new ManualTime());
        Check(!without.ToggleDictationAsync().GetAwaiter().GetResult() && without.HasDictationNote && !without.IsDictating, without.DictationNote);
    }
}
