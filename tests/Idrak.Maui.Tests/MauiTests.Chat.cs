// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Models;
using Idrak.Text.Plugins.Qwen;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The Chat lab: its view model streaming from a model behind the library's chat contract (IChatModel), the batching of
// its UI updates, Stop, the speeds, the settings as generation options, and the chat model service over a tiny generated
// Qwen3 decoder (the test registers the family, as the app does).
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
            var info = new ChatModelInfo(request.Model, "cpu", request.Weights ?? "float32", request.KeyValueFormat ?? "float32", request.Context,
                TimeSpan.FromSeconds(1.5), thinking, Estimate(request));
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
              && page.ModelText.Contains("loaded in 1.5 s", StringComparison.Ordinal) && page.ModelText.Contains("weights float32", StringComparison.Ordinal), page.ModelText);
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

    private static void ChatSettings()
    {
        var library = new GenerationOptions();
        Check(ChatViewModel.Options(new IdrakMauiOptions()) == library, "no setting: the library's options");
        var options = new IdrakMauiOptions { ChatTemperature = 0, ChatTopP = 0.5, ChatTopK = 7, ChatMaxNewTokens = 33, ChatContext = 512, ChatSeed = 9 };
        var mapped = ChatViewModel.Options(options);
        Check(mapped == library with { Temperature = 0, TopP = 0.5f, TopK = 7, NumPredict = 33, NumCtx = 512, Seed = 9 }, $"mapped {mapped}");

        IdrakSetting Find(string key) => IdrakMauiSettings.Definitions.Single(s => s.Key == key);
        foreach (var (key, text) in new[] { ("chat_temperature", "-1"), ("chat_top_p", "1.5"), ("chat_top_k", "-2"), ("chat_max_new_tokens", "0"), ("chat_context", "8"),
                     ("chat_thinking", "maybe"), ("chat_weights", "int9"), ("chat_kv", "int3") })
        {
            bool refused = false;
            try
            {
                Find(key).Set(options, text);
            }
            catch (FormatException e)
            {
                refused = e.Message.Contains(text, StringComparison.Ordinal);
            }

            Check(refused, $"{key}={text} was taken, or refused without naming the value");
        }

        Find("chat_weights").Set(options, "INT8");
        Find("chat_kv").Set(options, "bf16");
        Find("chat_thinking").Set(options, "on");
        Find("chat_system_prompt").Set(options, "  Answer in French.  ");
        Check(options.ChatWeights == "int8" && options.ChatKeyValueFormat == "bfloat16" && options.ChatThinking == true && options.ChatSystemPrompt == "Answer in French.",
            $"{options.ChatWeights}, {options.ChatKeyValueFormat}, {options.ChatThinking}, '{options.ChatSystemPrompt}'");
        Find("chat_weights").Set(options, "default");
        Find("chat_temperature").Set(options, "");
        Check(options.ChatWeights is null && options.ChatTemperature is null, "default and empty: the library's");
        Check(Find("chat_temperature").Placeholder == library.Temperature.ToString("R", System.Globalization.CultureInfo.InvariantCulture)
              && Find("chat_context").Placeholder == "2048", $"placeholders {Find("chat_temperature").Placeholder}, {Find("chat_context").Placeholder}");

        var choice = ChatModelChoice.From([Stored(ChatFolder("chat-request"))]).Single();
        var request = ChatViewModel.LoadRequest(choice, new IdrakMauiOptions { Device = "cpu", ChatWeights = "int4", ChatKeyValueFormat = "int8", ChatContext = 1024 });
        Check(request is { Device: "cpu", Weights: "int4", KeyValueFormat: "int8", Context: 1024, Model: "lab/chat" } && request.Path == choice.Path, $"{request}");
        Check(ChatViewModel.LoadRequest(choice, new IdrakMauiOptions()).Context == library.NumCtx, "the library's context");
    }

    // A tiny Qwen3 decoder made from a spec (random weights from a seed), with a byte-level tokenizer holding ChatML's
    // tokens and the ChatML template: what the library makes with PretrainedModel.Create.
    private static string TinyChatModel(string name)
    {
        string folder = Temp(name), tokenizerFolder = Temp(name + "-tokenizer");
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        var vocab = new JsonObject();
        for (int b = 0, n = 0; b < 256; b++)
        {
            vocab[((char)(printable.Contains(b) ? b : 256 + n++)).ToString()] = b;
        }

        File.WriteAllText(Path.Combine(tokenizerFolder, "tokenizer.json"), new JsonObject
        {
            ["added_tokens"] = new JsonArray(
                new JsonObject { ["id"] = 256, ["content"] = "<|im_start|>", ["special"] = true },
                new JsonObject { ["id"] = 257, ["content"] = "<|im_end|>", ["special"] = true },
                new JsonObject { ["id"] = 258, ["content"] = "<|endoftext|>", ["special"] = true }),
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray() },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = true },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        }.ToJsonString());
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 64, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        var files = new ModelFiles { Tokenizer = PretrainedTokenizers.Load(tokenizerFolder), EosToken = "<|im_end|>", StopTokens = ["<|endoftext|>"] };
        using var _ = PretrainedModel.Create(folder, spec, QwenPlugin.Qwen3ForCausalLM, files, new CreateOptions { Device = Device.Cpu, Seed = 7, Overwrite = true });
        return folder;
    }

    private static void ChatRefused()
    {
        QwenPlugin.Register();                                                   // the app's family, as MauiProgram registers it
        string folder = TinyChatModel("chat-refused");
        long? limit = ComputeResources.CpuMemoryLimit;
        var pressure = new ManualMemoryPressure();
        using var models = new ModelService(pressure, new IdrakMauiOptions());
        var service = new ChatModelService(models);
        try
        {
            // The measured need: the weights in float32, and less in int8; the cache by the context.
            var request = new ChatLoadRequest("lab/tiny", folder, "cpu", null, null, 128);
            var estimate = service.Estimate(request);
            var spec = new DecoderSpec
            {
                Vocabulary = 260, Dim = 32, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 64, MaxPositions = 256,
                Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
            };
            Check(estimate.WeightBytes == spec.ParameterCount * 4 && estimate.CacheBytes == 2L * 2 * 1 * 128 * 16 * 4 && estimate.AvailableBytes > 0 && estimate.Fits,
                $"estimate {estimate}: weights {estimate.WeightBytes} of {spec.ParameterCount} parameters, cache {estimate.CacheBytes}");
            var packed = service.Estimate(request with { Weights = "int8", KeyValueFormat = "int8" });
            Check(packed.WeightBytes < estimate.WeightBytes && packed.CacheBytes < estimate.CacheBytes, $"int8: {packed}");

            // A limit below the need refuses before loading, naming it; the page shows the refusal.
            ComputeResources.CpuMemoryLimit = 4096;
            var refused = Throws<ModelDoesNotFitException>(() => service.AcquireAsync(request), "a model above the limit");
            Check(refused.Message.Contains("does not fit", StringComparison.Ordinal) && refused.Message.Contains("CPU memory limit of 4 KB", StringComparison.Ordinal)
                  && refused.Estimate.AvailableBytes <= 4096 && models.Loaded.Count == 0, refused.Message);
            var page = new ChatViewModel(new ListedStore(Stored(folder, "lab/tiny")), service, new IdrakMauiOptions { Device = "cpu" });
            page.RefreshAsync().GetAwaiter().GetResult();
            Check(!page.LoadAsync(page.Models[0]).GetAwaiter().GetResult() && page.Error.Contains("does not fit", StringComparison.Ordinal)
                  && page.Error.Contains("CPU memory limit", StringComparison.Ordinal) && page.Model is null, page.Error);
            ComputeResources.CpuMemoryLimit = limit;

            // A family nobody registered fails loudly, named.
            string unknown = Temp("chat-unregistered");
            File.WriteAllText(Path.Combine(unknown, "config.json"), """{"architectures": ["NoSuchFamilyForCausalLM"]}""");
            Throws<NotRegisteredException>(() => service.AcquireAsync(request with { Path = unknown }), "an unregistered family");
            var other = new ChatViewModel(new ListedStore(Stored(unknown, "lab/unknown")), service, new IdrakMauiOptions { Device = "cpu" });
            other.RefreshAsync().GetAwaiter().GetResult();
            Check(!other.LoadAsync(other.Models[0]).GetAwaiter().GetResult() && other.Error.Contains("'NoSuchFamilyForCausalLM' is not registered in PretrainedArchitectures", StringComparison.Ordinal)
                  && other.Error.Contains(QwenPlugin.Qwen3ForCausalLM, StringComparison.Ordinal), other.Error);
        }
        finally
        {
            ComputeResources.CpuMemoryLimit = limit;
        }
    }

    private static void ChatEndToEnd()
    {
        QwenPlugin.Register();
        string folder = TinyChatModel("chat-end-to-end");
        var pressure = new ManualMemoryPressure();
        var options = new IdrakMauiOptions { Device = "cpu", ChatMaxNewTokens = 6, ChatTemperature = 0, ChatContext = 128, ModelKeepAlive = null };
        using var models = new ModelService(pressure, options);
        var page = new ChatViewModel(new ListedStore(Stored(folder, "lab/tiny")), new ChatModelService(models), options);
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models.Single();
        Check(page.Loading!.GetAwaiter().GetResult(), $"not loaded: {page.Error}");
        Check(page.Model is { Device: "cpu", Weights: "float32", Context: 128, ThinkingDefault: null } && page.ModelText.Contains("loaded in", StringComparison.Ordinal)
              && models.Loaded.Single().Host == ChatModelService.HostName, $"{page.Model}; {page.ModelText}");

        string Answer(string question)
        {
            page.Draft = question;
            Check(page.SendAsync().GetAwaiter().GetResult(), $"no answer: {page.Error}");
            return page.Messages[^1].Footer;
        }

        // At most the 6 tokens asked for (fewer when the random weights sample the end of the turn), counted as they came.
        static int Tokens(string footer) => int.Parse(footer.Split(' ')[0], System.Globalization.NumberStyles.AllowThousands, System.Globalization.CultureInfo.InvariantCulture);
        string first = Answer("hello");
        Check(Tokens(first) is >= 1 and <= 6 && first.EndsWith("cold", StringComparison.Ordinal) && page.History.Count == 2 && page.HasCold
              && page.LiveText.StartsWith(first.Split(' ')[0] + " token", StringComparison.Ordinal), $"first answer: '{first}', '{page.ColdText}', '{page.LiveText}'");
        string second = Answer("and then?");
        Check(Tokens(second) is >= 1 and <= 6 && second.EndsWith("warm", StringComparison.Ordinal) && page.HasWarm, $"second answer: '{second}', '{page.WarmText}'");

        // Memory pressure takes the model between answers: the next one loads it again and is cold again.
        pressure.Raise(MemoryPressureLevel.Critical, "onLowMemory");
        Check(models.Loaded.Count == 0, "the idle model stayed loaded under pressure");
        string third = Answer("still there?");
        Check(third.EndsWith("cold", StringComparison.Ordinal) && models.Loaded.Count == 1 && page.History.Count == 6, $"after the pressure: '{third}'");
    }

    private static void ChatPageOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var page = MobileFeatures.Open("chat", services);
        Check(page is ChatPage { BindingContext: ChatViewModel { Copy: not null, OpenModels: not null } } && page.Title == "Chat", $"page {page.GetType().Name}, title {page.Title}");
        Check(MobileFeatures.Origin("chat") == Overrides.Library && MobileFeatures.Get("chat") is ChatFeature { Order: 20, Needs: PhoneNeeds.Model }, "the library's chat feature");
    }
}
