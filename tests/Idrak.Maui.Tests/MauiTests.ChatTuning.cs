// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Specialized;
using System.Text.Json.Nodes;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

namespace Idrak.Maui.Tests;

// The chat page's header from one state, its tune sheet (each model's generation settings, kept per model and taken by the
// next reply), and the text families the app registers loaded from their own configs and answering through the model
// service (plan 34, the owner's phone runs).
internal static partial class MauiTests
{
    // The header's title, avatar and status line never disagree: a refresh whose picker drops its choice while its list is
    // made again keeps the chosen and loaded model named; one the store no longer lists stays named by the loaded model.
    private static void ChatHeaderOneState()
    {
        string first = ChatFolder("chat-header-a"), second = ChatFolder("chat-header-b");
        var listed = new List<StoredModel> { Stored(first, "lab/alpha") };
        var store = new ChangingStore(listed);
        var chats = new OneChat(new ScriptedChat(2, "ab", Stats(2, 1)));
        var page = new ChatViewModel(store, chats, new IdrakMauiOptions(), new InlineDispatcher(), new ManualTime());
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models.Single();
        Check(page.Loading!.GetAwaiter().GetResult(), page.Error);

        // As MAUI's Picker does: its list emptied, it sets the choice to null.
        page.Models.CollectionChanged += (_, e) =>
        {
            if (e.Action == NotifyCollectionChangedAction.Reset)
            {
                page.SelectedModel = null;
            }
        };
        listed.Add(Stored(second, "lab/beta"));
        page.RefreshAsync().GetAwaiter().GetResult();
        Check(page.SelectedModel?.Title == "lab/alpha" && page.ModelName == "alpha" && page.ModelInitial == "A" && page.ModelMeta.Contains("loaded in", StringComparison.Ordinal),
            $"after a refresh: '{page.ModelName}', '{page.ModelInitial}', '{page.ModelMeta}', chosen {page.SelectedModel?.Title}");

        listed.RemoveAt(0);                                                     // the chosen model left the store; it stays loaded
        page.RefreshAsync().GetAwaiter().GetResult();
        Check(page.SelectedModel is null && page.ModelName == "alpha" && page.ModelInitial == "A" && page.IsModelLoaded,
            $"the loaded model named: '{page.ModelName}', '{page.ModelInitial}', '{page.ModelMeta}'");
    }

    // Every chat setting the library has, with its default, kept per model and taken by the next reply and load.
    private static void ChatTuningSheet()
    {
        string folder = ChatFolder("chat-tuning");
        var model = new ScriptedChat(2, "ab", Stats(2, 1));
        var chats = new OneChat(model, thinking: true);
        var settings = new MemorySettingsStore();
        var options = new IdrakMauiOptions { ChatTemperature = 0.3, ChatSystemPrompt = "Be brief." };
        ChatViewModel Page()
        {
            var made = new ChatViewModel(new ListedStore(Stored(folder, "lab/tuned")), chats, options, new InlineDispatcher(), new ManualTime(), settings: settings);
            made.RefreshAsync().GetAwaiter().GetResult();
            return made;
        }

        var page = Page();
        page.OpenTuningCommand.Execute(null);
        Check(!page.IsTuning && page.Error.Contains("Choose a model", StringComparison.Ordinal), "no model chosen: the sheet says to choose one");
        page.SelectedModel = page.Models.Single();
        Check(page.Loading!.GetAwaiter().GetResult(), page.Error);
        page.OpenTuningCommand.Execute(null);
        var entries = page.Tuning.SelectMany(g => g).ToDictionary(e => e.Key);
        Check(page.IsTuning && page.Tuning.Select(g => g.Title).SequenceEqual(ChatReplySettings.Groups)
              && entries.Count == ChatReplySettings.Definitions.Count && entries.Values.All(e => e.Text.Length == 0 && e.Default.Length > 0),
            $"the sheet: {string.Join(", ", page.Tuning.Select(g => $"{g.Title} {g.Count}"))}");
        var library = new GenerationOptions();
        Check(entries["temperature"].Default == "0.3" && entries["top_k"].Default == library.TopK.ToString(System.Globalization.CultureInfo.InvariantCulture)
              && entries["system_prompt"].Default == "Be brief." && entries["context"].Default == "2048" && entries["thinking"].Default == "the model's",
            $"defaults: temperature {entries["temperature"].Default}, top-k {entries["top_k"].Default}, system '{entries["system_prompt"].Default}', context {entries["context"].Default}");

        // GenerationOptions' every field a decoder-only chat reads is on the sheet (the beam and returned-ids ones are not a chat's).
        var notAChats = new[] { "NumBeams", "LengthPenalty", "EarlyStopping", "ReturnTokens", "ReturnLogProbabilities" };
        var read = typeof(GenerationOptions).GetProperties().Select(p => p.Name).Except(notAChats).ToList();
        var mapped = typeof(ChatReplyOptions).GetProperties().Select(p => p.Name switch { "MaxNewTokens" => "NumPredict", "Context" => "NumCtx", var n => n }).ToHashSet();
        Check(read.All(mapped.Contains), $"not on the sheet: {string.Join(", ", read.Where(n => !mapped.Contains(n)))}");

        entries["temperature"].Text = "0.2";
        entries["top_k"].Text = "5";
        entries["min_p"].Text = "0.05";
        entries["repeat_penalty"].Text = "1.3";
        entries["stop"].Text = "</s>\nEND";
        entries["system_prompt"].Text = "Answer in French.";
        entries["context"].Text = "512";
        entries["repeat_stop"].Text = "1";
        entries["thinking"].SelectedChoice = "off";
        Check(entries["repeat_stop"].HasError && entries["repeat_stop"].Error.Contains("'1'", StringComparison.Ordinal), $"a refused value names it: '{entries["repeat_stop"].Error}'");
        entries["repeat_stop"].Text = "3";
        Check(!entries["repeat_stop"].HasError && page.TuningStatus.Contains("next reply", StringComparison.Ordinal), page.TuningStatus);
        entries["top_p"].SliderValue = 0.537;
        Check(entries["top_p"].Text == "0.54" && Math.Abs(entries["top_p"].SliderValue - 0.54) < 1e-9, $"the slider rounds to its step: {entries["top_p"].Text}");
        page.Temperature = 0.25;                                                // the inspector's quick setting is the model's too
        Check(entries["temperature"].Text == "0.25" && options.ChatTemperature == 0.3, $"the inspector: {entries["temperature"].Text}, app {options.ChatTemperature}");

        page.Draft = "hello";
        Check(page.SendAsync().GetAwaiter().GetResult(), page.Error);
        var request = model.Requests.Single();
        var want = ChatViewModel.Options(options) with
        {
            Temperature = 0.25f, TopK = 5, TopP = 0.54f, MinP = 0.05f, RepeatPenalty = 1.3f, Stop = ["</s>", "END"], NumCtx = 512, RepeatStop = 3,
        };
        Check(request.Options is { } given && given with { Stop = want.Stop } == want && given.Stop.SequenceEqual(want.Stop) && request.Think == false
              && request.Messages[0] is { Role: "system", Content: "Answer in French." } && chats.Requests[^1].Context == 512,
            $"the reply took them: {request.Options}, think {request.Think}, system '{request.Messages[0].Content}', load context {chats.Requests[^1].Context}");

        // Kept for the model: a page made again reads them back; Reset puts the defaults back and forgets them.
        var again = Page();
        again.SelectedModel = again.Models.Single();
        again.Loading!.GetAwaiter().GetResult();
        var kept = again.Tuning.SelectMany(g => g).ToDictionary(e => e.Key);
        Check(kept["temperature"].Text == "0.25" && kept["stop"].Text == "</s>\nEND" && kept["thinking"].Text == "off" && again.ReplyGeneration().NumCtx == 512,
            $"kept: temperature {kept["temperature"].Text}, stop '{kept["stop"].Text}', thinking '{kept["thinking"].Text}'");
        again.ResetTuningCommand.Execute(null);
        Check(again.Tuning.SelectMany(g => g).All(e => e.Text.Length == 0) && again.ReplyGeneration() == ChatViewModel.Options(options)
              && settings.Get(ChatReplySettings.KeyOf("lab/tuned") + "temperature") is null, "Reset: the defaults again, nothing kept");
    }

    // The text families the app registers (Qwen, Llama, Gemma, Mistral), each read by its config alone through the chat
    // model service and answering on the CPU in the weights formats the settings offer; a Gemma 3 whose tokenizer has a token
    // past the model's vocabulary (as gemma-3-270m-it's does) answers too.
    private static void ChatFamiliesLoad()
    {
        QwenPlugin.Register();
        LlamaPlugin.Register();
        GemmaPlugin.Register();
        MistralPlugin.Register();
        string data = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "gguf");
        string Copy(string name, string from, Action<string>? change = null)
        {
            string folder = Temp(name);
            foreach (string file in Directory.GetFiles(Path.Combine(data, from)))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            change?.Invoke(folder);
            return folder;
        }

        void Edit(string file, Action<JsonObject> change)
        {
            var json = JsonNode.Parse(File.ReadAllText(file))!.AsObject();
            change(json);
            File.WriteAllText(file, json.ToJsonString());
        }

        var models = new (string Name, string Folder)[]
        {
            ("qwen3", Copy("family-qwen3", "tiny-qwen3-q8-hf")),
            ("llama", Copy("family-llama", "tiny-llama-hf")),
            ("gemma3", Copy("family-gemma3", "tiny-gemma3-hf")),
            ("mistral", Copy("family-mistral", "tiny-llama-hf", f => Edit(Path.Combine(f, "config.json"), c =>
            {
                c["architectures"] = new JsonArray("MistralForCausalLM");
                c["model_type"] = "mistral";
            }))),
            ("gemma3-added-token", Copy("family-gemma3-added", "tiny-gemma3-hf", f => Edit(Path.Combine(f, "tokenizer.json"), t =>
            {
                int next = t["added_tokens"]!.AsArray().Max(a => (int)a!["id"]!) + 1;
                int vocab = t["model"]!["vocab"] is JsonObject words ? words.Count : t["model"]!["vocab"]!.AsArray().Count;
                t["added_tokens"]!.AsArray().Add(new JsonObject { ["id"] = Math.Max(next, vocab), ["content"] = "<image_soft_token>", ["special"] = true });
            }))),
        };

        var failures = new List<string>();
        foreach (string? weights in new[] { null, "int8", "int4" })
        {
            var options = new IdrakMauiOptions { Device = "cpu", ChatWeights = weights, ChatMaxNewTokens = 4, ChatTemperature = 0, ChatContext = 64, ModelKeepAlive = null };
            using var service = new ModelService(new ManualMemoryPressure(), options);
            var chats = new ChatModelService(service);                          // one for the model service, as the app has
            foreach (var (name, folder) in models)
            {
                var page = new ChatViewModel(new ListedStore(Stored(folder, "lab/" + name)), chats, options, new InlineDispatcher(), new ManualTime());
                page.RefreshAsync().GetAwaiter().GetResult();
                page.SelectedModel = page.Models.Single();
                if (!page.Loading!.GetAwaiter().GetResult())
                {
                    failures.Add($"{name} {weights ?? "float32"}: not loaded: {page.Error}");
                    continue;
                }

                page.Draft = "hello";
                if (!page.SendAsync().GetAwaiter().GetResult())
                {
                    failures.Add($"{name} {weights ?? "float32"}: no answer: {page.Error}");
                }
            }
        }

        Check(failures.Count == 0, string.Join("; ", failures));
    }

    // A store whose list the test changes between refreshes.
    private sealed class ChangingStore(List<StoredModel> models) : IModelStore
    {
        public string Folder => Path.GetTempPath();

        public bool HasToken => false;

        public Task<StoredModel> PullAsync(string model, string? revision = null, IProgressReporter? progress = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public IReadOnlyList<StoredModel> List() => [.. models];

        public bool Remove(string id) => false;

        public StorageSpace? FreeSpace() => null;
    }
}
