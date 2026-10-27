// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// One chat model's own generation settings (the chat page's tune sheet): every <see cref="GenerationOptions"/> value a
/// decoder-only chat reads, and the chat's own (the system prompt, thinking, the tokenizer's special tokens). A null value
/// is the default: the app's chat setting (<see cref="IdrakMauiOptions"/>'s Chat values) where it has one, else the
/// library's. Kept per model by <see cref="ChatReplySettings"/>; the next reply uses them. The beam settings
/// (<see cref="GenerationOptions.NumBeams"/>, <see cref="GenerationOptions.LengthPenalty"/>,
/// <see cref="GenerationOptions.EarlyStopping"/>) are an encoder-decoder's, and the returned ids and log-probabilities
/// (<see cref="GenerationOptions.ReturnTokens"/>) training's: a chat reads neither, so neither is here.
/// </summary>
public sealed class ChatReplyOptions
{
    /// <summary>The system prompt (a system message the model's template renders); null: the app's.</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>Whether the reply reasons, where the template has a switch (<see cref="ChatRequest.Think"/>); null: the app's, else the model's own.</summary>
    public bool? Thinking { get; set; }

    /// <summary><see cref="ChatRequest.AddSpecialTokens"/>; null: off (the template's tokens alone).</summary>
    public bool? AddSpecialTokens { get; set; }

    /// <summary><see cref="GenerationOptions.Temperature"/>.</summary>
    public double? Temperature { get; set; }

    /// <summary><see cref="GenerationOptions.TopK"/>.</summary>
    public int? TopK { get; set; }

    /// <summary><see cref="GenerationOptions.TopP"/>.</summary>
    public double? TopP { get; set; }

    /// <summary><see cref="GenerationOptions.MinP"/>.</summary>
    public double? MinP { get; set; }

    /// <summary><see cref="GenerationOptions.Seed"/>.</summary>
    public int? Seed { get; set; }

    /// <summary><see cref="GenerationOptions.RepeatPenalty"/>.</summary>
    public double? RepeatPenalty { get; set; }

    /// <summary><see cref="GenerationOptions.RepeatLastN"/>.</summary>
    public int? RepeatLastN { get; set; }

    /// <summary><see cref="GenerationOptions.PresencePenalty"/>.</summary>
    public double? PresencePenalty { get; set; }

    /// <summary><see cref="GenerationOptions.FrequencyPenalty"/>.</summary>
    public double? FrequencyPenalty { get; set; }

    /// <summary><see cref="GenerationOptions.RepeatStop"/>.</summary>
    public int? RepeatStop { get; set; }

    /// <summary><see cref="GenerationOptions.RepeatStopMinTokens"/>.</summary>
    public int? RepeatStopMinTokens { get; set; }

    /// <summary><see cref="GenerationOptions.NumPredict"/> (the most new tokens of a reply).</summary>
    public int? MaxNewTokens { get; set; }

    /// <summary><see cref="GenerationOptions.NumCtx"/> (the context window; a change loads the model again with its cache).</summary>
    public int? Context { get; set; }

    /// <summary><see cref="GenerationOptions.SlideContext"/>.</summary>
    public bool? SlideContext { get; set; }

    /// <summary><see cref="GenerationOptions.Stop"/>: the texts a reply ends at; null: none.</summary>
    public IReadOnlyList<string>? Stop { get; set; }

    /// <summary><see cref="GenerationOptions.ChunkSize"/>.</summary>
    public int? ChunkSize { get; set; }

    /// <summary><see cref="GenerationOptions.PrefillChunk"/>.</summary>
    public int? PrefillChunk { get; set; }

    /// <summary><see cref="GenerationOptions.PrefillMinChunk"/>.</summary>
    public int? PrefillMinChunk { get; set; }

    /// <summary><see cref="GenerationOptions.UseCache"/>.</summary>
    public bool? UseCache { get; set; }

    /// <summary><see cref="GenerationOptions.UseGraph"/>.</summary>
    public bool? UseGraph { get; set; }

    /// <summary>A copy.</summary>
    public ChatReplyOptions Clone() => (ChatReplyOptions)MemberwiseClone();

    /// <summary>The generation options of a reply: these values over <paramref name="defaults"/> (the app's, over the library's).</summary>
    public GenerationOptions Over(GenerationOptions defaults)
    {
        ArgumentNullException.ThrowIfNull(defaults);
        return defaults with
        {
            Temperature = Temperature is { } temperature ? (float)temperature : defaults.Temperature,
            TopK = TopK ?? defaults.TopK,
            TopP = TopP is { } topP ? (float)topP : defaults.TopP,
            MinP = MinP is { } minP ? (float)minP : defaults.MinP,
            Seed = Seed ?? defaults.Seed,
            RepeatPenalty = RepeatPenalty is { } repeat ? (float)repeat : defaults.RepeatPenalty,
            RepeatLastN = RepeatLastN ?? defaults.RepeatLastN,
            PresencePenalty = PresencePenalty is { } presence ? (float)presence : defaults.PresencePenalty,
            FrequencyPenalty = FrequencyPenalty is { } frequency ? (float)frequency : defaults.FrequencyPenalty,
            RepeatStop = RepeatStop ?? defaults.RepeatStop,
            RepeatStopMinTokens = RepeatStopMinTokens ?? defaults.RepeatStopMinTokens,
            NumPredict = MaxNewTokens ?? defaults.NumPredict,
            NumCtx = Context ?? defaults.NumCtx,
            SlideContext = SlideContext ?? defaults.SlideContext,
            Stop = Stop ?? defaults.Stop,
            ChunkSize = ChunkSize ?? defaults.ChunkSize,
            PrefillChunk = PrefillChunk ?? defaults.PrefillChunk,
            PrefillMinChunk = PrefillMinChunk ?? defaults.PrefillMinChunk,
            UseCache = UseCache ?? defaults.UseCache,
            UseGraph = UseGraph ?? defaults.UseGraph,
        };
    }
}

/// <summary>
/// One setting of the tune sheet: its definition (key, title, help, how it reads and writes), its group, its default as
/// the app has it now, and the range a slider shows for a number that has one (typing takes any value the setting takes).
/// </summary>
/// <param name="Setting">Its definition.</param>
/// <param name="Group">Its group on the sheet ("Sampling").</param>
/// <param name="Default">Its default as text, from the app's options (its chat setting, else the library's value).</param>
/// <param name="Range">A slider's least, most and step, or null for none.</param>
/// <param name="Lines">Edited over several lines (the system prompt, the stop texts: one a line).</param>
/// <param name="Reloads">A change loads the model again (the context sizes its key/value cache).</param>
public sealed record ChatReplySetting(LabSetting<ChatReplyOptions> Setting, string Group, Func<IdrakMauiOptions, string> Default,
    (double Minimum, double Maximum, double Step)? Range = null, bool Lines = false, bool Reloads = false);

/// <summary>
/// The tune sheet's settings kept per chat model, each under <see cref="Prefix"/>, the model's key and its own key in the
/// app's <see cref="ISettingsStore"/> (an empty value removed: the default).
/// </summary>
public static class ChatReplySettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "chat.model.";

    /// <summary>The groups, in the sheet's order.</summary>
    public static IReadOnlyList<string> Groups { get; } = ["Sampling", "Repetition", "Length", "Conversation", "Speed"];

    /// <summary>Every setting, in the sheet's order.</summary>
    public static IReadOnlyList<ChatReplySetting> Definitions { get; } = Define();

    /// <summary>The prefix of <paramref name="model"/>'s keys ("idrak.chat.model.Qwen_Qwen3-0.6B.").</summary>
    public static string KeyOf(string model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        return Prefix + string.Create(model.Length, model, static (span, m) =>
        {
            for (int i = 0; i < m.Length; i++)
            {
                span[i] = char.IsAsciiLetterOrDigit(m[i]) || m[i] is '-' or '_' or '@' ? m[i] : '_';
            }
        }) + ".";
    }

    /// <summary><paramref name="model"/>'s settings kept in <paramref name="store"/> (null values where none is); a kept value that no longer reads is skipped and named.</summary>
    public static ChatReplyOptions Load(ISettingsStore store, string model, out IReadOnlyList<string> skipped)
    {
        ArgumentNullException.ThrowIfNull(store);
        return LabSettings.Load(store, KeyOf(model), [.. Definitions.Select(d => d.Setting)], new ChatReplyOptions(), out skipped);
    }

    /// <summary>Keeps every value of <paramref name="options"/> for <paramref name="model"/> (an empty one removed: the default).</summary>
    public static void Save(ISettingsStore store, string model, ChatReplyOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        LabSettings.Save(store, KeyOf(model), [.. Definitions.Select(d => d.Setting)], options);
    }

    /// <summary>Keeps <paramref name="setting"/>'s value of <paramref name="options"/> for <paramref name="model"/> alone (an empty one removed).</summary>
    public static void Save(ISettingsStore store, string model, ChatReplySetting setting, ChatReplyOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(setting);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(KeyOf(model) + setting.Setting.Key, LabSettings.Empty(setting.Setting.Get(options)));
    }

    /// <summary>Forgets <paramref name="model"/>'s settings: its replies take the defaults again.</summary>
    public static void Clear(ISettingsStore store, string model) => Save(store, model, new ChatReplyOptions());

    private static IReadOnlyList<ChatReplySetting> Define()
    {
        static string Show(float value) => value.ToString(CultureInfo.InvariantCulture);       // the shortest text that reads back as the float

        // The app's generation options (its chat settings over the library's).
        static GenerationOptions App(IdrakMauiOptions o) => Idrak.Maui.ViewModels.ChatViewModel.Options(o);

        static LabSetting<ChatReplyOptions> Number(string key, string title, string summary, Func<ChatReplyOptions, string> read, Action<ChatReplyOptions, string> write) =>
            new(key, title, summary, LabSettingKind.Number, "", read, write);

        static LabSetting<ChatReplyOptions> Switch(string key, string title, string summary, Func<ChatReplyOptions, bool?> read, Action<ChatReplyOptions, bool?> write) =>
            new(key, title, summary, LabSettingKind.Choice, "default", o => LabSettings.Show(read(o)), (o, t) => write(o, LabSettings.Switch(t)), () => ["default", "on", "off"]);

        static string OnOff(bool value) => value ? "on" : "off";

        return
        [
            new(Number("temperature", "Temperature", "How varied the reply is: 0 takes the likeliest token, above 1 more varied (GenerationOptions.Temperature).",
                o => LabSettings.Show(o.Temperature), (o, t) => o.Temperature = LabSettings.Real(t, atLeast: 0)), "Sampling", o => Show(App(o).Temperature), (0, 2, 0.05)),
            new(Number("top_k", "Top-k", "Sample among this many likeliest tokens; 0: all (GenerationOptions.TopK).",
                o => LabSettings.Show(o.TopK), (o, t) => o.TopK = LabSettings.Whole(t, 0)), "Sampling", o => App(o).TopK.ToString(CultureInfo.InvariantCulture), (0, 200, 1)),
            new(Number("top_p", "Top-p", "Sample among the likeliest tokens holding this share of the probability, in (0, 1]; 1 is off (GenerationOptions.TopP).",
                o => LabSettings.Show(o.TopP), (o, t) => o.TopP = LabSettings.Real(t, above: 0, atMost: 1)), "Sampling", o => Show(App(o).TopP), (0.01, 1, 0.01)),
            new(Number("min_p", "Min-p", "Drop tokens less likely than this share of the likeliest one; 0 is off (GenerationOptions.MinP).",
                o => LabSettings.Show(o.MinP), (o, t) => o.MinP = LabSettings.Real(t, atLeast: 0, atMost: 1)), "Sampling", o => Show(App(o).MinP), (0, 1, 0.01)),
            new(Number("seed", "Seed", "The random seed, for replies that repeat; empty: random (GenerationOptions.Seed).",
                o => LabSettings.Show(o.Seed), (o, t) => o.Seed = LabSettings.Whole(t, int.MinValue)), "Sampling",
                o => App(o).Seed is { } seed ? seed.ToString(CultureInfo.InvariantCulture) : "random"),
            new(Number("repeat_penalty", "Repetition penalty", "Lowers the tokens seen in the recent window; 1 is off (GenerationOptions.RepeatPenalty).",
                o => LabSettings.Show(o.RepeatPenalty), (o, t) => o.RepeatPenalty = LabSettings.Real(t, above: 0)), "Repetition", o => Show(App(o).RepeatPenalty), (0.5, 2, 0.01)),
            new(Number("repeat_last_n", "Repetition window", "The recent tokens (the prompt's too) the penalties look at; 0 turns them off (GenerationOptions.RepeatLastN).",
                o => LabSettings.Show(o.RepeatLastN), (o, t) => o.RepeatLastN = LabSettings.Whole(t, 0)), "Repetition", o => App(o).RepeatLastN.ToString(CultureInfo.InvariantCulture),
                (0, 1024, 8)),
            new(Number("presence_penalty", "Presence penalty", "Subtracted from every token in the window (GenerationOptions.PresencePenalty).",
                o => LabSettings.Show(o.PresencePenalty), (o, t) => o.PresencePenalty = LabSettings.Real(t)), "Repetition", o => Show(App(o).PresencePenalty), (-2, 2, 0.05)),
            new(Number("frequency_penalty", "Frequency penalty", "Subtracted from a token once per time it is in the window (GenerationOptions.FrequencyPenalty).",
                o => LabSettings.Show(o.FrequencyPenalty), (o, t) => o.FrequencyPenalty = LabSettings.Real(t)), "Repetition", o => Show(App(o).FrequencyPenalty), (-2, 2, 0.05)),
            new(Number("repeat_stop", "Stop a loop after", "Ends the reply when one run of tokens repeats this many times in a row; 0 is off, at least 2 (GenerationOptions.RepeatStop).",
                o => LabSettings.Show(o.RepeatStop), (o, t) => o.RepeatStop = LoopCount(t)), "Repetition", o => App(o).RepeatStop.ToString(CultureInfo.InvariantCulture), (0, 16, 1)),
            new(Number("repeat_stop_min_tokens", "Shortest loop (tokens)", "The shortest run the loop stop counts (GenerationOptions.RepeatStopMinTokens).",
                o => LabSettings.Show(o.RepeatStopMinTokens), (o, t) => o.RepeatStopMinTokens = LabSettings.Whole(t, 1, 64)), "Repetition",
                o => App(o).RepeatStopMinTokens.ToString(CultureInfo.InvariantCulture), (1, 64, 1)),
            new(Number("max_new_tokens", "Max new tokens", "The most tokens one reply has (GenerationOptions.NumPredict); empty: until the turn ends or the context is full.",
                o => LabSettings.Show(o.MaxNewTokens), (o, t) => o.MaxNewTokens = LabSettings.Whole(t, 1)), "Length",
                o => App(o).NumPredict > 0 ? App(o).NumPredict.ToString(CultureInfo.InvariantCulture) : "until the turn ends"),
            new(Number("context", "Context (tokens)", "The context window, capped by the model's own; it sizes the key/value cache, so a change loads the model again (GenerationOptions.NumCtx).",
                o => LabSettings.Show(o.Context), (o, t) => o.Context = LabSettings.Whole(t, 16)), "Length", o => App(o).NumCtx.ToString(CultureInfo.InvariantCulture), Reloads: true),
            new(Switch("slide_context", "Slide the context", "A reply that fills the context goes on with the window's last half re-read (on) or ends there (off) (GenerationOptions.SlideContext).",
                o => o.SlideContext, (o, v) => o.SlideContext = v), "Length", o => OnOff(App(o).SlideContext)),
            new(new LabSetting<ChatReplyOptions>("stop", "Stop texts", "The reply ends where one of these appears, not shown; one a line (GenerationOptions.Stop).", LabSettingKind.Text, "",
                o => o.Stop is { Count: > 0 } stop ? string.Join('\n', stop) : "",
                (o, t) => o.Stop = t.Split(['\n', '\r'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is { Length: > 0 } lines ? lines : null), "Length",
                o => App(o).Stop.Count > 0 ? string.Join(", ", App(o).Stop) : "none", Lines: true),
            new(new LabSetting<ChatReplyOptions>("system_prompt", "System prompt", "The instructions the chat starts with, as a system message the model's template renders.", LabSettingKind.Text, "",
                o => o.SystemPrompt ?? "", (o, t) => o.SystemPrompt = t.Length == 0 ? null : t), "Conversation", o => o.ChatSystemPrompt is { Length: > 0 } prompt ? prompt : "none", Lines: true),
            new(Switch("thinking", "Thinking", "Whether the reply reasons first, where the model's template has a switch (ChatRequest.Think).",
                o => o.Thinking, (o, v) => o.Thinking = v), "Conversation", o => o.ChatThinking is { } think ? OnOff(think) : "the model's"),
            new(Switch("add_special_tokens", "Add special tokens", "Encode the rendered prompt with the tokenizer's special tokens added, as a pipeline that renders the template to text does (ChatRequest.AddSpecialTokens).",
                o => o.AddSpecialTokens, (o, v) => o.AddSpecialTokens = v), "Conversation", _ => "off"),
            new(Number("chunk_size", "Tokens a step", "Tokens generated between two reads of the device while streaming (GenerationOptions.ChunkSize).",
                o => LabSettings.Show(o.ChunkSize), (o, t) => o.ChunkSize = LabSettings.Whole(t, 1)), "Speed", o => App(o).ChunkSize.ToString(CultureInfo.InvariantCulture), (1, 64, 1)),
            new(Number("prefill_chunk", "Prompt piece (positions)", "The positions a prompt's pass feeds at a time; 0: measured from the memory free (GenerationOptions.PrefillChunk, IDRAK_PREFILL_CHUNK).",
                o => LabSettings.Show(o.PrefillChunk), (o, t) => o.PrefillChunk = LabSettings.Whole(t, 0)), "Speed",
                o => App(o).PrefillChunk > 0 ? App(o).PrefillChunk.ToString(CultureInfo.InvariantCulture) : "measured"),
            new(Number("prefill_min_chunk", "Least prompt piece (positions)", "The fewest positions a measured piece takes; 0: the library's (GenerationOptions.PrefillMinChunk, IDRAK_PREFILL_MIN_CHUNK).",
                o => LabSettings.Show(o.PrefillMinChunk), (o, t) => o.PrefillMinChunk = LabSettings.Whole(t, 0)), "Speed",
                o => App(o).PrefillMinChunk > 0 ? App(o).PrefillMinChunk.ToString(CultureInfo.InvariantCulture) : GenerationOptions.DefaultPrefillMinChunk.ToString(CultureInfo.InvariantCulture)),
            new(Switch("use_cache", "Key/value cache", "Decode a token at a time from the cache (on), or read the whole window again for every token (off) (GenerationOptions.UseCache).",
                o => o.UseCache, (o, v) => o.UseCache = v), "Speed", o => OnOff(App(o).UseCache)),
            new(Switch("use_graph", "Recorded steps", "Record the decoding step once and replay it where the device can (GenerationOptions.UseGraph).",
                o => o.UseGraph, (o, v) => o.UseGraph = v), "Speed", o => OnOff(App(o).UseGraph)),
        ];
    }

    // 0 (off) or a count of 2 or more.
    private static int? LoopCount(string text) => LabSettings.Whole(text, 0) is { } count && count == 1
        ? throw new FormatException("'1' is not a loop count: 0 (off) or 2 or more.") : LabSettings.Whole(text, 0);
}
