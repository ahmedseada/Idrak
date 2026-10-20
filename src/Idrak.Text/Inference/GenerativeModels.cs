// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Inference;
using Idrak.Text.Generation;

namespace Idrak.Text.Inference;

/// <summary>
/// Text, chat and encoder-decoder models in the <see cref="InferenceEngine"/>: each copy is a <see cref="TextGenerator"/>
/// (an encoder-decoder's: a <see cref="Seq2SeqGenerator"/>) and runs one generation at a time, because it owns its KV
/// cache. Callers reach them through <see cref="ITextModel"/> (every kind) and <see cref="IChatModel"/> (chat models):
/// <c>engine.Model&lt;IChatModel&gt;("my-gpt")</c>.
/// </summary>
public static class GenerativeModels
{
    // ------------------------------------------------------------------ text models

    /// <summary>A text model from a package holding a network (with its architecture) and a tokenizer named <paramref name="tokenizer"/>.</summary>
    public static InferenceEngineBuilder TextModel(this InferenceEngineBuilder engine, string name, string packagePath, string tokenizer,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddText(engine, name, configure, device => FromPackage(packagePath, tokenizer, device), reloadable: true, owns: true);

    /// <summary>A text model made by <paramref name="load"/> (called once per copy; the engine disposes the model on unload).</summary>
    public static InferenceEngineBuilder TextModel(this InferenceEngineBuilder engine, string name, Func<TextGenerator> load,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddText(engine, name, configure, _ => load(), reloadable: true, owns: true);

    /// <summary>A text model you already created (one copy; never unloaded or disposed by the engine).</summary>
    public static InferenceEngineBuilder TextModel(this InferenceEngineBuilder engine, string name, TextGenerator generator,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddText(engine, name, configure, _ => generator, reloadable: false, owns: false);

    // ------------------------------------------------------------------ chat models

    /// <summary>A chat model from a package holding a network (with its architecture) and a tokenizer named <paramref name="tokenizer"/>.</summary>
    public static InferenceEngineBuilder ChatModel(this InferenceEngineBuilder engine, string name, string packagePath, string tokenizer,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddChat(engine, name, configure, device => FromPackage(packagePath, tokenizer, device), reloadable: true, owns: true);

    /// <summary>A chat model whose text generator is made by <paramref name="load"/> (called once per copy).</summary>
    public static InferenceEngineBuilder ChatModel(this InferenceEngineBuilder engine, string name, Func<TextGenerator> load,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddChat(engine, name, configure, _ => load(), reloadable: true, owns: true);

    /// <summary>A chat model over a text generator you already created (one copy; never unloaded or disposed by the engine).</summary>
    public static InferenceEngineBuilder ChatModel(this InferenceEngineBuilder engine, string name, TextGenerator generator,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddChat(engine, name, configure, _ => generator, reloadable: false, owns: false);

    // ------------------------------------------------------------------ encoder-decoder models

    /// <summary>
    /// An encoder-decoder (sequence to sequence) model read from <paramref name="folder"/> by the family its architecture
    /// is registered under (<c>PretrainedSeq2Seq</c>, a plug-in's), with the folder's tokenizer: the engine's "seq2seq"
    /// kind, reached through <see cref="ITextModel"/> (<c>engine.Model&lt;ITextModel&gt;("translator")</c>). Each copy is a
    /// <see cref="Seq2SeqGenerator"/> on the builder's device; the engine disposes it on unload.
    /// </summary>
    public static InferenceEngineBuilder Seq2SeqModel(this InferenceEngineBuilder engine, string name, string folder,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        return AddSeq2Seq(engine, name, configure, device => Idrak.Models.PretrainedSeq2SeqModel.Load(folder, new Idrak.Models.PretrainedEncoderOptions { Device = device }).CreateGenerator(),
            reloadable: true, owns: true);
    }

    /// <summary>An encoder-decoder model whose generator is made by <paramref name="load"/> (called once per copy; the engine disposes its model on unload).</summary>
    public static InferenceEngineBuilder Seq2SeqModel(this InferenceEngineBuilder engine, string name, Func<Seq2SeqGenerator> load,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(load);
        return AddSeq2Seq(engine, name, configure, _ => load(), reloadable: true, owns: true);
    }

    /// <summary>An encoder-decoder generator you already created (one copy; never unloaded or disposed by the engine).</summary>
    public static InferenceEngineBuilder Seq2SeqModel(this InferenceEngineBuilder engine, string name, Seq2SeqGenerator generator,
        Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        return AddSeq2Seq(engine, name, configure, _ => generator, reloadable: false, owns: false);
    }

    private static InferenceEngineBuilder AddSeq2Seq(InferenceEngineBuilder engine, string name, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure,
        Func<Device?, Seq2SeqGenerator> load, bool reloadable, bool owns)
    {
        ArgumentNullException.ThrowIfNull(engine);
        var settings = Configure(name, configure, reloadable);
        if (settings.ChatTemplate is not null || settings.ToolRegistry is not null || settings.ImageReader is not null || settings.PartReader is not null)
        {
            throw new InvalidOperationException($"'{name}' is an encoder-decoder model; Template, Tools, Images and Parts apply to chat models.");
        }

        return engine.Add(name, new Seq2SeqEngineModel(settings, load, owns, reloadable));
    }

    private static InferenceEngineBuilder AddText(InferenceEngineBuilder engine, string name, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure,
        Func<Device?, TextGenerator> load, bool reloadable, bool owns)
    {
        ArgumentNullException.ThrowIfNull(engine);
        var settings = Configure(name, configure, reloadable);
        if (settings.ChatTemplate is not null || settings.ToolRegistry is not null || settings.ImageReader is not null || settings.PartReader is not null)
        {
            throw new InvalidOperationException($"'{name}' is a text model; Template, Tools and Images apply to chat models.");
        }

        return engine.Add(name, new TextEngineModel(settings, load, owns, reloadable));
    }

    private static InferenceEngineBuilder AddChat(InferenceEngineBuilder engine, string name, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure,
        Func<Device?, TextGenerator> load, bool reloadable, bool owns)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.Add(name, new ChatEngineModel(Configure(name, configure, reloadable), load, owns, reloadable));
    }

    private static GenerativeModelBuilder Configure(string name, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure, bool reloadable)
    {
        var settings = (configure ?? (b => b))(new GenerativeModelBuilder());
        if (!reloadable && settings.KeepAliveSet)
        {
            throw new InvalidOperationException($"'{name}' uses a model object you created, which the engine cannot copy or reload; " +
                "Instances and KeepAlive need a package or a factory.");
        }

        return settings;
    }

    private static TextGenerator FromPackage(string path, string tokenizer, Device? device)
    {
        using var package = ModelPackage.Open(path);
        return package.TextGenerator(tokenizer, device: device);
    }
}

/// <summary>Settings for a text or chat model in the engine. Each is off (or the underlying class's own default) unless set.</summary>
public sealed class GenerativeModelBuilder
{
    internal GenerativeModelBuilder()
    {
    }

    internal int? InstanceCount { get; private set; }
    internal TimeSpan? KeepAliveTime { get; private set; }
    internal bool KeepAliveSet { get; private set; }
    internal int? QueueLimitCount { get; private set; }
    internal TimeSpan? TimeoutLimit { get; private set; }
    internal string? WarmUpPrompt { get; private set; }
    internal Device? TargetDevice { get; private set; }
    internal ChatTemplate? ChatTemplate { get; private set; }
    internal IToolRegistry? ToolRegistry { get; private set; }
    internal Func<TextGenerator, ChatImages>? ImageReader { get; private set; }
    internal Func<TextGenerator, IReadOnlyList<ISoftPartSource>>? PartReader { get; private set; }
    internal IReadOnlyCollection<string> PartReaderKinds { get; private set; } = [];

    /// <summary>Loads <paramref name="count"/> copies, so that many generations run at the same time.</summary>
    public GenerativeModelBuilder Instances(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        InstanceCount = count;
        return this;
    }

    /// <summary>Unloads the model after it has been idle this long (null keeps it forever, <see cref="TimeSpan.Zero"/> unloads after each request).</summary>
    public GenerativeModelBuilder KeepAlive(TimeSpan? idle)
    {
        KeepAliveTime = idle;
        KeepAliveSet = true;
        return this;
    }

    /// <summary>Rejects new requests while <paramref name="waiting"/> are already waiting for a free copy.</summary>
    public GenerativeModelBuilder QueueLimit(int waiting)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(waiting);
        QueueLimitCount = waiting;
        return this;
    }

    /// <summary>Stops a request that has not finished after <paramref name="limit"/> (queue wait included).</summary>
    public GenerativeModelBuilder Timeout(TimeSpan limit)
    {
        TimeoutLimit = limit;
        return this;
    }

    /// <summary>Generates one token from <paramref name="prompt"/> right after each copy loads.</summary>
    public GenerativeModelBuilder WarmUp(string prompt)
    {
        WarmUpPrompt = prompt;
        return this;
    }

    /// <summary>The device for package sources (factories place their models themselves).</summary>
    public GenerativeModelBuilder Device(Device device)
    {
        TargetDevice = device;
        return this;
    }

    /// <summary>Chat models: the prompt format (the <see cref="ChatGenerator"/> constructor's <c>template</c>).</summary>
    public GenerativeModelBuilder Template(ChatTemplate template)
    {
        ChatTemplate = template;
        return this;
    }

    /// <summary>
    /// Chat models: tools that can be run on the server (<see cref="IToolChatModel.Tools"/>; for example by the ASP.NET Core chat
    /// endpoint with server-side execution).
    /// </summary>
    public GenerativeModelBuilder Tools(IToolRegistry tools)
    {
        ToolRegistry = tools;
        return this;
    }

    /// <summary>
    /// Chat models: they read images (a vision-language model). <paramref name="images"/> is called for each copy once
    /// its text generator loads, and gives how that copy reads images (<see cref="ChatGenerator.Images"/>: the model's
    /// image tokens, its prompt format and its encoder on the copy's device); the engine disposes
    /// <see cref="ChatImages.Owner"/> when it unloads the copy. The model then takes image parts
    /// (<see cref="IChatModel.PartKinds"/>) when its chat template renders them; a request is answered with its images
    /// encoded once each time (no cache across requests).
    /// </summary>
    public GenerativeModelBuilder Images(Func<TextGenerator, ChatImages> images)
    {
        ImageReader = images ?? throw new ArgumentNullException(nameof(images));
        return this;
    }

    /// <summary>
    /// Chat models: they read other soft parts (an audio-language model's sounds; <see cref="ChatGenerator.Parts"/>).
    /// <paramref name="parts"/> is called for each copy once its text generator loads and gives the copy's sources (on its
    /// device); the engine disposes each one that is <see cref="IDisposable"/> when it unloads the copy. The model then
    /// takes parts of <paramref name="kinds"/> (<see cref="IChatModel.PartKinds"/>) that its chat template renders.
    /// </summary>
    public GenerativeModelBuilder Parts(IReadOnlyCollection<string> kinds, Func<TextGenerator, IReadOnlyList<ISoftPartSource>> parts)
    {
        ArgumentNullException.ThrowIfNull(kinds);
        PartReader = parts ?? throw new ArgumentNullException(nameof(parts));
        PartReaderKinds = [.. kinds];
        return this;
    }

    /// <summary>How the engine hosts the copies: one generation at a time per copy.</summary>
    internal EngineHosting Hosting(bool reloadable) => new()
    {
        Instances = InstanceCount ?? 1,
        SharedCopies = false,
        Reloadable = reloadable,
        KeepAlive = KeepAliveTime,
        QueueLimit = QueueLimitCount,
        Timeout = TimeoutLimit,
    };

    /// <summary>Loads one text generator on the chosen device and warms it up.</summary>
    internal TextGenerator Load(Func<Device?, TextGenerator> load)
    {
        var generator = load(TargetDevice);
        if (WarmUpPrompt is { } prompt)
        {
            generator.Generate(prompt, new GenerationOptions { NumPredict = 1 });
        }

        return generator;
    }
}
