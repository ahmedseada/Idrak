// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;

namespace Idrak.Inference;

/// <summary>
/// Text and chat models in the <see cref="InferenceEngine"/>: each copy is a <see cref="TextGenerator"/> and runs one
/// generation at a time, because it owns its KV cache. Callers reach them through <see cref="ITextModel"/> (both kinds)
/// and <see cref="IChatModel"/> (chat models): <c>engine.Model&lt;IChatModel&gt;("my-gpt")</c>.
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

    /// <summary>The tools registered with the chat model named <paramref name="name"/> (<see cref="GenerativeModelBuilder.Tools"/>), or null.</summary>
    public static ToolRegistry? ToolsOf(this InferenceEngine engine, string name)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.Model<ChatEngineModel>(name).Tools;
    }

    private static InferenceEngineBuilder AddText(InferenceEngineBuilder engine, string name, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure,
        Func<Device?, TextGenerator> load, bool reloadable, bool owns)
    {
        ArgumentNullException.ThrowIfNull(engine);
        var settings = Configure(name, configure, reloadable);
        if (settings.ChatTemplate is not null || settings.ToolRegistry is not null)
        {
            throw new InvalidOperationException($"'{name}' is a text model; Template and Tools apply to chat models.");
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
    internal ToolRegistry? ToolRegistry { get; private set; }

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

    /// <summary>Chat models: tools that can be run on the server (for example by the ASP.NET Core chat endpoint with server-side execution).</summary>
    public GenerativeModelBuilder Tools(ToolRegistry tools)
    {
        ToolRegistry = tools;
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
