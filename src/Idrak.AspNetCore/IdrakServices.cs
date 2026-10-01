// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;

namespace Idrak.AspNetCore;

/// <summary>Registers Idrak's <see cref="InferenceEngine"/> in dependency injection.</summary>
public static class IdrakServiceCollectionExtensions
{
    /// <summary>
    /// Adds one <see cref="InferenceEngine"/> (built when the host starts, disposed when it stops) and returns a builder
    /// to add models to it. Resolve <see cref="InferenceEngine"/>, or <see cref="IPredictor{TIn, TOut}"/> (keyed by model
    /// name, and unkeyed for the last predictor of each type), in endpoints and services.
    /// </summary>
    public static IdrakBuilder AddIdrak(this IServiceCollection services)
    {
        var existing = services.FirstOrDefault(d => d.ServiceType == typeof(IdrakBuilder))?.ImplementationInstance;
        if (existing is IdrakBuilder builder)
        {
            return builder;
        }

        builder = new IdrakBuilder(services);
        services.AddSingleton(builder);
        services.AddSingleton<IdrakEngineHost>();
        services.AddHostedService(sp => sp.GetRequiredService<IdrakEngineHost>());
        services.AddSingleton(sp => sp.GetRequiredService<IdrakEngineHost>().Engine);
        return builder;
    }
}

/// <summary>
/// Adds models to the engine registered by <see cref="IdrakServiceCollectionExtensions.AddIdrak"/>. Each method
/// matches an <see cref="InferenceEngineBuilder"/> method; overloads taking an <see cref="IServiceProvider"/> let
/// model factories and tools use services from dependency injection.
/// </summary>
public sealed class IdrakBuilder
{
    private readonly List<Action<IServiceProvider, InferenceEngineBuilder>> _steps = [];

    internal IdrakBuilder(IServiceCollection services) => Services = services;

    /// <summary>The service collection.</summary>
    public IServiceCollection Services { get; }

    /// <summary>A predictor from a package; see <see cref="InferenceEngineBuilder.Predictor{TIn, TOut}(string, string, Func{PredictorBuilder{float[], float[]}, PredictorBuilder{TIn, TOut}})"/>.</summary>
    public IdrakBuilder AddPredictor<TIn, TOut>(string name, string packagePath,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure)
    {
        _steps.Add((_, engine) => engine.Predictor(name, packagePath, configure));
        return RegisterPredictor<TIn, TOut>(name);
    }

    /// <summary>A predictor whose model is made by <paramref name="load"/>, which may use services.</summary>
    public IdrakBuilder AddPredictor<TIn, TOut>(string name, Func<IServiceProvider, Module> load,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure)
    {
        _steps.Add((services, engine) => engine.Predictor(name, () => load(services), configure));
        return RegisterPredictor<TIn, TOut>(name);
    }

    /// <summary>A text model from a package with a tokenizer.</summary>
    public IdrakBuilder AddTextModel(string name, string packagePath, string tokenizer, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        Step((_, engine) => engine.TextModel(name, packagePath, tokenizer, configure));

    /// <summary>A text model made by <paramref name="load"/>; <paramref name="configure"/> may use services.</summary>
    public IdrakBuilder AddTextModel(string name, Func<IServiceProvider, TextGenerator> load,
        Func<IServiceProvider, GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        Step((services, engine) => engine.TextModel(name, () => load(services), configure is null ? null : b => configure(services, b)));

    /// <summary>A chat model from a package with a tokenizer.</summary>
    public IdrakBuilder AddChatModel(string name, string packagePath, string tokenizer, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        Step((_, engine) => engine.ChatModel(name, packagePath, tokenizer, configure));

    /// <summary>A chat model from a package; <paramref name="configure"/> may use services (for example to build its tools).</summary>
    public IdrakBuilder AddChatModel(string name, string packagePath, string tokenizer,
        Func<IServiceProvider, GenerativeModelBuilder, GenerativeModelBuilder> configure) =>
        Step((services, engine) => engine.ChatModel(name, packagePath, tokenizer, b => configure(services, b)));

    /// <summary>A chat model whose text generator is made by <paramref name="load"/>; <paramref name="configure"/> may use services.</summary>
    public IdrakBuilder AddChatModel(string name, Func<IServiceProvider, TextGenerator> load,
        Func<IServiceProvider, GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        Step((services, engine) => engine.ChatModel(name, () => load(services), configure is null ? null : b => configure(services, b)));

    /// <summary><see cref="InferenceEngineBuilder.LoadOnFirstUse"/>: the app starts without loading models.</summary>
    public IdrakBuilder LoadOnFirstUse() => Step((_, engine) => engine.LoadOnFirstUse());

    /// <summary><see cref="InferenceEngineBuilder.Telemetry"/>.</summary>
    public IdrakBuilder Telemetry() => Step((_, engine) => engine.Telemetry());

    /// <summary>Any other engine setting.</summary>
    public IdrakBuilder Configure(Action<IServiceProvider, InferenceEngineBuilder> configure) => Step(configure);

    internal InferenceEngineBuilder CreateEngine(IServiceProvider services)
    {
        var engine = InferenceEngine.Create();
        foreach (var step in _steps)
        {
            step(services, engine);
        }

        return engine;
    }

    private IdrakBuilder Step(Action<IServiceProvider, InferenceEngineBuilder> step)
    {
        _steps.Add(step);
        return this;
    }

    private IdrakBuilder RegisterPredictor<TIn, TOut>(string name)
    {
        Services.AddKeyedSingleton<IPredictor<TIn, TOut>>(name, (sp, _) => new EnginePredictor<TIn, TOut>(sp.GetRequiredService<InferenceEngine>(), name));
        Services.AddSingleton<IPredictor<TIn, TOut>>(sp => new EnginePredictor<TIn, TOut>(sp.GetRequiredService<InferenceEngine>(), name));
        return this;
    }

    private sealed class EnginePredictor<TIn, TOut>(InferenceEngine engine, string name) : IPredictor<TIn, TOut>
    {
        public ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default) =>
            engine.PredictAsync<TIn, TOut>(name, input, cancellationToken);

        public ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default) =>
            engine.PredictAsync<TIn, TOut>(name, inputs, cancellationToken);
    }
}

/// <summary>Builds the engine when the host starts and disposes it when the host stops.</summary>
internal sealed class IdrakEngineHost(IdrakBuilder builder, IServiceProvider services) : IHostedService, IAsyncDisposable
{
    private InferenceEngine? _engine;

    public InferenceEngine Engine => _engine ?? throw new InvalidOperationException("The Idrak engine is created when the host starts.");

    public async Task StartAsync(CancellationToken cancellationToken) =>
        _engine = await builder.CreateEngine(services).BuildAsync(cancellationToken).ConfigureAwait(false);

    public async Task StopAsync(CancellationToken cancellationToken) => await DisposeAsync().ConfigureAwait(false);

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _engine, null) is { } engine)
        {
            await engine.DisposeAsync().ConfigureAwait(false);
        }
    }
}
