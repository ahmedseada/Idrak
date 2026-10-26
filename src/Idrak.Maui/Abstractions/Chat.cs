// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;

namespace Idrak.Maui.Abstractions;

/// <summary>A chat model to load: where it is and how (the device, the weights and the key/value cache formats, the context).</summary>
/// <param name="Model">Its name as the user knows it ("Qwen/Qwen3-0.6B").</param>
/// <param name="Path">Its folder (config.json and weights) or its .gguf file.</param>
/// <param name="Device">The device: "auto" (<see cref="Device.Default"/>), "cpu", "vulkan:0" and the like.</param>
/// <param name="Weights">The projections' format once loaded ("int8", "int4", "bfloat16", a registered packed format), or null for the library's (float32).</param>
/// <param name="KeyValueFormat">The key/value cache's format (a registered layout), or null for the library's (float32).</param>
/// <param name="Context">The context window in tokens (capped by the model's own).</param>
public sealed record ChatLoadRequest(string Model, string Path, string Device, string? Weights, string? KeyValueFormat, int Context)
{
    /// <summary>The key the model is loaded under in the model service's "chat" host: one loaded model a key.</summary>
    public string Key => string.Join('\n', System.IO.Path.GetFullPath(Path), Device.ToLowerInvariant(), Weights ?? "", KeyValueFormat ?? "", Context.ToString(System.Globalization.CultureInfo.InvariantCulture));
}

/// <summary>What a chat model needs and what the device has, measured before it is loaded.</summary>
/// <param name="Model">The model.</param>
/// <param name="Device">The device it is for ("cpu", "vulkan:0").</param>
/// <param name="WeightBytes">The weights' bytes in the format asked for (projections in it, embeddings and norms in float32).</param>
/// <param name="CacheBytes">The key/value cache's bytes for the context.</param>
/// <param name="AvailableBytes">The bytes the device can still take, as measured now (within a limit the settings set), or null when it reports none.</param>
/// <param name="Limit">What bounds <paramref name="AvailableBytes"/>: the device's measured memory, or the limit the settings set (named).</param>
public sealed record ChatMemoryEstimate(string Model, string Device, long WeightBytes, long CacheBytes, long? AvailableBytes, string Limit)
{
    /// <summary>The weights and the cache.</summary>
    public long NeededBytes => WeightBytes + CacheBytes;

    /// <summary>Whether it fits (true where the device reports no memory: the load itself then says).</summary>
    public bool Fits => AvailableBytes is not { } available || NeededBytes <= available;

    /// <summary>"about 1.2 GB (weights 1.1 GB, key/value cache 117 MB) of 5.6 GB available on cpu (its measured memory)".</summary>
    public override string ToString() =>
        $"about {ProgressText.Bytes(NeededBytes)} (weights {ProgressText.Bytes(WeightBytes)}, key/value cache {ProgressText.Bytes(CacheBytes)})"
        + (AvailableBytes is { } a ? $" of {ProgressText.Bytes(a)} available on {Device} ({Limit})" : $" on {Device}, which reports no free memory");
}

/// <summary>A chat model refused before loading: it does not fit the memory the device has now.</summary>
public sealed class ModelDoesNotFitException : InvalidOperationException
{
    /// <summary>The refusal of <paramref name="estimate"/>'s model.</summary>
    public ModelDoesNotFitException(ChatMemoryEstimate estimate)
        : base($"{estimate?.Model} does not fit: it needs {estimate} — the limit is {estimate?.Limit}. "
               + "Choose a smaller weights format (int8, int4) or key/value cache format, a shorter context, another device, or a smaller model in Settings.")
    {
        Estimate = estimate ?? throw new ArgumentNullException(nameof(estimate));
    }

    /// <summary>What was measured.</summary>
    public ChatMemoryEstimate Estimate { get; }
}

/// <summary>A loaded chat model as the chat page shows it.</summary>
/// <param name="Model">Its name.</param>
/// <param name="Device">The device it is on ("cpu", "vulkan:0").</param>
/// <param name="Weights">Its projections' format ("float32", "int8").</param>
/// <param name="KeyValueFormat">Its key/value cache's format ("float32", "int8").</param>
/// <param name="Context">Its context window in tokens.</param>
/// <param name="LoadTime">How long its load took.</param>
/// <param name="ThinkingDefault">
/// Whether its template reasons when not told (true or false) where it has a switch for it (asking for reasoning and
/// suppressing it render different prompts); null when it has none.
/// </param>
/// <param name="Estimate">What it was measured to need before it was loaded.</param>
public sealed record ChatModelInfo(string Model, string Device, string Weights, string KeyValueFormat, int Context, TimeSpan LoadTime, bool? ThinkingDefault,
    ChatMemoryEstimate? Estimate)
{
    /// <summary>Whether the template has a reasoning switch.</summary>
    public bool HasThinking => ThinkingDefault is not null;
}

/// <summary>
/// A chat model in use (one answer, or a load to show its facts): dispose it when done, and the model stays loaded for the
/// keep-alive (<see cref="IdrakMauiOptions.ModelKeepAlive"/>) unless memory pressure takes it.
/// </summary>
public sealed class ChatLease : IDisposable
{
    private IDisposable? _release;

    /// <summary>A lease of <paramref name="chat"/>, released by disposing <paramref name="release"/>.</summary>
    /// <param name="chat">The model (the library's chat contract: Idrak.Text's <c>ChatGenerator</c> over the loaded model).</param>
    /// <param name="info">What the page shows of it.</param>
    /// <param name="loadDuration">The time this lease spent loading it (zero when it was loaded already).</param>
    /// <param name="release">What gives the model back (the model host's lease), or null.</param>
    public ChatLease(IChatModel chat, ChatModelInfo info, TimeSpan loadDuration, IDisposable? release = null)
    {
        Chat = chat ?? throw new ArgumentNullException(nameof(chat));
        Info = info ?? throw new ArgumentNullException(nameof(info));
        LoadDuration = loadDuration;
        _release = release;
    }

    /// <summary>The model.</summary>
    public IChatModel Chat { get; }

    /// <summary>What the page shows of it.</summary>
    public ChatModelInfo Info { get; }

    /// <summary>The time this lease spent loading it: zero when it was loaded already (its next answer is a warm one).</summary>
    public TimeSpan LoadDuration { get; }

    /// <summary>Whether this lease loaded the model.</summary>
    public bool Loaded => LoadDuration > TimeSpan.Zero;

    /// <inheritdoc />
    public void Dispose() => Interlocked.Exchange(ref _release, null)?.Dispose();
}

/// <summary>
/// The Chat lab's models: a model from the model store loaded once through the <see cref="IModelService"/> (its "chat"
/// host: shared, kept for the keep-alive, released on memory pressure) on the device of the settings, its weights in the
/// format they ask for, after checking that it fits the memory the device has now. The families are the app's: a model
/// whose family is not registered fails as the library does (<see cref="NotRegisteredException"/>).
/// </summary>
public interface IChatModelService
{
    /// <summary>What <paramref name="request"/>'s model needs against what its device has now (reads its config; loads nothing).</summary>
    /// <exception cref="NotRegisteredException">The model's family is not registered.</exception>
    ChatMemoryEstimate Estimate(ChatLoadRequest request);

    /// <summary>
    /// <paramref name="request"/>'s model, loaded once (a model loaded already is shared): first measured against the
    /// device's memory (the chat models not in use are unloaded to make room, and the cached device memory returned), then
    /// loaded with its own chat template. Runs on a worker thread.
    /// </summary>
    /// <exception cref="ModelDoesNotFitException">It does not fit the memory measured (nothing was loaded).</exception>
    /// <exception cref="NotRegisteredException">Its family is not registered.</exception>
    /// <exception cref="InvalidOperationException">It has no chat template, or the device cannot start.</exception>
    Task<ChatLease> AcquireAsync(ChatLoadRequest request, CancellationToken cancellationToken = default);
}
