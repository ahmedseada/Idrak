// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text.Generation;

namespace Idrak.Maui.Abstractions;

/// <summary>A vision-language model to load for reading pages: where it is and how (the device, the weights' format, the context).</summary>
/// <param name="Model">Its name as the user knows it ("Qwen/Qwen2-VL-2B-Instruct").</param>
/// <param name="Path">Its folder (config.json, the weights and the vision part's files).</param>
/// <param name="Device">The device: "auto" (<see cref="Device.Default"/>), "cpu", "vulkan:0" and the like.</param>
/// <param name="Weights">The decoder's projections' format once loaded ("int8", "int4", "bfloat16", a registered packed format), or null for the library's (float32).</param>
/// <param name="CacheFormat">The attention cache's format (<see cref="ImageReaderOptions.CacheFormat"/>).</param>
/// <param name="Context">The longest prompt and answer in tokens, the page's image tokens included (capped by the model's own).</param>
public sealed record VisionLoadRequest(string Model, string Path, string Device, string? Weights, KeyValueFormat CacheFormat, int Context)
{
    /// <summary>The key the model is loaded under in the model service's vision host: one loaded model a key.</summary>
    public string Key => string.Join('\n', System.IO.Path.GetFullPath(Path), Device.ToLowerInvariant(), Weights ?? "", CacheFormat.ToString(),
        Context.ToString(System.Globalization.CultureInfo.InvariantCulture));
}

/// <summary>A loaded vision-language reader as the OCR page shows it.</summary>
/// <param name="Model">Its name.</param>
/// <param name="Architecture">The architecture its config.json names (a registered vision family's).</param>
/// <param name="Device">The device it is on ("cpu", "vulkan:0").</param>
/// <param name="Weights">Its projections' format ("float32", "int8").</param>
/// <param name="CacheFormat">Its attention cache's format.</param>
/// <param name="Context">Its context in tokens.</param>
/// <param name="LoadTime">How long its load took.</param>
/// <param name="Estimate">What it was measured to need before it was loaded (its weights with the vision part's, and the attention cache).</param>
public sealed record VisionReaderInfo(string Model, string Architecture, string Device, string Weights, KeyValueFormat CacheFormat, int Context, TimeSpan LoadTime, ChatMemoryEstimate? Estimate);

/// <summary>
/// A model reading pages: each an image and an instruction, the answer streamed. The library's is
/// <see cref="ImageReader"/> (greedy unless sampled; the family's encoder, prompt format and attention rule); the
/// tests give a scripted one.
/// </summary>
public interface IVisionReader
{
    /// <summary>
    /// Hands over <paramref name="request"/>'s page already decoded (upright) and run through its transforms, so the read
    /// that follows decodes and transforms nothing (<see cref="ImageReader.Keep"/>).
    /// </summary>
    void Keep(ImageReadRequest request, ImageData prepared);

    /// <summary>The answer to <paramref name="request"/> as it is generated (the last chunk is done, with the figures).</summary>
    IEnumerable<ChatChunk> Stream(ImageReadRequest request, CancellationToken cancellationToken = default);
}

/// <summary>
/// A vision reader in use (one read, or a load to show its facts): dispose it when done, and the model stays loaded for the
/// keep-alive (<see cref="IdrakMauiOptions.ModelKeepAlive"/>) unless memory pressure takes it.
/// </summary>
public sealed class VisionReaderLease : IDisposable
{
    private IDisposable? _release;

    /// <summary>A lease of <paramref name="reader"/>, released by disposing <paramref name="release"/>.</summary>
    /// <param name="reader">The reader.</param>
    /// <param name="info">What the page shows of it.</param>
    /// <param name="loadDuration">The time this lease spent loading it (zero when it was loaded already).</param>
    /// <param name="release">What gives the model back (the model host's lease), or null.</param>
    public VisionReaderLease(IVisionReader reader, VisionReaderInfo info, TimeSpan loadDuration, IDisposable? release = null)
    {
        Reader = reader ?? throw new ArgumentNullException(nameof(reader));
        Info = info ?? throw new ArgumentNullException(nameof(info));
        LoadDuration = loadDuration;
        _release = release;
    }

    /// <summary>The reader.</summary>
    public IVisionReader Reader { get; }

    /// <summary>What the page shows of it.</summary>
    public VisionReaderInfo Info { get; }

    /// <summary>The time this lease spent loading it: zero when it was loaded already (its next read is a warm one).</summary>
    public TimeSpan LoadDuration { get; }

    /// <summary>Whether this lease loaded the model.</summary>
    public bool Loaded => LoadDuration > TimeSpan.Zero;

    /// <inheritdoc />
    public void Dispose() => Interlocked.Exchange(ref _release, null)?.Dispose();
}

/// <summary>
/// The OCR lab's readers: a vision-language model from the model store loaded once through the <see cref="IModelService"/>
/// (its vision host: shared, kept for the keep-alive, released on memory pressure) on the device of the settings, after
/// the same measured check as the Chat lab's (<see cref="IChatModelService"/>): its decoder's weights in the format asked
/// for, its vision part's and the attention cache for the context, against what the device has now. The families are the
/// app's: a model whose vision family or text decoder is not registered fails as the library does.
/// </summary>
public interface IVisionReaderService
{
    /// <summary>What <paramref name="request"/>'s model needs against what its device has now (reads its config and its weights' shapes; loads nothing).</summary>
    /// <exception cref="NotRegisteredException">The model's text decoder is not registered.</exception>
    /// <exception cref="NotSupportedException">The model's vision family is not registered.</exception>
    ChatMemoryEstimate Estimate(VisionLoadRequest request);

    /// <summary>
    /// <paramref name="request"/>'s model, loaded once (a model loaded already is shared): first measured against the
    /// device's memory (the readers not in use are unloaded to make room, and the cached device memory returned), then
    /// loaded with its vision part. Runs on a worker thread.
    /// </summary>
    /// <exception cref="ModelDoesNotFitException">It does not fit the memory measured (nothing was loaded).</exception>
    /// <exception cref="NotRegisteredException">Its text decoder is not registered.</exception>
    /// <exception cref="NotSupportedException">Its vision family is not registered.</exception>
    /// <exception cref="InvalidOperationException">It reads text only, or the device cannot start.</exception>
    Task<VisionReaderLease> AcquireAsync(VisionLoadRequest request, CancellationToken cancellationToken = default);
}
