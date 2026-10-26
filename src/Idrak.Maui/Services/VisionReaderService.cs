// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using Idrak.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Generation;

namespace Idrak.Maui;

/// <summary>
/// The OCR lab's readers (<see cref="IVisionReaderService"/>) in the <see cref="IModelService"/>'s "vision" host: each a
/// vision-language checkpoint loaded once by <see cref="PretrainedModel.Load"/> in the weights format asked for, read
/// through the library's <see cref="ImageReader"/>, after the Chat lab's measured check (<see cref="ChatModelService"/>):
/// the decoder's weights by their format, the vision part's (the checkpoint's tensors beyond the decoder's, in float32)
/// and the attention cache for the context, against the memory the device reports now (within the settings' limit).
/// Family-neutral: the architecture is the one config.json names, read by whichever vision family the app registered.
/// </summary>
public sealed class VisionReaderService : IVisionReaderService
{
    /// <summary>The name of the model service's host the readers live in.</summary>
    public const string HostName = "vision";

    // The vision part's parameters of a checkpoint folder, counted once from its tensors' shapes.
    private static readonly ConcurrentDictionary<string, long> VisionParameters = new(StringComparer.Ordinal);
    private readonly IModelService _models;
    private readonly ConcurrentDictionary<string, (VisionLoadRequest Request, string Folder, string Architecture, ChatMemoryEstimate Estimate)> _requests = new(StringComparer.Ordinal);
    private readonly Lazy<ModelHost<HostedReader>> _host;

    /// <summary>Readers kept in <paramref name="models"/>' "vision" host.</summary>
    public VisionReaderService(IModelService models)
    {
        _models = models ?? throw new ArgumentNullException(nameof(models));
        _host = new Lazy<ModelHost<HostedReader>>(() => _models.Host<HostedReader>(HostName, Load));
    }

    /// <inheritdoc />
    public ChatMemoryEstimate Estimate(VisionLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Measure(request).Estimate;
    }

    /// <summary>
    /// The architecture a model folder's config.json names (its first "architectures" entry), when a registered vision
    /// family reads it; null otherwise (no config.json, a text model, or a family the app did not register).
    /// </summary>
    public static string? VisionArchitecture(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        string path = Path.Combine(folder, "config.json");
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            return JsonNode.Parse(File.ReadAllBytes(path)) is JsonObject config && config["architectures"] is JsonArray { Count: > 0 } names
                   && names[0] is JsonValue first && first.TryGetValue(out string? name) && VisionFamilies.Find(name) is not null
                ? name : null;
        }
        catch (Exception e) when (e is System.Text.Json.JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    // The estimate, the folder the library reads and the architecture.
    private static (ChatMemoryEstimate Estimate, string Folder, string Architecture) Measure(VisionLoadRequest request)
    {
        var device = ChatModelService.DeviceOf(request.Device);
        string folder = CheckpointFormats.For(request.Path).Prepare(request.Path);
        string configPath = Path.Combine(folder, "config.json");
        var config = JsonNode.Parse(File.ReadAllText(configPath)) as JsonObject ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        string architecture = (string?)config["architectures"]?[0] ?? throw new InvalidDataException($"{configPath} names no architecture.");
        if (VisionFamilies.For(architecture, config) is null)
        {
            throw new InvalidOperationException($"{request.Model} reads text only (no vision part in its config.json): give a vision-language model.");
        }

        var spec = PretrainedArchitectures.Get(architecture).Spec(config, []);
        int context = Math.Min(request.Context, spec.MaxPositions);
        long vision = VisionParameters.GetOrAdd(Path.GetFullPath(folder), _ => CountVision(folder, spec));
        long weights = ChatModelService.WeightBytes(spec, request.Weights) + vision * sizeof(float);
        long cache = ModelMemory.KeyValueCacheBytes(spec, context, KeyValueLayouts.For(request.CacheFormat));
        var (available, limit) = ChatModelService.Available(device);
        return (new ChatMemoryEstimate(request.Model, device.ToString(), weights, cache, available, limit), folder, architecture);
    }

    // The checkpoint's values beyond the decoder's (the vision tower and its projector): its tensors' shapes, read from the
    // weights' headers (nothing is read into memory), less the decoder's parameters.
    private static long CountVision(string folder, DecoderSpec spec)
    {
        using var store = CheckpointFormats.For(folder).Open(folder);
        long total = 0;
        foreach (string name in store.Names)
        {
            long count = 1;
            foreach (int size in store.ShapeOf(name))
            {
                count *= size;
            }

            total += count;
        }

        return Math.Max(0, total - spec.ParameterCount);
    }

    /// <inheritdoc />
    public Task<VisionReaderLease> AcquireAsync(VisionLoadRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Task.Run(() => Acquire(request, cancellationToken), cancellationToken);
    }

    private VisionReaderLease Acquire(VisionLoadRequest request, CancellationToken cancellationToken)
    {
        var host = _host.Value;
        string key = request.Key;
        if (!host.Loaded.Any(m => m.Name == key))
        {
            var (estimate, folder, architecture) = Measure(request);
            if (!estimate.Fits)
            {
                // Room first: the readers no read uses, and the device's cached blocks.
                foreach (var idle in host.Loaded.Where(m => m.ActiveUsers == 0 && m.Name != key))
                {
                    host.Unload(idle.Name);
                }

                ComputeResources.ReleaseCachedMemory(ChatModelService.DeviceOf(request.Device));
                estimate = Measure(request).Estimate;
                if (!estimate.Fits)
                {
                    throw new ModelDoesNotFitException(estimate);
                }
            }

            _requests[key] = (request, folder, architecture, estimate);
        }

        cancellationToken.ThrowIfCancellationRequested();
        var lease = host.Acquire(key);
        return new VisionReaderLease(lease.Model, lease.Model.Info, lease.LoadDuration, lease);
    }

    // The model host's loader: the request kept under its key, loaded with its format on its device, read by ImageReader.
    private HostedReader Load(string key)
    {
        if (!_requests.TryGetValue(key, out var asked))
        {
            throw new InvalidOperationException($"No vision reader was asked for under '{key.Replace('\n', ' ')}'.");
        }

        var (request, folder, architecture, estimate) = asked;
        var device = ChatModelService.DeviceOf(request.Device);
        string? weights = request.Weights?.ToLowerInvariant();
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        var model = PretrainedModel.Load(folder, new PretrainedOptions
        {
            Device = device,
            Int8 = weights == "int8",
            Int4 = weights == "int4",
            BFloat16 = weights is "bfloat16" or "bf16",
            PackedFormatName = weights is null or "int8" or "int4" or "bfloat16" or "bf16" ? null : weights,
            MaxPositions = request.Context,
        });
        try
        {
            int context = Math.Min(request.Context, model.MaxPositions);
            var reader = new ImageReader(model, new ImageReaderOptions { ContextLength = context, CacheFormat = request.CacheFormat });
            var info = new VisionReaderInfo(request.Model, architecture, device.ToString(), weights ?? "float32", request.CacheFormat, context,
                System.Diagnostics.Stopwatch.GetElapsedTime(start), estimate);
            return new HostedReader(model, reader, info);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    // A loaded reader: the pretrained model and the library's reader over it (disposing it frees both, the reader first).
    private sealed class HostedReader(PretrainedModel model, ImageReader reader, VisionReaderInfo info) : IVisionReader, IDisposable
    {
        public VisionReaderInfo Info { get; } = info;

        public void Keep(ImageReadRequest request, ImageData prepared) => reader.Keep(request, prepared);

        public IEnumerable<ChatChunk> Stream(ImageReadRequest request, CancellationToken cancellationToken = default) => reader.Stream(request, cancellationToken);

        public void Dispose()
        {
            reader.Dispose();
            model.Dispose();
        }
    }
}
