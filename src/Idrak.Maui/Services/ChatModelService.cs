// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using Idrak.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;

namespace Idrak.Maui;

/// <summary>
/// The Chat lab's models (<see cref="IChatModelService"/>) in the <see cref="IModelService"/>'s "chat" host: each loaded
/// once by <see cref="PretrainedModel.Load"/> in the weights format asked for, with its own chat template
/// (<see cref="PretrainedModelExtensions"/>' <c>CreateChat</c>), after measuring that its weights and key/value cache fit
/// the memory the device reports now. Family-neutral: the architecture is the one config.json names, read by whichever
/// family the app registered.
/// </summary>
public sealed class ChatModelService : IChatModelService
{
    /// <summary>The name of the model service's host the chat models live in.</summary>
    public const string HostName = "chat";

    private static readonly ConcurrentDictionary<string, double> BytesPerValue = new(StringComparer.OrdinalIgnoreCase);
    private readonly IModelService _models;
    private readonly ConcurrentDictionary<string, (ChatLoadRequest Request, string Folder, ChatMemoryEstimate Estimate)> _requests = new(StringComparer.Ordinal);
    private readonly Lazy<ModelHost<HostedChat>> _host;

    /// <summary>Chat models kept in <paramref name="models"/>' "chat" host.</summary>
    public ChatModelService(IModelService models)
    {
        _models = models ?? throw new ArgumentNullException(nameof(models));
        _host = new Lazy<ModelHost<HostedChat>>(() => _models.Host<HostedChat>(HostName, Load));
    }

    /// <inheritdoc />
    public ChatMemoryEstimate Estimate(ChatLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Measure(request).Estimate;
    }

    // The estimate, and the folder the library reads (prepared once: Load takes it as it is).
    private static (ChatMemoryEstimate Estimate, string Folder) Measure(ChatLoadRequest request)
    {
        var device = DeviceOf(request.Device);
        var (folder, spec) = Read(request.Path);
        int context = Math.Min(request.Context, spec.MaxPositions);
        long weights = WeightBytes(spec, request.Weights);
        var layout = request.KeyValueFormat is { } name ? KeyValueLayouts.Get(name) : KeyValueLayouts.For(KeyValueFormat.Float32);
        long cache = ModelMemory.KeyValueCacheBytes(spec, context, layout);
        var (available, limit) = Available(device);
        return (new ChatMemoryEstimate(request.Model, device.ToString(), weights, cache, available, limit), folder);
    }

    // The decoder's weights in the format asked for: the projections (and a separate head) packed, the token table and the
    // norms in float32.
    internal static long WeightBytes(DecoderSpec spec, string? format)
    {
        long embedding = (long)spec.Vocabulary * spec.Dim;
        long table = embedding * (spec.TieEmbeddings ? 1 : 2) + (spec.LearnedPositions ? (long)spec.MaxPositions * spec.Dim : 0);
        long projections = Math.Max(0, spec.ParameterCount - table);
        long head = spec.TieEmbeddings ? 0 : embedding;
        return (long)((projections + head) * Bytes(format)) + (table - head) * sizeof(float);
    }

    /// <inheritdoc />
    public Task<ChatLease> AcquireAsync(ChatLoadRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Task.Run(() => Acquire(request, cancellationToken), cancellationToken);
    }

    private ChatLease Acquire(ChatLoadRequest request, CancellationToken cancellationToken)
    {
        var host = _host.Value;
        string key = request.Key;
        if (!host.Loaded.Any(m => m.Name == key))
        {
            var (estimate, folder) = Measure(request);
            if (!estimate.Fits)
            {
                // Room first: the chat models no answer uses, and the device's cached blocks.
                foreach (var idle in host.Loaded.Where(m => m.ActiveUsers == 0 && m.Name != key))
                {
                    host.Unload(idle.Name);
                }

                ComputeResources.ReleaseCachedMemory(DeviceOf(request.Device));
                estimate = Measure(request).Estimate;
                if (!estimate.Fits)
                {
                    throw new ModelDoesNotFitException(estimate);
                }
            }

            _requests[key] = (request, folder, estimate);
        }

        cancellationToken.ThrowIfCancellationRequested();
        var lease = host.Acquire(key);
        return new ChatLease(lease.Model.Chat, lease.Model.Info, lease.LoadDuration, lease);
    }

    // The model host's loader: the request kept under its key, loaded with its formats on its device.
    private HostedChat Load(string key)
    {
        if (!_requests.TryGetValue(key, out var asked))
        {
            throw new InvalidOperationException($"No chat model was asked for under '{key.Replace('\n', ' ')}'.");
        }

        var (request, folder, estimate) = asked;
        var device = DeviceOf(request.Device);
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
            if (model.ChatTemplate is null)
            {
                throw new InvalidOperationException($"{request.Model} has no chat template (tokenizer_config.json, chat_template.jinja): it is not a chat model.");
            }

            var layout = request.KeyValueFormat is { } kv ? KeyValueLayouts.Get(kv) : KeyValueLayouts.For(KeyValueFormat.Float32);
            int context = Math.Min(request.Context, model.MaxPositions);
            var chat = model.CreateChat(layout, context);
            var info = new ChatModelInfo(request.Model, device.ToString(), weights ?? "float32", request.KeyValueFormat ?? "float32", context,
                System.Diagnostics.Stopwatch.GetElapsedTime(start), ThinkingDefault(chat.Template), estimate);
            return new HostedChat(model, chat, info);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Whether <paramref name="template"/> reasons when not told, where it has a switch for it: true or false when asking for
    /// reasoning and suppressing it render different prompts (the one it renders untold says which it does), null when they
    /// render the same (no switch). Family-neutral: the template itself answers.
    /// </summary>
    public static bool? ThinkingDefault(IChatTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        try
        {
            IReadOnlyList<ChatMessage> messages = [new ChatMessage("user", "hi")];
            string on = template.Render(messages, [], true), off = template.Render(messages, [], false), untold = template.Render(messages, [], null);
            return on == off ? null : untold != off;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;                                                        // a template that cannot render this is not one to switch
        }
    }

    // The folder the library reads (a .gguf file prepared once into the cache, as loading it does) and its spec through the
    // family the app registered (NotRegisteredException when none is). The vision readers' service measures with it too.
    internal static (string Folder, DecoderSpec Spec) Read(string path)
    {
        string folder = Prepared(path);
        string configPath = Path.Combine(folder, "config.json");
        var config = JsonNode.Parse(File.ReadAllText(configPath)) as JsonObject ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        string architecture = (string?)config["architectures"]?[0] ?? throw new InvalidDataException($"{configPath} names no architecture.");
        return (folder, PretrainedArchitectures.Get(architecture).Spec(config, []));
    }

    private static string Prepared(string path) => CheckpointFormats.For(path).Prepare(path);

    internal static Device DeviceOf(string device) => IdrakMauiSettings.IsAuto(device) ? Device.Default : Device.Parse(device);

    // Bytes a weight value takes in the format: 4 in float32, else measured once by packing a sample on the CPU.
    internal static double Bytes(string? format) => format is null ? sizeof(float) : BytesPerValue.GetOrAdd(format is "bf16" ? "bfloat16" : format, static name =>
    {
        const int Side = 256;
        var values = new float[Side * Side];
        var random = new Random(1);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        using var weight = PackedWeightFormats.Pack(name, values, Side, Side, Device.Cpu);
        return weight.Bytes / (double)values.Length;
    });

    // What the device can take now (its own report: free memory and the pool's reusable blocks), within the limit the
    // settings set for its kind; and which of the two bounds it.
    internal static (long? Bytes, string Limit) Available(Device device)
    {
        long? measured = device.Backend.AvailableMemory();
        long? limit = device.IsGpu ? ComputeResources.GpuMemoryLimit : ComputeResources.CpuMemoryLimit;
        if (limit is { } set)
        {
            long room = Math.Max(0, set - ComputeResources.GetMemoryUsage(device).InUse);
            if (measured is not { } m || room < m)
            {
                return (room, $"the {(device.IsGpu ? "GPU" : "CPU")} memory limit of {ProgressText.Bytes(set)} in Settings");
            }
        }

        return (measured, measured is null ? "not reported" : $"the memory {device} reports free now");
    }

    // A loaded chat model: the pretrained model and its chat generator (disposing it frees the model's memory).
    private sealed class HostedChat(PretrainedModel model, ChatGenerator chat, ChatModelInfo info) : IDisposable
    {
        public ChatGenerator Chat { get; } = chat;

        public ChatModelInfo Info { get; } = info;

        public void Dispose() => model.Dispose();
    }
}
