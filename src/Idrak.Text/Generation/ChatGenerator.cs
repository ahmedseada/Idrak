// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Imaging;
using Idrak.Text.Retrieval.Abstractions;

namespace Idrak.Text.Generation;

/// <summary>
/// What a <see cref="ChatGenerator"/> needs to read the images of a conversation (a vision-language model), all of it
/// from the model's vision family (<c>PretrainedModel.Vision</c>): the encoder that turns decoded images into their
/// tokens' features (<see cref="IVisionEncoder"/>, with the family's preprocessing and token counts), how its prompt
/// holds an image (<see cref="IImagePromptFormat"/>, with the family's token ids) and how image tokens attend
/// (<see cref="IImageAttentionRule"/>). Nothing here assumes a family.
/// </summary>
/// <param name="encoder">The family's encoder, on the model's device (its features are disposed by the generator; the encoder is not).</param>
/// <param name="format">How the rendered prompt's image markers expand (<c>PretrainedVision.PromptFormat</c>).</param>
/// <param name="attention">How the image tokens attend (<c>PretrainedVision.Attention</c>).</param>
public sealed class ChatImages(IVisionEncoder encoder, IImagePromptFormat format, IImageAttentionRule attention)
{
    /// <summary>The family's encoder.</summary>
    public IVisionEncoder Encoder { get; } = encoder ?? throw new ArgumentNullException(nameof(encoder));

    /// <summary>How the rendered prompt's image markers expand.</summary>
    public IImagePromptFormat Format { get; } = format ?? throw new ArgumentNullException(nameof(format));

    /// <summary>How the image tokens attend.</summary>
    public IImageAttentionRule Attention { get; } = attention ?? throw new ArgumentNullException(nameof(attention));

    /// <summary>
    /// How a chat image's bytes become pixels: by default the registered codecs, then the EXIF orientation
    /// (<see cref="ChatImageDecoder.Decode(ChatImage)"/>, as transformers' <c>load_image</c>).
    /// </summary>
    public Func<ChatImage, ImageData> Decode { get; init; } = ChatImageDecoder.Decode;

    /// <summary>
    /// The image transforms run on every decoded image before the encoder (a request's
    /// <see cref="ChatRequest.ImageTransforms"/> replaces them); none unless set.
    /// </summary>
    public ImageTransformPipeline Transforms { get; init; } = ImageTransformPipeline.Empty;

    /// <summary>
    /// A chat image's pixels as the encoder reads them for <paramref name="request"/>: decoded, then transformed (the
    /// request's transforms, else <see cref="Transforms"/>). The last image read is kept with its content's hash and the
    /// transforms until the same image is read again with them (a page shown as the model sees it, then answered), which
    /// takes it: decoded and transformed once, and at most one image held. The pixels returned are shared with that
    /// earlier caller and must not be changed.
    /// </summary>
    public ImageData Read(ChatImage image, ChatRequest? request = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var pipeline = request?.ImageTransforms ?? Transforms;
        string transforms = pipeline.ToString();
        lock (_lastLock)
        {
            if (_last is { } last && last.Hash == image.Hash && last.Transforms == transforms)
            {
                _last = null;                                              // taken: nothing held past the read that used it
                return last.Pixels;
            }
        }

        long start = Telemetry.Start(TelemetryLevel.Inference);
        var decoded = Decode(image);
        Telemetry.InferenceStage(InferenceStageKind.ImageDecode, null, start, (long)decoded.Channels * decoded.Height * decoded.Width, image.Data.Length);
        start = Telemetry.Start(TelemetryLevel.Inference);
        var pixels = pipeline.IsEmpty ? decoded : pipeline.Apply(decoded);
        if (!pipeline.IsEmpty)
        {
            Telemetry.InferenceStage(InferenceStageKind.ImageTransforms, null, start, (long)pixels.Channels * pixels.Height * pixels.Width);
        }
        lock (_lastLock)
        {
            _last = (image.Hash, transforms, pixels);
        }

        return pixels;
    }

    private readonly Lock _lastLock = new();
    private (string Hash, string Transforms, ImageData Pixels)? _last;

    /// <summary>
    /// The arithmetic of the encoder's matrix products (<see cref="MixedPrecision.Use"/> around the encoder only), or null
    /// for the caller's (<see cref="MixedPrecision.Current"/>). <see cref="MatMulPrecision.BFloat16"/> runs a vision
    /// tower on tensor cores where the device has them: several times faster, its features within bfloat16's rounding of
    /// float32 (check a fine-tune's answers before relying on it). The language model's products are not affected.
    /// </summary>
    public MatMulPrecision? EncoderPrecision { get; init; }

    /// <summary>
    /// What the encoder holds (its module, built on first use for example), or null: the inference engine disposes it
    /// when it unloads the chat model these images were made for (<see cref="Idrak.Text.Inference.GenerativeModelBuilder.Images"/>);
    /// otherwise whoever made them disposes it.
    /// </summary>
    public IDisposable? Owner { get; init; }
}

/// <summary>
/// Chat on top of a <see cref="TextGenerator"/>: renders the conversation with a <see cref="ChatTemplate"/>, generates
/// until the end-of-turn marker, and splits the output into reasoning, answer and tool calls as it streams.
/// </summary>
public sealed class ChatGenerator(TextGenerator generator, ChatTemplate? template = null) : IChatModel
{
    /// <summary>The underlying text generator.</summary>
    public TextGenerator Generator { get; } = generator;

    /// <summary>The prompt format.</summary>
    public ChatTemplate Template { get; } = template ?? new ChatMLTemplate();

    /// <summary>
    /// How it reads images (a vision-language model); null (the default): text only. Each request's images are encoded
    /// when it is answered (a later turn encodes the conversation's images again); ChatBatch takes text only (StreamBatch decodes requests with images together).
    /// </summary>
    public ChatImages? Images { get; init; }

    /// <summary>
    /// The sources of soft tokens for the other kinds of message part (an audio part through Idrak.Audio's
    /// <c>ChatAudioParts</c>, one source per kind, <see cref="ISoftPartSource"/>); none (the default): those kinds are
    /// refused. Each request's parts of a source's kind are encoded when it is answered, their markers expanded by the
    /// source's format in message order and their features put at the soft tokens by the prefill, as images are.
    /// Images keep their own path (<see cref="Images"/>); a request whose soft parts need more than one soft token id or
    /// attention rule (an image beside an audio, say) is refused until the prefill takes several.
    /// </summary>
    public IReadOnlyList<ISoftPartSource> Parts
    {
        get => _parts;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            foreach (var source in value)
            {
                ArgumentNullException.ThrowIfNull(source, nameof(value));
                if (source.Kind is ChatParts.Text or ChatParts.Image || value.Count(s => s.Kind == source.Kind) > 1)
                {
                    throw new ArgumentException($"Parts holds one source per kind other than text and image; '{source.Kind}' is {(source.Kind is ChatParts.Text or ChatParts.Image ? "not one" : "given twice")}.", nameof(value));
                }
            }

            _parts = value;
            _partKinds = null;
        }
    }

    private IReadOnlyList<ISoftPartSource> _parts = [];
    private IReadOnlySet<string>? _partKinds;

    private static readonly IReadOnlySet<string> TextAndImages = FrozenSet.Create(StringComparer.Ordinal, ChatParts.Text, ChatParts.Image);

    /// <summary>
    /// The kinds of message parts it takes: text, images when <see cref="Images"/> is set, and each kind of
    /// <see cref="Parts"/>. A request with another kind throws <see cref="NotSupportedException"/>.
    /// </summary>
    public IReadOnlySet<string> PartKinds => _parts.Count == 0 ? Images is null ? ChatParts.TextOnly : TextAndImages
        : _partKinds ??= FrozenSet.Create(StringComparer.Ordinal, [ChatParts.Text, .. Images is null ? [] : new[] { ChatParts.Image }, .. _parts.Select(p => p.Kind)]);

    /// <summary>
    /// The prompt text for a request (useful for debugging templates), its images' markers expanded to the model's image
    /// tokens (<see cref="ChatImages.Format"/>, each image as many blocks and tokens as its encoder gives it under the
    /// request's <see cref="ChatRequest.VisionOptions"/>, so its images are decoded) and its other soft parts' markers by
    /// their sources (<see cref="Parts"/>); throws for a message part it does not take (<see cref="PartKinds"/>).
    /// </summary>
    public string RenderPrompt(ChatRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var images = Images is null ? [] : ImagesOf(request);
        return RenderPrompt(request, images.Count == 0 ? [] : [.. images.Select(i => Images!.Encoder.Blocks(Images.Read(i, request), request.VisionOptions))], SoftLayouts(request));
    }

    private string RenderPrompt(ChatRequest request, IReadOnlyList<IReadOnlyList<ImageTokenLayout>> layouts, IReadOnlyList<(ISoftPartSource Source, List<IReadOnlyList<ImageTokenLayout>> Layouts)>? soft = null)
    {
        ChatParts.ThrowIfUnsupported(this, request);
        string prompt = Template.Render(request.Messages, request.Tools ?? [], request.Think);
        prompt = Images is { } images && layouts.Count > 0 ? images.Format.Expand(prompt, layouts, Generator.Tokenizer) : prompt;
        foreach (var (source, own) in soft ?? [])
        {
            prompt = source.Format.Expand(prompt, own, Generator.Tokenizer);          // each source touches its own markers only
        }

        return request.AddSpecialTokens ? Template.AddedPrefix + prompt : prompt;
    }

    // The image parts of a request, in the order the template renders them.
    private static List<ChatImage> ImagesOf(ChatRequest request) => [.. request.Messages.SelectMany(m => m.Parts).OfType<ChatImage>()];

    // Each source's parts of a request, in the order the template renders them (sources with none left out).
    private List<(ISoftPartSource Source, List<ChatPart> Parts)> SoftPartsOf(ChatRequest request)
    {
        var found = new List<(ISoftPartSource, List<ChatPart>)>();
        foreach (var source in _parts)
        {
            var own = request.Messages.SelectMany(m => m.Parts).Where(p => p.Kind == source.Kind).ToList();
            if (own.Count > 0)
            {
                found.Add((source, own));
            }
        }

        return found;
    }

    // Each source's parts' blocks, for expanding the prompt.
    private List<(ISoftPartSource Source, List<IReadOnlyList<ImageTokenLayout>> Layouts)> SoftLayouts(ChatRequest request) =>
        [.. SoftPartsOf(request).Select(s => (s.Source, s.Parts.Select(s.Source.Blocks).ToList()))];

    private static bool HasSoftParts(ChatRequest request) => request.Messages.Any(m => m.Parts.Any(p => p is not ChatText and not ChatImage));

    // A text-only model refuses images as RenderPrompt does; a model that reads them answers them one at a time.
    private void ThrowIfImages(IReadOnlyList<ChatRequest> requests)
    {
        foreach (var request in requests)
        {
            ChatParts.ThrowIfUnsupported(this, request);
        }

        if (requests.Any(r => r.Messages.Any(m => m.Parts.Any(p => p is ChatImage))))
        {
            throw new NotSupportedException("ChatBatch takes text only; answer requests with images with StreamBatch, or each on its own (Chat or Stream).");
        }

        if (_parts.Count > 0 && requests.Any(HasSoftParts))
        {
            throw new NotSupportedException("ChatBatch takes text only; answer requests with audio or other soft parts with StreamBatch, or each on its own (Chat or Stream).");
        }
    }

    /// <summary>Generates the complete reply.</summary>
    public ChatChunk Chat(ChatRequest request, CancellationToken cancellationToken = default) =>
        Stream(request, cancellationToken).Last();

    /// <summary><see cref="Stream"/> on a background thread, as an <c>await foreach</c> stream.</summary>
    public IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(request, token), cancellationToken);

    /// <summary><see cref="Chat"/> on a background thread.</summary>
    public Task<ChatChunk> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default) =>
        Task.Run(() => Chat(request, cancellationToken), CancellationToken.None);

    /// <summary>
    /// The complete replies to several requests, generated together (<see cref="TextGenerator.GenerateBatch"/>): the
    /// options of the first request apply to all. Each reply is what <see cref="Chat"/> returns for its request.
    /// </summary>
    public IReadOnlyList<ChatChunk> ChatBatch(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
            return [];
        }

        ThrowIfImages(requests);
        var options = requests[0].Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var outputs = Generator.GenerateBatch([.. requests.Select(RenderPrompt)], options, cancellationToken);
        var replies = new List<ChatChunk>(requests.Count);
        for (int i = 0; i < requests.Count; i++)
        {
            var request = requests[i];
            var (text, reason, stats) = outputs[i];
            var parser = new ChatOutputParser(Template, request.Tools, separateThinking: request.Think != false);
            var first = parser.Feed(text);
            var last = parser.Finish();
            var calls = first.ToolCalls.Concat(last.ToolCalls).ToList();
            string thinking = first.Thinking + last.Thinking;
            var message = new ChatMessage("assistant", (first.Content + last.Content).Trim(), thinking.Length > 0 ? thinking.Trim() : null,
                calls.Count > 0 ? calls : null);
            replies.Add(new ChatChunk(last, true, reason, message, stats));
        }

        return replies;
    }

    /// <summary>
    /// <see cref="ChatBatch"/> streamed: each request's reply as it is generated (content, thinking and tool calls parsed
    /// as <see cref="Stream"/> parses them), tagged with the request's index, then each request's final chunk (with the
    /// whole message and statistics) once all are done. The options of the first request apply to all. Requests with
    /// images are decoded together where they can be (<see cref="TextGenerator.StreamBatch(IReadOnlyList{string}, int, IReadOnlyList{IReadOnlyList{ImageFeatures}}, IImageAttentionRule, GenerationOptions, CancellationToken)"/>:
    /// prompts of one length, such as pages read with one instruction, go through the model as one batch, and prompts of
    /// different lengths where the model attends from per-row starts, <see cref="TextGenerator.SupportsBatches"/>), every
    /// request's images through the encoder in one call (when their vision options are the same object), each reply
    /// parsed on its own and its done chunk given when its batch ends; requests that cannot share a batch (tools, other
    /// generation options, text beside images) run one by one, as <see cref="Stream"/>.
    /// </summary>
    public IEnumerable<(int Index, ChatChunk Chunk)> StreamBatch(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
            yield break;
        }

        if (_parts.Count > 0 && requests.Any(HasSoftParts))
        {
            // Soft parts other than images: each request on its own (the prefill takes one soft token id per prompt).
            for (int i = 0; i < requests.Count; i++)
            {
                foreach (var chunk in Stream(requests[i], cancellationToken))
                {
                    yield return (i, chunk);
                }
            }

            yield break;
        }

        if (Images is not null && requests.Any(r => ImagesOf(r).Count > 0))
        {
            foreach (var item in StreamBatchWithImages(requests, cancellationToken))
            {
                yield return item;
            }

            yield break;
        }

        ThrowIfImages(requests);
        var options = requests[0].Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var parsers = requests.Select(r => new ChatOutputParser(Template, r.Tools, separateThinking: r.Think != false)).ToArray();
        var content = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
        var thinking = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
        var calls = requests.Select(_ => new List<ToolCall>()).ToArray();
        foreach (var chunk in Generator.StreamBatch([.. requests.Select(RenderPrompt)], options, cancellationToken))
        {
            int i = chunk.Index;
            var delta = chunk.Done ? parsers[i].Finish() : parsers[i].Feed(chunk.Text);
            content[i].Append(delta.Content);
            thinking[i].Append(delta.Thinking);
            calls[i].AddRange(delta.ToolCalls);
            if (!chunk.Done)
            {
                if (delta.Content.Length > 0 || delta.Thinking.Length > 0 || delta.ToolCalls.Count > 0)
                {
                    yield return (i, new ChatChunk(delta));
                }

                continue;
            }

            string thought = thinking[i].ToString();
            var message = Reply(parsers[i], content[i].ToString(), thought, calls[i]);
            yield return (i, new ChatChunk(delta, true, chunk.DoneReason, message, chunk.Stats));
        }
    }

    /// <summary><see cref="StreamBatch"/> on a background thread, as an <c>await foreach</c> stream.</summary>
    public IAsyncEnumerable<(int Index, ChatChunk Chunk)> StreamBatchAsync(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => StreamBatch(requests, token), cancellationToken);

    // The images' features. The KV cache kept from the last reply (reusable up to a prompt's first image only) holds device
    // memory while the encoder runs: when the encoder finds no room beside it, the cache is freed and the images encoded
    // once more, so a full device never leaves every later prompt with images failing the same way.
    private IReadOnlyList<ImageFeatures> EncodeImages(IVisionEncoder encoder, IReadOnlyList<ImageData> images, VisionOptions? options)
    {
        long start = Telemetry.Start(TelemetryLevel.Inference);
        IReadOnlyList<ImageFeatures> features;
        try
        {
            features = encoder.Encode(images, options);
        }
        catch (ResourceLimitExceededException) when (Generator.HoldsCache)
        {
            Generator.ReleaseCache();
            features = encoder.Encode(images, options);
        }

        if (start != 0)
        {
            long bytes = 0;
            foreach (var feature in features)
            {
                bytes += 4L * feature.Features.Size;
            }

            Telemetry.InferenceStage(InferenceStageKind.Encoder, Generator.Device, start, images.Count, bytes);
        }

        return features;
    }

    // StreamBatch of requests with images (see there).
    private IEnumerable<(int Index, ChatChunk Chunk)> StreamBatchWithImages(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken)
    {
        var options = requests[0].Options ?? new GenerationOptions();
        bool batched = requests.Count > 1 && Images is not null
            && requests.All(r => r.Tools is not { Count: > 0 } && ImagesOf(r).Count > 0 && SameOptions(r.Options ?? new GenerationOptions(), options));
        if (!batched)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                foreach (var chunk in Stream(requests[i], cancellationToken))
                {
                    yield return (i, chunk);
                }
            }

            yield break;
        }

        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var reader = Images!;
        var prompts = new List<string>(requests.Count);
        var decoded = new List<List<ImageData>>(requests.Count);
        var blocks = new List<List<ImageTokenLayout>>(requests.Count);
        foreach (var request in requests)
        {
            ChatParts.ThrowIfUnsupported(this, request);
            var images = ImagesOf(request).Select(i => reader.Read(i, request)).ToList();
            var layouts = images.Select(d => reader.Encoder.Blocks(d, request.VisionOptions)).ToList();
            prompts.Add(RenderPrompt(request, layouts));
            decoded.Add(images);
            blocks.Add([.. layouts.SelectMany(l => l)]);
        }

        // Every request's images through the encoder at once when they share vision options (one batch of views), else a
        // request at a time; then each request's own features, checked against its blocks.
        cancellationToken.ThrowIfCancellationRequested();
        TensorOffloading.BeforeInference(Generator.Device, Generator.Model);
        var features = new List<IReadOnlyList<ImageFeatures>>(requests.Count);
        try
        {
            using (reader.EncoderPrecision is { } precision ? MixedPrecision.Use(precision) : default(MixedPrecision.Scope?))
            {
                if (requests.All(r => ReferenceEquals(r.VisionOptions, requests[0].VisionOptions)))
                {
                    var all = EncodeImages(reader.Encoder, [.. decoded.SelectMany(d => d)], requests[0].VisionOptions);
                    int at = 0;
                    foreach (var own in blocks)
                    {
                        features.Add([.. all.Skip(at).Take(own.Count)]);
                        at += own.Count;
                    }

                    if (at != all.Count)
                    {
                        features.Add([.. all.Skip(at)]);                       // disposed below with the rest
                        throw new InvalidOperationException($"The image encoder gave {all.Count} blocks' features for {at} blocks.");
                    }
                }
                else
                {
                    for (int r = 0; r < requests.Count; r++)
                    {
                        features.Add(EncodeImages(reader.Encoder, decoded[r], requests[r].VisionOptions));
                    }
                }
            }

            for (int r = 0; r < requests.Count; r++)
            {
                if (features[r].Count != blocks[r].Count)
                {
                    throw new InvalidOperationException($"Request {r}: the image encoder gave {features[r].Count} blocks' features for {blocks[r].Count} blocks.");
                }

                for (int i = 0; i < features[r].Count; i++)
                {
                    if (!features[r][i].Layout.Equals(blocks[r][i]) || features[r][i].Features.Shape[^1] != reader.Encoder.Width)
                    {
                        throw new InvalidOperationException($"Request {r}, image block {i}: the encoder gave {Tensor.FormatShape(features[r][i].Features.Shape)} "
                            + $"({features[r][i].Layout}); its layout says {blocks[r][i]} of width {reader.Encoder.Width}.");
                    }
                }
            }
        }
        catch
        {
            foreach (var f in features.SelectMany(f => f))
            {
                f.Dispose();
            }

            throw;
        }

        try
        {
            var parsers = requests.Select(r => new ChatOutputParser(Template, r.Tools, separateThinking: r.Think != false)).ToArray();
            var content = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
            var thinking = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
            var calls = requests.Select(_ => new List<ToolCall>()).ToArray();
            foreach (var chunk in Generator.StreamBatch(prompts, reader.Format.ImageToken, features, reader.Attention, options, cancellationToken))
            {
                int r = chunk.Index;
                var delta = chunk.Done ? parsers[r].Finish() : parsers[r].Feed(chunk.Text);
                content[r].Append(delta.Content);
                thinking[r].Append(delta.Thinking);
                calls[r].AddRange(delta.ToolCalls);
                if (chunk.Done)
                {
                    yield return (r, new ChatChunk(delta, true, chunk.DoneReason, Reply(parsers[r], content[r].ToString(), thinking[r].ToString(), calls[r]),
                        chunk.Stats));
                }
                else if (!delta.IsEmpty)
                {
                    yield return (r, new ChatChunk(delta));
                }
            }
        }
        finally
        {
            foreach (var f in features.SelectMany(f => f))
            {
                f.Dispose();
            }
        }

        // Whether two requests generate alike (their stop sequences compared by value).
        static bool SameOptions(GenerationOptions a, GenerationOptions b) =>
            a with { Stop = [] } == b with { Stop = [] } && a.Stop.SequenceEqual(b.Stop);
    }

    /// <summary>Streams the reply.</summary>
    public IEnumerable<ChatChunk> Stream(ChatRequest request, CancellationToken cancellationToken = default)
    {
        var options = request.Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var parser = new ChatOutputParser(Template, request.Tools, separateThinking: request.Think != false);
        var content = new System.Text.StringBuilder();
        var thinking = new System.Text.StringBuilder();
        var calls = new List<ToolCall>();
        var images = Images is null ? [] : ImagesOf(request);
        IReadOnlyList<ImageFeatures> features = [];
        string prompt;
        var soft = _parts.Count == 0 ? [] : SoftPartsOf(request);
        (int Token, IImageAttentionRule Attention)? softInput = null;
        if (soft.Count > 0)
        {
            // One soft token id and one attention rule per prompt (the prefill replaces the runs of one token): the soft
            // parts must share them, and images (their own token and rule) go in a request of their own.
            ChatParts.ThrowIfUnsupported(this, request);
            var first = soft[0].Source;
            if (images.Count > 0)
            {
                throw new NotSupportedException($"The request holds images and '{first.Kind}' parts; the prefill takes one soft token id per prompt, so give them in separate requests.");
            }

            var other = soft.Skip(1).Select(s => s.Source).FirstOrDefault(s => s.Format.ImageToken != first.Format.ImageToken || s.Attention.Name != first.Attention.Name);
            if (other is not null)
            {
                throw new NotSupportedException($"The request holds '{first.Kind}' and '{other.Kind}' parts, whose soft tokens or attention rules differ ({first.Format.Name} against "
                    + $"{other.Format.Name}); the prefill takes one soft token id per prompt, so give them in separate requests.");
            }

            softInput = (first.Format.ImageToken, first.Attention);
            var softLayouts = soft.Select(s => (s.Source, s.Parts.Select(s.Source.Blocks).ToList())).ToList();
            prompt = RenderPrompt(request, [], softLayouts);
            var expected = softLayouts.SelectMany(s => s.Item2).SelectMany(l => l).ToList();
            cancellationToken.ThrowIfCancellationRequested();                    // before the encoders, and after them below
            TensorOffloading.BeforeInference(Generator.Device, Generator.Model);  // with offloading: the encoders' room
            var encoded = new List<ImageFeatures>(expected.Count);
            try
            {
                foreach (var (source, parts) in soft)
                {
                    long start = Telemetry.Start(TelemetryLevel.Inference);
                    var own = source.Encode(parts);
                    encoded.AddRange(own);
                    if (start != 0)
                    {
                        Telemetry.InferenceStage(InferenceStageKind.Encoder, Generator.Device, start, parts.Count, own.Sum(f => 4L * f.Features.Size));
                    }
                }

                if (encoded.Count != expected.Count)
                {
                    throw new InvalidOperationException($"The '{first.Kind}' source gave {encoded.Count} blocks' features for {expected.Count} blocks of soft tokens.");
                }

                cancellationToken.ThrowIfCancellationRequested();
                for (int i = 0; i < encoded.Count; i++)
                {
                    if (!encoded[i].Layout.Equals(expected[i]) || encoded[i].Features.Shape[^1] != first.Width)
                    {
                        throw new InvalidOperationException($"Soft block {i} ('{first.Kind}'): the source gave {Tensor.FormatShape(encoded[i].Features.Shape)} ({encoded[i].Layout}); "
                            + $"its layout says {expected[i]} of width {first.Width}.");
                    }
                }

                features = encoded;
            }
            catch
            {
                foreach (var f in encoded)
                {
                    f.Dispose();
                }

                throw;
            }
        }
        else if (images.Count == 0)
        {
            prompt = RenderPrompt(request, []);
        }
        else
        {
            // Each image decoded once: its blocks expand the prompt (template errors before the encoder runs), then the
            // family's encoder gives one features per block, which must have that block's layout.
            ChatParts.ThrowIfUnsupported(this, request);
            var reader = Images!;
            var decoded = images.Select(i => reader.Read(i, request)).ToList();
            var layouts = decoded.Select(d => reader.Encoder.Blocks(d, request.VisionOptions)).ToList();
            prompt = RenderPrompt(request, layouts);
            var blocks = layouts.SelectMany(l => l).ToList();
            cancellationToken.ThrowIfCancellationRequested();                    // before the encoder, and after it below
            TensorOffloading.BeforeInference(Generator.Device, Generator.Model);  // with offloading: the encoder's room
            using (reader.EncoderPrecision is { } precision ? MixedPrecision.Use(precision) : default(MixedPrecision.Scope?))
            {
                features = EncodeImages(reader.Encoder, decoded, request.VisionOptions);
            }
            try
            {
                if (features.Count != blocks.Count)
                {
                    throw new InvalidOperationException($"The image encoder gave {features.Count} blocks' features for {images.Count} images of {blocks.Count} blocks.");
                }

                cancellationToken.ThrowIfCancellationRequested();
                for (int i = 0; i < features.Count; i++)
                {
                    if (!features[i].Layout.Equals(blocks[i]) || features[i].Features.Shape[^1] != reader.Encoder.Width)
                    {
                        throw new InvalidOperationException($"Image block {i}: the encoder gave {Tensor.FormatShape(features[i].Features.Shape)} ({features[i].Layout}); "
                            + $"its layout says {blocks[i]} of width {reader.Encoder.Width}.");
                    }
                }
            }
            catch
            {
                foreach (var f in features)
                {
                    f.Dispose();
                }

                throw;
            }
        }

        try
        {
            var source = softInput is { } input ? Generator.Stream(prompt, input.Token, features, input.Attention, options, cancellationToken)
                : images.Count > 0
                ? Generator.Stream(prompt, Images!.Format.ImageToken, features, Images.Attention, options, cancellationToken)
                : Generator.Stream(prompt, options, cancellationToken);
            foreach (var chunk in source)
            {
                var delta = chunk.Done ? parser.Finish() : parser.Feed(chunk.Text);
                content.Append(delta.Content);
                thinking.Append(delta.Thinking);
                calls.AddRange(delta.ToolCalls);
                if (chunk.Done)
                {
                    var message = Reply(parser, content.ToString(), thinking.ToString(), calls);
                    yield return new ChatChunk(delta, true, chunk.DoneReason, message, chunk.Stats);
                }
                else if (!delta.IsEmpty)
                {
                    yield return new ChatChunk(delta);
                }
            }
        }
        finally
        {
            foreach (var f in features)
            {
                f.Dispose();
            }
        }
    }

    // The finished reply; a call written as JSON in the answer (not in the template's format) becomes that call.
    private static ChatMessage Reply(ChatOutputParser parser, string content, string thinking, List<ToolCall> calls)
    {
        if (calls.Count == 0 && parser.CallsInAnswer(content) is { } written)
        {
            calls = [.. written.Calls];
            content = written.Text;
        }

        return new ChatMessage("assistant", content.Trim(), thinking.Length > 0 ? thinking.Trim() : null, calls.Count > 0 ? [.. calls] : null);
    }
}
