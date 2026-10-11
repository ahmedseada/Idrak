// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using Idrak.Abstraction.Devices;
using Idrak.Data;

namespace Idrak.Generation;

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

        var decoded = Decode(image);
        var pixels = pipeline.IsEmpty ? decoded : pipeline.Apply(decoded);
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
    /// when it unloads the chat model these images were made for (<see cref="Idrak.Inference.GenerativeModelBuilder.Images"/>);
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
    /// when it is answered (a later turn encodes the conversation's images again); chat batches take text only.
    /// </summary>
    public ChatImages? Images { get; init; }

    private static readonly IReadOnlySet<string> TextAndImages = FrozenSet.Create(StringComparer.Ordinal, ChatParts.Text, ChatParts.Image);

    /// <summary>
    /// The kinds of message parts it takes: text, and images when <see cref="Images"/> is set. A request with another
    /// kind throws <see cref="NotSupportedException"/>.
    /// </summary>
    public IReadOnlySet<string> PartKinds => Images is null ? ChatParts.TextOnly : TextAndImages;

    /// <summary>
    /// The prompt text for a request (useful for debugging templates), its images' markers expanded to the model's image
    /// tokens (<see cref="ChatImages.Format"/>, each image as many blocks and tokens as its encoder gives it under the
    /// request's <see cref="ChatRequest.VisionOptions"/>, so its images are decoded); throws for a message part it does
    /// not take (<see cref="PartKinds"/>).
    /// </summary>
    public string RenderPrompt(ChatRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var images = Images is null ? [] : ImagesOf(request);
        return RenderPrompt(request, images.Count == 0 ? [] : [.. images.Select(i => Images!.Encoder.Blocks(Images.Read(i, request), request.VisionOptions))]);
    }

    private string RenderPrompt(ChatRequest request, IReadOnlyList<IReadOnlyList<ImageTokenLayout>> layouts)
    {
        ChatParts.ThrowIfUnsupported(this, request);
        string prompt = Template.Render(request.Messages, request.Tools ?? [], request.Think);
        prompt = Images is { } images && layouts.Count > 0 ? images.Format.Expand(prompt, layouts, Generator.Tokenizer) : prompt;
        return request.AddSpecialTokens ? Template.AddedPrefix + prompt : prompt;
    }

    // The image parts of a request, in the order the template renders them.
    private static List<ChatImage> ImagesOf(ChatRequest request) => [.. request.Messages.SelectMany(m => m.Parts).OfType<ChatImage>()];

    // A text-only model refuses images as RenderPrompt does; a model that reads them answers them one at a time.
    private void ThrowIfImages(IReadOnlyList<ChatRequest> requests)
    {
        foreach (var request in requests)
        {
            ChatParts.ThrowIfUnsupported(this, request);
        }

        if (requests.Any(r => r.Messages.Any(m => m.Parts.Any(p => p is ChatImage))))
        {
            throw new NotSupportedException("Chat batches take text only; answer a request with images on its own (Chat or Stream).");
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
    /// whole message and statistics) once all are done. The options of the first request apply to all.
    /// </summary>
    public IEnumerable<(int Index, ChatChunk Chunk)> StreamBatch(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
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
        try
        {
            return encoder.Encode(images, options);
        }
        catch (ResourceLimitExceededException) when (Generator.HoldsCache)
        {
            Generator.ReleaseCache();
            return encoder.Encode(images, options);
        }
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
        if (images.Count == 0)
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
            var source = images.Count > 0
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
