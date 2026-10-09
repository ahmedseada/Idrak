// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
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
    /// tokens (<see cref="ChatImages.Format"/>, each image as many as its encoder's layout gives it, so its images are
    /// decoded); throws for a message part it does not take (<see cref="PartKinds"/>).
    /// </summary>
    public string RenderPrompt(ChatRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var images = Images is null ? [] : ImagesOf(request);
        return RenderPrompt(request, images.Count == 0 ? [] : [.. images.Select(i => Images!.Encoder.Layout(Images.Decode(i)))]);
    }

    private string RenderPrompt(ChatRequest request, IReadOnlyList<ImageTokenLayout> layouts)
    {
        ChatParts.ThrowIfUnsupported(this, request);
        string prompt = Template.Render(request.Messages, request.Tools ?? [], request.Think);
        return Images is { } images && layouts.Count > 0 ? images.Format.Expand(prompt, layouts, Generator.Tokenizer) : prompt;
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
            // Each image decoded once: its layout expands the prompt (template errors before the encoder runs), then the
            // family's encoder gives its features, which must have that layout.
            ChatParts.ThrowIfUnsupported(this, request);
            var decoded = images.Select(Images!.Decode).ToList();
            var layouts = decoded.Select(Images.Encoder.Layout).ToList();
            prompt = RenderPrompt(request, layouts);
            features = Images.Encoder.Encode(decoded);
            try
            {
                if (features.Count != images.Count)
                {
                    throw new InvalidOperationException($"The image encoder gave {features.Count} images' features for {images.Count} images.");
                }

                for (int i = 0; i < features.Count; i++)
                {
                    if (!features[i].Layout.Equals(layouts[i]) || features[i].Features.Shape[^1] != Images.Encoder.Width)
                    {
                        throw new InvalidOperationException($"Image {i}: the encoder gave {Tensor.FormatShape(features[i].Features.Shape)} ({features[i].Layout}); "
                            + $"its layout says {layouts[i]} of width {Images.Encoder.Width}.");
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
