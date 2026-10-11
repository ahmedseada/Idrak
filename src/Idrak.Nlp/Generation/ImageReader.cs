// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Models;
using Idrak.Nlp;

namespace Idrak.Generation;

/// <summary>How an <see cref="ImageReader"/> reads: the images' preparation and the attention cache.</summary>
public sealed record ImageReaderOptions
{
    /// <summary>The image transforms every page goes through before the family's own processor (a read may give its own).</summary>
    public ImageTransformPipeline Transforms { get; init; } = ImageTransformPipeline.Empty;

    /// <summary>The vision family's options for every image (null: the family's defaults).</summary>
    public VisionOptions? VisionOptions { get; init; }

    /// <summary>The attention cache's format.</summary>
    public KeyValueFormat CacheFormat { get; init; } = KeyValueFormat.Float32;

    /// <summary>The longest prompt and answer in tokens, the image's included (the model's maximum when null).</summary>
    public int? ContextLength { get; init; }

    /// <summary>
    /// Keep the attention cache after each read for the next one (see <see cref="TextGenerator.KeepCache"/>). Off by default:
    /// a cache is reused only up to a prompt's first image, a few template tokens when every read is a new page, so keeping
    /// it would hold the whole context's cache on the device between reads for almost nothing.
    /// </summary>
    public bool KeepCache { get; init; }

    /// <summary>Makes the vision encoder instead of the model's own (tests feed reference features with it).</summary>
    public Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? EncoderFactory { get; init; }
}

/// <summary>One page to read: the image, the instruction, and what replaces the reader's own settings for it.</summary>
/// <param name="Image">The page.</param>
/// <param name="Prompt">The instruction sent after the image in the user turn.</param>
public sealed record ImageReadRequest(ChatImage Image, string Prompt)
{
    /// <summary>A system turn before the user turn (none when null).</summary>
    public string? System { get; init; }

    /// <summary>The most tokens the answer may take.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>This page's image transforms instead of the reader's (<see cref="ImageReaderOptions.Transforms"/>).</summary>
    public ImageTransformPipeline? Transforms { get; init; }

    /// <summary>This page's vision options instead of the reader's.</summary>
    public VisionOptions? VisionOptions { get; init; }

    /// <summary>Divides the scores of tokens seen in the last <see cref="RepeatLastN"/> tokens (1 = off, as greedy reading is by default).</summary>
    public float RepeatPenalty { get; init; } = 1f;

    /// <summary>Subtracted from the score of every token seen in the last <see cref="RepeatLastN"/> tokens (0 = off).</summary>
    public float PresencePenalty { get; init; }

    /// <summary>Subtracted from a token's score once per time it was seen in the last <see cref="RepeatLastN"/> tokens (0 = off).</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary>How many recent tokens (prompt included) the penalties look at.</summary>
    public int RepeatLastN { get; init; } = 64;

    /// <summary>
    /// Ends the answer when it repeats one run of tokens this many times in a row (0 = off; see
    /// <see cref="GenerationOptions.RepeatStop"/>): a read caught in a loop stops instead of filling <see cref="MaxTokens"/>.
    /// </summary>
    public int RepeatStop { get; init; }
}

/// <summary>A page read: the answer, the generation's figures and the time it took.</summary>
public sealed record ImageReading(string Text, GenerationStats? Stats, TimeSpan Elapsed)
{
    /// <summary>When the answer's first text came (null when it had none).</summary>
    public TimeSpan? FirstText { get; init; }

    /// <summary>Whether the answer stopped at its token limit.</summary>
    public bool Truncated { get; init; }

    /// <summary>Why the answer ended: "stop" (the model finished), "length" (the token limit), "repetition" (a loop, see <see cref="ImageReadRequest.RepeatStop"/>).</summary>
    public string? DoneReason { get; init; }
}

/// <summary>
/// A vision-language model reading pages: each one an image and an instruction in one user turn, the answer generated
/// greedily (the same answer every time, as document reading wants). Its images go through <see cref="ModelImages"/>
/// (the family's encoder, prompt format and attention rule; the reader's transforms unless a read gives its own). The
/// reader does not own the model; dispose the model after the reader.
/// </summary>
public sealed class ImageReader : IDisposable
{
    private readonly ModelImages _images;
    private readonly ChatGenerator _chat;
    private readonly int _contextLength;

    /// <summary>A reader of <paramref name="model"/>'s images.</summary>
    /// <exception cref="InvalidOperationException">The model reads text only.</exception>
    /// <exception cref="ArgumentException">A vision option its family does not know.</exception>
    public ImageReader(PretrainedModel model, ImageReaderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        options ??= new ImageReaderOptions();
        Model = model;
        Options = options;
        _images = ModelImages.For(model, options.Transforms, options.VisionOptions, options.EncoderFactory)
                  ?? throw new InvalidOperationException($"{model.Folder} reads text only (no vision encoder): give a vision-language checkpoint.");
        try
        {
            _contextLength = Math.Min(options.ContextLength ?? model.MaxPositions, model.MaxPositions);
            var chat = model.CreateChat(options.CacheFormat, _contextLength);
            chat.Generator.KeepCache = options.KeepCache;
            _chat = new ChatGenerator(chat.Generator, chat.Template) { Images = _images.Images };
        }
        catch
        {
            _images.Dispose();
            throw;
        }
    }

    // The chat the pages are read with (the tests look at its generator).
    internal ChatGenerator Chat => _chat;

    /// <summary>The model read with.</summary>
    public PretrainedModel Model { get; }

    /// <summary>The reader's settings.</summary>
    public ImageReaderOptions Options { get; }

    /// <summary>The model's vision family.</summary>
    public PretrainedVision Vision => Model.Vision!;

    /// <summary>The image as the family's encoder reads it for <paramref name="request"/>: decoded, then the read's transforms (or the reader's).</summary>
    public ImageData Prepared(ImageReadRequest request) =>
        (request.Transforms ?? Options.Transforms).Apply(ChatImageDecoder.Decode(request.Image));

    /// <summary>The answer to <paramref name="request"/> as it is generated (the last chunk is done, with the figures).</summary>
    public IEnumerable<ChatChunk> Stream(ImageReadRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var messages = new List<ChatMessage>();
        if (request.System is { Length: > 0 } system)
        {
            messages.Add(new ChatMessage("system", system));
        }

        messages.Add(new ChatMessage("user", [request.Image, new ChatText(request.Prompt)]));
        var options = new GenerationOptions
        {
            Temperature = 1f, TopK = 1, TopP = 1f, Seed = 0,                                             // greedy
            RepeatPenalty = request.RepeatPenalty, PresencePenalty = request.PresencePenalty, FrequencyPenalty = request.FrequencyPenalty,
            RepeatLastN = request.RepeatLastN, RepeatStop = request.RepeatStop,
            NumCtx = _contextLength, NumPredict = Math.Max(1, request.MaxTokens),
        };
        return _chat.Stream(new ChatRequest(messages, Options: options)
        {
            ImageTransforms = request.Transforms, VisionOptions = request.VisionOptions,
        }, cancellationToken);
    }

    /// <summary>
    /// Reads <paramref name="request"/>: the whole answer, with each piece of it given to <paramref name="onText"/> as it
    /// is generated.
    /// </summary>
    public ImageReading Read(ImageReadRequest request, Action<string>? onText = null, CancellationToken cancellationToken = default)
    {
        var clock = Stopwatch.StartNew();
        string text = "";
        GenerationStats? stats = null;
        TimeSpan? first = null;
        string? reason = null;
        foreach (var chunk in Stream(request, cancellationToken))
        {
            if (chunk.Delta.Content.Length > 0)
            {
                first ??= clock.Elapsed;
                onText?.Invoke(chunk.Delta.Content);
            }

            if (chunk.Done)
            {
                text = chunk.Message?.Content ?? text;
                stats = chunk.Stats;
                reason = chunk.DoneReason;
            }
        }

        return new ImageReading(text.Trim(), stats, clock.Elapsed)
        {
            FirstText = first, Truncated = reason == "length" || reason is null && stats?.GeneratedTokens >= request.MaxTokens, DoneReason = reason,
        };
    }

    /// <summary>Disposes the vision encoder (not the model).</summary>
    public void Dispose() => _images.Dispose();
}
