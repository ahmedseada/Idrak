// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Text;

namespace Idrak.Text.Generation;

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

    /// <summary>
    /// Keep the decoding context, the sampler and the recorded decoding step from one read to the next (see
    /// <see cref="TextGenerator.ReuseDecoding"/>): a read of the same size and sampling takes them, reset, instead of
    /// allocating and clearing the attention cache, making a sampler and recording the step again; every read still starts
    /// from its own first token, so its answer is the same. Off by default: the context stays allocated on the device
    /// between reads (until the reader is disposed).
    /// </summary>
    public bool ReuseDecoding { get; init; }

    /// <summary>
    /// The arithmetic of the vision encoder's matrix products (<see cref="ChatImages.EncoderPrecision"/>); null: the
    /// caller's. <see cref="MatMulPrecision.BFloat16"/> runs the tower on tensor cores where the device has them (a page's
    /// encoding several times faster), its features within bfloat16's rounding of float32: compare a model's answers
    /// on the same pages before reading with it.
    /// </summary>
    public MatMulPrecision? EncoderPrecision { get; init; }

    /// <summary>
    /// The arithmetic of the prompt's pass (<see cref="TextGenerator.PrefillPrecision"/>: the page's image tokens and the
    /// instruction through the language model); null: the caller's. <see cref="MatMulPrecision.BFloat16"/> runs the
    /// prefill's products on tensor cores where the device has them; the answer's decoding steps are unaffected. Compare
    /// a model's answers on the same pages before reading with it.
    /// </summary>
    public MatMulPrecision? PrefillPrecision { get; init; }

    /// <summary>Makes the vision encoder instead of the model's own (tests feed reference features with it).</summary>
    public Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? EncoderFactory { get; init; }

    /// <summary>
    /// What <see cref="ImageReader.ReadBatch"/> does when the device has no room for the pages together:
    /// <see cref="BatchFit.Split"/> (the default) reads them in halves until they fit, <see cref="BatchFit.Fail"/> raises
    /// the device's <see cref="ResourceLimitExceededException"/>.
    /// </summary>
    public BatchFit BatchFit { get; init; } = BatchFit.Split;

    /// <summary>The most pages <see cref="ImageReader.ReadBatch"/> reads together (null: as many as asked, while they fit).</summary>
    public int? MaxBatch { get; init; }

    /// <summary>
    /// Whether a batch that had to be split sets where later batches start (the pages that fit together), so the device's
    /// shortage is paid for once (the default); false: every batch starts with all its pages.
    /// </summary>
    public bool RememberBatchFit { get; init; } = true;

    /// <summary>
    /// Whether the reader measures its first read for suggested settings (<see cref="ImageReader.Suggestions"/>: the
    /// device's peak during it above what it holds before and after, the peak reset for the read, and what it has left): null (the default) when telemetry listens at
    /// <see cref="TelemetryLevel.Suggestions"/> (the suggestions are published there, each once), true always, false never.
    /// </summary>
    public bool? Suggest { get; init; }

    /// <summary>
    /// The reads before the one measured for the suggestions (1 by default: the first read runs cold work a page does not
    /// need again, kernels compiled and tuned, weights loaded on the way, and its peak is not a page's).
    /// </summary>
    public int SuggestAfterReads { get; init; } = 1;

    /// <summary>
    /// The share of the device's memory the suggested settings leave free beyond the device's own reserve
    /// (<see cref="ComputeResources.GpuMemoryReserve"/>, which what the device reports as available already leaves): 0
    /// by default; more for what a read did not show (other programs, longer answers).
    /// </summary>
    public double SuggestionHeadroom { get; init; }
}

/// <summary>What a batch of pages does when the device has no room for them together (<see cref="ImageReaderOptions.BatchFit"/>).</summary>
public enum BatchFit
{
    /// <summary>Read them in halves (and halves of those) until they fit; each reading says how many it was read with.</summary>
    Split = 0,

    /// <summary>Raise the device's out-of-memory error.</summary>
    Fail = 1,
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
    /// Sample each token from the model's probabilities (shaped by <see cref="Temperature"/>, <see cref="TopK"/>,
    /// <see cref="TopP"/> and <see cref="MinP"/>) instead of taking the most likely one: answers vary from read to read
    /// unless <see cref="Seed"/> is set. Off by default (greedy: the same answer every time).
    /// </summary>
    public bool Sample { get; init; }

    /// <summary>With <see cref="Sample"/>: the softmax temperature (above 0; below 1 more predictable, above 1 more varied).</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>With <see cref="Sample"/>: sample only among the k most likely tokens (0 = all).</summary>
    public int TopK { get; init; }

    /// <summary>With <see cref="Sample"/>: the smallest set of likely tokens holding this probability (1 = off).</summary>
    public float TopP { get; init; } = 1f;

    /// <summary>With <see cref="Sample"/>: drop tokens less than this fraction as likely as the best one (0 = off).</summary>
    public float MinP { get; init; }

    /// <summary>With <see cref="Sample"/>: the random seed, so a read can be repeated exactly (null = a different one each read).</summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Encode the prompt with the tokenizer's special tokens added, as a transformers pipeline that renders the template
    /// to text and then calls the processor on it does (see <see cref="ChatRequest.AddSpecialTokens"/>). Off by default.
    /// </summary>
    public bool AddSpecialTokens { get; init; }

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

    /// <summary>
    /// How many pages were read together with this one, itself included (<see cref="ImageReader.ReadBatch"/>; fewer than
    /// asked when the device had no room for them all, see <see cref="ImageReaderOptions.BatchFit"/>); 1 for a page read alone.
    /// </summary>
    public int BatchSize { get; init; } = 1;
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
    private readonly ChatImages _reading;                                   // the model's images with the reader's encoder precision
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
            chat.Generator.ReuseDecoding = options.ReuseDecoding;
            chat.Generator.PrefillPrecision = options.PrefillPrecision;
            var images = _images.Images;
            _reading = new ChatImages(images.Encoder, images.Format, images.Attention)
            {
                Decode = images.Decode, Transforms = images.Transforms, Owner = images.Owner, EncoderPrecision = options.EncoderPrecision,
            };
            _chat = new ChatGenerator(chat.Generator, chat.Template) { Images = _reading };
            Publish();                                                      // what the device alone decides
        }
        catch
        {
            _images.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Settings worth changing for this reader (every advisor of <see cref="SettingAdvisors"/>), from what its device
    /// reports and, once a read was measured (<see cref="ImageReaderOptions.Suggest"/>), from that read's figures; none
    /// where a figure is missing.
    /// </summary>
    public IReadOnlyList<SettingSuggestion> Suggestions() => SettingAdvisors.Suggest(Situation);

    private AdviceSituation Situation => new(Model.Device, this, _measured) { Headroom = Options.SuggestionHeadroom };

    private AdviceMeasure? _measured;
    private int _reads;                                                     // reads seen while waiting to measure one
    private readonly HashSet<(string Setting, string Value)> _published = [];

    private bool Suggesting => Options.Suggest ?? Telemetry.IsEnabled(TelemetryLevel.Suggestions);

    // Each suggestion once, to the hooks listening at TelemetryLevel.Suggestions.
    private void Publish() => SettingAdvisors.Publish(Situation, _published);

    // The first read after SuggestAfterReads, measured for the suggestions (when wanted and the device reports its free
    // memory): the device's peak above what it held before and after, and what it has left after.
    private T Measured<T>(Func<T> read, Func<T, (int Pages, int Positions)> figures)
    {
        var backend = Model.Device.Backend;
        if (_measured is not null || !Suggesting || _reads++ < Options.SuggestAfterReads || backend.AvailableMemory() is null)
        {
            return read();
        }

        long start = backend.GetMemoryUsage().InUse;
        backend.ResetPeakMemoryUsage();
        var result = read();
        var (pages, positions) = figures(result);

        // What the read needed for its pages: its peak above what is held after it as well as before (weights it loaded
        // on the way, as the encoder's on a first read, stay and are not a page's).
        var after = backend.GetMemoryUsage();
        _measured = new AdviceMeasure(pages, positions, Math.Max(0, after.Peak - Math.Max(start, after.InUse)), backend.AvailableMemory() ?? 0);
        Publish();
        return result;
    }

    // The chat the pages are read with (the tests look at its generator).
    internal ChatGenerator Chat => _chat;

    /// <summary>The model read with.</summary>
    public PretrainedModel Model { get; }

    /// <summary>The reader's settings.</summary>
    public ImageReaderOptions Options { get; }

    /// <summary>The model's vision family.</summary>
    public IPretrainedVision Vision => Model.Vision!;

    /// <summary>
    /// The image as the family's encoder reads it for <paramref name="request"/>: decoded, then the read's transforms (or
    /// the reader's). Kept for the read that follows (<see cref="ChatImages.Read"/>), so showing a page and reading it
    /// decodes and transforms it once; the pixels are shared and must not be changed.
    /// </summary>
    public ImageData Prepared(ImageReadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return _reading.Read(request.Image, new ChatRequest([]) { ImageTransforms = request.Transforms ?? Options.Transforms });
    }

    /// <summary>
    /// Keeps <paramref name="prepared"/> as <paramref name="request"/>'s page for the read that follows
    /// (<see cref="ChatImages.Keep"/>): the caller decoded the page (the codecs, then its EXIF orientation:
    /// <c>ChatImageDecoder.Decode</c>) and ran the read's transforms (or the reader's) on it already, to show it before
    /// reading, so the read decodes and transforms nothing. <see cref="Prepared"/> gives it back too. The pixels are
    /// shared with the read and must not be changed.
    /// </summary>
    public void Keep(ImageReadRequest request, ImageData prepared)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(prepared);
        _reading.Keep(request.Image, prepared, new ChatRequest([]) { ImageTransforms = request.Transforms ?? Options.Transforms });
    }

    /// <summary>The answer to <paramref name="request"/> as it is generated (the last chunk is done, with the figures).</summary>
    public IEnumerable<ChatChunk> Stream(ImageReadRequest request, CancellationToken cancellationToken = default) =>
        _chat.Stream(ChatRequestFor(request), cancellationToken);

    /// <summary>
    /// Reads several pages together (<see cref="ChatGenerator.StreamBatch"/>): pages read with one instruction and the same
    /// settings go through the encoder in one call and through the language model as one batch, one pass a token for all
    /// of them, so a device that is bound by reading its weights reads several pages in about the time of one. Each answer
    /// is the page's own (stopped by its end, its stop sequences or a loop), within the batch's rounding;
    /// <paramref name="onText"/> gets each page's pieces with its index. Pages that cannot share a batch are read one by
    /// one. Each reading's <see cref="ImageReading.Elapsed"/> is the time until its answer was done.
    /// </summary>
    /// <remarks>
    /// When the device has no room for the pages together, they are read in halves until they fit (see
    /// <see cref="ImageReaderOptions.BatchFit"/>, <see cref="ImageReaderOptions.MaxBatch"/> and
    /// <see cref="ImageReaderOptions.RememberBatchFit"/>); each reading's <see cref="ImageReading.BatchSize"/> says how many
    /// it was read with. A batch is split only before any of its text reached <paramref name="onText"/>.
    /// </remarks>
    public IReadOnlyList<ImageReading> ReadBatch(IReadOnlyList<ImageReadRequest> requests, Action<int, string>? onText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        return requests.Count == 0 ? [] : Measured(() => ReadBatchCore(requests, onText, cancellationToken),
            read => (read.Max(r => r.BatchSize), read.Max(r => r.Stats?.PromptTokens ?? 0) + requests.Max(r => r.MaxTokens)));
    }

    private IReadOnlyList<ImageReading> ReadBatchCore(IReadOnlyList<ImageReadRequest> requests, Action<int, string>? onText, CancellationToken cancellationToken)
    {
        var clock = Stopwatch.StartNew();
        var results = new ImageReading?[requests.Count];
        var firsts = new TimeSpan?[requests.Count];
        int size = Math.Max(1, Math.Min(requests.Count, Math.Min(Options.MaxBatch ?? int.MaxValue, Options.RememberBatchFit ? _batchFit : int.MaxValue)));
        for (int start = 0; start < requests.Count;)
        {
            int count = Math.Min(size, requests.Count - start);
            bool streamed = false;
            try
            {
                ReadGroup(start, count);
                start += count;
            }
            catch (ResourceLimitExceededException) when (Options.BatchFit == BatchFit.Split && count > 1 && !streamed)
            {
                // No room for these pages together: what the attempt held goes back, and they are read in halves.
                ComputeResources.ReleaseCachedMemory(Model.Device);
                size = (count + 1) / 2;
                if (Options.RememberBatchFit)
                {
                    _batchFit = size;
                }
            }

            void ReadGroup(int first, int pages)
            {
                var group = new ChatRequest[pages];
                for (int i = 0; i < pages; i++)
                {
                    group[i] = ChatRequestFor(requests[first + i]);
                }

                foreach (var (offset, chunk) in _chat.StreamBatch(group, cancellationToken))
                {
                    int index = first + offset;
                    if (chunk.Delta.Content.Length > 0)
                    {
                        firsts[index] ??= clock.Elapsed;
                        streamed = onText is not null;
                        onText?.Invoke(index, chunk.Delta.Content);
                    }

                    if (chunk.Done)
                    {
                        var stats = chunk.Stats;
                        results[index] = new ImageReading((chunk.Message?.Content ?? "").Trim(), stats, clock.Elapsed)
                        {
                            FirstText = firsts[index],
                            Truncated = chunk.DoneReason == "length" || chunk.DoneReason is null && stats?.GeneratedTokens >= requests[index].MaxTokens,
                            DoneReason = chunk.DoneReason,
                            BatchSize = pages,
                        };
                    }
                }
            }
        }

        return [.. results.Select((r, i) => r ?? throw new InvalidOperationException($"Page {i} ended without an answer."))];
    }

    // The most pages a batch found room for (ImageReaderOptions.RememberBatchFit): where later batches start.
    private int _batchFit = int.MaxValue;

    // The chat request of a page read: the system text, the page and its instruction in one user turn, greedy unless sampled.
    private ChatRequest ChatRequestFor(ImageReadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var messages = new List<ChatMessage>();
        if (request.System is { Length: > 0 } system)
        {
            messages.Add(new ChatMessage("system", system));
        }

        messages.Add(new ChatMessage("user", [request.Image, new ChatText(request.Prompt)]));
        if (request.Sample && (!(request.Temperature > 0f) || request.TopK < 0 || !(request.TopP is > 0f and <= 1f) || !(request.MinP is >= 0f and < 1f)))
        {
            throw new ArgumentException($"Sampling takes a temperature above 0, top-k of 0 or more, top-p in (0, 1] and min-p in [0, 1); "
                + $"got {request.Temperature}, {request.TopK}, {request.TopP}, {request.MinP}.", nameof(request));
        }

        var options = new GenerationOptions
        {
            Temperature = request.Sample ? request.Temperature : 1f, TopK = request.Sample ? request.TopK : 1,   // greedy: the most likely token
            TopP = request.Sample ? request.TopP : 1f, MinP = request.Sample ? request.MinP : 0f, Seed = request.Sample ? request.Seed : 0,
            RepeatPenalty = request.RepeatPenalty, PresencePenalty = request.PresencePenalty, FrequencyPenalty = request.FrequencyPenalty,
            RepeatLastN = request.RepeatLastN, RepeatStop = request.RepeatStop,
            NumCtx = _contextLength, NumPredict = Math.Max(1, request.MaxTokens),
        };
        return new ChatRequest(messages, Options: options)
        {
            ImageTransforms = request.Transforms, VisionOptions = request.VisionOptions, AddSpecialTokens = request.AddSpecialTokens,
        };
    }

    /// <summary>
    /// Reads <paramref name="request"/>: the whole answer, with each piece of it given to <paramref name="onText"/> as it
    /// is generated.
    /// </summary>
    public ImageReading Read(ImageReadRequest request, Action<string>? onText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Measured(() => ReadCore(request, onText, cancellationToken), read => (1, (read.Stats?.PromptTokens ?? 0) + request.MaxTokens));
    }

    private ImageReading ReadCore(ImageReadRequest request, Action<string>? onText, CancellationToken cancellationToken)
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

    /// <summary>Disposes the vision encoder and what the reads kept on the device (attention cache, decoding context; not the model).</summary>
    public void Dispose()
    {
        _chat.Generator.ReleaseCache();
        _images.Dispose();
    }
}
