// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Inference.Abstractions;
using Idrak.Models;
using Idrak.Nlp;

namespace Idrak.Samples.ArabicOcr;

/// <summary>A page's reading: its text and the details for JSON output.</summary>
internal sealed record PageReading(string Text, JsonObject Details);

/// <summary>A way of reading a page (the app's own choice between its two readers; see <see cref="Readers"/>).</summary>
internal interface IPageReader : IDisposable
{
    /// <summary>"lines" or "vlm".</summary>
    string Name { get; }

    /// <summary>Reads <paramref name="page"/>; <paramref name="stream"/>, when given, gets the text as it is produced.</summary>
    PageReading Read(Page page, TextWriter? stream = null);

    /// <summary>How many pages <see cref="ReadMany"/> takes at once (1: a page at a time).</summary>
    int PagesAtOnce => 1;

    /// <summary>Reads <paramref name="pages"/> (at most <see cref="PagesAtOnce"/>) in order; by default one by one.</summary>
    IReadOnlyList<PageReading> ReadMany(IReadOnlyList<Page> pages) => [.. pages.Select(p => Read(p))];
}

/// <summary>The two readers by name, and the options each takes.</summary>
internal static class Readers
{
    public const string Lines = "lines";
    public const string Vlm = "vlm";

    /// <summary>Options of the line reader.</summary>
    public static readonly string[] LineValues = ["--decoder", "--beam-width", "--batch", "--max-skew"];

    /// <summary>Flags of the line reader.</summary>
    public static readonly string[] LineFlags = ["--single-line"];

    /// <summary>Options of the vision-language reader.</summary>
    public static readonly string[] VlmValues = ["--adapter", "--image-transform", "--vision-option", "--prompt", "--system", "--max-tokens", "--context"];

    /// <summary>Flags of the vision-language reader.</summary>
    public static readonly string[] VlmFlags = ["--grayscale", "--pan-and-scan"];

    /// <summary>The reader <paramref name="name"/> made from the command's options (<c>--model</c> is each reader's model).</summary>
    public static IPageReader Create(string name, OcrContext context, string modelOption = "--model") => name switch
    {
        Lines => new LinesReader(context, context.Args.Required(modelOption, "the recognizer's folder (train writes one)")),
        Vlm => new VlmReader(context, context.Args.Required(modelOption, "a Gemma 3 vision checkpoint folder")),
        _ => throw new UsageException($"--reader is {Lines} or {Vlm}, not '{name}'."),
    };
}

/// <summary>
/// The line reader: the page cut into lines (<see cref="LineSegmenter"/>), each line read by the trained recognizer and
/// decoded by a <see cref="CtcDecoders"/> decoder; the page's text is its lines, top to bottom, one per line.
/// </summary>
internal sealed class LinesReader : IPageReader
{
    private readonly Recognizer _recognizer;
    private readonly string _decoder;
    private readonly CtcDecodeOptions _options;
    private readonly bool _singleLine;
    private readonly SegmentationSettings _segmentation;
    private readonly MeasuredBatches _batches;

    public LinesReader(OcrContext context, string folder)
    {
        var args = context.Args;
        _recognizer = Recognizer.Load(folder, context.Device);
        _decoder = args.Option("--decoder") ?? CtcDecoders.Greedy;
        if (!CtcDecoders.Names.Contains(_decoder))
        {
            throw new UsageException($"--decoder: no CTC decoder '{_decoder}' (registered: {string.Join(", ", CtcDecoders.Names)}).");
        }

        _options = new CtcDecodeOptions { BeamWidth = args.Integer("--beam-width", 10, 1) };
        _singleLine = args.Flag("--single-line");
        _segmentation = new SegmentationSettings { MaxSkew = args.Number("--max-skew", 3, 0) };
        _batches = new MeasuredBatches(context.Device, args.Option("--batch") is null ? null : args.Integer("--batch", 1, 1));
        Alphabet = _recognizer.Settings.Alphabet;
    }

    public string Name => Readers.Lines;

    /// <summary>The recognizer's characters.</summary>
    public IReadOnlyList<string> Alphabet { get; }

    /// <summary>Batches run so far: the largest, and how often one was halved for memory.</summary>
    public MeasuredBatches Batches => _batches;

    /// <summary>The texts of <paramref name="lines"/> (each one line, already cut), through the network together.</summary>
    public IReadOnlyList<string> ReadLines(IReadOnlyList<ImageData> lines) =>
        lines.Count == 0 ? [] : [.. _recognizer.Read(lines, _decoder, _options, _batches).Select(r => r.Text)];

    public PageReading Read(Page page, TextWriter? stream = null)
    {
        var reading = ReadMany([page])[0];
        stream?.WriteLine(OcrText.EndLines(reading.Text));
        return reading;
    }

    /// <summary>Single lines go 256 at a time, pages 8: enough lines for the network's batches, few pages held at once.</summary>
    public int PagesAtOnce => _singleLine ? 256 : 8;

    /// <summary>
    /// The pages decoded and their lines found in parallel (independent pages, rule 79), then every line of them through
    /// the network together (the batches of <see cref="Recognizer.Read"/>), the readings handed back page by page. A page's
    /// seconds are the call's, shared evenly.
    /// </summary>
    public IReadOnlyList<PageReading> ReadMany(IReadOnlyList<Page> pages)
    {
        var clock = Stopwatch.StartNew();
        var found = new (IReadOnlyList<TextLine> Lines, double Skew)[pages.Count];
        Parallel.For(0, pages.Count, i =>
        {
            var image = pages[i].Decode();
            found[i] = _singleLine ? ([new TextLine(1, 0, 0, image.Width, image.Height, image)], 0.0) : LineSegmenter.Find(image, _segmentation);
        });

        var all = found.SelectMany(f => f.Lines.Select(l => l.Image)).ToArray();
        var readings = all.Length == 0 ? [] : _recognizer.Read(all, _decoder, _options, _batches);
        double seconds = Math.Round(clock.Elapsed.TotalSeconds / Math.Max(1, pages.Count), 3);
        var result = new PageReading[pages.Count];
        int first = 0;
        for (int i = 0; i < pages.Count; i++)
        {
            var (lines, skew) = found[i];
            result[i] = Reading(lines, readings.Skip(first).Take(lines.Count).ToArray(), skew, seconds);
            first += lines.Count;
        }

        return result;
    }

    private PageReading Reading(IReadOnlyList<TextLine> lines, IReadOnlyList<LineReading> readings, double skew, double seconds)
    {
        string text = string.Join("\n", readings.Select(r => r.Text));
        var details = new JsonObject
        {
            ["reader"] = Name,
            ["skew_degrees"] = skew,
            ["seconds"] = seconds,
            ["lines"] = new JsonArray([.. lines.Zip(readings).Select(p => (JsonNode)new JsonObject
            {
                ["index"] = p.First.Index,
                ["box"] = new JsonObject { ["x"] = p.First.X, ["y"] = p.First.Y, ["width"] = p.First.Width, ["height"] = p.First.Height },
                ["text"] = p.Second.Text,
                ["score"] = Math.Round(p.Second.Confidence, 4),
            })]),
        };
        return new PageReading(text, details);
    }

    public void Dispose() => _recognizer.Dispose();
}

/// <summary>
/// The vision-language reader: a Gemma 3 vision checkpoint (through the Gemma 3 plug-in the app registers), with an
/// adapter tuned on images (plan 12) merged as it loads and its saved image preparation (<see cref="TuningImages"/>)
/// applied; the page and a prompt in one user turn, the answer generated greedily.
/// </summary>
internal sealed class VlmReader : IPageReader
{
    /// <summary>The prompt when neither <c>--prompt</c> nor the data gives one.</summary>
    public const string DefaultPrompt = "Extract the text of this image.";

    private readonly PretrainedModel _model;
    private readonly IVisionEncoder _encoder;
    private readonly ChatGenerator _chat;
    private readonly string? _prompt, _system;
    private readonly GenerationOptions _generation;

    public VlmReader(OcrContext context, string folder)
    {
        var args = context.Args;
        if (!Directory.Exists(folder))
        {
            throw new UsageException($"--model {folder}: no such folder (a Gemma 3 vision checkpoint: config.json, safetensors, tokenizer).");
        }

        // The adapter's preparation (how its images were tuned), the command line's over it, as idrak run does.
        string? adapter = args.Option("--adapter");
        var tuned = adapter is null ? null : TuningImages.Read(adapter) ?? TuningImages.None;
        var given = args.Options("--image-transform") is { Count: > 0 } pipelines ? ImageTransformPipeline.Parse(string.Join(",", pipelines)) : null;
        var transforms = given ?? tuned?.Pipeline ?? ImageTransformPipeline.Empty;
        if (args.Flag("--grayscale") && !transforms.Contains("grayscale"))
        {
            transforms = ImageTransformPipeline.Parse("grayscale").Then(transforms);
        }

        var vision = (tuned?.VisionOptions ?? VisionOptions.Empty).With(VisionOptions.Parse(args.Options("--vision-option")));
        if (args.Flag("--pan-and-scan"))
        {
            vision = vision.With(VisionOptions.Parse(["do_pan_and_scan=true"]));
        }

        Transforms = transforms;
        VisionOptions = vision;
        _model = PretrainedModel.Load(folder, new PretrainedOptions { Device = context.Device, MergeAdapter = adapter });
        try
        {
            var family = _model.Vision ?? throw new UsageException($"--model {folder} reads text only (no vision encoder): give a vision-language checkpoint.");
            tuned?.ThrowIfOtherFamily(family.Family, $"the adapter {adapter}");
            vision.ThrowIfUnknown(family.Family, family.VisionOptionKeys);
            _encoder = _model.CreateVisionEncoder(new VisionEncoderOptions { Device = _model.Device, VisionOptions = vision.Count > 0 ? vision : null });
            int contextLength = Math.Min(args.Integer("--context", 8192, 256), _model.MaxPositions);
            var chat = _model.CreateChat(KeyValueFormat.Float32, contextLength);
            _chat = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(_encoder, family.PromptFormat, family.Attention) { Transforms = transforms } };
        }
        catch
        {
            _model.Dispose();
            throw;
        }

        _prompt = args.Option("--prompt");
        _system = args.Option("--system");
        _generation = new GenerationOptions
        {
            Temperature = 1f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, Seed = 0,                          // greedy
            NumCtx = _model.MaxPositions, NumPredict = args.Integer("--max-tokens", 2048, 1),
        };
    }

    public string Name => Readers.Vlm;

    /// <summary>The image transforms every page goes through first.</summary>
    public ImageTransformPipeline Transforms { get; }

    /// <summary>The family's options for every page.</summary>
    public VisionOptions VisionOptions { get; }

    public PageReading Read(Page page, TextWriter? stream = null)
    {
        var clock = Stopwatch.StartNew();
        string prompt = _prompt ?? page.Prompt ?? DefaultPrompt;
        var messages = new List<ChatMessage>();
        if ((_system ?? page.System) is { Length: > 0 } system)
        {
            messages.Add(new ChatMessage("system", system));
        }

        messages.Add(new ChatMessage("user", [page.Image, new ChatText(prompt)]));
        string text = "";
        GenerationStats? stats = null;
        foreach (var chunk in _chat.Stream(new ChatRequest(messages, Options: _generation)))
        {
            if (stream is not null && chunk.Delta.Content.Length > 0)
            {
                stream.Write(chunk.Delta.Content);
                stream.Flush();
            }

            if (chunk.Done)
            {
                text = chunk.Message?.Content ?? text;
                stats = chunk.Stats;
            }
        }

        stream?.WriteLine();
        var details = new JsonObject
        {
            ["reader"] = Name,
            ["prompt"] = prompt,
            ["image_transforms"] = Transforms.ToString(),
            ["vision_options"] = VisionOptions.ToJson(),
            ["prompt_tokens"] = stats?.PromptTokens,
            ["generated_tokens"] = stats?.GeneratedTokens,
            ["seconds"] = Math.Round(clock.Elapsed.TotalSeconds, 3),
        };
        return new PageReading(text.Trim(), details);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _model.Dispose();
    }
}
