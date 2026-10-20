// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Abstraction.Generation;

namespace Idrak.Text;

/// <summary>Settings for <see cref="ConversationTuning.Prepare"/>: the data, its images and evaluation, and the tuning.</summary>
public sealed record ConversationTuningOptions
{
    /// <summary>A data file of evaluation conversations (any <see cref="TuningDataFormats"/> format), or null.</summary>
    public string? Evaluation { get; init; }

    /// <summary>Without <see cref="Evaluation"/>: the share of the training conversations held out for evaluation (seeded by <see cref="Seed"/>; 0: none).</summary>
    public double EvaluationFraction { get; init; }

    /// <summary>The seed of the held-out share.</summary>
    public int Seed { get; init; }

    /// <summary>The data files' format (<see cref="TuningDataFormats"/>); null: told from each file's first record.</summary>
    public string? DataFormat { get; init; }

    /// <summary>Where relative image paths are found (a folder or a zip); null: each data file's folder.</summary>
    public string? Images { get; init; }

    /// <summary>
    /// Where each training data file's images are, by the file's path as given, over <see cref="Images"/> (a null root:
    /// the file's own folder): real pages from a zip beside pages drawn into a folder (<see cref="SyntheticPages"/>).
    /// </summary>
    public IReadOnlyDictionary<string, string?>? ImagesByData { get; init; }

    /// <summary>Where the evaluation file's images are; null: <see cref="Images"/>.</summary>
    public string? EvaluationImages { get; init; }

    /// <summary>The system prompt added to conversations without one (written to the adapter's <see cref="TuningManifest"/>), or null.</summary>
    public string? System { get; init; }

    /// <summary>Train on the first this many conversations of the data (after the held-out share is taken); 0: all.</summary>
    public int MaxConversations { get; init; }

    /// <summary>How every image is prepared (saved beside the adapter as <see cref="TuningImages.FileName"/>).</summary>
    public TuningImages Preparation { get; init; } = TuningImages.None;

    /// <summary>The vision parts that train beside the adapters (<see cref="VisionTuningParts"/>); none by default.</summary>
    public IReadOnlyList<string> VisionParts { get; init; } = [];

    /// <summary>Where relative audio paths are found (a folder or a zip); null: where the images are.</summary>
    public string? Audios { get; init; }

    /// <summary>The "transcripts" format's prompt (<see cref="TuningDataOptions.Prompt"/>); null: "Transcribe.".</summary>
    public string? Prompt { get; init; }

    /// <summary>
    /// The parts of the model's soft parts (<see cref="PretrainedModel.Parts"/>: "audio-encoder", "audio-projector") that
    /// train beside the adapters, saved with them; none by default (the parts read their features frozen).
    /// </summary>
    public IReadOnlyList<string> TrainedParts { get; init; } = [];

    /// <summary>Where image features are kept between steps and epochs (<see cref="FeatureCaches"/>).</summary>
    public string FeatureCache { get; init; } = FeatureCaches.Memory;

    /// <summary>The disk feature cache's folder (null: the library's).</summary>
    public string? FeatureFolder { get; init; }

    /// <summary>A metric of <see cref="TuningMetrics"/> scoring answers generated on the evaluation conversations while training, or null.</summary>
    public string? Metric { get; init; }

    /// <summary>Score answers every this many optimizer steps (0: each epoch).</summary>
    public int MetricEvery { get; init; }

    /// <summary>Score the first this many evaluation conversations (0: all).</summary>
    public int MetricSamples { get; init; }

    /// <summary>The most tokens an answer scored by <see cref="Metric"/> may take.</summary>
    public int MaxNewTokens { get; init; } = 512;

    /// <summary>
    /// End a scored answer when one phrase repeats this many times in a row (<see cref="GenerationOptions.RepeatStop"/>;
    /// 0, the default: off). A model that loops then costs a few repeats, not <see cref="MaxNewTokens"/>; the answer up to
    /// the stop is scored.
    /// </summary>
    public int MetricRepeatStop { get; init; }

    /// <summary>
    /// The text of a scored answer and its reference that <see cref="Metric"/> compares (<see cref="TuningAnswerScorer.AnswerText"/>):
    /// null (the default) for both as they are, or a name in <see cref="AnswerTexts"/>, such as <see cref="AnswerTexts.Values"/>
    /// for the values of JSON answers.
    /// </summary>
    public string? MetricText { get; init; }

    /// <summary>
    /// The evaluation loss over the first this many evaluation conversations (0, the default: all of them); the metric's
    /// own <see cref="MetricSamples"/> are counted apart. Each evaluation of the loss is a forward pass per conversation.
    /// </summary>
    public int EvaluationSamples { get; init; }

    /// <summary>The reasoning mode of scored answers (null: the template's default).</summary>
    public bool? Think { get; init; }

    /// <summary>The fine-tuning settings (<see cref="FineTuningOptions.Vision"/> and <see cref="FineTuningOptions.Answers"/> are set by the preparation).</summary>
    public FineTuningOptions Tuning { get; init; } = new();
}

/// <summary>A stage of <see cref="ConversationTuning.Prepare"/> as it goes ("tokenizing training": conversations done of the total).</summary>
public sealed record TuningStage(string Name, int Done, int Total, TimeSpan Elapsed);

/// <summary>How a <see cref="ConversationTuning.Train"/> run ended.</summary>
/// <param name="Folder">The adapter folder (full path).</param>
/// <param name="Stopped">True when it was stopped before the end (the adapter so far was saved).</param>
/// <param name="Elapsed">The training time.</param>
/// <param name="EvaluationLosses">The evaluation losses, in order.</param>
/// <param name="Last">The last step's report, or null when none ran.</param>
/// <param name="Answers">The last answer scores, or null.</param>
public sealed record ConversationTuningResult(string Folder, bool Stopped, TimeSpan Elapsed, IReadOnlyList<float> EvaluationLosses, FineTuningProgress? Last,
    TuningAnswerReport? Answers);

/// <summary>
/// Fine-tuning a chat model on conversation files, with their images when the model reads images: the data read through
/// <see cref="TuningDataFormats"/> (messages or ShareGPT, images from a folder or a zip), an evaluation file or a held-out
/// share, the image side from the model's own vision family (<see cref="TuningVision"/>, its features kept in a
/// <see cref="FeatureCaches"/> cache), the conversations tokenized with the model's chat template
/// (<see cref="ChatTranscriptEncoder"/>), answers scored by a <see cref="TuningMetrics"/> metric while training
/// (<see cref="TuningAnswerScorer"/>), then <see cref="FineTuner"/>'s training. <see cref="Train"/> saves the adapter, the
/// trained vision parts, the image preparation and the <see cref="TuningManifest"/>, also when it is stopped.
/// <para>
/// Between <see cref="Prepare"/> and <see cref="Train"/> a caller may change <see cref="TrainingSequences"/> (balancing
/// answers) and <see cref="Tuning"/>. Disposing it disposes the image side and the feature cache, not the model.
/// </para>
/// </summary>
/// <example>
/// <code>
/// using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = Device.Cuda(), BFloat16 = true });
/// using var tuning = ConversationTuning.Prepare(model, ["train.json"], new ConversationTuningOptions
/// {
///     Evaluation = "val.json", Images = "images.zip", Preparation = TuningImages.Parse("max_width=1024"), Metric = TuningMetrics.CharacterErrorRate,
/// });
/// var result = tuning.Train("adapters/legal", baseModel: "google/gemma-3-4b-it", progress: new Progress&lt;FineTuningProgress&gt;(p => Console.WriteLine(p.Loss)));
/// </code>
/// </example>
public sealed class ConversationTuning : IDisposable
{
    private readonly IFeatureCache? _cache;

    private ConversationTuning(PretrainedModel model, ConversationTuningOptions options, TuningVision? vision, IFeatureCache? cache, ChatTranscriptEncoder encoder)
    {
        Model = model;
        Options = options;
        Vision = vision;
        _cache = cache;
        Encoder = encoder;
    }

    /// <summary>The model being tuned (not owned).</summary>
    public PretrainedModel Model { get; }

    /// <summary>The settings it was prepared with.</summary>
    public ConversationTuningOptions Options { get; }

    /// <summary>The image side, or null for a model without a vision part.</summary>
    public TuningVision? Vision { get; }

    // The model's other soft parts (an audio part) as sources, and the parts each trains.
    private List<(ITrainableSoftParts Source, IReadOnlyList<string> Trained)> _sources = [];

    /// <summary>The sources of the model's soft parts other than images (its audio part), one per kind; none for most models.</summary>
    public IReadOnlyList<ISoftPartSource> Parts => [.. _sources.Select(s => s.Source)];

    /// <summary>The model's chat template as training sequences.</summary>
    public ChatTranscriptEncoder Encoder { get; }

    /// <summary>The training conversations.</summary>
    public IReadOnlyList<ChatTranscript> TrainingConversations { get; private set; } = [];

    /// <summary>The evaluation conversations, or null.</summary>
    public IReadOnlyList<ChatTranscript>? EvaluationConversations { get; private set; }

    /// <summary>The training sequences (a caller may replace them before <see cref="Train"/>, e.g. balanced).</summary>
    public IReadOnlyList<TrainingSequence> TrainingSequences { get; set; } = [];

    /// <summary>The evaluation sequences, or null.</summary>
    public IReadOnlyList<TrainingSequence>? EvaluationSequences { get; private set; }

    /// <summary>The answer scorer (when <see cref="ConversationTuningOptions.Metric"/> is set), or null.</summary>
    public TuningAnswerScorer? Scorer { get; private set; }

    /// <summary>The settings <see cref="Train"/> uses (the options' tuning with the image side, the scorer and a length within the model's context).</summary>
    public FineTuningOptions Tuning { get; set; } = new();

    /// <summary>
    /// Reads the data files, builds the image side, holds out the evaluation share and tokenizes the conversations.
    /// </summary>
    /// <param name="model">The model to tune (adapters already loaded continue training; others are added by <see cref="Train"/>).</param>
    /// <param name="data">The training data files.</param>
    /// <param name="options">The settings (defaults when null).</param>
    /// <param name="progress">Receives each stage as it goes (tokenizing).</param>
    /// <param name="log">Receives a line per finished stage (conversations read, sequences, images).</param>
    /// <param name="cancellationToken">Stops between conversations.</param>
    /// <exception cref="ArgumentException">No data file, or a metric without evaluation conversations.</exception>
    /// <exception cref="InvalidOperationException">Nothing to train on, a model without a chat template or tokenizer, or image options for a model without a vision part.</exception>
    public static ConversationTuning Prepare(PretrainedModel model, IReadOnlyList<string> data, ConversationTuningOptions? options = null, IProgress<TuningStage>? progress = null,
        Action<string>? log = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(data);
        if (data.Count == 0)
        {
            throw new ArgumentException("Give at least one training data file.", nameof(data));
        }

        options ??= new ConversationTuningOptions();
        var template = model.JinjaTemplate ?? throw new InvalidOperationException("The model has no chat template.");
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer.");
        var tuning = options.Tuning;
        if (tuning.MaxLength >= model.MaxPositions)
        {
            log?.Invoke($"max length {tuning.MaxLength} is beyond the model's context of {model.MaxPositions} positions; using {model.MaxPositions - 1}");
            tuning = tuning with { MaxLength = model.MaxPositions - 1 };
        }

        TuningVision? vision = null;
        IFeatureCache? cache = null;
        if (model.Vision is not null)
        {
            cache = FeatureCaches.Create(options.FeatureCache, new FeatureCacheOptions { Folder = options.FeatureFolder });
            try
            {
                vision = TuningVision.Create(model, options.Preparation with { Family = null }, options.VisionParts, cache);
            }
            catch
            {
                cache.Dispose();
                throw;
            }

            log?.Invoke($"images: {vision}");
        }
        else if (!options.Preparation.IsEmpty || options.VisionParts.Count > 0)
        {
            throw new InvalidOperationException("The model has no vision part (a text-only checkpoint): an image preparation and vision parts are for a vision-language model.");
        }

        // The parts stored beside the decoder (an audio part), opened as sources: frozen, or training the parts asked for.
        var sources = new List<(ITrainableSoftParts Source, IReadOnlyList<string> Trained)>();
        try
        {
            foreach (string kind in model.Parts)
            {
                var source = model.OpenPart(kind, model.Device);
                var trained = source.TrainableParts.Where(p => options.TrainedParts.Contains(p, StringComparer.OrdinalIgnoreCase)).ToList();
                sources.Add((source, trained));
                source.Train(trained);
                source.SetTraining(trained.Count > 0);
                log?.Invoke($"{kind}: a soft-part source{(trained.Count > 0 ? $", training {string.Join(", ", trained)}" : " (frozen)")}");
            }

            var unknown = options.TrainedParts.Where(p => !sources.Any(s => s.Source.TrainableParts.Contains(p, StringComparer.OrdinalIgnoreCase))).ToList();
            if (unknown.Count > 0)
            {
                throw new ArgumentException($"The model's parts offer no '{string.Join("', '", unknown)}' "
                    + $"({(sources.Count == 0 ? "it has none" : string.Join(", ", sources.SelectMany(s => s.Source.TrainableParts)))}).", nameof(options));
            }
        }
        catch
        {
            sources.ForEach(s => s.Source.Dispose());
            vision?.Dispose();
            cache?.Dispose();
            throw;
        }

        var run = new ConversationTuning(model, options, vision, cache, new ChatTranscriptEncoder(template, tokenizer) { Vision = vision, Parts = [.. sources.Select(s => s.Source)] })
        {
            _sources = sources,
        };
        try
        {
            run.Load(data, tuning, progress, log, cancellationToken);
            return run;
        }
        catch
        {
            run.Dispose();
            throw;
        }
    }

    /// <summary>The mean loss per trained token of the evaluation sequences, or null when there are none.</summary>
    public float? EvaluationLoss() => EvaluationSequences is { Count: > 0 } evaluation
        ? FineTuner.Evaluate(Model, evaluation, Tuning.BatchTokens, Tuning.LossChunkRows, vision: Vision, parts: Parts)
        : null;

    /// <summary>The scorer's answers with the model as it is now (before training, say), or null without a scorer.</summary>
    public TuningAnswerReport? ScoreAnswers(IProgress<TuningAnswer>? progress = null, CancellationToken cancellationToken = default) =>
        Scorer?.Score(Model, Vision, progress, cancellationToken);

    /// <summary>
    /// Trains the adapters (and the vision parts asked for) and writes them to <paramref name="folder"/> with the image
    /// preparation and the <see cref="TuningManifest"/>. Cancelling stops after the current step and saves what was
    /// trained so far (<see cref="ConversationTuningResult.Stopped"/>).
    /// </summary>
    /// <param name="folder">The adapter folder (created).</param>
    /// <param name="baseModel">The base model as it is named for loading again (a Hugging Face id or a folder), for the manifest.</param>
    /// <param name="progress">Receives one report per optimizer step (with the evaluation loss and the answer scores when they were taken).</param>
    /// <param name="cancellationToken">Stops after the current step.</param>
    /// <param name="trace">Receives the tuner's lines (each batch, graph recording, checkpoints).</param>
    public ConversationTuningResult Train(string folder, string baseModel, IProgress<FineTuningProgress>? progress = null, CancellationToken cancellationToken = default,
        Action<string>? trace = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseModel);
        var clock = Stopwatch.StartNew();
        FineTuningProgress? last = null;
        TuningAnswerReport? answers = null;
        var losses = new List<float>();
        var reports = new Relay<FineTuningProgress>(p =>
        {
            last = p;
            answers = p.Answers ?? answers;
            if (p.EvaluationLoss is { } loss)
            {
                losses.Add(loss);
            }

            progress?.Report(p);
        });

        bool stopped = false;
        try
        {
            FineTuner.Train(Model, TrainingSequences, EvaluationSequences, Tuning, folder, reports, cancellationToken, trace);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            stopped = true;
            Save(folder);
        }

        new TuningManifest { BaseModel = baseModel, System = Options.System, MaxLength = Tuning.MaxLength }.Save(folder);
        return new ConversationTuningResult(Path.GetFullPath(folder), stopped, clock.Elapsed, losses, last, answers);
    }

    /// <summary>Writes the adapters as they are now, the trained vision parts and the image preparation (when the training data has images) to <paramref name="folder"/>.</summary>
    public void Save(string folder)
    {
        if (Vision is { Parts.Count: > 0 })
        {
            Model.KeepTrainedVision(Vision.Encoder, Vision.Parts);
        }

        foreach (var (source, trained) in _sources.Where(s => s.Trained.Count > 0))
        {
            Model.KeepTrainedParts(source.Export(trained));                   // saved with the adapters (modules_to_save)
        }

        Model.SaveAdapter(folder);
        if (Vision is not null && TrainingSequences.Any(s => s.Images.Count > 0))
        {
            (Vision.Images with { Family = Vision.Vision.Family }).Save(folder);
        }
    }

    /// <summary>Disposes the image side, the part sources and the feature cache (not the model).</summary>
    public void Dispose()
    {
        _sources.ForEach(s => s.Source.Dispose());
        _sources.Clear();
        Vision?.Dispose();
        _cache?.Dispose();
    }

    // The conversations read, the held-out share taken, the sequences tokenized, the scorer made.
    private void Load(IReadOnlyList<string> data, FineTuningOptions tuning, IProgress<TuningStage>? progress, Action<string>? log, CancellationToken cancellationToken)
    {
        // Every record read and checked first, its images left unread; the held-out share and the first N chosen by the
        // conversations' places; then only the chosen conversations' images read and hashed (rule 75: a page left out is
        // never inflated). The choice is the one made over the conversations themselves before.
        using var trainingData = Read(data, Options.Images, "training", log, cancellationToken, Options.ImagesByData);
        using var evaluationData = Options.Evaluation is { } file ? Read([file], Options.EvaluationImages ?? Options.Images, "evaluation", log, cancellationToken) : null;
        var trainAt = Enumerable.Range(0, trainingData.Count).ToList();
        List<int>? heldAt = null;
        if (evaluationData is null && Options.EvaluationFraction > 0 && trainAt.Count > 1)
        {
            // A seeded share of the conversations held out (at least one, never all).
            int held = Math.Clamp((int)Math.Round(trainAt.Count * Options.EvaluationFraction), 1, trainAt.Count - 1);
            var order = Enumerable.Range(0, trainAt.Count).ToArray();
            new Random(Options.Seed).Shuffle(order);
            var heldOut = order.Take(held).ToHashSet();
            heldAt = [.. heldOut.Order()];
            trainAt = [.. trainAt.Where(i => !heldOut.Contains(i))];
            log?.Invoke($"evaluation: {held:N0} conversations held out ({Options.EvaluationFraction:P1})");
        }

        if (Options.MaxConversations > 0 && trainAt.Count > Options.MaxConversations)
        {
            log?.Invoke($"training on the first {Options.MaxConversations:N0} of {trainAt.Count:N0} conversations");
            trainAt = trainAt[..Options.MaxConversations];
        }

        var train = trainingData.Take(trainAt, Options.System, "training", log, cancellationToken);
        var evaluation = evaluationData is not null ? evaluationData.Take(Enumerable.Range(0, evaluationData.Count), Options.System, "evaluation", log, cancellationToken)
            : heldAt is not null ? trainingData.Take(heldAt, Options.System, "evaluation", log, cancellationToken)
            : null;
        TrainingConversations = train;
        EvaluationConversations = evaluation;
        TrainingSequences = Encode(train, tuning.MaxLength, "training", progress, log, cancellationToken);
        EvaluationSequences = evaluation is null ? null
            : Encode(Options.EvaluationSamples > 0 && evaluation.Count > Options.EvaluationSamples ? evaluation[..Options.EvaluationSamples] : evaluation,
                tuning.MaxLength, "evaluation", progress, log, cancellationToken);
        if (TrainingSequences.Count == 0)
        {
            throw new InvalidOperationException("Nothing to train on: no conversation with an assistant turn.");
        }

        if (Options.Metric is { } metric)
        {
            if (evaluation is not { Count: > 0 })
            {
                throw new ArgumentException($"The metric {metric} scores answers on evaluation conversations: give an evaluation file or an evaluation fraction.");
            }

            if (Options.MetricText is { } text)
            {
                _ = AnswerTexts.Get(text);                                    // not registered: said now, not after an epoch
            }

            Scorer = new TuningAnswerScorer(evaluation, metric, Options.MetricSamples)
            {
                AnswerText = Options.MetricText,
                Every = Options.MetricEvery, MaxNewTokens = Options.MaxNewTokens, ContextLength = Math.Min(Model.MaxPositions, tuning.MaxLength + Options.MaxNewTokens),
                Think = Options.Think, RepeatStop = Options.MetricRepeatStop, Parts = Parts,
            };
        }

        Tuning = tuning with { Vision = Vision, Answers = Scorer, Parts = Parts };
    }

    // Conversations from data files through their format, their images not read yet (ITuningDataFormat.ReadPending);
    // each file's images under `byFile`'s root for it, else `images`.
    private PendingData Read(IEnumerable<string> files, string? images, string what, Action<string>? log, CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string?>? byFile = null)
    {
        var watch = Stopwatch.StartNew();
        var read = new PendingData();
        try
        {
            foreach (string file in files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var options = new TuningDataOptions { Images = byFile is not null && byFile.TryGetValue(file, out var root) ? root : images, Audios = Options.Audios, Prompt = Options.Prompt };
                var format = Options.DataFormat is { } name && !name.Equals("auto", StringComparison.OrdinalIgnoreCase) ? TuningDataFormats.Get(name) : TuningDataFormats.Detect(file);
                var pending = format.ReadPending(file, options);
                read.Add(pending);
                log?.Invoke($"{what}: {pending.Count:N0} conversations ({format.Name}) from {file}, {pending.Images:N0} images ({watch.Elapsed.TotalSeconds:F1} s)");
            }

            return read;
        }
        catch
        {
            read.Dispose();
            throw;
        }
    }

    // The conversations of several data files as one list, by their place in it, each taken with its images read.
    private sealed class PendingData : IDisposable
    {
        private readonly List<PendingConversations> _files = [];
        private readonly List<int> _starts = [];                              // each file's first place

        public int Count { get; private set; }

        public void Add(PendingConversations file)
        {
            _files.Add(file);
            _starts.Add(Count);
            Count += file.Count;
        }

        // The conversations at `places`, in that order, their images read now (the system prompt added to those without one).
        public List<ChatTranscript> Take(IEnumerable<int> places, string? system, string what, Action<string>? log, CancellationToken cancellationToken)
        {
            var watch = Stopwatch.StartNew();
            var taken = new List<ChatTranscript>();
            int images = 0;
            foreach (int place in places)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int file = _starts.BinarySearch(place);
                file = file >= 0 ? LastStartingAt(file) : ~file - 1;
                var transcript = _files[file].Read(place - _starts[file]);
                foreach (var message in transcript.Messages)
                {
                    images += message.Parts.Count(p => p is ChatImage);
                }

                taken.Add(system is not null && !transcript.Messages.Any(m => m.Role == "system")
                    ? transcript with { Messages = [new ChatMessage("system", system), .. transcript.Messages] }
                    : transcript);
            }

            if (images > 0)
            {
                log?.Invoke($"{what}: the images of {taken.Count:N0} conversations read, {images:N0} ({watch.Elapsed.TotalSeconds:F1} s)");
            }

            return taken;
        }

        public void Dispose()
        {
            foreach (var file in _files)
            {
                file.Dispose();
            }
        }

        // The last of the files starting at the place of file `at` (empty files start where the next one does).
        private int LastStartingAt(int at)
        {
            while (at + 1 < _starts.Count && _starts[at + 1] == _starts[at])
            {
                at++;
            }

            return at;
        }
    }

    // Conversations tokenized into training sequences (their images' blocks of tokens from the family's encoder).
    private List<TrainingSequence> Encode(IReadOnlyList<ChatTranscript> transcripts, int maxLength, string what, IProgress<TuningStage>? progress, Action<string>? log,
        CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        if (Encoder.Vision is { } vision)
        {
            // The images' blocks first, on worker threads (each page decoded and transformed once), then the tokenizing below
            // finds them kept.
            var images = transcripts.SelectMany(t => t.Messages).SelectMany(m => m.Parts).OfType<ChatImage>().ToList();
            if (images.Count > 0)
            {
                string reading = $"reading {what} images";
                int total = images.DistinctBy(i => i.Hash).Count();
                var lockObject = new Lock();
                vision.PrepareBlocks(images, done =>
                {
                    lock (lockObject)
                    {
                        progress?.Report(new TuningStage(reading, done, total, watch.Elapsed));
                    }
                }, cancellationToken);
                log?.Invoke($"{what}: {total:N0} images read ({watch.Elapsed.TotalSeconds:F1} s, {Environment.ProcessorCount} threads)");
            }
        }

        // Each conversation rendered and tokenized on worker threads (independent of the others; the template and the
        // tokenizer serve several threads, the image side calls the family one call at a time), kept in the conversations'
        // order; the first failing conversation's error is the one thrown, as when they were encoded one after another.
        string stage = $"tokenizing {what}";                                    // made once, not per conversation
        var encoded = new TrainingSequence?[transcripts.Count];
        var errors = new Exception?[transcripts.Count];
        var gate = new Lock();
        int finished = 0;
        Parallel.For(0, transcripts.Count, new ParallelOptions { CancellationToken = cancellationToken }, i =>
        {
            try
            {
                encoded[i] = Encoder.Encode(transcripts[i], maxLength);
            }
            catch (Exception ex)
            {
                errors[i] = ex;
            }

            lock (gate)
            {
                progress?.Report(new TuningStage(stage, ++finished, transcripts.Count, watch.Elapsed));
            }
        });
        if (Array.FindIndex(errors, e => e is not null) is >= 0 and var failed)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(errors[failed]!).Throw();
        }

        var sequences = new List<TrainingSequence>(transcripts.Count);
        foreach (var sequence in encoded)
        {
            if (sequence is not null)
            {
                sequences.Add(sequence);
            }
        }

        long tokens = 0, trained = 0;
        foreach (var sequence in sequences)
        {
            tokens += sequence.Tokens.Length;
            trained += sequence.TrainedTokens;
        }

        log?.Invoke($"{what}: {sequences.Count:N0} sequences, {tokens:N0} tokens, {trained:N0} trained; "
                    + $"{transcripts.Count - sequences.Count:N0} conversations without trainable tokens skipped ({watch.Elapsed.TotalSeconds:F1} s)");
        return sequences;
    }

    // An IProgress that reports on the caller's thread.
    private sealed class Relay<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}
