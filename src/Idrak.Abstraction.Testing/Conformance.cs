// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Data;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// Checks an implementation against the library default on fixed and random cases, with tolerances, and reports every
/// failing case. Nothing here asserts: each method returns a <see cref="ConformanceReport"/> for any test framework (or
/// none) to check, print and save (<see cref="ConformanceReport.SaveFailures"/>).
/// </summary>
public static class Conformance
{

    /// <summary>Runs a contract's suite (fixed cases, then random ones) on an implementation.</summary>
    public static ConformanceReport Check<T>(T implementation, ContractSuite<T> suite, ContractCheckOptions? options = null)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(suite);
        options ??= new ContractCheckOptions();
        return suite.Check(implementation, [.. suite.Cases(options)], "", options.MaxFailures);
    }

    /// <summary>
    /// Checks a tokenizer: ids in range, the same ids every time, decoding the same text from a list or a span, text
    /// stable through a round trip, and, given a <paramref name="reference"/> (the library tokenizer it replaces), the
    /// same ids and text as it (<see cref="TokenizerSuite"/>).
    /// </summary>
    public static ConformanceReport Check(ITokenizer tokenizer, ITokenizer? reference = null, ContractCheckOptions? options = null) =>
        Check(tokenizer, new TokenizerSuite(reference), options);

    /// <summary>
    /// Checks a pretrained tokenizer (one a tokenizer model built): every tokenizer check, ranges encoded as their text is,
    /// tokens found by name, and its tokenizer.json naming its model; given a <paramref name="reference"/> (the library's
    /// model it replaces), the same ids and text as it (<see cref="PretrainedTokenizerSuite"/>).
    /// </summary>
    public static ConformanceReport CheckPretrained(IPretrainedTokenizer tokenizer, IPretrainedTokenizer? reference = null, ContractCheckOptions? options = null) =>
        Check(tokenizer, new PretrainedTokenizerSuite(reference), options);

    /// <summary>
    /// Checks a keyword index (Idrak.Text's <c>KeywordIndexes</c>, given as <see cref="KeywordIndexUnderTest"/>;
    /// <see cref="KeywordIndexSuite"/>): hits within the count asked for, ids in range and each once, finite positive
    /// scores best first with ties by id, nothing for absent words, the same hits again and from several threads, and,
    /// given a <paramref name="reference"/> (the library's index it replaces), the same ids and scores.
    /// </summary>
    /// <example>
    /// <code>
    /// static KeywordIndexUnderTest Of(string name) => new(texts =&gt; KeywordIndexes.Create(name, texts).Search);
    /// Conformance.CheckKeywordIndex(Of("mine"), Of("bm25")).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckKeywordIndex(KeywordIndexUnderTest index, KeywordIndexUnderTest? reference = null, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(index);
        return Check(index, new KeywordIndexSuite(reference), options);
    }

    /// <summary>
    /// Checks a text line segmenter (Idrak.Vision's <c>TextLineSegmenters</c>, given as its cutting: a page in, its lines
    /// out; <see cref="TextLineSegmenterSuite"/>) on pages drawn by the suite: no lines on a blank page, boxes on the page
    /// top to bottom with images their size, a line per drawn line (dots joined to it), no skew on a straight page, the same
    /// lines again and from several threads, and, given a <paramref name="reference"/> (the library's segmenter it
    /// replaces), the same lines. Six random cases unless <paramref name="options"/> says otherwise.
    /// </summary>
    public static ConformanceReport CheckTextLineSegmenter(Func<Data.ImageData, SegmentedPage> segmenter, Func<Data.ImageData, SegmentedPage>? reference = null,
        ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segmenter);
        return Check(segmenter, new TextLineSegmenterSuite(reference), options ?? new ContractCheckOptions { RandomCases = 6 });
    }

    /// <summary>
    /// Checks an approximate text matcher (Idrak.Text's <c>TextMatchers</c>, given as its search; <see cref="TextMatcherSuite"/>):
    /// nothing for nothing to match, a span within the text, the least edit distance of any span, runs of the text found
    /// exactly, spans at word boundaries (unless <paramref name="wordBoundaries"/> is false), the same span again and from
    /// several threads, and, given a <paramref name="reference"/> (the library's matcher it replaces), the same spans.
    /// </summary>
    public static ConformanceReport CheckTextMatcher(TextMatchSearch matcher, TextMatchSearch? reference = null, bool wordBoundaries = true, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(matcher);
        return Check(matcher, new TextMatcherSuite(reference, wordBoundaries), options);
    }

    /// <summary>
    /// Checks a glyph source (<see cref="GlyphSourceSuite"/>) on images of its own, as given, padded with paper and blank:
    /// every glyph on the image with ink and inside its cell or ink piece, labelled once, the known text's characters all
    /// accounted for or reported, the glyphs known found where they are, no glyph on a blank page, the same glyphs again
    /// and from several threads, an option it does not take refused. Six random (padded) cases unless
    /// <paramref name="options"/> says otherwise.
    /// </summary>
    /// <param name="source">The source.</param>
    /// <param name="samples">Images in its kind (at least one), with their known text and glyphs where known.</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckGlyphSource(IGlyphSource source, IReadOnlyList<GlyphSourceSample> samples, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return Check(source, new GlyphSourceSuite(samples), options ?? new ContractCheckOptions { RandomCases = 6 });
    }

    /// <summary>
    /// Checks a network recipe (Idrak's, given as <see cref="NetworkRecipeUnderTest"/>;
    /// <see cref="NetworkRecipeSuite"/>): the same seed gives the same weights and predictions, the predictions are finite
    /// and keep the batch, and, given a <paramref name="reference"/> (the library's recipe of that name), the same
    /// parameters, weights and predictions. A library recipe has samples of its own; give <paramref name="samples"/> for
    /// one of your own. Four random cases unless <paramref name="options"/> says otherwise.
    /// </summary>
    public static ConformanceReport CheckNetworkRecipe(string recipe, NetworkRecipeUnderTest builder, NetworkRecipeUnderTest? reference = null,
        IReadOnlyList<NetworkRecipeSample>? samples = null, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(builder);
        return Check(builder, new NetworkRecipeSuite(recipe, samples, reference), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>
    /// Checks a vision family's encoder (<see cref="VisionEncoderSuite"/>) on images of many sizes and channels: the same
    /// features every time, one features per block it says each image becomes, as many tokens as each block's layout says
    /// (width, grid, position ids), an image the same alone and in a batch, and, given <paramref name="cpu"/> (the same
    /// family's encoder on the CPU), the same features on the encoder's device within <paramref name="tolerance"/>; every
    /// call with the family's <paramref name="visionOptions"/> when given. Six random cases unless
    /// <paramref name="options"/> says otherwise.
    /// </summary>
    public static ConformanceReport CheckVisionEncoder(IVisionEncoder encoder, IVisionEncoder? cpu = null, IReadOnlyList<Data.ImageData>? samples = null,
        float tolerance = 1e-4f, ContractCheckOptions? options = null, VisionOptions? visionOptions = null)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        return Check(encoder, new VisionEncoderSuite(cpu, samples, tolerance, visionOptions) { On = encoder.Device }, options ?? new ContractCheckOptions { RandomCases = 6 });
    }

    /// <summary>
    /// Checks a fine-tuning data format (Idrak.Text's <c>ITuningDataFormat</c>, given as its reading: a path in, each
    /// conversation's messages out) on data files of its own (<see cref="TuningDataFormatSuite"/>): valid files read the
    /// same every time, from several threads, and stopped early; image parts carry their bytes; the expected conversations
    /// when given; invalid files refused with an error naming the file. Four random cases unless <paramref name="options"/>
    /// says otherwise.
    /// </summary>
    /// <example>
    /// <code>
    /// var format = TuningDataFormats.Get("sharegpt");
    /// Conformance.CheckTuningDataFormat(path =&gt; format.Read(path, options).Select(t =&gt; t.Messages), samples).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckTuningDataFormat(Func<string, IEnumerable<IReadOnlyList<ChatMessage>>> read, IReadOnlyList<TuningDataSample> samples,
        ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(read);
        return Check(read, new TuningDataFormatSuite(samples), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>
    /// Checks a sound codec (Idrak.Audio's <c>IAudioCodec</c>, given as its two methods in <see cref="AudioCodecUnderTest"/>)
    /// on sound files of its own (<see cref="AudioCodecSuite"/>): the header from the first bytes, the decoded sound's rate,
    /// channels, length and range, the same samples every time and from two threads, the expected samples when given, invalid
    /// files refused saying why; then files cut at random lengths, which decode shorter or are refused as damaged. Eight
    /// random cases unless <paramref name="options"/> says otherwise.
    /// </summary>
    /// <example>
    /// <code>
    /// var codec = AudioCodecs.Get("wav");
    /// Conformance.CheckAudioCodec(new AudioCodecUnderTest(
    ///     bytes =&gt; codec.ReadInfo(bytes.Span) is { } i ? new AudioHeader(i.SampleRate, i.Channels, i.Samples) : null,
    ///     bytes =&gt; { var a = codec.Decode(bytes.Span); return new DecodedAudio(a.SampleRate, a.Channels, a.Samples); }), samples).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckAudioCodec(AudioCodecUnderTest codec, IReadOnlyList<AudioSample> samples, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(codec);
        return Check(codec, new AudioCodecSuite(samples), options ?? new ContractCheckOptions { RandomCases = 8 });
    }

    /// <summary>
    /// Checks a short-time Fourier transform (Idrak.Audio's <c>Stft</c>, or one of your own, given as <see cref="StftUnderTest"/>)
    /// on signals the suite makes (<see cref="AudioFeatureSuite"/>): the frame count and bins, zero power on silence, a
    /// tone's bin, Parseval's identity per frame, the same spectra again. Six random cases unless <paramref name="options"/> says otherwise.
    /// </summary>
    public static ConformanceReport CheckStft(StftUnderTest stft, int sampleRate = 16000, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stft);
        return Check(stft, new AudioFeatureSuite(sampleRate), options ?? new ContractCheckOptions { RandomCases = 6 });
    }

    /// <summary>
    /// Checks a speech recognizer (Idrak.Audio's <c>ISpeechRecognizer</c>, or one of your own, given as
    /// <see cref="SpeechRecognizerUnderTest"/>) on sounds the suite makes (<see cref="SpeechRecognizerSuite"/>): it
    /// transcribes silence, tones, noise, a sound of milliseconds and an empty one without throwing, the segments in order
    /// within the sound and in the text, the same text twice and from two threads at once, and, with
    /// <paramref name="silenceMaxChars"/>, little text for silence. Two random cases unless <paramref name="options"/>
    /// says otherwise (each transcribes four times).
    /// </summary>
    /// <example>
    /// <code>
    /// Conformance.CheckSpeechRecognizer(new SpeechRecognizerUnderTest(16000, samples =&gt;
    /// {
    ///     var t = recognizer.Transcribe(new AudioData(samples, 1, 16000));
    ///     return new RecognizedSpeech(t.Text, [.. t.Segments.Select(s =&gt; new RecognizedSegment(s.Start, s.End, s.Text))]);
    /// })).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckSpeechRecognizer(SpeechRecognizerUnderTest recognizer, int? silenceMaxChars = null, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(recognizer);
        return Check(recognizer, new SpeechRecognizerSuite(silenceMaxChars), options ?? new ContractCheckOptions { RandomCases = 2 });
    }

    /// <summary>
    /// Checks a voice converter (Idrak.Audio's <c>IVoiceConverter</c>, or one of your own, given as
    /// <see cref="VoiceConverterUnderTest"/>) on tones and noise the suite makes (<see cref="VoiceConverterSuite"/>): finite
    /// embeddings of the said width, the same twice; finite conversions of the input's length (within the slack), the same
    /// twice and from two threads at once; a conversion to the source's own embedding. One random case unless
    /// <paramref name="options"/> says otherwise.
    /// </summary>
    /// <example>
    /// <code>
    /// Conformance.CheckVoiceConverter(new VoiceConverterUnderTest(converter.SampleRate, converter.EmbeddingSize,
    ///     clips =&gt; converter.Embed([.. clips.Select(c =&gt; new AudioData(c, 1, converter.SampleRate))]),
    ///     (speech, target) =&gt; converter.Convert(new AudioData(speech, 1, converter.SampleRate), target).Samples, hop)).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckVoiceConverter(VoiceConverterUnderTest converter, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(converter);
        return Check(converter, new VoiceConverterSuite(), options ?? new ContractCheckOptions { RandomCases = 1 });
    }

    /// <summary>
    /// Checks a voice store (Idrak.Audio's <c>IVoiceStore</c> with an encoder, or one of your own, given as
    /// <see cref="VoiceStoreUnderTest"/>) on voices of WAV clips the suite writes into <paramref name="folder"/>
    /// (<see cref="VoiceStoreSuite"/>): listed and found ignoring case once added, refused without consent or under an
    /// installed name, the same embedding after export, removal and import. One random case unless
    /// <paramref name="options"/> says otherwise.
    /// </summary>
    public static ConformanceReport CheckVoiceStore(VoiceStoreUnderTest store, string folder, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrEmpty(folder);
        return Check(store, new VoiceStoreSuite(folder), options ?? new ContractCheckOptions { RandomCases = 1 });
    }

    /// <summary>
    /// Checks an image model family (Idrak.Vision's <c>IImageModelFamily</c>, given as its loading: a checkpoint's path in, the
    /// loaded model wrapped as <see cref="ImageModelUnderTest"/> out) on checkpoints of its own
    /// (<see cref="ImageModelFamilySuite"/>): it loads, predicts finite outputs, the same every time, from several threads,
    /// alone and in a batch, and after loading again; the reference's outputs when given; invalid checkpoints refused saying
    /// why. Four random cases (random images of many sizes) unless <paramref name="options"/> says otherwise.
    /// </summary>
    /// <example>
    /// <code>
    /// Conformance.CheckImageModelFamily(path =&gt;
    /// {
    ///     var model = ImageModels.Load(path);
    ///     var outputs = model.Outputs().Build();
    ///     return new ImageModelUnderTest("classification", model.Labels, images =&gt; outputs.Predict(images), model);
    /// }, samples).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport CheckImageModelFamily(Func<string, ImageModelUnderTest> load, IReadOnlyList<ImageFamilySample> samples, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(load);
        return Check(load, new ImageModelFamilySuite(samples), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>
    /// Checks a feature cache (<see cref="IFeatureCache"/>) with <see cref="FeatureCacheSuite"/>: misses when empty or cleared,
    /// exact values and shapes back on the device asked for, its own copy, a miss for a key differing in any field, a put
    /// replacing the last, and many threads at once. Six random cases unless <paramref name="options"/> says otherwise.
    /// </summary>
    /// <example>
    /// <code>
    /// using var cache = FeatureCaches.Create("disk", new FeatureCacheOptions { Folder = "cache" });
    /// Conformance.CheckFeatureCache(cache).ThrowIfFailed();
    /// </code>
    /// </example>
    /// <param name="cache">The cache.</param>
    /// <param name="device">Where values are put and hits asked for (the CPU when null).</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckFeatureCache(IFeatureCache cache, Device? device = null, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        return Check(cache, new FeatureCacheSuite(device), options ?? new ContractCheckOptions { RandomCases = 6 });
    }

    /// <summary>
    /// Checks a cluster transport (plan 23) on groups of 2 to 8 ranks (<see cref="TransportSuite"/>): every message whole
    /// and in order under its tag, barriers, refusals and a timeout naming the silent rank. Four random cases unless
    /// <paramref name="options"/> says otherwise.
    /// </summary>
    /// <param name="transport">Connects a group of ranks of the transport.</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckTransport(TransportUnderTest transport, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        return Check(transport, new TransportSuite(), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>
    /// Checks a collective algorithm (plan 23) on groups of 2 to 8 ranks (<see cref="CollectivesSuite"/>): every operation
    /// against the sum, largest and smallest in rank order, identical bytes on every rank and on a rerun. Four random cases
    /// unless <paramref name="options"/> says otherwise.
    /// </summary>
    /// <param name="collectives">Connects a group of ranks of the algorithm.</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckCollectives(CollectivesUnderTest collectives, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        return Check(collectives, new CollectivesSuite(), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>Checks a link profile (plan 23, <see cref="LinkProfileSuite"/>).</summary>
    /// <param name="profile">The profile.</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckLinkProfile(Cluster.ILinkProfile profile, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return Check(profile, new LinkProfileSuite(), options ?? new ContractCheckOptions { RandomCases = 4 });
    }

    /// <summary>Checks a cluster topology (plan 23, <see cref="TopologySuite"/>).</summary>
    /// <param name="topology">The topology.</param>
    /// <param name="options">The cases to run.</param>
    public static ConformanceReport CheckTopology(Cluster.IClusterTopology topology, ContractCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(topology);
        return Check(topology, new TopologySuite(), options ?? new ContractCheckOptions { RandomCases = 2 });
    }

}
