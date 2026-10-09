// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How many tokens one image becomes in a vision-language model's prompt, and how they are arranged: a family with a
/// fixed size gives every image the same count (Gemma 3: 256 tokens, a 16 x 16 grid), a family with dynamic resolution
/// a count per image (from its size), with the grid its tokens cover when the family has one (rows and columns, or
/// frames, rows and columns), outermost first.
/// </summary>
/// <param name="Tokens">The image's tokens in the prompt (rows of its <see cref="ImageFeatures.Features"/>).</param>
public sealed record ImageTokenLayout(int Tokens)
{
    private readonly int[] _grid = [];

    /// <summary>The image's tokens.</summary>
    public int Tokens { get; } = Tokens > 0 ? Tokens : throw new ArgumentOutOfRangeException(nameof(Tokens), Tokens, "An image takes at least one token.");

    /// <summary>
    /// The grid the tokens cover, outermost first ([rows, columns], or [frames, rows, columns]), tokens in row-major order;
    /// empty when the family has none. Its product is <see cref="Tokens"/>.
    /// </summary>
    public IReadOnlyList<int> Grid
    {
        get => _grid;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            if (value.Count > 0 && (value.Any(v => v <= 0) || value.Aggregate(1L, (a, v) => a * v) != Tokens))
            {
                throw new ArgumentException($"A grid of [{string.Join(", ", value)}] does not hold {Tokens} tokens.", nameof(value));
            }

            _grid = [.. value];
        }
    }

    /// <inheritdoc />
    public bool Equals(ImageTokenLayout? other) => other is not null && Tokens == other.Tokens && _grid.AsSpan().SequenceEqual(other._grid);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Tokens, _grid.Length > 0 ? _grid[0] : 0, _grid.Length > 1 ? _grid[^1] : 0);

    /// <inheritdoc />
    public override string ToString() => _grid.Length == 0 ? $"{Tokens} tokens" : $"{Tokens} tokens ({string.Join(" x ", _grid)})";
}

/// <summary>
/// One image as a vision encoder gives it (<see cref="IVisionEncoder.Encode"/>): the embeddings of its tokens in the
/// language model's prompt, its token layout, and, for a family that places image tokens by several axes (M-RoPE:
/// time, height, width), their position ids. Owns its tensors: dispose it when done.
/// </summary>
public sealed class ImageFeatures : IDisposable
{
    /// <summary>Wraps an image's features (owned from now on).</summary>
    /// <param name="features">[tokens, width]: one embedding per image token, the language model's width, on its device.</param>
    /// <param name="layout">The image's token layout; its <see cref="ImageTokenLayout.Tokens"/> is the features' rows.</param>
    /// <param name="positions">
    /// Null (the usual case): the image's tokens take the next positions of the prompt, one each, as text does. Otherwise
    /// [axes, tokens]: each token's position on each axis of a multi-axis rotary embedding (M-RoPE), whole numbers as floats.
    /// </param>
    /// <exception cref="ArgumentException">The shapes do not agree with each other or with the layout.</exception>
    public ImageFeatures(Tensor features, ImageTokenLayout layout, Tensor? positions = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(layout);
        if (features.Rank != 2 || features.Shape[0] != layout.Tokens)
        {
            throw new ArgumentException($"An image's features are [tokens, width] with {layout.Tokens} tokens ({layout}), got {Tensor.FormatShape(features.Shape)}.", nameof(features));
        }

        if (positions is not null && (positions.Rank != 2 || positions.Shape[1] != layout.Tokens))
        {
            throw new ArgumentException($"An image's position ids are [axes, {layout.Tokens}], got {Tensor.FormatShape(positions.Shape)}.", nameof(positions));
        }

        Features = features;
        Layout = layout;
        Positions = positions;
    }

    /// <summary>[tokens, width]: the image tokens' embeddings.</summary>
    public Tensor Features { get; }

    /// <summary>The image's token layout.</summary>
    public ImageTokenLayout Layout { get; }

    /// <summary>[axes, tokens] position ids for a multi-axis rotary embedding, or null: the prompt's next positions.</summary>
    public Tensor? Positions { get; }

    /// <summary>The image's tokens.</summary>
    public int Tokens => Layout.Tokens;

    /// <inheritdoc />
    public void Dispose()
    {
        Features.Dispose();
        Positions?.Dispose();
    }
}

/// <summary>
/// The image side of a vision-language model: decoded images in, each image's token features out, with as many tokens
/// per image as its family gives it (<see cref="Layout"/>). It does its family's own preprocessing (resize, crop,
/// normalize, tiling or dynamic resolution) on the decoded pixels it is given. A family makes one from its checkpoint;
/// the testing kit checks one (<c>Conformance.CheckVisionEncoder</c>: deterministic, counts matching the layout, a batch
/// equal to single images, every device equal to the CPU).
/// </summary>
public interface IVisionEncoder : IDisposable
{
    /// <summary>The features' width: the language model's (each image token's embedding).</summary>
    int Width { get; }

    /// <summary>Where the features are made.</summary>
    Device Device { get; }

    /// <summary>How many tokens <paramref name="image"/> becomes and how they are arranged, without encoding it.</summary>
    ImageTokenLayout Layout(ImageData image);

    /// <summary>
    /// The features of <paramref name="images"/> in order, one <see cref="ImageFeatures"/> each (the caller disposes them),
    /// each with the layout <see cref="Layout"/> gives that image. Runs without recording gradients. An image encodes the
    /// same alone and in a batch.
    /// </summary>
    IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images);
}

/// <summary>
/// A vision encoder that also shows its stages, for comparing them one by one with a reference (<c>idrak vlm check</c>
/// against transformers): the pixel values its preprocessing gives, the vision tower's output, and the features. A
/// family's encoder may implement it; nothing requires it.
/// </summary>
public interface IVisionEncoderStages
{
    /// <summary>The pixel values the encoder's preprocessing gives <paramref name="image"/>, [channels, height, width], on the CPU.</summary>
    Tensor PixelValues(ImageData image);

    /// <summary>The vision tower's output for pixel values [images, channels, height, width] (before any projection), without gradients.</summary>
    Tensor Tower(Tensor pixelValues);

    /// <summary>The features [images, tokens, width] of pixel values [images, channels, height, width], without gradients.</summary>
    Tensor Features(Tensor pixelValues);
}

/// <summary>
/// How a vision-language model's prompt holds its images: the chat template writes one marker per image part, and the
/// model reads more than the marker (the image's tokens, and whatever its processor puts around them). <see cref="Expand"/>
/// does that second step on the rendered text, before it is tokenized; the embeddings of the
/// <see cref="ImageToken"/> runs are then replaced by the images' features. Each family's registration gives its own
/// (with its token ids from its configuration); <see cref="ImageTokenFormat"/> is a building block that covers the
/// common shapes.
/// </summary>
public interface IImagePromptFormat
{
    /// <summary>A name for messages (the family's).</summary>
    string Name { get; }

    /// <summary>The token whose embeddings an image's features replace (one per image token).</summary>
    int ImageToken { get; }

    /// <summary>
    /// The rendered prompt <paramref name="prompt"/> with each image's marker expanded to the text the model reads, the
    /// image's <see cref="ImageTokenLayout.Tokens"/> image tokens included (written as the tokenizer's text of
    /// <see cref="ImageToken"/>, so that tokenizing it gives their ids), the images in order.
    /// </summary>
    /// <exception cref="InvalidOperationException">The prompt does not hold one marker per image.</exception>
    string Expand(string prompt, IReadOnlyList<ImageTokenLayout> images, ITokenizer tokenizer);
}

/// <summary>
/// An image prompt format built from token ids: each marker token the chat template wrote (in order, one per image)
/// becomes <see cref="Before"/>, the begin token (if any), the image's count of image tokens, the end token (if any) and
/// <see cref="After"/>. Gemma 3's processor is the marker <c>&lt;start_of_image&gt;</c>, begin and end
/// <c>&lt;start_of_image&gt;</c> and <c>&lt;end_of_image&gt;</c>, "\n\n" around; LLaVA's is the marker <c>&lt;image&gt;</c>
/// replaced by the image tokens alone (the marker is the image token).
/// </summary>
/// <param name="name">A name for messages.</param>
/// <param name="marker">The token the chat template writes once per image.</param>
/// <param name="imageToken">The image token (repeated once per image token).</param>
/// <param name="begin">The token before the image tokens, or null for none.</param>
/// <param name="end">The token after the image tokens, or null for none.</param>
/// <param name="before">Text before the begin token.</param>
/// <param name="after">Text after the end token.</param>
public sealed class ImageTokenFormat(string name, int marker, int imageToken, int? begin = null, int? end = null, string before = "", string after = "") : IImagePromptFormat
{
    /// <inheritdoc />
    public string Name { get; } = name ?? throw new ArgumentNullException(nameof(name));

    /// <summary>The token the chat template writes once per image.</summary>
    public int Marker { get; } = marker;

    /// <inheritdoc />
    public int ImageToken { get; } = imageToken;

    /// <summary>The token before the image tokens, or null.</summary>
    public int? Begin { get; } = begin;

    /// <summary>The token after the image tokens, or null.</summary>
    public int? End { get; } = end;

    /// <summary>Text before the begin token.</summary>
    public string Before { get; } = before ?? "";

    /// <summary>Text after the end token.</summary>
    public string After { get; } = after ?? "";

    /// <inheritdoc />
    public string Expand(string prompt, IReadOnlyList<ImageTokenLayout> images, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string markerText = Token(tokenizer, Marker, "image marker"), image = Token(tokenizer, ImageToken, "image");
        string begin = Begin is { } b ? Token(tokenizer, b, "begin-image") : "", end = End is { } e ? Token(tokenizer, e, "end-image") : "";
        var at = new List<int>();
        for (int i = prompt.IndexOf(markerText, StringComparison.Ordinal); i >= 0; i = prompt.IndexOf(markerText, i + markerText.Length, StringComparison.Ordinal))
        {
            at.Add(i);
        }

        if (at.Count != images.Count)
        {
            throw new InvalidOperationException($"The rendered prompt holds {at.Count} image markers ({markerText}) for {images.Count} images: the chat template does not write one per image part.");
        }

        if (images.Count == 0)
        {
            return prompt;
        }

        var text = new StringBuilder(prompt.Length + images.Sum(l => l.Tokens) * image.Length);
        int from = 0;
        for (int k = 0; k < at.Count; k++)
        {
            text.Append(prompt, from, at[k] - from).Append(Before).Append(begin).Insert(text.Length, image, images[k].Tokens).Append(end).Append(After);
            from = at[k] + markerText.Length;
        }

        return text.Append(prompt, from, prompt.Length - from).ToString();
    }

    /// <inheritdoc />
    public override string ToString() => $"ImageTokenFormat({Name})";

    private static string Token(ITokenizer tokenizer, int id, string what) =>
        tokenizer.TokenOf(id) is { Length: > 0 } text ? text
            : throw new InvalidOperationException($"The tokenizer has no {what} token {id} (vocabulary {tokenizer.VocabularySize}).");
}

/// <summary>
/// How a vision-language family's image tokens attend in the prompt: which keys each row sees in a sequence holding image
/// blocks. Text rows are causal in every rule the library ships; what differs is the image rows (Gemma 3: an image's tokens
/// see each other in both directions; LLaVA: plainly causal). Each family picks its rule (by name, from
/// <see cref="ImageAttentionRules"/>), so the decoder never assumes one.
/// </summary>
public interface IImageAttentionRule
{
    /// <summary>The rule's name in <see cref="ImageAttentionRules"/>.</summary>
    string Name { get; }

    /// <summary>
    /// True when image rows attend exactly as text rows (plain causal attention, windowed on sliding-window layers): the
    /// decoder then only substitutes the images' features and keeps its causal kernels; <see cref="Spans"/> is not asked.
    /// </summary>
    bool Causal { get; }

    /// <summary>
    /// Each row's range of keys for one sequence of <paramref name="rows"/> positions (keys are the same positions) holding
    /// <paramref name="blocks"/> (each image's first position and token count, in order, not overlapping), on a layer that
    /// sees <paramref name="window"/> keys back (its own included; 0: every earlier key).
    /// </summary>
    KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window);
}

/// <summary>
/// The attention rules for image tokens, by name. The library has one, <see cref="Causal"/>: image rows attend as the
/// decoder's text rows do, which is the decoder's own behaviour rather than a family's choice. A family whose image tokens
/// attend otherwise registers its rule here (Gemma 3's registration adds "image-blocks", each image's rows seeing their
/// whole image through <see cref="KeySpans.ImageBlocks"/>) and names it in its vision part; an app can then take that
/// name's place. Nothing falls back to another rule: an unknown name is an error naming this registry.
/// </summary>
public static class ImageAttentionRules
{
    /// <summary>Image tokens attend as text does: causal (and windowed on sliding-window layers). LLaVA's rule.</summary>
    public const string Causal = "causal";

    private static readonly SlotTable<string, IImageAttentionRule> Table = BuiltIn();

    private static SlotTable<string, IImageAttentionRule> BuiltIn()
    {
        var table = new SlotTable<string, IImageAttentionRule>(nameof(ImageAttentionRules), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Causal, new CausalRule());
        return table;
    }

    // Each Spans is one call: an app's rule falls back (or is shadowed) per call. Causal is read when the rule is chosen.
    private static IImageAttentionRule Guard(Slot slot, IImageAttentionRule app, IImageAttentionRule library) => new GuardedRule(slot, app, library);

    /// <summary>Registers <paramref name="rule"/> under its <see cref="IImageAttentionRule.Name"/>; under a library name it takes that rule's place.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImageAttentionRule rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        ArgumentException.ThrowIfNullOrEmpty(rule.Name);
        Table.Register(rule.Name, rule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's rule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The rule <paramref name="name"/>, or null when none is registered.</summary>
    public static IImageAttentionRule? Find(string name) => Table.Find(name);

    /// <summary>The rule <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message names <see cref="Register"/>.</exception>
    public static IImageAttentionRule Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"No image attention rule '{name}' is registered ({string.Join(", ", Table.Keys)}); add it with ImageAttentionRules.Register.");

    /// <summary>The library's rule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImageAttentionRule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the rule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's rule <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    private sealed class CausalRule : IImageAttentionRule
    {
        public string Name => ImageAttentionRules.Causal;

        public bool Causal => true;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.Causal(rows, window);

        public override string ToString() => "causal image tokens";
    }

    private sealed class GuardedRule(Slot slot, IImageAttentionRule app, IImageAttentionRule library) : IImageAttentionRule
    {
        public string Name => app.Name;

        public bool Causal => app.Causal;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) =>
            slot.Call(() => app.Spans(rows, blocks, window), () => library.Spans(rows, blocks, window), (a, b) => a.Starts.SequenceEqual(b.Starts) && a.Ends.SequenceEqual(b.Ends) ? null : "different key ranges");
    }
}
